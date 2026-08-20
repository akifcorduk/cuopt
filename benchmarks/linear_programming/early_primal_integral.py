#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and score short-horizon MIP primal-integral benchmarks.

The solver keeps its normal long time limit so internal stage budgets do not
change.  A process-level timeout bounds measurement runs, while scoring ignores
incumbents reported after the requested horizon.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Iterable


FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
HEURISTIC_SOLUTION_RE = re.compile(
    rf"New solution from .*primal heuristics.*?Objective\s+({FLOAT_PATTERN}).*?"
    rf"Time\s+({FLOAT_PATTERN})"
)
TREE_SOLUTION_RE = re.compile(
    rf"^\s*[A-Z]\s+.*?([+-]\d+\.\d+[eE][+-]\d+).*?({FLOAT_PATTERN})\s*$"
)
OPTIMAL_ROOT_RE = re.compile(
    rf"Optimal solution found at root node\. Objective\s+({FLOAT_PATTERN})\. "
    rf"Time\s+({FLOAT_PATTERN})"
)
PAPILO_TIME_RE = re.compile(rf"Papilo presolve time:\s+({FLOAT_PATTERN})")
BEST_OBJECTIVE_RE = re.compile(rf"Best objective\s+({FLOAT_PATTERN})")


def read_optimal_objectives(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        result = {}
        for row in rows:
            name = row["InstanceInst."]
            value = row["ObjectiveObje."]
            try:
                result[name] = float(value)
            except ValueError:
                # Infeasible instances have no primal-integral reference value.
                continue
        return result


def parse_incumbents(
    path: Path, horizon: float, legacy_local_early_time: bool = False
) -> list[tuple[float, float]]:
    incumbents = []
    papilo_elapsed = 0.0
    presolve_optimal_pending = False
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            papilo_match = PAPILO_TIME_RE.search(line)
            if papilo_match is not None:
                papilo_elapsed = float(papilo_match.group(1))
            if "Optimal solution found during presolve." in line:
                presolve_optimal_pending = True
                continue
            if presolve_optimal_pending:
                best_match = BEST_OBJECTIVE_RE.search(line)
                if best_match is not None:
                    objective = float(best_match.group(1))
                    if math.isfinite(objective) and papilo_elapsed <= horizon:
                        incumbents.append((papilo_elapsed, objective))
                    presolve_optimal_pending = False
                    continue
            match = HEURISTIC_SOLUTION_RE.search(line)
            if match is None:
                match = OPTIMAL_ROOT_RE.search(line)
            if match is None:
                match = TREE_SOLUTION_RE.match(line)
            if match is None:
                continue
            objective = float(match.group(1))
            elapsed = float(match.group(2))
            if (
                legacy_local_early_time
                and papilo_elapsed > 0.0
                and "early primal heuristics" in line
            ):
                # CPUFJ restarted on the PaPILO-reduced problem used a local timer
                # before cuOpt switched the log to the global solve timer.
                elapsed += papilo_elapsed
            if math.isfinite(objective) and 0.0 <= elapsed <= horizon:
                incumbents.append((elapsed, objective))
    incumbents.sort()
    return incumbents


def primal_gap(objective: float, optimal: float) -> float:
    if objective * optimal < 0.0:
        return 1.0
    return abs(objective - optimal) / max(abs(objective), abs(optimal), 1.0)


def primal_integral(
    incumbents: Iterable[tuple[float, float]], optimal: float, horizon: float
) -> float:
    last_time = 0.0
    last_value = 1.0
    integral = 0.0
    best_gap = math.inf
    for elapsed, objective in incumbents:
        elapsed = min(elapsed, horizon)
        integral += (elapsed - last_time) * last_value
        last_time = elapsed
        gap = primal_gap(objective, optimal)
        if gap < best_gap:
            best_gap = gap
            last_value = gap
    integral += (horizon - last_time) * last_value
    return integral / horizon


def collect_instances(directories: Iterable[Path]) -> list[Path]:
    instances: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob("*.mps")):
            instances.setdefault(path.stem, path)
    return list(instances.values())


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_instances(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances = collect_instances(args.instance_dir)
    if args.instance:
        selected = set(args.instance)
        instances = [
            instance for instance in instances if instance.stem in selected
        ]
        missing = selected.difference(instance.stem for instance in instances)
        if missing:
            raise RuntimeError(
                f"instances not found: {', '.join(sorted(missing))}"
            )
    for repeat in range(args.repeats):
        for index, instance in enumerate(instances, start=1):
            log_path = args.output_dir / f"{instance.stem}.r{repeat}.log"
            command = [
                str(args.binary),
                str(instance),
                "--time-limit",
                str(args.solver_time_limit),
                "--num-cpu-threads",
                str(args.num_cpu_threads),
                "--random-seed",
                str(args.seed),
                "--log-to-console",
                "0",
                "--log-file",
                str(log_path),
            ]
            if args.params_file is not None:
                command.extend(["--params-file", str(args.params_file)])
            print(
                f"[{repeat + 1}/{args.repeats} {index}/{len(instances)}] {instance.stem}",
                flush=True,
            )
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=os.environ.copy(),
            )
            try:
                return_code = process.wait(timeout=args.wall_timeout)
            except subprocess.TimeoutExpired:
                terminate_process_group(process)
            else:
                if return_code != 0:
                    raise RuntimeError(
                        f"solver exited with status {return_code} for {instance.stem}; "
                        f"see {log_path}"
                    )


def score_logs(args: argparse.Namespace) -> None:
    optimal = read_optimal_objectives(args.optimal_objectives)
    rows = []
    for log_path in sorted(args.log_dir.glob("*.log")):
        match = re.match(r"(.+)\.r(\d+)\.log$", log_path.name)
        if match is None or match.group(1) not in optimal:
            continue
        instance = match.group(1)
        incumbents = parse_incumbents(
            log_path,
            args.horizon,
            legacy_local_early_time=args.legacy_local_early_time,
        )
        score = primal_integral(incumbents, optimal[instance], args.horizon)
        rows.append((instance, int(match.group(2)), score, len(incumbents)))

    if not rows:
        raise RuntimeError(f"no scoreable logs found in {args.log_dir}")

    scores = [row[2] for row in rows]
    shifted_geomean = (
        math.exp(
            sum(math.log(score + args.shift) for score in scores) / len(scores)
        )
        - args.shift
    )
    feasible = sum(score < 1.0 for score in scores)
    print(
        f"instances={len(rows)} feasible={feasible} horizon={args.horizon:g} "
        f"shift={args.shift:g} shifted_geomean={shifted_geomean:.9f}"
    )
    print("instance,repeat,primal_integral,incumbents")
    for instance, repeat, score, count in rows:
        print(f"{instance},{repeat},{score:.12g},{count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="run instances with a process-level cutoff"
    )
    run_parser.add_argument("--binary", type=Path, required=True)
    run_parser.add_argument(
        "--instance-dir", type=Path, action="append", required=True
    )
    run_parser.add_argument(
        "--instance", action="append", help="instance stem to include"
    )
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--params-file", type=Path)
    run_parser.add_argument("--solver-time-limit", type=float, default=600.0)
    run_parser.add_argument("--wall-timeout", type=float, default=20.0)
    run_parser.add_argument("--num-cpu-threads", type=int, default=4)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.set_defaults(function=run_instances)

    score_parser = subparsers.add_parser(
        "score", help="score incumbent events in solver logs"
    )
    score_parser.add_argument("--log-dir", type=Path, required=True)
    score_parser.add_argument("--optimal-objectives", type=Path, required=True)
    score_parser.add_argument("--horizon", type=float, default=10.0)
    score_parser.add_argument("--shift", type=float, default=0.001)
    score_parser.add_argument("--legacy-local-early-time", action="store_true")
    score_parser.set_defaults(function=score_logs)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
