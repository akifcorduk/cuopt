#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and summarize the standalone CPU-LNS feasibility benchmark.

The default corpus contains the 20 instances from the CPU-LNS feasibility
request plus ten additional, diverse MIPLIB 2017 instances.  Runs are
sequential to avoid cross-instance CPU/GPU contention.  Every raw cuOpt log is
kept next to CSV and JSON summaries so a throughput or feasibility claim is
auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ATTACHMENT_INSTANCES = (
    "30n20b8",
    "academictimetablesmall",
    "app1-2",
    "bnatt400",
    "chromaticindex1024-7",
    "csched007",
    "dano3_5",
    "dws008-01",
    "irish-electricity",
    "milo-v12-6-r2-40-1",
    "momentum1",
    "neos-1354092",
    "neos-3381206-awhea",
    "neos-3555904-turama",
    "neos-4722843-widden",
    "neos-5114902-kasavu",
    "ns1116954",
    "s250r10",
    "savsched1",
    "supportcase10",
)

ADDITIONAL_INSTANCES = (
    "air05",
    "binkar10_1",
    "cod105",
    "enlight_hard",
    "ex10",
    "gen-ip054",
    "gmu-35-50",
    "neos-3004026-krka",
    "rmatr200-p5",
    "seymour1",
)

DEFAULT_INSTANCES = ATTACHMENT_INSTANCES + ADDITIONAL_INSTANCES
ABLATION_VARIANTS = (
    "none",
    "legacy_rounding",
    "no_lp_reference",
    "no_seed_bp",
    "seed_natural_only",
    "seed_lp_order_only",
    "seed_degree_only",
    "seed_random_only",
    "no_initial_fj",
    "no_periodic_fj",
    "no_fj",
    "no_bandit",
    "fixed_ruin",
    "no_violated_ruin",
    "no_similarity",
    "no_random_walk",
    "no_propagate_repair",
    "with_propagate_repair",
    "no_shift_repair",
    "no_greedy_repair",
    "no_cosine",
    "no_type_filter",
    "no_saturation",
    "no_tardiness",
    "no_weight_decay",
    "raw_violation_excess",
    "no_projection",
    "with_projection",
    "no_sa",
    "full_refresh",
)
ABLATION_DESCRIPTIONS = {
    "none": "selected default CPU-LNS portfolio",
    "legacy_rounding": "round the LP seed in the caller and discard its fractional reference",
    "no_lp_reference": "keep deterministic working rounding but discard fractional LP guidance",
    "no_seed_bp": "disable the bounded seed bounds-propagation constructor",
    "seed_natural_only": "run only natural-order seed bounds propagation",
    "seed_lp_order_only": "run only LP-confidence seed bounds propagation",
    "seed_degree_only": "run only degree-ordered stochastic seed bounds propagation",
    "seed_random_only": "run only shuffled stochastic seed bounds propagation",
    "no_initial_fj": "disable the initial Feasibility Jump burst",
    "no_periodic_fj": "disable near-feasible periodic Feasibility Jump",
    "no_fj": "disable both initial and periodic Feasibility Jump",
    "no_bandit": "replace UCB operator selection with round-robin selection",
    "fixed_ruin": "replace adaptive ruin sizes with a fixed size of 16",
    "no_violated_ruin": "remove the violated-row ruin operator",
    "no_similarity": "remove similarity scoring and the similarity ruin operator",
    "no_random_walk": "remove the constraint-graph random-walk ruin operator",
    "no_propagate_repair": "remove the ordered propagation repair arm",
    "with_propagate_repair": "enable the ordered propagation repair arm",
    "no_shift_repair": "remove the randomized shift/propagation repair arm",
    "no_greedy_repair": "remove greedy repair and its intensification polish",
    "no_cosine": "remove cosine similarity from variable relatedness",
    "no_type_filter": "allow similarity candidates from different domain classes",
    "no_saturation": "remove LP/bound saturation from seeds and similarity ties",
    "no_tardiness": "use unit constraint weights",
    "no_weight_decay": "accumulate tardiness weights without decay",
    "raw_violation_excess": "score feasibility with raw instead of row-scaled excess",
    "no_projection": "disable near-feasible row projection while retaining greedy polish",
    "with_projection": "enable near-feasible row projection in greedy polish",
    "no_sa": "reject worsening feasible objective moves",
    "full_refresh": "recompute all activities and dynamic state after every LNS attempt",
}
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class Result:
    instance: str
    attachment: bool
    file_bytes: int
    return_code: int
    wall_time_s: float
    solver_time_s: float | None = None
    presolve_time_s: float | None = None
    lns_time_s: float | None = None
    first_feasible_lns_s: float | None = None
    first_feasible_solve_s: float | None = None
    first_feasible_source: str | None = None
    first_feasible_objective_solver: float | None = None
    first_feasible_objective_user: float | None = None
    final_objective_solver: float | None = None
    final_objective_user: float | None = None
    objective_improvement_solver: float | None = None
    feasible: bool = False
    constraint_violation: float | None = None
    integrality_violation: float | None = None
    bounds_violation: float | None = None
    variables: int | None = None
    integer_variables: int | None = None
    constraints: int | None = None
    nonzeros: int | None = None
    start_unsat: int | None = None
    final_unsat: int | None = None
    final_excess: float | None = None
    iterations: int | None = None
    iterations_per_second: float | None = None
    diagnostics: dict[str, Any] | None = None
    objective_trajectory: list[dict[str, Any]] = field(default_factory=list)


def parse_key_values(line: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, raw_value in re.findall(r"([a-zA-Z][a-zA-Z0-9_]*)=([^ ]+)", line):
        value = raw_value.rstrip(",")
        try:
            parsed[key] = float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def last_match(pattern: str, text: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    return matches[-1] if matches else None


def parse_log(result: Result, text: str, feasibility_tolerance: float) -> Result:
    start = last_match(
        rf"CPU LNS start:.*vars (\d+), integer vars (\d+), constraints (\d+), "
        rf"(?:nnz (\d+), )?unsat (\d+), excess ({FLOAT})",
        text,
    )
    if start:
        result.variables = int(start.group(1))
        result.integer_variables = int(start.group(2))
        result.constraints = int(start.group(3))
        result.nonzeros = int(start.group(4)) if start.group(4) else None
        result.start_unsat = int(start.group(5))

    finish = last_match(
        rf"CPU LNS finished: unsat (\d+), excess ({FLOAT}),.*?(\d+) iterations "
        rf"in ({FLOAT})s \(({FLOAT}) it/s\)",
        text,
    )
    if finish:
        result.final_unsat = int(finish.group(1))
        result.final_excess = float(finish.group(2))
        result.iterations = int(finish.group(3))
        result.lns_time_s = float(finish.group(4))
        result.iterations_per_second = float(finish.group(5))
    else:
        progress = last_match(
            rf"CPU LNS progress: ({FLOAT})s elapsed, (\d+) iterations, ({FLOAT}) it/s, "
            rf"best unsat (\d+), excess ({FLOAT})",
            text,
        )
        if progress:
            result.lns_time_s = float(progress.group(1))
            result.iterations = int(progress.group(2))
            result.iterations_per_second = float(progress.group(3))
            result.final_unsat = int(progress.group(4))
            result.final_excess = float(progress.group(5))

    first_feasible_line = last_match(r"^CPU_LNS_FIRST_FEASIBLE .*$", text)
    if first_feasible_line:
        first_values = parse_key_values(first_feasible_line.group(0))
        result.first_feasible_source = str(first_values.get("source"))
        result.first_feasible_lns_s = float(first_values["elapsed_s"])
        result.first_feasible_solve_s = float(first_values["solve_elapsed_s"])
        if "objective_solver" in first_values:
            result.first_feasible_objective_solver = float(
                first_values["objective_solver"]
            )
        if "objective_user" in first_values:
            result.first_feasible_objective_user = float(first_values["objective_user"])
    else:
        first_feasible = last_match(
            rf"CPU LNS feasible .*?\(({FLOAT})s elapsed\)", text
        )
        if first_feasible:
            result.first_feasible_lns_s = float(first_feasible.group(1))

    result.objective_trajectory = [
        parse_key_values(line)
        for line in re.findall(r"^CPU_LNS_OBJECTIVE (.*)$", text, re.MULTILINE)
    ]

    solution_line = last_match(r"^Solution objective:.*$", text)
    if solution_line:
        line = solution_line.group(0)
        objective = re.search(rf"Solution objective: ({FLOAT})", line)
        solver_time = re.search(rf"total_solve_time ({FLOAT})", line)
        presolve_time = re.search(rf"presolve_time ({FLOAT})", line)
        constraint_violation = re.search(rf"max constraint violation ({FLOAT})", line)
        integrality_violation = re.search(rf"max int violation ({FLOAT})", line)
        bounds_violation = re.search(rf"max var bounds violation ({FLOAT})", line)
        if solver_time:
            result.solver_time_s = float(solver_time.group(1))
        if objective:
            result.final_objective_user = float(objective.group(1))
        if presolve_time:
            result.presolve_time_s = float(presolve_time.group(1))
        if constraint_violation:
            result.constraint_violation = float(constraint_violation.group(1))
        if integrality_violation:
            result.integrality_violation = float(integrality_violation.group(1))
        if bounds_violation:
            result.bounds_violation = float(bounds_violation.group(1))

    violations = (
        result.constraint_violation,
        result.integrality_violation,
        result.bounds_violation,
    )
    result.feasible = result.return_code == 0 and all(
        value is not None and value <= feasibility_tolerance for value in violations
    )

    diagnostic_lines = re.findall(r"^CPU_LNS_DIAGNOSTICS (.*)$", text, re.MULTILINE)
    diagnostic_lines.extend(
        re.findall(r"^CPU_LNS_PAIR_DIAGNOSTICS (.*)$", text, re.MULTILINE)
    )
    if diagnostic_lines:
        diagnostics: dict[str, Any] = {}
        for line in diagnostic_lines:
            diagnostics.update(parse_key_values(line))
        result.diagnostics = diagnostics
        result.iterations = int(diagnostics.get("iterations", result.iterations or 0))
        result.iterations_per_second = float(
            diagnostics.get("iterations_per_second", result.iterations_per_second or 0.0)
        )
        result.lns_time_s = float(diagnostics.get("elapsed_s", result.lns_time_s or 0.0))
        if "best_unsat" in diagnostics:
            result.final_unsat = int(diagnostics["best_unsat"])
        if "best_excess" in diagnostics:
            result.final_excess = float(diagnostics["best_excess"])
        if "best_objective" in diagnostics:
            result.final_objective_solver = float(diagnostics["best_objective"])
        if float(diagnostics.get("first_feasible_solve_s", -1.0)) >= 0:
            result.first_feasible_solve_s = float(diagnostics["first_feasible_solve_s"])
            if "first_feasible_objective" in diagnostics:
                first_objective = float(diagnostics["first_feasible_objective"])
                if math.isfinite(first_objective):
                    result.first_feasible_objective_solver = first_objective
    elif result.feasible:
        # The baseline implementation returned immediately on feasibility, so its final solve
        # time is also its time to first feasible. New implementations publish the exact value.
        result.first_feasible_solve_s = result.solver_time_s
    if result.final_objective_solver is None and result.objective_trajectory:
        result.final_objective_solver = float(
            result.objective_trajectory[-1]["best_solver"]
        )
    if (
        result.first_feasible_objective_solver is not None
        and result.final_objective_solver is not None
        and math.isfinite(result.first_feasible_objective_solver)
        and math.isfinite(result.final_objective_solver)
    ):
        result.objective_improvement_solver = (
            result.first_feasible_objective_solver - result.final_objective_solver
        )
    return result


def read_instances(path: Path | None, minimum_instances: int) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_INSTANCES
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", maxsplit=1)[0].strip()
        if name:
            names.append(Path(name).stem)
    if len(names) < minimum_instances:
        raise ValueError(
            f"benchmark corpus must contain at least {minimum_instances} instances; "
            f"got {len(names)}"
        )
    return tuple(names)


def write_csv(path: Path, results: list[Result]) -> None:
    rows = []
    for result in results:
        row = asdict(result)
        row["diagnostics"] = json.dumps(row["diagnostics"], sort_keys=True)
        row["objective_trajectory"] = json.dumps(
            row["objective_trajectory"], sort_keys=True
        )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[Result], time_limit_s: float) -> dict[str, Any]:
    attachment = [result for result in results if result.attachment]
    large = [result for result in results if result.file_bytes >= 10_000_000]
    rates = [
        result.iterations_per_second
        for result in results
        if result.iterations_per_second is not None
    ]
    large_rates = [
        result.iterations_per_second
        for result in large
        if result.iterations_per_second is not None
    ]
    within_limit = [
        result
        for result in results
        if result.feasible
        and result.first_feasible_solve_s is not None
        and result.first_feasible_solve_s <= time_limit_s
    ]
    objective_runs = [
        result
        for result in results
        if result.objective_improvement_solver is not None
        and math.isfinite(result.objective_improvement_solver)
    ]
    return {
        "instance_count": len(results),
        "time_limit_s": time_limit_s,
        "feasible_count": sum(result.feasible for result in results),
        "feasible_within_limit_count": len(within_limit),
        "attachment_count": len(attachment),
        "attachment_feasible_count": sum(result.feasible for result in attachment),
        "attachment_feasible_within_limit_count": sum(
            result in within_limit for result in attachment
        ),
        "large_instance_count": len(large),
        "large_feasible_count": sum(result.feasible for result in large),
        "large_feasible_within_limit_count": sum(result in within_limit for result in large),
        "wall_overrun_count": sum(result.wall_time_s > time_limit_s * 1.25 for result in results),
        "iterations_per_second_min": min(rates, default=None),
        "iterations_per_second_median": sorted(rates)[len(rates) // 2] if rates else None,
        "large_iterations_per_second_min": min(large_rates, default=None),
        "large_iterations_per_second_median": (
            sorted(large_rates)[len(large_rates) // 2] if large_rates else None
        ),
        "large_instances_at_1000_ips": sum(rate >= 1000.0 for rate in large_rates),
        "objective_trackable_count": len(objective_runs),
        "objective_improved_count": sum(
            result.objective_improvement_solver > 1.0e-9
            for result in objective_runs
        ),
        "objective_improvement_solver_sum": sum(
            result.objective_improvement_solver for result in objective_runs
        ),
        "objective_best_updates": sum(
            int((result.diagnostics or {}).get("objective_best_updates", 0))
            for result in results
        ),
        "sa_uphill_accepts": sum(
            int((result.diagnostics or {}).get("sa_uphill_accepts", 0))
            for result in results
        ),
        "sa_escape_best_updates": sum(
            int((result.diagnostics or {}).get("sa_escape_best_updates", 0))
            for result in results
        ),
        "failures": [result.instance for result in results if result.return_code != 0],
    }


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo", type=Path, default=repo_default)
    parser.add_argument("--cuopt-cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instances-file", type=Path)
    parser.add_argument("--minimum-instances", type=int, default=30)
    parser.add_argument("--time-limit", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-cpu-threads", type=int, default=1)
    parser.add_argument("--determinism-mode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--presolve", type=int, choices=(-1, 0, 1, 2), default=-1)
    parser.add_argument("--feasibility-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--ablation", choices=ABLATION_VARIANTS, default="none")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    dataset_dir = repo / "datasets" / "miplib2017"
    output_dir = args.output_dir.resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    instances = read_instances(args.instances_file, args.minimum_instances)

    missing = [name for name in instances if not (dataset_dir / f"{name}.mps").is_file()]
    if missing:
        parser.error(f"missing MIPLIB instances: {', '.join(missing)}")
    if not args.cuopt_cli.is_file():
        parser.error(f"cuopt_cli does not exist: {args.cuopt_cli}")

    partial_path = output_dir / "results.partial.json"
    cached_results: dict[str, Result] = {}
    if args.resume and partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("ablation") != args.ablation:
            parser.error(
                f"partial results are for ablation {partial.get('ablation')}, "
                f"not {args.ablation}"
            )
        cached_results = {
            item["instance"]: Result(**item) for item in partial["results"]
        }
    results: list[Result] = []
    environment = os.environ.copy()
    environment["CUOPT_CPU_LNS_DIAGNOSTICS"] = "1"
    environment["CUOPT_CPU_LNS_ABLATION"] = args.ablation
    environment["PYTHONNOUSERSITE"] = "1"

    for index, name in enumerate(instances, start=1):
        if name in cached_results:
            result = cached_results[name]
            results.append(result)
            print(
                f"[{index:02d}/{len(instances):02d}] {name} resumed "
                f"feasible={int(result.feasible)}",
                flush=True,
            )
            continue
        instance_path = dataset_dir / f"{name}.mps"
        log_path = log_dir / f"{name}.log"
        command = [
            str(args.cuopt_cli),
            "--time-limit",
            str(args.time_limit),
            "--mip-heuristics-only",
            "true",
            "--num-cpu-threads",
            str(args.num_cpu_threads),
            "--random-seed",
            str(args.seed),
            "--mip-determinism-mode",
            str(args.determinism_mode),
            "--presolve",
            str(args.presolve),
            "--log-to-console",
            "1",
            str(instance_path),
        ]
        print(f"[{index:02d}/{len(instances):02d}] {name}", flush=True)
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        wall_time_s = time.monotonic() - start
        text = log_path.read_text(encoding="utf-8", errors="replace")
        result = Result(
            instance=name,
            attachment=name in ATTACHMENT_INSTANCES,
            file_bytes=instance_path.stat().st_size,
            return_code=completed.returncode,
            wall_time_s=wall_time_s,
        )
        parse_log(result, text, args.feasibility_tolerance)
        results.append(result)
        partial_path.write_text(
            json.dumps(
                {"ablation": args.ablation, "results": [asdict(item) for item in results]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        rate = (
            "NA"
            if result.iterations_per_second is None
            else f"{result.iterations_per_second:.1f}"
        )
        print(
            f"    feasible={int(result.feasible)} ips={rate} "
            f"wall={result.wall_time_s:.2f}s rc={result.return_code}",
            flush=True,
        )

    write_csv(output_dir / "results.csv", results)
    metadata = {
        "command": vars(args),
        "git": git_metadata(repo),
        "artifacts": {
            "cuopt_cli": str(args.cuopt_cli.resolve()),
            "cuopt_cli_sha256": sha256(args.cuopt_cli.resolve()),
            "libcuopt_sha256": sha256(args.cuopt_cli.resolve().parent / "libcuopt.so"),
        },
        "ablation": {
            "variant": args.ablation,
            "description": ABLATION_DESCRIPTIONS[args.ablation],
        },
        "instances": list(instances),
        "results": [asdict(result) for result in results],
        "summary": summarize(results, args.time_limit),
    }
    metadata["command"] = {key: str(value) for key, value in metadata["command"].items()}
    (output_dir / "results.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["summary"], indent=2, sort_keys=True))
    return 0 if not metadata["summary"]["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
