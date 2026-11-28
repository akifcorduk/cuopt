# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run gprs_check on the CSR dumps of testbed instances and list the checks.

The known objective value of an instance comes from the testbed file; the
dump of a maximisation holds the negated objective, so the value is negated
when the statistics of the dump (--stats-dir) say the instance maximises. A
value marked "*" is a best known one and is passed as an upper bound. An
instance listed as infeasible or unbounded is passed as such.

Solution files (<instance>.gprssol, see solution_file.hpp) are taken from
every --solutions-dir that has one for the instance; they feed the crush
check. With --make-solutions the tool solves the original MIP of every
instance instead and writes the solution files into the first
--solutions-dir.

Every tool run is killed after --kill-after seconds. A run record with the
identity of the run (tools/run_info.py) is written next to each result, and
--skip-done reuses a check result only if source tree, tool binary, command,
dump and solution files are those of the stored run. A solution file does
not depend on the engine and is tested when it is read: --skip-done keeps
it when command and dump are those of the stored run.

The summary of an earlier run is removed before anything runs, so a run
that is killed or ends with an error leaves no summary of other results.
With --require-points an instance that has a solution, according to the
testbed, and no usable point is a failure: without a solver its check is
the presolve status and the structure check alone.

--tool-arg passes one more argument to the tool, for example a flag of
tools/option_flags.hpp (--tool-arg=--primal-projections).
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "tools"))

import run_info  # noqa: E402


def parse_args():
    """Return the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--dumps", action="append", required=True)
    parser.add_argument("--stats-dir", action="append", default=[])
    parser.add_argument("--out")
    parser.add_argument(
        "--set", choices=["dev", "large", "status", "all"], default="dev"
    )
    parser.add_argument("--only")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lp-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-time-limit", type=float, default=60.0)
    parser.add_argument("--primal-only", action="store_true")
    parser.add_argument("--solutions-dir", action="append", default=[])
    parser.add_argument("--make-solutions", action="store_true")
    parser.add_argument("--solution-time-limit", type=float, default=300.0)
    parser.add_argument("--max-points", type=int, default=20)
    parser.add_argument("--cost-variants", type=int, default=2)
    parser.add_argument("--push-points", type=int, default=4)
    parser.add_argument("--require-points", action="store_true")
    parser.add_argument("--kill-after", type=float)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument(
        "--testbed", default=str(PROJECT_DIR / "testbed" / "testbed.json")
    )
    parser.add_argument("--tool-arg", action="append", default=[])
    args = parser.parse_args()
    if args.make_solutions and not args.solutions_dir:
        parser.error("--make-solutions needs a --solutions-dir")
    if not args.make_solutions and not args.out:
        parser.error("a check needs --out")
    if args.kill_after is None:
        # Twice what the limits of the solver add up to, and time for the
        # presolves, the dump and the solution files of a large instance.
        solver = (
            args.solution_time_limit
            if args.make_solutions
            else 2 * args.lp_time_limit + args.mip_time_limit
        )
        args.kill_after = 2 * solver + 1200.0
    return args


def select(testbed, set_name, only):
    """Return the testbed entries of the set, restricted to the given names."""
    main = testbed["main"]
    sets = {
        "dev": [e for e in main if e["where"] == "dev"],
        "large": [e for e in main if e["where"] == "dev-large"],
        "status": testbed["status"],
        "all": main + testbed["status"],
    }
    selected = sets[set_name]
    if only:
        names = set(only.split(","))
        unknown = names - {e["name"] for e in selected}
        if unknown:
            listed = ", ".join(sorted(unknown))
            sys.exit("not in the set %s: %s" % (set_name, listed))
        selected = [e for e in selected if e["name"] in names]
    return selected


def find_file(name, dirs):
    """Return the first existing file of that name in the directories."""
    for d in dirs:
        path = Path(d) / name
        if path.is_file():
            return path
    return None


def known(entry, stats_dirs):
    """Return what is known of the instance as arguments of the tool.

    An error text instead when the objective value cannot be used.
    """
    text = str(entry.get("objective", ""))
    if text in ("Infeasible", "Unbounded"):
        return ["--expect", text.lower()], None
    try:
        value = float(text.rstrip("*"))
    except ValueError:
        return [], None
    stats = find_file(entry["name"] + ".json", stats_dirs)
    sense = None
    if stats is not None:
        with open(stats) as f:
            sense = json.load(f).get("objective_sense")
    if sense not in ("min", "max"):
        return None, "no objective sense in the statistics directories"
    arguments = [
        "--known-objective",
        repr(-value if sense == "max" else value),
    ]
    if text.endswith("*"):
        arguments.append("--known-is-bound")
    return arguments, None


def command_of(args, entry, out, dump):
    """Return the tool command of the instance and its error, if any."""
    name = entry["name"]
    if args.make_solutions:
        return [
            args.tool,
            "--solve",
            str(dump),
            str(out / (name + ".gprssol")),
            "--mip-time-limit",
            str(args.solution_time_limit),
            "--threads",
            str(args.threads),
            "--max-points",
            str(args.max_points),
        ], None
    command = [
        args.tool,
        str(dump),
        str(out / (name + ".json")),
        "--lp-time-limit",
        str(args.lp_time_limit),
        "--mip-time-limit",
        str(args.mip_time_limit),
        "--threads",
        str(args.threads),
        "--cost-variants",
        str(args.cost_variants),
        "--push-points",
        str(args.push_points),
    ]
    if args.primal_only:
        command.append("--primal-only")
    command += args.tool_arg
    arguments, error = known(entry, args.stats_dir)
    if error:
        return command, error
    command += arguments
    for d in args.solutions_dir:
        path = Path(d) / (name + ".gprssol")
        if path.is_file():
            command += ["--solutions", str(path)]
    return command, None


def inputs_of(command, dump):
    """Return size and time of the dump and the solution files of a command."""
    files = [str(dump)] + [
        command[i + 1] for i, a in enumerate(command) if a == "--solutions"
    ]
    return [
        [f, Path(f).stat().st_size, Path(f).stat().st_mtime] for f in files
    ]


def run_one(args, identity, entry, out):
    """Run the tool on one instance and write its run record."""
    name = entry["name"]
    record_path = out / (name + ".run.json")
    report = out / (
        name + (".gprssol.json" if args.make_solutions else ".json")
    )
    dump = find_file(name + ".csr", args.dumps)
    record = {"name": name, "identity": identity}
    if dump is None:
        record["error"] = "no dump"
    else:
        command, error = command_of(args, entry, out, dump)
        record["command"] = command
        record["inputs"] = inputs_of(command, dump)
        if error:
            record["error"] = error
    if args.skip_done and "error" not in record and record_path.is_file():
        with open(record_path) as f:
            old = json.load(f)
        if (
            old.get("exit_code") == 0
            and report.is_file()
            and (
                args.make_solutions
                or run_info.same_identity(old.get("identity"), identity)
            )
            and old.get("command") == record["command"]
            and old.get("inputs") == record["inputs"]
        ):
            return
    if "error" not in record:
        report.unlink(missing_ok=True)
        record.update(run_info.host_state())
        start = time.time()
        with open(out / (name + ".log"), "w") as log:
            try:
                done = subprocess.run(
                    record["command"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.kill_after,
                    check=False,
                )
                record["exit_code"] = done.returncode
                if done.returncode != 0:
                    record["error"] = "exit code %d" % done.returncode
            except subprocess.TimeoutExpired:
                record["exit_code"] = None
                record["error"] = "killed after %g s" % args.kill_after
        record["wall_seconds"] = time.time() - start
        record["end_load_average"] = run_info.host_state()["load_average"]
    tmp = record_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(record, f)
    tmp.rename(record_path)
    print("%s: %s" % (name, record.get("error") or "ran"), flush=True)


def load_result(args, entry, out):
    """Return the run record and the report of an instance, or the error."""
    name = entry["name"]
    record_path = out / (name + ".run.json")
    report = out / (
        name + (".gprssol.json" if args.make_solutions else ".json")
    )
    if not record_path.is_file():
        return None, None, "not run"
    with open(record_path) as f:
        record = json.load(f)
    if "error" in record:
        return record, None, record["error"]
    if not report.is_file():
        return record, None, "no result"
    with open(report) as f:
        return record, json.load(f), None


def run_summary(record):
    """Return when and under which load a result was made."""
    keys = (
        "host",
        "start_time",
        "load_average",
        "end_load_average",
        "wall_seconds",
        "exit_code",
    )
    return {k: record.get(k) for k in keys}


def has_solution(entry):
    """Return whether the testbed lists the instance with a solution."""
    return entry.get("objective") not in ("Infeasible", "Unbounded")


def list_solutions(args, entries, out):
    """Print one line per instance and return rows and the failure count."""
    rows = []
    failures = 0
    for entry in entries:
        record, r, error = load_result(args, entry, out)
        row = {"name": entry["name"]}
        if record:
            row["run"] = run_summary(record)
        if error:
            row["error"] = error
            failures += 1
            print("%-24s %s" % (entry["name"][:24], error))
        else:
            row.update(r)
            mip = r["mip_original"]
            print(
                "%-24s %-20.20s points %2d rejected %2d best %s"
                % (
                    entry["name"][:24],
                    mip["status"],
                    r["points_written"],
                    r["points_rejected"],
                    r["best_point_objective"],
                )
            )
        rows.append(row)
    with_points = sum(1 for r in rows if r.get("points_written"))
    print(
        "%d instances, %d with a solution file that has points, %d failed"
        % (len(entries), with_points, failures)
    )
    return rows, failures


def list_checks(args, entries, out):
    """Print one line per instance and return rows and the failure count."""
    rows = []
    failures = 0
    with_points = 0
    for entry in entries:
        record, r, error = load_result(args, entry, out)
        row = {"name": entry["name"]}
        if record:
            row["run"] = run_summary(record)
        if error:
            row["error"] = error
            failures += 1
            print("%-24s %s" % (entry["name"][:24], error))
            rows.append(row)
            continue
        row.update(r)
        rows.append(row)
        failed = [k for k, v in r["checks"].items() if v == "FAIL"]
        passed = sum(1 for v in r["checks"].values() if v == "pass")
        used = r["crush"]["points_used"]
        with_points += 1 if used else 0
        if args.require_points and not used and has_solution(entry):
            failed.append("no usable point")
            row["no_usable_point"] = True
        failures += 1 if failed else 0
        print(
            "%-24s %-10.10s lp %-12.12s -> %-12.12s mip %-18.18s points %2d "
            "passed %2d%s"
            % (
                entry["name"][:24],
                r["presolve_status"],
                r["lp_original"]["status"],
                r["lp_reduced"]["status"],
                r["mip_reduced"]["status"],
                used,
                passed,
                "  FAILED: " + ", ".join(failed) if failed else "",
            )
        )
    print(
        "%d instances, %d with a usable point for the crush check, "
        "%d with a failed check or without a result"
        % (len(entries), with_points, failures)
    )
    return rows, failures


def main():
    """Run the tool on the instances, list the results, write the summary."""
    args = parse_args()
    with open(args.testbed) as f:
        entries = select(json.load(f), args.set, args.only)
    if args.make_solutions:
        # An instance without a solution has none to look for.
        entries = [e for e in entries if has_solution(e)]
    out = Path(args.solutions_dir[0] if args.make_solutions else args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").unlink(missing_ok=True)
    identity = run_info.run_identity(args.tool)
    if identity["tool_checksum"] is None:
        sys.exit("cannot read the tool %s" % args.tool)
    summary = {"identity": identity}
    summary.update(run_info.host_state())
    summary["arguments"] = {
        k: v for k, v in vars(args).items() if k not in ("tool", "testbed")
    }
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda e: run_one(args, identity, e, out), entries))

    if args.make_solutions:
        rows, failures = list_solutions(args, entries, out)
    else:
        rows, failures = list_checks(args, entries, out)
    summary["end_time"] = run_info.host_state()["start_time"]
    summary["end_load_average"] = run_info.host_state()["load_average"]
    summary["failures"] = failures
    summary["instances"] = rows
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
