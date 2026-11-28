# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run gprs_presolve on the CSR dumps of testbed instances and summarize.

Every instance is run --reps times; the repetitions go round the whole set, so
the runs of one instance are not back to back. The summary keeps the smallest
presolve time, all times, and whether the reduced-model hash is the same in
every repetition. Results are written per repetition, so a restart with
--skip-done continues where the last run stopped.

Every run record and the summary hold the identity of the run
(run_info.py: commit of the source tree, checksum of the tool binary, host,
start time, load average). --skip-done reuses a repetition only if source
tree, tool binary, command and dump are those of the stored run. On the
main set a status without a reduced model (anything but reduced, unchanged,
reduced_to_empty) is a failure. The summary of an earlier run is removed
before anything runs, so a run that is killed or ends with an error leaves
no summary of other results.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_info

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_STATUSES = ("reduced", "unchanged", "reduced_to_empty")


def parse_args():
    """Return the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--dumps", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--set", choices=["dev", "large", "status", "all"], default="dev"
    )
    parser.add_argument("--only")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--kill-after", type=float, default=3600.0)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--tool-arg", action="append", default=[])
    parser.add_argument(
        "--testbed", default=str(PROJECT_DIR / "testbed" / "testbed.json")
    )
    return parser.parse_args()


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


def find_dump(name, dirs):
    """Return the dump file of the instance, or None."""
    for d in dirs:
        path = Path(d) / (name + ".csr")
        if path.is_file():
            return path
    return None


def run_one(args, identity, entry, rep, out):
    """Run one repetition of one instance and write its run record."""
    name = entry["name"]
    record_path = out / ("%s.rep%d.run.json" % (name, rep))
    dump = find_dump(name, args.dumps)
    report = out / ("%s.rep%d.json" % (name, rep))
    command = [args.tool, str(dump), "--json", str(report)]
    if args.time_limit is not None:
        command += ["--time-limit", str(args.time_limit)]
    command += args.tool_arg
    inputs = None
    if dump is not None:
        inputs = [str(dump), dump.stat().st_size, dump.stat().st_mtime]
    if args.skip_done and record_path.is_file():
        with open(record_path) as f:
            old = json.load(f)
        if (
            "report" in old
            and run_info.same_identity(old.get("identity"), identity)
            and old.get("command") == command
            and old.get("inputs") == inputs
        ):
            return
    record = {"name": name, "rep": rep, "identity": identity}
    if dump is None:
        record["error"] = "no dump"
    else:
        record["command"] = command
        record["inputs"] = inputs
        record.update(run_info.host_state())
        report.unlink(missing_ok=True)
        start = time.time()
        with open(out / ("%s.rep%d.log" % (name, rep)), "w") as log:
            try:
                done = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    timeout=args.kill_after,
                )
                record["exit_code"] = done.returncode
            except subprocess.TimeoutExpired:
                record["exit_code"] = None
                record["error"] = "killed after %g s" % args.kill_after
        record["wall_seconds"] = time.time() - start
        record["end_load_average"] = run_info.host_state()["load_average"]
        if record["exit_code"] == 0 and report.is_file():
            with open(report) as f:
                record["report"] = json.load(f)
        elif "error" not in record:
            record["error"] = "exit code %s" % record["exit_code"]
    tmp = record_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(record, f)
    tmp.rename(record_path)
    print(
        "%s rep %d: %s"
        % (
            name,
            rep,
            record.get("error")
            or "%.3f s, %d nnz"
            % (
                record["report"]["seconds"],
                record["report"]["reduced"]["nnz"],
            ),
        ),
        flush=True,
    )


def summarize(args, summary, entries, out):
    """Write summary.json from the run records and return the failure count."""
    instances = []
    failures = 0
    for entry in entries:
        name = entry["name"]
        row = {
            "name": name,
            "band": entry.get("band"),
            "rows": entry["rows"],
            "cols": entry["cols"],
            "nnz": entry["nnz"],
        }
        reports = []
        errors = []
        runs = []
        for rep in range(args.reps):
            path = out / ("%s.rep%d.run.json" % (name, rep))
            if not path.is_file():
                errors.append("rep %d: not run" % rep)
                continue
            with open(path) as f:
                record = json.load(f)
            runs.append(
                {
                    k: record.get(k)
                    for k in (
                        "rep",
                        "host",
                        "start_time",
                        "load_average",
                        "end_load_average",
                        "wall_seconds",
                    )
                }
            )
            if "report" in record:
                reports.append(record["report"])
            else:
                errors.append("rep %d: %s" % (rep, record.get("error")))
        row["runs"] = runs
        failed = bool(errors)
        if reports:
            best = min(reports, key=lambda r: r["seconds"])
            row.update(
                {
                    "status": best["status"],
                    "reduced": best["reduced"],
                    "seconds": best["seconds"],
                    "all_seconds": [r["seconds"] for r in reports],
                    "hash": best["hash"],
                    "same_hash": len({r["hash"] for r in reports}) == 1,
                    "rounds": best["rounds"],
                    "rebuilds": best["rebuilds"],
                    "rules": best["rules"],
                }
            )
            if "seconds_outside_rules" in best:
                row["seconds_outside_rules"] = best["seconds_outside_rules"]
            # Reports of engines before the start step had its own entry do not have these.
            for key in (
                "start",
                "fixed_point",
                "explicit",
                "parallel_rows",
                "dual_findings",
                "sub_rules",
                "columns",
                "col_from_rows",
                "objective_nnz",
                "options",
                "column_counts",
                "left",
                "second",
            ):
                if key in best:
                    row[key] = best[key]
            failed = failed or not row["same_hash"]
            statuses = sorted({r["status"] for r in reports})
            if entry.get("role") == "main" and any(
                s not in MODEL_STATUSES for s in statuses
            ):
                errors.append(
                    "status without a reduced model: %s" % ", ".join(statuses)
                )
                failed = True
        if errors:
            row["errors"] = errors
        failures += 1 if failed else 0
        instances.append(row)
    summary.update(
        {
            "tool": args.tool,
            "tool_args": args.tool_arg,
            "reps": args.reps,
            "end_time": run_info.host_state()["start_time"],
            "end_load_average": run_info.host_state()["load_average"],
            "failures": failures,
            "instances": instances,
        }
    )
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    for row in instances:
        for error in row.get("errors", []):
            print("%s: %s" % (row["name"], error))
    return failures


def main():
    """Run the repetitions and write the summary."""
    args = parse_args()
    with open(args.testbed) as f:
        entries = select(json.load(f), args.set, args.only)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").unlink(missing_ok=True)
    identity = run_info.run_identity(args.tool)
    if identity["tool_checksum"] is None:
        sys.exit("cannot read the tool %s" % args.tool)
    summary = {"identity": identity}
    summary.update(run_info.host_state())
    for rep in range(args.reps):
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(
                pool.map(
                    lambda e: run_one(args, identity, e, rep, out), entries
                )
            )
    failures = summarize(args, summary, entries, out)
    print("%d instances, %d with a failure" % (len(entries), failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
