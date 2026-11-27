# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run gprs_check on the CSR dumps of testbed instances and list the checks.

The known objective value of an instance comes from the testbed file; the
dump of a maximisation holds the negated objective, so the value is negated
when the statistics of the dump (--stats-dir) say the instance maximises.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    """Return the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--dumps", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lp-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-time-limit", type=float, default=60.0)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument(
        "--testbed", default=str(PROJECT_DIR / "testbed" / "testbed.json")
    )
    return parser.parse_args()


def known_objective(entry, stats_dir):
    """Return the known objective of the minimisation in the dump, or None."""
    text = str(entry.get("objective", "")).rstrip("*")
    try:
        value = float(text)
    except ValueError:
        return None
    stats = Path(stats_dir) / (entry["name"] + ".json")
    if not stats.is_file():
        return None
    with open(stats) as f:
        sense = json.load(f).get("objective_sense")
    if sense not in ("min", "max"):
        return None
    return -value if sense == "max" else value


def run_one(args, entry, out):
    """Run the tool on one instance."""
    name = entry["name"]
    report = out / (name + ".json")
    if args.skip_done and report.is_file():
        return
    dump = Path(args.dumps) / (name + ".csr")
    command = [
        args.tool,
        str(dump),
        str(report),
        "--lp-time-limit",
        str(args.lp_time_limit),
        "--mip-time-limit",
        str(args.mip_time_limit),
        "--threads",
        str(args.threads),
    ]
    known = known_objective(entry, args.stats_dir)
    if known is not None:
        command += ["--known-objective", repr(known)]
    with open(out / (name + ".log"), "w") as log:
        done = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    print("%s: exit %d" % (name, done.returncode), flush=True)


def main():
    """Run the checks and print one line per instance."""
    args = parse_args()
    with open(args.testbed) as f:
        entries = [e for e in json.load(f)["main"] if e["where"] == "dev"]
    if args.only:
        names = set(args.only.split(","))
        entries = [e for e in entries if e["name"] in names]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda e: run_one(args, e, out), entries))

    failures = 0
    rows = []
    for entry in entries:
        report = out / (entry["name"] + ".json")
        if not report.is_file():
            print("%-24s no result" % entry["name"])
            failures += 1
            continue
        with open(report) as f:
            r = json.load(f)
        r["name"] = entry["name"]
        rows.append(r)
        failed = [k for k, v in r["checks"].items() if v == "FAIL"]
        passed = sum(1 for v in r["checks"].values() if v == "pass")
        failures += 1 if failed else 0
        print(
            "%-24s lp %-12.12s -> %-12.12s mip %-18.18s passed %d%s"
            % (
                entry["name"][:24],
                r["lp_original"]["status"],
                r["lp_reduced"]["status"],
                r["mip_reduced"]["status"],
                passed,
                "  FAILED: " + ", ".join(failed) if failed else "",
            )
        )
    with open(out / "summary.json", "w") as f:
        json.dump({"instances": rows}, f, indent=1)
    print("%d instances, %d with a failed check" % (len(entries), failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
