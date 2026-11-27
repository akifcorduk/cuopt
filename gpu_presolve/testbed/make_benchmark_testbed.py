# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make a testbed file for the MIPLIB 2017 benchmark set that cuOpt is run on.

The runners (tools/run_engine.py, check/run_check.py) take any testbed file
with --testbed. This one lists every MPS file of the benchmark directory:
sizes from the statistics of baseline_tool, which also writes the CSR dump,
and the objective from cuOpt's table of best known values
(benchmarks/linear_programming/cuopt/miplib2017_bks.hpp). A best known value
is written with a trailing "*": the check takes it as a bound, not as the
optimum. Instances the table lists as infeasible go to the "status" list
with the objective "Infeasible".

Run on the node, from the worktree root:
  python3 gpu_presolve/testbed/make_benchmark_testbed.py --jobs 8
An instance whose statistics exist is not run again.
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEPS = Path("/home/scratch.acoerduek_sw/gpu_presolve_deps")
BKS = (
    PROJECT_DIR.parent
    / "benchmarks"
    / "linear_programming"
    / "cuopt"
    / "miplib2017_bks.hpp"
)
ENTRY = re.compile(r'\{\s*"([^"]+)"\s*,\s*([^}]+?)\s*\}')
NAME = re.compile(r'^\s*"([^"]+)"\s*,\s*$')


def parse_args():
    """Return the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="/home/scratch.acoerduek_sw/cuopt_dev/datasets/mip/miplib2017",
    )
    parser.add_argument(
        "--tool", default=str(DEPS / "build/baseline_stock/baseline_tool")
    )
    parser.add_argument("--dumps-dir", default=str(DEPS / "csr_dumps_bench"))
    parser.add_argument(
        "--stats-dir", default=str(DEPS / "results/bench240/stats")
    )
    parser.add_argument(
        "--out", default=str(PROJECT_DIR / "testbed" / "benchmark240.json")
    )
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def known_values():
    """Return the best known objective by name and the infeasible names."""
    text = BKS.read_text()
    start = text.index("kBenchmarkInfeasible()")
    values = {}
    for name, value in ENTRY.findall(text[:start]):
        try:
            values[name.lower()] = float(value)
        except ValueError:
            continue
    block = text[start : text.index("};", start)]
    infeasible = {
        m.group(1).lower()
        for line in block.splitlines()
        if (m := NAME.match(line))
    }
    return values, infeasible


def statistics(args, mps):
    """Run baseline_tool on one instance unless its statistics exist."""
    name = mps.name[: -len(".mps")]
    stats = Path(args.stats_dir) / (name + ".json")
    dump = Path(args.dumps_dir) / (name + ".csr")
    if not (stats.is_file() and dump.is_file()):
        done = subprocess.run(
            [
                args.tool,
                str(mps),
                str(stats),
                "--stats-only",
                "--dump-csr",
                str(dump),
            ],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0 or not stats.is_file():
            return name, None, done.stderr.strip()[-300:]
    with open(stats) as f:
        return name, json.load(f), None


def band(nnz):
    """Return a size class: S below the target sizes, then A, B and C."""
    if nnz < 5e4:
        return "S"
    if nnz < 5e5:
        return "A"
    if nnz < 1e7:
        return "B"
    return "C"


def main():
    """Write the testbed file and report what could not be listed."""
    args = parse_args()
    values, infeasible = known_values()
    Path(args.stats_dir).mkdir(parents=True, exist_ok=True)
    Path(args.dumps_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.dataset_dir).glob("*.mps"))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda f: statistics(args, f), files))

    main_list = []
    status_list = []
    failed = []
    for name, stats, error in results:
        if stats is None:
            failed.append((name, error))
            continue
        original = stats["original"]
        entry = {
            "name": name,
            "band": band(original["nnz"]),
            "rows": original["rows"],
            "cols": original["cols"],
            "nnz": original["nnz"],
            "where": "dev" if original["nnz"] < 1e7 else "dev-large",
            "in_benchmark_set": True,
        }
        if name.lower() in infeasible:
            entry.update(role="status", objective="Infeasible")
            status_list.append(entry)
        else:
            entry["role"] = "main"
            if name.lower() in values:
                entry["objective"] = repr(values[name.lower()]) + "*"
            main_list.append(entry)
    main_list.sort(key=lambda e: e["nnz"])
    testbed = {
        "source": "MIPLIB 2017 benchmark set as cuOpt is run on it; sizes "
        "from baseline_tool, objectives from miplib2017_bks.hpp",
        "dataset_dir": args.dataset_dir,
        "extra_dataset_dirs": [],
        "main": main_list,
        "status": status_list,
        "heldout": [],
    }
    with open(args.out, "w") as f:
        json.dump(testbed, f, indent=1)
        f.write("\n")
    without = [e["name"] for e in main_list if "objective" not in e]
    print(
        "%d instances: %d main (%d without a known objective), %d infeasible, "
        "%d failed"
        % (
            len(files),
            len(main_list),
            len(without),
            len(status_list),
            len(failed),
        )
    )
    for name, error in failed:
        print("failed: %s: %s" % (name, error))
    counts = {}
    for e in main_list + status_list:
        counts[e["band"]] = counts.get(e["band"], 0) + 1
    print("by size class:", dict(sorted(counts.items())))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
