# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical quality of reduced models: run tools/gprs_quality, compare two runs.

Usage: quality_compare.py run <gprs_quality> <dump dir> <out.jsonl> [--jobs N]
                              [tool option]...
       quality_compare.py compare <before.jsonl> <after.jsonl> [--list N]

run      the quality tool on every *.csr of the directory, one JSON line per
         instance (see tools/gprs_quality.cpp for the fields); options after
         the output file go to the tool.
compare  two such files over the instances that both hold: sums of nonzeros,
         objective nonzeros, rows above the coefficient ratios, equations,
         ranged rows, singleton and free columns for the input and the two
         reduced models; the geometric mean of after / before of the
         coefficient range of the model, of the largest ratio within a row
         and of the cost range; and per measure (coefficient range of the
         model, largest and 99% ratio of the largest to the smallest absolute
         coefficient within a row, rows with a ratio above 1e6, largest side,
         objective nonzeros, cost range, largest and smallest cost) the
         number of instances where it is larger and where it is smaller
         after, with the --list (default 10) largest changes by name.
"""

import argparse
import glob
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SUMS = (
    "nnz",
    "obj_nnz",
    "rows_dyn_1e3",
    "rows_dyn_1e4",
    "rows_dyn_1e6",
    "rows_dyn_1e8",
    "equations",
    "ranged",
    "singleton_cols",
    "free_cols",
)


def run(tool, dumps, out, jobs, options):
    """Write one line of the quality tool per dump of the directory."""
    paths = sorted(glob.glob(dumps + "/*.csr"))

    def one(path):
        done = subprocess.run(
            [tool, path] + options, capture_output=True, text=True
        )
        if done.returncode != 0:
            sys.exit("%s: exit code %d" % (path, done.returncode))
        return done.stdout.strip()

    with ThreadPoolExecutor(jobs) as pool, open(out, "w") as f:
        for line in pool.map(one, paths):
            f.write(line + "\n")
    print("%s: %d instances" % (out, len(paths)))


def load(path):
    """Return the lines of a run by instance."""
    with open(path) as f:
        return {r["instance"]: r for r in map(json.loads, f)}


def spread(row, low, high):
    """Return the ratio of two fields, 1 when the smaller one is zero."""
    return row[high] / row[low] if row[low] > 0 else 1.0


def compare(before, after, listed):
    """Print the comparison of the reduced models of two runs."""
    a, b = load(before), load(after)
    names = sorted(set(a) & set(b))
    print("%d instances in both runs" % len(names))
    print("%-22s %12s %12s %12s" % ("sums", "input", "before", "after"))
    for key in SUMS:
        print(
            "%-22s %12d %12d %12d"
            % (
                key,
                sum(a[n]["input"][key] for n in names),
                sum(a[n]["reduced"][key] for n in names),
                sum(b[n]["reduced"][key] for n in names),
            )
        )
    measures = (
        "coefficient range of the model",
        "largest ratio within a row",
        "99% ratio within a row",
        "rows with a ratio above 1e6",
        "largest side",
        "objective nonzeros",
        "cost range",
        "largest cost",
        "smallest cost (as its inverse)",
    )
    larger = {m: [] for m in measures}
    smaller = {m: 0 for m in measures}
    logs = {"model": [], "row": [], "cost": []}
    for n in names:
        x, y = a[n]["reduced"], b[n]["reduced"]
        if x["nnz"] == 0 or y["nnz"] == 0:
            continue
        mx = spread(x, "coef_min", "coef_max")
        my = spread(y, "coef_min", "coef_max")
        logs["model"].append(math.log(my / mx))
        logs["row"].append(math.log(y["row_dyn_max"] / x["row_dyn_max"]))
        values = [mx, my, x["row_dyn_max"], y["row_dyn_max"]]
        values += [x["row_dyn_p99"], y["row_dyn_p99"]]
        values += [x["rows_dyn_1e6"], y["rows_dyn_1e6"]]
        values += [x["side_max"], y["side_max"], x["obj_nnz"], y["obj_nnz"]]
        if x["obj_nnz"] and y["obj_nnz"]:
            cx = spread(x, "cost_min", "cost_max")
            cy = spread(y, "cost_min", "cost_max")
            logs["cost"].append(math.log(cy / cx))
            values += [cx, cy, x["cost_max"], y["cost_max"]]
            values += [1 / x["cost_min"], 1 / y["cost_min"]]
        for k in range(len(values) // 2):
            u, v = values[2 * k], values[2 * k + 1]
            if v > u * (1 + 1e-9):
                ratio = v / u if u > 0 else math.inf
                larger[measures[k]].append((ratio, n, u, v))
            elif v < u * (1 - 1e-9):
                smaller[measures[k]] += 1

    def mean(values):
        return math.exp(sum(values) / len(values)) if values else 1.0

    print(
        "geometric mean of after / before: coefficient range of the model "
        "%.4f, largest ratio within a row %.4f (%d instances with a "
        "nonzero in both), cost range %.4f (%d instances with a cost in both)"
        % (
            mean(logs["model"]),
            mean(logs["row"]),
            len(logs["model"]),
            mean(logs["cost"]),
            len(logs["cost"]),
        )
    )
    for measure in measures:
        items = sorted(larger[measure], reverse=True)
        print(
            "%s: larger after on %d instances, smaller on %d"
            % (measure, len(items), smaller[measure])
        )
        for ratio, n, u, v in items[:listed]:
            print("   %-30s %.6g -> %.6g (x %.4g)" % (n, u, v, ratio))


def main():
    """Run the tool or compare two runs."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["run", "compare"])
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("third", nargs="?")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--list", type=int, default=10)
    args, options = parser.parse_known_args()
    if args.command == "run":
        if args.third is None:
            parser.error("run needs a tool, a dump directory and an output")
        run(args.first, args.second, args.third, args.jobs, options)
    else:
        compare(args.first, args.second, args.list)


if __name__ == "__main__":
    main()
