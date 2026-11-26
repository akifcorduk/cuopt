# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the targets of the Stage 1 and Stage 2 gates and write them as CSV.

Reads the HiGHS reference runs made with the extended diagnostic patch
(results/stage1/refs) and full HiGHS (results/stage0/diag, stock). Takes no
arguments. Sizes are those HiGHS holds at the end of presolve (DIAG total
line). The time limit of an instance is 1.2 * t + 0.02 s.
"""

import csv
import json
import math
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
STAGE0 = RESULTS / "stage0"
REFS = RESULTS / "stage1" / "refs"
SHIFT = 0.1


def load(path):
    """Return the instances of a runner summary by name."""
    with open(path) as f:
        return {r["name"]: r for r in json.load(f)["instances"]}


def sizes(row):
    """Return rows, columns and nonzeros after presolve from the DIAG total line."""
    total = row["diag"]["total"][0]
    return [int(total[k].split(":")[1]) for k in ("rows", "cols", "nnz")]


def shifted_mean(values):
    """Return the shifted geometric mean of the values."""
    return (
        math.exp(sum(math.log(v + SHIFT) for v in values) / len(values))
        - SHIFT
    )


def geometric_mean(values):
    """Return the geometric mean of the values."""
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    """Print the tables and write targets_stage1.csv."""
    full = load(STAGE0 / "diag" / "summary.json")
    stock = load(STAGE0 / "stock" / "summary.json")
    variants = {
        "stage1c": "Stage 1 reference",
        "stage1d": "Stage 1 reference, dual fixing off",
        "stage1c_redexp": "Stage 1 reference, redundancy on explicit sums",
        "stage12c": "Stage 2 reference",
    }
    runs = {v: load(REFS / v / "summary.json") for v in variants}
    names = sorted(full, key=lambda n: full[n]["nnz"])

    print("Remaining nonzeros against full HiGHS, and time")
    print(
        "%-48s %8s %8s %6s %9s %9s"
        % ("variant", "geomean", "median", ">1%", "sum s", "shifted")
    )
    for v, label in variants.items():
        ratio = [
            sizes(runs[v][n])[2] / max(sizes(full[n])[2], 1) for n in names
        ]
        seconds = [runs[v][n]["presolve_core_seconds"] for n in names]
        print(
            "%-48s %8.3f %8.3f %6d %9.1f %9.3f"
            % (
                label,
                geometric_mean(ratio),
                statistics.median(ratio),
                sum(1 for r in ratio if r > 1.01),
                sum(seconds),
                shifted_mean(seconds),
            )
        )

    print()
    print("Stage 1 reference, time per band (shifted mean, median, maximum)")
    bands = {}
    for n in names:
        bands.setdefault(full[n]["band"], []).append(n)
    for band in sorted(bands):
        seconds = [
            runs["stage1c"][n]["presolve_core_seconds"] for n in bands[band]
        ]
        print(
            "%-3s %7.3f %7.3f %7.3f"
            % (
                band,
                shifted_mean(seconds),
                statistics.median(seconds),
                max(seconds),
            )
        )

    header = [
        "band",
        "instance",
        "rows",
        "cols",
        "nnz",
        "full_rows",
        "full_cols",
        "full_nnz",
        "full_seconds",
        "full_limit_seconds",
        "stage1_rows",
        "stage1_cols",
        "stage1_nnz",
        "stage1_seconds",
        "stage1_limit_seconds",
        "stage2_rows",
        "stage2_cols",
        "stage2_nnz",
        "stage2_seconds",
        "stage2_limit_seconds",
    ]
    rows = []
    for n in names:
        t_full = min(
            stock[n]["presolve_core_seconds"], full[n]["presolve_core_seconds"]
        )
        t1 = runs["stage1c"][n]["presolve_core_seconds"]
        t2 = runs["stage12c"][n]["presolve_core_seconds"]
        rows.append(
            [
                full[n]["band"],
                n,
                full[n]["rows"],
                full[n]["cols"],
                full[n]["nnz"],
            ]
            + sizes(full[n])
            + ["%.6f" % t_full, "%.3f" % (1.2 * t_full + 0.02)]
            + sizes(runs["stage1c"][n])
            + ["%.6f" % t1, "%.3f" % (1.2 * t1 + 0.02)]
            + sizes(runs["stage12c"][n])
            + ["%.6f" % t2, "%.3f" % (1.2 * t2 + 0.02)]
        )
    out = RESULTS / "stage1" / "targets_stage1.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print()
    print(
        "%-22s %9s | %9s %8s | %9s %8s %8s"
        % ("instance", "nnz", "stage1", "s", "stage2", "s", "full")
    )
    for r in rows:
        print(
            "%-22s %9d | %9d %8.3f | %9d %8.3f %8d"
            % (r[1][:22], r[4], r[12], float(r[13]), r[17], float(r[18]), r[7])
        )
    print("written: %s" % out)


if __name__ == "__main__":
    main()
