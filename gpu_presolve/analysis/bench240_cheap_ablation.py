# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank HiGHS's cheap presolve rules by what the 240 instances lose without each.

Usage: bench240_cheap_ablation.py
Reads results/bench240/cheap_ablation: all_on.csv is HiGHS with its
expensive stages off (probing, enumeration, clique rules, aggregator,
sparsify, Fourier-Motzkin, inequality strengthening, pairwise dominated
columns, the implied-integer sweep), and each off_<rule>.csv the same with
one more rule off (the summary.csv files of baseline/run_baseline.py with the
instrumented HiGHS build). Per rule: the geometric mean of remaining
nonzeros, rows and columns without the rule over those with it, the number
of instances that change, the change of HiGHS's summed presolve seconds, and
the largest losses. ex9, ex10 and enlight_hard are left out: a full presolve
reduces them to almost nothing and they would carry a geometric mean.
"""

import csv
import math
import sys
from pathlib import Path

DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "bench240"
    / "cheap_ablation"
)
LEFT_OUT = {"ex9", "ex10", "enlight_hard"}
RULES = {
    "off_rule5": "dominated column",
    "off_rule7": "forcing column",
    "off_rule8": "free column substitution",
    "off_rule17": "dual fixing (all of it)",
    "off_rule18": "zero-cost singleton",
    "off_rule19": "column stuffing",
    "off_rule23": "weakly dominated column",
    "off_parallel_columns": "parallel columns",
    "off_simple_probing": "simple probing on equations",
    "off_dual_fixing_substitution": "dual-fixing substitution",
    "off_implied_equations": "implied equations",
    "off_implied_integer_conversion": "implied-integer conversion",
}


def load(path):
    """Return rows, columns, nonzeros after presolve and seconds by name."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("reached_time_limit") in ("1", "True", "true"):
                continue
            sizes = [
                int(float(r[key])) if r.get(key, "") != "" else 0
                for key in ("rows_after", "cols_after", "nnz_after")
            ]
            out[r["name"]] = (sizes, float(r.get("presolve_seconds") or 0.0))
    return out


def main():
    """Print one line per rule, the most valuable first."""
    base = load(DIR / "all_on.csv")
    print(
        "all cheap rules on: %d instances, %.0f s"
        % (len(base), sum(v[1] for v in base.values()))
    )
    lines = []
    for path in sorted(DIR.glob("off_*.csv")):
        off = load(path)
        common = [n for n in base if n in off and n not in LEFT_OUT]

        def ratio(n, k):
            return max(off[n][0][k], 1) / max(base[n][0][k], 1)

        means = [
            math.exp(sum(math.log(ratio(n, k)) for n in common) / len(common))
            for k in range(3)
        ]
        changed = sum(1 for n in common if off[n][0] != base[n][0])
        seconds = sum(off[n][1] - base[n][1] for n in common)
        worst = sorted(common, key=lambda n: -ratio(n, 2))[:3]
        lines.append(
            (
                means[2],
                RULES.get(path.stem, path.stem),
                means,
                changed,
                seconds,
                ", ".join("%s %.2f" % (n, ratio(n, 2)) for n in worst),
            )
        )
    print(
        "%-30s %8s %8s %8s %8s %8s  %s"
        % (
            "rule switched off",
            "nnz",
            "rows",
            "cols",
            "changed",
            "seconds",
            "largest losses (nnz without / with)",
        )
    )
    for _, name, means, changed, seconds, worst in sorted(lines, reverse=True):
        print(
            "%-30s %8.4f %8.4f %8.4f %8d %+8.0f  %s"
            % (name, means[2], means[0], means[1], changed, seconds, worst)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
