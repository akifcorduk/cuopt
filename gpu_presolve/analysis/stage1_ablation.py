# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Show what each rule family of the host engine contributes.

Usage: stage1_ablation.py [tag]   (default v1)
Reads results/stage1/host/<tag>/release_dev, release_large and the runs with
one family switched off (off_<rule>/dev, off_<rule>/large). For each family:
the geometric mean of nnz(without the family) / nnz(with all families), the
number of instances that change, the summed time, and the instances that lose
most. The times of the runs with a family off are from one repetition with six
runs side by side, so they are indicative only.
"""

import json
import math
import sys
from pathlib import Path

HOST = Path(__file__).resolve().parent.parent / "results" / "stage1" / "host"
RULES = [
    "row_rules",
    "propagation",
    "tightening",
    "doubleton",
    "parallel_rows",
    "dual_fixing",
]


def load(*paths):
    """Return the instances of several runner summaries by name."""
    out = {}
    for path in paths:
        if path.is_file():
            with open(path) as f:
                for r in json.load(f)["instances"]:
                    if "reduced" in r:
                        out[r["name"]] = r
    return out


def main():
    """Print one line per rule family."""
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    host = HOST / tag
    full = load(
        host / "release_dev" / "summary.json",
        host / "release_large" / "summary.json",
    )
    print(
        "%-14s %4s %9s %8s %9s %9s  %s"
        % (
            "family off",
            "n",
            "geomean",
            "changed",
            "sum s",
            "all on s",
            "largest losses (nnz without / nnz with)",
        )
    )
    for rule in RULES:
        off = load(
            host / ("off_" + rule) / "dev" / "summary.json",
            host / ("off_" + rule) / "large" / "summary.json",
        )
        names = [n for n in full if n in off]
        if not names:
            print("%-14s no result" % rule)
            continue
        ratio = {
            n: max(off[n]["reduced"]["nnz"], 1)
            / max(full[n]["reduced"]["nnz"], 1)
            for n in names
        }
        worst = sorted(names, key=lambda n: -ratio[n])[:4]
        print(
            "%-14s %4d %9.4f %8d %9.1f %9.1f  %s"
            % (
                rule,
                len(names),
                math.exp(
                    sum(math.log(v) for v in ratio.values()) / len(names)
                ),
                sum(
                    1 for n in names if off[n]["reduced"] != full[n]["reduced"]
                ),
                sum(off[n]["seconds"] for n in names),
                sum(full[n]["seconds"] for n in names),
                ", ".join("%s %.3f" % (n[:18], ratio[n]) for n in worst),
            )
        )


if __name__ == "__main__":
    main()
