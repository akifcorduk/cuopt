# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two runs of tools/run_engine.py instance by instance.

Usage: compare_runs.py <run a> <run b> [--ignore KEY[,KEY]]

A run is a directory with a summary.json, or such a file. Two runs are one
when every instance has, in both, the same status, reduced sizes, hash of
the reduced model, rounds, rebuilds, fixed-point flag, rounds on explicit
bounds, dual findings, findings per reduction, column statistics, counts of
records and of objective nonzeros, and the same calls, work, first work,
findings, records and removals of the start step and of every rule family.
Seconds are not compared. A key that only one side reports (statistics that
a later engine added, the options of the run) is not compared either and is
named once. --ignore leaves top-level keys of an instance out.

Prints the number of instances compared and every difference; the exit code
is 1 when there is one.
"""

import argparse
import json
import sys
from pathlib import Path

KEYS = (
    "status",
    "reduced",
    "hash",
    "rounds",
    "rebuilds",
    "fixed_point",
    "explicit",
    "parallel_rows",
    "dual_findings",
    "sub_rules",
    "columns",
    "col_from_rows",
    "objective_nnz",
    "column_counts",
    "left",
    "second",
    "start",
    "rules",
)
TIMES = ("seconds",)


def load(path):
    """Return the instances of a run by name."""
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    with open(path) as f:
        return {row["name"]: row for row in json.load(f)["instances"]}


def differences(a, b, where, out, one_sided):
    """Append the paths at which two values differ; times are left out."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key in TIMES:
                continue
            if key not in a or key not in b:
                one_sided.add("%s.%s" % (where.split(".", 1)[0], key))
                continue
            differences(a[key], b[key], "%s.%s" % (where, key), out, one_sided)
    elif a != b:
        out.append("%s: %s | %s" % (where, a, b))


def main():
    """Compare the two runs and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--ignore", default="")
    args = parser.parse_args()
    ignored = set(filter(None, args.ignore.split(",")))
    a, b = load(args.a), load(args.b)
    found = []
    one_sided = set()
    for name in sorted(set(a) ^ set(b)):
        found.append("%s: in one run only" % name)
    compared = sorted(set(a) & set(b))
    differing = 0
    for name in compared:
        out = []
        for key in KEYS:
            if key in ignored:
                continue
            if (key in a[name]) != (key in b[name]):
                one_sided.add(key)
                continue
            if key in a[name]:
                differences(a[name][key], b[name][key], key, out, one_sided)
        differing += 1 if out else 0
        found.extend("%s: %s" % (name, line) for line in out)
    print(
        "%d instances compared, %d differ, %d differences"
        % (len(compared), differing, len(found))
    )
    if one_sided:
        print(
            "in one run only, not compared: %s" % ", ".join(sorted(one_sided))
        )
    for line in found[:200]:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
