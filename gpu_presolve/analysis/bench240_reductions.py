# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare what each presolve leaves of the 240 benchmark instances.

Usage: bench240_reductions.py [engine tag] [cuOpt report directory]
       (defaults: v4pre, cuopt_report_v3)
Presolve only, no solve, no scaling. Sources, all under results/bench240:
- <engine tag>/release_all/summary.json: the standalone host engine
  (tools/run_engine.py with testbed/benchmark240.json);
- highs_stock/summary.json: HiGHS's full MIP presolve
  (baseline/run_baseline.py --variant stock);
- <report directory>/report.txt: the lines of the presolve-only report of
  cuOpt (gprs_presolve_report.DISABLED_reductions_report): mode 0 is cuOpt's
  PaPILO call with cuOpt's own reductions, mode 2 the new presolve alone inside
  cuOpt, mode 3 the new presolve followed by cuOpt's own reductions.

For every presolver the remaining nonzeros are compared; an instance that a
presolver solves or proves infeasible counts as one remaining nonzero in a
ratio. Prints per size class and in total: the geometric mean of
nnz(after) / nnz(before), the geometric mean of nnz(other) / nnz(engine) for
each other presolver (below 1: the other leaves less), and how often the
engine leaves less, the same, or more. Lists the instances with the largest
distance in both directions and writes <engine tag>_reductions.csv.
"""

import csv
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BENCH = PROJECT / "results" / "bench240"
CUOPT_STATUS = {
    0: "infeasible",
    1: "unbounded",
    2: "unbounded_or_infeasible",
    3: "solved",
    4: "reduced",
    5: "unchanged",
}
CUOPT_MODES = {0: "papilo", 2: "cuopt_new", 3: "cuopt_new_own"}


def engine(tag):
    """Return rows, cols, nnz, seconds and status of the engine by name."""
    path = BENCH / tag / "release_all" / "summary.json"
    out = {}
    with open(path) as f:
        for r in json.load(f)["instances"]:
            if "reduced" not in r:
                continue
            reduced = r["reduced"]
            out[r["name"]] = {
                "rows": reduced["rows"],
                "cols": reduced["cols"],
                "nnz": reduced["nnz"],
                "seconds": r["seconds"],
                "status": r["status"],
            }
    return out


def highs():
    """Return what HiGHS's full presolve leaves, by name."""
    path = BENCH / "highs_stock" / "summary.json"
    out = {}
    if not path.is_file():
        return out
    with open(path) as f:
        for r in json.load(f)["instances"]:
            status = str(r.get("status"))
            if r.get("nnz_after") is None:
                # No presolved model: solved or infeasible in presolve, or a time limit.
                if r.get("reached_time_limit"):
                    continue
                size = {"rows": 0, "cols": 0, "nnz": 0}
            else:
                size = {
                    "rows": r["rows_after"],
                    "cols": r["cols_after"],
                    "nnz": r["nnz_after"],
                }
            size.update(seconds=r.get("presolve_seconds"), status=status)
            out[r["name"]] = size
    return out


def cuopt(directory):
    """Return the report lines of cuOpt by presolver label and name."""
    path = BENCH / directory / "report.txt"
    out = {label: {} for label in CUOPT_MODES.values()}
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            if "GPRS_REDUCTIONS" not in line:
                continue
            fields = dict(
                word.split("=", 1) for word in line.split() if "=" in word
            )
            name = Path(fields["path"]).name
            name = name[: -len(".mps")] if name.endswith(".mps") else name
            label = CUOPT_MODES.get(int(fields["mode"]))
            if label is None:
                continue
            out[label][name] = {
                "rows": int(fields["rows"]),
                "cols": int(fields["cols"]),
                "nnz": int(fields["nnz"]),
                "seconds": float(fields["seconds"]),
                "status": CUOPT_STATUS.get(int(fields["status"]), "?"),
            }
    return out


def geomean(values):
    """Return the geometric mean of positive values, or None when empty."""
    values = list(values)
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def left(size):
    """Return the remaining nonzeros for a ratio, at least one."""
    return max(size["nnz"], 1)


def show(value, form="%.4f"):
    """Return a number for a table cell, a dash for a missing one."""
    return "-" if value is None else form % value


def main():
    """Print the comparison and write the table."""
    tag = sys.argv[1] if len(sys.argv) > 1 else "v4pre"
    report = sys.argv[2] if len(sys.argv) > 2 else "cuopt_report_v3"
    with open(PROJECT / "testbed" / "benchmark240.json") as f:
        testbed = json.load(f)
    entries = {e["name"]: e for e in testbed["main"] + testbed["status"]}
    sides = {"engine": engine(tag), "highs": highs()}
    sides.update(cuopt(report))
    others = [s for s in sides if s != "engine"]
    for label, data in sides.items():
        print("%-14s %3d instances" % (label, len(data)))

    names = sorted(sides["engine"], key=lambda n: entries[n]["nnz"])
    with open(BENCH / (tag + "_reductions.csv"), "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        header = ["instance", "class", "rows", "cols", "nnz"]
        for label in sides:
            header += [
                label + "_" + key
                for key in ("status", "rows", "cols", "nnz", "seconds")
            ]
        writer.writerow(header)
        for n in names:
            e = entries[n]
            row = [n, e["band"], e["rows"], e["cols"], e["nnz"]]
            for label in sides:
                size = sides[label].get(n)
                row += (
                    [""] * 5
                    if size is None
                    else [
                        size["status"],
                        size["rows"],
                        size["cols"],
                        size["nnz"],
                        size["seconds"],
                    ]
                )
            writer.writerow(row)

    classes = ["S", "A", "B", "C", "all"]
    print(
        "\nGeometric mean of nnz(after) / nnz(before); S below 5e4 nonzeros, "
        "A to 5e5, B to 1e7, C above"
    )
    print(
        "%-6s %4s  %s" % ("class", "n", "  ".join("%-14s" % s for s in sides))
    )
    for c in classes:
        chosen = [n for n in names if c == "all" or entries[n]["band"] == c]
        cells = []
        for label in sides:
            kept = [
                left(sides[label][n]) / entries[n]["nnz"]
                for n in chosen
                if n in sides[label]
            ]
            cells.append("%-14s" % show(geomean(kept)))
        print("%-6s %4d  %s" % (c, len(chosen), "  ".join(cells)))

    for other in others:
        data = sides[other]
        if not data:
            continue
        print(
            "\n%s against the engine: geometric mean of nnz(%s) / nnz(engine); "
            "below 1 the engine leaves more" % (other, other)
        )
        print(
            "%-6s %4s %8s | %s"
            % ("class", "n", "geomean", "engine leaves less / equal / more")
        )
        for c in classes:
            chosen = [
                n
                for n in names
                if n in data and (c == "all" or entries[n]["band"] == c)
            ]
            if not chosen:
                continue
            ratio = [left(data[n]) / left(sides["engine"][n]) for n in chosen]
            less = sum(1 for r in ratio if r > 1.0)
            equal = sum(1 for r in ratio if r == 1.0)
            print(
                "%-6s %4d %8s | %d / %d / %d"
                % (
                    c,
                    len(chosen),
                    show(geomean(ratio)),
                    less,
                    equal,
                    len(ratio) - less - equal,
                )
            )
        ranked = sorted(
            (n for n in names if n in data),
            key=lambda n: left(data[n]) / left(sides["engine"][n]),
        )
        print("  the engine is furthest behind on:")
        for n in ranked[:12]:
            print(
                "    %-28s %9d nnz: engine %9d, %s %9d (%s)"
                % (
                    n,
                    entries[n]["nnz"],
                    sides["engine"][n]["nnz"],
                    other,
                    data[n]["nnz"],
                    data[n]["status"],
                )
            )
        print("  the engine is furthest ahead on:")
        for n in ranked[::-1][:6]:
            print(
                "    %-28s %9d nnz: engine %9d, %s %9d"
                % (
                    n,
                    entries[n]["nnz"],
                    sides["engine"][n]["nnz"],
                    other,
                    data[n]["nnz"],
                )
            )

    print("\nSummed presolve seconds on the instances every side has:")
    common = [
        n for n in names if all(n in sides[s] for s in sides if sides[s])
    ]
    for label, data in sides.items():
        if data:
            total = sum(data[n]["seconds"] or 0.0 for n in common)
            print(
                "  %-14s %8.1f s over %d instances"
                % (label, total, len(common))
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
