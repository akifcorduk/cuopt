# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the engine once per value of an option and print what each value changes.

Usage: switch_table.py --tool <gprs_presolve> --dumps <dir> --out <dir>
                       [--testbed FILE] [--set all] [--jobs N] [--skip-done]
                       [--with ARG]... [--reference DIR] [--store DIR]
                       <label>=<tool option>[,<tool option>]...

One run of tools/run_engine.py (one repetition) with the options of --with
alone, the base, into <out>/<base label>, and one per value with its options
added, into <out>/<label>. The base label is "default", or "default_" and the
--with options when there are any. --skip-done keeps a run whose summary is
there. --store writes the reduced summary.json of every run
(tools/reduce_summary.py) to <DIR>/<label>/.

Printed per value, against the base:
- the geometric mean of base size / size with the value in nonzeros, rows
  and columns over the instances without ex9, ex10 and enlight_hard (above 1
  the value leaves less; a size of zero counts as one), and over all;
- the instances whose reduced model changes (another hash), and of them how
  many are smaller, larger or of the same size in each measure;
- the five largest changes by name, and every instance that gets larger in
  a measure;
- the fixed-point flags: how many are set, which are lost or gained; the
  instances with a failure of the run;
- with --reference (a HiGHS reference of analysis/bench240_stage1c.py, for
  example results/bench240/stage1c_refs/m1_a): the geometric mean of
  reference / engine of that script, for the base and for the value.

Example, the growth ratio on the 240 benchmark dumps:
  switch_table.py --tool build/release/gprs_presolve --dumps <dumps> --out <scratch>
    --testbed testbed/benchmark240.json --jobs 2
    --reference results/bench240/stage1c_refs/m1_a
    growth_ratio_continuous=--growth-ratio,continuous
    growth_ratio_integer_exempt=--growth-ratio,integer-exempt
"""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import reduce_summary

PROJECT_DIR = Path(__file__).resolve().parent.parent
LEFT_OUT = ("ex9", "ex10", "enlight_hard")
MEASURES = ("nnz", "rows", "cols")


def parse_args():
    """Return the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tool", required=True)
    parser.add_argument("--dumps", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--testbed", default=str(PROJECT_DIR / "testbed" / "benchmark240.json")
    )
    parser.add_argument("--set", default="all")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--with", dest="base", action="append", default=[])
    parser.add_argument("--reference")
    parser.add_argument("--store")
    parser.add_argument("values", nargs="*")
    return parser.parse_args()


def run(args, label, options):
    """Run the engine with the options into <out>/<label>; return the rows."""
    out = Path(args.out) / label
    summary = out / "summary.json"
    if not (args.skip_done and summary.is_file()):
        command = [
            sys.executable,
            str(PROJECT_DIR / "tools" / "run_engine.py"),
        ]
        command += ["--tool", args.tool, "--dumps", args.dumps]
        command += ["--testbed", args.testbed, "--set", args.set]
        command += ["--reps", "1", "--jobs", str(args.jobs), "--out", str(out)]
        command += ["--tool-arg=" + option for option in options]
        subprocess.run(command, stdout=subprocess.DEVNULL, check=False)
    if not summary.is_file():
        sys.exit("%s: the run left no summary" % label)
    if args.store:
        reduce_summary.reduce_summary(
            summary, Path(args.store) / label / "summary.json"
        )
    with open(summary) as f:
        data = json.load(f)
    return {row["name"]: row for row in data["instances"]}, data["failures"]


def sizes(row):
    """Return nonzeros, rows and columns of the reduced model, at least 1."""
    return [max(row["reduced"][key], 1) for key in MEASURES]


def geomean(values):
    """Return the geometric mean of positive values."""
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def reference_means(args, label):
    """Return the three means of reference / engine of bench240_stage1c."""
    script = PROJECT_DIR / "analysis" / "bench240_stage1c.py"
    command = [sys.executable, str(script)]
    command += ["--engine", str(Path(args.out) / label)]
    command += ["--ref", args.reference]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    for line in done.stdout.splitlines():
        if line.startswith("without ") or line.startswith("all "):
            found = re.findall(r"\d+\.\d{4}", line)
            if len(found) == 3:
                return " / ".join(found)
    return "not computed"


def compare(args, label, base, rows):
    """Print what the value changes against the base."""
    names = [
        n
        for n in base
        if n in rows and "reduced" in base[n] and "reduced" in rows[n]
    ]
    ratios = {
        n: [b / v for b, v in zip(sizes(base[n]), sizes(rows[n]))]
        for n in names
    }
    main = [n for n in names if n not in LEFT_OUT]
    changed = [n for n in names if base[n]["hash"] != rows[n]["hash"]]
    print(
        "  base / value over %d instances without %s: %s; over all %d: %s"
        % (
            len(main),
            ", ".join(LEFT_OUT),
            " / ".join(
                "%.4f" % geomean(ratios[n][k] for n in main) for k in range(3)
            ),
            len(names),
            " / ".join(
                "%.4f" % geomean(ratios[n][k] for n in names) for k in range(3)
            ),
        )
    )
    print(
        "  reduced model changes on %d of %d instances"
        % (len(changed), len(names))
    )
    for k, measure in enumerate(("nonzeros", "rows", "columns")):
        print(
            "    %-8s smaller on %d, larger on %d, equal on %d of them"
            % (
                measure,
                sum(1 for n in changed if ratios[n][k] > 1),
                sum(1 for n in changed if ratios[n][k] < 1),
                sum(1 for n in changed if ratios[n][k] == 1),
            )
        )

    def distance(name):
        return max(abs(math.log(r)) for r in ratios[name])

    largest = sorted(changed, key=lambda n: (-distance(n), n))[:5]
    for n in largest:
        if distance(n) == 0:
            continue
        print(
            "    %-28s %s -> %s (base / value %s)"
            % (
                n,
                " / ".join(str(base[n]["reduced"][m]) for m in MEASURES),
                " / ".join(str(rows[n]["reduced"][m]) for m in MEASURES),
                " / ".join("%.4f" % r for r in ratios[n]),
            )
        )
    larger = [n for n in names if min(ratios[n]) < 1]
    print(
        "  larger in a measure: %d%s"
        % (len(larger), ": " + ", ".join(larger) if larger else "")
    )
    set_base = {n for n in names if base[n].get("fixed_point")}
    set_rows = {n for n in names if rows[n].get("fixed_point")}
    lost, gained = sorted(set_base - set_rows), sorted(set_rows - set_base)
    print(
        "  fixed-point flags %d (base %d)%s%s"
        % (
            len(set_rows),
            len(set_base),
            "; lost: " + ", ".join(lost) if lost else "",
            "; gained: " + ", ".join(gained) if gained else "",
        )
    )
    rounds = sum(rows[n]["rounds"] for n in names)
    print(
        "  rounds %d (base %d), most %d"
        % (
            rounds,
            sum(base[n]["rounds"] for n in names),
            max(rows[n]["rounds"] for n in names),
        )
    )
    if args.reference:
        print(
            "  %s / engine: %s (base %s)"
            % (
                Path(args.reference).name,
                reference_means(args, label),
                args.base_reference,
            )
        )


def main():
    """Run the base and the values and print the table."""
    args = parse_args()
    base_label = "default"
    if args.base:
        base_label += "_" + "_".join(o.strip("-") for o in args.base)
    base, failures = run(args, base_label, args.base)
    print(
        "base %s: %d instances, %d with a failure"
        % (base_label, len(base), failures)
    )
    if args.reference:
        args.base_reference = reference_means(args, base_label)
    for value in args.values:
        label, _, text = value.partition("=")
        options = args.base + text.split(",")
        rows, failures = run(args, label, options)
        print(
            "%s (%s): %d with a failure" % (label, " ".join(options), failures)
        )
        compare(args, label, base, rows)


if __name__ == "__main__":
    main()
