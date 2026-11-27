# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare a host engine run with the Stage 1 reference and apply the gates.

Usage: stage1_compare.py [tag]   (default v1)
Reads results/stage1/host/<tag>/release_dev (and release_large when present),
results/stage1/targets_stage1.csv and results/stage1/refs/stage1c_large.
Reductions gate: geometric mean of nnz(reference) / nnz(engine) at least 0.99,
every instance below 0.95 listed. Time gate: t <= 1.2 * t(reference) + 0.02 s
per instance, and the median per band not above the reference.
Writes results/stage1/host/<tag>/compare.csv.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
STAGE1 = RESULTS / "stage1"
SHIFT = 0.1


def load(path):
    """Return the instances of a runner summary by name."""
    with open(path) as f:
        return {r["name"]: r for r in json.load(f)["instances"]}


def reference():
    """Return the Stage 1 reference by instance name: band, sizes, seconds."""
    ref = {}
    with open(STAGE1 / "targets_stage1.csv") as f:
        for r in csv.DictReader(f):
            ref[r["instance"]] = {
                "band": r["band"],
                "nnz": int(r["nnz"]),
                "ref": [
                    int(r["stage1_" + k]) for k in ("rows", "cols", "nnz")
                ],
                "seconds": float(r["stage1_seconds"]),
            }
    large = STAGE1 / "refs" / "stage1c_large" / "summary.json"
    if large.is_file():
        for name, r in load(large).items():
            if "diag" not in r or not r["diag"].get("total"):
                continue
            total = r["diag"]["total"][0]
            ref[name] = {
                "band": r["band"],
                "nnz": r["nnz"],
                "ref": [
                    int(total[k].split(":")[1])
                    for k in ("rows", "cols", "nnz")
                ],
                "seconds": r["presolve_core_seconds"],
            }
    return ref


def shifted_mean(values):
    """Return the shifted geometric mean of the values."""
    return (
        math.exp(sum(math.log(v + SHIFT) for v in values) / len(values))
        - SHIFT
    )


def main():
    """Print the comparison and write compare.csv."""
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    host = STAGE1 / "host" / tag
    runs = load(host / "release_dev" / "summary.json")
    if (host / "release_large" / "summary.json").is_file():
        runs.update(load(host / "release_large" / "summary.json"))
    ref = reference()
    names = sorted(
        (n for n in runs if n in ref and "reduced" in runs[n]),
        key=lambda n: ref[n]["nnz"],
    )
    missing = sorted(n for n in runs if n not in names)

    rows = []
    print(
        "%-3s %-24s %10s | %10s %10s %6s | %8s %8s %7s  %s"
        % (
            "",
            "instance",
            "nnz",
            "reference",
            "engine",
            "ratio",
            "ref s",
            "engine s",
            "limit",
            "status",
        )
    )
    for n in names:
        r = runs[n]
        ours = [r["reduced"][k] for k in ("rows", "cols", "nnz")]
        theirs = ref[n]["ref"]
        ratio = max(theirs[2], 1) / max(ours[2], 1)
        limit = 1.2 * ref[n]["seconds"] + 0.02
        flags = []
        if ratio < 0.95:
            flags.append("fewer reductions")
        if r["seconds"] > limit:
            flags.append("slow")
        if not r["same_hash"]:
            flags.append("hash differs between runs")
        if r["status"] not in ("reduced", "unchanged", "reduced_to_empty"):
            flags.append(r["status"])
        rows.append(
            [ref[n]["band"], n, ref[n]["nnz"]]
            + theirs
            + ours
            + [
                "%.4f" % ratio,
                "%.6f" % ref[n]["seconds"],
                "%.6f" % r["seconds"],
                "%.3f" % limit,
                r["status"],
                r["rounds"],
                "; ".join(flags),
            ]
        )
        print(
            "%-3s %-24s %10d | %10d %10d %6.3f | %8.3f %8.3f %7.3f  %s"
            % (
                ref[n]["band"],
                n[:24],
                ref[n]["nnz"],
                theirs[2],
                ours[2],
                ratio,
                ref[n]["seconds"],
                r["seconds"],
                limit,
                "; ".join(flags) or r["status"],
            )
        )
    for n in missing:
        print("no result: %s %s" % (n, runs[n].get("errors")))

    ratios = [float(r[9]) for r in rows]
    print()
    print(
        "reductions gate: geometric mean of nnz(reference) / nnz(engine) = "
        "%.4f over %d instances (needs 0.99); below 0.95: %d; above 1.05: %d"
        % (
            math.exp(sum(math.log(v) for v in ratios) / len(ratios)),
            len(ratios),
            sum(1 for v in ratios if v < 0.95),
            sum(1 for v in ratios if v > 1.05),
        )
    )
    slow = [r[1] for r in rows if float(r[11]) > float(r[12])]
    print(
        "time gate: %d of %d instances above 1.2 * t + 0.02 s%s"
        % (len(slow), len(rows), ": " + ", ".join(slow) if slow else "")
    )
    print(
        "%-3s %3s %9s %9s %9s %9s"
        % ("", "n", "ref med", "eng med", "ref sgm", "eng sgm")
    )
    bands = {}
    for r in rows:
        bands.setdefault(r[0], []).append(r)
    for band in sorted(bands):
        theirs = [float(r[10]) for r in bands[band]]
        ours = [float(r[11]) for r in bands[band]]
        print(
            "%-3s %3d %9.3f %9.3f %9.3f %9.3f%s"
            % (
                band,
                len(ours),
                statistics.median(theirs),
                statistics.median(ours),
                shifted_mean(theirs),
                shifted_mean(ours),
                "  median above the reference"
                if statistics.median(ours) > statistics.median(theirs)
                else "",
            )
        )
    print(
        "sum of seconds: reference %.1f, engine %.1f"
        % (sum(float(r[10]) for r in rows), sum(float(r[11]) for r in rows))
    )

    header = [
        "band",
        "instance",
        "nnz",
        "ref_rows",
        "ref_cols",
        "ref_nnz",
        "engine_rows",
        "engine_cols",
        "engine_nnz",
        "nnz_ratio",
        "ref_seconds",
        "engine_seconds",
        "limit_seconds",
        "status",
        "rounds",
        "flags",
    ]
    with open(host / "compare.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("written: %s" % (host / "compare.csv"))


if __name__ == "__main__":
    main()
