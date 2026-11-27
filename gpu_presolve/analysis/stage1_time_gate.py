# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply the time rule of Stage 1 to runs of both sides on one machine.

Usage: stage1_time_gate.py <name>
Reads results/stage1/timing/<name>, written by
baseline/stage1_timing_runs.sh: passes ref_dev_<k>, engine_dev_<k>,
ref_large_<k>, engine_large_<k> (k = 1, 2, 3), the reference and the engine
in turn; or by baseline/bench240_timing_runs.sh on the 240 benchmark
instances: ref_all_<k>, engine_all_<k>. Per instance the time of a side is
the minimum over its passes.

Time rule: t <= 1.2 * t(reference) + 0.02 s. It is a gate in bands A and B;
the other bands are reported. An instance with fewer than three passes on a
side is flagged, and so is a pass whose reduced model differs from the
other passes. Exit code 1 when an instance of bands A or B is above the
limit or has no time on a side.
"""

import json
import statistics
import sys
from pathlib import Path

TIMING = (
    Path(__file__).resolve().parent.parent / "results" / "stage1" / "timing"
)
PASSES = (1, 2, 3)


def instances(path):
    """Return the summary of a pass and its instances by name."""
    if not path.is_file():
        return None, {}
    with open(path) as f:
        summary = json.load(f)
    return summary, {r["name"]: r for r in summary["instances"]}


def collect(directory, side, seconds_key):
    """Return per instance the times and records of a side over all passes."""
    times = {}
    records = {}
    summaries = []
    for where in ("dev", "large", "all"):
        for k in PASSES:
            path = directory / ("%s_%s_%d" % (side, where, k)) / "summary.json"
            summary, rows = instances(path)
            if summary is None:
                continue
            summaries.append((path.parent.name, summary))
            for name, row in rows.items():
                seconds = row.get(seconds_key)
                if seconds is None:
                    continue
                times.setdefault(name, []).append(seconds)
                records.setdefault(name, []).append(row)
    return times, records, summaries


def main():
    """Print the time rule per instance and the sums."""
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    directory = TIMING / sys.argv[1]
    ref, ref_rows, ref_summaries = collect(
        directory, "ref", "presolve_core_seconds"
    )
    eng, eng_rows, eng_summaries = collect(directory, "engine", "seconds")
    if not ref or not eng:
        sys.exit("no passes of both sides in %s" % directory)

    names = sorted(
        set(ref) | set(eng),
        key=lambda n: (ref_rows.get(n) or eng_rows[n])[0]["nnz"],
    )
    print(
        "%-3s %-28s %10s | %9s %9s %8s %7s | %s"
        % ("", "instance", "nnz", "ref s", "engine s", "limit", "ratio", "")
    )
    failed = []
    above = []
    by_band = {}
    total_ref = total_eng = 0.0
    for n in names:
        row = (ref_rows.get(n) or eng_rows[n])[0]
        band = row["band"]
        gated = band[0] in "AB"
        flags = []
        if n not in ref or n not in eng:
            flags.append("no time on a side")
            if gated:
                failed.append(n)
            print("%-3s %-28s %10d | %s" % (band, n, row["nnz"], flags[0]))
            continue
        t_ref = min(ref[n])
        t_eng = min(eng[n])
        limit = 1.2 * t_ref + 0.02
        if len(ref[n]) < len(PASSES) or len(eng[n]) < len(PASSES):
            flags.append(
                "passes: ref %d, engine %d" % (len(ref[n]), len(eng[n]))
            )
        if len({r.get("hash") for r in eng_rows[n]}) > 1:
            flags.append("engine results differ between passes")
        if t_eng > limit:
            flags.append("ABOVE THE LIMIT")
            above.append(n)
            if gated:
                failed.append(n)
        total_ref += t_ref
        total_eng += t_eng
        by_band.setdefault(band, []).append((t_ref, t_eng))
        print(
            "%-3s %-28s %10d | %9.3f %9.3f %8.3f %7.2f | %s"
            % (
                band,
                n,
                row["nnz"],
                t_ref,
                t_eng,
                limit,
                t_ref / t_eng if t_eng > 0 else float("inf"),
                "; ".join(flags),
            )
        )

    print()
    print(
        "%-4s %3s %10s %10s %8s" % ("band", "n", "ref med", "eng med", "ratio")
    )
    for band in sorted(by_band):
        pairs = by_band[band]
        med_ref = statistics.median(p[0] for p in pairs)
        med_eng = statistics.median(p[1] for p in pairs)
        print(
            "%-4s %3d %10.3f %10.3f %8.2f"
            % (band, len(pairs), med_ref, med_eng, med_ref / med_eng)
        )
    print(
        "sum of the minima: reference %.1f s, engine %.1f s, ratio %.2f"
        % (total_ref, total_eng, total_ref / total_eng)
    )
    print(
        "time rule: %s. %d of %d instances above 1.2 * t + 0.02 s; in bands "
        "A and B (the gate): %d"
        % (
            "FAIL" if failed else "pass",
            len(above),
            len(names),
            len(failed),
        )
    )
    for label, summary in eng_summaries:
        identity = summary.get("identity") or {}
        print(
            "%s: host %s, load %s, tool checksum %s, source checksum %s"
            % (
                label,
                summary.get("host"),
                "%.1f" % summary["load_average"][0]
                if summary.get("load_average")
                else "?",
                str(identity.get("tool_checksum"))[:12],
                str(identity.get("source_checksum"))[:12],
            )
        )
    for label, summary in ref_summaries:
        print("%s: tool %s" % (label, summary.get("tool")))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
