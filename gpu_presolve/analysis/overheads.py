# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track the time of the host engine that is not spent in reduction rules.

Usage: overheads.py <tag> [timing name]
       overheads.py --bench <tag> [cuOpt report directory]

The engine reports its seconds as rule families plus the parts outside them:
state (the state built from the model, and the postsolve stack), start,
storage (compaction and rebuilds between phases), finish (the reduced model
read from the state) and hash (the hash of the reduced model). These parts
are the overheads. They are tracked with every run and optimised only when
they stand between the engine and HiGHS on time.

First form: results/stage1/host/<tag>/release_dev and release_large, against
the HiGHS Stage 1 reference: the minimum over the passes of
results/stage1/timing/<timing name> when that is given (both sides timed in
turn on one machine), else results/stage1/refs/stage1c and stage1c_large,
which were measured on another machine and are printed with that warning.
Second form: results/bench240/<tag>/release_all against HiGHS with the Stage 1
rule set on the 240 benchmark instances (results/bench240/highs_stage1c), and
what the same presolve costs inside cuOpt when a report directory is given.

Printed: the summed seconds split into rules and overheads, per size class;
the time allowance of the plan, 1.2 * t(HiGHS) + 0.02 s per instance, and how
much of it the engine and its overheads alone use; the instances where the
overheads alone use more than half of the allowance. A line per tag is kept
in results/stage1/overheads.csv, so that the overheads can be followed from
run to run.
"""

import csv
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
STAGE1 = RESULTS / "stage1"
BENCH = RESULTS / "bench240"
PARTS = ("state", "start", "storage", "finish", "hash")
TRACK = STAGE1 / "overheads.csv"


def instances(path):
    """Return the instances of a runner summary by name, or an empty dict."""
    if not path.is_file():
        return {}
    with open(path) as f:
        return {r["name"]: r for r in json.load(f)["instances"]}


def engine_rows(directories):
    """Return per instance its class, nonzeros and seconds by part."""
    out = {}
    for directory in directories:
        for name, r in instances(directory / "summary.json").items():
            if "seconds" not in r:
                continue
            outside = r.get("seconds_outside_rules") or {}
            rules = {
                family: stats.get("seconds", 0.0)
                for family, stats in (r.get("rules") or {}).items()
            }
            out[name] = {
                "band": r.get("band") or "?",
                "nnz": r["nnz"],
                "total": r["seconds"],
                "rules": rules,
                "parts": {p: outside.get(p, 0.0) for p in PARTS},
            }
    return out


def timed_reference(name):
    """Return the minimum reference seconds over the passes of a timing run."""
    out = {}
    for where in ("dev", "large"):
        for k in (1, 2, 3):
            path = (
                STAGE1
                / "timing"
                / name
                / ("ref_%s_%d" % (where, k))
                / "summary.json"
            )
            for n, r in instances(path).items():
                seconds = r.get("presolve_core_seconds")
                if seconds is not None:
                    out[n] = min(seconds, out.get(n, seconds))
    return out


def stored_reference(paths, key):
    """Return the reference seconds of stored summaries."""
    out = {}
    for path in paths:
        for n, r in instances(path).items():
            if r.get(key) is not None and not r.get("reached_time_limit"):
                out[n] = r[key]
    return out


def report(rows, reference, label, source):
    """Print the split and the allowance; return the line for the tracking file."""
    names = sorted(rows, key=lambda n: rows[n]["nnz"])
    total = sum(rows[n]["total"] for n in names)
    families = sorted({f for n in names for f in rows[n]["rules"]})
    rule_seconds = {
        f: sum(rows[n]["rules"].get(f, 0.0) for n in names) for f in families
    }
    part_seconds = {p: sum(rows[n]["parts"][p] for n in names) for p in PARTS}
    rules = sum(rule_seconds.values())
    overhead = sum(part_seconds.values())
    print("%s: %d instances, %.1f s" % (label, len(names), total))
    print(
        "  rules     %7.2f s %5.1f%%   (%s)"
        % (
            rules,
            100 * rules / total,
            ", ".join(
                "%s %.1f%%" % (f, 100 * rule_seconds[f] / total)
                for f in sorted(families, key=lambda f: -rule_seconds[f])
            ),
        )
    )
    print(
        "  overheads %7.2f s %5.1f%%   (%s)"
        % (
            overhead,
            100 * overhead / total,
            ", ".join(
                "%s %.1f%%" % (p, 100 * part_seconds[p] / total)
                for p in sorted(PARTS, key=lambda p: -part_seconds[p])
            ),
        )
    )
    print(
        "\n  %-5s %4s %9s %9s %9s %8s"
        % ("class", "n", "total s", "rules s", "overhd s", "overhd %")
    )
    for band in sorted({rows[n]["band"] for n in names}):
        chosen = [n for n in names if rows[n]["band"] == band]
        t = sum(rows[n]["total"] for n in chosen)
        o = sum(sum(rows[n]["parts"].values()) for n in chosen)
        r = sum(sum(rows[n]["rules"].values()) for n in chosen)
        print(
            "  %-5s %4d %9.3f %9.3f %9.3f %8.1f"
            % (band, len(chosen), t, r, o, 100 * o / t if t else 0.0)
        )

    timed = [n for n in names if n in reference]
    line = {
        "run": label,
        "instances": len(names),
        "total_seconds": round(total, 3),
        "rules_seconds": round(rules, 3),
        "overhead_seconds": round(overhead, 3),
    }
    line.update({p + "_seconds": round(part_seconds[p], 3) for p in PARTS})
    if not timed:
        print("\n  no HiGHS reference seconds: the allowance is not computed")
        return line
    allowance = {n: 1.2 * reference[n] + 0.02 for n in timed}
    used = sum(rows[n]["total"] for n in timed)
    used_overhead = sum(sum(rows[n]["parts"].values()) for n in timed)
    budget = sum(allowance.values())
    print(
        "\n  against HiGHS (%s), %d instances: HiGHS %.1f s, allowance 1.2 * t + "
        "0.02 s summed %.1f s"
        % (source, len(timed), sum(reference[n] for n in timed), budget)
    )
    print(
        "  the engine uses %.1f%% of the allowance, its overheads alone %.1f%%"
        % (100 * used / budget, 100 * used_overhead / budget)
    )
    above = [n for n in timed if rows[n]["total"] > allowance[n]]
    heavy = sorted(
        (
            n
            for n in timed
            if sum(rows[n]["parts"].values()) > 0.5 * allowance[n]
        ),
        key=lambda n: -sum(rows[n]["parts"].values()) / allowance[n],
    )
    print(
        "  instances above the allowance: %d%s"
        % (len(above), (" (" + ", ".join(above[:8]) + ")") if above else "")
    )
    print(
        "  instances whose overheads alone use more than half of it: %d"
        % len(heavy)
    )
    for n in heavy[:10]:
        parts = rows[n]["parts"]
        largest = max(PARTS, key=lambda p: parts[p])
        print(
            "    %-26s %10d nnz: overheads %.3f s of an allowance of %.3f s "
            "(%.0f%%), mostly %s"
            % (
                n,
                rows[n]["nnz"],
                sum(parts.values()),
                allowance[n],
                100 * sum(parts.values()) / allowance[n],
                largest,
            )
        )
    line.update(
        {
            "reference": source,
            "reference_seconds": round(sum(reference[n] for n in timed), 3),
            "engine_share_of_allowance": round(used / budget, 4),
            "overhead_share_of_allowance": round(used_overhead / budget, 4),
            "instances_above_allowance": len(above),
            "instances_overhead_above_half": len(heavy),
        }
    )
    return line


def track(line):
    """Keep one line per run in the tracking file."""
    lines = []
    if TRACK.is_file():
        with open(TRACK) as f:
            lines = [r for r in csv.DictReader(f) if r["run"] != line["run"]]
    lines.append({k: str(v) for k, v in line.items()})
    keys = []
    for r in lines:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(TRACK, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)
    print("\n  kept in %s" % TRACK)


def cuopt_cost(directory, rows):
    """Print what the engine's work costs inside cuOpt."""
    path = BENCH / directory / "report.txt"
    if not path.is_file():
        return
    inside = {}
    with open(path) as f:
        for text in f:
            fields = dict(w.split("=", 1) for w in text.split() if "=" in w)
            if fields.get("mode") == "2":
                name = Path(fields["path"]).name
                name = name[: -len(".mps")] if name.endswith(".mps") else name
                inside[name] = float(fields["seconds"])
    common = [n for n in rows if n in inside]
    if not common:
        return
    alone = sum(rows[n]["total"] for n in common)
    in_cuopt = sum(inside[n] for n in common)
    print(
        "\n  inside cuOpt (%s, %d instances): %.1f s for the presolve call against "
        "%.1f s for the engine alone; the adapter and its conversions cost %.1f s "
        "(%.1f times the engine). The two were not timed in the same run."
        % (
            directory,
            len(common),
            in_cuopt,
            alone,
            in_cuopt - alone,
            in_cuopt / alone,
        )
    )


def main():
    """Print the overhead report of one run."""
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--bench":
        tag = args[1]
        rows = engine_rows([BENCH / tag / "release_all"])
        reference = stored_reference(
            [BENCH / "highs_stage1c" / "summary.json"], "presolve_seconds"
        )
        line = report(
            rows,
            reference,
            "bench240 " + tag,
            "HiGHS with the Stage 1 rule set, 240 instances, a busy node",
        )
        if len(args) > 2:
            cuopt_cost(args[2], rows)
    else:
        tag = args[0]
        host = STAGE1 / "host" / tag
        rows = engine_rows([host / "release_dev", host / "release_large"])
        if len(args) > 1:
            reference = timed_reference(args[1])
            source = (
                "timing run "
                + args[1]
                + ", same machine, minimum of the passes"
            )
        else:
            reference = stored_reference(
                [
                    STAGE1 / "refs" / v / "summary.json"
                    for v in ("stage1c", "stage1c_large")
                ],
                "presolve_core_seconds",
            )
            source = "stored reference, ANOTHER machine, one run"
        line = report(rows, reference, "testbed " + tag, source)
    if not rows:
        sys.exit("no engine results for that tag")
    track(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
