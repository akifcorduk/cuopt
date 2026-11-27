# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the timing passes of an engine version against its base version.

Usage: stage1c_time.py <tag> [--limit PERCENT]

Reads results/stage1c/timing/<tag>/<side>_<set>_<pass>/summary.json as
written by tools/stage1c_engine_runs.sh: side base (the version before),
new (this version) and off (this version with its new rules switched off),
set all (the 240 benchmark instances), dev (the testbed below 1e7 nonzeros)
and large (the testbed above), passes 1 to 3 run in turn on one machine,
one process at a time.

Per set and side the seconds of an instance are the minimum over the
passes, and the parts (rule families, time outside the rules) are those of
the pass with that minimum. Printed: the sums and the ratio to the base,
the same split into rules and the parts outside them, the work units of the
rule families that both sides have, and the instances of at least 10 ms
whose ratio is above the limit (default 3 percent).
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
SETS = (
    ("all", "the 240 benchmark instances"),
    ("dev", "testbed below 1e7 nonzeros"),
    ("large", "testbed above 1e7 nonzeros"),
)
SIDES = ("base", "new", "off")
PARTS = ("state", "start", "storage", "finish", "hash")


def best(directory, side, set_name):
    """Return per instance the result of the pass with the least seconds."""
    out = {}
    passes = 0
    for k in (1, 2, 3):
        path = directory / ("%s_%s_%d" % (side, set_name, k)) / "summary.json"
        if not path.is_file():
            continue
        passes += 1
        with open(path) as f:
            for r in json.load(f)["instances"]:
                if "seconds" not in r:
                    continue
                kept = out.get(r["name"])
                if kept is None or r["seconds"] < kept["seconds"]:
                    out[r["name"]] = r
    return out, passes


def split(rows, names):
    """Return the summed seconds by rule family and by part outside them."""
    rules, parts = {}, {p: 0.0 for p in PARTS}
    for n in names:
        for family, stats in (rows[n].get("rules") or {}).items():
            rules[family] = rules.get(family, 0.0) + stats.get("seconds", 0.0)
        outside = rows[n].get("seconds_outside_rules") or {}
        for p in PARTS:
            parts[p] += outside.get(p, 0.0)
    return rules, parts


def work(rows, names, families):
    """Return the summed work units of the given rule families."""
    return sum(
        (rows[n].get("rules") or {}).get(f, {}).get("work", 0)
        for n in names
        for f in families
    )


def report(directory, set_name, label, limit):
    """Print one instance set; return whether it had results."""
    sides = {}
    for side in SIDES:
        rows, passes = best(directory, side, set_name)
        if rows:
            sides[side] = (rows, passes)
    if "base" not in sides or "new" not in sides:
        return False
    names = sorted(
        set.intersection(*(set(rows) for rows, _ in sides.values()))
    )
    base_rows = sides["base"][0]
    base_sum = sum(base_rows[n]["seconds"] for n in names)
    print(
        "%s: %d instances in every side, passes %s"
        % (
            label,
            len(names),
            ", ".join("%s %d" % (s, sides[s][1]) for s in sides),
        )
    )
    families = set.intersection(
        *(
            {f for n in names for f in (rows[n].get("rules") or {})}
            for rows, _ in sides.values()
        )
    )
    base_rules, base_parts = split(base_rows, names)
    for side, (rows, _) in sides.items():
        total = sum(rows[n]["seconds"] for n in names)
        rules, parts = split(rows, names)
        print(
            "  %-4s %9.3f s  %+6.2f%%   rules %8.3f s, outside %8.3f s, "
            "work of the shared families %d"
            % (
                side,
                total,
                100 * (total / base_sum - 1),
                sum(rules.values()),
                sum(parts.values()),
                work(rows, names, families),
            )
        )
        if side == "base":
            continue
        changes = [
            (rules.get(f, 0.0) - base_rules.get(f, 0.0), f)
            for f in sorted(set(rules) | set(base_rules))
        ] + [(parts[p] - base_parts[p], p) for p in PARTS]
        changes.sort(key=lambda c: -abs(c[0]))
        print(
            "       largest changes: %s"
            % ", ".join("%s %+.3f s" % (f, d) for d, f in changes[:6])
        )
        above = sorted(
            (
                (rows[n]["seconds"] / base_rows[n]["seconds"] - 1, n)
                for n in names
                if base_rows[n]["seconds"] >= 0.01
                and rows[n]["seconds"]
                > (1 + limit / 100) * base_rows[n]["seconds"]
            ),
            reverse=True,
        )
        timed = sum(1 for n in names if base_rows[n]["seconds"] >= 0.01)
        print(
            "       above %+.0f%% among the %d instances of at least 10 ms: %d%s"
            % (
                limit,
                timed,
                len(above),
                (
                    " ("
                    + ", ".join(
                        "%s %+.1f%%" % (n, 100 * r) for r, n in above[:8]
                    )
                    + ")"
                )
                if above
                else "",
            )
        )
    return True


def main():
    """Print the timing report of one tag; return the exit code."""
    args = sys.argv[1:]
    limit = 3.0
    if "--limit" in args:
        k = args.index("--limit")
        limit = float(args[k + 1])
        del args[k : k + 2]
    if len(args) != 1:
        sys.exit(__doc__)
    directory = RESULTS / "stage1c" / "timing" / args[0]
    found = False
    for set_name, label in SETS:
        found = report(directory, set_name, label, limit) or found
    if not found:
        print("no timing passes with a base and a new side in %s" % directory)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
