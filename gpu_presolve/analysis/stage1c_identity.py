# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the results of two engine versions instance by instance.

Usage: stage1c_identity.py <tag A> <tag B> [--sizes-only]

Reads the release runs of tools/stage1c_engine_runs.sh (or of
tools/stage1_host_runs.sh) of both tags on the three instance sets:
results/bench240/<tag>/release_all (the 240 benchmark instances),
results/stage1/host/<tag>/release_dev (the testbed below 1e7 nonzeros) and
release_large (the testbed above). A set that neither tag has is left out; a
set that only one has is an error.

Compared per instance: status, reduced rows, columns and nonzeros, hash of
the reduced model, fixed-point flag, rounds, rebuilds, rounds on explicit
bounds, parallel-row merges and subtractions, and for the start step and
every rule family that both versions have its calls, work, findings, records
and removals. Seconds are not compared. With --sizes-only the comparison
ends after the fixed-point flag: that is what must hold when a version with
its new rules switched off is compared with the version before it, where
the number of rounds may differ.

Exit code 0 when every compared instance is equal and none is missing.
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
SETS = (
    ("the 240 benchmark instances", "bench240/%s/release_all"),
    ("testbed below 1e7 nonzeros", "stage1/host/%s/release_dev"),
    ("testbed above 1e7 nonzeros", "stage1/host/%s/release_large"),
)
SIZES = ("status", "reduced", "hash", "fixed_point")
COUNTS = ("rounds", "rebuilds", "parallel_rows")
STATS = (
    "calls",
    "work",
    "findings",
    "records",
    "rows_removed",
    "cols_removed",
    "nnz_removed",
)


def instances(path):
    """Return the instances of a runner summary by name, or None."""
    if not path.is_file():
        return None
    with open(path) as f:
        return {r["name"]: r for r in json.load(f)["instances"]}


def differences(a, b, sizes_only):
    """Return the names of the fields in which two results differ."""
    out = [k for k in SIZES if a.get(k) != b.get(k)]
    if a.get("same_hash") is False or b.get("same_hash") is False:
        out.append("repetitions disagree")
    if sizes_only:
        return out
    out += [k for k in COUNTS if a.get(k) != b.get(k)]
    if (a.get("explicit") or {}).get("rounds") != (
        b.get("explicit") or {}
    ).get("rounds"):
        out.append("explicit rounds")
    for k in STATS:
        if (a.get("start") or {}).get(k) != (b.get("start") or {}).get(k):
            out.append("start." + k)
    rules_a, rules_b = a.get("rules") or {}, b.get("rules") or {}
    for family in rules_a:
        if family not in rules_b:
            continue
        for k in STATS:
            if rules_a[family].get(k) != rules_b[family].get(k):
                out.append("%s.%s" % (family, k))
    return out


def main():
    """Print the comparison; return the exit code."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sizes_only = "--sizes-only" in sys.argv[1:]
    if len(args) != 2:
        sys.exit(__doc__)
    tag_a, tag_b = args
    failed = False
    compared = 0
    for label, pattern in SETS:
        a = instances(RESULTS / (pattern % tag_a) / "summary.json")
        b = instances(RESULTS / (pattern % tag_b) / "summary.json")
        if a is None and b is None:
            continue
        if a is None or b is None:
            print(
                "%s: no summary for %s"
                % (label, tag_a if a is None else tag_b)
            )
            failed = True
            continue
        missing = sorted(set(a) ^ set(b))
        common = sorted(set(a) & set(b))
        differing = {}
        for name in common:
            fields = differences(a[name], b[name], sizes_only)
            if fields:
                differing[name] = fields
        compared += len(common)
        print(
            "%s: %d compared, %d equal, %d differ, %d only in one version"
            % (
                label,
                len(common),
                len(common) - len(differing),
                len(differing),
                len(missing),
            )
        )
        for name in sorted(differing)[:25]:
            print("    %-28s %s" % (name, ", ".join(differing[name][:8])))
        if len(differing) > 25:
            print("    ... and %d more" % (len(differing) - 25))
        if missing:
            print("    only in one version: %s" % ", ".join(missing[:12]))
        failed = failed or bool(differing) or bool(missing)
    if compared == 0:
        print("nothing to compare for %s and %s" % (tag_a, tag_b))
        return 1
    print(
        "%s against %s, %s: %s"
        % (
            tag_a,
            tag_b,
            "sizes, hash and fixed-point flag" if sizes_only else "all counts",
            "DIFFERENT" if failed else "identical",
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
