# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Show which stages of HiGHS's presolve remove what on the 240 benchmark instances.

Usage: bench240_highs_stages.py [variant]   (default highs_diag2_full)
Reads results/bench240/<variant>/summary.json, made by
baseline/run_baseline.py with the instrumented HiGHS build (diag2), which
counts rows, columns and nonzeros removed and the seconds of every stage of
one presolve call. Per stage, over the instances:
- the mean share of an instance's nonzeros (rows, columns) that the stage
  removes, so that every instance counts the same;
- on how many instances the stage removes at least 1% of the nonzeros;
- the share of the summed seconds.
A stage can add nonzeros (substitutions, clique merging); its share is then
negative. The stages are grouped by the stage of our plan that holds their
rules; the grouping is rough for initial_scan and fast_loop, which also run
single-column rules that our Stage 1 does not have.
"""

import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "results" / "bench240"
PLAN = {
    "initial_scan": "1 (cheap row and column rules)",
    "fast_loop": "1 (cheap row and column rules)",
    "parallel_rows_cols": "1 rows, 3 columns",
    "enumeration": "2 (probing, cliques)",
    "probing": "2 (probing, cliques)",
    "clique_substitutions": "2 (probing, cliques)",
    "aggregator": "3 (substitution)",
    "sparsify": "3 (sparsify)",
    "dominated_columns": "3 (dominated columns)",
    "fourier_motzkin": "3",
    "strengthen_inequalities": "3 (cover strengthening)",
    "implied_integers": "3",
}


def main():
    """Print one line per stage."""
    variant = sys.argv[1] if len(sys.argv) > 1 else "highs_diag2_full"
    with open(BENCH / variant / "summary.json") as f:
        rows = json.load(f)["instances"]
    used = [
        r
        for r in rows
        if isinstance((r.get("diag") or {}).get("stage"), dict)
        and not r.get("reached_time_limit")
    ]
    print(
        "%d instances, %d with stage counts and inside the time limit"
        % (len(rows), len(used))
    )
    stages = []
    for r in used:
        for name in r["diag"]["stage"]:
            if name not in stages:
                stages.append(name)
    total_seconds = sum(
        s.get("seconds", 0.0)
        for r in used
        for s in r["diag"]["stage"].values()
    )
    removed = {
        key: sum(
            1.0 - r[key + "_after"] / r[key]
            for r in used
            if r.get(key + "_after") is not None and r[key]
        )
        / len(used)
        for key in ("nnz", "rows", "cols")
    }
    print(
        "mean share removed by the whole presolve: nonzeros %.1f%%, rows "
        "%.1f%%, columns %.1f%%; %.0f s in stages"
        % (
            100 * removed["nnz"],
            100 * removed["rows"],
            100 * removed["cols"],
            total_seconds,
        )
    )
    print(
        "\n%-24s %8s %8s %8s %10s %8s  %s"
        % (
            "stage",
            "nnz %",
            "rows %",
            "cols %",
            ">=1% nnz",
            "time %",
            "stage of our plan",
        )
    )
    lines = []
    for name in stages:
        share = {"nnz": 0.0, "rows": 0.0, "cols": 0.0}
        hits = 0
        seconds = 0.0
        for r in used:
            stage = r["diag"]["stage"].get(name)
            if not stage:
                continue
            seconds += stage.get("seconds", 0.0)
            for key in share:
                if r[key]:
                    share[key] += stage.get(key + "_removed", 0) / r[key]
            if r["nnz"] and stage.get("nnz_removed", 0) >= 0.01 * r["nnz"]:
                hits += 1
        lines.append((share["nnz"] / len(used), name, share, hits, seconds))
    for mean, name, share, hits, seconds in sorted(lines, reverse=True):
        print(
            "%-24s %8.2f %8.2f %8.2f %10d %8.1f  %s"
            % (
                name,
                100 * mean,
                100 * share["rows"] / len(used),
                100 * share["cols"] / len(used),
                hits,
                100 * seconds / total_seconds,
                PLAN.get(name, ""),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
