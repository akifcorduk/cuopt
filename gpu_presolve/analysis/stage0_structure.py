# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 structure analysis of the GPU presolve testbed.

Reads the Stage 0 result files next to this script and prints markdown tables on what the
structure of the testbed instances says about kernels and data layout:

1. row and column length distributions, long rows and columns, length bins of DESIGN.md 2.3;
2. infinite activity terms per row (DESIGN.md 2.4) and column bounds;
3. row classes, and what HiGHS's clique table, clique merging and variable-bound aggregation
   did on the same instances;
4. coefficient range and row dynamism: raw, with HiGHS's scaling, and of the presolved model;
5. variable types before and after HiGHS presolve, implied integers;
6. bands A1 and A2: model shapes and the GPU fixed-cost arithmetic against HiGHS's time.

Inputs, relative to this file:
  ../testbed/testbed.json                      "main" list (names, bands)
  ../results/stage0/stats/<name>.json          structure statistics, dev-size instances
  ../results/stage0/stats_large/<name>.json    structure statistics, instances above 1e7 nonzeros
  ../results/stage0/stock/<name>.json          stock HiGHS presolve: time, presolved statistics
  ../results/stage0/diag/summary.json          patched HiGHS: parsed DIAG lines
  ../results/stage0/fixed_costs.json           GPU fixed costs
  ../results/stage0/sweeps/<name>.json         kernel micro-benchmark on the real matrix: exact
                                               rows and nonzeros per length class, rows with a
                                               finite lower / upper side, measured kernel times
  ../results/stage0/diag_stage1/<name>.json    patched HiGHS with enumeration, probing and the
  ../results/stage0/diag_stage1/summary.json   Stage 2 to 4 rules switched off (prepareProbing is
                                               never called)

The last two sets are optional: a table that needs them leaves out the instances without them.

No arguments. Python standard library only. Nothing is written; a main-list instance without a
statistics file is listed and left out.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE0 = os.path.normpath(os.path.join(HERE, "..", "results", "stage0"))
TESTBED = os.path.normpath(os.path.join(HERE, "..", "testbed", "testbed.json"))

# One device thread takes 0.5 to 0.6 microseconds per row entry on the H200
# (gpu_presolve/microbench/README_sweeps.md, "Things to know when reading the numbers").
THREAD_US_PER_ENTRY = 0.5

SUBWARP_MAX = 64
BLOCK_MAX = 8192
LENGTH_CLASSES = [
    (2, "0-2"),
    (4, "3-4"),
    (8, "5-8"),
    (16, "9-16"),
    (32, "17-32"),
    (64, "33-64"),
    (8192, "65-8192"),
]
CLASS_NAMES = [c[1] for c in LENGTH_CLASSES] + [">8192"]

ROW_CLASSES = [
    "empty",
    "singleton",
    "doubleton_equation",
    "set_packing",
    "set_partitioning",
    "set_covering",
    "cardinality",
    "binary_knapsack_non_unit",
    "variable_bound",
    "other_two_entry",
    "all_integer",
    "mixed",
    "all_continuous",
]
ROW_CLASS_SHORT = [
    "empty",
    "single",
    "dbl eq",
    "pack",
    "part",
    "cover",
    "card",
    "knap",
    "var bnd",
    "other 2",
    "all int",
    "mixed",
    "all cont",
]

ALL_INSTANCES = []


def load(path):
    with open(path) as f:
        return json.load(f)


def load_optional(path):
    return load(path) if os.path.exists(path) else None


def num(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, int):
        return f"{x:,}"
    if x != x:
        return "nan"
    if x in (math.inf, -math.inf):
        return "inf"
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1e7 or a < 1e-3:
        return f"{x:.{digits - 1}e}"
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:.0f}" if digits <= 3 else f"{x:.{digits - 3}f}"
    if a >= 10:
        return f"{x:.{max(digits - 2, 0)}f}"
    if a >= 1:
        return f"{x:.{max(digits - 1, 0)}f}"
    return f"{x:.{digits}g}"


def pct(part, whole, digits=1):
    if part is None or whole is None or whole == 0:
        return "-"
    return f"{100.0 * part / whole:.{digits}f}"


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def table(title, headers, rows, note=None):
    print()
    print(f"### {title}")
    print()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    if note:
        print()
        print(note)


def names_of(instances):
    """Names of a subset; the complement when that is the shorter list."""
    if not instances:
        return "none"
    names = [i["name"] for i in instances]
    rest = [i["name"] for i in ALL_INSTANCES if i["name"] not in names]
    if ALL_INSTANCES and not rest:
        return "all"
    if rest and len(rest) < len(names) and len(names) > 12:
        return "all but " + ", ".join(rest)
    return ", ".join(names)


def length_class(length):
    for upper, name in LENGTH_CLASSES:
        if length <= upper:
            return name
    return ">8192"


def share_at_most(dist, limit):
    """Largest known percentile whose value is at most `limit` (lower bound on the share)."""
    for p, key in (
        (100, "max"),
        (99, "p99"),
        (90, "p90"),
        (50, "p50"),
        (10, "p10"),
    ):
        if dist[key] <= limit:
            return p
    return 0


def rank_segments(dist, n):
    """Bounds on the sorted lengths from the nearest-rank percentiles.

    The statistics give the value at the positions ceil(p n / 100) of the ascending list, plus
    min and max. A length between two known positions is bracketed by their two values.
    Returns segments (first rank, last rank, lower bound, upper bound), ranks 1-based.
    """
    known = {1: dist["min"]}
    for p, key in ((10, "p10"), (50, "p50"), (90, "p90"), (99, "p99")):
        known[max(1, math.ceil(p * n / 100))] = dist[key]
    known[n] = dist["max"]
    ranks = sorted(known)
    segments = []
    for a, b in zip(ranks, ranks[1:] + [None]):
        segments.append((a, a, known[a], known[a]))
        if b is not None and b > a + 1:
            segments.append((a + 1, b - 1, known[a], known[b]))
    return segments


def sum_bounds(segments, first, last, floor=None, cap=None):
    low = high = 0
    for a, b, lb, ub in segments:
        count = min(b, last) - max(a, first) + 1
        if count <= 0:
            continue
        low += count * (max(lb, floor) if floor is not None else lb)
        high += count * (min(ub, cap) if cap is not None else ub)
    return low, high


def longest_share_bounds(dist, n, nnz, k, floor=None, cap=None):
    """Bounds on the share of the nonzeros held by the k longest rows (or columns).

    `floor`: known lower bound on each of the k (used for the long rows, which are longer than
    their threshold); `cap`: known upper bound on each of the others.
    """
    if k <= 0:
        return 0.0, 0.0
    segments = rank_segments(dist, n)
    top_low, top_high = sum_bounds(segments, n - k + 1, n, floor=floor)
    rest_low, rest_high = sum_bounds(segments, 1, n - k, cap=cap)
    low = max(top_low, nnz - rest_high)
    high = min(top_high, nnz - rest_low)
    return low / nnz, high / nnz


def sweep_ms(instance, section, name):
    """Median milliseconds of one variant in sweeps/<name>.json, or None."""
    sweeps = instance.get("sweeps")
    if not sweeps:
        return None
    for r in sweeps["results"].get(section, []):
        if r["name"] == name:
            return r["ms_median"]
    return None


def sweep_ms_prefix(instance, section, prefix):
    sweeps = instance.get("sweeps")
    if not sweeps:
        return None
    for r in sweeps["results"].get(section, []):
        if r["name"].startswith(prefix):
            return r["ms_median"]
    return None


def interpolate(points, x):
    """Piecewise-linear interpolation through sorted (x, y) points, clamped at the ends."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def read_instances():
    testbed = load(TESTBED)
    diag_summary = load_optional(os.path.join(STAGE0, "diag", "summary.json"))
    diag_by_name = {}
    if diag_summary:
        diag_by_name = {i["name"]: i for i in diag_summary["instances"]}
    stage1_summary = load_optional(
        os.path.join(STAGE0, "diag_stage1", "summary.json")
    )
    stage1_by_name = {}
    if stage1_summary:
        stage1_by_name = {i["name"]: i for i in stage1_summary["instances"]}
    instances = []
    missing = []
    for entry in testbed["main"]:
        name = entry["name"]
        stats = None
        for directory in ("stats", "stats_large"):
            stats = load_optional(
                os.path.join(STAGE0, directory, name + ".json")
            )
            if stats:
                stats_dir = directory
                break
        if not stats:
            missing.append(name)
            continue
        stock = load_optional(os.path.join(STAGE0, "stock", name + ".json"))
        diag_row = diag_by_name.get(name)
        sweeps = load_optional(os.path.join(STAGE0, "sweeps", name + ".json"))
        if sweeps:
            m, o = sweeps["matrix"], stats["original"]
            if (m["rows"], m["cols"], m["nnz"]) != (
                o["rows"],
                o["cols"],
                o["nnz"],
            ):
                sweeps = None
        stage1 = load_optional(
            os.path.join(STAGE0, "diag_stage1", name + ".json")
        )
        stage1_row = stage1_by_name.get(name)
        instances.append(
            {
                "sweeps": sweeps,
                "stage1": stage1,
                "stage1_diag": stage1_row["diag"]
                if stage1_row and stage1_row.get("diag")
                else None,
                "name": name,
                "band": entry["band"],
                "stats_dir": stats_dir,
                "o": stats["original"],
                "stock": stock,
                "p": stock.get("presolved") if stock else None,
                "presolve": stock.get("presolve") if stock else None,
                "diag": diag_row["diag"]
                if diag_row and diag_row.get("diag")
                else None,
            }
        )
    return instances, missing


def fixed_cost_table():
    data = load(os.path.join(STAGE0, "fixed_costs.json"))
    costs = {"gather": {}, "library": {}, "upload": []}
    for r in data["results"]:
        group, name, variant, size = (
            r["group"],
            r["name"],
            r.get("variant", ""),
            r.get("size", ""),
        )
        if group == "launch":
            costs[name] = r["median_us"]
        elif group == "readback":
            costs["readback_" + variant] = r["median_us"]
            costs["readback_" + variant + "_p90"] = r["p90_us"]
        elif group == "library" and variant == "rmm_pool":
            costs["library"].setdefault(name, {})[size] = r["median_us"]
        elif (
            group == "handover_copy"
            and name == "host_to_device"
            and variant == "pinned"
        ):
            costs["upload"].append((r["bytes"], r["median_us"]))
        elif group == "gather" and name == "gather_rows_to_host":
            phases = {p["name"]: p["median_us"] for p in r.get("phases", [])}
            costs["gather"].setdefault(size, {})[variant] = {
                "total": r["median_us"],
                "kernel": phases.get("scan_gather_count"),
                "rows": r.get("frontier_rows"),
                "entries": r.get("elements"),
            }
    costs["upload"].sort()
    costs["system"] = data.get("system", {})
    return costs


# --------------------------------------------------------------------------------------------
# 1. Row and column lengths
# --------------------------------------------------------------------------------------------


def section_lengths(instances, costs):
    print()
    print("## 1. Row and column lengths (original models)")

    rows = []
    for i in instances:
        o, d = i["o"], i["o"]["row_length"]
        rows.append(
            [
                i["band"],
                i["name"],
                num(o["rows"]),
                num(o["nnz"]),
                num(d["p50"]),
                num(d["p90"]),
                num(d["p99"]),
                num(d["max"]),
                num(d["mean"]),
                pct(d["max"], o["cols"]),
                num(o["long_rows"]["count"]),
                num(o["long_rows"]["threshold"]),
            ]
        )
    table(
        "1a. Row length per instance",
        [
            "band",
            "instance",
            "rows",
            "nonzeros",
            "p50",
            "p90",
            "p99",
            "max",
            "mean",
            "max as % of columns",
            "long rows",
            "long-row threshold",
        ],
        rows,
        "long rows = rows longer than max(1000, 1% of columns).",
    )

    rows = []
    for i in instances:
        o, d = i["o"], i["o"]["col_length"]
        rows.append(
            [
                i["band"],
                i["name"],
                num(o["cols"]),
                num(d["p50"]),
                num(d["p90"]),
                num(d["p99"]),
                num(d["max"]),
                num(d["mean"]),
                pct(d["max"], o["rows"]),
                num(o["long_cols"]["count"]),
                num(o["long_cols"]["threshold"]),
            ]
        )
    table(
        "1b. Column length per instance",
        [
            "band",
            "instance",
            "columns",
            "p50",
            "p90",
            "p99",
            "max",
            "mean",
            "max as % of rows",
            "long columns",
            "long-column threshold",
        ],
        rows,
        "long columns = columns longer than max(1000, 1% of rows).",
    )

    rows = []
    for limit in (64, 1000, 8192, 10000, 100000):
        with_rows = [
            i for i in instances if i["o"]["row_length"]["max"] > limit
        ]
        with_cols = [
            i for i in instances if i["o"]["col_length"]["max"] > limit
        ]
        either = [i for i in instances if i in with_rows or i in with_cols]
        rows.append(
            [
                f"> {limit:,}",
                len(with_rows),
                names_of(with_rows),
                len(with_cols),
                names_of(with_cols),
                len(either),
            ]
        )
    table(
        f"1c. Instances (of {len(instances)}) whose longest row or column exceeds a length",
        [
            "length",
            "rows: instances",
            "rows: which",
            "columns: instances",
            "columns: which",
            "either",
        ],
        rows,
    )

    for kind, key, count_key, long_key in (
        ("rows", "row_length", "rows", "long_rows"),
        ("columns", "col_length", "cols", "long_cols"),
    ):
        rows = []
        for i in instances:
            o, d = i["o"], i["o"][key]
            n, nnz = o[count_key], o["nnz"]
            k = n - math.ceil(99 * n / 100)
            low, high = longest_share_bounds(d, n, nnz, k)
            m, threshold = o[long_key]["count"], o[long_key]["threshold"]
            long_low, long_high = longest_share_bounds(
                d, n, nnz, m, floor=threshold + 1, cap=threshold
            )
            rows.append(
                [
                    i["band"],
                    i["name"],
                    num(k),
                    num(d["p99"]),
                    num(d["max"]),
                    pct(d["max"], nnz, 2),
                    f"{100 * low:.1f}",
                    f"{100 * high:.1f}",
                    "yes" if d["max"] == d["p99"] else "no",
                    num(m),
                    f"{100 * long_low:.1f} to {100 * long_high:.1f}"
                    if m
                    else "0",
                ]
            )
        table(
            f"1d. Share of the nonzeros in the longest 1% of {kind} "
            "(bounds from the percentiles)",
            [
                "band",
                "instance",
                f"{kind} above the p99 position (k)",
                "p99",
                "max",
                "longest one, % of nonzeros (exact)",
                "k longest: at least %",
                "k longest: at most %",
                "max = p99",
                f"long {kind}",
                f"long {kind}: % of nonzeros (bounds)",
            ],
            rows,
            f"The statistics hold percentiles, not a histogram. The k longest {kind} each have "
            "between p99 and max entries and the others are bracketed by the percentiles "
            f"around them, which gives the two bounds. The {kind} strictly longer than p99 are "
            "a subset of these k (none when max = p99), so their share is at most the upper "
            f"bound. For the long {kind} the bounds also use the threshold. The exact shares "
            "per length class are in table 1h (sweeps/<name>.json); the exact share of the "
            "longest 1% needs the per-row lengths (the CSR dumps on the node).",
        )

    for kind, key in (("rows", "row_length"), ("columns", "col_length")):
        rows = []
        for i in instances:
            d = i["o"][key]
            floor_ms = d["max"] * THREAD_US_PER_ENTRY / 1000.0
            rows.append(
                [
                    i["band"],
                    i["name"],
                    length_class(d["p50"]),
                    length_class(d["p90"]),
                    length_class(d["p99"]),
                    length_class(d["max"]),
                    share_at_most(d, 2),
                    share_at_most(d, 4),
                    share_at_most(d, 32),
                    share_at_most(d, SUBWARP_MAX),
                    share_at_most(d, BLOCK_MAX),
                    num(floor_ms),
                ]
            )
        table(
            f"1e. Length classes of the binned matrix: {kind}",
            [
                "band",
                "instance",
                "class of p50",
                "class of p90",
                "class of p99",
                "class of max",
                "% with <= 2 (at least)",
                "% with <= 4 (at least)",
                "% with <= 32 (at least)",
                "% with <= 64 (at least)",
                "% with <= 8192 (at least)",
                "one thread on the longest: ms",
            ],
            rows,
            "Classes: sub-warp groups 0-2, 3-4, 5-8, 9-16, 17-32, 33-64 (one launch), one block "
            "per row for 65-8192, chunks above 8192 (DESIGN.md 2.3, README_sweeps.md variant B). "
            '"at least" = the largest of the percentiles 10, 50, 90, 99, 100 whose value is '
            "within the limit; 0 means even p10 is above it. Last column: longest length times "
            f"{THREAD_US_PER_ENTRY} microseconds, the per-entry latency of one device thread "
            "stated in README_sweeps.md; it is an estimated floor of a kernel that gives the "
            f"longest one of the {kind} to a single thread, not a measurement. The measured "
            "time of the one-thread-per-row activity kernel on each matrix is in table 1i; "
            "no one-thread-per-column kernel was measured.",
        )

    rows = []
    for name in CLASS_NAMES:
        row = [name]
        for key in ("row_length", "col_length"):
            for q in ("p50", "p90", "p99", "max"):
                row.append(
                    sum(
                        1
                        for i in instances
                        if length_class(i["o"][key][q]) == name
                    )
                )
        rows.append(row)
    table(
        f"1f. Number of instances (of {len(instances)}) by the length class of each percentile",
        [
            "class",
            "row p50",
            "row p90",
            "row p99",
            "row max",
            "col p50",
            "col p90",
            "col p99",
            "col max",
        ],
        rows,
    )

    rows = []
    for size in sorted(costs["gather"], key=float):
        g = costs["gather"][size]
        t, b, w = (
            g["thread_per_row"],
            g["thread_per_row_block_32"],
            g["warp_per_row"],
        )
        rows.append(
            [
                size,
                num(t["rows"]),
                num(t["kernel"], 4),
                num(b["kernel"], 4),
                num(w["kernel"], 4),
                num(t["kernel"] / w["kernel"]),
                num(b["kernel"] / w["kernel"]),
                num(t["total"], 4),
                num(w["total"], 4),
            ]
        )
    table(
        "1g. fixed_costs gather: one thread per row against one warp per row (microseconds)",
        [
            "gathered entries",
            "frontier rows",
            "thread per row (256 rows per block): kernel part",
            "thread per row (32 rows per block): kernel part",
            "warp per row: kernel part",
            "thread / warp",
            "thread (block 32) / warp",
            "thread per row: total",
            "warp per row: total",
        ],
        rows,
        "kernel part = phase scan_gather_count (prefix sum, gather kernel, count read back); "
        "total adds the copy of the buffer to the host. The benchmark matrix has rows of 4 to "
        "64 entries (uniform), so it says nothing about rows of 1 to 3 entries or above 64.",
    )

    in_range = [i for i in instances if 4 <= i["o"]["row_length"]["p50"] <= 64]
    below = [i for i in instances if i["o"]["row_length"]["p50"] < 4]
    above = [i for i in instances if i["o"]["row_length"]["p50"] > 64]
    print()
    print(
        f"Median row length inside the benchmark's 4 to 64 range: {len(in_range)} instances "
        f"({names_of(in_range)}); below 4: {len(below)} ({names_of(below)}); above 64: "
        f"{len(above)} ({names_of(above)})."
    )
    all_short = [
        i for i in instances if i["o"]["row_length"]["max"] <= SUBWARP_MAX
    ]
    all_short_cols = [
        i for i in instances if i["o"]["col_length"]["max"] <= SUBWARP_MAX
    ]
    p99_short = [
        i for i in instances if i["o"]["row_length"]["p99"] <= SUBWARP_MAX
    ]
    p99_short_cols = [
        i for i in instances if i["o"]["col_length"]["p99"] <= SUBWARP_MAX
    ]
    print(
        f"All rows within {SUBWARP_MAX} entries: {len(all_short)} instances "
        f"({names_of(all_short)}); at least 99% of the rows: {len(p99_short)}. All columns "
        f"within {SUBWARP_MAX}: {len(all_short_cols)} ({names_of(all_short_cols)}); at least "
        f"99% of the columns: {len(p99_short_cols)}."
    )

    section_length_classes_exact(instances)
    section_kernels_measured(instances)


def class_shares(matrix, classes_key, count_key):
    """Rows (or columns) and nonzeros in the sub-warp, block and chunk classes, exact."""
    out = {"sub": [0, 0], "block": [0, 0], "chunk": [0, 0], "le4": [0, 0]}
    for c in matrix[classes_key]:
        if c["len"] == ">8192":
            key = "chunk"
        elif c["len"] == "65-8192":
            key = "block"
        else:
            key = "sub"
        out[key][0] += c[count_key]
        out[key][1] += c["nnz"]
        if c["len"] in ("0-2", "3-4"):
            out["le4"][0] += c[count_key]
            out["le4"][1] += c["nnz"]
    return out


def section_length_classes_exact(instances):
    with_sweeps = [i for i in instances if i["sweeps"]]
    if not with_sweeps:
        return
    rows = []
    shares = {}
    for i in with_sweeps:
        m = i["sweeps"]["matrix"]
        r = class_shares(m, "row_classes", "rows")
        c = class_shares(m, "col_classes", "cols")
        shares[i["name"]] = (r, c)
        nnz = m["nnz"]
        rows.append(
            [
                i["band"],
                i["name"],
                pct(r["le4"][0], m["rows"]),
                pct(r["le4"][1], nnz),
                pct(r["sub"][0], m["rows"]),
                pct(r["sub"][1], nnz),
                num(r["block"][0]),
                pct(r["block"][1], nnz),
                num(r["chunk"][0]),
                pct(r["chunk"][1], nnz),
                pct(c["sub"][0], m["cols"]),
                pct(c["sub"][1], nnz),
                num(c["block"][0]),
                pct(c["block"][1], nnz),
                num(c["chunk"][0]),
                pct(c["chunk"][1], nnz),
            ]
        )
    table(
        "1h. Exact rows and nonzeros per length class (sweeps/<name>.json, matrix.row_classes "
        "and col_classes)",
        [
            "band",
            "instance",
            "rows <= 4: % of rows",
            "% of nonzeros",
            "rows <= 64: % of rows",
            "% of nonzeros",
            "rows 65-8192",
            "% of nonzeros",
            "rows > 8192",
            "% of nonzeros",
            "columns <= 64: % of columns",
            "% of nonzeros",
            "columns 65-8192",
            "% of nonzeros",
            "columns > 8192",
            "% of nonzeros",
        ],
        rows,
        "Counted by the sweeps tool on the CSR of the original model. These are the classes "
        "of the binned matrix (DESIGN.md 2.3): lane groups up to 64 entries, one block per "
        "row or column up to 8,192, chunks above.",
    )

    def share(name, orient, key):
        r = shares[name][orient]
        total = sum(r[k][1] for k in ("sub", "block", "chunk"))
        return r[key][1] / total if total else 0.0

    def above64(name, orient):
        return share(name, orient, "block") + share(name, orient, "chunk")

    names = [i["name"] for i in with_sweeps]
    for orient, kind in ((0, "rows"), (1, "columns")):
        half = [n for n in names if above64(n, orient) >= 0.5]
        tenth = [n for n in names if above64(n, orient) >= 0.1]
        none = [n for n in names if above64(n, orient) == 0]
        chunk = [n for n in names if shares[n][orient]["chunk"][0] > 0]
        print()
        print(
            f"Of {len(names)} instances, {kind} above 64 entries hold at least half of the "
            f"nonzeros on {len(half)} ({', '.join(half)}), at least 10% on {len(tenth)}, none "
            f"on {len(none)}. {kind.capitalize()} above 8,192 entries exist on {len(chunk)}: "
            + ", ".join(
                f"{n} ({shares[n][orient]['chunk'][0]:,} holding "
                f"{100 * share(n, orient, 'chunk'):.1f}% of the nonzeros)"
                for n in chunk
            )
            + "."
        )
    either = [n for n in names if above64(n, 0) >= 0.5 or above64(n, 1) >= 0.5]
    le4_rows = [
        n
        for n in names
        if shares[n][0]["le4"][0]
        >= 0.5 * sum(shares[n][0][k][0] for k in ("sub", "block", "chunk"))
    ]
    le4_nnz = [n for n in names if share(n, 0, "le4") >= 0.5]
    print(
        f"In at least one orientation the classes above 64 hold at least half of the nonzeros "
        f"on {len(either)} of {len(names)}. Rows of at most 4 entries are at least half of "
        f"the rows on {len(le4_rows)} and hold at least half of the nonzeros on "
        f"{len(le4_nnz)} ({', '.join(le4_nnz)})."
    )


def section_kernels_measured(instances):
    with_sweeps = [i for i in instances if i["sweeps"]]
    if not with_sweeps:
        return
    rows = []
    t_over_b, h_over_b, t_over_est, cand = {}, {}, {}, {}
    for i in with_sweeps:
        m = i["sweeps"]["matrix"]
        t = sweep_ms(i, "activity", "T/S/-/plain")
        b = sweep_ms(i, "activity", "B/S/A/plain")
        h = sweep_ms(i, "activity", "H/S/A/plain")
        c = sweep_ms(i, "activity", "C/S/A/plain")
        cpu = sweep_ms(i, "cpu", "activity, full, plain")
        cb = sweep_ms(i, "candidate", "B/S/plain")
        ch = sweep_ms(i, "candidate", "H/S/plain")
        if None in (t, b, h, c, cpu, cb, ch):
            continue
        estimate = m["max_row_len"] * THREAD_US_PER_ENTRY / 1000.0
        t_over_b[i["name"]], h_over_b[i["name"]] = t / b, h / b
        t_over_est[i["name"]] = t / estimate
        cand[i["name"]] = cb / ch
        rows.append(
            [
                i["band"],
                i["name"],
                num(m["max_row_len"]),
                num(t, 4),
                num(estimate, 4),
                num(t / estimate),
                num(b, 4),
                num(h, 4),
                num(c, 4),
                num(t / b),
                num(h / b),
                num(c / b),
                num(cpu, 4),
                num(cpu / b),
                num(cb, 4),
                num(ch, 4),
                num(cb / ch),
            ]
        )
    table(
        "1i. Measured kernels on the testbed matrices (sweeps/<name>.json, median "
        "milliseconds for one full pass, plain sums, H200)",
        [
            "band",
            "instance",
            "longest row",
            "T: one thread per row",
            f"longest row x {THREAD_US_PER_ENTRY} us (estimate of 1e)",
            "T / estimate",
            "B: binned, lane groups",
            "H: binned, one thread per row up to 64",
            "C: one block per row",
            "T / B",
            "H / B",
            "C / B",
            "one CPU thread",
            "CPU / B",
            "candidates (columns): B",
            "candidates: H",
            "candidates B / H",
        ],
        rows,
        "Activity variants T/S/-/plain, B/S/A/plain, H/S/A/plain, C/S/A/plain and the CPU "
        'reference "activity, full, plain"; candidate variants B/S/plain and H/S/plain '
        "(README_sweeps.md). B and H treat rows above 64 entries the same way (one block up "
        "to 8,192, chunks above), so H / B compares one thread per row with lane groups on "
        "the rows up to 64 entries only. One run per matrix.",
    )
    if not t_over_b:
        return
    long_rows = sorted(
        (n for n in t_over_b if t_over_b[n] >= 10), key=lambda n: -t_over_b[n]
    )
    h_faster = [n for n in h_over_b if h_over_b[n] < 0.98]
    b_faster = [n for n in h_over_b if h_over_b[n] > 1.02]
    cand_b = [n for n in cand if cand[n] < 0.98]
    cand_h = [n for n in cand if cand[n] > 1.02]
    print()
    print(
        f"One thread per row (T) is at least 10 times slower than the binned kernel (B) on "
        f"{len(long_rows)} of {len(t_over_b)} instances: "
        + ", ".join(f"{n} ({t_over_b[n]:.0f}x)" for n in long_rows)
        + "."
    )
    long_est = [
        t_over_est[i["name"]]
        for i in with_sweeps
        if i["name"] in t_over_est
        and i["sweeps"]["matrix"]["max_row_len"] > 1000
    ]
    if long_est:
        print(
            f"Measured T against the estimate of table 1e (longest row x "
            f"{THREAD_US_PER_ENTRY} us) on the {len(long_est)} instances with a row above "
            f"1,000 entries: {min(long_est):.2f} to {max(long_est):.2f} times the estimate; "
            "the estimate is a floor. On matrices with short rows the time of T is launch "
            "cost, not the longest row."
        )
    print(
        f"Rows up to 64 entries, activity: lane groups (B) are faster than one thread per row "
        f"(H) by more than 2% on {len(b_faster)} instances (largest H / B "
        f"{max(h_over_b.values()):.2f}, {max(h_over_b, key=h_over_b.get)}), slower on "
        f"{len(h_faster)} ({', '.join(h_faster)}; smallest H / B "
        f"{min(h_over_b.values()):.2f}). Candidates over the columns: B faster by more than "
        f"2% on {len(cand_b)}, H faster on {len(cand_h)} ({', '.join(cand_h)})."
    )


# --------------------------------------------------------------------------------------------
# 2. Infinite activity terms and bounds
# --------------------------------------------------------------------------------------------


def single_infinity_bounds(model, side_rows=None):
    """Bounds on the rows in which a side the row has sees exactly one infinite term.

    `side_rows`: (rows with a finite upper side, rows with a finite lower side) when known
    (sweeps/<name>.json for the original models). The upper side uses the minimum activity,
    the lower side the maximum activity. Without it only the two-sided rows are known to have
    both sides, which gives weaker bounds.
    """
    t, n = model["rows_by_infinite_activity_terms"], model["rows"]
    one_min, one_max = t["min_activity"]["one"], t["max_activity"]["one"]
    sides = model["row_sides"]
    two_sided = sides["equation"] + sides["ranged"]
    low = max(
        0,
        one_min + one_max - n - sides["free"],
        one_min + two_sided - n,
        one_max + two_sided - n,
    )
    high = min(n - sides["free"], one_min + one_max)
    if side_rows:
        upper_rows, lower_rows = side_rows
        low = max(low, one_min + upper_rows - n, one_max + lower_rows - n)
        high = min(high, min(one_min, upper_rows) + min(one_max, lower_rows))
    return low, high


def side_rows_of(instance):
    """(rows with a finite upper side, rows with a finite lower side) of the original model."""
    sweeps = instance.get("sweeps")
    if not sweeps:
        return None
    m, t = sweeps["matrix"], instance["o"]["rows_by_infinite_activity_terms"]
    if "rows_finite_upper" not in m or "rows_finite_lower" not in m:
        return None
    for key, side in (
        ("rows_by_infinite_terms_min", "min_activity"),
        ("rows_by_infinite_terms_max", "max_activity"),
    ):
        if key in m and list(m[key]) != [
            t[side]["none"],
            t[side]["one"],
            t[side]["two_or_more"],
        ]:
            return None
    return m["rows_finite_upper"], m["rows_finite_lower"]


def infinite_terms_rows(instances, key):
    rows = []
    for i in instances:
        m = i[key]
        if not m:
            continue
        t = m["rows_by_infinite_activity_terms"]
        mn, mx, n = t["min_activity"], t["max_activity"], m["rows"]
        sides = m["row_sides"]
        side_rows = side_rows_of(i) if key == "o" else None
        low, high = single_infinity_bounds(m, side_rows)
        rows.append(
            [
                i["band"],
                i["name"],
                num(n),
                pct(sides["equation"] + sides["ranged"], n),
                pct(side_rows[0], n) if side_rows else "-",
                pct(side_rows[1], n) if side_rows else "-",
                num(mn["one"]),
                pct(mn["one"], n),
                num(mn["two_or_more"]),
                pct(mn["two_or_more"], n),
                num(mx["one"]),
                pct(mx["one"], n),
                num(mx["two_or_more"]),
                pct(mx["two_or_more"], n),
                f"{pct(low, n)} to {pct(high, n)}",
            ]
        )
    return rows


def section_infinite(instances):
    print()
    print("## 2. Infinite activity terms and column bounds")

    headers = [
        "band",
        "instance",
        "rows",
        "two-sided rows %",
        "rows with an upper side %",
        "rows with a lower side %",
        "min activity: one",
        "%",
        "min activity: two or more",
        "%",
        "max activity: one",
        "%",
        "max activity: two or more",
        "%",
        "rows where a side the row has sees exactly one, % (bounds)",
    ]
    note = (
        "The statistics count the two activities separately and for every row, whether or "
        "not the row has the side that uses that activity (the upper side uses the minimum "
        "activity, the lower side the maximum). There is no cross-table of the two "
        "activities or of activity against row side. The numbers of rows with a finite "
        'upper and a finite lower side come from sweeps/<name>.json ("-" without it). '
        "Last column, with u = rows with an upper side and l = rows with a lower side: at "
        "most min(count min, u) + min(count max, l); at least the largest of count min + u "
        "- rows, count max + l - rows and count min + count max - rows - free rows. Without "
        "the side counts only the two-sided rows stand in for u and l."
    )
    table(
        "2a. Rows by the number of infinite terms in their activities (original models)",
        headers,
        infinite_terms_rows(instances, "o"),
        note,
    )

    def larger_share(i, what):
        t, n = i["o"]["rows_by_infinite_activity_terms"], i["o"]["rows"]
        return max(t[s][what] for s in ("min_activity", "max_activity")) / n

    def side_share(i, side, what):
        return (
            i["o"]["rows_by_infinite_activity_terms"][side][what]
            / i["o"]["rows"]
        )

    limits = (
        (0.0, "more than 0"),
        (0.01, "at least 1%"),
        (0.10, "at least 10%"),
        (0.50, "at least 50%"),
    )

    def passes(share, limit):
        return share > 0 if limit == 0.0 else share >= limit

    rows = []
    for limit, label in limits:
        mn = [
            i
            for i in instances
            if passes(side_share(i, "min_activity", "one"), limit)
        ]
        mx = [
            i
            for i in instances
            if passes(side_share(i, "max_activity", "one"), limit)
        ]
        both = [i for i in instances if passes(larger_share(i, "one"), limit)]
        sure = [
            i
            for i in instances
            if passes(
                single_infinity_bounds(i["o"], side_rows_of(i))[0]
                / i["o"]["rows"],
                limit,
            )
        ]
        possible = [
            i
            for i in instances
            if passes(
                single_infinity_bounds(i["o"], side_rows_of(i))[1]
                / i["o"]["rows"],
                limit,
            )
        ]
        rows.append(
            [
                label,
                len(mn),
                len(mx),
                len(both),
                names_of(both),
                len(possible),
                names_of(possible),
                len(sure),
                names_of(sure),
            ]
        )
    table(
        f"2b. Instances (of {len(instances)}) by the share of rows with exactly one infinite "
        "term",
        [
            "share of rows",
            "in the minimum activity",
            "in the maximum activity",
            "in either activity, row sides ignored (larger of the two shares)",
            "which",
            "possible on a side the row has (upper bound of 2a)",
            "which",
            "certain on a side the row has (lower bound of 2a)",
            "which",
        ],
        rows,
        '"row sides ignored" counts an activity even when the row lacks the side that uses '
        'it. "possible" and "certain" use the bounds of the last column of 2a, which '
        "take the row sides into account.",
    )

    rows = []
    for limit, label in limits:
        which = [
            i
            for i in instances
            if passes(larger_share(i, "two_or_more"), limit)
        ]
        rows.append([label, len(which), names_of(which)])
    table(
        f"2c. Instances (of {len(instances)}) by the share of rows with two or more infinite "
        "terms in one activity (no bound can be derived from that side)",
        ["share of rows (larger of the two activities)", "instances", "which"],
        rows,
    )

    total_rows = sum(i["o"]["rows"] for i in instances)
    print()
    for side in ("min_activity", "max_activity"):
        one = sum(
            i["o"]["rows_by_infinite_activity_terms"][side]["one"]
            for i in instances
        )
        two = sum(
            i["o"]["rows_by_infinite_activity_terms"][side]["two_or_more"]
            for i in instances
        )
        print(
            f"All {len(instances)} instances pooled ({total_rows:,} rows), {side}: exactly one "
            f"infinite term {one:,} rows ({pct(one, total_rows)}%), two or more {two:,} "
            f"({pct(two, total_rows)}%)."
        )
    shares = sorted(larger_share(i, "one") for i in instances)
    mid = len(shares) // 2
    median = (
        shares[mid]
        if len(shares) % 2
        else 0.5 * (shares[mid - 1] + shares[mid])
    )
    print(
        f"Median over the instances of the larger single-infinity share: {100 * median:.2f}% "
        "of the rows."
    )

    rows = []
    for i in instances:
        o, p = i["o"], i["p"]
        if not p:
            continue
        t = p["rows_by_infinite_activity_terms"]
        mn, mx, n = t["min_activity"], t["max_activity"], p["rows"]
        ob, pb = o["bounds"], p["bounds"]
        low, high = single_infinity_bounds(p)
        rows.append(
            [
                i["band"],
                i["name"],
                i["presolve"]["status"],
                num(n),
                pct(mn["one"], n),
                pct(mn["two_or_more"], n),
                pct(mx["one"], n),
                pct(mx["two_or_more"], n),
                f"{pct(low, n)} to {pct(high, n)}",
                f"{o['cols']:,} -> {p['cols']:,}",
                f"{ob['upper_infinite']:,} -> {pb['upper_infinite']:,}",
                f"{ob['lower_infinite']:,} -> {pb['lower_infinite']:,}",
                f"{ob['free_cols']:,} -> {pb['free_cols']:,}",
            ]
        )
    table(
        "2d. After stock HiGHS presolve (presolved models): shares of the rows, and columns "
        "with infinite bounds before -> after",
        [
            "band",
            "instance",
            "status",
            "rows after",
            "min activity: one %",
            "min activity: two or more %",
            "max activity: one %",
            "max activity: two or more %",
            "a side the row has sees exactly one, % (bounds)",
            "columns",
            "upper infinite columns",
            "lower infinite columns",
            "free columns",
        ],
        rows,
        'rmatr200-p5 is "Not reduced": its presolved model is the original one. A column '
        "with an infinite bound leaves the count either because HiGHS made its bound finite "
        'or because the column was removed; compare with the "columns" column '
        "(supportcase7 and neos-4306827-ravan lose most of their columns, "
        "neos-5114902-kasavu and neos-5116085-kenana keep them). HiGHS makes recorded "
        "implied bounds explicit in HPresolve::prepareProbing, at the start of enumeration "
        "and of probing; table 2f has the same counts for a run in which neither is called.",
    )
    reduced = [
        i
        for i in instances
        if i["p"] and i["presolve"]["status"] != "Not reduced"
    ]

    def any_infinite(model):
        t = model["rows_by_infinite_activity_terms"]
        return sum(
            t[s][w]
            for s in ("min_activity", "max_activity")
            for w in ("one", "two_or_more")
        )

    before = [i for i in reduced if any_infinite(i["o"]) > 0]
    after = [i for i in reduced if any_infinite(i["p"]) > 0]
    print()
    print(
        f"Of the {len(reduced)} reduced instances, {len(before)} have a row with an infinite "
        f"activity term before presolve and {len(after)} after ({names_of(after)})."
    )

    rows = []
    for i in instances:
        o, b = i["o"], i["o"]["bounds"]
        n = o["cols"]
        rows.append(
            [
                i["band"],
                i["name"],
                num(n),
                num(b["free_cols"]),
                pct(b["free_cols"], n),
                num(b["lower_infinite"]),
                pct(b["lower_infinite"], n),
                num(b["upper_infinite"]),
                pct(b["upper_infinite"], n),
                num(b["lower_huge"]),
                num(b["upper_huge"]),
                num(b["cols_with_huge_bound"]),
                num(b["rows_with_huge_bound_col"]),
            ]
        )
    table(
        "2e. Column bounds (original models)",
        [
            "band",
            "instance",
            "columns",
            "free",
            "%",
            "lower infinite",
            "%",
            "upper infinite",
            "%",
            "huge lower",
            "huge upper",
            "columns with a huge bound",
            "rows with such a column",
        ],
        rows,
        "huge = finite with |bound| >= 1e8. free = both bounds infinite (free columns are "
        "counted in both infinite columns as well).",
    )

    free = [i for i in instances if i["o"]["bounds"]["free_cols"] > 0]
    upper = [i for i in instances if i["o"]["bounds"]["upper_infinite"] > 0]
    upper10 = [
        i
        for i in instances
        if i["o"]["bounds"]["upper_infinite"] >= 0.1 * i["o"]["cols"]
    ]
    lower = [i for i in instances if i["o"]["bounds"]["lower_infinite"] > 0]
    huge = [
        i for i in instances if i["o"]["bounds"]["cols_with_huge_bound"] > 0
    ]
    print()
    print(
        f"Instances with free columns: {len(free)} ({names_of(free)}). With an infinite upper "
        f"bound: {len(upper)}; on at least 10% of the columns: {len(upper10)} "
        f"({names_of(upper10)}). With an infinite lower bound: {len(lower)} "
        f"({names_of(lower)}). With a huge finite bound: {len(huge)} ({names_of(huge)})."
    )

    def infinite_rows(model):
        t = model["rows_by_infinite_activity_terms"]
        return max(
            t[s]["one"] + t[s]["two_or_more"]
            for s in ("min_activity", "max_activity")
        )

    rows = []
    stage1_kept = []
    stage1_reduced = []
    for i in instances:
        s1 = i["stage1"]
        if not s1 or not s1.get("presolved") or not i["p"]:
            continue
        o, p, q = i["o"], i["p"], s1["presolved"]
        if s1["presolve"]["status"] != "Not reduced":
            stage1_reduced.append(i)
            if any_infinite(q) > 0:
                stage1_kept.append(i)
        if o["bounds"]["upper_infinite"] + o["bounds"]["lower_infinite"] == 0:
            continue
        rows.append(
            [
                i["band"],
                i["name"],
                num(o["bounds"]["upper_infinite"]),
                num(p["bounds"]["upper_infinite"]),
                num(q["bounds"]["upper_infinite"]),
                num(o["cols"]),
                num(p["cols"]),
                num(q["cols"]),
                f"{infinite_rows(o):,} ({pct(infinite_rows(o), o['rows'])}%)",
                f"{infinite_rows(p):,} ({pct(infinite_rows(p), p['rows'])}%)",
                f"{infinite_rows(q):,} ({pct(infinite_rows(q), q['rows'])}%)",
                s1["presolve"]["status"],
            ]
        )
    if rows:
        table(
            "2f. Infinite bounds with and without HiGHS's prepareProbing (stock run against "
            "the diag_stage1 run, in which enumeration and probing are switched off)",
            [
                "band",
                "instance",
                "upper infinite columns: original",
                "stock presolved",
                "stage-1-only presolved",
                "columns: original",
                "stock presolved",
                "stage-1-only presolved",
                "rows with an infinite term: original",
                "stock presolved",
                "stage-1-only presolved",
                "stage-1-only status",
            ],
            rows,
            "rows with an infinite term = the larger of the two activities' counts of rows "
            "with one or more infinite terms. diag_stage1 = patched HiGHS with "
            "HIGHS_DIAG_OFF=probing, enumeration, clique_substitutions, clique_merging, "
            "implied_integers, aggregator, sparsify, fourier_motzkin, strengthen_inequalities, "
            "dominated_columns, parallel_columns, so prepareProbing is never called. Other "
            "rules are off too, so the difference is not prepareProbing alone. Instances "
            "without an infinite column bound in the original model are left out.",
        )
        print()
        print(
            f"Of the {len(stage1_reduced)} instances the stage-1-only run reduces, "
            f"{len(stage1_kept)} keep a row with an infinite activity term "
            f"({names_of(stage1_kept)}), against {len(after)} of {len(reduced)} in the stock "
            "run."
        )


# --------------------------------------------------------------------------------------------
# 3. Row classes
# --------------------------------------------------------------------------------------------


def clique_rows(model):
    c = model["row_classes"]
    return c["set_packing"] + c["set_partitioning"]


def section_classes(instances):
    print()
    print("## 3. Row classes")

    rows = []
    for i in instances:
        c = i["o"]["row_classes"]
        rows.append(
            [i["band"], i["name"], num(i["o"]["rows"])]
            + [num(c[k]) for k in ROW_CLASSES]
            + [num(i["o"]["two_binary_rows"])]
        )
    table(
        "3a. Row classes of the original models (rows; every row in exactly one class)",
        ["band", "instance", "rows"] + ROW_CLASS_SHORT + ["two-binary rows"],
        rows,
        "two-binary rows overlap the classes (two entries, both binary, any sides).",
    )

    rows = []
    for i in instances:
        o, c, n = i["o"], i["o"]["row_classes"], i["o"]["rows"]
        trivial = c["empty"] + c["singleton"] + c["doubleton_equation"]
        rows.append(
            [
                i["band"],
                i["name"],
                pct(clique_rows(o), n),
                pct(c["set_covering"], n),
                pct(c["cardinality"], n),
                pct(c["binary_knapsack_non_unit"], n),
                pct(c["variable_bound"], n),
                pct(o["two_binary_rows"], n),
                pct(c["all_integer"], n),
                pct(c["mixed"], n),
                pct(c["all_continuous"], n),
                pct(trivial, n),
            ]
        )
    table(
        "3b. Shares of the rows, original models (%)",
        [
            "band",
            "instance",
            "clique rows (packing + partitioning)",
            "covering",
            "cardinality",
            "knapsack",
            "variable bound",
            "two-binary",
            "all integer",
            "mixed",
            "all continuous",
            "empty + singleton + doubleton equation",
        ],
        rows,
    )

    rows = []
    for i in instances:
        if not i["p"]:
            continue
        o, p = i["o"], i["p"]
        c = p["row_classes"]
        n = p["rows"]
        rows.append(
            [
                i["band"],
                i["name"],
                i["presolve"]["status"],
                num(n),
                num(c["set_packing"]),
                num(c["set_partitioning"]),
                num(c["set_covering"]),
                num(c["cardinality"]),
                num(c["binary_knapsack_non_unit"]),
                num(c["variable_bound"]),
                num(p["two_binary_rows"]),
                num(c["all_integer"]),
                num(c["mixed"]),
                num(c["all_continuous"]),
                f"{pct(clique_rows(o), o['rows'])} -> {pct(clique_rows(p), n)}",
                f"{pct(o['row_classes']['variable_bound'], o['rows'])} -> "
                f"{pct(c['variable_bound'], n)}",
                f"{pct(o['row_classes']['binary_knapsack_non_unit'], o['rows'])} -> "
                f"{pct(c['binary_knapsack_non_unit'], n)}",
            ]
        )
    table(
        "3c. Row classes after stock HiGHS presolve (rows), and the shares before -> after (%)",
        [
            "band",
            "instance",
            "status",
            "rows after",
            "pack",
            "part",
            "cover",
            "card",
            "knap",
            "var bnd",
            "two-binary",
            "all int",
            "mixed",
            "all cont",
            "clique rows %",
            "variable bound %",
            "knapsack %",
        ],
        rows,
        "In the presolved model an implied integer column with bounds 0 and 1 counts as "
        "binary. HiGHS's clique merging has already extended and deleted clique rows in it.",
    )

    rows = []
    for i in instances:
        d = i["diag"]
        if not d:
            continue
        o, p = i["o"], i["p"]
        probing = d["probing"][0]
        merge = d["cliquemerge"][0]
        calls = d.get("cliquemerge_call", [])
        first = calls[0] if calls else None
        vb = d["varbound_aggregation"]
        vb_removed = (
            vb["enumeration"]["rows_removed"] + vb["probing"]["rows_removed"]
        )
        vb_modified = (
            vb["enumeration"]["rows_modified"] + vb["probing"]["rows_modified"]
        )
        vb_lifted = (
            vb["enumeration"]["vars_lifted"] + vb["probing"]["vars_lifted"]
        )
        rows.append(
            [
                i["band"],
                i["name"],
                num(clique_rows(o)),
                num(o["row_classes"]["binary_knapsack_non_unit"]),
                num(clique_rows(p)) if p else "-",
                f"{first['cliques_before']:,} ({first['stage']})"
                if first
                else "-",
                num(probing["cliques_before"]),
                num(probing["cliques_after"]),
                probing["stop"],
                num(merge["calls"]),
                num(merge["rows_changed"]),
                num(merge["rows_deleted"]),
                num(merge["nnz_added"]),
                num(merge["nnz_removed"]),
                num(merge["fixings"]),
                pct(
                    merge["rows_changed"] + merge["rows_deleted"], o["rows"], 2
                ),
                num(o["row_classes"]["variable_bound"]),
                num(vb_removed),
                num(vb_modified),
                num(vb_lifted),
            ]
        )
    table(
        "3d. What HiGHS did with cliques and variable-bound rows "
        "(patched run, diag/summary.json)",
        [
            "band",
            "instance",
            "clique rows, original",
            "knapsack rows, original",
            "clique rows, presolved",
            "clique table at the first merge call (stage)",
            "clique table: probing start",
            "clique table: probing end",
            "probing stop",
            "merge calls",
            "merge: rows extended",
            "merge: rows deleted",
            "merge: nonzeros added",
            "merge: nonzeros removed",
            "merge: fixings",
            "merge: rows extended + deleted as % of original rows",
            "variable-bound rows, original",
            "vb aggregation: rows removed",
            "vb aggregation: rows rewritten",
            "vb aggregation: binaries lifted",
        ],
        rows,
        "clique table = HighsCliqueTable::numCliques(). probing start / end = diag.probing "
        "cliques_before (first round, start of the loop) / cliques_after (last round, end of "
        "the loop, before merging). merge = diag.cliquemerge, summed over the calls in "
        "enumeration and probing. vb aggregation = diag.varbound_aggregation, enumeration plus "
        "probing.",
    )

    diag_instances = [i for i in instances if i["diag"]]

    def merge_of(i):
        return i["diag"]["cliquemerge"][0]

    def touched(i):
        return merge_of(i)["rows_changed"] + merge_of(i)["rows_deleted"]

    changed = [i for i in diag_instances if touched(i) > 0]
    extended = [i for i in diag_instances if merge_of(i)["rows_changed"] > 0]
    deleted = [i for i in diag_instances if merge_of(i)["rows_deleted"] > 0]
    big = [i for i in diag_instances if touched(i) >= 0.01 * i["o"]["rows"]]
    fix = [i for i in diag_instances if merge_of(i)["fixings"] > 0]
    vbagg = [
        i
        for i in diag_instances
        if sum(
            i["diag"]["varbound_aggregation"][s][k]
            for s in ("enumeration", "probing")
            for k in ("rows_removed", "rows_modified")
        )
        > 0
    ]
    grew = [
        i
        for i in diag_instances
        if i["diag"]["probing"][0]["cliques_after"]
        > i["diag"]["probing"][0]["cliques_before"]
    ]
    grew10 = [
        i
        for i in diag_instances
        if i["diag"]["probing"][0]["cliques_after"]
        >= 10 * max(i["diag"]["probing"][0]["cliques_before"], 1)
    ]
    over_rows = [
        i
        for i in diag_instances
        if i["diag"]["probing"][0]["cliques_after"] > i["o"]["rows"]
    ]
    empty_table = [
        i
        for i in diag_instances
        if i["diag"]["probing"][0]["cliques_after"] == 0
    ]
    no_class = [i for i in diag_instances if clique_rows(i["o"]) == 0]
    no_class_but_table = [
        i for i in no_class if i["diag"]["probing"][0]["cliques_before"] > 0
    ]
    many = [
        i
        for i in diag_instances
        if clique_rows(i["o"]) >= 0.1 * i["o"]["rows"]
    ]
    many_changed = [i for i in many if touched(i) > 0]
    few_changed = [
        i for i in diag_instances if i not in many and touched(i) > 0
    ]
    print()
    print(f"Of the {len(diag_instances)} instances with a patched run:")
    print(
        f"- clique merging extended or deleted a row on {len(changed)} ({names_of(changed)}); "
        f"extended on {len(extended)}, deleted on {len(deleted)};"
    )
    print(
        f"- it touched at least 1% of the original rows on {len(big)} ({names_of(big)});"
    )
    print(
        f"- of the {len(many)} instances with at least 10% clique rows in the original model "
        f"it changed rows on {len(many_changed)} (not on "
        f"{names_of([i for i in many if i not in many_changed])}); of the other "
        f"{len(diag_instances) - len(many)} on {len(few_changed)} ({names_of(few_changed)});"
    )
    print(f"- it fixed columns on {len(fix)} ({names_of(fix)});")
    print(
        f"- variable-bound aggregation removed or rewrote rows on {len(vbagg)} "
        f"({names_of(vbagg)});"
    )
    print(
        f"- the clique table grew during the probing loop on {len(grew)}, by 10 times or more "
        f"on {len(grew10)} ({names_of(grew10)}); at the end of the loop it holds more cliques "
        f"than the original model has rows on {len(over_rows)} ({names_of(over_rows)}); it is "
        f"empty on {len(empty_table)} ({names_of(empty_table)});"
    )
    print(
        f"- {len(no_class)} have no packing or partitioning row in the original model "
        f"({names_of(no_class)}); on {len(no_class_but_table)} of these HiGHS still has "
        f"cliques when probing starts ({names_of(no_class_but_table)})."
    )

    rows = []
    for i in diag_instances:
        merge = merge_of(i)
        total = i["diag"]["total"][0]["presolve_seconds"]
        if merge["seconds"] < 0.01 * total:
            continue
        calls = i["diag"].get("cliquemerge_call", [])
        active = [
            c
            for c in calls
            if c["rows_changed"] + c["rows_deleted"] + c["fixings"] > 0
        ]
        rows.append(
            [
                i["band"],
                i["name"],
                num(merge["seconds"], 4),
                num(total, 4),
                pct(merge["seconds"], total),
                len(calls),
                len(active),
                num(sum(c["seconds"] for c in active), 4),
            ]
        )
    table(
        "3e. Time in HiGHS's clique merging where it is at least 1% of the presolve time "
        "(patched run)",
        [
            "band",
            "instance",
            "merge seconds",
            "presolve seconds",
            "%",
            "merge calls",
            "calls that changed a row or fixed a column",
            "seconds of those calls",
        ],
        rows,
        "merge seconds = diag.cliquemerge seconds (time inside runCliqueMerging; applying the "
        "result is not in it). One run each.",
    )

    rows = []
    for threshold in (0.10, 0.50):
        for label, fn in (
            ("clique rows (packing + partitioning)", clique_rows),
            ("covering", lambda m: m["row_classes"]["set_covering"]),
            (
                "knapsack",
                lambda m: m["row_classes"]["binary_knapsack_non_unit"],
            ),
            ("variable bound", lambda m: m["row_classes"]["variable_bound"]),
            ("two-binary rows", lambda m: m["two_binary_rows"]),
            ("all continuous", lambda m: m["row_classes"]["all_continuous"]),
            ("mixed", lambda m: m["row_classes"]["mixed"]),
        ):
            orig = [
                i
                for i in instances
                if fn(i["o"]) >= threshold * i["o"]["rows"]
            ]
            pres = [
                i
                for i in instances
                if i["p"] and fn(i["p"]) >= threshold * i["p"]["rows"]
            ]
            rows.append(
                [
                    f">= {int(100 * threshold)}%",
                    label,
                    len(orig),
                    names_of(orig),
                    len(pres),
                    names_of(pres),
                ]
            )
    table(
        "3f. Instances in which a class holds a large share of the rows",
        [
            "share of rows",
            "class",
            f"original (of {len(instances)})",
            "which",
            f"presolved (of {sum(1 for i in instances if i['p'])})",
            "which",
        ],
        rows,
    )


# --------------------------------------------------------------------------------------------
# 4. Numerics
# --------------------------------------------------------------------------------------------


def section_numerics(instances):
    print()
    print("## 4. Numerics: coefficient range and row dynamism")

    def matrix_range(model):
        return (
            model["coefficient"]["max_abs"] / model["coefficient"]["min_abs"]
        )

    rows = []
    for i in instances:
        o, p = i["o"], i["p"]
        c, raw, sc = (
            o["coefficient"],
            o["row_dynamism"],
            o["row_dynamism_highs_scaled"],
        )
        row = [
            i["band"],
            i["name"],
            num(c["min_abs"]),
            num(c["max_abs"]),
            num(matrix_range(o)),
            num(raw["p50"]),
            num(raw["p99"]),
            num(raw["max"]),
            num(sc["p50"]),
            num(sc["p99"]),
            num(sc["max"]),
        ]
        if p:
            pd = p["row_dynamism"]
            row += [
                num(matrix_range(p)),
                num(pd["p50"]),
                num(pd["p99"]),
                num(pd["max"]),
            ]
        else:
            row += ["-", "-", "-", "-"]
        rows.append(row)
    table(
        "4a. Coefficient range and row dynamism (max |a| / min |a| per row)",
        [
            "band",
            "instance",
            "min |a|",
            "max |a|",
            "matrix range",
            "raw p50",
            "raw p99",
            "raw max",
            "scaled p50",
            "scaled p99",
            "scaled max",
            "presolved: matrix range",
            "presolved p50",
            "presolved p99",
            "presolved max",
        ],
        rows,
        "raw = original model; scaled = original model with HiGHS's final MIP scaling applied "
        "(row_dynamism_highs_scaled); presolved = the model stock HiGHS returns, which carries "
        "that scaling. Compare presolved with scaled. matrix range = max |a| / min |a| over "
        "the whole matrix; the statistics have it for the raw original and for the presolved "
        "model, not for the scaled original, so its change mixes scaling and presolve. "
        'rmatr200-p5 is "Not reduced": its presolved model is the unscaled original.',
    )

    rows = []
    for i in instances:
        o, p = i["o"], i["p"]
        if not p:
            continue
        sc, pd = o["row_dynamism_highs_scaled"], p["row_dynamism"]
        not_reduced = i["presolve"]["status"] == "Not reduced"
        base = o["row_dynamism"] if not_reduced else sc
        r99, rmax = (
            ratio(pd["p99"], base["p99"]),
            ratio(pd["max"], base["max"]),
        )

        def verdict(r):
            if r is None:
                return "-"
            if r > 1.0 + 1e-9:
                return "worse"
            if r < 1.0 - 1e-9:
                return "better"
            return "same"

        rows.append(
            [
                i["band"],
                i["name"],
                i["presolve"]["status"],
                num(base["p99"]),
                num(pd["p99"]),
                num(r99),
                verdict(r99),
                num(base["max"]),
                num(pd["max"]),
                num(rmax),
                verdict(rmax),
                num(10 * pd["p99"]),
            ]
        )
    table(
        "4b. HiGHS presolve against the scaled original, and the plan's ceiling",
        [
            "band",
            "instance",
            "status",
            "p99 before",
            "p99 after",
            "after / before",
            "p99",
            "max before",
            "max after",
            "after / before",
            "max",
            "our p99 ceiling (10 x HiGHS p99)",
        ],
        rows,
        'before = original with HiGHS\'s scaling (for the "Not reduced" instance: raw, '
        "because its returned model is unscaled). The plan allows our 99th-percentile row "
        "dynamism to be at most 10 times HiGHS's (DESIGN.md section 12).",
    )

    rows = []
    for limit in (1e4, 1e5, 1e6):
        for label, key, stat in (
            ("raw", "o", "row_dynamism"),
            ("scaled", "o", "row_dynamism_highs_scaled"),
            ("presolved", "p", "row_dynamism"),
        ):
            for q in ("max", "p99"):
                which = [
                    i for i in instances if i[key] and i[key][stat][q] > limit
                ]
                rows.append(
                    [
                        f"> {limit:.0e}",
                        "row dynamism",
                        label,
                        q,
                        len(which),
                        ", ".join(
                            f"{i['name']} ({num(i[key][stat][q])})"
                            for i in which
                        )
                        or "none",
                    ]
                )
        for label, key in (("raw", "o"), ("presolved", "p")):
            which = [
                i for i in instances if i[key] and matrix_range(i[key]) > limit
            ]
            rows.append(
                [
                    f"> {limit:.0e}",
                    "matrix range",
                    label,
                    "-",
                    len(which),
                    ", ".join(
                        f"{i['name']} ({num(matrix_range(i[key]))})"
                        for i in which
                    )
                    or "none",
                ]
            )
    table(
        "4c. Instances above a dynamism threshold",
        [
            "threshold",
            "quantity",
            "model",
            "statistic",
            "instances",
            "which (value)",
        ],
        rows,
        f"raw and scaled: {len(instances)} instances; presolved: "
        f"{sum(1 for i in instances if i['p'])}.",
    )

    hard_set = [
        i
        for i in instances
        if i["o"]["row_dynamism"]["max"] > 1e4
        or i["o"]["row_dynamism_highs_scaled"]["max"] > 1e4
        or (i["p"] and i["p"]["row_dynamism"]["max"] > 1e4)
    ]
    rows = []
    for i in hard_set:
        o, p = i["o"], i["p"]
        raw, sc = o["row_dynamism"], o["row_dynamism_highs_scaled"]
        pd = p["row_dynamism"] if p else None
        rows.append(
            [
                i["band"],
                i["name"],
                num(raw["p50"]),
                num(sc["p50"]),
                num(pd["p50"]) if pd else "-",
                num(raw["p99"]),
                num(sc["p99"]),
                num(pd["p99"]) if pd else "-",
                num(raw["max"]),
                num(sc["max"]),
                num(pd["max"]) if pd else "-",
                num(ratio(sc["max"], raw["max"])),
                num(ratio(pd["max"], sc["max"])) if pd else "-",
            ]
        )
    table(
        "4d. Instances with a maximum row dynamism above 1e4 in any of the three models",
        [
            "band",
            "instance",
            "p50 raw",
            "p50 scaled",
            "p50 presolved",
            "p99 raw",
            "p99 scaled",
            "p99 presolved",
            "max raw",
            "max scaled",
            "max presolved",
            "max scaled / max raw",
            "max presolved / max scaled",
        ],
        rows,
    )

    with_p = [
        i
        for i in instances
        if i["p"] and i["presolve"]["status"] != "Not reduced"
    ]

    def cmp(i, q, factor):
        return i["p"]["row_dynamism"][q], i["o"]["row_dynamism_highs_scaled"][
            q
        ] * factor

    worse99 = [
        i
        for i in with_p
        if cmp(i, "p99", 1 + 1e-9)[0] > cmp(i, "p99", 1 + 1e-9)[1]
    ]
    better99 = [
        i
        for i in with_p
        if cmp(i, "p99", 1 - 1e-9)[0] < cmp(i, "p99", 1 - 1e-9)[1]
    ]
    worse_max = [
        i
        for i in with_p
        if cmp(i, "max", 1 + 1e-9)[0] > cmp(i, "max", 1 + 1e-9)[1]
    ]
    better_max = [
        i
        for i in with_p
        if cmp(i, "max", 1 - 1e-9)[0] < cmp(i, "max", 1 - 1e-9)[1]
    ]
    worse10 = [
        i for i in with_p if cmp(i, "p99", 10)[0] > cmp(i, "p99", 10)[1]
    ]
    print()
    print(
        f"Of the {len(with_p)} reduced instances, presolved against scaled original: p99 worse "
        f"on {len(worse99)} ({names_of(worse99)}), better on {len(better99)} "
        f"({names_of(better99)}); max worse on {len(worse_max)} ({names_of(worse_max)}), "
        f"better on {len(better_max)} ({names_of(better_max)}). HiGHS's own p99 grows by more "
        f"than 10 times on {len(worse10)} ({names_of(worse10)})."
    )
    scaled_differs = [
        i
        for i in instances
        if i["o"]["row_dynamism_highs_scaled"] != i["o"]["row_dynamism"]
    ]
    print(
        f"HiGHS's scaling changes a row-dynamism statistic (p50, p99 or max) of the original "
        f"model on {len(scaled_differs)} instances ({names_of(scaled_differs)})."
    )
    unit = [
        i for i in instances if i["p"] and i["p"]["row_dynamism"]["p99"] <= 1.0
    ]
    print(
        f"HiGHS's presolved p99 is 1 on {len(unit)} instances ({names_of(unit)}): the ceiling "
        "there is a p99 of 10."
    )
    all_unit = [i for i in instances if i["o"]["row_dynamism"]["max"] <= 1.0]
    print(
        f"Every row of the original model has dynamism 1 on {len(all_unit)} instances "
        f"({names_of(all_unit)})."
    )


# --------------------------------------------------------------------------------------------
# 5. Variable types
# --------------------------------------------------------------------------------------------


def section_types(instances):
    print()
    print("## 5. Variable types before and after HiGHS presolve")

    rows = []
    for i in instances:
        o, p, d = i["o"], i["p"], i["diag"]
        t = o["col_types"]
        n = o["cols"]
        integer = t["binary"] + t["general_integer"]
        row = [
            i["band"],
            i["name"],
            num(n),
            num(t["binary"]),
            num(t["general_integer"]),
            num(t["continuous"]),
            pct(integer, n),
        ]
        if p:
            pt, pn = p["col_types"], p["cols"]
            pint = pt["binary"] + pt["general_integer"]
            row += [
                num(pn),
                num(pt["binary"]),
                num(pt["general_integer"]),
                num(pt["implied_integer"]),
                num(pt["continuous"]),
                pct(pint, pn),
                pct(pt["implied_integer"], pn),
                pct(
                    pt["implied_integer"],
                    pt["implied_integer"] + pt["continuous"],
                ),
            ]
        else:
            row += ["-"] * 8
        if d:
            total = d["total"][0]
            before, after = (
                int(x) for x in str(total["integer_cols"]).split(":")
            )
            row += [
                f"{before:,} -> {after:,}",
                num(total["implied_integer_cols"]),
            ]
        else:
            row += ["-", "-"]
        rows.append(row)
    table(
        "5a. Column types",
        [
            "band",
            "instance",
            "columns",
            "binary",
            "general integer",
            "continuous",
            "integer %",
            "columns after",
            "binary",
            "general integer",
            "implied integer",
            "continuous",
            "integer % after",
            "implied integer % after",
            "implied integer as % of non-integer columns after",
            "integer columns before -> after (DIAG total)",
            "implied integers (DIAG total)",
        ],
        rows,
        "integer % = binary + general integer over all columns. The columns from the "
        "presolved model are those of stock/<name>.json; the last two columns are HiGHS's "
        'internal counts on the patched run. For the "Not reduced" instance the presolved '
        "model is the original, so implied integers would show only in the DIAG count.",
    )

    semi = [i for i in instances if i["o"]["col_types"]["semi"] > 0]
    print()
    print(
        f"Instances with semi-continuous or semi-integer columns: {len(semi)}."
    )

    def share(i):
        t = i["o"]["col_types"]
        return (t["binary"] + t["general_integer"]) / i["o"]["cols"]

    groups = [
        ("no integer column", lambda s: s == 0),
        ("below 10%", lambda s: 0 < s < 0.1),
        ("10% to below 50%", lambda s: 0.1 <= s < 0.5),
        ("50% to below 100%", lambda s: 0.5 <= s < 1),
        ("all integer", lambda s: s == 1),
    ]
    rows = []
    for label, test in groups:
        which = [i for i in instances if test(share(i))]
        rows.append([label, len(which), names_of(which)])
    table(
        f"5b. Instances (of {len(instances)}) by the share of integer columns, original models",
        ["integer share", "instances", "which"],
        rows,
    )
    general = [
        i for i in instances if i["o"]["col_types"]["general_integer"] > 0
    ]
    general10 = [
        i
        for i in instances
        if i["o"]["col_types"]["general_integer"] >= 0.1 * i["o"]["cols"]
    ]
    print()
    print(
        f"Instances with general integer columns: {len(general)}; on at least 10% of the "
        f"columns: {len(general10)} ({names_of(general10)})."
    )

    with_diag = [i for i in instances if i["diag"]]
    found = [
        i
        for i in with_diag
        if i["diag"]["total"][0]["implied_integer_cols"] > 0
    ]
    rows = []
    for i in found:
        total = i["diag"]["total"][0]
        o, p = i["o"], i["p"]
        cont = o["col_types"]["continuous"]
        s1 = i["stage1_diag"]
        s1_count = (
            s1["total"][0]["implied_integer_cols"]
            if s1 and s1.get("total")
            else None
        )
        rows.append(
            [
                i["band"],
                i["name"],
                num(total["implied_integer_cols"]),
                num(cont),
                pct(total["implied_integer_cols"], cont),
                num(o["col_types"]["general_integer"]),
                num(p["col_types"]["general_integer"]) if p else "-",
                num(p["col_types"]["implied_integer"]) if p else "-",
                num(p["col_types"]["continuous"]) if p else "-",
                num(s1_count),
            ]
        )
    table(
        f"5c. Implied integer columns in HiGHS's presolved model ({len(found)} of "
        f"{len(with_diag)} instances)",
        [
            "band",
            "instance",
            "implied integer columns (DIAG total)",
            "continuous columns, original",
            "implied as % of original continuous",
            "general integer, original",
            "general integer, presolved",
            "implied integer, presolved model",
            "continuous, presolved",
            "implied integer columns, stage-1-only run (DIAG total)",
        ],
        rows,
        "The DIAG total is the number of implied integer columns left at the end of presolve; "
        "a column that was converted and then removed is not in it, so the number of "
        "conversions is not known. An implied integer can come from a continuous column or "
        "from a general integer column that HiGHS re-types; the files do not say which, so "
        "the percentage is only meaningful where the general-integer count did not drop. "
        "Last column: the diag_stage1 run, in which the implied-integer sweep, probing and "
        "the Stage 2 to 4 rules are switched off; the conversions of column presolve inside "
        'the fast loop stay on (HIGHS_DIAG.md, "Not measured").',
    )
    mismatch = [
        i
        for i in with_diag
        if i["p"]
        and i["presolve"]["status"] != "Not reduced"
        and i["p"]["col_types"]["implied_integer"]
        != i["diag"]["total"][0]["implied_integer_cols"]
    ]
    with_cont = [i for i in with_diag if i["o"]["col_types"]["continuous"] > 0]
    print()
    print(
        f"{len(with_cont)} of the {len(with_diag)} instances have continuous columns; HiGHS's "
        f"presolved model has implied integer columns on {len(found)}, more than 100 on "
        f"{sum(1 for i in found if i['diag']['total'][0]['implied_integer_cols'] > 100)}."
    )
    s1_found = [
        i
        for i in with_diag
        if i["stage1_diag"]
        and i["stage1_diag"].get("total")
        and i["stage1_diag"]["total"][0]["implied_integer_cols"] > 0
    ]
    if any(i["stage1_diag"] for i in with_diag):
        print(
            f"In the stage-1-only run (implied-integer sweep off) the presolved model has "
            f"implied integer columns on {len(s1_found)} ({names_of(s1_found)})."
        )
    print(
        "Stock presolved model and DIAG total disagree on the implied-integer count on "
        f"{len(mismatch)} reduced instances ({names_of(mismatch)})."
    )


# --------------------------------------------------------------------------------------------
# 6. Bands A1 and A2
# --------------------------------------------------------------------------------------------


def section_band_a(instances, costs):
    print()
    print("## 6. The 50,000-nonzero end: bands A1 and A2")
    band_a = [i for i in instances if i["band"] in ("A1", "A2")]

    rows = []
    for i in band_a:
        o = i["o"]
        r, c, t = o["row_length"], o["col_length"], o["col_types"]
        cls = o["row_classes"]
        short = (
            cls["empty"]
            + cls["singleton"]
            + cls["doubleton_equation"]
            + cls["variable_bound"]
            + cls["other_two_entry"]
        )
        rows.append(
            [
                i["band"],
                i["name"],
                num(o["rows"]),
                num(o["cols"]),
                num(o["nnz"]),
                num(r["p50"]),
                num(r["p90"]),
                num(r["p99"]),
                num(r["max"]),
                num(r["mean"]),
                num(c["p50"]),
                num(c["p90"]),
                num(c["p99"]),
                num(c["max"]),
                num(c["mean"]),
                pct(t["binary"] + t["general_integer"], o["cols"]),
                pct(short, o["rows"]),
            ]
        )
    table(
        "6a. Shapes of the band A1 and A2 models",
        [
            "band",
            "instance",
            "rows",
            "columns",
            "nonzeros",
            "row p50",
            "row p90",
            "row p99",
            "row max",
            "row mean",
            "col p50",
            "col p90",
            "col p99",
            "col max",
            "col mean",
            "integer %",
            "rows with at most two entries, % (class count)",
        ],
        rows,
        "Last column: empty, singleton, doubleton-equation, variable-bound and other "
        "two-entry rows; two-entry rows over two binaries are in the binary classes and not "
        "counted here.",
    )

    launch = costs["empty_kernel_launch_sync"]
    readback = costs["readback_pinned"]
    readback_p90 = costs["readback_pinned_p90"]
    # The readback row of fixed_costs is "kernel writes 8 bytes, copy to host, synchronise"
    # (README_fixed_costs.md): it already holds one kernel launch, so it is the cost of one
    # round of "one kernel, then the host reads a flag". Adding a launch to it counts the
    # kernel twice.
    round_cost = readback
    lib = costs["library"]
    reduce_us = lib["thrust::reduce<double>"]["1e5"]
    copy_if_us = lib["thrust::copy_if<u64>"]["1e5"]
    sort_us = lib["thrust::sort<u64>"]["1e5"]
    print()
    print(
        "Fixed costs used (fixed_costs.json, medians in microseconds): empty kernel launch + "
        f"synchronise {launch:.2f}; launch without synchronise "
        f"{costs['empty_kernel_launch_no_sync']:.2f}; ten launches + one synchronise "
        f"{costs['empty_kernel_10_launches_one_sync']:.2f}; kernel + 8-byte read-back to "
        f"pinned memory {readback:.2f} (90th percentile {costs['readback_pinned_p90']:.2f}), "
        f"to pageable memory {costs['readback_pageable']:.2f} (90th percentile "
        f"{costs['readback_pageable_p90']:.2f})."
    )
    rows = []
    for name in sorted(lib):
        rows.append(
            [name]
            + [num(lib[name][s], 4) for s in ("1e3", "1e4", "1e5", "1e6")]
        )
    table(
        "6b. Library primitives with the rmm pool, median microseconds (fixed_costs.json)",
        ["primitive", "1e3", "1e4", "1e5", "1e6"],
        rows,
    )

    rows = []
    for i in band_a:
        o = i["o"]
        core = i["presolve"]["core_seconds"]
        budget = core * 1e6 * 0.01
        matrix_bytes = (
            2 * o["nnz"] * 12 + 8 * (o["rows"] + 1) + 8 * (o["cols"] + 1)
        )
        vector_bytes = 16 * o["rows"] + 25 * o["cols"]
        upload = interpolate(costs["upload"], matrix_bytes + vector_bytes)
        rows.append(
            [
                i["band"],
                i["name"],
                num(core, 4),
                num(budget, 4),
                num(int(budget / launch)),
                num(int(budget / readback)),
                num(int(budget / readback_p90)),
                num(int(budget / reduce_us)),
                num(int(budget / copy_if_us)),
                num(int(budget / sort_us)),
                num((matrix_bytes + vector_bytes) / 2**20),
                num(upload),
                pct(upload, core * 1e6, 3),
            ]
        )
    table(
        "6c. What fits into 1% of HiGHS's presolve time (stock, presolve.core_seconds)",
        [
            "band",
            "instance",
            "HiGHS seconds",
            "1% in microseconds",
            "kernel launches with sync",
            "rounds of one kernel + one 8-byte read-back (median cost)",
            "the same at the 90th-percentile cost",
            "reduce calls (1e5)",
            "copy_if calls (1e5)",
            "sort calls (1e5)",
            "model on the device, MB",
            "one upload, microseconds (estimate)",
            "upload as % of HiGHS time",
        ],
        rows,
        f"empty kernel launch with synchronise {launch:.2f}; one kernel plus an 8-byte "
        f"read-back {readback:.2f} (90th percentile {readback_p90:.2f}); reduce "
        f"{reduce_us:.1f}, copy_if {copy_if_us:.1f}, sort {sort_us:.1f} microseconds. The "
        'read-back figure is the whole interval "kernel writes 8 bytes, copy to host, '
        'synchronise", so it includes its kernel launch. Model on '
        "the device = CSR and CSC with values (12 bytes per nonzero each, 64-bit offsets), row "
        "sides, column bounds, cost and type. Upload = pinned host-to-device copy time "
        "interpolated linearly between the measured sizes of the handover_copy group; an "
        "estimate, not a measurement of this model.",
    )

    rows = []
    for i in band_a:
        d = i["diag"]
        if not d:
            continue
        st = d["stage"]
        total = d["total"][0]["presolve_seconds"]
        cascade = st["initial_scan"]["seconds"] + st["fast_loop"]["seconds"]
        wide = st["probing"]["seconds"] + st["enumeration"]["seconds"]
        rows.append(
            [
                i["band"],
                i["name"],
                num(total, 4),
                num(cascade * 1000),
                pct(cascade, total),
                num(cascade * 1e9 / i["o"]["nnz"]),
                num(int(cascade * 1e4 / launch)),
                num(st["enumeration"]["seconds"] * 1000),
                pct(st["enumeration"]["seconds"], total),
                num(st["probing"]["seconds"] * 1000),
                pct(st["probing"]["seconds"], total),
                pct(wide, total),
                num(int(wide * 1e4 / launch)),
            ]
        )
    table(
        "6d. Where HiGHS spends the time on band A1 and A2 (patched run, DIAG stage lines)",
        [
            "band",
            "instance",
            "presolve seconds",
            "initial scan + fast loop, ms",
            "%",
            "ns per original nonzero",
            "launches in 1% of it",
            "enumeration, ms",
            "%",
            "probing, ms",
            "%",
            "enumeration + probing %",
            "launches in 1% of enumeration + probing",
        ],
        rows,
        "Times are those of the patched run on the dev node (one run each).",
    )

    rows = []
    for i in band_a:
        d = i["diag"]
        if not d:
            continue
        pr = d["probing"][0]
        probed, loop, sweeps = (
            pr["probed"],
            pr["loop_seconds"],
            pr["prop_sweeps"],
        )
        rows.append(
            [
                i["band"],
                i["name"],
                num(pr["binaries"]),
                num(probed),
                pr["rounds"],
                pr["stop"],
                num(loop * 1000, 4),
                num(loop * 1e6 / probed) if probed else "-",
                num(sweeps / probed) if probed else "-",
                num(pr["max_probe_sweeps"]),
                num(pr["probing_work"] / probed) if probed else "-",
                num(pr["max_probe_work"]),
                num(pr["probing_work"] / sweeps) if sweeps else "-",
                num(loop * 1e6 / sweeps) if sweeps else "-",
                num(pr["probing_work"] / loop / 1e6) if loop else "-",
            ]
        )
    table(
        "6e. HiGHS's probing loop on band A1 and A2 (patched run, diag.probing)",
        [
            "band",
            "instance",
            "candidates (first round)",
            "probed",
            "runProbing calls",
            "stop",
            "loop, ms",
            "microseconds per probe",
            "sweeps per probe, mean",
            "sweeps of one probe, max",
            "entries per probe, mean",
            "entries of one probe, max",
            "entries per sweep, mean",
            "microseconds per sweep",
            "million entries per second",
        ],
        rows,
        "probe = one binary, both values. sweeps = propagation rounds (prop_sweeps); entries = "
        "probing_work (row entries scanned by propagation plus column entries visited for "
        "activity updates). The loop time also holds HiGHS's clique-table and implication "
        "work, so the per-sweep and per-entry figures are upper readings of its propagation "
        "cost.",
    )

    rows = []
    for i in band_a:
        d = i["diag"]
        if not d:
            continue
        pr = d["probing"][0]
        probed, loop, sweeps = (
            pr["probed"],
            pr["loop_seconds"],
            pr["prop_sweeps"],
        )
        if not probed or not loop or not sweeps:
            rows.append([i["band"], i["name"]] + ["-"] * 7)
            continue
        us_per_probe = loop * 1e6 / probed
        us_per_sweep = loop * 1e6 / sweeps
        fixed_max = pr["max_probe_sweeps"] * round_cost
        rows.append(
            [
                i["band"],
                i["name"],
                num(us_per_sweep),
                num(round_cost / us_per_sweep),
                num(us_per_probe),
                num(pr["max_probe_sweeps"]),
                num(fixed_max),
                num(fixed_max / us_per_probe),
                num(fixed_max / 1000.0),
                pct(fixed_max, loop * 1e6, 2),
            ]
        )
    table(
        "6f. Probing: fixed cost of synchronised GPU rounds against HiGHS's loop time",
        [
            "band",
            "instance",
            "HiGHS microseconds per sweep",
            "probes that must share a round for its fixed cost to equal HiGHS's sweep time",
            "HiGHS microseconds per probe",
            "rounds of the deepest probe",
            "their fixed cost, microseconds",
            "as a multiple of HiGHS's time per probe",
            "all probes in one batch: fixed cost, ms",
            "as % of HiGHS's loop time",
        ],
        rows,
        f"Arithmetic, not a measurement: one GPU round is charged {round_cost:.2f} "
        "microseconds (one kernel, one 8-byte read-back and one synchronise: the median of "
        f"the readback row of fixed_costs; its 90th percentile is {readback_p90:.2f}). That "
        "is a floor: it is the least a round costs when the host decides whether another "
        "round follows, and a real round has several kernels and a frontier compaction (table "
        "6g has the measured frontier passes on the band A matrices). A batch of probes "
        "needs as many rounds as its deepest member has sweeps "
        "(max_probe_sweeps counts both values of the binary, so this is an upper count); the "
        "kernels' own work is charged nothing, and HiGHS's probes run one after another and "
        "use each other's results, which a batch does not. A value above 1 in the fourth or "
        "eighth column is the number of probes a round or a batch must hold before this "
        "floor alone drops below HiGHS's time for the same probes.",
    )
    section_band_a_measured(band_a)


def section_band_a_measured(band_a):
    with_sweeps = [i for i in band_a if i["sweeps"]]
    if not with_sweeps:
        return
    rows = []
    for i in with_sweeps:
        gpu_round = sweep_ms_prefix(i, "round", "best plain")
        cpu_act = sweep_ms(i, "cpu", "activity, full, plain")
        cpu_cand = sweep_ms(i, "cpu", "candidates by column, full, plain")
        if None in (gpu_round, cpu_act, cpu_cand):
            continue
        row = [
            i["band"],
            i["name"],
            num(gpu_round * 1000),
            num((cpu_act + cpu_cand) * 1000),
            num((cpu_act + cpu_cand) / gpu_round),
        ]
        for share in ("10%", "1%", "0.1%"):
            gpu = cpu_warm = cpu_cold = None
            for r in i["sweeps"]["results"].get("frontier", []):
                if r["name"].startswith(share + " (") and r["name"].endswith(
                    "compaction + H/S/A/plain"
                ):
                    gpu = r["ms_median"]
                    label = r["name"].split(":")[0]
                    cpu_warm = sweep_ms(
                        i, "cpu", f"activity, {label} warm, plain"
                    )
                    cpu_cold = sweep_ms(
                        i, "cpu", f"activity, {label} cold, plain"
                    )
            row += [
                num(gpu * 1000) if gpu is not None else "-",
                num(cpu_warm * 1000) if cpu_warm is not None else "-",
                num(cpu_cold * 1000) if cpu_cold is not None else "-",
            ]
        if i["presolve"]:
            row.append(
                pct(
                    (cpu_act + cpu_cand) / 1000.0,
                    i["presolve"]["core_seconds"],
                    2,
                )
            )
        else:
            row.append("-")
        rows.append(row)
    table(
        "6g. Measured sweeps on the band A1 and A2 matrices (sweeps/<name>.json, median "
        "microseconds, plain sums)",
        [
            "band",
            "instance",
            "GPU: one full round (activities + candidates)",
            "one CPU thread: activities + candidates by column",
            "CPU / GPU",
            "10% of the rows, activities: GPU",
            "CPU warm",
            "CPU cold",
            "1% of the rows: GPU",
            "CPU warm",
            "CPU cold",
            "0.1% of the rows: GPU",
            "CPU warm",
            "CPU cold",
            "the CPU full round as % of HiGHS's presolve time",
        ],
        rows,
        'GPU full round = the "best plain" row of the round section; CPU = "activity, '
        'full, plain" plus "candidates by column, full, plain". Frontier rows: GPU = flag '
        "compaction plus the H/S/A/plain activity kernel on a random share of the rows; CPU = "
        "the activities of the same rows on one thread, with warm and with evicted caches "
        "(README_sweeps.md sections 4 and 5). One run per matrix.",
    )


def main():
    instances, missing = read_instances()
    ALL_INSTANCES.extend(instances)
    costs = fixed_cost_table()
    print("# Stage 0: structure of the testbed instances")
    print()
    bands = {}
    for i in instances:
        bands.setdefault(i["band"], []).append(i["name"])
    print(
        f"{len(instances)} instances with structure statistics: "
        + "; ".join(f"{b}: {len(n)}" for b, n in bands.items())
        + "."
    )
    print(
        f"With a stock HiGHS result: {sum(1 for i in instances if i['stock'])}; with a "
        f"patched (DIAG) result: {sum(1 for i in instances if i['diag'])}."
    )
    large = [i for i in instances if i["stats_dir"] != "stats"]
    print(
        "From stats_large (statistics only, no HiGHS result used here): "
        + ", ".join(i["name"] for i in large)
        + "."
    )
    print(
        f"Main-list instances without a statistics file ({len(missing)}): "
        + (", ".join(missing) if missing else "none")
        + "."
    )
    system = costs["system"]
    print(
        f"Fixed costs measured on: {system.get('gpu_name', '?')}, CPU "
        f"{system.get('cpu_model', '?')}."
    )

    section_lengths(instances, costs)
    section_infinite(instances)
    section_classes(instances)
    section_numerics(instances)
    section_types(instances)
    section_band_a(instances, costs)


if __name__ == "__main__":
    main()
