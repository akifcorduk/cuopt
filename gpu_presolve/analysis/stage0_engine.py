#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 engine analysis: when is the GPU engine cheaper than the one-thread host engine for
propagation work, and what do the GPU fixed costs mean at the 50,000-nonzero end.

Reads, relative to this file:

    ../testbed/testbed.json                   "main" list (band of every matrix)
    ../results/stage0/sweeps/<name>.json      sweep-kernel micro-benchmark, 32 matrices, H200
    ../results/stage0/fixed_costs.json        GPU fixed costs on the same H200
    ../results/stage0/stock/summary.json      stock HiGHS 1.16.0 presolve (presolve_core_seconds)
    ../results/stage0/diag/summary.json       patched HiGHS (work of the propagation sweeps in probing)
    ../results/stage0/diag_stage1/summary.json  patched HiGHS restricted towards the Stage 1 rule set
                                              (optional; band A table of section 4)

and prints markdown tables:

    1  full sweep: GPU round against the CPU full sweeps per matrix, fits (fixed part plus
       per-nonzero part), crossover size and how far it is extrapolated
    2  frontier sweeps: GPU (compaction plus kernel) against the CPU, warm and cold, per matrix
       and frontier size; crossover in frontier nonzeros and rows; candidate thresholds and the
       time lost at each; the same with the hand-over costs of fixed_costs.json added; a work
       unit with a row term; the GPU's choice between a frontier list and all rows; the
       propagation sweeps of HiGHS's probing for scale
    3  hand-over on x86: refresh of a pinned mirror against gathering the needed rows
    4  GPU fixed cost of a propagation round and the number of rounds that fit into HiGHS's
       presolve time, per instance and per band
    5  cost constants in work units and the numbers behind the recommended defaults
    6  rebuild: GPU against CPU (an engine threshold of DESIGN.md 2.3; not one of the five tasks)

Definitions used throughout (every number is a median of the benchmark's repetitions):

    "GPU round"        results.round "best plain" / "best comp" (activities then candidates, one
                       timed unit; the variants differ between matrices)
    "GPU designed"     activity H/S/A/comp plus the faster of the candidate variants
                       H/S/comp/gated and B/S/comp/gated, timed separately (two synchronisations)
    "CPU by row"       cpu "activity, full" + "candidates by row, full" (what a host engine does)
    "CPU by column"    cpu "activity, full" + "candidates by column, full" (like the device kernel)
    "comp-fma"         compensated sums, CPU built with -mfma; "comp" is the build without it
    "GPU list"         frontier "compaction + H/S/A/comp" (activities of the frontier rows)
    "GPU A"            the cheaper of GPU list and activity H/S/A/comp over all rows without
                       compaction: the GPU engine can always sweep all rows instead
    "CPU A"            cpu "activity, <frontier> warm|cold"
    "CPU S"            the same plus "candidates by row, <frontier> warm|cold" (a host step)
    "GPU step"         no frontier-restricted candidate kernel exists. The model used for a GPU
                       step is 2 x GPU A (assumption: the candidate part costs what the activity
                       part costs); "S hi" replaces it by GPU A + the full-matrix candidate kernel
                       (measured upper bound) and "S lo" by GPU A alone (lower bound)

"Cold" is the CPU measurement with the core's private caches evicted before every repetition,
"warm" the one that repeats the same list. A host engine that has just received its rows through
a copy from the device is cold or worse (the copy does not go through the core's caches); warm
is the lower bound for a host engine that returns to rows it has just touched.

A crossover is the frontier size at which the CPU time equals the GPU time, found by linear
interpolation of log(CPU / GPU) against log(frontier nonzeros) between the measured frontiers
(0.01%, 0.1%, 1%, 10% of the rows, and the full matrix as the 100% point). Below it the host is
cheaper, above it the GPU. When the GPU is already cheaper at the smallest measured frontier the
crossover is extrapolated with a constant GPU time and a CPU time proportional to the frontier
nonzeros, and marked "<". The 100% point is the full CPU sweep, which the benchmark does not
evict: it is the same value for "warm" and "cold".

Scenarios marked "sensitivity" in section 2 vary what the two sides are taken to be (the fastest
split-layout activity kernel instead of H/S/A on the GPU, a host that keeps its activities
incrementally and only computes candidates, other factors in the GPU step model). They show how
far the crossover moves; the thresholds in REC were not chosen from them.

The values in REC below are the defaults proposed in the report. They are inputs, chosen from
the tables this script prints; the tables show what each of them costs on every matrix.

No arguments. Python 3 standard library only. Nothing is written.
"""

import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE0 = HERE.parent / "results" / "stage0"
TESTBED = HERE.parent / "testbed" / "testbed.json"

BANDS = ["A1", "A2", "A3", "B1", "B2", "B3", "C"]

# Assumptions of the hand-over model (section 3), stated in the output too.
BYTES_PER_NNZ = 12  # 4-byte index + 8-byte value of one CSR entry
STATE_BYTES_LOW = (
    16  # per row and per column: bounds and activities, lower figure
)
STATE_BYTES_HIGH = 32  # upper figure
MIB = 1048576.0

# Candidate frontier thresholds examined in section 2: in frontier nonzeros, and in the work
# unit W = nonzeros + kappa * rows.
THRESHOLDS = [500, 1000, 2000, 4000, 8000, 12000, 16000, 32000, 64000]
W_THRESHOLDS = [
    2000,
    4000,
    8000,
    12000,
    16000,
    20000,
    24000,
    32000,
    48000,
    64000,
]
KAPPAS = [0, 2, 4, 8, 16, 32, 64]
# Launch counts of the fixed-cost model of section 4.
LAUNCH_COUNTS = [4, 8, 16]

# The defaults proposed in the report. They are inputs of this script, not results: the tables
# of sections 2, 3, 5 and 6 show what each of them costs on every matrix.
REC = {
    "kappa": 8,  # a frontier row counts as this many nonzeros in the host work estimate
    "unit_ns": 10,  # one work unit in nanoseconds, for converting device constants
    "frontier_work": 20000,  # host below, GPU from here on, in W = nnz + kappa * rows
    "frontier_nnz": 8000,  # the same rule without the row term (frontier nonzeros)
    "device_owned_mirror_work": 8000,  # device owns the state, host has a mirror: host below this W
    "device_owned_gather_work": 4000,  # device owns the state, no mirror: host below this W
    "full_sweep_nnz": 32000,  # full sweep on the GPU from this many matrix nonzeros
    "gpu_list_min_nnz": 500000,  # GPU: frontier list only on matrices above this size, else all rows
    "full_sweep_fraction": 0.5,  # GPU: all rows instead of a list from this share of the nonzeros
    "mirror_bytes": 64
    * 1048576,  # pinned mirror up to this many bytes of matrix + state
    "rebuild_nnz": 100000,  # rebuild on the GPU from this many nonzeros
    "budget_units_per_nnz": 5,  # device fixed-cost budget of a presolve call per model nonzero
}


# ----------------------------------------------------------------------------------------------
# small helpers


def table(header, rows, align=None):
    """Print a markdown table. Cells are converted with str()."""
    align = align or ["l"] + ["r"] * (len(header) - 1)
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if a == "l" else "---:" for a in align) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.{digits}f}"


def count(x):
    return f"{int(round(x)):,}"


def geomean(values):
    values = [v for v in values if v is not None and v > 0]
    return (
        math.exp(sum(math.log(v) for v in values) / len(values))
        if values
        else None
    )


def quantile(values, q):
    """Linear interpolation between order statistics."""
    v = sorted(values)
    if not v:
        return None
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (pos - lo) * (v[hi] - v[lo])


def loglog(xs, ys, x):
    """Piecewise linear interpolation of log y against log x. Below the first point the value of
    the first point is returned (a fixed cost); above the last point y grows in proportion to x.
    """
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1] * x / xs[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (math.log(x) - math.log(xs[i])) / (
                math.log(xs[i + 1]) - math.log(xs[i])
            )
            return math.exp(
                math.log(ys[i]) + t * (math.log(ys[i + 1]) - math.log(ys[i]))
            )
    raise ValueError(x)


def fit_affine(xs, ys, relative=True):
    """Least squares fit y = a + b x. relative=True minimises the relative residuals
    (weights 1 / y^2), so that every matrix counts the same whatever its size.
    """
    w = [1.0 / (y * y) if relative else 1.0 for y in ys]
    sw = sum(w)
    sx = sum(wi * x for wi, x in zip(w, xs))
    sy = sum(wi * y for wi, y in zip(w, ys))
    sxx = sum(wi * x * x for wi, x in zip(w, xs))
    sxy = sum(wi * x * y for wi, x, y in zip(w, xs, ys))
    det = sw * sxx - sx * sx
    b = (sw * sxy - sx * sy) / det
    a = (sy - b * sx) / sw
    return a, b


def crossing(a_gpu, b_gpu, a_cpu, b_cpu):
    """Size at which a_gpu + b_gpu n = a_cpu + b_cpu n; None when the lines do not cross at a
    positive size.
    """
    if b_cpu <= b_gpu:
        return None
    n = (a_gpu - a_cpu) / (b_cpu - b_gpu)
    return n if n > 0 else None


# ----------------------------------------------------------------------------------------------
# loading

FRONTIER_RE = re.compile(r"^([\d.]+)% \((\d+) rows\): (.*)$")
CPU_RE = re.compile(
    r"^(activity|candidates by row|candidates by column), "
    r"(?:full|([\d.]+)% \((\d+) rows\) (warm|cold)), (plain|comp|comp-fma)$"
)


def load_sweeps(bands):
    out = []
    for path in sorted((STAGE0 / "sweeps").glob("*.json")):
        d = json.loads(path.read_text())
        name = path.stem
        m = d["matrix"]
        res = d["results"]
        s = {
            "name": name,
            "band": bands.get(name, "?"),
            "nnz": m["nnz"],
            "rows": m["rows"],
            "cols": m["cols"],
            "max_row_len": m["max_row_len"],
            "max_col_len": m["max_col_len"],
            "gpu": d["gpu"]["name"],
            "quick": d["quick"],
            "verified": d["verification"]["ok"],
            "min_reps": min(
                e["repetitions"] for sec in res.values() for e in sec
            ),
            "activity": {e["name"]: e["ms_median"] for e in res["activity"]},
            "candidate": {e["name"]: e["ms_median"] for e in res["candidate"]},
            "rebuild": {e["name"]: e["ms_median"] for e in res["rebuild"]},
            "round": {},
            "cpu_full": {},
            "frontier": {},
        }
        for e in res["round"]:
            key, variant = e["name"].split(": ", 1)
            s["round"][key] = (
                e["ms_median"],
                variant,
                e["ms_p10"],
                e["ms_p90"],
            )
        for e in res["frontier"]:
            mt = FRONTIER_RE.match(e["name"])
            pct, rows, what = mt.group(1), int(mt.group(2)), mt.group(3)
            f = s["frontier"].setdefault(
                pct, {"pct": pct, "rows": rows, "nnz": e["nnz"], "cpu": {}}
            )
            key = {
                "compaction only": "compaction",
                "compaction + H/S/A/plain": "plain",
                "compaction + H/S/A/comp": "comp",
            }[what]
            f[key] = e["ms_median"]
        for e in res["cpu"]:
            mt = CPU_RE.match(e["name"])
            kind, pct, rows, temp, arith = mt.groups()
            if pct is None:
                s["cpu_full"][(kind, arith)] = e["ms_median"]
            else:
                s["frontier"][pct]["cpu"][(kind, temp, arith)] = e["ms_median"]
        out.append(s)
    out.sort(key=lambda s: s["nnz"])
    return out


def load_fixed():
    d = json.loads((STAGE0 / "fixed_costs.json").read_text())
    fc = {"system": d["system"], "series": {}, "single": {}}
    for r in d["results"]:
        key = (r["group"], r["name"], r.get("variant", ""))
        size = r.get("elements", r.get("bytes"))
        if r["group"] in ("launch", "readback", "unified_memory"):
            fc["single"][key] = r["median_us"]
            continue
        if r["group"] in ("handover_copy", "allocation"):
            size = r["bytes"]
        fc["series"].setdefault(key, []).append((size, r["median_us"]))
    for k in fc["series"]:
        fc["series"][k].sort()
    return fc


class Fixed:
    """Access to fixed_costs.json with interpolation by size."""

    def __init__(self, fc):
        self.fc = fc
        s = fc["single"]
        self.launch_sync = s[("launch", "empty_kernel_launch_sync", "")]
        self.launch_only = s[("launch", "empty_kernel_launch_no_sync", "")]
        self.ten_launches = s[
            ("launch", "empty_kernel_10_launches_one_sync", "")
        ]
        self.readback = s[
            ("readback", "kernel_write_8_bytes_copy_to_host", "pinned")
        ]
        # cost of one more launch in a chain that ends with one synchronisation
        self.per_launch = (self.ten_launches - self.launch_sync) / 9.0
        # what a read-back adds to a launch chain that is synchronised anyway
        self.readback_extra = self.readback - self.launch_sync

    def series(self, group, name, variant=""):
        return self.fc["series"][(group, name, variant)]

    def at(self, group, name, variant, size):
        pts = self.series(group, name, variant)
        return loglog([p[0] for p in pts], [p[1] for p in pts], size)

    def round_fixed(self, k):
        """Cost of k kernel launches, one synchronisation, one read-back (microseconds)."""
        return (
            self.launch_sync + (k - 1) * self.per_launch + self.readback_extra
        )

    def round_fixed_separate(self, k):
        """The same when the read-back is a separate launch + copy + synchronisation."""
        return self.launch_sync + (k - 1) * self.per_launch + self.readback

    def copy_if(self, n):
        return self.at("library", "thrust::copy_if<u64>", "rmm_pool", n)

    def d2h_pinned(self, nbytes):
        return self.at("handover_copy", "device_to_host", "pinned", nbytes)

    def h2d_pinned(self, nbytes):
        return self.at("handover_copy", "host_to_device", "pinned", nbytes)

    def h2d_pageable(self, nbytes):
        return self.at("handover_copy", "host_to_device", "pageable", nbytes)

    def host_alloc(self, nbytes):
        return self.at("allocation", "cudaHostAlloc", "", nbytes)

    def dirty_d2h(self, k, variant="frontier_emit_atomic"):
        return self.at("dirty_list", "device_to_host", variant, k)

    def dirty_h2d(self, k):
        return self.at(
            "dirty_list", "host_to_device", "pinned_copy_scatter", k
        )

    def gather_total(self, entries, variant="warp_per_row"):
        """Row gather to the host plus the upload of the frontier row ids, by gathered entries
        (the two series are measured at the same five frontiers).
        """
        g = self.series("gather", "gather_rows_to_host", variant)
        u = self.series("gather", "frontier_upload", "")
        return loglog(
            [p[0] for p in g], [p[1] + q[1] for p, q in zip(g, u)], entries
        )


# ----------------------------------------------------------------------------------------------
# section 1: full sweep


def cpu_full(s, arith, how):
    return (
        s["cpu_full"][("activity", arith)]
        + s["cpu_full"][("candidates by " + how, arith)]
    )


def cpu_full_best(s, arith):
    return min(cpu_full(s, arith, "row"), cpu_full(s, arith, "column"))


def gpu_candidate_comp(s):
    return min(
        s["candidate"]["H/S/comp/gated"], s["candidate"]["B/S/comp/gated"]
    )


def gpu_designed(s):
    return s["activity"]["H/S/A/comp"] + gpu_candidate_comp(s)


def section_full(sweeps, fixed):
    print("## 1. Full sweep: GPU round against the one-thread CPU sweeps")
    print()
    print(
        "Times in milliseconds (medians). CPU = activity + candidates, by row / by column. Speedup = "
        "cheaper of the two CPU sweeps divided by the GPU round of the same arithmetic "
        "(plain against plain, comp against comp-fma and against comp without -mfma)."
    )
    print()
    rows = []
    for s in sweeps:
        gp, gc = s["round"]["best plain"][0], s["round"]["best comp"][0]
        rows.append(
            [
                s["name"],
                s["band"],
                count(s["nnz"]),
                fmt(gp),
                fmt(gc),
                fmt(gpu_designed(s)),
                fmt(cpu_full(s, "plain", "row"))
                + " / "
                + fmt(cpu_full(s, "plain", "column")),
                fmt(cpu_full(s, "comp-fma", "row"))
                + " / "
                + fmt(cpu_full(s, "comp-fma", "column")),
                fmt(cpu_full(s, "comp", "row"))
                + " / "
                + fmt(cpu_full(s, "comp", "column")),
                fmt(cpu_full_best(s, "plain") / gp),
                fmt(cpu_full_best(s, "comp-fma") / gc),
                fmt(cpu_full_best(s, "comp") / gc),
            ]
        )
    table(
        [
            "matrix",
            "band",
            "nnz",
            "GPU plain",
            "GPU comp",
            "GPU designed",
            "CPU plain row / col",
            "CPU comp-fma row / col",
            "CPU comp row / col",
            "speedup plain",
            "speedup comp-fma",
            "speedup comp",
        ],
        rows,
        ["l", "l"] + ["r"] * 10,
    )

    sp_plain = [
        (cpu_full_best(s, "plain") / s["round"]["best plain"][0], s["name"])
        for s in sweeps
    ]
    sp_fma = [
        (cpu_full_best(s, "comp-fma") / s["round"]["best comp"][0], s["name"])
        for s in sweeps
    ]
    sp_comp = [
        (cpu_full_best(s, "comp") / s["round"]["best comp"][0], s["name"])
        for s in sweeps
    ]
    sp_row = [
        (
            cpu_full(s, "comp-fma", "row") / s["round"]["best comp"][0],
            s["name"],
        )
        for s in sweeps
    ]
    sp_des = [
        (cpu_full_best(s, "comp-fma") / gpu_designed(s), s["name"])
        for s in sweeps
    ]
    rows = []
    for label, sp in [
        ("plain, cheaper CPU sweep / GPU best plain", sp_plain),
        ("comp-fma, cheaper CPU sweep / GPU best comp", sp_fma),
        ("comp without -mfma, cheaper CPU sweep / GPU best comp", sp_comp),
        ("comp-fma, CPU by row / GPU best comp", sp_row),
        (
            "comp-fma, cheaper CPU sweep / GPU designed (H/S/A/comp + S/comp/gated)",
            sp_des,
        ),
    ]:
        lo, hi = min(sp), max(sp)
        rows.append(
            [
                label,
                f"{fmt(lo[0])} ({lo[1]})",
                fmt(quantile([x[0] for x in sp], 0.5)),
                fmt(geomean([x[0] for x in sp])),
                f"{fmt(hi[0])} ({hi[1]})",
                sum(1 for x in sp if x[0] < 1),
            ]
        )
    print("Speedup of the GPU over the CPU, all 32 matrices:")
    print()
    table(
        [
            "comparison",
            "smallest",
            "median",
            "geomean",
            "largest",
            "matrices with speedup < 1",
        ],
        rows,
    )

    des_ratio = [
        (gpu_designed(s) / s["round"]["best comp"][0], s["name"])
        for s in sweeps
    ]
    print(
        f"GPU designed / GPU best comp: median {fmt(quantile([x[0] for x in des_ratio], 0.5))}, "
        f"largest {fmt(max(des_ratio)[0])} ({max(des_ratio)[1]}), smallest {fmt(min(des_ratio)[0])} ({min(des_ratio)[1]}). "
        "The designed pair is timed as two units, so it carries one synchronisation more than a round."
    )
    print()

    # fits
    series = {
        "GPU best plain": [s["round"]["best plain"][0] for s in sweeps],
        "GPU best comp": [s["round"]["best comp"][0] for s in sweeps],
        "GPU designed comp": [gpu_designed(s) for s in sweeps],
        "CPU plain by row": [cpu_full(s, "plain", "row") for s in sweeps],
        "CPU plain by column": [
            cpu_full(s, "plain", "column") for s in sweeps
        ],
        "CPU comp-fma by row": [
            cpu_full(s, "comp-fma", "row") for s in sweeps
        ],
        "CPU comp-fma by column": [
            cpu_full(s, "comp-fma", "column") for s in sweeps
        ],
        "CPU comp by row": [cpu_full(s, "comp", "row") for s in sweeps],
        "CPU comp by column": [cpu_full(s, "comp", "column") for s in sweeps],
    }
    nnz = [s["nnz"] for s in sweeps]
    names = [s["name"] for s in sweeps]
    idx_all = list(range(len(sweeps)))
    idx_a = [i for i, s in enumerate(sweeps) if s["band"].startswith("A")]
    subsets = [
        ("all 32", idx_all),
        ("band A (16 matrices, below 5e5 nnz)", idx_a),
    ]
    fits = {}
    print(
        "Fit time = a + b * nnz (a in microseconds, b in nanoseconds per nonzero), least squares on "
        "relative residuals so that every matrix counts the same. 'proportional' rows force a = 0 for "
        "the CPU (a host sweep has no launch or synchronisation) and take b as the geometric mean of "
        "the time per nonzero. Ordinary least squares is not used: it is decided by square47 alone."
    )
    print()
    rows = []
    for subset, idx in subsets:
        for sname, ys in series.items():
            kinds = [
                (
                    "affine",
                    fit_affine(
                        [nnz[i] for i in idx], [ys[i] for i in idx], True
                    ),
                )
            ]
            if sname.startswith("CPU"):
                kinds.append(
                    (
                        "proportional",
                        (0.0, geomean([ys[i] / nnz[i] for i in idx])),
                    )
                )
            for kind, (a, b) in kinds:
                fits[(subset, kind, sname)] = (a, b)
                resid = [
                    ((a + b * nnz[i]) / ys[i] - 1.0, names[i]) for i in idx
                ]
                worst = max(resid, key=lambda r: abs(r[0]))
                rows.append(
                    [
                        subset,
                        sname,
                        kind,
                        fmt(a * 1000.0),
                        fmt(b * 1e6),
                        f"{100 * quantile([abs(r[0]) for r in resid], 0.5):.0f}%",
                        f"{100 * worst[0]:+.0f}% ({worst[1]})",
                    ]
                )
    table(
        [
            "matrices",
            "series",
            "model",
            "a [us]",
            "b [ns/nnz]",
            "median abs rel. residual",
            "largest rel. residual (fit / measured - 1)",
        ],
        rows,
        ["l", "l", "l", "r", "r", "r", "l"],
    )

    pairs = [
        ("GPU best plain", "CPU plain by row"),
        ("GPU best plain", "CPU plain by column"),
        ("GPU best comp", "CPU comp-fma by row"),
        ("GPU best comp", "CPU comp-fma by column"),
        ("GPU best comp", "CPU comp by row"),
        ("GPU best comp", "CPU comp by column"),
        ("GPU designed comp", "CPU comp-fma by row"),
        ("GPU designed comp", "CPU comp-fma by column"),
    ]
    smallest = nnz[0]
    print(
        f"Crossover size from the fits (GPU cheaper above it). The smallest measured matrix is "
        f"{names[0]} with {count(smallest)} nonzeros; 'factor' is that size divided by the crossover, "
        "the distance of the extrapolation below the measured range. 'none' = the fitted GPU line is "
        "below the fitted CPU line at every positive size."
    )
    print()
    rows = []
    all_cross = []
    for subset, idx in subsets:
        for kind in ["affine", "proportional"]:
            for g, c in pairs:
                ag, bg = fits[(subset, "affine", g)]
                ac, bc = fits[(subset, kind, c)]
                n = crossing(ag, bg, ac, bc)
                if n:
                    all_cross.append(n)
                rows.append(
                    [
                        subset,
                        kind,
                        g,
                        c,
                        count(n) if n else "none",
                        fmt(smallest / n) if n else "-",
                        fmt((ac + bc * 50000) / (ag + bg * 50000)),
                    ]
                )
    table(
        [
            "matrices",
            "CPU model",
            "GPU",
            "CPU",
            "crossover [nnz]",
            "factor below smallest matrix",
            "fitted speedup at 50,000 nnz",
        ],
        rows,
        ["l", "l", "l", "l", "r", "r", "r"],
    )
    print(
        f"Fitted crossovers that exist: {len(all_cross)} of {len(rows)}, from {count(min(all_cross))} to {count(max(all_cross))} "
        f"nonzeros; {sum(1 for n in all_cross if n >= 50000)} at or above 50,000 and {sum(1 for n in all_cross if n >= smallest)} inside the measured range."
    )
    print()

    # leave-one-matrix-out sensitivity, CPU proportional
    loo = []
    for g, c in [
        ("GPU best comp", "CPU comp-fma by column"),
        ("GPU best comp", "CPU comp-fma by row"),
        ("GPU best plain", "CPU plain by column"),
        ("GPU designed comp", "CPU comp-fma by column"),
    ]:
        vals = []
        for drop in idx_all:
            keep = [i for i in idx_all if i != drop]
            ag, bg = fit_affine(
                [nnz[i] for i in keep], [series[g][i] for i in keep], True
            )
            bc = geomean([series[c][i] / nnz[i] for i in keep])
            n = crossing(ag, bg, 0.0, bc)
            vals.append((n if n else 0.0, names[drop]))
        loo.append(
            [
                g + " against " + c,
                count(min(vals)[0]) + f" (without {min(vals)[1]})",
                count(max(vals)[0]) + f" (without {max(vals)[1]})",
            ]
        )
    print(
        "Leave-one-matrix-out range of the crossover (GPU affine, CPU proportional, all 32):"
    )
    print()
    table(
        ["pair", "smallest crossover", "largest crossover"],
        loo,
        ["l", "r", "r"],
    )

    # per-matrix implied crossover: GPU fixed part and slope from the fit, the matrix's own CPU cost
    print(
        "Per-matrix implied crossover: GPU fixed part and slope from the relative fit over all 32, the "
        "matrix's own CPU time per nonzero, CPU fixed part taken as 0: n = a_gpu / (cpu_ns_per_nnz - b_gpu)."
    )
    print()
    rows = []
    band_a = subsets[1][0]
    for label, subset, g, arith in [
        ("plain", "all 32", "GPU best plain", "plain"),
        ("comp-fma", "all 32", "GPU best comp", "comp-fma"),
        ("comp without -mfma", "all 32", "GPU best comp", "comp"),
        (
            "comp-fma, designed kernel pair",
            "all 32",
            "GPU designed comp",
            "comp-fma",
        ),
        (
            "comp-fma, designed kernel pair, GPU fit over band A",
            band_a,
            "GPU designed comp",
            "comp-fma",
        ),
    ]:
        ag, bg = fits[(subset, "affine", g)]
        vals = []
        for s in sweeps:
            c = cpu_full_best(s, arith) / s["nnz"]
            vals.append((ag / (c - bg) if c > bg else float("inf"), s["name"]))
        rows.append(
            [
                label,
                f"{count(min(vals)[0])} ({min(vals)[1]})",
                count(quantile([v[0] for v in vals], 0.25)),
                count(quantile([v[0] for v in vals], 0.5)),
                count(quantile([v[0] for v in vals], 0.75)),
                f"{count(max(vals)[0])} ({max(vals)[1]})",
                sum(1 for v in vals if v[0] >= 50000),
            ]
        )
    table(
        [
            "arithmetic",
            "smallest",
            "25%",
            "median",
            "75%",
            "largest",
            "matrices with implied crossover >= 50,000",
        ],
        rows,
    )
    print(
        "The first three rows use the fastest round variant of each matrix, which differs between matrices "
        "and includes layouts DESIGN.md 2.4 rules out. The last two use the kernel pair of DESIGN.md 2.4; "
        "with it the implied crossover of a matrix whose host sweep is cheap per nonzero (most rows with "
        "two or more infinite activity terms, so that the candidate pass has nothing to do) reaches the "
        "50,000-nonzero end of the target range."
    )
    print()

    # with the upload of the matrix when it is not on the device yet
    print(
        "The same full sweep when the matrix is not on the device yet: GPU best comp plus one upload of "
        "CSR and CSC (24 bytes per nonzero, 8 bytes per row and column offset, 16 bytes per row and "
        "column of bounds and sides; assumption), pinned or pageable host memory (handover_copy)."
    )
    print()
    rows = []
    up = {"pinned": [], "pageable": []}
    for s in sweeps:
        nbytes = 24 * s["nnz"] + 24 * (s["rows"] + s["cols"])
        for kind, f in [
            ("pinned", fixed.h2d_pinned),
            ("pageable", fixed.h2d_pageable),
        ]:
            up[kind].append(s["round"]["best comp"][0] + f(nbytes) / 1000.0)
    for kind in ["pinned", "pageable"]:
        sp = [
            (cpu_full_best(s, "comp-fma") / t, s["name"])
            for s, t in zip(sweeps, up[kind])
        ]
        ag, bg = fit_affine(nnz, up[kind], True)
        ac, bc = fits[("all 32", "proportional", "CPU comp-fma by column")]
        n = crossing(ag, bg, ac, bc)
        rows.append(
            [
                kind,
                fmt(ag * 1000),
                fmt(bg * 1e6),
                f"{fmt(min(sp)[0])} ({min(sp)[1]})",
                fmt(quantile([x[0] for x in sp], 0.5)),
                sum(1 for x in sp if x[0] < 1),
                count(n) if n else "none",
            ]
        )
    table(
        [
            "upload from",
            "a [us]",
            "b [ns/nnz]",
            "smallest speedup over CPU comp-fma",
            "median speedup",
            "matrices with speedup < 1",
            "crossover against CPU comp-fma by column, proportional [nnz]",
        ],
        rows,
    )
    return fits


# ----------------------------------------------------------------------------------------------
# section 2: frontier sweeps

PCTS = ["0.01", "0.1", "1", "10"]


def frontier_points(s, arith="comp"):
    """Measured points of one matrix, sorted by frontier nonzeros, the full matrix last.

    arith "comp": GPU H/S/A/comp, CPU comp-fma. arith "plain": GPU H/S/A/plain, CPU plain.
    Each point: label, rows, nnz, gpu_list (compaction + activity kernel over the list), gpu_a
    (the cheaper of gpu_list and the same kernel over all rows without compaction: the GPU engine
    can always sweep all rows instead), cand (the full-matrix candidate kernel), cpu_a / cpu_s
    for warm and cold (cpu_s = activity + candidates by row).
    """
    cpu_arith = "comp-fma" if arith == "comp" else "plain"
    if arith == "comp":
        cand = gpu_candidate_comp(s)
        full_act = s["activity"]["H/S/A/comp"]
    else:
        cand = min(s["candidate"]["H/S/plain"], s["candidate"]["B/S/plain"])
        full_act = s["activity"]["H/S/A/plain"]
    # fastest activity kernel over all rows among the split-layout variants (any scheduling)
    fast_act = min(
        t
        for name, t in s["activity"].items()
        if name.split("/")[1] == "S" and name.endswith("/" + arith)
    )
    pts = []
    for pct in PCTS:
        f = s["frontier"][pct]
        if not f["nnz"]:
            continue  # a list of empty rows: no work on either side
        p = {
            "label": pct + "%",
            "rows": f["rows"],
            "nnz": f["nnz"],
            "gpu_list": f[arith],
            "gpu_a": min(f[arith], full_act),
            "gpu_b": min(f[arith], fast_act),
            "cand": cand,
        }
        for temp in ["warm", "cold"]:
            a = f["cpu"][("activity", temp, cpu_arith)]
            p["cpu_a_" + temp] = a
            p["cpu_c_" + temp] = f["cpu"][
                ("candidates by row", temp, cpu_arith)
            ]
            p["cpu_s_" + temp] = a + p["cpu_c_" + temp]
        pts.append(p)
    full = {
        "label": "100%",
        "rows": s["rows"],
        "nnz": s["nnz"],
        "gpu_list": full_act,
        "gpu_a": full_act,
        "gpu_b": fast_act,
        "cand": cand,
    }
    a = s["cpu_full"][("activity", cpu_arith)]
    for temp in [
        "warm",
        "cold",
    ]:  # the benchmark does not evict before a full sweep: one value for both
        full["cpu_a_" + temp] = a
        full["cpu_c_" + temp] = s["cpu_full"][("candidates by row", cpu_arith)]
        full["cpu_s_" + temp] = a + full["cpu_c_" + temp]
    pts.append(full)
    pts.sort(key=lambda p: p["nnz"])
    out = []
    for p in pts:  # two lists can hold the same number of nonzeros: keep the first of each size
        if not out or p["nnz"] > out[-1]["nnz"]:
            out.append(p)
    return out


def interp(pts, fn, x, below):
    """fn(point) at x frontier nonzeros: log-log interpolation between the measured frontiers.
    Below the smallest: constant (below="const") or proportional to x ("prop").
    """
    xs = [p["nnz"] for p in pts]
    ys = [fn(p) for p in pts]
    if x <= xs[0]:
        return ys[0] if below == "const" else ys[0] * x / xs[0]
    return loglog(xs, ys, x)


def crossover(pts, cpu, gpu):
    """Frontier size above which the GPU is cheaper at every larger measured point.

    cpu and gpu are functions of a point (milliseconds). Returns (nnz, rows, flag, where):
    flag "<" = the GPU is already cheaper at the smallest measured frontier (extrapolated),
    ">" = the CPU is cheaper even on the full matrix, "" = inside the measured range; where = the
    two measured frontiers that bracket it.
    """
    d = [math.log(cpu(p) / gpu(p)) for p in pts]
    last_cpu = max((i for i, v in enumerate(d) if v < 0), default=None)
    if last_cpu is None:
        p = pts[0]
        ratio = gpu(p) / cpu(p)  # < 1
        return p["nnz"] * ratio, p["rows"] * ratio, "<", "below " + p["label"]
    if last_cpu == len(pts) - 1:
        return float("inf"), float("inf"), ">", "above " + pts[-1]["label"]
    p, q = pts[last_cpu], pts[last_cpu + 1]
    t = -d[last_cpu] / (d[last_cpu + 1] - d[last_cpu])
    nnz = math.exp(
        math.log(p["nnz"]) + t * (math.log(q["nnz"]) - math.log(p["nnz"]))
    )
    rows = math.exp(
        math.log(p["rows"]) + t * (math.log(q["rows"]) - math.log(p["rows"]))
    )
    return nnz, rows, "", p["label"] + " to " + q["label"]


def nnz_at_work(pts, kappa, work):
    """Frontier nonzeros of this matrix at which nnz + kappa * rows equals work."""
    ws, xs = [], []
    for w, x in sorted((p["nnz"] + kappa * p["rows"], p["nnz"]) for p in pts):
        if not ws or (
            w > ws[-1] and x > xs[-1]
        ):  # tiny lists need not be ordered alike
            ws.append(w)
            xs.append(x)
    if work <= ws[0]:
        return xs[0] * work / ws[0]
    if work >= ws[-1]:
        return xs[-1]
    return loglog(ws, xs, work)


def scenarios(fixed):
    """(key, description, cpu function, gpu function) of every comparison of section 2."""

    def d2h(
        p,
    ):  # state the host needs: at most one record per frontier nonzero
        return fixed.dirty_d2h(p["nnz"]) / 1000.0

    def h2d(p):  # results sent back: one record per frontier row
        return fixed.dirty_h2d(p["rows"]) / 1000.0

    def gather(
        p,
    ):  # rows of the frontier copied to the host, when there is no mirror
        return fixed.gather_total(p["nnz"]) / 1000.0

    def step(
        p,
    ):  # model of a GPU step: the candidate part costs what the activity part costs
        return 2.0 * p["gpu_a"]

    return [
        (
            "A cold",
            "activities only, CPU cold (measured on both sides)",
            lambda p: p["cpu_a_cold"],
            lambda p: p["gpu_a"],
        ),
        (
            "A warm",
            "activities only, CPU warm (measured on both sides)",
            lambda p: p["cpu_a_warm"],
            lambda p: p["gpu_a"],
        ),
        (
            "S cold",
            "step, CPU cold, GPU model 2 x GPU A (assumption)",
            lambda p: p["cpu_s_cold"],
            step,
        ),
        (
            "S warm",
            "step, CPU warm, GPU model 2 x GPU A (assumption)",
            lambda p: p["cpu_s_warm"],
            step,
        ),
        (
            "S hi",
            "step, CPU cold, GPU A + full-matrix candidate kernel (measured upper bound of the GPU)",
            lambda p: p["cpu_s_cold"],
            lambda p: p["gpu_a"] + p["cand"],
        ),
        (
            "S lo",
            "step, CPU cold, GPU candidates free (lower bound of the GPU)",
            lambda p: p["cpu_s_cold"],
            lambda p: p["gpu_a"],
        ),
        (
            "H cold",
            "host owns the state: GPU model step + dirty lists up and down, CPU cold",
            lambda p: p["cpu_s_cold"],
            lambda p: step(p) + h2d(p) + d2h(p),
        ),
        (
            "H warm",
            "host owns the state: GPU model step + dirty lists up and down, CPU warm",
            lambda p: p["cpu_s_warm"],
            lambda p: step(p) + h2d(p) + d2h(p),
        ),
        (
            "D mirror",
            "device owns the state, host has a mirror: CPU cold + dirty lists down and up, GPU model step",
            lambda p: p["cpu_s_cold"] + d2h(p) + h2d(p),
            step,
        ),
        (
            "D gather",
            "device owns the state, no mirror: CPU cold + dirty lists + row gather, GPU model step",
            lambda p: p["cpu_s_cold"] + d2h(p) + h2d(p) + gather(p),
            step,
        ),
        # sensitivity of the crossover to what the two sides are taken to be (not used for the thresholds)
        (
            "A cold fast",
            "sensitivity: activities only, CPU cold, GPU = cheaper of the list and the fastest split-layout activity kernel over all rows (measured on both sides)",
            lambda p: p["cpu_a_cold"],
            lambda p: p["gpu_b"],
        ),
        (
            "C cold",
            "sensitivity: host with incrementally updated activities (CPU candidates by row only), cold, GPU model step",
            lambda p: p["cpu_c_cold"],
            step,
        ),
        (
            "C warm",
            "sensitivity: the same, CPU warm",
            lambda p: p["cpu_c_warm"],
            step,
        ),
        (
            "S cold 1.5",
            "sensitivity: step, CPU cold, GPU model 1.5 x GPU A",
            lambda p: p["cpu_s_cold"],
            lambda p: 1.5 * p["gpu_a"],
        ),
        (
            "S cold 3",
            "sensitivity: step, CPU cold, GPU model 3 x GPU A",
            lambda p: p["cpu_s_cold"],
            lambda p: 3.0 * p["gpu_a"],
        ),
    ]


def dist_row(label, vals):
    """vals: list of (value, name); infinite values are counted, not averaged."""
    finite = [v for v in vals if math.isfinite(v[0])]
    xs = [v[0] for v in finite]
    lo, hi = min(finite), max(finite)
    return [
        label,
        f"{count(lo[0])} ({lo[1]})",
        count(quantile(xs, 0.25)),
        count(quantile(xs, 0.5)),
        count(quantile(xs, 0.75)),
        f"{count(hi[0])} ({hi[1]})",
        count(geomean(xs)),
        fmt(hi[0] / lo[0]),
        len(vals) - len(finite),
    ]


DIST_HEADER = [
    "scenario",
    "smallest",
    "25%",
    "median",
    "75%",
    "largest",
    "geomean",
    "largest / smallest",
    "no crossover",
]


def losses_at(sweeps, cpu, gpu, threshold, kappa=None):
    """Per matrix: (loss in microseconds at the threshold, name, factor, engine that is too slow
    there, frontier nonzeros at the threshold).
    """
    out = []
    for s in sweeps:
        pts = frontier_points(s)
        x = threshold if kappa is None else nnz_at_work(pts, kappa, threshold)
        # no frontier is larger than the matrix: above it the rule keeps the host for a full sweep
        x = min(x, pts[-1]["nnz"])
        c = interp(pts, cpu, x, "prop")
        g = interp(pts, gpu, x, "const")
        out.append(
            (
                abs(c - g) * 1000.0,
                s["name"],
                max(c, g) / min(c, g),
                "host" if c > g else "GPU",
                x,
                c,
                g,
            )
        )
    return out


def threshold_rows(sweeps, scen, cross, keys, thresholds, kappa=None):
    rows = []
    for key in keys:
        cpu, gpu = next((c, g) for k, _, c, g in scen if k == key)
        xs = [
            c[0] if kappa is None else c[0] + kappa * c[2] for c in cross[key]
        ]
        for T in thresholds:
            losses = losses_at(sweeps, cpu, gpu, T, kappa)
            host = [x for x in losses if x[3] == "host"]
            gpul = [x for x in losses if x[3] == "GPU"]
            rows.append(
                [
                    key,
                    count(T),
                    sum(1 for x in xs if x < T / 2),
                    sum(1 for x in xs if x > 2 * T),
                    f"{fmt(max(host)[0])} ({max(host)[1]})" if host else "-",
                    f"{fmt(max(gpul)[0])} ({max(gpul)[1]})" if gpul else "-",
                    fmt(statistics.median([x[0] for x in losses])),
                    fmt(sum(x[0] for x in losses)),
                    f"{fmt(max(losses, key=lambda x: x[2])[2])} ({max(losses, key=lambda x: x[2])[1]})",
                ]
            )
    return rows


THRESHOLD_HEADER = [
    "scenario",
    "T",
    "matrices with crossover < T/2",
    "matrices with crossover > 2T",
    "largest loss, host kept too long [us]",
    "largest loss, GPU used too early [us]",
    "median loss [us]",
    "sum of losses over 32 [us]",
    "largest factor",
]


def section_frontier(sweeps, fixed, diag):
    print(
        "## 2. Frontier sweeps: GPU (compaction + kernel) against the CPU, warm and cold"
    )
    print()
    print(
        "Compensated arithmetic (GPU H/S/A/comp, CPU comp-fma), microseconds. 'GPU list' = compaction + "
        "activity kernel over the frontier rows; 'GPU all rows' = the same kernel over all rows without "
        "compaction; 'GPU A' = the cheaper of the two (the GPU engine can always sweep all rows); 'cand' = "
        "the full-matrix candidate kernel (no frontier-restricted candidate kernel exists). 'CPU A' = "
        "activity of the frontier rows, 'CPU S' = activity + candidates by row; cold / warm."
    )
    print()
    rows = []
    for s in sweeps:
        cand = gpu_candidate_comp(s)
        full_act = s["activity"]["H/S/A/comp"]
        for pct in PCTS:
            f = s["frontier"][pct]
            c = f["cpu"]
            a_cold, a_warm = (
                c[("activity", "cold", "comp-fma")],
                c[("activity", "warm", "comp-fma")],
            )
            s_cold = a_cold + c[("candidates by row", "cold", "comp-fma")]
            s_warm = a_warm + c[("candidates by row", "warm", "comp-fma")]
            best = min(f["comp"], full_act)
            rows.append(
                [
                    s["name"],
                    pct + "%",
                    count(f["rows"]),
                    count(f["nnz"]),
                    fmt(f["compaction"] * 1000),
                    fmt(f["plain"] * 1000),
                    fmt(f["comp"] * 1000),
                    fmt(full_act * 1000),
                    fmt(cand * 1000),
                    fmt(a_cold * 1000) + " / " + fmt(a_warm * 1000),
                    fmt(s_cold * 1000) + " / " + fmt(s_warm * 1000),
                    fmt(a_cold / best),
                    fmt(a_warm / best),
                    fmt(s_cold / (2 * best)),
                ]
            )
    table(
        [
            "matrix",
            "frontier",
            "rows",
            "nnz",
            "compaction only",
            "GPU list plain",
            "GPU list comp",
            "GPU all rows comp",
            "cand",
            "CPU A cold / warm",
            "CPU S cold / warm",
            "CPU A cold / GPU A",
            "CPU A warm / GPU A",
            "CPU S cold / (2 x GPU A)",
        ],
        rows,
        ["l", "l"] + ["r"] * 12,
    )

    print(
        "The same per matrix in one line: for each frontier 'frontier nnz: GPU A / CPU A cold / CPU S cold / "
        "CPU S warm' in microseconds."
    )
    print()
    rows = []
    for s in sweeps:
        row = [s["name"], s["band"], fmt(s["activity"]["H/S/A/comp"] * 1000)]
        for pct in PCTS:
            f = s["frontier"][pct]
            c = f["cpu"]
            a_cold, a_warm = (
                c[("activity", "cold", "comp-fma")],
                c[("activity", "warm", "comp-fma")],
            )
            s_cold = a_cold + c[("candidates by row", "cold", "comp-fma")]
            s_warm = a_warm + c[("candidates by row", "warm", "comp-fma")]
            best = min(f["comp"], s["activity"]["H/S/A/comp"])
            row.append(
                f"{count(f['nnz'])}: {fmt(best * 1000)} / {fmt(a_cold * 1000)} / {fmt(s_cold * 1000)} / {fmt(s_warm * 1000)}"
            )
        rows.append(row)
    table(
        [
            "matrix",
            "band",
            "GPU all rows",
            "0.01% of the rows",
            "0.1%",
            "1%",
            "10%",
        ],
        rows,
        ["l", "l"] + ["r"] * 5,
    )

    rows = []
    for pct in PCTS:
        ga = [s["frontier"][pct]["comp"] for s in sweeps]
        gb = [
            min(s["frontier"][pct]["comp"], s["activity"]["H/S/A/comp"])
            for s in sweeps
        ]
        co = [s["frontier"][pct]["compaction"] for s in sweeps]
        per = {}
        for temp in ["cold", "warm"]:
            per[temp] = [
                1e6
                * (
                    s["frontier"][pct]["cpu"][("activity", temp, "comp-fma")]
                    + s["frontier"][pct]["cpu"][
                        ("candidates by row", temp, "comp-fma")
                    ]
                )
                / s["frontier"][pct]["nnz"]
                for s in sweeps
            ]
        rows.append(
            [
                pct + "%",
                f"{fmt(min(co) * 1000)} / {fmt(statistics.median(co) * 1000)} / {fmt(max(co) * 1000)}",
                f"{fmt(min(ga) * 1000)} / {fmt(statistics.median(ga) * 1000)} / {fmt(max(ga) * 1000)}",
                f"{fmt(min(gb) * 1000)} / {fmt(statistics.median(gb) * 1000)} / {fmt(max(gb) * 1000)}",
                f"{fmt(min(per['cold']))} / {fmt(statistics.median(per['cold']))} / {fmt(max(per['cold']))}",
                f"{fmt(min(per['warm']))} / {fmt(statistics.median(per['warm']))} / {fmt(max(per['warm']))}",
                fmt(
                    statistics.median(
                        [c / w for c, w in zip(per["cold"], per["warm"])]
                    )
                ),
            ]
        )
    full = [
        1e6
        * (
            s["cpu_full"][("activity", "comp-fma")]
            + s["cpu_full"][("candidates by row", "comp-fma")]
        )
        / s["nnz"]
        for s in sweeps
    ]
    ga = [s["activity"]["H/S/A/comp"] for s in sweeps]
    rows.append(
        [
            "100% (full sweep, no compaction)",
            "-",
            "-",
            f"{fmt(min(ga) * 1000)} / {fmt(statistics.median(ga) * 1000)} / {fmt(max(ga) * 1000)}",
            f"{fmt(min(full))} / {fmt(statistics.median(full))} / {fmt(max(full))}",
            "same",
            "1",
        ]
    )
    print("Over the 32 matrices, min / median / max:")
    print()
    table(
        [
            "frontier",
            "compaction only [us]",
            "GPU list comp [us]",
            "GPU A comp [us]",
            "CPU S cold [ns per frontier nnz]",
            "CPU S warm [ns per frontier nnz]",
            "median cold / warm",
        ],
        rows,
    )

    # crossovers per matrix
    scen = scenarios(fixed)
    keys = [k for k, _, _, _ in scen]
    cross = {k: [] for k in keys}
    cross_plain = {k: [] for k in keys[:2]}
    rows = []
    for s in sweeps:
        pts = frontier_points(s)
        row = [
            s["name"],
            s["band"],
            count(s["nnz"]),
            fmt(1000 * pts[0]["gpu_a"]),
            fmt(1000 * pts[0]["cand"]),
        ]
        for key, _, cpu, gpu in scen:
            nnz, nrows, flag, where = crossover(pts, cpu, gpu)
            cross[key].append((nnz, s["name"], nrows, flag, where))
            if not math.isfinite(nnz):
                row.append("never")
            elif key in ("A cold", "S cold"):
                row.append(
                    f"{flag}{count(nnz)} ({count(max(nrows, 1))} rows; {where})"
                )
            else:
                row.append(f"{flag}{count(nnz)}")
        rows.append(row)
        pts_p = frontier_points(s, "plain")
        for key, _, cpu, gpu in scen[:2]:
            nnz, nrows, flag, where = crossover(pts_p, cpu, gpu)
            cross_plain[key].append((nnz, s["name"], nrows, flag, where))
    print(
        "Crossover per matrix in frontier nonzeros (for A cold and S cold also the frontier rows and the "
        "bracketing measured frontiers). '<' = the GPU is already cheaper at the smallest measured "
        "frontier, value extrapolated. Scenarios:"
    )
    print()
    for key, desc, _, _ in scen:
        print(f"- {key}: {desc}")
    print(
        "- hand-over assumptions of H and D: dirty list to the device (pinned_copy_scatter) with one record per "
        "frontier row; dirty list to the host (frontier_emit_atomic) with one record per frontier nonzero; "
        "row gather = gather_rows_to_host warp_per_row + frontier_upload."
    )
    print()
    table(
        [
            "matrix",
            "band",
            "nnz",
            "GPU A at the smallest frontier [us]",
            "cand [us]",
        ]
        + keys,
        rows,
        ["l", "l"] + ["r"] * (3 + len(keys)),
    )

    print(
        "Distribution of the crossover over the 32 matrices (frontier nonzeros):"
    )
    print()
    rows = [
        dist_row(f"{key}: {desc}", [(c[0], c[1]) for c in cross[key]])
        for key, desc, _, _ in scen
    ]
    for key in cross_plain:
        rows.append(
            dist_row(
                f"{key}, plain arithmetic on both sides",
                [(c[0], c[1]) for c in cross_plain[key]],
            )
        )
    table(DIST_HEADER, rows)

    print("The same in frontier rows:")
    print()
    rows = [
        dist_row(key, [(max(c[2], 1.0), c[1]) for c in cross[key]])
        for key in ["A cold", "A warm", "S cold", "S warm"]
    ]
    table(DIST_HEADER, rows)

    print("By band (geometric mean of the crossover in frontier nonzeros):")
    print()
    rows = []
    for group, test in [
        ("A (5e4 to 5e5)", lambda b: b.startswith("A")),
        ("B (5e5 to 1e7)", lambda b: b.startswith("B")),
        ("C (square47)", lambda b: b == "C"),
    ]:
        idx = [i for i, s in enumerate(sweeps) if test(s["band"])]
        row = [group, len(idx)]
        for key in keys:
            vals = [
                cross[key][i][0]
                for i in idx
                if math.isfinite(cross[key][i][0])
            ]
            row.append(count(geomean(vals)) if vals else "-")
        rows.append(row)
    table(["band", "matrices"] + keys, rows)

    for key in ["A cold", "A warm", "S cold", "S warm"]:
        w = {}
        for c in cross[key]:
            w[c[4]] = w.get(c[4], 0) + 1
        print(
            f"Bracket of the {key} crossover: "
            + ", ".join(
                f"{k}: {v}"
                for k, v in sorted(w.items(), key=lambda kv: -kv[1])
            )
            + "."
        )
    gap = [c[1] for c in cross["A cold"] if c[4] == "10% to 100%"]
    print(
        "The 100% point is the full CPU sweep, which the benchmark does not evict, and no frontier between 10% and "
        "100% of the rows was measured. A crossover bracketed '10% to 100%' is therefore an interpolation over a "
        "decade from a cold point to a warm one; what is measured there is only that the host is cheaper at 10% of "
        f"the rows and the GPU on the full matrix. A cold: {len(gap)} matrices ({', '.join(gap)})."
    )
    print()

    # thresholds in frontier nonzeros
    print(
        "Candidate thresholds T in frontier nonzeros: host below T, GPU from T on. For a matrix the loss "
        "of the rule is largest at T itself and equals |CPU(T) - GPU(T)| there (interpolated between the "
        "measured frontiers); 'host kept too long' = the GPU was cheaper at T, 'GPU used too early' = the "
        "host was cheaper at T. Factor = slower engine / faster engine at T."
    )
    print()
    table(
        THRESHOLD_HEADER,
        threshold_rows(
            sweeps,
            scen,
            cross,
            [
                "A cold",
                "A warm",
                "S cold",
                "S warm",
                "H cold",
                "H warm",
                "D mirror",
                "D gather",
            ],
            THRESHOLDS,
        ),
        ["l"] + ["r"] * 8,
    )

    # the row term of the work unit
    print(
        "Work unit with a row term: W = frontier nonzeros + kappa * frontier rows. Spread of the per-matrix "
        "crossover expressed in W (standard deviation of the natural log; largest / smallest):"
    )
    print()
    rows = []
    for kappa in KAPPAS:
        row = [kappa]
        for key in [
            "A cold",
            "A warm",
            "S cold",
            "S warm",
            "H cold",
            "H warm",
        ]:
            ws = [c[0] + kappa * c[2] for c in cross[key]]
            row.append(
                f"{statistics.pstdev([math.log(w) for w in ws]):.2f}; {fmt(max(ws) / min(ws))}; geomean {count(geomean(ws))}"
            )
        rows.append(row)
    table(
        ["kappa", "A cold", "A warm", "S cold", "S warm", "H cold", "H warm"],
        rows,
        ["r"] + ["r"] * 6,
    )

    # host cost model alpha * rows + beta * nnz
    print(
        "Host cost model time = alpha * frontier rows + beta * frontier nonzeros, least squares on relative "
        "residuals over the measured frontiers (32 matrices x 4 frontiers; '<= 1%' leaves the 10% lists out):"
    )
    print()
    rows = []
    host_model = {}
    for key, label in [
        ("cpu_s_cold", "CPU S cold"),
        ("cpu_s_warm", "CPU S warm"),
        ("cpu_a_cold", "CPU A cold"),
        ("cpu_a_warm", "CPU A warm"),
    ]:
        for sel, labels in [
            ("0.01% to 10%", ("0.01%", "0.1%", "1%", "10%")),
            ("<= 1%", ("0.01%", "0.1%", "1%")),
        ]:
            x1, x2, y = [], [], []
            for s in sweeps:
                for p in frontier_points(s):
                    if p["label"] in labels:
                        x1.append(p["rows"])
                        x2.append(p["nnz"])
                        y.append(p[key])
            alpha, beta = fit_two(x1, x2, y)
            host_model[(key, sel)] = (alpha, beta)
            resid = [
                abs((alpha * a + beta * b) / t - 1.0)
                for a, b, t in zip(x1, x2, y)
            ]
            rows.append(
                [
                    label,
                    sel,
                    len(y),
                    fmt(alpha * 1e6),
                    fmt(beta * 1e6),
                    fmt(alpha / beta),
                    f"{100 * statistics.median(resid):.0f}%",
                ]
            )
    table(
        [
            "series",
            "frontiers",
            "points",
            "alpha [ns/row]",
            "beta [ns/nnz]",
            "kappa = alpha / beta",
            "median abs rel. residual",
        ],
        rows,
        ["l", "l"] + ["r"] * 5,
    )

    print(
        f"Candidate thresholds in W = nnz + {REC['kappa']} * rows (same columns as above):"
    )
    print()
    table(
        THRESHOLD_HEADER,
        threshold_rows(
            sweeps,
            scen,
            cross,
            [
                "A cold",
                "A warm",
                "S cold",
                "S warm",
                "H cold",
                "H warm",
                "D mirror",
                "D gather",
            ],
            W_THRESHOLDS,
            REC["kappa"],
        ),
        ["l"] + ["r"] * 8,
    )

    print(
        "Threshold with the smallest sum of losses over the 32 matrices on a fine grid (factor 2^(1/4) between "
        "grid points, 250 to 256,000), and the range of thresholds whose sum is within 10% of that minimum:"
    )
    print()
    grid = [250.0 * 2 ** (i / 4.0) for i in range(41)]
    rows = []
    best = {}
    for key in [
        "A cold",
        "A warm",
        "S cold",
        "S warm",
        "H cold",
        "H warm",
        "D mirror",
        "D gather",
    ]:
        cpu, gpu = next((c, g) for k, _, c, g in scen if k == key)
        row = [key]
        for kappa in [None, REC["kappa"]]:
            per = [losses_at(sweeps, cpu, gpu, T, kappa) for T in grid]
            sums = [sum(x[0] for x in v) for v in per]
            worst = [max(x[2] for x in v) for v in per]
            lo = min(sums)
            ok = [T for T, v in zip(grid, sums) if v <= 1.1 * lo]
            best[(key, kappa)] = (grid[sums.index(lo)], lo, min(ok), max(ok))
            row += [
                count(grid[sums.index(lo)]),
                fmt(lo),
                f"{count(min(ok))} to {count(max(ok))}",
                f"{count(grid[worst.index(min(worst))])} (x{fmt(min(worst))})",
            ]
        rows.append(row)
    table(
        [
            "scenario",
            "best T [nnz]",
            "sum of losses [us]",
            "T within 10%",
            "T with the smallest largest factor",
            f"best W (kappa = {REC['kappa']})",
            "sum of losses [us]",
            "W within 10%",
            "W with the smallest largest factor",
        ],
        rows,
        ["l"] + ["r"] * 8,
    )
    print(
        "The sum of losses is one criterion among several: the loss of a matrix is its worst case per round (at the "
        "threshold itself), not an expected loss, because the distribution of real frontier sizes is not measured. "
        "'Smallest largest factor' is the threshold at which the slower engine is slower by the smallest factor on "
        "the worst matrix; it lies above the smallest-sum threshold in every scenario."
    )
    print()
    rows = []
    for label, kappa, thr in [
        (
            f"T = {count(REC['frontier_nnz'])} nonzeros",
            None,
            REC["frontier_nnz"],
        ),
        (
            f"W = {count(REC['frontier_work'])}, kappa = {REC['kappa']}",
            REC["kappa"],
            REC["frontier_work"],
        ),
        (
            f"W = {count(REC['device_owned_mirror_work'])}, kappa = {REC['kappa']}",
            REC["kappa"],
            REC["device_owned_mirror_work"],
        ),
        (
            f"W = {count(REC['device_owned_gather_work'])}, kappa = {REC['kappa']}",
            REC["kappa"],
            REC["device_owned_gather_work"],
        ),
    ]:
        row = [label]
        for key in [
            "A cold",
            "A warm",
            "S cold",
            "S warm",
            "H cold",
            "H warm",
            "D mirror",
            "D gather",
        ]:
            cpu, gpu = next((c, g) for k, _, c, g in scen if k == key)
            v = sum(x[0] for x in losses_at(sweeps, cpu, gpu, thr, kappa))
            row.append(f"{v / best[(key, kappa)][1]:.2f}")
        rows.append(row)
    print(
        "Sum of losses of the proposed thresholds divided by the smallest sum of the same scenario and unit:"
    )
    print()
    table(
        [
            "threshold",
            "A cold",
            "A warm",
            "S cold",
            "S warm",
            "H cold",
            "H warm",
            "D mirror",
            "D gather",
        ],
        rows,
    )

    # GPU only: frontier list against all rows
    print(
        "GPU only: compaction + H/S/A/comp over a frontier list against the same kernel over all rows without "
        "compaction (activity H/S/A/comp, full). Ratio frontier / full; above 1 the full sweep is cheaper:"
    )
    print()
    rows = []
    for pct in PCTS:
        r = [
            (
                s["frontier"][pct]["comp"] / s["activity"]["H/S/A/comp"],
                s["name"],
            )
            for s in sweeps
        ]
        rows.append(
            [
                pct + "%",
                f"{fmt(min(r)[0])} ({min(r)[1]})",
                fmt(quantile([x[0] for x in r], 0.25)),
                fmt(statistics.median([x[0] for x in r])),
                fmt(quantile([x[0] for x in r], 0.75)),
                f"{fmt(max(r)[0])} ({max(r)[1]})",
                sum(1 for x in r if x[0] > 1),
                sum(
                    1
                    for x, s in zip(r, sweeps)
                    if x[0] > 1 and s["band"].startswith("A")
                ),
            ]
        )
    table(
        [
            "frontier (share of the rows)",
            "smallest",
            "25%",
            "median",
            "75%",
            "largest",
            "matrices with ratio > 1",
            "of them in band A (16)",
        ],
        rows,
    )

    def rule_cost(rule):
        wrong, loss, worst = 0, 0.0, (0.0, "")
        for s in sweeps:
            full_act = s["activity"]["H/S/A/comp"]
            for pct in PCTS:
                f = s["frontier"][pct]
                t = f["comp"] if rule(s, f) else full_act
                extra = (t - min(f["comp"], full_act)) * 1000.0
                if extra > 0:
                    wrong += 1
                    loss += extra
                    worst = max(worst, (extra, f"{s['name']} at {pct}%"))
        return [wrong, fmt(loss), f"{fmt(worst[0])} ({worst[1]})"]

    print(
        "Rules for the GPU's choice between the list and all rows, on the 128 measured cases (frontiers up to 10% of the rows):"
    )
    print()
    rows = [
        ["always the list"] + rule_cost(lambda s, f: True),
        ["always all rows"] + rule_cost(lambda s, f: False),
    ]
    for share in [0.001, 0.01, 0.1]:
        rows.append(
            [
                f"list when the frontier holds less than {100 * share:g}% of the nonzeros"
            ]
            + rule_cost(lambda s, f, share=share: f["nnz"] < share * s["nnz"])
        )
    for size in [200000, 500000, 1000000]:
        rows.append(
            [f"list when the matrix has more than {count(size)} nonzeros"]
            + rule_cost(lambda s, f, size=size: s["nnz"] > size)
        )
    for saved in [200000, 400000, 1000000]:
        rows.append(
            [f"list when it leaves out more than {count(saved)} nonzeros"]
            + rule_cost(lambda s, f, saved=saved: s["nnz"] - f["nnz"] > saved)
        )
    table(
        [
            "rule",
            "cases with the slower choice",
            "sum of the extra time [us]",
            "largest extra time [us]",
        ],
        rows,
    )

    # per-matrix loss at the recommended thresholds
    print(
        "Per matrix at the proposed thresholds: CPU / GPU time in microseconds at the threshold, then the engine "
        "that is the slower one there, the time lost and the factor. A = activities only (measured on both "
        "sides), S = step with the GPU model."
    )
    print()
    sums = {}
    for label, thr, kappa in [
        (
            f"W = nnz + {REC['kappa']} * rows = {count(REC['frontier_work'])}",
            REC["frontier_work"],
            REC["kappa"],
        ),
        (
            f"T = {count(REC['frontier_nnz'])} frontier nonzeros",
            REC["frontier_nnz"],
            None,
        ),
    ]:
        rows = []
        for s in sweeps:
            row = [s["name"], s["band"]]
            for key in ["A cold", "S cold", "S warm"]:
                cpu, gpu = next((c, g) for k, _, c, g in scen if k == key)
                loss, _, factor, slow, x, c, g = losses_at(
                    [s], cpu, gpu, thr, kappa
                )[0]
                sums.setdefault((key, label), []).append(
                    (loss, s["name"], factor, slow)
                )
                if len(row) == 2:
                    row.append(count(x))
                row.append(
                    f"{fmt(c * 1000)} / {fmt(g * 1000)}; {slow} +{fmt(loss)} (x{fmt(factor)})"
                )
            rows.append(row)
        print(f"At {label}:")
        print()
        table(
            [
                "matrix",
                "band",
                "frontier nnz at the threshold",
                "A cold: CPU / GPU; slower engine, loss",
                "S cold",
                "S warm",
            ],
            rows,
            ["l", "l"] + ["r"] * 4,
        )
    rows = []
    for (key, label), v in sums.items():
        worst = max(v)
        worst_f = max(v, key=lambda x: x[2])
        rows.append(
            [
                key,
                label,
                fmt(statistics.median([x[0] for x in v])),
                fmt(sum(x[0] for x in v)),
                f"{fmt(worst[0])} ({worst[1]}, {worst[3]} too slow)",
                f"{fmt(worst_f[2])} ({worst_f[1]})",
                sum(1 for x in v if x[2] > 2),
                sum(1 for x in v if x[3] == "host"),
            ]
        )
    table(
        [
            "scenario",
            "threshold",
            "median loss [us]",
            "sum of losses [us]",
            "largest loss",
            "largest factor",
            "matrices with factor > 2",
            "matrices where the host is the slower one",
        ],
        rows,
        ["l", "l"] + ["r"] * 6,
    )

    # what HiGHS's own propagation sweeps look like (probing)
    print(
        "For scale: the propagation sweeps of HiGHS's probing on the same instances (diag/summary.json, "
        "DIAG probing: prop_sweeps = passes over the row queue, prop_rows = rows taken from the queue, "
        "prop_entries = row entries scanned, activity_entries = column entries visited for activity updates). "
        "These are HiGHS's sweeps, one probe at a time, not the sweeps of the planned presolve."
    )
    print()
    rows = []
    below = [0, 0, 0]
    total = 0
    for inst in diag["instances"]:
        p = inst["diag"].get("probing")
        p = p[0] if isinstance(p, list) and p else p
        if not p or not p.get("prop_sweeps"):
            rows.append(
                [
                    inst["name"],
                    inst["band"],
                    count(p.get("probed", 0)) if p else 0,
                    0,
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                ]
            )
            continue
        sw = p["prop_sweeps"]
        rows_per = p["prop_rows"] / sw
        ent_per = p["prop_entries"] / sw
        work_per = p["probing_work"] / sw
        w_unit = ent_per + REC["kappa"] * rows_per
        total += 1
        below[0] += ent_per < REC["frontier_nnz"]
        below[1] += work_per < REC["frontier_nnz"]
        below[2] += w_unit < REC["frontier_work"]
        rows.append(
            [
                inst["name"],
                inst["band"],
                count(p["probed"]),
                count(sw),
                fmt(rows_per),
                fmt(ent_per),
                fmt(work_per),
                count(p["max_probe_work"]),
                fmt(w_unit),
            ]
        )
    table(
        [
            "instance",
            "band",
            "probes",
            "sweeps",
            "rows per sweep",
            "row entries per sweep",
            "probing work per sweep",
            "largest work of one probe",
            f"W per sweep (entries + {REC['kappa']} rows)",
        ],
        rows,
        ["l", "l"] + ["r"] * 7,
    )
    print(
        f"Instances with sweeps: {total}. Mean row entries per sweep below {count(REC['frontier_nnz'])}: {below[0]}; "
        f"mean probing work per sweep below {count(REC['frontier_nnz'])}: {below[1]}; mean W per sweep below {count(REC['frontier_work'])}: {below[2]}."
    )
    print()
    return cross, host_model


def fit_two(x1, x2, y):
    """Least squares on relative residuals for y = a * x1 + b * x2."""
    s11 = sum((a / t) ** 2 for a, t in zip(x1, y))
    s22 = sum((b / t) ** 2 for b, t in zip(x2, y))
    s12 = sum(a * b / (t * t) for a, b, t in zip(x1, x2, y))
    r1 = sum(a / t for a, t in zip(x1, y))
    r2 = sum(b / t for b, t in zip(x2, y))
    det = s11 * s22 - s12 * s12
    return (r1 * s22 - r2 * s12) / det, (s11 * r2 - s12 * r1) / det


# ----------------------------------------------------------------------------------------------
# section 3: hand-over on x86


def mirror_bytes(nnz, rows_cols, state_bytes, csc=False):
    return BYTES_PER_NNZ * nnz * (2 if csc else 1) + state_bytes * rows_cols


def solve_bytes(fixed, target_us):
    """Number of bytes whose pinned device-to-host copy takes target_us (bisection)."""
    lo, hi = 1.0, 1e12
    if fixed.d2h_pinned(lo) >= target_us:
        return 0.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if fixed.d2h_pinned(mid) < target_us:
            lo = mid
        else:
            hi = mid
    return lo


def section_handover(sweeps, fixed, stock):
    print("## 3. Hand-over between the engines on x86")
    print()
    print(
        "Measured primitives used below (fixed_costs.json, medians in microseconds):"
    )
    print()
    rows = []
    for label, series in [
        (
            "device_to_host copy, pinned [by bytes]",
            fixed.series("handover_copy", "device_to_host", "pinned"),
        ),
        (
            "host_to_device copy, pinned [by bytes]",
            fixed.series("handover_copy", "host_to_device", "pinned"),
        ),
        (
            "host_to_device copy, pageable [by bytes]",
            fixed.series("handover_copy", "host_to_device", "pageable"),
        ),
        (
            "gather_rows_to_host warp_per_row [by entries]",
            fixed.series("gather", "gather_rows_to_host", "warp_per_row"),
        ),
        (
            "gather_rows_to_host thread_per_row_block_32 [by entries]",
            fixed.series(
                "gather", "gather_rows_to_host", "thread_per_row_block_32"
            ),
        ),
        (
            "frontier_upload [by frontier rows]",
            fixed.series("gather", "frontier_upload", ""),
        ),
        (
            "dirty list to host, frontier_emit_atomic [by records]",
            fixed.series(
                "dirty_list", "device_to_host", "frontier_emit_atomic"
            ),
        ),
        (
            "dirty list to host, flag_scan_ordered [by records]",
            fixed.series("dirty_list", "device_to_host", "flag_scan_ordered"),
        ),
        (
            "dirty list to device, pinned_copy_scatter [by records]",
            fixed.series(
                "dirty_list", "host_to_device", "pinned_copy_scatter"
            ),
        ),
        (
            "cudaHostAlloc [by bytes]",
            fixed.series("allocation", "cudaHostAlloc", ""),
        ),
    ]:
        rows.append(
            [label, "; ".join(f"{count(n)}: {fmt(us)}" for n, us in series)]
        )
    table(["primitive", "size: microseconds"], rows, ["l", "l"])
    um = fixed.fc["single"]
    print(
        "Managed memory on this x86 node, 64 MB (unified_memory): host touch after a kernel "
        f"{fmt(um[('unified_memory', 'host_touch_after_kernel', 'managed')])} us, kernel touch after the host "
        f"{fmt(um[('unified_memory', 'kernel_touch_after_host', 'managed')])} us; a pinned device-to-host copy of 64 MB is "
        f"{fmt(fixed.d2h_pinned(64 * MIB))} us (interpolated between the 16 MB and 256 MB copies) and the host loop on pinned memory {fmt(um[('unified_memory', 'host_touch_reference', 'pinned')])} us. "
        "The managed figure contains that host loop, so like for like it is "
        f"{fmt(um[('unified_memory', 'host_touch_after_kernel', 'managed')])} us against "
        f"{fmt(fixed.d2h_pinned(64 * MIB) + um[('unified_memory', 'host_touch_reference', 'pinned')])} us for copy + loop "
        f"(factor {fmt(um[('unified_memory', 'host_touch_after_kernel', 'managed')] / (fixed.d2h_pinned(64 * MIB) + um[('unified_memory', 'host_touch_reference', 'pinned')]))})."
    )
    print()

    print(
        f"Assumptions: a mirror holds {BYTES_PER_NNZ} bytes per nonzero (4-byte index + 8-byte value of the CSR; the row "
        f"offsets are counted in the per-row figure) plus {STATE_BYTES_LOW} or {STATE_BYTES_HIGH} bytes per row and per column "
        "(bounds, sides, activities, counters). 'with CSC' doubles the per-nonzero part. A refresh is one pinned "
        "device-to-host copy of that many bytes. A gather is gather_rows_to_host warp_per_row plus frontier_upload "
        "at the same benchmark size. Break-even count = refresh time / gather time: the number of row "
        "fetches between two rebuilds from which the mirror is cheaper."
    )
    print()
    rows = []
    small = fixed.gather_total(1000)
    for s in sweeps:
        rc = s["rows"] + s["cols"]
        b_lo = mirror_bytes(s["nnz"], rc, STATE_BYTES_LOW)
        b_hi = mirror_bytes(s["nnz"], rc, STATE_BYTES_HIGH)
        b_csc = mirror_bytes(s["nnz"], rc, STATE_BYTES_HIGH, True)
        r_lo, r_hi, r_csc = (
            fixed.d2h_pinned(b_lo),
            fixed.d2h_pinned(b_hi),
            fixed.d2h_pinned(b_csc),
        )
        f1, f10 = s["frontier"]["1"], s["frontier"]["10"]
        g1, g10 = fixed.gather_total(f1["nnz"]), fixed.gather_total(f10["nnz"])
        rebuild = s["rebuild"]["total: compact CSR + radix sort"] * 1000.0
        rows.append(
            [
                s["name"],
                s["band"],
                count(s["nnz"]),
                count(rc),
                f"{fmt(b_lo / MIB)} / {fmt(b_hi / MIB)}",
                f"{fmt(r_lo)} / {fmt(r_hi)}",
                fmt(r_csc),
                fmt(r_hi / small),
                f"{fmt(g1)} -> {fmt(r_hi / g1)}",
                f"{fmt(g10)} -> {fmt(r_hi / g10)}",
                fmt(fixed.gather_total(s["nnz"]) / r_hi),
                fmt(fixed.host_alloc(b_hi) / 1000.0),
                fmt(rebuild),
                f"{100 * r_hi / rebuild:.0f}%",
            ]
        )
    table(
        [
            "matrix",
            "band",
            "nnz",
            "rows + cols",
            "mirror MiB (16 / 32 B state)",
            "refresh us (16 / 32 B)",
            "refresh us with CSC (32 B)",
            f"break-even count, gathers of <= 1,000 entries ({fmt(small)} us)",
            "gather of the 1% frontier us -> break-even count",
            "gather of the 10% frontier us -> break-even count",
            "gather of all rows / refresh",
            "cudaHostAlloc of the mirror ms (one-off)",
            "GPU rebuild us (compact CSR + radix sort)",
            "refresh / rebuild",
        ],
        rows,
        ["l", "l"] + ["r"] * 12,
    )

    ratios = sorted((s["rows"] + s["cols"]) / s["nnz"] for s in sweeps)
    rho = quantile(ratios, 0.5)
    print(
        f"As a function of the matrix size, with rows + cols = {fmt(rho)} * nnz (median of the 32 matrices; "
        f"quartiles {fmt(quantile(ratios, 0.25))} and {fmt(quantile(ratios, 0.75))}, range {fmt(ratios[0])} to {fmt(ratios[-1])}) "
        f"and {STATE_BYTES_HIGH} bytes of state per row and column:"
    )
    print()
    rows = []
    for nnz in [5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 5e7, 2.6e8]:
        b = mirror_bytes(nnz, rho * nnz, STATE_BYTES_HIGH)
        r = fixed.d2h_pinned(b)
        # frontier size whose gather costs one refresh
        lo, hi = 1.0, nnz * 100
        for _ in range(100):
            mid = math.sqrt(lo * hi)
            if fixed.gather_total(mid) < r:
                lo = mid
            else:
                hi = mid
        rows.append(
            [
                count(nnz),
                fmt(b / MIB),
                fmt(r),
                fmt(1000 * r / nnz),
                fmt(r / fixed.gather_total(1e3)),
                fmt(r / fixed.gather_total(1e4)),
                fmt(r / fixed.gather_total(1e5)),
                fmt(r / fixed.gather_total(1e6)),
                "no gather is that cheap"
                if r <= fixed.gather_total(1.0)
                else f"{count(lo)} ({100 * lo / nnz:.1f}% of the nonzeros)"
                if lo < nnz
                else "more than the matrix",
                fmt(fixed.host_alloc(b) / 1000.0),
            ]
        )
    table(
        [
            "nnz",
            "mirror MiB",
            "refresh us",
            "refresh ns per nnz",
            "break-even count: gathers of 1e3 entries",
            "of 1e4",
            "of 1e5",
            "of 1e6",
            "one gather of this many entries costs one refresh",
            "cudaHostAlloc ms",
        ],
        rows,
        ["r"] * 10,
    )

    print(
        "Mirror size at which one refresh costs as much as n gathers of at most 1,000 entries "
        f"({fmt(small)} us each), and the matrix size that corresponds to it (rows + cols = {fmt(rho)} * nnz):"
    )
    print()
    rows = []
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        b = solve_bytes(fixed, n * small)
        per_lo = BYTES_PER_NNZ + STATE_BYTES_LOW * rho
        per_hi = BYTES_PER_NNZ + STATE_BYTES_HIGH * rho
        per_csc = 2 * BYTES_PER_NNZ + STATE_BYTES_HIGH * rho
        rows.append(
            [
                n,
                fmt(n * small),
                fmt(b / MIB),
                count(b / per_lo),
                count(b / per_hi),
                count(b / per_csc),
            ]
        )
    table(
        [
            "n",
            "n gathers us",
            "mirror MiB",
            "nnz (16 B state)",
            "nnz (32 B state)",
            "nnz (32 B state, CSR and CSC)",
        ],
        rows,
        ["r"] * 6,
    )

    print(
        "Mutable state alone (bounds and activities of every row and column) copied whole, against a dirty "
        "list to the host. The flag-scan lists were measured on a state array of 1e7 elements only, so their "
        "fixed part is an upper bound for smaller models."
    )
    print()
    emit = fixed.series("dirty_list", "device_to_host", "frontier_emit_atomic")
    ordered = fixed.series("dirty_list", "device_to_host", "flag_scan_ordered")
    rows = []
    for group, test in [
        ("A1", None),
        ("A2", None),
        ("A3", None),
        ("B1", None),
        ("B2", None),
        ("B3", None),
        ("C", None),
    ]:
        ms = [s for s in sweeps if s["band"] == group]
        b_lo = [STATE_BYTES_LOW * (s["rows"] + s["cols"]) for s in ms]
        b_hi = [STATE_BYTES_HIGH * (s["rows"] + s["cols"]) for s in ms]
        c_hi = [fixed.d2h_pinned(b) for b in b_hi]
        # records whose emit list costs as much as the whole copy
        ks = []
        for c in c_hi:
            lo, hi = 1.0, 1e9
            if fixed.dirty_d2h(lo) >= c:
                ks.append(0.0)
                continue
            for _ in range(100):
                mid = math.sqrt(lo * hi)
                if fixed.dirty_d2h(mid) < c:
                    lo = mid
                else:
                    hi = mid
            ks.append(lo)
        rows.append(
            [
                group,
                len(ms),
                f"{fmt(min(b_lo) / MIB)} to {fmt(max(b_hi) / MIB)}",
                f"{fmt(min(fixed.d2h_pinned(b) for b in b_lo))} to {fmt(max(c_hi))}",
                f"{count(min(ks))} to {count(max(ks))}",
                sum(1 for c in c_hi if c <= emit[0][1]),
                sum(1 for c in c_hi if c <= ordered[0][1]),
            ]
        )
    b_emit = solve_bytes(fixed, emit[0][1])
    b_ord = solve_bytes(fixed, ordered[0][1])
    print(
        f"A pinned copy costs as much as the smallest emit list ({fmt(emit[0][1])} us) at {fmt(b_emit / MIB)} MiB, "
        f"that is {count(b_emit / STATE_BYTES_HIGH)} rows + columns at {STATE_BYTES_HIGH} bytes or {count(b_emit / STATE_BYTES_LOW)} at "
        f"{STATE_BYTES_LOW}; as much as the smallest ordered flag-scan list ({fmt(ordered[0][1])} us) at {fmt(b_ord / MIB)} MiB."
    )
    print()
    table(
        [
            "band",
            "matrices",
            "state MiB (16 B low to 32 B high)",
            "whole copy us (low to high)",
            "records at which an emit list costs the whole copy (32 B)",
            f"matrices: whole copy <= smallest emit list ({fmt(emit[0][1])} us)",
            f"matrices: whole copy <= smallest ordered flag-scan list ({fmt(ordered[0][1])} us)",
        ],
        rows,
        ["l"] + ["r"] * 6,
    )


# ----------------------------------------------------------------------------------------------
# section 4: fixed cost of a GPU round against HiGHS's presolve time


def section_fixed(sweeps, fixed, stock, diag, stage1=None):
    print(
        "## 4. GPU fixed cost per propagation round against HiGHS's presolve time"
    )
    print()
    print(
        f"From fixed_costs.json: one launch + synchronisation {fmt(fixed.launch_sync)} us; ten launches + one "
        f"synchronisation {fmt(fixed.ten_launches)} us, so each further launch {fmt(fixed.per_launch)} us; launch without "
        f"synchronisation (enqueue only) {fmt(fixed.launch_only)} us; kernel + 8-byte read-back to pinned memory "
        f"{fmt(fixed.readback)} us, so the read-back adds {fmt(fixed.readback_extra)} us to a chain that is synchronised anyway."
    )
    print()
    rows = []
    for k in LAUNCH_COUNTS:
        rows.append(
            [
                k,
                fmt(fixed.round_fixed(k)),
                fmt(fixed.round_fixed_separate(k)),
                fmt(fixed.round_fixed(k) + fixed.copy_if(1e3)),
                fmt(fixed.round_fixed(k) + fixed.copy_if(1e5)),
                fmt(fixed.round_fixed(k) + fixed.copy_if(1e6)),
            ]
        )
    table(
        [
            "k launches",
            "k launches + 1 sync + 1 read-back [us]",
            "the same with the read-back as its own launch + sync [us]",
            "+ copy_if over 1e3 flags",
            "+ copy_if over 1e5 flags",
            "+ copy_if over 1e6 flags",
        ],
        rows,
        ["r"] * 6,
    )
    ci = fixed.series("library", "thrust::copy_if<u64>", "rmm_pool")
    cm = fixed.series("library", "thrust::copy_if<u64>", "cuda_malloc")
    print(
        "thrust::copy_if with an rmm pool: "
        + "; ".join(f"{count(n)}: {fmt(us)} us" for n, us in ci)
        + ". With cudaMalloc temporaries: "
        + "; ".join(f"{count(n)}: {fmt(us)}" for n, us in cm)
        + "."
    )
    floor = sorted(s["frontier"]["0.01"]["comp"] * 1000 for s in sweeps)
    comp_only = sorted(
        s["frontier"]["0.01"]["compaction"] * 1000 for s in sweeps
    )
    print(
        f"Measured in the sweeps for comparison: compaction only (two cub::DeviceSelect::Flagged and one copy of two counts) "
        f"{fmt(comp_only[0])} to {fmt(comp_only[-1])} us, median {fmt(statistics.median(comp_only))}; compaction + H/S/A/comp on the "
        f"0.01% frontier {fmt(floor[0])} to {fmt(floor[-1])} us, median {fmt(statistics.median(floor))}. "
        f"The model for k = 4 with one copy_if over the rows gives {fmt(fixed.round_fixed(4) + fixed.copy_if(1e3))} to "
        f"{fmt(fixed.round_fixed(4) + fixed.copy_if(1e6))} us."
    )
    print()

    by_name = {s["name"]: s for s in sweeps}
    probing = {}
    for inst in diag["instances"]:
        p = inst["diag"].get("probing")
        probing[inst["name"]] = (
            p[0] if isinstance(p, list) and p else (p or {})
        )
    print(
        "Per instance: round fixed cost = k launches + 1 sync + 1 read-back + one thrust::copy_if (rmm pool) over the "
        "rows. Rounds = HiGHS presolve_core_seconds * share / round cost, for k = 4 / 8 / 16. The last two columns "
        "use measured times instead of the model: the smallest-frontier step (compaction + H/S/A/comp, 0.01% of the "
        "rows) and the best compensated full round."
    )
    print()
    rows = []
    per_inst = []
    for inst in stock["instances"]:
        s = by_name[inst["name"]]
        t = inst["presolve_core_seconds"]
        p = fixed.copy_if(inst["rows"])
        cost = {k: fixed.round_fixed(k) + p for k in LAUNCH_COUNTS}
        step = s["frontier"]["0.01"]["comp"] * 1000.0
        full = s["round"]["best comp"][0] * 1000.0
        rounds = {
            (k, share): t * share * 1e6 / cost[k]
            for k in LAUNCH_COUNTS
            for share in (0.01, 0.1, 1.0)
        }
        per_inst.append(
            {
                "name": inst["name"],
                "band": inst["band"],
                "t": t,
                "nnz": inst["nnz"],
                "rounds": rounds,
                "step": t * 1e6 / step,
                "full": t * 1e6 / full,
                "cost": cost,
            }
        )
        rows.append(
            [
                inst["name"],
                inst["band"],
                count(inst["nnz"]),
                fmt(t),
                fmt(1e9 * t / inst["nnz"]),
                fmt(p),
                " / ".join(fmt(cost[k]) for k in LAUNCH_COUNTS),
                " / ".join(
                    count(math.floor(rounds[(k, 0.01)])) for k in LAUNCH_COUNTS
                ),
                " / ".join(
                    count(math.floor(rounds[(k, 0.1)])) for k in LAUNCH_COUNTS
                ),
                " / ".join(
                    count(math.floor(rounds[(k, 1.0)])) for k in LAUNCH_COUNTS
                ),
                f"{fmt(step)} -> {count(math.floor(t * 1e6 / step))}",
                f"{fmt(full)} -> {count(math.floor(t * 1e6 / full))}",
            ]
        )
    table(
        [
            "instance",
            "band",
            "nnz",
            "HiGHS s",
            "HiGHS ns per nnz",
            "copy_if us",
            "round cost us (k = 4 / 8 / 16)",
            "rounds in 1%",
            "rounds in 10%",
            "rounds in 100%",
            "measured smallest-frontier step us -> steps in 100%",
            "measured full round us -> rounds in 100%",
        ],
        rows,
        ["l", "l"] + ["r"] * 10,
    )

    print(
        "Per band (k = 8; smallest / median / largest over the instances of the band):"
    )
    print()
    rows = []
    for band in BANDS[:-1]:
        v = [x for x in per_inst if x["band"] == band]
        ts = [x["t"] for x in v]

        def three(vals):
            return f"{count(math.floor(min(vals)))} / {count(math.floor(statistics.median(vals)))} / {count(math.floor(max(vals)))}"

        rows.append(
            [
                band,
                len(v),
                f"{fmt(min(ts))} / {fmt(statistics.median(ts))} / {fmt(max(ts))}",
                f"{fmt(min(x['cost'][8] for x in v))} to {fmt(max(x['cost'][8] for x in v))}",
                three([x["rounds"][(8, 0.01)] for x in v]),
                three([x["rounds"][(8, 0.1)] for x in v]),
                three([x["rounds"][(8, 1.0)] for x in v]),
                three([x["full"] for x in v]),
            ]
        )
    table(
        [
            "band",
            "instances",
            "HiGHS s",
            "round cost us",
            "rounds in 1%",
            "rounds in 10%",
            "rounds in 100%",
            "measured full rounds in 100%",
        ],
        rows,
        ["l"] + ["r"] * 7,
    )

    print(
        "Band A instances on which HiGHS needs less than 0.25 s (least room first):"
    )
    print()
    rows = []
    tight = sorted(
        (x for x in per_inst if x["band"].startswith("A") and x["t"] < 0.25),
        key=lambda x: x["t"],
    )
    for x in tight:
        p = probing.get(x["name"], {})
        rows.append(
            [
                x["name"],
                x["band"],
                count(x["nnz"]),
                fmt(x["t"]),
                " / ".join(
                    count(math.floor(x["rounds"][(k, 1.0)]))
                    for k in LAUNCH_COUNTS
                ),
                " / ".join(
                    count(math.floor(x["rounds"][(k, 0.1)]))
                    for k in LAUNCH_COUNTS
                ),
                count(math.floor(x["step"])),
                count(math.floor(x["full"])),
                count(p.get("probed", 0)),
                count(p.get("prop_sweeps", 0)),
            ]
        )
    table(
        [
            "instance",
            "band",
            "nnz",
            "HiGHS s",
            "rounds in 100% (k = 4 / 8 / 16)",
            "rounds in 10%",
            "measured smallest-frontier steps in 100%",
            "measured full rounds in 100%",
            "HiGHS probes",
            "HiGHS propagation sweeps in probing",
        ],
        rows,
        ["l", "l"] + ["r"] * 8,
    )
    rate = sorted((1e9 * x["t"] / x["nnz"], x["name"]) for x in per_inst)
    unit = REC["unit_ns"]
    b = REC["budget_units_per_nnz"]
    print(
        f"HiGHS presolve time per nonzero: smallest {fmt(rate[0][0])} ns ({rate[0][1]}), second {fmt(rate[1][0])} ns ({rate[1][1]}), "
        f"median {fmt(statistics.median([r[0] for r in rate]))} ns, largest {fmt(rate[-1][0])} ns ({rate[-1][1]}). "
        f"A device fixed-cost budget of {b} work units ({b * unit} ns) per model nonzero is {100 * b * unit / rate[0][0]:.0f}% of the smallest rate."
    )
    print()
    rows = []
    for nnz in [50000, 100000, 200000, 500000, 1000000, 5000000]:
        budget_us = nnz * b * unit / 1000.0
        rows.append(
            [
                count(nnz),
                fmt(budget_us / 1000.0),
                " / ".join(
                    count(
                        math.floor(
                            budget_us
                            / (fixed.round_fixed(k) + fixed.copy_if(nnz * 0.1))
                        )
                    )
                    for k in LAUNCH_COUNTS
                ),
                count(math.floor(budget_us / statistics.median(floor))),
            ]
        )
    table(
        [
            "model nnz",
            "budget ms",
            "round fixed costs that fit (k = 4 / 8 / 16, copy_if over nnz / 10 rows)",
            "measured smallest-frontier steps that fit (median step)",
        ],
        rows,
        ["r"] * 4,
    )
    share = sorted(
        (100 * b * unit * x["nnz"] / (x["t"] * 1e9), x["name"])
        for x in per_inst
    )
    print(
        f"That budget as a share of HiGHS's presolve time per instance: largest {fmt(share[-1][0])}% ({share[-1][1]}), then {fmt(share[-2][0])}% ({share[-2][1]}); median {fmt(statistics.median([v[0] for v in share]))}%; smallest {fmt(share[0][0])}% ({share[0][1]})."
    )
    print()
    others = sorted(
        (x for x in per_inst if x["band"].startswith("A") and x["t"] >= 0.25),
        key=lambda x: x["t"],
    )
    print(
        "Band A instances at or above 0.25 s: "
        + ", ".join(f"{x['name']} {fmt(x['t'])} s" for x in others)
        + "."
    )
    print()

    if stage1:
        # The Stage 1 time gate of DESIGN.md 13 is against HiGHS restricted to the same rules, not stock.
        t1 = {
            inst["name"]: inst["presolve_core_seconds"]
            for inst in stage1["instances"]
        }
        off = stage1.get("env", {}).get("HIGHS_DIAG_OFF", "?")
        print(
            "Band A against HiGHS restricted towards the Stage 1 rule set (diag_stage1/summary.json, patched HiGHS "
            f"with HIGHS_DIAG_OFF={off}). The rules of HiGHS's fast loop and initial scan cannot be switched off "
            "(HIGHS_DIAG.md), so this time is an upper bound of a HiGHS limited to the Stage 1 rules. Same round "
            "costs as above."
        )
        print()
        rows = []
        v = sorted(
            (
                x
                for x in per_inst
                if x["band"].startswith("A") and x["name"] in t1
            ),
            key=lambda x: t1[x["name"]],
        )
        for x in v:
            t = t1[x["name"]]
            full = by_name[x["name"]]["round"]["best comp"][0] * 1000.0
            rows.append(
                [
                    x["name"],
                    x["band"],
                    fmt(x["t"]),
                    fmt(t),
                    fmt(t / x["t"]),
                    " / ".join(
                        count(math.floor(t * 1e6 / x["cost"][k]))
                        for k in LAUNCH_COUNTS
                    ),
                    " / ".join(
                        count(math.floor(t * 1e5 / x["cost"][k]))
                        for k in LAUNCH_COUNTS
                    ),
                    count(math.floor(t * 1e6 / full)),
                    f"{100 * b * unit * x['nnz'] / (t * 1e9):.1f}%",
                ]
            )
        table(
            [
                "instance",
                "band",
                "stock HiGHS s",
                "restricted HiGHS s",
                "restricted / stock",
                "rounds in 100% (k = 4 / 8 / 16)",
                "rounds in 10%",
                "measured full rounds in 100%",
                f"budget of {b} units per nnz as a share of the restricted time",
            ],
            rows,
            ["l", "l"] + ["r"] * 7,
        )
        ts = [t1[x["name"]] for x in v]
        print(
            f"Band A, restricted HiGHS: {fmt(min(ts))} / {fmt(statistics.median(ts))} / {fmt(max(ts))} s (smallest / median / largest); "
            f"{sum(1 for t in ts if t < 0.25)} of {len(ts)} instances below 0.25 s, {sum(1 for t in ts if t < 0.06)} below 0.06 s."
        )
        print()
    return per_inst


# ----------------------------------------------------------------------------------------------
# section 5: work units and the numbers behind the defaults


def section_units(sweeps, fixed, host_model, per_inst, cross):
    print(
        "## 5. Cost constants in work units and the numbers behind the recommended defaults"
    )
    print()
    alpha, beta = host_model[("cpu_s_cold", "0.01% to 10%")]
    unit_ns = REC["unit_ns"]
    print(
        f"Work unit: one frontier nonzero of a host step. The fitted cold host step costs {fmt(beta * 1e6)} ns per "
        f"nonzero and {fmt(alpha * 1e6)} ns per row (section 2); the conversion below uses {unit_ns} ns per unit "
        f"and counts a frontier row as {REC['kappa']} units (fitted kappa: {fmt(alpha / beta)} cold over all frontiers). Device constants in units = microseconds * 1000 / {unit_ns}."
    )
    print()
    floor = [s["frontier"]["0.01"]["comp"] * 1000 for s in sweeps]
    one = [s["frontier"]["1"]["comp"] * 1000 for s in sweeps]
    rows = []
    for label, us in [
        ("kernel launch in a chain", fixed.per_launch),
        ("first launch + synchronisation", fixed.launch_sync),
        ("read-back added to a synchronised chain", fixed.readback_extra),
        (
            "read-back as its own launch + copy + synchronisation",
            fixed.readback,
        ),
        (
            "thrust::copy_if (rmm pool), 1e3 to 1e5 elements",
            fixed.copy_if(1e4),
        ),
        (
            "measured frontier step floor: compaction + H/S/A/comp at 0.01% of the rows, median",
            statistics.median(floor),
        ),
        ("the same, largest of the 32 matrices", max(floor)),
        (
            "compaction + H/S/A/comp at 1% of the rows, median",
            statistics.median(one),
        ),
        (
            "dirty list to the host, emit, up to 1e4 records",
            fixed.dirty_d2h(1e3),
        ),
        (
            "dirty list to the host, ordered flag scan, up to 1e4 records",
            fixed.dirty_d2h(1e3, "flag_scan_ordered"),
        ),
        ("dirty list to the device, 1e3 records", fixed.dirty_h2d(1e3)),
        (
            "row gather to the host, up to 1e3 entries, with the frontier upload",
            fixed.gather_total(1e3),
        ),
        ("pinned copy of 1 MiB, device to host", fixed.d2h_pinned(MIB)),
        ("pinned copy of 16 MiB, device to host", fixed.d2h_pinned(16 * MIB)),
    ]:
        rows.append([label, fmt(us), count(us * 1000.0 / unit_ns)])
    table(["device cost", "microseconds", "work units"], rows, ["l", "r", "r"])

    print(
        "Numbers behind the recommended defaults (the values are set in REC at the top of this script):"
    )
    print()
    rows = []
    a = [c[0] for c in cross["A cold"]]
    aw = [c[0] for c in cross["A warm"]]
    sc = [c[0] for c in cross["S cold"]]
    sw = [c[0] for c in cross["S warm"]]
    hc = [c[0] for c in cross["H cold"]]
    hw = [c[0] for c in cross["H warm"]]
    dm = [c[0] for c in cross["D mirror"]]
    dg = [c[0] for c in cross["D gather"]]
    k = REC["kappa"]

    def w(key):
        return [c[0] + k * c[2] for c in cross[key]]

    rows.append(
        [
            f"frontier threshold, W = nnz + {k} rows",
            count(REC["frontier_work"]),
            f"geomean crossover in W: A cold {count(geomean(w('A cold')))}, A warm {count(geomean(w('A warm')))}, S cold {count(geomean(w('S cold')))}, S warm {count(geomean(w('S warm')))}; smallest-loss thresholds in section 2",
        ]
    )
    rows.append(
        [
            "the same without the row term, frontier nonzeros",
            count(REC["frontier_nnz"]),
            f"median crossover A cold {count(statistics.median(a))}, A warm {count(statistics.median(aw))}, S cold {count(statistics.median(sc))}, S warm {count(statistics.median(sw))}",
        ]
    )
    rows.append(
        [
            "host owns the state: the same threshold",
            count(REC["frontier_work"]),
            f"H cold: geomean in W {count(geomean(w('H cold')))}, median nnz {count(statistics.median(hc))}; H warm: geomean in W {count(geomean(w('H warm')))}, median nnz {count(statistics.median(hw))}; smallest-loss thresholds in section 2",
        ]
    )
    rows.append(
        [
            "device owns the state, mirror: host below W",
            count(REC["device_owned_mirror_work"]),
            f"D mirror: geomean in W {count(geomean(w('D mirror')))}; nnz median {count(statistics.median(dm))}, quartiles {count(quantile(dm, 0.25))} and {count(quantile(dm, 0.75))}",
        ]
    )
    rows.append(
        [
            "device owns the state, no mirror: host below W",
            count(REC["device_owned_gather_work"]),
            f"D gather: geomean in W {count(geomean(w('D gather')))}; nnz median {count(statistics.median(dg))}, quartiles {count(quantile(dg, 0.25))} and {count(quantile(dg, 0.75))}",
        ]
    )
    rows.append(
        [
            "full-sweep threshold, matrix nonzeros",
            count(REC["full_sweep_nnz"]),
            "section 1: fitted and per-matrix implied crossovers, all below the smallest measured matrix",
        ]
    )
    rows.append(
        [
            "GPU: frontier list only above this matrix size, nonzeros",
            count(REC["gpu_list_min_nnz"]),
            "section 2: rules for the choice between the list and all rows",
        ]
    )
    rows.append(
        [
            "GPU: all rows instead of a list, share of the nonzeros",
            f"{100 * REC['full_sweep_fraction']:.0f}%",
            "not measured between 10% and 100% of the rows",
        ]
    )
    rows.append(
        [
            "mirror threshold, bytes of matrix + state",
            f"{count(REC['mirror_bytes'] / MIB)} MiB",
            f"refresh {fmt(fixed.d2h_pinned(REC['mirror_bytes']))} us = {fmt(fixed.d2h_pinned(REC['mirror_bytes']) / fixed.gather_total(1e3))} gathers of <= 1,000 entries; cudaHostAlloc {fmt(fixed.host_alloc(REC['mirror_bytes']) / 1000)} ms",
        ]
    )
    rows.append(
        [
            "rebuild threshold, matrix nonzeros",
            count(REC["rebuild_nnz"]),
            "section 6: crossover inside the measured range",
        ]
    )
    rows.append(
        [
            "device fixed-cost budget, work units per model nonzero",
            REC["budget_units_per_nnz"],
            "section 4: HiGHS presolve time per nonzero",
        ]
    )
    table(
        ["default", "value", "measured numbers it rests on"],
        rows,
        ["l", "r", "l"],
    )

    mb = REC["mirror_bytes"]
    inside = [
        s["name"]
        for s in sweeps
        if mirror_bytes(s["nnz"], s["rows"] + s["cols"], STATE_BYTES_HIGH)
        <= mb
    ]
    outside = [
        s["name"]
        for s in sweeps
        if mirror_bytes(s["nnz"], s["rows"] + s["cols"], STATE_BYTES_HIGH) > mb
    ]
    print(
        f"Matrices whose mirror (32 B state, CSR only) fits the mirror threshold: {len(inside)} of {len(sweeps)}; above it: "
        + (", ".join(outside) if outside else "none")
        + "."
    )
    inside2 = [
        s["name"]
        for s in sweeps
        if mirror_bytes(
            s["nnz"], s["rows"] + s["cols"], STATE_BYTES_HIGH, True
        )
        <= mb
    ]
    print(
        f"With CSR and CSC in the mirror: {len(inside2)} of {len(sweeps)} fit."
    )
    print()


# ----------------------------------------------------------------------------------------------
# section 6: rebuild


def section_rebuild(sweeps):
    print(
        "## 6. Rebuild: GPU against one CPU thread (engine threshold of DESIGN.md 2.3)"
    )
    print()
    print(
        "Milliseconds. GPU totals: compact CSR followed by the CSC method. Deterministic methods: radix sort, "
        "scatter + segmented sort, and compaction of the old CSC (deletions only). CPU: compaction and a counting "
        "transpose. 20% of the rows and of the columns are deleted in every run."
    )
    print()
    names = {
        "radix": "total: compact CSR + radix sort",
        "seg": "total: compact CSR + scatter + segmented sort",
        "csc": "total: compact CSR + compact CSC",
        "cpu": "cpu: compact CSR + counting transpose",
    }
    rows = []
    for s in sweeps:
        r = {k: s["rebuild"][v] for k, v in names.items()}
        rows.append(
            [
                s["name"],
                s["band"],
                count(s["nnz"]),
                fmt(r["radix"]),
                fmt(r["seg"]),
                fmt(r["csc"]),
                fmt(r["cpu"]),
                fmt(r["cpu"] / r["radix"]),
                fmt(r["cpu"] / r["csc"]),
            ]
        )
    table(
        [
            "matrix",
            "band",
            "nnz",
            "GPU radix sort",
            "GPU scatter + segmented sort",
            "GPU compact CSC",
            "CPU",
            "CPU / GPU radix sort",
            "CPU / GPU compact CSC",
        ],
        rows,
        ["l", "l"] + ["r"] * 7,
    )
    nnz = [s["nnz"] for s in sweeps]
    idx_a = [i for i, s in enumerate(sweeps) if s["band"].startswith("A")]
    rows = []
    for key, label in [
        ("radix", "GPU compact CSR + radix sort"),
        ("csc", "GPU compact CSR + compact CSC"),
    ]:
        g = [s["rebuild"][names[key]] for s in sweeps]
        c = [s["rebuild"][names["cpu"]] for s in sweeps]
        slower = [s["name"] for s, gi, ci in zip(sweeps, g, c) if gi > ci]
        for subset, idx in [
            ("all 32", list(range(len(sweeps)))),
            ("band A", idx_a),
        ]:
            ag, bg = fit_affine(
                [nnz[i] for i in idx], [g[i] for i in idx], True
            )
            bc = geomean([c[i] / nnz[i] for i in idx])
            n = crossing(ag, bg, 0.0, bc)
            rows.append(
                [
                    label,
                    subset,
                    fmt(ag * 1000),
                    fmt(bg * 1e6),
                    fmt(bc * 1e6),
                    count(n) if n else "none",
                    len(slower),
                    ", ".join(slower) if subset == "all 32" else "",
                ]
            )
    table(
        [
            "GPU method",
            "matrices",
            "GPU a [us]",
            "GPU b [ns/nnz]",
            "CPU b [ns/nnz], proportional",
            "crossover [nnz]",
            "matrices where the GPU is slower",
            "which",
        ],
        rows,
        ["l", "l", "r", "r", "r", "r", "r", "l"],
    )


# ----------------------------------------------------------------------------------------------
# main


def main():
    testbed = json.loads(TESTBED.read_text())
    bands = {e["name"]: e["band"] for e in testbed["main"]}
    sweeps = load_sweeps(bands)
    fixed = Fixed(load_fixed())
    stock = json.loads((STAGE0 / "stock" / "summary.json").read_text())
    diag = json.loads((STAGE0 / "diag" / "summary.json").read_text())
    stage1_path = STAGE0 / "diag_stage1" / "summary.json"
    stage1 = (
        json.loads(stage1_path.read_text()) if stage1_path.exists() else None
    )

    print("# Stage 0 engine analysis")
    print()
    sysinfo = fixed.fc["system"]
    print(
        f"Sweep files: {len(sweeps)} matrices, GPU {sorted(set(s['gpu'] for s in sweeps))}, "
        f"quick mode in {sum(1 for s in sweeps if s['quick'])}, checks failed in {sum(1 for s in sweeps if not s['verified'])}, "
        f"fewest repetitions of any measurement {min(s['min_reps'] for s in sweeps)}."
    )
    print(
        f"Fixed costs: GPU {sysinfo['gpu_name']}, CPU {sysinfo['cpu_model']}; other GPU processes observed: "
        f"{sysinfo['other_gpu_processes_observed']}; other CPU load observed: {sysinfo['other_cpu_load_observed']} "
        f"(1-minute load average before the run: {sysinfo['cpu_loadavg_before'].split()[0]})."
    )
    by_band = {}
    for s in sweeps:
        by_band.setdefault(s["band"], []).append(s["name"])
    print(
        "Matrices per band: "
        + ", ".join(f"{b} {len(by_band.get(b, []))}" for b in BANDS)
        + "."
    )
    print(
        f"HiGHS baseline: {len(stock['instances'])} instances in stock/summary.json, {len(diag['instances'])} in diag/summary.json."
    )
    print()

    section_full(sweeps, fixed)
    cross, host_model = section_frontier(sweeps, fixed, diag)
    section_handover(sweeps, fixed, stock)
    per_inst = section_fixed(sweeps, fixed, stock, diag, stage1)
    section_units(sweeps, fixed, host_model, per_inst, cross)
    section_rebuild(sweeps)


if __name__ == "__main__":
    main()
