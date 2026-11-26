#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 kernel analysis: which variants should the propagation sweep use.

Reads, relative to this file:

    ../results/stage0/sweeps/<name>.json   micro-benchmark of the sweep kernels on 32 matrices
                                           (variant names: ../microbench/README_sweeps.md)
    ../results/stage0/fixed_costs.json     kernel launch and synchronise cost on the same GPU
    ../testbed/testbed.json                band of every matrix

and prints markdown tables:

    0  the runs (repetitions, checks), the structure of the matrices, the spread of the timings
    1  activity kernels: time and ns per nonzero of every variant on every matrix; split against
       paired layout, atomic against reduced counts, compensated against plain sums, and the four
       schedulings against each other, one factor at a time; the one-block-per-row scheme (C) by
       matrix structure; a cost model of every scheduling by structure; what the run's own checks
       say about the summation order and about compensated against plain results
    2  candidate kernels: the same for scheduling, layout, gated against unconditional lo loads
    3  full round: best and worst combination, ns per nonzero by size, fits of ms against
       nonzeros, matrices far from the fit, every fixed combination scored against the best
       combination of each matrix (also by band), scheduling chosen by a rule, one CPU thread
       for scale, the frontier-restricted activity sweep against the full sweep
    4  rebuild: the CSC methods against each other and against one CPU thread, fits
    5  the numbers behind each choice (constants CHOICE_* below) and where each choice loses

Conventions.

- A comparison "X against Y" is the ratio ms_median(X) / ms_median(Y) on one matrix, so a value
  below 1 means X is faster. "geomean" is the geometric mean of the ratio over the matrices.
- "X faster" counts the matrices where Y takes more than 3% longer than X (ratio below 1/1.03)
  AND the two measurements are apart by more than their own spread (ms_p90 of X below ms_p10 of
  Y). "Y faster" is the mirror image. "inside spread" counts the matrices where the medians
  differ by more than 3% but the p10..p90 ranges overlap; "within 3%" is the rest.
- That test only says that a difference repeats inside one process. It does not say that the
  difference comes from the factor that is varied: section 1.3 has a null comparison (two
  variants that do the same work at run time and differ only in the compiled template instance)
  in which the test still reports differences of up to 12%, and in section 2.5 the compensated
  candidate kernel, which does strictly more arithmetic, is reported faster than the plain one.
  Ratios inside that band are not evidence for or against a variant.
- Fits are ms = a + b * nnz (and variants with more terms). "ols" minimises the squared error in
  ms, so the largest matrices decide it; "rel" minimises the squared relative error, so every
  matrix counts the same. b is printed in ns per nonzero, a in microseconds.
- A time summed from two separately timed kernels (activity + candidates) is marked "sum of
  parts"; section 3.2 compares such sums with the rounds that were timed as one unit.
- "entries per block thread" of the rows (columns) of a matrix: the number of entries one thread
  of a 256-thread block walks on the longest row (column) above 64 entries. Without a row above
  8192 entries it is the longest row / 256. With one, the chunk kernel walks 8192 / 256 = 32 and
  the block kernel walks at least the mean length of the 65..8192 class / 256; the sum of the
  two is used, which is a lower bound because the file gives only the mean of that class.

No arguments. Python 3 standard library only. Nothing is written.
"""

import collections
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE0 = HERE.parent / "results" / "stage0"
SWEEPS = STAGE0 / "sweeps"
FIXED = STAGE0 / "fixed_costs.json"
TESTBED = HERE.parent / "testbed" / "testbed.json"

MARGIN = 1.03
OFF_FIT = 1.3
ARITH = ("plain", "comp")
LIGHT = ("0-2", "3-4", "5-8", "9-16", "17-32", "33-64")
LIGHT_MAX = {
    "none": 0,
    "0-2": 2,
    "3-4": 4,
    "5-8": 8,
    "9-16": 16,
    "17-32": 32,
    "33-64": 64,
}
BLOCK = "65-8192"
CHUNK = ">8192"
BANDS = ("A1", "A2", "A3", "B1", "B2", "B3", "C")

# The combination scored in section 5. Set after reading sections 1 to 4; change and re-run to
# score another one.
CHOICE_LAYOUT = "S"
CHOICE_COUNT = "A"
CHOICE_ROWS = "B"
CHOICE_COLS = "B"
CHOICE_LO = "gated"

REB_CSR = "compact CSR"
REB_RADIX = "CSC: stable radix sort"
REB_UNORD = "CSC: histogram + scan + scatter, unordered"
REB_SEG = "CSC: histogram + scan + scatter + segmented sort"
REB_OLD = "CSC: compaction of the old CSC"
REB_T_RADIX = "total: compact CSR + radix sort"
REB_T_SEG = "total: compact CSR + scatter + segmented sort"
REB_T_OLD = "total: compact CSR + compact CSC"
REB_CPU = "cpu: compact CSR + counting transpose"
REB_ALL = (
    REB_CSR,
    REB_RADIX,
    REB_UNORD,
    REB_SEG,
    REB_OLD,
    REB_T_RADIX,
    REB_T_SEG,
    REB_T_OLD,
    REB_CPU,
)
REB_SHORT = {
    REB_CSR: "compact CSR",
    REB_RADIX: "CSC radix",
    REB_UNORD: "CSC scatter unordered",
    REB_SEG: "CSC scatter + seg. sort",
    REB_OLD: "CSC compact old",
    REB_T_RADIX: "total radix",
    REB_T_SEG: "total seg. sort",
    REB_T_OLD: "total compact old",
    REB_CPU: "cpu",
}


# ----------------------------------------------------------------------------------------------
# input


class Matrix:
    def __init__(self, path, band):
        with open(path) as f:
            d = json.load(f)
        self.name = path.stem
        self.band = band
        self.m = d["matrix"]
        self.nnz = self.m["nnz"]
        self.rows = self.m["rows"]
        self.cols = self.m["cols"]
        self.max_row = self.m["max_row_len"]
        self.max_col = self.m["max_col_len"]
        self.res = {
            sec: {e["name"]: e for e in lst}
            for sec, lst in d["results"].items()
        }
        self.round = {}
        for e in d["results"]["round"]:
            kind, combo = e["name"].split(": ", 1)
            a, c = combo.split(" + ")
            self.round[kind] = (a, c, e)
        self.ok = d["verification"]["ok"]
        self.ver = d["verification"]
        self.quick = d["quick"]
        self.gpu = d["gpu"]["name"]
        self.reps = sorted(
            {e["repetitions"] for lst in d["results"].values() for e in lst}
        )
        self.rc = {c["len"]: c for c in self.m["row_classes"]}
        self.cc = {c["len"]: c for c in self.m["col_classes"]}
        # "10% (2035 rows): compaction + H/S/A/comp" -> ("10", "compaction + H/S/A/comp")
        self.frontier = {}
        for e in d["results"]["frontier"]:
            label, kind = e["name"].split(": ", 1)
            self.frontier[(label.split("%")[0], kind)] = e

    def ms(self, sec, name):
        return self.res[sec][name]["ms_median"]

    def ns(self, sec, name):
        return self.res[sec][name]["ms_median"] * 1e6 / self.nnz

    def side(self, which):
        if which == "row":
            return self.rc, "rows", self.rows, self.max_row
        return self.cc, "cols", self.cols, self.max_col

    def share(self, which, classes, field):
        table, key, total, _ = self.side(which)
        if field == "nnz":
            return sum(table[c]["nnz"] for c in classes) / self.nnz
        return sum(table[c][key] for c in classes) / total if total else 0.0

    def count(self, which, classes):
        table, key, _, _ = self.side(which)
        return sum(table[c][key] for c in classes)

    def top_light(self, which):
        table, key, _, _ = self.side(which)
        top = "none"
        for c in LIGHT:
            if table[c][key] > 0:
                top = c
        return top

    def blocks(self, which):
        """256-thread blocks launched for the rows (columns) above 64 entries."""
        table, key, _, _ = self.side(which)
        return (
            table[BLOCK][key]
            + table[CHUNK][key]
            + table[CHUNK]["nnz"] / 8192.0
        )

    def block_thread_entries(self, which):
        """See 'entries per block thread' in the module docstring."""
        table, key, _, longest = self.side(which)
        if table[CHUNK][key] > 0:
            n = table[BLOCK][key]
            return 32.0 + (table[BLOCK]["nnz"] / n / 256.0 if n else 0.0)
        return longest / 256.0 if longest > 64 else 0.0

    def light_thread_entries(self, which):
        """Upper bound of the entries one thread walks when rows (columns) up to 64 entries get
        one thread each (scheme H).
        """
        _, _, _, longest = self.side(which)
        return longest if longest <= 64 else LIGHT_MAX[self.top_light(which)]


def load():
    with open(TESTBED) as f:
        bands = {e["name"]: e["band"] for e in json.load(f)["main"]}
    mats = [
        Matrix(p, bands.get(p.stem, "?"))
        for p in sorted(SWEEPS.glob("*.json"))
    ]
    mats.sort(key=lambda m: (m.nnz, m.name))
    return mats


# ----------------------------------------------------------------------------------------------
# helpers


def gmean(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def num(x, sig=4):
    if x is None or x != x:
        return "-"
    if x == 0:
        return "0"
    digits = sig - 1 - int(math.floor(math.log10(abs(x))))
    return f"{x:.{max(0, digits)}f}"


def pct(x):
    return f"{100.0 * x:.1f}"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def text(*parts):
    print(" ".join(parts) + "\n")


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def lstsq(xs, ys, ws):
    """Weighted least squares by the normal equations; xs is a list of regressor rows."""
    k = len(xs[0])
    a = [[0.0] * (k + 1) for _ in range(k)]
    for row, y, w in zip(xs, ys, ws):
        for i in range(k):
            a[i][k] += w * row[i] * y
            for j in range(k):
                a[i][j] += w * row[i] * row[j]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        for r in range(k):
            if r != i:
                f = a[r][i] / a[i][i]
                for c in range(i, k + 1):
                    a[r][c] -= f * a[i][c]
    return [a[i][k] / a[i][i] for i in range(k)]


def fit(mats, ys, regressors, rel):
    """Fit ys, given in ms. Every regressor is divided by 1e6, so a coefficient is in ms per million units,
    which is ns per unit. Returns the coefficients (constant first, in ms), the fitted values and
    the ratios measured / fitted.
    """
    xs = [[1.0] + [g(m) / 1e6 for g in regressors] for m in mats]
    ws = [1.0 / (y * y) if rel else 1.0 for y in ys]
    coef = lstsq(xs, ys, ws)
    pred = [sum(c * x for c, x in zip(coef, row)) for row in xs]
    ratio = [y / p if p > 0 else float("nan") for y, p in zip(ys, pred)]
    return coef, pred, ratio


def fit_cells(mats, coef, ratio):
    good = [(r, m.name) for r, m in zip(ratio, mats) if r == r]
    lo, hi = min(good), max(good)
    med_err = statistics.median(abs(r - 1.0) for r, _ in good)
    return [
        pct(med_err),
        f"{num(lo[0], 3)} ({lo[1]})",
        f"{num(hi[0], 3)} ({hi[1]})",
    ]


def fit_row(label, mats, ys, regressors, rel):
    coef, _, ratio = fit(mats, ys, regressors, rel)
    return (
        [label, "rel" if rel else "ols", len(mats), num(coef[0] * 1e3)]
        + [num(c) for c in coef[1:]]
        + fit_cells(mats, coef, ratio)
    )


def compare(mats, sec, x, y):
    """Ratio ms(x) / ms(y) per matrix and the class described in the module docstring."""
    per = []
    for m in mats:
        ex, ey = m.res[sec][x], m.res[sec][y]
        r = ex["ms_median"] / ey["ms_median"]
        x_clear = ex["ms_p90"] < ey["ms_p10"]
        y_clear = ey["ms_p90"] < ex["ms_p10"]
        if r < 1.0 / MARGIN:
            cls = "x" if x_clear else "spread"
        elif r > MARGIN:
            cls = "y" if y_clear else "spread"
        else:
            cls = "tie"
        per.append((m.name, r, cls))
    return per


def summary_row(label, per):
    lo = min(per, key=lambda t: t[1])
    hi = max(per, key=lambda t: t[1])
    n = collections.Counter(c for _, _, c in per)
    return [
        label,
        num(gmean(r for _, r, _ in per), 4),
        f"{num(lo[1], 4)} ({lo[0]})",
        f"{num(hi[1], 4)} ({hi[0]})",
        n["x"],
        n["y"],
        n["tie"],
        n["spread"],
    ]


def comparison_block(mats, sec, title, x_name, y_name, pairs):
    """pairs: list of (label, variant x, variant y). Prints the summary and the per-matrix
    ratios; returns {label: per-matrix list}.
    """
    print(
        f"**{title}: ratio {x_name} / {y_name}, below 1 = {x_name} faster.**\n"
    )
    all_per = []
    rows = []
    for label, x, y in pairs:
        per = compare(mats, sec, x, y)
        all_per.append(per)
        rows.append(summary_row(label, per))
    pooled = [r for per in all_per for _, r, _ in per]
    n = collections.Counter(c for per in all_per for _, _, c in per)
    rows.append(
        [
            "all rows pooled",
            num(gmean(pooled), 4),
            num(min(pooled), 4),
            num(max(pooled), 4),
            n["x"],
            n["y"],
            n["tie"],
            n["spread"],
        ]
    )
    table(
        [
            "others fixed at",
            "geomean",
            "best (matrix)",
            "worst (matrix)",
            f"{x_name} faster",
            f"{y_name} faster",
            "within 3%",
            "inside spread",
        ],
        rows,
    )
    print(f"Per matrix, ratio {x_name} / {y_name}:\n")
    table(
        ["matrix"] + [label for label, _, _ in pairs],
        [
            [m.name] + [num(per[i][1], 4) for per in all_per]
            for i, m in enumerate(mats)
        ],
    )
    return {label: per for (label, _, _), per in zip(pairs, all_per)}


def losers(per, limit=8):
    """Matrices where x is slower than y by more than the margin and outside the spread."""
    out = sorted(((r, n) for n, r, c in per if c == "y"), reverse=True)
    if not out:
        return "none"
    s = ", ".join(f"{n} {num(r, 4)}" for r, n in out[:limit])
    return s + (f" and {len(out) - limit} more" if len(out) > limit else "")


def extremes(values, mats):
    i_lo, i_hi = values.index(min(values)), values.index(max(values))
    return (
        f"{num(values[i_lo], 4)} ({mats[i_lo].name})",
        f"{num(values[i_hi], 4)} ({mats[i_hi].name})",
    )


def act_name(s, layout, count, arith):
    return f"{s}/{layout}/{'-' if s == 'T' else count}/{arith}"


def act_variants(arith):
    out = []
    for s in "TBHC":
        for layout in "PS":
            for count in ("-",) if s == "T" else ("R", "A"):
                out.append(act_name(s, layout, count, arith))
    return out


def cand_variants(arith):
    out = []
    for s in "BH":
        if arith == "plain":
            out += [f"{s}/P/plain", f"{s}/S/plain"]
        else:
            out += [f"{s}/P/comp", f"{s}/S/comp/gated", f"{s}/S/comp/always"]
    return out


def cand_name(s, layout, arith, lo="gated"):
    if arith == "plain":
        return f"{s}/{layout}/plain"
    return f"{s}/P/comp" if layout == "P" else f"{s}/S/comp/{lo}"


def combos(arith):
    out = []
    for a in act_variants(arith):
        for c in cand_variants(arith):
            if a.split("/")[1] == c.split("/")[1]:
                out.append((a, c))
    return out


def combo_ms(m, a, c):
    return m.ms("activity", a) + m.ms("candidate", c)


def inf_share(m):
    zero = min(
        m.m["rows_by_infinite_terms_min"][0],
        m.m["rows_by_infinite_terms_max"][0],
    )
    return 1.0 - zero / m.rows


def lo_share(m):
    return max(m.m["rows_nonzero_lo_min"], m.m["rows_nonzero_lo_max"]) / m.rows


# ----------------------------------------------------------------------------------------------
# section 0


def section_inputs(mats):
    print("## 0. Inputs\n")
    gpus = sorted({m.gpu for m in mats})
    reps = sorted({r for m in mats for r in m.reps})
    counts = {
        sec: sorted({len(m.res[sec]) for m in mats}) for sec in mats[0].res
    }
    with open(FIXED) as f:
        fixed = {
            r["name"]: r
            for r in json.load(f)["results"]
            if r["group"] == "launch"
        }
    print(
        f"- {len(mats)} files in {SWEEPS.relative_to(HERE.parent)}; GPU: {', '.join(gpus)}."
    )
    print(
        f"- repetitions per measurement: {reps}; quick runs: {sum(m.quick for m in mats)}; "
        f"verification ok in {sum(bool(m.ok) for m in mats)} of {len(mats)} files."
    )
    print(f"- entries per section (all files): {counts}.")
    print(
        "- fixed_costs.json: empty kernel launch + synchronise "
        f"{num(fixed['empty_kernel_launch_sync']['median_us'])} us, launch without synchronise "
        f"{num(fixed['empty_kernel_launch_no_sync']['median_us'])} us, 10 launches + one "
        f"synchronise {num(fixed['empty_kernel_10_launches_one_sync']['median_us'])} us.\n"
    )

    print("### 0.1 Structure of the matrices (sorted by nonzeros)\n")
    text(
        "Shares are in percent. 'rows <= 2' is the share of rows with at most 2 entries. 'thread",
        "entries' is the 'entries per block thread' of the module docstring, rows / columns.",
        "'inf' is the share of rows with at least one infinite term on the minimum / maximum",
        "side; 'lo' the share of rows whose compensated sum has a nonzero low part (larger of",
        "the two sides).",
    )
    rows = []
    for m in mats:
        inf_min = 1.0 - m.m["rows_by_infinite_terms_min"][0] / m.rows
        inf_max = 1.0 - m.m["rows_by_infinite_terms_max"][0] / m.rows
        rows.append(
            [
                m.name,
                m.band,
                m.rows,
                m.cols,
                m.nnz,
                num(m.nnz / m.rows, 3),
                num(m.nnz / m.cols, 3),
                m.max_row,
                m.max_col,
                pct(m.share("row", ("0-2",), "count")),
                pct(m.share("row", LIGHT, "nnz")),
                pct(m.share("row", (BLOCK,), "nnz")),
                pct(m.share("row", (CHUNK,), "nnz")),
                pct(m.share("col", LIGHT, "nnz")),
                pct(m.share("col", (BLOCK,), "nnz")),
                pct(m.share("col", (CHUNK,), "nnz")),
                f"{num(m.block_thread_entries('row'), 3)} / {num(m.block_thread_entries('col'), 3)}",
                f"{pct(inf_min)} / {pct(inf_max)}",
                pct(lo_share(m)),
            ]
        )
    table(
        [
            "matrix",
            "band",
            "rows",
            "cols",
            "nnz",
            "mean row",
            "mean col",
            "max row",
            "max col",
            "rows <= 2",
            "nnz in rows <= 64",
            "nnz in rows 65-8192",
            "nnz in rows > 8192",
            "nnz in cols <= 64",
            "nnz in cols 65-8192",
            "nnz in cols > 8192",
            "thread entries rows / cols",
            "inf min / max",
            "lo",
        ],
        rows,
    )

    print("### 0.2 Spread of the timings\n")
    print(
        "(ms_p90 - ms_p10) / ms_median over all entries of a section and all matrices.\n"
    )
    rows = []
    for sec in ("activity", "candidate", "round", "rebuild"):
        sp = sorted(
            (e["ms_p90"] - e["ms_p10"]) / e["ms_median"]
            for m in mats
            for e in m.res[sec].values()
        )
        over = sum(1 for s in sp if s > MARGIN - 1.0)
        rows.append(
            [
                sec,
                len(sp),
                pct(statistics.median(sp)),
                pct(sp[int(0.9 * len(sp))]),
                pct(sp[-1]),
                over,
            ]
        )
    table(
        [
            "section",
            "entries",
            "median %",
            "90th percentile %",
            "max %",
            "entries above 3%",
        ],
        rows,
    )


# ----------------------------------------------------------------------------------------------
# section 1


def variant_tables(mats, sec, variants, title):
    print(f"{title}: ms, median\n")
    table(
        ["matrix", "nnz"] + variants,
        [
            [m.name, m.nnz] + [num(m.ms(sec, v)) for v in variants]
            for m in mats
        ],
    )
    print(f"{title}: ns per nonzero, median\n")
    rows = [
        [m.name, m.nnz] + [num(m.ns(sec, v)) for v in variants] for m in mats
    ]
    rows.append(
        ["geomean", ""]
        + [num(gmean(m.ns(sec, v) for m in mats)) for v in variants]
    )
    rows.append(
        ["median", ""]
        + [
            num(statistics.median(m.ns(sec, v) for m in mats))
            for v in variants
        ]
    )
    table(["matrix", "nnz"] + variants, rows)


MODEL_HEADER = [
    "variant",
    "a us",
    "ns per nnz",
    "ns per row or column up to 64",
    "ns per block",
    "us per entry of a block thread",
    "us per entry of a single thread",
    "median abs error %",
    "lowest measured / fit",
    "highest measured / fit",
]


def model_row(mats, sec, variant, which, scheme):
    """Relative fit of one variant's ms on structure terms. scheme B: lane groups up to 64, one
    block above; H: one thread up to 64, one block above; C: one block per row; T: one thread per
    row. Terms that a scheme does not have are printed as '-'.
    """
    ys = [m.ms(sec, variant) for m in mats]
    light = [lambda m: m.count(which, LIGHT)]
    blocks = [lambda m: m.blocks(which)]
    walk = [lambda m: m.block_thread_entries(which) * 1e3]
    if scheme == "B":
        regs, slots = [lambda m: m.nnz] + light + blocks + walk, (1, 2, 3, 4)
    elif scheme == "H":
        single = [lambda m: m.light_thread_entries(which) * 1e3]
        regs, slots = (
            [lambda m: m.nnz] + light + blocks + walk + single,
            (1, 2, 3, 4, 5),
        )
    elif scheme == "C":
        regs = [
            lambda m: m.nnz,
            lambda m: m.rows,
            lambda m: m.max_row / 256.0 * 1e3,
        ]
        slots = (1, 3, 4)
    else:
        regs = [lambda m: m.nnz, lambda m: m.rows, lambda m: m.max_row * 1e3]
        slots = (1, 2, 5)
    coef, _, ratio = fit(mats, ys, regs, True)
    cells = ["-"] * 6
    cells[0] = num(coef[0] * 1e3)
    for c, slot in zip(coef[1:], slots):
        cells[slot] = num(c)
    return [variant] + cells + fit_cells(mats, coef, ratio)


def section_activity(mats):
    print("## 1. Activity kernels\n")
    print(
        "Variant name: scheduling / layout / count / arithmetic (README_sweeps.md section 1).\n"
    )
    print("### 1.1 Every variant on every matrix\n")
    for arith in ARITH:
        variant_tables(
            mats, "activity", act_variants(arith), f"Activity, {arith}"
        )

    out = {}
    print(
        "### 1.2 (a) Layout: split arrays (S) against the paired value (P)\n"
    )
    pairs = []
    for arith in ARITH:
        for s in "TBHC":
            for count in ("-",) if s == "T" else ("R", "A"):
                pairs.append(
                    (
                        f"{s}/./{count}/{arith}",
                        act_name(s, "S", count, arith),
                        act_name(s, "P", count, arith),
                    )
                )
    out["layout"] = comparison_block(
        mats, "activity", "Layout", "S", "P", pairs
    )

    print(
        "### 1.3 (b) Counts: scan and integer atomic (A) against a reduction (R)\n"
    )
    pairs = []
    for arith in ARITH:
        for s in "BHC":
            for layout in "PS":
                pairs.append(
                    (
                        f"{s}/{layout}/./{arith}",
                        act_name(s, layout, "A", arith),
                        act_name(s, layout, "R", arith),
                    )
                )
    out["count"] = comparison_block(
        mats, "activity", "Counts", "A", "R", pairs
    )
    text(
        "A / R in scheduling B by where the nonzeros are. In the lane-group kernel (rows up to 64)",
        "R carries the count through the same shuffles as the sum and A adds to a counter in",
        "shared memory; in the block kernels (rows above 64) R is a second cub::BlockReduce pair.",
    )
    groups = (
        (
            "at least 99% of the nonzeros in rows <= 64",
            lambda m: m.share("row", LIGHT, "nnz") >= 0.99,
        ),
        (
            "at most 1% of the nonzeros in rows <= 64",
            lambda m: m.share("row", LIGHT, "nnz") <= 0.01,
        ),
        ("in between", lambda m: 0.01 < m.share("row", LIGHT, "nnz") < 0.99),
    )
    rows = []
    for label, pick in groups:
        idx = [i for i, m in enumerate(mats) if pick(m)]
        r = [label, len(idx)]
        for key in ("B/P/./plain", "B/S/./plain", "B/P/./comp", "B/S/./comp"):
            v = [out["count"][key][i][1] for i in idx]
            r.append(
                f"{num(gmean(v), 4)} ({num(min(v), 4)} to {num(max(v), 4)})"
            )
        rows.append(r)
    table(
        [
            "matrices",
            "n",
            "B/P/./plain geomean (range)",
            "B/S/./plain",
            "B/P/./comp",
            "B/S/./comp",
        ],
        rows,
    )
    shares = [inf_share(m) for m in mats]
    rows = []
    for key in (
        "B/S/./plain",
        "B/S/./comp",
        "B/P/./plain",
        "C/P/./plain",
        "C/S/./plain",
    ):
        rows.append(
            [
                key,
                num(spearman(shares, [r for _, r, _ in out["count"][key]]), 3),
            ]
        )
    print(
        "Spearman rank correlation of A / R with the share of rows that hold an infinite term:\n"
    )
    table(["others fixed at", "Spearman"], rows)
    i_max = shares.index(max(shares))
    text(
        f"Matrices with at least one row holding an infinite term: {sum(1 for s in shares if s > 0)}",
        f"of {len(mats)}; above half of the rows:",
        ", ".join(
            f"{m.name} {pct(s)}%" for m, s in zip(mats, shares) if s > 0.5
        )
        + ";",
        f"largest share {pct(shares[i_max])}% ({mats[i_max].name}).",
    )
    print(
        "A / R in scheduling B, layout S, on the matrices where more than half of the rows hold an "
        "infinite term, by the number of infinite terms per row (minimum / maximum side):\n"
    )
    rows = []
    for i, m in enumerate(mats):
        if shares[i] <= 0.5:
            continue
        t_min, t_max = (
            m.m["rows_by_infinite_terms_min"],
            m.m["rows_by_infinite_terms_max"],
        )
        rows.append(
            [
                m.name,
                pct(m.share("row", LIGHT, "nnz")),
                f"{pct(t_min[1] / m.rows)} / {pct(t_max[1] / m.rows)}",
                f"{pct(t_min[2] / m.rows)} / {pct(t_max[2] / m.rows)}",
                num(out["count"]["B/S/./plain"][i][1], 4),
                num(out["count"]["B/S/./comp"][i][1], 4),
            ]
        )
    table(
        [
            "matrix",
            "% nnz in rows <= 64",
            "% rows with exactly one infinite term",
            "% rows with two or more",
            "A / R, B/S/./plain",
            "A / R, B/S/./comp",
        ],
        rows,
    )

    print(
        "Null comparison: H with counts A against H with counts R, matrices without a row above 64 entries.\n"
    )
    text(
        "Scheme H folds a row of at most 64 entries with one thread, and the count code of",
        "act_light_kernel (sweeps.cu) runs only for lane groups of two or more lanes; these",
        "matrices launch no block or chunk kernel. The two variants therefore do the same work at",
        "run time and differ only in the template instance that was compiled. A ratio away from 1",
        "in this table is not an effect of the count method; it shows how large a difference between",
        "two variants can be without one.",
    )
    labels = {
        "x": "A faster",
        "y": "R faster",
        "tie": "within 3%",
        "spread": "inside spread",
    }
    rows, null = [], {a: [] for a in ARITH}
    for m in mats:
        if m.max_row > 64:
            continue
        for arith in ARITH:
            for layout in "PS":
                a_name = act_name("H", layout, "A", arith)
                r_name = act_name("H", layout, "R", arith)
                _, r, cls = compare([m], "activity", a_name, r_name)[0]
                null[arith].append((r, cls))
                rows.append(
                    [
                        m.name,
                        m.max_row,
                        f"H/{layout}/./{arith}",
                        num(m.ms("activity", a_name)),
                        num(m.ms("activity", r_name)),
                        num(r, 4),
                        labels[cls],
                    ]
                )
    table(
        [
            "matrix",
            "max row",
            "others fixed at",
            "A ms",
            "R ms",
            "A / R",
            "class by the 3% and spread test",
        ],
        rows,
    )
    for arith in ARITH:
        v = [r for r, _ in null[arith]]
        flagged = sum(1 for _, c in null[arith] if c in ("x", "y"))
        text(
            f"{arith}: A / R from {num(min(v), 4)} to {num(max(v), 4)} over {len(v)} pairs;",
            f"{flagged} of {len(v)} classed as faster outside the spread.",
        )

    print("### 1.4 (c) Arithmetic: compensated (comp) against plain\n")
    pairs = []
    for s in "TBHC":
        for layout in "PS":
            for count in ("-",) if s == "T" else ("R", "A"):
                pairs.append(
                    (
                        f"{s}/{layout}/{count}/.",
                        act_name(s, layout, count, "comp"),
                        act_name(s, layout, count, "plain"),
                    )
                )
    out["arith"] = comparison_block(
        mats, "activity", "Arithmetic", "comp", "plain", pairs
    )
    text("comp / plain of B/S/A by where the nonzeros are:")
    rows = []
    for label, pick in groups:
        idx = [i for i, m in enumerate(mats) if pick(m)]
        v = [out["arith"]["B/S/A/."][i][1] for i in idx]
        d = [
            (
                mats[i].ms("activity", "B/S/A/comp")
                - mats[i].ms("activity", "B/S/A/plain")
            )
            * 1e6
            / mats[i].nnz
            for i in idx
        ]
        rows.append(
            [
                label,
                len(idx),
                num(gmean(v), 4),
                f"{num(min(v), 4)} to {num(max(v), 4)}",
                num(statistics.median(d), 3),
            ]
        )
    table(
        [
            "matrices",
            "n",
            "geomean comp / plain",
            "range",
            "median extra ns per nnz",
        ],
        rows,
    )

    print("### 1.5 (d) Scheduling: T, B, H, C against each other\n")
    print(
        "T has no count variant: the same T measurement is used for both count methods.\n"
    )
    sched_pairs = (
        ("H", "B"),
        ("C", "B"),
        ("C", "H"),
        ("T", "B"),
        ("T", "H"),
        ("T", "C"),
    )
    rows = []
    for x, y in sched_pairs:
        for arith in ARITH:
            for layout in "PS":
                for count in "RA":
                    per = compare(
                        mats,
                        "activity",
                        act_name(x, layout, count, arith),
                        act_name(y, layout, count, arith),
                    )
                    rows.append(
                        [f"{x} / {y}"]
                        + summary_row(f"./{layout}/{count}/{arith}", per)
                    )
    table(
        [
            "ratio",
            "others fixed at",
            "geomean",
            "best (matrix)",
            "worst (matrix)",
            "first faster",
            "second faster",
            "within 3%",
            "inside spread",
        ],
        rows,
    )

    print("Fastest scheduling per matrix, number of matrices:\n")
    rows = []
    for arith in ARITH:
        for layout in "PS":
            for count in "RA":
                win = collections.Counter(
                    min(
                        "TBHC",
                        key=lambda s: m.ms(
                            "activity", act_name(s, layout, count, arith)
                        ),
                    )
                    for m in mats
                )
                rows.append(
                    [f"./{layout}/{count}/{arith}"] + [win[s] for s in "TBHC"]
                )
    table(["others fixed at", "T", "B", "H", "C"], rows)

    for arith in ARITH:
        print(f"Per matrix, layout S, counts A, {arith}: ms and ratios\n")
        rows = []
        for m in mats:
            t = {
                s: m.ms("activity", act_name(s, "S", "A", arith))
                for s in "TBHC"
            }
            rows.append(
                [m.name, m.rows, num(m.nnz / m.rows, 3), m.max_row]
                + [num(t[s]) for s in "TBHC"]
                + [
                    num(t["H"] / t["B"], 4),
                    num(t["C"] / t["B"], 4),
                    num(t["C"] / t["H"], 4),
                    num(t["T"] / t["B"], 4),
                    min(t, key=t.get),
                ]
            )
        table(
            [
                "matrix",
                "rows",
                "mean row",
                "max row",
                "T ms",
                "B ms",
                "H ms",
                "C ms",
                "H / B",
                "C / B",
                "C / H",
                "T / B",
                "fastest",
            ],
            rows,
        )

    print(
        "### 1.6 C (one 256-thread block per row, cuOpt today) by matrix structure\n"
    )
    text(
        "C/P/R/plain is scheme C with the paired layout and reduced counts of cuOpt PR 2372;",
        "B/S/A is lane groups up to 64 entries, split arrays, atomic counts.",
    )
    rows = []
    key = "C/S/A/comp", "B/S/A/comp"
    for m in sorted(
        mats, key=lambda m: m.ms("activity", key[0]) / m.ms("activity", key[1])
    ):
        c_pr = m.ms("activity", "C/P/R/plain")
        rows.append(
            [
                m.name,
                m.rows,
                num(m.nnz / m.rows, 3),
                pct(m.share("row", LIGHT, "count")),
                pct(m.share("row", LIGHT, "nnz")),
                num(
                    m.ms("activity", "C/S/A/plain")
                    / m.ms("activity", "B/S/A/plain"),
                    4,
                ),
                num(
                    m.ms("activity", "C/S/A/comp")
                    / m.ms("activity", "B/S/A/comp"),
                    4,
                ),
                num(
                    m.ms("activity", "C/S/A/comp")
                    / m.ms("activity", "H/S/A/comp"),
                    4,
                ),
                num(c_pr / m.ms("activity", "B/S/A/plain"), 4),
                num(c_pr / m.ms("activity", "B/S/A/comp"), 4),
                num(m.ms("activity", "C/S/A/plain") * 1e6 / m.rows, 4),
                num(m.ms("activity", "C/S/A/comp") * 1e6 / m.rows, 4),
            ]
        )
    table(
        [
            "matrix (sorted by C/B comp)",
            "rows",
            "mean row",
            "% rows <= 64",
            "% nnz in rows <= 64",
            "C / B plain (S/A)",
            "C / B comp (S/A)",
            "C / H comp (S/A)",
            "C/P/R/plain / B/S/A/plain",
            "C/P/R/plain / B/S/A/comp",
            "C/S/A/plain ns per row",
            "C/S/A/comp ns per row",
        ],
        rows,
    )
    for arith in ARITH:
        ratio = [
            m.ms("activity", f"C/S/A/{arith}")
            / m.ms("activity", f"B/S/A/{arith}")
            for m in mats
        ]
        feats = (
            ("rows", [m.rows for m in mats]),
            ("mean row length", [m.nnz / m.rows for m in mats]),
            (
                "share of nnz in rows <= 64",
                [m.share("row", LIGHT, "nnz") for m in mats],
            ),
            ("max row length", [m.max_row for m in mats]),
            ("nnz", [m.nnz for m in mats]),
        )
        text(
            f"Spearman rank correlation of C / B (S/A/{arith}) with:",
            "; ".join(f"{n} {num(spearman(v, ratio), 3)}" for n, v in feats)
            + ".",
        )
    rows = []
    for label, pick in (
        ("mean row length <= 16", lambda m: m.nnz / m.rows <= 16),
        ("mean row length 16 to 64", lambda m: 16 < m.nnz / m.rows <= 64),
        ("mean row length > 64", lambda m: m.nnz / m.rows > 64),
    ):
        sub = [m for m in mats if pick(m)]
        r = [label, len(sub)]
        for num_v, den_v in (
            ("C/S/A/plain", "B/S/A/plain"),
            ("C/S/A/comp", "B/S/A/comp"),
            ("C/P/R/plain", "B/S/A/plain"),
            ("C/P/R/plain", "B/S/A/comp"),
        ):
            v = [
                m.ms("activity", num_v) / m.ms("activity", den_v) for m in sub
            ]
            r.append(
                f"{num(gmean(v), 4)} ({num(min(v), 4)} to {num(max(v), 4)})"
            )
        rows.append(r)
    table(
        [
            "matrices",
            "n",
            "C/S/A/plain / B/S/A/plain geomean (range)",
            "C/S/A/comp / B/S/A/comp",
            "C/P/R/plain / B/S/A/plain",
            "C/P/R/plain / B/S/A/comp",
        ],
        rows,
    )

    print(
        "### 1.7 Cost of every scheduling by structure, and the latency of one thread\n"
    )
    text(
        "Relative fits of the activity time on structure terms (layout S, counts A). 'ns per row or",
        "column up to 64' is per row with at most 64 entries (per row for T); 'ns per block' is per",
        "256-thread block (per row for C); 'us per entry of a block thread' multiplies the entries",
        "per block thread of the module docstring (the longest row / 256 for C); 'us per entry of",
        "a single thread' multiplies the longest row for T and, for H, the upper end of the longest",
        "occupied row class up to 64. The terms are correlated across these 32 matrices, so a",
        "coefficient is a description of this set, not a measured kernel constant.",
    )
    rows = []
    for arith in ARITH:
        for s in "BHCT":
            rows.append(
                model_row(
                    mats, "activity", act_name(s, "S", "A", arith), "row", s
                )
            )
    table(MODEL_HEADER, rows)
    print(
        "T: time divided by the longest row, for matrices whose longest row has 1000 entries or more.\n"
    )
    rows = []
    per_entry = {a: [] for a in ARITH}
    for m in mats:
        if m.max_row >= 1000:
            r = [m.name, m.max_row]
            for arith in ARITH:
                t = m.ms("activity", f"T/S/-/{arith}")
                per_entry[arith].append(t * 1e3 / m.max_row)
                r += [num(t), num(t * 1e3 / m.max_row, 3)]
            rows.append(r)
    rows.append(
        [
            "median",
            "",
            "",
            num(statistics.median(per_entry["plain"]), 3),
            "",
            num(statistics.median(per_entry["comp"]), 3),
        ]
    )
    table(
        [
            "matrix",
            "max row",
            "T/S/-/plain ms",
            "us per entry of the longest row",
            "T/S/-/comp ms",
            "us per entry of the longest row",
        ],
        rows,
    )
    print(
        "H against B by the longest occupied row class up to 64 entries (layout S, counts A):\n"
    )
    rows = []
    for top in ("none",) + LIGHT:
        sub = [m for m in mats if m.top_light("row") == top]
        if not sub:
            continue
        r = [top, len(sub)]
        for arith in ARITH:
            v = [
                m.ms("activity", f"H/S/A/{arith}")
                / m.ms("activity", f"B/S/A/{arith}")
                for m in sub
            ]
            r += [num(gmean(v), 4), f"{num(min(v), 4)} to {num(max(v), 4)}"]
        rows.append(r)
    table(
        [
            "longest occupied row class <= 64",
            "matrices",
            "H / B plain geomean",
            "range",
            "H / B comp geomean",
            "range",
        ],
        rows,
    )

    print(
        "### 1.8 What the run's own checks say (verification section of the files)\n"
    )
    text(
        "Every activity variant was compared with the one-thread CPU sum of the same arithmetic.",
        "'bitwise' counts the matrices on which every variant of the scheduling returned the same",
        "bits (for comp: the same hi and the same lo). 'largest difference' is the largest",
        "max_diff_over_term_scale: |hi + lo - reference| relative to the row's sum of term",
        "magnitudes.",
    )
    rows = []
    for s in "TBHC":
        for arith in ARITH:
            equal, worst = 0, (0.0, "-")
            for m in mats:
                es = [
                    e
                    for e in m.ver["activity"]
                    if e["name"].startswith(s + "/")
                    and e["name"].endswith("/" + arith)
                ]
                equal += all(e["bitwise_equal_cpu"] for e in es)
                worst = max(
                    worst,
                    (max(e["max_diff_over_term_scale"] for e in es), m.name),
                )
            rows.append(
                [
                    s,
                    arith,
                    f"{equal} of {len(mats)}",
                    f"{worst[0]:.3g}" + (f" ({worst[1]})" if worst[0] else ""),
                ]
            )
    table(
        [
            "scheduling",
            "arithmetic",
            "bitwise equal to the one-thread sum",
            "largest difference",
        ],
        rows,
    )
    text(
        "Compensated against plain on the CPU reference (one full recomputation from the original",
        "bounds, no incremental updates): matrices where a result differs.",
    )
    rows = []
    for m in mats:
        o = [
            x
            for x in m.ver["other"]
            if x["check"] == "compensated against plain"
        ][0]
        if o["activity_max_abs_diff"] or o["candidates_differing"]:
            rows.append(
                [
                    m.name,
                    f"{o['activity_max_abs_diff']:.3g}",
                    f"{o['activity_max_rel_diff']:.3g}",
                    o["candidates_differing"],
                    2 * m.cols,
                    f"{o['candidate_max_abs_diff']:.3g}",
                ]
            )
    table(
        [
            "matrix",
            "largest activity difference",
            "relative",
            "candidate bounds that differ",
            "candidate bounds (2 per column)",
            "largest candidate difference",
        ],
        rows,
    )
    return out


# ----------------------------------------------------------------------------------------------
# section 2


def section_candidates(mats):
    print("## 2. Candidate kernels\n")
    print(
        "Variant name: scheduling / layout / arithmetic [/ lo load] (README_sweeps.md section 2).\n"
    )
    print("### 2.1 Every variant on every matrix\n")
    variant_tables(
        mats,
        "candidate",
        cand_variants("plain") + cand_variants("comp"),
        "Candidates",
    )

    out = {}
    print(
        "### 2.2 Scheduling: one thread per column up to 64 (H) against lane groups (B)\n"
    )
    tails = ("P/plain", "S/plain", "P/comp", "S/comp/gated", "S/comp/always")
    out["sched"] = comparison_block(
        mats,
        "candidate",
        "Scheduling",
        "H",
        "B",
        [(f"./{t}", f"H/{t}", f"B/{t}") for t in tails],
    )

    print("### 2.3 Layout: split arrays (S) against the paired value (P)\n")
    pairs = []
    for s in "BH":
        pairs.append((f"{s}/./plain", f"{s}/S/plain", f"{s}/P/plain"))
        pairs.append(
            (f"{s}/./comp, S gated", f"{s}/S/comp/gated", f"{s}/P/comp")
        )
        pairs.append(
            (f"{s}/./comp, S always", f"{s}/S/comp/always", f"{s}/P/comp")
        )
    out["layout"] = comparison_block(
        mats, "candidate", "Layout", "S", "P", pairs
    )
    text(
        "P/comp always loads the low part. 'S gated' against P therefore mixes the layout with the",
        "gating of the low part; 'S always' against P is the like-for-like layout comparison.",
    )
    two = [
        max(
            m.m["rows_by_infinite_terms_min"][2],
            m.m["rows_by_infinite_terms_max"][2],
        )
        / m.rows
        for m in mats
    ]
    text(
        "Layout S skips the load of the sum when the count is 2 or more. Spearman rank correlation",
        "of S / P with the share of rows that hold two or more infinite terms (larger of the two",
        "sides):",
        "; ".join(
            f"{k} {num(spearman(two, [r for _, r, _ in out['layout'][k]]), 3)}"
            for k in (
                "B/./plain",
                "B/./comp, S gated",
                "H/./plain",
                "H/./comp, S gated",
            )
        )
        + ".",
    )

    print(
        "### 2.4 Low part: loaded only when flagged (gated) against always\n"
    )
    out["gate"] = comparison_block(
        mats,
        "candidate",
        "Low part",
        "gated",
        "always",
        [
            (f"{s}/S/comp/.", f"{s}/S/comp/gated", f"{s}/S/comp/always")
            for s in "BH"
        ],
    )
    los = [lo_share(m) for m in mats]
    text(
        "Share of rows with a nonzero low part (larger side): median",
        f"{pct(statistics.median(los))}%, max {pct(max(los))}% ({mats[los.index(max(los))].name});",
        "matrices above 10%:",
        ", ".join(f"{m.name} {pct(s)}%" for m, s in zip(mats, los) if s > 0.10)
        + ".",
        "Spearman rank correlation of gated / always with that share:",
        "; ".join(
            f"{s} {num(spearman(los, [x for _, x, _ in out['gate'][f'{s}/S/comp/.']]), 3)}"
            for s in "BH"
        )
        + ".",
    )

    print("### 2.5 Arithmetic: compensated against plain\n")
    text(
        "In the benchmark all plain candidate variants are timed before all compensated ones",
        "(sweeps.cu, section_candidates), on the reference activities of the same arithmetic.",
    )
    pairs = []
    for s in "BH":
        pairs.append((f"{s}/P/.", f"{s}/P/comp", f"{s}/P/plain"))
        pairs.append((f"{s}/S/. gated", f"{s}/S/comp/gated", f"{s}/S/plain"))
        pairs.append((f"{s}/S/. always", f"{s}/S/comp/always", f"{s}/S/plain"))
    out["arith"] = comparison_block(
        mats, "candidate", "Arithmetic", "comp", "plain", pairs
    )

    print("### 2.6 H against B by column structure (layout S, comp gated)\n")
    b_name, h_name = "B/S/comp/gated", "H/S/comp/gated"
    rows = []
    for m in sorted(
        mats,
        key=lambda m: m.ms("candidate", h_name) / m.ms("candidate", b_name),
    ):
        b, h = m.ms("candidate", b_name), m.ms("candidate", h_name)
        rows.append(
            [
                m.name,
                m.nnz,
                m.cols,
                num(m.nnz / m.cols, 3),
                m.max_col,
                m.top_light("col"),
                pct(m.share("col", LIGHT, "nnz")),
                num(m.block_thread_entries("col"), 3),
                num(b),
                num(h),
                num(h / b, 4),
                num((h - b) * 1e3, 3),
            ]
        )
    table(
        [
            "matrix (sorted by H/B)",
            "nnz",
            "cols",
            "mean col",
            "max col",
            "longest occupied col class <= 64",
            "% nnz in cols <= 64",
            "entries per block thread",
            "B ms",
            "H ms",
            "H / B",
            "H - B us",
        ],
        rows,
    )
    ratio = [
        m.ms("candidate", h_name) / m.ms("candidate", b_name) for m in mats
    ]
    feats = (
        ("nnz", [m.nnz for m in mats]),
        ("cols", [m.cols for m in mats]),
        ("cols up to 64 entries", [m.count("col", LIGHT) for m in mats]),
        ("mean col length", [m.nnz / m.cols for m in mats]),
        (
            "share of nnz in cols <= 64",
            [m.share("col", LIGHT, "nnz") for m in mats],
        ),
    )
    text(
        "Spearman rank correlation of H / B with:",
        "; ".join(f"{n} {num(spearman(v, ratio), 3)}" for n, v in feats) + ".",
    )
    print("H / B by the longest occupied column class up to 64 entries:\n")
    rows = []
    for top in ("none",) + LIGHT:
        sub = [m for m in mats if m.top_light("col") == top]
        if not sub:
            continue
        v = [
            m.ms("candidate", h_name) / m.ms("candidate", b_name) for m in sub
        ]
        fast = [
            m.name
            for m in sub
            if m.ms("candidate", h_name) < m.ms("candidate", b_name) / MARGIN
        ]
        rows.append(
            [
                top,
                len(sub),
                num(gmean(v), 4),
                f"{num(min(v), 4)} to {num(max(v), 4)}",
                ", ".join(fast) or "none",
            ]
        )
    table(
        [
            "longest occupied col class <= 64",
            "matrices",
            "H / B geomean",
            "range",
            "H faster by more than 3% on",
        ],
        rows,
    )

    print("### 2.7 Cost of the two schedulings by structure\n")
    text(
        "Relative fits as in section 1.7, on columns. The second table drops the two latency terms",
        "to show what they explain.",
    )
    names = ("B/S/plain", "H/S/plain", b_name, h_name)
    table(
        MODEL_HEADER,
        [model_row(mats, "candidate", v, "col", v[0]) for v in names],
    )
    rows = []
    for v in names:
        ys = [m.ms("candidate", v) for m in mats]
        regs = (
            lambda m: m.nnz,
            lambda m: m.count("col", LIGHT),
            lambda m: m.blocks("col"),
        )
        rows.append(fit_row(v, mats, ys, regs, True))
    table(
        [
            "variant",
            "fit",
            "matrices",
            "a us",
            "ns per nnz",
            "ns per column up to 64",
            "ns per block",
            "median abs error %",
            "lowest measured / fit",
            "highest measured / fit",
        ],
        rows,
    )
    print(
        "Candidate time of the matrices with at least 16 entries per block thread on the columns:\n"
    )
    rows = []
    for m in mats:
        e = m.block_thread_entries("col")
        if e >= 16:
            rows.append(
                [
                    m.name,
                    m.nnz,
                    m.max_col,
                    m.cc[BLOCK]["cols"],
                    m.cc[CHUNK]["cols"],
                    num(e, 3),
                    pct(inf_share(m)),
                    num(m.ms("candidate", "B/S/plain")),
                    num(m.ms("candidate", b_name)),
                    num(m.ms("candidate", b_name) * 1e3 / e, 3),
                ]
            )
    table(
        [
            "matrix",
            "nnz",
            "max col",
            "cols 65-8192",
            "cols > 8192",
            "entries per block thread",
            "% rows with an infinite term",
            "B/S/plain ms",
            "B/S/comp/gated ms",
            "whole kernel us / entry per block thread",
        ],
        rows,
    )
    return out


# ----------------------------------------------------------------------------------------------
# section 3


def rule_table(mats, title, fixed_label, other_sec, fixed_ms, name_of, rules):
    """Scores scheduling rules for one side of the round (rows or columns); the other side is
    fixed. name_of(s) gives the variant of the ruled side for scheduling s.
    """
    cs = combos("comp")
    best = [min(combo_ms(m, a, c) for a, c in cs) for m in mats]
    rows = []
    for label, pick_h in rules:
        on_h = [bool(pick_h(m)) for m in mats]
        cost = [
            fixed_ms(m) + m.ms(other_sec, name_of("H" if h else "B"))
            for m, h in zip(mats, on_h)
        ]
        reg = [c / b for c, b in zip(cost, best)]
        rows.append(
            [
                label,
                sum(on_h),
                num(gmean(reg), 4),
                extremes(reg, mats)[1],
                sum(1 for r in reg if r > 1.10),
                sum(1 for r in reg if r > 1.5),
            ]
        )
    print(f"{title} ({fixed_label}):\n")
    table(
        [
            "rule",
            "matrices on H",
            "geomean regret",
            "worst regret (matrix)",
            "> 1.10",
            "> 1.5",
        ],
        rows,
    )


def section_round(mats):
    print("## 3. Full round (activities, then candidates)\n")
    print(
        "### 3.1 Best and worst combination per matrix (timed as one unit)\n"
    )
    for arith in ARITH:
        rows = []
        best, worst, ratio = [], [], []
        for m in mats:
            ba, bc, be = m.round[f"best {arith}"]
            wa, wc, we = m.round[f"worst {arith}"]
            best.append(be["ms_median"] * 1e6 / m.nnz)
            worst.append(we["ms_median"] * 1e6 / m.nnz)
            ratio.append(we["ms_median"] / be["ms_median"])
            rows.append(
                [
                    m.name,
                    m.band,
                    m.nnz,
                    f"{ba} + {bc}",
                    num(be["ms_median"]),
                    num(best[-1]),
                    f"{wa} + {wc}",
                    num(we["ms_median"]),
                    num(worst[-1]),
                    num(ratio[-1], 4),
                ]
            )
        rows.append(
            [
                "geomean",
                "",
                "",
                "",
                "",
                num(gmean(best)),
                "",
                "",
                num(gmean(worst)),
                num(gmean(ratio), 4),
            ]
        )
        rows.append(
            [
                "median",
                "",
                "",
                "",
                "",
                num(statistics.median(best)),
                "",
                "",
                num(statistics.median(worst)),
                num(statistics.median(ratio), 4),
            ]
        )
        table(
            [
                "matrix",
                "band",
                "nnz",
                f"best {arith}",
                "ms",
                "ns per nnz",
                f"worst {arith}",
                "ms",
                "ns per nnz",
                "worst / best",
            ],
            rows,
        )
    print("How often each combination is the best / the worst:\n")
    rows = []
    for kind in ("best plain", "best comp", "worst plain", "worst comp"):
        n = collections.Counter(
            f"{m.round[kind][0]} + {m.round[kind][1]}" for m in mats
        )
        rows.append([kind, "; ".join(f"{k}: {v}" for k, v in n.most_common())])
        n = collections.Counter(m.round[kind][0].split("/")[0] for m in mats)
        rows.append(
            [
                f"{kind}, row scheduling only",
                "; ".join(f"{k}: {v}" for k, v in n.most_common()),
            ]
        )
        n = collections.Counter(m.round[kind][1].split("/")[0] for m in mats)
        rows.append(
            [
                f"{kind}, column scheduling only",
                "; ".join(f"{k}: {v}" for k, v in n.most_common()),
            ]
        )
    table(["", "matrices"], rows)
    cost = [
        m.round["best comp"][2]["ms_median"]
        / m.round["best plain"][2]["ms_median"]
        for m in mats
    ]
    lo, hi = extremes(cost, mats)
    text(
        f"Best comp round / best plain round: geomean {num(gmean(cost), 4)}, lowest {lo}, highest {hi}."
    )

    print(
        "### 3.2 Round timed as one unit against the sum of its two parts timed separately\n"
    )
    rows = []
    for kind in ("best plain", "best comp", "worst plain", "worst comp"):
        parts = [combo_ms(m, m.round[kind][0], m.round[kind][1]) for m in mats]
        r = [m.round[kind][2]["ms_median"] / p for m, p in zip(mats, parts)]
        d = [
            (m.round[kind][2]["ms_median"] - p) * 1e3
            for m, p in zip(mats, parts)
        ]
        lo, hi = extremes(r, mats)
        rows.append(
            [kind, num(gmean(r), 4), lo, hi, num(statistics.median(d), 3)]
        )
    table(
        [
            "round",
            "geomean round / (activity + candidates)",
            "lowest",
            "highest",
            "median difference us",
        ],
        rows,
    )

    print("### 3.3 The best round by band\n")
    rows = []
    for band in BANDS:
        sub = [m for m in mats if m.band == band]
        if not sub:
            continue
        r = [
            band,
            len(sub),
            f"{min(m.nnz for m in sub)} to {max(m.nnz for m in sub)}",
        ]
        for arith in ARITH:
            v = [
                m.round[f"best {arith}"][2]["ms_median"] * 1e6 / m.nnz
                for m in sub
            ]
            t = [m.round[f"best {arith}"][2]["ms_median"] for m in sub]
            r += [
                f"{num(statistics.median(t))} ({num(min(t))} to {num(max(t))})",
                f"{num(statistics.median(v))} ({num(min(v))} to {num(max(v))})",
            ]
        rows.append(r)
    table(
        [
            "band",
            "matrices",
            "nnz",
            "plain ms median (range)",
            "plain ns per nnz median (range)",
            "comp ms median (range)",
            "comp ns per nnz median (range)",
        ],
        rows,
    )
    for arith in ARITH:
        v = [
            m.round[f"best {arith}"][2]["ms_median"] * 1e6 / m.nnz
            for m in mats
        ]
        text(
            f"Spearman rank correlation of the best {arith} round's ns per nonzero with: nnz",
            num(spearman([m.nnz for m in mats], v), 3)
            + "; (rows + cols) / nnz",
            num(spearman([(m.rows + m.cols) / m.nnz for m in mats], v), 3)
            + "; max row length",
            num(spearman([m.max_row for m in mats], v), 3)
            + "; max col length",
            num(spearman([m.max_col for m in mats], v), 3) + ".",
        )

    print(
        "### 3.4 Fixed part and per-nonzero part: fits of ms against nonzeros\n"
    )
    text(
        "'without square47' drops the only matrix above 1e7 nonzeros. The second table adds one",
        "term per row and column.",
    )
    small = [m for m in mats if m.name != "square47"]
    series = {}
    for arith in ARITH:
        series[f"best {arith} round (timed)"] = (
            lambda m, k=f"best {arith}": m.round[k][2]["ms_median"]
        )
    for arith in ARITH:
        a = act_name(CHOICE_ROWS, CHOICE_LAYOUT, CHOICE_COUNT, arith)
        for s in "BH":
            c = cand_name(s, CHOICE_LAYOUT, arith, CHOICE_LO)
            series[f"{a} + {c} (sum of parts)"] = lambda m, a=a, c=c: combo_ms(
                m, a, c
            )
    series["C/P/R/plain + B/P/plain (sum of parts)"] = lambda m: combo_ms(
        m, "C/P/R/plain", "B/P/plain"
    )
    rows2, rows3 = [], []
    for label, f in series.items():
        for sub, tag in ((mats, ""), (small, ", without square47")):
            ys = [f(m) for m in sub]
            for rel in (False, True):
                rows2.append(
                    fit_row(label + tag, sub, ys, (lambda m: m.nnz,), rel)
                )
            rows3.append(
                fit_row(
                    label + tag,
                    sub,
                    ys,
                    (lambda m: m.nnz, lambda m: m.rows + m.cols),
                    True,
                )
            )
    table(
        [
            "series",
            "fit",
            "matrices",
            "a us",
            "b ns per nnz",
            "median abs error %",
            "lowest measured / fit",
            "highest measured / fit",
        ],
        rows2,
    )
    table(
        [
            "series",
            "fit",
            "matrices",
            "a us",
            "b ns per nnz",
            "c ns per row or column",
            "median abs error %",
            "lowest measured / fit",
            "highest measured / fit",
        ],
        rows3,
    )
    ys = [m.round["best comp"][2]["ms_median"] for m in mats]
    coef, _, ratio = fit(mats, ys, (lambda m: m.nnz,), True)
    print(
        "Share of the fitted constant in the fitted time of the best comp round (relative fit), by size:\n"
    )
    rows = []
    for n in (5e4, 1e5, 2e5, 5e5, 1e6, 3e6, 1e7, 3e7):
        t = coef[0] + coef[1] * n / 1e6
        rows.append([f"{n:.0e}", num(t), num(t * 1e6 / n), pct(coef[0] / t)])
    table(["nnz", "fitted ms", "fitted ns per nnz", "constant share %"], rows)
    print(
        "The fitted constant is a fit parameter, not a floor. Fastest measured rounds:\n"
    )
    rows = []
    for arith in ARITH:
        t = [m.round[f"best {arith}"][2]["ms_median"] for m in mats]
        c = fit(mats, t, (lambda m: m.nnz,), True)[0]
        order = sorted(range(len(mats)), key=lambda i: t[i])[:3]
        rows.append(
            [
                f"best {arith} round",
                num(c[0] * 1e3),
                "; ".join(
                    f"{num(t[i] * 1e3)} us ({mats[i].name}, {mats[i].nnz} nnz)"
                    for i in order
                ),
            ]
        )
    table(
        [
            "series",
            "constant of the relative fit, us",
            "three fastest measured rounds",
        ],
        rows,
    )

    print(
        f"### 3.5 Matrices more than {OFF_FIT}x off the fit (best comp round)\n"
    )
    ya = [m.ms("activity", m.round["best comp"][0]) for m in mats]
    yc = [m.ms("candidate", m.round["best comp"][1]) for m in mats]
    coef_a, _, ratio_a = fit(mats, ya, (lambda m: m.nnz,), True)
    coef_c, _, ratio_c = fit(mats, yc, (lambda m: m.nnz,), True)
    text(
        f"Round: ms = {num(coef[0] * 1e3)} us + {num(coef[1])} ns * nnz. Its two variants timed",
        f"alone, fitted the same way: activity {num(coef_a[0] * 1e3)} us + {num(coef_a[1])} ns *",
        f"nnz, candidates {num(coef_c[0] * 1e3)} us + {num(coef_c[1])} ns * nnz. 'thread entries'",
        "is the entries per block thread of the module docstring.",
    )
    rows = []
    for i, m in enumerate(mats):
        if 1.0 / OFF_FIT <= ratio[i] <= OFF_FIT:
            continue
        rows.append(
            [
                m.name,
                m.band,
                m.nnz,
                num(ys[i]),
                num(ratio[i], 3),
                f"{m.round['best comp'][0]}: {num(ya[i])}",
                num(ratio_a[i], 3),
                f"{m.round['best comp'][1]}: {num(yc[i])}",
                num(ratio_c[i], 3),
                num(m.nnz / m.rows, 3),
                num(m.nnz / m.cols, 3),
                m.count("row", (BLOCK, CHUNK)),
                m.count("col", (BLOCK, CHUNK)),
                num(m.block_thread_entries("row"), 3),
                num(m.block_thread_entries("col"), 3),
                num((m.rows + m.cols) / m.nnz, 3),
            ]
        )
    rows.sort(key=lambda r: float(r[4]))
    table(
        [
            "matrix",
            "band",
            "nnz",
            "round ms",
            "round / fit",
            "activity variant: ms",
            "activity / fit",
            "candidate variant: ms",
            "candidates / fit",
            "mean row",
            "mean col",
            "rows > 64",
            "cols > 64",
            "thread entries rows",
            "thread entries cols",
            "(rows + cols) / nnz",
        ],
        rows,
    )
    sub = [(ratio[i], m) for i, m in enumerate(mats)]
    feats = (
        (
            "entries per block thread, columns",
            [m.block_thread_entries("col") for _, m in sub],
        ),
        (
            "entries per block thread, rows",
            [m.block_thread_entries("row") for _, m in sub],
        ),
        ("(rows + cols) / nnz", [(m.rows + m.cols) / m.nnz for _, m in sub]),
        ("blocks per nnz, rows", [m.blocks("row") / m.nnz for _, m in sub]),
        ("blocks per nnz, columns", [m.blocks("col") / m.nnz for _, m in sub]),
        ("nnz", [m.nnz for _, m in sub]),
    )
    text(
        "Spearman rank correlation of round / fit over all 32 matrices with:",
        "; ".join(
            f"{n} {num(spearman(v, [r for r, _ in sub]), 3)}" for n, v in feats
        )
        + ".",
    )

    print(
        "### 3.6 Every fixed combination against the best combination of each matrix (sum of parts)\n"
    )
    text(
        "Cost of a combination on a matrix = activity ms + candidate ms of the two variants, same",
        "layout. Regret = that cost / the lowest cost of any combination of the same arithmetic on",
        "that matrix. Sorted by the geometric mean of the regret.",
    )
    policy = {}
    for arith in ARITH:
        cs = combos(arith)
        best = [min(combo_ms(m, a, c) for a, c in cs) for m in mats]
        rows = []
        for a, c in cs:
            reg = [combo_ms(m, a, c) / b for m, b in zip(mats, best)]
            policy[(a, c)] = reg
            rows.append(
                (
                    gmean(reg),
                    [
                        f"{a} + {c}",
                        num(gmean(reg), 4),
                        extremes(reg, mats)[1],
                        sum(1 for r in reg if r <= MARGIN),
                        sum(1 for r in reg if r > 1.10),
                        sum(1 for r in reg if r > 1.5),
                        sum(1 for r in reg if r > 2.0),
                        num(sum(combo_ms(m, a, c) for m in mats)),
                    ],
                )
            )
        rows.sort(key=lambda t: t[0])
        table(
            [
                f"combination ({arith})",
                "geomean regret",
                "worst regret (matrix)",
                "within 3% of best",
                "> 1.10",
                "> 1.5",
                "> 2",
                "sum of ms over the 32",
            ],
            [r for _, r in rows],
        )

    text(
        "Scheduling chosen per matrix by a rule (layout S, counts A, comp, low part gated). Regret",
        "as above. The thresholds were read off the same 32 matrices they are scored on, so these",
        "regrets are in-sample; 'the faster of B and H' is not a rule but the bound a rule can reach.",
    )
    rule_table(
        mats,
        "Column scheduling",
        "rows B",
        "candidate",
        lambda m: m.ms("activity", "B/S/A/comp"),
        lambda s: f"{s}/S/comp/gated",
        [
            ("always B", lambda m: False),
            ("always H", lambda m: True),
            (
                "the faster of B and H",
                lambda m: m.ms("candidate", "H/S/comp/gated")
                < m.ms("candidate", "B/S/comp/gated"),
            ),
            ("H when nnz >= 5e5", lambda m: m.nnz >= 5e5),
            ("H when nnz >= 1e6", lambda m: m.nnz >= 1e6),
            ("H when cols >= 1e5", lambda m: m.cols >= 1e5),
            ("H when cols >= 2e5", lambda m: m.cols >= 2e5),
            (
                "H when the mean column length <= 8",
                lambda m: m.nnz / m.cols <= 8,
            ),
            (
                "H when cols >= 1e5 and no column above 8192",
                lambda m: m.cols >= 1e5 and m.max_col <= 8192,
            ),
        ],
    )
    rule_table(
        mats,
        "Row scheduling",
        "columns B",
        "activity",
        lambda m: m.ms("candidate", "B/S/comp/gated"),
        lambda s: f"{s}/S/A/comp",
        [
            ("always B", lambda m: False),
            ("always H", lambda m: True),
            (
                "the faster of B and H",
                lambda m: m.ms("activity", "H/S/A/comp")
                < m.ms("activity", "B/S/A/comp"),
            ),
            ("H when rows >= 1e5", lambda m: m.rows >= 1e5),
            ("H when rows >= 5e5", lambda m: m.rows >= 5e5),
            (
                "H when no row class above 9-16 is occupied below 65",
                lambda m: m.top_light("row")
                in ("none", "0-2", "3-4", "5-8", "9-16"),
            ),
        ],
    )
    cs = combos("comp")
    best = [min(combo_ms(m, a, c) for a, c in cs) for m in mats]
    both = [
        min(m.ms("activity", "B/S/A/comp"), m.ms("activity", "H/S/A/comp"))
        + min(
            m.ms("candidate", "B/S/comp/gated"),
            m.ms("candidate", "H/S/comp/gated"),
        )
        for m in mats
    ]
    reg = [c / b for c, b in zip(both, best)]
    text(
        "The faster of B and H on both sides (bound for any B/H rule):",
        f"geomean regret {num(gmean(reg), 4)}, worst {extremes(reg, mats)[1]}.",
    )

    print(
        "Regret by band (rows B, layout S, counts A, comp, low part gated; sum of parts):\n"
    )

    def col_cost(m, s):
        return m.ms("activity", "B/S/A/comp") + m.ms(
            "candidate", f"{s}/S/comp/gated"
        )

    picks = (
        ("columns always B", lambda m: "B"),
        (
            "columns H when cols >= 1e5",
            lambda m: "H" if m.cols >= 1e5 else "B",
        ),
        (
            "columns the faster of B and H",
            lambda m: min("BH", key=lambda s: col_cost(m, s)),
        ),
    )
    rows = []
    for band in BANDS:
        idx = [i for i, m in enumerate(mats) if m.band == band]
        if not idx:
            continue
        r = [band, len(idx)]
        for _, pick in picks:
            v = [col_cost(mats[i], pick(mats[i])) / best[i] for i in idx]
            r.append(f"{num(gmean(v), 4)} (max {num(max(v), 4)})")
        rows.append(r)
    table(
        ["band", "matrices"]
        + [f"{label}: geomean regret (max)" for label, _ in picks],
        rows,
    )
    text(
        "Columns B against H on the candidate kernel alone (S, comp gated), total ms over the",
        "matrices of a size group:",
        "; ".join(
            f"{label} B {num(sum(m.ms('candidate', 'B/S/comp/gated') for m in sub))} H "
            f"{num(sum(m.ms('candidate', 'H/S/comp/gated') for m in sub))} (n = {len(sub)})"
            for label, sub in (
                ("nnz below 1e6", [m for m in mats if m.nnz < 1e6]),
                ("nnz 1e6 to 1e7", [m for m in mats if 1e6 <= m.nnz < 1e7]),
                ("square47", [m for m in mats if m.nnz >= 1e7]),
            )
        )
        + ".",
    )
    print(
        "Where the column threshold can sit (rule 'H when cols >= t', in-sample):\n"
    )
    rows = []
    for t in (2e4, 3e4, 4e4, 5e4, 7e4, 9e4, 1e5, 1.2e5, 1.5e5, 2e5, 5e5):
        v = [
            col_cost(m, "H" if m.cols >= t else "B") / b
            for m, b in zip(mats, best)
        ]
        rows.append(
            [
                f"{t:.0f}",
                sum(1 for m in mats if m.cols >= t),
                num(gmean(v), 4),
                extremes(v, mats)[1],
            ]
        )
    table(
        ["t", "matrices on H", "geomean regret", "worst regret (matrix)"], rows
    )
    n_cols = sorted(m.cols for m in mats)
    gap = max(
        zip(n_cols, n_cols[1:]),
        key=lambda p: p[1] / p[0] if 3e4 <= p[0] < 2e5 else 0.0,
    )
    text(
        f"No matrix has between {gap[0]} and {gap[1]} columns, so the data cannot place the",
        "threshold inside that range.",
    )

    print("### 3.7 One CPU thread, for scale (full sweep, same arithmetic)\n")
    rows = []
    sp = {k: [] for k in ARITH}
    for m in mats:
        r = [m.name, m.nnz]
        for arith, cpu_arith in (("plain", "plain"), ("comp", "comp-fma")):
            cpu = m.ms("cpu", f"activity, full, {cpu_arith}") + m.ms(
                "cpu", f"candidates by column, full, {cpu_arith}"
            )
            gpu = m.round[f"best {arith}"][2]["ms_median"]
            sp[arith].append(cpu / gpu)
            r += [num(cpu), num(gpu), num(cpu / gpu, 3)]
        rows.append(r)
    rows.append(
        [
            "geomean",
            "",
            "",
            "",
            num(gmean(sp["plain"]), 3),
            "",
            "",
            num(gmean(sp["comp"]), 3),
        ]
    )
    table(
        [
            "matrix",
            "nnz",
            "cpu plain ms (activity + candidates by column)",
            "gpu best plain round ms",
            "cpu / gpu",
            "cpu comp-fma ms",
            "gpu best comp round ms",
            "cpu / gpu",
        ],
        rows,
    )

    print(
        "### 3.8 Frontier-restricted activity sweep against the full sweep (results.frontier)\n"
    )
    text(
        "The frontier path of the benchmark compacts a flag array into two row lists",
        "(cub::DeviceSelect::Flagged twice, one read-back of the two counts) and runs the H/S/A",
        "kernels over the lists; it exists only for scheduling H, and the frontier is a random share",
        "of the rows. 'frontier / full B' divides that time by the full sweep of all rows with",
        "B/S/A of the same arithmetic: above 1, recomputing every row is cheaper than the frontier",
        "path. A frontier kernel without the compaction (a flag test inside the full kernel) was",
        "not measured.",
    )
    fracs = ("10", "1", "0.1", "0.01")
    comp_only = [
        m.frontier[(f, "compaction only")]["ms_median"]
        for m in mats
        for f in fracs
    ]
    text(
        f"Compaction alone: median {num(statistics.median(comp_only) * 1e3)} us, from",
        f"{num(min(comp_only) * 1e3)} to {num(max(comp_only) * 1e3)} us over the {len(mats)} matrices",
        "and the four frontier sizes.",
    )
    groups = (
        ("all", lambda m: True),
        ("nnz below 5e5", lambda m: m.nnz < 5e5),
        ("nnz 5e5 to 1e6", lambda m: 5e5 <= m.nnz < 1e6),
        ("nnz 1e6 and above", lambda m: m.nnz >= 1e6),
    )
    rows = []
    for arith in ARITH:
        for f in fracs:
            r = [arith, f + "%"]
            for _, pick in groups:
                sub = [m for m in mats if pick(m)]
                v = [
                    m.frontier[(f, f"compaction + H/S/A/{arith}")]["ms_median"]
                    / m.ms("activity", f"B/S/A/{arith}")
                    for m in sub
                ]
                r.append(
                    f"{num(gmean(v), 3)}; {sum(1 for x in v if x > 1.0)} of {len(sub)}"
                )
            rows.append(r)
    table(
        ["arithmetic", "frontier (share of rows)"]
        + [
            f"{label}: geomean frontier / full B; matrices where the frontier path is slower"
            for label, _ in groups
        ],
        rows,
    )
    print("Per matrix, comp: ms, median\n")
    rows = []
    for m in mats:
        rows.append(
            [
                m.name,
                m.nnz,
                num(m.ms("activity", "B/S/A/comp")),
                num(m.ms("activity", "H/S/A/comp")),
                num(m.frontier[("0.01", "compaction only")]["ms_median"]),
            ]
            + [
                num(m.frontier[(f, "compaction + H/S/A/comp")]["ms_median"])
                for f in fracs
            ]
        )
    table(
        [
            "matrix",
            "nnz",
            "full B/S/A/comp",
            "full H/S/A/comp",
            "compaction only (0.01%)",
        ]
        + [f"frontier {f}%" for f in fracs],
        rows,
    )
    return policy


# ----------------------------------------------------------------------------------------------
# section 4


def section_rebuild(mats):
    print("## 4. Rebuild (20% of the rows and columns removed)\n")
    print("### 4.1 Every method on every matrix: ms, median\n")
    table(
        ["matrix", "nnz", "cols"] + [REB_SHORT[n] for n in REB_ALL],
        [
            [m.name, m.nnz, m.cols]
            + [num(m.ms("rebuild", n)) for n in REB_ALL]
            for m in mats
        ],
    )
    print("ns per nonzero of the original matrix, median\n")
    rows = [
        [m.name, m.nnz] + [num(m.ns("rebuild", n)) for n in REB_ALL]
        for m in mats
    ]
    rows.append(
        ["geomean", ""]
        + [num(gmean(m.ns("rebuild", n) for m in mats)) for n in REB_ALL]
    )
    table(["matrix", "nnz"] + [REB_SHORT[n] for n in REB_ALL], rows)

    print("### 4.2 Methods against each other and against one CPU thread\n")
    pairs = [
        (
            "CSC step: scatter + segmented sort / radix sort",
            REB_SEG,
            REB_RADIX,
        ),
        ("CSC step: scatter unordered / radix sort", REB_UNORD, REB_RADIX),
        (
            "CSC step: scatter unordered / scatter + segmented sort",
            REB_UNORD,
            REB_SEG,
        ),
        (
            "CSC step: compaction of the old CSC / radix sort",
            REB_OLD,
            REB_RADIX,
        ),
        (
            "CSC step: compaction of the old CSC / scatter + segmented sort",
            REB_OLD,
            REB_SEG,
        ),
        ("total: segmented sort / radix sort", REB_T_SEG, REB_T_RADIX),
        ("total: compact old CSC / radix sort", REB_T_OLD, REB_T_RADIX),
        ("total: compact old CSC / segmented sort", REB_T_OLD, REB_T_SEG),
        ("total radix sort / cpu", REB_T_RADIX, REB_CPU),
        ("total segmented sort / cpu", REB_T_SEG, REB_CPU),
        ("total compact old CSC / cpu", REB_T_OLD, REB_CPU),
    ]
    reb = {label: compare(mats, "rebuild", x, y) for label, x, y in pairs}
    table(
        [
            "ratio first / second",
            "geomean",
            "best (matrix)",
            "worst (matrix)",
            "first faster",
            "second faster",
            "within 3%",
            "inside spread",
        ],
        [summary_row(label, reb[label]) for label, _, _ in pairs],
    )
    print("Per matrix:\n")
    rows = []
    for i, m in enumerate(mats):
        seg_alone = (
            (m.ms("rebuild", REB_SEG) - m.ms("rebuild", REB_UNORD))
            * 1e6
            / m.nnz
        )
        rows.append(
            [
                m.name,
                m.nnz,
                m.cols,
                num(m.nnz / m.cols, 3),
                m.max_col,
                num(reb[pairs[0][0]][i][1], 4),
                num(reb[pairs[5][0]][i][1], 4),
                num(reb[pairs[6][0]][i][1], 4),
                num(1 / reb[pairs[8][0]][i][1], 3),
                num(1 / reb[pairs[9][0]][i][1], 3),
                num(1 / reb[pairs[10][0]][i][1], 3),
                num(seg_alone, 3),
            ]
        )
    table(
        [
            "matrix",
            "nnz",
            "cols",
            "mean col",
            "max col",
            "CSC seg. sort / radix",
            "total seg. sort / radix",
            "total compact old / radix",
            "cpu / total radix",
            "cpu / total seg. sort",
            "cpu / total compact old",
            "segmented sort alone ns per nnz",
        ],
        rows,
    )
    ratio = [
        m.ms("rebuild", REB_SEG) / m.ms("rebuild", REB_RADIX) for m in mats
    ]
    feats = (
        ("nnz", [m.nnz for m in mats]),
        ("cols", [m.cols for m in mats]),
        ("mean col length", [m.nnz / m.cols for m in mats]),
        ("max col length", [m.max_col for m in mats]),
        (
            "share of nnz in cols above 64",
            [m.share("col", (BLOCK, CHUNK), "nnz") for m in mats],
        ),
    )
    text(
        "Spearman rank correlation of CSC seg. sort / radix with:",
        "; ".join(f"{n} {num(spearman(v, ratio), 3)}" for n, v in feats) + ".",
    )
    r = [
        m.ms("rebuild", REB_T_RADIX)
        / (m.ms("rebuild", REB_CSR) + m.ms("rebuild", REB_RADIX))
        for m in mats
    ]
    text(
        "Total radix timed as one unit / (compact CSR + CSC radix timed apart): geomean",
        f"{num(gmean(r), 4)}, range {num(min(r), 4)} to {num(max(r), 4)}.",
    )
    slower = [
        (
            m.ms("rebuild", n) / m.ms("rebuild", REB_CPU),
            m.name,
            m.nnz,
            REB_SHORT[n],
        )
        for m in mats
        for n in (REB_T_RADIX, REB_T_OLD)
        if m.ms("rebuild", n) > m.ms("rebuild", REB_CPU)
    ]
    text(
        "GPU slower than one CPU thread:",
        "; ".join(
            f"{n} ({nnz} nnz) {s} {num(r, 3)}x"
            for r, n, nnz, s in sorted(slower, reverse=True)
        )
        + ".",
    )

    print("### 4.3 Fits of ms against nonzeros of the original matrix\n")
    small = [m for m in mats if m.name != "square47"]
    rows = []
    coefs = {}
    for n in (REB_T_RADIX, REB_T_SEG, REB_T_OLD, REB_CPU):
        for sub, tag in ((mats, ""), (small, ", without square47")):
            ys = [m.ms("rebuild", n) for m in sub]
            for rel in (False, True):
                rows.append(
                    fit_row(
                        REB_SHORT[n] + tag, sub, ys, (lambda m: m.nnz,), rel
                    )
                )
            if sub is small:
                coefs[n] = fit(sub, ys, (lambda m: m.nnz,), True)[0]
    table(
        [
            "series",
            "fit",
            "matrices",
            "a us",
            "b ns per nnz",
            "median abs error %",
            "lowest measured / fit",
            "highest measured / fit",
        ],
        rows,
    )
    text(
        "Break-even with one CPU thread from the relative fits without square47 (nnz at which the",
        f"two lines cross). The smallest measured matrix has {min(m.nnz for m in mats)} nonzeros;",
        "a value below that is an extrapolation.",
    )
    rows = []
    ca, cb = coefs[REB_CPU]
    for n in (REB_T_RADIX, REB_T_SEG, REB_T_OLD):
        a, b = coefs[n]
        x = (a - ca) / (cb - b) * 1e6
        worst = max(
            (m.ms("rebuild", n) / m.ms("rebuild", REB_CPU), m.name)
            for m in mats
        )
        rows.append(
            [
                REB_SHORT[n],
                num(a * 1e3),
                num(b),
                num(ca * 1e3),
                num(cb),
                f"{x:.0f}",
                f"{num(worst[0], 3)} ({worst[1]})",
            ]
        )
    table(
        [
            "GPU method",
            "a us",
            "b ns per nnz",
            "cpu a us",
            "cpu b ns per nnz",
            "break-even nnz",
            "highest gpu / cpu measured (matrix)",
        ],
        rows,
    )
    return reb


# ----------------------------------------------------------------------------------------------
# section 5


def section_choice(mats, act, cand, policy, reb):
    print("## 5. Numbers behind each choice\n")
    text(
        f"Choice scored: layout {CHOICE_LAYOUT}, counts {CHOICE_COUNT}, rows {CHOICE_ROWS}, columns",
        f"{CHOICE_COLS}, compensated sums with the low part {CHOICE_LO}, rebuild by stable radix",
        "sort (compaction of the old CSC when the rebuild is deletions only). 'loses on' lists the",
        "matrices where the alternative is faster by more than 3% and outside the spread, with the",
        "ratio choice / alternative (up to 8).",
    )
    rows = []

    def add(decision, alt, per):
        n = collections.Counter(c for _, _, c in per)
        rows.append(
            [
                decision,
                alt,
                num(gmean(r for _, r, _ in per), 4),
                n["x"],
                n["y"],
                losers(per),
            ]
        )

    r, lay, cnt, col = CHOICE_ROWS, CHOICE_LAYOUT, CHOICE_COUNT, CHOICE_COLS
    for arith in ARITH:
        add(
            f"layout S, activity {r}/./{cnt}/{arith}",
            "P",
            act["layout"][f"{r}/./{cnt}/{arith}"],
        )
    add(
        f"layout S, candidates {col}/./plain",
        "P",
        cand["layout"][f"{col}/./plain"],
    )
    add(
        f"layout S, candidates {col}/./comp gated",
        "P",
        cand["layout"][f"{col}/./comp, S gated"],
    )
    add(
        "layout S, cuOpt's scheme C/./R/plain",
        "P",
        act["layout"]["C/./R/plain"],
    )
    for arith in ARITH:
        add(
            f"counts A, activity {r}/{lay}/./{arith}",
            "R",
            act["count"][f"{r}/{lay}/./{arith}"],
        )
    add(
        "counts A, cuOpt's scheme C/P/./plain",
        "R",
        act["count"]["C/P/./plain"],
    )
    add("counts A, cuOpt's scheme C/P/./comp", "R", act["count"]["C/P/./comp"])
    for arith in ARITH:
        per = compare(mats, "activity", f"C/S/A/{arith}", f"C/P/R/{arith}")
        add(
            f"layout S and counts A together, cuOpt's scheme ({arith})",
            "C/P/R",
            per,
        )
    for arith in ARITH:
        for other in "HCT":
            per = compare(
                mats,
                "activity",
                act_name(r, lay, cnt, arith),
                act_name(other, lay, cnt, arith),
            )
            add(f"rows {r}, ./{lay}/{cnt}/{arith}", other, per)
    for arith in ARITH:
        per = compare(
            mats,
            "candidate",
            cand_name(col, lay, arith, CHOICE_LO),
            cand_name("H", lay, arith, CHOICE_LO),
        )
        add(f"columns {col}, ./{lay}/{arith}", "H", per)
    add(
        f"comp, activity {r}/{lay}/{cnt}",
        "plain",
        act["arith"][f"{r}/{lay}/{cnt}/."],
    )
    add(
        f"comp gated, candidates {col}/S",
        "plain",
        cand["arith"][f"{col}/S/. gated"],
    )
    add(
        f"low part gated, candidates {col}/S/comp",
        "always",
        cand["gate"][f"{col}/S/comp/."],
    )
    add(
        "rebuild CSC step: radix sort",
        "scatter + segmented sort",
        [
            (n, 1 / x, {"x": "y", "y": "x"}.get(c, c))
            for n, x, c in reb[
                "CSC step: scatter + segmented sort / radix sort"
            ]
        ],
    )
    add(
        "rebuild total: radix sort",
        "segmented sort",
        [
            (n, 1 / x, {"x": "y", "y": "x"}.get(c, c))
            for n, x, c in reb["total: segmented sort / radix sort"]
        ],
    )
    add(
        "rebuild total: radix sort",
        "compact old CSC (deletions only)",
        [
            (n, 1 / x, {"x": "y", "y": "x"}.get(c, c))
            for n, x, c in reb["total: compact old CSC / radix sort"]
        ],
    )
    add(
        "rebuild total: radix sort",
        "one CPU thread",
        reb["total radix sort / cpu"],
    )
    add(
        "rebuild total: compact old CSC",
        "one CPU thread",
        reb["total compact old CSC / cpu"],
    )
    table(
        [
            "decision (others fixed)",
            "alternative",
            "geomean choice / alternative",
            "choice faster",
            "alternative faster",
            "loses on (ratio)",
        ],
        rows,
    )

    print(
        "Whole round of the choice, sum of parts, against other whole rounds:\n"
    )
    rows = []
    for arith in ARITH:
        a = act_name(r, lay, cnt, arith)
        c = cand_name(col, lay, arith, CHOICE_LO)
        mine = [combo_ms(m, a, c) for m in mats]
        refs = [
            ("the best combination of each matrix", None),
            (
                "C/P/R/plain + B/P/plain (one block per row, paired, reduced counts)",
                ("C/P/R/plain", "B/P/plain"),
            ),
            (
                f"H/S/A/{arith} + {cand_name('H', 'S', arith)} (one thread per row or column up to 64)",
                (f"H/S/A/{arith}", cand_name("H", "S", arith)),
            ),
            (
                f"{a} + {cand_name('H', 'S', arith)}",
                (a, cand_name("H", "S", arith)),
            ),
        ]
        if arith == "comp":
            refs.insert(
                2, ("C/P/R/comp + B/P/comp", ("C/P/R/comp", "B/P/comp"))
            )
            refs.append(
                (
                    "B/S/A/plain + B/S/plain (the same choice in plain arithmetic)",
                    ("B/S/A/plain", "B/S/plain"),
                )
            )
        for label, ref in refs:
            if ref is None:
                ratio = policy[(a, c)]
            else:
                ratio = [
                    x / combo_ms(m, ref[0], ref[1]) for x, m in zip(mine, mats)
                ]
            lo, hi = extremes(ratio, mats)
            worse = sorted(
                ((x, m.name) for x, m in zip(ratio, mats) if x > MARGIN),
                reverse=True,
            )
            rows.append(
                [
                    f"{a} + {c}",
                    label,
                    num(gmean(ratio), 4),
                    lo,
                    hi,
                    len(worse),
                    ", ".join(f"{n} {num(x, 4)}" for x, n in worse[:8])
                    + (
                        f" and {len(worse) - 8} more" if len(worse) > 8 else ""
                    ),
                ]
            )
    table(
        [
            "choice",
            "against",
            "geomean choice / other",
            "lowest (matrix)",
            "highest (matrix)",
            "matrices where the choice is slower by more than 3%",
            "largest (up to 8)",
        ],
        rows,
    )
    print("Time of the choice in absolute terms (sum of parts), by band:\n")
    rows = []
    for band in BANDS:
        sub = [m for m in mats if m.band == band]
        if not sub:
            continue
        row = [band, len(sub)]
        for arith in ARITH:
            a = act_name(r, lay, cnt, arith)
            c = cand_name(col, lay, arith, CHOICE_LO)
            t = [combo_ms(m, a, c) for m in sub]
            row.append(
                f"{num(statistics.median(t))} ({num(min(t))} to {num(max(t))})"
            )
        d = [
            (
                combo_ms(
                    m,
                    act_name(r, lay, cnt, "comp"),
                    cand_name(col, lay, "comp", CHOICE_LO),
                )
                - combo_ms(
                    m,
                    act_name(r, lay, cnt, "plain"),
                    cand_name(col, lay, "plain"),
                )
            )
            * 1e3
            for m in sub
        ]
        row.append(
            f"{num(statistics.median(d), 3)} ({num(min(d), 3)} to {num(max(d), 3)})"
        )
        rows.append(row)
    table(
        [
            "band",
            "matrices",
            "plain ms median (range)",
            "comp ms median (range)",
            "comp - plain us median (range)",
        ],
        rows,
    )


def main():
    mats = load()
    print("# Stage 0: propagation sweep kernels (sweeps/*.json)\n")
    section_inputs(mats)
    act = section_activity(mats)
    cand = section_candidates(mats)
    policy = section_round(mats)
    reb = section_rebuild(mats)
    section_choice(mats, act, cand, policy, reb)


if __name__ == "__main__":
    main()
