#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 time analysis: where HiGHS presolve spends its time on the dev-size testbed.

Reads, relative to this file:

    ../testbed/testbed.json                   "main" list, instances with where == "dev"
    ../results/stage0/stock/summary.json      stock HiGHS 1.16.0 (time used: presolve_core_seconds)
    ../results/stage0/diag/summary.json       patched HiGHS, parsed DIAG lines per instance
    ../results/stage0/diag/<name>.json.log    HiGHS log (yields that have no DIAG field, progress lines)
    ../results/stage0/stats/<name>.json       structure statistics of the original model
    ../results/stage0/stock/<name>.json       structure statistics of the presolved model ("presolved")
    ../results/stage0/fixed_costs.json        GPU fixed costs (H200)
    ../results/stage0/diag_noprobing/         optional supplement (HIGHS_DIAG_OFF=probing); section 3.15
                                              is printed only when its summary.json exists
    ../results/stage0/diag_stage1/            optional supplements (HiGHS restricted to the Stage 1 rules,
    ../results/stage0/diag_stage12/           and to the Stage 1 and 2 rules); section 5.3 is printed only
                                              when diag_stage1/summary.json exists

and prints markdown tables:

    1  per instance: time (stock, diag), time per nonzero, seconds and share of every DIAG stage;
       per band and overall: median, geometric mean, maximum, shifted geometric means;
       agreement of the patched build with the stock build
    2  time against size (log-log fits), instances far from the trend
    3  stages that take more than half of a presolve above 2 s, and the numbers that test the
       cost explanation of each (row removal, dominated columns, probing work, clique merging)
    4  what the time of every expensive stage run removed; runs that removed nothing
    5  numbers for time targets (time by rule group per band, GPU fixed costs)

No arguments. Python 3 standard library only. Nothing is written.
"""

import collections
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE0 = HERE.parent / "results" / "stage0"
TESTBED = HERE.parent / "testbed" / "testbed.json"

STAGES = [
    "setup",
    "initial_scan",
    "fast_loop",
    "clique_substitutions",
    "fourier_motzkin",
    "aggregator",
    "sparsify",
    "parallel_rows_cols",
    "strengthen_inequalities",
    "dominated_columns",
    "enumeration",
    "implied_integers",
    "probing",
    "shrink",
    "final_scaling",
    "other",
]
SHORT = {
    "setup": "setup",
    "initial_scan": "scan",
    "fast_loop": "fast",
    "clique_substitutions": "clqsub",
    "fourier_motzkin": "fm",
    "aggregator": "aggr",
    "sparsify": "spars",
    "parallel_rows_cols": "par",
    "strengthen_inequalities": "streng",
    "dominated_columns": "domcol",
    "enumeration": "enum",
    "implied_integers": "implint",
    "probing": "prob",
    "shrink": "shrink",
    "final_scaling": "scale",
    "other": "other",
}
# stages whose purpose is a reduction (the others are bookkeeping)
REDUCTION_STAGES = [
    "initial_scan",
    "fast_loop",
    "fourier_motzkin",
    "aggregator",
    "sparsify",
    "parallel_rows_cols",
    "strengthen_inequalities",
    "dominated_columns",
    "enumeration",
    "probing",
]
# rule groups of DESIGN.md sections 5 and 13, by the HiGHS stage that holds them
GROUPS = collections.OrderedDict(
    [
        (
            "rows+cols loop",
            [
                "setup",
                "initial_scan",
                "fast_loop",
                "parallel_rows_cols",
                "shrink",
                "final_scaling",
                "other",
            ],
        ),
        (
            "probing+enum",
            [
                "probing",
                "enumeration",
                "clique_substitutions",
                "implied_integers",
            ],
        ),
        (
            "aggr+domcol+fm",
            ["aggregator", "dominated_columns", "fourier_motzkin"],
        ),
        ("spars+streng", ["sparsify", "strengthen_inequalities"]),
    ]
)
BANDS = ["A1", "A2", "A3", "B1", "B2", "B3"]

DOMINANT_MIN_SECONDS = 2.0
DOMINANT_MIN_SHARE = 0.5
YIELD_MIN_SECONDS = 0.5
YIELD_MIN_SHARE = 0.2
ALMOST_NOTHING = (
    1e-3  # removed rows + cols + nonzeros below this fraction of the model
)
TREND_FACTOR = 3.0


# ----------------------------------------------------------------------------- helpers


def load(path):
    with open(path) as f:
        return json.load(f)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def sgm(xs, shift):
    return math.exp(sum(math.log(x + shift) for x in xs) / len(xs)) - shift


def ols(xs, ys):
    """Least squares y = a + b x. Returns a, b, r2, residuals."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sum(r * r for r in res) / sst
    return a, b, r2, res


def ols_multi(cols, ys):
    """Least squares with an intercept and several regressors. Returns r2."""
    n = len(ys)
    rows = [[1.0] + [c[i] for c in cols] for i in range(n)]
    k = len(rows[0])
    ata = [
        [sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)
    ]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    m = [ata[i] + [aty[i]] for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda q: abs(m[q][i]))
        m[i], m[p] = m[p], m[i]
        for q in range(k):
            if q != i:
                f = m[q][i] / m[i][i]
                m[q] = [u - f * v for u, v in zip(m[q], m[i])]
    beta = [m[i][k] / m[i][i] for i in range(k)]
    my = sum(ys) / n
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum(
        (y - sum(b * v for b, v in zip(beta, r))) ** 2
        for r, y in zip(rows, ys)
    )
    return 1.0 - ssr / sst


def theil_sen(xs, ys):
    """Median of pairwise slopes; intercept is the median residual."""
    slopes = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] != xs[j]:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    b = statistics.median(slopes)
    a = statistics.median([y - b * x for x, y in zip(xs, ys)])
    return a, b


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for q in range(i, j + 1):
            r[order[q]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(xs, ys):
    return ols(ranks(xs), ranks(ys))[2] ** 0.5 * (
        1 if ols(ranks(xs), ranks(ys))[1] >= 0 else -1
    )


def fs(x, digits=3):
    """Seconds."""
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def fn(x):
    """Integer with thousands separators."""
    if x is None:
        return "-"
    return f"{int(round(x)):,}"


def fg(x, sig=3):
    if x is None:
        return "-"
    if x == 0:
        return "0"
    return f"{x:.{sig}g}"


def pct(x, digits=0):
    if x is None:
        return "-"
    return f"{100.0 * x:.{digits}f}%"


def table(headers, rows, right=None):
    """Print a markdown table. `right` is the set of right-aligned column indices
    (default: all but the first).
    """
    if right is None:
        right = set(range(1, len(headers)))
    cells = [[str(c) for c in r] for r in rows]
    width = [
        max([len(h)] + [len(r[i]) for r in cells])
        for i, h in enumerate(headers)
    ]

    def line(vals):
        out = []
        for i, v in enumerate(vals):
            out.append(v.rjust(width[i]) if i in right else v.ljust(width[i]))
        return "| " + " | ".join(out) + " |"

    print(line(headers))
    print(
        "|"
        + "|".join(
            ("-" * (width[i] + 1) + ":")
            if i in right
            else (":" + "-" * (width[i] + 1))
            for i in range(len(headers))
        )
        + "|"
    )
    for r in cells:
        print(line(r))
    print()


def heading(text, level=2):
    print("#" * level + " " + text)
    print()


def parse_pair(text):
    a, b = text.split(":")
    return int(a), int(b)


LOG_PATTERNS = {
    "strengthened_coefficients": re.compile(
        r"^Strengthened (\d+) coefficients"
    ),
    "domcol": re.compile(
        r"^Fixed (\d+) dominated columns and strengthened (\d+) bounds"
    ),
    "enumeration": re.compile(
        r"^Enumeration presolve: (\d+) deleted columns, (\d+) tightened bounds, (\d+) substitutions, "
        r"(\d+) deleted rows, (\d+) lifted nonzeros"
    ),
}


def parse_log(path):
    """Yields that HiGHS prints in its own log and the DIAG lines do not carry."""
    out = {
        "strengthened_coefficients": 0,
        "domcol_fixed": 0,
        "domcol_bounds": 0,
        "enum_tightened_bounds": 0,
        "enum_lifted": 0,
        "found": False,
    }
    if not path.exists():
        return out
    out["found"] = True
    with open(path, errors="replace") as f:
        for ln in f:
            m = LOG_PATTERNS["strengthened_coefficients"].match(ln)
            if m:
                out["strengthened_coefficients"] += int(m.group(1))
                continue
            m = LOG_PATTERNS["domcol"].match(ln)
            if m:
                out["domcol_fixed"] += int(m.group(1))
                out["domcol_bounds"] += int(m.group(2))
                continue
            m = LOG_PATTERNS["enumeration"].match(ln)
            if m:
                out["enum_tightened_bounds"] += int(m.group(2))
                out["enum_lifted"] += int(m.group(5))
    return out


# ----------------------------------------------------------------------------- data


class Inst:
    pass


def read_instances():
    testbed = load(TESTBED)
    stock = {
        x["name"]: x
        for x in load(STAGE0 / "stock" / "summary.json")["instances"]
    }
    diag = {
        x["name"]: x
        for x in load(STAGE0 / "diag" / "summary.json")["instances"]
    }
    out = []
    skipped = []
    for t in testbed["main"]:
        if t.get("where") != "dev":
            continue
        name = t["name"]
        s = stock.get(name)
        d = diag.get(name)
        stats_file = STAGE0 / "stats" / f"{name}.json"
        if (
            s is None
            or d is None
            or not stats_file.exists()
            or not d.get("diag")
            or s.get("presolve_core_seconds") is None
        ):
            skipped.append(name)
            continue
        i = Inst()
        i.name = name
        i.band = t["band"]
        i.rows, i.cols, i.nnz = s["rows"], s["cols"], s["nnz"]
        i.stock = s
        i.diag_row = d
        i.dg = d["diag"]
        i.t_stock = s["presolve_core_seconds"]
        i.t_diag = d["presolve_core_seconds"]
        i.t_total = i.dg["total"][0]["presolve_seconds"]
        i.stage = i.dg["stage"]
        i.sec = {st: i.stage[st]["seconds"] for st in STAGES}
        i.orig = load(stats_file)["original"]
        # structure of the presolved model (stock run); a stage that runs late in the main loop
        # sees a model between the original and this one
        stock_file = STAGE0 / "stock" / f"{name}.json"
        i.pre = (
            load(stock_file).get("presolved") if stock_file.exists() else None
        )
        i.red = {
            r["stage"]: {k: v for k, v in r.items() if k != "stage"}
            for r in i.dg.get("reductions", [])
        }
        i.log = parse_log(STAGE0 / "diag" / f"{name}.json.log")
        i.rows_ba = parse_pair(i.dg["total"][0]["rows"])
        i.cols_ba = parse_pair(i.dg["total"][0]["cols"])
        i.nnz_ba = parse_pair(i.dg["total"][0]["nnz"])
        i.probing = i.dg["probing"][0]
        i.rounds = i.dg.get("probing_round", [])
        i.cm_calls = i.dg.get("cliquemerge_call", [])
        i.parallel = i.dg["parallel"][0]
        i.vb = i.dg.get("varbound_aggregation", {})
        i.subst = i.dg["substitutions"][0]
        out.append(i)
    return out, skipped


def top_stages(i, k=3):
    return sorted(STAGES, key=lambda st: -i.sec[st])[:k]


VERDICTS = [
    "nothing",
    "almost nothing",
    "bounds or coefficients only",
    "productive",
]


def stage_yield(i, st):
    """What a stage removed on an instance, with the yields that are not removals.

    Verdict: 'nothing' = no row, column or nonzero removed, no postsolve entry, no bound
    tightened, no coefficient strengthened; 'productive' = removed rows + cols + nonzeros
    at least ALMOST_NOTHING of the model; 'bounds or coefficients only' = no row, column or
    nonzero removed, but bounds tightened or coefficients strengthened; 'almost nothing' =
    everything else below ALMOST_NOTHING (a run that removes a few entries and also tightens
    bounds is 'almost nothing', its bounds are in the by-products column).
    """
    s = i.stage[st]
    extra = []
    tightened = 0  # bounds tightened or coefficients strengthened: a yield that removes nothing
    entries = sum(i.red.get(st, {}).values())
    if st == "probing":
        p = i.probing
        for key, label in (
            ("fixed", "fixed"),
            ("substitutions", "subst"),
            ("bounds_tightened", "bounds"),
            ("lifted_nonzeros", "lifted"),
        ):
            if p.get(key, 0):
                extra.append(f"{label} {p[key]:,}")
        tightened = p.get("bounds_tightened", 0)
        new_cliques = p["cliques_after"] - p["cliques_before"]
        extra.append(f"cliques {new_cliques:+,}")
        extra.append(f"implications {p.get('implications', 0):,}")
    elif st == "enumeration":
        tightened = i.log["enum_tightened_bounds"]
        if tightened:
            extra.append(f"bounds {tightened:,}")
    elif st == "dominated_columns":
        tightened = i.log["domcol_bounds"]
        if tightened:
            extra.append(f"bounds {tightened:,}")
    elif st == "strengthen_inequalities":
        tightened = i.log["strengthened_coefficients"]
        if tightened:
            extra.append(f"coefficients {tightened:,}")
    removed = (
        abs(s["rows_removed"]) + abs(s["cols_removed"]) + abs(s["nnz_removed"])
    )
    frac = removed / float(i.rows + i.cols + i.nnz)
    if removed == 0 and entries == 0 and tightened == 0:
        verdict = "nothing"
    elif frac >= ALMOST_NOTHING:
        verdict = "productive"
    elif removed == 0 and tightened:
        verdict = "bounds or coefficients only"
    else:
        verdict = "almost nothing"
    return s, entries, extra, frac, verdict


PROGRESS = re.compile(r"^(\d+) rows, (\d+) cols, (\d+) nonzeros (\d+)s")


def progress_lines(path):
    """The 'R rows, C cols, Z nonzeros Ts' lines HiGHS prints at the top of its main loop
    (when the size fell by 15% or 10 s passed) and once at the end with the time of the
    last loop top. T is the whole second of the HiGHS clock.
    """
    out = []
    if not path.exists():
        return out
    with open(path, errors="replace") as f:
        for ln in f:
            m = PROGRESS.match(ln)
            if m:
                out.append(tuple(int(g) for g in m.groups()))
    return out


# ----------------------------------------------------------------------------- section 1


def section_1(insts):
    heading("1. Presolve time per instance")
    print(
        "Times are presolve_core_seconds (stock, diag). Stage seconds and shares are from the diag run "
        "(DIAG stage lines; share of DIAG total presolve_seconds). us/nnz is the stock time per original nonzero."
    )
    print()
    rows = []
    for i in insts:
        tops = top_stages(i)
        rows.append(
            [
                i.name,
                i.band,
                fn(i.rows),
                fn(i.cols),
                fn(i.nnz),
                fs(i.t_stock),
                fs(i.t_diag),
                f"{i.t_diag / i.t_stock:.3f}",
                f"{1e6 * i.t_stock / i.nnz:.2f}",
            ]
            + [f"{SHORT[st]} {pct(i.sec[st] / i.t_total)}" for st in tops]
        )
    table(
        [
            "instance",
            "band",
            "rows",
            "cols",
            "nnz",
            "stock s",
            "diag s",
            "diag/stock",
            "us/nnz",
            "1st stage",
            "2nd stage",
            "3rd stage",
        ],
        rows,
        right={2, 3, 4, 5, 6, 7, 8},
    )

    heading("1.1 Stage seconds (diag run)", 3)
    rows = []
    for i in insts:
        rows.append([i.name, fs(i.t_total)] + [fs(i.sec[st]) for st in STAGES])
    table(["instance", "total"] + [SHORT[st] for st in STAGES], rows)

    heading("1.2 Stage share of the presolve time, percent (diag run)", 3)
    rows = []
    for i in insts:
        rows.append(
            [i.name, fs(i.t_total, 2)]
            + [f"{100.0 * i.sec[st] / i.t_total:.1f}" for st in STAGES]
        )
    table(["instance", "total s"] + [SHORT[st] for st in STAGES], rows)

    heading("1.3 Per band and overall", 3)
    print(
        "SGM(s) = exp(mean(log(t + s))) - s. us/nnz is the median over the instances of the band."
    )
    print()
    rows = []
    for variant, key in (("stock", "t_stock"), ("diag", "t_diag")):
        for band in BANDS + ["all"]:
            sel = [i for i in insts if band == "all" or i.band == band]
            if not sel:
                continue
            ts = [getattr(i, key) for i in sel]
            rows.append(
                [
                    variant,
                    band,
                    str(len(sel)),
                    fs(min(ts)),
                    fs(statistics.median(ts)),
                    fs(geomean(ts)),
                    fs(sgm(ts, 0.1)),
                    fs(sgm(ts, 1.0)),
                    fs(max(ts)),
                    max(sel, key=lambda i: getattr(i, key)).name,
                    fs(sum(ts), 2),
                    f"{statistics.median([1e6 * getattr(i, key) / i.nnz for i in sel]):.2f}",
                ]
            )
    table(
        [
            "run",
            "band",
            "n",
            "min",
            "median",
            "geomean",
            "SGM(0.1)",
            "SGM(1)",
            "max",
            "max instance",
            "sum",
            "us/nnz",
        ],
        rows,
        right={2, 3, 4, 5, 6, 7, 8, 10, 11},
    )

    heading("1.4 Patched build against stock build", 3)
    ratios = [(i.t_diag / i.t_stock, i.name) for i in insts]
    rs = [r for r, _ in ratios]
    same = sum(
        1
        for i in insts
        if (i.stock["rows_after"], i.stock["cols_after"], i.stock["nnz_after"])
        == (
            i.diag_row["rows_after"],
            i.diag_row["cols_after"],
            i.diag_row["nnz_after"],
        )
    )
    print(
        f"- presolved sizes (rows, cols, nnz) identical in stock and diag: {same} of {len(insts)} instances"
    )
    print(
        f"- diag/stock time ratio: min {min(rs):.3f} ({min(ratios)[1]}), median {statistics.median(rs):.3f}, geomean {geomean(rs):.3f}, max {max(rs):.3f} ({max(ratios)[1]})"
    )
    for tol in (0.02, 0.05, 0.10):
        print(
            f"- instances with |diag/stock - 1| <= {tol:.2f}: {sum(1 for r in rs if abs(r - 1) <= tol)} of {len(rs)}"
        )
    big = [i for i in insts if i.t_stock >= 1.0]
    rb = [i.t_diag / i.t_stock for i in big]
    print(
        f"- instances with stock time >= 1 s ({len(big)}): ratio min {min(rb):.3f}, median {statistics.median(rb):.3f}, max {max(rb):.3f}"
    )
    small = [i for i in insts if i.t_stock < 1.0]
    worst = max(small, key=lambda i: abs(i.t_diag - i.t_stock))
    print(
        f"- instances with stock time below 1 s ({len(small)}): largest |diag - stock| {1e3 * abs(worst.t_diag - worst.t_stock):.1f} ms ({worst.name}), median {1e3 * statistics.median([abs(i.t_diag - i.t_stock) for i in small]):.1f} ms"
    )
    print(
        f"- sum over instances: stock {sum(i.t_stock for i in insts):.2f} s, diag {sum(i.t_diag for i in insts):.2f} s (ratio {sum(i.t_diag for i in insts) / sum(i.t_stock for i in insts):.4f})"
    )
    for shift in (0.1, 1.0):
        a = sgm([i.t_stock for i in insts], shift)
        b = sgm([i.t_diag for i in insts], shift)
        print(
            f"- SGM({shift:g}): stock {a:.4f} s, diag {b:.4f} s (ratio {b / a:.4f})"
        )
    gap = [abs(sum(i.sec.values()) - i.t_total) for i in insts]
    print(
        f"- largest |sum of stage seconds - DIAG total presolve_seconds|: {max(gap):.2e} s"
    )
    gap2 = [(abs(i.t_total - i.t_diag) / i.t_diag, i.name) for i in insts]
    print(
        f"- largest relative gap between DIAG total presolve_seconds and the tool's core_seconds of the same run: {max(gap2)[0]:.2e} ({max(gap2)[1]})"
    )
    print()
    rows = [[n, f"{r:.3f}"] for r, n in sorted(ratios) if abs(r - 1) > 0.02]
    if rows:
        print("Instances with a diag/stock ratio outside 0.98 to 1.02:")
        print()
        table(["instance", "diag/stock"], rows)


# ----------------------------------------------------------------------------- section 2


def section_2(insts):
    heading("2. Time against size")
    lx = [math.log10(i.nnz) for i in insts]
    ly = [math.log10(i.t_stock) for i in insts]
    a, b, r2, res = ols(lx, ly)
    sd = (sum(r * r for r in res) / (len(res) - 2)) ** 0.5
    ta, tb = theil_sen(lx, ly)
    print(
        "Fits of log10(time) on log10(size), stock times, 31 instances. R2 is the share of the variance of log10(time) that the fit explains."
    )
    print()
    rows = []
    rows.append(
        [
            "stock time",
            "nnz",
            f"{b:.2f}",
            f"{a:.2f}",
            f"{r2:.2f}",
            f"{sd:.2f}",
            f"{10**sd:.1f}x",
        ]
    )
    for label, f in (
        ("rows", lambda i: i.rows),
        ("cols", lambda i: i.cols),
        ("rows+cols", lambda i: i.rows + i.cols),
    ):
        xa, xb, xr2, xres = ols([math.log10(f(i)) for i in insts], ly)
        xsd = (sum(r * r for r in xres) / (len(xres) - 2)) ** 0.5
        rows.append(
            [
                "stock time",
                label,
                f"{xb:.2f}",
                f"{xa:.2f}",
                f"{xr2:.2f}",
                f"{xsd:.2f}",
                f"{10**xsd:.1f}x",
            ]
        )
    r2m = ols_multi(
        [
            lx,
            [math.log10(i.rows) for i in insts],
            [math.log10(i.cols) for i in insts],
        ],
        ly,
    )
    rows.append(
        [
            "stock time",
            "nnz, rows, cols (3 regressors)",
            "-",
            "-",
            f"{r2m:.2f}",
            "-",
            "-",
        ]
    )
    # parts of the diag time
    parts = collections.OrderedDict()
    parts["diag time minus its largest stage"] = lambda i: i.t_total - max(
        i.sec.values()
    )
    parts["setup+scan+shrink+scale (single passes)"] = lambda i: sum(
        i.sec[st]
        for st in ("setup", "initial_scan", "shrink", "final_scaling")
    )
    for g, sts in GROUPS.items():
        parts[f"group {g}"] = lambda i, sts=sts: sum(i.sec[st] for st in sts)
    for label, f in parts.items():
        ys = [math.log10(max(f(i), 1e-6)) for i in insts]
        xa, xb, xr2, xres = ols(lx, ys)
        xsd = (sum(r * r for r in xres) / (len(xres) - 2)) ** 0.5
        rows.append(
            [
                label,
                "nnz",
                f"{xb:.2f}",
                f"{xa:.2f}",
                f"{xr2:.2f}",
                f"{xsd:.2f}",
                f"{10**xsd:.1f}x",
            ]
        )
    table(
        [
            "time",
            "size",
            "slope",
            "intercept",
            "R2",
            "residual sd (log10)",
            "sd as factor",
        ],
        rows,
        right={2, 3, 4, 5, 6},
    )
    print(
        f"- Spearman rank correlation of stock time with nnz: {spearman(lx, ly):.2f}"
    )
    print(
        f"- least squares: time = {10**a:.3g} * nnz^{b:.2f} s; at 1e5 nnz {10 ** (a + 5 * b):.3f} s, at 1e6 {10 ** (a + 6 * b):.2f} s, at 5e6 {10 ** (a + b * math.log10(5e6)):.2f} s"
    )
    print(
        f"- Theil-Sen (median of pairwise slopes, robust to outliers): time = {10**ta:.3g} * nnz^{tb:.2f} s; at 1e5 nnz {10 ** (ta + 5 * tb):.3f} s, at 1e6 {10 ** (ta + 6 * tb):.2f} s, at 5e6 {10 ** (ta + tb * math.log10(5e6)):.2f} s"
    )
    print()
    heading("2.1 Distance from the trend", 3)
    print(
        f"Factor = stock time / fitted time. Listed: every instance; flagged when the factor against the Theil-Sen line is above {TREND_FACTOR:g} or below 1/{TREND_FACTOR:g}."
    )
    print()
    rows = []
    for i, x, y, r in sorted(
        zip(insts, lx, ly, res), key=lambda q: -(q[2] - (ta + tb * q[1]))
    ):
        f_ts = 10 ** (y - (ta + tb * x))
        flag = (
            "above"
            if f_ts > TREND_FACTOR
            else ("below" if f_ts < 1.0 / TREND_FACTOR else "")
        )
        tops = top_stages(i, 1)[0]
        rows.append(
            [
                i.name,
                i.band,
                fn(i.nnz),
                fs(i.t_stock),
                f"{10**r:.2f}",
                f"{f_ts:.2f}",
                flag,
                f"{SHORT[tops]} {pct(i.sec[tops] / i.t_total)}",
            ]
        )
    table(
        [
            "instance",
            "band",
            "nnz",
            "stock s",
            "factor (least squares)",
            "factor (Theil-Sen)",
            "flag",
            "largest stage",
        ],
        rows,
        right={2, 3, 4, 5},
    )
    return ta, tb


# ----------------------------------------------------------------------------- section 3


def dominant_cases(insts):
    out = []
    for i in insts:
        if i.t_total <= DOMINANT_MIN_SECONDS:
            continue
        for st in STAGES:
            if i.sec[st] > DOMINANT_MIN_SHARE * i.t_total:
                out.append((i, st))
    return out


def section_3(insts):
    heading("3. Where the time goes, and why")
    heading("3.1 Stage totals over the 31 instances", 3)
    total = sum(i.t_total for i in insts)
    rows = []
    for st in sorted(STAGES, key=lambda st: -sum(i.sec[st] for i in insts)):
        s = sum(i.sec[st] for i in insts)
        shares = [i.sec[st] / i.t_total for i in insts]
        big = max(insts, key=lambda i: i.sec[st])
        rows.append(
            [
                st,
                fs(s, 2),
                pct(s / total, 1),
                pct(sum(shares) / len(shares), 1),
                pct(statistics.median(shares), 1),
                str(sum(1 for i in insts if top_stages(i, 1)[0] == st)),
                str(sum(1 for i in insts if i.sec[st] > 0.5 * i.t_total)),
                f"{big.name} {fs(big.sec[st], 2)}",
            ]
        )
    rows.append(
        ["all", fs(total, 2), "100.0%", "", "", str(len(insts)), "", ""]
    )
    table(
        [
            "stage",
            "sum s",
            "share of sum",
            "mean share",
            "median share",
            "largest on n",
            "above 50% on n",
            "largest single run",
        ],
        rows,
        right={1, 2, 3, 4, 5, 6},
    )

    heading("3.2 Share of the time by rule group, per band", 3)
    print(
        "Sum of the stage seconds of the group over the band, divided by the sum of the presolve seconds of the band; in brackets the mean of the per-instance shares."
    )
    print()
    rows = []
    for band in BANDS + ["all"]:
        sel = [i for i in insts if band == "all" or i.band == band]
        tot = sum(i.t_total for i in sel)
        r = [band, str(len(sel)), fs(tot, 2)]
        for g, sts in GROUPS.items():
            gs = sum(i.sec[st] for i in sel for st in sts)
            ms = sum(
                sum(i.sec[st] for st in sts) / i.t_total for i in sel
            ) / len(sel)
            r.append(f"{pct(gs / tot)} ({pct(ms)})")
        rows.append(r)
    table(["band", "n", "sum s"] + list(GROUPS.keys()), rows)

    heading(
        f"3.3 Instances above {DOMINANT_MIN_SECONDS:g} s where one stage takes more than half",
        3,
    )
    cases = dominant_cases(insts)
    rows = []
    for i, st in sorted(cases, key=lambda c: -c[0].sec[c[1]]):
        s = i.stage[st]
        o = i.orig
        rows.append(
            [
                i.name,
                i.band,
                fs(i.t_total, 2),
                st,
                fs(i.sec[st], 2),
                pct(i.sec[st] / i.t_total),
                str(s["calls"]),
                f"{o['row_length']['p50']}/{o['row_length']['p99']}/{o['row_length']['max']}",
                f"{o['col_length']['p50']}/{o['col_length']['p99']}/{o['col_length']['max']}",
                str(o["long_rows"]["count"]),
                str(o["long_cols"]["count"]),
            ]
        )
    table(
        [
            "instance",
            "band",
            "total s",
            "stage",
            "stage s",
            "share",
            "calls",
            "row len p50/p99/max",
            "col len p50/p99/max",
            "long rows",
            "long cols",
        ],
        rows,
        right={2, 4, 5, 6, 7, 8, 9, 10},
    )
    print(
        f"Sum of these dominant stage runs: {sum(i.sec[st] for i, st in cases):.2f} s of {total:.2f} s ({pct(sum(i.sec[st] for i, st in cases) / total)})."
    )
    print()

    # ---- row removal
    heading("3.4 Cost of removing rows: parallel rows and the fast loop", 3)
    print(
        "us per removed nonzero = seconds / nonzeros the stage removed (parallel: seconds of the row loop). "
        "'ns per column entry' divides that by the longest original column: the cost per entry if every removed "
        "nonzero walked a column of that length (a lower bound on the per-entry cost under that explanation). "
        "Listed: stage runs that removed at least 1,000 nonzeros."
    )
    print()
    rows = []
    for i in insts:
        cl = i.orig["col_length"]
        for st, secs in (
            ("parallel_rows_cols", i.parallel.get("row_seconds", 0.0)),
            ("fast_loop", i.sec["fast_loop"]),
        ):
            s = i.stage[st]
            if s["nnz_removed"] < 1000 or s["rows_removed"] <= 0:
                continue
            us = 1e6 * secs / s["nnz_removed"]
            rows.append(
                (
                    us,
                    [
                        i.name,
                        "parallel (row loop)"
                        if st == "parallel_rows_cols"
                        else "fast_loop",
                        fs(secs),
                        fn(s["rows_removed"]),
                        fn(s["cols_removed"]),
                        fn(s["nnz_removed"]),
                        f"{1e6 * secs / s['rows_removed']:.1f}",
                        f"{us:.2f}",
                        f"{cl['p50']}/{cl['p99']}/{cl['max']}",
                        f"{1e3 * us / cl['max']:.2f}",
                    ],
                )
            )
    table(
        [
            "instance",
            "stage",
            "s",
            "rows removed",
            "cols removed",
            "nnz removed",
            "us per row",
            "us per nnz",
            "col len p50/p99/max",
            "ns per column entry",
        ],
        [r for _, r in sorted(rows, key=lambda q: -q[0])],
        right={2, 3, 4, 5, 6, 7, 8, 9},
    )

    heading("3.5 Parallel rows and columns: hashing, column loop, row loop", 3)
    print(
        "ns/nnz/call = hash seconds / (original nnz * calls): a lower estimate of the cost per live nonzero, because later calls hash a smaller model."
    )
    print()
    rows = []
    for i in sorted(insts, key=lambda i: -i.sec["parallel_rows_cols"]):
        p = i.parallel
        s = i.stage["parallel_rows_cols"]
        if s["seconds"] < 0.05:
            continue
        rows.append(
            [
                i.name,
                str(s["calls"]),
                fs(s["seconds"]),
                pct(s["seconds"] / i.t_total),
                fs(p["hash_seconds"]),
                fs(p["col_seconds"]),
                fs(p["row_seconds"]),
                fs(
                    s["seconds"]
                    - p["hash_seconds"]
                    - p["col_seconds"]
                    - p["row_seconds"]
                ),
                f"{1e9 * p['hash_seconds'] / (i.nnz * max(s['calls'], 1)):.0f}",
                fn(p["duplicate_rows"]),
                fn(p["row_additions"]),
                fn(p["cols_merged"] + p["cols_dominated"]),
                fn(s["rows_removed"]),
                fn(s["cols_removed"]),
                fn(s["nnz_removed"]),
            ]
        )
    table(
        [
            "instance",
            "calls",
            "stage s",
            "share",
            "hash s",
            "col loop s",
            "row loop s",
            "rest s",
            "hash ns/nnz/call",
            "dup rows",
            "row adds",
            "cols merged+dominated",
            "rows rem",
            "cols rem",
            "nnz rem",
        ],
        rows,
    )

    # ---- dominated columns
    heading("3.6 Dominated columns", 3)
    print(
        "'fixed' and 'bounds' are HiGHS's own log line ('Fixed N dominated columns and strengthened M bounds', summed over the calls); "
        "'fixed' counts fixings only, 'cols rem' also holds the columns that the doubleton equations following a fixing substitute. "
        "'pairs bound' = long columns * longest row * 4 (two rows per column, two signs): an upper bound on the column pairs per call "
        "only if the cost sits in the long columns (a pair of a short column with a long one also walks the long column)."
    )
    print()
    rows = []
    for i in sorted(insts, key=lambda i: -i.sec["dominated_columns"]):
        s = i.stage["dominated_columns"]
        if s["seconds"] < 0.02:
            continue
        o = i.orig
        lc = o["long_cols"]["count"]
        bound = lc * o["row_length"]["max"] * 4
        rows.append(
            [
                i.name,
                str(s["calls"]),
                fs(s["seconds"]),
                pct(s["seconds"] / i.t_total),
                fn(i.cols),
                f"{o['row_length']['p10']}/{o['row_length']['p50']}/{o['row_length']['p90']}/{o['row_length']['max']}",
                f"{o['col_length']['p50']}/{o['col_length']['p90']}/{o['col_length']['max']}",
                str(o["long_rows"]["count"]),
                str(lc),
                f"{1e9 * s['seconds'] / max(s['calls'], 1) / (i.cols * o['row_length']['p50']):.0f}",
                fg(bound) if lc else "-",
                f"{1e6 * s['seconds'] / max(s['calls'], 1) / bound:.2f}"
                if lc
                else "-",
                fn(i.log["domcol_fixed"]),
                fn(i.log["domcol_bounds"]),
                fn(s["rows_removed"]),
                fn(s["cols_removed"]),
                fn(s["nnz_removed"]),
            ]
        )
    table(
        [
            "instance",
            "calls",
            "s",
            "share",
            "cols",
            "row len p10/p50/p90/max",
            "col len p50/p90/max",
            "long rows",
            "long cols",
            "ns per (col * median row) per call",
            "pairs bound per call",
            "us per pair of the bound",
            "fixed",
            "bounds",
            "rows rem",
            "cols rem",
            "nnz rem",
        ],
        rows,
    )
    print(
        "The structure columns above are those of the ORIGINAL model. dominatedColumns is called late in a main-loop pass (after the fast loop, "
        "the aggregator, sparsify and parallel rows; a pass in which parallel rows removes more than 0.05% of the rows or columns restarts the loop "
        "before it), so the stage sees a model between the original and the presolved one. The table below gives the presolved structure "
        "(stock/<name>.json 'presolved') for the runs of at least 0.5 s. 'last loop-top line' is the last progress line HiGHS printed at a top of "
        "its main loop (the final line of a log is printed after the loop and is left out). 'stage s after that line, at least' = stage seconds minus "
        "(stamp of that line + 1 - setup - initial scan): that part of the stage ran on the model of that line or a smaller one."
    )
    print()
    rows = []
    for i in sorted(insts, key=lambda i: -i.sec["dominated_columns"]):
        s = i.stage["dominated_columns"]
        if s["seconds"] < 0.5 or i.pre is None:
            continue
        o, p = i.orig, i.pre
        lines = progress_lines(STAGE0 / "diag" / f"{i.name}.json.log")
        last = lines[-2] if len(lines) >= 2 else None
        after = (
            max(
                s["seconds"]
                - max(
                    last[3] + 1 - i.sec["setup"] - i.sec["initial_scan"], 0.0
                ),
                0.0,
            )
            if last
            else None
        )
        lcp = p["long_cols"]["count"]
        boundp = lcp * p["row_length"]["max"] * 4
        rows.append(
            [
                i.name,
                fs(s["seconds"]),
                f"{fn(o['rows'])} / {fn(o['cols'])} / {fn(o['nnz'])}",
                f"{fn(p['rows'])} / {fn(p['cols'])} / {fn(p['nnz'])}",
                f"{o['long_rows']['count']} -> {p['long_rows']['count']}",
                f"{o['long_cols']['count']} -> {lcp}",
                f"{o['col_length']['p90']}/{o['col_length']['max']} -> {p['col_length']['p90']}/{p['col_length']['max']}",
                f"{o['row_length']['p90']}/{o['row_length']['max']} -> {p['row_length']['p90']}/{p['row_length']['max']}",
                fg(boundp) if lcp else "-",
                f"{1e6 * s['seconds'] / max(s['calls'], 1) / boundp:.2f}"
                if lcp
                else "-",
                f"{last[0]:,} / {last[1]:,} / {last[2]:,} at {last[3]}s"
                if last
                else "-",
                fs(after, 2) if after is not None else "-",
            ]
        )
    table(
        [
            "instance",
            "stage s",
            "original rows / cols / nnz",
            "presolved rows / cols / nnz",
            "long rows",
            "long cols",
            "col len p90/max",
            "row len p90/max",
            "pairs bound per call, presolved",
            "us per pair of that bound",
            "last loop-top line (rows / cols / nnz at stamp)",
            "stage s after that line, at least",
        ],
        rows,
    )

    # ---- probing
    heading("3.7 Probing: cost", 3)
    print(
        "work = probing_work (propagated row entries + activity updates per bound change), in the probing loop. "
        "work/nnz uses the original nnz. 'per probe/nnz' = work per probe as a fraction of the original nonzeros "
        "(1.0 = one pass over the whole matrix per probed variable). ns/work = loop seconds / work: about 10 ns when "
        "propagation is the cost, more when the time is outside the counted work (clique table, implication storage)."
    )
    print()
    rows = []
    for i in sorted(insts, key=lambda i: -i.sec["probing"]):
        p = i.probing
        if not p.get("probed"):
            continue
        work = p.get("probing_work", 0)
        loop = p.get("loop_seconds", 0.0)
        rows.append(
            [
                i.name,
                fs(i.sec["probing"]),
                pct(i.sec["probing"] / i.t_total),
                fs(loop),
                fs(sum(r["prepare_seconds"] for r in i.rounds)),
                fs(sum(r["finalise_seconds"] for r in i.rounds)),
                str(p["rounds"]),
                fn(p["binaries"]),
                fn(p["probed"]),
                p["stop"],
                fg(work),
                f"{work / i.nnz:.0f}",
                f"{work / p['probed'] / i.nnz:.3f}",
                f"{1e9 * loop / work:.0f}" if work else "-",
                pct(p.get("prop_entries", 0) / work) if work else "-",
                f"{p.get('prop_sweeps', 0) / p['probed']:.1f}",
                f"{p.get('prop_rows', 0) / max(p.get('prop_sweeps', 0), 1):.0f}",
                f"{p.get('prop_entries', 0) / max(p.get('prop_rows', 0), 1):.0f}",
                f"{1e3 * loop / p['probed']:.2f}",
            ]
        )
    table(
        [
            "instance",
            "stage s",
            "share",
            "loop s",
            "prepare s",
            "finalise s",
            "rounds",
            "binaries",
            "probed",
            "stop (last round)",
            "work",
            "work/nnz",
            "per probe/nnz",
            "ns/work",
            "row scans in work",
            "sweeps/probe",
            "rows/sweep",
            "entries/row scan",
            "ms/probe",
        ],
        rows,
        right={1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18},
    )

    heading("3.8 Probing: yield", 3)
    print(
        "'cols/s' = columns removed by the probing stage per second of the stage. 'found per 1e6 work' = (fixed + substitutions + bounds tightened) per million work units."
    )
    print()
    rows = []
    for i in sorted(insts, key=lambda i: -i.sec["probing"]):
        p = i.probing
        if not p.get("probed"):
            continue
        s = i.stage["probing"]
        found = (
            p.get("fixed", 0)
            + p.get("substitutions", 0)
            + p.get("bounds_tightened", 0)
        )
        work = p.get("probing_work", 0)
        rows.append(
            [
                i.name,
                fs(i.sec["probing"]),
                fn(p["probed"]),
                fn(p.get("fixed", 0)),
                fn(p.get("substitutions", 0)),
                fn(p.get("bounds_tightened", 0)),
                fn(p.get("lifted_nonzeros", 0)),
                fn(p["cliques_before"]),
                fn(p["cliques_after"]),
                fn(p.get("implications", 0)),
                fn(s["rows_removed"]),
                fn(s["cols_removed"]),
                fn(s["nnz_removed"]),
                pct(s["cols_removed"] / i.cols, 2),
                f"{s['cols_removed'] / i.sec['probing']:.0f}",
                f"{1e6 * found / work:.2f}" if work else "-",
            ]
        )
    table(
        [
            "instance",
            "stage s",
            "probed",
            "fixed",
            "subst",
            "bounds",
            "lifted nnz",
            "cliques before",
            "cliques after",
            "implications",
            "rows rem",
            "cols rem",
            "nnz rem",
            "cols rem / cols",
            "cols/s",
            "found per 1e6 work",
        ],
        rows,
    )

    heading(
        "3.9 Probing rounds of the instances where probing takes more than half",
        3,
    )
    rows = []
    for i, st in sorted(dominant_cases(insts), key=lambda c: -c[0].sec[c[1]]):
        if st != "probing":
            continue
        for r in i.rounds:
            rows.append(
                [
                    i.name,
                    str(r["round"]),
                    fn(r["binaries"]),
                    fn(r["probed"]),
                    r["stop"],
                    fs(r["seconds"]),
                    fs(r["loop_seconds"]),
                    fn(r["fixed"]),
                    fn(r["substitutions"]),
                    fn(r["bounds_tightened"]),
                    fn(r["cliques_after"] - r["cliques_before"]),
                    fn(r["implications"] - r["implications_before"]),
                    fg(r["probing_work"]),
                    f"{1e9 * r['loop_seconds'] / r['probing_work']:.0f}"
                    if r["probing_work"]
                    else "-",
                    f"{r['prop_entries'] / max(r['prop_rows'], 1):.0f}",
                    fn(r["max_probe_work"]),
                ]
            )
    table(
        [
            "instance",
            "round",
            "binaries",
            "probed",
            "stop",
            "s",
            "loop s",
            "fixed",
            "subst",
            "bounds",
            "new cliques",
            "new implications",
            "work",
            "ns/work",
            "entries/row scan",
            "max work of one probe",
        ],
        rows,
        right={1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    )

    # ---- clique merging
    heading("3.10 Clique merging", 3)
    print(
        "A call 'changed nothing' when rows_changed, rows_deleted and fixings are all 0. Listed: instances with at least 0.05 s of clique merging."
    )
    print()
    rows = []
    for i in sorted(
        insts, key=lambda i: -sum(c["seconds"] for c in i.cm_calls)
    ):
        tot = sum(c["seconds"] for c in i.cm_calls)
        if tot < 0.05:
            continue
        idle = [
            c
            for c in i.cm_calls
            if c["rows_changed"] == 0
            and c["rows_deleted"] == 0
            and c["fixings"] == 0
        ]
        rows.append(
            [
                i.name,
                str(len(i.cm_calls)),
                fs(tot),
                pct(tot / i.t_total),
                str(len(idle)),
                fs(sum(c["seconds"] for c in idle)),
                pct(sum(c["seconds"] for c in idle) / i.t_total),
                fn(sum(c["rows_changed"] for c in i.cm_calls)),
                fn(sum(c["rows_deleted"] for c in i.cm_calls)),
                fn(sum(c["nnz_added"] for c in i.cm_calls)),
                fn(sum(c["nnz_removed"] for c in i.cm_calls)),
                fn(max(c["cliques_before"] for c in i.cm_calls)),
            ]
        )
    table(
        [
            "instance",
            "calls",
            "merge s",
            "share",
            "calls that changed nothing",
            "their s",
            "their share",
            "rows changed",
            "rows deleted",
            "nnz added",
            "nnz removed",
            "max cliques before",
        ],
        rows,
    )
    rows = []
    for i in insts:
        if sum(c["seconds"] for c in i.cm_calls) < 1.0:
            continue
        for k, c in enumerate(i.cm_calls, 1):
            rows.append(
                [
                    i.name,
                    str(k),
                    c["stage"],
                    fs(c["seconds"]),
                    fn(c["cliques_before"]),
                    fn(c["cliques_after"]),
                    fn(c["rows_changed"]),
                    fn(c["rows_deleted"]),
                    fn(c["nnz_added"]),
                    fn(c["nnz_removed"]),
                    fn(c["fixings"]),
                ]
            )
    print("Calls of the instances with at least 1 s of clique merging:")
    print()
    table(
        [
            "instance",
            "call",
            "in stage",
            "s",
            "cliques before",
            "cliques after",
            "rows changed",
            "rows deleted",
            "nnz added",
            "nnz removed",
            "fixings",
        ],
        rows,
        right={1, 3, 4, 5, 6, 7, 8, 9, 10},
    )

    # ---- enumeration
    heading("3.11 Enumeration stage: what is inside", 3)
    print(
        "'merge s' = clique merging inside enumeration calls, 'varbound s' = variable-bound aggregation, 'rest' = prepareProbing (shrink, matrix rebuild, propagation, clique extraction), the enumeration itself and applying the results. Listed: at least 0.3 s."
    )
    print()
    rows = []
    for i in sorted(insts, key=lambda i: -i.sec["enumeration"]):
        s = i.stage["enumeration"]
        if s["seconds"] < 0.3:
            continue
        cm = sum(
            c["seconds"] for c in i.cm_calls if c["stage"] == "enumeration"
        )
        vb = i.vb.get("enumeration", {}).get("seconds", 0.0)
        rows.append(
            [
                i.name,
                str(s["calls"]),
                fs(s["seconds"]),
                pct(s["seconds"] / i.t_total),
                fs(cm),
                fs(vb),
                fs(s["seconds"] - cm - vb),
                f"{1e9 * (s['seconds'] - cm - vb) / (s['calls'] * i.nnz):.0f}",
                fn(s["rows_removed"]),
                fn(s["cols_removed"]),
                fn(s["nnz_removed"]),
                fn(i.log["enum_tightened_bounds"]),
            ]
        )
    table(
        [
            "instance",
            "calls",
            "stage s",
            "share",
            "merge s",
            "varbound s",
            "rest s",
            "rest ns per original nnz per call",
            "rows rem",
            "cols rem",
            "nnz rem",
            "bounds tightened",
        ],
        rows,
    )

    # ---- shrink
    heading(
        "3.12 Shrink stage against the size of the clique table and the implication store",
        3,
    )
    print("Listed: shrink at least 0.1 s or at least 5% of the presolve.")
    print()
    rows = []
    for i in sorted(insts, key=lambda i: -i.sec["shrink"]):
        s = i.stage["shrink"]
        if s["seconds"] < 0.1 and s["seconds"] < 0.05 * i.t_total:
            continue
        p = i.probing
        rows.append(
            [
                i.name,
                str(s["calls"]),
                fs(s["seconds"]),
                pct(s["seconds"] / i.t_total),
                fn(i.nnz),
                fn(p["cliques_after"]),
                fn(p.get("implications", 0)),
                f"{1e9 * s['seconds'] / (s['calls'] * i.nnz):.0f}",
            ]
        )
    table(
        [
            "instance",
            "calls",
            "shrink s",
            "share",
            "nnz",
            "cliques after probing",
            "implications",
            "ns per original nnz per call",
        ],
        rows,
    )

    # ---- passes
    heading(
        "3.13 Cost of the passes over the model, ns per original nonzero", 3
    )
    print(
        "setup and initial scan run once on the full model. The other columns divide by the calls and by the original nonzeros, although later calls "
        "see a smaller model, so they are lower estimates of the cost per live nonzero: parallel hash (hash_seconds), strengthen inequalities, "
        "probing prepare (prepare_seconds per round: shrink, matrix rebuild, propagation, clique extraction), fast loop per call."
    )
    print()
    rows = []
    cols = collections.defaultdict(list)
    for i in insts:
        vals = collections.OrderedDict()
        vals["setup"] = 1e9 * i.sec["setup"] / i.nnz
        vals["initial scan"] = 1e9 * i.sec["initial_scan"] / i.nnz
        c = i.stage["parallel_rows_cols"]["calls"]
        vals["parallel hash / call"] = (
            1e9 * i.parallel["hash_seconds"] / (c * i.nnz) if c else None
        )
        c = i.stage["strengthen_inequalities"]["calls"]
        vals["strengthen / call"] = (
            1e9 * i.sec["strengthen_inequalities"] / (c * i.nnz) if c else None
        )
        vals["probing prepare / round"] = (
            1e9
            * sum(r["prepare_seconds"] for r in i.rounds)
            / (len(i.rounds) * i.nnz)
            if i.rounds
            else None
        )
        c = i.stage["fast_loop"]["calls"]
        vals["fast loop / call"] = (
            1e9 * i.sec["fast_loop"] / (c * i.nnz) if c else None
        )
        for k, v in vals.items():
            if v is not None:
                cols[k].append(v)
        rows.append(
            [i.name, i.band, fn(i.nnz)]
            + [f"{v:.0f}" if v is not None else "-" for v in vals.values()]
        )
    names = list(vals.keys())
    rows.append(
        ["median", "", ""]
        + [f"{statistics.median(cols[k]):.0f}" for k in names]
    )
    rows.append(["minimum", "", ""] + [f"{min(cols[k]):.0f}" for k in names])
    rows.append(["maximum", "", ""] + [f"{max(cols[k]):.0f}" for k in names])
    table(
        ["instance", "band", "nnz"] + names,
        rows,
        right=set(range(2, 3 + len(names))),
    )

    # ---- progress lines
    heading(
        "3.14 When the fast loop removes its rows (HiGHS progress lines of the diag run)",
        3,
    )
    print(
        "HiGHS prints 'R rows, C cols, Z nonzeros Ts' at a top of its main loop; T is the whole second of its clock, so the presolve had run less than T + 1 s. "
        "'fast-loop rows by then, at least' = rows removed so far minus everything the other stages removed in the whole run. "
        "'fast-loop s by then, at most' = T + 1 minus setup and initial scan. The last line of a log repeats the time of the last loop top and is left out. "
        "Listed: instances with at least 5 s in the fast loop."
    )
    print()
    rows = []
    for i in insts:
        if i.sec["fast_loop"] < 5.0:
            continue
        other_rows = sum(
            max(i.stage[st]["rows_removed"], 0)
            for st in STAGES
            if st != "fast_loop"
        )
        fl = i.stage["fast_loop"]
        lines = progress_lines(STAGE0 / "diag" / f"{i.name}.json.log")
        for r, c, z, t in lines[1:-1]:
            removed = i.rows - r
            lb = max(removed - other_rows, 0)
            ub_s = min(
                max(t + 1 - i.sec["setup"] - i.sec["initial_scan"], 0.0),
                fl["seconds"],
            )
            rows.append(
                [
                    i.name,
                    f"{t}s",
                    fn(r),
                    fn(removed),
                    fn(fl["rows_removed"]),
                    fn(lb),
                    fs(fl["seconds"], 2),
                    fs(ub_s, 2),
                    fs(fl["seconds"] - ub_s, 2),
                    fn(fl["rows_removed"] - lb),
                ]
            )
    table(
        [
            "instance",
            "line stamp",
            "rows at the line",
            "rows removed so far",
            "fast-loop rows, whole run",
            "fast-loop rows by then, at least",
            "fast-loop s, whole run",
            "fast-loop s by then, at most",
            "fast-loop s after, at least",
            "fast-loop rows after, at most",
        ],
        rows,
    )

    # ---- optional: the run with probing switched off
    np_file = STAGE0 / "diag_noprobing" / "summary.json"
    if np_file.exists():
        heading(
            "3.15 Supplement: the same stages with probing switched off (diag_noprobing, HIGHS_DIAG_OFF=probing)",
            3,
        )
        print(
            "This directory is not one of the inputs named for the Stage 0 time analysis. Its counts are deterministic; its times are single runs, "
            "and its summary does not record how many instances ran at the same time. Listed: instances with at least 1 s in the fast loop, "
            "parallel rows or enumeration of the diag run."
        )
        print()
        nop = {
            x["name"]: x for x in load(np_file)["instances"] if x.get("diag")
        }
        rows = []
        for i in insts:
            if i.name not in nop:
                continue
            n = nop[i.name]["diag"]["stage"]
            for st in ("fast_loop", "parallel_rows_cols", "enumeration"):
                if i.sec[st] < 1.0:
                    continue
                a = i.stage[st]
                b = n[st]
                rows.append(
                    [
                        i.name,
                        st,
                        str(a["calls"]),
                        fs(a["seconds"], 2),
                        fn(a["rows_removed"]),
                        fn(a["nnz_removed"]),
                        str(b["calls"]),
                        fs(b["seconds"], 2),
                        fn(b["rows_removed"]),
                        fn(b["nnz_removed"]),
                    ]
                )
        table(
            [
                "instance",
                "stage",
                "calls",
                "s",
                "rows rem",
                "nnz rem",
                "calls, no probing",
                "s, no probing",
                "rows rem, no probing",
                "nnz rem, no probing",
            ],
            rows,
            right={2, 3, 4, 5, 6, 7, 8, 9},
        )
        rows = []
        for name in ("rd-rplusc-21", "neos-4306827-ravan"):
            if name not in nop:
                continue
            tot = nop[name]["diag"]["total"][0]["presolve_seconds"]
            for r, c, z, t in progress_lines(
                STAGE0 / "diag_noprobing" / f"{name}.json.log"
            ):
                rows.append([name, f"{t}s", fn(r), fn(c), fn(z), fs(tot, 2)])
        if rows:
            print(
                "Progress lines of the run with probing off (the last line of each instance repeats the time of the last loop top):"
            )
            print()
            table(
                [
                    "instance",
                    "line stamp",
                    "rows",
                    "cols",
                    "nnz",
                    "presolve s of the run",
                ],
                rows,
            )


# ----------------------------------------------------------------------------- section 4


def section_4(insts):
    heading("4. What the time bought")
    print(
        f"Every run of a reduction stage with at least {YIELD_MIN_SECONDS:g} s and at least {pct(YIELD_MIN_SHARE)} of the presolve. "
        "'removed/model' = (rows + cols + nonzeros removed by the stage) / (rows + cols + nonzeros of the model). "
        f"Verdict: 'nothing' = no row, column or nonzero removed, no postsolve entry, no bound tightened, no coefficient strengthened; "
        f"'productive' = removed/model at least {pct(ALMOST_NOTHING, 1)}; 'bounds or coefficients only' = no row, column or nonzero removed, but bounds tightened or "
        "coefficients strengthened (probing: DIAG bounds_tightened; enumeration, dominated columns, strengthen inequalities: HiGHS's own log lines); "
        f"'almost nothing' = everything else below {pct(ALMOST_NOTHING, 1)}. "
        "For probing, new cliques and implications are listed but do not count as a yield of the presolve. "
        "The shares of the instance's removed rows / cols / nnz are printed with one decimal: 100.0% can still hide a remainder below 0.05%."
    )
    print()
    rows = []
    for i in insts:
        for st in REDUCTION_STAGES:
            if (
                i.sec[st] < YIELD_MIN_SECONDS
                or i.sec[st] < YIELD_MIN_SHARE * i.t_total
            ):
                continue
            s, entries, extra, frac, verdict = stage_yield(i, st)
            rr = i.rows_ba[0] - i.rows_ba[1]
            cc = i.cols_ba[0] - i.cols_ba[1]
            zz = i.nnz_ba[0] - i.nnz_ba[1]
            rows.append(
                (
                    -i.sec[st],
                    [
                        i.name,
                        st,
                        fs(i.sec[st], 2),
                        pct(i.sec[st] / i.t_total),
                        fn(s["rows_removed"]),
                        fn(s["cols_removed"]),
                        fn(s["nnz_removed"]),
                        f"{pct(s['rows_removed'] / rr, 1) if rr else '-'} / {pct(s['cols_removed'] / cc, 1) if cc else '-'} / {pct(s['nnz_removed'] / zz, 1) if zz else '-'}",
                        "; ".join(extra) if extra else "",
                        pct(frac, 2),
                        verdict,
                    ],
                )
            )
    table(
        [
            "instance",
            "stage",
            "s",
            "share",
            "rows rem",
            "cols rem",
            "nnz rem",
            "share of the instance's removed rows / cols / nnz",
            "other yield and by-products",
            "removed/model",
            "verdict",
        ],
        [r for _, r in sorted(rows)],
        right={2, 3, 4, 5, 6, 9},
    )

    heading(
        "4.1 Reduction-stage runs that removed nothing or almost nothing", 3
    )
    total = sum(i.t_total for i in insts)
    acc = collections.defaultdict(float)
    listed = []
    for i in insts:
        for st in REDUCTION_STAGES:
            if i.stage[st]["calls"] == 0:
                continue
            s, entries, extra, frac, verdict = stage_yield(i, st)
            acc[verdict] += i.sec[st]
            acc[(verdict, st)] += i.sec[st]
            if verdict != "productive" and i.sec[st] >= 0.25:
                listed.append(
                    (
                        -i.sec[st],
                        [
                            i.name,
                            st,
                            fs(i.sec[st], 2),
                            pct(i.sec[st] / i.t_total),
                            verdict,
                            fn(s["rows_removed"]),
                            fn(s["cols_removed"]),
                            fn(s["nnz_removed"]),
                            "; ".join(extra),
                        ],
                    )
                )
    print(
        f"Sum of the presolve seconds of the {len(insts)} instances: {total:.2f} s."
    )
    for v in VERDICTS:
        print(
            f"- reduction-stage runs with verdict '{v}': {acc[v]:.2f} s ({pct(acc[v] / total, 1)} of the sum)"
        )
    print(
        f"- stages that are not reduction stages (setup, shrink, final scaling, implied-integer sweep, clique substitutions, other): {total - sum(acc[v] for v in VERDICTS):.2f} s"
    )
    print()
    rows = []
    for st in REDUCTION_STAGES:
        rows.append([st] + [fs(acc[(v, st)], 2) for v in VERDICTS])
    table(["stage"] + [v + " s" for v in VERDICTS], rows)
    print("Runs of at least 0.25 s with a verdict other than 'productive':")
    print()
    table(
        [
            "instance",
            "stage",
            "s",
            "share",
            "verdict",
            "rows rem",
            "cols rem",
            "nnz rem",
            "other yield and by-products",
        ],
        [r for _, r in sorted(listed)],
        right={2, 3, 5, 6, 7},
    )

    heading(
        "4.2 Instances where the largest stage removed nothing or almost nothing",
        3,
    )
    rows = []
    for i in sorted(insts, key=lambda i: -i.t_total):
        st = top_stages(i, 1)[0]
        if st not in REDUCTION_STAGES:
            continue
        s, entries, extra, frac, verdict = stage_yield(i, st)
        if verdict == "productive":
            continue
        rows.append(
            [
                i.name,
                fs(i.t_total, 2),
                st,
                fs(i.sec[st], 2),
                pct(i.sec[st] / i.t_total),
                verdict,
                fn(s["rows_removed"]),
                fn(s["cols_removed"]),
                fn(s["nnz_removed"]),
                "; ".join(extra),
            ]
        )
    table(
        [
            "instance",
            "total s",
            "largest stage",
            "s",
            "share",
            "verdict",
            "rows rem",
            "cols rem",
            "nnz rem",
            "other yield and by-products",
        ],
        rows,
        right={1, 3, 4, 6, 7, 8},
    )


# ----------------------------------------------------------------------------- section 5


def section_5(insts):
    heading("5. Numbers for time targets")
    heading("5.1 Time by rule group, per band (diag run)", 3)
    print(
        "Each cell: SGM(0.1) / SGM(1) / max of the seconds in the stages of the group. The groups follow DESIGN.md sections 5 and 13 by the HiGHS stage that holds "
        "the rule: 'rows+cols loop' = setup, initial scan, fast loop, parallel rows and columns, shrink, final scaling, other (Stage 1 rules, and the Stage 3 "
        "column rules HiGHS runs inside its column presolve); 'probing+enum' = probing, enumeration, clique substitutions, implied-integer sweep (Stage 2, "
        "enumeration is left out of the first versions); 'aggr+domcol+fm' = aggregator, dominated columns, Fourier-Motzkin (Stage 3; Fourier-Motzkin is left out); "
        "'spars+streng' = sparsify, strengthen inequalities (Stage 4). The sum of stage seconds is not the time HiGHS would take with only that group switched on."
    )
    print()
    rows = []
    for band in BANDS + ["all"]:
        sel = [i for i in insts if band == "all" or i.band == band]
        r = [band, str(len(sel))]
        ts = [i.t_total for i in sel]
        r.append(f"{sgm(ts, 0.1):.3f} / {sgm(ts, 1.0):.3f} / {max(ts):.2f}")
        for g, sts in GROUPS.items():
            gs = [sum(i.sec[st] for st in sts) for i in sel]
            r.append(
                f"{sgm(gs, 0.1):.3f} / {sgm(gs, 1.0):.3f} / {max(gs):.2f}"
            )
        rows.append(r)
    table(["band", "n", "full presolve"] + list(GROUPS.keys()), rows)

    heading("5.2 The 20% allowance against GPU fixed costs", 3)
    fc_file = STAGE0 / "fixed_costs.json"
    launch = readback = None
    if fc_file.exists():
        for r in load(fc_file)["results"]:
            if (
                r.get("group") == "launch"
                and r.get("name") == "empty_kernel_launch_sync"
            ):
                launch = r["median_us"]
            if (
                r.get("group") == "readback"
                and r.get("name") == "kernel_write_8_bytes_copy_to_host"
                and r.get("variant") == "pinned"
            ):
                readback = r["median_us"]
    print(
        f"Fixed costs on the H200 (fixed_costs.json, medians): kernel launch with synchronisation {launch} us, kernel + 8-byte read-back (pinned) {readback} us."
    )
    print()
    rows = []
    for i in sorted(insts, key=lambda i: i.t_stock)[:8]:
        allow = 0.2 * i.t_stock
        rows.append(
            [
                i.name,
                i.band,
                fs(i.t_stock),
                f"{1e3 * allow:.1f}",
                fn(1e6 * allow / launch) if launch else "-",
                fn(1e6 * allow / readback) if readback else "-",
            ]
        )
    table(
        [
            "instance (8 fastest)",
            "band",
            "stock s",
            "20% of it, ms",
            "= kernel launches",
            "= read-backs",
        ],
        rows,
        right={2, 3, 4, 5},
    )
    ts = sorted(i.t_stock for i in insts)
    print(
        f"Stock times: min {ts[0]:.3f} s, 10th percentile {ts[int(math.ceil(0.1 * len(ts))) - 1]:.3f} s, median {statistics.median(ts):.3f} s, 90th percentile {ts[int(math.ceil(0.9 * len(ts))) - 1]:.3f} s, max {ts[-1]:.2f} s."
    )
    for lim in (0.3, 1.0, 2.0, 10.0):
        print(
            f"- instances with stock time below {lim:g} s: {sum(1 for t in ts if t < lim)} of {len(ts)}"
        )
    print()

    section_5_3(insts)


def section_5_3(insts):
    """Optional supplement: the patched HiGHS with the stages outside the Stage 1 rule set
    switched off (diag_stage1), and with Stage 1 and 2 rules (diag_stage12).
    """
    s1_file = STAGE0 / "diag_stage1" / "summary.json"
    if not s1_file.exists():
        return
    s1_sum = load(s1_file)
    s1 = {
        x["name"]: x
        for x in s1_sum["instances"]
        if x.get("diag") and x.get("presolve_core_seconds") is not None
    }
    s12_file = STAGE0 / "diag_stage12" / "summary.json"
    s12_sum = load(s12_file) if s12_file.exists() else None
    s12 = (
        {
            x["name"]: x
            for x in s12_sum["instances"]
            if x.get("diag") and x.get("presolve_core_seconds") is not None
        }
        if s12_sum
        else {}
    )
    sel_all = [i for i in insts if i.name in s1]
    heading(
        "5.3 Supplement: HiGHS restricted to the Stage 1 rule set (diag_stage1)",
        3,
    )
    print(
        "These directories are not among the inputs named for the Stage 0 time analysis. Single runs; the summaries do not record how many instances "
        "ran at the same time. They are the measured reference for the Stage 1 time gate ('HiGHS on the same rule set'); the stage sums of table 5.1 are not."
    )
    print()
    print(
        f"- diag_stage1: HIGHS_DIAG_OFF={s1_sum.get('env', {}).get('HIGHS_DIAG_OFF', '?')} ({len(s1)} of {len(insts)} instances)"
    )
    if s12_sum:
        print(
            f"- diag_stage12: HIGHS_DIAG_OFF={s12_sum.get('env', {}).get('HIGHS_DIAG_OFF', '?')} ({len(s12)} of {len(insts)} instances)"
        )
    print(
        "- what stays on in diag_stage1: initial scan, fast loop (row and column presolve, which hold some Stage 3 column rules), parallel rows."
    )
    print()
    rows = []
    for band in BANDS + ["all", "all but rd-rplusc-21"]:
        sel = [i for i in sel_all if band.startswith("all") or i.band == band]
        if band == "all but rd-rplusc-21":
            sel = [i for i in sel if i.name != "rd-rplusc-21"]
        if not sel:
            continue
        ts = [s1[i.name]["presolve_core_seconds"] for i in sel]
        grp = [sum(i.sec[st] for st in GROUPS["rows+cols loop"]) for i in sel]
        full = [i.t_diag for i in sel]
        top = max(sel, key=lambda i: s1[i.name]["presolve_core_seconds"])
        r = [
            band,
            str(len(sel)),
            fs(statistics.median(ts)),
            fs(sgm(ts, 0.1)),
            fs(sgm(ts, 1.0)),
            f"{max(ts):.2f} ({top.name})",
            fs(sum(ts), 2),
            fs(sgm(grp, 0.1)),
            fs(sgm(full, 0.1)),
        ]
        if s12:
            t12 = [
                s12[i.name]["presolve_core_seconds"]
                for i in sel
                if i.name in s12
            ]
            r.append(fs(sgm(t12, 0.1)) if len(t12) == len(sel) else "-")
        rows.append(r)
    table(
        [
            "band",
            "n",
            "stage1 median",
            "stage1 SGM(0.1)",
            "stage1 SGM(1)",
            "stage1 max",
            "stage1 sum",
            "'rows+cols loop' stage sum of the full run, SGM(0.1)",
            "full diag SGM(0.1)",
        ]
        + (["stage12 SGM(0.1)"] if s12 else []),
        rows,
        right={1, 2, 3, 4, 6, 7, 8, 9},
    )
    print(
        "Per instance (fast loop and parallel rows: seconds / calls / rows removed):"
    )
    print()
    rows = []
    for i in sel_all:
        a = s1[i.name]
        st = a["diag"]["stage"]
        fl, pr = st["fast_loop"], st["parallel_rows_cols"]
        r = [
            i.name,
            i.band,
            fs(i.t_diag),
            fs(sum(i.sec[s] for s in GROUPS["rows+cols loop"])),
            fs(a["presolve_core_seconds"]),
            f"{1e6 * a['presolve_core_seconds'] / i.nnz:.2f}",
            f"{fs(i.sec['fast_loop'], 2)} / {i.stage['fast_loop']['calls']} / {fn(i.stage['fast_loop']['rows_removed'])}",
            f"{fs(fl['seconds'], 2)} / {fl['calls']} / {fn(fl['rows_removed'])}",
            f"{fs(i.sec['parallel_rows_cols'], 2)} / {i.stage['parallel_rows_cols']['calls']} / {fn(i.stage['parallel_rows_cols']['rows_removed'])}",
            f"{fs(pr['seconds'], 2)} / {pr['calls']} / {fn(pr['rows_removed'])}",
            f"{fn(i.diag_row['rows_after'])} / {fn(a['rows_after'])}",
        ]
        if s12:
            b = s12.get(i.name)
            if b:
                f12 = b["diag"]["stage"]["fast_loop"]
                r += [
                    fs(b["presolve_core_seconds"]),
                    f"{fs(f12['seconds'], 2)} / {f12['calls']} / {fn(f12['rows_removed'])}",
                ]
            else:
                r += ["-", "-"]
        rows.append(r)
    table(
        [
            "instance",
            "band",
            "full diag s",
            "rows+cols stage sum, full run",
            "stage1 s",
            "stage1 us/nnz",
            "fast loop, full run",
            "fast loop, stage1",
            "parallel, full run",
            "parallel, stage1",
            "rows after: full / stage1",
        ]
        + (["stage12 s", "fast loop, stage12"] if s12 else []),
        rows,
        right={2, 3, 4, 5},
    )
    rows = []
    for i in sel_all:
        a = s1[i.name]
        if a["presolve_core_seconds"] < 5.0:
            continue
        tot = a["diag"]["total"][0]["presolve_seconds"]
        for r, c, z, t in progress_lines(
            STAGE0 / "diag_stage1" / f"{i.name}.json.log"
        ):
            rows.append([i.name, f"{t}s", fn(r), fn(c), fn(z), fs(tot, 2)])
    if rows:
        print(
            "Progress lines of the diag_stage1 run, instances with at least 5 s (the last line of each instance repeats the time of the last loop top):"
        )
        print()
        table(
            [
                "instance",
                "line stamp",
                "rows",
                "cols",
                "nnz",
                "presolve s of the run",
            ],
            rows,
        )


# ----------------------------------------------------------------------------- main


def main():
    insts, skipped = read_instances()
    heading(
        "Stage 0: where HiGHS presolve spends its time (dev-size testbed)", 1
    )
    print(
        f"Instances: {len(insts)} (testbed 'main', where == 'dev', with stock, diag and stats results). Skipped: {', '.join(skipped) if skipped else 'none'}."
    )
    logs = sum(1 for i in insts if i.log["found"])
    print(
        f"HiGHS logs of the diag run found for {logs} of {len(insts)} instances."
    )
    print()
    section_1(insts)
    section_2(insts)
    section_3(insts)
    section_4(insts)
    section_5(insts)


if __name__ == "__main__":
    main()
