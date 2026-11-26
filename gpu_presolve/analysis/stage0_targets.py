#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 targets: per-instance targets for the stage gates, and what the HiGHS switch-off runs
say about the stage order and the open decisions of the plan.

Reads, relative to this file:

    ../testbed/testbed.json                         "main" list, instances below 1e7 nonzeros (31)
    ../results/stage0/stock/summary.json            stock HiGHS 1.16.0
    ../results/stage0/diag/summary.json             patched HiGHS, no switch (DIAG lines parsed)
    ../results/stage0/diag_stage1/summary.json      Stage 1 reference (11 switches off)
    ../results/stage0/diag_stage12/summary.json     Stage 1 + 2 reference (8 switches off)
    ../results/stage0/diag_noprobing/summary.json   probing off
    ../results/stage0/diag_nocliquemerge/summary.json  clique merging off
    ../results/stage0/<variant>/<name>.json         row classes of the presolved model (section 4)
    ../results/stage0/stats/<name>.json             structure statistics of the original model
    ../results/stage0/stock_status/summary.json     the four status instances

Prints markdown tables:

    1  target table (sizes and presolve_core_seconds of every variant), checks that every run
       ended normally, and the odd cases: a restricted variant that is smaller or slower than full
       HiGHS, with the DIAG numbers of both runs
    2  how good each restricted variant is as a reference: Stage 3 rules that still run in
       diag_stage1 (counted by postsolve type), what diag_stage12 loses with the enumeration stage,
       the switch sets proposed for the next runs, and what HiGHS probing finds (Stage 2 gate)
    3  paired comparisons per rule family (probing, clique merging, beyond Stage 1, Stage 3 + 4,
       Stage 2 on top of Stage 1)
    4  the clique decision: effect and cost of HiGHS clique merging per instance, with structure
    5  time-target constants: run-to-run noise, shift of the shifted geometric mean, the constant of
       the per-instance rule, and the per-instance limits they give
    6  status instances

Sizes of the patched runs are taken from the "DIAG total" line (the model HiGHS holds at the end of
presolve). The tool's rows_after / cols_after / nnz_after are the model Highs::presolve() returns,
which is the original model when the status is "Not reduced"; the two differ on one run (section 1).

Writes ../results/stage0/targets.csv: the target table, the probing counts of the full and the
diag_stage12 run, and the proposed per-instance time limits.

No arguments. Python 3 standard library only.
"""

import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE0 = HERE.parent / "results" / "stage0"
TESTBED = HERE.parent / "testbed" / "testbed.json"
CSV_OUT = STAGE0 / "targets.csv"

DIAG_VARIANTS = [
    "diag",
    "diag_stage1",
    "diag_stage12",
    "diag_noprobing",
    "diag_nocliquemerge",
]
VARIANTS = ["stock"] + DIAG_VARIANTS
SHORT = {
    "stock": "full (stock)",
    "diag": "full (diag)",
    "diag_stage1": "stage1",
    "diag_stage12": "stage12",
    "diag_noprobing": "noprobing",
    "diag_nocliquemerge": "nocliquemerge",
}
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
BAND_GROUP = {"A1": "A", "A2": "A", "A3": "A", "B1": "B", "B2": "B", "B3": "B"}

# Proposed constants (section 5 prints the numbers behind them).
SHIFT_SECONDS = 0.1
RULE_FACTOR = 1.2
RULE_CONSTANT_SECONDS = 0.05

# Postsolve types that only rules of Stage 3 of the plan produce while the aggregator, sparsify and
# parallel columns are off (the diag_stage1 run). fixed_col is not in the list: it holds every
# column fixing (Stage 1 rules and dominated columns alike) and cannot be split.
STAGE3_COLUMN_TYPES = [
    "free_col_substitution",
    "zero_obj_singleton_col",
    "forcing_column",
]
STAGE3_OTHER_TYPES = [
    "implied_equation",
    "forcing_column_removed_row",
    "linear_transform",
]


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def geomean(values):
    return math.exp(statistics.fmean(math.log(v) for v in values))


def shifted_geomean(values, shift):
    return (
        math.exp(statistics.fmean(math.log(v + shift) for v in values)) - shift
    )


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered) / 100) - 1)]


def table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if a == "l" else "---:" for a in align) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def fmt_s(seconds):
    if seconds is None:
        return "-"
    return f"{seconds:.3f}" if seconds < 100 else f"{seconds:.1f}"


def fmt_ratio(value):
    return f"{value:.4f}"


def pct(value):
    return f"{100 * value:+.2f}%"


class Data:
    def __init__(self):
        testbed = load_json(TESTBED)
        self.testbed = {
            i["name"]: i for i in testbed["main"] if i["nnz"] < 1e7
        }
        self.names = list(self.testbed)
        self.summary = {
            v: load_json(STAGE0 / v / "summary.json") for v in VARIANTS
        }
        self.inst = {
            v: {i["name"]: i for i in self.summary[v]["instances"]}
            for v in VARIANTS
        }
        self.status = load_json(STAGE0 / "stock_status" / "summary.json")
        self.status_testbed = {i["name"]: i for i in testbed["status"]}

    def band(self, name):
        return self.testbed[name]["band"]

    def seconds(self, variant, name):
        return self.inst[variant][name]["presolve_core_seconds"]

    def diag(self, variant, name):
        return self.inst[variant][name]["diag"]

    def total(self, variant, name):
        return self.diag(variant, name)["total"][0]

    def tool_sizes(self, variant, name):
        i = self.inst[variant][name]
        return (i["rows_after"], i["cols_after"], i["nnz_after"])

    def sizes(self, variant, name):
        """(rows, cols, nnz) HiGHS holds at the end of presolve."""
        if variant == "stock":
            return self.tool_sizes(variant, name)
        total = self.total(variant, name)
        return tuple(
            int(total[k].split(":")[1]) for k in ("rows", "cols", "nnz")
        )

    def stage(self, variant, name, stage):
        return self.diag(variant, name)["stage"][stage]

    def reductions(self, variant, name, stage="all"):
        for line in self.diag(variant, name)["reductions"]:
            if line["stage"] == stage:
                return line
        return {}

    def merge_calls(self, variant, name, stage=None):
        calls = self.diag(variant, name).get("cliquemerge_call", [])
        return [c for c in calls if stage is None or c["stage"] == stage]

    def probing(self, variant, name):
        return self.diag(variant, name)["probing"][0]

    def full_seconds(self, name):
        """Best estimate of the full HiGHS time: the smaller of the two measurements."""
        return min(self.seconds("stock", name), self.seconds("diag", name))


# --------------------------------------------------------------------------------------------
# 5 (computed first: the noise bound is used to flag slow variants in section 1)
# --------------------------------------------------------------------------------------------


def noise_pairs(data):
    pairs = []
    for name in data.names:
        a = data.seconds("stock", name)
        b = data.seconds("diag", name)
        pairs.append((name, a, b, b - a, (b - a) / ((a + b) / 2)))
    return pairs


def noise_bound(data):
    """(absolute seconds below 1 s, relative above): the largest stock-diag difference seen."""
    pairs = noise_pairs(data)
    absolute = max(abs(p[3]) for p in pairs if min(p[1], p[2]) < 1.0)
    relative = max(abs(p[4]) for p in pairs)
    return absolute, relative


# --------------------------------------------------------------------------------------------
# 1 target table, checks, odd cases
# --------------------------------------------------------------------------------------------


def section_targets(data):
    print("## 1. Target table\n")
    print(
        "Sizes are rows / columns / nonzeros after presolve; t is presolve_core_seconds. Full HiGHS"
        " gives the same sizes with the stock and the patched binary (checked below), so they are"
        " printed once.\n"
    )
    rows = []
    for name in data.names:
        t = data.testbed[name]
        full = data.sizes("diag", name)
        rows.append(
            [
                name,
                t["band"],
                t["rows"],
                t["cols"],
                t["nnz"],
                *full,
                f"{100 * (1 - full[2] / t['nnz']):.1f}%",
                fmt_s(data.seconds("stock", name)),
                fmt_s(data.seconds("diag", name)),
            ]
        )
    table(
        [
            "instance",
            "band",
            "rows",
            "cols",
            "nnz",
            "full rows",
            "full cols",
            "full nnz",
            "nnz removed",
            "t stock",
            "t diag",
        ],
        rows,
        ["l", "l"] + ["r"] * 9,
    )
    for pair in (
        ("diag_stage1", "diag_stage12"),
        ("diag_noprobing", "diag_nocliquemerge"),
    ):
        header = ["instance"]
        for v in pair:
            header += [f"{SHORT[v]} rows", "cols", "nnz", "t"]
        rows = []
        for name in data.names:
            row = [name]
            for v in pair:
                row += [*data.sizes(v, name), fmt_s(data.seconds(v, name))]
            rows.append(row)
        table(header, rows)

    print("### 1.1 Did every run end normally?\n")
    problems = []
    max_clock = (0.0, 0.0)
    not_reduced = []
    for v in VARIANTS:
        env = data.summary[v]["env"].get("HIGHS_DIAG_OFF", "none")
        if set(data.inst[v]) != set(data.names):
            problems.append(f"{v}: instance list differs from the testbed")
        for name in data.names:
            i = data.inst[v][name]
            t = data.testbed[name]
            if i["status"] not in ("Reduced", "Not reduced"):
                problems.append(f"{v} {name}: status {i['status']}")
            if i["status"] == "Not reduced":
                not_reduced.append((v, name))
            if i["reached_time_limit"]:
                problems.append(f"{v} {name}: reached the time limit")
            if (i["rows"], i["cols"], i["nnz"]) != (
                t["rows"],
                t["cols"],
                t["nnz"],
            ):
                problems.append(
                    f"{v} {name}: original size differs from testbed.json"
                )
            if v == "stock":
                continue
            diag = i["diag"]
            if len(diag["total"]) != 1:
                problems.append(
                    f"{v} {name}: {len(diag['total'])} DIAG total lines"
                )
            total = diag["total"][0]
            if total["stopped"] != 0 or total["restart"] != 0:
                problems.append(
                    f"{v} {name}: stopped={total['stopped']} restart={total['restart']}"
                )
            if set(diag["switches"][0]["off"].split(",")) != set(
                env.split(",")
            ):
                problems.append(
                    f"{v} {name}: DIAG switches differ from the run's environment"
                )
            if diag.get("_replaced"):
                problems.append(
                    f"{v} {name}: repeated DIAG lines {diag['_replaced']}"
                )
            if data.sizes(v, name) != data.tool_sizes(v, name):
                problems.append(
                    f"{v} {name}: status {i['status']}; the tool reports "
                    f"{data.tool_sizes(v, name)} but DIAG total has {data.sizes(v, name)}"
                )
            diff = abs(total["presolve_seconds"] - i["presolve_core_seconds"])
            max_clock = max(
                max_clock, (diff, diff / i["presolve_core_seconds"])
            )
    for name in data.names:
        if data.tool_sizes("stock", name) != data.tool_sizes("diag", name):
            problems.append(
                f"{name}: stock and patched HiGHS give different sizes"
            )
    print(
        f"- {len(VARIANTS)} variants x {len(data.names)} instances = "
        f"{len(VARIANTS) * len(data.names)} runs read."
    )
    print(
        "- Checked per run: status Reduced or Not reduced; reached_time_limit false; original"
        " size equal to testbed.json; for the patched runs one DIAG total line with stopped=0"
        " and restart=0, DIAG switches equal to the run's HIGHS_DIAG_OFF, no repeated DIAG"
        " line, DIAG total sizes equal to the tool's sizes; stock sizes equal to the patched"
        " sizes without switch."
    )
    print(
        f"- Largest difference between DIAG total presolve_seconds and presolve_core_seconds of"
        f" the same run: {max_clock[0]:.4f} s ({100 * max_clock[1]:.2f}%)."
    )
    print(
        "- Runs with status Not reduced: "
        + ", ".join(f"{name} ({SHORT[v]})" for v, name in not_reduced)
        + "."
    )
    print(f"- Checks that failed: {len(problems)}.")
    for p in problems:
        print(f"  - {p}")
    print()

    abs_noise, rel_noise = noise_bound(data)
    print("### 1.2 Restricted variant with a smaller result than full HiGHS\n")
    print(
        "Difference = restricted minus full (negative = the restricted run ends smaller)."
        " 'ext' = clique-extension nonzeros added by clique merging, 'vb' = nonzeros added by"
        " variable-bound aggregation, 'fill' = nonzeros added by the aggregator (each summed"
        " over the stage calls that added nonzeros), all in the run named.\n"
    )
    rows = []
    for v in DIAG_VARIANTS[1:]:
        for name in data.names:
            full = data.sizes("diag", name)
            mine = data.sizes(v, name)
            delta = [m - f for m, f in zip(mine, full)]
            if min(delta) >= 0:
                continue
            rows.append(
                [
                    name,
                    SHORT[v],
                    *delta,
                    added_nonzeros(data, "diag", name),
                    added_nonzeros(data, v, name),
                    probing_text(data, "diag", name),
                    probing_text(data, v, name),
                ]
            )
    table(
        [
            "instance",
            "variant",
            "d rows",
            "d cols",
            "d nnz",
            "full: ext/vb/fill",
            "variant: ext/vb/fill",
            "full probing",
            "variant probing",
        ],
        rows,
        ["l", "l", "r", "r", "r", "r", "r", "l", "l"],
    )
    print(
        "Probing text: rounds / probes / fixed / substitutions / bounds tightened / stop of the"
        " last round.\n"
    )
    only_nnz = sum(1 for r in rows if r[2] >= 0 and r[3] >= 0)
    all_three = sum(1 for r in rows if r[2] < 0 and r[3] < 0 and r[4] < 0)
    print(
        f"- {len(rows)} (instance, variant) pairs; fewer nonzeros only: {only_nnz}; fewer rows,"
        f" columns and nonzeros: {all_three}; other (fewer rows or columns, not all three):"
        f" {len(rows) - only_nnz - all_three}.\n"
    )
    print(
        "Stage with the largest change of removed nonzeros, per case (full -> variant):\n"
    )
    rows = []
    for v in DIAG_VARIANTS[1:]:
        for name in data.names:
            full = data.sizes("diag", name)
            mine = data.sizes(v, name)
            if min(m - f for m, f in zip(mine, full)) >= 0:
                continue
            rows.append(
                [
                    name,
                    SHORT[v],
                    stage_changes(data, v, name, "nnz_removed", 3),
                ]
            )
    table(
        ["instance", "variant", "nnz removed per stage, full -> variant"],
        rows,
        ["l", "l", "l"],
    )

    print("### 1.3 Restricted variant slower than full HiGHS\n")
    print(
        f"Listed when t(variant) - t(full, patched) exceeds the largest stock-diag difference"
        f" seen for the same work: {abs_noise:.3f} s below 1 s, {100 * rel_noise:.1f}% above"
        " (section 5).\n"
    )
    rows = []
    for v in DIAG_VARIANTS[1:]:
        for name in data.names:
            full = data.seconds("diag", name)
            mine = data.seconds(v, name)
            if mine - full <= (abs_noise if full < 1.0 else rel_noise * full):
                continue
            rows.append(
                [
                    name,
                    SHORT[v],
                    fmt_s(full),
                    fmt_s(mine),
                    f"{mine / full:.2f}",
                    stage_changes(data, v, name, "seconds", 2),
                    probing_text(data, "diag", name),
                    probing_text(data, v, name),
                ]
            )
    table(
        [
            "instance",
            "variant",
            "t full",
            "t variant",
            "ratio",
            "seconds per stage, full -> variant (calls)",
            "full probing",
            "variant probing",
        ],
        rows,
        ["l", "l", "r", "r", "r", "l", "l", "l"],
    )


def added_nonzeros(data, variant, name):
    diag = data.diag(variant, name)
    ext = diag["cliquemerge"][0]["nnz_added"]
    vb = sum(
        -s["nnz_removed"]
        for s in diag["varbound_aggregation"].values()
        if s["nnz_removed"] < 0
    )
    fill = -data.stage(variant, name, "aggregator")["nnz_removed"]
    return f"{ext}/{vb}/{max(fill, 0)}"


def probing_text(data, variant, name):
    p = data.probing(variant, name)
    if p["rounds"] == 0:
        return "not run"
    return (
        f"{p['rounds']} / {p['probed']} / {p['fixed']} / {p['substitutions']} / "
        f"{p['bounds_tightened']} / {p['stop']}"
    )


def stage_changes(data, variant, name, field, count):
    changes = []
    for stage in STAGES:
        a = data.stage("diag", name, stage)
        b = data.stage(variant, name, stage)
        changes.append((abs(b[field] - a[field]), stage, a, b))
    changes.sort(key=lambda c: -c[0])
    parts = []
    for size, stage, a, b in changes[:count]:
        if size == 0:
            continue
        if field == "seconds":
            parts.append(
                f"{stage} {a[field]:.2f} ({a['calls']}) -> {b[field]:.2f} ({b['calls']})"
            )
        else:
            parts.append(f"{stage} {a[field]} -> {b[field]}")
    return "; ".join(parts)


# --------------------------------------------------------------------------------------------
# 2 quality of the references
# --------------------------------------------------------------------------------------------


def stage3_in_stage1(data, name):
    """Counts of Stage 3 rule applications in the diag_stage1 run."""
    red = data.reductions("diag_stage1", name)
    subs = data.diag("diag_stage1", name)["substitutions"][0]
    counts = {
        t: red.get(t, 0) for t in STAGE3_COLUMN_TYPES + STAGE3_OTHER_TYPES
    }
    counts["simple_probing"] = subs["simple_probing"]
    counts["dual_fixing"] = subs["dual_fixing"]
    counts["columns"] = (
        counts["simple_probing"]
        + counts["dual_fixing"]
        + sum(counts[t] for t in STAGE3_COLUMN_TYPES)
    )
    counts["implied_integer_cols"] = data.total("diag_stage1", name)[
        "implied_integer_cols"
    ]
    counts["fixed_col"] = red.get("fixed_col", 0)
    counts["doubleton"] = data.diag("diag_stage1", name)["doubleton"][0][
        "removed"
    ]
    return counts


def section_references(data):
    print("## 2. How good is each restricted variant as a reference?\n")
    print("### 2.1 diag_stage1: Stage 3 rules that still run\n")
    ran = set()
    for name in data.names:
        for stage in STAGES:
            if data.stage("diag_stage1", name, stage)["calls"] > 0:
                ran.add(stage)
    print(
        "Stages with at least one call in the diag_stage1 runs: "
        + ", ".join(s for s in STAGES if s in ran)
        + ".\n"
    )
    print(
        "Column removals by Stage 3 rules in the diag_stage1 run, counted from the postsolve"
        " types and the DIAG substitutions line. 'S3 cols' = simple probing + dual-fixing"
        " substitution + free column substitution + zero-cost singleton + forcing column (one"
        " column each). 'share' = S3 cols / columns removed by the run. fixed_col also holds"
        " the fixings of dominated and weakly dominated columns and cannot be split, so the"
        " share is a lower bound. 'dbl eq' = doubleton equations (a Stage 1 rule), for scale.\n"
    )
    rows = []
    affected = []
    for name in data.names:
        c = stage3_in_stage1(data, name)
        removed = (
            data.testbed[name]["cols"] - data.sizes("diag_stage1", name)[1]
        )
        share = c["columns"] / removed if removed else 0.0
        other = (
            c["implied_equation"]
            + c["forcing_column_removed_row"]
            + c["linear_transform"]
        )
        if c["columns"] or other or c["implied_integer_cols"]:
            affected.append(
                (
                    name,
                    c["columns"],
                    share,
                    c["columns"] / data.testbed[name]["cols"],
                )
            )
        rows.append(
            [
                name,
                removed,
                c["simple_probing"],
                c["dual_fixing"],
                c["free_col_substitution"],
                c["zero_obj_singleton_col"],
                c["forcing_column"],
                c["columns"],
                f"{100 * share:.1f}%",
                c["implied_equation"],
                c["forcing_column_removed_row"],
                c["linear_transform"],
                c["implied_integer_cols"],
                c["fixed_col"],
                c["doubleton"],
            ]
        )
    table(
        [
            "instance",
            "cols removed",
            "simple probing",
            "dual-fix subst",
            "free col subst",
            "zero-cost singleton",
            "forcing col",
            "S3 cols",
            "share",
            "implied eq",
            "forcing-col rows",
            "linear transform",
            "implied int cols",
            "fixed_col (unsplit)",
            "dbl eq",
        ],
        rows,
    )
    with_cols = [a for a in affected if a[1] > 0]
    print(
        f"- Instances with at least one Stage 3 application of any listed kind: {len(affected)}"
        f" of {len(data.names)}; with at least one S3 column removal: {len(with_cols)}."
    )
    for limit in (0.01, 0.10, 0.50):
        hit = [a for a in with_cols if a[2] > limit]
        print(
            f"- S3 cols above {100 * limit:.0f}% of the columns the run removes: {len(hit)}"
            + (
                " ("
                + ", ".join(f"{a[0]} {100 * a[2]:.0f}%" for a in hit)
                + ")"
                if hit
                else ""
            )
        )
    hit = [a for a in with_cols if a[3] > 0.01]
    print(
        f"- S3 cols above 1% of the original columns: {len(hit)} ("
        + ", ".join(f"{a[0]} {100 * a[3]:.1f}%" for a in hit)
        + ")"
    )
    for key, label in [
        ("simple_probing", "simple probing on equations"),
        ("dual_fixing", "dual-fixing substitution"),
        (
            "free_col_substitution",
            "free column substitution (singleton columns)",
        ),
        ("zero_obj_singleton_col", "zero-cost singleton"),
        ("forcing_column", "forcing column"),
    ]:
        per = sorted(
            ((stage3_in_stage1(data, n)[key], n) for n in data.names),
            reverse=True,
        )
        per = [p for p in per if p[0] > 0]
        print(
            f"- {label}: {sum(p[0] for p in per)} columns on {len(per)} instances ("
            + ", ".join(f"{n} {c}" for c, n in per[:4])
            + (", ..." if len(per) > 4 else "")
            + ")"
        )
    implied = [
        (data.total("diag_stage1", n)["implied_integer_cols"], n)
        for n in data.names
    ]
    implied = sorted((p for p in implied if p[0] > 0), reverse=True)
    print(
        f"- implied-integer columns at the end of the run (conversion in column presolve):"
        f" {len(implied)} instances ("
        + ", ".join(f"{n} {c}" for c, n in implied)
        + ")"
    )
    budget = 0.99 ** len(data.names)
    print(
        f"- Room in a 99% geometric mean over {len(data.names)} instances: one instance at"
        f" {100 * budget:.1f}% with the others at 100%, or two at"
        f" {100 * math.sqrt(budget):.1f}%."
    )
    print()

    print(
        "Rules the plan leaves out of the first versions, in the full run (they are part of"
        " every reference that keeps their stage):\n"
    )
    rows = []
    for name in data.names:
        fm = data.stage("diag", name, "fourier_motzkin")
        vb = data.diag("diag", name)["varbound_aggregation"]
        vb_rows = sum(s["rows_removed"] for s in vb.values())
        vb_mod = sum(s["rows_modified"] for s in vb.values())
        vb_nnz = sum(s["nnz_removed"] for s in vb.values())
        if not (fm["rows_removed"] or fm["cols_removed"] or vb_rows or vb_mod):
            continue
        rows.append(
            [
                name,
                f"{fm['rows_removed']} / {fm['cols_removed']} / {fm['nnz_removed']}",
                vb_rows,
                vb_mod,
                vb_nnz,
            ]
        )
    table(
        [
            "instance",
            "Fourier-Motzkin rows / cols / nnz removed",
            "VLB/VUB aggregation rows removed",
            "rows rewritten",
            "nnz removed",
        ],
        rows,
    )

    print(
        "### 2.2 diag_stage12: what goes missing with the enumeration stage\n"
    )
    print(
        "Full run, clique merging by the stage that called it (sums of the cliquemerge_call"
        " lines), variable-bound aggregation by stage, and what the enumeration stage removed in"
        " total. Only instances where one of these numbers is nonzero.\n"
    )
    rows = []
    tot = {"enumeration": [0, 0, 0, 0], "probing": [0, 0, 0, 0]}
    for name in data.names:
        diag = data.diag("diag", name)
        cells = []
        nonzero = False
        for stage in ("enumeration", "probing"):
            calls = data.merge_calls("diag", name, stage)
            changed = sum(c["rows_changed"] for c in calls)
            deleted = sum(c["rows_deleted"] for c in calls)
            added = sum(c["nnz_added"] for c in calls)
            fix = sum(c["fixings"] for c in calls)
            before = sum(c["rows_deleted_before_merge"] for c in calls)
            tot[stage][0] += changed
            tot[stage][1] += deleted
            tot[stage][2] += added
            tot[stage][3] += fix
            nonzero = nonzero or changed or deleted or fix or before
            cells += [len(calls), changed, deleted, added, fix, before]
        vb = diag["varbound_aggregation"]
        enum = data.stage("diag", name, "enumeration")
        red = data.reductions("diag", name, "enumeration")
        subs = diag["substitutions"][0]["in_enumeration"]
        nonzero = (
            nonzero
            or enum["rows_removed"]
            or enum["cols_removed"]
            or enum["nnz_removed"]
            or any(
                s["rows_removed"] or s["rows_modified"] for s in vb.values()
            )
        )
        if not nonzero:
            continue
        rows.append(
            [
                name,
                *cells,
                f"{vb['enumeration']['rows_removed']}/{vb['enumeration']['rows_modified']}",
                f"{vb['probing']['rows_removed']}/{vb['probing']['rows_modified']}",
                f"{enum['rows_removed']}/{enum['cols_removed']}/{enum['nnz_removed']}",
                red.get("fixed_col", 0),
                subs,
                fmt_s(enum["seconds"]),
            ]
        )
    table(
        [
            "instance",
            "enum calls",
            "rows ext",
            "rows del",
            "nnz ext",
            "fix",
            "del before",
            "prob calls",
            "rows ext",
            "rows del",
            "nnz ext",
            "fix",
            "del before",
            "vb enum del/mod",
            "vb prob del/mod",
            "enum stage rows/cols/nnz removed",
            "enum fixed_col",
            "enum subst",
            "enum s",
        ],
        rows,
    )
    for stage in ("enumeration", "probing"):
        t = tot[stage]
        print(
            f"- Clique merging called from the {stage} stage, all instances: rows extended"
            f" {t[0]}, rows deleted {t[1]}, extension nonzeros {t[2]}, fixings {t[3]}."
        )
    both = [tot["enumeration"][k] + tot["probing"][k] for k in range(4)]
    print(
        f"- Share of the enumeration-stage calls: rows extended"
        f" {100 * tot['enumeration'][0] / both[0]:.1f}%, rows deleted"
        f" {100 * tot['enumeration'][1] / both[1]:.1f}%."
    )
    wo = [
        tot["enumeration"][1]
        - sum(
            c["rows_deleted"]
            for c in data.merge_calls("diag", "sorrell3", "enumeration")
        ),
        tot["probing"][1],
    ]
    print(
        f"- Without sorrell3: rows deleted in enumeration-stage calls {wo[0]}, in probing-stage"
        f" calls {wo[1]}."
    )
    print()

    print(
        "Clique merging in the diag_stage12 run against the full run (totals of the"
        " cliquemerge line), and the probing totals of both. Only instances where merging acts"
        " in one of the two runs.\n"
    )
    rows = []
    for name in data.names:
        a = data.diag("diag", name)["cliquemerge"][0]
        b = data.diag("diag_stage12", name)["cliquemerge"][0]
        if not any(
            x[k]
            for x in (a, b)
            for k in ("rows_changed", "rows_deleted", "fixings")
        ):
            continue
        rows.append(
            [
                name,
                a["calls"],
                a["rows_changed"],
                a["rows_deleted"],
                a["nnz_added"],
                b["calls"],
                b["rows_changed"],
                b["rows_deleted"],
                b["nnz_added"],
                probing_text(data, "diag", name),
                probing_text(data, "diag_stage12", name),
            ]
        )
    table(
        [
            "instance",
            "full calls",
            "rows ext",
            "rows del",
            "nnz ext",
            "stage12 calls",
            "rows ext",
            "rows del",
            "nnz ext",
            "full probing",
            "stage12 probing",
        ],
        rows,
        ["l"] + ["r"] * 8 + ["l", "l"],
    )
    for key, label in [
        ("rows_changed", "rows extended"),
        ("rows_deleted", "rows deleted"),
        ("nnz_added", "extension nonzeros"),
        ("calls", "calls"),
    ]:
        a = sum(
            data.diag("diag", n)["cliquemerge"][0][key] for n in data.names
        )
        b = sum(
            data.diag("diag_stage12", n)["cliquemerge"][0][key]
            for n in data.names
        )
        big = ("sorrell3", "savsched1")
        a2 = a - sum(data.diag("diag", n)["cliquemerge"][0][key] for n in big)
        b2 = b - sum(
            data.diag("diag_stage12", n)["cliquemerge"][0][key] for n in big
        )
        print(
            f"- Clique merging, {label}, all instances: full {a}, diag_stage12 {b}; without"
            f" sorrell3 and savsched1: full {a2}, diag_stage12 {b2}."
        )
    lost = [
        name
        for name in data.names
        if data.diag("diag", name)["cliquemerge"][0]["calls"] > 0
        and data.diag("diag_stage12", name)["cliquemerge"][0]["calls"] == 0
    ]
    print(
        "- Instances where the full run calls clique merging and diag_stage12 never does: "
        + (
            ", ".join(
                f"{n} (probing stop: {data.probing('diag_stage12', n)['stop']})"
                for n in lost
            )
            if lost
            else "none"
        )
        + "."
    )
    fixed = sum(
        data.reductions("diag", n, "enumeration").get("fixed_col", 0)
        for n in data.names
    )
    subst = sum(
        data.diag("diag", n)["substitutions"][0]["in_enumeration"]
        for n in data.names
    )
    print(
        f"- Columns fixed during the enumeration stage of the full run (enumeration itself,"
        f" propagation in prepareProbing, clique table; not split): {fixed}; substitutions"
        f" applied at its end: {subst}."
    )
    name = "sorrell3"
    print(
        f"- {name}, diag_stage12: tool sizes {data.tool_sizes('diag_stage12', name)} with status"
        f" {data.inst['diag_stage12'][name]['status']}; DIAG total sizes"
        f" {data.sizes('diag_stage12', name)}; postsolve entries of the run:"
        f" {sum(v for k, v in data.reductions('diag_stage12', name).items() if k != 'stage')}."
        f" Full run: {data.sizes('diag', name)}; diag_noprobing:"
        f" {data.sizes('diag_noprobing', name)}."
    )
    print()

    print("### 2.3 Switch sets proposed for the next runs\n")
    print("```")
    for label, value in PROPOSED_SWITCHES:
        print(f"{label}\n    HIGHS_DIAG_OFF={value}")
    print("```\n")
    mask = sum(1 << bit for bit, _ in STOCK_RULE_BITS)
    print(
        "Stock option presolve_rule_off, for a Stage 1 reference without the Stage 3 rules of"
        " column presolve that have a stock switch (to be combined with S1; the baseline tool"
        " cannot pass a HiGHS option today):\n"
    )
    print("```")
    for bit, label in STOCK_RULE_BITS:
        print(f"bit {bit:2d} = {1 << bit:8d}  {label}")
    print(f"presolve_rule_off = {mask}")
    print(
        f"presolve_rule_off = {mask + (1 << STOCK_DUAL_FIXING_BIT)}  also bit"
        f" {STOCK_DUAL_FIXING_BIT} (dual fixing: the lock rule of Stage 1 and its Stage 3"
        " extensions together)"
    )
    print(
        f"bit {STOCK_DUAL_FIX_PROBING_BIT} = {1 << STOCK_DUAL_FIX_PROBING_BIT}  dual-fix"
        " probing inside the probing loop (left out by the plan), for the S2 and S3 sets"
    )
    print("```\n")

    print("### 2.4 Stage 2 gate: what HiGHS probing finds, per instance\n")
    print(
        "Counts of the DIAG probing line (sums over the rounds; implications = stored"
        " implications after the last round) and the seconds of the probing stage, in the full"
        " run and in the diag_stage12 run (probing on the Stage 1 model, set S2b).\n"
    )
    rows = []
    for name in data.names:
        row = [name, data.band(name)]
        for v in ("diag", "diag_stage12"):
            p = data.probing(v, name)
            row += [
                p["rounds"],
                p["probed"],
                p["fixed"],
                p["substitutions"],
                p["implications"],
                p["bounds_tightened"],
                p["stop"],
                fmt_s(data.stage(v, name, "probing")["seconds"]),
            ]
        rows.append(row)
    table(
        [
            "instance",
            "band",
            "full rounds",
            "probes",
            "fixed",
            "subst",
            "implications",
            "bounds",
            "stop",
            "probing s",
            "stage12 rounds",
            "probes",
            "fixed",
            "subst",
            "implications",
            "bounds",
            "stop",
            "probing s",
        ],
        rows,
        ["l", "l"] + (["r"] * 6 + ["l", "r"]) * 2,
    )
    for v in ("diag", "diag_stage12"):
        tot = [
            sum(data.probing(v, n)[k] for n in data.names)
            for k in ("probed", "fixed", "substitutions", "bounds_tightened")
        ]
        seconds = sum(
            data.stage(v, n, "probing")["seconds"] for n in data.names
        )
        print(
            f"- {SHORT[v]}: {tot[0]} probes, {tot[1]} fixed, {tot[2]} substitutions, {tot[3]}"
            f" bounds tightened, {seconds:.1f} s in the probing stage over the"
            f" {len(data.names)} instances."
        )
    print()
    print(
        "'probing s' is the whole probing stage. Its parts (sums of the probing_round lines):"
        " prepare = clique extraction and propagation, loop = the probes, finalise = clique"
        " merging, fixings, substitutions, variable-bound aggregation; 'merge' ="
        " runCliqueMerging inside finalise. Instances where the loop is less than half of a"
        " stage of 0.5 s or more are listed.\n"
    )
    rows = []
    for v in ("diag", "diag_stage12"):
        parts = {}
        small_loop = []
        for name in data.names:
            rounds = data.diag(v, name).get("probing_round", [])
            p = {
                k: sum(r[k] for r in rounds)
                for k in (
                    "prepare_seconds",
                    "loop_seconds",
                    "finalise_seconds",
                )
            }
            p["merge"] = sum(
                c["seconds"] for c in data.merge_calls(v, name, "probing")
            )
            p["stage"] = data.stage(v, name, "probing")["seconds"]
            for k, value in p.items():
                parts[k] = parts.get(k, 0.0) + value
            if p["stage"] >= 0.5 and p["loop_seconds"] < 0.5 * p["stage"]:
                small_loop.append(
                    f"{name} {p['loop_seconds']:.2f} of {p['stage']:.2f}"
                )
        rows.append(
            [
                SHORT[v],
                f"{parts['stage']:.1f}",
                f"{parts['prepare_seconds']:.1f}",
                f"{parts['loop_seconds']:.1f}",
                f"{parts['finalise_seconds']:.1f}",
                f"{parts['merge']:.1f}",
                "; ".join(small_loop),
            ]
        )
    table(
        [
            "run",
            "stage s",
            "prepare s",
            "loop s",
            "finalise s",
            "merge s",
            "loop s of stage s where the loop is less than half",
        ],
        rows,
        ["l"] + ["r"] * 5 + ["l"],
    )


PROPOSED_SWITCHES = [
    (
        "S1   Stage 1 gate (the existing diag_stage1; no HIGHS_DIAG_OFF name is left to add)",
        "probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,"
        "sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns,parallel_columns",
    ),
    (
        "S2a  Stage 2 gate, with the enumeration stage (clique extraction and merging in every pass)",
        "implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,"
        "dominated_columns,parallel_columns",
    ),
    (
        "S2b  Stage 2 gate, without the enumeration stage (the existing diag_stage12)",
        "enumeration,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,"
        "dominated_columns,parallel_columns",
    ),
    (
        "S2c  Stage 1 + probing, no clique merging (probing part of the Stage 2 gate)",
        "enumeration,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,"
        "strengthen_inequalities,dominated_columns,parallel_columns",
    ),
    (
        "S2d  Stage 1 + clique merging, no probing (clique part of the Stage 2 gate)",
        "probing,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,"
        "dominated_columns,parallel_columns",
    ),
    (
        "S3a  Stage 3 gate, with the enumeration stage (Stage 4 rules and Fourier-Motzkin off)",
        "sparsify,strengthen_inequalities,fourier_motzkin",
    ),
    (
        "S3b  Stage 3 gate, without the enumeration stage",
        "sparsify,strengthen_inequalities,fourier_motzkin,enumeration",
    ),
    (
        "E    Full HiGHS without the enumeration stage (net effect of that stage alone)",
        "enumeration",
    ),
]

# Stock HiGHS option presolve_rule_off (bit = 1 << rule number of HConst.h). The baseline tool has
# no argument that sets it, so these need a tool change before they can be run.
STOCK_RULE_BITS = [
    (
        5,
        "dominated column (dominated, weakly dominated and forcing columns in column presolve)",
    ),
    (7, "forcing column"),
    (8, "free column substitution (singleton columns)"),
    (18, "zero-cost singleton"),
    (19, "column stuffing"),
    (23, "weakly dominated column"),
]
STOCK_DUAL_FIXING_BIT = 17
STOCK_DUAL_FIX_PROBING_BIT = 20


# --------------------------------------------------------------------------------------------
# 3 paired comparisons
# --------------------------------------------------------------------------------------------

FAMILIES = [
    ("probing", "diag", "diag_noprobing", "full against probing off"),
    (
        "clique merging",
        "diag",
        "diag_nocliquemerge",
        "full against clique merging off",
    ),
    (
        "everything beyond Stage 1",
        "diag",
        "diag_stage1",
        "full against the Stage 1 reference",
    ),
    (
        "Stage 3 + 4 rules",
        "diag",
        "diag_stage12",
        "full against the Stage 1 + 2 reference",
    ),
    (
        "Stage 2 rules on top of Stage 1",
        "diag_stage12",
        "diag_stage1",
        "the Stage 1 + 2 reference against the Stage 1 reference",
    ),
]


def section_families(data):
    print(
        "## 3. Net effect of each rule family (paired over the 31 instances)\n"
    )
    print(
        "Ratio = remaining size with the family / remaining size without it, so 0.90 means the"
        " family leaves 10% less. 'With' is full HiGHS in the first four rows and the Stage 1 +"
        " 2 reference in the last. Times are presolve_core_seconds of the patched binary in"
        " both runs.\n"
    )
    summary = []
    carriers = {}
    for label, with_variant, variant, _ in FAMILIES:
        ratios = {}
        for name in data.names:
            full = data.sizes(with_variant, name)
            other = data.sizes(variant, name)
            ratios[name] = tuple(f / o for f, o in zip(full, other))
        gm = [geomean([ratios[n][k] for n in data.names]) for k in range(3)]
        over1 = [
            sum(1 for n in data.names if abs(ratios[n][k] - 1) > 0.01)
            for k in range(3)
        ]
        over10 = [
            sum(1 for n in data.names if abs(ratios[n][k] - 1) > 0.10)
            for k in range(3)
        ]
        any1 = sum(
            1 for n in data.names if any(abs(r - 1) > 0.01 for r in ratios[n])
        )
        any10 = sum(
            1 for n in data.names if any(abs(r - 1) > 0.10 for r in ratios[n])
        )
        larger = sum(1 for n in data.names if ratios[n][2] > 1.0)
        t_full = [data.seconds(with_variant, n) for n in data.names]
        t_other = [data.seconds(variant, n) for n in data.names]
        summary.append(
            [
                label,
                fmt_ratio(gm[2]),
                fmt_ratio(gm[0]),
                fmt_ratio(gm[1]),
                f"{over1[2]} / {over1[0]} / {over1[1]}",
                f"{over10[2]} / {over10[0]} / {over10[1]}",
                f"{any1} / {any10}",
                larger,
                f"{sum(t_full):.1f}",
                f"{sum(t_other):.1f}",
                f"{sum(t_full) - sum(t_other):+.1f}",
                " / ".join(
                    f"{shifted_geomean(t, 0.1):.3f}" for t in (t_full, t_other)
                ),
                " / ".join(
                    f"{shifted_geomean(t, 1.0):.3f}" for t in (t_full, t_other)
                ),
            ]
        )
        order = sorted(
            (
                n
                for n in data.names
                if any(abs(r - 1) > 0.01 for r in ratios[n])
            ),
            key=lambda n: ratios[n][2],
        )
        carriers[label] = [(n, ratios[n]) for n in order]
    table(
        [
            "family",
            "gm nnz",
            "gm rows",
            "gm cols",
            ">1% nnz / rows / cols",
            ">10% nnz / rows / cols",
            "any size >1% / >10%",
            "nnz larger with",
            "sum t with",
            "sum t without",
            "family s",
            "SGM 0.1 with / without",
            "SGM 1 with / without",
        ],
        summary,
    )
    print(
        "'>1%' and '>10%' count instances whose ratio differs from 1 by more than that, in"
        " either direction. 'nnz larger with' = instances that end with more nonzeros with the"
        " family than without it. 'family s' is a difference between two runs made at different"
        " times: section 5.2 (c) gives the speed of each run relative to the full run on"
        " identical work.\n"
    )
    print(
        "How concentrated each effect is: the geometric means without sorrell3, the median of"
        " the nonzero ratio over the 31 instances, and the number of instances that end with at"
        " least 1% and at least 10% fewer nonzeros with the family.\n"
    )
    rest = [n for n in data.names if n != "sorrell3"]
    rows = []
    for label, with_variant, variant, _ in FAMILIES:
        ratio = {
            n: [
                a / b
                for a, b in zip(
                    data.sizes(with_variant, n), data.sizes(variant, n)
                )
            ]
            for n in data.names
        }
        rows.append(
            [
                label,
                fmt_ratio(geomean([ratio[n][2] for n in rest])),
                fmt_ratio(geomean([ratio[n][0] for n in rest])),
                fmt_ratio(statistics.median(ratio[n][2] for n in data.names)),
                sum(1 for n in data.names if ratio[n][2] < 0.99),
                sum(1 for n in data.names if ratio[n][2] < 0.90),
            ]
        )
    table(
        [
            "family",
            "gm nnz without sorrell3",
            "gm rows without sorrell3",
            "median nnz ratio",
            "nnz at least 1% fewer",
            "nnz at least 10% fewer",
        ],
        rows,
    )
    for label, with_variant, variant, text in FAMILIES:
        print(f"### {label}: {text}\n")
        rows = []
        for name, r in carriers[label]:
            full = data.seconds(with_variant, name)
            other = data.seconds(variant, name)
            rows.append(
                [
                    name,
                    data.band(name),
                    fmt_ratio(r[2]),
                    fmt_ratio(r[0]),
                    fmt_ratio(r[1]),
                    fmt_s(full),
                    fmt_s(other),
                    f"{full - other:+.2f}",
                ]
            )
        table(
            [
                "instance (any size changed by more than 1%)",
                "band",
                "nnz ratio",
                "rows ratio",
                "cols ratio",
                "t with",
                "t without",
                "family s",
            ],
            rows,
            ["l", "l"] + ["r"] * 6,
        )
        costs = sorted(
            (
                (data.seconds(with_variant, n) - data.seconds(variant, n), n)
                for n in data.names
            ),
            reverse=True,
        )
        total = sum(c for c, _ in costs)
        top = ", ".join(f"{n} {c:+.1f}" for c, n in costs[:5])
        print(
            f"Time: the family costs {total:+.1f} s in total; largest: {top}."
            f" Smallest (negative = faster with the family): "
            + ", ".join(f"{n} {c:+.1f}" for c, n in costs[-3:])
            + ".\n"
        )
    print(
        "Stage seconds of the two families in the full run (direct measurement, sum over the"
        " 31 instances):\n"
    )
    probing = sum(
        data.stage("diag", n, "probing")["seconds"] for n in data.names
    )
    enum = sum(
        data.stage("diag", n, "enumeration")["seconds"] for n in data.names
    )
    merge = sum(
        data.diag("diag", n)["cliquemerge"][0]["seconds"] for n in data.names
    )
    total = sum(data.seconds("diag", n) for n in data.names)
    print(
        f"- probing stage {probing:.1f} s, enumeration stage {enum:.1f} s, runCliqueMerging"
        f" inside both {merge:.1f} s, of {total:.1f} s presolve in total.\n"
    )
    name = "sorrell3"
    print(
        f"Note on Stage 3 + 4: with the tool's sizes (the original model for sorrell3, status"
        f" Not reduced) the nnz ratio of sorrell3 would be"
        f" {data.sizes('diag', name)[2] / data.tool_sizes('diag_stage12', name)[2]:.4f}"
        f" instead of {data.sizes('diag', name)[2] / data.sizes('diag_stage12', name)[2]:.4f},"
        " and the geometric mean over the 31 instances "
        + fmt_ratio(
            geomean(
                [
                    data.sizes("diag", n)[2]
                    / data.tool_sizes("diag_stage12", n)[2]
                    for n in data.names
                ]
            )
        )
        + ".\n"
    )
    print(
        "Split of the reduction beyond the Stage 1 reference (mean of log ratios; the three"
        " runs are nested, so the two parts add up exactly), over all instances and without the"
        " instances that carry most of the Stage 2 part:\n"
    )
    rows = []
    subsets = [
        ("all 31", data.names),
        ("without sorrell3", rest),
        (
            "without sorrell3, rd-rplusc-21",
            [n for n in rest if n != "rd-rplusc-21"],
        ),
    ]
    for subset, names in subsets:
        logs = {}
        for key, with_variant, variant in [
            ("beyond", "diag", "diag_stage1"),
            ("stage2", "diag_stage12", "diag_stage1"),
            ("stage34", "diag", "diag_stage12"),
        ]:
            logs[key] = [
                statistics.fmean(
                    math.log(
                        data.sizes(with_variant, n)[k]
                        / data.sizes(variant, n)[k]
                    )
                    for n in names
                )
                for k in range(3)
            ]
        for k, label in ((2, "nonzeros"), (0, "rows"), (1, "columns")):
            rows.append(
                [
                    subset,
                    label,
                    f"{logs['beyond'][k]:+.4f}",
                    f"{logs['stage2'][k]:+.4f}",
                    f"{100 * logs['stage2'][k] / logs['beyond'][k]:.0f}%",
                    f"{logs['stage34'][k]:+.4f}",
                    f"{100 * logs['stage34'][k] / logs['beyond'][k]:.0f}%",
                ]
            )
    table(
        [
            "instances",
            "size",
            "full vs stage1",
            "stage12 vs stage1",
            "share",
            "full vs stage12",
            "share",
        ],
        rows,
        ["l", "l"] + ["r"] * 5,
    )

    print(
        "Distance of each restricted run from full HiGHS (geometric mean of remaining size,"
        " restricted / full) and the instance with the largest nonzero ratio:\n"
    )
    rows = []
    for v in DIAG_VARIANTS[1:]:
        ratio = {
            n: [a / b for a, b in zip(data.sizes(v, n), data.sizes("diag", n))]
            for n in data.names
        }
        worst = max(data.names, key=lambda n: ratio[n][2])
        rows.append(
            [
                SHORT[v],
                fmt_ratio(geomean([ratio[n][2] for n in data.names])),
                fmt_ratio(geomean([ratio[n][0] for n in data.names])),
                fmt_ratio(geomean([ratio[n][1] for n in data.names])),
                f"{worst} {ratio[worst][2]:.3f}",
            ]
        )
    table(
        ["run", "gm nnz", "gm rows", "gm cols", "largest nnz ratio"],
        rows,
        ["l", "r", "r", "r", "l"],
    )

    print(
        "Full run, per stage, summed over the 31 instances: seconds and what the stage itself"
        " removes (what later fast-loop calls remove because of it is in fast_loop). The two"
        " instances with the most seconds are named.\n"
    )
    rows = []
    for stage in STAGES:
        per = {n: data.stage("diag", n, stage) for n in data.names}
        seconds = sum(s["seconds"] for s in per.values())
        if seconds < 0.05 and not any(
            s["rows_removed"] or s["cols_removed"] or s["nnz_removed"]
            for s in per.values()
        ):
            continue
        top = sorted(data.names, key=lambda n: -per[n]["seconds"])[:2]
        rows.append(
            [
                stage,
                f"{seconds:.1f}",
                sum(s["rows_removed"] for s in per.values()),
                sum(s["cols_removed"] for s in per.values()),
                sum(s["nnz_removed"] for s in per.values()),
                ", ".join(
                    f"{n} {per[n]['seconds']:.1f} s ({per[n]['rows_removed']} /"
                    f" {per[n]['cols_removed']} / {per[n]['nnz_removed']})"
                    for n in top
                ),
            ]
        )
    table(
        [
            "stage",
            "seconds",
            "rows removed",
            "cols removed",
            "nnz removed",
            "most seconds: instance, seconds (rows / cols / nnz it removes there)",
        ],
        rows,
        ["l", "r", "r", "r", "r", "l"],
    )


# --------------------------------------------------------------------------------------------
# 4 clique decision
# --------------------------------------------------------------------------------------------


def presolved_classes(variant, name):
    presolved = load_json(STAGE0 / variant / f"{name}.json").get("presolved")
    if not presolved:
        return None
    classes = presolved["row_classes"]
    return {
        "set_packing": classes["set_packing"],
        "set_partitioning": classes["set_partitioning"],
        "two_binary_rows": presolved["two_binary_rows"],
    }


def section_cliques(data):
    print("## 4. The clique decision\n")
    print(
        "Full run: totals of the DIAG cliquemerge line ('rows ext' = rows that got extension"
        " nonzeros, 'rows del' = rows deleted because merging made their clique redundant),"
        " time in runCliqueMerging and its share of the presolve. Paired: full against"
        " diag_nocliquemerge; 'rows fewer' = 1 - rows(full) / rows(no merging), the same for"
        " nonzeros (negative = merging leaves more). Structure of the original model:"
        " set-packing rows (SPP), set-partitioning rows (SPA), rows with two binary entries"
        " (2-bin), binaries.\n"
    )

    def fewer(name, k):
        return (
            1
            - data.sizes("diag", name)[k]
            / data.sizes("diag_nocliquemerge", name)[k]
        )

    rows = []
    acts = []
    changed = []
    for name in data.names:
        orig = load_json(STAGE0 / "stats" / f"{name}.json")["original"]
        classes = orig["row_classes"]
        cm = data.diag("diag", name)["cliquemerge"][0]
        full = data.sizes("diag", name)
        without = data.sizes("diag_nocliquemerge", name)
        t_full = data.seconds("diag", name)
        t_without = data.seconds("diag_nocliquemerge", name)
        if cm["rows_changed"] + cm["rows_deleted"] + cm["fixings"] > 0:
            acts.append(name)
        if full != without:
            changed.append(name)
        rows.append(
            [
                name,
                classes["set_packing"],
                classes["set_partitioning"],
                orig["two_binary_rows"],
                orig["col_types"]["binary"],
                cm["calls"],
                cm["rows_changed"],
                cm["rows_deleted"],
                cm["nnz_added"],
                cm["nnz_removed"],
                cm["fixings"],
                f"{cm['seconds']:.3f}",
                f"{100 * cm['seconds'] / t_full:.1f}%",
                full[0],
                without[0],
                f"{100 * fewer(name, 0):.1f}%",
                f"{100 * fewer(name, 2):+.1f}%",
                without[1] - full[1],
                f"{t_full - t_without:+.2f}",
            ]
        )
    table(
        [
            "instance",
            "SPP",
            "SPA",
            "2-bin",
            "binaries",
            "calls",
            "rows ext",
            "rows del",
            "nnz ext",
            "nnz del",
            "fix",
            "merge s",
            "share of presolve",
            "rows full",
            "rows no merging",
            "rows fewer",
            "nnz fewer",
            "cols fewer (count)",
            "t full - t no merging",
        ],
        rows,
    )

    n = len(data.names)
    print(
        f"- Merging does something (extends or deletes a row, or fixes a column) on"
        f" {len(acts)} of {n} instances: {', '.join(acts)}."
    )
    print(
        f"- The presolve result differs between the two runs on {len(changed)} of {n}:"
        f" {', '.join(changed)}."
    )
    for limit_ in (0.01, 0.10):
        r = [x for x in data.names if fewer(x, 0) > limit_]
        z = [x for x in data.names if fewer(x, 2) > limit_]
        zneg = [x for x in data.names if fewer(x, 2) < -limit_]
        print(
            f"- More than {100 * limit_:.0f}% fewer rows with merging: {len(r)} of {n} ("
            + ", ".join(f"{x} {100 * fewer(x, 0):.1f}%" for x in r)
            + f"). More than {100 * limit_:.0f}% fewer nonzeros: {len(z)} ("
            + ", ".join(f"{x} {100 * fewer(x, 2):.1f}%" for x in z)
            + f"). More than {100 * limit_:.0f}% more nonzeros: {len(zneg)}"
            + (
                " ("
                + ", ".join(f"{x} {-100 * fewer(x, 2):.1f}%" for x in zneg)
                + ")"
                if zneg
                else ""
            )
            + "."
        )
    more_nnz = [
        (x, data.sizes("diag", x)[2] - data.sizes("diag_nocliquemerge", x)[2])
        for x in data.names
        if fewer(x, 2) < 0
    ]
    print(
        f"- Merging leaves more nonzeros than no merging on {len(more_nnz)} instances ("
        + ", ".join(f"{x} {d:+d}" for x, d in more_nnz)
        + ")."
    )
    gm = [
        geomean(
            [
                data.sizes("diag", x)[k]
                / data.sizes("diag_nocliquemerge", x)[k]
                for x in data.names
            ]
        )
        for k in range(3)
    ]
    wo = [x for x in data.names if x != "sorrell3"]
    gm_wo = [
        geomean(
            [
                data.sizes("diag", x)[k]
                / data.sizes("diag_nocliquemerge", x)[k]
                for x in wo
            ]
        )
        for k in range(3)
    ]
    print(
        f"- Geometric mean full / no merging: rows {gm[0]:.4f}, columns {gm[1]:.4f},"
        f" nonzeros {gm[2]:.4f}; without sorrell3: rows {gm_wo[0]:.4f}, columns {gm_wo[1]:.4f},"
        f" nonzeros {gm_wo[2]:.4f}."
    )
    merge = sum(
        data.diag("diag", x)["cliquemerge"][0]["seconds"] for x in data.names
    )
    sorrell = data.diag("diag", "sorrell3")["cliquemerge"][0]["seconds"]
    total = sum(data.seconds("diag", x) for x in data.names)
    paired = sum(
        data.seconds("diag", x) - data.seconds("diag_nocliquemerge", x)
        for x in data.names
    )
    idle = 0.0
    idle_sorrell = 0.0
    for x in data.names:
        for c in data.merge_calls("diag", x):
            if (
                c["rows_changed"] == 0
                and c["rows_deleted"] == 0
                and c["fixings"] == 0
            ):
                idle += c["seconds"]
                if x == "sorrell3":
                    idle_sorrell += c["seconds"]
    print(
        f"- Time in runCliqueMerging, full run: {merge:.2f} s over all instances"
        f" ({100 * merge / total:.1f}% of {total:.1f} s), {sorrell:.2f} s of it on sorrell3;"
        f" largest shares of an instance's presolve: "
        + ", ".join(
            f"{x} {100 * s:.0f}%"
            for s, x in sorted(
                (
                    (
                        data.diag("diag", x)["cliquemerge"][0]["seconds"]
                        / data.seconds("diag", x),
                        x,
                    )
                    for x in data.names
                ),
                reverse=True,
            )[:4]
        )
        + "."
    )
    print(
        f"- Of that, {idle:.2f} s is in calls that change no row and fix no column"
        f" ({idle_sorrell:.2f} s on sorrell3)."
    )
    print(
        f"- Paired time, full minus no merging: {paired:+.1f} s in total (includes what the"
        " later stages do differently and the run-to-run noise of section 5)."
    )
    print()

    def row_gain(name):
        return fewer(name, 0)

    print(
        "Structure against effect. Instances grouped by the set-packing and set-partitioning"
        " rows and the two-binary rows of the original model:\n"
    )
    groups = [
        (
            "no SPP/SPA row and no 2-bin row",
            lambda c, o: c["set_packing"] + c["set_partitioning"] == 0
            and o["two_binary_rows"] == 0,
        ),
        (
            "SPP + SPA rows below 1% of rows",
            lambda c, o: 0
            < c["set_packing"] + c["set_partitioning"] + o["two_binary_rows"]
            and c["set_packing"] + c["set_partitioning"] < 0.01 * o["rows"],
        ),
        (
            "SPP + SPA rows 1% of rows or more",
            lambda c, o: c["set_packing"] + c["set_partitioning"]
            >= 0.01 * o["rows"],
        ),
    ]
    rows = []
    for label, test in groups:
        members = []
        for name in data.names:
            orig = load_json(STAGE0 / "stats" / f"{name}.json")["original"]
            if test(orig["row_classes"], orig):
                members.append(name)
        act = [x for x in members if x in acts]
        gain = [x for x in members if row_gain(x) > 0.01]
        rows.append([label, len(members), len(act), len(gain), ", ".join(act)])
    table(
        [
            "group",
            "instances",
            "merging acts",
            "rows >1% fewer",
            "instances where it acts",
        ],
        rows,
        ["l", "r", "r", "r", "l"],
    )

    print(
        "Row classes after presolve, full against no merging (set-packing rows, two-binary"
        " rows), for the instances where merging acts:\n"
    )
    rows = []
    for name in acts:
        a = presolved_classes("diag", name)
        b = presolved_classes("diag_nocliquemerge", name)
        orig = load_json(STAGE0 / "stats" / f"{name}.json")["original"]
        note = ""
        if data.inst["diag_nocliquemerge"][name]["status"] == "Not reduced":
            note = "no merging: Not reduced, original model"
        rows.append(
            [
                name,
                orig["row_classes"]["set_packing"],
                a["set_packing"],
                b["set_packing"],
                orig["two_binary_rows"],
                a["two_binary_rows"],
                b["two_binary_rows"],
                note,
            ]
        )
    table(
        [
            "instance",
            "SPP original",
            "SPP full",
            "SPP no merging",
            "2-bin original",
            "2-bin full",
            "2-bin no merging",
            "note",
        ],
        rows,
        ["l"] + ["r"] * 6 + ["l"],
    )


# --------------------------------------------------------------------------------------------
# 5 time-target constants
# --------------------------------------------------------------------------------------------


def section_time(data):
    print("## 5. Time-target constants\n")
    print(
        "### 5.1 Run-to-run noise: stock against patched HiGHS, no switch (the same work)\n"
    )
    pairs = sorted(noise_pairs(data), key=lambda p: min(p[1], p[2]))
    table(
        ["instance", "t stock", "t diag", "diag - stock (s)", "relative"],
        [
            [p[0], f"{p[1]:.4f}", f"{p[2]:.4f}", f"{p[3]:+.4f}", pct(p[4])]
            for p in pairs
        ],
    )
    slower = sum(1 for p in pairs if p[3] > 0)
    print(
        f"- The patched binary is slower on {slower} of {len(pairs)} instances; mean relative"
        f" difference {pct(statistics.fmean(p[4] for p in pairs))}, median"
        f" {pct(statistics.median(p[4] for p in pairs))}. No systematic overhead of the"
        " counters is visible at this noise level."
    )
    bins = [
        ("below 0.5 s", 0.0, 0.5),
        ("0.5 to 5 s", 0.5, 5.0),
        ("5 s and above", 5.0, 1e9),
    ]
    rows = []
    for label, low, high in bins:
        sel = [p for p in pairs if low <= min(p[1], p[2]) < high]
        rows.append(
            [
                label,
                len(sel),
                f"{statistics.median(abs(p[3]) for p in sel):.4f}",
                f"{max(abs(p[3]) for p in sel):.4f}",
                f"{100 * statistics.median(abs(p[4]) for p in sel):.2f}%",
                f"{100 * max(abs(p[4]) for p in sel):.2f}%",
            ]
        )
    print()
    table(
        [
            "time",
            "pairs",
            "median abs diff (s)",
            "max abs diff (s)",
            "median relative",
            "max relative",
        ],
        rows,
    )

    print("### 5.2 More pairs of the same work\n")
    print(
        "(a) Patched full run against diag_nocliquemerge on instances where clique merging is a"
        " no-op in the full run (every cliquemerge_call line has rows_changed = rows_deleted ="
        " fixings = 0 and cliques_before = cliques_after) and the results are equal. The two"
        " runs then do the same work apart from the runCliqueMerging calls ('merge s').\n"
    )
    rows = []
    same_work = []
    for name in data.names:
        calls = data.merge_calls("diag", name)
        noop = all(
            c["cliques_before"] == c["cliques_after"]
            and c["rows_changed"] == 0
            and c["rows_deleted"] == 0
            and c["fixings"] == 0
            for c in calls
        )
        if not noop or data.sizes("diag", name) != data.sizes(
            "diag_nocliquemerge", name
        ):
            continue
        a = data.seconds("diag", name)
        b = data.seconds("diag_nocliquemerge", name)
        rel = (b - a) / ((a + b) / 2)
        same_work.append((name, a, b, rel))
        scan = [
            data.stage(v, name, "initial_scan")["seconds"]
            for v in ("diag", "diag_nocliquemerge")
        ]
        read = [
            data.inst[v][name]["read_seconds"]
            for v in ("diag", "diag_nocliquemerge")
        ]
        rows.append(
            [
                name,
                f"{a:.4f}",
                f"{b:.4f}",
                f"{b - a:+.4f}",
                pct(rel),
                f"{data.diag('diag', name)['cliquemerge'][0]['seconds']:.5f}",
                pct((scan[1] - scan[0]) / ((scan[0] + scan[1]) / 2)),
                pct((read[1] - read[0]) / ((read[0] + read[1]) / 2)),
            ]
        )
    table(
        [
            "instance",
            "t diag",
            "t nocliquemerge",
            "diff (s)",
            "relative",
            "merge s",
            "initial scan relative",
            "file read relative",
        ],
        rows,
    )
    print(
        f"- Largest relative difference: "
        f"{100 * max(abs(s[3]) for s in same_work):.1f}%"
        f" ({max(same_work, key=lambda s: abs(s[3]))[0]}).\n"
    )

    print(
        "(a2) The same comparison with a wider test: merging extends and deletes no row and"
        " fixes no column in the full run (it may drop redundant cliques from the table), the"
        " results are equal, and the probing loop does the same work in both runs (probes and"
        " probing_work equal). Pairs of (a) are not repeated.\n"
    )
    rows = []
    wider = []
    strict = {s[0] for s in same_work}
    for name in data.names:
        cm = data.diag("diag", name)["cliquemerge"][0]
        pa = data.probing("diag", name)
        pb = data.probing("diag_nocliquemerge", name)
        if cm["rows_changed"] or cm["rows_deleted"] or cm["fixings"]:
            continue
        if data.sizes("diag", name) != data.sizes("diag_nocliquemerge", name):
            continue
        if (pa["probed"], pa["probing_work"]) != (
            pb["probed"],
            pb["probing_work"],
        ):
            continue
        a = data.seconds("diag", name)
        b = data.seconds("diag_nocliquemerge", name)
        rel = (b - a) / ((a + b) / 2)
        wider.append((name, rel))
        if name in strict:
            continue
        scan = [
            data.stage(v, name, "initial_scan")["seconds"]
            for v in ("diag", "diag_nocliquemerge")
        ]
        rows.append(
            [
                name,
                f"{a:.4f}",
                f"{b:.4f}",
                f"{b - a:+.4f}",
                pct(rel),
                f"{cm['seconds']:.5f}",
                pct((scan[1] - scan[0]) / ((scan[0] + scan[1]) / 2)),
            ]
        )
    table(
        [
            "instance",
            "t diag",
            "t nocliquemerge",
            "diff (s)",
            "relative",
            "merge s",
            "initial scan relative",
        ],
        rows,
    )
    worst = max(wider, key=lambda s: abs(s[1]))
    print(
        f"- {len(wider)} pairs with (a); the diag_nocliquemerge run is the faster one in"
        f" {sum(1 for s in wider if s[1] < 0)}; largest relative difference"
        f" {100 * abs(worst[1]):.1f}% ({worst[0]}).\n"
    )

    print(
        "(b) The initial scan is the same work in all five patched runs of an instance (no"
        " switch reaches it; its removed rows, columns and nonzeros are equal in all five,"
        " checked here). Spread = (max - min) / median of its five times.\n"
    )
    spreads = []
    mismatch = []
    for name in data.names:
        scans = [data.stage(v, name, "initial_scan") for v in DIAG_VARIANTS]
        if any(
            (s["rows_removed"], s["cols_removed"], s["nnz_removed"])
            != (
                scans[0]["rows_removed"],
                scans[0]["cols_removed"],
                scans[0]["nnz_removed"],
            )
            for s in scans
        ):
            mismatch.append(name)
        times = [s["seconds"] for s in scans]
        median = statistics.median(times)
        spreads.append(
            (
                median,
                name,
                (max(times) - min(times)) / median,
                max(times) - min(times),
            )
        )
    print(
        "- Instances where the initial scan removes different amounts in the five runs: "
        + (", ".join(mismatch) if mismatch else "none")
        + "."
    )
    rows = []
    for label, low, high in [
        ("below 0.01 s", 0, 0.01),
        ("0.01 to 0.1 s", 0.01, 0.1),
        ("0.1 s and above", 0.1, 1e9),
    ]:
        sel = [s for s in spreads if low <= s[0] < high]
        if not sel:
            continue
        worst = max(sel, key=lambda s: s[2])
        rows.append(
            [
                label,
                len(sel),
                f"{100 * statistics.median(s[2] for s in sel):.1f}%",
                f"{100 * worst[2]:.1f}% ({worst[1]})",
                f"{max(s[3] for s in sel):.4f}",
            ]
        )
    print()
    table(
        [
            "initial scan time (median of 5)",
            "instances",
            "median spread",
            "max spread",
            "max abs spread (s)",
        ],
        rows,
        ["l", "r", "r", "l", "r"],
    )
    wide = sorted((s for s in spreads if s[2] > 0.15), key=lambda s: -s[2])
    print(
        "- Spread above 15%: "
        + ", ".join(
            f"{s[1]} {100 * s[2]:.0f}% ("
            + " ".join(
                f"{data.stage(v, s[1], 'initial_scan')['seconds']:.3f}"
                for v in DIAG_VARIANTS
            )
            + " s)"
            for s in wide
        )
        + ". Order of the five times: "
        + ", ".join(SHORT[v] for v in DIAG_VARIANTS)
        + ".\n"
    )

    print(
        "(c) Speed of each restricted run relative to the patched full run on identical work:"
        " the initial scan and the file read of the same instance (ratio variant / full). A"
        " ratio below 1 means the whole run of that variant was made under faster conditions;"
        " a paired time difference of that size between the two runs is not an effect of the"
        " switch. Last column: the variant's total time divided by the geometric mean of the"
        " initial-scan ratio, minus the measured total; subtract it from 'family s' of"
        " section 3 for the families measured against that run.\n"
    )
    rows = []
    for v in DIAG_VARIANTS[1:]:
        scan = [
            data.stage(v, n, "initial_scan")["seconds"]
            / data.stage("diag", n, "initial_scan")["seconds"]
            for n in data.names
        ]
        read = [
            data.inst[v][n]["read_seconds"]
            / data.inst["diag"][n]["read_seconds"]
            for n in data.names
        ]
        scan_sum = sum(
            data.stage(v, n, "initial_scan")["seconds"] for n in data.names
        ) / sum(
            data.stage("diag", n, "initial_scan")["seconds"]
            for n in data.names
        )
        total = sum(data.seconds(v, n) for n in data.names)
        rows.append(
            [
                SHORT[v],
                fmt_ratio(geomean(scan)),
                fmt_ratio(scan_sum),
                f"{sum(1 for r in scan if r < 1)} of {len(scan)}",
                fmt_ratio(geomean(read)),
                f"{total:.1f}",
                f"{(1 / geomean(scan) - 1) * total:+.1f}",
            ]
        )
    table(
        [
            "run",
            "initial scan, geometric mean",
            "initial scan, ratio of sums",
            "instances with ratio below 1",
            "file read, geometric mean",
            "sum t (s)",
            "sum t at the full run's speed minus measured (s)",
        ],
        rows,
    )

    print("### 5.3 Distribution of the HiGHS times\n")
    rows = []
    series = [
        ("full HiGHS (stock)", lambda n: data.seconds("stock", n)),
        (
            "Stage 1 reference (diag_stage1)",
            lambda n: data.seconds("diag_stage1", n),
        ),
    ]
    for label, get in series:
        for group in ("A", "B", "A+B"):
            sel = [
                get(n)
                for n in data.names
                if group == "A+B" or BAND_GROUP[data.band(n)] == group
            ]
            rows.append(
                [
                    label,
                    group,
                    len(sel),
                    fmt_s(min(sel)),
                    fmt_s(percentile(sel, 25)),
                    fmt_s(statistics.median(sel)),
                    fmt_s(percentile(sel, 75)),
                    fmt_s(max(sel)),
                    sum(1 for t in sel if t < 0.1),
                    sum(1 for t in sel if t < 1.0),
                    f"{sum(sel):.1f}",
                ]
            )
    table(
        [
            "times",
            "bands",
            "n",
            "min",
            "p25",
            "median",
            "p75",
            "max",
            "below 0.1 s",
            "below 1 s",
            "sum",
        ],
        rows,
        ["l", "l"] + ["r"] * 9,
    )

    print("### 5.4 Choice of the shift\n")
    print(
        "For each candidate shift: the shifted geometric mean (SGM) of the reference and the"
        " factor by which it rises when a subset of the instances takes twice as long: the 4"
        " fastest, the 4 slowest, those below 0.5 s, those of 5 s and more (count in"
        " brackets). Last column: SGM of the patched full run / SGM of the stock full run,"
        " i.e. what the noise between two measurements does to the mean.\n"
    )
    rows = []
    for label, get in series:
        for group in ("A", "B"):
            names = [
                n for n in data.names if BAND_GROUP[data.band(n)] == group
            ]
            base = sorted(get(n) for n in names)
            n_small = sum(1 for t in base if t < 0.5)
            n_large = sum(1 for t in base if t >= 5)
            for shift in (0.01, 0.1, 1.0, 10.0):
                sgm = shifted_geomean(base, shift)
                fast4 = (
                    shifted_geomean(
                        [2 * t if k < 4 else t for k, t in enumerate(base)],
                        shift,
                    )
                    / sgm
                )
                slow4 = (
                    shifted_geomean(
                        [
                            2 * t if k >= len(base) - 4 else t
                            for k, t in enumerate(base)
                        ],
                        shift,
                    )
                    / sgm
                )
                small2 = (
                    shifted_geomean(
                        [2 * t if t < 0.5 else t for t in base], shift
                    )
                    / sgm
                )
                large2 = (
                    shifted_geomean(
                        [2 * t if t >= 5 else t for t in base], shift
                    )
                    / sgm
                )
                noise = shifted_geomean(
                    [data.seconds("diag", n) for n in names], shift
                ) / shifted_geomean(
                    [data.seconds("stock", n) for n in names], shift
                )
                rows.append(
                    [
                        label,
                        group,
                        shift,
                        f"{sgm:.3f}",
                        f"{fast4:.3f}",
                        f"{slow4:.3f}",
                        f"{small2:.3f} ({n_small})",
                        f"{large2:.3f} ({n_large})",
                        f"{noise:.4f}" if label.startswith("full") else "-",
                    ]
                )
    table(
        [
            "times",
            "band",
            "shift (s)",
            "SGM (s)",
            "x2 fastest 4",
            "x2 slowest 4",
            "x2 below 0.5 s",
            "x2 from 5 s",
            "diag / stock",
        ],
        rows,
        ["l", "l"] + ["r"] * 7,
    )

    print("### 5.5 Proposed constants and the limits they give\n")
    abs_noise, rel_noise = noise_bound(data)
    print(f"- Shift of the shifted geometric mean: {SHIFT_SECONDS} s.")
    print(
        f"- Per-instance rule: t(ours) <= {RULE_FACTOR} * t(HiGHS) + {RULE_CONSTANT_SECONDS} s."
    )
    print(
        f"- Largest stock-diag difference below 1 s: {abs_noise:.4f} s; the constant is"
        f" {RULE_CONSTANT_SECONDS / abs_noise:.1f} times that. Largest relative difference of a"
        f" stock-diag pair: {100 * rel_noise:.1f}%."
    )
    s1 = sorted(data.seconds("diag_stage1", n) for n in data.names)
    s1_a = statistics.median(
        data.seconds("diag_stage1", n)
        for n in data.names
        if BAND_GROUP[data.band(n)] == "A"
    )
    for constant in (0.01, 0.02, RULE_CONSTANT_SECONDS, 0.1):
        ratios = [(RULE_FACTOR * t + constant) / t for t in s1]
        loose = sum(1 for r in ratios if r > 1.5)
        print(
            f"- With a constant of {constant} s the Stage 1 limit is {max(ratios):.2f} times the"
            f" HiGHS time at the fastest instance ({s1[0]:.3f} s),"
            f" {(RULE_FACTOR * s1_a + constant) / s1_a:.2f} at the band A median"
            f" ({s1_a:.3f} s), {statistics.median(ratios):.2f} at the median of all, and above"
            f" 1.5 times on {loose} of {len(ratios)} instances."
        )
    small = [abs(p[3]) for p in noise_pairs(data) if min(p[1], p[2]) < 1.0]
    print(
        f"- Stock-diag differences below 1 s ({len(small)} pairs), largest first: "
        + ", ".join(f"{1000 * d:.1f}" for d in sorted(small, reverse=True))
        + " ms."
    )
    same_work_max = 0.0
    for name in data.names:
        calls = data.merge_calls("diag", name)
        if all(
            c["cliques_before"] == c["cliques_after"]
            and c["rows_changed"] == 0
            and c["rows_deleted"] == 0
            and c["fixings"] == 0
            for c in calls
        ) and data.sizes("diag", name) == data.sizes(
            "diag_nocliquemerge", name
        ):
            a = data.seconds("diag", name)
            b = data.seconds("diag_nocliquemerge", name)
            same_work_max = max(same_work_max, abs(b - a) / ((a + b) / 2))
    print(
        f"- The 20% of the rule against single-run noise: {100 * rel_noise:.1f}% is the largest"
        f" stock-diag difference, {100 * same_work_max:.1f}% the largest difference of the"
        " same-work pairs of 5.2 (a). One run per side is not enough to apply the rule."
    )
    print()
    print(
        f"Band targets, shifted geometric mean with shift {SHIFT_SECONDS} s:\n"
    )
    rows = []
    groups = [
        ("A", lambda n: BAND_GROUP[data.band(n)] == "A"),
        ("B", lambda n: BAND_GROUP[data.band(n)] == "B"),
        (
            "B without rd-rplusc-21",
            lambda n: BAND_GROUP[data.band(n)] == "B" and n != "rd-rplusc-21",
        ),
    ]
    for group, test in groups:
        names = [n for n in data.names if test(n)]
        rows.append(
            [group, len(names)]
            + [
                f"{shifted_geomean([data.seconds(v, n) for n in names], SHIFT_SECONDS):.3f}"
                for v in ("diag_stage1", "diag_stage12", "stock", "diag")
            ]
        )
    table(
        [
            "band",
            "n",
            "Stage 1 gate: diag_stage1",
            "diag_stage12",
            "full, stock",
            "full, diag",
        ],
        rows,
        ["l"] + ["r"] * 5,
    )
    print(
        "Instances where one stage takes more than 60% of the Stage 1 reference time:\n"
    )
    rows = []
    for name in data.names:
        total = data.seconds("diag_stage1", name)
        for stage in STAGES:
            seconds = data.stage("diag_stage1", name, stage)["seconds"]
            if seconds > 0.6 * total:
                extra = ""
                if stage == "parallel_rows_cols":
                    par = data.diag("diag_stage1", name)["parallel"][0]
                    extra = (
                        f"row loop {par['row_seconds']:.2f} s, hashing"
                        f" {par['hash_seconds']:.2f} s,"
                        f" duplicate rows {par['duplicate_rows']}"
                    )
                rows.append(
                    [
                        name,
                        fmt_s(total),
                        stage,
                        fmt_s(seconds),
                        f"{100 * seconds / total:.0f}%",
                        extra,
                    ]
                )
    table(
        ["instance", "t stage1", "stage", "stage s", "share", "detail"],
        rows,
        ["l", "r", "l", "r", "r", "l"],
    )
    print(
        "Per-instance limits. Full HiGHS time = the smaller of the stock and the patched run.\n"
    )
    rows = []
    for name in data.names:
        s1 = data.seconds("diag_stage1", name)
        full = data.full_seconds(name)
        l1 = limit(s1)
        lf = limit(full)
        rows.append(
            [
                name,
                data.band(name),
                fmt_s(s1),
                fmt_s(l1),
                f"{l1 / s1:.2f}",
                fmt_s(full),
                fmt_s(lf),
                f"{lf / full:.2f}",
            ]
        )
    table(
        [
            "instance",
            "band",
            "t stage1",
            "Stage 1 limit",
            "limit / t",
            "t full",
            "full limit",
            "limit / t",
        ],
        rows,
        ["l", "l"] + ["r"] * 6,
    )


def limit(seconds):
    return RULE_FACTOR * seconds + RULE_CONSTANT_SECONDS


# --------------------------------------------------------------------------------------------
# 6 status instances
# --------------------------------------------------------------------------------------------


def section_status(data):
    print("## 6. Status instances (stock HiGHS presolve)\n")
    rows = []
    for i in data.status["instances"]:
        t = data.status_testbed[i["name"]]
        after = "-"
        if i["rows_after"] is not None:
            after = f"{i['rows_after']} / {i['cols_after']} / {i['nnz_after']}"
        core = i["presolve_core_seconds"]
        rule = "-"
        if i["band"] in BAND_GROUP and core is not None:
            rule = fmt_s(limit(core))
        rows.append(
            [
                i["name"],
                i["band"],
                t["objective"],
                f"{i['rows']} / {i['cols']} / {i['nnz']}",
                i["status"],
                after,
                fmt_s(core),
                fmt_s(i["presolve_seconds"]),
                "yes" if i["reached_time_limit"] else "no",
                rule,
            ]
        )
    table(
        [
            "instance",
            "band",
            "MIPLIB",
            "rows / cols / nnz",
            "HiGHS presolve status",
            "after presolve",
            "core s",
            "call s",
            "time limit",
            f"{RULE_FACTOR} t + {RULE_CONSTANT_SECONDS} s",
        ],
        rows,
        ["l", "l", "l", "r", "l", "r", "r", "r", "l", "r"],
    )
    print(
        "The per-instance time rule of the plan covers bands A and B (5e4 to 1e7 nonzeros), so"
        " the last column is filled only there: 'below' is under 50,000 nonzeros and band D is"
        " reported only. Without a 'Presolve reductions' line HiGHS gives no core time; 'call s'"
        " then includes the set-up of the MIP solver.\n"
    )


# --------------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------------


def write_csv(data):
    header = ["band", "instance", "rows", "cols", "nnz"]
    for v in VARIANTS:
        header += [
            f"{v}_status",
            f"{v}_rows",
            f"{v}_cols",
            f"{v}_nnz",
            f"{v}_core_seconds",
        ]
    probing_fields = ["probed", "fixed", "substitutions", "implications"]
    for v in ("diag", "diag_stage12"):
        header += [f"{v}_probing_{k}" for k in probing_fields] + [
            f"{v}_probing_seconds"
        ]
    header += [
        "stage1_limit_seconds",
        "full_limit_seconds",
        "stage1_stage3_col_removals",
        "notes",
    ]
    with open(CSV_OUT, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for name in data.names:
            t = data.testbed[name]
            row = [t["band"], name, t["rows"], t["cols"], t["nnz"]]
            notes = []
            for v in VARIANTS:
                i = data.inst[v][name]
                row += [
                    i["status"],
                    *data.sizes(v, name),
                    f"{data.seconds(v, name):.6f}",
                ]
                if data.sizes(v, name) != data.tool_sizes(v, name):
                    notes.append(
                        f"{v}: sizes from DIAG total; the tool reports the original"
                        f" model (status {i['status']})"
                    )
            for v in ("diag", "diag_stage12"):
                row += [data.probing(v, name)[k] for k in probing_fields]
                row.append(f"{data.stage(v, name, 'probing')['seconds']:.6f}")
            row += [
                f"{limit(data.seconds('diag_stage1', name)):.3f}",
                f"{limit(data.full_seconds(name)):.3f}",
                stage3_in_stage1(data, name)["columns"],
                "; ".join(notes),
            ]
            writer.writerow(row)
    print(
        f"Wrote {CSV_OUT} ({len(data.names)} rows; limits = {RULE_FACTOR} * t +"
        f" {RULE_CONSTANT_SECONDS} s, full time = min(stock, diag))."
    )


def main():
    data = Data()
    print("# Stage 0 targets\n")
    section_targets(data)
    section_references(data)
    section_families(data)
    section_cliques(data)
    section_time(data)
    section_status(data)
    write_csv(data)


if __name__ == "__main__":
    main()
