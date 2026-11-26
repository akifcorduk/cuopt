# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 reductions analysis: which HiGHS presolve rules remove rows, columns and nonzeros.

Reads the Stage 0 result files next to this script and prints markdown tables for the dev-size
instances of the main testbed (below 1e7 nonzeros):

1. sizes before and after HiGHS presolve, and the removal per DIAG stage (with the check that the
   stages add up to the total);
2. postsolve entries by type, the split of `doubleton_equation` and of the column rules;
3. rows and columns by the rule that executed the removal, mapped to the stages of DESIGN.md
   section 5 (S1 row rules / propagation / doubleton equations / parallel rows / dual fixing by
   locks, S2 probing and cliques, S3 dual and column rules and the aggregator, S4 sparsify and
   cover strengthening, OUT = left out of the first versions);
4. lower and upper estimates of the share of HiGHS's removal that S1, S1+S2 and S1+S2+S3 reach;
5. how often each stage, postsolve type and counter fires;
6. integer and implied-integer columns and the shape of the reduced model.

What the data cannot separate (see HIGHS_DIAG.md and HPresolve.cpp of the measured commit):
- Nonzeros are counted per DIAG stage only. The initial scan and the fast loop run S1 row rules
  and S3 column rules in one pass, so their nonzeros are mixed.
- `fixed_col` is pushed by every fixing rule. In the initial scan, the fast loop and the cascades
  inside other stages it is a fixed or empty column, a singleton or forcing row, dual fixing by
  locks (S1), or a dominated / weakly dominated column, the dual-fixing extensions, stuffing or a
  gcd fixing (S3).
- A removal is counted for the rule that executed it. A row that an S1 rule removes only after a
  later stage changed the model (for example a redundant row after a probing fixing) is still an
  S1-rule removal here; only the run with probing off measures such a cascade.
- HiGHS tests row redundancy on the activity sums over implied column bounds (isRedundant uses
  impliedRowBounds.getSumLower / getSumUpper, not the *Orig sums). The plan's Stage 1 keeps the
  sums on explicit bounds only (DESIGN.md 2.4). The counts do not say which redundant rows need
  the implied bounds, so "S1 rule" below means the rule, not the Stage 1 activity design.

Estimates (section 4), per measure, as a share of HiGHS's net removal:
- S1 low: certain S1 removals of the initial scan only (rows: singleton, redundant, doubleton
  equation; columns: doubleton equations and columns removed with no nonzero; nonzeros: the whole
  scan if every column it removes is one of those, else 2 per doubleton equation + 1 per singleton
  row, which is the least those rules remove).
- S1 high: everything an S1 rule executed or may have executed in any stage (rows: singleton,
  redundant, doubleton, parallel; columns: doubleton equations and every `fixed_col` outside the
  probing stage, the pairwise dominated-column stage and parallel-column dominance; nonzeros:
  initial scan + fast loop + parallel rows + the sparsify calls that also remove rows + the
  fixings of the enumeration stage). It is taken from the run with probing off when that run
  exists, because there no S1-rule removal can follow from a probing result; the value of the
  normal run is printed next to it.
- S1+S2 low / high: the S1 low plus the certain S2 removals (probing stage, clique merging, clique
  and implication substitutions) / the S1 high of the normal run plus the same S2 removals.
- S1+S2+S3 low / high: total minus S4 (sparsify cancellation, cover strengthening) minus OUT
  (Fourier-Motzkin, variable-bound aggregation); the low value also drops the fixings of the
  enumeration stage and the sparsify calls that mix cancellation with row removals.
A share above 100% means the rule set would leave fewer nonzeros than HiGHS, because a later
HiGHS stage adds nonzeros (aggregator fill-in, clique extension, variable-bound aggregation). A
negative S1+S2 low means the S2 stages add more nonzeros than the certain S1 part removes.

These estimates are counts by executing rule, not bounds:
- The low values count S1-rule entries of the whole initial scan. The scan runs colPresolve on
  all columns after its row pass, so rows that become singleton or redundant after a column
  fixing of unknown rule are included (supportcase40: 20 singleton rows in the original model,
  6435 singleton_row entries in the scan).
- The high values keep every S1-rule removal that follows an aggregator, sparsify, pairwise
  dominated-column, Fourier-Motzkin or enumeration step, and the whole sparsify cell when the call
  also removes rows. Only the cascade after probing is taken out.
The measured restricted runs (section 4, "Measured restricted runs") are the check: HiGHS with
stages switched off by HIGHS_DIAG_OFF. Their sizes are read from the DIAG total line, not from
the returned model: HiGHS returns status "Not reduced" and the original model when no postsolve
entry was pushed, also when clique merging deleted rows (sorrell3 in diag_stage12). The stage
switches do not reach the S3 rules inside row and column presolve (dominated columns, dual-fixing
extensions and substitution, singleton-column rules, simple probing, implied equations), so a
restricted run is "the switched-on stages plus those rules".

Inputs, relative to this file:
  ../testbed/testbed.json                         "main" list; instances with where == "dev"
  ../results/stage0/diag/summary.json             patched HiGHS: sizes and parsed DIAG lines
  ../results/stage0/diag/<name>.json              structure statistics before / after presolve
  ../results/stage0/diag/<name>.json.log          HiGHS log (three dev-level count lines)
  ../results/stage0/stock/summary.json            stock HiGHS: sizes, to check the patched run
  ../results/stage0/diag_noprobing/summary.json   optional: the same run with HIGHS_DIAG_OFF=probing
                                                  (runProbing is not called; every other stage runs)
  ../results/stage0/diag_nocliquemerge/summary.json  optional: HIGHS_DIAG_OFF=clique_merging
  ../results/stage0/diag_stage1/summary.json      optional: only the initial scan, the fast loop and
                                                  parallel rows are left on
  ../results/stage0/diag_stage12/summary.json     optional: the same plus probing, clique merging and
                                                  clique / implication substitutions

No arguments. Python standard library only. Nothing is written.
"""

import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE0 = os.path.normpath(os.path.join(HERE, "..", "results", "stage0"))
TESTBED = os.path.normpath(os.path.join(HERE, "..", "testbed", "testbed.json"))

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

TYPES = [
    "singleton_row",
    "redundant_row",
    "forcing_row",
    "fixed_col",
    "doubleton_equation",
    "free_col_substitution",
    "duplicate_row",
    "duplicate_column",
    "implied_equation",
    "forcing_column",
    "forcing_column_removed_row",
    "zero_obj_singleton_col",
    "linear_transform",
    "equality_row_addition",
    "equality_row_additions",
    "fourier_motzkin_block",
    "fourier_motzkin_obj_col",
]

# Stages whose fixed_col entries come from row and column presolve (rule not identified).
ROWCOL_FIX_STAGES = [
    s
    for s in STAGES
    if s not in ("probing", "enumeration", "dominated_columns")
]

ROW_RULES = [
    ("singleton_row", "S1"),
    ("redundant_row", "S1"),
    ("doubleton_eq", "S1"),
    ("parallel_row", "S1"),
    ("clique_merging", "S2"),
    ("probing_lifting", "S2"),
    ("free_col_subst", "S3"),
    ("forcing_col", "S3"),
    ("simple_probing", "S3"),
    ("vb_aggregation", "OUT"),
    ("fourier_motzkin", "OUT"),
]

COL_RULES = [
    ("doubleton_eq", "S1"),
    ("fix_empty_col", "S1"),
    ("fix_row_col_presolve", "S1|S3"),
    ("fix_enumeration_stage", "S1|S2|OUT"),
    ("fix_probing_stage", "S2"),
    ("subst_clique", "S2"),
    ("subst_implication", "S2"),
    ("fix_pairwise_dominated", "S3"),
    ("fix_parallel_dominated", "S3"),
    ("parallel_col_merge", "S3"),
    ("free_col_aggregator", "S3"),
    ("free_col_singleton", "S3"),
    ("forcing_col", "S3"),
    ("zero_obj_singleton", "S3"),
    ("subst_simple_probing", "S3"),
    ("subst_dual_fixing", "S3"),
    ("fourier_motzkin", "OUT"),
]

NNZ_CELLS = [
    "initial_scan",
    "fast_loop",
    "parallel_rows_cols",
    "aggregator",
    "dominated_columns",
    "sparsify",
    "strengthen_inequalities",
    "probing",
    "enum_clique_merging",
    "enum_other",
    "vb_aggregation",
    "fourier_motzkin",
    "rest",
]

STAGE_MAP = [
    (
        "setup, shrink, final_scaling, other",
        "none",
        "data structures, compaction, scaling; remove nothing",
    ),
    (
        "initial_scan",
        "S1 + S3 mixed",
        "rowPresolve on all rows (S1 row rules, doubleton equations, bound tightening of integer and "
        "implied-integer columns from the explicit-bound residual activity, with the column's rows "
        "queued again by changeColLower / changeColUpper so that it chains over later fast-loop "
        "passes, integral scaling / Chvatal-Gomory / coefficient strengthening; S3: implied "
        "equations, simple probing on equations, gcd transform), then colPresolve on all columns "
        "(S1: fixed and empty columns, dual fixing by locks; S3: dominated / weakly dominated / "
        "forcing columns, dual-fixing extensions, stuffing, singleton-column substitution, "
        "zero-cost singleton, implied integers)",
    ),
    (
        "fast_loop",
        "S1 + S3 mixed, plus cascade",
        "the same rules on changed rows and columns after every other stage",
    ),
    (
        "clique_substitutions",
        "S2",
        "empty at the root presolve (substitutions are applied inside probing / enumeration)",
    ),
    ("fourier_motzkin", "OUT", "left out of the first versions"),
    (
        "aggregator",
        "S3",
        "free-column substitution; singleton-row cascade inside",
    ),
    (
        "sparsify",
        "S4, with S1 cascade",
        "starts and ends with removeRowSingletons / removeDoubletonEquations, so its rows and "
        "columns are S1-rule removals; its nonzeros are cancellation (S4) plus those cascades",
    ),
    (
        "parallel_rows_cols",
        "rows S1, columns S3",
        "column loop: merge (duplicate_column) and dominance (fixed_col, cols_dominated) = S3; row "
        "loop: duplicate_row and nearly parallel rows (equality_row_addition, then a doubleton "
        "equation) = S1",
    ),
    (
        "strengthen_inequalities",
        "S4",
        "cover-based coefficient strengthening; changes coefficients, removes no row or column",
    ),
    (
        "dominated_columns",
        "S3",
        "pairwise dominated columns and predictive bound analysis; row cascade inside",
    ),
    (
        "enumeration",
        "OUT, S2 and S1 mixed",
        "prepareProbing (implied bounds made explicit, HighsDomain propagation to a fixpoint = S1 "
        "propagation; clique extraction = S2), enumeration of binary rows up to 8 entries (OUT), "
        "finaliseProbing (clique merging and clique substitutions = S2, variable-bound aggregation "
        "= OUT)",
    ),
    ("implied_integers", "S3", "type conversion only"),
    (
        "probing",
        "S2",
        "probing, clique merging, substitutions, lifting; its variable-bound aggregation is OUT",
    ),
]

TYPE_MAP = [
    (
        "singleton_row",
        "S1",
        "HPresolve::singletonRow (also reached from removeRowSingletons in every stage)",
    ),
    (
        "redundant_row",
        "S1 (S2 in stage probing)",
        "rowPresolve checkRowRedundant / emptyRow / redundant singleton; also pushed once per "
        "forcing row; in stage probing it is lifting for probing",
    ),
    (
        "forcing_row",
        "S1",
        "rowPresolve checkForcingRow; its fixings are fixed_col; the row is also counted in redundant_row",
    ),
    (
        "fixed_col",
        "mixed S1 / S2 / S3 / OUT",
        "S1: colPresolve fixed column, emptyCol, singletonRow with equal bounds, forcing row, "
        "dualFixing by locks; S3: detectDominatedCol (dominated, weakly dominated), dualFixing "
        "single-equation extension and worst-case bound, singletonColStuffing, gcd fixing, simple "
        "probing, dominatedColumns, parallel-column dominance; S2 / OUT / S1: finaliseProbing in "
        "probing and enumeration (probing, clique table, enumeration, propagation fixpoint)",
    ),
    (
        "doubleton_equation",
        "mixed S1 / S2 / S3",
        "S1: doubletonEq (DIAG doubleton); S2: clique and implication substitutions (DIAG "
        "substitutions applied); S3: simple probing on an equation and the dual-fixing substitution",
    ),
    (
        "free_col_substitution",
        "S3",
        "aggregator (DIAG aggregator substituted) and singletonCol in column presolve (the rest)",
    ),
    ("duplicate_row", "S1", "detectParallelRowsAndCols row loop"),
    (
        "equality_row_addition",
        "S1 (parallel rows)",
        "nearly parallel rows: an equation added to the other row, which then goes through rowPresolve",
    ),
    ("duplicate_column", "S3", "parallel columns merged"),
    ("implied_equation", "S3", "rowPresolve, from row-dual bounds"),
    (
        "forcing_column, forcing_column_removed_row",
        "S3",
        "detectDominatedCol, weakly dominated column with an infinite bound",
    ),
    (
        "zero_obj_singleton_col",
        "S3",
        "zeroCostSingleton: zero-cost continuous singleton column in a ranged row",
    ),
    (
        "linear_transform",
        "S3 (no removal)",
        "transformColumn: gcd substitution and implied-integer scaling in rowPresolve, shift of a "
        "binary to [0, 1] in colPresolve, and the final scaling of continuous columns (scaleMIP)",
    ),
    ("equality_row_additions", "S4", "sparsify"),
    (
        "fourier_motzkin_block, fourier_motzkin_obj_col",
        "OUT",
        "Fourier-Motzkin",
    ),
    (
        "(no entry) rows of clique merging and of the clique table",
        "S2",
        "DIAG cliquemerge_call",
    ),
    (
        "(no entry) rows of variable-bound aggregation",
        "OUT",
        "DIAG varbound_aggregation",
    ),
    (
        "(no entry) row of simple probing on an equation",
        "S3",
        "rowPresolve removes the row without an entry; seen as rows of the scan / fast loop that no entry explains",
    ),
]


# Runs of the patched HiGHS with stages switched off (directory under results/stage0, what is on).
RESTRICTED_RUNS = [
    ("diag_noprobing", "every stage except probing"),
    ("diag_nocliquemerge", "every stage, clique merging off"),
    ("diag_stage1", "initial scan, fast loop, parallel rows"),
    (
        "diag_stage12",
        "initial scan, fast loop, parallel rows, probing, clique merging and substitutions",
    ),
]


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def diag_removed(inst):
    """Rows, columns and nonzeros removed according to the DIAG total line."""
    total = inst["diag"]["total"][0]
    out = {}
    for key in ("rows", "cols", "nnz"):
        before, after = pair(total[key])
        out[key] = before - after
    return out


def pair(text):
    before, after = text.split(":")
    return int(before), int(after)


def pct(part, whole):
    if whole is None or whole <= 0 or part is None:
        return None
    return 100.0 * part / whole


def fmt_pct(value, digits=1):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_num(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                fmt_num(cell) if not isinstance(cell, str) else cell
                for cell in row
            )
            + " |"
        )
    print()


def log_counts(path):
    """Counts from three HiGHS dev-level lines of the presolve log."""
    out = {
        "strengthened_coefs": 0,
        "pairwise_dominated_fixed": 0,
        "pairwise_bounds_strengthened": 0,
        "fm_lines": 0,
    }
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as handle:
        for line in handle:
            words = line.split()
            if (
                line.startswith("Strengthened ")
                and words[-1] == "coefficients"
            ):
                out["strengthened_coefs"] += int(words[1])
            elif (
                line.startswith("Fixed ")
                and "dominated columns and strengthened" in line
            ):
                out["pairwise_dominated_fixed"] += int(words[1])
                out["pairwise_bounds_strengthened"] += int(words[6])
            elif line.startswith("Fourier-Motzkin"):
                out["fm_lines"] += 1
    return out


def analyse(inst):
    diag = inst["diag"]
    stage = diag["stage"]
    assert list(stage.keys()) == STAGES, inst["name"]
    assert len(diag["total"]) == 1, inst["name"]
    total = diag["total"][0]
    rows_b, rows_a = pair(total["rows"])
    cols_b, cols_a = pair(total["cols"])
    nnz_b, nnz_a = pair(total["nnz"])
    int_b, int_a = pair(total["integer_cols"])
    out = {
        "name": inst["name"],
        "band": inst["band"],
        "status": total["status"],
        "before": (rows_b, cols_b, nnz_b),
        "after": (rows_a, cols_a, nnz_a),
        "removed": {
            "rows": rows_b - rows_a,
            "cols": cols_b - cols_a,
            "nnz": nnz_b - nnz_a,
        },
        "integer": (int_b, int_a),
        "implied_integer": total["implied_integer_cols"],
        "stage": stage,
    }
    removed = out["removed"]

    red = {entry["stage"]: entry for entry in diag["reductions"]}
    all_red = red["all"]

    def ent(stage_name, kind):
        return red.get(stage_name, {}).get(kind, 0)

    out["types"] = {kind: all_red.get(kind, 0) for kind in TYPES}
    out["ent"] = ent
    dbl = diag["doubleton"][0]
    sub = diag["substitutions"][0]
    agg = diag["aggregator"][0]
    par = diag["parallel"][0]
    vba = diag["varbound_aggregation"]
    probing = diag["probing"][0]
    merge = diag["cliquemerge"][0]
    out.update(
        dbl=dbl,
        sub=sub,
        agg=agg,
        par=par,
        vba=vba,
        probing=probing,
        merge=merge,
    )

    merge_rows = defaultdict(int)
    merge_net = defaultdict(int)
    # No cliquemerge_call line is printed when neither enumeration nor probing runs.
    for call in diag.get("cliquemerge_call", []):
        merge_rows[call["stage"]] += (
            call["rows_deleted"] + call["rows_deleted_before_merge"]
        )
        merge_net[call["stage"]] += call["nnz_removed"] - call["nnz_added"]

    # ---- checks -------------------------------------------------------------------------
    checks = []
    for key, field in (
        ("rows", "rows_removed"),
        ("cols", "cols_removed"),
        ("nnz", "nnz_removed"),
    ):
        stage_sum = sum(stage[s][field] for s in STAGES)
        if stage_sum != removed[key]:
            checks.append(
                f"stage {key} sum {stage_sum} != total {removed[key]}"
            )
    if (inst["rows"], inst["cols"], inst["nnz"]) != out["before"]:
        checks.append("sizes before differ between the tool and DIAG total")
    if (inst["rows_after"], inst["cols_after"], inst["nnz_after"]) != out[
        "after"
    ]:
        checks.append("sizes after differ between the tool and DIAG total")
    for kind in all_red:
        if kind == "stage":
            continue
        if (
            sum(entry[kind] for name, entry in red.items() if name != "all")
            != all_red[kind]
        ):
            checks.append(
                f"reductions {kind}: stage lines do not add up to stage=all"
            )
    other_subst = sub["applied"] + sub["simple_probing"] + sub["dual_fixing"]
    if all_red["doubleton_equation"] != dbl["removed"] + other_subst:
        checks.append(
            "doubleton_equation != doubleton removed + substitutions"
        )
    if sub["applied"] != sub["from_cliques"] + sub["from_implications"]:
        checks.append(
            "substitutions applied != from_cliques + from_implications"
        )
    if agg["substituted"] != ent("aggregator", "free_col_substitution"):
        checks.append(
            "aggregator substituted != free_col_substitution of stage aggregator"
        )
    if par["duplicate_rows"] != all_red["duplicate_row"]:
        checks.append("parallel duplicate_rows != duplicate_row entries")
    if par["cols_merged"] != all_red["duplicate_column"]:
        checks.append("parallel cols_merged != duplicate_column entries")
    if par["row_additions"] != all_red["equality_row_addition"]:
        checks.append(
            "parallel row_additions != equality_row_addition entries"
        )
    if ent("parallel_rows_cols", "fixed_col") < par["cols_dominated"]:
        checks.append(
            "parallel cols_dominated > fixed_col of stage parallel_rows_cols"
        )

    # ---- rows by executing rule ------------------------------------------------------------
    row_resid = {}
    col_resid = {}
    for name in STAGES:
        explained = (
            ent(name, "singleton_row")
            + ent(name, "redundant_row")
            + ent(name, "duplicate_row")
            + ent(name, "forcing_column_removed_row")
            + ent(name, "free_col_substitution")
            + dbl.get(name, 0)
            + merge_rows.get(name, 0)
            + (vba[name]["rows_removed"] if name in vba else 0)
        )
        row_resid[name] = stage[name]["rows_removed"] - explained
        explained_cols = (
            ent(name, "fixed_col")
            + ent(name, "doubleton_equation")
            + ent(name, "free_col_substitution")
            + ent(name, "duplicate_column")
            + ent(name, "forcing_column")
            + ent(name, "zero_obj_singleton_col")
        )
        col_resid[name] = stage[name]["cols_removed"] - explained_cols
    simple_probing_rows = sum(
        v for s, v in row_resid.items() if s != "fourier_motzkin"
    )
    if simple_probing_rows < 0 or (
        simple_probing_rows > 0 and sub["simple_probing"] == 0
    ):
        checks.append(
            f"{simple_probing_rows} rows without an entry and no simple probing"
        )
    if any(v != 0 for s, v in col_resid.items() if s != "fourier_motzkin"):
        checks.append(
            "columns of a stage not explained by its postsolve entries"
        )
    out["simple_probing_rows_by_stage"] = {
        s: v for s, v in row_resid.items() if v and s != "fourier_motzkin"
    }

    rows = {
        "singleton_row": all_red["singleton_row"],
        "redundant_row": all_red["redundant_row"]
        - ent("probing", "redundant_row"),
        "doubleton_eq": dbl["removed"],
        "parallel_row": all_red["duplicate_row"],
        "clique_merging": sum(merge_rows.values()),
        "probing_lifting": ent("probing", "redundant_row"),
        "free_col_subst": all_red["free_col_substitution"],
        "forcing_col": all_red["forcing_column_removed_row"],
        "simple_probing": simple_probing_rows,
        "vb_aggregation": sum(v["rows_removed"] for v in vba.values()),
        "fourier_motzkin": row_resid["fourier_motzkin"],
    }
    if sum(rows.values()) != removed["rows"]:
        checks.append("row rules do not add up to the rows removed")
    out["rows"] = rows

    # ---- columns by executing rule ---------------------------------------------------------
    fixed = {s: ent(s, "fixed_col") for s in STAGES}
    out["fixed_by_stage"] = fixed

    def empty_cols_cell(name):
        cell = stage[name]
        own = fixed[name] - (
            par["cols_dominated"] if name == "parallel_rows_cols" else 0
        )
        if (
            own > 0
            and cell["rows_removed"] == 0
            and cell["nnz_removed"] == 0
            and cell["cols_removed"] == own
        ):
            return own
        return 0

    empty_by_stage = {s: empty_cols_cell(s) for s in ROWCOL_FIX_STAGES}
    rowcol_fixed = (
        sum(fixed[s] for s in ROWCOL_FIX_STAGES) - par["cols_dominated"]
    )
    cols = {
        "doubleton_eq": dbl["removed"],
        "fix_empty_col": sum(empty_by_stage.values()),
        "fix_row_col_presolve": rowcol_fixed - sum(empty_by_stage.values()),
        "fix_enumeration_stage": fixed["enumeration"],
        "fix_probing_stage": fixed["probing"],
        "subst_clique": sub["from_cliques"],
        "subst_implication": sub["from_implications"],
        "fix_pairwise_dominated": fixed["dominated_columns"],
        "fix_parallel_dominated": par["cols_dominated"],
        "parallel_col_merge": all_red["duplicate_column"],
        "free_col_aggregator": agg["substituted"],
        "free_col_singleton": all_red["free_col_substitution"]
        - agg["substituted"],
        "forcing_col": all_red["forcing_column"],
        "zero_obj_singleton": all_red["zero_obj_singleton_col"],
        "subst_simple_probing": sub["simple_probing"],
        "subst_dual_fixing": sub["dual_fixing"],
        "fourier_motzkin": col_resid["fourier_motzkin"],
    }
    if sum(cols.values()) != removed["cols"]:
        checks.append("column rules do not add up to the columns removed")
    out["cols"] = cols

    # ---- nonzeros by stage cell ------------------------------------------------------------
    vba_nnz = {name: vba[name]["nnz_removed"] for name in vba}
    cells = {
        "initial_scan": stage["initial_scan"]["nnz_removed"],
        "fast_loop": stage["fast_loop"]["nnz_removed"],
        "parallel_rows_cols": stage["parallel_rows_cols"]["nnz_removed"],
        "aggregator": stage["aggregator"]["nnz_removed"],
        "dominated_columns": stage["dominated_columns"]["nnz_removed"],
        "sparsify": stage["sparsify"]["nnz_removed"],
        "strengthen_inequalities": stage["strengthen_inequalities"][
            "nnz_removed"
        ],
        "probing": stage["probing"]["nnz_removed"] - vba_nnz.get("probing", 0),
        "enum_clique_merging": merge_net.get("enumeration", 0),
        "enum_other": stage["enumeration"]["nnz_removed"]
        - merge_net.get("enumeration", 0)
        - vba_nnz.get("enumeration", 0),
        "vb_aggregation": sum(vba_nnz.values()),
        "fourier_motzkin": stage["fourier_motzkin"]["nnz_removed"],
    }
    cells["rest"] = removed["nnz"] - sum(cells.values())
    if cells["rest"] != 0:
        checks.append(
            f"{cells['rest']} nonzeros in stages that should remove nothing"
        )
    out["cells"] = cells

    par_rows = par["duplicate_rows"] + par["row_additions"]
    par_cols = par["cols_merged"] + par["cols_dominated"]
    par_cell = stage["parallel_rows_cols"]
    if (
        par_cols == 0
        and par_cell["cols_removed"]
        == dbl["parallel_rows_cols"] + fixed["parallel_rows_cols"]
    ):
        par_class = "S1"
    elif par_rows == 0 and par_cell["rows_removed"] == 0:
        par_class = "S3"
    else:
        par_class = "S1|S3"
    if not (
        par_cell["rows_removed"]
        or par_cell["cols_removed"]
        or par_cell["nnz_removed"]
    ):
        par_class = "-"
    out["par_class"] = par_class
    spars = stage["sparsify"]
    if not (
        spars["rows_removed"] or spars["cols_removed"] or spars["nnz_removed"]
    ):
        out["sparsify_class"] = "-"
    elif spars["rows_removed"] == 0 and spars["cols_removed"] == 0:
        out["sparsify_class"] = "S4"
    else:
        out["sparsify_class"] = "S4|S1"

    # ---- evidence of S3 rules inside the scan and the fast loop -----------------------------
    def s3_entries(name):
        return (
            ent(name, "free_col_substitution")
            + ent(name, "forcing_column")
            + ent(name, "zero_obj_singleton_col")
            + ent(name, "implied_equation")
            + ent(name, "doubleton_equation")
            - dbl.get(name, 0)
        )

    out["scan_s3"] = s3_entries("initial_scan")
    out["fast_s3"] = s3_entries("fast_loop")
    out["scan_fixed"] = fixed["initial_scan"]
    out["fast_fixed"] = fixed["fast_loop"]

    # ---- estimates ---------------------------------------------------------------------------
    scan = stage["initial_scan"]
    scan_dbl = dbl["initial_scan"]
    scan_empty = empty_by_stage["initial_scan"]
    scan_pure = (
        scan["cols_removed"] == scan_dbl + scan_empty
        and ent("initial_scan", "implied_equation") == 0
    )
    low1 = {
        "rows": ent("initial_scan", "singleton_row")
        + ent("initial_scan", "redundant_row")
        + scan_dbl,
        "cols": scan_dbl + scan_empty,
        "nnz": scan["nnz_removed"]
        if scan_pure
        else max(
            0,
            min(
                scan["nnz_removed"],
                2 * scan_dbl + ent("initial_scan", "singleton_row"),
            ),
        ),
    }
    high1 = {
        "rows": rows["singleton_row"]
        + rows["redundant_row"]
        + rows["doubleton_eq"]
        + rows["parallel_row"],
        "cols": cols["doubleton_eq"]
        + cols["fix_empty_col"]
        + cols["fix_row_col_presolve"]
        + cols["fix_enumeration_stage"],
        "nnz": cells["initial_scan"]
        + cells["fast_loop"]
        + (cells["parallel_rows_cols"] if par_class in ("S1", "S1|S3") else 0)
        + (cells["sparsify"] if out["sparsify_class"] == "S4|S1" else 0)
        + max(0, cells["enum_other"]),
    }
    s2 = {
        "rows": rows["clique_merging"] + rows["probing_lifting"],
        "cols": cols["fix_probing_stage"]
        + cols["subst_clique"]
        + cols["subst_implication"],
        "nnz": cells["probing"] + cells["enum_clique_merging"],
    }
    out_part = {
        "rows": rows["vb_aggregation"] + rows["fourier_motzkin"],
        "cols": cols["fourier_motzkin"],
        "nnz": cells["vb_aggregation"] + cells["fourier_motzkin"],
    }
    s4_nnz_low = cells["sparsify"] + cells["strengthen_inequalities"]
    s4_nnz_high = (
        cells["sparsify"] if out["sparsify_class"] == "S4" else 0
    ) + cells["strengthen_inequalities"]
    low123 = {
        "rows": removed["rows"] - out_part["rows"],
        "cols": removed["cols"]
        - out_part["cols"]
        - cols["fix_enumeration_stage"],
        "nnz": removed["nnz"]
        - out_part["nnz"]
        - s4_nnz_low
        - max(0, cells["enum_other"]),
    }
    high123 = {
        "rows": removed["rows"] - out_part["rows"],
        "cols": removed["cols"] - out_part["cols"],
        "nnz": removed["nnz"] - out_part["nnz"] - s4_nnz_high,
    }
    out["parts"] = {
        "low1": low1,
        "high1": high1,
        "s2": s2,
        "low123": low123,
        "high123": high123,
    }
    out["checks"] = checks
    return out


def estimates(res, off):
    """Estimates of one instance; `off` is the analysis of the run with probing off, or None."""
    parts = res["parts"]
    est = {}
    for key in ("rows", "cols", "nnz"):
        est[key] = {
            "low1": parts["low1"][key],
            "high1_on": parts["high1"][key],
            "high1": parts["high1"][key]
            if off is None
            else off["parts"]["high1"][key],
            "noprobing": None if off is None else off["removed"][key],
            "low12": parts["low1"][key] + parts["s2"][key],
            "high12": parts["high1"][key] + parts["s2"][key],
            "low123": parts["low123"][key],
            "high123": parts["high123"][key],
        }
    return est


def top_share(parts, whole):
    """Largest positive part and its share of the net removal."""
    best = max(parts, key=lambda k: parts[k])
    if whole <= 0 or parts[best] <= 0:
        return "-", None
    return best, 100.0 * parts[best] / whole


def main():
    testbed = load_json(TESTBED)
    dev = [entry for entry in testbed["main"] if entry["where"] == "dev"]
    diag_summary = {
        inst["name"]: inst
        for inst in load_json(os.path.join(STAGE0, "diag", "summary.json"))[
            "instances"
        ]
    }
    stock_summary = {
        inst["name"]: inst
        for inst in load_json(os.path.join(STAGE0, "stock", "summary.json"))[
            "instances"
        ]
    }
    noprobing_path = os.path.join(STAGE0, "diag_noprobing", "summary.json")
    noprobing = {}
    if os.path.exists(noprobing_path):
        noprobing = {
            inst["name"]: inst
            for inst in load_json(noprobing_path)["instances"]
        }

    restricted = {}
    for variant, _ in RESTRICTED_RUNS:
        path = os.path.join(STAGE0, variant, "summary.json")
        if not os.path.exists(path):
            continue
        summary = load_json(path)
        restricted[variant] = {
            "off": summary.get("env", {}).get("HIGHS_DIAG_OFF", ""),
            "inst": {
                inst["name"]: inst
                for inst in summary["instances"]
                if inst.get("diag") and inst["diag"].get("total")
            },
        }

    missing = [
        entry["name"]
        for entry in dev
        if entry["name"] not in diag_summary
        or not diag_summary[entry["name"]].get("diag")
    ]
    results = []
    for entry in dev:
        name = entry["name"]
        if name in missing:
            continue
        res = analyse(diag_summary[name])
        np_inst = noprobing.get(name)
        res["off"] = None
        if (
            np_inst is not None
            and np_inst.get("diag")
            and np_inst.get("rows_after") is not None
        ):
            res["off"] = analyse(np_inst)
            if res["off"]["before"] != res["before"]:
                res["checks"].append(
                    "the run with probing off starts from a different model"
                )
            if res["off"]["checks"]:
                res["checks"].append(
                    "run with probing off: " + "; ".join(res["off"]["checks"])
                )
        res["est"] = estimates(res, res["off"])
        stock = stock_summary.get(name)
        if stock is None:
            res["checks"].append("no stock result")
        elif (
            stock["rows_after"],
            stock["cols_after"],
            stock["nnz_after"],
        ) != res["after"]:
            res["checks"].append(
                "stock and patched HiGHS give different presolved sizes"
            )
        res["log"] = log_counts(
            os.path.join(STAGE0, "diag", name + ".json.log")
        )
        stats_path = os.path.join(STAGE0, "diag", name + ".json")
        res["stats"] = (
            load_json(stats_path) if os.path.exists(stats_path) else None
        )
        results.append(res)

    print("# Stage 0: reductions of HiGHS presolve on the dev-size testbed")
    print()
    print(
        f"Instances: {len(results)} of {len(dev)} dev-size instances of the main testbed have a patched-HiGHS result."
    )
    if missing:
        print("Missing: " + ", ".join(missing))
    with_off = sum(1 for r in results if r["off"] is not None)
    print(
        f"Run with probing off (HIGHS_DIAG_OFF=probing): {with_off} instances; the S1 upper estimate is taken from it."
    )
    print()

    # ------------------------------------------------------------------ 1. sizes
    print("## 1. Sizes before and after, and the removal by DIAG stage")
    print()
    rows = []
    for r in results:
        b, a, rem = r["before"], r["after"], r["removed"]
        rows.append(
            [
                r["name"],
                r["band"],
                r["status"],
                f"{b[0]} -> {a[0]}",
                fmt_pct(pct(rem["rows"], b[0])),
                f"{b[1]} -> {a[1]}",
                fmt_pct(pct(rem["cols"], b[1])),
                f"{b[2]} -> {a[2]}",
                fmt_pct(pct(rem["nnz"], b[2])),
            ]
        )
    table(
        [
            "instance",
            "band",
            "status",
            "rows",
            "rows removed %",
            "columns",
            "columns removed %",
            "nonzeros",
            "nonzeros removed %",
        ],
        rows,
    )

    bad = [(r["name"], r["checks"]) for r in results if r["checks"]]
    print(
        "Checks per instance: the stage lines add up to DIAG total in rows, columns and nonzeros; the tool's sizes"
    )
    print(
        "equal DIAG total; stock and patched HiGHS give the same presolved sizes; the DIAG reductions lines add up to"
    )
    print(
        "stage=all; doubleton_equation = doubleton removed + substitutions applied + simple_probing + dual_fixing;"
    )
    print(
        "the DIAG parallel and aggregator counters equal the matching postsolve entries; every row and column a"
    )
    print(
        "stage removes is explained by its entries, clique merging and variable-bound aggregation (rows without an"
    )
    print(
        "entry only where simple probing ran; Fourier-Motzkin excepted). The run with probing off is checked the"
    )
    print(
        "same way (without the comparison with stock HiGHS) and must start from the same model."
    )
    if bad:
        print("Instances that fail a check:")
        for name, messages in bad:
            print(f"- {name}: " + "; ".join(messages))
    else:
        print(f"All checks hold on all {len(results)} instances.")
    print()

    live = [
        s
        for s in STAGES
        if any(
            r["stage"][s]["rows_removed"]
            or r["stage"][s]["cols_removed"]
            or r["stage"][s]["nnz_removed"]
            for r in results
        )
    ]
    for field, title in (
        ("rows_removed", "Rows"),
        ("cols_removed", "Columns"),
        ("nnz_removed", "Nonzeros"),
    ):
        print(
            f"### {title} removed by stage"
            + (
                " (negative = the stage added nonzeros)"
                if field == "nnz_removed"
                else ""
            )
        )
        print()
        rows = []
        for r in results:
            rows.append(
                [r["name"]]
                + [r["stage"][s][field] for s in live]
                + [sum(r["stage"][s][field] for s in STAGES)]
            )
        table(["instance"] + live + ["total"], rows)
    dead = [s for s in STAGES if s not in live]
    print(
        "Stages that remove nothing on any instance: " + ", ".join(dead) + "."
    )
    print()

    print("### Nonzeros by stage, % of the net nonzeros removed")
    print()
    print(
        "`enumeration` is split with the DIAG cliquemerge_call and varbound_aggregation lines; `probing` is shown"
    )
    print(
        "without its variable-bound aggregation, which is in `vb_aggregation`."
    )
    print()
    rows = []
    for r in results:
        z = r["removed"]["nnz"]
        rows.append(
            [r["name"], z]
            + [
                fmt_pct(pct(r["cells"][c], z)) if r["cells"][c] else "0"
                for c in NNZ_CELLS[:-1]
            ]
            + [r["par_class"], r["sparsify_class"]]
        )
    table(
        ["instance", "net removed"]
        + NNZ_CELLS[:-1]
        + ["parallel cell is", "sparsify cell is"],
        rows,
    )

    # ------------------------------------------------------------------ 2. postsolve entries
    print("## 2. Postsolve entries by type (DIAG reductions, stage=all)")
    print()
    live_types = [t for t in TYPES if any(r["types"][t] for r in results)]
    rows = [[r["name"]] + [r["types"][t] for t in live_types] for r in results]
    table(["instance"] + live_types, rows)
    dead_types = [t for t in TYPES if t not in live_types]
    print(
        "Types with no entry on any instance: "
        + (", ".join(dead_types) if dead_types else "none")
        + "."
    )
    print()

    print("### Split of `doubleton_equation` (every substitution y = a + b x)")
    print()
    rows = []
    for r in results:
        d, s = r["dbl"], r["sub"]
        if not r["types"]["doubleton_equation"]:
            continue
        rows.append(
            [
                r["name"],
                r["types"]["doubleton_equation"],
                d["removed"],
                d["initial_scan"],
                d["fast_loop"],
                d["removed"] - d["initial_scan"] - d["fast_loop"],
                s["from_cliques"],
                s["from_implications"],
                s["in_enumeration"],
                s["in_probing"],
                s["simple_probing"],
                s["dual_fixing"],
            ]
        )
    table(
        [
            "instance",
            "entries",
            "doubletonEq (S1)",
            "of which initial scan",
            "fast loop",
            "inside other stages",
            "clique subst (S2)",
            "implication subst (S2)",
            "applied in enumeration",
            "applied in probing",
            "simple probing (S3)",
            "dual-fixing subst (S3)",
        ],
        rows,
    )
    print("Instances not listed have no `doubleton_equation` entry.")
    print()

    print("### `fixed_col` by stage")
    print()
    fix_stages = [
        s for s in STAGES if any(r["fixed_by_stage"][s] for r in results)
    ]
    rows = []
    for r in results:
        if not r["types"]["fixed_col"]:
            continue
        rows.append(
            [r["name"], r["types"]["fixed_col"]]
            + [r["fixed_by_stage"][s] for s in fix_stages]
            + [
                r["par"]["cols_dominated"],
                r["cols"]["fix_empty_col"],
                r["types"]["forcing_row"],
                r["types"]["singleton_row"],
            ]
        )
    table(
        ["instance", "fixed_col"]
        + fix_stages
        + [
            "of parallel: cols_dominated",
            "certain empty columns",
            "forcing_row entries",
            "singleton_row entries",
        ],
        rows,
    )
    print(
        "`certain empty columns`: a stage that removes only fixed columns, no row and no nonzero."
    )
    print()

    print(
        "### Column rules: free-column substitution, aggregator, parallel rows and columns"
    )
    print()
    rows = []
    for r in results:
        a, p = r["agg"], r["par"]
        if not (
            r["types"]["free_col_substitution"]
            or a["candidates"]
            or p["rows_removed"]
            or p["cols_removed"]
        ):
            continue
        rows.append(
            [
                r["name"],
                r["types"]["free_col_substitution"],
                a["substituted"],
                r["cols"]["free_col_singleton"],
                a["candidates"],
                a["examined"],
                a["size2"],
                a["rejected_fillin"],
                a["rejected_integer"],
                a["stale"],
                a["stopped_by_failures"],
                p["duplicate_rows"],
                p["row_additions"],
                p["cols_merged"],
                p["cols_dominated"],
            ]
        )
    table(
        [
            "instance",
            "free_col_substitution",
            "by aggregator",
            "by singleton column",
            "aggregator candidates",
            "examined",
            "size2",
            "rejected fill-in",
            "rejected integer",
            "stale",
            "calls stopped by failures",
            "duplicate rows",
            "nearly parallel rows",
            "columns merged",
            "columns dominated",
        ],
        rows,
    )

    # ------------------------------------------------------------------ 3. mapping
    print(
        "## 3. Stages and reduction types against the plan (DESIGN.md section 5)"
    )
    print()
    table(
        [
            "DIAG stage",
            "plan stage",
            "what runs in it (HPresolve.cpp and the HPresolve*.cpp files of commit 283bfc859)",
        ],
        [list(row) for row in STAGE_MAP],
    )
    table(
        ["postsolve type", "plan stage", "pushed by"],
        [list(row) for row in TYPE_MAP],
    )

    print("### Rows by the rule that removed them")
    print()
    live_rules = [
        name for name, _ in ROW_RULES if any(r["rows"][name] for r in results)
    ]
    plan = dict(ROW_RULES)
    rows = [
        [r["name"], r["removed"]["rows"]]
        + [r["rows"][name] for name in live_rules]
        for r in results
    ]
    table(
        ["instance", "rows removed"]
        + [f"{name} ({plan[name]})" for name in live_rules],
        rows,
    )
    print(
        "`redundant_row` includes one entry per forcing row. `simple_probing` = rows of the scan and fast loop with no"
    )
    print(
        "postsolve entry. Row rules with no removal on any instance: "
        + (", ".join(n for n, _ in ROW_RULES if n not in live_rules) or "none")
        + "."
    )
    print()

    print("### Columns by the rule that removed them")
    print()
    live_rules = [
        name for name, _ in COL_RULES if any(r["cols"][name] for r in results)
    ]
    plan = dict(COL_RULES)
    rows = [
        [r["name"], r["removed"]["cols"]]
        + [r["cols"][name] for name in live_rules]
        for r in results
    ]
    table(
        ["instance", "columns removed"]
        + [f"{name} ({plan[name]})" for name in live_rules],
        rows,
    )
    print(
        "Column rules with no removal on any instance: "
        + (", ".join(n for n, _ in COL_RULES if n not in live_rules) or "none")
        + "."
    )
    print()

    print("### S3 rules visible inside the initial scan and the fast loop")
    print()
    rows = []
    for r in results:
        ent = r["ent"]
        rows.append(
            [
                r["name"],
                r["scan_fixed"],
                r["scan_s3"],
                r["fast_fixed"],
                r["fast_s3"],
                r["sub"]["simple_probing"],
                r["sub"]["dual_fixing"],
                ent("initial_scan", "free_col_substitution")
                + ent("fast_loop", "free_col_substitution"),
                ent("initial_scan", "zero_obj_singleton_col")
                + ent("fast_loop", "zero_obj_singleton_col"),
                ent("initial_scan", "forcing_column")
                + ent("fast_loop", "forcing_column"),
                ent("initial_scan", "implied_equation")
                + ent("fast_loop", "implied_equation"),
                ent("initial_scan", "linear_transform")
                + ent("fast_loop", "linear_transform"),
            ]
        )
    table(
        [
            "instance",
            "scan fixed_col (rule unknown)",
            "scan S3 entries",
            "fast loop fixed_col (rule unknown)",
            "fast loop S3 entries",
            "simple probing subst (all stages)",
            "dual-fixing subst (all stages)",
            "singleton-column subst",
            "zero-cost singleton",
            "forcing column",
            "implied equation",
            "linear transform",
        ],
        rows,
    )
    print(
        "`S3 entries` = free_col_substitution + forcing_column + zero_obj_singleton_col + implied_equation +"
    )
    print(
        "(doubleton_equation entries of the stage - doubleton equations of the stage)."
    )
    print()

    print("### Rule families over the testbed")
    print()
    print(
        "Share = the family's part of the instance's net removal of that measure. Nonzeros are per stage cell."
    )
    print()
    rows = []
    for measure, key, families in (
        ("rows", "rows", ROW_RULES),
        ("columns", "cols", COL_RULES),
        ("nonzeros", "cells", [(c, "") for c in NNZ_CELLS[:-1]]),
    ):
        removed_key = "nnz" if key == "cells" else key
        grand = sum(r["removed"][removed_key] for r in results)
        for family, plan_stage in families:
            total_family = sum(r[key][family] for r in results)
            if not any(r[key][family] for r in results):
                continue
            shares = [
                (r["name"], pct(r[key][family], r["removed"][removed_key]))
                for r in results
                if r[key][family]
            ]
            big = [
                name
                for name, share in shares
                if share is not None and share >= 50.0
            ]
            rows.append(
                [
                    measure,
                    family + (f" ({plan_stage})" if plan_stage else ""),
                    total_family,
                    fmt_pct(pct(total_family, grand)),
                    len(shares),
                    sum(
                        1
                        for _, share in shares
                        if share is not None and share >= 10.0
                    ),
                    len(big),
                    ", ".join(big) if big else "-",
                ]
            )
    table(
        [
            "measure",
            "family",
            "sum over the testbed",
            "% of the summed removal",
            "instances where it is nonzero",
            "instances at 10% or more",
            "instances at 50% or more",
            "which",
        ],
        rows,
    )

    mixed = [r for r in results if r["par_class"] == "S1|S3"]
    if mixed:
        print("### Parallel-stage cells that remove rows and columns")
        print()
        print(
            "The bound assumes that no column is longer than the longest column of the original model when the stage"
        )
        print(
            "runs (true when nothing before it adds entries to a column; substitutions can)."
        )
        print()
        rows = []
        for r in mixed:
            par = r["par"]
            cell = r["stage"]["parallel_rows_cols"]
            longest = (
                r["stats"]["original"]["col_length"]["max"]
                if r["stats"]
                else None
            )
            bound = (
                None
                if longest is None
                else (par["cols_merged"] + par["cols_dominated"]) * longest
            )
            binding = bound is not None and bound < cell["nnz_removed"]
            rows.append(
                [
                    r["name"],
                    par["duplicate_rows"],
                    par["row_additions"],
                    par["cols_merged"],
                    par["cols_dominated"],
                    cell["nnz_removed"],
                    longest,
                    bound,
                    fmt_pct(pct(bound, cell["nnz_removed"]))
                    if binding
                    else "no bound (above the cell)",
                ]
            )
        table(
            [
                "instance",
                "duplicate rows",
                "nearly parallel rows",
                "columns merged",
                "columns dominated",
                "nonzeros of the cell",
                "longest original column",
                "most nonzeros the removed columns can hold",
                "% of the cell",
            ],
            rows,
        )

    # ------------------------------------------------------------------ 4. estimates
    print(
        "## 4. Share of HiGHS's removal that the plan stages reach: lower and upper estimates"
    )
    print()
    print(
        "Definitions are in the module docstring. All values are % of HiGHS's net removal of that measure in the"
    )
    print(
        "normal run. `all rules, probing off` is measured: the same HiGHS with HIGHS_DIAG_OFF=probing. `S1 high` is"
    )
    print(
        "the S1-rule removal of that run; `S1 high in the normal run` is the same count where probing ran."
    )
    print()
    titles = {"rows": "Rows", "cols": "Columns", "nnz": "Nonzeros"}
    for key in ("rows", "cols", "nnz"):
        print(f"### {titles[key]}")
        print()
        rows = []
        for r in results:
            whole = r["removed"][key]
            e = r["est"][key]
            rows.append(
                [
                    r["name"],
                    whole,
                    fmt_pct(pct(e["low1"], whole)),
                    fmt_pct(pct(e["high1"], whole)),
                    fmt_pct(pct(e["high1_on"], whole)),
                    fmt_pct(pct(e["noprobing"], whole)),
                    fmt_pct(pct(e["low12"], whole)),
                    fmt_pct(pct(e["high12"], whole)),
                    fmt_pct(pct(e["low123"], whole)),
                    fmt_pct(pct(e["high123"], whole)),
                ]
            )
        table(
            [
                "instance",
                "HiGHS removed",
                "S1 low",
                "S1 high",
                "S1 high in the normal run",
                "all rules, probing off (measured)",
                "S1+S2 low",
                "S1+S2 high",
                "S1+S2+S3 low",
                "S1+S2+S3 high",
            ],
            rows,
        )

    print("### Rows that S1 removes by their shape in the original model")
    print()
    print(
        "From the structure statistics of the original model: empty rows and singleton rows are always removed by"
    )
    print(
        "S1; a doubleton equation is removed unless both columns are integer and the coefficient ratio is not"
    )
    print(
        "integral. The last column is what HiGHS's initial scan removed with S1 rules (the `S1 low` of rows)."
    )
    print()
    rows = []
    for r in results:
        if not r["stats"]:
            continue
        classes = r["stats"]["original"]["row_classes"]
        shape = (
            classes["empty"]
            + classes["singleton"]
            + classes["doubleton_equation"]
        )
        if shape == 0:
            continue
        rows.append(
            [
                r["name"],
                classes["empty"],
                classes["singleton"],
                classes["doubleton_equation"],
                fmt_pct(
                    pct(
                        classes["empty"] + classes["singleton"],
                        r["removed"]["rows"],
                    )
                ),
                fmt_pct(pct(shape, r["removed"]["rows"])),
                fmt_pct(pct(r["est"]["rows"]["low1"], r["removed"]["rows"])),
            ]
        )
    table(
        [
            "instance",
            "empty rows",
            "singleton rows",
            "doubleton equations",
            "empty + singleton, % of HiGHS's rows removed",
            "all three, %",
            "S1 rules in HiGHS's initial scan, %",
        ],
        rows,
    )
    print("Instances not listed have no such row.")
    print()

    changed = [
        r
        for r in results
        if r["off"] is not None and r["off"]["removed"] != r["removed"]
    ]
    if any(r["off"] is not None for r in results):
        print("### What changes when probing is switched off (measured)")
        print()
        print(
            "Normal run -> run with HIGHS_DIAG_OFF=probing. Instances with the same rows, columns and nonzeros removed in"
        )
        print("both runs are not listed.")
        print()
        rows = []
        for r in changed:
            off = r["off"]
            on_stage, off_stage = r["stage"], off["stage"]
            rows.append(
                [
                    r["name"],
                    f"{r['removed']['rows']} -> {off['removed']['rows']}",
                    f"{r['removed']['cols']} -> {off['removed']['cols']}",
                    f"{r['removed']['nnz']} -> {off['removed']['nnz']}",
                    f"{on_stage['probing']['rows_removed']} / {on_stage['probing']['cols_removed']} / {on_stage['probing']['nnz_removed']}",
                    f"{r['parts']['high1']['rows']} -> {off['parts']['high1']['rows']}",
                    f"{on_stage['fast_loop']['nnz_removed']} -> {off_stage['fast_loop']['nnz_removed']}",
                    f"{on_stage['sparsify']['nnz_removed']} -> {off_stage['sparsify']['nnz_removed']}",
                    f"{on_stage['sparsify']['calls']} -> {off_stage['sparsify']['calls']}",
                    f"{on_stage['parallel_rows_cols']['nnz_removed']} -> {off_stage['parallel_rows_cols']['nnz_removed']}",
                    f"{on_stage['enumeration']['nnz_removed']} -> {off_stage['enumeration']['nnz_removed']}",
                ]
            )
        table(
            [
                "instance",
                "rows removed",
                "columns removed",
                "nonzeros removed",
                "probing stage itself: rows / columns / nonzeros",
                "rows removed by S1 rules",
                "fast loop nonzeros",
                "sparsify nonzeros",
                "sparsify calls",
                "parallel stage nonzeros",
                "enumeration nonzeros",
            ],
            rows,
        )
        same = [
            r["name"]
            for r in results
            if r["off"] is not None and r["off"]["removed"] == r["removed"]
        ]
        print(
            f"Same result in both runs ({len(same)}): " + ", ".join(same) + "."
        )
        print()

        print(
            "### Where the loss with probing off sits (normal run minus probing-off run, summed)"
        )
        print()
        rows = []
        with_off_run = [r for r in results if r["off"] is not None]
        for field, title in (
            ("rows_removed", "rows"),
            ("cols_removed", "columns"),
            ("nnz_removed", "nonzeros"),
        ):
            diff = {
                s: sum(
                    r["stage"][s][field] - r["off"]["stage"][s][field]
                    for r in with_off_run
                )
                for s in STAGES
            }
            whole = sum(diff.values())
            rows.append(
                [title, whole]
                + [
                    f"{diff[s]} ({fmt_pct(pct(diff[s], whole))}%)"
                    for s in (
                        "probing",
                        "fast_loop",
                        "sparsify",
                        "parallel_rows_cols",
                        "aggregator",
                        "enumeration",
                    )
                ]
                + [
                    whole
                    - sum(
                        diff[s]
                        for s in (
                            "probing",
                            "fast_loop",
                            "sparsify",
                            "parallel_rows_cols",
                            "aggregator",
                            "enumeration",
                        )
                    )
                ]
            )
        table(
            [
                "measure",
                "lost without probing",
                "probing stage itself",
                "fast loop",
                "sparsify",
                "parallel stage",
                "aggregator",
                "enumeration",
                "other stages",
            ],
            rows,
        )

    if restricted:
        keys = ("rows", "cols", "nnz")
        print("### Measured restricted runs (HiGHS with stages switched off)")
        print()
        print(
            "Sizes are those of the DIAG total line. Shares are % of what HiGHS removes in the normal run. The S3 rules"
        )
        print(
            "inside row and column presolve stay on in every run (see the module docstring). The instance columns leave"
        )
        print(
            "out rmatr200-p5 (nothing removed) and supportcase19 (one nonzero removed)."
        )
        print()
        mismatches = []
        rows = []
        for variant, label in RESTRICTED_RUNS:
            if variant not in restricted:
                continue
            run = restricted[variant]
            used = [r for r in results if r["name"] in run["inst"]]
            removed = {
                r["name"]: diag_removed(run["inst"][r["name"]]) for r in used
            }
            for r in used:
                inst = run["inst"][r["name"]]
                tool = {
                    "rows": inst["rows"] - inst["rows_after"],
                    "cols": inst["cols"] - inst["cols_after"],
                    "nnz": inst["nnz"] - inst["nnz_after"],
                }
                if tool != removed[r["name"]]:
                    mismatches.append(
                        f"{variant} {r['name']}: status {inst['status']}, returned model removes "
                        f"{tool['rows']} / {tool['cols']} / {tool['nnz']}, DIAG total "
                        f"{removed[r['name']]['rows']} / {removed[r['name']]['cols']} / {removed[r['name']]['nnz']}"
                    )
            line = [
                variant,
                label,
                len(used),
                sum(1 for r in used if removed[r["name"]] != r["removed"]),
            ]
            for key in keys:
                whole = sum(r["removed"][key] for r in used)
                line.append(
                    fmt_pct(
                        pct(sum(removed[r["name"]][key] for r in used), whole)
                    )
                )
            # supportcase19 loses one nonzero in all, so its share means nothing
            shares = [
                (
                    r["name"],
                    100.0 * removed[r["name"]]["nnz"] / r["removed"]["nnz"],
                )
                for r in used
                if r["removed"]["nnz"] > 1
            ]
            line.append(
                fmt_pct(statistics.median(min(100.0, s) for _, s in shares))
                if shares
                else "n/a"
            )
            line.append(sum(1 for _, s in shares if s >= 90.0))
            line.append(
                ", ".join(
                    f"{name} {share:.1f}"
                    for name, share in shares
                    if share < 50.0
                )
                or "-"
            )
            line.append(
                ", ".join(
                    f"{name} {share:.1f}"
                    for name, share in shares
                    if share > 105.0
                )
                or "-"
            )
            rows.append(line)
        table(
            [
                "run",
                "what is on",
                "instances",
                "results that differ from the normal run",
                "rows %",
                "columns %",
                "nonzeros %",
                "median nonzero share of the instances",
                "instances at 90% of the nonzeros or more",
                "instances below 50% of the nonzeros",
                "instances above 105% of the nonzeros",
            ],
            rows,
        )
        if mismatches:
            print(
                "Returned model and DIAG total differ (HiGHS reports Not reduced when no postsolve entry was pushed):"
            )
            for text in mismatches:
                print("- " + text)
            print()

        stage1 = restricted.get("diag_stage1")
        stage12 = restricted.get("diag_stage12")
        if stage1 is not None:
            print("### S1 estimates against the restricted runs")
            print()
            print(
                "`S1 high` and `S1+S2 high` are the estimates of the tables above. `S1-rule rows` = singleton, redundant,"
            )
            print(
                "doubleton-equation and parallel rows of the restricted run. An estimate above the measured run counts"
            )
            print("removals that need a stage the run has switched off.")
            print()
            rows = []
            over = []
            s1_rule_rows = 0
            entries = defaultdict(int)
            for r in results:
                inst = stage1["inst"].get(r["name"])
                if inst is None:
                    continue
                sub = analyse(inst)
                rem1 = diag_removed(inst)
                rem12 = (
                    diag_removed(stage12["inst"][r["name"]])
                    if stage12 is not None and r["name"] in stage12["inst"]
                    else None
                )
                s1_rule_rows += sub["parts"]["high1"]["rows"]
                for kind in (
                    "fixed_col",
                    "zero_obj_singleton_col",
                    "forcing_column",
                    "implied_equation",
                    "duplicate_row",
                    "equality_row_addition",
                ):
                    entries[kind] += sub["types"][kind]
                entries["free_col_substitution"] += sub["cols"][
                    "free_col_singleton"
                ]
                entries["doubleton_eq"] += sub["dbl"]["removed"]
                entries["simple_probing"] += sub["sub"]["simple_probing"]
                entries["dual_fixing"] += sub["sub"]["dual_fixing"]
                entries["simple_probing_rows"] += sub["rows"]["simple_probing"]
                whole = r["removed"]
                est = r["est"]
                rows.append(
                    [
                        r["name"],
                        whole["rows"],
                        fmt_pct(pct(est["rows"]["high1"], whole["rows"])),
                        fmt_pct(pct(rem1["rows"], whole["rows"])),
                        fmt_pct(
                            pct(sub["parts"]["high1"]["rows"], whole["rows"])
                        ),
                        fmt_pct(pct(est["rows"]["high12"], whole["rows"])),
                        fmt_pct(pct(rem12["rows"], whole["rows"]))
                        if rem12
                        else "n/a",
                        whole["cols"],
                        fmt_pct(pct(rem1["cols"], whole["cols"])),
                        fmt_pct(pct(rem12["cols"], whole["cols"]))
                        if rem12
                        else "n/a",
                        whole["nnz"],
                        fmt_pct(pct(est["nnz"]["high1"], whole["nnz"])),
                        fmt_pct(pct(rem1["nnz"], whole["nnz"])),
                        fmt_pct(pct(est["nnz"]["high12"], whole["nnz"])),
                        fmt_pct(pct(rem12["nnz"], whole["nnz"]))
                        if rem12
                        else "n/a",
                    ]
                )
                if (
                    whole["nnz"] > 1
                    and est["nnz"]["high1"] is not None
                    and 100.0
                    * (est["nnz"]["high1"] - rem1["nnz"])
                    / whole["nnz"]
                    > 5.0
                ):
                    over.append(
                        f"{r['name']} {fmt_pct(pct(est['nnz']['high1'], whole['nnz']))} against {fmt_pct(pct(rem1['nnz'], whole['nnz']))}"
                    )
            table(
                [
                    "instance",
                    "rows removed by HiGHS",
                    "S1 high (estimate)",
                    "diag_stage1 (measured)",
                    "of which S1-rule rows",
                    "S1+S2 high (estimate)",
                    "diag_stage12 (measured)",
                    "columns removed by HiGHS",
                    "diag_stage1 (measured)",
                    "diag_stage12 (measured)",
                    "nonzeros removed by HiGHS",
                    "S1 high (estimate)",
                    "diag_stage1 (measured)",
                    "S1+S2 high (estimate)",
                    "diag_stage12 (measured)",
                ],
                rows,
            )
            print(
                "Nonzeros, S1 high more than 5 points above diag_stage1: "
                + ("; ".join(over) if over else "none")
                + "."
            )
            grand_rows = sum(r["removed"]["rows"] for r in results)
            grand_cols = sum(r["removed"]["cols"] for r in results)
            print(
                f"diag_stage1 over the testbed: S1-rule rows {s1_rule_rows} ({fmt_pct(pct(s1_rule_rows, grand_rows))}% of HiGHS's rows); "
                f"duplicate rows {entries['duplicate_row']}, nearly parallel rows {entries['equality_row_addition']}; "
                f"columns: fixed_col {entries['fixed_col']} ({fmt_pct(pct(entries['fixed_col'], grand_cols))}% of HiGHS's columns), "
                f"doubleton equations {entries['doubleton_eq']}; S3 rules that stay on inside row and column presolve: "
                f"dual-fixing substitution {entries['dual_fixing']}, zero-cost singleton {entries['zero_obj_singleton_col']}, "
                f"simple probing {entries['simple_probing']} columns and {entries['simple_probing_rows']} rows, "
                f"singleton-column substitution {entries['free_col_substitution']}, forcing column {entries['forcing_column']}, "
                f"implied equation {entries['implied_equation']}."
            )
            print()

    print("### Aggregates over the instances with a positive removal")
    print()
    rows = []
    for key in ("rows", "cols", "nnz"):
        used = [r for r in results if r["removed"][key] > 0]
        whole = sum(r["removed"][key] for r in used)
        for field, label in (
            ("low1", "S1 low"),
            ("high1", "S1 high"),
            ("high1_on", "S1 high in the normal run"),
            ("noprobing", "all rules, probing off (measured)"),
            ("low12", "S1+S2 low"),
            ("high12", "S1+S2 high"),
            ("low123", "S1+S2+S3 low"),
            ("high123", "S1+S2+S3 high"),
        ):
            values = [r["est"][key][field] for r in used]
            if any(v is None for v in values):
                continue
            shares = [
                min(100.0, 100.0 * v / r["removed"][key])
                for v, r in zip(values, used)
            ]
            rows.append(
                [
                    titles[key],
                    label,
                    len(used),
                    fmt_pct(100.0 * sum(values) / whole),
                    fmt_pct(statistics.median(shares)),
                    fmt_pct(statistics.mean(shares)),
                    sum(1 for s in shares if s >= 90.0),
                    sum(1 for s in shares if s < 50.0),
                ]
            )
    table(
        [
            "measure",
            "estimate",
            "instances",
            "% of the summed removal",
            "median of the instance shares",
            "mean of the instance shares",
            "instances at 90% or more",
            "instances below 50%",
        ],
        rows,
    )
    print(
        "Instance shares are cut at 100% for the median, the mean and the counts. Largest parts of the summed removal:"
    )
    for key in ("rows", "cols", "nnz"):
        whole = sum(r["removed"][key] for r in results)
        top = sorted(results, key=lambda r: -r["removed"][key])[:4]
        print(
            f"- {titles[key].lower()}: "
            + ", ".join(
                f"{r['name']} {fmt_pct(pct(r['removed'][key], whole))}%"
                for r in top
            )
        )
    print()

    print("### Largest contributor per instance")
    print()
    rows = []
    for r in results:
        rem = r["removed"]
        row_rule, row_share = top_share(r["rows"], rem["rows"])
        col_rule, col_share = top_share(r["cols"], rem["cols"])
        cell, cell_share = top_share(r["cells"], rem["nnz"])
        rows.append(
            [
                r["name"],
                row_rule,
                fmt_pct(row_share, 0),
                col_rule,
                fmt_pct(col_share, 0),
                cell,
                fmt_pct(cell_share, 0),
            ]
        )
    table(
        [
            "instance",
            "rows: rule",
            "% of rows",
            "columns: rule",
            "% of columns",
            "nonzeros: stage",
            "% of nonzeros",
        ],
        rows,
    )

    # ------------------------------------------------------------------ 5. how often rules fire
    print("## 5. How often stages, types and counters fire")
    print()
    n = len(results)
    rows = []
    for s in STAGES:
        called = sum(1 for r in results if r["stage"][s]["calls"] > 0)
        acts = [
            r
            for r in results
            if r["stage"][s]["rows_removed"]
            or r["stage"][s]["cols_removed"]
            or r["stage"][s]["nnz_removed"]
        ]
        entries = sum(1 for r in results if any(r["ent"](s, t) for t in TYPES))
        rows.append(
            [
                s,
                called,
                len(acts),
                entries,
                sum(r["stage"][s]["rows_removed"] for r in results),
                sum(r["stage"][s]["cols_removed"] for r in results),
                sum(r["stage"][s]["nnz_removed"] for r in results),
            ]
        )
    table(
        [
            "stage",
            "instances where it is called",
            "instances where it changes the size",
            "instances where it pushes an entry",
            "rows removed (sum)",
            "columns removed (sum)",
            "nonzeros removed (sum)",
        ],
        rows,
    )

    rows = []
    for t in TYPES:
        rows.append(
            [
                t,
                sum(1 for r in results if r["types"][t]),
                sum(r["types"][t] for r in results),
            ]
        )
    table(
        [
            "postsolve type",
            f"instances with an entry (of {n})",
            "entries (sum)",
        ],
        rows,
    )

    def count(getter):
        values = [getter(r) for r in results]
        return sum(1 for v in values if v), sum(values)

    counters = [
        ("doubleton equations (doubletonEq)", lambda r: r["dbl"]["removed"]),
        ("clique substitutions", lambda r: r["sub"]["from_cliques"]),
        ("implication substitutions", lambda r: r["sub"]["from_implications"]),
        (
            "substitutions applied in the clique_substitutions stage",
            lambda r: r["sub"]["in_clique_substitutions"],
        ),
        (
            "substitutions applied in enumeration",
            lambda r: r["sub"]["in_enumeration"],
        ),
        ("simple probing substitutions", lambda r: r["sub"]["simple_probing"]),
        ("dual-fixing substitutions", lambda r: r["sub"]["dual_fixing"]),
        ("aggregator substitutions", lambda r: r["agg"]["substituted"]),
        (
            "aggregator substitutions with a row or column of size 2",
            lambda r: r["agg"]["size2"],
        ),
        (
            "singleton-column substitutions",
            lambda r: r["cols"]["free_col_singleton"],
        ),
        ("duplicate rows", lambda r: r["par"]["duplicate_rows"]),
        ("nearly parallel rows", lambda r: r["par"]["row_additions"]),
        ("parallel columns merged", lambda r: r["par"]["cols_merged"]),
        (
            "parallel columns fixed by dominance",
            lambda r: r["par"]["cols_dominated"],
        ),
        (
            "fixed_col in the pairwise dominated-columns stage",
            lambda r: r["fixed_by_stage"]["dominated_columns"],
        ),
        (
            "fixed_col in the probing stage",
            lambda r: r["fixed_by_stage"]["probing"],
        ),
        (
            "fixed_col in the enumeration stage",
            lambda r: r["fixed_by_stage"]["enumeration"],
        ),
        (
            "rows deleted by clique merging and the clique table",
            lambda r: r["rows"]["clique_merging"],
        ),
        (
            "rows changed by clique merging (extension)",
            lambda r: r["merge"]["rows_changed"],
        ),
        (
            "clique-extension nonzeros added by merging",
            lambda r: r["merge"]["nnz_added"],
        ),
        (
            "rows removed by lifting for probing",
            lambda r: r["rows"]["probing_lifting"],
        ),
        (
            "rows removed by variable-bound aggregation",
            lambda r: r["rows"]["vb_aggregation"],
        ),
        (
            "rows rewritten by variable-bound aggregation",
            lambda r: sum(v["rows_modified"] for v in r["vba"].values()),
        ),
        (
            "rows without an entry (simple probing)",
            lambda r: r["rows"]["simple_probing"],
        ),
        (
            "columns removed with no nonzero (empty columns, certain)",
            lambda r: r["cols"]["fix_empty_col"],
        ),
    ]
    rows = []
    for label, getter in counters:
        inst_count, total = count(getter)
        rows.append([label, inst_count, total])
    logs = [r for r in results if r["log"] is not None]
    if logs:
        for label, key in (
            (
                "coefficients changed by strengthenInequalities (log)",
                "strengthened_coefs",
            ),
            (
                "columns fixed by pairwise dominance (log)",
                "pairwise_dominated_fixed",
            ),
            (
                "bounds strengthened by pairwise dominance (log)",
                "pairwise_bounds_strengthened",
            ),
            ("Fourier-Motzkin report lines (log)", "fm_lines"),
        ):
            rows.append(
                [
                    label,
                    sum(1 for r in logs if r["log"][key]),
                    sum(r["log"][key] for r in logs),
                ]
            )
    table(["counter", f"instances where it is nonzero (of {n})", "sum"], rows)

    def names(pred):
        return ", ".join(r["name"] for r in results if pred(r)) or "none"

    print("Instances per rare rule:")
    print(
        "- Fourier-Motzkin removes something: "
        + names(
            lambda r: r["rows"]["fourier_motzkin"]
            or r["cols"]["fourier_motzkin"]
        )
    )
    print("- forcing column: " + names(lambda r: r["types"]["forcing_column"]))
    print(
        "- implied equation: "
        + names(lambda r: r["types"]["implied_equation"])
    )
    print(
        "- zero-cost singleton column: "
        + names(lambda r: r["types"]["zero_obj_singleton_col"])
    )
    print(
        "- simple probing on an equation: "
        + names(lambda r: r["sub"]["simple_probing"])
    )
    print(
        "- dual-fixing substitution: "
        + names(lambda r: r["sub"]["dual_fixing"])
    )
    print(
        "- pairwise dominated columns fix a column: "
        + names(lambda r: r["fixed_by_stage"]["dominated_columns"])
    )
    print(
        "- parallel columns (merge or dominance): "
        + names(
            lambda r: r["par"]["cols_merged"] or r["par"]["cols_dominated"]
        )
    )
    print(
        "- nearly parallel rows: " + names(lambda r: r["par"]["row_additions"])
    )
    print(
        "- variable-bound aggregation removes or rewrites a row: "
        + names(
            lambda r: sum(
                v["rows_removed"] + v["rows_modified"]
                for v in r["vba"].values()
            )
        )
    )
    print(
        "- clique merging deletes a row: "
        + names(lambda r: r["rows"]["clique_merging"])
    )
    print(
        "- strengthenInequalities changes a coefficient (log): "
        + names(lambda r: r["log"] and r["log"]["strengthened_coefs"])
    )
    print(
        "- strengthenInequalities removes a nonzero: "
        + names(lambda r: r["stage"]["strengthen_inequalities"]["nnz_removed"])
    )
    print(
        "- sparsify cancels (equality_row_additions): "
        + names(lambda r: r["types"]["equality_row_additions"])
    )
    print()

    # ------------------------------------------------------------------ 6. quality
    print(
        "## 6. Integer columns, implied integers and the shape of the reduced model"
    )
    print()
    rows = []
    for r in results:
        stats = r["stats"]
        orig = stats["original"] if stats else None
        pres = stats.get("presolved") if stats else None
        int_b, int_a = r["integer"]
        cols_b, cols_a = r["before"][1], r["after"][1]
        cont_b = cols_b - int_b
        cont_a = cols_a - int_a - r["implied_integer"]
        rows.append(
            [
                r["name"],
                f"{int_b} -> {int_a}",
                fmt_pct(pct(int_b - int_a, int_b)),
                f"{cont_b} -> {cont_a}",
                r["implied_integer"],
                fmt_pct(
                    pct(r["implied_integer"], cont_a + r["implied_integer"])
                ),
                r["types"]["linear_transform"],
                f"{orig['col_types']['binary']} -> {pres['col_types']['binary']}"
                if orig and pres
                else "n/a",
                f"{orig['col_types']['general_integer']} -> {pres['col_types']['general_integer']}"
                if orig and pres
                else "n/a",
            ]
        )
    table(
        [
            "instance",
            "integer columns",
            "integer removed %",
            "continuous columns (implied integers not counted after)",
            "implied integer after",
            "implied integer % of non-integer columns after",
            "linear_transform entries",
            "binary (tool)",
            "general integer (tool)",
        ],
        rows,
    )
    print(
        "`integer columns` and `implied integer after` are from DIAG total; `binary` and `general integer` from the"
    )
    print(
        "tool's statistics of the returned model (for status Not reduced that model is the original)."
    )
    print()

    rows = []
    for r in results:
        stats = r["stats"]
        if not stats or not stats.get("presolved"):
            continue
        orig, pres = stats["original"], stats["presolved"]
        scaled = orig.get("row_dynamism_highs_scaled", orig["row_dynamism"])
        added = r["merge"]["nnz_added"]
        fill = sum(
            -r["stage"][s]["nnz_removed"]
            for s in STAGES
            if r["stage"][s]["nnz_removed"] < 0
        )
        rows.append(
            [
                r["name"],
                f"{orig['row_length']['mean']:.2f} -> {pres['row_length']['mean']:.2f}",
                f"{orig['row_length']['max']} -> {pres['row_length']['max']}",
                f"{orig['col_length']['mean']:.2f} -> {pres['col_length']['mean']:.2f}",
                f"{scaled['p99']:.3g} -> {pres['row_dynamism']['p99']:.3g}",
                f"{scaled['max']:.3g} -> {pres['row_dynamism']['max']:.3g}",
                added,
                sum(v["vars_lifted"] for v in r["vba"].values()),
                fill,
                f"{orig['row_classes']['set_packing'] + orig['row_classes']['set_partitioning']} -> {pres['row_classes']['set_packing'] + pres['row_classes']['set_partitioning']}",
                f"{orig['row_sides']['equation']} -> {pres['row_sides']['equation']}",
            ]
        )
    table(
        [
            "instance",
            "mean row length",
            "longest row",
            "mean column length",
            "row dynamism p99 (HiGHS-scaled original -> presolved)",
            "row dynamism max",
            "clique-extension nonzeros added",
            "binaries lifted by variable-bound aggregation",
            "nonzeros added by stages with a net increase",
            "set packing + partitioning rows",
            "equations",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
