"""Wrong engines of the validity review of M1 (beyond check/make_mutants.py).

Usage: my_mutants.py <project dir> <out dir> [name ...]
Each is <out dir>/<name>/src with exact text replacements, each found exactly once.
"""

import shutil
import sys
from pathlib import Path

R = "rules_singleton_cols.cpp"
P = "postsolve.cpp"

MUTANTS = {
    # ---- R2 / R5: the free singleton column ----
    # The implied-bound test reads the side less the fixed terms with activities that still
    # hold those terms: the fixed terms are taken off twice (review item D.9 a).
    "v_r2_fixed_twice": [
        (R, "  if (!implied_free(c, row, s, side)) {", "  if (!implied_free(c, row, s, free_side)) {"),
    ],
    # Both ends of the range are computed with the wrong extreme of the rest.
    "v_r2_ends": [
        (R, "    if (!range_end(c, row, s.slot, side, s.a > 0, k)) { return false; }",
            "    if (!range_end(c, row, s.slot, side, s.a < 0, k)) { return false; }"),
        (R, "    if (!range_end(c, row, s.slot, side, s.a < 0, k)) { return false; }\n"
            "    if (!(k.bound - candidate_slack(k)",
            "    if (!range_end(c, row, s.slot, side, s.a > 0, k)) { return false; }\n"
            "    if (!(k.bound - candidate_slack(k)"),
    ],
    # Record and offset get the stored side, with the fixed terms in it.
    "v_r2_side_stored": [
        (R, "  return substitute(c, row, s, free_side, !equation, growth, largest_other);",
            "  return substitute(c, row, s, side, !equation, growth, largest_other);"),
    ],
    # The side of the row is not asked to be integral for an integer column.
    "v_r2_int_side": [
        ("integral.hpp", "  integral_form_t f;\n  integral_side(tol, a, side, f);\n  for (int64_t s = csr.begin(row)",
                         "  integral_form_t f;\n  (void)side;\n  (void)tol;\n  for (int64_t s = csr.begin(row)"),
    ],
    # No guard on the side less the fixed terms.
    "v_r2_free_guard": [
        (R, "  if (!free_side_readable(c.tol, free_side)) { return false; }\n  double q = cost / s.a;",
            "  double q = cost / s.a;"),
    ],
    # The implied test with a band of 1e-3 instead of implied_tolerance.
    "v_r2_tol": [
        (R, "    if (!(k.bound + candidate_slack(k) <= upper + implied_tolerance(c.tol, upper))) {",
            "    if (!(k.bound + candidate_slack(k) <= upper + 1e-3)) {"),
        (R, "    if (!(k.bound - candidate_slack(k) >= lower - implied_tolerance(c.tol, lower))) {",
            "    if (!(k.bound - candidate_slack(k) >= lower - 1e-3)) {"),
    ],
    # The drift guard of an integer column is dropped (the ratios are still integral to tol.zero).
    "v_r2_drift": [
        (R, "    if (!form.integral ||\n        !drift_within(c.tol, c.tol.integral_drift_share, s.a, form, drift_units_t::row)) {",
            "    if (!form.integral) {"),
    ],
    # ---- R5 only ----
    # Only the bound ahead of the move is tested.
    "v_r5_ahead": [
        (R, "bool implied_free(context_t& c, int32_t row, const single_t& s, double side)\n{",
            "bool implied_free(context_t& c, int32_t row, const single_t& s, double side, int32_t skip = 0)\n{"),
        (R, "  if (upper < inf) {\n    if (!range_end(c, row, s.slot, side, s.a > 0, k)) { return false; }",
            "  if (upper < inf && skip != 1) {\n    if (!range_end(c, row, s.slot, side, s.a > 0, k)) { return false; }"),
        (R, "  if (lower > -inf) {\n    if (!range_end(c, row, s.slot, side, s.a < 0, k)) { return false; }",
            "  if (lower > -inf && skip != -1) {\n    if (!range_end(c, row, s.slot, side, s.a < 0, k)) { return false; }"),
        (R, "  if (!implied_free(c, row, s, side)) {",
            "  if (!implied_free(c, row, s, side, equation ? 0 : cost > 0 ? 1 : -1)) {"),
    ],
    # The inequality pivot is recorded and counted as a primal reduction.
    "v_r5_not_dual": [
        (R, "  return substitute(c, row, s, free_side, !equation, growth, largest_other);",
            "  return substitute(c, row, s, free_side, false, growth, largest_other);"),
    ],
    # When the side that the cost pushes against is infinite the other side is taken.
    "v_r5_any_side": [
        (R, "  if (!side_readable(c.tol, side)) { return false; }\n  double free_side = dd_residual(side, scanned.fixed_terms);",
            "  if (!equation && std::isinf(side)) {\n"
            "    side = side > 0 ? st.row_lower[row] : st.row_upper[row];\n  }\n"
            "  if (!side_readable(c.tol, side)) { return false; }\n  double free_side = dd_residual(side, scanned.fixed_terms);"),
    ],
    # ---- R3: zero-cost singleton columns ----
    # An infinite bound adds nothing to the range, so the new side stays finite.
    "v_r3_infinite": [
        (R, "    if (std::isinf(at_most)) {\n      most_infinite = true;\n    } else {",
            "    if (std::isinf(at_most)) {\n    } else {"),
    ],
    # The record holds the stored sides, with the fixed terms in them.
    "v_r3_saved_stored": [
        (R, "  saved.lower = free_lower;\n  saved.upper = free_upper;",
            "  saved.lower = lower;\n  saved.upper = upper;"),
    ],
    # The undo measures the room of a column without its coefficient.
    "v_r3_room": [
        (P, "    double room           = std::fabs(a) * std::fabs(target - c.prefer);",
            "    double room           = std::fabs(target - c.prefer);"),
    ],
    # The upper side is not widened.
    "v_r3_only_lower": [
        (R, "    if (!upper_first && new_upper != upper) { st.set_row_upper(row, new_upper); }\n", ""),
    ],
    # The new sides are taken from the sides less the fixed terms although the terms stay.
    "v_r3_fixed_side": [
        (R, "  double new_lower  = most_infinite ? -inf : dd_residual(lower, most);",
            "  double new_lower  = most_infinite ? -inf : dd_residual(dd_residual(lower, scanned.fixed_terms), most);"),
    ],
    # No guard on the sides less the fixed terms.
    "v_r3_free_guard": [
        (R, "      (new_upper < inf && !free_side_readable(c.tol, new_upper)) ||\n"
            "      !free_side_readable(c.tol, free_lower) || !free_side_readable(c.tol, free_upper)) {",
            "      (new_upper < inf && !free_side_readable(c.tol, new_upper))) {"),
    ],
    # ---- R4: the forcing column ----
    # A finite bound in the free direction is accepted.
    "v_r4_bound": [
        (R, "  bool down = st.down_locks[col] == 0 && lower == -inf && st.up_locks[col] >= 1;",
            "  bool down = st.down_locks[col] == 0 && st.up_locks[col] >= 1;"),
    ],
    # The undo reads the first row only.
    "v_r4_first_row": [
        (P, "    for (int32_t k = r.rows_begin; k < r.rows_end; ++k) {\n"
            "      const saved_row_t& row = ps.saved_rows[k];\n"
            "      double a               = ps.record_coefficients[r.coefficients + (k - r.rows_begin)];\n"
            "      double side            = from_above == (a > 0) ? row.upper : row.lower;",
            "    for (int32_t k = r.rows_begin; k < r.rows_begin + 1; ++k) {\n"
            "      const saved_row_t& row = ps.saved_rows[k];\n"
            "      double a               = ps.record_coefficients[r.coefficients + (k - r.rows_begin)];\n"
            "      double side            = from_above == (a > 0) ? row.upper : row.lower;"),
    ],
    # The saved rows keep the stored sides, with the fixed terms in them.
    "v_r4_stored_side": [
        (R, "    saved.push_back(copy);\n    values[kept++] = values[k];",
            "    copy.lower = st.row_lower[row];\n    copy.upper = st.row_upper[row];\n"
            "    saved.push_back(copy);\n    values[kept++] = values[k];"),
    ],
    # The coefficients are not moved up when a row without a finite side is skipped.
    "v_r4_coef_shift": [
        (R, "    values[kept++] = values[k];", "    ++kept;"),
    ],
    # The side is chosen as for a column that is free to go down, for the mirror as well.
    "v_r4_mirror": [
        (R, "    bool upper_side = down == (values[k] > 0);\n"
            "    double side     = upper_side ? st.row_upper[row] : st.row_lower[row];\n"
            "    gprs_assert(std::isinf(upper_side ? st.row_lower[row] : st.row_upper[row]));",
            "    bool upper_side = values[k] > 0;\n"
            "    double side     = upper_side ? st.row_upper[row] : st.row_lower[row];"),
    ],
    # No guard on the sides of the rows.
    "v_r4_noguard": [
        (R, "  if (!usable) {", "  if (!usable && c.opt.max_rounds < 0) {"),
    ],
    # ---- integrality in the alias forest: the state before the repair ----
    "v_af_drift": [
        ("alias_forest.cpp",
         "      st.is_integer(parent) && usable(scale, offset) && form.integral &&\n"
         "      drift_within(st.tol, st.tol.integral_drift_share, child_coef, form, drift_units_t::row);",
         "      st.is_integer(parent) && usable(scale, offset) && form.integral;"),
    ],
    # ---- lists ----
    # The family drops its row list: a bound change next to a singleton column is not served.
    "v_ev_rows": [
        (R, "    st.row_frontier.take(row_list::singleton_cols, w.rows);",
            "    st.row_frontier.take(row_list::singleton_cols, w.rows);\n    w.rows.clear();"),
    ],
    # The pass counter does not advance: a row is visited once in a presolve.
    "v_ev_stamp": [
        (R, "    ++w.pass;\n", "    w.pass = 1;\n"),
    ],
    # A history refusal does not report limited.
    "v_ev_limited": [
        (R, "    if (st.is_live(col)) { c.result.limited = true; }", "    (void)col;"),
    ],
}


def main():
    project = Path(sys.argv[1])
    out = Path(sys.argv[2])
    for name in sys.argv[3:] or MUTANTS:
        target = out / name / "src"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(project / "src", target)
        for file, old, new in MUTANTS[name]:
            text = (target / file).read_text()
            if text.count(old) != 1:
                print("%s: the text to replace occurs %d times in %s" % (name, text.count(old), file))
                return 1
            (target / file).write_text(text.replace(old, new))
        print("%s: %s" % (name, target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
