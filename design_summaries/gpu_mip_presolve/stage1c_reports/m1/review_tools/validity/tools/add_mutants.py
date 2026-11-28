"""add_mutants.py <check/make_mutants.py>: appends the wrong engines of the validity review."""

import sys
from pathlib import Path

p = Path(sys.argv[1])
t = p.read_text()
old = '''    "fc_round": (
        "postsolve.cpp",
        "      value = from_above ? std::floor(value + rounding) : "
        "std::ceil(value - rounding);",
        "      value = from_above ? std::ceil(value + rounding) : "
        "std::floor(value - rounding);",
    ),
}
'''
assert t.count(old) == 1
new = '''    "fc_round": (
        "postsolve.cpp",
        "      value = from_above ? std::floor(value + rounding) : "
        "std::ceil(value - rounding);",
        "      value = from_above ? std::ceil(value + rounding) : "
        "std::floor(value - rounding);",
    ),
    # The mutants of the validity review of the singleton-column rules.
    # The implied-bound test reads the side less the fixed terms with
    # activities that still hold those terms, so the terms come off
    # twice. The box part of the structure check; the oracle; the second
    # presolve of the campaign.
    "fs_fixed": (
        "rules_singleton_cols.cpp",
        "  if (!implied_free(c, row, s, side)) {",
        "  if (!implied_free(c, row, s, free_side)) {",
    ),
    # Each end of the range is computed with the other extreme of the
    # rest. The box part of the structure check; the oracle.
    "fs_ends": [
        (
            "rules_singleton_cols.cpp",
            "    if (!range_end(c, row, s.slot, side, s.a > 0, k)) "
            "{ return false; }",
            "    if (!range_end(c, row, s.slot, side, s.a == 0, k)) "
            "{ return false; }",
        ),
        (
            "rules_singleton_cols.cpp",
            "    if (!range_end(c, row, s.slot, side, s.a < 0, k)) "
            "{ return false; }",
            "    if (!range_end(c, row, s.slot, side, s.a > 0, k)) "
            "{ return false; }",
        ),
        (
            "rules_singleton_cols.cpp",
            "    if (!range_end(c, row, s.slot, side, s.a == 0, k)) "
            "{ return false; }",
            "    if (!range_end(c, row, s.slot, side, s.a < 0, k)) "
            "{ return false; }",
        ),
    ],
    # The record and the offset get the stored side, with the terms of the
    # fixed columns in it. The rows and columns parts of the structure
    # check; the oracle.
    "fs_stored": (
        "rules_singleton_cols.cpp",
        "  return substitute(c, row, s, free_side, !equation, growth, "
        "largest_other);",
        "  return substitute(c, row, s, side, !equation, growth, "
        "largest_other);",
    ),
    # The side of the row is not asked to be integral for an integer
    # column (2 k + 2 y = 3). The integral test and the postsolve part of
    # the structure check; the oracle.
    "fs_int_side": (
        "integral.hpp",
        "  integral_form_t f;\\n  integral_side(tol, a, side, f);\\n",
        "  integral_form_t f;\\n  (void)side;\\n",
    ),
    # A band of 1e-3 in place of implied_tolerance. The box part of the
    # structure check; the oracle.
    "fs_band": [
        (
            "rules_singleton_cols.cpp",
            "<= upper + implied_tolerance(c.tol, upper))) {",
            "<= upper + 1e-3)) {",
        ),
        (
            "rules_singleton_cols.cpp",
            ">= lower - implied_tolerance(c.tol, lower))) {",
            ">= lower - 1e-3)) {",
        ),
    ],
    # With an inequality as pivot row only the bound ahead of the move is
    # tested: the value passes the bound behind it at points of the
    # reduced model. The box part of the structure check; the oracle.
    "fs_ahead": [
        (
            "rules_singleton_cols.cpp",
            "bool implied_free(context_t& c, int32_t row, "
            "const single_t& s, double side)\\n{",
            "bool implied_free(\\n  context_t& c, int32_t row, "
            "const single_t& s, double side, int32_t skip = 0)\\n{",
        ),
        (
            "rules_singleton_cols.cpp",
            "  if (upper < inf) {\\n",
            "  if (upper < inf && skip != 1) {\\n",
        ),
        (
            "rules_singleton_cols.cpp",
            "  if (lower > -inf) {\\n    if (!range_end(",
            "  if (lower > -inf && skip != -1) {\\n    if (!range_end(",
        ),
        (
            "rules_singleton_cols.cpp",
            "  if (!implied_free(c, row, s, side)) {",
            "  if (!implied_free(c, row, s, side, "
            "equation ? 0 : cost > 0 ? 1 : -1)) {",
        ),
    ],
    # The inequality as pivot row is recorded and counted as a primal
    # reduction. The crush test of the oracle (a crushed point that costs
    # more than its point); requirement 5 of the crush check. Not the
    # structure check.
    "fs_not_dual": (
        "rules_singleton_cols.cpp",
        "  return substitute(c, row, s, free_side, !equation, growth, "
        "largest_other);",
        "  return substitute(c, row, s, free_side, false, growth, "
        "largest_other);",
    ),
    # When the side that the cost pushes the column against is infinite,
    # the other side is taken. The optimum and the crush test of the
    # oracle. Not the structure check: a restriction.
    "fs_any_side": (
        "rules_singleton_cols.cpp",
        "  if (!side_readable(c.tol, side)) { return false; }\\n"
        "  double free_side = dd_residual(side, scanned.fixed_terms);",
        "  if (!equation && std::isinf(side)) {\\n"
        "    side = side > 0 ? st.row_lower[row] : st.row_upper[row];\\n"
        "  }\\n"
        "  if (!side_readable(c.tol, side)) { return false; }\\n"
        "  double free_side = dd_residual(side, scanned.fixed_terms);",
    ),
    # An infinite bound of a zero-cost singleton column adds nothing to
    # the range of its term, so the new side stays finite: a restriction.
    # The oracle. Not the structure check.
    "zs_infinite": (
        "rules_singleton_cols.cpp",
        "    if (std::isinf(at_most)) {\\n      most_infinite = true;\\n"
        "    } else {",
        "    if (std::isinf(at_most)) {\\n    } else {",
    ),
    # The record of the zero-cost singleton columns holds the stored
    # sides, with the terms of the fixed columns in them. The kept-rows
    # and columns parts of the structure check; the oracle.
    "zs_stored": (
        "rules_singleton_cols.cpp",
        "  saved.lower = free_lower;\\n  saved.upper = free_upper;",
        "  saved.lower = lower;\\n  saved.upper = upper;",
    ),
    # The undo measures the room of a column without its coefficient.
    # The postsolve part of the structure check; the oracle.
    "zs_room": (
        "postsolve.cpp",
        "    double room           = std::fabs(a) * "
        "std::fabs(target - c.prefer);",
        "    double room           = std::fabs(target - c.prefer);",
    ),
    # The upper side is not widened: a restriction. The oracle.
    "zs_one_side": (
        "rules_singleton_cols.cpp",
        "    if (!upper_first && new_upper != upper) "
        "{ st.set_row_upper(row, new_upper); }\\n",
        "",
    ),
    # The new lower side is taken from the side less the fixed terms
    # although the terms stay in the row. The columns part of the
    # structure check; the oracle.
    "zs_fixed": (
        "rules_singleton_cols.cpp",
        "  double new_lower  = most_infinite ? -inf : "
        "dd_residual(lower, most);",
        "  double new_lower  = most_infinite\\n"
        "                        ? -inf\\n"
        "                        : dd_residual(dd_residual(lower, "
        "scanned.fixed_terms), most);",
    ),
    # A forcing column with a finite bound in its free direction. The
    # stack part of the structure check; the oracle.
    "fc_bound": (
        "rules_singleton_cols.cpp",
        "  bool down = st.down_locks[col] == 0 && lower == -inf && "
        "st.up_locks[col] >= 1;",
        "  bool down = st.down_locks[col] == 0 && st.up_locks[col] >= 1;",
    ),
    # The undo of a forcing column reads its first row only. The
    # postsolve part of the structure check; the oracle.
    "fc_first_row": (
        "postsolve.cpp",
        "    for (int32_t k = r.rows_begin; k < r.rows_end; ++k) {\\n"
        "      const saved_row_t& row = ps.saved_rows[k];\\n"
        "      double a               = ",
        "    for (int32_t k = r.rows_begin; k < r.rows_begin + 1; ++k) {\\n"
        "      const saved_row_t& row = ps.saved_rows[k];\\n"
        "      double a               = ",
    ),
    # The saved rows of a forcing column keep the stored sides. The
    # kept-rows part of the structure check; the oracle.
    "fc_stored": (
        "rules_singleton_cols.cpp",
        "    saved.push_back(copy);\\n    values[kept++] = values[k];",
        "    copy.lower = st.row_lower[row];\\n"
        "    copy.upper = st.row_upper[row];\\n"
        "    saved.push_back(copy);\\n    values[kept++] = values[k];",
    ),
    # The coefficients of a forcing record are not moved up when a row
    # without a finite side is skipped. The stack and kept-rows parts of
    # the structure check; the oracle; tests/test_columns_review.cpp.
    "fc_shift": (
        "rules_singleton_cols.cpp",
        "    values[kept++] = values[k];",
        "    ++kept;",
    ),
    # The family drops its row list: a bound that changes next to a
    # singleton column is not served. The second presolve of the campaign.
    "ev_rows": (
        "rules_singleton_cols.cpp",
        "    st.row_frontier.take(row_list::singleton_cols, w.rows);\\n"
        "    if (w.cols.empty()",
        "    st.row_frontier.take(row_list::singleton_cols, w.rows);\\n"
        "    w.rows.clear();\\n"
        "    if (w.cols.empty()",
    ),
    # The pass counter stands still: a row is visited once in a presolve.
    # The second presolve of the campaign.
    "ev_stamp": (
        "rules_singleton_cols.cpp",
        "    ++w.pass;\\n",
        "    w.pass = 1;\\n",
    ),
    # Wrong engines that the generator of the campaign does not reach.
    # Each fails the unit test named for it and no check of the campaign:
    # no generated model has a side that loses digits next to a fixed
    # term beside a column of these rules, a ratio that is an integer only
    # within the tolerance, a chain of pivot ratios above the growth limit
    # or a side that turns readable.
    # No guard on the side less the fixed terms, for each of the three
    # readers (columns_singleton_side_less_the_fixed_terms,
    # columns_singleton_forcing_column,
    # columns_review_side_swallowed_by_a_fixed_term).
    "fs_guard": (
        "rules_singleton_cols.cpp",
        "  if (!free_side_readable(c.tol, free_side)) { return false; }\\n"
        "  double q = cost / s.a;",
        "  double q = cost / s.a;",
    ),
    "zs_guard": (
        "rules_singleton_cols.cpp",
        "      (new_upper < inf && !free_side_readable(c.tol, new_upper)) "
        "||\\n      !free_side_readable(c.tol, free_lower) || "
        "!free_side_readable(c.tol, free_upper)) {",
        "      (new_upper < inf && !free_side_readable(c.tol, new_upper))) "
        "{",
    ),
    "fc_guard": (
        "rules_singleton_cols.cpp",
        "  if (!usable) {",
        "  if (!usable && c.opt.max_rounds < 0) {",
    ),
    # No drift guard for an integer free singleton column
    # (columns_singleton_integer_column_and_the_drift), and none in the
    # alias forest (alias_integer_edge_needs_integral_scale_and_offset).
    "fs_drift": (
        "rules_singleton_cols.cpp",
        "    if (!form.integral ||\\n"
        "        !drift_within(c.tol, c.tol.integral_drift_share, s.a, form, "
        "drift_units_t::row)) {",
        "    if (!form.integral) {",
    ),
    "r_alias_drift": (
        "alias_forest.cpp",
        "form.integral &&\\n      drift_within(st.tol, "
        "st.tol.integral_drift_share, child_coef, form, drift_units_t::row);",
        "form.integral;",
    ),
    # A column that only the growth factor of earlier substitutions kept
    # does not report limited
    # (columns_singleton_growth_factor_follows_a_chain).
    "fs_limited": (
        "rules_singleton_cols.cpp",
        "    if (st.is_live(col)) { c.result.limited = true; }",
        "    (void)col;",
    ),
    # A row whose side a forcing column was left for is not marked: the
    # column is not tested again when the side changes
    # (columns_review_forcing_column_follows_the_sides_of_its_rows,
    # columns_review_fixed_point_with_a_side_that_becomes_readable).
    "ev_sides": (
        "rules_singleton_cols.cpp",
        "    if (!usable) { c.w.unreadable[row] = 1; }\\n",
        "",
    ),
}
'''
t = t.replace(old, new)
p.write_text(t)
