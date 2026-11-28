# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write deliberately wrong copies of the engine sources, to test the check.

Usage: make_mutants.py <out dir> [name ...]
Each mutant is <out dir>/<name>/src: the engine sources, internal headers
included, with one exact text replacement, or a few. The script stops when
the text of a replacement is not found exactly once, so a mutant that the
engine has outgrown is noticed. A replacement in a header changes every
source that includes it: a build of a mutant compiles all of its sources.

A check tool with a mutant engine is built with
  cmake -S gpu_presolve/check -B <build dir> -DGPRS_ENGINE_SRC=<out dir>/<name>/src
and run with run_check.py as the tool of the real engine. The check is
expected to fail on the instances whose presolve the mutant changes; where
it does not, it cannot see that kind of error on that instance.
"""

import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DUAL_FIXING = (
    "bool to_lower = cost >= 0 && st.down_locks[col] == 0;\n"
    "    bool to_upper = cost <= 0 && st.up_locks[col] == 0;"
)
PROPAGATED_UPPER = (
    "bound_change_t change = is_upper "
    "? tighten_upper(st, c.ps, col, bound + rounding)"
)
MERGE_SWAP = "if (ratio < 0) { std::swap(lower, upper); }"
# Dual fixing with locks counted from the column, an equation as a <= row.
DUAL_FIXING_EQUATION = """\
int32_t down_count = 0, up_count = 0;
    for (int64_t p = st.csc.begin(col); p < st.csc.end(col); ++p) {
      if (!st.csc.live(p)) { continue; }
      int32_t row       = st.csc.index[p];
      bool positive     = st.csc.value[p] > 0;
      bool equation     = st.row_lower[row] == st.row_upper[row];
      bool lower_finite = st.row_lower[row] > -inf && !equation;
      bool upper_finite = st.row_upper[row] < inf;
      up_count += positive ? upper_finite : lower_finite;
      down_count += positive ? lower_finite : upper_finite;
    }
    bool to_lower = cost >= 0 && down_count == 0;
    bool to_upper = cost <= 0 && up_count == 0;"""
FORCING_TEST = "  if (!(std::fabs(gap) + error <= %s)) { return false; }"
SUBTRACT_SHIFT = "st.shift_row_sides(target.row, factor, rhs);"
SUBTRACT_RECORD = "  ps.push_row_addition(target.row, equation.row, factor);\n"
EXPLICIT_BOUND = "    bound        = is_upper ? bound + slack : bound - slack;"
EXPLICIT_SLACK = "    double slack = candidate_slack(k);\n" + EXPLICIT_BOUND
COST_CANCEL = (
    "if (std::fabs(sum) <= tol.cancellation * (std::fabs(old) + "
    "std::fabs(moved))) { sum = 0.; }"
)
DUAL_FIX = (
    "    budget.spend(st.csc.used[col]);\n    fix_column(st, ps, col, value);"
)

# name: (file, text, replacement), or a list of them
MUTANTS = {
    # A wrong "infeasible": a row whose minimum activity meets its upper
    # side (a forcing row) is reported infeasible.
    "m_infeasible": (
        "rules_rows.cpp",
        "if (upper_gap < -(feas + min_error) || lower_gap > feas + max_error) {",
        "if (upper_gap < feas + min_error || lower_gap > feas + max_error) {",
    ),
    # Dual fixing with the lock direction swapped.
    "m_duallock": (
        "rules_dual_fixing.cpp",
        DUAL_FIXING,
        "bool to_lower = cost >= 0 && st.up_locks[col] == 0;\n"
        "    bool to_upper = cost <= 0 && st.down_locks[col] == 0;",
    ),
    # Dual fixing that ignores the sign of the cost.
    "m_dualcost": (
        "rules_dual_fixing.cpp",
        DUAL_FIXING,
        "bool to_lower = st.down_locks[col] == 0;\n"
        "    bool to_upper = st.up_locks[col] == 0;",
    ),
    # A row deleted as redundant although its maximum activity is up to 1
    # above its upper side.
    "m_redundant": (
        "rules_rows.cpp",
        "upper == inf || (max_finite && "
        "dd_residual(upper, max_sum) >= max_error - feas);",
        "upper == inf || (max_finite && "
        "dd_residual(upper, max_sum) >= max_error - feas - 1.);",
    ),
    # A wrong objective offset: half the cost of a fixed column.
    "m_offset": (
        "state_ops.cpp",
        "add_objective_offset(cost[col], value);",
        "add_objective_offset(0.5 * cost[col], value);",
    ),
    # An upper bound from propagation one unit too tight, on every integer
    # column: most instances end infeasible.
    "m_propcut": (
        "rules_propagation.cpp",
        PROPAGATED_UPPER,
        "bound_change_t change = is_upper ? tighten_upper(st, c.ps, col, "
        "bound + rounding - "
        "(st.col_upper[col] - st.col_lower[col] >= 1 ? 1. : 0.))",
    ),
    # The same on one integer column in five, and only where the bound
    # stays at or above the lower bound: a restriction that does not make
    # the model infeasible at once.
    "m_bound_sparse": (
        "rules_propagation.cpp",
        PROPAGATED_UPPER,
        "bound_change_t change = is_upper ? tighten_upper(st, c.ps, col, "
        "bound + rounding - (col % 5 == 0 && "
        "std::floor(bound + rounding) - 1. >= st.col_lower[col] ? 1. : 0.))",
    ),
    # A coefficient strengthened too far: half the valid magnitude.
    "m_coef": (
        "rules_tightening.cpp",
        "double magnitude = d + error;",
        "double magnitude = 0.5 * (d + error);",
    ),
    # A wrong side after a parallel-row merge: the sides of a row with a
    # negative ratio are not swapped.
    "m_parallel_swap": ("rules_parallel_rows.cpp", MERGE_SWAP, ""),
    # ... the side is multiplied by the ratio instead of divided: the
    # quotient of the two rows is taken the wrong way round.
    "m_parallel_ratio": (
        "rules_parallel_rows.cpp",
        "    parts_t numerator = parts_of(kept_first), "
        "denominator = parts_of(removed_first);",
        "    parts_t numerator = parts_of(removed_first), "
        "denominator = parts_of(kept_first);",
    ),
    # ... the upper side taken from the removed row is one unit too small.
    "m_parallel_tight": (
        "rules_parallel_rows.cpp",
        MERGE_SWAP,
        MERGE_SWAP + "\n  upper.value -= 1.;",
    ),
    # ... the sides of the removed row are dropped.
    "m_parallel_loose": (
        "rules_parallel_rows.cpp",
        MERGE_SWAP,
        "lower       = image_t();\n  lower.value = -inf;\n"
        "  upper       = image_t();\n  upper.value = inf;",
    ),
    # Dual fixing left on in primal-only mode.
    "m_primal": (
        "presolve.cpp",
        "return opt.enabled[rule] && "
        "!(rule == rule::dual_fixing && opt.primal_only);",
        "return opt.enabled[rule];",
    ),
    # The mutants of the review of the check (r_*). Relaxations first.
    # The bounds of a substituted column do not reach its root.
    "r_dbl_bounds": (
        "rules_doubleton.cpp",
        "result.outcome = transfer_bounds(st, ps, tol, map);",
        "result.outcome = ok;",
    ),
    # An integer column is substituted by a continuous one.
    "r_alias_dir": [
        (
            "alias_forest.cpp",
            "st.is_integer(parent) && usable(scale, offset) && "
            "form.integral &&",
            "usable(scale, offset) && form.integral &&",
        ),
        (
            "alias_forest.cpp",
            "if (st.is_integer(col)) { best = v; }",
            "if (!st.is_integer(col)) { best = v; }",
        ),
    ],
    # One integer column in ten is continuous in the reduced model.
    "r_coltype": (
        "state_storage.cpp",
        "    model.col_type[cols]  = (uint8_t)st.col_type[j];",
        "    model.col_type[cols]  = "
        "j % 10 == 0 ? 0 : (uint8_t)st.col_type[j];",
    ),
    # One one-sided row in fifty is deleted whatever its activity is.
    "r_rowdrop": (
        "rules_rows.cpp",
        "  if (upper_redundant && lower_redundant) {",
        "  if ((upper_redundant && lower_redundant) ||\n"
        "      (row % 50 == 0 && (upper_redundant || lower_redundant))) {",
    ),
    # The upper bound of a singleton row is lost, one row in three.
    "r_singleton": (
        "rules_rows.cpp",
        "  change = tighten_upper(st, c.ps, col, upper);",
        "  change = row % 3 == 0 ? bound_change_t{} : "
        "tighten_upper(st, c.ps, col, upper);",
    ),
    # The lower bound of an integer column from a singleton row is one
    # unit too loose.
    "r_singleton_round": (
        "rules_rows.cpp",
        "  bound_change_t change = tighten_lower(st, c.ps, col, lower);",
        "  bound_change_t change = tighten_lower(st, c.ps, col, "
        "st.is_integer(col) && std::isfinite(lower) ? "
        "std::floor(lower) - 1. : lower);",
    ),
    # Coefficient strengthening without the move of the side.
    "r_strengthen_side": (
        "rules_tightening.cpp",
        "  if (value == side) { return; }\n",
        "  if (value == value) { return; }\n",
    ),
    # The side divided by the gcd is rounded outward.
    "r_gcd": (
        "rules_tightening.cpp",
        "is_lower ? std::ceil(integral / divisor) : "
        "std::floor(integral / divisor);",
        "is_lower ? std::floor(integral / divisor) : "
        "std::ceil(integral / divisor);",
    ),
    # The side of an integer row is rounded outward (no local instance
    # has such a side).
    "r_gcd_side": (
        "rules_tightening.cpp",
        "is_lower ? std::ceil(scaled - feas) : std::floor(scaled + feas);",
        "is_lower ? std::floor(scaled + feas) : std::ceil(scaled - feas);",
    ),
    # The reduced model drops the fixed terms of one column in four from
    # the sides of its rows: looser or tighter by the sign of the term.
    "r_rebuild": (
        "state_storage.cpp",
        "dd_add_product(fixed_terms, csr.value[e], st.col_lower[j]);",
        "dd_add_product(fixed_terms, csr.value[e], "
        "j % 4 == 0 ? 0. : st.col_lower[j]);",
    ),
    # A parallel-row merge forgets the fixed terms of the kept row.
    "r_parallel_fixed": (
        "rules_parallel_rows.cpp",
        "  double fixed = dd_value(kept_fixed);",
        "  double fixed = 0.;",
    ),
    # Wrong substitutions. The sides of the rows of a substituted column
    # move the wrong way.
    "r_dbl_side": (
        "rules_doubleton.cpp",
        "st.shift_row_sides(row, a, map.offset);",
        "st.shift_row_sides(row, a, -map.offset);",
    ),
    # The substituted coefficient loses the sign of the scale.
    "r_dbl_scale": (
        "rules_doubleton.cpp",
        "double moved = a * map.scale;",
        "double moved = a * std::fabs(map.scale);",
    ),
    # The substituted cost is not scaled.
    "r_dbl_cost": (
        "rules_doubleton.cpp",
        "double moved = cost * map.scale;",
        "double moved = cost;",
    ),
    # The offset of a substituted column does not reach the objective.
    "r_dbl_objoff": (
        "rules_doubleton.cpp",
        "    st.add_objective_offset(cost, map.offset);\n",
        "",
    ),
    # Wrong postsolve. The alias record holds a wrong offset.
    "r_alias_offset": (
        "rules_doubleton.cpp",
        "ps.push_alias(col, root, map.row, map.scale, map.offset);",
        "ps.push_alias(col, root, map.row, map.scale, map.offset + 1.);",
    ),
    # The records are undone in the order they were pushed.
    "r_order": [
        (
            "postsolve.cpp",
            "  size_t fixed = fixed_cols.size(), alias = aliases.size();\n"
            "  for (size_t k = order.size(); k-- > 0;) {",
            "  size_t fixed = 0, alias = 0;\n"
            "  for (size_t k = 0; k < order.size(); ++k) {",
        ),
        ("postsolve.cpp", "fixed_cols[--fixed];", "fixed_cols[fixed++];"),
        ("postsolve.cpp", "aliases[--alias];", "aliases[alias++];"),
    ],
    # Restrictions. A row whose minimum activity is within 1 of its side
    # is taken as forcing, one row in seven.
    "r_forcing": [
        (
            "rules_rows.cpp",
            FORCING_TEST % "c.tol.feasibility",
            "  double wrong = row % 7 == 0 ? 1. : 0.;\n"
            + FORCING_TEST % "c.tol.feasibility + wrong",
        ),
        (
            "rules_rows.cpp",
            FORCING_TEST % "tolerance",
            FORCING_TEST % "tolerance + wrong",
        ),
    ],
    # A forcing row fixes its columns at the other bound.
    "r_forcing_end": (
        "rules_rows.cpp",
        "    c.work.values.push_back(at_minimum",
        "    c.work.values.push_back(!at_minimum",
    ),
    # Propagation over a row with one infinite term as if it had none,
    # one row in three. The candidate of an entry is computed in implied.hpp,
    # by the function that bound propagation shares with the column rules:
    # candidate_of_finite_rest, which tests no infinite term itself. The
    # test of candidate(), which the column rules call, stays as it is.
    "r_prop_inf": [
        (
            "rules_propagation.cpp",
            "  if (act.inf_count[row] == 1) {",
            "  if (act.inf_count[row] == 1 && row % 3 != 0) {",
        ),
        (
            "implied.hpp",
            "  if (infinite == 0) {",
            "  if (infinite == 0 || act.inf_slot[row] != e) {",
        ),
    ],
    # An implied upper bound made explicit a little too tight.
    "r_explicit": (
        "rules_propagation.cpp",
        "bound_change_t change = tighten_upper(st, ps, col, bound);",
        "bound_change_t change = tighten_upper(st, ps, col, "
        "bound - 1e-4 * (1. + std::fabs(bound)));",
    ),
    # Dual fixing takes an equation as a <= row.
    "r_dual_eq": ("rules_dual_fixing.cpp", DUAL_FIXING, DUAL_FIXING_EQUATION),
    # Dual fixing at the other bound when the wanted one is infinite.
    "r_dual_inf": (
        "rules_dual_fixing.cpp",
        "        continue;\n      }\n    }\n"
        "    budget.spend(st.csc.used[col]);",
        "        value = to_lower ? upper : lower;\n"
        "        if (!std::isfinite(value)) { continue; }\n"
        "      }\n    }\n    budget.spend(st.csc.used[col]);",
    ),
    # The mutants of the two rules of the merge 0ee2ce720 (a_*). First the
    # subtraction of an equation from a row that is parallel to it up to
    # singleton columns: row <- row - factor * equation. The first five
    # leave a row that is off by a multiple of the right-hand side or of
    # an entry: each relaxes the row where the error moves a side outward
    # (for the structure check) and restricts it where it moves one inward
    # (for the oracle and the crush check), by the signs of the instance.
    # The sides of the row move with the wrong sign: off by twice the
    # factor times the right-hand side.
    "a_sub_sign": (
        "rules_parallel_rows.cpp",
        SUBTRACT_SHIFT,
        "st.shift_row_sides(target.row, -factor, rhs);",
    ),
    # The sides of the row are not moved: off by the factor times the
    # right-hand side.
    "a_sub_side": (
        "rules_parallel_rows.cpp",
        "  if (factor * rhs != 0. && (lower > -inf || upper < inf)) {\n"
        "    " + SUBTRACT_SHIFT + "\n  }\n",
        "",
    ),
    # A wrong factor, 1.5 times the ratio of the first shared entries, in
    # the sides, in the entry that enters the row and in the record: the
    # sides are off by half the factor times the right-hand side, and the
    # entry of the equation's singleton column is half too large. The
    # limit on the factor reads the right one, so that the subtractions
    # are those of the engine. Where the right-hand side is zero and the
    # equation has no singleton column, only the factor of the record is
    # wrong, which no check reads.
    "a_sub_factor": [
        (
            "rules_parallel_rows.cpp",
            "  double factor           = target.first / equation.first;",
            "  double factor           = 1.5 * target.first / equation.first;",
        ),
        (
            "rules_parallel_rows.cpp",
            "  double scale = std::fabs(factor);",
            "  double scale = std::fabs(factor) / 1.5;",
        ),
    ],
    # The entry of the equation's singleton column enters the row with the
    # wrong sign: off by twice that entry.
    "a_sub_single": (
        "rules_parallel_rows.cpp",
        "equation.singles == 1 ? -factor * equation.single_value : 0.;",
        "equation.singles == 1 ? factor * equation.single_value : 0.;",
    ),
    # A row with a finite upper side is taken as an equation at that side.
    # Relaxes the side of the target that needs the lower side of that row
    # (the row holds below its upper side only) and restricts the other.
    "a_sub_inequality": (
        "rules_parallel_rows.cpp",
        "  return is_equation(st, equation.row) &&",
        "  return (is_equation(st, equation.row) ||\n"
        "          st.row_upper[equation.row] < inf) &&",
    ),
    # The subtraction leaves no record. Neither relaxes nor restricts: the
    # reduced model is the right one and the primal postsolve ignores the
    # record. Only the structure check sees it, which cannot prove the row
    # from its own reduced row.
    "a_sub_record": (
        "rules_parallel_rows.cpp",
        SUBTRACT_RECORD,
        "  (void)ps;\n",
    ),
    # The record names the two rows the other way round. As a_sub_record
    # for the row; the equation is tested with the row as its equation.
    "a_sub_swap": (
        "rules_parallel_rows.cpp",
        SUBTRACT_RECORD,
        "  ps.push_row_addition(equation.row, target.row, factor);\n",
    ),
    # The sides of the row are off by the factor, whatever the right-hand
    # side is: an error that does not grow with the sides of the equation,
    # for the scaling of a violation (a huge side must not forgive it).
    "a_sub_unit": (
        "rules_parallel_rows.cpp",
        SUBTRACT_SHIFT,
        "st.shift_row_sides(target.row, factor, rhs - 1.);",
    ),
    # ... off by a thousandth of the factor: 100 times the tolerance of the
    # structure check on a row with terms of size 1.
    "a_sub_small": (
        "rules_parallel_rows.cpp",
        SUBTRACT_SHIFT,
        "st.shift_row_sides(target.row, factor, rhs - 1e-3);",
    ),
    # The last entry that the two rows share stays in the row.
    "a_sub_keep": (
        "rules_parallel_rows.cpp",
        "  for (; k < slots.size(); ++k) {\n    st.delete_entry(slots[k]);",
        "  for (; k + 1 < slots.size(); ++k) {\n"
        "    st.delete_entry(slots[k]);",
    ),
    # The entry of the equation's singleton column enters the row with
    # half its value.
    "a_sub_half": (
        "rules_parallel_rows.cpp",
        "equation.singles == 1 ? -factor * equation.single_value : 0.;",
        "equation.singles == 1 ? -0.5 * factor * equation.single_value : 0.;",
    ),
    # Then the rounds on explicit bounds. Both restrict: a bound of the
    # reduced model that is tighter than the rows allow cuts feasible
    # points off and relaxes nothing, so the structure check passes.
    # A candidate bound of a continuous column is taken a little too tight.
    "a_exp_tight": (
        "rules_propagation.cpp",
        EXPLICIT_SLACK,
        "    double slack = candidate_slack(k);\n"
        "    if (c.work.explicit_bounds) {\n"
        "      slack -= 1e-4 * (1. + std::fabs(bound));\n"
        "    }\n" + EXPLICIT_BOUND,
    ),
    # A candidate of a continuous column is taken on the wrong side, the
    # upper bound that a row gives as a lower bound, one row in three.
    "a_exp_side": (
        "rules_propagation.cpp",
        "  bool is_upper = (st.csr.value[e] > 0) == upper_side;",
        "  bool is_upper = ((st.csr.value[e] > 0) == upper_side) !=\n"
        "                  (c.work.explicit_bounds && !st.is_integer(col) &&\n"
        "                   row % 3 == 0);",
    ),
    # The mutants of the groundwork of the column rules (ev_*): an event
    # that their lists do not get. A presolve result shows such an engine
    # only through a reduction that reads the list: a second presolve of a
    # fixed point then still reduces. Until the families hold reductions,
    # the tests of the state are what fails (tests/test_state_ops.cpp,
    # test_state_random.cpp, test_bounds_tighten.cpp), and the test that
    # runs the Stage 1 families and reads the lists after each
    # (tests/test_columns_events.cpp), with gprs_tests built on the mutant
    # sources.
    # A bound change whose activity propagation does not read (the side
    # that reads it is infinite) does not list the row for the column rules.
    "ev_unread": (
        "state_ops.cpp",
        "st.row_frontier.push(propagates ? row_event::activity "
        ": row_event::activity_unread, row);",
        "st.row_frontier.push(\n"
        "      propagates\n"
        "        ? row_event::activity\n"
        "        : (list_mask_t)(row_event::activity_unread &\n"
        "                        ~row_event::column_rules),\n"
        "      row);",
    ),
    # scale_row lists no column.
    "ev_scale": (
        "state_ops.cpp",
        "    if (is_live(csr.index[e])) { "
        "col_frontier.push(col_event::entries, csr.index[e]); }\n",
        "",
    ),
    # The mutants of a cost or an offset (obj_*): the objective part of the
    # structure check must catch each, next to the optimum of the oracle.
    # A fixing books cost * (old lower bound) where it fixes elsewhere.
    "obj_fix_bound": (
        "state_ops.cpp",
        "  add_objective_offset(cost[col], value);",
        "  add_objective_offset(cost[col], lower > -inf ? lower : value);",
    ),
    # A substitution moves the cost with the absolute value of the scale.
    "obj_alias_abs": (
        "rules_doubleton.cpp",
        "    double moved = cost * map.scale;",
        "    double moved = cost * std::fabs(map.scale);",
    ),
    # A moved cost of size up to 1 counts as a cancellation.
    "obj_cancel_wide": (
        "rules_doubleton.cpp",
        COST_CANCEL,
        "if (std::fabs(sum) <= 1.) { sum = 0.; }",
    ),
    # The offset of a substitution is booked with the moved cost.
    "obj_offset_scale": (
        "rules_doubleton.cpp",
        "    st.add_objective_offset(cost, map.offset);",
        "    st.add_objective_offset(moved, map.offset);",
    ),
    # The mutants of the forward map (cr_*): the optimum and every
    # postsolved point are right, so the crush test of the oracle is the
    # check that must catch each, on results with a dual finding.
    # Dual fixing does not report its findings as dual: a crushed point
    # must then cost the same and costs less.
    "cr_uncounted": (
        "rules_dual_fixing.cpp",
        "    ++result.dual_findings;\n",
        "",
    ),
    # An integer column with cost > 0 and no down lock is not fixed: it
    # keeps its lower bound and the value above it and stays in the model.
    # Every optimal point is kept, but no record moves a feasible point
    # with a larger value into the new bounds.
    "cr_clip": (
        "rules_dual_fixing.cpp",
        DUAL_FIX,
        "    if (cost > 0 && st.is_integer(col) && lower > -inf && "
        "upper - lower >= 2) {\n"
        "      tighten_upper(st, ps, col, lower + 1);\n"
        "      ++result.findings;\n"
        "      ++result.dual_findings;\n"
        "      continue;\n"
        "    }\n" + DUAL_FIX,
    ),
    # crush exchanges the first two columns of the reduced point.
    "cr_crush_swap": (
        "postsolve.cpp",
        "    x_reduced[k] = x_original[col_origin[k]];\n  }\n",
        "    x_reduced[k] = x_original[col_origin[k]];\n  }\n"
        "  if (x_reduced.size() >= 2) {\n"
        "    double first = x_reduced[0];\n"
        "    x_reduced[0] = x_reduced[1];\n"
        "    x_reduced[1] = first;\n"
        "  }\n",
    ),
    # The mutants of the singleton-column rules, with the check that must
    # catch each. First the free singleton column (fs_*).
    # The lower end of the range that the pivot row gives the column is not
    # tested against its bound: a relaxation of that bound. The box part
    # of the structure check; the oracle.
    "fs_implied": (
        "rules_singleton_cols.cpp",
        "  if (lower > -inf) {\n"
        "    if (!range_end(c, row, s.slot, side, s.a < 0, k)) "
        "{ return false; }",
        "  if (lower > -inf && c.opt.max_rounds < 0) {\n"
        "    if (!range_end(c, row, s.slot, side, s.a < 0, k)) "
        "{ return false; }",
    ),
    # The cost of the column is not moved onto the other columns of its
    # pivot row. The objective part of the structure check; requirement 4
    # of the crush check; the optimum of the oracle.
    "fs_cost": (
        "rules_singleton_cols.cpp",
        "      st.set_cost(other, sum);\n",
        "",
    ),
    # ... the offset is not moved. The same checks.
    "fs_offset": (
        "rules_singleton_cols.cpp",
        "    st.add_objective_offset(q, free_side);\n",
        "",
    ),
    # An integer column is defined by a row that is not integral for it.
    # The integral test of the structure check; the oracle. Not the crush
    # check: a point of the model has integers that meet the row.
    "fs_int": (
        "rules_singleton_cols.cpp",
        "    if (!form.integral ||\n"
        "        !drift_within(c.tol, c.tol.integral_drift_share, s.a, form, "
        "drift_units_t::row)) {",
        "    if (c.opt.max_rounds < 0 && !form.integral &&\n"
        "        !drift_within(c.tol, c.tol.integral_drift_share, s.a, form, "
        "drift_units_t::row)) {",
    ),
    # An inequality is taken as tight at the side that the cost of the
    # column pushes it away from: a restriction to points that cost more.
    # Requirement 4 of the crush check; the optimum of the oracle.
    "fs_side": (
        "rules_singleton_cols.cpp",
        "    side       = rises ? st.row_upper[row] : st.row_lower[row];",
        "    side       = rises ? st.row_lower[row] : st.row_upper[row];",
    ),
    # The inequality as pivot row in primal-only mode. The stack part of
    # the structure check (a dual finding and a dual record in a result of
    # that mode); requirement 5 of the crush check.
    "fs_primal": (
        "rules_singleton_cols.cpp",
        "  if (scanned.free_entries >= 2 && sub_rule_active(c.opt, sub)) {",
        "  if (scanned.free_entries >= 2 && c.opt.sub_enabled[sub]) {",
    ),
    # Then the zero-cost singleton columns (zs_*). The new lower side is
    # one unit too wide: a relaxation of the row. The snapshot of the
    # structure check; the oracle.
    "zs_wide": (
        "rules_singleton_cols.cpp",
        "  double new_lower  = most_infinite ? -inf : "
        "dd_residual(lower, most);",
        "  double new_lower  = most_infinite ? -inf : "
        "dd_residual(lower, most) - 1.;",
    ),
    # The largest and the smallest sum of the terms are exchanged: sides
    # that are too tight, a restriction. Requirement 2 of the crush check;
    # the oracle. Not the structure check.
    "zs_side": (
        "rules_singleton_cols.cpp",
        "  double new_lower  = most_infinite ? -inf : "
        "dd_residual(lower, most);\n"
        "  double new_upper  = least_infinite ? inf : "
        "dd_residual(upper, least);",
        "  double new_lower  = least_infinite ? -inf : "
        "dd_residual(lower, least);\n"
        "  double new_upper  = most_infinite ? inf : "
        "dd_residual(upper, most);",
    ),
    # A column with a cost is taken. The objective part of the structure
    # check; the oracle.
    "zs_cost": (
        "rules_singleton_cols.cpp",
        "    if (st.is_integer(s.col) || st.cost[s.col] != 0.) { continue; }",
        "    if (st.is_integer(s.col)) { continue; }",
    ),
    # An integer column is taken. The stack part of the structure check;
    # the oracle.
    "zs_int": (
        "rules_singleton_cols.cpp",
        "    if (st.is_integer(s.col) || st.cost[s.col] != 0.) { continue; }",
        "    if (st.cost[s.col] != 0.) { continue; }",
    ),
    # The undo leaves the columns at prefer whatever the row needs.
    # Requirement 3 of the crush check; the oracle.
    "zs_undo": (
        "postsolve.cpp",
        "  if (!(below > 0) && !(above > 0)) { return; }",
        "  if (!(below > 1e300) && !(above > 1e300)) { return; }",
    ),
    # Last the forcing column (fc_*). A lock in the free direction is
    # ignored: rows that hold the column from both ends are deleted with
    # it. The stack part of the structure check (a row of the record that
    # does not bound the column from the one end); the oracle.
    "fc_lock": (
        "rules_singleton_cols.cpp",
        "  bool down = st.down_locks[col] == 0 && lower == -inf && "
        "st.up_locks[col] >= 1;",
        "  bool down = lower == -inf && st.up_locks[col] >= 1;",
    ),
    # A column with a cost is taken. The objective part of the structure
    # check.
    "fc_cost": (
        "rules_singleton_cols.cpp",
        "  if (st.cost[col] != 0.) { return true; }\n",
        "",
    ),
    # The undo rounds an integer column out of its rows. The postsolve part
    # of the structure check; requirement 3 of the crush check; the oracle.
    "fc_round": (
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
        "  integral_form_t f;\n  integral_side(tol, a, side, f);\n",
        "  integral_form_t f;\n  (void)side;\n",
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
            "const single_t& s, double side)\n{",
            "bool implied_free(\n  context_t& c, int32_t row, "
            "const single_t& s, double side, int32_t skip = 0)\n{",
        ),
        (
            "rules_singleton_cols.cpp",
            "  if (upper < inf) {\n",
            "  if (upper < inf && skip != 1) {\n",
        ),
        (
            "rules_singleton_cols.cpp",
            "  if (lower > -inf) {\n    if (!range_end(",
            "  if (lower > -inf && skip != -1) {\n    if (!range_end(",
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
        "  if (!side_readable(c.tol, side)) { return false; }\n"
        "  double free_side = dd_residual(side, scanned.fixed_terms);",
        "  if (!equation && std::isinf(side)) {\n"
        "    side = side > 0 ? st.row_lower[row] : st.row_upper[row];\n"
        "  }\n"
        "  if (!side_readable(c.tol, side)) { return false; }\n"
        "  double free_side = dd_residual(side, scanned.fixed_terms);",
    ),
    # An infinite bound of a zero-cost singleton column adds nothing to
    # the range of its term, so the new side stays finite: a restriction.
    # The oracle. Not the structure check.
    "zs_infinite": (
        "rules_singleton_cols.cpp",
        "    if (std::isinf(at_most)) {\n      most_infinite = true;\n"
        "    } else {",
        "    if (std::isinf(at_most)) {\n    } else {",
    ),
    # The record of the zero-cost singleton columns holds the stored
    # sides, with the terms of the fixed columns in them. The kept-rows
    # and columns parts of the structure check; the oracle.
    "zs_stored": (
        "rules_singleton_cols.cpp",
        "  saved.lower = free_lower;\n  saved.upper = free_upper;",
        "  saved.lower = lower;\n  saved.upper = upper;",
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
        "{ st.set_row_upper(row, new_upper); }\n",
        "",
    ),
    # The new lower side is taken from the side less the fixed terms
    # although the terms stay in the row. The columns part of the
    # structure check; the oracle.
    "zs_fixed": (
        "rules_singleton_cols.cpp",
        "  double new_lower  = most_infinite ? -inf : "
        "dd_residual(lower, most);",
        "  double new_lower  = most_infinite\n"
        "                        ? -inf\n"
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
        "    for (int32_t k = r.rows_begin; k < r.rows_end; ++k) {\n"
        "      const saved_row_t& row = ps.saved_rows[k];\n"
        "      double a               = ",
        "    for (int32_t k = r.rows_begin; k < r.rows_begin + 1; ++k) {\n"
        "      const saved_row_t& row = ps.saved_rows[k];\n"
        "      double a               = ",
    ),
    # The saved rows of a forcing column keep the stored sides. The
    # kept-rows part of the structure check; the oracle.
    "fc_stored": (
        "rules_singleton_cols.cpp",
        "    saved.push_back(copy);\n    values[kept++] = values[k];",
        "    copy.lower = st.row_lower[row];\n"
        "    copy.upper = st.row_upper[row];\n"
        "    saved.push_back(copy);\n    values[kept++] = values[k];",
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
        "    st.row_frontier.take(row_list::singleton_cols, w.rows);\n"
        "    if (w.cols.empty()",
        "    st.row_frontier.take(row_list::singleton_cols, w.rows);\n"
        "    w.rows.clear();\n"
        "    if (w.cols.empty()",
    ),
    # The pass counter stands still: a row is visited once in a presolve.
    # The second presolve of the campaign.
    "ev_stamp": (
        "rules_singleton_cols.cpp",
        "    ++w.pass;\n",
        "    w.pass = 1;\n",
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
        "  if (!free_side_readable(c.tol, free_side)) { return false; }\n"
        "  double q = cost / s.a;",
        "  double q = cost / s.a;",
    ),
    "zs_guard": (
        "rules_singleton_cols.cpp",
        "      (new_upper < inf && !free_side_readable(c.tol, new_upper)) "
        "||\n      !free_side_readable(c.tol, free_lower) || "
        "!free_side_readable(c.tol, free_upper)) {",
        "      (new_upper < inf && !free_side_readable(c.tol, new_upper))) {",
    ),
    "fc_guard": (
        "rules_singleton_cols.cpp",
        "  if (!usable) {\n    pool.col.resize(pool_begin);",
        "  if (!usable && c.opt.max_rounds < 0) {\n"
        "    pool.col.resize(pool_begin);",
    ),
    # No drift guard for an integer free singleton column
    # (columns_singleton_integer_column_and_the_drift), and none in the
    # alias forest (alias_integer_edge_needs_integral_scale_and_offset).
    "fs_drift": (
        "rules_singleton_cols.cpp",
        "    if (!form.integral ||\n"
        "        !drift_within(c.tol, c.tol.integral_drift_share, s.a, form, "
        "drift_units_t::row)) {",
        "    if (!form.integral) {",
    ),
    "r_alias_drift": (
        "alias_forest.cpp",
        "form.integral &&\n      drift_within(st.tol, "
        "st.tol.integral_drift_share, child_coef, form, drift_units_t::row);",
        "form.integral;",
    ),
    # The side that is taken as integral and the drift each keep their own
    # limit and not the tolerance together
    # (columns_review_integral_side_and_drift_share_the_tolerance,
    # alias_integer_edge_needs_integral_scale_and_offset).
    "fs_shared": (
        "integral.hpp",
        "  return size * f.drift <= limit && "
        "f.side_error + size * f.drift <= tol.feasibility;",
        "  return size * f.drift <= limit;",
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
        "    if (!usable) { c.w.unreadable[row] = 1; }\n",
        "",
    ),
    # Which of two parallel rows is kept. The row with the smaller index
    # stays whatever the sizes, as before the larger row was kept: what a
    # later rule rounds in the kept row reaches a row that is a multiple of
    # it times the ratio of the two (the verdict sides of the campaign
    # profile scale_pair;
    # parallel_larger_row_is_kept_before_a_singleton_column_leaves).
    "pr_keep_index": (
        "rules_parallel_rows.cpp",
        "  if (keep_larger && size_a != size_b) { return size_a > size_b; }",
        "  if (keep_larger && size_a != size_b && false) "
        "{ return size_a > size_b; }",
    ),
    # The fixed terms of the two rows change places in the image of a
    # side: those of the kept row are taken off the side of the removed
    # row, and those of the removed row are added.
    "pr_fixed_roles": (
        "rules_parallel_rows.cpp",
        "    return image_of(side, removed.fixed_terms, kept.fixed_terms, "
        "kept.first, removed.first, tol);",
        "    return image_of(side, kept.fixed_terms, removed.fixed_terms, "
        "kept.first, removed.first, tol);",
    ),
    # Sides that cross are measured in the units of the kept row: a pair
    # whose removed row holds within its tolerance at the side of the kept
    # row is reported infeasible
    # (parallel_crossing_in_the_units_of_the_removed_row). The generated
    # models have no crossing below one unit of the smaller row, so no
    # campaign check sees it.
    "pr_cross_kept": [
        (
            "rules_parallel_rows.cpp",
            "    if (scale * (crossing - error) - drift > tol.feasibility) "
            "{ return merge_t::infeasible; }",
            "    if ((crossing - error) - drift > tol.feasibility) "
            "{ return merge_t::infeasible; }",
        ),
        (
            "rules_parallel_rows.cpp",
            "    if (scale * (crossing + error) + drift <= tol.feasibility) "
            "{ return merge_t::made; }",
            "    if ((crossing + error) + drift <= tol.feasibility) "
            "{ return merge_t::made; }\n    (void)scale;",
        ),
    ],
    # Rows that are parallel within tol.zero are merged whatever they
    # differ by over the bounds of their columns
    # (parallel_rows_that_are_not_exact_multiples). The generated rows are
    # exact multiples, so no campaign check sees it.
    "pr_no_drift": (
        "rules_parallel_rows.cpp",
        "          if (!(drift <= opt.parallel_drift_share * tol.feasibility)) {",
        "          if (false) {",
    ),
}


def main():
    """Write the mutants and return the exit code."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    for name in sys.argv[2:] or MUTANTS:
        edits = MUTANTS[name]
        target = out / name / "src"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(PROJECT_DIR / "src", target)
        for file, old, new in [edits] if isinstance(edits, tuple) else edits:
            text = (target / file).read_text()
            if text.count(old) != 1:
                print(
                    "%s: the text to replace occurs %d times in %s"
                    % (name, text.count(old), file)
                )
                return 1
            (target / file).write_text(text.replace(old, new))
        print("%s: %s" % (name, target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
