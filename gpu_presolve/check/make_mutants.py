# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write deliberately wrong copies of the engine sources, to test the check.

Usage: make_mutants.py <out dir> [name ...]
Each mutant is <out dir>/<name>/src: the engine sources with one exact text
replacement, or a few. The script stops when the text of a replacement is
not found exactly once, so a mutant that the engine has outgrown is noticed.

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
EXPLICIT_BOUND = "    bound        = is_upper ? bound + slack : bound - slack;"
EXPLICIT_SLACK = "    double slack = candidate_slack(k);\n" + EXPLICIT_BOUND

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
    # ... the side is multiplied by the ratio instead of divided.
    "m_parallel_ratio": (
        "rules_parallel_rows.cpp",
        "return free_side / ratio + dd_value(kept_fixed);",
        "return free_side * ratio + dd_value(kept_fixed);",
    ),
    # ... the upper side taken from the removed row is one unit too small.
    "m_parallel_tight": (
        "rules_parallel_rows.cpp",
        MERGE_SWAP,
        MERGE_SWAP + "\n  upper -= 1.;",
    ),
    # ... the sides of the removed row are dropped.
    "m_parallel_loose": (
        "rules_parallel_rows.cpp",
        MERGE_SWAP,
        "lower = -inf;\n  upper = inf;",
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
            "map.admissible = st.is_integer(parent) && "
            "usable(scale, offset) &&",
            "map.admissible = usable(scale, offset) &&",
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
        "return free_side / ratio + dd_value(kept_fixed);",
        "return free_side / ratio;",
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
    # one row in three.
    "r_prop_inf": [
        (
            "rules_propagation.cpp",
            "  if (act.inf_count[row] == 1) {",
            "  if (act.inf_count[row] == 1 && row % 3 != 0) {",
        ),
        (
            "rules_propagation.cpp",
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
        "  ps.push_row_addition(target.row, equation.row, factor);\n",
        "  (void)ps;\n",
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
