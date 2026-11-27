# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write deliberately wrong copies of the engine sources, to test the check.

Usage: make_mutants.py <out dir>
Each mutant is <out dir>/<name>/src: the engine sources with one exact text
replacement. The script stops when the text of a replacement is not found
exactly once, so a mutant that the engine has outgrown is noticed.

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

# name: (file, text, replacement)
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
}


def main():
    """Write the mutants and return the exit code."""
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    for name, (file, old, new) in MUTANTS.items():
        target = out / name / "src"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(PROJECT_DIR / "src", target)
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
