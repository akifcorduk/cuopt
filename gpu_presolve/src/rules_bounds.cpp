/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/rules.hpp>

#include <cmath>

namespace gprs {

bound_change_t tighten_lower(state_t& st, postsolve_t& ps, int32_t col, double lower)
{
  gprs_assert(st.is_live(col));
  gprs_assert(!std::isnan(lower));
  if (st.is_integer(col)) { lower = round_lower_integer(lower, st.tol); }
  if (!(lower > st.col_lower[col])) { return {}; }
  double upper = st.col_upper[col];
  if (lower == inf || lower - upper > st.tol.feasibility) {
    return {rule_outcome_t::infeasible, false};
  }
  if (lower >= upper) {
    fix_column(st, ps, col, upper);
  } else {
    st.set_col_lower(col, lower);
  }
  return {rule_outcome_t::ok, true};
}

bound_change_t tighten_upper(state_t& st, postsolve_t& ps, int32_t col, double upper)
{
  gprs_assert(st.is_live(col));
  gprs_assert(!std::isnan(upper));
  if (st.is_integer(col)) { upper = round_upper_integer(upper, st.tol); }
  if (!(upper < st.col_upper[col])) { return {}; }
  double lower = st.col_lower[col];
  if (upper == -inf || lower - upper > st.tol.feasibility) {
    return {rule_outcome_t::infeasible, false};
  }
  if (upper <= lower) {
    fix_column(st, ps, col, lower);
  } else {
    st.set_col_upper(col, upper);
  }
  return {rule_outcome_t::ok, true};
}

void fix_column(state_t& st, postsolve_t& ps, int32_t col, double value)
{
  gprs_assert(st.is_live(col));
  gprs_assert(std::isfinite(value) && st.col_lower[col] <= value && value <= st.col_upper[col]);
  gprs_assert(!st.is_integer(col) || value == std::round(value));
  st.fix_col(col, value);
  ps.push_fixed_col(col, value);
}

bool activity_fresh(const state_t& st, const activity_t& act, int32_t row)
{
  // A summation counts one update per finite term.
  return act.updates[row] + act.inf_count[row] == (uint32_t)st.csr.size[row];
}

bool refresh_activity(state_t& st, int32_t row)
{
  gprs_assert(st.row_alive[row]);
  double limit = st.tol.zero * st.tol.feasibility;
  bool stale   = (st.min_activity_error(row) > limit && !activity_fresh(st, st.min_act, row)) ||
               (st.max_activity_error(row) > limit && !activity_fresh(st, st.max_act, row));
  if (stale) { st.recompute_row_activity(row); }
  return stale;
}

}  // namespace gprs
