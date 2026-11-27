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

}  // namespace gprs
