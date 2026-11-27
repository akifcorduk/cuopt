/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <vector>

namespace gprs {

struct dual_fixing_t::work_t {
  std::vector<int32_t> cols;
};

dual_fixing_t::dual_fixing_t() : work(new work_t) {}

dual_fixing_t::~dual_fixing_t() = default;

rule_result_t dual_fixing_t::run(state_t& st,
                                 work_budget_t& budget,
                                 postsolve_t& ps,
                                 const options_t& opt)
{
  gprs_assert(!opt.primal_only);
  rule_result_t result;
  std::vector<int32_t>& cols = work->cols;
  st.col_frontier.take(col_list::dual_fixing, cols);
  size_t k = 0;
  for (; k < cols.size() && !budget.stop(); ++k) {
    int32_t col = cols[k];
    if (!st.is_live(col)) { continue; }
    budget.spend(line_visit_work);
    double cost   = st.cost[col];
    bool to_lower = cost >= 0 && st.down_locks[col] == 0;
    bool to_upper = cost <= 0 && st.up_locks[col] == 0;
    if (!to_lower && !to_upper) { continue; }
    double lower = st.col_lower[col];
    double upper = st.col_upper[col];
    double value;
    if (to_lower && to_upper) {
      value = lower > -inf ? lower : upper < inf ? upper : 0.;
    } else {
      value = to_lower ? lower : upper;
      if (value == -inf || value == inf) {
        if (cost != 0) {
          result.outcome = rule_outcome_t::unbounded_or_infeasible;
          return result;
        }
        continue;
      }
    }
    budget.spend(st.csc.used[col]);
    fix_column(st, ps, col, value);
    budget.found();
    ++result.findings;
    ++result.dual_findings;
  }
  result.stopped = k < cols.size();
  for (; k < cols.size(); ++k) {
    st.col_frontier.push(list_bit(col_list::dual_fixing), cols[k]);
  }
  return result;
}

}  // namespace gprs
