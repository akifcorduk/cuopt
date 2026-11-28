/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Support of the tests of the singleton-column rules: one call of the family on the state of a
// small model, with what it leaves in the state, on the stack and for the structure check.

#include "../check/structure_check.hpp"
#include "harness.hpp"
#include "test_rows_support.hpp"

#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace columns_test {

using namespace gprs;
using gprs_check::structure_report_t;
using rule_test::col_t;
using rule_test::row_t;

inline const col_t binary{0., 1., true};
inline const col_t unit{0., 1., false};
inline const col_t free_col{-inf, inf, false};

// One call of the family on the state of a model, and what is left afterwards.
struct call_t {
  model_t model;
  state_t st;
  postsolve_t ps;
  options_t opt;
  work_budget_t budget;
  singleton_cols_t family;
  rule_result_t result;
  model_t reduced;

  call_t(const std::vector<col_t>& cols,
         const std::vector<row_t>& rows,
         const std::vector<double>& cost = {})
    : model(with_cost(rule_test::make_model(cols, rows), cost)), st(model, tolerances_t())
  {
    GPRS_CHECK(well_formed(model));
    ps.init(model);
    budget.limit       = rule_test::unlimited;
    budget.stall_limit = rule_test::unlimited;
  }

  static model_t with_cost(model_t m, const std::vector<double>& cost)
  {
    if (!cost.empty()) { m.cost = cost; }
    return m;
  }

  // Runs the family and reads the reduced model.
  call_t& run()
  {
    result = family.run(st, budget, ps, opt);
    GPRS_CHECK_EQ(st.consistency_error(), std::string());
    GPRS_CHECK(st.col_frontier.empty(col_list::singleton_cols) || result.stopped);
    GPRS_CHECK(st.row_frontier.empty(row_list::singleton_cols) || result.stopped);
    reduced = reduced_model(st, ps.row_origin, ps.col_origin);
    return *this;
  }

  // The values of the original columns that the stack gives for values of the columns that
  // are left, named by their original index.
  std::vector<double> postsolved(const std::vector<std::pair<int32_t, double>>& left) const
  {
    std::vector<double> x_reduced(ps.col_origin.size(), 0.), x;
    for (const auto& value : left) {
      bool found = false;
      for (size_t k = 0; k < ps.col_origin.size(); ++k) {
        if (ps.col_origin[k] == value.first) {
          x_reduced[k] = value.second;
          found        = true;
        }
      }
      GPRS_CHECK(found);
    }
    ps.postsolve(x_reduced, x);
    return x;
  }

  // The structure check of what the call did, as the result of a presolve.
  structure_report_t structure() const
  {
    result_t r;
    r.status        = status_t::reduced;
    r.reduced       = reduced;
    r.postsolve     = ps;
    r.dual_findings = result.dual_findings;
    return gprs_check::structure_check(model, r, opt.primal_only);
  }
};

inline bool silent(const structure_report_t& s)
{
  return s.composed && s.lost_integers == 0 && s.bound_failures == 0 && s.row_failures == 0 &&
         s.kept_failures == 0 && s.added_failures == 0 && s.box_failures == 0 &&
         s.integral_failures == 0 && s.snapshot_failures == 0 && s.objective_failures == 0 &&
         s.postsolve_failures == 0;
}

// Row activity of the model at x.
inline double activity(const model_t& m, int32_t row, const std::vector<double>& x)
{
  dd_t sum;
  for (int64_t e = m.row_start[row]; e < m.row_start[row + 1]; ++e) {
    dd_add_product(sum, m.value[e], x[m.col_index[e]]);
  }
  return dd_value(sum);
}

inline double objective(const model_t& m, const std::vector<double>& x)
{
  dd_t sum{m.objective_offset, 0.};
  for (int32_t j = 0; j < m.ncols; ++j) {
    dd_add_product(sum, m.cost[j], x[j]);
  }
  return dd_value(sum);
}

}  // namespace columns_test
