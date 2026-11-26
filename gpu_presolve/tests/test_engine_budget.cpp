/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"

#include <gprs/presolve.hpp>

using namespace gprs;

namespace {

// rows x cols, every row with the entries (i, i) and (i, (i + 1) % cols).
model_t two_entry_rows(int32_t rows, int32_t cols)
{
  model_t m;
  m.nrows = rows;
  m.ncols = cols;
  m.row_start.push_back(0);
  for (int32_t i = 0; i < rows; ++i) {
    int32_t a = i % cols, b = (i + 1) % cols;
    m.col_index.push_back(a < b ? a : b);
    m.col_index.push_back(a < b ? b : a);
    m.value.push_back(1.);
    m.value.push_back(1.);
    m.row_start.push_back(m.value.size());
  }
  m.row_lower.assign(rows, 0.);
  m.row_upper.assign(rows, 1.);
  m.col_lower.assign(cols, 0.);
  m.col_upper.assign(cols, 1.);
  m.cost.assign(cols, 0.);
  m.col_type.assign(cols, 0);
  return m;
}

}  // namespace

GPRS_TEST(engine_budget_stops_on_limit_and_on_stall)
{
  work_budget_t budget;
  budget.limit       = 100;
  budget.stall_limit = 30;
  GPRS_CHECK(!budget.stop());
  budget.spend(29);
  GPRS_CHECK(!budget.stop());
  budget.spend(1);
  GPRS_CHECK(budget.stop());
  budget.found();
  GPRS_CHECK(!budget.stop());
  budget.spend(29);
  budget.found();
  budget.spend(29);
  GPRS_CHECK(!budget.stop());
  budget.spend(12);
  GPRS_CHECK_EQ(budget.used, 100);
  GPRS_CHECK(budget.stop());
  budget.found();
  GPRS_CHECK(budget.stop());
}

GPRS_TEST(engine_budget_from_one_pass)
{
  model_t m = two_entry_rows(1000, 7);
  GPRS_REQUIRE(well_formed(m));
  state_t st(m, tolerances_t());
  int64_t pass = 2000 + line_visit_work * (1000 + 7);
  GPRS_CHECK_EQ(pass, 10056);

  work_budget_t budget = make_budget(budget_t{16., 2., 100000}, st);
  GPRS_CHECK_EQ(budget.limit, 100000 + 16 * pass);
  GPRS_CHECK_EQ(budget.stall_limit, 100000 + 2 * pass);
  GPRS_CHECK_EQ(budget.used, 0);
  GPRS_CHECK_EQ(budget.used_at_finding, 0);

  work_budget_t fraction = make_budget(budget_t{0.5, 0.25, 0}, st);
  GPRS_CHECK_EQ(fraction.limit, 5028);
  GPRS_CHECK_EQ(fraction.stall_limit, 2514);
}

// One visit of every row of a model with two entries per row, without a finding, fits into
// the stall limit of every family's default budget with the floor taken away.
GPRS_TEST(engine_budget_default_stall_covers_one_pass)
{
  model_t m = two_entry_rows(50000, 50000);
  state_t st(m, tolerances_t());
  int64_t row_pass = st.free_nnz() + line_visit_work * st.live_rows;
  int64_t col_pass = st.free_nnz() + line_visit_work * st.live_cols;
  options_t options;
  for (int32_t r = 0; r < rule::count; ++r) {
    budget_t b           = options.budget[r];
    b.floor              = 0;
    work_budget_t budget = make_budget(b, st);
    GPRS_CHECK(budget.stall_limit >= row_pass);
    GPRS_CHECK(budget.stall_limit >= col_pass);
    GPRS_CHECK(budget.limit >= budget.stall_limit);
  }
}
