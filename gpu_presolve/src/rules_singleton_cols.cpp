/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <vector>

namespace gprs {

struct singleton_cols_t::work_t {
  std::vector<int32_t> cols;
  std::vector<int32_t> rows;
};

singleton_cols_t::singleton_cols_t() : work(new work_t) {}

singleton_cols_t::~singleton_cols_t() = default;

rule_result_t singleton_cols_t::run(state_t& st, work_budget_t&, postsolve_t&, const options_t&)
{
  st.col_frontier.take(col_list::singleton_cols, work->cols);
  st.row_frontier.take(row_list::singleton_cols, work->rows);
  return rule_result_t();
}

}  // namespace gprs
