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

struct simple_probing_t::work_t {
  std::vector<int32_t> rows;
};

simple_probing_t::simple_probing_t() : work(new work_t) {}

simple_probing_t::~simple_probing_t() = default;

rule_result_t simple_probing_t::run(state_t& st, work_budget_t&, postsolve_t&, const options_t&)
{
  st.row_frontier.take(row_list::simple_probing, work->rows);
  return rule_result_t();
}

}  // namespace gprs
