/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/rules.hpp>

namespace gprs {

struct parallel_rows_t::work_t {};

parallel_rows_t::parallel_rows_t() : work(new work_t) {}

parallel_rows_t::~parallel_rows_t() = default;

rule_result_t parallel_rows_t::run(state_t&, work_budget_t&, postsolve_t&, const options_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

}  // namespace gprs
