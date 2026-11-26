/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/rules.hpp>

namespace gprs {

struct row_tightening_t::work_t {};

row_tightening_t::row_tightening_t() : work(new work_t) {}

row_tightening_t::~row_tightening_t() = default;

rule_result_t row_tightening_t::run(state_t&, work_budget_t&, postsolve_t&, const options_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

}  // namespace gprs
