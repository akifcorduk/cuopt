/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/rules.hpp>

namespace gprs {

struct bound_propagation_t::work_t {};

bound_propagation_t::bound_propagation_t() : work(new work_t) {}

bound_propagation_t::~bound_propagation_t() = default;

rule_result_t bound_propagation_t::run(state_t&, work_budget_t&, postsolve_t&, const options_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

rule_result_t bound_propagation_t::make_explicit(state_t&, postsolve_t&, const options_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

}  // namespace gprs
