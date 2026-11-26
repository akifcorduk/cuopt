/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/postsolve.hpp>

namespace gprs {

void postsolve_t::init(const model_t&) { gprs_assert(false && "not implemented"); }

void postsolve_t::push_fixed_col(int32_t, double) { gprs_assert(false && "not implemented"); }

void postsolve_t::push_alias(int32_t, int32_t, int32_t, double, double)
{
  gprs_assert(false && "not implemented");
}

void postsolve_t::push_parallel_row(int32_t, int32_t, double)
{
  gprs_assert(false && "not implemented");
}

void postsolve_t::postsolve(const std::vector<double>&, std::vector<double>&) const
{
  gprs_assert(false && "not implemented");
}

void postsolve_t::crush(const std::vector<double>&, std::vector<double>&) const
{
  gprs_assert(false && "not implemented");
}

point_check_t check_point(const model_t&, const std::vector<double>&)
{
  gprs_assert(false && "not implemented");
  return {};
}

bool point_feasible(const point_check_t&, const tolerances_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

}  // namespace gprs
