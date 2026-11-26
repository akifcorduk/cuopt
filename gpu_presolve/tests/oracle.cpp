/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "oracle.hpp"

#include <gprs/assert.hpp>

namespace gprs {
namespace oracle {

rng_t::rng_t(uint64_t) : state(0) { gprs_assert(false && "not implemented"); }

uint64_t rng_t::next()
{
  gprs_assert(false && "not implemented");
  return {};
}

int32_t rng_t::uniform(int32_t, int32_t)
{
  gprs_assert(false && "not implemented");
  return {};
}

bool rng_t::chance(double)
{
  gprs_assert(false && "not implemented");
  return {};
}

case_t generate(uint64_t, const generator_options_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

bool classify(const model_t&, int32_t, std::vector<uint8_t>&)
{
  gprs_assert(false && "not implemented");
  return {};
}

solution_t solve(const model_t&, const std::vector<uint8_t>&, int32_t)
{
  gprs_assert(false && "not implemented");
  return {};
}

verdict_t check_presolve(const case_t&, const options_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

}  // namespace oracle
}  // namespace gprs
