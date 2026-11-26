/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>

namespace gprs {

uint64_t model_hash(const model_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

uint64_t state_hash(const state_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

uint64_t postsolve_hash(const postsolve_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

}  // namespace gprs
