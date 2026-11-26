/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>

namespace gprs {

const char* rule_name(int32_t)
{
  gprs_assert(false && "not implemented");
  return {};
}

const char* status_name(status_t)
{
  gprs_assert(false && "not implemented");
  return {};
}

result_t presolve(const model_t&, const options_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

}  // namespace gprs
