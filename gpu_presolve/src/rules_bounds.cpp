/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/rules.hpp>

namespace gprs {

bound_change_t tighten_lower(state_t&, postsolve_t&, int32_t, double)
{
  gprs_assert(false && "not implemented");
  return {};
}

bound_change_t tighten_upper(state_t&, postsolve_t&, int32_t, double)
{
  gprs_assert(false && "not implemented");
  return {};
}

void fix_column(state_t&, postsolve_t&, int32_t, double)
{
  gprs_assert(false && "not implemented");
}

}  // namespace gprs
