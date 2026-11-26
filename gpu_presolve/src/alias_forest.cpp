/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/postsolve.hpp>

namespace gprs {

void alias_forest_t::init(int32_t) { gprs_assert(false && "not implemented"); }

bool alias_forest_t::add_edge(const alias_edge_t&)
{
  gprs_assert(false && "not implemented");
  return {};
}

void alias_forest_t::freeze(const state_t&, std::vector<alias_map_t>&)
{
  gprs_assert(false && "not implemented");
}

void alias_forest_t::clear() { gprs_assert(false && "not implemented"); }

}  // namespace gprs
