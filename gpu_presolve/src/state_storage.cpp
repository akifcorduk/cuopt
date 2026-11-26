/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/state.hpp>

namespace gprs {

void state_t::compact_sparse_lines() { gprs_assert(false && "not implemented"); }

void state_t::rebuild() { gprs_assert(false && "not implemented"); }

void state_t::check_consistency() { gprs_assert(false && "not implemented"); }

model_t extract_model(const state_t&, std::vector<int32_t>&, std::vector<int32_t>&)
{
  gprs_assert(false && "not implemented");
  return {};
}

}  // namespace gprs
