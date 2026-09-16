/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../internal.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct row_repair_move_t {
  f_t effect;
  i_t var;
  f_t coeff;
  f_t new_val;
};

template <typename i_t, typename f_t>
void collect_row_repair_moves(fj_cpu_climber_t<i_t, f_t>& c,
                              i_t row_begin,
                              i_t row_end,
                              f_t direction,
                              f_t tolerance,
                              std::vector<row_repair_move_t<i_t, f_t>>& out);

template <typename i_t, typename f_t>
void apply_greedy_covering_seed(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_aggressive_constraint_seed(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_exact_k_seed(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_exact_one_repair_seed(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_ordinal_midpoint_seed(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void repair_difficult_anchor(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_bipartite_matching_seed(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
bool apply_cumulative_chain_seed(fj_cpu_climber_t<i_t, f_t>& c, uint32_t seed);

template <typename i_t, typename f_t>
void apply_precedence_completion_seed(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_affine_equality_seed(fj_cpu_climber_t<i_t, f_t>& c, double budget);

template <typename i_t, typename f_t>
void apply_unit_commitment_seed(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
bool apply_fixed_charge_network_seed(fj_cpu_climber_t<i_t, f_t>& c, double budget);

template <typename i_t, typename f_t>
bool apply_pmedian_seed(fj_cpu_climber_t<i_t, f_t>& c, double budget);

template <typename i_t, typename f_t>
void apply_structural_completion_seed(fj_cpu_climber_t<i_t, f_t>& c);

}  // namespace cuopt::mathematical_optimization::mip

