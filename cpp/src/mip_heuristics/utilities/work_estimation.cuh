/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstddef>

namespace cuopt::linear_programming::detail {

/**
 * @brief Tunable coefficients for the approximate GPU-kernel work-unit model.
 *
 * These are the "formulation" knobs: they define how a kernel launch is converted into
 * work units as a function of access pattern and problem dimensions. They are pure
 * scalars (no wall clock, no device queries) so the estimate is deterministic and
 * reproducible. The work unit is scaled to be roughly comparable to the CPUFJ
 * byte-touch unit (~1e10 byte-touches) via unit_divisor, so budgets are comparable
 * across components.
 */
// Coarse default pass count for a bounds-propagation "round" when the exact iteration count
// is not readily available. Only affects work-unit accounting magnitude (tunable later).
inline constexpr int default_bounds_prop_work_iters = 50;

struct kernel_work_coeffs_t {
  double nnz_coeff{1.0};        // weight on nonzeros (the dominant sparse-pattern term)
  double var_coeff{0.0};        // weight on number of variables
  double con_coeff{0.0};        // weight on number of constraints
  double unit_divisor{1.0e10};  // converts estimated byte-touches into work units
};

/**
 * @brief Approximate, deterministic work-unit estimate for a single sparse-pattern
 * kernel launch.
 *
 * Models a kernel that touches roughly @p bytes_per_pass bytes per nonzero, plus
 * optional per-variable and per-constraint terms. Pure function of its inputs.
 */
__host__ inline double estimate_sparse_kernel_work(std::size_t nnz,
                                                   std::size_t n_vars,
                                                   std::size_t n_constraints,
                                                   double bytes_per_pass,
                                                   const kernel_work_coeffs_t& c) noexcept
{
  const double touches = c.nnz_coeff * (double)nnz * bytes_per_pass + c.var_coeff * (double)n_vars +
                         c.con_coeff * (double)n_constraints;
  return touches / c.unit_divisor;
}

// Approximate work for an iterative sparse op (e.g. relaxed LP, bounds propagation) that runs
// @p iterations passes over the constraint matrix. Deterministic given the inputs.
__host__ inline double estimate_iterative_op_work(std::size_t nnz,
                                                  std::size_t n_vars,
                                                  std::size_t n_constraints,
                                                  double bytes_per_pass,
                                                  int iterations,
                                                  const kernel_work_coeffs_t& c) noexcept
{
  const double iters = iterations > 0 ? (double)iterations : 1.0;
  return estimate_sparse_kernel_work(nnz, n_vars, n_constraints, bytes_per_pass, c) * iters;
}

}  // namespace cuopt::linear_programming::detail
