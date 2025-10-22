/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/presolve.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class node_presolve_t {
 public:
  // For pure LP bounds strengthening, var_types should be defaulted (i.e. left empty)
  node_presolve_t(const lp_problem_t<i_t, f_t>& problem,
                  const std::vector<char>& row_sense,
                  const csc_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types);

  bool bound_strengthening(std::vector<f_t>& lower_bounds,
                           std::vector<f_t>& upper_bounds,
                           const simplex_solver_settings_t<i_t, f_t>& settings);

  std::vector<bool> bounds_changed;

 private:
  const lp_problem_t<i_t, f_t>& problem;
  const csc_matrix_t<i_t, f_t>& Arow;
  const std::vector<variable_type_t>& var_types;

  std::vector<f_t> delta_min_activity;
  std::vector<f_t> delta_max_activity;
  std::vector<f_t> constraint_lb;
  std::vector<f_t> constraint_ub;
};
}  // namespace cuopt::linear_programming::dual_simplex
