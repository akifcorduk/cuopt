/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <raft/core/handle.hpp>
#include <utilities/logger.hpp>

#include "../../../benchmarks/linear_programming/cuopt/initial_problem_check.hpp"

#include <gtest/gtest.h>

TEST(BenchmarkValidation, UsesConfiguredAbsoluteAndRelativeRowTolerances)
{
  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> problem;
  const std::vector<double> coefficients{1}, lower{1e6}, upper{1e6 + 1}, objective{0};
  const std::vector<int> columns{0}, offsets{0, 1};
  const std::vector<char> types{'C'};
  problem.set_csr_constraint_matrix(coefficients, columns, offsets);
  problem.set_variable_lower_bounds(lower);
  problem.set_variable_upper_bounds(upper);
  problem.set_variable_types(types);
  problem.set_objective_coefficients(objective);
  problem.set_constraint_lower_bounds(lower);
  problem.set_constraint_upper_bounds(lower);
  cuopt::mathematical_optimization::mip_solver_settings_t<int, double>::tolerances_t tolerances;
  tolerances.absolute_tolerance = 1e-7;
  tolerances.relative_tolerance = 2e-8;
  EXPECT_TRUE(verify_solution(problem, {1e6 + .01}, 0, tolerances, 0));
  EXPECT_FALSE(verify_solution(problem, {1e6 + .03}, 0, tolerances, 1));
  tolerances.relative_tolerance = 0;
  EXPECT_FALSE(verify_solution(problem, {1e6 + .01}, 0, tolerances, 2));
}

TEST(BenchmarkValidation, InfiniteBoundsDoNotMakeRowToleranceInfinite)
{
  const double inf = std::numeric_limits<double>::infinity();
  const auto lower = solver_row_limits(1e-7, 2e-8, -1e6, inf);
  EXPECT_DOUBLE_EQ(lower.first, -1e6 - (1e-7 + .02));
  EXPECT_EQ(lower.second, inf);
  const auto upper = solver_row_limits(1e-7, 2e-8, -inf, 1e6);
  EXPECT_EQ(upper.first, -inf);
  EXPECT_DOUBLE_EQ(upper.second, 1e6 + (1e-7 + .02));
}
