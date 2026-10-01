/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/pre_presolve_primal.cuh>
#include <mip_heuristics/utils.cuh>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip::test {
namespace {
using host_problem_t = simplex::user_problem_t<int, double>;
using tolerances_t   = mip_solver_settings_t<int, double>::tolerances_t;

host_problem_t one_row_problem(char sense = 'L', double rhs = 100.)
{
  host_problem_t problem(nullptr);
  problem.num_rows = 1;
  problem.num_cols = 1;
  problem.A.resize(1, 1, 1);
  problem.A.col_start    = {0, 1};
  problem.A.i            = {0};
  problem.A.x            = {1.};
  problem.objective      = {1.};
  problem.lower          = {-std::numeric_limits<double>::infinity()};
  problem.upper          = {std::numeric_limits<double>::infinity()};
  problem.rhs            = {rhs};
  problem.row_sense      = {sense};
  problem.var_types      = {simplex::variable_type_t::CONTINUOUS};
  problem.num_range_rows = 0;
  return problem;
}

tolerances_t active_tolerances()
{
  tolerances_t tolerances;
  tolerances.absolute_tolerance    = 1e-6;
  tolerances.relative_tolerance    = 1e-12;
  tolerances.integrality_tolerance = 1e-5;
  return tolerances;
}
}  // namespace

TEST(pre_presolve_validation, enforces_strict_variable_domains)
{
  auto problem          = one_row_problem();
  problem.lower         = {0.};
  problem.upper         = {1.};
  const auto tolerances = active_tolerances();
  EXPECT_TRUE(verify_pre_presolve_primal_solution(problem, tolerances, {0.}));
  EXPECT_TRUE(verify_pre_presolve_primal_solution(problem, tolerances, {1.}));
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {-1e-10}));
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {1. + 1e-10}));
  problem.var_types = {simplex::variable_type_t::BINARY};
  problem.upper     = {2.};
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {2.}));
}

TEST(pre_presolve_validation, uses_configured_integrality_tolerance)
{
  auto problem      = one_row_problem();
  problem.var_types = {simplex::variable_type_t::INTEGER};
  auto tolerances   = active_tolerances();
  EXPECT_TRUE(verify_pre_presolve_primal_solution(problem, tolerances, {1. + 5e-6}));
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {1. + 2e-5}));
  tolerances.integrality_tolerance = 1e-4;
  EXPECT_TRUE(verify_pre_presolve_primal_solution(problem, tolerances, {1. + 2e-5}));
  problem.var_types = {simplex::variable_type_t::CONTINUOUS};
  EXPECT_TRUE(verify_pre_presolve_primal_solution(problem, tolerances, {1.25}));
}

TEST(pre_presolve_validation, uses_configured_absolute_row_tolerance)
{
  auto tolerances               = active_tolerances();
  tolerances.relative_tolerance = 0.;
  for (char sense : std::array<char, 3>{'E', 'G', 'L'}) {
    auto problem           = one_row_problem(sense, 1.);
    const double direction = sense == 'G' ? -1. : 1.;
    EXPECT_TRUE(verify_pre_presolve_primal_solution(
      problem, tolerances, {1. + direction * 0.5 * tolerances.absolute_tolerance}));
    EXPECT_FALSE(verify_pre_presolve_primal_solution(
      problem, tolerances, {1. + direction * 2. * tolerances.absolute_tolerance}));
  }
}

TEST(pre_presolve_validation, uses_configured_relative_row_tolerance)
{
  auto problem                  = one_row_problem('L', 1e8);
  auto tolerances               = active_tolerances();
  tolerances.relative_tolerance = 1e-8;
  const double row_tolerance =
    get_cstr_tolerance<int, double>(-std::numeric_limits<double>::infinity(),
                                    problem.rhs[0],
                                    tolerances.absolute_tolerance,
                                    tolerances.relative_tolerance);
  ASSERT_GT(row_tolerance, tolerances.absolute_tolerance);
  const std::vector<double> candidate{problem.rhs[0] + 0.5 * row_tolerance};
  EXPECT_TRUE(verify_pre_presolve_primal_solution(problem, tolerances, candidate));
  EXPECT_FALSE(verify_pre_presolve_primal_solution(
    problem, tolerances, {problem.rhs[0] + 2. * row_tolerance}));
  tolerances.relative_tolerance = 0.;
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, candidate));
}

TEST(pre_presolve_validation, rejects_wrong_size_nonfinite_values_and_objectives)
{
  auto problem          = one_row_problem();
  const auto tolerances = active_tolerances();
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {}));
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {0., 0.}));
  EXPECT_FALSE(verify_pre_presolve_primal_solution(
    problem, tolerances, {std::numeric_limits<double>::quiet_NaN()}));
  EXPECT_FALSE(verify_pre_presolve_primal_solution(
    problem, tolerances, {std::numeric_limits<double>::infinity()}));
  problem.objective = {std::numeric_limits<double>::max()};
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {2.}));
  problem.objective    = {1.};
  problem.obj_constant = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {0.}));
}

TEST(pre_presolve_validation, interprets_signed_ranges_for_each_row_sense)
{
  auto tolerances               = active_tolerances();
  tolerances.relative_tolerance = 0.;
  for (char sense : std::array<char, 3>{'E', 'G', 'L'}) {
    for (double range : std::array<double, 2>{-2., 2.}) {
      auto problem           = one_row_problem(sense, 3.);
      problem.range_rows     = {0};
      problem.range_value    = {range};
      problem.num_range_rows = 1;
      const double lower     = sense == 'L' || (sense == 'E' && range < 0.) ? 1. : 3.;
      const double upper     = lower + 2.;
      EXPECT_TRUE(verify_pre_presolve_primal_solution(problem, tolerances, {lower}));
      EXPECT_TRUE(verify_pre_presolve_primal_solution(problem, tolerances, {upper}));
      EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {lower - 1e-4}));
      EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, tolerances, {upper + 1e-4}));
    }
  }
}

TEST(pre_presolve_validation, accumulates_each_csc_row)
{
  auto problem     = one_row_problem();
  problem.num_rows = 2;
  problem.num_cols = 2;
  problem.A.resize(2, 2, 4);
  problem.A.col_start = {0, 2, 4};
  problem.A.i         = {0, 1, 0, 1};
  problem.A.x         = {1., 1., 1., -1.};
  problem.objective   = {1., 1.};
  problem.lower       = {0., 0.};
  problem.upper       = {10., 10.};
  problem.rhs         = {3., 1.};
  problem.row_sense   = {'E', 'G'};
  problem.var_types.assign(2, simplex::variable_type_t::CONTINUOUS);
  EXPECT_TRUE(verify_pre_presolve_primal_solution(problem, active_tolerances(), {2., 1.}));
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, active_tolerances(), {1., 2.}));
  EXPECT_FALSE(verify_pre_presolve_primal_solution(problem, active_tolerances(), {3., 1.}));
}

TEST(pre_presolve_eligibility, preserves_inclusive_size_and_team_limits)
{
  EXPECT_EQ(pre_presolve_diving_workers, 3);
  EXPECT_EQ(pre_presolve_task_slots, 4);
  EXPECT_TRUE(is_pre_presolve_primal_eligible(pre_presolve_max_rows,
                                              pre_presolve_max_cols,
                                              pre_presolve_max_nnz,
                                              false,
                                              false,
                                              true,
                                              true,
                                              true,
                                              pre_presolve_min_team_threads));
  EXPECT_TRUE(is_pre_presolve_primal_eligible(
    1, 1, 0, false, false, true, true, true, pre_presolve_min_team_threads));
}

TEST(pre_presolve_eligibility, rejects_out_of_range_or_empty_dimensions)
{
  const auto eligible = [](int rows, int cols, int nnz) {
    return is_pre_presolve_primal_eligible(
      rows, cols, nnz, false, false, true, true, true, pre_presolve_min_team_threads);
  };
  EXPECT_FALSE(eligible(0, 1, 0));
  EXPECT_FALSE(eligible(1, 0, 0));
  EXPECT_FALSE(eligible(-1, 1, 0));
  EXPECT_FALSE(eligible(1, -1, 0));
  EXPECT_FALSE(eligible(1, 1, -1));
  EXPECT_FALSE(eligible(pre_presolve_max_rows + 1, 1, 0));
  EXPECT_FALSE(eligible(1, pre_presolve_max_cols + 1, 0));
  EXPECT_FALSE(eligible(1, 1, pre_presolve_max_nnz + 1));
}

TEST(pre_presolve_eligibility, rejects_unsupported_modes)
{
  const auto eligible = [](bool quadratic_objective,
                           bool quadratic_constraints,
                           bool is_mip,
                           bool presolve,
                           bool opportunistic,
                           int threads) {
    return is_pre_presolve_primal_eligible(1,
                                           1,
                                           0,
                                           quadratic_objective,
                                           quadratic_constraints,
                                           is_mip,
                                           presolve,
                                           opportunistic,
                                           threads);
  };
  const int threads = pre_presolve_min_team_threads;
  EXPECT_FALSE(eligible(true, false, true, true, true, threads));
  EXPECT_FALSE(eligible(false, true, true, true, true, threads));
  EXPECT_FALSE(eligible(false, false, false, true, true, threads));
  EXPECT_FALSE(eligible(false, false, true, false, true, threads));
  EXPECT_FALSE(eligible(false, false, true, true, false, threads));
  EXPECT_FALSE(eligible(false, false, true, true, true, threads - 1));
}
}  // namespace cuopt::mathematical_optimization::mip::test
