/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <linear_algebra/sort_csr.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <presolve/presolve_budget_policy.hpp>
#include <presolve/third_party_presolve.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

namespace cuopt::mathematical_optimization::test {

// gprs removes the fixed variable f, the singleton row cap, the row twice, which is parallel to
// cover, and y through the doubleton equation link.
static io::mps_data_model_t<int, double> gprs_test_problem(bool maximize)
{
  auto problem = io::read_lp_from_string<int, double>(R"LP(
Minimize
 obj: 3 x + 2 y + 4 z + w + 5 f - 2 v
Subject To
 link: -2 x + y = 1
 cover: x + z + w >= 3
 pack: 2 z + 3 w + v <= 9
 cap: 4 z <= 12
 twice: 2 x + 2 z + 2 w >= 4
Bounds
 0 <= x <= 4
 1 <= y <= 9
 0 <= z <= 5
 0 <= w <= 3
 f = 2
 0 <= v <= 6
General
 x y z w v
End
)LP");
  problem.set_maximize(maximize);
  problem.set_objective_offset(7.25);
  problem.set_objective_scaling_factor(2.5);
  return problem;
}

static double user_objective(const io::mps_data_model_t<int, double>& problem,
                             const std::vector<double>& solution)
{
  const auto& objective_coefficients = problem.get_objective_coefficients();
  return problem.get_objective_scaling_factor() *
         std::inner_product(solution.begin(),
                            solution.end(),
                            objective_coefficients.begin(),
                            problem.get_objective_offset());
}

// Instance, mip_hyper_params.presolve_gprs.
class gprs_presolve_solve_test : public ::testing::TestWithParam<std::tuple<std::string, int>> {};

TEST_P(gprs_presolve_solve_test, same_objective_as_without_gprs)
{
  const auto& [instance, mode] = GetParam();
  const raft::handle_t handle{};
  auto problem = io::read_mps<int, double>(make_path_absolute(instance), false);
  handle.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit                     = 20;
  auto reference                          = solve_mip(&handle, problem, settings);
  settings.mip_hyper_params.presolve_gprs = mode;
  auto solution                           = solve_mip(&handle, problem, settings);

  ASSERT_EQ(reference.get_termination_status(), mip_termination_status_t::Optimal);
  ASSERT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), reference.get_objective_value(), 1e-6);
  test_variable_bounds(problem, solution.get_solution(), settings);
  test_constraint_sanity_per_row(problem,
                                 solution.get_solution(),
                                 settings.tolerances.absolute_tolerance,
                                 settings.tolerances.relative_tolerance);
}

// gprs reduces sudoku and fixed-problem to an empty problem, leaves dominating_set as it is and
// tightens bounds of sample.
INSTANTIATE_TEST_SUITE_P(gprs_presolve,
                         gprs_presolve_solve_test,
                         ::testing::Combine(::testing::Values(std::string("mip/sudoku.mps"),
                                                              std::string("mip/fixed-problem.mps"),
                                                              std::string("mip/dominating_set.mps"),
                                                              std::string("mip/sample.mps")),
                                            ::testing::Values(1, 2, 3)));

class gprs_presolve_infeasible_test : public ::testing::TestWithParam<int> {};

TEST_P(gprs_presolve_infeasible_test, infeasible_problem)
{
  const raft::handle_t handle{};
  auto problem =
    io::read_mps<int, double>(make_path_absolute("mip/presolve-infeasible.mps"), false);
  handle.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit                     = 20;
  settings.mip_hyper_params.presolve_gprs = GetParam();
  auto solution                           = solve_mip(&handle, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Infeasible);
}

INSTANTIATE_TEST_SUITE_P(gprs_presolve, gprs_presolve_infeasible_test, ::testing::Values(1, 2, 3));

// mip_hyper_params.presolve_gprs, maximize, dual reductions.
class gprs_presolve_undo_test : public ::testing::TestWithParam<std::tuple<int, bool, bool>> {};

TEST_P(gprs_presolve_undo_test, undo_gives_an_optimal_point_of_the_original_problem)
{
  const auto [mode, maximize, dual_reductions] = GetParam();
  const raft::handle_t handle{};
  const auto problem = gprs_test_problem(maximize);

  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 20;
  auto reference      = solve_mip(&handle, problem, settings);
  ASSERT_EQ(reference.get_termination_status(), mip_termination_status_t::Optimal);

  auto op_problem = mps_data_model_to_optimization_problem(&handle, problem);
  mip::third_party_presolve_t<int, double> presolver;
  presolver.set_gprs_presolve((mip::gprs_presolve_mode_t)mode);
  presolver.set_dual_reductions(dual_reductions);
  auto reduced = presolver.apply_presolve_from_op_problem(
    op_problem, problem_category_t::MIP, presolver_t::Papilo, false, 1e-6, 1e-12, 20., 1);
  ASSERT_TRUE(reduced.status == mip::third_party_presolve_status_t::REDUCED ||
              reduced.status == mip::third_party_presolve_status_t::OPTIMAL);

  // The maps refer to the problem the presolver was given.
  const int n_original = problem.get_n_variables();
  const int n_reduced  = reduced.reduced_problem.get_n_variables();
  ASSERT_LT(n_reduced, n_original);
  ASSERT_EQ(reduced.reduced_to_original_map.size(), n_reduced);
  ASSERT_EQ(reduced.original_to_reduced_map.size(), n_original);
  for (int j = 0; j < n_reduced; ++j) {
    const int original = reduced.reduced_to_original_map[j];
    ASSERT_GE(original, 0);
    ASSERT_LT(original, n_original);
    EXPECT_EQ(reduced.original_to_reduced_map[original], j);
  }

  std::vector<double> solution;
  if (n_reduced > 0) {
    // solve_mip scales the problem it is given.
    optimization_problem_t<int, double> reduced_problem(reduced.reduced_problem);
    settings.presolver  = presolver_t::None;
    auto reduced_result = solve_mip(reduced_problem, settings);
    ASSERT_EQ(reduced_result.get_termination_status(), mip_termination_status_t::Optimal);
    solution = host_copy(reduced_result.get_solution(), handle.get_stream());
  }
  std::vector<double> dual_solution, reduced_costs;
  presolver.undo(solution, dual_solution, reduced_costs, problem_category_t::MIP, false, false);
  ASSERT_EQ(solution.size(), n_original);

  mip::problem_t<int, double> original_problem(op_problem);
  mip::solution_t<int, double> original_solution(original_problem);
  original_solution.copy_new_assignment(solution);
  EXPECT_TRUE(original_solution.compute_feasibility());
  EXPECT_NEAR(user_objective(problem, solution), reference.get_objective_value(), 1e-6);

  const auto& tolerances = settings.tolerances;
  EXPECT_TRUE(presolver.is_original_primal_solution_feasible(solution,
                                                             tolerances.absolute_tolerance,
                                                             tolerances.relative_tolerance,
                                                             tolerances.integrality_tolerance));
  std::vector<double> crushed, uncrushed;
  presolver.crush_primal_solution(reduced.reduced_problem, solution, crushed);
  ASSERT_EQ(crushed.size(), n_reduced);
  presolver.uncrush_primal_solution(crushed, uncrushed);
  ASSERT_EQ(uncrushed.size(), n_original);
  EXPECT_TRUE(presolver.is_original_primal_solution_feasible(uncrushed,
                                                             tolerances.absolute_tolerance,
                                                             tolerances.relative_tolerance,
                                                             tolerances.integrality_tolerance));
  EXPECT_NEAR(user_objective(problem, uncrushed), reference.get_objective_value(), 1e-6);

  // x leaves its upper bound.
  solution[0] = 5;
  EXPECT_FALSE(presolver.is_original_primal_solution_feasible(solution,
                                                              tolerances.absolute_tolerance,
                                                              tolerances.relative_tolerance,
                                                              tolerances.integrality_tolerance));
}

INSTANTIATE_TEST_SUITE_P(gprs_presolve,
                         gprs_presolve_undo_test,
                         ::testing::Combine(::testing::Values(1, 2, 3),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

// Not a test: prints what each presolve mode leaves of the instances listed in the file named by
// GPRS_REDUCTIONS_LIST (one MPS path per line), without solving and without scaling, as the
// solver would call the presolve. Modes: 0 the third-party presolve, 2 gprs alone, 3 gprs and
// cuOpt's own reductions. GPRS_REDUCTIONS_THREADS is the thread count given to Papilo (4),
// GPRS_REDUCTIONS_MODES the modes to run as digits (default "023").
// Run with --gtest_also_run_disabled_tests --gtest_filter='*reductions_report*'.
TEST(gprs_presolve_report, DISABLED_reductions_report)
{
  const char* list = std::getenv("GPRS_REDUCTIONS_LIST");
  ASSERT_NE(list, nullptr);
  const char* threads_text = std::getenv("GPRS_REDUCTIONS_THREADS");
  const int threads        = threads_text ? std::atoi(threads_text) : 4;
  const char* modes_text   = std::getenv("GPRS_REDUCTIONS_MODES");
  const std::string modes  = modes_text ? modes_text : "023";
  const raft::handle_t handle{};
  std::ifstream in(list);
  ASSERT_TRUE(in.good());
  std::string path;
  while (std::getline(in, path)) {
    if (path.empty()) { continue; }
    const auto model = io::read_mps<int, double>(path, false);
    auto op_problem  = mps_data_model_to_optimization_problem(&handle, model);
    sort_csr(op_problem);
    handle.sync_stream();
    mip_solver_settings_t<int, double> settings;
    const auto& hp      = settings.mip_hyper_params;
    const auto features = mip::papilo_presolve_features(op_problem);
    const auto budget   = mip::evaluate_presolve_budget(hp, features);
    for (char digit : modes) {
      const int mode = digit - '0';
      mip::third_party_presolve_t<int, double> presolver;
      presolver.set_indicator_strengthening(hp.presolve_indicator_strengthening);
      presolver.set_dual_reductions(true);
      presolver.set_gprs_presolve((mip::gprs_presolve_mode_t)mode);
      const auto start = std::chrono::steady_clock::now();
      const auto result =
        presolver.apply_presolve_from_op_problem(op_problem,
                                                 problem_category_t::MIP,
                                                 presolver_t::Papilo,
                                                 false,
                                                 settings.tolerances.absolute_tolerance,
                                                 settings.tolerances.relative_tolerance,
                                                 600.,
                                                 threads,
                                                 budget.papilo_max_rounds,
                                                 budget.papilo_max_badgesize);
      const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      const bool has_model = result.status == mip::third_party_presolve_status_t::REDUCED ||
                             result.status == mip::third_party_presolve_status_t::UNCHANGED;
      printf(
        "GPRS_REDUCTIONS path=%s mode=%d status=%d rows=%d cols=%d nnz=%d seconds=%.3f "
        "rows0=%d cols0=%d nnz0=%d\n",
        path.c_str(),
        mode,
        (int)result.status,
        has_model ? result.reduced_problem.get_n_constraints() : 0,
        has_model ? result.reduced_problem.get_n_variables() : 0,
        has_model ? result.reduced_problem.get_nnz() : 0,
        seconds,
        op_problem.get_n_constraints(),
        op_problem.get_n_variables(),
        op_problem.get_nnz());
      fflush(stdout);
    }
  }
}

}  // namespace cuopt::mathematical_optimization::test
