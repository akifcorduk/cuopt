/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/local_search/feasibility_pump/feasibility_pump.cuh>
#include <mip_heuristics/mip_scaling_strategy.cuh>
#include <mip_heuristics/solver.cuh>
#include <pdlp/utilities/problem_checking.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/inline_lp_test_utils.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

namespace cuopt::mathematical_optimization::test {

TEST(FeasibilityPumpTest, ExternalSolutionImprovementMargin)
{
  using mip::external_solution_improves_fp_incumbent;

  EXPECT_TRUE(external_solution_improves_fp_incumbent(89.9, 100.0));
  EXPECT_FALSE(external_solution_improves_fp_incumbent(90.0, 100.0));
  EXPECT_FALSE(external_solution_improves_fp_incumbent(90.1, 100.0));
  EXPECT_TRUE(external_solution_improves_fp_incumbent(-110.1, -100.0));
  EXPECT_FALSE(external_solution_improves_fp_incumbent(-110.0, -100.0));
  EXPECT_FALSE(external_solution_improves_fp_incumbent(-109.9, -100.0));
  EXPECT_TRUE(external_solution_improves_fp_incumbent(-2. * mip::OBJECTIVE_EPSILON, 0.0));
  EXPECT_FALSE(external_solution_improves_fp_incumbent(-0.5 * mip::OBJECTIVE_EPSILON, 0.0));
  EXPECT_TRUE(
    external_solution_improves_fp_incumbent(1.0, std::numeric_limits<double>::infinity()));
  EXPECT_TRUE(external_solution_improves_fp_incumbent(1.0, std::numeric_limits<double>::max()));
  EXPECT_FALSE(external_solution_improves_fp_incumbent(std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::infinity()));
}

TEST(FeasibilityPumpTest, ImprovementMarginsAreInvariantToPresolveOffset)
{
  using mip::external_solution_improves_fp_incumbent;
  using mip::fp_objective_improvement_margin;
  for (double offset : {-1000000.0, -1000.0, 0.0, 1000.0, 1000000.0}) {
    SCOPED_TRACE(offset);
    EXPECT_TRUE(external_solution_improves_fp_incumbent(89.0 - offset, 100.0 - offset, offset));
    EXPECT_FALSE(external_solution_improves_fp_incumbent(90.0 - offset, 100.0 - offset, offset));
    EXPECT_TRUE(external_solution_improves_fp_incumbent(-111.0 - offset, -100.0 - offset, offset));
    EXPECT_FALSE(external_solution_improves_fp_incumbent(-110.0 - offset, -100.0 - offset, offset));
    EXPECT_DOUBLE_EQ(fp_objective_improvement_margin(100.0 - offset, offset, 0.001), 0.1);
    EXPECT_DOUBLE_EQ(fp_objective_improvement_margin(-100.0 - offset, offset, 0.001), 0.1);
    EXPECT_DOUBLE_EQ(fp_objective_improvement_margin(-offset, offset, 0.001),
                     mip::OBJECTIVE_EPSILON);
  }
}

TEST(PrimalIntegralTest, TracksImprovingMinimizationIncumbents)
{
  benchmark_info_t benchmark_info;
  benchmark_info.initialize_primal_integral(100.0, 10.0, false);

  benchmark_info.update_primal_integral(200.0, 2.0);
  benchmark_info.update_primal_integral(150.0, 6.0);
  benchmark_info.update_primal_integral(175.0, 7.0);
  benchmark_info.update_primal_integral(100.0, 8.0);
  benchmark_info.finalize_primal_integral(9.0);

  EXPECT_NEAR(benchmark_info.primal_integral, 7.0 / 15.0, 1e-12);
}

TEST(PrimalIntegralTest, HandlesMaximizationAndMissingIncumbents)
{
  benchmark_info_t maximization_info;
  maximization_info.initialize_primal_integral(100.0, 10.0, true);
  maximization_info.update_primal_integral(0.0, 1.0);
  maximization_info.update_primal_integral(50.0, 4.0);
  maximization_info.finalize_primal_integral(10.0);
  EXPECT_NEAR(maximization_info.primal_integral, 0.7, 1e-12);

  benchmark_info_t no_incumbent_info;
  no_incumbent_info.initialize_primal_integral(100.0, 10.0, false);
  no_incumbent_info.finalize_primal_integral(10.0);
  EXPECT_EQ(no_incumbent_info.primal_integral, 2.0);
}

io::mps_data_model_t<int, double> create_std_lp_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: 1.2 x1 + 1.7 x2
Subject To
  c1_ub: x1 + x2 <= 5000
  c1_lb: x1 + x2 >= 0
Bounds
  0 <= x1 <= 3000
  0 <= x2 <= 5000
End
)LP");
}

io::mps_data_model_t<int, double> create_single_var_lp_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: -0.23 x
Subject To
  c1: x = 0
Bounds
  x = 0
End
)LP");
}

io::mps_data_model_t<int, double> create_std_milp_problem(bool maximize)
{
  auto problem = create_std_lp_problem();
  problem.set_maximize(maximize);
  std::vector<char> var_types = {'I', 'C'};
  problem.set_variable_types(var_types);
  return problem;
}

io::mps_data_model_t<int, double> create_single_var_milp_problem(bool maximize)
{
  auto problem = create_single_var_lp_problem();
  problem.set_maximize(maximize);
  std::vector<char> var_types = {'I'};
  problem.set_variable_types(var_types);
  return problem;
}

TEST(FeasibilityPumpTest, ConsumesQueuedFeasibleSolutionWhenPopulationIsInfeasible)
{
  raft::handle_t handle;
  auto model      = create_std_milp_problem(false);
  auto op_problem = mps_data_model_to_optimization_problem(&handle, model);

  mip_solver_settings_t<int, double> settings{};
  mip::problem_t<int, double> problem(op_problem, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_t<int, double> solver(problem, settings, timer_t(5.0));
  mip::diversity_manager_t<int, double> diversity_manager(solver.context);
  solver.context.diversity_manager_ptr = &diversity_manager;
  diversity_manager.population.initialize_population();
  diversity_manager.population.allocate_solutions();

  ASSERT_FALSE(diversity_manager.population.is_feasible());
  diversity_manager.population.add_external_solution(
    {0.0, 0.0}, 0.0, mip::solution_origin_t::BRANCH_AND_BOUND);
  ASSERT_TRUE(diversity_manager.population.solutions_in_external_queue_.load());
  ASSERT_FALSE(diversity_manager.population.is_feasible());

  mip::solution_t<int, double> fp_solution(problem);
  fp_solution.copy_new_assignment(std::vector<double>{1.0, 0.0});
  diversity_manager.ls.fp.timer = timer_t(1.0);

  EXPECT_FALSE(diversity_manager.ls.fp.run_single_fp_descent(fp_solution, 100.0));
  EXPECT_FALSE(diversity_manager.population.solutions_in_external_queue_.load());
  ASSERT_TRUE(diversity_manager.population.is_feasible());
  EXPECT_NEAR(diversity_manager.population.best_feasible().get_objective(), 0.0, 1e-12);
}

TEST(FeasibilityPumpTest, ExternalRestartRefreshesState)
{
  raft::handle_t handle;
  auto model      = create_std_milp_problem(false);
  auto op_problem = mps_data_model_to_optimization_problem(&handle, model);
  mip_solver_settings_t<int, double> settings{};
  mip::problem_t<int, double> problem(op_problem, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_t<int, double> solver(problem, settings, timer_t(5.0));
  mip::diversity_manager_t<int, double> diversity_manager(solver.context);
  solver.context.diversity_manager_ptr = &diversity_manager;
  auto& population                     = diversity_manager.population;
  population.initialize_population();
  population.allocate_solutions();
  population.add_external_solution({0.0, 0.0}, 0.0, mip::solution_origin_t::BRANCH_AND_BOUND);
  // Exercise the restart without spending time on the population sweep.
  diversity_manager.timer = timer_t(0.0);
  auto& ls                = diversity_manager.ls;
  ls.fp.timer             = timer_t(5.0);
  ls.fp.config.alpha      = 0.2;
  ls.fp.best_excess       = 7.0;
  ls.fp.max_n_of_integers = 2;
  ls.fp.last_distances.push_back(3.0);
  mip::solution_t<int, double> solution(problem);
  solution.copy_new_assignment(std::vector<double>{1.0, 0.0});
  solution.compute_feasibility();
  rmm::device_uvector<double> best_solution(solution.assignment, handle.get_stream());
  double best_objective = solution.get_objective();
  ASSERT_TRUE(
    ls.restart_fp_from_external_solution(solution, &population, best_solution, best_objective));
  EXPECT_DOUBLE_EQ(ls.fp.config.alpha, mip::default_alpha);
  EXPECT_TRUE(std::isinf(ls.fp.best_excess));
  EXPECT_EQ(ls.fp.max_n_of_integers, 0);
  EXPECT_TRUE(ls.fp.last_distances.empty());
  EXPECT_DOUBLE_EQ(best_objective, 0.0);
  EXPECT_TRUE(ls.cutting_plane_added_for_active_run);
}

TEST(FeasibilityPumpTest, ExternalRestartAccountsForObjectiveOffset)
{
  for (bool should_restart : {false, true}) {
    SCOPED_TRACE(should_restart);
    raft::handle_t handle;
    auto model      = create_std_milp_problem(false);
    auto op_problem = mps_data_model_to_optimization_problem(&handle, model);
    mip_solver_settings_t<int, double> settings{};
    mip::problem_t<int, double> problem(op_problem, settings.get_tolerances());
    problem.preprocess_problem();
    problem.presolve_data.objective_offset = should_restart ? -1000.0 : 1000.0;
    mip::mip_solver_t<int, double> solver(problem, settings, timer_t(5.0));
    mip::diversity_manager_t<int, double> diversity_manager(solver.context);
    solver.context.diversity_manager_ptr = &diversity_manager;
    auto& population                     = diversity_manager.population;
    population.initialize_population();
    population.allocate_solutions();
    const double candidate = should_restart ? 950.0 : 0.0;
    population.add_external_solution(
      {candidate, 0.0}, 1.2 * candidate, mip::solution_origin_t::BRANCH_AND_BOUND);
    diversity_manager.timer = timer_t(0.0);
    auto& ls                = diversity_manager.ls;
    ls.fp.timer             = timer_t(5.0);
    ls.fp.config.alpha      = 0.2;
    mip::solution_t<int, double> solution(problem);
    const double incumbent = should_restart ? 1000.0 : 1.0;
    solution.copy_new_assignment(std::vector<double>{incumbent, 0.0});
    solution.compute_feasibility();
    rmm::device_uvector<double> best_solution(solution.assignment, handle.get_stream());
    double best_objective = solution.get_objective();
    // User objectives: 200 -> 140 (30%) or 1001.2 -> 1000 (less than 10%).
    EXPECT_EQ(
      ls.restart_fp_from_external_solution(solution, &population, best_solution, best_objective),
      should_restart);
    EXPECT_DOUBLE_EQ(ls.fp.config.alpha, should_restart ? mip::default_alpha : 0.2);
    EXPECT_NEAR(best_objective, 1.2 * (should_restart ? candidate : incumbent), 1e-12);
  }
}

TEST(FeasibilityPumpTest, ObjectiveCutAccountsForObjectiveOffset)
{
  for (bool initial_cut : {false, true}) {
    SCOPED_TRACE(initial_cut);
    for (double offset : {1000.0, -1000.0}) {
      SCOPED_TRACE(offset);
      raft::handle_t handle;
      auto model      = create_std_milp_problem(false);
      auto op_problem = mps_data_model_to_optimization_problem(&handle, model);
      mip_solver_settings_t<int, double> settings{};
      mip::problem_t<int, double> problem(op_problem, settings.get_tolerances());
      problem.preprocess_problem();
      problem.presolve_data.objective_offset = offset;
      mip::mip_solver_t<int, double> solver(problem, settings, timer_t(5.0));
      mip::diversity_manager_t<int, double> diversity_manager(solver.context);
      solver.context.diversity_manager_ptr = &diversity_manager;
      mip::solution_t<int, double> solution(problem);
      solution.copy_new_assignment(std::vector<double>{1.0, 0.0});
      solution.compute_feasibility();
      rmm::device_uvector<double> best_solution(solution.assignment, handle.get_stream());
      double best_objective = std::numeric_limits<double>::infinity();
      auto& ls              = diversity_manager.ls;
      if (initial_cut) {
        auto& population = diversity_manager.population;
        population.initialize_population();
        population.allocate_solutions();
        ls.run_fp(solution, timer_t(0.0), &population);
      } else {
        ls.save_solution_and_add_cutting_plane(solution, best_solution, best_objective);
      }
      const auto& cut_problem = ls.problem_with_objective_cut;
      const double rhs = cut_problem.constraint_upper_bounds.element(cut_problem.n_constraints - 1,
                                                                     handle.get_stream());
      const double expected_rhs = 1.2 - 0.001 * std::abs(1.2 + offset);
      EXPECT_NEAR(rhs, expected_rhs, 1e-12);
    }
  }
}

TEST(PopulationTest, TopSolutionsSnapshot)
{
  raft::handle_t handle;
  auto model      = create_std_milp_problem(false);
  auto op_problem = mps_data_model_to_optimization_problem(&handle, model);

  mip_solver_settings_t<int, double> settings{};
  mip::problem_t<int, double> problem(op_problem, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_t<int, double> solver(problem, settings, timer_t(5.0));
  mip::diversity_manager_t<int, double> diversity_manager(solver.context);
  auto& population = diversity_manager.population;
  EXPECT_TRUE(population.get_top_solutions(2).empty());
  population.initialize_population();
  population.allocate_solutions();
  population.var_threshold = 0;
  EXPECT_TRUE(population.get_top_solutions(2).empty());

  auto add_solution = [&](double value) {
    mip::solution_t<int, double> solution(problem);
    solution.copy_new_assignment(std::vector<double>{value, 0.0});
    solution.compute_feasibility();
    population.add_solution(std::move(solution));
  };

  add_solution(3.0);
  ASSERT_TRUE(population.is_feasible());
  ASSERT_EQ(population.get_top_solutions(2).size(), 1);
  EXPECT_TRUE(population.get_top_solutions(0).empty());

  add_solution(1.0);
  add_solution(2.0);
  ASSERT_EQ(population.current_size(), 3);
  auto top_solutions = population.get_top_solutions(2);
  ASSERT_EQ(top_solutions.size(), 2);
  EXPECT_NEAR(top_solutions[0].get_objective(), 1.2, 1e-12);
  EXPECT_NEAR(top_solutions[1].get_objective(), 2.4, 1e-12);
  EXPECT_EQ(population.get_top_solutions(10).size(), 3);
  // The full snapshot still includes the archived best feasible solution.
  EXPECT_EQ(population.population_to_vector().size(), 4);

  // Recombination may change the population while iterating over its snapshot.
  add_solution(0.0);
  EXPECT_NEAR(population.best().get_objective(), 0.0, 1e-12);
  EXPECT_NEAR(top_solutions[0].get_objective(), 1.2, 1e-12);
  EXPECT_NEAR(top_solutions[1].get_objective(), 2.4, 1e-12);

  population.clear();
  EXPECT_TRUE(population.get_top_solutions(2).empty());
  add_solution(-3.0);
  add_solution(-1.0);
  add_solution(-2.0);
  ASSERT_FALSE(population.is_feasible());
  ASSERT_EQ(population.current_size(), 3);
  auto infeasible_solutions = population.get_top_solutions(2);
  ASSERT_EQ(infeasible_solutions.size(), 2);
  EXPECT_FALSE(infeasible_solutions[0].get_feasible());
  EXPECT_FALSE(infeasible_solutions[1].get_feasible());
  EXPECT_DOUBLE_EQ(infeasible_solutions[0].get_objective(),
                   population.solution_at_index(1).get_objective());
  EXPECT_DOUBLE_EQ(infeasible_solutions[1].get_objective(),
                   population.solution_at_index(2).get_objective());
}

TEST(LPTest, TestSampleLP2)
{
  raft::handle_t handle;

  // Two identical row constraints exercise duplicate-row handling.
  auto problem = cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: x
Subject To
  c1: x <= 1
  c2: x <= 1
End
)LP");

  cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double> settings{};
  settings.set_optimality_tolerance(1e-2);
  settings.method     = cuopt::mathematical_optimization::method_t::PDLP;
  settings.time_limit = 5;

  // Solve
  auto result = cuopt::mathematical_optimization::solve_lp(&handle, problem, settings);

  // Check results
  EXPECT_EQ(result.get_termination_status(),
            cuopt::mathematical_optimization::pdlp_termination_status_t::Optimal);
  ASSERT_EQ(result.get_primal_solution().size(), 1);

  // Copy solution to host to access values
  auto primal_host = cuopt::host_copy(result.get_primal_solution(), handle.get_stream());
  EXPECT_NEAR(primal_host[0], 0.0, 1e-6);

  EXPECT_NEAR(result.get_additional_termination_information().primal_objective, 0.0, 1e-6);
  EXPECT_NEAR(result.get_additional_termination_information().dual_objective, 0.0, 1e-6);
}

TEST(LPTest, TestSampleLP)
{
  raft::handle_t handle;
  auto problem = create_std_lp_problem();

  cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double> settings{};
  settings.set_optimality_tolerance(1e-4);
  settings.time_limit = 5;
  settings.presolver  = cuopt::mathematical_optimization::presolver_t::None;

  auto result = cuopt::mathematical_optimization::solve_lp(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::mathematical_optimization::pdlp_termination_status_t::Optimal);
}

TEST(ErrorTest, TestError)
{
  raft::handle_t handle;
  auto problem = create_std_milp_problem(false);

  cuopt::mathematical_optimization::mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 5;
  settings.presolver  = cuopt::mathematical_optimization::presolver_t::None;

  // Set constraint bounds
  std::vector<double> lower_bounds = {1.0};
  std::vector<double> upper_bounds = {1.0, 1.0};
  problem.set_constraint_lower_bounds(lower_bounds);
  problem.set_constraint_upper_bounds(upper_bounds);

  auto result = cuopt::mathematical_optimization::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::mathematical_optimization::mip_termination_status_t::NoTermination);
}

class MILPTestParams
  : public testing::TestWithParam<
      std::tuple<bool, int, bool, cuopt::mathematical_optimization::mip_termination_status_t>> {};

TEST_P(MILPTestParams, TestSampleMILP)
{
  bool maximize                    = std::get<0>(GetParam());
  int scaling                      = std::get<1>(GetParam());
  bool heuristics_only             = std::get<2>(GetParam());
  auto expected_termination_status = std::get<3>(GetParam());

  raft::handle_t handle;
  auto problem = create_std_milp_problem(maximize);

  cuopt::mathematical_optimization::mip_solver_settings_t<int, double> settings{};
  settings.time_limit      = 5;
  settings.mip_scaling     = scaling;
  settings.heuristics_only = heuristics_only;
  settings.presolver       = cuopt::mathematical_optimization::presolver_t::None;

  auto result = cuopt::mathematical_optimization::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(), expected_termination_status);
}

TEST_P(MILPTestParams, TestSingleVarMILP)
{
  bool maximize                    = std::get<0>(GetParam());
  int scaling                      = std::get<1>(GetParam());
  bool heuristics_only             = std::get<2>(GetParam());
  auto expected_termination_status = std::get<3>(GetParam());

  raft::handle_t handle;
  auto problem = create_single_var_milp_problem(maximize);

  cuopt::mathematical_optimization::mip_solver_settings_t<int, double> settings{};
  settings.time_limit      = 5;
  settings.mip_scaling     = scaling;
  settings.heuristics_only = heuristics_only;
  settings.presolver       = cuopt::mathematical_optimization::presolver_t::None;

  auto result = cuopt::mathematical_optimization::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::mathematical_optimization::mip_termination_status_t::Optimal);
}

INSTANTIATE_TEST_SUITE_P(
  MILPTests,
  MILPTestParams,
  testing::Values(
    std::make_tuple(true,
                    CUOPT_MIP_SCALING_ON,
                    true,
                    cuopt::mathematical_optimization::mip_termination_status_t::Optimal),
    std::make_tuple(false,
                    CUOPT_MIP_SCALING_ON,
                    false,
                    cuopt::mathematical_optimization::mip_termination_status_t::Optimal),
    std::make_tuple(true,
                    CUOPT_MIP_SCALING_OFF,
                    true,
                    cuopt::mathematical_optimization::mip_termination_status_t::Optimal),
    std::make_tuple(false,
                    CUOPT_MIP_SCALING_OFF,
                    false,
                    cuopt::mathematical_optimization::mip_termination_status_t::Optimal)));

// ---------------------------------------------------------------------------
// Scaling integrality preservation test
// ---------------------------------------------------------------------------

// Coefficient spread (~log2(100000/1) ≈ 17) exceeds the scaler's 12-threshold
// so the scaling path is exercised; row 4 omits x3 so the integer-only row
// stays integer.
static io::mps_data_model_t<int, double> create_wide_spread_milp()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: x0 + 2 x1 + 3 x2 + 0.5 x3
Subject To
  c0: 3 x0 + 7 x1 + 2 x2 + 1.5 x3 <= 1e6
  c1: 100000 x0 + 50000 x1 + 25000 x2 + 999.9 x3 <= 1e8
  c2: 5 x0 + 11 x1 + 13 x2 + 0.3 x3 <= 1e4
  c3: 60000 x0 + 30000 x1 + 9000 x2 + 42.42 x3 <= 1e8
  c4: x0 + x1 + x2 <= 100
  c5: 8 x0 + 4 x1 + 6 x2 + 3.14 x3 <= 1e4
Bounds
  0 <= x0 <= 1000
  0 <= x1 <= 1000
  0 <= x2 <= 1000
  0 <= x3 <= 1e6
Generals
  x0
  x1
  x2
End
)LP");
}

TEST(ScalingIntegrity, IntegerCoefficientsPreservedAfterScaling)
{
  raft::handle_t handle;
  auto mps_problem = create_wide_spread_milp();
  auto op_problem  = mps_data_model_to_optimization_problem(&handle, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  const int nnz = op_problem.get_nnz();

  auto pre_values =
    cuopt::host_copy(op_problem.get_constraint_matrix_values(), handle.get_stream());
  auto col_indices =
    cuopt::host_copy(op_problem.get_constraint_matrix_indices(), handle.get_stream());
  auto var_types = cuopt::host_copy(op_problem.get_variable_types(), handle.get_stream());
  handle.sync_stream();

  std::vector<bool> was_integer(nnz, false);
  for (int k = 0; k < nnz; ++k) {
    int col = col_indices[k];
    if (var_types[col] == var_t::INTEGER) {
      double abs_val = std::abs(pre_values[k]);
      if (abs_val > 0.0 &&
          std::abs(abs_val - std::round(abs_val)) <= 1e-6 * std::max(1.0, abs_val)) {
        was_integer[k] = true;
      }
    }
  }

  mip::mip_scaling_strategy_t<int, double> scaling(op_problem);
  scaling.scale_problem();

  auto post_values =
    cuopt::host_copy(op_problem.get_constraint_matrix_values(), handle.get_stream());
  handle.sync_stream();

  int violations = 0;
  for (int k = 0; k < nnz; ++k) {
    if (!was_integer[k]) { continue; }
    double abs_val  = std::abs(post_values[k]);
    double frac_err = std::abs(abs_val - std::round(abs_val));
    double rel_tol  = 1e-6 * std::max(1.0, abs_val);
    if (frac_err > rel_tol) {
      ++violations;
      ADD_FAILURE() << "Coefficient [" << k << "] col=" << col_indices[k] << " was integer ("
                    << pre_values[k] << ") but after scaling is " << post_values[k]
                    << " (frac_err=" << frac_err << ")";
    }
  }
  EXPECT_EQ(violations, 0) << violations << " integer coefficients lost integrality after scaling";
}

TEST(ScalingIntegrity, NoObjectiveScalingPreservesIntegerCoefficients)
{
  raft::handle_t handle;
  auto mps_problem = create_wide_spread_milp();
  auto op_problem  = mps_data_model_to_optimization_problem(&handle, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  const int nnz = op_problem.get_nnz();

  auto pre_values =
    cuopt::host_copy(op_problem.get_constraint_matrix_values(), handle.get_stream());
  auto col_indices =
    cuopt::host_copy(op_problem.get_constraint_matrix_indices(), handle.get_stream());
  auto var_types = cuopt::host_copy(op_problem.get_variable_types(), handle.get_stream());
  handle.sync_stream();

  std::vector<bool> was_integer(nnz, false);
  for (int k = 0; k < nnz; ++k) {
    int col = col_indices[k];
    if (var_types[col] == var_t::INTEGER) {
      double abs_val = std::abs(pre_values[k]);
      if (abs_val > 0.0 &&
          std::abs(abs_val - std::round(abs_val)) <= 1e-6 * std::max(1.0, abs_val)) {
        was_integer[k] = true;
      }
    }
  }

  mip::mip_scaling_strategy_t<int, double> scaling(op_problem);
  scaling.scale_problem(/*scale_objective=*/false);

  auto post_values =
    cuopt::host_copy(op_problem.get_constraint_matrix_values(), handle.get_stream());
  handle.sync_stream();

  int violations = 0;
  for (int k = 0; k < nnz; ++k) {
    if (!was_integer[k]) { continue; }
    double abs_val  = std::abs(post_values[k]);
    double frac_err = std::abs(abs_val - std::round(abs_val));
    double rel_tol  = 1e-6 * std::max(1.0, abs_val);
    if (frac_err > rel_tol) { ++violations; }
  }
  EXPECT_EQ(violations, 0) << violations
                           << " integer coefficients lost integrality after scaling (no-obj mode)";
}

}  // namespace cuopt::mathematical_optimization::test
