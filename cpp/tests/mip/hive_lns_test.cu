/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../benchmarks/linear_programming/cuopt/c_api_check.hpp"
#include "../../../experiments/hive_lns/bridge.cuh"

#include <mip_heuristics/diversity/diversity_manager.cuh>

#include <gtest/gtest.h>

#include <atomic>
#include <limits>

namespace cuopt::hive_lns::test {
namespace mip = cuopt::mathematical_optimization::mip;
namespace opt = cuopt::mathematical_optimization;

model_t make_model(bool integer = false)
{
  model_t model;
  model.lower                 = {0.0};
  model.upper                 = {1.0};
  model.integer               = {integer};
  model.objective             = {1.0};
  model.offsets               = {0, 1};
  model.columns               = {0};
  model.coefficients          = {1.0};
  model.row_lower             = {0.0};
  model.row_upper             = {1.0};
  model.feasibility_tolerance = 1e-7;
  model.relative_tolerance    = 2e-8;
  model.integrality_tolerance = 1e-4;
  model.row_tolerances        = {mip::get_cstr_tolerance<int, double>(
    0.0, 1.0, model.feasibility_tolerance, model.relative_tolerance)};
  return model;
}

TEST(HiveLns, NormalizesPrivateSeedAndKeepsStrictDomains)
{
  auto model      = make_model();
  model.row_lower = {-1.0};
  const std::vector<double> population_member{-5e-5};
  auto seed = population_member;
  ASSERT_TRUE(model.feasible(seed));
  ASSERT_TRUE(model.normalize_seed(seed));
  EXPECT_EQ(seed[0], 0.0);
  EXPECT_EQ(population_member[0], -5e-5);

  repair_request_t request;
  request.start = request.lower = request.upper = seed;
  EXPECT_TRUE(make_repair_problem(model, request).possible);
  request.lower = request.upper = population_member;
  EXPECT_FALSE(make_repair_problem(model, request).possible);
  request.lower = {0.0};
  request.upper = {1.0 + 1e-10};
  EXPECT_FALSE(make_repair_problem(model, request).possible);
}

TEST(HiveLns, RoundsIntegersAndRejectsEmptyIntegerDomains)
{
  auto model = make_model(true);
  std::vector<double> seed{1.0 - 5e-5};
  ASSERT_TRUE(model.normalize_seed(seed));
  EXPECT_EQ(seed[0], 1.0);
  repair_request_t request;
  request.start = request.lower = request.upper = seed;
  EXPECT_TRUE(make_repair_problem(model, request).possible);

  model.lower = {.2};
  model.upper = {.99999};
  seed        = {.99999};
  ASSERT_TRUE(model.feasible(seed));
  EXPECT_FALSE(model.normalize_seed(seed));
}

TEST(HiveLns, RevalidatesRowsAfterRoundingOrClamping)
{
  auto model      = make_model(true);
  model.row_lower = model.row_upper = {1.0 - 5e-5};
  std::vector<double> seed{1.0 - 5e-5};
  ASSERT_TRUE(model.feasible(seed));
  EXPECT_FALSE(model.normalize_seed(seed));

  model.integer   = {false};
  model.row_lower = model.row_upper = {-5e-5};
  seed                              = {-5e-5};
  ASSERT_TRUE(model.feasible(seed));
  EXPECT_FALSE(model.normalize_seed(seed));
}

TEST(HiveLns, HonorsConfiguredBoundsIntegralityAndRelativeRowTolerance)
{
  auto model      = make_model(true);
  model.row_upper = {2.0};
  EXPECT_TRUE(model.feasible({1.0 + 5e-5}));
  model.integrality_tolerance = 1e-6;
  EXPECT_FALSE(model.feasible({1.0 + 5e-5}));
  EXPECT_FALSE(model.feasible({std::numeric_limits<double>::quiet_NaN()}));

  model.integer   = {false};
  model.lower     = {1e6};
  model.upper     = {1e6 + 1.0};
  model.row_lower = model.row_upper = {1e6};
  model.row_tolerances              = {mip::get_cstr_tolerance<int, double>(
    1e6, 1e6, model.feasibility_tolerance, model.relative_tolerance)};
  EXPECT_TRUE(model.feasible({1e6 + .01}));
  EXPECT_FALSE(model.feasible({1e6 + .03}));
  repair_request_t request;
  request.start = request.lower = request.upper = {1e6 + .01};
  EXPECT_TRUE(make_repair_problem(model, request).possible);
}

TEST(HiveLns, BothRepairBackendsContainWidenedNeighborhoods)
{
  auto model = make_model();
  repair_request_t request;
  request.start = {0.0};
  request.lower = {-1e-10};
  request.upper = {1.0};
  std::atomic<bool> preemption{false};
  for (auto backend : {repair_backend_t::cpufj, repair_backend_t::submip}) {
    auto result =
      repair_neighborhood(model, request, backend, [] { return false; }, nullptr, 1.0, preemption);
    EXPECT_FALSE(result.feasible);
    EXPECT_TRUE(result.assignment.empty());
    auto valid_request  = request;
    valid_request.lower = valid_request.upper = valid_request.start;
    auto recovered                            = repair_neighborhood(
      model, valid_request, backend, [] { return false; }, nullptr, 1.0, preemption);
    EXPECT_TRUE(recovered.feasible);
    EXPECT_EQ(recovered.assignment, valid_request.start);
  }
}

TEST(HiveLns, RenormalizesBackendOutputBeforeNextFixing)
{
  auto model = make_model(true);
  model.lower.assign(40, 0.0);
  model.upper.assign(40, 1.0);
  model.integer.assign(40, true);
  model.objective.assign(40, 1.0);
  model.offsets.resize(41);
  model.columns.resize(40);
  std::iota(model.offsets.begin(), model.offsets.end(), 0);
  std::iota(model.columns.begin(), model.columns.end(), 0);
  model.coefficients.assign(40, 1.0);
  model.row_lower.assign(40, 0.0);
  model.row_upper.assign(40, 1.0);
  model.row_tolerances.assign(40, model.row_tolerances.front());
  const population_t population{std::vector<double>(40, 1.0 - 5e-5)};
  int repairs = 0;
  auto repair = [&](const repair_request_t& request) {
    ++repairs;
    for (size_t j = 0; j < request.start.size(); ++j) {
      EXPECT_EQ(request.start[j], 1.0);
      EXPECT_GE(request.lower[j], model.lower[j]);
      EXPECT_LE(request.upper[j], model.upper[j]);
      EXPECT_EQ(request.lower[j], std::round(request.lower[j]));
      EXPECT_EQ(request.upper[j], std::round(request.upper[j]));
    }
    repair_result_t result;
    result.feasible   = true;
    result.assignment = population.front();
    result.objective  = model.cost(result.assignment);
    return result;
  };
  model.repair.cpufj  = repair;
  model.repair.submip = repair;
  int polls           = 0;
  run_lns(
    model,
    [&] { return population; },
    [](const auto&) {},
    [&] { return repairs >= 2 || ++polls > 1000; },
    42);
  EXPECT_EQ(repairs, 2);
  EXPECT_EQ(population.front()[1], 1.0 - 5e-5);
}

std::atomic<bool> worker_ran{false};

void failing_worker(
  const model_t& model, const snapshot_fn& snapshot, const submit_fn&, const stop_fn&, uint64_t)
{
  worker_ran = true;
  EXPECT_EQ(model.feasibility_tolerance, 3e-7);
  EXPECT_EQ(model.relative_tolerance, 4e-8);
  EXPECT_EQ(model.integrality_tolerance, 2e-4);
  EXPECT_EQ(model.row_tolerances[0], (mip::get_cstr_tolerance<int, double>(1, 2, 3e-7, 4e-8)));
  EXPECT_FALSE(snapshot().empty());
  throw std::runtime_error("injected optional worker failure");
}

TEST(HiveLns, WorkerFailurePreservesValidatedPopulationIncumbent)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  const std::vector<double> coefficients{1, 1}, lower{0, 0}, upper{1, 1}, objective{1, 2};
  const std::vector<double> row_lower{1}, row_upper{2};
  const std::vector<int> columns{0, 1}, offsets{0, 2};
  const std::vector<opt::var_t> types(2, opt::var_t::INTEGER);
  op.set_csr_constraint_matrix(coefficients.data(), 2, columns.data(), 2, offsets.data(), 2);
  op.set_variable_lower_bounds(lower.data(), 2);
  op.set_variable_upper_bounds(upper.data(), 2);
  op.set_variable_types(types.data(), 2);
  op.set_objective_coefficients(objective.data(), 2);
  op.set_constraint_lower_bounds(row_lower.data(), 1);
  op.set_constraint_upper_bounds(row_upper.data(), 1);
  opt::mip_solver_settings_t<int, double> settings;
  settings.tolerances.absolute_tolerance    = 3e-7;
  settings.tolerances.relative_tolerance    = 4e-8;
  settings.tolerances.integrality_tolerance = 2e-4;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();
  mip::solution_t<int, double> incumbent(problem);
  incumbent.copy_new_assignment(std::vector<double>{1, 0});
  ASSERT_TRUE(incumbent.compute_feasibility());
  dm.population.add_solution(std::move(incumbent));
  ASSERT_TRUE(dm.population.is_feasible());
  const auto before = dm.population.best_feasible().get_host_assignment();
  worker_ran        = false;
  mip::hive_lns_bridge_t<int, double> worker(
    context, dm.population, cuopt::timer_t(10), failing_worker);
  EXPECT_NO_THROW(worker.finish());
  EXPECT_TRUE(worker_ran.load());
  EXPECT_TRUE(dm.population.best_feasible().compute_feasibility());
  EXPECT_EQ(dm.population.best_feasible().get_host_assignment(), before);
  EXPECT_FALSE(context.preempt_heuristic_solver_.load());
  EXPECT_FALSE(dm.population.lns_observer);
}

TEST(HiveLns, BenchmarkApiErrorsAreDistinctFromNoSolution)
{
  EXPECT_NO_THROW(cuopt_bench::check_c_api(CUOPT_SUCCESS, "cuOptSolve"));
  try {
    cuopt_bench::check_c_api(CUOPT_RUNTIME_ERROR, "cuOptSolve");
    FAIL() << "API error was ignored";
  } catch (const cuopt_bench::c_api_error_t& error) {
    EXPECT_EQ(error.code, CUOPT_RUNTIME_ERROR);
    EXPECT_STREQ(error.operation, "cuOptSolve");
  }
}
}  // namespace cuopt::hive_lns::test
