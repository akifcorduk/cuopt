/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/local_search/feasibility_pump/pdfp_projection.cuh>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <raft/core/handle.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::test {
namespace {

constexpr double inf = std::numeric_limits<double>::infinity();
using clock_type     = std::chrono::steady_clock;

struct projection_model_t {
  std::vector<int> offsets, indices;
  std::vector<double> values, row_lower, row_upper, lower, upper, objective;
  std::vector<var_t> types;

  optimization_problem_t<int, double> device_problem(const raft::handle_t& handle) const
  {
    optimization_problem_t<int, double> result(&handle);
    result.set_csr_constraint_matrix(
      values.data(), values.size(), indices.data(), indices.size(), offsets.data(), offsets.size());
    result.set_objective_coefficients(objective.data(), objective.size());
    result.set_variable_lower_bounds(lower.data(), lower.size());
    result.set_variable_upper_bounds(upper.data(), upper.size());
    result.set_variable_types(types.data(), types.size());
    result.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
    result.set_constraint_upper_bounds(row_upper.data(), row_upper.size());
    return result;
  }
};

projection_model_t copy_model(mip::problem_t<int, double>& problem)
{
  auto stream = problem.handle_ptr->get_stream();
  projection_model_t result;
  result.offsets   = cuopt::host_copy(problem.offsets, stream);
  result.indices   = cuopt::host_copy(problem.variables, stream);
  result.values    = cuopt::host_copy(problem.coefficients, stream);
  result.row_lower = cuopt::host_copy(problem.constraint_lower_bounds, stream);
  result.row_upper = cuopt::host_copy(problem.constraint_upper_bounds, stream);
  result.objective = cuopt::host_copy(problem.objective_coefficients, stream);
  result.types     = cuopt::host_copy(problem.variable_types, stream);
  auto bounds      = cuopt::host_copy(problem.variable_bounds, stream);
  for (auto bound : bounds) {
    result.lower.push_back(cuopt::get_lower(bound));
    result.upper.push_back(cuopt::get_upper(bound));
  }
  return result;
}

struct projection_metrics_t {
  double objective              = 0.;
  double row_violation          = 0.;
  double relative_row_violation = 0.;
  double bound_violation        = 0.;
};

projection_metrics_t measure(const projection_model_t& model,
                             const std::vector<double>& target,
                             const std::vector<double>& x,
                             const mip::pdfp_projection_settings_t& settings)
{
  projection_metrics_t metrics;
  for (size_t j = 0; j < model.lower.size(); ++j) {
    if (!std::isfinite(x[j])) {
      metrics.objective = metrics.row_violation = metrics.bound_violation = inf;
      return metrics;
    }
    metrics.objective += settings.objective_weight * model.objective[j] * x[j];
    if (model.types[j] == var_t::INTEGER) {
      metrics.objective += settings.distance_weight * std::abs(x[j] - target[j]);
    }
    metrics.bound_violation =
      std::max({metrics.bound_violation, model.lower[j] - x[j], x[j] - model.upper[j]});
  }
  for (size_t i = 0; i < model.row_lower.size(); ++i) {
    double activity = 0., scale = 1.;
    for (int k = model.offsets[i]; k < model.offsets[i + 1]; ++k) {
      activity += model.values[k] * x[model.indices[k]];
    }
    if (std::isfinite(model.row_lower[i])) scale = std::max(scale, std::abs(model.row_lower[i]));
    if (std::isfinite(model.row_upper[i])) scale = std::max(scale, std::abs(model.row_upper[i]));
    const double violation =
      std::max({0., model.row_lower[i] - activity, activity - model.row_upper[i]});
    metrics.row_violation          = std::max(metrics.row_violation, violation);
    metrics.relative_row_violation = std::max(metrics.relative_row_violation, violation / scale);
  }
  return metrics;
}

struct oracle_result_t {
  std::vector<double> x;
  pdlp_termination_status_t status;
  double setup_time, solve_time;
  int rows, columns, nnz;
  int iterations = -1;
};

// warm_dual, when given, warm-starts PDLP as FP does: primal at the target, dual from the previous
// projection when the row count matches; it receives this projection's dual.
oracle_result_t epigraph_projection(const raft::handle_t& handle,
                                    const projection_model_t& original,
                                    const std::vector<double>& target,
                                    const mip::pdfp_projection_settings_t& settings,
                                    std::vector<double>* warm_dual = nullptr)
{
  const auto begin = clock_type::now();
  auto model       = original;
  const int n      = model.lower.size();
  for (int j = 0; j < n; ++j) {
    model.objective[j] *= settings.objective_weight;
    if (model.types[j] != var_t::INTEGER) continue;
    if (target[j] == model.lower[j]) {
      model.objective[j] += settings.distance_weight;
    } else if (target[j] == model.upper[j]) {
      model.objective[j] -= settings.distance_weight;
    } else {
      const int auxiliary = model.lower.size();
      model.lower.push_back(0.);
      model.upper.push_back(inf);
      model.objective.push_back(settings.distance_weight);
      model.types.push_back(var_t::CONTINUOUS);
      model.indices.insert(model.indices.end(), {j, auxiliary, j, auxiliary});
      model.values.insert(model.values.end(), {1., -1., -1., -1.});
      model.offsets.push_back(model.offsets.back() + 2);
      model.offsets.push_back(model.offsets.back() + 2);
      model.row_lower.insert(model.row_lower.end(), {-inf, -inf});
      model.row_upper.insert(model.row_upper.end(), {target[j], -target[j]});
    }
  }
  std::fill(model.types.begin(), model.types.end(), var_t::CONTINUOUS);
  auto problem = model.device_problem(handle);
  handle.sync_stream();
  const auto setup_end = clock_type::now();
  pdlp_solver_settings_t<int, double> lp_settings;
  lp_settings.method           = method_t::PDLP;
  lp_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable2;
  lp_settings.presolver        = presolver_t::None;
  lp_settings.time_limit       = settings.time_limit;
  lp_settings.set_optimality_tolerance(settings.tolerance);
  lp_settings.log_to_console = false;
  std::vector<double> warm_primal;
  if (warm_dual != nullptr) {
    // FP's relaxed LP keeps the relative primal and dual tolerances at tolerance * rel / abs.
    lp_settings.tolerances.relative_primal_tolerance = settings.tolerance * 1e-6;
    lp_settings.tolerances.relative_dual_tolerance   = settings.tolerance * 1e-6;
    warm_primal                                      = target;
    warm_primal.resize(model.lower.size(), 0.);
    lp_settings.set_initial_primal_solution(warm_primal.data(), warm_primal.size());
    if (warm_dual->size() == model.row_lower.size()) {
      lp_settings.set_initial_dual_solution(warm_dual->data(), warm_dual->size());
    }
  }
  auto result = solve_lp(problem, lp_settings);
  handle.sync_stream();
  const auto solve_end = clock_type::now();
  auto x               = cuopt::host_copy(result.get_primal_solution(), handle.get_stream());
  if (warm_dual != nullptr) {
    *warm_dual = cuopt::host_copy(result.get_dual_solution(), handle.get_stream());
  }
  if (x.size() < static_cast<size_t>(n)) {
    ADD_FAILURE() << "Epigraph solver returned no complete primal solution";
    throw std::runtime_error("Missing epigraph primal solution");
  }
  x.resize(n);
  return {std::move(x),
          result.get_termination_status(),
          std::chrono::duration<double>(setup_end - begin).count(),
          std::chrono::duration<double>(solve_end - setup_end).count(),
          static_cast<int>(model.row_lower.size()),
          static_cast<int>(model.lower.size()),
          static_cast<int>(model.values.size()),
          result.get_additional_termination_information().number_of_steps_taken};
}

mip::pdfp_projection_settings_t accurate_settings()
{
  mip::pdfp_projection_settings_t result;
  result.tolerance      = 1e-7;
  result.time_limit     = 20.;
  result.check_interval = 20;
  return result;
}

std::vector<double> check_projection(const projection_model_t& model,
                                     const std::vector<double>& target,
                                     double expected_objective,
                                     mip::pdfp_projection_settings_t settings = accurate_settings())
{
  raft::handle_t handle;
  auto op_problem = model.device_problem(handle);
  mip::problem_t<int, double> problem(op_problem);
  mip::pdfp_projection_t<int, double> projector(problem);
  rmm::device_uvector<double> d_target(target.size(), handle.get_stream());
  rmm::device_uvector<double> d_output(target.size(), handle.get_stream());
  raft::copy(d_target.data(), target.data(), target.size(), handle.get_stream());
  const auto result = projector.project(d_target, d_output, settings);
  const auto output = cuopt::host_copy(d_output, handle.get_stream());
  EXPECT_TRUE(result.converged);
  const auto metrics = measure(model, target, output, settings);
  EXPECT_LT(metrics.row_violation, 2e-5);
  EXPECT_LT(metrics.bound_violation, 1e-10);
  EXPECT_NEAR(metrics.objective, expected_objective, 2e-5);
  EXPECT_NEAR(result.objective, metrics.objective, 2e-5);
  auto oracle = epigraph_projection(handle, model, target, settings);
  EXPECT_EQ(oracle.status, pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(measure(model, target, oracle.x, settings).objective, metrics.objective, 5e-5);
  // The prox form of the distance term solves the same projection.
  mip::pdfp_projection_t<int, double> prox_projector(problem, false, true, true);
  rmm::device_uvector<double> d_prox(target.size(), handle.get_stream());
  const auto prox_result = prox_projector.project(d_target, d_prox, settings);
  const auto prox_output = cuopt::host_copy(d_prox, handle.get_stream());
  EXPECT_TRUE(prox_result.converged);
  const auto prox_metrics = measure(model, target, prox_output, settings);
  EXPECT_LT(prox_metrics.row_violation, 2e-5);
  EXPECT_LT(prox_metrics.bound_violation, 1e-10);
  EXPECT_NEAR(prox_metrics.objective, expected_objective, 2e-5);
  EXPECT_NEAR(prox_result.objective, prox_metrics.objective, 2e-5);
  // PDLP's stopping rule (duality gap, row tolerance abs + rel * |bound|) reaches the same optimum
  // with either form of the distance term.
  auto gap_settings               = settings;
  gap_settings.duality_gap_test   = true;
  gap_settings.primal_ratio       = 1.;
  gap_settings.primal_bound_ratio = 1e-6;
  for (int prox = 0; prox < 2; ++prox) {
    mip::pdfp_projection_t<int, double> gap_projector(problem, false, true, prox == 1);
    rmm::device_uvector<double> d_gap(target.size(), handle.get_stream());
    const auto gap_result = gap_projector.project(d_target, d_gap, gap_settings);
    const auto gap_output = cuopt::host_copy(d_gap, handle.get_stream());
    EXPECT_TRUE(gap_result.converged);
    const auto gap_metrics = measure(model, target, gap_output, gap_settings);
    EXPECT_LT(gap_metrics.row_violation, 2e-5);
    EXPECT_LT(gap_metrics.bound_violation, 1e-10);
    EXPECT_NEAR(
      gap_metrics.objective, expected_objective, 2e-5 * (1. + std::abs(expected_objective)));
  }
  // The adaptive step reaches the same optimum, also from the step a previous projection left.
  auto adaptive_settings          = gap_settings;
  adaptive_settings.adaptive_step = true;
  mip::pdfp_projection_t<int, double> adaptive_projector(problem, true, true, true);
  rmm::device_uvector<double> d_adaptive(target.size(), handle.get_stream());
  for (int pass = 0; pass < 2; ++pass) {
    const auto adaptive_result =
      adaptive_projector.project(d_target, d_adaptive, adaptive_settings);
    const auto adaptive_output = cuopt::host_copy(d_adaptive, handle.get_stream());
    EXPECT_TRUE(adaptive_result.converged);
    const auto adaptive_metrics = measure(model, target, adaptive_output, adaptive_settings);
    EXPECT_LT(adaptive_metrics.row_violation, 2e-5);
    EXPECT_LT(adaptive_metrics.bound_violation, 1e-10);
    EXPECT_NEAR(
      adaptive_metrics.objective, expected_objective, 2e-5 * (1. + std::abs(expected_objective)));
  }
  // Working-set steps solve the same projection.
  auto held_settings        = gap_settings;
  held_settings.held_weight = .01;
  mip::pdfp_projection_t<int, double> held_projector(problem, true, true, true);
  rmm::device_uvector<double> d_held(target.size(), handle.get_stream());
  const auto held_result = held_projector.project(d_target, d_held, held_settings);
  const auto held_output = cuopt::host_copy(d_held, handle.get_stream());
  EXPECT_TRUE(held_result.converged);
  const auto held_metrics = measure(model, target, held_output, held_settings);
  EXPECT_LT(held_metrics.row_violation, 2e-5);
  EXPECT_LT(held_metrics.bound_violation, 1e-10);
  EXPECT_NEAR(
    held_metrics.objective, expected_objective, 2e-5 * (1. + std::abs(expected_objective)));
  return output;
}

projection_model_t paired_model(int n)
{
  projection_model_t model;
  model.offsets.push_back(0);
  model.lower.assign(n, -10.);
  model.upper.assign(n, 10.);
  model.objective.assign(n, 0.);
  model.types.assign(n, var_t::INTEGER);
  for (int j = 0; j < n; j += 2) {
    const double scale = j % 6 == 0 ? 0.25 : (j % 6 == 2 ? 1. : 4.);
    model.indices.insert(model.indices.end(), {j, j + 1, j, j + 1});
    model.values.insert(model.values.end(), {scale, scale, 1., -1.});
    model.offsets.push_back(model.offsets.back() + 2);
    model.offsets.push_back(model.offsets.back() + 2);
    model.row_lower.insert(model.row_lower.end(), {8. * scale, 0.});
    model.row_upper.insert(model.row_upper.end(), {inf, 0.});
  }
  return model;
}

void print_benchmark(const char* method,
                     const char* phase,
                     const projection_model_t& model,
                     const std::vector<double>& target,
                     const std::vector<double>& x,
                     const mip::pdfp_projection_settings_t& settings,
                     double setup_time,
                     double solve_time,
                     int iterations,
                     bool converged,
                     int rows,
                     int columns,
                     int nnz)
{
  const auto metrics = measure(model, target, x, settings);
  std::cout << std::setprecision(10) << "PDFP_BENCHMARK method=" << method << " phase=" << phase
            << " rows=" << rows << " columns=" << columns << " nnz=" << nnz
            << " setup_seconds=" << setup_time << " solve_seconds=" << solve_time
            << " iterations=" << iterations << " converged=" << converged
            << " objective=" << metrics.objective << " max_row_violation=" << metrics.row_violation
            << " max_relative_row_violation=" << metrics.relative_row_violation
            << " max_bound_violation=" << metrics.bound_violation << '\n';
}

void benchmark_model(const projection_model_t& model,
                     std::vector<double> target,
                     double expected_objective = -1.)
{
  int integer_count = 0, general_integer_count = 0;
  for (size_t j = 0; j < model.types.size(); ++j) {
    if (model.types[j] == var_t::INTEGER) {
      ++integer_count;
      if (model.lower[j] < 0. || model.upper[j] > 1.) ++general_integer_count;
    }
  }
  std::cout << "PDFP_BENCHMARK integer_variables=" << integer_count
            << " general_integer_variables=" << general_integer_count << '\n';
  raft::handle_t handle;
  auto op_problem = model.device_problem(handle);
  mip::problem_t<int, double> problem(op_problem);
  mip::pdfp_projection_settings_t settings;
  settings.tolerance  = 1e-5;
  settings.time_limit = 30.;
  if (const char* time = std::getenv("PDFP_BENCHMARK_TIME_LIMIT")) {
    settings.time_limit = std::stod(time);
  }
  handle.sync_stream();
  const auto begin = clock_type::now();
  mip::pdfp_projection_t<int, double> projector(problem);
  handle.sync_stream();
  const double setup_time = std::chrono::duration<double>(clock_type::now() - begin).count();
  rmm::device_uvector<double> d_target(target.size(), handle.get_stream());
  rmm::device_uvector<double> d_output(target.size(), handle.get_stream());
  for (int pass = 0; pass < 2; ++pass) {
    const char* phase = pass == 0 ? "cold" : "warm_changed_target";
    raft::copy(d_target.data(), target.data(), target.size(), handle.get_stream());
    handle.sync_stream();
    const auto solve_begin = clock_type::now();
    const auto result      = projector.project(d_target, d_output, settings);
    handle.sync_stream();
    const double solve_time =
      std::chrono::duration<double>(clock_type::now() - solve_begin).count();
    const auto output = cuopt::host_copy(d_output, handle.get_stream());
    print_benchmark("direct",
                    phase,
                    model,
                    target,
                    output,
                    settings,
                    pass == 0 ? setup_time : 0.,
                    solve_time,
                    result.iterations,
                    result.converged,
                    problem.n_constraints,
                    problem.n_variables,
                    problem.nnz);
    if (expected_objective >= 0. && pass == 0) {
      EXPECT_TRUE(result.converged);
      const auto metrics = measure(model, target, output, settings);
      EXPECT_NEAR(metrics.objective, expected_objective, 1e-4 * (1. + expected_objective));
      EXPECT_LT(metrics.row_violation, 1e-3);
      EXPECT_LT(metrics.bound_violation, 1e-10);
    }
    auto oracle = epigraph_projection(handle, model, target, settings);
    print_benchmark("epigraph_pdlp",
                    phase,
                    model,
                    target,
                    oracle.x,
                    settings,
                    oracle.setup_time,
                    oracle.solve_time,
                    -1,
                    oracle.status == pdlp_termination_status_t::Optimal,
                    oracle.rows,
                    oracle.columns,
                    oracle.nnz);
    for (size_t j = 0; j < target.size(); ++j) {
      if (model.types[j] == var_t::INTEGER) {
        target[j] = std::clamp(
          std::round(output[j]) + (j % 2 == 0 ? 1. : -1.), model.lower[j], model.upper[j]);
      }
    }
  }
}

}  // namespace

TEST(PdfpProjection, GeneralIntegerWithContinuousCompletion)
{
  projection_model_t model{{0, 1, 3},
                           {0, 0, 1},
                           {1., -2., 1.},
                           {5., 0.},
                           {inf, 0.},
                           {0., -inf},
                           {10., inf},
                           {0., 0.},
                           {var_t::INTEGER, var_t::CONTINUOUS}};
  auto x = check_projection(model, {3., -100.}, 2.);
  EXPECT_NEAR(x[0], 5., 2e-5);
  EXPECT_NEAR(x[1], 10., 2e-5);
}

TEST(PdfpProjection, RangedRowsAndFiniteBounds)
{
  projection_model_t model{{0, 1, 3},
                           {0, 0, 1},
                           {1., 1., 1.},
                           {2., 7.},
                           {4., 7.},
                           {0., 0.},
                           {10., 3.},
                           {0., 0.},
                           {var_t::INTEGER, var_t::INTEGER}};
  auto x = check_projection(model, {3., 2.}, 2.);
  EXPECT_NEAR(x[0], 4., 2e-5);
  EXPECT_NEAR(x[1], 3., 2e-5);
}

TEST(PdfpProjection, WeightedOriginalObjective)
{
  projection_model_t model{{0, 1}, {0}, {1.}, {1.}, {inf}, {0.}, {10.}, {-2.}, {var_t::INTEGER}};
  auto settings             = accurate_settings();
  settings.objective_weight = 1.;
  auto x                    = check_projection(model, {3.}, -13., settings);
  EXPECT_NEAR(x[0], 10., 2e-5);
  settings.distance_weight = 4.;
  x                        = check_projection(model, {3.}, -6., settings);
  EXPECT_NEAR(x[0], 3., 2e-5);
}

TEST(PdfpProjection, ZeroDistanceWeight)
{
  projection_model_t model{{0, 1}, {0}, {1.}, {1.}, {inf}, {0.}, {10.}, {2.}, {var_t::INTEGER}};
  auto settings             = accurate_settings();
  settings.objective_weight = 1.;
  settings.distance_weight  = 0.;
  check_projection(model, {7.}, 2., settings);
}

TEST(PdfpProjection, NoIntegerVariables)
{
  projection_model_t model{
    {0, 1}, {0}, {1.}, {5.}, {inf}, {-inf}, {10.}, {1.}, {var_t::CONTINUOUS}};
  auto settings             = accurate_settings();
  settings.objective_weight = 1.;
  check_projection(model, {-20.}, 5., settings);
}

TEST(PdfpProjection, ScalingPreservesUnscaledDistance)
{
  projection_model_t model{{0, 2},
                           {0, 1},
                           {1e-3, 1e3},
                           {5000.},
                           {5000.},
                           {0., 0.},
                           {10., 10.},
                           {0., 0.},
                           {var_t::INTEGER, var_t::INTEGER}};
  auto x = check_projection(model, {3., 2.}, 2.999997);
  EXPECT_NEAR(x[0], 3., 2e-5);
  EXPECT_NEAR(x[1], 4.999997, 2e-5);
}

TEST(PdfpProjection, WarmStartsTargetChangesAliasingAndReset)
{
  projection_model_t model{{0, 1}, {0}, {1.}, {5.}, {inf}, {0.}, {10.}, {0.}, {var_t::INTEGER}};
  raft::handle_t handle;
  auto op_problem = model.device_problem(handle);
  mip::problem_t<int, double> problem(op_problem);
  mip::pdfp_projection_t<int, double> projector(problem);
  rmm::device_uvector<double> target(1, handle.get_stream());
  const auto settings = accurate_settings();
  for (double value : {3., 8., 1.}) {
    raft::copy(target.data(), &value, 1, handle.get_stream());
    const auto result = projector.project(target, target, settings);
    auto output       = cuopt::host_copy(target, handle.get_stream());
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(output[0], std::max(5., value), 2e-5);
    EXPECT_NEAR(result.objective, std::max(0., 5. - value), 2e-5);
    if (value == 8.) projector.reset();
  }
}

TEST(PdfpProjection, IterationLimitDoesNotClaimConvergence)
{
  const auto model = paired_model(100);
  raft::handle_t handle;
  auto op_problem = model.device_problem(handle);
  mip::problem_t<int, double> problem(op_problem);
  mip::pdfp_projection_t<int, double> projector(problem);
  std::vector<double> target(100, 2.);
  rmm::device_uvector<double> d_target(100, handle.get_stream());
  raft::copy(d_target.data(), target.data(), target.size(), handle.get_stream());
  auto settings            = accurate_settings();
  settings.iteration_limit = 1;
  settings.check_interval  = 1;
  const auto result        = projector.project(d_target, d_target, settings);
  EXPECT_FALSE(result.converged);
  EXPECT_LE(result.iterations, 1);
  auto output = cuopt::host_copy(d_target, handle.get_stream());
  EXPECT_EQ(output.size(), target.size());
  for (auto value : output)
    EXPECT_TRUE(std::isfinite(value));
}

TEST(PdfpProjection, EmptyMatrixUsesVariableBounds)
{
  projection_model_t model{
    {0}, {}, {}, {}, {}, {0., 4.}, {10., 4.}, {0., 0.}, {var_t::INTEGER, var_t::INTEGER}};
  raft::handle_t handle;
  auto op_problem = model.device_problem(handle);
  mip::problem_t<int, double> problem(op_problem);
  mip::pdfp_projection_t<int, double> projector(problem);
  std::vector<double> target{3., -2.};
  rmm::device_uvector<double> d_target(2, handle.get_stream());
  raft::copy(d_target.data(), target.data(), target.size(), handle.get_stream());
  const auto result = projector.project(d_target, d_target, accurate_settings());
  auto output       = cuopt::host_copy(d_target, handle.get_stream());
  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(output[0], 3., 1e-6);
  EXPECT_NEAR(output[1], 4., 1e-6);
  EXPECT_NEAR(result.objective, 6., 1e-6);
}

TEST(PdfpProjection, InfeasibleProjectionDoesNotClaimConvergence)
{
  projection_model_t model{{0, 1}, {0}, {1.}, {5.}, {inf}, {0.}, {2.}, {0.}, {var_t::INTEGER}};
  raft::handle_t handle;
  auto op_problem = model.device_problem(handle);
  mip::problem_t<int, double> problem(op_problem);
  mip::pdfp_projection_t<int, double> projector(problem);
  rmm::device_uvector<double> target(1, handle.get_stream());
  double value = 1.;
  raft::copy(target.data(), &value, 1, handle.get_stream());
  auto settings            = accurate_settings();
  settings.iteration_limit = 1000;
  const auto result        = projector.project(target, target, settings);
  EXPECT_FALSE(result.converged);
  auto output = cuopt::host_copy(target, handle.get_stream());
  EXPECT_GE(output[0], 0.);
  EXPECT_LE(output[0], 2.);
  EXPECT_GE(result.primal_residual, 3.);
}

TEST(PdfpProjection, ZeroTimeLimitReturnsBoundedPoint)
{
  projection_model_t model{{0, 1}, {0}, {1.}, {5.}, {inf}, {0.}, {10.}, {0.}, {var_t::INTEGER}};
  raft::handle_t handle;
  auto op_problem = model.device_problem(handle);
  mip::problem_t<int, double> problem(op_problem);
  mip::pdfp_projection_t<int, double> projector(problem);
  rmm::device_uvector<double> target(1, handle.get_stream());
  double value = 1.;
  raft::copy(target.data(), &value, 1, handle.get_stream());
  auto settings       = accurate_settings();
  settings.time_limit = 0.;
  const auto result   = projector.project(target, target, settings);
  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.iterations, 0);
  auto output = cuopt::host_copy(target, handle.get_stream());
  EXPECT_NEAR(output[0], 1., 1e-12);
}

TEST(PdfpProjection, IntegerMaskChangesBetweenWarmProjections)
{
  projection_model_t model{{0, 2},
                           {0, 1},
                           {1., 1.},
                           {8.},
                           {8.},
                           {0., 0.},
                           {10., 10.},
                           {0., 0.},
                           {var_t::INTEGER, var_t::INTEGER}};
  raft::handle_t handle;
  auto op_problem = model.device_problem(handle);
  mip::problem_t<int, double> problem(op_problem);
  mip::pdfp_projection_t<int, double> projector(problem);
  std::vector<double> target{3., 2.};
  rmm::device_uvector<double> d_target(2, handle.get_stream());
  rmm::device_uvector<double> d_output(2, handle.get_stream());
  raft::copy(d_target.data(), target.data(), target.size(), handle.get_stream());
  auto settings = accurate_settings();
  auto result   = projector.project(d_target, d_output, settings);
  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.objective, 3., 2e-5);
  model.types[1] = var_t::CONTINUOUS;
  raft::copy(
    problem.variable_types.data(), model.types.data(), model.types.size(), handle.get_stream());
  problem.compute_n_integer_vars();
  result      = projector.project(d_target, d_output, settings);
  auto output = cuopt::host_copy(d_output, handle.get_stream());
  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(output[0], 3., 2e-5);
  EXPECT_NEAR(output[1], 5., 2e-5);
  EXPECT_NEAR(result.objective, 0., 2e-5);
}

// The fused, graph-replayed iterations between residual checks must reproduce the plain kernels,
// across restarts, primal-weight updates and warm projections with changing weights.
TEST(PdfpProjection, FusedGraphMatchesPlainIterations)
{
  constexpr int n    = 4000;
  const auto model   = paired_model(n);
  constexpr int runs = 3;
  for (int prox = 0; prox < 2; ++prox) {
    std::vector<std::vector<double>> outputs[2];
    std::vector<int> iterations[2];
    for (int fused = 0; fused < 2; ++fused) {
      raft::handle_t handle;
      auto op_problem = model.device_problem(handle);
      mip::problem_t<int, double> problem(op_problem);
      mip::pdfp_projection_t<int, double> projector(problem, false, true, prox == 1);
      rmm::device_uvector<double> target(n, handle.get_stream());
      auto settings            = accurate_settings();
      settings.tolerance       = 1e-14;
      settings.iteration_limit = 600;
      settings.fused_graph     = fused == 1;
      for (int run = 0; run < runs; ++run) {
        std::vector<double> h_target(n);
        for (int j = 0; j < n; ++j)
          h_target[j] = (j + run) % 3 == 0 ? 2. : 3.5 - run;
        raft::copy(target.data(), h_target.data(), n, handle.get_stream());
        settings.distance_weight = 1. / (1 + run);
        const auto result        = projector.project(target, target, settings);
        iterations[fused].push_back(result.iterations);
        outputs[fused].push_back(cuopt::host_copy(target, handle.get_stream()));
      }
    }
    EXPECT_EQ(iterations[0], iterations[1]);
    double max_difference = 0.;
    for (int run = 0; run < runs; ++run)
      for (int j = 0; j < n; ++j)
        max_difference = std::max(
          max_difference,
          std::abs(outputs[0][run][j] - outputs[1][run][j]) / (1. + std::abs(outputs[0][run][j])));
    EXPECT_LT(max_difference, 1e-9);
  }
}

// Warm projections with targets at variable bounds, inside them and changing between calls:
// the prox form must reach the objective of the bounded-dual form.
TEST(PdfpProjection, ProxDistanceMatchesBoundedDualOnWarmProjections)
{
  constexpr int n    = 400;
  const auto model   = paired_model(n);
  constexpr int runs = 4;
  std::vector<double> objectives[2];
  for (int prox = 0; prox < 2; ++prox) {
    raft::handle_t handle;
    auto op_problem = model.device_problem(handle);
    mip::problem_t<int, double> problem(op_problem);
    mip::pdfp_projection_t<int, double> projector(problem, false, true, prox == 1);
    rmm::device_uvector<double> target(n, handle.get_stream());
    rmm::device_uvector<double> output(n, handle.get_stream());
    auto settings = accurate_settings();
    for (int run = 0; run < runs; ++run) {
      std::vector<double> h_target(n);
      for (int j = 0; j < n; ++j) {
        const int kind = (j + run) % 4;
        h_target[j]    = kind == 0 ? -10. : kind == 1 ? 10. : kind == 2 ? 3. : -1. - run;
      }
      raft::copy(target.data(), h_target.data(), n, handle.get_stream());
      settings.distance_weight  = 1. / (1 + run);
      settings.objective_weight = run % 2 == 0 ? 0. : 0.05;
      const auto result         = projector.project(target, output, settings);
      EXPECT_TRUE(result.converged);
      const auto x       = cuopt::host_copy(output, handle.get_stream());
      const auto metrics = measure(model, h_target, x, settings);
      EXPECT_LT(metrics.row_violation, 2e-5);
      EXPECT_LT(metrics.bound_violation, 1e-10);
      objectives[prox].push_back(metrics.objective);
    }
  }
  for (int run = 0; run < runs; ++run)
    EXPECT_NEAR(objectives[0][run], objectives[1][run], 1e-4 * (1. + std::abs(objectives[0][run])));
}

TEST(PdfpProjection, DISABLED_Benchmark100kGeneralIntegers)
{
  constexpr int n = 100000;
  std::vector<double> target(n);
  for (int j = 0; j < n; ++j)
    target[j] = j % 2 == 0 ? 2. : 3.;
  benchmark_model(paired_model(n), std::move(target), 1.5 * n);
}

TEST(PdfpProjection, DISABLED_BenchmarkMillionGeneralIntegers)
{
  constexpr int n = 1000000;
  std::vector<double> target(n);
  for (int j = 0; j < n; ++j)
    target[j] = j % 2 == 0 ? 2. : 3.;
  benchmark_model(paired_model(n), std::move(target), 1.5 * n);
}

TEST(PdfpProjection, DISABLED_BenchmarkMps)
{
  const char* path = std::getenv("PDFP_BENCHMARK_MPS");
  ASSERT_NE(path, nullptr) << "Set PDFP_BENCHMARK_MPS to an MPS file.";
  raft::handle_t handle;
  auto mps        = io::read_mps<int, double>(path, false);
  auto op_problem = mps_data_model_to_optimization_problem(&handle, mps);
  mip::problem_t<int, double> problem(op_problem);
  auto model = copy_model(problem);
  pdlp_solver_settings_t<int, double> settings;
  settings.method     = method_t::PDLP;
  settings.time_limit = 10.;
  settings.set_optimality_tolerance(1e-3);
  settings.log_to_console = false;
  auto root               = solve_lp(op_problem, settings);
  auto target             = cuopt::host_copy(root.get_primal_solution(), handle.get_stream());
  ASSERT_EQ(target.size(), model.lower.size());
  for (size_t j = 0; j < target.size(); ++j) {
    ASSERT_TRUE(std::isfinite(target[j]));
    if (model.types[j] == var_t::INTEGER) target[j] = std::round(target[j]);
    target[j] = std::clamp(target[j], model.lower[j], model.upper[j]);
  }
  std::cout << "PDFP_BENCHMARK instance=" << path << '\n';
  benchmark_model(model, std::move(target));
}

// FP-like sequence on an MPS. Every method projects the same targets with FP's objective mixing:
// the nearest rounding of the epigraph oracle's previous projection, alpha 0.99 * 0.9^k. Each
// method is warm-started from its own previous projection.
TEST(PdfpProjection, DISABLED_BenchmarkFpSequence)
{
  const char* path = std::getenv("PDFP_BENCHMARK_MPS");
  ASSERT_NE(path, nullptr) << "Set PDFP_BENCHMARK_MPS to an MPS file.";
  auto env_double = [](const char* name, double fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::stod(value);
  };
  const double tolerance  = env_double("PDFP_BENCHMARK_TOL", 1e-6);
  const double time_limit = env_double("PDFP_BENCHMARK_TIME_LIMIT", 5.);
  const int steps         = int(env_double("PDFP_BENCHMARK_STEPS", 6));
  const int interval      = int(env_double("PDFP_BENCHMARK_CHECK", 200));
  raft::handle_t handle;
  auto mps        = io::read_mps<int, double>(path, false);
  auto op_problem = mps_data_model_to_optimization_problem(&handle, mps);
  mip::problem_t<int, double> problem(op_problem);
  auto model  = copy_model(problem);
  const int n = model.lower.size();
  pdlp_solver_settings_t<int, double> root_settings;
  root_settings.method     = method_t::PDLP;
  root_settings.time_limit = 30.;
  root_settings.set_optimality_tolerance(1e-4);
  root_settings.log_to_console = false;
  auto root                    = solve_lp(op_problem, root_settings);
  auto x                       = cuopt::host_copy(root.get_primal_solution(), handle.get_stream());
  ASSERT_EQ(x.size(), size_t(n));
  int n_integer         = 0;
  double objective_norm = 0.;
  for (int j = 0; j < n; ++j) {
    n_integer += model.types[j] == var_t::INTEGER;
    objective_norm += model.objective[j] * model.objective[j];
  }
  objective_norm = std::sqrt(objective_norm);
  auto rounding  = [&](const std::vector<double>& point) {
    std::vector<double> target(point);
    for (int j = 0; j < n; ++j) {
      if (model.types[j] == var_t::INTEGER) target[j] = std::round(target[j]);
      target[j] = std::clamp(target[j], model.lower[j], model.upper[j]);
    }
    return target;
  };
  struct variant_t {
    const char* name;
    bool prox, row_tolerance, gap_test, balanced_weight;
    double weight_gain;
  };
  // PDFP_BENCHMARK_VARIANTS keeps the variants whose name is in the comma-separated list.
  const char* wanted = std::getenv("PDFP_BENCHMARK_VARIANTS");
  std::vector<variant_t> variants;
  for (const auto& variant :
       std::vector<variant_t>{{"dual", false, false, false, false, .99},
                              {"dual+ptol", false, true, false, false, .99},
                              {"dual+ptol+gap", false, true, true, false, .99},
                              {"prox", true, false, false, false, .99},
                              {"prox+ptol", true, true, false, false, .99},
                              {"prox+ptol+gap", true, true, true, false, .99},
                              {"pg+w0", true, true, true, true, .99},
                              {"pg+kp", true, true, true, false, .5},
                              {"pg+w0+kp", true, true, true, true, .5},
                              {"dg+w0+kp", false, true, true, true, .5}}) {
    const std::string list = wanted == nullptr ? "" : std::string(",") + wanted + ",";
    if (wanted == nullptr || wanted[0] == '\0' ||
        list.find(std::string(",") + variant.name + ",") != std::string::npos) {
      variants.push_back(variant);
    }
  }
  std::vector<std::unique_ptr<mip::pdfp_projection_t<int, double>>> projectors;
  for (const auto& variant : variants) {
    projectors.push_back(
      std::make_unique<mip::pdfp_projection_t<int, double>>(problem, true, true, variant.prox));
  }
  rmm::device_uvector<double> d_target(n, handle.get_stream());
  rmm::device_uvector<double> d_output(n, handle.get_stream());
  std::vector<double> oracle_dual;
  auto target  = rounding(x);
  double alpha = .99;
  std::cout << "FPSEQ instance=" << path << " rows=" << problem.n_constraints << " cols=" << n
            << " integers=" << n_integer << " nnz=" << problem.nnz << " tol=" << tolerance << '\n';
  auto report = [&](int step,
                    const char* method,
                    const std::vector<double>& point,
                    const mip::pdfp_projection_settings_t& settings,
                    int iterations,
                    double seconds,
                    bool converged,
                    double weight,
                    int restarts) {
    const auto metrics = measure(model, target, point, settings);
    int integral       = 0;
    for (int j = 0; j < n; ++j) {
      integral +=
        model.types[j] == var_t::INTEGER && std::abs(point[j] - std::round(point[j])) <= 1e-5;
    }
    std::cout << std::setprecision(9) << "FPSEQ step=" << step << " method=" << method
              << " iters=" << iterations << " sec=" << seconds << " conv=" << converged
              << " obj=" << metrics.objective << std::setprecision(3)
              << " viol=" << metrics.row_violation << " relviol=" << metrics.relative_row_violation
              << " integral=" << integral << " weight=" << weight << " restarts=" << restarts
              << '\n';
  };
  for (int step = 1; step <= steps; ++step) {
    alpha *= .9;
    mip::pdfp_projection_settings_t settings;
    settings.tolerance        = tolerance;
    settings.time_limit       = time_limit;
    settings.check_interval   = interval;
    settings.distance_weight  = n_integer > 0 ? (1. - alpha) / std::sqrt(double(n_integer)) : 0.;
    settings.objective_weight = objective_norm > 0. ? alpha / objective_norm : 0.;
    raft::copy(d_target.data(), target.data(), n, handle.get_stream());
    for (size_t v = 0; v < variants.size(); ++v) {
      auto variant_settings = settings;
      if (variants[v].row_tolerance) {
        variant_settings.primal_ratio       = 1.;
        variant_settings.primal_bound_ratio = 1e-6;
      }
      variant_settings.duality_gap_test = variants[v].gap_test;
      variant_settings.balanced_weight  = variants[v].balanced_weight;
      variant_settings.weight_gain      = variants[v].weight_gain;
      handle.sync_stream();
      const auto begin  = clock_type::now();
      const auto result = projectors[v]->project(d_target, d_output, variant_settings);
      handle.sync_stream();
      const double seconds = std::chrono::duration<double>(clock_type::now() - begin).count();
      report(step,
             variants[v].name,
             cuopt::host_copy(d_output, handle.get_stream()),
             settings,
             result.iterations,
             seconds,
             result.converged,
             result.primal_weight,
             result.restarts);
    }
    auto oracle = epigraph_projection(handle, model, target, settings, &oracle_dual);
    report(step,
           "epigraph_pdlp",
           oracle.x,
           settings,
           oracle.iterations,
           oracle.solve_time,
           oracle.status == pdlp_termination_status_t::Optimal,
           0.,
           0);
    auto next = rounding(oracle.x);
    if (next == target) {
      // The pump would cycle here; move on by flipping about 1% of the integers.
      unsigned state = 12345u + step;
      for (int j = 0; j < n; ++j) {
        state = state * 1664525u + 1013904223u;
        if (model.types[j] != var_t::INTEGER || (state >> 8) % 100 != 0) continue;
        next[j] = next[j] < model.upper[j] ? next[j] + 1. : next[j] - 1.;
        next[j] = std::clamp(next[j], model.lower[j], model.upper[j]);
      }
    }
    target = std::move(next);
  }
}

}  // namespace cuopt::mathematical_optimization::test
