/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::test {
namespace {

optimization_problem_t<int, double> make_adaptive_step_problem(const raft::handle_t& handle)
{
  optimization_problem_t<int, double> op(&handle);
  const std::vector<double> values{1}, objective{1}, lower{-1000}, upper{1000}, rhs{0};
  const std::vector<int> columns{0}, offsets{0, 1};
  op.set_csr_constraint_matrix(
    values.data(), values.size(), columns.data(), columns.size(), offsets.data(), offsets.size());
  op.set_objective_coefficients(objective.data(), objective.size());
  op.set_variable_lower_bounds(lower.data(), lower.size());
  op.set_variable_upper_bounds(upper.data(), upper.size());
  op.set_constraint_lower_bounds(rhs.data(), rhs.size());
  op.set_constraint_upper_bounds(rhs.data(), rhs.size());
  return op;
}

struct adaptive_attempt_t {
  int valid;
  int host_attempts;
  int device_attempts;
  // Current and tentative states, A^T y caches, deltas, and next step sizes.
  std::array<double, 11> values;
};

class adaptive_step_state_t {
 public:
  adaptive_step_state_t()
    : op_(make_adaptive_step_problem(handle_)),
      problem_(op_),
      climbers_(1),
      primal_weight_(1, handle_.get_stream()),
      step_size_(1, handle_.get_stream()),
      primal_step_size_(1, handle_.get_stream()),
      dual_step_size_(1, handle_.get_stream()),
      unused_bounds_(0, handle_.get_stream())
  {
    settings_.pdlp_solver_mode = pdlp_solver_mode_t::Stable2;
    set_pdlp_solver_mode(settings_);
    problem_.compute_transpose_of_problem();
    pdhg_ = std::make_unique<pdlp::pdhg_solver_t<int, double>>(
      &handle_, problem_, false, climbers_, settings_.hyper_params, settings_.new_bounds);
    strategy_ = std::make_unique<pdlp::adaptive_step_size_strategy_t<int, double>>(
      &handle_, &primal_weight_, &step_size_, false, 1, 1, climbers_, settings_.hyper_params);
    write(primal_weight_, 1);
    set_step_size(10);
    write(pdhg_->get_primal_solution(), 1);
    handle_.sync_stream();
  }

  void set_step_size(double value)
  {
    write(step_size_, value);
    write(primal_step_size_, value);
    write(dual_step_size_, value);
  }

  void set_current(double primal, double dual)
  {
    write(pdhg_->get_primal_solution(), primal);
    write(pdhg_->get_dual_solution(), dual);
    // A restart replaces the iterate but leaves this cache stale until the step.
    write(pdhg_->get_saddle_point_state().get_current_AtY(), 987);
  }

  void begin_iteration() { strategy_->set_valid_step_size(0); }

  adaptive_attempt_t attempt(bool fused, int accepted_iteration, bool restart_to_average)
  {
    const int since_restart = restart_to_average ? 0 : accepted_iteration;
    if (fused) {
      strategy_->take_fused_step(*pdhg_,
                                 primal_step_size_,
                                 dual_step_size_,
                                 since_restart,
                                 restart_to_average,
                                 accepted_iteration);
    } else {
      pdhg_->take_step(primal_step_size_,
                       dual_step_size_,
                       unused_bounds_,
                       pdhg_->get_primal_solution(),
                       pdhg_->get_dual_solution(),
                       since_restart,
                       nullptr,
                       restart_to_average,
                       accepted_iteration,
                       false);
      strategy_->compute_step_sizes(*pdhg_, primal_step_size_, dual_step_size_, accepted_iteration);
    }
    auto& saddle = pdhg_->get_saddle_point_state();
    return {strategy_->get_valid_step_size(),
            pdhg_->total_pdhg_iterations_,
            pdhg_->get_d_total_pdhg_iterations().value(handle_.get_stream()),
            {read(pdhg_->get_primal_solution()),
             read(pdhg_->get_dual_solution()),
             read(pdhg_->get_potential_next_primal_solution()),
             read(pdhg_->get_potential_next_dual_solution()),
             read(saddle.get_current_AtY()),
             read(saddle.get_next_AtY()),
             read(saddle.get_delta_primal()),
             read(saddle.get_delta_dual()),
             read(step_size_),
             read(primal_step_size_),
             read(dual_step_size_)}};
  }

  void accept()
  {
    // Reusing this view is sufficient here: it updates the same vector descriptors
    // as the separate evaluation view in pdlp_solver_t::take_adaptive_step.
    pdhg_->update_solution(pdhg_->get_cusparse_view());
  }

 private:
  void write(rmm::device_uvector<double>& vector, double value)
  {
    raft::update_device(vector.data(), &value, 1, handle_.get_stream());
    handle_.sync_stream();
  }

  double read(const rmm::device_uvector<double>& vector)
  {
    return vector.element(0, handle_.get_stream());
  }

  raft::handle_t handle_;
  optimization_problem_t<int, double> op_;
  mip::problem_t<int, double> problem_;
  pdlp_solver_settings_t<int, double> settings_;
  std::vector<pdlp_climber_strategy_t> climbers_;
  rmm::device_uvector<double> primal_weight_;
  rmm::device_uvector<double> step_size_;
  rmm::device_uvector<double> primal_step_size_;
  rmm::device_uvector<double> dual_step_size_;
  rmm::device_uvector<double> unused_bounds_;
  std::unique_ptr<pdlp::pdhg_solver_t<int, double>> pdhg_;
  std::unique_ptr<pdlp::adaptive_step_size_strategy_t<int, double>> strategy_;
};

void expect_same_attempt(const adaptive_attempt_t& separate, const adaptive_attempt_t& fused)
{
  EXPECT_EQ(fused.valid, separate.valid);
  EXPECT_EQ(fused.host_attempts, separate.host_attempts);
  EXPECT_EQ(fused.device_attempts, separate.device_attempts);
  for (size_t i = 0; i < separate.values.size(); ++i) {
    EXPECT_DOUBLE_EQ(fused.values[i], separate.values[i]) << "state index " << i;
  }
}

TEST(PdlpFusedAdaptiveStep, MatchesRejectedAcceptedAndRestartedAttemptsAcrossParities)
{
  adaptive_step_state_t separate;
  adaptive_step_state_t fused;
  int rejected = 0;
  for (int iteration = 0; iteration < 8; ++iteration) {
    const bool restart = iteration == 2 || iteration == 3 || iteration >= 6;
    if (restart) {
      separate.set_current(2, -0.5);
      fused.set_current(2, -0.5);
    }
    separate.begin_iteration();
    fused.begin_iteration();
    bool accepted = false;
    for (int retry = 0; retry < 32; ++retry) {
      SCOPED_TRACE("accepted iteration " + std::to_string(iteration) + " retry " +
                   std::to_string(retry));
      auto expected = separate.attempt(false, iteration, restart);
      auto actual   = fused.attempt(true, iteration, restart);
      expect_same_attempt(expected, actual);
      ASSERT_GE(actual.valid, 0);
      if (iteration == 0 && retry == 0) { ASSERT_EQ(actual.valid, 0); }
      if (restart) { EXPECT_DOUBLE_EQ(actual.values[4], -0.5); }
      if (actual.valid == 1) {
        accepted = true;
        separate.accept();
        fused.accept();
        break;
      }
      ++rejected;
      if (iteration == 0) { EXPECT_DOUBLE_EQ(actual.values[0], 1); }
    }
    ASSERT_TRUE(accepted);
  }
  EXPECT_GT(rejected, 0);
}

TEST(PdlpFusedAdaptiveStep, PreservesZeroMovementErrorAndAttemptCounters)
{
  adaptive_step_state_t separate;
  adaptive_step_state_t fused;
  separate.set_current(0, 1);
  fused.set_current(0, 1);
  separate.begin_iteration();
  fused.begin_iteration();
  auto expected = separate.attempt(false, 0, true);
  auto actual   = fused.attempt(true, 0, true);
  expect_same_attempt(expected, actual);
  EXPECT_EQ(actual.valid, -1);
  EXPECT_EQ(actual.host_attempts, 1);
  EXPECT_EQ(actual.device_attempts, 0);
}

struct adaptive_solve_result_t {
  pdlp_termination_status_t status;
  int iterations;
  int attempts;
  double objective;
  std::vector<double> primal;
  std::vector<double> dual;
};

adaptive_solve_result_t solve_with_adaptive_graph(bool fused)
{
  raft::handle_t handle;
  auto op = make_adaptive_step_problem(handle);
  mip::problem_t<int, double> problem(op);
  pdlp_solver_settings_t<int, double> settings;
  settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable2;
  settings.iteration_limit  = 2000;
  settings.set_optimality_tolerance(1e-7);
  set_pdlp_solver_mode(settings);
  pdlp::pdlp_solver_t<int, double> solver(problem, settings);
  solver.set_fused_adaptive_step(fused);
  rmm::device_uvector<double> initial_primal(1, handle.get_stream());
  const double primal = 1;
  raft::update_device(initial_primal.data(), &primal, 1, handle.get_stream());
  solver.set_initial_primal_solution(initial_primal);
  solver.set_initial_step_size(10);
  solver.set_initial_primal_weight(1);
  auto result = solver.run_solver(timer_t(30));
  return {result.get_termination_status(),
          result.get_additional_termination_information().number_of_steps_taken,
          solver.get_total_pdhg_iterations(),
          result.get_objective_value(),
          host_copy(result.get_primal_solution(), handle.get_stream()),
          host_copy(result.get_dual_solution(), handle.get_stream())};
}

TEST(PdlpFusedAdaptiveStep, Stable2SolvePreservesObjectiveIteratesAndIterationCounts)
{
  const auto expected = solve_with_adaptive_graph(false);
  const auto actual   = solve_with_adaptive_graph(true);
  EXPECT_EQ(expected.status, pdlp_termination_status_t::Optimal);
  EXPECT_EQ(actual.status, expected.status);
  EXPECT_EQ(actual.iterations, expected.iterations);
  EXPECT_EQ(actual.attempts, expected.attempts);
  EXPECT_DOUBLE_EQ(actual.objective, expected.objective);
  EXPECT_NEAR(actual.objective, 0, 1e-6);
  EXPECT_EQ(actual.primal, expected.primal);
  EXPECT_EQ(actual.dual, expected.dual);
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::test
