/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "vanilla_fp.cuh"

#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>

#include <chrono>

namespace cuopt::mathematical_optimization::mip {
namespace {

enum class cycle_kind_t : int { none, distance, integer };

void finish_iteration(feasibility_pump_t<int, double>& fp,
                      cycle_kind_t cycle_kind,
                      bool perturbed,
                      bool timed_out,
                      bool feasible)
{
  fp.finish_iteration_metrics(cycle_kind != cycle_kind_t::none, timed_out, feasible);
  if (fp.metrics == nullptr) { return; }
  fp.metrics->iterations.back().cycle_kind = (int)cycle_kind;
  fp.metrics->iterations.back().perturbed  = perturbed;
}

bool nearest_round(feasibility_pump_t<int, double>& fp, solution_t<int, double>& solution)
{
  using timing_clock = std::chrono::steady_clock;
  if (fp.metrics != nullptr) { solution.handle_ptr->sync_stream(); }
  const auto rounding_begin = timing_clock::now();
  bool feasible             = solution.round_nearest();
  raft::copy(fp.last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  if (fp.metrics != nullptr) { solution.handle_ptr->sync_stream(); }
  fp.record_rounding_metrics(
    solution,
    feasible,
    std::chrono::duration<double>(timing_clock::now() - rounding_begin).count());
  return feasible;
}

bool perturb_cycle(feasibility_pump_t<int, double>& fp,
                   solution_t<int, double>& solution,
                   cycle_kind_t cycle_kind)
{
  fp.perturbate(solution);
  cuopt_assert(solution.test_number_all_integer(), "Vanilla FP perturbation must remain integer");
  raft::copy(fp.last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  fp.last_distances.clear();
  fp.cycle_queue.reset(solution);
  bool feasible = solution.get_feasible();
  finish_iteration(fp, cycle_kind, true, false, feasible);
  return feasible;
}

}  // namespace

bool run_vanilla_fp_descent(feasibility_pump_t<int, double>& fp,
                            solution_t<int, double>& solution,
                            const vanilla_fp_config_t& config)
{
  using timing_clock = std::chrono::steady_clock;
  cuopt_assert(config.diversity_climber >= 1, "Diversity climber index must be positive");

  if (fp.unified_problem == nullptr) { fp.build_unified_projection_problem(solution); }
  auto stream        = solution.handle_ptr->get_stream();
  const int n_points = config.diversity_climber + 1;
  const int n_vars   = solution.problem_ptr->n_variables;
  fp.diversity_rng   = std::mt19937(config.diversity_seed);
  rmm::device_uvector<double> cloud(0, stream);
  fp.seed_cloud_from_assignment_gpu(solution, n_points, cloud);
  raft::copy(solution.assignment.data(),
             cloud.data() + (size_t)config.diversity_climber * n_vars,
             n_vars,
             stream);
  solution.round_nearest();
  raft::copy(fp.last_rounding.data(), solution.assignment.data(), n_vars, stream);
  fp.cycle_queue.reset(solution);
  fp.last_distances.clear();

  while (true) {
    if (fp.context.diversity_manager_ptr->check_b_b_preemption() || fp.timer.check_time_limit()) {
      return false;
    }

    fp.proj_begin = fp.timer.remaining_time();
    if (fp.metrics != nullptr) { solution.handle_ptr->sync_stream(); }
    const auto projection_begin = timing_clock::now();
    const double ratio_of_assigned_integers =
      (double)solution.n_assigned_integers / solution.problem_ptr->n_integer_vars;
    bool feasible = fp.linear_project_onto_polytope(solution, ratio_of_assigned_integers);
    if (fp.metrics != nullptr) { solution.handle_ptr->sync_stream(); }
    const double projection_time =
      std::chrono::duration<double>(timing_clock::now() - projection_begin).count();
    int n_integers = solution.compute_number_of_integers();
    fp.record_projection_metrics(solution, n_integers, 1, projection_time);

    bool distance_cycle = fp.config.check_distance_cycle && fp.check_distance_cycle(solution);
    if (distance_cycle) {
      feasible = nearest_round(fp, solution);
      if (feasible) {
        finish_iteration(fp, cycle_kind_t::distance, false, false, true);
        return true;
      }
      if (perturb_cycle(fp, solution, cycle_kind_t::distance)) { return true; }
      continue;
    }

    if (n_integers == solution.problem_ptr->n_integer_vars) {
      if (feasible) {
        finish_iteration(fp, cycle_kind_t::none, false, false, true);
        return true;
      }
      if (!fp.last_distances.empty() && fp.last_distances[0] < distance_to_check_for_feasible) {
        relaxed_lp_settings_t lp_settings;
        lp_settings.time_limit            = 5.;
        lp_settings.tolerance             = solution.problem_ptr->tolerances.absolute_tolerance;
        lp_settings.return_first_feasible = true;
        lp_settings.save_state            = true;
        lp_settings.check_infeasibility   = false;
        run_lp_with_vars_fixed(*solution.problem_ptr,
                               solution,
                               solution.problem_ptr->integer_indices,
                               lp_settings,
                               &fp.constraint_prop.bounds_update);
        feasible   = solution.get_feasible();
        n_integers = solution.compute_number_of_integers();
        if (feasible && n_integers == solution.problem_ptr->n_integer_vars) {
          finish_iteration(fp, cycle_kind_t::none, false, false, true);
          return true;
        }
      }
    }

    cuopt_func_call(solution.test_variable_bounds(false));
    feasible = nearest_round(fp, solution);
    cuopt_func_call(solution.test_variable_bounds(true));
    fp.proj_and_round_time = fp.proj_begin - fp.timer.remaining_time();
    if (feasible) {
      bool verified = solution.compute_feasibility();
      cuopt_assert(verified, "Vanilla FP feasibility verification failed");
      finish_iteration(fp, cycle_kind_t::none, false, false, true);
      return true;
    }
    if (fp.timer.check_time_limit()) {
      finish_iteration(fp, cycle_kind_t::none, false, true, false);
      return false;
    }

    bool integer_cycle    = false;
    double previous_alpha = fp.config.alpha / fp.config.alpha_decrease_factor;
    if (previous_alpha - fp.config.alpha < 0.005) {
      integer_cycle = fp.cycle_queue.check_cycle(solution);
    }
    fp.cycle_queue.update_recent_solutions(solution);
    if (integer_cycle) {
      if (perturb_cycle(fp, solution, cycle_kind_t::integer)) { return true; }
      continue;
    }
    fp.cycle_queue.n_iterations_without_cycle++;
    finish_iteration(fp, cycle_kind_t::none, false, false, false);
  }
}

}  // namespace cuopt::mathematical_optimization::mip
