/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "feasibility_pump.cuh"

#include <cuopt/error.hpp>
#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/host_helper.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <mip_heuristics/utils.cuh>

#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>

#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/linalg/binary_op.cuh>
#include <utilities/seed_generator.cuh>

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

namespace cuopt::mathematical_optimization::mip {

int resolve_fp_quality_config_id(const char* config)
{
  return config == nullptr ? default_fp_quality_config : std::stoi(config);
}

fp_batch_config_t make_base_fp_batch_config()
{
  fp_batch_config_t config;
  config.max_work_without_feasible   = std::numeric_limits<int>::max();
  config.restart_batch_exhausted     = false;
  config.skip_restart_for_large_pure = false;
  return config;
}

fp_batch_config_t make_fp_batch_config(int quality_config_id)
{
  cuopt_assert(quality_config_id >= 0 && quality_config_id < n_fp_quality_configs,
               "FP quality config ID out of bounds");
  fp_batch_config_t config = make_base_fp_batch_config();
  config.quality_config_id = quality_config_id;
  auto make_probe          = [&config]() {
    config.probe_projection       = true;
    config.adaptive_cloud         = true;
    config.structural_selector    = 2;
    config.target_min_batch_size  = 1;
    config.target_max_batch_size  = 64;
    config.latency_max_batch_size = 64;
    config.fallback_threshold     = 1;
  };
  switch (quality_config_id) {
    case 0: make_probe(); break;
    case 1:
      make_probe();
      config.probe_width_on_stagnation = true;
      break;
    case 2:
      make_probe();
      config.probe_width_on_stagnation = true;
      config.probe_binary_only         = true;
      break;
    default: break;
  }
  return config;
}

template <typename i_t, typename f_t>
feasibility_pump_t<i_t, f_t>::feasibility_pump_t(
  mip_solver_context_t<i_t, f_t>& context_,
  fj_t<i_t, f_t>& fj_,
  //  fj_tree_t<i_t, f_t>& fj_tree_,
  constraint_prop_t<i_t, f_t>& constraint_prop_,
  line_segment_search_t<i_t, f_t>& line_segment_search_,
  rmm::device_uvector<f_t>& lp_optimal_solution_)
  : context(context_),
    fj(fj_),
    // fj_tree(fj_tree_),
    line_segment_search(line_segment_search_),
    cycle_queue(*context.problem_ptr, context.settings.heuristic_params.cycle_detection_length),
    constraint_prop(constraint_prop_),
    last_rounding(context.problem_ptr->n_variables, context.problem_ptr->handle_ptr->get_stream()),
    last_projection(context.problem_ptr->n_variables,
                    context.problem_ptr->handle_ptr->get_stream()),
    orig_variable_types(context.problem_ptr->n_variables,
                        context.problem_ptr->handle_ptr->get_stream()),
    lp_optimal_solution(lp_optimal_solution_),
    diversity_rng(cuopt::seed_generator::get_seed()),
    timer(20.),
    batch_primal_init(0, context.problem_ptr->handle_ptr->get_stream()),
    d_aux_integer_indices_cache(0, context.problem_ptr->handle_ptr->get_stream()),
    probe_prev_dual(0, context.problem_ptr->handle_ptr->get_stream()),
    probe_batch_assignments(0, context.problem_ptr->handle_ptr->get_stream())
{
  int max_config             = n_fp_quality_configs;
  const char* max_config_env = std::getenv("CUOPT_MAX_CONFIG");
  if (max_config_env != nullptr) { max_config = std::stoi(max_config_env); }
  cuopt_assert(max_config > 0 && max_config <= n_fp_quality_configs,
               "CUOPT_MAX_CONFIG must be in [1, n_fp_quality_configs]");

  const char* config  = std::getenv("CUOPT_CONFIG_ID");
  const int config_id = resolve_fp_quality_config_id(config);
  cuopt_assert(config_id >= 0 && config_id < n_fp_quality_configs,
               "CUOPT_CONFIG_ID must be in [0, n_fp_quality_configs)");
  cuopt_assert(config == nullptr || config_id < max_config,
               "CUOPT_CONFIG_ID must be in [0, CUOPT_MAX_CONFIG)");
  batch_config = make_fp_batch_config(config_id);
  CUOPT_LOG_INFO("Using batched FP quality configuration %d of %d", config_id, max_config);
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::set_projection_solver_metrics(i_t pdlp_iterations,
                                                                 i_t pdhg_iterations,
                                                                 i_t termination_status,
                                                                 f_t primal_residual,
                                                                 f_t dual_residual,
                                                                 f_t gap,
                                                                 double batch_mean_pdlp_iterations,
                                                                 i_t batch_max_pdlp_iterations)
{
  last_pdlp_iterations            = pdlp_iterations;
  last_pdhg_iterations            = pdhg_iterations;
  last_projection_status          = termination_status;
  last_pdlp_primal_residual       = primal_residual;
  last_pdlp_dual_residual         = dual_residual;
  last_pdlp_gap                   = gap;
  last_batch_mean_pdlp_iterations = batch_mean_pdlp_iterations;
  last_batch_max_pdlp_iterations  = batch_max_pdlp_iterations;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::record_projection_metrics(solution_t<i_t, f_t>& solution,
                                                             i_t n_integers,
                                                             i_t batch_size,
                                                             double elapsed)
{
  if (metrics == nullptr) { return; }
  auto stream = solution.handle_ptr->get_stream();
  solution.handle_ptr->sync_stream();
  auto& entry                  = metrics->iterations.emplace_back();
  entry.iteration              = (i_t)metrics->iterations.size() - 1;
  entry.outer_iteration        = current_outer_iteration;
  entry.batch_size             = batch_size;
  entry.projection_time        = elapsed;
  entry.projection_solve_time  = last_projection_solve_time;
  last_fj_time                 = 0.;
  last_verify_lp_time          = 0.;
  entry.projected_integers     = n_integers;
  entry.projection_l1_distance = compute_l1_distance<i_t, f_t>(
    solution.problem_ptr->integer_indices, last_rounding, last_projection, solution.handle_ptr);
  entry.projection_feasible           = solution.get_feasible();
  entry.projection_total_violation    = solution.get_total_excess();
  entry.projection_adjusted_violation = solution.get_adjusted_total_excess();
  entry.projection_objective          = solution.get_objective();
  entry.projection_violated_constraints =
    solution.problem_ptr->n_constraints - solution.n_feasible_constraints.value(stream);
  entry.pdlp_iterations               = last_pdlp_iterations;
  entry.pdhg_iterations               = last_pdhg_iterations;
  entry.projection_termination_status = last_projection_status;
  entry.pdlp_primal_residual          = last_pdlp_primal_residual;
  entry.pdlp_dual_residual            = last_pdlp_dual_residual;
  entry.pdlp_gap                      = last_pdlp_gap;
  entry.batch_mean_pdlp_iterations    = last_batch_mean_pdlp_iterations;
  entry.batch_max_pdlp_iterations     = last_batch_max_pdlp_iterations;
  entry.dual_warm_start_used          = last_warm_start_stats.dual_used;
  entry.dual_warm_start_nonfinite     = last_warm_start_stats.nonfinite;
  entry.dual_warm_start_zeroed_tail   = last_warm_start_stats.zeroed_tail;
  entry.probes_emitted                = last_probes_emitted;
  entry.probe_winner                  = last_probe_winner;
  entry.probe_win_margin              = last_probe_win_margin;
  entry.probe_winner_fixings          = last_probe_winner_fixings;
  entry.probing_cache_implications    = last_probing_cache_implications;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::record_rounding_metrics(solution_t<i_t, f_t>& solution,
                                                           bool is_feasible,
                                                           double elapsed)
{
  if (metrics == nullptr) { return; }
  cuopt_assert(!metrics->iterations.empty(), "Projection metrics must precede rounding metrics");
  auto stream = solution.handle_ptr->get_stream();
  solution.handle_ptr->sync_stream();
  auto& entry                = metrics->iterations.back();
  entry.rounding_time        = elapsed;
  entry.rounded_integers     = solution.compute_number_of_integers();
  entry.rounding_l1_movement = compute_l1_distance<i_t, f_t>(solution.problem_ptr->integer_indices,
                                                             last_projection,
                                                             solution.assignment,
                                                             solution.handle_ptr);
  entry.rounding_feasible    = is_feasible;
  entry.rounding_total_violation    = solution.get_total_excess();
  entry.rounding_adjusted_violation = solution.get_adjusted_total_excess();
  entry.rounding_violated_constraints =
    solution.problem_ptr->n_constraints - solution.n_feasible_constraints.value(stream);
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::finish_iteration_metrics(bool cycle,
                                                            bool timed_out,
                                                            bool feasible)
{
  if (metrics == nullptr) { return; }
  cuopt_assert(!metrics->iterations.empty(), "Projection metrics must precede iteration outcome");
  auto& entry          = metrics->iterations.back();
  entry.cycle          = cycle;
  entry.timed_out      = timed_out;
  entry.feasible       = feasible;
  entry.fj_time        = last_fj_time;
  entry.verify_lp_time = last_verify_lp_time;
  // proj_begin is stamped at the top of every descent loop body, so this is the whole iteration.
  entry.iteration_time = proj_begin - timer.remaining_time();
  metrics->feasible_events += feasible;
}

template <typename Iter_T>
long double vector_norm(Iter_T first, Iter_T last)
{
  return sqrt(inner_product(first, last, first, 0.0));
}

// this function creates a weighted objective between the distance to the polytope and the original
// objective in the beginning the solution will favor the original objective but later it favors the
// feasibility(distance)
template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::adjust_objective_with_original(solution_t<i_t, f_t>& solution,
                                                                  std::vector<f_t>& dist_objective,
                                                                  bool longer_fp_run)
{
  // TODO set alpha to zero after some point
  if (!longer_fp_run) {
    CUOPT_LOG_TRACE(
      "changing alpha from %f to %f", config.alpha, config.alpha * config.alpha_decrease_factor);
    config.alpha = config.alpha * config.alpha_decrease_factor;
  }
  f_t distance_weight         = 1. - config.alpha;
  std::vector<f_t> obj_vector = cuopt::host_copy(solution.problem_ptr->objective_coefficients,
                                                 solution.handle_ptr->get_stream());
  solution.handle_ptr->sync_stream();
  const f_t l2_norm_of_original_obj = vector_norm(obj_vector.begin(), obj_vector.end());
  const f_t l2_norm_of_distance_obj = vector_norm(dist_objective.begin(), dist_objective.end());
  CUOPT_LOG_TRACE("l2_norm_of_original_obj %f l2_norm_of_distance_obj %f",
                  l2_norm_of_original_obj,
                  l2_norm_of_distance_obj);
  // f_t orig_obj_weight = config.alpha * (l2_norm_of_distance_obj / l2_norm_of_original_obj);
  f_t orig_obj_weight = config.alpha / l2_norm_of_original_obj;
  distance_weight     = distance_weight / l2_norm_of_distance_obj;
  if (!isfinite(orig_obj_weight)) {
    CUOPT_LOG_TRACE("orig_obj_weight is not finite, setting to zero");
    orig_obj_weight = 0.;
  }
  cuopt_expects(isfinite(orig_obj_weight), error_type_t::RuntimeError, "Weight should be finite!");
  CUOPT_LOG_TRACE("dist weight %f obj weight %f", distance_weight, orig_obj_weight);
  for (i_t i = 0; i < (i_t)dist_objective.size(); ++i) {
    f_t orig_obj      = i < (i_t)obj_vector.size() ? obj_vector[i] : 0.;
    dist_objective[i] = dist_objective[i] * distance_weight + orig_obj_weight * orig_obj;
    cuopt_expects(
      isfinite(dist_objective[i]), error_type_t::RuntimeError, "Weight should be finite!");
  }
}

// TODO adjust this tolerance for runs of lower prec(10-8)
double get_tolerance_from_ratio(double ratio_integer, double absolute_tol)
{
  if (ratio_integer < 0.80) {
    return 0.1;
  } else if (ratio_integer < 0.93) {
    return 0.01;
  } else if (ratio_integer < 0.97) {
    return 0.001;
  } else {
    return absolute_tol;
  }
}

// projects the current integer solution to the polytope.
// the epsilon can be larger here maybe 10-1,10-2.
// finding the projection requires running LP that minimizes the distance of the current solution to
// the polytope. the distance to polytope is integrated into the linear programming constraint.
// following is done if current integer value is within the bounds:
// the distance is added as an additional variable for each original variable.
// minimize the distance where distance is at least |x_j-val(x_j)|.
// two constraints are added to handle abs value.
// if we are at the end of the interval.(i.e x_j is u_j or l_j)
// we can get rid of the additional variables and constraints. because the distance can only be to a
// single direction. we won't need a variable and two constraints for the abs value.

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::linear_project_onto_polytope(solution_t<i_t, f_t>& solution,
                                                                f_t ratio_of_set_integers,
                                                                bool longer_lp_run)
{
  raft::common::nvtx::range fun_scope("linear_project_onto_polytope");
  auto h_assignment = solution.get_host_assignment();
  auto h_variable_bounds =
    cuopt::host_copy(solution.problem_ptr->variable_bounds, solution.handle_ptr->get_stream());
  auto h_last_projection = cuopt::host_copy(last_projection, solution.handle_ptr->get_stream());
  const f_t int_tol      = context.settings.tolerances.integrality_tolerance;
  constraints_delta_t<i_t, f_t> h_constraints;
  variables_delta_t<i_t, f_t> h_variables;
  h_variables.n_vars = solution.problem_ptr->n_variables;
  std::vector<f_t> obj_coefficients(solution.problem_ptr->n_variables, 0.);
  problem_t<i_t, f_t> temp_p(*solution.problem_ptr);
  auto h_integer_indices =
    cuopt::host_copy(solution.problem_ptr->integer_indices, solution.handle_ptr->get_stream());
  f_t obj_offset = 0;
  // for each integer add the variable and the distance constraints
  for (auto i : h_integer_indices) {
    auto h_var_bounds = h_variable_bounds[i];
    if (solution.problem_ptr->integer_equal(h_assignment[i], get_upper(h_var_bounds))) {
      obj_offset += get_upper(h_var_bounds);
      // set the objective weight to -1,  u - x
      obj_coefficients[i] = -1;
    } else if (solution.problem_ptr->integer_equal(h_assignment[i], get_lower(h_var_bounds))) {
      obj_offset -= get_lower(h_var_bounds);
      // set the objective weight to +1,  x - l
      obj_coefficients[i] = 1;
    } else {
      // objective weight is 1
      const f_t obj_weight = 1.;
      // the distance should always be positive
      i_t var_id =
        h_variables.add_variable(0,
                                 (get_upper(h_var_bounds) - get_lower(h_var_bounds)) + int_tol,
                                 obj_weight,
                                 var_t::CONTINUOUS);
      obj_coefficients.push_back(obj_weight);
      f_t dist_val = abs(h_assignment[i] - h_last_projection[i]);
      // if it is out of bounds, because of the approximation issues,or init issues
      // the first projection doesn't have a value
      if (!isfinite(dist_val)) { dist_val = 0; }
      h_assignment.push_back(dist_val);
      std::vector<i_t> constr_indices{var_id, i};
      // d_j - x_j >= -val(x_j)
      std::vector<f_t> constr_coeffs_1{1, -1};
      h_constraints.add_constraint(
        constr_indices, constr_coeffs_1, -h_assignment[i], (f_t)default_cont_upper);
      // d_j + x_j >= val(x_j)
      std::vector<f_t> constr_coeffs_2{1, 1};
      h_constraints.add_constraint(
        constr_indices, constr_coeffs_2, h_assignment[i], (f_t)default_cont_upper);
    }
  }
  adjust_objective_with_original(solution, obj_coefficients, longer_lp_run);
  // commit all the changes that were done by the host
  if (h_variables.size() > 0) { temp_p.insert_variables(h_variables); }
  if (h_constraints.n_constraints() > 0) { temp_p.insert_constraints(h_constraints); }
  if (h_constraints.n_constraints() > 0 || h_variables.size() > 0) {
    temp_p.compute_transpose_of_problem();
  }
  CUOPT_LOG_INFO("linear projection of fp: n_vars %d n_constr %d nnz %d aux %d",
                 temp_p.n_variables,
                 temp_p.n_constraints,
                 temp_p.nnz,
                 (int)h_variables.size());
  cuopt_assert(h_assignment.size() == temp_p.n_variables, "Var count mismatch!");
  cuopt_assert(temp_p.objective_coefficients.size() == temp_p.n_variables, "Var count mismatch!");
  solution.copy_new_assignment(h_assignment);
  cuopt_assert(solution.assignment.size() == temp_p.n_variables, "Var count mismatch!");
  // copy new objective coefficients
  raft::copy(temp_p.objective_coefficients.data(),
             obj_coefficients.data(),
             obj_coefficients.size(),
             solution.handle_ptr->get_stream());
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream());
  temp_p.presolve_data.objective_offset = obj_offset;
  // change the precision between 1. and 10-4 depending on the integer ratio
  // the lp tolerance can be pretty high
  const double lp_tolerance =
    get_tolerance_from_ratio(ratio_of_set_integers, context.settings.tolerances.absolute_tolerance);
  temp_p.check_problem_representation(true);
  const f_t rlp_base = context.settings.heuristic_params.relaxed_lp_time_limit;
  f_t time_limit     = longer_lp_run ? 5. * rlp_base : rlp_base;
  time_limit         = std::max(0.05, std::min(time_limit, timer.remaining_time() / 10.));
  static f_t lp_time = 0;
  static i_t n_calls = 0;
  f_t old_remaining  = timer.remaining_time();
  cuopt_func_call(solution.test_variable_bounds(false));
  relaxed_lp_settings_t lp_settings;
  lp_settings.time_limit          = time_limit;
  lp_settings.tolerance           = lp_tolerance;
  lp_settings.check_infeasibility = false;
  last_warm_start_stats           = {};
  if (metrics != nullptr) { lp_settings.warm_start_stats = &last_warm_start_stats; }
  const auto solve_begin = std::chrono::steady_clock::now();
  auto solver_response   = get_relaxed_lp_solution(temp_p, solution, lp_settings);
  last_projection_solve_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_begin).count();
  const auto term_info = solver_response.get_additional_termination_information();
  set_projection_solver_metrics(term_info.number_of_steps_taken,
                                term_info.total_number_of_attempted_steps,
                                (i_t)solver_response.get_termination_status(),
                                term_info.l2_primal_residual,
                                term_info.l2_dual_residual,
                                term_info.gap,
                                term_info.number_of_steps_taken,
                                term_info.number_of_steps_taken);
  cuopt_func_call(solution.test_variable_bounds(false));
  last_lp_time = old_remaining - timer.remaining_time();
  lp_time += last_lp_time;
  n_calls++;
  CUOPT_LOG_INFO("lp_time %f average lp_time %f", last_lp_time, lp_time / n_calls);
  solution.assignment.resize(solution.problem_ptr->n_variables, solution.handle_ptr->get_stream());
  raft::copy(last_projection.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  // Projection result might be feasible but not optimal, due to time limits
  bool is_feasible = solution.compute_feasibility();
  cuopt_func_call(solution.test_variable_bounds(false));
  if (!is_feasible) {
    CUOPT_LOG_INFO("LP is infeasible returning the current PDLP solution! Code %d",
                   (int)solver_response.get_termination_status());
    return false;
  }
  // normal feasible return
  return true;
}

// round will use inevitable infeasibility while propagating the bounds
template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::round(solution_t<i_t, f_t>& solution, bool update_last_rounding)
{
  bool result;
  CUOPT_LOG_INFO("Rounding the point");
  timer_t bounds_prop_timer(std::max(0.05, std::min(0.5, timer.remaining_time() / 10.)));
  const f_t lp_run_time_after_feasible     = 0.;
  bool old_var                             = constraint_prop.round_all_vars;
  f_t old_time                             = constraint_prop.max_time_for_bounds_prop;
  constraint_prop.round_all_vars           = false;
  constraint_prop.max_time_for_bounds_prop = 0.7;
  result = constraint_prop.apply_round(solution, lp_run_time_after_feasible, bounds_prop_timer);
  constraint_prop.round_all_vars           = old_var;
  constraint_prop.max_time_for_bounds_prop = old_time;
  // result = solution.round_nearest();
  cuopt_func_call(solution.test_variable_bounds(true));
  // copy the last rounding (skipped when rounding a side candidate, e.g. the best cloud climber,
  // so it does not disturb climber 0's classic-FP trajectory state)
  if (update_last_rounding) {
    raft::copy(last_rounding.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
  }
  if (result) {
    CUOPT_LOG_INFO("[FP_FEASIBLE] New feasible solution with objective %g",
                   solution.get_user_objective());
  }
  return result;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::perturbate(solution_t<i_t, f_t>& solution)
{
  constexpr f_t change_ratio = 0.1;
  solution.assign_random_within_bounds(change_ratio, true);
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_fj_cycle_escape(solution_t<i_t, f_t>& solution)
{
  bool is_feasible;
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = false;
  fj.settings.n_of_minimums_for_exit = 5000;
  fj.settings.time_limit             = std::min(3., timer.remaining_time());
  is_feasible                        = fj.solve(solution);
  // if FJ didn't change the solution, take last incumbent solution
  if (!is_feasible && cycle_queue.check_cycle(solution)) {
    CUOPT_LOG_INFO("cycle detected after FJ, taking last incumbent of fj");
    raft::copy(solution.assignment.data(),
               fj.climbers[0]->incumbent_assignment.data(),
               solution.problem_ptr->n_variables,
               solution.handle_ptr->get_stream());
  }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::test_fj_feasible(solution_t<i_t, f_t>& solution,
                                                    f_t time_limit,
                                                    i_t trajectory_capacity)
{
  CUOPT_LOG_INFO("Running 20%% with %f time limit", time_limit);
  auto stream = solution.handle_ptr->get_stream();
  if (trajectory_capacity > 0) {
    fj.set_trajectory_capacity(trajectory_capacity, stream);
    fj.settings.record_trajectory = true;
  }
  bool is_feasible;
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.n_of_minimums_for_exit = 5000;
  fj.settings.time_limit             = std::min(time_limit, timer.remaining_time());
  cuopt_func_call(solution.test_variable_bounds(true));
  is_feasible                   = fj.solve(solution);
  fj.settings.record_trajectory = false;
  cuopt_func_call(solution.test_variable_bounds(true));
  // if FJ didn't change the solution, take last incumbent solution
  if (!is_feasible) {
    raft::copy(solution.assignment.data(),
               last_rounding.data(),
               solution.problem_ptr->n_variables,
               solution.handle_ptr->get_stream());
    cuopt_func_call(solution.test_variable_bounds(true));
  } else {
    CUOPT_LOG_INFO("[FP_FEASIBLE] 20%% FJ run found feasible!");
  }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::handle_cycle(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("handle_cycle");
  CUOPT_LOG_INFO("running handle cycle");
  bool is_feasible       = false;
  fp_fj_cycle_time_begin = timer.remaining_time();
  CUOPT_LOG_INFO("Running longer FJ on last rounding");
  raft::copy(solution.assignment.data(),
             last_rounding.data(),
             last_rounding.size(),
             solution.handle_ptr->get_stream());
  cuopt_func_call(solution.test_variable_bounds(true));
  cuopt_assert(solution.test_number_all_integer(), "All must be integers before fj");
  is_feasible = run_fj_cycle_escape(solution);
  cuopt_assert(solution.test_number_all_integer(), "All must be integers after fj");
  if (cycle_queue.check_cycle(solution)) {
    CUOPT_LOG_INFO("Cycle couldn't be broken. Perturbating FP");
    perturbate(solution);
    is_feasible = solution.get_feasible();
  }
  cycle_queue.n_iterations_without_cycle = 0;
  cycle_queue.update_recent_solutions(solution);
  if (is_feasible) {
    solution.test_feasibility();
    CUOPT_LOG_INFO("[FP_FEASIBLE] Feasible found cycle breaking long FJ");
  }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::restart_fp(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("restart_fp");
  bool is_feasible = handle_cycle(solution);
  // reset the distance
  last_distances.resize(0);
  thrust::transform(solution.handle_ptr->get_thrust_policy(),
                    fj.cstr_weights.begin(),
                    fj.cstr_weights.end(),
                    fj.cstr_weights.begin(),
                    [] __device__(f_t val) {
                      constexpr f_t weight_divisor = 10.;
                      return std::max(f_t(10.), std::round(val / weight_divisor));
                    });
  return is_feasible;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::reset()
{
  best_excess               = std::numeric_limits<f_t>::infinity();
  total_fp_time_until_cycle = 0;
  fp_fj_cycle_time_begin    = timer.remaining_time();
  max_n_of_integers         = 0;
  config.alpha              = default_alpha;
  last_distances.resize(0);
  probe_best_rounding_violation = std::numeric_limits<f_t>::infinity();
  probe_no_improve              = 0;
  thrust::fill(context.problem_ptr->handle_ptr->get_thrust_policy(),
               last_projection.begin(),
               last_projection.end(),
               std::numeric_limits<f_t>::quiet_NaN());
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::resize_vectors(problem_t<i_t, f_t>& problem,
                                                  const raft::handle_t* handle_ptr)
{
  last_rounding.resize(problem.n_variables, handle_ptr->get_stream());
  last_projection.resize(problem.n_variables, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::check_distance_cycle(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("check_distance_cycle");
  f_t distance_to_last_rounding = compute_l1_distance<i_t, f_t>(
    solution.problem_ptr->integer_indices, last_rounding, solution.assignment, solution.handle_ptr);

  bool is_cycle = false;
  if (last_distances.size() == (size_t)config.first_stage_kk) {
    // perf is not important, very small array
    f_t avg_distance =
      std::accumulate(last_distances.begin(), last_distances.end(), 0.0) / last_distances.size();
    if (avg_distance - distance_to_last_rounding <
        config.cycle_distance_reduction_ration * avg_distance) {
      CUOPT_LOG_INFO("Distance cycle detected curr %f avg %f for last %d iter",
                     distance_to_last_rounding,
                     avg_distance,
                     last_distances.size());
      is_cycle = true;
    }
    last_distances.pop_back();
  } else {
    CUOPT_LOG_INFO("Distance of projection: %f", distance_to_last_rounding);
  }
  last_distances.push_front(distance_to_last_rounding);
  return is_cycle;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::relax_general_integers(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("relax_general_integers");
  orig_variable_types.resize(solution.problem_ptr->n_variables, solution.handle_ptr->get_stream());

  auto var_types  = make_span(solution.problem_ptr->variable_types);
  auto var_bnds   = make_span(solution.problem_ptr->variable_bounds);
  auto copy_types = make_span(orig_variable_types);

  raft::copy(orig_variable_types.data(),
             solution.problem_ptr->variable_types.data(),
             orig_variable_types.size(),
             solution.handle_ptr->get_stream());
  thrust::for_each(
    solution.handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator<i_t>(0),
    thrust::make_counting_iterator<i_t>(solution.problem_ptr->n_variables),
    [var_types, var_bnds, copy_types, pb = solution.problem_ptr->view()] __device__(auto v_idx) {
      auto orig_v_type = var_types[v_idx];
      auto var_bounds  = var_bnds[v_idx];
      auto lb          = get_lower(var_bounds);
      auto ub          = get_upper(var_bounds);
      bool var_binary  = (pb.integer_equal(lb, 0) && pb.integer_equal(ub, 1));
      auto copy_type =
        (orig_v_type == var_t::INTEGER) && var_binary ? var_t::INTEGER : var_t::CONTINUOUS;
      var_types[v_idx] = copy_type;
    });
  solution.handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream());
  solution.problem_ptr->compute_n_integer_vars();
  solution.problem_ptr->compute_binary_var_table();
  CUOPT_LOG_INFO("Integers are relaxed n_int vars %d n_binary vars %d n_vars %d",
                 solution.problem_ptr->n_integer_vars,
                 solution.problem_ptr->n_binary_vars,
                 solution.problem_ptr->n_variables);
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::revert_relaxation(solution_t<i_t, f_t>& solution)
{
  cuopt_assert(orig_variable_types.size() == solution.problem_ptr->variable_types.size(),
               "variable size mismatch");
  std::swap(orig_variable_types, solution.problem_ptr->variable_types);
  solution.problem_ptr->compute_n_integer_vars();
  solution.problem_ptr->compute_binary_var_table();
  unified_problem.reset();
  cached_cloud_batch_size = -1;
  cloud_batch_capacity    = 0;
  climber_alphas.clear();
  climber_hash_history.clear();
  if (batch_config.reset_stage_state) {
    reset();
    cycle_queue.reset(solution);
  } else {
    last_distances.resize(0);
  }
  solution.compute_feasibility();
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_single_fp_descent(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("run_single_fp_descent");
  using timing_clock = std::chrono::steady_clock;
  // start by doing nearest rounding
  solution.round_nearest();
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  while (true) {
    if (context.diversity_manager_ptr->check_b_b_preemption() || timer.check_time_limit()) {
      CUOPT_LOG_INFO("FP time limit reached!");
      round(solution);
      return false;
    }
    proj_begin = timer.remaining_time();
    // pass n_assigned_integers from the previous iteration
    f_t ratio_of_assigned_integers =
      f_t(solution.n_assigned_integers) / solution.problem_ptr->n_integer_vars;
    if (metrics != nullptr) { solution.handle_ptr->sync_stream(); }
    const auto projection_begin = timing_clock::now();
    bool is_feasible = linear_project_onto_polytope(solution, ratio_of_assigned_integers);
    if (metrics != nullptr) { solution.handle_ptr->sync_stream(); }
    const double projection_time =
      std::chrono::duration<double>(timing_clock::now() - projection_begin).count();
    i_t n_integers = solution.compute_number_of_integers();
    CUOPT_LOG_INFO("after fp projection n_integers %d total n_integes %d",
                   n_integers,
                   solution.problem_ptr->n_integer_vars);
    record_projection_metrics(solution, n_integers, 1, projection_time);
    bool is_cycle = true;
    // temp comment for presolve run
    if (config.check_distance_cycle) {
      // use distance cycle if we are running ii or objective FP
      is_cycle = check_distance_cycle(solution);
      if (is_cycle) {
        if (metrics != nullptr) { solution.handle_ptr->sync_stream(); }
        const auto rounding_begin = timing_clock::now();
        is_feasible               = round(solution);
        if (metrics != nullptr) { solution.handle_ptr->sync_stream(); }
        record_rounding_metrics(
          solution,
          is_feasible,
          std::chrono::duration<double>(timing_clock::now() - rounding_begin).count());
        cuopt_func_call(solution.test_variable_bounds(true));
        if (is_feasible) {
          bool res = solution.compute_feasibility();
          cuopt_assert(res, "Feasibility issue");
          finish_iteration_metrics(true, false, true);
          return true;
        }
        cuopt::default_logger().flush();
        f_t remaining_time_end_fp = timer.remaining_time();
        total_fp_time_until_cycle = fp_fj_cycle_time_begin - remaining_time_end_fp;
        CUOPT_LOG_INFO("total_fp_time_until_cycle: %f", total_fp_time_until_cycle);
        finish_iteration_metrics(true, false, false);
        return false;
      }
    }
    // if it is feasible check if all are still integer
    if (n_integers == solution.problem_ptr->n_integer_vars) {
      if (is_feasible) {
        CUOPT_LOG_INFO("[FP_FEASIBLE] Feasible solution found after LP with relative tolerance");
        finish_iteration_metrics(false, false, true);
        return true;
      }
      // if the solution is almost on polytope
      else if (last_distances[0] < distance_to_check_for_feasible) {
        // run the LP with full precision to check if it actually is feasible
        const f_t lp_verify_time_limit = 5.;
        relaxed_lp_settings_t lp_settings;
        lp_settings.time_limit            = lp_verify_time_limit;
        lp_settings.tolerance             = solution.problem_ptr->tolerances.absolute_tolerance;
        lp_settings.return_first_feasible = true;
        lp_settings.save_state            = true;
        // Verifying a candidate we believe is on the polytope; leaving PDLP's infeasibility
        // detection on makes it return a spurious PrimalInfeasible for a near-feasible point.
        lp_settings.check_infeasibility = false;
        const auto verify_begin         = timing_clock::now();
        run_lp_with_vars_fixed(*solution.problem_ptr,
                               solution,
                               solution.problem_ptr->integer_indices,
                               lp_settings,
                               &constraint_prop.bounds_update);
        last_verify_lp_time =
          std::chrono::duration<double>(timing_clock::now() - verify_begin).count();
        is_feasible = solution.get_feasible();
        n_integers  = solution.compute_number_of_integers();
        if (is_feasible && n_integers == solution.problem_ptr->n_integer_vars) {
          CUOPT_LOG_INFO("[FP_FEASIBLE] Feasible solution verified with LP!");
          finish_iteration_metrics(false, false, true);
          return true;
        }
      }
    }
    cuopt_func_call(solution.test_variable_bounds(false));
    if (metrics != nullptr) { solution.handle_ptr->sync_stream(); }
    const auto rounding_begin = timing_clock::now();
    is_feasible               = round(solution);
    if (metrics != nullptr) { solution.handle_ptr->sync_stream(); }
    record_rounding_metrics(
      solution,
      is_feasible,
      std::chrono::duration<double>(timing_clock::now() - rounding_begin).count());
    cuopt_func_call(solution.test_variable_bounds(true));
    proj_and_round_time = proj_begin - timer.remaining_time();
    if (!is_feasible) {
      const f_t time_ratio = 0.2;
      const auto fj_begin  = timing_clock::now();
      is_feasible          = test_fj_feasible(solution, time_ratio * proj_and_round_time);
      last_fj_time         = std::chrono::duration<double>(timing_clock::now() - fj_begin).count();
    }
    if (is_feasible) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      finish_iteration_metrics(false, false, true);
      return true;
    }
    if (timer.check_time_limit()) {
      CUOPT_LOG_INFO("FP time limit reached!");
      finish_iteration_metrics(false, true, false);
      return false;
    }
    // do the cycle check if alpha diff is small enough
    f_t alpha_at_earlier_iter = config.alpha / config.alpha_decrease_factor;
    if (alpha_at_earlier_iter - config.alpha < 0.005) {
      is_cycle = cycle_queue.check_cycle(solution);
    }
    cycle_queue.update_recent_solutions(solution);
    if (is_cycle) {
      CUOPT_LOG_INFO("FP cycle encountered");
      f_t remaining_time_end_fp = timer.remaining_time();
      total_fp_time_until_cycle = fp_fj_cycle_time_begin - remaining_time_end_fp;
      CUOPT_LOG_INFO(
        "remaining_time_end_fp %f fp_fj_cycle_time_begin %f total_fp_time_until_cycle: %f",
        remaining_time_end_fp,
        fp_fj_cycle_time_begin,
        total_fp_time_until_cycle);
      finish_iteration_metrics(true, false, false);
      return false;
    }
    cycle_queue.n_iterations_without_cycle++;
    finish_iteration_metrics(false, false, false);
  }
  // unreachable
  return false;
}

#if MIP_INSTANTIATE_FLOAT
template class feasibility_pump_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class feasibility_pump_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
