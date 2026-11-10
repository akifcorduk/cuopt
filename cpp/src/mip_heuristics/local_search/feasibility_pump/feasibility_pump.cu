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

#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <pdlp/pdlp.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/linalg/binary_op.cuh>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>

#include <cuda/functional>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace cuopt::mathematical_optimization::mip {

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
    rng(derive_seed(context.base_seed, rng_id_t::feasibility_pump)),
    timer(20.)
{
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
  const f_t alpha             = fp_experiment::pure_distance() ? f_t(0) : f_t(config.alpha);
  last_alpha_used             = alpha;
  f_t distance_weight         = 1. - alpha;
  std::vector<f_t> obj_vector = cuopt::host_copy(solution.problem_ptr->objective_coefficients,
                                                 solution.handle_ptr->get_stream());
  solution.handle_ptr->sync_stream();
  const f_t l2_norm_of_original_obj = vector_norm(obj_vector.begin(), obj_vector.end());
  const f_t l2_norm_of_distance_obj = vector_norm(dist_objective.begin(), dist_objective.end());
  CUOPT_LOG_TRACE("l2_norm_of_original_obj %f l2_norm_of_distance_obj %f",
                  l2_norm_of_original_obj,
                  l2_norm_of_distance_obj);
  // f_t orig_obj_weight = config.alpha * (l2_norm_of_distance_obj / l2_norm_of_original_obj);
  f_t orig_obj_weight = alpha / l2_norm_of_original_obj;
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
  CUOPT_LOG_DEBUG("linear projection of fp");
  if (config.use_pdfp_projection) {
    auto& problem = *solution.problem_ptr;
    if (!longer_lp_run) { config.alpha *= config.alpha_decrease_factor; }
    const f_t alpha = fp_experiment::pure_distance() ? f_t(0) : f_t(config.alpha);
    last_alpha_used = alpha;
    auto objective =
      cuopt::host_copy(problem.objective_coefficients, solution.handle_ptr->get_stream());
    const f_t objective_norm   = vector_norm(objective.begin(), objective.end());
    const bool paper_tolerance = fp_experiment::paper_tolerance();
    // Two-regime policy: before the first incumbent of this FP run use the fast truncated
    // projection; afterwards use converged paper-style projections with a decaying tolerance.
    const bool converged_mode                      = paper_tolerance || incumbent_in_run;
    constexpr double converged_projection_time_cap = 8.0;
    pdfp_projection_settings_t settings;
    settings.distance_weight =
      problem.n_integer_vars > 0 ? (1. - alpha) / std::sqrt(f_t(problem.n_integer_vars)) : 0.;
    settings.objective_weight = objective_norm > 0. ? alpha / objective_norm : 0.;
    settings.tolerance =
      converged_mode ? projection_tolerance
                     : get_tolerance_from_ratio(ratio_of_set_integers,
                                                context.settings.tolerances.absolute_tolerance);
    settings.relative_primal = converged_mode;
    settings.original_units  = problem.n_integer_vars < problem.n_variables;
    settings.anchor_halpern  = fp_experiment::pdfp_paper_warm_start();
    settings.fused_graph     = fp_experiment::pdfp_cuda_graph();
    const f_t base           = context.settings.heuristic_params.relaxed_lp_time_limit;
    if (paper_tolerance) {
      settings.time_limit = std::max(0.05, timer.remaining_time());
    } else if (incumbent_in_run) {
      settings.time_limit =
        std::max(0.05, std::min(converged_projection_time_cap, timer.remaining_time() / 4.));
    } else {
      settings.time_limit =
        std::max(0.05, std::min(longer_lp_run ? 5. * base : base, timer.remaining_time() / 10.));
    }
    const f_t old_remaining = timer.remaining_time();
    double setup_time       = 0.;
    if (!pdfp_projector) {
      const auto setup_start = std::chrono::steady_clock::now();
      pdfp_projector         = std::make_unique<pdfp_projection_t<i_t, f_t>>(
        problem, fp_experiment::pdfp_spectral_step(), fp_experiment::pdfp_cusparse_spmv());
      pdfp_problem = solution.problem_ptr;
      if (fp_experiment::pdfp_paper_warm_start() && context.diversity_manager_ptr != nullptr &&
          context.diversity_manager_ptr->ls.lp_optimal_exists &&
          lp_optimal_solution.size() == size_t(problem.n_variables)) {
        pdfp_projector->set_cold_start(lp_optimal_solution);
      }
      setup_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count();
    }
    cuopt_assert(pdfp_problem == solution.problem_ptr,
                 "The PDFP projector was built for another problem");
    last_pdfp_result = pdfp_projector->project(solution.assignment, solution.assignment, settings);
    last_lp_time     = old_remaining - timer.remaining_time();
    if (converged_mode) { projection_tolerance = std::max(projection_tolerance * 0.98, 1e-4); }
    raft::copy(last_projection.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
    auto& trace           = last_projection_trace;
    trace.projector       = "pdfp";
    trace.status          = last_pdfp_result.converged                             ? "Optimal"
                            : last_pdfp_result.elapsed_time >= settings.time_limit ? "Time_Limit"
                                                                                   : "Iteration_Limit";
    trace.tolerance       = settings.tolerance;
    trace.time_limit      = settings.time_limit;
    trace.solve_time      = last_pdfp_result.elapsed_time;
    trace.setup_time      = setup_time;
    trace.iterations      = last_pdfp_result.iterations;
    trace.primal_residual = last_pdfp_result.primal_residual;
    trace.dual_residual   = last_pdfp_result.dual_residual;
    trace.relative_gap    = last_pdfp_result.relative_gap;
    trace.n_rows          = problem.n_constraints;
    trace.n_cols          = problem.n_variables;
    trace.n_aux           = 0;
    return solution.compute_feasibility();
  }
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
  cuopt_assert(h_assignment.size() == temp_p.n_variables, "Var count mismatch!");
  cuopt_assert(temp_p.objective_coefficients.size() == temp_p.n_variables, "Var count mismatch!");
  solution.copy_new_assignment(h_assignment);
  cuopt_assert(solution.assignment.size() == temp_p.n_variables, "Var count mismatch!");
  // copy new objective coefficients
  raft::copy(temp_p.objective_coefficients.data(),
             obj_coefficients.data(),
             obj_coefficients.size(),
             solution.handle_ptr->get_stream());
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream().get());
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
  const bool paper_tolerance = fp_experiment::paper_tolerance();
  lp_settings.time_limit = paper_tolerance ? std::max(0.05, timer.remaining_time()) : time_limit;
  lp_settings.tolerance  = paper_tolerance ? projection_tolerance : lp_tolerance;
  lp_settings.relative_criteria = paper_tolerance;
  if (paper_tolerance) {
    // Same right-hand-side norm as PDFP (original rows only, not the epigraph rows).
    const f_t* lower            = solution.problem_ptr->constraint_lower_bounds.data();
    const f_t* upper            = solution.problem_ptr->constraint_upper_bounds.data();
    lp_settings.primal_rhs_norm = std::sqrt(thrust::transform_reduce(
      solution.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(solution.problem_ptr->n_constraints),
      cuda::proclaim_return_type<f_t>([lower, upper] __device__(i_t i) -> f_t {
        const f_t rhs = max(isfinite(lower[i]) ? abs(lower[i]) : f_t(0),
                            isfinite(upper[i]) ? abs(upper[i]) : f_t(0));
        return rhs * rhs;
      }),
      f_t(0),
      thrust::plus<f_t>()));
  }
  lp_settings.check_infeasibility = false;
  const auto lp_start             = std::chrono::steady_clock::now();
  auto solver_response            = get_relaxed_lp_solution(temp_p, solution, lp_settings);
  const double lp_call_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - lp_start).count();
  if (paper_tolerance) { projection_tolerance = std::max(projection_tolerance * 0.98, 1e-4); }
  cuopt_func_call(solution.test_variable_bounds(false));
  last_lp_time = old_remaining - timer.remaining_time();
  lp_time += last_lp_time;
  n_calls++;
  CUOPT_LOG_DEBUG("lp_time %f average lp_time %f", last_lp_time, lp_time / n_calls);
  {
    const auto& info = solver_response.get_additional_termination_information();
    auto& trace      = last_projection_trace;
    trace.projector  = "epigraph";
    trace.status     = solver_response.get_termination_status_string();
    std::replace(trace.status.begin(), trace.status.end(), ' ', '_');
    trace.tolerance       = lp_settings.tolerance;
    trace.time_limit      = lp_settings.time_limit;
    trace.solve_time      = lp_call_time;
    trace.setup_time      = 0.;
    trace.iterations      = info.number_of_steps_taken;
    trace.primal_residual = info.l2_relative_primal_residual;
    trace.dual_residual   = info.l2_relative_dual_residual;
    trace.relative_gap    = info.relative_gap;
    trace.n_rows          = temp_p.n_constraints;
    trace.n_cols          = temp_p.n_variables;
    trace.n_aux           = h_variables.size();
  }
  solution.assignment.resize(solution.problem_ptr->n_variables, solution.handle_ptr->get_stream());
  raft::copy(last_projection.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  // Projection result might be feasible but not optimal, due to time limits
  bool is_feasible = solution.compute_feasibility();
  cuopt_func_call(solution.test_variable_bounds(false));
  if (!is_feasible) {
    CUOPT_LOG_DEBUG("LP is infeasible returning the current PDLP solution! Code %d",
                    (int)solver_response.get_termination_status());
    return false;
  }
  // normal feasible return
  return true;
}

// round will use inevitable infeasibility while propagating the bounds
template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::round(solution_t<i_t, f_t>& solution)
{
  bool result;
  CUOPT_LOG_DEBUG("Rounding the point");
  const auto round_start = std::chrono::steady_clock::now();
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
  const f_t* rounded  = solution.assignment.data();
  const f_t* previous = last_rounding.data();
  const i_t n_flips =
    thrust::count_if(solution.handle_ptr->get_thrust_policy(),
                     solution.problem_ptr->integer_indices.begin(),
                     solution.problem_ptr->integer_indices.end(),
                     [pb = solution.problem_ptr->view(), rounded, previous] __device__(i_t j) {
                       return !pb.integer_equal(rounded[j], previous[j]);
                     });
  // copy the last rounding
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  if (result) {
    CUOPT_LOG_DEBUG("New feasible solution with objective %g", solution.get_user_objective());
  }
  const double round_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - round_start).count();
  ++trace_totals.n_roundings;
  trace_totals.rounding_time += round_time;
  CUOPT_LOG_INFO("FPR src=%s run=%d desc=%d it=%d t=%.3f rtime=%.4f flips=%d feas=%d excess=%.6g",
                 trace_source,
                 trace_run,
                 trace_descent,
                 trace_iteration,
                 trace_time(),
                 round_time,
                 n_flips,
                 (int)result,
                 (double)solution.get_total_excess());
  return result;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::perturbate(solution_t<i_t, f_t>& solution)
{
  constexpr f_t change_ratio = 0.1;
  solution.assign_random_within_bounds(rng(), change_ratio, true);
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
  is_feasible =
    general_integers_relaxed ? fj.solve(solution) : fj.solve(solution, improvement_callback);
  // if FJ didn't change the solution, take last incumbent solution
  if (!is_feasible && cycle_queue.check_cycle(solution)) {
    CUOPT_LOG_DEBUG("cycle detected after FJ, taking last incumbent of fj");
    raft::copy(solution.assignment.data(),
               fj.climbers[0]->incumbent_assignment.data(),
               solution.problem_ptr->n_variables,
               solution.handle_ptr->get_stream());
  }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::test_fj_feasible(solution_t<i_t, f_t>& solution, f_t time_limit)
{
  CUOPT_LOG_DEBUG("Running 20%% with %f time limit", time_limit);
  bool is_feasible;
  const auto fj_start                = std::chrono::steady_clock::now();
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.n_of_minimums_for_exit = 5000;
  fj.settings.time_limit             = std::min(time_limit, timer.remaining_time());
  cuopt_func_call(solution.test_variable_bounds(true));
  is_feasible =
    general_integers_relaxed ? fj.solve(solution) : fj.solve(solution, improvement_callback);
  cuopt_func_call(solution.test_variable_bounds(true));
  const double fj_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - fj_start).count();
  trace_totals.fj_time += fj_time;
  CUOPT_LOG_INFO("FPJ src=%s run=%d desc=%d it=%d t=%.3f fjtime=%.4f tlim=%.4f feas=%d",
                 trace_source,
                 trace_run,
                 trace_descent,
                 trace_iteration,
                 trace_time(),
                 fj_time,
                 (double)fj.settings.time_limit,
                 (int)is_feasible);
  // if FJ didn't change the solution, take last incumbent solution
  if (!is_feasible) {
    raft::copy(solution.assignment.data(),
               last_rounding.data(),
               solution.problem_ptr->n_variables,
               solution.handle_ptr->get_stream());
    cuopt_func_call(solution.test_variable_bounds(true));
  } else {
    CUOPT_LOG_DEBUG("20%% FJ run found feasible!");
  }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::handle_cycle(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("handle_cycle");
  CUOPT_LOG_DEBUG("running handle cycle");
  bool is_feasible       = false;
  const auto cycle_start = std::chrono::steady_clock::now();
  fp_fj_cycle_time_begin = timer.remaining_time();
  CUOPT_LOG_DEBUG("Running longer FJ on last rounding");
  raft::copy(solution.assignment.data(),
             last_rounding.data(),
             last_rounding.size(),
             solution.handle_ptr->get_stream());
  cuopt_func_call(solution.test_variable_bounds(true));
  cuopt_assert(solution.test_number_all_integer(), "All must be integers before fj");
  is_feasible = run_fj_cycle_escape(solution);
  cuopt_assert(solution.test_number_all_integer(), "All must be integers after fj");
  const bool fj_feasible = is_feasible;
  bool perturbed         = false;
  if (cycle_queue.check_cycle(solution)) {
    CUOPT_LOG_DEBUG("Cycle couldn't be broken. Perturbating FP");
    perturbate(solution);
    is_feasible = solution.get_feasible();
    perturbed   = true;
  }
  cycle_queue.n_iterations_without_cycle = 0;
  cycle_queue.update_recent_solutions(solution);
  if (is_feasible) {
    solution.test_feasibility();
    CUOPT_LOG_DEBUG("Feasible found cycle breaking long FJ");
  }
  const double cycle_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - cycle_start).count();
  ++trace_totals.n_cycle_restarts;
  trace_totals.cycle_time += cycle_time;
  CUOPT_LOG_INFO("FPX src=%s run=%d desc=%d t=%.3f ctime=%.4f fjfeas=%d perturbed=%d feas=%d",
                 trace_source,
                 trace_run,
                 trace_descent,
                 trace_time(),
                 cycle_time,
                 (int)fj_feasible,
                 (int)perturbed,
                 (int)is_feasible);
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
  ++trace_run;
  projection_tolerance = 1e-2;
  incumbent_in_run     = false;
  pdfp_projector.reset();
  best_excess               = std::numeric_limits<f_t>::infinity();
  total_fp_time_until_cycle = 0;
  fp_fj_cycle_time_begin    = timer.remaining_time();
  max_n_of_integers         = 0;
  config.alpha              = default_alpha;
  last_distances.resize(0);
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::resize_vectors(problem_t<i_t, f_t>& problem,
                                                  const raft::handle_t* handle_ptr)
{
  pdfp_projector.reset();
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
      CUOPT_LOG_DEBUG("Distance cycle detected curr %f avg %f for last %d iter",
                      distance_to_last_rounding,
                      avg_distance,
                      last_distances.size());
      is_cycle = true;
    }
    last_distances.pop_back();
  } else {
    CUOPT_LOG_DEBUG("Distance of projection: %f", distance_to_last_rounding);
  }
  last_distances.push_front(distance_to_last_rounding);
  return is_cycle;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::relax_general_integers(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("relax_general_integers");
  cuopt_assert(!general_integers_relaxed, "General integers are already relaxed");
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
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream().get());
  solution.problem_ptr->compute_n_integer_vars();
  solution.problem_ptr->compute_binary_var_table();
  general_integers_relaxed = true;
  CUOPT_LOG_DEBUG("Integers are relaxed n_int vars %d n_binary vars %d n_vars %d",
                  solution.problem_ptr->n_integer_vars,
                  solution.problem_ptr->n_binary_vars,
                  solution.problem_ptr->n_variables);
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::revert_relaxation(solution_t<i_t, f_t>& solution)
{
  cuopt_assert(general_integers_relaxed, "General integers are not relaxed");
  cuopt_assert(orig_variable_types.size() == solution.problem_ptr->variable_types.size(),
               "variable size mismatch");
  std::swap(orig_variable_types, solution.problem_ptr->variable_types);
  solution.problem_ptr->compute_n_integer_vars();
  solution.problem_ptr->compute_binary_var_table();
  general_integers_relaxed = false;
  solution.compute_feasibility();
}

template <typename i_t, typename f_t>
double feasibility_pump_t<i_t, f_t>::trace_time()
{
  if (context.diversity_manager_ptr == nullptr) { return -1.; }
  return context.diversity_manager_ptr->timer.elapsed_time();
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::trace_projection(solution_t<i_t, f_t>& solution, i_t n_integers)
{
  auto& problem      = *solution.problem_ptr;
  auto policy        = solution.handle_ptr->get_thrust_policy();
  const f_t int_tol  = context.settings.tolerances.integrality_tolerance;
  const f_t* x       = solution.assignment.data();
  const f_t* target  = last_rounding.data();
  const f_t distance = compute_l1_distance<i_t, f_t>(
    problem.integer_indices, last_rounding, solution.assignment, solution.handle_ptr);
  const i_t n_match = thrust::count_if(
    policy,
    problem.integer_indices.begin(),
    problem.integer_indices.end(),
    [x, target, int_tol] __device__(i_t j) { return abs(x[j] - target[j]) <= int_tol; });
  const f_t* lower_excess   = solution.lower_excess.data();
  const f_t* upper_excess   = solution.upper_excess.data();
  const f_t l2_violation    = std::sqrt(thrust::transform_reduce(
    policy,
    thrust::make_counting_iterator<i_t>(0),
    thrust::make_counting_iterator<i_t>(problem.n_constraints),
    cuda::proclaim_return_type<f_t>([lower_excess, upper_excess] __device__(i_t i) -> f_t {
      const f_t violation = max(lower_excess[i], upper_excess[i]);
      return violation * violation;
    }),
    f_t(0),
    thrust::plus<f_t>()));
  const f_t max_violation   = solution.compute_max_constraint_violation();
  const f_t total_violation = solution.get_total_excess();
  const i_t n_violated      = problem.n_constraints -
                         solution.n_feasible_constraints.value(solution.handle_ptr->get_stream());
  const f_t distance_reduction =
    trace_last_distance > 0 ? (trace_last_distance - distance) / trace_last_distance : NAN;
  trace_last_distance = distance;

  const auto& p = last_projection_trace;
  ++trace_totals.n_projections;
  if (p.status == "Optimal") { ++trace_totals.n_converged; }
  trace_totals.projection_time += p.total_time;
  if (std::isfinite(p.solve_time)) { trace_totals.solve_time += p.solve_time; }
  trace_totals.setup_time += p.setup_time;
  CUOPT_LOG_INFO(
    "FPT src=%s run=%d desc=%d it=%d t=%.3f stage=%s proj=%s alpha=%.4g tol=%.3g tlim=%.3g "
    "ptime=%.4f stime=%.4f setup=%.4f iters=%d status=%s nint=%d integral=%d match=%d "
    "dist=%.6g dred=%.4g pinf_max=%.3e pinf_l2=%.3e pinf_sum=%.3e nviol=%d pres=%.3e dres=%.3e "
    "gap=%.3e obj=%.10g rows=%d cols=%d naux=%d",
    trace_source,
    trace_run,
    trace_descent,
    trace_iteration,
    trace_time(),
    general_integers_relaxed ? "binary" : "all",
    p.projector,
    (double)last_alpha_used,
    p.tolerance,
    p.time_limit,
    p.total_time,
    p.solve_time,
    p.setup_time,
    p.iterations,
    p.status.c_str(),
    problem.n_integer_vars,
    n_integers,
    n_match,
    (double)distance,
    (double)distance_reduction,
    (double)max_violation,
    (double)l2_violation,
    (double)total_violation,
    n_violated,
    p.primal_residual,
    p.dual_residual,
    p.relative_gap,
    (double)solution.get_user_objective(),
    p.n_rows,
    p.n_cols,
    p.n_aux);
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::log_trace_summary()
{
  const auto& s = trace_totals;
  CUOPT_LOG_INFO(
    "FPSUM src=%s runs=%d descents=%ld feasible_descents=%ld projections=%ld converged=%ld "
    "proj_time=%.3f solve_time=%.3f setup_time=%.3f roundings=%ld round_time=%.3f fj20_time=%.3f "
    "cycle_restarts=%ld cycle_time=%.3f",
    trace_source,
    trace_run,
    s.n_descents,
    s.n_feasible_descents,
    s.n_projections,
    s.n_converged,
    s.projection_time,
    s.solve_time,
    s.setup_time,
    s.n_roundings,
    s.rounding_time,
    s.fj_time,
    s.n_cycle_restarts,
    s.cycle_time);
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_single_fp_descent(solution_t<i_t, f_t>& solution)
{
  ++trace_descent;
  trace_iteration     = 0;
  trace_last_distance = NAN;
  trace_exit          = "";
  const double start  = trace_time();
  const bool feasible = run_single_fp_descent_impl(solution);
  if (feasible) { incumbent_in_run = true; }
  const double end = trace_time();
  ++trace_totals.n_descents;
  if (feasible) { ++trace_totals.n_feasible_descents; }
  CUOPT_LOG_INFO(
    "FPD src=%s run=%d desc=%d t=%.3f dur=%.3f iters=%d exit=%s feas=%d obj=%.10g nint=%d "
    "stage=%s",
    trace_source,
    trace_run,
    trace_descent,
    end,
    end - start,
    trace_iteration,
    trace_exit,
    (int)feasible,
    feasible ? (double)solution.get_user_objective() : NAN,
    solution.problem_ptr->n_integer_vars,
    general_integers_relaxed ? "binary" : "all");
  return feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_single_fp_descent_impl(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("run_single_fp_descent");
  // start by doing nearest rounding
  solution.round_nearest(rng());
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  while (true) {
    if (context.diversity_manager_ptr->check_b_b_preemption() || timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("FP time limit reached!");
      round(solution);
      trace_exit = "limit";
      return false;
    }
    ++trace_iteration;
    proj_begin = timer.remaining_time();
    // pass n_assigned_integers from the previous iteration
    f_t ratio_of_assigned_integers =
      f_t(solution.n_assigned_integers) / solution.problem_ptr->n_integer_vars;
    const auto projection_start = std::chrono::steady_clock::now();
    bool is_feasible = linear_project_onto_polytope(solution, ratio_of_assigned_integers);
    last_projection_trace.total_time =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - projection_start).count();
    i_t n_integers = solution.compute_number_of_integers();
    trace_projection(solution, n_integers);
    CUOPT_LOG_DEBUG("after fp projection n_integers %d total n_integes %d",
                    n_integers,
                    solution.problem_ptr->n_integer_vars);
    bool is_cycle = true;
    // temp comment for presolve run
    if (config.check_distance_cycle) {
      // use distance cycle if we are running ii or objective FP
      is_cycle = check_distance_cycle(solution);
      if (is_cycle) {
        is_feasible = round(solution);
        cuopt_func_call(solution.test_variable_bounds(true));
        if (is_feasible) {
          bool res = solution.compute_feasibility();
          cuopt_assert(res, "Feasibility issue");
          trace_exit = "distance_cycle_round";
          return true;
        }
        cuopt::default_logger().flush();
        f_t remaining_time_end_fp = timer.remaining_time();
        total_fp_time_until_cycle = fp_fj_cycle_time_begin - remaining_time_end_fp;
        CUOPT_LOG_DEBUG("total_fp_time_until_cycle: %f", total_fp_time_until_cycle);
        trace_exit = "distance_cycle";
        return false;
      }
    }
    // if it is feasible check if all are still integer
    if (n_integers == solution.problem_ptr->n_integer_vars) {
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Feasible solution found after LP with relative tolerance");
        trace_exit = "projection";
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
        run_lp_with_vars_fixed(*solution.problem_ptr,
                               solution,
                               solution.problem_ptr->integer_indices,
                               lp_settings,
                               &constraint_prop.bounds_update);
        is_feasible = solution.get_feasible();
        n_integers  = solution.compute_number_of_integers();
        if (is_feasible && n_integers == solution.problem_ptr->n_integer_vars) {
          CUOPT_LOG_DEBUG("Feasible solution verified with LP!");
          trace_exit = "fixed_lp";
          return true;
        }
      }
    }
    cuopt_func_call(solution.test_variable_bounds(false));
    is_feasible = round(solution);
    cuopt_func_call(solution.test_variable_bounds(true));
    proj_and_round_time       = proj_begin - timer.remaining_time();
    const bool round_feasible = is_feasible;
    if (!is_feasible) {
      const f_t time_ratio = 0.2;
      is_feasible          = test_fj_feasible(solution, time_ratio * proj_and_round_time);
    }
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("FP time limit reached!");
      trace_exit = "limit";
      return false;
    }
    if (is_feasible) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      trace_exit = round_feasible ? "round" : "fj20";
      return true;
    }
    // do the cycle check if alpha diff is small enough
    f_t alpha_at_earlier_iter = config.alpha / config.alpha_decrease_factor;
    if (alpha_at_earlier_iter - config.alpha < 0.005) {
      is_cycle = cycle_queue.check_cycle(solution);
    }
    cycle_queue.update_recent_solutions(solution);
    if (is_cycle) {
      CUOPT_LOG_DEBUG("FP cycle encountered");
      f_t remaining_time_end_fp = timer.remaining_time();
      total_fp_time_until_cycle = fp_fj_cycle_time_begin - remaining_time_end_fp;
      CUOPT_LOG_DEBUG(
        "remaining_time_end_fp %f fp_fj_cycle_time_begin %f total_fp_time_until_cycle: %f",
        remaining_time_end_fp,
        fp_fj_cycle_time_begin,
        total_fp_time_until_cycle);
      trace_exit = "cycle";
      return false;
    }
    cycle_queue.n_iterations_without_cycle++;
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
