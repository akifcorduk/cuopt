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
#include <utilities/logger.hpp>
#include <utilities/timer.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/linalg/binary_op.cuh>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>

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

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::set_projection_host_metadata(
  const std::vector<f_t>& objective, const std::vector<i_t>& integer_indices)
{
  cuopt_expects(
    objective.size() == static_cast<size_t>(context.problem_ptr->n_variables) &&
      integer_indices.size() == static_cast<size_t>(context.problem_ptr->n_integer_vars),
    error_type_t::ValidationError,
    "FP cached host metadata does not match the GPU problem");
  auto& data           = projection_host_data.emplace();
  data.objective       = objective;
  data.integer_indices = integer_indices;
  data.objective_norm  = vector_norm(objective.begin(), objective.end());
  data.assignment.resize(objective.size());
  data.last_projection.resize(objective.size());
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::set_projection_host_bounds(
  const std::vector<typename type_2<f_t>::type>& bounds)
{
  cuopt_expects(projection_host_data && bounds.size() == projection_host_data->objective.size(),
                error_type_t::ValidationError,
                "FP cached host bounds do not match the GPU problem");
  projection_host_data->variable_bounds = bounds;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::copy_projection_host_iterates(
  solution_t<i_t, f_t>& solution,
  std::vector<f_t>& assignment,
  std::vector<f_t>& previous_projection)
{
  auto& data = *projection_host_data;
  cuopt_expects(data.assignment.size() == solution.assignment.size() &&
                  data.last_projection.size() == last_projection.size() &&
                  data.variable_bounds.size() == data.assignment.size(),
                error_type_t::ValidationError,
                "FP cached host inputs require a new model or refreshed node bounds");
  const auto stream     = solution.handle_ptr->get_stream();
  auto* host_assignment = thrust::raw_pointer_cast(data.assignment.data());
  auto* host_projection = thrust::raw_pointer_cast(data.last_projection.data());
  raft::copy(host_assignment, solution.assignment.data(), solution.assignment.size(), stream);
  raft::copy(host_projection, last_projection.data(), last_projection.size(), stream);
  // Both CPU inputs share one completion boundary and persistent pinned storage.
  stream.sync();
  assignment.assign(host_assignment, host_assignment + data.assignment.size());
  previous_projection.assign(host_projection, host_projection + data.last_projection.size());
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
  f_t distance_weight = 1. - config.alpha;
  std::vector<f_t> uncached_objective;
  if (!projection_host_data) {
    uncached_objective = cuopt::host_copy(solution.problem_ptr->objective_coefficients,
                                          solution.handle_ptr->get_stream());
  }
  const auto& obj_vector =
    projection_host_data ? projection_host_data->objective : uncached_objective;
  const f_t l2_norm_of_original_obj = projection_host_data
                                        ? projection_host_data->objective_norm
                                        : vector_norm(obj_vector.begin(), obj_vector.end());
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
  CUOPT_LOG_DEBUG("linear projection of fp");
  std::vector<f_t> h_assignment;
  std::vector<f_t> h_last_projection;
  std::vector<typename type_2<f_t>::type> uncached_bounds;
  std::vector<i_t> uncached_integer_indices;
  if (projection_host_data) {
    copy_projection_host_iterates(solution, h_assignment, h_last_projection);
  } else {
    h_assignment = solution.get_host_assignment();
    uncached_bounds =
      cuopt::host_copy(solution.problem_ptr->variable_bounds, solution.handle_ptr->get_stream());
    h_last_projection = cuopt::host_copy(last_projection, solution.handle_ptr->get_stream());
    uncached_integer_indices =
      cuopt::host_copy(solution.problem_ptr->integer_indices, solution.handle_ptr->get_stream());
  }
  const auto& h_variable_bounds =
    projection_host_data ? projection_host_data->variable_bounds : uncached_bounds;
  const auto& h_integer_indices =
    projection_host_data ? projection_host_data->integer_indices : uncached_integer_indices;
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;
  constraints_delta_t<i_t, f_t> h_constraints;
  variables_delta_t<i_t, f_t> h_variables;
  h_variables.n_vars = solution.problem_ptr->n_variables;
  std::vector<f_t> obj_coefficients(solution.problem_ptr->n_variables, 0.);
  problem_t<i_t, f_t> temp_p(*solution.problem_ptr);
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
  if (config.use_projection_warm_start) {
    // The distance rows above refer to the rounded target. PDLP starts from the
    // node LP solution or the preceding continuous projection instead.
    std::copy(h_last_projection.begin(), h_last_projection.end(), h_assignment.begin());
    // Distance rows can change identity even when their count stays unchanged.
    // Only duals belonging to the stable subproblem rows can be reused.
    auto& dual = solution.lp_state.prev_dual;
    if (dual.size() > static_cast<size_t>(solution.problem_ptr->n_constraints)) {
      thrust::fill(solution.handle_ptr->get_thrust_policy(),
                   dual.begin() + solution.problem_ptr->n_constraints,
                   dual.end(),
                   f_t(0));
    }
  }
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
  if (config.projection_time_limit > 0) {
    time_limit = std::min(time_limit, f_t(config.projection_time_limit));
  }
  if (context.diversity_manager_ptr == nullptr) {
    time_limit = std::max(f_t(0), std::min(time_limit, f_t(timer.remaining_time())));
  }
  const double projection_begin = timer.elapsed_time();
  cuopt_func_call(solution.test_variable_bounds(false));
  relaxed_lp_settings_t lp_settings;
  lp_settings.time_limit          = time_limit;
  lp_settings.tolerance           = lp_tolerance;
  lp_settings.check_infeasibility = false;
  lp_settings.concurrent_halt     = concurrent_halt;
  lp_settings.fused_adaptive_step = config.fused_adaptive_step;
  auto solver_response            = get_relaxed_lp_solution(temp_p, solution, lp_settings);
  cuopt_func_call(solution.test_variable_bounds(false));
  last_lp_time = timer.elapsed_time() - projection_begin;
  total_lp_time += last_lp_time;
  n_lp_calls++;
  CUOPT_LOG_DEBUG("lp_time %f average lp_time %f", last_lp_time, total_lp_time / n_lp_calls);
  solution.assignment.resize(solution.problem_ptr->n_variables, solution.handle_ptr->get_stream());
  raft::copy(last_projection.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  // Projection result might be feasible but not optimal, due to time limits
  bool is_feasible = solution.compute_feasibility();
  if (config.log_descent_diagnostics) {
    const auto& information = solver_response.get_additional_termination_information();
    ++diagnostics.projections;
    diagnostics.pdlp_limit_seconds = time_limit;
    diagnostics.last_fractional =
      solution.problem_ptr->n_integer_vars - solution.n_assigned_integers;
    diagnostics.best_fractional =
      std::min(diagnostics.best_fractional, diagnostics.last_fractional);
    diagnostics.pdlp_status          = static_cast<int>(solver_response.get_termination_status());
    diagnostics.pdlp_steps           = information.number_of_steps_taken;
    diagnostics.pdlp_primal_residual = information.l2_primal_residual;
  }
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
  const double rounding_begin = config.log_descent_diagnostics ? timer.elapsed_time() : 0.;
  bool result;
  CUOPT_LOG_DEBUG("Rounding the point");
  auto round_time =
    std::max(0.05, std::min(config.rounding_time_limit, timer.remaining_time() / 10.));
  if (context.diversity_manager_ptr == nullptr) {
    round_time = std::max(0., std::min(round_time, timer.remaining_time()));
  }
  timer_t bounds_prop_timer(round_time);
  const f_t lp_run_time_after_feasible = 0.;
  bool old_var                         = constraint_prop.round_all_vars;
  f_t old_time                         = constraint_prop.max_time_for_bounds_prop;
  constraint_prop.round_all_vars       = false;
  constraint_prop.max_time_for_bounds_prop =
    context.diversity_manager_ptr == nullptr
      ? std::max(0., std::min(config.propagation_time_limit, timer.remaining_time()))
      : config.propagation_time_limit;
  const double propagation_limit = constraint_prop.max_time_for_bounds_prop;
  result = constraint_prop.apply_round(solution, lp_run_time_after_feasible, bounds_prop_timer);
  constraint_prop.round_all_vars           = old_var;
  constraint_prop.max_time_for_bounds_prop = old_time;
  // result = solution.round_nearest();
  cuopt_func_call(solution.test_variable_bounds(true));
  // copy the last rounding
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  if (result) {
    CUOPT_LOG_DEBUG("New feasible solution with objective %g", solution.get_user_objective());
  }
  if (config.log_descent_diagnostics) {
    const double elapsed = timer.elapsed_time() - rounding_begin;
    diagnostics.rounding_seconds += elapsed;
    if (diagnostics.projections == 1 || diagnostics.projections % 10 == 0) {
      CUOPT_LOG_INFO(
        "GpuSubmipFpRounding job=%llu iter=%d fractional=%d total_excess=%.9g "
        "cutoff_excess=%.9g seconds=%.6f rounding_limit_seconds=%.6f "
        "propagation_limit_seconds=%.6f",
        (unsigned long long)diagnostics.job_id,
        diagnostics.projections,
        int(solution.problem_ptr->n_integer_vars - solution.n_assigned_integers),
        double(solution.h_infeasibility_cost),
        std::max(0., double(solution.h_obj) - diagnostics.cutoff),
        elapsed,
        round_time,
        propagation_limit);
    }
  }
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
  is_feasible                        = fj.solve(solution);
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
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.n_of_minimums_for_exit = 5000;
  fj.settings.time_limit             = std::min(time_limit, timer.remaining_time());
  cuopt_func_call(solution.test_variable_bounds(true));
  const double fj_begin = config.log_descent_diagnostics ? timer.elapsed_time() : 0.;
  is_feasible           = fj.solve(solution);
  if (config.log_descent_diagnostics) { diagnostics.fj_seconds += timer.elapsed_time() - fj_begin; }
  cuopt_func_call(solution.test_variable_bounds(true));
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
  fp_fj_cycle_time_begin = timer.elapsed_time();
  CUOPT_LOG_DEBUG("Running longer FJ on last rounding");
  raft::copy(solution.assignment.data(),
             last_rounding.data(),
             last_rounding.size(),
             solution.handle_ptr->get_stream());
  cuopt_func_call(solution.test_variable_bounds(true));
  cuopt_assert(solution.test_number_all_integer(), "All must be integers before fj");
  is_feasible = run_fj_cycle_escape(solution);
  cuopt_assert(solution.test_number_all_integer(), "All must be integers after fj");
  if (cycle_queue.check_cycle(solution)) {
    CUOPT_LOG_DEBUG("Cycle couldn't be broken. Perturbating FP");
    perturbate(solution);
    is_feasible = solution.get_feasible();
  }
  cycle_queue.n_iterations_without_cycle = 0;
  cycle_queue.update_recent_solutions(solution);
  if (is_feasible) {
    solution.test_feasibility();
    CUOPT_LOG_DEBUG("Feasible found cycle breaking long FJ");
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
  fp_fj_cycle_time_begin    = timer.elapsed_time();
  max_n_of_integers         = 0;
  config.alpha              = config.initial_alpha;
  last_distances.resize(0);
  if (config.log_descent_diagnostics) { diagnostics = {}; }
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
  if (config.log_descent_diagnostics) {
    diagnostics.last_distance = distance_to_last_rounding;
    log_projection_diagnostics(solution);
  }
  return is_cycle;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::log_projection_diagnostics(const solution_t<i_t, f_t>& solution)
{
  if (diagnostics.projections != 1 && diagnostics.projections % 10 != 0) { return; }
  const auto status = optimization_problem_solution_t<i_t, f_t>::get_termination_status_string(
    static_cast<pdlp_termination_status_t>(diagnostics.pdlp_status));
  CUOPT_LOG_INFO(
    "GpuSubmipFpProjection job=%llu iter=%d fractional=%d free_integer=%d distance=%.9g "
    "total_excess=%.9g cutoff_excess=%.9g alpha=%.6g pdlp_status=%s "
    "pdlp_steps=%d pdlp_primal_residual=%.9g pdlp_seconds=%.6f pdlp_limit_seconds=%.6f",
    (unsigned long long)diagnostics.job_id,
    diagnostics.projections,
    diagnostics.last_fractional,
    diagnostics.free_integers,
    diagnostics.last_distance,
    double(solution.h_infeasibility_cost),
    std::max(0., double(solution.h_obj) - diagnostics.cutoff),
    config.alpha,
    status.c_str(),
    diagnostics.pdlp_steps,
    diagnostics.pdlp_primal_residual,
    double(last_lp_time),
    diagnostics.pdlp_limit_seconds);
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::finish_descent(bool feasible, const char* reason)
{
  if (config.log_descent_diagnostics) { diagnostics.exit_reason = reason; }
  return feasible;
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
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream().get());
  solution.problem_ptr->compute_n_integer_vars();
  solution.problem_ptr->compute_binary_var_table();
  CUOPT_LOG_DEBUG("Integers are relaxed n_int vars %d n_binary vars %d n_vars %d",
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
  solution.compute_feasibility();
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::stop_requested()
{
  if (context.diversity_manager_ptr && context.diversity_manager_ptr->check_b_b_preemption()) {
    if (config.log_descent_diagnostics) { diagnostics.exit_reason = "cancel"; }
    return true;
  }
  if ((concurrent_halt && concurrent_halt->load(std::memory_order_acquire)) ||
      context.preempt_heuristic_solver_.load()) {
    if (config.log_descent_diagnostics) { diagnostics.exit_reason = "cancel"; }
    return true;
  }
  const bool expired = timer.check_time_limit();
  if (expired && config.log_descent_diagnostics) { diagnostics.exit_reason = "deadline"; }
  return expired;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_single_fp_descent(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("run_single_fp_descent");
  // start by doing nearest rounding
  solution.round_nearest(rng());
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  while (true) {
    if (stop_requested()) {
      CUOPT_LOG_DEBUG("FP time limit reached!");
      if (context.diversity_manager_ptr != nullptr) { round(solution); }
      return false;
    }
    proj_begin = timer.elapsed_time();
    // pass n_assigned_integers from the previous iteration
    f_t ratio_of_assigned_integers =
      f_t(solution.n_assigned_integers) / solution.problem_ptr->n_integer_vars;
    bool is_feasible = linear_project_onto_polytope(solution, ratio_of_assigned_integers);
    if (context.diversity_manager_ptr == nullptr && stop_requested()) { return false; }
    // Projection already refreshed this count as part of compute_feasibility.
    i_t n_integers = solution.n_assigned_integers;
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
          return finish_descent(true, "feasible_rounding");
        }
        cuopt::default_logger().flush();
        const double elapsed_time_end_fp = timer.elapsed_time();
        total_fp_time_until_cycle        = elapsed_time_end_fp - fp_fj_cycle_time_begin;
        CUOPT_LOG_DEBUG("total_fp_time_until_cycle: %f", total_fp_time_until_cycle);
        return finish_descent(false, "distance_cycle");
      }
    }
    // if it is feasible check if all are still integer
    if (n_integers == solution.problem_ptr->n_integer_vars) {
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Feasible solution found after LP with relative tolerance");
        return finish_descent(true, "feasible_projection");
      }
      // if the solution is almost on polytope
      else if (last_distances[0] < distance_to_check_for_feasible) {
        // run the LP with full precision to check if it actually is feasible
        const f_t lp_verify_time_limit = 5.;
        relaxed_lp_settings_t lp_settings;
        lp_settings.time_limit =
          context.diversity_manager_ptr == nullptr
            ? std::max(0.,
                       std::min(double(context.settings.heuristic_params.relaxed_lp_time_limit),
                                timer.remaining_time()))
            : lp_verify_time_limit;
        lp_settings.concurrent_halt       = concurrent_halt;
        lp_settings.fused_adaptive_step   = config.fused_adaptive_step;
        lp_settings.tolerance             = solution.problem_ptr->tolerances.absolute_tolerance;
        lp_settings.return_first_feasible = true;
        lp_settings.save_state            = true;
        const auto verification_begin     = timer.elapsed_time();
        run_lp_with_vars_fixed(*solution.problem_ptr,
                               solution,
                               solution.problem_ptr->integer_indices,
                               lp_settings,
                               &constraint_prop.bounds_update);
        total_verification_time += timer.elapsed_time() - verification_begin;
        ++n_verification_calls;
        is_feasible = solution.get_feasible();
        n_integers  = solution.compute_number_of_integers();
        if (is_feasible && n_integers == solution.problem_ptr->n_integer_vars) {
          CUOPT_LOG_DEBUG("Feasible solution verified with LP!");
          return finish_descent(true, "feasible_verification");
        }
      }
    }
    cuopt_func_call(solution.test_variable_bounds(false));
    is_feasible = round(solution);
    cuopt_func_call(solution.test_variable_bounds(true));
    if (context.diversity_manager_ptr == nullptr && stop_requested()) { return false; }
    proj_and_round_time = timer.elapsed_time() - proj_begin;
    if (!is_feasible) {
      const f_t time_ratio = 0.2;
      is_feasible          = test_fj_feasible(solution, time_ratio * proj_and_round_time);
    }
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("FP time limit reached!");
      return finish_descent(false, "deadline");
    }
    if (is_feasible) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      return finish_descent(true, "feasible_rounding_or_fj");
    }
    // do the cycle check if alpha diff is small enough
    f_t alpha_at_earlier_iter = config.alpha / config.alpha_decrease_factor;
    if (alpha_at_earlier_iter - config.alpha < 0.005) {
      is_cycle = cycle_queue.check_cycle(solution);
    }
    cycle_queue.update_recent_solutions(solution);
    if (is_cycle) {
      CUOPT_LOG_DEBUG("FP cycle encountered");
      const double elapsed_time_end_fp = timer.elapsed_time();
      total_fp_time_until_cycle        = elapsed_time_end_fp - fp_fj_cycle_time_begin;
      CUOPT_LOG_DEBUG(
        "elapsed_time_end_fp %f fp_fj_cycle_time_begin %f total_fp_time_until_cycle: %f",
        elapsed_time_end_fp,
        fp_fj_cycle_time_begin,
        total_fp_time_until_cycle);
      return finish_descent(false, "integer_cycle");
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
