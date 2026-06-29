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

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>

#include <cmath>

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

namespace cuopt::linear_programming::detail {

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
    rng(cuopt::seed_generator::get_seed()),
    timer(20.),
    reseed_points(0, context.problem_ptr->handle_ptr->get_stream())
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
  CUOPT_LOG_DEBUG("linear projection of fp");
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
  auto solver_response            = get_relaxed_lp_solution(temp_p, solution, lp_settings);
  cuopt_func_call(solution.test_variable_bounds(false));
  last_lp_time = old_remaining - timer.remaining_time();
  lp_time += last_lp_time;
  n_calls++;
  CUOPT_LOG_DEBUG("lp_time %f average lp_time %f", last_lp_time, lp_time / n_calls);
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
  // copy the last rounding
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  if (result) {
    CUOPT_LOG_DEBUG("New feasible solution with objective %g", solution.get_user_objective());
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
  is_feasible = fj.solve(solution);
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
  fp_fj_cycle_time_begin    = timer.remaining_time();
  max_n_of_integers         = 0;
  config.alpha              = default_alpha;
  last_distances.resize(0);
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
bool feasibility_pump_t<i_t, f_t>::run_single_fp_descent(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("run_single_fp_descent");
  // start by doing nearest rounding
  solution.round_nearest();
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  while (true) {
    if (context.diversity_manager_ptr->check_b_b_preemption() || timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("FP time limit reached!");
      round(solution);
      return false;
    }
    proj_begin = timer.remaining_time();
    // pass n_assigned_integers from the previous iteration
    f_t ratio_of_assigned_integers =
      f_t(solution.n_assigned_integers) / solution.problem_ptr->n_integer_vars;
    bool is_feasible = linear_project_onto_polytope(solution, ratio_of_assigned_integers);
    i_t n_integers   = solution.compute_number_of_integers();
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
          return true;
        }
        cuopt::default_logger().flush();
        f_t remaining_time_end_fp = timer.remaining_time();
        total_fp_time_until_cycle = fp_fj_cycle_time_begin - remaining_time_end_fp;
        CUOPT_LOG_DEBUG("total_fp_time_until_cycle: %f", total_fp_time_until_cycle);
        return false;
      }
    }
    // if it is feasible check if all are still integer
    if (n_integers == solution.problem_ptr->n_integer_vars) {
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Feasible solution found after LP with relative tolerance");
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
          return true;
        }
      }
    }
    cuopt_func_call(solution.test_variable_bounds(false));
    is_feasible = round(solution);
    cuopt_func_call(solution.test_variable_bounds(true));
    proj_and_round_time = proj_begin - timer.remaining_time();
    if (!is_feasible) {
      const f_t time_ratio = 0.2;
      is_feasible          = test_fj_feasible(solution, time_ratio * proj_and_round_time);
    }
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("FP time limit reached!");
      return false;
    }
    if (is_feasible) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
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
      return false;
    }
    cycle_queue.n_iterations_without_cycle++;
  }
  // unreachable
  return false;
}

// ---------------------------------------------------------------------------
// Batched-PDLP feasibility pump (cloud projection)
// ---------------------------------------------------------------------------

// Build the fixed unified projection problem once: for every integer variable j we always create
// an aux distance var d_j >= 0 (upper (u_j - l_j) + int_tol) and two abs-value constraints
//   d_j - x_j >= -val_j   and   d_j + x_j >= val_j
// so that d_j = |x_j - val_j| at optimum. The structure is shared across all climbers and outer
// iterations; only the two constraints' lower bounds (+/- val_j) and the alpha-blended objective
// change later.
template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::build_unified_projection_problem(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("build_unified_projection_problem");
  auto* pb          = solution.problem_ptr;
  auto stream       = solution.handle_ptr->get_stream();
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;

  unified_n_vars          = pb->n_variables;
  unified_n_constr        = pb->n_constraints;
  h_integer_indices_cache = cuopt::host_copy(pb->integer_indices, stream);
  unified_n_int           = (i_t)h_integer_indices_cache.size();
  unified_n_vars_total    = unified_n_vars + unified_n_int;
  unified_n_constr_total  = unified_n_constr + 2 * unified_n_int;

  auto h_offsets          = cuopt::host_copy(pb->offsets, stream);
  auto h_indices          = cuopt::host_copy(pb->variables, stream);
  auto h_values           = cuopt::host_copy(pb->coefficients, stream);
  auto h_var_bounds       = cuopt::host_copy(pb->variable_bounds, stream);
  h_base_constraint_lower = cuopt::host_copy(pb->constraint_lower_bounds, stream);
  h_base_constraint_upper = cuopt::host_copy(pb->constraint_upper_bounds, stream);
  solution.handle_ptr->sync_stream();

  h_var_lower.resize(unified_n_vars);
  h_var_upper.resize(unified_n_vars);
  std::vector<f_t> vlb(unified_n_vars_total);
  std::vector<f_t> vub(unified_n_vars_total);
  std::vector<var_t> vtypes(unified_n_vars_total, var_t::CONTINUOUS);
  for (i_t j = 0; j < unified_n_vars; ++j) {
    h_var_lower[j] = get_lower(h_var_bounds[j]);
    h_var_upper[j] = get_upper(h_var_bounds[j]);
    vlb[j]         = h_var_lower[j];
    vub[j]         = h_var_upper[j];
  }
  for (i_t k = 0; k < unified_n_int; ++k) {
    i_t col                    = h_integer_indices_cache[k];
    vlb[unified_n_vars + k]    = 0.;
    vub[unified_n_vars + k]    = (h_var_upper[col] - h_var_lower[col]) + int_tol;
  }

  std::vector<i_t> offsets;
  std::vector<i_t> indices;
  std::vector<f_t> values;
  offsets.reserve(unified_n_constr_total + 1);
  indices.reserve(h_values.size() + 4 * unified_n_int);
  values.reserve(h_values.size() + 4 * unified_n_int);
  for (i_t r = 0; r < unified_n_constr; ++r) {
    offsets.push_back((i_t)values.size());
    for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
      indices.push_back(h_indices[p]);
      values.push_back(h_values[p]);
    }
  }

  std::vector<f_t> clb(unified_n_constr_total);
  std::vector<f_t> cub(unified_n_constr_total);
  for (i_t r = 0; r < unified_n_constr; ++r) {
    clb[r] = h_base_constraint_lower[r];
    cub[r] = h_base_constraint_upper[r];
  }
  for (i_t k = 0; k < unified_n_int; ++k) {
    i_t col_x = h_integer_indices_cache[k];
    i_t col_d = unified_n_vars + k;
    // C1: d - x >= -val (columns sorted ascending: x first, then aux d)
    offsets.push_back((i_t)values.size());
    indices.push_back(col_x);
    values.push_back(-1.);
    indices.push_back(col_d);
    values.push_back(1.);
    // C2: d + x >= val
    offsets.push_back((i_t)values.size());
    indices.push_back(col_x);
    values.push_back(1.);
    indices.push_back(col_d);
    values.push_back(1.);
    clb[unified_n_constr + 2 * k]     = 0.;
    cub[unified_n_constr + 2 * k]     = (f_t)default_cont_upper;
    clb[unified_n_constr + 2 * k + 1] = 0.;
    cub[unified_n_constr + 2 * k + 1] = (f_t)default_cont_upper;
  }
  offsets.push_back((i_t)values.size());

  // Initial (distance-only) objective so the device vector is sized n_vars_total; rebuilt each
  // outer iteration with the alpha-blended objective.
  std::vector<f_t> obj(unified_n_vars_total, 0.);
  for (i_t k = 0; k < unified_n_int; ++k) {
    obj[unified_n_vars + k] = 1.;
  }

  unified_problem =
    std::make_unique<cuopt::linear_programming::optimization_problem_t<i_t, f_t>>(
      solution.handle_ptr);
  auto& op = *unified_problem;
  op.set_maximize(false);
  op.set_csr_constraint_matrix(values.data(),
                               (i_t)values.size(),
                               indices.data(),
                               (i_t)indices.size(),
                               offsets.data(),
                               (i_t)offsets.size());
  op.set_constraint_lower_bounds(clb.data(), (i_t)clb.size());
  op.set_constraint_upper_bounds(cub.data(), (i_t)cub.size());
  op.set_objective_coefficients(obj.data(), (i_t)obj.size());
  op.set_variable_lower_bounds(vlb.data(), (i_t)vlb.size());
  op.set_variable_upper_bounds(vub.data(), (i_t)vub.size());
  op.set_variable_types(vtypes.data(), (i_t)vtypes.size());
  CUOPT_LOG_DEBUG(
    "Built unified projection problem: n_vars %d (+%d aux), n_constr %d (+%d abs-value)",
    unified_n_vars,
    unified_n_int,
    unified_n_constr,
    2 * unified_n_int);
}

template <typename i_t, typename f_t>
i_t feasibility_pump_t<i_t, f_t>::compute_cloud_batch_size(solution_t<i_t, f_t>& solution)
{
  cuopt_assert(unified_problem != nullptr, "Unified problem must be built first");
  // Per-climber constraint bounds (the +/- val_j rows) drive the dominant memory term; the
  // objective is shared across climbers.
  const size_t mem_cap = cuopt::linear_programming::compute_optimal_batch_size(
    *unified_problem, /*per_climber_objectives=*/false, /*per_climber_constraint_bounds=*/true);
  if (mem_cap < (size_t)batch_config.fallback_threshold) {
    CUOPT_LOG_DEBUG("Cloud batch size memory cap %zu below fallback threshold %d; using single FP",
                    mem_cap,
                    batch_config.fallback_threshold);
    return 0;
  }
  i_t bs = (i_t)std::min<size_t>(mem_cap, (size_t)batch_config.target_max_batch_size);
  CUOPT_LOG_DEBUG("Cloud batch size %d (memory cap %zu, target [%d, %d])",
                  bs,
                  mem_cap,
                  batch_config.target_min_batch_size,
                  batch_config.target_max_batch_size);
  return bs;
}

template <typename i_t, typename f_t>
i_t feasibility_pump_t<i_t, f_t>::assemble_cloud(solution_t<i_t, f_t>& solution,
                                                 i_t batch_size,
                                                 bool first_iteration,
                                                 rmm::device_uvector<f_t>& d_cloud,
                                                 bool& seed_found_feasible)
{
  raft::common::nvtx::range fun_scope("assemble_cloud");
  auto stream         = solution.handle_ptr->get_stream();
  const f_t int_tol   = context.settings.tolerances.integrality_tolerance;
  const i_t n_vars    = unified_n_vars;
  seed_found_feasible = false;

  auto round_to_int = [&](f_t v, f_t lb, f_t ub) -> f_t {
    f_t int_lb = std::ceil(lb - int_tol);
    f_t int_ub = std::floor(ub + int_tol);
    f_t r      = std::round(v);
    if (r < int_lb) r = int_lb;
    if (r > int_ub) r = int_ub;
    return r;
  };

  std::vector<f_t> h_points((size_t)batch_size * n_vars, 0.);
  i_t count = 0;

  auto append_point = [&](const std::vector<f_t>& pt) {
    if (count >= batch_size) return;
    std::copy(pt.begin(), pt.begin() + n_vars, h_points.begin() + (size_t)count * n_vars);
    count++;
  };

  // (1) Promising reseed points carried from the previous iteration. Capped at reseed_fraction of
  // the cloud so fresh FJ-trajectory points and padding always add diversity.
  if (!first_iteration && reseed_count > 0) {
    auto h_reseed = cuopt::host_copy(reseed_points, stream);
    solution.handle_ptr->sync_stream();
    i_t reseed_cap = std::max(1, (i_t)std::floor(batch_config.reseed_fraction * batch_size));
    i_t take       = std::min(reseed_count, reseed_cap);
    for (i_t c = 0; c < take && count < batch_size; ++c) {
      std::copy(h_reseed.begin() + (size_t)c * n_vars,
                h_reseed.begin() + (size_t)(c + 1) * n_vars,
                h_points.begin() + (size_t)count * n_vars);
      count++;
    }
  }

  // (2) Fresh 20% FJ trajectory points captured from a single descent (Option B).
  if (count < batch_size) {
    cuopt_func_call(solution.test_variable_bounds(true));
    solution.round_nearest();
    cuopt_assert(solution.test_number_all_integer(), "Cloud FJ seed must be integer");
    f_t fj_time =
      first_iteration
        ? std::max(0.05, std::min(1.0, timer.remaining_time() / 20.))
        : std::max(0.05, batch_config.fj_seed_time_ratio * std::max(proj_and_round_time, 0.05));
    fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
    fj.settings.update_weights         = true;
    fj.settings.feasibility_run        = true;
    fj.settings.n_of_minimums_for_exit = 5000;
    fj.settings.time_limit             = std::min(fj_time, timer.remaining_time());
    fj.settings.record_trajectory      = true;
    fj.set_trajectory_capacity(batch_size - count, stream);
    // The seeding 20% FJ both records its visited trajectory (cloud seeds) AND may itself reach a
    // feasible solution; if it does, signal the caller to exit instead of projecting a cloud.
    bool fj_feasible = fj.solve(solution);
    fj.settings.record_trajectory = false;
    if (fj_feasible && solution.compute_feasibility()) { seed_found_feasible = true; }
    i_t traj = fj.get_trajectory_count();
    if (traj > 0) {
      auto h_traj = cuopt::host_copy(fj.get_trajectory_buffer(), stream);
      solution.handle_ptr->sync_stream();
      for (i_t t = 0; t < traj && count < batch_size; ++t) {
        std::copy(h_traj.begin() + (size_t)t * n_vars,
                  h_traj.begin() + (size_t)(t + 1) * n_vars,
                  h_points.begin() + (size_t)count * n_vars);
        count++;
      }
    }
  }

  // (3) Pad to batch_size with perturbed nearest-roundings of the LP optimal.
  if (count < batch_size && lp_optimal_solution.size() == (size_t)n_vars) {
    auto h_lp_opt = cuopt::host_copy(lp_optimal_solution, stream);
    solution.handle_ptr->sync_stream();
    std::vector<f_t> base(n_vars);
    for (i_t j = 0; j < n_vars; ++j) {
      base[j] = h_lp_opt[j];
    }
    for (i_t k = 0; k < unified_n_int; ++k) {
      i_t col  = h_integer_indices_cache[k];
      base[col] = round_to_int(h_lp_opt[col], h_var_lower[col], h_var_upper[col]);
    }
    std::uniform_real_distribution<double> unit(0., 1.);
    constexpr double perturb_ratio = 0.1;
    while (count < batch_size) {
      std::vector<f_t> pt = base;
      for (i_t k = 0; k < unified_n_int; ++k) {
        if (unit(rng) >= perturb_ratio) continue;
        i_t col    = h_integer_indices_cache[k];
        f_t int_lb = std::ceil(h_var_lower[col] - int_tol);
        f_t int_ub = std::floor(h_var_upper[col] + int_tol);
        if (int_ub < int_lb) continue;
        f_t span = int_ub - int_lb;
        f_t val  = int_lb + std::floor(unit(rng) * (span + 1.));
        if (val > int_ub) val = int_ub;
        pt[col] = val;
      }
      append_point(pt);
    }
  }

  if (count == 0) return 0;
  d_cloud.resize((size_t)count * n_vars, stream);
  raft::copy(d_cloud.data(), h_points.data(), (size_t)count * n_vars, stream);
  solution.handle_ptr->sync_stream();
  return count;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::project_cloud(solution_t<i_t, f_t>& solution,
                                                 i_t n_points,
                                                 const rmm::device_uvector<f_t>& d_cloud,
                                                 rmm::device_uvector<f_t>& d_projected)
{
  raft::common::nvtx::range fun_scope("project_cloud");
  auto stream      = solution.handle_ptr->get_stream();
  auto& op         = *unified_problem;
  const i_t n_vars = unified_n_vars;
  const i_t nct    = unified_n_constr_total;

  // Shared alpha-blended objective (distance part + alpha * original objective).
  std::vector<f_t> obj(unified_n_vars_total, 0.);
  for (i_t k = 0; k < unified_n_int; ++k) {
    obj[unified_n_vars + k] = 1.;
  }
  adjust_objective_with_original(solution, obj);
  raft::copy(op.get_objective_coefficients().data(), obj.data(), obj.size(), stream);

  // Per-climber constraint bounds expanded directly on device (this is the dominant per-iteration
  // term, 2 * n_points * n_constr_total): original rows reuse the problem's shared bounds, the
  // abs-value rows carry +/- val_j read straight from the device cloud. No host round-trip.
  auto& d_clb = op.get_constraint_lower_bounds();
  auto& d_cub = op.get_constraint_upper_bounds();
  d_clb.resize((size_t)n_points * nct, stream);
  d_cub.resize((size_t)n_points * nct, stream);

  const i_t n_constr      = unified_n_constr;
  const f_t cont_upper    = (f_t)default_cont_upper;
  const f_t* base_clb_ptr = solution.problem_ptr->constraint_lower_bounds.data();
  const f_t* base_cub_ptr = solution.problem_ptr->constraint_upper_bounds.data();
  const i_t* int_idx_ptr  = solution.problem_ptr->integer_indices.data();
  const f_t* cloud_ptr    = d_cloud.data();
  f_t* clb_ptr            = d_clb.data();
  f_t* cub_ptr            = d_cub.data();
  thrust::for_each(
    solution.handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>((size_t)n_points * nct),
    [=] __device__(size_t g) {
      const size_t c = g / (size_t)nct;
      const i_t r    = (i_t)(g % (size_t)nct);
      if (r < n_constr) {
        clb_ptr[g] = base_clb_ptr[r];
        cub_ptr[g] = base_cub_ptr[r];
      } else {
        const i_t rr  = r - n_constr;
        const i_t k   = rr / 2;
        const i_t col = int_idx_ptr[k];
        const f_t val = cloud_ptr[c * (size_t)n_vars + (size_t)col];
        clb_ptr[g]    = (rr & 1) ? val : -val;
        cub_ptr[g]    = cont_upper;
      }
    });

  pdlp_solver_settings_t<i_t, f_t> settings;
  settings.method                          = cuopt::linear_programming::method_t::PDLP;
  settings.presolver                       = presolver_t::None;
  settings.fixed_batch_size                = n_points;
  settings.generate_batch_primal_dual_solution = true;
  const f_t rlp_base = context.settings.heuristic_params.relaxed_lp_time_limit;
  settings.time_limit =
    std::max(0.05, std::min((double)rlp_base, timer.remaining_time() / 10.));
  settings.set_optimality_tolerance(1e-4);

  auto sol      = cuopt::linear_programming::run_batch_pdlp(op, settings);
  auto& primal  = sol.get_primal_solution();
  if (primal.size() != (size_t)n_points * unified_n_vars_total) {
    CUOPT_LOG_DEBUG("Batch projection produced no usable primal (size %zu, expected %zu)",
                    primal.size(),
                    (size_t)n_points * unified_n_vars_total);
    return false;
  }
  d_projected.resize((size_t)n_points * n_vars, stream);
  for (i_t c = 0; c < n_points; ++c) {
    raft::copy(d_projected.data() + (size_t)c * n_vars,
               primal.data() + (size_t)c * unified_n_vars_total,
               n_vars,
               stream);
  }
  solution.handle_ptr->sync_stream();
  return true;
}

template <typename i_t, typename f_t>
i_t feasibility_pump_t<i_t, f_t>::select_cloud_point(solution_t<i_t, f_t>& solution,
                                                     i_t n_points,
                                                     const rmm::device_uvector<f_t>& d_cloud,
                                                     const rmm::device_uvector<f_t>& d_projected)
{
  raft::common::nvtx::range fun_scope("select_cloud_point");
  auto stream       = solution.handle_ptr->get_stream();
  const i_t n_vars  = unified_n_vars;
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;

  auto h_cloud     = cuopt::host_copy(d_cloud, stream);
  auto h_projected = cuopt::host_copy(d_projected, stream);
  solution.handle_ptr->sync_stream();

  i_t best_c          = 0;
  f_t best_l1         = std::numeric_limits<f_t>::infinity();
  i_t best_int_count  = -1;
  for (i_t c = 0; c < n_points; ++c) {
    f_t l1        = 0.;
    i_t int_count = 0;
    for (i_t k = 0; k < unified_n_int; ++k) {
      i_t col   = h_integer_indices_cache[k];
      f_t pv    = h_projected[(size_t)c * n_vars + col];
      f_t seed  = h_cloud[(size_t)c * n_vars + col];
      l1 += std::abs(pv - seed);
      if (std::abs(pv - std::round(pv)) <= int_tol) int_count++;
    }
    bool better = (l1 < best_l1) || (l1 == best_l1 && int_count > best_int_count);
    if (better) {
      best_l1        = l1;
      best_int_count = int_count;
      best_c         = c;
    }
  }
  CUOPT_LOG_DEBUG("Selected cloud point %d / %d (L1 %g, integer comps %d)",
                  best_c,
                  n_points,
                  best_l1,
                  best_int_count);
  raft::copy(solution.assignment.data(),
             d_projected.data() + (size_t)best_c * n_vars,
             n_vars,
             stream);
  solution.handle_ptr->sync_stream();
  return best_c;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_batched_fp_cloud(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("run_batched_fp_cloud");
  // Batch PDLP requires double precision; fall back to the single-point pump otherwise.
  if constexpr (!std::is_same_v<f_t, double>) {
    CUOPT_LOG_DEBUG("Batched FP cloud requires double precision; falling back to single FP");
    return run_single_fp_descent(solution);
  }

  if (unified_problem == nullptr || unified_n_vars != solution.problem_ptr->n_variables ||
      unified_n_constr != solution.problem_ptr->n_constraints) {
    build_unified_projection_problem(solution);
  }

  const i_t batch_size = compute_cloud_batch_size(solution);
  if (batch_size == 0) { return run_single_fp_descent(solution); }

  reseed_count = 0;
  solution.round_nearest();
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());

  rmm::device_uvector<f_t> d_cloud(0, solution.handle_ptr->get_stream());
  rmm::device_uvector<f_t> d_projected(0, solution.handle_ptr->get_stream());
  bool first_iteration = true;

  while (true) {
    if (context.diversity_manager_ptr->check_b_b_preemption() || timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Batched FP time limit reached!");
      round(solution);
      return false;
    }
    proj_begin = timer.remaining_time();

    bool seed_found_feasible = false;
    i_t n_points = assemble_cloud(solution, batch_size, first_iteration, d_cloud, seed_found_feasible);
    first_iteration = false;
    // The 20% FJ that seeds the cloud may already land on a feasible solution: take it and exit.
    if (seed_found_feasible) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      CUOPT_LOG_DEBUG("Seeding 20%% FJ found feasible; exiting batched FP");
      return res;
    }
    if (n_points == 0) {
      CUOPT_LOG_DEBUG("Empty cloud; falling back to single FP");
      return run_single_fp_descent(solution);
    }

    if (!project_cloud(solution, n_points, d_cloud, d_projected)) {
      bool is_feasible = round(solution);
      if (is_feasible && solution.compute_feasibility()) { return true; }
      return false;
    }

    select_cloud_point(solution, n_points, d_cloud, d_projected);
    raft::copy(last_projection.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());

    // Reseed the next cloud: quick nearest rounding of every projected point.
    {
      auto stream  = solution.handle_ptr->get_stream();
      auto h_proj  = cuopt::host_copy(d_projected, stream);
      solution.handle_ptr->sync_stream();
      const f_t int_tol = context.settings.tolerances.integrality_tolerance;
      std::vector<f_t> h_reseed((size_t)n_points * unified_n_vars);
      std::copy(h_proj.begin(), h_proj.end(), h_reseed.begin());
      for (i_t c = 0; c < n_points; ++c) {
        for (i_t k = 0; k < unified_n_int; ++k) {
          i_t col    = h_integer_indices_cache[k];
          f_t int_lb = std::ceil(h_var_lower[col] - int_tol);
          f_t int_ub = std::floor(h_var_upper[col] + int_tol);
          f_t r      = std::round(h_proj[(size_t)c * unified_n_vars + col]);
          if (r < int_lb) r = int_lb;
          if (r > int_ub) r = int_ub;
          h_reseed[(size_t)c * unified_n_vars + col] = r;
        }
      }
      reseed_points.resize(h_reseed.size(), stream);
      raft::copy(reseed_points.data(), h_reseed.data(), h_reseed.size(), stream);
      solution.handle_ptr->sync_stream();
      reseed_count = n_points;
    }

    i_t n_integers   = solution.compute_number_of_integers();
    bool is_feasible = solution.compute_feasibility();
    CUOPT_LOG_DEBUG("after batched fp projection n_integers %d total %d",
                    n_integers,
                    solution.problem_ptr->n_integer_vars);

    bool is_cycle = true;
    if (config.check_distance_cycle) {
      is_cycle = check_distance_cycle(solution);
      if (is_cycle) {
        is_feasible = round(solution);
        cuopt_func_call(solution.test_variable_bounds(true));
        if (is_feasible) {
          bool res = solution.compute_feasibility();
          cuopt_assert(res, "Feasibility issue");
          return true;
        }
        cuopt::default_logger().flush();
        total_fp_time_until_cycle = fp_fj_cycle_time_begin - timer.remaining_time();
        return false;
      }
    }

    if (n_integers == solution.problem_ptr->n_integer_vars) {
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Feasible solution found after batched projection");
        return true;
      }
      // The selected projection is fully integer but PDLP's loose batch tolerance can leave a
      // sub-MIP-tolerance violation. When it is essentially on the polytope, verify it with a
      // full-precision LP (integers fixed) to convert near-feasible points instead of cycling.
      else if (!last_distances.empty() && last_distances[0] < distance_to_check_for_feasible) {
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
          CUOPT_LOG_DEBUG("Feasible solution verified with LP after batched projection!");
          return true;
        }
      }
    }

    cuopt_func_call(solution.test_variable_bounds(false));
    is_feasible         = round(solution);
    cuopt_func_call(solution.test_variable_bounds(true));
    proj_and_round_time = proj_begin - timer.remaining_time();
    if (!is_feasible) {
      // Record this 20% FJ's visited iterates too, so its work is reused as extra cloud seeds for
      // the next iteration instead of being discarded.
      fj.settings.record_trajectory = true;
      fj.set_trajectory_capacity(batch_size, solution.handle_ptr->get_stream());
      is_feasible = test_fj_feasible(solution, batch_config.fj_seed_time_ratio * proj_and_round_time);
      fj.settings.record_trajectory = false;
      if (!is_feasible) {
        i_t traj = fj.get_trajectory_count();
        if (traj > 0) {
          auto stream = solution.handle_ptr->get_stream();
          auto h_traj = cuopt::host_copy(fj.get_trajectory_buffer(), stream);
          auto h_cur  = cuopt::host_copy(reseed_points, stream);
          solution.handle_ptr->sync_stream();
          h_cur.insert(h_cur.end(),
                       h_traj.begin(),
                       h_traj.begin() + (size_t)traj * unified_n_vars);
          reseed_points.resize(h_cur.size(), stream);
          raft::copy(reseed_points.data(), h_cur.data(), h_cur.size(), stream);
          solution.handle_ptr->sync_stream();
          reseed_count += traj;
        }
      }
    }
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Batched FP time limit reached!");
      return false;
    }
    if (is_feasible) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      return true;
    }

    f_t alpha_at_earlier_iter = config.alpha / config.alpha_decrease_factor;
    if (alpha_at_earlier_iter - config.alpha < 0.005) {
      is_cycle = cycle_queue.check_cycle(solution);
    }
    cycle_queue.update_recent_solutions(solution);
    if (is_cycle) {
      CUOPT_LOG_DEBUG("Batched FP cycle encountered");
      total_fp_time_until_cycle = fp_fj_cycle_time_begin - timer.remaining_time();
      return false;
    }
    cycle_queue.n_iterations_without_cycle++;
  }
  return false;
}

#if MIP_INSTANTIATE_FLOAT
template class feasibility_pump_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class feasibility_pump_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
