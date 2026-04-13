/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "batched_feasibility_pump.cuh"

#include <cuopt/error.hpp>
#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/diversity/population.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/host_helper.cuh>
#include <mip_heuristics/relaxed_lp/batched_relaxed_lp.cuh>
#include <mip_heuristics/utils.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>
#include <utilities/timer.hpp>

#include <raft/core/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
batched_feasibility_pump_t<i_t, f_t>::batched_feasibility_pump_t(
  mip_solver_context_t<i_t, f_t>& context_,
  fj_t<i_t, f_t>& fj_,
  constraint_prop_t<i_t, f_t>& constraint_prop_,
  line_segment_search_t<i_t, f_t>& line_segment_search_,
  rmm::device_uvector<f_t>& lp_optimal_solution_)
  : context(context_),
    fj(fj_),
    constraint_prop(constraint_prop_),
    line_segment_search(line_segment_search_),
    lp_optimal_solution(lp_optimal_solution_),
    path_roundings(0, context.handle_ptr->get_stream()),
    path_projections(0, context.handle_ptr->get_stream()),
    rng(cuopt::seed_generator::get_seed()),
    timer(20.)
{
}

template <typename i_t, typename f_t>
void batched_feasibility_pump_t<i_t, f_t>::reset()
{
  path_distances.clear();
  path_alphas.clear();
  augmented_problem_.reset();
  dist_var_id_.clear();
  n_orig_vars_    = 0;
  dist_cstr_base_ = 0;
}

template <typename i_t, typename f_t>
void batched_feasibility_pump_t<i_t, f_t>::resize_vectors(problem_t<i_t, f_t>& problem,
                                                          const raft::handle_t* handle_ptr)
{
  const i_t B      = config.batch_size;
  const size_t len = static_cast<size_t>(B) * problem.n_variables;
  auto stream      = handle_ptr->get_stream();
  path_roundings.resize(len, stream);
  path_projections.resize(len, stream);
}

// ──────────────────────────────────────────────────────────────
//  initialize_paths: create N random integer roundings from LP
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
void batched_feasibility_pump_t<i_t, f_t>::initialize_paths(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("batched_fp_initialize_paths");

  const i_t B      = config.batch_size;
  const i_t n_vars = solution.problem_ptr->n_variables;
  auto stream      = solution.handle_ptr->get_stream();

  path_distances.resize(B);
  path_alphas.resize(B);
  for (i_t k = 0; k < B; ++k) {
    path_distances[k].clear();
    path_alphas[k] = config.alpha;
  }

  thrust::fill(rmm::exec_policy(stream), path_projections.begin(), path_projections.end(), NAN);

  // Path 0: nearest rounding of LP relaxation (the standard FP starting point)
  solution.round_nearest();
  const f_t* src = solution.assignment.data();
  f_t* rnd_dst   = path_roundings.data();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<i_t>(0),
                     n_vars,
                     [src, rnd_dst, B] __device__(i_t j) { rnd_dst[0 + j * B] = src[j]; });

  // Paths 1..B-1: random rounding within bounds (integer vars get random integer)
  for (i_t k = 1; k < B; ++k) {
    solution.assign_random_within_bounds(1.0, true);
    solution.round_nearest();
    const f_t* rsrc = solution.assignment.data();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<i_t>(0),
                       n_vars,
                       [rsrc, rnd_dst, k, B] __device__(i_t j) { rnd_dst[k + j * B] = rsrc[j]; });
  }

  // Restore solution to the LP relaxation (path 0 source) for subsequent use
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<i_t>(0),
                     n_vars,
                     [rnd_dst, src = solution.assignment.data(), B] __device__(i_t j) {
                       src[j] = rnd_dst[0 + j * B];
                     });
}

// ──────────────────────────────────────────────────────────────
//  Objective blending helper (same logic as single-path FP)
// ──────────────────────────────────────────────────────────────
namespace {
template <typename f_t>
f_t vector_norm_host(const std::vector<f_t>& v)
{
  long double sum = 0;
  for (auto x : v)
    sum += static_cast<long double>(x) * x;
  return static_cast<f_t>(std::sqrt(sum));
}

template <typename i_t, typename f_t>
void blend_objective_with_original(std::vector<f_t>& dist_obj,
                                   const std::vector<f_t>& orig_obj,
                                   double alpha)
{
  f_t l2_orig = vector_norm_host(orig_obj);
  f_t l2_dist = vector_norm_host(dist_obj);
  f_t orig_w  = static_cast<f_t>(alpha) / l2_orig;
  f_t dist_w  = static_cast<f_t>(1.0 - alpha) / l2_dist;
  if (!std::isfinite(orig_w)) orig_w = f_t{0};
  cuopt_expects(std::isfinite(dist_w), error_type_t::RuntimeError, "Distance weight not finite");
  for (size_t i = 0; i < dist_obj.size(); ++i) {
    f_t ow      = (i < orig_obj.size()) ? orig_obj[i] : f_t{0};
    dist_obj[i] = dist_obj[i] * dist_w + orig_w * ow;
  }
}
}  // namespace

// ──────────────────────────────────────────────────────────────
//  build_augmented_problem
//  Build the augmented problem (original + distance vars/constraints) ONCE.
//  The A matrix structure is fixed for the entire FP run.
//  We conservatively add distance variables for ALL integers with lb != ub
//  so the structure doesn't depend on the current roundings.
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
void batched_feasibility_pump_t<i_t, f_t>::build_augmented_problem(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("batched_fp_build_augmented_problem");

  auto stream       = solution.handle_ptr->get_stream();
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;

  n_orig_vars_ = solution.problem_ptr->n_variables;

  auto h_variable_bounds = cuopt::host_copy(solution.problem_ptr->variable_bounds, stream);
  auto h_integer_indices = cuopt::host_copy(solution.problem_ptr->integer_indices, stream);
  solution.handle_ptr->sync_stream();

  constraints_delta_t<i_t, f_t> h_constraints;
  variables_delta_t<i_t, f_t> h_variables;
  h_variables.n_vars = n_orig_vars_;

  dist_var_id_.assign(h_integer_indices.size(), -1);

  for (size_t ii = 0; ii < h_integer_indices.size(); ++ii) {
    i_t i         = h_integer_indices[ii];
    auto h_bounds = h_variable_bounds[i];
    f_t lb        = get_lower(h_bounds);
    f_t ub        = get_upper(h_bounds);

    if (std::abs(ub - lb) < int_tol) continue;

    const f_t obj_weight = 1.;
    i_t var_id = h_variables.add_variable(0, (ub - lb) + int_tol, obj_weight, var_t::CONTINUOUS);
    dist_var_id_[ii] = var_id;

    std::vector<i_t> constr_indices{var_id, i};
    // d_j - x_j >= -ub  (relaxed; tightened per-path via RHS each iteration)
    std::vector<f_t> constr_coeffs_1{1, -1};
    h_constraints.add_constraint(
      constr_indices, constr_coeffs_1, f_t{-ub}, (f_t)default_cont_upper);
    // d_j + x_j >= lb
    std::vector<f_t> constr_coeffs_2{1, 1};
    h_constraints.add_constraint(constr_indices, constr_coeffs_2, f_t{lb}, (f_t)default_cont_upper);
  }

  augmented_problem_ = std::make_unique<problem_t<i_t, f_t>>(*solution.problem_ptr);
  if (h_variables.size() > 0) { augmented_problem_->insert_variables(h_variables); }
  if (h_constraints.n_constraints() > 0) { augmented_problem_->insert_constraints(h_constraints); }
  if (h_constraints.n_constraints() > 0 || h_variables.size() > 0) {
    augmented_problem_->compute_transpose_of_problem();
  }

  dist_cstr_base_ = solution.problem_ptr->n_constraints;

  CUOPT_LOG_DEBUG("Batched FP: augmented problem built once: %d vars (%d orig), %d constraints",
                  augmented_problem_->n_variables,
                  n_orig_vars_,
                  augmented_problem_->n_constraints);
}

// ──────────────────────────────────────────────────────────────
//  update_batched_input
//  Fill per-path objectives, bounds, RHS, and initial primals
//  into a batched_lp_input_t.  Does NOT touch the A matrix.
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
batched_lp_input_t<i_t, f_t> batched_feasibility_pump_t<i_t, f_t>::update_batched_input(
  solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("batched_fp_update_batched_input");

  const i_t B          = config.batch_size;
  const i_t n_vars     = n_orig_vars_;
  auto stream          = solution.handle_ptr->get_stream();
  const f_t int_tol    = context.settings.tolerances.integrality_tolerance;
  auto& aug_p          = *augmented_problem_;
  const i_t n_aug_vars = aug_p.n_variables;
  const i_t n_aug_cstr = aug_p.n_constraints;

  auto h_variable_bounds = cuopt::host_copy(solution.problem_ptr->variable_bounds, stream);
  auto h_integer_indices = cuopt::host_copy(solution.problem_ptr->integer_indices, stream);
  auto h_orig_obj        = cuopt::host_copy(solution.problem_ptr->objective_coefficients, stream);
  auto h_roundings       = cuopt::host_copy(path_roundings, stream);
  auto h_projections     = cuopt::host_copy(path_projections, stream);

  auto h_cstr_lb        = cuopt::host_copy(aug_p.constraint_lower_bounds, stream);
  auto h_cstr_ub        = cuopt::host_copy(aug_p.constraint_upper_bounds, stream);
  auto h_aug_var_bounds = cuopt::host_copy(aug_p.variable_bounds, stream);
  solution.handle_ptr->sync_stream();

  batched_lp_input_t<i_t, f_t> input(B, aug_p, stream);

  std::vector<f_t> h_obj_batch(static_cast<size_t>(B) * n_aug_vars, f_t{0});
  std::vector<f_t> h_vlb_batch(static_cast<size_t>(B) * n_aug_vars);
  std::vector<f_t> h_vub_batch(static_cast<size_t>(B) * n_aug_vars);
  std::vector<f_t> h_clb_batch(static_cast<size_t>(B) * n_aug_cstr);
  std::vector<f_t> h_cub_batch(static_cast<size_t>(B) * n_aug_cstr);
  std::vector<f_t> h_init_primal(static_cast<size_t>(B) * n_aug_vars, f_t{0});

  for (i_t k = 0; k < B; ++k) {
    for (i_t j = 0; j < n_aug_vars; ++j) {
      h_vlb_batch[k + static_cast<size_t>(j) * B] = get_lower(h_aug_var_bounds[j]);
      h_vub_batch[k + static_cast<size_t>(j) * B] = get_upper(h_aug_var_bounds[j]);
    }
  }

  for (i_t k = 0; k < B; ++k) {
    for (i_t c = 0; c < n_aug_cstr; ++c) {
      h_clb_batch[k + static_cast<size_t>(c) * B] = h_cstr_lb[c];
      h_cub_batch[k + static_cast<size_t>(c) * B] = h_cstr_ub[c];
    }
  }

  for (i_t k = 0; k < B; ++k) {
    std::vector<f_t> obj_coefficients(n_aug_vars, f_t{0});
    f_t obj_offset    = f_t{0};
    i_t dist_cstr_idx = 0;

    for (size_t ii = 0; ii < h_integer_indices.size(); ++ii) {
      i_t i         = h_integer_indices[ii];
      auto h_bounds = h_variable_bounds[i];
      f_t lb        = get_lower(h_bounds);
      f_t ub        = get_upper(h_bounds);
      if (std::abs(ub - lb) < int_tol) continue;

      f_t val = h_roundings[k + static_cast<size_t>(i) * B];

      if (solution.problem_ptr->integer_equal(val, ub)) {
        obj_offset += ub;
        obj_coefficients[i] = f_t{-1};
      } else if (solution.problem_ptr->integer_equal(val, lb)) {
        obj_offset -= lb;
        obj_coefficients[i] = f_t{1};
      } else {
        i_t var_id = dist_var_id_[ii];
        cuopt_assert(var_id >= 0, "Expected distance variable for interior integer");
        obj_coefficients[var_id] = f_t{1};

        f_t proj_val = h_projections[k + static_cast<size_t>(i) * B];
        f_t dist_val = std::abs(val - proj_val);
        if (!std::isfinite(dist_val)) dist_val = f_t{0};
        h_init_primal[k + static_cast<size_t>(var_id) * B] = dist_val;
      }

      if (dist_var_id_[ii] >= 0) {
        i_t c1                                       = dist_cstr_base_ + dist_cstr_idx * 2;
        i_t c2                                       = dist_cstr_base_ + dist_cstr_idx * 2 + 1;
        h_clb_batch[k + static_cast<size_t>(c1) * B] = -val;
        h_clb_batch[k + static_cast<size_t>(c2) * B] = val;
        dist_cstr_idx++;
      }
    }

    for (i_t j = 0; j < n_vars; ++j) {
      f_t proj                                      = h_projections[k + static_cast<size_t>(j) * B];
      f_t rnd                                       = h_roundings[k + static_cast<size_t>(j) * B];
      h_init_primal[k + static_cast<size_t>(j) * B] = std::isfinite(proj) ? proj : rnd;
    }

    path_alphas[k] *= config.alpha_decrease_factor;
    blend_objective_with_original<i_t, f_t>(obj_coefficients, h_orig_obj, path_alphas[k]);

    input.objective_offsets[k] = obj_offset;

    for (i_t j = 0; j < n_aug_vars; ++j) {
      h_obj_batch[k + static_cast<size_t>(j) * B] = obj_coefficients[j];
    }
  }

  raft::copy(input.objective_coefficients.data(), h_obj_batch.data(), h_obj_batch.size(), stream);
  raft::copy(input.variable_lower_bounds.data(), h_vlb_batch.data(), h_vlb_batch.size(), stream);
  raft::copy(input.variable_upper_bounds.data(), h_vub_batch.data(), h_vub_batch.size(), stream);
  raft::copy(input.constraint_lower_bounds.data(), h_clb_batch.data(), h_clb_batch.size(), stream);
  raft::copy(input.constraint_upper_bounds.data(), h_cub_batch.data(), h_cub_batch.size(), stream);
  raft::copy(
    input.initial_primal_solutions.data(), h_init_primal.data(), h_init_primal.size(), stream);
  input.has_initial_primal = true;

  return input;
}

// ──────────────────────────────────────────────────────────────
//  extract_projections
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
void batched_feasibility_pump_t<i_t, f_t>::extract_projections(
  const batched_lp_output_t<i_t, f_t>& output, i_t n_orig_vars)
{
  raft::common::nvtx::range fun_scope("batched_fp_extract_projections");

  const i_t B          = config.batch_size;
  auto stream          = context.handle_ptr->get_stream();
  const i_t n_aug_vars = output.primal_solutions.size() / B;

  // Copy only original variables from output (skip distance vars)
  const f_t* out_src = output.primal_solutions.data();
  f_t* proj_dst      = path_projections.data();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<i_t>(0),
                     static_cast<i_t>(B) * n_orig_vars,
                     [out_src, proj_dst, B, n_aug_vars, n_orig_vars] __device__(i_t flat) {
                       i_t k               = flat % B;
                       i_t j               = flat / B;
                       proj_dst[k + j * B] = out_src[k + j * B];
                     });
}

// ──────────────────────────────────────────────────────────────
//  round_all_paths: nearest rounding of all paths
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
void batched_feasibility_pump_t<i_t, f_t>::round_all_paths(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("batched_fp_round_all_paths");

  const i_t B      = config.batch_size;
  const i_t n_vars = solution.problem_ptr->n_variables;
  auto stream      = solution.handle_ptr->get_stream();

  f_t* proj_ptr     = path_projections.data();
  f_t* rnd_ptr      = path_roundings.data();
  auto int_indices  = make_span(solution.problem_ptr->integer_indices);
  auto var_bounds   = make_span(solution.problem_ptr->variable_bounds);
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;

  // For each path and each variable, copy projection to rounding.
  // For integer variables, round to nearest integer within bounds.
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<i_t>(0),
                     B * n_vars,
                     [proj_ptr, rnd_ptr, var_bounds, B, n_vars] __device__(i_t flat) {
                       i_t k              = flat % B;
                       i_t j              = flat / B;
                       f_t val            = proj_ptr[k + j * B];
                       rnd_ptr[k + j * B] = val;
                     });

  // Round integer variables
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<i_t>(0),
                     B * solution.problem_ptr->n_integer_vars,
                     [rnd_ptr, int_indices, var_bounds, B, int_tol] __device__(i_t flat) {
                       i_t k              = flat % B;
                       i_t ii             = flat / B;
                       i_t j              = int_indices[ii];
                       f_t val            = rnd_ptr[k + j * B];
                       f_t lb             = get_lower(var_bounds[j]);
                       f_t ub             = get_upper(var_bounds[j]);
                       f_t rounded        = rint(val);
                       rounded            = max(lb, min(ub, rounded));
                       rnd_ptr[k + j * B] = rounded;
                     });
}

// ──────────────────────────────────────────────────────────────
//  check_path_cycles: distance-based cycle detection per path
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
std::vector<bool> batched_feasibility_pump_t<i_t, f_t>::check_path_cycles(
  solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("batched_fp_check_cycles");

  const i_t B      = config.batch_size;
  const i_t n_vars = solution.problem_ptr->n_variables;
  auto stream      = solution.handle_ptr->get_stream();

  std::vector<bool> cycling(B, false);

  // Compute L1 distance between projection and rounding for each path
  // We do this on device per-path using a transform_reduce
  for (i_t k = 0; k < B; ++k) {
    const f_t* proj_ptr = path_projections.data();
    const f_t* rnd_ptr  = path_roundings.data();

    f_t distance = thrust::transform_reduce(
      rmm::exec_policy(stream),
      solution.problem_ptr->integer_indices.begin(),
      solution.problem_ptr->integer_indices.end(),
      [proj_ptr, rnd_ptr, k, B] __device__(i_t j) -> f_t {
        return abs(proj_ptr[k + j * B] - rnd_ptr[k + j * B]);
      },
      f_t{0},
      thrust::plus<f_t>());

    auto& hist = path_distances[k];
    if (static_cast<i_t>(hist.size()) >= config.cycle_detection_window) {
      f_t avg = std::accumulate(hist.begin(), hist.end(), f_t{0}) / hist.size();
      if (avg - distance < config.cycle_distance_reduction_ratio * avg) {
        CUOPT_LOG_DEBUG("Batched FP: path %d distance cycle (curr %f avg %f)", k, distance, avg);
        cycling[k] = true;
      }
      hist.pop_back();
    }
    hist.push_front(distance);
  }

  return cycling;
}

// ──────────────────────────────────────────────────────────────
//  replace_cycling_paths: perturb cycling paths
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
void batched_feasibility_pump_t<i_t, f_t>::replace_cycling_paths(
  solution_t<i_t, f_t>& solution, const std::vector<bool>& cycling_mask)
{
  raft::common::nvtx::range fun_scope("batched_fp_replace_cycling");

  const i_t B      = config.batch_size;
  const i_t n_vars = solution.problem_ptr->n_variables;
  auto stream      = solution.handle_ptr->get_stream();

  for (i_t k = 0; k < B; ++k) {
    if (!cycling_mask[k]) continue;

    CUOPT_LOG_DEBUG("Batched FP: replacing cycling path %d", k);

    // Reset distance history and alpha for this path
    path_distances[k].clear();
    path_alphas[k] = config.alpha;

    // Generate a new random rounding
    solution.assign_random_within_bounds(config.perturbation_ratio, true);
    solution.round_nearest();

    const f_t* src = solution.assignment.data();
    f_t* rnd_dst   = path_roundings.data();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<i_t>(0),
                       n_vars,
                       [src, rnd_dst, k, B] __device__(i_t j) { rnd_dst[k + j * B] = src[j]; });
  }
}

// ──────────────────────────────────────────────────────────────
//  check_and_submit_feasible
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
i_t batched_feasibility_pump_t<i_t, f_t>::check_and_submit_feasible(
  solution_t<i_t, f_t>& solution, population_t<i_t, f_t>* population_ptr)
{
  raft::common::nvtx::range fun_scope("batched_fp_check_feasible");

  const i_t B      = config.batch_size;
  const i_t n_vars = solution.problem_ptr->n_variables;
  auto stream      = solution.handle_ptr->get_stream();

  i_t best_feasible = -1;
  i_t best_n_ints   = -1;

  for (i_t k = 0; k < B; ++k) {
    // Load this path's rounding into the solution
    const f_t* rnd_src = path_roundings.data();
    f_t* dst           = solution.assignment.data();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<i_t>(0),
                       n_vars,
                       [rnd_src, dst, k, B] __device__(i_t j) { dst[j] = rnd_src[k + j * B]; });

    bool is_feasible = solution.compute_feasibility();
    i_t n_ints       = solution.compute_number_of_integers();

    if (is_feasible) {
      CUOPT_LOG_DEBUG("Batched FP: path %d feasible! obj=%f", k, solution.get_user_objective());
      if (population_ptr != nullptr) {
        solution_t<i_t, f_t> feasible_copy(solution);
        population_ptr->add_solution(std::move(feasible_copy));
      }
      if (best_feasible < 0 || n_ints > best_n_ints) {
        best_feasible = k;
        best_n_ints   = n_ints;
      }
    }
  }

  return best_feasible;
}

// ──────────────────────────────────────────────────────────────
//  try_fj_repair: short FJ run on a specific path
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
bool batched_feasibility_pump_t<i_t, f_t>::try_fj_repair(solution_t<i_t, f_t>& solution,
                                                         i_t path_idx)
{
  raft::common::nvtx::range fun_scope("batched_fp_fj_repair");

  const i_t B      = config.batch_size;
  const i_t n_vars = solution.problem_ptr->n_variables;
  auto stream      = solution.handle_ptr->get_stream();

  // Load the path's rounding into solution
  const f_t* rnd_src = path_roundings.data();
  f_t* dst           = solution.assignment.data();
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<i_t>(0),
    n_vars,
    [rnd_src, dst, path_idx, B] __device__(i_t j) { dst[j] = rnd_src[path_idx + j * B]; });

  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.n_of_minimums_for_exit = 5000;
  fj.settings.time_limit             = std::min(1.0, static_cast<double>(timer.remaining_time()));

  return fj.solve(solution);
}

// ──────────────────────────────────────────────────────────────
//  run: main batched FP loop
// ──────────────────────────────────────────────────────────────
template <typename i_t, typename f_t>
bool batched_feasibility_pump_t<i_t, f_t>::run(solution_t<i_t, f_t>& solution,
                                               timer_t run_timer,
                                               population_t<i_t, f_t>* population_ptr)
{
  raft::common::nvtx::range fun_scope("batched_feasibility_pump");
  timer = run_timer;

  const i_t B      = config.batch_size;
  const i_t n_vars = solution.problem_ptr->n_variables;
  auto stream      = solution.handle_ptr->get_stream();

  CUOPT_LOG_DEBUG("Batched FP starting with %d paths, %d vars, %d int vars",
                  B,
                  n_vars,
                  solution.problem_ptr->n_integer_vars);

  resize_vectors(*solution.problem_ptr, solution.handle_ptr);
  reset();
  initialize_paths(solution);

  // Build the augmented problem (original + distance vars) ONCE.
  // The A matrix is reused across all iterations; only per-path
  // objectives, bounds, and RHS are updated each iteration.
  build_augmented_problem(solution);

  bool found_feasible = false;
  const f_t rlp_base  = context.settings.heuristic_params.relaxed_lp_time_limit;

  for (i_t iter = 0; iter < config.max_iterations; ++iter) {
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Batched FP: time limit at iteration %d", iter);
      break;
    }
    if (context.diversity_manager_ptr != nullptr &&
        context.diversity_manager_ptr->check_b_b_preemption()) {
      CUOPT_LOG_DEBUG("Batched FP: preempted at iteration %d", iter);
      break;
    }
    if (context.preempt_heuristic_solver_.load()) {
      CUOPT_LOG_DEBUG("Batched FP: heuristic preempted at iteration %d", iter);
      break;
    }

    // 1. Update per-path data (A matrix is reused from augmented_problem_)
    auto batched_input               = update_batched_input(solution);
    batched_input.shared_problem_ptr = augmented_problem_.get();

    // 2. Solve batched LP
    f_t lp_time =
      std::max(0.05, std::min(static_cast<double>(rlp_base), timer.remaining_time() / 10.));
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit          = lp_time;
    lp_settings.tolerance           = 0.01;
    lp_settings.check_infeasibility = false;
    lp_settings.save_state          = false;

    auto batched_output = solve_batched_lp(batched_input, lp_settings, timer);

    // 3. Extract projections (original vars only, skip distance vars)
    extract_projections(batched_output, n_vars);

    // 4. Round all paths
    round_all_paths(solution);

    // 5. Check feasibility and submit to population
    if (population_ptr != nullptr) { population_ptr->add_external_solutions_to_population(); }
    i_t feasible_path = check_and_submit_feasible(solution, population_ptr);
    if (feasible_path >= 0) {
      CUOPT_LOG_DEBUG("Batched FP: feasible path %d at iteration %d", feasible_path, iter);

      // Load best feasible path into solution
      const f_t* rnd_src = path_roundings.data();
      f_t* dst           = solution.assignment.data();
      thrust::for_each_n(rmm::exec_policy(stream),
                         thrust::make_counting_iterator<i_t>(0),
                         n_vars,
                         [rnd_src, dst, feasible_path, B] __device__(i_t j) {
                           dst[j] = rnd_src[feasible_path + j * B];
                         });
      solution.compute_feasibility();
      found_feasible = true;
      break;
    }

    // 6. Try FJ repair on the path with most integers
    if (!found_feasible && timer.remaining_time() > 1.0) {
      // Find path with most integers set
      i_t best_path   = 0;
      i_t best_n_ints = 0;
      for (i_t k = 0; k < B; ++k) {
        const f_t* rnd_src = path_roundings.data();
        f_t* dst           = solution.assignment.data();
        thrust::for_each_n(rmm::exec_policy(stream),
                           thrust::make_counting_iterator<i_t>(0),
                           n_vars,
                           [rnd_src, dst, k, B] __device__(i_t j) { dst[j] = rnd_src[k + j * B]; });
        i_t n_ints = solution.compute_number_of_integers();
        if (n_ints > best_n_ints) {
          best_n_ints = n_ints;
          best_path   = k;
        }
      }

      bool fj_feasible = try_fj_repair(solution, best_path);
      if (fj_feasible) {
        CUOPT_LOG_DEBUG(
          "Batched FP: FJ repair found feasible on path %d at iteration %d", best_path, iter);
        solution.compute_feasibility();
        if (population_ptr != nullptr) {
          solution_t<i_t, f_t> feasible_copy(solution);
          population_ptr->add_solution(std::move(feasible_copy));
        }
        found_feasible = true;
        break;
      } else {
        // Restore from last rounding (FJ may have modified it)
        const f_t* rnd_src = path_roundings.data();
        f_t* dst           = solution.assignment.data();
        thrust::for_each_n(
          rmm::exec_policy(stream),
          thrust::make_counting_iterator<i_t>(0),
          n_vars,
          [rnd_src, dst, best_path, B] __device__(i_t j) { dst[j] = rnd_src[best_path + j * B]; });
      }
    }

    // 7. Cycle detection and replacement
    auto cycling = check_path_cycles(solution);
    replace_cycling_paths(solution, cycling);

    CUOPT_LOG_DEBUG(
      "Batched FP iteration %d done, remaining time %f", iter, timer.remaining_time());
  }

  if (!found_feasible) {
    // Return the best path's rounding as the solution even if infeasible
    // (closest to feasible by most integers)
    i_t best_path   = 0;
    i_t best_n_ints = 0;
    for (i_t k = 0; k < B; ++k) {
      const f_t* rnd_src = path_roundings.data();
      f_t* dst           = solution.assignment.data();
      thrust::for_each_n(rmm::exec_policy(stream),
                         thrust::make_counting_iterator<i_t>(0),
                         n_vars,
                         [rnd_src, dst, k, B] __device__(i_t j) { dst[j] = rnd_src[k + j * B]; });
      i_t n_ints = solution.compute_number_of_integers();
      if (n_ints > best_n_ints) {
        best_n_ints = n_ints;
        best_path   = k;
      }
    }
    // Load best into solution
    const f_t* rnd_src = path_roundings.data();
    f_t* dst           = solution.assignment.data();
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<i_t>(0),
      n_vars,
      [rnd_src, dst, best_path, B] __device__(i_t j) { dst[j] = rnd_src[best_path + j * B]; });
    solution.compute_feasibility();
  }

  CUOPT_LOG_DEBUG("Batched FP finished: feasible=%d", found_feasible);
  return found_feasible;
}

#if MIP_INSTANTIATE_FLOAT
template class batched_feasibility_pump_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class batched_feasibility_pump_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
