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
    d_dist_var_id_(0, context.handle_ptr->get_stream()),
    d_orig_obj_padded_(0, context.handle_ptr->get_stream()),
    cached_vlb_batch_(0, context.handle_ptr->get_stream()),
    cached_vub_batch_(0, context.handle_ptr->get_stream()),
    cached_cub_batch_(0, context.handle_ptr->get_stream()),
    cached_clb_base_(0, context.handle_ptr->get_stream()),
    d_obj_offsets_(0, context.handle_ptr->get_stream()),
    rng(cuopt::seed_generator::get_seed()),
    timer(20.)
{
}

template <typename i_t, typename f_t>
void batched_feasibility_pump_t<i_t, f_t>::reset()
{
  path_distances.clear();
  path_alphas.clear();
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

// (Objective blending is now done on the GPU inside update_batched_input.)

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

  // --- Cache invariant device data for update_batched_input ---
  auto& aug_p          = *augmented_problem_;
  const i_t n_aug_vars = aug_p.n_variables;
  const i_t n_aug_cstr = aug_p.n_constraints;
  const i_t B          = config.batch_size;

  // dist_var_id on device
  d_dist_var_id_.resize(dist_var_id_.size(), stream);
  raft::copy(d_dist_var_id_.data(), dist_var_id_.data(), dist_var_id_.size(), stream);

  // Original objective padded to n_aug_vars (zeros for distance variables)
  auto h_orig_obj = cuopt::host_copy(solution.problem_ptr->objective_coefficients, stream);
  solution.handle_ptr->sync_stream();
  std::vector<f_t> h_orig_obj_padded(n_aug_vars, f_t{0});
  for (i_t j = 0; j < n_orig_vars_; ++j)
    h_orig_obj_padded[j] = h_orig_obj[j];
  d_orig_obj_padded_.resize(n_aug_vars, stream);
  raft::copy(d_orig_obj_padded_.data(), h_orig_obj_padded.data(), n_aug_vars, stream);

  // Precompute L2 norm of original objective
  long double sum_sq = 0;
  for (auto x : h_orig_obj)
    sum_sq += static_cast<long double>(x) * x;
  l2_orig_obj_ = static_cast<f_t>(std::sqrt(sum_sq));

  // Build invariant batched variable bounds
  auto h_aug_var_bounds = cuopt::host_copy(aug_p.variable_bounds, stream);
  solution.handle_ptr->sync_stream();
  const size_t vb_len = static_cast<size_t>(B) * n_aug_vars;
  std::vector<f_t> h_vlb(vb_len), h_vub(vb_len);
  for (i_t k = 0; k < B; ++k) {
    for (i_t j = 0; j < n_aug_vars; ++j) {
      h_vlb[k + static_cast<size_t>(j) * B] = get_lower(h_aug_var_bounds[j]);
      h_vub[k + static_cast<size_t>(j) * B] = get_upper(h_aug_var_bounds[j]);
    }
  }
  cached_vlb_batch_.resize(vb_len, stream);
  cached_vub_batch_.resize(vb_len, stream);
  raft::copy(cached_vlb_batch_.data(), h_vlb.data(), vb_len, stream);
  raft::copy(cached_vub_batch_.data(), h_vub.data(), vb_len, stream);

  // Build invariant batched constraint upper bounds + base constraint lower bounds
  auto h_cstr_ub = cuopt::host_copy(aug_p.constraint_upper_bounds, stream);
  auto h_cstr_lb = cuopt::host_copy(aug_p.constraint_lower_bounds, stream);
  solution.handle_ptr->sync_stream();
  const size_t cb_len = static_cast<size_t>(B) * n_aug_cstr;
  std::vector<f_t> h_cub(cb_len);
  for (i_t k = 0; k < B; ++k) {
    for (i_t c = 0; c < n_aug_cstr; ++c)
      h_cub[k + static_cast<size_t>(c) * B] = h_cstr_ub[c];
  }
  cached_cub_batch_.resize(cb_len, stream);
  raft::copy(cached_cub_batch_.data(), h_cub.data(), cb_len, stream);

  cached_clb_base_.resize(n_aug_cstr, stream);
  raft::copy(cached_clb_base_.data(), h_cstr_lb.data(), n_aug_cstr, stream);

  d_obj_offsets_.resize(B, stream);

  CUOPT_LOG_DEBUG("Batched FP: augmented problem built once: %d vars (%d orig), %d constraints",
                  n_aug_vars,
                  n_orig_vars_,
                  n_aug_cstr);
}

// ──────────────────────────────────────────────────────────────
//  update_batched_input (GPU-based)
//  Invariant data (variable bounds, constraint upper bounds) is D->D
//  copied from cached buffers built once in build_augmented_problem.
//  Per-path objectives, constraint RHS, and initial primals are
//  computed entirely on the GPU — no D->H->D round-trips.
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
  const i_t n_int_vars = solution.problem_ptr->n_integer_vars;

  batched_lp_input_t<i_t, f_t> input(B, aug_p, stream);

  // ── Invariant data: D->D copy from cached buffers ──
  raft::copy(
    input.variable_lower_bounds.data(), cached_vlb_batch_.data(), cached_vlb_batch_.size(), stream);
  raft::copy(
    input.variable_upper_bounds.data(), cached_vub_batch_.data(), cached_vub_batch_.size(), stream);
  raft::copy(input.constraint_upper_bounds.data(),
             cached_cub_batch_.data(),
             cached_cub_batch_.size(),
             stream);

  // ── Broadcast base constraint lower bounds to all paths ──
  {
    const f_t* clb_base = cached_clb_base_.data();
    f_t* clb_dst        = input.constraint_lower_bounds.data();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<i_t>(0),
                       static_cast<i_t>(B) * n_aug_cstr,
                       [clb_base, clb_dst, B] __device__(i_t flat) {
                         i_t c         = flat / B;
                         clb_dst[flat] = clb_base[c];
                       });
  }

  // ── Zero-fill per-path objectives and initial primals ──
  thrust::fill(rmm::exec_policy(stream),
               input.objective_coefficients.begin(),
               input.objective_coefficients.end(),
               f_t{0});
  thrust::fill(rmm::exec_policy(stream),
               input.initial_primal_solutions.begin(),
               input.initial_primal_solutions.end(),
               f_t{0});

  // ── Set initial primals for original variables: prefer projection, fall back to rounding ──
  {
    const f_t* proj_ptr = path_projections.data();
    const f_t* rnd_ptr  = path_roundings.data();
    f_t* primal_ptr     = input.initial_primal_solutions.data();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<i_t>(0),
                       static_cast<i_t>(B) * n_vars,
                       [proj_ptr, rnd_ptr, primal_ptr, B] __device__(i_t flat) {
                         i_t k                 = flat % B;
                         i_t j                 = flat / B;
                         f_t proj              = proj_ptr[k + j * B];
                         f_t rnd               = rnd_ptr[k + j * B];
                         primal_ptr[k + j * B] = isfinite(proj) ? proj : rnd;
                       });
  }

  // ── Process integer variables on GPU:
  //    set objectives, distance constraint RHS, distance variable primals,
  //    and store per-element offset contributions for deterministic reduction. ──
  {
    const f_t* rnd_ptr  = path_roundings.data();
    const f_t* proj_ptr = path_projections.data();
    f_t* obj_ptr        = input.objective_coefficients.data();
    f_t* clb_ptr        = input.constraint_lower_bounds.data();
    f_t* primal_ptr     = input.initial_primal_solutions.data();
    const i_t* dv_ptr   = d_dist_var_id_.data();
    auto int_idx        = make_span(solution.problem_ptr->integer_indices);
    auto vbounds        = make_span(solution.problem_ptr->variable_bounds);
    const i_t dcb       = dist_cstr_base_;
    const i_t n_orig    = n_orig_vars_;

    const size_t contrib_len = static_cast<size_t>(B) * n_int_vars;
    rmm::device_uvector<f_t> d_offset_contribs(contrib_len, stream);
    f_t* contrib_ptr = d_offset_contribs.data();

    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<i_t>(0),
                       static_cast<i_t>(B) * n_int_vars,
                       [rnd_ptr,
                        proj_ptr,
                        obj_ptr,
                        clb_ptr,
                        primal_ptr,
                        contrib_ptr,
                        dv_ptr,
                        int_idx,
                        vbounds,
                        int_tol,
                        B,
                        n_orig,
                        dcb] __device__(i_t flat) {
                         i_t k     = flat % B;
                         i_t ii    = flat / B;
                         i_t i     = int_idx[ii];
                         auto bnds = vbounds[i];
                         f_t lb    = get_lower(bnds);
                         f_t ub    = get_upper(bnds);
                         if (abs(ub - lb) < int_tol) {
                           contrib_ptr[flat] = f_t{0};
                           return;
                         }

                         f_t val  = rnd_ptr[k + i * B];
                         i_t dvid = dv_ptr[ii];

                         f_t offset_contrib = f_t{0};
                         if (integer_equal<f_t>(val, ub, int_tol)) {
                           offset_contrib     = ub;
                           obj_ptr[k + i * B] = f_t{-1};
                         } else if (integer_equal<f_t>(val, lb, int_tol)) {
                           offset_contrib     = -lb;
                           obj_ptr[k + i * B] = f_t{1};
                         } else {
                           cuopt_assert(dvid >= 0,
                                        "Expected distance variable for interior integer");
                           obj_ptr[k + dvid * B] = f_t{1};
                           f_t proj_val          = proj_ptr[k + i * B];
                           f_t dist_val          = abs(val - proj_val);
                           if (!isfinite(dist_val)) dist_val = f_t{0};
                           primal_ptr[k + dvid * B] = dist_val;
                         }
                         contrib_ptr[flat] = offset_contrib;

                         if (dvid >= 0) {
                           i_t rank            = dvid - n_orig;
                           i_t c1              = dcb + rank * 2;
                           i_t c2              = dcb + rank * 2 + 1;
                           clb_ptr[k + c1 * B] = -val;
                           clb_ptr[k + c2 * B] = val;
                         }
                       });

    // Deterministic per-path reduction: each thread sums its path in fixed order
    f_t* offsets_ptr    = d_obj_offsets_.data();
    const f_t* cptr     = d_offset_contribs.data();
    const i_t n_int_loc = n_int_vars;
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<i_t>(0),
                       B,
                       [cptr, offsets_ptr, n_int_loc, B] __device__(i_t k) {
                         f_t sum = f_t{0};
                         for (i_t ii = 0; ii < n_int_loc; ++ii) {
                           sum += cptr[k + static_cast<size_t>(ii) * B];
                         }
                         offsets_ptr[k] = sum;
                       });
  }

  // ── Objective blending: decay alpha, compute per-path L2 norms, blend on GPU ──
  {
    f_t* obj_ptr = input.objective_coefficients.data();
    std::vector<f_t> h_orig_w(B), h_dist_w(B);
    for (i_t k = 0; k < B; ++k) {
      path_alphas[k] *= config.alpha_decrease_factor;
      f_t l2_dist_sq = thrust::transform_reduce(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator<i_t>(0),
        thrust::make_counting_iterator<i_t>(n_aug_vars),
        [obj_ptr, k, B] __device__(i_t j) -> f_t {
          f_t v = obj_ptr[k + j * B];
          return v * v;
        },
        f_t{0},
        thrust::plus<f_t>());
      f_t l2_dist = std::sqrt(l2_dist_sq);
      h_orig_w[k] = static_cast<f_t>(path_alphas[k]) / l2_orig_obj_;
      h_dist_w[k] = static_cast<f_t>(1.0 - path_alphas[k]) / l2_dist;
      if (!std::isfinite(h_orig_w[k])) h_orig_w[k] = f_t{0};
      cuopt_expects(
        std::isfinite(h_dist_w[k]), error_type_t::RuntimeError, "Distance weight not finite");
    }

    rmm::device_uvector<f_t> d_orig_w(B, stream);
    rmm::device_uvector<f_t> d_dist_w(B, stream);
    raft::copy(d_orig_w.data(), h_orig_w.data(), B, stream);
    raft::copy(d_dist_w.data(), h_dist_w.data(), B, stream);

    const f_t* orig_obj_ptr = d_orig_obj_padded_.data();
    const f_t* ow_ptr       = d_orig_w.data();
    const f_t* dw_ptr       = d_dist_w.data();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<i_t>(0),
                       static_cast<i_t>(B) * n_aug_vars,
                       [obj_ptr, orig_obj_ptr, ow_ptr, dw_ptr, B] __device__(i_t flat) {
                         i_t k         = flat % B;
                         i_t j         = flat / B;
                         obj_ptr[flat] = obj_ptr[flat] * dw_ptr[k] + ow_ptr[k] * orig_obj_ptr[j];
                       });
  }

  // ── Copy per-path objective offsets back to host ──
  std::vector<f_t> h_offsets(B);
  raft::copy(h_offsets.data(), d_obj_offsets_.data(), B, stream);
  solution.handle_ptr->sync_stream();
  for (i_t k = 0; k < B; ++k)
    input.objective_offsets[k] = h_offsets[k];

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

  // Build the augmented problem once; reuse across iterations AND across runs.
  // The base problem (A, integer indices, variable types) is stable.
  if (!augmented_problem_) { build_augmented_problem(solution); }

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
