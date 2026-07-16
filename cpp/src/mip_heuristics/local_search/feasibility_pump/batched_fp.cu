/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "feasibility_pump.cuh"

#include <cuopt/error.hpp>
#include <mip_heuristics/diversity/assignment_hash_map.cuh>
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/linalg/binary_op.cuh>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

namespace cuopt::mathematical_optimization::mip {

// Defined in feasibility_pump.cu: maps the fraction of integral integer-vars to an LP tolerance
// (looser when far from integral, tighter as it converges).
double get_tolerance_from_ratio(double ratio_integer, double absolute_tol);

class cuda_profiler_scope_t {
 public:
  explicit cuda_profiler_scope_t(bool enable)
  {
    if (enable) cudaProfilerStart();
    active_ = enable;
  }
  ~cuda_profiler_scope_t()
  {
    if (active_) cudaProfilerStop();
  }
  cuda_profiler_scope_t(const cuda_profiler_scope_t&)            = delete;
  cuda_profiler_scope_t& operator=(const cuda_profiler_scope_t&) = delete;

 private:
  bool active_{false};
};

template <typename Iterator>
static double vector_l2_norm(Iterator first, Iterator last)
{
  return std::sqrt(std::inner_product(first, last, first, 0.0));
}

static bool is_memory_allocation_failure(const cuopt::logic_error& err)
{
  if (err.get_error_type() != error_type_t::RuntimeError &&
      err.get_error_type() != error_type_t::OutOfMemoryError) {
    return false;
  }
  return std::strstr(err.what(), "Memory allocation") != nullptr;
}

template <typename i_t, typename f_t>
static bool is_memory_allocation_failure(
  const cuopt::mathematical_optimization::optimization_problem_solution_t<i_t, f_t>& sol)
{
  const auto& err = sol.get_error_status();
  return err.get_error_type() != error_type_t::Success && is_memory_allocation_failure(err);
}

template <typename i_t, typename f_t>
static thread_local int climber0_projection_status = 0;

// ---------------------------------------------------------------------------
// Batched-PDLP feasibility pump (cloud projection)
// ---------------------------------------------------------------------------

// Build the fixed unified projection problem once: for every non-binary integer variable j create
// an aux distance var d_j >= 0 (upper (u_j - l_j) + int_tol) and two abs-value constraints
//   d_j - x_j >= -val_j   and   d_j + x_j >= val_j
// so that d_j = |x_j - val_j| at optimum. Binary distance is linear: +x_j for target 0 and -x_j
// for target 1. The structure is shared across all climbers and outer iterations.
template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::build_unified_projection_problem(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("build_unified_projection_problem");
  auto* pb          = solution.problem_ptr;
  auto stream       = solution.handle_ptr->get_stream();
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;

  unified_n_vars              = pb->n_variables;
  unified_n_constr            = pb->n_constraints;
  h_integer_indices_cache     = cuopt::host_copy(pb->integer_indices, stream);
  h_aux_integer_indices_cache = cuopt::host_copy(pb->nonbinary_indices, stream);
  unified_n_int               = (i_t)h_integer_indices_cache.size();
  unified_n_aux               = (i_t)h_aux_integer_indices_cache.size();
  unified_n_vars_total        = unified_n_vars + unified_n_aux;
  unified_n_constr_total      = unified_n_constr + 2 * unified_n_aux;

  auto h_var_bounds       = cuopt::host_copy(pb->variable_bounds, stream);
  h_base_constraint_lower = cuopt::host_copy(pb->constraint_lower_bounds, stream);
  h_base_constraint_upper = cuopt::host_copy(pb->constraint_upper_bounds, stream);
  h_csr_offsets           = cuopt::host_copy(pb->offsets, stream);
  h_csr_indices           = cuopt::host_copy(pb->variables, stream);
  h_csr_values            = cuopt::host_copy(pb->coefficients, stream);
  solution.handle_ptr->sync_stream();
  h_csr_offsets.reserve(unified_n_constr_total + 1);
  h_csr_indices.reserve(h_csr_values.size() + 4 * unified_n_aux);
  h_csr_values.reserve(h_csr_values.size() + 4 * unified_n_aux);

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
  for (i_t k = 0; k < unified_n_aux; ++k) {
    i_t col                 = h_aux_integer_indices_cache[k];
    vlb[unified_n_vars + k] = 0.;
    vub[unified_n_vars + k] = (h_var_upper[col] - h_var_lower[col]) + int_tol;
  }

  std::vector<f_t> clb(unified_n_constr_total);
  std::vector<f_t> cub(unified_n_constr_total);
  for (i_t r = 0; r < unified_n_constr; ++r) {
    clb[r] = h_base_constraint_lower[r];
    cub[r] = h_base_constraint_upper[r];
  }
  for (i_t k = 0; k < unified_n_aux; ++k) {
    i_t col_x = h_aux_integer_indices_cache[k];
    i_t col_d = unified_n_vars + k;
    // C1: d - x >= -val (columns sorted ascending: x first, then aux d)
    h_csr_indices.push_back(col_x);
    h_csr_values.push_back(-1.);
    h_csr_indices.push_back(col_d);
    h_csr_values.push_back(1.);
    h_csr_offsets.push_back((i_t)h_csr_values.size());
    // C2: d + x >= val
    h_csr_indices.push_back(col_x);
    h_csr_values.push_back(1.);
    h_csr_indices.push_back(col_d);
    h_csr_values.push_back(1.);
    h_csr_offsets.push_back((i_t)h_csr_values.size());
    clb[unified_n_constr + 2 * k]     = 0.;
    cub[unified_n_constr + 2 * k]     = (f_t)default_cont_upper;
    clb[unified_n_constr + 2 * k + 1] = 0.;
    cub[unified_n_constr + 2 * k + 1] = (f_t)default_cont_upper;
  }

  // Initial (distance-only) objective so the device vector is sized n_vars_total; rebuilt each
  // outer iteration with the alpha-blended objective.
  std::vector<f_t> obj(unified_n_vars_total, 0.);
  for (i_t k = 0; k < unified_n_aux; ++k) {
    obj[unified_n_vars + k] = 1.;
  }

  unified_problem =
    std::make_unique<cuopt::mathematical_optimization::optimization_problem_t<i_t, f_t>>(
      solution.handle_ptr);
  auto& op = *unified_problem;
  op.set_maximize(false);
  op.set_csr_constraint_matrix(h_csr_values.data(),
                               (i_t)h_csr_values.size(),
                               h_csr_indices.data(),
                               (i_t)h_csr_indices.size(),
                               h_csr_offsets.data(),
                               (i_t)h_csr_offsets.size());
  op.set_constraint_lower_bounds(clb.data(), (i_t)clb.size());
  op.set_constraint_upper_bounds(cub.data(), (i_t)cub.size());
  op.set_objective_coefficients(obj.data(), (i_t)obj.size());
  op.set_variable_lower_bounds(vlb.data(), (i_t)vlb.size());
  op.set_variable_upper_bounds(vub.data(), (i_t)vub.size());
  op.set_variable_types(vtypes.data(), (i_t)vtypes.size());
  cloud_batch_capacity    = 0;
  cached_cloud_batch_size = -1;
}

template <typename i_t, typename f_t>
i_t feasibility_pump_t<i_t, f_t>::compute_cloud_batch_size(solution_t<i_t, f_t>& solution)
{
  cuopt_assert(unified_problem != nullptr, "Unified problem must be built first");
  if (cached_cloud_batch_size >= 0) { return cached_cloud_batch_size; }
  const size_t batch_cap = cuopt::mathematical_optimization::compute_optimal_batch_size(
    *unified_problem,
    /*per_climber_objectives=*/true,
    /*per_climber_constraint_bounds=*/true,
    /*collect_solutions=*/true);
  if (batch_cap < (size_t)batch_config.fallback_threshold) {
    cached_cloud_batch_size = 0;
    return 0;
  }
  const size_t clamped = std::clamp(batch_cap,
                                    static_cast<size_t>(batch_config.target_min_batch_size),
                                    static_cast<size_t>(batch_config.target_max_batch_size));
  const size_t latency_clamped =
    std::min(clamped, static_cast<size_t>(batch_config.latency_max_batch_size));
  cached_cloud_batch_size = (i_t)latency_clamped;
  return cached_cloud_batch_size;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::expand_unified_projection_batch_buffers(
  solution_t<i_t, f_t>& solution, i_t batch_capacity)
{
  raft::common::nvtx::range fun_scope("expand_unified_projection_batch_buffers");
  cuopt_assert(unified_problem != nullptr, "Unified problem must be built first");
  cuopt_assert(batch_capacity > 0, "Batch capacity must be positive");
  auto stream            = solution.handle_ptr->get_stream();
  auto& op               = *unified_problem;
  cloud_batch_capacity   = batch_capacity;
  const size_t obj_size  = (size_t)batch_capacity * unified_n_vars_total;
  const size_t cstr_size = (size_t)batch_capacity * unified_n_constr_total;
  op.get_objective_coefficients().resize(obj_size, stream);
  op.get_constraint_lower_bounds().resize(cstr_size, stream);
  op.get_constraint_upper_bounds().resize(cstr_size, stream);
  batch_primal_init.resize(obj_size, stream);
  solution.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::ensure_batch_problem_views(solution_t<i_t, f_t>& solution,
                                                              i_t try_n)
{
  auto stream            = solution.handle_ptr->get_stream();
  auto& op               = *unified_problem;
  const size_t obj_size  = (size_t)try_n * unified_n_vars_total;
  const size_t cstr_size = (size_t)try_n * unified_n_constr_total;
  op.get_objective_coefficients().resize(obj_size, stream);
  op.get_constraint_lower_bounds().resize(cstr_size, stream);
  op.get_constraint_upper_bounds().resize(cstr_size, stream);
  batch_primal_init.resize(obj_size, stream);
  solution.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::project_cloud(solution_t<i_t, f_t>& solution,
                                                 i_t& n_points,
                                                 rmm::device_uvector<f_t>& d_batch_assignments,
                                                 i_t)
{
  raft::common::nvtx::range fun_scope("project_cloud");
  auto stream      = solution.handle_ptr->get_stream();
  auto& op         = *unified_problem;
  const i_t n_vars = unified_n_vars;
  const i_t nct    = unified_n_constr_total;
  const i_t nvt    = unified_n_vars_total;

  if ((i_t)climber_alphas.size() < n_points) { climber_alphas.resize(n_points, default_alpha); }

  std::vector<f_t> orig_obj_vector =
    cuopt::host_copy(solution.problem_ptr->objective_coefficients, stream);
  solution.handle_ptr->sync_stream();
  const f_t l2_norm_of_original_obj =
    vector_l2_norm(orig_obj_vector.begin(), orig_obj_vector.end());
  const f_t l2_norm_of_distance_obj = std::sqrt((f_t)unified_n_int);
  CUOPT_LOG_TRACE("l2_norm_of_original_obj %f l2_norm_of_distance_obj %f",
                  l2_norm_of_original_obj,
                  l2_norm_of_distance_obj);

  const i_t n_constr   = unified_n_constr;
  const f_t cont_upper = (f_t)default_cont_upper;
  const f_t int_tol    = context.settings.tolerances.integrality_tolerance;
  const f_t rlp_base   = context.settings.heuristic_params.relaxed_lp_time_limit;
  const double lp_tolerance =
    get_tolerance_from_ratio(climber0_int_ratio, context.settings.tolerances.absolute_tolerance);

  CUOPT_LOG_TRACE(
    "changing alpha from %f to %f", config.alpha, config.alpha * config.alpha_decrease_factor);
  for (i_t c = 0; c < n_points; ++c) {
    climber_alphas[c] *= config.alpha_decrease_factor;
  }
  config.alpha                       = climber_alphas[0];
  f_t climber0_orig_obj_weight       = config.alpha / l2_norm_of_original_obj;
  const f_t climber0_distance_weight = (1. - config.alpha) / l2_norm_of_distance_obj;
  if (!isfinite(climber0_orig_obj_weight)) {
    CUOPT_LOG_TRACE("orig_obj_weight is not finite, setting to zero");
    climber0_orig_obj_weight = 0.;
  }
  CUOPT_LOG_TRACE(
    "dist weight %f obj weight %f", climber0_distance_weight, climber0_orig_obj_weight);

  i_t try_n = n_points;
  while (try_n >= 1) {
    bool retry_after_oom = false;
    try {
      cuopt_assert(try_n <= cloud_batch_capacity, "try_n exceeds pre-expanded batch capacity");
      const size_t obj_view  = (size_t)try_n * nvt;
      const size_t cstr_view = (size_t)try_n * nct;
      if (op.get_objective_coefficients().size() != obj_view) {
        ensure_batch_problem_views(solution, try_n);
      }

      auto& d_obj = op.get_objective_coefficients();
      rmm::device_uvector<f_t> d_climber_alphas((size_t)try_n, stream);
      raft::copy(d_climber_alphas.data(), climber_alphas.data(), try_n, stream);
      const f_t* orig_obj_ptr = solution.problem_ptr->objective_coefficients.data();
      const i_t* binary_ptr   = solution.problem_ptr->is_binary_variable.data();
      const var_t* var_types  = solution.problem_ptr->variable_types.data();
      const i_t* aux_idx_ptr  = solution.problem_ptr->nonbinary_indices.data();
      const f_t* vlb_ptr      = op.get_variable_lower_bounds().data();
      const f_t* vub_ptr      = op.get_variable_upper_bounds().data();
      const f_t* cloud_ptr    = d_batch_assignments.data();
      const f_t* alpha_ptr    = d_climber_alphas.data();
      f_t* obj_ptr            = d_obj.data();
      thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                       thrust::make_counting_iterator<size_t>(0),
                       thrust::make_counting_iterator<size_t>(obj_view),
                       [=] __device__(size_t g) {
                         const size_t c            = g / (size_t)nvt;
                         const i_t loc             = (i_t)(g % (size_t)nvt);
                         const f_t alpha           = alpha_ptr[c];
                         const f_t distance_weight = (1. - alpha) / l2_norm_of_distance_obj;
                         f_t orig_obj_weight       = alpha / l2_norm_of_original_obj;
                         if (!isfinite(orig_obj_weight)) { orig_obj_weight = 0.; }
                         f_t dist_obj = 0.;
                         if (loc < n_vars && var_types[loc] == var_t::INTEGER) {
                           const f_t target    = cloud_ptr[c * (size_t)n_vars + (size_t)loc];
                           const bool at_lower = abs(target - vlb_ptr[loc]) <= int_tol;
                           const bool at_upper = abs(target - vub_ptr[loc]) <= int_tol;
                           if (binary_ptr[loc] || at_lower || at_upper) {
                             dist_obj = at_upper ? f_t(-1.) : f_t(1.);
                           }
                         } else if (loc >= n_vars) {
                           const i_t col       = aux_idx_ptr[loc - n_vars];
                           const f_t target    = cloud_ptr[c * (size_t)n_vars + (size_t)col];
                           const bool at_lower = abs(target - vlb_ptr[col]) <= int_tol;
                           const bool at_upper = abs(target - vub_ptr[col]) <= int_tol;
                           dist_obj            = at_lower || at_upper ? f_t(0.) : f_t(1.);
                         }
                         const f_t orig_obj = loc < n_vars ? orig_obj_ptr[loc] : f_t(0.);
                         obj_ptr[g] = dist_obj * distance_weight + orig_obj_weight * orig_obj;
                       });
      solution.handle_ptr->sync_stream();

      auto& d_clb             = op.get_constraint_lower_bounds();
      auto& d_cub             = op.get_constraint_upper_bounds();
      const f_t* base_clb_ptr = solution.problem_ptr->constraint_lower_bounds.data();
      const f_t* base_cub_ptr = solution.problem_ptr->constraint_upper_bounds.data();
      f_t* clb_ptr            = d_clb.data();
      f_t* cub_ptr            = d_cub.data();
      thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                       thrust::make_counting_iterator<size_t>(0),
                       thrust::make_counting_iterator<size_t>((size_t)try_n * nct),
                       [=] __device__(size_t g) {
                         const size_t c = g / (size_t)nct;
                         const i_t r    = (i_t)(g % (size_t)nct);
                         if (r < n_constr) {
                           clb_ptr[g] = base_clb_ptr[r];
                           cub_ptr[g] = base_cub_ptr[r];
                         } else {
                           const i_t rr        = r - n_constr;
                           const i_t k         = rr / 2;
                           const i_t col       = aux_idx_ptr[k];
                           const f_t val       = cloud_ptr[c * (size_t)n_vars + (size_t)col];
                           const bool at_lower = abs(val - vlb_ptr[col]) <= int_tol;
                           const bool at_upper = abs(val - vub_ptr[col]) <= int_tol;
                           clb_ptr[g] =
                             at_lower || at_upper ? -cont_upper : ((rr & 1) ? val : -val);
                           cub_ptr[g] = cont_upper;
                         }
                       });
      solution.handle_ptr->sync_stream();

      pdlp_solver_settings_t<i_t, f_t> settings;
      settings.method           = cuopt::mathematical_optimization::method_t::PDLP;
      settings.presolver        = presolver_t::None;
      settings.fixed_batch_size = try_n;
      settings.generate_batch_primal_dual_solution = true;
      const double base_time_limit =
        std::max(0.05, std::min((double)rlp_base, timer.remaining_time() / 10.));
      settings.time_limit =
        std::min(timer.remaining_time(), std::sqrt((double)try_n) * base_time_limit);
      settings.set_optimality_tolerance(lp_tolerance);
      settings.per_constraint_residual = true;
      settings.save_best_primal_so_far = false;
      settings.detect_infeasibility    = false;

      {
        const f_t* cloud_pp            = d_batch_assignments.data();
        const f_t* vlb_ptr             = op.get_variable_lower_bounds().data();
        const f_t* vub_ptr             = op.get_variable_upper_bounds().data();
        const f_t* last_projection_ptr = last_projection.data();
        f_t* pinit                     = batch_primal_init.data();
        thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                         thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(obj_view),
                         [=] __device__(size_t g) {
                           const size_t c = g / (size_t)nvt;
                           const i_t loc  = (i_t)(g % (size_t)nvt);
                           f_t v          = 0.;
                           if (loc < n_vars) {
                             v = cloud_pp[c * (size_t)n_vars + (size_t)loc];
                           } else if (c == 0) {
                             const i_t col = aux_idx_ptr[loc - n_vars];
                             v             = abs(cloud_pp[col] - last_projection_ptr[col]);
                           }
                           if (!isfinite(v)) v = f_t(0);
                           const f_t lo = vlb_ptr[loc];
                           const f_t hi = vub_ptr[loc];
                           pinit[g]     = (v < lo) ? lo : ((v > hi) ? hi : v);
                         });
      }
      settings.set_initial_primal_solution(batch_primal_init.data(), (i_t)obj_view, stream);
      solution.handle_ptr->sync_stream();

      auto sol   = cuopt::mathematical_optimization::run_batch_pdlp(op, settings);
      auto& term = sol.get_terminations_status();
      climber0_projection_status<i_t, f_t> = term.empty() ? 0 : (int)term[0];
      auto& primal                         = sol.get_primal_solution();
      auto& dual                           = sol.get_dual_solution();
      const size_t exp_primal              = (size_t)try_n * unified_n_vars_total;
      const size_t exp_dual                = (size_t)try_n * nct;
      const bool wrong_size                = primal.size() != exp_primal || dual.size() != exp_dual;
      if (wrong_size) {
        const auto es = sol.get_error_status();
        const std::string tstr =
          term.empty() ? std::string("none") : sol.get_termination_status_string(0);
        CUOPT_LOG_ERROR(
          "Batch projection WRONG SIZE at try_n=%d: primal %zu dual %zu (expected %zu/%zu); "
          "error_type=%d msg='%s' term_status[0]='%s' n_term=%zu | inputs: nvt=%d nct=%d "
          "obj_size=%zu clb_size=%zu cub_size=%zu primal_init=%zu dual_init=%zu",
          try_n,
          primal.size(),
          dual.size(),
          exp_primal,
          exp_dual,
          (int)es.get_error_type(),
          es.what(),
          tstr.c_str(),
          term.size(),
          nvt,
          nct,
          op.get_objective_coefficients().size(),
          op.get_constraint_lower_bounds().size(),
          op.get_constraint_upper_bounds().size(),
          (size_t)obj_view,
          (size_t)0);
      }
      if (is_memory_allocation_failure(sol) || (wrong_size && try_n > 1)) {
        retry_after_oom = true;
      } else if (wrong_size) {
        // try_n == 1 and the batch solve still returned an empty/undersized primal-dual (LP error,
        // or OOM under concurrent B&B pressure that PDLP did not surface as an allocation failure).
        // Do not crash: signal the caller (n_points == 0) to fall back to a plain rounding step.
        CUOPT_LOG_ERROR(
          "Batch projection returned no usable solution at batch size 1 (primal %zu, dual %zu; "
          "expected %zu / %zu); falling back to single rounding",
          primal.size(),
          dual.size(),
          exp_primal,
          exp_dual);
        n_points = 0;
        return;
      } else {
        const auto term_infos = sol.get_additional_termination_informations();
        cuopt_assert(term_infos.size() == (size_t)try_n,
                     "Batch termination information size mismatch");
        double mean_iterations = 0.;
        i_t max_iterations     = 0;
        for (const auto& info : term_infos) {
          mean_iterations += info.number_of_steps_taken;
          max_iterations = std::max(max_iterations, info.number_of_steps_taken);
        }
        mean_iterations /= try_n;
        const auto& climber0_info = term_infos[0];
        set_projection_solver_metrics(climber0_info.number_of_steps_taken,
                                      climber0_info.total_number_of_attempted_steps,
                                      (i_t)term[0],
                                      climber0_info.l2_primal_residual,
                                      climber0_info.l2_dual_residual,
                                      climber0_info.gap,
                                      mean_iterations,
                                      max_iterations);
        for (i_t c = 0; c < try_n; ++c) {
          raft::copy(d_batch_assignments.data() + (size_t)c * n_vars,
                     primal.data() + (size_t)c * unified_n_vars_total,
                     n_vars,
                     stream);
        }
        solution.handle_ptr->sync_stream();
        if (try_n < n_points) {
          CUOPT_LOG_WARN(
            "Batch projection reduced cloud from %d to %d climbers after OOM", n_points, try_n);
        }
        n_points = try_n;
        return;
      }
    } catch (const rmm::out_of_memory&) {
      retry_after_oom = true;
      cudaGetLastError();
    } catch (const std::bad_alloc&) {
      retry_after_oom = true;
    }

    if (!retry_after_oom) break;

    if (try_n == 1) {
      // Even a single-climber projection cannot allocate. Fall back to a plain rounding step
      // (n_points == 0) instead of crashing the whole solve.
      CUOPT_LOG_ERROR(
        "Batch projection OOM at minimum batch size 1; falling back to single rounding");
      n_points = 0;
      return;
    }
    CUOPT_LOG_WARN("Batch projection OOM at %d climbers; halving batch size", try_n);
    try_n = std::max((i_t)1, try_n / 2);
  }
}

// One round of the original single-point FP applied to climber 0's projection (already loaded into
// solution.assignment): distance-cycle check, full-integer + near-feasible LP-verify, CP round,
// then the 20% FJ fallback. Mirrors run_single_fp_descent's per-iteration body and uses the shared
// FP state. Returns whether climber 0 reached feasibility; sets climber0_cycle when it cycled.
template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_climber0_step(solution_t<i_t, f_t>& solution,
                                                     f_t proj_begin,
                                                     bool& climber0_cycle,
                                                     i_t batch_size)
{
  raft::common::nvtx::range fun_scope("run_climber0_step");
  using timing_clock = std::chrono::steady_clock;
  climber0_cycle     = false;
  bool is_feasible   = solution.compute_feasibility();
  i_t n_integers     = solution.compute_number_of_integers();
  // Record climber 0's projection integrality to drive the next projection's adaptive LP tolerance.
  climber0_int_ratio = (f_t)n_integers / solution.problem_ptr->n_integer_vars;
  CUOPT_LOG_INFO("after fp projection n_integers %d total n_integes %d",
                 n_integers,
                 solution.problem_ptr->n_integer_vars);
  record_projection_metrics(solution, n_integers, batch_size, proj_begin - timer.remaining_time());
  if (!is_feasible) {
    CUOPT_LOG_INFO("LP is infeasible returning the current PDLP solution! Code %d",
                   climber0_projection_status<i_t, f_t>);
  }

  bool is_cycle = true;
  if (config.check_distance_cycle) {
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
      climber0_cycle = true;
      finish_iteration_metrics(true, false, false);
      return false;
    }
  }

  if (n_integers == solution.problem_ptr->n_integer_vars) {
    if (is_feasible) {
      CUOPT_LOG_INFO(
        "[FP_FEASIBLE][climber=0] Feasible solution found after LP with relative tolerance");
      finish_iteration_metrics(false, false, true);
      return true;
    }
    // Fully integer but PDLP's loose batch tolerance can leave a sub-MIP-tolerance violation; when
    // essentially on the polytope, verify with a full-precision LP (integers fixed).
    else if (!last_distances.empty() && last_distances[0] < distance_to_check_for_feasible) {
      const f_t lp_verify_time_limit = 5.;
      relaxed_lp_settings_t lp_settings;
      lp_settings.time_limit            = lp_verify_time_limit;
      lp_settings.tolerance             = solution.problem_ptr->tolerances.absolute_tolerance;
      lp_settings.return_first_feasible = true;
      lp_settings.save_state            = true;
      // We are verifying a candidate we believe is on the polytope; leaving PDLP's infeasibility
      // detection on makes it return a spurious PrimalInfeasible certificate for a near-feasible
      // (but not-yet-converged) point instead of converging to the feasible completion.
      lp_settings.check_infeasibility = false;
      run_lp_with_vars_fixed(*solution.problem_ptr,
                             solution,
                             solution.problem_ptr->integer_indices,
                             lp_settings,
                             &constraint_prop.bounds_update);
      is_feasible = solution.get_feasible();
      n_integers  = solution.compute_number_of_integers();
      if (is_feasible && n_integers == solution.problem_ptr->n_integer_vars) {
        CUOPT_LOG_INFO("[FP_FEASIBLE][climber=0] Feasible solution verified with LP");
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
    is_feasible = test_fj_feasible(solution, batch_config.fj_ratio * proj_and_round_time);
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
    climber0_cycle = true;
    finish_iteration_metrics(true, false, false);
    return false;
  }
  cycle_queue.n_iterations_without_cycle++;
  finish_iteration_metrics(false, false, false);
  return false;
}

template <typename i_t, typename f_t>
__global__ void batch_diversity_nearest_round_kernel(f_t* __restrict__ assignments,
                                                     const i_t* __restrict__ integer_cols,
                                                     const f_t* __restrict__ var_lower,
                                                     const f_t* __restrict__ var_upper,
                                                     i_t n_points,
                                                     i_t n_vars,
                                                     i_t n_int,
                                                     f_t int_tol)
{
  const i_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const i_t n_work = (n_points - 1) * n_int;
  if (tid >= n_work) return;
  const i_t climber = tid / n_int + 1;
  const i_t k       = tid % n_int;
  const i_t col     = integer_cols[k];
  const size_t pos  = (size_t)climber * (size_t)n_vars + (size_t)col;
  const f_t v       = assignments[pos];
  f_t int_lb        = ceil(var_lower[col] - int_tol);
  f_t int_ub        = floor(var_upper[col] + int_tol);
  f_t nearest       = round(v);
  if (nearest < int_lb) nearest = int_lb;
  if (nearest > int_ub) nearest = int_ub;
  assignments[pos] = nearest;
}

template <typename i_t, typename f_t>
__global__ void batch_climber_int_hash_kernel(const f_t* __restrict__ assignments,
                                              const i_t* __restrict__ integer_cols,
                                              size_t* __restrict__ hashes_out,
                                              i_t n_points,
                                              i_t n_vars,
                                              i_t n_int)
{
  const i_t climber = blockIdx.x + 1;
  if (climber >= n_points || threadIdx.x != 0) return;
  combine_hash combine;
  size_t h      = (size_t)n_int;
  const f_t* pt = assignments + (size_t)climber * (size_t)n_vars;
  for (i_t k = 0; k < n_int; ++k) {
    i_t col = integer_cols[k];
    h       = combine(h, (size_t)llround(pt[col]));
  }
  hashes_out[climber] = h;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::seed_cloud_from_assignment_gpu(solution_t<i_t, f_t>& solution,
                                                                  i_t n_points,
                                                                  rmm::device_uvector<f_t>& d_cloud)
{
  raft::common::nvtx::range fun_scope("seed_cloud_from_assignment_gpu");
  auto stream           = solution.handle_ptr->get_stream();
  const i_t n_vars      = unified_n_vars;
  const i_t n_int       = unified_n_int;
  const f_t int_tol     = context.settings.tolerances.integrality_tolerance;
  const size_t cloud_sz = (size_t)n_points * (size_t)n_vars;
  if (d_cloud.size() != cloud_sz) { d_cloud.resize(cloud_sz, stream); }
  const f_t* assignment = solution.assignment.data();
  thrust::tabulate(solution.handle_ptr->get_thrust_policy(),
                   d_cloud.begin(),
                   d_cloud.end(),
                   [=] __device__(size_t g) { return assignment[g % (size_t)n_vars]; });
  raft::copy(d_cloud.data(), last_rounding.data(), n_vars, stream);
  const i_t* d_int_cols = solution.problem_ptr->integer_indices.data();
  const f_t* d_vlb      = unified_problem->get_variable_lower_bounds().data();
  const f_t* d_vub      = unified_problem->get_variable_upper_bounds().data();
  f_t* cloud            = d_cloud.data();
  const uint64_t seed   = (uint64_t)diversity_rng();
  thrust::for_each(
    solution.handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>((size_t)(n_points - 1) * n_int),
    [=] __device__(size_t g) {
      const i_t climber              = (i_t)(g / (size_t)n_int) + 1;
      const i_t k                    = (i_t)(g % (size_t)n_int);
      const i_t col                  = d_int_cols[k];
      f_t val                        = assignment[col];
      const i_t exponent             = std::min(climber - 1, (i_t)4);
      const f_t target_perturbations = (f_t)(8 << exponent);
      const f_t perturb_ratio        = std::min(f_t(0.1), target_perturbations / (f_t)n_int);
      raft::random::PCGenerator rng(seed, (uint64_t)climber * (uint64_t)n_int + (uint64_t)k, 0);
      if (rng.next_float() < perturb_ratio) {
        f_t int_lb = ceil(d_vlb[col] - int_tol);
        f_t int_ub = floor(d_vub[col] + int_tol);
        if (int_lb <= int_ub) {
          cuopt_assert(isfinite(int_lb) || isfinite(int_ub),
                       "Free integer variable cannot be perturbed");
          if (!isfinite(int_lb)) {
            val = int_ub;
          } else if (!isfinite(int_ub)) {
            val = int_lb;
          } else {
            f_t span = int_ub - int_lb + 1.;
            val      = int_lb + floor(rng.next_float() * span);
          }
        }
      }
      cloud[(size_t)climber * (size_t)n_vars + (size_t)col] = val;
    });
  solution.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::replace_flagged_climbers_diverse(
  solution_t<i_t, f_t>& solution,
  i_t n_points,
  rmm::device_uvector<f_t>& d_cloud,
  const std::vector<char>& flagged,
  i_t& n_diverse,
  i_t& n_fallback)
{
  raft::common::nvtx::range fun_scope("replace_flagged_climbers_diverse");
  n_diverse         = 0;
  n_fallback        = 0;
  auto stream       = solution.handle_ptr->get_stream();
  const i_t n_vars  = unified_n_vars;
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;
  rmm::device_uvector<char> d_flagged((size_t)n_points, stream);
  raft::copy(d_flagged.data(), flagged.data(), n_points, stream);
  const f_t* assignment = solution.assignment.data();
  f_t* cloud            = d_cloud.data();
  const char* flags     = d_flagged.data();
  thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>((size_t)n_points * n_vars),
                   [=] __device__(size_t g) {
                     const i_t climber = (i_t)(g / (size_t)n_vars);
                     if (climber == 0 || flags[climber] == 0) { return; }
                     const i_t j = (i_t)(g % (size_t)n_vars);
                     cloud[g]    = assignment[j];
                   });
  const i_t* d_int_cols = solution.problem_ptr->integer_indices.data();
  const f_t* d_vlb      = unified_problem->get_variable_lower_bounds().data();
  const f_t* d_vub      = unified_problem->get_variable_upper_bounds().data();
  const i_t n_int       = unified_n_int;
  const uint64_t seed   = (uint64_t)diversity_rng();
  thrust::for_each(
    solution.handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>((size_t)n_points * n_int),
    [=] __device__(size_t g) {
      const i_t climber = (i_t)(g / (size_t)n_int);
      if (climber == 0 || flags[climber] == 0) { return; }
      const i_t k                    = (i_t)(g % (size_t)n_int);
      const i_t col                  = d_int_cols[k];
      f_t val                        = assignment[col];
      const i_t exponent             = std::min(climber - 1, (i_t)4);
      const f_t target_perturbations = (f_t)(8 << exponent);
      const f_t perturb_ratio        = std::min(f_t(0.1), target_perturbations / (f_t)n_int);
      raft::random::PCGenerator rng(seed, (uint64_t)climber * (uint64_t)n_int + (uint64_t)k, 0);
      if (rng.next_float() < perturb_ratio) {
        f_t int_lb = ceil(d_vlb[col] - int_tol);
        f_t int_ub = floor(d_vub[col] + int_tol);
        if (int_lb <= int_ub) {
          cuopt_assert(isfinite(int_lb) || isfinite(int_ub),
                       "Free integer variable cannot be perturbed");
          if (!isfinite(int_lb)) {
            val = int_ub;
          } else if (!isfinite(int_ub)) {
            val = int_lb;
          } else {
            f_t span = int_ub - int_lb + 1.;
            val      = int_lb + floor(rng.next_float() * span);
          }
        }
      }
      cloud[(size_t)climber * (size_t)n_vars + (size_t)col] = val;
    });
  for (i_t c = 1; c < n_points; ++c) {
    n_diverse += flagged[c];
  }
  solution.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::advance_diversity_climbers_gpu(
  solution_t<i_t, f_t>& solution,
  i_t n_points,
  rmm::device_uvector<f_t>& d_batch_assignments,
  std::vector<char>& flagged,
  std::vector<size_t>& climber_hashes)
{
  raft::common::nvtx::range fun_scope("advance_diversity_climbers_gpu");
  if (n_points <= 1) {
    flagged.assign((size_t)n_points, 0);
    climber_hashes.assign((size_t)n_points, 0);
    return;
  }
  auto stream       = solution.handle_ptr->get_stream();
  const i_t n_vars  = unified_n_vars;
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;

  const i_t* d_int_cols = solution.problem_ptr->integer_indices.data();
  const f_t* d_vlb      = unified_problem->get_variable_lower_bounds().data();
  const f_t* d_vub      = unified_problem->get_variable_upper_bounds().data();
  constexpr i_t TPB     = 256;
  const i_t n_work      = (n_points - 1) * unified_n_int;
  const i_t n_blocks    = (n_work + TPB - 1) / TPB;
  batch_diversity_nearest_round_kernel<i_t, f_t><<<n_blocks, TPB, 0, stream>>>(
    d_batch_assignments.data(), d_int_cols, d_vlb, d_vub, n_points, n_vars, unified_n_int, int_tol);

  rmm::device_uvector<size_t> d_hashes((size_t)n_points, stream);
  batch_climber_int_hash_kernel<i_t, f_t><<<n_points - 1, 1, 0, stream>>>(
    d_batch_assignments.data(), d_int_cols, d_hashes.data(), n_points, n_vars, unified_n_int);
  solution.handle_ptr->sync_stream();

  auto h_hashes = cuopt::host_copy(d_hashes, stream);
  solution.handle_ptr->sync_stream();
  climber_hashes.assign(h_hashes.begin(), h_hashes.end());
  flagged.assign((size_t)n_points, 0);
  std::unordered_map<size_t, i_t> seen_hash;
  seen_hash.reserve((size_t)n_points * 2);
  for (i_t c = 1; c < n_points; ++c) {
    size_t hsh = h_hashes[c];
    bool cyc   = false;
    for (size_t prev : climber_hash_history[c]) {
      if (prev == hsh) {
        cyc = true;
        break;
      }
    }
    bool dup   = !seen_hash.insert({hsh, c}).second;
    flagged[c] = (cyc || dup) ? 1 : 0;
  }
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::host_assignment_feasible(const f_t* x)
{
  const f_t tol = context.settings.tolerances.absolute_tolerance;
  for (i_t r = 0; r < unified_n_constr; ++r) {
    f_t act = 0.;
    for (i_t p = h_csr_offsets[r]; p < h_csr_offsets[r + 1]; ++p) {
      act += h_csr_values[p] * x[h_csr_indices[p]];
    }
    if (act < h_base_constraint_lower[r] - tol || act > h_base_constraint_upper[r] + tol) {
      return false;
    }
  }
  return true;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_batched_fp_cloud(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("run_batched_fp_cloud");
  // Batched-PDLP cloud is the default path; set CUOPT_FP_SINGLE to force the classic single-point
  // FP (the outer loop drives restarts either way).
  static const bool use_single_fp = std::getenv("CUOPT_FP_SINGLE") != nullptr;
  if (use_single_fp) { return run_single_fp_descent(solution); }

  if (unified_problem == nullptr || unified_n_vars != solution.problem_ptr->n_variables ||
      unified_n_constr != solution.problem_ptr->n_constraints) {
    build_unified_projection_problem(solution);
  }

  const i_t batch_size = compute_cloud_batch_size(solution);
  if (batch_size == 0) { return run_single_fp_descent(solution); }
  const size_t expected_obj_size = (size_t)batch_size * unified_n_vars_total;
  if (cloud_batch_capacity != batch_size ||
      unified_problem->get_objective_coefficients().size() != expected_obj_size) {
    expand_unified_projection_batch_buffers(solution, batch_size);
  }

  auto stream = solution.handle_ptr->get_stream();
  solution.round_nearest();
  climber0_int_ratio = (f_t)solution.n_assigned_integers / solution.problem_ptr->n_integer_vars;
  raft::copy(last_rounding.data(), solution.assignment.data(), solution.assignment.size(), stream);

  rmm::device_uvector<f_t> d_batch_assignments(0, stream);
  std::vector<char> flagged((size_t)batch_size, 0);
  std::vector<size_t> climber_hashes((size_t)batch_size, 0);

  return run_batched_fp_cloud_descent(
    solution, batch_size, d_batch_assignments, flagged, climber_hashes);
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_batched_fp_cloud_descent(
  solution_t<i_t, f_t>& solution,
  i_t batch_size,
  rmm::device_uvector<f_t>& d_batch_assignments,
  std::vector<char>& flagged,
  std::vector<size_t>& climber_hashes)
{
  raft::common::nvtx::range fun_scope("run_batched_fp_cloud_descent");
  static const bool profile_one_iter = std::getenv("CUOPT_FP_PROFILE_ONE_ITER") != nullptr;
  auto stream                        = solution.handle_ptr->get_stream();
  const i_t n_vars                   = unified_n_vars;
  const f_t inf                      = std::numeric_limits<f_t>::infinity();
  rmm::device_uvector<f_t> cand_best(n_vars, stream);
  rmm::device_uvector<f_t> climber0_assign(n_vars, stream);
  rmm::device_uvector<f_t> diversity_candidate(n_vars, stream);
  i_t n_points         = 0;
  bool first_iteration = true;
  i_t trajectory       = 0;

  while (true) {
    if (context.diversity_manager_ptr->check_b_b_preemption() || timer.check_time_limit()) {
      CUOPT_LOG_INFO("FP time limit reached!");
      round(solution);
      return false;
    }
    const cuda_profiler_scope_t profiler_scope(profile_one_iter && trajectory == 0);

    // ---- Build (first round) or refresh the working cloud ----
    if (first_iteration) {
      n_points = batch_size;
      seed_cloud_from_assignment_gpu(solution, n_points, d_batch_assignments);
      climber_alphas.assign(n_points, default_alpha);
      climber_alphas[0] = config.alpha;
      for (i_t c = 1; c < n_points; ++c) {
        climber_alphas[c] = default_alpha * (f_t)(c - 1) / (f_t)std::max((i_t)1, n_points - 2);
      }
      climber_hash_history.assign(n_points, {});
      first_iteration = false;
    } else {
      raft::copy(d_batch_assignments.data(), last_rounding.data(), n_vars, stream);
      solution.handle_ptr->sync_stream();
    }

    CUOPT_LOG_INFO("linear projection of fp");
    proj_begin        = timer.remaining_time();
    f_t old_remaining = timer.remaining_time();
    project_cloud(solution, n_points, d_batch_assignments, trajectory);
    static f_t lp_time = 0;
    static i_t n_calls = 0;
    last_lp_time       = old_remaining - timer.remaining_time();
    lp_time += last_lp_time;
    n_calls++;
    CUOPT_LOG_INFO("lp_time %f average lp_time %f", last_lp_time, lp_time / n_calls);
    if (n_points == 0) {
      cuopt_expects(false, error_type_t::RuntimeError, "Projection produced no usable solution");
    }
    if ((i_t)climber_alphas.size() > n_points) { climber_alphas.resize(n_points); }
    if ((i_t)climber_hash_history.size() > n_points) { climber_hash_history.resize(n_points); }

    // ---- Climber 0 (slot 0): full classic FP with internal restarts (the only CP round) ----
    raft::copy(solution.assignment.data(), d_batch_assignments.data(), n_vars, stream);
    // Batch PDLP's loose tolerance can leave the primal marginally outside variable bounds; clamp.
    solution.clamp_within_bounds();
    raft::copy(
      last_projection.data(), solution.assignment.data(), solution.assignment.size(), stream);
    bool climber0_cycle = false;
    bool feasible0      = run_climber0_step(solution, proj_begin, climber0_cycle, n_points);
    // Best feasible found this trajectory across climber 0 and the diversity climbers.
    f_t best_obj              = inf;
    bool have_feasible        = false;
    i_t best_feasible_climber = -1;
    if (feasible0) {
      std::cout << "[FP_FEASIBLE][climber=0] Feasible candidate found" << std::endl;
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      best_obj              = solution.get_objective();
      have_feasible         = true;
      best_feasible_climber = 0;
      raft::copy(cand_best.data(), solution.assignment.data(), n_vars, stream);
      solution.handle_ptr->sync_stream();
    }
    // The diversity feasibility check below overwrites solution.assignment, but the reseed step
    // perturbs from climber 0's assignment; stash it so we can restore it afterwards.
    raft::copy(d_batch_assignments.data(), solution.assignment.data(), n_vars, stream);
    raft::copy(climber0_assign.data(), solution.assignment.data(), n_vars, stream);
    solution.handle_ptr->sync_stream();
    f_t best_infeasible_excess   = solution.get_total_excess();
    i_t best_infeasible_integers = solution.compute_number_of_integers();
    bool use_diversity_candidate = false;

    // ---- Diversity climbers 1..N: GPU nearest-integer rounding + hash dedup ----
    advance_diversity_climbers_gpu(
      solution, n_points, d_batch_assignments, flagged, climber_hashes);

    // ---- CP-round diversity climbers and keep the best feasible objective ----
    for (i_t c = 1; c < n_points; ++c) {
      raft::copy(solution.assignment.data(),
                 d_batch_assignments.data() + (size_t)c * n_vars,
                 n_vars,
                 stream);
      solution.handle_ptr->sync_stream();
      bool diversity_feasible = solution.compute_feasibility();
      if (!diversity_feasible) {
        const bool old_backtracking            = constraint_prop.enable_backtracking;
        const bool old_repair                  = constraint_prop.enable_repair;
        const i_t old_divisor                  = constraint_prop.bulk_rounding_divisor;
        const i_t old_pre_round_target         = constraint_prop.pre_round_target_unset;
        constraint_prop.enable_backtracking    = false;
        constraint_prop.enable_repair          = false;
        constraint_prop.bulk_rounding_divisor  = 10;
        constraint_prop.pre_round_target_unset = std::numeric_limits<i_t>::max();
        round(solution, false);
        constraint_prop.enable_backtracking    = old_backtracking;
        constraint_prop.enable_repair          = old_repair;
        constraint_prop.bulk_rounding_divisor  = old_divisor;
        constraint_prop.pre_round_target_unset = old_pre_round_target;
        diversity_feasible                     = solution.compute_feasibility();
        raft::copy(d_batch_assignments.data() + (size_t)c * n_vars,
                   solution.assignment.data(),
                   n_vars,
                   stream);
        solution.handle_ptr->sync_stream();
      }
      if (diversity_feasible) {
        std::cout << "[FP_FEASIBLE][climber=" << c << "] Diversity candidate found feasible"
                  << std::endl;
        if (metrics != nullptr) {
          metrics->iterations.back().diversity_feasible_candidates++;
          if (c == 1) { metrics->iterations.back().climber1_feasible_candidates++; }
        }
        f_t obj = solution.get_objective();
        if (obj < best_obj) {
          std::cout << "[FP_FEASIBLE][climber=" << c << "] New best feasible objective " << obj
                    << std::endl;
          best_obj              = obj;
          have_feasible         = true;
          best_feasible_climber = c;
          if (metrics != nullptr) {
            metrics->iterations.back().diversity_best_updates++;
            if (c == 1) { metrics->iterations.back().climber1_best_updates++; }
          }
          raft::copy(cand_best.data(), solution.assignment.data(), n_vars, stream);
          solution.handle_ptr->sync_stream();
        }
      } else {
        const f_t excess   = solution.get_total_excess();
        const i_t integers = solution.compute_number_of_integers();
        if (!solution.problem_ptr->cutting_plane_added &&
            (excess < best_infeasible_excess ||
             (excess == best_infeasible_excess && integers > best_infeasible_integers))) {
          best_infeasible_excess   = excess;
          best_infeasible_integers = integers;
          use_diversity_candidate  = true;
          if (metrics != nullptr) {
            metrics->iterations.back().diversity_best_updates++;
            if (c == 1) { metrics->iterations.back().climber1_best_updates++; }
          }
          raft::copy(diversity_candidate.data(), solution.assignment.data(), n_vars, stream);
          solution.handle_ptr->sync_stream();
        }
      }
    }

    raft::copy(solution.assignment.data(), climber0_assign.data(), n_vars, stream);
    solution.handle_ptr->sync_stream();
    solution.compute_feasibility();
    if (!have_feasible && use_diversity_candidate) {
      raft::copy(solution.assignment.data(), diversity_candidate.data(), n_vars, stream);
      raft::copy(d_batch_assignments.data(), diversity_candidate.data(), n_vars, stream);
      raft::copy(last_rounding.data(), diversity_candidate.data(), n_vars, stream);
      solution.handle_ptr->sync_stream();
      solution.compute_feasibility();
      climber0_int_ratio = (f_t)best_infeasible_integers / solution.problem_ptr->n_integer_vars;
      cycle_queue.update_recent_solutions(solution);
    }

    // ---- Return the best feasible found (climber 0 or a diversity climber) ----
    if (have_feasible) {
      std::cout << "[FP_FEASIBLE][selected_climber=" << best_feasible_climber
                << "] Returning feasible objective " << best_obj << std::endl;
      raft::copy(solution.assignment.data(), cand_best.data(), n_vars, stream);
      solution.handle_ptr->sync_stream();
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      return true;
    }

    i_t n_flagged = 0;
    for (i_t c = 1; c < n_points; ++c) {
      n_flagged += flagged[c];
    }

    if (timer.check_time_limit()) {
      CUOPT_LOG_INFO("FP time limit reached!");
      return false;
    }

    if (n_flagged > 0) {
      i_t n_diverse_perturb  = 0;
      i_t n_fallback_perturb = 0;
      replace_flagged_climbers_diverse(
        solution, n_points, d_batch_assignments, flagged, n_diverse_perturb, n_fallback_perturb);
      for (i_t c = 1; c < n_points; ++c) {
        if (!flagged[c]) continue;
        climber_alphas[c] = default_alpha * (f_t)(c - 1) / (f_t)std::max((i_t)1, n_points - 2);
        climber_hash_history[c].clear();
      }
    }

    raft::copy(last_rounding.data(), d_batch_assignments.data(), n_vars, stream);
    solution.handle_ptr->sync_stream();
    for (i_t c = 1; c < n_points; ++c) {
      if (flagged[c]) continue;
      auto& hist = climber_hash_history[c];
      hist.push_back(climber_hashes[c]);
      while ((i_t)hist.size() > cycle_queue.cycle_detection_length) {
        hist.pop_front();
      }
    }

    if (climber0_cycle) { return false; }
    if (profile_one_iter && trajectory == 0) { return false; }
    trajectory++;
    if (trajectory >= batch_config.max_trajectories_before_restart) { return false; }
  }
  return false;
}

// Explicit instantiation of the batched-PDLP members. The non-batched members are instantiated by
// the `template class feasibility_pump_t<...>` in feasibility_pump.cu; these definitions live in a
// separate translation unit, so the specific members are instantiated here instead.
#define INSTANTIATE_BATCHED(F_TYPE)                                                       \
  template void feasibility_pump_t<int, F_TYPE>::build_unified_projection_problem(        \
    solution_t<int, F_TYPE>&);                                                            \
  template void feasibility_pump_t<int, F_TYPE>::expand_unified_projection_batch_buffers( \
    solution_t<int, F_TYPE>&, int);                                                       \
  template int feasibility_pump_t<int, F_TYPE>::compute_cloud_batch_size(                 \
    solution_t<int, F_TYPE>&);                                                            \
  template void feasibility_pump_t<int, F_TYPE>::seed_cloud_from_assignment_gpu(          \
    solution_t<int, F_TYPE>&, int, rmm::device_uvector<F_TYPE>&);                         \
  template void feasibility_pump_t<int, F_TYPE>::replace_flagged_climbers_diverse(        \
    solution_t<int, F_TYPE>&,                                                             \
    int,                                                                                  \
    rmm::device_uvector<F_TYPE>&,                                                         \
    const std::vector<char>&,                                                             \
    int&,                                                                                 \
    int&);                                                                                \
  template void feasibility_pump_t<int, F_TYPE>::advance_diversity_climbers_gpu(          \
    solution_t<int, F_TYPE>&,                                                             \
    int,                                                                                  \
    rmm::device_uvector<F_TYPE>&,                                                         \
    std::vector<char>&,                                                                   \
    std::vector<size_t>&);                                                                \
  template void feasibility_pump_t<int, F_TYPE>::project_cloud(                           \
    solution_t<int, F_TYPE>&, int&, rmm::device_uvector<F_TYPE>&, int);                   \
  template bool feasibility_pump_t<int, F_TYPE>::run_batched_fp_cloud_descent(            \
    solution_t<int, F_TYPE>&,                                                             \
    int,                                                                                  \
    rmm::device_uvector<F_TYPE>&,                                                         \
    std::vector<char>&,                                                                   \
    std::vector<size_t>&);                                                                \
  template bool feasibility_pump_t<int, F_TYPE>::run_climber0_step(                       \
    solution_t<int, F_TYPE>&, F_TYPE, bool&, int);                                        \
  template bool feasibility_pump_t<int, F_TYPE>::host_assignment_feasible(const F_TYPE*); \
  template bool feasibility_pump_t<int, F_TYPE>::run_batched_fp_cloud(solution_t<int, F_TYPE>&);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE_BATCHED(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE_BATCHED(double)
#endif

#undef INSTANTIATE_BATCHED

}  // namespace cuopt::mathematical_optimization::mip
