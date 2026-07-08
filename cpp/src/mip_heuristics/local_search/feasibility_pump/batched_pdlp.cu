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
#include <cstring>
#include <unordered_map>

#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/linalg/binary_op.cuh>

#include <thrust/copy.h>
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

// Minimum integer-component difference required to accept a new cloud point as diverse. Maps the
// problem's integer-variable count to [1, 25] on a log10 scale (uniform in orders of magnitude):
// ~1 for tiny problems (100 vars), ~25 for very large ones (500k vars). Linear fit through
// (log10(100), 1) and (log10(500000), 25).
template <typename i_t>
static i_t compute_min_int_diff(i_t n_int)
{
  if (n_int <= 1) return 1;
  constexpr double slope     = 6.48836;    // 24 / (log10(500000) - log10(100))
  constexpr double intercept = -11.97672;  // 1 - slope * log10(100)
  const double d             = slope * std::log10((double)n_int) + intercept;
  const i_t rounded          = (i_t)std::lround(d);
  return std::clamp(rounded, i_t(1), std::min(i_t(25), n_int));
}

// Hash the integer components of a host assignment slice, using the shared combine_hash recurrence
// (same as the device assignment_hash_map_t). Seeded with the integer count, matching the device
// hasher's th_hash = assignment.size() convention.
template <typename i_t, typename f_t>
static size_t integer_assignment_hash(const std::vector<i_t>& integer_cols, const f_t* x)
{
  combine_hash combine;
  size_t h = integer_cols.size();
  for (i_t col : integer_cols) {
    h = combine(h, (size_t)std::llround(x[col]));
  }
  return h;
}

// Tracks accepted integer assignments for cloud-point diversity filtering.
template <typename i_t, typename f_t>
struct cloud_point_diversity_t {
  const std::vector<i_t>& integer_cols;
  const std::vector<std::deque<size_t>>& climber_hash_history;
  i_t min_int_diff;
  std::vector<long long> accepted_int_vals;
  std::vector<long long> cand;

  cloud_point_diversity_t(const std::vector<i_t>& cols,
                          const std::vector<std::deque<size_t>>& history,
                          i_t n_int)
    : integer_cols(cols),
      climber_hash_history(history),
      min_int_diff(compute_min_int_diff(n_int)),
      cand(n_int)
  {
    accepted_int_vals.reserve((size_t)n_int * 64);
  }

  void extract_int(const f_t* pt)
  {
    for (i_t k = 0; k < (i_t)integer_cols.size(); ++k) {
      cand[k] = std::llround(pt[integer_cols[k]]);
    }
  }

  bool hash_in_climber_history(size_t hsh) const
  {
    for (const auto& hist : climber_hash_history) {
      for (size_t prev : hist) {
        if (prev == hsh) return true;
      }
    }
    return false;
  }

  bool too_close() const
  {
    const i_t n_int = (i_t)integer_cols.size();
    i_t n_accepted  = (i_t)(accepted_int_vals.size() / std::max(n_int, 1));
    for (i_t a = 0; a < n_accepted; ++a) {
      const long long* base = accepted_int_vals.data() + (size_t)a * n_int;
      i_t diff              = 0;
      for (i_t k = 0; k < n_int && diff < min_int_diff; ++k) {
        if (cand[k] != base[k]) ++diff;
      }
      if (diff < min_int_diff) return true;
    }
    return false;
  }

  bool passes_diversity(const f_t* pt)
  {
    size_t hsh = integer_assignment_hash(integer_cols, pt);
    if (hash_in_climber_history(hsh)) return false;
    extract_int(pt);
    if (too_close()) return false;
    return true;
  }

  void accept(const f_t* pt)
  {
    extract_int(pt);
    accepted_int_vals.insert(accepted_int_vals.end(), cand.begin(), cand.end());
  }

  void seed_from_cloud(const f_t* h_cloud, i_t n_points, i_t n_vars, const std::vector<char>* skip)
  {
    for (i_t c = 0; c < n_points; ++c) {
      if (skip != nullptr && c < (i_t)skip->size() && (*skip)[c]) continue;
      accept(h_cloud + (size_t)c * n_vars);
    }
  }
};

// Blend distance + original objective for one climber at the given alpha (does not mutate alpha).
template <typename f_t>
static void blend_projection_objective(f_t alpha,
                                       f_t l2_norm_of_original_obj,
                                       f_t l2_norm_of_distance_obj,
                                       const std::vector<f_t>& orig_obj_vector,
                                       const std::vector<f_t>& dist_objective,
                                       std::vector<f_t>& out)
{
  f_t distance_weight = (1. - alpha) / l2_norm_of_distance_obj;
  f_t orig_obj_weight = alpha / l2_norm_of_original_obj;
  if (!isfinite(orig_obj_weight)) { orig_obj_weight = 0.; }
  cuopt_expects(isfinite(orig_obj_weight), error_type_t::RuntimeError, "Weight should be finite!");
  for (size_t i = 0; i < out.size(); ++i) {
    f_t orig_obj = i < orig_obj_vector.size() ? orig_obj_vector[i] : 0.;
    out[i]       = dist_objective[i] * distance_weight + orig_obj_weight * orig_obj;
    cuopt_expects(isfinite(out[i]), error_type_t::RuntimeError, "Weight should be finite!");
  }
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

  // Cache the original CSR for the host-side feasibility filter of diversity climbers.
  h_csr_offsets = h_offsets;
  h_csr_indices = h_indices;
  h_csr_values  = h_values;

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
    i_t col                 = h_integer_indices_cache[k];
    vlb[unified_n_vars + k] = 0.;
    vub[unified_n_vars + k] = (h_var_upper[col] - h_var_lower[col]) + int_tol;
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
    std::make_unique<cuopt::mathematical_optimization::optimization_problem_t<i_t, f_t>>(
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
  cloud_batch_capacity = 0;
  // The per-climber sizes just changed, so any stored warm start is stale.
  warm_start_n_points = 0;
  CUOPT_LOG_INFO(
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
  const size_t batch_cap = cuopt::mathematical_optimization::compute_optimal_batch_size(
    *unified_problem,
    /*per_climber_objectives=*/true,
    /*per_climber_constraint_bounds=*/true,
    /*collect_solutions=*/true);
  if (batch_cap < (size_t)batch_config.fallback_threshold) {
    CUOPT_LOG_INFO("Cloud batch size cap %zu below fallback threshold %d; using single FP",
                   batch_cap,
                   batch_config.fallback_threshold);
    return 0;
  }
  const size_t clamped = std::clamp(batch_cap,
                                    static_cast<size_t>(batch_config.target_min_batch_size),
                                    static_cast<size_t>(batch_config.target_max_batch_size));
  const size_t latency_clamped =
    std::min(clamped, static_cast<size_t>(batch_config.latency_max_batch_size));
  CUOPT_LOG_INFO(
    "Cloud batch size %zu (compute_optimal_batch_size cap %zu, target [%d, %d], latency cap %d)",
    latency_clamped,
    batch_cap,
    batch_config.target_min_batch_size,
    batch_config.target_max_batch_size,
    batch_config.latency_max_batch_size);
  return static_cast<i_t>(latency_clamped);
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
  h_batch_obj.resize(obj_size);
  batch_primal_init.resize(obj_size, stream);
  batch_dual_init.resize(cstr_size, stream);
  warm_start_dual.resize(cstr_size, stream);
  warm_start_n_points = 0;
  solution.handle_ptr->sync_stream();
  CUOPT_LOG_INFO("Expanded unified projection batch buffers for %d climbers", batch_capacity);
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
  batch_dual_init.resize(cstr_size, stream);
  solution.handle_ptr->sync_stream();
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

  std::vector<f_t> h_points((size_t)batch_size * n_vars, 0.);
  i_t count = 0;
  // Leading cloud climbers that are carried-over reseed points (positionally aligned with the
  // previous projection's climbers). Used by project_cloud to decide which climbers reuse their own
  // previous dual vs the previous best climber's dual.
  n_carried_over_points = 0;

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
    n_carried_over_points = take;
  }

  // (2) Remaining slots are filled below with perturbed-assignment padding.
  (void)first_iteration;

  // (3) Fill every remaining slot with perturbed integers of the current assignment. The cloud is
  // always seeded to the full batch_size (never shrinks): FJ contributes what it can, perturbation
  // covers the rest with diverse points.
  if (count < batch_size) {
    auto h_assignment = cuopt::host_copy(solution.assignment, stream);
    solution.handle_ptr->sync_stream();
    const i_t pad_start = count;
    std::uniform_real_distribution<double> unit(0., 1.);
    constexpr double perturb_ratio = 0.1;
    while (count < batch_size) {
      std::vector<f_t> pt(h_assignment.begin(), h_assignment.begin() + n_vars);
      for (i_t k = 0; k < unified_n_int; ++k) {
        if (unit(rng) >= perturb_ratio) continue;
        i_t col = h_integer_indices_cache[k];
        f_t lb  = h_var_lower[col];
        f_t ub  = h_var_upper[col];
        if (lb == -std::numeric_limits<f_t>::infinity()) {
          pt[col] = std::floor(ub + int_tol);
        } else if (ub == std::numeric_limits<f_t>::infinity()) {
          pt[col] = std::ceil(lb - int_tol);
        } else {
          std::uniform_int_distribution<i_t> unif((i_t)std::ceil(lb - int_tol),
                                                  (i_t)std::floor(ub + int_tol));
          pt[col] = unif(rng);
        }
      }
      append_point(pt);
    }
    CUOPT_LOG_INFO("Cloud seeding: padded %d points from perturbed assignment to fill batch",
                   count - pad_start);
  }

  if (count == 0) return 0;
  // Reserve cloud slot 0 for climber 0: its dedicated classic-FP trajectory point (last_rounding).
  // Climber 0 is always slot 0 across iterations, so its warm start and dual carry over cleanly.
  {
    auto h_last_rounding = cuopt::host_copy(last_rounding, stream);
    solution.handle_ptr->sync_stream();
    std::copy(h_last_rounding.begin(), h_last_rounding.begin() + n_vars, h_points.begin());
  }
  const size_t cloud_size = (size_t)count * n_vars;
  if (d_cloud.size() != cloud_size) { d_cloud.resize(cloud_size, stream); }
  raft::copy(d_cloud.data(), h_points.data(), cloud_size, stream);
  solution.handle_ptr->sync_stream();
  return count;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::project_cloud(solution_t<i_t, f_t>& solution,
                                                 i_t& n_points,
                                                 const rmm::device_uvector<f_t>& d_cloud,
                                                 rmm::device_uvector<f_t>& d_projected)
{
  raft::common::nvtx::range fun_scope("project_cloud");
  auto stream      = solution.handle_ptr->get_stream();
  auto& op         = *unified_problem;
  const i_t n_vars = unified_n_vars;
  const i_t nct    = unified_n_constr_total;
  const i_t nvt    = unified_n_vars_total;

  if ((i_t)climber_alphas.size() < n_points) { climber_alphas.resize(n_points, default_alpha); }

  std::vector<f_t> dist_base(nvt, 0.);
  for (i_t k = 0; k < unified_n_int; ++k) {
    dist_base[unified_n_vars + k] = 1.;
  }
  std::vector<f_t> orig_obj_vector =
    cuopt::host_copy(solution.problem_ptr->objective_coefficients, stream);
  solution.handle_ptr->sync_stream();
  const f_t l2_norm_of_original_obj =
    vector_l2_norm(orig_obj_vector.begin(), orig_obj_vector.end());
  const f_t l2_norm_of_distance_obj = vector_l2_norm(dist_base.begin(), dist_base.end());

  const i_t n_constr   = unified_n_constr;
  const f_t cont_upper = (f_t)default_cont_upper;
  const f_t rlp_base   = batch_config.projection_time_limit;
  const double lp_tolerance =
    get_tolerance_from_ratio(climber0_int_ratio, context.settings.tolerances.absolute_tolerance);

  for (i_t c = 0; c < n_points; ++c) {
    climber_alphas[c] *= config.alpha_decrease_factor;
  }
  config.alpha = climber_alphas[0];

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

      std::vector<f_t> climber_obj(nvt);
      for (i_t c = 0; c < try_n; ++c) {
        climber_obj = dist_base;
        blend_projection_objective(climber_alphas[c],
                                   l2_norm_of_original_obj,
                                   l2_norm_of_distance_obj,
                                   orig_obj_vector,
                                   dist_base,
                                   climber_obj);
        std::copy(climber_obj.begin(), climber_obj.end(), h_batch_obj.begin() + (size_t)c * nvt);
      }

      auto& d_obj = op.get_objective_coefficients();
      raft::copy(d_obj.data(), h_batch_obj.data(), obj_view, stream);

      auto& d_clb             = op.get_constraint_lower_bounds();
      auto& d_cub             = op.get_constraint_upper_bounds();
      const f_t* base_clb_ptr = solution.problem_ptr->constraint_lower_bounds.data();
      const f_t* base_cub_ptr = solution.problem_ptr->constraint_upper_bounds.data();
      const i_t* int_idx_ptr  = solution.problem_ptr->integer_indices.data();
      const f_t* cloud_ptr    = d_cloud.data();
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
                           const i_t rr  = r - n_constr;
                           const i_t k   = rr / 2;
                           const i_t col = int_idx_ptr[k];
                           const f_t val = cloud_ptr[c * (size_t)n_vars + (size_t)col];
                           clb_ptr[g]    = (rr & 1) ? val : -val;
                           cub_ptr[g]    = cont_upper;
                         }
                       });

      pdlp_solver_settings_t<i_t, f_t> settings;
      settings.method           = cuopt::mathematical_optimization::method_t::PDLP;
      settings.presolver        = presolver_t::None;
      settings.fixed_batch_size = try_n;
      settings.generate_batch_primal_dual_solution = true;
      settings.time_limit =
        std::max(0.05, std::min((double)rlp_base, timer.remaining_time() / 10.));
      settings.set_optimality_tolerance(lp_tolerance);

      {
        const f_t* cloud_pp = d_cloud.data();
        const f_t* vlb_ptr  = op.get_variable_lower_bounds().data();
        const f_t* vub_ptr  = op.get_variable_upper_bounds().data();
        f_t* pinit          = batch_primal_init.data();
        thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                         thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(obj_view),
                         [=] __device__(size_t g) {
                           const size_t c = g / (size_t)nvt;
                           const i_t loc  = (i_t)(g % (size_t)nvt);
                           f_t v =
                             (loc < n_vars) ? cloud_pp[c * (size_t)n_vars + (size_t)loc] : f_t(0);
                           if (!isfinite(v)) v = f_t(0);
                           const f_t lo = vlb_ptr[loc];
                           const f_t hi = vub_ptr[loc];
                           pinit[g]     = (v < lo) ? lo : ((v > hi) ? hi : v);
                         });
      }
      settings.set_initial_primal_solution(batch_primal_init.data(), (i_t)obj_view, stream);

      const bool dual_warm_available =
        warm_start_n_points > 0 && warm_start_dual.size() >= (size_t)warm_start_n_points * nct &&
        warm_start_best_c >= 0 && warm_start_best_c < warm_start_n_points;
      if (dual_warm_available) {
        const i_t carry  = std::min(n_carried_over_points, warm_start_n_points);
        const i_t best   = warm_start_best_c;
        const i_t nct_l  = nct;
        const f_t* wdual = warm_start_dual.data();
        f_t* dinit       = batch_dual_init.data();
        thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                         thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(cstr_view),
                         [=] __device__(size_t g) {
                           const size_t c = g / (size_t)nct_l;
                           const i_t r    = (i_t)(g % (size_t)nct_l);
                           const i_t src  = (c < (size_t)carry) ? (i_t)c : best;
                           const f_t v    = wdual[(size_t)src * (size_t)nct_l + (size_t)r];
                           dinit[g]       = isfinite(v) ? v : f_t(0);
                         });
        settings.set_initial_dual_solution(batch_dual_init.data(), (i_t)cstr_view, stream);
        CUOPT_LOG_INFO(
          "Batch projection warm start: primal from current points; dual %d/%d climbers carried "
          "over, rest from best climber %d",
          std::min(carry, try_n),
          try_n,
          best);
      } else {
        CUOPT_LOG_INFO("Batch projection warm start: primal from current points (cold dual)");
      }

      auto sol                = cuopt::mathematical_optimization::run_batch_pdlp(op, settings);
      auto& primal            = sol.get_primal_solution();
      auto& dual              = sol.get_dual_solution();
      const size_t exp_primal = (size_t)try_n * unified_n_vars_total;
      const size_t exp_dual   = (size_t)try_n * nct;
      const bool wrong_size   = primal.size() != exp_primal || dual.size() != exp_dual;
      if (is_memory_allocation_failure(sol) || (wrong_size && try_n > 1)) {
        retry_after_oom = true;
      } else if (wrong_size) {
        // try_n == 1 and the batch solve still returned an empty/undersized primal-dual (LP error,
        // or OOM under concurrent B&B pressure that PDLP did not surface as an allocation failure).
        // Do not crash: signal the caller (n_points == 0) to fall back to a plain rounding step.
        CUOPT_LOG_INFO(
          "Batch projection returned no usable solution at batch size 1 (primal %zu, dual %zu; "
          "expected %zu / %zu); falling back to single rounding",
          primal.size(),
          dual.size(),
          exp_primal,
          exp_dual);
        n_points = 0;
        return;
      } else {
        for (i_t c = 0; c < try_n; ++c) {
          raft::copy(d_projected.data() + (size_t)c * n_vars,
                     primal.data() + (size_t)c * unified_n_vars_total,
                     n_vars,
                     stream);
        }
        raft::copy(warm_start_dual.data(), dual.data(), dual.size(), stream);
        warm_start_n_points = try_n;
        solution.handle_ptr->sync_stream();
        if (try_n < n_points) {
          CUOPT_LOG_INFO(
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
      CUOPT_LOG_INFO(
        "Batch projection OOM at minimum batch size 1; falling back to single rounding");
      n_points = 0;
      return;
    }
    CUOPT_LOG_INFO("Batch projection OOM at %d climbers; halving batch size", try_n);
    try_n = std::max((i_t)1, try_n / 2);
  }
}

template <typename i_t, typename f_t>
i_t feasibility_pump_t<i_t, f_t>::select_cloud_point(solution_t<i_t, f_t>& solution,
                                                     i_t n_points,
                                                     const rmm::device_uvector<f_t>& d_cloud,
                                                     const rmm::device_uvector<f_t>& d_projected,
                                                     i_t start_climber)
{
  raft::common::nvtx::range fun_scope("select_cloud_point");
  auto stream       = solution.handle_ptr->get_stream();
  const i_t n_vars  = unified_n_vars;
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;

  auto h_cloud     = cuopt::host_copy(d_cloud, stream);
  auto h_projected = cuopt::host_copy(d_projected, stream);
  solution.handle_ptr->sync_stream();

  i_t best_c         = start_climber;
  f_t best_l1        = std::numeric_limits<f_t>::infinity();
  i_t best_int_count = -1;
  for (i_t c = start_climber; c < n_points; ++c) {
    f_t l1        = 0.;
    i_t int_count = 0;
    for (i_t k = 0; k < unified_n_int; ++k) {
      i_t col  = h_integer_indices_cache[k];
      f_t pv   = h_projected[(size_t)c * n_vars + col];
      f_t seed = h_cloud[(size_t)c * n_vars + col];
      l1 += std::abs(pv - seed);
      if (std::abs(pv - std::round(pv)) <= int_tol) int_count++;
    }
    bool better = (l1 < best_l1) || (int_count > best_int_count);
    if (better) {
      best_l1        = l1;
      best_int_count = int_count;
      best_c         = c;
    }
  }
  last_selected_l1 = best_l1;
  // Remember the best climber so the next projection can seed new climbers' dual from it.
  warm_start_best_c = best_c;
  CUOPT_LOG_INFO("Selected cloud point %d / %d (L1 %g, integer comps %d / %d)",
                 best_c,
                 n_points,
                 best_l1,
                 best_int_count,
                 unified_n_int);
  raft::copy(
    solution.assignment.data(), d_projected.data() + (size_t)best_c * n_vars, n_vars, stream);
  solution.handle_ptr->sync_stream();
  return best_c;
}

// One round of the original single-point FP applied to climber 0's projection (already loaded into
// solution.assignment): distance-cycle check, full-integer + near-feasible LP-verify, CP round,
// then the 20% FJ fallback. Mirrors run_single_fp_descent's per-iteration body and uses the shared
// FP state. Returns whether climber 0 reached feasibility; sets climber0_cycle when it cycled.
template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_climber0_step(solution_t<i_t, f_t>& solution,
                                                     f_t proj_begin,
                                                     bool& climber0_cycle,
                                                     i_t fj_traj_capacity)
{
  raft::common::nvtx::range fun_scope("run_climber0_step");
  climber0_cycle   = false;
  bool is_feasible = solution.compute_feasibility();
  i_t n_integers   = solution.compute_number_of_integers();
  // Record climber 0's projection integrality to drive the next projection's adaptive LP tolerance.
  climber0_int_ratio = (f_t)n_integers / solution.problem_ptr->n_integer_vars;

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
      climber0_cycle            = true;
      return false;
    }
  }

  if (n_integers == solution.problem_ptr->n_integer_vars) {
    if (is_feasible) {
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
      run_lp_with_vars_fixed(*solution.problem_ptr,
                             solution,
                             solution.problem_ptr->integer_indices,
                             lp_settings,
                             &constraint_prop.bounds_update);
      is_feasible = solution.get_feasible();
      n_integers  = solution.compute_number_of_integers();
      if (is_feasible && n_integers == solution.problem_ptr->n_integer_vars) { return true; }
    }
  }

  cuopt_func_call(solution.test_variable_bounds(false));
  is_feasible = round(solution);
  cuopt_func_call(solution.test_variable_bounds(true));
  proj_and_round_time = proj_begin - timer.remaining_time();
  if (!is_feasible) {
    is_feasible =
      test_fj_feasible(solution,
                       batch_config.fj_seed_time_ratio * std::max(proj_and_round_time, 0.05),
                       fj_traj_capacity);
  }
  if (timer.check_time_limit()) { return false; }
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
    total_fp_time_until_cycle = fp_fj_cycle_time_begin - timer.remaining_time();
    climber0_cycle            = true;
    return false;
  }
  cycle_queue.n_iterations_without_cycle++;
  return false;
}

// Sequential probing-cache rounding of a single projected climber (inspired by
// constraint_prop's generate_bulk_rounding_vector, but no propagation: only the precomputed
// probing cache is consulted). For each integer variable we take the nearest rounding as the base
// probe and, if the cache has an entry, pick the least-conflicting value while the cache's implied
// bounds accumulate into the h_lb/h_ub scratch -- so later variables respect earlier choices.
template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::probing_cache_sequential_round(solution_t<i_t, f_t>& solution,
                                                                  const f_t* h_projection,
                                                                  std::vector<f_t>& h_lb,
                                                                  std::vector<f_t>& h_ub,
                                                                  f_t* out_assignment)
{
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;
  auto& pcache      = constraint_prop.bounds_update.probing_cache;
  for (i_t j = 0; j < unified_n_vars; ++j) {
    h_lb[j] = h_var_lower[j];
    h_ub[j] = h_var_upper[j];
  }
  for (i_t k = 0; k < unified_n_int; ++k) {
    i_t col     = h_integer_indices_cache[k];
    f_t v       = h_projection[col];
    f_t int_lb  = std::ceil(h_var_lower[col] - int_tol);
    f_t int_ub  = std::floor(h_var_upper[col] + int_tol);
    f_t nearest = std::round(v);
    if (nearest < int_lb) nearest = int_lb;
    if (nearest > int_ub) nearest = int_ub;
    // The other candidate is the adjacent integer toward the fractional part.
    f_t other = (v >= nearest) ? nearest + 1. : nearest - 1.;
    if (other < int_lb) other = int_lb;
    if (other > int_ub) other = int_ub;
    f_t val = nearest;
    if (pcache.contains(*solution.problem_ptr, col)) {
      val = pcache.get_least_conflicting_rounding(
        *solution.problem_ptr, h_lb, h_ub, col, nearest, other, int_tol);
    }
    if (val < int_lb) val = int_lb;
    if (val > int_ub) val = int_ub;
    out_assignment[col] = val;
  }
}

template <typename i_t, typename f_t>
__global__ void batch_diversity_nearest_round_kernel(const f_t* __restrict__ projected,
                                                     f_t* __restrict__ seeds,
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
  const f_t v       = projected[(size_t)climber * (size_t)n_vars + (size_t)col];
  f_t int_lb        = ceil(var_lower[col] - int_tol);
  f_t int_ub        = floor(var_upper[col] + int_tol);
  f_t nearest       = round(v);
  if (nearest < int_lb) nearest = int_lb;
  if (nearest > int_ub) nearest = int_ub;
  seeds[(size_t)climber * (size_t)n_vars + (size_t)col] = nearest;
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

__device__ __forceinline__ uint64_t fp_lcg_step(uint64_t state)
{
  return state * 6364136223846793005ULL + 1ULL;
}

template <typename i_t, typename f_t>
__global__ void fill_cloud_rows_from_template_kernel(f_t* __restrict__ cloud,
                                                     const f_t* __restrict__ template_x,
                                                     i_t n_points,
                                                     i_t n_vars)
{
  const size_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n = (size_t)n_points * (size_t)n_vars;
  if (g >= n) return;
  cloud[g] = template_x[g % (size_t)n_vars];
}

template <typename i_t, typename f_t>
__global__ void perturb_cloud_integer_slots_kernel(f_t* __restrict__ cloud,
                                                   const f_t* __restrict__ template_x,
                                                   const f_t* __restrict__ slot0_x,
                                                   const i_t* __restrict__ integer_cols,
                                                   const f_t* __restrict__ var_lower,
                                                   const f_t* __restrict__ var_upper,
                                                   const char* __restrict__ flagged,
                                                   i_t n_points,
                                                   i_t n_vars,
                                                   i_t n_int,
                                                   f_t int_tol,
                                                   bool flagged_only,
                                                   uint64_t seed)
{
  const i_t climber = blockIdx.x;
  if (climber >= n_points) return;
  if (flagged_only) {
    if (climber == 0 || flagged == nullptr || flagged[climber] == 0) return;
  } else if (climber == 0) {
    return;
  }
  uint64_t rng = seed ^ ((uint64_t)climber * 1315423911ULL);
  for (i_t k = threadIdx.x; k < n_int; k += blockDim.x) {
    i_t col = integer_cols[k];
    f_t val = (climber == 0) ? slot0_x[col] : template_x[col];
    if (climber != 0) {
      rng = fp_lcg_step(rng);
      if ((rng >> 32) % 10U == 0U) {
        f_t int_lb = ceil(var_lower[col] - int_tol);
        f_t int_ub = floor(var_upper[col] + int_tol);
        rng        = fp_lcg_step(rng);
        if (int_lb <= int_ub) {
          f_t span = int_ub - int_lb + 1.;
          val      = int_lb + fmod((f_t)(rng >> 32), span);
        }
      }
    }
    cloud[(size_t)climber * (size_t)n_vars + (size_t)col] = val;
  }
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::seed_cloud_from_assignment_gpu(solution_t<i_t, f_t>& solution,
                                                                  i_t n_points,
                                                                  rmm::device_uvector<f_t>& d_cloud)
{
  raft::common::nvtx::range fun_scope("seed_cloud_from_assignment_gpu");
  auto stream           = solution.handle_ptr->get_stream();
  const i_t n_vars      = unified_n_vars;
  const f_t int_tol     = context.settings.tolerances.integrality_tolerance;
  const size_t cloud_sz = (size_t)n_points * (size_t)n_vars;
  if (d_cloud.size() != cloud_sz) { d_cloud.resize(cloud_sz, stream); }
  constexpr i_t TPB     = 256;
  const i_t fill_blocks = (i_t)((cloud_sz + TPB - 1) / TPB);
  fill_cloud_rows_from_template_kernel<i_t, f_t>
    <<<fill_blocks, TPB, 0, stream>>>(d_cloud.data(), solution.assignment.data(), n_points, n_vars);
  raft::copy(d_cloud.data(), last_rounding.data(), n_vars, stream);
  const i_t* d_int_cols = solution.problem_ptr->integer_indices.data();
  const f_t* d_vlb      = unified_problem->get_variable_lower_bounds().data();
  const f_t* d_vub      = unified_problem->get_variable_upper_bounds().data();
  perturb_cloud_integer_slots_kernel<i_t, f_t>
    <<<n_points, TPB, 0, stream>>>(d_cloud.data(),
                                   solution.assignment.data(),
                                   last_rounding.data(),
                                   d_int_cols,
                                   d_vlb,
                                   d_vub,
                                   nullptr,
                                   n_points,
                                   n_vars,
                                   unified_n_int,
                                   int_tol,
                                   false,
                                   (uint64_t)rng());
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
  auto h_assignment = cuopt::host_copy(solution.assignment, stream);
  auto h_cloud      = cuopt::host_copy(d_cloud, stream);
  solution.handle_ptr->sync_stream();

  cloud_point_diversity_t<i_t, f_t> div(
    h_integer_indices_cache, climber_hash_history, unified_n_int);
  div.seed_from_cloud(h_cloud.data(), n_points, n_vars, &flagged);

  std::uniform_real_distribution<double> unit(0., 1.);
  constexpr double perturb_ratio = 0.1;
  constexpr i_t max_attempts     = 64;

  auto perturb_point = [&](std::vector<f_t>& pt) {
    for (i_t k = 0; k < unified_n_int; ++k) {
      if (unit(rng) >= perturb_ratio) continue;
      i_t col = h_integer_indices_cache[k];
      f_t lb  = h_var_lower[col];
      f_t ub  = h_var_upper[col];
      if (lb == -std::numeric_limits<f_t>::infinity()) {
        pt[col] = std::floor(ub + int_tol);
      } else if (ub == std::numeric_limits<f_t>::infinity()) {
        pt[col] = std::ceil(lb - int_tol);
      } else {
        std::uniform_int_distribution<i_t> unif((i_t)std::ceil(lb - int_tol),
                                                (i_t)std::floor(ub + int_tol));
        pt[col] = unif(rng);
      }
    }
  };

  for (i_t c = 1; c < n_points; ++c) {
    if (!flagged[c]) continue;
    std::vector<f_t> pt(h_assignment.begin(), h_assignment.begin() + n_vars);
    bool placed = false;
    for (i_t attempt = 0; attempt < max_attempts; ++attempt) {
      pt = std::vector<f_t>(h_assignment.begin(), h_assignment.begin() + n_vars);
      perturb_point(pt);
      if (div.passes_diversity(pt.data())) {
        div.accept(pt.data());
        std::copy(pt.begin(), pt.end(), h_cloud.begin() + (size_t)c * n_vars);
        n_diverse++;
        placed = true;
        break;
      }
    }
    if (!placed) {
      pt = std::vector<f_t>(h_assignment.begin(), h_assignment.begin() + n_vars);
      perturb_point(pt);
      div.accept(pt.data());
      std::copy(pt.begin(), pt.end(), h_cloud.begin() + (size_t)c * n_vars);
      n_fallback++;
    }
  }
  raft::copy(d_cloud.data(), h_cloud.data(), (size_t)n_points * n_vars, stream);
  solution.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
i_t feasibility_pump_t<i_t, f_t>::reseed_flagged_from_fj_trajectory(
  solution_t<i_t, f_t>& solution,
  i_t n_points,
  rmm::device_uvector<f_t>& d_cloud,
  const std::vector<char>& flagged)
{
  raft::common::nvtx::range fun_scope("reseed_flagged_from_fj_trajectory");
  const i_t n_vars = unified_n_vars;
  const i_t traj   = fj.get_trajectory_count();
  if (traj <= 0 || n_points <= 1) return 0;

  i_t n_flagged = 0;
  for (i_t c = 1; c < n_points; ++c) {
    n_flagged += flagged[c];
  }
  if (n_flagged == 0) return 0;

  auto stream  = solution.handle_ptr->get_stream();
  auto h_traj  = cuopt::host_copy(fj.get_trajectory_buffer(), stream);
  auto h_cloud = cuopt::host_copy(d_cloud, stream);
  solution.handle_ptr->sync_stream();

  cloud_point_diversity_t<i_t, f_t> div(
    h_integer_indices_cache, climber_hash_history, unified_n_int);
  div.seed_from_cloud(h_cloud.data(), n_points, n_vars, &flagged);

  std::vector<i_t> pool;
  pool.reserve((size_t)traj);
  for (i_t t = 0; t < traj; ++t) {
    const f_t* pt = h_traj.data() + (size_t)t * n_vars;
    if (!div.passes_diversity(pt)) continue;
    pool.push_back(t);
    div.accept(pt);
  }

  i_t pool_idx = 0;
  i_t applied  = 0;
  for (i_t c = 1; c < n_points; ++c) {
    if (!flagged[c]) continue;
    if (pool_idx >= (i_t)pool.size()) break;
    i_t t = pool[(size_t)pool_idx++];
    std::copy(h_traj.begin() + (size_t)t * n_vars,
              h_traj.begin() + (size_t)(t + 1) * n_vars,
              h_cloud.begin() + (size_t)c * n_vars);
    applied++;
  }
  if (applied > 0) {
    raft::copy(d_cloud.data(), h_cloud.data(), (size_t)n_points * n_vars, stream);
    solution.handle_ptr->sync_stream();
  }
  CUOPT_LOG_INFO(
    "Climber 0 FJ trajectory reseed: %d raw, %d pool, %d applied (>= %d integer diffs apart)",
    traj,
    (i_t)pool.size(),
    applied,
    div.min_int_diff);
  return applied;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::advance_diversity_climbers_gpu(
  solution_t<i_t, f_t>& solution,
  i_t n_points,
  const rmm::device_uvector<f_t>& d_projected,
  rmm::device_uvector<f_t>& d_seeds,
  std::vector<char>& flagged,
  std::vector<size_t>& climber_hashes)
{
  raft::common::nvtx::range fun_scope("advance_diversity_climbers_gpu");
  if (n_points <= 1) {
    flagged.assign((size_t)n_points, 0);
    climber_hashes.assign((size_t)n_points, 0);
    return;
  }
  auto stream           = solution.handle_ptr->get_stream();
  const i_t n_vars      = unified_n_vars;
  const f_t int_tol     = context.settings.tolerances.integrality_tolerance;
  const size_t cloud_sz = (size_t)n_points * (size_t)n_vars;
  if (d_seeds.size() != cloud_sz) { d_seeds.resize(cloud_sz, stream); }
  raft::copy(d_seeds.data(), d_projected.data(), cloud_sz, stream);

  const i_t* d_int_cols = solution.problem_ptr->integer_indices.data();
  const f_t* d_vlb      = unified_problem->get_variable_lower_bounds().data();
  const f_t* d_vub      = unified_problem->get_variable_upper_bounds().data();
  constexpr i_t TPB     = 256;
  const i_t n_work      = (n_points - 1) * unified_n_int;
  const i_t n_blocks    = (n_work + TPB - 1) / TPB;
  batch_diversity_nearest_round_kernel<i_t, f_t><<<n_blocks, TPB, 0, stream>>>(d_projected.data(),
                                                                               d_seeds.data(),
                                                                               d_int_cols,
                                                                               d_vlb,
                                                                               d_vub,
                                                                               n_points,
                                                                               n_vars,
                                                                               unified_n_int,
                                                                               int_tol);

  rmm::device_uvector<size_t> d_hashes((size_t)n_points, stream);
  batch_climber_int_hash_kernel<i_t, f_t><<<n_points - 1, 1, 0, stream>>>(
    d_seeds.data(), d_int_cols, d_hashes.data(), n_points, n_vars, unified_n_int);
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
  static const bool use_single_fp    = std::getenv("CUOPT_FP_SINGLE") != nullptr;
  static const bool profile_one_iter = std::getenv("CUOPT_FP_PROFILE_ONE_ITER") != nullptr;
  if (use_single_fp) { return run_single_fp_descent(solution); }

  if (unified_problem == nullptr || unified_n_vars != solution.problem_ptr->n_variables ||
      unified_n_constr != solution.problem_ptr->n_constraints) {
    build_unified_projection_problem(solution);
  }

  const i_t batch_size = compute_cloud_batch_size(solution);
  if (batch_size == 0) { return run_single_fp_descent(solution); }
  expand_unified_projection_batch_buffers(solution, batch_size);

  auto stream         = solution.handle_ptr->get_stream();
  const i_t n_vars    = unified_n_vars;
  reseed_count        = 0;   // persistent trajectories — no assemble_cloud reseed step
  warm_start_n_points = 0;   // cold dual at descent start
  climber0_int_ratio  = 0.;  // start loose; tightens as climber 0 converges (like the single FP)
  config.alpha        = default_alpha;
  // Climber 0 begins from the nearest rounding (classic FP start).
  solution.round_nearest();
  raft::copy(last_rounding.data(), solution.assignment.data(), solution.assignment.size(), stream);

  rmm::device_uvector<f_t> d_cloud(0, stream);
  rmm::device_uvector<f_t> d_projected((size_t)batch_size * n_vars, stream);
  std::vector<char> flagged((size_t)batch_size, 0);
  std::vector<size_t> climber_hashes((size_t)batch_size, 0);
  const f_t inf = std::numeric_limits<f_t>::infinity();
  rmm::device_uvector<f_t> cand_best(n_vars, stream);  // best feasible found this trajectory
  rmm::device_uvector<f_t> climber0_assign(n_vars,
                                           stream);  // climber 0 base for reseed perturbation
  i_t n_points         = 0;
  bool first_iteration = true;
  i_t trajectory       = 0;

  CUOPT_LOG_INFO("Starting batched FP cloud: batch size %d, integers %d, vars %d, constraints %d",
                 batch_size,
                 unified_n_int,
                 unified_n_vars,
                 unified_n_constr);

  using timing_clock = std::chrono::steady_clock;
  auto elapsed_since = [](timing_clock::time_point t0) {
    return std::chrono::duration<double>(timing_clock::now() - t0).count();
  };

  while (true) {
    if (context.diversity_manager_ptr->check_b_b_preemption() || timer.check_time_limit()) {
      CUOPT_LOG_INFO("Batched FP time limit reached after %d trajectories", trajectory);
      round(solution);
      return false;
    }
    proj_begin = timer.remaining_time();
    const cuda_profiler_scope_t profiler_scope(profile_one_iter && trajectory == 0);

    // ---- Build (first round) or refresh the working cloud ----
    if (first_iteration) {
      auto t_reseed = timing_clock::now();
      n_points      = batch_size;
      seed_cloud_from_assignment_gpu(solution, n_points, d_cloud);
      CUOPT_LOG_INFO("Batched FP trajectory %d timing: reseed (initial cloud) %.3fs",
                     trajectory,
                     elapsed_since(t_reseed));
      if (n_points == 0) {
        CUOPT_LOG_INFO("Empty cloud; falling back to single FP");
        return run_single_fp_descent(solution);
      }
      climber_alphas.assign(n_points, default_alpha);
      climber_alphas[0] = config.alpha;
      climber_hash_history.assign(n_points, {});
      first_iteration = false;
    } else {
      // Slots 1..N persist their own trajectory in d_cloud; only slot 0 (climber 0) is refreshed.
      raft::copy(d_cloud.data(), last_rounding.data(), n_vars, stream);
      solution.handle_ptr->sync_stream();
    }
    // Persistent trajectories: slot c is the same climber across rounds, so its own previous dual
    // is the warm start (per-climber carry-over).
    n_carried_over_points = n_points;

    CUOPT_LOG_INFO(
      "Batched FP trajectory %d: cloud %d points (slot 0 = climber 0)", trajectory, n_points);

    auto t_proj = timing_clock::now();
    project_cloud(solution, n_points, d_cloud, d_projected);
    CUOPT_LOG_INFO(
      "Batched FP trajectory %d timing: projection %.3fs", trajectory, elapsed_since(t_proj));
    if (n_points == 0) {
      CUOPT_LOG_INFO(
        "Batched FP trajectory %d: projection produced no usable solution; single-round fallback",
        trajectory);
      bool is_feasible = round(solution);
      if (is_feasible && solution.compute_feasibility()) {
        CUOPT_LOG_INFO("New feasible solution via rounding fallback (objective %g)",
                       solution.get_user_objective());
        return true;
      }
      return false;
    }
    if ((i_t)climber_alphas.size() > n_points) { climber_alphas.resize(n_points); }
    if ((i_t)climber_hash_history.size() > n_points) { climber_hash_history.resize(n_points); }
    n_carried_over_points = std::min(n_carried_over_points, n_points);

    // ---- Climber 0 (slot 0): full classic FP with internal restarts (the only CP round) ----
    auto t_climber0 = timing_clock::now();
    raft::copy(solution.assignment.data(), d_projected.data(), n_vars, stream);
    // Batch PDLP's loose tolerance can leave the primal marginally outside variable bounds; clamp.
    solution.clamp_within_bounds();
    raft::copy(
      last_projection.data(), solution.assignment.data(), solution.assignment.size(), stream);
    bool climber0_cycle = false;
    bool feasible0      = run_climber0_step(solution, proj_begin, climber0_cycle, 10 * n_points);
    CUOPT_LOG_INFO("Batched FP trajectory %d timing: climber0 round %.3fs",
                   trajectory,
                   elapsed_since(t_climber0));
    // Best feasible found this trajectory across climber 0 and the diversity climbers.
    f_t best_obj       = inf;
    bool have_feasible = false;
    if (feasible0) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      best_obj      = solution.get_objective();
      have_feasible = true;
      raft::copy(cand_best.data(), solution.assignment.data(), n_vars, stream);
      solution.handle_ptr->sync_stream();
    }
    // The diversity feasibility check below overwrites solution.assignment, but the reseed step
    // perturbs from climber 0's assignment; stash it so we can restore it afterwards.
    raft::copy(climber0_assign.data(), solution.assignment.data(), n_vars, stream);
    solution.handle_ptr->sync_stream();

    // ---- Diversity climbers 1..N: GPU nearest-integer rounding + hash dedup ----
    auto t_diversity = timing_clock::now();
    advance_diversity_climbers_gpu(
      solution, n_points, d_projected, d_cloud, flagged, climber_hashes);
    CUOPT_LOG_INFO("Batched FP trajectory %d timing: diversity round %.3fs",
                   trajectory,
                   elapsed_since(t_diversity));

    // ---- Device-confirm host-feasible diversity climbers; keep the best objective ----
    {
      auto h_cloud = cuopt::host_copy(d_cloud, stream);
      solution.handle_ptr->sync_stream();
      i_t n_feas_div = 0;
      for (i_t c = 1; c < n_points; ++c) {
        const f_t* seed_c = h_cloud.data() + (size_t)c * n_vars;
        if (!host_assignment_feasible(seed_c)) continue;
        raft::copy(solution.assignment.data(), d_cloud.data() + (size_t)c * n_vars, n_vars, stream);
        solution.handle_ptr->sync_stream();
        if (solution.compute_feasibility()) {
          n_feas_div++;
          f_t obj = solution.get_objective();
          if (obj < best_obj) {
            best_obj      = obj;
            have_feasible = true;
            raft::copy(cand_best.data(), solution.assignment.data(), n_vars, stream);
            solution.handle_ptr->sync_stream();
          }
        }
      }
      if (n_feas_div > 0) {
        CUOPT_LOG_INFO(
          "Batched FP trajectory %d: %d diversity climbers feasible", trajectory, n_feas_div);
      }
    }

    // Restore climber 0's assignment as the reseed perturbation base.
    raft::copy(solution.assignment.data(), climber0_assign.data(), n_vars, stream);
    solution.handle_ptr->sync_stream();

    // ---- Return the best feasible found (climber 0 or a diversity climber) ----
    if (have_feasible) {
      raft::copy(solution.assignment.data(), cand_best.data(), n_vars, stream);
      solution.handle_ptr->sync_stream();
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      CUOPT_LOG_INFO("New feasible solution at trajectory %d (objective %g)",
                     trajectory,
                     solution.get_user_objective());
      return true;
    }

    i_t n_flagged = 0;
    for (i_t c = 1; c < n_points; ++c) {
      n_flagged += flagged[c];
    }
    CUOPT_LOG_INFO("Batched FP trajectory %d: climber0 feasible %d; diversity flagged %d / %d",
                   trajectory,
                   (int)feasible0,
                   n_flagged,
                   n_points - 1);

    if (timer.check_time_limit()) {
      CUOPT_LOG_INFO("Batched FP time limit reached after %d trajectories", trajectory);
      return false;
    }

    // ---- Replace cycled / duplicate diversity climbers with cheap padding-only seeds ----
    if (n_flagged > 0) {
      auto t_reseed   = timing_clock::now();
      i_t n_fj_reseed = reseed_flagged_from_fj_trajectory(solution, n_points, d_cloud, flagged);
      std::vector<char> still_flagged = flagged;
      i_t n_diverse_perturb           = 0;
      i_t n_fallback_perturb          = 0;
      if (n_fj_reseed > 0) {
        i_t cleared = 0;
        for (i_t c = 1; c < n_points && cleared < n_fj_reseed; ++c) {
          if (!flagged[c]) continue;
          still_flagged[c]  = 0;
          climber_alphas[c] = default_alpha;
          climber_hash_history[c].clear();
          cleared++;
        }
      }
      replace_flagged_climbers_diverse(
        solution, n_points, d_cloud, still_flagged, n_diverse_perturb, n_fallback_perturb);
      CUOPT_LOG_INFO(
        "Reseed breakdown trajectory %d: %d FJ, %d diverse perturb, %d fallback perturb (of %d "
        "flagged)",
        trajectory,
        n_fj_reseed,
        n_diverse_perturb,
        n_fallback_perturb,
        n_flagged);
      for (i_t c = 1; c < n_points; ++c) {
        if (!still_flagged[c]) continue;
        climber_alphas[c] = default_alpha;
        climber_hash_history[c].clear();
      }
      CUOPT_LOG_INFO("Batched FP trajectory %d timing: reseed (replacement pool) %.3fs",
                     trajectory,
                     elapsed_since(t_reseed));
    }

    // ---- Commit the new persistent cloud (slot 0 = climber 0; 1..N = advanced or replaced) ----
    raft::copy(d_cloud.data(), last_rounding.data(), n_vars, stream);
    solution.handle_ptr->sync_stream();
    for (i_t c = 1; c < n_points; ++c) {
      if (flagged[c]) continue;
      auto& hist = climber_hash_history[c];
      hist.push_back(climber_hashes[c]);
      while ((i_t)hist.size() > cycle_queue.cycle_detection_length) {
        hist.pop_front();
      }
    }

    // ---- Only climber 0 restarts (self-contained, like the original FP) ----
    if (climber0_cycle) {
      CUOPT_LOG_INFO("Climber 0 cycle at trajectory %d; restarting (FJ escape + perturbate)",
                     trajectory);
      auto t_restart = timing_clock::now();
      restart_fp(solution);
      // Re-anchor climber 0 from the perturbed point for the next round's slot 0.
      solution.round_nearest();
      raft::copy(
        last_rounding.data(), solution.assignment.data(), solution.assignment.size(), stream);
      CUOPT_LOG_INFO(
        "Batched FP trajectory %d timing: restart %.3fs", trajectory, elapsed_since(t_restart));
    }
    if (profile_one_iter && trajectory == 0) {
      CUOPT_LOG_INFO("Batched FP profile-one-iter: exiting after trajectory 0");
      return false;
    }
    trajectory++;
  }
  return false;
}

// Explicit instantiation of the batched-PDLP members. The non-batched members are instantiated by
// the `template class feasibility_pump_t<...>` in feasibility_pump.cu; these definitions live in a
// separate translation unit, so the specific members are instantiated here instead.
#define INSTANTIATE_BATCHED(F_TYPE)                                                                \
  template void feasibility_pump_t<int, F_TYPE>::build_unified_projection_problem(                 \
    solution_t<int, F_TYPE>&);                                                                     \
  template void feasibility_pump_t<int, F_TYPE>::expand_unified_projection_batch_buffers(          \
    solution_t<int, F_TYPE>&, int);                                                                \
  template int feasibility_pump_t<int, F_TYPE>::compute_cloud_batch_size(                          \
    solution_t<int, F_TYPE>&);                                                                     \
  template int feasibility_pump_t<int, F_TYPE>::assemble_cloud(                                    \
    solution_t<int, F_TYPE>&, int, bool, rmm::device_uvector<F_TYPE>&, bool&);                     \
  template void feasibility_pump_t<int, F_TYPE>::seed_cloud_from_assignment_gpu(                   \
    solution_t<int, F_TYPE>&, int, rmm::device_uvector<F_TYPE>&);                                  \
  template void feasibility_pump_t<int, F_TYPE>::replace_flagged_climbers_diverse(                 \
    solution_t<int, F_TYPE>&,                                                                      \
    int,                                                                                           \
    rmm::device_uvector<F_TYPE>&,                                                                  \
    const std::vector<char>&,                                                                      \
    int&,                                                                                          \
    int&);                                                                                         \
  template int feasibility_pump_t<int, F_TYPE>::reseed_flagged_from_fj_trajectory(                 \
    solution_t<int, F_TYPE>&, int, rmm::device_uvector<F_TYPE>&, const std::vector<char>&);        \
  template void feasibility_pump_t<int, F_TYPE>::advance_diversity_climbers_gpu(                   \
    solution_t<int, F_TYPE>&,                                                                      \
    int,                                                                                           \
    const rmm::device_uvector<F_TYPE>&,                                                            \
    rmm::device_uvector<F_TYPE>&,                                                                  \
    std::vector<char>&,                                                                            \
    std::vector<size_t>&);                                                                         \
  template void feasibility_pump_t<int, F_TYPE>::project_cloud(solution_t<int, F_TYPE>&,           \
                                                               int&,                               \
                                                               const rmm::device_uvector<F_TYPE>&, \
                                                               rmm::device_uvector<F_TYPE>&);      \
  template int feasibility_pump_t<int, F_TYPE>::select_cloud_point(                                \
    solution_t<int, F_TYPE>&,                                                                      \
    int,                                                                                           \
    const rmm::device_uvector<F_TYPE>&,                                                            \
    const rmm::device_uvector<F_TYPE>&,                                                            \
    int);                                                                                          \
  template bool feasibility_pump_t<int, F_TYPE>::run_climber0_step(                                \
    solution_t<int, F_TYPE>&, F_TYPE, bool&, int);                                                 \
  template void feasibility_pump_t<int, F_TYPE>::probing_cache_sequential_round(                   \
    solution_t<int, F_TYPE>&, const F_TYPE*, std::vector<F_TYPE>&, std::vector<F_TYPE>&, F_TYPE*); \
  template bool feasibility_pump_t<int, F_TYPE>::host_assignment_feasible(const F_TYPE*);          \
  template bool feasibility_pump_t<int, F_TYPE>::run_batched_fp_cloud(solution_t<int, F_TYPE>&);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE_BATCHED(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE_BATCHED(double)
#endif

#undef INSTANTIATE_BATCHED

}  // namespace cuopt::mathematical_optimization::mip
