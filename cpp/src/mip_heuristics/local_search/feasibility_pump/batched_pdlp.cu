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

  unified_problem = std::make_unique<cuopt::linear_programming::optimization_problem_t<i_t, f_t>>(
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
  // Per-climber constraint bounds (the +/- val_j rows) drive the dominant memory term; the
  // objective is shared across climbers.
  const size_t mem_cap = cuopt::linear_programming::compute_optimal_batch_size(
    *unified_problem, /*per_climber_objectives=*/false, /*per_climber_constraint_bounds=*/true);
  if (mem_cap < (size_t)batch_config.fallback_threshold) {
    CUOPT_LOG_INFO("Cloud batch size memory cap %zu below fallback threshold %d; using single FP",
                   mem_cap,
                   batch_config.fallback_threshold);
    return 0;
  }
  i_t bs = (i_t)std::min<size_t>(mem_cap, (size_t)batch_config.target_max_batch_size);
  CUOPT_LOG_INFO("Cloud batch size %d (memory cap %zu, target [%d, %d])",
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
    // Record up to a full cloud worth of candidates (not just the remaining slots): many will share
    // integer values and get filtered below, so oversampling lets FJ still contribute enough
    // integer-distinct points.
    fj.set_trajectory_capacity(batch_size, stream);
    // The seeding 20% FJ both records its visited trajectory (cloud seeds) AND may itself reach a
    // feasible solution; if it does, signal the caller to exit instead of projecting a cloud.
    bool fj_feasible              = fj.solve(solution);
    fj.settings.record_trajectory = false;
    if (fj_feasible && solution.compute_feasibility()) { seed_found_feasible = true; }
    i_t traj = fj.get_trajectory_count();
    if (traj > 0) {
      auto h_traj = cuopt::host_copy(fj.get_trajectory_buffer(), stream);
      solution.handle_ptr->sync_stream();
      // Keep the cloud integer-diverse. The FJ incumbent often only moves its continuous part
      // between sync points, so consecutive iterates differ in just a handful of integer
      // components; projecting those near-duplicates wastes cloud slots. Instead of a strict
      // one-to-one match, require a candidate to differ from every already-added point (including
      // the reseed points above) in at least max(1% of the integers, 2) integer components.
      const i_t min_int_diff =
        std::min(std::max(unified_n_int, 1), std::max(2, (i_t)std::ceil(0.01 * unified_n_int)));
      std::vector<long long> accepted_int_vals;  // flat [n_accepted * unified_n_int]
      accepted_int_vals.reserve((size_t)batch_size * unified_n_int);
      std::vector<long long> cand(unified_n_int);
      auto extract_int = [&](const f_t* pt) {
        for (i_t k = 0; k < unified_n_int; ++k) {
          cand[k] = std::llround(pt[h_integer_indices_cache[k]]);
        }
      };
      auto too_close = [&]() {
        i_t n_accepted = (i_t)(accepted_int_vals.size() / std::max(unified_n_int, 1));
        for (i_t a = 0; a < n_accepted; ++a) {
          const long long* base = accepted_int_vals.data() + (size_t)a * unified_n_int;
          i_t diff              = 0;
          for (i_t k = 0; k < unified_n_int && diff < min_int_diff; ++k) {
            if (cand[k] != base[k]) ++diff;
          }
          if (diff < min_int_diff) return true;
        }
        return false;
      };
      for (i_t c = 0; c < count; ++c) {
        extract_int(h_points.data() + (size_t)c * n_vars);
        accepted_int_vals.insert(accepted_int_vals.end(), cand.begin(), cand.end());
      }
      i_t added_unique = 0;
      for (i_t t = 0; t < traj && count < batch_size; ++t) {
        const f_t* pt = h_traj.data() + (size_t)t * n_vars;
        extract_int(pt);
        if (too_close()) continue;
        std::copy(pt, pt + n_vars, h_points.begin() + (size_t)count * n_vars);
        accepted_int_vals.insert(accepted_int_vals.end(), cand.begin(), cand.end());
        count++;
        added_unique++;
      }
      CUOPT_LOG_INFO("Cloud seeding: %d FJ trajectory points, %d added (>= %d integer diffs apart)",
                     traj,
                     added_unique,
                     min_int_diff);
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
      i_t col   = h_integer_indices_cache[k];
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
  thrust::for_each(solution.handle_ptr->get_thrust_policy(),
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
  settings.method                              = cuopt::linear_programming::method_t::PDLP;
  settings.presolver                           = presolver_t::None;
  settings.fixed_batch_size                    = n_points;
  settings.generate_batch_primal_dual_solution = true;
  const f_t rlp_base  = context.settings.heuristic_params.relaxed_lp_time_limit;
  settings.time_limit = std::max(0.05, std::min((double)rlp_base, timer.remaining_time() / 10.));
  settings.set_optimality_tolerance(1e-4);

  // Per-climber warm start: seed each climber from its own primal/dual of the previous projection.
  // Only valid when the climber count is unchanged (the unified problem is fixed across iterations,
  // so the per-climber sizes match). The unified problem's n_variables == unified_n_vars_total and
  // n_constraints == unified_n_constr_total, so the stored full primal/dual map 1:1 onto the batch.
  const bool warm_start_available =
    warm_start_n_points == n_points &&
    warm_start_primal.size() == (size_t)n_points * unified_n_vars_total &&
    warm_start_dual.size() == (size_t)n_points * nct;
  if (warm_start_available) {
    settings.set_initial_primal_solution(
      warm_start_primal.data(), (i_t)warm_start_primal.size(), stream);
    settings.set_initial_dual_solution(warm_start_dual.data(), (i_t)warm_start_dual.size(), stream);
    CUOPT_LOG_INFO(
      "Batch projection warm-started per-climber from previous projection (%d climbers)", n_points);
  }

  auto sol     = cuopt::linear_programming::run_batch_pdlp(op, settings);
  auto& primal = sol.get_primal_solution();
  if (primal.size() != (size_t)n_points * unified_n_vars_total) {
    CUOPT_LOG_DEBUG("Batch projection produced no usable primal (size %zu, expected %zu)",
                    primal.size(),
                    (size_t)n_points * unified_n_vars_total);
    warm_start_n_points = 0;
    return false;
  }
  d_projected.resize((size_t)n_points * n_vars, stream);
  for (i_t c = 0; c < n_points; ++c) {
    raft::copy(d_projected.data() + (size_t)c * n_vars,
               primal.data() + (size_t)c * unified_n_vars_total,
               n_vars,
               stream);
  }

  // Persist the full per-climber primal/dual to warm start the next projection in this descent.
  auto& dual = sol.get_dual_solution();
  if (dual.size() == (size_t)n_points * nct) {
    warm_start_primal.resize(primal.size(), stream);
    raft::copy(warm_start_primal.data(), primal.data(), primal.size(), stream);
    warm_start_dual.resize(dual.size(), stream);
    raft::copy(warm_start_dual.data(), dual.data(), dual.size(), stream);
    warm_start_n_points = n_points;
  } else {
    warm_start_n_points = 0;
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

  i_t best_c         = 0;
  f_t best_l1        = std::numeric_limits<f_t>::infinity();
  i_t best_int_count = -1;
  for (i_t c = 0; c < n_points; ++c) {
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

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_batched_fp_cloud(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("run_batched_fp_cloud");
  // Batch PDLP requires double precision; fall back to the single-point pump otherwise.
  if constexpr (!std::is_same_v<f_t, double>) {
    CUOPT_LOG_INFO("Batched FP cloud requires double precision; falling back to single FP");
    return run_single_fp_descent(solution);
  }

  if (unified_problem == nullptr || unified_n_vars != solution.problem_ptr->n_variables ||
      unified_n_constr != solution.problem_ptr->n_constraints) {
    build_unified_projection_problem(solution);
  }

  const i_t batch_size = compute_cloud_batch_size(solution);
  if (batch_size == 0) { return run_single_fp_descent(solution); }

  reseed_count = 0;
  // Each descent starts cold; warm starts only chain across iterations within this descent.
  warm_start_n_points = 0;
  solution.round_nearest();
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());

  rmm::device_uvector<f_t> d_cloud(0, solution.handle_ptr->get_stream());
  rmm::device_uvector<f_t> d_projected(0, solution.handle_ptr->get_stream());
  bool first_iteration = true;
  i_t trajectory       = 0;

  CUOPT_LOG_INFO("Starting batched FP cloud: batch size %d, integers %d, vars %d, constraints %d",
                 batch_size,
                 unified_n_int,
                 unified_n_vars,
                 unified_n_constr);

  while (true) {
    if (context.diversity_manager_ptr->check_b_b_preemption() || timer.check_time_limit()) {
      CUOPT_LOG_INFO("Batched FP time limit reached after %d trajectories", trajectory);
      round(solution);
      return false;
    }
    proj_begin = timer.remaining_time();

    bool seed_found_feasible = false;
    i_t n_points =
      assemble_cloud(solution, batch_size, first_iteration, d_cloud, seed_found_feasible);
    first_iteration = false;
    CUOPT_LOG_INFO("Batched FP trajectory %d: assembled cloud of %d points", trajectory, n_points);
    // The 20% FJ that seeds the cloud may already land on a feasible solution: take it and exit.
    if (seed_found_feasible) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      CUOPT_LOG_INFO(
        "New feasible solution: seeding 20%% FJ found feasible at trajectory %d (objective %g)",
        trajectory,
        solution.get_user_objective());
      return res;
    }
    if (n_points == 0) {
      CUOPT_LOG_INFO("Empty cloud at trajectory %d; falling back to single FP", trajectory);
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
      auto stream = solution.handle_ptr->get_stream();
      auto h_proj = cuopt::host_copy(d_projected, stream);
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
    CUOPT_LOG_INFO(
      "Batched FP trajectory %d: projection L1 distance %g, integers %d / %d, %d cloud points",
      trajectory,
      last_selected_l1,
      n_integers,
      solution.problem_ptr->n_integer_vars,
      n_points);

    bool is_cycle = true;
    if (config.check_distance_cycle) {
      is_cycle = check_distance_cycle(solution);
      if (is_cycle) {
        is_feasible = round(solution);
        cuopt_func_call(solution.test_variable_bounds(true));
        if (is_feasible) {
          bool res = solution.compute_feasibility();
          cuopt_assert(res, "Feasibility issue");
          CUOPT_LOG_INFO(
            "New feasible solution: distance-cycle round at trajectory %d (objective %g)",
            trajectory,
            solution.get_user_objective());
          return true;
        }
        cuopt::default_logger().flush();
        total_fp_time_until_cycle = fp_fj_cycle_time_begin - timer.remaining_time();
        return false;
      }
    }

    if (n_integers == solution.problem_ptr->n_integer_vars) {
      if (is_feasible) {
        CUOPT_LOG_INFO(
          "New feasible solution found after batched projection at trajectory %d (objective %g)",
          trajectory,
          solution.get_user_objective());
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
          CUOPT_LOG_INFO(
            "New feasible solution verified with LP after batched projection at trajectory %d "
            "(objective %g)",
            trajectory,
            solution.get_user_objective());
          return true;
        }
      }
    }

    cuopt_func_call(solution.test_variable_bounds(false));
    is_feasible = round(solution);
    cuopt_func_call(solution.test_variable_bounds(true));
    proj_and_round_time = proj_begin - timer.remaining_time();
    if (!is_feasible) {
      // Record this 20% FJ's visited iterates too, so its work is reused as extra cloud seeds for
      // the next iteration instead of being discarded.
      fj.settings.record_trajectory = true;
      fj.set_trajectory_capacity(batch_size, solution.handle_ptr->get_stream());
      is_feasible =
        test_fj_feasible(solution, batch_config.fj_seed_time_ratio * proj_and_round_time);
      fj.settings.record_trajectory = false;
      if (!is_feasible) {
        i_t traj = fj.get_trajectory_count();
        if (traj > 0) {
          auto stream = solution.handle_ptr->get_stream();
          auto h_traj = cuopt::host_copy(fj.get_trajectory_buffer(), stream);
          auto h_cur  = cuopt::host_copy(reseed_points, stream);
          solution.handle_ptr->sync_stream();
          h_cur.insert(h_cur.end(), h_traj.begin(), h_traj.begin() + (size_t)traj * unified_n_vars);
          reseed_points.resize(h_cur.size(), stream);
          raft::copy(reseed_points.data(), h_cur.data(), h_cur.size(), stream);
          solution.handle_ptr->sync_stream();
          reseed_count += traj;
        }
      }
    }
    if (timer.check_time_limit()) {
      CUOPT_LOG_INFO("Batched FP time limit reached after %d trajectories", trajectory);
      return false;
    }
    if (is_feasible) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      CUOPT_LOG_INFO("New feasible solution found after round/FJ at trajectory %d (objective %g)",
                     trajectory,
                     solution.get_user_objective());
      return true;
    }

    f_t alpha_at_earlier_iter = config.alpha / config.alpha_decrease_factor;
    if (alpha_at_earlier_iter - config.alpha < 0.005) {
      is_cycle = cycle_queue.check_cycle(solution);
    }
    cycle_queue.update_recent_solutions(solution);
    if (is_cycle) {
      CUOPT_LOG_INFO("Batched FP cycle encountered at trajectory %d", trajectory);
      total_fp_time_until_cycle = fp_fj_cycle_time_begin - timer.remaining_time();
      return false;
    }
    cycle_queue.n_iterations_without_cycle++;
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
  template int feasibility_pump_t<int, F_TYPE>::compute_cloud_batch_size(                          \
    solution_t<int, F_TYPE>&);                                                                     \
  template int feasibility_pump_t<int, F_TYPE>::assemble_cloud(                                    \
    solution_t<int, F_TYPE>&, int, bool, rmm::device_uvector<F_TYPE>&, bool&);                     \
  template bool feasibility_pump_t<int, F_TYPE>::project_cloud(solution_t<int, F_TYPE>&,           \
                                                               int,                                \
                                                               const rmm::device_uvector<F_TYPE>&, \
                                                               rmm::device_uvector<F_TYPE>&);      \
  template int feasibility_pump_t<int, F_TYPE>::select_cloud_point(                                \
    solution_t<int, F_TYPE>&,                                                                      \
    int,                                                                                           \
    const rmm::device_uvector<F_TYPE>&,                                                            \
    const rmm::device_uvector<F_TYPE>&);                                                           \
  template bool feasibility_pump_t<int, F_TYPE>::run_batched_fp_cloud(solution_t<int, F_TYPE>&);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE_BATCHED(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE_BATCHED(double)
#endif

#undef INSTANTIATE_BATCHED

}  // namespace cuopt::linear_programming::detail
