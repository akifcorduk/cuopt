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

#include <cmath>
#include <unordered_map>

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

// Defined in feasibility_pump.cu: maps the fraction of integral integer-vars to an LP tolerance
// (looser when far from integral, tighter as it converges).
double get_tolerance_from_ratio(double ratio_integer, double absolute_tol);

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
  // objective is shared across climbers. collect_solutions=true budgets the per-climber
  // primal/dual/reduced-cost outputs we request via generate_batch_primal_dual_solution (otherwise
  // they are unaccounted and the cloud OOMs on large problems).
  const size_t mem_cap = cuopt::mathematical_optimization::compute_optimal_batch_size(
    *unified_problem,
    /*per_climber_objectives=*/false,
    /*per_climber_constraint_bounds=*/true,
    /*collect_solutions=*/true);
  // PDLP's estimate still ignores our warm-start primal/dual buffers and the concurrent B&B
  // allocations, and it sizes against a single free-memory snapshot; use only a fraction of it.
  const size_t conservative_cap =
    (size_t)std::floor(batch_config.memory_headroom_fraction * (double)mem_cap);
  if (conservative_cap < (size_t)batch_config.fallback_threshold) {
    CUOPT_LOG_INFO(
      "Cloud batch size cap %zu (conservative, mem-fit %zu) below fallback threshold %d; using "
      "single FP",
      conservative_cap,
      mem_cap,
      batch_config.fallback_threshold);
    return 0;
  }
  i_t bs = (i_t)std::min<size_t>(conservative_cap, (size_t)batch_config.target_max_batch_size);
  CUOPT_LOG_INFO("Cloud batch size %d (mem-fit cap %zu, conservative cap %zu, target [%d, %d])",
                 bs,
                 mem_cap,
                 conservative_cap,
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
    raft::random::PCGenerator pcg(cuopt::seed_generator::get_seed(), 0, 0);
    for (i_t k = 0; k < unified_n_int; ++k) {
      i_t col   = h_integer_indices_cache[k];
      base[col] = round_nearest(h_lp_opt[col], h_var_lower[col], h_var_upper[col], int_tol, pcg);
    }
    std::uniform_real_distribution<double> unit(0., 1.);
    constexpr double perturb_ratio = 0.1;
    while (count < batch_size) {
      std::vector<f_t> pt = base;
      for (i_t k = 0; k < unified_n_int; ++k) {
        if (unit(rng) >= perturb_ratio) continue;
        i_t col = h_integer_indices_cache[k];
        f_t lb  = h_var_lower[col];
        f_t ub  = h_var_upper[col];
        // Same uniform-within-bounds integer sampling as solution_t::assign_random_within_bounds.
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
  }

  if (count == 0) return 0;
  // Reserve cloud slot 0 for climber 0: its dedicated classic-FP trajectory point (last_rounding).
  // Climber 0 is always slot 0 across iterations, so its warm start and dual carry over cleanly.
  {
    auto h_last_rounding = cuopt::host_copy(last_rounding, stream);
    solution.handle_ptr->sync_stream();
    std::copy(h_last_rounding.begin(), h_last_rounding.begin() + n_vars, h_points.begin());
  }
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
  settings.method                              = cuopt::mathematical_optimization::method_t::PDLP;
  settings.presolver                           = presolver_t::None;
  settings.fixed_batch_size                    = n_points;
  settings.generate_batch_primal_dual_solution = true;
  // Projection base time: each batch-PDLP projection gets a 1s base budget (still capped so a
  // single projection cannot consume more than a tenth of the remaining time).
  const f_t rlp_base  = 1.0;
  settings.time_limit = std::max(0.05, std::min((double)rlp_base, timer.remaining_time() / 10.));
  // Integer-ratio-adaptive LP tolerance, driven by climber 0's integrality (same schedule as the
  // single-point FP): loose while climber 0 is far from integral, tightening as it converges. The
  // whole cloud rides on climber 0's ratio, so the other climbers get the adaptivity for free.
  const double lp_tolerance =
    get_tolerance_from_ratio(climber0_int_ratio, context.settings.tolerances.absolute_tolerance);
  settings.set_optimality_tolerance(lp_tolerance);

  // Primal warm start: seed every climber from the current point it is projecting (the integer
  // cloud seed x, with the aux distance vars d = 0 -- exactly feasible for the |x - val| rows since
  // val == the seed). Built fresh from the current cloud, so it applies on every projection,
  // including the first of the descent. Lives until after the solve (run_batch_pdlp copies it into
  // the settings).
  rmm::device_uvector<f_t> primal_init((size_t)n_points * unified_n_vars_total, stream);
  {
    const i_t nvt       = unified_n_vars_total;
    const i_t nv        = n_vars;
    const f_t* cloud_pp = d_cloud.data();
    const f_t* vlb_ptr  = op.get_variable_lower_bounds().data();
    const f_t* vub_ptr  = op.get_variable_upper_bounds().data();
    f_t* pinit          = primal_init.data();
    // The warm start must be finite and within the variable bounds: an out-of-bounds or non-finite
    // primal seed makes PDHG diverge into a NaN primal weight. Mirrors relaxed_lp's
    // clamp_within_var_bounds before set_initial_primal_solution.
    thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                     thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>((size_t)n_points * nvt),
                     [=] __device__(size_t g) {
                       const size_t c = g / (size_t)nvt;
                       const i_t loc  = (i_t)(g % (size_t)nvt);
                       f_t v = (loc < nv) ? cloud_pp[c * (size_t)nv + (size_t)loc] : f_t(0);
                       if (!isfinite(v)) v = f_t(0);
                       const f_t lo = vlb_ptr[loc];
                       const f_t hi = vub_ptr[loc];
                       pinit[g]     = (v < lo) ? lo : ((v > hi) ? hi : v);
                     });
  }
  settings.set_initial_primal_solution(primal_init.data(), (i_t)primal_init.size(), stream);

  // Dual warm start (skipped on the first projection of a descent): climbers carried over from the
  // previous cloud reuse their own previous-iteration dual; every new climber reuses the previous
  // iteration's best (selected) climber's dual.
  rmm::device_uvector<f_t> dual_init(0, stream);
  const bool dual_warm_available =
    warm_start_n_points > 0 && warm_start_dual.size() == (size_t)warm_start_n_points * nct &&
    warm_start_best_c >= 0 && warm_start_best_c < warm_start_n_points;
  if (dual_warm_available) {
    dual_init.resize((size_t)n_points * nct, stream);
    const i_t carry  = std::min(n_carried_over_points, warm_start_n_points);
    const i_t best   = warm_start_best_c;
    const i_t nct_l  = nct;
    const f_t* wdual = warm_start_dual.data();
    f_t* dinit       = dual_init.data();
    thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                     thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>((size_t)n_points * nct_l),
                     [=] __device__(size_t g) {
                       const size_t c = g / (size_t)nct_l;
                       const i_t r    = (i_t)(g % (size_t)nct_l);
                       const i_t src  = (c < (size_t)carry) ? (i_t)c : best;
                       // Sanitize the carried-over dual: a non-finite dual makes A^T y NaN.
                       const f_t v = wdual[(size_t)src * (size_t)nct_l + (size_t)r];
                       dinit[g]    = isfinite(v) ? v : f_t(0);
                     });
    settings.set_initial_dual_solution(dual_init.data(), (i_t)dual_init.size(), stream);
    CUOPT_LOG_INFO(
      "Batch projection warm start: primal from current points; dual %d/%d climbers carried over, "
      "rest from best climber %d",
      std::min(carry, n_points),
      n_points,
      best);
  } else {
    CUOPT_LOG_INFO("Batch projection warm start: primal from current points (cold dual)");
  }

  auto sol     = cuopt::mathematical_optimization::run_batch_pdlp(op, settings);
  auto& primal = sol.get_primal_solution();
  auto& dual   = sol.get_dual_solution();
  // A failed batch solve (LP error, or out of memory under concurrent B&B pressure) surfaces as an
  // empty / undersized primal-dual rather than throwing. Skip this projection so the caller can
  // round the current point and continue, instead of copying out of bounds.
  if (primal.size() != (size_t)n_points * unified_n_vars_total ||
      dual.size() != (size_t)n_points * nct) {
    CUOPT_LOG_INFO(
      "Batch projection returned no usable solution (primal %zu, dual %zu; expected %zu / %zu); "
      "skipping projection",
      primal.size(),
      dual.size(),
      (size_t)n_points * unified_n_vars_total,
      (size_t)n_points * nct);
    return false;
  }
  d_projected.resize((size_t)n_points * n_vars, stream);
  for (i_t c = 0; c < n_points; ++c) {
    raft::copy(d_projected.data() + (size_t)c * n_vars,
               primal.data() + (size_t)c * unified_n_vars_total,
               n_vars,
               stream);
  }

  // Persist this projection's full per-climber dual to warm start the next projection's dual.
  warm_start_dual.resize(dual.size(), stream);
  raft::copy(warm_start_dual.data(), dual.data(), dual.size(), stream);
  warm_start_n_points = n_points;
  solution.handle_ptr->sync_stream();
  return true;
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
                                                     bool& climber0_cycle)
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
    is_feasible = test_fj_feasible(solution, batch_config.fj_seed_time_ratio * proj_and_round_time);
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

// Second CP rounding on the best cloud climber (excluding climber 0). Selects the best of climbers
// [1, n_points), verifies a near-feasible fully-integer projection with a full-precision LP, then
// CP-rounds it without disturbing climber 0's last_rounding. No FJ fallback, no cycle/restart.
template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_best_climber_step(
  solution_t<i_t, f_t>& solution,
  i_t n_points,
  const rmm::device_uvector<f_t>& d_cloud,
  const rmm::device_uvector<f_t>& d_projected)
{
  raft::common::nvtx::range fun_scope("run_best_climber_step");
  // Loads the best of climbers [1, n_points) into solution.assignment and records
  // warm_start_best_c.
  select_cloud_point(solution, n_points, d_cloud, d_projected, /*start_climber=*/1);
  // Clamp the marginal out-of-bounds slack left by batch PDLP's loose tolerance.
  solution.clamp_within_bounds();

  bool is_feasible = solution.compute_feasibility();
  i_t n_integers   = solution.compute_number_of_integers();
  if (n_integers == solution.problem_ptr->n_integer_vars) {
    if (is_feasible) {
      return true;
    } else if (last_selected_l1 < distance_to_check_for_feasible) {
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
  is_feasible = round(solution, /*update_last_rounding=*/false);
  if (is_feasible) {
    bool res = solution.compute_feasibility();
    cuopt_assert(res, "Feasibility issue");
    return true;
  }
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

  auto stream         = solution.handle_ptr->get_stream();
  const i_t n_vars    = unified_n_vars;
  const f_t inf       = std::numeric_limits<f_t>::infinity();
  reseed_count        = 0;   // persistent trajectories don't use assemble_cloud's reseed step
  warm_start_n_points = 0;   // cold dual at descent start
  climber0_int_ratio  = 0.;  // start loose; tightens as climber 0 converges (like the single FP)
  // Climber 0 begins from the nearest rounding (classic FP start).
  solution.round_nearest();
  raft::copy(last_rounding.data(), solution.assignment.data(), solution.assignment.size(), stream);

  rmm::device_uvector<f_t> d_cloud(0, stream);
  rmm::device_uvector<f_t> d_pool(0, stream);
  rmm::device_uvector<f_t> d_projected(0, stream);
  rmm::device_uvector<f_t> cand_best(n_vars, stream);  // best feasible found this round
  std::vector<f_t> h_lb_scratch(n_vars), h_ub_scratch(n_vars);
  i_t n_points         = 0;
  bool first_iteration = true;
  i_t trajectory       = 0;

  // FNV-1a hash of the integer components of a host assignment slice.
  auto int_hash = [&](const f_t* x) {
    size_t h = 1469598103934665603ull;
    for (i_t k = 0; k < unified_n_int; ++k) {
      long long v = std::llround(x[h_integer_indices_cache[k]]);
      h           = (h ^ (size_t)v) * 1099511628211ull;
    }
    return h;
  };

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

    // ---- Build (first round) or refresh the working cloud ----
    if (first_iteration) {
      // Slot 0 = climber 0 (forced by assemble_cloud); slots 1..N = fresh diversity points.
      bool seed_found_feasible = false;
      n_points = assemble_cloud(solution, batch_size, true, d_cloud, seed_found_feasible);
      if (seed_found_feasible) {
        bool res = solution.compute_feasibility();
        cuopt_assert(res, "Feasibility issue");
        CUOPT_LOG_INFO("New feasible solution: seeding 20%% FJ found feasible (objective %g)",
                       solution.get_user_objective());
        return res;
      }
      if (n_points == 0) {
        CUOPT_LOG_INFO("Empty cloud; falling back to single FP");
        return run_single_fp_descent(solution);
      }
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

    if (!project_cloud(solution, n_points, d_cloud, d_projected)) {
      bool is_feasible = round(solution);
      if (is_feasible && solution.compute_feasibility()) { return true; }
      return false;
    }

    // ---- Climber 0 (slot 0): full classic FP with internal restarts (the only CP round) ----
    raft::copy(solution.assignment.data(), d_projected.data(), n_vars, stream);
    // Batch PDLP's loose tolerance can leave the primal marginally outside variable bounds; clamp.
    solution.clamp_within_bounds();
    raft::copy(
      last_projection.data(), solution.assignment.data(), solution.assignment.size(), stream);
    bool climber0_cycle = false;
    bool feasible0      = run_climber0_step(solution, proj_begin, climber0_cycle);
    f_t best_obj        = inf;
    bool have_feasible  = false;
    if (feasible0) {
      best_obj      = solution.get_objective();
      have_feasible = true;
      raft::copy(cand_best.data(), solution.assignment.data(), n_vars, stream);
      solution.handle_ptr->sync_stream();
    }

    // ---- Diversity climbers 1..N: advance by sequential probing-cache rounding (no propagation)
    // ----
    auto h_proj = cuopt::host_copy(d_projected, stream);
    solution.handle_ptr->sync_stream();
    // Keep continuous columns at the projection; integer columns get the probing-cache rounding.
    std::vector<f_t> new_seed(h_proj.begin(), h_proj.end());
    std::vector<size_t> climber_hash(n_points, 0);
    std::vector<char> flagged(n_points, 0);
    std::vector<char> feas_host(n_points, 0);
    std::unordered_map<size_t, i_t> seen_hash;
    seen_hash.reserve((size_t)n_points * 2);
    for (i_t c = 1; c < n_points; ++c) {
      f_t* seed_c = new_seed.data() + (size_t)c * n_vars;
      probing_cache_sequential_round(
        solution, h_proj.data() + (size_t)c * n_vars, h_lb_scratch, h_ub_scratch, seed_c);
      size_t hsh      = int_hash(seed_c);
      climber_hash[c] = hsh;
      feas_host[c]    = host_assignment_feasible(seed_c) ? 1 : 0;
      // integer-assignment cycle: this climber's new rounding repeats one of its recent roundings
      bool cyc = false;
      for (size_t prev : climber_hash_history[c]) {
        if (prev == hsh) {
          cyc = true;
          break;
        }
      }
      // diversity dedup: keep the lower-indexed climber, replace later duplicates
      bool dup   = !seen_hash.insert({hsh, c}).second;
      flagged[c] = (cyc || dup) ? 1 : 0;
    }

    // ---- Device-confirm the host-feasible diversity climbers; track the best objective ----
    i_t n_feas_div = 0;
    for (i_t c = 1; c < n_points; ++c) {
      if (!feas_host[c]) continue;
      raft::copy(solution.assignment.data(), new_seed.data() + (size_t)c * n_vars, n_vars, stream);
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

    i_t n_flagged = 0;
    for (i_t c = 1; c < n_points; ++c) {
      n_flagged += flagged[c];
    }
    CUOPT_LOG_INFO(
      "Batched FP trajectory %d: climber0 feasible %d; diversity feasible %d, flagged %d / %d",
      trajectory,
      (int)feasible0,
      n_feas_div,
      n_flagged,
      n_points - 1);

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

    if (timer.check_time_limit()) {
      CUOPT_LOG_INFO("Batched FP time limit reached after %d trajectories", trajectory);
      return false;
    }

    // ---- Replace cycled / duplicate diversity climbers with fresh points (FJ then padding) ----
    if (n_flagged > 0) {
      bool pool_feasible = false;
      i_t pool_count     = assemble_cloud(solution, batch_size, false, d_pool, pool_feasible);
      if (pool_feasible) {
        bool res = solution.compute_feasibility();
        cuopt_assert(res, "Feasibility issue");
        CUOPT_LOG_INFO("New feasible solution: replacement-pool FJ found feasible (objective %g)",
                       solution.get_user_objective());
        return res;
      }
      auto h_pool = cuopt::host_copy(d_pool, stream);
      solution.handle_ptr->sync_stream();
      i_t pool_idx = 0;  // pool slot 0 is climber 0; draw replacements from slots 1..
      for (i_t c = 1; c < n_points; ++c) {
        if (!flagged[c]) continue;
        i_t src = (pool_count > 1) ? (1 + (pool_idx % (pool_count - 1))) : 0;
        std::copy(h_pool.begin() + (size_t)src * n_vars,
                  h_pool.begin() + (size_t)(src + 1) * n_vars,
                  new_seed.begin() + (size_t)c * n_vars);
        pool_idx++;
        climber_hash_history[c].clear();
      }
    }

    // ---- Commit the new persistent cloud (slot 0 = climber 0; 1..N = advanced or replaced) ----
    {
      auto h_last_rounding = cuopt::host_copy(last_rounding, stream);
      solution.handle_ptr->sync_stream();
      std::copy(h_last_rounding.begin(), h_last_rounding.begin() + n_vars, new_seed.begin());
      raft::copy(d_cloud.data(), new_seed.data(), (size_t)n_points * n_vars, stream);
      solution.handle_ptr->sync_stream();
    }
    // Push the new rounding into each advanced (non-replaced) climber's cycle history.
    for (i_t c = 1; c < n_points; ++c) {
      if (flagged[c]) continue;
      auto& hist = climber_hash_history[c];
      hist.push_back(climber_hash[c]);
      while ((i_t)hist.size() > cycle_queue.cycle_detection_length) {
        hist.pop_front();
      }
    }

    // ---- Only climber 0 restarts (self-contained, like the original FP) ----
    if (climber0_cycle) {
      CUOPT_LOG_INFO("Climber 0 cycle at trajectory %d; restarting (FJ escape + perturbate)",
                     trajectory);
      restart_fp(solution);
      // Re-anchor climber 0 from the perturbed point for the next round's slot 0.
      solution.round_nearest();
      raft::copy(
        last_rounding.data(), solution.assignment.data(), solution.assignment.size(), stream);
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
    const rmm::device_uvector<F_TYPE>&,                                                            \
    int);                                                                                          \
  template bool feasibility_pump_t<int, F_TYPE>::run_climber0_step(                                \
    solution_t<int, F_TYPE>&, F_TYPE, bool&);                                                      \
  template bool feasibility_pump_t<int, F_TYPE>::run_best_climber_step(                            \
    solution_t<int, F_TYPE>&,                                                                      \
    int,                                                                                           \
    const rmm::device_uvector<F_TYPE>&,                                                            \
    const rmm::device_uvector<F_TYPE>&);                                                           \
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
