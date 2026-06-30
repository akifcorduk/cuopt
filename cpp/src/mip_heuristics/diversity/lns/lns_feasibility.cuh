/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/diversity/recombiners/recombiner_configs.hpp>
#include <mip_heuristics/local_search/local_search.cuh>
#include <mip_heuristics/local_search/rounding/constraint_prop.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>

#include <branch_and_bound/branch_and_bound.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <pdlp/initial_scaling_strategy/initial_scaling.cuh>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <utility>
#include <vector>

#ifndef LNS_DEBUG
#define LNS_DEBUG 1
#endif

namespace cuopt::linear_programming::detail {

template <typename i_t>
struct lns_not_ruined_var_t {
  const i_t* ruined_vars;

  __device__ bool operator()(i_t var_idx) const { return ruined_vars[var_idx] == 0; }
};

template <typename i_t>
struct lns_mark_var_t {
  i_t* ruined_vars;

  __device__ void operator()(i_t var_idx) const { ruined_vars[var_idx] = 1; }
};

template <typename i_t, typename f_t>
struct lns_negative_score_t {
  const f_t* scores;

  __device__ bool operator()(i_t idx) const { return scores[idx] < f_t{0}; }
};

template <typename i_t, typename f_t>
struct lns_update_constraint_tardiness_t {
  typename solution_t<i_t, f_t>::view_t sol_view;
  i_t* tardiness;
  f_t rel_tol;

  __device__ void operator()(i_t idx)
  {
    if ((sol_view.lower_excess[idx] + sol_view.upper_excess[idx]) > rel_tol) { ++tardiness[idx]; }
  }
};

template <typename i_t, typename f_t>
struct lns_tardiness_penalty_t {
  typename solution_t<i_t, f_t>::view_t sol_view;
  const i_t* tardiness;
  f_t rel_tol;

  __device__ f_t operator()(i_t idx) const
  {
    const f_t excess = sol_view.lower_excess[idx] + sol_view.upper_excess[idx];
    return excess > rel_tol ? excess * static_cast<f_t>(1 + tardiness[idx]) : f_t{0};
  }
};

template <typename i_t, typename f_t>
struct lns_weighted_violation_count_t {
  typename solution_t<i_t, f_t>::view_t sol_view;
  const i_t* tardiness;
  f_t rel_tol;

  __device__ f_t operator()(i_t idx) const
  {
    const f_t excess = sol_view.lower_excess[idx] + sol_view.upper_excess[idx];
    return excess > rel_tol ? static_cast<f_t>(1 + tardiness[idx]) : f_t{0};
  }
};

template <typename i_t, typename f_t>
struct lns_lower_violated_t {
  typename solution_t<i_t, f_t>::view_t sol_view;
  f_t rel_tol;

  __device__ bool operator()(i_t idx) const { return sol_view.lower_excess[idx] > rel_tol; }
};

template <typename i_t, typename f_t>
struct lns_upper_violated_t {
  typename solution_t<i_t, f_t>::view_t sol_view;
  f_t rel_tol;

  __device__ bool operator()(i_t idx) const { return sol_view.upper_excess[idx] > rel_tol; }
};

template <typename i_t, typename f_t>
struct lns_constraint_violated_t {
  typename solution_t<i_t, f_t>::view_t sol_view;
  f_t rel_tol;

  __device__ bool operator()(i_t idx) const
  {
    return (sol_view.lower_excess[idx] + sol_view.upper_excess[idx]) > rel_tol;
  }
};

template <typename i_t, typename f_t>
struct lns_constraint_excess_t {
  typename solution_t<i_t, f_t>::view_t sol_view;

  __device__ f_t operator()(i_t idx) const
  {
    return sol_view.lower_excess[idx] + sol_view.upper_excess[idx];
  }
};

template <typename i_t, typename f_t>
struct lns_violated_constraint_excess_t {
  typename solution_t<i_t, f_t>::view_t sol_view;
  f_t rel_tol;

  __device__ f_t operator()(i_t idx) const
  {
    const f_t excess = sol_view.lower_excess[idx] + sol_view.upper_excess[idx];
    return excess > rel_tol ? excess : std::numeric_limits<f_t>::infinity();
  }
};

template <typename i_t, typename f_t>
struct lns_worse_constraint_t {
  typename solution_t<i_t, f_t>::view_t sol_view;

  __device__ bool operator()(i_t lhs, i_t rhs) const
  {
    const f_t lhs_excess = sol_view.lower_excess[lhs] + sol_view.upper_excess[lhs];
    const f_t rhs_excess = sol_view.lower_excess[rhs] + sol_view.upper_excess[rhs];
    return lhs_excess < rhs_excess;
  }
};

template <typename i_t, typename f_t>
__global__ void lns_mark_violated_constraint_integer_vars_kernel(
  typename solution_t<i_t, f_t>::view_t sol_view, i_t* ruined_vars, f_t rel_tol)
{
  for (i_t constraint_idx = blockIdx.x * blockDim.x + threadIdx.x;
       constraint_idx < sol_view.problem.n_constraints;
       constraint_idx += blockDim.x * gridDim.x) {
    if ((sol_view.lower_excess[constraint_idx] + sol_view.upper_excess[constraint_idx]) <=
        rel_tol) {
      continue;
    }
    for (i_t offset = sol_view.problem.offsets[constraint_idx];
         offset < sol_view.problem.offsets[constraint_idx + 1];
         ++offset) {
      const i_t var_idx = sol_view.problem.variables[offset];
      if (sol_view.problem.is_integer_var(var_idx)) { atomicExch(&ruined_vars[var_idx], i_t{1}); }
    }
  }
}

template <typename i_t, typename f_t>
__global__ void lns_compute_seed_neighbor_scores_kernel(
  const f_t* assignment,
  i_t seed_var,
  const i_t* neighbor_vars,
  const i_t* pair_offsets,
  const typename type_2<f_t>::type* pair_coefficients,
  i_t neighbor_count,
  const i_t* ruined_vars,
  f_t* candidate_scores,
  i_t* candidate_vars)
{
  static constexpr i_t warp_size = 32;
  const i_t global_thread        = blockIdx.x * blockDim.x + threadIdx.x;
  const i_t global_warp          = global_thread / warp_size;
  const i_t lane                 = threadIdx.x & (warp_size - 1);
  const i_t n_warps              = (blockDim.x * gridDim.x) / warp_size;
  const f_t seed_value           = assignment[seed_var];

  for (i_t neighbor_pos = global_warp; neighbor_pos < neighbor_count; neighbor_pos += n_warps) {
    const i_t candidate_var = neighbor_vars[neighbor_pos];
    const i_t pair_begin    = pair_offsets[neighbor_pos];
    const i_t pair_end      = pair_offsets[neighbor_pos + 1];

    f_t difference_sum = 0.;
    for (i_t pair_pos = pair_begin + lane; pair_pos < pair_end; pair_pos += warp_size) {
      const auto coeff_pair = pair_coefficients[pair_pos];
      difference_sum += abs(coeff_pair.x * seed_value - coeff_pair.y * assignment[candidate_var]);
    }
    difference_sum = raft::warpReduce(difference_sum);

    if (lane == 0) {
      candidate_vars[neighbor_pos] = candidate_var;
      candidate_scores[neighbor_pos] =
        ruined_vars[candidate_var] == 0
          ? -feasibility_lns_config_t::alpha * static_cast<f_t>(pair_end - pair_begin) +
              feasibility_lns_config_t::beta * difference_sum
          : std::numeric_limits<f_t>::infinity();
    }
  }
}

template <typename i_t, typename f_t>
__global__ void lns_compute_violation_seed_scores_kernel(typename solution_t<i_t, f_t>::view_t sol,
                                                         const i_t* integer_indices,
                                                         i_t n_integer_vars,
                                                         f_t* seed_scores)
{
  const f_t rel_tol = sol.problem.tolerances.relative_tolerance;
  for (i_t integer_pos = blockIdx.x * blockDim.x + threadIdx.x; integer_pos < n_integer_vars;
       integer_pos += blockDim.x * gridDim.x) {
    const i_t var_idx = integer_indices[integer_pos];
    f_t violation_sum = 0.;
    for (i_t offset = sol.problem.reverse_offsets[var_idx];
         offset < sol.problem.reverse_offsets[var_idx + 1];
         ++offset) {
      const i_t constraint_idx = sol.problem.reverse_constraints[offset];
      violation_sum += sol.lower_excess[constraint_idx] + sol.upper_excess[constraint_idx];
    }
    seed_scores[integer_pos] =
      violation_sum > rel_tol ? -violation_sum : std::numeric_limits<f_t>::infinity();
  }
}

template <typename i_t>
__global__ void lns_seed_ruin_kernel(i_t seed_var, i_t* ruined_vars, i_t* ruin_vars)
{
  ruined_vars[seed_var] = 1;
  ruin_vars[0]          = seed_var;
}

template <typename i_t>
__global__ void lns_mark_selected_neighbor_ruin_kernel(const i_t* candidate_vars,
                                                       i_t selected_count,
                                                       i_t* ruined_vars,
                                                       i_t* ruin_vars)
{
  for (i_t selected_pos = blockIdx.x * blockDim.x + threadIdx.x; selected_pos < selected_count;
       selected_pos += blockDim.x * gridDim.x) {
    const i_t var_idx           = candidate_vars[selected_pos];
    ruined_vars[var_idx]        = 1;
    ruin_vars[selected_pos + 1] = var_idx;
  }
}

template <typename i_t, typename f_t>
__device__ f_t lns_ruined_value(typename problem_t<i_t, f_t>::view_t pb,
                                i_t fixed_var_idx,
                                i_t original_var_idx,
                                uint64_t seed)
{
  const auto bounds = pb.variable_bounds[fixed_var_idx];
  const f_t lb      = get_lower(bounds);
  const f_t ub      = get_upper(bounds);

  raft::random::PCGenerator rng(seed, static_cast<uint64_t>(original_var_idx), 0);
  const f_t frac = rng.next_double();

  f_t value = f_t{0.5};
  if (isfinite(lb) && isfinite(ub)) {
    value = lb + frac * (ub - lb);
  } else if (isfinite(lb)) {
    value = lb + frac;
  } else if (isfinite(ub)) {
    value = ub - frac;
  }

  return value < lb ? lb : (value > ub ? ub : value);
}

template <typename i_t, typename f_t>
__global__ void lns_unset_ruined_values_kernel(typename solution_t<i_t, f_t>::view_t fixed_sol,
                                               const i_t* variable_map,
                                               const i_t* ruined_vars,
                                               uint64_t seed)
{
  for (i_t fixed_var_idx = blockIdx.x * blockDim.x + threadIdx.x;
       fixed_var_idx < fixed_sol.problem.n_variables;
       fixed_var_idx += blockDim.x * gridDim.x) {
    const i_t original_var_idx = variable_map[fixed_var_idx];
    if (ruined_vars[original_var_idx] == 0) { continue; }
    fixed_sol.assignment[fixed_var_idx] =
      lns_ruined_value<i_t, f_t>(fixed_sol.problem, fixed_var_idx, original_var_idx, seed);
  }
}

#if LNS_DEBUG
template <typename i_t, typename f_t>
struct lns_ruined_changed_t {
  const f_t* before;
  const f_t* after;
  const i_t* ruined;
  f_t change_tol;

  __device__ bool operator()(i_t idx) const
  {
    return ruined[idx] != 0 && abs(after[idx] - before[idx]) > change_tol;
  }
};

template <typename i_t, typename f_t>
struct lns_ruined_delta_t {
  const f_t* before;
  const f_t* after;
  const i_t* ruined;

  __device__ f_t operator()(i_t idx) const
  {
    return ruined[idx] != 0 ? abs(after[idx] - before[idx]) : f_t{0};
  }
};
#endif

template <typename i_t, typename f_t>
class lns_feasibility_t {
 public:
  lns_feasibility_t(mip_solver_context_t<i_t, f_t>& context_, local_search_t<i_t, f_t>& ls_)
    : context(context_),
      ls(ls_),
      problem_ptr(context.problem_ptr),
      constraint_prop(ls_.constraint_prop),
      candidate_scores(problem_ptr->n_integer_vars, problem_ptr->handle_ptr->get_stream()),
      related_candidate_vars(problem_ptr->n_integer_vars, problem_ptr->handle_ptr->get_stream()),
      violated_seed_pool(problem_ptr->n_integer_vars, problem_ptr->handle_ptr->get_stream()),
      related_neighbor_offsets(0, problem_ptr->handle_ptr->get_stream()),
      related_neighbor_vars(0, problem_ptr->handle_ptr->get_stream()),
      related_pair_offsets(0, problem_ptr->handle_ptr->get_stream()),
      related_pair_coefficients(0, problem_ptr->handle_ptr->get_stream()),
      ruin_vars(problem_ptr->n_integer_vars, problem_ptr->handle_ptr->get_stream()),
      vars_to_fix(problem_ptr->n_integer_vars, problem_ptr->handle_ptr->get_stream()),
      ruined_var_flags(problem_ptr->n_variables, problem_ptr->handle_ptr->get_stream()),
      constraint_tardiness(problem_ptr->n_constraints, problem_ptr->handle_ptr->get_stream()),
      rng(cuopt::seed_generator::get_seed())
  {
    auto stream = problem_ptr->handle_ptr->get_stream();
    thrust::fill(problem_ptr->handle_ptr->get_thrust_policy(),
                 constraint_tardiness.begin(),
                 constraint_tardiness.end(),
                 i_t{0});
    build_relatedness_cache();
  }

  bool run(solution_t<i_t, f_t>& solution, f_t time_limit)
  {
    raft::common::nvtx::range fun_scope("run_standalone_lns_feasibility");
    if (time_limit <= 0.) { return solution.get_feasible(); }

    lns_timer = timer_t(time_limit);
    solution.compute_feasibility();
    CUOPT_LOG_INFO(
      "Standalone LNS start: time_limit %.2fs, vars %d, integer vars %d, constraints %d, "
      "initial feasible %d, unsat %d, excess %.6e",
      time_limit,
      problem_ptr->n_variables,
      problem_ptr->n_integer_vars,
      problem_ptr->n_constraints,
      solution.get_feasible(),
      n_unsatisfied_constraints(solution),
      solution.get_total_excess());

    bool accepted_any = seed_repair(solution, lns_timer);
    solution.compute_feasibility();
    CUOPT_LOG_INFO("Standalone LNS seed repair: improved %d, feasible %d, unsat %d, excess %.6e",
                   accepted_any,
                   solution.get_feasible(),
                   n_unsatisfied_constraints(solution),
                   solution.get_total_excess());
    if (solution.get_feasible()) {
      CUOPT_LOG_INFO("Standalone LNS feasible after seed repair (%.2fs elapsed)",
                     lns_timer.elapsed_time());
      return finalize_feasible_solution(solution);
    }

    // Quick Feasibility-Jump burst from the seed (POC step 1: "super quick FJ"). FJ is the
    // strongest feasibility engine; starting the ruin/repair loop from its result gives a much
    // better incumbent than the LP-rounded seed alone.
    if (!lns_timer.check_time_limit()) {
      solution_t<i_t, f_t> fj_sol(solution);
      const f_t fj_budget = std::min<f_t>(feasibility_lns_config_t::fj_polish_initial_time_limit,
                                          lns_timer.remaining_time());
      const bool fj_feasible = fj_polish(fj_sol, fj_budget, "lns_fj_initial");
      fj_sol.compute_feasibility();
      CUOPT_LOG_INFO("Standalone LNS initial FJ: feasible %d, unsat %d, excess %.6e (%.2fs elapsed)",
                     fj_feasible,
                     n_unsatisfied_constraints(fj_sol),
                     fj_sol.get_total_excess(),
                     lns_timer.elapsed_time());
      if (fj_feasible) {
        solution.copy_from(fj_sol);
        return finalize_feasible_solution(solution);
      }
      if (is_better_feasibility_state(fj_sol, solution)) { solution.copy_from(fj_sol); }
    }
    solution_t<i_t, f_t> best_solution(solution);

    weight_t<i_t, f_t> weights(problem_ptr->n_constraints, problem_ptr->handle_ptr);
    // Keep ruin sets small: small neighborhoods repair fast and preserve most of the
    // current structure. Growing the ruin set toward "re-randomize everything" destroys
    // progress, so cap it and grow additively (not by doubling) on stalls.
    const i_t min_ruin = 8;
    const i_t max_ruin = std::max(min_ruin, std::min<i_t>(64, problem_ptr->n_integer_vars / 2));
    const i_t grow_after_failures = 5;
    i_t ruin_count                = min_ruin;
    i_t failure_streak            = 0;
    size_t attempted_repairs      = 0;
    size_t accepted_repairs       = 0;
    size_t attempts_since_fj          = feasibility_lns_config_t::fj_polish_min_attempts_between;
    size_t attempts_since_sub_mip     = feasibility_lns_config_t::sub_mip_repair_min_attempts_between;
    for (size_t attempt = 0; attempt < feasibility_lns_config_t::max_attempts; ++attempt) {
      ++attempts_since_fj;
      ++attempts_since_sub_mip;
      if (lns_timer.check_time_limit()) {
        CUOPT_LOG_INFO("Standalone LNS time limit hit after %lu attempts (%.2fs elapsed)",
                       attempted_repairs,
                       lns_timer.elapsed_time());
        break;
      }

      // Periodic Feasibility-Jump polish whenever the incumbent is close to feasible. The
      // bounds-propagation repair grinds the last few violations down very slowly (often one
      // unit per accepted move), so we periodically hand the near-feasible incumbent to FJ,
      // which can resolve the remaining violations in a single short burst. FJ keeps its own
      // (persistent, weight-updated) state, so repeated calls explore different moves.
      const i_t best_unsat = n_unsatisfied_constraints(best_solution);
      if (best_unsat > 0 &&
          best_unsat <= static_cast<i_t>(feasibility_lns_config_t::fj_polish_unsat_threshold) &&
          attempts_since_fj >= feasibility_lns_config_t::fj_polish_min_attempts_between) {
        attempts_since_fj = 0;
        solution_t<i_t, f_t> fj_sol(best_solution);
        const f_t fj_budget = std::min<f_t>(feasibility_lns_config_t::fj_polish_time_limit,
                                            lns_timer.remaining_time());
        const bool fj_feasible = fj_polish(fj_sol, fj_budget, "lns_fj_polish");
        fj_sol.compute_feasibility();
        CUOPT_LOG_INFO(
          "Standalone LNS FJ polish: feasible %d, unsat %d -> %d, excess %.6e (%.2fs elapsed)",
          fj_feasible,
          best_unsat,
          n_unsatisfied_constraints(fj_sol),
          fj_sol.get_total_excess(),
          lns_timer.elapsed_time());
        if (fj_feasible) {
          solution.copy_from(fj_sol);
          best_solution.copy_from(fj_sol);
          return finalize_feasible_solution(solution);
        }
        if (is_better_feasibility_state(fj_sol, best_solution)) {
          best_solution.copy_from(fj_sol);
          solution.copy_from(fj_sol);
        }
      }

      // Mini-MIP repair when stuck at a small number of violated constraints that the cheaper
      // repairs cannot close.
      const i_t sub_mip_best_unsat = n_unsatisfied_constraints(best_solution);
      if (sub_mip_best_unsat > 0 &&
          sub_mip_best_unsat <=
            static_cast<i_t>(feasibility_lns_config_t::sub_mip_repair_unsat_threshold) &&
          attempts_since_sub_mip >=
            feasibility_lns_config_t::sub_mip_repair_min_attempts_between &&
          !lns_timer.check_time_limit()) {
        attempts_since_sub_mip       = 0;
        const i_t sub_mip_unsat_pre  = sub_mip_best_unsat;
        const bool sub_mip_feasible  = sub_mip_repair(best_solution);
        best_solution.compute_feasibility();
        CUOPT_LOG_INFO(
          "Standalone LNS sub-MIP repair: feasible %d, unsat %d -> %d (%.2fs elapsed)",
          sub_mip_feasible,
          sub_mip_unsat_pre,
          n_unsatisfied_constraints(best_solution),
          lns_timer.elapsed_time());
        if (sub_mip_feasible) {
          solution.copy_from(best_solution);
          return finalize_feasible_solution(solution);
        }
        if (is_better_feasibility_state(best_solution, solution)) {
          solution.copy_from(best_solution);
        }
      }

      const i_t unsat_before = n_unsatisfied_constraints(solution);
      const bool use_violated_constraint_ruin =
        unsat_before <=
          static_cast<i_t>(feasibility_lns_config_t::violated_constraint_ruin_unsat_limit) &&
        attempt % 2 == 0;
      auto [candidate, accepted] = use_violated_constraint_ruin
                                     ? violated_constraint_ruin_repair(solution, weights, attempt)
                                     : ruin_repair(solution, weights, ruin_count, attempt);
      ++attempted_repairs;
      candidate.compute_feasibility();
      CUOPT_LOG_INFO(
        "Standalone LNS attempt %lu: strategy %s, ruin_count %d, accepted %d, "
        "unsat %d -> %d, excess %.6e -> %.6e (%.2fs elapsed)",
        attempt,
        use_violated_constraint_ruin ? "violated_constraint" : "related",
        ruin_count,
        accepted,
        unsat_before,
        n_unsatisfied_constraints(candidate),
        solution.get_total_excess(),
        candidate.get_total_excess(),
        lns_timer.elapsed_time());
      if (!use_violated_constraint_ruin) {
        if (accepted) {
          ruin_count     = std::max(min_ruin, ruin_count - min_ruin);
          failure_streak = 0;
        } else if (++failure_streak >= grow_after_failures) {
          ruin_count     = std::min(max_ruin, ruin_count + min_ruin);
          failure_streak = 0;
        }
      }
      if (!accepted) {
        CUOPT_LOG_INFO(" ");
        continue;
      }

      solution.copy_from(candidate);
      accepted_any = true;
      ++accepted_repairs;
      if (is_better_feasibility_state(solution, best_solution)) {
        best_solution.copy_from(solution);
        CUOPT_LOG_INFO("Standalone LNS new best: unsat %d, excess %.6e (attempt %lu)",
                       n_unsatisfied_constraints(best_solution),
                       best_solution.get_total_excess(),
                       attempt);
      }
      if (solution.get_feasible()) {
        CUOPT_LOG_INFO("Standalone LNS feasible at attempt %lu (%lu/%lu accepted, %.2fs elapsed)",
                       attempt,
                       accepted_repairs,
                       attempted_repairs,
                       lns_timer.elapsed_time());
        return finalize_feasible_solution(solution);
      }
      CUOPT_LOG_INFO(" ");
    }
    if (is_better_feasibility_state(best_solution, solution)) { solution.copy_from(best_solution); }
    if (solution.get_feasible()) { finalize_feasible_solution(solution); }
#if LNS_DEBUG
    if (!solution.get_feasible()) { log_violation_summary(solution, "Standalone LNS final"); }
#endif
    CUOPT_LOG_INFO(
      "Standalone LNS feasibility finished: feasible %d, unsat %d, excess %.6e, "
      "repairs %lu, accepted %lu",
      solution.get_feasible(),
      n_unsatisfied_constraints(solution),
      solution.get_total_excess(),
      attempted_repairs,
      accepted_repairs);
    return accepted_any;
  }

 private:
  using f_t2 = typename type_2<f_t>::type;

  void build_relatedness_cache()
  {
    raft::common::nvtx::range fun_scope("standalone_lns_build_relatedness_cache");
    auto stream              = problem_ptr->handle_ptr->get_stream();
    const i_t n_variables    = problem_ptr->n_variables;
    const i_t n_integer_vars = problem_ptr->n_integer_vars;

    auto h_integer_indices = cuopt::host_copy(problem_ptr->integer_indices, stream);
    integer_indices_host   = h_integer_indices;

    std::vector<i_t> h_neighbor_offsets(n_variables + 1, 0);
    std::vector<i_t> h_neighbor_vars;
    std::vector<i_t> h_pair_offsets(1, 0);
    std::vector<f_t2> h_pair_coefficients;

    if (n_variables == 0 || n_integer_vars == 0) {
      related_neighbor_offsets_host = h_neighbor_offsets;
      related_neighbor_offsets      = cuopt::device_copy(h_neighbor_offsets, stream);
      related_neighbor_vars.resize(0, stream);
      related_pair_offsets = cuopt::device_copy(h_pair_offsets, stream);
      related_pair_coefficients.resize(0, stream);
      return;
    }

    auto h_reverse_offsets      = cuopt::host_copy(problem_ptr->reverse_offsets, stream);
    auto h_reverse_constraints  = cuopt::host_copy(problem_ptr->reverse_constraints, stream);
    auto h_reverse_coefficients = cuopt::host_copy(problem_ptr->reverse_coefficients, stream);
    auto h_offsets              = cuopt::host_copy(problem_ptr->offsets, stream);
    auto h_variables            = cuopt::host_copy(problem_ptr->variables, stream);
    auto h_coefficients         = cuopt::host_copy(problem_ptr->coefficients, stream);

    std::vector<uint8_t> is_integer_var(n_variables, 0);
    for (const auto var_idx : h_integer_indices) {
      is_integer_var[var_idx] = 1;
    }

    for (i_t var_idx = 0; var_idx < n_variables; ++var_idx) {
      if (!is_integer_var[var_idx]) {
        h_neighbor_offsets[var_idx + 1] = static_cast<i_t>(h_neighbor_vars.size());
        continue;
      }

      std::map<i_t, std::vector<f_t2>> neighbor_coefficients;
      for (i_t reverse_pos = h_reverse_offsets[var_idx];
           reverse_pos < h_reverse_offsets[var_idx + 1];
           ++reverse_pos) {
        const i_t constraint_idx = h_reverse_constraints[reverse_pos];
        const f_t var_coeff      = h_reverse_coefficients[reverse_pos];
        for (i_t row_pos = h_offsets[constraint_idx]; row_pos < h_offsets[constraint_idx + 1];
             ++row_pos) {
          const i_t neighbor_var = h_variables[row_pos];
          if (neighbor_var == var_idx || !is_integer_var[neighbor_var]) { continue; }
          f_t2 coeff_pair;
          coeff_pair.x = var_coeff;
          coeff_pair.y = h_coefficients[row_pos];
          neighbor_coefficients[neighbor_var].push_back(coeff_pair);
        }
      }

      for (auto& [neighbor_var, coeff_pairs] : neighbor_coefficients) {
        h_neighbor_vars.push_back(neighbor_var);
        h_pair_coefficients.insert(
          h_pair_coefficients.end(), coeff_pairs.begin(), coeff_pairs.end());
        h_pair_offsets.push_back(static_cast<i_t>(h_pair_coefficients.size()));
      }
      h_neighbor_offsets[var_idx + 1] = static_cast<i_t>(h_neighbor_vars.size());
    }

    related_neighbor_offsets_host = h_neighbor_offsets;
    related_neighbor_offsets      = cuopt::device_copy(h_neighbor_offsets, stream);
    related_neighbor_vars         = cuopt::device_copy(h_neighbor_vars, stream);
    related_pair_offsets          = cuopt::device_copy(h_pair_offsets, stream);
    related_pair_coefficients     = cuopt::device_copy(h_pair_coefficients, stream);

#if LNS_DEBUG
    CUOPT_LOG_INFO(
      "Standalone LNS relatedness cache: variables %d, integer vars %d, "
      "neighbor pairs %lu, coefficient pairs %lu",
      n_variables,
      n_integer_vars,
      static_cast<unsigned long>(h_neighbor_vars.size()),
      static_cast<unsigned long>(h_pair_coefficients.size()));
#endif
  }

  bool finalize_feasible_solution(solution_t<i_t, f_t>& solution)
  {
    solution.compute_feasibility();
    if (!solution.get_feasible()) { return false; }
    if (problem_ptr->n_integer_vars == 0) { return true; }

    solution_t<i_t, f_t> rounded(solution);
    if (rounded.round_nearest()) { solution.copy_from(rounded); }
    return true;
  }

  bool is_better_feasibility_state(solution_t<i_t, f_t>& candidate,
                                   solution_t<i_t, f_t>& incumbent) const
  {
    candidate.compute_feasibility();
    incumbent.compute_feasibility();
    if (candidate.get_feasible() && !incumbent.get_feasible()) { return true; }
    if (!candidate.get_feasible() && incumbent.get_feasible()) { return false; }

    const i_t candidate_unsat = n_unsatisfied_constraints(candidate);
    const i_t incumbent_unsat = n_unsatisfied_constraints(incumbent);
    const f_t candidate_score = get_tardiness_penalty(candidate);
    const f_t incumbent_score = get_tardiness_penalty(incumbent);

    // Reject candidates whose infeasibility regresses by more than this magnitude.
    const i_t max_unsat_increase = std::max<i_t>(5 * incumbent_unsat, 10);
    if (candidate_unsat - incumbent_unsat > max_unsat_increase) { return false; }

    if (candidate_unsat < incumbent_unsat) { return true; }
    if (candidate_score + OBJECTIVE_EPSILON < incumbent_score) { return true; }
    return candidate_unsat == incumbent_unsat &&
           candidate_score <= incumbent_score + OBJECTIVE_EPSILON;
  }

  bool seed_repair(solution_t<i_t, f_t>& solution, timer_t& lns_timer)
  {
    solution_t<i_t, f_t> repaired(solution);
    const auto old_repair_iterations               = constraint_prop.max_n_failed_repair_iterations;
    constraint_prop.max_n_failed_repair_iterations = feasibility_lns_config_t::n_repair_iterations;
    timer_t seed_timer(
      std::min<f_t>(feasibility_lns_config_t::seed_repair_time_limit, lns_timer.remaining_time()));
    constraint_prop.apply_round(
      repaired, feasibility_lns_config_t::lp_after_bounds_prop_time_limit, seed_timer);
    constraint_prop.max_n_failed_repair_iterations = old_repair_iterations;
    repaired.compute_feasibility();

    solution_t<i_t, f_t> best_repaired(repaired);
    if (use_seed_lp_polish()) {
      solution_t<i_t, f_t> polished(repaired);
      lp_polish_with_integers_fixed(polished);
      if (is_better_feasibility_state(polished, best_repaired)) {
        best_repaired.copy_from(polished);
      }
    }

    if (best_repaired.get_feasible() || is_better_feasibility_state(best_repaired, solution) ||
        best_repaired.get_total_excess() + OBJECTIVE_EPSILON < solution.get_total_excess()) {
      solution.copy_from(best_repaired);
      return true;
    }
    return false;
  }

  bool use_seed_lp_polish() const { return problem_ptr->n_constraints > problem_ptr->n_variables; }

  i_t n_unsatisfied_constraints(solution_t<i_t, f_t>& sol) const
  {
    return sol.problem_ptr->n_constraints -
           sol.n_feasible_constraints.value(sol.handle_ptr->get_stream());
  }

  void ensure_constraint_tardiness_size(solution_t<i_t, f_t>& sol)
  {
    if (constraint_tardiness.size() == static_cast<size_t>(sol.problem_ptr->n_constraints)) {
      return;
    }
    constraint_tardiness.resize(sol.problem_ptr->n_constraints, sol.handle_ptr->get_stream());
    thrust::fill(sol.handle_ptr->get_thrust_policy(),
                 constraint_tardiness.begin(),
                 constraint_tardiness.end(),
                 i_t{0});
  }

  void update_constraint_tardiness(solution_t<i_t, f_t>& sol)
  {
    ensure_constraint_tardiness_size(sol);
    const auto sol_view = sol.view();
    const f_t rel_tol   = sol.problem_ptr->tolerances.relative_tolerance;
    thrust::for_each(
      sol.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(sol.problem_ptr->n_constraints),
      lns_update_constraint_tardiness_t<i_t, f_t>{sol_view, constraint_tardiness.data(), rel_tol});
  }

  f_t get_tardiness_penalty(solution_t<i_t, f_t>& sol) const
  {
    const auto sol_view = sol.view();
    const f_t rel_tol   = sol.problem_ptr->tolerances.relative_tolerance;
    return thrust::transform_reduce(
      sol.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(sol.problem_ptr->n_constraints),
      lns_tardiness_penalty_t<i_t, f_t>{sol_view, constraint_tardiness.data(), rel_tol},
      f_t{0},
      thrust::plus<f_t>());
  }

  // Guided-Local-Search score: sum over currently violated constraints of their weight
  // (1 + tardiness). Unlike the excess-weighted penalty, this weights the violation
  // *indicator*, so it does not reward spreading the same total excess across many
  // constraints. Minimizing it drives the search toward fewer (weighted) violations.
  f_t get_weighted_violation_count(solution_t<i_t, f_t>& sol) const
  {
    const auto sol_view = sol.view();
    const f_t rel_tol   = sol.problem_ptr->tolerances.relative_tolerance;
    return thrust::transform_reduce(
      sol.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(sol.problem_ptr->n_constraints),
      lns_weighted_violation_count_t<i_t, f_t>{sol_view, constraint_tardiness.data(), rel_tol},
      f_t{0},
      thrust::plus<f_t>());
  }

#if LNS_DEBUG
  std::pair<cudaEvent_t, cudaEvent_t> start_kernel_timer(cudaStream_t stream) const
  {
    cudaEvent_t start;
    cudaEvent_t stop;
    RAFT_CUDA_TRY(cudaEventCreate(&start));
    RAFT_CUDA_TRY(cudaEventCreate(&stop));
    RAFT_CUDA_TRY(cudaEventRecord(start, stream));
    return {start, stop};
  }

  void log_kernel_timer(std::pair<cudaEvent_t, cudaEvent_t> events,
                        const char* label,
                        cudaStream_t stream) const
  {
    RAFT_CUDA_TRY(cudaEventRecord(events.second, stream));
    RAFT_CUDA_TRY(cudaEventSynchronize(events.second));
    float elapsed_ms = 0.f;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed_ms, events.first, events.second));
    RAFT_CUDA_TRY(cudaEventDestroy(events.first));
    RAFT_CUDA_TRY(cudaEventDestroy(events.second));
    CUOPT_LOG_INFO("Standalone LNS kernel %s took %.3f ms", label, static_cast<double>(elapsed_ms));
  }

  void log_violation_summary(solution_t<i_t, f_t>& sol, const char* label) const
  {
    sol.compute_feasibility();
    const auto policy = sol.handle_ptr->get_thrust_policy();
    const auto view   = sol.view();
    const f_t rel_tol = sol.problem_ptr->tolerances.relative_tolerance;
    const i_t infeasible_count =
      thrust::count_if(policy,
                       thrust::make_counting_iterator<i_t>(0),
                       thrust::make_counting_iterator<i_t>(sol.problem_ptr->n_constraints),
                       lns_constraint_violated_t<i_t, f_t>{view, rel_tol});
    const i_t lower_count =
      thrust::count_if(policy,
                       thrust::make_counting_iterator<i_t>(0),
                       thrust::make_counting_iterator<i_t>(sol.problem_ptr->n_constraints),
                       lns_lower_violated_t<i_t, f_t>{view, rel_tol});
    const i_t upper_count =
      thrust::count_if(policy,
                       thrust::make_counting_iterator<i_t>(0),
                       thrust::make_counting_iterator<i_t>(sol.problem_ptr->n_constraints),
                       lns_upper_violated_t<i_t, f_t>{view, rel_tol});
    f_t min_infeasibility = f_t{0};
    f_t max_infeasibility = f_t{0};
    if (infeasible_count > 0) {
      min_infeasibility = thrust::transform_reduce(
        policy,
        thrust::make_counting_iterator<i_t>(0),
        thrust::make_counting_iterator<i_t>(sol.problem_ptr->n_constraints),
        lns_violated_constraint_excess_t<i_t, f_t>{view, rel_tol},
        std::numeric_limits<f_t>::infinity(),
        thrust::minimum<f_t>());
      max_infeasibility = thrust::transform_reduce(
        policy,
        thrust::make_counting_iterator<i_t>(0),
        thrust::make_counting_iterator<i_t>(sol.problem_ptr->n_constraints),
        lns_constraint_excess_t<i_t, f_t>{view},
        f_t{0},
        thrust::maximum<f_t>());
    }
    CUOPT_LOG_INFO(
      "%s infeasibility: constraints %d, lower %d, upper %d, min %.6e, max %.6e, total %.6e",
      label,
      infeasible_count,
      lower_count,
      upper_count,
      static_cast<double>(min_infeasibility),
      static_cast<double>(max_infeasibility),
      static_cast<double>(sol.get_total_excess()));
  }

  void log_ruined_assignment_delta(solution_t<i_t, f_t>& before,
                                   solution_t<i_t, f_t>& after,
                                   const char* label) const
  {
    const auto policy       = after.handle_ptr->get_thrust_policy();
    const f_t* before_ptr   = before.assignment.data();
    const f_t* after_ptr    = after.assignment.data();
    const i_t* ruined_ptr   = ruined_var_flags.data();
    const f_t change_tol    = after.problem_ptr->tolerances.absolute_tolerance;
    const i_t changed_count = thrust::count_if(
      policy,
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(after.problem_ptr->n_variables),
      lns_ruined_changed_t<i_t, f_t>{before_ptr, after_ptr, ruined_ptr, change_tol});
    const f_t max_delta =
      thrust::transform_reduce(policy,
                               thrust::make_counting_iterator<i_t>(0),
                               thrust::make_counting_iterator<i_t>(after.problem_ptr->n_variables),
                               lns_ruined_delta_t<i_t, f_t>{before_ptr, after_ptr, ruined_ptr},
                               f_t{0},
                               thrust::maximum<f_t>());
    CUOPT_LOG_INFO("%s ruined assignment delta: changed %d, max %.6e",
                   label,
                   changed_count,
                   static_cast<double>(max_delta));
  }

  const char* acceptance_reason(solution_t<i_t, f_t>& current,
                                solution_t<i_t, f_t>& candidate,
                                const weight_t<i_t, f_t>& weights) const
  {
    const i_t current_unsat   = n_unsatisfied_constraints(current);
    const i_t candidate_unsat = n_unsatisfied_constraints(candidate);
    if (candidate_unsat < current_unsat) { return "fewer infeasible constraints"; }
    if (candidate_unsat > current_unsat) { return "more infeasible constraints"; }

    const f_t current_excess   = current.get_total_excess();
    const f_t candidate_excess = candidate.get_total_excess();
    if (candidate_excess + OBJECTIVE_EPSILON < current_excess) { return "lower total excess"; }
    if (current_excess + OBJECTIVE_EPSILON < candidate_excess) { return "higher total excess"; }

    const f_t current_tardiness_penalty   = get_tardiness_penalty(current);
    const f_t candidate_tardiness_penalty = get_tardiness_penalty(candidate);
    if (candidate_tardiness_penalty + OBJECTIVE_EPSILON < current_tardiness_penalty) {
      return "lower tardiness penalty";
    }
    if (current_tardiness_penalty + OBJECTIVE_EPSILON < candidate_tardiness_penalty) {
      return "higher tardiness penalty";
    }

    const f_t current_quality   = current.get_quality(weights);
    const f_t candidate_quality = candidate.get_quality(weights);
    if (candidate_quality + OBJECTIVE_EPSILON < current_quality) {
      return "lower weighted quality";
    }
    return "not improving weighted quality";
  }

  void log_acceptance_diagnostics(solution_t<i_t, f_t>& current,
                                  solution_t<i_t, f_t>& candidate,
                                  const weight_t<i_t, f_t>& weights,
                                  bool accepted) const
  {
    CUOPT_LOG_INFO(
      "Standalone LNS acceptance: accepted %d, reason %s, unsat %d -> %d, excess %.6e -> "
      "%.6e, tardiness %.6e -> %.6e, quality %.6e -> %.6e",
      accepted,
      acceptance_reason(current, candidate, weights),
      n_unsatisfied_constraints(current),
      n_unsatisfied_constraints(candidate),
      static_cast<double>(current.get_total_excess()),
      static_cast<double>(candidate.get_total_excess()),
      static_cast<double>(get_tardiness_penalty(current)),
      static_cast<double>(get_tardiness_penalty(candidate)),
      static_cast<double>(current.get_quality(weights)),
      static_cast<double>(candidate.get_quality(weights)));
  }
#endif

  bool accept_feasibility_move(solution_t<i_t, f_t>& current,
                               solution_t<i_t, f_t>& candidate,
                               const weight_t<i_t, f_t>& weights) const
  {
    // Feasibility-phase acceptance (Guided-Local-Search style): accept the candidate if it
    // reduces the number of unsatisfied constraints OR lowers the tardiness-weighted penalty
    // sum. Because persistently violated constraints accumulate tardiness (their weight grows),
    // this rule lets the search trade one set of violations for another and escape the plateaus
    // that a strict descent on the raw unsatisfied count gets stuck in.
    const i_t current_unsat   = n_unsatisfied_constraints(current);
    const i_t candidate_unsat = n_unsatisfied_constraints(candidate);
    if (candidate_unsat < current_unsat) { return true; }

    const f_t current_score   = get_weighted_violation_count(current);
    const f_t candidate_score = get_weighted_violation_count(candidate);
    if (candidate_score + OBJECTIVE_EPSILON < current_score) { return true; }
    // On ties in weighted violation count, prefer the state that is numerically closer to
    // feasibility (smaller total excess), which is easier for the next repair to finish off.
    if (candidate_score <= current_score + OBJECTIVE_EPSILON &&
        candidate.get_total_excess() + OBJECTIVE_EPSILON < current.get_total_excess()) {
      return true;
    }
    return false;
  }

 public:
  // Pick the seed variable for the related-ruin expansion. With high probability the seed is
  // drawn from the integer variables that participate in a currently violated constraint, so
  // the ruined set covers infeasibility and the repair can reduce the number of unsatisfied
  // constraints. Otherwise (or when the solution is feasible) it falls back to uniform random.
  i_t pick_seed_var(solution_t<i_t, f_t>& sol)
  {
    const i_t n_integer_vars = problem_ptr->n_integer_vars;
    auto stream              = sol.handle_ptr->get_stream();
    std::uniform_int_distribution<i_t> uniform_dist(0, n_integer_vars - 1);

    std::uniform_real_distribution<double> coin(0., 1.);
    if (coin(rng) < feasibility_lns_config_t::violated_seed_probability) {
      constexpr i_t block_size = 256;
      const i_t grid_size = std::min<i_t>(4096, (n_integer_vars + block_size - 1) / block_size);
      lns_compute_violation_seed_scores_kernel<i_t, f_t><<<grid_size, block_size, 0, stream>>>(
        sol.view(), problem_ptr->integer_indices.data(), n_integer_vars, candidate_scores.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      auto pool_end = thrust::copy_if(sol.handle_ptr->get_thrust_policy(),
                                      thrust::make_counting_iterator<i_t>(0),
                                      thrust::make_counting_iterator<i_t>(n_integer_vars),
                                      violated_seed_pool.begin(),
                                      lns_negative_score_t<i_t, f_t>{candidate_scores.data()});
      const i_t violated_count = static_cast<i_t>(pool_end - violated_seed_pool.begin());
      if (violated_count > 0) {
        std::uniform_int_distribution<i_t> pool_dist(0, violated_count - 1);
        const i_t integer_pos = violated_seed_pool.element(pool_dist(rng), stream);
        return integer_indices_host[integer_pos];
      }
    }
    return integer_indices_host[uniform_dist(rng)];
  }

  i_t select_related_ruin_vars(solution_t<i_t, f_t>& sol, i_t target_ruin_count)
  {
    const i_t n_integer_vars = problem_ptr->n_integer_vars;
    if (n_integer_vars == 0) { return 0; }

    // TODO do target_count as small as possible. 10-20 variables
    const i_t target_count   = target_ruin_count <= 0
                                 ? static_cast<i_t>(feasibility_lns_config_t::ruin_count)
                                 : target_ruin_count;
    const i_t selected_count = std::min(std::max<i_t>(2, target_count), n_integer_vars / 2);
    auto stream              = sol.handle_ptr->get_stream();
    if (selected_count <= 0) { return 0; }

    thrust::fill(sol.handle_ptr->get_thrust_policy(),
                 ruined_var_flags.begin(),
                 ruined_var_flags.end(),
                 i_t{0});

    const i_t seed_var = pick_seed_var(sol);
#if LNS_DEBUG
    auto seed_kernel_timer = start_kernel_timer(stream);
#endif
    lns_seed_ruin_kernel<i_t>
      <<<1, 1, 0, stream>>>(seed_var, ruined_var_flags.data(), ruin_vars.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
#if LNS_DEBUG
    log_kernel_timer(seed_kernel_timer, "lns_seed_ruin_kernel", stream);
#endif
    if (selected_count <= 1) { return 1; }

    const i_t neighbor_begin = related_neighbor_offsets_host[seed_var];
    const i_t neighbor_end   = related_neighbor_offsets_host[seed_var + 1];
    const i_t neighbor_count = neighbor_end - neighbor_begin;
#if LNS_DEBUG
    CUOPT_LOG_INFO(
      "Standalone LNS related ruin seed: var %d, selected target %d, "
      "neighbor candidates %d",
      seed_var,
      selected_count,
      neighbor_count);
#endif
    if (neighbor_count <= 0) { return 1; }

    static constexpr i_t warp_size = 32;
    const i_t warp_grid_size       = std::min<i_t>(
      4096, (neighbor_count + (block_size / warp_size) - 1) / (block_size / warp_size));
#if LNS_DEBUG
    auto score_kernel_timer = start_kernel_timer(stream);
#endif
    lns_compute_seed_neighbor_scores_kernel<i_t, f_t>
      <<<warp_grid_size, block_size, 0, stream>>>(sol.assignment.data(),
                                                  seed_var,
                                                  related_neighbor_vars.data() + neighbor_begin,
                                                  related_pair_offsets.data() + neighbor_begin,
                                                  related_pair_coefficients.data(),
                                                  neighbor_count,
                                                  ruined_var_flags.data(),
                                                  candidate_scores.data(),
                                                  related_candidate_vars.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
#if LNS_DEBUG
    log_kernel_timer(score_kernel_timer, "lns_compute_seed_neighbor_scores_kernel", stream);
#endif

    thrust::sort_by_key(sol.handle_ptr->get_thrust_policy(),
                        candidate_scores.begin(),
                        candidate_scores.begin() + neighbor_count,
                        related_candidate_vars.begin());

    const i_t additional_count = std::min(selected_count - 1, neighbor_count);
#if LNS_DEBUG
    auto mark_neighbor_kernel_timer = start_kernel_timer(stream);
#endif
    thrust::for_each_n(
      sol.handle_ptr->get_thrust_policy(),
      thrust::counting_iterator<i_t>(0),
      additional_count,
      [ruined_var_flags_ptr       = ruined_var_flags.data(),
       ruin_vars_ptr              = ruin_vars.data(),
       related_candidate_vars_ptr = related_candidate_vars.data()] __device__(i_t idx) {
        i_t var                   = related_candidate_vars_ptr[idx];
        ruined_var_flags_ptr[var] = 1;
        ruin_vars_ptr[var]        = 1;
      });
    RAFT_CUDA_TRY(cudaPeekAtLastError());
#if LNS_DEBUG
    log_kernel_timer(mark_neighbor_kernel_timer, "lns_mark_selected_neighbor_ruin", stream);
    CUOPT_LOG_INFO("Standalone LNS related ruin selected: seed 1, neighbors %d, total %d",
                   additional_count,
                   additional_count + 1);
#endif
    return additional_count + 1;
  }

 private:
  i_t build_vars_to_fix()
  {
    vars_to_fix.resize(problem_ptr->n_integer_vars, problem_ptr->handle_ptr->get_stream());
    auto end = thrust::copy_if(problem_ptr->handle_ptr->get_thrust_policy(),
                               problem_ptr->integer_indices.begin(),
                               problem_ptr->integer_indices.end(),
                               vars_to_fix.begin(),
                               lns_not_ruined_var_t<i_t>{ruined_var_flags.data()});
    return static_cast<i_t>(end - vars_to_fix.begin());
  }

  void unset_ruined_integer_values(solution_t<i_t, f_t>& fixed_sol,
                                   const rmm::device_uvector<i_t>& variable_map)
  {
    constexpr i_t block_size = 256;
    const i_t grid_size =
      std::min<i_t>(4096, (fixed_sol.problem_ptr->n_variables + block_size - 1) / block_size);
    const uint64_t seed = static_cast<uint64_t>(cuopt::seed_generator::get_seed());
#if LNS_DEBUG
    auto unset_kernel_timer = start_kernel_timer(fixed_sol.handle_ptr->get_stream());
#endif
    lns_unset_ruined_values_kernel<i_t, f_t>
      <<<grid_size, block_size, 0, fixed_sol.handle_ptr->get_stream()>>>(
        fixed_sol.view(), variable_map.data(), ruined_var_flags.data(), seed);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
#if LNS_DEBUG
    log_kernel_timer(
      unset_kernel_timer, "lns_unset_ruined_values_kernel", fixed_sol.handle_ptr->get_stream());
#endif
  }

  std::pair<solution_t<i_t, f_t>, bool> repair_current_ruin_set(solution_t<i_t, f_t>& current,
                                                                const weight_t<i_t, f_t>& weights,
                                                                size_t attempt,
                                                                const char* ruin_strategy)
  {
    raft::common::nvtx::range fun_scope("standalone_lns_feasibility_ruin_repair");
    const i_t n_vars_to_fix = build_vars_to_fix();
    const i_t ruin_count    = problem_ptr->n_integer_vars - n_vars_to_fix;
    if (ruin_count < 1) { return std::make_pair(solution_t<i_t, f_t>(current), false); }
    vars_to_fix.resize(n_vars_to_fix, current.handle_ptr->get_stream());
#if LNS_DEBUG
    CUOPT_LOG_INFO("Standalone LNS repair attempt %lu: strategy %s, ruin %d, fix %d",
                   static_cast<unsigned long>(attempt),
                   ruin_strategy,
                   ruin_count,
                   n_vars_to_fix);
    log_violation_summary(current, "Standalone LNS original before repair");
#endif

    solution_t<i_t, f_t> offspring(current);
#if LNS_DEBUG
    const auto fix_start = std::chrono::steady_clock::now();
#endif
    auto [fixed_problem, fixed_assignment, variable_map] = offspring.fix_variables(vars_to_fix);
#if LNS_DEBUG
    const auto fix_end  = std::chrono::steady_clock::now();
    const double fix_ms = std::chrono::duration<double, std::milli>(fix_end - fix_start).count();
    CUOPT_LOG_INFO(
      "Standalone LNS fixed problem: vars %d -> %d, constraints %d -> %d, "
      "nnz %d -> %d, fix_variables %.3f ms",
      current.problem_ptr->n_variables,
      fixed_problem.n_variables,
      current.problem_ptr->n_constraints,
      fixed_problem.n_constraints,
      current.problem_ptr->nnz,
      fixed_problem.nnz,
      fix_ms);
#endif
    rmm::device_uvector<f_t> old_assignment(offspring.assignment,
                                            offspring.handle_ptr->get_stream());

    offspring.assignment  = std::move(fixed_assignment);
    offspring.problem_ptr = &fixed_problem;
    unset_ruined_integer_values(offspring, variable_map);
    cuopt_func_call(offspring.test_variable_bounds(false));
#if LNS_DEBUG
    log_violation_summary(offspring, "Standalone LNS fixed after unset");
#endif

    const auto old_repair_iterations = constraint_prop.max_n_failed_repair_iterations;
    constraint_prop.max_n_failed_repair_iterations =
      feasibility_lns_config_t::repair_max_failed_iterations;
    const f_t cp_remaining_before = lns_timer.remaining_time();
    // Bound each repair to a small slice so the ruin/repair loop iterates fast instead of
    // letting a single constraint-propagation call consume all remaining time.
    const f_t cp_budget =
      std::min<f_t>(feasibility_lns_config_t::repair_time_limit, lns_timer.remaining_time());
    timer_t cp_timer(cp_budget);
    constraint_prop.apply_round(
      offspring, feasibility_lns_config_t::lp_after_bounds_prop_time_limit, cp_timer);
    constraint_prop.max_n_failed_repair_iterations = old_repair_iterations;
#if LNS_DEBUG
    CUOPT_LOG_INFO("Standalone LNS constraint propagation: feasible %d, remaining %.6e -> %.6e",
                   offspring.get_feasible(),
                   static_cast<double>(cp_remaining_before),
                   static_cast<double>(lns_timer.remaining_time()));
    log_violation_summary(offspring, "Standalone LNS fixed after constraint propagation");
#endif

    offspring.handle_ptr->sync_stream();
    offspring.problem_ptr = current.problem_ptr;
    fixed_assignment      = std::move(offspring.assignment);
    offspring.assignment  = std::move(old_assignment);
    offspring.unfix_variables(fixed_assignment, variable_map);
    offspring.compute_feasibility();
#if LNS_DEBUG
    log_violation_summary(offspring, "Standalone LNS original after unfix");
    log_ruined_assignment_delta(current, offspring, "Standalone LNS after unfix");
    const i_t lp_unsat_before  = n_unsatisfied_constraints(offspring);
    const f_t lp_excess_before = offspring.get_total_excess();
#endif
    lp_polish_with_integers_fixed(offspring);
#if LNS_DEBUG
    CUOPT_LOG_INFO("Standalone LNS LP polish: unsat %d -> %d, excess %.6e -> %.6e",
                   lp_unsat_before,
                   n_unsatisfied_constraints(offspring),
                   static_cast<double>(lp_excess_before),
                   static_cast<double>(offspring.get_total_excess()));
    log_violation_summary(offspring, "Standalone LNS original after LP polish");
    log_ruined_assignment_delta(current, offspring, "Standalone LNS after LP polish");
#endif

    bool accepted = accept_feasibility_move(current, offspring, weights);
    update_constraint_tardiness(offspring);
#if LNS_DEBUG
    log_acceptance_diagnostics(current, offspring, weights, accepted);
#endif
    if (accepted) {
#if LNS_DEBUG
      CUOPT_LOG_INFO("Standalone LNS feasibility accepted: unsat %d -> %d, excess %.6e -> %.6e",
                     n_unsatisfied_constraints(current),
                     n_unsatisfied_constraints(offspring),
                     current.get_total_excess(),
                     offspring.get_total_excess());
#endif
      return std::make_pair(std::move(offspring), true);
    }
    return std::make_pair(std::move(offspring), false);
  }

  std::pair<solution_t<i_t, f_t>, bool> ruin_repair(solution_t<i_t, f_t>& current,
                                                    const weight_t<i_t, f_t>& weights,
                                                    i_t target_ruin_count,
                                                    size_t attempt)
  {
    current.compute_feasibility();
    ensure_constraint_tardiness_size(current);
    update_constraint_tardiness(current);

    const i_t ruin_count = select_related_ruin_vars(current, target_ruin_count);
    if (ruin_count < 2) { return std::make_pair(solution_t<i_t, f_t>(current), false); }
    return repair_current_ruin_set(current, weights, attempt, "related");
  }

  std::pair<solution_t<i_t, f_t>, bool> violated_constraint_ruin_repair(
    solution_t<i_t, f_t>& current, const weight_t<i_t, f_t>& weights, size_t attempt)
  {
    raft::common::nvtx::range fun_scope("standalone_lns_feasibility_violated_constraint_ruin");
    current.compute_feasibility();
    ensure_constraint_tardiness_size(current);
    update_constraint_tardiness(current);

    auto stream = current.handle_ptr->get_stream();
    thrust::fill(current.handle_ptr->get_thrust_policy(),
                 ruined_var_flags.begin(),
                 ruined_var_flags.end(),
                 i_t{0});
    constexpr i_t block_size = 256;
    const i_t grid_size =
      std::min<i_t>(4096, (current.problem_ptr->n_constraints + block_size - 1) / block_size);
#if LNS_DEBUG
    auto mark_kernel_timer = start_kernel_timer(stream);
#endif
    lns_mark_violated_constraint_integer_vars_kernel<i_t, f_t>
      <<<grid_size, block_size, 0, stream>>>(current.view(),
                                             ruined_var_flags.data(),
                                             current.problem_ptr->tolerances.relative_tolerance);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
#if LNS_DEBUG
    log_kernel_timer(mark_kernel_timer, "lns_mark_violated_constraint_integer_vars_kernel", stream);
    const i_t marked_count = thrust::count(current.handle_ptr->get_thrust_policy(),
                                           ruined_var_flags.begin(),
                                           ruined_var_flags.end(),
                                           i_t{1});
    CUOPT_LOG_INFO("Standalone LNS violated-constraint ruin selected %d integer variables",
                   marked_count);
#endif
    return repair_current_ruin_set(current, weights, attempt, "violated_constraint");
  }

  // Run cuOpt's Feasibility Jump (feasibility mode, with weight updates) on the given solution
  // for a bounded budget. Used to build a strong starting point and to close out near-feasible
  // incumbents that the bounds-propagation repair would otherwise reduce too slowly.
  bool fj_polish(solution_t<i_t, f_t>& sol, f_t time_limit, const char* source)
  {
    if (time_limit <= 0.) { return sol.get_feasible(); }
    // Reset FJ constraint weights to the same baseline (1.0) that a fresh FJ run uses, before
    // each burst. do_fj_solve updates weights in place and the LNS calls it many times over a
    // run; without a reset the weights accumulate across calls and eventually distort FJ's
    // landscape, making later polishes ineffective.
    ls.fj.reset_weights(sol.handle_ptr->get_stream(), f_t{1});
    ls.fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
    ls.fj.settings.n_of_minimums_for_exit = 500;
    ls.fj.settings.update_weights         = true;
    ls.fj.settings.feasibility_run        = true;
    ls.do_fj_solve(sol, ls.fj, time_limit, source);
    return sol.compute_feasibility();
  }

  // Mini-MIP repair: free the integer variables that appear in violated constraints, fix every
  // other integer variable, and solve the residual sub-MIP exactly with dual-simplex branch and
  // bound under a strict time budget. This resolves the "stuck at a few violated constraints"
  // plateaus that the bounds-propagation and FJ repairs grind on indefinitely. Mirrors the
  // sub-MIP recombiner. Returns true if it produced a feasible solution (written into best).
  bool sub_mip_repair(solution_t<i_t, f_t>& best_solution)
  {
    namespace dual_simplex = cuopt::linear_programming::dual_simplex;
    // The host-side B&B build cost scales with the full problem size, so only attempt this on
    // small models where it is cheap; on large models it would burn the whole time budget.
    if (problem_ptr->n_constraints >
          static_cast<i_t>(feasibility_lns_config_t::sub_mip_repair_max_constraints) ||
        problem_ptr->n_variables >
          static_cast<i_t>(feasibility_lns_config_t::sub_mip_repair_max_problem_vars)) {
      return false;
    }
    best_solution.compute_feasibility();
    auto stream = best_solution.handle_ptr->get_stream();

    thrust::fill(best_solution.handle_ptr->get_thrust_policy(),
                 ruined_var_flags.begin(),
                 ruined_var_flags.end(),
                 i_t{0});
    constexpr i_t block_size = 256;
    const i_t grid_size =
      std::min<i_t>(4096, (best_solution.problem_ptr->n_constraints + block_size - 1) / block_size);
    lns_mark_violated_constraint_integer_vars_kernel<i_t, f_t><<<grid_size, block_size, 0, stream>>>(
      best_solution.view(),
      ruined_var_flags.data(),
      best_solution.problem_ptr->tolerances.relative_tolerance);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    const i_t n_free = thrust::count(best_solution.handle_ptr->get_thrust_policy(),
                                     ruined_var_flags.begin(),
                                     ruined_var_flags.end(),
                                     i_t{1});
    if (n_free < 1 ||
        n_free > static_cast<i_t>(feasibility_lns_config_t::sub_mip_repair_max_free_vars)) {
      return false;
    }
    const i_t n_vars_to_fix = build_vars_to_fix();
    vars_to_fix.resize(n_vars_to_fix, stream);

    solution_t<i_t, f_t> offspring(best_solution);
    auto [fixed_problem, fixed_assignment, variable_map] = offspring.fix_variables(vars_to_fix);

    pdlp_initial_scaling_strategy_t<i_t, f_t> scaling(
      fixed_problem.handle_ptr,
      fixed_problem,
      context.settings.hyper_params.default_l_inf_ruiz_iterations,
      static_cast<f_t>(context.settings.hyper_params.default_alpha_pock_chambolle_rescaling),
      fixed_problem.reverse_coefficients,
      fixed_problem.reverse_offsets,
      fixed_problem.reverse_constraints,
      nullptr,
      context.settings.hyper_params,
      static_cast<i_t>(1),
      true);
    scaling.scale_problem();
    fixed_problem.presolve_data.reset_additional_vars(fixed_problem, offspring.handle_ptr);
    fixed_problem.presolve_data.initialize_var_mapping(fixed_problem, offspring.handle_ptr);
    trivial_presolve(fixed_problem);
    fixed_problem.check_problem_representation(true);
    if (fixed_problem.n_integer_vars <= 8) { return false; }

    sub_mip_solution_vector.clear();
    dual_simplex::user_problem_t<i_t, f_t> bb_problem(offspring.handle_ptr);
    dual_simplex::simplex_solver_settings_t<i_t, f_t> bb_settings;
    fixed_problem.get_host_user_problem(bb_problem);
    dual_simplex::mip_solution_t<i_t, f_t> bb_solution(1);
    bb_solution.resize(bb_problem.num_cols);
    bb_settings.time_limit           = std::min<f_t>(
      feasibility_lns_config_t::sub_mip_repair_time_limit, lns_timer.remaining_time());
    bb_settings.print_presolve_stats = false;
    bb_settings.absolute_mip_gap_tol = context.settings.tolerances.absolute_mip_gap;
    bb_settings.relative_mip_gap_tol = context.settings.tolerances.relative_mip_gap;
    bb_settings.integer_tol          = context.settings.tolerances.integrality_tolerance;
    bb_settings.num_threads          = 1;
    bb_settings.reliability_branching = 0;
    bb_settings.max_cut_passes        = 0;
    bb_settings.clique_cuts           = 0;
    bb_settings.sub_mip               = 1;
    bb_settings.strong_branching_simplex_iteration_limit = 200;
    bb_settings.log.log               = false;
    bb_settings.solution_callback =
      [this](std::vector<f_t>& sol, f_t /*objective*/) { sub_mip_solution_vector.push_back(sol); };
    dual_simplex::probing_implied_bound_t<i_t, f_t> empty_probing(bb_problem.num_cols);
    dual_simplex::branch_and_bound_t<i_t, f_t> bb(
      bb_problem, bb_settings, dual_simplex::tic(), empty_probing);
    bb.solve(bb_solution);
    if (sub_mip_solution_vector.empty()) { return false; }

    rmm::device_uvector<f_t> post(bb_solution.x.size(), offspring.handle_ptr->get_stream());
    raft::copy(post.data(),
               bb_solution.x.data(),
               bb_solution.x.size(),
               offspring.handle_ptr->get_stream());
    fixed_problem.post_process_assignment(post, false);
    offspring.handle_ptr->sync_stream();
    std::swap(fixed_assignment, post);
    rmm::device_uvector<f_t> dummy(0, offspring.handle_ptr->get_stream());
    scaling.unscale_solutions(fixed_assignment, dummy);
    offspring.unfix_variables(fixed_assignment, variable_map);
    offspring.clamp_within_bounds();
    const bool feasible = offspring.compute_feasibility();
    if (feasible || is_better_feasibility_state(offspring, best_solution)) {
      best_solution.copy_from(offspring);
      return feasible;
    }
    return false;
  }

  void lp_polish_with_integers_fixed(
    solution_t<i_t, f_t>& solution,
    f_t time_limit = static_cast<f_t>(feasibility_lns_config_t::lp_after_bounds_prop_time_limit))
  {
    if (solution.get_feasible() || time_limit <= 0.) { return; }
    if (solution.problem_ptr->n_variables == solution.problem_ptr->n_integer_vars) { return; }

    // The LP polish (continuous vars re-optimized with integers fixed) can occasionally
    // return an iterate that violates more constraints than the input. Snapshot first and
    // revert if the polish does not improve the feasibility state.
    solution_t<i_t, f_t> pre_polish(solution);
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit            = time_limit;
    lp_settings.tolerance             = solution.problem_ptr->tolerances.absolute_tolerance;
    lp_settings.save_state            = false;
    lp_settings.return_first_feasible = true;
    run_lp_with_vars_fixed(*solution.problem_ptr,
                           solution,
                           solution.problem_ptr->integer_indices,
                           lp_settings,
                           static_cast<bound_presolve_t<i_t, f_t>*>(nullptr));
    solution.compute_feasibility();
    if (!solution.get_feasible() && !is_better_feasibility_state(solution, pre_polish)) {
      solution.copy_from(pre_polish);
    }
  }

  mip_solver_context_t<i_t, f_t>& context;
  local_search_t<i_t, f_t>& ls;
  problem_t<i_t, f_t>* problem_ptr;
  constraint_prop_t<i_t, f_t>& constraint_prop;
  rmm::device_uvector<f_t> candidate_scores;
  rmm::device_uvector<i_t> related_candidate_vars;
  rmm::device_uvector<i_t> violated_seed_pool;
  rmm::device_uvector<i_t> related_neighbor_offsets;
  rmm::device_uvector<i_t> related_neighbor_vars;
  rmm::device_uvector<i_t> related_pair_offsets;
  rmm::device_uvector<f_t2> related_pair_coefficients;
  rmm::device_uvector<i_t> ruin_vars;
  rmm::device_uvector<i_t> vars_to_fix;
  rmm::device_uvector<i_t> ruined_var_flags;
  rmm::device_uvector<i_t> constraint_tardiness;
  std::vector<i_t> integer_indices_host;
  std::vector<i_t> related_neighbor_offsets_host;
  std::vector<std::vector<f_t>> sub_mip_solution_vector;
  std::mt19937 rng;
  timer_t lns_timer{0.};
};

}  // namespace cuopt::linear_programming::detail
