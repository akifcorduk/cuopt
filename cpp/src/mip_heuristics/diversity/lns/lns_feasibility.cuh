/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/diversity/recombiners/recombiner_configs.hpp>
#include <mip_heuristics/local_search/rounding/constraint_prop.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <utility>
#include <vector>

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
      difference_sum +=
        abs(coeff_pair.x * seed_value - coeff_pair.y * assignment[candidate_var]);
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
__global__ void lns_compute_violation_seed_scores_kernel(
  typename solution_t<i_t, f_t>::view_t sol,
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
    const i_t var_idx          = candidate_vars[selected_pos];
    ruined_vars[var_idx]       = 1;
    ruin_vars[selected_pos + 1] = var_idx;
  }
}

template <typename i_t, typename f_t>
__device__ f_t lns_ruined_value(typename problem_t<i_t, f_t>::view_t pb,
                                i_t fixed_var_idx,
                                i_t original_var_idx,
                                i_t attempt_seed)
{
  const auto bounds = pb.variable_bounds[fixed_var_idx];
  const f_t lb      = get_lower(bounds);
  const f_t ub      = get_upper(bounds);
  const f_t int_tol = pb.tolerances.integrality_tolerance;

  f_t value = f_t{0.5};
  if (isfinite(lb) && isfinite(ub)) {
    if (ub - lb <= int_tol) { return lb; }

    const f_t rounded_lb = ceil(lb - int_tol);
    const f_t rounded_ub = floor(ub + int_tol);
    if (rounded_lb <= rounded_ub &&
        rounded_lb >= static_cast<f_t>(std::numeric_limits<i_t>::lowest()) &&
        rounded_ub <= static_cast<f_t>(std::numeric_limits<i_t>::max())) {
      const i_t int_lb = static_cast<i_t>(rounded_lb);
      const i_t int_ub = static_cast<i_t>(rounded_ub);
      const i_t span   = int_ub - int_lb + 1;
      unsigned int hash =
        static_cast<unsigned int>(original_var_idx) * 1664525u +
        static_cast<unsigned int>(attempt_seed + 1013904223u);
      hash ^= hash >> 16;
      hash *= 2246822519u;
      hash ^= hash >> 13;
      value = static_cast<f_t>(int_lb + static_cast<i_t>(hash % static_cast<unsigned int>(span)));
    } else {
      value = (lb + ub) / 2;
    }
  } else if (isfinite(lb)) {
    value = lb + f_t{0.5};
  } else if (isfinite(ub)) {
    value = ub - f_t{0.5};
  }

  return value < lb ? lb : (value > ub ? ub : value);
}

template <typename i_t, typename f_t>
__global__ void lns_unset_ruined_values_kernel(typename solution_t<i_t, f_t>::view_t fixed_sol,
                                               const i_t* variable_map,
                                               const i_t* ruined_vars,
                                               i_t attempt_seed)
{
  for (i_t fixed_var_idx = blockIdx.x * blockDim.x + threadIdx.x;
       fixed_var_idx < fixed_sol.problem.n_variables;
       fixed_var_idx += blockDim.x * gridDim.x) {
    const i_t original_var_idx = variable_map[fixed_var_idx];
    if (ruined_vars[original_var_idx] == 0) { continue; }
    fixed_sol.assignment[fixed_var_idx] =
      lns_ruined_value<i_t, f_t>(fixed_sol.problem, fixed_var_idx, original_var_idx, attempt_seed);
  }
}

template <typename i_t, typename f_t>
class lns_feasibility_t {
 public:
  lns_feasibility_t(mip_solver_context_t<i_t, f_t>& context_,
                    constraint_prop_t<i_t, f_t>& constraint_prop_)
    : context(context_),
      problem_ptr(context.problem_ptr),
      constraint_prop(constraint_prop_),
      candidate_scores(problem_ptr->n_integer_vars, problem_ptr->handle_ptr->get_stream()),
      related_candidate_vars(problem_ptr->n_integer_vars, problem_ptr->handle_ptr->get_stream()),
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

    timer_t lns_timer(time_limit);
    bool accepted_any = seed_repair(solution, lns_timer);
    if (solution.get_feasible()) { return finalize_feasible_solution(solution); }
    solution_t<i_t, f_t> best_solution(solution);

    static constexpr i_t ruin_schedule[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
    static constexpr size_t ruin_schedule_size =
      sizeof(ruin_schedule) / sizeof(ruin_schedule[0]);
    weight_t<i_t, f_t> weights(problem_ptr->n_constraints, problem_ptr->handle_ptr);
    size_t attempted_repairs = 0;
    size_t accepted_repairs  = 0;
    for (size_t attempt = 0; attempt < feasibility_lns_config_t::max_attempts; ++attempt) {
      if (lns_timer.check_time_limit()) { break; }
      const bool use_violated_constraint_ruin =
        n_unsatisfied_constraints(solution) <=
          static_cast<i_t>(feasibility_lns_config_t::violated_constraint_ruin_unsat_limit) &&
        attempt % 2 == 0;
      const i_t ruin_count = ruin_schedule[attempt % ruin_schedule_size];
      auto [candidate, accepted] =
        use_violated_constraint_ruin
          ? violated_constraint_ruin_repair(solution, weights, attempt)
          : ruin_repair(solution, weights, ruin_count, attempt);
      ++attempted_repairs;
      if (!accepted) { continue; }

      solution.copy_from(candidate);
      accepted_any = true;
      ++accepted_repairs;
      if (is_better_feasibility_state(solution, best_solution)) { best_solution.copy_from(solution); }
      if (solution.get_feasible()) { return finalize_feasible_solution(solution); }
    }
    if (is_better_feasibility_state(best_solution, solution)) { solution.copy_from(best_solution); }
    if (solution.get_feasible()) { finalize_feasible_solution(solution); }
    if (!solution.get_feasible()) { log_violation_summary(solution, "Standalone LNS final"); }
    CUOPT_LOG_INFO("Standalone LNS feasibility finished: feasible %d, unsat %d, excess %.6e, "
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

    CUOPT_LOG_DEBUG("Standalone LNS relatedness cache: variables %d, integer vars %d, "
                    "neighbor pairs %lu, coefficient pairs %lu",
                    n_variables,
                    n_integer_vars,
                    static_cast<unsigned long>(h_neighbor_vars.size()),
                    static_cast<unsigned long>(h_pair_coefficients.size()));
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
    if (candidate_unsat < incumbent_unsat) { return true; }
    if (candidate_unsat > incumbent_unsat) { return false; }

    return candidate.get_total_excess() + OBJECTIVE_EPSILON < incumbent.get_total_excess();
  }

  bool seed_repair(solution_t<i_t, f_t>& solution, timer_t& lns_timer)
  {
    solution_t<i_t, f_t> repaired(solution);
    const auto old_repair_iterations = constraint_prop.max_n_failed_repair_iterations;
    constraint_prop.max_n_failed_repair_iterations =
      feasibility_lns_config_t::n_repair_iterations;
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

  bool use_seed_lp_polish() const
  {
    return problem_ptr->n_constraints > problem_ptr->n_variables;
  }

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

  void log_violation_summary(solution_t<i_t, f_t>& sol, const char* label) const
  {
    sol.compute_feasibility();
    const auto policy = sol.handle_ptr->get_thrust_policy();
    const auto view   = sol.view();
    const f_t rel_tol = sol.problem_ptr->tolerances.relative_tolerance;
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
    auto worst_it =
      thrust::max_element(policy,
                          thrust::make_counting_iterator<i_t>(0),
                          thrust::make_counting_iterator<i_t>(sol.problem_ptr->n_constraints),
                          lns_worse_constraint_t<i_t, f_t>{view});
    const i_t worst_idx = static_cast<i_t>(*worst_it);
    const auto stream   = sol.handle_ptr->get_stream();
    CUOPT_LOG_INFO(
      "%s violation summary: lower %d, upper %d, worst row %d, lower_excess %.6e, "
      "upper_excess %.6e, activity %.6e, lb %.6e, ub %.6e, row_nnz %d",
      label,
      lower_count,
      upper_count,
      worst_idx,
      static_cast<double>(sol.lower_excess.element(worst_idx, stream)),
      static_cast<double>(sol.upper_excess.element(worst_idx, stream)),
      static_cast<double>(sol.constraint_value.element(worst_idx, stream)),
      static_cast<double>(sol.problem_ptr->constraint_lower_bounds.element(worst_idx, stream)),
      static_cast<double>(sol.problem_ptr->constraint_upper_bounds.element(worst_idx, stream)),
      sol.problem_ptr->offsets.element(worst_idx + 1, stream) -
        sol.problem_ptr->offsets.element(worst_idx, stream));
  }

  bool accept_feasibility_move(solution_t<i_t, f_t>& current,
                               solution_t<i_t, f_t>& candidate,
                               const weight_t<i_t, f_t>& weights) const
  {
    const i_t current_unsat   = n_unsatisfied_constraints(current);
    const i_t candidate_unsat = n_unsatisfied_constraints(candidate);
    if (candidate_unsat < current_unsat) { return true; }
    if (candidate_unsat > current_unsat) { return false; }

    const f_t current_excess   = current.get_total_excess();
    const f_t candidate_excess = candidate.get_total_excess();
    if (candidate_excess + OBJECTIVE_EPSILON < current_excess) { return true; }
    if (current_excess + OBJECTIVE_EPSILON < candidate_excess) { return false; }

    const f_t current_tardiness_penalty   = get_tardiness_penalty(current);
    const f_t candidate_tardiness_penalty = get_tardiness_penalty(candidate);
    if (candidate_tardiness_penalty + OBJECTIVE_EPSILON < current_tardiness_penalty) {
      return true;
    }
    if (current_tardiness_penalty + OBJECTIVE_EPSILON < candidate_tardiness_penalty) {
      return false;
    }

    return candidate.get_quality(weights) + OBJECTIVE_EPSILON < current.get_quality(weights);
  }

  i_t select_related_ruin_vars(solution_t<i_t, f_t>& sol, i_t target_ruin_count)
  {
    const i_t n_integer_vars = problem_ptr->n_integer_vars;
    if (n_integer_vars == 0) { return 0; }
    const i_t* integer_indices = problem_ptr->integer_indices.data();

    const i_t target_count = target_ruin_count <= 0
                               ? static_cast<i_t>(feasibility_lns_config_t::ruin_count)
                               : target_ruin_count;
    const i_t selected_count = std::min(std::max<i_t>(2, target_count), n_integer_vars/2);
    auto stream              = sol.handle_ptr->get_stream();
    if (selected_count <= 0) { return 0; }

    thrust::fill(sol.handle_ptr->get_thrust_policy(),
                 ruined_var_flags.begin(),
                 ruined_var_flags.end(),
                 i_t{0});

    i_t seed_pos = 0;
    std::uniform_int_distribution<i_t> seed_dist(0, n_integer_vars - 1);
    seed_pos = seed_dist(rng);

    const i_t seed_var = integer_indices_host[seed_pos];
    lns_seed_ruin_kernel<i_t>
      <<<1, 1, 0, stream>>>(seed_var, ruined_var_flags.data(), ruin_vars.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    if (selected_count <= 1) { return 1; }

    const i_t neighbor_begin = related_neighbor_offsets_host[seed_var];
    const i_t neighbor_end   = related_neighbor_offsets_host[seed_var + 1];
    const i_t neighbor_count = neighbor_end - neighbor_begin;
    if (neighbor_count <= 0) { return 1; }

    static constexpr i_t warp_size = 32;
    const i_t warp_grid_size       = std::min<i_t>(
      4096, (neighbor_count + (block_size / warp_size) - 1) / (block_size / warp_size));
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

    thrust::sort_by_key(sol.handle_ptr->get_thrust_policy(),
                        candidate_scores.begin(),
                        candidate_scores.begin() + neighbor_count,
                        related_candidate_vars.begin());

    const i_t additional_count = std::min(selected_count - 1, neighbor_count);
    thrust::for_each_n(
        sol.handle_ptr->get_thrust_policy(),
        thrust::counting_iterator<i_t>(0),
        additional_count,
        [ruined_var_flags_ptr = ruined_var_flags.data(),
         ruin_vars_ptr        = ruin_vars.data(),
         related_candidate_vars_ptr = related_candidate_vars.data()] __device__ (i_t idx) {
            i_t var = related_candidate_vars_ptr[idx];
            ruined_var_flags_ptr[var] = 1;
            ruin_vars_ptr[var] = 1;
        }
    );
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    return additional_count + 1;
  }

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
                                   const rmm::device_uvector<i_t>& variable_map,
                                   size_t attempt)
  {
    constexpr i_t block_size = 256;
    const i_t grid_size =
      std::min<i_t>(4096, (fixed_sol.problem_ptr->n_variables + block_size - 1) / block_size);
    lns_unset_ruined_values_kernel<i_t, f_t>
      <<<grid_size, block_size, 0, fixed_sol.handle_ptr->get_stream()>>>(
        fixed_sol.view(),
        variable_map.data(),
        ruined_var_flags.data(),
        static_cast<i_t>(attempt));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  std::pair<solution_t<i_t, f_t>, bool> repair_current_ruin_set(
    solution_t<i_t, f_t>& current, const weight_t<i_t, f_t>& weights, size_t attempt)
  {
    raft::common::nvtx::range fun_scope("standalone_lns_feasibility_ruin_repair");
    const i_t n_vars_to_fix = build_vars_to_fix();
    const i_t ruin_count    = problem_ptr->n_integer_vars - n_vars_to_fix;
    if (ruin_count < 1) { return std::make_pair(solution_t<i_t, f_t>(current), false); }
    vars_to_fix.resize(n_vars_to_fix, current.handle_ptr->get_stream());

    solution_t<i_t, f_t> offspring(current);
    auto [fixed_problem, fixed_assignment, variable_map] = offspring.fix_variables(vars_to_fix);
    timer_t timer(feasibility_lns_config_t::bounds_prop_time_limit);
    rmm::device_uvector<f_t> old_assignment(offspring.assignment,
                                            offspring.handle_ptr->get_stream());

    offspring.assignment  = std::move(fixed_assignment);
    offspring.problem_ptr = &fixed_problem;
    unset_ruined_integer_values(offspring, variable_map, attempt);
    cuopt_func_call(offspring.test_variable_bounds(false));

    const auto old_repair_iterations               = constraint_prop.max_n_failed_repair_iterations;
    constraint_prop.max_n_failed_repair_iterations = feasibility_lns_config_t::n_repair_iterations;
    constraint_prop.apply_round(
      offspring, feasibility_lns_config_t::lp_after_bounds_prop_time_limit, timer);
    constraint_prop.max_n_failed_repair_iterations = old_repair_iterations;

    offspring.handle_ptr->sync_stream();
    offspring.problem_ptr = current.problem_ptr;
    fixed_assignment      = std::move(offspring.assignment);
    offspring.assignment  = std::move(old_assignment);
    offspring.unfix_variables(fixed_assignment, variable_map);
    offspring.compute_feasibility();
    lp_polish_with_integers_fixed(offspring);

    bool accepted = accept_feasibility_move(current, offspring, weights);
    update_constraint_tardiness(offspring);
    if (accepted) {
      CUOPT_LOG_DEBUG("Standalone LNS feasibility accepted: unsat %d -> %d, excess %.6e -> %.6e",
                      n_unsatisfied_constraints(current),
                      n_unsatisfied_constraints(offspring),
                      current.get_total_excess(),
                      offspring.get_total_excess());
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
    return repair_current_ruin_set(current, weights, attempt);
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
    lns_mark_violated_constraint_integer_vars_kernel<i_t, f_t>
      <<<grid_size, block_size, 0, stream>>>(
        current.view(),
        ruined_var_flags.data(),
        current.problem_ptr->tolerances.relative_tolerance);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    return repair_current_ruin_set(current, weights, attempt);
  }

  void lp_polish_with_integers_fixed(
    solution_t<i_t, f_t>& solution,
    f_t time_limit = static_cast<f_t>(feasibility_lns_config_t::lp_after_bounds_prop_time_limit))
  {
    if (solution.get_feasible() || time_limit <= 0.) {
      return;
    }
    if (solution.problem_ptr->n_variables == solution.problem_ptr->n_integer_vars) { return; }

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
  }

  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;
  constraint_prop_t<i_t, f_t>& constraint_prop;
  rmm::device_uvector<f_t> candidate_scores;
  rmm::device_uvector<i_t> related_candidate_vars;
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
  std::mt19937 rng;
};

}  // namespace cuopt::linear_programming::detail
