/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pdlp/solve.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/linalg/binary_op.cuh>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace cuopt::mathematical_optimization::mip {

// ---------------------------------------------------------------------------
// Probe-based batched feasibility pump
//
// Width is a decision amplifier on a single trajectory rather than a population: every member
// shares one objective and one warm start and differs only by one hard variable fixing (plus that
// fixing's implied-bound closure). The members are ranked and exactly one is committed, so the
// trajectory depth is identical to the single-path FP.
// ---------------------------------------------------------------------------

namespace {

// A candidate hard fixing for one batch member.
template <typename i_t, typename f_t>
struct probe_candidate_t {
  i_t var;
  f_t lower;
  f_t upper;
  f_t ambiguity;     // |lp_value - rounded_value|, the generator's own preference
  i_t implications;  // size of the implied-bound closure, -1 when the cache has no entry
};

}  // namespace

// Scores every batch member against the original problem. Unlike the diversity-cloud kernels this
// includes member 0, because member 0 is the reference step that the probes must beat. Each row
// accumulates straight into its member's totals, so no per-row excess is materialised and the
// reduction over rows is the grid rather than a loop.
template <typename i_t, typename f_t>
__global__ void probe_member_row_excess_kernel(typename problem_t<i_t, f_t>::view_t problem,
                                               const f_t* __restrict__ assignments,
                                               f_t* __restrict__ adjusted_excess,
                                               i_t* __restrict__ violated_counts,
                                               i_t n_vars)
{
  __shared__ f_t shmem[raft::WarpSize];
  const i_t constraint = blockIdx.x;
  const i_t member     = blockIdx.y;
  auto [begin, end]    = problem.range_for_constraint(constraint);
  const f_t* x         = assignments + (size_t)member * (size_t)n_vars;
  f_t activity         = 0.;
  for (i_t p = begin + threadIdx.x; p < end; p += blockDim.x) {
    activity += problem.coefficients[p] * x[problem.variables[p]];
  }
  activity = raft::blockReduce(activity, (char*)shmem);
  if (threadIdx.x == 0) {
    const f_t lower_bound = problem.constraint_lower_bounds[constraint];
    const f_t upper_bound = problem.constraint_upper_bounds[constraint];
    const f_t lower       = max((f_t)0., lower_bound - activity);
    const f_t upper       = max((f_t)0., activity - upper_bound);
    const f_t tolerance   = get_cstr_tolerance<i_t, f_t>(lower_bound,
                                                       upper_bound,
                                                       problem.tolerances.absolute_tolerance,
                                                       problem.tolerances.relative_tolerance);
    const f_t excess      = max((f_t)0., lower - tolerance) + max((f_t)0., upper - tolerance);
    if (excess > (f_t)0.) { atomicAdd(&adjusted_excess[member], excess); }
    if (lower > tolerance || upper > tolerance) { atomicAdd(&violated_counts[member], 1); }
  }
}

template <typename i_t, typename f_t>
__global__ void probe_member_integer_count_kernel(typename problem_t<i_t, f_t>::view_t problem,
                                                  const f_t* __restrict__ assignments,
                                                  i_t* __restrict__ integer_counts,
                                                  i_t n_vars)
{
  __shared__ i_t shmem[raft::WarpSize];
  const i_t member = blockIdx.y;
  const f_t* x     = assignments + (size_t)member * (size_t)n_vars;
  i_t count        = 0;
  for (i_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n_vars; j += blockDim.x * gridDim.x) {
    count += problem.is_integer_var(j) && problem.is_integer(x[j]);
  }
  count = raft::blockReduce(count, (char*)shmem);
  if (threadIdx.x == 0 && count > 0) { atomicAdd(&integer_counts[member], count); }
}

// The trajectory dual is resized against the current projection problem and then broadcast to every
// member. rmm::device_uvector::resize does not initialise the grown tail, and those bits are
// frequently finite garbage that PDLP's own isfinite clamp would let through, so the tail has to be
// zeroed here. The previous size must be captured before the resize; reading it afterwards observes
// exactly the garbage it is meant to remove.
template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::prepare_probe_dual_warm_start(solution_t<i_t, f_t>& solution)
{
  auto stream           = solution.handle_ptr->get_stream();
  const i_t n_constr    = unified_n_constr_total;
  const i_t prev_size   = probe_prev_dual_size;
  last_warm_start_stats = {};

  if (!probe_dual_seeded) {
    probe_prev_dual.resize(n_constr, stream);
    thrust::fill(solution.handle_ptr->get_thrust_policy(),
                 probe_prev_dual.begin(),
                 probe_prev_dual.end(),
                 f_t(0));
    probe_prev_dual_size = n_constr;
    solution.handle_ptr->sync_stream();
    return;
  }

  const i_t carried = std::min(prev_size, n_constr);
  probe_prev_dual.resize(n_constr, stream);
  if (metrics != nullptr) {
    last_warm_start_stats.dual_used   = true;
    last_warm_start_stats.zeroed_tail = (int)(n_constr - carried);
    last_warm_start_stats.nonfinite =
      (int)thrust::count_if(solution.handle_ptr->get_thrust_policy(),
                            probe_prev_dual.data(),
                            probe_prev_dual.data() + carried,
                            [] __device__(f_t x) { return !isfinite(x); });
  }
  f_t* dual = probe_prev_dual.data();
  thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator<i_t>(0),
                   thrust::make_counting_iterator<i_t>(n_constr),
                   [dual, carried] __device__(i_t i) {
                     if (i >= carried || !isfinite(dual[i])) { dual[i] = f_t(0); }
                   });
  probe_prev_dual_size = n_constr;
  solution.handle_ptr->sync_stream();
}

// Selects the probe fixings and expands each into its implied-bound closure. Everything here is on
// the host: the probing cache is a host unordered_map and new_bounds is assembled on the host
// anyway.
template <typename i_t, typename f_t>
i_t feasibility_pump_t<i_t, f_t>::build_probe_new_bounds(
  solution_t<i_t, f_t>& solution,
  i_t max_probes,
  std::vector<std::tuple<i_t, i_t, f_t, f_t>>& new_bounds)
{
  new_bounds.clear();
  last_probing_cache_implications = 0;
  if (max_probes <= 0) { return 0; }

  auto* pb          = solution.problem_ptr;
  auto stream       = solution.handle_ptr->get_stream();
  const f_t int_tol = context.settings.tolerances.integrality_tolerance;
  const f_t abs_tol = context.settings.tolerances.absolute_tolerance;

  // The rounding drives the probe (which row is broken, which way a variable was rounded); the
  // pre-rounding projection supplies the LP values that rank the candidates.
  const auto h_rounding  = cuopt::host_copy(last_rounding, stream);
  const auto h_lp_values = cuopt::host_copy(last_projection, stream);
  solution.handle_ptr->sync_stream();

  auto& probing_cache  = constraint_prop.bounds_update.probing_cache;
  const bool use_cache = !probing_cache.probing_cache.empty() && !pb->original_ids.empty() &&
                         !pb->reverse_original_ids.empty();

  // Cache entry matching the probe value, or nullptr when the cache cannot describe this fixing.
  auto find_cache_entry = [&](i_t var, f_t probe_value) -> const cache_entry_t<i_t, f_t>* {
    if (!use_cache || !probing_cache.contains(*pb, var)) { return nullptr; }
    auto& cache_row = probing_cache.probing_cache[pb->original_ids[var]];
    i_t hit_first   = -1;
    i_t hit_second  = -1;
    for (i_t e = 0; e < 2; ++e) {
      if (cache_row[e].var_to_cached_bound_map.empty()) { continue; }
      cache_row[e].val_interval.fill_cache_hits(e, probe_value, probe_value, hit_first, hit_second);
    }
    if (hit_first < 0) { return nullptr; }
    return &cache_row[hit_first];
  };

  auto implication_count = [&](i_t var, f_t probe_value) -> i_t {
    const auto* entry = find_cache_entry(var, probe_value);
    return entry == nullptr ? -1 : (i_t)entry->var_to_cached_bound_map.size();
  };

  std::vector<probe_candidate_t<i_t, f_t>> candidates;
  std::vector<char> chosen((size_t)unified_n_vars, 0);

  auto push_candidate = [&](i_t var, f_t lower, f_t upper, f_t ambiguity) {
    if (var < 0 || var >= unified_n_vars || chosen[(size_t)var]) { return; }
    // Never emit a fixing outside the variable's own bounds; that is an infeasible member, not a
    // probe.
    if (lower < h_var_lower[(size_t)var] - int_tol || upper > h_var_upper[(size_t)var] + int_tol) {
      return;
    }
    chosen[(size_t)var] = 1;
    candidates.push_back({var, lower, upper, ambiguity, implication_count(var, lower)});
  };

  const bool want_row_repair =
    batch_config.probe_generator == 0 || batch_config.probe_generator == 1;
  const bool want_ambiguous =
    batch_config.probe_generator == 0 || batch_config.probe_generator == 2;

  // Generator 1 - row repair. Nearest rounding drives an equality row to zero whenever every value
  // in it is below one half and can never recover it, which is the documented set-partitioning
  // failure. Fix the most promising variable in the row up instead.
  if (want_row_repair) {
    struct violated_row_t {
      i_t row;
      f_t violation;
    };
    std::vector<violated_row_t> rows;
    for (i_t r = 0; r < unified_n_constr; ++r) {
      const f_t lo = h_base_constraint_lower[(size_t)r];
      const f_t hi = h_base_constraint_upper[(size_t)r];
      if (std::abs(hi - lo) > abs_tol) { continue; }
      f_t activity = 0.;
      for (i_t p = h_csr_offsets[(size_t)r]; p < h_csr_offsets[(size_t)r + 1]; ++p) {
        activity += h_csr_values[(size_t)p] * h_rounding[(size_t)h_csr_indices[(size_t)p]];
      }
      const f_t violation = std::abs(activity - lo);
      if (violation > abs_tol) { rows.push_back({r, violation}); }
    }
    std::sort(rows.begin(), rows.end(), [](const violated_row_t& a, const violated_row_t& b) {
      return a.violation > b.violation;
    });

    for (const auto& row : rows) {
      if ((i_t)candidates.size() >= max_probes) { break; }
      i_t best_var      = -1;
      i_t best_implied  = -2;
      f_t best_lp_value = -std::numeric_limits<f_t>::infinity();
      for (i_t p = h_csr_offsets[(size_t)row.row]; p < h_csr_offsets[(size_t)row.row + 1]; ++p) {
        const i_t var = h_csr_indices[(size_t)p];
        if (!h_probe_is_integer[(size_t)var] || chosen[(size_t)var]) { continue; }
        // Only variables the rounding pushed down can repair the row by going up.
        if (h_rounding[(size_t)var] > h_var_lower[(size_t)var] + int_tol) { continue; }
        const f_t target = h_var_lower[(size_t)var] + 1.;
        if (target > h_var_upper[(size_t)var] + int_tol) { continue; }
        const i_t implied  = implication_count(var, target);
        const f_t lp_value = h_lp_values[(size_t)var];
        // Prefer the fixing that propagates widest; fall back to the argmax LP value when the cache
        // cannot separate the candidates.
        const bool better =
          implied > best_implied || (implied == best_implied && lp_value > best_lp_value);
        if (better) {
          best_var      = var;
          best_implied  = implied;
          best_lp_value = lp_value;
        }
      }
      if (best_var >= 0) {
        const f_t target = h_var_lower[(size_t)best_var] + 1.;
        push_candidate(best_var, target, target, row.violation);
      }
    }
  }

  // Generator 2 - most ambiguous. The integers whose LP value sits farthest from the value nearest
  // rounding picked, fixed to the opposite side.
  if (want_ambiguous && (i_t)candidates.size() < max_probes) {
    std::vector<probe_candidate_t<i_t, f_t>> flips;
    for (i_t var : h_integer_indices_cache) {
      if (chosen[(size_t)var]) { continue; }
      const f_t lp_value = h_lp_values[(size_t)var];
      const f_t rounded  = h_rounding[(size_t)var];
      if (!std::isfinite(lp_value) || !std::isfinite(rounded)) { continue; }
      const f_t ambiguity = std::abs(lp_value - rounded);
      if (ambiguity <= int_tol) { continue; }
      // Opposite side of the rounding: up when it was rounded down, down when it was rounded up.
      const f_t target = lp_value > rounded ? rounded + 1. : rounded - 1.;
      if (target < h_var_lower[(size_t)var] - int_tol ||
          target > h_var_upper[(size_t)var] + int_tol) {
        continue;
      }
      flips.push_back({var, target, target, ambiguity, implication_count(var, target)});
    }
    std::sort(flips.begin(),
              flips.end(),
              [](const probe_candidate_t<i_t, f_t>& a, const probe_candidate_t<i_t, f_t>& b) {
                if (a.implications != b.implications) { return a.implications > b.implications; }
                return a.ambiguity > b.ambiguity;
              });
    for (const auto& flip : flips) {
      if ((i_t)candidates.size() >= max_probes) { break; }
      push_candidate(flip.var, flip.lower, flip.upper, flip.ambiguity);
    }
  }

  if (candidates.empty()) { return 0; }

  // Expand each surviving candidate into its member. Implications are a presolve-time snapshot, so
  // a probe whose closure already contradicts the current bounds is dropped rather than trusted.
  i_t member = 1;
  for (const auto& candidate : candidates) {
    const auto* entry = find_cache_entry(candidate.var, candidate.lower);
    if (entry != nullptr) {
      const i_t conflicts = probing_cache.check_number_of_conflicting_vars(
        h_var_lower, h_var_upper, *entry, int_tol, pb->reverse_original_ids);
      if (conflicts > 0) { continue; }
    }

    // (member, variable) pairs must be unique, so the closure is deduped against the probe variable
    // itself and against anything already emitted for this member.
    std::vector<std::tuple<i_t, i_t, f_t, f_t>> member_bounds;
    member_bounds.push_back({member, candidate.var, candidate.lower, candidate.upper});
    if (entry != nullptr) {
      for (const auto& [original_var, bound] : entry->var_to_cached_bound_map) {
        const i_t var = pb->reverse_original_ids[(size_t)original_var];
        // -1 means the variable was fixed away and no longer exists in the current problem.
        if (var < 0 || var >= unified_n_vars || var == candidate.var) { continue; }
        if (bound.lb > bound.ub) { continue; }
        member_bounds.push_back({member, var, bound.lb, bound.ub});
        ++last_probing_cache_implications;
      }
    }
    std::sort(member_bounds.begin(), member_bounds.end(), [](const auto& a, const auto& b) {
      return std::get<1>(a) < std::get<1>(b);
    });
    member_bounds.erase(
      std::unique(member_bounds.begin(),
                  member_bounds.end(),
                  [](const auto& a, const auto& b) { return std::get<1>(a) == std::get<1>(b); }),
      member_bounds.end());
    new_bounds.insert(new_bounds.end(), member_bounds.begin(), member_bounds.end());
    ++member;
  }

  return member - 1;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::probe_project_onto_polytope(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("probe_project_onto_polytope");
  CUOPT_LOG_INFO("linear projection of fp");
  auto* pb             = solution.problem_ptr;
  auto stream          = solution.handle_ptr->get_stream();
  auto& op             = *unified_problem;
  const i_t n_vars     = unified_n_vars;
  const i_t nvt        = unified_n_vars_total;
  const i_t nct        = unified_n_constr_total;
  const f_t int_tol    = context.settings.tolerances.integrality_tolerance;
  const f_t cont_upper = (f_t)default_cont_upper;

  last_probes_emitted       = 0;
  last_probe_winner         = 0;
  last_probe_win_margin     = 0.;
  last_probe_winner_fixings = 0;
  probe_all_members_optimal = false;

  // Depth is the binding resource only where the projection is cut off by its budget; where it
  // converges with time to spare, the batch-mode step is absorbed for free. So width is chosen from
  // the previous projection's slack, and is only ever 1 or the cap.
  i_t target_width = cloud_batch_capacity;
  if (batch_config.probe_adaptive_width) {
    target_width = (probe_width_had_slack && probe_width_backoff == 0) ? cloud_batch_capacity : 1;
  }

  // Probe selection first: the batch size follows from how many probes actually survive, which
  // keeps max(climber_id) + 1 == fixed_batch_size exactly as the fixed path asserts.
  std::vector<std::tuple<i_t, i_t, f_t, f_t>> new_bounds;
  const i_t max_probes = std::max((i_t)0, target_width - 1);
  const i_t n_probes   = build_probe_new_bounds(solution, max_probes, new_bounds);
  const i_t n_members  = n_probes + 1;
  last_probes_emitted  = n_probes;
  cuopt_assert(n_members <= cloud_batch_capacity, "Probe batch exceeds pre-expanded capacity");

  if (op.get_objective_coefficients().size() != (size_t)n_members * nvt) {
    ensure_batch_problem_views(solution, n_members);
  }

  // Every member shares one objective, so alpha is scalar here rather than per-climber. Decaying it
  // once per projection matches adjust_objective_with_original in the single path.
  config.alpha                           = config.alpha * config.alpha_decrease_factor;
  const std::vector<f_t> orig_obj_vector = cuopt::host_copy(pb->objective_coefficients, stream);
  solution.handle_ptr->sync_stream();
  const f_t l2_norm_of_original_obj = std::sqrt(std::inner_product(
    orig_obj_vector.begin(), orig_obj_vector.end(), orig_obj_vector.begin(), 0.0));
  const f_t l2_norm_of_distance_obj = std::sqrt((f_t)unified_n_int);
  const f_t distance_weight         = (1. - config.alpha) / l2_norm_of_distance_obj;
  f_t orig_obj_weight               = config.alpha / l2_norm_of_original_obj;
  if (!isfinite(orig_obj_weight)) { orig_obj_weight = 0.; }

  // The rounding is the same for every member; they differ only through new_bounds.
  probe_batch_assignments.resize((size_t)n_members * n_vars, stream);
  for (i_t c = 0; c < n_members; ++c) {
    raft::copy(
      probe_batch_assignments.data() + (size_t)c * n_vars, last_rounding.data(), n_vars, stream);
  }

  const f_t* orig_obj_ptr = pb->objective_coefficients.data();
  const i_t* binary_ptr   = pb->is_binary_variable.data();
  const var_t* var_types  = pb->variable_types.data();
  const i_t* aux_idx_ptr  = d_aux_integer_indices_cache.data();
  const f_t* vlb_ptr      = op.get_variable_lower_bounds().data();
  const f_t* vub_ptr      = op.get_variable_upper_bounds().data();
  const f_t* cloud_ptr    = probe_batch_assignments.data();
  const size_t obj_view   = (size_t)n_members * nvt;

  // Single-path projection objective: an integer sitting on a bound contributes +-1 directly and
  // needs no auxiliary structure; only an interior general integer uses its aux distance variable.
  f_t* obj_ptr = op.get_objective_coefficients().data();
  thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>(obj_view),
                   [=] __device__(size_t g) {
                     const size_t c = g / (size_t)nvt;
                     const i_t loc  = (i_t)(g % (size_t)nvt);
                     f_t dist_obj   = 0.;
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
                     obj_ptr[g]         = dist_obj * distance_weight + orig_obj_weight * orig_obj;
                   });

  const i_t n_constr      = unified_n_constr;
  const f_t* base_clb_ptr = pb->constraint_lower_bounds.data();
  const f_t* base_cub_ptr = pb->constraint_upper_bounds.data();
  f_t* clb_ptr            = op.get_constraint_lower_bounds().data();
  f_t* cub_ptr            = op.get_constraint_upper_bounds().data();
  thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>((size_t)n_members * nct),
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
                       clb_ptr[g] = at_lower || at_upper ? -cont_upper : ((rr & 1) ? val : -val);
                       cub_ptr[g] = cont_upper;
                     }
                   });
  solution.handle_ptr->sync_stream();

  pdlp_solver_settings_t<i_t, f_t> settings;
  settings.method                              = cuopt::mathematical_optimization::method_t::PDLP;
  settings.presolver                           = presolver_t::None;
  settings.fixed_batch_size                    = n_members;
  settings.generate_batch_primal_dual_solution = true;
  settings.new_bounds                          = new_bounds;
  const f_t rlp_base = context.settings.heuristic_params.relaxed_lp_time_limit;
  // Base limit with no batch inflation: the whole point is that width must not cost depth.
  const double base_time_limit =
    std::max(0.05, std::min((double)rlp_base, timer.remaining_time() / 10.));
  settings.time_limit       = std::min(timer.remaining_time(), base_time_limit);
  probe_projection_budget   = settings.time_limit;
  const double lp_tolerance = context.settings.tolerances.absolute_tolerance;
  settings.set_optimality_tolerance(lp_tolerance);
  // set_optimality_tolerance also overwrites the relative tolerances, which would make the per-row
  // criterion abs + rel * |bound| nearly vacuous on rows with large bounds. Mirror the single-path
  // divisor so the batch projects to MIP accuracy.
  double lp_tolerance_divisor =
    context.settings.tolerances.absolute_tolerance / context.settings.tolerances.relative_tolerance;
  if (lp_tolerance_divisor == 0) { lp_tolerance_divisor = 1; }
  settings.tolerances.relative_primal_tolerance = lp_tolerance / lp_tolerance_divisor;
  settings.tolerances.relative_dual_tolerance   = lp_tolerance / lp_tolerance_divisor;
  settings.per_constraint_residual              = true;
  settings.save_best_primal_so_far              = false;
  settings.detect_infeasibility                 = false;

  batch_primal_init.resize(obj_view, stream);
  {
    const f_t* last_projection_ptr = last_projection.data();
    f_t* pinit                     = batch_primal_init.data();
    thrust::for_each(
      solution.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(obj_view),
      [=] __device__(size_t g) {
        const size_t c = g / (size_t)nvt;
        const i_t loc  = (i_t)(g % (size_t)nvt);
        f_t v          = 0.;
        if (loc < n_vars) {
          v = cloud_ptr[c * (size_t)n_vars + (size_t)loc];
        } else {
          const i_t col = aux_idx_ptr[loc - n_vars];
          v = abs(cloud_ptr[c * (size_t)n_vars + (size_t)col] - last_projection_ptr[col]);
        }
        if (!isfinite(v)) v = f_t(0);
        const f_t lo = vlb_ptr[loc];
        const f_t hi = vub_ptr[loc];
        pinit[g]     = (v < lo) ? lo : ((v > hi) ? hi : v);
      });
  }
  settings.set_initial_primal_solution(batch_primal_init.data(), (i_t)obj_view, stream);

  // Member 0 is one continuous trajectory with the same point and objective, so its dual is the
  // dual of a nearby problem; the probe members are perturbations of that same point. Broadcasting
  // the committed dual is therefore the right default here, unlike the discontinuous diversity
  // climbers.
  if (batch_config.probe_dual_warm_start) {
    prepare_probe_dual_warm_start(solution);
    cuopt_assert(probe_prev_dual.size() == (size_t)nct,
                 "Probe dual warm start must be exactly one projection problem's constraint count");
    settings.set_initial_dual_solution(probe_prev_dual.data(), (i_t)nct, stream);
  }
  solution.handle_ptr->sync_stream();

  const auto solve_begin = std::chrono::steady_clock::now();
  auto sol               = cuopt::mathematical_optimization::run_batch_pdlp(op, settings);
  last_projection_solve_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_begin).count();
  auto& term                   = sol.get_terminations_status();
  auto& primal                 = sol.get_primal_solution();
  auto& dual                   = sol.get_dual_solution();
  const size_t expected_primal = (size_t)n_members * nvt;
  const size_t expected_dual   = (size_t)n_members * nct;
  if (primal.size() != expected_primal || dual.size() != expected_dual) {
    const auto es = sol.get_error_status();
    CUOPT_LOG_ERROR(
      "Probe projection returned no usable solution: primal %zu dual %zu (expected %zu/%zu); "
      "error_type=%d msg='%s' members=%d probes=%d",
      primal.size(),
      dual.size(),
      expected_primal,
      expected_dual,
      (int)es.get_error_type(),
      es.what(),
      n_members,
      n_probes);
    return false;
  }

  const auto term_infos = sol.get_additional_termination_informations();
  cuopt_assert(term_infos.size() == (size_t)n_members, "Probe termination information mismatch");
  probe_all_members_optimal = !term.empty();
  for (const auto status : term) {
    probe_all_members_optimal &= status == pdlp_termination_status_t::Optimal;
  }
  double mean_iterations = 0.;
  i_t max_iterations     = 0;
  for (const auto& info : term_infos) {
    mean_iterations += info.number_of_steps_taken;
    max_iterations = std::max(max_iterations, info.number_of_steps_taken);
  }
  mean_iterations /= n_members;
  const auto& member0_info = term_infos[0];
  set_projection_solver_metrics(member0_info.number_of_steps_taken,
                                member0_info.total_number_of_attempted_steps,
                                term.empty() ? -1 : (i_t)term[0],
                                member0_info.l2_primal_residual,
                                member0_info.l2_dual_residual,
                                member0_info.gap,
                                mean_iterations,
                                max_iterations);

  // ---- Rank the members and commit exactly one ----
  probe_batch_assignments.resize((size_t)n_members * n_vars, stream);
  for (i_t c = 0; c < n_members; ++c) {
    raft::copy(probe_batch_assignments.data() + (size_t)c * n_vars,
               primal.data() + (size_t)c * nvt,
               n_vars,
               stream);
  }

  rmm::device_uvector<f_t> adjusted_excess((size_t)n_members, stream);
  rmm::device_uvector<i_t> violated_counts((size_t)n_members, stream);
  rmm::device_uvector<i_t> integer_counts((size_t)n_members, stream);
  {
    auto policy = solution.handle_ptr->get_thrust_policy();
    thrust::fill(policy, adjusted_excess.begin(), adjusted_excess.end(), f_t(0));
    thrust::fill(policy, violated_counts.begin(), violated_counts.end(), i_t(0));
    thrust::fill(policy, integer_counts.begin(), integer_counts.end(), i_t(0));
    constexpr i_t row_tpb   = 64;
    constexpr i_t count_tpb = 128;
    const dim3 row_grid(pb->n_constraints, n_members);
    probe_member_row_excess_kernel<i_t, f_t>
      <<<row_grid, row_tpb, 0, stream>>>(pb->view(),
                                         probe_batch_assignments.data(),
                                         adjusted_excess.data(),
                                         violated_counts.data(),
                                         n_vars);
    RAFT_CHECK_CUDA(stream);
    const i_t count_blocks = std::max((i_t)1, (n_vars + count_tpb - 1) / count_tpb);
    const dim3 count_grid(count_blocks, n_members);
    probe_member_integer_count_kernel<i_t, f_t><<<count_grid, count_tpb, 0, stream>>>(
      pb->view(), probe_batch_assignments.data(), integer_counts.data(), n_vars);
    RAFT_CHECK_CUDA(stream);
  }
  const auto h_adjusted = cuopt::host_copy(adjusted_excess, stream);
  const auto h_violated = cuopt::host_copy(violated_counts, stream);
  const auto h_integers = cuopt::host_copy(integer_counts, stream);
  solution.handle_ptr->sync_stream();

  // Feasible first, then adjusted violation, then integer ratio. Ties go to member 0 so the scheme
  // degrades to plain FP rather than drifting away from the reference trajectory.
  i_t winner = 0;
  for (i_t c = 1; c < n_members; ++c) {
    const bool feasible        = h_violated[c] == 0;
    const bool winner_feasible = h_violated[winner] == 0;
    bool better;
    if (feasible != winner_feasible) {
      better = feasible;
    } else if (h_adjusted[c] != h_adjusted[winner]) {
      better = h_adjusted[c] < h_adjusted[winner];
    } else {
      better = h_integers[c] > h_integers[winner];
    }
    if (better) { winner = c; }
  }
  last_probe_winner     = winner;
  last_probe_win_margin = h_adjusted[0] - h_adjusted[winner];
  last_probe_winner_fixings =
    (i_t)std::count_if(new_bounds.begin(), new_bounds.end(), [winner](const auto& entry) {
      return std::get<0>(entry) == winner;
    });
  if (winner != 0) {
    CUOPT_LOG_INFO(
      "Probe projection: member %d of %d wins, adjusted violation %g against member 0 %g",
      winner,
      n_members,
      (double)h_adjusted[winner],
      (double)h_adjusted[0]);
  }

  raft::copy(solution.assignment.data(),
             probe_batch_assignments.data() + (size_t)winner * n_vars,
             n_vars,
             stream);
  // The batch primal can land marginally outside the variable bounds; clamp before anything reads
  // it as an assignment.
  solution.clamp_within_bounds();
  raft::copy(last_projection.data(), solution.assignment.data(), n_vars, stream);

  // Carry the winner's dual, and only when the solve actually produced one. A projection that dies
  // early must leave the previous dual intact rather than overwrite it with a partial vector.
  if (batch_config.probe_dual_warm_start && dual.size() == expected_dual) {
    probe_prev_dual.resize(nct, stream);
    raft::copy(probe_prev_dual.data(), dual.data() + (size_t)winner * nct, nct, stream);
    probe_prev_dual_size = nct;
    probe_dual_seeded    = true;
  }
  solution.handle_ptr->sync_stream();

  const bool is_feasible = solution.compute_feasibility();
  if (!is_feasible) {
    CUOPT_LOG_INFO("LP is infeasible returning the current PDLP solution! Code %d",
                   term.empty() ? -1 : (int)term[0]);
  }
  return is_feasible;
}

// Mirrors run_single_fp_descent step for step - round_nearest, project, distance-cycle check,
// round, the all-integers full-precision verification and the FJ fallback. Only the projection
// differs, so any behavioural difference is attributable to the projection alone.
template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_probe_fp_descent(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("run_probe_fp_descent");
  using timing_clock = std::chrono::steady_clock;
  auto stream        = solution.handle_ptr->get_stream();

  if (unified_problem == nullptr || unified_n_vars != solution.problem_ptr->n_variables ||
      unified_n_constr != solution.problem_ptr->n_constraints ||
      unified_n_int != solution.problem_ptr->n_integer_vars ||
      unified_n_aux !=
        solution.problem_ptr->n_integer_vars - solution.problem_ptr->get_n_binary_variables()) {
    build_unified_projection_problem(solution);
    probe_dual_seeded    = false;
    probe_prev_dual_size = 0;
  }
  i_t batch_size = compute_cloud_batch_size(solution);
  if (batch_size == 0) { return run_single_fp_descent(solution); }
  const size_t expected_obj_size = (size_t)batch_size * unified_n_vars_total;
  if (cloud_batch_capacity != batch_size ||
      unified_problem->get_objective_coefficients().size() != expected_obj_size) {
    expand_unified_projection_batch_buffers(solution, batch_size);
  }
  h_probe_is_integer.assign((size_t)unified_n_vars, 0);
  for (i_t col : h_integer_indices_cache) {
    h_probe_is_integer[(size_t)col] = 1;
  }

  // solution.assignment holds the relaxed-LP point that is about to be rounded, so seeding
  // last_projection from it gives the first projection real LP values to rank probes and to
  // initialise the auxiliary distances with, instead of whatever the allocator left behind.
  raft::copy(
    last_projection.data(), solution.assignment.data(), solution.assignment.size(), stream);
  solution.round_nearest();
  raft::copy(last_rounding.data(), solution.assignment.data(), solution.assignment.size(), stream);
  while (true) {
    if (context.diversity_manager_ptr->check_b_b_preemption() || timer.check_time_limit()) {
      CUOPT_LOG_INFO("FP time limit reached!");
      round(solution);
      return false;
    }
    proj_begin = timer.remaining_time();
    if (metrics != nullptr) { solution.handle_ptr->sync_stream(); }
    const auto projection_begin = timing_clock::now();
    bool is_feasible            = probe_project_onto_polytope(solution);
    if (metrics != nullptr) { solution.handle_ptr->sync_stream(); }
    const double projection_time =
      std::chrono::duration<double>(timing_clock::now() - projection_begin).count();
    if (batch_config.probe_adaptive_width) {
      const bool had_slack =
        probe_all_members_optimal &&
        projection_time <= batch_config.probe_width_slack_ratio * probe_projection_budget;
      // The cap keeping its slack is the only evidence that width is affordable here; losing it
      // means width is being paid for with depth, so back off rather than retrying it every
      // projection.
      if (last_probes_emitted > 0 && !had_slack) {
        probe_width_backoff = batch_config.probe_width_backoff_projections;
      } else if (probe_width_backoff > 0) {
        --probe_width_backoff;
      }
      probe_width_had_slack = had_slack;
    }
    i_t n_integers = solution.compute_number_of_integers();
    CUOPT_LOG_INFO("after fp projection n_integers %d total n_integes %d",
                   n_integers,
                   solution.problem_ptr->n_integer_vars);
    record_projection_metrics(solution, n_integers, last_probes_emitted + 1, projection_time);
    if (metrics != nullptr) {
      metrics->iterations.back().effective_cloud_size = last_probes_emitted + 1;
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
        finish_iteration_metrics(true, false, false);
        return false;
      }
    }
    if (n_integers == solution.problem_ptr->n_integer_vars) {
      if (is_feasible) {
        CUOPT_LOG_INFO("[FP_FEASIBLE] Feasible solution found after LP with relative tolerance");
        finish_iteration_metrics(false, false, true);
        return true;
      } else if (!last_distances.empty() && last_distances[0] < distance_to_check_for_feasible) {
        const f_t lp_verify_time_limit = 5.;
        relaxed_lp_settings_t lp_settings;
        lp_settings.time_limit            = lp_verify_time_limit;
        lp_settings.tolerance             = solution.problem_ptr->tolerances.absolute_tolerance;
        lp_settings.return_first_feasible = true;
        lp_settings.save_state            = true;
        lp_settings.check_infeasibility   = false;
        const auto verify_begin           = timing_clock::now();
        run_lp_with_vars_fixed(*solution.problem_ptr,
                               solution,
                               solution.problem_ptr->integer_indices,
                               lp_settings,
                               &constraint_prop.bounds_update);
        last_verify_lp_time =
          std::chrono::duration<double>(timing_clock::now() - verify_begin).count();
        is_feasible = solution.get_feasible();
        n_integers  = solution.compute_number_of_integers();
        if (is_feasible && n_integers == solution.problem_ptr->n_integer_vars) {
          CUOPT_LOG_INFO("[FP_FEASIBLE] Feasible solution verified with LP!");
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
      const auto fj_begin = timing_clock::now();
      is_feasible         = test_fj_feasible(solution, batch_config.fj_ratio * proj_and_round_time);
      last_fj_time        = std::chrono::duration<double>(timing_clock::now() - fj_begin).count();
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
      CUOPT_LOG_INFO("total_fp_time_until_cycle: %f", total_fp_time_until_cycle);
      finish_iteration_metrics(true, false, false);
      return false;
    }
    cycle_queue.n_iterations_without_cycle++;
    finish_iteration_metrics(false, false, false);
  }
  return false;
}

#define INSTANTIATE_PROBE(F_TYPE)                                                                \
  template bool feasibility_pump_t<int, F_TYPE>::run_probe_fp_descent(solution_t<int, F_TYPE>&); \
  template bool feasibility_pump_t<int, F_TYPE>::probe_project_onto_polytope(                    \
    solution_t<int, F_TYPE>&);                                                                   \
  template int feasibility_pump_t<int, F_TYPE>::build_probe_new_bounds(                          \
    solution_t<int, F_TYPE>&, int, std::vector<std::tuple<int, int, F_TYPE, F_TYPE>>&);          \
  template void feasibility_pump_t<int, F_TYPE>::prepare_probe_dual_warm_start(                  \
    solution_t<int, F_TYPE>&);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE_PROBE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE_PROBE(double)
#endif

#undef INSTANTIATE_PROBE

}  // namespace cuopt::mathematical_optimization::mip
