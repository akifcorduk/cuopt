/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// CPU port of the standalone LNS feasibility loop (see lns_feasibility.cuh for the GPU version).
// The whole ruin/repair loop runs on host copies of the problem: seed selection, related /
// violated-constraint ruin, similarity/divergence neighbor scoring, tardiness bookkeeping and
// acceptance are plain host loops. Repair is a simple constraint-propagation rounding based on
// dual_simplex/bounds_strengthening.cpp, and the feasibility bursts reuse the CPU Feasibility
// Jump. All tuning numbers are hard coded here on purpose.

#include <mip_heuristics/feasibility_jump/cpu_fj_thread.cuh>
#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>
#include <mip_heuristics/logger.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <mip_heuristics/utils.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>
#include <utilities/timer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class lns_feasibility_cpu_t {
 public:
  explicit lns_feasibility_cpu_t(mip_solver_context_t<i_t, f_t>& context_)
    : context(context_), problem_ptr(context_.problem_ptr), rng(cuopt::seed_generator::get_seed())
  {
    init_host_data();
  }

  // Entry point mirroring lns_feasibility_t::run. Reads the current assignment from `solution`,
  // improves it on the host, and writes the best assignment found back into `solution`.
  bool run(solution_t<i_t, f_t>& solution, f_t time_limit)
  {
    if (time_limit <= f_t{0}) { return solution.get_feasible(); }
    lns_timer = timer_t(time_limit);

    std::vector<f_t> x = solution.get_host_assignment();
    state_t start      = eval_state(x);
    CUOPT_LOG_INFO(
      "CPU LNS start: time_limit %.2fs, vars %d, integer vars %d, constraints %d, unsat %d, "
      "excess %.6e",
      time_limit,
      n_vars,
      n_int,
      n_cons,
      start.unsat,
      start.total_excess);

    bool accepted_any = false;

    // Seed repair: a single constraint-propagation rounding over all integer variables.
    {
      std::vector<f_t> repaired = x;
      std::vector<char> ruined(n_vars, 0);
      for (i_t v : integer_indices) {
        ruined[v] = 1;
      }
      if (constraint_prop_round(repaired, ruined) &&
          is_better_state(eval_state(repaired), eval_state(x))) {
        x            = std::move(repaired);
        accepted_any = true;
      }
    }
    if (eval_state(x).unsat == 0) {
      CUOPT_LOG_INFO("CPU LNS feasible via seed constraint-prop repair (%.2fs elapsed)",
                     lns_timer.elapsed_time());
      return finalize(solution, x);
    }

    // Initial Feasibility-Jump burst from the seed (strongest feasibility engine first).
    if (!lns_timer.check_time_limit()) {
      std::vector<f_t> xf = x;
      const f_t budget    = std::min<f_t>(f_t{1.0}, lns_timer.remaining_time());
      const bool feasible = fj_burst(xf, budget, solution);
      if (feasible) {
        CUOPT_LOG_INFO("CPU LNS feasible via initial FJ burst (%.2fs elapsed)",
                       lns_timer.elapsed_time());
        x = std::move(xf);
        return finalize(solution, x);
      }
      if (is_better_state(eval_state(xf), eval_state(x))) {
        x            = std::move(xf);
        accepted_any = true;
      }
    }

    std::vector<f_t> best = x;

    // Small ruin sets repair fast and preserve structure; grow additively on stalls.
    const i_t min_ruin = 8;
    const i_t max_ruin = std::max(min_ruin, std::min<i_t>(64, n_int / 2));
    i_t ruin_count     = min_ruin;
    i_t failure_streak = 0;

    const bool related_graph_available = related_offsets.size() == static_cast<size_t>(n_vars) + 1;
    size_t attempts_since_fj           = 20;

    for (size_t attempt = 0; attempt < 1000000; ++attempt) {
      ++attempts_since_fj;
      if (lns_timer.check_time_limit()) { break; }

      // Periodic FJ polish when the incumbent is close to feasible.
      state_t best_state = eval_state(best);
      if (best_state.unsat > 0 && best_state.unsat <= 64 && attempts_since_fj >= 20) {
        attempts_since_fj   = 0;
        std::vector<f_t> xf = best;
        const f_t budget    = std::min<f_t>(f_t{0.5}, lns_timer.remaining_time());
        const bool feasible = fj_burst(xf, budget, solution);
        if (feasible) {
          CUOPT_LOG_INFO("CPU LNS feasible via periodic FJ polish at attempt %lu (%.2fs elapsed)",
                         attempt,
                         lns_timer.elapsed_time());
          best = xf;
          x    = std::move(xf);
          return finalize(solution, best);
        }
        if (is_better_state(eval_state(xf), eval_state(best))) {
          best = xf;
          x    = std::move(xf);
        }
      }

      std::vector<f_t> lhs = compute_lhs(x);
      update_tardiness(lhs);

      std::vector<char> ruined(n_vars, 0);
      const bool use_violated = !related_graph_available;
      const i_t count         = use_violated ? violated_constraint_ruin(lhs, ruined)
                                             : select_related_ruin(x, lhs, ruin_count, attempt, ruined);
      if ((use_violated && count < 1) || (!use_violated && count < 2)) { continue; }

      std::vector<f_t> candidate = x;
      const bool repaired        = constraint_prop_round(candidate, ruined);

      bool accepted = false;
      if (repaired) {
        std::vector<f_t> cand_lhs = compute_lhs(candidate);
        accepted                  = accept_feasibility_move(lhs, cand_lhs);
        update_tardiness(cand_lhs);
      }

      if (!use_violated) {
        if (accepted) {
          ruin_count     = std::max(min_ruin, ruin_count - min_ruin);
          failure_streak = 0;
        } else if (++failure_streak >= 5) {
          ruin_count     = std::min(max_ruin, ruin_count + min_ruin);
          failure_streak = 0;
        }
      }
      if (!accepted) { continue; }

      x            = candidate;
      accepted_any = true;
      if (is_better_state(eval_state(x), eval_state(best))) { best = x; }
      if (eval_state(x).unsat == 0) {
        CUOPT_LOG_INFO("CPU LNS feasible via %s ruin/repair at attempt %lu (%.2fs elapsed)",
                       use_violated ? "violated-constraint" : "related",
                       attempt,
                       lns_timer.elapsed_time());
        return finalize(solution, best);
      }
    }

    if (is_better_state(eval_state(best), eval_state(x))) { x = best; }
    state_t final_state = eval_state(x);
    CUOPT_LOG_INFO("CPU LNS finished: unsat %d, excess %.6e, accepted_any %d",
                   final_state.unsat,
                   final_state.total_excess,
                   accepted_any);
    if (final_state.unsat == 0) {
      CUOPT_LOG_INFO("CPU LNS feasible at loop exit (%.2fs elapsed)", lns_timer.elapsed_time());
      return finalize(solution, x);
    }
    solution.copy_new_assignment(x);
    solution.compute_feasibility();
    return accepted_any;
  }

 private:
  struct state_t {
    i_t unsat;
    f_t total_excess;
    f_t weighted_violation;
    f_t tardiness_penalty;
  };

  void init_host_data()
  {
    auto stream             = problem_ptr->handle_ptr->get_stream();
    problem_t<i_t, f_t>& pb = *problem_ptr;
    n_vars                  = pb.n_variables;
    n_cons                  = pb.n_constraints;
    n_int                   = pb.n_integer_vars;
    tols                    = pb.tolerances;

    offsets          = cuopt::host_copy(pb.offsets, stream);
    variables        = cuopt::host_copy(pb.variables, stream);
    coefficients     = cuopt::host_copy(pb.coefficients, stream);
    rev_offsets      = cuopt::host_copy(pb.reverse_offsets, stream);
    rev_constraints  = cuopt::host_copy(pb.reverse_constraints, stream);
    rev_coefficients = cuopt::host_copy(pb.reverse_coefficients, stream);
    cons_lb          = cuopt::host_copy(pb.constraint_lower_bounds, stream);
    cons_ub          = cuopt::host_copy(pb.constraint_upper_bounds, stream);
    integer_indices  = cuopt::host_copy(pb.integer_indices, stream);

    auto bounds = cuopt::host_copy(pb.variable_bounds, stream);
    var_lb.resize(n_vars);
    var_ub.resize(n_vars);
    for (i_t j = 0; j < n_vars; ++j) {
      var_lb[j] = get_lower(bounds[j]);
      var_ub[j] = get_upper(bounds[j]);
    }
    auto vtypes = cuopt::host_copy(pb.variable_types, stream);
    var_is_int.resize(n_vars);
    for (i_t j = 0; j < n_vars; ++j) {
      var_is_int[j] = (vtypes[j] == var_t::INTEGER);
    }

    if (pb.related_variables_offsets.size() == static_cast<size_t>(n_vars) + 1) {
      related_offsets = cuopt::host_copy(pb.related_variables_offsets, stream);
      related_vars    = cuopt::host_copy(pb.related_variables, stream);
    }

    constraint_tardiness.assign(n_cons, 0);
    compute_row_norms();

    accum_dissim.assign(n_vars, f_t{0});
    accum_weight.assign(n_vars, f_t{0});
    accum_inter.assign(n_vars, f_t{0});
  }

  void compute_row_norms()
  {
    row_inf_norm.assign(n_cons, f_t{0});
    row_l1_norm.assign(n_cons, f_t{0});
    for (i_t c = 0; c < n_cons; ++c) {
      f_t inf = 0;
      f_t l1  = 0;
      for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
        const f_t a = std::abs(coefficients[p]);
        inf         = std::max(inf, a);
        l1 += a;
      }
      row_inf_norm[c] = inf;
      row_l1_norm[c]  = l1;
    }
  }

  std::vector<f_t> compute_lhs(const std::vector<f_t>& x) const
  {
    std::vector<f_t> lhs(n_cons, f_t{0});
    for (i_t c = 0; c < n_cons; ++c) {
      f_t sum = 0;
      for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
        sum += coefficients[p] * x[variables[p]];
      }
      lhs[c] = sum;
    }
    return lhs;
  }

  bool constraint_violated(i_t c, f_t val) const
  {
    return !is_constraint_feasible<i_t, f_t>(val, cons_lb[c], cons_ub[c], tols);
  }

  f_t excess_of(i_t c, f_t val) const
  {
    return std::max<f_t>(0, cons_lb[c] - val) + std::max<f_t>(0, val - cons_ub[c]);
  }

  state_t eval_state(const std::vector<f_t>& x) const { return state_from_lhs(compute_lhs(x)); }

  state_t state_from_lhs(const std::vector<f_t>& lhs) const
  {
    state_t s{0, f_t{0}, f_t{0}, f_t{0}};
    for (i_t c = 0; c < n_cons; ++c) {
      const f_t excess = excess_of(c, lhs[c]);
      s.total_excess += excess;
      if (constraint_violated(c, lhs[c])) {
        ++s.unsat;
        const f_t weight = static_cast<f_t>(1 + constraint_tardiness[c]);
        s.weighted_violation += weight;
        s.tardiness_penalty += excess * weight;
      }
    }
    return s;
  }

  void update_tardiness(const std::vector<f_t>& lhs)
  {
    for (i_t c = 0; c < n_cons; ++c) {
      if (constraint_violated(c, lhs[c])) { ++constraint_tardiness[c]; }
    }
  }

  // Guided-Local-Search acceptance: accept on fewer unsatisfied constraints, else a lower
  // tardiness-weighted violation count, else (on ties) lower total excess.
  bool accept_feasibility_move(const std::vector<f_t>& cur_lhs, const std::vector<f_t>& cand_lhs)
  {
    const state_t cur  = state_from_lhs(cur_lhs);
    const state_t cand = state_from_lhs(cand_lhs);
    if (cand.unsat < cur.unsat) { return true; }
    if (cand.weighted_violation + OBJECTIVE_EPSILON < cur.weighted_violation) { return true; }
    if (cand.weighted_violation <= cur.weighted_violation + OBJECTIVE_EPSILON &&
        cand.total_excess + OBJECTIVE_EPSILON < cur.total_excess) {
      return true;
    }
    return false;
  }

  bool is_better_state(const state_t& cand, const state_t& inc) const
  {
    if (cand.unsat == 0 && inc.unsat != 0) { return true; }
    if (cand.unsat != 0 && inc.unsat == 0) { return false; }
    const i_t max_unsat_increase = std::max<i_t>(5 * inc.unsat, 10);
    if (cand.unsat - inc.unsat > max_unsat_increase) { return false; }
    if (cand.unsat < inc.unsat) { return true; }
    if (cand.tardiness_penalty + OBJECTIVE_EPSILON < inc.tardiness_penalty) { return true; }
    return cand.unsat == inc.unsat &&
           cand.tardiness_penalty <= inc.tardiness_penalty + OBJECTIVE_EPSILON;
  }

  // With high probability draw the seed from an integer variable that sits in a violated
  // constraint, so the ruined neighborhood actually covers infeasibility.
  i_t pick_seed_var(const std::vector<f_t>& lhs)
  {
    std::uniform_real_distribution<double> coin(0., 1.);
    if (!integer_indices.empty() && coin(rng) < 0.9) {
      std::vector<i_t> pool;
      pool.reserve(integer_indices.size());
      for (i_t v : integer_indices) {
        f_t violation = 0;
        for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
          violation += excess_of(rev_constraints[p], lhs[rev_constraints[p]]);
        }
        if (violation > f_t{0}) { pool.push_back(v); }
      }
      if (!pool.empty()) {
        std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
        return pool[dist(rng)];
      }
    }
    std::uniform_int_distribution<size_t> dist(0, integer_indices.size() - 1);
    return integer_indices[dist(rng)];
  }

  bool eligible_neighbor(i_t candidate, i_t seed, const std::vector<char>& ruined) const
  {
    return candidate != seed && var_is_int[candidate] && ruined[candidate] == 0;
  }

  // Related ruin: seed + its most similar (structurally/state) related-integer neighbors.
  i_t select_related_ruin(const std::vector<f_t>& x,
                          const std::vector<f_t>& lhs,
                          i_t target_ruin_count,
                          size_t attempt,
                          std::vector<char>& ruined)
  {
    if (n_int == 0) { return 0; }
    const i_t target         = target_ruin_count <= 0 ? 128 : target_ruin_count;
    const i_t selected_count = std::min(std::max<i_t>(2, target), n_int / 2);
    if (selected_count <= 0) { return 0; }

    const i_t seed = pick_seed_var(lhs);
    ruined[seed]   = 1;
    if (selected_count <= 1) { return 1; }
    if (related_offsets.size() != static_cast<size_t>(n_vars) + 1) { return 1; }

    const i_t nb_begin = related_offsets[seed];
    const i_t nb_end   = related_offsets[seed + 1];
    const i_t nb_count = nb_end - nb_begin;
    if (nb_count <= 0) { return 1; }

    // Similarity scoring with an alpha that ramps from structural toward state over the
    // attempts, capped to [0.2, 0.8] so both the structural and state terms always contribute.
    const f_t alpha_raw  = static_cast<f_t>(1.0 - static_cast<double>(attempt) / 50.0);
    const f_t alpha_this = std::min<f_t>(0.8, std::max<f_t>(0.2, alpha_raw));

    std::vector<std::pair<f_t, i_t>> scored(nb_count);
    score_similarity(x, seed, nb_begin, nb_count, alpha_this, ruined, scored);

    std::sort(
      scored.begin(), scored.end(), [](const std::pair<f_t, i_t>& a, const std::pair<f_t, i_t>& b) {
        return a.first < b.first;
      });

    const i_t additional = std::min(selected_count - 1, nb_count);
    i_t marked           = 0;
    for (i_t i = 0; i < additional; ++i) {
      if (!std::isfinite(scored[i].first)) { break; }
      ruined[scored[i].second] = 1;
      ++marked;
    }
    return marked + 1;
  }

  // Similarity scoring: scatter the seed's shared-constraint contributions into per-variable
  // accumulators, then finalize each neighbor to weighted-mean dissimilarity minus a Jaccard
  // (shared-constraint) bonus. Lower score = more alike/coupled and sorts first.
  void score_similarity(const std::vector<f_t>& x,
                        i_t seed,
                        i_t nb_begin,
                        i_t nb_count,
                        f_t alpha,
                        const std::vector<char>& ruined,
                        std::vector<std::pair<f_t, i_t>>& scored)
  {
    const f_t seed_val = x[seed];
    std::vector<i_t> touched;
    for (i_t rev = rev_offsets[seed]; rev < rev_offsets[seed + 1]; ++rev) {
      const i_t c   = rev_constraints[rev];
      const f_t inf = row_inf_norm[c];
      if (!(inf > f_t{0})) { continue; }
      const f_t seed_hat = rev_coefficients[rev] / inf;
      const f_t weight   = row_l1_norm[c] / inf;
      for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
        const i_t k = variables[p];
        if (k == seed) { continue; }
        if (accum_inter[k] == f_t{0}) { touched.push_back(k); }
        const f_t cand_hat   = coefficients[p] / inf;
        const f_t structural = std::abs(seed_hat - cand_hat);
        const f_t stt        = std::abs(seed_hat * seed_val - cand_hat * x[k]);
        accum_dissim[k] += weight * (alpha * structural + (f_t{1} - alpha) * stt);
        accum_weight[k] += weight;
        accum_inter[k] += f_t{1};
      }
    }

    const f_t seed_degree = static_cast<f_t>(rev_offsets[seed + 1] - rev_offsets[seed]);
    for (i_t i = 0; i < nb_count; ++i) {
      const i_t k = related_vars[nb_begin + i];
      if (!eligible_neighbor(k, seed, ruined)) {
        scored[i] = {std::numeric_limits<f_t>::infinity(), k};
        continue;
      }
      const f_t weight_sum = accum_weight[k];
      const f_t inter      = accum_inter[k];
      const f_t sim_mean   = weight_sum > f_t{0} ? accum_dissim[k] / weight_sum : f_t{0};
      const f_t cand_deg   = static_cast<f_t>(rev_offsets[k + 1] - rev_offsets[k]);
      const f_t uni        = seed_degree + cand_deg - inter;
      const f_t jaccard    = uni > f_t{0} ? inter / uni : f_t{0};
      scored[i]            = {sim_mean - jaccard, k};
    }

    for (i_t k : touched) {
      accum_dissim[k] = f_t{0};
      accum_weight[k] = f_t{0};
      accum_inter[k]  = f_t{0};
    }
  }

  // Free every integer variable that appears in a violated constraint.
  i_t violated_constraint_ruin(const std::vector<f_t>& lhs, std::vector<char>& ruined)
  {
    i_t count = 0;
    for (i_t c = 0; c < n_cons; ++c) {
      if (!constraint_violated(c, lhs[c])) { continue; }
      for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
        const i_t v = variables[p];
        if (var_is_int[v] && ruined[v] == 0) {
          ruined[v] = 1;
          ++count;
        }
      }
    }
    return count;
  }

  // Simple constraint-propagation rounding (based on dual_simplex/bounds_strengthening.cpp):
  // fix the non-ruined integer variables, propagate bounds, then round the ruined integer
  // variables one at a time (re-propagating after each). Continuous variables are clamped into
  // the tightened bounds and left for the FJ burst to finish. Returns false when propagation
  // proves the fixed neighborhood infeasible.
  bool constraint_prop_round(std::vector<f_t>& x, const std::vector<char>& ruined)
  {
    const f_t eps     = tols.absolute_tolerance;
    const f_t int_tol = tols.integrality_tolerance;

    std::vector<f_t> lb = var_lb;
    std::vector<f_t> ub = var_ub;
    for (i_t j = 0; j < n_vars; ++j) {
      if (var_is_int[j] && ruined[j] == 0) {
        const f_t v = std::round(std::min(std::max(x[j], var_lb[j]), var_ub[j]));
        lb[j] = ub[j] = v;
      }
    }

    if (!propagate(lb, ub, std::vector<char>())) { return false; }

    for (i_t v : integer_indices) {
      if (ruined[v] == 0) { continue; }
      if (lb[v] > ub[v] + eps) { return false; }
      f_t target = std::min(std::max(x[v], lb[v]), ub[v]);
      f_t val    = std::round(target);
      if (val < lb[v] - int_tol) { val = std::ceil(lb[v] - int_tol); }
      if (val > ub[v] + int_tol) { val = std::floor(ub[v] + int_tol); }
      lb[v] = ub[v] = val;
      std::vector<char> bounds_changed(n_vars, 0);
      bounds_changed[v] = 1;
      if (!propagate(lb, ub, bounds_changed)) { return false; }
    }

    for (i_t j = 0; j < n_vars; ++j) {
      if (var_is_int[j]) {
        x[j] = lb[j];
      } else {
        x[j] = std::min(std::max(x[j], lb[j]), ub[j]);
      }
    }
    return true;
  }

  // Iterative interval propagation with integer rounding. lb/ub are both the working bounds and
  // the floors to clamp against; results are written back into them. Returns false on a proven
  // infeasibility. If bounds_changed is empty every constraint is scanned.
  bool propagate(std::vector<f_t>& lb,
                 std::vector<f_t>& ub,
                 const std::vector<char>& bounds_changed)
  {
    const f_t eps                   = tols.absolute_tolerance;
    const f_t int_tol               = tols.integrality_tolerance;
    constexpr f_t inf               = std::numeric_limits<f_t>::infinity();
    const std::vector<f_t> lb_floor = lb;
    const std::vector<f_t> ub_floor = ub;

    std::vector<char> cons_changed(n_cons, 1);
    std::vector<char> cons_changed_next(n_cons, 0);
    std::vector<char> var_changed(n_vars, 0);
    std::vector<f_t> delta_min_activity(n_cons, f_t{0});
    std::vector<f_t> delta_max_activity(n_cons, f_t{0});

    if (!bounds_changed.empty()) {
      std::fill(cons_changed.begin(), cons_changed.end(), 0);
      for (i_t j = 0; j < n_vars; ++j) {
        if (!bounds_changed[j]) { continue; }
        for (i_t p = rev_offsets[j]; p < rev_offsets[j + 1]; ++p) {
          cons_changed[rev_constraints[p]] = 1;
        }
      }
    }

    for (i_t iter = 0; iter < 10; ++iter) {
      for (i_t c = 0; c < n_cons; ++c) {
        if (!cons_changed[c]) { continue; }
        f_t min_a = 0;
        f_t max_a = 0;
        for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
          const i_t j    = variables[p];
          const f_t a    = coefficients[p];
          var_changed[j] = 1;
          if (a > 0) {
            min_a += a * lb[j];
            max_a += a * ub[j];
          } else if (a < 0) {
            min_a += a * ub[j];
            max_a += a * lb[j];
          }
          if (ub[j] == inf && a > 0) { max_a = inf; }
          if (lb[j] == -inf && a < 0) { max_a = inf; }
          if (lb[j] == -inf && a > 0) { min_a = -inf; }
          if (ub[j] == inf && a < 0) { min_a = -inf; }
        }
        if (min_a > cons_ub[c] + eps || max_a < cons_lb[c] - eps) { return false; }
        delta_min_activity[c] = cons_ub[c] - min_a;
        delta_max_activity[c] = cons_lb[c] - max_a;
      }

      i_t num_bounds_changed = 0;
      for (i_t k = 0; k < n_vars; ++k) {
        if (!var_changed[k]) { continue; }
        const f_t old_lb = lb[k];
        const f_t old_ub = ub[k];
        f_t new_lb       = old_lb;
        f_t new_ub       = old_ub;
        for (i_t p = rev_offsets[k]; p < rev_offsets[k + 1]; ++p) {
          const i_t c = rev_constraints[p];
          if (!cons_changed[c]) { continue; }
          const f_t a       = rev_coefficients[p];
          f_t delta_min_act = delta_min_activity[c] + ((a < 0) ? a * old_ub : a * old_lb);
          f_t delta_max_act = delta_max_activity[c] + ((a > 0) ? a * old_ub : a * old_lb);
          const f_t comp_lb = (a < 0) ? delta_min_act / a : delta_max_act / a;
          const f_t comp_ub = (a < 0) ? delta_max_act / a : delta_min_act / a;
          new_lb            = std::max(new_lb, comp_lb);
          new_ub            = std::min(new_ub, comp_ub);
        }
        if (var_is_int[k]) {
          new_lb = std::ceil(new_lb - int_tol);
          new_ub = std::floor(new_ub + int_tol);
        }
        new_lb = std::max(new_lb, lb_floor[k]);
        new_ub = std::min(new_ub, ub_floor[k]);
        if (new_lb > new_ub + eps) { return false; }
        if (new_lb != old_lb || new_ub != old_ub) {
          for (i_t p = rev_offsets[k]; p < rev_offsets[k + 1]; ++p) {
            cons_changed_next[rev_constraints[p]] = 1;
          }
        }
        lb[k] = std::min(new_lb, new_ub);
        ub[k] = std::max(new_lb, new_ub);
        if (std::abs(new_lb - old_lb) > 1e3 * eps || std::abs(new_ub - old_ub) > 1e3 * eps) {
          ++num_bounds_changed;
        }
      }

      if (num_bounds_changed == 0) { break; }
      std::swap(cons_changed, cons_changed_next);
      std::fill(cons_changed_next.begin(), cons_changed_next.end(), 0);
      std::fill(var_changed.begin(), var_changed.end(), 0);
    }
    return true;
  }

  // Run a CPU Feasibility-Jump burst seeded from `x`; write the best state found back into `x`.
  bool fj_burst(std::vector<f_t>& x, f_t time_limit, solution_t<i_t, f_t>& solution)
  {
    if (time_limit <= f_t{0}) { return eval_state(x).unsat == 0; }
    solution.copy_new_assignment(x);

    fj_settings_t settings;
    settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
    settings.n_of_minimums_for_exit = 500;
    settings.update_weights         = true;
    settings.feasibility_run        = true;
    auto fj_cpu                     = init_fj_cpu_standalone<i_t, f_t>(
      *problem_ptr, solution, context.preempt_heuristic_solver_, settings);
    fj_cpu->log_prefix = "[LNS CPUFJ] ";
    cpufj_solve<i_t, f_t>(fj_cpu.get(), time_limit);

    state_t best  = eval_state(x);
    bool feasible = best.unsat == 0;
    auto consider = [&](std::vector<f_t>& cand) {
      const state_t s = eval_state(cand);
      if (is_better_state(s, best)) {
        best     = s;
        x        = cand;
        feasible = s.unsat == 0;
      }
    };
    consider(fj_cpu->h_assignment.underlying());
    if (fj_cpu->feasible_found) { consider(fj_cpu->h_best_assignment.underlying()); }
    return feasible;
  }

  bool finalize(solution_t<i_t, f_t>& solution, std::vector<f_t>& x)
  {
    solution.copy_new_assignment(x);
    if (!solution.compute_feasibility()) { return false; }
    if (n_int == 0) { return true; }
    solution_t<i_t, f_t> rounded(solution);
    if (rounded.round_nearest()) { solution.copy_from(rounded); }
    return true;
  }

  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;

  i_t n_vars{0};
  i_t n_cons{0};
  i_t n_int{0};
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tols{};

  std::vector<i_t> offsets;
  std::vector<i_t> variables;
  std::vector<f_t> coefficients;
  std::vector<i_t> rev_offsets;
  std::vector<i_t> rev_constraints;
  std::vector<f_t> rev_coefficients;
  std::vector<f_t> cons_lb;
  std::vector<f_t> cons_ub;
  std::vector<f_t> var_lb;
  std::vector<f_t> var_ub;
  std::vector<char> var_is_int;
  std::vector<i_t> integer_indices;
  std::vector<i_t> related_offsets;
  std::vector<i_t> related_vars;

  std::vector<f_t> row_inf_norm;
  std::vector<f_t> row_l1_norm;
  std::vector<f_t> accum_dissim;
  std::vector<f_t> accum_weight;
  std::vector<f_t> accum_inter;

  std::vector<i_t> constraint_tardiness;
  std::mt19937 rng;
  timer_t lns_timer{0.};
};

}  // namespace cuopt::linear_programming::detail
