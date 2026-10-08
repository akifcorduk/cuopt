/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/feasibility_jump/cpu/search/api.hpp>
#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/timer.hpp>
#include "cpufj_experiment.cuh"
#include "cpufj_geometry.cuh"
#include "cpufj_similarity.cuh"
#include "cpufj_validation.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <thread>

namespace cuopt::mathematical_optimization::mip {
// CPUFJ solves a fresh neighborhood on every call. Keep its setup contract and
// the LNS worker's validated incumbent separate from that solve-local state.
template <typename i_t, typename f_t>
bool repair_cpufj_lns_neighborhood(fj_cpu_climber_t<i_t, f_t>* ptr,
                                   f_t time_limit,
                                   bool reset_local_incumbent)
{
  auto archived_assignment = ptr->h_best_assignment.underlying();
  const bool have_archive =
    ptr->feasible_found && clamp_and_validate_cpufj_lns_seed(
                             *ptr->problem, ptr->h_var_bounds.underlying(), archived_assignment);
  const f_t archived_objective = have_archive
                                   ? std::inner_product(archived_assignment.begin(),
                                                        archived_assignment.end(),
                                                        ptr->problem->h_obj_coeffs.begin(),
                                                        f_t{0})
                                   : std::numeric_limits<f_t>::infinity();

  // A previous scalar repair expands equalities/ranged constraints into search
  // rows. Restore model-row membership before calling the unchanged setup again.
  ptr->n_rows = 0;
  ptr->violated_constraints.resize(ptr->problem->n_constraints);
  ptr->satisfied_constraints.resize(ptr->problem->n_constraints);
  // CPUFJ setup releases these arrays after constructing its private search rows.
  // Recreate the LNS climber's initial unit weights for the next repair.
  ptr->h_initial_left_weights.resize(ptr->problem->n_constraints, f_t{1});
  ptr->h_initial_right_weights.resize(ptr->problem->n_constraints, f_t{1});
  recompute_lhs(*ptr);
  invalidate_mtm_cache(*ptr);
  // The geometric repair must first repair the ruined assignment. Keep the
  // validated incumbent in the archive above while starting this local search
  // without an incumbent; the merge below retains the better validated point.
  if (reset_local_incumbent) {
    ptr->feasible_found   = false;
    ptr->h_best_objective = std::numeric_limits<f_t>::max();
  }
  cpufj_solve(ptr, time_limit, std::numeric_limits<double>::infinity());

  // A feasible ruined start can replace CPUFJ's best even when it is worse than
  // the previous neighborhood's incumbent. Retain only a validated improvement.
  auto candidate = ptr->h_best_assignment.underlying();
  const bool valid =
    ptr->feasible_found &&
    clamp_and_validate_cpufj_lns_seed(*ptr->problem, ptr->h_var_bounds.underlying(), candidate);
  const f_t objective =
    valid ? std::inner_product(
              candidate.begin(), candidate.end(), ptr->problem->h_obj_coeffs.begin(), f_t{0})
          : std::numeric_limits<f_t>::infinity();
  const bool improved = valid && objective + OBJECTIVE_EPSILON < archived_objective;
  ptr->feasible_found = improved || have_archive;
  if (improved) {
    ptr->h_best_assignment = std::move(candidate);
    ptr->h_best_objective  = objective;
  } else if (have_archive) {
    ptr->h_best_assignment = std::move(archived_assignment);
    ptr->h_best_objective  = archived_objective;
  } else {
    ptr->h_best_objective = std::numeric_limits<f_t>::max();
  }
  return improved;
}

// Incumbent-guided ruin-and-repair worker body. Runs entirely on one spare OMP thread, driving
// a single, already-constructed CPU FJ climber (`ptr`) through repeated ruin+repair bursts once
// the population has a feasible incumbent. It never touches device memory or population
// internals directly -- all communication is through the host snapshot and improvement callback, so
// it cannot race the main solve thread or the feasibility-finding scratch CPUFJ lanes.
template <typename i_t, typename f_t, typename model_validator_t>
void run_cpufj_lns_ruin_repair(
  fj_cpu_climber_t<i_t, f_t>* ptr,
  const std::function<bool(std::vector<f_t>&)>& snapshot,
  const model_validator_t& model_feasible,
  const std::function<void(std::vector<std::vector<f_t>>&)>& recent = {})
{
  cpufj_lns_stats_t stats;
  CUOPT_LOG_INFO("CPUFJ LNS starting mode=%s", cpufj_lns_experiment_name(cpufj_lns_experiment));
  const bool geometric_repair = configure_cpufj_lns_geometry(*ptr);
  if (geometric_repair) { CUOPT_LOG_DEBUG("CPUFJ LNS: enabling bound-aware geometric repair"); }
  const i_t n_vars = ptr->problem->n_variables;

  std::vector<i_t> integer_vars;
  integer_vars.reserve(n_vars);
  for (i_t v = 0; v < n_vars; ++v) {
    if (ptr->problem->h_var_types[v] == var_t::CONTINUOUS) continue;
    const auto bounds = ptr->h_var_bounds[v].get();
    if (get_upper(bounds) - get_lower(bounds) <= ptr->problem->tolerances.absolute_tolerance) {
      continue;
    }
    integer_vars.push_back(v);
  }
  if (integer_vars.empty()) return;

  std::mt19937 rng(static_cast<std::mt19937::result_type>(ptr->settings.seed));
  std::mt19937 policy_rng(static_cast<std::mt19937::result_type>(
    derive_seed(ptr->settings.seed, rng_id_t::local_search_cpu_fj, 1)));
  std::uniform_real_distribution<double> random_unit(0.0, 1.0);
  std::unique_ptr<cpufj_lns_similarity_t<i_t, f_t>> similarity;
  if constexpr (cpufj_lns_experiment == cpufj_lns_experiment_t::structural ||
                cpufj_lns_experiment == cpufj_lns_experiment_t::structural_state) {
    cuopt::timer_t setup_timer(0);
    similarity = std::make_unique<cpufj_lns_similarity_t<i_t, f_t>>(*ptr->problem, integer_vars);
    stats.selection_s += setup_timer.elapsed_time();
  }
  // Variables where the last adopted population incumbent disagreed with this climber's own
  // best-known point. Ruining preferentially from this pool is a crossover-style, population
  // guided neighborhood rather than uniform-random ruin.
  std::vector<i_t> guidance_pool;
  std::vector<uint8_t> chosen(n_vars, 0);
  std::vector<i_t> ruin_set;
  std::vector<f_t> pop_assignment, population_seed, rejected_seed, previous_seed;
  cuopt::timer_t rejection_log_timer(0.0);
  i_t consecutive_no_improve = 0;

  while (!ptr->halted.load(std::memory_order_relaxed) &&
         !ptr->preemption_flag.load(std::memory_order_relaxed)) {
    const bool available = snapshot(population_seed);
    const bool repeated_rejection =
      available && !rejected_seed.empty() && population_seed == rejected_seed;
    if (!available || repeated_rejection) {
      cuopt::timer_t wait_timer(0);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      (repeated_rejection ? stats.wait_rejected_s : stats.wait_empty_s) +=
        wait_timer.elapsed_time();
      continue;
    }
    if (population_seed != previous_seed) {
      ++stats.seed_versions;
      previous_seed = population_seed;
    }

    pop_assignment = population_seed;
    cpufj_lns_seed_validation_t validation;
    bool valid = false;
    if (population_seed.size() != static_cast<size_t>(n_vars)) {
      validation.rejection = cpufj_lns_rejection_t::size;
    } else {
      valid = clamp_and_validate_cpufj_lns_seed(*ptr, pop_assignment, model_feasible, &validation);
    }
    stats.record(validation, valid);
    if (!valid) {
      rejected_seed = population_seed;
      if (rejection_log_timer.check_time_limit()) {
        CUOPT_LOG_INFO(
          "CPUFJ LNS: skipping population seed that fails solver-model or private-domain "
          "validation (stage=%d reason=%d); waiting for a different seed",
          static_cast<int>(validation.stage),
          static_cast<int>(validation.rejection));
        rejection_log_timer = cuopt::timer_t(5.0);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    rejected_seed.clear();
    const f_t pop_objective = std::inner_product(
      pop_assignment.begin(), pop_assignment.end(), ptr->problem->h_obj_coeffs.begin(), f_t{0});
    const bool adopt_population_incumbent =
      !ptr->feasible_found || pop_objective + OBJECTIVE_EPSILON < (f_t)ptr->h_best_objective;
    if (adopt_population_incumbent) {
      guidance_pool.clear();
      for (i_t v : integer_vars) {
        if (!ptr->problem->integer_equal((f_t)ptr->h_assignment[v], pop_assignment[v])) {
          guidance_pool.push_back(v);
        }
      }
      ptr->h_assignment      = pop_assignment;
      ptr->h_best_assignment = pop_assignment;
      ptr->h_best_objective  = pop_objective;
      ptr->feasible_found    = true;
      consecutive_no_improve = 0;
    } else {
      // Restart the local search from its own best-known point before ruining it again.
      ptr->h_assignment = ptr->h_best_assignment.underlying();
    }

    cuopt::timer_t selection_timer(0);
    bool alternative_start = false;
    if constexpr (cpufj_lns_experiment == cpufj_lns_experiment_t::diverse_seeds) {
      if (recent && random_unit(policy_rng) < 0.1) {
        ++stats.alternative_draws;
        std::vector<std::vector<f_t>> candidates;
        std::vector<f_t> alternative;
        recent(candidates);
        if (choose_cpufj_lns_alternative(candidates,
                                         population_seed,
                                         ptr->h_best_assignment.underlying(),
                                         policy_rng,
                                         alternative)) {
          ++stats.alternative_candidates;
          const bool usable =
            clamp_and_validate_cpufj_lns_seed(*ptr, alternative, model_feasible, &validation);
          stats.record(validation, usable);
          if (!usable) {
            ++stats.alternative_rejections;
          } else if (alternative != pop_assignment &&
                     alternative != ptr->h_best_assignment.underlying()) {
            // Preserve h_best_assignment: the repair helper archives and merges it.
            ptr->h_assignment = std::move(alternative);
            alternative_start = true;
            ++stats.alternative_starts;
          }
        }
      }
    }

    const i_t base_size =
      std::min<i_t>(40, std::max<i_t>(6, static_cast<i_t>(integer_vars.size()) / 200));
    const i_t ruin_size =
      std::min<i_t>(static_cast<i_t>(integer_vars.size()),
                    base_size * (1 + std::min<i_t>(consecutive_no_improve / 6, 6)));

    ruin_set.clear();
    if (!guidance_pool.empty()) {
      std::shuffle(guidance_pool.begin(), guidance_pool.end(), rng);
      for (i_t v : guidance_pool) {
        if (static_cast<i_t>(ruin_set.size()) >= ruin_size) break;
        if (chosen[v]) continue;
        chosen[v] = 1;
        ruin_set.push_back(v);
      }
    }
    std::uniform_int_distribution<size_t> pick_dist(0, integer_vars.size() - 1);
    size_t guard = 0;
    while (static_cast<i_t>(ruin_set.size()) < ruin_size &&
           guard++ < integer_vars.size() * 4 + 16) {
      const i_t v = integer_vars[pick_dist(rng)];
      if (chosen[v]) continue;
      chosen[v] = 1;
      ruin_set.push_back(v);
    }
    for (i_t v : ruin_set) {
      chosen[v] = 0;
    }
    if (ruin_set.empty()) continue;

    if (similarity) {
      std::vector<i_t> neighbors;
      const f_t alpha = cpufj_lns_experiment == cpufj_lns_experiment_t::structural ? 1.0 : 0.5;
      similarity->select(
        ruin_set.front(), ptr->h_assignment.underlying(), ruin_size, alpha, policy_rng, neighbors);
      ++stats.structural_selections;
      stats.structural_neighbors += neighbors.size() - 1;
      for (i_t v : neighbors)
        chosen[v] = 1;
      // Keep the original fallback and its RNG consumption identical across C and D.
      for (i_t v : ruin_set) {
        if (static_cast<i_t>(neighbors.size()) >= ruin_size) break;
        if (!chosen[v]) {
          chosen[v] = 1;
          neighbors.push_back(v);
        }
      }
      ruin_set = std::move(neighbors);
      for (i_t v : ruin_set)
        chosen[v] = 0;
    }
    stats.selection_s += selection_timer.elapsed_time();

    // Ruin: force the chosen variables to re-decide, biased half the time toward the population
    // incumbent's value at that variable (a directed, crossover-like perturbation) and otherwise
    // toward a uniformly random point in-domain.
    for (i_t v : ruin_set) {
      const auto bounds = ptr->h_var_bounds[v].get();
      const f_t lo = std::ceil(get_lower(bounds)), hi = std::floor(get_upper(bounds));
      f_t new_value;
      if (random_unit(rng) < 0.5) {
        new_value = pop_assignment[v];
      } else if (std::isfinite(lo) && std::isfinite(hi) &&
                 lo >= static_cast<f_t>(std::numeric_limits<int64_t>::min()) &&
                 hi < static_cast<f_t>(std::numeric_limits<int64_t>::max())) {
        std::uniform_int_distribution<int64_t> value_dist(static_cast<int64_t>(lo),
                                                          static_cast<int64_t>(hi));
        new_value = static_cast<f_t>(value_dist(rng));
      } else {
        new_value = random_unit(rng) < 0.5 ? lo : hi;
        if (!std::isfinite(new_value)) new_value = (f_t)ptr->h_assignment[v];
      }
      ptr->h_assignment[v] = new_value;
    }

    const f_t repair_time_limit =
      std::min<f_t>(2., 0.15 + 0.02 * static_cast<f_t>(ruin_set.size()));
    cuopt::timer_t repair_timer(0);
    ++stats.repairs;
    if (repair_cpufj_lns_neighborhood(
          ptr, repair_time_limit, geometric_repair || alternative_start)) {
      ++stats.improvements;
      consecutive_no_improve = 0;
    } else {
      ++consecutive_no_improve;
    }
    stats.repair_s += repair_timer.elapsed_time();
  }
}

}  // namespace cuopt::mathematical_optimization::mip
