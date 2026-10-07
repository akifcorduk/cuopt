/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/feasibility_jump/cpu/search/api.hpp>
#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>
#include "cpufj_geometry.cuh"
#include "cpufj_validation.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
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

// Retain the RNG, guidance, and incumbent history when another LNS method shares this thread.
template <typename i_t, typename f_t>
class cpufj_lns_search_t {
 public:
  explicit cpufj_lns_search_t(fj_cpu_climber_t<i_t, f_t>* ptr)
    : ptr_(ptr),
      geometric_repair_(configure_cpufj_lns_geometry(*ptr)),
      rng_(static_cast<std::mt19937::result_type>(ptr->settings.seed)),
      chosen_(ptr->problem->n_variables, 0)
  {
    if (geometric_repair_) { CUOPT_LOG_DEBUG("CPUFJ LNS: enabling bound-aware geometric repair"); }
    const i_t n_vars = ptr->problem->n_variables;

    integer_vars_.reserve(n_vars);
    for (i_t v = 0; v < n_vars; ++v) {
      if (ptr->problem->h_var_types[v] == var_t::CONTINUOUS) continue;
      const auto bounds = ptr->h_var_bounds[v].get();
      if (get_upper(bounds) - get_lower(bounds) <= ptr->problem->tolerances.absolute_tolerance) {
        continue;
      }
      integer_vars_.push_back(v);
    }
  }

  bool stopped() const
  {
    return integer_vars_.empty() || ptr_->halted.load(std::memory_order_relaxed) ||
           ptr_->preemption_flag.load(std::memory_order_relaxed);
  }

  // One bounded ruin-and-repair neighborhood; no search state is reset between calls.
  void run_once(const std::function<bool(std::vector<f_t>&)>& snapshot)
  {
    if (stopped()) return;
    auto* ptr        = ptr_;
    const i_t n_vars = ptr->problem->n_variables;
    if (!snapshot(pop_assignment_) || pop_assignment_.size() != static_cast<size_t>(n_vars)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      return;
    }

    if (!clamp_and_validate_cpufj_lns_seed(
          *ptr->problem, ptr->h_var_bounds.underlying(), pop_assignment_)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      return;
    }
    const f_t pop_objective = std::inner_product(
      pop_assignment_.begin(), pop_assignment_.end(), ptr->problem->h_obj_coeffs.begin(), f_t{0});
    const bool adopt_population_incumbent =
      !ptr->feasible_found || pop_objective + OBJECTIVE_EPSILON < (f_t)ptr->h_best_objective;
    if (adopt_population_incumbent) {
      guidance_pool_.clear();
      for (i_t v : integer_vars_) {
        if (!ptr->problem->integer_equal((f_t)ptr->h_assignment[v], pop_assignment_[v])) {
          guidance_pool_.push_back(v);
        }
      }
      ptr->h_assignment       = pop_assignment_;
      ptr->h_best_assignment  = pop_assignment_;
      ptr->h_best_objective   = pop_objective;
      ptr->feasible_found     = true;
      consecutive_no_improve_ = 0;
    } else {
      // Restart the local search from its own best-known point before ruining it again.
      ptr->h_assignment = ptr->h_best_assignment.underlying();
    }

    const i_t base_size =
      std::min<i_t>(40, std::max<i_t>(6, static_cast<i_t>(integer_vars_.size()) / 200));
    const i_t ruin_size =
      std::min<i_t>(static_cast<i_t>(integer_vars_.size()),
                    base_size * (1 + std::min<i_t>(consecutive_no_improve_ / 6, 6)));

    ruin_set_.clear();
    if (!guidance_pool_.empty()) {
      std::shuffle(guidance_pool_.begin(), guidance_pool_.end(), rng_);
      for (i_t v : guidance_pool_) {
        if (static_cast<i_t>(ruin_set_.size()) >= ruin_size) break;
        if (chosen_[v]) continue;
        chosen_[v] = 1;
        ruin_set_.push_back(v);
      }
    }
    std::uniform_int_distribution<size_t> pick_dist(0, integer_vars_.size() - 1);
    size_t guard = 0;
    while (static_cast<i_t>(ruin_set_.size()) < ruin_size &&
           guard++ < integer_vars_.size() * 4 + 16) {
      const i_t v = integer_vars_[pick_dist(rng_)];
      if (chosen_[v]) continue;
      chosen_[v] = 1;
      ruin_set_.push_back(v);
    }
    for (i_t v : ruin_set_) {
      chosen_[v] = 0;
    }
    if (ruin_set_.empty()) return;

    // Ruin: force the chosen variables to re-decide, biased half the time toward the population
    // incumbent's value at that variable (a directed, crossover-like perturbation) and otherwise
    // toward a uniformly random point in-domain.
    for (i_t v : ruin_set_) {
      const auto bounds = ptr->h_var_bounds[v].get();
      const f_t lo = std::ceil(get_lower(bounds)), hi = std::floor(get_upper(bounds));
      f_t new_value;
      if (random_unit_(rng_) < 0.5) {
        new_value = pop_assignment_[v];
      } else if (std::isfinite(lo) && std::isfinite(hi) &&
                 lo >= static_cast<f_t>(std::numeric_limits<int64_t>::min()) &&
                 hi < static_cast<f_t>(std::numeric_limits<int64_t>::max())) {
        std::uniform_int_distribution<int64_t> value_dist(static_cast<int64_t>(lo),
                                                          static_cast<int64_t>(hi));
        new_value = static_cast<f_t>(value_dist(rng_));
      } else {
        new_value = random_unit_(rng_) < 0.5 ? lo : hi;
        if (!std::isfinite(new_value)) new_value = (f_t)ptr->h_assignment[v];
      }
      ptr->h_assignment[v] = new_value;
    }

    const f_t repair_time_limit =
      std::min<f_t>(2., 0.15 + 0.02 * static_cast<f_t>(ruin_set_.size()));
    if (repair_cpufj_lns_neighborhood(ptr, repair_time_limit, geometric_repair_)) {
      consecutive_no_improve_ = 0;
    } else {
      ++consecutive_no_improve_;
    }
  }

 private:
  fj_cpu_climber_t<i_t, f_t>* ptr_;
  bool geometric_repair_;
  std::vector<i_t> integer_vars_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> random_unit_{0.0, 1.0};
  // Prefer variables that disagree with the last adopted population incumbent.
  std::vector<i_t> guidance_pool_;
  std::vector<uint8_t> chosen_;
  std::vector<i_t> ruin_set_;
  std::vector<f_t> pop_assignment_;
  i_t consecutive_no_improve_{0};
};

// All communication uses host snapshots and the climber's improvement callback.
template <typename i_t, typename f_t>
void run_cpufj_lns_ruin_repair(fj_cpu_climber_t<i_t, f_t>* ptr,
                               const std::function<bool(std::vector<f_t>&)>& snapshot)
{
  cpufj_lns_search_t<i_t, f_t> search(ptr);
  while (!search.stopped())
    search.run_once(snapshot);
}

}  // namespace cuopt::mathematical_optimization::mip
