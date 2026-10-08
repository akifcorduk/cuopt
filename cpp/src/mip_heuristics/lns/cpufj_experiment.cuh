/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cpufj_validation.cuh"

#include <utilities/logger.hpp>
#include <utilities/timer.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

enum class cpufj_lns_experiment_t { control, diverse_seeds, structural, structural_state };
// Each benchmark commit changes only this selector, with identical instrumentation.
inline constexpr auto cpufj_lns_experiment = cpufj_lns_experiment_t::structural_state;

inline const char* cpufj_lns_experiment_name(cpufj_lns_experiment_t experiment)
{
  switch (experiment) {
    case cpufj_lns_experiment_t::diverse_seeds: return "B-diverse-seeds";
    case cpufj_lns_experiment_t::structural: return "C-structural";
    case cpufj_lns_experiment_t::structural_state: return "D-structural-state";
    default: return "A-control";
  }
}

// Uniform over distinct alternatives, excluding both population and worker best.
// The existing feed holds at most eight recent members.
template <typename f_t>
bool choose_cpufj_lns_alternative(const std::vector<std::vector<f_t>>& recent,
                                  const std::vector<f_t>& population_best,
                                  const std::vector<f_t>& worker_best,
                                  std::mt19937& rng,
                                  std::vector<f_t>& out)
{
  std::vector<size_t> distinct;
  for (size_t i = 0; i < recent.size(); ++i) {
    if (recent[i].size() != population_best.size() || recent[i] == population_best ||
        recent[i] == worker_best)
      continue;
    const bool duplicate = std::any_of(
      distinct.begin(), distinct.end(), [&](size_t j) { return recent[i] == recent[j]; });
    if (!duplicate) distinct.push_back(i);
  }
  if (distinct.empty()) return false;
  out = recent[distinct[std::uniform_int_distribution<size_t>(0, distinct.size() - 1)(rng)]];
  return true;
}

struct cpufj_lns_stats_t {
  uint64_t seed_versions = 0, validations = 0, accepted = 0, projections = 0;
  uint64_t alternative_draws = 0, alternative_candidates = 0, alternative_starts = 0;
  uint64_t alternative_rejections = 0, repairs = 0, improvements = 0;
  uint64_t structural_selections = 0, structural_neighbors = 0;
  // First failure, partitioned by stage and reason; successful revalidations are not failures.
  std::array<std::array<uint64_t, 6>, 3> rejected{};
  double wait_empty_s = 0, wait_rejected_s = 0, selection_s = 0, repair_s = 0;
  cuopt::timer_t lifetime{0};

  void record(const cpufj_lns_seed_validation_t& result, bool valid)
  {
    ++validations;
    projections += result.projected;
    accepted += valid;
    if (!valid)
      ++rejected[static_cast<size_t>(result.stage)][static_cast<size_t>(result.rejection)];
  }

  // Scope lifetime also emits the counters if a worker unwinds with an exception.
  ~cpufj_lns_stats_t()
  {
    CUOPT_LOG_INFO(
      "LNS stats mode=%s seed_versions=%llu validations=%llu accepted=%llu projections=%llu "
      "wait_empty_s=%.6f wait_rejected_s=%.6f",
      cpufj_lns_experiment_name(cpufj_lns_experiment),
      (unsigned long long)seed_versions,
      (unsigned long long)validations,
      (unsigned long long)accepted,
      (unsigned long long)projections,
      wait_empty_s,
      wait_rejected_s);
    const std::array<const char*, 3> stages{"model", "private", "projected_model"};
    for (size_t i = 0; i < stages.size(); ++i) {
      CUOPT_LOG_INFO(
        "LNS rejects stage=%s size=%llu domain=%llu objective=%llu rows=%llu unknown=%llu",
        stages[i],
        (unsigned long long)rejected[i][1],
        (unsigned long long)rejected[i][2],
        (unsigned long long)rejected[i][3],
        (unsigned long long)rejected[i][4],
        (unsigned long long)rejected[i][5]);
    }
    CUOPT_LOG_INFO(
      "LNS work alternative_draws=%llu alternative_candidates=%llu alternative_starts=%llu "
      "alternative_rejections=%llu structural_selections=%llu structural_neighbors=%llu "
      "repairs=%llu improvements=%llu selection_s=%.6f repair_s=%.6f lifetime_s=%.6f",
      (unsigned long long)alternative_draws,
      (unsigned long long)alternative_candidates,
      (unsigned long long)alternative_starts,
      (unsigned long long)alternative_rejections,
      (unsigned long long)structural_selections,
      (unsigned long long)structural_neighbors,
      (unsigned long long)repairs,
      (unsigned long long)improvements,
      selection_s,
      repair_s,
      lifetime.elapsed_time());
  }
};

}  // namespace cuopt::mathematical_optimization::mip
