/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

#include "work_calibration.hpp"

namespace cuopt {

// Static structural descriptor of the problem, computed once. Used by the structural
// budget policy to scale per-algorithm effort. Pure data; no device handles.
struct problem_features_t {
  std::size_t n_vars{0};
  std::size_t n_constraints{0};
  std::size_t nnz{0};
  double row_nnz_mean{0.0};
  double row_nnz_std{0.0};  // TODO(work-units): fill from CSR offsets
  double col_nnz_mean{0.0};
  double col_nnz_std{0.0};  // TODO(work-units): fill from CSR reverse_offsets
  double integer_fraction{0.0};
};

// Deterministic snapshot of solver state used to adapt budgets during the solve.
// MUST be populated only from deterministic counters / replay state (never wall clock)
// so deterministic mode stays reproducible.
struct solution_state_view_t {
  double incumbent_objective{std::numeric_limits<double>::infinity()};
  double bound{-std::numeric_limits<double>::infinity()};
  double mip_gap{std::numeric_limits<double>::infinity()};
  bool has_incumbent{false};
  int stagnation{0};
};

// Deterministic stop-gap for PDLP-based LP solves: convert a wall-clock time cap into a
// PDLP iteration cap (PDLP's iteration count is deterministic, unlike wall time). This is a
// rough guess to be calibrated; for now it is a tunable constant rate times the time cap.
// TODO(work-units): make size-aware (per-iteration PDLP cost scales with nnz) and/or
// calibrate online, then replace with a real work-unit budget.
inline int lp_iteration_limit_from_time(double time_limit_seconds,
                                        const problem_features_t& /*features*/,
                                        double iters_per_sec) noexcept
{
  const double iters = std::max(0.0, iters_per_sec) * std::max(0.0, time_limit_seconds);
  return iters < 1.0 ? 1 : (int)iters;
}

// Identifies the sub-algorithm requesting a budget, so the policy can apply
// per-algorithm limits / formulation.
enum class work_algorithm_t {
  feasibility_jump,
  feasibility_pump,
  line_search,
  rounding,
  relaxed_lp,
  recombiner,
  presolve,
  root_lp,
  rins,
  other
};

// Pluggable source of per-sub-algorithm work-unit budgets. Components ask the policy
// for an absolute work-unit limit; they never call the calibrator directly. Swapping
// the policy implementation changes the whole solver's budgeting without touching call
// sites.
class work_budget_policy_t {
 public:
  virtual ~work_budget_policy_t() = default;

  // Returns an absolute work-unit budget for one invocation of @p algo.
  // @p legacy_time_cap_seconds is the component's existing time cap, consumed by the
  // transitional time-calibrated policy; structural policies use @p features / @p state.
  virtual double budget_for(work_algorithm_t algo,
                            double legacy_time_cap_seconds,
                            const problem_features_t& features,
                            const solution_state_view_t& state) const = 0;
};

// Transitional policy: derive the budget from the legacy time cap via the calibrator,
// reproducing today's time-limit behavior.
class time_calibrated_policy_t : public work_budget_policy_t {
 public:
  explicit time_calibrated_policy_t(const work_calibrator_t& calibrator) : calibrator_(&calibrator)
  {
  }

  double budget_for(work_algorithm_t /*algo*/,
                    double legacy_time_cap_seconds,
                    const problem_features_t& /*features*/,
                    const solution_state_view_t& /*state*/) const override
  {
    return calibrator_->work_units_from_time(legacy_time_cap_seconds);
  }

 private:
  const work_calibrator_t* calibrator_;
};

// Goal policy: time-independent budget as a function of problem structure and solver
// state. Stubbed to the calibrated behavior for now so it is a safe drop-in; the
// structural model is filled in once benchmark statistics are available.
class structural_policy_t : public work_budget_policy_t {
 public:
  explicit structural_policy_t(const work_calibrator_t& calibrator) : calibrator_(&calibrator) {}

  double budget_for(work_algorithm_t /*algo*/,
                    double legacy_time_cap_seconds,
                    const problem_features_t& /*features*/,
                    const solution_state_view_t& /*state*/) const override
  {
    // TODO(work-units): replace with work_limit = f(features, state) once calibrated.
    return calibrator_->work_units_from_time(legacy_time_cap_seconds);
  }

 private:
  const work_calibrator_t* calibrator_;
};

}  // namespace cuopt
