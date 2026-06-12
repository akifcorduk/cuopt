/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <string>

#include <mip_heuristics/logger.hpp>

#include "timer.hpp"
#include "work_unit_scheduler.hpp"

namespace cuopt {

struct work_limit_context_t {
  double global_work_units_elapsed{0.0};
  double total_sync_time{0.0};  // Total time spent waiting at sync barriers (seconds)
  bool deterministic{false};
  work_unit_scheduler_t* scheduler{nullptr};
  std::string name;

  work_limit_context_t(const std::string& name) : name(name) {}

  // Accumulate work units in BOTH modes. This is pure accounting (no synchronization),
  // so it is safe in opportunistic mode where it only feeds work_meter_t budgets and
  // never gates control flow (work_limit is infinity there).
  void record_work(double work) { global_work_units_elapsed += work; }

  // Accumulate work and, in deterministic mode only, synchronize on horizon boundaries.
  void record_work_sync_on_horizon(double work)
  {
    record_work(work);
    if (!deterministic) return;
    if (scheduler) { scheduler->on_work_recorded(*this, global_work_units_elapsed); }
  }
};

// A work-budget "clock" that mirrors timer_t but is measured in work units accumulated
// in a work_limit_context_t. Components express sub-budgets in work units instead of
// wall-clock seconds; the API intentionally parallels timer_t to make conversion of
// existing call sites mechanical.
class work_meter_t {
 public:
  work_meter_t()                    = delete;
  work_meter_t(const work_meter_t&) = default;

  work_meter_t(const work_limit_context_t& ctx, double work_budget)
    : ctx_(&ctx), start_(ctx.global_work_units_elapsed), work_budget_(work_budget)
  {
  }

  double elapsed_work() const noexcept { return ctx_->global_work_units_elapsed - start_; }

  bool check_work_limit() const noexcept { return elapsed_work() >= work_budget_; }

  double remaining_work() const noexcept
  {
    return std::max<double>(0.0, work_budget_ - elapsed_work());
  }

  double clamp_remaining_work(double desired_work) const noexcept
  {
    return std::min<double>(desired_work, remaining_work());
  }

  double get_work_limit() const noexcept { return work_budget_; }

 private:
  const work_limit_context_t* ctx_;
  double start_;
  double work_budget_;
};

}  // namespace cuopt
