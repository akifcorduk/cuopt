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
#include <array>
#include <chrono>
#include <string>

#include <mip_heuristics/logger.hpp>

#include "timer.hpp"
#include "work_unit_scheduler.hpp"

namespace cuopt {

// Per-leaf attribution buckets for the heuristic phase. Used only for diagnostic logging (wall vs
// charged work per leaf): a leaf whose accumulated wall greatly exceeds its accumulated work is
// under-charged (mis- or un-calibrated). Measuring wall never gates control flow, so it does not
// affect determinism.
enum class heur_leaf_t : int {
  root_lp = 0,
  fj,
  relaxed_lp,
  bounds_presolve,
  multi_probe,
  bounds_repair,
  line_segment,
  SIZE
};

inline const char* heur_leaf_name(heur_leaf_t l)
{
  switch (l) {
    case heur_leaf_t::root_lp: return "root_lp";
    case heur_leaf_t::fj: return "fj";
    case heur_leaf_t::relaxed_lp: return "relaxed_lp";
    case heur_leaf_t::bounds_presolve: return "bounds_presolve";
    case heur_leaf_t::multi_probe: return "multi_probe";
    case heur_leaf_t::bounds_repair: return "bounds_repair";
    case heur_leaf_t::line_segment: return "line_segment";
    default: return "unknown";
  }
}

struct work_limit_context_t {
  double global_work_units_elapsed{0.0};
  double total_sync_time{0.0};  // Total time spent waiting at sync barriers (seconds)
  bool deterministic{false};
  work_unit_scheduler_t* scheduler{nullptr};
  std::string name;

  // Per-leaf (wall seconds, charged work units, call count) accumulators for diagnostic logging.
  static constexpr int n_leaves = static_cast<int>(heur_leaf_t::SIZE);
  std::array<double, n_leaves> leaf_wall{};
  std::array<double, n_leaves> leaf_work{};
  std::array<long, n_leaves> leaf_calls{};

  void add_leaf(heur_leaf_t l, double wall, double work)
  {
    const int i = static_cast<int>(l);
    leaf_wall[i] += wall;
    leaf_work[i] += work;
    leaf_calls[i] += 1;
  }

  work_limit_context_t(const std::string& name) : name(name) {}

  // Current accumulated work units (interface used by termination_checker_t).
  double current_work() const noexcept { return global_work_units_elapsed; }

  // Accumulate work units in BOTH modes (pure accounting; no synchronization). Safe in
  // opportunistic mode where nothing gates control flow on it.
  void record_work(double work) { global_work_units_elapsed += work; }

  // Accumulate work and, in deterministic mode only, synchronize on horizon boundaries.
  void record_work_sync_on_horizon(double work)
  {
    record_work(work);
    if (!deterministic) return;
    if (scheduler) { scheduler->on_work_recorded(*this, global_work_units_elapsed); }
  }
};

// RAII scope that attributes the wall time and the work charged during its lifetime to one leaf.
// Place at a leaf's top-level entry; scopes for different leaves must not overlap (otherwise wall
// is double counted). Diagnostic only.
struct leaf_work_scope_t {
  work_limit_context_t& ctx;
  heur_leaf_t leaf;
  std::chrono::steady_clock::time_point t0;
  double work0;

  leaf_work_scope_t(work_limit_context_t& ctx_, heur_leaf_t leaf_)
    : ctx(ctx_),
      leaf(leaf_),
      t0(std::chrono::steady_clock::now()),
      work0(ctx_.global_work_units_elapsed)
  {
  }

  ~leaf_work_scope_t()
  {
    const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ctx.add_leaf(leaf, wall, ctx.global_work_units_elapsed - work0);
  }

  leaf_work_scope_t(const leaf_work_scope_t&)            = delete;
  leaf_work_scope_t& operator=(const leaf_work_scope_t&) = delete;
};

}  // namespace cuopt
