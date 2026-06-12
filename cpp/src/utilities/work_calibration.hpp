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

namespace cuopt {

/**
 * @brief Converts wall-clock seconds into work units.
 *
 * This exists ONLY to seed sensible default absolute work-unit limits so the first
 * version of the work-limit machinery reproduces today's time-limit behavior. There is
 * no global work-unit budget and no ratios: each sub-algorithm gets its own absolute
 * work-unit limit, and this helper is what turns a legacy time cap into a starting value.
 *
 * Modes:
 *   - Opportunistic: uses a measured rate (work units recorded / wall seconds elapsed),
 *     falling back to default_wups until a measurement is available.
 *   - Deterministic: uses a fixed default_wups so seeded budgets are fully reproducible
 *     (a wall-clock-derived rate would reintroduce nondeterminism).
 */
struct work_calibrator_t {
  double default_wups{1.0e9};  // fixed deterministic rate / opportunistic fallback
  bool deterministic{false};
  double measured_wups{0.0};  // running estimate, opportunistic only

  double current_wups() const noexcept
  {
    if (deterministic) return default_wups;
    return measured_wups > 0.0 ? measured_wups : default_wups;
  }

  // Update the measured rate from cumulative work units and elapsed wall seconds.
  // No-op in deterministic mode to preserve reproducibility.
  void update(double total_work_units, double elapsed_seconds) noexcept
  {
    if (deterministic) return;
    if (elapsed_seconds > 1e-6 && total_work_units > 0.0) {
      measured_wups = total_work_units / elapsed_seconds;
    }
  }

  double work_units_from_time(double seconds) const noexcept
  {
    return std::max<double>(0.0, seconds) * current_wups();
  }
};

}  // namespace cuopt
