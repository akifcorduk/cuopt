/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <atomic>
#include <chrono>
#include <thread>

namespace cuopt {

// Strong branching oracle experiment: wall time excluded from every solver clock, and a freeze that
// background heuristics honor while that time elapses.
inline std::atomic<double>& excluded_wall_time()
{
  static std::atomic<double> seconds{0.0};
  return seconds;
}

inline std::atomic<bool>& solver_frozen()
{
  static std::atomic<bool> frozen{false};
  return frozen;
}

inline void wait_while_solver_frozen()
{
  while (solver_frozen().load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

}  // namespace cuopt
