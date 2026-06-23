/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

// Shared device-aware feature recipe for the GPU-generic work model.
//
// A leaf's per-iteration time is modeled as a sum of resource terms: each raw problem/dynamic count
// is divided by the device rate that bounds it (memory bandwidth for data volume, sm*clock for
// occupancy/compute), plus a clock-scaled launch latency and a constant. The fitted coefficients
// then carry units of "bytes per count" / "ops per count" and are GPU-INDEPENDENT: the same vector
// reproduces per-iteration time on any device once that device's gpu_features_t is plugged in.
//
// CRITICAL: the calibrator (fit) and the solver (runtime evaluation) MUST build the feature vector
// with this exact function, in this exact order, so a fitted header transfers across GPUs.

#include <mip_heuristics/deterministic_calibrator/gpu_features.hpp>

#include <vector>

namespace cuopt::linear_programming::detail::calib {

// Build the device-aware regression terms from a leaf's raw count features and the device
// descriptor. `hinge_excess` is the saturation term max(0, warp_loads - k*warp_capacity) already
// evaluated by the caller (0 if the leaf has no hinge).
//
// Layout (must stay stable):
//   [0]            constant            (fixed per-call overhead)
//   [1]            1 / sm_clock_ghz    (clock-scaled launch latency)
//   [2 + 2j]       raw[j] / mem_bw     (memory-bound contribution of count j)
//   [3 + 2j]       raw[j] / (sm*clock) (occupancy/compute contribution of count j)
//   [last]         hinge_excess / mem_bw   (reduction beyond hidden capacity, memory-bound)
inline std::vector<double> device_terms(const std::vector<double>& raw,
                                        const gpu_features_t& g,
                                        double hinge_excess)
{
  const double bw      = g.mem_bandwidth_gb_s > 0.0 ? g.mem_bandwidth_gb_s : 1.0;
  const double occ     = (g.sm_count * g.sm_clock_ghz) > 0.0 ? g.sm_count * g.sm_clock_ghz : 1.0;
  const double inv_clk = g.sm_clock_ghz > 0.0 ? 1.0 / g.sm_clock_ghz : 1.0;

  std::vector<double> x;
  x.reserve(2 * raw.size() + 3);
  x.push_back(1.0);
  x.push_back(inv_clk);
  for (double r : raw) {
    x.push_back(r / bw);
    x.push_back(r / occ);
  }
  x.push_back(hinge_excess / bw);
  return x;
}

// Number of device terms produced for `n_raw` raw count features (+ the trailing hinge term).
inline std::size_t device_term_count(std::size_t n_raw) { return 2 * n_raw + 3; }

}  // namespace cuopt::linear_programming::detail::calib
