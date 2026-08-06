/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <mip_heuristics/deterministic_calibrator/calibrator.hpp>

#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::mip::calib {

// Names of the bound-presolve feature vector entries (base structural features + bound-presolve
// dynamic per-iteration features). Both per-kernel sub-models share this layout.
std::vector<std::string> bp_feature_names();

// Activity and bounds-update kernels are timed separately; per-iteration bound-presolve work is the
// sum of the two sub-models. `activity[i]` and `update[i]` are the same feature point (paired).
struct bp_samples_t {
  std::vector<calibration_sample_t> activity;
  std::vector<calibration_sample_t> update;
};

// Load `mps_path` and run the real bound-presolve fixpoint loop, timing calculate_activity and
// calculate_bounds_update separately each iteration. Returns one paired sample per distinct
// dynamic-feature point (deduped; median time over repeats).
bp_samples_t run_bp_calibration_samples(const std::string& mps_path,
                                        const std::string& instance_name);

}  // namespace cuopt::mathematical_optimization::mip::calib
