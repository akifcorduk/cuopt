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

// Names of the FJ feature vector entries (base structural features + FJ-specific dynamic feature).
std::vector<std::string> fj_feature_names();

// Load `mps_path`, run real GPU feasibility-jump for two iteration counts, and return one
// calibration sample: the FJ feature vector for this instance plus the measured steady-state
// per-step wall time (seconds). Uses a two-point measurement so fixed setup/teardown cancels.
calibration_sample_t run_fj_calibration_sample(const std::string& mps_path,
                                               const std::string& instance_name);

}  // namespace cuopt::mathematical_optimization::mip::calib
