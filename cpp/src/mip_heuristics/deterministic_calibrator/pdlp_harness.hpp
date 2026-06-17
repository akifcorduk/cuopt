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

namespace cuopt::linear_programming::detail::calib {

// Feature layout for the PDLP per-iteration work model (base structural features + SpMV warp-load
// depth terms). The last entry is total_warp_loads, on which the saturation hinge is built.
std::vector<std::string> pdlp_feature_names();

// Load `mps_path` and run real PDLP for two iteration counts (two-point timing cancels
// setup/scaling), returning one calibration sample (PDLP inner iterations are uniform).
calibration_sample_t run_pdlp_calibration_sample(const std::string& mps_path,
                                                 const std::string& instance_name);

}  // namespace cuopt::linear_programming::detail::calib
