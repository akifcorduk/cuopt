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

// Names of the bound-presolve feature vector entries (base structural features + bound-presolve
// dynamic per-iteration features).
std::vector<std::string> bp_feature_names();

// Load `mps_path` and run the real bound-presolve fixpoint loop, timing each iteration. Returns one
// calibration sample per iteration (the per-iteration dynamic features vary as the loop converges:
// iter 0 touches all constraints, later iters only the changed set). Per-iteration time is the
// median over repeats (iteration counts are deterministic across repeats).
std::vector<calibration_sample_t> run_bp_calibration_samples(const std::string& mps_path,
                                                             const std::string& instance_name);

}  // namespace cuopt::linear_programming::detail::calib
