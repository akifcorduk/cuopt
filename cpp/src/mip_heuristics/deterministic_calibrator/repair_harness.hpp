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

// bounds_repair per-move work model. Each repair move: compute_best_shift (chosen constraint row),
// compute_damages (per candidate column + sort), apply_move, get_ii_violation (full-matrix activity
// pass -> dominant, uniform). Feature layout (hinge on the last = total_warp_loads):
//   base(6) + [n_candidates, cur_cstr_rowsize, total_warp_loads]
std::vector<std::string> repair_feature_names();

// Load `mps_path`, fix integers to create an infeasible state, run the repair loop, and return one
// calibration sample (median per-move time + median dynamic features). Empty if the fixed problem
// is not bound-infeasible (no repair moves).
std::vector<calibration_sample_t> run_repair_calibration_sample(const std::string& mps_path,
                                                                const std::string& instance_name);

}  // namespace cuopt::linear_programming::detail::calib
