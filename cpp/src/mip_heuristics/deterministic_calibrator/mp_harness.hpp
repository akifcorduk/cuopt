/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <mip_heuristics/deterministic_calibrator/bp_harness.hpp>  // bp_samples_t

#include <string>
#include <vector>

namespace cuopt::linear_programming::detail::calib {

// multi_probe runs the same activity/bounds-update kernels as bound presolve, over two probe
// buffers. The activity kernel scales with changed *constraints*; the bounds-update kernel scales
// with changed *variables*. Feature layout (shared by both per-kernel sub-models; each NNLS fit
// picks the relevant ones, plus its own saturation hinge):
//   base(6) + max_row_nnz + [n_changed_cons, changed_cons_nnz, changed_cons_warp_loads]
//           + [n_changed_vars, changed_vars_warp_loads]
// changed_cons_warp_loads is at index `mp_cons_warp_index()`, changed_vars_warp_loads at
// `mp_vars_warp_index()` (the activity / update hinges respectively).
std::vector<std::string> mp_feature_names();
std::size_t mp_cons_warp_index();
std::size_t mp_vars_warp_index();

// Returns paired per-kernel samples (activity + bounds-update) per distinct feature point. Both
// share the mp_feature_names() layout; dynamic features are summed over both probes.
bp_samples_t run_mp_calibration_samples(const std::string& mps_path,
                                        const std::string& instance_name);

}  // namespace cuopt::linear_programming::detail::calib
