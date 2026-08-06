/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <string>

namespace cuopt::mathematical_optimization::mip::calib {

// Standalone FP-leaf work-unit verification for one instance. For a set of pseudo-second work
// budgets it converts each into a deterministic per-leaf iteration budget via the calibrated work
// model (context.pdlp_iters_for_budget / fj_steps_for_budget), runs the leaf to that budget TWICE,
// and checks: (a) the iteration budget is deterministic, (b) the run is reproducible (identical
// step count + objective/hash across runs), (c) measured wall time ~ budget (calibration accuracy).
// Prints a per-instance report. Returns true if all checks pass.
bool run_fp_verify(const std::string& mps_path, const std::string& instance_name);

}  // namespace cuopt::mathematical_optimization::mip::calib
