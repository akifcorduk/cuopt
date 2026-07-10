/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "feasibility_pump.cuh"

namespace cuopt::mathematical_optimization::mip {

struct vanilla_fp_config_t {
  int diversity_climber = 1;
  int diversity_seed    = 42;
};

bool run_vanilla_fp_descent(feasibility_pump_t<int, double>& fp,
                            solution_t<int, double>& solution,
                            const vanilla_fp_config_t& config);

}  // namespace cuopt::mathematical_optimization::mip
