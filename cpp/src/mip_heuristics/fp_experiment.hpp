/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdlib>

// Switches of the FP projection experiment (epigraph PDLP vs PDFP). The compiled defaults are what
// benchmark runs use; the environment variables override them for local runs.
namespace cuopt::mathematical_optimization::mip::fp_experiment {

constexpr bool default_use_pdfp_projection = false;
constexpr bool default_gpu_heuristics_only = true;

inline bool env_flag(const char* name, bool compiled_default)
{
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') { return compiled_default; }
  return value[0] == '1' || value[0] == 't' || value[0] == 'T' || value[0] == 'y' ||
         value[0] == 'Y';
}

inline bool use_pdfp_projection()
{
  return env_flag("CUOPT_FP_EXP_PDFP", default_use_pdfp_projection);
}

// Run only the GPU heuristic thread: no pre-solve/presolve-time FJ or structural heuristics, no
// branch and bound, no root structural heuristic, no CPUFJ threads, and a PDLP-only root LP.
inline bool gpu_heuristics_only()
{
  return env_flag("CUOPT_FP_EXP_GPU_ONLY", default_gpu_heuristics_only);
}

}  // namespace cuopt::mathematical_optimization::mip::fp_experiment
