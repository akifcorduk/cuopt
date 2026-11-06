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

constexpr bool default_use_pdfp_projection   = false;
constexpr bool default_gpu_heuristics_only   = true;
constexpr bool default_skip_initial_fj       = true;
constexpr bool default_pdfp_spectral_step    = false;
constexpr bool default_paper_tolerance       = false;
constexpr bool default_pdfp_paper_warm_start = false;
constexpr bool default_pure_distance         = false;

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

// Skip the FJ phases before the first FP (rounding + FJ from the LP optimum, FJ from zero); FP
// starts from the root LP solution.
inline bool skip_initial_fj() { return env_flag("CUOPT_FP_EXP_SKIP_FJ", default_skip_initial_fj); }

// PDFP step 0.998/sqrt(2): the Pock-Chambolle scaling already certifies ||A||_2 <= 1.
inline bool pdfp_spectral_step()
{
  return env_flag("CUOPT_FP_EXP_PDFP_STEP", default_pdfp_spectral_step);
}

// Paper tolerance policy for either projector: relative residuals (primal 0.1 * eps), eps from 1e-2
// decaying by 0.98 per projection to 1e-4, and no per-projection time cap.
inline bool paper_tolerance()
{
  return env_flag("CUOPT_FP_EXP_PAPER_TOL", default_paper_tolerance);
}

// Paper warm start for PDFP: first projection from the root LP primal, later projections anchored
// at the last Halpern iterate.
inline bool pdfp_paper_warm_start()
{
  return env_flag("CUOPT_FP_EXP_PDFP_WARM", default_pdfp_paper_warm_start);
}

// Pure L1 distance projections (no objective term), as in the paper.
inline bool pure_distance() { return env_flag("CUOPT_FP_EXP_PURE_DIST", default_pure_distance); }

}  // namespace cuopt::mathematical_optimization::mip::fp_experiment
