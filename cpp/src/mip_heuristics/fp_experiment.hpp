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
constexpr bool default_pdfp_cusparse_spmv    = true;
constexpr bool default_pdfp_cuda_graph       = true;
constexpr bool default_pdfp_prox_distance    = false;
constexpr bool default_pdfp_primal_tolerance = false;
constexpr bool default_pdfp_gap_test         = false;
constexpr bool default_pdfp_balanced_weight  = false;
constexpr bool default_pdfp_damped_weight    = false;
constexpr bool default_pdfp_wide_weight      = false;
constexpr bool default_pdfp_target_start     = false;
constexpr bool default_pdfp_shadow           = false;
constexpr bool default_ratio_tolerance       = false;
constexpr int default_pdfp_check_interval    = 200;
constexpr int default_fj_check_floor_ms      = 0;

inline bool env_flag(const char* name, bool compiled_default)
{
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') { return compiled_default; }
  return value[0] == '1' || value[0] == 't' || value[0] == 'T' || value[0] == 'y' ||
         value[0] == 'Y';
}

inline int env_int(const char* name, int compiled_default)
{
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') { return compiled_default; }
  return std::atoi(value);
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

// PDFP SpMV through cuSPARSE (CSR ALG2 on scaled copies of A and A^T, as in PDLP) instead of the
// one-warp-per-row kernel, which serializes long rows/columns and the dense objective cut row.
inline bool pdfp_cusparse_spmv()
{
  return env_flag("CUOPT_FP_EXP_PDFP_CUSPARSE", default_pdfp_cusparse_spmv);
}

// PDFP iterations between residual checks with the Halpern update fused into the primal and dual
// steps, replayed as captured CUDA graphs.
inline bool pdfp_cuda_graph()
{
  return env_flag("CUOPT_FP_EXP_PDFP_GRAPH", default_pdfp_cuda_graph);
}

// PDFP handles the L1 distance in the primal step (shrinkage towards the target) instead of a
// bounded dual per integer variable. The operator is then A alone, so the step is sqrt(2) larger
// and a target at a variable bound costs nothing extra, as in the epigraph LP.
inline bool pdfp_prox_distance()
{
  return env_flag("CUOPT_FP_EXP_PDFP_PROX", default_pdfp_prox_distance);
}

// PDFP primal test as cuOpt judges a row: violation <= tolerance * (1 + rel/abs * |bound|),
// instead of the paper's absolute 0.1 * tolerance on every row.
inline bool pdfp_primal_tolerance()
{
  return env_flag("CUOPT_FP_EXP_PDFP_PTOL", default_pdfp_primal_tolerance);
}

// PDFP stops as PDLP stops the epigraph LP: on the duality gap, with reduced costs that finite
// bounds absorb not counted as dual residual.
inline bool pdfp_gap_test() { return env_flag("CUOPT_FP_EXP_PDFP_GAP", default_pdfp_gap_test); }

// PDFP starts each projection from the primal weight (||cost|| + 1) / (||row bounds|| + 1).
inline bool pdfp_balanced_weight()
{
  return env_flag("CUOPT_FP_EXP_PDFP_W0", default_pdfp_balanced_weight);
}

// PDFP updates the primal weight with PDLP's smoothing 0.5 instead of the gain 0.99.
inline bool pdfp_damped_weight()
{
  return env_flag("CUOPT_FP_EXP_PDFP_KP", default_pdfp_damped_weight);
}

// PDFP primal weight limited to [1e-12, 1e12] instead of [1e-6, 1e6].
inline bool pdfp_wide_weight()
{
  return env_flag("CUOPT_FP_EXP_PDFP_WIDE", default_pdfp_wide_weight);
}

// PDFP starts the primal iterate at the target, as the epigraph LP is warm-started, instead of
// at the previous projection.
inline bool pdfp_target_start()
{
  return env_flag("CUOPT_FP_EXP_PDFP_TSTART", default_pdfp_target_start);
}

// Diagnostic for the epigraph FP: PDFP also solves every projection target (FPS trace line),
// warm-started from its own previous shadow projection. The pump uses the epigraph result.
inline bool pdfp_shadow() { return env_flag("CUOPT_FP_EXP_PDFP_SHADOW", default_pdfp_shadow); }

// Projection tolerance from the previous projection's integrality (0.1 / 0.01 / 0.001 / absolute),
// as get_tolerance_from_ratio intends; the rounding's count makes the ratio always 1.
inline bool ratio_tolerance() { return env_flag("CUOPT_FP_EXP_RATIO", default_ratio_tolerance); }

// PDFP iterations between residual checks (convergence and restart tests). The reference uses
// 200; PDLP evaluates the epigraph LP every 40.
inline int pdfp_check_interval()
{
  return env_int("CUOPT_FP_EXP_PDFP_CHECK", default_pdfp_check_interval);
}

// Minimum time limit of the FJ check after a rounding, in milliseconds. The check gets 20% of the
// projection + rounding time, which a fast projector shrinks below what FJ can use.
inline int fj_check_floor_ms()
{
  return env_int("CUOPT_FP_EXP_FJ_FLOOR_MS", default_fj_check_floor_ms);
}

}  // namespace cuopt::mathematical_optimization::mip::fp_experiment
