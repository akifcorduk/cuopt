/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdlib>

namespace cuopt::mathematical_optimization::mip::gpu_submip_experiment {

// Defaults of the GPU sub-MIP experiment. Benchmark commits change them; the environment variable
// named next to each one overrides it.
// CUOPT_GPU_SUBMIP: run the GPU sub-MIP worker.
inline constexpr bool default_enabled = true;
// CUOPT_SUBMIP_POOL_DIAGNOSTICS: log the CPU sub-MIP pool's tasks and schedule.
inline constexpr bool default_pool_diagnostics = true;
// CUOPT_GPU_SUBMIP_PDFP: PDFP instead of PDLP as the FP session's projector.
inline constexpr bool default_pdfp = false;
// CUOPT_GPU_SUBMIP_PROJECTION_SECONDS: time limit of one projection; 0 keeps the FP preset's.
inline constexpr double default_projection_seconds = 0.;
// CUOPT_GPU_SUBMIP_SCREEN_SECONDS: time for the screen's LP run before the descent; 0 turns the
// screen off. Needs PDFP.
inline constexpr double default_screen_seconds = 0.;
// CUOPT_GPU_SUBMIP_TARGET_SHARE: share of the screen's LP gain the descent has to reach. Needs
// the screen.
inline constexpr double default_target_share = 0.;
// CUOPT_GPU_SUBMIP_LP_START: start the descent from the screen's LP point. Needs the screen.
inline constexpr bool default_lp_start = false;
// CUOPT_GPU_SUBMIP_RENS_NO_INCUMBENT: before the first incumbent, give the worker the RENS
// neighbourhoods of the tree nodes, searched without a cutoff.
inline constexpr bool default_rens_without_incumbent = false;

// Benchmark commits set this to false, so that their defaults hold whatever the environment
// says: the cluster's benchmark script exports CUOPT_GPU_SUBMIP, CUOPT_SUBMIP_POOL_DIAGNOSTICS,
// CUOPT_GPU_SUBMIP_CONFIG_FAMILY, CUOPT_CONFIG_ID and CUOPT_MAX_CONFIG.
inline constexpr bool default_read_environment = false;

// The value of a variable of the experiment, or null when the environment is not read.
inline const char* environment(const char* name)
{
  return default_read_environment ? std::getenv(name) : nullptr;
}

inline bool flag(const char* name, bool fallback)
{
  const char* value = environment(name);
  return value == nullptr || value[0] == '\0' ? fallback : std::atoi(value) != 0;
}

inline double number(const char* name, double fallback)
{
  const char* value = environment(name);
  return value == nullptr || value[0] == '\0' ? fallback : std::atof(value);
}

inline bool enabled() { return flag("CUOPT_GPU_SUBMIP", default_enabled); }
inline bool pool_diagnostics()
{
  return flag("CUOPT_SUBMIP_POOL_DIAGNOSTICS", default_pool_diagnostics);
}
inline bool pdfp() { return flag("CUOPT_GPU_SUBMIP_PDFP", default_pdfp); }
inline double projection_seconds()
{
  return number("CUOPT_GPU_SUBMIP_PROJECTION_SECONDS", default_projection_seconds);
}
inline double screen_seconds()
{
  return number("CUOPT_GPU_SUBMIP_SCREEN_SECONDS", default_screen_seconds);
}
inline double target_share()
{
  return number("CUOPT_GPU_SUBMIP_TARGET_SHARE", default_target_share);
}
inline bool lp_start() { return flag("CUOPT_GPU_SUBMIP_LP_START", default_lp_start); }
inline bool rens_without_incumbent()
{
  return flag("CUOPT_GPU_SUBMIP_RENS_NO_INCUMBENT", default_rens_without_incumbent);
}

}  // namespace cuopt::mathematical_optimization::mip::gpu_submip_experiment
