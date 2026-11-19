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
inline constexpr bool default_enabled = false;
// CUOPT_SUBMIP_POOL_DIAGNOSTICS: log the CPU sub-MIP pool's tasks and schedule.
inline constexpr bool default_pool_diagnostics = false;
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
// CUOPT_GPU_METHOD_BANDIT: no GPU sub-MIP worker thread. The regular GPU thread leaves the FP loop
// when it stops producing and then gives its time, one slice at a time, to the recombiners and to
// the RINS / RENS / mutation neighbourhoods of the tree, chosen by a bandit.
inline constexpr bool default_method_bandit = false;
// CUOPT_GPU_METHOD_BANDIT_FP_FAILURES: failed FP iterations in a row that end the FP loop, once
// the population has a feasible solution.
inline constexpr int default_fp_failures = 5;
// CUOPT_GPU_METHOD_BANDIT_ARM_SECONDS: seconds of one slice.
inline constexpr double default_arm_seconds = 5.;
// CUOPT_GPU_METHOD_BANDIT_DYNAMIC: each method's slice doubles (up to four times the setting above)
// after a slice that improves the incumbent, halves (down to half of it) after one that adds
// nothing to the population, and all go back to it after 60 s without an improvement.
inline constexpr bool default_dynamic_arm_seconds = false;

// Benchmark commits set this to false, so that their defaults hold whatever the environment
// says: the cluster's benchmark script exports CUOPT_GPU_SUBMIP, CUOPT_SUBMIP_POOL_DIAGNOSTICS,
// CUOPT_GPU_SUBMIP_CONFIG_FAMILY, CUOPT_CONFIG_ID and CUOPT_MAX_CONFIG.
inline constexpr bool default_read_environment = true;

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
inline bool method_bandit() { return flag("CUOPT_GPU_METHOD_BANDIT", default_method_bandit); }
inline int fp_failures()
{
  return (int)number("CUOPT_GPU_METHOD_BANDIT_FP_FAILURES", default_fp_failures);
}
inline double arm_seconds()
{
  return number("CUOPT_GPU_METHOD_BANDIT_ARM_SECONDS", default_arm_seconds);
}
inline bool dynamic_arm_seconds()
{
  return flag("CUOPT_GPU_METHOD_BANDIT_DYNAMIC", default_dynamic_arm_seconds);
}

}  // namespace cuopt::mathematical_optimization::mip::gpu_submip_experiment
