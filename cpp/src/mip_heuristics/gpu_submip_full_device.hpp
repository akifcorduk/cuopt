/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <branch_and_bound/gpu_submip_worker.hpp>

namespace cuopt::mathematical_optimization::mip {

// Keep the CUDA-facing handle definition out of host-only worker implementations.
int gpu_submip_device_id(const raft::handle_t* handle);

// Execute on the caller's thread using the selected device's ordinary CUDA context.
// Each call owns its stream, RAFT resources, and FP session. Callbacks run on this
// thread; concurrent callers must synchronize any shared publication state.
// CUDA and callback errors propagate after cleanup, and the prior device is restored.
template <typename i_t, typename f_t>
bool run_gpu_submip_job_full_device(const gpu_submip_job_t<i_t, f_t>& job,
                                    const mip_solver_settings_t<i_t, f_t>& settings,
                                    std::atomic<int>* halt,
                                    const gpu_submip_options_t& options,
                                    std::function<f_t()> current_user_cutoff,
                                    std::function<void(const std::vector<f_t>&, f_t)> publish,
                                    int device);

// Retain one stream/model across descents until the paired CPU sub-MIP ends.
// pair_halt must remain alive through this call; root cancellation is mirrored
// into it without ever resetting it. Each descent refreshes the live cutoff.
// options.time_limit is a single-job replay limit and is ignored in this mode;
// job.deadline and the two cancellation tokens govern the paired session.
template <typename i_t, typename f_t>
bool run_gpu_submip_job_full_device_continuous(
  const gpu_submip_job_t<i_t, f_t>& job,
  const mip_solver_settings_t<i_t, f_t>& settings,
  std::atomic<int>* root_halt,
  std::atomic<int>* pair_halt,
  const gpu_submip_options_t& options,
  std::function<f_t()> current_user_cutoff,
  std::function<void(const std::vector<f_t>&, f_t)> publish,
  int device);

}  // namespace cuopt::mathematical_optimization::mip
