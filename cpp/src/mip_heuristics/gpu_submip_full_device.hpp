/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <branch_and_bound/gpu_submip_worker.hpp>

namespace cuopt::mathematical_optimization::mip {

// Keep the CUDA-facing handle definition out of host-only worker implementations.
int gpu_submip_device_id(const raft::handle_t* handle);

// Exclusive leases keep streams and library handles alive across paired jobs. Keep
// the pool alive until all solver workers have stopped: cuBLAS handle destruction
// synchronizes the device and cannot overlap another worker's graph capture.
class gpu_submip_device_pool_t {
 public:
  gpu_submip_device_pool_t();
  ~gpu_submip_device_pool_t();
  gpu_submip_device_pool_t(const gpu_submip_device_pool_t&)            = delete;
  gpu_submip_device_pool_t& operator=(const gpu_submip_device_pool_t&) = delete;
  std::shared_ptr<const raft::handle_t> acquire(int device);

 private:
  struct impl_t;
  std::shared_ptr<impl_t> impl_;
};

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

struct gpu_submip_session_info_t {
  uint64_t descents{0};
  int projections{0};
  const char* exit_reason{"not_started"};
};

// Retain one stream/model across descents until the paired CPU sub-MIP ends.
// pair_halt must remain alive through this call; root cancellation is mirrored
// into it without ever resetting it. Each descent refreshes the live cutoff.
// options.time_limit is a single-job replay limit and is ignored in this mode;
// job.deadline and the two cancellation tokens govern the paired session.
// With bounded_effort, honor options.time_limit/max_descents/max_projections instead;
// pass root_halt as both tokens to run independently of any CPU producer's lifetime.
template <typename i_t, typename f_t>
bool run_gpu_submip_job_full_device_continuous(
  const gpu_submip_job_t<i_t, f_t>& job,
  const mip_solver_settings_t<i_t, f_t>& settings,
  std::atomic<int>* root_halt,
  std::atomic<int>* pair_halt,
  const gpu_submip_options_t& options,
  std::function<f_t()> current_user_cutoff,
  std::function<void(const std::vector<f_t>&, f_t)> publish,
  int device,
  gpu_submip_device_pool_t* device_pool         = nullptr,
  bool bounded_effort                           = false,
  gpu_submip_session_info_t* session_info       = nullptr,
  gpu_submip_seed_cache_t<i_t, f_t>* seed_cache = nullptr);

}  // namespace cuopt::mathematical_optimization::mip
