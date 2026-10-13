/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpu_submip_full_device.hpp"

#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>
#include <utilities/pcgenerator.hpp>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace cuopt::mathematical_optimization::mip {
namespace {

class stream_drain_t {
 public:
  explicit stream_drain_t(cudaStream_t stream) : stream_(stream) {}

  ~stream_drain_t()
  {
    if (!drained_) { RAFT_CUDA_TRY_NO_THROW(cudaStreamSynchronize(stream_)); }
  }

  void synchronize()
  {
    const auto status = cudaStreamSynchronize(stream_);
    drained_          = true;
    RAFT_CUDA_TRY(status);
  }

 private:
  cudaStream_t stream_;
  bool drained_{false};
};

// CPU completion writes pair_halt directly, so PDLP/FJ see it immediately.
// Only root cancellation needs a watcher; no GPU resources are touched here.
class paired_cancel_t {
 public:
  paired_cancel_t(std::atomic<int>* root, std::atomic<int>* pair) : root_(root), pair_(pair)
  {
    if (!pair_) { throw std::invalid_argument("Continuous GPU FP requires a pair halt token"); }
    if (root_ && !cancelled()) {
      watcher_ = std::thread([this] { watch_root(); });
    }
  }

  ~paired_cancel_t()
  {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    if (watcher_.joinable()) { watcher_.join(); }
  }

  bool cancelled() const
  {
    if (root_ && root_->load(std::memory_order_acquire)) {
      pair_->store(1, std::memory_order_release);
    }
    return pair_->load(std::memory_order_acquire) != 0;
  }

  const char* end_reason() const
  {
    if (root_ && root_->load(std::memory_order_acquire)) { return "root_cancel"; }
    return pair_->load(std::memory_order_acquire) ? "pair_complete" : "deadline";
  }

  void wait_until(double deadline)
  {
    std::unique_lock lock(mutex_);
    while (!cancelled() && tic() < deadline) {
      condition_.wait_for(lock, std::chrono::milliseconds(5), [this] { return cancelled(); });
    }
  }

 private:
  void watch_root()
  {
    std::unique_lock lock(mutex_);
    while (!stopping_ && !cancelled()) {
      condition_.wait_for(lock, std::chrono::milliseconds(5), [this] { return stopping_; });
    }
    condition_.notify_all();
  }

  std::atomic<int>* root_;
  std::atomic<int>* pair_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool stopping_{false};
  std::thread watcher_;
};

}  // namespace

struct gpu_submip_device_pool_t::impl_t {
  struct entry_t {
    explicit entry_t(int device_) : device(device_) {}

    int device;
    bool busy{true};
    rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
    raft::handle_t handle{stream};
  };

  ~impl_t()
  {
    for (auto& entry : entries) {
      raft::device_setter device_guard(entry->device);
      entry.reset();
    }
  }

  std::mutex mutex;
  std::vector<std::unique_ptr<entry_t>> entries;
};

gpu_submip_device_pool_t::gpu_submip_device_pool_t() : impl_(std::make_shared<impl_t>()) {}
gpu_submip_device_pool_t::~gpu_submip_device_pool_t() = default;

std::shared_ptr<const raft::handle_t> gpu_submip_device_pool_t::acquire(int device)
{
  raft::device_setter device_guard(device);
  std::lock_guard lock(impl_->mutex);
  impl_t::entry_t* selected = nullptr;
  for (auto& entry : impl_->entries) {
    if (!entry->busy && entry->device == device) {
      selected = entry.get();
      break;
    }
  }
  if (!selected) {
    impl_->entries.push_back(std::make_unique<impl_t::entry_t>(device));
    selected = impl_->entries.back().get();
  }
  selected->busy = true;
  return {&selected->handle, [pool = impl_, selected](const raft::handle_t*) {
            std::lock_guard lock(pool->mutex);
            selected->busy = false;
          }};
}

int gpu_submip_device_id(const raft::handle_t* handle) { return handle->get_device(); }

template <typename i_t, typename f_t>
bool run_gpu_submip_job_full_device(const gpu_submip_job_t<i_t, f_t>& job,
                                    const mip_solver_settings_t<i_t, f_t>& settings,
                                    std::atomic<int>* halt,
                                    const gpu_submip_options_t& options,
                                    std::function<f_t()> current_user_cutoff,
                                    std::function<void(const std::vector<f_t>&, f_t)> publish,
                                    int device)
{
  raft::device_setter device_guard(device);
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  stream_drain_t drain(stream.value());
  bool found = false;
  {
    // Construct rather than copy: cuBLAS/cuSPARSE handles and their stream bindings
    // belong exclusively to this call. The job creates and destroys its own session.
    raft::handle_t handle(stream);
    CUOPT_LOG_INFO("GpuSubmipFullDevice job=%llu device=%d stream=%p sm_count=%d",
                   (unsigned long long)job.job_id,
                   device,
                   reinterpret_cast<void*>(stream.value()),
                   handle.get_device_properties().multiProcessorCount);
    found = run_gpu_submip_job(
      job, settings, &handle, halt, options, std::move(current_user_cutoff), std::move(publish));
  }
  // Drain after session and handle destruction, while the stream and device guard
  // are still alive. The guard also drains when a callback or GPU operation throws.
  drain.synchronize();
  return found;
}

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
  gpu_submip_device_pool_t* device_pool)
{
  paired_cancel_t cancellation(root_halt, pair_halt);
  raft::device_setter device_guard(device);
  gpu_submip_device_pool_t local_pool;
  auto lease         = (device_pool ? device_pool : &local_pool)->acquire(device);
  const auto& handle = *lease;
  stream_drain_t drain(handle.get_stream().get());
  bool published_any   = false;
  uint64_t descent     = 0;
  const double started = tic();
  {
    std::unique_ptr<gpu_submip_fp_t<i_t, f_t>> session;
    const auto base_seed = get_base_seed(settings.seed);
    pcgenerator_t rng(derive_seed(base_seed, rng_id_t::gpu_submip_worker, job.job_id),
                      derive_stream(base_seed, rng_id_t::gpu_submip_worker, job.job_id));
    auto continuous_options       = options;
    continuous_options.time_limit = std::numeric_limits<double>::infinity();
    std::vector<f_t> restart_seed;
    const char* restart_reason = "initial_seed";
    i_t perturbed              = 0;
    CUOPT_LOG_INFO("GpuSubmipFullDevice job=%llu device=%d stream=%p sm_count=%d continuous=1",
                   (unsigned long long)job.job_id,
                   device,
                   reinterpret_cast<void*>(handle.get_stream().get()),
                   handle.get_device_properties().multiProcessorCount);
    while (!cancellation.cancelled() && tic() < job.deadline) {
      ++descent;
      const bool restarting = !restart_seed.empty();
      CUOPT_LOG_INFO(
        "GpuSubmipPairDescent job=%llu descent=%llu phase=start restart_reason=%s "
        "seed=%s perturbed=%d",
        (unsigned long long)job.job_id,
        (unsigned long long)descent,
        restart_reason,
        restarting ? "perturbed_projection" : "cpu_initial",
        int(perturbed));
      gpu_submip_fp_run_info_t info;
      const double descent_start = tic();
      bool published             = false;
      const auto guarded_publish = [&](const std::vector<f_t>& candidate, f_t objective) {
        if (!cancellation.cancelled() && tic() < job.deadline && publish) {
          publish(candidate, objective);
          published = true;
        }
      };
      run_gpu_submip_job<i_t, f_t>(job,
                                   settings,
                                   &handle,
                                   pair_halt,
                                   continuous_options,
                                   current_user_cutoff,
                                   guarded_publish,
                                   &session,
                                   restarting ? &restart_seed : nullptr,
                                   "perturbed_projection",
                                   &info);
      published_any = published_any || published;
      CUOPT_LOG_INFO(
        "GpuSubmipPairDescent job=%llu descent=%llu phase=end seconds=%.6f "
        "exit=%s projections=%d published=%d",
        (unsigned long long)job.job_id,
        (unsigned long long)descent,
        toc(descent_start),
        info.exit_reason,
        info.projections,
        int(published));
      if (cancellation.cancelled() || tic() >= job.deadline) { break; }
      if (!info.started && std::strcmp(info.exit_reason, "no_incumbent") == 0) {
        CUOPT_LOG_INFO("GpuSubmipPairIdle job=%llu descent=%llu reason=await_incumbent",
                       (unsigned long long)job.job_id,
                       (unsigned long long)descent);
        // An incumbent may arrive from the still-running CPU solve. Wait without
        // rebuilding the session, flooding logs, or replacing the initial seed.
        while (!cancellation.cancelled() && tic() < job.deadline) {
          cancellation.wait_until(std::min(job.deadline, tic() + 0.05));
          if (current_user_cutoff && std::isfinite(current_user_cutoff())) { break; }
        }
        continue;
      }
      restart_reason          = info.exit_reason;
      const char* idle_reason = nullptr;
      if (!info.started) {
        idle_reason = "descent_not_started";
      } else if (!session->get_restart_seed(restart_seed)) {
        idle_reason = "no_finite_restart_seed";
      } else {
        perturbed = perturb_gpu_submip_restart_seed(job.lp, job.var_types, restart_seed, rng);
        if (perturbed == 0) { idle_reason = "no_perturbable_integer"; }
      }
      if (idle_reason) {
        CUOPT_LOG_INFO("GpuSubmipPairIdle job=%llu descent=%llu reason=%s",
                       (unsigned long long)job.job_id,
                       (unsigned long long)descent,
                       idle_reason);
        // Keep ownership until CPU completion even when no useful restart exists.
        cancellation.wait_until(job.deadline);
        break;
      }
      if (info.projections == 0) {
        CUOPT_LOG_INFO(
          "GpuSubmipPairBackoff job=%llu descent=%llu reason=no_projection seconds=0.05",
          (unsigned long long)job.job_id,
          (unsigned long long)descent);
        cancellation.wait_until(std::min(job.deadline, tic() + 0.05));
      }
    }
  }
  drain.synchronize();
  CUOPT_LOG_INFO(
    "GpuSubmipPairSessionEnd job=%llu descents=%llu seconds=%.6f reason=%s published=%d",
    (unsigned long long)job.job_id,
    (unsigned long long)descent,
    toc(started),
    cancellation.end_reason(),
    int(published_any));
  return published_any;
}

template bool run_gpu_submip_job_full_device<int, double>(
  const gpu_submip_job_t<int, double>&,
  const mip_solver_settings_t<int, double>&,
  std::atomic<int>*,
  const gpu_submip_options_t&,
  std::function<double()>,
  std::function<void(const std::vector<double>&, double)>,
  int);

template bool run_gpu_submip_job_full_device_continuous<int, double>(
  const gpu_submip_job_t<int, double>&,
  const mip_solver_settings_t<int, double>&,
  std::atomic<int>*,
  std::atomic<int>*,
  const gpu_submip_options_t&,
  std::function<double()>,
  std::function<void(const std::vector<double>&, double)>,
  int,
  gpu_submip_device_pool_t*);

}  // namespace cuopt::mathematical_optimization::mip
