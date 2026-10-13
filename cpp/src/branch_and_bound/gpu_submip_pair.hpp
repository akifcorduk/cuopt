/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <thread>
#include <utility>

namespace cuopt::mathematical_optimization::mip {

// Count a job from admission through completed device cleanup. The release callback is
// host-only and nonthrowing; an unstarted thread or exceptional descent still releases once.
class gpu_submip_demand_t {
 public:
  gpu_submip_demand_t(const std::function<bool()>& acquire, std::function<void()> release)
    : release_(std::move(release)), active_(acquire ? acquire() : true)
  {
  }
  gpu_submip_demand_t(const gpu_submip_demand_t&)            = delete;
  gpu_submip_demand_t& operator=(const gpu_submip_demand_t&) = delete;
  ~gpu_submip_demand_t() { release(); }
  explicit operator bool() const { return active_.load(); }
  void release() noexcept
  {
    if (active_.exchange(false) && release_) { release_(); }
  }

 private:
  std::function<void()> release_;
  std::atomic<bool> active_;
};

// GPU work belongs to the entire existing CPU sub-MIP invocation, including its DFS/child
// solve. Once that scope finishes, request cooperative cancellation before joining cleanup.
class gpu_submip_pair_t {
 public:
  gpu_submip_pair_t(std::function<void(std::atomic<int>*)> run,
                    std::function<void(std::exception_ptr)> report_error,
                    std::function<void(bool, bool)> join_observer = {},
                    std::function<void()> stop_observer           = {})
    : halt_(std::make_shared<std::atomic<int>>(0)),
      finished_(std::make_shared<std::atomic<bool>>(false)),
      join_observer_(std::move(join_observer)),
      stop_observer_(std::move(stop_observer)),
      worker_([run          = std::move(run),
               report_error = std::move(report_error),
               halt         = halt_,
               finished     = finished_]() {
        try {
          run(halt.get());
        } catch (...) {
          report_error(std::current_exception());
        }
        finished->store(true, std::memory_order_release);
      })
  {
  }
  gpu_submip_pair_t(const gpu_submip_pair_t&)            = delete;
  gpu_submip_pair_t& operator=(const gpu_submip_pair_t&) = delete;
  ~gpu_submip_pair_t() { join(); }

  void request_stop() noexcept
  {
    halt_->store(1, std::memory_order_release);
    // Global cancellation may already have set halt_. CPU scope completion still
    // needs its own one-time notification and cleanup timestamp.
    if (cpu_stop_requested_.exchange(true, std::memory_order_acq_rel)) { return; }
    try {
      if (stop_observer_) { stop_observer_(); }
    } catch (...) {
      // Diagnostics cannot interfere with the stop request or cleanup.
    }
  }

  void join()
  {
    request_stop();
    if (!worker_.joinable()) { return; }
    const bool already_finished = finished_->load(std::memory_order_acquire);
    notify_join(true, already_finished);
    worker_.join();
    notify_join(false, already_finished);
  }

 private:
  void notify_join(bool starting, bool already_finished) noexcept
  {
    try {
      if (join_observer_) { join_observer_(starting, already_finished); }
    } catch (...) {
      // Diagnostics cannot bypass the lifetime join or change CPU search.
    }
  }
  std::shared_ptr<std::atomic<int>> halt_;
  std::atomic<bool> cpu_stop_requested_{false};
  std::shared_ptr<std::atomic<bool>> finished_;
  std::function<void(bool, bool)> join_observer_;
  std::function<void()> stop_observer_;
  std::thread worker_;
};

}  // namespace cuopt::mathematical_optimization::mip
