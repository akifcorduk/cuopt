/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/logger.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace cuopt {

// The regular GPU worker owns activation; the subMIP worker observes ready only after all
// auxiliary streams have moved. Both workers must finish before this object is destroyed.
class mip_green_context_t {
 public:
  explicit mip_green_context_t(const raft::handle_t* handle)
    : regular_handle_(const_cast<raft::handle_t*>(handle)), original_stream_(handle->get_stream())
  {
  }

  mip_green_context_t(const mip_green_context_t&)            = delete;
  mip_green_context_t& operator=(const mip_green_context_t&) = delete;

  ~mip_green_context_t()
  {
    try {
      restore_regular();
    } catch (const std::exception& e) {
      CUOPT_LOG_WARN("GPU subMIP stream restoration failed: %s", e.what());
    } catch (...) {
      CUOPT_LOG_WARN("GPU subMIP stream restoration failed with an unknown exception");
    }
    submip_handle_.reset();
#if CUDART_VERSION >= 12040
    destroy_resources();
#endif
  }

  void request() { requested_.store(true, std::memory_order_release); }

  bool activate_regular()
  {
    if (!requested_.load(std::memory_order_acquire) || attempted_) { return false; }
    attempted_ = true;
#if CUDART_VERSION >= 12040
    try {
      load_functions();
      regular_handle_->sync_stream();
      check(get_current_(&original_context_), "cuCtxGetCurrent");
      original_context_saved_ = true;
      CUdevResource all{}, submip{}, regular{};
      check(get_resource_(regular_handle_->get_device(), &all, CU_DEV_RESOURCE_TYPE_SM),
            "cuDeviceGetDevResource");
      unsigned int groups = 1;
      unsigned int target = std::max(1u, static_cast<unsigned int>(all.sm.smCount * 0.4));
      check(split_(&submip,
                   &groups,
                   &all,
                   &regular,
                   CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING,
                   target),
            "cuDevSmResourceSplitByCount");
      if (groups != 1 || submip.sm.smCount == 0 || regular.sm.smCount == 0) {
        throw std::runtime_error("The device cannot provide two nonempty SM partitions");
      }
      create_partition(submip, submip_context_, submip_stream_, submip_cuda_context_);
      create_partition(regular, regular_context_, regular_stream_, regular_cuda_context_);
      check(set_current_(submip_cuda_context_), "cuCtxSetCurrent");
      submip_handle_ = std::make_unique<raft::handle_t>(rmm::cuda_stream_view{submip_stream_});
      raft::resource::get_device_properties(*submip_handle_).multiProcessorCount =
        submip.sm.smCount;
      check(set_current_(regular_cuda_context_), "cuCtxSetCurrent");
      original_sm_count_ = regular_handle_->get_device_properties().multiProcessorCount;
      raft::resource::get_device_properties(*regular_handle_).multiProcessorCount =
        regular.sm.smCount;
      active_ = true;
      rebind_handle(regular_stream_);
      CUOPT_LOG_INFO(
        "GPU subMIP partition: %u/%u SMs (%.1f%%), regular heuristics: %u/%u SMs "
        "(%.1f%%)",
        submip.sm.smCount,
        all.sm.smCount,
        100.0 * submip.sm.smCount / all.sm.smCount,
        regular.sm.smCount,
        all.sm.smCount,
        100.0 * regular.sm.smCount / all.sm.smCount);
      return true;
    } catch (const std::exception& e) {
      std::exception_ptr cleanup_error;
      if (active_) {
        attempt_cleanup(cleanup_error, [&]() { restore_regular(); });
      }
      if (set_current_ && original_context_saved_) {
        attempt_cleanup(cleanup_error,
                        [&]() { check(set_current_(original_context_), "cuCtxSetCurrent"); });
      }
      CUOPT_LOG_WARN("GPU subMIP disabled: green-context activation failed: %s", e.what());
      if (cleanup_error) { std::rethrow_exception(cleanup_error); }
      return false;
    }
#else
    CUOPT_LOG_WARN("GPU subMIP disabled: this CUDA toolkit has no green-context support");
    return false;
#endif
  }

  void publish_ready() { ready_.store(active_, std::memory_order_release); }
  bool active() const { return active_; }

  const raft::handle_t* submip_handle()
  {
    if (!ready_.load(std::memory_order_acquire)) { return nullptr; }
#if CUDART_VERSION >= 12040
    if (!submip_bound_) {
      check(get_current_(&submip_original_context_), "cuCtxGetCurrent");
      check(set_current_(submip_cuda_context_), "cuCtxSetCurrent");
      submip_bound_ = true;
    }
#endif
    return submip_handle_.get();
  }

  void release_submip()
  {
#if CUDART_VERSION >= 12040
    if (submip_bound_) {
      std::exception_ptr cleanup_error;
      attempt_cleanup(cleanup_error, [&]() { submip_handle_->sync_stream(); });
      attempt_cleanup(cleanup_error,
                      [&]() { check(set_current_(submip_original_context_), "cuCtxSetCurrent"); });
      submip_bound_ = false;
      if (cleanup_error) { std::rethrow_exception(cleanup_error); }
    }
#endif
  }

  bool restore_regular()
  {
    if (!active_) { return false; }
    ready_.store(false, std::memory_order_release);
    std::exception_ptr cleanup_error;
    attempt_cleanup(cleanup_error, [&]() { regular_handle_->sync_stream(); });
    attempt_cleanup(cleanup_error, [&]() { rebind_handle(original_stream_.get()); });
    attempt_cleanup(cleanup_error, [&]() {
      raft::resource::get_device_properties(*regular_handle_).multiProcessorCount =
        original_sm_count_;
    });
#if CUDART_VERSION >= 12040
    attempt_cleanup(cleanup_error,
                    [&]() { check(set_current_(original_context_), "cuCtxSetCurrent"); });
#endif
    active_ = false;
    if (cleanup_error) { std::rethrow_exception(cleanup_error); }
    return true;
  }

 private:
  template <typename cleanup_t>
  static void attempt_cleanup(std::exception_ptr& first_error, cleanup_t&& cleanup)
  {
    try {
      cleanup();
    } catch (...) {
      if (!first_error) { first_error = std::current_exception(); }
    }
  }

  void rebind_handle(cudaStream_t stream)
  {
    std::exception_ptr cleanup_error;
    attempt_cleanup(cleanup_error, [&]() {
      raft::resource::set_cuda_stream(*regular_handle_, rmm::cuda_stream_view{stream});
    });
    attempt_cleanup(cleanup_error, [&]() {
      regular_handle_->add_resource_factory(
        std::make_shared<raft::resource::thrust_policy_resource_factory>(
          rmm::cuda_stream_view{stream}));
    });
    attempt_cleanup(cleanup_error, [&]() {
      RAFT_CUBLAS_TRY(cublasSetStream(regular_handle_->get_cublas_handle(), stream));
    });
    attempt_cleanup(cleanup_error, [&]() {
      RAFT_CUSPARSE_TRY(cusparseSetStream(regular_handle_->get_cusparse_handle(), stream));
    });
    if (cleanup_error) { std::rethrow_exception(cleanup_error); }
  }

#if CUDART_VERSION >= 12040
  template <typename function_t>
  static function_t load_function(const char* name)
  {
    void* pointer = nullptr;
    cudaDriverEntryPointQueryResult status;
    RAFT_CUDA_TRY(
      cudaGetDriverEntryPointByVersion(name, &pointer, CUDA_VERSION, cudaEnableDefault, &status));
    if (status != cudaDriverEntryPointSuccess || pointer == nullptr) {
      throw std::runtime_error(std::string("Missing CUDA driver function ") + name);
    }
    return reinterpret_cast<function_t>(pointer);
  }

  static void check(CUresult result, const char* operation)
  {
    if (result != CUDA_SUCCESS) {
      throw std::runtime_error(std::string(operation) + " failed with CUDA driver status " +
                               std::to_string(static_cast<int>(result)));
    }
  }

  void load_functions()
  {
    get_current_   = load_function<decltype(get_current_)>("cuCtxGetCurrent");
    set_current_   = load_function<decltype(set_current_)>("cuCtxSetCurrent");
    get_resource_  = load_function<decltype(get_resource_)>("cuDeviceGetDevResource");
    split_         = load_function<decltype(split_)>("cuDevSmResourceSplitByCount");
    generate_desc_ = load_function<decltype(generate_desc_)>("cuDevResourceGenerateDesc");
    create_        = load_function<decltype(create_)>("cuGreenCtxCreate");
    destroy_       = load_function<decltype(destroy_)>("cuGreenCtxDestroy");
    create_stream_ = load_function<decltype(create_stream_)>("cuGreenCtxStreamCreate");
    convert_       = load_function<decltype(convert_)>("cuCtxFromGreenCtx");
  }

  void create_partition(CUdevResource& resource,
                        CUgreenCtx& context,
                        CUstream& stream,
                        CUcontext& cuda_context)
  {
    CUdevResourceDesc descriptor;
    check(generate_desc_(&descriptor, &resource, 1), "cuDevResourceGenerateDesc");
    check(create_(&context, descriptor, regular_handle_->get_device(), CU_GREEN_CTX_DEFAULT_STREAM),
          "cuGreenCtxCreate");
    check(create_stream_(&stream, context, CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate");
    check(convert_(&cuda_context, context), "cuCtxFromGreenCtx");
  }

  void destroy_resources() noexcept
  {
    if (regular_stream_) {
      cudaStreamSynchronize(regular_stream_);
      cudaStreamDestroy(regular_stream_);
    }
    if (submip_stream_) {
      cudaStreamSynchronize(submip_stream_);
      cudaStreamDestroy(submip_stream_);
    }
    if (regular_context_ && destroy_) { destroy_(regular_context_); }
    if (submip_context_ && destroy_) { destroy_(submip_context_); }
  }

  decltype(&cuCtxGetCurrent) get_current_{nullptr};
  decltype(&cuCtxSetCurrent) set_current_{nullptr};
  decltype(&cuDeviceGetDevResource) get_resource_{nullptr};
  decltype(&cuDevSmResourceSplitByCount) split_{nullptr};
  decltype(&cuDevResourceGenerateDesc) generate_desc_{nullptr};
  decltype(&cuGreenCtxCreate) create_{nullptr};
  decltype(&cuGreenCtxDestroy) destroy_{nullptr};
  decltype(&cuGreenCtxStreamCreate) create_stream_{nullptr};
  decltype(&cuCtxFromGreenCtx) convert_{nullptr};
  CUgreenCtx regular_context_{nullptr};
  CUgreenCtx submip_context_{nullptr};
  CUcontext regular_cuda_context_{nullptr};
  CUcontext submip_cuda_context_{nullptr};
  CUcontext original_context_{nullptr};
  CUcontext submip_original_context_{nullptr};
  CUstream regular_stream_{nullptr};
  CUstream submip_stream_{nullptr};
  bool submip_bound_{false};
  bool original_context_saved_{false};
#endif
  raft::handle_t* regular_handle_;
  rmm::cuda_stream_view original_stream_;
  std::unique_ptr<raft::handle_t> submip_handle_;
  std::atomic<bool> requested_{false};
  std::atomic<bool> ready_{false};
  bool attempted_{false};
  bool active_{false};
  int original_sm_count_{0};
};

}  // namespace cuopt
