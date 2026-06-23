/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

// Device hardware descriptor for the GPU-generic work model. All fields come from cudaDeviceProp /
// cudaDeviceGetAttribute, which are constant for a given device, so reading them never injects
// run-to-run nondeterminism. The work model expresses per-iteration time as problem-count features
// divided by these device rates (memory bandwidth, occupancy capacity, clock), so one set of fitted
// coefficients transfers across GPUs once the device descriptor is plugged in.

#include <cuda_runtime.h>
#include <cstdio>

namespace cuopt::linear_programming::detail::calib {

struct gpu_features_t {
  double mem_bandwidth_gb_s{0.0};  // peak global memory bandwidth (GB/s)
  double sm_count{0.0};            // multiprocessor count
  double sm_clock_ghz{0.0};        // SM core clock (GHz)
  double l2_cache_mb{0.0};         // L2 cache size (MiB)
  double warp_capacity{0.0};       // sm_count * maxThreadsPerSM / 32 (resident-warp ceiling)
  char name[256]{0};
};

// Query the descriptor for `device`. Uses cudaDeviceGetAttribute for the clock/bus-width values
// (more robust than the deprecated cudaDeviceProp fields) and falls back to the prop struct.
inline gpu_features_t query_gpu_features(int device = 0)
{
  gpu_features_t g;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, device) != cudaSuccess) { return g; }

  int mem_clock_khz  = 0;  // kHz
  int bus_width_bits = 0;  // bits
  int sm_clock_khz   = 0;  // kHz
  int l2_bytes       = 0;
  int max_threads_sm = 0;
  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);
  cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device);
  cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, device);
  cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device);
  cudaDeviceGetAttribute(&max_threads_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device);

  // Fall back to prop fields if an attribute came back as 0.
  if (mem_clock_khz == 0) { mem_clock_khz = p.memoryClockRate; }
  if (bus_width_bits == 0) { bus_width_bits = p.memoryBusWidth; }
  if (sm_clock_khz == 0) { sm_clock_khz = p.clockRate; }
  if (l2_bytes == 0) { l2_bytes = p.l2CacheSize; }
  if (max_threads_sm == 0) { max_threads_sm = p.maxThreadsPerMultiProcessor; }

  // DDR memory: effective transfers are 2x the clock; bytes = bus_width_bits / 8.
  g.mem_bandwidth_gb_s = 2.0 * ((double)mem_clock_khz * 1e3) * ((double)bus_width_bits / 8.0) / 1e9;
  g.sm_count           = (double)p.multiProcessorCount;
  g.sm_clock_ghz       = (double)sm_clock_khz / 1e6;
  g.l2_cache_mb        = (double)l2_bytes / (1024.0 * 1024.0);
  g.warp_capacity      = (double)p.multiProcessorCount * (double)max_threads_sm / 32.0;
  std::snprintf(g.name, sizeof(g.name), "%s", p.name);
  return g;
}

inline void print_gpu_features(const gpu_features_t& g)
{
  std::printf(
    "GPU '%s': mem_bw=%.1f GB/s  sm_count=%.0f  sm_clock=%.3f GHz  l2=%.1f MiB  warp_cap=%.0f\n",
    g.name,
    g.mem_bandwidth_gb_s,
    g.sm_count,
    g.sm_clock_ghz,
    g.l2_cache_mb,
    g.warp_capacity);
}

}  // namespace cuopt::linear_programming::detail::calib
