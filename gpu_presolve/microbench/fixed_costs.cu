/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <rmm/aligned.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/version_config.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform_scan.h>
#include <thrust/version.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

#define CUDA_CHECK(call)                        \
  do {                                          \
    cudaError_t cuda_status = (call);           \
    if (cuda_status != cudaSuccess) {           \
      fprintf(stderr,                           \
              "%s:%d: %s failed: %s\n",         \
              __FILE__,                         \
              __LINE__,                         \
              #call,                            \
              cudaGetErrorString(cuda_status)); \
      abort();                                  \
    }                                           \
  } while (0)

#define REQUIRE(condition)                                                          \
  do {                                                                              \
    if (!(condition)) {                                                             \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
      abort();                                                                      \
    }                                                                               \
  } while (0)

using clock_type = std::chrono::steady_clock;

double elapsed_us(clock_type::time_point begin, clock_type::time_point end)
{
  return std::chrono::duration<double, std::micro>(end - begin).count();
}

struct stats_t {
  int reps;
  double median_us;
  double p10_us;
  double p90_us;
};

double quantile(const std::vector<double>& sorted, double q)
{
  double pos   = q * (sorted.size() - 1);
  size_t below = pos;
  size_t above = std::min(below + 1, sorted.size() - 1);
  return sorted[below] + (pos - below) * (sorted[above] - sorted[below]);
}

stats_t summarize(std::vector<double>& samples)
{
  std::sort(samples.begin(), samples.end());
  return {
    (int)samples.size(), quantile(samples, 0.5), quantile(samples, 0.1), quantile(samples, 0.9)};
}

constexpr int max_phases = 5;
constexpr size_t kb      = 1024;
constexpr size_t mb      = 1024 * kb;
constexpr size_t gb      = 1024 * mb;

// A body calls lap() at the end of each timed phase; code after the last lap() is not timed.
struct lap_t {
  clock_type::time_point last;
  double us[max_phases];
  int count;

  void start()
  {
    count = 0;
    last  = clock_type::now();
  }

  void lap()
  {
    auto now    = clock_type::now();
    us[count++] = elapsed_us(last, now);
    last        = now;
  }
};

struct measurement_t {
  stats_t total;
  std::vector<stats_t> phases;
};

struct harness_t {
  int reps;
  int min_reps;
  int warmup;
  double budget_us;
  double warmup_budget_us;

  template <typename setup_t, typename body_t>
  measurement_t measure(setup_t&& setup, body_t&& body) const
  {
    lap_t lap;
    double iteration_us = 0;
    auto warmup_begin   = clock_type::now();
    for (int i = 0; i < warmup; ++i) {
      auto begin = clock_type::now();
      setup();
      lap.start();
      body(lap);
      auto end     = clock_type::now();
      iteration_us = elapsed_us(begin, end);
      if (i > 0 && elapsed_us(warmup_begin, end) > warmup_budget_us) { break; }
    }

    int n = reps;
    if (iteration_us * n > budget_us) { n = std::max(min_reps, (int)(budget_us / iteration_us)); }

    int phases = lap.count;
    std::vector<std::vector<double>> samples(phases + 1);
    for (auto& s : samples) {
      s.reserve(n);
    }
    for (int i = 0; i < n; ++i) {
      setup();
      lap.start();
      body(lap);
      double total = 0;
      for (int p = 0; p < phases; ++p) {
        samples[p].push_back(lap.us[p]);
        total += lap.us[p];
      }
      samples[phases].push_back(total);
    }

    measurement_t m;
    m.total = summarize(samples[phases]);
    for (int p = 0; p < phases; ++p) {
      m.phases.push_back(summarize(samples[p]));
    }
    return m;
  }

  template <typename body_t>
  measurement_t measure(body_t&& body) const
  {
    return measure([] {}, body);
  }
};

struct result_t {
  std::string group;
  std::string name;
  std::string variant;
  std::string size_label;
  double elements = 0;
  double bytes    = 0;
  bool byte_rate  = false;
  stats_t stats{};
  std::vector<std::pair<std::string, stats_t>> phases;
  std::vector<std::pair<std::string, double>> extra;
};

struct context_t {
  cudaStream_t stream;
  harness_t harness;
  bool quick;
  rmm::mr::pool_memory_resource* pool;
  std::vector<result_t> results;

  void add(const result_t& r)
  {
    fprintf(stderr,
            "[%s] %s %s %s: median %.3f us (%d reps)\n",
            r.group.c_str(),
            r.name.c_str(),
            r.variant.c_str(),
            r.size_label.c_str(),
            r.stats.median_us,
            r.stats.reps);
    results.push_back(r);
  }
};

result_t make_result(const char* group,
                     const char* name,
                     const char* variant,
                     const std::string& size_label,
                     const stats_t& stats)
{
  result_t r;
  r.group      = group;
  r.name       = name;
  r.variant    = variant;
  r.size_label = size_label;
  r.stats      = stats;
  return r;
}

std::string count_label(size_t n)
{
  int exponent = 0;
  while (n % 10 == 0) {
    n /= 10;
    ++exponent;
  }
  return std::to_string(n) + "e" + std::to_string(exponent);
}

std::string bytes_label(size_t bytes)
{
  if (bytes >= gb) { return std::to_string(bytes / gb) + " GB"; }
  if (bytes >= mb) { return std::to_string(bytes / mb) + " MB"; }
  return std::to_string(bytes / kb) + " KB";
}

template <typename T>
struct device_array {
  T* ptr;
  size_t size;

  explicit device_array(size_t n) : size(n) { CUDA_CHECK(cudaMalloc((void**)&ptr, n * sizeof(T))); }
  ~device_array() { CUDA_CHECK(cudaFree(ptr)); }
  device_array(const device_array&)            = delete;
  device_array& operator=(const device_array&) = delete;
};

template <typename T>
struct pinned_array {
  T* ptr;
  size_t size;

  explicit pinned_array(size_t n) : size(n)
  {
    CUDA_CHECK(cudaHostAlloc((void**)&ptr, n * sizeof(T), cudaHostAllocDefault));
  }
  ~pinned_array() { CUDA_CHECK(cudaFreeHost(ptr)); }
  pinned_array(const pinned_array&)            = delete;
  pinned_array& operator=(const pinned_array&) = delete;
};

__host__ __device__ inline uint64_t mix64(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

__host__ __device__ inline uint64_t hash2(uint64_t seed, uint64_t i)
{
  return mix64(mix64(seed) ^ i);
}

__host__ __device__ inline double unit_double(uint64_t bits) { return (bits >> 11) * 0x1.0p-53; }

std::vector<int> random_permutation(int n, uint64_t seed)
{
  std::vector<int> perm(n);
  for (int i = 0; i < n; ++i) {
    perm[i] = i;
  }
  for (int i = n - 1; i > 0; --i) {
    int j = hash2(seed, i) % (i + 1);
    std::swap(perm[i], perm[j]);
  }
  return perm;
}

__global__ void empty_kernel() {}

void bench_launch(context_t& ctx)
{
  cudaStream_t stream = ctx.stream;

  auto idle_sync = ctx.harness.measure([&](lap_t& lap) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
  });
  ctx.add(make_result("launch", "sync_idle_stream", "", "", idle_sync.total));

  auto launch_sync = ctx.harness.measure([&](lap_t& lap) {
    empty_kernel<<<1, 1, 0, stream>>>();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
    CUDA_CHECK(cudaGetLastError());
  });
  ctx.add(make_result("launch", "empty_kernel_launch_sync", "", "", launch_sync.total));

  auto launch_only = ctx.harness.measure([&](lap_t& lap) {
    empty_kernel<<<1, 1, 0, stream>>>();
    lap.lap();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  });
  ctx.add(make_result("launch", "empty_kernel_launch_no_sync", "", "", launch_only.total));

  auto chained = ctx.harness.measure([&](lap_t& lap) {
    for (int i = 0; i < 10; ++i) {
      empty_kernel<<<1, 1, 0, stream>>>();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
    CUDA_CHECK(cudaGetLastError());
  });
  ctx.add(make_result("launch", "empty_kernel_10_launches_one_sync", "", "", chained.total));
}

__global__ void write_value_kernel(uint64_t* out, uint64_t value) { *out = value; }

void bench_readback(context_t& ctx)
{
  cudaStream_t stream = ctx.stream;
  device_array<uint64_t> d_value(1);
  pinned_array<uint64_t> pinned(1);
  std::vector<uint64_t> pageable(1);
  uint64_t counter = 0;

  auto run = [&](uint64_t* host, const char* variant) {
    auto m     = ctx.harness.measure([&](lap_t& lap) {
      ++counter;
      write_value_kernel<<<1, 1, 0, stream>>>(d_value.ptr, counter);
      CUDA_CHECK(
        cudaMemcpyAsync(host, d_value.ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      lap.lap();
      CUDA_CHECK(cudaGetLastError());
      REQUIRE(*host == counter);
    });
    result_t r = make_result("readback", "kernel_write_8_bytes_copy_to_host", variant, "", m.total);
    r.bytes    = sizeof(uint64_t);
    ctx.add(r);
  };
  run(pageable.data(), "pageable");
  run(pinned.ptr, "pinned");
}

struct hash_fn {
  uint64_t seed;
  __host__ __device__ uint64_t operator()(uint64_t i) const { return hash2(seed, i); }
};

struct unit_double_fn {
  uint64_t seed;
  __host__ __device__ double operator()(uint64_t i) const { return unit_double(hash2(seed, i)); }
};

struct group_key_fn {
  __host__ __device__ int operator()(uint64_t i) const { return i / 16; }
};

struct row_length_fn {
  uint64_t seed;
  __host__ __device__ int64_t operator()(uint64_t i) const { return 4 + hash2(seed, i) % 61; }
};

struct keep_tenth_fn {
  __host__ __device__ bool operator()(uint64_t x) const { return x % 10 == 0; }
};

struct library_data_t {
  size_t n;
  size_t num_groups;
  device_array<double> values;
  device_array<int> group_keys;
  device_array<int> out_group_keys;
  device_array<double> out_group_values;
  device_array<uint64_t> keys;
  device_array<uint64_t> payload;
  device_array<uint64_t> work_keys;
  device_array<uint64_t> work_payload;
  device_array<int64_t> lengths;
  device_array<int64_t> offsets;

  library_data_t(size_t n_, cudaStream_t stream)
    : n(n_),
      num_groups((n_ + 15) / 16),
      values(n_),
      group_keys(n_),
      out_group_keys(num_groups),
      out_group_values(num_groups),
      keys(n_),
      payload(n_),
      work_keys(n_),
      work_payload(n_),
      lengths(n_),
      offsets(n_)
  {
    auto policy = thrust::cuda::par.on(stream);
    thrust::tabulate(policy, values.ptr, values.ptr + n, unit_double_fn{1});
    thrust::tabulate(policy, group_keys.ptr, group_keys.ptr + n, group_key_fn{});
    thrust::tabulate(policy, keys.ptr, keys.ptr + n, hash_fn{2});
    thrust::tabulate(policy, payload.ptr, payload.ptr + n, hash_fn{3});
    thrust::tabulate(policy, lengths.ptr, lengths.ptr + n, row_length_fn{4});
  }
};

void add_library_result(
  context_t& ctx, const char* name, const char* variant, size_t n, const measurement_t& m)
{
  result_t r = make_result("library", name, variant, count_label(n), m.total);
  r.elements = n;
  ctx.add(r);
}

template <typename policy_t>
void bench_library(context_t& ctx, const policy_t& policy, const char* variant, library_data_t& d)
{
  cudaStream_t stream = ctx.stream;
  size_t n            = d.n;

  double sum  = 0;
  auto reduce = ctx.harness.measure([&](lap_t& lap) {
    sum = thrust::reduce(policy, d.values.ptr, d.values.ptr + n);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
  });
  REQUIRE(sum > 0 && sum < n);
  add_library_result(ctx, "thrust::reduce<double>", variant, n, reduce);

  size_t groups      = 0;
  auto reduce_by_key = ctx.harness.measure([&](lap_t& lap) {
    auto ends = thrust::reduce_by_key(policy,
                                      d.group_keys.ptr,
                                      d.group_keys.ptr + n,
                                      d.values.ptr,
                                      d.out_group_keys.ptr,
                                      d.out_group_values.ptr);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
    groups = ends.first - d.out_group_keys.ptr;
  });
  REQUIRE(groups == d.num_groups);
  add_library_result(ctx, "thrust::reduce_by_key<int,double>", variant, n, reduce_by_key);

  auto restore_keys = [&] {
    CUDA_CHECK(cudaMemcpyAsync(
      d.work_keys.ptr, d.keys.ptr, n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
      d.work_payload.ptr, d.payload.ptr, n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  auto sort = ctx.harness.measure(restore_keys, [&](lap_t& lap) {
    thrust::sort(policy, d.work_keys.ptr, d.work_keys.ptr + n);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
  });
  REQUIRE(thrust::is_sorted(thrust::cuda::par.on(stream), d.work_keys.ptr, d.work_keys.ptr + n));
  add_library_result(ctx, "thrust::sort<u64>", variant, n, sort);

  auto sort_by_key = ctx.harness.measure(restore_keys, [&](lap_t& lap) {
    thrust::sort_by_key(policy, d.work_keys.ptr, d.work_keys.ptr + n, d.work_payload.ptr);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
  });
  REQUIRE(thrust::is_sorted(thrust::cuda::par.on(stream), d.work_keys.ptr, d.work_keys.ptr + n));
  add_library_result(ctx, "thrust::sort_by_key<u64,u64>", variant, n, sort_by_key);

  size_t kept  = 0;
  auto copy_if = ctx.harness.measure([&](lap_t& lap) {
    auto end =
      thrust::copy_if(policy, d.keys.ptr, d.keys.ptr + n, d.work_keys.ptr, keep_tenth_fn{});
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
    kept = end - d.work_keys.ptr;
  });
  result_t copy_if_result =
    make_result("library", "thrust::copy_if<u64>", variant, count_label(n), copy_if.total);
  copy_if_result.elements = n;
  copy_if_result.extra.emplace_back("kept", kept);
  ctx.add(copy_if_result);

  auto scan = ctx.harness.measure([&](lap_t& lap) {
    thrust::exclusive_scan(policy, d.lengths.ptr, d.lengths.ptr + n, d.offsets.ptr);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
  });
  add_library_result(ctx, "thrust::exclusive_scan<i64>", variant, n, scan);
}

void bench_cub_sort_pairs(context_t& ctx, library_data_t& d)
{
  cudaStream_t stream = ctx.stream;
  size_t n            = d.n;
  size_t temp_bytes   = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,
                                             temp_bytes,
                                             d.keys.ptr,
                                             d.work_keys.ptr,
                                             d.payload.ptr,
                                             d.work_payload.ptr,
                                             n,
                                             0,
                                             64,
                                             stream));
  device_array<char> temp(temp_bytes);

  auto m = ctx.harness.measure([&](lap_t& lap) {
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp.ptr,
                                               temp_bytes,
                                               d.keys.ptr,
                                               d.work_keys.ptr,
                                               d.payload.ptr,
                                               d.work_payload.ptr,
                                               n,
                                               0,
                                               64,
                                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
  });
  REQUIRE(thrust::is_sorted(thrust::cuda::par.on(stream), d.work_keys.ptr, d.work_keys.ptr + n));
  result_t r = make_result(
    "library", "cub::DeviceRadixSort::SortPairs<u64,u64>", "preallocated", count_label(n), m.total);
  r.elements = n;
  r.extra.emplace_back("temp_storage_bytes", temp_bytes);
  ctx.add(r);
}

void bench_library_all(context_t& ctx)
{
  std::vector<size_t> sizes = ctx.quick
                                ? std::vector<size_t>{1000, 100000, 1000000}
                                : std::vector<size_t>{1000, 10000, 100000, 1000000, 10000000};
  for (size_t n : sizes) {
    library_data_t data(n, ctx.stream);
    bench_library(ctx, rmm::exec_policy_nosync(ctx.stream, *ctx.pool), "rmm_pool", data);
    bench_library(ctx, thrust::cuda::par_nosync.on(ctx.stream), "cuda_malloc", data);
    bench_cub_sort_pairs(ctx, data);
  }
}

void bench_handover_copy(context_t& ctx)
{
  cudaStream_t stream = ctx.stream;
  std::vector<size_t> sizes =
    ctx.quick ? std::vector<size_t>{4 * kb, mb, 16 * mb}
              : std::vector<size_t>{4 * kb, 64 * kb, mb, 16 * mb, 256 * mb, 2 * gb};
  size_t max_bytes = sizes.back();
  pinned_array<char> pinned(max_bytes);
  std::vector<char> pageable(max_bytes, 1);
  device_array<char> device(max_bytes);
  memset(pinned.ptr, 1, max_bytes);

  auto run = [&](char* host, const char* variant, size_t bytes) {
    auto h2d = ctx.harness.measure([&](lap_t& lap) {
      CUDA_CHECK(cudaMemcpyAsync(device.ptr, host, bytes, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      lap.lap();
    });
    result_t up =
      make_result("handover_copy", "host_to_device", variant, bytes_label(bytes), h2d.total);
    up.bytes     = bytes;
    up.byte_rate = true;
    ctx.add(up);

    auto d2h = ctx.harness.measure([&](lap_t& lap) {
      CUDA_CHECK(cudaMemcpyAsync(host, device.ptr, bytes, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      lap.lap();
    });
    result_t down =
      make_result("handover_copy", "device_to_host", variant, bytes_label(bytes), d2h.total);
    down.bytes     = bytes;
    down.byte_rate = true;
    ctx.add(down);
  };

  for (size_t bytes : sizes) {
    run(pinned.ptr, "pinned", bytes);
    run(pageable.data(), "pageable", bytes);
  }
}

void bench_allocation(context_t& ctx)
{
  cudaStream_t stream = ctx.stream;
  std::vector<size_t> sizes =
    ctx.quick ? std::vector<size_t>{mb, 64 * mb} : std::vector<size_t>{mb, 64 * mb, gb, 4 * gb};
  auto add_pair =
    [&](const char* alloc_name, const char* free_name, size_t bytes, const measurement_t& m) {
      result_t a = make_result("allocation", alloc_name, "", bytes_label(bytes), m.phases[0]);
      a.bytes    = bytes;
      ctx.add(a);
      result_t f = make_result("allocation", free_name, "", bytes_label(bytes), m.phases[1]);
      f.bytes    = bytes;
      ctx.add(f);
    };

  for (size_t bytes : sizes) {
    void* p   = nullptr;
    auto host = ctx.harness.measure([&](lap_t& lap) {
      CUDA_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
      lap.lap();
      CUDA_CHECK(cudaFreeHost(p));
      lap.lap();
    });
    add_pair("cudaHostAlloc", "cudaFreeHost", bytes, host);

    auto device = ctx.harness.measure([&](lap_t& lap) {
      CUDA_CHECK(cudaMalloc(&p, bytes));
      lap.lap();
      CUDA_CHECK(cudaFree(p));
      lap.lap();
    });
    add_pair("cudaMalloc", "cudaFree", bytes, device);

    auto pool = ctx.harness.measure([&](lap_t& lap) {
      p = ctx.pool->allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      lap.lap();
      ctx.pool->deallocate(stream, p, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      lap.lap();
    });
    add_pair("rmm_pool_allocate", "rmm_pool_deallocate", bytes, pool);
  }
}

struct value_t {
  double v[2];
};

struct record_t {
  int index;
  int pad;
  value_t value;
};

__global__ void scatter_records_kernel(const record_t* records, int k, value_t* state)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) { state[records[i].index] = records[i].value; }
}

__global__ void set_flags_kernel(const int* indices, int k, uint8_t* flags)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) { flags[indices[i]] = 1; }
}

__global__ void compact_flagged_kernel(
  const uint8_t* changed, const value_t* state, int n, record_t* out, unsigned int* count)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && changed[i]) {
    unsigned int slot = atomicAdd(count, 1u);
    out[slot]         = {i, 0, state[i]};
  }
}

__global__ void emit_frontier_kernel(const int* frontier,
                                     int k,
                                     const uint8_t* changed,
                                     const value_t* state,
                                     record_t* out,
                                     unsigned int* count)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) {
    int j = frontier[i];
    if (changed[j]) {
      unsigned int slot = atomicAdd(count, 1u);
      out[slot]         = {j, 0, state[j]};
    }
  }
}

__global__ void gather_records_kernel(const int* indices,
                                      int k,
                                      const value_t* state,
                                      record_t* out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) { out[i] = {indices[i], 0, state[indices[i]]}; }
}

void bench_dirty_list(context_t& ctx)
{
  cudaStream_t stream = ctx.stream;
  int n               = ctx.quick ? 1000000 : 10000000;
  std::vector<int> ks = ctx.quick ? std::vector<int>{100, 10000, 1000000}
                                  : std::vector<int>{100, 1000, 10000, 100000, 1000000, 10000000};
  int kmax            = ks.back();

  std::vector<value_t> host_state(n);
  for (int i = 0; i < n; ++i) {
    host_state[i] = {unit_double(hash2(11, i)), unit_double(hash2(12, i))};
  }
  std::vector<int> dirty = random_permutation(n, 13);
  std::vector<value_t> host_check(n);
  std::vector<uint8_t> seen(n);

  pinned_array<record_t> pinned_records(kmax);
  pinned_array<unsigned int> pinned_count(1);
  device_array<record_t> d_records(kmax);
  device_array<value_t> d_state(n);
  device_array<uint8_t> d_changed(n);
  device_array<int> d_dirty(kmax);
  device_array<int> d_selected(kmax);
  device_array<unsigned int> d_count(1);
  CUDA_CHECK(cudaMemcpy(d_dirty.ptr, dirty.data(), kmax * sizeof(int), cudaMemcpyHostToDevice));

  auto add = [&](const char* name,
                 const char* variant,
                 int k,
                 const measurement_t& m,
                 const char* phase0,
                 const char* phase1) {
    result_t r  = make_result("dirty_list", name, variant, count_label(k), m.total);
    r.elements  = k;
    r.bytes     = k * sizeof(record_t);
    r.byte_rate = true;
    r.phases.push_back({phase0, m.phases[0]});
    r.phases.push_back({phase1, m.phases[1]});
    ctx.add(r);
  };

  for (int k : ks) {
    CUDA_CHECK(cudaMemsetAsync(d_state.ptr, 0, n * sizeof(value_t), stream));
    auto m = ctx.harness.measure([&](lap_t& lap) {
      for (int i = 0; i < k; ++i) {
        int j                 = dirty[i];
        pinned_records.ptr[i] = {j, 0, host_state[j]};
      }
      lap.lap();
      CUDA_CHECK(cudaMemcpyAsync(
        d_records.ptr, pinned_records.ptr, k * sizeof(record_t), cudaMemcpyHostToDevice, stream));
      scatter_records_kernel<<<(k + 255) / 256, 256, 0, stream>>>(d_records.ptr, k, d_state.ptr);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      lap.lap();
      CUDA_CHECK(cudaGetLastError());
    });
    CUDA_CHECK(
      cudaMemcpy(host_check.data(), d_state.ptr, n * sizeof(value_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < k; ++i) {
      int j = dirty[i];
      REQUIRE(memcmp(&host_check[j], &host_state[j], sizeof(value_t)) == 0);
    }
    add("host_to_device", "pinned_copy_scatter", k, m, "host_fill", "copy_scatter_sync");
  }

  CUDA_CHECK(
    cudaMemcpy(d_state.ptr, host_state.data(), n * sizeof(value_t), cudaMemcpyHostToDevice));

  size_t select_temp_bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,
                                        select_temp_bytes,
                                        thrust::make_counting_iterator(0),
                                        d_changed.ptr,
                                        d_selected.ptr,
                                        d_count.ptr,
                                        n,
                                        stream));
  device_array<char> select_temp(select_temp_bytes);

  auto check_records = [&](int k, bool ordered) {
    int count = *pinned_count.ptr;
    REQUIRE(count == k);
    std::fill(seen.begin(), seen.end(), 0);
    for (int i = 0; i < k; ++i) {
      const record_t& r = pinned_records.ptr[i];
      REQUIRE(memcmp(&r.value, &host_state[r.index], sizeof(value_t)) == 0);
      REQUIRE(!ordered || i == 0 || pinned_records.ptr[i - 1].index < r.index);
      ++seen[r.index];
    }
    for (int i = 0; i < k; ++i) {
      REQUIRE(seen[dirty[i]] == 1);
    }
  };

  auto copy_records = [&](lap_t& lap) {
    CUDA_CHECK(cudaMemcpyAsync(pinned_records.ptr,
                               d_records.ptr,
                               *pinned_count.ptr * sizeof(record_t),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
    CUDA_CHECK(cudaGetLastError());
  };

  for (int k : ks) {
    CUDA_CHECK(cudaMemsetAsync(d_changed.ptr, 0, n, stream));
    set_flags_kernel<<<(k + 255) / 256, 256, 0, stream>>>(d_dirty.ptr, k, d_changed.ptr);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemsetAsync(d_records.ptr, 0, k * sizeof(record_t), stream));
    auto scan_atomic = ctx.harness.measure([&](lap_t& lap) {
      CUDA_CHECK(cudaMemsetAsync(d_count.ptr, 0, sizeof(unsigned int), stream));
      compact_flagged_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        d_changed.ptr, d_state.ptr, n, d_records.ptr, d_count.ptr);
      CUDA_CHECK(cudaMemcpyAsync(
        pinned_count.ptr, d_count.ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      lap.lap();
      copy_records(lap);
    });
    check_records(k, false);
    add("device_to_host", "flag_scan_atomic", k, scan_atomic, "compact_count", "copy_records");

    CUDA_CHECK(cudaMemsetAsync(d_records.ptr, 0, k * sizeof(record_t), stream));
    auto scan_ordered = ctx.harness.measure([&](lap_t& lap) {
      CUDA_CHECK(cub::DeviceSelect::Flagged(select_temp.ptr,
                                            select_temp_bytes,
                                            thrust::make_counting_iterator(0),
                                            d_changed.ptr,
                                            d_selected.ptr,
                                            d_count.ptr,
                                            n,
                                            stream));
      CUDA_CHECK(cudaMemcpyAsync(
        pinned_count.ptr, d_count.ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      lap.lap();
      int count = *pinned_count.ptr;
      gather_records_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        d_selected.ptr, count, d_state.ptr, d_records.ptr);
      copy_records(lap);
    });
    check_records(k, true);
    add("device_to_host",
        "flag_scan_ordered",
        k,
        scan_ordered,
        "select_count",
        "gather_copy_records");

    CUDA_CHECK(cudaMemsetAsync(d_records.ptr, 0, k * sizeof(record_t), stream));
    auto emit = ctx.harness.measure([&](lap_t& lap) {
      CUDA_CHECK(cudaMemsetAsync(d_count.ptr, 0, sizeof(unsigned int), stream));
      emit_frontier_kernel<<<(k + 255) / 256, 256, 0, stream>>>(
        d_dirty.ptr, k, d_changed.ptr, d_state.ptr, d_records.ptr, d_count.ptr);
      CUDA_CHECK(cudaMemcpyAsync(
        pinned_count.ptr, d_count.ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      lap.lap();
      copy_records(lap);
    });
    check_records(k, false);
    add("device_to_host", "frontier_emit_atomic", k, emit, "compact_count", "copy_records");
  }
}

struct csr_column_fn {
  uint64_t seed;
  int num_cols;
  __host__ __device__ int operator()(uint64_t e) const { return hash2(seed, e) % num_cols; }
};

struct frontier_length_fn {
  const int* frontier;
  const int64_t* row_offsets;
  int f;
  __host__ __device__ int64_t operator()(int i) const
  {
    if (i == f) { return 0; }
    int row = frontier[i];
    return row_offsets[row + 1] - row_offsets[row];
  }
};

// Buffer layout: f + 1 offsets (int64), then all values (double), then all column ids (int).
// threads_per_row = 1 copies a row serially in one thread; 32 spreads it over one warp.
__global__ void gather_rows_kernel(const int* frontier,
                                   int f,
                                   const int64_t* row_offsets,
                                   const int* __restrict__ cols,
                                   const double* __restrict__ vals,
                                   const int64_t* out_offsets,
                                   int threads_per_row)
{
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int i      = thread / threads_per_row;
  if (i < f) {
    double* out_vals = (double*)(out_offsets + f + 1);
    int* out_cols    = (int*)(out_vals + out_offsets[f]);
    int64_t src      = row_offsets[frontier[i]];
    int64_t dst      = out_offsets[i];
    int64_t len      = out_offsets[i + 1] - dst;
    for (int64_t e = thread % threads_per_row; e < len; e += threads_per_row) {
      out_vals[dst + e] = vals[src + e];
      out_cols[dst + e] = cols[src + e];
    }
  }
}

void bench_gather(context_t& ctx)
{
  cudaStream_t stream          = ctx.stream;
  int64_t target_nnz           = ctx.quick ? 10000000 : 100000000;
  std::vector<int64_t> targets = ctx.quick
                                   ? std::vector<int64_t>{1000, 100000}
                                   : std::vector<int64_t>{1000, 10000, 100000, 1000000, 10000000};
  auto plain                   = thrust::cuda::par.on(stream);

  // Mean row length is 34; generate 5% more rows than needed and cut at the target.
  int generated_rows = target_nnz / 34 + target_nnz / 680;
  device_array<int64_t> d_row_offsets(generated_rows + 1);
  {
    device_array<int64_t> d_lengths(generated_rows + 1);
    thrust::tabulate(plain, d_lengths.ptr, d_lengths.ptr + generated_rows + 1, row_length_fn{21});
    thrust::exclusive_scan(
      plain, d_lengths.ptr, d_lengths.ptr + generated_rows + 1, d_row_offsets.ptr);
  }
  std::vector<int64_t> row_offsets(generated_rows + 1);
  CUDA_CHECK(cudaMemcpy(row_offsets.data(),
                        d_row_offsets.ptr,
                        (generated_rows + 1) * sizeof(int64_t),
                        cudaMemcpyDeviceToHost));
  int m =
    std::upper_bound(row_offsets.begin(), row_offsets.end(), target_nnz) - row_offsets.begin() - 1;
  REQUIRE(m < generated_rows);
  int64_t nnz = row_offsets[m];

  csr_column_fn column_fn{22, m};
  unit_double_fn value_fn{23};
  device_array<int> d_cols(nnz);
  device_array<double> d_vals(nnz);
  thrust::tabulate(plain, d_cols.ptr, d_cols.ptr + nnz, column_fn);
  thrust::tabulate(plain, d_vals.ptr, d_vals.ptr + nnz, value_fn);

  std::vector<int> order = random_permutation(m, 24);
  std::vector<int> frontier_rows;
  std::vector<int64_t> frontier_entries;
  {
    int f         = 0;
    int64_t total = 0;
    for (int64_t target : targets) {
      while (total + row_offsets[order[f] + 1] - row_offsets[order[f]] <= target) {
        total += row_offsets[order[f] + 1] - row_offsets[order[f]];
        ++f;
      }
      frontier_rows.push_back(f);
      frontier_entries.push_back(total);
    }
  }
  int fmax = frontier_rows.back();
  size_t max_buf_bytes =
    (fmax + 1) * sizeof(int64_t) + frontier_entries.back() * (sizeof(double) + sizeof(int));

  pinned_array<int> pinned_frontier(fmax);
  pinned_array<int64_t> pinned_total(1);
  pinned_array<char> pinned_buf(max_buf_bytes);
  device_array<int> d_frontier(fmax);
  device_array<char> d_buf(max_buf_bytes);
  memcpy(pinned_frontier.ptr, order.data(), fmax * sizeof(int));

  auto policy = rmm::exec_policy_nosync(stream, *ctx.pool);

  for (size_t t = 0; t < targets.size(); ++t) {
    int f                = frontier_rows[t];
    int64_t entries      = frontier_entries[t];
    size_t buf_bytes     = (f + 1) * sizeof(int64_t) + entries * (sizeof(double) + sizeof(int));
    int64_t* out_offsets = (int64_t*)d_buf.ptr;
    std::string label    = count_label(targets[t]);

    auto upload             = ctx.harness.measure([&](lap_t& lap) {
      CUDA_CHECK(cudaMemcpyAsync(
        d_frontier.ptr, pinned_frontier.ptr, f * sizeof(int), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      lap.lap();
    });
    result_t upload_result  = make_result("gather", "frontier_upload", "", label, upload.total);
    upload_result.elements  = f;
    upload_result.bytes     = f * sizeof(int);
    upload_result.byte_rate = true;
    ctx.add(upload_result);

    auto run = [&](const char* variant, int threads_per_row, int block_size) {
      CUDA_CHECK(cudaMemsetAsync(d_buf.ptr, 0, buf_bytes, stream));
      int blocks    = (f * threads_per_row + block_size - 1) / block_size;
      auto m_gather = ctx.harness.measure([&](lap_t& lap) {
        thrust::transform_exclusive_scan(policy,
                                         thrust::make_counting_iterator(0),
                                         thrust::make_counting_iterator(f + 1),
                                         out_offsets,
                                         frontier_length_fn{d_frontier.ptr, d_row_offsets.ptr, f},
                                         (int64_t)0,
                                         thrust::plus<int64_t>());
        gather_rows_kernel<<<blocks, block_size, 0, stream>>>(d_frontier.ptr,
                                                              f,
                                                              d_row_offsets.ptr,
                                                              d_cols.ptr,
                                                              d_vals.ptr,
                                                              out_offsets,
                                                              threads_per_row);
        CUDA_CHECK(cudaMemcpyAsync(
          pinned_total.ptr, out_offsets + f, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        lap.lap();
        size_t bytes =
          (f + 1) * sizeof(int64_t) + *pinned_total.ptr * (sizeof(double) + sizeof(int));
        CUDA_CHECK(
          cudaMemcpyAsync(pinned_buf.ptr, d_buf.ptr, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        lap.lap();
        CUDA_CHECK(cudaGetLastError());
      });

      REQUIRE(*pinned_total.ptr == entries);
      const int64_t* got_offsets = (const int64_t*)pinned_buf.ptr;
      const double* got_vals     = (const double*)(got_offsets + f + 1);
      const int* got_cols        = (const int*)(got_vals + entries);
      for (int i = 0; i < f; ++i) {
        int64_t src = row_offsets[order[i]];
        int64_t dst = got_offsets[i];
        REQUIRE(got_offsets[i + 1] - dst == row_offsets[order[i] + 1] - src);
        for (int64_t e = 0; e < got_offsets[i + 1] - dst; ++e) {
          REQUIRE(got_vals[dst + e] == value_fn(src + e));
          REQUIRE(got_cols[dst + e] == column_fn(src + e));
        }
      }

      result_t r  = make_result("gather", "gather_rows_to_host", variant, label, m_gather.total);
      r.elements  = entries;
      r.bytes     = buf_bytes;
      r.byte_rate = true;
      r.phases.push_back({"scan_gather_count", m_gather.phases[0]});
      r.phases.push_back({"copy_buffer", m_gather.phases[1]});
      r.extra.emplace_back("frontier_rows", f);
      r.extra.emplace_back("matrix_rows", m);
      r.extra.emplace_back("matrix_nonzeros", nnz);
      ctx.add(r);
    };
    run("thread_per_row", 1, 256);
    run("thread_per_row_block_32", 1, 32);
    run("warp_per_row", 32, 256);
  }
}

__global__ void add_one_kernel(uint64_t* data, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { data[i] += 1; }
}

void bench_unified_memory(context_t& ctx)
{
  cudaStream_t stream = ctx.stream;
  size_t bytes        = 64 * mb;
  size_t n            = bytes / sizeof(uint64_t);
  int blocks          = (n + 255) / 256;
  std::string label   = bytes_label(bytes);

  uint64_t* data             = nullptr;
  auto managed               = ctx.harness.measure([&](lap_t& lap) {
    CUDA_CHECK(cudaMallocManaged((void**)&data, bytes, cudaMemAttachGlobal));
    lap.lap();
    for (size_t i = 0; i < n; ++i) {
      data[i] = i;
    }
    lap.lap();
    add_one_kernel<<<blocks, 256, 0, stream>>>(data, n);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
    for (size_t i = 0; i < n; ++i) {
      data[i] += 1;
    }
    lap.lap();
    uint64_t first = data[0];
    uint64_t last  = data[n - 1];
    CUDA_CHECK(cudaFree(data));
    lap.lap();
    CUDA_CHECK(cudaGetLastError());
    REQUIRE(first == 2 && last == n + 1);
  });
  const char* phase_names[5] = {"cudaMallocManaged",
                                "host_first_touch",
                                "kernel_touch_after_host",
                                "host_touch_after_kernel",
                                "cudaFree"};
  for (int p = 0; p < 5; ++p) {
    result_t r = make_result("unified_memory", phase_names[p], "managed", label, managed.phases[p]);
    r.bytes    = bytes;
    r.byte_rate = p >= 1 && p <= 3;
    ctx.add(r);
  }

  pinned_array<uint64_t> pinned(n);
  device_array<uint64_t> device(n);
  CUDA_CHECK(cudaMemsetAsync(device.ptr, 0, bytes, stream));
  memset(pinned.ptr, 0, bytes);

  auto host_reference = ctx.harness.measure([&](lap_t& lap) {
    for (size_t i = 0; i < n; ++i) {
      pinned.ptr[i] += 1;
    }
    lap.lap();
  });
  result_t host_result =
    make_result("unified_memory", "host_touch_reference", "pinned", label, host_reference.total);
  host_result.bytes     = bytes;
  host_result.byte_rate = true;
  ctx.add(host_result);

  auto kernel_reference  = ctx.harness.measure([&](lap_t& lap) {
    add_one_kernel<<<blocks, 256, 0, stream>>>(device.ptr, n);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    lap.lap();
    CUDA_CHECK(cudaGetLastError());
  });
  result_t kernel_result = make_result(
    "unified_memory", "kernel_touch_reference", "device", label, kernel_reference.total);
  kernel_result.bytes     = bytes;
  kernel_result.byte_rate = true;
  ctx.add(kernel_result);
}

std::string trim(const std::string& text)
{
  size_t begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) { return ""; }
  size_t end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1);
}

std::string run_command(const char* command)
{
  FILE* pipe = popen(command, "r");
  REQUIRE(pipe != nullptr);
  std::string output;
  char buffer[4096];
  size_t got;
  while ((got = fread(buffer, 1, sizeof(buffer), pipe)) > 0) {
    output.append(buffer, got);
  }
  pclose(pipe);
  return trim(output);
}

std::vector<std::string> split(const std::string& text, char separator)
{
  std::vector<std::string> parts;
  size_t begin = 0;
  while (begin <= text.size()) {
    size_t end = text.find(separator, begin);
    if (end == std::string::npos) { end = text.size(); }
    parts.push_back(trim(text.substr(begin, end - begin)));
    begin = end + 1;
  }
  return parts;
}

std::string lines_containing(const std::string& text, const std::string& needle)
{
  std::string found;
  for (const std::string& line : split(text, '\n')) {
    if (line.find(needle) != std::string::npos) { found += (found.empty() ? "" : "\n") + line; }
  }
  return found;
}

size_t line_count(const std::string& text) { return text.empty() ? 0 : split(text, '\n').size(); }

const char* gpu_query =
  "nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu "
  "--format=csv,noheader,nounits";
const char* apps_query =
  "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits";

struct system_info_t {
  std::string gpu_name;
  std::string driver_version;
  std::string cpu_model;
  std::string cpu_threads;
  std::string compute_capability;
  std::string apps_before;
  std::string apps_after;
  std::string apps_most;
  std::string loadavg_before;
  std::string loadavg_after;
  double gpu_memory_total_mib;
  double gpu_memory_used_mib_before;
  double gpu_utilization_before;
  int cuda_runtime_version;
  int cuda_driver_api_version;
  bool other_gpu_processes;
  bool other_cpu_load;
};

std::vector<std::string> make_notes(const system_info_t& sys)
{
  return {
    "Times are wall-clock microseconds (std::chrono::steady_clock) around the stated interval. "
    "Every interval that uses the device ends after cudaStreamSynchronize, except "
    "empty_kernel_launch_no_sync. Percentiles interpolate linearly between order statistics.",
    "Each measurement runs warm-up iterations first, then 'reps' repetitions. The count is the "
    "default in 'harness' unless that many repetitions would exceed the time budget; then it is "
    "budget / iteration time, and at least min_repetitions.",
    "One user stream (cudaStreamCreate), default scheduling flag (cudaDeviceScheduleAuto), one "
    "host thread.",
    "library: variant rmm_pool uses rmm::exec_policy_nosync with a pool memory resource for "
    "Thrust temporaries; variant cuda_malloc uses thrust::cuda::par_nosync, whose temporaries "
    "come from cudaMalloc. Both are followed by cudaStreamSynchronize inside the interval. "
    "reduce_by_key has one key per 16 elements; copy_if keeps about 10% (exact count in "
    "'kept'); every sort repetition starts from the same unsorted input, restored outside the "
    "interval.",
    "Sizes are binary (1 MB = 2^20 bytes) and gb_per_s uses 2^30 bytes. Pageable copies use "
    "cudaMemcpyAsync on the same stream followed by a synchronisation, like the pinned ones.",
    "allocation: cudaHostAlloc with cudaHostAllocDefault. The rmm pool is created once with a "
    "fixed size before the measurements, so pool growth is never inside an interval.",
    "dirty_list: a record is 24 bytes (int index, int pad, 16-byte value); the state array has "
    "1e7 16-byte values (1e6 in quick mode); indices are random and distinct. host_to_device = "
    "the host fills k records into a pinned buffer (phase host_fill), one copy, one scatter "
    "kernel, synchronise. host_fill reads each value from a random position of a host copy of "
    "the state array, so above about 1e3 records it is mostly host cache misses and not the "
    "write into the pinned buffer; copy_scatter_sync is the device part. device_to_host has "
    "three variants. flag_scan_atomic: one kernel over "
    "all changed flags of the state array with an integer atomic slot counter (record order "
    "not deterministic). flag_scan_ordered: cub::DeviceSelect::Flagged over the same flags with "
    "preallocated temporary storage, count read back, one gather kernel (deterministic order). "
    "frontier_emit_atomic: one kernel over a device list of the k candidate indices with an "
    "atomic slot counter. Each variant reads the count back to pinned memory, then copies count "
    "records to pinned memory.",
    "gather: random CSR generated on the device (row lengths uniform in 4..64, 32-bit column "
    "ids, double values, 64-bit offsets). The frontier is a random set of distinct rows whose "
    "total length is the largest not above the size label. gather_rows_to_host = "
    "transform_exclusive_scan of the frontier row lengths, one gather kernel into one "
    "contiguous buffer (offsets, values, column ids), count read back (phase "
    "scan_gather_count), one copy of the buffer to pinned memory (phase copy_buffer). The three "
    "variants are one kernel with different launch shapes. thread_per_row copies each row "
    "serially in one thread, 256 rows per block; thread_per_row_block_32 does the same with 32 "
    "rows per block; warp_per_row spreads each row over the 32 threads of one warp, 8 rows per "
    "block. In the one-thread-per-row variants a warp works on 32 different rows at once, and "
    "the time depends on the block size, not only on the row lengths. frontier_upload is the "
    "separate cost of sending the frontier row ids from pinned memory and is not part of "
    "gather_rows_to_host.",
    "unified_memory: cudaMallocManaged on an x86 node with a discrete GPU. Every hand-over "
    "between host and device is served by page faults and page migration. This is NOT the "
    "Grace Hopper behaviour and must not be used to estimate it. The reference rows give the "
    "same host loop on pinned memory and the same kernel on device memory.",
    sys.other_gpu_processes
      ? "Other processes used this GPU during the run (system.gpu_*_before, "
        "system.gpu_compute_apps_*); timings are disturbed."
      : "nvidia-smi showed no other process on this GPU before the run or after any group. A "
        "process that came and went inside one group is not seen.",
    sys.other_cpu_load
      ? "Other processes loaded the CPUs of this node when the run started (1-minute load "
        "average of 1 or more, system.cpu_loadavg_before); host-side timings are disturbed."
      : "The 1-minute load average was below 1 when the run started."};
}

std::string json_string(const std::string& text)
{
  std::string out = "\"";
  for (char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out += c;
    }
  }
  return out + "\"";
}

void write_stats(FILE* out, const stats_t& s)
{
  fprintf(out,
          "\"reps\": %d, \"median_us\": %.9g, \"p10_us\": %.9g, \"p90_us\": %.9g",
          s.reps,
          s.median_us,
          s.p10_us,
          s.p90_us);
}

double elements_per_s(const result_t& r) { return r.elements / (r.stats.median_us * 1e-6); }

double gb_per_s(const result_t& r) { return r.bytes / gb / (r.stats.median_us * 1e-6); }

void write_json(const char* path,
                const context_t& ctx,
                const system_info_t& sys,
                const std::vector<std::string>& notes)
{
  FILE* out = fopen(path, "w");
  REQUIRE(out != nullptr);
  fprintf(
    out, "{\n  \"tool\": \"fixed_costs\",\n  \"mode\": \"%s\",\n", ctx.quick ? "quick" : "full");
  fprintf(out, "  \"system\": {\n");
  fprintf(out, "    \"gpu_name\": %s,\n", json_string(sys.gpu_name).c_str());
  fprintf(out, "    \"gpu_memory_total_mib\": %.0f,\n", sys.gpu_memory_total_mib);
  fprintf(
    out, "    \"gpu_compute_capability\": %s,\n", json_string(sys.compute_capability).c_str());
  fprintf(out, "    \"driver_version\": %s,\n", json_string(sys.driver_version).c_str());
  fprintf(out, "    \"cuda_runtime_version\": %d,\n", sys.cuda_runtime_version);
  fprintf(out, "    \"cuda_driver_api_version\": %d,\n", sys.cuda_driver_api_version);
  fprintf(out, "    \"cuda_compile_version\": %d,\n", CUDART_VERSION);
  fprintf(out,
          "    \"cccl_version\": \"%d.%d.%d\",\n",
          THRUST_MAJOR_VERSION,
          THRUST_MINOR_VERSION,
          THRUST_SUBMINOR_VERSION);
  fprintf(out, "    \"rmm_version\": \"%s\",\n", RMM_VERSION);
  fprintf(out, "    \"cpu_model\": %s,\n", json_string(sys.cpu_model).c_str());
  fprintf(out, "    \"cpu_threads\": %s,\n", sys.cpu_threads.c_str());
  fprintf(
    out, "    \"other_gpu_processes_observed\": %s,\n", sys.other_gpu_processes ? "true" : "false");
  fprintf(out, "    \"other_cpu_load_observed\": %s,\n", sys.other_cpu_load ? "true" : "false");
  fprintf(out, "    \"gpu_memory_used_mib_before\": %.0f,\n", sys.gpu_memory_used_mib_before);
  fprintf(out, "    \"gpu_utilization_percent_before\": %.0f,\n", sys.gpu_utilization_before);
  fprintf(out, "    \"gpu_compute_apps_before\": %s,\n", json_string(sys.apps_before).c_str());
  fprintf(out, "    \"gpu_compute_apps_after\": %s,\n", json_string(sys.apps_after).c_str());
  fprintf(out, "    \"gpu_compute_apps_most_seen\": %s,\n", json_string(sys.apps_most).c_str());
  fprintf(out, "    \"cpu_loadavg_before\": %s,\n", json_string(sys.loadavg_before).c_str());
  fprintf(out, "    \"cpu_loadavg_after\": %s\n", json_string(sys.loadavg_after).c_str());
  fprintf(out, "  },\n");
  fprintf(out,
          "  \"harness\": {\"repetitions\": %d, \"min_repetitions\": %d, \"budget_s\": %.3g, "
          "\"max_warmup_iterations\": %d, \"warmup_budget_s\": %.3g},\n",
          ctx.harness.reps,
          ctx.harness.min_reps,
          ctx.harness.budget_us * 1e-6,
          ctx.harness.warmup,
          ctx.harness.warmup_budget_us * 1e-6);
  fprintf(out, "  \"notes\": [\n");
  for (size_t i = 0; i < notes.size(); ++i) {
    fprintf(out, "    %s%s\n", json_string(notes[i]).c_str(), i + 1 < notes.size() ? "," : "");
  }
  fprintf(out, "  ],\n  \"results\": [\n");
  for (size_t i = 0; i < ctx.results.size(); ++i) {
    const result_t& r = ctx.results[i];
    fprintf(out,
            "    {\"group\": %s, \"name\": %s, \"variant\": %s, \"size\": %s, ",
            json_string(r.group).c_str(),
            json_string(r.name).c_str(),
            json_string(r.variant).c_str(),
            json_string(r.size_label).c_str());
    if (r.elements > 0) { fprintf(out, "\"elements\": %.0f, ", r.elements); }
    if (r.bytes > 0) { fprintf(out, "\"bytes\": %.0f, ", r.bytes); }
    write_stats(out, r.stats);
    if (r.elements > 0) { fprintf(out, ", \"elements_per_s\": %.9g", elements_per_s(r)); }
    if (r.byte_rate) { fprintf(out, ", \"gb_per_s\": %.9g", gb_per_s(r)); }
    for (const auto& e : r.extra) {
      fprintf(out, ", %s: %.0f", json_string(e.first).c_str(), e.second);
    }
    if (!r.phases.empty()) {
      fprintf(out, ", \"phases\": [");
      for (size_t p = 0; p < r.phases.size(); ++p) {
        fprintf(
          out, "%s{\"name\": %s, ", p > 0 ? ", " : "", json_string(r.phases[p].first).c_str());
        write_stats(out, r.phases[p].second);
        fprintf(out, "}");
      }
      fprintf(out, "]");
    }
    fprintf(out, "}%s\n", i + 1 < ctx.results.size() ? "," : "");
  }
  fprintf(out, "  ]\n}\n");
  REQUIRE(fclose(out) == 0);
}

std::string format_value(double v)
{
  char buffer[32];
  snprintf(buffer,
           sizeof(buffer),
           v >= 1000  ? "%.0f"
           : v >= 100 ? "%.1f"
           : v >= 10  ? "%.2f"
                      : "%.3f",
           v);
  return buffer;
}

struct cell_t {
  std::string row;
  std::string column;
  double value;
};

void print_pivot(const std::string& title, const std::vector<cell_t>& cells)
{
  std::vector<std::string> rows;
  std::vector<std::string> columns;
  size_t row_width = 0;
  for (const cell_t& c : cells) {
    if (std::find(rows.begin(), rows.end(), c.row) == rows.end()) { rows.push_back(c.row); }
    if (std::find(columns.begin(), columns.end(), c.column) == columns.end()) {
      columns.push_back(c.column);
    }
    row_width = std::max(row_width, c.row.size());
  }
  printf("\n%s\n%-*s", title.c_str(), (int)row_width, "");
  for (const std::string& column : columns) {
    printf(" %11s", column.c_str());
  }
  printf("\n");
  for (const std::string& row : rows) {
    printf("%-*s", (int)row_width, row.c_str());
    for (const std::string& column : columns) {
      std::string text = "-";
      for (const cell_t& c : cells) {
        if (c.row == row && c.column == column) { text = format_value(c.value); }
      }
      printf(" %11s", text.c_str());
    }
    printf("\n");
  }
}

std::string row_label(const result_t& r)
{
  return r.variant.empty() ? r.name : r.name + " [" + r.variant + "]";
}

void print_tables(const std::vector<result_t>& results)
{
  std::vector<std::string> groups;
  for (const result_t& r : results) {
    if (std::find(groups.begin(), groups.end(), r.group) == groups.end()) {
      groups.push_back(r.group);
    }
  }
  for (const std::string& group : groups) {
    std::vector<cell_t> times;
    std::vector<cell_t> element_rates;
    std::vector<cell_t> byte_rates;
    bool sized = false;
    for (const result_t& r : results) {
      if (r.group != group) { continue; }
      sized = sized || !r.size_label.empty();
      times.push_back({row_label(r), r.size_label, r.stats.median_us});
      for (const auto& phase : r.phases) {
        times.push_back({row_label(r) + " / " + phase.first, r.size_label, phase.second.median_us});
      }
      if (r.elements > 0) {
        element_rates.push_back({row_label(r), r.size_label, elements_per_s(r) * 1e-6});
      } else if (r.byte_rate) {
        byte_rates.push_back({row_label(r), r.size_label, gb_per_s(r)});
      }
    }
    if (!sized) {
      printf("\n%s: microseconds, median (p10, p90)\n", group.c_str());
      for (const result_t& r : results) {
        if (r.group != group) { continue; }
        printf("%-44s %11s (%s, %s)\n",
               row_label(r).c_str(),
               format_value(r.stats.median_us).c_str(),
               format_value(r.stats.p10_us).c_str(),
               format_value(r.stats.p90_us).c_str());
      }
      continue;
    }
    print_pivot(group + ": median microseconds", times);
    if (!element_rates.empty()) {
      print_pivot(group + ": million elements per second", element_rates);
    }
    if (!byte_rates.empty()) { print_pivot(group + ": GB per second", byte_rates); }
  }
}

bool selected(const std::string& only, const char* group)
{
  if (only.empty()) { return true; }
  std::vector<std::string> names = split(only, ',');
  return std::find(names.begin(), names.end(), group) != names.end();
}

}  // namespace

int main(int argc, char** argv)
{
  const char* out_path = nullptr;
  bool quick           = false;
  std::string only;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--quick") == 0) {
      quick = true;
    } else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
      only = argv[++i];
    } else if (out_path == nullptr && argv[i][0] != '-') {
      out_path = argv[i];
    } else {
      out_path = nullptr;
      break;
    }
  }
  if (out_path == nullptr) {
    fprintf(stderr,
            "usage: fixed_costs <output.json> [--quick] [--only group,group,...]\n"
            "groups: launch readback library handover_copy allocation dirty_list gather "
            "unified_memory\n");
    return 1;
  }

  // Taken before this process creates a CUDA context, so anything seen belongs to other processes.
  std::string gpus_before = run_command(gpu_query);
  system_info_t sys;
  sys.apps_before    = run_command(apps_query);
  sys.loadavg_before = run_command("cat /proc/loadavg");
  sys.cpu_model      = run_command("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-");
  sys.cpu_threads    = run_command("nproc");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaRuntimeGetVersion(&sys.cuda_runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&sys.cuda_driver_api_version));
  std::string uuid = "GPU-";
  for (int i = 0; i < 16; ++i) {
    char hex[3];
    snprintf(hex, sizeof(hex), "%02x", (unsigned char)prop.uuid.bytes[i]);
    uuid += hex;
    if (i == 3 || i == 5 || i == 7 || i == 9) { uuid += '-'; }
  }
  std::vector<std::string> gpu_fields = split(lines_containing(gpus_before, uuid), ',');
  REQUIRE(gpu_fields.size() == 6);
  sys.gpu_name                   = prop.name;
  sys.compute_capability         = std::to_string(prop.major) + "." + std::to_string(prop.minor);
  sys.driver_version             = gpu_fields[2];
  sys.gpu_memory_total_mib       = std::stod(gpu_fields[3]);
  sys.gpu_memory_used_mib_before = std::stod(gpu_fields[4]);
  sys.gpu_utilization_before     = std::stod(gpu_fields[5]);
  sys.apps_before                = lines_containing(sys.apps_before, uuid);

  context_t ctx;
  ctx.quick   = quick;
  ctx.harness = quick ? harness_t{30, 3, 5, 0.2e6, 0.1e6} : harness_t{200, 10, 20, 8e6, 0.5e6};
  CUDA_CHECK(cudaStreamCreate(&ctx.stream));
  rmm::mr::pool_memory_resource pool(rmm::mr::cuda_memory_resource(), (quick ? 1 : 8) * gb);
  ctx.pool = &pool;

  auto run_group = [&](const char* group, void (*bench)(context_t&)) {
    if (!selected(only, group)) { return; }
    bench(ctx);
    std::string apps = lines_containing(run_command(apps_query), uuid);
    if (line_count(apps) > line_count(sys.apps_most)) { sys.apps_most = apps; }
  };
  run_group("launch", bench_launch);
  run_group("readback", bench_readback);
  run_group("library", bench_library_all);
  run_group("handover_copy", bench_handover_copy);
  run_group("allocation", bench_allocation);
  run_group("dirty_list", bench_dirty_list);
  run_group("gather", bench_gather);
  run_group("unified_memory", bench_unified_memory);
  CUDA_CHECK(cudaStreamDestroy(ctx.stream));

  sys.apps_after    = lines_containing(run_command(apps_query), uuid);
  sys.loadavg_after = run_command("cat /proc/loadavg");
  // An idle GPU reports a few MiB at most; this process alone gives one line in the app list.
  sys.other_gpu_processes = sys.gpu_memory_used_mib_before >= 16 ||
                            sys.gpu_utilization_before > 0 || !sys.apps_before.empty() ||
                            line_count(sys.apps_most) > 1 || line_count(sys.apps_after) > 1;
  sys.other_cpu_load = std::stod(sys.loadavg_before) >= 1;

  write_json(out_path, ctx, sys, make_notes(sys));
  print_tables(ctx.results);
  printf(
    "\nGPU: %s, driver %s, CUDA runtime %d; CPU: %s\n"
    "other GPU processes observed: %s; other CPU load observed: %s (load average %s)\n",
    sys.gpu_name.c_str(),
    sys.driver_version.c_str(),
    sys.cuda_runtime_version,
    sys.cpu_model.c_str(),
    sys.other_gpu_processes ? "yes" : "no",
    sys.other_cpu_load ? "yes" : "no",
    sys.loadavg_before.c_str());
  printf("JSON written to %s\n", out_path);
  return 0;
}
