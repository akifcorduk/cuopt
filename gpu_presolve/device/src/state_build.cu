/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/device/state_build.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/version.cuh>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace gprs {
namespace device {

namespace {

#define CUDA_CHECK(call)                              \
  do {                                                \
    cudaError_t cuda_status_ = (call);                \
    if (cuda_status_ != cudaSuccess) {                \
      fprintf(stderr,                                 \
              "gprs device: %s failed: %s (%s:%d)\n", \
              #call,                                  \
              cudaGetErrorString(cuda_status_),       \
              __FILE__,                               \
              __LINE__);                              \
      abort();                                        \
    }                                                 \
  } while (0)

#define CUSPARSE_CHECK(call)                            \
  do {                                                  \
    cusparseStatus_t cusparse_status_ = (call);         \
    if (cusparse_status_ != CUSPARSE_STATUS_SUCCESS) {  \
      fprintf(stderr,                                   \
              "gprs device: %s failed: %s (%s:%d)\n",   \
              #call,                                    \
              cusparseGetErrorString(cusparse_status_), \
              __FILE__,                                 \
              __LINE__);                                \
      abort();                                          \
    }                                                   \
  } while (0)

// Thrown where the device has no memory left and caught by build().
struct out_of_device_memory_t {};

// A cuSPARSE call that allocates on the device.
#define CUSPARSE_CHECK_MEMORY(call)                                \
  do {                                                             \
    cusparseStatus_t cusparse_memory_status_ = (call);             \
    if (cusparse_memory_status_ == CUSPARSE_STATUS_ALLOC_FAILED) { \
      throw out_of_device_memory_t();                              \
    }                                                              \
    CUSPARSE_CHECK(cusparse_memory_status_);                       \
  } while (0)

constexpr int block_dim         = 256;
constexpr int walk_chunk        = 32;
constexpr int walk_linear_steps = 8;
constexpr int n_outputs         = 8;

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point begin)
{
  return std::chrono::duration<double>(clock_type::now() - begin).count();
}

// lap() ends an interval: the device is idle when the clock is read.
struct lap_t {
  clock_type::time_point last = clock_type::now();

  double lap()
  {
    CUDA_CHECK(cudaDeviceSynchronize());
    double seconds = seconds_since(last);
    last           = clock_type::now();
    return seconds;
  }
};

struct device_memory_t {
  std::vector<std::pair<void*, size_t>> blocks;
  size_t now  = 0;
  size_t peak = 0;

  template <typename T>
  T* allocate(size_t n)
  {
    size_t bytes = std::max<size_t>(n, 1) * sizeof(T);
    void* p;
    cudaError_t status = cudaMalloc(&p, bytes);
    if (status == cudaErrorMemoryAllocation) { throw out_of_device_memory_t(); }
    CUDA_CHECK(status);
    blocks.push_back({p, bytes});
    now += bytes;
    peak = std::max(peak, now);
    return (T*)p;
  }

  void release_last()
  {
    CUDA_CHECK(cudaFree(blocks.back().first));
    now -= blocks.back().second;
    blocks.pop_back();
  }

  void release_from(size_t first)
  {
    while (blocks.size() > first) {
      release_last();
    }
  }
};

struct session_t {
  cudaStream_t stream;
  cusparseHandle_t cusparse;
  device_memory_t memory;
  device_info_t info;
  size_t library_buffer_bytes = 0;

  char* staging        = nullptr;
  size_t staging_bytes = 0;
  cudaEvent_t array_done[n_outputs];

  char* chunk[2]     = {nullptr, nullptr};
  size_t chunk_bytes = 0;
  int next_chunk     = 0;
  cudaEvent_t chunk_done[2];
};

// The model and the derived arrays on the device. csc_start and csc_size have ncols + 1
// elements: csc_start[ncols] is the number of nonzeros, as the walks need it.
struct arrays_t {
  int32_t nrows;
  int32_t ncols;
  int64_t nnz;

  int64_t* row_start;
  int32_t* col;
  double* value;
  double* row_lower;
  double* row_upper;

  int64_t* csc_start;
  int32_t* csc_size;
  int32_t* csc_index;
  double* csc_value;
  int64_t* csc_twin;
  int64_t* csr_twin;
  int32_t* down_locks;
  int32_t* up_locks;
};

unsigned blocks_for(int64_t n)
{
  return (unsigned)std::max<int64_t>(1, (n + block_dim - 1) / block_dim);
}

unsigned walk_blocks_for(int64_t slots)
{
  return blocks_for((slots + walk_chunk - 1) / walk_chunk);
}

struct plus_i64_t {
  __host__ __device__ int64_t operator()(int64_t a, int64_t b) const { return a + b; }
};

// Last line in [lo, hi) whose start is <= slot; start[lo] <= slot < start[hi] on entry.
__device__ __forceinline__ int line_of(const int64_t* start, int lo, int hi, int64_t slot)
{
  while (hi - lo > 1) {
    int mid = lo + (hi - lo) / 2;
    if (start[mid] <= slot) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// The line of slot, which is `line` or a later one. A long run of empty lines is skipped by a
// search, not one by one.
__device__ __forceinline__ int next_line(const int64_t* start, int n_lines, int line, int64_t slot)
{
  for (int step = 0; start[line + 1] <= slot; ++step) {
    if (step == walk_linear_steps) { return line_of(start, line, n_lines, slot); }
    ++line;
  }
  return line;
}

// Thread t owns the slots [t * walk_chunk, (t + 1) * walk_chunk) and walks the offsets with
// them, so the cost does not depend on how the entries are distributed over the lines.
__global__ void entry_rows_kernel(const int64_t* row_start,
                                  int nrows,
                                  int64_t nnz,
                                  int32_t* entry_row)
{
  int64_t e = ((int64_t)blockIdx.x * block_dim + threadIdx.x) * walk_chunk;
  if (e >= nnz) { return; }
  int64_t end = e + walk_chunk < nnz ? e + walk_chunk : nnz;
  int row     = line_of(row_start, 0, nrows, e);
  for (; e < end; ++e) {
    row          = next_line(row_start, nrows, row, e);
    entry_row[e] = row;
  }
}

template <typename pos_t>
__global__ void positions_kernel(int64_t n, pos_t* position)
{
  int64_t e = (int64_t)blockIdx.x * block_dim + threadIdx.x;
  if (e < n) { position[e] = (pos_t)e; }
}

__global__ void ones_kernel(int64_t n, double* x)
{
  int64_t i = (int64_t)blockIdx.x * block_dim + threadIdx.x;
  if (i < n) { x[i] = 1.; }
}

__global__ void narrow_start_kernel(int n, const int64_t* start, int32_t* start32)
{
  int i = blockIdx.x * block_dim + threadIdx.x;
  if (i < n) { start32[i] = (int32_t)start[i]; }
}

__global__ void column_counts_kernel(int64_t nnz, const int32_t* col, int32_t* count)
{
  int64_t e = (int64_t)blockIdx.x * block_dim + threadIdx.x;
  if (e < nnz) { atomicAdd(&count[col[e]], 1); }
}

template <typename pos_t>
__global__ void scatter_positions_kernel(int64_t nnz,
                                         const int32_t* col,
                                         unsigned long long* cursor,
                                         pos_t* position)
{
  int64_t e = (int64_t)blockIdx.x * block_dim + threadIdx.x;
  if (e < nnz) { position[atomicAdd(&cursor[col[e]], 1ull)] = (pos_t)e; }
}

// perm[p] is the CSR position of the entry in CSC slot p.
template <typename pos_t>
__global__ void gather_kernel(int64_t nnz,
                              const pos_t* __restrict__ perm,
                              const int32_t* __restrict__ entry_row,
                              const double* __restrict__ value,
                              int32_t* __restrict__ csc_index,
                              double* __restrict__ csc_value,
                              int64_t* __restrict__ csc_twin,
                              int64_t* __restrict__ csr_twin)
{
  int64_t p = (int64_t)blockIdx.x * block_dim + threadIdx.x;
  if (p < nnz) {
    int64_t e    = perm[p];
    csc_index[p] = entry_row[e];
    csc_value[p] = value[e];
    csc_twin[p]  = e;
    csr_twin[e]  = p;
  }
}

// csc_twin holds, as doubles, the CSR positions that cusparseCsr2cscEx2 moved as its values.
__global__ void twins_from_positions_kernel(int64_t nnz,
                                            const double* __restrict__ value,
                                            double* __restrict__ csc_value,
                                            int64_t* csc_twin,
                                            int64_t* csr_twin)
{
  int64_t p = (int64_t)blockIdx.x * block_dim + threadIdx.x;
  if (p < nnz) {
    int64_t e    = (int64_t)__longlong_as_double(csc_twin[p]);
    csc_twin[p]  = e;
    csr_twin[e]  = p;
    csc_value[p] = value[e];
  }
}

__global__ void column_lines_kernel(int ncols,
                                    const int32_t* col_start32,
                                    int64_t* csc_start,
                                    int32_t* csc_size)
{
  int j = blockIdx.x * block_dim + threadIdx.x;
  if (j > ncols) { return; }
  csc_start[j] = col_start32[j];
  if (j < ncols) { csc_size[j] = col_start32[j + 1] - col_start32[j]; }
}

// Bit 0: the lower side is finite; bit 1: the upper side is.
__global__ void finite_sides_kernel(int nrows,
                                    const double* row_lower,
                                    const double* row_upper,
                                    uint8_t* finite_sides)
{
  int i = blockIdx.x * block_dim + threadIdx.x;
  if (i < nrows) { finite_sides[i] = (row_lower[i] > -inf) | ((row_upper[i] < inf) << 1); }
}

// The walk of entry_rows_kernel over the columns. A thread adds what its slots give a column
// once, when it leaves the column: integer sums, the same in any order.
__global__ void locks_kernel(const int64_t* csc_start,
                             int ncols,
                             int64_t nnz,
                             const int32_t* csc_index,
                             const double* csc_value,
                             const uint8_t* finite_sides,
                             int32_t* down_locks,
                             int32_t* up_locks)
{
  int64_t p = ((int64_t)blockIdx.x * block_dim + threadIdx.x) * walk_chunk;
  if (p >= nnz) { return; }
  int64_t end = p + walk_chunk < nnz ? p + walk_chunk : nnz;
  int col     = line_of(csc_start, 0, ncols, p);
  int down    = 0;
  int up      = 0;
  for (; p < end; ++p) {
    if (csc_start[col + 1] <= p) {
      atomicAdd(&down_locks[col], down);
      atomicAdd(&up_locks[col], up);
      down = 0;
      up   = 0;
      col  = next_line(csc_start, ncols, col, p);
    }
    int sides        = finite_sides[csc_index[p]];
    int lower_finite = sides & 1;
    int upper_finite = sides >> 1;
    bool positive    = csc_value[p] > 0;
    up += positive ? upper_finite : lower_finite;
    down += positive ? lower_finite : upper_finite;
  }
  atomicAdd(&down_locks[col], down);
  atomicAdd(&up_locks[col], up);
}

// One thread, after the build.
__global__ void fault_kernel(fault_t fault,
                             int ncols,
                             const int64_t* csc_start,
                             int32_t* csc_index,
                             double* csc_value,
                             int64_t* csc_twin,
                             int64_t* csr_twin,
                             int32_t* up_locks)
{
  for (int j = 0; j < ncols; ++j) {
    int64_t p = csc_start[j];
    if (fault == fault_t::reverse_column && csc_start[j + 1] - p >= 2) {
      int32_t index             = csc_index[p];
      double value              = csc_value[p];
      int64_t twin              = csc_twin[p];
      csc_index[p]              = csc_index[p + 1];
      csc_value[p]              = csc_value[p + 1];
      csc_twin[p]               = csc_twin[p + 1];
      csc_index[p + 1]          = index;
      csc_value[p + 1]          = value;
      csc_twin[p + 1]           = twin;
      csr_twin[csc_twin[p]]     = p;
      csr_twin[csc_twin[p + 1]] = p + 1;
      return;
    }
    if (fault == fault_t::drop_lock && up_locks[j] > 0) {
      --up_locks[j];
      return;
    }
  }
}

// Runs a CUB algorithm with temporary storage that lives until the temporaries of the build are
// released.
template <typename F>
void cub_call(session_t& s, F&& f)
{
  size_t bytes = 0;
  CUDA_CHECK(f(nullptr, bytes));
  void* temp = s.memory.allocate<char>(bytes);
  CUDA_CHECK(f(temp, bytes));
  s.library_buffer_bytes = std::max(s.library_buffer_bytes, bytes);
}

void transpose_cusparse(session_t& s, const arrays_t& a)
{
  gprs_assert(a.nnz <= INT32_MAX);
  int32_t* row_start32 = s.memory.allocate<int32_t>((size_t)a.nrows + 1);
  int32_t* col_start32 = s.memory.allocate<int32_t>((size_t)a.ncols + 1);
  // The positions go in through the memory of csr_twin and come out through that of csc_twin:
  // both are written only after the conversion, and a double holds a position exactly.
  double* position     = (double*)a.csr_twin;
  double* csc_position = (double*)a.csc_twin;
  narrow_start_kernel<<<blocks_for(a.nrows + 1), block_dim, 0, s.stream>>>(
    a.nrows + 1, a.row_start, row_start32);
  positions_kernel<double><<<blocks_for(a.nnz), block_dim, 0, s.stream>>>(a.nnz, position);

  size_t buffer_bytes = 0;
  CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(s.cusparse,
                                               a.nrows,
                                               a.ncols,
                                               (int)a.nnz,
                                               position,
                                               row_start32,
                                               a.col,
                                               csc_position,
                                               col_start32,
                                               a.csc_index,
                                               CUDA_R_64F,
                                               CUSPARSE_ACTION_NUMERIC,
                                               CUSPARSE_INDEX_BASE_ZERO,
                                               CUSPARSE_CSR2CSC_ALG1,
                                               &buffer_bytes));
  void* buffer           = s.memory.allocate<char>(buffer_bytes);
  s.library_buffer_bytes = buffer_bytes;
  CUSPARSE_CHECK_MEMORY(cusparseCsr2cscEx2(s.cusparse,
                                           a.nrows,
                                           a.ncols,
                                           (int)a.nnz,
                                           position,
                                           row_start32,
                                           a.col,
                                           csc_position,
                                           col_start32,
                                           a.csc_index,
                                           CUDA_R_64F,
                                           CUSPARSE_ACTION_NUMERIC,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUSPARSE_CSR2CSC_ALG1,
                                           buffer));
  twins_from_positions_kernel<<<blocks_for(a.nnz), block_dim, 0, s.stream>>>(
    a.nnz, a.value, a.csc_value, a.csc_twin, a.csr_twin);
  column_lines_kernel<<<blocks_for(a.ncols + 1), block_dim, 0, s.stream>>>(
    a.ncols, col_start32, a.csc_start, a.csc_size);
}

void count_columns(session_t& s, const arrays_t& a)
{
  CUDA_CHECK(cudaMemsetAsync(a.csc_size, 0, ((size_t)a.ncols + 1) * sizeof(int32_t), s.stream));
  column_counts_kernel<<<blocks_for(a.nnz), block_dim, 0, s.stream>>>(a.nnz, a.col, a.csc_size);
  cub_call(s, [&](void* temp, size_t& bytes) {
    return cub::DeviceScan::ExclusiveScan(
      temp, bytes, a.csc_size, a.csc_start, plus_i64_t{}, (int64_t)0, a.ncols + 1, s.stream);
  });
}

template <typename pos_t>
void transpose_sort(session_t& s, const arrays_t& a, transpose_t transpose)
{
  count_columns(s, a);
  pos_t* perm = s.memory.allocate<pos_t>(a.nnz);
  if (transpose == transpose_t::radix_sort) {
    pos_t* position     = s.memory.allocate<pos_t>(a.nnz);
    int32_t* sorted_col = s.memory.allocate<int32_t>(a.nnz);
    positions_kernel<pos_t><<<blocks_for(a.nnz), block_dim, 0, s.stream>>>(a.nnz, position);
    int end_bit = a.ncols > 1 ? 32 - __builtin_clz(a.ncols - 1) : 1;
    cub_call(s, [&](void* temp, size_t& bytes) {
      return cub::DeviceRadixSort::SortPairs(
        temp, bytes, a.col, sorted_col, position, perm, a.nnz, 0, end_bit, s.stream);
    });
  } else {
    // cub::DeviceSegmentedSort takes int sizes before CCCL 3.
    gprs_assert(CUB_VERSION >= 300000 || a.nnz <= INT32_MAX);
    unsigned long long* cursor = s.memory.allocate<unsigned long long>(a.ncols);
    pos_t* unsorted            = s.memory.allocate<pos_t>(a.nnz);
    CUDA_CHECK(cudaMemcpyAsync(
      cursor, a.csc_start, (size_t)a.ncols * sizeof(int64_t), cudaMemcpyDeviceToDevice, s.stream));
    scatter_positions_kernel<pos_t>
      <<<blocks_for(a.nnz), block_dim, 0, s.stream>>>(a.nnz, a.col, cursor, unsorted);
    cub_call(s, [&](void* temp, size_t& bytes) {
      return cub::DeviceSegmentedSort::SortKeys(temp,
                                                bytes,
                                                unsorted,
                                                perm,
                                                a.nnz,
                                                (int64_t)a.ncols,
                                                a.csc_start,
                                                a.csc_start + 1,
                                                s.stream);
    });
  }
  int32_t* entry_row = s.memory.allocate<int32_t>(a.nnz);
  entry_rows_kernel<<<walk_blocks_for(a.nnz), block_dim, 0, s.stream>>>(
    a.row_start, a.nrows, a.nnz, entry_row);
  gather_kernel<pos_t><<<blocks_for(a.nnz), block_dim, 0, s.stream>>>(
    a.nnz, perm, entry_row, a.value, a.csc_index, a.csc_value, a.csc_twin, a.csr_twin);
}

void count_locks(session_t& s, const arrays_t& a)
{
  uint8_t* finite_sides = s.memory.allocate<uint8_t>(a.nrows);
  finite_sides_kernel<<<blocks_for(a.nrows), block_dim, 0, s.stream>>>(
    a.nrows, a.row_lower, a.row_upper, finite_sides);
  CUDA_CHECK(cudaMemsetAsync(a.down_locks, 0, (size_t)a.ncols * sizeof(int32_t), s.stream));
  CUDA_CHECK(cudaMemsetAsync(a.up_locks, 0, (size_t)a.ncols * sizeof(int32_t), s.stream));
  locks_kernel<<<walk_blocks_for(a.nnz), block_dim, 0, s.stream>>>(
    a.csc_start, a.ncols, a.nnz, a.csc_index, a.csc_value, finite_sides, a.down_locks, a.up_locks);
}

template <typename F>
void for_each_input(const model_t& model, const arrays_t& a, F&& f)
{
  f(model.row_start, a.row_start);
  f(model.col_index, a.col);
  f(model.value, a.value);
  f(model.row_lower, a.row_lower);
  f(model.row_upper, a.row_upper);
}

template <typename F>
void for_each_output(const arrays_t& a, state_arrays_t& out, F&& f)
{
  size_t ncols = a.ncols, nnz = a.nnz;
  f(a.csc_start, ncols, out.csc_start);
  f(a.csc_size, ncols, out.csc_size);
  f(a.csc_index, nnz, out.csc_index);
  f(a.csc_value, nnz, out.csc_value);
  f(a.csc_twin, nnz, out.csc_twin);
  f(a.csr_twin, nnz, out.csr_twin);
  f(a.down_locks, ncols, out.down_locks);
  f(a.up_locks, ncols, out.up_locks);
}

size_t staged_size(size_t bytes) { return (bytes + 7) & ~(size_t)7; }

template <typename T>
void upload(
  session_t& s, transfer_t transfer, const std::vector<T>& host, T* device, size_t& staged)
{
  size_t bytes     = host.size() * sizeof(T);
  const char* from = (const char*)host.data();
  if (transfer == transfer_t::pageable) {
    CUDA_CHECK(cudaMemcpyAsync(device, from, bytes, cudaMemcpyHostToDevice, s.stream));
  } else if (transfer == transfer_t::pinned) {
    gprs_assert(staged + bytes <= s.staging_bytes);
    memcpy(s.staging + staged, from, bytes);
    CUDA_CHECK(
      cudaMemcpyAsync(device, s.staging + staged, bytes, cudaMemcpyHostToDevice, s.stream));
    staged += staged_size(bytes);
  } else {
    // A chunk is refilled only after its last transfer has finished.
    for (size_t done = 0; done < bytes; done += s.chunk_bytes) {
      int b      = s.next_chunk;
      size_t len = std::min(s.chunk_bytes, bytes - done);
      CUDA_CHECK(cudaEventSynchronize(s.chunk_done[b]));
      memcpy(s.chunk[b], from + done, len);
      CUDA_CHECK(
        cudaMemcpyAsync((char*)device + done, s.chunk[b], len, cudaMemcpyHostToDevice, s.stream));
      CUDA_CHECK(cudaEventRecord(s.chunk_done[b], s.stream));
      s.next_chunk = b ^ 1;
    }
  }
}

// The transfer of chunk c + 1 runs while the host appends chunk c to the vector. The vector is
// sized by reserve and filled by insert, so its memory is written once.
template <typename T>
void download_chunked(session_t& s, const T* device, size_t n, std::vector<T>& host)
{
  size_t bytes = n * sizeof(T);
  auto issue   = [&](size_t offset, int b) {
    size_t len = std::min(s.chunk_bytes, bytes - offset);
    CUDA_CHECK(cudaMemcpyAsync(
      s.chunk[b], (const char*)device + offset, len, cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaEventRecord(s.chunk_done[b], s.stream));
  };
  host.clear();
  host.reserve(n);
  if (bytes > 0) { issue(0, 0); }
  int b = 0;
  for (size_t offset = 0; offset < bytes; offset += s.chunk_bytes, b ^= 1) {
    if (offset + s.chunk_bytes < bytes) { issue(offset + s.chunk_bytes, b ^ 1); }
    CUDA_CHECK(cudaEventSynchronize(s.chunk_done[b]));
    const T* from = (const T*)s.chunk[b];
    host.insert(host.end(), from, from + std::min(s.chunk_bytes, bytes - offset) / sizeof(T));
  }
}

void download(session_t& s,
              transfer_t transfer,
              const arrays_t& a,
              state_arrays_t& out,
              build_timing_t& timing,
              void** untouched)
{
  if (untouched) {
    int k = 0;
    for_each_output(a, out, [&](const auto* device, size_t n, auto&) {
      size_t bytes = n * sizeof(*device);
      untouched[k] = malloc(bytes);
      CUDA_CHECK(cudaMemcpyAsync(untouched[k++], device, bytes, cudaMemcpyDeviceToHost, s.stream));
    });
  } else if (transfer == transfer_t::pageable) {
    auto begin = clock_type::now();
    for_each_output(a, out, [&](const auto*, size_t n, auto& host) { host.resize(n); });
    timing.download_resize_seconds = seconds_since(begin);
    for_each_output(a, out, [&](const auto* device, size_t, auto& host) {
      CUDA_CHECK(cudaMemcpyAsync(
        host.data(), device, host.size() * sizeof(host[0]), cudaMemcpyDeviceToHost, s.stream));
    });
  } else if (transfer == transfer_t::pinned) {
    size_t staged = 0;
    int k         = 0;
    for_each_output(a, out, [&](const auto* device, size_t n, auto&) {
      size_t bytes = n * sizeof(*device);
      gprs_assert(staged + bytes <= s.staging_bytes);
      CUDA_CHECK(
        cudaMemcpyAsync(s.staging + staged, device, bytes, cudaMemcpyDeviceToHost, s.stream));
      CUDA_CHECK(cudaEventRecord(s.array_done[k++], s.stream));
      staged += staged_size(bytes);
    });
    staged = 0;
    k      = 0;
    for_each_output(a, out, [&](const auto* device, size_t n, auto& host) {
      CUDA_CHECK(cudaEventSynchronize(s.array_done[k++]));
      auto* from = (decltype(device))(s.staging + staged);
      host.assign(from, from + n);
      staged += staged_size(n * sizeof(*device));
    });
  } else {
    for_each_output(a, out, [&](const auto* device, size_t n, auto& host) {
      download_chunked(s, device, n, host);
    });
  }
}

}  // namespace

struct state_builder_t::impl_t {
  session_t session;
};

const char* name(transpose_t transpose)
{
  switch (transpose) {
    case transpose_t::cusparse: return "cusparse";
    case transpose_t::radix_sort: return "radix_sort";
    case transpose_t::scatter_sort: return "scatter_sort";
  }
  return "";
}

const char* name(transfer_t transfer)
{
  switch (transfer) {
    case transfer_t::pageable: return "pageable";
    case transfer_t::pinned: return "pinned";
    case transfer_t::pinned_chunked: return "pinned_chunked";
  }
  return "";
}

state_builder_t::state_builder_t() : impl(new impl_t)
{
  session_t& s = impl->session;
  auto begin   = clock_type::now();
  CUDA_CHECK(cudaStreamCreate(&s.stream));
  CUDA_CHECK(cudaStreamSynchronize(s.stream));
  s.info.context_seconds = seconds_since(begin);

  begin = clock_type::now();
  CUSPARSE_CHECK(cusparseCreate(&s.cusparse));
  CUSPARSE_CHECK(cusparseSetStream(s.cusparse, s.stream));
  s.info.cusparse_seconds = seconds_since(begin);

  for (cudaEvent_t& event : s.array_done) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  for (cudaEvent_t& event : s.chunk_done) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }

  int device;
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  s.info.gpu_name         = properties.name;
  s.info.gpu_memory_bytes = properties.totalGlobalMem;
  CUDA_CHECK(cudaRuntimeGetVersion(&s.info.cuda_runtime_version));
  CUSPARSE_CHECK(cusparseGetVersion(s.cusparse, &s.info.cusparse_version));
  s.info.cub_version = CUB_VERSION;
}

state_builder_t::~state_builder_t()
{
  session_t& s = impl->session;
  release_staging();
  for (cudaEvent_t event : s.array_done) {
    CUDA_CHECK(cudaEventDestroy(event));
  }
  for (cudaEvent_t event : s.chunk_done) {
    CUDA_CHECK(cudaEventDestroy(event));
  }
  CUSPARSE_CHECK(cusparseDestroy(s.cusparse));
  CUDA_CHECK(cudaStreamDestroy(s.stream));
  delete impl;
}

const device_info_t& state_builder_t::info() const { return impl->session.info; }

size_t state_builder_t::device_used_bytes() const
{
  size_t free_bytes, total_bytes;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return total_bytes - free_bytes;
}

double state_builder_t::reserve_staging(const model_t& model,
                                        transfer_t transfer,
                                        size_t chunk_bytes,
                                        size_t& bytes)
{
  session_t& s = impl->session;
  auto begin   = clock_type::now();
  bytes        = 0;
  if (transfer == transfer_t::pinned) {
    gprs_assert(!s.staging);
    size_t nrows = model.nrows, ncols = model.ncols, nnz = model.nnz();
    size_t up       = (nrows + 1) * 8 + nnz * 12 + nrows * 16 + 5 * 8;
    size_t down     = ncols * 20 + nnz * 28 + n_outputs * 8;
    s.staging_bytes = std::max(up, down);
    CUDA_CHECK(cudaHostAlloc((void**)&s.staging, s.staging_bytes, cudaHostAllocDefault));
    bytes = s.staging_bytes;
  } else if (transfer == transfer_t::pinned_chunked) {
    gprs_assert(!s.chunk[0] && chunk_bytes % 8 == 0);
    s.chunk_bytes = chunk_bytes;
    for (char*& chunk : s.chunk) {
      CUDA_CHECK(cudaHostAlloc((void**)&chunk, chunk_bytes, cudaHostAllocDefault));
    }
    bytes = 2 * chunk_bytes;
  }
  return seconds_since(begin);
}

void state_builder_t::release_staging()
{
  session_t& s = impl->session;
  if (s.staging) { CUDA_CHECK(cudaFreeHost(s.staging)); }
  s.staging       = nullptr;
  s.staging_bytes = 0;
  for (char*& chunk : s.chunk) {
    if (chunk) { CUDA_CHECK(cudaFreeHost(chunk)); }
    chunk = nullptr;
  }
  s.chunk_bytes = 0;
}

bool state_builder_t::build(const model_t& model,
                            const build_options_t& options,
                            state_arrays_t& out,
                            build_timing_t& timing)
{
  session_t& s         = impl->session;
  device_memory_t& mem = s.memory;
  gprs_assert(mem.blocks.empty());
  mem.peak               = 0;
  s.library_buffer_bytes = 0;
  timing                 = build_timing_t();

  arrays_t a;
  a.nrows      = model.nrows;
  a.ncols      = model.ncols;
  a.nnz        = model.nnz();
  size_t nrows = a.nrows, ncols = a.ncols, nnz = a.nnz;
  bool wide = options.wide_positions || nnz > UINT32_MAX;

  lap_t lap;
  try {
    a.row_start  = mem.allocate<int64_t>(nrows + 1);
    a.col        = mem.allocate<int32_t>(nnz);
    a.value      = mem.allocate<double>(nnz);
    a.row_lower  = mem.allocate<double>(nrows);
    a.row_upper  = mem.allocate<double>(nrows);
    a.csc_start  = mem.allocate<int64_t>(ncols + 1);
    a.csc_size   = mem.allocate<int32_t>(ncols + 1);
    a.csc_index  = mem.allocate<int32_t>(nnz);
    a.csc_value  = mem.allocate<double>(nnz);
    a.csc_twin   = mem.allocate<int64_t>(nnz);
    a.csr_twin   = mem.allocate<int64_t>(nnz);
    a.down_locks = mem.allocate<int32_t>(ncols);
    a.up_locks   = mem.allocate<int32_t>(ncols);
    if (options.poison) {
      CUDA_CHECK(cudaMemsetAsync(a.csc_start, 0xff, (ncols + 1) * sizeof(int64_t), s.stream));
      CUDA_CHECK(cudaMemsetAsync(a.csc_size, 0xff, (ncols + 1) * sizeof(int32_t), s.stream));
      CUDA_CHECK(cudaMemsetAsync(a.csc_index, 0xff, nnz * sizeof(int32_t), s.stream));
      CUDA_CHECK(cudaMemsetAsync(a.csc_value, 0xff, nnz * sizeof(double), s.stream));
      CUDA_CHECK(cudaMemsetAsync(a.csc_twin, 0xff, nnz * sizeof(int64_t), s.stream));
      CUDA_CHECK(cudaMemsetAsync(a.csr_twin, 0xff, nnz * sizeof(int64_t), s.stream));
      CUDA_CHECK(cudaMemsetAsync(a.down_locks, 0xff, ncols * sizeof(int32_t), s.stream));
      CUDA_CHECK(cudaMemsetAsync(a.up_locks, 0xff, ncols * sizeof(int32_t), s.stream));
    }
    timing.alloc_seconds = lap.lap();

    size_t staged = 0;
    for_each_input(model, a, [&](const auto& host, auto* device) {
      upload(s, options.transfer, host, device, staged);
    });
    timing.upload_seconds = lap.lap();

    size_t fixed_blocks = mem.blocks.size();
    if (options.transpose == transpose_t::cusparse) {
      transpose_cusparse(s, a);
      timing.position_bits = 32;
    } else if (wide) {
      transpose_sort<uint64_t>(s, a, options.transpose);
      timing.position_bits = 64;
    } else {
      transpose_sort<uint32_t>(s, a, options.transpose);
      timing.position_bits = 32;
    }
    count_locks(s, a);
    if (options.fault != fault_t::none) {
      fault_kernel<<<1, 1, 0, s.stream>>>(options.fault,
                                          a.ncols,
                                          a.csc_start,
                                          a.csc_index,
                                          a.csc_value,
                                          a.csc_twin,
                                          a.csr_twin,
                                          a.up_locks);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    mem.release_from(fixed_blocks);
    timing.library_buffer_bytes = s.library_buffer_bytes;
    timing.device_seconds       = lap.lap();
  } catch (const out_of_device_memory_t&) {
    // The failed allocation stays the last error of the thread until it is read.
    cudaGetLastError();
    CUDA_CHECK(cudaDeviceSynchronize());
    mem.release_from(0);
    return false;
  }

  gprs_assert(!options.untouched_destination || options.transfer == transfer_t::pageable);
  void* untouched[n_outputs];
  download(
    s, options.transfer, a, out, timing, options.untouched_destination ? untouched : nullptr);
  timing.download_seconds = lap.lap();

  mem.release_from(0);
  timing.free_seconds      = lap.lap();
  timing.device_bytes_peak = mem.peak;

  if (options.untouched_destination) {
    int k = 0;
    for_each_output(a, out, [&](const auto* device, size_t n, auto& host) {
      auto* from = (decltype(device))untouched[k];
      host.assign(from, from + n);
      free(untouched[k++]);
    });
  }
  return true;
}

void state_builder_t::spmv_ones(const model_t& model,
                                int repetitions,
                                bool deterministic,
                                std::vector<double>& y,
                                spmv_timing_t& timing)
{
  session_t& s         = impl->session;
  device_memory_t& mem = s.memory;
  gprs_assert(mem.blocks.empty() && model.nnz() <= INT32_MAX);
  size_t nrows = model.nrows, ncols = model.ncols, nnz = model.nnz();
  timing = spmv_timing_t();

  lap_t lap;
  int64_t* row_start   = mem.allocate<int64_t>(nrows + 1);
  int32_t* row_start32 = mem.allocate<int32_t>(nrows + 1);
  int32_t* col         = mem.allocate<int32_t>(nnz);
  double* value        = mem.allocate<double>(nnz);
  double* x            = mem.allocate<double>(ncols);
  double* product      = mem.allocate<double>(nrows);
  CUDA_CHECK(cudaMemcpyAsync(row_start,
                             model.row_start.data(),
                             (nrows + 1) * sizeof(int64_t),
                             cudaMemcpyHostToDevice,
                             s.stream));
  CUDA_CHECK(cudaMemcpyAsync(
    col, model.col_index.data(), nnz * sizeof(int32_t), cudaMemcpyHostToDevice, s.stream));
  CUDA_CHECK(cudaMemcpyAsync(
    value, model.value.data(), nnz * sizeof(double), cudaMemcpyHostToDevice, s.stream));
  narrow_start_kernel<<<blocks_for(nrows + 1), block_dim, 0, s.stream>>>(
    model.nrows + 1, row_start, row_start32);
  ones_kernel<<<blocks_for(ncols), block_dim, 0, s.stream>>>(ncols, x);
  timing.upload_seconds = lap.lap();

  cusparseSpMatDescr_t matrix;
  cusparseDnVecDescr_t vec_x, vec_y;
  CUSPARSE_CHECK(cusparseCreateCsr(&matrix,
                                   nrows,
                                   ncols,
                                   nnz,
                                   row_start32,
                                   col,
                                   value,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, ncols, x, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, nrows, product, CUDA_R_64F));
  double alpha              = 1.;
  double beta               = 0.;
  cusparseSpMVAlg_t method  = deterministic ? CUSPARSE_SPMV_CSR_ALG2 : CUSPARSE_SPMV_ALG_DEFAULT;
  cusparseOperation_t plain = CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(s.cusparse,
                                         plain,
                                         &alpha,
                                         matrix,
                                         vec_x,
                                         &beta,
                                         vec_y,
                                         CUDA_R_64F,
                                         method,
                                         &timing.buffer_bytes));
  void* buffer = mem.allocate<char>(timing.buffer_bytes);
  CUSPARSE_CHECK(cusparseSpMV_preprocess(
    s.cusparse, plain, &alpha, matrix, vec_x, &beta, vec_y, CUDA_R_64F, method, buffer));
  timing.setup_seconds = lap.lap();

  timing.repetitions = repetitions;
  for (int k = 0; k < repetitions; ++k) {
    CUSPARSE_CHECK(cusparseSpMV(
      s.cusparse, plain, &alpha, matrix, vec_x, &beta, vec_y, CUDA_R_64F, method, buffer));
    double seconds = lap.lap();
    if (k == 0) { timing.first_seconds = seconds; }
    if (k == 0 || seconds < timing.min_seconds) { timing.min_seconds = seconds; }
  }

  y.resize(nrows);
  CUDA_CHECK(
    cudaMemcpyAsync(y.data(), product, nrows * sizeof(double), cudaMemcpyDeviceToHost, s.stream));
  timing.download_seconds = lap.lap();

  CUSPARSE_CHECK(cusparseDestroySpMat(matrix));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
  mem.release_from(0);
}

}  // namespace device
}  // namespace gprs
