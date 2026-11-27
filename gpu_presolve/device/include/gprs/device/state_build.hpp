/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <gprs/state.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gprs {
namespace device {

// How the entries are brought into column order. The host has the entries of a column in
// increasing CSR position, which is increasing row order; radix_sort and scatter_sort give that
// order by construction, cusparse only as far as the library keeps doing what it does today.
enum class transpose_t : int {
  // cusparseCsr2cscEx2 with the entry positions as its values. 32-bit sizes. cusparse.h says
  // nothing about the order inside a column: it has been the order of a stable sort in every
  // run (also with unsorted rows and duplicate entries), which is observed, not promised. The
  // first build of a process pays about 20 ms for the handle and the kernels of the library.
  cusparse = 0,
  // cub::DeviceRadixSort of the entry positions by column, which is documented as stable.
  radix_sort = 1,
  // One integer cursor per column hands out the slots (order inside a column depends on thread
  // timing), then cub::DeviceSegmentedSort orders the positions inside each column. With a
  // CCCL before 3 the sort takes int sizes, so the nonzeros must fit an int.
  scatter_sort = 2
};

// How an array moves between a std::vector and the device.
enum class transfer_t : int {
  // cudaMemcpy on the memory of the vector.
  pageable = 0,
  // Through one pinned buffer that holds all arrays of a direction.
  pinned = 1,
  // Through two pinned chunks: the host copies one while the device transfers the other.
  pinned_chunked = 2
};

// A wrong result on purpose, made on the device after the build, to see that a comparison
// with the host notices it.
enum class fault_t : int {
  none = 0,
  // The first two entries of the first column that has two change places, with their twins: a
  // column-wise copy of the same matrix in another order.
  reverse_column = 1,
  // The first column with an up-lock has one less.
  drop_lock = 2
};

const char* name(transpose_t transpose);
const char* name(transfer_t transfer);

// What the constructor of state_t derives from the matrix and the sides of the rows, in the
// layout of the state: csc.start, csc.size, csc.index, csc.value, csc.twin, csr.twin and the
// locks.
struct state_arrays_t {
  std::vector<int64_t> csc_start;
  std::vector<int32_t> csc_size;
  std::vector<int32_t> csc_index;
  std::vector<double> csc_value;
  std::vector<int64_t> csc_twin;
  std::vector<int64_t> csr_twin;
  std::vector<int32_t> down_locks;
  std::vector<int32_t> up_locks;
};

struct build_options_t {
  transpose_t transpose = transpose_t::radix_sort;
  transfer_t transfer   = transfer_t::pageable;
  // 64-bit positions in the two sorts even when the nonzeros fit 32 bits.
  bool wide_positions = false;
  // Fills the device outputs with 0xff before anything else, so that an element a variant does
  // not write cannot hold the right value from an earlier run. For checks; it costs time.
  bool poison = false;
  // Measurement only, with transfer_t::pageable: the download goes into memory that is
  // allocated and not yet touched, as a vector type without value initialisation would offer
  // it, and the vectors are filled from it after the last timed interval.
  bool untouched_destination = false;
  fault_t fault              = fault_t::none;
};

// Consecutive intervals of one build; each ends after cudaDeviceSynchronize.
struct build_timing_t {
  double alloc_seconds    = 0;  // cudaMalloc of the inputs, the outputs and the fixed temporaries
  double upload_seconds   = 0;
  double device_seconds   = 0;  // library buffers are allocated and all temporaries freed in here
  double download_seconds = 0;
  double free_seconds     = 0;
  // Part of download_seconds: sizing the vectors before a pageable copy (the other transfers
  // fill a vector as they size it).
  double download_resize_seconds = 0;
  // Largest sum of the device allocations made through this library during the build.
  size_t device_bytes_peak = 0;
  // Buffer of cusparseCsr2cscEx2, or the temporary storage of the sort.
  size_t library_buffer_bytes = 0;
  int position_bits           = 0;

  double total_seconds() const
  {
    return alloc_seconds + upload_seconds + device_seconds + download_seconds + free_seconds;
  }
};

struct spmv_timing_t {
  double upload_seconds   = 0;
  double setup_seconds    = 0;  // descriptors, buffer, cusparseSpMV_preprocess
  double first_seconds    = 0;
  double min_seconds      = 0;
  double download_seconds = 0;
  size_t buffer_bytes     = 0;
  int repetitions         = 0;
};

struct device_info_t {
  std::string gpu_name;
  size_t gpu_memory_bytes  = 0;
  int cuda_runtime_version = 0;
  int cusparse_version     = 0;
  int cub_version          = 0;
  double context_seconds   = 0;  // first CUDA call to a usable stream
  double cusparse_seconds  = 0;  // cusparseCreate
};

class state_builder_t {
 public:
  state_builder_t();
  ~state_builder_t();
  state_builder_t(const state_builder_t&)            = delete;
  state_builder_t& operator=(const state_builder_t&) = delete;

  // Pinned memory for the given transfer of this model; a build with that transfer needs it.
  // Returns the seconds the allocation took and the bytes in `bytes`.
  // chunk_bytes is the size of each of the two chunks of transfer_t::pinned_chunked.
  double reserve_staging(const model_t& model,
                         transfer_t transfer,
                         size_t chunk_bytes,
                         size_t& bytes);
  void release_staging();

  // The model must be well_formed; with transpose_t::cusparse its nonzeros must fit an int.
  // False when the device has not the memory for the build: nothing stays allocated, `out` and
  // `timing` hold nothing of use, and the caller builds on the host.
  bool build(const model_t& model,
             const build_options_t& options,
             state_arrays_t& out,
             build_timing_t& timing);

  // y = A * x for x = (1, ..., 1) with cusparseSpMV on the CSR, CUSPARSE_SPMV_CSR_ALG2 when
  // deterministic and the default algorithm otherwise. Plain double sums in the order of the
  // library: not the compensated sums of the engine and no replacement for them. The nonzeros
  // must fit an int.
  void spmv_ones(const model_t& model,
                 int repetitions,
                 bool deterministic,
                 std::vector<double>& y,
                 spmv_timing_t& timing);

  // Device memory in use on the GPU, by every process. Callable from another thread.
  size_t device_used_bytes() const;

  const device_info_t& info() const;

 private:
  struct impl_t;
  impl_t* impl;
};

}  // namespace device
}  // namespace gprs
