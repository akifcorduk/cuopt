/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The host side of the comparison, shared by state_build_bench and state_build_test: the loop of
// the constructor of state_t on the model arrays, and the bit comparison of the eight arrays.

#pragma once

#include <gprs/device/state_build.hpp>
#include <gprs/state.hpp>

#include <cstring>
#include <vector>

namespace gprs {
namespace device {

// The statements of the constructor of state_t that the device replaces, on the model arrays.
// They do not rely on the order inside a row, so this is also the reference for a model whose
// rows are not sorted.
inline void host_derived_arrays(const model_t& model, state_arrays_t& out)
{
  int32_t nrows = model.nrows, ncols = model.ncols;
  int64_t nnz = model.nnz();
  out.csr_twin.resize(nnz);

  out.csc_size.assign(ncols, 0);
  for (int64_t e = 0; e < nnz; ++e) {
    ++out.csc_size[model.col_index[e]];
  }
  out.csc_start.resize(ncols);
  int64_t next_start = 0;
  for (int32_t j = 0; j < ncols; ++j) {
    out.csc_start[j] = next_start;
    next_start += out.csc_size[j];
  }
  out.csc_index.resize(nnz);
  out.csc_value.resize(nnz);
  out.csc_twin.resize(nnz);

  out.down_locks.assign(ncols, 0);
  out.up_locks.assign(ncols, 0);
  std::vector<int64_t> fill(out.csc_start);
  for (int32_t i = 0; i < nrows; ++i) {
    int32_t lower_finite = model.row_lower[i] > -inf;
    int32_t upper_finite = model.row_upper[i] < inf;
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      int32_t j        = model.col_index[e];
      double a         = model.value[e];
      int64_t p        = fill[j]++;
      out.csc_index[p] = i;
      out.csc_value[p] = a;
      out.csc_twin[p]  = e;
      out.csr_twin[e]  = p;
      bool positive    = a > 0;
      out.up_locks[j] += positive ? upper_finite : lower_finite;
      out.down_locks[j] += positive ? lower_finite : upper_finite;
    }
  }
}

struct reference_t {
  const std::vector<int64_t>* csc_start;
  const std::vector<int32_t>* csc_size;
  const std::vector<int32_t>* csc_index;
  const std::vector<double>* csc_value;
  const std::vector<int64_t>* csc_twin;
  const std::vector<int64_t>* csr_twin;
  const std::vector<int32_t>* down_locks;
  const std::vector<int32_t>* up_locks;
};

inline reference_t reference_of(const state_t& st)
{
  return {&st.csc.start,
          &st.csc.size,
          &st.csc.index,
          &st.csc.value,
          &st.csc.twin,
          &st.csr.twin,
          &st.down_locks,
          &st.up_locks};
}

inline reference_t reference_of(const state_arrays_t& a)
{
  return {&a.csc_start,
          &a.csc_size,
          &a.csc_index,
          &a.csc_value,
          &a.csc_twin,
          &a.csr_twin,
          &a.down_locks,
          &a.up_locks};
}

// different is the number of elements whose bits differ, or -1 when the sizes differ.
struct array_check_t {
  const char* name;
  int64_t different;
  int64_t first;
};

template <typename T>
array_check_t check_array(const char* name, const std::vector<T>& got, const std::vector<T>& want)
{
  array_check_t check{name, 0, -1};
  if (got.size() != want.size()) {
    check.different = -1;
    return check;
  }
  if (got.empty() || memcmp(got.data(), want.data(), got.size() * sizeof(T)) == 0) { return check; }
  for (size_t k = 0; k < got.size(); ++k) {
    if (memcmp(&got[k], &want[k], sizeof(T)) != 0) {
      if (check.first < 0) { check.first = k; }
      ++check.different;
    }
  }
  return check;
}

inline std::vector<array_check_t> check_arrays(const state_arrays_t& got, const reference_t& want)
{
  return {check_array("csc_start", got.csc_start, *want.csc_start),
          check_array("csc_size", got.csc_size, *want.csc_size),
          check_array("csc_index", got.csc_index, *want.csc_index),
          check_array("csc_value", got.csc_value, *want.csc_value),
          check_array("csc_twin", got.csc_twin, *want.csc_twin),
          check_array("csr_twin", got.csr_twin, *want.csr_twin),
          check_array("down_locks", got.down_locks, *want.down_locks),
          check_array("up_locks", got.up_locks, *want.up_locks)};
}

inline bool all_equal(const std::vector<array_check_t>& checks)
{
  for (const array_check_t& check : checks) {
    if (check.different != 0) { return false; }
  }
  return true;
}

}  // namespace device
}  // namespace gprs
