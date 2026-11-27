/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The device build of the derived arrays on small models made here, through every transpose,
// position width and transfer, against the host bit for bit: the state_t of the host library
// when the model is well_formed, the host loop of bench/reference.hpp when it is not (rows that
// are not sorted, duplicate entries). The outputs are poisoned in every build and the pinned
// chunks are 64 bytes, so every array crosses chunks.
//
// usage: state_build_test             the models and the two faults
//        state_build_test --memory    builds while another allocation holds the device memory
//        state_build_test --big <nonzeros> [columns]
//                                     one generated model through radix_sort, pageable; for
//                                     sizes that no instance has (above 2^31 nonzeros)
// Returns 0 when everything is as expected.

#include "../bench/reference.hpp"

#include <gprs/device/state_build.hpp>
#include <gprs/state.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace gprs::device;

namespace {

struct random_t {
  uint64_t state;

  uint32_t next()
  {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return state >> 33;
  }

  uint32_t below(uint32_t n) { return next() % n; }
};

// Both signs, the largest and the smallest magnitudes of a double.
double some_value(random_t& random)
{
  static const double magnitude[] = {1.,
                                     0.5,
                                     3.,
                                     1e-300,
                                     1e300,
                                     std::numeric_limits<double>::denorm_min(),
                                     std::numeric_limits<double>::max(),
                                     0.1};
  double a                        = magnitude[random.below(8)];
  return random.below(2) ? a : -a;
}

void set_columns(gprs::model_t& model, int32_t ncols)
{
  model.ncols = ncols;
  model.col_lower.assign(ncols, 0.);
  model.col_upper.assign(ncols, 1.);
  model.cost.assign(ncols, 0.);
  model.col_type.assign(ncols, 0);
}

// The four kinds of sides in turn: both finite, lower only, upper only, none.
void add_row(gprs::model_t& model, const std::vector<std::pair<int32_t, double>>& entries)
{
  if (model.row_start.empty()) { model.row_start.push_back(0); }
  int kind = model.nrows % 4;
  model.row_lower.push_back(kind == 0 || kind == 1 ? -2. : -gprs::inf);
  model.row_upper.push_back(kind == 0 || kind == 2 ? 3. : gprs::inf);
  for (const std::pair<int32_t, double>& entry : entries) {
    model.col_index.push_back(entry.first);
    model.value.push_back(entry.second);
  }
  model.row_start.push_back(model.col_index.size());
  ++model.nrows;
}

gprs::model_t empty_model(int32_t nrows, int32_t ncols)
{
  gprs::model_t model;
  set_columns(model, ncols);
  model.row_start.push_back(0);
  for (int32_t i = 0; i < nrows; ++i) {
    add_row(model, {});
  }
  return model;
}

// Every entry (i, j) for which keep(i, j) holds, rows sorted.
template <typename F>
gprs::model_t pattern_model(int32_t nrows, int32_t ncols, uint64_t seed, F&& keep)
{
  random_t random{seed};
  gprs::model_t model;
  set_columns(model, ncols);
  model.row_start.push_back(0);
  std::vector<std::pair<int32_t, double>> entries;
  for (int32_t i = 0; i < nrows; ++i) {
    entries.clear();
    for (int32_t j = 0; j < ncols; ++j) {
      if (keep(i, j)) { entries.push_back({j, some_value(random)}); }
    }
    add_row(model, entries);
  }
  return model;
}

gprs::model_t random_model(int32_t nrows, int32_t ncols, uint32_t one_in, uint64_t seed)
{
  random_t pick{seed ^ 0x9e3779b97f4a7c15ull};
  return pattern_model(
    nrows, ncols, seed, [&](int32_t, int32_t) { return pick.below(one_in) == 0; });
}

void shuffle_rows(gprs::model_t& model, uint64_t seed)
{
  random_t random{seed};
  for (int32_t i = 0; i < model.nrows; ++i) {
    for (int64_t e = model.row_start[i + 1] - 1; e > model.row_start[i]; --e) {
      int64_t other = model.row_start[i] + random.below(e - model.row_start[i] + 1);
      std::swap(model.col_index[e], model.col_index[other]);
      std::swap(model.value[e], model.value[other]);
    }
  }
}

void reverse_rows(gprs::model_t& model)
{
  for (int32_t i = 0; i < model.nrows; ++i) {
    std::reverse(model.col_index.begin() + model.row_start[i],
                 model.col_index.begin() + model.row_start[i + 1]);
    std::reverse(model.value.begin() + model.row_start[i],
                 model.value.begin() + model.row_start[i + 1]);
  }
}

// Every third entry gets the column of the entry before it.
void duplicate_entries(gprs::model_t& model)
{
  for (int32_t i = 0; i < model.nrows; ++i) {
    for (int64_t e = model.row_start[i] + 1; e < model.row_start[i + 1]; e += 3) {
      model.col_index[e] = model.col_index[e - 1];
    }
  }
}

struct tester_t {
  state_builder_t builder;
  int failures = 0;
  int builds   = 0;

  // The eight arrays of the host for this model.
  state_arrays_t host(const gprs::model_t& model, bool& from_state)
  {
    state_arrays_t want;
    host_derived_arrays(model, want);
    from_state = gprs::well_formed(model);
    if (from_state) {
      gprs::tolerances_t tol;
      gprs::state_t st(model, tol);
      if (!all_equal(check_arrays(want, reference_of(st)))) {
        printf("FAIL the host loop differs from state_t\n");
        ++failures;
      }
    }
    return want;
  }

  bool build_one(const gprs::model_t& model,
                 const build_options_t& options,
                 const state_arrays_t& want,
                 const char* name_of_model)
  {
    size_t bytes;
    builder.reserve_staging(model, options.transfer, 64, bytes);
    state_arrays_t out;
    build_timing_t timing;
    bool built = builder.build(model, options, out, timing);
    builder.release_staging();
    ++builds;
    if (!built) {
      printf("FAIL %s: no device memory\n", name_of_model);
      return false;
    }
    bool equal = true;
    for (const array_check_t& check : check_arrays(out, reference_of(want))) {
      if (check.different == 0) { continue; }
      printf("FAIL %s: %s %s %d-bit%s: %s differs in %lld elements, first %lld\n",
             name_of_model,
             name(options.transpose),
             name(options.transfer),
             timing.position_bits,
             options.untouched_destination ? " untouched" : "",
             check.name,
             (long long)check.different,
             (long long)check.first);
      equal = false;
    }
    return equal;
  }

  void check(const char* name_of_model, const gprs::model_t& model)
  {
    bool from_state;
    state_arrays_t want = host(model, from_state);
    int failed          = 0;
    int before          = builds;
    for (transpose_t transpose :
         {transpose_t::cusparse, transpose_t::radix_sort, transpose_t::scatter_sort}) {
      for (bool wide : {false, true}) {
        if (wide && transpose == transpose_t::cusparse) { continue; }
        for (transfer_t transfer :
             {transfer_t::pageable, transfer_t::pinned, transfer_t::pinned_chunked}) {
          build_options_t options;
          options.transpose      = transpose;
          options.transfer       = transfer;
          options.wide_positions = wide;
          options.poison         = true;
          failed += !build_one(model, options, want, name_of_model);
          if (transfer == transfer_t::pageable) {
            options.untouched_destination = true;
            failed += !build_one(model, options, want, name_of_model);
          }
        }
      }
    }
    printf("%s %s: %d rows, %d columns, %lld nonzeros, %d builds against %s\n",
           failed ? "FAIL" : "ok  ",
           name_of_model,
           model.nrows,
           model.ncols,
           (long long)model.nnz(),
           builds - before,
           from_state ? "state_t" : "the host loop (not well_formed)");
    failures += failed != 0;
  }

  // The comparison must see a fault, in the arrays the fault touches and in no other.
  void check_fault(const gprs::model_t& model, fault_t fault)
  {
    bool from_state;
    state_arrays_t want = host(model, from_state);
    bool as_expected    = true;
    for (transpose_t transpose :
         {transpose_t::cusparse, transpose_t::radix_sort, transpose_t::scatter_sort}) {
      build_options_t options;
      options.transpose = transpose;
      options.poison    = true;
      options.fault     = fault;
      state_arrays_t out;
      build_timing_t timing;
      as_expected = builder.build(model, options, out, timing) && as_expected;
      ++builds;
      for (const array_check_t& check : check_arrays(out, reference_of(want))) {
        std::string array = check.name;
        int64_t expected  = 0;
        if (fault == fault_t::reverse_column) {
          expected = array == "csc_index" || array == "csc_twin" || array == "csr_twin" ? 2 : 0;
          // The two values that changed places can be equal.
          if (array == "csc_value") { expected = check.different; }
        } else if (array == "up_locks") {
          expected = 1;
        }
        if (check.different != expected) {
          printf("FAIL fault %d, %s: %s differs in %lld elements, expected %lld\n",
                 (int)fault,
                 name(transpose),
                 check.name,
                 (long long)check.different,
                 (long long)expected);
          as_expected = false;
        }
      }
    }
    printf("%s fault %s: the comparison reports it for every transpose\n",
           as_expected ? "ok  " : "FAIL",
           fault == fault_t::reverse_column ? "reverse_column" : "drop_lock");
    failures += !as_expected;
  }
};

#define CUDA_CHECK(call)                                                                      \
  do {                                                                                        \
    cudaError_t cuda_status_ = (call);                                                        \
    if (cuda_status_ != cudaSuccess) {                                                        \
      fprintf(stderr, "state_build_test: %s: %s\n", #call, cudaGetErrorString(cuda_status_)); \
      abort();                                                                                \
    }                                                                                         \
  } while (0)

// `leave` bytes of the device stay free while the build runs.
bool build_with_device_held(tester_t& tester,
                            const gprs::model_t& model,
                            transpose_t transpose,
                            size_t leave,
                            state_arrays_t& out)
{
  size_t free_bytes, total_bytes;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  void* hold = nullptr;
  if (free_bytes > leave) { CUDA_CHECK(cudaMalloc(&hold, free_bytes - leave)); }
  build_options_t options;
  options.transpose = transpose;
  options.poison    = true;
  build_timing_t timing;
  bool built = tester.builder.build(model, options, out, timing);
  if (hold) { CUDA_CHECK(cudaFree(hold)); }
  return built;
}

int memory_test()
{
  tester_t tester;
  gprs::model_t model =
    pattern_model(200000, 1000, 7, [](int32_t i, int32_t j) { return (i * 7 + j) % 10 == 0; });
  bool from_state;
  state_arrays_t want = tester.host(model, from_state);
  size_t nrows = model.nrows, ncols = model.ncols, nnz = model.nnz();
  // The inputs and the outputs, which a build allocates first; the peak adds the temporaries.
  size_t fixed = (nrows + 1) * 8 + nnz * 12 + nrows * 16 + (ncols + 1) * 12 + nnz * 28 + ncols * 8;
  size_t megabyte = 1 << 20;
  int failures    = tester.failures;
  for (transpose_t transpose :
       {transpose_t::cusparse, transpose_t::radix_sort, transpose_t::scatter_sort}) {
    build_options_t options;
    options.transpose = transpose;
    state_arrays_t out;
    build_timing_t timing;
    bool built  = tester.builder.build(model, options, out, timing);
    size_t peak = timing.device_bytes_peak;
    printf("%s %s on the free device: build returned %s, inputs and outputs %zu MB, peak %zu MB\n",
           built ? "ok  " : "FAIL",
           name(transpose),
           built ? "true" : "false",
           fixed / megabyte,
           peak / megabyte);
    failures += !built;
    // Not even the inputs; the inputs and outputs but not the temporaries; everything.
    for (size_t leave : {8 * megabyte, (fixed + peak) / 2, peak + 512 * megabyte}) {
      built         = build_with_device_held(tester, model, transpose, leave, out);
      bool expected = leave > peak;
      bool equal    = built && all_equal(check_arrays(out, reference_of(want)));
      bool ok       = built == expected && (!built || equal);
      printf("%s %s with %zu MB of the device free: build returned %s%s\n",
             ok ? "ok  " : "FAIL",
             name(transpose),
             leave / megabyte,
             built ? "true" : "false",
             built ? (equal ? ", bits equal" : ", bits differ") : "");
      failures += !ok;
    }
  }
  return failures ? 3 : 0;
}

int big_test(int64_t nnz, int32_t ncols)
{
  int32_t per_row = 32;
  int32_t stride  = ncols / per_row;
  gprs::model_t model;
  set_columns(model, ncols);
  int64_t nrows = (nnz + per_row - 1) / per_row;
  model.nrows   = nrows;
  model.row_start.resize(nrows + 1);
  model.row_lower.resize(nrows);
  model.row_upper.resize(nrows);
  model.col_index.resize(nnz);
  model.value.resize(nnz);
  for (int64_t i = 0; i < nrows; ++i) {
    int64_t begin      = i * per_row;
    int64_t end        = std::min(begin + per_row, nnz);
    model.row_start[i] = begin;
    model.row_lower[i] = i % 4 < 2 ? -2. : -gprs::inf;
    model.row_upper[i] = i % 2 == 0 ? 3. : gprs::inf;
    int32_t first      = (uint32_t)(i * 2654435761u) % stride;
    for (int64_t e = begin; e < end; ++e) {
      model.col_index[e] = first + (e - begin) * stride;
      model.value[e]     = (e % 3 == 0 ? -1. : 1.) * (1 + e % 5);
    }
  }
  model.row_start[nrows] = nnz;
  printf(
    "generated: %lld rows, %d columns, %lld nonzeros\n", (long long)nrows, ncols, (long long)nnz);
  fflush(stdout);

  state_arrays_t want;
  auto start = std::chrono::steady_clock::now();
  host_derived_arrays(model, want);
  double host_seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  printf("host loop: %.2f s\n", host_seconds);
  fflush(stdout);

  state_builder_t builder;
  build_options_t options;
  options.transpose = transpose_t::radix_sort;
  options.poison    = true;
  state_arrays_t out;
  build_timing_t timing;
  if (!builder.build(model, options, out, timing)) {
    printf("the device has not the memory for this model\n");
    return 1;
  }
  bool equal = true;
  for (const array_check_t& check : check_arrays(out, reference_of(want))) {
    if (check.different == 0) { continue; }
    printf("FAIL %s differs in %lld elements, first %lld\n",
           check.name,
           (long long)check.different,
           (long long)check.first);
    equal = false;
  }
  printf(
    "%s radix_sort pageable, %d-bit positions: alloc %.3f, upload %.3f, device %.3f, "
    "download %.3f, free %.3f, total %.3f s; device peak %.2f GB (%.1f bytes per nonzero)\n",
    equal ? "ok  " : "FAIL",
    timing.position_bits,
    timing.alloc_seconds,
    timing.upload_seconds,
    timing.device_seconds,
    timing.download_seconds,
    timing.free_seconds,
    timing.total_seconds(),
    timing.device_bytes_peak / 1e9,
    (double)timing.device_bytes_peak / nnz);
  return equal ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "--memory") { return memory_test(); }
  if (mode == "--big") {
    return big_test(std::atoll(argv[2]), argc > 3 ? std::atoi(argv[3]) : 1 << 20);
  }

  tester_t tester;
  tester.check("no rows, no columns", empty_model(0, 0));
  tester.check("rows, no columns", empty_model(3, 0));
  tester.check("columns, no rows", empty_model(0, 3));
  tester.check("rows and columns, no entry", empty_model(5, 7));
  tester.check("one entry", pattern_model(1, 1, 1, [](int32_t, int32_t) { return true; }));
  tester.check("one dense row", pattern_model(1, 1000, 2, [](int32_t, int32_t) { return true; }));
  tester.check("one dense column",
               pattern_model(1000, 1, 3, [](int32_t, int32_t) { return true; }));
  tester.check("dense 37 x 41", pattern_model(37, 41, 4, [](int32_t, int32_t) { return true; }));
  tester.check("diagonal",
               pattern_model(3000, 3000, 5, [](int32_t i, int32_t j) { return i == j; }));
  tester.check("first and last row and column empty",
               pattern_model(6, 6, 6, [](int32_t i, int32_t j) {
                 return i > 0 && i < 5 && j > 0 && j < 5 && i != 3 && j != 2;
               }));
  // Runs of 96 empty rows and 112 empty columns: longer than a chunk of the walks.
  tester.check("long runs of empty rows and columns",
               pattern_model(9000, 9000, 7, [](int32_t i, int32_t j) {
                 return i % 97 == 0 && j % 113 == 0 && (i / 97 + j / 113) % 3 == 0;
               }));
  tester.check(
    "only the last row and the last column",
    pattern_model(5000, 5000, 8, [](int32_t i, int32_t j) { return i == 4999 || j == 4999; }));
  tester.check("only the first row and the first column",
               pattern_model(5000, 5000, 9, [](int32_t i, int32_t j) { return i == 0 || j == 0; }));
  // More than the 8192 slots of one block of a walk in one row and in one column.
  tester.check("three dense rows of 20000",
               pattern_model(3, 20000, 10, [](int32_t, int32_t) { return true; }));
  tester.check("three dense columns of 20000",
               pattern_model(20000, 3, 11, [](int32_t, int32_t) { return true; }));
  tester.check("random 300 x 300, one in 20", random_model(300, 300, 20, 12));
  tester.check("random 4000 x 4000, one in 1500", random_model(4000, 4000, 1500, 13));
  tester.check("random 50 x 20000, one in 20", random_model(50, 20000, 20, 14));
  tester.check("random 20000 x 50, one in 20", random_model(20000, 50, 20, 15));
  tester.check("random 2000 x 3000, one in 40", random_model(2000, 3000, 40, 16));

  gprs::model_t unsorted = random_model(2000, 3000, 40, 17);
  shuffle_rows(unsorted, 18);
  tester.check("rows in random order", unsorted);
  gprs::model_t reversed = random_model(2000, 3000, 40, 19);
  reverse_rows(reversed);
  tester.check("rows in decreasing column order", reversed);
  gprs::model_t duplicates = random_model(2000, 3000, 40, 20);
  duplicate_entries(duplicates);
  tester.check("duplicate entries", duplicates);
  shuffle_rows(duplicates, 21);
  tester.check("duplicate entries, rows in random order", duplicates);

  gprs::model_t faulty = random_model(300, 300, 20, 22);
  tester.check_fault(faulty, fault_t::reverse_column);
  tester.check_fault(faulty, fault_t::drop_lock);

  printf("%d builds, %d failures\n", tester.builds, tester.failures);
  return tester.failures ? 3 : 0;
}
