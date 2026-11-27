/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Times the arrays that the constructor of state_t derives from the matrix (the column-wise
// copy, the twin positions, the locks) built on the device and brought back to std::vector,
// against the host, and compares every array of every device run with the host state bit for
// bit. Also times one cusparseSpMV against a host loop.
//
// usage: state_build_bench <model.csr> <out.json> [options]
//   --reps n            timed device runs per variant (default 5 below 5e7 nonzeros, else 3)
//   --host-reps n       host runs (default 5)
//   --transposes a,b    of cusparse, radix_sort, scatter_sort (default all)
//   --transfers a,b     of pageable, pinned, pinned_chunked (default all)
//   --chunk-mb n        size of each of the two chunks of pinned_chunked (default 16)
//   --wide              64-bit positions in the two sorts
//   --untouched         the download goes into allocated memory that was never touched instead
//                       of sized vectors (pageable transfer only)
//   --reverse-rows      check only: the entries of every row in decreasing column order, the
//                       reference is the host loop of reference.hpp, nothing is timed
//   --fault kind        reverse_column or drop_lock: the device result is made wrong on purpose
//                       (fault_t), so the tool must report a difference and return 3
//   --fault-run k       the fault in one run only: -1 the untimed run, 0.. a timed run
//   --no-spmv
// Every device variant runs once untimed first (outputs poisoned, device memory sampled), then
// the timed runs. Times are seconds; "best" is the run with the smallest total. These are the
// times of a build in a process that has built before: what the first build of a process
// costs, which is the one a presolve call gets, is measured by the two modes below.
//
//   --cold              one variant (--transposes x --transfers y), nothing before it but the
//                       builder and one small allocation and transfer each way: the first
//                       build of the process, then a second one. The comparison with the host
//                       state follows the two builds.
//   --cold-host         the host loop of the derived arrays as the first work of the process,
//                       then --host-reps - 1 more runs
// Both print one JSON line to stdout; <out.json> is not written (pass "-").

#include "reference.hpp"

#include <gprs/device/state_build.hpp>
#include <gprs/io.hpp>
#include <gprs/state.hpp>

#include <cuda_runtime.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace gprs::device;

namespace {

double seconds_since(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

struct memory_sampler_t {
  std::atomic<bool> stop{false};
  size_t peak     = 0;
  int64_t samples = 0;
  std::thread thread;

  void start(const state_builder_t& builder)
  {
    thread = std::thread([this, &builder] {
      while (!stop) {
        peak = std::max(peak, builder.device_used_bytes());
        ++samples;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });
  }

  void finish()
  {
    stop = true;
    thread.join();
  }
};

struct variant_t {
  transpose_t transpose;
  transfer_t transfer;
  std::vector<build_timing_t> runs;
  int best         = 0;
  int runs_checked = 0;
  int runs_equal   = 0;
  // Of the first run that differs, or of the last run.
  std::vector<array_check_t> checks;
  size_t sampled_bytes   = 0;
  int64_t samples        = 0;
  double untimed_seconds = 0;
};

struct spmv_t {
  bool deterministic;
  spmv_timing_t timing;
  int64_t rows_different = 0;
  double max_difference  = 0;
};

std::string first_line(const char* path)
{
  std::ifstream file(path);
  std::string line;
  std::getline(file, line);
  return line;
}

std::string loadavg() { return first_line("/proc/loadavg"); }

std::vector<std::string> split(const std::string& text)
{
  std::vector<std::string> parts;
  size_t begin = 0;
  while (begin <= text.size()) {
    size_t end = text.find(',', begin);
    if (end == std::string::npos) { end = text.size(); }
    parts.push_back(text.substr(begin, end - begin));
    begin = end + 1;
  }
  return parts;
}

// In the order of "run_columns".
void print_timing(FILE* out, const build_timing_t& t)
{
  fprintf(out,
          "[%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]",
          t.alloc_seconds,
          t.upload_seconds,
          t.device_seconds,
          t.download_seconds,
          t.download_resize_seconds,
          t.free_seconds,
          t.total_seconds());
}

void print_runs(FILE* out, const std::vector<double>& runs)
{
  fprintf(out, "[");
  for (size_t k = 0; k < runs.size(); ++k) {
    fprintf(out, "%s%.6f", k ? ", " : "", runs[k]);
  }
  fprintf(out, "]");
}

double minimum(const std::vector<double>& runs)
{
  return *std::min_element(runs.begin(), runs.end());
}

// The mean number of busy CPUs, among those this process may run on, since the constructor;
// this process is one of them while it computes. The load average of the node says nothing
// about the 28 CPUs of the container.
struct cpu_busy_t {
  cpu_set_t allowed;
  double ticks_begin;
  std::chrono::steady_clock::time_point begin;

  cpu_busy_t()
  {
    sched_getaffinity(0, sizeof(allowed), &allowed);
    ticks_begin = ticks();
    begin       = std::chrono::steady_clock::now();
  }

  // user + nice + system + irq + softirq + steal of the allowed CPUs.
  double ticks() const
  {
    std::ifstream file("/proc/stat");
    std::string line;
    double sum = 0;
    while (std::getline(file, line)) {
      int cpu;
      // "cpu " is the sum over all CPUs.
      if (line.size() < 4 || line[3] == ' ' || sscanf(line.c_str(), "cpu%d", &cpu) != 1 ||
          !CPU_ISSET(cpu, &allowed)) {
        continue;
      }
      std::istringstream fields(line.substr(line.find(' ')));
      double t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (double& field : t) {
        fields >> field;
      }
      sum += t[0] + t[1] + t[2] + t[5] + t[6] + t[7];
    }
    return sum;
  }

  int allowed_cpus() const { return CPU_COUNT(&allowed); }

  double mean() const
  {
    return (ticks() - ticks_begin) / (seconds_since(begin) * sysconf(_SC_CLK_TCK));
  }
};

#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t cuda_status_ = (call);                                                         \
    if (cuda_status_ != cudaSuccess) {                                                         \
      fprintf(stderr, "state_build_bench: %s: %s\n", #call, cudaGetErrorString(cuda_status_)); \
      abort();                                                                                 \
    }                                                                                          \
  } while (0)

// What a process that has used CUDA before has behind it when it comes to the presolve: device
// memory and a transfer each way. Nothing of the device library is touched.
double prewarm()
{
  auto start = std::chrono::steady_clock::now();
  std::vector<char> host(1 << 20, 1);
  void* device;
  CUDA_CHECK(cudaMalloc(&device, host.size()));
  CUDA_CHECK(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device));
  return seconds_since(start);
}

void print_system(FILE* out, const cpu_busy_t& busy)
{
  const char* tunables = std::getenv("GLIBC_TUNABLES");
  fprintf(out,
          "\"loadavg\": \"%s\", \"cpus_allowed\": %d, \"cpus_busy_mean\": %.2f, "
          "\"glibc_tunables\": \"%s\"",
          loadavg().c_str(),
          busy.allowed_cpus(),
          busy.mean(),
          tunables ? tunables : "");
}

int run_cold_host(const gprs::model_t& model, const std::string& instance, int host_reps)
{
  cpu_busy_t busy;
  std::vector<double> derived_runs;
  state_arrays_t derived;
  for (int k = 0; k < host_reps; ++k) {
    derived    = state_arrays_t();
    auto start = std::chrono::steady_clock::now();
    host_derived_arrays(model, derived);
    derived_runs.push_back(seconds_since(start));
  }
  gprs::tolerances_t tol;
  auto start = std::chrono::steady_clock::now();
  gprs::state_t st(model, tol);
  double build_seconds = seconds_since(start);
  bool equal           = all_equal(check_arrays(derived, reference_of(st)));

  printf(
    "{\"mode\": \"cold_host\", \"instance\": \"%s\", \"rows\": %d, \"cols\": %d, "
    "\"nnz\": %lld, ",
    instance.c_str(),
    model.nrows,
    model.ncols,
    (long long)model.nnz());
  print_system(stdout, busy);
  printf(", \"derived_arrays_first_seconds\": %.6f, \"derived_arrays_runs\": ", derived_runs[0]);
  print_runs(stdout, derived_runs);
  printf(", \"state_build_after_seconds\": %.6f, \"derived_arrays_equal_state\": %s}\n",
         build_seconds,
         equal ? "true" : "false");
  return equal ? 0 : 3;
}

int run_cold_device(const gprs::model_t& model,
                    const std::string& instance,
                    const build_options_t& options,
                    size_t chunk_bytes)
{
  cpu_busy_t busy;
  state_builder_t builder;
  double prewarm_seconds = prewarm();
  size_t staging_bytes   = 0;
  double staging_seconds =
    builder.reserve_staging(model, options.transfer, chunk_bytes, staging_bytes);
  state_arrays_t out[2];
  build_timing_t timing[2];
  for (int k = 0; k < 2; ++k) {
    if (!builder.build(model, options, out[k], timing[k])) {
      fprintf(
        stderr, "state_build_bench: the device has not the memory for %s\n", instance.c_str());
      return 1;
    }
  }
  builder.release_staging();

  gprs::tolerances_t tol;
  gprs::state_t st(model, tol);
  bool equal[2];
  for (int k = 0; k < 2; ++k) {
    equal[k] = all_equal(check_arrays(out[k], reference_of(st)));
  }

  // What the first build of a process costs a call: its own time, the pinned memory of the
  // transfer, and the cuSPARSE handle when the transpose uses it.
  const device_info_t& info = builder.info();
  double handle_seconds = options.transpose == transpose_t::cusparse ? info.cusparse_seconds : 0;
  printf(
    "{\"mode\": \"cold\", \"instance\": \"%s\", \"rows\": %d, \"cols\": %d, \"nnz\": %lld, "
    "\"transpose\": \"%s\", \"transfer\": \"%s\", \"position_bits\": %d, ",
    instance.c_str(),
    model.nrows,
    model.ncols,
    (long long)model.nnz(),
    name(options.transpose),
    name(options.transfer),
    timing[0].position_bits);
  print_system(stdout, busy);
  printf(
    ", \"cuda_context_seconds\": %.6f, \"cusparse_create_seconds\": %.6f, "
    "\"prewarm_seconds\": %.6f, \"staging_seconds\": %.6f, \"staging_bytes\": %zu, "
    "\"first\": ",
    info.context_seconds,
    info.cusparse_seconds,
    prewarm_seconds,
    staging_seconds,
    staging_bytes);
  print_timing(stdout, timing[0]);
  printf(", \"second\": ");
  print_timing(stdout, timing[1]);
  printf(
    ", \"first_call_seconds\": %.6f, \"second_call_seconds\": %.6f, "
    "\"first_equal\": %s, \"second_equal\": %s}\n",
    timing[0].total_seconds() + staging_seconds + handle_seconds,
    timing[1].total_seconds() + staging_seconds,
    equal[0] ? "true" : "false",
    equal[1] ? "true" : "false");
  return equal[0] && equal[1] ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3) {
    fprintf(stderr, "usage: state_build_bench <model.csr> <out.json> [options]\n");
    return 2;
  }
  std::string path = argv[1];
  int reps = 0, host_reps = 0;
  size_t chunk_bytes = 16 << 20;
  bool wide = false, untouched = false, reverse_rows = false, with_spmv = true;
  bool cold = false, cold_host = false;
  fault_t fault                       = fault_t::none;
  int fault_run                       = -2;
  std::vector<transpose_t> transposes = {
    transpose_t::cusparse, transpose_t::radix_sort, transpose_t::scatter_sort};
  std::vector<transfer_t> transfers = {
    transfer_t::pageable, transfer_t::pinned, transfer_t::pinned_chunked};
  for (int k = 3; k < argc; ++k) {
    std::string arg = argv[k];
    if (arg == "--reps") {
      reps = std::atoi(argv[++k]);
    } else if (arg == "--host-reps") {
      host_reps = std::atoi(argv[++k]);
    } else if (arg == "--chunk-mb") {
      chunk_bytes = (size_t)std::atoi(argv[++k]) << 20;
    } else if (arg == "--wide") {
      wide = true;
    } else if (arg == "--untouched") {
      untouched = true;
      transfers = {transfer_t::pageable};
    } else if (arg == "--reverse-rows") {
      reverse_rows = true;
    } else if (arg == "--no-spmv") {
      with_spmv = false;
    } else if (arg == "--cold") {
      cold = true;
    } else if (arg == "--cold-host") {
      cold_host = true;
    } else if (arg == "--fault") {
      std::string kind = argv[++k];
      fault            = kind == "reverse_column" ? fault_t::reverse_column
                         : kind == "drop_lock"    ? fault_t::drop_lock
                                                  : fault_t::none;
      if (fault == fault_t::none) {
        fprintf(stderr, "state_build_bench: unknown fault %s\n", kind.c_str());
        return 2;
      }
    } else if (arg == "--fault-run") {
      fault_run = std::atoi(argv[++k]);
    } else if (arg == "--transposes") {
      transposes.clear();
      for (const std::string& part : split(argv[++k])) {
        for (transpose_t t :
             {transpose_t::cusparse, transpose_t::radix_sort, transpose_t::scatter_sort}) {
          if (part == name(t)) { transposes.push_back(t); }
        }
      }
    } else if (arg == "--transfers") {
      transfers.clear();
      for (const std::string& part : split(argv[++k])) {
        for (transfer_t t :
             {transfer_t::pageable, transfer_t::pinned, transfer_t::pinned_chunked}) {
          if (part == name(t)) { transfers.push_back(t); }
        }
      }
    } else {
      fprintf(stderr, "state_build_bench: unknown option %s\n", arg.c_str());
      return 2;
    }
  }

  gprs::model_t model;
  std::string error;
  auto start = std::chrono::steady_clock::now();
  if (!gprs::read_csr_dump(path, model, error)) {
    fprintf(stderr, "state_build_bench: %s\n", error.c_str());
    return 1;
  }
  double read_seconds = seconds_since(start);
  int64_t nnz         = model.nnz();
  if (reps == 0) { reps = nnz < 50000000 ? 5 : 3; }
  if (host_reps == 0) { host_reps = 5; }
  if (reverse_rows) {
    for (int32_t i = 0; i < model.nrows; ++i) {
      std::reverse(model.col_index.begin() + model.row_start[i],
                   model.col_index.begin() + model.row_start[i + 1]);
      std::reverse(model.value.begin() + model.row_start[i],
                   model.value.begin() + model.row_start[i + 1]);
    }
    reps      = 1;
    host_reps = 1;
    with_spmv = false;
  }
  std::string instance = path.substr(path.find_last_of('/') + 1);
  instance             = instance.substr(0, instance.rfind(".csr"));
  if (cold_host) { return run_cold_host(model, instance, host_reps); }
  if (cold) {
    if (transposes.size() != 1 || transfers.size() != 1) {
      fprintf(stderr, "state_build_bench: --cold takes one transpose and one transfer\n");
      return 2;
    }
    build_options_t options;
    options.transpose      = transposes[0];
    options.transfer       = transfers[0];
    options.wide_positions = wide;
    return run_cold_device(model, instance, options, chunk_bytes);
  }
  std::string load_start = loadavg();
  cpu_busy_t busy;
  fprintf(stderr,
          "%s: %d rows, %d columns, %lld nonzeros, read in %.2f s\n",
          instance.c_str(),
          model.nrows,
          model.ncols,
          (long long)nnz,
          read_seconds);

  // Host: the constructor and the fold as state_build_probe times them, then the derived arrays
  // alone.
  std::unique_ptr<gprs::state_t> st;
  std::vector<double> build_runs, fold_runs, derived_runs;
  double checksum = 0;
  if (!reverse_rows) {
    gprs::tolerances_t tol;
    for (int k = 0; k < host_reps; ++k) {
      st.reset();
      start = std::chrono::steady_clock::now();
      st    = std::make_unique<gprs::state_t>(model, tol);
      build_runs.push_back(seconds_since(start));

      start = std::chrono::steady_clock::now();
      gprs::side_sum_t low, high;
      double sum = 0;
      for (int32_t i = 0; i < st->nrows; ++i) {
        st->fold_row_activity(i, low, high);
        sum += low.sum.hi + high.sum.hi;
      }
      fold_runs.push_back(seconds_since(start));
      checksum = sum;
    }
  }
  state_arrays_t derived;
  for (int k = 0; k < host_reps; ++k) {
    derived = state_arrays_t();
    start   = std::chrono::steady_clock::now();
    host_derived_arrays(model, derived);
    derived_runs.push_back(seconds_since(start));
  }
  bool derived_equals_state = true;
  if (!reverse_rows) {
    derived_equals_state = all_equal(check_arrays(derived, reference_of(*st)));
    derived              = state_arrays_t();
  }
  reference_t reference = reverse_rows ? reference_of(derived) : reference_of(*st);
  fprintf(stderr, "host done, load %s\n", loadavg().c_str());

  state_builder_t builder;
  const device_info_t& info = builder.info();

  std::vector<variant_t> variants;
  double staging_seconds[3] = {0, 0, 0};
  size_t staging_bytes[3]   = {0, 0, 0};
  for (transfer_t transfer : transfers) {
    staging_seconds[(int)transfer] =
      builder.reserve_staging(model, transfer, chunk_bytes, staging_bytes[(int)transfer]);
    for (transpose_t transpose : transposes) {
      variant_t v;
      v.transpose = transpose;
      v.transfer  = transfer;
      build_options_t options;
      options.transpose             = transpose;
      options.transfer              = transfer;
      options.wide_positions        = wide;
      options.untouched_destination = untouched;
      for (int k = -1; k < reps; ++k) {
        state_arrays_t out;
        build_timing_t timing;
        options.fault = fault_run == -2 || fault_run == k ? fault : fault_t::none;
        bool built;
        if (k < 0) {
          options.poison  = true;
          size_t baseline = builder.device_used_bytes();
          memory_sampler_t sampler;
          sampler.start(builder);
          start             = std::chrono::steady_clock::now();
          built             = builder.build(model, options, out, timing);
          v.untimed_seconds = seconds_since(start);
          sampler.finish();
          v.sampled_bytes = sampler.peak > baseline ? sampler.peak - baseline : 0;
          v.samples       = sampler.samples;
          options.poison  = false;
        } else {
          built = builder.build(model, options, out, timing);
          v.runs.push_back(timing);
          if (timing.total_seconds() < v.runs[v.best].total_seconds()) { v.best = k; }
        }
        if (!built) {
          fprintf(stderr, "state_build_bench: the device has not the memory for this model\n");
          return 1;
        }
        std::vector<array_check_t> checks = check_arrays(out, reference);
        bool equal                        = all_equal(checks);
        if (v.runs_equal == v.runs_checked) { v.checks = checks; }
        ++v.runs_checked;
        v.runs_equal += equal;
      }
      const build_timing_t& best = v.runs[v.best];
      fprintf(stderr,
              "%s %s: total %.4f s, %d of %d runs equal\n",
              name(transpose),
              name(transfer),
              best.total_seconds(),
              v.runs_equal,
              v.runs_checked);
      variants.push_back(v);
    }
    builder.release_staging();
  }

  std::vector<spmv_t> spmvs;
  std::vector<double> host_product_runs;
  if (with_spmv) {
    std::vector<double> x(model.ncols, 1.), host_product(model.nrows);
    for (int k = 0; k < host_reps; ++k) {
      start = std::chrono::steady_clock::now();
      for (int32_t i = 0; i < model.nrows; ++i) {
        double sum = 0;
        for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
          sum += model.value[e] * x[model.col_index[e]];
        }
        host_product[i] = sum;
      }
      host_product_runs.push_back(seconds_since(start));
    }
    for (bool deterministic : {false, true}) {
      spmv_t r;
      r.deterministic = deterministic;
      std::vector<double> product;
      builder.spmv_ones(model, 10, deterministic, product, r.timing);
      for (int32_t i = 0; i < model.nrows; ++i) {
        if (memcmp(&product[i], &host_product[i], sizeof(double)) != 0) {
          ++r.rows_different;
          r.max_difference = std::max(r.max_difference, std::fabs(product[i] - host_product[i]));
        }
      }
      spmvs.push_back(r);
    }
  }
  std::string load_end = loadavg();
  double cpus_busy     = busy.mean();
  const char* tunables = std::getenv("GLIBC_TUNABLES");

  size_t nrows = model.nrows, ncols = model.ncols;
  size_t upload_bytes   = (nrows + 1) * 8 + (size_t)nnz * 12 + nrows * 16;
  size_t download_bytes = ncols * 20 + (size_t)nnz * 28;
  bool every_run_equal  = true;

  FILE* out = fopen(argv[2], "w");
  if (!out) {
    fprintf(stderr, "state_build_bench: cannot write %s\n", argv[2]);
    return 1;
  }
  fprintf(out, "{\n  \"tool\": \"state_build_bench\",\n");
  fprintf(out, "  \"instance\": \"%s\",\n  \"file\": \"%s\",\n", instance.c_str(), path.c_str());
  fprintf(out,
          "  \"rows\": %d,\n  \"cols\": %d,\n  \"nnz\": %lld,\n",
          model.nrows,
          model.ncols,
          (long long)nnz);
  fprintf(out,
          "  \"mode\": \"%s\",\n  \"wide_positions\": %s,\n  \"untouched_destination\": %s,\n",
          reverse_rows ? "reverse_rows_check" : "timing",
          wide ? "true" : "false",
          untouched ? "true" : "false");
  fprintf(out,
          "  \"system\": {\"gpu\": \"%s\", \"gpu_memory_bytes\": %zu, \"cuda_runtime\": %d, "
          "\"cusparse\": %d, \"cub\": %d, \"loadavg_start\": \"%s\", \"loadavg_end\": \"%s\", "
          "\"cpus_allowed\": %d, \"cpus_busy_mean\": %.2f, "
          "\"transparent_hugepage\": \"%s\", \"glibc_tunables\": \"%s\"},\n",
          info.gpu_name.c_str(),
          info.gpu_memory_bytes,
          info.cuda_runtime_version,
          info.cusparse_version,
          info.cub_version,
          load_start.c_str(),
          load_end.c_str(),
          busy.allowed_cpus(),
          cpus_busy,
          first_line("/sys/kernel/mm/transparent_hugepage/enabled").c_str(),
          tunables ? tunables : "");
  if (fault != fault_t::none) {
    fprintf(out,
            "  \"fault\": {\"kind\": \"%s\", \"run\": \"%s\"},\n",
            fault == fault_t::reverse_column ? "reverse_column" : "drop_lock",
            fault_run == -2 ? "every" : std::to_string(fault_run).c_str());
  }
  fprintf(out,
          "  \"once\": {\"cuda_context_seconds\": %.6f, \"cusparse_create_seconds\": %.6f, "
          "\"pinned_seconds\": %.6f, \"pinned_bytes\": %zu, \"pinned_chunks_seconds\": %.6f, "
          "\"pinned_chunks_bytes\": %zu},\n",
          info.context_seconds,
          info.cusparse_seconds,
          staging_seconds[(int)transfer_t::pinned],
          staging_bytes[(int)transfer_t::pinned],
          staging_seconds[(int)transfer_t::pinned_chunked],
          staging_bytes[(int)transfer_t::pinned_chunked]);
  fprintf(out,
          "  \"transfer_bytes\": {\"upload\": %zu, \"download\": %zu},\n",
          upload_bytes,
          download_bytes);
  fprintf(out,
          "  \"run_columns\": [\"alloc_seconds\", \"upload_seconds\", \"device_seconds\", "
          "\"download_seconds\", \"download_resize_seconds\", \"free_seconds\", "
          "\"total_seconds\"],\n");
  fprintf(out, "  \"host\": {\"repetitions\": %d", host_reps);
  if (!reverse_rows) {
    double build = minimum(build_runs), fold = minimum(fold_runs);
    fprintf(out,
            ", \"state_build_seconds\": %.6f, \"fold_all_rows_seconds\": %.6f, "
            "\"build_without_fold_seconds\": %.6f, \"checksum\": %.17g, "
            "\"derived_arrays_equal_state\": %s, \"state_build_runs\": ",
            build,
            fold,
            build - fold,
            checksum,
            derived_equals_state ? "true" : "false");
    print_runs(out, build_runs);
    fprintf(out, ", \"fold_all_rows_runs\": ");
    print_runs(out, fold_runs);
  }
  fprintf(
    out, ", \"derived_arrays_seconds\": %.6f, \"derived_arrays_runs\": ", minimum(derived_runs));
  print_runs(out, derived_runs);
  fprintf(out, "},\n  \"variants\": [\n");
  for (size_t k = 0; k < variants.size(); ++k) {
    const variant_t& v = variants[k];
    every_run_equal    = every_run_equal && v.runs_equal == v.runs_checked;
    fprintf(out,
            "    {\"transpose\": \"%s\", \"transfer\": \"%s\", \"position_bits\": %d, "
            "\"repetitions\": %d, \"best\": ",
            name(v.transpose),
            name(v.transfer),
            v.runs[0].position_bits,
            reps);
    print_timing(out, v.runs[v.best]);
    fprintf(out, ",\n     \"runs\": [");
    for (size_t r = 0; r < v.runs.size(); ++r) {
      fprintf(out, "%s", r ? ", " : "");
      print_timing(out, v.runs[r]);
    }
    fprintf(out,
            "],\n     \"device_bytes_allocated_peak\": %zu, \"device_bytes_sampled_peak\": %zu, "
            "\"memory_samples\": %lld, \"library_buffer_bytes\": %zu, "
            "\"untimed_run_seconds\": %.6f,\n"
            "     \"runs_checked\": %d, \"runs_equal\": %d, \"bit_equal\": %s, "
            "\"arrays_different\": {",
            v.runs[0].device_bytes_peak,
            v.sampled_bytes,
            (long long)v.samples,
            v.runs[0].library_buffer_bytes,
            v.untimed_seconds,
            v.runs_checked,
            v.runs_equal,
            v.runs_equal == v.runs_checked ? "true" : "false");
    const char* separator = "";
    for (const array_check_t& check : v.checks) {
      if (check.different == 0) { continue; }
      fprintf(out,
              "%s\"%s\": {\"different\": %lld, \"first\": %lld}",
              separator,
              check.name,
              (long long)check.different,
              (long long)check.first);
      separator = ", ";
    }
    fprintf(out, "}}%s\n", k + 1 < variants.size() ? "," : "");
  }
  fprintf(out, "  ],\n  \"every_run_bit_equal\": %s", every_run_equal ? "true" : "false");
  if (with_spmv) {
    fprintf(out,
            ",\n  \"spmv_ones\": {\"host_loop_seconds\": %.6f, \"host_loop_runs\": ",
            minimum(host_product_runs));
    print_runs(out, host_product_runs);
    fprintf(out, ", \"cusparse\": [");
    for (size_t k = 0; k < spmvs.size(); ++k) {
      const spmv_t& r = spmvs[k];
      fprintf(out,
              "%s\n    {\"algorithm\": \"%s\", \"repetitions\": %d, \"spmv_seconds\": %.6f, "
              "\"first_spmv_seconds\": %.6f, \"upload_seconds\": %.6f, \"setup_seconds\": %.6f, "
              "\"download_seconds\": %.6f, \"buffer_bytes\": %zu, "
              "\"rows_different_from_host_loop\": %lld, \"max_difference\": %.6g}",
              k ? "," : "",
              r.deterministic ? "CUSPARSE_SPMV_CSR_ALG2" : "CUSPARSE_SPMV_ALG_DEFAULT",
              r.timing.repetitions,
              r.timing.min_seconds,
              r.timing.first_seconds,
              r.timing.upload_seconds,
              r.timing.setup_seconds,
              r.timing.download_seconds,
              r.timing.buffer_bytes,
              (long long)r.rows_different,
              r.max_difference);
    }
    fprintf(out, "]}");
  }
  fprintf(out, "\n}\n");
  fclose(out);

  printf("%s: %d rows, %d columns, %lld nonzeros",
         instance.c_str(),
         model.nrows,
         model.ncols,
         (long long)nnz);
  double host_rest = 0;
  if (!reverse_rows) {
    host_rest = minimum(build_runs) - minimum(fold_runs);
    printf(
      "\nhost (min of %d): state_t %.4f s, fold %.4f s, state_t without fold %.4f s, "
      "derived arrays alone %.4f s",
      host_reps,
      minimum(build_runs),
      minimum(fold_runs),
      host_rest,
      minimum(derived_runs));
  }
  printf("\n%-13s %-15s %8s %8s %8s %8s %8s %8s %7s %7s %6s %7s\n",
         "transpose",
         "transfer",
         "alloc",
         "upload",
         "device",
         "download",
         "free",
         "total",
         "rest/t",
         "deriv/t",
         "equal",
         "dev GB");
  for (const variant_t& v : variants) {
    const build_timing_t& t = v.runs[v.best];
    printf("%-13s %-15s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %7.2f %7.2f %3d/%-2d %7.2f\n",
           name(v.transpose),
           name(v.transfer),
           t.alloc_seconds,
           t.upload_seconds,
           t.device_seconds,
           t.download_seconds,
           t.free_seconds,
           t.total_seconds(),
           host_rest / t.total_seconds(),
           minimum(derived_runs) / t.total_seconds(),
           v.runs_equal,
           v.runs_checked,
           std::max(v.runs[0].device_bytes_peak, v.sampled_bytes) / 1e9);
  }
  for (const spmv_t& r : spmvs) {
    printf("spmv %s: device %.6f s, host loop %.6f s, %lld rows differ, max difference %.3g\n",
           r.deterministic ? "CSR_ALG2" : "default",
           r.timing.min_seconds,
           minimum(host_product_runs),
           (long long)r.rows_different,
           r.max_difference);
  }
  return every_run_equal && derived_equals_state ? 0 : 3;
}
