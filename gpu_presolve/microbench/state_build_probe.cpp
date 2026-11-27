/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Splits the time the host engine needs to build its state from a model: the whole constructor
// of state_t, and inside it the fold of the two activities of every row, which is timed again
// on the finished state. What is left is the copy of the arrays, the transpose with the twin
// positions and the locks. Also timed: the presolve, with its own split, and reading the dump.
//
// usage: state_build_probe <model.csr> [repetitions]
// Prints one JSON line. Times are the minimum over the repetitions.

#include <gprs/io.hpp>
#include <gprs/presolve.hpp>
#include <gprs/state.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

static double seconds_since(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    fprintf(stderr, "usage: state_build_probe <model.csr> [repetitions]\n");
    return 2;
  }
  int repetitions = argc > 2 ? std::atoi(argv[2]) : 3;
  gprs::model_t model;
  std::string error;
  auto start = std::chrono::steady_clock::now();
  if (!gprs::read_csr_dump(argv[1], model, error)) {
    fprintf(stderr, "state_build_probe: %s\n", error.c_str());
    return 1;
  }
  double read_seconds = seconds_since(start);

  gprs::tolerances_t tol;
  double build = 1e300, fold = 1e300;
  double checksum = 0;
  for (int k = 0; k < repetitions; ++k) {
    start = std::chrono::steady_clock::now();
    gprs::state_t st(model, tol);
    build = std::min(build, seconds_since(start));

    start = std::chrono::steady_clock::now();
    gprs::side_sum_t low, high;
    double sum = 0;
    for (int32_t i = 0; i < st.nrows; ++i) {
      st.fold_row_activity(i, low, high);
      sum += low.sum.hi + high.sum.hi;
    }
    fold     = std::min(fold, seconds_since(start));
    checksum = sum;
  }

  gprs::options_t options;
  gprs::result_t result = gprs::presolve(model, options);
  printf(
    "{\"file\": \"%s\", \"rows\": %d, \"cols\": %d, \"nnz\": %lld, \"read_seconds\": %.6f, "
    "\"state_build_seconds\": %.6f, \"fold_all_rows_seconds\": %.6f, "
    "\"build_without_fold_seconds\": %.6f, \"presolve_seconds\": %.6f, "
    "\"presolve_state_seconds\": %.6f, \"presolve_finish_seconds\": %.6f, "
    "\"presolve_hash_seconds\": %.6f, \"status\": \"%s\", \"checksum\": %.17g}\n",
    argv[1],
    model.nrows,
    model.ncols,
    (long long)model.nnz(),
    read_seconds,
    build,
    fold,
    build - fold,
    result.seconds,
    result.seconds_state,
    result.seconds_finish,
    result.seconds_hash,
    gprs::status_name(result.status),
    checksum);
  return 0;
}
