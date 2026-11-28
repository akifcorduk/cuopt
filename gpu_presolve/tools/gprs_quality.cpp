/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// gprs_quality <model.csr> [--primal-only] [--off <rule>]... [--pivot-growth-limit <x>]
//              [--saved-row-factor <x>] [--dump-reduced <file.csr>]
//              [the flags of option_flags.hpp]
// Runs the presolve and prints one JSON line with what the numbers of the input model and of
// the reduced model look like, for a comparison of two engines or of two values of an option
// (analysis/quality_compare.py reads the lines of two runs):
//   rows, cols, nnz, obj_nnz     sizes, and the columns with a cost
//   coef_min, coef_max           smallest and largest absolute coefficient of the matrix
//   row_dyn_max, row_dyn_p99,    largest ratio of the largest to the smallest absolute
//   row_dyn_mean_log10           coefficient within a row, its 99% quantile over the rows and
//                                the mean of its decimal logarithm
//   rows_dyn_1e3 .. rows_dyn_1e8 rows whose ratio is above 1e3, 1e4, 1e6, 1e8
//   cost_min, cost_max           smallest and largest absolute cost that is not zero
//   side_max, bound_max          largest finite side and largest finite bound
//   equations, ranged            rows with equal finite sides, with two different finite sides
//   integers, free_cols,         integer columns, columns without a finite bound, columns with
//   singleton_cols               one entry
//   offset                       the objective offset
// --off takes a rule family or a reduction of the column rules, as gprs_presolve does.
// --dump-reduced writes the reduced model as a GPRSCSR1 dump.
// Exit code 0 when the presolve ran, 1 when a file cannot be read or written, 2 on a usage
// error.

#include "option_flags.hpp"

#include <gprs/io.hpp>
#include <gprs/presolve.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void stats(const char* name, const gprs::model_t& m)
{
  double coef_min = gprs::inf, coef_max = 0.;
  double row_dyn_max = 1.;
  long long rows_1e3 = 0, rows_1e4 = 0, rows_1e6 = 0, rows_1e8 = 0;
  std::vector<double> dyn;
  dyn.reserve(m.nrows);
  for (int32_t r = 0; r < m.nrows; ++r) {
    double lo = gprs::inf, hi = 0.;
    for (int64_t e = m.row_start[r]; e < m.row_start[r + 1]; ++e) {
      double a = std::fabs(m.value[e]);
      lo       = std::min(lo, a);
      hi       = std::max(hi, a);
    }
    if (hi == 0.) { continue; }
    coef_min = std::min(coef_min, lo);
    coef_max = std::max(coef_max, hi);
    double d = hi / lo;
    dyn.push_back(d);
    row_dyn_max = std::max(row_dyn_max, d);
    rows_1e3 += d > 1e3;
    rows_1e4 += d > 1e4;
    rows_1e6 += d > 1e6;
    rows_1e8 += d > 1e8;
  }
  double row_dyn_p99 = 1.;
  double log_mean    = 0.;
  if (!dyn.empty()) {
    std::sort(dyn.begin(), dyn.end());
    row_dyn_p99 = dyn[(size_t)(0.99 * (dyn.size() - 1))];
    for (double d : dyn) {
      log_mean += std::log10(d);
    }
    log_mean /= dyn.size();
  }
  double cost_min = gprs::inf, cost_max = 0.;
  long long obj_nnz = 0;
  for (double c : m.cost) {
    if (c == 0.) { continue; }
    ++obj_nnz;
    cost_min = std::min(cost_min, std::fabs(c));
    cost_max = std::max(cost_max, std::fabs(c));
  }
  double side_max = 0., bound_max = 0.;
  long long ranged = 0, equations = 0;
  for (int32_t r = 0; r < m.nrows; ++r) {
    for (double s : {m.row_lower[r], m.row_upper[r]}) {
      if (std::isfinite(s)) { side_max = std::max(side_max, std::fabs(s)); }
    }
    bool two = std::isfinite(m.row_lower[r]) && std::isfinite(m.row_upper[r]);
    equations += two && m.row_lower[r] == m.row_upper[r];
    ranged += two && m.row_lower[r] != m.row_upper[r];
  }
  long long integers = 0, free_cols = 0, singletons = 0;
  std::vector<int32_t> size(m.ncols, 0);
  for (int64_t e = 0; e < m.nnz(); ++e) {
    ++size[m.col_index[e]];
  }
  for (int32_t c = 0; c < m.ncols; ++c) {
    for (double b : {m.col_lower[c], m.col_upper[c]}) {
      if (std::isfinite(b)) { bound_max = std::max(bound_max, std::fabs(b)); }
    }
    integers += m.col_type[c] != 0;
    free_cols += !std::isfinite(m.col_lower[c]) && !std::isfinite(m.col_upper[c]);
    singletons += size[c] == 1;
  }
  if (coef_max == 0.) { coef_min = 0.; }
  if (obj_nnz == 0) { cost_min = 0.; }
  std::printf(
    "\"%s\": {\"rows\": %d, \"cols\": %d, \"nnz\": %lld, \"obj_nnz\": %lld, \"coef_min\": %.6g, "
    "\"coef_max\": %.6g, \"row_dyn_max\": %.6g, \"row_dyn_p99\": %.6g, \"row_dyn_mean_log10\": "
    "%.6g, \"rows_dyn_1e3\": %lld, \"rows_dyn_1e4\": %lld, \"rows_dyn_1e6\": %lld, "
    "\"rows_dyn_1e8\": %lld, \"cost_min\": %.6g, \"cost_max\": %.6g, \"side_max\": %.6g, "
    "\"bound_max\": %.6g, \"equations\": %lld, \"ranged\": %lld, \"integers\": %lld, "
    "\"free_cols\": %lld, \"singleton_cols\": %lld, \"offset\": %.17g}",
    name,
    m.nrows,
    m.ncols,
    (long long)m.nnz(),
    obj_nnz,
    coef_min,
    coef_max,
    row_dyn_max,
    row_dyn_p99,
    log_mean,
    rows_1e3,
    rows_1e4,
    rows_1e6,
    rows_1e8,
    cost_min,
    cost_max,
    side_max,
    bound_max,
    equations,
    ranged,
    integers,
    free_cols,
    singletons,
    m.objective_offset);
}

int usage()
{
  std::fprintf(stderr,
               "usage: gprs_quality <model.csr> [--primal-only] [--off <rule>]... "
               "[--pivot-growth-limit <x>] [--saved-row-factor <x>] "
               "[--dump-reduced <file.csr>] %s\n",
               gprs_tools::switch_usage);
  return 2;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string path, dump;
  gprs::options_t options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    int taken       = gprs_tools::take_switch(argc, argv, i, options);
    if (taken != 0) {
      if (taken < 0) { return usage(); }
    } else if (arg == "--primal-only") {
      options.primal_only = true;
    } else if (arg == "--dump-reduced" && i + 1 < argc) {
      dump = argv[++i];
    } else if (arg == "--pivot-growth-limit" && i + 1 < argc) {
      options.pivot_growth_limit = std::atof(argv[++i]);
    } else if (arg == "--saved-row-factor" && i + 1 < argc) {
      options.saved_row_factor = std::atof(argv[++i]);
    } else if (arg == "--off" && i + 1 < argc) {
      int32_t rule = 0, sub = 0;
      ++i;
      while (rule < gprs::rule::count && std::strcmp(gprs::rule_name(rule), argv[i]) != 0) {
        ++rule;
      }
      while (sub < gprs::sub_rule::count && std::strcmp(gprs::sub_rule_name(sub), argv[i]) != 0) {
        ++sub;
      }
      if (rule < gprs::rule::count) {
        options.enabled[rule] = false;
      } else if (sub < gprs::sub_rule::count) {
        options.sub_enabled[sub] = false;
      } else {
        return usage();
      }
    } else if (path.empty() && arg[0] != '-') {
      path = arg;
    } else {
      return usage();
    }
  }
  if (path.empty()) { return usage(); }
  gprs::model_t model;
  std::string error;
  if (!gprs::read_csr_dump(path, model, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  gprs::result_t result = gprs::presolve(model, options);
  if (!dump.empty() && !gprs::write_csr_dump(dump, result.reduced, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::string name = path.substr(path.find_last_of('/') + 1);
  if (name.size() > 4 && name.compare(name.size() - 4, 4, ".csr") == 0) {
    name.resize(name.size() - 4);
  }
  std::printf(
    "{\"instance\": \"%s\", \"status\": \"%s\", ", name.c_str(), gprs::status_name(result.status));
  stats("input", model);
  std::printf(", ");
  stats("reduced", result.reduced);
  std::printf("}\n");
  return 0;
}
