/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The command line flags of the choices of the singleton-column rules (options_t, from
// dual_integer_pivots on), shared by the tools that run the engine: gprs_presolve, gprs_quality
// and check/gprs_check.
//   --no-dual-integer-pivots            dual_integer_pivots off
//   --primal-projections                primal_projections on
//   --growth-ratio <all|continuous|integer-exempt>
//   --zero-cost-one-sided               zero_cost_one_sided on
//   --zero-cost-whole-row               zero_cost_whole_row on
//   --singleton-order <index|largest-pivot|zero-cost-first|single-pass>
//   --interval-start <lower|upper|nearest-zero|middle>
//   --interval-order <ascending|descending>
//   --forcing-rounding-units <x>
//   --side-guard-scale <x>
//   --singleton-cols-early              singleton_cols_early on

#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gprs_tools {

constexpr const char* switch_usage =
  "[--no-dual-integer-pivots] [--primal-projections] "
  "[--growth-ratio <all|continuous|integer-exempt>] [--zero-cost-one-sided] "
  "[--zero-cost-whole-row] [--singleton-order <index|largest-pivot|zero-cost-first|single-pass>] "
  "[--interval-start <lower|upper|nearest-zero|middle>] [--interval-order <ascending|descending>] "
  "[--forcing-rounding-units <x>] [--side-guard-scale <x>] "
  "[--singleton-cols-early]";

constexpr const char* growth_ratio_names[]    = {"all", "continuous", "integer-exempt"};
constexpr const char* singleton_order_names[] = {
  "index", "largest-pivot", "zero-cost-first", "single-pass"};
constexpr const char* interval_start_names[] = {"lower", "upper", "nearest-zero", "middle"};
constexpr const char* interval_order_names[] = {"ascending", "descending"};

// The index of `text` among `count` names, or -1.
inline int name_index(const char* const* names, int count, const char* text)
{
  for (int k = 0; k < count; ++k) {
    if (!std::strcmp(names[k], text)) { return k; }
  }
  return -1;
}

// Reads the flag at argv[i] when it is one of the list above, with its value. Returns 1 when
// it took the flag (i is then at its last argument), 0 when argv[i] is none of them, and -1
// for a value that is missing, not a name of the flag or outside the range of the option
// (gprs::choices_in_range).
inline int take_switch(int argc, char** argv, int& i, gprs::options_t& o)
{
  const char* arg = argv[i];
  auto flag       = [arg](const char* name) { return !std::strcmp(arg, name); };
  if (flag("--no-dual-integer-pivots")) {
    o.dual_integer_pivots = false;
    return 1;
  }
  if (flag("--primal-projections")) {
    o.primal_projections = true;
    return 1;
  }
  if (flag("--zero-cost-one-sided")) {
    o.zero_cost_one_sided = true;
    return 1;
  }
  if (flag("--zero-cost-whole-row")) {
    o.zero_cost_whole_row = true;
    return 1;
  }
  if (flag("--singleton-cols-early")) {
    o.singleton_cols_early = true;
    return 1;
  }
  bool named = flag("--growth-ratio") || flag("--singleton-order") || flag("--interval-start") ||
               flag("--interval-order");
  bool number = flag("--forcing-rounding-units") || flag("--side-guard-scale");
  if (!named && !number) { return 0; }
  if (i + 1 >= argc) { return -1; }
  const char* value = argv[++i];
  if (flag("--growth-ratio")) {
    int k = name_index(growth_ratio_names, 3, value);
    if (k < 0) { return -1; }
    o.growth_ratio = (gprs::growth_ratio_t)k;
  } else if (flag("--singleton-order")) {
    int k = name_index(singleton_order_names, 4, value);
    if (k < 0) { return -1; }
    o.singleton_order = (gprs::singleton_order_t)k;
  } else if (flag("--interval-start")) {
    int k = name_index(interval_start_names, 4, value);
    if (k < 0) { return -1; }
    o.interval_start = (gprs::interval_start_t)k;
  } else if (flag("--interval-order")) {
    int k = name_index(interval_order_names, 2, value);
    if (k < 0) { return -1; }
    o.interval_order = (gprs::interval_order_t)k;
  } else if (flag("--forcing-rounding-units")) {
    o.forcing_rounding_units = std::atof(value);
  } else {
    o.side_guard_scale = std::atof(value);
  }
  return gprs::choices_in_range(o) ? 1 : -1;
}

// The options that decide a result, as one JSON object: the mode, the families and reductions
// that are switched off, the limits that have a flag, and the choices of the singleton-column
// rules.
inline void write_options(FILE* out, const gprs::options_t& o)
{
  std::fprintf(out, "{\"primal_only\": %s, \"off\": [", o.primal_only ? "true" : "false");
  const char* separator = "";
  for (int32_t r = 0; r < gprs::rule::count; ++r) {
    if (!o.enabled[r]) {
      std::fprintf(out, "%s\"%s\"", separator, gprs::rule_name(r));
      separator = ", ";
    }
  }
  for (int32_t s = 0; s < gprs::sub_rule::count; ++s) {
    if (!o.sub_enabled[s]) {
      std::fprintf(out, "%s\"%s\"", separator, gprs::sub_rule_name(s));
      separator = ", ";
    }
  }
  // A limit that is switched off is infinite, which JSON has no number for.
  auto number = [out](const char* name, double v) {
    if (std::isfinite(v)) {
      std::fprintf(out, ", \"%s\": %.17g", name, v);
    } else {
      std::fprintf(out, ", \"%s\": \"%s\"", name, v > 0 ? "inf" : v < 0 ? "-inf" : "nan");
    }
  };
  auto flag = [out](const char* name, bool v) {
    std::fprintf(out, ", \"%s\": %s", name, v ? "true" : "false");
  };
  auto name = [out](const char* key, const char* v) {
    std::fprintf(out, ", \"%s\": \"%s\"", key, v);
  };
  std::fprintf(out, "]");
  flag("near_parallel_rows", o.near_parallel_rows);
  number("pivot_growth_limit", o.pivot_growth_limit);
  number("saved_row_factor", o.saved_row_factor);
  number("integral_drift_share", o.tol.integral_drift_share);
  flag("dual_integer_pivots", o.dual_integer_pivots);
  flag("primal_projections", o.primal_projections);
  name("growth_ratio", growth_ratio_names[(int)o.growth_ratio]);
  flag("zero_cost_one_sided", o.zero_cost_one_sided);
  flag("zero_cost_whole_row", o.zero_cost_whole_row);
  name("singleton_order", singleton_order_names[(int)o.singleton_order]);
  name("interval_start", interval_start_names[(int)o.interval_start]);
  name("interval_order", interval_order_names[(int)o.interval_order]);
  number("forcing_rounding_units", o.forcing_rounding_units);
  number("side_guard_scale", o.side_guard_scale);
  flag("singleton_cols_early", o.singleton_cols_early);
  std::fprintf(out, "}");
}

}  // namespace gprs_tools
