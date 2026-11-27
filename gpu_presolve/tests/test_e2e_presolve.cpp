/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Generated models through presolve() and oracle::check_presolve: every rule family together,
// each alone, each switched off, primal-only mode, the engine's thresholds at their extremes,
// and a second presolve of the reduced model. A failure prints the case as a test that can be
// pasted into this file.

#include "harness.hpp"
#include "oracle.hpp"

#include <cstdio>

using namespace gprs;
using oracle::case_t;
using oracle::generator_options_t;
using oracle::verdict_t;

namespace {

struct profile_t {
  const char* name;
  generator_options_t options;
};

std::vector<profile_t> profiles()
{
  std::vector<profile_t> list;
  generator_options_t o;
  list.push_back({"default", o});

  o               = generator_options_t();
  o.max_line_cols = 0;
  o.max_enum_cols = 6;
  o.max_rows      = 6;
  list.push_back({"integer", o});

  o                      = generator_options_t();
  o.p_doubleton_equation = 0.45;
  o.p_equation           = 0.4;
  o.p_singleton_row      = 0.05;
  o.p_empty_row          = 0.02;
  o.p_zero_cost          = 0.5;
  list.push_back({"equations", o});

  o                = generator_options_t();
  o.p_parallel_row = 0.5;
  o.max_rows       = 7;
  list.push_back({"parallel", o});

  o          = generator_options_t();
  o.p_scaled = 1.;
  list.push_back({"scaled", o});

  o                  = generator_options_t();
  o.max_line_cols    = 3;
  o.p_infinite_bound = 0.7;
  o.p_huge_bound     = 0.6;
  o.p_zero_cost      = 0.5;
  list.push_back({"open_ends", o});

  o                 = generator_options_t();
  o.density         = 0.9;
  o.max_rows        = 8;
  o.max_enum_cols   = 4;
  o.max_width       = 2;
  o.p_singleton_row = 0.05;
  o.p_empty_row     = 0.02;
  o.p_forcing_side  = 0.3;
  list.push_back({"dense", o});

  o                 = generator_options_t();
  o.max_width       = 1;
  o.max_side        = 6;
  o.max_coefficient = 4;
  o.max_line_cols   = 1;
  o.p_common_factor = 0.4;
  o.p_fixed_col     = 0.05;
  list.push_back({"binary", o});
  return list;
}

const char* const family_name[rule::count] = {
  "row_rules", "propagation", "tightening", "doubleton", "parallel_rows", "dual_fixing"};

std::string real(double v)
{
  char text[40];
  std::snprintf(text, sizeof(text), "%.17g", v);
  return text;
}

// The fields of the options that the suites of this file change, as statements.
std::string options_code(const options_t& options)
{
  options_t defaults;
  std::string out = "  options_t options;\n";
  if (options.primal_only) { out += "  options.primal_only = true;\n"; }
  if (options.phase_hashes) { out += "  options.phase_hashes = true;\n"; }
  if (options.enabled != defaults.enabled) {
    out += "  options.enabled = {";
    for (int32_t r = 0; r < rule::count; ++r) {
      out += std::string(r > 0 ? ", " : "") + (options.enabled[r] ? "true" : "false");
    }
    out += "};\n";
  }
  for (int32_t r = 0; r < rule::count; ++r) {
    const budget_t& b = options.budget[r];
    const budget_t& d = defaults.budget[r];
    if (b.passes == d.passes && b.stall_passes == d.stall_passes && b.floor == d.floor) {
      continue;
    }
    out += "  options.budget[" + std::to_string(r) + "] = {" + real(b.passes) + ", " +
           real(b.stall_passes) + ", " + std::to_string(b.floor) + "};\n";
  }
  if (options.max_rounds != defaults.max_rounds) {
    out += "  options.max_rounds = " + std::to_string(options.max_rounds) + ";\n";
  }
  if (options.max_cascade_passes != defaults.max_cascade_passes) {
    out += "  options.max_cascade_passes = " + std::to_string(options.max_cascade_passes) + ";\n";
  }
  if (options.rebuild_dead_fraction != defaults.rebuild_dead_fraction) {
    out += "  options.rebuild_dead_fraction = " + real(options.rebuild_dead_fraction) + ";\n";
  }
  if (options.rebuild_fixed_fraction != defaults.rebuild_fixed_fraction) {
    out += "  options.rebuild_fixed_fraction = " + real(options.rebuild_fixed_fraction) + ";\n";
  }
  if (options.parallel_bucket_limit != defaults.parallel_bucket_limit) {
    out +=
      "  options.parallel_bucket_limit = " + std::to_string(options.parallel_bucket_limit) + ";\n";
  }
  return out;
}

void report(const std::string& where,
            const std::string& test_name,
            const std::string& message,
            const std::string& case_code,
            const options_t& options,
            const char* check)
{
  std::string text = where + ": " + message + "\nGPRS_TEST(e2e_case_" + test_name + ")\n{\n" +
                     case_code + options_code(options) + check +
                     "  if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }\n}";
  test::fail(__FILE__, __LINE__, text);
}

const char* const check_once =
  "  oracle::verdict_t verdict = oracle::check_presolve(c, options);\n";

constexpr uint64_t seed_blocks[] = {1000000, 2000000, 3000000};
constexpr int32_t max_reports    = 3;

// count cases per profile and seed block; stops after max_reports failures.
void run_suite(const char* suite, const options_t& options, int32_t count)
{
  int32_t failures = 0;
  for (const profile_t& profile : profiles()) {
    for (uint64_t block : seed_blocks) {
      for (uint64_t seed = block; seed < block + count; ++seed) {
        case_t c    = oracle::generate(seed, profile.options);
        verdict_t v = oracle::check_presolve(c, options);
        if (v.ok) { continue; }
        std::string id = std::string(profile.name) + "_" + std::to_string(seed);
        report(std::string(suite) + ", profile " + profile.name + ", seed " + std::to_string(seed),
               std::string(suite) + "_" + id,
               v.message,
               oracle::to_code(c),
               options,
               check_once);
        if (++failures == max_reports) { return; }
      }
    }
  }
}

options_t only(int32_t family)
{
  options_t options;
  options.enabled.fill(false);
  options.enabled[family] = true;
  return options;
}

options_t without(int32_t family)
{
  options_t options;
  options.enabled[family] = false;
  return options;
}

void run_only(int32_t family)
{
  run_suite((std::string("only_") + family_name[family]).c_str(), only(family), 120);
}

void run_without(int32_t family)
{
  run_suite((std::string("without_") + family_name[family]).c_str(), without(family), 120);
}

bool same_model(const model_t& a, const model_t& b)
{
  return a.nrows == b.nrows && a.ncols == b.ncols && a.row_start == b.row_start &&
         a.col_index == b.col_index && a.value == b.value && a.row_lower == b.row_lower &&
         a.row_upper == b.row_upper && a.col_lower == b.col_lower && a.col_upper == b.col_upper &&
         a.cost == b.cost && a.col_type == b.col_type && a.objective_offset == b.objective_offset;
}

bool has_model(status_t status)
{
  return status == status_t::reduced || status == status_t::unchanged ||
         status == status_t::reduced_to_empty || status == status_t::limit;
}

}  // namespace

GPRS_TEST(e2e_all_rules) { run_suite("all_rules", options_t(), 400); }

GPRS_TEST(e2e_primal_only)
{
  options_t options;
  options.primal_only = true;
  run_suite("primal_only", options, 300);
}

GPRS_TEST(e2e_only_row_rules) { run_only(rule::row_rules); }
GPRS_TEST(e2e_only_propagation) { run_only(rule::propagation); }
GPRS_TEST(e2e_only_tightening) { run_only(rule::tightening); }
GPRS_TEST(e2e_only_doubleton) { run_only(rule::doubleton); }
GPRS_TEST(e2e_only_parallel_rows) { run_only(rule::parallel_rows); }
GPRS_TEST(e2e_only_dual_fixing) { run_only(rule::dual_fixing); }

GPRS_TEST(e2e_without_row_rules) { run_without(rule::row_rules); }
GPRS_TEST(e2e_without_propagation) { run_without(rule::propagation); }
GPRS_TEST(e2e_without_tightening) { run_without(rule::tightening); }
GPRS_TEST(e2e_without_doubleton) { run_without(rule::doubleton); }
GPRS_TEST(e2e_without_parallel_rows) { run_without(rule::parallel_rows); }
GPRS_TEST(e2e_without_dual_fixing) { run_without(rule::dual_fixing); }

// The start step alone.
GPRS_TEST(e2e_no_rule)
{
  options_t options;
  options.enabled.fill(false);
  run_suite("no_rule", options, 100);
}

// The thresholds of the engine at their extremes, so that every path runs on tiny models.
GPRS_TEST(e2e_thresholds_rebuild_always)
{
  options_t options;
  options.rebuild_dead_fraction  = 0.;
  options.rebuild_fixed_fraction = 0.;
  run_suite("rebuild_always", options, 80);
}

GPRS_TEST(e2e_thresholds_rebuild_never)
{
  options_t options;
  options.rebuild_dead_fraction  = 2.;
  options.rebuild_fixed_fraction = 2.;
  run_suite("rebuild_never", options, 80);
}

GPRS_TEST(e2e_thresholds_one_round)
{
  options_t options;
  options.max_rounds = 1;
  run_suite("one_round", options, 80);
  options.max_cascade_passes = 1;
  run_suite("one_cascade_pass", options, 80);
}

// A family that stops on its budget leaves a valid state, whatever part of its list is left.
GPRS_TEST(e2e_thresholds_small_budgets)
{
  options_t options;
  for (budget_t& b : options.budget) {
    b = budget_t{0., 0., 20};
  }
  run_suite("budget_20_units", options, 80);
  for (budget_t& b : options.budget) {
    b = budget_t{0.25, 0.125, 0};
  }
  run_suite("budget_quarter_pass", options, 80);
  for (budget_t& b : options.budget) {
    b = budget_t{0., 0., 0};
  }
  run_suite("budget_zero", options, 40);
}

GPRS_TEST(e2e_thresholds_parallel_bucket_of_one)
{
  options_t options;
  options.parallel_bucket_limit = 1;
  run_suite("parallel_bucket_of_one", options, 80);
}

// The same model gives the same result twice.
GPRS_TEST(e2e_presolve_is_repeatable)
{
  options_t options;
  options.phase_hashes = true;
  int32_t failures     = 0;
  for (const profile_t& profile : profiles()) {
    for (uint64_t seed = 4000000; seed < 4000100 && failures < max_reports; ++seed) {
      case_t c   = oracle::generate(seed, profile.options);
      result_t a = presolve(c.model, options);
      result_t b = presolve(c.model, options);
      bool same  = a.status == b.status && a.hash == b.hash && a.rounds == b.rounds;
      same       = same && a.rebuilds == b.rebuilds && a.phase_hash == b.phase_hash;
      same       = same && (!has_model(a.status) || same_model(a.reduced, b.reduced));
      same       = same &&
             (!has_model(a.status) || postsolve_hash(a.postsolve) == postsolve_hash(b.postsolve));
      same = same && a.postsolve.col_origin == b.postsolve.col_origin &&
             a.postsolve.row_origin == b.postsolve.row_origin;
      if (same) { continue; }
      ++failures;
      report(std::string("repeatable, profile ") + profile.name + ", seed " + std::to_string(seed),
             std::string("repeatable_") + profile.name + "_" + std::to_string(seed),
             "two presolves of one model differ",
             oracle::to_code(c),
             options,
             check_once);
    }
  }
}

// Multi-round sequence: the reduced model is a model like any other. Its data need not be
// integers, so the oracle compares with its tolerance here.
GPRS_TEST(e2e_second_presolve_of_the_reduced_model)
{
  const char* check_twice =
    "  result_t first = presolve(c.model, options);\n"
    "  oracle::case_t second = oracle::make_case(first.reduced, {}, {});\n"
    "  oracle::verdict_t verdict = oracle::check_presolve(second, options);\n";
  for (int32_t mode = 0; mode < 2; ++mode) {
    options_t options;
    options.primal_only = mode == 1;
    int32_t failures    = 0;
    for (const profile_t& profile : profiles()) {
      for (uint64_t seed = 5000000; seed < 5000150 && failures < max_reports; ++seed) {
        case_t c       = oracle::generate(seed, profile.options);
        result_t first = presolve(c.model, options);
        if (!has_model(first.status) || !well_formed(first.reduced)) { continue; }
        std::vector<uint8_t> line;
        if (!oracle::classify_widest(first.reduced, oracle::check_max_assignments, line)) {
          continue;
        }
        verdict_t v = oracle::check_presolve(oracle::make_case(first.reduced, {}, {}), options);
        if (v.ok) { continue; }
        ++failures;
        report(std::string("second presolve, profile ") + profile.name + ", seed " +
                 std::to_string(seed),
               std::string("second_presolve_") + profile.name + "_" + std::to_string(seed),
               v.message,
               oracle::to_code(c),
               options,
               check_twice);
      }
    }
  }
}
