/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Generated models through presolve() and oracle::check_presolve: every rule family together,
// each alone, each switched off, primal-only mode, the engine's thresholds at their extremes,
// and a second presolve of the reduced model, which is checked like the first and has nothing
// left to change. A failure prints the case as a test that can be pasted into this file.

#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
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

  // Rows parallel up to singleton columns, equations among them, and little that the cascade
  // settles before the parallel rows see them: no side on an end of the activity range, no
  // fixed column, hardly a short row. First many such rows with narrow integer singleton
  // columns for the most part, then singleton columns that are line columns for the most part.
  o                      = generator_options_t();
  o.p_parallel_row       = 0.65;
  o.p_equation           = 0.45;
  o.p_singleton_col      = 0.8;
  o.p_singleton_line     = 0.25;
  o.max_singleton_cols   = 4;
  o.max_rows             = 8;
  o.max_enum_cols        = 3;
  o.max_side             = 30;
  o.density              = 0.8;
  o.p_forcing_side       = 0.;
  o.p_outside_side       = 0.;
  o.p_fixed_col          = 0.;
  o.p_doubleton_equation = 0.02;
  o.p_singleton_row      = 0.;
  o.p_empty_row          = 0.02;
  list.push_back({"near_parallel", o});

  o.p_parallel_row     = 0.5;
  o.p_singleton_col    = 0.6;
  o.p_singleton_line   = 0.9;
  o.max_singleton_cols = 3;
  o.max_rows           = 6;
  o.max_enum_cols      = 4;
  o.max_line_cols      = 1;
  o.p_infinite_bound   = 0.3;
  list.push_back({"near_parallel_lines", o});

  // What the singleton-column rules act on: up to two singleton columns per row, most of them
  // line columns and half of them without a cost, in rows of which a third are equations, and
  // forcing columns.
  o                       = generator_options_t();
  o.p_singleton_col       = 0.5;
  o.p_singleton_line      = 0.7;
  o.p_singleton_zero_cost = 0.5;
  o.max_singleton_cols    = 4;
  o.max_row_singletons    = 2;
  o.p_forcing_col         = 0.5;
  o.p_equation            = 0.35;
  list.push_back({"columns", o});

  // What simple probing acts on: up to three equations in which a binary carries half of the
  // activity range, over columns that other rows hold, and their near misses.
  o                  = generator_options_t();
  o.p_probing_row    = 0.8;
  o.max_probing_rows = 3;
  o.max_width        = 2;
  o.max_enum_cols    = 6;
  o.max_line_cols    = 1;
  o.p_infinite_bound = 0.2;
  list.push_back({"probing", o});
  return list;
}

// The two profiles with rows that are parallel up to singleton columns.
std::vector<profile_t> near_profiles()
{
  std::vector<profile_t> list;
  for (const profile_t& profile : profiles()) {
    if (profile.options.p_parallel_row >= 0.5 && profile.options.p_singleton_col > 0.) {
      list.push_back(profile);
    }
  }
  return list;
}

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
  if (!options.near_parallel_rows) { out += "  options.near_parallel_rows = false;\n"; }
  for (int32_t s = 0; s < sub_rule::count; ++s) {
    if (options.sub_enabled[s]) { continue; }
    out += "  options.sub_enabled[" + std::to_string(s) + "] = false;\n";
  }
  if (options.pivot_growth_limit != defaults.pivot_growth_limit) {
    out += "  options.pivot_growth_limit = " + real(options.pivot_growth_limit) + ";\n";
  }
  if (options.saved_row_factor != defaults.saved_row_factor) {
    out += "  options.saved_row_factor = " + real(options.saved_row_factor) + ";\n";
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
  run_suite((std::string("only_") + rule_name(family)).c_str(), only(family), 120);
}

void run_without(int32_t family)
{
  run_suite((std::string("without_") + rule_name(family)).c_str(), without(family), 120);
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

// A row that the row rules would remove, found in the model alone: a row without entries, with
// one entry, redundant or forcing on the bounds of its columns. The sums are long doubles, and
// a row counts only where the answer is clear: no term above 1e6, redundant by half the
// feasibility tolerance, forcing with a gap a hundredth of the tolerance of the row rules where
// that tolerance is at least 1e-12. Empty when there is none.
std::string removable_row(const model_t& m)
{
  const double feas = tolerances_t().feasibility;
  for (int32_t i = 0; i < m.nrows; ++i) {
    int64_t begin = m.row_start[i], end = m.row_start[i + 1];
    std::string row = "row " + std::to_string(i) + " of the reduced model ";
    if (end == begin) { return row + "has no entry"; }
    if (end == begin + 1) { return row + "has one entry"; }
    long double low = 0., high = 0.;
    int32_t low_infinite = 0, high_infinite = 0;
    double smallest = inf, largest = 0., largest_term = 0.;
    for (int64_t e = begin; e < end; ++e) {
      double a      = m.value[e];
      int32_t j     = m.col_index[e];
      double at_min = a > 0 ? m.col_lower[j] : m.col_upper[j];
      double at_max = a > 0 ? m.col_upper[j] : m.col_lower[j];
      smallest      = std::fmin(smallest, std::fabs(a));
      largest       = std::fmax(largest, std::fabs(a));
      if (std::isinf(at_min)) {
        ++low_infinite;
      } else {
        low += (long double)a * at_min;
        largest_term = std::fmax(largest_term, std::fabs(a * at_min));
      }
      if (std::isinf(at_max)) {
        ++high_infinite;
      } else {
        high += (long double)a * at_max;
        largest_term = std::fmax(largest_term, std::fabs(a * at_max));
      }
    }
    double lower = m.row_lower[i], upper = m.row_upper[i];
    if (largest_term > 1e6 || std::fabs(lower) > 1e6 || std::fabs(upper) > 1e6) {
      if (!std::isinf(lower) || !std::isinf(upper) || largest_term > 1e6) { continue; }
    }
    bool upper_redundant = upper == inf || (high_infinite == 0 && high <= upper + 0.5 * feas);
    bool lower_redundant = lower == -inf || (low_infinite == 0 && low >= lower - 0.5 * feas);
    if (upper_redundant && lower_redundant) { return row + "is redundant on its bounds"; }
    double tolerance = feas * smallest / std::fmax(1., largest);
    if (tolerance < 1e-12) { continue; }
    if (upper < inf && low_infinite == 0 && std::fabs((double)(low - upper)) <= 0.01 * tolerance) {
      return row + "is forcing at its upper side";
    }
    if (lower > -inf && high_infinite == 0 &&
        std::fabs((double)(high - lower)) <= 0.01 * tolerance) {
      return row + "is forcing at its lower side";
    }
  }
  return {};
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
GPRS_TEST(e2e_only_simple_probing) { run_only(rule::simple_probing); }
GPRS_TEST(e2e_only_singleton_cols) { run_only(rule::singleton_cols); }
GPRS_TEST(e2e_only_dual_cols) { run_only(rule::dual_cols); }
GPRS_TEST(e2e_only_parallel_cols) { run_only(rule::parallel_cols); }

GPRS_TEST(e2e_without_row_rules) { run_without(rule::row_rules); }
GPRS_TEST(e2e_without_propagation) { run_without(rule::propagation); }
GPRS_TEST(e2e_without_tightening) { run_without(rule::tightening); }
GPRS_TEST(e2e_without_doubleton) { run_without(rule::doubleton); }
GPRS_TEST(e2e_without_parallel_rows) { run_without(rule::parallel_rows); }
GPRS_TEST(e2e_without_dual_fixing) { run_without(rule::dual_fixing); }
GPRS_TEST(e2e_without_simple_probing) { run_without(rule::simple_probing); }
GPRS_TEST(e2e_without_singleton_cols) { run_without(rule::singleton_cols); }
GPRS_TEST(e2e_without_dual_cols) { run_without(rule::dual_cols); }
GPRS_TEST(e2e_without_parallel_cols) { run_without(rule::parallel_cols); }

// The reductions of the singleton-column rules, each as the only one of the column rules and
// each switched off, and the two limits of the family at their tightest.
GPRS_TEST(e2e_singleton_cols_reductions)
{
  for (int32_t sub : {sub_rule::free_singleton,
                      sub_rule::zero_cost_singleton,
                      sub_rule::forcing_column,
                      sub_rule::free_singleton_dual}) {
    options_t alone;
    alone.sub_enabled.fill(false);
    alone.sub_enabled[sub] = true;
    run_suite((std::string("only_") + sub_rule_name(sub)).c_str(), alone, 60);
    options_t off;
    off.sub_enabled[sub] = false;
    run_suite((std::string("without_") + sub_rule_name(sub)).c_str(), off, 60);
  }
  options_t tight;
  tight.pivot_growth_limit = 1.;
  tight.saved_row_factor   = 0.;
  run_suite("tight_singleton_limits", tight, 60);
}

// Fixed model 6: 2 k + y = 4 with an integer singleton column k in [0, 2] and y integer in
// [0, 3]. The row is no definition of k: y = 1 would give 1.5. And 2 k + 2 y = 3, which has no
// integer point, with the tightening family off, which would say so.
GPRS_TEST(e2e_case_integer_singleton_column_needs_an_integral_row)
{
  model_t base;
  base.nrows            = 2;
  base.ncols            = 3;
  base.row_start        = {0, 2, 4};
  base.col_index        = {0, 1, 1, 2};
  base.value            = {2., 1., 1., 1.};
  base.row_lower        = {4., -inf};
  base.row_upper        = {4., 4.};
  base.col_lower        = {0., 0., 0.};
  base.col_upper        = {2., 3., 3.};
  base.cost             = {1., -1., -1.};
  base.col_type         = {1, 1, 1};
  base.objective_offset = 0.;
  for (bool primal_only : {false, true}) {
    options_t options;
    options.primal_only = primal_only;
    result_t result     = presolve(base, options);
    GPRS_CHECK_EQ(result.sub_findings[sub_rule::free_singleton], 0);
    oracle::verdict_t verdict =
      oracle::check_result(oracle::make_case(base, {}, {}), options, result);
    if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }

    model_t odd                       = base;
    odd.value[1]                      = 2.;
    odd.row_lower[0]                  = 3.;
    odd.row_upper[0]                  = 3.;
    options_t without                 = options;
    without.enabled[rule::tightening] = false;
    result                            = presolve(odd, without);
    verdict = oracle::check_result(oracle::make_case(odd, {}, {}), without, result);
    if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
  }
}

// Found by the campaign (profile dense, seed 200045869, variant 9): the zero-cost singleton
// column x0 with a lower bound of -1.28e9 after scaling, whose undo stepped from that bound to
// 597.33 and lost the digits that row 6, parallel to its row at 2^18 times the scale, reads.
GPRS_TEST(e2e_case_interval_undo_far_from_prefer)
{
  model_t base;
  base.nrows            = 7;
  base.ncols            = 3;
  base.row_start        = {0, 2, 5, 7, 9, 12, 12, 15};
  base.col_index        = {0, 1, 0, 1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2};
  base.value            = {2., -3., -2., 1., 1., 3., 3., 2., -3., -3., -3., 2., -3., -3., 2.};
  base.row_lower        = {-inf, -10., -inf, -inf, -12., 0., -12.};
  base.row_upper        = {10., inf, inf, 10., -12., 0., -12.};
  base.col_lower        = {-10000000., 0., 0.};
  base.col_upper        = {5., 2., 2.};
  base.cost             = {0., 0., 0.};
  base.col_type         = {0, 1, 1};
  base.objective_offset = 3.;
  oracle::case_t c      = oracle::make_case(
    base, {0.015625, 0.125, 0.5, 0.5, 0.00390625, 64., 1024.}, {0.0078125, 1., 1.});
  options_t options;
  options.max_rounds            = 1;
  options.max_cascade_passes    = 2;
  options.parallel_bucket_limit = 1;
  for (const options_t& o : {options, options_t()}) {
    result_t result = presolve(c.model, o);
    GPRS_CHECK_EQ(result.sub_findings[sub_rule::zero_cost_singleton], 1);
    oracle::verdict_t verdict = oracle::check_result(c, o, result);
    if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
  }
}

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

// Parallel rows as they were before rows parallel up to singleton columns.
GPRS_TEST(e2e_near_parallel_rows_off)
{
  options_t options;
  options.near_parallel_rows = false;
  run_suite("near_parallel_rows_off", options, 120);
}

// The suites reach the subtraction of an equation from a row that is parallel to it up to
// singleton columns on the two profiles made for it, with every rule and more often with the
// parallel rows alone or without the bound propagation, and the oracle decides the reduced
// models: also those in which the row is left with two line columns, which it takes back into
// its class.
GPRS_TEST(e2e_near_parallel_rows_are_reached)
{
  options_t without_propagation = without(rule::propagation);
  options_t off;
  off.near_parallel_rows = false;
  int64_t with_all = 0, with_few = 0, models_with = 0, two_line_columns = 0, switched_off = 0;
  int32_t failures = 0;
  for (const profile_t& profile : near_profiles()) {
    for (uint64_t seed = 6000000; seed < 6000500 && failures < max_reports; ++seed) {
      case_t c = oracle::generate(seed, profile.options);
      for (const options_t& options :
           {options_t(), only(rule::parallel_rows), without_propagation}) {
        result_t r  = presolve(c.model, options);
        verdict_t v = oracle::check_result(c, options, r);
        if (!v.ok) {
          ++failures;
          report(std::string("reached, profile ") + profile.name + ", seed " + std::to_string(seed),
                 std::string("reached_") + profile.name + "_" + std::to_string(seed),
                 v.message,
                 oracle::to_code(c),
                 options,
                 check_once);
          continue;
        }
        size_t n = r.postsolve.row_additions.size();
        (options.enabled[rule::row_rules] && options.enabled[rule::propagation] ? with_all
                                                                                : with_few) += n;
        models_with += n > 0;
        std::vector<uint8_t> line;
        two_line_columns +=
          has_model(r.status) && n > 0 &&
          !oracle::classify_widest(r.reduced, oracle::check_max_assignments, line);
      }
      switched_off += presolve(c.model, off).postsolve.row_additions.size();
    }
  }
  GPRS_CHECK(with_all > 40);
  GPRS_CHECK(with_few > 400);
  GPRS_CHECK(models_with > 300);
  GPRS_CHECK(two_line_columns > 8);
  GPRS_CHECK_EQ(switched_off, 0);
}

// Rows x0 + x1 + x2 = 1 and x0 + x1 + x2 + s0 + s1 + s2 = 4: the first is subtracted from the
// second once, after which neither row has a core. Subtracting the equation with the singleton
// columns from the other one instead puts those columns into both rows, the next round takes
// them out again, and the rounds end only at their limit.
GPRS_TEST(e2e_case_near_parallel_rows_end)
{
  for (uint8_t type : {0, 1}) {
    model_t base;
    base.nrows            = 2;
    base.ncols            = 6;
    base.row_start        = {0, 3, 9};
    base.col_index        = {0, 1, 2, 0, 1, 2, 3, 4, 5};
    base.value            = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
    base.row_lower        = {1., 4.};
    base.row_upper        = {1., 4.};
    base.col_lower        = {0., 0., 0., 0., 0., 0.};
    base.col_upper        = {1., 1., 1., 5., 5., 5.};
    base.cost             = {1., 2., 3., -1., 2., 0.};
    base.col_type         = {1, 1, 1, type, type, type};
    base.objective_offset = 0.;
    // Without the singleton-column rules the two rows are what is left; with them the row of
    // the singleton columns is taken apart as well, and the rounds still end early.
    for (bool singleton_cols : {false, true}) {
      options_t options;
      options.max_rounds                    = 200;
      options.enabled[rule::singleton_cols] = singleton_cols;
      result_t result                       = presolve(base, options);
      GPRS_CHECK(result.status == status_t::reduced);
      GPRS_CHECK(result.rounds <= (singleton_cols ? 8 : 4));
      GPRS_CHECK_EQ(result.postsolve.row_additions.size(), 1u);
      if (!singleton_cols) { GPRS_CHECK_EQ(result.reduced.nnz(), 6); }
      // Three continuous columns in one row are outside the oracle class; integer ones are
      // enumerated.
      if (type == 1) {
        oracle::verdict_t verdict =
          oracle::check_result(oracle::make_case(base, {}, {}), options, result);
        if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
      }
    }
  }
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

// Cases that longer runs of the suites above found (seeds from 60000000 on).

// Dual fixing puts x1 at its bound of 1e22, and rows 0 and 2 are parallel on what is left. A
// side of row 0 is stored with the term of x1 in it: the lower side that row 2 gives it was
// lost in that term and the presolve returned infeasible.
GPRS_TEST(e2e_case_all_rules_dense_101108360)
{
  model_t base;
  base.nrows            = 3;
  base.ncols            = 4;
  base.row_start        = {0, 3, 6, 8};
  base.col_index        = {0, 1, 2, 0, 1, 2, 0, 2};
  base.value            = {3., 3., -3., 3., -1., 3., -2., 2.};
  base.row_lower        = {3., -inf, -inf};
  base.row_upper        = {inf, 3., 5.};
  base.col_lower        = {0., -inf, 3., 0.};
  base.col_upper        = {2., 1e+22, 4., 2.};
  base.cost             = {-3., 0., -2., 0.};
  base.col_type         = {1, 1, 1, 1};
  base.objective_offset = 0.;

  oracle::case_t c = oracle::make_case(base, {0.00390625, 0.125, 0.00390625}, {1., 1., 1., 1.});
  options_t options;
  oracle::verdict_t verdict = oracle::check_presolve(c, options);
  if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
}

// The same with x2 fixed at -2e20 and no rebuild before the parallel rows: row 4 gave row 2 the
// lower side 3 - 4e20, which is -4e20, and the reduced model had the optimum 6 instead of 6.5.
GPRS_TEST(e2e_case_rebuild_never_open_ends_60163257)
{
  model_t base;
  base.nrows            = 5;
  base.ncols            = 8;
  base.row_start        = {0, 2, 5, 9, 14, 16};
  base.col_index        = {3, 7, 0, 1, 6, 0, 2, 5, 6, 0, 1, 4, 5, 6, 0, 6};
  base.value            = {-2., -3., 2., -2., 1., 3., 2., -3., -3., -1., 1., 1., -2., 3., 2., -2.};
  base.row_lower        = {3., -inf, -inf, -inf, 1.};
  base.row_upper        = {3., 3., 12., inf, inf};
  base.col_lower        = {3., -inf, -2e+20, 0., 0., -3., 0., -inf};
  base.col_upper        = {5., inf, -2., 3., 1., 0., 3., inf};
  base.cost             = {0., 3., 0., 0., -2., 0., -2., 0.};
  base.col_type         = {1, 0, 1, 1, 1, 1, 1, 1};
  base.objective_offset = 5.;

  oracle::case_t c = oracle::make_case(base, {}, {});
  options_t options;
  options.rebuild_dead_fraction  = 2.;
  options.rebuild_fixed_fraction = 2.;
  oracle::verdict_t verdict      = oracle::check_presolve(c, options);
  if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
}

// The lower bound of x0 comes down from -1e15 to -2, and coefficient strengthening sets its
// coefficient to 1 plus the error bound of the activity, one rounding step above 1. x0 is an
// integer line column of the reduced model: the oracle rounded the end that the row gives it
// inwards without its tolerance and found the optimum 0 instead of -2.
GPRS_TEST(e2e_case_all_rules_open_ends_60152816)
{
  model_t base;
  base.nrows            = 3;
  base.ncols            = 3;
  base.row_start        = {0, 3, 6, 9};
  base.col_index        = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  base.value            = {-2., -3., -1., 2., 3., 1., 2., 3., 1.};
  base.row_lower        = {-inf, 3., 3.};
  base.row_upper        = {-3., inf, inf};
  base.col_lower        = {-1000000000000000., 1., 0.};
  base.col_upper        = {inf, 2., 2.};
  base.cost             = {2., 0., 2.};
  base.col_type         = {1, 1, 1};
  base.objective_offset = 0.;

  oracle::case_t c = oracle::make_case(base, {}, {});
  options_t options;
  oracle::verdict_t verdict = oracle::check_presolve(c, options);
  if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
}

// Cases that a longer run with rows parallel up to singleton columns next to large fixed
// columns found (seeds from 7200000000 on, models with rows and columns added to the generated
// ones).

// Row 0, 3 x0 + 3 x1 + z = 30000009 with z fixed at 3e7, is divided by 3 in the row tightening:
// the coefficient of z is 1 / 3 as a double, and x0 + x1 = 3 holds with an error of 5.6e-10,
// far below the tolerance. Row 1 has the same core 3072 times larger and the continuous
// singleton column y. Subtracting row 0 from it with the factor 3072 left 1024 y = 2048 - 1.7e-6
// and the postsolved point violated row 1 by 1.7e-6.
GPRS_TEST(e2e_case_near_parallel_rows_scaled_equation)
{
  model_t base;
  base.nrows            = 2;
  base.ncols            = 4;
  base.row_start        = {0, 3, 6};
  base.col_index        = {0, 1, 3, 0, 1, 2};
  base.value            = {3., 3., 1., 3072., 3072., 1024.};
  base.row_lower        = {30000009., 11264.};
  base.row_upper        = {30000009., 11264.};
  base.col_lower        = {0., 0., 0., 30000000.};
  base.col_upper        = {2., 2., 10., 30000000.};
  base.cost             = {0., 0., 1., 0.};
  base.col_type         = {1, 1, 0, 1};
  base.objective_offset = 0.;

  oracle::case_t c = oracle::make_case(base, {}, {});
  options_t options;
  oracle::verdict_t verdict = oracle::check_presolve(c, options);
  if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
}

// The other way round: row 1 is the row that the tightening has scaled next to x3, fixed at
// 9e7, and row 0 has the same core 3072 times larger and the continuous singleton column x4.
// Row 0 was subtracted from row 1 with the factor -1 / 3072, the bound that row 1 then gave x4
// went into row 0 times 2048, and the presolve returned infeasible.
GPRS_TEST(e2e_case_near_parallel_rows_scaled_target)
{
  model_t base;
  base.nrows            = 3;
  base.ncols            = 5;
  base.row_start        = {0, 3, 7, 9};
  base.col_index        = {0, 1, 4, 0, 1, 2, 3, 0, 1};
  base.value            = {3., 3., -2., -6., -6., -1., -1., 6., 6.};
  base.row_lower        = {-20., -89999988., -12.};
  base.row_upper        = {-20., inf, -12.};
  base.col_lower        = {-3., 0., -2., 90000000., -1.};
  base.col_upper        = {-2., 1., 0., 90000000., 9.};
  base.cost             = {1., 1., 3., -1., 3.};
  base.col_type         = {1, 1, 1, 1, 0};
  base.objective_offset = 0.;

  oracle::case_t c = oracle::make_case(base, {1024., 1., 0.25}, {1., 1., 1., 1., 1.});
  options_t options;
  oracle::verdict_t verdict = oracle::check_presolve(c, options);
  if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
}

// The factor is 1, and the target has the continuous singleton column y with the coefficient
// 2^-21. Row 0 is again the row that the tightening has scaled, off by 5.6e-10: the bound that
// the target gave y after the subtraction was off by 2^21 times that, and the optimal point of
// the model, crushed, violated it by 1.2e-3.
GPRS_TEST(e2e_case_near_parallel_rows_small_singleton_coefficient)
{
  model_t base;
  base.nrows            = 2;
  base.ncols            = 4;
  base.row_start        = {0, 3, 6};
  base.col_index        = {0, 1, 3, 0, 1, 2};
  base.value            = {3., 3., 1., 1., 1., 1.};
  base.row_lower        = {30000009., -inf};
  base.row_upper        = {30000009., 4.};
  base.col_lower        = {0., 0., 0., 30000000.};
  base.col_upper        = {2., 2., 3., 30000000.};
  base.cost             = {0., 0., -1., 0.};
  base.col_type         = {1, 1, 0, 1};
  base.objective_offset = 0.;

  oracle::case_t c = oracle::make_case(base, {}, {1., 1., 0x1p-21, 1.});
  options_t options;
  options.primal_only       = true;
  oracle::verdict_t verdict = oracle::check_presolve(c, options);
  if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
}

// No row is changed by the presolve here (the tightening is off). Row 0 is 26.4 times
// x0 + x1 - w = -99997 with w fixed at 1e5: as doubles, its right-hand side less its fixed term
// is 79.2 up to the rounding of 2.6e6, 4e-11. Row 1 has the same core 3072000 / 26.4 times
// larger: after the subtraction it was off by 5e-6, and so was the postsolved point.
GPRS_TEST(e2e_case_near_parallel_rows_large_factor)
{
  model_t base;
  base.nrows            = 2;
  base.ncols            = 4;
  base.row_start        = {0, 3, 6};
  base.col_index        = {0, 1, 3, 0, 1, 2};
  base.value            = {1., 1., -1., 3000., 3000., 1.};
  base.row_lower        = {-99997., 9002.};
  base.row_upper        = {-99997., 9002.};
  base.col_lower        = {0., 0., 0., 100000.};
  base.col_upper        = {2., 2., 10., 100000.};
  base.cost             = {0., 0., 1., 0.};
  base.col_type         = {1, 1, 0, 1};
  base.objective_offset = 0.;

  oracle::case_t c = oracle::make_case(base, {1., 1024.}, {});
  // Row 0 times a factor that is no power of two.
  constexpr double factor = 26.4;
  for (int64_t e = c.model.row_start[0]; e < c.model.row_start[1]; ++e) {
    c.model.value[e] *= factor;
  }
  c.model.row_lower[0] *= factor;
  c.model.row_upper[0] *= factor;
  c.row_scale[0] *= factor;
  options_t options;
  options.enabled[rule::tightening] = false;
  oracle::verdict_t verdict         = oracle::check_presolve(c, options);
  if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
}

// Row 1, x0 + x1 + x2 + y = 4, is as the model gave it, and row 2 has its core 4096 times
// larger: subtracted with the factor 4096, row 1 is from then on what row 2 says of the core,
// times 4096. y is its singleton column now, and row 0, which the tightening has scaled (off by
// 5.6e-10), was subtracted from it with the factor 1: y = 1 - 5.6e-10, right for row 1 and
// 2.3e-6 off for row 2. A subtraction with a factor above 1 makes the equation carry a row
// that is stricter than itself.
GPRS_TEST(e2e_case_near_parallel_rows_equation_changed_later)
{
  model_t base;
  base.nrows            = 3;
  base.ncols            = 6;
  base.row_start        = {0, 4, 8, 13};
  base.col_index        = {0, 1, 2, 5, 0, 1, 2, 3, 0, 1, 2, 3, 4};
  base.value            = {3., 3., 3., 1., 1., 1., 1., 1., 4., 4., 4., 4., 1.};
  base.row_lower        = {30000009., 4., 18.};
  base.row_upper        = {30000009., 4., 18.};
  base.col_lower        = {0., 0., 0., 0., 0., 30000000.};
  base.col_upper        = {3., 3., 3., 10., 3., 30000000.};
  base.cost             = {0., 0., 0., 1., 0., 0.};
  base.col_type         = {1, 1, 1, 0, 1, 1};
  base.objective_offset = 0.;

  oracle::case_t c = oracle::make_case(base, {1., 1., 1024.}, {});
  options_t options;
  oracle::verdict_t verdict = oracle::check_presolve(c, options);
  if (!verdict.ok) { test::fail(__FILE__, __LINE__, verdict.message); }
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

// The reduced model is a fixed point of the rules: with result_t::fixed_point set it has no row
// that the row rules would remove, and a second presolve with the same options returns it
// unchanged, in every array. The test above only checks that a second presolve is correct.
GPRS_TEST(e2e_reduced_model_is_a_fixed_point)
{
  const char* check_fixed =
    "  result_t first = presolve(c.model, options);\n"
    "  result_t second = presolve(first.reduced, options);\n"
    "  oracle::verdict_t verdict;\n"
    "  verdict.ok = !first.fixed_point || second.status == status_t::unchanged;\n"
    "  verdict.message = \"the second presolve changes the reduced model\";\n";
  int64_t with_model = 0, fixed = 0, reduced = 0;
  for (int32_t mode = 0; mode < 4; ++mode) {
    options_t options;
    options.primal_only = mode == 1;
    if (mode >= 2) {
      // No rebuild between the phases, and one after every phase.
      options.rebuild_dead_fraction  = mode == 2 ? 2. : 0.;
      options.rebuild_fixed_fraction = mode == 2 ? 2. : 0.;
    }
    int32_t failures = 0;
    for (const profile_t& profile : profiles()) {
      for (uint64_t seed = 5200000; seed < 5200300 && failures < max_reports; ++seed) {
        case_t c       = oracle::generate(seed, profile.options);
        result_t first = presolve(c.model, options);
        if (!has_model(first.status)) { continue; }
        ++with_model;
        if (!first.fixed_point) { continue; }
        ++fixed;
        reduced += first.status == status_t::reduced;
        std::string problem = removable_row(first.reduced);
        if (problem.empty()) {
          result_t second = presolve(first.reduced, options);
          if (second.status != status_t::unchanged || !same_model(second.reduced, first.reduced)) {
            problem = std::string("the second presolve returns ") + status_name(second.status);
            for (int32_t r = 0; r < rule::count; ++r) {
              if (second.stats[r].findings == 0) { continue; }
              problem +=
                std::string(", ") + rule_name(r) + " " + std::to_string(second.stats[r].findings);
            }
          }
        }
        if (problem.empty()) { continue; }
        ++failures;
        report(
          std::string("fixed point, profile ") + profile.name + ", seed " + std::to_string(seed),
          std::string("fixed_point_") + profile.name + "_" + std::to_string(seed),
          problem,
          oracle::to_code(c),
          options,
          check_fixed);
      }
    }
  }
  // The property is not an empty one: with the default limits nearly every presolve of these
  // models ends at a fixed point, and many of them return a reduced model that is not empty.
  GPRS_CHECK(with_model > 4000);
  GPRS_CHECK(fixed * 100 >= with_model * 99);
  GPRS_CHECK(reduced > 1000);
}
