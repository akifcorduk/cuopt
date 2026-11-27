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
                std::string(", ") + family_name[r] + " " + std::to_string(second.stats[r].findings);
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
