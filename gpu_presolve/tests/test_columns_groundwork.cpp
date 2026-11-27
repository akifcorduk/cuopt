/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The groundwork of the column rules: the four families in the round, their options and
// statistics, and the dual findings of a result. While the families hold no reduction, a
// presolve with them is the presolve without them.

#include "harness.hpp"
#include "oracle.hpp"
#include "test_engine_support.hpp"

#include <gprs/presolve.hpp>

#include <string>
#include <vector>

using namespace gprs;
using namespace gprs::engine_test;

namespace {

options_t without_column_rules(options_t options)
{
  for (int32_t r = rule::simple_probing; r <= rule::parallel_cols; ++r) {
    options.enabled[r] = false;
  }
  return options;
}

// Everything of a result but its seconds, and for the column families but their calls.
std::string difference(const result_t& a, const result_t& b)
{
  if (a.status != b.status) { return "status"; }
  if (a.hash != b.hash) { return "hash of the reduced model"; }
  if (postsolve_hash(a.postsolve) != postsolve_hash(b.postsolve)) { return "postsolve stack"; }
  if (a.fixed_point != b.fixed_point) { return "fixed-point flag"; }
  if (a.rounds != b.rounds || a.explicit_rounds != b.explicit_rounds) { return "rounds"; }
  if (a.rebuilds != b.rebuilds) { return "rebuilds"; }
  if (a.dual_findings != b.dual_findings) { return "dual findings"; }
  for (int32_t r = 0; r < rule::simple_probing; ++r) {
    const rule_stats_t& s = a.stats[r];
    const rule_stats_t& t = b.stats[r];
    if (s.calls != t.calls || s.work != t.work || s.findings != t.findings ||
        s.records != t.records || s.rows_removed != t.rows_removed ||
        s.cols_removed != t.cols_removed || s.nnz_removed != t.nnz_removed) {
      return std::string("statistics of ") + rule_name(r);
    }
  }
  return std::string();
}

}  // namespace

GPRS_TEST(columns_options_and_names)
{
  options_t options;
  for (int32_t r = 0; r < rule::count; ++r) {
    GPRS_CHECK(options.enabled[r]);
  }
  for (int32_t s = 0; s < sub_rule::count; ++s) {
    GPRS_CHECK(options.sub_enabled[s]);
  }
  GPRS_CHECK_EQ(rule::simple_probing, 6);
  GPRS_CHECK_EQ(rule::singleton_cols, 7);
  GPRS_CHECK_EQ(rule::dual_cols, 8);
  GPRS_CHECK_EQ(rule::parallel_cols, 9);
  GPRS_CHECK_EQ(rule::count, 10);
  const budget_t expected[] = {
    {4., 2., 100000}, {8., 4., 100000}, {8., 4., 100000}, {8., 8., 100000}};
  for (int32_t r = rule::simple_probing; r <= rule::parallel_cols; ++r) {
    const budget_t& b = options.budget[r];
    const budget_t& e = expected[r - rule::simple_probing];
    GPRS_CHECK_EQ(b.passes, e.passes);
    GPRS_CHECK_EQ(b.stall_passes, e.stall_passes);
    GPRS_CHECK_EQ(b.floor, e.floor);
  }
  GPRS_CHECK_EQ(options.merge_integers, merge_integers_t::range_kept);
  GPRS_CHECK_EQ(options.saved_row_factor, 4.);
  GPRS_CHECK_EQ(options.pivot_growth_limit, 1000.);
  GPRS_CHECK_EQ(options.lock_alias_row_limit, 0);
  GPRS_CHECK_EQ(alias_scale_limit, 1e6);
}

// In primal-only mode a reduction of the column rules runs only when it keeps every feasible
// point and its objective value.
GPRS_TEST(columns_sub_rules_by_mode)
{
  options_t options;
  for (int32_t s = 0; s < sub_rule::count; ++s) {
    GPRS_CHECK(sub_rule_active(options, s));
  }
  options.primal_only = true;
  for (int32_t s = 0; s < sub_rule::count; ++s) {
    bool primal = s == sub_rule::free_singleton || s == sub_rule::parallel_merge;
    GPRS_CHECK_EQ(sub_rule_active(options, s), primal);
  }
  options.sub_enabled[sub_rule::free_singleton] = false;
  GPRS_CHECK(!sub_rule_active(options, sub_rule::free_singleton));
  GPRS_CHECK(sub_rule_active(options, sub_rule::parallel_merge));
  options.primal_only                       = false;
  options.sub_enabled[sub_rule::lock_alias] = false;
  GPRS_CHECK(!sub_rule_active(options, sub_rule::lock_alias));
  GPRS_CHECK(sub_rule_active(options, sub_rule::dual_bound));
}

// The families are called once per round, find nothing and leave no list behind: the result is
// a fixed point with them as without them.
GPRS_TEST(columns_families_are_called_and_find_nothing)
{
  //   r0:  x0 + x1 + x2 <= 4
  //   r1:  x0 - x1       >= -1
  // Nothing to reduce but x2, which no row stops from moving to its lower bound.
  model_t m = make_model({{1., 1., 1.}, {1., -1., 0.}},
                         {-inf, -1.},
                         {4., inf},
                         {0., 0., 0.},
                         {3., 3., 3.},
                         {-1., -1., 1.},
                         {1, 1, 1});
  for (bool primal_only : {false, true}) {
    options_t options;
    options.primal_only = primal_only;
    result_t result     = presolve(m, options);
    GPRS_CHECK(result.fixed_point);
    GPRS_CHECK_EQ(result.dual_findings, primal_only ? 0 : 1);
    GPRS_CHECK_EQ(result.dual_findings, result.stats[rule::dual_fixing].findings);
    for (int32_t r = rule::simple_probing; r <= rule::parallel_cols; ++r) {
      GPRS_CHECK(result.stats[r].calls >= 1);
      GPRS_CHECK(result.stats[r].calls <= result.rounds);
      GPRS_CHECK_EQ(result.stats[r].findings, 0);
      GPRS_CHECK_EQ(result.stats[r].work, 0);
      GPRS_CHECK_EQ(result.stats[r].rows_removed, 0);
      GPRS_CHECK_EQ(result.stats[r].cols_removed, 0);
    }
    for (int32_t s = 0; s < sub_rule::count; ++s) {
      GPRS_CHECK_EQ(result.sub_findings[s], 0);
    }
    GPRS_CHECK_EQ(result.columns.cost_fill, 0);
    GPRS_CHECK_EQ(result.columns.saved_entries, 0);
    GPRS_CHECK_EQ(result.columns.max_growth, 0.);
    GPRS_CHECK_EQ(result.columns.alias_depth, 0);
    GPRS_CHECK_EQ(result.columns.free_singleton_rejected_explicit, 0);
    result_t second = presolve(result.reduced, options);
    GPRS_CHECK_EQ(second.status, status_t::unchanged);

    result_t off = presolve(m, without_column_rules(options));
    GPRS_CHECK_EQ(difference(result, off), std::string());
    for (int32_t r = rule::simple_probing; r <= rule::parallel_cols; ++r) {
      GPRS_CHECK_EQ(off.stats[r].calls, 0);
    }
  }
}

// The column rules read sides, so the term of a fixed column leaves its row before their last
// round, and a row that lost one other than plainly is theirs to visit again. With one of them
// alone switched on, that is what the second round is for; without any, the term stays in the
// row until the reduced model is read.
GPRS_TEST(columns_family_sees_a_row_that_lost_a_fixed_term)
{
  // x0 + x1 + x2 <= 2e6 with x2 fixed at 1e6 by its bounds: a term above the size up to which
  // it leaves a row without an event.
  model_t m = make_model(
    {{1., 1., 1.}}, {-inf}, {2e6}, {0., 0., 1e6}, {1., 1., 1e6}, {0., 0., 0.}, {0, 0, 0});
  // No rebuild between the phases, which would move the term as well.
  options_t none;
  none.enabled.fill(false);
  none.rebuild_dead_fraction  = 2.;
  none.rebuild_fixed_fraction = 2.;
  for (int32_t family = rule::simple_probing; family < rule::count; ++family) {
    options_t options       = none;
    options.enabled[family] = true;
    result_t result         = presolve(m, options);
    GPRS_CHECK_EQ(result.status, status_t::reduced);
    GPRS_CHECK(result.fixed_point);
    GPRS_CHECK_EQ(result.rounds, 2);
    GPRS_CHECK_EQ(result.stats[family].calls, 2);
    GPRS_REQUIRE(result.reduced.nrows == 1);
    GPRS_CHECK_EQ(result.reduced.row_upper[0], 1e6);
    GPRS_CHECK_EQ(presolve(result.reduced, options).status, status_t::unchanged);
  }
  result_t result = presolve(m, none);
  GPRS_CHECK(result.fixed_point);
  GPRS_CHECK_EQ(result.rounds, 1);
  GPRS_REQUIRE(result.reduced.nrows == 1);
  GPRS_CHECK_EQ(result.reduced.row_upper[0], 1e6);
}

// On generated models, in both modes and with a rebuild after every phase or none: the reduced
// model, the stack, the flag, the rounds and the statistics of the Stage 1 families are the same
// with the four families switched on and switched off. Seven phase hashes per round.
GPRS_TEST(columns_empty_families_change_no_result)
{
  std::vector<oracle::generator_options_t> shapes(3);
  shapes[1].p_doubleton_equation = 0.4;
  shapes[1].p_equation           = 0.4;
  shapes[1].p_parallel_row       = 0.3;
  shapes[2].p_singleton_col      = 0.6;
  shapes[2].p_parallel_row       = 0.4;
  shapes[2].p_huge_bound         = 0.6;
  shapes[2].max_line_cols        = 3;
  int64_t compared = 0, reduced = 0, failures = 0;
  for (const oracle::generator_options_t& shape : shapes) {
    for (uint64_t seed = 7300000; seed < 7302000 && failures < 3; ++seed) {
      model_t m = oracle::generate(seed, shape).model;
      for (int32_t variant = 0; variant < 4; ++variant) {
        options_t options;
        options.primal_only  = variant & 1;
        options.phase_hashes = true;
        if (variant & 2) {
          options.rebuild_dead_fraction  = 0.;
          options.rebuild_fixed_fraction = 0.;
        }
        result_t on  = presolve(m, options);
        result_t off = presolve(m, without_column_rules(options));
        ++compared;
        reduced += on.status == status_t::reduced;
        std::string what = difference(on, off);
        bool carried =
          on.status != status_t::infeasible && on.status != status_t::unbounded_or_infeasible;
        if (what.empty() && carried && on.phase_hash.size() != 2 + 7 * (size_t)on.rounds) {
          what = "number of phase hashes";
        }
        if (what.empty()) { continue; }
        ++failures;
        test::fail(__FILE__,
                   __LINE__,
                   "seed " + std::to_string(seed) + " variant " + std::to_string(variant) +
                     ": another " + what);
      }
    }
  }
  GPRS_CHECK_EQ(failures, 0);
  GPRS_CHECK_EQ(compared, 3 * 2000 * 4);
  GPRS_CHECK(reduced > 5000);
}

// result_t::dual_findings counts the fixings by locks: none in primal-only mode, none with the
// family switched off, and with none the default result is the primal-only one.
GPRS_TEST(columns_dual_findings_are_the_lock_fixings)
{
  oracle::generator_options_t shape;
  shape.p_zero_cost = 0.5;
  int64_t with = 0, without = 0;
  for (uint64_t seed = 7310000; seed < 7313000; ++seed) {
    model_t m = oracle::generate(seed, shape).model;
    options_t options;
    result_t result = presolve(m, options);
    GPRS_CHECK_EQ(result.dual_findings, result.stats[rule::dual_fixing].findings);
    with += result.dual_findings > 0;
    options.primal_only = true;
    result_t primal     = presolve(m, options);
    GPRS_CHECK_EQ(primal.dual_findings, 0);
    GPRS_CHECK_EQ(primal.stats[rule::dual_fixing].calls, 0);
    if (result.dual_findings == 0 && result.status != status_t::unbounded_or_infeasible) {
      ++without;
      GPRS_CHECK_EQ(result.hash, primal.hash);
      GPRS_CHECK_EQ(postsolve_hash(result.postsolve), postsolve_hash(primal.postsolve));
    }
    options                            = options_t();
    options.enabled[rule::dual_fixing] = false;
    GPRS_CHECK_EQ(presolve(m, options).dual_findings, 0);
  }
  GPRS_CHECK(with > 500 && without > 500);
}
