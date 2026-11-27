/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The driver of presolve(): start step, statuses, statistics, stop rules, rebuilds and phase
// hashes. Every test switches the rule families it does not need off, so it holds for any set
// of families: here the start step alone and the start step with dual fixing.

#include "harness.hpp"
#include "test_engine_support.hpp"

#include <gprs/presolve.hpp>

#include <string>
#include <vector>

using namespace gprs;
using namespace gprs::engine_test;

namespace {

using list_t = std::vector<int32_t>;

options_t no_rule()
{
  options_t options;
  options.enabled.fill(false);
  return options;
}

options_t only_dual_fixing()
{
  options_t options                  = no_rule();
  options.enabled[rule::dual_fixing] = true;
  return options;
}

bool same_model(const model_t& a, const model_t& b)
{
  return a.nrows == b.nrows && a.ncols == b.ncols && a.row_start == b.row_start &&
         a.col_index == b.col_index && a.value == b.value && a.row_lower == b.row_lower &&
         a.row_upper == b.row_upper && a.col_lower == b.col_lower && a.col_upper == b.col_upper &&
         a.cost == b.cost && a.col_type == b.col_type && a.objective_offset == b.objective_offset;
}

bool no_call(const result_t& result)
{
  for (const rule_stats_t& s : result.stats) {
    if (s.calls != 0 || s.work != 0 || s.findings != 0 || s.rows_removed != 0 ||
        s.cols_removed != 0 || s.nnz_removed != 0) {
      return false;
    }
  }
  return true;
}

//   r0:  x0 + 2 x1 - x2 <= 6
//   r1:  x0 - x1 + x2 free
// x0 integer in [1, 4] cost 2, x1 in [0, 3] cost -1, x2 in [0, 2] cost -3, x3 integer in
// [-2, 7] cost 0 without entries; offset 10. Dual fixing: x0 = 1, x2 = 2, x3 = -2; x1 has an
// up lock and stays.
model_t four_columns()
{
  model_t m          = make_model({{1., 2., -1., 0.}, {1., -1., 1., 0.}},
                                  {-inf, -inf},
                                  {6., inf},
                                  {1., 0., 0., -2.},
                                  {4., 3., 2., 7.},
                                  {2., -1., -3., 0.},
                                  {1, 0, 0, 1});
  m.objective_offset = 10.;
  return m;
}

// What dual fixing leaves of four_columns: 2 x1 <= 7, -x1 free, min -x1 + 6.
model_t four_columns_reduced()
{
  model_t m          = make_model({{2.}, {-1.}}, {-inf, -inf}, {7., inf}, {0.}, {3.}, {-1.}, {0});
  m.objective_offset = 6.;
  return m;
}

// Two equations lock both columns in both directions: no family of this file changes it.
model_t locked_model()
{
  return make_model(
    {{1., 1.}, {1., -1.}}, {3., 1.}, {3., 1.}, {0., 0.}, {5., 5.}, {1., -2.}, {0, 1});
}

// One row over ten columns, column j with coefficient j + 1 and cost 1, offset 0.5.
model_t start_columns()
{
  double below_one   = 1. + 0x1p-21;
  model_t m          = make_model({{1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}},
                                  {-100.},
                                  {100.},
                                  {0.5, 1.2, 2.0000005, below_one, 3., 0.5, -inf, -0.0000001, 4., 3.9999995},
                                  {2.5, 1.9999999, 1.9999995, 1., 3., 2.5, 3.2, 3.0000001, 4., 4.0000005},
                         reals_t(10, 1.),
                                  {1, 1, 1, 0, 0, 0, 1, 1, 1, 1});
  m.objective_offset = 0.5;
  return m;
}

}  // namespace

GPRS_TEST(engine_names)
{
  const char* rules[rule::count] = {
    "row_rules", "propagation", "tightening", "doubleton", "parallel_rows", "dual_fixing"};
  for (int32_t r = 0; r < rule::count; ++r) {
    GPRS_CHECK_EQ(std::string(rule_name(r)), std::string(rules[r]));
  }
  GPRS_CHECK_EQ(std::string(rule_name(rule::tightening)), std::string("tightening"));
  GPRS_CHECK_EQ(std::string(rule_name(rule::dual_fixing)), std::string("dual_fixing"));
  GPRS_CHECK_EQ(std::string(status_name(status_t::reduced)), std::string("reduced"));
  GPRS_CHECK_EQ(std::string(status_name(status_t::unchanged)), std::string("unchanged"));
  GPRS_CHECK_EQ(std::string(status_name(status_t::reduced_to_empty)),
                std::string("reduced_to_empty"));
  GPRS_CHECK_EQ(std::string(status_name(status_t::infeasible)), std::string("infeasible"));
  GPRS_CHECK_EQ(std::string(status_name(status_t::unbounded_or_infeasible)),
                std::string("unbounded_or_infeasible"));
  GPRS_CHECK_EQ(std::string(status_name(status_t::limit)), std::string("limit"));
  GPRS_CHECK_EQ(std::string(status_name(status_t::failed)), std::string("failed"));
}

// Sides that cross by no more than the feasibility tolerance become an equation at the upper
// side; by more, the model is infeasible.
GPRS_TEST(engine_start_step_rows)
{
  double within = 1. + 0x1p-20;  // 9.5e-7 above 1
  double beyond = 1. + 0x1p-19;  // 1.9e-6 above 1
  model_t m     = make_model({{1., 1.}, {1., -1.}, {1., 0.}},
                             {within, -2., -inf},
                             {1., 3., 4.},
                             {0., 0.},
                             {5., 5.},
                             {1., 1.},
                             {0, 0});
  GPRS_REQUIRE(well_formed(m));
  result_t result = presolve(m, no_rule());
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  model_t expected      = m;
  expected.row_lower[0] = 1.;
  GPRS_CHECK(same_model(result.reduced, expected));
  GPRS_CHECK(result.postsolve.order.empty());
  GPRS_CHECK(result.postsolve.row_origin == (list_t{0, 1, 2}));
  GPRS_CHECK(result.postsolve.col_origin == (list_t{0, 1}));
  GPRS_CHECK(no_call(result));

  m.row_lower[0] = beyond;
  GPRS_CHECK_EQ(presolve(m, no_rule()).status, status_t::infeasible);
  m.row_lower[0] = 1.;
  GPRS_CHECK_EQ(presolve(m, no_rule()).status, status_t::unchanged);
  // Row 1, so that an earlier row is no excuse.
  m.row_lower[1] = 3.5;
  GPRS_CHECK_EQ(presolve(m, no_rule()).status, status_t::infeasible);

  // The tolerance is the one of the options.
  m.row_lower[1] = 3.0005;
  GPRS_CHECK_EQ(presolve(m, no_rule()).status, status_t::infeasible);
  options_t loose       = no_rule();
  loose.tol.feasibility = 1e-3;
  result                = presolve(m, loose);
  GPRS_REQUIRE(result.status == status_t::reduced);
  GPRS_CHECK_EQ(result.reduced.row_lower[1], 3.);
}

// Integer bounds are rounded, then crossing bounds are dealt with, then lower == upper is fixed.
GPRS_TEST(engine_start_step_columns)
{
  model_t m = start_columns();
  GPRS_REQUIRE(well_formed(m));
  result_t result = presolve(m, no_rule());
  GPRS_CHECK_EQ(result.status, status_t::reduced);

  // Kept: columns 0, 5, 6, 7. Fixed: 1, 2, 8, 9 (integer), 3 at its upper bound, 4.
  // Fixed terms of the row: 2 * 2 + 3 * 2 + 4 * 1 + 5 * 3 + 9 * 4 + 10 * 4 = 105.
  model_t expected          = make_model({{1., 6., 7., 8.}},
                                         {-205.},
                                         {-5.},
                                         {1., 0.5, -inf, 0.},
                                         {2., 2.5, 3., 3.},
                                reals_t(4, 1.),
                                         {1, 0, 1, 1});
  expected.objective_offset = 0.5 + 2. + 2. + 1. + 3. + 4. + 4.;
  GPRS_CHECK(same_model(result.reduced, expected));
  GPRS_CHECK(well_formed(result.reduced));
  GPRS_CHECK(result.postsolve.col_origin == (list_t{0, 5, 6, 7}));
  GPRS_CHECK(result.postsolve.row_origin == (list_t{0}));
  GPRS_CHECK_EQ(result.hash, model_hash(result.reduced));

  // In index order.
  const std::vector<fixed_col_record_t>& fixed = result.postsolve.fixed_cols;
  list_t cols;
  reals_t values;
  for (const fixed_col_record_t& record : fixed) {
    cols.push_back(record.col);
    values.push_back(record.value);
  }
  GPRS_CHECK(cols == (list_t{1, 2, 3, 4, 8, 9}));
  GPRS_CHECK(values == (reals_t{2., 2., 1., 3., 4., 4.}));
  GPRS_CHECK_EQ(result.postsolve.order.size(), (size_t)6);

  reals_t x;
  result.postsolve.postsolve({1., 0.5, 3., 0.}, x);
  GPRS_CHECK(x == (reals_t{1., 2., 2., 1., 3., 0.5, 3., 0., 4., 4.}));
  GPRS_CHECK(no_call(result));
  GPRS_CHECK_EQ(result.rounds, 1);
}

GPRS_TEST(engine_start_step_infeasible_columns)
{
  struct bounds_t {
    int32_t col;
    double lower;
    double upper;
    status_t status;
  };
  double beyond = 1. + 0x1p-19;
  for (const bounds_t& b : {bounds_t{0, 0.3, 0.7, status_t::infeasible},        // no integer
                            bounds_t{0, 0.5, 0.9999985, status_t::infeasible},  // 1 > 0 rounded
                            bounds_t{0, 2.6, 3.4, status_t::reduced},           // the integer 3
                            bounds_t{5, 3., 1., status_t::infeasible},
                            bounds_t{5, beyond, 1., status_t::infeasible},
                            bounds_t{9, 4.0000005, 3.9999995, status_t::reduced},  // still 4
                            bounds_t{9, 4.4, 3.6, status_t::infeasible}}) {
    model_t m          = start_columns();
    m.col_lower[b.col] = b.lower;
    m.col_upper[b.col] = b.upper;
    GPRS_REQUIRE(well_formed(m));
    result_t result = presolve(m, no_rule());
    GPRS_CHECK_EQ(result.status, b.status);
    if (b.status == status_t::infeasible) {
      GPRS_CHECK_EQ(result.reduced.ncols, 0);
      GPRS_CHECK_EQ(result.rounds, 0);
    }
  }
}

GPRS_TEST(engine_unchanged_model)
{
  model_t m = locked_model();
  for (const options_t& options : {no_rule(), only_dual_fixing()}) {
    result_t result = presolve(m, options);
    GPRS_CHECK_EQ(result.status, status_t::unchanged);
    GPRS_CHECK(same_model(result.reduced, m));
    GPRS_CHECK_EQ(result.hash, model_hash(m));
    GPRS_CHECK(result.postsolve.order.empty());
    GPRS_CHECK(result.postsolve.row_origin == (list_t{0, 1}));
    GPRS_CHECK(result.postsolve.col_origin == (list_t{0, 1}));
    GPRS_CHECK_EQ(result.rounds, 1);
    GPRS_CHECK_EQ(result.rebuilds, 1);
    GPRS_CHECK(result.seconds >= 0.);
  }
  result_t result = presolve(m, only_dual_fixing());
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].calls, 1);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].work, 2 * line_visit_work);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].findings, 0);

  // A model without rows and columns.
  model_t empty;
  empty.row_start.push_back(0);
  GPRS_REQUIRE(well_formed(empty));
  result_t nothing = presolve(empty, options_t());
  GPRS_CHECK_EQ(nothing.status, status_t::unchanged);
  GPRS_CHECK(same_model(nothing.reduced, empty));
}

GPRS_TEST(engine_reduced_to_empty)
{
  model_t m = make_model({}, {}, {}, {0., 1., -1.}, {2., 3., 1.}, {1., -1., 0.}, {1, 0, 1});
  m.objective_offset = 5.;
  GPRS_REQUIRE(well_formed(m));
  result_t result = presolve(m, only_dual_fixing());
  GPRS_CHECK_EQ(result.status, status_t::reduced_to_empty);
  GPRS_CHECK_EQ(result.reduced.nrows, 0);
  GPRS_CHECK_EQ(result.reduced.ncols, 0);
  GPRS_CHECK(well_formed(result.reduced));
  GPRS_CHECK_EQ(result.reduced.objective_offset, 5. + 0. - 3.);
  reals_t x;
  result.postsolve.postsolve({}, x);
  GPRS_CHECK(x == (reals_t{0., 3., -1.}));
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].cols_removed, 3);
  // The second round finds nothing on the list.
  GPRS_CHECK_EQ(result.rounds, 2);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].calls, 1);

  // A row without a column left stays a row while no row rule runs: reduced, not empty.
  model_t with_row = make_model({{1., 1.}}, {-inf}, {inf}, {0., 0.}, {1., 1.}, {1., 1.}, {0, 0});
  result           = presolve(with_row, only_dual_fixing());
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK_EQ(result.reduced.nrows, 1);
  GPRS_CHECK_EQ(result.reduced.ncols, 0);
  GPRS_CHECK(well_formed(result.reduced));
}

GPRS_TEST(engine_unbounded_or_infeasible)
{
  model_t m       = make_model({{1., 1.}}, {-inf}, {4.}, {-inf, 0.}, {5., 1.}, {1., 0.}, {0, 0});
  result_t result = presolve(m, only_dual_fixing());
  GPRS_CHECK_EQ(result.status, status_t::unbounded_or_infeasible);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].calls, 1);
  GPRS_CHECK_EQ(result.rounds, 1);

  // No lock argument, no status.
  options_t primal   = only_dual_fixing();
  primal.primal_only = true;
  GPRS_CHECK_EQ(presolve(m, primal).status, status_t::unchanged);
  GPRS_CHECK_EQ(presolve(m, no_rule()).status, status_t::unchanged);
}

GPRS_TEST(engine_statistics_of_dual_fixing)
{
  model_t m       = four_columns();
  result_t result = presolve(m, only_dual_fixing());
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK(same_model(result.reduced, four_columns_reduced()));
  GPRS_CHECK(result.postsolve.col_origin == (list_t{1}));
  GPRS_CHECK(result.postsolve.row_origin == (list_t{0, 1}));
  GPRS_CHECK_EQ(result.hash, model_hash(four_columns_reduced()));

  const rule_stats_t& stats = result.stats[rule::dual_fixing];
  GPRS_CHECK_EQ(stats.calls, 1);
  GPRS_CHECK_EQ(stats.work, 4 * line_visit_work + 4);
  GPRS_CHECK_EQ(stats.findings, 3);
  GPRS_CHECK_EQ(stats.rows_removed, 0);
  GPRS_CHECK_EQ(stats.cols_removed, 3);
  GPRS_CHECK_EQ(stats.nnz_removed, 4);
  GPRS_CHECK(stats.seconds >= 0. && result.seconds >= stats.seconds);
  for (int32_t r = 0; r < rule::dual_fixing; ++r) {
    GPRS_CHECK_EQ(result.stats[r].calls, 0);
    GPRS_CHECK_EQ(result.stats[r].findings, 0);
  }
  GPRS_CHECK_EQ(result.rounds, 2);
  // Four of six entries belong to fixed columns after the cascade: one rebuild there, one at
  // the end.
  GPRS_CHECK_EQ(result.rebuilds, 2);

  reals_t x;
  result.postsolve.postsolve({3.}, x);
  GPRS_CHECK(x == (reals_t{1., 3., 2., -2.}));
  point_check_t check = check_point(m, x);
  GPRS_CHECK(point_feasible(check, options_t().tol));
  GPRS_CHECK_EQ(check.objective, 10. + 2. - 3. - 6.);

  // x0 + x1 + x2 <= 5 with x0 = 2 from the start step, whose entry is not counted for the
  // family: dual fixing removes x1 (cost 1, to 0) and its one entry; x2 (cost -1) stays.
  model_t with_fixed =
    make_model({{1., 1., 1.}}, {-inf}, {5.}, {2., 0., 0.}, {2., 3., 3.}, {1., 1., -1.}, {0, 0, 0});
  result = presolve(with_fixed, only_dual_fixing());
  GPRS_CHECK_EQ(stats.findings, 1);
  GPRS_CHECK_EQ(stats.cols_removed, 1);
  GPRS_CHECK_EQ(stats.nnz_removed, 1);
  model_t left          = make_model({{1.}}, {-inf}, {3.}, {0.}, {3.}, {-1.}, {0});
  left.objective_offset = 2.;
  GPRS_CHECK(same_model(result.reduced, left));
  GPRS_CHECK(result.postsolve.col_origin == (list_t{2}));
}

// The fractions decide the rebuilds between the phases, never the reduced model.
GPRS_TEST(engine_rebuild_fractions)
{
  model_t m                    = four_columns();
  options_t never              = only_dual_fixing();
  never.rebuild_dead_fraction  = 2.;
  never.rebuild_fixed_fraction = 2.;
  result_t result              = presolve(m, never);
  GPRS_CHECK_EQ(result.rebuilds, 1);
  GPRS_CHECK(same_model(result.reduced, four_columns_reduced()));

  // 4 of 6 live entries are fixed: 0.66 is below the fraction, 0.67 above.
  options_t below              = only_dual_fixing();
  below.rebuild_fixed_fraction = 0.67;
  GPRS_CHECK_EQ(presolve(m, below).rebuilds, 1);
  options_t above              = only_dual_fixing();
  above.rebuild_fixed_fraction = 0.66;
  GPRS_CHECK_EQ(presolve(m, above).rebuilds, 2);

  options_t always              = only_dual_fixing();
  always.rebuild_dead_fraction  = 0.;
  always.rebuild_fixed_fraction = 0.;
  result                        = presolve(m, always);
  GPRS_CHECK_EQ(result.rebuilds, 2);
  GPRS_CHECK(same_model(result.reduced, four_columns_reduced()));
}

// The time limit ends the presolve before the next call of a family, with the state as it is.
GPRS_TEST(engine_time_limit)
{
  model_t m          = four_columns();
  options_t options  = only_dual_fixing();
  options.time_limit = 0.;
  result_t result    = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::limit);
  GPRS_CHECK(same_model(result.reduced, m));
  GPRS_CHECK(well_formed(result.reduced));
  GPRS_CHECK(result.postsolve.col_origin == (list_t{0, 1, 2, 3}));
  GPRS_CHECK(no_call(result));
  GPRS_CHECK_EQ(result.hash, model_hash(m));

  // The start step is done before the limit is looked at.
  model_t start      = start_columns();
  result_t limited   = presolve(start, options);
  result_t unlimited = presolve(start, no_rule());
  GPRS_CHECK_EQ(limited.status, status_t::limit);
  GPRS_CHECK(same_model(limited.reduced, unlimited.reduced));

  // No family to call: nothing is cut short.
  options_t idle  = no_rule();
  idle.time_limit = 0.;
  GPRS_CHECK_EQ(presolve(m, idle).status, status_t::unchanged);

  options.time_limit = 3600.;
  GPRS_CHECK_EQ(presolve(m, options).status, status_t::reduced);
}

GPRS_TEST(engine_round_limits)
{
  model_t m          = four_columns();
  options_t options  = only_dual_fixing();
  options.max_rounds = 0;
  result_t result    = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::unchanged);
  GPRS_CHECK_EQ(result.rounds, 0);
  GPRS_CHECK(no_call(result));

  options.max_rounds = 1;
  result             = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK_EQ(result.rounds, 1);
  GPRS_CHECK(same_model(result.reduced, four_columns_reduced()));

  options.max_rounds         = 20;
  options.max_cascade_passes = 0;
  result                     = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::unchanged);
  GPRS_CHECK_EQ(result.rounds, 1);
  GPRS_CHECK(no_call(result));
}

// A call gets make_budget of its family on the state as it is. A call that stops on its budget
// ends the cascade; a round without a finding ends the presolve.
GPRS_TEST(engine_budget_per_call)
{
  model_t m = four_columns();

  options_t zero                 = only_dual_fixing();
  zero.budget[rule::dual_fixing] = budget_t{0., 0., 0};
  result_t result                = presolve(m, zero);
  GPRS_CHECK_EQ(result.status, status_t::unchanged);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].calls, 1);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].work, 0);
  GPRS_CHECK_EQ(result.rounds, 1);

  // Limit 8. Round 1: x0 is fixed (8 + 2 units) and the call stops. Round 2: x1 gives nothing
  // and the call stops; the round has no finding.
  options_t one                 = only_dual_fixing();
  one.budget[rule::dual_fixing] = budget_t{0., 0., 8};
  result                        = presolve(m, one);
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK_EQ(result.reduced.ncols, 3);
  GPRS_CHECK(result.postsolve.col_origin == (list_t{1, 2, 3}));
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].calls, 2);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].work, 10 + 8);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].findings, 1);
  GPRS_CHECK_EQ(result.rounds, 2);

  // Limit 11. Round 1: x0 (10 units), x1 (18), stop. Round 2: x2 and x3. Round 3: no list.
  options_t two                 = only_dual_fixing();
  two.budget[rule::dual_fixing] = budget_t{0., 0., 11};
  result                        = presolve(m, two);
  GPRS_CHECK(same_model(result.reduced, four_columns_reduced()));
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].calls, 2);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].work, 18 + 18);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].findings, 3);
  GPRS_CHECK_EQ(result.rounds, 3);

  // The budget follows the size of the state: a quarter of a pass over four_columns is
  // (6 + 8 * (2 + 4)) / 4 = 13.5 units, which is x0 and x1 in the first call.
  options_t quarter                 = only_dual_fixing();
  quarter.budget[rule::dual_fixing] = budget_t{0.25, 0.25, 0};
  result                            = presolve(m, quarter);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].work, 18 + 18);
  GPRS_CHECK_EQ(result.rounds, 3);
}

// The cascade repeats its families while one of their lists is not empty. x0 + x1 >= 1 with
// x0, x1 in [0, 1], cost 1 and 0: dual fixing sets x1 = 1 (no up lock), the row is redundant
// and the row rules delete it in the next pass, and dual fixing then moves x0, which has no
// lock left, to 0. This test needs the row rules (part C) besides dual fixing.
GPRS_TEST(engine_cascade_repeats_on_released_locks)
{
  model_t m         = make_model({{1., 1.}}, {1.}, {inf}, {0., 0.}, {1., 1.}, {1., 0.}, {0, 0});
  options_t options = only_dual_fixing();
  options.enabled[rule::row_rules] = true;
  for (int32_t passes : {100, 1}) {
    options.max_cascade_passes = passes;
    result_t result            = presolve(m, options);
    GPRS_CHECK_EQ(result.status, status_t::reduced_to_empty);
    GPRS_CHECK_EQ(result.reduced.objective_offset, 0.);
    reals_t x;
    result.postsolve.postsolve({}, x);
    GPRS_CHECK(x == (reals_t{0., 1.}));
    GPRS_CHECK_EQ(result.stats[rule::row_rules].rows_removed, 1);
    GPRS_CHECK_EQ(result.stats[rule::row_rules].cols_removed, 0);
    GPRS_CHECK_EQ(result.stats[rule::dual_fixing].rows_removed, 0);
    GPRS_CHECK_EQ(result.stats[rule::dual_fixing].cols_removed, 2);
    GPRS_CHECK_EQ(result.stats[rule::dual_fixing].calls, 2);
    // With one pass per cascade the second pass is the cascade of the next round.
    GPRS_CHECK_EQ(result.rounds, passes == 1 ? 3 : 2);
  }

  // Inside a pass dual fixing comes last and sees the locks the row families leave: x0 <= 5
  // with x0 in [0, 1] and cost -1 is redundant, and x0 goes to 1 in the same pass.
  model_t order   = make_model({{1.}}, {-inf}, {5.}, {0.}, {1.}, {-1.}, {0});
  result_t result = presolve(order, options);
  GPRS_CHECK_EQ(result.status, status_t::reduced_to_empty);
  GPRS_CHECK_EQ(result.reduced.objective_offset, -1.);
  GPRS_CHECK_EQ(result.stats[rule::row_rules].calls, 1);
  GPRS_CHECK_EQ(result.stats[rule::dual_fixing].calls, 1);
  GPRS_CHECK_EQ(result.rounds, 2);
}

// One hash after the start step, one after each of the three phases of a round and one at the
// end; the same on every run.
GPRS_TEST(engine_phase_hashes)
{
  model_t m         = four_columns();
  options_t options = only_dual_fixing();
  GPRS_CHECK(presolve(m, options).phase_hash.empty());

  options.phase_hashes = true;
  result_t a           = presolve(m, options);
  result_t b           = presolve(model_t(m), options);
  GPRS_CHECK_EQ(a.rounds, 2);
  GPRS_REQUIRE(a.phase_hash.size() == 8);
  GPRS_CHECK(a.phase_hash == b.phase_hash);
  GPRS_CHECK_EQ(a.hash, b.hash);
  GPRS_CHECK_EQ(postsolve_hash(a.postsolve), postsolve_hash(b.postsolve));
  // The cascade of round 1 changes the state; nothing does until the end, where the reduced
  // model gets its origins.
  GPRS_CHECK(a.phase_hash[0] != a.phase_hash[1]);
  for (size_t k = 2; k < 7; ++k) {
    GPRS_CHECK_EQ(a.phase_hash[k], a.phase_hash[1]);
  }
  GPRS_CHECK(a.phase_hash[7] != a.phase_hash[6]);

  // The hashes follow the model.
  model_t other = m;
  other.cost[1] = -2.;
  result_t c    = presolve(other, options);
  GPRS_REQUIRE(c.phase_hash.size() == 8);
  for (size_t k = 0; k < 8; ++k) {
    GPRS_CHECK(c.phase_hash[k] != a.phase_hash[k]);
  }
  GPRS_CHECK(c.hash != a.hash);
}

namespace {

// presolve() against enumeration on tiny integer models. Windows as in the dual-fixing test;
// 6 holds every finite bound of the generator with its half.
int32_t check_against_enumeration(const options_t& options, uint64_t seed, const char* name)
{
  random_options_t generator;
  generator.start_step_cases = true;
  model_t m                  = random_model(seed, generator);
  std::string label          = std::string(name) + ", seed " + std::to_string(seed) + ": ";
  if (!well_formed(m)) {
    gprs::test::fail(__FILE__, __LINE__, label + "generator");
    return 1;
  }
  result_t result = presolve(m, options);
  if (result.status == status_t::unbounded_or_infeasible) {
    optimum_t narrow = enumerate(m, 6.);
    if (!narrow.feasible) { return 0; }
    optimum_t wide = enumerate(m, 8.);
    if (wide.feasible && wide.objective < narrow.objective) { return 0; }
    gprs::test::fail(__FILE__, __LINE__, label + "unbounded claimed, optimum does not fall");
    return 1;
  }
  std::string error;
  optimum_t before = enumerate(m, 6.);
  optimum_t after;
  if (result.status != status_t::reduced && result.status != status_t::unchanged &&
      result.status != status_t::reduced_to_empty) {
    error = std::string("status ") + status_name(result.status);
  } else if (!well_formed(result.reduced)) {
    error = "reduced model is not well formed";
  } else if ((result.status == status_t::unchanged) != same_model(result.reduced, m)) {
    error = "status unchanged against the reduced model";
  } else if ((result.status == status_t::reduced_to_empty) !=
             (result.reduced.nrows == 0 && result.reduced.ncols == 0 && m.ncols > 0)) {
    error = "status reduced_to_empty against the reduced model";
  } else if (result.hash != model_hash(result.reduced)) {
    error = "hash";
  } else if ((int32_t)result.postsolve.col_origin.size() != result.reduced.ncols ||
             (int32_t)result.postsolve.row_origin.size() != result.reduced.nrows) {
    error = "origins";
  } else {
    after = enumerate(result.reduced, 6.);
    if (before.feasible != after.feasible) {
      error = "feasibility changed";
    } else if (before.feasible && before.objective != after.objective) {
      error = "optimum " + std::to_string(before.objective) + " became " +
              std::to_string(after.objective);
    } else if (before.feasible) {
      reals_t x;
      result.postsolve.postsolve(after.x, x);
      point_check_t check = check_point(m, x);
      if (!point_feasible(check, options.tol)) {
        error = "postsolved optimum is infeasible";
      } else if (check.objective != before.objective) {
        error = "postsolved optimum has another objective value";
      }
    }
  }
  // Nothing is left to do: the reduced model is a fixed point of the same presolve.
  if (error.empty() && presolve(result.reduced, options).status != status_t::unchanged) {
    error = "a second presolve changes the reduced model";
  }
  if (error.empty()) { return 0; }
  gprs::test::fail(__FILE__, __LINE__, label + error);
  return 1;
}

void run_against_enumeration(const options_t& options, const char* name)
{
  int32_t failures = 0;
  for (uint64_t seed = 100000; seed < 103000 && failures < 5; ++seed) {
    failures += check_against_enumeration(options, seed, name);
  }
}

}  // namespace

GPRS_TEST(engine_start_step_keeps_the_optimum_by_enumeration)
{
  run_against_enumeration(no_rule(), "no rule");
}

GPRS_TEST(engine_dual_fixing_keeps_the_optimum_by_enumeration)
{
  run_against_enumeration(only_dual_fixing(), "dual fixing");
  options_t always              = only_dual_fixing();
  always.rebuild_dead_fraction  = 0.;
  always.rebuild_fixed_fraction = 0.;
  run_against_enumeration(always, "dual fixing, rebuild always");
  options_t small                 = only_dual_fixing();
  small.budget[rule::dual_fixing] = budget_t{0., 0., 12};
  run_against_enumeration(small, "dual fixing, 12 units per call");
}

// In primal-only mode the start step is all that is left of this set of families.
GPRS_TEST(engine_primal_only_is_the_start_step)
{
  options_t primal   = only_dual_fixing();
  primal.primal_only = true;
  random_options_t generator;
  generator.start_step_cases = true;
  for (uint64_t seed = 200000; seed < 200300; ++seed) {
    model_t m  = random_model(seed, generator);
    result_t a = presolve(m, primal);
    result_t b = presolve(m, no_rule());
    GPRS_CHECK_EQ(a.status, b.status);
    GPRS_CHECK_EQ(a.hash, b.hash);
    GPRS_CHECK_EQ(a.stats[rule::dual_fixing].calls, 0);
  }
}
