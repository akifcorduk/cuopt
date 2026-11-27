/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The end of presolve(): the implied bounds of continuous columns become explicit and the rounds
// apply what follows from them, the terms of fixed columns leave the rows before the last
// round, and the reduced model is a fixed point of the rules (result_t::fixed_point). Also the
// statistics that go with it: records apart from findings, the start step as its own entry,
// and no empty row whatever stops the rounds.

#include "harness.hpp"
#include "test_rows_support.hpp"

#include <gprs/postsolve.hpp>
#include <gprs/presolve.hpp>

#include <cmath>
#include <vector>

using namespace gprs;
using namespace rule_test;

namespace {

bool same_model(const model_t& a, const model_t& b)
{
  return a.nrows == b.nrows && a.ncols == b.ncols && a.row_start == b.row_start &&
         a.col_index == b.col_index && a.value == b.value && a.row_lower == b.row_lower &&
         a.row_upper == b.row_upper && a.col_lower == b.col_lower && a.col_upper == b.col_upper &&
         a.cost == b.cost && a.col_type == b.col_type && a.objective_offset == b.objective_offset;
}

model_t with_cost(model_t m, const std::vector<double>& cost)
{
  m.cost = cost;
  return m;
}

// The presolve ended at a fixed point and a second presolve of its result changes nothing.
void check_fixed_point(const result_t& result, const options_t& options)
{
  GPRS_CHECK(result.fixed_point);
  result_t again = presolve(result.reduced, options);
  GPRS_CHECK_EQ(again.status, status_t::unchanged);
  GPRS_CHECK(same_model(again.reduced, result.reduced));
  GPRS_CHECK(again.fixed_point);
  for (const rule_stats_t& stats : again.stats) {
    GPRS_CHECK_EQ(stats.findings, 0);
  }
}

// x <= 1000 u and y <= 1000 v with binaries u and v give x and y the upper bound 1000, which
// makes -1000 <= y - x <= 1000 redundant. x and y have no explicit upper bound.
model_t two_implied_bounds()
{
  std::vector<col_t> cols = {{0., inf, false}, {0., inf, false}, {0., 1., true}, {0., 1., true}};
  std::vector<row_t> rows = {{{{0, 1.}, {2, -1000.}}, -inf, 0.},
                             {{{1, 1.}, {3, -1000.}}, -inf, 0.},
                             {{{0, -1.}, {1, 1.}}, -1000., 1000.}};
  return with_cost(make_model(cols, rows), {-1., -1., 1., 1.});
}

}  // namespace

// A row that is redundant on the bounds that other rows imply is removed, and the bounds are in
// the reduced model. (669 such rows stayed in unitcal_7.)
GPRS_TEST(engine_implied_bounds_make_a_row_redundant)
{
  model_t m = two_implied_bounds();
  GPRS_REQUIRE(well_formed(m));
  options_t options;
  result_t result = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK_EQ(result.reduced.nrows, 2);
  GPRS_CHECK(result.postsolve.row_origin == std::vector<int32_t>({0, 1}));
  GPRS_CHECK_EQ(result.reduced.ncols, 4);
  GPRS_CHECK_EQ(result.reduced.col_upper[0], 1000.);
  GPRS_CHECK_EQ(result.reduced.col_upper[1], 1000.);
  GPRS_CHECK_EQ(result.stats[rule::row_rules].rows_removed, 1);
  // The first round only records. The bounds become explicit after it, the second round removes
  // the row and the third finds nothing.
  GPRS_CHECK_EQ(result.rounds, 3);
  GPRS_CHECK_EQ(result.explicit_rounds, 2);
  GPRS_CHECK_EQ(result.stats[rule::propagation].findings, 2);
  GPRS_CHECK(result.stats[rule::propagation].records >= 2);
  GPRS_CHECK(result.seconds_explicit >= 0. && result.seconds_explicit <= result.seconds);
  check_fixed_point(result, options);

  // Without bound propagation no bound is implied and the row stays.
  options.enabled[rule::propagation] = false;
  result                             = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::unchanged);
  GPRS_CHECK_EQ(result.explicit_rounds, 0);
  check_fixed_point(result, options);
}

// w <= z and w <= 100 z with a binary z: the first row implies w <= 1, on which the 100 of the
// second row is strengthened to 1. The two rows are then parallel and one of them goes.
// (rd-rplusc-21 kept coefficients of 1e7 in 47,567 rows this way.)
GPRS_TEST(engine_implied_bounds_strengthen_a_coefficient)
{
  std::vector<col_t> cols = {{0., 1., true}, {0., inf, false}};
  std::vector<row_t> rows = {{{{0, -1.}, {1, 1.}}, -inf, 0.}, {{{0, -100.}, {1, 1.}}, -inf, 0.}};
  model_t m               = with_cost(make_model(cols, rows), {1., -1.});
  options_t options;
  result_t result = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK_EQ(result.reduced.nrows, 1);
  GPRS_CHECK(result.reduced.value == std::vector<double>({-1., 1.}));
  GPRS_CHECK_EQ(result.reduced.row_upper[0], 0.);
  GPRS_CHECK_EQ(result.reduced.col_upper[1], 1.);
  GPRS_CHECK(result.stats[rule::tightening].findings >= 1);
  GPRS_CHECK_EQ(result.stats[rule::parallel_rows].rows_removed, 1);
  check_fixed_point(result, options);
}

// A chain of bounds: x1 <= x0 + 1, x2 <= x1 + 1, x3 <= x2 + 1 with x0 <= 5, and x3 <= 9, which
// the bound 8 of the chain makes redundant. One recorded bound per round would need a round per
// link; after the switch the cascade follows the chain.
GPRS_TEST(engine_implied_bounds_follow_a_chain)
{
  std::vector<col_t> cols = {{0., 5., false}, {0., inf, false}, {0., inf, false}, {0., inf, false}};
  std::vector<row_t> rows = {{{{2, -1.}, {3, 1.}}, -inf, 1.},
                             {{{1, -1.}, {2, 1.}}, -inf, 1.},
                             {{{0, -1.}, {1, 1.}}, -inf, 1.},
                             {{{2, 1.}, {3, 1.}}, -inf, 15.}};
  // The cost keeps every column away from the bound that dual fixing would give it.
  model_t m = with_cost(make_model(cols, rows), {1., -1., 1., -1.});
  options_t options;
  result_t result = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK_EQ(result.reduced.nrows, 3);
  GPRS_CHECK(result.reduced.col_upper == std::vector<double>({5., 6., 7., 8.}));
  GPRS_CHECK_EQ(result.rounds, 3);
  check_fixed_point(result, options);
}

// What stops the rounds before nothing is left to do: the result is a valid reduced model and
// not a fixed point.
GPRS_TEST(engine_fixed_point_needs_the_rounds_to_end)
{
  model_t m = two_implied_bounds();
  options_t options;
  GPRS_CHECK(presolve(m, options).fixed_point);

  // One round is the round that records. The bounds become explicit, and no round is left for
  // what follows from them.
  options.max_rounds = 1;
  result_t result    = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK(!result.fixed_point);
  GPRS_CHECK_EQ(result.rounds, 1);
  GPRS_CHECK_EQ(result.reduced.nrows, 3);
  GPRS_CHECK_EQ(result.reduced.col_upper[0], 1000.);
  GPRS_CHECK(well_formed(result.reduced));
  // The second presolve goes on where the first stopped.
  result_t again = presolve(result.reduced, options_t());
  GPRS_CHECK_EQ(again.status, status_t::reduced);
  GPRS_CHECK_EQ(again.reduced.nrows, 2);

  // A family that stops on its budget in the last round.
  options                         = options_t();
  options.budget[rule::row_rules] = budget_t{0., 0., 0};
  result                          = presolve(m, options);
  GPRS_CHECK(!result.fixed_point);
  GPRS_CHECK_EQ(result.reduced.nrows, 3);

  // A cascade that max_cascade_passes ends with a list left.
  options                    = options_t();
  options.max_cascade_passes = 0;
  result                     = presolve(m, options);
  GPRS_CHECK(!result.fixed_point);
  GPRS_CHECK_EQ(result.status, status_t::unchanged);

  // The time limit.
  options            = options_t();
  options.time_limit = 0.;
  result             = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::limit);
  GPRS_CHECK(!result.fixed_point);
}

// Bounds that improve each other in ever smaller steps: every bound gets its steps and the
// presolve ends without a fixed point, in a few rounds.
GPRS_TEST(engine_fixed_point_and_the_step_limit)
{
  std::vector<col_t> cols = {{0., 1e6, false}, {0., 1e6, false}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, -0.999}}, -inf, 1.},
                             {{{0, -0.999}, {1, 1.}}, -inf, 1.}};
  model_t m               = with_cost(make_model(cols, rows), {-1., -1.});
  result_t result         = presolve(m, options_t());
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK(!result.fixed_point);
  GPRS_CHECK(result.rounds <= 3);
  GPRS_CHECK_EQ(result.stats[rule::propagation].findings, 34);
  GPRS_CHECK(result.reduced.col_upper[0] > 9.6e5 && result.reduced.col_upper[0] < 9.7e5);
}

// A round ends on changes of the model. A model on which bound propagation can only record
// has one round and status unchanged, and the records are counted apart.
GPRS_TEST(engine_records_are_no_findings)
{
  std::vector<col_t> cols = {{0., 10., false}, {0., 10., false}};
  model_t m = with_cost(make_model(cols, {{{{0, 1.}, {1, 1.}}, -inf, 15.}}), {-1., -1.});
  options_t options;
  result_t result = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::unchanged);
  GPRS_CHECK_EQ(result.rounds, 1);
  GPRS_CHECK_EQ(result.explicit_rounds, 0);
  GPRS_CHECK_EQ(result.stats[rule::propagation].calls, 1);
  GPRS_CHECK_EQ(result.stats[rule::propagation].records, 2);
  for (int32_t r = 0; r < rule::count; ++r) {
    GPRS_CHECK_EQ(result.stats[r].findings, 0);
    if (r != rule::propagation) { GPRS_CHECK_EQ(result.stats[r].records, 0); }
  }
  check_fixed_point(result, options);
}

// The start step has its own entry in the statistics.
GPRS_TEST(engine_start_step_statistics)
{
  // x0 is fixed in the model, x1 gets the integer bounds [1, 2], row 1 the side 4 for [4, 4 -
  // 5e-7].
  std::vector<col_t> cols = {{2., 2., false}, {0.5, 2.5, true}, {0., 10., false}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}, {2, 1.}}, -inf, 9.},
                             {{{0, 1.}, {2, 1.}}, 4., 4. - 5e-7}};
  model_t m               = with_cost(make_model(cols, rows), {1., 1., 1.});
  options_t options;
  options.enabled.fill(false);
  result_t result = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK_EQ(result.start.calls, 1);
  GPRS_CHECK_EQ(result.start.work, 0);
  GPRS_CHECK_EQ(result.start.records, 0);
  // One side, two bounds and one fixed column.
  GPRS_CHECK_EQ(result.start.findings, 4);
  GPRS_CHECK_EQ(result.start.rows_removed, 0);
  GPRS_CHECK_EQ(result.start.cols_removed, 1);
  GPRS_CHECK_EQ(result.start.nnz_removed, 2);
  GPRS_CHECK_EQ(result.start.seconds, result.seconds_start);
  for (const rule_stats_t& stats : result.stats) {
    GPRS_CHECK_EQ(stats.calls, 0);
    GPRS_CHECK_EQ(stats.cols_removed, 0);
  }

  // Nothing to do, and an infeasible start.
  result = presolve(two_implied_bounds(), options);
  GPRS_CHECK_EQ(result.start.calls, 1);
  GPRS_CHECK_EQ(result.start.findings, 0);
  m.col_lower[2] = 11.;
  result         = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::infeasible);
  GPRS_CHECK_EQ(result.start.calls, 1);
  GPRS_CHECK_EQ(result.start.findings, 0);
}

// A reduced model has no row without entries, whatever stopped the rounds.
GPRS_TEST(engine_no_empty_row_after_a_limit)
{
  // Dual fixing puts x0 and x1 at 0 in the first cascade pass; the row is then without a free
  // entry, and with one pass and one round no call of the row rules follows.
  std::vector<col_t> cols = {{0., 3., true}, {0., 3., true}, {0., 3., true}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}}, -inf, 5.}, {{{2, 1.}}, 1., 2.}};
  model_t m               = with_cost(make_model(cols, rows), {1., 1., 0.});
  options_t options;
  options.max_rounds                 = 1;
  options.max_cascade_passes         = 1;
  options.enabled[rule::propagation] = false;
  options.enabled[rule::tightening]  = false;
  // Row 1 stays for the list of the row rules: their budget ends the first call at once.
  options.budget[rule::row_rules] = budget_t{0., 0., 0};
  result_t result                 = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK(!result.fixed_point);
  GPRS_CHECK_EQ(result.reduced.nrows, 1);
  GPRS_CHECK(result.postsolve.row_origin == std::vector<int32_t>({1}));
  GPRS_CHECK_EQ(result.reduced.nnz(), 1);
  GPRS_CHECK_EQ(result.stats[rule::row_rules].rows_removed, 1);
  GPRS_CHECK_EQ(result.stats[rule::row_rules].findings, 1);
  GPRS_CHECK_EQ(result.stats[rule::row_rules].calls, 1);

  // The row without a free entry can be infeasible: x0 + x1 >= 1 with both columns fixed at 0
  // by the start step.
  std::vector<col_t> fixed = {{0., 0., true}, {0., 0., true}, {0., 3., true}};
  std::vector<row_t> wrong = {{{{0, 1.}, {1, 1.}}, 1., inf}, {{{2, 1.}}, 1., 2.}};
  result                   = presolve(make_model(fixed, wrong), options);
  GPRS_CHECK_EQ(result.status, status_t::infeasible);
  GPRS_CHECK_EQ(result.reduced.nrows, 0);

  // With the row rules switched off an empty row stays, as the options ask.
  options.enabled[rule::row_rules] = false;
  result                           = presolve(m, options);
  GPRS_CHECK_EQ(result.reduced.nrows, 2);
  GPRS_CHECK_EQ(result.reduced.row_start[1], 0);
  GPRS_CHECK_EQ(result.stats[rule::row_rules].rows_removed, 0);
}

// 2 x0 - 3 x1 - x2 + 3 x4 <= -2, all integer. Dual fixing puts x0 at 0 and x2 at 3, and the
// row is divided by 3 with those terms in it: its upper side, less the terms, is
// -5.55e-17 where it should be 0. The side is set to 0 only once the terms have left the row,
// which a second presolve did and the first did not.
GPRS_TEST(engine_fixed_terms_leave_the_rows_before_the_last_round)
{
  std::vector<col_t> cols = {
    {0., 2., true}, {-0.5, 1., true}, {0., 3., true}, {0., 2., true}, {0., 1., true}};
  std::vector<row_t> rows = {{{{0, 2.}, {1, -3.}, {2, -1.}, {4, 3.}}, -inf, -2.}};
  model_t m               = with_cost(make_model(cols, rows), {0., 1., -1., 2., -2.});
  m.objective_offset      = 9.;
  options_t options;
  result_t result = presolve(m, options);
  GPRS_CHECK_EQ(result.status, status_t::reduced);
  GPRS_CHECK_EQ(result.reduced.nrows, 1);
  GPRS_CHECK(result.reduced.value == std::vector<double>({-1., 1.}));
  GPRS_CHECK_EQ(result.reduced.row_upper[0], 0.);
  check_fixed_point(result, options);

  // The same without a rebuild between the phases: the terms leave at the end.
  options.rebuild_dead_fraction  = 2.;
  options.rebuild_fixed_fraction = 2.;
  result                         = presolve(m, options);
  GPRS_CHECK_EQ(result.reduced.row_upper[0], 0.);
  check_fixed_point(result, options);
}

// -x1 - 3 x2 + 2 x3 <= 7 with x3 fixed at its lower bound of -1e21 by dual fixing: the term
// counts as infinite while it is in the row, and the row is redundant once it is in the side.
GPRS_TEST(engine_infinite_fixed_term_leaves_the_row)
{
  std::vector<col_t> cols = {{0., 1., true}, {0., 3., true}, {0., 2., true}, {-1e21, 4., false}};
  std::vector<row_t> rows = {{{{1, 2.}, {2, 3.}}, 7., 11.},
                             {{{1, -3.}, {2, -3.}}, -inf, -3.},
                             {{{0, 2.}, {2, -2.}}, -inf, 2.},
                             {{{1, 3.}, {3, 2.}}, -inf, 12.},
                             {{{1, -1.}, {2, -3.}, {3, 2.}}, -inf, 7.}};
  model_t m               = with_cost(make_model(cols, rows), {1., 0., 0., 0.});
  for (double fraction : {0.25, 2.}) {
    options_t options;
    options.rebuild_dead_fraction  = fraction;
    options.rebuild_fixed_fraction = fraction;
    result_t result                = presolve(m, options);
    GPRS_CHECK_EQ(result.status, status_t::reduced);
    GPRS_CHECK_EQ(result.reduced.nrows, 1);
    GPRS_CHECK(result.postsolve.row_origin == std::vector<int32_t>({0}));
    check_fixed_point(result, options);
  }
}

// Row 0 holds x3, which the start step fixes at 0, and x2 with a lower bound of -2.56e24: a term
// of 7.8e19 in its maximum activity. The term of x3 is 0 and its leaving changes no number, but
// with it the activity has one term more, and its error bound is then above what the guard of
// bound propagation allows for the small coefficient of x2. Without it the guard passes and the
// row gives x2 <= 0. The row is visited again when the term has left.
GPRS_TEST(engine_fixed_term_counts_in_the_error_bound)
{
  std::vector<col_t> cols = {{-inf, inf, false},
                             {0.001953125, inf, false},
                             {-2.56e24, inf, false},
                             {-0.5, 0., true},
                             {1., 2., true}};
  std::vector<row_t> rows = {
    {{{2, -3.0517578125e-05}, {3, -0.00390625}, {4, -0.01171875}}, -0.01171875, inf},
    {{{1, -3072.}}, -11., -11.},
    {{{1, -393216.}}, -inf, -896.},
    {{{0, -256.}, {3, 16.}}, -24., inf}};
  model_t m = with_cost(make_model(cols, rows), {0., -2048., -0.0078125, 1., -2.});
  for (double fraction : {0.25, 2., 0.}) {
    options_t options;
    options.rebuild_dead_fraction  = fraction;
    options.rebuild_fixed_fraction = fraction;
    result_t result                = presolve(m, options);
    GPRS_CHECK_EQ(result.status, status_t::reduced);
    GPRS_CHECK_EQ(result.reduced.nrows, 1);
    GPRS_CHECK_EQ(result.reduced.ncols, 2);
    GPRS_CHECK_EQ(result.reduced.col_lower[0], -2.56e24);
    // The bound is 0 moved outward by the error bound of the activity over the coefficient,
    // which the guard keeps below a tenth of the tolerance.
    GPRS_CHECK(result.reduced.col_upper[0] >= 0. && result.reduced.col_upper[0] <= 1e-7);
    check_fixed_point(result, options);
  }
}

// -x0 / 1024 + x1 >= 3.25 - 4 / 1024 and 3 x0 - x1 >= 8.75 with x0 in [-1, 4] have the one
// point (4, 3.25). Propagation takes the bounds towards it in steps that shrink by 3072 each
// and ends, at the step limit of a continuous bound, with x0 >= 3.9999994...: the first row
// then misses its lower side by 5e-10, inside the forcing tolerance, and fixing x0 at that bound
// violates the second row by 1.6e-6. The presolve reported a feasible model as infeasible.
GPRS_TEST(engine_forcing_row_keeps_a_continuous_column_that_is_off_its_bound)
{
  double c                  = 1. / 1024.;
  std::vector<col_t> cols   = {{-1., 4., false}, {-inf, inf, false}};
  std::vector<row_t> rows   = {{{{0, -c}, {1, 1.}}, 3.25 - 4. * c, inf},
                               {{{0, 3.}, {1, -1.}}, 8.75, inf}};
  model_t m                 = make_model(cols, rows);
  std::vector<double> point = {4., 3.25};
  GPRS_REQUIRE(point_feasible(check_point(m, point), tolerances_t()));
  for (bool primal_only : {false, true}) {
    options_t options;
    options.primal_only = primal_only;
    result_t result     = presolve(m, options);
    GPRS_REQUIRE(result.status == status_t::reduced);
    std::vector<double> crushed;
    result.postsolve.crush(point, crushed);
    point_check_t check = check_point(result.reduced, crushed);
    GPRS_CHECK_EQ(check.row_violation, 0.);
    GPRS_CHECK_EQ(check.bound_violation, 0.);
    check_fixed_point(result, options);
  }
}

// -3 x0 - x1 + x2 >= 3 * 2^60 with x1 fixed at -1 holds at x0 = -2^60, x2 = -1, the lower bound
// of x0. The side less the term of x1 is 3 * 2^60 - 1, which no double holds: with the term
// moved into the side before the last rounds the row read 3 * 2^60, one more than its largest
// activity, and the presolve reported a feasible model infeasible. The terms stay in the rows
// and the presolve ends without the fixed point.
GPRS_TEST(engine_fixed_terms_stay_in_a_row_that_cannot_take_them)
{
  double big              = 0x1p60;
  std::vector<col_t> cols = {{-big, 3.25, false}, {-1., -1., true}, {-1.25, 1e12, false}};
  std::vector<row_t> rows = {
    {{{0, -0.1}, {1, -1.}, {2, -3.}}, 1.1529215046068469e+17, 1.1586861121298814e+17},
    {{{0, -3.}, {1, -1.}, {2, 1.}}, 3. * big, inf},
    {{{0, -2.}, {2, 3.}}, 2.0752587082923244e+18, inf},
    {{{1, 1.}, {2, -2.}}, 0.75, inf}};
  model_t m                 = with_cost(make_model(cols, rows), {0., -2., -3.});
  std::vector<double> point = {-big, -1., -1.};
  point_check_t in          = check_point(m, point);
  GPRS_REQUIRE(in.row_violation == 0. && in.bound_violation == 0.);
  for (bool primal_only : {false, true}) {
    options_t options;
    options.primal_only = primal_only;
    result_t result     = presolve(m, options);
    GPRS_CHECK(result.status == status_t::reduced || result.status == status_t::reduced_to_empty);
    GPRS_CHECK(!result.fixed_point);
  }
}

// 2.9 x0 + 1.3 x1 <= 7.18 gives x0 an upper bound next to 1.4; the three rows below carry it
// on with a coefficient of 131072 each, to x4, whose upper bound 3 the point
// (1.4, 2.4, 1, 2, 3) meets. A bound of x0 one rounding step below the largest x0 of the first
// row arrives as x3 <= 2 - 3.8e-6 against x3 >= 2: infeasible, for a feasible model.
GPRS_TEST(engine_rounding_of_a_bound_is_not_carried_into_other_rows)
{
  double gain               = 131072.;
  std::vector<col_t> cols   = {{-inf, inf, false},
                               {2.4000000000000004, 7.4000000000000004, false},
                               {-inf, inf, false},
                               {-inf, inf, false},
                               {-inf, 3., false}};
  std::vector<row_t> rows   = {{{{0, 2.9000000000000004}, {1, 1.3}}, -inf, 7.1800000000000015},
                               {{{0, gain}, {2, 1.}}, 183501.80000000002, inf},
                               {{{2, gain}, {3, 1.}}, -inf, 131074.},
                               {{{3, gain}, {4, 1.}}, 262147., inf}};
  model_t m                 = make_model(cols, rows);
  std::vector<double> point = {1.4000000000000001, 2.4000000000000004, 1., 2., 3.};
  point_check_t in          = check_point(m, point);
  GPRS_REQUIRE(in.row_violation == 0. && in.bound_violation == 0.);
  options_t options;
  options.primal_only = true;
  result_t result     = presolve(m, options);
  GPRS_REQUIRE(result.status == status_t::reduced);
  std::vector<double> crushed;
  result.postsolve.crush(point, crushed);
  GPRS_CHECK(point_feasible(check_point(result.reduced, crushed), options.tol));
  check_fixed_point(result, options);
}
