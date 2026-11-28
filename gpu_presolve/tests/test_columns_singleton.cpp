/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The singleton-column rules (rules.hpp (h)) one call at a time on small states: what each
// reduction asks for, what it leaves in the state and on the stack, and what the undo gives.
// Every result is also read by the structure check, which proves it from the records alone.

#include "test_columns_support.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace gprs;
using columns_test::activity;
using columns_test::binary;
using columns_test::call_t;
using columns_test::free_col;
using columns_test::objective;
using columns_test::silent;
using columns_test::unit;
using gprs_check::structure_report_t;
using rule_test::col_t;
using rule_test::row_t;

namespace {

// Columns k, y, z: row 0 is the pivot row k + 2 y + 3 z, rows 1 and 2 keep y and z in the model.
std::vector<row_t> pivot_rows(double lower, double upper)
{
  return {{{{0, 1.}, {1, 2.}, {2, 3.}}, lower, upper},
          {{{1, 1.}, {2, 1.}}, -inf, 2.},
          {{{1, 1.}, {2, -1.}}, -1., inf}};
}

}  // namespace

// k + 2 y + 3 z = 6 with a free k: the row and the column leave, the cost of k moves onto y and
// z and into the offset, and the undo solves the row for k.
GPRS_TEST(columns_singleton_free_column_of_an_equation)
{
  call_t c({free_col, unit, unit}, pivot_rows(6., 6.), {2., 5., 6.});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 1);
  GPRS_CHECK_EQ(c.result.dual_findings, 0);
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton], 1);
  GPRS_CHECK(!c.result.limited && !c.result.stopped);
  GPRS_CHECK(!c.st.row_alive[0]);
  GPRS_CHECK(c.st.col_status[0] == col_status_t::removed);
  GPRS_CHECK_EQ(c.reduced.nrows, 2);
  GPRS_CHECK_EQ(c.reduced.ncols, 2);
  // cost - (2 / 1) * coefficient, and the offset 2 * 6.
  GPRS_CHECK(c.reduced.cost == std::vector<double>({1., 0.}));
  GPRS_CHECK_EQ(c.reduced.objective_offset, 12.);
  GPRS_CHECK_EQ(c.result.columns.cost_fill, 0);
  GPRS_CHECK_EQ(c.result.columns.saved_entries, 2);
  GPRS_CHECK_EQ(c.result.columns.max_growth, 0.);

  GPRS_REQUIRE(c.ps.col_from_rows.size() == 1);
  const col_from_rows_record_t& r = c.ps.col_from_rows[0];
  GPRS_CHECK(r.kind == col_from_rows_kind_t::defined && !r.dual);
  GPRS_CHECK_EQ(c.ps.record_cols[r.cols_begin].col, 0);
  GPRS_CHECK_EQ(c.ps.record_cols[r.cols_begin].lower, -inf);
  GPRS_CHECK_EQ(c.ps.record_coefficients[r.coefficients], 1.);
  GPRS_CHECK_EQ(c.ps.saved_rows[r.rows_begin].row, 0);
  GPRS_CHECK_EQ(c.ps.saved_rows[r.rows_begin].lower, 6.);
  GPRS_CHECK_EQ(c.ps.saved_rows[r.rows_begin].upper, 6.);
  GPRS_CHECK(c.ps.col_removed == std::vector<uint8_t>({1, 0, 0}));

  std::vector<double> x = c.postsolved({{1, 0.5}, {2, 0.25}});
  GPRS_CHECK(x == std::vector<double>({6. - 1. - 0.75, 0.5, 0.25}));
  GPRS_CHECK_EQ(objective(c.model, x), objective(c.reduced, {0.5, 0.25}));
  std::vector<double> crushed;
  c.ps.crush(x, crushed);
  GPRS_CHECK(crushed == std::vector<double>({0.5, 0.25}));
  structure_report_t s = c.structure();
  GPRS_CHECK(silent(s));
  GPRS_CHECK_EQ(s.defined_records, 1);
  GPRS_CHECK_EQ(s.box_tested, 0);
  // The pivot row is proved from its saved row, the two other rows from their reduced rows.
  GPRS_CHECK_EQ(s.kept_tested, 3);
}

// A cost that the substitution cancels is zero, and a column that gets one is counted.
GPRS_TEST(columns_singleton_moved_cost)
{
  call_t c({free_col, unit, unit}, pivot_rows(6., 6.), {1., 2., 0.});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 1);
  // 2 - 1 * 2 and 0 - 1 * 3.
  GPRS_CHECK(c.reduced.cost == std::vector<double>({0., -3.}));
  GPRS_CHECK_EQ(c.result.columns.cost_fill, 1);
  GPRS_CHECK(silent(c.structure()));

  // Without a cost nothing moves and no list of another column is touched for it.
  call_t plain({free_col, unit, unit}, pivot_rows(6., 6.), {0., 2., 0.});
  plain.run();
  GPRS_CHECK_EQ(plain.result.findings, 1);
  GPRS_CHECK(plain.reduced.cost == std::vector<double>({2., 0.}));
  GPRS_CHECK_EQ(plain.reduced.objective_offset, 0.);
  GPRS_CHECK(silent(plain.structure()));
}

// The bounds of the column must follow from the row: k = 6 - 2 y - 3 z lies in [1, 6].
GPRS_TEST(columns_singleton_needs_implied_bounds)
{
  struct bounds_t {
    double lower;
    double upper;
    bool taken;
  };
  for (bounds_t b : {bounds_t{1., 6., true},
                     bounds_t{0., 7., true},
                     bounds_t{-inf, 6., true},
                     bounds_t{1., inf, true},
                     bounds_t{2., 6., false},
                     bounds_t{1., 5., false},
                     bounds_t{1. + 1e-5, 6., false},
                     bounds_t{1., 6. - 1e-5, false},
                     // Within the tolerance of the test (implied_tolerance).
                     bounds_t{1. + 1e-8, 6. - 1e-8, true}}) {
    call_t c({{b.lower, b.upper, false}, unit, unit}, pivot_rows(6., 6.), {1., 0., 0.});
    c.run();
    GPRS_CHECK_EQ(c.result.findings, b.taken ? 1 : 0);
    GPRS_CHECK_EQ(c.ps.col_from_rows.size(), b.taken ? 1u : 0u);
    if (!b.taken) {
      GPRS_CHECK(c.st.row_alive[0] && c.st.is_live(0));
      GPRS_CHECK_EQ(c.result.columns.free_singleton_rejected_explicit, 0);
      continue;
    }
    structure_report_t s = c.structure();
    GPRS_CHECK(silent(s));
    GPRS_CHECK_EQ(s.box_tested, (b.lower > -inf) + (b.upper < inf));
    GPRS_CHECK_EQ(c.result.columns.max_growth, 3.);
  }
  // An end with an infinite term is no end: z without an upper bound leaves k unbounded below.
  call_t open({{1., inf, false}, unit, {0., inf, false}}, pivot_rows(6., 6.), {1., 0., 0.});
  open.run();
  GPRS_CHECK_EQ(open.result.findings, 0);
  call_t fine({{-inf, 6., false}, unit, {0., inf, false}}, pivot_rows(6., 6.), {1., 0., 0.});
  fine.run();
  GPRS_CHECK_EQ(fine.result.findings, 1);
  GPRS_CHECK(silent(fine.structure()));

  // After the switch to explicit bounds a refusal of the test is counted.
  call_t counted({{2., 6., false}, unit, unit}, pivot_rows(6., 6.), {1., 0., 0.});
  counted.family.bounds_became_explicit();
  counted.run();
  GPRS_CHECK_EQ(counted.result.findings, 0);
  GPRS_CHECK_EQ(counted.result.columns.free_singleton_rejected_explicit, 1);
}

// The structure check recomputes the test on the reduced model: a record whose column has
// bounds that the saved row does not imply fails its box.
GPRS_TEST(columns_singleton_box_is_proved_by_the_check)
{
  call_t c({{1., 6., false}, unit, unit}, pivot_rows(6., 6.));
  c.run();
  GPRS_REQUIRE(c.ps.col_from_rows.size() == 1);
  c.ps.record_cols[0].lower  = 2.;
  c.ps.record_cols[0].prefer = 2.;
  structure_report_t s       = c.structure();
  GPRS_CHECK_EQ(s.box_tested, 2);
  GPRS_CHECK_EQ(s.box_failures, 1);
  GPRS_CHECK_EQ(s.box_worst_col, 0);
  GPRS_CHECK_EQ(s.box_worst, 1. / 6.);
  // A bound of the record that is wider than that of the model is a bound failure.
  c.ps.record_cols[0].lower  = 0.;
  c.ps.record_cols[0].prefer = 0.;
  s                          = c.structure();
  GPRS_CHECK_EQ(s.box_failures, 0);
  GPRS_CHECK_EQ(s.bound_failures, 1);
}

// An integer column needs a row that is integral for it (fixed model 6): 2 k + y = 4 with y = 1
// would give k = 1.5.
GPRS_TEST(columns_singleton_integer_column_needs_an_integral_row)
{
  std::vector<row_t> keep = {{{{1, 1.}, {2, 1.}}, -inf, 3.}, {{{1, 1.}, {2, -1.}}, -2., inf}};
  auto rows               = [&](double a, double b, double side) {
    std::vector<row_t> all = {{{{0, a}, {1, b}}, side, side}};
    all.insert(all.end(), keep.begin(), keep.end());
    return all;
  };
  const col_t y{0., 3., true}, z{0., 3., true};
  // 2 k + y = 4: the ratio 1 / 2 is no integer.
  call_t half({{0., 2., true}, y, z}, rows(2., 1., 4.));
  half.run();
  GPRS_CHECK_EQ(half.result.findings, 0);
  // 2 k + 2 y = 3 has an integral ratio and no integral side.
  call_t side({{-inf, inf, true}, y, z}, rows(2., 2., 3.));
  side.run();
  GPRS_CHECK_EQ(side.result.findings, 0);
  // 2 k + 2 y = 4 is k = 2 - y, inside [-1, 2].
  call_t whole({{-1., 2., true}, y, z}, rows(2., 2., 4.));
  whole.run();
  GPRS_CHECK_EQ(whole.result.findings, 1);
  structure_report_t s = whole.structure();
  GPRS_CHECK(silent(s));
  GPRS_CHECK(whole.postsolved({{1, 3.}, {2, 1.}}) == std::vector<double>({-1., 3., 1.}));
  // A continuous column next to it gives the integer column no integer value.
  call_t mixed({{-inf, inf, true}, {0., 3., false}, z}, rows(2., 2., 4.));
  mixed.run();
  GPRS_CHECK_EQ(mixed.result.findings, 0);
  // A continuous singleton column takes any row.
  call_t continuous({{0., 2., false}, y, z}, rows(2., 1., 4.));
  continuous.run();
  GPRS_CHECK_EQ(continuous.result.findings, 1);
  GPRS_CHECK(silent(continuous.structure()));

  // The record of the first case, written by hand, fails the integral test of the check.
  whole.ps.saved_entries.value[0] = 1.;
  s                               = whole.structure();
  GPRS_CHECK_EQ(s.integral_failures, 1);
  GPRS_CHECK_EQ(s.first_integral_col, 0);
}

// The drift of a ratio that is an integer only within the tolerance: k - 1.0000000005 y = 0
// with y up to 2000000 is off by 1e-3 at k = y; with y up to 100 by 5e-8.
GPRS_TEST(columns_singleton_integer_column_and_the_drift)
{
  const std::vector<row_t> rows = {{{{0, 1.}, {1, -1.0000000005}}, 0., 0.},
                                   {{{1, 1.}, {2, 1.}}, -inf, 1e7},
                                   {{{1, 1.}, {2, -1.}}, -1e7, inf}};
  const col_t k{-inf, inf, true}, z{0., 1., true};
  call_t far({k, {0., 2000000., true}, z}, rows);
  far.run();
  GPRS_CHECK_EQ(far.result.findings, 0);
  call_t near({k, {0., 100., true}, z}, rows);
  near.run();
  GPRS_CHECK_EQ(near.result.findings, 1);
  GPRS_CHECK(silent(near.structure()));
  GPRS_CHECK(near.postsolved({{1, 100.}, {2, 0.}}) == std::vector<double>({100., 100., 0.}));
  // The share of the tolerance is a setting, inside the tolerance: the row may not be off by
  // more than the feasibility tolerance whatever the share is.
  call_t tight({k, {0., 100., true}, z}, rows);
  tight.opt.tol.integral_drift_share = 0.01;
  tight.run();
  GPRS_CHECK_EQ(tight.result.findings, 0);
  call_t loose({k, {0., 2000000., true}, z}, rows);
  loose.opt.tol.integral_drift_share = 2000.;
  loose.run();
  GPRS_CHECK_EQ(loose.result.findings, 0);
  call_t wide({k, {0., 2000000., true}, z}, rows);
  wide.opt.tol.integral_drift_share = 1.;
  wide.opt.tol.feasibility          = 1e-2;
  wide.run();
  GPRS_CHECK_EQ(wide.result.findings, 1);
  // The check holds such a record to its own tolerance in the units of the row.
  GPRS_CHECK_EQ(wide.structure().integral_failures, 1);
}

// An inequality as pivot row: the cost of the column pushes it against one side, which becomes
// the equation that defines it. A dual finding, and none in primal-only mode.
GPRS_TEST(columns_singleton_free_column_of_an_inequality)
{
  // min k: k + 2 y + 3 z >= 6 is tight, k = 6 - 2 y - 3 z in [1, 6].
  for (bool primal_only : {false, true}) {
    call_t c({{0., inf, false}, unit, unit}, pivot_rows(6., inf), {1., 0., 0.});
    c.opt.primal_only = primal_only;
    c.run();
    GPRS_CHECK_EQ(c.result.findings, primal_only ? 0 : 1);
    if (primal_only) { continue; }
    GPRS_CHECK_EQ(c.result.dual_findings, 1);
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton_dual], 1);
    GPRS_REQUIRE(c.ps.col_from_rows.size() == 1);
    GPRS_CHECK(c.ps.col_from_rows[0].kind == col_from_rows_kind_t::defined);
    GPRS_CHECK(c.ps.col_from_rows[0].dual);
    GPRS_CHECK_EQ(c.ps.saved_rows[0].lower, 6.);
    GPRS_CHECK_EQ(c.ps.saved_rows[0].upper, 6.);
    GPRS_CHECK(c.reduced.cost == std::vector<double>({-2., -3.}));
    GPRS_CHECK_EQ(c.reduced.objective_offset, 6.);
    GPRS_CHECK(c.postsolved({{1, 1.}, {2, 1.}}) == std::vector<double>({1., 1., 1.}));
    GPRS_CHECK(silent(c.structure()));
    // The same record in a result of primal-only mode breaks the contract of the stack.
    c.opt.primal_only      = true;
    c.result.dual_findings = 0;
    GPRS_CHECK_EQ(c.structure().stack_error, "dual record in a primal-only result at 0");
  }
  // The side that the cost pushes against, for each sign of the cost and of the coefficient,
  // on a ranged row 3 <= a k + 2 y + 3 z <= 6 with a free k.
  struct side_t {
    double a;
    double cost;
    double side;
  };
  for (side_t t :
       {side_t{1., 1., 3.}, side_t{1., -1., 6.}, side_t{-1., 1., 6.}, side_t{-1., -1., 3.}}) {
    call_t c({free_col, unit, unit},
             {{{{0, t.a}, {1, 2.}, {2, 3.}}, 3., 6.},
              {{{1, 1.}, {2, 1.}}, -inf, 2.},
              {{{1, 1.}, {2, -1.}}, -1., inf}},
             {t.cost, 0., 0.});
    c.run();
    GPRS_CHECK_EQ(c.result.dual_findings, 1);
    GPRS_REQUIRE(c.ps.saved_rows.size() == 1);
    GPRS_CHECK_EQ(c.ps.saved_rows[0].upper, t.side);
    std::vector<double> x = c.postsolved({{1, 0.5}, {2, 0.5}});
    GPRS_CHECK_EQ(activity(c.model, 0, x), t.side);
    GPRS_CHECK(silent(c.structure()));
  }
  // Without a cost no side is tight, and a side that is infinite stops nothing: the column
  // with cost -1 in a row >= is the lock rule's.
  call_t no_cost({{0., inf, false}, unit, unit}, pivot_rows(6., inf));
  no_cost.opt.sub_enabled[sub_rule::forcing_column] = false;
  no_cost.run();
  GPRS_CHECK_EQ(no_cost.result.findings, 0);
  call_t no_lock({{0., inf, false}, unit, unit}, pivot_rows(6., inf), {-1., 0., 0.});
  no_lock.run();
  GPRS_CHECK_EQ(no_lock.result.findings, 0);
  // The bound behind the move is tested as well: with k <= 5 the value 6 - 2 y - 3 z can pass
  // it at a point of the reduced model.
  call_t behind({{0., 5., false}, unit, unit}, pivot_rows(6., inf), {1., 0., 0.});
  behind.run();
  GPRS_CHECK_EQ(behind.result.findings, 0);
}

// The switches of the two reductions.
GPRS_TEST(columns_singleton_pivot_switches)
{
  call_t equation({free_col, unit, unit}, pivot_rows(6., 6.));
  equation.opt.sub_enabled[sub_rule::free_singleton]      = false;
  equation.opt.sub_enabled[sub_rule::zero_cost_singleton] = false;
  equation.run();
  GPRS_CHECK_EQ(equation.result.findings, 0);
  call_t inequality({{0., inf, false}, unit, unit}, pivot_rows(6., inf), {1., 0., 0.});
  inequality.opt.sub_enabled[sub_rule::free_singleton_dual] = false;
  inequality.run();
  GPRS_CHECK_EQ(inequality.result.findings, 0);
  // The switch of one does not stop the other.
  call_t other({free_col, unit, unit}, pivot_rows(6., 6.));
  other.opt.sub_enabled[sub_rule::free_singleton_dual] = false;
  other.run();
  GPRS_CHECK_EQ(other.result.findings, 1);
}

// Of several singleton columns of a row that pass, the one with the smallest index leaves.
GPRS_TEST(columns_singleton_smallest_index_first)
{
  // k0 + k1 + 2 y + 3 z = 6 with both free: k0 goes, and k1 is then left in a row that is gone.
  call_t c({free_col, free_col, unit, unit},
           {{{{0, 1.}, {1, 1.}, {2, 2.}, {3, 3.}}, 6., 6.},
            {{{2, 1.}, {3, 1.}}, -inf, 2.},
            {{{2, 1.}, {3, -1.}}, -1., inf}},
           {1., 1., 0., 0.});
  c.opt.sub_enabled[sub_rule::forcing_column] = false;
  c.run();
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton], 1);
  GPRS_REQUIRE(c.ps.col_from_rows.size() == 1);
  GPRS_CHECK_EQ(c.ps.record_cols[0].col, 0);
  // The saved row holds k1, which is live when the record is pushed.
  GPRS_CHECK_EQ(c.ps.saved_rows[0].end - c.ps.saved_rows[0].begin, 3);
  GPRS_CHECK(c.st.is_live(1) && c.st.csc.size[1] == 0);
  GPRS_CHECK_EQ(c.st.cost[1], 0.);
  GPRS_CHECK(silent(c.structure()));
  // The first that passes: k0 with bounds that the row does not imply is skipped.
  call_t second({{0., 1., false}, free_col, unit, unit},
                {{{{0, 1.}, {1, 1.}, {2, 2.}, {3, 3.}}, 6., 6.},
                 {{{2, 1.}, {3, 1.}}, -inf, 2.},
                 {{{2, 1.}, {3, -1.}}, -1., inf}},
                {1., 1., 0., 0.});
  second.run();
  GPRS_REQUIRE(second.ps.col_from_rows.size() == 1);
  GPRS_CHECK_EQ(second.ps.record_cols[0].col, 1);
  GPRS_CHECK(silent(second.structure()));
}

// The side of the free part is what the record holds, and it must be readable: a fixed column
// at 1e15 swallows a side of 3 (the guard of free_side, src/saved_row.hpp).
GPRS_TEST(columns_singleton_side_less_the_fixed_terms)
{
  auto rows = pivot_rows(6., 6.);
  rows[0].entries.push_back({3, 1.});
  for (double fixed : {1000., 1e15}) {
    call_t c({free_col, unit, unit, {fixed, fixed + 1., false}}, rows, {1., 0., 0., 0.});
    fix_column(c.st, c.ps, 3, fixed);
    c.run();
    bool readable = fixed == 1000.;
    GPRS_CHECK_EQ(c.result.findings, readable ? 1 : 0);
    if (!readable) { continue; }
    GPRS_CHECK_EQ(c.ps.saved_rows[0].upper, 6. - fixed);
    GPRS_CHECK_EQ(c.ps.saved_rows[0].end - c.ps.saved_rows[0].begin, 2);
    GPRS_CHECK(c.postsolved({{1, 1.}, {2, 1.}}) ==
               std::vector<double>({1. - fixed, 1., 1., fixed}));
    GPRS_CHECK_EQ(c.reduced.objective_offset, 6. - fixed);
    GPRS_CHECK(silent(c.structure()));
  }
  // A side that is itself beyond the size is not read either.
  call_t huge({free_col, unit, unit}, pivot_rows(3e9, 3e9));
  huge.run();
  GPRS_CHECK_EQ(huge.result.findings, 0);
}

// The growth factor (fixed model 8): x1 + 1000 x2 = 0, x2 + 1000 x3 = 0 with x3 in [-1e-3, 1e-3].
// The first substitution gives x2 the factor 1000; the second would reach 1e6 and is refused,
// which the call reports as limited. Both columns have a cost, so that they are no zero-cost
// singleton columns.
GPRS_TEST(columns_singleton_growth_factor_follows_a_chain)
{
  std::vector<col_t> cols = {{-1000., 1000., false}, {-1., 1., false}, {-1e-3, 1e-3, false}, unit};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1000.}}, 0., 0.},
                             {{{1, 1.}, {2, 1000.}}, 0., 0.},
                             {{{2, 1.}, {3, 1.}}, -inf, 1.},
                             {{{2, 1.}, {3, -1.}}, -1., inf}};
  const std::vector<double> cost = {1., 1., 0., 0.};
  call_t c(cols, rows, cost);
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 1);
  GPRS_CHECK(c.result.limited);
  GPRS_CHECK_EQ(c.result.columns.max_growth, 1000.);
  GPRS_CHECK_EQ(c.result.columns.growth_refused, 1);
  GPRS_CHECK_EQ(c.result.columns.growth_refused_row, 0);
  GPRS_CHECK(c.st.row_alive[1] && c.st.is_live(1));
  GPRS_CHECK(silent(c.structure()));
  // Without a cost the refused column leaves as a zero-cost singleton column of its equation,
  // and nothing is left undone.
  call_t free_of_cost(cols, rows, {1., 1000., 0., 0.});
  free_of_cost.run();
  GPRS_CHECK_EQ(free_of_cost.result.sub_findings[sub_rule::free_singleton], 1);
  GPRS_CHECK_EQ(free_of_cost.result.sub_findings[sub_rule::zero_cost_singleton], 1);
  GPRS_CHECK_EQ(free_of_cost.result.columns.growth_refused, 1);
  GPRS_CHECK(!free_of_cost.result.limited);
  GPRS_CHECK(silent(free_of_cost.structure()));
  // The same in the engine: no fixed point, and the second presolve, which starts without the
  // factor, takes the second equation.
  options_t options;
  options.enabled.fill(false);
  options.enabled[rule::singleton_cols] = true;
  result_t result                       = presolve(c.model, options);
  GPRS_CHECK(!result.fixed_point);
  GPRS_CHECK_EQ(result.reduced.nrows, 3);
  GPRS_CHECK_EQ(result.columns.growth_refused, 1);
  GPRS_CHECK_EQ(presolve(result.reduced, options).reduced.nrows, 2);

  // A ratio above the limit in one row is a property of the state: no limited.
  rows[0].entries[1].second = 2000.;
  cols[0]                   = {-2000., 2000., false};
  call_t steep(cols, rows, cost);
  steep.run();
  GPRS_CHECK_EQ(steep.result.columns.growth_refused_row, 1);
  // x1 is refused, and x2, which is no singleton column, is left as well.
  GPRS_CHECK_EQ(steep.result.findings, 0);
  GPRS_CHECK(!steep.result.limited);
  // A column without a bound has none to pass: any ratio.
  cols[0] = free_col;
  call_t unbounded(cols, rows, cost);
  unbounded.run();
  GPRS_CHECK_EQ(unbounded.result.sub_findings[sub_rule::free_singleton], 2);
  GPRS_CHECK(!unbounded.result.limited);
  GPRS_CHECK_EQ(unbounded.result.columns.max_growth, 1000.);
  // The limit is a setting.
  cols[0] = {-2000., 2000., false};
  call_t wide(cols, rows, cost);
  wide.opt.pivot_growth_limit = 1e7;
  wide.run();
  GPRS_CHECK_EQ(wide.result.sub_findings[sub_rule::free_singleton], 2);
  GPRS_CHECK_EQ(wide.result.columns.max_growth, 2e6);
}

// A chain of equations (fixed model 9): x(t-1) + x(t) + u(t) + v(t) = d(t) for t = 1 .. 30,
// with only x(30) a singleton column. Each deleted row leaves the next column as one, and the
// call follows the chain to its end.
GPRS_TEST(columns_singleton_chain_ends_in_one_call)
{
  const int32_t links = 30;
  // Columns x(0) .. x(links), then u(1), v(1), ..; two rows hold every u and v, and a third
  // x(0), which has bounds.
  std::vector<col_t> cols(links + 1, free_col);
  cols[0] = {0., 5., false};
  std::vector<row_t> rows;
  for (int32_t t = 1; t <= links; ++t) {
    int32_t u = cols.size();
    cols.push_back(unit);
    cols.push_back(unit);
    rows.push_back({{{t - 1, 1.}, {t, 1.}, {u, 1.}, {u + 1, 1.}}, (double)t, (double)t});
  }
  row_t all_u{{}, -inf, 20.}, all_v{{}, 5., inf}, first{{{0, 1.}, {links + 1, 1.}}, -inf, 3.};
  for (int32_t t = 1; t <= links; ++t) {
    all_u.entries.push_back({links + 2 * t - 1, 1.});
    all_v.entries.push_back({links + 2 * t, 1.});
  }
  rows.push_back(all_u);
  rows.push_back(all_v);
  rows.push_back(first);
  std::vector<double> cost(cols.size(), 0.);
  cost[links] = 1.;
  call_t c(cols, rows, cost);
  c.run();
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton], links);
  GPRS_CHECK_EQ(c.ps.col_from_rows.size(), (size_t)links);
  GPRS_CHECK_EQ(c.reduced.nrows, 3);
  GPRS_CHECK_EQ(c.ps.record_cols[0].col, links);
  GPRS_CHECK_EQ(c.ps.record_cols[links - 1].col, 1);
  GPRS_CHECK(c.st.is_live(0));
  structure_report_t s = c.structure();
  GPRS_CHECK(silent(s));
  GPRS_CHECK_EQ(s.defined_records, links);

  // In the engine: one call of the family, and a fixed point.
  options_t options;
  result_t result = presolve(c.model, options);
  GPRS_CHECK(result.fixed_point);
  GPRS_CHECK_EQ(result.sub_findings[sub_rule::free_singleton], links);
  GPRS_CHECK_EQ(result.stats[rule::singleton_cols].rows_removed, links);
  GPRS_CHECK(result.rounds <= 4);
  if (result.status == status_t::reduced) {
    GPRS_CHECK_EQ(presolve(result.reduced, options).status, status_t::unchanged);
  }
}

// Rows with two finite sides lose their zero-cost continuous singleton columns in one record
// (fixed model 15): 2 <= s1 + 2 s2 - s3 + y + z <= 5 with s1, s3 in [0, 1] and s2 in [0, 2].
GPRS_TEST(columns_singleton_zero_cost_columns_leave_together)
{
  // Columns s1, y, s2, z, s3.
  std::vector<col_t> cols = {unit, {0., 4., false}, {0., 2., false}, {0., 4., false}, unit};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}, {2, 2.}, {3, 1.}, {4, -1.}}, 2., 5.},
                             {{{1, 1.}, {3, 1.}}, -inf, 7.},
                             {{{1, 1.}, {3, -1.}}, -3., inf}};
  call_t c(cols, rows);
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 3);
  GPRS_CHECK_EQ(c.result.dual_findings, 0);
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], 3);
  GPRS_CHECK_EQ(c.result.columns.saved_entries, 2);
  GPRS_REQUIRE(c.ps.col_from_rows.size() == 1);
  const col_from_rows_record_t& r = c.ps.col_from_rows[0];
  GPRS_CHECK(r.kind == col_from_rows_kind_t::interval);
  GPRS_CHECK_EQ(r.cols_end - r.cols_begin, 3);
  GPRS_CHECK_EQ(c.ps.record_cols[0].col, 0);
  GPRS_CHECK_EQ(c.ps.record_cols[1].col, 2);
  GPRS_CHECK_EQ(c.ps.record_cols[2].col, 4);
  GPRS_CHECK(c.ps.record_coefficients == std::vector<double>({1., 2., -1.}));
  GPRS_CHECK_EQ(c.ps.saved_rows[0].lower, 2.);
  GPRS_CHECK_EQ(c.ps.saved_rows[0].upper, 5.);
  // The sides widen by the sums of the ranges: 2 - (1 + 4 + 0) and 5 - (0 + 0 - 1).
  GPRS_CHECK(c.st.row_alive[0]);
  GPRS_CHECK_EQ(c.st.row_lower[0], -3.);
  GPRS_CHECK_EQ(c.st.row_upper[0], 6.);
  GPRS_CHECK_EQ(c.reduced.ncols, 2);
  GPRS_CHECK_EQ(c.reduced.nnz(), 6);
  structure_report_t s = c.structure();
  GPRS_CHECK(silent(s));
  GPRS_CHECK_EQ(s.interval_records, 1);
  GPRS_CHECK_EQ(s.snapshot_tested, 1);

  // The undo: the columns start at their lower bounds and move, in the order of the record, as
  // far as the row needs.
  struct point_t {
    double y;
    double z;
    std::vector<double> s;
  };
  for (const point_t& p : {point_t{1., 2., {0., 0., 0.}},   // 3 is inside [2, 5]
                           point_t{0., 0., {1., 0.5, 0.}},  // 0: 2 is missing
                           point_t{0., -0., {1., 0.5, 0.}},
                           point_t{2., 4., {0., 0., 1.}},  // 6: one too many, only s3 lowers
                           point_t{0.5, 1., {0.5, 0., 0.}}}) {
    std::vector<double> x = c.postsolved({{1, p.y}, {3, p.z}});
    GPRS_CHECK(x == std::vector<double>({p.s[0], p.y, p.s[1], p.z, p.s[2]}));
    double row = activity(c.model, 0, x);
    GPRS_CHECK(row >= 2. && row <= 5.);
  }
  // At the ends of the widened sides every column is at the end of its range.
  // (A point of the reduced model has y + z in [-3, 6]; y and z are not below 0 here.)
  std::vector<double> low = c.postsolved({{1, 0.}, {3, 0.}});
  GPRS_CHECK_EQ(activity(c.model, 0, low), 2.);
}

// What the zero-cost rule leaves alone, and the sides it gives.
GPRS_TEST(columns_singleton_zero_cost_conditions)
{
  // Columns s, y, z and the row lower <= s + y + z <= upper; two rows keep y and z.
  auto rows = [](double lower, double upper) {
    return std::vector<row_t>{{{{0, 1.}, {1, 1.}, {2, 1.}}, lower, upper},
                              {{{1, 1.}, {2, 1.}}, -inf, 7.},
                              {{{1, 1.}, {2, -1.}}, -3., inf}};
  };
  const col_t y{0., 4., false};
  auto taken = [&](const col_t& s, double lower, double upper, double cost) {
    call_t c({s, y, y}, rows(lower, upper), {cost, 0., 0.});
    c.opt.sub_enabled[sub_rule::forcing_column] = false;
    c.run();
    if (c.result.findings > 0) { GPRS_CHECK(silent(c.structure())); }
    return c.result.sub_findings[sub_rule::zero_cost_singleton];
  };
  GPRS_CHECK_EQ(taken(unit, 2., 5., 0.), 1);
  // An equation becomes a range.
  GPRS_CHECK_EQ(taken({0., 3., false}, 4., 4., 0.), 1);
  // A cost, an integer column, a row with one finite side.
  GPRS_CHECK_EQ(taken(unit, 2., 5., 1.), 0);
  GPRS_CHECK_EQ(taken(binary, 2., 5., 0.), 0);
  GPRS_CHECK_EQ(taken(unit, -inf, 5., 0.), 0);
  GPRS_CHECK_EQ(taken(unit, 2., inf, 0.), 0);
  // A bound whose term is beyond the size of a readable side.
  GPRS_CHECK_EQ(taken({0., 1e15, false}, 2., 5., 0.), 0);
  // The equation k + y + z = 4 with a free k is the free singleton column's.
  {
    call_t c({free_col, y, y}, rows(4., 4.));
    c.run();
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton], 1);
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], 0);
  }
  // An infinite bound makes a side infinite: s >= 0 leaves s + y + z in [2, 5] as y + z <= 5.
  {
    call_t c({{0., inf, false}, y, y}, rows(2., 5.));
    c.opt.sub_enabled[sub_rule::forcing_column] = false;
    c.run();
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], 1);
    GPRS_CHECK_EQ(c.st.row_lower[0], -inf);
    GPRS_CHECK_EQ(c.st.row_upper[0], 5.);
    GPRS_CHECK(c.postsolved({{1, 0.}, {2, 1.}}) == std::vector<double>({1., 0., 1.}));
    GPRS_CHECK(silent(c.structure()));
  }
  // Two infinite bounds delete the row, which then counts nothing against the saved rows.
  {
    call_t c({free_col, y, y}, rows(2., 5.));
    c.opt.sub_enabled[sub_rule::forcing_column] = false;
    c.opt.saved_row_factor                      = 0.;
    c.run();
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], 1);
    GPRS_CHECK(!c.st.row_alive[0]);
    GPRS_CHECK(!c.result.limited);
    // prefer is 0: 4 is inside, 6 is one too many, 1 is one short.
    GPRS_CHECK(c.postsolved({{1, 2.}, {2, 2.}}) == std::vector<double>({0., 2., 2.}));
    GPRS_CHECK(c.postsolved({{1, 4.}, {2, 2.}}) == std::vector<double>({-1., 4., 2.}));
    GPRS_CHECK(c.postsolved({{1, 0.}, {2, 1.}}) == std::vector<double>({1., 0., 1.}));
    GPRS_CHECK(silent(c.structure()));
  }
  // A row whose free entries are all such columns keeps them.
  {
    call_t c({unit, unit}, {{{{0, 1.}, {1, 1.}}, 0.5, 1.5}});
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
  }
  // The switch, and primal-only mode.
  {
    call_t c({unit, y, y}, rows(2., 5.));
    c.opt.sub_enabled[sub_rule::zero_cost_singleton] = false;
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
    call_t primal({unit, y, y}, rows(2., 5.));
    primal.opt.primal_only = true;
    primal.run();
    GPRS_CHECK_EQ(primal.result.findings, 0);
  }
}

// A row that stays is saved at every visit that takes columns from it. Beyond
// opt.saved_row_factor times the nonzeros of the model the visit does nothing and reports
// limited.
GPRS_TEST(columns_singleton_saved_row_limit)
{
  std::vector<col_t> cols = {unit, {0., 4., false}, {0., 4., false}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}, {2, 1.}}, 2., 5.},
                             {{{1, 1.}, {2, 1.}}, -inf, 7.},
                             {{{1, 1.}, {2, -1.}}, -3., inf}};
  // 7 nonzeros; the visit saves 2 entries.
  for (double factor : {0., 0.25, 0.3, 4.}) {
    call_t c(cols, rows);
    c.opt.saved_row_factor = factor;
    c.run();
    bool fits = 2. <= factor * 7.;
    GPRS_CHECK_EQ(c.result.findings, fits ? 1 : 0);
    GPRS_CHECK_EQ(c.result.limited, !fits);
    GPRS_CHECK_EQ(c.result.columns.saved_row_refused, fits ? 0 : 1);
    GPRS_CHECK_EQ(c.result.columns.saved_entries, fits ? 2 : 0);
  }
}

// The forcing column: no cost, free to go down and held from above by all its rows. The rows
// leave with it, and the undo gives it the largest value that they and its bound allow.
GPRS_TEST(columns_singleton_forcing_column)
{
  // k + y <= 4 and -2 k + z >= -6, with k <= 3 and no lower bound.
  auto rows = [](double other_lower) {
    return std::vector<row_t>{{{{0, 1.}, {1, 1.}}, other_lower, 4.},
                              {{{0, -2.}, {2, 1.}}, -6., inf},
                              {{{1, 1.}, {2, 1.}}, -inf, 7.},
                              {{{1, 1.}, {2, -1.}}, -3., inf}};
  };
  const col_t y{0., 4., false};
  for (bool integer : {false, true}) {
    call_t c({{-inf, 3., integer}, y, y}, rows(-inf));
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 1);
    GPRS_CHECK_EQ(c.result.dual_findings, 0);
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::forcing_column], 1);
    GPRS_CHECK(!c.st.row_alive[0] && !c.st.row_alive[1]);
    GPRS_CHECK(c.st.col_status[0] == col_status_t::removed);
    GPRS_REQUIRE(c.ps.col_from_rows.size() == 1);
    const col_from_rows_record_t& r = c.ps.col_from_rows[0];
    GPRS_CHECK(r.kind == col_from_rows_kind_t::forcing);
    GPRS_CHECK_EQ(r.rows_end - r.rows_begin, 2);
    GPRS_CHECK(c.ps.record_coefficients == std::vector<double>({1., -2.}));
    GPRS_CHECK_EQ(c.ps.saved_rows[0].upper, 4.);
    GPRS_CHECK_EQ(c.ps.saved_rows[0].lower, -inf);
    GPRS_CHECK_EQ(c.ps.saved_rows[1].lower, -6.);
    GPRS_CHECK_EQ(c.ps.saved_rows[1].upper, inf);
    GPRS_CHECK_EQ(c.result.columns.saved_entries, 2);
    // min(3, 4 - y, 3 + z / 2).
    GPRS_CHECK_EQ(c.postsolved({{1, 0.}, {2, 4.}})[0], 3.);
    GPRS_CHECK_EQ(c.postsolved({{1, 2.5}, {2, 4.}})[0], integer ? 1. : 1.5);
    GPRS_CHECK_EQ(c.postsolved({{1, 4.}, {2, -3.}})[0], integer ? 0. : 0.);
    GPRS_CHECK_EQ(c.postsolved({{1, 0.}, {2, -3.}})[0], integer ? 1. : 1.5);
    structure_report_t s = c.structure();
    GPRS_CHECK(silent(s));
    GPRS_CHECK_EQ(s.forcing_records, 1);
  }
  // The mirror: free to go up, held from below; the smallest value, rounded up.
  {
    call_t c({{-3., inf, true}, y, y},
             {{{{0, -1.}, {1, 1.}}, -inf, 4.},
              {{{0, 2.}, {2, 1.}}, -6., inf},
              {{{1, 1.}, {2, 1.}}, -inf, 7.},
              {{{1, 1.}, {2, -1.}}, -3., inf}});
    c.run();
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::forcing_column], 1);
    // max(-3, y - 4, -3 - z / 2).
    GPRS_CHECK_EQ(c.postsolved({{1, 2.5}, {2, 4.}})[0], -1.);
    GPRS_CHECK_EQ(c.postsolved({{1, 0.}, {2, -3.}})[0], -1.);
    GPRS_CHECK(silent(c.structure()));
  }
  // A lock in the free direction, a cost, a finite bound in the free direction, the switch,
  // primal-only mode.
  {
    call_t locked({{-inf, 3., false}, y, y}, rows(0.));
    locked.run();
    GPRS_CHECK_EQ(locked.result.findings, 0);
    call_t cost({{-inf, 3., false}, y, y}, rows(-inf), {-1., 0., 0.});
    cost.run();
    GPRS_CHECK_EQ(cost.result.findings, 0);
    call_t bounded({{-9., 3., false}, y, y}, rows(-inf));
    bounded.run();
    GPRS_CHECK_EQ(bounded.result.findings, 0);
    call_t off({{-inf, 3., false}, y, y}, rows(-inf));
    off.opt.sub_enabled[sub_rule::forcing_column] = false;
    off.run();
    GPRS_CHECK_EQ(off.result.findings, 0);
    call_t primal({{-inf, 3., false}, y, y}, rows(-inf));
    primal.opt.primal_only = true;
    primal.run();
    GPRS_CHECK_EQ(primal.result.findings, 0);
  }
  // A row of the column without a finite side is deleted and not saved; one whose side less
  // the fixed terms is not readable leaves the column where it is.
  {
    std::vector<row_t> with_free = rows(-inf);
    with_free.push_back({{{0, 5.}, {1, 1.}}, -inf, inf});
    call_t c({{-inf, 3., false}, y, y}, with_free);
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 1);
    GPRS_CHECK(!c.st.row_alive[4]);
    GPRS_CHECK_EQ(c.ps.saved_rows.size(), 2u);
    GPRS_CHECK(silent(c.structure()));

    std::vector<row_t> with_fixed = rows(-inf);
    with_fixed[0].entries.push_back({3, 1.});
    call_t swallowed({{-inf, 3., false}, y, y, {1e15, 1e15 + 1., false}}, with_fixed);
    fix_column(swallowed.st, swallowed.ps, 3, 1e15);
    swallowed.run();
    GPRS_CHECK_EQ(swallowed.result.findings, 0);
    GPRS_CHECK(swallowed.ps.saved_entries.col.empty());
  }
}

// The contract of the forcing record is what proves its rows: the check rejects a record with
// a row that holds the column from the other end.
GPRS_TEST(columns_singleton_forcing_record_is_read_by_the_check)
{
  call_t c({{-inf, 3., false}, {0., 4., false}, {0., 4., false}},
           {{{{0, 1.}, {1, 1.}}, -inf, 4.},
            {{{0, -2.}, {2, 1.}}, -6., inf},
            {{{1, 1.}, {2, 1.}}, -inf, 7.},
            {{{1, 1.}, {2, -1.}}, -3., inf}});
  c.run();
  GPRS_REQUIRE(c.ps.col_from_rows.size() == 1);
  postsolve_t right           = c.ps;
  c.ps.record_coefficients[1] = 2.;
  GPRS_CHECK_EQ(c.structure().stack_error, "row of forcing record 0");
  c.ps                     = right;
  c.ps.saved_rows[0].lower = 0.;
  GPRS_CHECK_EQ(c.structure().stack_error, "row of forcing record 0");
  c.ps                       = right;
  c.ps.record_cols[0].lower  = -9.;
  c.ps.record_cols[0].prefer = -9.;
  GPRS_CHECK_EQ(c.structure().stack_error, "bound of forcing record 0");
  // A side that is one too wide lets the row be violated: the kept-rows part reads it.
  c.ps                     = right;
  c.ps.saved_rows[0].upper = 5.;
  structure_report_t s     = c.structure();
  GPRS_CHECK(s.stack_error.empty());
  GPRS_CHECK_EQ(s.kept_failures, 1);
  GPRS_CHECK_EQ(s.kept_worst_row, 0);
}

// The budget: the call stops between two columns or rows and returns what it has not served; a
// listed row without a singleton column costs one visit and no scan.
GPRS_TEST(columns_singleton_budget_and_work)
{
  // Twenty rows x(i) + x(i + 1) <= 1 over binaries: no singleton column but the two ends, which
  // nothing takes (a cost, an inequality whose side the cost does not push against).
  const int32_t n = 21;
  std::vector<col_t> cols(n, binary);
  std::vector<row_t> rows;
  for (int32_t i = 0; i + 1 < n; ++i) {
    rows.push_back({{{i, 1.}, {i + 1, 1.}}, -inf, 1.});
  }
  std::vector<double> cost(n, 1.);
  call_t c(cols, rows, cost);
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 0);
  // Every column is listed at construction and no row: one visit per column, and for the two
  // singleton columns their entry and the scan of their row.
  GPRS_CHECK_EQ(c.budget.used, line_visit_work * n + 2 * (1 + 2));
  // A row event lists the row: one visit, no scan.
  c.st.set_col_upper(5, 0.5);
  work_budget_t budget;
  budget.limit = budget.stall_limit = rule_test::unlimited;
  rule_result_t again               = c.family.run(c.st, budget, c.ps, c.opt);
  GPRS_CHECK_EQ(again.findings, 0);
  GPRS_CHECK_EQ(budget.used, 2 * line_visit_work);

  // A budget of one visit serves one column and returns the rest.
  call_t tight(cols, rows, cost);
  tight.budget.limit = 1;
  tight.run();
  GPRS_CHECK(tight.result.stopped);
  GPRS_CHECK_EQ(tight.budget.used, line_visit_work + 1 + 2);
  GPRS_CHECK_EQ(tight.st.col_frontier.list[col_list::singleton_cols].size(), (size_t)(n - 1));
  // The next call goes on with them.
  tight.budget.limit = rule_test::unlimited;
  tight.budget.used  = 0;
  tight.run();
  GPRS_CHECK(!tight.result.stopped);
  GPRS_CHECK_EQ(tight.budget.used, line_visit_work * (n - 1) + 1 + 2);
}

// A forcing column whose rows do not fit the rest of the budget waits, unless it is the first
// reduction of the call.
GPRS_TEST(columns_singleton_forcing_column_waits_for_the_budget)
{
  // Two forcing columns k0 and k1, each in one row of five entries.
  std::vector<col_t> cols = {{-inf, 3., false}, {-inf, 3., false}, unit, unit, unit, unit};
  std::vector<row_t> rows = {{{{0, 1.}, {2, 1.}, {3, 1.}, {4, 1.}, {5, 1.}}, -inf, 4.},
                             {{{1, 1.}, {2, 1.}, {3, 1.}, {4, 1.}, {5, 1.}}, -inf, 4.},
                             {{{2, 1.}, {3, 1.}, {4, 1.}, {5, 1.}}, 1., 3.}};
  call_t c(cols, rows);
  c.budget.limit = 30;
  c.run();
  // The first column is taken although its rows cost more than is left; the second waits.
  GPRS_CHECK_EQ(c.result.findings, 1);
  GPRS_CHECK(c.result.stopped);
  GPRS_CHECK(c.st.is_live(1));
  GPRS_CHECK((c.st.col_frontier.pending[1] & list_bit(col_list::singleton_cols)) != 0);
  c.budget.limit = rule_test::unlimited;
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 1);
  GPRS_CHECK(!c.result.stopped);
  GPRS_CHECK_EQ(c.ps.col_from_rows.size(), 2u);
  GPRS_CHECK(silent(c.structure()));
}

// The test of the bound behind the move reads the end of the activity that no finite side of
// the pivot row reads: a bound change there lists the row for the family all the same.
GPRS_TEST(columns_singleton_bound_that_no_side_reads)
{
  // min k: k + y >= 2 with k in [0, 1] and y in [0, 1.5]. k = 2 - y passes 1 while y can be 0.
  call_t c({unit, {0., 1.5, false}, unit},
           {{{{0, 1.}, {1, 1.}}, 2., inf},
            {{{1, 1.}, {2, 1.}}, -inf, 2.25},
            {{{1, 1.}, {2, -1.}}, -1., inf}},
           {1., 0., 0.});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 0);
  // y >= 1.25 changes the smallest activity of row 0, whose upper side is infinite.
  c.st.set_col_lower(1, 1.25);
  GPRS_CHECK((c.st.row_frontier.pending[0] & list_bit(row_list::singleton_cols)) != 0);
  c.run();
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton_dual], 1);
  GPRS_CHECK(c.st.col_status[0] == col_status_t::removed);
  GPRS_CHECK(!c.st.row_alive[0]);
  GPRS_CHECK(silent(c.structure()));
}

// The family keeps nothing of a call but the growth factors: a second call on the state that
// the first left finds nothing, and a state whose lists are empty costs nothing.
GPRS_TEST(columns_singleton_second_call_finds_nothing)
{
  call_t c({free_col, unit, unit}, pivot_rows(6., 6.), {2., 5., 6.});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 1);
  uint64_t before = state_hash(c.st);
  c.budget.used   = 0;
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 0);
  GPRS_CHECK_EQ(c.budget.used, 0);
  GPRS_CHECK_EQ(state_hash(c.st), before);
}
