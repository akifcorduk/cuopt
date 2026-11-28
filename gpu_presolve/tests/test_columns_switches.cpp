/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The choices of the singleton-column rules (options_t, from dual_integer_pivots on): every
// value of every option on a model where the values differ, and the counters of the family
// (column_count::).

#include "../check/crush_check.hpp"
#include "oracle.hpp"
#include "test_columns_support.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace gprs;
using columns_test::binary;
using columns_test::call_t;
using columns_test::free_col;
using columns_test::silent;
using columns_test::unit;
using gprs_check::structure_report_t;
using rule_test::col_t;
using rule_test::row_t;

namespace {

// Columns k, y, z: row 0 is a * k + 2 y + 3 z, rows 1 and 2 keep y and z in the model.
std::vector<row_t> rows_with(double a, double lower, double upper)
{
  return {{{{0, a}, {1, 2.}, {2, 3.}}, lower, upper},
          {{{1, 1.}, {2, 1.}}, -inf, 2.},
          {{{1, 1.}, {2, -1.}}, -1., inf}};
}

int64_t counted(const call_t& c, int32_t counter) { return c.result.columns.counts[counter]; }

const col_from_rows_record_t& record(const call_t& c) { return c.ps.col_from_rows[0]; }

}  // namespace

// k + 2 y + 3 z >= 4 with an integer k that its cost pushes down onto the row: the dual pivot
// takes it, and leaves it with dual_integer_pivots off; a continuous column leaves either way.
GPRS_TEST(switches_dual_integer_pivots)
{
  for (bool integer : {true, false}) {
    for (bool on : {true, false}) {
      call_t c({{-inf, inf, integer}, binary, binary}, rows_with(1., 4., inf), {1., 0., 0.});
      c.opt.dual_integer_pivots = on;
      c.run();
      bool taken = on || !integer;
      GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton_dual], taken ? 1 : 0);
      GPRS_CHECK_EQ(c.result.dual_findings, taken ? 1 : 0);
      GPRS_CHECK_EQ(counted(c, column_count::integer_pivots_dual), taken && integer ? 1 : 0);
      GPRS_CHECK_EQ(counted(c, column_count::refused_integer), taken ? 0 : 1);
      GPRS_CHECK_EQ(counted(c, column_count::pivots_tried), 1);
      GPRS_CHECK(silent(c.structure()));
    }
  }
  // An equation is no dual pivot: the option does not reach it.
  call_t e({{-inf, inf, true}, binary, binary}, rows_with(1., 4., 4.), {1., 0., 0.});
  e.opt.dual_integer_pivots = false;
  e.run();
  GPRS_CHECK_EQ(e.result.sub_findings[sub_rule::free_singleton], 1);
  GPRS_CHECK_EQ(counted(e, column_count::integer_pivots), 1);
}

GPRS_TEST(switches_primal_projections_which_reductions)
{
  options_t o;
  o.primal_only = true;
  for (bool on : {false, true}) {
    o.primal_projections = on;
    GPRS_CHECK(sub_rule_active(o, sub_rule::free_singleton));
    GPRS_CHECK(sub_rule_active(o, sub_rule::parallel_merge));
    GPRS_CHECK_EQ(sub_rule_active(o, sub_rule::zero_cost_singleton), on);
    GPRS_CHECK_EQ(sub_rule_active(o, sub_rule::forcing_column), on);
    for (int32_t sub : {sub_rule::free_singleton_dual,
                        sub_rule::parallel_dominance,
                        sub_rule::dual_bound,
                        sub_rule::lock_alias}) {
      GPRS_CHECK(!sub_rule_active(o, sub));
    }
    o.sub_enabled[sub_rule::forcing_column] = false;
    GPRS_CHECK(!sub_rule_active(o, sub_rule::forcing_column));
    o.sub_enabled[sub_rule::forcing_column] = true;
  }
  o.primal_only = false;
  GPRS_CHECK(sub_rule_active(o, sub_rule::zero_cost_singleton));
}

// Primal-only mode: s + 2 y + 3 z in [2, 4] with a zero-cost s in [0, 1] keeps s unless the
// projections are switched on; the record is then one that the structure check takes in that
// mode only with the option.
GPRS_TEST(switches_primal_projections_zero_cost)
{
  for (bool on : {false, true}) {
    call_t c({unit, unit, unit}, rows_with(1., 2., 4.), {0., 1., 1.});
    c.opt.primal_only        = true;
    c.opt.primal_projections = on;
    c.run();
    GPRS_CHECK_EQ(c.result.findings, on ? 1 : 0);
    GPRS_CHECK_EQ(c.result.dual_findings, 0);
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], on ? 1 : 0);
    GPRS_CHECK(silent(c.structure()));
    if (!on) { continue; }
    GPRS_CHECK(record(c).kind == col_from_rows_kind_t::interval);
    GPRS_CHECK_EQ(c.reduced.row_lower[0], 1.);
    GPRS_CHECK_EQ(c.reduced.row_upper[0], 4.);
    // The same stack without the option is no stack of the mode.
    c.opt.primal_projections = false;
    GPRS_CHECK_EQ(c.structure().stack_error,
                  "interval or forcing record in a primal-only result at 0");
  }
}

// Primal-only mode: k without a cost and without a lower bound, held from above by its two
// rows, leaves with them when the projections are on. A column with a cost on an inequality
// stays in both settings: the dual pivot is no projection.
GPRS_TEST(switches_primal_projections_forcing_column)
{
  for (bool on : {false, true}) {
    call_t c({{-inf, 3., false}, {0., 4., false}, {0., 4., false}},
             {{{{0, 1.}, {1, 1.}}, -inf, 4.},
              {{{0, -2.}, {2, 1.}}, -6., inf},
              {{{1, 1.}, {2, 1.}}, -inf, 7.},
              {{{1, 1.}, {2, -1.}}, -3., inf}});
    c.opt.primal_only        = true;
    c.opt.primal_projections = on;
    c.run();
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::forcing_column], on ? 1 : 0);
    GPRS_CHECK_EQ(c.result.dual_findings, 0);
    GPRS_CHECK_EQ(c.reduced.nrows, on ? 2 : 4);
    GPRS_CHECK(silent(c.structure()));

    call_t d({{-inf, inf, false}, unit, unit}, rows_with(1., 4., inf), {1., 0., 0.});
    d.opt.primal_only        = true;
    d.opt.primal_projections = on;
    d.run();
    GPRS_CHECK_EQ(d.result.findings, 0);
  }
}

// The whole presolve in primal-only mode with the projections: no dual finding, a feasible
// point crushes to a feasible point of the same objective value, and the round trip holds with
// the columns of the interval record exempt.
GPRS_TEST(switches_primal_projections_presolve)
{
  model_t m = rule_test::make_model({unit, unit, unit, {0., 3., false}},
                                    {{{{0, 1.}, {1, 2.}, {2, 3.}}, 2., 4.},
                                     {{{1, 1.}, {2, 1.}, {3, 1.}}, 1., 3.},
                                     {{{1, 1.}, {2, -1.}, {3, 2.}}, -1., 5.}});
  m.cost    = {0., 1., -1., 2.};
  options_t o;
  o.primal_only = true;
  result_t off  = presolve(m, o);
  GPRS_CHECK_EQ(off.sub_findings[sub_rule::zero_cost_singleton], 0);
  o.primal_projections = true;
  result_t on          = presolve(m, o);
  GPRS_CHECK_EQ(on.sub_findings[sub_rule::zero_cost_singleton], 1);
  GPRS_CHECK_EQ(on.dual_findings, 0);
  GPRS_CHECK_EQ(on.reduced.ncols, off.reduced.ncols - 1);
  GPRS_CHECK(silent(gprs_check::structure_check(m, on, o)));
  GPRS_CHECK(!gprs_check::structure_check(m, on, true).stack_error.empty());
  gprs_check::point_set_t set;
  set.add(m, {0.75, 0.5, 0.5, 0.25});
  GPRS_REQUIRE(set.points.size() == 1);
  gprs_check::crush_report_t c = gprs_check::crush_check(m, on, true, set);
  GPRS_CHECK(c.exact);
  GPRS_CHECK_EQ(c.exempt_columns, 1);
  GPRS_CHECK_EQ(c.failed_reduced + c.failed_postsolved + c.failed_objective, 0);
  GPRS_CHECK_EQ(c.failed_round_trip, 0);
}

// k + 2000 y + z = 2001 with k in [0, 2001], y binary and z in [0, 1]: the ratio over all
// columns is 2000 and that over the continuous ones 1. k has a cost, so that the zero-cost rule
// does not take what the limit leaves.
GPRS_TEST(switches_growth_ratio_of_a_continuous_column)
{
  std::vector<row_t> rows = {{{{0, 1.}, {1, 2000.}, {2, 1.}}, 2001., 2001.},
                             {{{1, 1.}, {2, 1.}}, -inf, 2.},
                             {{{1, 1.}, {2, -1.}}, -1., inf}};
  for (growth_ratio_t form : {growth_ratio_t::all_columns,
                              growth_ratio_t::continuous_columns,
                              growth_ratio_t::integer_pivot_exempt}) {
    call_t c({{0., 2001., false}, binary, unit}, rows, {1., 0., 0.});
    c.opt.growth_ratio = form;
    c.run();
    bool taken = form == growth_ratio_t::continuous_columns;
    GPRS_CHECK_EQ(c.result.findings, taken ? 1 : 0);
    GPRS_CHECK_EQ(c.result.columns.growth_refused_row, taken ? 0 : 1);
    GPRS_CHECK_EQ(counted(c, column_count::refused_growth_row), taken ? 0 : 1);
    GPRS_CHECK_EQ(c.result.columns.max_growth, taken ? 1. : 0.);
    GPRS_CHECK(!c.result.limited);
    GPRS_CHECK(silent(c.structure()));
    if (!taken) { continue; }
    // The factor that the columns of the saved row get is that of the whole row.
    std::vector<double> x = c.postsolved({{1, 1.}, {2, 0.5}});
    GPRS_CHECK_EQ(x[0], 0.5);
  }
  // With the limit above the ratio every form takes the column.
  call_t wide({{0., 2001., false}, binary, unit}, rows, {1., 0., 0.});
  wide.opt.pivot_growth_limit = 2000.;
  wide.run();
  GPRS_CHECK_EQ(wide.result.findings, 1);
  GPRS_CHECK_EQ(wide.result.columns.max_growth, 2000.);
}

// k + 2000 y + 3 z = 2003 with an integer k in [0, 2003] and binaries y and z: only integer
// columns stand next to k, so the ratio over the continuous ones is 0 and the integer pivot is
// exempt.
GPRS_TEST(switches_growth_ratio_of_an_integer_column)
{
  std::vector<row_t> rows = {{{{0, 1.}, {1, 2000.}, {2, 3.}}, 2003., 2003.},
                             {{{1, 1.}, {2, 1.}}, -inf, 2.},
                             {{{1, 1.}, {2, -1.}}, -1., inf}};
  for (growth_ratio_t form : {growth_ratio_t::all_columns,
                              growth_ratio_t::continuous_columns,
                              growth_ratio_t::integer_pivot_exempt}) {
    call_t c({{0., 2003., true}, binary, binary}, rows, {1., 0., 0.});
    c.opt.growth_ratio = form;
    c.run();
    bool taken = form != growth_ratio_t::all_columns;
    GPRS_CHECK_EQ(c.result.findings, taken ? 1 : 0);
    GPRS_CHECK_EQ(counted(c, column_count::integer_pivots), taken ? 1 : 0);
    GPRS_CHECK_EQ(c.result.columns.max_growth, 0.);
    GPRS_CHECK(silent(c.structure()));
  }
}

// s + 2 y + 3 z <= 4 with a zero-cost s in [1, 5]: a row with one finite side is not the
// rule's, unless zero_cost_one_sided is on. The row then keeps its infinite side and loses
// the smallest term of s from the other.
GPRS_TEST(switches_zero_cost_one_sided)
{
  for (bool on : {false, true}) {
    call_t c({{1., 5., false}, unit, unit}, rows_with(1., -inf, 4.));
    c.opt.zero_cost_one_sided = on;
    c.run();
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], on ? 1 : 0);
    GPRS_CHECK_EQ(counted(c, column_count::zero_cost_one_sided), on ? 0 : 1);
    GPRS_CHECK_EQ(counted(c, column_count::refused_zero_cost), 1);
    GPRS_CHECK(silent(c.structure()));
    if (!on) { continue; }
    GPRS_CHECK_EQ(c.result.dual_findings, 0);
    GPRS_CHECK_EQ(c.reduced.row_lower[0], -inf);
    GPRS_CHECK_EQ(c.reduced.row_upper[0], 3.);
    GPRS_CHECK(record(c).kind == col_from_rows_kind_t::interval);
    GPRS_CHECK_EQ(c.ps.saved_rows[0].lower, -inf);
    GPRS_CHECK_EQ(c.ps.saved_rows[0].upper, 4.);
    GPRS_CHECK_EQ(c.postsolved({{1, 1.}, {2, 1. / 3.}})[0], 1.);
    GPRS_CHECK_EQ(c.postsolved({{1, 0.}, {2, 0.}})[0], 1.);
    // Without the option the stack is none that the check takes.
    c.opt.zero_cost_one_sided = false;
    GPRS_CHECK_EQ(c.structure().stack_error, "sides of interval record 0");
  }
}

// -s + 2 y + 3 z <= 4: with s in [0, 5] the row is left with the side 9, and the undo moves s
// up from its lower bound as far as the row asks; with s in [0, inf) nothing is left to ask of
// y and z and the row leaves.
GPRS_TEST(switches_zero_cost_one_sided_undo)
{
  call_t c({{0., 5., false}, unit, unit}, rows_with(-1., -inf, 4.));
  c.opt.zero_cost_one_sided = true;
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 1);
  GPRS_CHECK_EQ(c.reduced.row_upper[0], 9.);
  GPRS_CHECK_EQ(c.postsolved({{1, 1.}, {2, 1.}})[0], 1.);
  GPRS_CHECK_EQ(c.postsolved({{1, 1.}, {2, 0.5}})[0], 0.);
  GPRS_CHECK(silent(c.structure()));

  call_t open({{0., inf, false}, unit, unit}, rows_with(-1., -inf, 4.));
  open.opt.zero_cost_one_sided = true;
  open.run();
  GPRS_CHECK_EQ(open.result.findings, 1);
  GPRS_CHECK_EQ(open.reduced.nrows, 2);
  GPRS_CHECK_EQ(open.postsolved({{1, 1.}, {2, 1.}})[0], 1.);
  GPRS_CHECK(silent(open.structure()));

  // The lower side alone, with the mirror of the first case.
  call_t low({{0., 5., false}, unit, unit}, rows_with(1., 2., inf));
  low.opt.zero_cost_one_sided = true;
  low.run();
  GPRS_CHECK_EQ(low.result.findings, 1);
  GPRS_CHECK_EQ(low.reduced.row_lower[0], -3.);
  GPRS_CHECK_EQ(low.reduced.row_upper[0], inf);
  GPRS_CHECK_EQ(low.postsolved({{1, 0.5}, {2, 0.}})[0], 1.);
  GPRS_CHECK(silent(low.structure()));
}

// s + t in [1, 3] with s and t in [0, 2] and nothing else in the row: the columns stay, unless
// zero_cost_whole_row is on. The row is then left without a free entry and with the sides
// that 0 has to meet.
GPRS_TEST(switches_zero_cost_whole_row)
{
  std::vector<col_t> cols = {{0., 2., false}, {0., 2., false}, unit, unit};
  std::vector<row_t> rows = {
    {{{0, 1.}, {1, 1.}}, 1., 3.}, {{{2, 1.}, {3, 1.}}, -inf, 2.}, {{{2, 1.}, {3, -1.}}, -1., inf}};
  for (bool on : {false, true}) {
    call_t c(cols, rows);
    c.opt.zero_cost_whole_row = on;
    c.run();
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], on ? 2 : 0);
    GPRS_CHECK_EQ(counted(c, column_count::zero_cost_whole_row), on ? 0 : 1);
    GPRS_CHECK_EQ(counted(c, column_count::zero_cost_whole_row_columns), on ? 0 : 2);
    GPRS_CHECK(silent(c.structure()));
    if (!on) { continue; }
    GPRS_CHECK(c.st.row_alive[0]);
    GPRS_CHECK_EQ(c.st.row_free_size[0], 0);
    GPRS_CHECK_EQ(c.st.row_lower[0], -3.);
    GPRS_CHECK_EQ(c.st.row_upper[0], 3.);
    std::vector<double> x = c.postsolved({{2, 0.5}, {3, 0.5}});
    GPRS_CHECK_EQ(x[0], 1.);
    GPRS_CHECK_EQ(x[1], 0.);
  }
  // The whole presolve: the row rules delete the row that is left, and the result is a fixed
  // point.
  model_t m = rule_test::make_model(cols, rows);
  m.cost    = {0., 0., 1., -1.};
  options_t o;
  result_t kept         = presolve(m, o);
  o.zero_cost_whole_row = true;
  result_t gone         = presolve(m, o);
  GPRS_CHECK_EQ(gone.sub_findings[sub_rule::zero_cost_singleton], 2);
  GPRS_CHECK_EQ(gone.reduced.nrows + 1, kept.reduced.nrows);
  GPRS_CHECK_EQ(gone.reduced.ncols + 2, kept.reduced.ncols);
  GPRS_CHECK(gone.fixed_point);
  GPRS_CHECK(silent(gprs_check::structure_check(m, gone, o)));
}

// k0 + 4 k1 + 2 y + 3 z = 6 with two free singleton columns: the smallest index takes k0, the
// largest pivot k1.
GPRS_TEST(switches_order_largest_pivot)
{
  std::vector<row_t> rows = {{{{0, 1.}, {1, 4.}, {2, 2.}, {3, 3.}}, 6., 6.},
                             {{{2, 1.}, {3, 1.}}, -inf, 2.},
                             {{{2, 1.}, {3, -1.}}, -1., inf}};
  for (singleton_order_t order :
       {singleton_order_t::smallest_index, singleton_order_t::largest_pivot}) {
    call_t c({free_col, free_col, unit, unit}, rows);
    c.opt.singleton_order = order;
    c.run();
    bool largest = order == singleton_order_t::largest_pivot;
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton], 1);
    GPRS_CHECK(c.st.col_status[largest ? 1 : 0] == col_status_t::removed);
    GPRS_CHECK(c.st.col_status[largest ? 0 : 1] == col_status_t::live);
    GPRS_CHECK_EQ(c.ps.record_coefficients[record(c).coefficients], largest ? 4. : 1.);
    GPRS_CHECK_EQ(counted(c, column_count::pivots_among_several), 1);
    GPRS_CHECK_EQ(counted(c, column_count::pivots_tried), 1);
    GPRS_CHECK(silent(c.structure()));
  }
  // Equal coefficients: the index decides.
  rows[0].entries[1].second = -1.;
  call_t tie({free_col, free_col, unit, unit}, rows);
  tie.opt.singleton_order = singleton_order_t::largest_pivot;
  tie.run();
  GPRS_CHECK(tie.st.col_status[0] == col_status_t::removed);
}

// k + 2 y + 3 z = 6 with a zero-cost k in [1, 6], which the row implies: the free singleton
// column takes the row with it; with zero_cost_first the column leaves as a projection and the
// row stays with the sides [0, 5].
GPRS_TEST(switches_order_zero_cost_first)
{
  for (singleton_order_t order :
       {singleton_order_t::smallest_index, singleton_order_t::zero_cost_first}) {
    call_t c({{1., 6., false}, unit, unit}, rows_with(1., 6., 6.));
    c.opt.singleton_order = order;
    c.run();
    bool first = order == singleton_order_t::zero_cost_first;
    GPRS_CHECK_EQ(c.result.findings, 1);
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], first ? 1 : 0);
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton], first ? 0 : 1);
    GPRS_CHECK_EQ(c.reduced.nrows, first ? 3 : 2);
    GPRS_CHECK_EQ(counted(c, column_count::pivots_tried), first ? 0 : 1);
    GPRS_CHECK(silent(c.structure()));
    if (first) {
      GPRS_CHECK_EQ(c.reduced.row_lower[0], 0.);
      GPRS_CHECK_EQ(c.reduced.row_upper[0], 5.);
    }
  }
  // A column with a cost is not the zero-cost rule's: the visit goes on to the pivot.
  call_t cost({{1., 6., false}, unit, unit}, rows_with(1., 6., 6.), {1., 0., 0.});
  cost.opt.singleton_order = singleton_order_t::zero_cost_first;
  cost.run();
  GPRS_CHECK_EQ(cost.result.sub_findings[sub_rule::free_singleton], 1);
}

// A chain: k leaves with row 0, which makes y a singleton column of row 1. Row 1 was visited
// in the same pass for its singleton column s, so y waits for the next pass: the next pass of
// the call, or with single_pass the call of the next round.
GPRS_TEST(switches_order_single_pass)
{
  constexpr int32_t s = 0, y = 1, k = 2, a = 3, b = 4;
  std::vector<col_t> cols  = {unit, free_col, free_col, unit, unit};
  std::vector<row_t> rows  = {{{{y, 1.}, {k, 1.}}, 3., 3.},
                              {{{s, 1.}, {y, 1.}, {a, 2.}, {b, 3.}}, 6., 6.},
                              {{{a, 1.}, {b, 1.}}, -inf, 2.},
                              {{{a, 1.}, {b, -1.}}, -1., inf}};
  std::vector<double> cost = {1., 0., 0., 0., 0.};

  call_t all(cols, rows, cost);
  all.run();
  GPRS_CHECK_EQ(all.result.sub_findings[sub_rule::free_singleton], 2);
  // The third pass serves what the second substitution listed.
  GPRS_CHECK_EQ(counted(all, column_count::passes), 3);

  call_t one(cols, rows, cost);
  one.opt.singleton_order = singleton_order_t::single_pass;
  one.result              = one.family.run(one.st, one.budget, one.ps, one.opt);
  GPRS_CHECK_EQ(one.result.sub_findings[sub_rule::free_singleton], 1);
  GPRS_CHECK_EQ(counted(one, column_count::passes), 1);
  GPRS_CHECK(!one.result.stopped && !one.result.limited);
  GPRS_CHECK(one.st.col_status[k] == col_status_t::removed);
  GPRS_CHECK(one.st.col_status[y] == col_status_t::live);
  GPRS_CHECK(!one.st.col_frontier.empty(col_list::singleton_cols) ||
             !one.st.row_frontier.empty(row_list::singleton_cols));
  // The next call makes the next link.
  one.result = one.family.run(one.st, one.budget, one.ps, one.opt);
  GPRS_CHECK_EQ(one.result.sub_findings[sub_rule::free_singleton], 1);
  GPRS_CHECK(one.st.col_status[y] == col_status_t::removed);

  // The whole presolve ends at the same model, one round later at most.
  model_t m = columns_test::call_t::with_cost(rule_test::make_model(cols, rows), cost);
  options_t o;
  result_t chained  = presolve(m, o);
  o.singleton_order = singleton_order_t::single_pass;
  result_t stepped  = presolve(m, o);
  GPRS_CHECK_EQ(stepped.hash, chained.hash);
  GPRS_CHECK(stepped.fixed_point);
  GPRS_CHECK(stepped.stats[rule::singleton_cols].calls >=
             chained.stats[rule::singleton_cols].calls);
}

// single_pass and a pass without a reduction. k + y <= 3e9 with k free to go down is left for
// its side and marks the row; when the side becomes 4 the row is listed and lists its columns.
// That pass reduces nothing, so the call goes on and takes k: it must not end with a list and
// without a finding, which the engine would read as a fixed point.
GPRS_TEST(switches_order_single_pass_without_a_reduction)
{
  const col_t y{0., 4., false};
  for (singleton_order_t order :
       {singleton_order_t::smallest_index, singleton_order_t::single_pass}) {
    call_t c({{-inf, 9., false}, y, y},
             {{{{0, 1.}, {1, 1.}}, -inf, 3e9},
              {{{1, 1.}, {2, 1.}}, -inf, 7.},
              {{{1, 1.}, {2, -1.}}, -3., inf}});
    c.opt.singleton_order = order;
    c.result              = c.family.run(c.st, c.budget, c.ps, c.opt);
    GPRS_CHECK_EQ(c.result.findings, 0);
    GPRS_CHECK_EQ(counted(c, column_count::forcing_side), 1);
    c.st.set_row_upper(0, 4.);
    c.result = c.family.run(c.st, c.budget, c.ps, c.opt);
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::forcing_column], 1);
    GPRS_CHECK(c.st.col_status[0] == col_status_t::removed);
    GPRS_CHECK(counted(c, column_count::passes) >= 2);
    GPRS_CHECK(!c.result.stopped);
  }
}

// s + t + 2 y + 3 z in [4, 9] with s in [1, 3] and t in [-2, 4]: the start values of the four
// settings, and the two orders of the move.
GPRS_TEST(switches_interval_start_and_order)
{
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}, {2, 2.}, {3, 3.}}, 4., 9.},
                             {{{2, 1.}, {3, 1.}}, -inf, 2.},
                             {{{2, 1.}, {3, -1.}}, -1., inf}};
  std::vector<col_t> cols = {{1., 3., false}, {-2., 4., false}, unit, unit};
  struct expected_t {
    interval_start_t start;
    double s;
    double t;
    // The values at y = z = 0, where the row asks for 4, moved in ascending order.
    double moved_s;
    double moved_t;
  };
  for (const expected_t& e : {expected_t{interval_start_t::lower, 1., -2., 3., 1.},
                              expected_t{interval_start_t::upper, 3., 4., 3., 4.},
                              expected_t{interval_start_t::nearest_zero, 1., 0., 3., 1.},
                              expected_t{interval_start_t::middle, 2., 1., 3., 1.}}) {
    call_t c(cols, rows);
    c.opt.interval_start = e.start;
    c.run();
    GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], 2);
    GPRS_CHECK_EQ(c.ps.record_cols[0].prefer, e.s);
    GPRS_CHECK_EQ(c.ps.record_cols[1].prefer, e.t);
    // The new sides do not depend on the start value.
    GPRS_CHECK_EQ(c.reduced.row_lower[0], -3.);
    GPRS_CHECK_EQ(c.reduced.row_upper[0], 10.);
    std::vector<double> x = c.postsolved({{2, 0.}, {3, 0.}});
    GPRS_CHECK_EQ(x[0], e.moved_s);
    GPRS_CHECK_EQ(x[1], e.moved_t);
    GPRS_CHECK(silent(c.structure()));
    // Descending: t is moved first and can give all that the row misses.
    c.ps.interval_descending = true;
    x                        = c.postsolved({{2, 0.}, {3, 0.}});
    double sum               = x[0] + x[1];
    GPRS_CHECK(sum >= 4. && sum <= 9.);
    GPRS_CHECK_EQ(x[0], e.s);
    GPRS_CHECK(silent(c.structure()));
  }
  // Columns without a finite bound: every start is a point of the bounds.
  for (interval_start_t start : {interval_start_t::lower,
                                 interval_start_t::upper,
                                 interval_start_t::nearest_zero,
                                 interval_start_t::middle}) {
    call_t c({{-inf, -1., false}, {2., inf, false}, unit, unit}, rows);
    c.opt.interval_start = start;
    c.run();
    GPRS_REQUIRE(c.ps.record_cols.size() == 2);
    GPRS_CHECK_EQ(c.ps.record_cols[0].prefer, -1.);
    GPRS_CHECK_EQ(c.ps.record_cols[1].prefer, 2.);
    GPRS_CHECK(silent(c.structure()));
  }
  call_t open({free_col, unit, unit}, rows_with(1., 2., 4.));
  open.opt.interval_start = interval_start_t::middle;
  open.run();
  GPRS_CHECK_EQ(open.ps.record_cols[0].prefer, 0.);
}

// s + 2 t + 2 y + 3 z in [4, 9] with s in [0, 1] and t in [0, 3], so that each column moves
// with a coefficient of its own. At y = z = 0 the row misses 4: in ascending order s gives 1
// and t the rest, in descending order t gives all of it. At y = z = 1, from the upper bounds,
// the row is 3 above its side: s gives 1 and t the rest, or t all of it.
GPRS_TEST(switches_interval_order_with_two_coefficients)
{
  std::vector<row_t> rows = {{{{0, 1.}, {1, 2.}, {2, 2.}, {3, 3.}}, 4., 9.},
                             {{{2, 1.}, {3, 1.}}, -inf, 2.},
                             {{{2, 1.}, {3, -1.}}, -1., inf}};
  std::vector<col_t> cols = {{0., 1., false}, {0., 3., false}, unit, unit};
  call_t low(cols, rows);
  low.run();
  GPRS_CHECK_EQ(low.result.sub_findings[sub_rule::zero_cost_singleton], 2);
  std::vector<double> x = low.postsolved({{2, 0.}, {3, 0.}});
  GPRS_CHECK_EQ(x[0], 1.);
  GPRS_CHECK_EQ(x[1], 1.5);
  low.ps.interval_descending = true;
  x                          = low.postsolved({{2, 0.}, {3, 0.}});
  GPRS_CHECK_EQ(x[0], 0.);
  GPRS_CHECK_EQ(x[1], 2.);
  GPRS_CHECK(silent(low.structure()));

  call_t high(cols, rows);
  high.opt.interval_start = interval_start_t::upper;
  high.run();
  x = high.postsolved({{2, 1.}, {3, 1.}});
  GPRS_CHECK_EQ(x[0], 0.);
  GPRS_CHECK_EQ(x[1], 2.);
  high.ps.interval_descending = true;
  x                           = high.postsolved({{2, 1.}, {3, 1.}});
  GPRS_CHECK_EQ(x[0], 1.);
  GPRS_CHECK_EQ(x[1], 1.5);
  GPRS_CHECK(silent(high.structure()));
}

// The presolve writes the two choices of the undo to the stack.
GPRS_TEST(switches_undo_choices_reach_the_stack)
{
  model_t m = rule_test::make_model({unit, unit, unit}, rows_with(1., 2., 4.));
  options_t o;
  result_t r = presolve(m, o);
  GPRS_CHECK(!r.postsolve.interval_descending);
  GPRS_CHECK_EQ(r.postsolve.forcing_rounding_units, 8.);
  uint64_t hash            = postsolve_hash(r.postsolve);
  o.interval_order         = interval_order_t::descending;
  o.forcing_rounding_units = 0.;
  r                        = presolve(m, o);
  GPRS_CHECK(r.postsolve.interval_descending);
  GPRS_CHECK_EQ(r.postsolve.forcing_rounding_units, 0.);
  // They are no part of the hash of the stack.
  GPRS_CHECK_EQ(postsolve_hash(r.postsolve), hash);
  r.postsolve.init(m);
  GPRS_CHECK(!r.postsolve.interval_descending);
  GPRS_CHECK_EQ(r.postsolve.forcing_rounding_units, 8.);
}

// k + (7/3) y <= 9 with an integer k that is free to go down: at y = 3 the row leaves k the
// value 1.9999999999999996, which a band takes as 2 and no band rounds down to 1. Both are
// values of the row, and the structure check follows either.
GPRS_TEST(switches_forcing_rounding_units)
{
  std::vector<row_t> rows = {{{{0, 1.}, {1, 7. / 3.}}, -inf, 9.},
                             {{{1, 1.}, {2, 1.}}, -inf, 5.},
                             {{{1, 1.}, {2, -1.}}, -1., inf}};
  call_t c({{-inf, 5., true}, {0., 3., false}, {0., 3., false}}, rows);
  c.run();
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::forcing_column], 1);
  GPRS_CHECK_EQ(c.ps.forcing_rounding_units, 8.);
  GPRS_CHECK_EQ(c.postsolved({{1, 3.}, {2, 2.}})[0], 2.);
  GPRS_CHECK(silent(c.structure()));
  c.ps.forcing_rounding_units = 0.;
  GPRS_CHECK_EQ(c.postsolved({{1, 3.}, {2, 2.}})[0], 1.);
  GPRS_CHECK(silent(c.structure()));
  c.ps.forcing_rounding_units = 4.;
  GPRS_CHECK_EQ(c.postsolved({{1, 3.}, {2, 2.}})[0], 2.);
  GPRS_CHECK(silent(c.structure()));
  // A value that is no integer within any band is rounded into the row with each setting.
  for (double units : {0., 4., 8.}) {
    c.ps.forcing_rounding_units = units;
    GPRS_CHECK_EQ(c.postsolved({{1, 1.5}, {2, 2.}})[0], 5.);
    GPRS_CHECK_EQ(c.postsolved({{1, 2.5}, {2, 2.}})[0], 3.);
  }
}

// A side of 5e8 is beyond what the family reads, and within eight times that; a side of 5e6
// is read unless the factor is 0.01.
GPRS_TEST(switches_side_guard_scale)
{
  for (double scale : {1., 8.}) {
    call_t c({free_col, unit, unit}, rows_with(1., 5e8, 5e8), {1., 0., 0.});
    c.opt.side_guard_scale = scale;
    c.run();
    GPRS_CHECK_EQ(c.result.findings, scale > 1. ? 1 : 0);
    GPRS_CHECK_EQ(counted(c, column_count::refused_side), scale > 1. ? 0 : 1);
    GPRS_CHECK(silent(c.structure()));
    if (scale > 1.) { GPRS_CHECK_EQ(c.postsolved({{1, 1.}, {2, 1.}})[0], 5e8 - 5.); }
  }
  for (double scale : {0.01, 1.}) {
    call_t c({free_col, unit, unit}, rows_with(1., 5e6, 5e6), {1., 0., 0.});
    c.opt.side_guard_scale = scale;
    c.run();
    GPRS_CHECK_EQ(c.result.findings, scale < 1. ? 0 : 1);
    // The zero-cost rule and the forcing column read their sides through the same guard.
    call_t z({unit, unit, unit}, rows_with(1., 2., 5e6));
    z.opt.side_guard_scale = scale;
    z.run();
    GPRS_CHECK_EQ(z.result.findings, scale < 1. ? 0 : 1);
    GPRS_CHECK_EQ(counted(z, column_count::zero_cost_side), scale < 1. ? 1 : 0);
    call_t f({{-inf, 3., false}, {0., 4., false}, {0., 4., false}},
             {{{{0, 1.}, {1, 1.}}, -inf, 5e6},
              {{{1, 1.}, {2, 1.}}, -inf, 7.},
              {{{1, 1.}, {2, -1.}}, -3., inf}});
    f.opt.side_guard_scale = scale;
    f.run();
    GPRS_CHECK_EQ(f.result.sub_findings[sub_rule::forcing_column], scale < 1. ? 0 : 1);
    GPRS_CHECK_EQ(counted(f, column_count::forcing_side), scale < 1. ? 1 : 0);
  }
}

GPRS_TEST(switches_ranges_of_the_choices)
{
  options_t o;
  GPRS_CHECK(choices_in_range(o));
  for (double units : {0., 4., 8.}) {
    o.forcing_rounding_units = units;
    GPRS_CHECK(choices_in_range(o));
  }
  for (double units : {-1., 8.5, 64., inf, std::nan("")}) {
    o.forcing_rounding_units = units;
    GPRS_CHECK(!choices_in_range(o));
  }
  o = options_t();
  for (double scale : {0.002, 0.5, 2., 11.}) {
    o.side_guard_scale = scale;
    GPRS_CHECK(choices_in_range(o));
  }
  for (double scale : {0., -1., 1e-4, 12., inf, std::nan("")}) {
    o.side_guard_scale = scale;
    GPRS_CHECK(!choices_in_range(o));
  }
}

// x + y = 5 with a singleton column x: after simple probing the equation has been the
// doubleton rule's for a round; as the first phase after the cascade the family takes x.
GPRS_TEST(switches_singleton_cols_early)
{
  model_t m = rule_test::make_model(
    {free_col, {0., 10., false}, {0., 10., false}},
    {{{{0, 1.}, {1, 1.}}, 5., 5.}, {{{1, 1.}, {2, 1.}}, 2., 12.}, {{{1, 1.}, {2, -3.}}, -4., 6.}});
  m.cost = {2., 1., 1.};
  options_t o;
  o.phase_hashes = true;
  result_t late  = presolve(m, o);
  GPRS_CHECK(late.postsolve.aliases.size() == 1);
  GPRS_CHECK_EQ(late.sub_findings[sub_rule::free_singleton], 0);
  o.singleton_cols_early = true;
  result_t early         = presolve(m, o);
  GPRS_CHECK(early.postsolve.aliases.empty());
  GPRS_CHECK_EQ(early.sub_findings[sub_rule::free_singleton], 1);
  GPRS_CHECK(early.fixed_point && late.fixed_point);
  GPRS_CHECK_EQ(early.reduced.nrows, late.reduced.nrows);
  GPRS_CHECK_EQ(early.reduced.ncols, late.reduced.ncols);
  // One start entry, seven per round, one at the end, whatever the order is.
  GPRS_CHECK_EQ(early.phase_hash.size(), 2 + 7 * (size_t)early.rounds);
  GPRS_CHECK(silent(gprs_check::structure_check(m, early, o)));
  std::vector<double> reduced(early.reduced.ncols, 1.), x;
  early.postsolve.postsolve(reduced, x);
  GPRS_CHECK_EQ(x[0] + x[1], 5.);
}

// The refusals of the free singleton column on an inequality by the end that fails: the bound
// ahead of the move, or only the one behind it.
GPRS_TEST(switches_counters_of_the_two_ends)
{
  // The cost moves k up to the upper side 6: its range is [1, 6].
  call_t behind({{3., inf, false}, unit, unit}, rows_with(1., -inf, 6.), {-1., 0., 0.});
  behind.run();
  GPRS_CHECK_EQ(behind.result.findings, 0);
  GPRS_CHECK_EQ(counted(behind, column_count::refused_bound), 1);
  GPRS_CHECK_EQ(counted(behind, column_count::refused_behind), 1);
  GPRS_CHECK_EQ(counted(behind, column_count::refused_ahead), 0);

  call_t ahead({{-inf, 4., false}, unit, unit}, rows_with(1., -inf, 6.), {-1., 0., 0.});
  ahead.run();
  GPRS_CHECK_EQ(counted(ahead, column_count::refused_ahead), 1);
  GPRS_CHECK_EQ(counted(ahead, column_count::refused_behind), 0);

  // The cost moves k down to the lower side 1: its range is [-4, 1]. The upper bound, which is
  // tested first, is the one behind.
  call_t first({{-10., 0., false}, unit, unit}, rows_with(1., 1., inf), {1., 0., 0.});
  first.run();
  GPRS_CHECK_EQ(first.result.findings, 0);
  GPRS_CHECK_EQ(counted(first, column_count::refused_behind), 1);
  GPRS_CHECK_EQ(counted(first, column_count::refused_ahead), 0);

  call_t both({{-2., 0., false}, unit, unit}, rows_with(1., 1., inf), {1., 0., 0.});
  both.run();
  GPRS_CHECK_EQ(counted(both, column_count::refused_ahead), 1);
  GPRS_CHECK_EQ(counted(both, column_count::refused_behind), 0);

  // An equation has no move: neither counter.
  call_t equation({{3., 6., false}, unit, unit}, rows_with(1., 6., 6.), {-1., 0., 0.});
  equation.run();
  GPRS_CHECK_EQ(counted(equation, column_count::refused_bound), 1);
  GPRS_CHECK_EQ(counted(equation, column_count::refused_ahead) +
                  counted(equation, column_count::refused_behind),
                0);
}

// The first test that a column fails is the one that counts it.
GPRS_TEST(switches_counters_of_the_refusals)
{
  // An inequality and no cost: the zero-cost rule takes the column afterwards. The row is
  // visited for its listed column and again for its own change, then without a singleton
  // column.
  call_t zero({unit, unit, unit}, rows_with(1., 2., 4.));
  zero.run();
  GPRS_CHECK_EQ(counted(zero, column_count::refused_zero_cost), 1);
  GPRS_CHECK_EQ(counted(zero, column_count::zero_cost_records), 1);
  GPRS_CHECK_EQ(counted(zero, column_count::rows_visited), 2);
  GPRS_CHECK_EQ(counted(zero, column_count::rows_without_singleton), 1);
  GPRS_CHECK_EQ(counted(zero, column_count::rows_scanned), 1);
  GPRS_CHECK_EQ(counted(zero, column_count::scanned_slots), 3);

  // The side that would stop the column is infinite.
  call_t open({unit, unit, unit}, rows_with(1., 2., inf), {-1., 0., 0.});
  open.run();
  GPRS_CHECK_EQ(counted(open, column_count::refused_side_infinite), 1);

  // An end of the range with an infinite term.
  call_t wide({unit, {-inf, 1., false}, unit}, rows_with(1., 6., 6.), {1., 0., 0.});
  wide.run();
  GPRS_CHECK_EQ(counted(wide, column_count::refused_infinite_rest), 1);

  // 2 k + 2 y + 3 z = 6 with integers: 3 / 2 is no integer.
  call_t odd({{-inf, inf, true}, binary, binary}, rows_with(2., 6., 6.), {1., 0., 0.});
  odd.run();
  GPRS_CHECK_EQ(counted(odd, column_count::refused_integral), 1);

  // A cost that the division loses.
  call_t tiny({free_col, unit, unit}, rows_with(4., 6., 6.), {4.9e-324, 0., 0.});
  tiny.run();
  GPRS_CHECK_EQ(counted(tiny, column_count::refused_cost), 1);

  // Every counter of a refusal is one of the tried columns.
  for (const call_t* c : {&zero, &open, &wide, &odd, &tiny}) {
    GPRS_CHECK_EQ(counted(*c, column_count::pivots_tried), 1);
  }
}

// What the zero-cost rule passes over: integer columns, and of them those for which the row
// is integral by its ratios.
GPRS_TEST(switches_counters_of_the_zero_cost_rule)
{
  // s + 2 y + 3 z in [2, 4], all integer: 1 divides 2 and 3.
  call_t whole({binary, binary, binary}, rows_with(1., 2., 4.));
  whole.run();
  GPRS_CHECK_EQ(whole.result.findings, 0);
  GPRS_CHECK_EQ(counted(whole, column_count::zero_cost_integer), 1);
  GPRS_CHECK_EQ(counted(whole, column_count::zero_cost_integer_integral), 1);

  // 2 does not divide 3.
  call_t odd({binary, binary, binary}, rows_with(2., 2., 4.));
  odd.run();
  GPRS_CHECK_EQ(counted(odd, column_count::zero_cost_integer), 1);
  GPRS_CHECK_EQ(counted(odd, column_count::zero_cost_integer_integral), 0);

  // A continuous column in the row.
  call_t mixed({binary, binary, unit}, rows_with(1., 2., 4.));
  mixed.run();
  GPRS_CHECK_EQ(counted(mixed, column_count::zero_cost_integer), 1);
  GPRS_CHECK_EQ(counted(mixed, column_count::zero_cost_integer_integral), 0);

  // A cost that is small and not zero, on a continuous column.
  call_t small({unit, unit, unit}, rows_with(1., 2., 4.), {1e-9, 0., 0.});
  small.opt.sub_enabled[sub_rule::free_singleton_dual] = false;
  small.run();
  GPRS_CHECK_EQ(small.result.findings, 0);
  GPRS_CHECK_EQ(counted(small, column_count::zero_cost_small_cost), 1);
}

// Counting changes nothing: the work and the findings of a call do not depend on a counter.
GPRS_TEST(switches_counters_spend_no_work)
{
  call_t c({binary, binary, binary}, rows_with(1., 2., 4.));
  c.budget.limit       = 1000;
  c.budget.stall_limit = 1000;
  c.run();
  // Three listed columns, the walk to the entry of the singleton column and one scan of its
  // row: the scans that count the integral ratios are not charged.
  GPRS_CHECK_EQ(counted(c, column_count::zero_cost_integer_integral), 1);
  GPRS_CHECK_EQ(c.budget.used, 3 * line_visit_work + 1 + 3);
}

// Primal-only mode with the projections on generated models: the structure check is silent
// with the options, and every feasible point that the exact solver lists crushes to a feasible
// point of the same objective value and comes back with only the columns of interval and
// forcing records moved (clause 5 of the crush check). Half of the seeds also take the rows
// with one side and the rows made of such columns.
GPRS_TEST(switches_primal_projections_checks_on_generated_models)
{
  oracle::generator_options_t g;
  g.p_singleton_col       = 0.5;
  g.p_singleton_line      = 0.7;
  g.p_singleton_zero_cost = 0.7;
  g.max_singleton_cols    = 4;
  g.max_row_singletons    = 2;
  g.p_forcing_col         = 0.5;
  g.p_equation            = 0.35;
  g.p_ranged_row          = 0.3;
  g.p_huge_bound          = 0.;
  g.p_scaled              = 0.;
  int64_t interval = 0, forcing = 0, one_sided = 0, exempt = 0, points = 0, failed = 0;
  for (uint64_t seed = 9300000; seed < 9304000 && failed < 3; ++seed) {
    oracle::case_t c = oracle::generate(seed, g);
    options_t options;
    options.primal_only         = true;
    options.primal_projections  = true;
    options.zero_cost_one_sided = seed % 2 == 1;
    options.zero_cost_whole_row = seed % 2 == 1;
    result_t result             = presolve(c.model, options);
    if (!gprs_check::carries_model(result.status)) { continue; }
    GPRS_CHECK_EQ(result.dual_findings, 0);
    for (const col_from_rows_record_t& record : result.postsolve.col_from_rows) {
      GPRS_CHECK(!record.dual);
      interval += record.kind == col_from_rows_kind_t::interval;
      forcing += record.kind == col_from_rows_kind_t::forcing;
      if (record.kind == col_from_rows_kind_t::interval) {
        const saved_row_t& row = result.postsolve.saved_rows[record.rows_begin];
        one_sided += std::isinf(row.lower) || std::isinf(row.upper);
      }
    }
    structure_report_t r = gprs_check::structure_check(c.model, result, options);
    std::vector<uint8_t> line;
    gprs_check::point_set_t set;
    if (oracle::classify_widest(c.base, oracle::check_max_assignments, line)) {
      oracle::solution_t base = oracle::solve(c.base, line, 8);
      for (std::vector<double> x : base.feasible) {
        for (size_t j = 0; j < x.size(); ++j) {
          x[j] /= c.col_scale[j];
        }
        set.add(c.model, x);
      }
    }
    gprs_check::crush_report_t k = gprs_check::crush_check(c.model, result, true, set);
    points += k.points;
    exempt += k.exempt_columns;
    bool crush_fails =
      k.failed_reduced + k.failed_postsolved + k.failed_objective + k.failed_round_trip != 0;
    if (silent(r) && !crush_fails) { continue; }
    ++failed;
    test::fail(__FILE__,
               __LINE__,
               "seed " + std::to_string(seed) + ": " + r.stack_error +
                 (crush_fails ? " crush check" : "") + "\n" + oracle::to_code(c));
  }
  GPRS_CHECK_EQ(failed, 0);
  GPRS_CHECK(interval > 100 && forcing > 100 && one_sided > 20);
  GPRS_CHECK(exempt > 200 && points > 2000);
}
