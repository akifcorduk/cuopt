/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Simple probing on an equation (rules.hpp (g)): the tests of a visit, the aliases, the batch
// through the alias forest, and the fixed models of its design.

#include "../check/structure_check.hpp"
#include "harness.hpp"
#include "oracle.hpp"
#include "test_columns_support.hpp"

#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace gprs;
using columns_test::binary;
using columns_test::silent;
using columns_test::unit;
using gprs_check::structure_report_t;
using rule_test::col_t;
using rule_test::row_t;

namespace {

constexpr rule_outcome_t ok  = rule_outcome_t::ok;
const alias_origin_t probing = alias_origin_t::probing;

// One call of the family on the state of a model, and what is left afterwards.
struct call_t {
  model_t model;
  state_t st;
  postsolve_t ps;
  options_t opt;
  work_budget_t budget;
  simple_probing_t family;
  rule_result_t result;
  model_t reduced;

  call_t(const std::vector<col_t>& cols,
         const std::vector<row_t>& rows,
         const std::vector<double>& cost = {})
    : model(with_cost(rule_test::make_model(cols, rows), cost)), st(model, tolerances_t())
  {
    GPRS_CHECK(well_formed(model));
    ps.init(model);
    budget.limit       = rule_test::unlimited;
    budget.stall_limit = rule_test::unlimited;
  }

  static model_t with_cost(model_t m, const std::vector<double>& cost)
  {
    if (!cost.empty()) { m.cost = cost; }
    return m;
  }

  call_t& run()
  {
    result = family.run(st, budget, ps, opt);
    if (result.outcome == ok) {
      GPRS_CHECK_EQ(st.consistency_error(), std::string());
      reduced = reduced_model(st, ps.row_origin, ps.col_origin);
    }
    return *this;
  }

  bool listed(int32_t row) const
  {
    return st.row_frontier.pending[row] & list_bit(row_list::simple_probing);
  }

  // The values of the original columns that the stack gives for values of the columns that
  // are left, named by their original index.
  std::vector<double> postsolved(const std::vector<std::pair<int32_t, double>>& left) const
  {
    std::vector<double> x_reduced(ps.col_origin.size(), 0.), x;
    for (const auto& value : left) {
      bool found = false;
      for (size_t k = 0; k < ps.col_origin.size(); ++k) {
        if (ps.col_origin[k] == value.first) {
          x_reduced[k] = value.second;
          found        = true;
        }
      }
      GPRS_CHECK(found);
    }
    ps.postsolve(x_reduced, x);
    return x;
  }

  structure_report_t structure() const
  {
    result_t r;
    r.status    = status_t::reduced;
    r.reduced   = reduced;
    r.postsolve = ps;
    return gprs_check::structure_check(model, r, opt.primal_only);
  }

  // The alias record of a column, which must have one.
  const alias_record_t& alias(int32_t col) const
  {
    for (const alias_record_t& r : ps.aliases) {
      if (r.col == col) { return r; }
    }
    GPRS_CHECK(!"the column has no alias");
    return ps.aliases.front();
  }
};

void check_alias(const alias_record_t& r, int32_t rep, int32_t row, double scale, double offset)
{
  GPRS_CHECK_EQ(r.rep, rep);
  GPRS_CHECK_EQ(r.row, row);
  GPRS_CHECK_EQ(r.scale, scale);
  GPRS_CHECK_EQ(r.offset, offset);
  GPRS_CHECK_EQ(r.origin, probing);
}

}  // namespace

// 2 p + x + y = 2 with x and y in [0, 1]: p = 1 puts both at 0 and p = 0 both at 1. The row
// leaves the model, the two columns become 1 - p in their other rows, and the cost follows.
GPRS_TEST(columns_probing_by_hand)
{
  //   r0:  2 p + x + y      = 2
  //   r1:      3 x     + z <= 4
  //   r2:  p     - 2 y + z >= -1
  call_t c({binary, unit, unit, {0., 5., false}},
           {{{{0, 2.}, {1, 1.}, {2, 1.}}, 2., 2.},
            {{{1, 3.}, {3, 1.}}, -inf, 4.},
            {{{0, 1.}, {2, -2.}, {3, 1.}}, -1., inf}},
           {1., 10., 100., 0.});
  c.run();
  GPRS_CHECK_EQ(c.result.outcome, ok);
  GPRS_CHECK_EQ(c.result.findings, 2);
  GPRS_CHECK_EQ(c.result.dual_findings, 0);
  GPRS_CHECK(!c.result.stopped && !c.result.limited);
  GPRS_CHECK_EQ(c.result.columns.probing_rows, 1);
  GPRS_CHECK_EQ(c.result.columns.probing_rows_deleted, 1);
  GPRS_CHECK_EQ(c.result.columns.probing_rows_cut, 0);
  GPRS_CHECK_EQ(c.result.columns.alias_depth, 1);
  GPRS_CHECK_EQ(c.result.columns.probing_deep_batches, 0);
  GPRS_CHECK_EQ(c.result.columns.probing_max_scale, 1.);
  // One entry of x moves, the entry of y joins that of p.
  GPRS_CHECK_EQ(c.result.columns.probing_entry_changes, 3);
  GPRS_CHECK(!c.st.row_alive[0]);
  GPRS_REQUIRE(c.ps.aliases.size() == 2);
  check_alias(c.alias(1), 0, 0, -1., 1.);
  check_alias(c.alias(2), 0, 0, -1., 1.);
  // r1: 3 (1 - p) + z <= 4, r2: p - 2 (1 - p) + z >= -1.
  GPRS_CHECK_EQ(rule_test::coefficient(c.st, 1, 0), -3.);
  GPRS_CHECK_EQ(c.st.row_upper[1], 1.);
  GPRS_CHECK_EQ(rule_test::coefficient(c.st, 2, 0), 3.);
  GPRS_CHECK_EQ(c.st.row_lower[2], 1.);
  // 1 p + 10 (1 - p) + 100 (1 - p).
  GPRS_CHECK_EQ(c.st.cost[0], -109.);
  GPRS_CHECK_EQ(dd_value(c.st.objective_offset), 110.);
  GPRS_CHECK_EQ(c.st.col_lower[0], 0.);
  GPRS_CHECK_EQ(c.st.col_upper[0], 1.);
  GPRS_CHECK(c.st.is_live(0) && !c.st.is_live(1) && !c.st.is_live(2));
  GPRS_CHECK(silent(c.structure()));
  std::vector<double> x = c.postsolved({{0, 1.}, {3, 2.}});
  GPRS_CHECK(x == std::vector<double>({1., 0., 0., 2.}));
  x = c.postsolved({{0, 0.}, {3, 2.}});
  GPRS_CHECK(x == std::vector<double>({0., 1., 1., 2.}));
  // The rows that changed are listed again, and a second call finds nothing.
  GPRS_CHECK(c.listed(1) && c.listed(2));
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 0);
  GPRS_CHECK(!c.listed(1) && !c.listed(2));
}

// The signs of the coefficients decide which bound a column takes with which value of the
// binary, and the range of a column is the scale: integer columns of any range and a binary
// with a negative coefficient.
GPRS_TEST(columns_probing_signs_and_ranges)
{
  // -6 p + 2 x - y = 1 with x in [1, 3] and y in [-1, 1], both integer: 2 x - y ranges over
  // [1, 7]. p = 1: 2 x - y = 7, x = 3 and y = -1; p = 0: 2 x - y = 1, x = 1 and y = 1.
  call_t c({binary, {1., 3., true}, {-1., 1., true}, unit},
           {{{{0, -6.}, {1, 2.}, {2, -1.}}, 1., 1.}, {{{1, 1.}, {2, 1.}, {3, 1.}}, -inf, 9.}});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 2);
  check_alias(c.alias(1), 0, 0, 2., 1.);
  check_alias(c.alias(2), 0, 0, -2., 1.);
  GPRS_CHECK_EQ(c.result.columns.probing_max_scale, 2.);
  GPRS_CHECK(silent(c.structure()));
  GPRS_CHECK(c.postsolved({{0, 1.}}) == std::vector<double>({1., 3., -1., 0.}));
  GPRS_CHECK(c.postsolved({{0, 0.}}) == std::vector<double>({0., 1., 1., 0.}));
}

// An integer column with bounds one apart can be the root of a tree of the batch, and the
// offset of every alias below it then holds its lower bound times a scale: only the bounds 0
// and 1 are taken, for the binary and for every other such column of the row.
GPRS_TEST(columns_probing_unit_columns_have_the_bounds_0_and_1)
{
  auto run = [](col_t p, col_t x) {
    // 5 p + 2 x + 3 y = side with y in [0, 1]: p at its upper bound puts x and y on their lower
    // bounds.
    double side = 5. * p.upper + 2. * x.lower;
    call_t c({p, x, binary, unit},
             {{{{0, 5.}, {1, 2.}, {2, 3.}}, side, side}, {{{0, 1.}, {1, 1.}, {3, 1.}}, -inf, 99.}});
    c.run();
    GPRS_CHECK(!c.result.limited);
    GPRS_CHECK_EQ(c.result.columns.probing_unit_refused, c.result.findings == 0 ? 1 : 0);
    GPRS_CHECK_EQ(c.result.columns.probing_scale_refused, 0);
    if (c.result.findings > 0) { GPRS_CHECK(silent(c.structure())); }
    return c.result.findings;
  };
  GPRS_CHECK_EQ(run(binary, binary), 2);
  GPRS_CHECK_EQ(run({3., 4., true}, binary), 0);
  GPRS_CHECK_EQ(run({-1., 0., true}, binary), 0);
  GPRS_CHECK_EQ(run(binary, {7., 8., true}), 0);
  // Bounds one apart on a continuous column, or further apart on an integer one, are the
  // bounds of a column that is no root: any are taken.
  GPRS_CHECK_EQ(run(binary, {7., 8., false}), 2);
}

// What ends a visit before a scan: no equation, two free entries, an infinite bound, a side
// beyond the size that a rule reads, activities that do not lie equally far from the side.
GPRS_TEST(columns_probing_rows_that_are_left)
{
  const std::vector<std::pair<int32_t, double>> entries = {{0, 2.}, {1, 1.}, {2, 1.}};
  auto findings = [&entries](const std::vector<col_t>& cols, double lower, double upper) {
    call_t c(cols, {{entries, lower, upper}});
    c.run();
    GPRS_CHECK_EQ(c.result.columns.probing_scale_refused, 0);
    GPRS_CHECK_EQ(c.result.columns.probing_size_refused, 0);
    return c.result.findings;
  };
  GPRS_CHECK_EQ(findings({binary, unit, unit}, 2., 2.), 2);
  GPRS_CHECK_EQ(findings({binary, unit, unit}, 2., 2.5), 0);
  GPRS_CHECK_EQ(findings({binary, unit, unit}, -inf, 2.), 0);
  GPRS_CHECK_EQ(findings({binary, unit, unit}, 3., 3.), 0);
  GPRS_CHECK_EQ(findings({binary, unit, {0., inf, false}}, 2., 2.), 0);
  GPRS_CHECK_EQ(findings({binary, unit, {-1e21, 1., false}}, 2., 2.), 0);
  // No integer column with bounds one apart carries half of the range.
  GPRS_CHECK_EQ(findings({unit, unit, unit}, 2., 2.), 0);
  GPRS_CHECK_EQ(findings({{0., 1., true}, {0., 1., true}, {0., 2., true}}, 2.5, 2.5), 0);
  // An integer column with three values that carries half of the range: 2 q + x + y = 4 with
  // x and y in [0, 2] is met by q = 1 with any x + y = 2.
  {
    call_t c({{0., 2., true}, {0., 2., true}, {0., 2., true}}, {{entries, 4., 4.}});
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
  }
  // Two free entries are a doubleton equation, whatever a fixed column adds.
  {
    call_t c({binary, unit, {1., 1., false}}, {{entries, 2., 2.}});
    fix_column(c.st, c.ps, 2, 1.);
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
  }
  // A side of 1e9 holds its last digits to 1e-7 only.
  {
    call_t c({binary, {0., 1e9, false}, unit},
             {{{{0, 1e9 + 1.}, {1, 1.}, {2, 1.}}, 1e9 + 1., 1e9 + 1.}});
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
  }
}

// The term of a fixed column is a constant of both activities: the row is read without it.
GPRS_TEST(columns_probing_row_with_a_fixed_column)
{
  //   2 p + x + y + 5 f = 7, f fixed at 1.
  call_t c({binary, unit, unit, {0., 3., false}},
           {{{{0, 2.}, {1, 1.}, {2, 1.}, {3, 5.}}, 7., 7.}, {{{1, 1.}, {2, 1.}}, -inf, 9.}});
  fix_column(c.st, c.ps, 3, 1.);
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 2);
  check_alias(c.alias(1), 0, 0, -1., 1.);
  check_alias(c.alias(2), 0, 0, -1., 1.);
  GPRS_CHECK(!c.st.row_alive[0]);
  GPRS_CHECK(silent(c.structure()));
}

// The forcing test of the row rules decides: the two gaps within the feasibility tolerance over
// the dynamism of the other entries, and with a continuous column among them positive by
// rounding only.
GPRS_TEST(columns_probing_tolerance_is_that_of_the_forcing_row)
{
  auto findings = [](const std::vector<col_t>& cols,
                     const std::vector<std::pair<int32_t, double>>& entries,
                     double side,
                     double balance = 1.) {
    call_t c(cols, {{entries, side, side}});
    c.opt.probing_balance = balance;
    c.run();
    return c.result.findings;
  };
  const std::vector<std::pair<int32_t, double>> ones = {{0, 2.}, {1, 1.}, {2, 1.}};
  // Integer columns: a side off by 4e-7 leaves the two gaps at 4e-7 and -4e-7.
  GPRS_CHECK_EQ(findings({binary, binary, binary}, ones, 2. + 4e-7), 2);
  GPRS_CHECK_EQ(findings({binary, binary, binary}, ones, 2. - 4e-7), 2);
  // Off by 8e-7 each gap is within the tolerance and the two are 1.6e-6 apart: the balance
  // test leaves the row, and takes it with two tolerances.
  GPRS_CHECK_EQ(findings({binary, binary, binary}, ones, 2. + 8e-7), 0);
  GPRS_CHECK_EQ(findings({binary, binary, binary}, ones, 2. + 8e-7, 2.), 2);
  GPRS_CHECK_EQ(findings({binary, binary, binary}, ones, 2. + 1.2e-6, 2.), 0);
  // A coefficient of 1000 among the others divides the tolerance: 4e-7 is beyond 1e-9.
  const std::vector<std::pair<int32_t, double>> steep = {{0, 1001.}, {1, 1000.}, {2, 1.}};
  GPRS_CHECK_EQ(findings({binary, binary, binary}, steep, 1001.), 2);
  GPRS_CHECK_EQ(findings({binary, binary, binary}, steep, 1001. + 4e-7), 0);
  // A continuous column: the gap at which the columns sit at their minimum may be negative,
  // and positive by rounding only.
  GPRS_CHECK_EQ(findings({binary, unit, binary}, ones, 2.), 2);
  GPRS_CHECK_EQ(findings({binary, unit, binary}, ones, 2. + 4e-7), 0);
  GPRS_CHECK_EQ(findings({binary, unit, binary}, ones, 2. - 4e-7), 0);
  // One rounding step of the side is rounding of the terms where the terms are not zero: with
  // x in [3, 4] the other terms are 3 at their minimum; with x in [0, 1] they are 0 there.
  const col_t shifted{3., 4., false};
  GPRS_CHECK_EQ(findings({binary, shifted, binary}, ones, 5.), 2);
  GPRS_CHECK_EQ(findings({binary, shifted, binary}, ones, std::nextafter(5., 6.)), 2);
  GPRS_CHECK_EQ(findings({binary, shifted, binary}, ones, 5. + 4e-7), 0);
  GPRS_CHECK_EQ(findings({binary, unit, binary}, ones, std::nextafter(2., 3.)), 0);
}

// Fixed model 7 of the design: 2 p + x1 + x2 = 2 + 2e-7 with x1 continuous in [0, 1] and x2
// binary. With p = 1 the row needs x1 = 2e-7, not 0, and p = 0 has no point. The alias
// x1 = 1 - p would put x1 on 0, which another row that holds x1 with a coefficient of 1000 does
// not allow: the rule leaves the row, as the forcing row leaves a continuous column that is off
// its bound.
GPRS_TEST(columns_probing_leaves_a_continuous_column_that_is_off_its_bound)
{
  // The model of the design, with the second row 1000 x1 >= 1.5e-4.
  {
    call_t c({binary, unit, binary},
             {{{{0, 2.}, {1, 1.}, {2, 1.}}, 2. + 2e-7, 2. + 2e-7}, {{{1, 1000.}}, 1.5e-4, inf}});
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
    GPRS_CHECK(c.st.row_alive[0]);
  }
  // The same through the engine and the oracle. The second row holds a second column, so that
  // no singleton row makes the bound of x1 explicit before the rule reads the row; q has a cost,
  // so that the lock rule leaves it.
  //   r0:  2 p + x1 + x2 = 2 + 2e-7
  //   r1:  1000 x1 + q  >= 1 + 1.5e-4
  // min -10 p + q. The only points have p = 1, x1 = 2e-7, x2 = 0 and q = 1.
  model_t base        = rule_test::make_model({binary, unit, binary, binary},
                                              {{{{0, 2.}, {1, 1.}, {2, 1.}}, 2. + 2e-7, 2. + 2e-7},
                                               {{{1, 1000.}, {3, 1.}}, 1. + 1.5e-4, inf}});
  base.cost           = {-10., 0., 0., 1.};
  oracle::case_t c    = oracle::make_case(base, {}, {});
  result_t result     = presolve(c.model, options_t());
  oracle::verdict_t v = oracle::check_result(c, options_t(), result);
  GPRS_CHECK(v.ok);
  if (!v.ok) { std::printf("    %s\n", v.message.c_str()); }
  for (const alias_record_t& r : result.postsolve.aliases) {
    GPRS_CHECK(r.origin != probing);
  }
  GPRS_CHECK_EQ(result.stats[rule::simple_probing].findings, 0);
  std::vector<uint8_t> line;
  GPRS_REQUIRE(oracle::classify(c.base, 3, line));
  oracle::solution_t s = oracle::solve(c.base, line);
  GPRS_CHECK_EQ(s.outcome, oracle::outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, -9.);
}

// The columns that can be the binary are tried in increasing index, and the first one whose
// other entries pass the forcing test is taken.
GPRS_TEST(columns_probing_tries_the_binaries_in_index_order)
{
  // 2 p + 2e-7 r + (2 + 2e-7) q = 2 + 2e-7, all binary. Both p and q carry half of the range
  // within the feasibility tolerance. With p as the binary the gaps are 2e-7 against a
  // tolerance of 1e-6 * 2e-7 / 2, the coefficient of r being among the others; q carries half
  // of the range exactly.
  call_t c({binary, binary, binary},
           {{{{0, 2.}, {1, 2e-7}, {2, 2. + 2e-7}}, 2. + 2e-7, 2. + 2e-7}});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 2);
  // p = 1 - q and r = 1 - q, whichever of the three is the root: r = p.
  GPRS_CHECK(c.st.is_live(0));
  check_alias(c.alias(2), 0, 0, -1., 1.);
  check_alias(c.alias(1), 0, 0, 1., 0.);
  GPRS_CHECK(silent(c.structure()));
}

// The scale of an alias is the range of its column: beyond opt.alias_scale_limit the row is
// left, by a test that reads the state alone.
GPRS_TEST(columns_probing_scale_limit)
{
  auto run = [](double range, double limit) {
    call_t c({binary, {0., range, false}, unit, unit},
             {{{{0, range + 1.}, {1, 1.}, {2, 1.}}, range + 1., range + 1.},
              {{{1, 1.}, {3, 1.}}, -inf, 9.}});
    c.opt.alias_scale_limit = limit;
    c.run();
    GPRS_CHECK(!c.result.limited);
    GPRS_CHECK_EQ(c.result.columns.probing_scale_refused, c.result.findings == 0 ? 1 : 0);
    if (c.result.findings > 0) { GPRS_CHECK_EQ(c.result.columns.probing_max_scale, range); }
    return c.result.findings;
  };
  GPRS_CHECK_EQ(run(1e6, 1e6), 2);
  GPRS_CHECK_EQ(run(1e6 + 1., 1e6), 0);
  GPRS_CHECK_EQ(run(1e6 + 1., 1e7), 2);
  GPRS_CHECK_EQ(run(4., 3.), 0);
}

// The offset of an alias, a bound of its column, enters the sides of the rows of the column,
// times its coefficient there: beyond the size that a side holds within the tolerance, or with
// such a row whose side is beyond that size already, the row is left, and the call reports
// limited, since the test reads rows that change without an event of the row it leaves.
GPRS_TEST(columns_probing_size_of_an_offset)
{
  // 2 p + x + y = start + 2 with x in [start, start + 1] and y in [0, 1]; x has `coefficient`
  // in a second row and p has `binary_coefficient` in a third one, with the side `side`.
  auto run = [](double start, double coefficient, double binary_coefficient, double side) {
    call_t c({binary, {start, start + 1., false}, unit, unit},
             {{{{0, 2.}, {1, 1.}, {2, 1.}}, start + 2., start + 2.},
              {{{1, coefficient}, {3, 1.}}, -inf, 9.},
              {{{0, binary_coefficient}, {3, 1.}}, -inf, side}});
    c.run();
    bool refused = c.result.findings == 0;
    GPRS_CHECK_EQ(c.result.columns.probing_size_refused, refused ? 1 : 0);
    GPRS_CHECK_EQ(c.result.columns.probing_scale_refused, 0);
    GPRS_CHECK_EQ(c.result.limited, refused);
    if (!refused) { GPRS_CHECK(silent(c.structure())); }
    return c.result.findings;
  };
  GPRS_CHECK_EQ(run(0., 1., 1., 9.), 2);
  // The bound 1 under a coefficient of 5e7 is within 1e8, under one of 2e8 beyond it.
  GPRS_CHECK_EQ(run(0., 5e7, 1., 9.), 2);
  GPRS_CHECK_EQ(run(0., 2e8, 1., 9.), 0);
  GPRS_CHECK_EQ(run(1e6 - 1., 50., 1., 9.), 2);
  GPRS_CHECK_EQ(run(1e6 - 1., 200., 1., 9.), 0);
  // The binary is substituted as well when another column is the root.
  GPRS_CHECK_EQ(run(0., 1., 5e7, 9.), 2);
  GPRS_CHECK_EQ(run(0., 1., 2e8, 9.), 0);
  // A row of a column with a side of 1e9.
  GPRS_CHECK_EQ(run(0., 1., 1., 5e7), 2);
  GPRS_CHECK_EQ(run(0., 1., 1., 1e9), 0);
}

// Two rows that share two columns: the second closes a cycle of the batch and waits on the list
// for the next call, which finds it as a doubleton equation.
GPRS_TEST(columns_probing_row_whose_columns_are_connected_waits)
{
  //   r0:  2 p + x + y = 2
  //   r1:  2 q + x + y = 2
  call_t c({binary, binary, unit, unit},
           {{{{0, 2.}, {2, 1.}, {3, 1.}}, 2., 2.}, {{{1, 2.}, {2, 1.}, {3, 1.}}, 2., 2.}});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 2);
  GPRS_CHECK_EQ(c.result.columns.probing_rows, 1);
  GPRS_CHECK_EQ(c.result.columns.probing_connected, 1);
  GPRS_CHECK(!c.st.row_alive[0] && c.st.row_alive[1]);
  GPRS_CHECK(c.listed(1));
  // r1 is 2 q - 2 p = 0 now.
  GPRS_CHECK_EQ(c.st.row_free_size[1], 2);
  GPRS_CHECK_EQ(rule_test::coefficient(c.st, 1, 0), -2.);
  GPRS_CHECK_EQ(c.st.row_upper[1], 0.);
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 0);
  GPRS_CHECK(!c.listed(1));
  GPRS_CHECK(silent(c.structure()));
}

// A row whose two connected columns the batch does not substitute stays as it is and has no
// event: it is on the list by the family's own push.
GPRS_TEST(columns_probing_connected_row_without_an_event_is_listed)
{
  //   r0:  2 p1 + x + w  = 2      p1 - x, p1 - w
  //   r1:  2 p2 + x + y  = 2      p2 - x, p2 - y: the edge from x to p2 is cut
  //   r2:  2 p3 + p1 + p2 = 2     p1 and p2 are connected through x
  call_t c({binary, binary, binary, unit, unit, unit},
           {{{{0, 2.}, {3, 1.}, {4, 1.}}, 2., 2.},
            {{{1, 2.}, {3, 1.}, {5, 1.}}, 2., 2.},
            {{{0, 1.}, {1, 1.}, {2, 2.}}, 2., 2.}});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 3);
  GPRS_CHECK_EQ(c.result.columns.probing_rows, 2);
  GPRS_CHECK_EQ(c.result.columns.probing_rows_deleted, 1);
  GPRS_CHECK_EQ(c.result.columns.probing_rows_cut, 1);
  GPRS_CHECK_EQ(c.result.columns.probing_connected, 1);
  GPRS_CHECK(c.st.is_live(0) && c.st.is_live(1) && c.st.is_live(2));
  GPRS_CHECK(c.st.row_alive[2] && c.listed(2));
  GPRS_CHECK_EQ(c.st.row_free_size[2], 3);
  // The next call takes it: p1 = 1 - p3 and p2 = 1 - p3, or with another root.
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 2);
  GPRS_CHECK(!c.st.row_alive[2]);
  GPRS_CHECK(silent(c.structure()));
}

// The model of the review of the design: a row whose edge to its binary is cut stays in the
// model, so the alias that it gave names no row, and a later record may name that row.
GPRS_TEST(columns_probing_row_with_a_cut_edge_names_no_row)
{
  //   A:  2 p1 + x + w = 2
  //   B:  2 p2 + x + y = 2
  //   C:  p2 - p1 + s <= 1, s a singleton column
  // With p1 as the root, x = 1 - p1 and w = 1 - p1 by A; p2 cannot hang from the continuous x
  // and is a root; y = 1 - p2 by B, which stays as p2 - p1 = 0.
  std::vector<col_t> cols = {binary, binary, unit, unit, unit, {0., 5., false}};
  std::vector<row_t> rows = {{{{0, 2.}, {2, 1.}, {3, 1.}}, 2., 2.},
                             {{{1, 2.}, {2, 1.}, {4, 1.}}, 2., 2.},
                             {{{0, -1.}, {1, 1.}, {5, 1.}}, -inf, 1.}};
  {
    call_t c(cols, rows);
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 3);
    GPRS_CHECK_EQ(c.result.columns.probing_rows, 2);
    GPRS_CHECK_EQ(c.result.columns.probing_rows_deleted, 1);
    GPRS_CHECK_EQ(c.result.columns.probing_rows_cut, 1);
    check_alias(c.alias(2), 0, 0, -1., 1.);
    check_alias(c.alias(3), 0, 0, -1., 1.);
    check_alias(c.alias(4), 1, -1, -1., 1.);
    GPRS_CHECK(!c.st.row_alive[0] && c.st.row_alive[1]);
    GPRS_CHECK_EQ(c.st.row_free_size[1], 2);
    GPRS_CHECK_EQ(rule_test::coefficient(c.st, 1, 0), -1.);
    GPRS_CHECK_EQ(rule_test::coefficient(c.st, 1, 1), 1.);
    GPRS_CHECK_EQ(c.st.row_lower[1], 0.);
    GPRS_CHECK_EQ(c.st.row_upper[1], 0.);
    // The row is on the lists by the change of its entries.
    GPRS_CHECK(c.listed(1));
    GPRS_CHECK(c.st.row_frontier.pending[1] & list_bit(row_list::doubleton));
    GPRS_CHECK(silent(c.structure()));
  }
  // Through the engine, with the singleton column kept in its row (a cost against its lock, no
  // singleton-column rule): the parallel rows subtract B from C in the next round, a record on
  // a row that an alias of the batch came from.
  model_t m = rule_test::make_model(cols, rows);
  m.cost    = {0., 0., 0., 0., 0., -1.};
  options_t options;
  options.enabled[rule::singleton_cols] = false;
  result_t result                       = presolve(m, options);
  GPRS_CHECK(gprs_check::carries_model(result.status));
  GPRS_CHECK(result.fixed_point);
  bool subtracted = false;
  for (const row_addition_record_t& r : result.postsolve.row_additions) {
    subtracted = subtracted || (r.row == 2 && r.equation_row == 1);
  }
  GPRS_CHECK(subtracted);
  int32_t without_row = 0;
  for (const alias_record_t& r : result.postsolve.aliases) {
    without_row += r.origin == probing && r.row == -1;
  }
  GPRS_CHECK_EQ(without_row, 1);
  structure_report_t s = gprs_check::structure_check(m, result, false);
  GPRS_CHECK_EQ(s.stack_error, std::string());
  GPRS_CHECK(silent(s));
  GPRS_CHECK_EQ(presolve(result.reduced, options).status, status_t::unchanged);
}

// Fixed model 10 of the design: a chain of equations x[k] + y[k] - 2 x[k + 1] = 0 over binaries
// makes every column a copy of one of them, k aliases deep, and x[1] sits in 100 further rows.
// The batch substitutes every column once, into its root: the entries of x[1] move once,
// however long the chain is.
GPRS_TEST(columns_probing_chain_moves_every_entry_once)
{
  const int32_t links = 30, further = 100;
  for (bool first_continuous : {true, false}) {
    // Columns: x[1..links + 1], then y[1..links], then w.
    std::vector<col_t> cols(2 * links + 2, binary);
    cols[0]     = first_continuous ? unit : binary;
    cols.back() = {0., 4., false};
    int32_t w   = 2 * links + 1;
    std::vector<row_t> rows;
    for (int32_t k = 0; k < links; ++k) {
      rows.push_back({{{k, 1.}, {k + 1, -2.}, {links + 1 + k, 1.}}, 0., 0.});
    }
    for (int32_t k = 0; k < further; ++k) {
      rows.push_back({{{0, 1. + k}, {w, 1.}}, -inf, 200.});
    }
    call_t c(cols, rows);
    c.run();
    GPRS_CHECK_EQ(c.result.outcome, ok);
    GPRS_CHECK_EQ(c.result.findings, 2 * links);
    GPRS_CHECK_EQ(c.result.columns.probing_rows, links);
    GPRS_CHECK_EQ(c.result.columns.probing_rows_deleted, links);
    GPRS_CHECK_EQ(c.result.columns.probing_rows_cut, 0);
    GPRS_CHECK_EQ(c.result.columns.probing_deep_batches, 1);
    GPRS_CHECK_EQ(c.st.live_cols, 2);
    // A continuous x[1] cannot be the root: the chain hangs from x[2], which has two entries
    // as every later x has, and x[1] moves into it. An integer x[1] has the most entries and
    // is the root: nothing moves.
    // The last y hangs from the last x.
    GPRS_CHECK_EQ(c.result.columns.alias_depth, first_continuous ? links : links + 1);
    GPRS_CHECK_EQ(c.result.columns.probing_entry_changes, first_continuous ? further : 0);
    GPRS_CHECK(c.st.is_live(first_continuous ? 1 : 0));
    // The rows of the chain once for their binaries and once for the binary that is tried, the
    // further rows not at all; the columns of the batch once for the size of their offsets and
    // once more, as the columns of their rows, for the substitutions.
    GPRS_CHECK(c.budget.used <= line_visit_work * (links + further) + 2 * 3 * links +
                                  (3 * links + further) + (5 * links + further));
    GPRS_CHECK(silent(c.structure()));
    std::vector<double> x = c.postsolved({{first_continuous ? 1 : 0, 1.}, {w, 2.}});
    for (int32_t j = 0; j < w; ++j) {
      GPRS_CHECK_EQ(x[j], 1.);
    }
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
  }
}

// The budget is tested between two rows: a call that stops returns the rows it did not visit,
// applies what it collected, and the next call goes on.
GPRS_TEST(columns_probing_budget)
{
  // Four separate rows 2 p + x + y = 2.
  std::vector<col_t> cols;
  std::vector<row_t> rows;
  for (int32_t k = 0; k < 4; ++k) {
    cols.insert(cols.end(), {binary, unit, unit});
    rows.push_back({{{3 * k, 2.}, {3 * k + 1, 1.}, {3 * k + 2, 1.}}, 2., 2.});
  }
  call_t c(cols, rows);
  // One row costs its visit, two scans and the entries of its three columns.
  c.budget.limit       = 2 * (line_visit_work + 6 + 3);
  c.budget.stall_limit = c.budget.limit;
  c.run();
  GPRS_CHECK(c.result.stopped);
  GPRS_CHECK_EQ(c.result.findings, 4);
  GPRS_CHECK(!c.st.row_alive[0] && !c.st.row_alive[1] && c.st.row_alive[2] && c.st.row_alive[3]);
  GPRS_CHECK(c.listed(2) && c.listed(3));
  c.budget.limit       = rule_test::unlimited;
  c.budget.stall_limit = rule_test::unlimited;
  c.run();
  GPRS_CHECK(!c.result.stopped);
  GPRS_CHECK_EQ(c.result.findings, 4);
  GPRS_CHECK_EQ(c.st.live_rows, 0);
  GPRS_CHECK(silent(c.structure()));
}

// An integer column with bounds one apart among the others: the batch may hang the binary from
// it, and both directions give the same model.
GPRS_TEST(columns_probing_root_is_an_integer_column_with_the_most_entries)
{
  //   r0:  2 p + x + y = 2, all binary
  //   r1..r3 hold y.
  call_t c({binary, binary, binary, unit},
           {{{{0, 2.}, {1, 1.}, {2, 1.}}, 2., 2.},
            {{{2, 1.}, {3, 1.}}, -inf, 9.},
            {{{2, 2.}, {3, 1.}}, -inf, 9.},
            {{{2, 3.}, {3, 1.}}, -inf, 9.}});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 2);
  GPRS_CHECK(c.st.is_live(2) && !c.st.is_live(0) && !c.st.is_live(1));
  // p = 1 - y and x = y.
  check_alias(c.alias(0), 2, 0, -1., 1.);
  check_alias(c.alias(1), 2, 0, 1., 0.);
  GPRS_CHECK_EQ(c.result.columns.alias_depth, 2);
  GPRS_CHECK_EQ(c.result.columns.probing_entry_changes, 0);
  GPRS_CHECK(silent(c.structure()));
}

// The engine with the family: a model whose only reduction is a probing row ends at a fixed
// point in both modes, and the presolve of the reduced model changes nothing.
GPRS_TEST(columns_probing_in_the_engine)
{
  //   r0:  2 p + x + y      = 2
  //   r1:      x     + z + u <= 2.5
  //   r2:  p     + y + z - u >= 0.5
  model_t m = rule_test::make_model({binary, unit, unit, {0., 3., false}, {0., 3., false}},
                                    {{{{0, 2.}, {1, 1.}, {2, 1.}}, 2., 2.},
                                     {{{1, 1.}, {3, 1.}, {4, 1.}}, -inf, 2.5},
                                     {{{0, 1.}, {2, 1.}, {3, 1.}, {4, -1.}}, 0.5, inf}});
  m.cost    = {1., 1., -1., 1., -1.};
  for (bool primal_only : {false, true}) {
    options_t options;
    options.primal_only = primal_only;
    result_t result     = presolve(m, options);
    GPRS_CHECK_EQ(result.status, status_t::reduced);
    GPRS_CHECK(result.fixed_point);
    GPRS_CHECK_EQ(result.stats[rule::simple_probing].findings, 2);
    GPRS_CHECK_EQ(result.stats[rule::simple_probing].rows_removed, 1);
    GPRS_CHECK_EQ(result.stats[rule::simple_probing].cols_removed, 2);
    GPRS_CHECK_EQ(result.columns.probing_rows_deleted, 1);
    GPRS_CHECK_EQ(result.columns.alias_depth, 1);
    GPRS_CHECK(silent(gprs_check::structure_check(m, result, primal_only)));
    GPRS_CHECK_EQ(presolve(result.reduced, options).status, status_t::unchanged);
    options.enabled[rule::simple_probing] = false;
    result_t off                          = presolve(m, options);
    GPRS_CHECK_EQ(off.stats[rule::simple_probing].calls, 0);
    GPRS_CHECK_EQ(off.reduced.nrows, 3);
  }
}

// The generator knob p_probing_row: the rule fires on its rows, in every way the batch can end
// for a row, and the oracle agrees with every result, with the family alone and with every
// rule.
GPRS_TEST(columns_probing_on_generated_rows)
{
  oracle::generator_options_t shape;
  shape.p_probing_row    = 0.8;
  shape.max_probing_rows = 3;
  shape.max_width        = 2;
  shape.max_enum_cols    = 6;
  shape.max_line_cols    = 1;
  shape.p_infinite_bound = 0.2;
  int64_t fired = 0, deleted = 0, cut = 0, connected = 0, deep = 0, refused = 0, failures = 0;
  int64_t without_row = 0, units = 0;
  for (uint64_t seed = 7500000; seed < 7506000 && failures < 3; ++seed) {
    oracle::case_t c = oracle::generate(seed, shape);
    for (bool alone : {true, false}) {
      options_t options;
      if (alone) {
        options.enabled.fill(false);
        options.enabled[rule::simple_probing] = true;
      }
      // One seed in four with a limit on the scale that the wider columns do not pass.
      if (seed % 4 == 0) { options.alias_scale_limit = 2.; }
      result_t result     = presolve(c.model, options);
      oracle::verdict_t v = oracle::check_result(c, options, result);
      if (!v.ok && v.message.find("outside the oracle class") == std::string::npos) {
        ++failures;
        test::fail(__FILE__, __LINE__, "seed " + std::to_string(seed) + ": " + v.message);
      }
      const column_stats_t& s = result.columns;
      fired += s.probing_rows > 0;
      deleted += s.probing_rows_deleted;
      cut += s.probing_rows_cut;
      connected += s.probing_connected;
      deep += s.probing_deep_batches;
      refused += s.probing_scale_refused;
      units += s.probing_unit_refused;
      for (const alias_record_t& r : result.postsolve.aliases) {
        without_row += r.origin == probing && r.row == -1;
      }
      if (gprs_check::carries_model(result.status)) {
        structure_report_t report =
          gprs_check::structure_check(c.model, result, options.primal_only);
        if (!silent(report) || !report.stack_error.empty()) {
          ++failures;
          test::fail(__FILE__, __LINE__, "seed " + std::to_string(seed) + ": structure check");
        }
      }
    }
  }
  GPRS_CHECK_EQ(failures, 0);
  if (std::getenv("GPRS_ORACLE_CENSUS")) {
    std::printf(
      "    fired %lld deleted %lld cut %lld connected %lld deep %lld refused %lld "
      "aliases without a row %lld bounds one apart, not 0 and 1: %lld\n",
      (long long)fired,
      (long long)deleted,
      (long long)cut,
      (long long)connected,
      (long long)deep,
      (long long)refused,
      (long long)without_row,
      (long long)units);
  }
  GPRS_CHECK(fired > 800 && deleted > 800 && cut >= 1 && connected > 400 && deep > 200);
  GPRS_CHECK(without_row >= 1 && refused > 40 && units > 400);
}

// The knob p_probing_row: without it a seed gives the case it gave before; with it the rows
// counted before come out as they were and equations follow them, most of them with the two
// activities equally far from the side and an enumerated column with two values that carries
// half of the range.
GPRS_TEST(columns_probing_generator_knob)
{
  oracle::generator_options_t plain, none, with;
  none.p_probing_row    = 1.;
  none.max_probing_rows = 0;
  with.p_probing_row    = 1.;
  with.p_scaled         = 0.;
  with.p_crossing       = 0.;
  plain.p_scaled        = 0.;
  plain.p_crossing      = 0.;
  int64_t added = 0, balanced = 0, carried = 0;
  for (uint64_t seed = 7600000; seed < 7602000; ++seed) {
    oracle::generator_options_t defaults;
    GPRS_CHECK_EQ(model_hash(oracle::generate(seed, defaults).model),
                  model_hash(oracle::generate(seed, none).model));
    model_t before = oracle::generate(seed, plain).base;
    model_t m      = oracle::generate(seed, with).base;
    GPRS_REQUIRE(m.nrows >= before.nrows && m.nrows <= before.nrows + with.max_probing_rows);
    GPRS_CHECK_EQ(m.ncols, before.ncols);
    std::vector<uint8_t> line;
    GPRS_CHECK(oracle::classify(m, with.max_width, line));
    for (int32_t i = 0; i < before.nrows; ++i) {
      int64_t length = before.row_start[i + 1] - before.row_start[i];
      GPRS_REQUIRE(m.row_start[i + 1] - m.row_start[i] == length);
      for (int64_t k = 0; k < length; ++k) {
        GPRS_CHECK_EQ(m.col_index[m.row_start[i] + k], before.col_index[before.row_start[i] + k]);
        GPRS_CHECK_EQ(m.value[m.row_start[i] + k], before.value[before.row_start[i] + k]);
      }
    }
    for (int32_t i = before.nrows; i < m.nrows; ++i) {
      ++added;
      GPRS_CHECK_EQ(m.row_lower[i], m.row_upper[i]);
      double low = 0., high = 0.;
      bool finite = true;
      for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
        int32_t j = m.col_index[e];
        double a  = m.value[e];
        double l = std::ceil(m.col_lower[j]), u = std::floor(m.col_upper[j]);
        finite = finite && std::fabs(l) <= 13. && std::fabs(u) <= 13.;
        low += a > 0 ? a * l : a * u;
        high += a > 0 ? a * u : a * l;
      }
      if (!finite || high - m.row_upper[i] != m.row_upper[i] - low) { continue; }
      ++balanced;
      for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
        int32_t j = m.col_index[e];
        if (m.col_type[j] == 1 && std::floor(m.col_upper[j]) - std::ceil(m.col_lower[j]) == 1. &&
            std::fabs(m.value[e]) == high - m.row_upper[i]) {
          ++carried;
          break;
        }
      }
    }
  }
  if (std::getenv("GPRS_ORACLE_CENSUS")) {
    std::printf("    probing rows %lld, balanced %lld, with a column that carries half %lld\n",
                (long long)added,
                (long long)balanced,
                (long long)carried);
  }
  GPRS_CHECK(added > 1500 && balanced > added / 4 && carried > added / 5);
}

// A continuous column whose two bounds are rounding apart is one value as far as its row can
// tell. The forcing test passes it, as it passes a column that it then fixes; an alias between
// the two bounds would have a scale of rounding size, and with it the other rows of the column
// would get coefficients of rounding size that decide the binary by a quotient of two
// roundings. The row is left.
GPRS_TEST(columns_probing_leaves_a_column_whose_range_is_rounding)
{
  // 3 x + 2 p - 2 y = 2 with x within six rounding steps of 2 / 3: p = y whatever x is.
  const double low = 0.66666666666666663, high = 0.66666666666666696;
  {
    call_t c({{low, high, false}, binary, binary}, {{{{0, 3.}, {1, 2.}, {2, -2.}}, 2., 2.}});
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
    GPRS_CHECK_EQ(c.result.columns.probing_narrow_refused, 1);
    GPRS_CHECK(!c.result.limited);
  }
  // The same with an exact range of 2^-40, which is far above the rounding of its bounds 0 and
  // 2^-40: the row is exact and is taken.
  {
    const double narrow = 0x1p-40;
    call_t c({{0., narrow, false}, binary, binary},
             {{{{0, 1.}, {1, 1. + narrow}, {2, 1.}}, 1. + narrow, 1. + narrow}});
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 2);
    GPRS_CHECK_EQ(c.result.columns.probing_narrow_refused, 0);
    GPRS_CHECK(silent(c.structure()));
  }
  // That range next to bounds of size 3: 2^-40 is not far above the rounding of 3.
  {
    const double narrow = 0x1p-40;
    call_t c({{3., 3. + narrow, false}, binary, binary},
             {{{{0, 1.}, {1, 1. + narrow}, {2, 1.}}, 4. + narrow, 4. + narrow}});
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
    GPRS_CHECK_EQ(c.result.columns.probing_narrow_refused, 1);
  }
  // A gap that is positive by one rounding step of the side is rounding against the terms of
  // the row, which are 3 and more, and as large as the whole range of z, 2^-50: the row cannot
  // tell at which of its bounds z sits. With the exact side it can.
  {
    const double tiny = 0x1p-50;
    for (bool exact : {false, true}) {
      double side = exact ? 5. + tiny : std::nextafter(5. + tiny, 9.);
      call_t c({{3., 4., false}, binary, binary, {0., tiny, false}},
               {{{{0, 1.}, {1, 2. + tiny}, {2, 1.}, {3, 1.}}, side, side}});
      c.run();
      GPRS_CHECK_EQ(c.result.findings, exact ? 3 : 0);
      GPRS_CHECK_EQ(c.result.columns.probing_narrow_refused, exact ? 0 : 1);
    }
  }
  // The model of the campaign that showed it, with the rules of its seed: without the row
  // rules no singleton row fixes x0, which row 1 gives the value 2 / 3 once x3 is fixed, and
  // propagation leaves it bounds that are rounding apart.
  //   r0:  3 x0 + 2 x1 - 2 x2 = 2
  //   r1: -3 x0 - 3 x3       = -2
  //   r2:   -x0             <= 2
  model_t base     = rule_test::make_model({{-inf, inf, false}, binary, binary, binary},
                                           {{{{0, 3.}, {1, 2.}, {2, -2.}}, 2., 2.},
                                            {{{0, -3.}, {3, -3.}}, -2., -2.},
                                            {{{0, -1.}}, -inf, 2.}});
  base.cost        = {-1., 2., 0., -3.};
  oracle::case_t c = oracle::make_case(base, {}, {});
  options_t options;
  options.enabled.fill(false);
  for (int32_t r : {rule::propagation,
                    rule::tightening,
                    rule::parallel_rows,
                    rule::dual_fixing,
                    rule::simple_probing,
                    rule::parallel_cols}) {
    options.enabled[r] = true;
  }
  result_t result     = presolve(c.model, options);
  oracle::verdict_t v = oracle::check_result(c, options, result);
  GPRS_CHECK(v.ok);
  if (!v.ok) { std::printf("    %s\n", v.message.c_str()); }
  GPRS_CHECK_EQ(result.stats[rule::simple_probing].findings, 0);
  GPRS_CHECK(result.columns.probing_narrow_refused >= 1);
}
