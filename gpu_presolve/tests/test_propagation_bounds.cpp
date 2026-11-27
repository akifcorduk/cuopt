/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "test_rows_support.hpp"

#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <cmath>
#include <vector>

using namespace gprs;
using namespace rule_test;

namespace {

constexpr rule_outcome_t ok         = rule_outcome_t::ok;
constexpr rule_outcome_t infeasible = rule_outcome_t::infeasible;

rule_result_t run_propagation(fixture_t& f)
{
  bound_propagation_t rule;
  return rule.run(f.st, f.budget, f.ps, f.opt);
}

// One rule object over a fixture: the marks of run() are what make_explicit works on.
struct propagation_t {
  fixture_t& f;
  bound_propagation_t rule;

  rule_result_t run() { return rule.run(f.st, f.budget, f.ps, f.opt); }
  rule_result_t make_explicit() { return rule.make_explicit(f.st, f.ps, f.opt); }
};

void check_bounds(const state_t& st, int32_t col, double lower, double upper)
{
  GPRS_CHECK_EQ(st.col_lower[col], lower);
  GPRS_CHECK_EQ(st.col_upper[col], upper);
}

// Bound propagation on the integer columns from scratch, in plain doubles, to its fixed point:
// the bounds a correct worklist reaches in any order, as long as no bound grows to the size
// where its term counts as infinite (the fixed point then depends on the order).
struct naive_t {
  std::vector<double> lower, upper;
  bool infeasible = false;
  bool converged  = false;

  bool small() const
  {
    for (size_t j = 0; j < lower.size(); ++j) {
      if (std::fabs(lower[j]) > 1e9 && lower[j] > -inf) { return false; }
      if (std::fabs(upper[j]) > 1e9 && upper[j] < inf) { return false; }
    }
    return true;
  }
};

naive_t naive_propagation(const state_t& st, double feas)
{
  naive_t n{st.col_lower, st.col_upper};
  for (int32_t round = 0; round < 300; ++round) {
    bool changed = false;
    for (int32_t i = 0; i < st.nrows; ++i) {
      if (!st.row_alive[i]) { continue; }
      for (bool upper_side : {true, false}) {
        double side = upper_side ? st.row_upper[i] : st.row_lower[i];
        if (std::isinf(side)) { continue; }
        for (int64_t e = st.csr.begin(i); e < st.csr.end(i); ++e) {
          if (!st.csr.live(e)) { continue; }
          int32_t j = st.csr.index[e];
          double a  = st.csr.value[e];
          if (!st.is_integer(j) || n.lower[j] == n.upper[j]) { continue; }
          double rest  = 0.;
          bool unknown = false;
          for (int64_t o = st.csr.begin(i); o < st.csr.end(i); ++o) {
            if (o == e || !st.csr.live(o)) { continue; }
            int32_t k    = st.csr.index[o];
            double b     = st.csr.value[o];
            double bound = (b > 0) == upper_side ? n.lower[k] : n.upper[k];
            if (!(std::fabs(b * bound) <= 1e20)) { unknown = true; }
            rest += b * bound;
          }
          if (unknown) { continue; }
          double bound = (side - rest) / a;
          if ((a > 0) == upper_side) {
            bound = std::floor(bound + feas);
            if (bound < n.upper[j]) { n.upper[j] = bound, changed = true; }
          } else {
            bound = std::ceil(bound - feas);
            if (bound > n.lower[j]) { n.lower[j] = bound, changed = true; }
          }
          if (n.lower[j] > n.upper[j]) {
            n.infeasible = true;
            return n;
          }
        }
      }
    }
    if (!changed) {
      n.converged = true;
      return n;
    }
  }
  return n;
}

}  // namespace

// No infinite term: every column of the row gets its bound, with the sign of its coefficient
// deciding which.
GPRS_TEST(propagation_every_column)
{
  std::vector<col_t> cols(7, col_t{0., 10., true});
  fixture_t f(cols,
              {{{{0, 2.}, {1, 3.}, {2, 1.}}, -inf, 7.},
               {{{3, -2.}, {4, 1.}}, -5., inf},
               {{{5, 1.}, {6, -2.}}, -inf, -3.}});
  rule_result_t result = run_propagation(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK_EQ(result.findings, 5);
  // 2 x0 + 3 x1 + x2 <= 7.
  check_bounds(f.st, 0, 0., 3.);
  check_bounds(f.st, 1, 0., 2.);
  check_bounds(f.st, 2, 0., 7.);
  // -2 x3 + x4 >= -5: x3 <= (10 + 5) / 2.
  check_bounds(f.st, 3, 0., 7.);
  check_bounds(f.st, 4, 0., 10.);
  // x5 - 2 x6 <= -3: x6 >= 3 / 2.
  check_bounds(f.st, 5, 0., 10.);
  check_bounds(f.st, 6, 2., 10.);
  GPRS_CHECK_EQ(f.st.live_rows, 3);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::propagation));
  f.check_consistent();
}

// The case of cuOpt PR 2372: x1 = 2 and x2 = 3 fixed, x3 free, x1 + x2 + x3 <= 10 gives x3 <= 5.
GPRS_TEST(propagation_fixed_columns_and_free_column)
{
  for (bool integer : {true, false}) {
    std::vector<col_t> cols = {{0., 5., true}, {0., 5., false}, {-inf, inf, integer}};
    fixture_t f(cols, {{{{0, 1.}, {1, 1.}, {2, 1.}}, -inf, 10.}});
    fix_column(f.st, f.ps, 0, 2.);
    fix_column(f.st, f.ps, 1, 3.);
    rule_result_t result = run_propagation(f);
    GPRS_CHECK_EQ(result.outcome, ok);
    // An explicit bound is a finding, a recorded one a record.
    GPRS_CHECK_EQ(result.findings, integer ? 1 : 0);
    GPRS_CHECK_EQ(result.records, integer ? 0 : 1);
    if (integer) {
      check_bounds(f.st, 2, -inf, 5.);
    } else {
      check_bounds(f.st, 2, -inf, inf);
      GPRS_CHECK(f.st.implied_upper_valid(2));
      GPRS_CHECK_EQ(f.st.implied_upper[2], 5.);
      GPRS_CHECK_EQ(f.st.implied_upper_row[2], 0);
      GPRS_CHECK(!f.st.implied_lower_valid(2));
    }
    f.check_consistent();
  }
}

GPRS_TEST(propagation_infinite_terms)
{
  {
    // One infinite term in the minimum activity, that of x0: x0 <= 10 - 0 - 1 and nothing for
    // the others, whose bound would need the lower bound of x0.
    std::vector<col_t> cols = {{-inf, 20., true}, {0., 20., true}, {1., 20., true}};
    fixture_t f(cols, {{{{0, 1.}, {1, 1.}, {2, 1.}}, -inf, 10.}});
    int64_t visits       = f.st.slot_visits;
    rule_result_t result = run_propagation(f);
    GPRS_CHECK_EQ(result.findings, 1);
    check_bounds(f.st, 0, -inf, 9.);
    check_bounds(f.st, 1, 0., 20.);
    check_bounds(f.st, 2, 1., 20.);
    // The row is read at one slot: the work is the row visit and the column of the new bound.
    // The row is not visited again: the new bound is in its maximum activity, which no side
    // reads.
    GPRS_CHECK_EQ(f.budget.used, line_visit_work + 1 + 1);
    GPRS_CHECK_EQ(f.st.slot_visits - visits, 1);
  }
  {
    // The same on the lower side with a negative coefficient: -x0 + x1 + x2 >= -50 with x0
    // free has the infinite term in the maximum activity.
    std::vector<col_t> cols = {{-inf, inf, true}, {0., 20., true}, {1., 20., true}};
    fixture_t f(cols, {{{{0, -1.}, {1, 1.}, {2, 1.}}, -50., inf}});
    GPRS_CHECK_EQ(run_propagation(f).findings, 1);
    // -x0 >= -50 - 40.
    check_bounds(f.st, 0, -inf, 90.);
    check_bounds(f.st, 1, 0., 20.);
    check_bounds(f.st, 2, 1., 20.);
  }
  {
    // Two infinite terms: nothing.
    std::vector<col_t> cols = {{-inf, 20., true}, {-inf, 20., true}, {1., 20., true}};
    fixture_t f(cols, {{{{0, 1.}, {1, 1.}, {2, 1.}}, -inf, 10.}});
    rule_result_t result = run_propagation(f);
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 0);
    GPRS_CHECK_EQ(f.st.bound_changes, 0);
  }
  {
    // The single infinite term belongs to a fixed column: 1e15 * 1e6 is above the limit.
    std::vector<col_t> cols = {{0., 2e6, false}, {0., 100., true}};
    fixture_t f(cols, {{{{0, 1e15}, {1, 1.}}, -inf, 5.}});
    fix_column(f.st, f.ps, 0, 1e6);
    GPRS_CHECK_EQ(f.st.min_act.inf_count[0], 1);
    rule_result_t result = run_propagation(f);
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 0);
    check_bounds(f.st, 1, 0., 100.);
  }
}

GPRS_TEST(propagation_integer_rounding)
{
  std::vector<col_t> cols = {{0., 5., true}, {0., 1., true}, {0., 9., true}, {0., 4., true}};
  fixture_t f(cols,
              {{{{0, 2.}, {1, 1.}}, 3., inf},
               // 5.9999995 / 3 is 2 within the tolerance, not 1.
               {{{2, 3.}, {3, 1.}}, -inf, 5.9999995}});
  rule_result_t result = run_propagation(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  check_bounds(f.st, 0, 1., 5.);
  check_bounds(f.st, 1, 0., 1.);
  check_bounds(f.st, 2, 0., 2.);
  check_bounds(f.st, 3, 0., 4.);
  GPRS_CHECK_EQ(result.findings, 2);
}

GPRS_TEST(propagation_infeasible)
{
  // x + y >= 9 with x <= 3 and y <= 4.
  fixture_t f({{0., 3., true}, {0., 4., true}}, {{{{0, 1.}, {1, 1.}}, 9., inf}});
  GPRS_CHECK_EQ(run_propagation(f).outcome, infeasible);

  // Continuous columns only get the records; making them explicit finds the crossing.
  fixture_t g({{0., 3., false}, {0., 4., false}}, {{{{0, 1.}, {1, 1.}}, 9., inf}});
  propagation_t p{g, {}};
  rule_result_t result = p.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(result.records, 2);
  GPRS_CHECK_EQ(g.st.implied_lower[0], 5.);
  GPRS_CHECK_EQ(g.st.implied_lower[1], 6.);
  GPRS_CHECK_EQ(g.st.bound_changes, 0);
  GPRS_CHECK_EQ(p.make_explicit().outcome, infeasible);

  // After make_explicit the visit of the row finds it.
  fixture_t h({{0., 3., false}, {0., 4., false}}, {{{{0, 1.}, {1, 1.}}, -inf, 20.}});
  propagation_t q{h, {}};
  GPRS_CHECK_EQ(q.run().records, 2);
  GPRS_CHECK_EQ(q.make_explicit().findings, 0);
  h.st.set_row_lower(0, 9.);
  GPRS_CHECK_EQ(q.run().outcome, infeasible);
}

// A continuous column gets a record and keeps its explicit bound; a record is replaced when
// the new bound is better by more than 1000 feasibility tolerances, or when it is void.
GPRS_TEST(propagation_continuous_records)
{
  std::vector<col_t> cols = {{0., 5., false}, {0., 10., false}};
  fixture_t f(cols, {{{{0, 1.}, {1, 1.}}, -inf, 8.}, {{{0, 1.}, {1, -1.}}, 1., inf}});
  rule_result_t result = run_propagation(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  // Row 0: x0 <= 8 (weaker than the explicit 5, recorded all the same) and x1 <= 8.
  // Row 1: x0 >= 1 + 0; x1 <= 5 - 1 replaces the record of row 0.
  GPRS_CHECK_EQ(result.records, 4);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(f.st.bound_changes, 0);
  check_bounds(f.st, 0, 0., 5.);
  check_bounds(f.st, 1, 0., 10.);
  GPRS_CHECK(f.st.implied_upper_valid(0));
  GPRS_CHECK_EQ(f.st.implied_upper[0], 8.);
  GPRS_CHECK_EQ(f.st.implied_upper_row[0], 0);
  GPRS_CHECK(f.st.implied_lower_valid(0));
  GPRS_CHECK_EQ(f.st.implied_lower[0], 1.);
  GPRS_CHECK_EQ(f.st.implied_lower_row[0], 1);
  GPRS_CHECK(f.st.implied_upper_valid(1));
  GPRS_CHECK_EQ(f.st.implied_upper[1], 4.);
  GPRS_CHECK_EQ(f.st.implied_upper_row[1], 1);
  GPRS_CHECK(!f.st.implied_lower_valid(1));

  // x1 >= 0.0005 improves the bounds of x0 by half the step: no new record.
  f.st.set_col_lower(1, 0.0005);
  result = run_propagation(f);
  GPRS_CHECK_EQ(result.records, 0);
  GPRS_CHECK_EQ(f.st.implied_upper[0], 8.);
  GPRS_CHECK_EQ(f.st.implied_lower[0], 1.);
  // By twice the step.
  f.st.set_col_lower(1, 0.002);
  result = run_propagation(f);
  GPRS_CHECK_EQ(result.records, 2);
  GPRS_CHECK(std::fabs(f.st.implied_upper[0] - 7.998) < 1e-12);
  GPRS_CHECK(std::fabs(f.st.implied_lower[0] - 1.002) < 1e-12);
  GPRS_CHECK_EQ(f.st.bound_changes, 2);

  // A new side voids the records of row 0; the weaker bound of the changed row is recorded.
  f.st.set_row_upper(0, 9.);
  GPRS_CHECK(!f.st.implied_upper_valid(0));
  result = run_propagation(f);
  GPRS_CHECK_EQ(result.records, 1);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK(f.st.implied_upper_valid(0));
  GPRS_CHECK(std::fabs(f.st.implied_upper[0] - 8.998) < 1e-12);
  // x1 keeps the record of row 1, which is better than the 9 of row 0.
  GPRS_CHECK_EQ(f.st.implied_upper[1], 4.);
  GPRS_CHECK_EQ(f.st.implied_upper_row[1], 1);
  check_bounds(f.st, 0, 0., 5.);
  f.check_consistent();
}

// 1e-6 x + y + w1 - w2 <= 5e-6 with w1 >= 1e20 and w2 <= 1e20: the two large terms cancel in the
// minimum activity and leave an error bound of 6e-12, too much for the coefficient 1e-6 and
// harmless for the coefficient 1.
GPRS_TEST(propagation_error_guard)
{
  std::vector<col_t> cols = {
    {0., 1e9, true}, {0., 10., true}, {1e20, 2e20, false}, {0., 1e20, false}};
  fixture_t f(cols, {{{{0, 1e-6}, {1, 1.}, {2, 1.}, {3, -1.}}, -inf, 5e-6}});
  GPRS_CHECK_EQ(f.st.min_act.inf_count[0], 0);
  GPRS_CHECK_EQ(f.st.min_activity(0), 0.);
  GPRS_CHECK(f.st.min_activity_error(0) > 0.1 * 1e-6 * 1e-6);
  GPRS_CHECK(f.st.min_activity_error(0) < 0.1 * 1e-6);
  rule_result_t result = run_propagation(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  check_bounds(f.st, 0, 0., 1e9);
  GPRS_CHECK(f.st.col_status[1] == col_status_t::fixed);
  check_bounds(f.st, 1, 0., 0.);
}

// No bound from a side b with |b| * 1e-14 above the feasibility tolerance.
GPRS_TEST(propagation_side_guard)
{
  std::vector<col_t> cols = {{0., 3e8, true}, {0., 3e8, true}};
  fixture_t f(cols, {{{{0, 1.}, {1, 1.}}, -inf, 2e8}});
  rule_result_t result = run_propagation(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 0);
  check_bounds(f.st, 0, 0., 3e8);

  fixture_t g(cols, {{{{0, 1.}, {1, 1.}}, -inf, 5e7}});
  GPRS_CHECK_EQ(run_propagation(g).findings, 2);
  check_bounds(g.st, 0, 0., 5e7);
  check_bounds(g.st, 1, 0., 5e7);

  // The guard is per side: the lower side of the same row is small.
  fixture_t h({{-3e8, 1e8, true}, {-3e8, 1e8, true}}, {{{{0, 1.}, {1, 1.}}, -5e7, 2e8}});
  GPRS_CHECK_EQ(run_propagation(h).findings, 2);
  check_bounds(h.st, 0, -1.5e8, 1e8);
  check_bounds(h.st, 1, -1.5e8, 1e8);
}

// Lower bounds 2^54, 2^54 (coefficient -1), 1 and 0: the first and the third share a lane of
// the summation, where 2^54 + 1 is 2^54 in plain doubles. The compensated minimum activity is 1,
// so x3 <= 9; a plain sum gives 0 and x3 <= 10.
GPRS_TEST(propagation_cancellation)
{
  double big              = 0x1p54;
  std::vector<col_t> cols = {
    {big, big + 1024., false}, {0., big, false}, {1., 100., true}, {0., 100., true}};
  fixture_t f(cols, {{{{0, 1.}, {1, -1.}, {2, 1.}, {3, 1.}}, -inf, 10.}});
  GPRS_CHECK_EQ(f.st.min_activity(0), 1.);
  rule_result_t result = run_propagation(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  check_bounds(f.st, 2, 1., 10.);
  check_bounds(f.st, 3, 0., 9.);
}

// 0.01 x0 - 0.9 x1 <= 96.92 with x1 <= 633000170: x0 = 56970024992 meets the row with x1 at its
// bound. The candidate is that number less a rounding error of 1e-5, ten times the feasibility
// tolerance, and rounded as it is would cut the point off.
GPRS_TEST(propagation_large_bound)
{
  std::vector<col_t> cols = {{0., inf, true}, {0., 633000170., true}};
  fixture_t f(cols, {{{{0, 0.01}, {1, -0.9}}, -inf, 96.92}});
  GPRS_CHECK(upper_excess(f.st, 0, {56970024992., 633000170.}) <= 0.);
  GPRS_CHECK(upper_excess(f.st, 0, {56970024993., 633000170.}) > 1e-3);
  rule_result_t result = run_propagation(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 1);
  check_bounds(f.st, 0, 0., 56970024992.);

  // The same row in -x0: a lower bound.
  std::vector<col_t> mirrored = {{-inf, 0., true}, {0., 633000170., true}};
  fixture_t g(mirrored, {{{{0, -0.01}, {1, -0.9}}, -inf, 96.92}});
  GPRS_CHECK_EQ(run_propagation(g).findings, 1);
  check_bounds(g.st, 0, -56970024992., 0.);

  // At a usual size the allowance changes nothing: 299.9999995 is 300 within the tolerance and
  // 299.999 is not.
  std::vector<col_t> small = {{0., 1000., true}, {0., 1000., true}};
  fixture_t h(small, {{{{0, 0.01}}, -inf, 2.999999995}, {{{1, 0.01}}, -inf, 2.99999}});
  GPRS_CHECK_EQ(run_propagation(h).findings, 2);
  check_bounds(h.st, 0, 0., 300.);
  check_bounds(h.st, 1, 0., 299.);
}

GPRS_TEST(propagation_make_explicit)
{
  std::vector<col_t> cols = {
    {0., 100., false}, {0., 10., false}, {0., 10., true}, {0., 100., false}, {0., 5., false}};
  std::vector<row_t> rows = {
    {{{0, 1.}, {1, 1.}}, -inf, 8.}, {{{2, 1.}, {3, 1.}}, 15., inf}, {{{1, 1.}, {4, 1.}}, -inf, 9.}};
  fixture_t f(cols, rows);
  propagation_t p{f, {}};
  GPRS_CHECK_EQ(p.run().outcome, ok);
  GPRS_CHECK_EQ(f.st.bound_changes, 0);
  f.only(row_list::row_rules, {});
  state_t before       = f.st;
  rule_result_t result = p.make_explicit();
  GPRS_CHECK_EQ(result.outcome, ok);
  // x0 <= 8 and x1 <= 8 from row 0, x3 >= 5 from row 1; x4 <= 9 of row 2 is weaker than 5.
  GPRS_CHECK_EQ(result.findings, 3);
  GPRS_CHECK_EQ(result.records, 0);
  check_bounds(f.st, 0, 0., 8.);
  check_bounds(f.st, 1, 0., 8.);
  check_bounds(f.st, 2, 0., 10.);
  check_bounds(f.st, 3, 5., 100.);
  check_bounds(f.st, 4, 0., 5.);
  check_same_set(before, f.st, result.outcome);
  // No record of these columns was ever void: only the rows that a new bound reaches are
  // listed. x0 <= 8 and x1 <= 8 are in the maximum activity of rows 0 and 2, which have no
  // lower side; x3 >= 5 is in the minimum activity of row 1, which has no upper side.
  GPRS_CHECK(f.st.row_frontier.empty(row_list::propagation));
  GPRS_CHECK(f.st.row_frontier.list[row_list::row_rules] == std::vector<int32_t>({0, 2, 1}));

  // A second call has no mark left.
  result = p.make_explicit();
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(f.st.bound_changes, 3);
}

// After make_explicit a continuous column gets its bounds as an integer one does, so that one
// call follows a chain of bounds: x1 <= x0 + 1, x2 <= x1 + 1, x3 <= x2 + 1 with x0 <= 5.
GPRS_TEST(propagation_explicit_chain)
{
  std::vector<col_t> cols = {{0., 5., false}, {0., inf, false}, {0., inf, false}, {0., inf, false}};
  std::vector<row_t> rows = {{{{2, -1.}, {3, 1.}}, -inf, 1.},
                             {{{1, -1.}, {2, 1.}}, -inf, 1.},
                             {{{0, -1.}, {1, 1.}}, -inf, 1.}};
  fixture_t f(cols, rows);
  propagation_t p{f, {}};
  rule_result_t result = p.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  // Only x1 <= 6 can be derived while the bounds of x1 and x2 are infinite.
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(f.st.bound_changes, 0);
  GPRS_CHECK_EQ(f.st.implied_upper[1], 6.);
  GPRS_CHECK(!f.st.implied_upper_valid(2));

  state_t before = f.st;
  result         = p.make_explicit();
  GPRS_CHECK_EQ(result.findings, 1);
  check_bounds(f.st, 1, 0., 6.);
  check_bounds(f.st, 2, 0., inf);
  // Row 1 reads the new bound of x1 with its upper side.
  GPRS_CHECK(f.st.row_frontier.list[row_list::propagation] == std::vector<int32_t>({1}));

  result = p.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 2);
  check_bounds(f.st, 2, 0., 7.);
  check_bounds(f.st, 3, 0., 8.);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::propagation));
  check_same_set(before, f.st, result.outcome);
  // The bounds are recorded with their rows as well.
  GPRS_CHECK_EQ(f.st.implied_upper[3], 8.);
  GPRS_CHECK_EQ(f.st.implied_upper_row[3], 0);
}

// An explicit bound is the candidate of a row as the row is when the bound is written. A record
// whose row was deleted or changed is not used, whatever it says.
GPRS_TEST(propagation_explicit_bound_after_the_source_row_changed)
{
  std::vector<col_t> cols = {{0., 100., false}, {0., 10., false}, {0., 100., false}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}}, -inf, 8.}, {{{1, 1.}, {2, 1.}}, -inf, 30.}};
  {
    // The source row gets a weaker side: x0 <= 20, not the recorded 8.
    fixture_t f(cols, rows);
    propagation_t p{f, {}};
    GPRS_CHECK_EQ(p.run().records, 3);
    GPRS_CHECK_EQ(f.st.implied_upper[0], 8.);
    f.st.set_row_upper(0, 20.);
    GPRS_CHECK(!f.st.implied_upper_valid(0));
    state_t before       = f.st;
    rule_result_t result = p.make_explicit();
    GPRS_CHECK_EQ(result.outcome, ok);
    // x2 <= 30 of the unchanged row 1 is taken; x0 waits for the visit of its row.
    GPRS_CHECK_EQ(result.findings, 1);
    check_bounds(f.st, 0, 0., 100.);
    check_bounds(f.st, 2, 0., 30.);
    result = p.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    check_bounds(f.st, 0, 0., 20.);
    check_bounds(f.st, 1, 0., 10.);
    check_same_set(before, f.st, result.outcome);
  }
  {
    // The source row is deleted and no other row implies the bound: x0 keeps its bound.
    fixture_t f(cols, rows);
    propagation_t p{f, {}};
    GPRS_CHECK_EQ(p.run().records, 3);
    f.st.delete_row(0);
    state_t before       = f.st;
    rule_result_t result = p.make_explicit();
    GPRS_CHECK_EQ(result.findings, 1);
    result = p.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 0);
    check_bounds(f.st, 0, 0., 100.);
    check_bounds(f.st, 1, 0., 10.);
    check_bounds(f.st, 2, 0., 30.);
    check_same_set(before, f.st, result.outcome);
  }
  {
    // The source row is scaled: the same constraint, and the bound comes from the scaled row.
    fixture_t f(cols, rows);
    propagation_t p{f, {}};
    GPRS_CHECK_EQ(p.run().records, 3);
    f.st.scale_row(0, 4., false);
    GPRS_CHECK(!f.st.implied_upper_valid(0));
    state_t before = f.st;
    GPRS_CHECK_EQ(p.make_explicit().findings, 1);
    rule_result_t result = p.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    check_bounds(f.st, 0, 0., 8.);
    check_bounds(f.st, 1, 0., 8.);
    check_same_set(before, f.st, result.outcome);
  }
}

// A record that was lost and replaced by a weaker one: row 0 gave x0 <= 8, row 1 gives 8.0005,
// which is not recorded next to it. Row 0 is deleted and the visit of row 2 records x0 <= 50.
// make_explicit takes the 50 and lists the rows of x0, because a visit found the record void:
// row 1 then gives its bound.
GPRS_TEST(propagation_explicit_bound_after_a_lost_record)
{
  std::vector<col_t> cols = {{0., 100., false}, {0., 10., false}, {0., 10., true}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}}, -inf, 8.},
                             {{{0, 1.}, {1, 1.}}, -inf, 8.0005},
                             {{{0, 1.}, {2, 1.}}, -inf, 50.}};
  fixture_t f(cols, rows);
  propagation_t p{f, {}};
  GPRS_CHECK_EQ(p.run().outcome, ok);
  GPRS_CHECK_EQ(f.st.implied_upper[0], 8.);
  GPRS_CHECK_EQ(f.st.implied_upper_row[0], 0);
  f.st.delete_row(0);
  f.only(row_list::propagation, {2});
  GPRS_CHECK_EQ(p.run().records, 1);
  GPRS_CHECK_EQ(f.st.implied_upper[0], 50.);

  state_t before       = f.st;
  rule_result_t result = p.make_explicit();
  GPRS_CHECK_EQ(result.findings, 1);
  check_bounds(f.st, 0, 0., 50.);
  GPRS_CHECK(f.st.row_frontier.list[row_list::propagation] == std::vector<int32_t>({1, 2}));
  result = p.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  check_bounds(f.st, 0, 0., 8.0005);
  check_same_set(before, f.st, result.outcome);
}

// The step: a bound of a continuous column is improved by more than 1000 feasibility
// tolerances, or by more than as many times what a bound of its size loses when it is stored.
GPRS_TEST(propagation_explicit_step)
{
  {
    std::vector<col_t> cols = {{0., 5., false}, {0., 5., false}};
    fixture_t f(cols, {{{{0, 1.}}, -inf, 4.9995}, {{{1, 1.}}, -inf, 4.998}});
    propagation_t p{f, {}};
    GPRS_CHECK_EQ(p.run().records, 2);
    GPRS_CHECK_EQ(p.make_explicit().findings, 1);
    check_bounds(f.st, 0, 0., 5.);
    check_bounds(f.st, 1, 0., 4.998);
  }
  {
    // 1e-11 of 4e15 is 4e4: 30000 below the bound does not count, 50000 does. A bound of this
    // size is moved outward by four rounding units of its size, 1.8, which is 2 here.
    double big              = 4e15;
    double a                = 0x1p-30;
    std::vector<col_t> cols = {{0., big, false}, {0., big, false}};
    fixture_t f(cols, {{{{0, a}}, -inf, a * (big - 30000.)}, {{{1, a}}, -inf, a * (big - 50000.)}});
    propagation_t p{f, {}};
    GPRS_CHECK_EQ(p.run().outcome, ok);
    GPRS_CHECK_EQ(f.st.implied_upper[1], big - 50000.);
    GPRS_CHECK_EQ(p.make_explicit().findings, 1);
    check_bounds(f.st, 0, 0., big);
    check_bounds(f.st, 1, 0., big - 50000. + 2.);
  }
}

// Two bounds that depend on each other with a factor below 1 improve in ever smaller steps:
// x <= 0.999 y + 1 and y <= 0.999 x + 1 from 1e6 down to 1000. Each bound gets its steps and
// stays; the call reports it.
GPRS_TEST(propagation_explicit_steps_are_limited)
{
  std::vector<col_t> cols = {{0., 1e6, false}, {0., 1e6, false}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, -0.999}}, -inf, 1.},
                             {{{0, -0.999}, {1, 1.}}, -inf, 1.}};
  fixture_t f(cols, rows);
  propagation_t p{f, {}};
  GPRS_CHECK_EQ(p.run().outcome, ok);
  state_t before       = f.st;
  rule_result_t result = p.make_explicit();
  GPRS_CHECK(!result.limited);
  result = p.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK(result.limited);
  GPRS_CHECK(!result.stopped);
  // One step from the record and sixteen from the visits, per bound.
  GPRS_CHECK_EQ(result.findings, 32);
  GPRS_CHECK_EQ(f.st.bound_changes, 34);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::propagation));
  GPRS_CHECK(f.st.col_upper[0] > 9.6e5 && f.st.col_upper[0] < 9.7e5);
  check_same_set(before, f.st, result.outcome);
  // A further call has nothing listed and reports nothing.
  result = p.run();
  GPRS_CHECK(!result.limited);
  GPRS_CHECK_EQ(result.findings, 0);
}

// The error bound of an activity counts the largest term the sum ever held. x1 had the upper
// bound 1.024e25, a term of 7.8e19 in the maximum activity; with the bound at 0 that term is
// gone, but its size still made the error guard drop the candidate of x1, whose coefficient is
// small. The activity is summed again and the row gives x1 >= -0.046875 / 7.62939453125e-06.
GPRS_TEST(propagation_error_guard_after_a_huge_term_left)
{
  double a                = 7.62939453125e-06;
  std::vector<col_t> cols = {
    {0., 2., true}, {-inf, 1.024e25, false}, {0., 3., true}, {0., 1., true}};
  fixture_t f(cols, {{{{0, 0.01171875}, {1, a}, {2, 0.00390625}, {3, 0.01171875}}, 0., inf}});
  GPRS_CHECK_EQ(f.st.max_act.inf_count[0], 0);
  GPRS_CHECK(f.st.max_act.max_term[0] > 7e19);
  f.st.set_col_upper(1, 0.);
  GPRS_CHECK(f.st.max_act.max_term[0] > 7e19);
  GPRS_CHECK(!activity_fresh(f.st, f.st.max_act, 0));
  GPRS_CHECK(f.st.max_activity_error(0) > 0.1 * 1e-6 * a);

  rule_result_t result = run_propagation(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.records, 1);
  GPRS_CHECK(activity_fresh(f.st, f.st.max_act, 0));
  GPRS_CHECK(f.st.max_act.max_term[0] < 1.);
  GPRS_CHECK(f.st.implied_lower_valid(1));
  GPRS_CHECK_EQ(f.st.implied_lower[1], -0.046875 / a);
  f.check_consistent();
}

GPRS_TEST(propagation_budget_stop)
{
  // No finding in any row; a visit costs line_visit_work and the two entries.
  std::vector<row_t> rows(4, row_t{{{0, 1.}, {1, 1.}}, -inf, 30.});
  fixture_t f({{0., 10., true}, {0., 10., true}}, rows);
  f.budget.limit       = line_visit_work + 5;
  rule_result_t result = run_propagation(f);
  GPRS_CHECK(result.stopped);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(f.budget.used, 2 * (line_visit_work + 2));
  GPRS_CHECK(f.st.row_frontier.list[row_list::propagation] == std::vector<int32_t>({2, 3}));

  f.budget = work_budget_t{unlimited, unlimited, 0, 0};
  f.st.delete_row(3);
  result = run_propagation(f);
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK_EQ(f.budget.used, line_visit_work + 2);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::propagation));
}

GPRS_TEST(propagation_property_feasible_set)
{
  int64_t findings = 0, infeasible_models = 0, compared = 0, records = 0, made_explicit = 0;
  int64_t explicit_findings = 0;
  std::vector<col_t> cols;
  std::vector<row_t> rows;
  for (uint64_t seed = 1; seed <= 1000; ++seed) {
    rng_t rng(seed);
    random_options_t options;
    options.integer_share = seed % 3 == 0 ? 1. : 0.6;
    random_model(rng, options, cols, rows);
    fixture_t f(cols, rows);
    for (int32_t j = 0; j < f.st.ncols; ++j) {
      if (cols[j].lower > -inf && rng.chance(0.15)) { fix_column(f.st, f.ps, j, cols[j].lower); }
    }
    f.budget.limit = 2000000;
    propagation_t p{f, {}};
    state_t before       = f.st;
    naive_t naive        = naive_propagation(before, f.opt.tol.feasibility);
    rule_result_t result = p.run();
    check_same_set(before, f.st, result.outcome);
    bool comparable = (naive.converged || naive.infeasible) && naive.small();
    if (comparable) {
      GPRS_CHECK(!result.stopped);
      GPRS_CHECK_EQ(result.outcome == infeasible, naive.infeasible);
      ++compared;
    }
    if (result.outcome != ok) {
      ++infeasible_models;
      continue;
    }
    if (comparable) {
      GPRS_CHECK(f.st.col_lower == naive.lower);
      GPRS_CHECK(f.st.col_upper == naive.upper);
      GPRS_CHECK(f.st.row_frontier.empty(row_list::propagation));
    }
    findings += result.findings;
    records += result.records;

    // Every valid record holds at every feasible point.
    int64_t broken = 0, valid = 0;
    for_each_point(make_grid(f.st), [&](const std::vector<double>& x) {
      if (!feasible(f.st, x)) { return; }
      for (int32_t j = 0; j < f.st.ncols; ++j) {
        broken += f.st.implied_lower_valid(j) && x[j] < f.st.implied_lower[j] - 1e-9;
        broken += f.st.implied_upper_valid(j) && x[j] > f.st.implied_upper[j] + 1e-9;
      }
    });
    GPRS_CHECK_EQ(broken, 0);
    for (int32_t j = 0; j < f.st.ncols; ++j) {
      valid += f.st.implied_lower_valid(j) + f.st.implied_upper_valid(j);
      GPRS_CHECK(!f.st.is_integer(j) || f.st.implied_lower_row[j] < 0);
      GPRS_CHECK(!f.st.is_integer(j) || f.st.implied_upper_row[j] < 0);
    }
    GPRS_CHECK(valid <= result.records);

    // The bounds become explicit, and the call after it keeps the feasible set as well.
    state_t recorded   = f.st;
    rule_result_t made = p.make_explicit();
    made_explicit += made.outcome == ok ? made.findings : 0;
    check_same_set(recorded, f.st, made.outcome);
    if (made.outcome != ok) { continue; }
    rule_result_t after = p.run();
    check_same_set(recorded, f.st, after.outcome);
    explicit_findings += after.outcome == ok ? after.findings : 0;
    if (after.outcome != ok || after.stopped || after.limited) { continue; }
    // No row has a bound left to give: a second call, on every row, finds nothing.
    for (int32_t i = 0; i < f.st.nrows; ++i) {
      if (f.st.row_alive[i]) { f.st.row_frontier.push(list_bit(row_list::propagation), i); }
    }
    int64_t changes = f.st.bound_changes;
    GPRS_CHECK_EQ(p.run().findings, 0);
    GPRS_CHECK_EQ(f.st.bound_changes, changes);
  }
  GPRS_CHECK(findings > 500);
  GPRS_CHECK(infeasible_models > 100);
  GPRS_CHECK(compared > 900);
  GPRS_CHECK(records > 400);
  GPRS_CHECK(made_explicit > 150);
  GPRS_CHECK(explicit_findings > 20);
}
