/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "test_engine_support.hpp"

#include <gprs/presolve.hpp>

#include <string>
#include <vector>

using namespace gprs;
using namespace gprs::engine_test;

namespace {

constexpr rule_outcome_t ok        = rule_outcome_t::ok;
constexpr rule_outcome_t unbounded = rule_outcome_t::unbounded_or_infeasible;

using list_t = std::vector<int32_t>;

work_budget_t large_budget()
{
  work_budget_t budget;
  budget.limit       = 1 << 30;
  budget.stall_limit = 1 << 30;
  return budget;
}

struct fixture_t {
  state_t st;
  postsolve_t ps;
  dual_fixing_t family;
  options_t options;

  explicit fixture_t(const model_t& m) : st(m, tolerances_t()) { ps.init(m); }

  rule_result_t run()
  {
    work_budget_t budget = large_budget();
    return run(budget);
  }

  rule_result_t run(work_budget_t& budget)
  {
    rule_result_t result = family.run(st, budget, ps, options);
    if (result.outcome == ok) { GPRS_CHECK_EQ(st.consistency_error(), std::string()); }
    return result;
  }
};

enum class side_t { none, le, ge, eq, range, free };
enum class expect_t { fixed, left, unbounded };

struct case_t {
  double cost;
  double a;  // coefficient of the column in its row; with side_t::none the column is empty
  side_t side;
  double lower;
  double upper;
  expect_t expect;
  double value;
};

// Column 0 is the column of the case. Column 1, in [0, 1] with zero cost, shares row 0 with it
// and is locked in both directions by the range row 1, so the rule must leave it.
model_t case_model(const case_t& c, uint8_t type)
{
  double a     = c.side == side_t::none ? 0. : c.a;
  double lower = -inf;
  double upper = inf;
  if (c.side == side_t::le || c.side == side_t::eq || c.side == side_t::range) { upper = 4.; }
  if (c.side == side_t::ge || c.side == side_t::eq) { lower = 4.; }
  if (c.side == side_t::range) { lower = 1.; }
  if (c.side == side_t::none) { upper = 4.; }
  return make_model({{a, 1.}, {0., 1.}},
                    {lower, 0.},
                    {upper, 1.},
                    {c.lower, 0.},
                    {c.upper, 1.},
                    {c.cost, 0.},
                    {type, 0});
}

void check_case(const case_t& c, uint8_t type, int32_t line)
{
  model_t m = case_model(c, type);
  GPRS_REQUIRE(well_formed(m));
  fixture_t f(m);
  rule_result_t result = f.run();
  bool as_expected     = false;
  if (c.expect == expect_t::unbounded) {
    as_expected = result.outcome == unbounded;
  } else if (c.expect == expect_t::left) {
    as_expected = result.outcome == ok && result.findings == 0 && !result.stopped &&
                  f.st.is_live(0) && f.st.col_lower[0] == c.lower && f.st.col_upper[0] == c.upper &&
                  f.ps.order.empty() && dd_value(f.st.objective_offset) == 0.;
  } else {
    as_expected = result.outcome == ok && result.findings == 1 && !result.stopped &&
                  f.st.col_status[0] == col_status_t::fixed && f.st.col_lower[0] == c.value &&
                  f.st.col_upper[0] == c.value && f.ps.order.size() == 1 &&
                  f.ps.fixed_cols.size() == 1 && f.ps.fixed_cols[0].col == 0 &&
                  f.ps.fixed_cols[0].value == c.value &&
                  dd_value(f.st.objective_offset) == c.cost * c.value;
  }
  if (result.outcome == ok) {
    as_expected = as_expected && f.st.is_live(1) && f.st.col_lower[1] == 0. &&
                  f.st.col_upper[1] == 1. && f.st.col_frontier.empty(col_list::dual_fixing);
  }
  if (!as_expected) {
    gprs::test::fail(
      __FILE__,
      line,
      "case with cost " + std::to_string(c.cost) + ", coefficient " + std::to_string(c.a) +
        ", bounds [" + std::to_string(c.lower) + ", " + std::to_string(c.upper) + "], type " +
        std::to_string(type) + ": outcome " + std::to_string((int)result.outcome) + ", findings " +
        std::to_string(result.findings) + ", column 0 in [" + std::to_string(f.st.col_lower[0]) +
        ", " + std::to_string(f.st.col_upper[0]) + "]");
  }
}

void check_cases(const std::vector<case_t>& cases, int32_t line)
{
  for (const case_t& c : cases) {
    for (uint8_t type : {0, 1}) {
      check_case(c, type, line);
    }
  }
}

constexpr expect_t fixed = expect_t::fixed;
constexpr expect_t left  = expect_t::left;

}  // namespace

// The sign of the cost and the lock that the row puts on the column decide; a negative
// coefficient swaps the lock.
GPRS_TEST(dual_fixing_cost_sign_against_lock)
{
  check_cases({{1., 1., side_t::le, 2., 5., fixed, 2.},
               {1., 1., side_t::ge, 2., 5., left, 0.},
               {1., -1., side_t::le, 2., 5., left, 0.},
               {1., -1., side_t::ge, 2., 5., fixed, 2.},
               {-1., 1., side_t::ge, 2., 5., fixed, 5.},
               {-1., 1., side_t::le, 2., 5., left, 0.},
               {-1., -1., side_t::le, 2., 5., fixed, 5.},
               {-1., -1., side_t::ge, 2., 5., left, 0.},
               {3., 2., side_t::le, -4., -1., fixed, -4.},
               {-3., 2., side_t::ge, -4., -1., fixed, -1.}},
              __LINE__);
}

// A cost of any size counts with its sign.
GPRS_TEST(dual_fixing_tiny_cost_has_a_sign)
{
  check_cases({{1e-12, 1., side_t::le, 2., 5., fixed, 2.},
               {1e-12, 1., side_t::ge, 2., 5., left, 0.},
               {-1e-12, 1., side_t::ge, 2., 5., fixed, 5.},
               {-1e-12, 1., side_t::le, 2., 5., left, 0.}},
              __LINE__);
}

// An equation and a range lock in both directions, whatever the cost.
GPRS_TEST(dual_fixing_two_sided_rows_lock_both_ways)
{
  check_cases({{1., 1., side_t::eq, 2., 5., left, 0.},
               {-1., 1., side_t::eq, 2., 5., left, 0.},
               {0., 1., side_t::eq, 2., 5., left, 0.},
               {1., -1., side_t::range, 2., 5., left, 0.},
               {-1., -1., side_t::range, 2., 5., left, 0.},
               {0., -1., side_t::range, 2., 5., left, 0.}},
              __LINE__);
}

// Zero cost: the column moves away from its only lock; without a lock it takes the lower bound
// if that is finite, else the upper bound, else 0.
GPRS_TEST(dual_fixing_zero_cost)
{
  check_cases({{0., 1., side_t::le, 2., 5., fixed, 2.},
               {0., 1., side_t::ge, 2., 5., fixed, 5.},
               {0., -1., side_t::le, 2., 5., fixed, 5.},
               {0., -1., side_t::ge, 2., 5., fixed, 2.},
               {0., 1., side_t::none, 2., 5., fixed, 2.},
               {0., 1., side_t::none, -inf, 5., fixed, 5.},
               {0., 1., side_t::none, 2., inf, fixed, 2.},
               {0., 1., side_t::none, -inf, inf, fixed, 0.},
               {0., 1., side_t::free, 2., 5., fixed, 2.},
               {0., -1., side_t::free, -inf, 5., fixed, 5.},
               {0., 1., side_t::free, -inf, inf, fixed, 0.}},
              __LINE__);
}

// A column without entries, or with entries in free rows only, has no lock: the cost decides.
GPRS_TEST(dual_fixing_column_without_locks)
{
  check_cases({{1., 1., side_t::none, 2., 5., fixed, 2.},
               {-1., 1., side_t::none, 2., 5., fixed, 5.},
               {1., 1., side_t::free, 2., 5., fixed, 2.},
               {-1., -1., side_t::free, 2., 5., fixed, 5.},
               {1., 1., side_t::none, 2., inf, fixed, 2.},
               {-1., 1., side_t::none, -inf, 5., fixed, 5.}},
              __LINE__);
}

// An infinite target bound with a cost that pushes towards it: unbounded or infeasible, also
// when the other bound is finite.
GPRS_TEST(dual_fixing_infinite_target_with_cost)
{
  check_cases({{1., 1., side_t::le, -inf, 5., expect_t::unbounded, 0.},
               {1., -1., side_t::ge, -inf, 5., expect_t::unbounded, 0.},
               {-1., 1., side_t::ge, 2., inf, expect_t::unbounded, 0.},
               {-1., -1., side_t::le, 2., inf, expect_t::unbounded, 0.},
               {1., 1., side_t::none, -inf, 5., expect_t::unbounded, 0.},
               {-1., 1., side_t::none, 2., inf, expect_t::unbounded, 0.},
               {1., 1., side_t::free, -inf, inf, expect_t::unbounded, 0.},
               {-1., 1., side_t::free, -inf, inf, expect_t::unbounded, 0.},
               {1e-12, 1., side_t::le, -inf, 5., expect_t::unbounded, 0.},
               {-1e-12, 1., side_t::ge, 2., inf, expect_t::unbounded, 0.}},
              __LINE__);
}

// No status and no fixing when the column is locked in the direction of its cost, when the
// infinite bound is not the target, or when the cost is zero and the other direction is locked.
GPRS_TEST(dual_fixing_infinite_bound_without_a_conclusion)
{
  check_cases({{1., 1., side_t::ge, -inf, 5., left, 0.},
               {-1., 1., side_t::le, 2., inf, left, 0.},
               {1., 1., side_t::eq, -inf, inf, left, 0.},
               {-1., 1., side_t::range, -inf, inf, left, 0.},
               {0., 1., side_t::le, -inf, 5., left, 0.},
               {0., 1., side_t::ge, 2., inf, left, 0.},
               {0., -1., side_t::le, 2., inf, left, 0.},
               {0., 1., side_t::le, -inf, inf, left, 0.},
               {1., 1., side_t::le, 2., inf, fixed, 2.},
               {-1., 1., side_t::ge, -inf, 5., fixed, 5.}},
              __LINE__);
}

// A huge finite bound is a value, not infinity: the column is fixed there and its term counts
// as infinite in the activities of the row.
GPRS_TEST(dual_fixing_huge_finite_bound_is_a_value)
{
  check_cases(
    {{1., 1., side_t::le, -1e30, 5., fixed, -1e30}, {-1., 1., side_t::ge, 2., 1e30, fixed, 1e30}},
    __LINE__);
  fixture_t f(case_model({1., 1., side_t::le, -1e30, 5., fixed, -1e30}, 0));
  f.run();
  GPRS_CHECK_EQ(f.st.min_act.inf_count[0], 1);
  GPRS_CHECK_EQ(f.st.max_act.inf_count[0], 1);
}

// Three columns in one <= row and one free row, a fourth without entries.
//   r0:  x0 + 2 x1 - x2 <= 6
//   r1:  x0 - x1 + x2 free
// x0 in [1, 4] cost 2 (no down lock: to 1), x1 in [0, 3] cost -1 (up lock: left),
// x2 in [0, 2] cost -3 (a < 0 in <=: no up lock: to 2), x3 in [-2, 7] cost 0 (to -2).
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

GPRS_TEST(dual_fixing_records_offset_and_rows)
{
  model_t m = four_columns();
  GPRS_REQUIRE(well_formed(m));
  fixture_t f(m);
  list_t rows;
  for (int32_t l = 0; l < row_list::compaction; ++l) {
    f.st.row_frontier.take(l, rows);
  }
  work_budget_t budget = large_budget();
  rule_result_t result = f.run(budget);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 3);
  GPRS_CHECK(!result.stopped);

  // In list order, each with its value.
  GPRS_REQUIRE(f.ps.fixed_cols.size() == 3 && f.ps.order.size() == 3);
  GPRS_CHECK(f.ps.aliases.empty() && f.ps.parallel_rows.empty());
  GPRS_CHECK_EQ(f.ps.fixed_cols[0].col, 0);
  GPRS_CHECK_EQ(f.ps.fixed_cols[0].value, 1.);
  GPRS_CHECK_EQ(f.ps.fixed_cols[1].col, 2);
  GPRS_CHECK_EQ(f.ps.fixed_cols[1].value, 2.);
  GPRS_CHECK_EQ(f.ps.fixed_cols[2].col, 3);
  GPRS_CHECK_EQ(f.ps.fixed_cols[2].value, -2.);

  GPRS_CHECK_EQ(f.st.col_status[0], col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_status[1], col_status_t::live);
  GPRS_CHECK_EQ(f.st.col_status[2], col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_status[3], col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_lower[1], 0.);
  GPRS_CHECK_EQ(f.st.col_upper[1], 3.);
  GPRS_CHECK_EQ(f.st.live_cols, 1);
  GPRS_CHECK_EQ(f.st.row_free_size[0], 1);
  GPRS_CHECK_EQ(f.st.row_free_size[1], 1);
  // 10 + 2 * 1 - 3 * 2 + 0 * -2
  GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 6.);
  // Activity of r0 over x1 in [0, 3] with x0 = 1 and x2 = 2: 1 - 2 + 2 * [0, 3].
  GPRS_CHECK_EQ(f.st.min_activity(0), -1.);
  GPRS_CHECK_EQ(f.st.max_activity(0), 5.);

  // The rows of the fixed columns are back on the lists of the row families.
  for (int32_t l = 0; l < row_list::compaction; ++l) {
    GPRS_CHECK(f.st.row_frontier.list[l] == (list_t{0, 1}));
  }
  GPRS_CHECK(f.st.col_frontier.empty(col_list::dual_fixing));

  // One visit per live column and the entries of the three fixed columns.
  GPRS_CHECK_EQ(budget.used, 4 * line_visit_work + 2 + 2 + 0);
  GPRS_CHECK_EQ(budget.used_at_finding, budget.used);
}

// Only the columns on the list are looked at, and the list is empty afterwards.
GPRS_TEST(dual_fixing_takes_its_list_only)
{
  fixture_t f(four_columns());
  list_t cols;
  f.st.col_frontier.take(col_list::dual_fixing, cols);
  GPRS_CHECK(cols == (list_t{0, 1, 2, 3}));
  rule_result_t none = f.run();
  GPRS_CHECK_EQ(none.findings, 0);
  GPRS_CHECK_EQ(f.st.live_cols, 4);

  f.st.col_frontier.push(col_event::locks, 2);
  rule_result_t one = f.run();
  GPRS_CHECK_EQ(one.findings, 1);
  GPRS_CHECK_EQ(f.st.col_status[2], col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.live_cols, 3);
  GPRS_CHECK(f.st.col_frontier.empty(col_list::dual_fixing));
}

// A column that was fixed or removed while it waited on the list is skipped: no second record,
// no work.
GPRS_TEST(dual_fixing_skips_columns_that_are_not_live)
{
  fixture_t f(four_columns());
  fix_column(f.st, f.ps, 0, 3.);
  f.st.remove_col(3);
  work_budget_t budget = large_budget();
  rule_result_t result = f.run(budget);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_REQUIRE(f.ps.fixed_cols.size() == 2);
  GPRS_CHECK_EQ(f.ps.fixed_cols[0].col, 0);
  GPRS_CHECK_EQ(f.ps.fixed_cols[0].value, 3.);
  GPRS_CHECK_EQ(f.ps.fixed_cols[1].col, 2);
  GPRS_CHECK_EQ(f.st.col_lower[0], 3.);
  GPRS_CHECK_EQ(f.st.col_status[3], col_status_t::removed);
  GPRS_CHECK_EQ(budget.used, 2 * line_visit_work + 2);
}

// The locks are those of the state at the call: a row that is deleted releases its columns,
// which come back on the list.
GPRS_TEST(dual_fixing_follows_released_locks)
{
  //   r0:  x0 + x1 >= 2      r1:  x0 - x1 <= 3
  // x0, x1 in [0, 5], cost 1 and 1: x0 is locked both ways, x1 down by r0 and by r1.
  model_t m =
    make_model({{1., 1.}, {1., -1.}}, {2., -inf}, {inf, 3.}, {0., 0.}, {5., 5.}, {1., 1.}, {0, 1});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 0);
  GPRS_CHECK_EQ(f.st.live_cols, 2);

  f.st.delete_row(0);
  // x0: up lock of r1 only, cost 1: to its lower bound. x1: down lock of r1: stays.
  rule_result_t result = f.run();
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_CHECK_EQ(f.st.col_status[0], col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_lower[0], 0.);
  GPRS_CHECK(f.st.is_live(1));

  f.st.delete_row(1);
  result = f.run();
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_CHECK_EQ(f.st.col_status[1], col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_lower[1], 0.);
}

// A cost that changes sign changes the direction (set_cost puts the column on the list).
GPRS_TEST(dual_fixing_follows_the_cost)
{
  // x0 + x1 <= 4: x0 has an up lock only.
  model_t m =
    make_model({{1., 1.}, {0., 1.}}, {-inf, 0.}, {4., 1.}, {2., 0.}, {5., 1.}, {-1., 0.}, {0, 0});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 0);
  f.st.set_cost(0, 1.);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK_EQ(f.st.col_lower[0], 2.);
  GPRS_CHECK_EQ(f.st.col_upper[0], 2.);
}

namespace {

// n columns without entries in [0, 1]; cost 1 where fixable[j] (to the lower bound), else the
// column is locked in both directions by its own equation row.
model_t budget_model(const std::vector<uint8_t>& fixable)
{
  int32_t n = fixable.size();
  dense_t rows;
  reals_t row_lower, row_upper;
  for (int32_t j = 0; j < n; ++j) {
    if (fixable[j]) { continue; }
    rows.push_back(reals_t(n, 0.));
    rows.back()[j] = 1.;
    row_lower.push_back(0.);
    row_upper.push_back(1.);
  }
  return make_model(rows,
                    row_lower,
                    row_upper,
                    reals_t(n, 0.),
                    reals_t(n, 1.),
                    reals_t(n, 1.),
                    std::vector<uint8_t>(n, 0));
}

}  // namespace

// The budget is tested between two columns; what is left goes back on the list in its order
// and the next call goes on there.
GPRS_TEST(dual_fixing_stops_on_the_budget)
{
  fixture_t f(budget_model({1, 1, 1, 1, 1, 1}));
  work_budget_t budget;
  budget.limit         = 2 * line_visit_work;
  budget.stall_limit   = 1 << 30;
  rule_result_t result = f.run(budget);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK(result.stopped);
  GPRS_CHECK_EQ(result.findings, 2);
  GPRS_CHECK_EQ(budget.used, 2 * line_visit_work);
  GPRS_CHECK(f.st.col_frontier.list[col_list::dual_fixing] == (list_t{2, 3, 4, 5}));
  GPRS_CHECK_EQ(f.st.col_status[1], col_status_t::fixed);
  GPRS_CHECK(f.st.is_live(2));

  // One unit below the limit is one more column.
  budget             = work_budget_t();
  budget.limit       = 2 * line_visit_work + 1;
  budget.stall_limit = 1 << 30;
  result             = f.run(budget);
  GPRS_CHECK(result.stopped);
  GPRS_CHECK_EQ(result.findings, 3);
  GPRS_CHECK(f.st.col_frontier.list[col_list::dual_fixing] == (list_t{5}));

  result = f.run();
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_CHECK_EQ(f.st.live_cols, 0);
  GPRS_CHECK(f.st.col_frontier.empty(col_list::dual_fixing));

  // A spent budget and an empty list: nothing is left, so the call did not stop.
  budget = work_budget_t();
  result = f.run(budget);
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK_EQ(result.findings, 0);
}

// Without a budget nothing is looked at and the whole list goes back.
GPRS_TEST(dual_fixing_with_a_spent_budget_changes_nothing)
{
  fixture_t f(budget_model({1, 1, 1}));
  work_budget_t budget;
  rule_result_t result = f.run(budget);
  GPRS_CHECK(result.stopped);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(budget.used, 0);
  GPRS_CHECK_EQ(f.st.live_cols, 3);
  GPRS_CHECK(f.ps.order.empty());
  GPRS_CHECK(f.st.col_frontier.list[col_list::dual_fixing] == (list_t{0, 1, 2}));
}

// The stall limit counts from the last finding.
GPRS_TEST(dual_fixing_stall_limit_counts_from_the_last_finding)
{
  // Column 0 is fixed after 8 units; columns 1 and 2 give nothing, 16 more units; the stall
  // limit of 16 stops the call before column 3. Counted from the start it would stop before
  // column 2.
  fixture_t f(budget_model({1, 0, 0, 1}));
  work_budget_t budget;
  budget.limit         = 1 << 30;
  budget.stall_limit   = 2 * line_visit_work;
  rule_result_t result = f.run(budget);
  GPRS_CHECK(result.stopped);
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_CHECK_EQ(budget.used, 3 * line_visit_work);
  GPRS_CHECK_EQ(budget.used_at_finding, line_visit_work);
  GPRS_CHECK(f.st.col_frontier.list[col_list::dual_fixing] == (list_t{3}));

  // Columns that give nothing from the start: the fixable column behind them is not reached.
  fixture_t g(budget_model({0, 0, 1}));
  budget             = work_budget_t();
  budget.limit       = 1 << 30;
  budget.stall_limit = 2 * line_visit_work;
  result             = g.run(budget);
  GPRS_CHECK(result.stopped);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK(g.st.col_frontier.list[col_list::dual_fixing] == (list_t{2}));
}

// The work of a fixing is the used slots of the column, which fix_col walks.
GPRS_TEST(dual_fixing_work_counts_the_entries_of_fixed_columns)
{
  // x0 in three <= rows with positive coefficients (up locks only), x1 locked both ways.
  model_t m = make_model({{1., 1.}, {2., 0.}, {3., 1.}, {0., 1.}},
                         {-inf, -inf, -inf, 0.},
                         {9., 9., 9., 1.},
                         {0., 0.},
                         {1., 1.},
                         {1., 0.},
                         {0, 0});
  fixture_t f(m);
  work_budget_t budget = large_budget();
  rule_result_t result = f.run(budget);
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_CHECK_EQ(budget.used, 2 * line_visit_work + 3);
  GPRS_CHECK_EQ(budget.used_at_finding, line_visit_work + 3);
}

// Fixing by locks keeps the optimal value: tiny integer models, enumerated before and after.
// An infinite bound is enumerated as a window that holds every finite bound of the generator,
// so a fixing at a finite bound is the same step in the windowed model.
GPRS_TEST(dual_fixing_keeps_the_optimum_by_enumeration)
{
  random_options_t generator;
  int32_t fixed_cols = 0, kept_cols = 0, unbounded_cases = 0, growing = 0, feasible_cases = 0;
  int32_t failures = 0;
  for (uint64_t seed = 1; seed <= 4000 && failures < 5; ++seed) {
    model_t m = random_model(seed, generator);
    GPRS_REQUIRE(well_formed(m));
    fixture_t f(m);
    rule_result_t result = f.run();
    std::string name     = "seed " + std::to_string(seed);

    if (result.outcome == unbounded) {
      // Feasible in a window: a wider window has a strictly smaller optimum.
      ++unbounded_cases;
      optimum_t narrow = enumerate(m, 5.);
      if (!narrow.feasible) { continue; }
      ++growing;
      optimum_t wide = enumerate(m, 7.);
      if (!(wide.feasible && wide.objective < narrow.objective)) {
        ++failures;
        gprs::test::fail(__FILE__, __LINE__, name + ": unbounded claimed, optimum does not fall");
      }
      continue;
    }
    if (result.outcome != ok) {
      ++failures;
      gprs::test::fail(__FILE__, __LINE__, name + ": unexpected outcome");
      continue;
    }

    model_t reduced = m;
    size_t records  = 0;
    for (int32_t j = 0; j < m.ncols; ++j) {
      if (f.st.col_status[j] == col_status_t::live) {
        ++kept_cols;
        continue;
      }
      ++fixed_cols;
      ++records;
      reduced.col_lower[j] = f.st.col_lower[j];
      reduced.col_upper[j] = f.st.col_lower[j];
    }
    optimum_t before = enumerate(m, 5.);
    optimum_t after  = enumerate(reduced, 5.);
    feasible_cases += before.feasible;
    bool same = before.feasible == after.feasible &&
                (!before.feasible || before.objective == after.objective);
    same = same && f.ps.fixed_cols.size() == records && result.findings == (int64_t)records;
    for (const fixed_col_record_t& record : f.ps.fixed_cols) {
      same = same && record.value == f.st.col_lower[record.col] &&
             f.st.col_status[record.col] == col_status_t::fixed;
    }
    if (!same) {
      ++failures;
      gprs::test::fail(__FILE__,
                       __LINE__,
                       name + ": optimum " + std::to_string(before.objective) + " before, " +
                         std::to_string(after.objective) + " after");
    }
  }
  // The generator reaches every branch often.
  GPRS_CHECK(fixed_cols > 1000);
  GPRS_CHECK(kept_cols > 1000);
  GPRS_CHECK(unbounded_cases > 100);
  GPRS_CHECK(growing > 50);
  GPRS_CHECK(feasible_cases > 1000);
}

// Primal-only mode switches the lock arguments off; with them on, the same column is fixed.
GPRS_TEST(dual_fixing_is_off_in_primal_only_mode)
{
  model_t m = four_columns();
  options_t options;
  options.enabled.fill(false);
  options.enabled[rule::dual_fixing] = true;

  result_t on = presolve(m, options);
  GPRS_CHECK_EQ(on.status, status_t::reduced);
  GPRS_CHECK_EQ(on.reduced.ncols, 1);
  GPRS_CHECK_EQ(on.stats[rule::dual_fixing].findings, 3);
  GPRS_CHECK_EQ(on.postsolve.fixed_cols.size(), (size_t)3);

  options.primal_only = true;
  result_t off        = presolve(m, options);
  GPRS_CHECK_EQ(off.status, status_t::unchanged);
  GPRS_CHECK_EQ(off.reduced.ncols, 4);
  GPRS_CHECK_EQ(off.stats[rule::dual_fixing].calls, 0);
  GPRS_CHECK_EQ(off.stats[rule::dual_fixing].findings, 0);
  GPRS_CHECK(off.postsolve.order.empty());
}
