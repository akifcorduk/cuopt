/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "test_state_support.hpp"

#include <gprs/postsolve.hpp>
#include <gprs/rules.hpp>
#include <gprs/state.hpp>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

using namespace gprs;
using namespace state_test;

namespace {

// 4 rows, 5 columns, 10 nonzeros.
//   r0:  2 x0 - 3 x1 + x2            <= 10
//   r1:         x1        + x3        =  5
//   r2:  - x0       + x2       + x4  >=  1
//   r3:                     x3 - x4  in [-4, 4]
//   x0 integer in [0, 4], x1 in [-2, 3], x2 >= 0, x3 integer and free, x4 in [1, 8].
// Cost {1, -1, 0, 2, 0.5}, offset 7.
model_t hand_model()
{
  model_t m;
  m.nrows            = 4;
  m.ncols            = 5;
  m.row_start        = {0, 3, 5, 8, 10};
  m.col_index        = {0, 1, 2, 1, 3, 0, 2, 4, 3, 4};
  m.value            = {2., -3., 1., 1., 1., -1., 1., 1., 1., -1.};
  m.row_lower        = {-inf, 5., 1., -4.};
  m.row_upper        = {10., 5., inf, 4.};
  m.col_lower        = {0., -2., 0., -inf, 1.};
  m.col_upper        = {4., 3., inf, inf, 8.};
  m.cost             = {1., -1., 0., 2., 0.5};
  m.col_type         = {1, 0, 0, 1, 0};
  m.objective_offset = 7.;
  return m;
}

// The state, its naive reference and the record stack, with empty rule lists.
struct fixture_t {
  state_t st;
  shadow_t sh;
  postsolve_t ps;

  explicit fixture_t(const model_t& m, const tolerances_t& tol = tolerances_t())
    : st(m, tol), sh(m, tol)
  {
    ps.init(m);
    clear_rule_lists(st);
  }
};

using list_t    = std::vector<int32_t>;
using records_t = std::vector<fixed_col_record_t>;

constexpr rule_outcome_t ok         = rule_outcome_t::ok;
constexpr rule_outcome_t infeasible = rule_outcome_t::infeasible;

void check_change(bound_change_t result, rule_outcome_t outcome, bool changed)
{
  GPRS_CHECK_EQ(result.outcome, outcome);
  GPRS_CHECK_EQ(result.changed, changed);
}

void check_rule_lists(const state_t& st, list_mask_t events, const list_t& rows)
{
  for (int32_t l = 0; l < row_list::compaction; ++l) {
    GPRS_CHECK(st.row_frontier.list[l] == ((events >> l) & 1 ? rows : list_t()));
  }
  GPRS_CHECK(st.col_frontier.empty(col_list::dual_fixing));
}

// After a bound change: the rows of the column on the lists of a changed activity, and on the
// propagation list those whose side reads that activity.
void check_activity_lists(const state_t& st, const list_t& rows, const list_t& read)
{
  for (int32_t l = 0; l < row_list::compaction; ++l) {
    bool listed = (row_event::activity >> l) & 1;
    GPRS_CHECK(st.row_frontier.list[l] == (!listed                      ? list_t()
                                           : l == row_list::propagation ? read
                                                                        : rows));
  }
  GPRS_CHECK(st.col_frontier.empty(col_list::dual_fixing));
}

std::string records_error(const postsolve_t& ps, const records_t& expected)
{
  if (ps.order.size() != expected.size() || ps.fixed_cols.size() != expected.size() ||
      !ps.aliases.empty() || !ps.parallel_rows.empty()) {
    return "number of records";
  }
  std::vector<uint8_t> removed(ps.ncols, 0);
  for (size_t k = 0; k < expected.size(); ++k) {
    if (ps.order[k] != record_kind_t::fixed_col || ps.fixed_cols[k].col != expected[k].col ||
        ps.fixed_cols[k].value != expected[k].value) {
      return "record " + std::to_string(k);
    }
    removed[expected[k].col] = 1;
  }
  if (ps.col_removed != removed) { return "col_removed"; }
  return {};
}

void check_records(const postsolve_t& ps, const records_t& expected)
{
  GPRS_CHECK_EQ(records_error(ps, expected), std::string());
}

// The state stands for the model of the reference, recomputed from scratch, and is consistent.
void check_state(fixture_t& f)
{
  GPRS_CHECK_EQ(compare(f.st, f.sh), std::string());
  GPRS_CHECK_EQ(f.st.consistency_error(), std::string());
}

}  // namespace

GPRS_TEST(bounds_not_tighter)
{
  fixture_t f(hand_model());
  int64_t visits = f.st.slot_visits;
  // 5e-7 and 3.9999995 round to the current bound of the integer column within the tolerance.
  const std::pair<int32_t, double> lower[] = {
    {1, -2.}, {1, -5.}, {1, -inf}, {0, 0.}, {0, -0.5}, {0, 5e-7}, {2, 0.}, {3, -inf}, {4, 1.}};
  const std::pair<int32_t, double> upper[] = {
    {1, 3.}, {1, 7.}, {1, inf}, {0, 4.}, {0, 4.5}, {0, 3.9999995}, {2, inf}, {3, inf}, {4, 8.}};
  for (const auto& [col, bound] : lower) {
    check_change(tighten_lower(f.st, f.ps, col, bound), ok, false);
  }
  for (const auto& [col, bound] : upper) {
    check_change(tighten_upper(f.st, f.ps, col, bound), ok, false);
  }
  GPRS_CHECK_EQ(f.st.slot_visits, visits);
  GPRS_CHECK_EQ(f.st.bound_changes, 0);
  GPRS_CHECK_EQ(f.st.live_cols, 5);
  check_rule_lists(f.st, 0, {});
  check_records(f.ps, {});
  check_state(f);
}

// A continuous column gets the bound as given; its rows go on the lists of a changed activity.
GPRS_TEST(bounds_tighter_continuous)
{
  fixture_t f(hand_model());
  int64_t visits = f.st.slot_visits;

  check_change(tighten_lower(f.st, f.ps, 1, 0.5), ok, true);
  f.sh.col_lower[1] = 0.5;
  GPRS_CHECK_EQ(f.st.col_lower[1], 0.5);
  GPRS_CHECK_EQ(f.st.col_upper[1], 3.);
  GPRS_CHECK(f.st.is_live(1));
  GPRS_CHECK_EQ(f.st.max_activity(0), 6.5);
  GPRS_CHECK_EQ(f.st.min_activity(1), 0.5);
  GPRS_CHECK_EQ(f.st.bound_changes, 1);
  GPRS_CHECK_EQ(f.st.slot_visits - visits, 2);
  // The lower bound is in the maximum activity of r0, which has no lower side to read it with.
  check_activity_lists(f.st, {0, 1}, {1});
  check_records(f.ps, {});
  check_state(f);

  clear_rule_lists(f.st);
  check_change(tighten_upper(f.st, f.ps, 1, 2.25), ok, true);
  f.sh.col_upper[1] = 2.25;
  GPRS_CHECK_EQ(f.st.col_lower[1], 0.5);
  GPRS_CHECK_EQ(f.st.col_upper[1], 2.25);
  GPRS_CHECK(f.st.is_live(1));
  GPRS_CHECK_EQ(f.st.min_activity(0), -6.75);
  GPRS_CHECK_EQ(f.st.max_activity(1), 2.25);
  GPRS_CHECK_EQ(f.st.bound_changes, 2);
  GPRS_CHECK_EQ(f.st.slot_visits - visits, 4);
  check_activity_lists(f.st, {0, 1}, {0, 1});
  check_records(f.ps, {});
  check_state(f);
}

GPRS_TEST(bounds_infinite_to_finite)
{
  fixture_t f(hand_model());
  GPRS_CHECK_EQ(f.st.max_act.inf_count[0], 1);
  GPRS_CHECK_EQ(f.st.max_act.inf_count[2], 1);

  check_change(tighten_upper(f.st, f.ps, 2, 6.), ok, true);
  f.sh.col_upper[2] = 6.;
  GPRS_CHECK_EQ(f.st.col_upper[2], 6.);
  GPRS_CHECK_EQ(f.st.max_act.inf_count[0], 0);
  GPRS_CHECK_EQ(f.st.max_activity(0), 20.);
  GPRS_CHECK_EQ(f.st.max_act.inf_count[2], 0);
  GPRS_CHECK_EQ(f.st.max_activity(2), 14.);
  // r0 has no lower side for its maximum activity.
  check_activity_lists(f.st, {0, 2}, {2});
  check_state(f);

  // Integer and free: -2.5 becomes -2.
  clear_rule_lists(f.st);
  check_change(tighten_lower(f.st, f.ps, 3, -2.5), ok, true);
  f.sh.col_lower[3] = -2.;
  GPRS_CHECK_EQ(f.st.col_lower[3], -2.);
  GPRS_CHECK_EQ(f.st.col_upper[3], inf);
  GPRS_CHECK(f.st.is_live(3));
  GPRS_CHECK_EQ(f.st.min_act.inf_count[1], 0);
  GPRS_CHECK_EQ(f.st.min_activity(1), -4.);
  GPRS_CHECK_EQ(f.st.min_act.inf_count[3], 0);
  GPRS_CHECK_EQ(f.st.min_activity(3), -10.);
  check_activity_lists(f.st, {1, 3}, {1, 3});
  check_state(f);

  // A finite bound whose term is still above the infinite-term limit.
  check_change(tighten_upper(f.st, f.ps, 3, 1e30), ok, true);
  f.sh.col_upper[3] = 1e30;
  GPRS_CHECK_EQ(f.st.col_upper[3], 1e30);
  GPRS_CHECK_EQ(f.st.max_act.inf_count[1], 1);
  GPRS_CHECK_EQ(f.st.max_act.inf_count[3], 1);
  GPRS_CHECK_EQ(f.st.bound_changes, 3);
  check_records(f.ps, {});
  check_state(f);
}

// ceil(lower - feasibility) and floor(upper + feasibility), and a rounded bound that meets the
// other one fixes the column.
GPRS_TEST(bounds_integer_rounding)
{
  fixture_t f(hand_model());

  check_change(tighten_lower(f.st, f.ps, 0, 0.3), ok, true);
  f.sh.col_lower[0] = 1.;
  GPRS_CHECK_EQ(f.st.col_lower[0], 1.);
  check_state(f);
  check_change(tighten_lower(f.st, f.ps, 0, 1.0000005), ok, false);
  check_change(tighten_lower(f.st, f.ps, 0, 1.000002), ok, true);
  f.sh.col_lower[0] = 2.;
  GPRS_CHECK_EQ(f.st.col_lower[0], 2.);
  check_state(f);

  check_change(tighten_upper(f.st, f.ps, 0, 3.7), ok, true);
  f.sh.col_upper[0] = 3.;
  GPRS_CHECK_EQ(f.st.col_upper[0], 3.);
  check_state(f);
  check_change(tighten_upper(f.st, f.ps, 0, 2.9999995), ok, false);
  GPRS_CHECK_EQ(f.st.bound_changes, 3);
  GPRS_CHECK(f.st.is_live(0));
  check_records(f.ps, {});

  clear_rule_lists(f.st);
  int64_t visits = f.st.slot_visits;
  check_change(tighten_upper(f.st, f.ps, 0, 2.999998), ok, true);
  f.sh.fix_col(0, 2.);
  GPRS_CHECK_EQ(f.st.col_status[0], col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_lower[0], 2.);
  GPRS_CHECK_EQ(f.st.col_upper[0], 2.);
  GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 9.);
  GPRS_CHECK_EQ(f.st.live_cols, 4);
  GPRS_CHECK_EQ(f.st.fixed_nnz, 2);
  GPRS_CHECK(f.st.row_free_size == list_t({2, 2, 2, 2}));
  GPRS_CHECK_EQ(f.st.bound_changes, 4);
  GPRS_CHECK_EQ(f.st.slot_visits - visits, 2);
  check_rule_lists(f.st, row_event::entries, {0, 2});
  check_records(f.ps, {{0, 2.}});
  check_state(f);
}

// The integer column [0, 4] against bounds at and around its opposite bound.
GPRS_TEST(bounds_integer_at_opposite_bound)
{
  {
    fixture_t f(hand_model());
    check_change(tighten_lower(f.st, f.ps, 0, 3.2), ok, true);
    f.sh.fix_col(0, 4.);
    GPRS_CHECK_EQ(f.st.col_status[0], col_status_t::fixed);
    GPRS_CHECK_EQ(f.st.col_lower[0], 4.);
    GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 11.);
    check_rule_lists(f.st, row_event::entries, {0, 2});
    check_records(f.ps, {{0, 4.}});
    check_state(f);
  }
  {
    fixture_t f(hand_model());
    check_change(tighten_lower(f.st, f.ps, 0, 4.0000005), ok, true);
    f.sh.fix_col(0, 4.);
    check_records(f.ps, {{0, 4.}});
    check_state(f);
  }
  {
    fixture_t f(hand_model());
    check_change(tighten_upper(f.st, f.ps, 0, -5e-7), ok, true);
    f.sh.fix_col(0, 0.);
    GPRS_CHECK_EQ(f.st.col_upper[0], 0.);
    check_records(f.ps, {{0, 0.}});
    check_state(f);
  }
  {
    fixture_t f(hand_model());
    check_change(tighten_lower(f.st, f.ps, 0, 4.000002), infeasible, false);
    check_change(tighten_upper(f.st, f.ps, 0, -0.000002), infeasible, false);
    GPRS_CHECK_EQ(f.st.bound_changes, 0);
    check_records(f.ps, {});
    check_state(f);
  }
}

// A crossing of at most the tolerance sets the bound to the opposite one, which fixes the column
// there and not at the bound that was passed.
GPRS_TEST(bounds_crossing_within_tolerance)
{
  {
    fixture_t f(hand_model());
    int64_t visits = f.st.slot_visits;
    check_change(tighten_lower(f.st, f.ps, 1, 3.0000005), ok, true);
    f.sh.fix_col(1, 3.);
    GPRS_CHECK_EQ(f.st.col_status[1], col_status_t::fixed);
    GPRS_CHECK_EQ(f.st.col_lower[1], 3.);
    GPRS_CHECK_EQ(f.st.col_upper[1], 3.);
    GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 4.);
    GPRS_CHECK_EQ(f.st.bound_changes, 1);
    GPRS_CHECK_EQ(f.st.slot_visits - visits, 2);
    check_rule_lists(f.st, row_event::entries, {0, 1});
    check_records(f.ps, {{1, 3.}});
    check_state(f);
  }
  {
    fixture_t f(hand_model());
    check_change(tighten_upper(f.st, f.ps, 1, -2.0000005), ok, true);
    f.sh.fix_col(1, -2.);
    GPRS_CHECK_EQ(f.st.col_status[1], col_status_t::fixed);
    GPRS_CHECK_EQ(f.st.col_lower[1], -2.);
    GPRS_CHECK_EQ(f.st.col_upper[1], -2.);
    GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 9.);
    GPRS_CHECK_EQ(f.st.bound_changes, 1);
    check_rule_lists(f.st, row_event::entries, {0, 1});
    check_records(f.ps, {{1, -2.}});
    check_state(f);
  }
  // No crossing: the new bound is the opposite bound.
  {
    fixture_t f(hand_model());
    check_change(tighten_lower(f.st, f.ps, 4, 8.), ok, true);
    f.sh.fix_col(4, 8.);
    check_change(tighten_upper(f.st, f.ps, 1, -2.), ok, true);
    f.sh.fix_col(1, -2.);
    GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 13.);
    GPRS_CHECK_EQ(f.st.live_cols, 3);
    check_records(f.ps, {{4, 8.}, {1, -2.}});
    check_state(f);
  }
}

GPRS_TEST(bounds_crossing_beyond_tolerance)
{
  fixture_t f(hand_model());
  int64_t visits = f.st.slot_visits;
  check_change(tighten_lower(f.st, f.ps, 1, 3.000002), infeasible, false);
  check_change(tighten_upper(f.st, f.ps, 1, -2.000002), infeasible, false);
  check_change(tighten_lower(f.st, f.ps, 4, 100.), infeasible, false);
  check_change(tighten_upper(f.st, f.ps, 4, -100.), infeasible, false);
  // Infinite bounds, also against an infinite opposite bound and on an integer column.
  check_change(tighten_lower(f.st, f.ps, 1, inf), infeasible, false);
  check_change(tighten_upper(f.st, f.ps, 1, -inf), infeasible, false);
  check_change(tighten_lower(f.st, f.ps, 2, inf), infeasible, false);
  check_change(tighten_upper(f.st, f.ps, 2, -inf), infeasible, false);
  check_change(tighten_lower(f.st, f.ps, 3, inf), infeasible, false);
  check_change(tighten_upper(f.st, f.ps, 3, -inf), infeasible, false);
  GPRS_CHECK_EQ(f.st.slot_visits, visits);
  GPRS_CHECK_EQ(f.st.bound_changes, 0);
  GPRS_CHECK_EQ(f.st.live_cols, 5);
  check_rule_lists(f.st, 0, {});
  check_records(f.ps, {});
  check_state(f);
}

// With a tolerance of 0.25 every comparison is exact: a crossing of exactly the tolerance is
// accepted, and the tolerance is the one of the state.
GPRS_TEST(bounds_tolerance_boundary)
{
  tolerances_t tol;
  tol.feasibility = 0.25;
  {
    fixture_t f(hand_model(), tol);
    check_change(tighten_lower(f.st, f.ps, 1, 3.25), ok, true);
    f.sh.fix_col(1, 3.);
    check_records(f.ps, {{1, 3.}});
    check_state(f);
  }
  {
    fixture_t f(hand_model(), tol);
    check_change(tighten_upper(f.st, f.ps, 1, -2.25), ok, true);
    f.sh.fix_col(1, -2.);
    check_records(f.ps, {{1, -2.}});
    check_state(f);
  }
  {
    fixture_t f(hand_model(), tol);
    check_change(tighten_lower(f.st, f.ps, 1, 3.5), infeasible, false);
    check_change(tighten_upper(f.st, f.ps, 1, -2.5), infeasible, false);
    check_records(f.ps, {});
    check_state(f);
  }
  {
    fixture_t f(hand_model(), tol);
    check_change(tighten_lower(f.st, f.ps, 0, 1.25), ok, true);
    f.sh.col_lower[0] = 1.;
    GPRS_CHECK_EQ(f.st.col_lower[0], 1.);
    check_change(tighten_lower(f.st, f.ps, 0, 1.5), ok, true);
    f.sh.col_lower[0] = 2.;
    GPRS_CHECK_EQ(f.st.col_lower[0], 2.);
    check_change(tighten_upper(f.st, f.ps, 0, 2.75), ok, true);
    f.sh.col_upper[0] = 3.;
    GPRS_CHECK_EQ(f.st.col_upper[0], 3.);
    check_state(f);
    check_change(tighten_upper(f.st, f.ps, 0, 2.5), ok, true);
    f.sh.fix_col(0, 2.);
    check_records(f.ps, {{0, 2.}});
    check_state(f);
  }
}

GPRS_TEST(bounds_fix_column)
{
  fixture_t f(hand_model());
  int64_t visits = f.st.slot_visits;

  fix_column(f.st, f.ps, 1, 0.5);
  f.sh.fix_col(1, 0.5);
  GPRS_CHECK_EQ(f.st.col_status[1], col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_lower[1], 0.5);
  GPRS_CHECK_EQ(f.st.col_upper[1], 0.5);
  GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 6.5);
  GPRS_CHECK_EQ(f.st.live_cols, 4);
  GPRS_CHECK_EQ(f.st.bound_changes, 2);
  GPRS_CHECK_EQ(f.st.slot_visits - visits, 2);
  check_rule_lists(f.st, row_event::entries, {0, 1});
  check_records(f.ps, {{1, 0.5}});
  check_state(f);

  // Integer and free; its row r1 has no free entry left.
  clear_rule_lists(f.st);
  fix_column(f.st, f.ps, 3, -3.);
  f.sh.fix_col(3, -3.);
  GPRS_CHECK_EQ(f.st.col_status[3], col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.row_free_size[1], 0);
  GPRS_CHECK_EQ(f.st.min_act.inf_count[1], 0);
  GPRS_CHECK_EQ(f.st.max_act.inf_count[1], 0);
  GPRS_CHECK_EQ(f.st.min_activity(1), -2.5);
  GPRS_CHECK_EQ(f.st.max_activity(1), -2.5);
  GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 0.5);
  check_rule_lists(f.st, row_event::entries, {1, 3});
  check_state(f);

  // At a bound.
  fix_column(f.st, f.ps, 0, 0.);
  f.sh.fix_col(0, 0.);
  GPRS_CHECK_EQ(f.st.bound_changes, 5);
  GPRS_CHECK_EQ(f.st.live_cols, 2);
  GPRS_CHECK_EQ(f.st.fixed_nnz, 6);
  check_records(f.ps, {{1, 0.5}, {3, -3.}, {0, 0.}});
  check_state(f);

  f.ps.col_origin = {2, 4};
  std::vector<double> x;
  f.ps.postsolve({1.5, 2.}, x);
  GPRS_CHECK(x == std::vector<double>({0., 0.5, 1.5, -3., 2.}));
}

namespace {

struct expected_t {
  rule_outcome_t outcome = ok;
  bool changed           = false;
  bool fixed             = false;
  double bound           = 0.;
};

// The contract of tighten_lower on a column with the bounds [lower, upper]. tighten_upper is
// the same on the negated column: lower bound -upper, upper bound -lower, candidate -v.
expected_t expect_tighten(
  double lower, double upper, bool integer, double v, const tolerances_t& tol)
{
  expected_t e;
  if (integer) { v = std::ceil(v - tol.feasibility); }
  if (v <= lower) { return e; }
  if (v == inf || v - upper > tol.feasibility) {
    e.outcome = infeasible;
    return e;
  }
  e.changed = true;
  e.bound   = v < upper ? v : upper;
  e.fixed   = e.bound == upper;
  return e;
}

// A candidate for the lower bound of a column with the bounds [lower, upper]: not tighter,
// inside, at the upper bound, across it within and beyond the tolerance, infinite, and next to
// an integer inside.
double random_candidate(rng_t& rng, double lower, double upper)
{
  double inside;
  if (lower > -inf && upper < inf) {
    inside = lower + (upper - lower) * rng.uniform();
  } else if (lower > -inf) {
    inside = lower + std::fabs(random_bound_value(rng));
  } else if (upper < inf) {
    inside = upper - std::fabs(random_bound_value(rng));
  } else {
    inside = random_bound_value(rng);
  }
  switch (rng.below(10)) {
    case 0: return lower;
    case 1: return rng.chance(0.5) ? -inf : lower - 1.;
    case 2: return inside;
    case 3: return upper;
    case 4: return upper + 5e-7;
    case 5: return upper + 3e-6;
    case 6: return inf;
    case 7: return random_bound_value(rng);
    case 8: return std::round(inside) + (rng.chance(0.5) ? 5e-7 : -5e-7);
    default: return std::round(inside) + (rng.chance(0.5) ? 2e-6 : -2e-6);
  }
}

double random_fix_value(rng_t& rng, double lower, double upper, bool integer)
{
  double v;
  if (lower > -inf && upper < inf) {
    int32_t pick = rng.below(3);
    v            = pick == 0 ? lower : pick == 1 ? upper : lower + (upper - lower) * rng.uniform();
  } else if (lower > -inf) {
    v = lower + rng.below(5);
  } else if (upper < inf) {
    v = upper - rng.below(5);
  } else {
    v = rng.below(11) - 5;
  }
  if (integer) { v = std::round(v); }
  return v < lower ? lower : v > upper ? upper : v;
}

// What presolve() does before the first rule: integral bounds on integer columns and
// lower <= upper.
void normalise_bounds(model_t& m)
{
  for (int32_t j = 0; j < m.ncols; ++j) {
    if (m.col_type[j]) {
      m.col_lower[j] = std::ceil(m.col_lower[j]);
      m.col_upper[j] = std::floor(m.col_upper[j]);
    }
    if (m.col_lower[j] > m.col_upper[j]) { m.col_upper[j] = m.col_lower[j]; }
  }
}

struct counts_t {
  int32_t unchanged  = 0;
  int32_t tightened  = 0;
  int32_t fixed      = 0;
  int32_t infeasible = 0;
  int32_t fix_column = 0;
};

// One random call on a live column, checked against the contract and the reference. Returns the
// first difference.
std::string random_step(fixture_t& f, rng_t& rng, int32_t col, records_t& records, counts_t& n)
{
  state_t& st       = f.st;
  shadow_t& sh      = f.sh;
  double lower      = sh.col_lower[col];
  double upper      = sh.col_upper[col];
  bool integer      = sh.is_integer[col] != 0;
  int32_t operation = rng.below(5);
  clear_rule_lists(st);
  int64_t visits = st.slot_visits;

  expected_t e;
  bool tightened_lower = operation < 2;
  if (operation == 4) {
    e.changed = true;
    e.fixed   = true;
    e.bound   = random_fix_value(rng, lower, upper, integer);
    fix_column(st, f.ps, col, e.bound);
    ++n.fix_column;
  } else {
    bool is_lower = operation < 2;
    bound_change_t result;
    if (is_lower) {
      double v = random_candidate(rng, lower, upper);
      e        = expect_tighten(lower, upper, integer, v, sh.tol);
      result   = tighten_lower(st, f.ps, col, v);
    } else {
      double v = -random_candidate(rng, -upper, -lower);
      e        = expect_tighten(-upper, -lower, integer, -v, sh.tol);
      e.bound  = -e.bound;
      result   = tighten_upper(st, f.ps, col, v);
    }
    if (result.outcome != e.outcome || result.changed != e.changed) {
      return std::string(is_lower ? "tighten_lower" : "tighten_upper") + ": outcome " +
             std::to_string((int)result.outcome) + " changed " + std::to_string(result.changed) +
             ", expected " + std::to_string((int)e.outcome) + " " + std::to_string(e.changed);
    }
    if (e.outcome == infeasible) {
      ++n.infeasible;
    } else if (!e.changed) {
      ++n.unchanged;
    } else if (e.fixed) {
      ++n.fixed;
    } else {
      ++n.tightened;
      (is_lower ? sh.col_lower : sh.col_upper)[col] = e.bound;
    }
  }
  if (e.fixed) {
    sh.fix_col(col, e.bound);
    records.push_back({col, e.bound});
  }

  std::string error = compare(st, sh);
  if (error.empty()) { error = st.consistency_error(); }
  if (error.empty()) { error = records_error(f.ps, records); }
  if (!error.empty()) { return error; }

  // The rows of the column are on the lists of the event, and nothing else is on a list. A
  // tightened bound changes one activity of a row, which is on the propagation list when its
  // side that reads that activity is finite.
  int64_t row_visits = 0;
  for (int32_t i = 0; i < sh.nrows; ++i) {
    auto entry        = sh.row[i].find(col);
    bool has_col      = entry != sh.row[i].end();
    list_mask_t event = 0;
    if (has_col && e.changed && e.fixed) {
      event = row_event::entries;
    } else if (has_col && e.changed) {
      bool in_min = (entry->second > 0) == tightened_lower;
      bool read   = in_min ? sh.row_upper[i] < inf : sh.row_lower[i] > -inf;
      event       = read ? row_event::activity : row_event::activity_unread;
    }
    if ((st.row_frontier.pending[i] & row_event::entries) != event) {
      return "row " + std::to_string(i) + ": lists " + std::to_string(st.row_frontier.pending[i]);
    }
    if (has_col) { row_visits += st.csr.used[i]; }
  }
  if (!st.col_frontier.empty(col_list::dual_fixing)) { return "a column on the dual-fixing list"; }

  // O(1) without a change; otherwise the column once, and its rows where they are summed again.
  int64_t spent = st.slot_visits - visits;
  if (!e.changed ? spent != 0 : spent < st.csc.used[col] || spent > st.csc.used[col] + row_visits) {
    return "slots visited: " + std::to_string(spent);
  }
  return {};
}

}  // namespace

// Random calls of the three functions on random models with integer and continuous columns,
// infinite bounds and terms around the infinite-term limit. After every call the state is
// compared with a reference that is recomputed from scratch.
GPRS_TEST(bounds_random_against_reference)
{
  counts_t n;
  for (uint64_t seed = 1; seed <= 400; ++seed) {
    rng_t rng(seed);
    model_options_t o;
    o.nrows     = 2 + rng.below(6);
    o.ncols     = 2 + rng.below(8);
    o.density   = 0.3 + 0.4 * rng.uniform();
    o.long_rows = rng.below(2);
    o.decimal   = rng.chance(0.5);
    model_t m   = random_model(rng, o);
    normalise_bounds(m);
    fixture_t f(m);
    records_t records;
    for (int32_t j = 0; j < m.ncols; ++j) {
      if (m.col_lower[j] != m.col_upper[j]) { continue; }
      fix_column(f.st, f.ps, j, m.col_lower[j]);
      f.sh.fix_col(j, m.col_lower[j]);
      records.push_back({j, m.col_lower[j]});
    }
    std::string error = compare(f.st, f.sh);
    for (int32_t step = 0; error.empty() && step < 60; ++step) {
      int32_t col = rng.below(m.ncols);
      if (!f.st.is_live(col)) { continue; }
      error = random_step(f, rng, col, records, n);
      if (!error.empty()) { error = "step " + std::to_string(step) + ": " + error; }
    }
    if (!error.empty()) {
      ::gprs::test::fail(__FILE__, __LINE__, "seed " + std::to_string(seed) + ": " + error);
      return;
    }

    // The records give the fixed columns their values.
    std::vector<double> x_reduced, x;
    for (int32_t j = 0; j < m.ncols; ++j) {
      if (!f.st.is_live(j)) { continue; }
      f.ps.col_origin.push_back(j);
      x_reduced.push_back(j);
    }
    f.ps.postsolve(x_reduced, x);
    for (int32_t j = 0; j < m.ncols; ++j) {
      GPRS_CHECK_EQ(x[j], f.st.is_live(j) ? j : f.st.col_lower[j]);
    }
  }
  GPRS_CHECK(n.unchanged > 500);
  GPRS_CHECK(n.tightened > 500);
  GPRS_CHECK(n.fixed > 200);
  GPRS_CHECK(n.infeasible > 500);
  GPRS_CHECK(n.fix_column > 200);
}

// An activity is fresh until a term of it changes; refresh_activity sums a row again only when
// an activity is not fresh and its error bound could decide a test.
GPRS_TEST(bounds_refresh_activity)
{
  model_t m      = hand_model();
  m.col_upper[4] = 1e19;
  fixture_t f(m);
  for (int32_t i = 0; i < f.st.nrows; ++i) {
    GPRS_CHECK(activity_fresh(f.st, f.st.min_act, i));
    GPRS_CHECK(activity_fresh(f.st, f.st.max_act, i));
    GPRS_CHECK(!refresh_activity(f.st, i));
  }

  // A small term changes: not fresh, and an error bound far below anything a test compares.
  f.st.set_col_upper(1, 2.);
  GPRS_CHECK(!activity_fresh(f.st, f.st.min_act, 0));
  GPRS_CHECK(activity_fresh(f.st, f.st.max_act, 0));
  GPRS_CHECK(f.st.min_activity_error(0) < 1e-25);
  uint32_t updates = f.st.min_act.updates[0];
  GPRS_CHECK(!refresh_activity(f.st, 0));
  GPRS_CHECK_EQ(f.st.min_act.updates[0], updates);

  // The term of 1e19 leaves the maximum activity of r2 and the minimum activity of r3. Its size
  // stays in their error bounds until they are summed again.
  f.st.set_col_upper(4, 8.);
  GPRS_CHECK_EQ(f.st.max_act.max_term[2], 1e19);
  GPRS_CHECK(f.st.max_activity_error(2) > 1e-13);
  GPRS_CHECK(!activity_fresh(f.st, f.st.max_act, 2));
  int64_t visits = f.st.slot_visits;
  GPRS_CHECK(refresh_activity(f.st, 2));
  GPRS_CHECK(activity_fresh(f.st, f.st.max_act, 2));
  GPRS_CHECK_EQ(f.st.max_act.max_term[2], 8.);
  GPRS_CHECK(f.st.max_activity_error(2) < 1e-25);
  GPRS_CHECK(!refresh_activity(f.st, 2));
  GPRS_CHECK(refresh_activity(f.st, 3));
  GPRS_CHECK(f.st.slot_visits >= visits);
  GPRS_CHECK_EQ(f.st.consistency_error(), std::string());

  // An infinite term is no update: x2 has no upper bound, and the maximum activity of r0 is
  // fresh with two finite terms and one infinite.
  GPRS_CHECK_EQ(f.st.max_act.inf_count[0], 1);
  GPRS_CHECK_EQ(f.st.max_act.updates[0], 2u);
  GPRS_CHECK(activity_fresh(f.st, f.st.max_act, 0));
}
