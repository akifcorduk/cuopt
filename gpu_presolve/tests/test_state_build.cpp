/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"

#include <gprs/state.hpp>

#include <cmath>
#include <cstring>

using namespace gprs;

namespace {

uint64_t bits(double value)
{
  uint64_t pattern;
  std::memcpy(&pattern, &value, sizeof(pattern));
  return pattern;
}

// 5 rows, 5 columns, 12 nonzeros:
//   r0:  2 x0 - 3 x1 +   x2            <= 10
//   r1:         x1        +   x3        =  5
//   r2:            -   x2 - 2 x3 + x4/2 >= 1
//   r3:  (empty)                        in [-1, 1]
//   r4:    x0 +   x1 +   x2      + x4   in [2, 8]
//   x0 integer in [0, 4], x1 in [-2, 3], x2 >= 0, x3 integer and free, x4 in [1, 1e21].
model_t hand_model()
{
  model_t m;
  m.nrows            = 5;
  m.ncols            = 5;
  m.row_start        = {0, 3, 5, 8, 8, 12};
  m.col_index        = {0, 1, 2, 1, 3, 2, 3, 4, 0, 1, 2, 4};
  m.value            = {2., -3., 1., 1., 1., -1., -2., 0.5, 1., 1., 1., 1.};
  m.row_lower        = {-inf, 5., 1., -1., 2.};
  m.row_upper        = {10., 5., inf, 1., 8.};
  m.col_lower        = {0., -2., 0., -inf, 1.};
  m.col_upper        = {4., 3., inf, inf, 1e21};
  m.cost             = {1., -1., 0., 2., 0.5};
  m.col_type         = {1, 0, 0, 1, 0};
  m.objective_offset = 7.;
  return m;
}

struct expected_side_t {
  double hi;
  double lo;
  int32_t inf_count;
  int64_t inf_slot;
  double max_term;
  uint32_t updates;
};

void check_side(const activity_t& act, int32_t row, const expected_side_t& expected)
{
  GPRS_CHECK_EQ(act.hi[row], expected.hi);
  GPRS_CHECK_EQ(act.lo[row], expected.lo);
  GPRS_CHECK_EQ(act.inf_count[row], expected.inf_count);
  GPRS_CHECK_EQ(act.inf_slot[row], expected.inf_slot);
  GPRS_CHECK_EQ(act.max_term[row], expected.max_term);
  GPRS_CHECK_EQ(act.updates[row], expected.updates);
}

// The summation order of numerics.hpp written out without any shortcut: all lanes, full tree.
void reference_fold(const state_t& st, int32_t row, side_sum_t& min_side, side_sum_t& max_side)
{
  std::vector<int64_t> live;
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (st.csr.live(e)) { live.push_back(e); }
  }
  int32_t n     = live.size();
  int32_t lanes = activity_lanes(n);
  std::vector<dd_t> lane_min(lanes), lane_max(lanes);
  min_side = side_sum_t{};
  max_side = side_sum_t{};
  for (int32_t k = 0; k < n; ++k) {
    int64_t e    = live[k];
    int32_t j    = st.csr.index[e];
    double a     = st.csr.value[e];
    double b_min = a > 0 ? st.col_lower[j] : st.col_upper[j];
    double b_max = a > 0 ? st.col_upper[j] : st.col_lower[j];
    if (std::fabs(a * b_min) > st.tol.infinite_term) {
      ++min_side.inf_count;
      min_side.inf_slot += e;
    } else {
      dd_add_product(lane_min[k % lanes], a, b_min);
      min_side.max_term = std::fmax(min_side.max_term, std::fabs(a * b_min));
      ++min_side.terms;
    }
    if (std::fabs(a * b_max) > st.tol.infinite_term) {
      ++max_side.inf_count;
      max_side.inf_slot += e;
    } else {
      dd_add_product(lane_max[k % lanes], a, b_max);
      max_side.max_term = std::fmax(max_side.max_term, std::fabs(a * b_max));
      ++max_side.terms;
    }
  }
  for (int32_t off = lanes / 2; off > 0; off /= 2) {
    for (int32_t l = 0; l < off; ++l) {
      lane_min[l] = dd_add(lane_min[l], lane_min[l + off]);
      lane_max[l] = dd_add(lane_max[l], lane_max[l + off]);
    }
  }
  min_side.sum = lane_min[0];
  max_side.sum = lane_max[0];
}

void check_sum_bits(const activity_t& act, int32_t row, const side_sum_t& expected)
{
  GPRS_CHECK_EQ(bits(act.hi[row]), bits(expected.sum.hi));
  GPRS_CHECK_EQ(bits(act.lo[row]), bits(expected.sum.lo));
  GPRS_CHECK_EQ(act.inf_count[row], expected.inf_count);
  GPRS_CHECK_EQ(act.inf_slot[row], expected.inf_slot);
  GPRS_CHECK_EQ(bits(act.max_term[row]), bits(expected.max_term));
  GPRS_CHECK_EQ(act.updates[row], expected.terms);
}

// One row over n columns whose terms need the compensation: magnitudes from 1e-3 to 1e15, both
// signs, a third of the columns with an infinite or huge bound.
model_t wide_row_model(int32_t n, uint64_t seed)
{
  model_t m;
  m.nrows     = 1;
  m.ncols     = n;
  m.row_start = {0, n};
  m.row_lower = {-1.};
  m.row_upper = {1.};
  uint64_t x  = seed;
  auto draw   = [&x]() {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    return (x >> 11) * 0x1p-53;
  };
  for (int32_t j = 0; j < n; ++j) {
    double a     = (draw() < 0.5 ? -1. : 1.) * std::pow(10., 6. * draw() - 3.) * (0.1 + draw());
    double lower = -std::pow(10., 9. * draw()) * draw();
    double upper = lower + std::pow(10., 9. * draw()) * draw();
    double kind  = draw();
    if (kind < 0.1) { lower = -inf; }
    if (kind > 0.9) { upper = inf; }
    if (kind > 0.45 && kind < 0.55) { upper = 1e24; }
    m.col_index.push_back(j);
    m.value.push_back(a);
    m.col_lower.push_back(lower);
    m.col_upper.push_back(upper);
    m.cost.push_back(0.);
    m.col_type.push_back(j % 2);
  }
  return m;
}

}  // namespace

GPRS_TEST(state_build_matrix)
{
  model_t m = hand_model();
  GPRS_REQUIRE(well_formed(m));
  state_t st(m, tolerances_t());

  GPRS_CHECK_EQ(st.nrows, 5);
  GPRS_CHECK_EQ(st.ncols, 5);
  GPRS_CHECK_EQ(st.live_rows, 5);
  GPRS_CHECK_EQ(st.live_cols, 5);
  GPRS_CHECK_EQ(st.live_nnz, 12);
  GPRS_CHECK_EQ(st.fixed_nnz, 0);
  GPRS_CHECK_EQ(st.free_nnz(), 12);
  GPRS_CHECK_EQ(st.bound_changes, 0);
  GPRS_CHECK_EQ(st.side_changes, 0);
  GPRS_CHECK_EQ(st.coefficient_changes, 0);
  GPRS_CHECK_EQ(st.objective_offset.hi, 7.);
  GPRS_CHECK_EQ(st.objective_offset.lo, 0.);
  GPRS_CHECK_EQ(st.tol.feasibility, 1e-6);

  GPRS_CHECK(st.csr.start == std::vector<int64_t>({0, 3, 5, 8, 8}));
  GPRS_CHECK(st.csr.size == std::vector<int32_t>({3, 2, 3, 0, 4}));
  GPRS_CHECK(st.csr.used == st.csr.size);
  GPRS_CHECK(st.csr.capacity == st.csr.size);
  GPRS_CHECK(st.csr.index == m.col_index);
  GPRS_CHECK(st.csr.value == m.value);
  GPRS_CHECK_EQ(st.csr.slots(), 12);
  GPRS_CHECK_EQ(st.csr.begin(2), 5);
  GPRS_CHECK_EQ(st.csr.end(2), 8);
  GPRS_CHECK_EQ(st.csr.begin(3), st.csr.end(3));

  GPRS_CHECK(st.csc.start == std::vector<int64_t>({0, 2, 5, 8, 10}));
  GPRS_CHECK(st.csc.size == std::vector<int32_t>({2, 3, 3, 2, 2}));
  GPRS_CHECK(st.csc.used == st.csc.size);
  GPRS_CHECK(st.csc.capacity == st.csc.size);
  GPRS_CHECK(st.csc.index == std::vector<int32_t>({0, 4, 0, 1, 4, 0, 2, 4, 1, 2, 2, 4}));
  GPRS_CHECK(st.csc.value ==
             std::vector<double>({2., 1., -3., 1., 1., 1., -1., 1., 1., -2., 0.5, 1.}));
  GPRS_CHECK(st.csc.twin == std::vector<int64_t>({0, 8, 1, 3, 9, 2, 5, 10, 4, 6, 7, 11}));
  GPRS_CHECK(st.csr.twin == std::vector<int64_t>({0, 2, 5, 3, 8, 6, 9, 10, 1, 4, 7, 11}));
  for (int32_t i = 0; i < st.nrows; ++i) {
    for (int64_t e = st.csr.begin(i); e < st.csr.end(i); ++e) {
      GPRS_CHECK(st.csr.live(e));
      int64_t p = st.csr.twin[e];
      GPRS_CHECK_EQ(st.csc.twin[p], e);
      GPRS_CHECK_EQ(st.csc.index[p], i);
      GPRS_CHECK_EQ(st.csc.value[p], st.csr.value[e]);
      GPRS_CHECK(p >= st.csc.begin(st.csr.index[e]) && p < st.csc.end(st.csr.index[e]));
    }
  }
}

GPRS_TEST(state_build_rows_and_columns)
{
  model_t m = hand_model();
  state_t st(m, tolerances_t());

  GPRS_CHECK(st.row_lower == m.row_lower);
  GPRS_CHECK(st.row_upper == m.row_upper);
  GPRS_CHECK(st.row_alive == std::vector<uint8_t>(5, 1));
  GPRS_CHECK(st.row_free_size == st.csr.size);
  GPRS_CHECK(st.row_version == std::vector<uint32_t>(5, 0));

  GPRS_CHECK(st.col_lower == m.col_lower);
  GPRS_CHECK(st.col_upper == m.col_upper);
  GPRS_CHECK(st.cost == m.cost);
  for (int32_t j = 0; j < st.ncols; ++j) {
    GPRS_CHECK_EQ(st.is_integer(j), m.col_type[j] == 1);
    GPRS_CHECK(st.is_live(j));
    GPRS_CHECK_EQ(st.col_status[j], col_status_t::live);
    GPRS_CHECK_EQ(st.implied_lower[j], -inf);
    GPRS_CHECK_EQ(st.implied_upper[j], inf);
    GPRS_CHECK_EQ(st.implied_lower_row[j], -1);
    GPRS_CHECK_EQ(st.implied_upper_row[j], -1);
    GPRS_CHECK_EQ(st.implied_lower_version[j], 0u);
    GPRS_CHECK_EQ(st.implied_upper_version[j], 0u);
  }
  GPRS_CHECK(st.up_locks == std::vector<int32_t>({2, 2, 3, 2, 1}));
  GPRS_CHECK(st.down_locks == std::vector<int32_t>({1, 3, 1, 1, 2}));
}

GPRS_TEST(state_build_activities)
{
  state_t st(hand_model(), tolerances_t());

  // r0: lanes {x0, x2} and {x1}. max: x2 has no upper bound.
  check_side(st.min_act, 0, {-9., 0., 0, 0, 9., 3});
  check_side(st.max_act, 0, {14., 0., 1, 2, 8., 2});
  GPRS_CHECK_EQ(st.min_activity(0), -9.);
  GPRS_CHECK_EQ(st.max_activity(0), 14.);
  GPRS_CHECK_EQ(st.csr.index[st.max_act.inf_slot[0]], 2);

  // r1: x3 is free, so it is the single infinite term of both sides.
  check_side(st.min_act, 1, {-2., 0., 1, 4, 2., 1});
  check_side(st.max_act, 1, {3., 0., 1, 4, 3., 1});
  GPRS_CHECK_EQ(st.csr.index[st.min_act.inf_slot[1]], 3);

  // r2: min has x2 and x3 infinite (slots 5 + 6). max has x3 infinite and the term of x4,
  // 0.5 * 1e21, above 1e20 (slots 6 + 7); its only finite term is -1 * 0.
  check_side(st.min_act, 2, {0.5, 0., 2, 11, 0.5, 1});
  check_side(st.max_act, 2, {0., 0., 2, 13, 0., 1});

  check_side(st.min_act, 3, {0., 0., 0, 0, 0., 0});
  check_side(st.max_act, 3, {0., 0., 0, 0, 0., 0});

  // r4: lanes {x0, x2} and {x1, x4}. max: x2 unbounded and 1 * 1e21 counted as infinite.
  check_side(st.min_act, 4, {-1., 0., 0, 0, 2., 4});
  check_side(st.max_act, 4, {7., 0., 2, 21, 4., 2});

  GPRS_CHECK_EQ(st.min_activity_error(0), activity_error(3, 9.));
  GPRS_CHECK_EQ(st.max_activity_error(4), activity_error(2, 4.));
  GPRS_CHECK_EQ(st.min_activity_error(3), 0.);

  for (int32_t i = 0; i < st.nrows; ++i) {
    side_sum_t min_side, max_side;
    st.fold_row_activity(i, min_side, max_side);
    check_sum_bits(st.min_act, i, min_side);
    check_sum_bits(st.max_act, i, max_side);
    reference_fold(st, i, min_side, max_side);
    check_sum_bits(st.min_act, i, min_side);
    check_sum_bits(st.max_act, i, max_side);
  }
}

// Terms at the limit of the infinite-term rule, negative coefficients on infinite bounds, a
// bound of -0, and a sum whose hi parts cancel so that the value sits in lo.
//   r0: -2 x0 + x1 - x2 + 4 x3 <= 5
//   r1: 1e10 x0 - 1e-10 x1      = 0
//   r2:               x2 +   x3 >= 1
//   x0 <= 5e19, x1 = 1e20, x2 = 3 (integer), x3 in [-0, 2.5e19].
GPRS_TEST(state_build_activity_limits)
{
  model_t m;
  m.nrows     = 3;
  m.ncols     = 4;
  m.row_start = {0, 4, 6, 8};
  m.col_index = {0, 1, 2, 3, 0, 1, 2, 3};
  m.value     = {-2., 1., -1., 4., 1e10, -1e-10, 1., 1.};
  m.row_lower = {-inf, 0., 1.};
  m.row_upper = {5., 0., inf};
  m.col_lower = {-inf, 1e20, 3., -0.};
  m.col_upper = {5e19, 1e20, 3., 2.5e19};
  m.cost      = {0., 0., 0., 0.};
  m.col_type  = {0, 0, 1, 0};
  GPRS_REQUIRE(well_formed(m));
  state_t st(m, tolerances_t());

  // r0 min: lanes {-2 * 5e19, -1 * 3} and {1 * 1e20, 4 * -0}; -1e20 and 1e20 are finite terms
  // and cancel in hi. r0 max: -2 * -inf is the single infinite term, the rest is 2e20 - 3.
  check_side(st.min_act, 0, {0., -3., 0, 0, 1e20, 4});
  GPRS_CHECK_EQ(st.min_activity(0), -3.);
  check_side(st.max_act, 0, {2e20, -3., 1, 0, 1e20, 3});
  GPRS_CHECK_EQ(st.csr.index[st.max_act.inf_slot[0]], 0);

  // r1: x0 is infinite on both sides, by its bound for the minimum and by 1e10 * 5e19 for the
  // maximum; the term of x1 is -1e-10 * 1e20 with its product error in lo.
  double product_error;
  double product = two_prod(-1e-10, 1e20, product_error);
  GPRS_CHECK(product_error != 0.);
  check_side(st.min_act, 1, {product, product_error, 1, 4, -product, 1});
  check_side(st.max_act, 1, {product, product_error, 1, 4, -product, 1});

  check_side(st.min_act, 2, {3., 0., 0, 0, 3., 2});
  check_side(st.max_act, 2, {2.5e19, 3., 0, 0, 2.5e19, 2});

  // x0: -2 in a <= row locks down, 1e10 in an equation locks both ways.
  GPRS_CHECK(st.up_locks == std::vector<int32_t>({1, 2, 0, 1}));
  GPRS_CHECK(st.down_locks == std::vector<int32_t>({2, 1, 2, 1}));
}

// A tombstone is no term and takes no lane: the row is folded as its live entries in storage
// order.
GPRS_TEST(state_build_fold_skips_tombstones)
{
  int32_t sizes[] = {3, 10, 70, 300};
  for (int32_t n : sizes) {
    state_t st(wide_row_model(n, 77 + n), tolerances_t());
    int32_t dead = 0;
    for (int64_t e = st.csr.begin(0); e < st.csr.end(0); e += 3) {
      st.csr.index[e] = dead_index;
      ++dead;
    }
    st.csr.size[0] -= dead;
    side_sum_t min_side, max_side, min_ref, max_ref;
    st.fold_row_activity(0, min_side, max_side);
    reference_fold(st, 0, min_ref, max_ref);
    GPRS_CHECK_EQ(bits(min_side.sum.hi), bits(min_ref.sum.hi));
    GPRS_CHECK_EQ(bits(min_side.sum.lo), bits(min_ref.sum.lo));
    GPRS_CHECK_EQ(bits(max_side.sum.hi), bits(max_ref.sum.hi));
    GPRS_CHECK_EQ(bits(max_side.sum.lo), bits(max_ref.sum.lo));
    GPRS_CHECK_EQ(min_side.inf_count, min_ref.inf_count);
    GPRS_CHECK_EQ(min_side.inf_slot, min_ref.inf_slot);
    GPRS_CHECK_EQ(max_side.inf_count, max_ref.inf_count);
    GPRS_CHECK_EQ(max_side.inf_slot, max_ref.inf_slot);
    GPRS_CHECK_EQ(min_side.inf_count + (int32_t)min_side.terms, n - dead);
  }
}

GPRS_TEST(state_build_lists)
{
  state_t st(hand_model(), tolerances_t());
  std::vector<int32_t> all = {0, 1, 2, 3, 4};
  std::vector<int32_t> taken;

  int32_t rule_lists[] = {row_list::row_rules,
                          row_list::propagation,
                          row_list::tightening,
                          row_list::doubleton,
                          row_list::parallel_rows};
  for (int32_t l : rule_lists) {
    GPRS_CHECK(st.row_frontier.list[l] == all);
  }
  GPRS_CHECK(st.row_frontier.empty(row_list::compaction));
  GPRS_CHECK(st.row_frontier.empty(row_list::singletons));
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == all);
  GPRS_CHECK(st.col_frontier.empty(col_list::compaction));
  // The column rules: every row on the list of simple probing, every column on the three
  // column lists, and no row on the row lists that only lead to columns.
  GPRS_CHECK(st.row_frontier.list[row_list::simple_probing] == all);
  for (int32_t l : {row_list::singleton_cols, row_list::dual_cols, row_list::parallel_cols}) {
    GPRS_CHECK(st.row_frontier.empty(l));
  }
  for (int32_t l : {col_list::singleton_cols, col_list::dual_cols, col_list::parallel_cols}) {
    GPRS_CHECK(st.col_frontier.list[l] == all);
  }
  GPRS_CHECK_EQ(st.row_frontier.pending[3], row_event::construction);
  GPRS_CHECK_EQ(st.col_frontier.pending[3], col_event::locks);

  st.row_frontier.take(row_list::propagation, taken);
  GPRS_CHECK(taken == all);
  GPRS_CHECK(st.row_frontier.empty(row_list::propagation));
  GPRS_CHECK(st.row_frontier.list[row_list::row_rules] == all);

  // An activity event reaches propagation again and no list twice.
  st.row_frontier.push(row_event::activity, 3);
  st.row_frontier.push(row_event::activity, 1);
  st.row_frontier.push(row_event::entries, 3);
  GPRS_CHECK(st.row_frontier.list[row_list::propagation] == std::vector<int32_t>({3, 1}));
  GPRS_CHECK(st.row_frontier.list[row_list::row_rules] == all);
  GPRS_CHECK(st.row_frontier.list[row_list::parallel_rows] == all);

  st.row_frontier.take(row_list::parallel_rows, taken);
  st.row_frontier.push(row_event::activity, 2);
  GPRS_CHECK(st.row_frontier.empty(row_list::parallel_rows));
  // A side is read by the parallel rows as well.
  st.row_frontier.push(row_event::side, 2);
  st.row_frontier.push(row_event::entries, 2);
  st.row_frontier.push(row_event::sparse, 4);
  st.row_frontier.push(row_event::sparse, 4);
  GPRS_CHECK(st.row_frontier.list[row_list::parallel_rows] == std::vector<int32_t>({2}));
  GPRS_CHECK(st.row_frontier.list[row_list::compaction] == std::vector<int32_t>({4}));

  st.row_frontier.take(row_list::doubleton, taken);
  GPRS_CHECK(taken == all);
  st.row_frontier.push(list_bit(row_list::doubleton), 4);
  st.row_frontier.push(list_bit(row_list::doubleton), 0);
  st.row_frontier.take(row_list::doubleton, taken);
  GPRS_CHECK(taken == std::vector<int32_t>({4, 0}));
  GPRS_CHECK(st.row_frontier.empty(row_list::doubleton));

  st.col_frontier.take(col_list::dual_fixing, taken);
  st.col_frontier.push(col_event::locks, 2);
  st.col_frontier.push(col_event::sparse, 2);
  st.col_frontier.push(col_event::locks, 2);
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == std::vector<int32_t>({2}));
  GPRS_CHECK(st.col_frontier.list[col_list::compaction] == std::vector<int32_t>({2}));

  // The row lists of the column rules get every activity, side and entries event, the one of
  // an activity that propagation does not read included, and no singleton or sparse event.
  for (int32_t l : {row_list::simple_probing,
                    row_list::singleton_cols,
                    row_list::dual_cols,
                    row_list::parallel_cols}) {
    st.row_frontier.take(l, taken);
    st.row_frontier.push(row_event::singleton, 0);
    st.row_frontier.push(row_event::sparse, 0);
    GPRS_CHECK(st.row_frontier.empty(l));
    st.row_frontier.push(row_event::activity_unread, 3);
    st.row_frontier.push(row_event::activity, 1);
    st.row_frontier.push(row_event::side, 3);
    st.row_frontier.push(row_event::entries, 0);
    GPRS_CHECK(st.row_frontier.list[l] == std::vector<int32_t>({3, 1, 0}));
  }
  // Their column lists get the events of locks and of entries; an entries event does not reach
  // dual fixing, which reads no coefficient.
  st.col_frontier.take(col_list::dual_fixing, taken);
  for (int32_t l : {col_list::singleton_cols, col_list::dual_cols, col_list::parallel_cols}) {
    st.col_frontier.take(l, taken);
    GPRS_CHECK(taken == all);
    st.col_frontier.push(col_event::sparse, 1);
    GPRS_CHECK(st.col_frontier.empty(l));
  }
  st.col_frontier.push(col_event::entries, 4);
  st.col_frontier.push(col_event::locks, 1);
  for (int32_t l : {col_list::singleton_cols, col_list::dual_cols, col_list::parallel_cols}) {
    GPRS_CHECK(st.col_frontier.list[l] == std::vector<int32_t>({4, 1}));
  }
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == std::vector<int32_t>({1}));
}

GPRS_TEST(state_build_event_masks)
{
  GPRS_CHECK_EQ(row_event::column_rules, 0x780);
  GPRS_CHECK_EQ(row_event::activity, 0x787);
  GPRS_CHECK_EQ(row_event::activity_unread, 0x785);
  GPRS_CHECK_EQ(row_event::side, 0x79f);
  GPRS_CHECK_EQ(row_event::entries, 0x79f);
  GPRS_CHECK_EQ(row_event::sparse, 0x20);
  GPRS_CHECK_EQ(row_event::singleton, 0x50);
  GPRS_CHECK_EQ(row_event::construction, 0x9f);
  GPRS_CHECK_EQ(col_event::column_rules, 0x1c);
  GPRS_CHECK_EQ(col_event::locks, 0x1d);
  GPRS_CHECK_EQ(col_event::entries, 0x1c);
  GPRS_CHECK_EQ(col_event::sparse, 0x02);
  // Lists of later rules have a bit left in both masks.
  GPRS_CHECK(row_list::count < 16 && col_list::count < 16);
  GPRS_CHECK_EQ(sizeof(list_mask_t), (size_t)2);
}

// state_t::row_singletons at construction: the live columns with one entry, by the row of it.
GPRS_TEST(state_build_singleton_counts)
{
  state_t st(hand_model(), tolerances_t());
  GPRS_CHECK(st.row_singletons == std::vector<int32_t>(5, 0));

  //   r0:  x0 + x1 + x2 <= 4
  //   r1:  x0      + x3  = 1
  //   r2:  (empty)
  // x1, x2 and x3 have one entry each, x4 has none; x2 has lower == upper and counts until the
  // start step of the engine fixes it.
  model_t m;
  m.nrows     = 3;
  m.ncols     = 5;
  m.row_start = {0, 3, 5, 5};
  m.col_index = {0, 1, 2, 0, 3};
  m.value     = {1., 1., 1., 1., 1.};
  m.row_lower = {-inf, 1., 0.};
  m.row_upper = {4., 1., 0.};
  m.col_lower = {0., 0., 2., 0., 0.};
  m.col_upper = {1., 1., 2., 1., 1.};
  m.cost      = {0., 0., 0., 0., 0.};
  m.col_type  = {0, 0, 0, 0, 0};
  state_t singles(m, tolerances_t());
  GPRS_CHECK(singles.row_singletons == std::vector<int32_t>({2, 1, 0}));
  GPRS_CHECK_EQ(singles.consistency_error(), std::string());
  singles.fix_col(2, 2.);
  GPRS_CHECK(singles.row_singletons == std::vector<int32_t>({1, 1, 0}));
  GPRS_CHECK_EQ(singles.consistency_error(), std::string());
}

// The stored activities follow the summation order of numerics.hpp bit for bit at every lane
// count, on terms whose plain sum loses digits.
GPRS_TEST(state_build_summation_order)
{
  int32_t sizes[] = {1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64, 65, 100, 255, 256, 257, 700, 5000};
  bool lo_used    = false;
  for (int32_t n : sizes) {
    state_t st(wide_row_model(n, 1000 + n), tolerances_t());
    GPRS_CHECK_EQ(st.csr.size[0], n);
    side_sum_t min_side, max_side;
    reference_fold(st, 0, min_side, max_side);
    check_sum_bits(st.min_act, 0, min_side);
    check_sum_bits(st.max_act, 0, max_side);
    GPRS_CHECK_EQ(min_side.inf_count + (int32_t)min_side.terms, n);
    GPRS_CHECK_EQ(max_side.inf_count + (int32_t)max_side.terms, n);
    lo_used = lo_used || st.min_act.lo[0] != 0. || st.max_act.lo[0] != 0.;
    if (n >= 100) { GPRS_CHECK(min_side.inf_count > 1 && max_side.inf_count > 1); }
  }
  GPRS_CHECK(lo_used);
}

GPRS_TEST(state_build_empty_model)
{
  model_t m;
  m.row_start = {0};
  GPRS_REQUIRE(well_formed(m));
  state_t st(m, tolerances_t());
  GPRS_CHECK_EQ(st.nrows, 0);
  GPRS_CHECK_EQ(st.ncols, 0);
  GPRS_CHECK_EQ(st.live_nnz, 0);
  GPRS_CHECK_EQ(st.csr.slots(), 0);
  GPRS_CHECK_EQ(st.csc.slots(), 0);
  GPRS_CHECK(st.row_frontier.empty(row_list::row_rules));
  GPRS_CHECK(st.col_frontier.empty(col_list::dual_fixing));
}

// Columns without entries and rows without entries at both ends of the index range.
GPRS_TEST(state_build_empty_lines)
{
  model_t m;
  m.nrows     = 3;
  m.ncols     = 4;
  m.row_start = {0, 0, 2, 2};
  m.col_index = {1, 2};
  m.value     = {-1., 4.};
  m.row_lower = {0., -inf, -inf};
  m.row_upper = {0., inf, 3.};
  m.col_lower = {0., 0., -5., 0.};
  m.col_upper = {1., 2., 5., inf};
  m.cost      = {0., 0., 0., 1.};
  m.col_type  = {0, 0, 0, 0};
  GPRS_REQUIRE(well_formed(m));
  state_t st(m, tolerances_t());
  GPRS_CHECK(st.csc.size == std::vector<int32_t>({0, 1, 1, 0}));
  GPRS_CHECK(st.csc.start == std::vector<int64_t>({0, 0, 1, 2}));
  GPRS_CHECK(st.csc.index == std::vector<int32_t>({1, 1}));
  // A free row locks nothing.
  GPRS_CHECK(st.up_locks == std::vector<int32_t>({0, 0, 0, 0}));
  GPRS_CHECK(st.down_locks == std::vector<int32_t>({0, 0, 0, 0}));
  check_side(st.min_act, 1, {-22., 0., 0, 0, 20., 2});
  check_side(st.max_act, 1, {20., 0., 0, 0, 20., 2});
  check_side(st.min_act, 0, {0., 0., 0, 0, 0., 0});
  check_side(st.max_act, 2, {0., 0., 0, 0, 0., 0});
}

GPRS_TEST(state_build_well_formed_rejects)
{
  model_t good = hand_model();
  GPRS_CHECK(well_formed(good));

  model_t m      = good;
  m.col_index[1] = 0;  // column index repeated inside a row
  GPRS_CHECK(!well_formed(m));
  m              = good;
  m.col_index[2] = 5;  // column out of range
  GPRS_CHECK(!well_formed(m));
  m              = good;
  m.col_index[0] = -1;
  GPRS_CHECK(!well_formed(m));
  m          = good;
  m.value[4] = 0.;
  GPRS_CHECK(!well_formed(m));
  m          = good;
  m.value[4] = inf;
  GPRS_CHECK(!well_formed(m));
  m              = good;
  m.row_start[2] = 9;  // decreasing
  GPRS_CHECK(!well_formed(m));
  m              = good;
  m.row_start[5] = 11;
  GPRS_CHECK(!well_formed(m));
  m = good;
  m.row_lower.pop_back();
  GPRS_CHECK(!well_formed(m));
  m = good;
  m.cost.push_back(0.);
  GPRS_CHECK(!well_formed(m));
  m              = good;
  m.row_lower[0] = inf;
  GPRS_CHECK(!well_formed(m));
  m              = good;
  m.col_upper[1] = -inf;
  GPRS_CHECK(!well_formed(m));
  m              = good;
  m.col_lower[1] = std::nan("");
  GPRS_CHECK(!well_formed(m));
  m         = good;
  m.cost[0] = inf;
  GPRS_CHECK(!well_formed(m));
  m             = good;
  m.col_type[0] = 2;
  GPRS_CHECK(!well_formed(m));
  m       = good;
  m.nrows = -1;
  GPRS_CHECK(!well_formed(m));

  // Crossing bounds and sides are a model, an infeasible one.
  m              = good;
  m.col_lower[0] = 9.;
  m.row_lower[4] = 20.;
  GPRS_CHECK(well_formed(m));
}
