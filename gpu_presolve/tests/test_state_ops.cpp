/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "test_state_support.hpp"

#include <gprs/state.hpp>

#include <cmath>
#include <vector>

using namespace gprs;
using namespace state_test;

namespace {

// 5 rows, 5 columns, 12 nonzeros; CSR slots in brackets.
//   r0:  2 x0 [0] - 3 x1 [1] +   x2 [2]                 <= 10
//   r1:               x1 [3]            +   x3 [4]       =  5
//   r2:                      -   x2 [5] - 2 x3 [6] + x4/2 [7] >= 1
//   r3:  (empty)                                         in [-1, 1]
//   r4:    x0 [8] +   x1 [9] +   x2 [10]        + x4 [11] in [2, 8]
//   x0 integer in [0, 4], x1 in [-2, 3], x2 >= 0, x3 integer and free, x4 in [1, 1e21].
// Locks at the start: up {2, 2, 3, 2, 1}, down {1, 3, 1, 1, 2}.
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

using list_t = std::vector<int32_t>;

void check_rule_lists(const state_t& st, uint8_t events, const list_t& rows)
{
  for (int32_t l = 0; l < row_list::compaction; ++l) {
    GPRS_CHECK(st.row_frontier.list[l] == ((events >> l) & 1 ? rows : list_t()));
  }
}

void check_consistent(state_t& st) { GPRS_CHECK_EQ(st.consistency_error(), std::string()); }

}  // namespace

// Each bound enters one side per row, by the sign of the coefficient, and changes between
// finite and infinite through the counter and the slot sum.
GPRS_TEST(state_ops_col_bounds)
{
  state_t st(hand_model(), tolerances_t());
  clear_rule_lists(st);

  st.set_col_upper(2, 6.);
  check_side(st.max_act, 0, {20., 0., 0, 0, 8., 3});
  check_side(st.min_act, 2, {-5.5, 0., 1, 6, 6., 2});
  check_side(st.max_act, 4, {13., 0., 1, 11, 6., 3});
  check_side(st.min_act, 0, {-9., 0., 0, 0, 9., 3});
  check_side(st.max_act, 2, {0., 0., 2, 13, 0., 1});
  check_side(st.min_act, 4, {-1., 0., 0, 0, 2., 4});
  check_side(st.min_act, 1, {-2., 0., 1, 4, 2., 1});
  GPRS_CHECK_EQ(st.col_upper[2], 6.);
  GPRS_CHECK_EQ(st.bound_changes, 1);
  check_rule_lists(st, row_event::activity, {0, 2, 4});
  GPRS_CHECK(st.col_frontier.empty(col_list::dual_fixing));
  GPRS_CHECK(st.row_version == std::vector<uint32_t>(5, 0));
  check_consistent(st);

  st.set_col_lower(2, -inf);
  check_side(st.min_act, 0, {-9., 0., 1, 2, 9., 4});
  check_side(st.max_act, 2, {0., 0., 3, 18, 0., 2});
  check_side(st.min_act, 4, {-1., 0., 1, 10, 2., 5});
  check_consistent(st);

  st.set_col_lower(2, 2.);
  check_side(st.min_act, 0, {-7., 0., 0, 0, 9., 5});
  check_side(st.max_act, 2, {-2., 0., 2, 13, 2., 3});
  check_side(st.min_act, 4, {1., 0., 0, 0, 2., 6});
  check_consistent(st);

  // A negative coefficient: the upper bound is in the minimum activity.
  st.set_col_upper(1, 1.);
  check_side(st.min_act, 0, {-1., 0., 0, 0, 9., 7});
  check_side(st.max_act, 1, {1., 0., 1, 4, 3., 3});
  check_side(st.max_act, 4, {11., 0., 1, 11, 6., 5});
  check_side(st.max_act, 0, {20., 0., 0, 0, 8., 3});
  GPRS_CHECK_EQ(st.bound_changes, 4);
  GPRS_CHECK(st.up_locks == list_t({2, 2, 3, 2, 1}));
  GPRS_CHECK(st.down_locks == list_t({1, 3, 1, 1, 2}));
  check_consistent(st);
}

// The old term leaves and the new one enters. With a * (new - old) the 0.001 of the old bound
// would stay in the sum, because 1e19 - 0.001 rounds to 1e19.
GPRS_TEST(state_ops_bound_change_is_two_terms)
{
  model_t m;
  m.nrows     = 1;
  m.ncols     = 2;
  m.row_start = {0, 2};
  m.col_index = {0, 1};
  m.value     = {1., 1.};
  m.row_lower = {-inf};
  m.row_upper = {1.};
  m.col_lower = {0.001, 0.};
  m.col_upper = {1e19, 0.};
  m.cost      = {0., 0.};
  m.col_type  = {0, 0};
  state_t st(m, tolerances_t());
  GPRS_CHECK_EQ(st.min_activity(0), 0.001);
  st.set_col_lower(0, 1e19);
  GPRS_CHECK_EQ(st.min_act.hi[0], 1e19);
  GPRS_CHECK_EQ(st.min_act.lo[0], 0.);
  check_consistent(st);
  st.set_col_lower(0, 0.001);
  GPRS_CHECK_EQ(st.min_act.hi[0], 0.001);
  GPRS_CHECK_EQ(st.min_act.lo[0], 0.);
  GPRS_CHECK_EQ(st.min_act.updates[0], 6u);
  GPRS_CHECK_EQ(st.min_act.max_term[0], 1e19);
  check_consistent(st);
}

GPRS_TEST(state_ops_fix_col)
{
  state_t st(hand_model(), tolerances_t());
  clear_rule_lists(st);

  st.fix_col(1, 1.);
  check_side(st.min_act, 0, {-3., 0., 0, 0, 9., 5});
  check_side(st.max_act, 0, {5., 0., 1, 2, 8., 4});
  check_side(st.min_act, 1, {1., 0., 1, 4, 2., 3});
  check_side(st.max_act, 1, {1., 0., 1, 4, 3., 3});
  check_side(st.min_act, 4, {2., 0., 0, 0, 2., 6});
  check_side(st.max_act, 4, {5., 0., 2, 21, 4., 4});
  GPRS_CHECK_EQ(st.col_lower[1], 1.);
  GPRS_CHECK_EQ(st.col_upper[1], 1.);
  GPRS_CHECK_EQ(st.col_status[1], col_status_t::fixed);
  GPRS_CHECK(!st.is_live(1));
  GPRS_CHECK(st.row_free_size == list_t({2, 1, 3, 0, 3}));
  GPRS_CHECK(st.csr.size == list_t({3, 2, 3, 0, 4}));
  GPRS_CHECK_EQ(st.live_cols, 4);
  GPRS_CHECK_EQ(st.live_nnz, 12);
  GPRS_CHECK_EQ(st.fixed_nnz, 3);
  GPRS_CHECK_EQ(st.free_nnz(), 9);
  GPRS_CHECK_EQ(st.objective_offset.hi, 6.);
  GPRS_CHECK_EQ(st.objective_offset.lo, 0.);
  GPRS_CHECK_EQ(st.bound_changes, 2);
  check_rule_lists(st, row_event::entries, {0, 1, 4});
  GPRS_CHECK(st.col_frontier.empty(col_list::dual_fixing));
  GPRS_CHECK(st.up_locks == list_t({2, 2, 3, 2, 1}));
  GPRS_CHECK(st.down_locks == list_t({1, 3, 1, 1, 2}));
  GPRS_CHECK(st.row_version == std::vector<uint32_t>(5, 0));
  GPRS_CHECK_EQ(st.find_entry(0, 1), 1);
  check_consistent(st);

  // At a bound only the other bound's side changes.
  st.fix_col(0, 0.);
  check_side(st.min_act, 0, {-3., 0., 0, 0, 9., 5});
  check_side(st.max_act, 0, {-3., 0., 1, 2, 8., 6});
  GPRS_CHECK_EQ(st.bound_changes, 3);
  GPRS_CHECK_EQ(st.objective_offset.hi, 6.);
  GPRS_CHECK_EQ(st.fixed_nnz, 5);
  check_consistent(st);

  // A fixed term above the infinite-term limit is counted, not summed.
  st.fix_col(4, 1e21);
  check_side(st.min_act, 4, {1., 0., 1, 11, 2., 7});
  GPRS_CHECK_EQ(st.max_act.inf_count[4], 2);
  GPRS_CHECK_EQ(dd_value(st.objective_offset), 6. + 0.5e21);
  check_consistent(st);
}

GPRS_TEST(state_ops_delete_row)
{
  state_t st(hand_model(), tolerances_t());
  clear_rule_lists(st);
  st.fix_col(1, 1.);
  clear_rule_lists(st);

  st.delete_row(4);
  GPRS_CHECK(st.row_alive == std::vector<uint8_t>({1, 1, 1, 1, 0}));
  GPRS_CHECK_EQ(st.live_rows, 4);
  GPRS_CHECK_EQ(st.live_nnz, 8);
  GPRS_CHECK_EQ(st.fixed_nnz, 2);
  GPRS_CHECK(st.csr.size == list_t({3, 2, 3, 0, 0}));
  GPRS_CHECK(st.csr.used == list_t({3, 2, 3, 0, 4}));
  GPRS_CHECK(st.row_free_size == list_t({2, 1, 3, 0, 0}));
  GPRS_CHECK(st.csc.size == list_t({1, 2, 2, 2, 1}));
  GPRS_CHECK(st.csc.used == list_t({2, 3, 3, 2, 2}));
  for (int64_t e = 8; e < 12; ++e) {
    GPRS_CHECK(!st.csr.live(e));
  }
  for (int64_t p : {1, 4, 7, 11}) {
    GPRS_CHECK(!st.csc.live(p));
  }
  // Both sides of r4 are finite: one lock each way per column, fixed column included.
  GPRS_CHECK(st.up_locks == list_t({1, 1, 2, 2, 0}));
  GPRS_CHECK(st.down_locks == list_t({0, 2, 0, 1, 1}));
  check_side(st.min_act, 4, {0., 0., 0, 0, 0., 0});
  check_side(st.max_act, 4, {0., 0., 0, 0, 0., 0});
  // Events for the live columns only, and no row event.
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == list_t({0, 2, 4}));
  check_rule_lists(st, 0, {});
  GPRS_CHECK(st.col_frontier.empty(col_list::compaction));
  GPRS_CHECK_EQ(st.find_entry(4, 0), -1);
  GPRS_CHECK_EQ(st.dead_fraction(), 4. / 12.);
  check_consistent(st);

  // r0 has an upper side only. Columns 0 and 2 become sparse; column 1 is fixed and is not
  // listed.
  st.delete_row(0);
  GPRS_CHECK(st.up_locks == list_t({0, 1, 1, 2, 0}));
  GPRS_CHECK(st.down_locks == list_t({0, 1, 0, 1, 1}));
  GPRS_CHECK(st.col_frontier.list[col_list::compaction] == list_t({0, 2}));
  GPRS_CHECK_EQ(st.fixed_nnz, 1);
  GPRS_CHECK_EQ(st.live_nnz, 5);
  check_consistent(st);

  st.compact_sparse_lines();
  GPRS_CHECK(st.csc.used == list_t({0, 3, 1, 2, 2}));
  GPRS_CHECK(st.col_frontier.empty(col_list::compaction));
  GPRS_CHECK_EQ(st.csc.index[st.csc.begin(2)], 2);
  GPRS_CHECK_EQ(st.csr.twin[5], st.csc.begin(2));
  check_consistent(st);
}

GPRS_TEST(state_ops_row_sides)
{
  state_t st(hand_model(), tolerances_t());
  clear_rule_lists(st);

  // -inf to finite: the row starts to lock x0 and x2 down and x1 (negative entry) up.
  st.set_row_lower(0, 3.);
  GPRS_CHECK_EQ(st.row_lower[0], 3.);
  GPRS_CHECK(st.up_locks == list_t({2, 3, 3, 2, 1}));
  GPRS_CHECK(st.down_locks == list_t({2, 3, 2, 1, 2}));
  GPRS_CHECK_EQ(st.row_version[0], 1u);
  GPRS_CHECK_EQ(st.side_changes, 1);
  check_rule_lists(st, row_event::side, {0});
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == list_t({0, 1, 2}));
  check_consistent(st);

  // Finite to +inf on an equation: the up locks of positive entries go.
  st.set_row_upper(1, inf);
  GPRS_CHECK(st.up_locks == list_t({2, 2, 3, 1, 1}));
  GPRS_CHECK(st.down_locks == list_t({2, 3, 2, 1, 2}));
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == list_t({0, 1, 2, 3}));
  check_consistent(st);

  // Finite to finite: no lock and no column event.
  clear_rule_lists(st);
  st.set_row_upper(4, 9.);
  st.set_row_lower(4, -1.);
  GPRS_CHECK(st.up_locks == list_t({2, 2, 3, 1, 1}));
  GPRS_CHECK(st.down_locks == list_t({2, 3, 2, 1, 2}));
  GPRS_CHECK(st.col_frontier.empty(col_list::dual_fixing));
  GPRS_CHECK_EQ(st.row_version[4], 2u);
  GPRS_CHECK_EQ(st.side_changes, 4);
  check_rule_lists(st, row_event::side, {4});
  check_consistent(st);

  // Both finite sides move, an infinite side stays.
  st.shift_row_sides(4, 2., 1.5);
  GPRS_CHECK_EQ(st.row_lower[4], -4.);
  GPRS_CHECK_EQ(st.row_upper[4], 6.);
  GPRS_CHECK_EQ(st.row_version[4], 3u);
  GPRS_CHECK_EQ(st.side_changes, 6);
  st.shift_row_sides(2, -1., 4.);
  GPRS_CHECK_EQ(st.row_lower[2], 5.);
  GPRS_CHECK_EQ(st.row_upper[2], inf);
  GPRS_CHECK_EQ(st.side_changes, 7);
  check_rule_lists(st, row_event::side, {4, 2});

  // The product error is carried: 0.3 - 0.1 * 3 is -2^-55 exactly, and 0.3 minus the rounded
  // product is twice that.
  st.set_row_upper(3, 2.);
  st.set_row_lower(3, 0.3);
  st.shift_row_sides(3, 0.1, 3.);
  GPRS_CHECK_EQ(st.row_lower[3], -0x1p-55);
  GPRS_CHECK(0.3 - 0.1 * 3. != -0x1p-55);
  check_consistent(st);
}

GPRS_TEST(state_ops_scale_row)
{
  state_t st(hand_model(), tolerances_t());
  clear_rule_lists(st);

  st.scale_row(2, 2., false);
  GPRS_CHECK_EQ(st.csr.value[5], -2.);
  GPRS_CHECK_EQ(st.csr.value[6], -4.);
  GPRS_CHECK_EQ(st.csr.value[7], 1.);
  for (int64_t e = 5; e < 8; ++e) {
    GPRS_CHECK_EQ(st.csc.value[st.csr.twin[e]], st.csr.value[e]);
  }
  GPRS_CHECK_EQ(st.row_lower[2], 2.);
  GPRS_CHECK_EQ(st.row_upper[2], inf);
  check_side(st.min_act, 2, {1., 0., 2, 11, 1., 1});
  check_side(st.max_act, 2, {0., 0., 2, 13, 0., 1});
  GPRS_CHECK_EQ(st.row_version[2], 1u);
  GPRS_CHECK_EQ(st.coefficient_changes, 3);
  GPRS_CHECK_EQ(st.side_changes, 1);
  check_rule_lists(st, row_event::entries, {2});
  GPRS_CHECK(st.col_frontier.empty(col_list::dual_fixing));
  GPRS_CHECK(st.up_locks == list_t({2, 2, 3, 2, 1}));
  check_consistent(st);

  // integral: the products of the live columns become integers, the product of a fixed column
  // and the sides do not.
  model_t m;
  m.nrows     = 1;
  m.ncols     = 4;
  m.row_start = {0, 4};
  m.col_index = {0, 1, 2, 3};
  m.value     = {0.57, -0.07, 0.125, 1.1};
  m.row_lower = {0.125};
  m.row_upper = {0.57};
  m.col_lower = {0., 0., 0., 0.};
  m.col_upper = {1., 1., 1., 1.};
  m.cost      = {0., 0., 0., 0.};
  m.col_type  = {1, 1, 0, 1};
  state_t scaled(m, tolerances_t());
  scaled.fix_col(2, 1.);
  GPRS_CHECK(0.57 * 100. != 57. || -0.07 * 100. != -7. || 1.1 * 100. != 110.);
  scaled.scale_row(0, 100., true);
  GPRS_CHECK_EQ(scaled.csr.value[0], 57.);
  GPRS_CHECK_EQ(scaled.csr.value[1], -7.);
  GPRS_CHECK_EQ(scaled.csr.value[2], 12.5);
  GPRS_CHECK_EQ(scaled.csr.value[3], 110.);
  GPRS_CHECK_EQ(scaled.row_lower[0], 12.5);
  GPRS_CHECK_EQ(scaled.row_upper[0], 0.57 * 100.);
  GPRS_CHECK_EQ(scaled.min_activity(0), -7. + 12.5);
  GPRS_CHECK_EQ(scaled.max_activity(0), 57. + 12.5 + 110.);
  check_consistent(scaled);
}

GPRS_TEST(state_ops_set_coefficient)
{
  state_t st(hand_model(), tolerances_t());
  clear_rule_lists(st);

  // -3 x1 becomes 4 x1 in r0 (<= 10): each activity takes the other bound, the down lock
  // becomes an up lock.
  st.set_coefficient(1, 4.);
  GPRS_CHECK_EQ(st.csr.value[1], 4.);
  GPRS_CHECK_EQ(st.csc.value[2], 4.);
  check_side(st.min_act, 0, {-8., 0., 0, 0, 9., 5});
  check_side(st.max_act, 0, {20., 0., 1, 2, 12., 4});
  GPRS_CHECK(st.up_locks == list_t({2, 3, 3, 2, 1}));
  GPRS_CHECK(st.down_locks == list_t({1, 2, 1, 1, 2}));
  GPRS_CHECK_EQ(st.row_version[0], 1u);
  GPRS_CHECK_EQ(st.coefficient_changes, 1);
  check_rule_lists(st, row_event::entries, {0});
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == list_t({1}));
  check_consistent(st);

  // An entry on an infinite bound: with the sign the infinite term changes sides.
  st.set_coefficient(2, -5.);
  check_side(st.min_act, 0, {-8., 0., 1, 2, 9., 6});
  check_side(st.max_act, 0, {20., 0., 0, 0, 12., 5});
  check_consistent(st);
}

GPRS_TEST(state_ops_delete_entry)
{
  state_t st(hand_model(), tolerances_t());
  clear_rule_lists(st);

  st.delete_entry(6);
  check_side(st.min_act, 2, {0.5, 0., 1, 5, 0.5, 1});
  check_side(st.max_act, 2, {0., 0., 1, 7, 0., 1});
  GPRS_CHECK(st.up_locks == list_t({2, 2, 3, 1, 1}));
  GPRS_CHECK(st.down_locks == list_t({1, 3, 1, 1, 2}));
  GPRS_CHECK(!st.csr.live(6));
  GPRS_CHECK(!st.csc.live(9));
  GPRS_CHECK_EQ(st.csr.size[2], 2);
  GPRS_CHECK_EQ(st.csr.used[2], 3);
  GPRS_CHECK_EQ(st.csc.size[3], 1);
  GPRS_CHECK_EQ(st.row_free_size[2], 2);
  GPRS_CHECK_EQ(st.live_nnz, 11);
  GPRS_CHECK_EQ(st.row_version[2], 1u);
  GPRS_CHECK_EQ(st.find_entry(2, 3), -1);
  GPRS_CHECK_EQ(st.find_entry(2, 4), 7);
  check_rule_lists(st, row_event::entries, {2});
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == list_t({3}));
  GPRS_CHECK(st.row_frontier.empty(row_list::compaction));
  GPRS_CHECK(st.col_frontier.empty(col_list::compaction));
  check_consistent(st);

  // Two tombstones against one live entry: the row goes on the compaction list.
  st.delete_entry(5);
  check_side(st.min_act, 2, {0.5, 0., 0, 0, 0.5, 1});
  check_side(st.max_act, 2, {0., 0., 1, 7, 0., 2});
  GPRS_CHECK(st.up_locks == list_t({2, 2, 2, 1, 1}));
  GPRS_CHECK(st.row_frontier.list[row_list::compaction] == list_t({2}));
  GPRS_CHECK(st.col_frontier.empty(col_list::compaction));
  check_consistent(st);

  st.compact_sparse_lines();
  GPRS_CHECK_EQ(st.csr.used[2], 1);
  GPRS_CHECK_EQ(st.csr.index[5], 4);
  GPRS_CHECK_EQ(st.csr.value[5], 0.5);
  GPRS_CHECK_EQ(st.find_entry(2, 4), 5);
  GPRS_CHECK_EQ(st.csc.twin[st.csr.twin[5]], 5);
  check_side(st.max_act, 2, {0., 0., 1, 5, 0., 0});
  check_side(st.min_act, 2, {0.5, 0., 0, 0, 0.5, 1});
  GPRS_CHECK(st.row_frontier.empty(row_list::compaction));
  check_consistent(st);
}

GPRS_TEST(state_ops_move_entry)
{
  state_t st(hand_model(), tolerances_t());
  clear_rule_lists(st);

  // r1: x1 + x3 = 5 becomes x1 + 3 x0 = 5. Column 0 is full and moves to the end of the store.
  st.move_entry(4, 0, 3.);
  GPRS_CHECK_EQ(st.csr.index[4], 0);
  GPRS_CHECK_EQ(st.csr.value[4], 3.);
  GPRS_CHECK_EQ(st.csr.twin[4], 14);
  check_side(st.min_act, 1, {-2., 0., 0, 0, 2., 2});
  check_side(st.max_act, 1, {15., 0., 0, 0, 12., 2});
  GPRS_CHECK(st.up_locks == list_t({3, 2, 3, 1, 1}));
  GPRS_CHECK(st.down_locks == list_t({2, 3, 1, 0, 2}));
  GPRS_CHECK_EQ(st.csc.slots(), 16);
  GPRS_CHECK_EQ(st.csc.start[0], 12);
  GPRS_CHECK_EQ(st.csc.capacity[0], 4);
  GPRS_CHECK_EQ(st.csc.used[0], 3);
  GPRS_CHECK_EQ(st.csc.size[0], 3);
  GPRS_CHECK(list_t(st.csc.index.begin() + 12, st.csc.index.end()) == list_t({0, 4, 1, -1}));
  GPRS_CHECK_EQ(st.csc.value[12], 2.);
  GPRS_CHECK_EQ(st.csc.value[13], 1.);
  GPRS_CHECK_EQ(st.csc.value[14], 3.);
  GPRS_CHECK_EQ(st.csr.twin[0], 12);
  GPRS_CHECK_EQ(st.csr.twin[8], 13);
  GPRS_CHECK(!st.csc.live(0) && !st.csc.live(1) && !st.csc.live(8));
  GPRS_CHECK_EQ(st.csc.size[3], 1);
  GPRS_CHECK_EQ(st.csc.used[3], 2);
  GPRS_CHECK(st.csr.size == list_t({3, 2, 3, 0, 4}));
  GPRS_CHECK(st.row_free_size == list_t({3, 2, 3, 0, 4}));
  GPRS_CHECK_EQ(st.live_nnz, 12);
  GPRS_CHECK_EQ(st.row_version[1], 1u);
  GPRS_CHECK_EQ(st.coefficient_changes, 1);
  GPRS_CHECK_EQ(st.find_entry(1, 0), 4);
  GPRS_CHECK_EQ(st.find_entry(1, 3), -1);
  GPRS_CHECK_EQ(st.dead_fraction(), 4. / 16.);
  check_rule_lists(st, row_event::entries, {1});
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == list_t({3, 0}));
  check_consistent(st);

  // The moved column has a spare slot: the next entry goes there.
  st.move_entry(7, 0, -1.);
  GPRS_CHECK_EQ(st.csc.slots(), 16);
  GPRS_CHECK_EQ(st.csc.start[0], 12);
  GPRS_CHECK_EQ(st.csc.used[0], 4);
  GPRS_CHECK_EQ(st.csc.index[15], 2);
  GPRS_CHECK_EQ(st.csr.twin[7], 15);
  // r2 (>= 1): -x2 - 2 x3 - x0. The term of x4 was infinite in the maximum only.
  check_side(st.min_act, 2, {-4., 0., 2, 11, 4., 3});
  check_side(st.max_act, 2, {0., 0., 1, 6, 0., 2});
  GPRS_CHECK(st.up_locks == list_t({4, 2, 3, 1, 1}));
  GPRS_CHECK(st.down_locks == list_t({2, 3, 1, 0, 1}));
  check_consistent(st);
}

// A column that keeps receiving entries doubles its capacity each time it is full, so n moves
// cost O(n) slots in all.
GPRS_TEST(state_ops_move_entry_amortised)
{
  const int32_t n = 64;
  model_t m;
  m.nrows = n;
  m.ncols = n + 1;
  m.row_start.push_back(0);
  for (int32_t i = 0; i < n; ++i) {
    m.col_index.push_back(i + 1);
    m.value.push_back(1.);
    m.row_start.push_back(i + 1);
  }
  m.row_lower.assign(n, 0.);
  m.row_upper.assign(n, 1.);
  m.col_lower.assign(n + 1, 0.);
  m.col_upper.assign(n + 1, 1.);
  m.cost.assign(n + 1, 0.);
  m.col_type.assign(n + 1, 0);
  GPRS_REQUIRE(well_formed(m));
  state_t st(m, tolerances_t());
  list_t capacities;
  for (int32_t i = 0; i < n; ++i) {
    st.move_entry(i, 0, 2.);
    if (capacities.empty() || capacities.back() != st.csc.capacity[0]) {
      capacities.push_back(st.csc.capacity[0]);
    }
  }
  GPRS_CHECK(capacities == list_t({4, 8, 16, 32, 64}));
  GPRS_CHECK_EQ(st.csc.size[0], n);
  GPRS_CHECK_EQ(st.csc.used[0], n);
  GPRS_CHECK_EQ(st.csc.slots(), n + 4 + 8 + 16 + 32 + 64);
  GPRS_CHECK(st.slot_visits <= 3 * n);
  for (int32_t i = 0; i < n; ++i) {
    GPRS_CHECK_EQ(st.csc.index[st.csc.begin(0) + i], i);
    GPRS_CHECK_EQ(st.csr.twin[i], st.csc.begin(0) + i);
  }
  check_consistent(st);
}

GPRS_TEST(state_ops_find_entry)
{
  model_t m = hand_model();
  state_t st(m, tolerances_t());
  for (int32_t i = 0; i < m.nrows; ++i) {
    for (int32_t j = 0; j < m.ncols; ++j) {
      int64_t expected = -1;
      for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
        if (m.col_index[e] == j) { expected = e; }
      }
      GPRS_CHECK_EQ(st.find_entry(i, j), expected);
    }
  }
}

GPRS_TEST(state_ops_implied_bounds)
{
  state_t st(hand_model(), tolerances_t());
  GPRS_CHECK(!st.implied_lower_valid(1));
  GPRS_CHECK(!st.implied_upper_valid(1));

  st.record_implied_lower(1, 0.5, 0);
  st.record_implied_upper(2, 7., 4);
  GPRS_CHECK(st.implied_lower_valid(1));
  GPRS_CHECK(!st.implied_upper_valid(1));
  GPRS_CHECK(st.implied_upper_valid(2));
  GPRS_CHECK_EQ(st.implied_lower[1], 0.5);
  GPRS_CHECK_EQ(st.implied_lower_row[1], 0);
  GPRS_CHECK_EQ(st.implied_upper[2], 7.);
  GPRS_CHECK_EQ(st.implied_upper_row[2], 4);
  // The explicit bounds and the activities do not move.
  GPRS_CHECK_EQ(st.col_lower[1], -2.);
  GPRS_CHECK_EQ(st.min_act.updates[0], 3u);

  // A bound of another column of the source row leaves the record valid; an entry or a side of
  // the row voids it.
  st.set_col_upper(0, 3.);
  st.fix_col(2, 1.);
  GPRS_CHECK(st.implied_lower_valid(1));
  st.set_coefficient(0, 5.);
  GPRS_CHECK(!st.implied_lower_valid(1));
  GPRS_CHECK(st.implied_upper_valid(2));
  st.record_implied_lower(1, 0.25, 0);
  GPRS_CHECK(st.implied_lower_valid(1));
  GPRS_CHECK_EQ(st.implied_lower_version[1], 1u);

  st.set_row_upper(4, 7.5);
  GPRS_CHECK(!st.implied_upper_valid(2));
  st.record_implied_upper(1, 2., 4);
  st.rebuild();
  GPRS_CHECK(st.implied_lower_valid(1));
  GPRS_CHECK(st.implied_upper_valid(1));
  st.delete_row(4);
  GPRS_CHECK(!st.implied_upper_valid(1));
  GPRS_CHECK(st.implied_lower_valid(1));
  st.delete_row(0);
  GPRS_CHECK(!st.implied_lower_valid(1));
  check_consistent(st);
}

GPRS_TEST(state_ops_cost_offset_remove)
{
  state_t st(hand_model(), tolerances_t());
  clear_rule_lists(st);

  st.set_cost(3, -4.);
  GPRS_CHECK_EQ(st.cost[3], -4.);
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == list_t({3}));

  double error;
  double product = two_prod(0.1, 3., error);
  GPRS_CHECK(error != 0.);
  st.add_objective_offset(0.1, 3.);
  GPRS_CHECK_EQ(st.objective_offset.hi, 7. + product);
  GPRS_CHECK(st.objective_offset.lo != 0.);
  st.add_objective_offset(-0.1, 3.);
  GPRS_CHECK_EQ(dd_value(st.objective_offset), 7.);

  st.delete_row(1);
  st.delete_row(2);
  GPRS_CHECK_EQ(st.csc.size[3], 0);
  st.set_cost(3, 0.);
  int64_t visits = st.slot_visits;
  st.remove_col(3);
  GPRS_CHECK_EQ(st.slot_visits, visits);
  GPRS_CHECK_EQ(st.col_status[3], col_status_t::removed);
  GPRS_CHECK_EQ(st.live_cols, 4);
  GPRS_CHECK_EQ(st.live_nnz, 7);
  check_consistent(st);
}

// A row is summed again when a side's error bound is above the share and the side has twice
// the row's size in updates, and not before.
GPRS_TEST(state_ops_resummation_rule)
{
  tolerances_t tol;
  tol.activity_error_share = 1e-30;
  state_t st(hand_model(), tol);
  st.set_col_lower(0, 1.);
  GPRS_CHECK_EQ(st.min_act.updates[0], 5u);
  GPRS_CHECK_EQ(st.min_act.updates[4], 6u);
  GPRS_CHECK_EQ(st.max_act.updates[0], 2u);
  st.set_col_lower(0, 2.);
  check_side(st.min_act, 0, {-5., 0., 0, 0, 9., 3});
  check_side(st.max_act, 0, {14., 0., 1, 2, 8., 2});
  check_side(st.min_act, 4, {1., 0., 0, 0, 2., 4});
  check_consistent(st);

  state_t plain(hand_model(), tolerances_t());
  plain.set_col_lower(0, 1.);
  plain.set_col_lower(0, 2.);
  check_side(plain.min_act, 0, {-5., 0., 0, 0, 9., 7});
  check_side(plain.min_act, 4, {1., 0., 0, 0, 2., 8});
  check_consistent(plain);
}
