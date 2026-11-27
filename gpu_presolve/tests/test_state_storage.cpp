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
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace gprs;
using namespace state_test;

namespace {

// The model of test_state_ops.cpp; CSR slots in brackets.
//   r0:  2 x0 [0] - 3 x1 [1] +   x2 [2]                 <= 10
//   r1:               x1 [3]            +   x3 [4]       =  5
//   r2:                      -   x2 [5] - 2 x3 [6] + x4/2 [7] >= 1
//   r3:  (empty)                                         in [-1, 1]
//   r4:    x0 [8] +   x1 [9] +   x2 [10]        + x4 [11] in [2, 8]
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

using list_t  = std::vector<int32_t>;
using slots_t = std::vector<int64_t>;

void check_consistent(state_t& st) { GPRS_CHECK_EQ(st.consistency_error(), std::string()); }

int64_t differing(const list_t& a, const list_t& b)
{
  int64_t count = 0;
  for (size_t k = 0; k < a.size(); ++k) {
    count += a[k] != b[k];
  }
  return count;
}

}  // namespace

// Deleting a row costs its own entries, however long its columns are, and writes nothing else.
GPRS_TEST(state_storage_deletion_cost)
{
  model_t m;
  m.nrows = 4000;
  m.ncols = 50;
  m.row_start.push_back(0);
  for (int32_t i = 0; i < m.nrows; ++i) {
    m.col_index.insert(m.col_index.end(), {0, 1 + i % 24, 25 + i % 25});
    m.value.insert(m.value.end(), {1., 1., -1.});
    m.row_start.push_back(m.col_index.size());
  }
  m.row_lower.assign(m.nrows, -inf);
  m.row_upper.assign(m.nrows, 1.);
  m.col_lower.assign(m.ncols, 0.);
  m.col_upper.assign(m.ncols, 1.);
  m.cost.assign(m.ncols, 0.);
  m.col_type.assign(m.ncols, 0);
  GPRS_REQUIRE(well_formed(m));
  state_t st(m, tolerances_t());
  GPRS_CHECK_EQ(st.csc.used[0], 4000);
  GPRS_CHECK_EQ(st.slot_visits, 0);

  for (int32_t i = 0; i < m.nrows; i += 40) {
    list_t csr_before = st.csr.index, csc_before = st.csc.index;
    int64_t visits = st.slot_visits;
    st.delete_row(i);
    GPRS_CHECK_EQ(st.slot_visits - visits, 3);
    GPRS_CHECK_EQ(differing(csr_before, st.csr.index), 3);
    GPRS_CHECK_EQ(differing(csc_before, st.csc.index), 3);
  }
  GPRS_CHECK_EQ(st.slot_visits, 300);
  GPRS_CHECK_EQ(st.csc.size[0], 3900);
  GPRS_CHECK_EQ(st.live_rows, 3900);
  GPRS_CHECK_EQ(st.up_locks[0], 3900);
  check_consistent(st);

  // An entry of the long column is deleted or changed at the cost of one slot.
  int64_t slot   = st.find_entry(1, 0);
  int64_t visits = st.slot_visits;
  st.delete_entry(slot);
  GPRS_CHECK_EQ(st.slot_visits - visits, 1);
  slot   = st.find_entry(2, 0);
  visits = st.slot_visits;
  st.set_coefficient(slot, -2.);
  GPRS_CHECK_EQ(st.slot_visits - visits, 1);
  visits = st.slot_visits;
  st.set_row_upper(3, 2.);
  st.shift_row_sides(3, 1., 1.);
  GPRS_CHECK_EQ(st.slot_visits - visits, 0);

  // A bound change and a fixing walk the column once.
  visits = st.slot_visits;
  st.set_col_upper(7, 0.5);
  GPRS_CHECK_EQ(st.slot_visits - visits, st.csc.used[7]);
  visits = st.slot_visits;
  st.fix_col(0, 1.);
  GPRS_CHECK_EQ(st.slot_visits - visits, 4000);
  GPRS_CHECK_EQ(st.fixed_nnz, 3899);
  check_consistent(st);

  // A side that changes between finite and infinite walks the row, not its columns.
  visits = st.slot_visits;
  st.set_row_lower(5, 0.);
  GPRS_CHECK_EQ(st.slot_visits - visits, 3);
  check_consistent(st);
}

// find_entry scans the shorter of the row and the column.
GPRS_TEST(state_storage_find_entry_cost)
{
  // Row 0 has an entry in each of the 100 columns; rows 1 to 199 have one entry, in column 0.
  model_t m;
  m.nrows = 200;
  m.ncols = 100;
  m.row_start.push_back(0);
  for (int32_t j = 0; j < m.ncols; ++j) {
    m.col_index.push_back(j);
    m.value.push_back(1.);
  }
  m.row_start.push_back(m.col_index.size());
  for (int32_t i = 1; i < m.nrows; ++i) {
    m.col_index.push_back(0);
    m.value.push_back(2.);
    m.row_start.push_back(m.col_index.size());
  }
  m.row_lower.assign(m.nrows, 0.);
  m.row_upper.assign(m.nrows, 1.);
  m.col_lower.assign(m.ncols, 0.);
  m.col_upper.assign(m.ncols, 1.);
  m.cost.assign(m.ncols, 0.);
  m.col_type.assign(m.ncols, 0);
  GPRS_REQUIRE(well_formed(m));
  const state_t st(m, tolerances_t());

  // Long row, column of one entry: the column is scanned.
  int64_t visits = st.slot_visits;
  GPRS_CHECK_EQ(st.find_entry(0, 57), 57);
  GPRS_CHECK_EQ(st.slot_visits - visits, 1);
  // Row of one entry, long column: the row is scanned.
  visits = st.slot_visits;
  GPRS_CHECK_EQ(st.find_entry(150, 0), 100 + 149);
  GPRS_CHECK_EQ(st.slot_visits - visits, 1);
  visits = st.slot_visits;
  GPRS_CHECK_EQ(st.find_entry(150, 57), -1);
  GPRS_CHECK_EQ(st.slot_visits - visits, 1);
  // Row of 100 against column of 200: at most the row.
  visits = st.slot_visits;
  GPRS_CHECK_EQ(st.find_entry(0, 0), 0);
  GPRS_CHECK(st.slot_visits - visits <= 100);
}

GPRS_TEST(state_storage_rebuild)
{
  state_t st(hand_model(), tolerances_t());
  st.fix_col(1, 1.);
  st.delete_row(2);
  st.delete_entry(8);
  st.record_implied_lower(2, 0.5, 0);
  clear_rule_lists(st);
  st.set_col_upper(0, 3.);
  st.set_cost(4, 1.);
  std::vector<list_t> row_lists = st.row_frontier.list, col_lists = st.col_frontier.list;
  std::vector<uint32_t> versions = st.row_version;
  check_consistent(st);

  st.rebuild();
  check_consistent(st);
  GPRS_CHECK(st.csr.start == slots_t({0, 2, 3, 3, 3}));
  GPRS_CHECK(st.csr.size == list_t({2, 1, 0, 0, 2}));
  GPRS_CHECK(st.csr.used == st.csr.size);
  GPRS_CHECK(st.csr.capacity == st.csr.size);
  GPRS_CHECK(st.row_free_size == st.csr.size);
  GPRS_CHECK(st.csr.index == list_t({0, 2, 3, 2, 4}));
  GPRS_CHECK(st.csr.value == std::vector<double>({2., 1., 1., 1., 1.}));
  GPRS_CHECK(st.csr.twin == slots_t({0, 1, 3, 2, 4}));
  GPRS_CHECK(st.csc.start == slots_t({0, 1, 1, 3, 4}));
  GPRS_CHECK(st.csc.size == list_t({1, 0, 2, 1, 1}));
  GPRS_CHECK(st.csc.used == st.csc.size);
  GPRS_CHECK(st.csc.capacity == st.csc.size);
  GPRS_CHECK(st.csc.index == list_t({0, 0, 4, 1, 4}));
  GPRS_CHECK(st.csc.twin == slots_t({0, 1, 3, 2, 4}));

  // The term of the fixed column leaves the finite sides: -3 * 1 from r0, 1 * 1 from r1 and r4.
  GPRS_CHECK(st.row_lower == std::vector<double>({-inf, 4., 1., -1., 1.}));
  GPRS_CHECK(st.row_upper == std::vector<double>({13., 4., inf, 1., 7.}));
  GPRS_CHECK_EQ(st.col_status[1], col_status_t::removed);
  GPRS_CHECK(st.row_alive == std::vector<uint8_t>({1, 1, 0, 1, 1}));
  GPRS_CHECK(st.up_locks == list_t({1, 0, 2, 1, 1}));
  GPRS_CHECK(st.down_locks == list_t({0, 0, 1, 1, 1}));
  GPRS_CHECK_EQ(st.live_rows, 4);
  GPRS_CHECK_EQ(st.live_cols, 4);
  GPRS_CHECK_EQ(st.live_nnz, 5);
  GPRS_CHECK_EQ(st.fixed_nnz, 0);
  GPRS_CHECK_EQ(st.dead_fraction(), 0.);
  GPRS_CHECK_EQ(st.objective_offset.hi, 6.);

  check_side(st.min_act, 0, {0., 0., 0, 0, 0., 2});
  check_side(st.max_act, 0, {6., 0., 1, 1, 6., 1});
  check_side(st.min_act, 1, {0., 0., 1, 2, 0., 0});
  check_side(st.max_act, 1, {0., 0., 1, 2, 0., 0});
  check_side(st.min_act, 4, {1., 0., 0, 0, 1., 2});
  check_side(st.max_act, 4, {0., 0., 2, 7, 0., 0});

  // Lists, versions and implied bounds are kept; the compaction lists are served.
  for (int32_t l = 0; l < row_list::compaction; ++l) {
    GPRS_CHECK(st.row_frontier.list[l] == row_lists[l]);
  }
  GPRS_CHECK(st.row_frontier.list[row_list::propagation] == list_t({0}));
  GPRS_CHECK(st.col_frontier.list[col_list::dual_fixing] == col_lists[col_list::dual_fixing]);
  GPRS_CHECK(st.row_frontier.empty(row_list::compaction));
  GPRS_CHECK(st.col_frontier.empty(col_list::compaction));
  GPRS_CHECK(st.row_version == versions);
  GPRS_CHECK(st.implied_lower_valid(2));

  std::vector<int32_t> row_origin, col_origin;
  model_t reduced = extract_model(st, row_origin, col_origin);
  GPRS_CHECK(well_formed(reduced));
  GPRS_CHECK(row_origin == list_t({0, 1, 3, 4}));
  GPRS_CHECK(col_origin == list_t({0, 2, 3, 4}));
  GPRS_CHECK_EQ(reduced.nrows, 4);
  GPRS_CHECK_EQ(reduced.ncols, 4);
  GPRS_CHECK(reduced.row_start == slots_t({0, 2, 3, 3, 5}));
  GPRS_CHECK(reduced.col_index == list_t({0, 1, 2, 1, 3}));
  GPRS_CHECK(reduced.value == std::vector<double>({2., 1., 1., 1., 1.}));
  GPRS_CHECK(reduced.row_lower == std::vector<double>({-inf, 4., -1., 1.}));
  GPRS_CHECK(reduced.row_upper == std::vector<double>({13., 4., 1., 7.}));
  GPRS_CHECK(reduced.col_lower == std::vector<double>({0., 0., -inf, 1.}));
  GPRS_CHECK(reduced.col_upper == std::vector<double>({3., inf, inf, 1e21}));
  GPRS_CHECK(reduced.cost == std::vector<double>({1., 0., 2., 1.}));
  GPRS_CHECK(reduced.col_type == std::vector<uint8_t>({1, 0, 1, 0}));
  GPRS_CHECK_EQ(reduced.objective_offset, 6.);

  // A second rebuild finds nothing to do.
  state_t again = st;
  again.rebuild();
  GPRS_CHECK(again.csr.index == st.csr.index && again.csr.twin == st.csr.twin);
  GPRS_CHECK(again.csc.index == st.csc.index && again.csc.twin == st.csc.twin);
  GPRS_CHECK(same_doubles(again.min_act.hi, st.min_act.hi));
  GPRS_CHECK(again.max_act.inf_slot == st.max_act.inf_slot);
  GPRS_CHECK(again.max_act.updates == st.max_act.updates);
  GPRS_CHECK(same_doubles(again.row_upper, st.row_upper));
}

// A row that keeps all its slots is not summed again: its sums and update counts stay and the
// slot sums follow the row.
GPRS_TEST(state_storage_rebuild_shifts_inf_slot)
{
  state_t st(hand_model(), tolerances_t());
  st.set_col_upper(1, 2.);
  check_side(st.max_act, 1, {2., 0., 1, 4, 3., 3});
  st.delete_row(0);
  st.rebuild();
  GPRS_CHECK(st.csr.start == slots_t({0, 0, 2, 5, 5}));
  check_side(st.min_act, 1, {-2., 0., 1, 1, 2., 1});
  check_side(st.max_act, 1, {2., 0., 1, 1, 3., 3});
  check_side(st.min_act, 2, {0.5, 0., 2, 5, 0.5, 1});
  check_side(st.max_act, 2, {0., 0., 2, 7, 0., 1});
  check_side(st.max_act, 4, {6., 0., 2, 15, 4., 4});
  GPRS_CHECK_EQ(st.csr.index[st.min_act.inf_slot[1]], 3);
  check_consistent(st);
}

// The rows of the extracted model are sorted by column whatever the storage order.
GPRS_TEST(state_storage_extract_sorts_rows)
{
  state_t st(hand_model(), tolerances_t());
  st.move_entry(4, 0, 3.);
  st.move_entry(1, 4, -6.);
  st.rebuild();
  check_consistent(st);
  GPRS_CHECK_EQ(st.csr.index[st.csr.begin(1)], 1);
  GPRS_CHECK_EQ(st.csr.index[st.csr.begin(1) + 1], 0);
  GPRS_CHECK(list_t(st.csr.index.begin(), st.csr.index.begin() + 3) == list_t({0, 4, 2}));
  // Column 0 holds its entries in the order they came: r0, r4, r1.
  GPRS_CHECK(list_t(st.csc.index.begin(), st.csc.index.begin() + 3) == list_t({0, 4, 1}));

  std::vector<int32_t> row_origin, col_origin;
  model_t reduced = extract_model(st, row_origin, col_origin);
  GPRS_CHECK(well_formed(reduced));
  GPRS_CHECK(reduced.row_start == slots_t({0, 3, 5, 8, 8, 12}));
  GPRS_CHECK(reduced.col_index == list_t({0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 2, 4}));
  GPRS_CHECK(reduced.value ==
             std::vector<double>({2., 1., -6., 3., 1., -1., -2., 0.5, 1., 1., 1., 1.}));
  GPRS_CHECK(row_origin == list_t({0, 1, 2, 3, 4}));
  GPRS_CHECK(col_origin == list_t({0, 1, 2, 3, 4}));
}

// A state that was never changed gives its model back, and a rebuild of it changes nothing.
GPRS_TEST(state_storage_extract_round_trip)
{
  rng_t rng(31);
  for (int32_t k = 0; k < 40; ++k) {
    model_options_t shape;
    shape.nrows   = rng.below(15);
    shape.ncols   = rng.below(15);
    shape.density = 0.4;
    model_t m     = random_model(rng, shape);
    GPRS_REQUIRE(well_formed(m));
    state_t st(m, tolerances_t());
    std::vector<int32_t> row_origin, col_origin;
    GPRS_CHECK(same_model(extract_model(st, row_origin, col_origin), m));
    state_t rebuilt = st;
    rebuilt.rebuild();
    check_consistent(rebuilt);
    GPRS_CHECK(rebuilt.csr.index == st.csr.index && rebuilt.csr.twin == st.csr.twin);
    GPRS_CHECK(rebuilt.csc.index == st.csc.index && rebuilt.csc.twin == st.csc.twin);
    GPRS_CHECK(rebuilt.csc.start == st.csc.start && rebuilt.csr.start == st.csr.start);
    GPRS_CHECK(same_doubles(rebuilt.min_act.hi, st.min_act.hi));
    GPRS_CHECK(same_doubles(rebuilt.max_act.lo, st.max_act.lo));
    GPRS_CHECK(rebuilt.min_act.inf_slot == st.min_act.inf_slot);
    GPRS_CHECK(same_model(extract_model(rebuilt, row_origin, col_origin), m));
    GPRS_CHECK_EQ((int32_t)row_origin.size(), m.nrows);
    GPRS_CHECK_EQ((int32_t)col_origin.size(), m.ncols);
  }
}

// Compaction keeps the order of the live entries, moves the twins and sums the row again.
GPRS_TEST(state_storage_compaction)
{
  model_t m;
  m.nrows     = 2;
  m.ncols     = 8;
  m.row_start = {0, 8, 10};
  m.col_index = {0, 1, 2, 3, 4, 5, 6, 7, 2, 5};
  m.value     = {1., 2., 3., 4., 5., 6., 7., 8., -1., -1.};
  m.row_lower = {-inf, 0.};
  m.row_upper = {100., 0.};
  m.col_lower = {0., 0., 0., 0., 0., -inf, 0., 0.};
  m.col_upper = {1., 1., 1., 1., 1., 1., 1., inf};
  m.cost.assign(8, 0.);
  m.col_type.assign(8, 0);
  state_t st(m, tolerances_t());
  for (int64_t e : {0, 1, 3, 4, 6}) {
    st.delete_entry(e);
  }
  GPRS_CHECK(st.row_frontier.list[row_list::compaction] == list_t({0}));
  GPRS_CHECK_EQ(st.csr.used[0], 8);
  GPRS_CHECK_EQ(st.max_act.inf_slot[0], 7);
  GPRS_CHECK_EQ(st.min_act.inf_slot[0], 5);
  GPRS_CHECK(st.min_act.updates[0] > 2u);
  // A rebuild serves the compaction lists as well.
  state_t rebuilt = st;
  rebuilt.rebuild();
  GPRS_CHECK(rebuilt.row_frontier.empty(row_list::compaction));
  GPRS_CHECK(rebuilt.col_frontier.empty(col_list::compaction));
  GPRS_CHECK_EQ(rebuilt.row_frontier.pending[0] & row_event::sparse, 0);
  check_consistent(rebuilt);

  int64_t visits = st.slot_visits;
  st.compact_sparse_lines();
  GPRS_CHECK(st.slot_visits - visits <= 2 * 8 + 5);
  GPRS_CHECK_EQ(st.csr.used[0], 3);
  GPRS_CHECK_EQ(st.csr.capacity[0], 8);
  GPRS_CHECK(list_t(st.csr.index.begin(), st.csr.index.begin() + 8) ==
             list_t({2, 5, 7, -1, -1, -1, -1, -1}));
  GPRS_CHECK_EQ(st.csr.value[0], 3.);
  GPRS_CHECK_EQ(st.csr.value[1], 6.);
  GPRS_CHECK_EQ(st.csr.value[2], 8.);
  GPRS_CHECK_EQ(st.max_act.inf_slot[0], 2);
  GPRS_CHECK_EQ(st.min_act.inf_slot[0], 1);
  check_side(st.max_act, 0, {9., 0., 1, 2, 6., 2});
  check_side(st.min_act, 0, {0., 0., 1, 1, 0., 2});
  GPRS_CHECK_EQ(st.find_entry(0, 5), 1);
  GPRS_CHECK_EQ(st.csc.twin[st.csr.twin[2]], 2);
  // The columns that lost their only entry are compacted too; row 1 and its columns stay.
  GPRS_CHECK(st.csc.used == list_t({0, 0, 2, 0, 0, 2, 0, 1}));
  GPRS_CHECK(st.row_frontier.empty(row_list::compaction));
  GPRS_CHECK(st.col_frontier.empty(col_list::compaction));
  check_consistent(st);
}

// consistency_error names the invariant that a corrupted state breaks.
GPRS_TEST(state_storage_consistency_detects)
{
  state_t base(hand_model(), tolerances_t());
  base.fix_col(1, 1.);
  base.delete_row(2);
  base.move_entry(4, 0, 3.);
  base.delete_entry(8);
  base.delete_entry(10);
  base.delete_entry(11);
  GPRS_REQUIRE(base.consistency_error().empty());
  GPRS_REQUIRE(base.csc.start[0] == 12 && base.csc.capacity[0] == 4 && base.csc.used[0] == 3);
  GPRS_REQUIRE(base.row_frontier.pending[4] & row_event::sparse);
  GPRS_REQUIRE(base.max_act.inf_count[0] == 1);

  struct corruption_t {
    const char* keyword;
    std::function<void(state_t&)> apply;
  };
  const double nan                      = std::numeric_limits<double>::quiet_NaN();
  std::vector<corruption_t> corruptions = {
    {"twin", [](state_t& s) { s.csr.twin[0] = 13; }},
    {"twin", [](state_t& s) { s.csr.index[0] = 2; }},
    {"value", [](state_t& s) { s.csc.value[12] = 9.; }},
    {"value", [](state_t& s) { s.csr.value[0] = s.csc.value[12] = 0.; }},
    {"size", [](state_t& s) { s.csr.size[0] = 2; }},
    {"size", [](state_t& s) { s.csc.size[0] = 3; }},
    {"capacity", [](state_t& s) { s.csr.used[0] = 4; }},
    {"outside the store", [](state_t& s) { s.csc.capacity[0] = 5; }},
    {"another line", [](state_t& s) { s.csc.start[2] = 3; }},
    {"tombstone", [](state_t& s) { s.csc.index[15] = 0; }},
    {"tombstone", [](state_t& s) { s.csc.index[0] = 0; }},
    {"locks", [](state_t& s) { s.up_locks[0] += 1; }},
    {"locks", [](state_t& s) { s.down_locks[1] -= 1; }},
    {"row_free_size", [](state_t& s) { s.row_free_size[0] -= 1; }},
    {"live_nnz", [](state_t& s) { s.live_nnz += 1; }},
    {"live_rows", [](state_t& s) { s.live_rows -= 1; }},
    {"live_cols", [](state_t& s) { s.live_cols += 1; }},
    {"fixed_nnz", [](state_t& s) { s.fixed_nnz += 1; }},
    {"inf_count", [](state_t& s) { s.min_act.inf_count[1] += 1; }},
    {"inf_slot", [](state_t& s) { s.max_act.inf_slot[0] += 1; }},
    {"differs", [](state_t& s) { s.min_act.hi[0] += 1e-9; }},
    {"differs", [](state_t& s) { s.max_act.lo[1] = 1e-13; }},
    {"max_term", [](state_t& s) { s.min_act.max_term[0] = 0.; }},
    {"updates", [](state_t& s) { s.min_act.updates[0] = 1; }},
    {"not summed again",
     [](state_t& s) {
       s.tol.activity_error_share = 1e-40;
       s.min_act.updates[0]       = 100;
     }},
    {"not finite", [nan](state_t& s) { s.max_act.hi[1] = nan; }},
    {"pending", [](state_t& s) { s.row_frontier.pending[0] ^= 1; }},
    {"pending", [](state_t& s) { s.col_frontier.pending[0] |= col_event::sparse; }},
    {"twice", [](state_t& s) { s.row_frontier.list[0].push_back(s.row_frontier.list[0][0]); }},
    {"compaction list",
     [](state_t& s) {
       s.row_frontier.pending[4] &= ~row_event::sparse;
       s.row_frontier.list[row_list::compaction].clear();
     }},
    {"fixed column", [](state_t& s) { s.col_lower[1] = 0.5; }},
    {"removed column", [](state_t& s) { s.col_status[0] = col_status_t::removed; }},
    {"deleted row", [](state_t& s) { s.min_act.hi[2] = 1.; }},
    {"deleted row", [](state_t& s) { s.row_free_size[2] = 1; }},
    {"alive flag", [](state_t& s) { s.row_alive[0] = 2; }},
    {"sides", [nan](state_t& s) { s.row_lower[0] = nan; }},
    {"bounds", [](state_t& s) { s.col_upper[0] = -inf; }},
    {"cost", [nan](state_t& s) { s.cost[0] = nan; }},
    {"source row", [](state_t& s) { s.implied_lower[0] = 3.; }},
    {"integer column", [](state_t& s) { s.implied_upper_row[0] = 1; }},
    {"objective offset", [](state_t& s) { s.objective_offset.lo = inf; }},
    {"one element per row", [](state_t& s) { s.row_version.pop_back(); }},
    {"one element per column", [](state_t& s) { s.up_locks.push_back(0); }},
  };
  for (const corruption_t& corruption : corruptions) {
    state_t broken = base;
    corruption.apply(broken);
    std::string error = broken.consistency_error();
    if (error.find(corruption.keyword) == std::string::npos) {
      test::fail(__FILE__,
                 __LINE__,
                 std::string("expected an error naming \"") + corruption.keyword + "\", got \"" +
                   error + "\"");
    }
  }
}
