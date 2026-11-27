/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "test_state_support.hpp"

#include <gprs/presolve.hpp>
#include <gprs/state.hpp>

#include <cmath>
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

using list_t  = std::vector<int32_t>;
using slots_t = std::vector<int64_t>;

// reduced_model of the state as it is against the naive reference and against the two steps it
// stands for, a rebuild and extract_model, bit for bit; the state stays as it was.
void check_reduced(const state_t& st)
{
  std::vector<int32_t> row_origin, col_origin, other_row_origin, other_col_origin;
  uint64_t before = state_hash(st);
  model_t reduced = reduced_model(st, row_origin, col_origin);
  GPRS_CHECK_EQ(state_hash(st), before);
  GPRS_CHECK(well_formed(reduced));

  model_t reference = reference_model(st, other_row_origin, other_col_origin);
  GPRS_CHECK(same_model(reduced, reference));
  GPRS_CHECK(row_origin == other_row_origin);
  GPRS_CHECK(col_origin == other_col_origin);

  state_t rebuilt = st;
  rebuilt.rebuild();
  model_t extracted = extract_model(rebuilt, other_row_origin, other_col_origin);
  GPRS_CHECK(same_model(reduced, extracted));
  GPRS_CHECK_EQ(model_hash(reduced), model_hash(extracted));
  GPRS_CHECK(row_origin == other_row_origin);
  GPRS_CHECK(col_origin == other_col_origin);
}

// What a rebuild leaves in the two stores: every line in index order with its kept entries in
// storage order.
struct layout_t {
  slots_t row_start, col_start;
  list_t row_index, col_index;
  std::vector<double> row_value, col_value;
};

layout_t expected_layout(const state_t& st)
{
  layout_t x;
  for (int32_t i = 0; i < st.nrows; ++i) {
    x.row_start.push_back(x.row_index.size());
    for (int64_t e = st.csr.begin(i); e < st.csr.end(i); ++e) {
      if (!st.csr.live(e) || st.col_status[st.csr.index[e]] == col_status_t::fixed) { continue; }
      x.row_index.push_back(st.csr.index[e]);
      x.row_value.push_back(st.csr.value[e]);
    }
  }
  for (int32_t j = 0; j < st.ncols; ++j) {
    x.col_start.push_back(x.col_index.size());
    if (st.col_status[j] == col_status_t::fixed) { continue; }
    for (int64_t p = st.csc.begin(j); p < st.csc.end(j); ++p) {
      if (!st.csc.live(p)) { continue; }
      x.col_index.push_back(st.csc.index[p]);
      x.col_value.push_back(st.csc.value[p]);
    }
  }
  return x;
}

void check_rebuild(const state_t& st)
{
  layout_t x      = expected_layout(st);
  state_t rebuilt = st;
  rebuilt.rebuild();
  GPRS_CHECK_EQ(rebuilt.consistency_error(), std::string());
  GPRS_CHECK(rebuilt.csr.start == x.row_start);
  GPRS_CHECK(rebuilt.csr.index == x.row_index);
  GPRS_CHECK(same_doubles(rebuilt.csr.value, x.row_value));
  GPRS_CHECK(rebuilt.csc.start == x.col_start);
  GPRS_CHECK(rebuilt.csc.index == x.col_index);
  GPRS_CHECK(same_doubles(rebuilt.csc.value, x.col_value));
  GPRS_CHECK(rebuilt.csr.used == rebuilt.csr.size && rebuilt.csr.capacity == rebuilt.csr.size);
  GPRS_CHECK(rebuilt.csc.used == rebuilt.csc.size && rebuilt.csc.capacity == rebuilt.csc.size);
  GPRS_CHECK_EQ(rebuilt.csr.slots(), rebuilt.live_nnz);
  GPRS_CHECK_EQ(rebuilt.csc.slots(), rebuilt.live_nnz);
  GPRS_CHECK_EQ(rebuilt.csr.twin.size(), rebuilt.csr.index.size());
  GPRS_CHECK_EQ(rebuilt.csc.twin.size(), rebuilt.csc.index.size());
}

// The columns lie in the store in index order; a moved column breaks it.
bool columns_in_order(const state_t& st)
{
  int64_t end = 0;
  for (int32_t j = 0; j < st.ncols; ++j) {
    if (st.csc.start[j] < end) { return false; }
    end = st.csc.start[j] + st.csc.capacity[j];
  }
  return true;
}

bool has_unsorted_row(const state_t& st)
{
  for (int32_t i = 0; i < st.nrows; ++i) {
    int32_t last = -1;
    for (int64_t e = st.csr.begin(i); e < st.csr.end(i); ++e) {
      if (!st.csr.live(e) || !st.is_live(st.csr.index[e])) { continue; }
      if (st.csr.index[e] < last) { return true; }
      last = st.csr.index[e];
    }
  }
  return false;
}

// A live entry of a live column in a random row, or -1.
int64_t random_free_slot(const state_t& st, rng_t& rng, int32_t& row)
{
  row = rng.below(st.nrows);
  if (st.csr.used[row] == 0) { return -1; }
  int64_t e = st.csr.begin(row) + rng.below(st.csr.used[row]);
  return st.csr.live(e) && st.is_live(st.csr.index[e]) ? e : -1;
}

// One random operation that leaves tombstones, fixed entries, deleted rows, rows out of column
// order or columns moved to the end of their store. False when it had no candidate.
bool random_operation(state_t& st, rng_t& rng)
{
  int32_t kind = rng.below(10);
  int32_t row;
  if (kind == 0) {
    int32_t j = rng.below(st.ncols);
    if (!st.is_live(j)) { return false; }
    double value = st.col_lower[j] > -inf ? st.col_lower[j] : st.col_upper[j];
    if (!std::isfinite(value)) { return false; }
    st.fix_col(j, value);
  } else if (kind == 1) {
    row = rng.below(st.nrows);
    if (!st.row_alive[row]) { return false; }
    st.delete_row(row);
  } else if (kind < 4) {
    int64_t e = random_free_slot(st, rng, row);
    if (e < 0) { return false; }
    st.delete_entry(e);
  } else if (kind < 9) {
    int64_t e   = random_free_slot(st, rng, row);
    int32_t col = rng.below(st.ncols);
    if (e < 0 || !st.is_live(col) || st.find_entry(row, col) >= 0) { return false; }
    st.move_entry(e, col, random_coefficient(rng, false));
  } else {
    st.compact_sparse_lines();
  }
  return true;
}

}  // namespace

// A state that was never changed gives its model back.
GPRS_TEST(state_reduced_round_trip)
{
  rng_t rng(47);
  for (int32_t k = 0; k < 40; ++k) {
    model_options_t shape;
    shape.nrows   = rng.below(15);
    shape.ncols   = rng.below(15);
    shape.density = 0.4;
    model_t m     = random_model(rng, shape);
    GPRS_REQUIRE(well_formed(m));
    state_t st(m, tolerances_t());
    std::vector<int32_t> row_origin, col_origin;
    GPRS_CHECK(same_model(reduced_model(st, row_origin, col_origin), m));
    GPRS_CHECK_EQ((int32_t)row_origin.size(), m.nrows);
    GPRS_CHECK_EQ((int32_t)col_origin.size(), m.ncols);
    for (int32_t i = 0; i < m.nrows; ++i) {
      GPRS_CHECK_EQ(row_origin[i], i);
    }
    for (int32_t j = 0; j < m.ncols; ++j) {
      GPRS_CHECK_EQ(col_origin[j], j);
    }
    check_reduced(st);
  }
}

// The state of state_storage_rebuild before its rebuild: a fixed column, a deleted row and a
// deleted entry. The model is the one that test extracts after the rebuild.
GPRS_TEST(state_reduced_drops_dead_and_fixed)
{
  state_t st(hand_model(), tolerances_t());
  st.fix_col(1, 1.);
  st.delete_row(2);
  st.delete_entry(8);
  st.set_col_upper(0, 3.);
  st.set_cost(4, 1.);

  std::vector<int32_t> row_origin, col_origin;
  model_t reduced = reduced_model(st, row_origin, col_origin);
  GPRS_CHECK_EQ(st.csr.slots(), 12);
  GPRS_CHECK_EQ(st.fixed_nnz, 3);
  GPRS_CHECK_EQ(st.col_status[1], col_status_t::fixed);
  GPRS_CHECK(st.row_upper == std::vector<double>({10., 5., inf, 1., 8.}));

  GPRS_CHECK(well_formed(reduced));
  GPRS_CHECK(row_origin == list_t({0, 1, 3, 4}));
  GPRS_CHECK(col_origin == list_t({0, 2, 3, 4}));
  GPRS_CHECK_EQ(reduced.nrows, 4);
  GPRS_CHECK_EQ(reduced.ncols, 4);
  GPRS_CHECK(reduced.row_start == slots_t({0, 2, 3, 3, 5}));
  GPRS_CHECK(reduced.col_index == list_t({0, 1, 2, 1, 3}));
  GPRS_CHECK(reduced.value == std::vector<double>({2., 1., 1., 1., 1.}));
  // The term of the fixed column leaves the finite sides: -3 * 1 from r0, 1 * 1 from r1 and r4.
  GPRS_CHECK(reduced.row_lower == std::vector<double>({-inf, 4., -1., 1.}));
  GPRS_CHECK(reduced.row_upper == std::vector<double>({13., 4., 1., 7.}));
  GPRS_CHECK(reduced.col_lower == std::vector<double>({0., 0., -inf, 1.}));
  GPRS_CHECK(reduced.col_upper == std::vector<double>({3., inf, inf, 1e21}));
  GPRS_CHECK(reduced.cost == std::vector<double>({1., 0., 2., 1.}));
  GPRS_CHECK(reduced.col_type == std::vector<uint8_t>({1, 0, 1, 0}));
  GPRS_CHECK_EQ(reduced.objective_offset, 6.);
  check_reduced(st);
}

// The rows of the model are sorted by column whatever the storage order: r0 holds x0, x4, x2
// and r1 holds x1, x0.
GPRS_TEST(state_reduced_sorts_rows)
{
  state_t st(hand_model(), tolerances_t());
  st.move_entry(4, 0, 3.);
  st.move_entry(1, 4, -6.);
  GPRS_CHECK(has_unsorted_row(st));
  GPRS_CHECK(!columns_in_order(st));

  std::vector<int32_t> row_origin, col_origin;
  model_t reduced = reduced_model(st, row_origin, col_origin);
  GPRS_CHECK(well_formed(reduced));
  GPRS_CHECK(reduced.row_start == slots_t({0, 3, 5, 8, 8, 12}));
  GPRS_CHECK(reduced.col_index == list_t({0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 2, 4}));
  GPRS_CHECK(reduced.value ==
             std::vector<double>({2., 1., -6., 3., 1., -1., -2., 0.5, 1., 1., 1., 1.}));
  GPRS_CHECK(row_origin == list_t({0, 1, 2, 3, 4}));
  GPRS_CHECK(col_origin == list_t({0, 1, 2, 3, 4}));
  check_reduced(st);
  check_rebuild(st);
}

// Random states with tombstones, fixed columns, deleted rows, rows out of column order and
// moved columns: reduced_model against the reference and against rebuild and extract_model, and
// the layout a rebuild leaves, with the columns in index order (compacted in place) and not.
GPRS_TEST(state_reduced_and_rebuild_random_states)
{
  rng_t rng(53);
  int failures     = test::failure_count();
  int32_t unsorted = 0, moved_columns = 0, columns_in_place = 0, with_fixed = 0;
  for (int32_t k = 0; k < 300; ++k) {
    model_options_t shape;
    shape.nrows     = 1 + rng.below(12);
    shape.ncols     = 2 + rng.below(10);
    shape.density   = 0.2 + 0.3 * rng.uniform();
    shape.long_rows = rng.below(3);
    model_t m       = random_model(rng, shape);
    GPRS_REQUIRE(well_formed(m));
    state_t st(m, tolerances_t());
    for (int32_t round = 0; round < 6; ++round) {
      int32_t operations = 1 + rng.below(8);
      for (int32_t o = 0; o < operations; ++o) {
        random_operation(st, rng);
      }
      GPRS_REQUIRE(st.consistency_error().empty());
      unsorted += has_unsorted_row(st);
      moved_columns += !columns_in_order(st);
      columns_in_place += columns_in_order(st);
      with_fixed += st.fixed_nnz > 0;
      check_reduced(st);
      check_rebuild(st);
      if (test::failure_count() != failures) { return; }
      if (rng.chance(0.3)) { st.rebuild(); }
    }
  }
  GPRS_CHECK(unsorted > 100);
  GPRS_CHECK(moved_columns > 100);
  GPRS_CHECK(columns_in_place > 100);
  GPRS_CHECK(with_fixed > 100);
}
