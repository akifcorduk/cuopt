/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/state.hpp>

#include <cmath>
#include <limits>

namespace gprs {

namespace {

inline void count_update(activity_t& act, int32_t row, double a, double bound)
{
  double magnitude = std::fabs(a * bound);
  if (magnitude > act.max_term[row]) { act.max_term[row] = magnitude; }
  gprs_assert(act.updates[row] < std::numeric_limits<uint32_t>::max());
  ++act.updates[row];
}

inline void remove_term(
  activity_t& act, int32_t row, int64_t slot, double a, double bound, const tolerances_t& tol)
{
  if (is_infinite_term(a, bound, tol)) {
    gprs_assert(act.inf_count[row] > 0);
    --act.inf_count[row];
    act.inf_slot[row] -= slot;
    return;
  }
  dd_t sum{act.hi[row], act.lo[row]};
  dd_sub_product(sum, a, bound);
  act.hi[row] = sum.hi;
  act.lo[row] = sum.lo;
  count_update(act, row, a, bound);
}

inline void add_term(
  activity_t& act, int32_t row, int64_t slot, double a, double bound, const tolerances_t& tol)
{
  if (is_infinite_term(a, bound, tol)) {
    ++act.inf_count[row];
    act.inf_slot[row] += slot;
    return;
  }
  dd_t sum{act.hi[row], act.lo[row]};
  dd_add_product(sum, a, bound);
  act.hi[row] = sum.hi;
  act.lo[row] = sum.lo;
  count_update(act, row, a, bound);
}

// The entry a at `slot` of `row`, in a column with the given bounds, leaves both activities.
inline void remove_entry_terms(
  state_t& st, int32_t row, int64_t slot, double a, double lower, double upper)
{
  remove_term(st.min_act, row, slot, a, min_activity_bound(a, lower, upper), st.tol);
  remove_term(st.max_act, row, slot, a, max_activity_bound(a, lower, upper), st.tol);
}

inline void add_entry_terms(
  state_t& st, int32_t row, int64_t slot, double a, double lower, double upper)
{
  add_term(st.min_act, row, slot, a, min_activity_bound(a, lower, upper), st.tol);
  add_term(st.max_act, row, slot, a, max_activity_bound(a, lower, upper), st.tol);
}

inline bool side_stale(const state_t& st, const activity_t& act, int32_t row)
{
  return act.updates[row] >= 2u * st.csr.size[row] &&
         activity_error(act.updates[row], act.max_term[row]) >
           st.tol.activity_error_share * st.tol.feasibility;
}

inline void refresh_stale_activity(state_t& st, int32_t row)
{
  if (side_stale(st, st.min_act, row) || side_stale(st, st.max_act, row)) {
    st.slot_visits += st.csr.used[row];
    st.recompute_row_activity(row);
  }
}

// The locks that the entry a of `row` puts on `col`, added (sign 1) or released (sign -1).
inline void change_locks(state_t& st, int32_t row, int32_t col, double a, int32_t sign)
{
  int32_t lower_finite = st.row_lower[row] > -inf ? sign : 0;
  int32_t upper_finite = st.row_upper[row] < inf ? sign : 0;
  if (a > 0) {
    st.up_locks[col] += upper_finite;
    st.down_locks[col] += lower_finite;
  } else {
    st.down_locks[col] += upper_finite;
    st.up_locks[col] += lower_finite;
  }
}

inline void note_row_tombstone(state_t& st, int32_t row)
{
  if (st.csr.used[row] - st.csr.size[row] > st.csr.size[row]) {
    st.row_frontier.push(row_event::sparse, row);
  }
}

inline void note_col_tombstone(state_t& st, int32_t col)
{
  if (st.is_live(col) && st.csc.used[col] - st.csc.size[col] > st.csc.size[col]) {
    st.col_frontier.push(col_event::sparse, col);
  }
}

inline int32_t row_of(const state_t& st, int64_t row_slot)
{
  return st.csc.index[st.csr.twin[row_slot]];
}

// CSR slot of the live entry (row, col) or -1, by a scan of the shorter line; `visits` counts
// the slots it reads.
int64_t locate_entry(const state_t& st, int32_t row, int32_t col, int64_t& visits)
{
  const line_store_t& csr = st.csr;
  const line_store_t& csc = st.csc;
  if (csr.used[row] <= csc.used[col]) {
    for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
      ++visits;
      if (csr.index[e] == col) { return e; }
    }
    return -1;
  }
  for (int64_t p = csc.begin(col); p < csc.end(col); ++p) {
    ++visits;
    if (csc.index[p] == row) { return csc.twin[p]; }
  }
  return -1;
}

// side - a * b with the product error carried.
inline double shifted_side(double side, double a, double b)
{
  dd_t product;
  product.hi = two_prod(a, b, product.lo);
  return dd_residual(side, product);
}

constexpr int32_t min_moved_capacity = 4;

// Moves the live entries of a column, in order, to the end of the CSC store, where the column
// gets twice its capacity.
void move_column_to_end(state_t& st, int32_t col)
{
  line_store_t& csc = st.csc;
  gprs_assert(csc.capacity[col] <= std::numeric_limits<int32_t>::max() / 2);
  int32_t capacity = 2 * csc.capacity[col];
  if (capacity < min_moved_capacity) { capacity = min_moved_capacity; }
  int64_t begin = csc.begin(col), end = csc.end(col);
  int64_t target = csc.slots();
  csc.index.resize(target + capacity, dead_index);
  csc.value.resize(target + capacity);
  csc.twin.resize(target + capacity);
  int64_t q = target;
  for (int64_t p = begin; p < end; ++p) {
    if (!csc.live(p)) { continue; }
    csc.index[q]             = csc.index[p];
    csc.value[q]             = csc.value[p];
    csc.twin[q]              = csc.twin[p];
    st.csr.twin[csc.twin[p]] = q;
    csc.index[p]             = dead_index;
    ++q;
  }
  st.slot_visits += csc.used[col];
  gprs_assert(q - target == csc.size[col]);
  csc.start[col]    = target;
  csc.used[col]     = csc.size[col];
  csc.capacity[col] = capacity;
}

void set_col_bound(state_t& st, int32_t col, double old_bound, double new_bound, bool is_lower)
{
  const line_store_t& csc = st.csc;
  for (int64_t p = csc.begin(col); p < csc.end(col); ++p) {
    if (!csc.live(p)) { continue; }
    int32_t row = csc.index[p];
    double a    = csc.value[p];
    // The lower bound is in the minimum activity for a > 0 and in the maximum for a < 0.
    bool in_min     = (a > 0) == is_lower;
    activity_t& act = in_min ? st.min_act : st.max_act;
    remove_term(act, row, csc.twin[p], a, old_bound, st.tol);
    add_term(act, row, csc.twin[p], a, new_bound, st.tol);
    // Bound propagation reads the minimum activity with the upper side of the row and the
    // maximum activity with the lower side.
    bool propagates = in_min ? st.row_upper[row] < inf : st.row_lower[row] > -inf;
    st.row_frontier.push(propagates ? row_event::activity : row_event::activity_unread, row);
    refresh_stale_activity(st, row);
  }
  st.slot_visits += csc.used[col];
  ++st.bound_changes;
}

// Adds delta to the locks that one side of the row puts on its columns.
void change_side_locks(state_t& st, int32_t row, bool is_lower, int32_t delta)
{
  const line_store_t& csr = st.csr;
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if ((csr.value[e] > 0) == is_lower) {
      st.down_locks[col] += delta;
    } else {
      st.up_locks[col] += delta;
    }
    if (st.is_live(col)) { st.col_frontier.push(col_event::locks, col); }
  }
  st.slot_visits += csr.used[row];
}

}  // namespace

double state_t::dead_fraction() const
{
  int64_t slots = csr.slots() > csc.slots() ? csr.slots() : csc.slots();
  if (slots == 0) { return 0.; }
  return (double)(slots - live_nnz) / (double)slots;
}

int64_t state_t::find_entry(int32_t row, int32_t col) const
{
  return locate_entry(*this, row, col, slot_visits);
}

bool state_t::implied_lower_valid(int32_t col) const
{
  int32_t row = implied_lower_row[col];
  return row >= 0 && row_alive[row] && row_version[row] == implied_lower_version[col];
}

bool state_t::implied_upper_valid(int32_t col) const
{
  int32_t row = implied_upper_row[col];
  return row >= 0 && row_alive[row] && row_version[row] == implied_upper_version[col];
}

void state_t::set_col_lower(int32_t col, double lower)
{
  gprs_assert(is_live(col));
  gprs_assert(lower < inf && lower <= col_upper[col] && lower != col_lower[col]);
  double old     = col_lower[col];
  col_lower[col] = lower;
  set_col_bound(*this, col, old, lower, true);
}

void state_t::set_col_upper(int32_t col, double upper)
{
  gprs_assert(is_live(col));
  gprs_assert(upper > -inf && upper >= col_lower[col] && upper != col_upper[col]);
  double old     = col_upper[col];
  col_upper[col] = upper;
  set_col_bound(*this, col, old, upper, false);
}

void state_t::fix_col(int32_t col, double value)
{
  gprs_assert(is_live(col));
  gprs_assert(std::isfinite(value) && col_lower[col] <= value && value <= col_upper[col]);
  double lower    = col_lower[col];
  double upper    = col_upper[col];
  col_lower[col]  = value;
  col_upper[col]  = value;
  col_status[col] = col_status_t::fixed;
  for (int64_t p = csc.begin(col); p < csc.end(col); ++p) {
    if (!csc.live(p)) { continue; }
    int32_t row     = csc.index[p];
    int64_t slot    = csc.twin[p];
    double a        = csc.value[p];
    double min_side = min_activity_bound(a, lower, upper);
    double max_side = max_activity_bound(a, lower, upper);
    if (min_side != value) {
      remove_term(min_act, row, slot, a, min_side, tol);
      add_term(min_act, row, slot, a, value, tol);
    }
    if (max_side != value) {
      remove_term(max_act, row, slot, a, max_side, tol);
      add_term(max_act, row, slot, a, value, tol);
    }
    --row_free_size[row];
    row_frontier.push(row_event::entries, row);
    refresh_stale_activity(*this, row);
  }
  slot_visits += csc.used[col];
  --live_cols;
  fixed_nnz += csc.size[col];
  bound_changes += (lower != value) + (upper != value);
  add_objective_offset(cost[col], value);
}

void state_t::remove_col(int32_t col)
{
  gprs_assert(is_live(col));
  gprs_assert(csc.size[col] == 0 && cost[col] == 0.);
  col_status[col] = col_status_t::removed;
  --live_cols;
}

void state_t::set_cost(int32_t col, double value)
{
  gprs_assert(std::isfinite(value));
  cost[col] = value;
  col_frontier.push(col_event::locks, col);
}

void state_t::add_objective_offset(double a, double b)
{
  gprs_assert(std::isfinite(a * b));
  dd_add_product(objective_offset, a, b);
}

void state_t::record_implied_lower(int32_t col, double lower, int32_t row)
{
  gprs_assert(!is_integer(col) && row_alive[row] && !std::isnan(lower));
  implied_lower[col]         = lower;
  implied_lower_row[col]     = row;
  implied_lower_version[col] = row_version[row];
}

void state_t::record_implied_upper(int32_t col, double upper, int32_t row)
{
  gprs_assert(!is_integer(col) && row_alive[row] && !std::isnan(upper));
  implied_upper[col]         = upper;
  implied_upper_row[col]     = row;
  implied_upper_version[col] = row_version[row];
}

void state_t::delete_row(int32_t row)
{
  gprs_assert(row_alive[row]);
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    change_locks(*this, row, col, csr.value[e], -1);
    csc.index[csr.twin[e]] = dead_index;
    --csc.size[col];
    csr.index[e] = dead_index;
    if (col_status[col] == col_status_t::fixed) {
      --fixed_nnz;
    } else {
      col_frontier.push(col_event::locks, col);
      note_col_tombstone(*this, col);
    }
  }
  slot_visits += csr.used[row];
  live_nnz -= csr.size[row];
  csr.size[row]      = 0;
  row_free_size[row] = 0;
  row_alive[row]     = 0;
  --live_rows;
  for (activity_t* act : {&min_act, &max_act}) {
    act->hi[row]        = 0.;
    act->lo[row]        = 0.;
    act->inf_count[row] = 0;
    act->inf_slot[row]  = 0;
    act->max_term[row]  = 0.;
    act->updates[row]   = 0;
  }
}

void state_t::set_row_lower(int32_t row, double lower)
{
  gprs_assert(row_alive[row]);
  gprs_assert(lower < inf && lower <= row_upper[row] && lower != row_lower[row]);
  bool was_finite = row_lower[row] > -inf;
  bool is_finite  = lower > -inf;
  row_lower[row]  = lower;
  if (was_finite != is_finite) { change_side_locks(*this, row, true, is_finite ? 1 : -1); }
  ++row_version[row];
  ++side_changes;
  row_frontier.push(row_event::side, row);
}

void state_t::set_row_upper(int32_t row, double upper)
{
  gprs_assert(row_alive[row]);
  gprs_assert(upper > -inf && upper >= row_lower[row] && upper != row_upper[row]);
  bool was_finite = row_upper[row] < inf;
  bool is_finite  = upper < inf;
  row_upper[row]  = upper;
  if (was_finite != is_finite) { change_side_locks(*this, row, false, is_finite ? 1 : -1); }
  ++row_version[row];
  ++side_changes;
  row_frontier.push(row_event::side, row);
}

void state_t::shift_row_sides(int32_t row, double a, double b)
{
  gprs_assert(row_alive[row]);
  gprs_assert(std::isfinite(a * b));
  bool ordered = row_lower[row] <= row_upper[row];
  if (row_lower[row] > -inf) {
    row_lower[row] = shifted_side(row_lower[row], a, b);
    gprs_assert(std::isfinite(row_lower[row]));
    ++side_changes;
  }
  if (row_upper[row] < inf) {
    row_upper[row] = shifted_side(row_upper[row], a, b);
    gprs_assert(std::isfinite(row_upper[row]));
    ++side_changes;
  }
  gprs_assert(!ordered || row_lower[row] <= row_upper[row]);
  ++row_version[row];
  row_frontier.push(row_event::side, row);
}

void state_t::scale_row(int32_t row, double factor, bool integral)
{
  gprs_assert(row_alive[row]);
  gprs_assert(factor > 0 && factor < inf);
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    double value = csr.value[e] * factor;
    // The term of a fixed column is a constant that no caller tests for integrality.
    if (integral && is_live(csr.index[e])) { value = std::round(value); }
    gprs_assert(std::isfinite(value) && value != 0.);
    csr.value[e]           = value;
    csc.value[csr.twin[e]] = value;
  }
  if (row_lower[row] > -inf) {
    row_lower[row] *= factor;
    gprs_assert(std::isfinite(row_lower[row]));
    ++side_changes;
  }
  if (row_upper[row] < inf) {
    row_upper[row] *= factor;
    gprs_assert(std::isfinite(row_upper[row]));
    ++side_changes;
  }
  coefficient_changes += csr.size[row];
  slot_visits += 2 * (int64_t)csr.used[row];
  recompute_row_activity(row);
  ++row_version[row];
  row_frontier.push(row_event::entries, row);
}

void state_t::set_coefficient(int64_t row_slot, double value)
{
  gprs_assert(csr.live(row_slot));
  gprs_assert(std::isfinite(value) && value != 0.);
  int32_t col = csr.index[row_slot];
  int32_t row = row_of(*this, row_slot);
  gprs_assert(is_live(col));
  double old = csr.value[row_slot];
  remove_entry_terms(*this, row, row_slot, old, col_lower[col], col_upper[col]);
  change_locks(*this, row, col, old, -1);
  csr.value[row_slot]           = value;
  csc.value[csr.twin[row_slot]] = value;
  add_entry_terms(*this, row, row_slot, value, col_lower[col], col_upper[col]);
  change_locks(*this, row, col, value, 1);
  ++slot_visits;
  ++coefficient_changes;
  ++row_version[row];
  row_frontier.push(row_event::entries, row);
  col_frontier.push(col_event::locks, col);
  refresh_stale_activity(*this, row);
}

void state_t::delete_entry(int64_t row_slot)
{
  gprs_assert(csr.live(row_slot));
  int32_t col = csr.index[row_slot];
  int32_t row = row_of(*this, row_slot);
  gprs_assert(is_live(col));
  double a = csr.value[row_slot];
  remove_entry_terms(*this, row, row_slot, a, col_lower[col], col_upper[col]);
  change_locks(*this, row, col, a, -1);
  csc.index[csr.twin[row_slot]] = dead_index;
  csr.index[row_slot]           = dead_index;
  --csc.size[col];
  --csr.size[row];
  --row_free_size[row];
  --live_nnz;
  ++slot_visits;
  ++coefficient_changes;
  ++row_version[row];
  row_frontier.push(row_event::entries, row);
  col_frontier.push(col_event::locks, col);
  note_row_tombstone(*this, row);
  note_col_tombstone(*this, col);
  refresh_stale_activity(*this, row);
}

void state_t::move_entry(int64_t row_slot, int32_t col, double value)
{
  gprs_assert(csr.live(row_slot));
  gprs_assert(std::isfinite(value) && value != 0.);
  int32_t from = csr.index[row_slot];
  int32_t row  = row_of(*this, row_slot);
  gprs_assert(is_live(from) && is_live(col));
  int64_t uncounted = 0;
  gprs_assert(locate_entry(*this, row, col, uncounted) < 0);
  double old = csr.value[row_slot];
  remove_entry_terms(*this, row, row_slot, old, col_lower[from], col_upper[from]);
  change_locks(*this, row, from, old, -1);
  csc.index[csr.twin[row_slot]] = dead_index;
  --csc.size[from];
  col_frontier.push(col_event::locks, from);
  note_col_tombstone(*this, from);

  if (csc.used[col] == csc.capacity[col]) { move_column_to_end(*this, col); }
  int64_t p    = csc.end(col);
  csc.index[p] = row;
  csc.value[p] = value;
  csc.twin[p]  = row_slot;
  ++csc.used[col];
  ++csc.size[col];
  csr.index[row_slot] = col;
  csr.value[row_slot] = value;
  csr.twin[row_slot]  = p;
  add_entry_terms(*this, row, row_slot, value, col_lower[col], col_upper[col]);
  change_locks(*this, row, col, value, 1);
  col_frontier.push(col_event::locks, col);

  ++slot_visits;
  ++coefficient_changes;
  ++row_version[row];
  row_frontier.push(row_event::entries, row);
  refresh_stale_activity(*this, row);
}

}  // namespace gprs
