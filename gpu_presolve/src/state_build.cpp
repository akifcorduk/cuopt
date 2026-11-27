/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/state.hpp>

#include <cmath>

namespace gprs {

bool well_formed(const model_t& model)
{
  if (model.nrows < 0 || model.ncols < 0) { return false; }
  size_t nrows = model.nrows, ncols = model.ncols;
  if (model.row_start.size() != nrows + 1 || model.row_lower.size() != nrows ||
      model.row_upper.size() != nrows || model.col_lower.size() != ncols ||
      model.col_upper.size() != ncols || model.cost.size() != ncols ||
      model.col_type.size() != ncols || model.col_index.size() != model.value.size()) {
    return false;
  }
  if (model.row_start[0] != 0 || model.row_start[nrows] != model.nnz()) { return false; }
  for (int32_t i = 0; i < model.nrows; ++i) {
    if (model.row_start[i] > model.row_start[i + 1]) { return false; }
  }
  for (int32_t i = 0; i < model.nrows; ++i) {
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      int32_t j = model.col_index[e];
      if (j < 0 || j >= model.ncols) { return false; }
      if (e > model.row_start[i] && model.col_index[e - 1] >= j) { return false; }
      if (!std::isfinite(model.value[e]) || model.value[e] == 0.) { return false; }
    }
    if (std::isnan(model.row_lower[i]) || std::isnan(model.row_upper[i])) { return false; }
    if (model.row_lower[i] == inf || model.row_upper[i] == -inf) { return false; }
  }
  for (int32_t j = 0; j < model.ncols; ++j) {
    if (std::isnan(model.col_lower[j]) || std::isnan(model.col_upper[j])) { return false; }
    if (model.col_lower[j] == inf || model.col_upper[j] == -inf) { return false; }
    if (!std::isfinite(model.cost[j]) || model.col_type[j] > 1) { return false; }
  }
  return std::isfinite(model.objective_offset);
}

void activity_t::resize(int32_t nrows)
{
  hi.resize(nrows);
  lo.resize(nrows);
  inf_count.resize(nrows);
  inf_slot.resize(nrows);
  max_term.resize(nrows);
  updates.resize(nrows);
}

namespace {

// For the sums of one row: a call, with its results passed through memory, costs more than the
// sums of a short row.
#if defined(__GNUC__)
#define GPRS_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define GPRS_ALWAYS_INLINE inline
#endif

// One term of one side, into `lane`; `first` when the lane holds no term yet.
// There is no branch on the term: the sign of a coefficient decides which bound a side takes,
// so whether a term is infinite can change from entry to entry and is not predictable. An
// infinite term enters the lane as a * 0 instead and leaves it as it is, bit for bit: no part
// of a lane is ever -0 (a sum is -0 only when both of its terms are, and a lane starts at
// (+0, +0)), and x + (+-0) is x for every other x.
// The first term of a lane is dd_add_product on (+0, +0) reduced to what the zeros leave of
// it, hi = 0 + p and lo = 0 + error. The two sums stay: 0 + (-0) is +0.
GPRS_ALWAYS_INLINE void add_term(side_sum_t& side,
                                 dd_t& lane,
                                 double a,
                                 double bound,
                                 int64_t slot,
                                 const tolerances_t& tol,
                                 bool first)
{
  bool infinite = is_infinite_term(a, bound, tol);
  side.inf_count += infinite;
  side.inf_slot += infinite ? slot : 0;
  double finite_bound = infinite ? 0. : bound;
  if (first) {
    double error;
    lane.hi = 0. + two_prod(a, finite_bound, error);
    lane.lo = 0. + error;
  } else {
    dd_add_product(lane, a, finite_bound);
  }
  double magnitude = std::fabs(a * finite_bound);
  side.max_term    = magnitude > side.max_term ? magnitude : side.max_term;
  side.terms += !infinite;
}

GPRS_ALWAYS_INLINE void add_entry(const state_t& st,
                                  int64_t slot,
                                  side_sum_t& min_side,
                                  dd_t& lane_min,
                                  side_sum_t& max_side,
                                  dd_t& lane_max,
                                  bool first)
{
  int32_t j    = st.csr.index[slot];
  double a     = st.csr.value[slot];
  double lower = st.col_lower[j];
  double upper = st.col_upper[j];
  add_term(min_side, lane_min, a, min_activity_bound(a, lower, upper), slot, st.tol, first);
  add_term(max_side, lane_max, a, max_activity_bound(a, lower, upper), slot, st.tol, first);
}

GPRS_ALWAYS_INLINE void merge_lanes(dd_t* lane_min, dd_t* lane_max, int32_t lanes, int32_t active)
{
  // Lanes from `active` on hold no term, and merging an empty lane changes no bit, so the tree
  // of numerics.hpp is walked over the active lanes only.
  for (int32_t off = lanes >> 1; off > 0; off >>= 1) {
    for (int32_t l = 0; l < off && l + off < active; ++l) {
      lane_min[l] = dd_add(lane_min[l], lane_min[l + off]);
      lane_max[l] = dd_add(lane_max[l], lane_max[l + off]);
    }
  }
}

// fold_row_activity for a row of at most max_light_row entries without a tombstone. Entry k is
// then at slot begin + k and a lane holds at most the entries l and l + lanes, so the lanes are
// summed one after the other, each in registers: every lane gets its terms in storage order,
// as in the walk over the row. The sums are locals for the same reason: the compiler cannot
// tell that the results, the lanes and the arrays of the state are different memory.
GPRS_ALWAYS_INLINE void fold_light_row(const state_t& st,
                                       int64_t begin,
                                       int32_t n,
                                       dd_t* scratch,
                                       side_sum_t& min_side,
                                       side_sum_t& max_side)
{
  gprs_assert(n >= 0 && n <= max_light_row);
  side_sum_t min_sum, max_sum;
  int32_t lanes = activity_lanes(n);
  if (lanes == 1) {
    if (n > 0) { add_entry(st, begin, min_sum, min_sum.sum, max_sum, max_sum.sum, true); }
    if (n > 1) { add_entry(st, begin + 1, min_sum, min_sum.sum, max_sum, max_sum.sum, false); }
  } else if (lanes == 2) {
    // Three or four entries: both lanes and their merge without the scratch lanes.
    dd_t odd_min, odd_max;
    add_entry(st, begin, min_sum, min_sum.sum, max_sum, max_sum.sum, true);
    add_entry(st, begin + 1, min_sum, odd_min, max_sum, odd_max, true);
    add_entry(st, begin + 2, min_sum, min_sum.sum, max_sum, max_sum.sum, false);
    if (n == 4) { add_entry(st, begin + 3, min_sum, odd_min, max_sum, odd_max, false); }
    min_sum.sum = dd_add(min_sum.sum, odd_min);
    max_sum.sum = dd_add(max_sum.sum, odd_max);
  } else {
    // More than one lane: every lane has an entry.
    dd_t* scratch_min = scratch;
    dd_t* scratch_max = scratch + max_activity_lanes;
    for (int32_t l = 0; l < lanes; ++l) {
      dd_t lane_min, lane_max;
      add_entry(st, begin + l, min_sum, lane_min, max_sum, lane_max, true);
      if (l + lanes < n) {
        add_entry(st, begin + l + lanes, min_sum, lane_min, max_sum, lane_max, false);
      }
      scratch_min[l] = lane_min;
      scratch_max[l] = lane_max;
    }
    merge_lanes(scratch_min, scratch_max, lanes, lanes);
    min_sum.sum = scratch_min[0];
    max_sum.sum = scratch_max[0];
  }
  min_side = min_sum;
  max_side = max_sum;
}

// fold_row_activity for every other row: the walk in storage order over the lanes of the state.
void fold_heavy_row(state_t& st, int32_t row, side_sum_t& min_side, side_sum_t& max_side)
{
  const line_store_t& csr = st.csr;
  int32_t n               = csr.size[row];
  int32_t lanes           = activity_lanes(n);
  int32_t active          = n < lanes ? n : lanes;
  dd_t* lane_min          = st.lane_scratch.data();
  dd_t* lane_max          = lane_min + max_activity_lanes;
  side_sum_t min_sum, max_sum;

  // The first `active` live entries open the lanes, one each.
  int64_t e = csr.begin(row);
  for (int32_t k = 0; k < active; ++e) {
    if (!csr.live(e)) { continue; }
    add_entry(st, e, min_sum, lane_min[k], max_sum, lane_max[k], true);
    ++k;
  }
  int32_t k = active;
  for (; e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t lane = k & (lanes - 1);
    ++k;
    add_entry(st, e, min_sum, lane_min[lane], max_sum, lane_max[lane], false);
  }
  gprs_assert(k == n);

  merge_lanes(lane_min, lane_max, lanes, active);
  if (active > 0) {
    min_sum.sum = lane_min[0];
    max_sum.sum = lane_max[0];
  }
  min_side = min_sum;
  max_side = max_sum;
}

inline bool light_row(const state_t& st, int32_t row)
{
  return st.csr.size[row] <= max_light_row && st.csr.used[row] == st.csr.size[row];
}

GPRS_ALWAYS_INLINE void store_side(activity_t& act, int32_t row, const side_sum_t& side)
{
  act.hi[row]        = side.sum.hi;
  act.lo[row]        = side.sum.lo;
  act.inf_count[row] = side.inf_count;
  act.inf_slot[row]  = side.inf_slot;
  act.max_term[row]  = side.max_term;
  act.updates[row]   = side.terms;
}

// The sums of a light row never leave the registers: its two results are no arguments of a
// call.
GPRS_ALWAYS_INLINE void store_row_activity(state_t& st, int32_t row)
{
  if (light_row(st, row)) {
    side_sum_t min_side, max_side;
    fold_light_row(
      st, st.csr.begin(row), st.csr.size[row], st.lane_scratch.data(), min_side, max_side);
    store_side(st.min_act, row, min_side);
    store_side(st.max_act, row, max_side);
  } else {
    side_sum_t min_side, max_side;
    fold_heavy_row(st, row, min_side, max_side);
    store_side(st.min_act, row, min_side);
    store_side(st.max_act, row, max_side);
  }
}

}  // namespace

state_t::state_t(const model_t& model, const tolerances_t& tolerances)
  : nrows(model.nrows),
    ncols(model.ncols),
    tol(tolerances),
    row_lower(model.row_lower),
    row_upper(model.row_upper),
    col_lower(model.col_lower),
    col_upper(model.col_upper),
    cost(model.cost),
    objective_offset{model.objective_offset, 0.},
    live_rows(model.nrows),
    live_cols(model.ncols),
    live_nnz(model.nnz()),
    fixed_nnz(0)
{
  gprs_assert(well_formed(model));
  int64_t nnz = model.nnz();

  csr.start.assign(model.row_start.begin(), model.row_start.end() - 1);
  csr.size.resize(nrows);
  for (int32_t i = 0; i < nrows; ++i) {
    csr.size[i] = model.row_start[i + 1] - model.row_start[i];
  }
  csr.used     = csr.size;
  csr.capacity = csr.size;
  csr.index    = model.col_index;
  csr.value    = model.value;
  csr.twin.resize(nnz);

  csc.size.assign(ncols, 0);
  for (int64_t e = 0; e < nnz; ++e) {
    ++csc.size[csr.index[e]];
  }
  csc.start.resize(ncols);
  int64_t next_start = 0;
  for (int32_t j = 0; j < ncols; ++j) {
    csc.start[j] = next_start;
    next_start += csc.size[j];
  }
  csc.used     = csc.size;
  csc.capacity = csc.size;
  csc.index.resize(nnz);
  csc.value.resize(nnz);
  csc.twin.resize(nnz);

  down_locks.assign(ncols, 0);
  up_locks.assign(ncols, 0);
  std::vector<int64_t> fill(csc.start);
  for (int32_t i = 0; i < nrows; ++i) {
    int32_t lower_finite = row_lower[i] > -inf;
    int32_t upper_finite = row_upper[i] < inf;
    for (int64_t e = csr.begin(i); e < csr.end(i); ++e) {
      int32_t j    = csr.index[e];
      double a     = csr.value[e];
      int64_t p    = fill[j]++;
      csc.index[p] = i;
      csc.value[p] = a;
      csc.twin[p]  = e;
      csr.twin[e]  = p;
      // No branch on the sign either.
      bool positive = a > 0;
      up_locks[j] += positive ? upper_finite : lower_finite;
      down_locks[j] += positive ? lower_finite : upper_finite;
    }
  }

  row_alive.assign(nrows, 1);
  row_free_size = csr.size;
  row_version.assign(nrows, 0);

  col_type.resize(ncols);
  for (int32_t j = 0; j < ncols; ++j) {
    col_type[j] = (col_type_t)model.col_type[j];
  }
  col_status.assign(ncols, col_status_t::live);
  implied_lower.assign(ncols, -inf);
  implied_upper.assign(ncols, inf);
  implied_lower_row.assign(ncols, -1);
  implied_upper_row.assign(ncols, -1);
  implied_lower_version.assign(ncols, 0);
  implied_upper_version.assign(ncols, 0);

  min_act.resize(nrows);
  max_act.resize(nrows);
  lane_scratch.resize(2 * max_activity_lanes);
  for (int32_t i = 0; i < nrows; ++i) {
    store_row_activity(*this, i);
  }

  row_frontier.init(row_list::count, nrows, row_event::entries);
  col_frontier.init(col_list::count, ncols, col_event::locks);
}

void state_t::fold_row_activity(int32_t row, side_sum_t& min_side, side_sum_t& max_side)
{
  if (light_row(*this, row)) {
    fold_light_row(*this, csr.begin(row), csr.size[row], lane_scratch.data(), min_side, max_side);
  } else {
    fold_heavy_row(*this, row, min_side, max_side);
  }
}

void state_t::recompute_row_activity(int32_t row) { store_row_activity(*this, row); }

}  // namespace gprs
