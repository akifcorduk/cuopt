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
      if (a > 0) {
        up_locks[j] += upper_finite;
        down_locks[j] += lower_finite;
      } else {
        down_locks[j] += upper_finite;
        up_locks[j] += lower_finite;
      }
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
    recompute_row_activity(i);
  }

  row_frontier.init(row_list::count, nrows, row_event::entries);
  col_frontier.init(col_list::count, ncols, col_event::locks);
}

namespace {

inline void add_term(
  side_sum_t& side, dd_t& lane, double a, double bound, int64_t slot, const tolerances_t& tol)
{
  if (is_infinite_term(a, bound, tol)) {
    ++side.inf_count;
    side.inf_slot += slot;
    return;
  }
  dd_add_product(lane, a, bound);
  double magnitude = std::fabs(a * bound);
  if (magnitude > side.max_term) { side.max_term = magnitude; }
  ++side.terms;
}

}  // namespace

void state_t::fold_row_activity(int32_t row, side_sum_t& min_side, side_sum_t& max_side)
{
  int32_t n      = csr.size[row];
  int32_t lanes  = activity_lanes(n);
  int32_t active = n < lanes ? n : lanes;
  dd_t* lane_min = lane_scratch.data();
  dd_t* lane_max = lane_min + max_activity_lanes;
  for (int32_t l = 0; l < active; ++l) {
    lane_min[l] = dd_t{};
    lane_max[l] = dd_t{};
  }
  min_side = side_sum_t{};
  max_side = side_sum_t{};

  int32_t k = 0;
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t lane = k & (lanes - 1);
    ++k;
    int32_t j = csr.index[e];
    double a  = csr.value[e];
    add_term(
      min_side, lane_min[lane], a, min_activity_bound(a, col_lower[j], col_upper[j]), e, tol);
    add_term(
      max_side, lane_max[lane], a, max_activity_bound(a, col_lower[j], col_upper[j]), e, tol);
  }
  gprs_assert(k == n);

  // Lanes from `active` on hold no term, and merging an empty lane changes no bit, so the tree
  // of numerics.hpp is walked over the active lanes only.
  for (int32_t off = lanes >> 1; off > 0; off >>= 1) {
    for (int32_t l = 0; l < off && l + off < active; ++l) {
      lane_min[l] = dd_add(lane_min[l], lane_min[l + off]);
      lane_max[l] = dd_add(lane_max[l], lane_max[l + off]);
    }
  }
  if (active > 0) {
    min_side.sum = lane_min[0];
    max_side.sum = lane_max[0];
  }
}

namespace {

inline void store_side(activity_t& act, int32_t row, const side_sum_t& side)
{
  act.hi[row]        = side.sum.hi;
  act.lo[row]        = side.sum.lo;
  act.inf_count[row] = side.inf_count;
  act.inf_slot[row]  = side.inf_slot;
  act.max_term[row]  = side.max_term;
  act.updates[row]   = side.terms;
}

}  // namespace

void state_t::recompute_row_activity(int32_t row)
{
  side_sum_t min_side, max_side;
  fold_row_activity(row, min_side, max_side);
  store_side(min_act, row, min_side);
  store_side(max_act, row, max_side);
}

}  // namespace gprs
