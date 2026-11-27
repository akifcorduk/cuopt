/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/state.hpp>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace gprs {

namespace {

// Moves the live entries of a line to its front, in order.
void compact_line(line_store_t& store, line_store_t& other, int32_t line)
{
  int64_t end = store.end(line);
  int64_t w   = store.begin(line);
  for (int64_t s = w; s < end; ++s) {
    if (!store.live(s)) { continue; }
    if (w != s) {
      store.index[w]            = store.index[s];
      store.value[w]            = store.value[s];
      store.twin[w]             = store.twin[s];
      other.twin[store.twin[s]] = w;
      store.index[s]            = dead_index;
    }
    ++w;
  }
  store.used[line] = store.size[line];
}

void resize_lines(line_store_t& store, int32_t lines, int64_t slots)
{
  store.start.resize(lines);
  store.used.resize(lines);
  store.capacity.resize(lines);
  store.size.resize(lines);
  store.index.resize(slots);
  store.value.resize(slots);
  store.twin.resize(slots);
}

inline void set_line(line_store_t& store, int32_t line, int64_t start, int64_t end)
{
  store.start[line]    = start;
  store.used[line]     = end - start;
  store.capacity[line] = end - start;
  store.size[line]     = end - start;
}

}  // namespace

void state_t::compact_sparse_lines()
{
  std::vector<int32_t> lines;
  row_frontier.take(row_list::compaction, lines);
  for (int32_t row : lines) {
    slot_visits += 2 * (int64_t)csr.used[row];
    compact_line(csr, csc, row);
    recompute_row_activity(row);
  }
  col_frontier.take(col_list::compaction, lines);
  for (int32_t col : lines) {
    slot_visits += csc.used[col];
    compact_line(csc, csr, col);
  }
}

namespace {

// Sides and sums of fixed terms up to this size are out of reach of every limit that a family
// puts on the size of a side: tol.feasibility / tol.side_error, 1e8 by default, for a row that
// row tightening has scaled by up to 1000.
constexpr double plain_size = 1e5;

// The finite sides of a row lose the sum of the terms of its fixed columns, `fixed_terms`, whose
// absolute values sum to `fixed_size`: one compensated sum in storage order, and the same
// arithmetic wherever a model is read from the state (dd_residual).
// Returns whether a family can treat the row otherwise now. It cannot when the move is plain:
// the sum is one double, every new side is the exact difference, no side, no sum and no term
// that the activities of the row have held since they were summed is above plain_size, and no
// new side lies within the feasibility tolerance of an integer without being one. Then a side
// of the free part, an activity and a candidate are the same numbers before and after, the
// error bounds of the activities are far below what a test compares them with, whatever the
// number of terms, and row tightening, which rounds the sides of an integer row only when they
// are off by more than the tolerance while the row holds such a term, has nothing to round.
// Called before the activities of the row are recomputed.
bool move_fixed_terms(state_t& st, int32_t row, dd_t fixed_terms, double fixed_size)
{
  bool plain = fixed_terms.lo == 0. && fixed_size <= plain_size &&
               st.min_act.max_term[row] <= plain_size && st.max_act.max_term[row] <= plain_size;
  bool ordered = st.row_lower[row] <= st.row_upper[row];
  for (double* side : {&st.row_lower[row], &st.row_upper[row]}) {
    if (std::isinf(*side)) { continue; }
    double error;
    two_sum(*side, -fixed_terms.hi, error);
    double moved = dd_residual(*side, fixed_terms);
    gprs_assert(std::isfinite(moved));
    double nearest = std::round(moved);
    plain          = plain && error == 0. && std::fabs(*side) <= plain_size &&
            std::fabs(moved) <= plain_size &&
            (moved == nearest || std::fabs(moved - nearest) > st.tol.feasibility);
    *side = moved;
  }
  gprs_assert(!ordered || st.row_lower[row] <= st.row_upper[row]);
  return !plain;
}

// The terms of the fixed columns of a row leave it for its sides, and the row is compacted.
void absorb_row(state_t& st, int32_t row)
{
  line_store_t& csr = st.csr;
  line_store_t& csc = st.csc;
  dd_t fixed_terms;
  double fixed_size = 0.;
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if (st.col_status[col] != col_status_t::fixed) { continue; }
    dd_add_product(fixed_terms, csr.value[e], st.col_lower[col]);
    fixed_size += std::fabs(csr.value[e] * st.col_lower[col]);
    csc.index[csr.twin[e]] = dead_index;
    csr.index[e]           = dead_index;
    --csc.size[col];
    --csr.size[row];
    --st.live_nnz;
    --st.fixed_nnz;
  }
  bool listed = move_fixed_terms(st, row, fixed_terms, fixed_size);
  st.slot_visits += 3 * (int64_t)csr.used[row];
  compact_line(csr, csc, row);
  st.recompute_row_activity(row);
  if (listed) { st.row_frontier.push(row_event::entries, row); }
}

}  // namespace

void state_t::absorb_fixed_terms()
{
  for (int32_t col = 0; col < ncols; ++col) {
    if (col_status[col] != col_status_t::fixed) { continue; }
    // absorb_row tombstones every entry of a fixed column in the row: the one it is called
    // for, and those of the fixed columns that this loop has yet to reach.
    if (csc.size[col] > 0) {
      slot_visits += csc.used[col];
      for (int64_t p = csc.begin(col); p < csc.end(col); ++p) {
        if (csc.live(p)) { absorb_row(*this, csc.index[p]); }
      }
    }
    gprs_assert(csc.size[col] == 0);
    down_locks[col] = 0;
    up_locks[col]   = 0;
  }
  gprs_assert(fixed_nnz == 0);
}

namespace {

// The entry at slot `from` of a store goes to the free slot `to` of the same store.
inline void move_slot(line_store_t& store, line_store_t& other, int64_t from, int64_t to)
{
  int64_t twin     = store.twin[from];
  store.index[to]  = store.index[from];
  store.value[to]  = store.value[from];
  store.twin[to]   = twin;
  other.twin[twin] = to;
}

void truncate(line_store_t& store, int64_t slots)
{
  store.index.resize(slots);
  store.value.resize(slots);
  store.twin.resize(slots);
}

// True when the lines lie in the store in index order, so that the kept entries of a line can
// move to the front of the store without running over an entry that is still to be read.
// move_entry breaks it for the columns: it moves a full column to the end of the store.
bool lines_in_order(const line_store_t& store, int32_t lines)
{
  int64_t end = 0;
  for (int32_t l = 0; l < lines; ++l) {
    if (store.start[l] < end) { return false; }
    end = store.start[l] + store.capacity[l];
  }
  return true;
}

}  // namespace

// Both stores are compacted in place where their lines are in index order, which the rows always
// are: no new store is allocated, and a line in front of the first dropped slot is not touched.
void state_t::rebuild()
{
  int64_t kept = live_nnz - fixed_nnz;
  slot_visits += csr.slots() + csc.slots() + nrows + ncols;

  int64_t w = 0;
  for (int32_t i = 0; i < nrows; ++i) {
    int64_t begin = csr.begin(i);
    int64_t end   = csr.end(i);
    int64_t start = w;
    gprs_assert(begin >= w);
    if (csr.used[i] == row_free_size[i]) {
      // Every used slot holds an entry of a live column: the row moves as it is.
      if (begin == w) {
        w = end;
      } else {
        for (int64_t e = begin; e < end; ++e, ++w) {
          move_slot(csr, csc, e, w);
        }
        int64_t shift = start - begin;
        min_act.inf_slot[i] += min_act.inf_count[i] * shift;
        max_act.inf_slot[i] += max_act.inf_count[i] * shift;
      }
      set_line(csr, i, start, w);
      continue;
    }
    dd_t fixed_terms;
    double fixed_size = 0.;
    bool has_fixed    = false;
    for (int64_t e = begin; e < end; ++e) {
      if (!csr.live(e)) { continue; }
      int32_t col = csr.index[e];
      if (col_status[col] == col_status_t::fixed) {
        dd_add_product(fixed_terms, csr.value[e], col_lower[col]);
        fixed_size += std::fabs(csr.value[e] * col_lower[col]);
        has_fixed = true;
        continue;
      }
      if (w != e) { move_slot(csr, csc, e, w); }
      ++w;
    }
    set_line(csr, i, start, w);
    gprs_assert(csr.size[i] == row_free_size[i]);
    if (has_fixed && move_fixed_terms(*this, i, fixed_terms, fixed_size)) {
      row_frontier.push(row_event::entries, i);
    }
    recompute_row_activity(i);
  }
  gprs_assert(w == kept);
  truncate(csr, kept);

  // The twin of a kept entry in the columns is by now its slot in the compacted rows.
  bool in_place = lines_in_order(csc, ncols);
  line_store_t cols;
  if (!in_place) { resize_lines(cols, ncols, kept); }
  line_store_t& target = in_place ? csc : cols;
  int64_t q            = 0;
  for (int32_t j = 0; j < ncols; ++j) {
    int64_t begin = csc.begin(j);
    int64_t end   = csc.end(j);
    int64_t start = q;
    if (col_status[j] == col_status_t::fixed) {
      col_status[j] = col_status_t::removed;
      down_locks[j] = 0;
      up_locks[j]   = 0;
    } else if (in_place && csc.used[j] == csc.size[j] && begin == q) {
      q = end;
    } else {
      for (int64_t p = begin; p < end; ++p) {
        if (!csc.live(p)) { continue; }
        if (!in_place || q != p) {
          int64_t e       = csc.twin[p];
          target.index[q] = csc.index[p];
          target.value[q] = csc.value[p];
          target.twin[q]  = e;
          csr.twin[e]     = q;
        }
        ++q;
      }
    }
    set_line(target, j, start, q);
  }
  gprs_assert(q == kept);
  if (in_place) {
    truncate(csc, kept);
  } else {
    csc = std::move(cols);
  }

  live_nnz  = kept;
  fixed_nnz = 0;
  std::vector<int32_t> served;
  row_frontier.take(row_list::compaction, served);
  col_frontier.take(col_list::compaction, served);
  // Plan 2.4: with assertions on, a rebuild is compared with a full recomputation.
  gprs_assert((check_consistency(), true));
}

model_t extract_model(const state_t& st,
                      std::vector<int32_t>& row_origin,
                      std::vector<int32_t>& col_origin)
{
  gprs_assert(st.fixed_nnz == 0);
  gprs_assert(st.csr.slots() == st.live_nnz && st.csc.slots() == st.live_nnz);
  model_t model;
  model.nrows = st.live_rows;
  model.ncols = st.live_cols;
  row_origin.clear();
  col_origin.clear();

  std::vector<int32_t> row_index(st.nrows, -1);
  std::vector<int64_t> fill;
  model.row_start.push_back(0);
  for (int32_t i = 0; i < st.nrows; ++i) {
    if (!st.row_alive[i]) { continue; }
    row_index[i] = row_origin.size();
    row_origin.push_back(i);
    fill.push_back(model.row_start.back());
    model.row_start.push_back(model.row_start.back() + st.csr.size[i]);
    model.row_lower.push_back(st.row_lower[i]);
    model.row_upper.push_back(st.row_upper[i]);
  }
  gprs_assert(model.row_start.back() == st.live_nnz);
  model.col_index.resize(st.live_nnz);
  model.value.resize(st.live_nnz);

  for (int32_t j = 0; j < st.ncols; ++j) {
    if (!st.is_live(j)) { continue; }
    int32_t col = col_origin.size();
    col_origin.push_back(j);
    model.col_lower.push_back(st.col_lower[j]);
    model.col_upper.push_back(st.col_upper[j]);
    model.cost.push_back(st.cost[j]);
    model.col_type.push_back((uint8_t)st.col_type[j]);
    for (int64_t p = st.csc.begin(j); p < st.csc.end(j); ++p) {
      int64_t e          = fill[row_index[st.csc.index[p]]]++;
      model.col_index[e] = col;
      model.value[e]     = st.csc.value[p];
    }
  }
  gprs_assert((int32_t)row_origin.size() == model.nrows);
  gprs_assert((int32_t)col_origin.size() == model.ncols);
  model.objective_offset = dd_value(st.objective_offset);
  return model;
}

// One pass over the rows of the state: a row is copied in storage order, which is increasing
// column order unless move_entry changed it, and sorted only when it is not.
model_t reduced_model(const state_t& st,
                      std::vector<int32_t>& row_origin,
                      std::vector<int32_t>& col_origin)
{
  const line_store_t& csr = st.csr;
  int64_t nnz             = st.free_nnz();
  model_t model;
  model.nrows = st.live_rows;
  model.ncols = st.live_cols;

  // Index in the model of a live column.
  std::vector<int32_t> new_col(st.ncols);
  col_origin.resize(model.ncols);
  model.col_lower.resize(model.ncols);
  model.col_upper.resize(model.ncols);
  model.cost.resize(model.ncols);
  model.col_type.resize(model.ncols);
  int32_t cols = 0;
  for (int32_t j = 0; j < st.ncols; ++j) {
    if (!st.is_live(j)) { continue; }
    new_col[j]            = cols;
    col_origin[cols]      = j;
    model.col_lower[cols] = st.col_lower[j];
    model.col_upper[cols] = st.col_upper[j];
    model.cost[cols]      = st.cost[j];
    model.col_type[cols]  = (uint8_t)st.col_type[j];
    ++cols;
  }
  gprs_assert(cols == model.ncols);

  row_origin.resize(model.nrows);
  model.row_start.resize(model.nrows + 1);
  model.row_lower.resize(model.nrows);
  model.row_upper.resize(model.nrows);
  model.col_index.resize(nnz);
  model.value.resize(nnz);
  std::vector<std::pair<int32_t, double>> unsorted;
  int64_t w    = 0;
  int32_t rows = 0;
  for (int32_t i = 0; i < st.nrows; ++i) {
    if (!st.row_alive[i]) { continue; }
    int64_t start = w;
    double lower  = st.row_lower[i];
    double upper  = st.row_upper[i];
    int32_t last  = -1;
    bool sorted   = true;
    if (csr.used[i] == st.row_free_size[i]) {
      // Every used slot holds an entry of a live column.
      for (int64_t e = csr.begin(i); e < csr.end(i); ++e, ++w) {
        gprs_assert(st.is_live(csr.index[e]));
        int32_t col        = new_col[csr.index[e]];
        sorted             = sorted & (col > last);
        last               = col;
        model.col_index[w] = col;
        model.value[w]     = csr.value[e];
      }
    } else {
      // As in rebuild(): the terms of the fixed columns, one compensated sum in storage order,
      // leave the finite sides.
      dd_t fixed_terms;
      bool has_fixed = false;
      for (int64_t e = csr.begin(i); e < csr.end(i); ++e) {
        if (!csr.live(e)) { continue; }
        int32_t j = csr.index[e];
        if (st.col_status[j] == col_status_t::fixed) {
          dd_add_product(fixed_terms, csr.value[e], st.col_lower[j]);
          has_fixed = true;
          continue;
        }
        gprs_assert(st.is_live(j));
        int32_t col        = new_col[j];
        sorted             = sorted & (col > last);
        last               = col;
        model.col_index[w] = col;
        model.value[w]     = csr.value[e];
        ++w;
      }
      if (has_fixed) {
        if (lower > -inf) { lower = dd_residual(lower, fixed_terms); }
        if (upper < inf) { upper = dd_residual(upper, fixed_terms); }
        gprs_assert(std::isfinite(lower) || lower == -inf);
        gprs_assert(std::isfinite(upper) || upper == inf);
        gprs_assert(!(st.row_lower[i] <= st.row_upper[i]) || lower <= upper);
      }
    }
    gprs_assert(w - start == st.row_free_size[i]);
    if (!sorted) {
      // A row holds a column at most once, so the order is unique.
      unsorted.clear();
      for (int64_t e = start; e < w; ++e) {
        unsorted.emplace_back(model.col_index[e], model.value[e]);
      }
      std::sort(unsorted.begin(),
                unsorted.end(),
                [](const std::pair<int32_t, double>& a, const std::pair<int32_t, double>& b) {
                  return a.first < b.first;
                });
      for (int64_t e = start; e < w; ++e) {
        model.col_index[e] = unsorted[e - start].first;
        model.value[e]     = unsorted[e - start].second;
      }
    }
    row_origin[rows]      = i;
    model.row_start[rows] = start;
    model.row_lower[rows] = lower;
    model.row_upper[rows] = upper;
    ++rows;
  }
  gprs_assert(rows == model.nrows && w == nnz);
  model.row_start[rows]  = w;
  model.objective_offset = dd_value(st.objective_offset);
  return model;
}

// ---- Consistency ----

namespace {

#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
std::string
describe(const char* format, ...)
{
  char text[400];
  va_list args;
  va_start(args, format);
  std::vsnprintf(text, sizeof(text), format, args);
  va_end(args);
  return text;
}

#define GPRS_STATE_INVARIANT(condition, ...)            \
  do {                                                  \
    if (!(condition)) { return describe(__VA_ARGS__); } \
  } while (false)

inline bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

// Sizes of the arrays of a store and the slot range of every line.
std::string check_lines(const line_store_t& store, int32_t lines, const char* name)
{
  size_t n = lines;
  GPRS_STATE_INVARIANT(store.start.size() == n && store.used.size() == n &&
                         store.capacity.size() == n && store.size.size() == n,
                       "%s: the line arrays do not have one element per line",
                       name);
  GPRS_STATE_INVARIANT(
    store.value.size() == store.index.size() && store.twin.size() == store.index.size(),
    "%s: index, value and twin differ in length",
    name);
  for (int32_t l = 0; l < lines; ++l) {
    GPRS_STATE_INVARIANT(
      0 <= store.size[l] && store.size[l] <= store.used[l] && store.used[l] <= store.capacity[l],
      "%s line %d: 0 <= size %d <= used %d <= capacity %d fails",
      name,
      l,
      store.size[l],
      store.used[l],
      store.capacity[l]);
    GPRS_STATE_INVARIANT(store.start[l] >= 0 && store.start[l] + store.capacity[l] <= store.slots(),
                         "%s line %d: slots [%lld, %lld) lie outside the store of %lld slots",
                         name,
                         l,
                         (long long)store.start[l],
                         (long long)(store.start[l] + store.capacity[l]),
                         (long long)store.slots());
  }
  return {};
}

// Ownership of the slots, tombstones outside the used parts, sizes, twins and values of the
// entries of one store against the other store.
std::string check_entries(const line_store_t& store,
                          const line_store_t& other,
                          int32_t lines,
                          int32_t other_lines,
                          const char* name,
                          const char* other_name)
{
  // 0: no line owns the slot, 1: spare slot of a line, 2: used slot of a line.
  std::vector<uint8_t> owner(store.slots(), 0);
  for (int32_t l = 0; l < lines; ++l) {
    for (int64_t s = store.start[l]; s < store.start[l] + store.capacity[l]; ++s) {
      GPRS_STATE_INVARIANT(
        owner[s] == 0, "%s line %d: slot %lld belongs to another line too", name, l, (long long)s);
      owner[s] = 1;
    }
    int32_t live = 0;
    for (int64_t s = store.begin(l); s < store.end(l); ++s) {
      owner[s] = 2;
      if (!store.live(s)) { continue; }
      ++live;
      int32_t k = store.index[s];
      GPRS_STATE_INVARIANT(k >= 0 && k < other_lines,
                           "%s line %d slot %lld: index %d is out of range",
                           name,
                           l,
                           (long long)s,
                           k);
      int64_t t = store.twin[s];
      GPRS_STATE_INVARIANT(t >= other.begin(k) && t < other.end(k),
                           "%s line %d slot %lld: twin %lld is not a used slot of %s line %d",
                           name,
                           l,
                           (long long)s,
                           (long long)t,
                           other_name,
                           k);
      GPRS_STATE_INVARIANT(other.index[t] == l,
                           "%s line %d slot %lld: twin %lld holds index %d",
                           name,
                           l,
                           (long long)s,
                           (long long)t,
                           other.index[t]);
      GPRS_STATE_INVARIANT(other.twin[t] == s,
                           "%s line %d slot %lld: twin %lld points back to slot %lld",
                           name,
                           l,
                           (long long)s,
                           (long long)t,
                           (long long)other.twin[t]);
      GPRS_STATE_INVARIANT(same_bits(other.value[t], store.value[s]),
                           "%s line %d slot %lld: value %.17g, twin value %.17g",
                           name,
                           l,
                           (long long)s,
                           store.value[s],
                           other.value[t]);
      GPRS_STATE_INVARIANT(std::isfinite(store.value[s]) && store.value[s] != 0.,
                           "%s line %d slot %lld: value %.17g is zero or not finite",
                           name,
                           l,
                           (long long)s,
                           store.value[s]);
    }
    GPRS_STATE_INVARIANT(
      live == store.size[l], "%s line %d: size %d, live entries %d", name, l, store.size[l], live);
  }
  for (int64_t s = 0; s < store.slots(); ++s) {
    GPRS_STATE_INVARIANT(owner[s] == 2 || !store.live(s),
                         "%s slot %lld: a slot outside the used part of every line is not a "
                         "tombstone",
                         name,
                         (long long)s);
  }
  return {};
}

std::string check_frontier(const frontier_set_t& frontier,
                           int32_t lists,
                           int32_t n,
                           const char* name)
{
  GPRS_STATE_INVARIANT(
    frontier.pending.size() == (size_t)n && frontier.list.size() == (size_t)lists,
    "%s: wrong number of lists or pending flags",
    name);
  std::vector<uint8_t> listed(n, 0);
  for (int32_t l = 0; l < lists; ++l) {
    for (int32_t i : frontier.list[l]) {
      GPRS_STATE_INVARIANT(i >= 0 && i < n, "%s list %d: index %d is out of range", name, l, i);
      GPRS_STATE_INVARIANT(
        !((listed[i] >> l) & 1), "%s list %d: index %d is on the list twice", name, l, i);
      listed[i] |= 1 << l;
    }
  }
  for (int32_t i = 0; i < n; ++i) {
    GPRS_STATE_INVARIANT(listed[i] == frontier.pending[i],
                         "%s index %d: pending flags %#x, on the lists %#x",
                         name,
                         i,
                         (unsigned)frontier.pending[i],
                         (unsigned)listed[i]);
  }
  return {};
}

// A bound of |hi + lo - sum of the terms| for an accumulator that took `updates` terms of
// absolute value up to max_term through dd_add_product, dd_sub_product and dd_add, in any order:
// |hi| stays below updates * max_term, so each error that enters a lo is below
// u * (updates + 1) * max_term and |lo| below 2 * updates times that, and each of the at most
// 4 * updates additions into a lo rounds by at most u * |lo|. The cube of updates + 1 also covers
// the rounding of the comparison in check_activity, the last term products that underflow.
// activity_error(updates, max_term) cannot serve as the tolerance of the check: valid states
// exceed it, because the rounding of an addition into lo follows the size of the sum, not of
// the largest term (a row of 8 terms of one sign and similar size, after two bound changes).
inline double proven_activity_error(uint32_t updates, double max_term)
{
  double n = updates + 1.;
  return 9. * n * n * n * rounding_unit * rounding_unit * max_term +
         n * std::numeric_limits<double>::denorm_min();
}

std::string check_activity(
  const state_t& st, const activity_t& act, const side_sum_t& fresh, int32_t row, const char* name)
{
  GPRS_STATE_INVARIANT(act.inf_count[row] == fresh.inf_count,
                       "%s activity of row %d: inf_count %d, from scratch %d",
                       name,
                       row,
                       act.inf_count[row],
                       fresh.inf_count);
  GPRS_STATE_INVARIANT(act.inf_slot[row] == fresh.inf_slot,
                       "%s activity of row %d: inf_slot %lld, from scratch %lld",
                       name,
                       row,
                       (long long)act.inf_slot[row],
                       (long long)fresh.inf_slot);
  GPRS_STATE_INVARIANT(std::isfinite(act.hi[row]) && std::isfinite(act.lo[row]),
                       "%s activity of row %d: hi %.17g or lo %.17g is not finite",
                       name,
                       row,
                       act.hi[row],
                       act.lo[row]);
  GPRS_STATE_INVARIANT(act.updates[row] >= fresh.terms,
                       "%s activity of row %d: %u updates, fewer than its %u finite terms",
                       name,
                       row,
                       act.updates[row],
                       fresh.terms);
  GPRS_STATE_INVARIANT(act.max_term[row] >= fresh.max_term,
                       "%s activity of row %d: max_term %.17g is below the largest term %.17g",
                       name,
                       row,
                       act.max_term[row],
                       fresh.max_term);
  double rounding;
  double difference = two_sum(act.hi[row], -fresh.sum.hi, rounding);
  difference += rounding + (act.lo[row] - fresh.sum.lo);
  double bound = proven_activity_error(act.updates[row], act.max_term[row]) +
                 proven_activity_error(fresh.terms, fresh.max_term);
  GPRS_STATE_INVARIANT(std::fabs(difference) <= bound,
                       "%s activity of row %d: (%.17g, %.17g) differs from the sum from scratch "
                       "(%.17g, %.17g) by %.3g, error bound %.3g",
                       name,
                       row,
                       act.hi[row],
                       act.lo[row],
                       fresh.sum.hi,
                       fresh.sum.lo,
                       difference,
                       bound);
  bool stale = act.updates[row] >= 2u * st.csr.size[row] &&
               activity_error(act.updates[row], act.max_term[row]) >
                 st.tol.activity_error_share * st.tol.feasibility;
  GPRS_STATE_INVARIANT(!stale,
                       "%s activity of row %d: error bound %.3g after %u updates on %d entries "
                       "and the row was not summed again",
                       name,
                       row,
                       activity_error(act.updates[row], act.max_term[row]),
                       act.updates[row],
                       st.csr.size[row]);
  return {};
}

inline bool activity_empty(const activity_t& act, int32_t row)
{
  return act.hi[row] == 0. && act.lo[row] == 0. && act.inf_count[row] == 0 &&
         act.inf_slot[row] == 0 && act.max_term[row] == 0. && act.updates[row] == 0;
}

}  // namespace

std::string state_t::consistency_error()
{
  size_t rows = nrows, cols = ncols;
  GPRS_STATE_INVARIANT(nrows >= 0 && ncols >= 0, "negative number of rows or columns");
  GPRS_STATE_INVARIANT(
    row_lower.size() == rows && row_upper.size() == rows && row_alive.size() == rows &&
      row_free_size.size() == rows && row_version.size() == rows && min_act.hi.size() == rows &&
      min_act.lo.size() == rows && min_act.inf_count.size() == rows &&
      min_act.inf_slot.size() == rows && min_act.max_term.size() == rows &&
      min_act.updates.size() == rows && max_act.hi.size() == rows && max_act.lo.size() == rows &&
      max_act.inf_count.size() == rows && max_act.inf_slot.size() == rows &&
      max_act.max_term.size() == rows && max_act.updates.size() == rows,
    "a row array does not have one element per row");
  GPRS_STATE_INVARIANT(
    col_lower.size() == cols && col_upper.size() == cols && cost.size() == cols &&
      col_type.size() == cols && col_status.size() == cols && down_locks.size() == cols &&
      up_locks.size() == cols && implied_lower.size() == cols && implied_upper.size() == cols &&
      implied_lower_row.size() == cols && implied_upper_row.size() == cols &&
      implied_lower_version.size() == cols && implied_upper_version.size() == cols,
    "a column array does not have one element per column");
  GPRS_STATE_INVARIANT(lane_scratch.size() == (size_t)(2 * max_activity_lanes),
                       "lane_scratch does not hold 2 * max_activity_lanes lanes");

  std::string error = check_lines(csr, nrows, "csr");
  if (error.empty()) { error = check_lines(csc, ncols, "csc"); }
  if (error.empty()) { error = check_entries(csr, csc, nrows, ncols, "csr", "csc"); }
  if (error.empty()) { error = check_entries(csc, csr, ncols, nrows, "csc", "csr"); }
  if (error.empty()) { error = check_frontier(row_frontier, row_list::count, nrows, "rows"); }
  if (error.empty()) { error = check_frontier(col_frontier, col_list::count, ncols, "columns"); }
  if (!error.empty()) { return error; }

  // Columns: flags, bounds and records.
  int32_t count_live_cols = 0;
  int64_t count_fixed_nnz = 0, count_col_nnz = 0;
  for (int32_t j = 0; j < ncols; ++j) {
    GPRS_STATE_INVARIANT(
      col_type[j] == col_type_t::continuous || col_type[j] == col_type_t::integer,
      "column %d: type %d",
      j,
      (int)col_type[j]);
    GPRS_STATE_INVARIANT(col_status[j] == col_status_t::live ||
                           col_status[j] == col_status_t::fixed ||
                           col_status[j] == col_status_t::removed,
                         "column %d: status %d",
                         j,
                         (int)col_status[j]);
    GPRS_STATE_INVARIANT(col_lower[j] < inf && col_upper[j] > -inf,
                         "column %d: bounds [%.17g, %.17g] hold NaN, lower +inf or upper -inf",
                         j,
                         col_lower[j],
                         col_upper[j]);
    GPRS_STATE_INVARIANT(std::isfinite(cost[j]), "column %d: cost %.17g", j, cost[j]);
    count_col_nnz += csc.size[j];
    if (col_status[j] == col_status_t::live) {
      ++count_live_cols;
      GPRS_STATE_INVARIANT(
        csc.used[j] - csc.size[j] <= csc.size[j] || (col_frontier.pending[j] & col_event::sparse),
        "column %d: %d live entries in %d used slots and not on the compaction list",
        j,
        csc.size[j],
        csc.used[j]);
    } else if (col_status[j] == col_status_t::fixed) {
      count_fixed_nnz += csc.size[j];
      GPRS_STATE_INVARIANT(col_lower[j] == col_upper[j] && std::isfinite(col_lower[j]),
                           "fixed column %d: bounds [%.17g, %.17g]",
                           j,
                           col_lower[j],
                           col_upper[j]);
    } else {
      GPRS_STATE_INVARIANT(csc.size[j] == 0, "removed column %d: %d live entries", j, csc.size[j]);
    }
    GPRS_STATE_INVARIANT(implied_lower_row[j] >= -1 && implied_lower_row[j] < nrows &&
                           implied_upper_row[j] >= -1 && implied_upper_row[j] < nrows,
                         "column %d: source row of an implied bound out of range (%d, %d)",
                         j,
                         implied_lower_row[j],
                         implied_upper_row[j]);
    GPRS_STATE_INVARIANT(!std::isnan(implied_lower[j]) && !std::isnan(implied_upper[j]),
                         "column %d: an implied bound is NaN",
                         j);
    GPRS_STATE_INVARIANT(implied_lower_row[j] >= 0 || implied_lower[j] == -inf,
                         "column %d: implied lower bound %.17g without a source row",
                         j,
                         implied_lower[j]);
    GPRS_STATE_INVARIANT(implied_upper_row[j] >= 0 || implied_upper[j] == inf,
                         "column %d: implied upper bound %.17g without a source row",
                         j,
                         implied_upper[j]);
    GPRS_STATE_INVARIANT(!is_integer(j) || (implied_lower_row[j] < 0 && implied_upper_row[j] < 0),
                         "integer column %d has a recorded implied bound",
                         j);
  }
  GPRS_STATE_INVARIANT(count_live_cols == live_cols,
                       "live_cols %d, columns with status live %d",
                       live_cols,
                       count_live_cols);
  GPRS_STATE_INVARIANT(count_col_nnz == live_nnz,
                       "live_nnz %lld, live entries of the csc %lld",
                       (long long)live_nnz,
                       (long long)count_col_nnz);
  GPRS_STATE_INVARIANT(count_fixed_nnz == fixed_nnz,
                       "fixed_nnz %lld, live entries of fixed columns %lld",
                       (long long)fixed_nnz,
                       (long long)count_fixed_nnz);

  // Rows: flags, free sizes, locks, duplicate entries and activities.
  std::vector<int32_t> count_down(ncols, 0), count_up(ncols, 0), last_row(ncols, -1);
  int32_t count_live_rows = 0;
  int64_t count_row_nnz   = 0;
  for (int32_t i = 0; i < nrows; ++i) {
    GPRS_STATE_INVARIANT(row_alive[i] <= 1, "row %d: alive flag %d", i, row_alive[i]);
    GPRS_STATE_INVARIANT(row_lower[i] < inf && row_upper[i] > -inf,
                         "row %d: sides [%.17g, %.17g] hold NaN, lower +inf or upper -inf",
                         i,
                         row_lower[i],
                         row_upper[i]);
    count_row_nnz += csr.size[i];
    if (!row_alive[i]) {
      GPRS_STATE_INVARIANT(csr.size[i] == 0 && row_free_size[i] == 0,
                           "deleted row %d: %d live entries, free size %d",
                           i,
                           csr.size[i],
                           row_free_size[i]);
      GPRS_STATE_INVARIANT(activity_empty(min_act, i) && activity_empty(max_act, i),
                           "deleted row %d: its activities are not those of an empty row",
                           i);
      continue;
    }
    ++count_live_rows;
    GPRS_STATE_INVARIANT(
      csr.used[i] - csr.size[i] <= csr.size[i] || (row_frontier.pending[i] & row_event::sparse),
      "row %d: %d live entries in %d used slots and not on the compaction list",
      i,
      csr.size[i],
      csr.used[i]);
    int32_t lower_finite = row_lower[i] > -inf;
    int32_t upper_finite = row_upper[i] < inf;
    int32_t free_size    = 0;
    for (int64_t e = csr.begin(i); e < csr.end(i); ++e) {
      if (!csr.live(e)) { continue; }
      int32_t j = csr.index[e];
      GPRS_STATE_INVARIANT(last_row[j] != i, "row %d: two live entries in column %d", i, j);
      last_row[j] = i;
      free_size += is_live(j);
      if (csr.value[e] > 0) {
        count_up[j] += upper_finite;
        count_down[j] += lower_finite;
      } else {
        count_down[j] += upper_finite;
        count_up[j] += lower_finite;
      }
    }
    GPRS_STATE_INVARIANT(free_size == row_free_size[i],
                         "row %d: row_free_size %d, live entries in live columns %d",
                         i,
                         row_free_size[i],
                         free_size);
    side_sum_t fresh_min, fresh_max;
    fold_row_activity(i, fresh_min, fresh_max);
    error = check_activity(*this, min_act, fresh_min, i, "minimum");
    if (error.empty()) { error = check_activity(*this, max_act, fresh_max, i, "maximum"); }
    if (!error.empty()) { return error; }
  }
  GPRS_STATE_INVARIANT(
    count_live_rows == live_rows, "live_rows %d, rows alive %d", live_rows, count_live_rows);
  GPRS_STATE_INVARIANT(count_row_nnz == live_nnz,
                       "live_nnz %lld, live entries of the csr %lld",
                       (long long)live_nnz,
                       (long long)count_row_nnz);
  for (int32_t j = 0; j < ncols; ++j) {
    GPRS_STATE_INVARIANT(down_locks[j] == count_down[j] && up_locks[j] == count_up[j],
                         "column %d: locks down %d up %d, counted down %d up %d",
                         j,
                         down_locks[j],
                         up_locks[j],
                         count_down[j],
                         count_up[j]);
  }
  GPRS_STATE_INVARIANT(std::isfinite(objective_offset.hi) && std::isfinite(objective_offset.lo),
                       "objective offset (%.17g, %.17g) is not finite",
                       objective_offset.hi,
                       objective_offset.lo);
  return {};
}

void state_t::check_consistency()
{
  std::string error = consistency_error();
  if (!error.empty()) { assert_fail(error.c_str(), __FILE__, __LINE__); }
}

}  // namespace gprs
