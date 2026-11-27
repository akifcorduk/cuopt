/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>

#include <cstring>
#include <type_traits>

namespace gprs {

namespace {

template <typename T>
inline uint64_t word(T v)
{
  if constexpr (std::is_floating_point_v<T>) {
    uint64_t bits;
    static_assert(sizeof(bits) == sizeof(v), "a double is hashed as one 64-bit word");
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
  } else if constexpr (std::is_enum_v<T>) {
    return (uint64_t)v;
  } else {
    return v;
  }
}

// A chain of 64-bit words through the splitmix64 finaliser, which is a bijection: with every
// other word the same, two values of one word never give the same hash. A vector is hashed as
// its length and its elements, so a value that moves from one array into the next changes the
// hash. Values are hashed, not bytes or addresses.
struct hasher_t {
  uint64_t h = 0x6a09e667f3bcc908;

  template <typename T>
  void add(T v)
  {
    uint64_t z = (h + 0x9e3779b97f4a7c15) ^ word(v);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    h          = z ^ (z >> 31);
  }

  template <typename T>
  void add(const std::vector<T>& values)
  {
    add(values.size());
    for (const T& v : values) {
      add(v);
    }
  }

  void add(dd_t v)
  {
    add(v.hi);
    add(v.lo);
  }

  // Every line with its place in the store and its used slots; value and twin of a tombstone
  // are leftovers and are not hashed.
  void add(const line_store_t& store)
  {
    add(store.start);
    add(store.used);
    add(store.capacity);
    add(store.size);
    add(store.slots());
    for (size_t line = 0; line < store.start.size(); ++line) {
      int64_t end = store.start[line] + store.used[line];
      for (int64_t s = store.start[line]; s < end; ++s) {
        add(store.index[s]);
        if (!store.live(s)) { continue; }
        add(store.value[s]);
        add(store.twin[s]);
      }
    }
  }

  void add(const activity_t& act)
  {
    add(act.hi);
    add(act.lo);
    add(act.inf_count);
    add(act.inf_slot);
    add(act.max_term);
    add(act.updates);
  }

  void add(const frontier_set_t& frontier)
  {
    add(frontier.list.size());
    for (const std::vector<int32_t>& list : frontier.list) {
      add(list);
    }
  }

  void add(const tolerances_t& tol)
  {
    add(tol.feasibility);
    add(tol.dual);
    add(tol.zero);
    add(tol.integrality);
    add(tol.infinite_term);
    add(tol.activity_error_share);
    add(tol.implied_bound_step);
    add(tol.side_error);
    add(tol.cancellation);
  }
};

}  // namespace

uint64_t model_hash(const model_t& model)
{
  hasher_t h;
  h.add(model.nrows);
  h.add(model.ncols);
  h.add(model.row_start);
  h.add(model.col_index);
  h.add(model.value);
  h.add(model.row_lower);
  h.add(model.row_upper);
  h.add(model.col_lower);
  h.add(model.col_upper);
  h.add(model.cost);
  h.add(model.col_type);
  h.add(model.objective_offset);
  return h.h;
}

// Everything a later step of the presolve reads: the fields the header names, and with them the
// layout of the stores (it decides when the engine rebuilds), versions, recorded implied bounds
// and the lists with their order. slot_visits and lane_scratch are no part of the state.
uint64_t state_hash(const state_t& st)
{
  hasher_t h;
  h.add(st.nrows);
  h.add(st.ncols);
  h.add(st.tol);
  h.add(st.csr);
  h.add(st.csc);
  h.add(st.row_lower);
  h.add(st.row_upper);
  h.add(st.row_alive);
  h.add(st.row_free_size);
  h.add(st.row_version);
  h.add(st.min_act);
  h.add(st.max_act);
  h.add(st.col_lower);
  h.add(st.col_upper);
  h.add(st.cost);
  h.add(st.col_type);
  h.add(st.col_status);
  h.add(st.down_locks);
  h.add(st.up_locks);
  h.add(st.implied_lower);
  h.add(st.implied_upper);
  h.add(st.implied_lower_row);
  h.add(st.implied_upper_row);
  h.add(st.implied_lower_version);
  h.add(st.implied_upper_version);
  h.add(st.objective_offset);
  h.add(st.live_rows);
  h.add(st.live_cols);
  h.add(st.live_nnz);
  h.add(st.fixed_nnz);
  h.add(st.bound_changes);
  h.add(st.side_changes);
  h.add(st.coefficient_changes);
  h.add(st.row_frontier);
  h.add(st.col_frontier);
  return h.h;
}

uint64_t postsolve_hash(const postsolve_t& ps)
{
  hasher_t h;
  h.add(ps.nrows);
  h.add(ps.ncols);
  h.add(ps.col_type);
  h.add(ps.row_origin);
  h.add(ps.col_origin);
  h.add(ps.order);
  h.add(ps.fixed_cols.size());
  for (const fixed_col_record_t& r : ps.fixed_cols) {
    h.add(r.col);
    h.add(r.value);
  }
  h.add(ps.aliases.size());
  for (const alias_record_t& r : ps.aliases) {
    h.add(r.col);
    h.add(r.rep);
    h.add(r.row);
    h.add(r.scale);
    h.add(r.offset);
  }
  h.add(ps.parallel_rows.size());
  for (const parallel_row_record_t& r : ps.parallel_rows) {
    h.add(r.removed_row);
    h.add(r.kept_row);
    h.add(r.ratio);
  }
  return h.h;
}

}  // namespace gprs
