/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <gprs/numerics.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace gprs {

// minimise cost * x + objective_offset
// subject to row_lower <= A x <= row_upper, col_lower <= x <= col_upper, x[j] integer where
// col_type[j] is 1.
// A is compact CSR: column indices strictly increasing inside a row, every value finite and not
// zero. Infinite sides and bounds are IEEE infinities. lower > upper is allowed (infeasible).
struct model_t {
  int32_t nrows = 0;
  int32_t ncols = 0;
  std::vector<int64_t> row_start;
  std::vector<int32_t> col_index;
  std::vector<double> value;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<double> cost;
  std::vector<uint8_t> col_type;
  double objective_offset = 0.;

  int64_t nnz() const { return value.size(); }
};

// The contract of model_t, checked in full. O(nonzeros).
bool well_formed(const model_t& model);

enum class col_type_t : uint8_t { continuous = 0, integer = 1 };

// live: in the problem.
// fixed: lower == upper; its entries stay in the rows as constant terms until the next rebuild.
// removed: substituted, or fixed and dropped by a rebuild; it has no entries.
enum class col_status_t : uint8_t { live = 0, fixed = 1, removed = 2 };

constexpr int32_t dead_index = -1;

// One orientation of the matrix. CSR: a line is a row, index holds columns. CSC: a line is a
// column, index holds rows. Line l owns the slots [start[l], start[l] + capacity[l]); the first
// used[l] of them hold its entries, size[l] live ones and tombstones (index == dead_index).
// twin[s] is the slot of the same entry in the other orientation.
// A slot of a line stays valid until that line is moved (move_entry, for the target column),
// compacted (compact_sparse_lines) or the state is rebuilt.
// Order: at construction the entries of a row are in increasing column order and those of a
// column in increasing row order. move_entry breaks both (the row keeps the slot for another
// column, the column gets the entry at its end) and no operation restores them, so nothing may
// rely on the order of a line; extract_model sorts the rows it returns.
struct line_store_t {
  std::vector<int64_t> start;
  std::vector<int32_t> used;
  std::vector<int32_t> capacity;
  std::vector<int32_t> size;
  std::vector<int32_t> index;
  std::vector<double> value;
  std::vector<int64_t> twin;

  int64_t begin(int32_t line) const { return start[line]; }
  int64_t end(int32_t line) const { return start[line] + used[line]; }
  bool live(int64_t slot) const { return index[slot] != dead_index; }
  int64_t slots() const { return index.size(); }
};

// Minimum or maximum activity of every row over the explicit bounds, entries of fixed columns
// included. hi + lo is the compensated sum of the finite terms a * bound. A term that
// is_infinite_term is counted in inf_count instead; inf_slot is the sum of the CSR slots of the
// counted terms, so it is the slot of the single infinite entry when inf_count is 1.
// updates counts the finite terms added to or removed from the sum since it was last summed
// from scratch, the terms of that summation included, and max_term is the largest |a * bound|
// among them; they give the error bound activity_error(updates, max_term).
// The pair is not normalised: after a cancellation |lo| can be far above the last bit of hi (hi
// can be 0 while lo holds the value), so hi alone says nothing about the sum.
struct activity_t {
  std::vector<double> hi;
  std::vector<double> lo;
  std::vector<int32_t> inf_count;
  std::vector<int64_t> inf_slot;
  std::vector<double> max_term;
  std::vector<uint32_t> updates;

  void resize(int32_t nrows);
};

// One side of one row summed from scratch.
struct side_sum_t {
  dd_t sum;
  int32_t inf_count = 0;
  int64_t inf_slot  = 0;
  double max_term   = 0.;
  uint32_t terms    = 0;
};

// Lists of changed rows or columns, one list per consumer. A list holds an index at most once,
// in the order of the first event since the consumer last took the list. An index can be on a
// list after its row was deleted or its column fixed or removed; consumers test that.
struct frontier_set_t {
  std::vector<uint8_t> pending;  // bit l set while the index is on list l
  std::vector<std::vector<int32_t>> list;

  // Every index, ascending, on the lists of the mask `lists`.
  void init(int32_t n_lists, int32_t n_indices, uint8_t lists)
  {
    pending.assign(n_indices, lists);
    list.assign(n_lists, std::vector<int32_t>());
    for (int32_t l = 0; l < n_lists; ++l) {
      if (!((lists >> l) & 1)) { continue; }
      list[l].resize(n_indices);
      for (int32_t i = 0; i < n_indices; ++i) {
        list[l][i] = i;
      }
    }
  }

  // O(1).
  void push(uint8_t lists, int32_t index)
  {
    uint8_t add = lists & ~pending[index];
    if (add == 0) { return; }
    pending[index] |= add;
    for (int32_t l = 0; add != 0; ++l, add >>= 1) {
      if (add & 1) { list[l].push_back(index); }
    }
  }

  // Moves list l into out, replacing its contents, and starts an empty list. O(list length).
  // A consumer that stops early pushes the indices it did not process back.
  void take(int32_t l, std::vector<int32_t>& out)
  {
    out.clear();
    out.swap(list[l]);
    uint8_t keep = ~(1 << l);
    for (int32_t i : out) {
      pending[i] &= keep;
    }
  }

  bool empty(int32_t l) const { return list[l].empty(); }
};

constexpr uint8_t list_bit(int32_t list) { return 1 << list; }

// Consumers of changed rows.
namespace row_list {
constexpr int32_t row_rules     = 0;
constexpr int32_t propagation   = 1;
constexpr int32_t tightening    = 2;
constexpr int32_t doubleton     = 3;
constexpr int32_t parallel_rows = 4;
constexpr int32_t compaction    = 5;  // rows with more tombstones than live entries
// Rows with a column that became a singleton column or stopped being one (row_event::singleton).
// Empty at construction.
constexpr int32_t singletons = 6;
constexpr int32_t count      = 7;
}  // namespace row_list

// Consumers of changed columns.
namespace col_list {
constexpr int32_t dual_fixing = 0;
constexpr int32_t compaction  = 1;  // columns with more tombstones than live entries
constexpr int32_t count       = 2;
}  // namespace col_list

// Which lists an event feeds.
namespace row_event {
// A bound of a column of the row changed, and with it one activity of the row.
constexpr uint8_t activity =
  (1 << row_list::row_rules) | (1 << row_list::propagation) | (1 << row_list::tightening);
// The same when the side of the row that bound propagation reads that activity with is infinite
// (the upper side for the minimum activity, the lower side for the maximum activity): the
// bounds that the row implies are those of its last visit.
constexpr uint8_t activity_unread = (1 << row_list::row_rules) | (1 << row_list::tightening);
// A side of the row changed.
constexpr uint8_t side = activity | (1 << row_list::doubleton);
// An entry of the row was changed, moved or deleted, the row was scaled, or one of its columns
// was fixed.
constexpr uint8_t entries = side | (1 << row_list::parallel_rows);
constexpr uint8_t sparse  = 1 << row_list::compaction;
// A singleton column is a live column with one live entry. The row of that entry gets this event
// when the column becomes one (delete_row, delete_entry or move_entry took its other entries)
// and when it stops being one (move_entry gives it a second entry). The row itself does not
// change: its entries, sides and version stay. No event when the column is fixed or when its
// only entry leaves, which are events of the row anyway.
constexpr uint8_t singleton = (1 << row_list::parallel_rows) | (1 << row_list::singletons);
}  // namespace row_event

namespace col_event {
// A lock count or the cost of the column changed.
constexpr uint8_t locks  = 1 << col_list::dual_fixing;
constexpr uint8_t sparse = 1 << col_list::compaction;
}  // namespace col_event

// The presolve state shared by the host engine and the GPU engine. Rows and columns keep the
// indices of the input model for the whole presolve; extract_model renumbers at the end.
// Rules read the arrays and change the state only through the member functions, which keep
// stores, twins, sizes, activities, counters, locks, versions and lists consistent. An operation
// that leaves a live row or column with more tombstones than live entries puts it on its
// compaction list (row_event::sparse, col_event::sparse).
struct state_t {
  // O(nonzeros). The model must be well_formed. Every row is on the five rule lists and every
  // column on the dual-fixing list. Bounds and sides may cross and columns with lower == upper
  // are live: presolve() deals with both before the first rule runs, and from then on
  // lower <= upper holds for every live row and column and lower < upper for every live column.
  state_t(const model_t& model, const tolerances_t& tolerances);

  int32_t nrows;
  int32_t ncols;
  tolerances_t tol;

  line_store_t csr;
  line_store_t csc;

  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::vector<uint8_t> row_alive;
  // Live entries of the row in live columns. csr.size also counts entries of fixed columns.
  std::vector<int32_t> row_free_size;
  // Incremented when an entry or a side of the row changes; an implied bound recorded from the
  // row is void afterwards.
  // Not incremented when a bound of one of its columns changes or the column is fixed, nor by
  // delete_row (a deleted source row voids the record) or rebuild (a fixed term moves into the
  // sides, the constraint stays the same).
  std::vector<uint32_t> row_version;
  activity_t min_act;
  activity_t max_act;

  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<double> cost;
  std::vector<col_type_t> col_type;
  std::vector<col_status_t> col_status;
  // Live rows that stop the column from moving down / up: an entry a > 0 in a row with a finite
  // upper side locks up and with a finite lower side locks down; a < 0 the other way round.
  std::vector<int32_t> down_locks;
  std::vector<int32_t> up_locks;

  // Bounds of continuous columns implied by one row and the explicit bounds of its other
  // columns, kept apart from the explicit bounds. -inf / +inf and row -1 without a record.
  std::vector<double> implied_lower;
  std::vector<double> implied_upper;
  std::vector<int32_t> implied_lower_row;
  std::vector<int32_t> implied_upper_row;
  std::vector<uint32_t> implied_lower_version;
  std::vector<uint32_t> implied_upper_version;

  // Model offset plus cost * value of the fixed columns plus the offsets of substitutions.
  dd_t objective_offset;

  int32_t live_rows;
  int32_t live_cols;  // status live
  int64_t live_nnz;   // live entries, entries of fixed columns included
  int64_t fixed_nnz;  // live entries of fixed columns
  // One per bound value that moved (fix_col: each of the two that moved), per side value that
  // changed (shift_row_sides and scale_row: each finite side) and per entry changed, deleted or
  // moved (scale_row: each live entry).
  int64_t bound_changes       = 0;
  int64_t side_changes        = 0;
  int64_t coefficient_changes = 0;
  // Slots of the stores that the mutating operations and find_entry have visited since
  // construction, the summations they trigger included. It is no part of the model: tests bound
  // the cost of an operation with it.
  mutable int64_t slot_visits = 0;

  frontier_set_t row_frontier;  // lists row_list::*
  frontier_set_t col_frontier;  // lists col_list::*

  // Lanes of fold_row_activity, 2 * max_activity_lanes.
  std::vector<dd_t> lane_scratch;

  bool is_integer(int32_t col) const { return col_type[col] == col_type_t::integer; }
  bool is_live(int32_t col) const { return col_status[col] == col_status_t::live; }
  // Nonzeros of the model that extract_model would return.
  int64_t free_nnz() const { return live_nnz - fixed_nnz; }
  // Finite part of an activity; it is the activity when the side's inf_count is 0.
  double min_activity(int32_t row) const { return min_act.hi[row] + min_act.lo[row]; }
  double max_activity(int32_t row) const { return max_act.hi[row] + max_act.lo[row]; }
  double min_activity_error(int32_t row) const
  {
    return activity_error(min_act.updates[row], min_act.max_term[row]);
  }
  double max_activity_error(int32_t row) const
  {
    return activity_error(max_act.updates[row], max_act.max_term[row]);
  }
  // Share of the slots of the larger store that hold no live entry.
  double dead_fraction() const;

  // CSR slot of the live entry (row, col), or -1. Scans the shorter of the row and the column:
  // O(min(csr.used[row], csc.used[col])).
  int64_t find_entry(int32_t row, int32_t col) const;

  // True when the column has a recorded implied bound whose source row is alive and unchanged.
  bool implied_lower_valid(int32_t col) const;
  bool implied_upper_valid(int32_t col) const;

  // ---- Bounds and columns ----

  // New explicit bound of a live column; not NaN, lower <= upper must hold afterwards and the
  // value must differ from the current one. In each row of the column the bound enters one side
  // (min_activity_bound / max_activity_bound), and that side is updated as two terms, never as
  // a * (new - old): the old term leaves (dd_sub_product, or inf_count and inf_slot when it
  // is_infinite_term) and the new one enters (dd_add_product, or inf_count and inf_slot).
  // row_event::activity for each of those rows, row_event::activity_unread where it applies.
  // O(csc.used[col]).
  // A lower bound of +inf and an upper bound of -inf are not accepted (the same for the sides of
  // a row below): the caller reports infeasibility instead.
  void set_col_lower(int32_t col, double lower);
  void set_col_upper(int32_t col, double upper);

  // Fixes a live column at a finite value inside its bounds: both bounds become the value, the
  // status fixed, cost * value goes to the offset, its rows lose one free entry
  // (row_event::entries). Its entries stay. O(csc.used[col]).
  void fix_col(int32_t col, double value);

  // Marks a live column without live entries and with zero cost as removed. O(1).
  void remove_col(int32_t col);

  // O(1); col_event::locks.
  void set_cost(int32_t col, double value);

  // objective_offset += a * b, compensated. O(1).
  void add_objective_offset(double a, double b);

  // Records an implied bound of a continuous column with its source row and the row's version.
  // O(1). The explicit bound and the activities do not change.
  void record_implied_lower(int32_t col, double lower, int32_t row);
  void record_implied_upper(int32_t col, double upper, int32_t row);

  // ---- Rows ----

  // Deletes a live row: tombstones its entries in both stores and releases its locks
  // (col_event::locks for its live columns). O(csr.used[row]): no column is walked.
  // Added with row_event::singleton: but for a live column that is left with one live entry,
  // whose used slots are read up to that entry to find its row. The same holds for delete_entry
  // and for both columns of move_entry.
  void delete_row(int32_t row);

  // New side of a live row; lower <= upper must hold afterwards and the value must differ from
  // the current one (row_event::side). O(1) when the side stays finite or stays infinite,
  // otherwise O(csr.used[row]) for the locks of its columns (col_event::locks).
  void set_row_lower(int32_t row, double lower);
  void set_row_upper(int32_t row, double upper);

  // Subtracts a * b from the finite sides of a live row, both at once, as a substitution or a
  // removed constant term requires (row_event::side). O(1).
  void shift_row_sides(int32_t row, double a, double b);

  // The terms of the fixed columns leave their rows for the sides, as in rebuild() but without
  // a pass over the stores: every live entry of a fixed column is tombstoned in both stores; the
  // finite sides of each of those rows lose the sum of the row's terms (one compensated sum in
  // storage order, the arithmetic of rebuild()), the row is compacted and its activities are
  // recomputed. Afterwards fixed_nnz is 0 and a fixed column has no entry and no lock; it stays
  // fixed until a rebuild removes it. Versions and implied bounds are kept: the constraint is
  // the same. The slots of the compacted rows change.
  // A family can treat a row with such a term otherwise than the same row without it: a side
  // stored with the term in it has another size and another rounding, a term above the
  // infinite-term limit makes an activity infinite, and row tightening rounds the sides of an
  // integer row only by more than the tolerance while the row holds a term. Such a row gets
  // row_event::entries, here and in rebuild(). A row whose terms leave it plainly does not: the
  // sum of its terms is one double, every new side is the exact difference and neither above
  // 1e5 nor within the feasibility tolerance of an integer without being one, and its
  // activities have held no term above 1e5 since they were summed (max_term), so that their
  // error bounds decide nothing with or without the terms. The engine calls this before its
  // last rounds: only then do the rules see the rows that the reduced model will have.
  // O(columns + used slots of the rows that hold such a term).
  void absorb_fixed_terms();

  // True when every row can take the terms of its fixed columns into its sides, as
  // absorb_fixed_terms and rebuild move them. A finite side can when one of these holds:
  // - the side less the sum of the terms is exact;
  // - side and terms are at most tol.feasibility / tol.side_error in size, where a double holds
  //   the difference far within the tolerance;
  // - the new side is out of reach of the free part of the row by more than the tolerance plus
  //   what the difference rounds away: above its maximum or below its minimum activity over the
  //   bounds. The side is then never met or never satisfiable, whatever it lost (12 next to a
  //   term of 2e21).
  // Any other side is another constraint afterwards, by more than the tolerance (3 * 2^60 less 1
  // is 3 * 2^60), and a family that reads the stored side as it is reports a feasible row
  // infeasible; with the terms in the row its activities hold them exactly.
  // O(rows + used slots of the rows that hold such a term).
  bool fixed_terms_absorbable() const;

  // Multiplies the entries and the finite sides of a live row by factor > 0 and recomputes its
  // activities (row_event::entries). With integral set, each product a * factor is replaced by
  // the nearest integer; the caller has checked that every product is one within its tolerance.
  // The sides are multiplied, not rounded. O(csr.used[row]).
  // Only the products of live columns are rounded: the entry of a fixed column is a constant of
  // the row and gets the plain product.
  void scale_row(int32_t row, double factor, bool integral);

  // ---- Entries, addressed by CSR slot ----
  // The three operations below take a live entry of a live column: the term of a fixed column
  // is a constant of its row and leaves only in rebuild.

  // New value, finite and not zero: the old term is removed from both activities of the row
  // and the new one added (two terms, as for a bound), the locks follow a sign change
  // (row_event::entries, col_event::locks). O(1).
  void set_coefficient(int64_t row_slot, double value);

  // Tombstones the entry in both stores and removes its term and its locks
  // (row_event::entries, col_event::locks). O(1).
  void delete_entry(int64_t row_slot);

  // Turns the entry (row, j) into the entry (row, col) with the given value, finite and not
  // zero, in the same CSR slot. The row must have no live entry in col, and col must be live.
  // The CSC entry of j is tombstoned and one is appended to col; terms and locks follow
  // (row_event::entries, col_event::locks for both columns). Amortised O(1): a column without a
  // free slot is first moved to the end of the store with twice the capacity, at the cost of
  // its live entries, which changes the CSC slots of col only. The arrays of csc can be
  // reallocated: no pointer or reference into them survives the call.
  // The moved column gets at least 4 slots; its old slots stay as tombstones that no line owns
  // until the next rebuild.
  void move_entry(int64_t row_slot, int32_t col, double value);

  // ---- Activities ----

  // Both sides of a row summed from scratch in the order defined in numerics.hpp. This function
  // is the only place where an activity is summed. O(csr.used[row]).
  void fold_row_activity(int32_t row, side_sum_t& min_side, side_sum_t& max_side);

  // Replaces the stored activities of the row by fold_row_activity. The other operations call
  // it when the error bound of a side exceeds tol.activity_error_share * tol.feasibility and
  // that side has at least 2 * csr.size[row] updates. The second condition makes the updates
  // since the last summation pay for the next one: without it a long row whose error bound is
  // above the share right after a summation would be summed again at every event. The error
  // bound of a side can therefore be above the share, and every test on an activity allows for
  // it (min_activity_error / max_activity_error).
  void recompute_row_activity(int32_t row);

  // ---- Storage ----

  // Compacts, in place and keeping the order of the live entries, every line on the compaction
  // lists, so that a scan of a line costs at most twice its live entries. Recomputes the
  // activities of the compacted rows. Slots of the compacted lines change. O(used slots of
  // those lines).
  void compact_sparse_lines();

  // Compacts both stores: drops tombstones, deleted rows and the entries of fixed columns, whose
  // terms are subtracted from the finite sides of their rows, per row as one compensated sum
  // in storage order (the columns become removed).
  // Leaves used == size == capacity for every line and row_free_size == csr.size. A row that
  // loses a slot gets its activities recomputed; for the other rows inf_slot follows the shift
  // of the row. The live entries of a line keep their order. All slots change; lists, versions
  // and implied bounds are kept, and a row that loses the term of a fixed column gets
  // row_event::entries unless the term leaves it plainly, as in absorb_fixed_terms and for its
  // reason. O(slots + rows + columns).
  // The two compaction lists are emptied: the rebuild has served them.
  // A build with assertions ends the rebuild with check_consistency.
  void rebuild();

  // Recomputes sizes, twins, locks, counters and activities from the stores and asserts that the
  // state agrees, activities within their error bound. O(slots). For tests and assertion builds.
  // It runs and fails in every build, GPRS_ASSERT or not: a caller that wants it in assertion
  // builds only writes gprs_assert((st.check_consistency(), true)).
  void check_consistency();

  // The check behind check_consistency: an empty string, or the first broken invariant with the
  // row, column or slot where it broke. check_consistency fails with this text, in every build.
  // A stored activity must agree with the fold within 9 * (n + 1)^3 * u^2 * max_term for each of
  // the two accumulators (n its updates or terms), which holds for every sequence of updates.
  // activity_error(updates, max_term) is not the tolerance: valid states exceed it.
  std::string consistency_error();
};

// The live rows and live columns of a rebuilt state as a well_formed model, in index order,
// the entries of each row in increasing column order whatever their order in the state (filled
// column by column from the CSC), objective_offset = dd_value(st.objective_offset), with
// row_origin / col_origin giving the state index of each row and column of the model.
// O(slots + rows + columns).
model_t extract_model(const state_t& st,
                      std::vector<int32_t>& row_origin,
                      std::vector<int32_t>& col_origin);

}  // namespace gprs
