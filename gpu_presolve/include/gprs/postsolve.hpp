/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <gprs/state.hpp>

#include <cstdint>
#include <vector>

namespace gprs {

// A record exists where a column disappears or where a later dual postsolve needs the step.
// Deleted rows that are empty, redundant, singleton or forcing leave no record in Stage 1.
// A reader of the stack names every kind it knows and takes any other as a broken stack.
enum class record_kind_t : uint8_t { fixed_col, alias, parallel_row, row_addition, col_from_rows };

// x[col] = value.
struct fixed_col_record_t {
  int32_t col;
  double value;
};

// The rule that an alias comes from.
// doubleton: a doubleton equation, which holds at every feasible point.
// probing: simple probing on an equation; the alias holds at every feasible point.
// lock: a column with a single lock against its cost, aliased to a binary of its lock row; the
// alias holds at an optimal point only, so no result of primal-only mode has one.
enum class alias_origin_t : uint8_t { doubleton, probing, lock };

// x[col] = scale * x[rep] + offset. rep is live when the record is pushed. `row` is the row that
// leaves the model with the alias, -1 when none does: the equation of origin doubleton; the
// equation of origin probing when its batch deletes it, else -1; always -1 for origin lock,
// whose lock row stays.
struct alias_record_t {
  int32_t col;
  int32_t rep;
  int32_t row;
  double scale;
  double offset;
  alias_origin_t origin;
};

// removed_row = ratio * kept_row, entry by entry; the sides of the kept row were intersected.
// The primal postsolve ignores it.
struct parallel_row_record_t {
  int32_t removed_row;
  int32_t kept_row;
  double ratio;
};

// row became row - factor * equation_row on the free entries: those that the two rows shared
// left the row, those that only the equation had entered it, and its finite sides moved by
// factor times the right-hand side of the equation less its fixed terms. Both rows stay. The
// primal postsolve ignores it.
struct row_addition_record_t {
  int32_t row;
  int32_t equation_row;
  double factor;
};

// Entries of saved rows, one row after the other.
struct row_pool_t {
  std::vector<int32_t> col;
  std::vector<double> value;
};

// The copy of a row at one moment: its sides less the terms of its fixed columns, and its
// entries [begin, end) of a pool. It names live columns only.
struct saved_row_t {
  int32_t row;
  double lower;
  double upper;
  int64_t begin;
  int64_t end;
};

// What a col_from_rows record says of its columns.
// defined: one column and one saved row, an equation lower == upper: the value of the column is
// what the row gives it at the values of the saved entries. The pivot row of a free singleton
// column, which left the model; with `dual` it was an inequality, tight at that side by the
// cost of the column.
// interval: one saved row with two finite sides (one at least with
// options_t::zero_cost_one_sided) and one or more continuous columns with zero
// cost, in increasing index: the row stays in the model without them, its sides widened by the
// ranges of their terms, and they take values inside their bounds with which the saved row
// holds.
// forcing: one column with zero cost and one or more saved rows, all of which left the model.
// Each has one finite side and bounds the column from the same end, at which the column has no
// bound: every row with a positive coefficient has the upper side and every row with a
// negative one the lower side, and the lower bound of the column is -inf (the column was free
// to go down); or the mirror.
enum class col_from_rows_kind_t : uint8_t { defined, interval, forcing };

// A column that a col_from_rows record removes, with its explicit bounds at the push.
// prefer is the value an interval column starts from: its lower bound if finite, else its upper
// bound if finite, else 0; the same value for a forcing column, which does not read it. The
// engine writes another finite value inside the bounds of an interval column with
// options_t::interval_start; the undo reads the value and not how it was chosen.
struct record_col_t {
  int32_t col;
  double lower;
  double upper;
  double prefer;
};

// Columns [cols_begin, cols_end) of postsolve_t::record_cols, rows [rows_begin, rows_end) of
// postsolve_t::saved_rows, and their coefficients from `coefficients` on in
// postsolve_t::record_coefficients: one per column (defined, interval), in the record's one
// row, or one per row (forcing), of the record's one column. A saved row holds the free entries
// of its row other than those of the record's columns.
// Undo, with rest the compensated sum of the saved entries of a row at the restored values:
// defined: x = (side - rest) / coefficient, for an integer column rounded to the nearest
//   integer; never moved into the bounds.
// interval: every column starts at prefer. While the row with them is below its lower side, the
//   columns are moved one after the other, in the order of the record, towards the bound that
//   raises their term: to that bound while the row misses at least what the column can give,
//   and the first column that can give more to (side - rest - the terms of the other columns of
//   the record) / coefficient, kept between prefer and the bound. Above its upper side the
//   mirror. The value is the quotient and not prefer plus a step, which loses the digits that
//   prefer is larger by. The columns stay inside their bounds, and the row holds when the rest
//   lies inside the widened sides. With postsolve_t::interval_descending the columns are moved
//   in the reverse order of the record. An infinite side of the saved row asks for nothing.
// forcing: the column free to go down gets min(upper, the smallest of (side - rest) /
//   coefficient over its rows), rounded down for an integer column; the column free to go up
//   max(lower, the largest of them), rounded up. A value of a row that lies within the rounding
//   of its computation of an integer (postsolve_t::forcing_rounding_units rounding units of
//   the side and the terms of the rest, over the coefficient) is that integer.
struct col_from_rows_record_t {
  col_from_rows_kind_t kind;
  bool dual;
  int32_t cols_begin;
  int32_t cols_end;
  int32_t rows_begin;
  int32_t rows_end;
  int64_t coefficients;
};

// The record stack. Each kind has its own array; order holds the kinds in application order, so
// the k-th occurrence of a kind in order is element k of that kind's array. Indices are those of
// the original model.
struct postsolve_t {
  int32_t nrows = 0;
  int32_t ncols = 0;
  // Nonzeros of the original model.
  int64_t nnz = 0;
  std::vector<uint8_t> col_type;
  // Reduced index -> original index, set by the engine together with the reduced model.
  std::vector<int32_t> row_origin;
  std::vector<int32_t> col_origin;

  std::vector<record_kind_t> order;
  std::vector<fixed_col_record_t> fixed_cols;
  std::vector<alias_record_t> aliases;
  std::vector<parallel_row_record_t> parallel_rows;
  std::vector<row_addition_record_t> row_additions;
  std::vector<col_from_rows_record_t> col_from_rows;
  // The columns, saved rows and coefficients of the col_from_rows records, record after record.
  // Later record kinds save rows into the same arrays: a reader finds the rows of a record
  // through the record and never by walking saved_rows.
  std::vector<record_col_t> record_cols;
  std::vector<saved_row_t> saved_rows;
  std::vector<double> record_coefficients;
  row_pool_t saved_entries;
  // 1 once a fixed_col, alias or col_from_rows record has removed the column.
  std::vector<uint8_t> col_removed;

  // How the undo chooses among the values that the saved rows of a col_from_rows record leave
  // its columns. The presolve writes them from options_t::interval_order and
  // options_t::forcing_rounding_units; a point that is restored with other values of them is as
  // feasible. They are no part of postsolve_hash.
  // The columns of an interval record are moved in the reverse order of the record.
  bool interval_descending = false;
  // The value that a row leaves an integer forcing column is rounded into the row, and one
  // that lies within this many rounding units of the terms it was computed from of an integer
  // counts as that integer: 9 - 3 * (7 / 3) is 1.9999999999999996 and stands for 2. With 0 the
  // value hangs on the last bit of a sum, which another summation order gives otherwise.
  double forcing_rounding_units = 8.;

  // Empties the stack, takes sizes and column types from the original model and sets the two
  // choices of the undo to their defaults.
  void init(const model_t& original);

  // The type of a column at a position of the stack (before record `position` of order; at
  // order.size() in the reduced model) and now, for a push. Every undo, push contract and check
  // that depends on a type reads it here. No record changes a type so far, so it is the type of
  // the original model.
  bool integer_at(int32_t col, size_t position) const
  {
    (void)position;
    return col_type[col] != 0;
  }
  bool integer_now(int32_t col) const { return integer_at(col, order.size()); }

  // O(1) each. A column that a fixed_col, alias or col_from_rows record removes must not appear
  // in a later record.
  // A fixed value is finite, and integral for an integer column. An alias has a finite scale
  // other than zero and a finite offset; an integer column has an integer rep and an integral
  // scale and offset; its row is as alias_record_t says for its origin. A ratio and a factor are
  // finite and not zero.
  void push_fixed_col(int32_t col, double value);
  void push_alias(int32_t col,
                  int32_t rep,
                  int32_t row,
                  double scale,
                  double offset,
                  alias_origin_t origin = alias_origin_t::doubleton);
  void push_parallel_row(int32_t removed_row, int32_t kept_row, double ratio);
  void push_row_addition(int32_t row, int32_t equation_row, double factor);

  // The three kinds of col_from_rows. Each removes its columns. The saved rows are the last
  // ones written to saved_entries (save_row, src/saved_row.hpp), in the order given, and name
  // columns that no record has removed and none of the record's own. Coefficients are finite
  // and not zero, bounds satisfy lower <= upper and prefer is as record_col_t says (for an
  // interval column a finite value inside its bounds).
  // O(entries of the saved rows) in a build with assertions, else O(columns and rows of the
  // record).
  // defined: the saved row is an equation with a finite side; an integer column has integer
  // columns in its saved row.
  void push_defined_col(
    int32_t col, double coefficient, double lower, double upper, const saved_row_t& row, bool dual);
  // interval: the columns are continuous and in increasing index, cols[k] with coefficients[k];
  // the saved row has lower <= upper and a finite side.
  void push_interval_cols(const std::vector<record_col_t>& cols,
                          const std::vector<double>& coefficients,
                          const saved_row_t& row);
  // forcing: rows[k] holds the column with coefficients[k]; rows and bounds are as
  // col_from_rows_kind_t says.
  void push_forcing_col(const record_col_t& col,
                        const std::vector<double>& coefficients,
                        const std::vector<saved_row_t>& rows);

  // Point of the reduced model -> point of the original model: integer columns of the reduced
  // point are rounded, then the records are undone in reverse order. Does not change the
  // object, so several threads can call it at once. O(columns + records + saved entries).
  void postsolve(const std::vector<double>& x_reduced, std::vector<double>& x_original) const;

  // Point of the original model -> point of the reduced model: the values of the columns that
  // are left. A feasible point gives a feasible point of the reduced model whose objective value
  // there is not above that of the point, and equal to it when no reduction rests on a dual
  // argument (result_t::dual_findings == 0, which primal-only mode guarantees). A col_from_rows
  // record only drops its columns: the reduced model holds what the other columns must meet for
  // values of the dropped ones to exist, and with `dual` its objective is that of the point with
  // the dropped column moved onto its tight row.
  // O(reduced columns).
  void crush(const std::vector<double>& x_original, std::vector<double>& x_reduced) const;
};

// Largest violations of a point, and its objective value including the offset.
struct point_check_t {
  double row_violation         = 0.;
  double bound_violation       = 0.;
  double integrality_violation = 0.;
  double objective             = 0.;
};

// O(nonzeros). Row activities are compensated sums.
point_check_t check_point(const model_t& model, const std::vector<double>& x);

// Rows and bounds within tol.feasibility, integer columns within tol.integrality.
bool point_feasible(const point_check_t& check, const tolerances_t& tol);

// a * x[col_a] + b * x[col_b] = rhs, read from row `row` with the terms of fixed columns moved to
// the right-hand side.
struct alias_edge_t {
  int32_t row;
  int32_t col_a;
  int32_t col_b;
  double a;
  double b;
  double rhs;
};

// x[col] = scale * x[root] + offset; `row` is the tree edge of col towards the root, an identity
// once col and its parent are substituted.
struct alias_map_t {
  int32_t col;
  int32_t root;
  int32_t row;
  double scale;
  double offset;
};

// The doubleton equations of one batch as a spanning forest over the columns. Per-column arrays
// live for the whole presolve; a batch touches only the columns of its edges.
struct alias_forest_t {
  std::vector<int32_t> link;        // union-find parent of a column of the batch, -1 otherwise
  std::vector<int32_t> touched;     // columns of the batch in order of first appearance
  std::vector<alias_edge_t> edges;  // tree edges in the order they were added
  std::vector<int32_t> position;    // index in touched of a column of the batch, -1 otherwise
  std::vector<int32_t> set_size;    // by position: columns of the set of a union-find root

  // O(columns), once.
  void init(int32_t ncols);

  // Adds the edge when its columns are not yet connected and returns true. A non-tree edge is
  // not stored and its row stays in the model. Amortised near O(1).
  bool add_edge(const alias_edge_t& edge);

  // An edge read as child = scale * parent + offset is admissible when the child is continuous,
  // or child and parent are integer and scale and offset are integral within tol.zero (they are
  // then rounded to integers). Between two integer columns with a ratio other than 1 or -1 at
  // most one direction is admissible, so the root decides which equations can be used.
  // Roots every tree at a column of the highest type (integer above continuous); among those
  // at the one that makes the most edges of the tree admissible, then the one with the most
  // live entries, then the smallest index. Walks each tree breadth-first from its root,
  // neighbours in the order their edges were added. An inadmissible edge is cut: its row stays
  // in the model and the child becomes the root of its subtree, which is walked after the tree.
  // Returns in maps every column that is not a root with its map composed to its root, the maps
  // of one root contiguous and parents before children. O(edges of the batch).
  // Added with the implementation. The columns of the edges are live. An edge between integers
  // must be integral in the units of its row as well, since a large coefficient turns a
  // deviation below tol.zero into a violated row: with the rounded scale and offset,
  // |child coefficient * scale + parent coefficient| <= tol.zero and
  // |rhs - child coefficient * offset| <= tol.feasibility; and what the rounding of the scale
  // moves the row by over the bounds of the parent, |child coefficient| * |scale as read -
  // scale| * max(|lower|, |upper|) of the parent, is at most tol.integral_drift_share *
  // tol.feasibility, and together with |rhs - child coefficient * offset| at most
  // tol.feasibility (integral_term and drift_within, src/integral.hpp): a scale off by 5e-10
  // next to a parent that reaches 2e6 leaves the row off by 1e-3. An edge is inadmissible as well
  // when its scale, rounded for an integer child, is zero or not finite or its offset is not
  // finite, and it is cut when the map composed to the root is (the root choice counts edges, not
  // composed maps). The trees are walked in the order of their first column in touched, and the
  // cut subtrees of a tree in the order they were cut.
  void freeze(const state_t& st, std::vector<alias_map_t>& maps);

  // Ends the batch. O(columns of the batch).
  void clear();
};

}  // namespace gprs
