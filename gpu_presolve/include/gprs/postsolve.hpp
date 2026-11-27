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
enum class record_kind_t : uint8_t { fixed_col, alias, parallel_row, row_addition };

// x[col] = value.
struct fixed_col_record_t {
  int32_t col;
  double value;
};

// x[col] = scale * x[rep] + offset, from the doubleton equation `row`. rep is live when the
// record is pushed.
struct alias_record_t {
  int32_t col;
  int32_t rep;
  int32_t row;
  double scale;
  double offset;
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

// The record stack. Each kind has its own array; order holds the kinds in application order, so
// the k-th occurrence of a kind in order is element k of that kind's array. Indices are those of
// the original model.
struct postsolve_t {
  int32_t nrows = 0;
  int32_t ncols = 0;
  std::vector<uint8_t> col_type;
  // Reduced index -> original index, set by the engine together with the reduced model.
  std::vector<int32_t> row_origin;
  std::vector<int32_t> col_origin;

  std::vector<record_kind_t> order;
  std::vector<fixed_col_record_t> fixed_cols;
  std::vector<alias_record_t> aliases;
  std::vector<parallel_row_record_t> parallel_rows;
  std::vector<row_addition_record_t> row_additions;
  // 1 once a fixed_col or alias record has removed the column.
  std::vector<uint8_t> col_removed;

  // Empties the stack and takes sizes and column types from the original model.
  void init(const model_t& original);

  // O(1) each. A column that a fixed_col or alias record removes must not appear in a later
  // record.
  // A fixed value is finite, and integral for an integer column. An alias has a finite scale
  // other than zero and a finite offset; an integer column has an integer rep and an integral
  // scale and offset. A ratio and a factor are finite and not zero.
  void push_fixed_col(int32_t col, double value);
  void push_alias(int32_t col, int32_t rep, int32_t row, double scale, double offset);
  void push_parallel_row(int32_t removed_row, int32_t kept_row, double ratio);
  void push_row_addition(int32_t row, int32_t equation_row, double factor);

  // Point of the reduced model -> point of the original model: integer columns of the reduced
  // point are rounded, then the records are undone in reverse order. Does not change the
  // object, so several threads can call it at once. O(columns + records).
  void postsolve(const std::vector<double>& x_reduced, std::vector<double>& x_original) const;

  // Point of the original model -> point of the reduced model: the values of the columns that
  // are left. In primal-only mode a feasible point gives a feasible point with the same
  // objective value. O(reduced columns).
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
  // |rhs - child coefficient * offset| <= tol.feasibility. An edge is inadmissible as well when
  // its scale, rounded for an integer child, is zero or not finite or its offset is not finite,
  // and it is cut when the map composed to the root is (the root choice counts edges, not
  // composed maps). The trees are walked in the order of their first column in touched, and the
  // cut subtrees of a tree in the order they were cut.
  void freeze(const state_t& st, std::vector<alias_map_t>& maps);

  // Ends the batch. O(columns of the batch).
  void clear();
};

}  // namespace gprs
