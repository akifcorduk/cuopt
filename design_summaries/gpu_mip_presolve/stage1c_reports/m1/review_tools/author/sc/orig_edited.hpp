/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The structure check: what one presolve result must satisfy whatever point is postsolved. It
// reads the postsolve stack, the origin maps and the reduced model, needs no solver and no
// solution file, and costs O(columns + records + nonzeros + saved entries) (a row whose coefficients were
// edited one by one costs its length times the number of its different coefficient ratios; a
// row that had equations subtracted costs the entries of its reduced row and of theirs times
// the points that its search visits, see added_rows).
// It sees what the crush check cannot: a reduction that relaxes.
//
//   stack      The stack and the maps keep the contract of postsolve.hpp: every record is of a
//              kind that this check reads; every column is left in the reduced model or removed
//              by exactly one record and named by no later record; a fixed value is finite, and
//              integral for an integer column; an alias has a finite scale other than zero and
//              a finite offset, an integer column an integer rep with an integral scale and
//              offset, and the row that its origin asks for (a row for a doubleton equation,
//              none for a lock, either for simple probing); a ratio is finite and not zero; a
//              row addition has a finite factor other than zero and names two different rows,
//              neither of which an earlier record took out of the model (the removed row of a
//              parallel record, the row of an alias that has one, the rows of a defined or a
//              forcing record). A col_from_rows record has the counts of its kind, finite
//              coefficients other than zero, bounds lower <= upper, saved rows that an earlier
//              record did not take out of the model and that name no removed column and none of
//              its own; a defined record an equation with a finite side; an interval record
//              continuous columns in increasing index and two finite sides; a forcing record
//              rows that each have one finite side and bound its column from one and the same
//              end, at which the column has no bound: a value of the column then exists
//              whatever the other columns are, which is all that such a record claims.
//              A result of primal-only mode has no dual finding (result_t::dual_findings), no
//              alias of origin lock, no defined record with the dual flag and no interval or
//              forcing record.
//   types      A column that is integer in the original model is integer in the reduced model.
//   The check variables are the reduced columns and, each with the bounds of its record as its
//   box, the columns of the col_from_rows records: their values at a postsolved point are not
//   functions of one reduced column. The anchors are the constraints over check variables that
//   hold at every postsolved point: the rows of the reduced model, and every row that a
//   col_from_rows record saved, with the record's own columns in it, because the undo of the
//   record makes it hold (a defined record by solving the row for its column, a forcing record
//   by its contract; an interval record where its snapshot holds, below). The anchor of a row
//   after a position of the stack is its first saved row after it, else its row of the reduced
//   model, else that of the row it was removed into as a parallel row after that.
//   bounds     With the records composed, every original column is a constant or
//              scale * (one check variable) + offset. Over the box of that variable it stays
//              within its original bounds.
//   rows       The activity of an original row at a postsolved point is a linear function g of
//              the check variables. A row without an anchor holds at the postsolve of every
//              point within the boxes: the range of g over the boxes lies within the sides of
//              the row.
//   kept_rows  A row with an anchor c with sides [l, u] holds at the postsolve of every point
//              within the boxes that meets the anchor: the maximum of g over
//              { boxes, l <= c * r <= u } is not above its upper side, and the minimum not below
//              its lower side. A row removed as a parallel row is tested in the same way
//              against the anchor of its kept row, or as a row without one. The pivot row of a
//              free singleton column and the rows of a forcing column are multiples of their
//              saved rows and pass with one multiplier.
//              The maximum is the smallest of the bounds  lambda * (u or l) + max over the
//              bounds of (g - lambda * c) * r,  which is reached at lambda = 0 or at a ratio
//              g[k] / c[k].
//   added_rows A row that had an equation subtracted (a row_addition record) is no multiple of
//              its anchor: it is that row plus multiples of the equations. Below, "reduced row"
//              stands for the anchor of a row and "bounds" for the boxes of the check
//              variables; without a col_from_rows record they are the same. The row holds at the
//              postsolve of every point within the bounds that meets its reduced row, if it has
//              one, and the reduced rows of those equations (proof_rows): the maximum of g
//              over { bounds, l <= c * r <= u, le <= e * r <= ue for each equation e } is not
//              above its upper side, and the minimum not below its lower side.
//              The maximum is bounded by weak duality: for any multipliers lambda and mu_e,
//                lambda * (u or l) + sum of mu_e * (ue or le)
//                  + max over the bounds of (g - lambda * c - sum of mu_e * e) * r,
//              each multiplier with the side of its sign. A pass is therefore a proof whatever
//              the multipliers are, and they are read from the data alone: the factor of a
//              record counts the rows as they were when it was pushed, before and after other
//              rules scaled them, and is not used. The bound is linear in the multipliers
//              between the hyperplanes where one of its terms changes its sign (a multiplier
//              is 0; the coefficient g[k] - lambda * c[k] - sum of mu_e * e[k] of a column is
//              0), so its smallest value, which is that maximum when a point meets the rows,
//              is taken where as many of them meet as there are multipliers. The search visits
//              all those points, and a row fails when the smallest bound is above its side.
//              A bound is a proof only as it is computed. Hyperplanes that are parallel, or
//              nearly so, meet in the rounding of their system or far out: the multipliers of
//              such a point are huge and cancel each other, a coefficient of the size of the
//              row's own vanishes in their sum, and the bound that is left can lie far below
//              the maximum. The search therefore takes no point from a system with a pivot
//              below singular_pivot, and a coefficient counts as zero only as a cancellation
//              of the row itself or as the rounding of the terms that make it
//              (combined_bound).
//              Not tested, and never failed or passed, is a row that does not pass while no
//              point within the bounds meets the reduced rows together: the bounds then have
//              no smallest one, which the same search finds as a negative bound of the
//              function zero (no_common_point).
//              The anchor of an equation is taken after the position of the record that
//              subtracted it, and the equations of a row are those of the records between the
//              position of the row and the position of its anchor: what is subtracted from the
//              row later is the business of the anchor. An equation without an anchor (deleted,
//              turned into a bound, substituted) gets no multiplier: its part of g is then
//              tested over the boxes like the rest. A row all of whose equations are without
//              one is tested as a row without an anchor or as a kept row.
//   columns    What the saved rows prove of the columns that left the model with them.
//              Box of a defined column: the range of (side - rest) / coefficient over the boxes
//              of the variables of its saved row lies inside the bounds of the record, which is
//              the engine's test "the row implies the bounds of the column" recomputed on the
//              reduced model, so the column is inside its bounds at the postsolve of every
//              point within the boxes. An integer column of a defined record has integer
//              columns in its saved row, and ratios and a side that are integral for it up to a
//              drift, over the boxes, within the tolerance in the units of the row: only then
//              does the rounded value leave the saved row true.
//              Snapshot of an interval record: the saved entries without the record's columns,
//              with the sides less the largest and the smallest sum of the terms of those
//              columns over their bounds (recomputed from the record), hold at the postsolve of
//              every point within the boxes that meets the anchor of that row after the record,
//              and the equations subtracted from the row between the record and that anchor.
//              Tested as a row is, against that anchor and those equations. Where it holds,
//              values of the record's columns exist inside their bounds with which the saved
//              row holds, and the undo finds them; so the saved row is an anchor only as far as
//              its snapshot passes, and a snapshot failure is read before any other.
//   objective  The objective of the original model at a postsolved point is the reduced
//              objective at the reduced point: with the records composed, the difference of the
//              two is a linear function D of the check variables. In the order of the stack,
//              the variable of every defined record is eliminated from D with its saved row,
//              which gives its value at every postsolved point; what is left is the zero
//              function. Every coefficient of D is zero within composed_zero of the sum of the
//              absolute values it is made of, and its constant within identity_tolerance of the
//              sum of the absolute values of its terms, an offset counted with the terms that
//              the aliases composed it of. This reads what the bounds and the rows do not: the
//              cost that a substitution or a free singleton column moves onto other columns,
//              the offset that they and a fixing leave, and the cost of a column of an interval
//              or a forcing record, which must be zero.
//   postsolve  postsolve_t::postsolve gives what the composed records give, with every column
//              of a col_from_rows record computed a second time from its anchors by the rule
//              of its kind (postsolve.hpp), at these points of the reduced model: every column
//              at its lower bound, at its upper bound and between them, and for up to
//              max_replay_points interval records a point at which the undo has to move the
//              record's columns. At each of them every saved row of a defined or a forcing
//              record holds at the postsolved point, that of an interval record where the rest
//              of the row lies inside the sides of its snapshot, and the columns of interval
//              and forcing records are inside their bounds.
//
// Why bounds, rows, kept_rows, added_rows and columns hold, in both modes. No rule relaxes a
// bound of a column that stays, and the bounds of a substituted column are transferred to its
// root. A row leaves the model because its sides hold over the bounds of its columns
// (redundant, empty), because its bound went to its only column (singleton), because all its
// columns were fixed (forcing), or because it defines a column that leaves with it: a
// doubleton equation, whose g is then a constant, and the pivot row of a free singleton column,
// which the undo solves for that column whatever the other columns are, so that the row holds
// by construction and what is left to prove is that the value is inside the bounds of the
// column (the box test). The rows of a forcing column leave because a value of the column
// exists that meets them all, whatever the other columns are. Bounds only tighten afterwards.
// Every edit of one row that stays gives a row that implies the row before the edit over the
// bounds, for real points as well: scaling and substitution keep it, a rounded or intersected
// side is tighter, and a strengthened coefficient a -> d with the side moved by
// (a - d) * bound differs from the old row by (a - d) * (bound - x) >= 0. Two edits are of
// another kind. The subtraction of an equation from a row that is parallel to it up to
// singleton columns (rules.hpp (e)) combines rows: the row before is the row after plus the
// factor times the equation, entry by entry and in the sides, so it holds wherever the row
// after and the equation hold; and an equation stays one, since sides only tighten. And the
// zero-cost singleton columns of a row (rules.hpp (h)) leave it with sides that are wider than
// before: the row after the edit does not imply the row before it, it is the projection of
// that row onto the other columns, and the row before holds because the undo then gives the
// removed columns values with which it does. That is why a saved row is an anchor and why the
// snapshot of the record is an obligation of its own. By induction over the records, from the
// end of the stack, every anchor holds at the postsolved point, and an original row is implied,
// over the boxes, by its anchor and the anchors of the equations it is made of. The proof that
// the induction gives has the multipliers of the subtractions on the equations, in the units
// of the rows as they are at their anchors. Where the bounds have a smallest one, the search
// finds it, and it is not above the bound of that proof. They have none only when no point
// within the boxes meets the anchors the row is made of, and the row is then counted as not
// tested unless the search proves it.
// Why objective holds: a fixing puts cost * value into the offset, a substitution puts
// cost * scale onto the cost of the root and cost * offset into the offset, a free singleton
// column puts its cost over its coefficient, times the row, onto the other columns of its pivot
// row and into the offset, the other column rules built so far remove columns without a cost,
// and no other rule touches a cost; a fixing by locks is a fixing here. What the check cannot
// see is a restriction: a column fixed or aliased where it must not be, or an inequality taken
// as tight at the wrong side, passes every part as long as it stays inside its bounds, and the
// crush check and the oracle test that.
// Like the crush check, this has to be reread when a rule is added that relaxes a bound,
// deletes a row for another reason or combines rows in another way (dominated columns,
// parallel columns, sparsification).
// The tolerance is wider than that of a tested point: a row can use the tolerance of its
// deletion and that of a rounded bound together. A row that is made of equations uses theirs
// as well: an equation holds within the tolerance of the rule that last moved its side (a
// rounded side, a side taken from a parallel row), and that reaches the row times the
// multiplier of the equation. A violation of such a row is therefore relative to the largest
// of: its bound, the terms of its own coefficients at the bounds that the bound uses, and the
// multiplier of each equation, in the units of the row per unit of the reduced equation; the
// largest multiplier of a tested row is reported (added_multiplier). The sides of the reduced
// rows and the terms of columns that only they hold are not the row's: the side of a row
// after a subtraction holds the right-hand side of the equation and the side of the equation
// gives it back, so they cancel in the bound whatever their size is. They count with the
// share that their rounding can reach (rounding_share), and a violation of 50 by a row with
// terms of 50 fails next to an equation with a side of 1e7.

#include "crush_check.hpp"

#include <gprs/presolve.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace gprs_check {

constexpr double structure_tolerance = 1e-5;
// A composed coefficient below this share of the sum of its parts is a cancellation.
constexpr double composed_zero = 1e-7;
// Two ratios g / c within this, relative, are one.
constexpr double same_ratio = 1e-9;
// postsolve() and the composed records agree within this, relative to the size of their terms.
constexpr double composed_tolerance = 1e-9;
// A kept row costs its length times its number of different ratios, a row that is made of
// equations the entries of its reduced rows times the points that its search visits. One above
// the first limit is not tested, and none once the rows together are above the second.
constexpr double max_kept_row_work = 5e7;
constexpr double max_kept_work     = 2e10;
// Equations a row can be made of and still be tested. The points of a search multiply with
// every reduced row that shares no column with the others, so the work limit ends it earlier.
constexpr size_t max_row_equations = 16;
// Hyperplanes meet in a point of a search when their system, with every column and then every
// row scaled to a largest entry of 1, has no pivot below this. A smaller one is the rounding of
// a zero, or gives multipliers that much larger than the coefficients they balance.
constexpr double singular_pivot = 1e-6;
// What the terms of several reduced rows leave of a coefficient is their rounding below this
// share of their sum.
constexpr double rounding_zero = 1e-12;
// The share of a term of a bound that its rounding can reach, in units of structure_tolerance:
// a side or a bound of the reduced model is stored with a relative error far below 1e-14.
constexpr double rounding_share = 1e-9;
// A difference of costs up to the smallest normal double is what a product loses when it
// underflows, whatever the sizes of its terms are.
constexpr double underflow_size = std::numeric_limits<double>::min();

// x[original column] = scale * x_reduced[root] + offset; root -1 for a constant.
struct column_map_t {
  int32_t root;
  double scale;
  double offset;
};

struct structure_report_t {
  bool ran = false;
  // Empty when the stack keeps its contract; the other parts need the composed records and are
  // left out when it does not.
  std::string stack_error;
  bool composed = false;

  int64_t lost_integers      = 0;
  int32_t first_lost_integer = -1;  // original index

  int64_t bound_failures  = 0;
  double bound_worst      = 0.;  // relative to max(1, |bound|)
  int32_t bound_worst_col = -1;

  // Rows that left the model. Violations are relative to max(1, largest term of the end that
  // is violated).
  int64_t rows_tested   = 0;
  int64_t row_failures  = 0;
  double row_worst      = 0.;
  int32_t row_worst_row = -1;

  // Rows tested against a reduced row: those in the reduced model and the parallel rows whose
  // kept row is. Not tested: a reduced row that no point within the bounds meets, and a row
  // above the work limits.
  int64_t kept_tested    = 0;
  int64_t kept_parallel  = 0;
  int64_t kept_untested  = 0;
  int64_t kept_failures  = 0;
  double kept_worst      = 0.;
  int32_t kept_worst_row = -1;

  // Rows that are made of equations as well (row_equations_t), whether these still have a
  // reduced row or not, and of them the rows tested against the reduced row of such an
  // equation: they count here and not above. Not tested: a row that does not pass while no
  // point within the bounds meets the reduced rows together, and a row above the work limits
  // or with more than max_row_equations equations.
  int64_t added_rows      = 0;
  int64_t added_tested    = 0;
  int64_t added_untested  = 0;
  int64_t added_failures  = 0;
  double added_worst      = 0.;
  int32_t added_worst_row = -1;
  // Largest multiplier of an equation in the bound that a tested row passed or failed with.
  double added_multiplier = 0.;

  // Coefficients of the objective difference that fail, and one more when its constant fails.
  // The largest coefficient is relative to the sum of its parts and belongs to the reduced
  // column of the original index objective_worst_col; the constant is relative to the sum of
  // its terms.
  int64_t objective_failures  = 0;
  double objective_worst      = 0.;
  int32_t objective_worst_col = -1;
  double objective_constant   = 0.;

  // Ends of the boxes of defined columns that were tested (a finite bound each), and those
  // that the saved row does not imply; the worst is relative to max(1, |bound|, largest term
  // of the row over the coefficient).
  int64_t box_tested    = 0;
  int64_t box_failures  = 0;
  double box_worst      = 0.;
  int32_t box_worst_col = -1;
  // Integer columns of defined records whose saved row is not integral for them.
  int64_t integral_failures  = 0;
  int32_t first_integral_col = -1;
  // Snapshots of interval records: tested, not tested (as a kept or an added row is not) and
  // failed, with the row of the worst one.
  int64_t snapshot_tested    = 0;
  int64_t snapshot_untested  = 0;
  int64_t snapshot_failures  = 0;
  double snapshot_worst      = 0.;
  int32_t snapshot_worst_row = -1;
  // Records by kind, for the reports.
  int64_t defined_records  = 0;
  int64_t interval_records = 0;
  int64_t forcing_records  = 0;

  int64_t postsolve_failures  = 0;
  double postsolve_worst      = 0.;  // relative to max(1, largest term)
  int32_t postsolve_worst_col = -1;
};

inline bool is_integral(double v) { return v == std::round(v); }

// Tests the contract of the stack, for a result of the given mode, and composes its records. On
// a broken contract `error` says which part, with the index of the record or column.
inline bool compose_records(const gprs::model_t& model,
                            const gprs::result_t& result,
                            bool primal_only,
                            std::vector<column_map_t>& map,
                            std::string& error)
{
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;
  auto fail                    = [&](const char* what, int64_t index) {
    error = std::string(what) + " " + std::to_string(index);
    return false;
  };
  if (ps.ncols != model.ncols || ps.nrows != model.nrows) { return fail("sizes of the stack", 0); }
  if (primal_only && result.dual_findings != 0) {
    return fail("dual findings in a primal-only result:", result.dual_findings);
  }
  if (ps.col_origin.size() != (size_t)reduced.ncols ||
      ps.row_origin.size() != (size_t)reduced.nrows) {
    return fail("sizes of the origin maps", 0);
  }
  const int32_t unset = -2;
  map.assign(model.ncols, column_map_t{unset, 0., 0.});
  for (int32_t k = 0; k < reduced.ncols; ++k) {
    int32_t col = ps.col_origin[k];
    if (col < 0 || col >= model.ncols || map[col].root != unset) {
      return fail("origin of reduced column", k);
    }
    map[col] = column_map_t{k, 1., 0.};
  }
  std::vector<uint8_t> seen(model.nrows, 0);
  for (int32_t k = 0; k < reduced.nrows; ++k) {
    int32_t row = ps.row_origin[k];
    if (row < 0 || row >= model.nrows || seen[row]) { return fail("origin of reduced row", k); }
    seen[row] = 1;
  }

  auto column_ok = [&](int32_t col) { return col >= 0 && col < model.ncols; };
  auto row_ok    = [&](int32_t row) { return row >= 0 && row < model.nrows; };
  std::vector<uint8_t> removed(model.ncols, 0);
  // Rows that a record took out of the model.
  std::vector<uint8_t> row_gone(model.nrows, 0);
  size_t fixed = 0, alias = 0, parallel = 0, addition = 0;
  int64_t left = model.ncols - reduced.ncols;
  for (size_t k = 0; k < ps.order.size(); ++k) {
    if (ps.order[k] == gprs::record_kind_t::fixed_col) {
      if (fixed == ps.fixed_cols.size()) { return fail("fixed records missing at", k); }
      const gprs::fixed_col_record_t& r = ps.fixed_cols[fixed++];
      if (!column_ok(r.col) || removed[r.col] || map[r.col].root != unset) {
        return fail("column of fixed record", k);
      }
      if (!std::isfinite(r.value) || (model.col_type[r.col] && !is_integral(r.value))) {
        return fail("value of fixed record", k);
      }
      removed[r.col] = 1;
      --left;
    } else if (ps.order[k] == gprs::record_kind_t::alias) {
      if (alias == ps.aliases.size()) { return fail("alias records missing at", k); }
      const gprs::alias_record_t& r = ps.aliases[alias++];
      if (!column_ok(r.col) || !column_ok(r.rep) || r.col == r.rep || removed[r.col] ||
          removed[r.rep] || map[r.col].root != unset) {
        return fail("columns of alias record", k);
      }
      bool doubleton = r.origin == gprs::alias_origin_t::doubleton;
      bool probing   = r.origin == gprs::alias_origin_t::probing;
      bool lock      = r.origin == gprs::alias_origin_t::lock;
      if (!doubleton && !probing && !lock) { return fail("origin of alias record", k); }
      if (lock && primal_only) { return fail("lock alias in a primal-only result at", k); }
      bool has_row = doubleton || (probing && r.row != -1);
      if (has_row ? !row_ok(r.row) : r.row != -1) { return fail("row of alias record", k); }
      if (!std::isfinite(r.scale) || r.scale == 0. || !std::isfinite(r.offset)) {
        return fail("scale or offset of alias record", k);
      }
      if (model.col_type[r.col] &&
          !(model.col_type[r.rep] && is_integral(r.scale) && is_integral(r.offset))) {
        return fail("integer column of alias record", k);
      }
      removed[r.col] = 1;
      if (has_row) { row_gone[r.row] = 1; }
      --left;
    } else if (ps.order[k] == gprs::record_kind_t::parallel_row) {
      if (parallel == ps.parallel_rows.size()) { return fail("parallel records missing at", k); }
      const gprs::parallel_row_record_t& r = ps.parallel_rows[parallel++];
      if (!row_ok(r.removed_row) || !row_ok(r.kept_row) || r.removed_row == r.kept_row ||
          !std::isfinite(r.ratio) || r.ratio == 0.) {
        return fail("parallel record", k);
      }
      row_gone[r.removed_row] = 1;
    } else if (ps.order[k] == gprs::record_kind_t::row_addition) {
      if (addition == ps.row_additions.size()) {
        return fail("row addition records missing at", k);
      }
      const gprs::row_addition_record_t& r = ps.row_additions[addition++];
      if (!row_ok(r.row) || !row_ok(r.equation_row) || r.row == r.equation_row ||
          !std::isfinite(r.factor) || r.factor == 0.) {
        return fail("row addition record", k);
      }
      if (row_gone[r.row] || row_gone[r.equation_row]) {
        return fail("rows of row addition record", k);
      }
    } else {
      return fail("unknown record kind at", k);
    }
  }
  if (fixed != ps.fixed_cols.size() || alias != ps.aliases.size() ||
      parallel != ps.parallel_rows.size() || addition != ps.row_additions.size()) {
    return fail("records outside the order", 0);
  }
  if (left != 0) { return fail("columns neither left nor removed:", left); }

  fixed = ps.fixed_cols.size();
  alias = ps.aliases.size();
  for (size_t k = ps.order.size(); k-- > 0;) {
    if (ps.order[k] == gprs::record_kind_t::fixed_col) {
      const gprs::fixed_col_record_t& r = ps.fixed_cols[--fixed];
      map[r.col]                        = column_map_t{-1, 0., r.value};
    } else if (ps.order[k] == gprs::record_kind_t::alias) {
      const gprs::alias_record_t& r = ps.aliases[--alias];
      const column_map_t& rep       = map[r.rep];
      map[r.col] = column_map_t{rep.root, r.scale * rep.scale, r.scale * rep.offset + r.offset};
    }
  }
  return true;
}

// One reduced column in the test of a row: g its coefficient in the original row, composed, c
// its coefficient in the reduced row.
struct row_term_t {
  double g;
  double c;
  double parts;  // sum of the absolute values g was composed of
  double lower;
  double upper;
};

// The largest value of sign * g * r over the bounds and low <= c * r <= high, as the smallest
// Lagrangian bound over the multipliers 0 and sign * g[k] / c[k]; infinity when none is
// finite. `largest` gets the largest term of that bound.
inline double implied_maximum(const std::vector<row_term_t>& terms,
                              double sign,
                              double low,
                              double high,
                              std::vector<double>& multipliers,
                              double& largest)
{
  double best = gprs::inf;
  double last = std::nan("");
  for (double lambda : multipliers) {
    lambda *= sign;
    if (std::fabs(lambda - last) <= same_ratio * std::fmax(std::fabs(lambda), std::fabs(last))) {
      continue;
    }
    last        = lambda;
    double side = lambda > 0 ? high : low;
    if (lambda != 0. && !std::isfinite(side)) { continue; }
    sum_t bound;
    double size = 1.;
    if (lambda != 0.) {
      bound.add(lambda, side);
      size = std::fmax(size, std::fabs(lambda * side));
    }
    bool finite = true;
    for (const row_term_t& t : terms) {
      double g     = sign * t.g;
      double moved = lambda * t.c;
      double w     = g - moved;
      if (std::fabs(w) <= composed_zero * std::fmax(std::fmax(1., t.parts), std::fabs(moved))) {
        continue;
      }
      double at = w > 0 ? t.upper : t.lower;
      if (!std::isfinite(at)) {
        finite = false;
        break;
      }
      bound.add(w, at);
      size = std::fmax(size, std::fabs(w * at));
    }
    if (finite && bound.value() < best) {
      best    = bound.value();
      largest = size;
    }
  }
  return best;
}

// The equations that the rows are made of, by original index. The row before a record
// row <- row - factor * equation_row is the row after it plus the factor times the equation as
// it is then, and that equation is in turn made of what is subtracted from it later; a row
// removed as a parallel row is, from then on, made of what its kept row is made of. Read from
// the records in reverse, each record therefore gives the row it names what the other row has
// collected so far. A row is not its own equation: that part is a multiple of its reduced row.
struct row_equations_t {
  // The list of a row in `lists`, -1 for a row that is made of itself alone.
  std::vector<int32_t> list;
  std::vector<std::vector<int32_t>> lists;

  // A list longer than max_row_equations is not completed: its row is not tested.
  void add(int32_t row, int32_t equation)
  {
    if (equation == row) { return; }
    if (list[row] < 0) {
      list[row] = lists.size();
      lists.emplace_back();
    }
    std::vector<int32_t>& of = lists[list[row]];
    if (of.size() > max_row_equations || std::find(of.begin(), of.end(), equation) != of.end()) {
      return;
    }
    of.push_back(equation);
  }

  void add_all(int32_t row, int32_t from)
  {
    if (list[from] < 0) { return; }
    for (size_t k = 0; k < lists[list[from]].size(); ++k) {
      add(row, lists[list[from]][k]);
    }
  }

  // The stack keeps its contract (compose_records).
  explicit row_equations_t(const gprs::postsolve_t& ps) : list(ps.nrows, -1)
  {
    size_t parallel = ps.parallel_rows.size(), addition = ps.row_additions.size();
    for (size_t k = ps.order.size(); k-- > 0;) {
      if (ps.order[k] == gprs::record_kind_t::parallel_row) {
        const gprs::parallel_row_record_t& r = ps.parallel_rows[--parallel];
        add_all(r.removed_row, r.kept_row);
      } else if (ps.order[k] == gprs::record_kind_t::row_addition) {
        const gprs::row_addition_record_t& r = ps.row_additions[--addition];
        add(r.row, r.equation_row);
        add_all(r.row, r.equation_row);
      }
    }
  }
};

// The test of a row against several reduced rows together: its own, if it has one, as row 0,
// and those of its equations. A point v holds one multiplier per row.
struct combined_t {
  struct column_t {
    double g;      // coefficient in the original row, composed
    double parts;  // sum of the absolute values g was composed of
    double lower;
    double upper;
  };
  bool own     = false;
  int32_t rows = 0;
  std::vector<double> low;
  std::vector<double> high;
  std::vector<column_t> columns;
  // Coefficient of column k in row i at k * rows + i.
  std::vector<double> a;
};

inline void sort_different(std::vector<double>& values)
{
  std::sort(values.begin(), values.end());
  size_t kept = 0;
  for (double a : values) {
    if (kept > 0 && std::fabs(a - values[kept - 1]) <=
                      same_ratio * std::fmax(std::fabs(a), std::fabs(values[kept - 1]))) {
      continue;
    }
    values[kept++] = a;
  }
  values.resize(kept);
}

// The Lagrangian bound of the largest value of sign * g * r over the bounds and the rows of p,
// for the multipliers m = sign * v:
//   sum of m[i] * (high[i] or low[i]) + max over the bounds of (sign * g - sum of m[i] * a[i]) * r
// with the side of the sign of each multiplier; infinity when a side or a bound it needs is.
// A coefficient of that maximum counts as zero when it is a cancellation of the row itself
// (composed_zero of its parts) or the rounding of the terms of the reduced rows (rounding_zero
// of their sum). No larger share of those terms: multipliers that cancel each other make the
// sum as large as they are, and a coefficient of the size of the row's own would vanish in it.
// `largest` gets the largest term of the bound, the term of an equation counted as at least its
// multiplier, and `multiplier` the largest multiplier of an equation.
// `size` gets what a violation of the row is relative to: the bound itself, the row's own
// coefficients times the bounds that the bound uses, and the multiplier of an equation. The
// sides of the reduced rows and the columns that only they hold count with rounding_share:
// their terms cancel in the bound (the side of the row after a subtraction holds the
// right-hand side of the equation, the side of the equation gives it back), so a huge one says
// what its rounding can do to the row and no more.
inline double combined_bound(const combined_t& p,
                             double sign,
                             const double* v,
                             double& size,
                             double& largest,
                             double& multiplier)
{
  sum_t bound;
  size       = 1.;
  largest    = 1.;
  multiplier = 0.;
  for (int32_t i = 0; i < p.rows; ++i) {
    double m = sign * v[i];
    if (m == 0.) { continue; }
    double side = m > 0 ? p.high[i] : p.low[i];
    if (!std::isfinite(side)) { return gprs::inf; }
    bound.add(m, side);
    largest = std::fmax(largest, std::fabs(m * side));
    if (!(p.own && i == 0)) {
      size       = std::fmax(size, std::fabs(m));
      largest    = std::fmax(largest, std::fabs(m));
      multiplier = std::fmax(multiplier, std::fabs(m));
    }
  }
  for (size_t k = 0; k < p.columns.size(); ++k) {
    const combined_t::column_t& c = p.columns[k];
    double w = sign * c.g, moved = 0.;
    for (int32_t i = 0; i < p.rows; ++i) {
      double term = sign * v[i] * p.a[k * p.rows + i];
      w -= term;
      moved += std::fabs(term);
    }
    if (std::fabs(w) <= std::fmax(composed_zero * std::fmax(1., c.parts), rounding_zero * moved)) {
      continue;
    }
    double at = w > 0 ? c.upper : c.lower;
    if (!std::isfinite(at)) { return gprs::inf; }
    bound.add(w, at);
    largest = std::fmax(largest, std::fabs(w * at));
    size    = std::fmax(size, std::fabs(c.g * at));
  }
  size = std::fmax(size, std::fmax(std::fabs(bound.value()), rounding_share * largest));
  return bound.value();
}

// The bound is linear in v between the hyperplanes where one of its terms changes its sign:
// v[i] = 0 for a row, and a[k] * v = g[k] for a column. Where it has a smallest value, it has
// it at a point where as many of them meet as there are rows. A column of one row gives a
// hyperplane v[i] = g[k] / a[k][i]: axis[i] holds the different values of that kind, with 0;
// `general` holds the columns of two rows or more.
struct hyperplanes_t {
  std::vector<std::vector<double>> axis;
  std::vector<int32_t> general;
  // Right-hand side of the hyperplane of each column of `general`.
  std::vector<double> rhs;

  explicit hyperplanes_t(const combined_t& p) : axis(p.rows)
  {
    for (size_t k = 0; k < p.columns.size(); ++k) {
      int32_t held = 0, row = 0;
      for (int32_t i = 0; i < p.rows; ++i) {
        if (p.a[k * p.rows + i] != 0.) {
          ++held;
          row = i;
        }
      }
      if (held == 1) { axis[row].push_back(p.columns[k].g / p.a[k * p.rows + row]); }
      if (held > 1) {
        general.push_back(k);
        rhs.push_back(p.columns[k].g);
      }
    }
    for (std::vector<double>& values : axis) {
      values.push_back(0.);
      sort_different(values);
    }
  }

  // The number of points that visit_vertices tries: every choice of the rows that take a value
  // of their axis, a value for each, and as many columns of `general` as rows are left.
  double count() const
  {
    // chosen[j]: the choices of j rows with a value each.
    std::vector<double> chosen(axis.size() + 1, 0.);
    chosen[0] = 1.;
    for (size_t i = 0; i < axis.size(); ++i) {
      for (size_t j = i + 1; j-- > 0;) {
        chosen[j + 1] += chosen[j] * (double)axis[i].size();
      }
    }
    double total = 0., subsets = 1.;
    for (size_t m = 0; m <= axis.size() && m <= general.size(); ++m) {
      total += subsets * chosen[axis.size() - m];
      subsets *= (double)(general.size() - m) / (double)(m + 1);
    }
    return total;
  }
};

// Calls visit(v) for every point v where p.rows of the hyperplanes meet in one point (no pivot
// of their system below singular_pivot); the rows of `fixed` (a bit per row) always take a
// value of their axis.
template <typename visit_t>
void visit_vertices(const combined_t& p, const hyperplanes_t& h, uint32_t fixed, visit_t visit)
{
  int32_t d = p.rows, planes = h.general.size();
  std::vector<double> v(d), matrix, scale, scaled;
  std::vector<int32_t> free_rows, fixed_rows, value, chosen;
  uint32_t masks = planes == 0 ? 1 : (uint32_t)1 << d;
  for (uint32_t free_mask = 0; free_mask < masks; ++free_mask) {
    if (free_mask & fixed) { continue; }
    free_rows.clear();
    fixed_rows.clear();
    for (int32_t i = 0; i < d; ++i) {
      ((free_mask >> i) & 1 ? free_rows : fixed_rows).push_back(i);
    }
    int32_t m = free_rows.size();
    if (m > planes) { continue; }
    value.assign(fixed_rows.size(), 0);
    for (bool more_values = true; more_values;) {
      for (size_t n = 0; n < fixed_rows.size(); ++n) {
        v[fixed_rows[n]] = h.axis[fixed_rows[n]][value[n]];
      }
      chosen.resize(m);
      for (int32_t n = 0; n < m; ++n) {
        chosen[n] = n;
      }
      for (bool more_planes = true; more_planes;) {
        // The free rows from the chosen hyperplanes: the system with every column and then
        // every row scaled to a largest entry of 1, elimination with row pivoting, back
        // substitution. A system with a pivot below singular_pivot gives no point: hyperplanes
        // that are parallel on the free rows leave a pivot that is the rounding of zero, and
        // its multipliers, as large as that rounding is small, would cancel each other in every
        // coefficient that the hyperplanes share and stand for a bound that is none.
        matrix.assign(m * (m + 1), 0.);
        for (int32_t r = 0; r < m; ++r) {
          int32_t k        = h.general[chosen[r]];
          double remainder = h.rhs[chosen[r]];
          for (int32_t i : fixed_rows) {
            remainder -= p.a[k * d + i] * v[i];
          }
          for (int32_t n = 0; n < m; ++n) {
            matrix[r * (m + 1) + n] = p.a[k * d + free_rows[n]];
          }
          matrix[r * (m + 1) + m] = remainder;
        }
        bool regular = true;
        scale.assign(m, 0.);
        scaled.assign(m, 0.);
        for (int32_t n = 0; n < m && regular; ++n) {
          for (int32_t r = 0; r < m; ++r) {
            scale[n] = std::fmax(scale[n], std::fabs(matrix[r * (m + 1) + n]));
          }
          regular = scale[n] > 0.;
        }
        for (int32_t r = 0; r < m && regular; ++r) {
          double most = 0.;
          for (int32_t n = 0; n < m; ++n) {
            matrix[r * (m + 1) + n] /= scale[n];
            most = std::fmax(most, std::fabs(matrix[r * (m + 1) + n]));
          }
          regular = most > 0.;
          if (!regular) { break; }
          for (int32_t c = 0; c <= m; ++c) {
            matrix[r * (m + 1) + c] /= most;
          }
        }
        for (int32_t n = 0; n < m && regular; ++n) {
          int32_t pivot = n;
          for (int32_t r = n + 1; r < m; ++r) {
            if (std::fabs(matrix[r * (m + 1) + n]) > std::fabs(matrix[pivot * (m + 1) + n])) {
              pivot = r;
            }
          }
          regular = std::fabs(matrix[pivot * (m + 1) + n]) >= singular_pivot;
          if (!regular) { break; }
          for (int32_t c = 0; c <= m; ++c) {
            std::swap(matrix[n * (m + 1) + c], matrix[pivot * (m + 1) + c]);
          }
          for (int32_t r = n + 1; r < m; ++r) {
            double factor = matrix[r * (m + 1) + n] / matrix[n * (m + 1) + n];
            for (int32_t c = n; c <= m; ++c) {
              matrix[r * (m + 1) + c] -= factor * matrix[n * (m + 1) + c];
            }
          }
        }
        for (int32_t n = m; regular && n-- > 0;) {
          double rest = matrix[n * (m + 1) + m];
          for (int32_t c = n + 1; c < m; ++c) {
            rest -= matrix[n * (m + 1) + c] * scaled[c];
          }
          scaled[n]       = rest / matrix[n * (m + 1) + n];
          v[free_rows[n]] = scaled[n] / scale[n];
          regular         = std::isfinite(v[free_rows[n]]);
        }
        if (regular) { visit(v.data()); }
        // The next choice of m of the columns.
        int32_t n = m - 1;
        while (n >= 0 && chosen[n] == planes - m + n) {
          --n;
        }
        more_planes = n >= 0;
        if (more_planes) {
          ++chosen[n];
          for (int32_t r = n + 1; r < m; ++r) {
            chosen[r] = chosen[r - 1] + 1;
          }
        }
      }
      // The next choice of a value for each fixed row.
      size_t n = 0;
      while (n < fixed_rows.size() && value[n] + 1 == (int32_t)h.axis[fixed_rows[n]].size()) {
        value[n++] = 0;
      }
      more_values = n < fixed_rows.size();
      if (more_values) { ++value[n]; }
    }
  }
}

// The smallest bound that a search has visited, with what a violation is relative to and its
// largest multiplier of an equation (combined_bound); infinity before the first finite one.
struct smallest_bound_t {
  double value      = gprs::inf;
  double size       = 1.;
  double multiplier = 0.;

  void visit(const combined_t& p, double sign, const double* v)
  {
    double bound_size, largest, most;
    double bound = combined_bound(p, sign, v, bound_size, largest, most);
    if (bound < value) {
      value      = bound;
      size       = bound_size;
      multiplier = most;
    }
  }
};

// No point within the bounds meets the rows of p together: the bound of the function zero is
// then negative for some multipliers, and a search for a row has no smallest bound. That bound
// is linear on every cone between the hyperplanes through the origin, a[k] * v = 0 and
// v[i] = 0, so it is negative somewhere when it is on a line where all but one of them meet:
// at a point with one multiplier 1 or -1 where as many of the others meet as there are other
// rows.
inline bool no_common_point(combined_t p)
{
  for (combined_t::column_t& c : p.columns) {
    c.g     = 0.;
    c.parts = 0.;
  }
  hyperplanes_t h(p);
  bool negative = false;
  for (int32_t unit = 0; unit < p.rows && !negative; ++unit) {
    h.axis[unit] = {-1., 1.};
    visit_vertices(p, h, (uint32_t)1 << unit, [&](const double* v) {
      double size, largest, most;
      negative =
        negative || combined_bound(p, 1., v, size, largest, most) < -check_tolerance * largest;
    });
    h.axis[unit] = {0.};
  }
  return negative;
}

// The objective part of the check. The stack keeps its contract and `map` holds its records
// composed (compose_records). O(columns + records).
inline void objective_check(const gprs::model_t& model,
                            const gprs::result_t& result,
                            const std::vector<column_map_t>& map,
                            structure_report_t& s)
{
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;
  // The sum of the absolute values that the composed offset of a column is made of: an alias
  // adds its own offset to the scaled offset of its rep, and the two can cancel.
  std::vector<double> offset_parts(model.ncols, 0.);
  size_t fixed = ps.fixed_cols.size(), alias = ps.aliases.size();
  for (size_t k = ps.order.size(); k-- > 0;) {
    if (ps.order[k] == gprs::record_kind_t::fixed_col) {
      const gprs::fixed_col_record_t& r = ps.fixed_cols[--fixed];
      offset_parts[r.col]               = std::fabs(r.value);
    } else if (ps.order[k] == gprs::record_kind_t::alias) {
      const gprs::alias_record_t& r = ps.aliases[--alias];
      offset_parts[r.col] = std::fabs(r.scale) * offset_parts[r.rep] + std::fabs(r.offset);
    }
  }

  std::vector<sum_t> coefficient(reduced.ncols);
  std::vector<double> parts(reduced.ncols, 0.);
  sum_t constant;
  constant.add(model.objective_offset, 1.);
  constant.add(-reduced.objective_offset, 1.);
  double terms = std::fabs(model.objective_offset) + std::fabs(reduced.objective_offset);
  for (int32_t j = 0; j < model.ncols; ++j) {
    double cost = model.cost[j];
    if (cost == 0.) { continue; }
    const column_map_t& m = map[j];
    constant.add(cost, m.offset);
    terms += std::fabs(cost) * offset_parts[j];
    if (m.root < 0) { continue; }
    coefficient[m.root].add(cost, m.scale);
    parts[m.root] += std::fabs(cost * m.scale);
  }
  for (int32_t k = 0; k < reduced.ncols; ++k) {
    coefficient[k].add(-reduced.cost[k], 1.);
    parts[k] += std::fabs(reduced.cost[k]);
    double difference = std::fabs(coefficient[k].value());
    if (difference <= underflow_size) { continue; }
    double relative = difference / parts[k];
    if (!(relative <= s.objective_worst)) { s.objective_worst_col = ps.col_origin[k]; }
    raise_to(s.objective_worst, relative);
    s.objective_failures += !(relative <= composed_zero);
  }
  double difference = std::fabs(constant.value());
  if (!(difference <= underflow_size)) {
    s.objective_constant = difference / terms;
    s.objective_failures += !(s.objective_constant <= identity_tolerance);
  }
}

// The check of one result of a presolve in the given mode (options_t::primal_only).
inline structure_report_t structure_check(const gprs::model_t& model,
                                          const gprs::result_t& result,
                                          bool primal_only)
{
  structure_report_t s;
  s.ran = true;
  std::vector<column_map_t> map;
  if (!compose_records(model, result, primal_only, map, s.stack_error)) { return s; }
  s.composed                   = true;
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;

  for (int32_t k = 0; k < reduced.ncols; ++k) {
    int32_t col = ps.col_origin[k];
    if (model.col_type[col] && !reduced.col_type[k]) {
      if (s.lost_integers++ == 0) { s.first_lost_integer = col; }
    }
  }

  objective_check(model, result, map, s);

  for (int32_t j = 0; j < model.ncols; ++j) {
    const column_map_t& m = map[j];
    double low = m.offset, high = m.offset;
    if (m.root >= 0) {
      double a = m.scale * reduced.col_lower[m.root] + m.offset;
      double b = m.scale * reduced.col_upper[m.root] + m.offset;
      low      = m.scale > 0 ? a : b;
      high     = m.scale > 0 ? b : a;
    }
    double below     = model.col_lower[j] == -gprs::inf ? -gprs::inf : model.col_lower[j] - low;
    double above     = model.col_upper[j] == gprs::inf ? -gprs::inf : high - model.col_upper[j];
    double violation = !(below <= above) ? below : above;
    if (violation <= 0.) { continue; }
    double bound  = below > above ? model.col_lower[j] : model.col_upper[j];
    double scaled = violation / std::fmax(1., std::fabs(bound));
    if (!(scaled <= s.bound_worst)) { s.bound_worst_col = j; }
    raise_to(s.bound_worst, scaled);
    s.bound_failures += !(scaled <= structure_tolerance);
  }

  // The reduced row an original row is tested against: its own, or that of the row it was
  // merged into as a parallel row, or none.
  std::vector<int32_t> reduced_row(model.nrows, -1), kept_row(model.nrows, -1);
  for (int32_t k = 0; k < reduced.nrows; ++k) {
    reduced_row[ps.row_origin[k]] = k;
  }
  for (const gprs::parallel_row_record_t& r : ps.parallel_rows) {
    kept_row[r.removed_row] = r.kept_row;
  }
  auto final_row = [&](int32_t row) {
    for (size_t hops = 0;
         reduced_row[row] < 0 && kept_row[row] >= 0 && hops <= ps.parallel_rows.size();
         ++hops) {
      row = kept_row[row];
    }
    return row;
  };
  row_equations_t equations(ps);
  std::vector<double> g(reduced.ncols, 0.), parts(reduced.ncols, 0.), c(reduced.ncols, 0.);
  std::vector<int32_t> touched;
  std::vector<row_term_t> terms;
  std::vector<double> multipliers;
  // For a row that is made of equations: their reduced rows, the columns of the test with the
  // place of each in it, and the entries of the rows of the test, up to entry_end for each row.
  std::vector<int32_t> equation_rows, cols, slot(reduced.ncols, -1), entry_col;
  std::vector<double> entry_value;
  std::vector<size_t> entry_end(max_row_equations + 1);
  combined_t p;
  double kept_work = 0.;
  for (int32_t i = 0; i < model.nrows; ++i) {
    int32_t against = final_row(i);
    int32_t k       = reduced_row[against];

    sum_t constant;
    double largest = 1.;
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      const column_map_t& m = map[model.col_index[e]];
      double a              = model.value[e];
      constant.add(a, m.offset);
      raise_to(largest, std::fabs(a * m.offset));
      if (m.root < 0) { continue; }
      if (parts[m.root] == 0.) { touched.push_back(m.root); }
      g[m.root] += a * m.scale;
      parts[m.root] += std::fabs(a * m.scale);
    }
    double lower = model.row_lower[i], upper = model.row_upper[i];
    double below = -gprs::inf, above = -gprs::inf;

    equation_rows.clear();
    bool too_many = false;
    if (equations.list[i] >= 0) {
      ++s.added_rows;
      const std::vector<int32_t>& of = equations.lists[equations.list[i]];
      too_many                       = of.size() > max_row_equations;
      for (int32_t equation : of) {
        int32_t e = reduced_row[final_row(equation)];
        if (e >= 0 && e != k &&
            std::find(equation_rows.begin(), equation_rows.end(), e) == equation_rows.end()) {
          equation_rows.push_back(e);
        }
      }
    }
    if (too_many) {
      for (int32_t col : touched) {
        g[col]     = 0.;
        parts[col] = 0.;
      }
      touched.clear();
      ++s.added_untested;
      continue;
    }
    if (!equation_rows.empty()) {
      p.own  = k >= 0;
      p.rows = p.own + equation_rows.size();
      p.low.clear();
      p.high.clear();
      p.columns.clear();
      cols.clear();
      entry_col.clear();
      entry_value.clear();
      for (int32_t col : touched) {
        slot[col] = p.columns.size();
        cols.push_back(col);
        p.columns.push_back(
          combined_t::column_t{g[col], parts[col], reduced.col_lower[col], reduced.col_upper[col]});
        g[col]     = 0.;
        parts[col] = 0.;
      }
      touched.clear();
      // The entries of the rows, row after row; a column that only they have gets g = 0.
      for (int32_t n = 0; n < p.rows; ++n) {
        int32_t row = p.own ? (n == 0 ? k : equation_rows[n - 1]) : equation_rows[n];
        p.low.push_back(reduced.row_lower[row]);
        p.high.push_back(reduced.row_upper[row]);
        for (int64_t e = reduced.row_start[row]; e < reduced.row_start[row + 1]; ++e) {
          int32_t col = reduced.col_index[e];
          if (slot[col] < 0) {
            slot[col] = p.columns.size();
            cols.push_back(col);
            p.columns.push_back(
              combined_t::column_t{0., 0., reduced.col_lower[col], reduced.col_upper[col]});
          }
          entry_col.push_back(slot[col]);
          entry_value.push_back(reduced.value[e]);
        }
        entry_end[n] = entry_col.size();
      }
      for (int32_t col : cols) {
        slot[col] = -1;
      }
      p.a.assign(p.columns.size() * p.rows, 0.);
      for (size_t e = 0, n = 0; e < entry_col.size(); ++e) {
        while (e == entry_end[n]) {
          ++n;
        }
        p.a[entry_col[e] * p.rows + n] = entry_value[e];
      }
      hyperplanes_t h(p);
      double work = h.count() * (double)p.a.size();
      if (work > max_kept_row_work || kept_work + work > max_kept_work) {
        ++s.added_untested;
        continue;
      }
      kept_work += work;
      smallest_bound_t most, least;
      visit_vertices(p, h, 0, [&](const double* v) {
        if (upper < gprs::inf) { most.visit(p, 1., v); }
        if (lower > -gprs::inf) { least.visit(p, -1., v); }
      });
      if (upper < gprs::inf) { above = constant.value() + most.value - upper; }
      if (lower > -gprs::inf) { below = lower - constant.value() + least.value; }
      double violation = !(below <= above) ? below : above;
      double scaled    = violation / std::fmax(largest, below > above ? least.size : most.size);
      // A bound that proves the row does so whether a point meets the reduced rows or not; a
      // search that finds none says nothing when no point meets them.
      if (violation > 0. && !(scaled <= structure_tolerance) && no_common_point(p)) {
        ++s.added_untested;
        continue;
      }
      ++s.added_tested;
      raise_to(s.added_multiplier, std::fmax(most.multiplier, least.multiplier));
      if (violation <= 0.) { continue; }
      if (!(scaled <= s.added_worst)) { s.added_worst_row = i; }
      raise_to(s.added_worst, scaled);
      s.added_failures += !(scaled <= structure_tolerance);
      continue;
    }

    if (k < 0) {
      // Each end has its own size: a huge bound at one end says nothing about the other.
      sum_t low = constant, high = constant;
      double largest_low = largest, largest_high = largest;
      bool low_infinite = false, high_infinite = false;
      for (int32_t col : touched) {
        double coefficient = g[col];
        bool cancelled     = std::fabs(coefficient) <= composed_zero * std::fmax(1., parts[col]);
        g[col]             = 0.;
        parts[col]         = 0.;
        if (cancelled) { continue; }
        double at_low  = coefficient > 0 ? reduced.col_lower[col] : reduced.col_upper[col];
        double at_high = coefficient > 0 ? reduced.col_upper[col] : reduced.col_lower[col];
        if (std::isfinite(at_low)) {
          low.add(coefficient, at_low);
          raise_to(largest_low, std::fabs(coefficient * at_low));
        } else {
          low_infinite = true;
        }
        if (std::isfinite(at_high)) {
          high.add(coefficient, at_high);
          raise_to(largest_high, std::fabs(coefficient * at_high));
        } else {
          high_infinite = true;
        }
      }
      touched.clear();
      if (lower > -gprs::inf) { below = low_infinite ? gprs::inf : lower - low.value(); }
      if (upper < gprs::inf) { above = high_infinite ? gprs::inf : high.value() - upper; }
      double violation = !(below <= above) ? below : above;
      ++s.rows_tested;
      if (violation <= 0.) { continue; }
      double scaled = violation / (!(below <= above) ? largest_low : largest_high);
      if (!(scaled <= s.row_worst)) { s.row_worst_row = i; }
      raise_to(s.row_worst, scaled);
      s.row_failures += !(scaled <= structure_tolerance);
      continue;
    }

    for (int64_t e = reduced.row_start[k]; e < reduced.row_start[k + 1]; ++e) {
      int32_t col = reduced.col_index[e];
      if (parts[col] == 0. && c[col] == 0.) { touched.push_back(col); }
      c[col] = reduced.value[e];
    }
    terms.clear();
    multipliers.assign(1, 0.);
    sum_t least, most;
    bool least_infinite = false, most_infinite = false;
    for (int32_t col : touched) {
      row_term_t t{g[col], c[col], parts[col], reduced.col_lower[col], reduced.col_upper[col]};
      g[col]     = 0.;
      parts[col] = 0.;
      c[col]     = 0.;
      terms.push_back(t);
      if (t.c == 0.) { continue; }
      multipliers.push_back(t.g / t.c);
      double at_least = t.c > 0 ? t.lower : t.upper;
      double at_most  = t.c > 0 ? t.upper : t.lower;
      if (std::isfinite(at_least)) {
        least.add(t.c, at_least);
      } else {
        least_infinite = true;
      }
      if (std::isfinite(at_most)) {
        most.add(t.c, at_most);
      } else {
        most_infinite = true;
      }
    }
    touched.clear();
    std::sort(multipliers.begin(), multipliers.end());
    size_t different = 1;
    for (size_t m = 1; m < multipliers.size(); ++m) {
      double a = multipliers[m - 1], b = multipliers[m];
      different += !(std::fabs(b - a) <= same_ratio * std::fmax(std::fabs(a), std::fabs(b)));
    }
    double low = reduced.row_lower[k], high = reduced.row_upper[k];
    bool empty =
      (!least_infinite &&
       least.value() - high > check_tolerance * std::fmax(1., std::fabs(high))) ||
      (!most_infinite && low - most.value() > check_tolerance * std::fmax(1., std::fabs(low)));
    double work = (double)different * (double)terms.size();
    if (empty || work > max_kept_row_work || kept_work + work > max_kept_work) {
      ++s.kept_untested;
      continue;
    }
    kept_work += work;
    ++s.kept_tested;
    s.kept_parallel += against != i;
    double size_above = 1., size_below = 1.;
    if (upper < gprs::inf) {
      above =
        constant.value() + implied_maximum(terms, 1., low, high, multipliers, size_above) - upper;
    }
    if (lower > -gprs::inf) {
      below =
        lower - constant.value() + implied_maximum(terms, -1., low, high, multipliers, size_below);
    }
    double violation = !(below <= above) ? below : above;
    if (violation <= 0.) { continue; }
    double scaled = violation / std::fmax(largest, below > above ? size_below : size_above);
    if (!(scaled <= s.kept_worst)) { s.kept_worst_row = i; }
    raise_to(s.kept_worst, scaled);
    s.kept_failures += !(scaled <= structure_tolerance);
  }

  std::vector<double> point(reduced.ncols), z;
  for (int32_t k = 0; k < reduced.ncols; ++k) {
    double v = 1.;
    if (v < reduced.col_lower[k]) { v = reduced.col_lower[k]; }
    if (v > reduced.col_upper[k]) { v = reduced.col_upper[k]; }
    point[k] = model.col_type[ps.col_origin[k]] ? std::round(v) : v;
  }
  ps.postsolve(point, z);
  for (int32_t j = 0; j < model.ncols; ++j) {
    const column_map_t& m = map[j];
    double term           = m.root < 0 ? 0. : m.scale * point[m.root];
    double size           = std::fmax(1., std::fmax(std::fabs(term), std::fabs(m.offset)));
    double difference     = std::fabs(z[j] - (term + m.offset)) / size;
    if (!(difference <= s.postsolve_worst)) { s.postsolve_worst_col = j; }
    raise_to(s.postsolve_worst, difference);
    s.postsolve_failures += !(difference <= composed_tolerance);
  }
  return s;
}

}  // namespace gprs_check
