/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The structure check: what one presolve result must satisfy whatever point is postsolved. It
// reads the postsolve stack, the origin maps and the reduced model, needs no solver and no
// solution file, and costs O(columns + records + nonzeros + saved entries) (a row whose
// coefficients were edited one by one costs its length times the number of its different
// coefficient ratios; a row that had equations subtracted costs the entries of its reduced row and
// of theirs times the points that its search visits, see added_rows). It sees what the crush check
// cannot: a reduction that relaxes.
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
//              forcing record; with options_t::primal_projections it may have interval and
//              forcing records, which are projections and no dual reductions. The saved row
//              of an interval record has two finite sides; with
//              options_t::zero_cost_one_sided it may have one.
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
//              record's columns. The second computation follows the two choices that the
//              stack carries for its undo (postsolve_t::interval_descending,
//              forcing_rounding_units). The value of an integer forcing column is a rounded
//              sum, and the two computations sum in another order. With the band of 8 rounding
//              units, which is also what the two sums can differ by, they must agree. With
//              another band of the stack they can give neighbouring integers: where they do
//              and the sum lies within forcing_rounding (or that band, if wider) of the
//              integer between them, the value of postsolve() stands, and the rows and the
//              bounds are tested with it. At each of the points every saved row of a defined
//              or a forcing record holds at the postsolved point, that of an interval record
//              where the rest of the row lies inside the sides of its snapshot, and the
//              columns of interval and forcing records are inside their bounds.
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
// A ratio of two coefficients within this of an integer is taken as that integer where a saved
// row defines an integer column; what the rounding moves the row by is tested apart.
constexpr double integral_ratio = 1e-7;
// Interval records that get a point of their own in the postsolve part.
constexpr size_t max_replay_points = 8;
// What the value that a row leaves an integer forcing column can differ by between two
// summations, as a share of the terms it is computed from. The band of the undo itself is
// postsolve_t::forcing_rounding_units, 8 rounding units as well by default.
constexpr double forcing_rounding = 8 * 0x1p-53;

// The larger absolute value of the finite sides of an anchor; 0 without one.
inline double side_size(double low, double high)
{
  double size = 0.;
  if (std::isfinite(low)) { size = std::fabs(low); }
  if (std::isfinite(high)) { size = std::fmax(size, std::fabs(high)); }
  return size;
}

// x[original column] = scale * v[root] + offset over the check variables; root -1 for a
// constant.
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

// What happens to an original row at one position of the stack.
struct row_event_t {
  enum kind_t : uint8_t { saved, parallel, addition };
  size_t position;
  kind_t kind;
  // saved: the anchor of the saved row; parallel: the kept row; addition: the equation row.
  int32_t index;
};

// The records composed: the check variables with their boxes, the map of every original column
// onto them, the anchors, and what the stack does to each row.
struct system_t {
  int32_t reduced_cols = 0;
  int32_t reduced_rows = 0;
  std::vector<column_map_t> map;
  // By original column: the sum of the absolute values that the offset of its map is made of.
  // An alias adds its own offset to the scaled offset of its rep, and the two can cancel.
  std::vector<double> offset_parts;
  // Variables: the reduced columns, then the columns of the col_from_rows records in the order
  // of the stack.
  std::vector<double> lower;
  std::vector<double> upper;
  std::vector<int32_t> origin;  // original column
  // Anchors: the rows of the reduced model, then the saved rows of the col_from_rows records,
  // each with its entries [start[a], start[a + 1]) over the variables, the record's own columns
  // among them, and its sides less the constants of the composition.
  std::vector<int64_t> start;
  std::vector<int32_t> var;
  std::vector<double> value;
  std::vector<double> low;
  std::vector<double> high;
  std::vector<int32_t> row;  // original row
  // By anchor: the sum of the absolute values that its sides are made of, which are the side of
  // the saved row and the constants of the composition; 0 for a row of the reduced model.
  std::vector<double> side_parts;
  // By original row: its row of the reduced model as an anchor, or -1, and its events
  // [first[row], first[row + 1]) in the order of the stack.
  std::vector<int32_t> reduced_anchor;
  std::vector<int32_t> first;
  std::vector<row_event_t> events;
  // By col_from_rows record: its position in the order of the stack, its first variable, and
  // the anchor of its first saved row; the others follow.
  std::vector<size_t> position;
  std::vector<int32_t> first_var;
  std::vector<int32_t> first_anchor;

  int32_t nvars() const { return lower.size(); }
  int32_t anchors() const { return low.size(); }
};

// A forcing record bounds its column from above (postsolve.hpp).
inline bool forcing_from_above(const gprs::postsolve_t& ps, const gprs::col_from_rows_record_t& r)
{
  return (ps.record_coefficients[r.coefficients] > 0) ==
         (ps.saved_rows[r.rows_begin].upper < gprs::inf);
}

// Tests the contract of the stack, for a result of the given mode, and composes its records. On
// a broken contract `error` says which part, with the index of the record or column.
// `projections`: a result of primal-only mode may hold interval and forcing records
// (options_t::primal_projections). `one_sided`: the saved row of an interval record may have
// one finite side (options_t::zero_cost_one_sided).
inline bool compose_records(const gprs::model_t& model,
                            const gprs::result_t& result,
                            bool primal_only,
                            bool projections,
                            bool one_sided,
                            system_t& sys,
                            std::string& error)
{
  const gprs::postsolve_t& ps    = result.postsolve;
  const gprs::model_t& reduced   = result.reduced;
  std::vector<column_map_t>& map = sys.map;
  auto fail                      = [&](const char* what, int64_t index) {
    error = std::string(what) + " " + std::to_string(index);
    return false;
  };
  if (ps.ncols != model.ncols || ps.nrows != model.nrows) { return fail("sizes of the stack", 0); }
  if (ps.col_type != model.col_type) { return fail("column types of the stack", 0); }
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
  size_t fixed = 0, alias = 0, parallel = 0, addition = 0, from_rows = 0;
  int32_t cols_used = 0, rows_used = 0;
  int64_t coefficients_used = 0, entries_used = 0;
  int64_t left = model.ncols - reduced.ncols;
  std::vector<int32_t> events_of(model.nrows + 1, 0);
  for (size_t k = 0; k < ps.order.size(); ++k) {
    if (ps.order[k] == gprs::record_kind_t::fixed_col) {
      if (fixed == ps.fixed_cols.size()) { return fail("fixed records missing at", k); }
      const gprs::fixed_col_record_t& r = ps.fixed_cols[fixed++];
      if (!column_ok(r.col) || removed[r.col] || map[r.col].root != unset) {
        return fail("column of fixed record", k);
      }
      if (!std::isfinite(r.value) || (ps.integer_at(r.col, k) && !is_integral(r.value))) {
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
      if (ps.integer_at(r.col, k) &&
          !(ps.integer_at(r.rep, k) && is_integral(r.scale) && is_integral(r.offset))) {
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
      ++events_of[r.removed_row + 1];
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
      ++events_of[r.row + 1];
    } else if (ps.order[k] == gprs::record_kind_t::col_from_rows) {
      if (from_rows == ps.col_from_rows.size()) {
        return fail("col_from_rows records missing at", k);
      }
      const gprs::col_from_rows_record_t& r = ps.col_from_rows[from_rows++];
      int64_t ncols                         = (int64_t)r.cols_end - r.cols_begin,
              nrows                         = (int64_t)r.rows_end - r.rows_begin;
      int64_t count                         = ncols > nrows ? ncols : nrows;
      if (r.cols_begin < cols_used || ncols < 1 || (size_t)r.cols_end > ps.record_cols.size() ||
          r.rows_begin < rows_used || nrows < 1 || (size_t)r.rows_end > ps.saved_rows.size() ||
          r.coefficients < coefficients_used ||
          (size_t)(r.coefficients + count) > ps.record_coefficients.size()) {
        return fail("ranges of col_from_rows record", k);
      }
      cols_used         = r.cols_end;
      rows_used         = r.rows_end;
      coefficients_used = r.coefficients + count;
      bool defined      = r.kind == gprs::col_from_rows_kind_t::defined;
      bool interval     = r.kind == gprs::col_from_rows_kind_t::interval;
      bool forcing      = r.kind == gprs::col_from_rows_kind_t::forcing;
      if ((!defined && !interval && !forcing) || (!interval && ncols != 1) ||
          (!forcing && nrows != 1) || (r.dual && !defined)) {
        return fail("kind or counts of col_from_rows record", k);
      }
      if (primal_only && r.dual) { return fail("dual record in a primal-only result at", k); }
      if (primal_only && !defined && !projections) {
        return fail("interval or forcing record in a primal-only result at", k);
      }
      for (int64_t n = 0; n < count; ++n) {
        double a = ps.record_coefficients[r.coefficients + n];
        if (!std::isfinite(a) || a == 0.) { return fail("coefficient of col_from_rows record", k); }
      }
      for (int32_t n = r.cols_begin; n < r.cols_end; ++n) {
        const gprs::record_col_t& c = ps.record_cols[n];
        if (!column_ok(c.col) || removed[c.col] || map[c.col].root != unset ||
            (n > r.cols_begin && ps.record_cols[n - 1].col >= c.col)) {
          return fail("column of col_from_rows record", k);
        }
        if (!(c.lower <= c.upper) || std::isnan(c.prefer) ||
            (interval &&
             !(std::isfinite(c.prefer) && c.lower <= c.prefer && c.prefer <= c.upper))) {
          return fail("bounds of col_from_rows record", k);
        }
        if (interval && ps.integer_at(c.col, k)) {
          return fail("integer column of interval record", k);
        }
        removed[c.col] = 1;
        --left;
      }
      const gprs::record_col_t& c = ps.record_cols[r.cols_begin];
      bool from_above             = forcing && forcing_from_above(ps, r);
      if (forcing && (from_above ? c.lower != -gprs::inf : c.upper != gprs::inf)) {
        return fail("bound of forcing record", k);
      }
      for (int32_t n = r.rows_begin; n < r.rows_end; ++n) {
        const gprs::saved_row_t& row = ps.saved_rows[n];
        if (!row_ok(row.row) || row_gone[row.row] || row.begin < entries_used ||
            row.begin > row.end || (size_t)row.end > ps.saved_entries.col.size() ||
            ps.saved_entries.value.size() != ps.saved_entries.col.size() ||
            !(row.lower <= row.upper)) {
          return fail("saved row of col_from_rows record", k);
        }
        entries_used = row.end;
        if (defined && !(row.lower == row.upper && std::isfinite(row.upper))) {
          return fail("side of defined record", k);
        }
        if (interval && !(one_sided ? std::isfinite(row.lower) || std::isfinite(row.upper)
                                    : std::isfinite(row.lower) && std::isfinite(row.upper))) {
          return fail("sides of interval record", k);
        }
        if (forcing) {
          // Every row bounds the column from the end that the first one does.
          double a   = ps.record_coefficients[r.coefficients + (n - r.rows_begin)];
          bool upper = from_above == (a > 0);
          if (!std::isfinite(upper ? row.upper : row.lower) ||
              !std::isinf(upper ? row.lower : row.upper)) {
            return fail("row of forcing record", k);
          }
        }
        for (int64_t e = row.begin; e < row.end; ++e) {
          int32_t col = ps.saved_entries.col[e];
          double a    = ps.saved_entries.value[e];
          if (!column_ok(col) || removed[col] || !std::isfinite(a) || a == 0.) {
            return fail("entry of col_from_rows record", k);
          }
        }
        if (!interval) { row_gone[row.row] = 1; }
        ++events_of[row.row + 1];
      }
    } else {
      return fail("unknown record kind at", k);
    }
  }
  if (fixed != ps.fixed_cols.size() || alias != ps.aliases.size() ||
      parallel != ps.parallel_rows.size() || addition != ps.row_additions.size() ||
      from_rows != ps.col_from_rows.size()) {
    return fail("records outside the order", 0);
  }
  if (left != 0) { return fail("columns neither left nor removed:", left); }

  // Variables and anchors of the reduced model.
  sys.reduced_cols = reduced.ncols;
  sys.reduced_rows = reduced.nrows;
  sys.lower        = reduced.col_lower;
  sys.upper        = reduced.col_upper;
  sys.origin       = ps.col_origin;
  sys.first_var.assign(ps.col_from_rows.size(), 0);
  sys.first_anchor.assign(ps.col_from_rows.size(), 0);
  sys.position.assign(ps.col_from_rows.size(), 0);
  int32_t saved_anchors = 0;
  for (size_t n = 0; n < ps.col_from_rows.size(); ++n) {
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    sys.first_var[n]                      = sys.lower.size();
    sys.first_anchor[n]                   = reduced.nrows + saved_anchors;
    saved_anchors += r.rows_end - r.rows_begin;
    for (int32_t c = r.cols_begin; c < r.cols_end; ++c) {
      sys.lower.push_back(ps.record_cols[c].lower);
      sys.upper.push_back(ps.record_cols[c].upper);
      sys.origin.push_back(ps.record_cols[c].col);
    }
  }
  sys.start.assign(reduced.row_start.begin(), reduced.row_start.end());
  sys.var   = reduced.col_index;
  sys.value = reduced.value;
  sys.low   = reduced.row_lower;
  sys.high  = reduced.row_upper;
  sys.row   = ps.row_origin;
  sys.reduced_anchor.assign(model.nrows, -1);
  for (int32_t k = 0; k < reduced.nrows; ++k) {
    sys.reduced_anchor[ps.row_origin[k]] = k;
  }

  // The maps, from the end of the stack. A saved row is composed with the maps as they are when
  // the pass reaches its record, which are the meanings of its columns at that position; only
  // then do the record's own columns get their variables. The anchors of a record are composed
  // into their places afterwards, since the pass meets the records in reverse.
  struct composed_row_t {
    std::vector<int32_t> var;
    std::vector<double> value;
    double low;
    double high;
    double side_parts;
    int32_t row;
  };
  std::vector<composed_row_t> composed(saved_anchors);
  std::vector<sum_t> sum(sys.nvars());
  std::vector<double> parts(sys.nvars(), 0.);
  std::vector<int32_t> touched;
  std::vector<double>& offset_parts = sys.offset_parts;
  offset_parts.assign(model.ncols, 0.);
  fixed     = ps.fixed_cols.size();
  alias     = ps.aliases.size();
  from_rows = ps.col_from_rows.size();
  for (size_t k = ps.order.size(); k-- > 0;) {
    if (ps.order[k] == gprs::record_kind_t::fixed_col) {
      const gprs::fixed_col_record_t& r = ps.fixed_cols[--fixed];
      map[r.col]                        = column_map_t{-1, 0., r.value};
      offset_parts[r.col]               = std::fabs(r.value);
    } else if (ps.order[k] == gprs::record_kind_t::alias) {
      const gprs::alias_record_t& r = ps.aliases[--alias];
      const column_map_t& rep       = map[r.rep];
      map[r.col] = column_map_t{rep.root, r.scale * rep.scale, r.scale * rep.offset + r.offset};
      offset_parts[r.col] = std::fabs(r.scale) * offset_parts[r.rep] + std::fabs(r.offset);
    } else if (ps.order[k] == gprs::record_kind_t::col_from_rows) {
      const gprs::col_from_rows_record_t& r = ps.col_from_rows[--from_rows];
      sys.position[from_rows]               = k;
      int32_t ncols                         = r.cols_end - r.cols_begin;
      for (int32_t n = r.rows_begin; n < r.rows_end; ++n) {
        const gprs::saved_row_t& row = ps.saved_rows[n];
        composed_row_t& out =
          composed[sys.first_anchor[from_rows] - reduced.nrows + (n - r.rows_begin)];
        sum_t constant;
        double side = 0.;
        if (std::isfinite(row.lower)) { side = std::fabs(row.lower); }
        if (std::isfinite(row.upper)) { side = std::fmax(side, std::fabs(row.upper)); }
        out.side_parts = side;
        for (int64_t e = row.begin; e < row.end; ++e) {
          const column_map_t& m = map[ps.saved_entries.col[e]];
          double a              = ps.saved_entries.value[e];
          constant.add(a, m.offset);
          out.side_parts += std::fabs(a) * offset_parts[ps.saved_entries.col[e]];
          if (m.root < 0) { continue; }
          if (parts[m.root] == 0.) { touched.push_back(m.root); }
          sum[m.root].add(a, m.scale);
          parts[m.root] += std::fabs(a * m.scale);
        }
        for (int32_t v : touched) {
          double a = sum[v].value();
          // What is left of terms that cancel is their rounding.
          if (std::fabs(a) > rounding_zero * parts[v]) {
            out.var.push_back(v);
            out.value.push_back(a);
          }
          sum[v]   = sum_t();
          parts[v] = 0.;
        }
        touched.clear();
        // The record's own columns: all of them in its one row, or its one column in each row.
        for (int32_t c = 0; c < (ncols > 1 ? ncols : 1); ++c) {
          int32_t at = ncols > 1 ? c : n - r.rows_begin;
          out.var.push_back(sys.first_var[from_rows] + c);
          out.value.push_back(ps.record_coefficients[r.coefficients + at]);
        }
        out.low  = row.lower - constant.value();
        out.high = row.upper - constant.value();
        out.row  = row.row;
      }
      for (int32_t c = 0; c < ncols; ++c) {
        map[ps.record_cols[r.cols_begin + c].col] =
          column_map_t{sys.first_var[from_rows] + c, 1., 0.};
      }
    }
  }
  sys.side_parts.assign(reduced.nrows, 0.);
  for (const composed_row_t& row : composed) {
    sys.var.insert(sys.var.end(), row.var.begin(), row.var.end());
    sys.value.insert(sys.value.end(), row.value.begin(), row.value.end());
    sys.start.push_back(sys.var.size());
    sys.low.push_back(row.low);
    sys.high.push_back(row.high);
    sys.side_parts.push_back(row.side_parts);
    sys.row.push_back(row.row);
  }

  // The events of the rows, in the order of the stack.
  for (int32_t i = 0; i < model.nrows; ++i) {
    events_of[i + 1] += events_of[i];
  }
  sys.first = events_of;
  sys.events.resize(events_of[model.nrows]);
  std::vector<int32_t> fill(events_of.begin(), events_of.end() - 1);
  parallel = addition = from_rows = 0;
  for (size_t k = 0; k < ps.order.size(); ++k) {
    if (ps.order[k] == gprs::record_kind_t::parallel_row) {
      const gprs::parallel_row_record_t& r = ps.parallel_rows[parallel++];
      sys.events[fill[r.removed_row]++]    = row_event_t{k, row_event_t::parallel, r.kept_row};
    } else if (ps.order[k] == gprs::record_kind_t::row_addition) {
      const gprs::row_addition_record_t& r = ps.row_additions[addition++];
      sys.events[fill[r.row]++]            = row_event_t{k, row_event_t::addition, r.equation_row};
    } else if (ps.order[k] == gprs::record_kind_t::col_from_rows) {
      const gprs::col_from_rows_record_t& r = ps.col_from_rows[from_rows];
      for (int32_t n = r.rows_begin; n < r.rows_end; ++n) {
        int32_t anchor                           = sys.first_anchor[from_rows] + (n - r.rows_begin);
        sys.events[fill[ps.saved_rows[n].row]++] = row_event_t{k, row_event_t::saved, anchor};
      }
      ++from_rows;
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

// The anchors that a function of a row at a position of the stack is proved from: the anchor of
// the row after that position, and the anchors of the equations that the records between the
// position and that anchor subtracted from it, each taken after its record, with the equations
// that were subtracted from those in turn. A row removed as a parallel row is, from then on,
// made of what its kept row is made of.
struct proof_rows_t {
  int32_t own = -1;
  // The own anchor belongs to another row: the kept row of a parallel record.
  bool other_row = false;
  // A record subtracted an equation other than the row itself.
  bool made_of_equations = false;
  // More than max_row_equations equations, or a walk above max_proof_visits: not tested.
  bool too_many = false;
  // Anchors of the equations, each once and none the own one.
  std::vector<int32_t> equations;

  struct visit_t {
    int32_t row;
    int64_t position;
  };
  std::vector<int32_t> rows;
  std::vector<visit_t> pending;
};

// Follows a row from a position of the stack: a saved row ends the walk with its anchor, a
// parallel record continues it with the kept row, and the equations subtracted on the way go
// to p.pending. Without a saved row the walk ends with the row of the reduced model, or -1.
// `last` gets the row that the anchor belongs to.
inline int32_t walk_row(const system_t& sys,
                        const gprs::postsolve_t& ps,
                        int32_t row,
                        int64_t position,
                        proof_rows_t& p,
                        int32_t& last)
{
  for (size_t hops = 0; hops <= ps.parallel_rows.size(); ++hops) {
    int32_t e = sys.first[row], end = sys.first[row + 1];
    while (e < end && (int64_t)sys.events[e].position <= position) {
      ++e;
    }
    bool moved = false;
    for (; e < end && !moved; ++e) {
      const row_event_t& event = sys.events[e];
      if (event.kind == row_event_t::saved) {
        last = row;
        return event.index;
      }
      if (event.kind == row_event_t::addition) {
        p.pending.push_back(proof_rows_t::visit_t{event.index, (int64_t)event.position});
        continue;
      }
      row      = event.index;
      position = event.position;
      moved    = true;
    }
    if (!moved) { break; }
  }
  last = row;
  return sys.reduced_anchor[row];
}

// Steps of the walks of one obligation; above it the obligation is not tested.
constexpr size_t max_proof_visits = 1024;

inline void proof_rows(
  const system_t& sys, const gprs::postsolve_t& ps, int32_t row, int64_t position, proof_rows_t& p)
{
  p.equations.clear();
  p.rows.clear();
  p.pending.clear();
  p.made_of_equations = false;
  int32_t last        = row;
  p.own               = walk_row(sys, ps, row, position, p, last);
  p.other_row         = last != row;
  for (size_t n = 0; n < p.pending.size(); ++n) {
    if (p.rows.size() > max_row_equations || p.pending.size() > max_proof_visits) { break; }
    proof_rows_t::visit_t visit = p.pending[n];
    bool again                  = false;
    for (size_t m = 0; m < n && !again; ++m) {
      again = p.pending[m].row == visit.row && p.pending[m].position == visit.position;
    }
    if (again) { continue; }
    if (visit.row != row) {
      p.made_of_equations = true;
      if (std::find(p.rows.begin(), p.rows.end(), visit.row) == p.rows.end()) {
        p.rows.push_back(visit.row);
      }
    }
    int32_t anchor = walk_row(sys, ps, visit.row, visit.position, p, last);
    if (anchor >= 0 && anchor != p.own &&
        std::find(p.equations.begin(), p.equations.end(), anchor) == p.equations.end()) {
      p.equations.push_back(anchor);
    }
  }
  p.too_many = p.rows.size() > max_row_equations || p.pending.size() > max_proof_visits;
}

// How an obligation was tested and what it gave.
struct proof_t {
  // 0: over the boxes; 1: against its anchor; 2: against anchors of equations as well.
  int32_t path = 0;
  // False for what the three paths count as not tested (kept_untested, added_untested).
  bool tested = true;
  // The violation, relative as the path defines it; 0 without one.
  double scaled = 0.;
  // Path 2: the largest multiplier of an equation.
  double multiplier = 0.;
};

// The three tests of a linear function of the check variables with sides. The caller fills the
// function into g and parts, with touched naming its variables, and prove() empties them.
struct prover_t {
  const system_t& sys;
  std::vector<double> g;
  std::vector<double> parts;
  std::vector<double> c;
  std::vector<int32_t> touched;
  std::vector<row_term_t> terms;
  std::vector<double> multipliers;
  // For a function that is made of equations: the variables of the test with the place of each
  // in it, and the entries of the anchors of the test, up to entry_end for each of them.
  std::vector<int32_t> cols;
  std::vector<int32_t> slot;
  std::vector<int32_t> entry_col;
  std::vector<double> entry_value;
  std::vector<size_t> entry_end;
  combined_t p;
  double kept_work = 0.;

  explicit prover_t(const system_t& system)
    : sys(system),
      g(system.nvars(), 0.),
      parts(system.nvars(), 0.),
      c(system.nvars(), 0.),
      slot(system.nvars(), -1),
      entry_end(max_row_equations + 1)
  {
  }

  void add_variable(int32_t v, double coefficient)
  {
    if (parts[v] == 0.) { touched.push_back(v); }
    g[v] += coefficient;
    parts[v] += std::fabs(coefficient);
  }

  // The term a * (original column) of the function.
  void add_column(const column_map_t& m, double a, sum_t& constant, double& largest)
  {
    constant.add(a, m.offset);
    raise_to(largest, std::fabs(a * m.offset));
    if (m.root >= 0) { add_variable(m.root, a * m.scale); }
  }

  void clear()
  {
    for (int32_t v : touched) {
      g[v]     = 0.;
      parts[v] = 0.;
    }
    touched.clear();
  }

  // lower <= constant + g * r <= upper at every point within the boxes that meets the anchors
  // of `with`; `largest` is the largest term of the constant, at least 1.
  proof_t prove(
    double lower, double upper, const sum_t& constant, double largest, const proof_rows_t& with)
  {
    proof_t out;
    int32_t k    = with.own;
    double below = -gprs::inf, above = -gprs::inf;
    if (with.too_many) {
      clear();
      out.path   = 2;
      out.tested = false;
      return out;
    }
    if (!with.equations.empty()) {
      out.path = 2;
      p.own    = k >= 0;
      p.rows   = p.own + with.equations.size();
      p.low.clear();
      p.high.clear();
      p.columns.clear();
      cols.clear();
      entry_col.clear();
      entry_value.clear();
      for (int32_t v : touched) {
        slot[v] = p.columns.size();
        cols.push_back(v);
        p.columns.push_back(combined_t::column_t{g[v], parts[v], sys.lower[v], sys.upper[v]});
        g[v]     = 0.;
        parts[v] = 0.;
      }
      touched.clear();
      // The entries of the anchors, one after the other; a variable that only they have gets
      // g = 0.
      for (int32_t n = 0; n < p.rows; ++n) {
        int32_t anchor = p.own ? (n == 0 ? k : with.equations[n - 1]) : with.equations[n];
        p.low.push_back(sys.low[anchor]);
        p.high.push_back(sys.high[anchor]);
        for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
          int32_t v = sys.var[e];
          if (slot[v] < 0) {
            slot[v] = p.columns.size();
            cols.push_back(v);
            p.columns.push_back(combined_t::column_t{0., 0., sys.lower[v], sys.upper[v]});
          }
          entry_col.push_back(slot[v]);
          entry_value.push_back(sys.value[e]);
        }
        entry_end[n] = entry_col.size();
      }
      for (int32_t v : cols) {
        slot[v] = -1;
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
        out.tested = false;
        return out;
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
      // A bound that proves the function does so whether a point meets the anchors or not; a
      // search that finds none says nothing when no point meets them.
      if (violation > 0. && !(scaled <= structure_tolerance) && no_common_point(p)) {
        out.tested = false;
        return out;
      }
      out.multiplier = std::fmax(most.multiplier, least.multiplier);
      out.scaled     = violation <= 0. ? 0. : scaled;
      return out;
    }

    if (k < 0) {
      // Each end has its own size: a huge bound at one end says nothing about the other.
      sum_t low = constant, high = constant;
      double largest_low = largest, largest_high = largest;
      bool low_infinite = false, high_infinite = false;
      for (int32_t v : touched) {
        double coefficient = g[v];
        bool cancelled     = std::fabs(coefficient) <= composed_zero * std::fmax(1., parts[v]);
        g[v]               = 0.;
        parts[v]           = 0.;
        if (cancelled) { continue; }
        double at_low  = coefficient > 0 ? sys.lower[v] : sys.upper[v];
        double at_high = coefficient > 0 ? sys.upper[v] : sys.lower[v];
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
      if (!(violation <= 0.)) {
        out.scaled = violation / (!(below <= above) ? largest_low : largest_high);
      }
      return out;
    }

    out.path = 1;
    for (int64_t e = sys.start[k]; e < sys.start[k + 1]; ++e) {
      int32_t v = sys.var[e];
      if (parts[v] == 0. && c[v] == 0.) { touched.push_back(v); }
      c[v] = sys.value[e];
    }
    terms.clear();
    multipliers.assign(1, 0.);
    sum_t least, most;
    bool least_infinite = false, most_infinite = false;
    for (int32_t v : touched) {
      row_term_t t{g[v], c[v], parts[v], sys.lower[v], sys.upper[v]};
      g[v]     = 0.;
      parts[v] = 0.;
      c[v]     = 0.;
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
    double low = sys.low[k], high = sys.high[k];
    bool empty =
      (!least_infinite &&
       least.value() - high > check_tolerance * std::fmax(1., std::fabs(high))) ||
      (!most_infinite && low - most.value() > check_tolerance * std::fmax(1., std::fabs(low)));
    double work = (double)different * (double)terms.size();
    if (empty || work > max_kept_row_work || kept_work + work > max_kept_work) {
      out.tested = false;
      return out;
    }
    kept_work += work;
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
    if (!(violation <= 0.)) {
      out.scaled = violation / std::fmax(largest, below > above ? size_below : size_above);
    }
    return out;
  }
};

// The largest and the smallest sum of the terms of the columns of an interval record over
// their bounds; an infinite one is flagged.
struct term_range_t {
  sum_t most;
  sum_t least;
  bool most_infinite  = false;
  bool least_infinite = false;
  // The sum of the terms at prefer.
  sum_t preferred;
};

inline term_range_t term_range(const gprs::postsolve_t& ps, const gprs::col_from_rows_record_t& r)
{
  term_range_t range;
  for (int32_t n = r.cols_begin; n < r.cols_end; ++n) {
    const gprs::record_col_t& col = ps.record_cols[n];
    double a                      = ps.record_coefficients[r.coefficients + (n - r.cols_begin)];
    double at_most = a > 0 ? col.upper : col.lower, at_least = a > 0 ? col.lower : col.upper;
    if (std::isinf(at_most)) {
      range.most_infinite = true;
    } else {
      range.most.add(a, at_most);
    }
    if (std::isinf(at_least)) {
      range.least_infinite = true;
    } else {
      range.least.add(a, at_least);
    }
    range.preferred.add(a, col.prefer);
  }
  return range;
}

// The columns part of the check: the boxes and the integral forms of defined records and the
// snapshots of interval records. The stack keeps its contract and `sys` holds its records
// composed.
inline void columns_check(const gprs::result_t& result,
                          const system_t& sys,
                          prover_t& prover,
                          structure_report_t& s)
{
  const gprs::postsolve_t& ps = result.postsolve;
  proof_rows_t with;
  for (size_t n = 0; n < ps.col_from_rows.size(); ++n) {
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    int32_t own                           = sys.first_var[n];
    int32_t anchor                        = sys.first_anchor[n];
    int32_t ncols                         = r.cols_end - r.cols_begin;
    if (r.kind == gprs::col_from_rows_kind_t::forcing) {
      ++s.forcing_records;
      continue;
    }
    if (r.kind == gprs::col_from_rows_kind_t::interval) {
      ++s.interval_records;
      for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
        if (sys.var[e] < own || sys.var[e] >= own + ncols) {
          prover.add_variable(sys.var[e], sys.value[e]);
        }
      }
      term_range_t range = term_range(ps, r);
      double lower       = range.most_infinite ? -gprs::inf : sys.low[anchor] - range.most.value();
      double upper = range.least_infinite ? gprs::inf : sys.high[anchor] - range.least.value();
      if (lower == -gprs::inf && upper == gprs::inf) {
        prover.clear();
        ++s.snapshot_tested;
        continue;
      }
      proof_rows(sys, ps, sys.row[anchor], sys.position[n], with);
      double size   = std::fmax(1., side_size(sys.low[anchor], sys.high[anchor]));
      proof_t proof = prover.prove(lower, upper, sum_t(), size, with);
      if (!proof.tested) {
        ++s.snapshot_untested;
        continue;
      }
      ++s.snapshot_tested;
      if (proof.scaled == 0.) { continue; }
      if (!(proof.scaled <= s.snapshot_worst)) { s.snapshot_worst_row = sys.row[anchor]; }
      raise_to(s.snapshot_worst, proof.scaled);
      s.snapshot_failures += !(proof.scaled <= structure_tolerance);
      continue;
    }

    ++s.defined_records;
    const gprs::record_col_t& col = ps.record_cols[r.cols_begin];
    double a                      = ps.record_coefficients[r.coefficients];
    double side                   = sys.low[anchor];
    // The range of the rest of the saved row over the boxes of its variables.
    sum_t most, least;
    bool most_infinite = false, least_infinite = false;
    double largest = std::fabs(side);
    for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
      int32_t v = sys.var[e];
      if (v == own) { continue; }
      double c        = sys.value[e];
      double at_most  = c > 0 ? sys.upper[v] : sys.lower[v];
      double at_least = c > 0 ? sys.lower[v] : sys.upper[v];
      if (std::isinf(at_most)) {
        most_infinite = true;
      } else {
        most.add(c, at_most);
        raise_to(largest, std::fabs(c * at_most));
      }
      if (std::isinf(at_least)) {
        least_infinite = true;
      } else {
        least.add(c, at_least);
        raise_to(largest, std::fabs(c * at_least));
      }
    }
    // The largest value of the column is (side - rest) / a at the smallest rest for a > 0.
    bool high_infinite = a > 0 ? least_infinite : most_infinite;
    bool low_infinite  = a > 0 ? most_infinite : least_infinite;
    double high        = (side - (a > 0 ? least.value() : most.value())) / a;
    double low         = (side - (a > 0 ? most.value() : least.value())) / a;
    auto end           = [&](double bound, bool infinite, double beyond) {
      if (std::isinf(bound)) { return; }
      ++s.box_tested;
      double violation = infinite ? gprs::inf : beyond;
      if (violation <= 0.) { return; }
      double scaled =
        violation / std::fmax(std::fmax(1., std::fabs(bound)), largest / std::fabs(a));
      if (!(scaled <= s.box_worst)) { s.box_worst_col = col.col; }
      raise_to(s.box_worst, scaled);
      s.box_failures += !(scaled <= structure_tolerance);
    };
    end(col.upper, high_infinite, high - col.upper);
    end(col.lower, low_infinite, col.lower - low);

    if (!ps.integer_at(col.col, sys.position[n])) { continue; }
    // Integral for the column: integer columns, integral ratios and side, and the drift of the
    // rounded ratios over the boxes within the tolerance, in the units of the row.
    const gprs::saved_row_t& row = ps.saved_rows[r.rows_begin];
    bool integral                = true;
    double drift                 = std::fabs(a * std::round(row.upper / a) - row.upper);
    for (int64_t e = row.begin; e < row.end && integral; ++e) {
      int32_t other         = ps.saved_entries.col[e];
      double ratio          = ps.saved_entries.value[e] / a;
      double delta          = std::fabs(ratio - std::round(ratio));
      const column_map_t& m = sys.map[other];
      integral              = ps.integer_at(other, sys.position[n]) && delta <= integral_ratio;
      if (delta == 0. || !integral) { continue; }
      double reach = std::fabs(m.offset);
      if (m.root >= 0) {
        reach += std::fabs(m.scale) *
                 std::fmax(std::fabs(sys.lower[m.root]), std::fabs(sys.upper[m.root]));
      }
      drift += std::fabs(a) * delta * reach;
    }
    if (!integral || !(drift <= structure_tolerance * std::fmax(1., std::fabs(row.upper)))) {
      if (s.integral_failures++ == 0) { s.first_integral_col = col.col; }
    }
  }
}

// The objective part of the check. The stack keeps its contract and `sys` holds its records
// composed (compose_records). O(columns + records + entries of the anchors).
inline void objective_check(const gprs::model_t& model,
                            const gprs::result_t& result,
                            const system_t& sys,
                            structure_report_t& s)
{
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;
  // The terms of the constant count with the absolute values that they are made of: the offset
  // of a column (system_t::offset_parts) and the side of a saved row (side_parts).
  const std::vector<double>& offset_parts = sys.offset_parts;

  std::vector<sum_t> coefficient(sys.nvars());
  std::vector<double> parts(sys.nvars(), 0.);
  sum_t constant;
  constant.add(model.objective_offset, 1.);
  constant.add(-reduced.objective_offset, 1.);
  double terms = std::fabs(model.objective_offset) + std::fabs(reduced.objective_offset);
  for (int32_t j = 0; j < model.ncols; ++j) {
    double cost = model.cost[j];
    if (cost == 0.) { continue; }
    const column_map_t& m = sys.map[j];
    constant.add(cost, m.offset);
    terms += std::fabs(cost) * offset_parts[j];
    if (m.root < 0) { continue; }
    coefficient[m.root].add(cost, m.scale);
    parts[m.root] += std::fabs(cost * m.scale);
  }
  for (int32_t k = 0; k < reduced.ncols; ++k) {
    coefficient[k].add(-reduced.cost[k], 1.);
    parts[k] += std::fabs(reduced.cost[k]);
  }
  // The variable of a defined record is what its saved row gives it. In the order of the stack
  // every variable of a saved row is one of the reduced model or of a later record.
  for (size_t n = 0; n < ps.col_from_rows.size(); ++n) {
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    if (r.kind != gprs::col_from_rows_kind_t::defined) { continue; }
    int32_t own    = sys.first_var[n];
    int32_t anchor = sys.first_anchor[n];
    double held    = coefficient[own].value();
    // What is left of a cost that an earlier elimination took away is the rounding of the
    // engine's own subtraction, and nothing to carry on.
    if (std::fabs(held) <= composed_zero * parts[own]) {
      coefficient[own] = sum_t();
      parts[own]       = 0.;
      continue;
    }
    double factor = held / ps.record_coefficients[r.coefficients];
    for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
      int32_t v = sys.var[e];
      if (v == own) { continue; }
      coefficient[v].add(-factor, sys.value[e]);
      parts[v] += std::fabs(factor * sys.value[e]);
    }
    constant.add(factor, sys.low[anchor]);
    terms += std::fabs(factor) * sys.side_parts[anchor];
    coefficient[own] = sum_t();
    parts[own]       = 0.;
  }
  for (int32_t v = 0; v < sys.nvars(); ++v) {
    double difference = std::fabs(coefficient[v].value());
    if (difference <= underflow_size) { continue; }
    double relative = difference / parts[v];
    if (!(relative <= s.objective_worst)) { s.objective_worst_col = sys.origin[v]; }
    raise_to(s.objective_worst, relative);
    s.objective_failures += !(relative <= composed_zero);
  }
  double difference = std::fabs(constant.value());
  if (!(difference <= underflow_size)) {
    s.objective_constant = difference / terms;
    s.objective_failures += !(s.objective_constant <= identity_tolerance);
  }
}

// The values of the check variables at one point of the reduced model, by the rules of
// postsolve.hpp applied to the anchors: the second implementation of the undo of the
// col_from_rows records. size is what a difference of a value is relative to.
struct replay_t {
  std::vector<double> value;
  std::vector<double> size;
};

// The rest of an anchor at the values of the replay: the sum of its terms outside the variables
// [own, own + ncols) of its record. `largest` gets the largest of those terms.
inline double anchor_rest(const system_t& sys,
                          const replay_t& at,
                          int32_t anchor,
                          int32_t own,
                          int32_t ncols,
                          double& largest)
{
  sum_t rest;
  for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
    int32_t v = sys.var[e];
    if (v >= own && v < own + ncols) { continue; }
    rest.add(sys.value[e], at.value[v]);
    raise_to(largest, std::fabs(sys.value[e]) * at.size[v]);
  }
  return rest.value();
}

// `engine` is the point that postsolve() gave for `point`, or null: an integer forcing column
// whose value hangs on the last bit of its sum takes the value of the engine (the postsolve
// part at the head of this file).
inline void replay(const gprs::result_t& result,
                   const system_t& sys,
                   const std::vector<double>& point,
                   replay_t& at,
                   const std::vector<double>* engine = nullptr)
{
  const gprs::postsolve_t& ps = result.postsolve;
  at.value.assign(sys.nvars(), 0.);
  at.size.assign(sys.nvars(), 0.);
  for (int32_t k = 0; k < sys.reduced_cols; ++k) {
    bool integer = ps.integer_at(ps.col_origin[k], ps.order.size());
    at.value[k]  = integer ? std::round(point[k]) : point[k];
    at.size[k]   = std::fabs(at.value[k]);
  }
  for (size_t n = ps.col_from_rows.size(); n-- > 0;) {
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    int32_t own = sys.first_var[n], anchor = sys.first_anchor[n];
    int32_t ncols                 = r.cols_end - r.cols_begin;
    const gprs::record_col_t& col = ps.record_cols[r.cols_begin];
    if (r.kind == gprs::col_from_rows_kind_t::defined) {
      double a       = ps.record_coefficients[r.coefficients];
      double largest = std::fabs(sys.low[anchor]);
      double value   = (sys.low[anchor] - anchor_rest(sys, at, anchor, own, 1, largest)) / a;
      at.value[own]  = ps.integer_at(col.col, sys.position[n]) ? std::round(value) : value;
      at.size[own]   = std::fmax(std::fabs(at.value[own]), largest / std::fabs(a));
    } else if (r.kind == gprs::col_from_rows_kind_t::interval) {
      double largest = side_size(sys.low[anchor], sys.high[anchor]);
      double rest    = anchor_rest(sys, at, anchor, own, ncols, largest);
      sum_t activity;
      activity.add(rest, 1.);
      for (int32_t c = 0; c < ncols; ++c) {
        const gprs::record_col_t& taken = ps.record_cols[r.cols_begin + c];
        double a                        = ps.record_coefficients[r.coefficients + c];
        at.value[own + c]               = taken.prefer;
        activity.add(a, taken.prefer);
        raise_to(largest, std::fabs(a * taken.prefer));
      }
      double below = sys.low[anchor] - activity.value();
      double above = activity.value() - sys.high[anchor];
      bool raise   = below > 0;
      double side  = raise ? sys.low[anchor] : sys.high[anchor];
      double need  = raise ? below : above;
      for (int32_t k = 0; k < ncols && need > 0; ++k) {
        int32_t c                       = ps.interval_descending ? ncols - 1 - k : k;
        const gprs::record_col_t& taken = ps.record_cols[r.cols_begin + c];
        double a                        = ps.record_coefficients[r.coefficients + c];
        bool up                         = (a > 0) == raise;
        double target                   = up ? taken.upper : taken.lower;
        double room                     = std::fabs(a) * std::fabs(target - taken.prefer);
        // The row without the term of this column: its value is a quotient, not a step.
        activity.add(-a, taken.prefer);
        if (room <= need) {
          at.value[own + c] = target;
          activity.add(a, target);
          need = raise ? side - activity.value() : activity.value() - side;
        } else {
          double value      = (side - activity.value()) / a;
          at.value[own + c] = up ? std::fmin(std::fmax(value, taken.prefer), target)
                                 : std::fmax(std::fmin(value, taken.prefer), target);
          need              = 0.;
        }
      }
      for (int32_t c = 0; c < ncols; ++c) {
        double a         = ps.record_coefficients[r.coefficients + c];
        at.size[own + c] = std::fmax(std::fabs(at.value[own + c]), largest / std::fabs(a));
      }
    } else {
      bool from_above = forcing_from_above(ps, r);
      double value    = from_above ? col.upper : col.lower;
      double size     = 0.;
      // What the value of the row that gives it can be off by; none for the bound. `rounding`
      // is the band of the undo, `band` what two summations can differ by.
      double rounding = 0., band = 0.;
      for (int32_t j = 0; j < r.rows_end - r.rows_begin; ++j) {
        double a       = ps.record_coefficients[r.coefficients + j];
        double side    = from_above == (a > 0) ? sys.high[anchor + j] : sys.low[anchor + j];
        double largest = std::fabs(side);
        double limit   = (side - anchor_rest(sys, at, anchor + j, own, 1, largest)) / a;
        raise_to(size, largest / std::fabs(a));
        if (from_above ? limit < value : limit > value) {
          double terms = sys.side_parts[anchor + j];
          for (int64_t e = sys.start[anchor + j]; e < sys.start[anchor + j + 1]; ++e) {
            if (sys.var[e] != own) { terms += std::fabs(sys.value[e] * at.value[sys.var[e]]); }
          }
          value    = limit;
          rounding = ps.forcing_rounding_units * 0x1p-53 * terms / std::fabs(a);
          // The two computations take the terms of the band from other sums as well.
          band = std::fmax(forcing_rounding * terms / std::fabs(a), rounding);
        }
      }
      if (ps.integer_at(col.col, sys.position[n])) {
        double sum    = from_above ? value + rounding : value - rounding;
        value         = from_above ? std::floor(sum) : std::ceil(sum);
        bool own_band = ps.forcing_rounding_units * 0x1p-53 == forcing_rounding;
        if (engine && !own_band && std::fabs((*engine)[col.col] - value) == 1.) {
          // The integer at which the rounding of the sum turns from one value to the other.
          double other = (*engine)[col.col];
          double turn  = from_above ? std::fmax(value, other) : std::fmin(value, other);
          if (std::fabs(sum - turn) <= band) { value = other; }
        }
      }
      at.value[own] = value;
      at.size[own]  = std::fmax(std::fabs(value), size);
    }
  }
}

// The postsolve part of the check at one point of the reduced model.
inline void postsolve_at(const gprs::model_t& model,
                         const gprs::result_t& result,
                         const system_t& sys,
                         const std::vector<double>& point,
                         replay_t& at,
                         std::vector<double>& z,
                         structure_report_t& s)
{
  const gprs::postsolve_t& ps = result.postsolve;
  ps.postsolve(point, z);
  replay(result, sys, point, at, &z);
  auto failure = [&](double scaled, int32_t col) {
    if (!(scaled <= s.postsolve_worst)) { s.postsolve_worst_col = col; }
    raise_to(s.postsolve_worst, scaled);
    ++s.postsolve_failures;
  };
  for (int32_t j = 0; j < model.ncols; ++j) {
    const column_map_t& m = sys.map[j];
    double term           = m.root < 0 ? 0. : m.scale * at.value[m.root];
    double reach          = m.root < 0 ? 0. : std::fabs(m.scale) * at.size[m.root];
    double size           = std::fmax(1., std::fmax(reach, std::fabs(m.offset)));
    double difference     = std::fabs(z[j] - (term + m.offset)) / size;
    if (!(difference <= s.postsolve_worst)) { s.postsolve_worst_col = j; }
    raise_to(s.postsolve_worst, difference);
    s.postsolve_failures += !(difference <= composed_tolerance);
  }
  // The saved rows and the boxes at the point that the engine postsolved, read from the records
  // as they are.
  for (const gprs::col_from_rows_record_t& r : ps.col_from_rows) {
    const gprs::record_col_t& col = ps.record_cols[r.cols_begin];
    bool interval                 = r.kind == gprs::col_from_rows_kind_t::interval;
    bool forcing                  = r.kind == gprs::col_from_rows_kind_t::forcing;
    bool from_above               = forcing && forcing_from_above(ps, r);
    for (int32_t n = r.rows_begin; n < r.rows_end; ++n) {
      const gprs::saved_row_t& row = ps.saved_rows[n];
      sum_t activity;
      double largest = 1.;
      for (int64_t e = row.begin; e < row.end; ++e) {
        double a = ps.saved_entries.value[e], v = z[ps.saved_entries.col[e]];
        activity.add(a, v);
        raise_to(largest, std::fabs(a * v));
      }
      double rest  = activity.value();
      double lower = row.lower, upper = row.upper;
      if (interval) {
        // The row need hold only where the rest lies inside the sides of the snapshot.
        term_range_t range = term_range(ps, r);
        double low         = range.most_infinite ? -gprs::inf : lower - range.most.value();
        double high        = range.least_infinite ? gprs::inf : upper - range.least.value();
        if (!(rest >= low && rest <= high)) { continue; }
        for (int32_t c = r.cols_begin; c < r.cols_end; ++c) {
          double a = ps.record_coefficients[r.coefficients + (c - r.cols_begin)];
          double v = z[ps.record_cols[c].col];
          activity.add(a, v);
          raise_to(largest, std::fabs(a * v));
        }
      } else {
        double a = ps.record_coefficients[r.coefficients + (forcing ? n - r.rows_begin : 0)];
        activity.add(a, z[col.col]);
        raise_to(largest, std::fabs(a * z[col.col]));
        if (forcing) {
          // Only the side that bounds the column is a side of the record.
          bool upper_side = from_above == (a > 0);
          lower           = upper_side ? -gprs::inf : lower;
          upper           = upper_side ? upper : gprs::inf;
        }
      }
      double value     = activity.value();
      double violation = std::isnan(value) ? value : std::fmax(lower - value, value - upper);
      double size = std::fmax(largest, lower > -gprs::inf ? std::fabs(lower) : std::fabs(upper));
      if (!(violation / size <= structure_tolerance)) { failure(violation / size, col.col); }
    }
    if (!interval && !forcing) { continue; }
    for (int32_t c = r.cols_begin; c < r.cols_end; ++c) {
      const gprs::record_col_t& taken = ps.record_cols[c];
      double v                        = z[taken.col];
      double violation = std::isnan(v) ? v : std::fmax(taken.lower - v, v - taken.upper);
      if (violation <= 0.) { continue; }
      double bound  = v < taken.lower ? taken.lower : taken.upper;
      double scaled = violation / std::fmax(1., std::fabs(bound));
      if (!(scaled <= structure_tolerance)) { failure(scaled, taken.col); }
    }
  }
}

// The postsolve part of the check: the points of the reduced model and postsolve_at for each.
inline void postsolve_check(const gprs::model_t& model,
                            const gprs::result_t& result,
                            const system_t& sys,
                            structure_report_t& s)
{
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;
  std::vector<double> point(reduced.ncols), z;
  replay_t at;
  // 0: 1 moved into the bounds; 1: the lower bound; 2: the upper bound; 3: between them. An
  // infinite bound is replaced by the other one less or plus 1, or by -1 and 1.
  for (int32_t variant = 0; variant < 4; ++variant) {
    for (int32_t k = 0; k < reduced.ncols; ++k) {
      double lower = reduced.col_lower[k], upper = reduced.col_upper[k];
      double low  = lower > -gprs::inf ? lower : upper < gprs::inf ? upper - 1. : -1.;
      double high = upper < gprs::inf ? upper : lower > -gprs::inf ? lower + 1. : 1.;
      double v    = variant == 1 ? low : variant == 2 ? high : 0.5 * low + 0.5 * high;
      if (variant == 0) {
        v = 1.;
        if (v < lower) { v = lower; }
        if (v > upper) { v = upper; }
      }
      point[k] = ps.integer_at(ps.col_origin[k], ps.order.size()) ? std::round(v) : v;
    }
    postsolve_at(model, result, sys, point, at, z, s);
  }
  // For some interval records a point at which the undo has to move the columns half of the way
  // they can go: from the point between the bounds, one reduced column of the saved row is
  // moved so that the rest of the row asks for it. It need not lie within the bounds.
  std::vector<size_t> interval;
  for (size_t n = 0; n < ps.col_from_rows.size(); ++n) {
    if (ps.col_from_rows[n].kind == gprs::col_from_rows_kind_t::interval) { interval.push_back(n); }
  }
  size_t count = interval.size() < max_replay_points ? interval.size() : max_replay_points;
  std::vector<double> moved;
  for (size_t pick = 0; pick < count; ++pick) {
    size_t n                              = interval[pick * interval.size() / count];
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    int32_t own = sys.first_var[n], anchor = sys.first_anchor[n];
    replay(result, sys, point, at);
    double largest     = 1.;
    double rest        = anchor_rest(sys, at, anchor, own, r.cols_end - r.cols_begin, largest);
    term_range_t range = term_range(ps, r);
    double preferred   = range.preferred.value();
    double up          = range.most_infinite ? gprs::inf : range.most.value() - preferred;
    double down        = range.least_infinite ? gprs::inf : preferred - range.least.value();
    double target;
    if (up > 0 && std::isfinite(sys.low[anchor])) {
      target = sys.low[anchor] - preferred - (std::isfinite(up) ? 0.5 * up : 1.);
    } else if (down > 0 && std::isfinite(sys.high[anchor])) {
      target = sys.high[anchor] - preferred + (std::isfinite(down) ? 0.5 * down : 1.);
    } else {
      continue;
    }
    int64_t best = -1;
    for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
      if (sys.var[e] < sys.reduced_cols &&
          (best < 0 || std::fabs(sys.value[e]) > std::fabs(sys.value[best]))) {
        best = e;
      }
    }
    if (best < 0) { continue; }
    moved = point;
    moved[sys.var[best]] += (target - rest) / sys.value[best];
    postsolve_at(model, result, sys, moved, at, z, s);
  }
}

// The check of one result of a presolve in the given mode (options_t::primal_only);
// `projections` and `one_sided` as in compose_records.
inline structure_report_t structure_check(const gprs::model_t& model,
                                          const gprs::result_t& result,
                                          bool primal_only,
                                          bool projections = false,
                                          bool one_sided   = false)
{
  structure_report_t s;
  s.ran = true;
  system_t sys;
  if (!compose_records(model, result, primal_only, projections, one_sided, sys, s.stack_error)) {
    return s;
  }
  s.composed                   = true;
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;

  for (int32_t k = 0; k < reduced.ncols; ++k) {
    int32_t col = ps.col_origin[k];
    if (model.col_type[col] && !reduced.col_type[k]) {
      if (s.lost_integers++ == 0) { s.first_lost_integer = col; }
    }
  }

  objective_check(model, result, sys, s);

  for (int32_t j = 0; j < model.ncols; ++j) {
    const column_map_t& m = sys.map[j];
    double low = m.offset, high = m.offset;
    if (m.root >= 0) {
      double a = m.scale * sys.lower[m.root] + m.offset;
      double b = m.scale * sys.upper[m.root] + m.offset;
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

  prover_t prover(sys);
  // The snapshots first: an anchor of an interval record holds only as far as its snapshot.
  columns_check(result, sys, prover, s);

  proof_rows_t with;
  for (int32_t i = 0; i < model.nrows; ++i) {
    sum_t constant;
    double largest = 1.;
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      prover.add_column(sys.map[model.col_index[e]], model.value[e], constant, largest);
    }
    proof_rows(sys, ps, i, -1, with);
    s.added_rows += with.made_of_equations;
    proof_t proof = prover.prove(model.row_lower[i], model.row_upper[i], constant, largest, with);
    if (proof.path == 2) {
      if (!proof.tested) {
        ++s.added_untested;
        continue;
      }
      ++s.added_tested;
      raise_to(s.added_multiplier, proof.multiplier);
      if (proof.scaled == 0.) { continue; }
      if (!(proof.scaled <= s.added_worst)) { s.added_worst_row = i; }
      raise_to(s.added_worst, proof.scaled);
      s.added_failures += !(proof.scaled <= structure_tolerance);
    } else if (proof.path == 0) {
      ++s.rows_tested;
      if (proof.scaled == 0.) { continue; }
      if (!(proof.scaled <= s.row_worst)) { s.row_worst_row = i; }
      raise_to(s.row_worst, proof.scaled);
      s.row_failures += !(proof.scaled <= structure_tolerance);
    } else {
      if (!proof.tested) {
        ++s.kept_untested;
        continue;
      }
      ++s.kept_tested;
      s.kept_parallel += with.other_row;
      if (proof.scaled == 0.) { continue; }
      if (!(proof.scaled <= s.kept_worst)) { s.kept_worst_row = i; }
      raise_to(s.kept_worst, proof.scaled);
      s.kept_failures += !(proof.scaled <= structure_tolerance);
    }
  }

  postsolve_check(model, result, sys, s);
  return s;
}

// The check of a result of presolve(model, options).
inline structure_report_t structure_check(const gprs::model_t& model,
                                          const gprs::result_t& result,
                                          const gprs::options_t& options)
{
  return structure_check(
    model, result, options.primal_only, options.primal_projections, options.zero_cost_one_sided);
}

}  // namespace gprs_check
