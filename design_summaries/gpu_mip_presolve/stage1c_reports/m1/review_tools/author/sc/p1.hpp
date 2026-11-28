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

