/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// "The row gives its column k an integer value at every integer value of its other columns":
// the row divided by the coefficient a of k has integral coefficients and an integral side.
// One predicate for every rule that needs it: the free singleton column of integer type
// (rules.hpp (h)), the edge of the alias forest between two integer columns, and the rules that
// follow (implied integers, implied equations with an integer source, column scaling).
// It reads explicit bounds, coefficients and types of the state as it is.

#include <gprs/numerics.hpp>
#include <gprs/state.hpp>

#include <cmath>

namespace gprs {

// The row divided by a, column after column.
// integral: each ratio b / a so far lies within tol.zero of an integer n with
// |a * n - b| <= tol.zero, and the side divided by a within tol.zero of an integer m with
// |a * m - side| <= tol.feasibility. integral_for also asks every other free column of the row to
// be integer; a caller that feeds the terms itself tests the types.
// drift: what the rounded ratios move the value that the row gives column k by, over the bounds
// of the other columns: the sum of |b / a - n| * max(|lower|, |upper|), in the units of column
// k; infinite when a ratio that is not an integer exactly belongs to a column without a finite
// bound. It is the drift of row tightening (rules.hpp (c)) measured in the row divided by a,
// where that rule measures it in the row times its scale.
// side_error: |a * m - side|, what the rounded side moves the row by, in the units of the row.
struct integral_form_t {
  bool integral     = true;
  double drift      = 0.;
  double side_error = 0.;
};

// One other free column of the row, an integer column with coefficient b and the given bounds.
inline void integral_term(
  const tolerances_t& tol, double a, double b, double lower, double upper, integral_form_t& f)
{
  double ratio   = b / a;
  double nearest = std::round(ratio);
  double delta   = std::fabs(ratio - nearest);
  if (!(delta <= tol.zero) || !(std::fabs(a * nearest - b) <= tol.zero)) {
    f.integral = false;
    return;
  }
  if (delta != 0.) { f.drift += delta * std::fmax(std::fabs(lower), std::fabs(upper)); }
}

// The side of the row less the terms of its fixed columns.
inline void integral_side(const tolerances_t& tol, double a, double side, integral_form_t& f)
{
  double ratio   = side / a;
  double nearest = std::round(ratio);
  f.side_error   = std::fabs(a * nearest - side);
  if (!(std::fabs(ratio - nearest) <= tol.zero) || !(f.side_error <= tol.feasibility)) {
    f.integral = false;
  }
}

// The units in which a drift is bounded.
// row: the row may be off by the drift times |a|. Enough where column k leaves the model with
// the row (a substituted column, whose value the postsolve rounds: the row then holds within
// that), or moves only in a direction in which nothing holds it.
// column: column k itself may be off an integer by the drift, as well. Needed where k stays in
// the model as an integer column and its bounds are rounded.
enum class drift_units_t : uint8_t { row, column };

// The drift stays below `share` of the feasibility tolerance in the given units: in the units of
// the row, |a| * drift <= share * tol.feasibility, and in those of the column also
// drift <= share * tol.feasibility, which together are
// |a| * drift <= share * tol.feasibility * min(1, |a|).
// The rounded side moves the same row: with the integers that the form stands for the row is
// off by up to side_error + |a| * drift, which must stay within tol.feasibility as well. Each
// alone within its limit would let a row be off by 1 + share tolerances.
inline bool drift_within(
  const tolerances_t& tol, double share, double a, const integral_form_t& f, drift_units_t units)
{
  double size  = std::fabs(a);
  double limit = share * tol.feasibility;
  if (units == drift_units_t::column && size < 1.) { limit *= size; }
  return size * f.drift <= limit && f.side_error + size * f.drift <= tol.feasibility;
}

// The form of the live row for the column of its live entry e, with `side` as the side of the
// free part: one scan of the row. O(csr.used[row]).
inline integral_form_t integral_for(
  const state_t& st, const tolerances_t& tol, int32_t row, int64_t e, double side)
{
  const line_store_t& csr = st.csr;
  double a                = csr.value[e];
  integral_form_t f;
  integral_side(tol, a, side, f);
  for (int64_t s = csr.begin(row); s < csr.end(row) && f.integral; ++s) {
    int32_t col = csr.index[s];
    if (s == e || !csr.live(s) || !st.is_live(col)) { continue; }
    if (!st.is_integer(col)) {
      f.integral = false;
    } else {
      integral_term(tol, a, csr.value[s], st.col_lower[col], st.col_upper[col], f);
    }
  }
  return f;
}

}  // namespace gprs
