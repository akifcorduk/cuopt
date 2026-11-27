/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The bound that one row implies for one of its columns over the explicit bounds of its other
// columns, in O(1) from the stored activities. Shared by bound propagation, which writes such
// bounds, and by the column rules, which test whether a bound of a column is implied by a row.
// Every function reads the state as it is and no recorded implied bound.

#include <gprs/assert.hpp>
#include <gprs/numerics.hpp>
#include <gprs/state.hpp>

#include <cmath>

namespace gprs {

// A candidate is rounded twice, in the residual and in the division, by up to this many rounding
// units of its size in all.
constexpr double candidate_rounding_units = 4.;

inline double bound_rounding(double bound)
{
  return candidate_rounding_units * rounding_unit * std::fabs(bound);
}

// What two candidates of one bound can differ by: each is moved outward by its rounding and by
// the error bound of its activity, which the guard of candidate() keeps below a share of the
// tolerance and which is another one in every presolve. A bound of a column counts as implied
// by a row when the candidate of the row is within this of it, so that the test gives one
// answer before and after the candidate has become the explicit bound.
inline double implied_tolerance(const tolerances_t& tol, double bound)
{
  return 2. * (bound_rounding(bound) + tol.activity_error_share * tol.feasibility);
}

// The activity of the row has no infinite term outside entry e: its rest, the activity without
// the term of the column of e, is finite.
inline bool rest_finite(const activity_t& act, int32_t row, int64_t e)
{
  int32_t infinite = act.inf_count[row];
  return infinite == 0 || (infinite == 1 && act.inf_slot[row] == e);
}

// The bound that one side of the row implies for the column of entry e: the upper side with
// the minimum activity (upper_side) or the lower side with the maximum activity. It is the same
// for every entry of the row: a candidate changes the bound of its column that this activity
// does not hold, and a column fixed by its candidate is fixed at the bound that the activity
// holds.
struct candidate_t {
  double bound;
  // What it was computed from: bound = dd_residual(side, rest) / a, with the error bound of rest.
  double side;
  dd_t rest;
  double a;
  double error;
};

// The candidate of a finite side of the row for the column of its live entry e, for a caller
// that knows the rest of that activity to be finite (rest_finite): bound propagation, which
// reads the infinite terms of a row before it visits an entry. False without a candidate: the
// error bound of the rest exceeds tol.activity_error_share * tol.feasibility * |a|, which sets
// `guarded` (a caller whose activity is not fresh sums it from scratch and asks again); or the
// division overflows, which says nothing about the column. O(1).
// The innermost step of bound propagation. It is written out, and takes the tolerances of its
// caller, because composed of row_rest and with the test of candidate() it made that family 4%
// slower on the largest local instance.
inline bool candidate_of_finite_rest(const state_t& st,
                                     const tolerances_t& tol,
                                     int32_t row,
                                     int64_t e,
                                     bool upper_side,
                                     candidate_t& k,
                                     bool& guarded)
{
  const activity_t& act = upper_side ? st.min_act : st.max_act;
  int32_t infinite      = act.inf_count[row];
  gprs_assert(infinite == 0 || (infinite == 1 && act.inf_slot[row] == e));
  int32_t col      = st.csr.index[e];
  k.a              = st.csr.value[e];
  k.side           = upper_side ? st.row_upper[row] : st.row_lower[row];
  k.rest           = dd_t{act.hi[row], act.lo[row]};
  uint32_t updates = act.updates[row];
  if (infinite == 0) {
    double own = upper_side ? min_activity_bound(k.a, st.col_lower[col], st.col_upper[col])
                            : max_activity_bound(k.a, st.col_lower[col], st.col_upper[col]);
    gprs_assert(!is_infinite_term(k.a, own, tol));
    dd_sub_product(k.rest, k.a, own);
    ++updates;
  }
  k.error = activity_error(updates, act.max_term[row]);
  if (k.error > tol.activity_error_share * tol.feasibility * std::fabs(k.a)) {
    guarded = true;
    return false;
  }
  k.bound = dd_residual(k.side, k.rest) / k.a;
  return std::isfinite(k.bound);
}

// The candidate of a finite side of the row for the column of any live entry e: false as well
// when the rest of the activity holds an infinite term. For the column rules, which ask of one
// row whether it implies a bound of one column. O(1).
inline bool candidate(const state_t& st,
                      const tolerances_t& tol,
                      int32_t row,
                      int64_t e,
                      bool upper_side,
                      candidate_t& k,
                      bool& guarded)
{
  if (!rest_finite(upper_side ? st.min_act : st.max_act, row, e)) { return false; }
  return candidate_of_finite_rest(st, tol, row, e, upper_side, k, guarded);
}

// One activity of a row without the term of the column of its live entry e, as a compensated
// sum with its error bound: candidate_t::rest and candidate_t::error, for a caller that needs
// the range of the row without the column and no bound.
struct row_rest_t {
  dd_t sum;
  double error;
};

// The minimum activity of the row (minimum) or its maximum activity, over the explicit bounds
// and with the terms of the fixed columns, less the term of the column of entry e. False when
// that rest holds an infinite term (rest_finite). O(1).
inline bool row_rest(const state_t& st, int32_t row, int64_t e, bool minimum, row_rest_t& rest)
{
  const activity_t& act = minimum ? st.min_act : st.max_act;
  if (!rest_finite(act, row, e)) { return false; }
  rest.sum         = dd_t{act.hi[row], act.lo[row]};
  uint32_t updates = act.updates[row];
  if (act.inf_count[row] == 0) {
    int32_t col = st.csr.index[e];
    double a    = st.csr.value[e];
    double own  = minimum ? min_activity_bound(a, st.col_lower[col], st.col_upper[col])
                          : max_activity_bound(a, st.col_lower[col], st.col_upper[col]);
    gprs_assert(!is_infinite_term(a, own, st.tol));
    dd_sub_product(rest.sum, a, own);
    ++updates;
  }
  rest.error = activity_error(updates, act.max_term[row]);
  return true;
}

// What the true quotient can lie beyond the candidate by: nothing when the residual and the
// division are exact for the stored sum (a candidate of 8 is 8, whatever the history of the
// sum), else their rounding and the error bound of the activity. A bound of a continuous column
// is the candidate moved outward by it, so that the bound holds for every point of the row: a
// bound that is a rounding step too tight cuts a point off, and a row with a large coefficient
// of the column carries that on, multiplied (131072 twice turns one step into 4e-6).
inline double candidate_slack(const candidate_t& k)
{
  // dd_residual(k.side, k.rest), with what its two additions round away.
  double carry, tail_rounding, residual_rounding;
  double difference = two_sum(k.side, -k.rest.hi, carry);
  double tail       = two_sum(carry, -k.rest.lo, tail_rounding);
  double residual   = two_sum(difference, tail, residual_rounding);
  if (tail_rounding == 0. && residual_rounding == 0. && std::fma(k.bound, k.a, -residual) == 0.) {
    return 0.;
  }
  return bound_rounding(k.bound) +
         (k.error + std::fabs(tail_rounding) + std::fabs(residual_rounding)) / std::fabs(k.a);
}

}  // namespace gprs
