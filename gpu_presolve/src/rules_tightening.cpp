/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <cmath>
#include <numeric>
#include <vector>

namespace gprs {

struct row_tightening_t::work_t {
  std::vector<int32_t> rows;
};

row_tightening_t::row_tightening_t() : work(new work_t) {}

row_tightening_t::~row_tightening_t() = default;

namespace {

// Largest factor that a row is multiplied with to make its coefficients integral.
constexpr int64_t max_integral_scale = 1000;
// Scaled coefficients above this are not taken as integers: a * (scale / gcd) must round to the
// integer that a * scale divided by the gcd is.
constexpr double max_integral_coefficient = 1e12;
// Share of the feasibility tolerance by which rounding the scaled coefficients to integers may
// move the activity of the row over the bounds of its columns.
constexpr double coefficient_rounding_share = 0.1;
// A side above this, in the row scaled to integers, is not rounded: below it the division of
// the rounded side by the gcd, both integers, gives a quotient whose floor and ceiling are exact.
constexpr double max_exact_side = 0x1p52;

struct context_t {
  state_t& st;
  work_budget_t& budget;
  const tolerances_t& tol;
  rule_result_t& result;

  void finding()
  {
    ++result.findings;
    budget.found();
  }
};

// Denominator q <= limit of a continued-fraction convergent of x >= 0 with x * q integral
// within eps, or 0 when there is none.
int64_t integral_denominator(double x, double eps, int64_t limit)
{
  double rest      = x - std::floor(x);
  int64_t previous = 0, q = 1;
  while (true) {
    double scaled = x * q;
    if (std::fabs(scaled - std::round(scaled)) <= eps) { return q; }
    rest         = 1. / rest;
    double digit = std::floor(rest);
    rest -= digit;
    // Also the end of an expansion whose last convergent missed eps by rounding (rest was 0).
    if (!(digit <= limit)) { return 0; }
    int64_t next = (int64_t)digit * q + previous;
    if (next > limit) { return 0; }
    previous = q;
    q        = next;
  }
}

// A side or a fixed term of this size is stored, as one double, with an error far below the
// feasibility tolerance (2^-53 / side_error of it). Above it the error can exceed the tolerance,
// and an edit that writes such a value changes the row by more than it proves.
bool holds_tolerance(double size, const tolerances_t& tol)
{
  return size * tol.side_error <= tol.feasibility;
}

struct row_scan_t {
  int32_t free_entries = 0;
  bool all_integer     = true;
  bool has_fixed       = false;
  dd_t fixed_terms;
  double fixed_size = 0.;  // sum of |a * value| over the fixed entries
};

row_scan_t scan_row(const state_t& st, int32_t row)
{
  row_scan_t scan;
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!st.csr.live(e)) { continue; }
    int32_t col = st.csr.index[e];
    if (st.is_live(col)) {
      ++scan.free_entries;
      if (!st.is_integer(col)) { scan.all_integer = false; }
    } else {
      scan.has_fixed = true;
      dd_add_product(scan.fixed_terms, st.csr.value[e], st.col_lower[col]);
      scan.fixed_size += std::fabs(st.csr.value[e] * st.col_lower[col]);
    }
  }
  return scan;
}

// Factor scale / divisor that turns the free coefficients of the row into integers without a
// common divisor; rounding is set when a product is not an integer exactly. False when the row
// has no such factor within the limits above.
bool integral_factor(
  const context_t& c, int32_t row, double& scale, int64_t& divisor, bool& rounding)
{
  const state_t& st = c.st;
  int64_t product   = 1;
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!st.csr.live(e) || !st.is_live(st.csr.index[e])) { continue; }
    int64_t q =
      integral_denominator(std::fabs(st.csr.value[e]) * product, c.tol.zero, max_integral_scale);
    if (q == 0 || product * q > max_integral_scale) { return false; }
    product *= q;
  }
  scale        = product;
  divisor      = 0;
  rounding     = false;
  double drift = 0.;
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!st.csr.live(e)) { continue; }
    int32_t col = st.csr.index[e];
    if (!st.is_live(col)) { continue; }
    double scaled  = st.csr.value[e] * scale;
    double nearest = std::round(scaled);
    double delta   = std::fabs(scaled - nearest);
    if (delta > c.tol.zero || nearest == 0. || std::fabs(nearest) > max_integral_coefficient) {
      return false;
    }
    if (delta != 0.) {
      rounding = true;
      drift += delta * std::fmax(std::fabs(st.col_lower[col]), std::fabs(st.col_upper[col]));
    }
    divisor = std::gcd(divisor, (int64_t)std::fabs(nearest));
  }
  gprs_assert(divisor >= 1);
  // Infinite when a rounded coefficient belongs to a column without a finite bound.
  return drift <= coefficient_rounding_share * c.tol.feasibility;
}

// One side of the free part of an integer row: scaled to integral coefficients, rounded within
// the feasibility tolerance of that row, divided by the gcd and rounded again, exactly.
struct rounded_side_t {
  double value = 0.;
  bool exact   = true;   // the scaled side is small enough for an exact division
  bool moved   = false;  // by more than the tolerance
};

rounded_side_t round_side(
  double side, dd_t fixed_terms, double scale, int64_t divisor, bool is_lower, double feas)
{
  rounded_side_t r;
  double scaled   = dd_residual(side, fixed_terms) * scale;
  double integral = is_lower ? std::ceil(scaled - feas) : std::floor(scaled + feas);
  r.exact         = std::fabs(integral) <= max_exact_side;
  r.value         = is_lower ? std::ceil(integral / divisor) : std::floor(integral / divisor);
  r.moved         = std::fabs(r.value * divisor - scaled) > feas;
  return r;
}

// Integral scaling and side rounding of a row whose free columns are all integer.
rule_outcome_t integer_row(context_t& c, int32_t row, row_scan_t scan)
{
  state_t& st = c.st;
  double scale;
  int64_t divisor;
  bool rounding;
  c.budget.spend(2 * (int64_t)st.csr.used[row]);
  if (!integral_factor(c, row, scale, divisor, rounding)) { return rule_outcome_t::ok; }

  bool has_lower = st.row_lower[row] > -inf;
  bool has_upper = st.row_upper[row] < inf;
  // With fixed terms the edited row keeps its sides and those terms as doubles, and the side of
  // the free part is their difference. When they are too large for that difference to hold the
  // tolerance, the next visit reads another integer side than this one wrote and rounds again,
  // one unit further each time.
  if (scan.has_fixed) {
    double size = scan.fixed_size;
    if (has_lower) { size = std::fmax(size, std::fabs(st.row_lower[row])); }
    if (has_upper) { size = std::fmax(size, std::fabs(st.row_upper[row])); }
    if (!holds_tolerance(size * (scale / divisor), c.tol)) { return rule_outcome_t::ok; }
  }
  rounded_side_t low, up;
  if (has_lower) {
    low = round_side(st.row_lower[row], scan.fixed_terms, scale, divisor, true, c.tol.feasibility);
  }
  if (has_upper) {
    up = round_side(st.row_upper[row], scan.fixed_terms, scale, divisor, false, c.tol.feasibility);
  }
  bool round_sides = low.exact && up.exact;
  if (round_sides && has_lower && has_upper && low.value > up.value) {
    return rule_outcome_t::infeasible;
  }

  if (scale != 1. || divisor != 1 || rounding) {
    c.budget.spend(st.csr.used[row]);
    st.scale_row(row, scale / divisor, true);
    c.finding();
    if (scan.has_fixed) {
      c.budget.spend(st.csr.used[row]);
      scan = scan_row(st, row);
    }
  }
  if (!round_sides) { return rule_outcome_t::ok; }

  // Without fixed terms the sides become the integers. With them a side is rewritten only when
  // one of the two moved by more than the tolerance, and then both are, as
  // fl(integer + fixed terms) with one double for the fixed terms: that sum is monotone, so the
  // new sides do not cross.
  double lower     = st.row_lower[row];
  double upper     = st.row_upper[row];
  double new_lower = lower, new_upper = upper;
  if (!scan.has_fixed) {
    if (has_lower) { new_lower = low.value; }
    if (has_upper) { new_upper = up.value; }
  } else if (low.moved || up.moved) {
    double fixed = dd_value(scan.fixed_terms);
    if (has_lower) { new_lower = low.value + fixed; }
    if (has_upper) { new_upper = up.value + fixed; }
    gprs_assert(!has_lower || std::fabs(dd_residual(new_lower, scan.fixed_terms) - low.value) <=
                                c.tol.feasibility);
    gprs_assert(!has_upper || std::fabs(dd_residual(new_upper, scan.fixed_terms) - up.value) <=
                                c.tol.feasibility);
  }
  gprs_assert(new_lower <= new_upper);
  // The order keeps lower <= upper between the two calls.
  if (new_upper >= upper) {
    if (new_upper != upper) {
      st.set_row_upper(row, new_upper);
      c.finding();
    }
    if (new_lower != lower) {
      st.set_row_lower(row, new_lower);
      c.finding();
    }
  } else {
    if (new_lower != lower) {
      st.set_row_lower(row, new_lower);
      c.finding();
    }
    st.set_row_upper(row, new_upper);
    c.finding();
  }
  return rule_outcome_t::ok;
}

// Coefficient strengthening of a row with exactly one finite side b: with d = M - b for the
// maximum activity M of a <= row (b - m for the minimum activity m of a >= row), an integer
// column with |a| > d gets the coefficient sign(a) * d and the side moves by the change of the
// column's term in that activity, which leaves d as it is. The new side is computed first: one
// that is too large to be stored within the tolerance would change d by its rounding, and the
// row would cut feasible points off or be strengthened again at every visit.
void strengthen_row(context_t& c, int32_t row)
{
  state_t& st     = c.st;
  bool upper_side = st.row_upper[row] < inf;
  if (upper_side == (st.row_lower[row] > -inf)) { return; }
  if (refresh_activity(st, row)) { c.budget.spend(st.csr.used[row]); }
  const activity_t& act = upper_side ? st.max_act : st.min_act;
  if (act.inf_count[row] != 0) { return; }
  double side  = upper_side ? st.row_upper[row] : st.row_lower[row];
  double error = activity_error(act.updates[row], act.max_term[row]);
  double d     = dd_residual(side, dd_t{act.hi[row], act.lo[row]});
  if (upper_side) { d = -d; }
  if (!(d - error > c.tol.feasibility)) { return; }
  // A coefficient below the true d would cut feasible points off, one above it does not.
  double magnitude = d + error;
  double threshold = magnitude + c.tol.feasibility;

  auto strengthened = [&](int64_t e) {
    if (!st.csr.live(e)) { return false; }
    int32_t col = st.csr.index[e];
    return st.is_live(col) && st.is_integer(col) && std::fabs(st.csr.value[e]) > threshold;
  };
  dd_t new_side{side, 0.};
  bool changed = false;
  c.budget.spend(st.csr.used[row]);
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!strengthened(e)) { continue; }
    int32_t col  = st.csr.index[e];
    double a     = st.csr.value[e];
    double bound = upper_side ? max_activity_bound(a, st.col_lower[col], st.col_upper[col])
                              : min_activity_bound(a, st.col_lower[col], st.col_upper[col]);
    dd_sub_product(new_side, a, bound);
    dd_add_product(new_side, a > 0 ? magnitude : -magnitude, bound);
    changed = true;
  }
  if (!changed) { return; }
  double value = dd_value(new_side);
  if (!holds_tolerance(std::fabs(value), c.tol)) { return; }

  c.budget.spend(st.csr.used[row]);
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!strengthened(e)) { continue; }
    st.set_coefficient(e, st.csr.value[e] > 0 ? magnitude : -magnitude);
    c.finding();
  }
  if (value == side) { return; }
  if (upper_side) {
    st.set_row_upper(row, value);
  } else {
    st.set_row_lower(row, value);
  }
  c.finding();
}

}  // namespace

rule_result_t row_tightening_t::run(state_t& st,
                                    work_budget_t& budget,
                                    postsolve_t&,
                                    const options_t& opt)
{
  rule_result_t result;
  context_t c{st, budget, opt.tol, result};
  std::vector<int32_t>& rows = work->rows;
  // An edit puts its row on the list again: the call runs until the list stays empty or the
  // budget stops it.
  while (!st.row_frontier.empty(row_list::tightening)) {
    st.row_frontier.take(row_list::tightening, rows);
    for (size_t k = 0; k < rows.size(); ++k) {
      if (budget.stop()) {
        for (; k < rows.size(); ++k) {
          if (st.row_alive[rows[k]]) {
            st.row_frontier.push(list_bit(row_list::tightening), rows[k]);
          }
        }
        result.stopped = true;
        return result;
      }
      int32_t row = rows[k];
      if (!st.row_alive[row]) { continue; }
      budget.spend(line_visit_work + st.csr.used[row]);
      row_scan_t scan = scan_row(st, row);
      if (scan.free_entries == 0) { continue; }
      if (scan.all_integer) {
        result.outcome = integer_row(c, row, scan);
        if (result.outcome != rule_outcome_t::ok) { return result; }
      }
      strengthen_row(c, row);
    }
  }
  return result;
}

}  // namespace gprs
