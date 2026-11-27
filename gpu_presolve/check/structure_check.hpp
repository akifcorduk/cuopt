/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The structure check: what one presolve result must satisfy whatever point is postsolved. It
// reads the postsolve stack, the origin maps and the reduced model, needs no solver and no
// solution file, and costs O(columns + records + nonzeros) (a row whose coefficients were
// edited one by one costs its length times the number of its different coefficient ratios).
// It sees what the crush check cannot: a reduction that relaxes.
//
//   stack      The stack and the maps keep the contract of postsolve.hpp: every column is left
//              in the reduced model or removed by exactly one record and named by no later
//              record; a fixed value is finite, and integral for an integer column; an alias
//              has a finite scale other than zero and a finite offset, and an integer column an
//              integer rep with an integral scale and offset; a ratio is finite and not zero.
//   types      A column that is integer in the original model is integer in the reduced model.
//   bounds     With the records composed, every original column is a constant or
//              scale * (one reduced column) + offset. Over the bounds of that reduced column it
//              stays within its original bounds.
//   rows       The activity of an original row at a postsolved point is a linear function g of
//              the reduced columns. A row that left the model holds at the postsolve of every
//              point within the bounds of the reduced model: the range of g over the bounds
//              lies within the sides of the row.
//   kept_rows  A row that is in the reduced model, as row c with sides [l, u], holds at the
//              postsolve of every point within the bounds that meets that reduced row: the
//              maximum of g over { bounds, l <= c * r <= u } is not above its upper side, and
//              the minimum not below its lower side. A row removed as a parallel row is tested
//              in the same way against the reduced row of its kept row, or as a row that left
//              the model when the kept row left it.
//              The maximum is the smallest of the bounds  lambda * (u or l) + max over the
//              bounds of (g - lambda * c) * r,  which is reached at lambda = 0 or at a ratio
//              g[k] / c[k].
//   postsolve  postsolve_t::postsolve of one point within the bounds of the reduced model gives
//              what the composed records give.
//
// Why bounds, rows and kept_rows hold for the Stage 1 rules, in both modes. No rule relaxes a
// bound, and the bounds of a substituted column are transferred to its root. A row leaves the
// model because its sides hold over the bounds of its columns (redundant, empty), because its
// bound went to its only column (singleton), because all its columns were fixed (forcing), or
// because it defines a substituted column (doubleton equation, whose g is then a constant); and
// bounds only tighten afterwards. Every edit of a row that stays gives a row that implies the
// row before the edit over the bounds, for real points as well: scaling and substitution keep
// it, a rounded or intersected side is tighter, and a strengthened coefficient a -> d with the
// side moved by (a - d) * bound differs from the old row by (a - d) * (bound - x) >= 0. Like
// the crush check, this has to be reread when a rule is added that relaxes a bound, deletes a
// row for another reason or combines rows (Stage 3: free column substitution, dominated
// columns, sparsification).
// The tolerance is wider than that of a tested point: a row can use the tolerance of its
// deletion and that of a rounded bound together.

#include "crush_check.hpp"

#include <gprs/presolve.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
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
// A kept row costs its length times its number of different ratios. One above the first limit
// is not tested, and none once the rows together are above the second.
constexpr double max_kept_row_work = 5e7;
constexpr double max_kept_work     = 2e10;

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

  // Rows that left the model. Violations are relative to max(1, largest term).
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

  int64_t postsolve_failures  = 0;
  double postsolve_worst      = 0.;  // relative to max(1, largest term)
  int32_t postsolve_worst_col = -1;
};

inline bool is_integral(double v) { return v == std::round(v); }

// Tests the contract of the stack and composes its records. On a broken contract `error` says
// which part, with the index of the record or column.
inline bool compose_records(const gprs::model_t& model,
                            const gprs::result_t& result,
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
  size_t fixed = 0, alias = 0, parallel = 0;
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
          removed[r.rep] || map[r.col].root != unset || !row_ok(r.row)) {
        return fail("columns of alias record", k);
      }
      if (!std::isfinite(r.scale) || r.scale == 0. || !std::isfinite(r.offset)) {
        return fail("scale or offset of alias record", k);
      }
      if (model.col_type[r.col] &&
          !(model.col_type[r.rep] && is_integral(r.scale) && is_integral(r.offset))) {
        return fail("integer column of alias record", k);
      }
      removed[r.col] = 1;
      --left;
    } else {
      if (parallel == ps.parallel_rows.size()) { return fail("parallel records missing at", k); }
      const gprs::parallel_row_record_t& r = ps.parallel_rows[parallel++];
      if (!row_ok(r.removed_row) || !row_ok(r.kept_row) || r.removed_row == r.kept_row ||
          !std::isfinite(r.ratio) || r.ratio == 0.) {
        return fail("parallel record", k);
      }
    }
  }
  if (fixed != ps.fixed_cols.size() || alias != ps.aliases.size() ||
      parallel != ps.parallel_rows.size()) {
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

inline structure_report_t structure_check(const gprs::model_t& model, const gprs::result_t& result)
{
  structure_report_t s;
  s.ran = true;
  std::vector<column_map_t> map;
  if (!compose_records(model, result, map, s.stack_error)) { return s; }
  s.composed                   = true;
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;

  for (int32_t k = 0; k < reduced.ncols; ++k) {
    int32_t col = ps.col_origin[k];
    if (model.col_type[col] && !reduced.col_type[k]) {
      if (s.lost_integers++ == 0) { s.first_lost_integer = col; }
    }
  }

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
  std::vector<double> g(reduced.ncols, 0.), parts(reduced.ncols, 0.), c(reduced.ncols, 0.);
  std::vector<int32_t> touched;
  std::vector<row_term_t> terms;
  std::vector<double> multipliers;
  double kept_work = 0.;
  for (int32_t i = 0; i < model.nrows; ++i) {
    int32_t against = i;
    for (size_t hops = 0;
         reduced_row[against] < 0 && kept_row[against] >= 0 && hops <= ps.parallel_rows.size();
         ++hops) {
      against = kept_row[against];
    }
    int32_t k = reduced_row[against];

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

    if (k < 0) {
      sum_t low = constant, high = constant;
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
          raise_to(largest, std::fabs(coefficient * at_low));
        } else {
          low_infinite = true;
        }
        if (std::isfinite(at_high)) {
          high.add(coefficient, at_high);
          raise_to(largest, std::fabs(coefficient * at_high));
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
      double scaled = violation / largest;
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
