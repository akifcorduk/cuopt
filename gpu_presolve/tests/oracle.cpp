/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "oracle.hpp"

#include <gprs/assert.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <sstream>

namespace gprs {
namespace oracle {

rng_t::rng_t(uint64_t seed) : state(seed) {}

uint64_t rng_t::next()
{
  state += 0x9e3779b97f4a7c15ull;
  uint64_t z = state;
  z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

int32_t rng_t::uniform(int32_t lo, int32_t hi)
{
  gprs_assert(lo <= hi);
  uint64_t range = (uint64_t)((int64_t)hi - lo) + 1;
  // Draws below 2^64 mod range are rejected, which leaves a multiple of range.
  uint64_t reject = (0ull - range) % range;
  uint64_t draw   = next();
  while (draw < reject) {
    draw = next();
  }
  return (int32_t)(lo + (int64_t)(draw % range));
}

bool rng_t::chance(double probability) { return (next() >> 11) * 0x1p-53 < probability; }

namespace {

// Every draw is a statement of its own: the order of two draws inside one expression is not
// fixed by the language.

double nonzero(rng_t& rng, int32_t largest)
{
  int32_t magnitude = rng.uniform(1, largest);
  return rng.chance(0.5) ? magnitude : -magnitude;
}

// Around the limit of an infinite activity term (1e20) for coefficients 1 to 3, and below it.
constexpr double huge_bounds[] = {1e7, 1e10, 1e15, 3e19, 4e19, 1e20, 2e20, 1e21, 1e22};

double huge_bound(rng_t& rng) { return huge_bounds[rng.uniform(0, 8)]; }

int32_t enumerated_lower(rng_t& rng, const generator_options_t& o, int32_t width)
{
  int32_t kind = rng.uniform(0, 9);
  if (kind < 6) { return 0; }
  int32_t from = -o.max_side, to = o.max_side - width;
  if (kind < 9) {
    from = std::max(from, -3);
    to   = std::min(to, 3);
  }
  return rng.uniform(from, to);
}

void draw_line_bounds(
  rng_t& rng, const generator_options_t& o, bool integer, bool fixed, double& lower, double& upper)
{
  if (fixed) {
    lower = upper = rng.uniform(-o.max_side, o.max_side);
    return;
  }
  bool lower_infinite = rng.chance(o.p_infinite_bound);
  bool upper_infinite = rng.chance(o.p_infinite_bound);
  bool lower_huge     = !lower_infinite && rng.chance(o.p_huge_bound);
  bool upper_huge     = !upper_infinite && rng.chance(o.p_huge_bound);
  bool lower_small    = !lower_infinite && !lower_huge;
  bool upper_small    = !upper_infinite && !upper_huge;
  lower               = -inf;
  upper               = inf;
  if (lower_huge) { lower = -huge_bound(rng); }
  if (upper_huge) { upper = huge_bound(rng); }
  if (lower_small && upper_small) {
    int32_t gap = integer ? o.max_width + 1 : 0;
    int32_t l   = rng.uniform(-o.max_side, o.max_side - gap);
    lower       = l;
    upper       = rng.uniform(l + gap, o.max_side);
  } else if (lower_small) {
    lower = rng.uniform(-o.max_side, o.max_side);
  } else if (upper_small) {
    upper = rng.uniform(-o.max_side, o.max_side);
  }
  if (integer && lower_small && rng.chance(o.p_fractional_bound)) { lower -= 0.5; }
  if (integer && upper_small && rng.chance(o.p_fractional_bound)) { upper += 0.5; }
}

enum class row_kind_t { parallel, empty, singleton, doubleton, general };

struct draft_row_t {
  row_kind_t kind = row_kind_t::empty;
  int32_t source  = -1;  // parallel: the row it is a multiple of
  int32_t factor  = 1;   // general: every coefficient is a multiple of it
  bool has_line   = false;
  std::vector<int32_t> col;
  std::vector<double> value;
  // The entries in columns behind the counted ones, in increasing column order: the singleton
  // columns of the row, then its entries in forcing columns. A copy of the row does not get
  // them.
  std::vector<int32_t> own_col;
  std::vector<double> own_value;
  double lower = -inf;
  double upper = inf;
};

void sort_entries(draft_row_t& row)
{
  std::vector<int32_t> order(row.col.size());
  for (size_t k = 0; k < order.size(); ++k) {
    order[k] = k;
  }
  std::sort(
    order.begin(), order.end(), [&](int32_t a, int32_t b) { return row.col[a] < row.col[b]; });
  draft_row_t sorted = row;
  for (size_t k = 0; k < order.size(); ++k) {
    sorted.col[k]   = row.col[order[k]];
    sorted.value[k] = row.value[order[k]];
  }
  row = sorted;
}

bool is_integer(double v) { return v == std::floor(v); }

bool on_half_grid(double v) { return 2. * v == std::floor(2. * v); }

// A coefficient that was read as an integer or scaled by a power of two has few significant
// bits; one that a division or a product has left has all of them.
bool few_bits(double v)
{
  constexpr int32_t bits = 30;
  int exponent           = 0;
  double scaled          = std::ldexp(std::frexp(v, &exponent), bits);
  return scaled == std::floor(scaled);
}

// First and last integer of [lower, upper]; a bound off the half grid gets the tolerance.
double first_integer(double lower)
{
  if (on_half_grid(lower)) { return std::ceil(lower); }
  return std::ceil(lower - solve_tolerance * std::max(1., std::fabs(lower)));
}

double last_integer(double upper)
{
  if (on_half_grid(upper)) { return std::floor(upper); }
  return std::floor(upper + solve_tolerance * std::max(1., std::fabs(upper)));
}

}  // namespace

case_t generate(uint64_t seed, const generator_options_t& o)
{
  gprs_assert(o.max_rows >= 0 && o.max_enum_cols >= 0 && o.max_line_cols >= 0);
  gprs_assert(o.max_coefficient >= 1 && o.max_width >= 0 && o.max_side > o.max_width);
  gprs_assert(o.max_scale_exp2 >= 0 && o.max_scale_exp2 <= 30);
  rng_t rng(seed);
  int32_t nrows  = rng.uniform(0, o.max_rows);
  int32_t n_enum = rng.uniform(0, o.max_enum_cols);
  int32_t n_line = rng.uniform(0, o.max_line_cols);
  int32_t ncols  = n_enum + n_line;

  std::vector<uint8_t> line(ncols, 0);
  std::fill(line.begin(), line.begin() + n_line, 1);
  for (int32_t j = ncols - 1; j > 0; --j) {
    std::swap(line[j], line[rng.uniform(0, j)]);
  }

  model_t base;
  base.nrows = nrows;
  base.ncols = ncols;
  base.col_lower.resize(ncols);
  base.col_upper.resize(ncols);
  base.cost.resize(ncols);
  base.col_type.resize(ncols);
  std::vector<int32_t> usable;
  for (int32_t j = 0; j < ncols; ++j) {
    if (!rng.chance(o.p_empty_col)) { usable.push_back(j); }
    bool fixed = rng.chance(o.p_fixed_col);
    if (line[j]) {
      base.col_type[j] = rng.chance(o.p_continuous) ? 0 : 1;
      draw_line_bounds(rng, o, base.col_type[j] == 1, fixed, base.col_lower[j], base.col_upper[j]);
    } else {
      base.col_type[j]  = 1;
      int32_t width     = fixed ? 0 : rng.uniform(std::min(1, o.max_width), o.max_width);
      int32_t lower     = enumerated_lower(rng, o, width);
      base.col_lower[j] = lower;
      base.col_upper[j] = lower + width;
      if (width < o.max_width && rng.chance(o.p_fractional_bound)) {
        if (rng.chance(0.5)) {
          base.col_lower[j] -= 0.5;
        } else {
          base.col_upper[j] += 0.5;
        }
      }
    }
    base.cost[j] = rng.chance(o.p_zero_cost) ? 0. : nonzero(rng, o.max_coefficient);
  }

  // Kind of every row, and the entries of the rows that are not copies.
  std::vector<draft_row_t> rows(nrows);
  std::vector<uint8_t> used(ncols, 0);
  auto add_entry = [&](draft_row_t& row, int32_t j, double a) {
    row.col.push_back(j);
    row.value.push_back(a);
    row.has_line = row.has_line || line[j];
    used[j]      = 1;
  };
  for (int32_t i = 0; i < nrows; ++i) {
    draft_row_t& row = rows[i];
    if (i > 0 && rng.chance(o.p_parallel_row)) {
      row.kind   = row_kind_t::parallel;
      row.source = rng.uniform(0, i - 1);
    } else if (rng.chance(o.p_empty_row)) {
    } else if (!usable.empty() && rng.chance(o.p_singleton_row)) {
      row.kind  = row_kind_t::singleton;
      int32_t j = usable[rng.uniform(0, (int32_t)usable.size() - 1)];
      add_entry(row, j, nonzero(rng, o.max_coefficient));
    } else if (usable.size() >= 2 && rng.chance(o.p_doubleton_equation)) {
      row.kind      = row_kind_t::doubleton;
      int32_t first = usable[rng.uniform(0, (int32_t)usable.size() - 1)];
      std::vector<int32_t> partners;
      for (int32_t j : usable) {
        if (j != first && !(line[first] && line[j])) { partners.push_back(j); }
      }
      add_entry(row, first, nonzero(rng, o.max_coefficient));
      if (!partners.empty()) {
        int32_t second = partners[rng.uniform(0, (int32_t)partners.size() - 1)];
        add_entry(row, second, nonzero(rng, o.max_coefficient));
      }
    } else {
      row.kind = row_kind_t::general;
      if (o.max_coefficient >= 2 && rng.chance(o.p_common_factor)) {
        row.factor = rng.uniform(2, o.max_coefficient);
      }
      std::vector<int32_t> left;
      for (int32_t j : usable) {
        if (rng.chance(o.density) && !(line[j] && row.has_line)) {
          add_entry(row, j, row.factor * nonzero(rng, o.max_coefficient / row.factor));
        } else {
          left.push_back(j);
        }
      }
      // At least two entries where the columns allow it; shorter rows have their own knobs.
      while (row.col.size() < 2 && !left.empty()) {
        int32_t pick = rng.uniform(0, (int32_t)left.size() - 1);
        int32_t j    = left[pick];
        left.erase(left.begin() + pick);
        if (line[j] && row.has_line) { continue; }
        add_entry(row, j, row.factor * nonzero(rng, o.max_coefficient / row.factor));
      }
    }
  }
  // A usable column that no row took goes into a row drawn entry by entry, if one can hold it.
  for (int32_t j : usable) {
    if (used[j]) { continue; }
    std::vector<int32_t> open;
    for (int32_t i = 0; i < nrows; ++i) {
      if (rows[i].kind == row_kind_t::general && !(line[j] && rows[i].has_line)) {
        open.push_back(i);
      }
    }
    if (open.empty()) { continue; }
    draft_row_t& row = rows[open[rng.uniform(0, (int32_t)open.size() - 1)]];
    add_entry(row, j, row.factor * nonzero(rng, o.max_coefficient / row.factor));
  }

  for (int32_t i = 0; i < nrows; ++i) {
    draft_row_t& row = rows[i];
    bool own_sides   = true;
    if (row.kind == row_kind_t::parallel) {
      const draft_row_t& other = rows[row.source];
      double largest           = 1.;
      for (double a : other.value) {
        largest = std::max(largest, std::fabs(a));
      }
      double factor = nonzero(rng, (int32_t)(o.max_coefficient / largest));
      row.col       = other.col;
      row.value     = other.value;
      for (double& a : row.value) {
        a *= factor;
      }
      if (std::fabs(factor) == 1. && rng.chance(1. / 3.)) {
        own_sides = false;
        row.lower = factor > 0 ? other.lower : -other.upper;
        row.upper = factor > 0 ? other.upper : -other.lower;
      }
    } else {
      sort_entries(row);
    }
    std::vector<int32_t> cols  = row.col;
    std::vector<double> values = row.value;
    double p_no_cost = o.p_singleton_zero_cost < 0. ? o.p_zero_cost : o.p_singleton_zero_cost;
    for (int32_t n = 0; n < o.max_row_singletons; ++n) {
      if (!(o.p_singleton_col > 0. && !row.col.empty() &&
            base.ncols - ncols < o.max_singleton_cols && rng.chance(o.p_singleton_col))) {
        break;
      }
      bool has_line = false;
      for (int32_t j : cols) {
        has_line = has_line || line[j];
      }
      bool is_line = !has_line && rng.chance(o.p_singleton_line);
      double lower = 0., upper = 0.;
      uint8_t type = 1;
      if (is_line) {
        type = rng.chance(o.p_continuous) ? 0 : 1;
        draw_line_bounds(rng, o, type == 1, false, lower, upper);
      } else {
        int32_t width = std::min(1, o.max_width);
        lower         = enumerated_lower(rng, o, width);
        upper         = lower + width;
      }
      double cost = rng.chance(p_no_cost) ? 0. : nonzero(rng, o.max_coefficient);
      row.own_col.push_back(base.ncols++);
      row.own_value.push_back(nonzero(rng, o.max_coefficient));
      base.col_type.push_back(type);
      base.col_lower.push_back(lower);
      base.col_upper.push_back(upper);
      base.cost.push_back(cost);
      line.push_back(is_line);
      cols.push_back(row.own_col.back());
      values.push_back(row.own_value.back());
    }
    if (!own_sides) { continue; }

    // Activity range over the small bounds. An end that an infinite or huge bound leaves open
    // is put four from the other end, so that the sides stay where the row can be met.
    double act_min = 0., act_max = 0.;
    bool min_open = false, max_open = false;
    for (size_t k = 0; k < cols.size(); ++k) {
      int32_t j    = cols[k];
      double a     = values[k];
      double first = std::ceil(base.col_lower[j]), last = std::floor(base.col_upper[j]);
      double low = a > 0 ? first : last, high = a > 0 ? last : first;
      if (std::fabs(low) <= o.max_side + 1) {
        act_min += a * low;
      } else {
        min_open = true;
      }
      if (std::fabs(high) <= o.max_side + 1) {
        act_max += a * high;
      } else {
        max_open = true;
      }
    }
    if (min_open && max_open) {
      act_min = -3.;
      act_max = 3.;
    } else if (min_open) {
      act_min = act_max - 4.;
    } else if (max_open) {
      act_max = act_min + 4.;
    }
    double limit = o.max_side;
    auto side    = [&]() -> double {
      if (rng.chance(o.p_forcing_side)) {
        return std::clamp(rng.chance(0.5) ? act_min : act_max, -limit, limit);
      }
      if (rng.chance(o.p_outside_side)) {
        return std::clamp(rng.chance(0.5) ? act_min - 1. : act_max + 1., -limit, limit);
      }
      int32_t from = std::clamp(act_min, -limit, limit);
      int32_t to   = std::clamp(act_max, -limit, limit);
      return rng.uniform(from, to);
    };
    if (row.kind == row_kind_t::doubleton) {
      row.lower = row.upper = side();
    } else if (rng.chance(o.p_free_row)) {
    } else if (rng.chance(o.p_equation)) {
      row.lower = row.upper = side();
    } else if (rng.chance(o.p_ranged_row)) {
      double a  = side();
      double b  = side();
      row.lower = std::min(a, b);
      row.upper = std::max(a, b);
    } else if (rng.chance(0.5)) {
      row.upper = side();
    } else {
      row.lower = side();
    }
  }

  for (int32_t n = 0; n < o.max_forcing_cols; ++n) {
    if (!(o.p_forcing_col > 0. && rng.chance(o.p_forcing_col))) { break; }
    bool any_sides = rng.chance(1. / 7.);
    bool one_wrong = rng.chance(1. / 7.);
    bool has_cost  = rng.chance(1. / 7.);
    std::vector<int32_t> open;
    for (int32_t i = 0; i < nrows; ++i) {
      const draft_row_t& row = rows[i];
      bool has_line          = false;
      for (int32_t j : row.col) {
        has_line = has_line || line[j];
      }
      for (int32_t j : row.own_col) {
        has_line = has_line || line[j];
      }
      int32_t sides = (row.lower > -inf) + (row.upper < inf);
      if (!row.col.empty() && !has_line && (sides == 1 || (any_sides && sides == 2))) {
        open.push_back(i);
      }
    }
    if (open.empty()) { break; }
    bool down      = rng.chance(0.5);
    uint8_t type   = rng.chance(o.p_continuous) ? 0 : 1;
    double bound   = rng.chance(0.5) ? inf : rng.uniform(-o.max_side, o.max_side);
    double cost    = has_cost ? nonzero(rng, o.max_coefficient) : 0.;
    int32_t col    = base.ncols++;
    int32_t length = rng.uniform(1, std::min(3, (int32_t)open.size()));
    base.col_type.push_back(type);
    base.col_lower.push_back(down ? -inf : -bound);
    base.col_upper.push_back(down ? bound : inf);
    base.cost.push_back(cost);
    line.push_back(1);
    for (int32_t k = 0; k < length; ++k) {
      int32_t pick     = rng.uniform(0, (int32_t)open.size() - 1);
      draft_row_t& row = rows[open[pick]];
      open.erase(open.begin() + pick);
      // The sign with which the finite side of the row stops the column on its way up, for a
      // column that is free to go down.
      double a = rng.uniform(1, o.max_coefficient);
      if ((row.upper < inf) != down) { a = -a; }
      if (one_wrong && k == 0) { a = -a; }
      row.own_col.push_back(col);
      row.own_value.push_back(a);
    }
  }

  for (int32_t n = 0; n < o.max_probing_rows; ++n) {
    if (!(o.p_probing_row > 0. && rng.chance(o.p_probing_row))) { break; }
    int32_t kind      = rng.uniform(0, 7);
    bool wide         = kind == 6;
    bool prefers_line = kind == 7 || (kind < 4 && rng.chance(0.5));
    // The integer points of an enumerated column, and for a line column its small bounds or,
    // where an end is open, a stretch of four from the other end.
    double largest = o.probing_wide ? huge_bounds[0] : o.max_side + 1;
    auto ends      = [&](int32_t j, double& first, double& last) {
      first           = std::ceil(base.col_lower[j]);
      last            = std::floor(base.col_upper[j]);
      bool low_small  = std::fabs(first) <= largest;
      bool high_small = std::fabs(last) <= largest;
      if (!low_small) { first = high_small ? last - 4. : -2.; }
      if (!high_small) { last = first + 4.; }
    };
    // An enumerated column with two values other than 0 and 1.
    auto shifted = [&](int32_t j) {
      double first, last;
      ends(j, first, last);
      return !line[j] && last - first == 1. && first != 0.;
    };
    std::vector<int32_t> binaries, plain, others;
    for (int32_t j : usable) {
      double first, last;
      ends(j, first, last);
      if (line[j] || last - first != (wide ? 2. : 1.)) { continue; }
      binaries.push_back(j);
      if (first == 0.) { plain.push_back(j); }
    }
    if (binaries.empty()) { continue; }
    // Nine times in ten a column with the values 0 and 1, where the model has one.
    if (!plain.empty() && rng.chance(0.9)) { binaries = plain; }
    int32_t binary = binaries[rng.uniform(0, (int32_t)binaries.size() - 1)];
    for (int32_t j : usable) {
      if (j == binary) { continue; }
      if (line[j] && prefers_line) {
        others.insert(others.begin(), j);
      } else {
        others.push_back(j);
      }
    }
    int32_t count = rng.uniform(2, 3);
    draft_row_t row;
    row.kind     = row_kind_t::general;
    double reach = 0., minimum = 0.;
    bool at_front = prefers_line;
    while ((int32_t)row.col.size() < count && !others.empty()) {
      int32_t pick = at_front ? 0 : rng.uniform(0, (int32_t)others.size() - 1);
      at_front     = false;
      int32_t j    = others[pick];
      others.erase(others.begin() + pick);
      if (line[j] && row.has_line) { continue; }
      if (shifted(j) && rng.chance(0.75)) { continue; }
      double a = nonzero(rng, o.max_coefficient);
      if (wide) { a *= 2.; }
      double first, last;
      ends(j, first, last);
      add_entry(row, j, a);
      reach += std::fabs(a) * (last - first);
      minimum += a > 0 ? a * first : a * last;
    }
    double first, last;
    ends(binary, first, last);
    double step = wide ? reach / 2. : reach;
    if (kind == 5) { step += rng.chance(0.5) ? 1. : -1.; }
    if (step <= 0.) { step = 1.; }
    double a = rng.chance(0.5) ? step : -step;
    add_entry(row, binary, a);
    minimum += a > 0 ? a * first : a * last;
    double side = minimum + step * (last - first);
    if (kind == 4) { side += rng.chance(0.5) ? 1. : -1.; }
    sort_entries(row);
    row.lower = row.upper = side;
    rows.push_back(row);
  }
  nrows      = rows.size();
  base.nrows = nrows;

  base.row_start.push_back(0);
  for (const draft_row_t& row : rows) {
    base.col_index.insert(base.col_index.end(), row.col.begin(), row.col.end());
    base.value.insert(base.value.end(), row.value.begin(), row.value.end());
    base.col_index.insert(base.col_index.end(), row.own_col.begin(), row.own_col.end());
    base.value.insert(base.value.end(), row.own_value.begin(), row.own_value.end());
    base.row_start.push_back(base.value.size());
    base.row_lower.push_back(row.lower);
    base.row_upper.push_back(row.upper);
  }
  if (rng.chance(0.5)) { base.objective_offset = rng.uniform(-o.max_side, o.max_side); }

  if (nrows + ncols > 0 && rng.chance(o.p_crossing)) {
    int32_t pick  = rng.uniform(0, nrows + ncols - 1);
    int32_t gap   = rng.uniform(1, std::min(3, 2 * o.max_side));
    int32_t upper = rng.uniform(-o.max_side, o.max_side - gap);
    if (pick < nrows) {
      base.row_upper[pick] = upper;
      base.row_lower[pick] = upper + gap;
    } else {
      base.col_upper[pick - nrows] = upper;
      base.col_lower[pick - nrows] = upper + gap;
    }
  }

  std::vector<double> row_scale(nrows, 1.), col_scale(base.ncols, 1.);
  if (rng.chance(o.p_scaled)) {
    for (int32_t i = 0; i < nrows; ++i) {
      row_scale[i] = std::ldexp(1., rng.uniform(-o.max_scale_exp2, o.max_scale_exp2));
    }
    for (int32_t j = 0; j < base.ncols; ++j) {
      if (base.col_type[j] == 1) { continue; }
      col_scale[j] = std::ldexp(1., rng.uniform(-o.max_scale_exp2, o.max_scale_exp2));
    }
  }
  if (o.p_parallel_scale > 0.) {
    gprs_assert(o.max_parallel_scale_exp2 >= 2 && o.max_parallel_scale_exp2 <= 20);
    static const double others[] = {
      3., 5., 6., 7., 10., 100., 1000., 12345., 65537., 999999., 1000003.};
    std::vector<uint8_t> has_factor(nrows, 0);
    for (int32_t i = 0; i < nrows; ++i) {
      if (rows[i].kind != row_kind_t::parallel || !rng.chance(o.p_parallel_scale)) { continue; }
      int32_t target = rng.chance(0.5) ? i : rows[i].source;
      double factor  = std::ldexp(1., rng.uniform(1, o.max_parallel_scale_exp2));
      if (rng.chance(0.5)) {
        double other = others[rng.uniform(0, 10)];
        factor       = other <= std::ldexp(1., o.max_parallel_scale_exp2) ? other : 3.;
      }
      if (has_factor[target]) { continue; }
      has_factor[target] = 1;
      row_scale[target] *= factor;
    }
  }
  case_t c = make_case(base, row_scale, col_scale);
  gprs_assert(classify(c.base, o.max_width, line));
  return c;
}

model_t scale_model(const model_t& base,
                    const std::vector<double>& row_scale,
                    const std::vector<double>& col_scale)
{
  gprs_assert((int32_t)row_scale.size() == base.nrows);
  gprs_assert((int32_t)col_scale.size() == base.ncols);
  model_t model = base;
  for (int32_t i = 0; i < base.nrows; ++i) {
    gprs_assert(row_scale[i] > 0 && std::isfinite(row_scale[i]));
    for (int64_t e = base.row_start[i]; e < base.row_start[i + 1]; ++e) {
      model.value[e] = row_scale[i] * base.value[e] * col_scale[base.col_index[e]];
    }
    model.row_lower[i] = row_scale[i] * base.row_lower[i];
    model.row_upper[i] = row_scale[i] * base.row_upper[i];
  }
  for (int32_t j = 0; j < base.ncols; ++j) {
    gprs_assert(col_scale[j] > 0 && std::isfinite(col_scale[j]));
    model.col_lower[j] = base.col_lower[j] / col_scale[j];
    model.col_upper[j] = base.col_upper[j] / col_scale[j];
    model.cost[j]      = base.cost[j] * col_scale[j];
  }
  return model;
}

case_t make_case(const model_t& base,
                 const std::vector<double>& row_scale,
                 const std::vector<double>& col_scale)
{
  gprs_assert(well_formed(base));
  case_t c;
  c.base      = base;
  c.row_scale = row_scale.empty() ? std::vector<double>(base.nrows, 1.) : row_scale;
  c.col_scale = col_scale.empty() ? std::vector<double>(base.ncols, 1.) : col_scale;
  c.model     = scale_model(base, c.row_scale, c.col_scale);
  gprs_assert(well_formed(c.model));
  return c;
}

namespace {

std::string number(double v)
{
  if (v == inf) { return "inf"; }
  if (v == -inf) { return "-inf"; }
  char text[40];
  std::snprintf(text, sizeof(text), "%.17g", v);
  std::string s = text;
  if (s.find_first_of(".en") == std::string::npos) { s += '.'; }
  return s;
}

template <typename T, typename F>
std::string list(const std::vector<T>& values, F text)
{
  std::string out = "{";
  size_t width    = 0;
  for (size_t k = 0; k < values.size(); ++k) {
    std::string item = text(values[k]);
    if (k > 0 && width + item.size() > 78) {
      out += ",\n    ";
      width = 0;
    } else if (k > 0) {
      out += ", ";
    }
    out += item;
    width += item.size() + 2;
  }
  return out + "}";
}

std::string integers(const std::vector<int64_t>& v)
{
  return list(v, [](int64_t a) { return std::to_string(a); });
}

std::string numbers(const std::vector<double>& v) { return list(v, number); }

}  // namespace

std::string to_code(const case_t& c)
{
  const model_t& b = c.base;
  std::vector<int64_t> col_index(b.col_index.begin(), b.col_index.end());
  std::vector<int64_t> col_type(b.col_type.begin(), b.col_type.end());
  std::string out;
  out += "  model_t base;\n";
  out += "  base.nrows = " + std::to_string(b.nrows) + ";\n";
  out += "  base.ncols = " + std::to_string(b.ncols) + ";\n";
  out += "  base.row_start = " + integers(b.row_start) + ";\n";
  out += "  base.col_index = " + integers(col_index) + ";\n";
  out += "  base.value = " + numbers(b.value) + ";\n";
  out += "  base.row_lower = " + numbers(b.row_lower) + ";\n";
  out += "  base.row_upper = " + numbers(b.row_upper) + ";\n";
  out += "  base.col_lower = " + numbers(b.col_lower) + ";\n";
  out += "  base.col_upper = " + numbers(b.col_upper) + ";\n";
  out += "  base.cost = " + numbers(b.cost) + ";\n";
  out += "  base.col_type = " + integers(col_type) + ";\n";
  out += "  base.objective_offset = " + number(b.objective_offset) + ";\n";
  out += "  oracle::case_t c = oracle::make_case(\n    base,\n    " + numbers(c.row_scale) +
         ",\n    " + numbers(c.col_scale) + ");\n";
  return out;
}

bool classify(const model_t& model, int32_t max_width, std::vector<uint8_t>& line_col)
{
  line_col.resize(model.ncols);
  for (int32_t j = 0; j < model.ncols; ++j) {
    double lower = model.col_lower[j], upper = model.col_upper[j];
    line_col[j] = !(model.col_type[j] == 1 && std::isfinite(lower) && std::isfinite(upper) &&
                    upper - lower <= max_width);
  }
  for (int32_t i = 0; i < model.nrows; ++i) {
    int32_t lines = 0;
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      lines += line_col[model.col_index[e]];
    }
    if (lines > 1) { return false; }
  }
  return true;
}

double assignments(const model_t& model, const std::vector<uint8_t>& line_col)
{
  double count = 1.;
  for (int32_t j = 0; j < model.ncols; ++j) {
    if (line_col[j]) { continue; }
    count *=
      std::max(0., last_integer(model.col_upper[j]) - first_integer(model.col_lower[j]) + 1.);
  }
  return count;
}

namespace {

// The values of max_width at which the line columns of the model change, ascending.
std::vector<int32_t> width_limits(const model_t& model)
{
  // Below every width: no column is enumerated.
  std::vector<int32_t> limits = {std::numeric_limits<int32_t>::min()};
  for (int32_t j = 0; j < model.ncols; ++j) {
    double width = model.col_upper[j] - model.col_lower[j];
    if (model.col_type[j] == 1 && std::fabs(width) <= 0x1p20) {
      limits.push_back((int32_t)std::ceil(width));
    }
  }
  std::sort(limits.begin(), limits.end());
  limits.erase(std::unique(limits.begin(), limits.end()), limits.end());
  return limits;
}

// The model with every row of several line columns reduced to one of them by equations, as
// into_class() states it. False when a row is left with two.
bool reduce_line_columns(const model_t& model, const std::vector<uint8_t>& line_col, model_t& out)
{
  constexpr double cancellation = 1e-12;
  std::vector<std::map<int32_t, double>> rows(model.nrows);
  for (int32_t i = 0; i < model.nrows; ++i) {
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      rows[i][model.col_index[e]] = model.value[e];
    }
  }
  std::vector<double> lower = model.row_lower, upper = model.row_upper;
  auto lines = [&](int32_t i) {
    std::vector<int32_t> cols;
    for (const auto& entry : rows[i]) {
      if (line_col[entry.first]) { cols.push_back(entry.first); }
    }
    return cols;
  };
  // An equation other than row i that holds j as its only line column, or -1.
  auto equation_of = [&](int32_t i, int32_t j) {
    for (int32_t k = 0; k < model.nrows; ++k) {
      if (k != i && lower[k] == upper[k] && std::isfinite(upper[k]) && rows[k].count(j) &&
          lines(k).size() == 1) {
        return k;
      }
    }
    return -1;
  };
  // Every step takes a line column from a row and gives it none, so the steps end.
  for (bool changed = true; changed;) {
    changed = false;
    for (int32_t i = 0; i < model.nrows && !changed; ++i) {
      std::vector<int32_t> cols = lines(i);
      if (cols.size() < 2) { continue; }
      for (int32_t j : cols) {
        int32_t k = equation_of(i, j);
        if (k < 0) { continue; }
        double multiple = rows[i][j] / rows[k][j];
        for (const auto& entry : rows[k]) {
          if (entry.first == j) { continue; }
          double moved = multiple * entry.second;
          auto mine    = rows[i].find(entry.first);
          double old   = mine == rows[i].end() ? 0. : mine->second;
          double sum   = old - moved;
          if (std::fabs(sum) <= cancellation * (std::fabs(old) + std::fabs(moved))) {
            if (mine != rows[i].end()) { rows[i].erase(mine); }
          } else {
            rows[i][entry.first] = sum;
          }
        }
        rows[i].erase(j);
        if (std::isfinite(lower[i])) { lower[i] -= multiple * upper[k]; }
        if (std::isfinite(upper[i])) { upper[i] -= multiple * upper[k]; }
        changed = true;
        break;
      }
    }
  }
  out = model;
  out.row_start.assign(1, 0);
  out.col_index.clear();
  out.value.clear();
  for (int32_t i = 0; i < model.nrows; ++i) {
    if (lines(i).size() > 1) { return false; }
    for (const auto& entry : rows[i]) {
      if (!std::isfinite(entry.second)) { return false; }
      out.col_index.push_back(entry.first);
      out.value.push_back(entry.second);
    }
    out.row_start.push_back(out.value.size());
  }
  out.row_lower = lower;
  out.row_upper = upper;
  return well_formed(out);
}

}  // namespace

bool classify_widest(const model_t& model, double max_assignments, std::vector<uint8_t>& line_col)
{
  bool found = false;
  std::vector<uint8_t> trial;
  for (int32_t limit : width_limits(model)) {
    if (!classify(model, limit, trial) || assignments(model, trial) > max_assignments) { continue; }
    line_col = trial;
    found    = true;
  }
  return found;
}

bool into_class(const model_t& model,
                double max_assignments,
                model_t& in_class,
                std::vector<uint8_t>& line_col)
{
  if (classify_widest(model, max_assignments, line_col)) {
    in_class = model;
    return true;
  }
  bool found = false;
  std::vector<uint8_t> trial;
  model_t reduced;
  for (int32_t limit : width_limits(model)) {
    classify(model, limit, trial);
    if (assignments(model, trial) > max_assignments) { continue; }
    if (!reduce_line_columns(model, trial, reduced)) { continue; }
    in_class = reduced;
    line_col = trial;
    found    = true;
  }
  return found;
}

solution_t solve(const model_t& model, const std::vector<uint8_t>& line_col, int32_t keep_points)
{
  gprs_assert(well_formed(model));
  gprs_assert((int32_t)line_col.size() == model.ncols);
  int32_t nrows = model.nrows, ncols = model.ncols;

  std::vector<int32_t> enumerated;
  std::vector<double> first, last;
  for (int32_t j = 0; j < ncols; ++j) {
    if (line_col[j]) { continue; }
    gprs_assert(model.col_type[j] == 1);
    gprs_assert(std::isfinite(model.col_lower[j]) && std::isfinite(model.col_upper[j]));
    enumerated.push_back(j);
    first.push_back(first_integer(model.col_lower[j]));
    last.push_back(last_integer(model.col_upper[j]));
  }

  std::vector<int32_t> row_line(nrows, -1);
  std::vector<double> row_line_value(nrows, 0.);
  std::vector<uint8_t> row_exact(nrows);
  for (int32_t i = 0; i < nrows; ++i) {
    double lower = model.row_lower[i], upper = model.row_upper[i];
    row_exact[i] = is_integer(lower) && is_integer(upper);
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      int32_t j = model.col_index[e];
      if (line_col[j]) {
        gprs_assert(row_line[i] < 0);
        row_line[i]       = j;
        row_line_value[i] = model.value[e];
      }
      // An end of an integer line column is rounded inwards, so its coefficient counts as well.
      // An end of a continuous one is a side divided by its coefficient: with a coefficient
      // that a division left (5 / 3 as a double), it lies one rounding step off the end that an
      // integer row gives the same column.
      bool exact = !line_col[j] || model.col_type[j] == 1 ? is_integer(model.value[e])
                                                          : few_bits(model.value[e]);
      if (!exact) { row_exact[i] = 0; }
    }
  }

  solution_t solution;
  bool unbounded = false, has_best = false;
  double best = 0.;
  std::vector<double> x(ncols, 0.), far(ncols, 0.), lower_end(ncols), upper_end(ncols);
  // Tolerance of each end: zero when the end is an exact bound or comes from an exact row, so
  // that a huge exact bound is never moved.
  std::vector<double> lower_slack(ncols), upper_slack(ncols);
  auto slack_of = [](double end, bool exact) {
    return exact || !std::isfinite(end) ? 0. : solve_tolerance * std::max(1., std::fabs(end));
  };

  // Completes the assignment of the enumerated columns held in x. False when it is infeasible;
  // ray is set when a line column can move without limit in the direction of its cost.
  auto complete = [&](bool& ray) {
    for (int32_t j = 0; j < ncols; ++j) {
      lower_end[j]   = model.col_lower[j];
      upper_end[j]   = model.col_upper[j];
      lower_slack[j] = slack_of(lower_end[j], on_half_grid(lower_end[j]));
      upper_slack[j] = slack_of(upper_end[j], on_half_grid(upper_end[j]));
    }
    for (int32_t i = 0; i < nrows; ++i) {
      double activity = 0., magnitude = 0.;
      for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
        int32_t j = model.col_index[e];
        if (line_col[j]) { continue; }
        double term = model.value[e] * x[j];
        activity += term;
        magnitude += std::fabs(term);
      }
      double lower = model.row_lower[i], upper = model.row_upper[i];
      if (row_line[i] < 0) {
        // Each side with its own size: a huge side must not loosen the other one.
        auto slack = [&](double side) {
          if (row_exact[i] || !std::isfinite(side)) { return 0.; }
          return solve_tolerance * std::max({1., magnitude, std::fabs(side)});
        };
        if (activity < lower - slack(lower) || activity > upper + slack(upper)) { return false; }
        continue;
      }
      int32_t j = row_line[i];
      double a  = row_line_value[i];
      // An infinite side gives the infinite end of the right sign for either sign of a.
      double from = (lower - activity) / a, to = (upper - activity) / a;
      if (a < 0) { std::swap(from, to); }
      if (from > lower_end[j]) {
        lower_end[j]   = from;
        lower_slack[j] = slack_of(from, row_exact[i]);
      }
      if (to < upper_end[j]) {
        upper_end[j]   = to;
        upper_slack[j] = slack_of(to, row_exact[i]);
      }
    }
    ray = false;
    for (int32_t j = 0; j < ncols; ++j) {
      if (!line_col[j]) { continue; }
      double from = lower_end[j], to = upper_end[j];
      if (model.col_type[j] == 1) {
        from = std::ceil(from - lower_slack[j]);
        to   = std::floor(to + upper_slack[j]);
        if (from > to) { return false; }
      } else {
        if (from > to + std::max(lower_slack[j], upper_slack[j])) { return false; }
        if (from > to) { from = to = 0.5 * (from + to); }
      }
      double cost = model.cost[j];
      if (cost < 0) {
        ray  = ray || to == inf;
        x[j] = std::isfinite(to) ? to : std::isfinite(from) ? from : 0.;
      } else {
        ray  = ray || (cost > 0 && from == -inf);
        x[j] = std::isfinite(from) ? from : std::isfinite(to) ? to : 0.;
      }
      double step = far_step * std::max(1., std::fabs(x[j]));
      if (from == to) {
        far[j] = x[j];
      } else if (x[j] == to) {
        far[j] = std::isfinite(from) ? from : to - step;
      } else {
        far[j] = std::isfinite(to) ? to : x[j] + step;
      }
    }
    return true;
  };

  int32_t n_enum = enumerated.size();
  bool has_point = true;
  for (int32_t k = 0; k < n_enum; ++k) {
    has_point        = has_point && first[k] <= last[k];
    x[enumerated[k]] = first[k];
  }
  while (has_point) {
    bool ray = false;
    if (complete(ray)) {
      if ((int32_t)solution.feasible.size() < keep_points) {
        solution.feasible.push_back(x);
        solution.far.push_back(x);
        for (int32_t j = 0; j < ncols; ++j) {
          if (line_col[j]) { solution.far.back()[j] = far[j]; }
        }
      }
      double objective = model.objective_offset;
      for (int32_t j = 0; j < ncols; ++j) {
        objective += model.cost[j] * x[j];
      }
      unbounded = unbounded || ray;
      if (!ray && (!has_best || objective < best)) {
        has_best   = true;
        best       = objective;
        solution.x = x;
      }
    }
    int32_t k = n_enum - 1;
    while (k >= 0 && x[enumerated[k]] == last[k]) {
      x[enumerated[k]] = first[k];
      --k;
    }
    if (k < 0) { break; }
    x[enumerated[k]] += 1.;
  }

  if (unbounded) {
    solution.outcome = outcome_t::unbounded;
    solution.x.clear();
  } else if (has_best) {
    solution.outcome   = outcome_t::optimal;
    solution.objective = best;
  }
  return solution;
}

namespace {

const char* outcome_name(outcome_t outcome)
{
  switch (outcome) {
    case outcome_t::optimal: return "optimal";
    case outcome_t::infeasible: return "infeasible";
    default: return "unbounded";
  }
}

bool close(double a, double b)
{
  return std::fabs(a - b) <= 1e-9 * std::max({1., std::fabs(a), std::fabs(b)});
}

verdict_t failure(const std::string& message) { return verdict_t{false, message}; }

std::string point_text(const std::vector<double>& x) { return numbers(x); }

std::string violation_text(const point_check_t& check)
{
  return "row violation " + number(check.row_violation) + ", bound violation " +
         number(check.bound_violation) + ", integrality violation " +
         number(check.integrality_violation);
}

// origin names distinct indices below size, one per row or column of the reduced model.
bool origin_valid(const std::vector<int32_t>& origin, int32_t reduced_size, int32_t size)
{
  if ((int32_t)origin.size() != reduced_size) { return false; }
  std::vector<uint8_t> seen(std::max(size, 0), 0);
  for (int32_t index : origin) {
    if (index < 0 || index >= size || seen[index]) { return false; }
    seen[index] = 1;
  }
  return true;
}

// Indices of at most checked_points of n points, evenly spaced.
std::vector<int64_t> sample(int64_t n)
{
  int64_t count = std::min<int64_t>(n, checked_points);
  std::vector<int64_t> picked(count);
  for (int64_t k = 0; k < count; ++k) {
    picked[k] = k * n / count;
  }
  return picked;
}

}  // namespace

verdict_t check_result(const case_t& c, const options_t& options, const result_t& result)
{
  gprs_assert(well_formed(c.base) && well_formed(c.model));
  gprs_assert((int32_t)c.col_scale.size() == c.base.ncols);
  int32_t keep = (int32_t)check_max_assignments;

  std::vector<uint8_t> base_line;
  if (!classify_widest(c.base, check_max_assignments, base_line)) {
    return failure("the base model is outside the oracle class");
  }
  solution_t base = solve(c.base, base_line, keep);

  if (result.status == status_t::failed) { return failure("presolve returned status failed"); }
  if (result.status == status_t::infeasible) {
    if (base.outcome == outcome_t::infeasible) { return verdict_t(); }
    return failure(std::string("presolve returned infeasible, the base model is ") +
                   outcome_name(base.outcome));
  }
  if (result.status == status_t::unbounded_or_infeasible) {
    if (base.outcome != outcome_t::optimal) { return verdict_t(); }
    return failure("presolve returned unbounded_or_infeasible, the base model is optimal with " +
                   number(base.objective));
  }

  const model_t& reduced = result.reduced;
  const postsolve_t& ps  = result.postsolve;
  if (!well_formed(reduced)) { return failure("the reduced model is not well_formed"); }
  if (ps.nrows != c.model.nrows || ps.ncols != c.model.ncols) {
    return failure("the postsolve stack does not have the sizes of the model");
  }
  if (!origin_valid(ps.row_origin, reduced.nrows, ps.nrows) ||
      !origin_valid(ps.col_origin, reduced.ncols, ps.ncols)) {
    return failure(
      "row_origin / col_origin are not distinct rows / columns of the model, one "
      "per row / column of the reduced model");
  }
  std::vector<uint8_t> reduced_line;
  model_t reduced_in_class;
  if (!into_class(reduced, check_max_assignments, reduced_in_class, reduced_line)) {
    return failure("the reduced model is outside the oracle class");
  }
  solution_t red = solve(reduced_in_class, reduced_line, keep);
  if (red.outcome != base.outcome) {
    return failure(std::string("the reduced model is ") + outcome_name(red.outcome) +
                   ", the base model is " + outcome_name(base.outcome));
  }

  // Postsolves a point of the reduced model; an empty text when it is feasible for c.model.
  std::vector<double> x;
  double objective = 0.;
  auto postsolved  = [&](const std::vector<double>& x_reduced) -> std::string {
    ps.postsolve(x_reduced, x);
    if ((int32_t)x.size() != c.model.ncols) { return "the postsolved point has the wrong size"; }
    point_check_t check = check_point(c.model, x);
    objective           = check.objective;
    if (point_feasible(check, options.tol)) { return std::string(); }
    return "reduced point " + point_text(x_reduced) + " postsolves to " + point_text(x) +
           ", infeasible for the model: " + violation_text(check);
  };

  if (red.outcome == outcome_t::optimal) {
    if (!close(red.objective, base.objective)) {
      return failure("the reduced optimum is " + number(red.objective) + ", the base optimum " +
                     number(base.objective));
    }
    std::string error = postsolved(red.x);
    if (!error.empty()) { return failure("optimal point: " + error); }
    if (!close(objective, base.objective)) {
      return failure("the postsolved optimal point " + point_text(x) + " has objective " +
                     number(objective) + ", the base optimum is " + number(base.objective));
    }
  }
  for (int64_t k : sample(red.feasible.size())) {
    std::string error = postsolved(red.feasible[k]);
    if (!error.empty()) { return failure("feasible point " + std::to_string(k) + ": " + error); }
    error = postsolved(red.far[k]);
    if (!error.empty()) { return failure("far point " + std::to_string(k) + ": " + error); }
  }

  verdict_t verdict;
  if (base.outcome == outcome_t::infeasible) { return verdict; }
  if (options.primal_only && result.dual_findings != 0) {
    return failure("a result of primal-only mode reports " + std::to_string(result.dual_findings) +
                   " dual findings");
  }
  // Without a dual finding every reduction keeps every feasible point and its objective value.
  bool exact = result.dual_findings == 0;
  std::vector<const std::vector<double>*> points;
  if (base.outcome == outcome_t::optimal) { points.push_back(&base.x); }
  for (int64_t k : sample(base.feasible.size())) {
    points.push_back(&base.feasible[k]);
    points.push_back(&base.far[k]);
  }
  std::vector<double> x_model(c.model.ncols), x_reduced;
  for (const std::vector<double>* point : points) {
    for (int32_t j = 0; j < c.model.ncols; ++j) {
      x_model[j] = (*point)[j] / c.col_scale[j];
    }
    // A point of the base model with a huge value is no point of the scaled model in doubles.
    point_check_t model_check = check_point(c.model, x_model);
    if (!point_feasible(model_check, options.tol)) {
      ++verdict.skipped_points;
      continue;
    }
    ++verdict.crushed_points;
    ps.crush(x_model, x_reduced);
    if ((int32_t)x_reduced.size() != reduced.ncols) {
      return failure("the crushed point has the wrong size");
    }
    point_check_t check = check_point(reduced, x_reduced);
    if (!point_feasible(check, options.tol)) {
      return failure("feasible point " + point_text(x_model) + " of the model crushes to " +
                     point_text(x_reduced) +
                     ", infeasible for the reduced model: " + violation_text(check));
    }
    double original = model_check.objective;
    if (close(check.objective, original)) { continue; }
    if (exact) {
      return failure("feasible point " + point_text(x_model) + " of the model has objective " +
                     number(original) + ", its crushed point " + number(check.objective));
    }
    if (check.objective > original) {
      return failure("feasible point " + point_text(x_model) + " of the model has objective " +
                     number(original) + ", its crushed point the higher " +
                     number(check.objective));
    }
  }
  return verdict;
}

verdict_t check_presolve(const case_t& c, const options_t& options)
{
  return check_result(c, options, presolve(c.model, options));
}

}  // namespace oracle
}  // namespace gprs
