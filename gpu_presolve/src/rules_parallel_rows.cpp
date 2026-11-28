/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iterator>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace gprs {

namespace {

constexpr rule_outcome_t ok         = rule_outcome_t::ok;
constexpr rule_outcome_t infeasible = rule_outcome_t::infeasible;

// A normalised coefficient enters the hash with its mantissa rounded to this many bits: far
// coarser than tol.zero, so that two rows that are parallel within the tolerance differ in
// their hash only when a coefficient lies on a rounding boundary.
constexpr int32_t hash_mantissa_bits = 20;

inline uint64_t mix(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

inline uint64_t entry_hash(int32_t col, double normalised)
{
  // The fields of a double.
  constexpr int32_t fraction_bits = 52, sign_bit = 63, exponent_values = 2048;
  constexpr int32_t exponent_of_half = 1022;
  constexpr int32_t dropped_bits     = fraction_bits + 1 - hash_mantissa_bits;
  constexpr uint64_t one             = 1;
  int exponent                       = 0;
  int64_t grid                       = 0;
  uint64_t bits;
  std::memcpy(&bits, &normalised, sizeof(bits));
  int32_t biased = (bits >> fraction_bits) % exponent_values;
  if (biased != 0 && biased != exponent_values - 1) {
    // A normal number, where the bits give what frexp, ldexp and llround give below: the
    // mantissa in [1/2, 1) is 1/2 + fraction / 2^53, and llround rounds a tie away from zero.
    uint64_t fraction = bits % (one << fraction_bits);
    grid =
      (1 << (hash_mantissa_bits - 1)) + ((fraction + (one << (dropped_bits - 1))) >> dropped_bits);
    exponent = biased - exponent_of_half;
    if (bits >> sign_bit) { grid = -grid; }
  } else if (std::isfinite(normalised) && normalised != 0.) {
    grid = std::llround(std::ldexp(std::frexp(normalised, &exponent), hash_mantissa_bits));
  }
  // A mantissa that rounds up to 1 is the mantissa 1/2 of the next exponent.
  if (grid == 1 << hash_mantissa_bits || grid == -(1 << hash_mantissa_bits)) {
    grid /= 2;
    ++exponent;
  }
  return mix(mix(mix(col) ^ grid) ^ exponent);
}

// A singleton column is a live column with one live entry. With `near` set, the core of a row
// is its free entries outside singleton columns; without it, all its free entries.
inline bool in_core(const state_t& st, int32_t col, bool near)
{
  return !near || st.csc.size[col] > 1;
}

struct row_key_t {
  uint64_t hash  = 0;
  int32_t length = 0;
};

// Hash and number of the core entries. The hash is a sum over them, so the storage order does
// not enter; a multiple of the row has the same normalised coefficients.
row_key_t row_key(const state_t& st, int32_t row, bool near)
{
  const line_store_t& csr = st.csr;
  int32_t first           = st.ncols;
  double first_value      = 0.;
  row_key_t key;
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if (!st.is_live(col) || !in_core(st, col, near)) { continue; }
    ++key.length;
    if (col < first) {
      first       = col;
      first_value = csr.value[e];
    }
  }
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if (!st.is_live(col) || !in_core(st, col, near)) { continue; }
    key.hash += entry_hash(col, csr.value[e] / first_value);
  }
  return key;
}

struct bucket_key_t {
  uint64_t hash;
  int32_t length;
  int32_t row;
};

inline bool operator<(const bucket_key_t& a, const bucket_key_t& b)
{
  if (a.hash != b.hash) { return a.hash < b.hash; }
  if (a.length != b.length) { return a.length < b.length; }
  return a.row < b.row;
}

inline bool same_bucket(const bucket_key_t& a, const bucket_key_t& b)
{
  return a.hash == b.hash && a.length == b.length;
}

constexpr int32_t no_row = -1;

// The keys of all rows are sorted afresh when the keys set since the last sort reach one in
// this many rows; below that, each new key searches its bucket. The search of one key costs
// about as much as the sort costs for this many rows.
constexpr size_t sort_share = 32;

// Keys up to which the comparison sort alone is used.
constexpr size_t comparison_sort_size = 64;

// Orders the keys by hash, free length, row. Stable counting passes on the leading bytes of
// the hash, the lowest of them first, order the keys by those bytes, as many bytes as tell n
// hashes that spread over their range apart; each run of equal leading bytes is then sorted by
// comparison. With hashes that share their leading bytes the comparison sort does all the
// work, so the cost is O(n log n) for any hashes and O(n) for spread ones.
void sort_keys(std::vector<bucket_key_t>& keys)
{
  constexpr int32_t byte_bits = 8, hash_bits = 64, max_passes = 4;
  constexpr size_t byte_values = 256;
  size_t n                     = keys.size();
  int32_t passes               = 0;
  if (n > comparison_sort_size) {
    for (size_t reach = 1; reach < n; reach *= byte_values) {
      ++passes;
    }
  }
  // A row index has 31 bits.
  gprs_assert(passes <= max_passes);
  int32_t low = hash_bits - byte_bits * passes;
  if (passes > 0) {
    std::array<std::array<size_t, byte_values>, max_passes> count{};
    for (const bucket_key_t& key : keys) {
      for (int32_t p = 0; p < passes; ++p) {
        ++count[p][(key.hash >> (low + byte_bits * p)) % byte_values];
      }
    }
    std::vector<bucket_key_t> other(n);
    for (int32_t p = 0; p < passes; ++p) {
      int32_t shift                         = low + byte_bits * p;
      std::array<size_t, byte_values>& next = count[p];
      if (next[(keys[0].hash >> shift) % byte_values] == n) { continue; }
      size_t sum = 0;
      for (size_t& slot : next) {
        size_t size = slot;
        slot        = sum;
        sum += size;
      }
      for (const bucket_key_t& key : keys) {
        other[next[(key.hash >> shift) % byte_values]++] = key;
      }
      keys.swap(other);
    }
  }
  uint64_t lead = 0;
  if (passes > 0) { lead = ~lead << low; }
  for (size_t begin = 0; begin < n;) {
    size_t end = begin + 1;
    while (end < n && ((keys[end].hash ^ keys[begin].hash) & lead) == 0) {
      ++end;
    }
    if (end - begin > 1) { std::sort(keys.begin() + begin, keys.begin() + end); }
    begin = end;
  }
}

// What an addition reads of one of two rows with parallel cores.
struct near_row_t {
  int32_t row = -1;
  // Coefficient in the smallest core column.
  double first = 0.;
  // Free entries in singleton columns, the last of them, and the smallest |a| among them.
  int32_t singles     = 0;
  int32_t single_col  = -1;
  double single_value = 0.;
  double single_min   = inf;
  dd_t fixed_terms;
};

// The core entries of one row scattered over the columns.
struct scattered_row_t {
  near_row_t self;
  int32_t length = 0;  // core entries
  int32_t first  = 0;  // smallest core column
};

// One live entry of a row: a constant term, an entry of a singleton column, or a core entry
// (true).
inline bool core_entry(const state_t& st, int64_t e, bool near, near_row_t& of)
{
  int32_t col = st.csr.index[e];
  if (!st.is_live(col)) {
    dd_add_product(of.fixed_terms, st.csr.value[e], st.col_lower[col]);
    return false;
  }
  if (!in_core(st, col, near)) {
    ++of.singles;
    of.single_col   = col;
    of.single_value = st.csr.value[e];
    of.single_min   = std::fmin(of.single_min, std::fabs(st.csr.value[e]));
    return false;
  }
  return true;
}

// cols gets the core columns: the row can be deleted before its scatter is cleared.
scattered_row_t scatter(const state_t& st,
                        int32_t row,
                        bool near,
                        std::vector<double>& coefficient,
                        std::vector<int32_t>& cols)
{
  const line_store_t& csr = st.csr;
  scattered_row_t s;
  s.self.row = row;
  s.first    = st.ncols;
  cols.clear();
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e) || !core_entry(st, e, near, s.self)) { continue; }
    int32_t col      = csr.index[e];
    coefficient[col] = csr.value[e];
    cols.push_back(col);
    if (col < s.first) {
      s.first      = col;
      s.self.first = csr.value[e];
    }
  }
  s.length = cols.size();
  return s;
}

struct comparison_t {
  bool parallel = false;
  // larger row = ratio * smaller row, on the core.
  double ratio = 0.;
  // The cores are equal entry by entry, or one is the negative of the other.
  bool identical = false;
  near_row_t other;
};

// Exact test of the core of `other` against the scattered row, on the entries as they are
// stored now.
comparison_t compare(const state_t& st,
                     const tolerances_t& tol,
                     const scattered_row_t& s,
                     const std::vector<double>& coefficient,
                     int32_t other,
                     bool near)
{
  const line_store_t& csr = st.csr;
  comparison_t result;
  result.other.row = other;
  int32_t length   = 0;
  for (int64_t e = csr.begin(other); e < csr.end(other); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    // A column of the scattered core that `other` holds as well is live and has two entries.
    if (coefficient[col] == 0.) {
      if (core_entry(st, e, near, result.other)) { return result; }
      continue;
    }
    ++length;
    if (col == s.first) { result.other.first = csr.value[e]; }
  }
  // Every core column of `other` is one of the scattered row, so equal lengths are equal sets.
  if (length != s.length) { return result; }
  bool scattered_is_smaller = s.self.row < other;
  result.ratio =
    scattered_is_smaller ? result.other.first / s.self.first : s.self.first / result.other.first;
  if (!std::isfinite(result.ratio) || result.ratio == 0.) { return result; }
  bool identical = std::fabs(result.ratio) == 1.;
  for (int64_t e = csr.begin(other); e < csr.end(other); ++e) {
    if (!csr.live(e) || coefficient[csr.index[e]] == 0.) { continue; }
    double a = coefficient[csr.index[e]], b = csr.value[e];
    if (!scattered_is_smaller) { std::swap(a, b); }
    double off = std::fabs(b - result.ratio * a);
    if (!(off <= tol.zero * std::fabs(b))) { return result; }
    identical = identical && off == 0.;
  }
  result.parallel  = true;
  result.identical = identical;
  return result;
}

// A side or a fixed term of this size is stored, as one double, with an error far below the
// feasibility tolerance; a larger one takes more than the tolerance from a value next to it
// (as holds_tolerance in the row tightening).
inline bool holds_tolerance(double size, const tolerances_t& tol)
{
  return size * tol.side_error <= tol.feasibility;
}

// A side of the kept row is stored as one double with the fixed terms of that row in it, and
// the sides of the two rows are compared in that form.
inline bool keeps_sides(dd_t fixed_terms, const tolerances_t& tol)
{
  return holds_tolerance(std::fabs(dd_value(fixed_terms)), tol);
}

// Of two parallel rows without singleton columns the one with the larger coefficients is kept,
// read from the smallest core column, and the one with the smaller index when the two are equal
// in size: the other row is then the kept one times a ratio of at most 1 in size, and what a
// later rule or the postsolve rounds in the kept row reaches it no larger. With keep_larger off
// the row with the smaller index is kept whatever the sizes.
inline bool first_is_kept(const near_row_t& a, const near_row_t& b, bool keep_larger)
{
  double size_a = std::fabs(a.first), size_b = std::fabs(b.first);
  if (keep_larger && size_a != size_b) { return size_a > size_b; }
  return a.row < b.row;
}

// value = mantissa * 2^exponent with 1/2 <= |mantissa| < 1: a product of two mantissas neither
// overflows nor underflows, whatever the sizes of the values are.
struct parts_t {
  double mantissa;
  int exponent;
};

inline parts_t parts_of(double value)
{
  parts_t parts;
  parts.mantissa = std::frexp(value, &parts.exponent);
  return parts;
}

// What the removed row can differ by from its multiple of the kept row, in its own units, over
// the explicit bounds of the core columns. With a and b the core coefficients of the kept and
// of the removed row, and af and bf those of the smallest core column, the removed row is the
// kept one times bf / af plus the differences b - a * bf / af, each of which is read without
// rounding from the two products b * af and a * bf. The drift is the sum of their sizes times
// max(|lower|, |upper|) of their columns: zero for two rows that are multiples of each other
// as they are stored, infinite when a difference belongs to a column without a finite bound.
// `scattered` holds the core of one of the two rows by column; `other` is the other row.
double core_drift(const state_t& st,
                  const std::vector<double>& scattered,
                  int32_t other,
                  bool scattered_is_kept,
                  double kept_first,
                  double removed_first)
{
  const line_store_t& csr = st.csr;
  parts_t af = parts_of(kept_first), bf = parts_of(removed_first);
  double drift = 0.;
  for (int64_t e = csr.begin(other); e < csr.end(other); ++e) {
    int32_t col = csr.index[e];
    if (!csr.live(e) || scattered[col] == 0.) { continue; }
    parts_t a = parts_of(scattered[col]), b = parts_of(csr.value[e]);
    if (!scattered_is_kept) { std::swap(a, b); }
    // b * af - a * bf = (of_removed * 2^shift - of_kept) * 2^(a.exponent + bf.exponent), with
    // the errors of the two products: 0 exactly when the two are equal.
    double removed_error, kept_error;
    double of_removed = two_prod(b.mantissa, af.mantissa, removed_error);
    double of_kept    = two_prod(a.mantissa, bf.mantissa, kept_error);
    int shift         = b.exponent + af.exponent - a.exponent - bf.exponent;
    double difference =
      (std::ldexp(of_removed, shift) - of_kept) + (std::ldexp(removed_error, shift) - kept_error);
    if (difference == 0.) { continue; }
    double size = std::ldexp(difference / af.mantissa, a.exponent + bf.exponent - af.exponent);
    drift +=
      std::fabs(size) * std::fmax(std::fabs(st.col_lower[col]), std::fabs(st.col_upper[col]));
  }
  return drift;
}

// A side of the removed row as a side of the kept row.
struct image_t {
  double value = 0.;
  // What the value can be off by, in the units of the kept row; 0 for a value without rounding.
  double error = 0.;
  // False for a value that is not finite, and for one that is rounded at a size at which a
  // double does not hold the tolerance of the removed row.
  bool usable = true;
  // The only rounding is that of the fixed terms of the kept row, which can leave that row
  // without an event (state_t::absorb_fixed_terms).
  bool rounded_by_fixed_terms = false;
};

// Rounding units of its size that a rounded image can be off by: the side less the fixed
// terms, the product, the quotient with its correction, the fixed terms of the kept row as one
// double, and the sum.
constexpr double image_error_units = 8.;

// A side below this, other than zero, has a product with a mantissa whose error is no double.
constexpr double min_scaled_side = 0x1p-960;

// The side less removed_fixed, times kept_first / removed_first, plus kept_fixed. The quotient
// takes the error of the product, so that the image is exact wherever it is a double (a side
// times 3, a side of a row that is 7 / 3 of the other) and is the side itself for two rows
// with equal coefficients: a quotient by the rounded ratio of the two rows would be off the
// integer by a rounding unit there, and that reads as a crossing of two sides that meet.
image_t image_of(double side,
                 dd_t removed_fixed,
                 dd_t kept_fixed,
                 double kept_first,
                 double removed_first,
                 const tolerances_t& tol)
{
  image_t image;
  if (!std::isfinite(side)) {
    image.value = (kept_first < 0) == (removed_first < 0) ? side : -side;
    return image;
  }
  double residual_error;
  double difference = two_sum(side, -removed_fixed.hi, residual_error);
  double rest       = residual_error - removed_fixed.lo;
  double free_side  = difference + rest;
  double scaled     = kept_first == removed_first ? free_side : -free_side;
  bool exact        = rest == 0.;
  if (std::fabs(kept_first) != std::fabs(removed_first)) {
    if (free_side != 0. && std::fabs(free_side) < min_scaled_side) {
      image.usable = false;
      return image;
    }
    parts_t numerator = parts_of(kept_first), denominator = parts_of(removed_first);
    double product_error, back_error;
    double product   = two_prod(free_side, numerator.mantissa, product_error);
    double quotient  = product / denominator.mantissa;
    double remainder = std::fma(-quotient, denominator.mantissa, product) + product_error;
    double mantissa  = quotient + remainder / denominator.mantissa;
    int exponent     = numerator.exponent - denominator.exponent;
    scaled           = std::ldexp(mantissa, exponent);
    double back      = two_prod(mantissa, denominator.mantissa, back_error);
    exact            = exact && back == product && back_error == product_error &&
            std::ldexp(scaled, -exponent) == mantissa;
  }
  double fixed = dd_value(kept_fixed);
  double sum_error;
  image.value = two_sum(scaled, fixed, sum_error);
  if (!std::isfinite(image.value)) {
    image.usable = false;
    return image;
  }
  bool exact_sum = sum_error == 0. && kept_fixed.lo == 0.;
  if (exact && exact_sum) { return image; }
  image.error  = image_error_units * rounding_unit * std::fmax(std::fabs(scaled), std::fabs(fixed));
  image.usable = holds_tolerance(std::fabs(free_side), tol);
  image.rounded_by_fixed_terms = exact;
  return image;
}

// What became of two parallel rows without singleton columns.
enum class merge_t : uint8_t {
  made,
  infeasible,
  // Nothing changed, and nothing would with the same rows in another presolve.
  left,
  // Nothing changed, by a test that can pass later without an event of the two rows: a bound
  // of a core column that moves, a fixed term that leaves the kept row.
  left_for_now
};

// removed = ratio * kept on the free entries, up to `drift` (core_drift): the sides of
// `removed` become sides of `kept`. The caller has tested keeps_sides(kept.fixed_terms).
// Where two sides cross, the removed row is off by the crossing times |ratio|, in its own
// units, when the kept row holds at its side, and by the drift on top. That is the measure:
// infeasible when it is above the tolerance whatever the rounding of the image and the drift
// do, a merge with the new side at the opposite side of the kept row when it is within the
// tolerance whatever they do, and no merge between the two.
merge_t merge(state_t& st,
              postsolve_t& ps,
              const tolerances_t& tol,
              const near_row_t& kept,
              const near_row_t& removed,
              double drift)
{
  gprs_assert(keeps_sides(kept.fixed_terms, tol));
  double ratio = removed.first / kept.first;
  if (!std::isfinite(ratio) || ratio == 0.) { return merge_t::left; }
  double scale = std::fabs(ratio);
  auto image   = [&](double side) {
    return image_of(side, removed.fixed_terms, kept.fixed_terms, kept.first, removed.first, tol);
  };
  image_t lower = image(st.row_lower[removed.row]), upper = image(st.row_upper[removed.row]);
  if (ratio < 0) { std::swap(lower, upper); }
  if (!lower.usable || !upper.usable) {
    bool for_now = (!lower.usable && lower.rounded_by_fixed_terms) ||
                   (!upper.usable && upper.rounded_by_fixed_terms);
    return for_now ? merge_t::left_for_now : merge_t::left;
  }
  gprs_assert(lower.value <= upper.value);
  double kept_lower = st.row_lower[kept.row], kept_upper = st.row_upper[kept.row];
  // What is done with two sides that cross by `crossing`, in the units of the kept row.
  auto closes = [&](double crossing, double error) {
    if (scale * (crossing - error) - drift > tol.feasibility) { return merge_t::infeasible; }
    if (scale * (crossing + error) + drift <= tol.feasibility) { return merge_t::made; }
    return merge_t::left_for_now;
  };
  double new_lower = kept_lower, new_upper = kept_upper;
  if (lower.value > kept_lower) {
    new_lower = lower.value;
    if (lower.value > kept_upper) {
      merge_t step = closes(lower.value - kept_upper, lower.error);
      if (step != merge_t::made) { return step; }
      new_lower = kept_upper;
    }
  }
  if (upper.value < kept_upper) {
    new_upper = upper.value;
    if (upper.value < new_lower) {
      merge_t step = closes(new_lower - upper.value, upper.error);
      if (step != merge_t::made) { return step; }
      new_upper = new_lower;
    }
  }
  if (new_lower != kept_lower) { st.set_row_lower(kept.row, new_lower); }
  if (new_upper != kept_upper) { st.set_row_upper(kept.row, new_upper); }
  st.delete_row(removed.row);
  ps.push_parallel_row(removed.row, kept.row, ratio);
  return merge_t::made;
}

inline bool is_equation(const state_t& st, int32_t row)
{
  return st.row_lower[row] == st.row_upper[row];
}

// Whether `equation` may be subtracted from `target`, whose core is parallel to its own. The
// target is left with its own singleton columns and that of the equation: with at most two
// entries, or with a part of the entries it had.
inline bool may_subtract(const state_t& st, const near_row_t& equation, const near_row_t& target)
{
  return is_equation(st, equation.row) &&
         (equation.singles == 0 || (equation.singles == 1 && target.singles <= 1));
}

// An addition is left out when it would raise max |a| / min |a| of the target above this and
// above what it was.
constexpr double max_dynamism = 1e6;

// An addition is left out when it would multiply the error of a row by more than this. With
// it, sides and fixed terms up to 1e5 are out of reach of the limits on their size below
// (tol.feasibility / tol.side_error, 1e8 by default), so that the terms of fixed columns that
// leave a row plainly (state_t::absorb_fixed_terms) change nothing for a pair.
constexpr double max_growth = 1e3;

// What became of a pair of which one row may be subtracted from the other.
enum class subtraction_t : uint8_t {
  made,
  // Nothing changed, and nothing would with the same rows in another presolve.
  left,
  // Nothing changed, for the one reason that the presolve has changed a row whose error the
  // step would multiply: a presolve that is given these rows as its model makes the step.
  left_to_a_changed_row
};

// target -= factor * equation with factor * core of equation = core of target: the core
// entries of the target leave, the entry of the equation's singleton column enters with
// -factor times its coefficient, and the finite sides of the target move by factor times the
// right-hand side of the equation less its fixed terms. The fixed terms of the target stay in
// its entries; they are read only for the limit on their size when the step multiplies the
// error of the target.
// Nothing is changed when a value involved is too large to hold the tolerance where the step
// takes it, or the step would multiply the error of a row by more than max_growth or that of a
// row that the presolve has changed.
subtraction_t subtract(state_t& st,
                       postsolve_t& ps,
                       const tolerances_t& tol,
                       work_budget_t& budget,
                       const near_row_t& equation,
                       const near_row_t& target,
                       std::vector<int64_t>& slots)
{
  constexpr subtraction_t left = subtraction_t::left;
  gprs_assert(may_subtract(st, equation, target));
  const line_store_t& csr = st.csr;
  double factor           = target.first / equation.first;
  if (!std::isfinite(factor) || factor == 0.) { return left; }
  // From here on the target row of the model is the new target plus factor times the
  // equation: whatever is done to the equation later, within its tolerance (a side rounded, a
  // subtraction, a merge), reaches that row times |factor|. And once the singleton column of
  // the equation has been taken out through the target, the equation is what the target was,
  // divided by the factor: whatever is done to the target reaches it divided by |factor|.
  // Neither may grow.
  double scale = std::fabs(factor);
  if (scale > 1. || (equation.singles == 1 && scale != 1.)) { return left; }
  // The constants of a row are stored with an error: that of a double for a row as the model
  // gave it, and the rounding of every change for a row that the presolve has changed (a side
  // scaled next to a fixed term of 1e7 is off by 1e-9, which holds the tolerance in that row
  // and nowhere else). The error of the equation goes into each singleton column of the target
  // times |factor| and divided by the coefficient of that column; the error of the target into
  // the singleton column of the equation divided by its coefficient.
  double grow_equation = std::fmax(1., scale / target.single_min);
  double grow_target   = 1.;
  if (equation.singles == 1) { grow_target = std::fmax(1., 1. / std::fabs(equation.single_value)); }
  if (!(grow_equation <= max_growth && grow_target <= max_growth)) { return left; }
  double rhs   = dd_residual(st.row_upper[equation.row], equation.fixed_terms);
  double lower = st.row_lower[target.row], upper = st.row_upper[target.row];
  // The error of a double must hold the tolerance where it arrives. The shift is rounded at the
  // size of the right-hand side, of the fixed terms it is freed from, of the product and of the
  // side that takes it.
  if (!holds_tolerance(std::fabs(st.row_upper[equation.row]) * grow_equation, tol) ||
      !holds_tolerance(std::fabs(dd_value(equation.fixed_terms)) * grow_equation, tol) ||
      !holds_tolerance(std::fabs(factor * rhs), tol) ||
      (lower > -inf && !holds_tolerance(std::fabs(lower) * grow_target, tol)) ||
      (upper < inf && !holds_tolerance(std::fabs(upper) * grow_target, tol)) ||
      (grow_target > 1. &&
       !holds_tolerance(std::fabs(dd_value(target.fixed_terms)) * grow_target, tol))) {
    return left;
  }
  double value = equation.singles == 1 ? -factor * equation.single_value : 0.;
  if (equation.singles == 1 && (!std::isfinite(value) || value == 0.)) { return left; }
  // An error of unknown size must not grow: the row has to be as the model gave it. The one
  // test that reads what the presolve has done and not the rows as they are; it comes after
  // those that the same two rows fail as the model of another presolve.
  if ((grow_equation > 1. && st.row_version[equation.row] != 0) ||
      (grow_target > 1. && st.row_version[target.row] != 0)) {
    return subtraction_t::left_to_a_changed_row;
  }

  budget.spend(csr.used[target.row]);
  slots.clear();
  double old_min = inf, old_max = 0., new_min = inf, new_max = 0.;
  for (int64_t e = csr.begin(target.row); e < csr.end(target.row); ++e) {
    if (!csr.live(e) || !st.is_live(csr.index[e])) { continue; }
    double magnitude = std::fabs(csr.value[e]);
    old_min          = std::fmin(old_min, magnitude);
    old_max          = std::fmax(old_max, magnitude);
    if (in_core(st, csr.index[e], true)) {
      slots.push_back(e);
    } else {
      new_min = std::fmin(new_min, magnitude);
      new_max = std::fmax(new_max, magnitude);
    }
  }
  gprs_assert(slots.size() >= 2);
  if (equation.singles == 1) {
    new_min = std::fmin(new_min, std::fabs(value));
    new_max = std::fmax(new_max, std::fabs(value));
  }
  // The target keeps an entry: the rows do not both lack a singleton column.
  gprs_assert(new_max > 0.);
  if (new_max / new_min > std::fmax(old_max / old_min, max_dynamism)) { return left; }

  size_t k = 0;
  if (equation.singles == 1) {
    st.move_entry(slots[0], equation.single_col, value);
    k = 1;
  }
  for (; k < slots.size(); ++k) {
    st.delete_entry(slots[k]);
  }
  if (factor * rhs != 0. && (lower > -inf || upper < inf)) {
    st.shift_row_sides(target.row, factor, rhs);
  }
  ps.push_row_addition(target.row, equation.row, factor);
  return subtraction_t::made;
}

// What the family keeps of one row, in one place: arrays indexed by row that each start on a
// page compete for one cache set in the loop over the listed rows, and one more of them than
// the set holds costs a miss per row (a quarter of a first call, measured).
struct row_record_t {
  // The key: the hash and the number of the core entries; length 0 without one.
  uint64_t hash  = 0;
  int32_t length = 0;
  // row_free_size and row_version when the row was last read for its key, free 0 for a row
  // that has to be read again: together they tell whether the free entries are those that were
  // read. Which of them are core entries can change without them, and the row is then on
  // row_list::singletons.
  int32_t free     = 0;
  uint32_t version = 0;
  // The key is still the one in work_t::base.
  uint8_t in_base = 0;
};

}  // namespace

struct parallel_rows_t::work_t {
  std::vector<row_record_t> record;
  // opt.near_parallel_rows of the calls so far.
  bool near = false;
  // A bucket is a list of the rows with one hash and one core length, in index order: the next
  // row of the bucket above and below each row, no_row at the ends.
  std::vector<int32_t> above;
  std::vector<int32_t> below;
  // Where a new key finds its bucket. base holds the keys of the last sort, ordered by hash,
  // core length, row; recent holds the keys set since.
  std::vector<bucket_key_t> base;
  std::set<bucket_key_t> recent;
  // Over the entries of base that a search found stale: skip_up[i] leads towards the first
  // entry at or above i that can be current, skip_down[i + 1] towards the last one at or below
  // i (union-find with path compression; base.size() and -1 without one). Empty until the
  // first search after a sort.
  std::vector<int32_t> skip_up;
  std::vector<int32_t> skip_down;
  // The rows that got a new key in this call.
  std::vector<int32_t> fresh;
  // The core entries of the listed row under comparison, by column and 0 elsewhere, and its
  // core columns.
  std::vector<double> coefficient;
  std::vector<int32_t> cols;
  std::vector<int32_t> rows;
  // The core slots of the target of an addition.
  std::vector<int64_t> slots;

  bool has_key(int32_t row) const { return record[row].length != 0; }

  bucket_key_t key(int32_t row) const
  {
    return bucket_key_t{record[row].hash, record[row].length, row};
  }

  // The key of the row is the key of its entries as they are now.
  bool key_is_current(const state_t& st, int32_t row) const
  {
    row_key_t now = row_key(st, row, near);
    return now.length == record[row].length && now.hash == record[row].hash;
  }

  // Reads the listed rows whose free entries are not those that were read and sets their keys;
  // fresh gets the rows with a new key. False when the budget stopped it.
  bool key_rows(const state_t& st, work_budget_t& budget)
  {
    fresh.clear();
    for (int32_t row : rows) {
      if (budget.stop()) { return false; }
      row_record_t& r = record[row];
      int32_t free    = st.row_alive[row] && st.row_free_size[row] >= 2 ? st.row_free_size[row] : 0;
      if (free != 0 && r.free == free && r.version == st.row_version[row]) { continue; }
      if (r.length != 0) { drop(row); }
      r.free = free;
      if (free == 0) { continue; }
      budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[row]);
      row_key_t key = row_key(st, row, near);
      r.version     = st.row_version[row];
      // A core of one entry has no partner: without their singleton columns the two rows would
      // be singleton rows.
      if (key.length < 2) { continue; }
      r.hash   = key.hash;
      r.length = key.length;
      fresh.push_back(row);
    }
    return true;
  }

  void drop(int32_t row)
  {
    row_record_t& r = record[row];
    gprs_assert(r.length != 0);
    int32_t lower = below[row], upper = above[row];
    if (lower != no_row) { above[lower] = upper; }
    if (upper != no_row) { below[upper] = lower; }
    below[row] = above[row] = no_row;
    if (r.in_base) {
      r.in_base = 0;
    } else {
      size_t erased = recent.erase(key(row));
      gprs_assert(erased == 1);
    }
    r.length = 0;
  }

  // Sorts the keys of all rows and links the buckets.
  void sort_all()
  {
    int32_t nrows = record.size();
    size_t keys   = 0;
    for (row_record_t& r : record) {
      r.in_base = r.length != 0;
      keys += r.in_base;
    }
    base.clear();
    base.reserve(keys);
    for (int32_t row = 0; row < nrows; ++row) {
      if (record[row].in_base) { base.push_back(key(row)); }
    }
    sort_keys(base);
    recent.clear();
    skip_up.clear();
    skip_down.clear();
    std::fill(above.begin(), above.end(), no_row);
    std::fill(below.begin(), below.end(), no_row);
    for (size_t i = 1; i < base.size(); ++i) {
      if (!same_bucket(base[i - 1], base[i])) { continue; }
      above[base[i - 1].row] = base[i].row;
      below[base[i].row]     = base[i - 1].row;
    }
  }

  // First entry of base at or above i whose key is current; base.size() without one.
  int32_t current_from(int32_t i)
  {
    int32_t n     = base.size();
    int32_t found = i;
    while (true) {
      while (skip_up[found] != found) {
        found = skip_up[found];
      }
      if (found == n || record[base[found].row].in_base) { break; }
      skip_up[found] = found + 1;
    }
    while (skip_up[i] != found) {
      int32_t next = skip_up[i];
      skip_up[i]   = found;
      i            = next;
    }
    return found;
  }

  // Last entry of base at or below i whose key is current; -1 without one.
  int32_t current_to(int32_t i)
  {
    int32_t found = i + 1;
    while (true) {
      while (skip_down[found] != found) {
        found = skip_down[found];
      }
      if (found == 0 || record[base[found - 1].row].in_base) { break; }
      skip_down[found] = found - 1;
    }
    for (int32_t at = i + 1; skip_down[at] != found;) {
      int32_t next  = skip_down[at];
      skip_down[at] = found;
      at            = next;
    }
    return found - 1;
  }

  // Puts a row whose new key is in neither base nor recent into its bucket: between the
  // nearest rows of the bucket below and above it, which are current entries of base or
  // entries of recent.
  void insert(int32_t row)
  {
    if (skip_up.empty()) {
      skip_up.resize(base.size() + 1);
      std::iota(skip_up.begin(), skip_up.end(), 0);
      skip_down = skip_up;
    }
    bucket_key_t k = key(row);
    gprs_assert(!record[row].in_base && recent.count(k) == 0);
    auto self     = recent.insert(k).first;
    int32_t lower = no_row, upper = no_row;
    if (self != recent.begin() && same_bucket(*std::prev(self), k)) {
      lower = std::prev(self)->row;
    }
    if (std::next(self) != recent.end() && same_bucket(*std::next(self), k)) {
      upper = std::next(self)->row;
    }
    int32_t n    = base.size();
    int32_t at   = std::lower_bound(base.begin(), base.end(), k) - base.begin();
    int32_t up   = current_from(at);
    int32_t down = current_to(at - 1);
    if (up < n && same_bucket(base[up], k) && (upper == no_row || base[up].row < upper)) {
      upper = base[up].row;
    }
    if (down >= 0 && same_bucket(base[down], k) && base[down].row > lower) {
      lower = base[down].row;
    }
    gprs_assert(lower == no_row || above[lower] == upper);
    gprs_assert(upper == no_row || below[upper] == lower);
    below[row] = lower;
    above[row] = upper;
    if (lower != no_row) { above[lower] = row; }
    if (upper != no_row) { below[upper] = row; }
  }
};

parallel_rows_t::parallel_rows_t() : work(new work_t) {}

parallel_rows_t::~parallel_rows_t() = default;

rule_result_t parallel_rows_t::run(state_t& st,
                                   work_budget_t& budget,
                                   postsolve_t& ps,
                                   const options_t& opt)
{
  work_t& w = *work;
  bool near = opt.near_parallel_rows;
  if (w.record.size() != (size_t)st.nrows) {
    w.record.assign(st.nrows, row_record_t());
    w.near = near;
    w.above.assign(st.nrows, no_row);
    w.below.assign(st.nrows, no_row);
    w.base.clear();
    w.recent.clear();
    w.skip_up.clear();
    w.skip_down.clear();
  }
  // The keys of the earlier calls are keys of the same kind.
  gprs_assert(w.near == near);
  if (w.coefficient.size() != (size_t)st.ncols) { w.coefficient.assign(st.ncols, 0.); }
  const tolerances_t& tol          = opt.tol;
  const list_mask_t own_list       = list_bit(row_list::parallel_rows);
  const list_mask_t singleton_list = list_bit(row_list::singletons);
  rule_result_t result;

  // A row of row_list::singletons has another core than the one its key was taken from, with
  // the entries and the version it had: its key goes and the row is read again. The event
  // that listed it feeds the family's own list as well.
  if (near) {
    st.row_frontier.take(row_list::singletons, w.rows);
    for (int32_t row : w.rows) {
      gprs_assert(st.row_frontier.pending[row] & own_list);
      if (w.has_key(row)) { w.drop(row); }
      w.record[row].free = 0;
    }
  }
  // The core of the row changed in this call.
  auto core_changed = [&st, near](int32_t row) {
    return near && (st.row_frontier.pending[row] & singleton_list) != 0;
  };

  st.row_frontier.take(row_list::parallel_rows, w.rows);
  if (!std::is_sorted(w.rows.begin(), w.rows.end())) { std::sort(w.rows.begin(), w.rows.end()); }
  auto give_back = [&](size_t from) {
    for (size_t k = from; k < w.rows.size(); ++k) {
      if (st.row_alive[w.rows[k]]) { st.row_frontier.push(own_list, w.rows[k]); }
    }
    result.stopped = true;
  };

  // Every listed row gets its key and its bucket before the first comparison, so that two
  // listed rows meet in one bucket. A row that comes back unchanged keeps its key at no cost.
  bool out_of_budget = !w.key_rows(st, budget);
  if (sort_share * (w.fresh.size() + w.recent.size()) < (size_t)st.nrows) {
    for (int32_t row : w.fresh) {
      w.insert(row);
    }
  } else if (!w.fresh.empty()) {
    w.sort_all();
  }
  if (out_of_budget) {
    give_back(0);
    return result;
  }

  for (size_t k = 0; k < w.rows.size() && result.outcome == ok; ++k) {
    if (budget.stop()) {
      give_back(k);
      break;
    }
    int32_t row = w.rows[k];
    if (!w.has_key(row)) { continue; }
    gprs_assert(st.row_alive[row]);
    budget.spend(line_visit_work + st.csr.used[row]);
    if (w.above[row] == no_row && w.below[row] == no_row) { continue; }
    // Every change of the core of a row puts it on a list that this call has read, or takes its
    // key at once (below), so a key is current.
    gprs_assert(!core_changed(row) && w.key_is_current(st, row));
    scattered_row_t s = scatter(st, row, near, w.coefficient, w.cols);
    // The listed row is gone, or it has other entries or another core than the scattered one.
    bool done = false;

    // The reduction of a pair with parallel cores. False when the pair is left alone, which
    // counts as not parallel towards the limit.
    auto reduce = [&](const comparison_t& c) {
      const near_row_t& other = c.other;
      if (s.self.singles == 0 && other.singles == 0) {
        bool keep_listed          = first_is_kept(s.self, other, opt.parallel_keep_larger);
        const near_row_t& kept    = keep_listed ? s.self : other;
        const near_row_t& removed = keep_listed ? other : s.self;
        if (!keeps_sides(kept.fixed_terms, tol)) { return false; }
        double drift = 0.;
        if (!c.identical) {
          budget.spend(st.csr.used[other.row]);
          drift = core_drift(st, w.coefficient, other.row, keep_listed, kept.first, removed.first);
          if (!(drift <= opt.parallel_drift_share * tol.feasibility)) {
            result.limited = true;
            return false;
          }
        }
        merge_t step = merge(st, ps, tol, kept, removed, drift);
        if (step == merge_t::infeasible) {
          result.outcome = infeasible;
          return true;
        }
        if (step != merge_t::made) {
          result.limited = result.limited || step == merge_t::left_for_now;
          return false;
        }
        w.drop(removed.row);
        // A column that only the two rows held is a singleton column of the kept row now.
        if (core_changed(kept.row)) { w.drop(kept.row); }
        done = !w.has_key(row);
        return true;
      }
      bool mine_may  = may_subtract(st, s.self, other);
      bool other_may = may_subtract(st, other, s.self);
      if (!mine_may && !other_may) { return false; }
      // Of two equations the one with fewer singleton columns is subtracted, then the one with
      // the smaller index.
      bool mine_is_equation      = mine_may && (!other_may || s.self.singles < other.singles ||
                                           (s.self.singles == other.singles && row < other.row));
      const near_row_t& equation = mine_is_equation ? s.self : other;
      const near_row_t& target   = mine_is_equation ? other : s.self;
      subtraction_t step         = subtract(st, ps, tol, budget, equation, target, w.slots);
      if (step != subtraction_t::made) {
        result.limited = result.limited || step == subtraction_t::left_to_a_changed_row;
        return false;
      }
      w.drop(target.row);
      // The singleton column of the equation has a second entry, and a column that only the
      // two rows held is a singleton column of the equation now.
      if (core_changed(equation.row)) { w.drop(equation.row); }
      done = !w.has_key(row);
      return true;
    };

    // Rows above first: those that this row keeps are merged into it before its sides go to a
    // row below.
    int32_t unlike = 0;
    int32_t other  = w.above[row];
    while (other != no_row && !done &&
           (unlike < opt.parallel_bucket_limit || !st.row_alive[other])) {
      int32_t next = w.above[other];
      if (!st.row_alive[other]) {
        w.drop(other);
        other = next;
        continue;
      }
      gprs_assert(!core_changed(other) && w.key_is_current(st, other));
      budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[other]);
      comparison_t c = compare(st, tol, s, w.coefficient, other, near);
      if (!c.parallel || !reduce(c)) {
        ++unlike;
        other = next;
        continue;
      }
      if (result.outcome != ok) { break; }
      other = next;
      ++result.findings;
      budget.found();
    }
    // The scan ended on the limit with a row of the bucket left: the listed row was not
    // compared with it, and will be only by a presolve that finds fewer rows between the two.
    if (result.outcome == ok && !done && other != no_row) { result.limited = true; }

    // at is the listed row or, below it, the last row found unlike.
    unlike     = 0;
    int32_t at = row;
    while (result.outcome == ok && !done) {
      other = w.below[at];
      if (other == no_row) { break; }
      if (!st.row_alive[other]) {
        w.drop(other);
        continue;
      }
      if (unlike == opt.parallel_bucket_limit) {
        result.limited = true;
        break;
      }
      gprs_assert(!core_changed(other) && w.key_is_current(st, other));
      budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[other]);
      comparison_t c = compare(st, tol, s, w.coefficient, other, near);
      if (!c.parallel || !reduce(c)) {
        ++unlike;
        at = other;
        continue;
      }
      if (result.outcome != ok) { break; }
      ++result.findings;
      budget.found();
    }
    for (int32_t col : w.cols) {
      w.coefficient[col] = 0.;
    }
  }
  return result;
}

}  // namespace gprs
