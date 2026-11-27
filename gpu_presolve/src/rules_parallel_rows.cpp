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

// Sum over the free entries, so the storage order does not enter; a multiple of the row has the
// same normalised coefficients.
uint64_t row_hash(const state_t& st, int32_t row)
{
  const line_store_t& csr = st.csr;
  int32_t first           = st.ncols;
  double first_value      = 0.;
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (csr.live(e) && st.is_live(csr.index[e]) && csr.index[e] < first) {
      first       = csr.index[e];
      first_value = csr.value[e];
    }
  }
  gprs_assert(first < st.ncols);
  uint64_t hash = 0;
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (csr.live(e) && st.is_live(csr.index[e])) {
      hash += entry_hash(csr.index[e], csr.value[e] / first_value);
    }
  }
  return hash;
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

// The free entries of one row scattered over the columns.
struct scattered_row_t {
  int32_t row;
  int32_t length;
  int32_t first;  // smallest free column
  dd_t fixed_terms;
};

// cols gets the free columns: the row can be deleted before its scatter is cleared.
scattered_row_t scatter(const state_t& st,
                        int32_t row,
                        std::vector<double>& coefficient,
                        std::vector<int32_t>& cols)
{
  const line_store_t& csr = st.csr;
  scattered_row_t s{row, 0, st.ncols, dd_t()};
  cols.clear();
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if (!st.is_live(col)) {
      dd_add_product(s.fixed_terms, csr.value[e], st.col_lower[col]);
      continue;
    }
    coefficient[col] = csr.value[e];
    cols.push_back(col);
    if (col < s.first) { s.first = col; }
  }
  s.length = cols.size();
  return s;
}

struct comparison_t {
  bool parallel = false;
  // larger row = ratio * smaller row.
  double ratio = 0.;
  dd_t fixed_terms;  // of the row that was not scattered
};

// Exact test of `other` against the scattered row, on the entries as they are stored now.
comparison_t compare(const state_t& st,
                     const tolerances_t& tol,
                     const scattered_row_t& s,
                     const std::vector<double>& coefficient,
                     int32_t other)
{
  const line_store_t& csr = st.csr;
  comparison_t result;
  int32_t length     = 0;
  double other_first = 0.;
  for (int64_t e = csr.begin(other); e < csr.end(other); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if (!st.is_live(col)) {
      dd_add_product(result.fixed_terms, csr.value[e], st.col_lower[col]);
      continue;
    }
    if (coefficient[col] == 0.) { return result; }
    ++length;
    if (col == s.first) { other_first = csr.value[e]; }
  }
  // Every free column of `other` is one of the scattered row, so equal lengths are equal sets.
  if (length != s.length) { return result; }
  bool scattered_is_smaller = s.row < other;
  result.ratio =
    scattered_is_smaller ? other_first / coefficient[s.first] : coefficient[s.first] / other_first;
  if (!std::isfinite(result.ratio) || result.ratio == 0.) { return result; }
  for (int64_t e = csr.begin(other); e < csr.end(other); ++e) {
    if (!csr.live(e) || !st.is_live(csr.index[e])) { continue; }
    double a = coefficient[csr.index[e]], b = csr.value[e];
    if (!scattered_is_smaller) { std::swap(a, b); }
    if (!(std::fabs(b - result.ratio * a) <= tol.zero * std::fabs(b))) { return result; }
  }
  result.parallel = true;
  return result;
}

// A side of the kept row is stored as one double with the fixed terms of that row in it, and
// the sides of the two rows are compared in that form. Fixed terms above this size take more
// than the tolerance from a side next to them (as holds_tolerance in the row tightening).
inline bool keeps_sides(dd_t fixed_terms, const tolerances_t& tol)
{
  return std::fabs(dd_value(fixed_terms)) * tol.side_error <= tol.feasibility;
}

// removed = ratio * kept on the free entries: the sides of `removed` become sides of `kept`.
// The caller has tested keeps_sides(kept_fixed).
rule_outcome_t merge(state_t& st,
                     postsolve_t& ps,
                     const tolerances_t& tol,
                     int32_t kept,
                     dd_t kept_fixed,
                     int32_t removed,
                     dd_t removed_fixed,
                     double ratio)
{
  gprs_assert(keeps_sides(kept_fixed, tol));
  auto image = [&](double side) {
    double free_side = std::isfinite(side) ? dd_residual(side, removed_fixed) : side;
    return free_side / ratio + dd_value(kept_fixed);
  };
  double lower = image(st.row_lower[removed]), upper = image(st.row_upper[removed]);
  if (ratio < 0) { std::swap(lower, upper); }
  double kept_lower = st.row_lower[kept], kept_upper = st.row_upper[kept];
  if (lower > kept_lower) {
    if (lower - kept_upper > tol.feasibility) { return infeasible; }
    if (lower > kept_upper) { lower = kept_upper; }
    if (lower != kept_lower) {
      st.set_row_lower(kept, lower);
      kept_lower = lower;
    }
  }
  if (upper < kept_upper) {
    if (kept_lower - upper > tol.feasibility) { return infeasible; }
    if (upper < kept_lower) { upper = kept_lower; }
    if (upper != kept_upper) { st.set_row_upper(kept, upper); }
  }
  st.delete_row(removed);
  ps.push_parallel_row(removed, kept, ratio);
  return ok;
}

}  // namespace

struct parallel_rows_t::work_t {
  // Key of each row; length 0 without one. version is row_version at hashing: with the free
  // length it tells whether the free entries are those that were hashed.
  std::vector<uint64_t> hash;
  std::vector<int32_t> length;
  std::vector<uint32_t> version;
  // A bucket is a list of the rows with one hash and one free length, in index order: the next
  // row of the bucket above and below each row, no_row at the ends.
  std::vector<int32_t> above;
  std::vector<int32_t> below;
  // Where a new key finds its bucket. base holds the keys of the last sort, ordered by hash,
  // free length, row; in_base marks the rows whose key is still the one in base. recent holds
  // the keys set since.
  std::vector<bucket_key_t> base;
  std::vector<uint8_t> in_base;
  std::set<bucket_key_t> recent;
  // Over the entries of base that a search found stale: skip_up[i] leads towards the first
  // entry at or above i that can be current, skip_down[i + 1] towards the last one at or below
  // i (union-find with path compression; base.size() and -1 without one). Empty until the
  // first search after a sort.
  std::vector<int32_t> skip_up;
  std::vector<int32_t> skip_down;
  // The rows that got a new key in this call.
  std::vector<int32_t> fresh;
  // The free entries of the listed row under comparison, by column and 0 elsewhere, and its
  // free columns.
  std::vector<double> coefficient;
  std::vector<int32_t> cols;
  std::vector<int32_t> rows;

  bucket_key_t key(int32_t row) const { return bucket_key_t{hash[row], length[row], row}; }

  void drop(int32_t row)
  {
    gprs_assert(length[row] != 0);
    int32_t lower = below[row], upper = above[row];
    if (lower != no_row) { above[lower] = upper; }
    if (upper != no_row) { below[upper] = lower; }
    below[row] = above[row] = no_row;
    if (in_base[row]) {
      in_base[row] = 0;
    } else {
      size_t erased = recent.erase(key(row));
      gprs_assert(erased == 1);
    }
    length[row] = 0;
  }

  // Sorts the keys of all rows and links the buckets.
  void sort_all()
  {
    int32_t nrows = length.size();
    size_t keys   = 0;
    for (int32_t row = 0; row < nrows; ++row) {
      in_base[row] = length[row] != 0;
      keys += in_base[row];
    }
    base.clear();
    base.reserve(keys);
    for (int32_t row = 0; row < nrows; ++row) {
      if (in_base[row]) { base.push_back(key(row)); }
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
      if (found == n || in_base[base[found].row]) { break; }
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
      if (found == 0 || in_base[base[found - 1].row]) { break; }
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
    gprs_assert(!in_base[row] && recent.count(k) == 0);
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
  if (w.length.size() != (size_t)st.nrows) {
    w.hash.assign(st.nrows, 0);
    w.length.assign(st.nrows, 0);
    w.version.assign(st.nrows, 0);
    w.above.assign(st.nrows, no_row);
    w.below.assign(st.nrows, no_row);
    w.base.clear();
    w.in_base.assign(st.nrows, 0);
    w.recent.clear();
    w.skip_up.clear();
    w.skip_down.clear();
  }
  if (w.coefficient.size() != (size_t)st.ncols) { w.coefficient.assign(st.ncols, 0.); }
  const tolerances_t& tol = opt.tol;
  const uint8_t own_list  = list_bit(row_list::parallel_rows);
  rule_result_t result;

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
  w.fresh.clear();
  bool out_of_budget = false;
  for (int32_t row : w.rows) {
    if (budget.stop()) {
      out_of_budget = true;
      break;
    }
    int32_t length = st.row_alive[row] && st.row_free_size[row] >= 2 ? st.row_free_size[row] : 0;
    if (w.length[row] != 0) {
      if (w.length[row] == length && w.version[row] == st.row_version[row]) { continue; }
      w.drop(row);
    }
    if (length == 0) { continue; }
    budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[row]);
    w.hash[row]    = row_hash(st, row);
    w.length[row]  = length;
    w.version[row] = st.row_version[row];
    w.fresh.push_back(row);
  }
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
    if (w.length[row] == 0) { continue; }
    gprs_assert(st.row_alive[row]);
    budget.spend(line_visit_work + st.csr.used[row]);
    if (w.above[row] == no_row && w.below[row] == no_row) { continue; }
    scattered_row_t s = scatter(st, row, w.coefficient, w.cols);

    // Rows above first: they are merged into this row, whose sides then go to a row below.
    int32_t unlike = 0;
    for (int32_t other = w.above[row]; other != no_row && unlike < opt.parallel_bucket_limit;) {
      int32_t next = w.above[other];
      if (!st.row_alive[other]) {
        w.drop(other);
        other = next;
        continue;
      }
      // Every change of the free entries of a row puts it on the list, so a key is current.
      gprs_assert(st.row_free_size[other] == w.length[row]);
      budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[other]);
      comparison_t c = compare(st, tol, s, w.coefficient, other);
      if (!c.parallel || !keeps_sides(s.fixed_terms, tol)) {
        ++unlike;
        other = next;
        continue;
      }
      result.outcome = merge(st, ps, tol, row, s.fixed_terms, other, c.fixed_terms, c.ratio);
      if (result.outcome != ok) { break; }
      w.drop(other);
      other = next;
      ++result.findings;
      budget.found();
    }

    // at is the listed row or, below it, the last row found unlike.
    unlike = 0;
    for (int32_t at = row; result.outcome == ok && unlike < opt.parallel_bucket_limit;) {
      int32_t other = w.below[at];
      if (other == no_row) { break; }
      if (!st.row_alive[other]) {
        w.drop(other);
        continue;
      }
      gprs_assert(st.row_free_size[other] == w.length[row]);
      budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[other]);
      comparison_t c = compare(st, tol, s, w.coefficient, other);
      if (!c.parallel || !keeps_sides(c.fixed_terms, tol)) {
        ++unlike;
        at = other;
        continue;
      }
      result.outcome = merge(st, ps, tol, other, c.fixed_terms, row, s.fixed_terms, c.ratio);
      if (result.outcome != ok) { break; }
      w.drop(row);
      ++result.findings;
      budget.found();
      break;
    }
    for (int32_t col : w.cols) {
      w.coefficient[col] = 0.;
    }
  }
  return result;
}

}  // namespace gprs
