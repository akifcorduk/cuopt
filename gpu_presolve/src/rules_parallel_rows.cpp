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
#include <cmath>
#include <iterator>
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
  int exponent = 0;
  int64_t grid = 0;
  if (std::isfinite(normalised) && normalised != 0.) {
    grid = std::llround(std::ldexp(std::frexp(normalised, &exponent), hash_mantissa_bits));
    // A mantissa that rounds up to 1 is the mantissa 1/2 of the next exponent.
    if (grid == 1 << hash_mantissa_bits || grid == -(1 << hash_mantissa_bits)) {
      grid /= 2;
      ++exponent;
    }
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

using buckets_t = std::set<bucket_key_t>;

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
  // Key of each row in buckets; length 0 without one. version is row_version at hashing: with
  // the free length it tells whether the free entries are those that were hashed.
  std::vector<uint64_t> hash;
  std::vector<int32_t> length;
  std::vector<uint32_t> version;
  // Ordered by hash, free length, row: a bucket is a range, in index order.
  buckets_t buckets;
  // The free entries of the listed row under comparison, by column and 0 elsewhere, and its
  // free columns.
  std::vector<double> coefficient;
  std::vector<int32_t> cols;
  std::vector<int32_t> rows;

  bucket_key_t key(int32_t row) const { return bucket_key_t{hash[row], length[row], row}; }

  void drop(int32_t row)
  {
    gprs_assert(length[row] != 0);
    buckets.erase(key(row));
    length[row] = 0;
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
    w.buckets.clear();
  }
  if (w.coefficient.size() != (size_t)st.ncols) { w.coefficient.assign(st.ncols, 0.); }
  const tolerances_t& tol = opt.tol;
  const uint8_t own_list  = list_bit(row_list::parallel_rows);
  rule_result_t result;

  st.row_frontier.take(row_list::parallel_rows, w.rows);
  std::sort(w.rows.begin(), w.rows.end());
  auto give_back = [&](size_t from) {
    for (size_t k = from; k < w.rows.size(); ++k) {
      if (st.row_alive[w.rows[k]]) { st.row_frontier.push(own_list, w.rows[k]); }
    }
    result.stopped = true;
  };

  // Every listed row gets its key before the first comparison, so that two listed rows meet in
  // one bucket. A row that comes back unchanged keeps its key at no cost.
  for (int32_t row : w.rows) {
    if (budget.stop()) {
      give_back(0);
      return result;
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
    w.buckets.insert(w.key(row));
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
    scattered_row_t s = scatter(st, row, w.coefficient, w.cols);
    bucket_key_t key  = w.key(row);
    auto self         = w.buckets.find(key);
    gprs_assert(self != w.buckets.end());
    auto same_bucket = [&](const bucket_key_t& other) {
      return other.hash == key.hash && other.length == key.length;
    };

    // Rows above first: they are merged into this row, whose sides then go to a row below.
    int32_t unlike = 0;
    for (auto it = std::next(self);
         it != w.buckets.end() && same_bucket(*it) && unlike < opt.parallel_bucket_limit;) {
      int32_t other = it->row;
      if (!st.row_alive[other]) {
        w.length[other] = 0;
        it              = w.buckets.erase(it);
        continue;
      }
      // Every change of the free entries of a row puts it on the list, so a key is current.
      gprs_assert(st.row_free_size[other] == key.length);
      budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[other]);
      comparison_t c = compare(st, tol, s, w.coefficient, other);
      if (!c.parallel || !keeps_sides(s.fixed_terms, tol)) {
        ++unlike;
        ++it;
        continue;
      }
      result.outcome = merge(st, ps, tol, row, s.fixed_terms, other, c.fixed_terms, c.ratio);
      if (result.outcome != ok) { break; }
      w.length[other] = 0;
      it              = w.buckets.erase(it);
      ++result.findings;
      budget.found();
    }

    unlike = 0;
    while (result.outcome == ok && self != w.buckets.begin() &&
           unlike < opt.parallel_bucket_limit) {
      auto below = std::prev(self);
      if (!same_bucket(*below)) { break; }
      int32_t other = below->row;
      if (!st.row_alive[other]) {
        w.length[other] = 0;
        w.buckets.erase(below);
        continue;
      }
      gprs_assert(st.row_free_size[other] == key.length);
      budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[other]);
      comparison_t c = compare(st, tol, s, w.coefficient, other);
      if (!c.parallel || !keeps_sides(c.fixed_terms, tol)) {
        ++unlike;
        self = below;
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
