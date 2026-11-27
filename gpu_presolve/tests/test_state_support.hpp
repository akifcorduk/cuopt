/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Support of the state tests: a generator of small models with the bounds that matter for the
// activities, and a naive reference of the state (shadow_t) that keeps the model as maps and
// recomputes everything from scratch.

#include "harness.hpp"

#include <gprs/state.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace state_test {

using namespace gprs;

struct rng_t {
  uint64_t x;
  explicit rng_t(uint64_t seed) : x(seed * 0x9e3779b97f4a7c15ull + 0x2545f4914f6cdd1dull)
  {
    if (x == 0) { x = 1; }
  }
  uint64_t next()
  {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
  }
  double uniform() { return (next() >> 11) * 0x1p-53; }
  int32_t below(int32_t n) { return next() % (uint64_t)n; }
  bool chance(double p) { return uniform() < p; }
};

// Zero, small integers, values over eight decades and values around the infinite-term rule.
inline double random_bound_value(rng_t& rng)
{
  static const double huge[] = {1e18, 1e19, 5e19, 1e20, 2e20, 1e21, 1e24};
  double pick                = rng.uniform();
  double sign                = rng.chance(0.5) ? -1. : 1.;
  if (pick < 0.15) { return sign * huge[rng.below(7)]; }
  if (pick < 0.25) { return 0.; }
  if (pick < 0.6) { return sign * rng.below(20); }
  return sign * std::pow(10., 6. * rng.uniform() - 2.);
}

// decimal: a multiple of 0.1, so that 10 * a is an integer up to rounding.
inline double random_coefficient(rng_t& rng, bool decimal)
{
  double sign = rng.chance(0.5) ? -1. : 1.;
  if (decimal) { return sign * (1 + rng.below(50)) / 10.; }
  return sign * std::pow(10., 6. * rng.uniform() - 3.);
}

struct model_options_t {
  int32_t nrows     = 0;
  int32_t ncols     = 0;
  double density    = 0.;
  int32_t long_rows = 0;  // leading rows with an entry in every column
  bool decimal      = false;
};

inline model_t random_model(rng_t& rng, const model_options_t& o)
{
  model_t m;
  m.nrows = o.nrows;
  m.ncols = o.ncols;
  m.row_start.push_back(0);
  for (int32_t i = 0; i < o.nrows; ++i) {
    double density = 2. * o.density * rng.uniform();
    if (rng.chance(0.1)) { density = 0.; }
    if (i < o.long_rows) { density = 1.; }
    for (int32_t j = 0; j < o.ncols; ++j) {
      if (rng.uniform() >= density) { continue; }
      m.col_index.push_back(j);
      m.value.push_back(random_coefficient(rng, o.decimal));
    }
    m.row_start.push_back(m.col_index.size());
    double lower = random_bound_value(rng);
    double upper = lower + std::fabs(random_bound_value(rng));
    int32_t kind = rng.below(5);
    if (kind == 0 || kind == 1) { lower = -inf; }
    if (kind == 0 || kind == 2) { upper = inf; }
    if (kind == 3) { upper = lower; }
    m.row_lower.push_back(lower);
    m.row_upper.push_back(upper);
  }
  for (int32_t j = 0; j < o.ncols; ++j) {
    double lower = random_bound_value(rng);
    double upper = lower + std::fabs(random_bound_value(rng));
    double kind  = rng.uniform();
    if (kind < 0.15 || (kind > 0.45 && kind < 0.5)) { lower = -inf; }
    if (kind > 0.85 || (kind > 0.45 && kind < 0.5)) { upper = inf; }
    m.col_lower.push_back(lower);
    m.col_upper.push_back(upper);
    m.cost.push_back(rng.chance(0.3) ? 0. : random_coefficient(rng, o.decimal));
    m.col_type.push_back(rng.chance(0.5));
  }
  m.objective_offset = rng.below(11) - 5;
  return m;
}

inline bool near(double x, double y, double scale)
{
  if (x == y) { return true; }
  return std::fabs(x - y) <= 1e-9 * (scale + std::fabs(x) + std::fabs(y));
}

// The model a state stands for, kept naively: one map per row with its live entries (those of
// fixed columns included until a rebuild), plain doubles for sides and offset. Sides and offset
// follow the state up to rounding (side_scale, offset_scale); everything else is exact.
struct shadow_t {
  int32_t nrows;
  int32_t ncols;
  tolerances_t tol;
  std::vector<std::map<int32_t, double>> row;
  std::vector<double> row_lower, row_upper, side_scale;
  std::vector<uint8_t> row_alive;
  std::vector<uint32_t> version;
  std::vector<double> col_lower, col_upper, cost;
  std::vector<uint8_t> is_integer;
  std::vector<col_status_t> status;
  std::vector<int32_t> implied_lower_row, implied_upper_row;
  std::vector<uint32_t> implied_lower_version, implied_upper_version;
  std::vector<double> implied_lower, implied_upper;
  double offset;
  double offset_scale;

  shadow_t(const model_t& m, const tolerances_t& tolerances)
    : nrows(m.nrows),
      ncols(m.ncols),
      tol(tolerances),
      row(m.nrows),
      row_lower(m.row_lower),
      row_upper(m.row_upper),
      side_scale(m.nrows, 0.),
      row_alive(m.nrows, 1),
      version(m.nrows, 0),
      col_lower(m.col_lower),
      col_upper(m.col_upper),
      cost(m.cost),
      is_integer(m.col_type),
      status(m.ncols, col_status_t::live),
      implied_lower_row(m.ncols, -1),
      implied_upper_row(m.ncols, -1),
      implied_lower_version(m.ncols, 0),
      implied_upper_version(m.ncols, 0),
      implied_lower(m.ncols, -inf),
      implied_upper(m.ncols, inf),
      offset(m.objective_offset),
      offset_scale(std::fabs(m.objective_offset))
  {
    for (int32_t i = 0; i < m.nrows; ++i) {
      for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
        row[i][m.col_index[e]] = m.value[e];
      }
    }
  }

  bool live(int32_t col) const { return status[col] == col_status_t::live; }

  void fix_col(int32_t col, double value)
  {
    col_lower[col] = value;
    col_upper[col] = value;
    status[col]    = col_status_t::fixed;
    offset += cost[col] * value;
    offset_scale += std::fabs(cost[col] * value);
  }

  void delete_row(int32_t i)
  {
    row[i].clear();
    row_alive[i] = 0;
  }

  void shift_row_sides(int32_t i, double a, double b)
  {
    side_scale[i] += std::fabs(a * b);
    if (row_lower[i] > -inf) { row_lower[i] -= a * b; }
    if (row_upper[i] < inf) { row_upper[i] -= a * b; }
    ++version[i];
  }

  void scale_row(int32_t i, double factor, bool integral)
  {
    for (auto& entry : row[i]) {
      entry.second *= factor;
      if (integral && live(entry.first)) { entry.second = std::round(entry.second); }
    }
    row_lower[i] *= factor;
    row_upper[i] *= factor;
    side_scale[i] *= factor;
    ++version[i];
  }

  // Fixed columns leave: their terms go into the sides.
  // The terms of the fixed columns leave the rows for the sides; the columns stay fixed.
  void absorb_fixed_terms()
  {
    for (int32_t i = 0; i < nrows; ++i) {
      for (auto it = row[i].begin(); it != row[i].end();) {
        int32_t col = it->first;
        if (status[col] != col_status_t::fixed) {
          ++it;
          continue;
        }
        double term = it->second * col_lower[col];
        side_scale[i] += std::fabs(term);
        if (row_lower[i] > -inf) { row_lower[i] -= term; }
        if (row_upper[i] < inf) { row_upper[i] -= term; }
        it = row[i].erase(it);
      }
    }
  }

  void rebuild()
  {
    absorb_fixed_terms();
    for (int32_t j = 0; j < ncols; ++j) {
      if (status[j] == col_status_t::fixed) { status[j] = col_status_t::removed; }
    }
  }

  bool implied_lower_valid(int32_t col) const
  {
    int32_t i = implied_lower_row[col];
    return i >= 0 && row_alive[i] && version[i] == implied_lower_version[col];
  }

  bool implied_upper_valid(int32_t col) const
  {
    int32_t i = implied_upper_row[col];
    return i >= 0 && row_alive[i] && version[i] == implied_upper_version[col];
  }
};

// A bound of |hi + lo - exact sum| for an accumulator that took `updates` terms of absolute
// value up to max_term, by dd_add_product, dd_sub_product and dd_add in any order: |hi| stays
// below updates * max_term, so every error that enters a lo is below u * (updates + 1) *
// max_term and |lo| below 2 * updates times that; each of the at most 4 * updates additions into
// a lo rounds by at most u * |lo|. The last term is for products that underflow.
// activity_error(updates, max_term) is smaller than the error can be: it has max_term where the
// rounding of an addition into lo follows the size of the sum.
inline double proven_activity_error(uint32_t updates, double max_term)
{
  double n = updates + 1.;
  return 9. * n * n * n * rounding_unit * rounding_unit * max_term +
         n * std::numeric_limits<double>::denorm_min();
}

struct reference_side_t {
  dd_t sum;
  int32_t inf_count = 0;
  int32_t inf_col   = -1;
  double max_term   = 0.;
  uint32_t terms    = 0;

  void add(double a, double bound, int32_t col, const tolerances_t& tol)
  {
    if (!(std::fabs(a * bound) <= tol.infinite_term)) {
      ++inf_count;
      inf_col = col;
      return;
    }
    dd_add_product(sum, a, bound);
    max_term = std::fmax(max_term, std::fabs(a * bound));
    ++terms;
  }
};

inline std::string compare_side(const state_t& st,
                                const activity_t& act,
                                const reference_side_t& ref,
                                int32_t i,
                                const char* name)
{
  std::ostringstream out;
  out.precision(17);
  if (act.inf_count[i] != ref.inf_count) {
    out << name << " activity of row " << i << ": inf_count " << act.inf_count[i] << ", reference "
        << ref.inf_count;
    return out.str();
  }
  if (ref.inf_count == 1) {
    int64_t slot = act.inf_slot[i];
    if (slot < st.csr.begin(i) || slot >= st.csr.end(i) || st.csr.index[slot] != ref.inf_col) {
      out << name << " activity of row " << i << ": inf_slot " << slot
          << " is not the entry of column " << ref.inf_col;
      return out.str();
    }
  }
  double rounding;
  double difference = two_sum(act.hi[i], -ref.sum.hi, rounding);
  difference += rounding + (act.lo[i] - ref.sum.lo);
  double bound = proven_activity_error(act.updates[i], act.max_term[i]) +
                 proven_activity_error(ref.terms, ref.max_term);
  if (!(std::fabs(difference) <= bound)) {
    out << name << " activity of row " << i << ": (" << act.hi[i] << ", " << act.lo[i]
        << ") against the reference (" << ref.sum.hi << ", " << ref.sum.lo << "), difference "
        << difference << ", error bound " << bound;
    return out.str();
  }
  return {};
}

// Empty when the state stands for the shadow's model, else the first difference.
inline std::string compare(const state_t& st, const shadow_t& sh)
{
  std::ostringstream out;
  out.precision(17);
  int32_t live_rows = 0, live_cols = 0;
  int64_t live_nnz = 0, fixed_nnz = 0;
  std::vector<int32_t> down(sh.ncols, 0), up(sh.ncols, 0);

  for (int32_t i = 0; i < sh.nrows; ++i) {
    if (st.row_alive[i] != sh.row_alive[i]) {
      out << "row " << i << ": alive " << +st.row_alive[i];
      return out.str();
    }
    if (!sh.row_alive[i]) { continue; }
    ++live_rows;
    if (!near(st.row_lower[i], sh.row_lower[i], sh.side_scale[i]) ||
        !near(st.row_upper[i], sh.row_upper[i], sh.side_scale[i])) {
      out << "row " << i << ": sides [" << st.row_lower[i] << ", " << st.row_upper[i]
          << "], reference [" << sh.row_lower[i] << ", " << sh.row_upper[i] << "]";
      return out.str();
    }
    if (st.row_version[i] != sh.version[i]) {
      out << "row " << i << ": version " << st.row_version[i] << ", reference " << sh.version[i];
      return out.str();
    }
    std::map<int32_t, double> entries;
    int32_t free_size = 0;
    for (int64_t e = st.csr.begin(i); e < st.csr.end(i); ++e) {
      if (!st.csr.live(e)) { continue; }
      if (!entries.emplace(st.csr.index[e], st.csr.value[e]).second) {
        out << "row " << i << ": column " << st.csr.index[e] << " twice";
        return out.str();
      }
    }
    if (entries != sh.row[i]) {
      out << "row " << i << ": entries differ from the reference (" << entries.size() << " against "
          << sh.row[i].size() << ")";
      for (const auto& entry : sh.row[i]) {
        auto it = entries.find(entry.first);
        if (it == entries.end()) {
          out << "; column " << entry.first << " missing";
        } else if (it->second != entry.second) {
          out << "; column " << entry.first << " " << it->second << " against " << entry.second;
        }
      }
      return out.str();
    }
    reference_side_t ref_min, ref_max;
    int32_t lower_finite = sh.row_lower[i] > -inf;
    int32_t upper_finite = sh.row_upper[i] < inf;
    for (const auto& entry : sh.row[i]) {
      int32_t j = entry.first;
      double a  = entry.second;
      ++live_nnz;
      free_size += sh.live(j);
      fixed_nnz += sh.status[j] == col_status_t::fixed;
      ref_min.add(a, a > 0 ? sh.col_lower[j] : sh.col_upper[j], j, sh.tol);
      ref_max.add(a, a > 0 ? sh.col_upper[j] : sh.col_lower[j], j, sh.tol);
      if (a > 0) {
        up[j] += upper_finite;
        down[j] += lower_finite;
      } else {
        down[j] += upper_finite;
        up[j] += lower_finite;
      }
    }
    if (st.row_free_size[i] != free_size) {
      out << "row " << i << ": row_free_size " << st.row_free_size[i] << ", reference "
          << free_size;
      return out.str();
    }
    std::string error = compare_side(st, st.min_act, ref_min, i, "minimum");
    if (error.empty()) { error = compare_side(st, st.max_act, ref_max, i, "maximum"); }
    if (!error.empty()) { return error; }
  }

  for (int32_t j = 0; j < sh.ncols; ++j) {
    live_cols += sh.live(j);
    if (st.col_status[j] != sh.status[j] || st.col_lower[j] != sh.col_lower[j] ||
        st.col_upper[j] != sh.col_upper[j] || st.cost[j] != sh.cost[j] ||
        st.is_integer(j) != (sh.is_integer[j] != 0)) {
      out << "column " << j << ": status " << +(uint8_t)st.col_status[j] << " bounds ["
          << st.col_lower[j] << ", " << st.col_upper[j] << "] cost " << st.cost[j]
          << ", reference status " << +(uint8_t)sh.status[j] << " bounds [" << sh.col_lower[j]
          << ", " << sh.col_upper[j] << "] cost " << sh.cost[j];
      return out.str();
    }
    if (st.down_locks[j] != down[j] || st.up_locks[j] != up[j]) {
      out << "column " << j << ": locks down " << st.down_locks[j] << " up " << st.up_locks[j]
          << ", reference down " << down[j] << " up " << up[j];
      return out.str();
    }
    if (st.implied_lower_valid(j) != sh.implied_lower_valid(j) ||
        st.implied_upper_valid(j) != sh.implied_upper_valid(j)) {
      out << "column " << j << ": validity of an implied bound differs from the reference";
      return out.str();
    }
    if (st.implied_lower[j] != sh.implied_lower[j] || st.implied_upper[j] != sh.implied_upper[j] ||
        st.implied_lower_row[j] != sh.implied_lower_row[j] ||
        st.implied_upper_row[j] != sh.implied_upper_row[j]) {
      out << "column " << j << ": implied bound record differs from the reference";
      return out.str();
    }
  }
  if (st.live_rows != live_rows || st.live_cols != live_cols || st.live_nnz != live_nnz ||
      st.fixed_nnz != fixed_nnz) {
    out << "counters: rows " << st.live_rows << " columns " << st.live_cols << " nnz "
        << st.live_nnz << " fixed nnz " << st.fixed_nnz << ", reference " << live_rows << " "
        << live_cols << " " << live_nnz << " " << fixed_nnz;
    return out.str();
  }
  if (!near(dd_value(st.objective_offset), sh.offset, sh.offset_scale)) {
    out << "objective offset " << dd_value(st.objective_offset) << ", reference " << sh.offset;
    return out.str();
  }
  return {};
}

// The model of a state that may hold tombstones, deleted rows and fixed columns: what
// extract_model returns after a rebuild, with the fixed terms of each row taken from its sides
// as one compensated sum in storage order.
inline model_t reference_model(const state_t& st,
                               std::vector<int32_t>& row_origin,
                               std::vector<int32_t>& col_origin)
{
  model_t m;
  row_origin.clear();
  col_origin.clear();
  std::vector<int32_t> col_index(st.ncols, -1);
  for (int32_t j = 0; j < st.ncols; ++j) {
    if (!st.is_live(j)) { continue; }
    col_index[j] = col_origin.size();
    col_origin.push_back(j);
    m.col_lower.push_back(st.col_lower[j]);
    m.col_upper.push_back(st.col_upper[j]);
    m.cost.push_back(st.cost[j]);
    m.col_type.push_back(st.is_integer(j));
  }
  m.row_start.push_back(0);
  for (int32_t i = 0; i < st.nrows; ++i) {
    if (!st.row_alive[i]) { continue; }
    row_origin.push_back(i);
    std::map<int32_t, double> entries;
    dd_t fixed_terms;
    bool has_fixed = false;
    for (int64_t e = st.csr.begin(i); e < st.csr.end(i); ++e) {
      if (!st.csr.live(e)) { continue; }
      int32_t j = st.csr.index[e];
      if (st.col_status[j] == col_status_t::fixed) {
        dd_add_product(fixed_terms, st.csr.value[e], st.col_lower[j]);
        has_fixed = true;
      } else {
        entries[col_index[j]] = st.csr.value[e];
      }
    }
    for (const auto& entry : entries) {
      m.col_index.push_back(entry.first);
      m.value.push_back(entry.second);
    }
    m.row_start.push_back(m.col_index.size());
    double lower = st.row_lower[i], upper = st.row_upper[i];
    if (has_fixed && lower > -inf) { lower = dd_residual(lower, fixed_terms); }
    if (has_fixed && upper < inf) { upper = dd_residual(upper, fixed_terms); }
    m.row_lower.push_back(lower);
    m.row_upper.push_back(upper);
  }
  m.nrows            = row_origin.size();
  m.ncols            = col_origin.size();
  m.objective_offset = dd_value(st.objective_offset);
  return m;
}

inline bool same_doubles(const std::vector<double>& a, const std::vector<double>& b)
{
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// Bit for bit.
inline bool same_model(const model_t& a, const model_t& b)
{
  return a.nrows == b.nrows && a.ncols == b.ncols && a.row_start == b.row_start &&
         a.col_index == b.col_index && same_doubles(a.value, b.value) &&
         same_doubles(a.row_lower, b.row_lower) && same_doubles(a.row_upper, b.row_upper) &&
         same_doubles(a.col_lower, b.col_lower) && same_doubles(a.col_upper, b.col_upper) &&
         same_doubles(a.cost, b.cost) && a.col_type == b.col_type &&
         same_doubles({a.objective_offset}, {b.objective_offset});
}

// Empties the rule lists, so that the events of the next operation can be read from pending.
inline void clear_rule_lists(state_t& st)
{
  std::vector<int32_t> taken;
  for (int32_t l = 0; l < row_list::compaction; ++l) {
    st.row_frontier.take(l, taken);
  }
  st.col_frontier.take(col_list::dual_fixing, taken);
}

}  // namespace state_test
