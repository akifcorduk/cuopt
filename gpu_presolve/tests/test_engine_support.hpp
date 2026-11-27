/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Support of the dual-fixing, engine and hash tests: models from dense rows, a random generator
// of tiny integer models, and their optimum by enumeration.

#include <gprs/state.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace gprs {
namespace engine_test {

using dense_t = std::vector<std::vector<double>>;
using reals_t = std::vector<double>;

// Zeros of the dense rows are no entries. The number of columns is that of the bounds.
inline model_t make_model(const dense_t& rows,
                          const reals_t& row_lower,
                          const reals_t& row_upper,
                          const reals_t& col_lower,
                          const reals_t& col_upper,
                          const reals_t& cost,
                          const std::vector<uint8_t>& col_type)
{
  model_t m;
  m.nrows = rows.size();
  m.ncols = col_lower.size();
  m.row_start.push_back(0);
  for (const reals_t& row : rows) {
    for (size_t j = 0; j < row.size(); ++j) {
      if (row[j] == 0.) { continue; }
      m.col_index.push_back(j);
      m.value.push_back(row[j]);
    }
    m.row_start.push_back(m.value.size());
  }
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.col_lower = col_lower;
  m.col_upper = col_upper;
  m.cost      = cost;
  m.col_type  = col_type;
  return m;
}

// splitmix64.
struct rng_t {
  uint64_t state;

  explicit rng_t(uint64_t seed) : state(seed) {}

  uint64_t next()
  {
    uint64_t z = (state += 0x9e3779b97f4a7c15);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

  // Uniform in [lo, hi].
  int32_t uniform(int32_t lo, int32_t hi) { return lo + (int32_t)(next() % (hi - lo + 1)); }

  bool chance(double probability) { return (next() >> 11) * 0x1p-53 < probability; }
};

struct random_options_t {
  int32_t max_rows = 3;
  int32_t max_cols = 4;
  // Columns that may get an infinite bound; each bound of such a column is infinite with
  // probability one half.
  int32_t max_infinite_cols = 2;
  // What the start step of presolve() deals with: columns with lower == upper and integer
  // columns with a bound moved outwards by 1/2.
  bool start_step_cases = false;
};

// Integer data: coefficients and cost in [-2, 2], finite bounds in [-3, 5] with lower < upper,
// sides in [-6, 6]; rows are <=, >=, equations, ranges or free. Every column is integer.
inline model_t random_model(uint64_t seed, const random_options_t& options)
{
  rng_t rng(seed);
  int32_t nrows = rng.uniform(0, options.max_rows);
  int32_t ncols = rng.uniform(1, options.max_cols);
  dense_t rows(nrows, reals_t(ncols, 0.));
  reals_t row_lower(nrows), row_upper(nrows), col_lower(ncols), col_upper(ncols), cost(ncols);
  for (int32_t i = 0; i < nrows; ++i) {
    for (int32_t j = 0; j < ncols; ++j) {
      if (!rng.chance(0.6)) { continue; }
      rows[i][j] = rng.chance(0.5) ? rng.uniform(1, 2) : -rng.uniform(1, 2);
    }
    double a     = rng.uniform(-6, 6);
    double b     = rng.uniform(-6, 6);
    int32_t kind = rng.uniform(0, 9);
    if (kind < 4) {
      row_lower[i] = -inf;
      row_upper[i] = a;
    } else if (kind < 7) {
      row_lower[i] = a;
      row_upper[i] = inf;
    } else if (kind == 7) {
      row_lower[i] = a;
      row_upper[i] = a;
    } else if (kind == 8) {
      row_lower[i] = a < b ? a : b;
      row_upper[i] = a < b ? b : a;
    } else {
      row_lower[i] = -inf;
      row_upper[i] = inf;
    }
  }
  int32_t infinite_cols = 0;
  for (int32_t j = 0; j < ncols; ++j) {
    col_lower[j] = rng.uniform(-3, 2);
    col_upper[j] = col_lower[j] + rng.uniform(1, 3);
    if (infinite_cols < options.max_infinite_cols && rng.chance(0.3)) {
      ++infinite_cols;
      if (rng.chance(0.5)) { col_lower[j] = -inf; }
      if (rng.chance(0.5)) { col_upper[j] = inf; }
    } else if (options.start_step_cases) {
      int32_t kind = rng.uniform(0, 9);
      if (kind == 0) { col_upper[j] = col_lower[j]; }
      if (kind == 1) { col_lower[j] -= 0.5; }
      if (kind == 2) { col_upper[j] += 0.5; }
    }
    cost[j] = rng.chance(0.35) ? 0. : rng.chance(0.5) ? rng.uniform(1, 2) : -rng.uniform(1, 2);
  }
  model_t m = make_model(
    rows, row_lower, row_upper, col_lower, col_upper, cost, std::vector<uint8_t>(ncols, 1));
  m.objective_offset = rng.uniform(-3, 3);
  return m;
}

struct optimum_t {
  bool feasible    = false;
  double objective = inf;
  reals_t x;
};

// Minimum over the integer points of the box of the model, an infinite bound replaced by
// -window or window. Exact on integer data. Work: points of the box times the nonzeros.
inline optimum_t enumerate(const model_t& m, double window)
{
  reals_t lower(m.ncols), upper(m.ncols);
  for (int32_t j = 0; j < m.ncols; ++j) {
    lower[j] = std::ceil(m.col_lower[j] > -inf ? m.col_lower[j] : -window);
    upper[j] = std::floor(m.col_upper[j] < inf ? m.col_upper[j] : window);
    if (lower[j] > upper[j]) { return {}; }
  }
  optimum_t best;
  reals_t x = lower;
  while (true) {
    bool feasible = true;
    for (int32_t i = 0; i < m.nrows && feasible; ++i) {
      double activity = 0.;
      for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
        activity += m.value[e] * x[m.col_index[e]];
      }
      feasible = m.row_lower[i] <= activity && activity <= m.row_upper[i];
    }
    if (feasible) {
      double objective = m.objective_offset;
      for (int32_t j = 0; j < m.ncols; ++j) {
        objective += m.cost[j] * x[j];
      }
      if (!best.feasible || objective < best.objective) {
        best.feasible  = true;
        best.objective = objective;
        best.x         = x;
      }
    }
    int32_t j = m.ncols - 1;
    while (j >= 0 && x[j] == upper[j]) {
      x[j] = lower[j];
      --j;
    }
    if (j < 0) { break; }
    x[j] += 1.;
  }
  return best;
}

}  // namespace engine_test
}  // namespace gprs
