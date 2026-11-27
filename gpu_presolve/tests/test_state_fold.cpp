/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "test_state_support.hpp"

#include <gprs/state.hpp>

#include <cmath>
#include <cstring>
#include <vector>

using namespace gprs;
using namespace state_test;

namespace {

uint64_t bits(double value)
{
  uint64_t pattern;
  std::memcpy(&pattern, &value, sizeof(pattern));
  return pattern;
}

// The summation order of numerics.hpp as it is defined: a branch on every term, every lane
// started at zero and filled with dd_add_product, the full tree.
void plain_fold(const state_t& st, int32_t row, side_sum_t& min_side, side_sum_t& max_side)
{
  std::vector<int64_t> live;
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (st.csr.live(e)) { live.push_back(e); }
  }
  int32_t n     = live.size();
  int32_t lanes = activity_lanes(n);
  std::vector<dd_t> lane_min(lanes), lane_max(lanes);
  min_side = side_sum_t{};
  max_side = side_sum_t{};
  for (int32_t k = 0; k < n; ++k) {
    int64_t e    = live[k];
    int32_t j    = st.csr.index[e];
    double a     = st.csr.value[e];
    double b_min = a > 0 ? st.col_lower[j] : st.col_upper[j];
    double b_max = a > 0 ? st.col_upper[j] : st.col_lower[j];
    if (is_infinite_term(a, b_min, st.tol)) {
      ++min_side.inf_count;
      min_side.inf_slot += e;
    } else {
      dd_add_product(lane_min[k % lanes], a, b_min);
      if (std::fabs(a * b_min) > min_side.max_term) { min_side.max_term = std::fabs(a * b_min); }
      ++min_side.terms;
    }
    if (is_infinite_term(a, b_max, st.tol)) {
      ++max_side.inf_count;
      max_side.inf_slot += e;
    } else {
      dd_add_product(lane_max[k % lanes], a, b_max);
      if (std::fabs(a * b_max) > max_side.max_term) { max_side.max_term = std::fabs(a * b_max); }
      ++max_side.terms;
    }
  }
  for (int32_t off = lanes / 2; off > 0; off /= 2) {
    for (int32_t l = 0; l < off; ++l) {
      lane_min[l] = dd_add(lane_min[l], lane_min[l + off]);
      lane_max[l] = dd_add(lane_max[l], lane_max[l + off]);
    }
  }
  min_side.sum = lane_min[0];
  max_side.sum = lane_max[0];
}

// Bit for bit, the sign of a zero included.
bool same_side(const side_sum_t& a, const side_sum_t& b)
{
  return bits(a.sum.hi) == bits(b.sum.hi) && bits(a.sum.lo) == bits(b.sum.lo) &&
         a.inf_count == b.inf_count && a.inf_slot == b.inf_slot &&
         bits(a.max_term) == bits(b.max_term) && a.terms == b.terms;
}

side_sum_t stored_side(const activity_t& act, int32_t row)
{
  side_sum_t side;
  side.sum.hi    = act.hi[row];
  side.sum.lo    = act.lo[row];
  side.inf_count = act.inf_count[row];
  side.inf_slot  = act.inf_slot[row];
  side.max_term  = act.max_term[row];
  side.terms     = act.updates[row];
  return side;
}

// Values at which a shortcut of the sums would show: zeros of both signs, products that
// underflow to a zero of either sign with an error of either sign, terms at and beyond the
// infinite-term rule, and ordinary values whose sum needs the compensation.
double special_value(rng_t& rng, bool coefficient)
{
  static const double values[] = {
    1.,    -1.,  0.1,   -0.3, 2.5,   -0.75, 1e-200, -1e-200, 1e-170, -1e-170, 5e-324, -5e-324, 1e19,
    -1e19, 1e20, -1e20, 3e20, -3e20, 1e8,   -1e8,   1e-9,    -1e-9,  7.,      -13.,   0.,      -0.};
  int32_t count = coefficient ? 24 : 26;
  double v      = values[rng.below(count)];
  if (rng.chance(0.3)) { v *= 1. + rng.uniform(); }
  return v;
}

// Rows of every length up to 80 and a few long ones; row i holds the columns 0 .. length - 1.
model_t special_model(rng_t& rng)
{
  std::vector<int32_t> lengths;
  for (int32_t n = 0; n <= 80; ++n) {
    lengths.push_back(n);
  }
  for (int32_t n : {100, 128, 255, 256, 257, 300, 513}) {
    lengths.push_back(n);
  }
  model_t m;
  m.nrows = lengths.size();
  m.ncols = 513;
  m.row_start.push_back(0);
  for (int32_t n : lengths) {
    for (int32_t j = 0; j < n; ++j) {
      m.col_index.push_back(j);
      m.value.push_back(special_value(rng, true));
    }
    m.row_start.push_back(m.col_index.size());
    m.row_lower.push_back(-1.);
    m.row_upper.push_back(1.);
  }
  for (int32_t j = 0; j < m.ncols; ++j) {
    double lower = rng.chance(0.15) ? -inf : special_value(rng, false);
    double upper = rng.chance(0.15) ? inf : special_value(rng, false);
    m.col_lower.push_back(lower);
    m.col_upper.push_back(upper);
    m.cost.push_back(0.);
    m.col_type.push_back(0);
  }
  return m;
}

}  // namespace

// The sums of the state against the definition of numerics.hpp, bit for bit: the sums of the
// constructor, fold_row_activity, recompute_row_activity, and the same with tombstones in the
// rows, which a short row folds like a long one.
GPRS_TEST(state_fold_special_values)
{
  rng_t rng(71);
  int64_t negative_zero_products = 0, infinite_terms = 0, finite_terms = 0;
  for (int32_t k = 0; k < 20; ++k) {
    model_t m = special_model(rng);
    GPRS_REQUIRE(well_formed(m));
    state_t st(m, tolerances_t());
    for (int32_t i = 0; i < st.nrows; ++i) {
      side_sum_t min_plain, max_plain, min_side, max_side;
      plain_fold(st, i, min_plain, max_plain);
      GPRS_CHECK(same_side(stored_side(st.min_act, i), min_plain));
      GPRS_CHECK(same_side(stored_side(st.max_act, i), max_plain));
      st.fold_row_activity(i, min_side, max_side);
      GPRS_CHECK(same_side(min_side, min_plain));
      GPRS_CHECK(same_side(max_side, max_plain));
      infinite_terms += min_plain.inf_count + max_plain.inf_count;
      finite_terms += min_plain.terms + max_plain.terms;
      for (int64_t e = st.csr.begin(i); e < st.csr.end(i); ++e) {
        double a      = st.csr.value[e];
        double lower  = st.col_lower[st.csr.index[e]];
        double upper  = st.col_upper[st.csr.index[e]];
        bool negative = (a * lower == 0. && std::signbit(a * lower)) ||
                        (a * upper == 0. && std::signbit(a * upper));
        negative_zero_products += negative;
      }
    }

    // Tombstones: every third slot of every row, then every row summed again.
    for (int32_t i = 0; i < st.nrows; ++i) {
      int32_t dead = 0;
      for (int64_t e = st.csr.begin(i) + rng.below(3); e < st.csr.end(i); e += 3) {
        st.csr.index[e] = dead_index;
        ++dead;
      }
      st.csr.size[i] -= dead;
      side_sum_t min_plain, max_plain, min_side, max_side;
      plain_fold(st, i, min_plain, max_plain);
      st.fold_row_activity(i, min_side, max_side);
      GPRS_CHECK(same_side(min_side, min_plain));
      GPRS_CHECK(same_side(max_side, max_plain));
      st.recompute_row_activity(i);
      GPRS_CHECK(same_side(stored_side(st.min_act, i), min_plain));
      GPRS_CHECK(same_side(stored_side(st.max_act, i), max_plain));
    }
  }
  GPRS_CHECK(negative_zero_products > 1000);
  GPRS_CHECK(infinite_terms > 1000);
  GPRS_CHECK(finite_terms > 1000);
}

// A lane that holds only infinite terms, or only terms that are zeros of either sign, is
// (+0, +0), as the lane of the definition that nothing was added to.
GPRS_TEST(state_fold_lanes_without_a_finite_term)
{
  for (int32_t n : {1, 2, 3, 4, 7, 8, 30, 64, 65, 200}) {
    for (int32_t kind = 0; kind < 3; ++kind) {
      model_t m;
      m.nrows     = 1;
      m.ncols     = n;
      m.row_start = {0, n};
      m.row_lower = {-inf};
      m.row_upper = {inf};
      for (int32_t j = 0; j < n; ++j) {
        m.col_index.push_back(j);
        // kind 0: every term infinite. kind 1: every term -0. kind 2: the two in turn.
        bool infinite = kind == 0 || (kind == 2 && j % 2 == 0);
        m.value.push_back(-1.);
        m.col_lower.push_back(infinite ? -inf : 0.);
        m.col_upper.push_back(infinite ? inf : 0.);
        m.cost.push_back(0.);
        m.col_type.push_back(0);
      }
      state_t st(m, tolerances_t());
      side_sum_t min_plain, max_plain;
      plain_fold(st, 0, min_plain, max_plain);
      GPRS_CHECK(same_side(stored_side(st.min_act, 0), min_plain));
      GPRS_CHECK(same_side(stored_side(st.max_act, 0), max_plain));
      GPRS_CHECK_EQ(bits(st.min_act.hi[0]), bits(0.));
      GPRS_CHECK_EQ(bits(st.min_act.lo[0]), bits(0.));
      GPRS_CHECK_EQ(bits(st.max_act.hi[0]), bits(0.));
      GPRS_CHECK_EQ(bits(st.max_act.lo[0]), bits(0.));
      GPRS_CHECK_EQ(st.min_act.max_term[0], 0.);
    }
  }
}
