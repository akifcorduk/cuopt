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
#include <sstream>
#include <string>
#include <vector>

using namespace gprs;
using namespace state_test;

namespace {

// A sum of doubles without rounding: the parts are doubles whose real sum is the value
// (Shewchuk's grow-expansion: non-overlapping, increasing in magnitude).
struct exact_sum_t {
  std::vector<double> part;

  void add(double x)
  {
    size_t kept = 0;
    for (size_t k = 0; k < part.size(); ++k) {
      double error;
      x = two_sum(x, part[k], error);
      if (error != 0.) { part[kept++] = error; }
    }
    part.resize(kept);
    part.push_back(x);
  }

  void add_product(double a, double b)
  {
    double error;
    double p = two_prod(a, b, error);
    add(p);
    add(error);
  }

  // The parts do not overlap, so the largest carries the value and this sum loses nothing that
  // matters to a comparison with a bound.
  double value() const
  {
    double sum = 0.;
    for (double p : part) {
      sum += p;
    }
    return sum;
  }
};

// (hi + lo) minus the finite terms of one side of a row, exactly.
double stored_minus_exact(const state_t& st, int32_t row, bool min_side)
{
  const activity_t& act = min_side ? st.min_act : st.max_act;
  exact_sum_t sum;
  sum.add(act.hi[row]);
  sum.add(act.lo[row]);
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!st.csr.live(e)) { continue; }
    int32_t j    = st.csr.index[e];
    double a     = st.csr.value[e];
    double bound = min_side ? min_activity_bound(a, st.col_lower[j], st.col_upper[j])
                            : max_activity_bound(a, st.col_lower[j], st.col_upper[j]);
    if (is_infinite_term(a, bound, st.tol)) { continue; }
    sum.add_product(-a, bound);
  }
  return sum.value();
}

struct accuracy_options_t {
  int32_t nrows   = 0;
  int32_t ncols   = 0;
  double density  = 1.;
  bool mixed_sign = false;
  int32_t steps   = 0;
  double share    = 0.1;
};

// Fractions that are no dyadic numbers: every product and nearly every sum rounds.
double fraction(rng_t& rng, double from, double to) { return from + (to - from) * rng.uniform(); }

// Rows whose terms have one sign and similar size, so that the running sum is far above the
// largest term: the case in which the rounding errors of the lo additions are largest against
// max_term. Old and new bound differ by more than a factor of two, so that new - old rounds.
// Bound and coefficient changes, deletions and fixings, with the stored activities compared
// after every operation with the exact sum of the terms and through consistency_error with the
// fold.
void run_accuracy(uint64_t seed, const accuracy_options_t& o, double& worst_share)
{
  rng_t rng(seed);
  model_t m;
  m.nrows = o.nrows;
  m.ncols = o.ncols;
  m.row_start.push_back(0);
  for (int32_t i = 0; i < o.nrows; ++i) {
    for (int32_t j = 0; j < o.ncols; ++j) {
      if (!rng.chance(o.density)) { continue; }
      double sign = o.mixed_sign && rng.chance(0.5) ? -1. : 1.;
      m.col_index.push_back(j);
      m.value.push_back(sign * fraction(rng, 0.5, 1.5));
    }
    m.row_start.push_back(m.col_index.size());
  }
  m.row_lower.assign(o.nrows, -inf);
  m.row_upper.assign(o.nrows, 1e9);
  for (int32_t j = 0; j < o.ncols; ++j) {
    m.col_lower.push_back(fraction(rng, 0.25, 2.5));
    m.col_upper.push_back(fraction(rng, 3., 40.));
  }
  m.cost.assign(o.ncols, 0.);
  m.col_type.assign(o.ncols, 0);
  GPRS_REQUIRE(well_formed(m));

  tolerances_t tol;
  tol.activity_error_share = o.share;
  state_t st(m, tol);
  std::string where = "seed " + std::to_string(seed);
  for (int32_t s = 0; s < o.steps; ++s) {
    int32_t j    = rng.below(o.ncols);
    int32_t kind = rng.below(100);
    if (!st.is_live(j)) { continue; }
    if (kind < 40) {
      double v = fraction(rng, 0.25, 2.5);
      if (v == st.col_lower[j]) { continue; }
      st.set_col_lower(j, v);
    } else if (kind < 80) {
      double v = fraction(rng, 3., 40.);
      if (v == st.col_upper[j]) { continue; }
      st.set_col_upper(j, v);
    } else if (kind < 96) {
      if (st.csc.size[j] == 0) { continue; }
      int64_t p = st.csc.begin(j) + rng.below(st.csc.used[j]);
      if (!st.csc.live(p)) { continue; }
      double sign = st.csc.value[p] < 0 ? -1. : 1.;
      st.set_coefficient(st.csc.twin[p], sign * fraction(rng, 0.5, 1.5));
    } else if (kind < 99) {
      if (st.csc.size[j] == 0) { continue; }
      int64_t p = st.csc.begin(j) + rng.below(st.csc.used[j]);
      if (!st.csc.live(p)) { continue; }
      st.delete_entry(st.csc.twin[p]);
    } else {
      st.fix_col(j, fraction(rng, st.col_lower[j], st.col_upper[j]));
    }
    std::string error = st.consistency_error();
    if (!error.empty()) {
      test::fail(__FILE__, __LINE__, where + " step " + std::to_string(s) + ": " + error);
      return;
    }
    for (int32_t i = 0; i < o.nrows; ++i) {
      for (bool min_side : {true, false}) {
        const activity_t& act = min_side ? st.min_act : st.max_act;
        double error_bound    = proven_activity_error(act.updates[i], act.max_term[i]);
        double difference     = std::fabs(stored_minus_exact(st, i, min_side));
        if (!(difference <= error_bound)) {
          std::ostringstream out;
          out << where << " step " << s << " row " << i << ": the " << (min_side ? "min" : "max")
              << "imum activity is off the exact sum by " << difference << ", proven bound "
              << error_bound;
          test::fail(__FILE__, __LINE__, out.str());
          return;
        }
        if (error_bound > 0. && difference / error_bound > worst_share) {
          worst_share = difference / error_bound;
        }
      }
    }
  }
  st.rebuild();
  GPRS_CHECK_EQ(st.consistency_error(), std::string());
}

}  // namespace

// The exact sum itself: 1e16 + 1 - 1e16 is 1, and 0.1 * 3 - 0.3 is the rounding error of 0.3
// minus that of the product.
GPRS_TEST(state_accuracy_exact_sum)
{
  exact_sum_t sum;
  sum.add(1e16);
  sum.add(1.);
  sum.add(-1e16);
  GPRS_CHECK_EQ(sum.value(), 1.);
  exact_sum_t product;
  product.add_product(0.1, 3.);
  product.add(-0.3);
  GPRS_CHECK_EQ(product.value(), 0x1p-55);
}

// Same-sign rows from 8 to 300 entries (one lane per two entries, 256 lanes).
GPRS_TEST(state_accuracy_same_sign_rows)
{
  double worst = 0.;
  for (uint64_t seed = 1; seed <= 4; ++seed) {
    accuracy_options_t o;
    o.nrows = 3;
    o.ncols = seed == 1 ? 8 : seed == 2 ? 20 : seed == 3 ? 64 : 300;
    o.steps = 1500;
    run_accuracy(seed, o, worst);
  }
  // The stored sums are not exact, so the comparison above had something to compare.
  GPRS_CHECK(worst > 0.);
}

// The same with rows summed again in between (share reached after a few hundred updates) and
// with sparse rows of both signs.
GPRS_TEST(state_accuracy_with_resummation)
{
  double worst = 0.;
  for (uint64_t seed = 11; seed <= 14; ++seed) {
    accuracy_options_t o;
    o.nrows      = 6;
    o.ncols      = seed <= 12 ? 40 : 150;
    o.density    = 0.6;
    o.mixed_sign = seed % 2 == 0;
    o.steps      = 1500;
    o.share      = 4e-23;
    run_accuracy(seed, o, worst);
  }
  GPRS_CHECK(worst > 0.);
}
