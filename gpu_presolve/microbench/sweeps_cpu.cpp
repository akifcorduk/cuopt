/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cmath>
#include <cstdint>
#include <limits>

#ifndef SWEEPS_CPU_NS
#define SWEEPS_CPU_NS cpu_base
#endif

struct bound_t {
  double lb, ub;
};

// Must stay identical to the declaration in sweeps.cu.
struct cpu_view_t {
  int64_t nrows, ncols;
  const int64_t* row_start;
  const int32_t* col;
  const double* val;
  const int64_t* col_start;
  const int32_t* row;
  const double* col_val;
  const bound_t* bnd;
  const bound_t* row_side;
  const uint8_t* col_type;
  double* min_hi;
  double* min_lo;
  double* max_hi;
  double* max_lo;
  uint16_t* min_cnt;
  uint16_t* max_cnt;
  double* cand_lb;
  double* cand_ub;
};

namespace {

constexpr double big_term     = 1e20;
constexpr double int_eps      = 1e-6;
constexpr double inf          = std::numeric_limits<double>::infinity();
constexpr uint16_t count_mask = 0x7fff;
constexpr uint16_t lo_flag    = 0x8000;

struct side_acc_t {
  double hi = 0., lo = 0.;
  int cnt = 0;
};

template <bool COMP>
inline void add_term(side_acc_t& s, double a, double b)
{
  double p = a * b;
  if (!(std::fabs(p) <= big_term)) {
    ++s.cnt;
    return;
  }
  if (!COMP) {
    s.hi += p;
    return;
  }
  double pe = std::fma(a, b, -p);
  double t  = s.hi + p;
  double z  = t - s.hi;
  s.lo += ((s.hi - (t - z)) + (p - z)) + pe;
  s.hi = t;
}

inline uint16_t pack_count(const side_acc_t& s)
{
  return (s.cnt < count_mask ? s.cnt : count_mask) | (s.lo != 0. ? lo_flag : 0);
}

// Precondition: cnt <= 1. residual = activity - a * own, kept as (hi, lo) in the compensated case.
template <bool COMP>
inline bool residual_candidate(
  double hi, double lo, int cnt, double a, double own, double side, double& cand)
{
  double p     = a * own;
  bool own_inf = !(std::fabs(p) <= big_term);
  if (cnt == 1 && !own_inf) { return false; }
  if (!COMP) {
    cand = (side - (own_inf ? hi : hi - p)) / a;
    return true;
  }
  double rh = hi, rl = lo;
  if (!own_inf) {
    double pe = std::fma(a, own, -p);
    double t  = hi - p;
    double z  = t - hi;
    rl        = lo + (((hi - (t - z)) + (-p - z)) - pe);
    rh        = t;
  }
  double d = side - rh;
  double z = d - side;
  cand     = (d + (((side - (d - z)) + (-rh - z)) - rl)) / a;
  return true;
}

template <bool COMP>
void row_activity_impl(const cpu_view_t& v, const int32_t* rows, int64_t n)
{
  for (int64_t k = 0; k < n; ++k) {
    int64_t r = rows ? rows[k] : k;
    side_acc_t mn, mx;
    for (int64_t e = v.row_start[r]; e < v.row_start[r + 1]; ++e) {
      double a  = v.val[e];
      bound_t b = v.bnd[v.col[e]];
      add_term<COMP>(mn, a, a > 0 ? b.lb : b.ub);
      add_term<COMP>(mx, a, a > 0 ? b.ub : b.lb);
    }
    v.min_hi[r]  = mn.hi;
    v.max_hi[r]  = mx.hi;
    v.min_cnt[r] = pack_count(mn);
    v.max_cnt[r] = pack_count(mx);
    if (COMP) {
      v.min_lo[r] = mn.lo;
      v.max_lo[r] = mx.lo;
    }
  }
}

template <bool COMP>
void candidates_by_row_impl(const cpu_view_t& v, const int32_t* rows, int64_t n)
{
  for (int64_t k = 0; k < n; ++k) {
    int64_t r     = rows ? rows[k] : k;
    bound_t side  = v.row_side[r];
    uint16_t cmin = v.min_cnt[r], cmax = v.max_cnt[r];
    int nmin = cmin & count_mask, nmax = cmax & count_mask;
    bool use_min = side.ub < inf && nmin <= 1;
    bool use_max = side.lb > -inf && nmax <= 1;
    if (!use_min && !use_max) { continue; }
    double min_hi = v.min_hi[r], max_hi = v.max_hi[r];
    double min_lo = COMP && (cmin & lo_flag) ? v.min_lo[r] : 0.;
    double max_lo = COMP && (cmax & lo_flag) ? v.max_lo[r] : 0.;
    for (int64_t e = v.row_start[r]; e < v.row_start[r + 1]; ++e) {
      int32_t j  = v.col[e];
      double a   = v.val[e];
      bound_t xb = v.bnd[j];
      double lb = -inf, ub = inf, c;
      if (use_min &&
          residual_candidate<COMP>(min_hi, min_lo, nmin, a, a > 0 ? xb.lb : xb.ub, side.ub, c)) {
        (a > 0 ? ub : lb) = c;
      }
      if (use_max &&
          residual_candidate<COMP>(max_hi, max_lo, nmax, a, a > 0 ? xb.ub : xb.lb, side.lb, c)) {
        (a > 0 ? lb : ub) = c;
      }
      if (v.col_type[j]) {
        lb = std::ceil(lb - int_eps);
        ub = std::floor(ub + int_eps);
      }
      if (lb > v.cand_lb[j]) { v.cand_lb[j] = lb; }
      if (ub < v.cand_ub[j]) { v.cand_ub[j] = ub; }
    }
  }
}

template <bool COMP>
void candidates_by_column_impl(const cpu_view_t& v)
{
  for (int64_t j = 0; j < v.ncols; ++j) {
    bound_t xb = v.bnd[j];
    double lb = -inf, ub = inf, c;
    for (int64_t e = v.col_start[j]; e < v.col_start[j + 1]; ++e) {
      int32_t r    = v.row[e];
      double a     = v.col_val[e];
      bound_t side = v.row_side[r];
      if (side.ub < inf) {
        uint16_t cnt = v.min_cnt[r];
        if ((cnt & count_mask) <= 1 &&
            residual_candidate<COMP>(v.min_hi[r],
                                     COMP && (cnt & lo_flag) ? v.min_lo[r] : 0.,
                                     cnt & count_mask,
                                     a,
                                     a > 0 ? xb.lb : xb.ub,
                                     side.ub,
                                     c)) {
          if (a > 0) {
            ub = c < ub ? c : ub;
          } else {
            lb = c > lb ? c : lb;
          }
        }
      }
      if (side.lb > -inf) {
        uint16_t cnt = v.max_cnt[r];
        if ((cnt & count_mask) <= 1 &&
            residual_candidate<COMP>(v.max_hi[r],
                                     COMP && (cnt & lo_flag) ? v.max_lo[r] : 0.,
                                     cnt & count_mask,
                                     a,
                                     a > 0 ? xb.ub : xb.lb,
                                     side.lb,
                                     c)) {
          if (a > 0) {
            lb = c > lb ? c : lb;
          } else {
            ub = c < ub ? c : ub;
          }
        }
      }
    }
    if (v.col_type[j]) {
      lb = std::ceil(lb - int_eps);
      ub = std::floor(ub + int_eps);
    }
    v.cand_lb[j] = lb;
    v.cand_ub[j] = ub;
  }
}

}  // namespace

namespace SWEEPS_CPU_NS {

void row_activity(const cpu_view_t& v, bool comp, const int32_t* rows, int64_t n)
{
  if (comp) {
    row_activity_impl<true>(v, rows, n);
  } else {
    row_activity_impl<false>(v, rows, n);
  }
}

void candidates_by_row(const cpu_view_t& v, bool comp, const int32_t* rows, int64_t n)
{
  if (comp) {
    candidates_by_row_impl<true>(v, rows, n);
  } else {
    candidates_by_row_impl<false>(v, rows, n);
  }
}

void candidates_by_column(const cpu_view_t& v, bool comp)
{
  if (comp) {
    candidates_by_column_impl<true>(v);
  } else {
    candidates_by_column_impl<false>(v);
  }
}

}  // namespace SWEEPS_CPU_NS
