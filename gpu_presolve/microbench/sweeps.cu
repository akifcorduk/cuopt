/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/version.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#define CHECK(cond)                                                                     \
  do {                                                                                  \
    if (!(cond)) {                                                                      \
      fprintf(stderr, "sweeps: check failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
      abort();                                                                          \
    }                                                                                   \
  } while (0)

#define CUDA_CHECK(call)                         \
  do {                                           \
    cudaError_t cuda_status_ = (call);           \
    if (cuda_status_ != cudaSuccess) {           \
      fprintf(stderr,                            \
              "sweeps: %s failed: %s (%s:%d)\n", \
              #call,                             \
              cudaGetErrorString(cuda_status_),  \
              __FILE__,                          \
              __LINE__);                         \
      abort();                                   \
    }                                            \
  } while (0)

struct bound_t {
  double lb, ub;
};

// Must stay identical to the declaration in sweeps_cpu.cpp.
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

namespace cpu_base {
void row_activity(const cpu_view_t& v, bool comp, const int32_t* rows, int64_t n);
void candidates_by_row(const cpu_view_t& v, bool comp, const int32_t* rows, int64_t n);
void candidates_by_column(const cpu_view_t& v, bool comp);
}  // namespace cpu_base

namespace cpu_fma {
void row_activity(const cpu_view_t& v, bool comp, const int32_t* rows, int64_t n);
void candidates_by_row(const cpu_view_t& v, bool comp, const int32_t* rows, int64_t n);
void candidates_by_column(const cpu_view_t& v, bool comp);
}  // namespace cpu_fma

namespace {

constexpr double big_term       = 1e20;
constexpr double int_eps        = 1e-6;
constexpr double inf            = std::numeric_limits<double>::infinity();
constexpr int block_dim         = 256;
constexpr int max_light_len     = 64;
constexpr int max_block_len     = 8192;
constexpr int n_light_classes   = 6;
constexpr int block_class       = n_light_classes;
constexpr int chunk_class       = n_light_classes + 1;
constexpr int n_classes         = n_light_classes + 2;
constexpr int walk_chunk        = 32;
constexpr int walk_linear_steps = 8;
constexpr uint16_t count_mask   = 0x7fff;
constexpr uint16_t lo_flag      = 0x8000;
constexpr double agree_tol      = 1e-9;

enum { PAIRED = 0, SPLIT = 1 };
enum { REDUCE = 0, ATOMIC = 1 };
enum { PLAIN = 0, COMP = 1 };

struct pair_t {
  double sum, count;
};

struct dd_t {
  double hi, lo;
};

struct matrix_view_t {
  const int64_t* start;
  const int32_t* index;
  const double* value;
};

struct act_view_t {
  pair_t* min_p;
  pair_t* max_p;
  double* min_hi;
  double* max_hi;
  uint16_t* min_cnt;
  uint16_t* max_cnt;
  double* min_lo;
  double* max_lo;
};

struct cand_view_t {
  matrix_view_t csc;
  const bound_t* bnd;
  const bound_t* row_side;
  const uint8_t* col_type;
  act_view_t act;
};

struct side_acc_t {
  double hi, lo;
  int cnt;
};

struct part_t {
  double min_hi, min_lo, max_hi, max_lo;
  int min_cnt, max_cnt;
};

// Rows or columns above max_block_len entries, cut into chunks of max_block_len: chunk k
// belongs to chunk_item[k] and is its chunk_index[k]-th; item i owns chunks
// [item_start[i], item_start[i + 1]).
struct chunk_view_t {
  const int32_t* chunk_item;
  const int32_t* chunk_index;
  const int32_t* items;
  const int32_t* item_start;
  int n_chunks, n_items;
};

// Class c of light_plan_t covers warps [warp_begin[c], warp_begin[c + 1]) and items
// [row_begin[c], row_begin[c + 1]); every item of class c is folded by 2^c lanes.
struct light_plan_t {
  int64_t warp_begin[n_light_classes + 1];
  int64_t row_begin[n_light_classes + 1];
  int n_classes;
};

struct light_slot_t {
  int c, g, sub;
  int64_t k;
  bool valid;
};

struct max_op_t {
  __device__ double operator()(double a, double b) const { return a > b ? a : b; }
};

struct min_op_t {
  __device__ double operator()(double a, double b) const { return a < b ? a : b; }
};

struct plus_i32_t {
  __host__ __device__ int32_t operator()(int32_t a, int32_t b) const { return a + b; }
};

struct plus_i64_t {
  __host__ __device__ int64_t operator()(int64_t a, int64_t b) const { return a + b; }
};

template <int ARITH>
__device__ __forceinline__ void add_term(side_acc_t& s, double a, double b)
{
  double p = a * b;
  if (!(fabs(p) <= big_term)) {
    ++s.cnt;
    return;
  }
  if constexpr (ARITH == PLAIN) {
    s.hi += p;
  } else {
    double pe = fma(a, b, -p);
    double t  = s.hi + p;
    double z  = t - s.hi;
    s.lo += ((s.hi - (t - z)) + (p - z)) + pe;
    s.hi = t;
  }
}

__device__ __forceinline__ dd_t dd_add(dd_t a, dd_t b)
{
  double t = a.hi + b.hi;
  double z = t - a.hi;
  return dd_t{t, (a.lo + b.lo) + ((a.hi - (t - z)) + (b.hi - z))};
}

struct dd_add_t {
  __device__ dd_t operator()(const dd_t& a, const dd_t& b) const { return dd_add(a, b); }
};

template <int ARITH>
__device__ __forceinline__ void merge(side_acc_t& s, double hi, double lo)
{
  if constexpr (ARITH == PLAIN) {
    s.hi += hi;
  } else {
    dd_t r = dd_add(dd_t{s.hi, s.lo}, dd_t{hi, lo});
    s.hi   = r.hi;
    s.lo   = r.lo;
  }
}

template <int ARITH>
__device__ __forceinline__ void shfl_merge(side_acc_t& s, int off)
{
  double hi = __shfl_down_sync(0xffffffff, s.hi, off);
  double lo = 0.;
  if constexpr (ARITH == COMP) { lo = __shfl_down_sync(0xffffffff, s.lo, off); }
  merge<ARITH>(s, hi, lo);
}

template <int ARITH>
__device__ __forceinline__ void block_reduce_sums(side_acc_t& mn, side_acc_t& mx)
{
  using sum_t    = std::conditional_t<ARITH == COMP, dd_t, double>;
  using reduce_t = cub::BlockReduce<sum_t, block_dim>;
  __shared__ typename reduce_t::TempStorage storage;
  if constexpr (ARITH == COMP) {
    dd_t a = reduce_t(storage).Reduce(dd_t{mn.hi, mn.lo}, dd_add_t{});
    __syncthreads();
    dd_t b = reduce_t(storage).Reduce(dd_t{mx.hi, mx.lo}, dd_add_t{});
    mn.hi  = a.hi;
    mn.lo  = a.lo;
    mx.hi  = b.hi;
    mx.lo  = b.lo;
  } else {
    mn.hi = reduce_t(storage).Sum(mn.hi);
    __syncthreads();
    mx.hi = reduce_t(storage).Sum(mx.hi);
  }
}

template <typename cnt_t>
__device__ __forceinline__ void block_reduce_counts(side_acc_t& mn, side_acc_t& mx)
{
  using reduce_t = cub::BlockReduce<cnt_t, block_dim>;
  __shared__ typename reduce_t::TempStorage storage;
  cnt_t a = mn.cnt, b = mx.cnt;
  a = reduce_t(storage).Sum(a);
  __syncthreads();
  b      = reduce_t(storage).Sum(b);
  mn.cnt = a;
  mx.cnt = b;
}

__device__ __forceinline__ uint16_t pack_count(const side_acc_t& s)
{
  return (s.cnt < count_mask ? s.cnt : count_mask) | (s.lo != 0. ? lo_flag : 0);
}

template <int LAYOUT, int ARITH>
__device__ __forceinline__ void store(act_view_t out,
                                      int row,
                                      const side_acc_t& mn,
                                      const side_acc_t& mx)
{
  if constexpr (LAYOUT == PAIRED) {
    out.min_p[row] = pair_t{mn.hi, (double)mn.cnt};
    out.max_p[row] = pair_t{mx.hi, (double)mx.cnt};
  } else {
    out.min_hi[row]  = mn.hi;
    out.max_hi[row]  = mx.hi;
    out.min_cnt[row] = pack_count(mn);
    out.max_cnt[row] = pack_count(mx);
  }
  if constexpr (ARITH == COMP) {
    out.min_lo[row] = mn.lo;
    out.max_lo[row] = mx.lo;
  }
}

template <int ARITH>
__device__ __forceinline__ void scan_entries(matrix_view_t m,
                                             const bound_t* bnd,
                                             int64_t beg,
                                             int64_t end,
                                             int step,
                                             side_acc_t& mn,
                                             side_acc_t& mx)
{
  for (int64_t e = beg; e < end; e += step) {
    double a  = m.value[e];
    bound_t b = bnd[m.index[e]];
    add_term<ARITH>(mn, a, a > 0 ? b.lb : b.ub);
    add_term<ARITH>(mx, a, a > 0 ? b.ub : b.lb);
  }
}

__device__ __forceinline__ light_slot_t light_slot(const light_plan_t& plan)
{
  int64_t warp = ((int64_t)blockIdx.x * block_dim + threadIdx.x) >> 5;
  int lane     = threadIdx.x & 31;
  light_slot_t s;
  s.c = 0;
  while (s.c + 1 < plan.n_classes && warp >= plan.warp_begin[s.c + 1]) {
    ++s.c;
  }
  s.g     = 1 << s.c;
  s.sub   = lane & (s.g - 1);
  s.k     = plan.row_begin[s.c] + (warp - plan.warp_begin[s.c]) * (32 >> s.c) + (lane >> s.c);
  s.valid = s.k < plan.row_begin[s.c + 1];
  return s;
}

// One launch for all rows of at most max_light_len entries. A row of class c is folded by a
// group of 2^c adjacent lanes; the group result is a tree fixed by lane position.
template <int LAYOUT, int COUNT, int ARITH>
__global__ void act_light_kernel(matrix_view_t m,
                                 const bound_t* bnd,
                                 light_plan_t plan,
                                 const int32_t* list,
                                 const int32_t* out_row,
                                 act_view_t out)
{
  __shared__ int s_cnt[2][block_dim];
  light_slot_t t = light_slot(plan);
  int r          = 0;
  side_acc_t mn{0., 0., 0}, mx{0., 0., 0};
  if (t.valid) {
    r = list ? list[t.k] : t.k;
    scan_entries<ARITH>(m, bnd, m.start[r] + t.sub, m.start[r + 1], t.g, mn, mx);
  }
  if (t.c > 0) {
    if constexpr (COUNT == ATOMIC) {
      int slot = threadIdx.x - t.sub;
      if (t.sub == 0) {
        s_cnt[0][slot] = 0;
        s_cnt[1][slot] = 0;
      }
      __syncwarp();
      if (mn.cnt) { atomicAdd(&s_cnt[0][slot], mn.cnt); }
      if (mx.cnt) { atomicAdd(&s_cnt[1][slot], mx.cnt); }
      __syncwarp();
      mn.cnt = s_cnt[0][slot];
      mx.cnt = s_cnt[1][slot];
    }
    for (int off = t.g >> 1; off > 0; off >>= 1) {
      shfl_merge<ARITH>(mn, off);
      shfl_merge<ARITH>(mx, off);
      if constexpr (COUNT == REDUCE) {
        mn.cnt += __shfl_down_sync(0xffffffff, mn.cnt, off);
        mx.cnt += __shfl_down_sync(0xffffffff, mx.cnt, off);
      }
    }
  }
  if (t.valid && t.sub == 0) { store<LAYOUT, ARITH>(out, out_row ? out_row[r] : r, mn, mx); }
}

// Entries [beg, end) folded by one block; the result is valid in thread 0.
template <int COUNT, int ARITH, typename cnt_t>
__device__ __forceinline__ void block_fold(
  matrix_view_t m, const bound_t* bnd, int64_t beg, int64_t end, side_acc_t& mn, side_acc_t& mx)
{
  __shared__ int s_cnt[2];
  if constexpr (COUNT == ATOMIC) {
    if (threadIdx.x == 0) {
      s_cnt[0] = 0;
      s_cnt[1] = 0;
    }
    __syncthreads();
  }
  scan_entries<ARITH>(m, bnd, beg + threadIdx.x, end, block_dim, mn, mx);
  if constexpr (COUNT == ATOMIC) {
    if (mn.cnt) { atomicAdd(&s_cnt[0], mn.cnt); }
    if (mx.cnt) { atomicAdd(&s_cnt[1], mx.cnt); }
  }
  block_reduce_sums<ARITH>(mn, mx);
  if constexpr (COUNT == REDUCE) {
    block_reduce_counts<cnt_t>(mn, mx);
  } else {
    __syncthreads();
    mn.cnt = s_cnt[0];
    mx.cnt = s_cnt[1];
  }
}

// One block per row: the cuOpt style when launched over all rows, the 65 to 8192 bin otherwise.
template <int LAYOUT, int COUNT, int ARITH>
__global__ void act_block_kernel(matrix_view_t m,
                                 const bound_t* bnd,
                                 int first,
                                 const int32_t* list,
                                 const int32_t* out_row,
                                 act_view_t out)
{
  using cnt_t = std::conditional_t<LAYOUT == PAIRED, double, int>;
  int k       = first + blockIdx.x;
  int r       = list ? list[k] : k;
  side_acc_t mn{0., 0., 0}, mx{0., 0., 0};
  block_fold<COUNT, ARITH, cnt_t>(m, bnd, m.start[r], m.start[r + 1], mn, mx);
  if (threadIdx.x == 0) { store<LAYOUT, ARITH>(out, out_row ? out_row[r] : r, mn, mx); }
}

// Rows above max_block_len entries: one block per chunk, then one block per row over the
// partial results. With a flag array only flagged rows are computed.
template <int COUNT, int ARITH>
__global__ void act_chunk_kernel(
  matrix_view_t m, const bound_t* bnd, chunk_view_t chunks, const uint8_t* flag, part_t* part)
{
  int r = chunks.chunk_item[blockIdx.x];
  if (flag && !flag[r]) { return; }
  int64_t beg = m.start[r] + (int64_t)chunks.chunk_index[blockIdx.x] * max_block_len;
  int64_t end = beg + max_block_len < m.start[r + 1] ? beg + max_block_len : m.start[r + 1];
  side_acc_t mn{0., 0., 0}, mx{0., 0., 0};
  block_fold<COUNT, ARITH, int>(m, bnd, beg, end, mn, mx);
  if (threadIdx.x == 0) { part[blockIdx.x] = part_t{mn.hi, mn.lo, mx.hi, mx.lo, mn.cnt, mx.cnt}; }
}

template <int LAYOUT, int ARITH>
__global__ void act_chunk_finalize_kernel(const part_t* part,
                                          chunk_view_t chunks,
                                          const uint8_t* flag,
                                          const int32_t* out_row,
                                          act_view_t out)
{
  int r = chunks.items[blockIdx.x];
  if (flag && !flag[r]) { return; }
  side_acc_t mn{0., 0., 0}, mx{0., 0., 0};
  int end = chunks.item_start[blockIdx.x + 1];
  for (int k = chunks.item_start[blockIdx.x] + threadIdx.x; k < end; k += block_dim) {
    part_t p = part[k];
    merge<ARITH>(mn, p.min_hi, p.min_lo);
    merge<ARITH>(mx, p.max_hi, p.max_lo);
    mn.cnt += p.min_cnt;
    mx.cnt += p.max_cnt;
  }
  block_reduce_sums<ARITH>(mn, mx);
  block_reduce_counts<int>(mn, mx);
  if (threadIdx.x == 0) { store<LAYOUT, ARITH>(out, out_row[r], mn, mx); }
}

template <int LAYOUT, int ARITH, bool GATED>
__device__ __forceinline__ bool load_side(const pair_t* pair,
                                          const double* hi_arr,
                                          const uint16_t* cnt_arr,
                                          const double* lo_arr,
                                          int row,
                                          double& hi,
                                          double& lo,
                                          int& cnt)
{
  lo = 0.;
  if constexpr (LAYOUT == PAIRED) {
    pair_t x = pair[row];
    if (x.count > 1.) { return false; }
    hi  = x.sum;
    cnt = x.count;
    if constexpr (ARITH == COMP) { lo = lo_arr[row]; }
  } else {
    uint16_t c = cnt_arr[row];
    cnt        = c & count_mask;
    if (cnt > 1) { return false; }
    hi = hi_arr[row];
    if constexpr (ARITH == COMP) {
      if (!GATED || (c & lo_flag)) { lo = lo_arr[row]; }
    }
  }
  return true;
}

// Precondition: cnt <= 1. residual = activity - a * own, kept as (hi, lo) in the compensated case.
template <int ARITH>
__device__ __forceinline__ bool residual_candidate(
  double hi, double lo, int cnt, double a, double own, double side, double& cand)
{
  double p     = a * own;
  bool own_inf = !(fabs(p) <= big_term);
  if (cnt == 1 && !own_inf) { return false; }
  if constexpr (ARITH == PLAIN) {
    cand = (side - (own_inf ? hi : hi - p)) / a;
  } else {
    double rh = hi, rl = lo;
    if (!own_inf) {
      double pe = fma(a, own, -p);
      double t  = hi - p;
      double z  = t - hi;
      rl        = lo + (((hi - (t - z)) + (-p - z)) - pe);
      rh        = t;
    }
    double d = side - rh;
    double z = d - side;
    cand     = (d + (((side - (d - z)) + (-rh - z)) - rl)) / a;
  }
  return true;
}

template <int LAYOUT, int ARITH, bool GATED>
__device__ __forceinline__ void entry_candidates(
  const cand_view_t& v, int row, double a, bound_t xb, double& lb, double& ub)
{
  bound_t side = v.row_side[row];
  double hi, lo, c;
  int cnt;
  if (side.ub < inf &&
      load_side<LAYOUT, ARITH, GATED>(
        v.act.min_p, v.act.min_hi, v.act.min_cnt, v.act.min_lo, row, hi, lo, cnt) &&
      residual_candidate<ARITH>(hi, lo, cnt, a, a > 0 ? xb.lb : xb.ub, side.ub, c)) {
    if (a > 0) {
      ub = c < ub ? c : ub;
    } else {
      lb = c > lb ? c : lb;
    }
  }
  if (side.lb > -inf &&
      load_side<LAYOUT, ARITH, GATED>(
        v.act.max_p, v.act.max_hi, v.act.max_cnt, v.act.max_lo, row, hi, lo, cnt) &&
      residual_candidate<ARITH>(hi, lo, cnt, a, a > 0 ? xb.ub : xb.lb, side.lb, c)) {
    if (a > 0) {
      lb = c > lb ? c : lb;
    } else {
      ub = c < ub ? c : ub;
    }
  }
}

__device__ __forceinline__ void store_candidate(
  const cand_view_t& v, int j, double lb, double ub, double* cand_lb, double* cand_ub)
{
  if (v.col_type[j]) {
    lb = ceil(lb - int_eps);
    ub = floor(ub + int_eps);
  }
  cand_lb[j] = lb;
  cand_ub[j] = ub;
}

// Columns of at most max_light_len entries, grouped like the rows of act_light_kernel; order
// lists the columns by length class.
template <int LAYOUT, int ARITH, bool GATED>
__global__ void cand_light_kernel(
  cand_view_t v, light_plan_t plan, const int32_t* order, double* cand_lb, double* cand_ub)
{
  light_slot_t t = light_slot(plan);
  int j          = 0;
  double lb = -inf, ub = inf;
  if (t.valid) {
    j           = order[t.k];
    bound_t xb  = v.bnd[j];
    int64_t end = v.csc.start[j + 1];
    for (int64_t e = v.csc.start[j] + t.sub; e < end; e += t.g) {
      entry_candidates<LAYOUT, ARITH, GATED>(v, v.csc.index[e], v.csc.value[e], xb, lb, ub);
    }
  }
  for (int off = t.g >> 1; off > 0; off >>= 1) {
    double other_lb = __shfl_down_sync(0xffffffff, lb, off);
    double other_ub = __shfl_down_sync(0xffffffff, ub, off);
    lb              = other_lb > lb ? other_lb : lb;
    ub              = other_ub < ub ? other_ub : ub;
  }
  if (t.valid && t.sub == 0) { store_candidate(v, j, lb, ub, cand_lb, cand_ub); }
}

__device__ __forceinline__ void block_reduce_bounds(double& lb, double& ub)
{
  using reduce_t = cub::BlockReduce<double, block_dim>;
  __shared__ typename reduce_t::TempStorage storage;
  lb = reduce_t(storage).Reduce(lb, max_op_t{});
  __syncthreads();
  ub = reduce_t(storage).Reduce(ub, min_op_t{});
}

template <int LAYOUT, int ARITH, bool GATED>
__device__ __forceinline__ void block_candidates(
  const cand_view_t& v, int j, int64_t beg, int64_t end, double& lb, double& ub)
{
  bound_t xb = v.bnd[j];
  for (int64_t e = beg + threadIdx.x; e < end; e += block_dim) {
    entry_candidates<LAYOUT, ARITH, GATED>(v, v.csc.index[e], v.csc.value[e], xb, lb, ub);
  }
  block_reduce_bounds(lb, ub);
}

template <int LAYOUT, int ARITH, bool GATED>
__global__ void cand_block_kernel(cand_view_t v,
                                  const int32_t* block_cols,
                                  double* cand_lb,
                                  double* cand_ub)
{
  int j     = block_cols[blockIdx.x];
  double lb = -inf, ub = inf;
  block_candidates<LAYOUT, ARITH, GATED>(v, j, v.csc.start[j], v.csc.start[j + 1], lb, ub);
  if (threadIdx.x == 0) { store_candidate(v, j, lb, ub, cand_lb, cand_ub); }
}

template <int LAYOUT, int ARITH, bool GATED>
__global__ void cand_chunk_kernel(cand_view_t v, chunk_view_t chunks, bound_t* part)
{
  int j       = chunks.chunk_item[blockIdx.x];
  int64_t beg = v.csc.start[j] + (int64_t)chunks.chunk_index[blockIdx.x] * max_block_len;
  int64_t end = beg + max_block_len < v.csc.start[j + 1] ? beg + max_block_len : v.csc.start[j + 1];
  double lb = -inf, ub = inf;
  block_candidates<LAYOUT, ARITH, GATED>(v, j, beg, end, lb, ub);
  if (threadIdx.x == 0) { part[blockIdx.x] = bound_t{lb, ub}; }
}

__global__ void cand_chunk_finalize_kernel(
  cand_view_t v, const bound_t* part, chunk_view_t chunks, double* cand_lb, double* cand_ub)
{
  double lb = -inf, ub = inf;
  int end = chunks.item_start[blockIdx.x + 1];
  for (int k = chunks.item_start[blockIdx.x] + threadIdx.x; k < end; k += block_dim) {
    bound_t p = part[k];
    lb        = p.lb > lb ? p.lb : lb;
    ub        = p.ub < ub ? p.ub : ub;
  }
  block_reduce_bounds(lb, ub);
  if (threadIdx.x == 0) { store_candidate(v, chunks.items[blockIdx.x], lb, ub, cand_lb, cand_ub); }
}

// Last outer index in [lo, hi) whose start is <= e; start[lo] <= e < start[hi] on entry.
__device__ __forceinline__ int outer_of(const int64_t* start, int lo, int hi, int64_t e)
{
  while (hi - lo > 1) {
    int mid = lo + (hi - lo) / 2;
    if (start[mid] <= e) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Thread t owns entries [t * walk_chunk, (t + 1) * walk_chunk) and walks the offsets with them,
// so the cost does not depend on how the entries are distributed over rows or columns. A long
// run of empty rows or columns is skipped by a search, not one by one.
template <typename op_t>
__global__ void walk_entries_kernel(const int64_t* start, int n_outer, int64_t nnz, op_t op)
{
  int64_t e = ((int64_t)blockIdx.x * block_dim + threadIdx.x) * walk_chunk;
  if (e >= nnz) { return; }
  int64_t end = e + walk_chunk < nnz ? e + walk_chunk : nnz;
  int r       = outer_of(start, 0, n_outer, e);
  for (; e < end; ++e) {
    for (int step = 0; start[r + 1] <= e; ++step) {
      if (step == walk_linear_steps) {
        r = outer_of(start, r, n_outer, e);
        break;
      }
      ++r;
    }
    op(e, r);
  }
}

struct keep_op_t {
  const int32_t* index;
  const uint8_t* outer_alive;
  const uint8_t* inner_alive;
  uint8_t* keep;
  __device__ void operator()(int64_t e, int r) const
  {
    keep[e] = outer_alive[r] & inner_alive[index[e]];
  }
};

struct expand_op_t {
  int32_t* entry_outer;
  int32_t* position;
  __device__ void operator()(int64_t e, int r) const
  {
    entry_outer[e] = r;
    position[e]    = e;
  }
};

struct scatter_op_t {
  const int32_t* index;
  const double* value;
  unsigned long long* cursor;
  int32_t* out_index;
  double* out_value;
  __device__ void operator()(int64_t e, int r) const
  {
    unsigned long long slot = atomicAdd(&cursor[index[e]], 1ull);
    out_index[slot]         = r;
    out_value[slot]         = value[e];
  }
};

__global__ void compact_scatter_kernel(int64_t nnz,
                                       const uint8_t* keep,
                                       const int32_t* pos,
                                       const int32_t* index,
                                       const double* value,
                                       const int32_t* new_inner,
                                       int32_t* out_index,
                                       double* out_value)
{
  int64_t e = (int64_t)blockIdx.x * block_dim + threadIdx.x;
  if (e < nnz && keep[e]) {
    int p        = pos[e];
    out_index[p] = new_inner[index[e]];
    out_value[p] = value[e];
  }
}

__global__ void compact_start_kernel(int n_outer,
                                     const int64_t* start,
                                     const uint8_t* outer_alive,
                                     const int32_t* new_outer,
                                     const int32_t* pos,
                                     int64_t* out_start)
{
  int r = blockIdx.x * block_dim + threadIdx.x;
  if (r > n_outer) { return; }
  if (r == n_outer || outer_alive[r]) { out_start[new_outer[r]] = pos[start[r]]; }
}

__global__ void histogram_kernel(int64_t nnz, const int32_t* index, int32_t* count)
{
  int64_t e = (int64_t)blockIdx.x * block_dim + threadIdx.x;
  if (e < nnz) { atomicAdd(&count[index[e]], 1); }
}

__global__ void gather_kernel(int64_t nnz,
                              const int32_t* perm,
                              const int32_t* entry_outer,
                              const double* value,
                              int32_t* out_index,
                              double* out_value)
{
  int64_t e = (int64_t)blockIdx.x * block_dim + threadIdx.x;
  if (e < nnz) {
    int p        = perm[e];
    out_index[e] = entry_outer[p];
    out_value[e] = value[p];
  }
}

template <int L, int C, int A>
void launch_act_light(int64_t blocks,
                      matrix_view_t m,
                      const bound_t* bnd,
                      light_plan_t plan,
                      const int32_t* list,
                      const int32_t* out_row,
                      act_view_t out)
{
  act_light_kernel<L, C, A><<<blocks, block_dim>>>(m, bnd, plan, list, out_row, out);
}

template <int L, int C, int A>
void launch_act_block(int64_t blocks,
                      matrix_view_t m,
                      const bound_t* bnd,
                      int first,
                      const int32_t* list,
                      const int32_t* out_row,
                      act_view_t out)
{
  act_block_kernel<L, C, A><<<blocks, block_dim>>>(m, bnd, first, list, out_row, out);
}

template <int C, int A>
void launch_act_chunk(
  matrix_view_t m, const bound_t* bnd, chunk_view_t chunks, const uint8_t* flag, part_t* part)
{
  act_chunk_kernel<C, A><<<chunks.n_chunks, block_dim>>>(m, bnd, chunks, flag, part);
}

template <int L, int A>
void launch_act_chunk_finalize(const part_t* part,
                               chunk_view_t chunks,
                               const uint8_t* flag,
                               const int32_t* out_row,
                               act_view_t out)
{
  act_chunk_finalize_kernel<L, A><<<chunks.n_items, block_dim>>>(part, chunks, flag, out_row, out);
}

struct col_sched_t {
  light_plan_t plan;
  int64_t light_blocks;
  const int32_t* order;
  int block_begin, n_block;
  chunk_view_t chunks;
  bound_t* part;
};

template <int L, int A, bool G>
void launch_cand(cand_view_t v, const col_sched_t& s, double* cand_lb, double* cand_ub)
{
  if (s.light_blocks > 0) {
    cand_light_kernel<L, A, G><<<s.light_blocks, block_dim>>>(v, s.plan, s.order, cand_lb, cand_ub);
  }
  if (s.n_block > 0) {
    cand_block_kernel<L, A, G>
      <<<s.n_block, block_dim>>>(v, s.order + s.block_begin, cand_lb, cand_ub);
  }
  if (s.chunks.n_chunks > 0) {
    cand_chunk_kernel<L, A, G><<<s.chunks.n_chunks, block_dim>>>(v, s.chunks, s.part);
    cand_chunk_finalize_kernel<<<s.chunks.n_items, block_dim>>>(
      v, s.part, s.chunks, cand_lb, cand_ub);
  }
}

using act_light_fn_t = void (*)(
  int64_t, matrix_view_t, const bound_t*, light_plan_t, const int32_t*, const int32_t*, act_view_t);
using act_block_fn_t =
  void (*)(int64_t, matrix_view_t, const bound_t*, int, const int32_t*, const int32_t*, act_view_t);
using act_chunk_fn_t =
  void (*)(matrix_view_t, const bound_t*, chunk_view_t, const uint8_t*, part_t*);
using act_chunk_finalize_fn_t =
  void (*)(const part_t*, chunk_view_t, const uint8_t*, const int32_t*, act_view_t);
using cand_fn_t = void (*)(cand_view_t, const col_sched_t&, double*, double*);

// Index = layout * 4 + count * 2 + arith.
const act_light_fn_t act_light_table[8] = {launch_act_light<0, 0, 0>,
                                           launch_act_light<0, 0, 1>,
                                           launch_act_light<0, 1, 0>,
                                           launch_act_light<0, 1, 1>,
                                           launch_act_light<1, 0, 0>,
                                           launch_act_light<1, 0, 1>,
                                           launch_act_light<1, 1, 0>,
                                           launch_act_light<1, 1, 1>};
const act_block_fn_t act_block_table[8] = {launch_act_block<0, 0, 0>,
                                           launch_act_block<0, 0, 1>,
                                           launch_act_block<0, 1, 0>,
                                           launch_act_block<0, 1, 1>,
                                           launch_act_block<1, 0, 0>,
                                           launch_act_block<1, 0, 1>,
                                           launch_act_block<1, 1, 0>,
                                           launch_act_block<1, 1, 1>};
// Index = count * 2 + arith, and layout * 2 + arith.
const act_chunk_fn_t act_chunk_table[4] = {
  launch_act_chunk<0, 0>, launch_act_chunk<0, 1>, launch_act_chunk<1, 0>, launch_act_chunk<1, 1>};
const act_chunk_finalize_fn_t act_chunk_finalize_table[4] = {launch_act_chunk_finalize<0, 0>,
                                                             launch_act_chunk_finalize<0, 1>,
                                                             launch_act_chunk_finalize<1, 0>,
                                                             launch_act_chunk_finalize<1, 1>};

struct rng_t {
  uint64_t s;
  uint64_t next()
  {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  double uniform() { return (next() >> 11) * 0x1.0p-53; }
  int64_t below(int64_t n) { return next() % n; }
};

struct model_t {
  int64_t nrows = 0, ncols = 0, nnz = 0;
  std::vector<int64_t> row_start;
  std::vector<int32_t> col;
  std::vector<double> val, row_lower, row_upper, col_lower, col_upper, cost;
  std::vector<uint8_t> col_type;
  double objective_offset = 0.;
};

template <typename T>
void read_array(FILE* f, std::vector<T>& v, size_t n)
{
  v.resize(n);
  CHECK(fread(v.data(), sizeof(T), n, f) == n);
}

void validate(const model_t& m)
{
  CHECK(m.nrows > 0 && m.ncols > 0 && m.nnz > 0);
  CHECK(m.nrows < INT32_MAX && m.ncols < INT32_MAX && m.nnz < INT32_MAX);
  CHECK(m.row_start[0] == 0 && m.row_start[m.nrows] == m.nnz);
  for (int64_t r = 0; r < m.nrows; ++r) {
    CHECK(m.row_start[r] <= m.row_start[r + 1]);
  }
  for (int64_t e = 0; e < m.nnz; ++e) {
    CHECK(m.col[e] >= 0 && m.col[e] < m.ncols);
    CHECK(m.val[e] != 0. && std::isfinite(m.val[e]));
  }
  for (int64_t j = 0; j < m.ncols; ++j) {
    CHECK(m.col_lower[j] == m.col_lower[j] && m.col_upper[j] == m.col_upper[j]);
  }
  for (int64_t r = 0; r < m.nrows; ++r) {
    CHECK(m.row_lower[r] == m.row_lower[r] && m.row_upper[r] == m.row_upper[r]);
  }
}

bool load_dump(const char* path, model_t& m)
{
  FILE* f = fopen(path, "rb");
  if (!f) { return false; }
  char magic[8];
  CHECK(fread(magic, 1, 8, f) == 8 && memcmp(magic, "GPRSCSR1", 8) == 0);
  int64_t dims[3];
  CHECK(fread(dims, sizeof(int64_t), 3, f) == 3);
  m.nrows = dims[0];
  m.ncols = dims[1];
  m.nnz   = dims[2];
  CHECK(m.nrows >= 0 && m.ncols >= 0 && m.nnz >= 0);
  read_array(f, m.row_start, m.nrows + 1);
  read_array(f, m.col, m.nnz);
  read_array(f, m.val, m.nnz);
  read_array(f, m.row_lower, m.nrows);
  read_array(f, m.row_upper, m.nrows);
  read_array(f, m.col_lower, m.ncols);
  read_array(f, m.col_upper, m.ncols);
  read_array(f, m.cost, m.ncols);
  read_array(f, m.col_type, m.ncols);
  CHECK(fread(&m.objective_offset, sizeof(double), 1, f) == 1);
  CHECK(fgetc(f) == EOF);
  fclose(f);
  return true;
}

void make_synthetic(int64_t nrows, int64_t ncols, int64_t nnz_per_row, model_t& m)
{
  rng_t rng{0x5eed0001ull};
  m.nrows = nrows;
  m.ncols = ncols;
  m.col_lower.resize(ncols);
  m.col_upper.resize(ncols);
  m.cost.resize(ncols);
  m.col_type.resize(ncols);
  for (int64_t j = 0; j < ncols; ++j) {
    int64_t t = rng.below(100);
    double lb, ub;
    if (t < 60) {
      lb = 0.;
      ub = 1.;
    } else if (t < 75) {
      lb = -rng.below(10);
      ub = 1. + rng.below(100);
    } else if (t < 80) {
      lb = 0.;
      ub = inf;
    } else if (t < 90) {
      lb = -10. * rng.uniform();
      ub = lb + 100. * rng.uniform();
    } else if (t < 95) {
      lb = 0.;
      ub = inf;
    } else if (t < 98) {
      lb = -inf;
      ub = 50. * rng.uniform();
    } else {
      lb = -inf;
      ub = inf;
    }
    m.col_lower[j] = lb;
    m.col_upper[j] = ub;
    m.col_type[j]  = t < 80;
    m.cost[j]      = rng.below(10) < 3 ? 0. : 20. * rng.uniform() - 10.;
  }

  m.row_start.assign(nrows + 1, 0);
  m.row_lower.resize(nrows);
  m.row_upper.resize(nrows);
  std::vector<int64_t> len(nrows);
  for (int64_t r = 0; r < nrows; ++r) {
    int64_t l = 1 + rng.below(2 * nnz_per_row - 1);
    if (r % 1000 == 500) { l = 65 + rng.below(400); }
    if (r % 997 == 7) { l = 0; }
    if (r == 0) { l = ncols / 2; }
    if (r == nrows / 3) { l = ncols / 8; }
    if (r == 2 * (nrows / 3)) { l = ncols / 32; }
    len[r]             = std::min(l, ncols);
    m.row_start[r + 1] = m.row_start[r] + len[r];
  }
  m.nnz = m.row_start[nrows];
  m.col.resize(m.nnz);
  m.val.resize(m.nnz);
  for (int64_t r = 0; r < nrows; ++r) {
    int64_t first  = rng.below(ncols);
    int64_t stride = 1;
    if (rng.below(2)) {
      do {
        stride = 1 + rng.below(ncols);
      } while (std::gcd(stride, ncols) != 1);
    }
    bool exact = rng.below(2);
    for (int64_t k = 0; k < len[r]; ++k) {
      int64_t e   = m.row_start[r] + k;
      m.col[e]    = (first + k * stride) % ncols;
      double sign = rng.below(2) ? 1. : -1.;
      if (exact) {
        m.val[e] = sign * (1. + rng.below(9));
      } else if (rng.below(100) == 0) {
        m.val[e] = sign * 1e6 * (1. + rng.uniform());
      } else {
        m.val[e] = sign * (0.01 + 9.99 * rng.uniform());
      }
    }
    int64_t t  = rng.below(10);
    double rhs = exact ? rng.below(101) - 50 : 100. * rng.uniform() - 50.;
    if (t < 4) {
      m.row_lower[r] = -inf;
      m.row_upper[r] = rhs;
    } else if (t < 7) {
      m.row_lower[r] = rhs;
      m.row_upper[r] = inf;
    } else if (t < 9) {
      m.row_lower[r] = rhs;
      m.row_upper[r] = rhs;
    } else {
      m.row_lower[r] = rhs;
      m.row_upper[r] = rhs + 1. + rng.below(20);
    }
  }
}

struct stat_t {
  double med = 0., p10 = 0., p90 = 0.;
  int reps = 0;
};

struct opts_t {
  int reps        = 50;
  int warmup      = 3;
  int min_reps    = 5;
  double budget_s = 30.;
};

double now_ms()
{
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

void sync()
{
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
}

template <typename F, typename P>
stat_t measure(const opts_t& o, bool device, F&& f, P&& prepare)
{
  auto run = [&] {
    f();
    if (device) { sync(); }
  };
  for (int i = 0; i < o.warmup; ++i) {
    run();
  }
  std::vector<double> ms;
  double begin = now_ms();
  for (int i = 0; i < o.reps; ++i) {
    prepare();
    double t0 = now_ms();
    run();
    double t1 = now_ms();
    ms.push_back(t1 - t0);
    if (i + 1 >= o.min_reps && t1 - begin > o.budget_s * 1e3) { break; }
  }
  std::sort(ms.begin(), ms.end());
  size_t n = ms.size();
  auto q   = [&](double p) {
    size_t i = p * (n - 1) + 0.5;
    return ms[i];
  };
  return stat_t{0.5 * (ms[(n - 1) / 2] + ms[n / 2]), q(0.1), q(0.9), (int)n};
}

template <typename F>
stat_t measure(const opts_t& o, bool device, F&& f)
{
  return measure(o, device, f, [] {});
}

std::string fmt(const char* format, ...)
{
  char buf[1024];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);
  return buf;
}

std::string json_num(double x)
{
  if (std::isinf(x)) { return x > 0 ? "\"inf\"" : "\"-inf\""; }
  return fmt("%.17g", x);
}

template <typename T, typename F>
std::string json_array(const std::vector<T>& v, F&& item)
{
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    s += (i ? ", " : "") + item(v[i]);
  }
  return s + "]";
}

template <typename T>
T* ptr(thrust::device_vector<T>& v)
{
  return thrust::raw_pointer_cast(v.data());
}

template <typename T>
std::vector<T> to_host(const thrust::device_vector<T>& d, size_t n)
{
  std::vector<T> h(n);
  thrust::copy(d.begin(), d.begin() + n, h.begin());
  return h;
}

template <typename T>
std::vector<T> to_host(const thrust::device_vector<T>& d)
{
  return to_host(d, d.size());
}

template <typename T>
void zero(thrust::device_vector<T>& d)
{
  CUDA_CHECK(cudaMemset(ptr(d), 0, d.size() * sizeof(T)));
}

struct temp_t {
  void* p    = nullptr;
  size_t cap = 0;
};

template <typename F>
void cub_call(temp_t& t, F&& f)
{
  size_t bytes = 0;
  CUDA_CHECK(f(nullptr, bytes));
  if (!t.p || bytes > t.cap) {
    sync();
    CUDA_CHECK(cudaFree(t.p));
    t.cap = std::max<size_t>(bytes, 1);
    CUDA_CHECK(cudaMalloc(&t.p, t.cap));
  }
  CUDA_CHECK(f(t.p, bytes));
}

int64_t blocks_for(int64_t n) { return std::max<int64_t>(1, (n + block_dim - 1) / block_dim); }

light_plan_t single_class_plan(int64_t n)
{
  light_plan_t p{};
  p.n_classes     = 1;
  p.warp_begin[1] = (n + 31) / 32;
  p.row_begin[1]  = n;
  return p;
}

int64_t plan_blocks(const light_plan_t& p) { return (p.warp_begin[p.n_classes] + 7) / 8; }

int length_class(int64_t len)
{
  if (len <= 2) { return 0; }
  if (len > max_block_len) { return chunk_class; }
  if (len > max_light_len) { return block_class; }
  return 63 - __builtin_clzll(len - 1);
}

// Items in order of length class (stable); class k is order[begin[k] .. begin[k + 1]).
struct classes_t {
  std::vector<int32_t> order;
  int64_t begin[n_classes + 1] = {};
  int64_t nnz[n_classes]       = {};
};

classes_t classify(const std::vector<int64_t>& start)
{
  classes_t c;
  int64_t n = start.size() - 1;
  int64_t cursor[n_classes];
  for (int64_t i = 0; i < n; ++i) {
    int64_t len = start[i + 1] - start[i];
    ++c.begin[length_class(len) + 1];
    c.nnz[length_class(len)] += len;
  }
  for (int k = 0; k < n_classes; ++k) {
    c.begin[k + 1] += c.begin[k];
    cursor[k] = c.begin[k];
  }
  c.order.resize(n);
  for (int64_t i = 0; i < n; ++i) {
    c.order[cursor[length_class(start[i + 1] - start[i])]++] = i;
  }
  return c;
}

light_plan_t binned_plan(const classes_t& c)
{
  light_plan_t p{};
  p.n_classes = n_light_classes;
  for (int k = 0; k < n_light_classes; ++k) {
    int64_t per_warp    = 32 >> k;
    p.row_begin[k]      = c.begin[k];
    p.warp_begin[k + 1] = p.warp_begin[k] + (c.begin[k + 1] - c.begin[k] + per_warp - 1) / per_warp;
  }
  p.row_begin[n_light_classes] = c.begin[block_class];
  return p;
}

std::vector<int32_t> long_items(const classes_t& c)
{
  return std::vector<int32_t>(c.order.begin() + c.begin[chunk_class], c.order.end());
}

struct binned_t {
  classes_t classes;
  std::vector<int64_t> start;
  std::vector<int32_t> col;
  std::vector<double> val;
};

struct chunks_t {
  std::vector<int32_t> chunk_item, chunk_index, item_start;
  thrust::device_vector<int32_t> d_chunk_item, d_chunk_index, d_items, d_item_start;
  chunk_view_t view;
  void build(const std::vector<int32_t>& items, const std::vector<int64_t>& start)
  {
    for (int32_t i : items) {
      item_start.push_back(chunk_item.size());
      int64_t n = (start[i + 1] - start[i] + max_block_len - 1) / max_block_len;
      for (int64_t k = 0; k < n; ++k) {
        chunk_item.push_back(i);
        chunk_index.push_back(k);
      }
    }
    item_start.push_back(chunk_item.size());
    d_chunk_item  = chunk_item;
    d_chunk_index = chunk_index;
    d_items       = items;
    d_item_start  = item_start;
    view          = chunk_view_t{thrust::raw_pointer_cast(d_chunk_item.data()),
                        thrust::raw_pointer_cast(d_chunk_index.data()),
                        thrust::raw_pointer_cast(d_items.data()),
                        thrust::raw_pointer_cast(d_item_start.data()),
                        (int)chunk_item.size(),
                        (int)items.size()};
  }
};

binned_t make_binned(const model_t& m)
{
  binned_t b;
  b.classes = classify(m.row_start);
  b.start.resize(m.nrows + 1);
  b.col.resize(m.nnz);
  b.val.resize(m.nnz);
  int64_t p = 0;
  for (int64_t k = 0; k < m.nrows; ++k) {
    int64_t r   = b.classes.order[k];
    int64_t len = m.row_start[r + 1] - m.row_start[r];
    b.start[k]  = p;
    std::copy_n(m.col.begin() + m.row_start[r], len, b.col.begin() + p);
    std::copy_n(m.val.begin() + m.row_start[r], len, b.val.begin() + p);
    p += len;
  }
  b.start[m.nrows] = p;
  return b;
}

struct host_act_t {
  std::vector<double> min_hi, min_lo, max_hi, max_lo;
  std::vector<uint16_t> min_cnt, max_cnt;
  explicit host_act_t(int64_t n = 0)
    : min_hi(n), min_lo(n), max_hi(n), max_lo(n), min_cnt(n), max_cnt(n)
  {
  }
};

struct host_matrix_t {
  int64_t n_outer = 0, nnz = 0;
  std::vector<int64_t> start;
  std::vector<int32_t> index;
  std::vector<double> value;
};

struct compressed_t {
  int n_outer = 0, n_inner = 0;
  int64_t nnz = 0;
  thrust::device_vector<int64_t> start;
  thrust::device_vector<int32_t> index;
  thrust::device_vector<double> value;
  compressed_t(int64_t max_outer, int64_t max_nnz)
    : start(max_outer + 1), index(max_nnz), value(max_nnz)
  {
  }
  matrix_view_t view() { return matrix_view_t{ptr(start), ptr(index), ptr(value)}; }
};

struct rebuild_scratch_t {
  thrust::device_vector<uint8_t> keep;
  thrust::device_vector<int32_t> pos, new_outer, new_inner, entry_outer, position, perm,
    sorted_keys, count, u_index;
  thrust::device_vector<unsigned long long> cursor;
  thrust::device_vector<double> u_value;
  temp_t temp;
  rebuild_scratch_t(int64_t max_dim, int64_t nnz)
    : keep(nnz + 1),
      pos(nnz + 1),
      new_outer(max_dim + 1),
      new_inner(max_dim + 1),
      entry_outer(nnz),
      position(nnz),
      perm(nnz),
      sorted_keys(nnz),
      count(max_dim + 1),
      u_index(nnz),
      cursor(max_dim + 1),
      u_value(nnz)
  {
  }
};

template <typename op_t>
void walk_entries(const int64_t* start, int n_outer, int64_t nnz, op_t op)
{
  walk_entries_kernel<<<blocks_for((nnz + walk_chunk - 1) / walk_chunk), block_dim>>>(
    start, n_outer, nnz, op);
}

template <typename in_t, typename out_t, typename op_t>
void exclusive_scan(temp_t& temp, const in_t* in, out_t* out, op_t op, int64_t n)
{
  cub_call(temp, [&](void* t, size_t& bytes) {
    return cub::DeviceScan::ExclusiveScan(t, bytes, in, out, op, out_t{0}, n);
  });
}

// Alive flags hold one extra zero entry, so the scans also give the totals.
void compact(matrix_view_t in,
             int n_outer,
             int n_inner,
             int64_t nnz,
             const uint8_t* outer_alive,
             const uint8_t* inner_alive,
             rebuild_scratch_t& s,
             compressed_t& out)
{
  exclusive_scan(s.temp, outer_alive, ptr(s.new_outer), plus_i32_t{}, n_outer + 1);
  exclusive_scan(s.temp, inner_alive, ptr(s.new_inner), plus_i32_t{}, n_inner + 1);
  walk_entries(in.start, n_outer, nnz, keep_op_t{in.index, outer_alive, inner_alive, ptr(s.keep)});
  exclusive_scan(s.temp, ptr(s.keep), ptr(s.pos), plus_i32_t{}, nnz + 1);
  compact_scatter_kernel<<<blocks_for(nnz), block_dim>>>(nnz,
                                                         ptr(s.keep),
                                                         ptr(s.pos),
                                                         in.index,
                                                         in.value,
                                                         ptr(s.new_inner),
                                                         ptr(out.index),
                                                         ptr(out.value));
  compact_start_kernel<<<blocks_for(n_outer + 1), block_dim>>>(
    n_outer, in.start, outer_alive, ptr(s.new_outer), ptr(s.pos), ptr(out.start));
  int32_t totals[3];
  CUDA_CHECK(
    cudaMemcpy(&totals[0], ptr(s.new_outer) + n_outer, sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
    cudaMemcpy(&totals[1], ptr(s.new_inner) + n_inner, sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&totals[2], ptr(s.pos) + nnz, sizeof(int32_t), cudaMemcpyDeviceToHost));
  out.n_outer = totals[0];
  out.n_inner = totals[1];
  out.nnz     = totals[2];
}

void count_inner(matrix_view_t in, int n_inner, int64_t nnz, rebuild_scratch_t& s, int64_t* start)
{
  CUDA_CHECK(cudaMemsetAsync(ptr(s.count), 0, (n_inner + 1) * sizeof(int32_t)));
  histogram_kernel<<<blocks_for(nnz), block_dim>>>(nnz, in.index, ptr(s.count));
  exclusive_scan(s.temp, ptr(s.count), start, plus_i64_t{}, n_inner + 1);
}

void transpose_radix(
  matrix_view_t in, int n_outer, int n_inner, int64_t nnz, rebuild_scratch_t& s, compressed_t& out)
{
  walk_entries(in.start, n_outer, nnz, expand_op_t{ptr(s.entry_outer), ptr(s.position)});
  count_inner(in, n_inner, nnz, s, ptr(out.start));
  int end_bit = n_inner > 1 ? 32 - __builtin_clz(n_inner - 1) : 1;
  cub_call(s.temp, [&](void* t, size_t& bytes) {
    return cub::DeviceRadixSort::SortPairs(
      t, bytes, in.index, ptr(s.sorted_keys), ptr(s.position), ptr(s.perm), nnz, 0, end_bit);
  });
  gather_kernel<<<blocks_for(nnz), block_dim>>>(
    nnz, ptr(s.perm), ptr(s.entry_outer), in.value, ptr(out.index), ptr(out.value));
  out.n_outer = n_inner;
  out.n_inner = n_outer;
  out.nnz     = nnz;
}

// The scatter leaves the entries of a column in an order that depends on thread timing; the
// segmented sort by row index restores the canonical order.
void transpose_scatter(matrix_view_t in,
                       int n_outer,
                       int n_inner,
                       int64_t nnz,
                       bool ordered,
                       rebuild_scratch_t& s,
                       compressed_t& out)
{
  count_inner(in, n_inner, nnz, s, ptr(out.start));
  CUDA_CHECK(cudaMemcpyAsync(
    ptr(s.cursor), ptr(out.start), n_inner * sizeof(int64_t), cudaMemcpyDeviceToDevice));
  walk_entries(in.start,
               n_outer,
               nnz,
               scatter_op_t{in.index, in.value, ptr(s.cursor), ptr(s.u_index), ptr(s.u_value)});
  if (ordered) {
    cub_call(s.temp, [&](void* t, size_t& bytes) {
      return cub::DeviceSegmentedSort::SortPairs(t,
                                                 bytes,
                                                 ptr(s.u_index),
                                                 ptr(out.index),
                                                 ptr(s.u_value),
                                                 ptr(out.value),
                                                 nnz,
                                                 n_inner,
                                                 ptr(out.start),
                                                 ptr(out.start) + 1);
    });
  }
  out.n_outer = n_inner;
  out.n_inner = n_outer;
  out.nnz     = nnz;
}

void host_rebuild(const model_t& m,
                  const std::vector<uint8_t>& row_alive,
                  const std::vector<uint8_t>& col_alive,
                  std::vector<int32_t>& new_col,
                  std::vector<int64_t>& cursor,
                  host_matrix_t& csr,
                  host_matrix_t& csc)
{
  int32_t n_cols = 0;
  for (int64_t j = 0; j < m.ncols; ++j) {
    new_col[j] = n_cols;
    n_cols += col_alive[j];
  }
  csc.start.assign(n_cols + 1, 0);
  int64_t p = 0, n_rows = 0;
  for (int64_t r = 0; r < m.nrows; ++r) {
    if (!row_alive[r]) { continue; }
    csr.start[n_rows++] = p;
    for (int64_t e = m.row_start[r]; e < m.row_start[r + 1]; ++e) {
      if (col_alive[m.col[e]]) {
        csr.index[p] = new_col[m.col[e]];
        csr.value[p] = m.val[e];
        ++csc.start[csr.index[p] + 1];
        ++p;
      }
    }
  }
  csr.start[n_rows] = p;
  csr.n_outer       = n_rows;
  csr.nnz           = p;
  for (int32_t j = 0; j < n_cols; ++j) {
    csc.start[j + 1] += csc.start[j];
    cursor[j] = csc.start[j];
  }
  for (int64_t r = 0; r < n_rows; ++r) {
    for (int64_t e = csr.start[r]; e < csr.start[r + 1]; ++e) {
      int64_t q    = cursor[csr.index[e]]++;
      csc.index[q] = r;
      csc.value[q] = csr.value[e];
    }
  }
  csc.n_outer = n_cols;
  csc.nnz     = p;
}

bool same_matrix(compressed_t& d, const host_matrix_t& h)
{
  if (d.n_outer != h.n_outer || d.nnz != h.nnz) { return false; }
  auto start = to_host(d.start, h.n_outer + 1);
  auto index = to_host(d.index, h.nnz);
  auto value = to_host(d.value, h.nnz);
  return std::equal(start.begin(), start.end(), h.start.begin()) &&
         std::equal(index.begin(), index.end(), h.index.begin()) &&
         std::equal(value.begin(), value.end(), h.value.begin());
}

struct variant_t {
  char sched;
  int layout, count, arith;
  std::string name() const
  {
    return fmt("%c/%c/%c/%s",
               sched,
               layout == PAIRED ? 'P' : 'S',
               sched == 'T' ? '-' : (count == REDUCE ? 'R' : 'A'),
               arith == PLAIN ? "plain" : "comp");
  }
};

struct cand_variant_t {
  char sched;
  int layout, arith;
  bool gated;
  cand_fn_t launch;
  double ms = 0.;
  std::string name() const
  {
    const char* lo = layout == SPLIT && arith == COMP ? (gated ? "/gated" : "/always") : "";
    return fmt(
      "%c/%c/%s%s", sched, layout == PAIRED ? 'P' : 'S', arith == PLAIN ? "plain" : "comp", lo);
  }
};

struct result_t {
  std::string section, name;
  stat_t st;
  int64_t work;
};

struct frontier_t {
  const char* label;
  double fraction;
  int64_t nnz = 0;
  std::vector<int32_t> rows;
  std::vector<uint8_t> in_list;
  thrust::device_vector<uint8_t> flag;
};

struct bench_t {
  model_t m;
  opts_t opts;
  int64_t nrows, ncols, nnz;

  std::vector<bound_t> h_bnd, h_side;
  binned_t binned;
  host_matrix_t h_csc;
  classes_t col_classes;
  chunks_t row_chunks, col_chunks;
  host_act_t ref[2];
  std::vector<double> scale_min, scale_max;
  std::vector<double> ref_cand_lb[2], ref_cand_ub[2];
  std::vector<frontier_t> frontiers;

  thrust::device_vector<int64_t> d_row_start, d_b_start;
  thrust::device_vector<int32_t> d_col, d_b_col, d_row_of, d_col_order;
  thrust::device_vector<part_t> d_row_part;
  thrust::device_vector<bound_t> d_col_part;
  thrust::device_vector<double> d_val, d_b_val;
  thrust::device_vector<bound_t> d_bnd, d_side;
  thrust::device_vector<uint8_t> d_col_type;
  thrust::device_vector<pair_t> d_min_p, d_max_p;
  thrust::device_vector<double> d_min_hi, d_max_hi, d_min_lo, d_max_lo, d_cand_lb, d_cand_ub;
  thrust::device_vector<uint16_t> d_min_cnt, d_max_cnt;
  thrust::device_vector<int32_t> d_list_light, d_list_block, d_num_selected;
  compressed_t d_csc{0, 0};
  temp_t temp;

  matrix_view_t csr, b_csr;
  act_view_t act;
  cand_view_t cand;
  light_plan_t plan_all, plan_binned, plan_hybrid, plan_cols_hybrid;
  col_sched_t col_sched;
  std::vector<cand_variant_t> cand_variants;
  int block_begin = 0, n_block = 0;

  std::vector<result_t> results;
  std::vector<std::string> check_activity, check_candidates, check_frontier, check_other;
  bool all_ok = true;
  struct {
    int64_t count_bad = 0, failed = 0;
    double scaled = 0., rel[2] = {0., 0.};
    bool one_thread_bitwise = true;
  } act_summary;

  void setup()
  {
    nrows = m.nrows;
    ncols = m.ncols;
    nnz   = m.nnz;
    h_bnd.resize(ncols);
    h_side.resize(nrows);
    for (int64_t j = 0; j < ncols; ++j) {
      h_bnd[j] = bound_t{m.col_lower[j], m.col_upper[j]};
    }
    for (int64_t r = 0; r < nrows; ++r) {
      h_side[r] = bound_t{m.row_lower[r], m.row_upper[r]};
    }
    binned      = make_binned(m);
    block_begin = binned.classes.begin[block_class];
    n_block     = binned.classes.begin[chunk_class] - block_begin;
    std::vector<int32_t> long_rows(nrows - binned.classes.begin[chunk_class]);
    std::iota(long_rows.begin(), long_rows.end(), binned.classes.begin[chunk_class]);
    row_chunks.build(long_rows, binned.start);
    d_row_part.resize(row_chunks.chunk_item.size());

    d_row_start = m.row_start;
    d_col       = m.col;
    d_val       = m.val;
    d_b_start   = binned.start;
    d_b_col     = binned.col;
    d_b_val     = binned.val;
    d_row_of    = binned.classes.order;
    d_bnd       = h_bnd;
    d_side      = h_side;
    d_col_type  = m.col_type;
    d_min_p.resize(nrows);
    d_max_p.resize(nrows);
    d_min_hi.resize(nrows);
    d_max_hi.resize(nrows);
    d_min_lo.resize(nrows);
    d_max_lo.resize(nrows);
    d_min_cnt.resize(nrows);
    d_max_cnt.resize(nrows);
    d_cand_lb.resize(ncols);
    d_cand_ub.resize(ncols);
    d_list_light.resize(nrows);
    d_list_block.resize(n_block);
    d_num_selected.resize(2);

    csr   = matrix_view_t{ptr(d_row_start), ptr(d_col), ptr(d_val)};
    b_csr = matrix_view_t{ptr(d_b_start), ptr(d_b_col), ptr(d_b_val)};
    act   = act_view_t{ptr(d_min_p),
                     ptr(d_max_p),
                     ptr(d_min_hi),
                     ptr(d_max_hi),
                     ptr(d_min_cnt),
                     ptr(d_max_cnt),
                     ptr(d_min_lo),
                     ptr(d_max_lo)};

    plan_all    = single_class_plan(nrows);
    plan_hybrid = single_class_plan(block_begin);
    plan_binned = binned_plan(binned.classes);

    {
      rebuild_scratch_t scratch(std::max(nrows, ncols), nnz);
      d_csc = compressed_t(ncols, nnz);
      transpose_radix(csr, nrows, ncols, nnz, scratch, d_csc);
      sync();
      CUDA_CHECK(cudaFree(scratch.temp.p));
    }
    h_csc.n_outer = ncols;
    h_csc.nnz     = nnz;
    h_csc.start   = to_host(d_csc.start);
    h_csc.index   = to_host(d_csc.index);
    h_csc.value   = to_host(d_csc.value);
    col_classes   = classify(h_csc.start);
    d_col_order   = col_classes.order;
    col_chunks.build(long_items(col_classes), h_csc.start);
    d_col_part.resize(col_chunks.chunk_item.size());
    plan_cols_hybrid       = single_class_plan(col_classes.begin[block_class]);
    col_sched.plan         = binned_plan(col_classes);
    col_sched.light_blocks = plan_blocks(col_sched.plan);
    col_sched.order        = ptr(d_col_order);
    col_sched.block_begin  = col_classes.begin[block_class];
    col_sched.n_block      = col_classes.begin[chunk_class] - col_classes.begin[block_class];
    col_sched.chunks       = col_chunks.view;
    col_sched.part         = ptr(d_col_part);
    for (char sched : {'B', 'H'}) {
      cand_variants.push_back({sched, PAIRED, PLAIN, true, launch_cand<PAIRED, PLAIN, true>});
      cand_variants.push_back({sched, SPLIT, PLAIN, true, launch_cand<SPLIT, PLAIN, true>});
      cand_variants.push_back({sched, PAIRED, COMP, true, launch_cand<PAIRED, COMP, true>});
      cand_variants.push_back({sched, SPLIT, COMP, true, launch_cand<SPLIT, COMP, true>});
      cand_variants.push_back({sched, SPLIT, COMP, false, launch_cand<SPLIT, COMP, false>});
    }
    cand = cand_view_t{d_csc.view(), ptr(d_bnd), ptr(d_side), ptr(d_col_type), act};

    scale_min.assign(nrows, 0.);
    scale_max.assign(nrows, 0.);
    for (int64_t r = 0; r < nrows; ++r) {
      for (int64_t e = m.row_start[r]; e < m.row_start[r + 1]; ++e) {
        double a   = m.val[e];
        bound_t b  = h_bnd[m.col[e]];
        double tmn = std::fabs(a * (a > 0 ? b.lb : b.ub));
        double tmx = std::fabs(a * (a > 0 ? b.ub : b.lb));
        if (tmn <= big_term) { scale_min[r] += tmn; }
        if (tmx <= big_term) { scale_max[r] += tmx; }
      }
    }
  }

  cpu_view_t cpu_view(host_act_t& a, std::vector<double>& lb, std::vector<double>& ub)
  {
    return cpu_view_t{nrows,
                      ncols,
                      m.row_start.data(),
                      m.col.data(),
                      m.val.data(),
                      h_csc.start.data(),
                      h_csc.index.data(),
                      h_csc.value.data(),
                      h_bnd.data(),
                      h_side.data(),
                      m.col_type.data(),
                      a.min_hi.data(),
                      a.min_lo.data(),
                      a.max_hi.data(),
                      a.max_lo.data(),
                      a.min_cnt.data(),
                      a.max_cnt.data(),
                      lb.data(),
                      ub.data()};
  }

  void add(const std::string& section, const std::string& name, stat_t st, int64_t work)
  {
    results.push_back(result_t{section, name, st, work});
    printf("%-9s %-52s %11.4f %11.4f %11.4f %10.4f %5d\n",
           section.c_str(),
           name.c_str(),
           st.med,
           st.p10,
           st.p90,
           st.med * 1e6 / work,
           st.reps);
    fflush(stdout);
  }

  void chunked_rows(int layout, int count, int arith, const uint8_t* flag)
  {
    if (row_chunks.view.n_chunks > 0) {
      act_chunk_table[count * 2 + arith](b_csr, ptr(d_bnd), row_chunks.view, flag, ptr(d_row_part));
      act_chunk_finalize_table[layout * 2 + arith](
        ptr(d_row_part), row_chunks.view, flag, ptr(d_row_of), act);
    }
  }

  void long_rows(const variant_t& v)
  {
    if (n_block > 0) {
      act_block_table[v.layout * 4 + v.count * 2 + v.arith](
        n_block, b_csr, ptr(d_bnd), block_begin, nullptr, ptr(d_row_of), act);
    }
    chunked_rows(v.layout, v.count, v.arith, nullptr);
  }

  void run_activity(const variant_t& v)
  {
    int idx = v.layout * 4 + v.count * 2 + v.arith;
    switch (v.sched) {
      case 'T':
        act_light_table[idx](
          plan_blocks(plan_all), csr, ptr(d_bnd), plan_all, nullptr, nullptr, act);
        break;
      case 'B':
        if (block_begin > 0) {
          act_light_table[idx](
            plan_blocks(plan_binned), b_csr, ptr(d_bnd), plan_binned, nullptr, ptr(d_row_of), act);
        }
        long_rows(v);
        break;
      case 'H':
        if (block_begin > 0) {
          act_light_table[idx](
            plan_blocks(plan_hybrid), b_csr, ptr(d_bnd), plan_hybrid, nullptr, ptr(d_row_of), act);
        }
        long_rows(v);
        break;
      case 'C': act_block_table[idx](nrows, csr, ptr(d_bnd), 0, nullptr, nullptr, act); break;
    }
  }

  void run_candidates(const cand_variant_t& c)
  {
    col_sched_t sched = col_sched;
    if (c.sched == 'H') {
      sched.plan         = plan_cols_hybrid;
      sched.light_blocks = plan_blocks(plan_cols_hybrid);
    }
    c.launch(cand, sched, ptr(d_cand_lb), ptr(d_cand_ub));
  }

  void zero_activities()
  {
    zero(d_min_p);
    zero(d_max_p);
    zero(d_min_hi);
    zero(d_max_hi);
    zero(d_min_lo);
    zero(d_max_lo);
    zero(d_min_cnt);
    zero(d_max_cnt);
  }

  host_act_t split_to_host()
  {
    host_act_t h;
    h.min_hi  = to_host(d_min_hi);
    h.max_hi  = to_host(d_max_hi);
    h.min_lo  = to_host(d_min_lo);
    h.max_lo  = to_host(d_max_lo);
    h.min_cnt = to_host(d_min_cnt);
    h.max_cnt = to_host(d_max_cnt);
    return h;
  }

  void upload_reference(int arith)
  {
    const host_act_t& a = ref[arith];
    std::vector<pair_t> min_p(nrows), max_p(nrows);
    for (int64_t r = 0; r < nrows; ++r) {
      min_p[r] = pair_t{a.min_hi[r], (double)(a.min_cnt[r] & count_mask)};
      max_p[r] = pair_t{a.max_hi[r], (double)(a.max_cnt[r] & count_mask)};
    }
    d_min_p   = min_p;
    d_max_p   = max_p;
    d_min_hi  = a.min_hi;
    d_max_hi  = a.max_hi;
    d_min_lo  = a.min_lo;
    d_max_lo  = a.max_lo;
    d_min_cnt = a.min_cnt;
    d_max_cnt = a.max_cnt;
  }

  void reference()
  {
    for (int arith : {PLAIN, COMP}) {
      ref[arith] = host_act_t(nrows);
      ref_cand_lb[arith].assign(ncols, 0.);
      ref_cand_ub[arith].assign(ncols, 0.);
      cpu_view_t v = cpu_view(ref[arith], ref_cand_lb[arith], ref_cand_ub[arith]);
      cpu_base::row_activity(v, arith == COMP, nullptr, nrows);
      cpu_base::candidates_by_column(v, arith == COMP);

      host_act_t other(nrows);
      std::vector<double> lb(ncols, -inf), ub(ncols, inf);
      cpu_view_t w = cpu_view(other, lb, ub);
      cpu_fma::row_activity(w, arith == COMP, nullptr, nrows);
      bool same_act = other.min_hi == ref[arith].min_hi && other.max_hi == ref[arith].max_hi &&
                      other.min_lo == ref[arith].min_lo && other.max_lo == ref[arith].max_lo &&
                      other.min_cnt == ref[arith].min_cnt && other.max_cnt == ref[arith].max_cnt;
      cpu_view_t x = cpu_view(ref[arith], lb, ub);
      cpu_base::candidates_by_row(x, arith == COMP, nullptr, nrows);
      bool same_by_row = lb == ref_cand_lb[arith] && ub == ref_cand_ub[arith];
      cpu_fma::candidates_by_column(x, arith == COMP);
      bool same_fma = lb == ref_cand_lb[arith] && ub == ref_cand_ub[arith];
      all_ok        = all_ok && same_act && same_by_row && same_fma;
      check_other.push_back(
        fmt("{\"check\": \"cpu %s\", \"hardware_fma_build_equals_base\": %s, "
            "\"candidates_by_row_equal_by_column\": %s}",
            arith == PLAIN ? "plain" : "comp",
            same_act && same_fma ? "true" : "false",
            same_by_row ? "true" : "false"));
    }

    double max_abs = 0., max_rel = 0.;
    int64_t at = 0;
    auto side  = [&](const std::vector<double>& ph,
                    const std::vector<double>& ch,
                    const std::vector<double>& cl) {
      for (int64_t r = 0; r < nrows; ++r) {
        double c = ch[r] + cl[r];
        double d = std::fabs(c - ph[r]);
        if (d > max_abs) {
          max_abs = d;
          at      = r;
        }
        max_rel = std::max(max_rel, d / std::max(1., std::fabs(c)));
      }
    };
    side(ref[PLAIN].min_hi, ref[COMP].min_hi, ref[COMP].min_lo);
    side(ref[PLAIN].max_hi, ref[COMP].max_hi, ref[COMP].max_lo);
    int64_t cand_diff = 0;
    double cand_max   = 0.;
    for (int64_t j = 0; j < ncols; ++j) {
      for (auto* pair : {ref_cand_lb, ref_cand_ub}) {
        double p = pair[PLAIN][j], c = pair[COMP][j];
        if (p != c) {
          ++cand_diff;
          if (std::isfinite(p) && std::isfinite(c)) {
            cand_max = std::max(cand_max, std::fabs(p - c));
          }
        }
      }
    }
    check_other.push_back(
      fmt("{\"check\": \"compensated against plain\", \"activity_max_abs_diff\": %.17g, "
          "\"activity_max_rel_diff\": %.17g, \"row\": %lld, \"candidates_differing\": %lld, "
          "\"candidate_max_abs_diff\": %.17g}",
          max_abs,
          max_rel,
          (long long)at,
          (long long)cand_diff,
          cand_max));
    printf(
      "compensated against plain: largest activity difference %.6g (relative %.6g) at row "
      "%lld; %lld candidate bounds differ, by at most %.6g\n",
      max_abs,
      max_rel,
      (long long)at,
      (long long)cand_diff,
      cand_max);
  }

  void verify_activity(const variant_t& v)
  {
    zero_activities();
    run_activity(v);
    sync();
    host_act_t got = split_to_host();
    std::vector<pair_t> min_p, max_p;
    if (v.layout == PAIRED) {
      min_p = to_host(d_min_p);
      max_p = to_host(d_max_p);
    }
    const host_act_t& want = ref[v.arith];
    int64_t count_bad = 0, flag_bad = 0, bit_bad = 0;
    double max_rel = 0., max_scaled = 0.;
    auto side = [&](const std::vector<pair_t>& pair,
                    const std::vector<double>& hi,
                    const std::vector<uint16_t>& cnt,
                    const std::vector<double>& lo,
                    const std::vector<double>& want_hi,
                    const std::vector<uint16_t>& want_cnt,
                    const std::vector<double>& want_lo,
                    const std::vector<double>& scale) {
      for (int64_t r = 0; r < nrows; ++r) {
        double h  = v.layout == PAIRED ? pair[r].sum : hi[r];
        double l  = v.arith == COMP ? lo[r] : 0.;
        double wl = v.arith == COMP ? want_lo[r] : 0.;
        int c =
          v.layout == PAIRED ? std::min<double>(pair[r].count, count_mask) : cnt[r] & count_mask;
        count_bad += c != (want_cnt[r] & count_mask);
        if (v.layout == SPLIT) { flag_bad += ((cnt[r] & lo_flag) != 0) != (l != 0.); }
        bit_bad += h != want_hi[r] || l != wl;
        double w   = want_hi[r] + wl;
        double d   = std::fabs((h + l) - w);
        max_rel    = std::max(max_rel, d / std::max(1., std::fabs(w)));
        max_scaled = std::max(max_scaled, d / std::max(1., scale[r]));
      }
    };
    side(min_p,
         got.min_hi,
         got.min_cnt,
         got.min_lo,
         want.min_hi,
         want.min_cnt,
         want.min_lo,
         scale_min);
    side(max_p,
         got.max_hi,
         got.max_cnt,
         got.max_lo,
         want.max_hi,
         want.max_cnt,
         want.max_lo,
         scale_max);
    bool ok = count_bad == 0 && flag_bad == 0 && max_scaled <= agree_tol;
    all_ok  = all_ok && ok;
    check_activity.push_back(fmt(
      "{\"name\": \"%s\", \"ok\": %s, \"count_mismatches\": %lld, \"flag_mismatches\": %lld, "
      "\"max_rel_diff\": %.17g, \"max_diff_over_term_scale\": %.17g, \"bitwise_equal_cpu\": %s}",
      v.name().c_str(),
      ok ? "true" : "false",
      (long long)count_bad,
      (long long)flag_bad,
      max_rel,
      max_scaled,
      bit_bad == 0 ? "true" : "false"));
    act_summary.count_bad += count_bad;
    act_summary.failed += !ok;
    act_summary.scaled       = std::max(act_summary.scaled, max_scaled);
    act_summary.rel[v.arith] = std::max(act_summary.rel[v.arith], max_rel);
    if (v.sched == 'T') { act_summary.one_thread_bitwise &= bit_bad == 0; }
    if (!ok) {
      printf(
        "verify %s FAIL: count mismatches %lld, flag mismatches %lld, difference relative "
        "to the sum of term magnitudes %.3g\n",
        v.name().c_str(),
        (long long)count_bad,
        (long long)flag_bad,
        max_scaled);
    }
  }

  void section_activity(std::vector<variant_t>& variants, variant_t best[2], variant_t worst[2])
  {
    for (char sched : {'T', 'B', 'H', 'C'}) {
      for (int arith : {PLAIN, COMP}) {
        for (int layout : {PAIRED, SPLIT}) {
          for (int count : {REDUCE, ATOMIC}) {
            if (sched == 'T' && count == REDUCE) { continue; }
            variants.push_back(variant_t{sched, layout, count, arith});
          }
        }
      }
    }
    for (const variant_t& v : variants) {
      verify_activity(v);
    }
    printf(
      "activity check: %zu variants against the CPU reference, %lld failed; count mismatches "
      "%lld; largest difference relative to the row's sum of term magnitudes %.3g, relative "
      "to the sum itself %.3g (plain) and %.3g (compensated); one-thread-per-row variants "
      "bitwise equal to the CPU: %s\n",
      variants.size(),
      (long long)act_summary.failed,
      (long long)act_summary.count_bad,
      act_summary.scaled,
      act_summary.rel[PLAIN],
      act_summary.rel[COMP],
      act_summary.one_thread_bitwise ? "yes" : "no");
    printf("%-9s %-52s %11s %11s %11s %10s %5s\n",
           "section",
           "variant",
           "ms median",
           "ms p10",
           "ms p90",
           "ns/nnz",
           "reps");
    double best_ms[2] = {inf, inf}, worst_ms[2] = {0., 0.};
    for (const variant_t& v : variants) {
      stat_t st = measure(opts, true, [&] { run_activity(v); });
      add("activity", v.name(), st, nnz);
      if (st.med < best_ms[v.arith]) {
        best_ms[v.arith] = st.med;
        best[v.arith]    = v;
      }
      if (st.med > worst_ms[v.arith]) {
        worst_ms[v.arith] = st.med;
        worst[v.arith]    = v;
      }
    }
  }

  const cand_variant_t& matching_candidates(const variant_t& v, bool fastest)
  {
    const cand_variant_t* pick = nullptr;
    for (const cand_variant_t& c : cand_variants) {
      if (c.layout != v.layout || c.arith != v.arith || !c.gated) { continue; }
      if (!pick || (fastest ? c.ms < pick->ms : c.ms > pick->ms)) { pick = &c; }
    }
    return *pick;
  }

  void section_candidates()
  {
    for (int arith : {PLAIN, COMP}) {
      upload_reference(arith);
      for (const cand_variant_t& c : cand_variants) {
        if (c.arith != arith) { continue; }
        run_candidates(c);
        sync();
        auto lb     = to_host(d_cand_lb);
        auto ub     = to_host(d_cand_ub);
        int64_t bad = 0;
        for (int64_t j = 0; j < ncols; ++j) {
          bad += lb[j] != ref_cand_lb[arith][j] || ub[j] != ref_cand_ub[arith][j];
        }
        all_ok = all_ok && bad == 0;
        check_candidates.push_back(
          fmt("{\"name\": \"%s\", \"ok\": %s, \"columns_differing_from_cpu\": %lld}",
              c.name().c_str(),
              bad == 0 ? "true" : "false",
              (long long)bad));
        if (bad) {
          printf(
            "verify candidates %s FAIL: %lld columns differ\n", c.name().c_str(), (long long)bad);
        }
      }
      for (cand_variant_t& c : cand_variants) {
        if (c.arith != arith) { continue; }
        stat_t st = measure(opts, true, [&] { run_candidates(c); });
        c.ms      = st.med;
        add("candidate", c.name(), st, nnz);
      }
    }
  }

  void section_round(variant_t best[2], variant_t worst[2])
  {
    for (int arith : {PLAIN, COMP}) {
      for (bool is_best : {true, false}) {
        const variant_t& v      = is_best ? best[arith] : worst[arith];
        const cand_variant_t& c = matching_candidates(v, is_best);
        stat_t st               = measure(opts, true, [&] {
          run_activity(v);
          run_candidates(c);
        });
        add("round",
            fmt("%s %s: %s + %s",
                is_best ? "best" : "worst",
                arith == PLAIN ? "plain" : "comp",
                v.name().c_str(),
                c.name().c_str()),
            st,
            nnz);
      }
    }
  }

  void make_frontiers()
  {
    const char* labels[4] = {"10%", "1%", "0.1%", "0.01%"};
    double fractions[4]   = {0.1, 0.01, 0.001, 0.0001};
    frontiers.resize(4);
    std::vector<int32_t> perm(nrows);
    for (int i = 0; i < 4; ++i) {
      frontier_t& f = frontiers[i];
      f.label       = labels[i];
      f.fraction    = fractions[i];
      rng_t rng{0xf407ull + i};
      std::iota(perm.begin(), perm.end(), 0);
      int64_t n = std::max<int64_t>(1, fractions[i] * nrows + 0.5);
      std::vector<uint8_t> flag(nrows, 0);
      f.in_list.assign(nrows, 0);
      for (int64_t k = 0; k < n; ++k) {
        std::swap(perm[k], perm[k + rng.below(nrows - k)]);
        int32_t r     = binned.classes.order[perm[k]];
        flag[perm[k]] = 1;
        f.in_list[r]  = 1;
        f.rows.push_back(r);
        f.nnz += m.row_start[r + 1] - m.row_start[r];
      }
      std::sort(f.rows.begin(), f.rows.end());
      f.flag = flag;
    }
  }

  void frontier_compact(frontier_t& f, int32_t selected[2])
  {
    if (block_begin > 0) {
      cub_call(temp, [&](void* t, size_t& bytes) {
        return cub::DeviceSelect::Flagged(t,
                                          bytes,
                                          thrust::counting_iterator<int32_t>(0),
                                          ptr(f.flag),
                                          ptr(d_list_light),
                                          ptr(d_num_selected),
                                          block_begin);
      });
    }
    if (n_block > 0) {
      cub_call(temp, [&](void* t, size_t& bytes) {
        return cub::DeviceSelect::Flagged(t,
                                          bytes,
                                          thrust::counting_iterator<int32_t>(block_begin),
                                          ptr(f.flag) + block_begin,
                                          ptr(d_list_block),
                                          ptr(d_num_selected) + 1,
                                          n_block);
      });
    }
    CUDA_CHECK(
      cudaMemcpy(selected, ptr(d_num_selected), 2 * sizeof(int32_t), cudaMemcpyDeviceToHost));
  }

  void frontier_sweep(frontier_t& f, int arith)
  {
    int32_t selected[2];
    frontier_compact(f, selected);
    int idx = SPLIT * 4 + ATOMIC * 2 + arith;
    if (selected[0] > 0) {
      light_plan_t plan = single_class_plan(selected[0]);
      act_light_table[idx](
        plan_blocks(plan), b_csr, ptr(d_bnd), plan, ptr(d_list_light), ptr(d_row_of), act);
    }
    if (selected[1] > 0) {
      act_block_table[idx](
        selected[1], b_csr, ptr(d_bnd), 0, ptr(d_list_block), ptr(d_row_of), act);
    }
    chunked_rows(SPLIT, ATOMIC, arith, ptr(f.flag));
  }

  void section_frontier()
  {
    for (int arith : {PLAIN, COMP}) {
      zero_activities();
      run_activity(variant_t{'H', SPLIT, ATOMIC, arith});
      sync();
      host_act_t full = split_to_host();
      for (frontier_t& f : frontiers) {
        zero_activities();
        frontier_sweep(f, arith);
        sync();
        host_act_t got = split_to_host();
        int64_t bad    = 0;
        for (int64_t r = 0; r < nrows; ++r) {
          bool in = f.in_list[r];
          bad += got.min_hi[r] != (in ? full.min_hi[r] : 0.) ||
                 got.max_hi[r] != (in ? full.max_hi[r] : 0.) ||
                 got.min_lo[r] != (in ? full.min_lo[r] : 0.) ||
                 got.max_lo[r] != (in ? full.max_lo[r] : 0.) ||
                 got.min_cnt[r] != (in ? full.min_cnt[r] : 0) ||
                 got.max_cnt[r] != (in ? full.max_cnt[r] : 0);
        }
        all_ok = all_ok && bad == 0;
        check_frontier.push_back(fmt("{\"name\": \"%s %s\", \"ok\": %s, \"rows_differing\": %lld}",
                                     f.label,
                                     arith == PLAIN ? "plain" : "comp",
                                     bad == 0 ? "true" : "false",
                                     (long long)bad));
        if (bad) { printf("verify frontier %s FAIL: %lld rows differ\n", f.label, (long long)bad); }
      }
    }
    for (frontier_t& f : frontiers) {
      int32_t selected[2];
      add("frontier",
          fmt("%s (%lld rows): compaction only", f.label, (long long)f.rows.size()),
          measure(opts, true, [&] { frontier_compact(f, selected); }),
          f.nnz);
      for (int arith : {PLAIN, COMP}) {
        add("frontier",
            fmt("%s (%lld rows): compaction + H/S/A/%s",
                f.label,
                (long long)f.rows.size(),
                arith == PLAIN ? "plain" : "comp"),
            measure(opts, true, [&] { frontier_sweep(f, arith); }),
            f.nnz);
      }
    }
  }

  void section_cpu()
  {
    host_act_t out(nrows);
    std::vector<double> lb(ncols, -inf), ub(ncols, inf);
    struct arith_t {
      const char* name;
      int arith;
      bool hardware_fma;
    };
    const arith_t ariths[3] = {
      {"plain", PLAIN, false}, {"comp", COMP, false}, {"comp-fma", COMP, true}};
    // A frontier list is timed twice: repeated as it is, which keeps a short list cache-resident
    // (warm), and with the core's private caches evicted before every repetition (cold).
    std::vector<uint8_t> evict_buffer(32 << 20, 1);
    volatile uint64_t evict_sink = 0;
    auto evict                   = [&] {
      uint64_t sum = 0;
      for (size_t i = 0; i < evict_buffer.size(); i += 64) {
        sum += evict_buffer[i];
      }
      evict_sink = sum;
    };
    auto run = [&](const char* what,
                   const std::string& scope,
                   const int32_t* rows,
                   int64_t n,
                   int64_t work,
                   bool cold) {
      auto prepare = [&] {
        if (cold) { evict(); }
      };
      for (const arith_t& a : ariths) {
        bool comp     = a.arith == COMP;
        cpu_view_t w  = cpu_view(out, lb, ub);
        cpu_view_t r  = cpu_view(ref[a.arith], lb, ub);
        auto activity = a.hardware_fma ? cpu_fma::row_activity : cpu_base::row_activity;
        auto by_row   = a.hardware_fma ? cpu_fma::candidates_by_row : cpu_base::candidates_by_row;
        auto by_column =
          a.hardware_fma ? cpu_fma::candidates_by_column : cpu_base::candidates_by_column;
        stat_t st;
        if (!strcmp(what, "activity")) {
          st = measure(opts, false, [&] { activity(w, comp, rows, n); }, prepare);
        } else if (!strcmp(what, "candidates by row")) {
          st = measure(opts, false, [&] { by_row(r, comp, rows, n); }, prepare);
        } else {
          st = measure(opts, false, [&] { by_column(r, comp); });
        }
        add("cpu", fmt("%s, %s, %s", what, scope.c_str(), a.name), st, work);
      }
    };
    run("activity", "full", nullptr, nrows, nnz, false);
    run("candidates by row", "full", nullptr, nrows, nnz, false);
    run("candidates by column", "full", nullptr, nrows, nnz, false);
    for (frontier_t& f : frontiers) {
      for (bool cold : {false, true}) {
        std::string scope =
          fmt("%s (%lld rows) %s", f.label, (long long)f.rows.size(), cold ? "cold" : "warm");
        run("activity", scope, f.rows.data(), f.rows.size(), f.nnz, cold);
        run("candidates by row", scope, f.rows.data(), f.rows.size(), f.nnz, cold);
      }
    }
  }

  void section_rebuild()
  {
    rng_t rng{0xdeadull};
    std::vector<uint8_t> h_row_alive(nrows + 1, 0), h_col_alive(ncols + 1, 0);
    for (int64_t r = 0; r < nrows; ++r) {
      h_row_alive[r] = rng.uniform() >= 0.2;
    }
    for (int64_t j = 0; j < ncols; ++j) {
      h_col_alive[j] = rng.uniform() >= 0.2;
    }
    host_matrix_t h_csr, h_new_csc;
    h_csr.start.resize(nrows + 1);
    h_csr.index.resize(nnz);
    h_csr.value.resize(nnz);
    h_new_csc.index.resize(nnz);
    h_new_csc.value.resize(nnz);
    std::vector<int32_t> new_col(ncols);
    std::vector<int64_t> cursor(ncols);
    auto on_host = [&] {
      host_rebuild(m, h_row_alive, h_col_alive, new_col, cursor, h_csr, h_new_csc);
    };
    on_host();

    thrust::device_vector<uint8_t> row_alive = h_row_alive, col_alive = h_col_alive;
    rebuild_scratch_t s(std::max(nrows, ncols), nnz);
    compressed_t new_csr(nrows, nnz), new_csc(ncols, nnz);
    auto compact_csr = [&] {
      compact(csr, nrows, ncols, nnz, ptr(row_alive), ptr(col_alive), s, new_csr);
    };
    auto radix = [&] {
      transpose_radix(new_csr.view(), new_csr.n_outer, new_csr.n_inner, new_csr.nnz, s, new_csc);
    };
    auto scatter = [&](bool ordered) {
      transpose_scatter(
        new_csr.view(), new_csr.n_outer, new_csr.n_inner, new_csr.nnz, ordered, s, new_csc);
    };
    auto compact_csc = [&] {
      compact(d_csc.view(), ncols, nrows, nnz, ptr(col_alive), ptr(row_alive), s, new_csc);
    };

    compact_csr();
    sync();
    bool ok_csr = same_matrix(new_csr, h_csr);
    radix();
    sync();
    bool ok_radix = same_matrix(new_csc, h_new_csc);
    zero(new_csc.index);
    scatter(true);
    sync();
    bool ok_scatter = same_matrix(new_csc, h_new_csc);
    zero(new_csc.index);
    compact_csc();
    sync();
    bool ok_compact = same_matrix(new_csc, h_new_csc);
    all_ok          = all_ok && ok_csr && ok_radix && ok_scatter && ok_compact;
    check_other.push_back(
      fmt("{\"check\": \"rebuild against host\", \"rows\": %lld, \"cols\": %lld, \"nnz\": %lld, "
          "\"compact_csr\": %s, \"csc_radix_sort\": %s, \"csc_histogram_scatter_ordered\": %s, "
          "\"csc_compaction\": %s}",
          (long long)h_csr.n_outer,
          (long long)h_new_csc.n_outer,
          (long long)h_csr.nnz,
          ok_csr ? "true" : "false",
          ok_radix ? "true" : "false",
          ok_scatter ? "true" : "false",
          ok_compact ? "true" : "false"));
    if (!(ok_csr && ok_radix && ok_scatter && ok_compact)) {
      printf("verify rebuild FAIL: csr %d radix %d scatter %d compaction %d\n",
             ok_csr,
             ok_radix,
             ok_scatter,
             ok_compact);
    }

    add("rebuild", "compact CSR", measure(opts, true, compact_csr), nnz);
    add("rebuild", "CSC: stable radix sort", measure(opts, true, radix), nnz);
    add("rebuild",
        "CSC: histogram + scan + scatter, unordered",
        measure(opts, true, [&] { scatter(false); }),
        nnz);
    add("rebuild",
        "CSC: histogram + scan + scatter + segmented sort",
        measure(opts, true, [&] { scatter(true); }),
        nnz);
    add("rebuild", "CSC: compaction of the old CSC", measure(opts, true, compact_csc), nnz);
    add("rebuild",
        "total: compact CSR + radix sort",
        measure(opts,
                true,
                [&] {
                  compact_csr();
                  radix();
                }),
        nnz);
    add("rebuild",
        "total: compact CSR + scatter + segmented sort",
        measure(opts,
                true,
                [&] {
                  compact_csr();
                  scatter(true);
                }),
        nnz);
    add("rebuild",
        "total: compact CSR + compact CSC",
        measure(opts,
                true,
                [&] {
                  compact_csr();
                  compact_csc();
                }),
        nnz);
    add("rebuild", "cpu: compact CSR + counting transpose", measure(opts, false, on_host), nnz);
    sync();
    CUDA_CHECK(cudaFree(s.temp.p));
  }

  std::string matrix_json()
  {
    int64_t max_row = 0, max_col = 0;
    for (int64_t r = 0; r < nrows; ++r) {
      max_row = std::max(max_row, m.row_start[r + 1] - m.row_start[r]);
    }
    for (int64_t j = 0; j < ncols; ++j) {
      max_col = std::max(max_col, h_csc.start[j + 1] - h_csc.start[j]);
    }
    int64_t inf_lb = 0, inf_ub = 0, integers = 0, finite_lower = 0, finite_upper = 0;
    for (int64_t j = 0; j < ncols; ++j) {
      inf_lb += m.col_lower[j] == -inf;
      inf_ub += m.col_upper[j] == inf;
      integers += m.col_type[j] != 0;
    }
    int64_t by_count[2][3] = {}, lo_nonzero[2] = {};
    for (int64_t r = 0; r < nrows; ++r) {
      finite_lower += m.row_lower[r] > -inf;
      finite_upper += m.row_upper[r] < inf;
      ++by_count[0][std::min(ref[COMP].min_cnt[r] & count_mask, 2)];
      ++by_count[1][std::min(ref[COMP].max_cnt[r] & count_mask, 2)];
      lo_nonzero[0] += ref[COMP].min_lo[r] != 0.;
      lo_nonzero[1] += ref[COMP].max_lo[r] != 0.;
    }
    std::string s = fmt(
      "{\"rows\": %lld, \"cols\": %lld, \"nnz\": %lld, \"max_row_len\": %lld, "
      "\"max_col_len\": %lld, ",
      (long long)nrows,
      (long long)ncols,
      (long long)nnz,
      (long long)max_row,
      (long long)max_col);
    const char* names[n_classes] = {
      "0-2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-8192", ">8192"};
    auto classes = [&](const char* key, const char* unit, const classes_t& c) {
      s += fmt("\"%s\": [", key);
      for (int k = 0; k < n_classes; ++k) {
        s += fmt("%s{\"len\": \"%s\", \"%s\": %lld, \"nnz\": %lld}",
                 k ? ", " : "",
                 names[k],
                 unit,
                 (long long)(c.begin[k + 1] - c.begin[k]),
                 (long long)c.nnz[k]);
      }
      s += "], ";
    };
    classes("row_classes", "rows", binned.classes);
    classes("col_classes", "cols", col_classes);
    s += fmt(
      "\"integer_cols\": %lld, \"cols_infinite_lower\": %lld, \"cols_infinite_upper\": %lld, "
      "\"rows_finite_lower\": %lld, \"rows_finite_upper\": %lld, ",
      (long long)integers,
      (long long)inf_lb,
      (long long)inf_ub,
      (long long)finite_lower,
      (long long)finite_upper);
    s += fmt(
      "\"rows_by_infinite_terms_min\": [%lld, %lld, %lld], "
      "\"rows_by_infinite_terms_max\": [%lld, %lld, %lld], "
      "\"rows_nonzero_lo_min\": %lld, \"rows_nonzero_lo_max\": %lld}",
      (long long)by_count[0][0],
      (long long)by_count[0][1],
      (long long)by_count[0][2],
      (long long)by_count[1][0],
      (long long)by_count[1][1],
      (long long)by_count[1][2],
      (long long)lo_nonzero[0],
      (long long)lo_nonzero[1]);
    return s;
  }

  std::string values_json()
  {
    const host_act_t& a = ref[COMP];
    std::vector<double> mn(nrows), mx(nrows);
    std::vector<int> cmn(nrows), cmx(nrows);
    for (int64_t r = 0; r < nrows; ++r) {
      mn[r]  = a.min_hi[r] + a.min_lo[r];
      mx[r]  = a.max_hi[r] + a.max_lo[r];
      cmn[r] = a.min_cnt[r] & count_mask;
      cmx[r] = a.max_cnt[r] & count_mask;
    }
    auto count = [](int c) { return fmt("%d", c); };
    return "{\"min_activity\": " + json_array(mn, json_num) +
           ",\n    \"min_infinite_terms\": " + json_array(cmn, count) +
           ",\n    \"max_activity\": " + json_array(mx, json_num) +
           ",\n    \"max_infinite_terms\": " + json_array(cmx, count) +
           ",\n    \"min_activity_plain\": " + json_array(ref[PLAIN].min_hi, json_num) +
           ",\n    \"max_activity_plain\": " + json_array(ref[PLAIN].max_hi, json_num) +
           ",\n    \"candidate_lower_plain\": " + json_array(ref_cand_lb[PLAIN], json_num) +
           ",\n    \"candidate_upper_plain\": " + json_array(ref_cand_ub[PLAIN], json_num) +
           ",\n    \"candidate_lower_comp\": " + json_array(ref_cand_lb[COMP], json_num) +
           ",\n    \"candidate_upper_comp\": " + json_array(ref_cand_ub[COMP], json_num) + "}";
  }

  void write_json(const char* path, const std::string& input, bool quick)
  {
    int device = 0, runtime = 0, driver = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    std::string escaped;
    for (char ch : input) {
      if (ch == '"' || ch == '\\') { escaped += '\\'; }
      escaped += ch;
    }
    FILE* f = fopen(path, "w");
    CHECK(f);
    fprintf(f,
            "{\n  \"tool\": \"sweeps\",\n  \"input\": \"%s\",\n  \"quick\": %s,\n",
            escaped.c_str(),
            quick ? "true" : "false");
    fprintf(f,
            "  \"repetitions\": %d,\n  \"warmup\": %d,\n  \"min_repetitions\": %d,\n  "
            "\"budget_seconds\": %g,\n",
            opts.reps,
            opts.warmup,
            opts.min_reps,
            opts.budget_s);
    fprintf(f,
            "  \"gpu\": {\"name\": \"%s\", \"compute_capability\": \"%d.%d\", \"memory_bytes\": "
            "%llu, \"cuda_runtime\": \"%d.%d\", \"cuda_driver\": \"%d.%d\", \"cuda_compiler\": "
            "\"%d.%d.%d\", \"cccl\": \"%d.%d.%d\"},\n",
            prop.name,
            prop.major,
            prop.minor,
            (unsigned long long)prop.totalGlobalMem,
            runtime / 1000,
            runtime % 1000 / 10,
            driver / 1000,
            driver % 1000 / 10,
            __CUDACC_VER_MAJOR__,
            __CUDACC_VER_MINOR__,
            __CUDACC_VER_BUILD__,
            CUB_MAJOR_VERSION,
            CUB_MINOR_VERSION,
            CUB_SUBMINOR_VERSION);
    fprintf(f, "  \"matrix\": %s,\n", matrix_json().c_str());
    fprintf(f, "  \"results\": {");
    const char* sections[6] = {"activity", "candidate", "round", "frontier", "cpu", "rebuild"};
    for (int i = 0; i < 6; ++i) {
      fprintf(f, "%s\n    \"%s\": [", i ? "," : "", sections[i]);
      bool first = true;
      for (const result_t& r : results) {
        if (r.section != sections[i]) { continue; }
        auto per_nnz = [&](double ms) {
          return r.work > 0 ? fmt("%.6g", ms * 1e6 / r.work) : std::string("null");
        };
        fprintf(f,
                "%s\n      {\"name\": \"%s\", \"ms_median\": %.6g, \"ms_p10\": %.6g, \"ms_p90\": "
                "%.6g, \"ns_per_nnz_median\": %s, \"ns_per_nnz_p10\": %s, \"ns_per_nnz_p90\": "
                "%s, \"nnz\": %lld, \"repetitions\": %d}",
                first ? "" : ",",
                r.name.c_str(),
                r.st.med,
                r.st.p10,
                r.st.p90,
                per_nnz(r.st.med).c_str(),
                per_nnz(r.st.p10).c_str(),
                per_nnz(r.st.p90).c_str(),
                (long long)r.work,
                r.st.reps);
        first = false;
      }
      fprintf(f, "\n    ]");
    }
    fprintf(f, "\n  },\n  \"verification\": {\n    \"ok\": %s", all_ok ? "true" : "false");
    auto list = [&](const char* name, const std::vector<std::string>& items) {
      fprintf(f, ",\n    \"%s\": [", name);
      for (size_t i = 0; i < items.size(); ++i) {
        fprintf(f, "%s\n      %s", i ? "," : "", items[i].c_str());
      }
      fprintf(f, "\n    ]");
    };
    list("activity", check_activity);
    list("candidates", check_candidates);
    list("frontier", check_frontier);
    list("other", check_other);
    fprintf(f, "\n  }");
    if (nrows <= 1024 && ncols <= 1024) {
      fprintf(f, ",\n  \"values\": %s", values_json().c_str());
    }
    fprintf(f, "\n}\n");
    fclose(f);
  }
};

}  // namespace

int main(int argc, char** argv)
{
  std::vector<std::string> args;
  bool quick = false;
  opts_t opts;
  int reps        = 0;
  double budget_s = 0.;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--quick") {
      quick = true;
    } else if (a == "--reps" && i + 1 < argc) {
      reps = atoi(argv[++i]);
    } else if (a == "--budget" && i + 1 < argc) {
      budget_s = atof(argv[++i]);
    } else {
      args.push_back(a);
    }
  }
  bool synthetic = args.size() == 5 && args[0] == "--synthetic";
  if (!synthetic && args.size() != 2) {
    fprintf(stderr,
            "usage: sweeps <matrix.csr | --synthetic rows cols nnz_per_row> <out.json> [--quick] "
            "[--reps n] [--budget seconds]\n");
    return 2;
  }
  if (quick) { opts = opts_t{5, 1, 3, 2.}; }
  if (reps > 0) { opts.reps = reps; }
  if (budget_s > 0.) { opts.budget_s = budget_s; }

  bench_t b;
  b.opts = opts;
  std::string input;
  if (synthetic) {
    make_synthetic(atoll(args[1].c_str()), atoll(args[2].c_str()), atoll(args[3].c_str()), b.m);
    input = "synthetic " + args[1] + " " + args[2] + " " + args[3];
  } else {
    if (!load_dump(args[0].c_str(), b.m)) {
      fprintf(stderr, "sweeps: cannot open %s\n", args[0].c_str());
      return 2;
    }
    input = args[0];
  }
  validate(b.m);
  const std::string& out_path = args.back();

  b.setup();
  printf("sweeps: %s: %lld rows, %lld cols, %lld nonzeros\n",
         input.c_str(),
         (long long)b.nrows,
         (long long)b.ncols,
         (long long)b.nnz);
  b.reference();

  std::vector<variant_t> variants;
  variant_t best[2], worst[2];
  b.section_activity(variants, best, worst);
  b.section_candidates();
  b.section_round(best, worst);
  b.make_frontiers();
  b.section_frontier();
  b.section_cpu();
  b.section_rebuild();
  b.write_json(out_path.c_str(), input, quick);
  printf("verification: %s\nwrote %s\n", b.all_ok ? "ok" : "FAILED", out_path.c_str());
  return b.all_ok ? 0 : 1;
}
