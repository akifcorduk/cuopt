/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pdfp_projection.cuh"

#include <cuopt/error.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <pdlp/cusparse_view.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/device_scalar_init.hpp>
#include <utilities/manual_cuda_graph.cuh>
#include <utilities/timer.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {
namespace {

template <typename f_t>
__device__ f_t clip(f_t x, f_t l, f_t u)
{
  return min(max(x, l), u);
}

// One warp per sparse row, for both CSR A and the model's CSR transpose.
template <bool absolute, typename i_t, typename f_t>
__global__ void scaled_spmv(i_t rows,
                            const i_t* offsets,
                            const i_t* indices,
                            const f_t* values,
                            const f_t* row_scale,
                            const f_t* col_scale,
                            const f_t* x,
                            f_t* output)
{
  i_t row  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  i_t lane = threadIdx.x % 32;
  if (row >= rows) return;
  f_t sum = 0;
  for (i_t k = offsets[row] + lane; k < offsets[row + 1]; k += 32) {
    f_t a = values[k] * col_scale[indices[k]];
    sum += (absolute ? abs(a) : a) * x[indices[k]];
  }
  sum = raft::warpReduce(sum);
  if (lane == 0) output[row] = row_scale[row] * sum;
}

// One thread per nonzero; the row comes from a binary search over the CSR offsets.
template <typename i_t, typename f_t>
__global__ void scale_csr_values(i_t rows,
                                 i_t nnz,
                                 const i_t* offsets,
                                 const i_t* indices,
                                 const f_t* values,
                                 const f_t* row_scale,
                                 const f_t* col_scale,
                                 f_t* scaled)
{
  i_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nnz) return;
  i_t lo = 0, hi = rows;
  while (hi - lo > 1) {
    i_t mid = lo + (hi - lo) / 2;
    if (offsets[mid] <= k)
      lo = mid;
    else
      hi = mid;
  }
  scaled[k] = values[k] * row_scale[lo] * col_scale[indices[k]];
}

template <typename i_t, typename f_t>
__global__ void initialize_projection(typename problem_t<i_t, f_t>::view_t pb,
                                      const f_t* scale,
                                      const f_t* input,
                                      const f_t* start,
                                      f_t* target,
                                      f_t* current,
                                      f_t distance_weight,
                                      bool cold,
                                      bool start_at_target)
{
  i_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= pb.n_variables) return;
  target[j] = input[j] / scale[j];
  if (cold) current[j] = start != nullptr ? start[j] / scale[j] : target[j];
  if (start_at_target) current[j] = target[j];
  current[j] = clip(current[j],
                    get_lower(pb.variable_bounds[j]) / scale[j],
                    get_upper(pb.variable_bounds[j]) / scale[j]);
  f_t w      = pb.variable_types.size() && pb.is_integer_var(j) ? distance_weight * scale[j] : 0.;
  current[pb.n_variables + j] = cold ? 0. : clip(current[pb.n_variables + j], -w, w);
}

// Working set at a point. A variable is held when it sits at a bound or at the target and its
// reduced cost lies strictly inside the subgradients there; a row is slack when its dual is zero
// and it is strictly satisfied. Held variables and slack rows get the weight `held`, the rest 1,
// both times the square root of their Pock-Chambolle l1 sum.
template <typename i_t, typename f_t>
__global__ void classify_working_set(typename problem_t<i_t, f_t>::view_t pb,
                                     const f_t* col_scale,
                                     const f_t* row_scale,
                                     const f_t* col_root,
                                     const f_t* row_root,
                                     const f_t* target,
                                     const f_t* point,
                                     const f_t* atz_point,
                                     const f_t* ax,
                                     f_t distance_weight,
                                     f_t objective_weight,
                                     f_t held,
                                     f_t* col_weight,
                                     f_t* row_weight)
{
  i_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < pb.n_variables) {
    f_t w =
      pb.variable_types.size() && pb.is_integer_var(j) ? distance_weight * col_scale[j] : f_t(0);
    f_t g = objective_weight * pb.objective_coefficients[j] * col_scale[j] + atz_point[j];
    f_t x = point[j];
    // Subgradients of w * |x - target| plus the box at x.
    f_t lo = x > target[j] ? w : -w;
    f_t hi = x < target[j] ? -w : w;
    if (x <= get_lower(pb.variable_bounds[j]) / col_scale[j]) lo = -INFINITY;
    if (x >= get_upper(pb.variable_bounds[j]) / col_scale[j]) hi = INFINITY;
    col_weight[j] = col_root[j] * (lo < -g && -g < hi ? held : f_t(1));
  }
  if (j < pb.n_constraints) {
    f_t l         = pb.constraint_lower_bounds[j] * row_scale[j];
    f_t u         = pb.constraint_upper_bounds[j] * row_scale[j];
    bool slack    = point[2 * pb.n_variables + j] == 0 && l < ax[j] && ax[j] < u;
    row_weight[j] = row_root[j] * (slack ? held : f_t(1));
  }
}

// Steps from the weights and the weighted absolute row and column sums left in the step arrays.
template <typename i_t, typename f_t>
__global__ void finish_steps(i_t n_variables,
                             i_t n_constraints,
                             const f_t* col_weight,
                             const f_t* row_weight,
                             f_t* col_step,
                             f_t* row_step)
{
  i_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n_variables) col_step[j] = col_step[j] > 0 ? col_weight[j] / col_step[j] : f_t(1);
  if (j < n_constraints) row_step[j] = row_step[j] > 0 ? row_weight[j] / row_step[j] : f_t(1);
}

// Step sizes and weights live on the device so a captured iteration graph stays valid across
// restarts and projections.
constexpr int tau_slot       = 0;
constexpr int sigma_slot     = 1;
constexpr int distance_slot  = 2;
constexpr int objective_slot = 3;
constexpr int n_param_slots  = 4;

template <typename i_t, typename f_t>
__global__ void set_step_state(f_t* params,
                               i_t* counters,
                               f_t tau,
                               f_t sigma,
                               f_t distance_weight,
                               f_t objective_weight,
                               i_t inner)
{
  params[tau_slot]       = tau;
  params[sigma_slot]     = sigma;
  params[distance_slot]  = distance_weight;
  params[objective_slot] = objective_weight;
  counters[0]            = inner;
}

// With halpern, the Halpern update is fused into the primal and dual steps. The inner iteration
// count ping-pongs between counters[0] (read by the primal step, written by the dual step) and
// counters[1] (the reverse), so it advances without another kernel.
template <bool halpern, typename i_t, typename f_t>
__global__ void primal_step(typename problem_t<i_t, f_t>::view_t pb,
                            const f_t* scale,
                            f_t* current,
                            const f_t* anchor,
                            const f_t* atz,
                            const f_t* target,
                            f_t* point,
                            f_t* reflected_x,
                            f_t* slack,
                            const f_t* params,
                            const f_t* col_step,
                            i_t* counters,
                            bool prox)
{
  i_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (halpern && j == 0) counters[1] = counters[0] + 1;
  if (j >= pb.n_variables) return;
  f_t tau = params[tau_slot] * col_step[j];
  f_t v   = current[j] - tau * (params[objective_slot] * pb.objective_coefficients[j] * scale[j] +
                              atz[j] - current[pb.n_variables + j]);
  f_t x   = v;
  if (prox && pb.variable_types.size() && pb.is_integer_var(j)) {
    // Proximal step of tau * w * |x - target|; clipping to the box afterwards is exact in 1-D.
    f_t shrink = tau * params[distance_slot] * scale[j];
    f_t d      = v - target[j];
    x          = target[j] + (d > shrink ? d - shrink : d < -shrink ? d + shrink : f_t(0));
  }
  x = clip(
    x, get_lower(pb.variable_bounds[j]) / scale[j], get_upper(pb.variable_bounds[j]) / scale[j]);
  point[j]       = x;
  reflected_x[j] = 2 * x - current[j];
  slack[j]       = (x - v) / tau;
  if (halpern) {
    f_t lambda = f_t(1) / (counters[0] + 2);
    current[j] = lambda * anchor[j] + (1 - lambda) * (2 * x - current[j]);
  }
}

template <bool halpern, typename i_t, typename f_t>
__global__ void dual_step(typename problem_t<i_t, f_t>::view_t pb,
                          const f_t* col_scale,
                          const f_t* row_scale,
                          const f_t* target,
                          f_t* current,
                          const f_t* anchor,
                          const f_t* reflected_x,
                          const f_t* ax,
                          f_t* point,
                          const f_t* params,
                          const f_t* row_step,
                          i_t* counters,
                          bool prox)
{
  i_t j      = blockIdx.x * blockDim.x + threadIdx.x;
  f_t sigma  = params[sigma_slot];
  f_t lambda = halpern ? f_t(1) / (counters[1] + 1) : f_t(0);
  if (halpern && j == 0) counters[0] = counters[1];
  if (j < pb.n_variables && !prox) {
    f_t w = pb.variable_types.size() && pb.is_integer_var(j) ? params[distance_slot] * col_scale[j]
                                                             : f_t(0);
    f_t y = clip(current[pb.n_variables + j] + sigma * (target[j] - reflected_x[j]), -w, w);
    point[pb.n_variables + j] = y;
    if (halpern) {
      current[pb.n_variables + j] =
        lambda * anchor[pb.n_variables + j] + (1 - lambda) * (2 * y - current[pb.n_variables + j]);
    }
  }
  if (j < pb.n_constraints) {
    i_t k    = 2 * pb.n_variables + j;
    f_t s    = sigma * row_step[j];
    f_t v    = current[k] + s * ax[j];
    f_t l    = pb.constraint_lower_bounds[j] * row_scale[j];
    f_t u    = pb.constraint_upper_bounds[j] * row_scale[j];
    f_t q    = v / s;
    f_t z    = q < l ? v - s * l : q > u ? v - s * u : f_t(0);
    point[k] = z;
    if (halpern) { current[k] = lambda * anchor[k] + (1 - lambda) * (2 * z - current[k]); }
  }
}

template <typename f_t>
__global__ void halpern_step(
  size_t size, f_t* current, const f_t* point, const f_t* anchor, f_t lambda)
{
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < size) current[j] = lambda * anchor[j] + (1 - lambda) * (2 * point[j] - current[j]);
}

template <typename i_t, typename f_t>
__global__ void unscale_output(i_t n, const f_t* point, const f_t* scale, f_t* output)
{
  i_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) output[j] = point[j] * scale[j];
}

struct metrics_t {
  double primal = 0., dual = 0., gap = 0., objective = 0., target = 0., cost = 0.;
  double dx = 0., dy = 0., cross = 0., anchor_x = 0., anchor_dual = 0.;
  double primal_sq = 0., primal_sq_scaled = 0., rhs_sq = 0., rhs_sq_scaled = 0.;
  double dual_objective = 0., dual_illegal = 0.;
};

struct add_metrics {
  __host__ __device__ metrics_t operator()(metrics_t a, metrics_t b) const
  {
    a.primal = a.primal > b.primal ? a.primal : b.primal;
    a.dual += b.dual;
    a.gap += b.gap;
    a.objective += b.objective;
    a.target += b.target;
    a.cost += b.cost;
    a.dx += b.dx;
    a.dy += b.dy;
    a.cross += b.cross;
    a.anchor_x += b.anchor_x;
    a.anchor_dual += b.anchor_dual;
    a.primal_sq += b.primal_sq;
    a.primal_sq_scaled += b.primal_sq_scaled;
    a.rhs_sq += b.rhs_sq;
    a.rhs_sq_scaled += b.rhs_sq_scaled;
    a.dual_objective += b.dual_objective;
    a.dual_illegal += b.dual_illegal;
    return a;
  }
};

template <typename i_t, typename f_t>
struct collect_metrics {
  typename problem_t<i_t, f_t>::view_t pb;
  const f_t *col_scale, *row_scale, *target, *current, *point, *anchor;
  const f_t *ax, *atz, *atz_point, *slack;
  const f_t *col_step, *row_step;
  f_t distance_weight, objective_weight, bound_ratio;
  bool prox;

  __device__ metrics_t operator()(i_t j) const
  {
    metrics_t r;
    if (j < pb.n_variables) {
      f_t w =
        pb.variable_types.size() && pb.is_integer_var(j) ? distance_weight * col_scale[j] : 0.;
      f_t c = objective_weight * pb.objective_coefficients[j] * col_scale[j];
      f_t x = point[j], y = point[pb.n_variables + j];
      f_t dist     = w * abs(x - target[j]);
      f_t residual = c + atz_point[j] - y - slack[j];
      r.dual       = residual * residual;
      r.cost       = c * c + w * w;
      // With prox the step's own subgradient of the distance term is exact at x.
      r.gap       = prox ? f_t(0) : y * (target[j] - x) - dist;
      r.objective = c * x + dist;
      r.target    = w * abs(target[j]);
      f_t dx = current[j] - x, dy = current[pb.n_variables + j] - y;
      r.dx    = dx * dx / col_step[j];
      r.dy    = dy * dy;
      r.cross = dx * (atz[j] - atz_point[j] - dy);
      f_t da = anchor[j] - x, db = anchor[pb.n_variables + j] - y;
      r.anchor_x    = da * da / col_step[j];
      r.anchor_dual = db * db;
      // Dual bound of the row duals alone: min over the box of g * x + w * |x - target|, attained
      // at a bound or at the target. A slope that an infinite bound leaves unbounded is the
      // illegal part of the reduced cost, as in PDLP.
      f_t lower = get_lower(pb.variable_bounds[j]) / col_scale[j];
      f_t upper = get_upper(pb.variable_bounds[j]) / col_scale[j];
      f_t g     = c + atz_point[j];
      if (!isfinite(upper) && g < -w) {
        r.dual_illegal += (g + w) * (g + w);
        g = -w;
      }
      if (!isfinite(lower) && g > w) {
        r.dual_illegal += (g - w) * (g - w);
        g = w;
      }
      f_t best = g * target[j];
      if (isfinite(upper)) best = min(best, g * upper + w * (upper - target[j]));
      if (isfinite(lower)) best = min(best, g * lower + w * (target[j] - lower));
      r.dual_objective = best;
    }
    if (j < pb.n_constraints) {
      f_t activity = ax[j] / row_scale[j];
      f_t l = pb.constraint_lower_bounds[j], u = pb.constraint_upper_bounds[j];
      f_t rhs            = max(isfinite(l) ? abs(l) : f_t(0), isfinite(u) ? abs(u) : f_t(0));
      r.primal           = isfinite(activity) ? abs(activity - clip(activity, l, u)) : INFINITY;
      r.primal_sq        = r.primal * r.primal;
      r.primal_sq_scaled = r.primal_sq * row_scale[j] * row_scale[j];
      r.rhs_sq           = rhs * rhs;
      r.rhs_sq_scaled    = r.rhs_sq * row_scale[j] * row_scale[j];
      // Violation in units of the row's own feasibility tolerance, abs + rel * |bound|.
      r.primal /= 1 + bound_ratio * rhs;
      f_t z = point[2 * pb.n_variables + j];
      // The zero case avoids multiplying an infinite row bound by zero.
      f_t support = z > 0 ? z * u * row_scale[j] : z < 0 ? z * l * row_scale[j] : 0.;
      r.gap += z * ax[j] - support;
      r.dual_objective -= support;
      f_t dz = current[2 * pb.n_variables + j] - z;
      f_t da = anchor[2 * pb.n_variables + j] - z;
      r.dy += dz * dz / row_step[j];
      r.anchor_dual += da * da / row_step[j];
    }
    return r;
  }
};

// Squared norms of the scaled cost (objective and distance slopes) and of the scaled row bounds.
struct norms_t {
  double cost = 0., bounds = 0.;
};

struct add_norms {
  __host__ __device__ norms_t operator()(norms_t a, norms_t b) const
  {
    a.cost += b.cost;
    a.bounds += b.bounds;
    return a;
  }
};

template <typename i_t, typename f_t>
struct collect_norms {
  typename problem_t<i_t, f_t>::view_t pb;
  const f_t *col_scale, *row_scale;
  f_t distance_weight, objective_weight;

  __device__ norms_t operator()(i_t j) const
  {
    norms_t r;
    if (j < pb.n_variables) {
      f_t w =
        pb.variable_types.size() && pb.is_integer_var(j) ? distance_weight * col_scale[j] : 0.;
      f_t c  = objective_weight * pb.objective_coefficients[j] * col_scale[j];
      r.cost = c * c + w * w;
    }
    if (j < pb.n_constraints) {
      f_t l = pb.constraint_lower_bounds[j], u = pb.constraint_upper_bounds[j];
      f_t bound = max(isfinite(l) ? abs(l) : f_t(0), isfinite(u) ? abs(u) : f_t(0)) * row_scale[j];
      r.bounds  = bound * bound;
    }
    return r;
  }
};

struct finite_value {
  __device__ bool operator()(double x) const { return isfinite(x); }
};

void update_result(pdfp_projection_result_t& result,
                   const metrics_t& metrics,
                   const pdfp_projection_settings_t& settings)
{
  result.primal_residual =
    !settings.relative_primal ? metrics.primal
    : settings.original_units
      ? std::sqrt(metrics.primal_sq) / (1. + std::sqrt(metrics.rhs_sq))
      : std::sqrt(metrics.primal_sq_scaled) / (1. + std::sqrt(metrics.rhs_sq_scaled));
  result.dual_residual = std::sqrt(metrics.dual) / (1. + std::sqrt(metrics.cost));
  result.relative_gap  = std::abs(metrics.gap) / (1. + metrics.target);
  if (settings.duality_gap_test) {
    result.dual_residual = std::sqrt(metrics.dual_illegal) / (1. + std::sqrt(metrics.cost));
    result.relative_gap  = std::abs(metrics.objective - metrics.dual_objective) /
                          (1. + std::abs(metrics.objective) + std::abs(metrics.dual_objective));
  }
  result.objective = metrics.objective;
  result.converged =
    result.iterations > 0 && std::isfinite(result.objective) && std::isfinite(metrics.cost) &&
    std::isfinite(metrics.target) && std::isfinite(result.primal_residual) &&
    std::isfinite(result.dual_residual) && std::isfinite(result.relative_gap) &&
    result.primal_residual <= settings.primal_ratio * settings.tolerance &&
    result.dual_residual <= settings.tolerance && result.relative_gap <= settings.tolerance;
}

}  // namespace

template <typename i_t, typename f_t>
struct pdfp_projection_t<i_t, f_t>::impl_t {
  // y = A_s x (or A_s^T x) between two of the projector's vectors, A_s = D_r A D_c.
  struct product_t {
    f_t* x;
    f_t* y;
    bool transpose;
    pdlp::cusparse_dn_vec_uptr x_descr, y_descr;
  };

  problem_t<i_t, f_t>& problem;
  rmm::device_uvector<f_t> col_scale, row_scale, current, point, anchor, target;
  rmm::device_uvector<f_t> ax, atz, atz_point, reflected_x, slack, start;
  // Per-variable and per-row multipliers of the primal and dual step, the working-set weights
  // they come from, and the square roots of the Pock-Chambolle l1 sums.
  rmm::device_uvector<f_t> col_step, row_step, col_weight, row_weight, col_root, row_root;
  rmm::device_uvector<f_t> a_values, at_values;
  rmm::device_uvector<uint8_t> a_buffer, at_buffer;
  rmm::device_scalar<f_t> one, zero;
  rmm::device_uvector<f_t> params;
  rmm::device_uvector<i_t> counters;
  pdlp::cusparse_sp_mat_uptr a_mat, at_mat;
  product_t at_current, a_reflected, a_point, at_point;
  std::array<const i_t*, 4> csr;
  cuopt::manual_cuda_graph_t chunk_graph, single_graph;
  std::array<const void*, 5> graph_view{};
  bool cusparse_spmv, prox, unit_norm;
  bool uniform_steps = true;
  f_t eta;
  bool cold                        = true;
  f_t carried_step                 = 0.;
  static constexpr int graph_chunk = 16;

  impl_t(problem_t<i_t, f_t>& p, bool spectral_step, bool use_cusparse, bool prox_distance)
    : problem(p),
      col_scale(p.n_variables, p.handle_ptr->get_stream()),
      row_scale(p.n_constraints, p.handle_ptr->get_stream()),
      current(2 * size_t(p.n_variables) + p.n_constraints, p.handle_ptr->get_stream()),
      point(current.size(), p.handle_ptr->get_stream()),
      anchor(current.size(), p.handle_ptr->get_stream()),
      target(p.n_variables, p.handle_ptr->get_stream()),
      ax(p.n_constraints, p.handle_ptr->get_stream()),
      atz(p.n_variables, p.handle_ptr->get_stream()),
      atz_point(p.n_variables, p.handle_ptr->get_stream()),
      reflected_x(p.n_variables, p.handle_ptr->get_stream()),
      slack(p.n_variables, p.handle_ptr->get_stream()),
      start(0, p.handle_ptr->get_stream()),
      col_step(p.n_variables, p.handle_ptr->get_stream()),
      row_step(p.n_constraints, p.handle_ptr->get_stream()),
      col_weight(p.n_variables, p.handle_ptr->get_stream()),
      row_weight(p.n_constraints, p.handle_ptr->get_stream()),
      col_root(p.n_variables, p.handle_ptr->get_stream()),
      row_root(p.n_constraints, p.handle_ptr->get_stream()),
      a_values(0, p.handle_ptr->get_stream()),
      at_values(0, p.handle_ptr->get_stream()),
      a_buffer(0, p.handle_ptr->get_stream()),
      at_buffer(0, p.handle_ptr->get_stream()),
      one(one_v<f_t>, p.handle_ptr->get_stream()),
      zero(zero_v<f_t>, p.handle_ptr->get_stream()),
      params(n_param_slots, p.handle_ptr->get_stream()),
      counters(2, p.handle_ptr->get_stream()),
      csr{p.offsets.data(),
          p.variables.data(),
          p.reverse_offsets.data(),
          p.reverse_constraints.data()},
      cusparse_spmv(use_cusparse),
      prox(prox_distance),
      unit_norm(spectral_step)
  {
    auto offsets = cuopt::host_copy(p.offsets, p.handle_ptr->get_stream());
    auto indices = cuopt::host_copy(p.variables, p.handle_ptr->get_stream());
    auto values  = cuopt::host_copy(p.coefficients, p.handle_ptr->get_stream());
    p.handle_ptr->sync_stream();
    std::vector<f_t> rs(p.n_constraints, 1.), cs(p.n_variables, 1.);
    std::vector<f_t> rm(p.n_constraints), cm(p.n_variables);
    std::vector<f_t> row_l1_root(p.n_constraints, 1.), col_l1_root(p.n_variables, 1.);
    // Ruiz infinity equilibration followed by Pock--Chambolle l1 equilibration.
    for (int pass = 0; pass < 11; ++pass) {
      std::fill(rm.begin(), rm.end(), 0.);
      std::fill(cm.begin(), cm.end(), 0.);
      for (i_t row = 0; row < p.n_constraints; ++row) {
        for (i_t k = offsets[row]; k < offsets[row + 1]; ++k) {
          f_t a = std::abs(values[k] * rs[row] * cs[indices[k]]);
          if (pass < 10) {
            rm[row]        = std::max(rm[row], a);
            cm[indices[k]] = std::max(cm[indices[k]], a);
          } else {
            rm[row] += a;
            cm[indices[k]] += a;
          }
        }
      }
      for (i_t row = 0; row < p.n_constraints; ++row)
        if (rm[row] > 0.) rs[row] /= std::sqrt(rm[row]);
      for (i_t j = 0; j < p.n_variables; ++j)
        if (cm[j] > 0.) cs[j] /= std::sqrt(cm[j]);
      if (pass == 10) {
        for (i_t row = 0; row < p.n_constraints; ++row)
          if (rm[row] > 0.) row_l1_root[row] = std::sqrt(rm[row]);
        for (i_t j = 0; j < p.n_variables; ++j)
          if (cm[j] > 0.) col_l1_root[j] = std::sqrt(cm[j]);
      }
    }
    // A certified spectral upper bound keeps the reflected operator nonexpansive
    // even when power iteration would underestimate a clustered leading singular value.
    std::fill(cm.begin(), cm.end(), 0.);
    f_t row_norm = 0., col_norm = 0.;
    for (i_t row = 0; row < p.n_constraints; ++row) {
      f_t sum = 0.;
      for (i_t k = offsets[row]; k < offsets[row + 1]; ++k) {
        f_t a = std::abs(values[k] * rs[row] * cs[indices[k]]);
        sum += a;
        cm[indices[k]] += a;
      }
      row_norm = std::max(row_norm, sum);
    }
    for (auto a : cm)
      col_norm = std::max(col_norm, a);
    // The bounded distance dual adds an identity block to the operator; prox has A alone, and
    // an empty A leaves the step free.
    const f_t operator_bound = (prox ? 0. : 1.) + (spectral_step ? 1. : row_norm * col_norm);
    eta                      = .998 / std::sqrt(operator_bound > 0. ? operator_bound : 1.);
    raft::copy(row_scale.data(), rs.data(), rs.size(), p.handle_ptr->get_stream());
    raft::copy(col_scale.data(), cs.data(), cs.size(), p.handle_ptr->get_stream());
    raft::copy(row_root.data(), row_l1_root.data(), rs.size(), p.handle_ptr->get_stream());
    raft::copy(col_root.data(), col_l1_root.data(), cs.size(), p.handle_ptr->get_stream());
    thrust::fill(p.handle_ptr->get_thrust_policy(), col_step.begin(), col_step.end(), f_t(1));
    thrust::fill(p.handle_ptr->get_thrust_policy(), row_step.begin(), row_step.end(), f_t(1));
    if (cusparse_spmv && !values.empty()) { build_cusparse_matrices(); }
    f_t* z_current = current.data() + 2 * size_t(p.n_variables);
    at_current     = make_product(z_current, atz.data(), true);
    a_reflected    = make_product(reflected_x.data(), ax.data(), false);
    a_point        = make_product(point.data(), ax.data(), false);
    at_point       = make_product(point.data() + 2 * size_t(p.n_variables), atz_point.data(), true);
    if (a_mat) { prepare_cusparse_buffers(); }
    p.handle_ptr->sync_stream();
  }

  void build_cusparse_matrices()
  {
    auto& p    = problem;
    auto st    = p.handle_ptr->get_stream();
    size_t nnz = p.coefficients.size();
    cuopt_assert(p.reverse_coefficients.size() == nnz, "Transpose must match the model matrix");
    a_values.resize(nnz, st);
    at_values.resize(nnz, st);
    scale_csr_values<i_t, f_t><<<(nnz + 255) / 256, 256, 0, st.get()>>>(p.n_constraints,
                                                                        nnz,
                                                                        p.offsets.data(),
                                                                        p.variables.data(),
                                                                        p.coefficients.data(),
                                                                        row_scale.data(),
                                                                        col_scale.data(),
                                                                        a_values.data());
    scale_csr_values<i_t, f_t>
      <<<(nnz + 255) / 256, 256, 0, st.get()>>>(p.n_variables,
                                                nnz,
                                                p.reverse_offsets.data(),
                                                p.reverse_constraints.data(),
                                                p.reverse_coefficients.data(),
                                                col_scale.data(),
                                                row_scale.data(),
                                                at_values.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    a_mat = pdlp::make_csr<i_t, f_t>(
      p.n_constraints, p.n_variables, nnz, p.offsets.data(), p.variables.data(), a_values.data());
    at_mat = pdlp::make_csr<i_t, f_t>(p.n_variables,
                                      p.n_constraints,
                                      nnz,
                                      p.reverse_offsets.data(),
                                      p.reverse_constraints.data(),
                                      at_values.data());
  }

  product_t make_product(f_t* x, f_t* y, bool transpose)
  {
    auto& p = problem;
    product_t product{x, y, transpose, {}, {}};
    if (a_mat) {
      product.x_descr = pdlp::make_dnvec<f_t>(transpose ? p.n_constraints : p.n_variables, x);
      product.y_descr = pdlp::make_dnvec<f_t>(transpose ? p.n_variables : p.n_constraints, y);
    }
    return product;
  }

  void prepare_cusparse_buffers()
  {
    auto& p     = problem;
    auto st     = p.handle_ptr->get_stream();
    auto handle = p.handle_ptr->get_cusparse_handle();
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsesetpointermode(handle, CUSPARSE_POINTER_MODE_DEVICE, st.get()));
    for (auto* product : {&a_reflected, &at_current}) {
      auto& buffer = product->transpose ? at_buffer : a_buffer;
      auto matrix  = product->transpose ? at_mat.get() : a_mat.get();
      size_t size  = 0;
      RAFT_CUSPARSE_TRY(
        raft::sparse::detail::cusparsespmv_buffersize(handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      one.data(),
                                                      matrix,
                                                      product->x_descr.get(),
                                                      zero.data(),
                                                      product->y_descr.get(),
                                                      CUSPARSE_SPMV_CSR_ALG2,
                                                      &size,
                                                      st.get()));
      buffer.resize(size, st);
#if CUDA_VER_12_4_UP
      pdlp::my_cusparsespmv_preprocess(handle,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       one.data(),
                                       matrix,
                                       product->x_descr.get(),
                                       zero.data(),
                                       product->y_descr.get(),
                                       CUSPARSE_SPMV_CSR_ALG2,
                                       buffer.data(),
                                       st.get());
#endif
    }
  }

  void multiply(product_t& product)
  {
    auto& p  = problem;
    auto st  = p.handle_ptr->get_stream();
    i_t rows = product.transpose ? p.n_variables : p.n_constraints;
    if (!rows) return;
    if (!cusparse_spmv) {
      scaled_spmv<false, i_t, f_t><<<(rows + 3) / 4, 128, 0, st.get()>>>(
        rows,
        product.transpose ? p.reverse_offsets.data() : p.offsets.data(),
        product.transpose ? p.reverse_constraints.data() : p.variables.data(),
        product.transpose ? p.reverse_coefficients.data() : p.coefficients.data(),
        product.transpose ? col_scale.data() : row_scale.data(),
        product.transpose ? row_scale.data() : col_scale.data(),
        product.x,
        product.y);
      return;
    }
    // A matrix without nonzeros has no cuSPARSE descriptor; its products are zero.
    if (!a_mat) {
      RAFT_CUDA_TRY(cudaMemsetAsync(product.y, 0, rows * sizeof(f_t), st.get()));
      return;
    }
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsespmv(p.handle_ptr->get_cusparse_handle(),
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         one.data(),
                                         product.transpose ? at_mat.get() : a_mat.get(),
                                         product.x_descr.get(),
                                         zero.data(),
                                         product.y_descr.get(),
                                         CUSPARSE_SPMV_CSR_ALG2,
                                         (f_t*)(product.transpose ? at_buffer : a_buffer).data(),
                                         st.get()));
  }

  // Captured graphs hold the problem view's pointers.
  std::array<const void*, 5> view_pointers() const
  {
    return {problem.variable_bounds.data(),
            problem.objective_coefficients.data(),
            problem.variable_types.data(),
            problem.constraint_lower_bounds.data(),
            problem.constraint_upper_bounds.data()};
  }

  void set_state(f_t step, f_t weight, const pdfp_projection_settings_t& settings, int inner)
  {
    set_step_state<i_t, f_t>
      <<<1, 1, 0, problem.handle_ptr->get_stream().get()>>>(params.data(),
                                                            counters.data(),
                                                            step / weight,
                                                            step * weight,
                                                            f_t(settings.distance_weight),
                                                            f_t(settings.objective_weight),
                                                            inner);
  }

  template <bool halpern>
  void iteration()
  {
    auto& p    = problem;
    auto st    = p.handle_ptr->get_stream();
    i_t n_dual = std::max(p.n_variables, p.n_constraints);
    multiply(at_current);
    if (p.n_variables)
      primal_step<halpern, i_t, f_t>
        <<<(p.n_variables + 255) / 256, 256, 0, st.get()>>>(p.view(),
                                                            col_scale.data(),
                                                            current.data(),
                                                            anchor.data(),
                                                            atz.data(),
                                                            target.data(),
                                                            point.data(),
                                                            reflected_x.data(),
                                                            slack.data(),
                                                            params.data(),
                                                            col_step.data(),
                                                            counters.data(),
                                                            prox);
    multiply(a_reflected);
    if (n_dual)
      dual_step<halpern, i_t, f_t><<<(n_dual + 255) / 256, 256, 0, st.get()>>>(p.view(),
                                                                               col_scale.data(),
                                                                               row_scale.data(),
                                                                               target.data(),
                                                                               current.data(),
                                                                               anchor.data(),
                                                                               reflected_x.data(),
                                                                               ax.data(),
                                                                               point.data(),
                                                                               params.data(),
                                                                               row_step.data(),
                                                                               counters.data(),
                                                                               prox);
  }

  // Steps for the working set at `point`, whose products ax and atz_point must be current. With
  // weights phi (variables) and psi (rows) on the Pock-Chambolle sums of the Ruiz-scaled matrix B,
  // tau_j = phi_j / sum_i |b_ij| psi_i and sigma_i = psi_i / sum_j |b_ij| phi_j keep
  // 2 |x' B' z| <= sum_j x_j^2 / tau_j + sum_i z_i^2 / sigma_i, as the unweighted sums do. A held
  // variable or a slack row with a small weight barely moves and lets the others step further.
  void update_steps(const pdfp_projection_settings_t& settings)
  {
    auto& p = problem;
    auto st = p.handle_ptr->get_stream();
    i_t n   = std::max(p.n_variables, p.n_constraints);
    if (!n) return;
    classify_working_set<i_t, f_t>
      <<<(n + 255) / 256, 256, 0, st.get()>>>(p.view(),
                                              col_scale.data(),
                                              row_scale.data(),
                                              col_root.data(),
                                              row_root.data(),
                                              target.data(),
                                              point.data(),
                                              atz_point.data(),
                                              ax.data(),
                                              f_t(settings.distance_weight),
                                              f_t(settings.objective_weight),
                                              f_t(settings.held_weight),
                                              col_weight.data(),
                                              row_weight.data());
    if (p.n_constraints)
      scaled_spmv<true, i_t, f_t>
        <<<(p.n_constraints + 3) / 4, 128, 0, st.get()>>>(p.n_constraints,
                                                          p.offsets.data(),
                                                          p.variables.data(),
                                                          p.coefficients.data(),
                                                          row_scale.data(),
                                                          col_scale.data(),
                                                          col_weight.data(),
                                                          row_step.data());
    if (p.n_variables)
      scaled_spmv<true, i_t, f_t>
        <<<(p.n_variables + 3) / 4, 128, 0, st.get()>>>(p.n_variables,
                                                        p.reverse_offsets.data(),
                                                        p.reverse_constraints.data(),
                                                        p.reverse_coefficients.data(),
                                                        col_scale.data(),
                                                        row_scale.data(),
                                                        row_weight.data(),
                                                        col_step.data());
    finish_steps<i_t, f_t><<<(n + 255) / 256, 256, 0, st.get()>>>(p.n_variables,
                                                                  p.n_constraints,
                                                                  col_weight.data(),
                                                                  row_weight.data(),
                                                                  col_step.data(),
                                                                  row_step.data());
    uniform_steps = false;
  }

  void halpern_update(int inner)
  {
    if (current.size())
      halpern_step<<<(current.size() + 255) / 256,
                     256,
                     0,
                     problem.handle_ptr->get_stream().get()>>>(
        current.size(), current.data(), point.data(), anchor.data(), f_t(1.) / (inner + 1));
  }

  // Runs up to n iterations without residual checks, starting after inner iterations since the
  // last restart; returns how many ran before the time limit. With fused_graph, chunks of
  // graph_chunk fused iterations replay as captured CUDA graphs.
  int run_unchecked(int n, int inner, const cuopt::timer_t& timer, bool fused_graph)
  {
    int done = 0;
    if (!fused_graph) {
      for (; done < n && !timer.check_time_limit(); ++done) {
        iteration<false>();
        halpern_update(inner + done + 1);
      }
      return done;
    }
    while (done < n && !timer.check_time_limit()) {
      int k      = n - done >= graph_chunk ? graph_chunk : 1;
      auto chunk = [this, k]() {
        for (int i = 0; i < k; ++i)
          this->template iteration<true>();
      };
      (k == graph_chunk ? chunk_graph : single_graph).run(problem.handle_ptr->get_stream(), chunk);
      done += k;
    }
    return done;
  }

  metrics_t metrics(const pdfp_projection_settings_t& settings)
  {
    auto& p = problem;
    multiply(a_point);
    multiply(at_point);
    return thrust::transform_reduce(
      p.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(std::max(p.n_variables, p.n_constraints)),
      collect_metrics<i_t, f_t>{p.view(),
                                col_scale.data(),
                                row_scale.data(),
                                target.data(),
                                current.data(),
                                point.data(),
                                anchor.data(),
                                ax.data(),
                                atz.data(),
                                atz_point.data(),
                                slack.data(),
                                col_step.data(),
                                row_step.data(),
                                f_t(settings.distance_weight),
                                f_t(settings.objective_weight),
                                f_t(settings.primal_bound_ratio),
                                prox},
      metrics_t{},
      add_metrics{});
  }
};

template <typename i_t, typename f_t>
pdfp_projection_t<i_t, f_t>::pdfp_projection_t(problem_t<i_t, f_t>& problem,
                                               bool spectral_step,
                                               bool cusparse_spmv,
                                               bool prox_distance)
  : impl_(std::make_unique<impl_t>(problem, spectral_step, cusparse_spmv, prox_distance))
{
}

template <typename i_t, typename f_t>
void pdfp_projection_t<i_t, f_t>::set_cold_start(const rmm::device_uvector<f_t>& primal)
{
  auto& p = impl_->problem;
  cuopt_assert(primal.size() == size_t(p.n_variables), "Cold start size must match the model");
  impl_->start.resize(primal.size(), p.handle_ptr->get_stream());
  raft::copy(impl_->start.data(), primal.data(), primal.size(), p.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
pdfp_projection_t<i_t, f_t>::~pdfp_projection_t() = default;

template <typename i_t, typename f_t>
void pdfp_projection_t<i_t, f_t>::reset()
{
  impl_->cold         = true;
  impl_->carried_step = 0.;
}

template <typename i_t, typename f_t>
pdfp_projection_result_t pdfp_projection_t<i_t, f_t>::project(
  const rmm::device_uvector<f_t>& input,
  rmm::device_uvector<f_t>& output,
  const pdfp_projection_settings_t& settings)
{
  auto& s     = *impl_;
  auto& p     = s.problem;
  auto stream = p.handle_ptr->get_stream();
  cuopt_expects(s.col_scale.size() == size_t(p.n_variables) &&
                  s.row_scale.size() == size_t(p.n_constraints) &&
                  s.csr == std::array<const i_t*, 4>{p.offsets.data(),
                                                     p.variables.data(),
                                                     p.reverse_offsets.data(),
                                                     p.reverse_constraints.data()},
                error_type_t::ValidationError,
                "Recreate the PDFP projector after changing the model matrix");
  cuopt_expects(input.size() == size_t(p.n_variables),
                error_type_t::ValidationError,
                "PDFP target size must match the model");
  cuopt_expects(std::isfinite(settings.tolerance) && settings.tolerance > 0. &&
                  settings.time_limit >= 0. && settings.iteration_limit >= 0 &&
                  settings.check_interval > 0 && std::isfinite(settings.distance_weight) &&
                  settings.distance_weight >= 0. && std::isfinite(settings.objective_weight) &&
                  std::isfinite(settings.primal_bound_ratio) && settings.primal_bound_ratio >= 0. &&
                  std::isfinite(settings.weight_gain) && settings.weight_gain > 0. &&
                  std::isfinite(settings.weight_limit) && settings.weight_limit >= 1. &&
                  std::isfinite(settings.step_growth_limit) && settings.step_growth_limit >= 1. &&
                  settings.held_weight > 0. && settings.held_weight <= 1.,
                error_type_t::ValidationError,
                "Invalid PDFP projection settings");
  bool finite = thrust::transform_reduce(p.handle_ptr->get_thrust_policy(),
                                         input.begin(),
                                         input.end(),
                                         finite_value{},
                                         true,
                                         thrust::logical_and<bool>{});
  cuopt_expects(finite, error_type_t::ValidationError, "PDFP target must be finite");
  if (s.a_mat) {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
      p.handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, stream.get()));
  }
  cuopt::timer_t timer(settings.time_limit);
  pdfp_projection_result_t result;
  if (s.cold)
    thrust::fill(p.handle_ptr->get_thrust_policy(), s.current.begin(), s.current.end(), 0.);
  if (p.n_variables)
    initialize_projection<i_t, f_t><<<(p.n_variables + 255) / 256, 256, 0, stream.get()>>>(
      p.view(),
      s.col_scale.data(),
      input.data(),
      s.start.size() ? s.start.data() : nullptr,
      s.target.data(),
      s.current.data(),
      s.prox ? f_t(0) : f_t(settings.distance_weight),
      s.cold,
      settings.start_at_target);
  s.cold = false;
  raft::copy(s.anchor.data(), s.current.data(), s.current.size(), stream);
  raft::copy(s.point.data(), s.current.data(), s.current.size(), stream);
  thrust::fill(p.handle_ptr->get_thrust_policy(), s.slack.begin(), s.slack.end(), 0.);
  thrust::fill(p.handle_ptr->get_thrust_policy(), s.atz.begin(), s.atz.end(), 0.);
  f_t weight = std::sqrt(1. + double(p.n_integer_vars) / std::max(1, p.n_constraints));
  if (settings.balanced_weight) {
    const auto norms = thrust::transform_reduce(
      p.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(std::max(p.n_variables, p.n_constraints)),
      collect_norms<i_t, f_t>{p.view(),
                              s.col_scale.data(),
                              s.row_scale.data(),
                              f_t(settings.distance_weight),
                              f_t(settings.objective_weight)},
      norms_t{},
      add_norms{});
    weight = (std::sqrt(norms.cost) + 1.) / (std::sqrt(norms.bounds) + 1.);
  }
  const bool weighted_steps = settings.held_weight < 1.;
  cuopt_expects(!weighted_steps || (s.prox && s.unit_norm),
                error_type_t::ValidationError,
                "Working-set steps need the prox distance and the unit-norm step");
  if (weighted_steps) {
    s.multiply(s.a_point);
    s.multiply(s.at_point);
    s.update_steps(settings);
  } else if (!s.uniform_steps) {
    thrust::fill(p.handle_ptr->get_thrust_policy(), s.col_step.begin(), s.col_step.end(), f_t(1));
    thrust::fill(p.handle_ptr->get_thrust_policy(), s.row_step.begin(), s.row_step.end(), f_t(1));
    s.uniform_steps = true;
  }
  const f_t safe_step = s.eta;
  f_t step            = settings.adaptive_step && s.carried_step > 0. ? s.carried_step : safe_step;
  // Smallest step bound seen at the checks since the last restart.
  double step_bound       = INFINITY;
  double initial_residual = -1., last_residual = INFINITY, integral = 0.;
  int inner      = 0;
  int last_check = -1;
  if (s.graph_view != s.view_pointers()) {
    s.chunk_graph.reset();
    s.single_graph.reset();
    s.graph_view = s.view_pointers();
  }
  int iter = 0;
  while (iter < settings.iteration_limit && !timer.check_time_limit()) {
    s.set_state(step, weight, settings, inner);
    // Iterations between residual checks run fused; checks come right after a restart, every
    // check_interval iterations and at the iteration limit.
    if (inner > 0) {
      int next_check = std::min((iter / settings.check_interval + 1) * settings.check_interval,
                                settings.iteration_limit);
      int fused      = next_check - iter - 1;
      int done       = s.run_unchecked(fused, inner, timer, settings.fused_graph);
      iter += done;
      inner += done;
      result.iterations = iter;
      if (done < fused || timer.check_time_limit()) break;
    }
    ++iter;
    ++inner;
    s.template iteration<false>();
    result.iterations = iter;
    bool restart      = false;
    auto metrics      = s.metrics(settings);
    update_result(result, metrics, settings);
    last_check = iter;
    if (result.converged) break;
    const double movement = weight * metrics.dx + metrics.dy / weight;
    if (settings.adaptive_step) {
      const double bound =
        std::abs(metrics.cross) > 0. ? movement / (2. * std::abs(metrics.cross)) : INFINITY;
      step_bound = std::min(step_bound, bound);
      if (step > safe_step && !(step <= bound)) {
        ++result.step_cuts;
        step = std::max(safe_step, std::isfinite(bound) ? f_t(.7 * bound) : f_t(.5 * step));
        // The period keeps its point only if it still reduced the fixed-point error.
        const double cut_residual = std::sqrt(std::max(0., movement - 2 * step * metrics.cross));
        if (inner > 1 && std::isfinite(cut_residual) && cut_residual <= initial_residual) {
          raft::copy(s.current.data(), s.point.data(), s.point.size(), stream);
          raft::copy(s.anchor.data(), s.point.data(), s.point.size(), stream);
        } else {
          raft::copy(s.current.data(), s.anchor.data(), s.anchor.size(), stream);
          raft::copy(s.point.data(), s.anchor.data(), s.anchor.size(), stream);
        }
        inner         = 0;
        step_bound    = INFINITY;
        last_residual = INFINITY;
        continue;
      }
    }
    double residual = std::sqrt(std::max(0., movement - 2 * step * metrics.cross));
    if (inner == 1)
      initial_residual = residual;
    else {
      restart = iter == settings.check_interval || residual <= .2 * initial_residual ||
                (residual <= .5 * initial_residual && residual > last_residual) ||
                inner >= .36 * iter;
    }
    if (restart && metrics.anchor_x > 1e-16 && metrics.anchor_dual > 1e-16 &&
        metrics.anchor_x < 1e24 && metrics.anchor_dual < 1e24) {
      double error = std::log(std::sqrt(metrics.anchor_dual / metrics.anchor_x) / weight);
      integral     = .3 * integral + error;
      weight       = std::clamp(weight * std::exp(settings.weight_gain * error + .01 * integral),
                          1. / settings.weight_limit,
                          settings.weight_limit);
    }
    last_residual = residual;
    if (restart && settings.adaptive_step) {
      step = std::clamp(
        std::min(f_t(step * (1. + std::pow(result.restarts + 1., -.6))), f_t(.8 * step_bound)),
        safe_step,
        f_t(safe_step * settings.step_growth_limit));
      step_bound = INFINITY;
    }
    if (restart) {
      raft::copy(s.current.data(), s.point.data(), s.point.size(), stream);
      raft::copy(s.anchor.data(), s.point.data(), s.point.size(), stream);
      inner = 0;
      ++result.restarts;
      if (weighted_steps) { s.update_steps(settings); }
    } else {
      s.halpern_update(inner);
    }
  }
  if (last_check != result.iterations) update_result(result, s.metrics(settings), settings);
  if (!settings.anchor_halpern) {
    raft::copy(s.current.data(), s.point.data(), s.point.size(), stream);
  }
  output.resize(p.n_variables, stream);
  if (p.n_variables)
    unscale_output<<<(p.n_variables + 255) / 256, 256, 0, stream.get()>>>(
      p.n_variables, s.point.data(), s.col_scale.data(), output.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  p.handle_ptr->sync_stream();
  result.elapsed_time  = timer.elapsed_time();
  result.primal_weight = weight;
  result.step          = step;
  if (weighted_steps) {
    result.moving_variables = thrust::count_if(
      p.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(p.n_variables),
      [w = s.col_weight.data(), r = s.col_root.data()] __device__(i_t j) { return w[j] == r[j]; });
    result.acting_rows = thrust::count_if(
      p.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(p.n_constraints),
      [w = s.row_weight.data(), r = s.row_root.data()] __device__(i_t i) { return w[i] == r[i]; });
  }
  if (settings.adaptive_step) { s.carried_step = step; }
  return result;
}

#if MIP_INSTANTIATE_FLOAT
template class pdfp_projection_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class pdfp_projection_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
