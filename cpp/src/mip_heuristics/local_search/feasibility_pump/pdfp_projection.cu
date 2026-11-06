/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pdfp_projection.cuh"

#include <cuopt/error.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/util/cuda_utils.cuh>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
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
template <typename i_t, typename f_t>
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
    sum += values[k] * col_scale[indices[k]] * x[indices[k]];
  }
  sum = raft::warpReduce(sum);
  if (lane == 0) output[row] = row_scale[row] * sum;
}

template <typename i_t, typename f_t>
__global__ void initialize_projection(typename problem_t<i_t, f_t>::view_t pb,
                                      const f_t* scale,
                                      const f_t* input,
                                      const f_t* start,
                                      f_t* target,
                                      f_t* current,
                                      f_t distance_weight,
                                      bool cold)
{
  i_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= pb.n_variables) return;
  target[j] = input[j] / scale[j];
  if (cold) current[j] = start != nullptr ? start[j] / scale[j] : target[j];
  current[j] = clip(current[j],
                    get_lower(pb.variable_bounds[j]) / scale[j],
                    get_upper(pb.variable_bounds[j]) / scale[j]);
  f_t w      = pb.variable_types.size() && pb.is_integer_var(j) ? distance_weight * scale[j] : 0.;
  current[pb.n_variables + j] = cold ? 0. : clip(current[pb.n_variables + j], -w, w);
}

template <typename i_t, typename f_t>
__global__ void primal_step(typename problem_t<i_t, f_t>::view_t pb,
                            const f_t* scale,
                            const f_t* current,
                            const f_t* atz,
                            f_t* point,
                            f_t* reflected_x,
                            f_t* slack,
                            f_t tau,
                            f_t objective_weight)
{
  i_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= pb.n_variables) return;
  f_t v = current[j] - tau * (objective_weight * pb.objective_coefficients[j] * scale[j] + atz[j] -
                              current[pb.n_variables + j]);
  f_t x = clip(
    v, get_lower(pb.variable_bounds[j]) / scale[j], get_upper(pb.variable_bounds[j]) / scale[j]);
  point[j]       = x;
  reflected_x[j] = 2 * x - current[j];
  slack[j]       = (x - v) / tau;
}

template <typename i_t, typename f_t>
__global__ void dual_step(typename problem_t<i_t, f_t>::view_t pb,
                          const f_t* col_scale,
                          const f_t* row_scale,
                          const f_t* target,
                          const f_t* current,
                          const f_t* reflected_x,
                          const f_t* ax,
                          f_t* point,
                          f_t sigma,
                          f_t distance_weight)
{
  i_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < pb.n_variables) {
    f_t w = pb.variable_types.size() && pb.is_integer_var(j) ? distance_weight * col_scale[j] : 0.;
    point[pb.n_variables + j] =
      clip(current[pb.n_variables + j] + sigma * (target[j] - reflected_x[j]), -w, w);
  }
  if (j < pb.n_constraints) {
    f_t v                         = current[2 * pb.n_variables + j] + sigma * ax[j];
    f_t l                         = pb.constraint_lower_bounds[j] * row_scale[j];
    f_t u                         = pb.constraint_upper_bounds[j] * row_scale[j];
    f_t q                         = v / sigma;
    point[2 * pb.n_variables + j] = q < l ? v - sigma * l : q > u ? v - sigma * u : 0.;
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
    return a;
  }
};

template <typename i_t, typename f_t>
struct collect_metrics {
  typename problem_t<i_t, f_t>::view_t pb;
  const f_t *col_scale, *row_scale, *target, *current, *point, *anchor;
  const f_t *ax, *atz, *atz_point, *slack;
  f_t distance_weight, objective_weight;

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
      r.gap        = y * (target[j] - x) - dist;
      r.objective  = c * x + dist;
      r.target     = w * abs(target[j]);
      f_t dx = current[j] - x, dy = current[pb.n_variables + j] - y;
      r.dx    = dx * dx;
      r.dy    = dy * dy;
      r.cross = dx * (atz[j] - atz_point[j] - dy);
      f_t da = anchor[j] - x, db = anchor[pb.n_variables + j] - y;
      r.anchor_x    = da * da;
      r.anchor_dual = db * db;
    }
    if (j < pb.n_constraints) {
      f_t activity = ax[j] / row_scale[j];
      f_t l = pb.constraint_lower_bounds[j], u = pb.constraint_upper_bounds[j];
      r.primal           = isfinite(activity) ? abs(activity - clip(activity, l, u)) : INFINITY;
      f_t rhs            = max(isfinite(l) ? abs(l) : f_t(0), isfinite(u) ? abs(u) : f_t(0));
      r.primal_sq        = r.primal * r.primal;
      r.primal_sq_scaled = r.primal_sq * row_scale[j] * row_scale[j];
      r.rhs_sq           = rhs * rhs;
      r.rhs_sq_scaled    = r.rhs_sq * row_scale[j] * row_scale[j];
      f_t z              = point[2 * pb.n_variables + j];
      // The zero case avoids multiplying an infinite row bound by zero.
      f_t support = z > 0 ? z * u * row_scale[j] : z < 0 ? z * l * row_scale[j] : 0.;
      r.gap += z * ax[j] - support;
      f_t dz = current[2 * pb.n_variables + j] - z;
      f_t da = anchor[2 * pb.n_variables + j] - z;
      r.dy += dz * dz;
      r.anchor_dual += da * da;
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
  result.objective     = metrics.objective;
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
  problem_t<i_t, f_t>& problem;
  rmm::device_uvector<f_t> col_scale, row_scale, current, point, anchor, target;
  rmm::device_uvector<f_t> ax, atz, atz_point, reflected_x, slack, start;
  f_t eta;
  bool cold = true;

  impl_t(problem_t<i_t, f_t>& p, bool spectral_step)
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
      start(0, p.handle_ptr->get_stream())
  {
    auto offsets = cuopt::host_copy(p.offsets, p.handle_ptr->get_stream());
    auto indices = cuopt::host_copy(p.variables, p.handle_ptr->get_stream());
    auto values  = cuopt::host_copy(p.coefficients, p.handle_ptr->get_stream());
    p.handle_ptr->sync_stream();
    std::vector<f_t> rs(p.n_constraints, 1.), cs(p.n_variables, 1.);
    std::vector<f_t> rm(p.n_constraints), cm(p.n_variables);
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
    eta = .998 / std::sqrt(1. + (spectral_step ? 1. : row_norm * col_norm));
    raft::copy(row_scale.data(), rs.data(), rs.size(), p.handle_ptr->get_stream());
    raft::copy(col_scale.data(), cs.data(), cs.size(), p.handle_ptr->get_stream());
    p.handle_ptr->sync_stream();
  }

  void multiply(const f_t* x, f_t* out, bool transpose)
  {
    auto& p  = problem;
    i_t rows = transpose ? p.n_variables : p.n_constraints;
    if (!rows) return;
    scaled_spmv<<<(rows + 3) / 4, 128, 0, p.handle_ptr->get_stream().get()>>>(
      rows,
      transpose ? p.reverse_offsets.data() : p.offsets.data(),
      transpose ? p.reverse_constraints.data() : p.variables.data(),
      transpose ? p.reverse_coefficients.data() : p.coefficients.data(),
      transpose ? col_scale.data() : row_scale.data(),
      transpose ? row_scale.data() : col_scale.data(),
      x,
      out);
  }

  metrics_t metrics(const pdfp_projection_settings_t& settings)
  {
    auto& p = problem;
    multiply(point.data(), ax.data(), false);
    multiply(point.data() + 2 * p.n_variables, atz_point.data(), true);
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
                                f_t(settings.distance_weight),
                                f_t(settings.objective_weight)},
      metrics_t{},
      add_metrics{});
  }
};

template <typename i_t, typename f_t>
pdfp_projection_t<i_t, f_t>::pdfp_projection_t(problem_t<i_t, f_t>& problem, bool spectral_step)
  : impl_(std::make_unique<impl_t>(problem, spectral_step))
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
  impl_->cold = true;
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
  cuopt_expects(
    s.col_scale.size() == size_t(p.n_variables) && s.row_scale.size() == size_t(p.n_constraints),
    error_type_t::ValidationError,
    "Recreate the PDFP projector after changing model dimensions");
  cuopt_expects(input.size() == size_t(p.n_variables),
                error_type_t::ValidationError,
                "PDFP target size must match the model");
  cuopt_expects(std::isfinite(settings.tolerance) && settings.tolerance > 0. &&
                  settings.time_limit >= 0. && settings.iteration_limit >= 0 &&
                  settings.check_interval > 0 && std::isfinite(settings.distance_weight) &&
                  settings.distance_weight >= 0. && std::isfinite(settings.objective_weight),
                error_type_t::ValidationError,
                "Invalid PDFP projection settings");
  bool finite = thrust::transform_reduce(p.handle_ptr->get_thrust_policy(),
                                         input.begin(),
                                         input.end(),
                                         finite_value{},
                                         true,
                                         thrust::logical_and<bool>{});
  cuopt_expects(finite, error_type_t::ValidationError, "PDFP target must be finite");
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
      f_t(settings.distance_weight),
      s.cold);
  s.cold = false;
  raft::copy(s.anchor.data(), s.current.data(), s.current.size(), stream);
  raft::copy(s.point.data(), s.current.data(), s.current.size(), stream);
  thrust::fill(p.handle_ptr->get_thrust_policy(), s.slack.begin(), s.slack.end(), 0.);
  thrust::fill(p.handle_ptr->get_thrust_policy(), s.atz.begin(), s.atz.end(), 0.);
  f_t weight              = std::sqrt(1. + double(p.n_integer_vars) / std::max(1, p.n_constraints));
  double initial_residual = -1., last_residual = INFINITY, integral = 0.;
  int inner      = 0;
  int last_check = -1;
  i_t count      = std::max(p.n_variables, p.n_constraints);
  for (int iter = 1; iter <= settings.iteration_limit && !timer.check_time_limit(); ++iter) {
    ++inner;
    s.multiply(s.current.data() + 2 * p.n_variables, s.atz.data(), true);
    if (p.n_variables)
      primal_step<i_t, f_t>
        <<<(p.n_variables + 255) / 256, 256, 0, stream.get()>>>(p.view(),
                                                                s.col_scale.data(),
                                                                s.current.data(),
                                                                s.atz.data(),
                                                                s.point.data(),
                                                                s.reflected_x.data(),
                                                                s.slack.data(),
                                                                s.eta / weight,
                                                                f_t(settings.objective_weight));
    s.multiply(s.reflected_x.data(), s.ax.data(), false);
    if (count)
      dual_step<i_t, f_t>
        <<<(count + 255) / 256, 256, 0, stream.get()>>>(p.view(),
                                                        s.col_scale.data(),
                                                        s.row_scale.data(),
                                                        s.target.data(),
                                                        s.current.data(),
                                                        s.reflected_x.data(),
                                                        s.ax.data(),
                                                        s.point.data(),
                                                        s.eta * weight,
                                                        f_t(settings.distance_weight));
    result.iterations = iter;
    bool check =
      inner == 1 || iter % settings.check_interval == 0 || iter == settings.iteration_limit;
    bool restart = false;
    if (check) {
      auto metrics = s.metrics(settings);
      update_result(result, metrics, settings);
      last_check = iter;
      if (result.converged) break;
      double residual = std::sqrt(
        std::max(0., weight * metrics.dx + metrics.dy / weight - 2 * s.eta * metrics.cross));
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
        weight       = std::clamp(weight * std::exp(.99 * error + .01 * integral), 1e-6, 1e6);
      }
      last_residual = residual;
    }
    if (restart) {
      raft::copy(s.current.data(), s.point.data(), s.point.size(), stream);
      raft::copy(s.anchor.data(), s.point.data(), s.point.size(), stream);
      inner = 0;
    } else if (s.current.size()) {
      halpern_step<<<(s.current.size() + 255) / 256, 256, 0, stream.get()>>>(
        s.current.size(), s.current.data(), s.point.data(), s.anchor.data(), f_t(1.) / (inner + 1));
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
  result.elapsed_time = timer.elapsed_time();
  return result;
}

#if MIP_INSTANTIATE_FLOAT
template class pdfp_projection_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class pdfp_projection_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
