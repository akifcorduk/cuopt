/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/problem/problem.cuh>

#include <memory>

namespace cuopt::mathematical_optimization::mip {

struct pdfp_projection_settings_t {
  double tolerance        = 1e-4;
  double time_limit       = 10.;
  int iteration_limit     = 500000;
  int check_interval      = 200;
  double distance_weight  = 1.;
  double objective_weight = 0.;
};

struct pdfp_projection_result_t {
  bool converged         = false;
  int iterations         = 0;
  double primal_residual = INFINITY;
  double dual_residual   = INFINITY;
  double relative_gap    = INFINITY;
  double objective       = INFINITY;
  double elapsed_time    = 0.;
};

// Solves min distance_weight * sum_{j in I}|x_j-target_j| + objective_weight*c'x
// over the LP relaxation, without augmenting the constraint matrix. Recreate the
// projector after matrix or dimension edits. Bounds, objective coefficients and
// integer masks are read afresh on each projection.
template <typename i_t, typename f_t>
class pdfp_projection_t {
 public:
  explicit pdfp_projection_t(problem_t<i_t, f_t>& problem);
  ~pdfp_projection_t();
  pdfp_projection_result_t project(const rmm::device_uvector<f_t>& target,
                                   rmm::device_uvector<f_t>& output,
                                   const pdfp_projection_settings_t& settings = {});
  void reset();

 private:
  struct impl_t;
  std::unique_ptr<impl_t> impl_;
};

}  // namespace cuopt::mathematical_optimization::mip
