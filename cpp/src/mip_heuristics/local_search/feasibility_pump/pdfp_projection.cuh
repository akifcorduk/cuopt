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
  // Primal test: max absolute row violation (original units) <= primal_ratio * tolerance, or with
  // relative_primal ||violation||_2 / (1 + ||b||_2) <= primal_ratio * tolerance, measured in
  // original row units when original_units is set and on the scaled rows otherwise.
  double primal_ratio  = .1;
  bool relative_primal = false;
  bool original_units  = true;
  // Anchor the next projection at the last Halpern iterate instead of the bounded PDHG point.
  bool anchor_halpern = false;
  // Between residual checks, fuse the Halpern update into the primal and dual steps and replay the
  // iterations as captured CUDA graphs; false launches the unfused kernels per iteration.
  bool fused_graph = true;
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
  // spectral_step uses ||A||_2 <= 1, which the final Pock-Chambolle scaling pass certifies,
  // instead of the looser bound sqrt(||A||_1 ||A||_inf). cusparse_spmv runs the products with
  // cuSPARSE on scaled copies of A and A^T instead of the one-warp-per-row kernel; the copies
  // borrow the model's CSR structure, so the model's matrix must not be reallocated while the
  // projector lives.
  explicit pdfp_projection_t(problem_t<i_t, f_t>& problem,
                             bool spectral_step = false,
                             bool cusparse_spmv = true);
  ~pdfp_projection_t();
  pdfp_projection_result_t project(const rmm::device_uvector<f_t>& target,
                                   rmm::device_uvector<f_t>& output,
                                   const pdfp_projection_settings_t& settings = {});
  // Primal point (original units) the next cold projection starts from, instead of the target.
  void set_cold_start(const rmm::device_uvector<f_t>& primal);
  void reset();

 private:
  struct impl_t;
  std::unique_ptr<impl_t> impl_;
};

}  // namespace cuopt::mathematical_optimization::mip
