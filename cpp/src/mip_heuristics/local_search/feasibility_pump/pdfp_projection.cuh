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
  double primal_ratio = .1;
  // The absolute primal test divides each row's violation by 1 + primal_bound_ratio * |bound|,
  // i.e. it measures the violation in units of a row tolerance abs + rel * |bound|.
  double primal_bound_ratio = 0.;
  bool relative_primal      = false;
  bool original_units       = true;
  // Anchor the next projection at the last Halpern iterate instead of the bounded PDHG point.
  bool anchor_halpern = false;
  // Stop on the duality gap |P - D| / (1 + |P| + |D|) of the row duals' own dual bound, with only
  // reduced costs that an infinite bound cannot absorb counted as dual residual, as PDLP does for
  // the epigraph LP; false tests the step's stationarity residual and complementarity instead.
  bool duality_gap_test = false;
  // Start each projection from the primal weight (||cost|| + 1) / (||row bounds|| + 1) of the
  // scaled problem instead of sqrt(1 + integers / rows).
  bool balanced_weight = false;
  // Proportional gain of the primal weight update at restarts; PDLP's smoothing is 0.5.
  double weight_gain = .99;
  // The primal weight stays within [1 / weight_limit, weight_limit].
  double weight_limit = 1e6;
  // Start the primal iterate at the target (as the epigraph LP is warm-started) instead of at the
  // previous projection; the duals carry over either way.
  bool start_at_target = false;
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
  // Primal weight at exit and the number of restarts taken.
  double primal_weight = 0.;
  int restarts         = 0;
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
  // prox_distance applies the L1 term in the primal step instead of through a bounded dual per
  // integer variable; the operator is then A alone and the step bound drops the identity block.
  explicit pdfp_projection_t(problem_t<i_t, f_t>& problem,
                             bool spectral_step = false,
                             bool cusparse_spmv = true,
                             bool prox_distance = false);
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
