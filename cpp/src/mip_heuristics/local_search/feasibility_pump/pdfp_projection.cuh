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
  // Adapt the step to the iterates as PDLP's adaptive rule does. Every residual check bounds the
  // step by (w |dx|^2 + |dy|^2 / w) / (2 |dx' A' dy|) of its own iteration. The step grows at
  // restarts while the checks of the period allow it, up to step_growth_limit times the certified
  // step; a check that finds it above its bound cuts it and restarts. The step carries over to the
  // next projection.
  bool adaptive_step       = false;
  double step_growth_limit = 64.;
  // Below 1: per-variable and per-row steps from the working set at each restart. Variables held
  // at a bound or at the target and slack rows get this weight in the Pock-Chambolle sums, which
  // enlarges the steps of the variables that move and of the rows that act on them. The steps
  // stay certified for the whole matrix. Needs the prox distance and the unit-norm step.
  double held_weight = 1.;
  // An objective bound as a pull instead of a row. At a restart whose point has c'x above
  // objective_bound, the objective weight in use grows by half, from at least pull_floor, when it
  // has settled there (the projection converged, or the period cut its fixed-point error to a
  // fifth) or when, ten check intervals after its last growth, the period's rate leaves the bound
  // further away than the iterations spent so far. The weight added to objective_weight stops at
  // pull_limit; two thirds of it carry over to the next projection. pull_limit = 0 turns it off.
  // Needs the prox distance.
  double objective_bound = INFINITY;
  double pull_limit      = 0.;
  double pull_floor      = 0.;
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
  // Step eta: tau = eta / weight, sigma = eta * weight.
  double step = 0.;
  // Checks that found the adaptive step above its bound.
  int step_cuts = 0;
  // With working-set steps: variables and rows with full weight at exit.
  int moving_variables = -1;
  int acting_rows      = -1;
  // c'x at exit, the weight the pull added to objective_weight, and the times it grew.
  double linear_objective = NAN;
  double objective_pull   = 0.;
  int pull_updates        = 0;
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
