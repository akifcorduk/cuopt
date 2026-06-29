/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

// Runtime evaluator for the calibrated per-iteration work-unit models. Includes the generated
// coefficient headers and exposes per-leaf "work per iteration" predictors. With wups == 1 a work
// unit is calibrated to ~1 second on the calibration GPU, so a leaf's work-unit budget converts to
// a reproducible iteration/work stop independent of wall clock. Static structural features are
// computed once per problem (work_features_t); dynamic per-iteration quantities are passed in at
// the call.

#include <mip_heuristics/deterministic_calibrator/device_model.hpp>
#include <mip_heuristics/deterministic_calibrator/generated/bp_device_two_kernel_coeffs.hpp>
#include <mip_heuristics/deterministic_calibrator/generated/bp_work_coeffs.hpp>
#include <mip_heuristics/deterministic_calibrator/generated/fj_device_coeffs.hpp>
#include <mip_heuristics/deterministic_calibrator/generated/fj_work_coeffs.hpp>
#include <mip_heuristics/deterministic_calibrator/generated/pdlp_device_coeffs.hpp>
#include <mip_heuristics/deterministic_calibrator/generated/pdlp_work_coeffs.hpp>
#include <mip_heuristics/deterministic_calibrator/generated/repair_device_coeffs.hpp>
#include <mip_heuristics/deterministic_calibrator/generated/repair_work_coeffs.hpp>
#include <mip_heuristics/deterministic_calibrator/gpu_features.hpp>
#include <mip_heuristics/deterministic_calibrator/linear_work_model.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace cuopt::linear_programming::detail::calib {

// Static structural features of a problem, computed once from its CSR offsets.
struct work_features_t {
  double n_vars{0.0};
  double n_constraints{0.0};
  double nnz{0.0};
  double row_nnz_var{0.0};         // variance of nnz per constraint (A rows)
  double col_nnz_var{0.0};         // variance of nnz per variable (A^T rows)
  double max_row_nnz{0.0};         // deepest constraint row
  double cons_warp_loads{0.0};     // sum ceil(rowsize/32) over constraints  (A SpMV / activity)
  double var_warp_loads{0.0};      // sum ceil(coldeg/32) over variables      (A^T SpMV / update)
  double frontier_work_mean{0.0};  // sum_c rowsize^2 / n_vars  (FJ per-step frontier expectation)
};

inline std::pair<double, double> mean_var_of_diffs(const std::vector<int>& offsets)
{
  if (offsets.size() < 2) { return {0.0, 0.0}; }
  const std::size_t n = offsets.size() - 1;
  double mean         = 0.0;
  for (std::size_t r = 0; r < n; ++r) {
    mean += (double)(offsets[r + 1] - offsets[r]);
  }
  mean /= (double)n;
  double var = 0.0;
  for (std::size_t r = 0; r < n; ++r) {
    const double d = (double)(offsets[r + 1] - offsets[r]) - mean;
    var += d * d;
  }
  return {mean, var / (double)n};
}

// Build the static features from host CSR offsets (A: row offsets, A^T: reverse/col offsets).
inline work_features_t compute_work_features(const std::vector<int>& row_offsets,
                                             const std::vector<int>& col_offsets,
                                             double n_vars,
                                             double nnz)
{
  work_features_t f;
  f.n_vars        = n_vars;
  f.n_constraints = row_offsets.size() > 0 ? (double)(row_offsets.size() - 1) : 0.0;
  f.nnz           = nnz;
  f.row_nnz_var   = mean_var_of_diffs(row_offsets).second;
  f.col_nnz_var   = mean_var_of_diffs(col_offsets).second;
  double sum_sq   = 0.0;
  for (std::size_t r = 0; r + 1 < row_offsets.size(); ++r) {
    const double rs = (double)(row_offsets[r + 1] - row_offsets[r]);
    f.max_row_nnz   = std::max(f.max_row_nnz, rs);
    f.cons_warp_loads += (double)(((long)rs + 31) / 32);
    sum_sq += rs * rs;
  }
  for (std::size_t v = 0; v + 1 < col_offsets.size(); ++v) {
    f.var_warp_loads += (double)(((long)(col_offsets[v + 1] - col_offsets[v]) + 31) / 32);
  }
  f.frontier_work_mean = n_vars > 0 ? sum_sq / n_vars : 0.0;
  return f;
}

template <std::size_t N>
inline double dot(const std::array<double, N>& c, const std::vector<double>& x)
{
  double w = 0.0;
  for (std::size_t i = 0; i < N && i < x.size(); ++i) {
    w += c[i] * x[i];
  }
  return w;
}

// ---- per-leaf work-per-iteration predictors (units: calibrated seconds, wups == 1) ----

// FJ per host-step. `frontier_work_step` is the dynamic per-step frontier work (FJ already computes
// it as deterministic_batch_work); for an a-priori estimate pass features.frontier_work_mean.
inline double fj_work_per_step(const work_features_t& f, double frontier_work_step)
{
  std::vector<double> x{
    1.0, f.n_vars, f.n_constraints, f.nnz, f.row_nnz_var, f.col_nnz_var, frontier_work_step};
  double w = dot(fj_work_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

// PDLP per inner iteration (uniform; constant per instance).
inline double pdlp_work_per_iter(const work_features_t& f)
{
  const double total = f.cons_warp_loads + f.var_warp_loads;
  std::vector<double> x{1.0,
                        f.n_vars,
                        f.n_constraints,
                        f.nnz,
                        f.row_nnz_var,
                        f.col_nnz_var,
                        f.cons_warp_loads,
                        f.var_warp_loads,
                        total,
                        std::max(0.0, total - pdlp_excess_warp_loads_threshold)};
  double w = dot(pdlp_work_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

// bounds-repair per move (compute_best_shift + compute_damages + apply_move + ii-violation pass).
inline double repair_work_per_move(const work_features_t& f,
                                   double n_candidates,
                                   double cur_cstr_rowsize)
{
  std::vector<double> x{1.0,
                        f.n_vars,
                        f.n_constraints,
                        f.nnz,
                        f.row_nnz_var,
                        f.col_nnz_var,
                        n_candidates,
                        cur_cstr_rowsize,
                        f.cons_warp_loads,
                        std::max(0.0, f.cons_warp_loads - repair_excess_warp_loads_threshold)};
  double w = dot(repair_work_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

// Cost of a single full constraint-activity pass (compute every constraint's LHS over all nnz).
// This is exactly the kernel the bound-presolve activity model was calibrated on, evaluated for the
// full constraint set, so it doubles as the per-point cost of line-segment search (get_quality) and
// any other one-shot full activity evaluation. Activity-only (no bounds-update term).
inline double activity_work_full(const work_features_t& f)
{
  std::vector<double> x{1.0,
                        f.n_vars,
                        f.n_constraints,
                        f.nnz,
                        f.row_nnz_var,
                        f.col_nnz_var,
                        f.max_row_nnz,
                        f.n_constraints,
                        f.nnz,
                        f.cons_warp_loads,
                        std::max(0.0, f.cons_warp_loads - bp_excess_warp_loads_threshold)};
  double w = dot(bp_activity_work_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

// Bound-presolve per iteration: activity sub-kernel + bounds-update sub-kernel. Dynamic features
// are the current changed-constraint set (n_changed, its nnz, its warp loads).
inline double bp_work_per_iter(const work_features_t& f,
                               double n_changed_constraints,
                               double changed_constraints_nnz,
                               double changed_warp_loads)
{
  std::vector<double> x{1.0,
                        f.n_vars,
                        f.n_constraints,
                        f.nnz,
                        f.row_nnz_var,
                        f.col_nnz_var,
                        f.max_row_nnz,
                        n_changed_constraints,
                        changed_constraints_nnz,
                        changed_warp_loads,
                        std::max(0.0, changed_warp_loads - bp_excess_warp_loads_threshold)};
  double w = dot(bp_activity_work_coeffs, x) + dot(bp_update_work_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

// ---- GPU-generic (device-aware) per-iteration predictors ----
// These use the multi-GPU-fitted coefficients (generated/*_device_coeffs.hpp) and the runtime
// device descriptor, so one coefficient set predicts per-iteration seconds on any GPU (no per-GPU
// recalibration). The raw count vector and term order MUST match the calibrator's dump/fit exactly
// (calibrate_main collect_samples + device_model::device_terms).

// PDLP per inner iteration. raw = [n_vars, n_constraints, nnz, row_var, col_var, cons_warp,
// var_warp, total_warp]; hinge on total_warp (threshold k*warp_capacity); serial = max_row_nnz.
inline double pdlp_device_work_per_iter(const work_features_t& f, const gpu_features_t& g)
{
  const double total_warp = f.cons_warp_loads + f.var_warp_loads;
  std::vector<double> raw{f.n_vars,
                          f.n_constraints,
                          f.nnz,
                          f.row_nnz_var,
                          f.col_nnz_var,
                          f.cons_warp_loads,
                          f.var_warp_loads,
                          total_warp};
  const double excess =
    pdlp_device_use_hinge ? std::max(0.0, total_warp - pdlp_device_hinge_k * g.warp_capacity) : 0.0;
  const std::vector<double> x = device_terms(raw, g, excess, f.max_row_nnz);
  const double w              = dot(pdlp_device_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

// FJ per host-step. raw = [n_vars, n_constraints, nnz, row_var, col_var, frontier_work_step]; no
// hinge; serial = max_row_nnz.
inline double fj_device_work_per_step(const work_features_t& f,
                                      const gpu_features_t& g,
                                      double frontier_work_step)
{
  std::vector<double> raw{
    f.n_vars, f.n_constraints, f.nnz, f.row_nnz_var, f.col_nnz_var, frontier_work_step};
  const double excess         = fj_device_use_hinge
                                  ? std::max(0.0, frontier_work_step - fj_device_hinge_k * g.warp_capacity)
                                  : 0.0;
  const std::vector<double> x = device_terms(raw, g, excess, f.max_row_nnz);
  const double w              = dot(fj_device_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

// Bound-presolve (and multi_probe) per iteration, GPU-generic. activity + bounds-update sub-kernels
// share the same raw vector. raw layout MUST match the calibrator's bp dump/fit exactly:
//   [n_vars, n_constraints, nnz, row_var, col_var, max_row_nnz, n_changed, changed_nnz,
//    changed_warp]
// hinge on changed_warp (k*warp_capacity); serial = static max_row_nnz; L2 working set =
// changed_nnz (the data a single iteration actually touches).
inline double bp_device_work_per_iter(const work_features_t& f,
                                      const gpu_features_t& g,
                                      double n_changed_constraints,
                                      double changed_constraints_nnz,
                                      double changed_warp_loads)
{
  std::vector<double> raw{f.n_vars,
                          f.n_constraints,
                          f.nnz,
                          f.row_nnz_var,
                          f.col_nnz_var,
                          f.max_row_nnz,
                          n_changed_constraints,
                          changed_constraints_nnz,
                          changed_warp_loads};
  const double excess = std::max(0.0, changed_warp_loads - bp_device_hinge_k * g.warp_capacity);
  const std::vector<double> x =
    device_terms(raw, g, excess, f.max_row_nnz, changed_constraints_nnz);
  const double w = dot(bp_device_activity_coeffs, x) + dot(bp_device_update_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

// Cost of a single full constraint-activity pass (line-segment get_quality etc.), GPU-generic.
// Activity-only evaluation of the bp device model over the FULL constraint set (changed == all).
inline double activity_work_full_device(const work_features_t& f, const gpu_features_t& g)
{
  std::vector<double> raw{f.n_vars,
                          f.n_constraints,
                          f.nnz,
                          f.row_nnz_var,
                          f.col_nnz_var,
                          f.max_row_nnz,
                          f.n_constraints,
                          f.nnz,
                          f.cons_warp_loads};
  const double excess = std::max(0.0, f.cons_warp_loads - bp_device_hinge_k * g.warp_capacity);
  const std::vector<double> x = device_terms(raw, g, excess, f.max_row_nnz, f.nnz);
  const double w              = dot(bp_device_activity_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

// Bound-repair per move, GPU-generic. raw layout MUST match the calibrator's repair dump/fit:
//   [n_vars, n_constraints, nnz, row_var, col_var, n_candidates, cur_cstr_rowsize,
//   total_warp_loads]
// hinge on total_warp_loads (k*warp_capacity) when enabled; serial = static max_row_nnz; working
// set = nnz (default).
inline double repair_device_work_per_move(const work_features_t& f,
                                          const gpu_features_t& g,
                                          double n_candidates,
                                          double cur_cstr_rowsize)
{
  std::vector<double> raw{f.n_vars,
                          f.n_constraints,
                          f.nnz,
                          f.row_nnz_var,
                          f.col_nnz_var,
                          n_candidates,
                          cur_cstr_rowsize,
                          f.cons_warp_loads};
  const double excess =
    repair_device_use_hinge
      ? std::max(0.0, f.cons_warp_loads - repair_device_hinge_k * g.warp_capacity)
      : 0.0;
  const std::vector<double> x = device_terms(raw, g, excess, f.max_row_nnz);
  const double w              = dot(repair_device_coeffs, x);
  return w > 0.0 ? w : 0.0;
}

}  // namespace cuopt::linear_programming::detail::calib
