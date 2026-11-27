/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Values of mip_hyper_params_t::presolve_gprs.
enum class gprs_presolve_mode_t : int {
  OFF                    = 0,
  BEFORE_THIRD_PARTY     = 1,  // the third-party presolve gets the problem gprs reduced
  INSTEAD_OF_THIRD_PARTY = 2
};

enum class gprs_presolve_status_t {
  // The original problem stays: it has something gprs cannot carry, or gprs stopped on its limit.
  NOT_RUN,
  UNCHANGED,
  REDUCED,
  REDUCED_TO_EMPTY,
  INFEASIBLE,
  UNBOUNDED_OR_INFEASIBLE
};

// The gprs host presolve (gpu_presolve/) on a host MIP. gprs has no dual postsolve.
template <typename i_t, typename f_t>
class gprs_presolve_t {
 public:
  gprs_presolve_t();
  ~gprs_presolve_t();
  gprs_presolve_t(const gprs_presolve_t&)            = delete;
  gprs_presolve_t& operator=(const gprs_presolve_t&) = delete;

  // reduced_problem is written for REDUCED and REDUCED_TO_EMPTY; it keeps the sense, the
  // objective scaling factor and the name of the problem and has no row or variable names.
  // With primal_only every feasible point of the problem crushes to a feasible point of the
  // reduced problem with the same objective value.
  gprs_presolve_status_t apply(io::mps_data_model_t<i_t, f_t> const& problem,
                               bool primal_only,
                               double time_limit,
                               io::mps_data_model_t<i_t, f_t>& reduced_problem);

  // The members below need a status of UNCHANGED, REDUCED or REDUCED_TO_EMPTY. All of them can be
  // called from several threads at once.

  // Point of the reduced problem -> point of the original problem, in place.
  void postsolve(std::vector<f_t>& primal_solution) const;

  // Point of the original problem -> point of the reduced problem.
  void crush(const std::vector<f_t>& original_primal, std::vector<f_t>& reduced_primal) const;

  bool is_original_primal_solution_feasible(const std::vector<f_t>& original_primal,
                                            f_t absolute_tolerance,
                                            f_t relative_tolerance,
                                            f_t integrality_tolerance) const;

  // Rewrites the variable maps of a presolve that ran on the reduced problem so that they refer
  // to the original problem.
  void compose_maps(std::vector<i_t>& reduced_to_original,
                    std::vector<i_t>& original_to_reduced) const;

  const std::vector<i_t>& get_reduced_to_original_map() const;
  const std::vector<i_t>& get_original_to_reduced_map() const;

  // Seconds spent in apply.
  double get_presolve_time() const;

 private:
  struct impl_t;
  std::unique_ptr<impl_t> impl_;
};

}  // namespace cuopt::mathematical_optimization::mip
