/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/diversity/lns/lns_feasibility_cpu.cuh>
#include <mip_heuristics/diversity/population.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver_context.cuh>

#include <utilities/logger.hpp>

#include <thrust/fill.h>

#include <memory>
#include <vector>

namespace cuopt::linear_programming::detail {

// Background CPU LNS worker, started alongside scratch CPUFJ. Feasible finds are fed into the
// population; B&B/heuristics can push better incumbents and LP references with offer_*().
template <typename i_t, typename f_t>
class cpu_lns_thread_t {
 public:
  explicit cpu_lns_thread_t(mip_solver_context_t<i_t, f_t>& context_) : context(context_) {}

  ~cpu_lns_thread_t() { stop(); }

  void start(population_t<i_t, f_t>& population, f_t time_limit, f_t solve_elapsed_at_start)
  {
    if (started_ || lns_) { return; }
    if (omp_get_num_threads() < CUOPT_MIP_LNS_REQUIRED_THREAD_COUNT) { return; }
    if (context.settings.determinism_mode == CUOPT_MODE_DETERMINISTIC) { return; }
    if (context.problem_ptr == nullptr || context.problem_ptr->n_integer_vars <= 0) { return; }
    if (time_limit <= f_t{0}) { return; }

    context.problem_ptr->handle_ptr->sync_stream();
    problem_copy_ = std::make_unique<problem_t<i_t, f_t>>(*context.problem_ptr, &lns_handle_);
    solution_     = std::make_unique<solution_t<i_t, f_t>>(*problem_copy_);
    thrust::fill(lns_handle_.get_thrust_policy(),
                 solution_->assignment.begin(),
                 solution_->assignment.end(),
                 f_t{0});
    solution_->clamp_within_bounds();

    lns_ = std::make_unique<lns_feasibility_cpu_t<i_t, f_t>>(context, problem_copy_.get());
    lns_->feasible_callback = [&population](f_t objective, const std::vector<f_t>& assignment) {
      population.add_external_solution(assignment, objective, solution_origin_t::LNS);
    };

    time_limit_             = time_limit;
    solve_elapsed_at_start_ = solve_elapsed_at_start;
    started_                = true;

    CUOPT_LOG_DEBUG("Launching background CPU LNS task");
    auto* runner = this;
#pragma omp task firstprivate(runner) priority(CUOPT_DEFAULT_TASK_PRIORITY) \
  depend(out : *runner->lns_) default(none)
    {
      RAFT_CUDA_TRY(cudaSetDevice(runner->context.handle_ptr->get_device()));
      runner->lns_->run(
        *runner->solution_, runner->time_limit_, runner->solve_elapsed_at_start_);
    }
  }

  void stop()
  {
    if (!started_ || !lns_) { return; }
    lns_->halted.store(true, std::memory_order_relaxed);
#pragma omp taskwait depend(in : *lns_)
    CUOPT_LOG_DEBUG("Background CPU LNS stopped");
    lns_.reset();
    solution_.reset();
    problem_copy_.reset();
    started_ = false;
  }

  void offer_incumbent(const std::vector<f_t>& assignment, f_t objective)
  {
    if (lns_) { lns_->offer_incumbent(assignment, objective); }
  }

  void offer_lp_reference(const std::vector<f_t>& assignment)
  {
    if (lns_) { lns_->offer_lp_reference(assignment); }
  }

  bool running() const { return started_ && lns_ != nullptr; }

 private:
  mip_solver_context_t<i_t, f_t>& context;
  raft::handle_t lns_handle_;
  std::unique_ptr<problem_t<i_t, f_t>> problem_copy_;
  std::unique_ptr<solution_t<i_t, f_t>> solution_;
  std::unique_ptr<lns_feasibility_cpu_t<i_t, f_t>> lns_;
  f_t time_limit_{0};
  f_t solve_elapsed_at_start_{0};
  bool started_{false};
};

}  // namespace cuopt::linear_programming::detail
