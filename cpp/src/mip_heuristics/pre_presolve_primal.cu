/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "pre_presolve_primal.cuh"

#include <branch_and_bound/branch_and_bound.hpp>
#include <cuopt/error.hpp>
#include <dual_simplex/solve.hpp>
#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

#include <omp.h>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace {

constexpr const char* arm_name = "PRE-BNB-ROOT-DIVES";

int parse_integer(std::string_view value, const char* variable_name)
{
  cuopt::cuopt_expects(!value.empty(),
                       cuopt::error_type_t::ValidationError,
                       "%s must be a nonempty integer",
                       variable_name);

  int parsed{};
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  cuopt::cuopt_expects(result.ec == std::errc{} && result.ptr == value.data() + value.size(),
                       cuopt::error_type_t::ValidationError,
                       "%s must be a strict base-10 integer",
                       variable_name);
  return parsed;
}

int checked_config_id(const pre_presolve_config_t& config)
{
  cuopt::cuopt_expects(config.config_id >= 0 && config.config_id < 4,
                       cuopt::error_type_t::ValidationError,
                       "pre-presolve configuration id must be in [0, 4)");
  return config.config_id;
}

template <typename i_t, typename f_t>
simplex::user_problem_t<i_t, f_t> snapshot_original_problem(const problem_t<i_t, f_t>& problem)
{
  simplex::user_problem_t<i_t, f_t> user_problem(problem.handle_ptr);
  problem.get_host_user_problem(user_problem);
  return user_problem;
}

template <typename i_t, typename f_t>
simplex::simplex_solver_settings_t<i_t, f_t> make_settings(
  const mip_solver_settings_t<i_t, f_t>& settings,
  const pre_presolve_config_t& config,
  std::function<bool(std::vector<f_t>&)> peer_incumbent_callback)
{
  simplex::simplex_solver_settings_t<i_t, f_t> bnb_settings;
  bnb_settings.iteration_limit                          = std::numeric_limits<i_t>::max();
  bnb_settings.node_limit                               = std::numeric_limits<i_t>::max();
  bnb_settings.time_limit                               = std::numeric_limits<f_t>::infinity();
  bnb_settings.work_limit                               = std::numeric_limits<f_t>::infinity();
  bnb_settings.branch_and_bound_simplex_iteration_limit = std::numeric_limits<int64_t>::max();
  bnb_settings.num_threads                              = config.task_slots();
  bnb_settings.root_diving_only                         = true;
  bnb_settings.root_diving_config                       = config.config_id;
  if (config.cooperative()) {
    bnb_settings.root_diving_peer_incumbent_callback = std::move(peer_incumbent_callback);
  } else {
    bnb_settings.root_diving_peer_incumbent_callback = nullptr;
  }
  bnb_settings.print_presolve_stats               = false;
  bnb_settings.preserve_advanced_basis_dimensions = true;
  bnb_settings.primal_tol                         = settings.tolerances.absolute_tolerance;
  bnb_settings.dual_tol                           = settings.tolerances.absolute_tolerance;
  bnb_settings.integer_tol                        = settings.tolerances.integrality_tolerance;
  bnb_settings.absolute_mip_gap_tol               = settings.tolerances.absolute_mip_gap;
  bnb_settings.relative_mip_gap_tol               = settings.tolerances.relative_mip_gap;
  bnb_settings.random_seed                        = settings.seed;

  bnb_settings.max_cut_passes                           = 0;
  bnb_settings.mir_cuts                                 = 0;
  bnb_settings.mixed_integer_gomory_cuts                = 0;
  bnb_settings.knapsack_cuts                            = 0;
  bnb_settings.flow_cover_cuts                          = 0;
  bnb_settings.implied_bound_cuts                       = 0;
  bnb_settings.clique_cuts                              = 0;
  bnb_settings.zero_half_cuts                           = 0;
  bnb_settings.strong_chvatal_gomory_cuts               = 0;
  bnb_settings.reduced_cost_strengthening               = 0;
  bnb_settings.reliability_branching                    = 0;
  bnb_settings.strong_branching_simplex_iteration_limit = 0;
  bnb_settings.mip_batch_pdlp_strong_branching          = 0;
  bnb_settings.mip_batch_pdlp_reliability_branching     = 0;
  bnb_settings.symmetry                                 = 0;

  bnb_settings.submip_settings.rins         = 0;
  bnb_settings.submip_settings.rens         = 0;
  bnb_settings.submip_settings.max_level    = 0;
  bnb_settings.submip_settings.enable_cpufj = false;

  bnb_settings.diving_settings                        = settings.diving_params;
  bnb_settings.diving_settings.line_search_diving     = 1;
  bnb_settings.diving_settings.pseudocost_diving      = 1;
  bnb_settings.diving_settings.guided_diving          = 1;
  bnb_settings.diving_settings.coefficient_diving     = 1;
  bnb_settings.diving_settings.farkas_diving          = 1;
  bnb_settings.diving_settings.vector_length_diving   = 1;
  bnb_settings.diving_settings.node_limit             = std::numeric_limits<i_t>::max();
  bnb_settings.diving_settings.iteration_limit_factor = std::numeric_limits<f_t>::infinity();
  bnb_settings.diving_settings.iteration_limit_offset = std::numeric_limits<int64_t>::max();

  bnb_settings.set_log(false);
  return bnb_settings;
}

}  // namespace

int pre_presolve_config_t::diving_workers() const
{
  if (!enabled) { return 0; }
  return checked_config_id(*this) == 3 ? 2 : 3;
}

int pre_presolve_config_t::cpufj_reserved_threads() const
{
  if (!enabled) { return 0; }
  return checked_config_id(*this) == 3 ? 7 : 8;
}

bool pre_presolve_config_t::cooperative() const
{
  if (!enabled) { return false; }
  return checked_config_id(*this) != 0;
}

const char* pre_presolve_config_t::name() const
{
  if (!enabled) { return "disabled"; }
  switch (checked_config_id(*this)) {
    case 0: return "independent-w3-all";
    case 1: return "cooperative-w3-all";
    case 2: return "refinement-w3-restricted";
    case 3: return "cooperative-w2-all";
  }
  return "invalid";
}

pre_presolve_config_t resolve_pre_presolve_config(std::optional<std::string_view> config_id,
                                                  std::optional<std::string_view> max_config)
{
  if (!config_id.has_value() && !max_config.has_value()) { return {}; }

  cuopt::cuopt_expects(config_id.has_value() && max_config.has_value(),
                       cuopt::error_type_t::ValidationError,
                       "CUOPT_CONFIG_ID and CUOPT_MAX_CONFIG must be set together");

  const int raw_id     = parse_integer(*config_id, "CUOPT_CONFIG_ID");
  const int parsed_max = parse_integer(*max_config, "CUOPT_MAX_CONFIG");
  cuopt::cuopt_expects(
    parsed_max > 0, cuopt::error_type_t::ValidationError, "CUOPT_MAX_CONFIG must be positive");
  cuopt::cuopt_expects(parsed_max % 4 == 0,
                       cuopt::error_type_t::ValidationError,
                       "CUOPT_MAX_CONFIG must be a multiple of four");
  cuopt::cuopt_expects(raw_id >= 0 && raw_id < parsed_max,
                       cuopt::error_type_t::ValidationError,
                       "CUOPT_CONFIG_ID must be in [0, CUOPT_MAX_CONFIG)");

  pre_presolve_config_t config;
  config.enabled    = true;
  config.raw_id     = raw_id;
  config.max_config = parsed_max;
  config.repeat     = raw_id / 4;
  config.config_id  = raw_id % 4;
  return config;
}

pre_presolve_config_t resolve_pre_presolve_config_from_env()
{
  const char* config_id  = std::getenv("CUOPT_CONFIG_ID");
  const char* max_config = std::getenv("CUOPT_MAX_CONFIG");
  return resolve_pre_presolve_config(
    config_id == nullptr ? std::nullopt
                         : std::optional<std::string_view>{std::string_view{config_id}},
    max_config == nullptr ? std::nullopt
                          : std::optional<std::string_view>{std::string_view{max_config}});
}

template <typename i_t, typename f_t>
struct pre_presolve_primal_t<i_t, f_t>::work_t {
  work_t(const problem_t<i_t, f_t>& problem,
         const mip_solver_settings_t<i_t, f_t>& settings,
         const pre_presolve_config_t& config,
         early_incumbent_callback_t<f_t> incumbent_callback,
         std::function<bool(std::vector<f_t>&)> peer_incumbent_callback)
    : user_problem(snapshot_original_problem(problem)),
      bnb_settings(make_settings(settings, config, std::move(peer_incumbent_callback)))
  {
    const f_t objective_scale    = user_problem.obj_scale;
    const f_t objective_constant = user_problem.obj_constant;
    bnb_settings.solution_callback =
      [incumbent_callback = std::move(incumbent_callback), objective_scale, objective_constant](
        std::vector<f_t>& assignment, f_t solver_obj) {
        if (!incumbent_callback) { return; }
        const f_t user_obj = objective_scale * (solver_obj + objective_constant);
        try {
          incumbent_callback(solver_obj, user_obj, assignment, arm_name);
        } catch (const std::exception& e) {
          CUOPT_LOG_ERROR("Pre-presolve root-diving incumbent callback failed: %s", e.what());
        } catch (...) {
          CUOPT_LOG_ERROR("Pre-presolve root-diving incumbent callback failed");
        }
      };
  }

  void run(task_state_t& task_state)
  {
    if (task_state.cancel_requested.load(std::memory_order_acquire) != 0) { return; }

    probing_implied_bound_t<i_t, f_t> probing_implied_bound(user_problem.num_cols);
    branch_and_bound_t<i_t, f_t> branch_and_bound(
      user_problem, bnb_settings, tic(), probing_implied_bound);
    branch_and_bound.set_concurrent_lp_root_solve(false);
    branch_and_bound.set_external_halt(
      &task_state.cancel_requested, &task_state.root_halt, &task_state.node_halt);

    // Share an incumbent that arrived before the auxiliary B&B existed. The callback and the
    // copy held by B&B share their generation cursor, so each mailbox generation is consumed once.
    if (bnb_settings.root_diving_peer_incumbent_callback) {
      std::vector<f_t> peer_incumbent;
      if (bnb_settings.root_diving_peer_incumbent_callback(peer_incumbent) &&
          !peer_incumbent.empty()) {
        branch_and_bound.set_solution_from_heuristics(peer_incumbent,
                                                      heuristics_origin_t::HEURISTICS);
      }
    }

    // A zero simplex-iteration limit selects an estimate path rather than disabling initial root
    // strong branching. Neutral zero-count pseudocosts skip that root block and learn at nodes.
    pseudo_costs_t<i_t, f_t> neutral_pseudocost(user_problem.num_cols, bnb_settings);
    std::vector<i_t> identity_reduced_to_original(user_problem.num_cols);
    std::iota(identity_reduced_to_original.begin(), identity_reduced_to_original.end(), i_t{0});
    branch_and_bound.set_initial_pseudocost(neutral_pseudocost, identity_reduced_to_original);

    simplex::mip_solution_t<i_t, f_t> solution(user_problem.num_cols);
    branch_and_bound.solve(solution);
  }

  simplex::user_problem_t<i_t, f_t> user_problem;
  simplex::simplex_solver_settings_t<i_t, f_t> bnb_settings;
};

template <typename i_t, typename f_t>
pre_presolve_primal_t<i_t, f_t>::pre_presolve_primal_t(
  const problem_t<i_t, f_t>& problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  pre_presolve_config_t config,
  early_incumbent_callback_t<f_t> aux_callback)
  : config_(config),
    task_state_(std::make_unique<task_state_t>()),
    objective_sense_(problem.maximize ? f_t{-1} : f_t{1})
{
  if (!config_.enabled) { return; }

  const bool locally_eligible =
    is_pre_presolve_primal_eligible(problem.n_constraints,
                                    problem.n_variables,
                                    problem.nnz,
                                    !problem.Q_values.empty(),
                                    false,
                                    problem.n_integer_vars > 0,
                                    settings.presolver != presolver_t::None,
                                    settings.determinism_mode == CUOPT_MODE_OPPORTUNISTIC,
                                    omp_get_num_threads());
  if (!locally_eligible) { return; }

  std::function<bool(std::vector<f_t>&)> peer_incumbent_callback;
  if (config_.cooperative()) {
    auto last_generation = std::make_shared<uint64_t>(0);
    peer_incumbent_callback =
      [this, last_generation = std::move(last_generation)](std::vector<f_t>& assignment) {
        return poll_external_incumbent(*last_generation, assignment);
      };
  }

  work_ = std::make_unique<work_t>(
    problem, settings, config_, std::move(aux_callback), std::move(peer_incumbent_callback));
}

template <typename i_t, typename f_t>
pre_presolve_primal_t<i_t, f_t>::~pre_presolve_primal_t()
{
  stop();
}

template <typename i_t, typename f_t>
bool pre_presolve_primal_t<i_t, f_t>::start()
{
  if (started_) { return true; }
  if (!work_ || task_state_->cancel_requested.load(std::memory_order_acquire) != 0) {
    return false;
  }

  CUOPT_LOG_INFO(
    "PRE_PRESOLVE_START raw_id=%d max_config=%d repeat=%d config_id=%d name=%s workers=%d "
    "cooperative=%d",
    config_.raw_id,
    config_.max_config,
    config_.repeat,
    config_.config_id,
    config_.name(),
    config_.diving_workers(),
    static_cast<int>(config_.cooperative()));

  started_         = true;
  auto* task_state = task_state_.get();
  auto* work       = work_.get();

#pragma omp task default(none) firstprivate(task_state, work) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) depend(out : *task_state)
  {
    try {
      work->run(*task_state);
    } catch (...) {
      task_state->exception = std::current_exception();
    }
  }
  return true;
}

template <typename i_t, typename f_t>
void pre_presolve_primal_t<i_t, f_t>::stop() noexcept
{
  if (!started_) { return; }

  task_state_->cancel_requested.store(1, std::memory_order_release);
  task_state_->root_halt.store(1, std::memory_order_release);
  task_state_->node_halt.store(1, std::memory_order_release);
  auto* task_state = task_state_.get();
#pragma omp taskwait depend(in : *task_state)
  started_ = false;

  if (task_state_->exception) {
    try {
      std::rethrow_exception(task_state_->exception);
    } catch (const std::exception& e) {
      CUOPT_LOG_ERROR("Pre-presolve root-diving task failed: %s", e.what());
    } catch (...) {
      CUOPT_LOG_ERROR("Pre-presolve root-diving task failed with an unknown exception");
    }
    task_state_->exception = nullptr;
  }
}

template <typename i_t, typename f_t>
void pre_presolve_primal_t<i_t, f_t>::offer_external_incumbent(f_t user_obj,
                                                               const std::vector<f_t>& assignment)
{
  if (!config_.cooperative() || !work_ || !std::isfinite(user_obj) ||
      assignment.size() != static_cast<std::size_t>(work_->user_problem.num_cols)) {
    return;
  }

  const f_t score = objective_sense_ * user_obj;
  if (!std::isfinite(score)) { return; }

  std::lock_guard<std::mutex> lock(mailbox_mutex_);
  if (score >= mailbox_best_score_) { return; }
  mailbox_best_score_ = score;
  mailbox_assignment_ = assignment;
  ++mailbox_generation_;
}

template <typename i_t, typename f_t>
bool pre_presolve_primal_t<i_t, f_t>::poll_external_incumbent(uint64_t& last_generation,
                                                              std::vector<f_t>& assignment)
{
  std::lock_guard<std::mutex> lock(mailbox_mutex_);
  if (mailbox_generation_ == 0 || mailbox_generation_ <= last_generation) { return false; }
  assignment      = mailbox_assignment_;
  last_generation = mailbox_generation_;
  return true;
}

#if MIP_INSTANTIATE_FLOAT
template class pre_presolve_primal_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class pre_presolve_primal_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
