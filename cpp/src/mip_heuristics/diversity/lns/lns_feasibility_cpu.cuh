/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Standalone CPU large-neighborhood search for MIP. The hot loop is incremental: an attempt only
// visits variables in the ruin set and their incident constraints. Full matrix scans are limited to
// initialization, rare Feasibility-Jump polishes, and numerical refreshes.

#include <mip_heuristics/feasibility_jump/cpu_fj_thread.cuh>
#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>
#include <mip_heuristics/logger.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <mip_heuristics/utils.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>
#include <utilities/timer.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class lns_feasibility_cpu_t {
 public:
  using feasible_callback_t = std::function<void(f_t objective, const std::vector<f_t>& assignment)>;

  explicit lns_feasibility_cpu_t(mip_solver_context_t<i_t, f_t>& context_,
                                 problem_t<i_t, f_t>* problem_override = nullptr)
    : context(context_),
      problem_ptr(problem_override != nullptr ? problem_override : context_.problem_ptr),
      rng(context_.settings.seed >= 0
            ? static_cast<typename std::mt19937::result_type>(context_.settings.seed)
            : static_cast<typename std::mt19937::result_type>(cuopt::seed_generator::get_seed()))
  {
  }

  // Background-thread control and simple cross-heuristic communication.
  std::atomic<bool> halted{false};
  feasible_callback_t feasible_callback;

  void offer_incumbent(const std::vector<f_t>& assignment, f_t objective)
  {
    if (assignment.size() != static_cast<size_t>(problem_ptr->n_variables)) { return; }
    std::lock_guard<std::mutex> lock(inject_mutex);
    if (has_injected_incumbent && objective >= injected_objective) { return; }
    injected_assignment     = assignment;
    injected_objective      = objective;
    has_injected_incumbent  = true;
  }

  void offer_lp_reference(const std::vector<f_t>& assignment)
  {
    if (assignment.size() != static_cast<size_t>(problem_ptr->n_variables)) { return; }
    std::lock_guard<std::mutex> lock(inject_mutex);
    injected_lp_reference = assignment;
    has_injected_lp       = true;
  }

  bool run(solution_t<i_t, f_t>& solution, f_t time_limit, f_t solve_elapsed_at_start = f_t{0})
  {
    if (time_limit <= f_t{0}) { return solution.get_feasible(); }
    lns_timer            = timer_t(time_limit);
    solve_elapsed_offset = solve_elapsed_at_start;
    diagnostics_enabled  = diagnostics_requested();
    configure_ablation();
    const auto setup_start = clock_t::now();
    init_host_data();

    std::vector<f_t> x = solution.get_host_assignment();
    lp_reference       = x;
    for (i_t v = 0; v < n_vars; ++v) {
      if (var_class[v] == 0) {
        x[v] = std::min(std::max(x[v], var_lb[v]), var_ub[v]);
      } else {
        const f_t integral_lb = std::ceil(var_lb[v] - tols.integrality_tolerance);
        const f_t integral_ub = std::floor(var_ub[v] + tols.integrality_tolerance);
        x[v]                  = std::min(std::max(std::round(x[v]), integral_lb), integral_ub);
      }
    }
    if (!ablation.use_fractional_lp_reference) { lp_reference = x; }
    current_lhs     = compute_lhs(x);
    state_t current = state_from_lhs(current_lhs, x);
    initialize_dynamic_state(current_lhs);
    diagnostics.setup_s = seconds_since(setup_start);

    CUOPT_LOG_INFO(
      "CPU LNS start: time_limit %.2fs, vars %d, integer vars %d, constraints %d, nnz %d, "
      "unsat %d, excess %.6e",
      time_limit,
      n_vars,
      n_int,
      n_cons,
      static_cast<i_t>(coefficients.size()),
      current.unsat,
      current.total_excess);
    CUOPT_LOG_INFO("CPU_LNS_ABLATION variant=%s", ablation.name.c_str());

    std::vector<f_t> best = x;
    state_t best_state    = current;
    bool have_feasible    = current.unsat == 0;
    if (have_feasible) {
      record_first_feasible("lp_seed", 0, current);
      report_feasible(best, best_state.objective);
    }

    // Full seed propagation is very effective on moderate, tightly structured models, but the
    // old unbounded implementation was quadratic and overran large models by minutes. Keep the
    // generic constructor only when its conservative work estimate is small, and check its own
    // deadline throughout.
    const uint64_t seed_work_estimate =
      static_cast<uint64_t>(n_int) * static_cast<uint64_t>(n_vars);
    if (!have_feasible && ablation.use_seed_bp && seed_work_estimate <= MAX_SEED_PROPAGATION_WORK &&
        n_vars <= MAX_SEED_PROPAGATION_VARIABLES && !should_stop()) {
      const auto seed_start = clock_t::now();
      const f_t seed_budget =
        std::min<f_t>(f_t{2}, std::max<f_t>(f_t{0.1}, lns_timer.remaining_time() * f_t{0.2}));
      // Natural order produced the fastest deterministic constructions, while the shuffled order
      // found an additional early solution. LP-confidence and degree order added no unique wins in
      // the 30-instance ablation and can starve the shuffled constructor, so retain them as
      // diagnostic single-strategy variants rather than default portfolio members.
      static constexpr std::array<size_t, 2> default_seed_strategies = {0, 3};
      const size_t strategy_count =
        ablation.seed_strategy >= 0 ? 1 : default_seed_strategies.size();
      for (size_t strategy_rank = 0; strategy_rank < strategy_count && !have_feasible;
           ++strategy_rank) {
        const size_t strategy = ablation.seed_strategy >= 0
                                  ? static_cast<size_t>(ablation.seed_strategy)
                                  : default_seed_strategies[strategy_rank];
        const f_t used        = static_cast<f_t>(seconds_since(seed_start));
        if (used >= seed_budget || should_stop()) { break; }
        std::vector<f_t> seed_candidate = x;
        ++diagnostics.seed_bp_attempted;
        if (seed_constraint_propagation(seed_candidate, seed_budget - used, strategy)) {
          ++diagnostics.seed_bp_completed;
          state_t seed_state = state_from_lhs(compute_lhs(seed_candidate), seed_candidate);
          if (better_for_best(seed_state, best_state)) {
            x           = std::move(seed_candidate);
            current_lhs = compute_lhs(x);
            current     = state_from_lhs(current_lhs, x);
            rebuild_dynamic_state(current_lhs);
            best       = x;
            best_state = current;
            if (current.unsat == 0) {
              have_feasible = true;
              record_first_feasible("seed_bp", 0, current);
              report_feasible(best, best_state.objective);
            }
          }
        }
      }
      diagnostics.seed_bp_s = seconds_since(seed_start);
    }

    // A short CPUFJ burst is the generic starting constructor. Unlike the old all-variable seed
    // propagation, its work is deadline bounded and does not become quadratic in the variable
    // count.
    if (!have_feasible && ablation.use_initial_fj && !should_stop()) {
      std::vector<f_t> fj_candidate = x;
      const f_t budget =
        std::min<f_t>(f_t{4.0}, std::max<f_t>(f_t{0.25}, lns_timer.remaining_time() * f_t{0.35}));
      if (run_fj_burst(fj_candidate, budget, solution, false)) {
        x           = std::move(fj_candidate);
        current_lhs = compute_lhs(x);
        current     = state_from_lhs(current_lhs, x);
        rebuild_dynamic_state(current_lhs);
        if (better_for_best(current, best_state)) {
          best       = x;
          best_state = current;
        }
        if (current.unsat == 0) {
          have_feasible = true;
          best          = x;
          best_state    = current;
          record_first_feasible("initial_fj", diagnostics.iterations, current);
          report_feasible(best, best_state.objective);
        }
      }
    }

    f_t temperature = initial_temperature(current.objective);
    f_t next_log_s  = f_t{2};
    size_t failure_streak{0};
    size_t attempts_since_refresh{0};
    size_t attempts_since_fj{0};
    size_t attempts_since_best{0};
    f_t next_fj_time = f_t{8};

    for (size_t attempt = 0; attempt < MAX_ATTEMPTS && !should_stop(); ++attempt) {
      ++diagnostics.iterations;
      ++weight_epoch;
      ++attempts_since_refresh;
      ++attempts_since_fj;
      ++attempts_since_best;

      if (consume_external_messages(
            x, current_lhs, current, best, best_state, have_feasible, temperature)) {
        failure_streak      = 0;
        attempts_since_best = 0;
      }

      const bool improvement_phase = have_feasible;
      const bool intensify         = !improvement_phase && ablation.ruin_enabled[0] &&
                             best_state.unsat > 0 && best_state.unsat <= 8 &&
                             attempts_since_best >= 4096;
      if (intensify) {
        x           = best;
        current_lhs = compute_lhs(x);
        current     = state_from_lhs(current_lhs, x);
        rebuild_dynamic_state(current_lhs);
        attempts_since_best = 0;
        ++diagnostics.intensifications;
      }
      const size_t operator_pair =
        intensify ? operator_pair_index(0, 2) : select_operator_pair(diagnostics.iterations);
      const size_t ruin_arm   = operator_pair / N_REPAIR_ARMS;
      const size_t repair_arm = operator_pair % N_REPAIR_ARMS;
      const i_t target_ruin   = intensify ? intensification_ruin_size(current_lhs)
                                : improvement_phase
                                  ? std::min<i_t>(8, adaptive_ruin_size(failure_streak, attempt))
                                  : adaptive_ruin_size(failure_streak, attempt);

      const auto select_start = clock_t::now();
      const i_t selected =
        intensify ? select_violated_union_ruin(target_ruin, x, current_lhs)
                  : select_ruin_set(ruin_arm, target_ruin, x, current_lhs, improvement_phase);
      diagnostics.select_s += seconds_since(select_start);
      ++diagnostics.ruin_attempts[ruin_arm];
      ++diagnostics.repair_attempts[repair_arm];

      if (selected == 0) {
        ++diagnostics.empty_ruins;
        if (!intensify) { update_arm(operator_pair_arms[operator_pair], REWARD_REJECT); }
        update_arm(ruin_arms[ruin_arm], REWARD_REJECT);
        update_arm(repair_arms[repair_arm], REWARD_REJECT);
        continue;
      }

      prepare_incremental_attempt(x, current_lhs);
      const state_t previous  = current;
      const auto repair_start = clock_t::now();
      const bool changed      = intensify
                                  ? repair_intensification(x, current_lhs)
                                  : repair_incrementally(repair_arm, x, current_lhs, improvement_phase);
      diagnostics.repair_s += seconds_since(repair_start);
      ++diagnostics.productive_attempts;

      candidate_metrics_t candidate = evaluate_candidate(previous, x, current_lhs);
      const bool improving = better_for_phase(candidate.state, previous, improvement_phase);
      bool accepted =
        changed && accept_candidate(candidate, previous, improvement_phase, temperature);

      bool best_updated = false;
      if (accepted) {
        commit_candidate(current_lhs);
        current = candidate.state;
        ++diagnostics.accepted;
        ++diagnostics.ruin_accepts[ruin_arm];
        ++diagnostics.repair_accepts[repair_arm];
        failure_streak = 0;
        if (improvement_phase &&
            candidate.state.objective > previous.objective + OBJECTIVE_EPSILON) {
          ++diagnostics.sa_uphill_accepts;
          sa_excursion_active = true;
        }

        if (!have_feasible && current.unsat == 0) {
          // Establish an exact objective at the phase boundary. Accumulated deltas can lose
          // accuracy after very large moves on models with unbounded integer domains.
          current.objective = compute_objective(x);
          have_feasible     = true;
          record_first_feasible(ruin_arm_name(ruin_arm), diagnostics.iterations, current);
        }
        if (current.unsat == 0 && better_for_best(current, best_state)) {
          // Validate a claimed incumbent in O(n) before saving it. This prevents numerical
          // cancellation during an SA excursion from becoming a false best without imposing an
          // O(nnz) full-state refresh on every iteration.
          current.objective = compute_objective(x);
        }
        if (better_for_best(current, best_state)) {
          best         = x;
          best_state   = current;
          best_updated = true;
          ++diagnostics.best_updates;
          ++diagnostics.ruin_best_updates[ruin_arm];
          ++diagnostics.repair_best_updates[repair_arm];
          attempts_since_best = 0;
          if (current.unsat == 0) { report_feasible(best, best_state.objective); }
          if (improvement_phase) {
            ++diagnostics.objective_best_updates;
            if (sa_excursion_active) {
              ++diagnostics.sa_escape_best_updates;
              sa_excursion_active = false;
            }
            log_objective("best_update", diagnostics.iterations, current, best_state, temperature);
          }
        }
      } else {
        restore_incremental_attempt(x, current_lhs);
        current = previous;
        ++failure_streak;
      }

      const f_t reward = best_updated            ? REWARD_BEST
                         : accepted && improving ? REWARD_BETTER
                         : accepted              ? (improvement_phase ? REWARD_ACCEPT : f_t{0.25})
                                                 : REWARD_REJECT;
      if (!intensify) { update_arm(operator_pair_arms[operator_pair], reward); }
      update_arm(ruin_arms[ruin_arm], reward);
      update_arm(repair_arms[repair_arm], reward);
      clear_ruin_set();

      if (improvement_phase) {
        temperature = std::max(MIN_TEMPERATURE, temperature * TEMPERATURE_COOLING);
        if (failure_streak > 512) {
          temperature =
            std::max(temperature, initial_temperature(best_state.objective) * f_t{0.25});
          failure_streak = 0;
          ++diagnostics.sa_reheats;
        }
      }

      // Correct accumulated LHS roundoff, but keep this outside the hot path.
      if (ablation.full_refresh || attempts_since_refresh >= LHS_REFRESH_ATTEMPTS) {
        attempts_since_refresh = 0;
        current_lhs            = compute_lhs(x);
        current                = state_from_lhs(current_lhs, x);
        rebuild_dynamic_state(current_lhs);
        ++diagnostics.lhs_refreshes;
      }

      // One near-feasible FJ polish is useful as a complementary repair operator. Reinitializing
      // CPUFJ copies the whole problem, so it is deliberately time- and call-gated.
      if (!have_feasible && ablation.use_periodic_fj && best_state.unsat > 0 &&
          best_state.unsat <= FJ_UNSAT_THRESHOLD && attempts_since_fj >= FJ_MIN_ATTEMPTS &&
          diagnostics.fj_calls < MAX_FJ_CALLS && lns_timer.elapsed_time() >= next_fj_time &&
          lns_timer.remaining_time() > f_t{0.15}) {
        attempts_since_fj = 0;
        next_fj_time += f_t{4};
        std::vector<f_t> fj_candidate = best;
        const f_t budget              = std::min<f_t>(f_t{1.25}, lns_timer.remaining_time());
        if (run_fj_burst(fj_candidate, budget, solution, false)) {
          state_t fj_state = state_from_lhs(compute_lhs(fj_candidate), fj_candidate);
          if (better_for_best(fj_state, best_state)) {
            x           = fj_candidate;
            current_lhs = compute_lhs(x);
            current     = state_from_lhs(current_lhs, x);
            rebuild_dynamic_state(current_lhs);
            best       = x;
            best_state = current;
            if (current.unsat == 0) {
              have_feasible = true;
              record_first_feasible("periodic_fj", diagnostics.iterations, current);
              report_feasible(best, best_state.objective);
            }
          }
        }
      }

      const f_t elapsed = lns_timer.elapsed_time();
      if (elapsed >= next_log_s) {
        log_progress(current, best_state, elapsed, temperature);
        next_log_s += f_t{2};
      }
    }

    // Always return the best state rather than the final SA/GLS state.
    x                     = std::move(best);
    current_lhs           = compute_lhs(x);
    best_state            = state_from_lhs(current_lhs, x);
    diagnostics.elapsed_s = lns_timer.elapsed_time();
    if (best_state.unsat == 0) {
      log_objective("final", diagnostics.iterations, best_state, best_state, temperature, true);
    }
    log_diagnostics(best_state);
    log_violation_diagnostics(current_lhs);

    CUOPT_LOG_INFO(
      "CPU LNS finished: unsat %d, excess %.6e, accepted_any %d, %lu iterations in %.2fs "
      "(%.1f it/s)",
      best_state.unsat,
      best_state.total_excess,
      diagnostics.accepted != 0,
      diagnostics.iterations,
      diagnostics.elapsed_s,
      wall_iterations_per_second());

    if (best_state.unsat == 0) { return finalize(solution, x); }
    solution.copy_new_assignment(x);
    solution.compute_feasibility();
    return diagnostics.accepted != 0;
  }

 private:
  using clock_t = std::chrono::steady_clock;

  static constexpr size_t MAX_ATTEMPTS                = 10000000;
  static constexpr i_t MIN_RUIN_SIZE                  = 4;
  static constexpr i_t MAX_RUIN_SIZE                  = 32;
  static constexpr size_t N_RUIN_ARMS                 = 3;
  static constexpr size_t N_REPAIR_ARMS               = 3;
  static constexpr size_t N_OPERATOR_PAIRS            = N_RUIN_ARMS * N_REPAIR_ARMS;
  static constexpr size_t LHS_REFRESH_ATTEMPTS        = 4096;
  static constexpr size_t FJ_MIN_ATTEMPTS             = 8192;
  static constexpr i_t FJ_UNSAT_THRESHOLD             = 64;
  static constexpr uint64_t MAX_SEED_PROPAGATION_WORK = 200000000;
  static constexpr i_t MAX_SEED_PROPAGATION_VARIABLES = 20000;
  static constexpr uint64_t MAX_FJ_CALLS              = 2;
  static constexpr i_t MAX_INTENSIFICATION_RUIN       = 4096;
  static constexpr f_t WEIGHT_DECAY                   = f_t{0.995};
  static constexpr f_t MAX_TARDINESS                  = f_t{200};
  static constexpr f_t TEMPERATURE_COOLING            = f_t{0.9995};
  static constexpr f_t MIN_TEMPERATURE                = f_t{1e-6};
  static constexpr f_t REWARD_BEST                    = f_t{8};
  static constexpr f_t REWARD_BETTER                  = f_t{4};
  static constexpr f_t REWARD_ACCEPT                  = f_t{2};
  static constexpr f_t REWARD_REJECT                  = f_t{0};

  struct state_t {
    i_t unsat{0};
    f_t total_excess{0};
    f_t normalized_excess{0};
    f_t objective{0};
  };

  struct candidate_metrics_t {
    state_t state;
    f_t weighted_count_delta{0};
    f_t weighted_excess_delta{0};
  };

  struct arm_stats_t {
    uint64_t pulls{0};
    f_t reward_sum{0};
  };

  struct ablation_config_t {
    std::string name{"none"};
    bool use_fractional_lp_reference{true};
    bool use_seed_bp{true};
    int seed_strategy{-1};
    bool use_initial_fj{true};
    bool use_periodic_fj{true};
    bool use_bandit{true};
    bool use_adaptive_ruin_size{true};
    bool use_similarity{true};
    bool use_cosine{true};
    bool use_type_filter{true};
    bool use_saturation{true};
    bool use_tardiness{true};
    bool use_weight_decay{true};
    bool use_normalized_excess{true};
    bool use_projection{false};
    bool use_sa{true};
    bool full_refresh{false};
    std::array<bool, N_RUIN_ARMS> ruin_enabled{true, true, true};
    std::array<bool, N_REPAIR_ARMS> repair_enabled{false, true, true};
  };

  struct diagnostics_t {
    uint64_t iterations{0};
    uint64_t productive_attempts{0};
    uint64_t empty_ruins{0};
    uint64_t accepted{0};
    uint64_t best_updates{0};
    uint64_t propagation_conflicts{0};
    uint64_t lhs_refreshes{0};
    uint64_t fj_calls{0};
    uint64_t fj_iterations{0};
    uint64_t sa_reheats{0};
    uint64_t sa_uphill_accepts{0};
    uint64_t sa_escape_best_updates{0};
    uint64_t objective_best_updates{0};
    uint64_t seed_bp_attempted{0};
    uint64_t seed_bp_completed{0};
    uint64_t intensifications{0};
    std::array<uint64_t, N_RUIN_ARMS> ruin_attempts{};
    std::array<uint64_t, N_RUIN_ARMS> ruin_accepts{};
    std::array<uint64_t, N_RUIN_ARMS> ruin_best_updates{};
    std::array<uint64_t, N_REPAIR_ARMS> repair_attempts{};
    std::array<uint64_t, N_REPAIR_ARMS> repair_accepts{};
    std::array<uint64_t, N_REPAIR_ARMS> repair_best_updates{};
    uint64_t projection_moves{0};
    double setup_s{0};
    double select_s{0};
    double repair_s{0};
    double evaluate_s{0};
    double fj_s{0};
    double seed_bp_s{0};
    double elapsed_s{0};
    double first_feasible_lns_s{-1};
    double first_feasible_solve_s{-1};
    f_t first_feasible_objective{std::numeric_limits<f_t>::quiet_NaN()};
  };

  static double seconds_since(const clock_t::time_point& start)
  {
    return std::chrono::duration<double>(clock_t::now() - start).count();
  }

  static bool diagnostics_requested()
  {
    const char* value = std::getenv("CUOPT_CPU_LNS_DIAGNOSTICS");
    return value != nullptr && std::string(value) != "0";
  }

  void configure_ablation()
  {
    const char* value = std::getenv("CUOPT_CPU_LNS_ABLATION");
    if (value == nullptr || std::string(value).empty() || std::string(value) == "none") { return; }
    ablation.name = value;
    if (ablation.name == "legacy_rounding") {
      ablation.use_fractional_lp_reference = false;
    } else if (ablation.name == "no_lp_reference") {
      ablation.use_fractional_lp_reference = false;
    } else if (ablation.name == "no_seed_bp") {
      ablation.use_seed_bp = false;
    } else if (ablation.name == "seed_natural_only") {
      ablation.seed_strategy = 0;
    } else if (ablation.name == "seed_lp_order_only") {
      ablation.seed_strategy = 1;
    } else if (ablation.name == "seed_degree_only") {
      ablation.seed_strategy = 2;
    } else if (ablation.name == "seed_random_only") {
      ablation.seed_strategy = 3;
    } else if (ablation.name == "no_initial_fj") {
      ablation.use_initial_fj = false;
    } else if (ablation.name == "no_periodic_fj") {
      ablation.use_periodic_fj = false;
    } else if (ablation.name == "no_fj") {
      ablation.use_initial_fj  = false;
      ablation.use_periodic_fj = false;
    } else if (ablation.name == "no_bandit") {
      ablation.use_bandit = false;
    } else if (ablation.name == "fixed_ruin") {
      ablation.use_adaptive_ruin_size = false;
    } else if (ablation.name == "no_violated_ruin") {
      ablation.ruin_enabled[0] = false;
    } else if (ablation.name == "no_similarity") {
      ablation.use_similarity  = false;
      ablation.ruin_enabled[1] = false;
    } else if (ablation.name == "no_random_walk") {
      ablation.ruin_enabled[2] = false;
    } else if (ablation.name == "no_propagate_repair") {
      ablation.repair_enabled[0] = false;
    } else if (ablation.name == "with_propagate_repair") {
      ablation.repair_enabled[0] = true;
    } else if (ablation.name == "no_shift_repair") {
      ablation.repair_enabled[1] = false;
    } else if (ablation.name == "no_greedy_repair") {
      ablation.repair_enabled[2] = false;
    } else if (ablation.name == "no_cosine") {
      ablation.use_cosine = false;
    } else if (ablation.name == "no_type_filter") {
      ablation.use_type_filter = false;
    } else if (ablation.name == "no_saturation") {
      ablation.use_saturation = false;
    } else if (ablation.name == "no_tardiness") {
      ablation.use_tardiness = false;
    } else if (ablation.name == "no_weight_decay") {
      ablation.use_weight_decay = false;
    } else if (ablation.name == "raw_violation_excess") {
      ablation.use_normalized_excess = false;
    } else if (ablation.name == "no_projection") {
      ablation.use_projection = false;
    } else if (ablation.name == "with_projection") {
      ablation.use_projection = true;
    } else if (ablation.name == "no_sa") {
      ablation.use_sa = false;
    } else if (ablation.name == "full_refresh") {
      ablation.full_refresh = true;
    } else {
      cuopt_expects(false,
                    error_type_t::RuntimeError,
                    "Unknown CPU LNS ablation variant: %s",
                    ablation.name.c_str());
    }
  }

  void init_host_data()
  {
    auto stream             = problem_ptr->handle_ptr->get_stream();
    problem_t<i_t, f_t>& pb = *problem_ptr;
    n_vars                  = pb.n_variables;
    n_cons                  = pb.n_constraints;
    n_int                   = pb.n_integer_vars;
    tols                    = pb.tolerances;

    offsets          = cuopt::host_copy(pb.offsets, stream);
    variables        = cuopt::host_copy(pb.variables, stream);
    coefficients     = cuopt::host_copy(pb.coefficients, stream);
    rev_offsets      = cuopt::host_copy(pb.reverse_offsets, stream);
    rev_constraints  = cuopt::host_copy(pb.reverse_constraints, stream);
    rev_coefficients = cuopt::host_copy(pb.reverse_coefficients, stream);
    cons_lb          = cuopt::host_copy(pb.constraint_lower_bounds, stream);
    cons_ub          = cuopt::host_copy(pb.constraint_upper_bounds, stream);
    objective        = cuopt::host_copy(pb.objective_coefficients, stream);

    auto bounds = cuopt::host_copy(pb.variable_bounds, stream);
    auto vtypes = cuopt::host_copy(pb.variable_types, stream);
    var_lb.resize(n_vars);
    var_ub.resize(n_vars);
    var_class.resize(n_vars);
    active_variables.reserve(n_vars);
    objective_variables.reserve(n_vars);
    for (i_t v = 0; v < n_vars; ++v) {
      var_lb[v]          = get_lower(bounds[v]);
      var_ub[v]          = get_upper(bounds[v]);
      const bool integer = vtypes[v] == var_t::INTEGER;
      const bool binary  = integer && var_lb[v] == f_t{0} && var_ub[v] == f_t{1};
      var_class[v]       = binary ? 1 : integer ? 2 : 0;
      if (var_ub[v] > var_lb[v] + tols.absolute_tolerance) { active_variables.push_back(v); }
      if (objective[v] != f_t{0} && var_ub[v] > var_lb[v] + tols.absolute_tolerance) {
        objective_variables.push_back(v);
      }
    }

    compute_row_scales();
    ruined.assign(n_vars, 0);
    candidate_stamp.assign(n_vars, 0);
    candidate_dissim.assign(n_vars, f_t{0});
    candidate_dot.assign(n_vars, f_t{0});
    candidate_norm_seed.assign(n_vars, f_t{0});
    candidate_norm_other.assign(n_vars, f_t{0});
    candidate_intersection.assign(n_vars, 0);
    projection_stamp.assign(n_vars, 0);
    constraint_stamp.assign(n_cons, 0);
    affected_position.assign(n_cons, -1);
    violated_position.assign(n_cons, -1);
    tardiness.assign(n_cons, f_t{0});
    tardiness_epoch.assign(n_cons, 0);
    tardiness_violated.assign(n_cons, 0);
    work_lb.assign(n_vars, f_t{0});
    work_ub.assign(n_vars, f_t{0});
  }

  void compute_row_scales()
  {
    row_inf_norm.assign(n_cons, f_t{0});
    row_scale.assign(n_cons, f_t{1});
    for (i_t c = 0; c < n_cons; ++c) {
      f_t inf_norm = 0;
      for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
        inf_norm = std::max(inf_norm, std::abs(coefficients[p]));
      }
      row_inf_norm[c] = inf_norm;
      f_t scale       = std::max(f_t{1}, inf_norm);
      if (std::isfinite(cons_lb[c])) { scale = std::max(scale, std::abs(cons_lb[c])); }
      if (std::isfinite(cons_ub[c])) { scale = std::max(scale, std::abs(cons_ub[c])); }
      row_scale[c] = scale;
    }
  }

  std::vector<f_t> compute_lhs(const std::vector<f_t>& x) const
  {
    std::vector<f_t> lhs(n_cons, f_t{0});
    for (i_t c = 0; c < n_cons; ++c) {
      f_t sum = 0;
      for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
        sum += coefficients[p] * x[variables[p]];
      }
      lhs[c] = sum;
    }
    return lhs;
  }

  f_t compute_objective(const std::vector<f_t>& x) const
  {
    return std::inner_product(x.begin(), x.end(), objective.begin(), f_t{0});
  }

  bool constraint_violated(i_t c, f_t value) const
  {
    return !is_constraint_feasible<i_t, f_t>(value, cons_lb[c], cons_ub[c], tols);
  }

  f_t excess_of(i_t c, f_t value) const
  {
    return std::max(f_t{0}, cons_lb[c] - value) + std::max(f_t{0}, value - cons_ub[c]);
  }

  f_t normalized_excess_of(i_t c, f_t value) const
  {
    const f_t excess = excess_of(c, value);
    return ablation.use_normalized_excess ? excess / row_scale[c] : excess;
  }

  state_t state_from_lhs(const std::vector<f_t>& lhs, const std::vector<f_t>& x) const
  {
    state_t state;
    for (i_t c = 0; c < n_cons; ++c) {
      const f_t excess = excess_of(c, lhs[c]);
      state.total_excess += excess;
      state.normalized_excess += normalized_excess_of(c, lhs[c]);
      state.unsat += constraint_violated(c, lhs[c]);
    }
    state.objective = compute_objective(x);
    return state;
  }

  void initialize_dynamic_state(const std::vector<f_t>& lhs)
  {
    violated_constraints.clear();
    std::fill(violated_position.begin(), violated_position.end(), -1);
    for (i_t c = 0; c < n_cons; ++c) {
      const bool violated   = constraint_violated(c, lhs[c]);
      tardiness_violated[c] = violated;
      if (violated) { add_violated_constraint(c); }
    }
  }

  void rebuild_dynamic_state(const std::vector<f_t>& lhs)
  {
    for (i_t c = 0; c < n_cons; ++c) {
      materialize_tardiness(c);
      tardiness_violated[c] = constraint_violated(c, lhs[c]);
    }
    initialize_dynamic_state(lhs);
  }

  void add_violated_constraint(i_t c)
  {
    if (violated_position[c] >= 0) { return; }
    violated_position[c] = static_cast<i_t>(violated_constraints.size());
    violated_constraints.push_back(c);
  }

  void remove_violated_constraint(i_t c)
  {
    const i_t position = violated_position[c];
    if (position < 0) { return; }
    const i_t moved                = violated_constraints.back();
    violated_constraints[position] = moved;
    violated_position[moved]       = position;
    violated_constraints.pop_back();
    violated_position[c] = -1;
  }

  void materialize_tardiness(i_t c)
  {
    if (!ablation.use_tardiness) { return; }
    const uint64_t delta = weight_epoch - tardiness_epoch[c];
    if (delta == 0) { return; }
    if (ablation.use_weight_decay) {
      const f_t decay = delta > 4096 ? f_t{0} : std::pow(WEIGHT_DECAY, static_cast<f_t>(delta));
      const f_t added = tardiness_violated[c] ? (f_t{1} - decay) / (f_t{1} - WEIGHT_DECAY) : f_t{0};
      tardiness[c]    = std::min(MAX_TARDINESS, tardiness[c] * decay + added);
    } else if (tardiness_violated[c]) {
      tardiness[c] = std::min(MAX_TARDINESS, tardiness[c] + static_cast<f_t>(delta));
    }
    tardiness_epoch[c] = weight_epoch;
  }

  f_t constraint_weight(i_t c)
  {
    if (!ablation.use_tardiness) { return f_t{1}; }
    materialize_tardiness(c);
    return f_t{1} + tardiness[c];
  }

  static constexpr size_t operator_pair_index(size_t ruin_arm, size_t repair_arm)
  {
    return ruin_arm * N_REPAIR_ARMS + repair_arm;
  }

  size_t select_operator_pair(uint64_t iteration)
  {
    std::array<bool, N_OPERATOR_PAIRS> enabled{};
    for (size_t ruin_arm = 0; ruin_arm < N_RUIN_ARMS; ++ruin_arm) {
      for (size_t repair_arm = 0; repair_arm < N_REPAIR_ARMS; ++repair_arm) {
        enabled[operator_pair_index(ruin_arm, repair_arm)] =
          ablation.ruin_enabled[ruin_arm] && ablation.repair_enabled[repair_arm];
      }
    }
    return select_arm(operator_pair_arms, enabled, iteration);
  }

  template <size_t n_arms>
  size_t select_arm(const std::array<arm_stats_t, n_arms>& arms,
                    const std::array<bool, n_arms>& enabled,
                    uint64_t iteration)
  {
    if (!ablation.use_bandit) {
      size_t enabled_count = 0;
      for (bool is_enabled : enabled) {
        enabled_count += is_enabled;
      }
      const size_t selected_rank = iteration % enabled_count;
      size_t rank                = 0;
      for (size_t arm = 0; arm < n_arms; ++arm) {
        if (enabled[arm] && rank++ == selected_rank) { return arm; }
      }
    }
    uint64_t total = 0;
    for (size_t arm = 0; arm < n_arms; ++arm) {
      if (!enabled[arm]) { continue; }
      if (arms[arm].pulls == 0) { return arm; }
      total += arms[arm].pulls;
    }
    size_t best_arm = n_arms;
    f_t best_score  = -std::numeric_limits<f_t>::infinity();
    for (size_t arm = 0; arm < n_arms; ++arm) {
      if (!enabled[arm]) { continue; }
      const f_t mean  = arms[arm].reward_sum / static_cast<f_t>(arms[arm].pulls);
      const f_t bonus = std::sqrt(f_t{2} * std::log(static_cast<f_t>(total) + f_t{1}) /
                                  static_cast<f_t>(arms[arm].pulls));
      if (mean + bonus > best_score) {
        best_score = mean + bonus;
        best_arm   = arm;
      }
    }
    cuopt_expects(best_arm < n_arms,
                  error_type_t::RuntimeError,
                  "CPU LNS ablation disabled every operator in a portfolio");
    return best_arm;
  }

  static void update_arm(arm_stats_t& arm, f_t reward)
  {
    ++arm.pulls;
    arm.reward_sum += reward;
  }

  i_t adaptive_ruin_size(size_t failure_streak, size_t attempt) const
  {
    if (!ablation.use_adaptive_ruin_size) {
      return std::min<i_t>(16, static_cast<i_t>(active_variables.size()));
    }
    const i_t portfolio_size = MIN_RUIN_SIZE << ((attempt / 64) % 4);
    const i_t stall_growth   = MIN_RUIN_SIZE << std::min<size_t>(failure_streak / 128, 3);
    const i_t active_limit   = std::max<i_t>(1, static_cast<i_t>(active_variables.size()));
    return std::min({MAX_RUIN_SIZE, std::max(portfolio_size, stall_growth), active_limit});
  }

  i_t intensification_ruin_size(const std::vector<f_t>& lhs)
  {
    (void)lhs;
    i_t union_bound = 0;
    for (i_t c : violated_constraints) {
      union_bound += offsets[c + 1] - offsets[c];
      if (union_bound >= MAX_INTENSIFICATION_RUIN) { break; }
    }
    const i_t cap =
      std::min<i_t>(MAX_INTENSIFICATION_RUIN, static_cast<i_t>(active_variables.size()));
    return std::min<i_t>(std::max<i_t>(128, 3 * union_bound), cap);
  }

  i_t select_violated_union_ruin(i_t target, const std::vector<f_t>& x, const std::vector<f_t>& lhs)
  {
    clear_ruin_set();
    std::vector<std::pair<f_t, i_t>> ordered_constraints;
    ordered_constraints.reserve(violated_constraints.size());
    for (i_t c : violated_constraints) {
      ordered_constraints.emplace_back(-constraint_weight(c) * normalized_excess_of(c, lhs[c]), c);
    }
    std::sort(ordered_constraints.begin(), ordered_constraints.end());
    for (const auto& [score, c] : ordered_constraints) {
      (void)score;
      for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
        mark_ruined(variables[p]);
        if (static_cast<i_t>(ruin_vars.size()) >= target) {
          return static_cast<i_t>(ruin_vars.size());
        }
      }
    }
    const std::vector<i_t> first_hop = ruin_vars;
    for (i_t v : first_hop) {
      for (i_t reverse = rev_offsets[v]; reverse < rev_offsets[v + 1]; ++reverse) {
        const i_t c = rev_constraints[reverse];
        for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
          mark_ruined(variables[p]);
          if (static_cast<i_t>(ruin_vars.size()) >= target) {
            return static_cast<i_t>(ruin_vars.size());
          }
        }
      }
    }
    if (ruin_vars.empty()) {
      const i_t seed = pick_seed_variable(x, lhs, false);
      mark_ruined(seed);
    }
    return static_cast<i_t>(ruin_vars.size());
  }

  i_t select_ruin_set(size_t arm,
                      i_t target,
                      const std::vector<f_t>& x,
                      const std::vector<f_t>& lhs,
                      bool improvement_phase)
  {
    clear_ruin_set();
    if (active_variables.empty()) { return 0; }
    const i_t seed = pick_seed_variable(x, lhs, improvement_phase);
    if (seed < 0) { return 0; }
    mark_ruined(seed);

    if (arm == 0) {
      expand_from_violated_row(target, x, lhs);
      if (ablation.use_similarity) { expand_similarity_chain(target, x); }
    } else if (arm == 1) {
      expand_similarity_chain(target, x);
    } else {
      expand_random_walk(target);
      if (ablation.use_similarity) { expand_similarity_chain(target, x); }
    }

    size_t fallback_attempts = 0;
    while (static_cast<i_t>(ruin_vars.size()) < target &&
           fallback_attempts++ < static_cast<size_t>(target) * 8) {
      std::uniform_int_distribution<size_t> dist(0, active_variables.size() - 1);
      mark_ruined(active_variables[dist(rng)]);
    }
    return static_cast<i_t>(ruin_vars.size());
  }

  i_t pick_seed_variable(const std::vector<f_t>& x,
                         const std::vector<f_t>& lhs,
                         bool improvement_phase)
  {
    if (!improvement_phase && !violated_constraints.empty()) {
      const i_t c = pick_violated_constraint(lhs);
      return pick_variable_from_row(c, x, lhs);
    }
    const auto& pool = objective_variables.empty() ? active_variables : objective_variables;
    if (pool.empty()) { return -1; }
    std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
    i_t best             = pool[dist(rng)];
    f_t best_score       = -std::numeric_limits<f_t>::infinity();
    const size_t samples = std::min<size_t>(32, pool.size());
    for (size_t sample = 0; sample < samples; ++sample) {
      const i_t v     = pool[dist(rng)];
      const f_t score = saturation_score(v, x) * (f_t{1} + std::abs(objective[v]));
      if (score > best_score) {
        best       = v;
        best_score = score;
      }
    }
    return best;
  }

  i_t pick_violated_constraint(const std::vector<f_t>& lhs)
  {
    std::uniform_int_distribution<size_t> dist(0, violated_constraints.size() - 1);
    i_t best             = violated_constraints[dist(rng)];
    f_t best_score       = -1;
    const size_t samples = std::min<size_t>(8, violated_constraints.size());
    for (size_t sample = 0; sample < samples; ++sample) {
      const i_t c     = violated_constraints[dist(rng)];
      const f_t score = constraint_weight(c) * normalized_excess_of(c, lhs[c]);
      if (score > best_score) {
        best       = c;
        best_score = score;
      }
    }
    return best;
  }

  i_t pick_variable_from_row(i_t c, const std::vector<f_t>& x, const std::vector<f_t>& lhs)
  {
    const i_t begin = offsets[c];
    const i_t count = offsets[c + 1] - begin;
    if (count <= 0) { return -1; }
    std::uniform_int_distribution<i_t> dist(0, count - 1);
    i_t best                   = -1;
    f_t best_score             = -1;
    const i_t samples          = std::min<i_t>(64, count);
    const f_t needed_direction = lhs[c] < cons_lb[c] ? f_t{1} : f_t{-1};
    for (i_t sample = 0; sample < samples; ++sample) {
      const i_t p = begin + (count <= samples ? sample : dist(rng));
      const i_t v = variables[p];
      if (!is_active(v)) { continue; }
      const f_t a               = coefficients[p];
      const f_t improving_bound = needed_direction * a > f_t{0} ? var_ub[v] : var_lb[v];
      f_t movement              = std::abs(improving_bound - x[v]);
      if (!std::isfinite(movement)) { movement = f_t{1} + std::abs(x[v]); }
      const f_t degree_penalty =
        std::sqrt(f_t{1} + static_cast<f_t>(rev_offsets[v + 1] - rev_offsets[v]));
      const f_t score =
        std::abs(a) * movement * saturation_score(v, x) / (row_scale[c] * degree_penalty);
      if (score > best_score) {
        best       = v;
        best_score = score;
      }
    }
    return best;
  }

  f_t saturation_score(i_t v, const std::vector<f_t>& x) const
  {
    if (!ablation.use_saturation) { return f_t{1}; }
    const f_t scale     = f_t{1} + std::abs(x[v]);
    const bool at_bound = std::abs(x[v] - var_lb[v]) <= tols.absolute_tolerance ||
                          std::abs(x[v] - var_ub[v]) <= tols.absolute_tolerance;
    const f_t agreement = f_t{1} / (f_t{1} + std::abs(x[v] - lp_reference[v]) / scale);
    return f_t{1} + static_cast<f_t>(at_bound) * (f_t{0.5} + f_t{0.5} * agreement);
  }

  bool is_active(i_t v) const
  {
    return v >= 0 && v < n_vars && var_ub[v] > var_lb[v] + tols.absolute_tolerance;
  }

  bool mark_ruined(i_t v)
  {
    if (!is_active(v) || ruined[v]) { return false; }
    ruined[v] = 1;
    ruin_vars.push_back(v);
    return true;
  }

  void clear_ruin_set()
  {
    for (i_t v : ruin_vars) {
      ruined[v] = 0;
    }
    ruin_vars.clear();
  }

  void expand_from_violated_row(i_t target, const std::vector<f_t>& x, const std::vector<f_t>& lhs)
  {
    if (violated_constraints.empty() || static_cast<i_t>(ruin_vars.size()) >= target) { return; }
    const i_t c     = pick_violated_constraint(lhs);
    const i_t begin = offsets[c];
    const i_t count = offsets[c + 1] - begin;
    if (count <= 0) { return; }
    std::uniform_int_distribution<i_t> dist(0, count - 1);
    const i_t samples = std::min<i_t>(std::max<i_t>(target * 4, 16), count);
    scored_candidates.clear();
    for (i_t sample = 0; sample < samples; ++sample) {
      const i_t p = begin + (count <= samples ? sample : dist(rng));
      const i_t v = variables[p];
      if (!is_active(v) || ruined[v]) { continue; }
      const f_t score = -std::abs(coefficients[p]) * saturation_score(v, x) / row_scale[c];
      scored_candidates.emplace_back(score, v);
    }
    std::sort(scored_candidates.begin(), scored_candidates.end());
    for (const auto& [score, v] : scored_candidates) {
      (void)score;
      if (static_cast<i_t>(ruin_vars.size()) >= target) { break; }
      mark_ruined(v);
    }
  }

  void expand_similarity_chain(i_t target, const std::vector<f_t>& x)
  {
    if (ruin_vars.empty()) { return; }
    i_t pivot = ruin_vars.back();
    while (static_cast<i_t>(ruin_vars.size()) < target) {
      score_similarity_candidates(pivot, x);
      if (scored_candidates.empty()) { break; }
      const i_t batch = std::min<i_t>(4, target - static_cast<i_t>(ruin_vars.size()));
      const i_t take  = std::min<i_t>(batch, static_cast<i_t>(scored_candidates.size()));
      std::partial_sort(
        scored_candidates.begin(), scored_candidates.begin() + take, scored_candidates.end());
      i_t added = 0;
      for (i_t index = 0; index < take; ++index) {
        if (mark_ruined(scored_candidates[index].second)) {
          pivot = scored_candidates[index].second;
          ++added;
        }
      }
      if (added == 0) { break; }
    }
  }

  void score_similarity_candidates(i_t seed, const std::vector<f_t>& x)
  {
    scored_candidates.clear();
    touched_candidates.clear();
    if (++candidate_generation == std::numeric_limits<i_t>::max()) {
      std::fill(candidate_stamp.begin(), candidate_stamp.end(), 0);
      candidate_generation = 1;
    }

    const i_t reverse_begin = rev_offsets[seed];
    const i_t reverse_count = rev_offsets[seed + 1] - reverse_begin;
    if (reverse_count <= 0) { return; }
    std::uniform_int_distribution<i_t> reverse_dist(0, reverse_count - 1);
    const i_t constraint_samples = std::min<i_t>(12, reverse_count);
    const f_t alpha =
      f_t{0.2} + f_t{0.6} / (f_t{1} + static_cast<f_t>(diagnostics.iterations) / f_t{256});

    for (i_t sample = 0; sample < constraint_samples; ++sample) {
      const i_t reverse_index =
        reverse_begin + (reverse_count <= constraint_samples ? sample : reverse_dist(rng));
      const i_t c   = rev_constraints[reverse_index];
      const f_t inf = row_inf_norm[c];
      if (!(inf > f_t{0})) { continue; }
      const f_t seed_hat = rev_coefficients[reverse_index] / inf;
      const i_t begin    = offsets[c];
      const i_t count    = offsets[c + 1] - begin;
      if (count <= 0) { continue; }
      std::uniform_int_distribution<i_t> row_dist(0, count - 1);
      const i_t row_samples = std::min<i_t>(48, count);
      for (i_t row_sample = 0; row_sample < row_samples; ++row_sample) {
        const i_t p = begin + (count <= row_samples ? row_sample : row_dist(rng));
        const i_t v = variables[p];
        if (v == seed || ruined[v] || !is_active(v) ||
            (ablation.use_type_filter && var_class[v] != var_class[seed])) {
          continue;
        }
        if (candidate_stamp[v] != candidate_generation) {
          candidate_stamp[v]        = candidate_generation;
          candidate_dissim[v]       = f_t{0};
          candidate_dot[v]          = f_t{0};
          candidate_norm_seed[v]    = f_t{0};
          candidate_norm_other[v]   = f_t{0};
          candidate_intersection[v] = 0;
          touched_candidates.push_back(v);
        }
        const f_t other_hat  = coefficients[p] / inf;
        const f_t structural = std::abs(seed_hat - other_hat);
        const f_t state      = std::abs(seed_hat * x[seed] - other_hat * x[v]) /
                          (f_t{1} + std::abs(x[seed]) + std::abs(x[v]));
        candidate_dissim[v] += alpha * structural + (f_t{1} - alpha) * state;
        candidate_dot[v] += seed_hat * other_hat;
        candidate_norm_seed[v] += seed_hat * seed_hat;
        candidate_norm_other[v] += other_hat * other_hat;
        ++candidate_intersection[v];
      }
    }

    const f_t seed_degree = static_cast<f_t>(reverse_count);
    for (i_t v : touched_candidates) {
      const f_t intersection = static_cast<f_t>(candidate_intersection[v]);
      const f_t cosine_denom = std::sqrt(candidate_norm_seed[v] * candidate_norm_other[v]);
      const f_t cosine       = cosine_denom > f_t{0} ? candidate_dot[v] / cosine_denom : f_t{0};
      const f_t degree       = static_cast<f_t>(rev_offsets[v + 1] - rev_offsets[v]);
      const f_t union_size   = seed_degree + degree - intersection;
      const f_t jaccard      = union_size > f_t{0} ? intersection / union_size : f_t{0};
      const f_t cosine_bonus = ablation.use_cosine ? f_t{0.5} * cosine : f_t{0};
      const f_t score        = candidate_dissim[v] / std::max(f_t{1}, intersection) - cosine_bonus -
                        jaccard - f_t{0.05} * saturation_score(v, x);
      scored_candidates.emplace_back(score, v);
    }
  }

  void expand_random_walk(i_t target)
  {
    if (ruin_vars.empty()) { return; }
    i_t pivot                    = ruin_vars.back();
    size_t no_progress           = 0;
    const size_t max_no_progress = static_cast<size_t>(8) * static_cast<size_t>(target);
    while (static_cast<i_t>(ruin_vars.size()) < target && no_progress++ < max_no_progress) {
      const i_t reverse_count = rev_offsets[pivot + 1] - rev_offsets[pivot];
      if (reverse_count <= 0) { break; }
      std::uniform_int_distribution<i_t> reverse_dist(0, reverse_count - 1);
      const i_t c         = rev_constraints[rev_offsets[pivot] + reverse_dist(rng)];
      const i_t row_count = offsets[c + 1] - offsets[c];
      if (row_count <= 0) { continue; }
      std::uniform_int_distribution<i_t> row_dist(0, row_count - 1);
      const i_t next = variables[offsets[c] + row_dist(rng)];
      if (mark_ruined(next)) { pivot = next; }
    }
  }

  void prepare_incremental_attempt(const std::vector<f_t>& x, const std::vector<f_t>& lhs)
  {
    old_values.clear();
    old_values.reserve(ruin_vars.size());
    affected_constraints.clear();
    old_lhs.clear();
    if (++constraint_generation == std::numeric_limits<i_t>::max()) {
      std::fill(constraint_stamp.begin(), constraint_stamp.end(), 0);
      constraint_generation = 1;
    }
    for (i_t v : ruin_vars) {
      old_values.push_back(x[v]);
      for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
        const i_t c = rev_constraints[p];
        if (constraint_stamp[c] == constraint_generation) { continue; }
        constraint_stamp[c]  = constraint_generation;
        affected_position[c] = static_cast<i_t>(affected_constraints.size());
        affected_constraints.push_back(c);
        old_lhs.push_back(lhs[c]);
      }
    }
  }

  bool extend_incremental_attempt(i_t v, const std::vector<f_t>& x, const std::vector<f_t>& lhs)
  {
    if (ruined[v]) { return true; }
    if (ruin_vars.size() >= static_cast<size_t>(MAX_INTENSIFICATION_RUIN) || !mark_ruined(v)) {
      return false;
    }
    old_values.push_back(x[v]);
    for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
      const i_t c = rev_constraints[p];
      if (constraint_stamp[c] == constraint_generation) { continue; }
      constraint_stamp[c]  = constraint_generation;
      affected_position[c] = static_cast<i_t>(affected_constraints.size());
      affected_constraints.push_back(c);
      // A newly reached row has not been modified by the existing ruin variables; otherwise it
      // would already be in affected_constraints. Its current value is therefore the pre-attempt
      // value needed for atomic rollback and candidate evaluation.
      old_lhs.push_back(lhs[c]);
    }
    return true;
  }

  void restore_incremental_attempt(std::vector<f_t>& x, std::vector<f_t>& lhs)
  {
    for (size_t index = 0; index < ruin_vars.size(); ++index) {
      x[ruin_vars[index]] = old_values[index];
    }
    for (size_t index = 0; index < affected_constraints.size(); ++index) {
      lhs[affected_constraints[index]] = old_lhs[index];
    }
  }

  bool seed_constraint_propagation(std::vector<f_t>& x, f_t budget, size_t strategy)
  {
    const auto start    = clock_t::now();
    std::vector<f_t> lb = var_lb;
    std::vector<f_t> ub = var_ub;
    initialize_seed_propagation_scratch();
    if (!seed_propagate(lb, ub, -1, start, budget)) { return false; }

    std::vector<i_t> fix_order;
    fix_order.reserve(static_cast<size_t>(n_int));
    for (i_t v = 0; v < n_vars; ++v) {
      if (var_class[v] != 0 && lb[v] < ub[v] - tols.absolute_tolerance) { fix_order.push_back(v); }
    }
    if (strategy == 1) {
      std::stable_sort(fix_order.begin(), fix_order.end(), [&](i_t lhs_var, i_t rhs_var) {
        const f_t lhs_fractionality =
          std::abs(lp_reference[lhs_var] - std::round(lp_reference[lhs_var]));
        const f_t rhs_fractionality =
          std::abs(lp_reference[rhs_var] - std::round(lp_reference[rhs_var]));
        if (std::abs(lhs_fractionality - rhs_fractionality) > OBJECTIVE_EPSILON) {
          return lhs_fractionality < rhs_fractionality;
        }
        return rev_offsets[lhs_var + 1] - rev_offsets[lhs_var] >
               rev_offsets[rhs_var + 1] - rev_offsets[rhs_var];
      });
    } else if (strategy == 2) {
      std::shuffle(fix_order.begin(), fix_order.end(), rng);
      std::stable_sort(fix_order.begin(), fix_order.end(), [&](i_t lhs_var, i_t rhs_var) {
        return rev_offsets[lhs_var + 1] - rev_offsets[lhs_var] >
               rev_offsets[rhs_var + 1] - rev_offsets[rhs_var];
      });
    } else if (strategy == 3) {
      std::shuffle(fix_order.begin(), fix_order.end(), rng);
    }

    for (i_t v : fix_order) {
      if (seed_deadline_reached(start, budget)) { return false; }
      if (lb[v] > ub[v] + tols.absolute_tolerance) { return false; }
      if (std::abs(lb[v] - ub[v]) <= tols.absolute_tolerance) { continue; }
      const f_t reference = std::min(std::max(lp_reference[v], lb[v]), ub[v]);
      f_t value           = std::round(reference);
      if (strategy >= 2) {
        const f_t lower = std::max(std::floor(reference), std::ceil(lb[v]));
        const f_t upper = std::min(std::ceil(reference), std::floor(ub[v]));
        if (lower < upper) {
          const f_t probability_upper = reference - std::floor(reference);
          std::uniform_real_distribution<f_t> coin(f_t{0}, f_t{1});
          value = coin(rng) < probability_upper ? upper : lower;
        }
      }
      value = std::max(value, std::ceil(lb[v] - tols.integrality_tolerance));
      value = std::min(value, std::floor(ub[v] + tols.integrality_tolerance));
      lb[v] = ub[v] = value;
      if (!seed_propagate(lb, ub, v, start, budget)) { return false; }
    }

    for (i_t v = 0; v < n_vars; ++v) {
      x[v] = var_class[v] == 0 ? std::min(std::max(x[v], lb[v]), ub[v]) : lb[v];
    }
    return true;
  }

  void initialize_seed_propagation_scratch()
  {
    seed_cons_active.assign(n_cons, 0);
    seed_cons_next.assign(n_cons, 0);
    seed_var_active.assign(n_vars, 0);
    seed_min_activity.assign(n_cons, f_t{0});
    seed_max_activity.assign(n_cons, f_t{0});
    seed_current_constraints.clear();
    seed_next_constraints.clear();
    seed_active_variables.clear();
  }

  bool seed_deadline_reached(const clock_t::time_point& start, f_t budget) const
  {
    return seconds_since(start) >= static_cast<double>(budget) || should_stop();
  }

  bool seed_propagate(std::vector<f_t>& lb,
                      std::vector<f_t>& ub,
                      i_t changed_variable,
                      const clock_t::time_point& start,
                      f_t budget)
  {
    seed_current_constraints.clear();
    if (changed_variable < 0) {
      seed_current_constraints.resize(n_cons);
      std::iota(seed_current_constraints.begin(), seed_current_constraints.end(), i_t{0});
    } else {
      for (i_t p = rev_offsets[changed_variable]; p < rev_offsets[changed_variable + 1]; ++p) {
        const i_t c = rev_constraints[p];
        if (!seed_cons_active[c]) {
          seed_cons_active[c] = 1;
          seed_current_constraints.push_back(c);
        }
      }
      for (i_t c : seed_current_constraints) {
        seed_cons_active[c] = 0;
      }
    }

    for (i_t propagation_pass = 0; propagation_pass < 10 && !seed_current_constraints.empty();
         ++propagation_pass) {
      seed_active_variables.clear();
      for (i_t c : seed_current_constraints) {
        seed_cons_active[c] = 1;
      }

      size_t processed_constraints = 0;
      for (i_t c : seed_current_constraints) {
        if (((++processed_constraints) & 255U) == 0U && seed_deadline_reached(start, budget)) {
          return false;
        }
        f_t min_value = 0;
        f_t max_value = 0;
        for (i_t p = offsets[c]; p < offsets[c + 1]; ++p) {
          const i_t v = variables[p];
          const f_t a = coefficients[p];
          min_value += std::min(a * lb[v], a * ub[v]);
          max_value += std::max(a * lb[v], a * ub[v]);
          if (!seed_var_active[v]) {
            seed_var_active[v] = 1;
            seed_active_variables.push_back(v);
          }
        }
        if (min_value > cons_ub[c] + tols.absolute_tolerance ||
            max_value < cons_lb[c] - tols.absolute_tolerance) {
          return false;
        }
        seed_min_activity[c] = min_value;
        seed_max_activity[c] = max_value;
      }

      seed_next_constraints.clear();
      size_t processed_variables = 0;
      for (i_t v : seed_active_variables) {
        if (((++processed_variables) & 255U) == 0U && seed_deadline_reached(start, budget)) {
          return false;
        }
        f_t new_lb = lb[v];
        f_t new_ub = ub[v];
        for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
          const i_t c = rev_constraints[p];
          if (!seed_cons_active[c]) { continue; }
          const f_t a = rev_coefficients[p];
          if (a == f_t{0}) { continue; }
          const f_t min_other = seed_min_activity[c] - std::min(a * lb[v], a * ub[v]);
          const f_t max_other = seed_max_activity[c] - std::max(a * lb[v], a * ub[v]);
          if (std::isfinite(cons_ub[c])) {
            const f_t bound = (cons_ub[c] - min_other) / a;
            if (a > 0) {
              new_ub = std::min(new_ub, bound);
            } else {
              new_lb = std::max(new_lb, bound);
            }
          }
          if (std::isfinite(cons_lb[c])) {
            const f_t bound = (cons_lb[c] - max_other) / a;
            if (a > 0) {
              new_lb = std::max(new_lb, bound);
            } else {
              new_ub = std::min(new_ub, bound);
            }
          }
        }
        if (var_class[v] != 0) {
          new_lb = std::ceil(new_lb - tols.integrality_tolerance);
          new_ub = std::floor(new_ub + tols.integrality_tolerance);
        }
        if (new_lb > new_ub + tols.absolute_tolerance) { return false; }
        if (new_lb > lb[v] || new_ub < ub[v]) {
          lb[v] = new_lb;
          ub[v] = new_ub;
          for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
            const i_t c = rev_constraints[p];
            if (!seed_cons_next[c]) {
              seed_cons_next[c] = 1;
              seed_next_constraints.push_back(c);
            }
          }
        }
      }

      for (i_t c : seed_current_constraints) {
        seed_cons_active[c] = 0;
      }
      for (i_t v : seed_active_variables) {
        seed_var_active[v] = 0;
      }
      for (i_t c : seed_next_constraints) {
        seed_cons_next[c] = 0;
      }
      seed_current_constraints.swap(seed_next_constraints);
    }
    return !seed_deadline_reached(start, budget);
  }

  void initialize_local_bound(i_t v, const std::vector<f_t>& x)
  {
    const f_t trust_multiplier = f_t{8} + static_cast<f_t>(ruin_vars.size());
    f_t lb                     = var_lb[v];
    f_t ub                     = var_ub[v];
    const f_t trust            = trust_multiplier * (f_t{1} + std::abs(x[v]));
    if (!std::isfinite(lb)) { lb = x[v] - trust; }
    if (!std::isfinite(ub)) { ub = x[v] + trust; }
    if (var_class[v] != 0) {
      lb = std::ceil(lb - tols.integrality_tolerance);
      ub = std::floor(ub + tols.integrality_tolerance);
    }
    work_lb[v] = lb;
    work_ub[v] = ub;
  }

  void initialize_local_bounds(const std::vector<f_t>& x)
  {
    for (i_t v : ruin_vars) {
      initialize_local_bound(v, x);
    }
  }

  bool propagate_local_bounds()
  {
    next_lb.resize(ruin_vars.size());
    next_ub.resize(ruin_vars.size());
    min_activity.resize(affected_constraints.size());
    max_activity.resize(affected_constraints.size());
    bool feasible = true;

    for (i_t pass = 0; pass < 2; ++pass) {
      min_activity = old_lhs;
      max_activity = old_lhs;
      for (size_t index = 0; index < ruin_vars.size(); ++index) {
        const i_t v       = ruin_vars[index];
        const f_t old_val = old_values[index];
        for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
          const i_t c   = rev_constraints[p];
          const i_t pos = affected_position[c];
          const f_t a   = rev_coefficients[p];
          min_activity[pos] += std::min(a * work_lb[v], a * work_ub[v]) - a * old_val;
          max_activity[pos] += std::max(a * work_lb[v], a * work_ub[v]) - a * old_val;
        }
      }

      bool any_change = false;
      for (size_t index = 0; index < ruin_vars.size(); ++index) {
        const i_t v = ruin_vars[index];
        f_t lb      = work_lb[v];
        f_t ub      = work_ub[v];
        for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
          const i_t c   = rev_constraints[p];
          const i_t pos = affected_position[c];
          const f_t a   = rev_coefficients[p];
          if (a == f_t{0}) { continue; }
          const f_t min_contribution = std::min(a * work_lb[v], a * work_ub[v]);
          const f_t max_contribution = std::max(a * work_lb[v], a * work_ub[v]);
          const f_t min_other        = min_activity[pos] - min_contribution;
          const f_t max_other        = max_activity[pos] - max_contribution;
          if (std::isfinite(cons_ub[c])) {
            const f_t bound = (cons_ub[c] - min_other) / a;
            if (a > 0) {
              ub = std::min(ub, bound);
            } else {
              lb = std::max(lb, bound);
            }
          }
          if (std::isfinite(cons_lb[c])) {
            const f_t bound = (cons_lb[c] - max_other) / a;
            if (a > 0) {
              lb = std::max(lb, bound);
            } else {
              ub = std::min(ub, bound);
            }
          }
        }
        if (var_class[v] != 0) {
          lb = std::ceil(lb - tols.integrality_tolerance);
          ub = std::floor(ub + tols.integrality_tolerance);
        }
        lb             = std::max(lb, work_lb[v]);
        ub             = std::min(ub, work_ub[v]);
        next_lb[index] = lb;
        next_ub[index] = ub;
        if (lb > ub + tols.absolute_tolerance) { feasible = false; }
        any_change |= lb > work_lb[v] || ub < work_ub[v];
      }
      if (!feasible) { return false; }
      for (size_t index = 0; index < ruin_vars.size(); ++index) {
        work_lb[ruin_vars[index]] = next_lb[index];
        work_ub[ruin_vars[index]] = next_ub[index];
      }
      if (!any_change) { break; }
    }
    return true;
  }

  // Greedy coordinate repair cannot cross a plateau when fixing one violated equality creates a
  // second violation that must be repaired by another variable.  Near feasibility, make a bounded
  // sequence of projections onto the currently worst row before the normal greedy polish.  The
  // whole neighborhood is still accepted or rejected atomically, so these deliberately
  // non-monotone intermediate moves cannot degrade the incumbent.
  bool repair_intensification(std::vector<f_t>& x, std::vector<f_t>& lhs)
  {
    initialize_local_bounds(x);
    if (++projection_generation == std::numeric_limits<i_t>::max()) {
      std::fill(projection_stamp.begin(), projection_stamp.end(), 0);
      projection_generation = 1;
    }

    bool changed               = false;
    const size_t desired_steps = std::min<size_t>(256, std::max<size_t>(8, ruin_vars.size() / 4));
    const size_t scan_budget_steps =
      std::max<size_t>(8, 2000000 / std::max<size_t>(1, affected_constraints.size()));
    const size_t max_steps =
      ablation.use_projection ? std::min(desired_steps, scan_budget_steps) : 0;
    for (size_t step = 0; step < max_steps && !should_stop(); ++step) {
      i_t target_row      = -1;
      f_t target_priority = -1;
      f_t target_excess   = 0;
      for (i_t c : affected_constraints) {
        const f_t normalized_excess = normalized_excess_of(c, lhs[c]);
        if (!constraint_violated(c, lhs[c]) || normalized_excess <= f_t{0}) { continue; }
        const f_t priority =
          normalized_excess * (f_t{1} + std::log1p(std::min(constraint_weight(c), f_t{20})));
        if (priority > target_priority) {
          target_row      = c;
          target_priority = priority;
          target_excess   = normalized_excess;
        }
      }
      if (target_row < 0) { break; }

      i_t best_variable          = -1;
      f_t best_value             = 0;
      f_t best_target_excess     = std::numeric_limits<f_t>::infinity();
      f_t best_weighted_unsat    = std::numeric_limits<f_t>::infinity();
      f_t best_weighted_excess   = std::numeric_limits<f_t>::infinity();
      f_t best_absolute_movement = std::numeric_limits<f_t>::infinity();
      const f_t row_target =
        lhs[target_row] < cons_lb[target_row] ? cons_lb[target_row] : cons_ub[target_row];

      for (i_t row_pos = offsets[target_row]; row_pos < offsets[target_row + 1]; ++row_pos) {
        const i_t v = variables[row_pos];
        const f_t a = coefficients[row_pos];
        if (!is_active(v) || a == f_t{0} || projection_stamp[v] == projection_generation ||
            (!ruined[v] && ruin_vars.size() >= static_cast<size_t>(MAX_INTENSIFICATION_RUIN))) {
          continue;
        }
        if (!ruined[v]) { initialize_local_bound(v, x); }

        const f_t projected = x[v] + (row_target - lhs[target_row]) / a;
        repair_values.clear();
        add_repair_value(v, projected);
        add_repair_value(v, lp_reference[v]);
        if (var_class[v] != 0) {
          add_repair_value(v, std::floor(projected));
          add_repair_value(v, std::ceil(projected));
        }
        add_repair_value(v, (row_target - lhs[target_row]) * a > f_t{0} ? work_ub[v] : work_lb[v]);

        for (f_t value : repair_values) {
          const f_t delta = value - x[v];
          if (std::abs(delta) <= tols.absolute_tolerance) { continue; }
          const f_t new_target_excess =
            normalized_excess_of(target_row, lhs[target_row] + a * delta);
          if (new_target_excess + OBJECTIVE_EPSILON >= target_excess) { continue; }

          f_t weighted_unsat  = 0;
          f_t weighted_excess = 0;
          for (i_t reverse = rev_offsets[v]; reverse < rev_offsets[v + 1]; ++reverse) {
            const i_t c       = rev_constraints[reverse];
            const f_t new_lhs = lhs[c] + rev_coefficients[reverse] * delta;
            const f_t weight  = constraint_weight(c);
            weighted_unsat += weight * static_cast<f_t>(constraint_violated(c, new_lhs));
            weighted_excess += weight * normalized_excess_of(c, new_lhs);
          }
          const f_t absolute_movement = std::abs(delta);
          const bool better = new_target_excess + OBJECTIVE_EPSILON < best_target_excess ||
                              (new_target_excess <= best_target_excess + OBJECTIVE_EPSILON &&
                               weighted_unsat + OBJECTIVE_EPSILON < best_weighted_unsat) ||
                              (new_target_excess <= best_target_excess + OBJECTIVE_EPSILON &&
                               weighted_unsat <= best_weighted_unsat + OBJECTIVE_EPSILON &&
                               weighted_excess + OBJECTIVE_EPSILON < best_weighted_excess) ||
                              (new_target_excess <= best_target_excess + OBJECTIVE_EPSILON &&
                               weighted_unsat <= best_weighted_unsat + OBJECTIVE_EPSILON &&
                               weighted_excess <= best_weighted_excess + OBJECTIVE_EPSILON &&
                               absolute_movement < best_absolute_movement);
          if (better) {
            best_variable          = v;
            best_value             = value;
            best_target_excess     = new_target_excess;
            best_weighted_unsat    = weighted_unsat;
            best_weighted_excess   = weighted_excess;
            best_absolute_movement = absolute_movement;
          }
        }
      }
      if (best_variable < 0) { break; }
      if (!extend_incremental_attempt(best_variable, x, lhs)) { break; }
      apply_value(best_variable, best_value, x, lhs);
      projection_stamp[best_variable] = projection_generation;
      ++diagnostics.projection_moves;
      changed = true;
    }

    if (!ablation.repair_enabled[2]) { return changed; }
    repair_order = ruin_vars;
    std::sort(repair_order.begin(), repair_order.end(), [&](i_t lhs_var, i_t rhs_var) {
      return variable_pressure(lhs_var, lhs) > variable_pressure(rhs_var, lhs);
    });
    for (size_t sweep = 0; sweep < 3; ++sweep) {
      if (sweep > 0) { std::shuffle(repair_order.begin(), repair_order.end(), rng); }
      bool sweep_changed = false;
      for (size_t order_index = 0; order_index < repair_order.size(); ++order_index) {
        if ((order_index & 255U) == 0U && should_stop()) { return changed; }
        const i_t v = repair_order[order_index];
        if (sweep == 0 && projection_stamp[v] == projection_generation) { continue; }
        const f_t value = choose_repair_value(v, 2, x, lhs, false);
        if (std::abs(value - x[v]) > tols.absolute_tolerance) {
          apply_value(v, value, x, lhs);
          changed       = true;
          sweep_changed = true;
        }
      }
      if (!sweep_changed) { break; }
    }
    return changed;
  }

  bool repair_incrementally(size_t arm,
                            std::vector<f_t>& x,
                            std::vector<f_t>& lhs,
                            bool improvement_phase)
  {
    initialize_local_bounds(x);
    const bool use_propagation = arm != 2;
    if (use_propagation && !propagate_local_bounds()) { ++diagnostics.propagation_conflicts; }

    repair_order = ruin_vars;
    if (arm == 1) {
      std::shuffle(repair_order.begin(), repair_order.end(), rng);
    } else {
      std::sort(repair_order.begin(), repair_order.end(), [&](i_t lhs_var, i_t rhs_var) {
        return variable_pressure(lhs_var, lhs) > variable_pressure(rhs_var, lhs);
      });
    }

    bool changed        = false;
    const size_t sweeps = arm == 2 && ruin_vars.size() > 32 ? 4 : 1;
    for (size_t sweep = 0; sweep < sweeps; ++sweep) {
      if (sweep > 0) { std::shuffle(repair_order.begin(), repair_order.end(), rng); }
      bool sweep_changed = false;
      for (size_t order_index = 0; order_index < repair_order.size(); ++order_index) {
        if ((order_index & 255U) == 0U && should_stop()) { return changed; }
        const i_t v     = repair_order[order_index];
        const f_t value = choose_repair_value(v, arm, x, lhs, improvement_phase);
        if (std::abs(value - x[v]) > tols.absolute_tolerance) {
          apply_value(v, value, x, lhs);
          changed       = true;
          sweep_changed = true;
        }
        if (use_propagation) {
          work_lb[v] = value;
          work_ub[v] = value;
          if ((order_index & 1U) != 0U && !propagate_local_bounds()) {
            ++diagnostics.propagation_conflicts;
          }
        }
      }
      if (!sweep_changed) { break; }
    }
    return changed;
  }

  f_t variable_pressure(i_t v, const std::vector<f_t>& lhs)
  {
    f_t pressure = 0;
    for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
      const i_t c = rev_constraints[p];
      if (constraint_violated(c, lhs[c])) {
        pressure += constraint_weight(c) * normalized_excess_of(c, lhs[c]);
      }
    }
    return pressure;
  }

  f_t choose_repair_value(i_t v,
                          size_t arm,
                          const std::vector<f_t>& x,
                          const std::vector<f_t>& lhs,
                          bool improvement_phase)
  {
    repair_values.clear();
    add_repair_value(v, x[v]);
    add_repair_value(v, lp_reference[v]);
    add_repair_value(v, work_lb[v]);
    add_repair_value(v, work_ub[v]);
    add_repair_value(v, (work_lb[v] + work_ub[v]) / f_t{2});
    if (improvement_phase) {
      add_repair_value(v, objective[v] <= f_t{0} ? work_ub[v] : work_lb[v]);
    }

    const i_t reverse_begin = rev_offsets[v];
    const i_t reverse_count = rev_offsets[v + 1] - reverse_begin;
    if (reverse_count > 0) {
      std::uniform_int_distribution<i_t> dist(0, reverse_count - 1);
      const i_t samples = std::min<i_t>(12, reverse_count);
      for (i_t sample = 0; sample < samples; ++sample) {
        const i_t p = reverse_begin + (reverse_count <= samples ? sample : dist(rng));
        const i_t c = rev_constraints[p];
        const f_t a = rev_coefficients[p];
        if (a == f_t{0}) { continue; }
        const f_t other = lhs[c] - a * x[v];
        if (std::isfinite(cons_lb[c])) { add_repair_value(v, (cons_lb[c] - other) / a); }
        if (std::isfinite(cons_ub[c])) { add_repair_value(v, (cons_ub[c] - other) / a); }
      }
    }

    if (arm == 1) {
      if (var_class[v] == 0) {
        std::uniform_real_distribution<f_t> dist(work_lb[v], work_ub[v]);
        add_repair_value(v, dist(rng));
      } else {
        const auto lb = static_cast<int64_t>(std::ceil(work_lb[v]));
        const auto ub = static_cast<int64_t>(std::floor(work_ub[v]));
        if (lb <= ub) {
          std::uniform_int_distribution<int64_t> dist(lb, ub);
          add_repair_value(v, static_cast<f_t>(dist(rng)));
        }
      }
    }

    f_t best_value            = x[v];
    i_t best_weighted_unsat   = std::numeric_limits<i_t>::max();
    f_t best_weighted_excess  = std::numeric_limits<f_t>::infinity();
    f_t best_objective_change = std::numeric_limits<f_t>::infinity();
    std::vector<size_t> ties;
    for (size_t index = 0; index < repair_values.size(); ++index) {
      const f_t value     = repair_values[index];
      i_t weighted_unsat  = 0;
      f_t weighted_excess = 0;
      const f_t delta     = value - x[v];
      for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
        const i_t c       = rev_constraints[p];
        const f_t new_lhs = lhs[c] + rev_coefficients[p] * delta;
        const f_t weight  = constraint_weight(c);
        weighted_unsat +=
          constraint_violated(c, new_lhs) ? static_cast<i_t>(std::min<f_t>(weight, f_t{1000})) : 0;
        weighted_excess += weight * normalized_excess_of(c, new_lhs);
      }
      const f_t objective_change = objective[v] * delta;
      const bool better =
        weighted_unsat < best_weighted_unsat ||
        (weighted_unsat == best_weighted_unsat &&
         weighted_excess + OBJECTIVE_EPSILON < best_weighted_excess) ||
        (weighted_unsat == best_weighted_unsat &&
         weighted_excess <= best_weighted_excess + OBJECTIVE_EPSILON && improvement_phase &&
         objective_change < best_objective_change - OBJECTIVE_EPSILON);
      if (better) {
        best_value            = value;
        best_weighted_unsat   = weighted_unsat;
        best_weighted_excess  = weighted_excess;
        best_objective_change = objective_change;
        ties.clear();
        ties.push_back(index);
      } else if (weighted_unsat == best_weighted_unsat &&
                 std::abs(weighted_excess - best_weighted_excess) <= OBJECTIVE_EPSILON) {
        ties.push_back(index);
      }
    }

    // Shift-and-propagate needs occasional non-greedy assignments to coordinate equality swaps.
    if (arm == 1 && ties.size() > 1) {
      std::uniform_int_distribution<size_t> tie_dist(0, ties.size() - 1);
      best_value = repair_values[ties[tie_dist(rng)]];
    }
    return best_value;
  }

  void add_repair_value(i_t v, f_t value)
  {
    if (!std::isfinite(value)) { return; }
    value = std::min(std::max(value, work_lb[v]), work_ub[v]);
    if (var_class[v] != 0) { value = std::round(value); }
    value = std::min(std::max(value, work_lb[v]), work_ub[v]);
    for (f_t existing : repair_values) {
      if (std::abs(existing - value) <= tols.absolute_tolerance) { return; }
    }
    repair_values.push_back(value);
  }

  void apply_value(i_t v, f_t value, std::vector<f_t>& x, std::vector<f_t>& lhs)
  {
    const f_t delta = value - x[v];
    x[v]            = value;
    for (i_t p = rev_offsets[v]; p < rev_offsets[v + 1]; ++p) {
      lhs[rev_constraints[p]] += rev_coefficients[p] * delta;
    }
  }

  candidate_metrics_t evaluate_candidate(const state_t& previous,
                                         const std::vector<f_t>& x,
                                         const std::vector<f_t>& lhs)
  {
    const auto start = clock_t::now();
    candidate_metrics_t candidate;
    candidate.state = previous;
    for (size_t index = 0; index < affected_constraints.size(); ++index) {
      const i_t c          = affected_constraints[index];
      const f_t old_value  = old_lhs[index];
      const f_t new_value  = lhs[c];
      const bool old_unsat = constraint_violated(c, old_value);
      const bool new_unsat = constraint_violated(c, new_value);
      const f_t old_excess = excess_of(c, old_value);
      const f_t new_excess = excess_of(c, new_value);
      const f_t weight     = constraint_weight(c);
      candidate.state.unsat += static_cast<i_t>(new_unsat) - static_cast<i_t>(old_unsat);
      candidate.state.total_excess += new_excess - old_excess;
      const f_t normalized_excess_delta =
        normalized_excess_of(c, new_value) - normalized_excess_of(c, old_value);
      candidate.state.normalized_excess += normalized_excess_delta;
      candidate.weighted_count_delta +=
        weight * (static_cast<f_t>(new_unsat) - static_cast<f_t>(old_unsat));
      candidate.weighted_excess_delta += weight * normalized_excess_delta;
    }
    candidate.state.objective = previous.objective;
    for (size_t index = 0; index < ruin_vars.size(); ++index) {
      const i_t v = ruin_vars[index];
      candidate.state.objective += objective[v] * (x[v] - old_values[index]);
    }
    diagnostics.evaluate_s += seconds_since(start);
    return candidate;
  }

  bool accept_candidate(const candidate_metrics_t& candidate,
                        const state_t& current,
                        bool improvement_phase,
                        f_t temperature)
  {
    if (!improvement_phase) {
      if (candidate.state.unsat < current.unsat) { return true; }
      const i_t max_unsat_increase = std::max<i_t>(2, current.unsat / 10);
      if (candidate.state.unsat <= current.unsat + max_unsat_increase &&
          candidate.weighted_count_delta < -OBJECTIVE_EPSILON) {
        return true;
      }
      return candidate.state.unsat == current.unsat &&
             candidate.weighted_count_delta <= OBJECTIVE_EPSILON &&
             (candidate.weighted_excess_delta < -OBJECTIVE_EPSILON ||
              candidate.state.normalized_excess + OBJECTIVE_EPSILON < current.normalized_excess);
    }

    if (candidate.state.unsat != 0) { return false; }
    const f_t delta = candidate.state.objective - current.objective;
    if (delta <= OBJECTIVE_EPSILON) { return true; }
    // `initial_temperature` is already expressed in objective units (one percent of the current
    // objective scale). Multiplying it by the objective scale again made acceptance effectively
    // exp(-delta / objective^2), so even enormous worsening moves were almost always accepted.
    const f_t probability = std::exp(-delta / std::max(temperature, MIN_TEMPERATURE));
    std::uniform_real_distribution<f_t> coin(f_t{0}, f_t{1});
    const bool sampled = coin(rng) < probability;
    return ablation.use_sa && sampled;
  }

  void commit_candidate(const std::vector<f_t>& lhs)
  {
    for (i_t c : affected_constraints) {
      materialize_tardiness(c);
      const bool violated   = constraint_violated(c, lhs[c]);
      tardiness_violated[c] = violated;
      if (violated) {
        add_violated_constraint(c);
      } else {
        remove_violated_constraint(c);
      }
    }
  }

  bool better_for_phase(const state_t& candidate,
                        const state_t& current,
                        bool improvement_phase) const
  {
    if (!improvement_phase) {
      return candidate.unsat < current.unsat ||
             (candidate.unsat == current.unsat &&
              candidate.normalized_excess + OBJECTIVE_EPSILON < current.normalized_excess);
    }
    return candidate.unsat == 0 && candidate.objective + OBJECTIVE_EPSILON < current.objective;
  }

  bool should_stop() const
  {
    return halted.load(std::memory_order_relaxed) || context.preempt_heuristic_solver_.load() ||
           lns_timer.check_time_limit();
  }

  void report_feasible(const std::vector<f_t>& assignment, f_t objective)
  {
    if (!feasible_callback) { return; }
    feasible_callback(objective, assignment);
  }

  bool consume_external_messages(std::vector<f_t>& x,
                                 std::vector<f_t>& lhs,
                                 state_t& current,
                                 std::vector<f_t>& best,
                                 state_t& best_state,
                                 bool& have_feasible,
                                 f_t& temperature)
  {
    std::vector<f_t> pending_assignment;
    std::vector<f_t> pending_lp;
    f_t pending_objective = std::numeric_limits<f_t>::infinity();
    bool take_incumbent   = false;
    bool take_lp          = false;
    {
      std::lock_guard<std::mutex> lock(inject_mutex);
      if (has_injected_lp) {
        pending_lp      = std::move(injected_lp_reference);
        has_injected_lp = false;
        take_lp         = true;
      }
      if (has_injected_incumbent) {
        pending_assignment     = std::move(injected_assignment);
        pending_objective      = injected_objective;
        has_injected_incumbent = false;
        take_incumbent         = true;
      }
    }

    bool adopted = false;
    if (take_lp && pending_lp.size() == static_cast<size_t>(n_vars)) {
      lp_reference = std::move(pending_lp);
    }
    if (take_incumbent && pending_assignment.size() == static_cast<size_t>(n_vars)) {
      state_t injected_state = state_from_lhs(compute_lhs(pending_assignment), pending_assignment);
      if (injected_state.unsat == 0 && better_for_best(injected_state, best_state)) {
        x           = std::move(pending_assignment);
        lhs         = compute_lhs(x);
        current     = state_from_lhs(lhs, x);
        rebuild_dynamic_state(lhs);
        best           = x;
        best_state     = current;
        have_feasible  = true;
        temperature    = initial_temperature(best_state.objective);
        adopted        = true;
        ++diagnostics.best_updates;
        record_first_feasible("external_incumbent", diagnostics.iterations, best_state);
        CUOPT_LOG_INFO("CPU LNS adopted external incumbent with objective %.6e",
                       problem_ptr->get_user_obj_from_solver_obj(pending_objective));
      }
    }
    return adopted;
  }

  bool better_for_best(const state_t& candidate, const state_t& incumbent) const
  {
    if (candidate.unsat == 0 && incumbent.unsat != 0) { return true; }
    if (candidate.unsat != 0 && incumbent.unsat == 0) { return false; }
    if (candidate.unsat != incumbent.unsat) { return candidate.unsat < incumbent.unsat; }
    if (candidate.unsat == 0) {
      return candidate.objective + OBJECTIVE_EPSILON < incumbent.objective;
    }
    return candidate.normalized_excess + OBJECTIVE_EPSILON < incumbent.normalized_excess;
  }

  f_t initial_temperature(f_t objective_value) const
  {
    return std::max(f_t{1e-3}, f_t{0.01} * (f_t{1} + std::abs(objective_value)));
  }

  bool run_fj_burst(std::vector<f_t>& x,
                    f_t time_limit,
                    solution_t<i_t, f_t>& solution,
                    bool improvement_phase)
  {
    if (time_limit <= f_t{0}) { return false; }
    const auto start = clock_t::now();
    solution.copy_new_assignment(x);

    fj_settings_t settings;
    settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
    settings.n_of_minimums_for_exit = 500;
    settings.update_weights         = true;
    settings.feasibility_run        = !improvement_phase;
    auto fj_cpu                     = init_fj_cpu_standalone<i_t, f_t>(
      *problem_ptr, solution, context.preempt_heuristic_solver_, settings);
    if (context.settings.seed >= 0) {
      fj_cpu->settings.seed =
        context.settings.seed + static_cast<i_t>(104729 * diagnostics.fj_calls);
    }
    fj_cpu->log_prefix = "[LNS CPUFJ] ";
    cpufj_solve<i_t, f_t>(fj_cpu.get(), time_limit);

    ++diagnostics.fj_calls;
    diagnostics.fj_iterations += fj_cpu->iterations;
    diagnostics.fj_s += seconds_since(start);

    state_t best_state = state_from_lhs(compute_lhs(x), x);
    bool changed       = false;
    auto consider      = [&](const std::vector<f_t>& candidate) {
      const state_t state = state_from_lhs(compute_lhs(candidate), candidate);
      if (better_for_best(state, best_state)) {
        best_state = state;
        x          = candidate;
        changed    = true;
      }
    };
    consider(fj_cpu->h_assignment.underlying());
    if (fj_cpu->feasible_found) { consider(fj_cpu->h_best_assignment.underlying()); }
    return changed;
  }

  void record_first_feasible(const char* source, uint64_t iteration, const state_t& state)
  {
    if (diagnostics.first_feasible_lns_s >= 0) { return; }
    diagnostics.first_feasible_lns_s     = lns_timer.elapsed_time();
    diagnostics.first_feasible_solve_s   = solve_elapsed_offset + diagnostics.first_feasible_lns_s;
    diagnostics.first_feasible_objective = state.objective;
    CUOPT_LOG_INFO(
      "CPU_LNS_FIRST_FEASIBLE source=%s iteration=%lu elapsed_s=%.6f solve_elapsed_s=%.6f "
      "objective_solver=%.12e objective_user=%.12e",
      source,
      iteration,
      diagnostics.first_feasible_lns_s,
      diagnostics.first_feasible_solve_s,
      state.objective,
      problem_ptr->get_user_obj_from_solver_obj(state.objective));
    log_objective("first_feasible", iteration, state, state, f_t{0}, true);
  }

  static const char* ruin_arm_name(size_t arm)
  {
    static constexpr std::array<const char*, N_RUIN_ARMS> names = {
      "violated_row", "similarity", "random_walk"};
    return names[arm];
  }

  double wall_iterations_per_second() const
  {
    return diagnostics.iterations /
           std::max(diagnostics.elapsed_s, std::numeric_limits<double>::epsilon());
  }

  double productive_iterations_per_second() const
  {
    return diagnostics.productive_attempts /
           std::max(diagnostics.elapsed_s, std::numeric_limits<double>::epsilon());
  }

  void log_progress(const state_t& current, const state_t& best_state, f_t elapsed, f_t temperature)
  {
    CUOPT_LOG_INFO(
      "CPU LNS progress: %.2fs elapsed, %lu iterations, %.1f it/s, productive %lu, accepted %lu, "
      "best unsat %d, excess %.6e, best objective %.12e",
      elapsed,
      diagnostics.iterations,
      diagnostics.iterations / std::max<f_t>(elapsed, f_t{1e-6}),
      diagnostics.productive_attempts,
      diagnostics.accepted,
      best_state.unsat,
      best_state.total_excess,
      best_state.objective);
    if (best_state.unsat == 0) {
      log_objective("sample", diagnostics.iterations, current, best_state, temperature, true);
    }
  }

  void log_objective(const char* event,
                     uint64_t iteration,
                     const state_t& current,
                     const state_t& best,
                     f_t temperature,
                     bool force = false)
  {
    if (!diagnostics_enabled || best.unsat != 0) { return; }
    const double elapsed = lns_timer.elapsed_time();
    if (!force && elapsed - last_objective_log_s < 0.25) { return; }
    last_objective_log_s = elapsed;
    CUOPT_LOG_INFO(
      "CPU_LNS_OBJECTIVE event=%s iteration=%lu elapsed_s=%.6f solve_elapsed_s=%.6f "
      "current_solver=%.12e best_solver=%.12e best_user=%.12e temperature=%.12e",
      event,
      iteration,
      elapsed,
      solve_elapsed_offset + elapsed,
      current.objective,
      best.objective,
      problem_ptr->get_user_obj_from_solver_obj(best.objective),
      temperature);
  }

  void log_diagnostics(const state_t& best_state) const
  {
    if (!diagnostics_enabled) { return; }
    CUOPT_LOG_INFO(
      "CPU_LNS_DIAGNOSTICS elapsed_s=%.6f setup_s=%.6f iterations=%lu productive_iterations=%lu "
      "iterations_per_second=%.3f productive_iterations_per_second=%.3f accepted=%lu "
      "best_updates=%lu intensifications=%lu empty_ruins=%lu propagation_conflicts=%lu "
      "lhs_refreshes=%lu fj_calls=%lu "
      "fj_iterations=%lu fj_s=%.6f seed_bp_attempted=%lu seed_bp_completed=%lu seed_bp_s=%.6f "
      "select_s=%.6f repair_s=%.6f evaluate_s=%.6f sa_reheats=%lu "
      "sa_uphill_accepts=%lu sa_escape_best_updates=%lu objective_best_updates=%lu "
      "best_unsat=%d best_excess=%.9e best_objective=%.12e first_feasible_lns_s=%.6f "
      "first_feasible_solve_s=%.6f first_feasible_objective=%.12e "
      "ruin_violated_attempts=%lu ruin_similarity_attempts=%lu ruin_walk_attempts=%lu "
      "ruin_violated_accepts=%lu ruin_similarity_accepts=%lu ruin_walk_accepts=%lu "
      "repair_propagate_attempts=%lu repair_shift_attempts=%lu repair_greedy_attempts=%lu "
      "repair_propagate_accepts=%lu repair_shift_accepts=%lu repair_greedy_accepts=%lu",
      diagnostics.elapsed_s,
      diagnostics.setup_s,
      diagnostics.iterations,
      diagnostics.productive_attempts,
      wall_iterations_per_second(),
      productive_iterations_per_second(),
      diagnostics.accepted,
      diagnostics.best_updates,
      diagnostics.intensifications,
      diagnostics.empty_ruins,
      diagnostics.propagation_conflicts,
      diagnostics.lhs_refreshes,
      diagnostics.fj_calls,
      diagnostics.fj_iterations,
      diagnostics.fj_s,
      diagnostics.seed_bp_attempted,
      diagnostics.seed_bp_completed,
      diagnostics.seed_bp_s,
      diagnostics.select_s,
      diagnostics.repair_s,
      diagnostics.evaluate_s,
      diagnostics.sa_reheats,
      diagnostics.sa_uphill_accepts,
      diagnostics.sa_escape_best_updates,
      diagnostics.objective_best_updates,
      best_state.unsat,
      best_state.total_excess,
      best_state.objective,
      diagnostics.first_feasible_lns_s,
      diagnostics.first_feasible_solve_s,
      diagnostics.first_feasible_objective,
      diagnostics.ruin_attempts[0],
      diagnostics.ruin_attempts[1],
      diagnostics.ruin_attempts[2],
      diagnostics.ruin_accepts[0],
      diagnostics.ruin_accepts[1],
      diagnostics.ruin_accepts[2],
      diagnostics.repair_attempts[0],
      diagnostics.repair_attempts[1],
      diagnostics.repair_attempts[2],
      diagnostics.repair_accepts[0],
      diagnostics.repair_accepts[1],
      diagnostics.repair_accepts[2]);
    CUOPT_LOG_INFO(
      "CPU_LNS_DIAGNOSTICS projection_moves=%lu "
      "ruin_violated_best=%lu ruin_similarity_best=%lu ruin_walk_best=%lu "
      "repair_propagate_best=%lu repair_shift_best=%lu repair_greedy_best=%lu "
      "ruin_violated_reward=%.3f ruin_similarity_reward=%.3f ruin_walk_reward=%.3f "
      "repair_propagate_reward=%.3f repair_shift_reward=%.3f repair_greedy_reward=%.3f",
      diagnostics.projection_moves,
      diagnostics.ruin_best_updates[0],
      diagnostics.ruin_best_updates[1],
      diagnostics.ruin_best_updates[2],
      diagnostics.repair_best_updates[0],
      diagnostics.repair_best_updates[1],
      diagnostics.repair_best_updates[2],
      static_cast<double>(ruin_arms[0].reward_sum),
      static_cast<double>(ruin_arms[1].reward_sum),
      static_cast<double>(ruin_arms[2].reward_sum),
      static_cast<double>(repair_arms[0].reward_sum),
      static_cast<double>(repair_arms[1].reward_sum),
      static_cast<double>(repair_arms[2].reward_sum));
    CUOPT_LOG_INFO(
      "CPU_LNS_PAIR_DIAGNOSTICS "
      "p00_pulls=%lu p01_pulls=%lu p02_pulls=%lu p10_pulls=%lu p11_pulls=%lu "
      "p12_pulls=%lu p20_pulls=%lu p21_pulls=%lu p22_pulls=%lu "
      "p00_reward=%.3f p01_reward=%.3f p02_reward=%.3f p10_reward=%.3f p11_reward=%.3f "
      "p12_reward=%.3f p20_reward=%.3f p21_reward=%.3f p22_reward=%.3f",
      operator_pair_arms[0].pulls,
      operator_pair_arms[1].pulls,
      operator_pair_arms[2].pulls,
      operator_pair_arms[3].pulls,
      operator_pair_arms[4].pulls,
      operator_pair_arms[5].pulls,
      operator_pair_arms[6].pulls,
      operator_pair_arms[7].pulls,
      operator_pair_arms[8].pulls,
      static_cast<double>(operator_pair_arms[0].reward_sum),
      static_cast<double>(operator_pair_arms[1].reward_sum),
      static_cast<double>(operator_pair_arms[2].reward_sum),
      static_cast<double>(operator_pair_arms[3].reward_sum),
      static_cast<double>(operator_pair_arms[4].reward_sum),
      static_cast<double>(operator_pair_arms[5].reward_sum),
      static_cast<double>(operator_pair_arms[6].reward_sum),
      static_cast<double>(operator_pair_arms[7].reward_sum),
      static_cast<double>(operator_pair_arms[8].reward_sum));
  }

  void log_violation_diagnostics(const std::vector<f_t>& lhs)
  {
    if (!diagnostics_enabled) { return; }
    std::vector<std::pair<f_t, i_t>> violations;
    violations.reserve(static_cast<size_t>(n_cons));
    for (i_t c = 0; c < n_cons; ++c) {
      if (constraint_violated(c, lhs[c])) {
        violations.emplace_back(normalized_excess_of(c, lhs[c]), c);
      }
    }
    const size_t count = std::min<size_t>(8, violations.size());
    std::partial_sort(violations.begin(),
                      violations.begin() + count,
                      violations.end(),
                      [](const auto& lhs_entry, const auto& rhs_entry) {
                        return lhs_entry.first > rhs_entry.first;
                      });
    for (size_t rank = 0; rank < count; ++rank) {
      const i_t c      = violations[rank].second;
      const char* name = static_cast<size_t>(c) < problem_ptr->row_names.size()
                           ? problem_ptr->row_names[c].c_str()
                           : "";
      CUOPT_LOG_INFO(
        "CPU_LNS_VIOLATION rank=%lu constraint=%d name=%s degree=%d lhs=%.9e lb=%.9e "
        "ub=%.9e excess=%.9e normalized_excess=%.9e weight=%.6f",
        rank,
        c,
        name,
        offsets[c + 1] - offsets[c],
        lhs[c],
        cons_lb[c],
        cons_ub[c],
        excess_of(c, lhs[c]),
        normalized_excess_of(c, lhs[c]),
        constraint_weight(c));
    }
  }

  bool finalize(solution_t<i_t, f_t>& solution, std::vector<f_t>& x)
  {
    solution.copy_new_assignment(x);
    if (!solution.compute_feasibility()) { return false; }
    if (n_int == 0) { return true; }
    solution_t<i_t, f_t> rounded(solution);
    if (rounded.round_nearest()) { solution.copy_from(rounded); }
    return true;
  }

  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;
  i_t n_vars{0};
  i_t n_cons{0};
  i_t n_int{0};
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tols{};

  std::vector<i_t> offsets;
  std::vector<i_t> variables;
  std::vector<f_t> coefficients;
  std::vector<i_t> rev_offsets;
  std::vector<i_t> rev_constraints;
  std::vector<f_t> rev_coefficients;
  std::vector<f_t> cons_lb;
  std::vector<f_t> cons_ub;
  std::vector<f_t> objective;
  std::vector<f_t> var_lb;
  std::vector<f_t> var_ub;
  std::vector<char> var_class;
  std::vector<i_t> active_variables;
  std::vector<i_t> objective_variables;
  std::vector<f_t> row_inf_norm;
  std::vector<f_t> row_scale;
  std::vector<f_t> lp_reference;
  std::vector<f_t> current_lhs;

  std::vector<char> ruined;
  std::vector<i_t> ruin_vars;
  std::vector<i_t> repair_order;
  std::vector<f_t> old_values;
  std::vector<i_t> affected_constraints;
  std::vector<f_t> old_lhs;
  std::vector<i_t> constraint_stamp;
  std::vector<i_t> affected_position;
  i_t constraint_generation{0};

  std::vector<f_t> work_lb;
  std::vector<f_t> work_ub;
  std::vector<f_t> next_lb;
  std::vector<f_t> next_ub;
  std::vector<f_t> min_activity;
  std::vector<f_t> max_activity;
  std::vector<f_t> repair_values;

  std::vector<char> seed_cons_active;
  std::vector<char> seed_cons_next;
  std::vector<char> seed_var_active;
  std::vector<i_t> seed_current_constraints;
  std::vector<i_t> seed_next_constraints;
  std::vector<i_t> seed_active_variables;
  std::vector<f_t> seed_min_activity;
  std::vector<f_t> seed_max_activity;

  std::vector<i_t> candidate_stamp;
  std::vector<f_t> candidate_dissim;
  std::vector<f_t> candidate_dot;
  std::vector<f_t> candidate_norm_seed;
  std::vector<f_t> candidate_norm_other;
  std::vector<i_t> candidate_intersection;
  std::vector<i_t> touched_candidates;
  std::vector<std::pair<f_t, i_t>> scored_candidates;
  i_t candidate_generation{0};
  std::vector<i_t> projection_stamp;
  i_t projection_generation{0};

  std::vector<i_t> violated_constraints;
  std::vector<i_t> violated_position;
  std::vector<f_t> tardiness;
  std::vector<uint64_t> tardiness_epoch;
  std::vector<char> tardiness_violated;
  uint64_t weight_epoch{0};

  std::array<arm_stats_t, N_RUIN_ARMS> ruin_arms{};
  std::array<arm_stats_t, N_REPAIR_ARMS> repair_arms{};
  std::array<arm_stats_t, N_OPERATOR_PAIRS> operator_pair_arms{};
  ablation_config_t ablation;
  diagnostics_t diagnostics;
  bool diagnostics_enabled{false};
  bool sa_excursion_active{false};
  double last_objective_log_s{-std::numeric_limits<double>::infinity()};
  f_t solve_elapsed_offset{0};
  std::mt19937 rng;
  timer_t lns_timer{0.};

  std::mutex inject_mutex;
  std::vector<f_t> injected_assignment;
  std::vector<f_t> injected_lp_reference;
  f_t injected_objective{std::numeric_limits<f_t>::infinity()};
  bool has_injected_incumbent{false};
  bool has_injected_lp{false};
};

}  // namespace cuopt::linear_programming::detail
