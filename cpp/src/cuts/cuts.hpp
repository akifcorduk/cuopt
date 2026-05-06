/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_vector.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/user_problem.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <future>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <cmath>
#include <cstdint>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
struct clique_table_t;
}

namespace cuopt::linear_programming::dual_simplex {

enum cut_type_t : int8_t {
  MIXED_INTEGER_GOMORY   = 0,
  MIXED_INTEGER_ROUNDING = 1,
  KNAPSACK               = 2,
  CHVATAL_GOMORY         = 3,
  CLIQUE                 = 4,
  IMPLIED_BOUND          = 5,
  MAX_CUT_TYPE           = 6
};

// P0-1: Per-cut-type per-event aggregate counters. Bumped at every accept /
// reject site in cut_pool_t::add_cut and cut_pool_t::score_cuts so generation
// and selection can be inspected per cut type per phase. See
// cpp/src/cuts/IMPROVEMENTS.md (Tier P0 / P0-1).
enum cut_pool_event_t : int8_t {
  // Events recorded inside add_cut (one bump per call to add_cut).
  CUT_EVENT_ADD_ACCEPT   = 0,  // cut stored in the pool
  CUT_EVENT_ADD_VARS_OOR = 1,  // cut had a column index >= original_vars_
  CUT_EVENT_ADD_EMPTY    = 2,  // cut had empty support after squeeze
  CUT_EVENT_ADD_DUP      = 3,  // P2-2: at-insert duplicate detection
  CUT_EVENT_ADD_NF_RHS   = 4,  // P2-1: non-finite rhs
  CUT_EVENT_ADD_NF_COEF  = 5,  // P2-1: non-finite coefficient
  CUT_EVENT_ADD_DEGEN    = 6,  // P2-1: degenerate (zero) norm
  CUT_EVENT_ADD_OVER_CAP = 7,  // generator already filled per-type per-pass cap
  // Events recorded inside score_cuts (per pool entry per call).
  CUT_EVENT_SCORE_ACCEPT    = 8,   // cut selected into best_cuts_
  CUT_EVENT_SCORE_THRESHOLD = 9,   // score below min_cut_distance_
  CUT_EVENT_SCORE_PARALLEL  = 10,  // rejected by orthogonality filter
  CUT_EVENT_SCORE_AGED      = 11,  // P1-1: dropped by drop_cuts
  CUT_EVENT_SCORE_DUP       = 12,  // removed by check_for_duplicate_cuts
  MAX_CUT_EVENT             = 13
};

struct cut_pool_stats_t {
  std::array<std::array<int64_t, MAX_CUT_EVENT>, MAX_CUT_TYPE> counts{};

  void reset()
  {
    for (auto& row : counts) {
      row.fill(0);
    }
  }

  void inc(cut_type_t cut_type, cut_pool_event_t event)
  {
    counts[static_cast<int>(cut_type)][static_cast<int>(event)]++;
  }

  int64_t get(cut_type_t cut_type, cut_pool_event_t event) const
  {
    return counts[static_cast<int>(cut_type)][static_cast<int>(event)];
  }
};

template <typename f_t>
struct cut_gap_closure_t {
  f_t initial_gap{0.0};
  f_t final_gap{0.0};
  f_t gap_closed{0.0};
  f_t gap_closed_ratio{0.0};
};

template <typename f_t>
cut_gap_closure_t<f_t> compute_cut_gap_closure(f_t objective_reference,
                                               f_t objective_before_cuts,
                                               f_t objective_after_cuts)
{
  const f_t initial_gap      = std::abs(objective_reference - objective_before_cuts);
  const f_t final_gap        = std::abs(objective_reference - objective_after_cuts);
  const f_t gap_closed       = initial_gap - final_gap;
  constexpr f_t eps          = static_cast<f_t>(1e-12);
  const f_t gap_closed_ratio = initial_gap > eps ? gap_closed / initial_gap : static_cast<f_t>(0.0);
  return {initial_gap, final_gap, gap_closed, gap_closed_ratio};
}

template <typename i_t, typename f_t>
struct probing_implied_bound_t {
  // Probing implications stored in CSR format, indexed by binary variable x_j.
  //
  // "zero" = implications discovered when probing x_j = 0.
  // "one"  = implications discovered when probing x_j = 1.
  //
  // For a binary variable x_j, the range
  //   zero_offsets[j] .. zero_offsets[j+1]
  // indexes into the flat arrays zero_variables, zero_lower_bound, zero_upper_bound.
  //
  // For each position p in that range:
  //   zero_variables[p]    = i if variable y_i bounds were tightened
  //                          when x_j was fixed to 0 and constraints were propagated.
  //   zero_lower_bound[p]  = tightened lower bound on y_i (i.e., x_j = 0  =>  y_i >=
  //   zero_lower_bound[p]). zero_upper_bound[p]  = tightened upper bound on y_i (i.e., x_j = 0  =>
  //   y_i <= zero_upper_bound[p]).
  //
  // The one arrays are analogous for probing x_j = 1.
  //
  // Non-binary variables have empty ranges (zero_offsets[j] == zero_offsets[j+1]).
  // Offsets vectors have size num_cols + 1.

  probing_implied_bound_t() = default;

  probing_implied_bound_t(i_t num_cols)
    : zero_offsets(num_cols + 1, 0), one_offsets(num_cols + 1, 0)
  {
  }

  std::vector<i_t> zero_offsets;
  std::vector<i_t> zero_variables;
  std::vector<f_t> zero_lower_bound;
  std::vector<f_t> zero_upper_bound;

  std::vector<i_t> one_offsets;
  std::vector<i_t> one_variables;
  std::vector<f_t> one_lower_bound;
  std::vector<f_t> one_upper_bound;
};

template <typename i_t, typename f_t>
struct inequality_t {
  inequality_t() : vector(), rhs(0.0) {}
  inequality_t(i_t num_cols) : vector(num_cols, 0), rhs(0.0) {}
  inequality_t(csr_matrix_t<i_t, f_t>& A, i_t row, f_t rhs_value) : vector(A, row), rhs(rhs_value)
  {
  }
  sparse_vector_t<i_t, f_t> vector;
  f_t rhs;

  void push_back(i_t j, f_t x)
  {
    vector.i.push_back(j);
    vector.x.push_back(x);
  }
  void clear()
  {
    vector.i.clear();
    vector.x.clear();
  }
  void reserve(size_t n)
  {
    vector.i.reserve(n);
    vector.x.reserve(n);
  }
  size_t size() const { return vector.i.size(); }
  i_t index(i_t k) const { return vector.i[k]; }
  f_t coeff(i_t k) const { return vector.x[k]; }
  void negate()
  {
    vector.negate();
    rhs *= -1.0;
  }
  void sort() { vector.sort(); }
  void squeeze(inequality_t<i_t, f_t>& out) const
  {
    vector.squeeze(out.vector);
    out.rhs = rhs;
  }
  void scale(f_t factor)
  {
    vector.scale(factor);
    rhs *= factor;
  }
  void print() const
  {
    for (i_t k = 0; k < size(); k++) {
      printf("%g x%d ", coeff(k), index(k));
    }
    printf("\nrhs %g\n", rhs);
  }
};

template <typename i_t, typename f_t>
struct cut_info_t {
  bool has_cuts() const
  {
    i_t total_cuts = 0;
    for (i_t i = 0; i < MAX_CUT_TYPE; i++) {
      total_cuts += num_cuts[i];
    }
    return total_cuts > 0;
  }
  void record_cut_types(const std::vector<cut_type_t>& cut_types)
  {
    for (cut_type_t cut_type : cut_types) {
      num_cuts[static_cast<int>(cut_type)]++;
    }
  }
  const char* cut_type_names[MAX_CUT_TYPE] = {"Gomory        ",
                                              "MIR           ",
                                              "Knapsack      ",
                                              "Strong CG     ",
                                              "Clique        ",
                                              "Implied Bounds"};
  std::array<i_t, MAX_CUT_TYPE> num_cuts   = {0};
};

template <typename i_t, typename f_t>
void print_cut_info(const simplex_solver_settings_t<i_t, f_t>& settings,
                    const cut_info_t<i_t, f_t>& cut_info)
{
  if (cut_info.has_cuts()) {
    for (i_t i = 0; i < MAX_CUT_TYPE; i++) {
      settings.log.printf("%s cuts : %d\n", cut_info.cut_type_names[i], cut_info.num_cuts[i]);
    }
  }
}

template <typename i_t, typename f_t>
void print_cut_types(const std::string& prefix,
                     const std::vector<cut_type_t>& cut_types,
                     const simplex_solver_settings_t<i_t, f_t>& settings)
{
  cut_info_t<i_t, f_t> cut_info;
  cut_info.record_cut_types(cut_types);
  settings.log.printf("%s: ", prefix.c_str());
  for (i_t i = 0; i < MAX_CUT_TYPE; i++) {
    settings.log.printf("%s cuts: %d\n", cut_info.cut_type_names[i], cut_info.num_cuts[i]);
  }
}

template <typename f_t>
f_t fractional_part(f_t a)
{
  return a - std::floor(a);
}

template <typename f_t>
bool add_work_estimate(f_t accesses,
                       f_t* work_estimate,
                       f_t max_work_estimate,
                       bool* work_limit_reached = nullptr)
{
  if (work_estimate == nullptr) { return false; }
  *work_estimate += accesses;
  const bool over_work_limit = *work_estimate > max_work_estimate;
  if (over_work_limit && work_limit_reached != nullptr) { *work_limit_reached = true; }
  return over_work_limit;
}

// Computes a permutation of a score vector that puts the highest scores first
template <typename i_t, typename f_t>
void best_score_first_permutation(std::vector<f_t>& scores, std::vector<i_t>& permutation)
{
  if (permutation.size() != scores.size()) { permutation.resize(scores.size()); }
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(
    permutation.begin(), permutation.end(), [&](i_t a, i_t b) { return scores[a] > scores[b]; });
}

// Computes a permutation of a score vector that puts the highest score last
template <typename i_t, typename f_t>
void best_score_last_permutation(std::vector<f_t>& scores, std::vector<i_t>& permutation)
{
  if (permutation.size() != scores.size()) { permutation.resize(scores.size()); }
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(
    permutation.begin(), permutation.end(), [&](i_t a, i_t b) { return scores[a] < scores[b]; });
}

// Routines for verifying cuts against a saved solution
template <typename i_t, typename f_t>
void read_saved_solution_for_cut_verification(const lp_problem_t<i_t, f_t>& lp,
                                              const simplex_solver_settings_t<i_t, f_t>& settings,
                                              std::vector<f_t>& saved_solution);

template <typename i_t, typename f_t>
void write_solution_for_cut_verification(const lp_problem_t<i_t, f_t>& lp,
                                         const std::vector<f_t>& solution);

template <typename i_t, typename f_t>
void verify_cuts_against_saved_solution(const csr_matrix_t<i_t, f_t>& cuts,
                                        const std::vector<f_t>& cut_rhs,
                                        const std::vector<f_t>& saved_solution);

// Test-only helper to run the production maximal-clique algorithm used by clique cuts.
// adjacency_list must contain local vertex indices in [0, n_vertices).
std::vector<std::vector<int>> find_maximal_cliques_for_test(
  const std::vector<std::vector<int>>& adjacency_list,
  const std::vector<double>& weights,
  double min_weight,
  int max_calls,
  double time_limit);

template <typename i_t, typename f_t>
class cut_pool_t {
 public:
  cut_pool_t(i_t original_vars, const simplex_solver_settings_t<i_t, f_t>& settings)
    : original_vars_(original_vars),
      settings_(settings),
      cut_storage_(0, original_vars, 0),
      rhs_storage_(0),
      cut_age_(0),
      cut_type_(0),
      scored_cuts_(0)
  {
  }

  // Add a cut in the form: cut'*x >= rhs.
  // We expect that the cut is violated by the current relaxation xstar
  // cut'*xstart < rhs
  void add_cut(cut_type_t cut_type, const inequality_t<i_t, f_t>& cut);

  // P1-2: when var_types and objective are non-null, the per-cut sort key
  // becomes the SCIP/Mops Achterberg-style additive composite:
  //
  //   score = efficacy_weight_       * efficacy
  //         + integer_support_weight_ * integer_support_fraction
  //         + obj_parallelism_weight_ * objective_parallelism
  //
  // where each component is paper §2.2 / Eq 6 / Eq 19 / Eq 25. When either
  // pointer is null, or sizes don't match the pool's variable space, the
  // corresponding term collapses to 0 and the score reduces to pure
  // efficacy (= the pre-P1-2 baseline). The hard `min_cut_distance_` floor
  // is still applied to raw efficacy so cuts that are barely violated are
  // rejected regardless of integer-support / obj-parallelism contribution.
  void score_cuts(std::vector<f_t>& x_relax,
                  const std::vector<variable_type_t>* var_types = nullptr,
                  const std::vector<f_t>* objective             = nullptr);

  // We return the cuts in the form best_cuts*x <= best_rhs
  i_t get_best_cuts(csr_matrix_t<i_t, f_t>& best_cuts,
                    std::vector<f_t>& best_rhs,
                    std::vector<cut_type_t>& best_cut_types);

  void age_cuts();

  void drop_cuts();

  i_t pool_size() const { return cut_storage_.m; }

  void print_cutpool_types() { print_cut_types("In cut pool", cut_type_, settings_); }

  void check_for_duplicate_cuts();

  // P0-1: stats lifecycle. Callers reset around each generation phase / each
  // score_cuts call and then call the matching log_*_stats_summary helper.
  void reset_stats() { stats_.reset(); }
  const cut_pool_stats_t& stats() const { return stats_; }
  void log_add_stats_summary(const char* label) const;
  void log_score_stats_summary(const char* label) const;

  // P0-2: gate per-cut "Reject cut row=... reason=..." log lines in add_cut /
  // score_cuts. Default off — these fire millions of times per solve and
  // fflush makes them very expensive on NFS-backed log files.
  void set_log_rejects(bool flag) { log_rejects_ = flag; }
  bool log_rejects() const { return log_rejects_; }

  // P1-1: aging knobs.
  //
  // pool_age_limit_  — cuts whose age has reached this value (or the
  //                    effective limit derived from it under overflow) are
  //                    evicted at the top of score_cuts. Mirrors HiGHS
  //                    `mip_pool_age_limit`. Default 5 (HiGHS default; the
  //                    Mops paper uses 3).
  // pool_soft_limit_ — soft cap on pool size; when exceeded, drop_cuts walks
  //                    the per-age histogram downward (never below 1) until
  //                    the post-eviction pool is at or below the cap.
  //                    Mirrors HiGHS `mip_pool_soft_limit`. Default 30000
  //                    (HiGHS default).
  //
  // Lifecycle convention follows HiGHS / Wesselmann–Suhl §3: every selected
  // cut is "touched" (age reset to 0) inside get_best_cuts, then age_cuts
  // increments every cut by one. A cut whose last selection was N rounds
  // ago has age N going into score_cuts.
  void set_pool_age_limit(i_t v) { pool_age_limit_ = v; }
  void set_pool_soft_limit(i_t v) { pool_soft_limit_ = v; }
  i_t pool_age_limit() const { return pool_age_limit_; }
  i_t pool_soft_limit() const { return pool_soft_limit_; }

  // P1-2: composite-score weights. Defaults match SCIP `sepa_cutsel_hybrid`
  // and the Achterberg-style composite that Wesselmann–Suhl Table 4
  // validates as fastest at 50% of instances and 89% solved.
  // Set obj_parallelism_weight_ to 0.0 to mirror SCIP `objparalfac=0.0` for
  // categories where objective parallelism is uninformative.
  void set_efficacy_weight(f_t v) { efficacy_weight_ = v; }
  void set_integer_support_weight(f_t v) { integer_support_weight_ = v; }
  void set_obj_parallelism_weight(f_t v) { obj_parallelism_weight_ = v; }
  f_t efficacy_weight() const { return efficacy_weight_; }
  f_t integer_support_weight() const { return integer_support_weight_; }
  f_t obj_parallelism_weight() const { return obj_parallelism_weight_; }

  // P1-3: score-tiered orthogonality filter. Unified on min_orthogonality
  // (= 1 - max_parallelism) so both tiers share the same units as the
  // existing settings_.cut_min_orthogonality knob.
  //
  // settings_.cut_min_orthogonality
  //                       — strict tier; applies to "ordinary" candidate
  //                         cuts. Equivalent to SCIP `minortho`, paper
  //                         `1 - p_max`. Default 0.9 (= max parallelism 0.1).
  //                         Read directly from settings_, so the user-facing
  //                         CLI/proto knob `mip_cut_min_orthogonality`
  //                         controls it without duplication here.
  // good_min_orthogonality_
  //                       — relaxed tier; applies when a candidate is "good"
  //                         (score >= good_cut_factor_ * top_round_score).
  //                         Equivalent to SCIP `1 - goodmaxparall`. Default
  //                         0.5 (matches the pre-P1-3 single threshold so
  //                         high-quality cuts in dense correlated families
  //                         aren't starved).
  // good_cut_factor_      — fractional threshold defining "good": a cut is
  //                         good iff its composite score is at least this
  //                         fraction of the top score in the round. SCIP
  //                         `GOODSCORE`, paper `skip_factor`. Default 0.9.
  //
  // The orthogonality scan runs for both tiers; only the threshold differs.
  // Good cuts must still satisfy the relaxed constraint — there is no
  // unconditional bypass.
  void set_good_min_orthogonality(f_t v) { good_min_orthogonality_ = v; }
  void set_good_cut_factor(f_t v) { good_cut_factor_ = v; }
  f_t good_min_orthogonality() const { return good_min_orthogonality_; }
  f_t good_cut_factor() const { return good_cut_factor_; }

  // Wall-clock deadline. The cut pool's heaviest loops are
  // `check_for_duplicate_cuts` (Tomlin-Welch O(m^2) in the worst case) and
  // `score_cuts` (orthogonality scan O(pool * best_cuts * nnz)). On
  // pathological instances (dense ImpBnd flood, e.g. splice1k1) either can
  // run for many minutes inside a single `score_cuts` call, blowing past
  // settings_.time_limit. Callers (B&B root cut loop) set this to the
  // wall-clock start of the solve; the pool then bails out of its long
  // loops once toc(start) >= settings_.time_limit. A negative value (the
  // default) disables the check so unit tests and any caller that does not
  // care about the deadline keep their previous semantics.
  void set_wall_clock_start(f_t start_time) { wall_clock_start_ = start_time; }

  // Per-generator-pass cap on accepted cuts of one type. The
  // `cut_generation_t` driver calls reset_stats() before each generator
  // (gomory / knapsack / mir / clique / implied_bound) so the
  // CUT_EVENT_ADD_ACCEPT counter doubles as the per-pass per-type accept
  // count. When that count reaches max_cuts_per_type_per_pass_, add_cut
  // short-circuits and bumps CUT_EVENT_ADD_OVER_CAP.
  //
  // This is the front-line defence against generator floods (1.13M
  // implied bounds on splice1k1 -> 580k+ ImpBnd cuts in one round). The
  // downstream Tomlin-Welch dedup and orthogonality scan are O(m^2) and
  // O(m * best_cuts * nnz) respectively; capping the inflow keeps both
  // bounded regardless of probing density. 5000 per type per pass × 6
  // types ≤ 30000 = pool_soft_limit_, so drop_cuts can keep the pool
  // bounded across passes too.
  //
  // Set to 0 or a negative value to disable the cap (revert to the
  // pre-cap behaviour where every generated cut enters the pool).
  void set_max_cuts_per_type_per_pass(i_t v) { max_cuts_per_type_per_pass_ = v; }
  i_t max_cuts_per_type_per_pass() const { return max_cuts_per_type_per_pass_; }

 private:
  bool over_deadline() const
  {
    if (wall_clock_start_ < 0.0) { return false; }
    return toc(wall_clock_start_) >= settings_.time_limit;
  }
  f_t cut_distance(i_t row, const std::vector<f_t>& x, f_t& cut_violation, f_t& cut_norm);
  f_t cut_density(i_t row);
  f_t cut_orthogonality(i_t i, i_t j);
  // P1-2: composite-score components.
  // integer_support_fraction = |supp(a) ∩ NI| / |supp(a)|   (paper Eq 25)
  // obj_parallelism          = |aᵀc| / (‖a‖ ‖c‖)            (paper Eq 19)
  // Both return 0 when the relevant input is missing or sizes don't match.
  f_t cut_integer_support_fraction(i_t row, const std::vector<variable_type_t>& var_types) const;
  f_t cut_obj_parallelism(i_t row,
                          const std::vector<f_t>& objective,
                          f_t cut_norm,
                          f_t obj_norm) const;

  i_t original_vars_;
  const simplex_solver_settings_t<i_t, f_t>& settings_;

  csr_matrix_t<i_t, f_t> cut_storage_;
  std::vector<f_t> rhs_storage_;
  std::vector<i_t> cut_age_;
  std::vector<cut_type_t> cut_type_;

  i_t scored_cuts_;
  std::vector<f_t> cut_distances_;
  std::vector<f_t> cut_norms_;
  std::vector<f_t> cut_orthogonality_;
  std::vector<f_t> cut_scores_;
  std::vector<i_t> best_cuts_;
  const f_t min_cut_distance_{1e-4};

  cut_pool_stats_t stats_{};
  bool log_rejects_{false};

  // P1-1: pool_age_limit_ defaults match HiGHS (`mip_pool_age_limit` = 5).
  // pool_soft_limit_ used to match HiGHS' 30000, but with the per-type
  // per-pass accept cap (max_cuts_per_type_per_pass_ = 5000 × 6 cut types
  // = 30000 absolute ceiling per pass) the 30000 soft cap could never
  // bite during a pass — it only mattered for inter-pass survivors.
  // Tightening to 20000 makes drop_cuts more aggressive on inter-pass
  // survivors without conflicting with fresh adds (age-0 cuts are still
  // unconditionally protected, so the in-pass max of 30000 is unchanged).
  i_t pool_age_limit_{5};
  i_t pool_soft_limit_{20000};

  // P1-2: SCIP `sepa_cutsel_hybrid` defaults — `efficacyfac=1.0`,
  // `intsupportfac=0.1`, `objparalfac=0.1`. Paper §4 / Figure 4: this
  // composite (the Achterberg-style scorer) is fastest on 50% of instances
  // and solves 89% to optimality vs 46% / 85% for plain distance.
  f_t efficacy_weight_{1.0};
  f_t integer_support_weight_{0.1};
  f_t obj_parallelism_weight_{0.1};

  // P1-3: score-tiered orthogonality. Defaults (unified on min_orthogonality):
  //   strict tier            = settings_.cut_min_orthogonality (default 0.9,
  //                            SCIP minortho, paper 1-p_max). Read live from
  //                            settings_ on every score round.
  //   good_min_orthogonality_ = 0.5  (SCIP 1-goodmaxparall; equals the
  //                                   pre-P1-3 single threshold so the
  //                                   relaxed tier reproduces baseline
  //                                   behaviour for the top scoring cuts)
  //   good_cut_factor_        = 0.9  (SCIP GOODSCORE, paper skip_factor)
  // See score_cuts in cuts.cpp for how these combine: a candidate's
  // effective min_orthogonality is good_min_orthogonality_ when its score is
  // at least good_cut_factor_ * top_round_score, otherwise the strict
  // settings_.cut_min_orthogonality.
  f_t good_min_orthogonality_{0.5};
  f_t good_cut_factor_{0.9};

  // Wall-clock anchor for over_deadline(). Negative => no deadline check.
  f_t wall_clock_start_{-1.0};

  // Per-cut-type per-generator-pass accept cap. 5000 keeps the cumulative
  // worst-case accept count under pool_soft_limit_ (= 30000) across the
  // 6 cut types, so drop_cuts can still bring the pool under cap between
  // rounds. <= 0 disables the cap.
  i_t max_cuts_per_type_per_pass_{5000};
};

template <typename i_t, typename f_t>
class knapsack_generation_t {
 public:
  knapsack_generation_t(const lp_problem_t<i_t, f_t>& lp,
                        const simplex_solver_settings_t<i_t, f_t>& settings,
                        csr_matrix_t<i_t, f_t>& Arow,
                        const std::vector<i_t>& new_slacks,
                        const std::vector<variable_type_t>& var_types);

  i_t generate_knapsack_cut(const lp_problem_t<i_t, f_t>& lp,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<i_t>& new_slacks,
                            const std::vector<variable_type_t>& var_types,
                            const std::vector<f_t>& xstar,
                            i_t knapsack_row,
                            inequality_t<i_t, f_t>& cut);

  i_t num_knapsack_constraints() const { return knapsack_constraints_.size(); }
  const std::vector<i_t>& get_knapsack_constraints() const { return knapsack_constraints_; }

 private:
  void restore_complemented(const std::vector<i_t>& complemented_variables)
  {
    for (i_t j : complemented_variables) {
      is_complemented_[j] = 0;
    }
  }
  bool is_minimal_cover(f_t cover_sum, f_t beta, const std::vector<f_t>& cover_coefficients);

  void minimal_cover_and_partition(const inequality_t<i_t, f_t>& knapsack_inequality,
                                   const inequality_t<i_t, f_t>& negated_base_cut,
                                   const std::vector<f_t>& xstar,
                                   inequality_t<i_t, f_t>& minimal_cover_cut,
                                   std::vector<i_t>& c1_partition,
                                   std::vector<i_t>& c2_partition);

  void lift_knapsack_cut(const inequality_t<i_t, f_t>& knapsack_inequality,
                         const inequality_t<i_t, f_t>& base_cut,
                         const std::vector<i_t>& c1_partition,
                         const std::vector<i_t>& c2_partition,
                         inequality_t<i_t, f_t>& lifted_cut);

  // Generate a heuristic solution to the 0-1 knapsack problem
  f_t greedy_knapsack_problem(const std::vector<f_t>& values,
                              const std::vector<f_t>& weights,
                              f_t rhs,
                              std::vector<f_t>& solution);

  // Solve a 0-1 knapsack problem using dynamic programming
  f_t solve_knapsack_problem(const std::vector<f_t>& values,
                             const std::vector<f_t>& weights,
                             f_t rhs,
                             std::vector<f_t>& solution);

  f_t exact_knapsack_problem_integer_values_fraction_values(const std::vector<i_t>& values,
                                                            const std::vector<f_t>& weights,
                                                            f_t rhs,
                                                            std::vector<f_t>& solution);

  std::vector<i_t> is_slack_;
  std::vector<i_t> knapsack_constraints_;
  std::vector<i_t> is_complemented_;
  std::vector<i_t> is_marked_;
  std::vector<f_t> workspace_;
  std::vector<f_t> complemented_xstar_;
  const simplex_solver_settings_t<i_t, f_t>& settings_;
};

// Forward declarations
template <typename i_t, typename f_t>
class mixed_integer_rounding_cut_t;

template <typename i_t, typename f_t>
class variable_bounds_t;

template <typename i_t, typename f_t>
class cut_generation_t {
 public:
  cut_generation_t(
    cut_pool_t<i_t, f_t>& cut_pool,
    const lp_problem_t<i_t, f_t>& lp,
    const simplex_solver_settings_t<i_t, f_t>& settings,
    csr_matrix_t<i_t, f_t>& Arow,
    const std::vector<i_t>& new_slacks,
    const std::vector<variable_type_t>& var_types,
    const user_problem_t<i_t, f_t>& user_problem,
    const probing_implied_bound_t<i_t, f_t>& probing_implied_bound,
    std::shared_ptr<detail::clique_table_t<i_t, f_t>> clique_table                      = nullptr,
    std::future<std::shared_ptr<detail::clique_table_t<i_t, f_t>>>* clique_table_future = nullptr,
    std::atomic<bool>* signal_extend                                                    = nullptr)
    : cut_pool_(cut_pool),
      knapsack_generation_(lp, settings, Arow, new_slacks, var_types),
      user_problem_(user_problem),
      probing_implied_bound_(probing_implied_bound),
      clique_table_(std::move(clique_table)),
      clique_table_future_(clique_table_future),
      signal_extend_(signal_extend)
  {
  }

  bool generate_cuts(const lp_problem_t<i_t, f_t>& lp,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     csr_matrix_t<i_t, f_t>& Arow,
                     const std::vector<i_t>& new_slacks,
                     const std::vector<variable_type_t>& var_types,
                     basis_update_mpf_t<i_t, f_t>& basis_update,
                     const std::vector<f_t>& xstar,
                     const std::vector<f_t>& ystar,
                     const std::vector<f_t>& zstar,
                     const std::vector<i_t>& basic_list,
                     const std::vector<i_t>& nonbasic_list,
                     variable_bounds_t<i_t, f_t>& variable_bounds,
                     f_t start_time);

 private:
  // Generate all mixed integer gomory cuts
  void generate_gomory_cuts(const lp_problem_t<i_t, f_t>& lp,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<i_t>& new_slacks,
                            const std::vector<variable_type_t>& var_types,
                            basis_update_mpf_t<i_t, f_t>& basis_update,
                            const std::vector<f_t>& xstar,
                            const std::vector<i_t>& basic_list,
                            const std::vector<i_t>& nonbasic_list);

  // Generate all mixed integer rounding cuts
  void generate_mir_cuts(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         csr_matrix_t<i_t, f_t>& Arow,
                         const std::vector<i_t>& new_slacks,
                         const std::vector<variable_type_t>& var_types,
                         const std::vector<f_t>& xstar,
                         const std::vector<f_t>& ystar,
                         variable_bounds_t<i_t, f_t>& variable_bounds);

  // Generate all knapsack cuts
  void generate_knapsack_cuts(const lp_problem_t<i_t, f_t>& lp,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              csr_matrix_t<i_t, f_t>& Arow,
                              const std::vector<i_t>& new_slacks,
                              const std::vector<variable_type_t>& var_types,
                              const std::vector<f_t>& xstar,
                              f_t start_time);

  // Generate clique cuts from conflict graph cliques
  bool generate_clique_cuts(const lp_problem_t<i_t, f_t>& lp,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            const std::vector<variable_type_t>& var_types,
                            const std::vector<f_t>& xstar,
                            const std::vector<f_t>& reduced_costs,
                            f_t start_time);

  // Generate implied bounds cuts from probing implications
  void generate_implied_bound_cuts(const lp_problem_t<i_t, f_t>& lp,
                                   const simplex_solver_settings_t<i_t, f_t>& settings,
                                   const std::vector<variable_type_t>& var_types,
                                   const std::vector<f_t>& xstar,
                                   f_t start_time);

  cut_pool_t<i_t, f_t>& cut_pool_;
  knapsack_generation_t<i_t, f_t> knapsack_generation_;
  const user_problem_t<i_t, f_t>& user_problem_;
  const probing_implied_bound_t<i_t, f_t>& probing_implied_bound_;
  std::shared_ptr<detail::clique_table_t<i_t, f_t>> clique_table_;
  std::future<std::shared_ptr<detail::clique_table_t<i_t, f_t>>>* clique_table_future_{nullptr};
  std::atomic<bool>* signal_extend_{nullptr};
};

template <typename i_t, typename f_t>
class scratch_pad_t {
 public:
  scratch_pad_t(i_t num_vars) : workspace_(num_vars, 0.0), mark_(num_vars, 0)
  {
    indices_.reserve(num_vars);
  }

  // O(1) to add a value to the pad
  void add_to_pad(i_t j, f_t value)
  {
    workspace_[j] += value;
    if (!mark_[j]) {
      mark_[j] = 1;
      indices_.push_back(j);
    }
  }

  // O(nz) to clear the pad
  void clear_pad()
  {
    for (i_t j : indices_) {
      workspace_[j] = 0.0;
      mark_[j]      = 0;
    }
    indices_.clear();
  }

  // O(nz) to get the pad
  void get_pad(std::vector<i_t>& indices, std::vector<f_t>& values)
  {
    indices.reserve(indices_.size());
    values.reserve(indices_.size());
    indices.clear();
    values.clear();
    const i_t nz = indices_.size();
    for (i_t k = 0; k < nz; k++) {
      const i_t j   = indices_[k];
      const f_t val = workspace_[j];
      if (val != 0.0) {
        indices.push_back(j);
        values.push_back(val);
      }
    }
  }

 private:
  std::vector<f_t> workspace_;
  std::vector<i_t> mark_;
  std::vector<i_t> indices_;
};

template <typename i_t, typename f_t>
class mixed_integer_gomory_cut_t {
 public:
  mixed_integer_gomory_cut_t() {}
};

template <typename i_t, typename f_t>
class tableau_equality_t {
 public:
  tableau_equality_t(const lp_problem_t<i_t, f_t>& lp,
                     basis_update_mpf_t<i_t, f_t>& basis_update,
                     const std::vector<i_t>& nonbasic_list)
    : b_bar_(lp.num_rows, 0.0),
      nonbasic_mark_(lp.num_cols, 0),
      x_workspace_(lp.num_cols, 0.0),
      x_mark_(lp.num_cols, 0),
      c_workspace_(lp.num_cols, 0.0)
  {
    basis_update.b_solve(lp.rhs, b_bar_);
    for (i_t j : nonbasic_list) {
      nonbasic_mark_[j] = 1;
    }
  }

  // Generates the base inequalities: C*x == d that will be turned into cuts
  i_t generate_base_equality(const lp_problem_t<i_t, f_t>& lp,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             csr_matrix_t<i_t, f_t>& Arow,
                             const std::vector<variable_type_t>& var_types,
                             basis_update_mpf_t<i_t, f_t>& basis_update,
                             const std::vector<f_t>& xstar,
                             const std::vector<i_t>& basic_list,
                             const std::vector<i_t>& nonbasic_list,
                             i_t i,
                             inequality_t<i_t, f_t>& inequality);

 private:
  std::vector<f_t> b_bar_;
  std::vector<i_t> nonbasic_mark_;
  std::vector<f_t> x_workspace_;
  std::vector<i_t> x_mark_;
  std::vector<f_t> c_workspace_;
};

template <typename i_t, typename f_t>
class variable_bounds_t {
 public:
  variable_bounds_t(const lp_problem_t<i_t, f_t>& lp,
                    const simplex_solver_settings_t<i_t, f_t>& settings,
                    const std::vector<variable_type_t>& var_types,
                    const csr_matrix_t<i_t, f_t>& Arow,
                    const std::vector<i_t>& new_slacks);

  std::vector<i_t> upper_offsets;
  std::vector<i_t> upper_variables;
  std::vector<f_t> upper_weights;
  std::vector<f_t> upper_biases;

  std::vector<i_t> lower_offsets;
  std::vector<i_t> lower_variables;
  std::vector<f_t> lower_weights;
  std::vector<f_t> lower_biases;

  void resize(i_t new_num_cols)
  {
    const i_t current_upper_nz = upper_offsets.back();
    upper_offsets.resize(new_num_cols + 1, current_upper_nz);
    const i_t current_lower_nz = lower_offsets.back();
    lower_offsets.resize(new_num_cols + 1, current_lower_nz);
  }

 private:
  f_t lower_activity(f_t lower_bound, f_t upper_bound, f_t coefficient)
  {
    return (coefficient > 0.0 ? lower_bound : upper_bound) * coefficient;
  }

  f_t upper_activity(f_t lower_bound, f_t upper_bound, f_t coefficient)
  {
    return (coefficient > 0.0 ? upper_bound : lower_bound) * coefficient;
  }

  // Returns the lower activity adjusted for the number of lower inf variables
  // adjusted_lower_activity = { activity - lower_activity_i - lower_activity_j, if num_lower_inf =
  // 0
  //                           { activity - lower_activity_i                   , if num_lower_inf =
  //                           1, lower_activity_j = -inf { activity - lower_activity_j , if
  //                           num_lower_inf = 1, lower_activity_i != -inf { activity , if
  //                           num_lower_inf = 2, lower_activity_i = lower_activity_j = -inf { -inf
  //                           , if num_lower_inf > 2
  f_t adjusted_lower_activity(f_t activity,
                              i_t num_lower_inf,
                              f_t lower_activity_i,
                              f_t lower_activity_j)
  {
    if (num_lower_inf == 0) {
      return activity - lower_activity_i - lower_activity_j;
    } else if (num_lower_inf == 1 && lower_activity_j == -inf) {
      return activity - lower_activity_i;
    } else if (num_lower_inf == 1 && lower_activity_i == -inf) {
      return activity - lower_activity_j;
    } else if (num_lower_inf == 2 && lower_activity_i == -inf && lower_activity_j == -inf) {
      return activity;
    } else {
      return -inf;
    }
  }

  // Returns the upper activity adjusted for the number of upper inf variables
  // adjusted_upper_activity = { activity - upper_activity_i - upper_activity_j, if num_upper_inf =
  // 0
  //                           { activity - upper_activity_i                   , if num_upper_inf =
  //                           1, upper_activity_j = inf { activity - upper_activity_j , if
  //                           num_upper_inf = 1, upper_activity_i != inf { activity , if
  //                           num_upper_inf = 2, upper_activity_i = upper_activity_j = inf { inf ,
  //                           if num_upper_inf > 2
  f_t adjusted_upper_activity(f_t activity,
                              i_t num_upper_inf,
                              f_t upper_activity_i,
                              f_t upper_activity_j)
  {
    if (num_upper_inf == 0) {
      return activity - upper_activity_i - upper_activity_j;
    } else if (num_upper_inf == 1 && upper_activity_j == inf) {
      return activity - upper_activity_i;
    } else if (num_upper_inf == 1 && upper_activity_i == inf) {
      return activity - upper_activity_j;
    } else if (num_upper_inf == 2 && upper_activity_i == inf && upper_activity_j == inf) {
      return activity;
    } else {
      return inf;
    }
  }

  std::vector<f_t> upper_activities_;
  std::vector<i_t> num_pos_inf_;
  std::vector<f_t> lower_activities_;
  std::vector<i_t> num_neg_inf_;

  std::vector<i_t> slack_map_;
};

template <typename i_t, typename f_t>
class complemented_mixed_integer_rounding_cut_t {
 public:
  complemented_mixed_integer_rounding_cut_t(const lp_problem_t<i_t, f_t>& lp,
                                            const simplex_solver_settings_t<i_t, f_t>& settings,
                                            const std::vector<i_t>& new_slacks);

  void compute_initial_scores_for_rows(const lp_problem_t<i_t, f_t>& lp,
                                       const simplex_solver_settings_t<i_t, f_t>& settings,
                                       const csr_matrix_t<i_t, f_t>& Arow,
                                       const std::vector<f_t>& xstar,
                                       const std::vector<f_t>& ystar,
                                       std::vector<f_t>& score);

  // Perform bound substitution for the continuous variables using simple bounds
  // and variable bounds. And bound substitution for the integer variables
  // using simple bounds.
  void bound_substitution(const lp_problem_t<i_t, f_t>& lp,
                          const variable_bounds_t<i_t, f_t>& variable_bounds,
                          const std::vector<variable_type_t>& var_types,
                          const std::vector<f_t>& xstar,
                          std::vector<f_t>& transformed_xstar);

  // Converts an inequality of the form: sum_j a_j x_j >= beta
  // with l_j <= x_j <= u_j into the form:
  // sum_{j not in L union U} d_j x_j + sum_{j in L} d_j v_j
  // + sum_{j in U} d_j w_j >= delta,
  // where v_j = x_j - l_j for j in L
  // and   w_j = u_j - x_j for j in U
  void transform_inequality(const variable_bounds_t<i_t, f_t>& variable_bounds,
                            const std::vector<variable_type_t>& var_type,
                            inequality_t<i_t, f_t>& inequality);

  // Converts an inequality of the form:
  // sum_{j not in L union U} d_j x_j + sum_{j in L} d_j v_j
  // + sum_{j in U} d_j w_j >= delta,
  // where v_j = x_j - l_j for j in L
  // and   w_j = u_j - x_j for j in U
  // back to the form: sum_j a_j x_j >= beta
  // with l_j <= x_j <= u_j
  void untransform_inequality(const variable_bounds_t<i_t, f_t>& variable_bounds,
                              const std::vector<variable_type_t>& var_type,
                              inequality_t<i_t, f_t>& inequality);

  bool cut_generation_heuristic(const inequality_t<i_t, f_t>& transformed_inequality,
                                const std::vector<variable_type_t>& var_types,
                                const std::vector<f_t>& transformed_xstar,
                                inequality_t<i_t, f_t>& transformed_cut,
                                f_t& work_estimate);

  bool scale_uncomplement_and_generate_cut(const std::vector<variable_type_t>& var_types,
                                           const std::vector<f_t>& transformed_xstar,
                                           const std::vector<i_t>& complemented_indices,
                                           const inequality_t<i_t, f_t>& complemented_inequality,
                                           f_t delta,
                                           inequality_t<i_t, f_t>& cut_delta,
                                           f_t& work_estimate);

  // This routine takes an inequality and generates the MIR cut
  bool generate_cut_nonnegative_maintain_indicies(const inequality_t<i_t, f_t>& inequality,
                                                  const std::vector<variable_type_t>& var_types,
                                                  inequality_t<i_t, f_t>& cut);

  f_t compute_violation(const inequality_t<i_t, f_t>& cut, const std::vector<f_t>& xstar);

  f_t new_upper(i_t j) const { return transformed_upper_[j]; }

  // Given a cut of the form sum_j d_j x_j >= beta
  // with l_j <= x_j <= u_j, try to remove coefficients d_j
  // with | d_j | < epsilon
  void remove_small_coefficients(const std::vector<f_t>& lower_bounds,
                                 const std::vector<f_t>& upper_bounds,
                                 inequality_t<i_t, f_t>& cut);

  void substitute_slacks(const lp_problem_t<i_t, f_t>& lp,
                         csr_matrix_t<i_t, f_t>& Arow,
                         inequality_t<i_t, f_t>& cut);

  // Combine the pivot row with the inequality to eliminate the variable j
  // The new inequality is returned in inequality and inequality_rhs
  // The multiplier for the pivot row is returned
  f_t combine_rows(const lp_problem_t<i_t, f_t>& lp,
                   csr_matrix_t<i_t, f_t>& Arow,
                   i_t j,
                   const inequality_t<i_t, f_t>& pivot_row,
                   inequality_t<i_t, f_t>& inequality);

  const f_t get_lb_star(i_t j) const { return lb_star_[j]; }
  const f_t get_ub_star(i_t j) const { return ub_star_[j]; }

  const i_t slack_rows(i_t j) const { return slack_rows_[j]; }
  const i_t slack_cols(i_t i) const { return slack_cols_[i]; }

  bool scale_and_generate_mir_cut(const std::vector<variable_type_t>& var_types,
                                  const std::vector<f_t>& transformed_xstar,
                                  const inequality_t<i_t, f_t>& inequality,
                                  f_t divisor,
                                  std::vector<inequality_t<i_t, f_t>>& cuts,
                                  std::vector<f_t>& violations,
                                  std::vector<f_t>& deltas);

  bool check_violation_and_add_cut(const inequality_t<i_t, f_t>& inequality,
                                   const std::vector<f_t>& xstar,
                                   f_t divisor,
                                   std::vector<inequality_t<i_t, f_t>>& cuts,
                                   std::vector<f_t>& violations,
                                   std::vector<f_t>& deltas);

 private:
  std::vector<i_t> is_slack_;
  std::vector<i_t>
    slack_rows_;  // slack_rows_[j] = i, if variable j is slack for row i, -1 is sentinal value
  std::vector<i_t>
    slack_cols_;  // slack_cols_[i] = j, if variable j is slack for row i  -1 is sentinal value

  std::vector<i_t> lb_variable_;
  std::vector<f_t> lb_star_;
  std::vector<i_t> ub_variable_;
  std::vector<f_t> ub_star_;

  std::vector<i_t> bound_changed_;
  std::vector<f_t> transformed_upper_;

  scratch_pad_t<i_t, f_t> scratch_pad_;
};

template <typename i_t, typename f_t>
class strong_cg_cut_t {
 public:
  strong_cg_cut_t(const lp_problem_t<i_t, f_t>& lp,
                  const std::vector<variable_type_t>& var_types,
                  const std::vector<f_t>& xstar);

  i_t generate_strong_cg_cut(const lp_problem_t<i_t, f_t>& lp,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             const std::vector<variable_type_t>& var_types,
                             const inequality_t<i_t, f_t>& inequality,
                             const std::vector<f_t>& xstar,
                             inequality_t<i_t, f_t>& cut);

  i_t remove_continuous_variables_integers_nonnegative(
    const lp_problem_t<i_t, f_t>& lp,
    const simplex_solver_settings_t<i_t, f_t>& settings,
    const std::vector<variable_type_t>& var_types,
    inequality_t<i_t, f_t>& inequality);

  void to_original_integer_variables(const lp_problem_t<i_t, f_t>& lp, inequality_t<i_t, f_t>& cut);

  i_t generate_strong_cg_cut_integer_only(const simplex_solver_settings_t<i_t, f_t>& settings,
                                          const std::vector<variable_type_t>& var_types,
                                          const inequality_t<i_t, f_t>& inequality,
                                          inequality_t<i_t, f_t>& cut);

 private:
  i_t generate_strong_cg_cut_helper(const std::vector<i_t>& indicies,
                                    const std::vector<f_t>& coefficients,
                                    f_t rhs,
                                    const std::vector<variable_type_t>& var_types,
                                    inequality_t<i_t, f_t>& cut);

  std::vector<i_t> transformed_variables_;
};

template <typename i_t, typename f_t>
i_t add_cuts(const simplex_solver_settings_t<i_t, f_t>& settings,
             const csr_matrix_t<i_t, f_t>& cuts,
             const std::vector<f_t>& cut_rhs,
             lp_problem_t<i_t, f_t>& lp,
             std::vector<i_t>& new_slacks,
             lp_solution_t<i_t, f_t>& solution,
             basis_update_mpf_t<i_t, f_t>& basis_update,
             std::vector<i_t>& basic_list,
             std::vector<i_t>& nonbasic_list,
             std::vector<variable_status_t>& vstatus,
             std::vector<f_t>& edge_norms);

template <typename i_t, typename f_t>
i_t remove_cuts(lp_problem_t<i_t, f_t>& lp,
                const simplex_solver_settings_t<i_t, f_t>& settings,
                f_t start_time,
                csr_matrix_t<i_t, f_t>& Arow,
                std::vector<i_t>& new_slacks,
                i_t original_rows,
                std::vector<variable_type_t>& var_types,
                std::vector<variable_status_t>& vstatus,
                std::vector<f_t>& edge_norms,
                std::vector<f_t>& x,
                std::vector<f_t>& y,
                std::vector<f_t>& z,
                std::vector<i_t>& basic_list,
                std::vector<i_t>& nonbasic_list,
                basis_update_mpf_t<i_t, f_t>& basis_update);

}  // namespace cuopt::linear_programming::dual_simplex
