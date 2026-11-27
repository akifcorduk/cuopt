/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// gprs_check <model.csr> <out.json> [--lp-time-limit <s>] [--mip-time-limit <s>]
//            [--known-objective <value>] [--threads <n>]
// Correctness check of the host engine on one instance, with HiGHS as the solver:
//   1. LP relaxation of the original model and of the reduced model. The reduced value must not
//      lie below the original one (no reduction relaxes the model), and not above the known
//      objective value of the instance (the reductions keep an optimal solution).
//   2. MIP solve of the reduced model with a time limit. A solution it finds is postsolved and
//      tested in the original model: rows, bounds, integrality and the objective value.
// The dump holds a minimisation, so --known-objective is given in that form.
// Exit code 0 when the checks ran (whatever they found), 1 on an input error.

#include <gprs/io.hpp>
#include <gprs/presolve.hpp>

#include <Highs.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

double now()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct solve_t {
  bool ran           = false;
  std::string status = "not run";
  bool has_solution  = false;
  double objective   = 0.;
  double dual_bound  = 0.;
  double seconds     = 0.;
  std::vector<double> x;
};

solve_t solve(const gprs::model_t& model, bool relaxation, double time_limit, int threads)
{
  solve_t out;
  out.ran = true;
  if (model.ncols == 0) {
    // Nothing to solve: the value is the offset; the rows of an empty model have no entries.
    out.status       = "empty";
    out.has_solution = true;
    out.objective    = model.objective_offset;
    out.dual_bound   = model.objective_offset;
    return out;
  }
  HighsLp lp;
  lp.num_col_           = model.ncols;
  lp.num_row_           = model.nrows;
  lp.sense_             = ObjSense::kMinimize;
  lp.offset_            = model.objective_offset;
  lp.col_cost_          = model.cost;
  lp.col_lower_         = model.col_lower;
  lp.col_upper_         = model.col_upper;
  lp.row_lower_         = model.row_lower;
  lp.row_upper_         = model.row_upper;
  lp.a_matrix_.format_  = MatrixFormat::kRowwise;
  lp.a_matrix_.num_col_ = model.ncols;
  lp.a_matrix_.num_row_ = model.nrows;
  lp.a_matrix_.start_.assign(model.row_start.begin(), model.row_start.end());
  lp.a_matrix_.index_.assign(model.col_index.begin(), model.col_index.end());
  lp.a_matrix_.value_ = model.value;
  if (!relaxation) {
    lp.integrality_.resize(model.ncols);
    for (int32_t j = 0; j < model.ncols; ++j) {
      lp.integrality_[j] = model.col_type[j] == (uint8_t)gprs::col_type_t::integer
                             ? HighsVarType::kInteger
                             : HighsVarType::kContinuous;
    }
  }
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("time_limit", time_limit);
  highs.setOptionValue("threads", threads);
  double start = now();
  if (highs.passModel(std::move(lp)) == HighsStatus::kError) {
    out.status = "passModel failed";
    return out;
  }
  highs.run();
  out.seconds           = now() - start;
  out.status            = highs.modelStatusToString(highs.getModelStatus());
  const HighsInfo& info = highs.getInfo();
  out.has_solution      = info.primal_solution_status == kSolutionStatusFeasible;
  out.objective         = info.objective_function_value;
  out.dual_bound        = relaxation ? info.objective_function_value : info.mip_dual_bound;
  if (out.has_solution) { out.x = highs.getSolution().col_value; }
  return out;
}

void put_solve(FILE* out, const char* name, const solve_t& s)
{
  std::fprintf(out,
               "  \"%s\": {\"status\": \"%s\", \"has_solution\": %s, \"objective\": %.17g, "
               "\"dual_bound\": %.17g, \"seconds\": %.3f},\n",
               name,
               s.status.c_str(),
               s.has_solution ? "true" : "false",
               std::isfinite(s.objective) ? s.objective : 0.,
               std::isfinite(s.dual_bound) ? s.dual_bound : 0.,
               s.seconds);
}

// "pass", "FAIL" or "skipped".
const char* verdict(bool applicable, bool ok)
{
  return !applicable ? "skipped" : ok ? "pass" : "FAIL";
}

}  // namespace

int main(int argc, char** argv)
{
  std::string path, json_path;
  double lp_limit = 120., mip_limit = 60., known = 0.;
  bool has_known = false;
  int threads    = 1;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--lp-time-limit" && i + 1 < argc) {
      lp_limit = std::atof(argv[++i]);
    } else if (arg == "--mip-time-limit" && i + 1 < argc) {
      mip_limit = std::atof(argv[++i]);
    } else if (arg == "--known-objective" && i + 1 < argc) {
      known     = std::atof(argv[++i]);
      has_known = true;
    } else if (arg == "--threads" && i + 1 < argc) {
      threads = std::atoi(argv[++i]);
    } else if (path.empty()) {
      path = arg;
    } else if (json_path.empty()) {
      json_path = arg;
    } else {
      std::fprintf(stderr, "usage: gprs_check <model.csr> <out.json> [options]\n");
      return 1;
    }
  }
  if (json_path.empty()) {
    std::fprintf(stderr, "usage: gprs_check <model.csr> <out.json> [options]\n");
    return 1;
  }

  gprs::model_t model;
  std::string error;
  if (!gprs::read_csr_dump(path, model, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  gprs::options_t options;
  gprs::result_t result = gprs::presolve(model, options);
  bool has_reduced      = result.status == gprs::status_t::reduced ||
                     result.status == gprs::status_t::unchanged ||
                     result.status == gprs::status_t::reduced_to_empty;

  solve_t lp_original = solve(model, true, lp_limit, threads);
  solve_t lp_reduced, mip_reduced;
  gprs::point_check_t point;
  bool has_point = false;
  if (has_reduced) {
    lp_reduced = solve(result.reduced, true, lp_limit, threads);
    if (mip_limit > 0) { mip_reduced = solve(result.reduced, false, mip_limit, threads); }
    if (mip_reduced.has_solution) {
      std::vector<double> x_reduced = mip_reduced.x;
      x_reduced.resize(result.reduced.ncols);
      std::vector<double> x_original;
      result.postsolve.postsolve(x_reduced, x_original);
      point     = gprs::check_point(model, x_original);
      has_point = true;
    }
  }

  // Tolerance of a comparison of two objective values: relative 1e-6, absolute 1e-6.
  auto close_or_above = [](double a, double b) {
    return a >= b - 1e-6 * std::fmax(1., std::fmax(std::fabs(a), std::fabs(b)));
  };
  bool lp_both = lp_original.status == "Optimal" &&
                 (lp_reduced.status == "Optimal" || lp_reduced.status == "empty");
  bool lp_red = lp_reduced.status == "Optimal" || lp_reduced.status == "empty";
  const char* relaxation_not_weaker =
    verdict(lp_both, close_or_above(lp_reduced.objective, lp_original.objective));
  const char* relaxation_below_known =
    verdict(has_known && lp_red, close_or_above(known, lp_reduced.objective));
  // An infeasible reduced relaxation of an instance with a known solution is a wrong reduction.
  const char* reduced_feasible =
    verdict(has_known && has_reduced && lp_reduced.ran && lp_reduced.status != "Time limit reached",
            lp_reduced.status != "Infeasible");
  const char* point_feasible_verdict =
    verdict(has_point,
            point.row_violation <= 1e-5 && point.bound_violation <= 1e-5 &&
              point.integrality_violation <= 1e-5);
  const char* point_objective = verdict(has_point,
                                        std::fabs(point.objective - mip_reduced.objective) <=
                                          1e-6 * std::fmax(1., std::fabs(mip_reduced.objective)));
  bool mip_optimal            = mip_reduced.status == "Optimal" || mip_reduced.status == "empty";
  const char* optimum_known =
    verdict(has_known && mip_optimal,
            std::fabs(mip_reduced.objective - known) <= 1e-4 * std::fmax(1., std::fabs(known)));
  // The MIP solution of the reduced model must not be better than the known optimum is not a
  // test (the known value may be a best known one); a dual bound above it is one.
  const char* bound_below_known =
    verdict(has_known && mip_reduced.ran && mip_reduced.status != "empty" &&
              std::isfinite(mip_reduced.dual_bound),
            close_or_above(known + 1e-4 * std::fmax(1., std::fabs(known)), mip_reduced.dual_bound));

  std::string tmp = json_path + ".tmp";
  FILE* out       = std::fopen(tmp.c_str(), "w");
  if (!out) {
    std::fprintf(stderr, "cannot open %s\n", tmp.c_str());
    return 1;
  }
  std::fprintf(out, "{\n  \"instance\": \"%s\",\n", path.c_str());
  std::fprintf(out, "  \"presolve_status\": \"%s\",\n", gprs::status_name(result.status));
  std::fprintf(out, "  \"presolve_seconds\": %.6f,\n", result.seconds);
  std::fprintf(out,
               "  \"original\": {\"rows\": %d, \"cols\": %d, \"nnz\": %lld},\n",
               model.nrows,
               model.ncols,
               (long long)model.nnz());
  std::fprintf(out,
               "  \"reduced\": {\"rows\": %d, \"cols\": %d, \"nnz\": %lld},\n",
               result.reduced.nrows,
               result.reduced.ncols,
               (long long)result.reduced.nnz());
  if (has_known) { std::fprintf(out, "  \"known_objective\": %.17g,\n", known); }
  put_solve(out, "lp_original", lp_original);
  put_solve(out, "lp_reduced", lp_reduced);
  put_solve(out, "mip_reduced", mip_reduced);
  if (has_point) {
    std::fprintf(out,
                 "  \"postsolved_point\": {\"row_violation\": %.3g, \"bound_violation\": %.3g, "
                 "\"integrality_violation\": %.3g, \"objective\": %.17g},\n",
                 point.row_violation,
                 point.bound_violation,
                 point.integrality_violation,
                 point.objective);
  }
  std::fprintf(out,
               "  \"checks\": {\"relaxation_not_weaker\": \"%s\", \"relaxation_below_known\": "
               "\"%s\", \"reduced_relaxation_feasible\": \"%s\", \"postsolved_point_feasible\": "
               "\"%s\", \"postsolved_point_objective\": \"%s\", \"optimum_equals_known\": \"%s\", "
               "\"dual_bound_below_known\": \"%s\"}\n}\n",
               relaxation_not_weaker,
               relaxation_below_known,
               reduced_feasible,
               point_feasible_verdict,
               point_objective,
               optimum_known,
               bound_below_known);
  if (std::fclose(out) != 0 || std::rename(tmp.c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot write %s\n", json_path.c_str());
    return 1;
  }
  return 0;
}
