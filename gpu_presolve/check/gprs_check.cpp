/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Correctness check of the host engine on one instance, with HiGHS as the solver.
//
// gprs_check <model.csr> <out.json> [--solutions <file>]... [--lp-time-limit <s>]
//            [--mip-time-limit <s>] [--known-objective <value>] [--known-is-bound]
//            [--expect infeasible|unbounded] [--primal-only] [--threads <n>]
//            [--cost-variants <n>] [--push-points <n>]
//   Presolve status. A status without a reduced model fails when the instance has a known
//     objective value or a usable point of a solution file; limit and failed always fail.
//   Structure check (structure_check.hpp), in default and in primal-only mode: the postsolve
//     stack, the column types, the bounds, and every original row against the bounds,
//     against its row of the reduced model or, when an equation was subtracted from it,
//     against that row and the reduced rows of the equations. It needs neither a point nor a
//     solver.
//   Crush check (crush_check.hpp), in default and in primal-only mode, with every usable point
//     of the solution files (solution_file.hpp) and up to --push-points (at most 4) points
//     made from each by moving its columns to where bounds and rows stop them.
//   Cost variants: structure and crush check on --cost-variants copies of the model with other
//     costs. The points are feasible whatever the objective is, and the costs of a variant push
//     every column towards a finite bound, so a variant with a usable point is neither
//     infeasible nor unbounded. They reach the cost paths (dual fixing, the cost and offset of a
//     substituted column) on columns that the objective of the instance leaves at zero.
//   Solver checks, on the presolve in default mode or with --primal-only in that mode:
//     LP relaxation of the original and of the reduced model: the reduced value is not below
//       the original one (no reduction relaxes) and not above a known solution value; the
//       reduced relaxation ends optimal or at the time limit.
//     MIP solve of the reduced model with gap 0: not infeasible; a solution is postsolved and
//       tested in the original model, rows relative to their largest term; a proved optimum
//       equals the known optimum within 1e-6 relative; the dual bound is not above a known
//       solution value.
//   --known-objective is the value of the minimisation in the dump. With --known-is-bound it is
//   a best known value: an upper bound of the optimum, as is the best usable point. A time
//   limit of 0 leaves the solve out. --expect names what is known of an instance without a
//   solution: its reduced MIP must then not have one.
//
// gprs_check --solve <model.csr> <solutions file> [--mip-time-limit <s>] [--threads <n>]
//            [--max-points <n>]
//   Solves the original MIP with HiGHS and writes every improving solution that is a usable
//   point to the solution file as it is found: the first max-points - 1 and the last one.
//   A report goes to <solutions file>.json when the solve ends.
//
// A solve that HiGHS does not end long after its time limit is left behind with status
// "Time limit ignored"; everything else is still checked and reported.
// Exit code 0 when the tool ran (whatever the checks found), 1 on an input error.

#include "crush_check.hpp"
#include "solution_file.hpp"
#include "structure_check.hpp"

#include <gprs/io.hpp>
#include <gprs/presolve.hpp>

#include <Highs.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using gprs_check::crush_report_t;
using gprs_check::point_report_t;
using gprs_check::point_set_t;
using gprs_check::structure_report_t;

double now()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct solve_t {
  bool ran           = false;
  std::string status = "not run";
  bool has_solution  = false;
  // Not a number until a solve has ended.
  double objective  = std::nan("");
  double dual_bound = std::nan("");
  double seconds    = 0.;
  std::vector<double> x;
};

HighsLp to_highs(const gprs::model_t& model, bool relaxation)
{
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
  return lp;
}

// The tolerance HiGHS solves a MIP with whose solution becomes a point of this tool: tighter
// than the tolerance the point is tested with.
constexpr double reduced_mip_tolerance  = 1e-7;
constexpr double original_mip_tolerance = 1e-9;

// HiGHS has run far past its time limit on some instances (var-smallemery-m6j6: 35 minutes
// with a limit of 60 s, in a phase that polls neither the limit nor the interrupt callback). A
// solve therefore runs in a thread of its own. One that has not ended at patience(limit) is
// left behind with status "Time limit ignored": what does not need it is still checked and
// reported, and the process ends without waiting for it (leave).
bool solve_left_behind = false;

#ifndef GPRS_CHECK_PATIENCE_SECONDS
#define GPRS_CHECK_PATIENCE_SECONDS 60.
#endif

double patience(double time_limit) { return 1.5 * time_limit + GPRS_CHECK_PATIENCE_SECONDS; }

// Runs work() in a thread; false when it was left behind. work owns everything it uses.
template <typename work_t>
bool run_bounded(work_t work, double time_limit)
{
  auto done               = std::make_shared<std::promise<void>>();
  std::future<void> ended = done->get_future();
  std::thread([work, done]() mutable {
    work();
    done->set_value();
  }).detach();
  std::chrono::duration<double> wait(patience(time_limit));
  if (ended.wait_for(wait) == std::future_status::ready) { return true; }
  solve_left_behind = true;
  return false;
}

// The exit of main: a thread that was left behind is not waited for.
int leave(int code)
{
  if (solve_left_behind) {
    std::fflush(nullptr);
    std::_Exit(code);
  }
  return code;
}

struct solve_job_t {
  HighsLp lp;
  bool relaxation   = false;
  double time_limit = 0.;
  int threads       = 1;
  solve_t out;
};

void run_solve(solve_job_t& job)
{
  solve_t& out = job.out;
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("time_limit", job.time_limit);
  highs.setOptionValue("threads", job.threads);
  if (!job.relaxation) {
    highs.setOptionValue("mip_rel_gap", 0.);
    highs.setOptionValue("mip_abs_gap", 0.);
    highs.setOptionValue("mip_feasibility_tolerance", reduced_mip_tolerance);
    highs.setOptionValue("primal_feasibility_tolerance", reduced_mip_tolerance);
  }
  double start = now();
  if (highs.passModel(std::move(job.lp)) == HighsStatus::kError) {
    out.status = "passModel failed";
    return;
  }
  highs.run();
  out.seconds           = now() - start;
  out.status            = highs.modelStatusToString(highs.getModelStatus());
  const HighsInfo& info = highs.getInfo();
  out.has_solution      = info.primal_solution_status == kSolutionStatusFeasible;
  out.objective         = info.objective_function_value;
  out.dual_bound        = job.relaxation ? info.objective_function_value : info.mip_dual_bound;
  if (out.has_solution) { out.x = highs.getSolution().col_value; }
}

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
  auto job        = std::make_shared<solve_job_t>();
  job->lp         = to_highs(model, relaxation);
  job->relaxation = relaxation;
  job->time_limit = time_limit;
  job->threads    = threads;
  job->out.ran    = true;
  double start    = now();
  if (run_bounded([job] { run_solve(*job); }, time_limit)) { return job->out; }
  out.status  = "Time limit ignored";
  out.seconds = now() - start;
  return out;
}

// A JSON number, or null for a value that is not finite.
std::string number(double v, const char* format = "%.17g")
{
  if (!std::isfinite(v)) { return "null"; }
  char buffer[64];
  std::snprintf(buffer, sizeof buffer, format, v);
  return buffer;
}

std::string quoted(const std::string& text)
{
  std::string out = "\"";
  for (char c : text) {
    if (c == '"' || c == '\\') { out += '\\'; }
    out += c;
  }
  return out + "\"";
}

void put_solve(FILE* out, const char* name, const solve_t& s)
{
  std::fprintf(out,
               "  \"%s\": {\"status\": \"%s\", \"has_solution\": %s, \"objective\": %s, "
               "\"dual_bound\": %s, \"seconds\": %.3f},\n",
               name,
               s.status.c_str(),
               s.has_solution ? "true" : "false",
               number(s.objective).c_str(),
               number(s.dual_bound).c_str(),
               s.seconds);
}

void put_highs(FILE* out)
{
  std::fprintf(out,
               "  \"highs\": {\"version\": \"%s\", \"githash\": \"%s\", \"int_bytes\": %d},\n",
               highsVersion(),
               highsGithash(),
               (int)sizeof(HighsInt));
}

void put_crush(FILE* out, const char* name, const crush_report_t& c, bool last)
{
  std::fprintf(out,
               "    \"%s\": {\"status\": \"%s\", \"hash\": \"%016llx\", \"points\": %d, "
               "\"first_failed_point\": %d,\n",
               name,
               c.status.c_str(),
               (unsigned long long)c.hash,
               c.points,
               c.first_failed_point);
  std::fprintf(out,
               "      \"reduced_point\": {\"row\": %s, \"bound\": %s, \"integrality\": %s, "
               "\"failed\": %d},\n",
               number(c.reduced_row, "%.3g").c_str(),
               number(c.reduced_bound, "%.3g").c_str(),
               number(c.reduced_integrality, "%.3g").c_str(),
               c.failed_reduced);
  std::fprintf(out,
               "      \"postsolved_point\": {\"row\": %s, \"bound\": %s, \"integrality\": %s, "
               "\"failed\": %d},\n",
               number(c.postsolved_row, "%.3g").c_str(),
               number(c.postsolved_bound, "%.3g").c_str(),
               number(c.postsolved_integrality, "%.3g").c_str(),
               c.failed_postsolved);
  std::fprintf(out,
               "      \"objective\": {\"identity\": %s, \"rise\": %s, \"fall\": %s, "
               "\"failed\": %d},\n",
               number(c.objective_identity, "%.3g").c_str(),
               number(c.objective_rise, "%.3g").c_str(),
               number(c.objective_fall, "%.3g").c_str(),
               c.failed_objective);
  std::fprintf(out,
               "      \"round_trip\": {\"integer\": %s, \"continuous\": %s, \"columns\": %lld, "
               "\"failed\": %d}}%s\n",
               number(c.moved_integer, "%.3g").c_str(),
               number(c.moved_continuous, "%.3g").c_str(),
               (long long)c.moved_columns,
               c.failed_round_trip,
               last ? "" : ",");
}

void put_structure(FILE* out, const char* name, const structure_report_t& s, bool last)
{
  std::fprintf(out,
               "    \"%s\": {\"ran\": %s, \"stack_error\": %s, \"lost_integers\": %lld, "
               "\"first_lost_integer\": %d,\n",
               name,
               s.ran ? "true" : "false",
               quoted(s.stack_error).c_str(),
               (long long)s.lost_integers,
               s.first_lost_integer);
  std::fprintf(out,
               "      \"bounds\": {\"worst\": %s, \"column\": %d, \"failed\": %lld},\n",
               number(s.bound_worst, "%.3g").c_str(),
               s.bound_worst_col,
               (long long)s.bound_failures);
  std::fprintf(
    out,
    "      \"rows\": {\"tested\": %lld, \"worst\": %s, \"row\": %d, \"failed\": %lld},\n",
    (long long)s.rows_tested,
    number(s.row_worst, "%.3g").c_str(),
    s.row_worst_row,
    (long long)s.row_failures);
  std::fprintf(out,
               "      \"kept_rows\": {\"tested\": %lld, \"parallel\": %lld, \"not_tested\": %lld, "
               "\"worst\": %s, \"row\": %d, \"failed\": %lld},\n",
               (long long)s.kept_tested,
               (long long)s.kept_parallel,
               (long long)s.kept_untested,
               number(s.kept_worst, "%.3g").c_str(),
               s.kept_worst_row,
               (long long)s.kept_failures);
  std::fprintf(out,
               "      \"added_rows\": {\"rows\": %lld, \"tested\": %lld, \"not_tested\": %lld, "
               "\"multiplier\": %s, \"worst\": %s, \"row\": %d, \"failed\": %lld},\n",
               (long long)s.added_rows,
               (long long)s.added_tested,
               (long long)s.added_untested,
               number(s.added_multiplier, "%.3g").c_str(),
               number(s.added_worst, "%.3g").c_str(),
               s.added_worst_row,
               (long long)s.added_failures);
  std::fprintf(out,
               "      \"postsolve\": {\"worst\": %s, \"column\": %d, \"failed\": %lld}}%s\n",
               number(s.postsolve_worst, "%.3g").c_str(),
               s.postsolve_worst_col,
               (long long)s.postsolve_failures,
               last ? "" : ",");
}

// "pass", "FAIL" or "skipped".
const char* verdict(bool applicable, bool ok)
{
  return !applicable ? "skipped" : ok ? "pass" : "FAIL";
}

// What needs no solver, for one presolve result.
struct mode_report_t {
  std::string status;
  bool has_reduced = false;
  uint64_t hash    = 0;
  structure_report_t structure;
  crush_report_t crush;
};

mode_report_t solver_free_checks(const gprs::model_t& model,
                                 const gprs::result_t& result,
                                 bool primal_only,
                                 const point_set_t& points)
{
  mode_report_t r;
  r.status      = gprs::status_name(result.status);
  r.has_reduced = gprs_check::carries_model(result.status);
  r.hash        = result.hash;
  if (r.has_reduced) { r.structure = gprs_check::structure_check(model, result); }
  if (!points.empty()) { r.crush = gprs_check::crush_check(model, result, primal_only, points); }
  return r;
}

struct named_verdict_t {
  std::string name;
  const char* value;
};

// The verdicts of one mode, named <check>_<mode>_<part>.
void mode_verdicts(const mode_report_t& r, bool primal_only, std::vector<named_verdict_t>& out)
{
  std::string mode            = primal_only ? "primal_" : "default_";
  const structure_report_t& s = r.structure;
  bool composed               = s.ran && s.composed;
  out.push_back({"structure_" + mode + "stack", verdict(s.ran, s.composed)});
  out.push_back({"structure_" + mode + "types", verdict(composed, s.lost_integers == 0)});
  out.push_back({"structure_" + mode + "bounds", verdict(composed, s.bound_failures == 0)});
  out.push_back({"structure_" + mode + "rows", verdict(composed, s.row_failures == 0)});
  out.push_back({"structure_" + mode + "kept_rows", verdict(composed, s.kept_failures == 0)});
  out.push_back({"structure_" + mode + "added_rows", verdict(composed, s.added_failures == 0)});
  out.push_back({"structure_" + mode + "postsolve", verdict(composed, s.postsolve_failures == 0)});
  const crush_report_t& c = r.crush;
  bool crushed            = c.ran && c.has_reduced;
  out.push_back({"crush_" + mode + "status", verdict(c.ran, c.has_reduced)});
  out.push_back({"crush_" + mode + "reduced_point", verdict(crushed, c.failed_reduced == 0)});
  out.push_back({"crush_" + mode + "postsolved_point", verdict(crushed, c.failed_postsolved == 0)});
  out.push_back({"crush_" + mode + "objective", verdict(crushed, c.failed_objective == 0)});
  if (primal_only) {
    out.push_back({"crush_" + mode + "round_trip", verdict(crushed, c.failed_round_trip == 0)});
  }
}

// The model with the costs of variant `number`: an integer of size 0 to 3 per column, with the
// sign that pushes the column towards a finite bound (either sign with two finite bounds, zero
// with none), so the objective is bounded on the feasible set.
gprs::model_t cost_variant(const gprs::model_t& model, int number)
{
  gprs::model_t variant    = model;
  variant.objective_offset = 0.;
  uint64_t state           = 0x9e3779b97f4a7c15ull * (uint64_t)number;
  for (int32_t j = 0; j < model.ncols; ++j) {
    state           = state * 6364136223846793005ull + 1442695040888963407ull;
    uint32_t draw   = (uint32_t)(state >> 33);
    bool lower      = model.col_lower[j] > -gprs::inf;
    bool upper      = model.col_upper[j] < gprs::inf;
    double sign     = lower && upper ? ((draw >> 8) & 1 ? 1. : -1.) : lower ? 1. : upper ? -1. : 0.;
    variant.cost[j] = sign * (draw % 4);
  }
  return variant;
}

struct variant_report_t {
  int number = 0;
  mode_report_t in_mode[2];  // default, primal-only
  std::vector<std::string> failed[2];
};

// Structure and crush check of both modes on the cost variants.
std::vector<variant_report_t> check_cost_variants(const gprs::model_t& model,
                                                  const point_set_t& points,
                                                  int count)
{
  std::vector<variant_report_t> reports(count);
  for (int v = 0; v < count; ++v) {
    variant_report_t& report = reports[v];
    report.number            = v + 1;
    gprs::model_t variant    = cost_variant(model, report.number);
    for (int mode = 0; mode < 2; ++mode) {
      gprs::options_t options;
      options.primal_only = mode == 1;
      report.in_mode[mode] =
        solver_free_checks(variant, gprs::presolve(variant, options), mode == 1, points);
      std::vector<named_verdict_t> verdicts;
      mode_verdicts(report.in_mode[mode], mode == 1, verdicts);
      for (const named_verdict_t& check : verdicts) {
        if (std::string(check.value) == "FAIL") { report.failed[mode].push_back(check.name); }
      }
    }
  }
  return reports;
}

bool write_file(const std::string& path, FILE* out)
{
  std::string tmp = path + ".tmp";
  return std::fclose(out) == 0 && std::rename(tmp.c_str(), path.c_str()) == 0;
}

struct arguments_t {
  bool solve_mode = false;
  std::string path;
  std::string out_path;
  std::vector<std::string> solutions;
  double lp_limit  = 120.;
  double mip_limit = 60.;
  bool has_known   = false;
  double known     = 0.;
  bool known_bound = false;
  std::string expect;
  bool primal_only  = false;
  int threads       = 1;
  int max_points    = 20;
  int cost_variants = 0;
  int push_points   = 0;
};

int usage()
{
  std::fprintf(stderr,
               "usage: gprs_check <model.csr> <out.json> [--solutions <file>]... "
               "[--lp-time-limit <s>] [--mip-time-limit <s>] [--known-objective <value>] "
               "[--known-is-bound] [--expect infeasible|unbounded] [--primal-only] "
               "[--threads <n>] [--cost-variants <n>] [--push-points <n>]\n"
               "       gprs_check --solve <model.csr> <solutions file> [--mip-time-limit <s>] "
               "[--threads <n>] [--max-points <n>]\n");
  return 1;
}

bool parse(int argc, char** argv, arguments_t& a)
{
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    bool value      = i + 1 < argc;
    if (arg == "--solve") {
      a.solve_mode = true;
    } else if (arg == "--solutions" && value) {
      a.solutions.push_back(argv[++i]);
    } else if (arg == "--lp-time-limit" && value) {
      a.lp_limit = std::atof(argv[++i]);
    } else if (arg == "--mip-time-limit" && value) {
      a.mip_limit = std::atof(argv[++i]);
    } else if (arg == "--known-objective" && value) {
      a.known     = std::atof(argv[++i]);
      a.has_known = true;
    } else if (arg == "--known-is-bound") {
      a.known_bound = true;
    } else if (arg == "--expect" && value) {
      a.expect = argv[++i];
    } else if (arg == "--primal-only") {
      a.primal_only = true;
    } else if (arg == "--threads" && value) {
      a.threads = std::atoi(argv[++i]);
    } else if (arg == "--max-points" && value) {
      a.max_points = std::atoi(argv[++i]);
    } else if (arg == "--cost-variants" && value) {
      a.cost_variants = std::atoi(argv[++i]);
    } else if (arg == "--push-points" && value) {
      a.push_points = std::atoi(argv[++i]);
    } else if (arg[0] == '-') {
      return false;
    } else if (a.path.empty()) {
      a.path = arg;
    } else if (a.out_path.empty()) {
      a.out_path = arg;
    } else {
      return false;
    }
  }
  return !a.out_path.empty() &&
         (a.expect.empty() || a.expect == "infeasible" || a.expect == "unbounded") &&
         a.max_points >= 1 && a.cost_variants >= 0 && a.push_points >= 0;
}

// The solution mode: what the solver thread and the main thread share.
struct collector_t {
  gprs::model_t model;
  gprs_check::solution_writer_t writer;
  // Held while a record is written, and for good by a main thread that leaves the solve behind.
  std::mutex mutex;
  int max_points    = 20;
  int found         = 0;
  int rejected      = 0;
  double last       = gprs::inf;  // objective of the last record
  bool write_fail   = false;
  double start      = 0.;
  double time_limit = 0.;
  int threads       = 1;
  solve_t out;

  void take(const std::vector<double>& x)
  {
    std::vector<double> rounded = x;
    point_report_t report;
    double integrality;
    bool usable = gprs_check::usable_point(model, rounded, report, integrality);
    std::lock_guard<std::mutex> lock(mutex);
    ++found;
    if (!usable) {
      ++rejected;
      return;
    }
    bool ok = writer.points < max_points ? writer.append(report.objective, x)
                                         : writer.replace_last(report.objective, x);
    last    = report.objective;
    if (!ok) { write_fail = true; }
  }
};

void run_original(collector_t& collector)
{
  solve_t& out = collector.out;
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("time_limit", collector.time_limit);
  highs.setOptionValue("threads", collector.threads);
  highs.setOptionValue("mip_feasibility_tolerance", original_mip_tolerance);
  highs.setOptionValue("primal_feasibility_tolerance", original_mip_tolerance);
  highs.setCallback(
    [&collector](int type,
                 const std::string&,
                 const HighsCallbackOutput* data,
                 HighsCallbackInput* input,
                 void*) {
      if (type == kCallbackMipImprovingSolution) {
        collector.take(data->mip_solution);
      } else if (type == kCallbackMipInterrupt &&
                 now() - collector.start > 1.1 * collector.time_limit + 10.) {
        input->user_interrupt = true;
      }
    },
    nullptr);
  highs.startCallback(kCallbackMipImprovingSolution);
  highs.startCallback(kCallbackMipInterrupt);
  if (highs.passModel(to_highs(collector.model, false)) == HighsStatus::kError) {
    out.status = "passModel failed";
    return;
  }
  highs.run();
  out.status            = highs.modelStatusToString(highs.getModelStatus());
  const HighsInfo& info = highs.getInfo();
  out.has_solution      = info.primal_solution_status == kSolutionStatusFeasible;
  out.objective         = info.objective_function_value;
  out.dual_bound        = info.mip_dual_bound;
  // The final solution, when no callback delivered it.
  if (out.has_solution && !(collector.last <= out.objective + 1e-9 * std::fabs(out.objective))) {
    collector.take(highs.getSolution().col_value);
  }
}

int solve_original(const arguments_t& a)
{
  auto collector = std::make_shared<collector_t>();
  std::string error;
  if (!gprs::read_csr_dump(a.path, collector->model, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  collector->max_points = a.max_points;
  collector->time_limit = a.mip_limit;
  collector->threads    = a.threads;
  collector->start      = now();
  if (!collector->writer.open(a.out_path, collector->model.ncols)) {
    std::fprintf(stderr, "cannot write %s\n", a.out_path.c_str());
    return 1;
  }
  solve_t out;
  std::unique_lock<std::mutex> lock(collector->mutex, std::defer_lock);
  if (run_bounded([collector] { run_original(*collector); }, a.mip_limit)) {
    out = collector->out;
  } else {
    // The lock is kept: the file holds whole records and the counts below are final.
    lock.lock();
    out.status = "Time limit ignored";
  }
  out.ran     = true;
  out.seconds = now() - collector->start;
  if (collector->write_fail || !collector->writer.close()) {
    std::fprintf(stderr, "cannot write %s\n", a.out_path.c_str());
    return leave(1);
  }

  std::string report = a.out_path + ".json";
  FILE* file         = std::fopen((report + ".tmp").c_str(), "w");
  if (!file) {
    std::fprintf(stderr, "cannot open %s.tmp\n", report.c_str());
    return leave(1);
  }
  std::fprintf(file, "{\n  \"instance\": %s,\n", quoted(a.path).c_str());
  put_highs(file);
  put_solve(file, "mip_original", out);
  std::fprintf(file,
               "  \"time_limit\": %s,\n  \"tolerance\": %g,\n  \"points_found\": %d,\n"
               "  \"points_rejected\": %d,\n  \"points_written\": %lld,\n"
               "  \"best_point_objective\": %s\n}\n",
               number(a.mip_limit, "%g").c_str(),
               original_mip_tolerance,
               collector->found,
               collector->rejected,
               (long long)collector->writer.points,
               number(collector->last).c_str());
  if (!write_file(report, file)) {
    std::fprintf(stderr, "cannot write %s\n", report.c_str());
    return leave(1);
  }
  std::printf("%s: %s, %lld points written, %d rejected, best %s\n",
              a.path.c_str(),
              out.status.c_str(),
              (long long)collector->writer.points,
              collector->rejected,
              number(collector->last, "%.10g").c_str());
  return leave(0);
}

struct solutions_t {
  point_set_t set;
  std::vector<std::string> files;
  std::vector<int64_t> records;
  std::vector<int64_t> tail_bytes;
};

bool read_points(const arguments_t& a, const gprs::model_t& model, solutions_t& s)
{
  for (const std::string& path : a.solutions) {
    gprs_check::solution_set_t file;
    std::string error;
    if (!gprs_check::read_solutions(path, file, error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return false;
    }
    if (file.ncols != model.ncols) {
      std::fprintf(stderr,
                   "%s has %lld columns, the model %d\n",
                   path.c_str(),
                   (long long)file.ncols,
                   model.ncols);
      return false;
    }
    s.files.push_back(path);
    s.records.push_back(file.points.size());
    s.tail_bytes.push_back(file.tail_bytes);
    for (std::vector<double>& x : file.points) {
      s.set.add(model, std::move(x));
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv)
{
  arguments_t a;
  if (!parse(argc, argv, a)) { return usage(); }
  if (a.solve_mode) { return solve_original(a); }

  gprs::model_t model;
  std::string error;
  if (!gprs::read_csr_dump(a.path, model, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  solutions_t solutions;
  if (!read_points(a, model, solutions)) { return 1; }
  if (a.push_points > 0 && !solutions.set.empty()) {
    gprs_check::add_pushed_points(model, solutions.set, a.push_points);
  }
  const point_set_t& points = solutions.set;
  bool has_points           = !points.empty();

  // What needs no solver, in both modes. The presolve of the mode that is not solved, and those
  // of the cost variants, are dropped before a solve.
  mode_report_t in_mode[2];  // default, primal-only
  gprs::options_t options;
  options.primal_only = !a.primal_only;
  in_mode[!a.primal_only] =
    solver_free_checks(model, gprs::presolve(model, options), !a.primal_only, points);
  std::vector<variant_report_t> variants = check_cost_variants(model, points, a.cost_variants);
  options.primal_only                    = a.primal_only;
  gprs::result_t result                  = gprs::presolve(model, options);
  bool has_reduced                       = gprs_check::carries_model(result.status);
  in_mode[a.primal_only]                 = solver_free_checks(model, result, a.primal_only, points);
  const crush_report_t& crush_default    = in_mode[0].crush;
  const crush_report_t& crush_primal     = in_mode[1].crush;

  solve_t lp_original, lp_reduced, mip_reduced;
  if (a.lp_limit > 0) { lp_original = solve(model, true, a.lp_limit, a.threads); }
  point_report_t point;
  bool has_point = false;
  if (has_reduced) {
    if (a.lp_limit > 0) { lp_reduced = solve(result.reduced, true, a.lp_limit, a.threads); }
    if (a.mip_limit > 0) { mip_reduced = solve(result.reduced, false, a.mip_limit, a.threads); }
    if (mip_reduced.has_solution) {
      std::vector<double> x_reduced = mip_reduced.x;
      x_reduced.resize(result.reduced.ncols);
      std::vector<double> x_original;
      result.postsolve.postsolve(x_reduced, x_original);
      point     = gprs_check::check_point_scaled(model, x_original);
      has_point = true;
    }
  }

  // What is known of the instance. A usable point makes it feasible, and its value is an upper
  // bound of the optimum like a best known value.
  bool feasible_known   = a.has_known || has_points;
  bool expect_none      = a.expect == "infeasible" && !feasible_known;
  bool expect_unbounded = a.expect == "unbounded";
  double upper          = points.best_objective();
  if (a.has_known && a.known < upper) { upper = a.known; }
  auto tolerance = [](double v) {
    return gprs_check::objective_tolerance * std::fmax(1., std::fabs(v));
  };

  bool no_model_status = result.status == gprs::status_t::infeasible ||
                         result.status == gprs::status_t::unbounded_or_infeasible;
  bool status_expected =
    no_model_status &&
    (expect_unbounded ? result.status == gprs::status_t::unbounded_or_infeasible : expect_none);
  const char* presolve_status = verdict(!no_model_status || feasible_known || !a.expect.empty(),
                                        has_reduced || status_expected);

  bool lp_red  = lp_reduced.status == "Optimal" || lp_reduced.status == "empty";
  bool lp_both = lp_original.status == "Optimal" && lp_red;
  const char* relaxation_not_weaker = verdict(
    lp_both, lp_reduced.objective >= lp_original.objective - tolerance(lp_original.objective));
  const char* relaxation_below_known =
    verdict(feasible_known && lp_red, lp_reduced.objective <= upper + tolerance(upper));
  // Without a known solution the reduced relaxation of an infeasible instance may be infeasible.
  auto at_limit = [](const solve_t& s) {
    return s.status == "Time limit reached" || s.status == "Time limit ignored";
  };
  bool lp_status_ok =
    lp_red || at_limit(lp_reduced) ||
    (!feasible_known && (lp_reduced.status == "Infeasible" ||
                         lp_reduced.status == "Primal infeasible or unbounded")) ||
    (expect_unbounded && lp_reduced.status == "Unbounded");
  const char* reduced_relaxation_status = verdict(lp_reduced.ran, lp_status_ok);

  bool mip_optimal = mip_reduced.status == "Optimal" || mip_reduced.status == "empty";
  bool mip_ended   = mip_optimal || at_limit(mip_reduced);
  const char* reduced_mip_status =
    feasible_known ? verdict(mip_reduced.ran, mip_ended)
                   : verdict(mip_reduced.ran && expect_none &&
                               (mip_reduced.status == "Infeasible" || mip_reduced.has_solution),
                             !mip_reduced.has_solution);
  const char* point_feasible = verdict(has_point, point.within(gprs_check::check_tolerance));
  const char* point_objective =
    verdict(has_point,
            std::fabs(point.objective - mip_reduced.objective) <= tolerance(mip_reduced.objective));
  // A proved optimum of the reduced model is not above any known solution value, and not below
  // a known optimum.
  const char* optimum_known = verdict(
    feasible_known && mip_optimal,
    mip_reduced.objective <= upper + tolerance(upper) &&
      (!a.has_known || a.known_bound || mip_reduced.objective >= a.known - tolerance(a.known)));
  const char* bound_below_known =
    verdict(feasible_known && mip_reduced.ran && mip_reduced.status != "empty" &&
              std::isfinite(mip_reduced.dual_bound),
            mip_reduced.dual_bound <= upper + tolerance(upper));

  // A file with points of which none is usable would make the crush check silent.
  const char* solution_points = verdict(points.read > 0, has_points);

  std::vector<named_verdict_t> checks = {{"presolve_status", presolve_status},
                                         {"relaxation_not_weaker", relaxation_not_weaker},
                                         {"relaxation_below_known", relaxation_below_known},
                                         {"reduced_relaxation_status", reduced_relaxation_status},
                                         {"reduced_mip_status", reduced_mip_status},
                                         {"postsolved_point_feasible", point_feasible},
                                         {"postsolved_point_objective", point_objective},
                                         {"optimum_equals_known", optimum_known},
                                         {"dual_bound_below_known", bound_below_known},
                                         {"solution_points", solution_points}};
  mode_verdicts(in_mode[0], false, checks);
  mode_verdicts(in_mode[1], true, checks);
  // A variant with a usable point has a reduced model and passes structure and crush check.
  for (int mode = 0; mode < 2; ++mode) {
    bool ok = true;
    for (const variant_report_t& v : variants) {
      ok = ok && v.failed[mode].empty() && (v.in_mode[mode].has_reduced || !has_points);
    }
    checks.push_back(
      {mode ? "cost_variants_primal" : "cost_variants_default", verdict(!variants.empty(), ok)});
  }

  FILE* out = std::fopen((a.out_path + ".tmp").c_str(), "w");
  if (!out) {
    std::fprintf(stderr, "cannot open %s.tmp\n", a.out_path.c_str());
    return leave(1);
  }
  std::fprintf(out, "{\n  \"instance\": %s,\n", quoted(a.path).c_str());
  put_highs(out);
  std::fprintf(out, "  \"presolve_mode\": \"%s\",\n", a.primal_only ? "primal_only" : "default");
  std::fprintf(out, "  \"presolve_status\": \"%s\",\n", gprs::status_name(result.status));
  std::fprintf(out, "  \"presolve_seconds\": %.6f,\n", result.seconds);
  std::fprintf(out, "  \"hash\": \"%016llx\",\n", (unsigned long long)result.hash);
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
  if (a.has_known) {
    std::fprintf(out,
                 "  \"known_objective\": %.17g,\n  \"known_is_bound\": %s,\n",
                 a.known,
                 a.known_bound ? "true" : "false");
  }
  if (!a.expect.empty()) { std::fprintf(out, "  \"expect\": \"%s\",\n", a.expect.c_str()); }
  std::fprintf(out,
               "  \"tolerances\": {\"point\": %g, \"check\": %g, \"objective\": %g, "
               "\"objective_identity\": %g, \"reduced_mip\": %g},\n",
               gprs_check::point_tolerance,
               gprs_check::check_tolerance,
               gprs_check::objective_tolerance,
               gprs_check::identity_tolerance,
               reduced_mip_tolerance);
  put_solve(out, "lp_original", lp_original);
  put_solve(out, "lp_reduced", lp_reduced);
  put_solve(out, "mip_reduced", mip_reduced);
  if (has_point) {
    std::fprintf(out,
                 "  \"postsolved_point\": {\"row_violation\": %s, \"row_violation_scaled\": %s, "
                 "\"bound_violation\": %s, \"integrality_violation\": %s, \"objective\": %s},\n",
                 number(point.row, "%.3g").c_str(),
                 number(point.row_scaled, "%.3g").c_str(),
                 number(point.bound, "%.3g").c_str(),
                 number(point.integrality, "%.3g").c_str(),
                 number(point.objective).c_str());
  }
  std::fprintf(out, "  \"crush\": {\n    \"files\": [");
  for (size_t k = 0; k < solutions.files.size(); ++k) {
    std::fprintf(out,
                 "%s{\"file\": %s, \"points\": %lld, \"incomplete_tail_bytes\": %lld}",
                 k ? ", " : "",
                 quoted(solutions.files[k]).c_str(),
                 (long long)solutions.records[k],
                 (long long)solutions.tail_bytes[k]);
  }
  std::fprintf(out,
               "],\n    \"points\": %d, \"points_used\": %d, \"points_rejected\": %d, "
               "\"points_pushed\": %d,\n"
               "    \"rejected\": {\"row\": %s, \"bound\": %s, \"integrality\": %s},\n"
               "    \"best_objective\": %s%s\n",
               points.read,
               (int)points.points.size(),
               points.rejected,
               points.pushed,
               number(points.rejected_row, "%.3g").c_str(),
               number(points.rejected_bound, "%.3g").c_str(),
               number(points.rejected_integrality, "%.3g").c_str(),
               number(points.best_objective()).c_str(),
               has_points ? "," : "");
  if (has_points) {
    put_crush(out, "default", crush_default, false);
    put_crush(out, "primal_only", crush_primal, true);
  }
  std::fprintf(out, "  },\n");
  std::fprintf(out,
               "  \"structure\": {\n    \"tolerance\": %g, \"other_mode_status\": \"%s\",\n",
               gprs_check::structure_tolerance,
               in_mode[!a.primal_only].status.c_str());
  put_structure(out, "default", in_mode[0].structure, false);
  put_structure(out, "primal_only", in_mode[1].structure, true);
  std::fprintf(out, "  },\n  \"cost_variants\": [");
  for (size_t k = 0; k < variants.size(); ++k) {
    for (int mode = 0; mode < 2; ++mode) {
      const mode_report_t& r = variants[k].in_mode[mode];
      std::fprintf(out,
                   "%s\n    {\"variant\": %d, \"mode\": \"%s\", \"status\": \"%s\", "
                   "\"hash\": \"%016llx\", \"moved_columns\": %lld, \"objective_fall\": %s, "
                   "\"failed\": [",
                   k + mode ? "," : "",
                   variants[k].number,
                   mode ? "primal_only" : "default",
                   r.status.c_str(),
                   (unsigned long long)r.hash,
                   (long long)r.crush.moved_columns,
                   number(r.crush.objective_fall, "%.3g").c_str());
      for (size_t f = 0; f < variants[k].failed[mode].size(); ++f) {
        std::fprintf(out, "%s%s", f ? ", " : "", quoted(variants[k].failed[mode][f]).c_str());
      }
      std::fprintf(out, "]}");
    }
  }
  std::fprintf(out, "],\n  \"checks\": {");
  for (size_t k = 0; k < checks.size(); ++k) {
    std::fprintf(out,
                 "%s\"%s\": \"%s\"",
                 k == 0  ? ""
                 : k % 2 ? ", "
                         : ",\n    ",
                 checks[k].name.c_str(),
                 checks[k].value);
  }
  std::fprintf(out, "}\n}\n");
  if (!write_file(a.out_path, out)) {
    std::fprintf(stderr, "cannot write %s\n", a.out_path.c_str());
    return leave(1);
  }
  return leave(0);
}
