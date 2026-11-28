/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// gprs_presolve <model.csr> [--json <file>] [--primal-only] [--time-limit <seconds>]
//               [--off <rule>]... [--exact-parallel-rows] [--pivot-growth-limit <x>]
//               [--saved-row-factor <x>] [--integral-drift-share <x>]
//               [--alias-scale-limit <x>] [--probing-balance <x>]
// Reads a GPRSCSR1 dump, runs the presolve, prints sizes before and after, seconds and the
// determinism hash of the reduced model, and writes the same as JSON.
// --off takes the name of a rule family (rule_name) or of a reduction of the column rules
// (sub_rule_name).
// --exact-parallel-rows: only rows whose free entries are proportional are merged
// (options_t::near_parallel_rows off).
// --pivot-growth-limit, --saved-row-factor: the two limits of the singleton-column rules
// (options_t). --integral-drift-share: tolerances_t::integral_drift_share.
// --alias-scale-limit, --probing-balance: the two settings of simple probing (options_t).
// Exit code 0 when the presolve ran, 1 when the dump cannot be read, 2 on a usage error.

#include <gprs/io.hpp>
#include <gprs/presolve.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void write_stats(FILE* out, const gprs::rule_stats_t& s)
{
  std::fprintf(out,
               "{\"calls\": %lld, \"work\": %lld, \"first_work\": %lld, \"findings\": %lld, "
               "\"records\": %lld, "
               "\"seconds\": %.6f, \"rows_removed\": %lld, \"cols_removed\": %lld, "
               "\"nnz_removed\": %lld}",
               (long long)s.calls,
               (long long)s.work,
               (long long)s.first_work,
               (long long)s.findings,
               (long long)s.records,
               s.seconds,
               (long long)s.rows_removed,
               (long long)s.cols_removed,
               (long long)s.nnz_removed);
}

void write_report(FILE* out,
                  const std::string& path,
                  const gprs::model_t& model,
                  const gprs::result_t& result)
{
  std::string escaped;
  for (char c : path) {
    if (c == '"' || c == '\\') { escaped += '\\'; }
    escaped += c;
  }
  std::fprintf(out, "{\n  \"instance\": \"%s\",\n", escaped.c_str());
  std::fprintf(out, "  \"status\": \"%s\",\n", gprs::status_name(result.status));
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
  std::fprintf(out, "  \"seconds\": %.6f,\n", result.seconds);
  std::fprintf(out,
               "  \"seconds_outside_rules\": {\"state\": %.6f, \"start\": %.6f, "
               "\"storage\": %.6f, \"finish\": %.6f, \"hash\": %.6f},\n",
               result.seconds_state,
               result.seconds_start,
               result.seconds_storage,
               result.seconds_finish,
               result.seconds_hash);
  std::fprintf(out, "  \"hash\": \"%016llx\",\n", (unsigned long long)result.hash);
  std::fprintf(out, "  \"rounds\": %d,\n  \"rebuilds\": %d,\n", result.rounds, result.rebuilds);
  std::fprintf(out,
               "  \"fixed_point\": %s,\n  \"explicit\": {\"rounds\": %d, \"seconds\": %.6f},\n",
               result.fixed_point ? "true" : "false",
               result.explicit_rounds,
               result.seconds_explicit);
  std::fprintf(out,
               "  \"parallel_rows\": {\"merges\": %zu, \"subtractions\": %zu},\n",
               result.postsolve.parallel_rows.size(),
               result.postsolve.row_additions.size());
  std::fprintf(out, "  \"dual_findings\": %lld,\n", (long long)result.dual_findings);
  std::fprintf(out, "  \"sub_rules\": {");
  for (int32_t s = 0; s < gprs::sub_rule::count; ++s) {
    std::fprintf(out,
                 "%s\"%s\": %lld",
                 s > 0 ? ", " : "",
                 gprs::sub_rule_name(s),
                 (long long)result.sub_findings[s]);
  }
  std::fprintf(out, "},\n");
  std::fprintf(out,
               "  \"columns\": {\"cost_fill\": %lld, \"saved_entries\": %lld, "
               "\"max_growth\": %.6g, \"growth_refused\": %lld, \"growth_refused_row\": %lld, "
               "\"saved_row_refused\": %lld, \"alias_depth\": %d, "
               "\"free_singleton_rejected_explicit\": %lld},\n",
               (long long)result.columns.cost_fill,
               (long long)result.columns.saved_entries,
               result.columns.max_growth,
               (long long)result.columns.growth_refused,
               (long long)result.columns.growth_refused_row,
               (long long)result.columns.saved_row_refused,
               result.columns.alias_depth,
               (long long)result.columns.free_singleton_rejected_explicit);
  int64_t probing_aliases = 0, probing_aliases_with_row = 0;
  for (const gprs::alias_record_t& r : result.postsolve.aliases) {
    if (r.origin != gprs::alias_origin_t::probing) { continue; }
    ++probing_aliases;
    probing_aliases_with_row += r.row >= 0;
  }
  std::fprintf(out,
               "  \"probing\": {\"rows\": %lld, \"rows_deleted\": %lld, \"rows_cut\": %lld, "
               "\"connected\": %lld, \"unit_refused\": %lld, \"scale_refused\": %lld, "
               "\"size_refused\": %lld, "
               "\"deep_batches\": %lld, \"entry_changes\": %lld, \"max_scale\": %.17g, "
               "\"aliases\": %lld, \"aliases_with_row\": %lld},\n",
               (long long)result.columns.probing_rows,
               (long long)result.columns.probing_rows_deleted,
               (long long)result.columns.probing_rows_cut,
               (long long)result.columns.probing_connected,
               (long long)result.columns.probing_unit_refused,
               (long long)result.columns.probing_scale_refused,
               (long long)result.columns.probing_size_refused,
               (long long)result.columns.probing_deep_batches,
               (long long)result.columns.probing_entry_changes,
               result.columns.probing_max_scale,
               (long long)probing_aliases,
               (long long)probing_aliases_with_row);
  int64_t kinds[4] = {}, interval_columns = 0, forcing_rows = 0;
  for (const gprs::col_from_rows_record_t& r : result.postsolve.col_from_rows) {
    int32_t kind = r.kind == gprs::col_from_rows_kind_t::defined    ? (r.dual ? 1 : 0)
                   : r.kind == gprs::col_from_rows_kind_t::interval ? 2
                                                                    : 3;
    ++kinds[kind];
    if (kind == 2) { interval_columns += r.cols_end - r.cols_begin; }
    if (kind == 3) { forcing_rows += r.rows_end - r.rows_begin; }
  }
  std::fprintf(
    out,
    "  \"col_from_rows\": {\"defined\": %lld, \"defined_dual\": %lld, \"interval\": %lld, "
    "\"interval_columns\": %lld, \"forcing\": %lld, \"forcing_rows\": %lld},\n",
    (long long)kinds[0],
    (long long)kinds[1],
    (long long)kinds[2],
    (long long)interval_columns,
    (long long)kinds[3],
    (long long)forcing_rows);
  int64_t cost_nnz = 0, reduced_cost_nnz = 0;
  for (double cost : model.cost) {
    cost_nnz += cost != 0.;
  }
  for (double cost : result.reduced.cost) {
    reduced_cost_nnz += cost != 0.;
  }
  std::fprintf(out,
               "  \"objective_nnz\": {\"original\": %lld, \"reduced\": %lld},\n",
               (long long)cost_nnz,
               (long long)reduced_cost_nnz);
  std::fprintf(out, "  \"start\": ");
  write_stats(out, result.start);
  std::fprintf(out, ",\n  \"rules\": {\n");
  for (int32_t r = 0; r < gprs::rule::count; ++r) {
    std::fprintf(out, "    \"%s\": ", gprs::rule_name(r));
    write_stats(out, result.stats[r]);
    std::fprintf(out, "%s\n", r + 1 < gprs::rule::count ? "," : "");
  }
  std::fprintf(out, "  }\n}\n");
}

int usage()
{
  std::fprintf(stderr,
               "usage: gprs_presolve <model.csr> [--json <file>] [--primal-only] "
               "[--time-limit <seconds>] [--off <rule>]... [--exact-parallel-rows] "
               "[--pivot-growth-limit <x>] [--saved-row-factor <x>] "
               "[--integral-drift-share <x>] [--alias-scale-limit <x>] "
               "[--probing-balance <x>]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string path, json_path;
  gprs::options_t options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--json" && i + 1 < argc) {
      json_path = argv[++i];
    } else if (arg == "--primal-only") {
      options.primal_only = true;
    } else if (arg == "--exact-parallel-rows") {
      options.near_parallel_rows = false;
    } else if (arg == "--time-limit" && i + 1 < argc) {
      options.time_limit = std::atof(argv[++i]);
    } else if (arg == "--pivot-growth-limit" && i + 1 < argc) {
      options.pivot_growth_limit = std::atof(argv[++i]);
    } else if (arg == "--saved-row-factor" && i + 1 < argc) {
      options.saved_row_factor = std::atof(argv[++i]);
    } else if (arg == "--integral-drift-share" && i + 1 < argc) {
      options.tol.integral_drift_share = std::atof(argv[++i]);
    } else if (arg == "--alias-scale-limit" && i + 1 < argc) {
      options.alias_scale_limit = std::atof(argv[++i]);
    } else if (arg == "--probing-balance" && i + 1 < argc) {
      options.probing_balance = std::atof(argv[++i]);
    } else if (arg == "--off" && i + 1 < argc) {
      int32_t rule = 0, sub = 0;
      ++i;
      while (rule < gprs::rule::count && std::strcmp(gprs::rule_name(rule), argv[i]) != 0) {
        ++rule;
      }
      while (sub < gprs::sub_rule::count && std::strcmp(gprs::sub_rule_name(sub), argv[i]) != 0) {
        ++sub;
      }
      if (rule < gprs::rule::count) {
        options.enabled[rule] = false;
      } else if (sub < gprs::sub_rule::count) {
        options.sub_enabled[sub] = false;
      } else {
        return usage();
      }
    } else if (path.empty() && arg[0] != '-') {
      path = arg;
    } else {
      return usage();
    }
  }
  if (path.empty()) { return usage(); }

  gprs::model_t model;
  std::string error;
  if (!gprs::read_csr_dump(path, model, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }

  gprs::result_t result = gprs::presolve(model, options);
  write_report(stdout, path, model, result);
  if (!json_path.empty()) {
    std::string tmp = json_path + ".tmp";
    FILE* out       = std::fopen(tmp.c_str(), "w");
    if (!out) {
      std::fprintf(stderr, "cannot open %s\n", tmp.c_str());
      return 1;
    }
    write_report(out, path, model, result);
    if (std::fclose(out) != 0 || std::rename(tmp.c_str(), json_path.c_str()) != 0) {
      std::fprintf(stderr, "cannot write %s\n", json_path.c_str());
      return 1;
    }
  }
  return 0;
}
