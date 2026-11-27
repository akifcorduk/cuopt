/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// gprs_presolve <model.csr> [--json <file>] [--primal-only] [--time-limit <seconds>]
//               [--off <rule>]...
// Reads a GPRSCSR1 dump, runs the presolve, prints sizes before and after, seconds and the
// determinism hash of the reduced model, and writes the same as JSON.
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
               "{\"calls\": %lld, \"work\": %lld, \"findings\": %lld, \"records\": %lld, "
               "\"seconds\": %.6f, \"rows_removed\": %lld, \"cols_removed\": %lld, "
               "\"nnz_removed\": %lld}",
               (long long)s.calls,
               (long long)s.work,
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
               "[--time-limit <seconds>] [--off <rule>]...\n");
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
    } else if (arg == "--time-limit" && i + 1 < argc) {
      options.time_limit = std::atof(argv[++i]);
    } else if (arg == "--off" && i + 1 < argc) {
      int32_t rule = 0;
      ++i;
      while (rule < gprs::rule::count && std::strcmp(gprs::rule_name(rule), argv[i]) != 0) {
        ++rule;
      }
      if (rule == gprs::rule::count) { return usage(); }
      options.enabled[rule] = false;
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
