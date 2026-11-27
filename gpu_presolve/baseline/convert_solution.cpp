/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Converts solutions that name their variables (the .sol files of MIPLIB: one "name value" line
// per nonzero variable, and a line "=obj= value") into a solution file of the correctness check
// (check/solution_file.hpp): one value per column, in the column order of the CSR dump.
//
// The dump has no names. It is written from the model HiGHS reads from the MPS file, so the
// names come from the same read here, and the dump is compared with that model (sizes, bounds,
// costs) before anything is written. Each point is evaluated on the model; the violations and
// the objective are printed, the check decides later whether it can use the point.
//
// usage: convert_solution <instance.mps[.gz]> <instance.csr> <out.gprssol> <a.sol> [<b.sol> ...]
// Prints one JSON object. Exit code 1 when a file cannot be read, a name is not a column of the
// model, or the dump is not the model of the MPS file.

#include "../check/solution_file.hpp"

#include <Highs.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct dump_t {
  int64_t nrows = 0, ncols = 0, nnz = 0;
  std::vector<double> row_lower, row_upper, col_lower, col_upper, cost;
  double offset = 0;
};

template <typename T>
static bool read_array(FILE* file, T* data, size_t count)
{
  return fread(data, sizeof(T), count, file) == count;
}

// The layout baseline_tool writes: magic, three int64 sizes, int64 row starts, int32 column
// indices, values, row bounds, column bounds, costs, one byte per column type, offset.
static bool read_dump(const std::string& path, dump_t& dump, std::string& error)
{
  FILE* file = fopen(path.c_str(), "rb");
  if (!file) {
    error = "cannot open " + path;
    return false;
  }
  char magic[8];
  bool ok = read_array(file, magic, 8) && std::string(magic, 8) == "GPRSCSR1" &&
            read_array(file, &dump.nrows, 1) && read_array(file, &dump.ncols, 1) &&
            read_array(file, &dump.nnz, 1) && dump.nrows >= 0 && dump.ncols >= 0 && dump.nnz >= 0;
  if (ok) {
    long matrix = (dump.nrows + 1) * (long)sizeof(int64_t) +
                  dump.nnz * (long)(sizeof(int32_t) + sizeof(double));
    dump.row_lower.resize(dump.nrows);
    dump.row_upper.resize(dump.nrows);
    dump.col_lower.resize(dump.ncols);
    dump.col_upper.resize(dump.ncols);
    dump.cost.resize(dump.ncols);
    ok = fseek(file, matrix, SEEK_CUR) == 0 &&
         read_array(file, dump.row_lower.data(), dump.nrows) &&
         read_array(file, dump.row_upper.data(), dump.nrows) &&
         read_array(file, dump.col_lower.data(), dump.ncols) &&
         read_array(file, dump.col_upper.data(), dump.ncols) &&
         read_array(file, dump.cost.data(), dump.ncols) && fseek(file, dump.ncols, SEEK_CUR) == 0 &&
         read_array(file, &dump.offset, 1);
  }
  fclose(file);
  if (!ok) { error = path + " is not a complete GPRSCSR1 dump"; }
  return ok;
}

static bool same_model(const dump_t& dump, const HighsLp& lp, std::string& error)
{
  bool maximize = lp.sense_ == ObjSense::kMaximize;
  if (dump.nrows != lp.num_row_ || dump.ncols != lp.num_col_ ||
      dump.nnz != (int64_t)lp.a_matrix_.numNz()) {
    error = "the dump and the MPS file differ in rows, columns or nonzeros";
    return false;
  }
  for (int64_t j = 0; j < dump.ncols; ++j) {
    double cost = maximize ? 0.0 - lp.col_cost_[j] : lp.col_cost_[j];
    if (dump.cost[j] != cost || dump.col_lower[j] != lp.col_lower_[j] ||
        dump.col_upper[j] != lp.col_upper_[j]) {
      error = "the dump and the MPS file differ in column " + std::to_string(j);
      return false;
    }
  }
  for (int64_t i = 0; i < dump.nrows; ++i) {
    if (dump.row_lower[i] != lp.row_lower_[i] || dump.row_upper[i] != lp.row_upper_[i]) {
      error = "the dump and the MPS file differ in row " + std::to_string(i);
      return false;
    }
  }
  return true;
}

struct point_t {
  std::vector<double> x;
  bool has_reported = false;
  double reported   = 0;
  int64_t named     = 0;
};

static bool read_point(const std::string& path,
                       const std::unordered_map<std::string, int64_t>& column_of,
                       int64_t ncols,
                       point_t& point,
                       std::string& error)
{
  std::ifstream in(path);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  point.x.assign(ncols, 0.0);
  std::vector<uint8_t> seen(ncols, 0);
  std::string line;
  int64_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    std::istringstream words(line);
    std::string name, value_text;
    if (!(words >> name) || name[0] == '#') { continue; }
    std::string where = path + " line " + std::to_string(line_number);
    if (!(words >> value_text)) {
      error = where + ": a name without a value";
      return false;
    }
    char* end    = nullptr;
    double value = std::strtod(value_text.c_str(), &end);
    if (end == value_text.c_str() || *end != '\0' || !std::isfinite(value)) {
      error = where + ": '" + value_text + "' is not a finite number";
      return false;
    }
    if (name == "=obj=") {
      point.has_reported = true;
      point.reported     = value;
      continue;
    }
    auto found = column_of.find(name);
    if (found == column_of.end()) {
      error = where + ": '" + name + "' is not a column of the model";
      return false;
    }
    if (seen[found->second]) {
      error = where + ": '" + name + "' has a second value";
      return false;
    }
    seen[found->second]    = 1;
    point.x[found->second] = value;
    ++point.named;
  }
  return true;
}

struct evaluation_t {
  double objective   = 0;
  double bound       = 0;
  double row         = 0;
  double integrality = 0;
};

static double excess(double value, double lower, double upper)
{
  return std::max({0.0, lower - value, value - upper});
}

static evaluation_t evaluate(const HighsLp& lp,
                             const HighsSparseMatrix& by_column,
                             const std::vector<double>& x)
{
  evaluation_t e;
  long double objective = lp.offset_;
  std::vector<long double> activity(lp.num_row_, 0.0L);
  for (HighsInt j = 0; j < lp.num_col_; ++j) {
    objective += (long double)lp.col_cost_[j] * x[j];
    e.bound = std::max(e.bound, excess(x[j], lp.col_lower_[j], lp.col_upper_[j]));
    if (!lp.integrality_.empty() && lp.integrality_[j] != HighsVarType::kContinuous) {
      e.integrality = std::max(e.integrality, std::fabs(x[j] - std::round(x[j])));
    }
    for (HighsInt k = by_column.start_[j]; k < by_column.start_[j + 1]; ++k) {
      activity[by_column.index_[k]] += (long double)by_column.value_[k] * x[j];
    }
  }
  for (HighsInt i = 0; i < lp.num_row_; ++i) {
    e.row = std::max(e.row, excess((double)activity[i], lp.row_lower_[i], lp.row_upper_[i]));
  }
  e.objective = (double)objective;
  return e;
}

int main(int argc, char** argv)
{
  if (argc < 5) {
    fprintf(stderr,
            "usage: convert_solution <instance.mps[.gz]> <instance.csr> <out.gprssol> <a.sol> "
            "[<b.sol> ...]\n");
    return 2;
  }
  std::string mps_path = argv[1], dump_path = argv[2], out_path = argv[3];

  Highs highs;
  highs.setOptionValue("output_flag", false);
  if (highs.readModel(mps_path) == HighsStatus::kError) {
    fprintf(stderr, "convert_solution: HiGHS cannot read %s\n", mps_path.c_str());
    return 1;
  }
  const HighsLp& lp = highs.getLp();
  HighsSparseMatrix converted;
  const HighsSparseMatrix* by_column = &lp.a_matrix_;
  if (!by_column->isColwise()) {
    converted = lp.a_matrix_;
    converted.ensureColwise();
    by_column = &converted;
  }
  std::string error;
  dump_t dump;
  if (!read_dump(dump_path, dump, error) || !same_model(dump, lp, error)) {
    fprintf(stderr, "convert_solution: %s\n", error.c_str());
    return 1;
  }
  if ((int64_t)lp.col_names_.size() != dump.ncols) {
    fprintf(stderr, "convert_solution: %s gives no column names\n", mps_path.c_str());
    return 1;
  }
  std::unordered_map<std::string, int64_t> column_of;
  for (int64_t j = 0; j < dump.ncols; ++j) {
    if (!column_of.emplace(lp.col_names_[j], j).second) {
      fprintf(stderr, "convert_solution: two columns are named %s\n", lp.col_names_[j].c_str());
      return 1;
    }
  }

  bool maximize = lp.sense_ == ObjSense::kMaximize;
  std::vector<point_t> points(argc - 4);
  for (int k = 4; k < argc; ++k) {
    if (!read_point(argv[k], column_of, dump.ncols, points[k - 4], error)) {
      fprintf(stderr, "convert_solution: %s\n", error.c_str());
      return 1;
    }
  }

  gprs_check::solution_writer_t writer;
  if (!writer.open(out_path, dump.ncols)) {
    fprintf(stderr, "convert_solution: cannot write %s\n", out_path.c_str());
    return 1;
  }
  printf(
    "{\"mps_file\": \"%s\", \"dump\": \"%s\", \"out\": \"%s\", \"columns\": %lld, "
    "\"objective_sense\": \"%s\", \"points\": [",
    mps_path.c_str(),
    dump_path.c_str(),
    out_path.c_str(),
    (long long)dump.ncols,
    maximize ? "max" : "min");
  for (size_t p = 0; p < points.size(); ++p) {
    evaluation_t e = evaluate(lp, *by_column, points[p].x);
    // The dump is a minimisation: the objective of a maximisation is stored negated.
    if (!writer.append(maximize ? 0.0 - e.objective : e.objective, points[p].x)) {
      fprintf(stderr, "convert_solution: cannot write %s\n", out_path.c_str());
      return 1;
    }
    printf("%s{\"file\": \"%s\", \"named_values\": %lld, \"objective\": %.17g, ",
           p ? ", " : "",
           argv[p + 4],
           (long long)points[p].named,
           e.objective);
    if (points[p].has_reported) {
      printf("\"reported_objective\": %.17g, ", points[p].reported);
    } else {
      printf("\"reported_objective\": null, ");
    }
    printf("\"bound_violation\": %.3g, \"row_violation\": %.3g, \"integrality_violation\": %.3g}",
           e.bound,
           e.row,
           e.integrality);
  }
  printf("]}\n");
  if (!writer.close()) {
    fprintf(stderr, "convert_solution: cannot write %s\n", out_path.c_str());
    return 1;
  }
  return 0;
}
