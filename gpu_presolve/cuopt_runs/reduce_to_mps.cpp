/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Writes the model the host engine leaves of a CSR dump as a free-format MPS file, so that any
// solver can be run on exactly that model. Rows are named R<i> and columns C<j> by their index
// in the reduced model. The file is a minimisation, as the dump is.
//
// usage: reduce_to_mps <in.csr> <out.mps> [--primal-only] [--no-presolve]
// --no-presolve writes the dump itself. Exit code 1 when the dump cannot be read, the presolve
// returns no model, or the file cannot be written.

#include <gprs/io.hpp>
#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdio>
#include <string>

static bool write_mps(const std::string& path, const gprs::model_t& m)
{
  FILE* f = fopen(path.c_str(), "w");
  if (!f) { return false; }
  fprintf(f, "NAME reduced\nROWS\n N OBJ\n");
  std::vector<uint8_t> kept(m.nrows, 1);
  for (int32_t i = 0; i < m.nrows; ++i) {
    bool low = std::isfinite(m.row_lower[i]), up = std::isfinite(m.row_upper[i]);
    if (!low && !up) {
      kept[i] = 0;
      continue;
    }
    char type = low && up && m.row_lower[i] == m.row_upper[i] ? 'E' : low ? 'G' : 'L';
    fprintf(f, " %c R%d\n", type, i);
  }

  // Column by column, as the format wants: the entries of the rows are spread first.
  std::vector<int64_t> start(m.ncols + 1, 0);
  for (int64_t p = 0; p < m.nnz(); ++p) {
    ++start[m.col_index[p] + 1];
  }
  for (int32_t j = 0; j < m.ncols; ++j) {
    start[j + 1] += start[j];
  }
  std::vector<int32_t> row(m.nnz());
  std::vector<double> value(m.nnz());
  std::vector<int64_t> next(start.begin(), start.end() - 1);
  for (int32_t i = 0; i < m.nrows; ++i) {
    for (int64_t p = m.row_start[i]; p < m.row_start[i + 1]; ++p) {
      int64_t q = next[m.col_index[p]]++;
      row[q]    = i;
      value[q]  = m.value[p];
    }
  }
  fprintf(f, "COLUMNS\n");
  bool in_integers = false;
  int marker       = 0;
  for (int32_t j = 0; j < m.ncols; ++j) {
    bool integer = m.col_type[j] != 0;
    if (integer != in_integers) {
      fprintf(f, " MARKER%d 'MARKER' '%s'\n", marker++, integer ? "INTORG" : "INTEND");
      in_integers = integer;
    }
    // Every column gets an objective entry, so a column without entries is still declared.
    fprintf(f, " C%d OBJ %.17g\n", j, m.cost[j]);
    for (int64_t q = start[j]; q < start[j + 1]; ++q) {
      if (kept[row[q]]) { fprintf(f, " C%d R%d %.17g\n", j, row[q], value[q]); }
    }
  }
  if (in_integers) { fprintf(f, " MARKER%d 'MARKER' 'INTEND'\n", marker++); }

  fprintf(f, "RHS\n");
  // The objective constant is minus the right-hand side of the objective row.
  if (m.objective_offset != 0.) { fprintf(f, " RHS OBJ %.17g\n", 0.0 - m.objective_offset); }
  for (int32_t i = 0; i < m.nrows; ++i) {
    if (!kept[i]) { continue; }
    double side = std::isfinite(m.row_lower[i]) ? m.row_lower[i] : m.row_upper[i];
    if (side != 0.) { fprintf(f, " RHS R%d %.17g\n", i, side); }
  }
  // A G row with a range R is rhs <= row <= rhs + |R|.
  bool ranges = false;
  for (int32_t i = 0; i < m.nrows; ++i) {
    if (std::isfinite(m.row_lower[i]) && std::isfinite(m.row_upper[i]) &&
        m.row_lower[i] != m.row_upper[i]) {
      if (!ranges) { fprintf(f, "RANGES\n"); }
      ranges = true;
      fprintf(f, " RNG R%d %.17g\n", i, m.row_upper[i] - m.row_lower[i]);
    }
  }
  // Both bounds of every column are written: the defaults differ between readers.
  fprintf(f, "BOUNDS\n");
  for (int32_t j = 0; j < m.ncols; ++j) {
    double lower = m.col_lower[j], upper = m.col_upper[j];
    if (!std::isfinite(lower) && !std::isfinite(upper)) {
      fprintf(f, " FR BND C%d\n", j);
    } else if (lower == upper) {
      fprintf(f, " FX BND C%d %.17g\n", j, lower);
    } else {
      if (std::isfinite(lower)) {
        fprintf(f, " LO BND C%d %.17g\n", j, lower);
      } else {
        fprintf(f, " MI BND C%d\n", j);
      }
      if (std::isfinite(upper)) {
        fprintf(f, " UP BND C%d %.17g\n", j, upper);
      } else {
        fprintf(f, " PL BND C%d\n", j);
      }
    }
  }
  fprintf(f, "ENDATA\n");
  return fclose(f) == 0;
}

int main(int argc, char** argv)
{
  std::string in, out;
  bool primal_only = false, presolve = true;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--primal-only") {
      primal_only = true;
    } else if (arg == "--no-presolve") {
      presolve = false;
    } else if (in.empty()) {
      in = arg;
    } else {
      out = arg;
    }
  }
  if (out.empty()) {
    fprintf(stderr, "usage: reduce_to_mps <in.csr> <out.mps> [--primal-only] [--no-presolve]\n");
    return 2;
  }
  gprs::model_t model;
  std::string error;
  if (!gprs::read_csr_dump(in, model, error)) {
    fprintf(stderr, "reduce_to_mps: %s\n", error.c_str());
    return 1;
  }
  if (presolve) {
    gprs::options_t options;
    options.primal_only   = primal_only;
    gprs::result_t result = gprs::presolve(model, options);
    printf("status %s, %d rows, %d columns, %lld nonzeros, objective offset %.17g\n",
           gprs::status_name(result.status),
           result.reduced.nrows,
           result.reduced.ncols,
           (long long)result.reduced.nnz(),
           result.reduced.objective_offset);
    if (result.status != gprs::status_t::reduced && result.status != gprs::status_t::unchanged) {
      fprintf(stderr, "reduce_to_mps: the presolve returned no model to write\n");
      return 1;
    }
    model = std::move(result.reduced);
  }
  if (!write_mps(out, model)) {
    fprintf(stderr, "reduce_to_mps: cannot write %s\n", out.c_str());
    return 1;
  }
  return 0;
}
