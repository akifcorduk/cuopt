/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Solution file "GPRSSOL1" of one instance, little endian, kept outside the repository:
//   char magic[8]; int64 ncols; then records up to the end of the file, each
//   double objective; double value[ncols].
// A record is one point of the original model: one value per column of the CSR dump of the
// instance, in the dump's column order. objective is what the producer of the point reported
// for the minimisation in the dump; a reader recomputes it.
// The file is written record by record, so a writer that is killed leaves the records it
// finished and at most one incomplete record at the end, which a reader ignores and counts.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gprs_check {

constexpr char solution_magic[8] = {'G', 'P', 'R', 'S', 'S', 'O', 'L', '1'};

struct solution_set_t {
  int64_t ncols = 0;
  std::vector<double> objective;
  std::vector<std::vector<double>> points;
  // Bytes of an incomplete record at the end of the file.
  int64_t tail_bytes = 0;
};

// Reads every complete record. False with a message when the file cannot be read or is not a
// solution file.
inline bool read_solutions(const std::string& path, solution_set_t& set, std::string& error)
{
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    error = "cannot open " + path;
    return false;
  }
  char magic[8];
  set = solution_set_t();
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, solution_magic, 8) != 0 ||
      std::fread(&set.ncols, sizeof(int64_t), 1, f) != 1 || set.ncols < 0) {
    std::fclose(f);
    error = path + " is not a GPRSSOL1 solution file";
    return false;
  }
  while (true) {
    double objective;
    size_t got = std::fread(&objective, 1, sizeof(double), f);
    if (got == 0) { break; }
    std::vector<double> x(set.ncols);
    size_t values = got == sizeof(double) ? std::fread(x.data(), sizeof(double), set.ncols, f) : 0;
    if (got != sizeof(double) || values != (size_t)set.ncols) {
      set.tail_bytes = got + values * sizeof(double);
      break;
    }
    set.objective.push_back(objective);
    set.points.push_back(std::move(x));
  }
  std::fclose(f);
  return true;
}

// Writes a solution file record by record; every record is on disk when append returns.
struct solution_writer_t {
  FILE* file     = nullptr;
  int64_t ncols  = 0;
  int64_t points = 0;

  ~solution_writer_t() { close(); }

  bool open(const std::string& path, int64_t columns)
  {
    file   = std::fopen(path.c_str(), "wb");
    ncols  = columns;
    points = 0;
    return file && std::fwrite(solution_magic, 1, 8, file) == 8 &&
           std::fwrite(&ncols, sizeof(int64_t), 1, file) == 1 && std::fflush(file) == 0;
  }

  bool write_record(double objective, const std::vector<double>& x)
  {
    return std::fwrite(&objective, sizeof(double), 1, file) == 1 &&
           std::fwrite(x.data(), sizeof(double), ncols, file) == (size_t)ncols &&
           std::fflush(file) == 0;
  }

  bool append(double objective, const std::vector<double>& x)
  {
    ++points;
    return std::fseek(file, 0, SEEK_END) == 0 && write_record(objective, x);
  }

  // Overwrites the last record, for a writer that keeps a bounded number of points.
  bool replace_last(double objective, const std::vector<double>& x)
  {
    long offset = 16 + (points - 1) * (long)(sizeof(double) * (ncols + 1));
    return std::fseek(file, offset, SEEK_SET) == 0 && write_record(objective, x);
  }

  bool close()
  {
    bool ok = !file || std::fclose(file) == 0;
    file    = nullptr;
    return ok;
  }
};

}  // namespace gprs_check
