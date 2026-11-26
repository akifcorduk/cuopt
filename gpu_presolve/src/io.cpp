/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/io.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <system_error>

namespace gprs {

namespace {

constexpr char magic[8] = {'G', 'P', 'R', 'S', 'C', 'S', 'R', '1'};

struct file_t {
  FILE* handle = nullptr;
  ~file_t()
  {
    if (handle) { std::fclose(handle); }
  }
};

bool little_endian()
{
  uint32_t one = 1;
  unsigned char first;
  std::memcpy(&first, &one, 1);
  return first == 1;
}

template <typename T>
bool read_array(FILE* file, std::vector<T>& data, int64_t count)
{
  data.resize(count);
  return count == 0 || std::fread(data.data(), sizeof(T), count, file) == (size_t)count;
}

template <typename T>
bool write_array(FILE* file, const T* data, int64_t count)
{
  return count == 0 || std::fwrite(data, sizeof(T), count, file) == (size_t)count;
}

}  // namespace

bool read_csr_dump(const std::string& path, model_t& model, std::string& error)
{
  gprs_assert(little_endian());
  std::error_code code;
  uintmax_t file_size = std::filesystem::file_size(path, code);
  file_t file;
  file.handle = code ? nullptr : std::fopen(path.c_str(), "rb");
  if (!file.handle) {
    error = "cannot open " + path;
    return false;
  }

  char found[8];
  int64_t header[3];
  if (std::fread(found, 1, 8, file.handle) != 8 || std::memcmp(found, magic, 8) != 0 ||
      std::fread(header, sizeof(int64_t), 3, file.handle) != 3) {
    error = path + " is not a GPRSCSR1 dump";
    return false;
  }
  int64_t nrows = header[0], ncols = header[1], nnz = header[2];
  int64_t max_index = std::numeric_limits<int32_t>::max();
  if (nrows < 0 || nrows > max_index || ncols < 0 || ncols > max_index || nnz < 0 ||
      (uintmax_t)nnz > file_size / 12) {
    error = path + " has a header outside the supported sizes";
    return false;
  }
  int64_t expected = 8 + 24 + 8 * (nrows + 1) + 12 * nnz + 16 * nrows + 25 * ncols + 8;
  if ((uintmax_t)expected != file_size) {
    error = path + " does not have the size its header gives";
    return false;
  }

  model.nrows = nrows;
  model.ncols = ncols;
  FILE* f     = file.handle;
  if (!read_array(f, model.row_start, nrows + 1) || !read_array(f, model.col_index, nnz) ||
      !read_array(f, model.value, nnz) || !read_array(f, model.row_lower, nrows) ||
      !read_array(f, model.row_upper, nrows) || !read_array(f, model.col_lower, ncols) ||
      !read_array(f, model.col_upper, ncols) || !read_array(f, model.cost, ncols) ||
      !read_array(f, model.col_type, ncols) ||
      std::fread(&model.objective_offset, sizeof(double), 1, f) != 1) {
    error = "cannot read " + path;
    return false;
  }
  if (!well_formed(model)) {
    error = path + " does not hold a well-formed model";
    return false;
  }
  return true;
}

bool write_csr_dump(const std::string& path, const model_t& model, std::string& error)
{
  gprs_assert(little_endian());
  gprs_assert(well_formed(model));
  file_t file;
  file.handle = std::fopen(path.c_str(), "wb");
  if (!file.handle) {
    error = "cannot open " + path;
    return false;
  }
  int64_t header[3] = {model.nrows, model.ncols, model.nnz()};
  int64_t nrows = model.nrows, ncols = model.ncols, nnz = model.nnz();
  FILE* f = file.handle;
  bool written =
    write_array(f, magic, 8) && write_array(f, header, 3) &&
    write_array(f, model.row_start.data(), nrows + 1) &&
    write_array(f, model.col_index.data(), nnz) && write_array(f, model.value.data(), nnz) &&
    write_array(f, model.row_lower.data(), nrows) &&
    write_array(f, model.row_upper.data(), nrows) &&
    write_array(f, model.col_lower.data(), ncols) &&
    write_array(f, model.col_upper.data(), ncols) && write_array(f, model.cost.data(), ncols) &&
    write_array(f, model.col_type.data(), ncols) && write_array(f, &model.objective_offset, 1);
  int closed  = std::fclose(file.handle);
  file.handle = nullptr;
  if (!written || closed != 0) {
    error = "cannot write " + path;
    return false;
  }
  return true;
}

}  // namespace gprs
