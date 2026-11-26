/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"

#include <gprs/io.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

using namespace gprs;

namespace {

// Removed when the test ends.
struct temp_file_t {
  std::string path;

  explicit temp_file_t(const char* name)
  {
    long long stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path =
      (std::filesystem::temp_directory_path() / ("gprs_test_" + std::to_string(stamp) + "_" + name))
        .string();
  }
  ~temp_file_t()
  {
    std::error_code code;
    std::filesystem::remove(path, code);
  }
};

model_t sample_model()
{
  model_t m;
  m.nrows            = 4;
  m.ncols            = 3;
  m.row_start        = {0, 2, 2, 5, 6};
  m.col_index        = {0, 2, 0, 1, 2, 1};
  m.value            = {1.5, -2., 3., 0.1, -1e-7, 4e12};
  m.row_lower        = {-inf, 0., 1., -3.25};
  m.row_upper        = {4., 0., 1., inf};
  m.col_lower        = {0., -inf, -1e30};
  m.col_upper        = {1., inf, 2.5};
  m.cost             = {1., 0., -0.};
  m.col_type         = {1, 0, 1};
  m.objective_offset = -12.5;
  return m;
}

uint64_t bits(double value)
{
  uint64_t pattern;
  std::memcpy(&pattern, &value, sizeof(pattern));
  return pattern;
}

void check_same_doubles(const std::vector<double>& a, const std::vector<double>& b)
{
  GPRS_REQUIRE(a.size() == b.size());
  for (size_t k = 0; k < a.size(); ++k) {
    GPRS_CHECK_EQ(bits(a[k]), bits(b[k]));
  }
}

void check_same_model(const model_t& a, const model_t& b)
{
  GPRS_CHECK_EQ(a.nrows, b.nrows);
  GPRS_CHECK_EQ(a.ncols, b.ncols);
  GPRS_CHECK(a.row_start == b.row_start);
  GPRS_CHECK(a.col_index == b.col_index);
  GPRS_CHECK(a.col_type == b.col_type);
  check_same_doubles(a.value, b.value);
  check_same_doubles(a.row_lower, b.row_lower);
  check_same_doubles(a.row_upper, b.row_upper);
  check_same_doubles(a.col_lower, b.col_lower);
  check_same_doubles(a.col_upper, b.col_upper);
  check_same_doubles(a.cost, b.cost);
  GPRS_CHECK_EQ(bits(a.objective_offset), bits(b.objective_offset));
}

std::string file_bytes(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void put_bytes(const std::string& path, const std::string& bytes)
{
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), bytes.size());
}

template <typename T>
T value_at(const std::string& bytes, size_t offset)
{
  T value;
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

}  // namespace

GPRS_TEST(io_round_trip)
{
  temp_file_t file("round_trip.csr");
  model_t written = sample_model();
  std::string error;
  GPRS_REQUIRE(write_csr_dump(file.path, written, error));
  model_t read;
  GPRS_REQUIRE(read_csr_dump(file.path, read, error));
  GPRS_CHECK(error.empty());
  check_same_model(written, read);

  // Reading into a model that already holds another one replaces it.
  model_t reused = sample_model();
  reused.nrows   = 1;
  reused.value.push_back(9.);
  GPRS_REQUIRE(read_csr_dump(file.path, reused, error));
  check_same_model(written, reused);
}

GPRS_TEST(io_round_trip_empty)
{
  temp_file_t file("empty.csr");
  model_t written;
  written.row_start = {0};
  std::string error;
  GPRS_REQUIRE(write_csr_dump(file.path, written, error));
  GPRS_CHECK_EQ(std::filesystem::file_size(file.path), (uintmax_t)(8 + 24 + 8 + 8));
  model_t read = sample_model();
  GPRS_REQUIRE(read_csr_dump(file.path, read, error));
  check_same_model(written, read);

  // No row and two columns; one row and no column.
  temp_file_t columns_only("columns_only.csr");
  written.ncols     = 2;
  written.col_lower = {0., -inf};
  written.col_upper = {inf, 3.};
  written.cost      = {1., -1.};
  written.col_type  = {0, 1};
  GPRS_REQUIRE(write_csr_dump(columns_only.path, written, error));
  GPRS_REQUIRE(read_csr_dump(columns_only.path, read, error));
  check_same_model(written, read);

  temp_file_t rows_only("rows_only.csr");
  model_t row;
  row.nrows     = 1;
  row.row_start = {0, 0};
  row.row_lower = {-1.};
  row.row_upper = {1.};
  GPRS_REQUIRE(write_csr_dump(rows_only.path, row, error));
  GPRS_REQUIRE(read_csr_dump(rows_only.path, read, error));
  check_same_model(row, read);
}

// The byte layout of baseline/README.md, field by field.
GPRS_TEST(io_layout)
{
  temp_file_t file("layout.csr");
  model_t m = sample_model();
  std::string error;
  GPRS_REQUIRE(write_csr_dump(file.path, m, error));
  std::string bytes = file_bytes(file.path);
  size_t nrows = 4, ncols = 3, nnz = 6;
  GPRS_REQUIRE(bytes.size() == 8 + 24 + 8 * (nrows + 1) + 12 * nnz + 16 * nrows + 25 * ncols + 8);
  GPRS_CHECK(bytes.compare(0, 8, "GPRSCSR1") == 0);
  GPRS_CHECK_EQ(value_at<int64_t>(bytes, 8), 4);
  GPRS_CHECK_EQ(value_at<int64_t>(bytes, 16), 3);
  GPRS_CHECK_EQ(value_at<int64_t>(bytes, 24), 6);
  size_t at = 32;
  GPRS_CHECK_EQ(value_at<int64_t>(bytes, at + 8 * 3), 5);
  at += 8 * (nrows + 1);
  GPRS_CHECK_EQ(value_at<int32_t>(bytes, at + 4 * 4), 2);
  at += 4 * nnz;
  GPRS_CHECK_EQ(value_at<double>(bytes, at + 8 * 5), 4e12);
  at += 8 * nnz;
  GPRS_CHECK_EQ(value_at<double>(bytes, at + 8 * 3), -3.25);
  at += 8 * nrows;
  GPRS_CHECK_EQ(value_at<double>(bytes, at), 4.);
  at += 8 * nrows;
  GPRS_CHECK_EQ(value_at<double>(bytes, at + 8 * 2), -1e30);
  at += 8 * ncols;
  GPRS_CHECK_EQ(value_at<double>(bytes, at + 8 * 1), inf);
  at += 8 * ncols;
  GPRS_CHECK_EQ(bits(value_at<double>(bytes, at + 8 * 2)), bits(-0.));
  at += 8 * ncols;
  GPRS_CHECK_EQ(value_at<uint8_t>(bytes, at), 1);
  GPRS_CHECK_EQ(value_at<uint8_t>(bytes, at + 1), 0);
  at += ncols;
  GPRS_CHECK_EQ(value_at<double>(bytes, at), -12.5);
  GPRS_CHECK_EQ(at + 8, bytes.size());
}

GPRS_TEST(io_read_rejects)
{
  temp_file_t file("rejects.csr");
  model_t m = sample_model();
  std::string error;
  GPRS_REQUIRE(write_csr_dump(file.path, m, error));
  std::string good = file_bytes(file.path);
  model_t read;

  GPRS_CHECK(!read_csr_dump(file.path + ".missing", read, error));
  GPRS_CHECK(!error.empty());

  std::string bad = good;
  bad[7]          = '2';
  put_bytes(file.path, bad);
  GPRS_CHECK(!read_csr_dump(file.path, read, error));

  put_bytes(file.path, good.substr(0, good.size() - 1));
  GPRS_CHECK(!read_csr_dump(file.path, read, error));

  put_bytes(file.path, good + "x");
  GPRS_CHECK(!read_csr_dump(file.path, read, error));

  put_bytes(file.path, good.substr(0, 20));
  GPRS_CHECK(!read_csr_dump(file.path, read, error));

  // A header that claims more nonzeros than the file can hold.
  bad          = good;
  int64_t huge = 1ll << 60;
  std::memcpy(&bad[24], &huge, 8);
  put_bytes(file.path, bad);
  GPRS_CHECK(!read_csr_dump(file.path, read, error));

  int64_t negative = -1;
  bad              = good;
  std::memcpy(&bad[8], &negative, 8);
  put_bytes(file.path, bad);
  GPRS_CHECK(!read_csr_dump(file.path, read, error));

  // Right size, columns of row 2 not increasing.
  bad            = good;
  int32_t column = 0;
  std::memcpy(&bad[32 + 8 * 5 + 4 * 3], &column, 4);
  put_bytes(file.path, bad);
  GPRS_CHECK(!read_csr_dump(file.path, read, error));
  GPRS_CHECK(error.find("well-formed") != std::string::npos);

  put_bytes(file.path, good);
  GPRS_CHECK(read_csr_dump(file.path, read, error));
  check_same_model(m, read);
}

GPRS_TEST(io_write_reports_unwritable_path)
{
  std::string error;
  std::string path =
    (std::filesystem::temp_directory_path() / "gprs_no_such_directory" / "model.csr").string();
  GPRS_CHECK(!write_csr_dump(path, sample_model(), error));
  GPRS_CHECK(!error.empty());
}
