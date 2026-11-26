/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <Highs.h>

#include <dlfcn.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static const double inf = std::numeric_limits<double>::infinity();

struct csr_t {
  int64_t nrows = 0;
  int64_t ncols = 0;
  int64_t nnz   = 0;
  std::vector<int64_t> row_start;
  std::vector<int32_t> col_index;
  std::vector<double> value;
};

struct json_writer_t {
  FILE* file;
  std::vector<int64_t> items{0};

  void write_string(const std::string& s)
  {
    fputc('"', file);
    for (unsigned char c : s) {
      if (c == '"' || c == '\\')
        fprintf(file, "\\%c", c);
      else if (c < 0x20)
        fprintf(file, "\\u%04x", c);
      else
        fputc(c, file);
    }
    fputc('"', file);
  }

  void begin_item(const char* key)
  {
    if (items.back()++ > 0) fputc(',', file);
    fputc('\n', file);
    for (size_t i = 0; i < items.size(); ++i)
      fputs("  ", file);
    write_string(key);
    fputs(": ", file);
  }

  void open(const char* key)
  {
    begin_item(key);
    fputc('{', file);
    items.push_back(0);
  }

  void close()
  {
    items.pop_back();
    fputc('\n', file);
    for (size_t i = 0; i < items.size(); ++i)
      fputs("  ", file);
    fputc('}', file);
  }

  void put_int(const char* key, int64_t v)
  {
    begin_item(key);
    fprintf(file, "%lld", (long long)v);
  }

  void put_real(const char* key, double v)
  {
    begin_item(key);
    if (std::isfinite(v))
      fprintf(file, "%.17g", v);
    else
      fputs("null", file);
  }

  void put_bool(const char* key, bool v)
  {
    begin_item(key);
    fputs(v ? "true" : "false", file);
  }

  void put_string(const char* key, const std::string& v)
  {
    begin_item(key);
    write_string(v);
  }

  void put_null(const char* key)
  {
    begin_item(key);
    fputs("null", file);
  }
};

using time_point_t = std::chrono::steady_clock::time_point;

static double seconds_between(time_point_t start, time_point_t end)
{
  return std::chrono::duration<double>(end - start).count();
}

static double seconds_since(time_point_t start)
{
  return seconds_between(start, std::chrono::steady_clock::now());
}

// HiGHS logs "Presolving model" when its MIP presolve starts working on the model and
// "Presolve reductions" right after it ends (not when the model is found infeasible or
// unbounded). The times of the two messages split the presolve() call into set-up, the presolve
// itself and the hand-over of the reduced model.
struct presolve_marks_t {
  bool started = false;
  bool ended   = false;
  time_point_t start;
  time_point_t end;
};

static void on_log_message(
  int, const std::string& message, const HighsCallbackOutput*, HighsCallbackInput*, void* data)
{
  presolve_marks_t* marks = (presolve_marks_t*)data;
  if (!marks->started && message.rfind("Presolving model", 0) == 0) {
    marks->started = true;
    marks->start   = std::chrono::steady_clock::now();
  } else if (!marks->ended && message.rfind("Presolve reductions", 0) == 0) {
    marks->ended = true;
    marks->end   = std::chrono::steady_clock::now();
  }
}

// Rows come out sorted by column index because the columns are visited in order.
static csr_t build_csr(const HighsLp& lp)
{
  HighsSparseMatrix converted;
  const HighsSparseMatrix* a = &lp.a_matrix_;
  if (!a->isColwise()) {
    converted = *a;
    converted.ensureColwise();
    a = &converted;
  }
  assert(lp.num_col_ <= std::numeric_limits<int32_t>::max());
  csr_t m;
  m.nrows = lp.num_row_;
  m.ncols = lp.num_col_;
  m.nnz   = a->start_[m.ncols];
  m.row_start.assign(m.nrows + 1, 0);
  for (int64_t k = 0; k < m.nnz; ++k)
    ++m.row_start[a->index_[k] + 1];
  for (int64_t i = 0; i < m.nrows; ++i)
    m.row_start[i + 1] += m.row_start[i];
  std::vector<int64_t> next(m.row_start.begin(), m.row_start.end() - 1);
  m.col_index.resize(m.nnz);
  m.value.resize(m.nnz);
  for (int64_t j = 0; j < m.ncols; ++j) {
    for (int64_t k = a->start_[j]; k < a->start_[j + 1]; ++k) {
      int64_t p      = next[a->index_[k]]++;
      m.col_index[p] = j;
      m.value[p]     = a->value_[k];
    }
  }
  return m;
}

// Nearest-rank percentile of a sorted, non-empty vector.
template <typename T>
static T percentile(const std::vector<T>& sorted, int64_t p)
{
  int64_t n = sorted.size();
  return sorted[std::max<int64_t>((p * n + 99) / 100 - 1, 0)];
}

static void write_length_stats(json_writer_t& js, const char* key, std::vector<int64_t> length)
{
  js.open(key);
  if (!length.empty()) {
    std::sort(length.begin(), length.end());
    int64_t total = 0;
    for (int64_t v : length)
      total += v;
    js.put_int("min", length.front());
    js.put_int("p10", percentile(length, 10));
    js.put_int("p50", percentile(length, 50));
    js.put_int("p90", percentile(length, 90));
    js.put_int("p99", percentile(length, 99));
    js.put_int("max", length.back());
    js.put_real("mean", (double)total / length.size());
  }
  js.close();
}

static void write_row_dynamism(json_writer_t& js,
                               const char* key,
                               const csr_t& m,
                               const std::vector<double>& value)
{
  std::vector<double> dynamism;
  for (int64_t i = 0; i < m.nrows; ++i) {
    if (m.row_start[i] == m.row_start[i + 1]) continue;
    double lo = inf, hi = 0;
    for (int64_t k = m.row_start[i]; k < m.row_start[i + 1]; ++k) {
      lo = std::min(lo, std::abs(value[k]));
      hi = std::max(hi, std::abs(value[k]));
    }
    dynamism.push_back(hi / lo);
  }
  js.open(key);
  js.put_int("rows", dynamism.size());
  if (!dynamism.empty()) {
    std::sort(dynamism.begin(), dynamism.end());
    js.put_real("p50", percentile(dynamism, 50));
    js.put_real("p99", percentile(dynamism, 99));
    js.put_real("max", dynamism.back());
  }
  js.close();
}

// The equilibration HiGHS applies at the end of its MIP presolve (HPresolve::scaleMIP): rows
// with a continuous entry are scaled by a power of two so that their largest continuous
// coefficient is near 1, then continuous columns so that their largest coefficient is near 1.
static std::vector<double> highs_mip_scaling(const csr_t& m, const std::vector<uint8_t>& is_integer)
{
  std::vector<double> value = m.value;
  for (int64_t i = 0; i < m.nrows; ++i) {
    double largest = 0;
    for (int64_t k = m.row_start[i]; k < m.row_start[i + 1]; ++k)
      if (!is_integer[m.col_index[k]]) largest = std::max(largest, std::abs(value[k]));
    if (largest == 0) continue;
    double scale = std::exp2(std::round(-std::log2(largest)));
    for (int64_t k = m.row_start[i]; k < m.row_start[i + 1]; ++k)
      value[k] *= scale;
  }
  std::vector<double> col_largest(m.ncols, 0);
  for (int64_t k = 0; k < m.nnz; ++k)
    col_largest[m.col_index[k]] = std::max(col_largest[m.col_index[k]], std::abs(value[k]));
  for (int64_t k = 0; k < m.nnz; ++k) {
    int32_t j = m.col_index[k];
    if (!is_integer[j]) value[k] *= std::exp2(std::round(-std::log2(col_largest[j])));
  }
  return value;
}

enum row_class_t {
  ROW_EMPTY,
  ROW_SINGLETON,
  ROW_DOUBLETON_EQUATION,
  ROW_SET_PACKING,
  ROW_SET_PARTITIONING,
  ROW_SET_COVERING,
  ROW_CARDINALITY,
  ROW_BINARY_KNAPSACK,
  ROW_VARIABLE_BOUND,
  ROW_OTHER_TWO_ENTRY,
  ROW_ALL_INTEGER,
  ROW_MIXED,
  ROW_ALL_CONTINUOUS,
  ROW_CLASS_COUNT
};

static const char* row_class_name[ROW_CLASS_COUNT] = {"empty",
                                                      "singleton",
                                                      "doubleton_equation",
                                                      "set_packing",
                                                      "set_partitioning",
                                                      "set_covering",
                                                      "cardinality",
                                                      "binary_knapsack_non_unit",
                                                      "variable_bound",
                                                      "other_two_entry",
                                                      "all_integer",
                                                      "mixed",
                                                      "all_continuous"};

// A row over binaries whose coefficients all have absolute value `scale`, n_pos of them positive
// and n_neg negative. Complementing the negative columns gives a sum of literals with sides
// shifted by n_neg; complementing the positive ones instead gives the mirrored row. A side that
// the literal sum cannot violate is dropped. The row is set packing, partitioning or covering
// when one of the two forms reads sum <= 1, sum = 1 or sum >= 1.
static row_class_t classify_unit_binary_row(
  double lower, double upper, double scale, int64_t n_pos, int64_t n_neg)
{
  lower = std::ceil(lower / scale - 1e-9);
  upper = std::floor(upper / scale + 1e-9);
  if (lower <= -n_neg) lower = -inf;
  if (upper >= n_pos) upper = inf;
  double one_direct   = 1 - n_neg;
  double one_mirrored = n_pos - 1;
  if (lower == upper)
    return lower == one_direct || lower == one_mirrored ? ROW_SET_PARTITIONING : ROW_CARDINALITY;
  if (lower == -inf && upper == one_direct) return ROW_SET_PACKING;
  if (upper == inf && lower == one_mirrored) return ROW_SET_PACKING;
  if (upper == inf && lower == one_direct) return ROW_SET_COVERING;
  if (lower == -inf && upper == one_mirrored) return ROW_SET_COVERING;
  return ROW_CARDINALITY;
}

static void write_structure(
  json_writer_t& js, const char* key, const csr_t& m, const HighsLp& lp, bool with_highs_scaling)
{
  const double huge_bound = 1e8;

  std::vector<uint8_t> is_integer(m.ncols), is_binary(m.ncols), has_huge_bound(m.ncols);
  std::vector<uint8_t> lower_infinite(m.ncols), upper_infinite(m.ncols);
  int64_t n_binary = 0, n_general_integer = 0, n_implied_integer = 0, n_continuous = 0, n_semi = 0;
  int64_t n_lower_infinite = 0, n_upper_infinite = 0, n_free_cols = 0;
  int64_t n_lower_huge = 0, n_upper_huge = 0, n_cols_huge = 0;
  for (int64_t j = 0; j < m.ncols; ++j) {
    HighsVarType type = lp.integrality_.empty() ? HighsVarType::kContinuous : lp.integrality_[j];
    double lower      = lp.col_lower_[j];
    double upper      = lp.col_upper_[j];
    is_integer[j]     = type != HighsVarType::kContinuous && type != HighsVarType::kSemiContinuous;
    is_binary[j]      = is_integer[j] && lower == 0 && upper == 1;
    if (type == HighsVarType::kContinuous)
      ++n_continuous;
    else if (type == HighsVarType::kImplicitInteger)
      ++n_implied_integer;
    else if (type != HighsVarType::kInteger)
      ++n_semi;
    else if (is_binary[j])
      ++n_binary;
    else
      ++n_general_integer;
    lower_infinite[j] = lower == -inf;
    upper_infinite[j] = upper == inf;
    bool lower_huge   = std::isfinite(lower) && std::abs(lower) >= huge_bound;
    bool upper_huge   = std::isfinite(upper) && std::abs(upper) >= huge_bound;
    has_huge_bound[j] = lower_huge || upper_huge;
    n_lower_infinite += lower_infinite[j];
    n_upper_infinite += upper_infinite[j];
    n_free_cols += lower_infinite[j] && upper_infinite[j];
    n_lower_huge += lower_huge;
    n_upper_huge += upper_huge;
    n_cols_huge += has_huge_bound[j];
  }

  std::vector<int64_t> row_length(m.nrows), col_length(m.ncols, 0);
  int64_t n_equation = 0, n_ranged = 0, n_one_sided = 0, n_free_rows = 0;
  int64_t row_class_count[ROW_CLASS_COUNT] = {};
  int64_t n_two_binary_rows = 0, n_rows_with_huge_bound = 0;
  int64_t min_activity_terms[3] = {}, max_activity_terms[3] = {};
  double coef_min = inf, coef_max = 0;
  for (int64_t i = 0; i < m.nrows; ++i) {
    int64_t begin = m.row_start[i], end = m.row_start[i + 1];
    int64_t length = end - begin;
    double lower   = lp.row_lower_[i];
    double upper   = lp.row_upper_[i];
    row_length[i]  = length;

    bool equation    = lower == upper;
    int64_t n_finite = std::isfinite(lower) + std::isfinite(upper);
    n_equation += equation;
    n_ranged += n_finite == 2 && !equation;
    n_one_sided += n_finite == 1;
    n_free_rows += n_finite == 0;

    int64_t n_bin = 0, n_int = 0, n_pos = 0, n_min_infinite = 0, n_max_infinite = 0;
    bool equal_abs = true, huge = false;
    for (int64_t k = begin; k < end; ++k) {
      int32_t j        = m.col_index[k];
      double a         = m.value[k];
      double magnitude = std::abs(a);
      ++col_length[j];
      coef_min = std::min(coef_min, magnitude);
      coef_max = std::max(coef_max, magnitude);
      n_bin += is_binary[j];
      n_int += is_integer[j];
      n_pos += a > 0;
      equal_abs &= magnitude == std::abs(m.value[begin]);
      huge |= has_huge_bound[j];
      n_min_infinite += a > 0 ? lower_infinite[j] : upper_infinite[j];
      n_max_infinite += a > 0 ? upper_infinite[j] : lower_infinite[j];
    }
    n_rows_with_huge_bound += huge;
    n_two_binary_rows += length == 2 && n_bin == 2;
    ++min_activity_terms[std::min<int64_t>(n_min_infinite, 2)];
    ++max_activity_terms[std::min<int64_t>(n_max_infinite, 2)];

    row_class_t row_class;
    if (length == 0)
      row_class = ROW_EMPTY;
    else if (length == 1)
      row_class = ROW_SINGLETON;
    else if (length == 2 && equation)
      row_class = ROW_DOUBLETON_EQUATION;
    else if (n_bin == length)
      row_class = equal_abs ? classify_unit_binary_row(
                                lower, upper, std::abs(m.value[begin]), n_pos, length - n_pos)
                            : ROW_BINARY_KNAPSACK;
    else if (length == 2)
      row_class = n_bin == 1 ? ROW_VARIABLE_BOUND : ROW_OTHER_TWO_ENTRY;
    else if (n_int == length)
      row_class = ROW_ALL_INTEGER;
    else if (n_int == 0)
      row_class = ROW_ALL_CONTINUOUS;
    else
      row_class = ROW_MIXED;
    ++row_class_count[row_class];
  }

  int64_t long_col_threshold = std::max<int64_t>(1000, m.nrows / 100);
  int64_t long_row_threshold = std::max<int64_t>(1000, m.ncols / 100);
  int64_t n_long_cols = 0, n_long_rows = 0;
  for (int64_t v : col_length)
    n_long_cols += v > long_col_threshold;
  for (int64_t v : row_length)
    n_long_rows += v > long_row_threshold;

  js.open(key);
  js.put_int("rows", m.nrows);
  js.put_int("cols", m.ncols);
  js.put_int("nnz", m.nnz);

  js.open("col_types");
  js.put_int("binary", n_binary);
  js.put_int("general_integer", n_general_integer);
  js.put_int("implied_integer", n_implied_integer);
  js.put_int("continuous", n_continuous);
  js.put_int("semi", n_semi);
  js.close();

  js.open("row_sides");
  js.put_int("equation", n_equation);
  js.put_int("ranged", n_ranged);
  js.put_int("one_sided", n_one_sided);
  js.put_int("free", n_free_rows);
  js.close();

  write_length_stats(js, "row_length", row_length);
  write_length_stats(js, "col_length", col_length);

  js.open("long_cols");
  js.put_int("threshold", long_col_threshold);
  js.put_int("count", n_long_cols);
  js.close();
  js.open("long_rows");
  js.put_int("threshold", long_row_threshold);
  js.put_int("count", n_long_rows);
  js.close();

  js.open("row_classes");
  for (int c = 0; c < ROW_CLASS_COUNT; ++c)
    js.put_int(row_class_name[c], row_class_count[c]);
  js.close();
  js.put_int("two_binary_rows", n_two_binary_rows);

  js.open("coefficient");
  if (m.nnz > 0) {
    js.put_real("min_abs", coef_min);
    js.put_real("max_abs", coef_max);
  }
  js.close();

  write_row_dynamism(js, "row_dynamism", m, m.value);
  if (with_highs_scaling)
    write_row_dynamism(js, "row_dynamism_highs_scaled", m, highs_mip_scaling(m, is_integer));

  js.open("bounds");
  js.put_int("lower_infinite", n_lower_infinite);
  js.put_int("upper_infinite", n_upper_infinite);
  js.put_int("free_cols", n_free_cols);
  js.put_real("huge_threshold", huge_bound);
  js.put_int("lower_huge", n_lower_huge);
  js.put_int("upper_huge", n_upper_huge);
  js.put_int("cols_with_huge_bound", n_cols_huge);
  js.put_int("rows_with_huge_bound_col", n_rows_with_huge_bound);
  js.close();

  static const char* term_count_name[3] = {"none", "one", "two_or_more"};
  js.open("rows_by_infinite_activity_terms");
  js.open("min_activity");
  for (int c = 0; c < 3; ++c)
    js.put_int(term_count_name[c], min_activity_terms[c]);
  js.close();
  js.open("max_activity");
  for (int c = 0; c < 3; ++c)
    js.put_int(term_count_name[c], max_activity_terms[c]);
  js.close();
  js.close();

  js.close();
}

template <typename T>
static void write_array(FILE* file, const T* data, size_t count)
{
  size_t written = fwrite(data, sizeof(T), count, file);
  assert(written == count);
}

static int64_t dump_csr(const std::string& path, const csr_t& m, const HighsLp& lp)
{
  // The file has no sense field: a maximisation is written as the equivalent minimisation.
  // 0.0 - x keeps zero costs at +0.0.
  bool maximize = lp.sense_ == ObjSense::kMaximize;
  std::vector<double> cost(m.ncols);
  std::vector<uint8_t> col_type(m.ncols);
  for (int64_t j = 0; j < m.ncols; ++j) {
    HighsVarType type = lp.integrality_.empty() ? HighsVarType::kContinuous : lp.integrality_[j];
    assert(type == HighsVarType::kContinuous || type == HighsVarType::kInteger);
    cost[j]     = maximize ? 0.0 - lp.col_cost_[j] : lp.col_cost_[j];
    col_type[j] = type == HighsVarType::kInteger;
  }
  double offset = maximize ? 0.0 - lp.offset_ : lp.offset_;

  FILE* file = fopen(path.c_str(), "wb");
  assert(file);
  write_array(file, "GPRSCSR1", 8);
  write_array(file, &m.nrows, 1);
  write_array(file, &m.ncols, 1);
  write_array(file, &m.nnz, 1);
  write_array(file, m.row_start.data(), m.row_start.size());
  write_array(file, m.col_index.data(), m.col_index.size());
  write_array(file, m.value.data(), m.value.size());
  write_array(file, lp.row_lower_.data(), m.nrows);
  write_array(file, lp.row_upper_.data(), m.nrows);
  write_array(file, lp.col_lower_.data(), m.ncols);
  write_array(file, lp.col_upper_.data(), m.ncols);
  write_array(file, cost.data(), m.ncols);
  write_array(file, col_type.data(), m.ncols);
  write_array(file, &offset, 1);
  int64_t bytes = ftell(file);
  int closed    = fclose(file);
  assert(closed == 0);
  return bytes;
}

template <typename T>
static void set_option(Highs& highs, const char* name, T value)
{
  HighsStatus status = highs.setOptionValue(name, value);
  assert(status == HighsStatus::kOk);
}

static std::string instance_name(const std::string& path)
{
  size_t slash     = path.find_last_of('/');
  std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  for (const char* suffix : {".gz", ".mps"}) {
    size_t n = strlen(suffix);
    if (name.size() > n && name.compare(name.size() - n, n, suffix) == 0)
      name.resize(name.size() - n);
  }
  return name;
}

static const char* highs_status_name(HighsStatus status)
{
  return status == HighsStatus::kOk ? "ok" : status == HighsStatus::kWarning ? "warning" : "error";
}

static bool has_presolved_model(HighsPresolveStatus status)
{
  return status == HighsPresolveStatus::kReduced ||
         status == HighsPresolveStatus::kReducedToEmpty ||
         status == HighsPresolveStatus::kNotReduced;
}

// The conda toolchain puts its own library directory first in the run path, so the HiGHS that
// gets loaded is checked against the install this binary was built for.
static std::string loaded_highs_library()
{
  Dl_info info;
  int found = dladdr((void*)&highsVersion, &info);
  assert(found);
  char* loaded = realpath(info.dli_fname, nullptr);
  char* root   = realpath(BASELINE_HIGHS_ROOT, nullptr);
  assert(loaded && root);
  size_t n = strlen(root);
  assert(strncmp(loaded, root, n) == 0 && loaded[n] == '/');
  std::string path = loaded;
  free(loaded);
  free(root);
  return path;
}

static double peak_rss_mb()
{
  rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss / 1024.0;
}

int main(int argc, char** argv)
{
  std::vector<std::string> positional;
  std::string dump_path;
  double time_limit = 1800;
  bool stats_only   = false;
  bool usage_error  = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--stats-only")
      stats_only = true;
    else if (arg == "--time-limit" && i + 1 < argc)
      time_limit = std::stod(argv[++i]);
    else if (arg == "--dump-csr" && i + 1 < argc)
      dump_path = argv[++i];
    else if (arg.rfind("--", 0) == 0)
      usage_error = true;
    else
      positional.push_back(arg);
  }
  if (usage_error || positional.size() != 2) {
    fprintf(stderr,
            "usage: baseline_tool <instance.mps> <out.json> [--time-limit <s>] "
            "[--dump-csr <file>] [--stats-only]\n");
    return 2;
  }
  const std::string& mps_path = positional[0];
  const std::string& out_path = positional[1];
  std::string log_path        = out_path + ".log";
  std::string tmp_path        = out_path + ".tmp";
  std::string highs_library   = loaded_highs_library();
  remove(log_path.c_str());

  Highs highs;
  set_option(highs, "output_flag", true);
  set_option(highs, "log_to_console", false);
  set_option(highs, "log_file", log_path);
  set_option(highs, "log_dev_level", 1);
  set_option(highs, "threads", 1);
  set_option(highs, "time_limit", time_limit);
  set_option(highs, "presolve", "on");

  auto start              = std::chrono::steady_clock::now();
  HighsStatus read_status = highs.readModel(mps_path);
  double read_seconds     = seconds_since(start);
  if (read_status == HighsStatus::kError) {
    fprintf(stderr, "baseline_tool: HiGHS cannot read %s\n", mps_path.c_str());
    return 1;
  }
  const HighsLp& lp = highs.getLp();

  json_writer_t js;
  js.file = fopen(tmp_path.c_str(), "w");
  assert(js.file);
  fputc('{', js.file);
  js.put_string("instance", instance_name(mps_path));
  js.put_string("mps_file", mps_path);

  js.open("highs");
  js.put_string("version", highsVersion());
  js.put_string("githash", highsGithash());
  js.put_int("highsint_bytes", sizeof(HighsInt));
  js.put_string("library", highs_library);
  js.close();

  js.put_real("read_seconds", read_seconds);
  js.put_string("read_status", highs_status_name(read_status));
  js.put_string("objective_sense", lp.sense_ == ObjSense::kMaximize ? "max" : "min");

  int64_t rows, cols, nnz;
  {
    start   = std::chrono::steady_clock::now();
    csr_t m = build_csr(lp);
    rows    = m.nrows;
    cols    = m.ncols;
    nnz     = m.nnz;
    write_structure(js, "original", m, lp, true);
    js.put_real("stats_seconds", seconds_since(start));
    if (!dump_path.empty()) {
      js.open("dump_csr");
      js.put_string("file", dump_path);
      js.put_int("bytes", dump_csr(dump_path, m, lp));
      js.put_bool("objective_negated", lp.sense_ == ObjSense::kMaximize);
      js.close();
    }
  }

  if (!stats_only) {
    presolve_marks_t marks;
    HighsStatus callback_status = highs.setCallback(on_log_message, &marks);
    assert(callback_status == HighsStatus::kOk);
    callback_status = highs.startCallback(kCallbackLogging);
    assert(callback_status == HighsStatus::kOk);

    start                      = std::chrono::steady_clock::now();
    HighsStatus run_status     = highs.presolve();
    time_point_t finish        = std::chrono::steady_clock::now();
    double presolve_seconds    = seconds_between(start, finish);
    HighsPresolveStatus status = highs.getModelPresolveStatus();
    bool has_model             = has_presolved_model(status);

    js.open("presolve");
    js.put_int("threads", 1);
    js.put_real("time_limit", time_limit);
    js.put_string("log_file", log_path);
    js.put_real("seconds", presolve_seconds);
    if (marks.started) js.put_real("setup_seconds", seconds_between(start, marks.start));
    if (marks.started && marks.ended) {
      js.put_real("core_seconds", seconds_between(marks.start, marks.end));
      js.put_real("handover_seconds", seconds_between(marks.end, finish));
    }
    js.put_bool("reached_time_limit", presolve_seconds >= time_limit);
    js.put_string("status", highs.presolveStatusToString(status));
    js.put_string("highs_return", highs_status_name(run_status));
    js.put_bool("has_model", has_model);
    if (has_model) {
      const HighsLp& presolved = highs.getPresolvedLp();
      csr_t m                  = build_csr(presolved);
      js.put_int("rows_removed", rows - m.nrows);
      js.put_int("cols_removed", cols - m.ncols);
      js.put_int("nnz_removed", nnz - m.nnz);
      js.close();
      write_structure(js, "presolved", m, presolved, false);
    } else {
      js.close();
      js.put_null("presolved");
    }
  }

  js.put_real("peak_rss_mb", peak_rss_mb());
  fputs("\n}\n", js.file);
  int closed = fclose(js.file);
  assert(closed == 0);
  int renamed = rename(tmp_path.c_str(), out_path.c_str());
  assert(renamed == 0);
  return 0;
}
