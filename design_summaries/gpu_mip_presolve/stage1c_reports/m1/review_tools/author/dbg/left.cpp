// left.cpp <dump> [--list N] [gprs options: --growth <limit>]: the singleton columns that the
// presolve leaves in the reduced model, by what stops the free singleton column on each:
// counted per class, and the first N listed.
#include <gprs/io.hpp>
#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace gprs;

int main(int argc, char** argv)
{
  if (argc < 2) { return 2; }
  int list = 0;
  options_t opt;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--list") && i + 1 < argc) { list = std::atoi(argv[++i]); }
    if (!std::strcmp(argv[i], "--growth") && i + 1 < argc) {
      opt.pivot_growth_limit = std::atof(argv[++i]);
    }
  }
  model_t model;
  std::string error;
  if (!read_csr_dump(argv[1], model, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  result_t r       = presolve(model, opt);
  const model_t& m = r.reduced;
  std::vector<int32_t> count(m.ncols, 0), row_of(m.ncols, -1);
  std::vector<double> coef(m.ncols, 0.);
  for (int32_t i = 0; i < m.nrows; ++i) {
    for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
      ++count[m.col_index[e]];
      row_of[m.col_index[e]] = i;
      coef[m.col_index[e]]   = m.value[e];
    }
  }
  std::map<std::string, int> classes;
  int listed = 0, singles = 0;
  for (int32_t j = 0; j < m.ncols; ++j) {
    if (count[j] != 1) { continue; }
    ++singles;
    int32_t i   = row_of[j];
    double a    = coef[j];
    double lo   = m.row_lower[i], up = m.row_upper[i];
    bool eq     = lo == up;
    double cost = m.cost[j];
    int32_t entries = m.row_start[i + 1] - m.row_start[i];
    double rest_min = 0., rest_max = 0., largest = 0.;
    bool others_integer = true;
    for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
      int32_t k = m.col_index[e];
      if (k == j) { continue; }
      double v = m.value[e];
      rest_min += v > 0 ? v * m.col_lower[k] : v * m.col_upper[k];
      rest_max += v > 0 ? v * m.col_upper[k] : v * m.col_lower[k];
      largest = std::fmax(largest, std::fabs(v));
      others_integer = others_integer && m.col_type[k] != 0;
    }
    std::string why;
    double side = up;
    if (!eq) {
      if (cost == 0.) {
        why = "inequality, no cost";
      } else {
        bool rises = (cost < 0) == (a > 0);
        side       = rises ? up : lo;
        if (std::isinf(side)) { why = "inequality, the cost pushes to an infinite side"; }
      }
    }
    if (why.empty() && entries < 2) { why = "alone in its row"; }
    if (why.empty()) {
      double v1 = (side - rest_min) / a, v2 = (side - rest_max) / a;
      double low = std::fmin(v1, v2), high = std::fmax(v1, v2);
      double tol = 1e-9;
      bool implied = (std::isinf(m.col_upper[j]) || high <= m.col_upper[j] + tol * std::fmax(1., std::fabs(m.col_upper[j]))) &&
                     (std::isinf(m.col_lower[j]) || low >= m.col_lower[j] - tol * std::fmax(1., std::fabs(m.col_lower[j])));
      bool bounded = !std::isinf(m.col_lower[j]) || !std::isinf(m.col_upper[j]);
      if (!implied) {
        why = std::string(eq ? "equation" : "inequality") + ", the row does not imply the bounds";
      } else if (m.col_type[j] != 0 && !others_integer) {
        why = "integer column next to a continuous one";
      } else if (m.col_type[j] != 0) {
        why = "integer column, all integer: ratios, side or drift";
      } else if (bounded && largest / std::fabs(a) > opt.pivot_growth_limit) {
        why = "growth limit by the row";
      } else {
        why = "passes these tests (growth history, readable sides or rounding)";
      }
    }
    why += m.col_type[j] != 0 ? " | integer" : " | continuous";
    ++classes[why];
    if (listed < list) {
      ++listed;
      std::printf("col %d (orig %d) %s [%g, %g] cost %g a %g | row %d [%g, %g] entries %d rest [%g, %g] largest %g | %s\n",
                  j, r.postsolve.col_origin[j], m.col_type[j] ? "int" : "cont", m.col_lower[j],
                  m.col_upper[j], cost, a, i, lo, up, entries, rest_min, rest_max, largest,
                  why.c_str());
    }
  }
  std::printf("%s: reduced %d rows %d cols %lld nnz, singleton columns left %d\n", argv[1], m.nrows,
              m.ncols, (long long)m.nnz(), singles);
  for (const auto& c : classes) {
    std::printf("  %6d %s\n", c.second, c.first.c_str());
  }
  return 0;
}
