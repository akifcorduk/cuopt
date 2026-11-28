// exact_driver < models.txt > results.txt
// The engine on the tiny models of the exact enumeration (exact/engine_check.py). Input, per
// model:
//   M <id> <ncols> <nrows>
//   C <lower> <upper> <integer 0|1> <cost>          one per column
//   R <lower> <upper> <n> (<col> <value>) * n       one per row, columns increasing
//   Q <option set> <values of the reduced columns>  a reduced point to postsolve (second pass)
//   P <values of the columns>                       a point of the model to crush (second pass)
//   E
// Output, per model and option set:
//   RES <id> <option set> <status> fp <0|1> dual <n> fs <n> fsd <n> zs <n> fc <n>
//   RM <ncols> <nrows> <offset>, RC <original column> <lower> <upper> <integer> <cost>,
//   RR <lower> <upper> <n> (<reduced col> <value>) * n          the reduced model
//   PS <option set> <values of the original columns>            for every Q of that set
//   CR <option set> <values of the reduced columns>             for every P
//   END <id>

#include "structure_check.hpp"

#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace gprs;

namespace {

struct option_set_t {
  const char* name;
  options_t options;
};

std::vector<option_set_t> option_sets()
{
  std::vector<option_set_t> sets;
  options_t o;
  sets.push_back({"default", o});
  o.primal_only = true;
  sets.push_back({"primal", o});
  o = options_t();
  o.enabled.fill(false);
  o.enabled[rule::singleton_cols] = true;
  sets.push_back({"sc", o});
  const int32_t subs[4]    = {sub_rule::free_singleton,
                              sub_rule::free_singleton_dual,
                              sub_rule::zero_cost_singleton,
                              sub_rule::forcing_column};
  const char* const names[4] = {"sc_r2", "sc_r5", "sc_r3", "sc_r4"};
  for (int32_t k = 0; k < 4; ++k) {
    options_t one = o;
    for (int32_t s : subs) {
      one.sub_enabled[s] = s == subs[k];
    }
    sets.push_back({names[k], one});
  }
  return sets;
}

bool carries(status_t s)
{
  return s == status_t::reduced || s == status_t::unchanged || s == status_t::reduced_to_empty ||
         s == status_t::limit;
}

void put(double v) { std::printf(" %.17g", v); }

}  // namespace

int main()
{
  std::vector<option_set_t> sets = option_sets();
  std::string line;
  model_t m;
  long long id = 0;
  std::vector<std::pair<std::string, std::vector<double>>> reduced_points;
  std::vector<std::vector<double>> points;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string tag;
    in >> tag;
    if (tag == "M") {
      m = model_t();
      reduced_points.clear();
      points.clear();
      in >> id >> m.ncols >> m.nrows;
      m.row_start.push_back(0);
    } else if (tag == "C") {
      std::string lower, upper;
      int integer;
      double cost;
      in >> lower >> upper >> integer >> cost;
      m.col_lower.push_back(std::strtod(lower.c_str(), nullptr));
      m.col_upper.push_back(std::strtod(upper.c_str(), nullptr));
      m.col_type.push_back(integer);
      m.cost.push_back(cost);
    } else if (tag == "R") {
      std::string lower, upper;
      int n;
      in >> lower >> upper >> n;
      m.row_lower.push_back(std::strtod(lower.c_str(), nullptr));
      m.row_upper.push_back(std::strtod(upper.c_str(), nullptr));
      for (int k = 0; k < n; ++k) {
        int col;
        double value;
        in >> col >> value;
        m.col_index.push_back(col);
        m.value.push_back(value);
      }
      m.row_start.push_back(m.col_index.size());
    } else if (tag == "Q") {
      std::string set;
      in >> set;
      std::vector<double> x;
      double v;
      while (in >> v) {
        x.push_back(v);
      }
      reduced_points.push_back({set, x});
    } else if (tag == "P") {
      std::vector<double> x;
      double v;
      while (in >> v) {
        x.push_back(v);
      }
      points.push_back(x);
    } else if (tag == "E") {
      if (!well_formed(m)) {
        std::printf("BAD %lld\n", id);
        continue;
      }
      for (const option_set_t& set : sets) {
        result_t r = presolve(m, set.options);
        std::printf("RES %lld %s %s fp %d dual %lld fs %lld fsd %lld zs %lld fc %lld\n",
                    id,
                    set.name,
                    status_name(r.status),
                    (int)r.fixed_point,
                    (long long)r.dual_findings,
                    (long long)r.sub_findings[sub_rule::free_singleton],
                    (long long)r.sub_findings[sub_rule::free_singleton_dual],
                    (long long)r.sub_findings[sub_rule::zero_cost_singleton],
                    (long long)r.sub_findings[sub_rule::forcing_column]);
        if (!carries(r.status)) { continue; }
        const model_t& q = r.reduced;
        gprs_check::structure_report_t s =
          gprs_check::structure_check(m, r, set.options.primal_only);
        bool silent = s.composed && s.lost_integers == 0 && s.bound_failures == 0 &&
                      s.row_failures == 0 && s.kept_failures == 0 && s.added_failures == 0 &&
                      s.box_failures == 0 && s.integral_failures == 0 &&
                      s.snapshot_failures == 0 && s.objective_failures == 0 &&
                      s.postsolve_failures == 0;
        std::printf("SC %s %d stack '%s' bounds %lld rows %lld kept %lld added %lld box %lld "
                    "integral %lld snapshot %lld objective %lld postsolve %lld\n",
                    set.name,
                    (int)silent,
                    s.stack_error.c_str(),
                    (long long)s.bound_failures,
                    (long long)s.row_failures,
                    (long long)s.kept_failures,
                    (long long)s.added_failures,
                    (long long)s.box_failures,
                    (long long)s.integral_failures,
                    (long long)s.snapshot_failures,
                    (long long)s.objective_failures,
                    (long long)s.postsolve_failures);
        std::printf("RM %d %d %.17g\n", q.ncols, q.nrows, q.objective_offset);
        for (int32_t j = 0; j < q.ncols; ++j) {
          std::printf("RC %d", r.postsolve.col_origin[j]);
          put(q.col_lower[j]);
          put(q.col_upper[j]);
          std::printf(" %d", (int)q.col_type[j]);
          put(q.cost[j]);
          std::printf("\n");
        }
        for (int32_t i = 0; i < q.nrows; ++i) {
          std::printf("RR");
          put(q.row_lower[i]);
          put(q.row_upper[i]);
          std::printf(" %lld", (long long)(q.row_start[i + 1] - q.row_start[i]));
          for (int64_t e = q.row_start[i]; e < q.row_start[i + 1]; ++e) {
            std::printf(" %d", q.col_index[e]);
            put(q.value[e]);
          }
          std::printf("\n");
        }
        for (const auto& p : reduced_points) {
          if (p.first != set.name || p.second.size() != (size_t)q.ncols) { continue; }
          std::vector<double> x;
          r.postsolve.postsolve(p.second, x);
          std::printf("PS %s", set.name);
          for (double v : x) {
            put(v);
          }
          std::printf("\n");
        }
        for (const std::vector<double>& p : points) {
          std::vector<double> x;
          r.postsolve.crush(p, x);
          std::printf("CR %s", set.name);
          for (double v : x) {
            put(v);
          }
          std::printf("\n");
        }
      }
      std::printf("END %lld\n", id);
    }
  }
  return 0;
}
