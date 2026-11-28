// The integral side and the drift of an integer free singleton column use the tolerance
// together: 1000 k + (1000 - 9e-10) y = 1000 + 9.9e-7 with k an integer singleton column and
// y an integer in [0, 100].
#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace gprs;

int main()
{
  model_t m;
  m.nrows     = 3;
  m.ncols     = 3;
  m.row_start = {0, 2, 4, 6};
  m.col_index = {0, 1, 1, 2, 1, 2};
  m.value     = {1000., 1000. - 9e-10, 1., 1., 1., -1.};
  m.row_lower = {1000. + 9.9e-7, -inf, -200.};
  m.row_upper = {1000. + 9.9e-7, 200., inf};
  m.col_lower = {-inf, 0., 0.};
  m.col_upper = {inf, 100., 1.};
  m.cost      = {1., 0., 0.};
  m.col_type  = {1, 1, 1};
  options_t o;
  o.enabled.fill(false);
  o.enabled[rule::singleton_cols] = true;
  result_t r                      = presolve(m, o);
  std::printf("status %s free_singleton %lld reduced cols %d\n",
              status_name(r.status),
              (long long)r.sub_findings[sub_rule::free_singleton],
              r.reduced.ncols);
  if (r.reduced.ncols != 2) { return 0; }
  for (double y : {0., 100.}) {
    std::vector<double> x;
    r.postsolve.postsolve({y, 0.}, x);
    point_check_t reduced  = check_point(r.reduced, {y, 0.});
    point_check_t original = check_point(m, x);
    std::printf("y %g: k %g reduced row violation %g original row violation %.3e (tolerance %g)\n",
                y,
                x[0],
                reduced.row_violation,
                original.row_violation,
                o.tol.feasibility);
  }
  return 0;
}
