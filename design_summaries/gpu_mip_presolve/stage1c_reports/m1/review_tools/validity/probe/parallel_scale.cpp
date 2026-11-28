// A row removed as a multiple (2^18) of a kept row, and then a zero-cost singleton column of the
// kept row with a term of 9e7: the new side of the kept row is one double, and its rounding
// reaches the removed row times 2^18.
//   r0: s + y + z in [0.1, 5.3]          r1: 2^18 * r0
//   s in [0, 9e7] continuous without cost; y in [-1e8, 10], z in [0, 1]; two rows keep y and z.
#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace gprs;

int main()
{
  const double f = 262144.;
  model_t m;
  m.nrows     = 4;
  m.ncols     = 3;
  m.row_start = {0, 3, 6, 8, 10};
  m.col_index = {0, 1, 2, 0, 1, 2, 1, 2, 1, 2};
  m.value     = {1., 1., 1., f, f, f, 1., 1., 1., -1.};
  m.row_lower = {0.1, f * 0.1, -inf, -2e8};
  m.row_upper = {5.3, f * 5.3, 100., inf};
  m.col_lower = {0., -1e8, 0.};
  m.col_upper = {9e7, 10., 1.};
  m.cost      = {0., 1., 1.};
  m.col_type  = {0, 0, 0};
  options_t o;
  o.enabled.fill(false);
  o.enabled[rule::parallel_rows]  = true;
  o.enabled[rule::singleton_cols] = true;
  result_t r = presolve(m, o);
  std::printf("status %s parallel rows %lld zero_cost %lld reduced rows %d cols %d\n",
              status_name(r.status),
              (long long)r.stats[rule::parallel_rows].findings,
              (long long)r.sub_findings[sub_rule::zero_cost_singleton],
              r.reduced.nrows,
              r.reduced.ncols);
  const model_t& q = r.reduced;
  for (int32_t i = 0; i < q.nrows; ++i) {
    std::printf("  reduced row %d (original %d): [%.17g, %.17g]\n",
                i,
                r.postsolve.row_origin[i],
                q.row_lower[i],
                q.row_upper[i]);
  }
  if (q.ncols != 2 || q.nrows < 1) { return 0; }
  // The reduced point with y + z at the lower side of the widened row.
  int32_t row = -1;
  for (int32_t i = 0; i < q.nrows; ++i) {
    if (r.postsolve.row_origin[i] == 0) { row = i; }
  }
  if (row < 0) { return 0; }
  std::vector<double> reduced = {q.row_lower[row], 0.}, x;
  point_check_t at_reduced    = check_point(q, reduced);
  r.postsolve.postsolve(reduced, x);
  point_check_t at_original = check_point(m, x);
  std::printf("reduced point {%.17g, 0}: violations rows %g bounds %g\n",
              reduced[0],
              at_reduced.row_violation,
              at_reduced.bound_violation);
  std::printf("postsolved {%.17g, %.17g, %.17g}: violations rows %.3e bounds %g\n",
              x[0],
              x[1],
              x[2],
              at_original.row_violation,
              at_original.bound_violation);
  return 0;
}
