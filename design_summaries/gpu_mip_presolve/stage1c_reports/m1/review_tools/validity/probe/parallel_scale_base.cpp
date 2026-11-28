// The Stage 1 form of probe/parallel_scale.cpp, without a column rule: a row removed as a
// multiple (2^18) of a kept row, and then a doubleton substitution y = v + 9e7 that shifts the
// sides of the kept row by 9e7. The shifted side is one double, and its rounding reaches the
// removed row times 2^18.
//   r0: y + z in [0.1, 5.3]      r1: 2^18 * r0      r2: y - v = 9e7
//   y in [-1e9, 1e9], v in [-2e8, 10], z in [0, 1]; rows 3 to 5 keep v and z, and give v more
//   entries than y, so that v is the column that stays.
#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace gprs;

int main()
{
  const double f = 262144.;
  model_t m;
  m.nrows     = 6;
  m.ncols     = 3;
  // columns: 0 y, 1 v, 2 z
  m.row_start = {0, 2, 4, 6, 8, 10, 12};
  m.col_index = {0, 2, 0, 2, 0, 1, 1, 2, 1, 2, 1, 2};
  m.value     = {1., 1., f, f, 1., -1., 1., 1., 1., -1., 2., 1.};
  m.row_lower = {0.1, f * 0.1, 9e7, -inf, -4e8, -inf};
  m.row_upper = {5.3, f * 5.3, 9e7, 100., inf, 300.};
  m.col_lower = {-1e9, -2e8, 0.};
  m.col_upper = {1e9, 10., 1.};
  m.cost      = {0., 1., 1.};
  m.col_type  = {0, 0, 0};
  options_t o;
  o.enabled.fill(false);
  o.enabled[rule::parallel_rows] = true;
  o.enabled[rule::doubleton]     = true;
  result_t r                     = presolve(m, o);
  std::printf("status %s parallel rows %lld doubleton %lld reduced rows %d cols %d aliases %zu\n",
              status_name(r.status),
              (long long)r.stats[rule::parallel_rows].findings,
              (long long)r.stats[rule::doubleton].findings,
              r.reduced.nrows,
              r.reduced.ncols,
              r.postsolve.aliases.size());
  const model_t& q = r.reduced;
  int32_t row      = -1;
  for (int32_t i = 0; i < q.nrows; ++i) {
    std::printf("  reduced row %d (original %d): [%.17g, %.17g]\n",
                i,
                r.postsolve.row_origin[i],
                q.row_lower[i],
                q.row_upper[i]);
    if (r.postsolve.row_origin[i] == 0) { row = i; }
  }
  if (q.ncols != 2 || row < 0) { return 0; }
  std::printf("  reduced columns are original %d and %d\n",
              r.postsolve.col_origin[0],
              r.postsolve.col_origin[1]);
  // The reduced point with the kept column at the lower side of row 0 and z = 0.
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
