// The test of a row against several reduced rows together: its own, if it has one, as row 0,
// and those of its equations. A point v holds one multiplier per row.
struct combined_t {
  struct column_t {
    double g;      // coefficient in the original row, composed
    double parts;  // sum of the absolute values g was composed of
    double lower;
    double upper;
  };
  bool own     = false;
  int32_t rows = 0;
  std::vector<double> low;
  std::vector<double> high;
  std::vector<column_t> columns;
  // Coefficient of column k in row i at k * rows + i.
  std::vector<double> a;
};

inline void sort_different(std::vector<double>& values)
{
  std::sort(values.begin(), values.end());
  size_t kept = 0;
  for (double a : values) {
    if (kept > 0 && std::fabs(a - values[kept - 1]) <=
                      same_ratio * std::fmax(std::fabs(a), std::fabs(values[kept - 1]))) {
      continue;
    }
    values[kept++] = a;
  }
  values.resize(kept);
}

// The Lagrangian bound of the largest value of sign * g * r over the bounds and the rows of p,
// for the multipliers m = sign * v:
//   sum of m[i] * (high[i] or low[i]) + max over the bounds of (sign * g - sum of m[i] * a[i]) * r
// with the side of the sign of each multiplier; infinity when a side or a bound it needs is.
// A coefficient of that maximum counts as zero when it is a cancellation of the row itself
// (composed_zero of its parts) or the rounding of the terms of the reduced rows (rounding_zero
// of their sum). No larger share of those terms: multipliers that cancel each other make the
// sum as large as they are, and a coefficient of the size of the row's own would vanish in it.
// `largest` gets the largest term of the bound, the term of an equation counted as at least its
// multiplier, and `multiplier` the largest multiplier of an equation.
// `size` gets what a violation of the row is relative to: the bound itself, the row's own
// coefficients times the bounds that the bound uses, and the multiplier of an equation. The
// sides of the reduced rows and the columns that only they hold count with rounding_share:
// their terms cancel in the bound (the side of the row after a subtraction holds the
// right-hand side of the equation, the side of the equation gives it back), so a huge one says
// what its rounding can do to the row and no more.
inline double combined_bound(const combined_t& p,
                             double sign,
                             const double* v,
                             double& size,
                             double& largest,
                             double& multiplier)
{
  sum_t bound;
  size       = 1.;
  largest    = 1.;
  multiplier = 0.;
  for (int32_t i = 0; i < p.rows; ++i) {
    double m = sign * v[i];
    if (m == 0.) { continue; }
    double side = m > 0 ? p.high[i] : p.low[i];
    if (!std::isfinite(side)) { return gprs::inf; }
    bound.add(m, side);
    largest = std::fmax(largest, std::fabs(m * side));
    if (!(p.own && i == 0)) {
      size       = std::fmax(size, std::fabs(m));
      largest    = std::fmax(largest, std::fabs(m));
      multiplier = std::fmax(multiplier, std::fabs(m));
    }
  }
  for (size_t k = 0; k < p.columns.size(); ++k) {
    const combined_t::column_t& c = p.columns[k];
    double w = sign * c.g, moved = 0.;
    for (int32_t i = 0; i < p.rows; ++i) {
      double term = sign * v[i] * p.a[k * p.rows + i];
      w -= term;
      moved += std::fabs(term);
    }
    if (std::fabs(w) <= std::fmax(composed_zero * std::fmax(1., c.parts), rounding_zero * moved)) {
      continue;
    }
    double at = w > 0 ? c.upper : c.lower;
    if (!std::isfinite(at)) { return gprs::inf; }
    bound.add(w, at);
    largest = std::fmax(largest, std::fabs(w * at));
    size    = std::fmax(size, std::fabs(c.g * at));
  }
  size = std::fmax(size, std::fmax(std::fabs(bound.value()), rounding_share * largest));
  return bound.value();
}

// The bound is linear in v between the hyperplanes where one of its terms changes its sign:
// v[i] = 0 for a row, and a[k] * v = g[k] for a column. Where it has a smallest value, it has
// it at a point where as many of them meet as there are rows. A column of one row gives a
// hyperplane v[i] = g[k] / a[k][i]: axis[i] holds the different values of that kind, with 0;
// `general` holds the columns of two rows or more.
struct hyperplanes_t {
  std::vector<std::vector<double>> axis;
  std::vector<int32_t> general;
  // Right-hand side of the hyperplane of each column of `general`.
  std::vector<double> rhs;

  explicit hyperplanes_t(const combined_t& p) : axis(p.rows)
  {
    for (size_t k = 0; k < p.columns.size(); ++k) {
      int32_t held = 0, row = 0;
      for (int32_t i = 0; i < p.rows; ++i) {
        if (p.a[k * p.rows + i] != 0.) {
          ++held;
          row = i;
        }
      }
      if (held == 1) { axis[row].push_back(p.columns[k].g / p.a[k * p.rows + row]); }
      if (held > 1) {
        general.push_back(k);
        rhs.push_back(p.columns[k].g);
      }
    }
    for (std::vector<double>& values : axis) {
      values.push_back(0.);
      sort_different(values);
    }
  }

  // The number of points that visit_vertices tries: every choice of the rows that take a value
  // of their axis, a value for each, and as many columns of `general` as rows are left.
  double count() const
  {
    // chosen[j]: the choices of j rows with a value each.
    std::vector<double> chosen(axis.size() + 1, 0.);
    chosen[0] = 1.;
    for (size_t i = 0; i < axis.size(); ++i) {
      for (size_t j = i + 1; j-- > 0;) {
        chosen[j + 1] += chosen[j] * (double)axis[i].size();
      }
    }
    double total = 0., subsets = 1.;
    for (size_t m = 0; m <= axis.size() && m <= general.size(); ++m) {
      total += subsets * chosen[axis.size() - m];
      subsets *= (double)(general.size() - m) / (double)(m + 1);
    }
    return total;
  }
};

// Calls visit(v) for every point v where p.rows of the hyperplanes meet in one point (no pivot
// of their system below singular_pivot); the rows of `fixed` (a bit per row) always take a
// value of their axis.
template <typename visit_t>
void visit_vertices(const combined_t& p, const hyperplanes_t& h, uint32_t fixed, visit_t visit)
{
  int32_t d = p.rows, planes = h.general.size();
  std::vector<double> v(d), matrix, scale, scaled;
  std::vector<int32_t> free_rows, fixed_rows, value, chosen;
  uint32_t masks = planes == 0 ? 1 : (uint32_t)1 << d;
  for (uint32_t free_mask = 0; free_mask < masks; ++free_mask) {
    if (free_mask & fixed) { continue; }
    free_rows.clear();
    fixed_rows.clear();
    for (int32_t i = 0; i < d; ++i) {
      ((free_mask >> i) & 1 ? free_rows : fixed_rows).push_back(i);
    }
    int32_t m = free_rows.size();
    if (m > planes) { continue; }
    value.assign(fixed_rows.size(), 0);
    for (bool more_values = true; more_values;) {
      for (size_t n = 0; n < fixed_rows.size(); ++n) {
        v[fixed_rows[n]] = h.axis[fixed_rows[n]][value[n]];
      }
      chosen.resize(m);
      for (int32_t n = 0; n < m; ++n) {
        chosen[n] = n;
      }
      for (bool more_planes = true; more_planes;) {
        // The free rows from the chosen hyperplanes: the system with every column and then
        // every row scaled to a largest entry of 1, elimination with row pivoting, back
        // substitution. A system with a pivot below singular_pivot gives no point: hyperplanes
        // that are parallel on the free rows leave a pivot that is the rounding of zero, and
        // its multipliers, as large as that rounding is small, would cancel each other in every
        // coefficient that the hyperplanes share and stand for a bound that is none.
        matrix.assign(m * (m + 1), 0.);
        for (int32_t r = 0; r < m; ++r) {
          int32_t k        = h.general[chosen[r]];
          double remainder = h.rhs[chosen[r]];
          for (int32_t i : fixed_rows) {
            remainder -= p.a[k * d + i] * v[i];
          }
          for (int32_t n = 0; n < m; ++n) {
            matrix[r * (m + 1) + n] = p.a[k * d + free_rows[n]];
          }
          matrix[r * (m + 1) + m] = remainder;
        }
        bool regular = true;
        scale.assign(m, 0.);
        scaled.assign(m, 0.);
        for (int32_t n = 0; n < m && regular; ++n) {
          for (int32_t r = 0; r < m; ++r) {
            scale[n] = std::fmax(scale[n], std::fabs(matrix[r * (m + 1) + n]));
          }
          regular = scale[n] > 0.;
        }
        for (int32_t r = 0; r < m && regular; ++r) {
          double most = 0.;
          for (int32_t n = 0; n < m; ++n) {
            matrix[r * (m + 1) + n] /= scale[n];
            most = std::fmax(most, std::fabs(matrix[r * (m + 1) + n]));
          }
          regular = most > 0.;
          if (!regular) { break; }
          for (int32_t c = 0; c <= m; ++c) {
            matrix[r * (m + 1) + c] /= most;
          }
        }
        for (int32_t n = 0; n < m && regular; ++n) {
          int32_t pivot = n;
          for (int32_t r = n + 1; r < m; ++r) {
            if (std::fabs(matrix[r * (m + 1) + n]) > std::fabs(matrix[pivot * (m + 1) + n])) {
              pivot = r;
            }
          }
          regular = std::fabs(matrix[pivot * (m + 1) + n]) >= singular_pivot;
          if (!regular) { break; }
          for (int32_t c = 0; c <= m; ++c) {
            std::swap(matrix[n * (m + 1) + c], matrix[pivot * (m + 1) + c]);
          }
          for (int32_t r = n + 1; r < m; ++r) {
            double factor = matrix[r * (m + 1) + n] / matrix[n * (m + 1) + n];
            for (int32_t c = n; c <= m; ++c) {
              matrix[r * (m + 1) + c] -= factor * matrix[n * (m + 1) + c];
            }
          }
        }
        for (int32_t n = m; regular && n-- > 0;) {
          double rest = matrix[n * (m + 1) + m];
          for (int32_t c = n + 1; c < m; ++c) {
            rest -= matrix[n * (m + 1) + c] * scaled[c];
          }
          scaled[n]       = rest / matrix[n * (m + 1) + n];
          v[free_rows[n]] = scaled[n] / scale[n];
          regular         = std::isfinite(v[free_rows[n]]);
        }
        if (regular) { visit(v.data()); }
        // The next choice of m of the columns.
        int32_t n = m - 1;
        while (n >= 0 && chosen[n] == planes - m + n) {
          --n;
        }
        more_planes = n >= 0;
        if (more_planes) {
          ++chosen[n];
          for (int32_t r = n + 1; r < m; ++r) {
            chosen[r] = chosen[r - 1] + 1;
          }
        }
      }
      // The next choice of a value for each fixed row.
      size_t n = 0;
      while (n < fixed_rows.size() && value[n] + 1 == (int32_t)h.axis[fixed_rows[n]].size()) {
        value[n++] = 0;
      }
      more_values = n < fixed_rows.size();
      if (more_values) { ++value[n]; }
    }
  }
}

// The smallest bound that a search has visited, with what a violation is relative to and its
// largest multiplier of an equation (combined_bound); infinity before the first finite one.
struct smallest_bound_t {
  double value      = gprs::inf;
  double size       = 1.;
  double multiplier = 0.;

  void visit(const combined_t& p, double sign, const double* v)
  {
    double bound_size, largest, most;
    double bound = combined_bound(p, sign, v, bound_size, largest, most);
    if (bound < value) {
      value      = bound;
      size       = bound_size;
      multiplier = most;
    }
  }
};

// No point within the bounds meets the rows of p together: the bound of the function zero is
// then negative for some multipliers, and a search for a row has no smallest bound. That bound
// is linear on every cone between the hyperplanes through the origin, a[k] * v = 0 and
// v[i] = 0, so it is negative somewhere when it is on a line where all but one of them meet:
// at a point with one multiplier 1 or -1 where as many of the others meet as there are other
// rows.
inline bool no_common_point(combined_t p)
{
  for (combined_t::column_t& c : p.columns) {
    c.g     = 0.;
    c.parts = 0.;
  }
  hyperplanes_t h(p);
  bool negative = false;
  for (int32_t unit = 0; unit < p.rows && !negative; ++unit) {
    h.axis[unit] = {-1., 1.};
    visit_vertices(p, h, (uint32_t)1 << unit, [&](const double* v) {
      double size, largest, most;
      negative =
        negative || combined_bound(p, 1., v, size, largest, most) < -check_tolerance * largest;
    });
    h.axis[unit] = {0.};
  }
  return negative;
}

