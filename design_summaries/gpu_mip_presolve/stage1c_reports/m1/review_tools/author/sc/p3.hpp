// One reduced column in the test of a row: g its coefficient in the original row, composed, c
// its coefficient in the reduced row.
struct row_term_t {
  double g;
  double c;
  double parts;  // sum of the absolute values g was composed of
  double lower;
  double upper;
};

// The largest value of sign * g * r over the bounds and low <= c * r <= high, as the smallest
// Lagrangian bound over the multipliers 0 and sign * g[k] / c[k]; infinity when none is
// finite. `largest` gets the largest term of that bound.
inline double implied_maximum(const std::vector<row_term_t>& terms,
                              double sign,
                              double low,
                              double high,
                              std::vector<double>& multipliers,
                              double& largest)
{
  double best = gprs::inf;
  double last = std::nan("");
  for (double lambda : multipliers) {
    lambda *= sign;
    if (std::fabs(lambda - last) <= same_ratio * std::fmax(std::fabs(lambda), std::fabs(last))) {
      continue;
    }
    last        = lambda;
    double side = lambda > 0 ? high : low;
    if (lambda != 0. && !std::isfinite(side)) { continue; }
    sum_t bound;
    double size = 1.;
    if (lambda != 0.) {
      bound.add(lambda, side);
      size = std::fmax(size, std::fabs(lambda * side));
    }
    bool finite = true;
    for (const row_term_t& t : terms) {
      double g     = sign * t.g;
      double moved = lambda * t.c;
      double w     = g - moved;
      if (std::fabs(w) <= composed_zero * std::fmax(std::fmax(1., t.parts), std::fabs(moved))) {
        continue;
      }
      double at = w > 0 ? t.upper : t.lower;
      if (!std::isfinite(at)) {
        finite = false;
        break;
      }
      bound.add(w, at);
      size = std::fmax(size, std::fabs(w * at));
    }
    if (finite && bound.value() < best) {
      best    = bound.value();
      largest = size;
    }
  }
  return best;
}

