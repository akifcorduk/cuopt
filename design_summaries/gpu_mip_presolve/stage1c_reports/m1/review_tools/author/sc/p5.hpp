// The anchors that a function of a row at a position of the stack is proved from: the anchor of
// the row after that position, and the anchors of the equations that the records between the
// position and that anchor subtracted from it, each taken after its record, with the equations
// that were subtracted from those in turn. A row removed as a parallel row is, from then on,
// made of what its kept row is made of.
struct proof_rows_t {
  int32_t own = -1;
  // The own anchor belongs to another row: the kept row of a parallel record.
  bool other_row = false;
  // A record subtracted an equation other than the row itself.
  bool made_of_equations = false;
  // More than max_row_equations equations, or a walk above max_proof_visits: not tested.
  bool too_many = false;
  // Anchors of the equations, each once and none the own one.
  std::vector<int32_t> equations;

  struct visit_t {
    int32_t row;
    int64_t position;
  };
  std::vector<int32_t> rows;
  std::vector<visit_t> pending;
};

// Follows a row from a position of the stack: a saved row ends the walk with its anchor, a
// parallel record continues it with the kept row, and the equations subtracted on the way go
// to p.pending. Without a saved row the walk ends with the row of the reduced model, or -1.
// `last` gets the row that the anchor belongs to.
inline int32_t walk_row(const system_t& sys,
                        const gprs::postsolve_t& ps,
                        int32_t row,
                        int64_t position,
                        proof_rows_t& p,
                        int32_t& last)
{
  for (size_t hops = 0; hops <= ps.parallel_rows.size(); ++hops) {
    int32_t e = sys.first[row], end = sys.first[row + 1];
    while (e < end && (int64_t)sys.events[e].position <= position) {
      ++e;
    }
    bool moved = false;
    for (; e < end && !moved; ++e) {
      const row_event_t& event = sys.events[e];
      if (event.kind == row_event_t::saved) {
        last = row;
        return event.index;
      }
      if (event.kind == row_event_t::addition) {
        p.pending.push_back(proof_rows_t::visit_t{event.index, (int64_t)event.position});
        continue;
      }
      row      = event.index;
      position = event.position;
      moved    = true;
    }
    if (!moved) { break; }
  }
  last = row;
  return sys.reduced_anchor[row];
}

// Steps of the walks of one obligation; above it the obligation is not tested.
constexpr size_t max_proof_visits = 1024;

inline void proof_rows(const system_t& sys,
                       const gprs::postsolve_t& ps,
                       int32_t row,
                       int64_t position,
                       proof_rows_t& p)
{
  p.equations.clear();
  p.rows.clear();
  p.pending.clear();
  p.made_of_equations = false;
  int32_t last        = row;
  p.own               = walk_row(sys, ps, row, position, p, last);
  p.other_row         = last != row;
  for (size_t n = 0; n < p.pending.size(); ++n) {
    if (p.rows.size() > max_row_equations || p.pending.size() > max_proof_visits) { break; }
    proof_rows_t::visit_t visit = p.pending[n];
    bool again                  = false;
    for (size_t m = 0; m < n && !again; ++m) {
      again = p.pending[m].row == visit.row && p.pending[m].position == visit.position;
    }
    if (again) { continue; }
    if (visit.row != row) {
      p.made_of_equations = true;
      if (std::find(p.rows.begin(), p.rows.end(), visit.row) == p.rows.end()) {
        p.rows.push_back(visit.row);
      }
    }
    int32_t anchor = walk_row(sys, ps, visit.row, visit.position, p, last);
    if (anchor >= 0 && anchor != p.own &&
        std::find(p.equations.begin(), p.equations.end(), anchor) == p.equations.end()) {
      p.equations.push_back(anchor);
    }
  }
  p.too_many = p.rows.size() > max_row_equations || p.pending.size() > max_proof_visits;
}

// How an obligation was tested and what it gave.
struct proof_t {
  // 0: over the boxes; 1: against its anchor; 2: against anchors of equations as well.
  int32_t path = 0;
  // False for what the three paths count as not tested (kept_untested, added_untested).
  bool tested = true;
  // The violation, relative as the path defines it; 0 without one.
  double scaled = 0.;
  // Path 2: the largest multiplier of an equation.
  double multiplier = 0.;
};

// The three tests of a linear function of the check variables with sides. The caller fills the
// function into g and parts, with touched naming its variables, and prove() empties them.
struct prover_t {
  const system_t& sys;
  std::vector<double> g;
  std::vector<double> parts;
  std::vector<double> c;
  std::vector<int32_t> touched;
  std::vector<row_term_t> terms;
  std::vector<double> multipliers;
  // For a function that is made of equations: the variables of the test with the place of each
  // in it, and the entries of the anchors of the test, up to entry_end for each of them.
  std::vector<int32_t> cols;
  std::vector<int32_t> slot;
  std::vector<int32_t> entry_col;
  std::vector<double> entry_value;
  std::vector<size_t> entry_end;
  combined_t p;
  double kept_work = 0.;

  explicit prover_t(const system_t& system)
    : sys(system),
      g(system.nvars(), 0.),
      parts(system.nvars(), 0.),
      c(system.nvars(), 0.),
      slot(system.nvars(), -1),
      entry_end(max_row_equations + 1)
  {
  }

  void add_variable(int32_t v, double coefficient)
  {
    if (parts[v] == 0.) { touched.push_back(v); }
    g[v] += coefficient;
    parts[v] += std::fabs(coefficient);
  }

  // The term a * (original column) of the function.
  void add_column(const column_map_t& m, double a, sum_t& constant, double& largest)
  {
    constant.add(a, m.offset);
    raise_to(largest, std::fabs(a * m.offset));
    if (m.root >= 0) { add_variable(m.root, a * m.scale); }
  }

  void clear()
  {
    for (int32_t v : touched) {
      g[v]     = 0.;
      parts[v] = 0.;
    }
    touched.clear();
  }

  // lower <= constant + g * r <= upper at every point within the boxes that meets the anchors
  // of `with`; `largest` is the largest term of the constant, at least 1.
  proof_t prove(
    double lower, double upper, const sum_t& constant, double largest, const proof_rows_t& with)
  {
    proof_t out;
    int32_t k    = with.own;
    double below = -gprs::inf, above = -gprs::inf;
    if (with.too_many) {
      clear();
      out.path   = 2;
      out.tested = false;
      return out;
    }
    if (!with.equations.empty()) {
      out.path = 2;
      p.own    = k >= 0;
      p.rows   = p.own + with.equations.size();
      p.low.clear();
      p.high.clear();
      p.columns.clear();
      cols.clear();
      entry_col.clear();
      entry_value.clear();
      for (int32_t v : touched) {
        slot[v] = p.columns.size();
        cols.push_back(v);
        p.columns.push_back(combined_t::column_t{g[v], parts[v], sys.lower[v], sys.upper[v]});
        g[v]     = 0.;
        parts[v] = 0.;
      }
      touched.clear();
      // The entries of the anchors, one after the other; a variable that only they have gets
      // g = 0.
      for (int32_t n = 0; n < p.rows; ++n) {
        int32_t anchor = p.own ? (n == 0 ? k : with.equations[n - 1]) : with.equations[n];
        p.low.push_back(sys.low[anchor]);
        p.high.push_back(sys.high[anchor]);
        for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
          int32_t v = sys.var[e];
          if (slot[v] < 0) {
            slot[v] = p.columns.size();
            cols.push_back(v);
            p.columns.push_back(combined_t::column_t{0., 0., sys.lower[v], sys.upper[v]});
          }
          entry_col.push_back(slot[v]);
          entry_value.push_back(sys.value[e]);
        }
        entry_end[n] = entry_col.size();
      }
      for (int32_t v : cols) {
        slot[v] = -1;
      }
      p.a.assign(p.columns.size() * p.rows, 0.);
      for (size_t e = 0, n = 0; e < entry_col.size(); ++e) {
        while (e == entry_end[n]) {
          ++n;
        }
        p.a[entry_col[e] * p.rows + n] = entry_value[e];
      }
      hyperplanes_t h(p);
      double work = h.count() * (double)p.a.size();
      if (work > max_kept_row_work || kept_work + work > max_kept_work) {
        out.tested = false;
        return out;
      }
      kept_work += work;
      smallest_bound_t most, least;
      visit_vertices(p, h, 0, [&](const double* v) {
        if (upper < gprs::inf) { most.visit(p, 1., v); }
        if (lower > -gprs::inf) { least.visit(p, -1., v); }
      });
      if (upper < gprs::inf) { above = constant.value() + most.value - upper; }
      if (lower > -gprs::inf) { below = lower - constant.value() + least.value; }
      double violation = !(below <= above) ? below : above;
      double scaled    = violation / std::fmax(largest, below > above ? least.size : most.size);
      // A bound that proves the function does so whether a point meets the anchors or not; a
      // search that finds none says nothing when no point meets them.
      if (violation > 0. && !(scaled <= structure_tolerance) && no_common_point(p)) {
        out.tested = false;
        return out;
      }
      out.multiplier = std::fmax(most.multiplier, least.multiplier);
      out.scaled     = violation <= 0. ? 0. : scaled;
      return out;
    }

    if (k < 0) {
      // Each end has its own size: a huge bound at one end says nothing about the other.
      sum_t low = constant, high = constant;
      double largest_low = largest, largest_high = largest;
      bool low_infinite = false, high_infinite = false;
      for (int32_t v : touched) {
        double coefficient = g[v];
        bool cancelled     = std::fabs(coefficient) <= composed_zero * std::fmax(1., parts[v]);
        g[v]               = 0.;
        parts[v]           = 0.;
        if (cancelled) { continue; }
        double at_low  = coefficient > 0 ? sys.lower[v] : sys.upper[v];
        double at_high = coefficient > 0 ? sys.upper[v] : sys.lower[v];
        if (std::isfinite(at_low)) {
          low.add(coefficient, at_low);
          raise_to(largest_low, std::fabs(coefficient * at_low));
        } else {
          low_infinite = true;
        }
        if (std::isfinite(at_high)) {
          high.add(coefficient, at_high);
          raise_to(largest_high, std::fabs(coefficient * at_high));
        } else {
          high_infinite = true;
        }
      }
      touched.clear();
      if (lower > -gprs::inf) { below = low_infinite ? gprs::inf : lower - low.value(); }
      if (upper < gprs::inf) { above = high_infinite ? gprs::inf : high.value() - upper; }
      double violation = !(below <= above) ? below : above;
      if (!(violation <= 0.)) {
        out.scaled = violation / (!(below <= above) ? largest_low : largest_high);
      }
      return out;
    }

    out.path = 1;
    for (int64_t e = sys.start[k]; e < sys.start[k + 1]; ++e) {
      int32_t v = sys.var[e];
      if (parts[v] == 0. && c[v] == 0.) { touched.push_back(v); }
      c[v] = sys.value[e];
    }
    terms.clear();
    multipliers.assign(1, 0.);
    sum_t least, most;
    bool least_infinite = false, most_infinite = false;
    for (int32_t v : touched) {
      row_term_t t{g[v], c[v], parts[v], sys.lower[v], sys.upper[v]};
      g[v]     = 0.;
      parts[v] = 0.;
      c[v]     = 0.;
      terms.push_back(t);
      if (t.c == 0.) { continue; }
      multipliers.push_back(t.g / t.c);
      double at_least = t.c > 0 ? t.lower : t.upper;
      double at_most  = t.c > 0 ? t.upper : t.lower;
      if (std::isfinite(at_least)) {
        least.add(t.c, at_least);
      } else {
        least_infinite = true;
      }
      if (std::isfinite(at_most)) {
        most.add(t.c, at_most);
      } else {
        most_infinite = true;
      }
    }
    touched.clear();
    std::sort(multipliers.begin(), multipliers.end());
    size_t different = 1;
    for (size_t m = 1; m < multipliers.size(); ++m) {
      double a = multipliers[m - 1], b = multipliers[m];
      different += !(std::fabs(b - a) <= same_ratio * std::fmax(std::fabs(a), std::fabs(b)));
    }
    double low = sys.low[k], high = sys.high[k];
    bool empty =
      (!least_infinite &&
       least.value() - high > check_tolerance * std::fmax(1., std::fabs(high))) ||
      (!most_infinite && low - most.value() > check_tolerance * std::fmax(1., std::fabs(low)));
    double work = (double)different * (double)terms.size();
    if (empty || work > max_kept_row_work || kept_work + work > max_kept_work) {
      out.tested = false;
      return out;
    }
    kept_work += work;
    double size_above = 1., size_below = 1.;
    if (upper < gprs::inf) {
      above =
        constant.value() + implied_maximum(terms, 1., low, high, multipliers, size_above) - upper;
    }
    if (lower > -gprs::inf) {
      below =
        lower - constant.value() + implied_maximum(terms, -1., low, high, multipliers, size_below);
    }
    double violation = !(below <= above) ? below : above;
    if (!(violation <= 0.)) {
      out.scaled = violation / std::fmax(largest, below > above ? size_below : size_above);
    }
    return out;
  }
};

// The largest and the smallest sum of the terms of the columns of an interval record over
// their bounds; an infinite one is flagged.
struct term_range_t {
  sum_t most;
  sum_t least;
  bool most_infinite  = false;
  bool least_infinite = false;
  // The sum of the terms at prefer.
  sum_t preferred;
};

inline term_range_t term_range(const gprs::postsolve_t& ps, const gprs::col_from_rows_record_t& r)
{
  term_range_t range;
  for (int32_t n = r.cols_begin; n < r.cols_end; ++n) {
    const gprs::record_col_t& col = ps.record_cols[n];
    double a       = ps.record_coefficients[r.coefficients + (n - r.cols_begin)];
    double at_most = a > 0 ? col.upper : col.lower, at_least = a > 0 ? col.lower : col.upper;
    if (std::isinf(at_most)) {
      range.most_infinite = true;
    } else {
      range.most.add(a, at_most);
    }
    if (std::isinf(at_least)) {
      range.least_infinite = true;
    } else {
      range.least.add(a, at_least);
    }
    range.preferred.add(a, col.prefer);
  }
  return range;
}

// The columns part of the check: the boxes and the integral forms of defined records and the
// snapshots of interval records. The stack keeps its contract and `sys` holds its records
// composed.
inline void columns_check(const gprs::result_t& result,
                          const system_t& sys,
                          prover_t& prover,
                          structure_report_t& s)
{
  const gprs::postsolve_t& ps = result.postsolve;
  proof_rows_t with;
  for (size_t n = 0; n < ps.col_from_rows.size(); ++n) {
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    int32_t own                           = sys.first_var[n];
    int32_t anchor                        = sys.first_anchor[n];
    int32_t ncols                         = r.cols_end - r.cols_begin;
    if (r.kind == gprs::col_from_rows_kind_t::forcing) {
      ++s.forcing_records;
      continue;
    }
    if (r.kind == gprs::col_from_rows_kind_t::interval) {
      ++s.interval_records;
      for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
        if (sys.var[e] < own || sys.var[e] >= own + ncols) {
          prover.add_variable(sys.var[e], sys.value[e]);
        }
      }
      term_range_t range = term_range(ps, r);
      double lower = range.most_infinite ? -gprs::inf : sys.low[anchor] - range.most.value();
      double upper = range.least_infinite ? gprs::inf : sys.high[anchor] - range.least.value();
      proof_rows(sys, ps, sys.row[anchor], sys.position[n], with);
      double size = std::fmax(1., std::fmax(std::fabs(sys.low[anchor]), std::fabs(sys.high[anchor])));
      proof_t proof = prover.prove(lower, upper, sum_t(), size, with);
      if (!proof.tested) {
        ++s.snapshot_untested;
        continue;
      }
      ++s.snapshot_tested;
      if (proof.scaled == 0.) { continue; }
      if (!(proof.scaled <= s.snapshot_worst)) { s.snapshot_worst_row = sys.row[anchor]; }
      raise_to(s.snapshot_worst, proof.scaled);
      s.snapshot_failures += !(proof.scaled <= structure_tolerance);
      continue;
    }

    ++s.defined_records;
    const gprs::record_col_t& col = ps.record_cols[r.cols_begin];
    double a                      = ps.record_coefficients[r.coefficients];
    double side                   = sys.low[anchor];
    // The range of the rest of the saved row over the boxes of its variables.
    sum_t most, least;
    bool most_infinite = false, least_infinite = false;
    double largest = std::fabs(side);
    for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
      int32_t v = sys.var[e];
      if (v == own) { continue; }
      double c       = sys.value[e];
      double at_most = c > 0 ? sys.upper[v] : sys.lower[v];
      double at_least = c > 0 ? sys.lower[v] : sys.upper[v];
      if (std::isinf(at_most)) {
        most_infinite = true;
      } else {
        most.add(c, at_most);
        raise_to(largest, std::fabs(c * at_most));
      }
      if (std::isinf(at_least)) {
        least_infinite = true;
      } else {
        least.add(c, at_least);
        raise_to(largest, std::fabs(c * at_least));
      }
    }
    // The largest value of the column is (side - rest) / a at the smallest rest for a > 0.
    bool high_infinite = a > 0 ? least_infinite : most_infinite;
    bool low_infinite  = a > 0 ? most_infinite : least_infinite;
    double high        = (side - (a > 0 ? least.value() : most.value())) / a;
    double low         = (side - (a > 0 ? most.value() : least.value())) / a;
    auto end           = [&](double bound, bool infinite, double beyond) {
      if (std::isinf(bound)) { return; }
      ++s.box_tested;
      double violation = infinite ? gprs::inf : beyond;
      if (violation <= 0.) { return; }
      double scaled =
        violation / std::fmax(std::fmax(1., std::fabs(bound)), largest / std::fabs(a));
      if (!(scaled <= s.box_worst)) { s.box_worst_col = col.col; }
      raise_to(s.box_worst, scaled);
      s.box_failures += !(scaled <= structure_tolerance);
    };
    end(col.upper, high_infinite, high - col.upper);
    end(col.lower, low_infinite, col.lower - low);

    if (!ps.integer_at(col.col, sys.position[n])) { continue; }
    // Integral for the column: integer columns, integral ratios and side, and the drift of the
    // rounded ratios over the boxes within the tolerance, in the units of the row.
    const gprs::saved_row_t& row = ps.saved_rows[r.rows_begin];
    bool integral                = true;
    double drift                 = std::fabs(a * std::round(row.upper / a) - row.upper);
    for (int64_t e = row.begin; e < row.end && integral; ++e) {
      int32_t other         = ps.saved_entries.col[e];
      double ratio          = ps.saved_entries.value[e] / a;
      double delta          = std::fabs(ratio - std::round(ratio));
      const column_map_t& m = sys.map[other];
      integral              = ps.integer_at(other, sys.position[n]) && delta <= integral_ratio;
      if (delta == 0. || !integral) { continue; }
      double reach = std::fabs(m.offset);
      if (m.root >= 0) {
        reach += std::fabs(m.scale) *
                 std::fmax(std::fabs(sys.lower[m.root]), std::fabs(sys.upper[m.root]));
      }
      drift += std::fabs(a) * delta * reach;
    }
    if (!integral || !(drift <= structure_tolerance * std::fmax(1., std::fabs(row.upper)))) {
      if (s.integral_failures++ == 0) { s.first_integral_col = col.col; }
    }
  }
}

// The objective part of the check. The stack keeps its contract and `sys` holds its records
// composed (compose_records). O(columns + records + entries of the anchors).
inline void objective_check(const gprs::model_t& model,
                            const gprs::result_t& result,
                            const system_t& sys,
                            structure_report_t& s)
{
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;
  // The sum of the absolute values that the composed offset of a column is made of: an alias
  // adds its own offset to the scaled offset of its rep, and the two can cancel. A column of a
  // col_from_rows record is a variable and has none.
  std::vector<double> offset_parts(model.ncols, 0.);
  size_t fixed = ps.fixed_cols.size(), alias = ps.aliases.size();
  for (size_t k = ps.order.size(); k-- > 0;) {
    if (ps.order[k] == gprs::record_kind_t::fixed_col) {
      const gprs::fixed_col_record_t& r = ps.fixed_cols[--fixed];
      offset_parts[r.col]               = std::fabs(r.value);
    } else if (ps.order[k] == gprs::record_kind_t::alias) {
      const gprs::alias_record_t& r = ps.aliases[--alias];
      offset_parts[r.col] = std::fabs(r.scale) * offset_parts[r.rep] + std::fabs(r.offset);
    }
  }

  std::vector<sum_t> coefficient(sys.nvars());
  std::vector<double> parts(sys.nvars(), 0.);
  sum_t constant;
  constant.add(model.objective_offset, 1.);
  constant.add(-reduced.objective_offset, 1.);
  double terms = std::fabs(model.objective_offset) + std::fabs(reduced.objective_offset);
  for (int32_t j = 0; j < model.ncols; ++j) {
    double cost = model.cost[j];
    if (cost == 0.) { continue; }
    const column_map_t& m = sys.map[j];
    constant.add(cost, m.offset);
    terms += std::fabs(cost) * offset_parts[j];
    if (m.root < 0) { continue; }
    coefficient[m.root].add(cost, m.scale);
    parts[m.root] += std::fabs(cost * m.scale);
  }
  for (int32_t k = 0; k < reduced.ncols; ++k) {
    coefficient[k].add(-reduced.cost[k], 1.);
    parts[k] += std::fabs(reduced.cost[k]);
  }
  // The variable of a defined record is what its saved row gives it. In the order of the stack
  // every variable of a saved row is one of the reduced model or of a later record.
  for (size_t n = 0; n < ps.col_from_rows.size(); ++n) {
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    if (r.kind != gprs::col_from_rows_kind_t::defined) { continue; }
    int32_t own    = sys.first_var[n];
    int32_t anchor = sys.first_anchor[n];
    double held    = coefficient[own].value();
    if (held == 0.) { continue; }
    double factor = held / ps.record_coefficients[r.coefficients];
    for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
      int32_t v = sys.var[e];
      if (v == own) { continue; }
      coefficient[v].add(-factor, sys.value[e]);
      parts[v] += std::fabs(factor * sys.value[e]);
    }
    constant.add(factor, sys.low[anchor]);
    terms += std::fabs(factor * sys.low[anchor]);
    coefficient[own] = sum_t();
    parts[own]       = 0.;
  }
  for (int32_t v = 0; v < sys.nvars(); ++v) {
    double difference = std::fabs(coefficient[v].value());
    if (difference <= underflow_size) { continue; }
    double relative = difference / parts[v];
    if (!(relative <= s.objective_worst)) { s.objective_worst_col = sys.origin[v]; }
    raise_to(s.objective_worst, relative);
    s.objective_failures += !(relative <= composed_zero);
  }
  double difference = std::fabs(constant.value());
  if (!(difference <= underflow_size)) {
    s.objective_constant = difference / terms;
    s.objective_failures += !(s.objective_constant <= identity_tolerance);
  }
}

// The values of the check variables at one point of the reduced model, by the rules of
// postsolve.hpp applied to the anchors: the second implementation of the undo of the
// col_from_rows records. size is what a difference of a value is relative to.
struct replay_t {
  std::vector<double> value;
  std::vector<double> size;
};

// The rest of an anchor at the values of the replay: the sum of its terms outside the variables
// [own, own + ncols) of its record. `largest` gets the largest of those terms.
inline double anchor_rest(const system_t& sys,
                          const replay_t& at,
                          int32_t anchor,
                          int32_t own,
                          int32_t ncols,
                          double& largest)
{
  sum_t rest;
  for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
    int32_t v = sys.var[e];
    if (v >= own && v < own + ncols) { continue; }
    rest.add(sys.value[e], at.value[v]);
    raise_to(largest, std::fabs(sys.value[e]) * at.size[v]);
  }
  return rest.value();
}

inline void replay(const gprs::result_t& result,
                   const system_t& sys,
                   const std::vector<double>& point,
                   replay_t& at)
{
  const gprs::postsolve_t& ps = result.postsolve;
  at.value.assign(sys.nvars(), 0.);
  at.size.assign(sys.nvars(), 0.);
  for (int32_t k = 0; k < sys.reduced_cols; ++k) {
    bool integer = ps.integer_at(ps.col_origin[k], ps.order.size());
    at.value[k]  = integer ? std::round(point[k]) : point[k];
    at.size[k]   = std::fabs(at.value[k]);
  }
  for (size_t n = ps.col_from_rows.size(); n-- > 0;) {
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    int32_t own = sys.first_var[n], anchor = sys.first_anchor[n];
    int32_t ncols                 = r.cols_end - r.cols_begin;
    const gprs::record_col_t& col = ps.record_cols[r.cols_begin];
    if (r.kind == gprs::col_from_rows_kind_t::defined) {
      double a       = ps.record_coefficients[r.coefficients];
      double largest = std::fabs(sys.low[anchor]);
      double value   = (sys.low[anchor] - anchor_rest(sys, at, anchor, own, 1, largest)) / a;
      at.value[own]  = ps.integer_at(col.col, sys.position[n]) ? std::round(value) : value;
      at.size[own]   = std::fmax(std::fabs(at.value[own]), largest / std::fabs(a));
    } else if (r.kind == gprs::col_from_rows_kind_t::interval) {
      double largest = std::fmax(std::fabs(sys.low[anchor]), std::fabs(sys.high[anchor]));
      double rest    = anchor_rest(sys, at, anchor, own, ncols, largest);
      sum_t activity;
      activity.add(rest, 1.);
      for (int32_t c = 0; c < ncols; ++c) {
        const gprs::record_col_t& taken = ps.record_cols[r.cols_begin + c];
        double a                        = ps.record_coefficients[r.coefficients + c];
        at.value[own + c]               = taken.prefer;
        activity.add(a, taken.prefer);
        raise_to(largest, std::fabs(a * taken.prefer));
      }
      double below = sys.low[anchor] - activity.value();
      double above = activity.value() - sys.high[anchor];
      bool raise   = below > 0;
      double need  = raise ? below : above;
      for (int32_t c = 0; c < ncols && need > 0; ++c) {
        const gprs::record_col_t& taken = ps.record_cols[r.cols_begin + c];
        double a                        = ps.record_coefficients[r.coefficients + c];
        double target                   = (a > 0) == raise ? taken.upper : taken.lower;
        double room                     = std::fabs(a) * std::fabs(target - taken.prefer);
        if (room <= need) {
          at.value[own + c] = target;
          need -= room;
        } else {
          at.value[own + c] = taken.prefer + (raise ? need : -need) / a;
          need              = 0.;
        }
      }
      for (int32_t c = 0; c < ncols; ++c) {
        double a         = ps.record_coefficients[r.coefficients + c];
        at.size[own + c] = std::fmax(std::fabs(at.value[own + c]), largest / std::fabs(a));
      }
    } else {
      bool from_above = forcing_from_above(ps, r);
      double value    = from_above ? col.upper : col.lower;
      double size     = 0.;
      for (int32_t j = 0; j < r.rows_end - r.rows_begin; ++j) {
        double a       = ps.record_coefficients[r.coefficients + j];
        double side    = from_above == (a > 0) ? sys.high[anchor + j] : sys.low[anchor + j];
        double largest = std::fabs(side);
        double limit   = (side - anchor_rest(sys, at, anchor + j, own, 1, largest)) / a;
        raise_to(size, largest / std::fabs(a));
        if (from_above ? limit < value : limit > value) { value = limit; }
      }
      if (ps.integer_at(col.col, sys.position[n])) {
        value = from_above ? std::floor(value) : std::ceil(value);
      }
      at.value[own] = value;
      at.size[own]  = std::fmax(std::fabs(value), size);
    }
  }
}

// The postsolve part of the check at one point of the reduced model.
inline void postsolve_at(const gprs::model_t& model,
                         const gprs::result_t& result,
                         const system_t& sys,
                         const std::vector<double>& point,
                         replay_t& at,
                         std::vector<double>& z,
                         structure_report_t& s)
{
  const gprs::postsolve_t& ps = result.postsolve;
  ps.postsolve(point, z);
  replay(result, sys, point, at);
  auto failure = [&](double scaled, int32_t col) {
    if (!(scaled <= s.postsolve_worst)) { s.postsolve_worst_col = col; }
    raise_to(s.postsolve_worst, scaled);
    ++s.postsolve_failures;
  };
  for (int32_t j = 0; j < model.ncols; ++j) {
    const column_map_t& m = sys.map[j];
    double term           = m.root < 0 ? 0. : m.scale * at.value[m.root];
    double reach          = m.root < 0 ? 0. : std::fabs(m.scale) * at.size[m.root];
    double size           = std::fmax(1., std::fmax(reach, std::fabs(m.offset)));
    double difference     = std::fabs(z[j] - (term + m.offset)) / size;
    if (!(difference <= s.postsolve_worst)) { s.postsolve_worst_col = j; }
    raise_to(s.postsolve_worst, difference);
    s.postsolve_failures += !(difference <= composed_tolerance);
  }
  // The saved rows and the boxes at the point that the engine postsolved, read from the records
  // as they are.
  for (const gprs::col_from_rows_record_t& r : ps.col_from_rows) {
    const gprs::record_col_t& col = ps.record_cols[r.cols_begin];
    bool interval                 = r.kind == gprs::col_from_rows_kind_t::interval;
    bool forcing                  = r.kind == gprs::col_from_rows_kind_t::forcing;
    bool from_above               = forcing && forcing_from_above(ps, r);
    for (int32_t n = r.rows_begin; n < r.rows_end; ++n) {
      const gprs::saved_row_t& row = ps.saved_rows[n];
      sum_t activity;
      double largest = 1.;
      for (int64_t e = row.begin; e < row.end; ++e) {
        double a = ps.saved_entries.value[e], v = z[ps.saved_entries.col[e]];
        activity.add(a, v);
        raise_to(largest, std::fabs(a * v));
      }
      double rest  = activity.value();
      double lower = row.lower, upper = row.upper;
      if (interval) {
        // The row need hold only where the rest lies inside the sides of the snapshot.
        term_range_t range = term_range(ps, r);
        double low  = range.most_infinite ? -gprs::inf : lower - range.most.value();
        double high = range.least_infinite ? gprs::inf : upper - range.least.value();
        double slack = structure_tolerance * std::fmax(largest, std::fmax(std::fabs(lower), std::fabs(upper)));
        if (!(rest >= low - slack && rest <= high + slack)) { continue; }
        for (int32_t c = r.cols_begin; c < r.cols_end; ++c) {
          double a = ps.record_coefficients[r.coefficients + (c - r.cols_begin)];
          double v = z[ps.record_cols[c].col];
          activity.add(a, v);
          raise_to(largest, std::fabs(a * v));
        }
      } else {
        double a = ps.record_coefficients[r.coefficients + (forcing ? n - r.rows_begin : 0)];
        activity.add(a, z[col.col]);
        raise_to(largest, std::fabs(a * z[col.col]));
        if (forcing) {
          // Only the side that bounds the column is a side of the record.
          bool upper_side = from_above == (a > 0);
          lower           = upper_side ? -gprs::inf : lower;
          upper           = upper_side ? upper : gprs::inf;
        }
      }
      double value     = activity.value();
      double violation = std::isnan(value) ? value : std::fmax(lower - value, value - upper);
      double size      = std::fmax(largest, lower > -gprs::inf ? std::fabs(lower) : std::fabs(upper));
      if (!(violation / size <= structure_tolerance)) { failure(violation / size, col.col); }
    }
    if (!interval && !forcing) { continue; }
    for (int32_t c = r.cols_begin; c < r.cols_end; ++c) {
      const gprs::record_col_t& taken = ps.record_cols[c];
      double v                        = z[taken.col];
      double violation = std::isnan(v) ? v : std::fmax(taken.lower - v, v - taken.upper);
      double bound     = v < taken.lower ? taken.lower : taken.upper;
      double scaled    = violation / std::fmax(1., std::fabs(bound));
      if (!(scaled <= structure_tolerance)) { failure(scaled, taken.col); }
    }
  }
}

// The postsolve part of the check: the points of the reduced model and postsolve_at for each.
inline void postsolve_check(const gprs::model_t& model,
                            const gprs::result_t& result,
                            const system_t& sys,
                            structure_report_t& s)
{
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;
  std::vector<double> point(reduced.ncols), z;
  replay_t at;
  // 0: 1 moved into the bounds; 1: the lower bound; 2: the upper bound; 3: between them. An
  // infinite bound is replaced by the other one less or plus 1, or by -1 and 1.
  for (int32_t variant = 0; variant < 4; ++variant) {
    for (int32_t k = 0; k < reduced.ncols; ++k) {
      double lower = reduced.col_lower[k], upper = reduced.col_upper[k];
      double low   = lower > -gprs::inf ? lower : upper < gprs::inf ? upper - 1. : -1.;
      double high  = upper < gprs::inf ? upper : lower > -gprs::inf ? lower + 1. : 1.;
      double v     = variant == 1 ? low : variant == 2 ? high : 0.5 * low + 0.5 * high;
      if (variant == 0) {
        v = 1.;
        if (v < lower) { v = lower; }
        if (v > upper) { v = upper; }
      }
      point[k] = ps.integer_at(ps.col_origin[k], ps.order.size()) ? std::round(v) : v;
    }
    postsolve_at(model, result, sys, point, at, z, s);
  }
  // For some interval records a point at which the undo has to move the columns half of the way
  // they can go: from the point between the bounds, one reduced column of the saved row is
  // moved so that the rest of the row asks for it. It need not lie within the bounds.
  std::vector<size_t> interval;
  for (size_t n = 0; n < ps.col_from_rows.size(); ++n) {
    if (ps.col_from_rows[n].kind == gprs::col_from_rows_kind_t::interval) { interval.push_back(n); }
  }
  size_t count = interval.size() < max_replay_points ? interval.size() : max_replay_points;
  std::vector<double> moved;
  for (size_t pick = 0; pick < count; ++pick) {
    size_t n                              = interval[pick * interval.size() / count];
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    int32_t own = sys.first_var[n], anchor = sys.first_anchor[n];
    replay(result, sys, point, at);
    double largest     = 1.;
    double rest        = anchor_rest(sys, at, anchor, own, r.cols_end - r.cols_begin, largest);
    term_range_t range = term_range(ps, r);
    double preferred   = range.preferred.value();
    double up   = range.most_infinite ? gprs::inf : range.most.value() - preferred;
    double down = range.least_infinite ? gprs::inf : preferred - range.least.value();
    double target;
    if (up > 0) {
      target = sys.low[anchor] - preferred - (std::isfinite(up) ? 0.5 * up : 1.);
    } else if (down > 0) {
      target = sys.high[anchor] - preferred + (std::isfinite(down) ? 0.5 * down : 1.);
    } else {
      continue;
    }
    int64_t best = -1;
    for (int64_t e = sys.start[anchor]; e < sys.start[anchor + 1]; ++e) {
      if (sys.var[e] < sys.reduced_cols &&
          (best < 0 || std::fabs(sys.value[e]) > std::fabs(sys.value[best]))) {
        best = e;
      }
    }
    if (best < 0) { continue; }
    moved = point;
    moved[sys.var[best]] += (target - rest) / sys.value[best];
    postsolve_at(model, result, sys, moved, at, z, s);
  }
}

// The check of one result of a presolve in the given mode (options_t::primal_only).
inline structure_report_t structure_check(const gprs::model_t& model,
                                          const gprs::result_t& result,
                                          bool primal_only)
{
  structure_report_t s;
  s.ran = true;
  system_t sys;
  if (!compose_records(model, result, primal_only, sys, s.stack_error)) { return s; }
  s.composed                   = true;
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;

  for (int32_t k = 0; k < reduced.ncols; ++k) {
    int32_t col = ps.col_origin[k];
    if (model.col_type[col] && !reduced.col_type[k]) {
      if (s.lost_integers++ == 0) { s.first_lost_integer = col; }
    }
  }

  objective_check(model, result, sys, s);

  for (int32_t j = 0; j < model.ncols; ++j) {
    const column_map_t& m = sys.map[j];
    double low = m.offset, high = m.offset;
    if (m.root >= 0) {
      double a = m.scale * sys.lower[m.root] + m.offset;
      double b = m.scale * sys.upper[m.root] + m.offset;
      low      = m.scale > 0 ? a : b;
      high     = m.scale > 0 ? b : a;
    }
    double below     = model.col_lower[j] == -gprs::inf ? -gprs::inf : model.col_lower[j] - low;
    double above     = model.col_upper[j] == gprs::inf ? -gprs::inf : high - model.col_upper[j];
    double violation = !(below <= above) ? below : above;
    if (violation <= 0.) { continue; }
    double bound  = below > above ? model.col_lower[j] : model.col_upper[j];
    double scaled = violation / std::fmax(1., std::fabs(bound));
    if (!(scaled <= s.bound_worst)) { s.bound_worst_col = j; }
    raise_to(s.bound_worst, scaled);
    s.bound_failures += !(scaled <= structure_tolerance);
  }

  prover_t prover(sys);
  // The snapshots first: an anchor of an interval record holds only as far as its snapshot.
  columns_check(result, sys, prover, s);

  proof_rows_t with;
  for (int32_t i = 0; i < model.nrows; ++i) {
    sum_t constant;
    double largest = 1.;
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      prover.add_column(sys.map[model.col_index[e]], model.value[e], constant, largest);
    }
    proof_rows(sys, ps, i, -1, with);
    s.added_rows += with.made_of_equations;
    proof_t proof = prover.prove(model.row_lower[i], model.row_upper[i], constant, largest, with);
    if (proof.path == 2) {
      if (!proof.tested) {
        ++s.added_untested;
        continue;
      }
      ++s.added_tested;
      raise_to(s.added_multiplier, proof.multiplier);
      if (proof.scaled == 0.) { continue; }
      if (!(proof.scaled <= s.added_worst)) { s.added_worst_row = i; }
      raise_to(s.added_worst, proof.scaled);
      s.added_failures += !(proof.scaled <= structure_tolerance);
    } else if (proof.path == 0) {
      ++s.rows_tested;
      if (proof.scaled == 0.) { continue; }
      if (!(proof.scaled <= s.row_worst)) { s.row_worst_row = i; }
      raise_to(s.row_worst, proof.scaled);
      s.row_failures += !(proof.scaled <= structure_tolerance);
    } else {
      if (!proof.tested) {
        ++s.kept_untested;
        continue;
      }
      ++s.kept_tested;
      s.kept_parallel += with.other_row;
      if (proof.scaled == 0.) { continue; }
      if (!(proof.scaled <= s.kept_worst)) { s.kept_worst_row = i; }
      raise_to(s.kept_worst, proof.scaled);
      s.kept_failures += !(proof.scaled <= structure_tolerance);
    }
  }

  postsolve_check(model, result, sys, s);
  return s;
}

}  // namespace gprs_check
