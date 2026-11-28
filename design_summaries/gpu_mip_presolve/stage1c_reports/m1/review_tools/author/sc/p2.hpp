// What happens to an original row at one position of the stack.
struct row_event_t {
  enum kind_t : uint8_t { saved, parallel, addition };
  size_t position;
  kind_t kind;
  // saved: the anchor of the saved row; parallel: the kept row; addition: the equation row.
  int32_t index;
};

// The records composed: the check variables with their boxes, the map of every original column
// onto them, the anchors, and what the stack does to each row.
struct system_t {
  int32_t reduced_cols = 0;
  int32_t reduced_rows = 0;
  std::vector<column_map_t> map;
  // Variables: the reduced columns, then the columns of the col_from_rows records in the order
  // of the stack.
  std::vector<double> lower;
  std::vector<double> upper;
  std::vector<int32_t> origin;  // original column
  // Anchors: the rows of the reduced model, then the saved rows of the col_from_rows records,
  // each with its entries [start[a], start[a + 1]) over the variables, the record's own columns
  // among them, and its sides less the constants of the composition.
  std::vector<int64_t> start;
  std::vector<int32_t> var;
  std::vector<double> value;
  std::vector<double> low;
  std::vector<double> high;
  std::vector<int32_t> row;  // original row
  // By original row: its row of the reduced model as an anchor, or -1, and its events
  // [first[row], first[row + 1]) in the order of the stack.
  std::vector<int32_t> reduced_anchor;
  std::vector<int32_t> first;
  std::vector<row_event_t> events;
  // By col_from_rows record: its position in the order of the stack, its first variable, and
  // the anchor of its first saved row; the others follow.
  std::vector<size_t> position;
  std::vector<int32_t> first_var;
  std::vector<int32_t> first_anchor;

  int32_t nvars() const { return lower.size(); }
  int32_t anchors() const { return low.size(); }
};

// A forcing record bounds its column from above (postsolve.hpp).
inline bool forcing_from_above(const gprs::postsolve_t& ps, const gprs::col_from_rows_record_t& r)
{
  return (ps.record_coefficients[r.coefficients] > 0) ==
         (ps.saved_rows[r.rows_begin].upper < gprs::inf);
}

// Tests the contract of the stack, for a result of the given mode, and composes its records. On
// a broken contract `error` says which part, with the index of the record or column.
inline bool compose_records(const gprs::model_t& model,
                            const gprs::result_t& result,
                            bool primal_only,
                            system_t& sys,
                            std::string& error)
{
  const gprs::postsolve_t& ps  = result.postsolve;
  const gprs::model_t& reduced = result.reduced;
  std::vector<column_map_t>& map = sys.map;
  auto fail                    = [&](const char* what, int64_t index) {
    error = std::string(what) + " " + std::to_string(index);
    return false;
  };
  if (ps.ncols != model.ncols || ps.nrows != model.nrows) { return fail("sizes of the stack", 0); }
  if (primal_only && result.dual_findings != 0) {
    return fail("dual findings in a primal-only result:", result.dual_findings);
  }
  if (ps.col_origin.size() != (size_t)reduced.ncols ||
      ps.row_origin.size() != (size_t)reduced.nrows) {
    return fail("sizes of the origin maps", 0);
  }
  const int32_t unset = -2;
  map.assign(model.ncols, column_map_t{unset, 0., 0.});
  for (int32_t k = 0; k < reduced.ncols; ++k) {
    int32_t col = ps.col_origin[k];
    if (col < 0 || col >= model.ncols || map[col].root != unset) {
      return fail("origin of reduced column", k);
    }
    map[col] = column_map_t{k, 1., 0.};
  }
  std::vector<uint8_t> seen(model.nrows, 0);
  for (int32_t k = 0; k < reduced.nrows; ++k) {
    int32_t row = ps.row_origin[k];
    if (row < 0 || row >= model.nrows || seen[row]) { return fail("origin of reduced row", k); }
    seen[row] = 1;
  }

  auto column_ok = [&](int32_t col) { return col >= 0 && col < model.ncols; };
  auto row_ok    = [&](int32_t row) { return row >= 0 && row < model.nrows; };
  std::vector<uint8_t> removed(model.ncols, 0);
  // Rows that a record took out of the model.
  std::vector<uint8_t> row_gone(model.nrows, 0);
  size_t fixed = 0, alias = 0, parallel = 0, addition = 0, from_rows = 0;
  int32_t cols_used = 0, rows_used = 0;
  int64_t coefficients_used = 0, entries_used = 0;
  int64_t left              = model.ncols - reduced.ncols;
  std::vector<int32_t> events_of(model.nrows + 1, 0);
  for (size_t k = 0; k < ps.order.size(); ++k) {
    if (ps.order[k] == gprs::record_kind_t::fixed_col) {
      if (fixed == ps.fixed_cols.size()) { return fail("fixed records missing at", k); }
      const gprs::fixed_col_record_t& r = ps.fixed_cols[fixed++];
      if (!column_ok(r.col) || removed[r.col] || map[r.col].root != unset) {
        return fail("column of fixed record", k);
      }
      if (!std::isfinite(r.value) || (ps.integer_at(r.col, k) && !is_integral(r.value))) {
        return fail("value of fixed record", k);
      }
      removed[r.col] = 1;
      --left;
    } else if (ps.order[k] == gprs::record_kind_t::alias) {
      if (alias == ps.aliases.size()) { return fail("alias records missing at", k); }
      const gprs::alias_record_t& r = ps.aliases[alias++];
      if (!column_ok(r.col) || !column_ok(r.rep) || r.col == r.rep || removed[r.col] ||
          removed[r.rep] || map[r.col].root != unset) {
        return fail("columns of alias record", k);
      }
      bool doubleton = r.origin == gprs::alias_origin_t::doubleton;
      bool probing   = r.origin == gprs::alias_origin_t::probing;
      bool lock      = r.origin == gprs::alias_origin_t::lock;
      if (!doubleton && !probing && !lock) { return fail("origin of alias record", k); }
      if (lock && primal_only) { return fail("lock alias in a primal-only result at", k); }
      bool has_row = doubleton || (probing && r.row != -1);
      if (has_row ? !row_ok(r.row) : r.row != -1) { return fail("row of alias record", k); }
      if (!std::isfinite(r.scale) || r.scale == 0. || !std::isfinite(r.offset)) {
        return fail("scale or offset of alias record", k);
      }
      if (ps.integer_at(r.col, k) &&
          !(ps.integer_at(r.rep, k) && is_integral(r.scale) && is_integral(r.offset))) {
        return fail("integer column of alias record", k);
      }
      removed[r.col] = 1;
      if (has_row) { row_gone[r.row] = 1; }
      --left;
    } else if (ps.order[k] == gprs::record_kind_t::parallel_row) {
      if (parallel == ps.parallel_rows.size()) { return fail("parallel records missing at", k); }
      const gprs::parallel_row_record_t& r = ps.parallel_rows[parallel++];
      if (!row_ok(r.removed_row) || !row_ok(r.kept_row) || r.removed_row == r.kept_row ||
          !std::isfinite(r.ratio) || r.ratio == 0.) {
        return fail("parallel record", k);
      }
      row_gone[r.removed_row] = 1;
      ++events_of[r.removed_row + 1];
    } else if (ps.order[k] == gprs::record_kind_t::row_addition) {
      if (addition == ps.row_additions.size()) {
        return fail("row addition records missing at", k);
      }
      const gprs::row_addition_record_t& r = ps.row_additions[addition++];
      if (!row_ok(r.row) || !row_ok(r.equation_row) || r.row == r.equation_row ||
          !std::isfinite(r.factor) || r.factor == 0.) {
        return fail("row addition record", k);
      }
      if (row_gone[r.row] || row_gone[r.equation_row]) {
        return fail("rows of row addition record", k);
      }
      ++events_of[r.row + 1];
    } else if (ps.order[k] == gprs::record_kind_t::col_from_rows) {
      if (from_rows == ps.col_from_rows.size()) {
        return fail("col_from_rows records missing at", k);
      }
      const gprs::col_from_rows_record_t& r = ps.col_from_rows[from_rows++];
      int64_t ncols = (int64_t)r.cols_end - r.cols_begin, nrows = (int64_t)r.rows_end - r.rows_begin;
      int64_t count = ncols > nrows ? ncols : nrows;
      if (r.cols_begin < cols_used || ncols < 1 || (size_t)r.cols_end > ps.record_cols.size() ||
          r.rows_begin < rows_used || nrows < 1 || (size_t)r.rows_end > ps.saved_rows.size() ||
          r.coefficients < coefficients_used ||
          (size_t)(r.coefficients + count) > ps.record_coefficients.size()) {
        return fail("ranges of col_from_rows record", k);
      }
      cols_used         = r.cols_end;
      rows_used         = r.rows_end;
      coefficients_used = r.coefficients + count;
      bool defined      = r.kind == gprs::col_from_rows_kind_t::defined;
      bool interval     = r.kind == gprs::col_from_rows_kind_t::interval;
      bool forcing      = r.kind == gprs::col_from_rows_kind_t::forcing;
      if ((!defined && !interval && !forcing) || (!interval && ncols != 1) ||
          (!forcing && nrows != 1) || (r.dual && !defined)) {
        return fail("kind or counts of col_from_rows record", k);
      }
      if (primal_only && r.dual) { return fail("dual record in a primal-only result at", k); }
      if (primal_only && !defined) {
        return fail("interval or forcing record in a primal-only result at", k);
      }
      for (int64_t n = 0; n < count; ++n) {
        double a = ps.record_coefficients[r.coefficients + n];
        if (!std::isfinite(a) || a == 0.) { return fail("coefficient of col_from_rows record", k); }
      }
      for (int32_t n = r.cols_begin; n < r.cols_end; ++n) {
        const gprs::record_col_t& c = ps.record_cols[n];
        if (!column_ok(c.col) || removed[c.col] || map[c.col].root != unset ||
            (n > r.cols_begin && ps.record_cols[n - 1].col >= c.col)) {
          return fail("column of col_from_rows record", k);
        }
        if (!(c.lower <= c.upper) || std::isnan(c.prefer)) {
          return fail("bounds of col_from_rows record", k);
        }
        if (interval && ps.integer_at(c.col, k)) {
          return fail("integer column of interval record", k);
        }
        removed[c.col] = 1;
        --left;
      }
      const gprs::record_col_t& c = ps.record_cols[r.cols_begin];
      bool from_above             = forcing && forcing_from_above(ps, r);
      if (forcing && (from_above ? c.lower != -gprs::inf : c.upper != gprs::inf)) {
        return fail("bound of forcing record", k);
      }
      for (int32_t n = r.rows_begin; n < r.rows_end; ++n) {
        const gprs::saved_row_t& row = ps.saved_rows[n];
        if (!row_ok(row.row) || row_gone[row.row] || row.begin < entries_used ||
            row.begin > row.end || (size_t)row.end > ps.saved_entries.col.size() ||
            ps.saved_entries.value.size() != ps.saved_entries.col.size() ||
            !(row.lower <= row.upper)) {
          return fail("saved row of col_from_rows record", k);
        }
        entries_used = row.end;
        if (defined && !(row.lower == row.upper && std::isfinite(row.upper))) {
          return fail("side of defined record", k);
        }
        if (interval && !(std::isfinite(row.lower) && std::isfinite(row.upper))) {
          return fail("sides of interval record", k);
        }
        if (forcing) {
          // Every row bounds the column from the end that the first one does.
          double a   = ps.record_coefficients[r.coefficients + (n - r.rows_begin)];
          bool upper = from_above == (a > 0);
          if (!std::isfinite(upper ? row.upper : row.lower) ||
              !std::isinf(upper ? row.lower : row.upper)) {
            return fail("row of forcing record", k);
          }
        }
        for (int64_t e = row.begin; e < row.end; ++e) {
          int32_t col = ps.saved_entries.col[e];
          double a    = ps.saved_entries.value[e];
          if (!column_ok(col) || removed[col] || !std::isfinite(a) || a == 0.) {
            return fail("entry of col_from_rows record", k);
          }
        }
        if (!interval) { row_gone[row.row] = 1; }
        ++events_of[row.row + 1];
      }
    } else {
      return fail("unknown record kind at", k);
    }
  }
  if (fixed != ps.fixed_cols.size() || alias != ps.aliases.size() ||
      parallel != ps.parallel_rows.size() || addition != ps.row_additions.size() ||
      from_rows != ps.col_from_rows.size()) {
    return fail("records outside the order", 0);
  }
  if (left != 0) { return fail("columns neither left nor removed:", left); }

  // Variables and anchors of the reduced model.
  sys.reduced_cols = reduced.ncols;
  sys.reduced_rows = reduced.nrows;
  sys.lower        = reduced.col_lower;
  sys.upper        = reduced.col_upper;
  sys.origin       = ps.col_origin;
  sys.first_var.assign(ps.col_from_rows.size(), 0);
  sys.first_anchor.assign(ps.col_from_rows.size(), 0);
  sys.position.assign(ps.col_from_rows.size(), 0);
  int32_t saved_anchors = 0;
  for (size_t n = 0; n < ps.col_from_rows.size(); ++n) {
    const gprs::col_from_rows_record_t& r = ps.col_from_rows[n];
    sys.first_var[n]                      = sys.lower.size();
    sys.first_anchor[n]                   = reduced.nrows + saved_anchors;
    saved_anchors += r.rows_end - r.rows_begin;
    for (int32_t c = r.cols_begin; c < r.cols_end; ++c) {
      sys.lower.push_back(ps.record_cols[c].lower);
      sys.upper.push_back(ps.record_cols[c].upper);
      sys.origin.push_back(ps.record_cols[c].col);
    }
  }
  sys.start.assign(reduced.row_start.begin(), reduced.row_start.end());
  sys.var   = reduced.col_index;
  sys.value = reduced.value;
  sys.low   = reduced.row_lower;
  sys.high  = reduced.row_upper;
  sys.row   = ps.row_origin;
  sys.reduced_anchor.assign(model.nrows, -1);
  for (int32_t k = 0; k < reduced.nrows; ++k) {
    sys.reduced_anchor[ps.row_origin[k]] = k;
  }

  // The maps, from the end of the stack. A saved row is composed with the maps as they are when
  // the pass reaches its record, which are the meanings of its columns at that position; only
  // then do the record's own columns get their variables. The anchors of a record are composed
  // into their places afterwards, since the pass meets the records in reverse.
  struct composed_row_t {
    std::vector<int32_t> var;
    std::vector<double> value;
    double low;
    double high;
    int32_t row;
  };
  std::vector<composed_row_t> composed(saved_anchors);
  std::vector<sum_t> sum(sys.nvars());
  std::vector<double> parts(sys.nvars(), 0.);
  std::vector<int32_t> touched;
  fixed     = ps.fixed_cols.size();
  alias     = ps.aliases.size();
  from_rows = ps.col_from_rows.size();
  for (size_t k = ps.order.size(); k-- > 0;) {
    if (ps.order[k] == gprs::record_kind_t::fixed_col) {
      const gprs::fixed_col_record_t& r = ps.fixed_cols[--fixed];
      map[r.col]                        = column_map_t{-1, 0., r.value};
    } else if (ps.order[k] == gprs::record_kind_t::alias) {
      const gprs::alias_record_t& r = ps.aliases[--alias];
      const column_map_t& rep       = map[r.rep];
      map[r.col] = column_map_t{rep.root, r.scale * rep.scale, r.scale * rep.offset + r.offset};
    } else if (ps.order[k] == gprs::record_kind_t::col_from_rows) {
      const gprs::col_from_rows_record_t& r = ps.col_from_rows[--from_rows];
      sys.position[from_rows]               = k;
      int32_t ncols                         = r.cols_end - r.cols_begin;
      for (int32_t n = r.rows_begin; n < r.rows_end; ++n) {
        const gprs::saved_row_t& row = ps.saved_rows[n];
        composed_row_t& out = composed[sys.first_anchor[from_rows] - reduced.nrows + (n - r.rows_begin)];
        sum_t constant;
        for (int64_t e = row.begin; e < row.end; ++e) {
          const column_map_t& m = map[ps.saved_entries.col[e]];
          double a              = ps.saved_entries.value[e];
          constant.add(a, m.offset);
          if (m.root < 0) { continue; }
          if (parts[m.root] == 0.) { touched.push_back(m.root); }
          sum[m.root].add(a, m.scale);
          parts[m.root] += std::fabs(a * m.scale);
        }
        for (int32_t v : touched) {
          double a = sum[v].value();
          // What is left of terms that cancel is their rounding.
          if (std::fabs(a) > rounding_zero * parts[v]) {
            out.var.push_back(v);
            out.value.push_back(a);
          }
          sum[v]   = sum_t();
          parts[v] = 0.;
        }
        touched.clear();
        // The record's own columns: all of them in its one row, or its one column in each row.
        for (int32_t c = 0; c < (ncols > 1 ? ncols : 1); ++c) {
          int32_t at = ncols > 1 ? c : n - r.rows_begin;
          out.var.push_back(sys.first_var[from_rows] + c);
          out.value.push_back(ps.record_coefficients[r.coefficients + at]);
        }
        out.low  = row.lower - constant.value();
        out.high = row.upper - constant.value();
        out.row  = row.row;
      }
      for (int32_t c = 0; c < ncols; ++c) {
        map[ps.record_cols[r.cols_begin + c].col] =
          column_map_t{sys.first_var[from_rows] + c, 1., 0.};
      }
    }
  }
  for (const composed_row_t& row : composed) {
    sys.var.insert(sys.var.end(), row.var.begin(), row.var.end());
    sys.value.insert(sys.value.end(), row.value.begin(), row.value.end());
    sys.start.push_back(sys.var.size());
    sys.low.push_back(row.low);
    sys.high.push_back(row.high);
    sys.row.push_back(row.row);
  }

  // The events of the rows, in the order of the stack.
  for (int32_t i = 0; i < model.nrows; ++i) {
    events_of[i + 1] += events_of[i];
  }
  sys.first = events_of;
  sys.events.resize(events_of[model.nrows]);
  std::vector<int32_t> fill(events_of.begin(), events_of.end() - 1);
  parallel = addition = from_rows = 0;
  for (size_t k = 0; k < ps.order.size(); ++k) {
    if (ps.order[k] == gprs::record_kind_t::parallel_row) {
      const gprs::parallel_row_record_t& r = ps.parallel_rows[parallel++];
      sys.events[fill[r.removed_row]++]    = row_event_t{k, row_event_t::parallel, r.kept_row};
    } else if (ps.order[k] == gprs::record_kind_t::row_addition) {
      const gprs::row_addition_record_t& r = ps.row_additions[addition++];
      sys.events[fill[r.row]++]            = row_event_t{k, row_event_t::addition, r.equation_row};
    } else if (ps.order[k] == gprs::record_kind_t::col_from_rows) {
      const gprs::col_from_rows_record_t& r = ps.col_from_rows[from_rows];
      for (int32_t n = r.rows_begin; n < r.rows_end; ++n) {
        int32_t anchor = sys.first_anchor[from_rows] + (n - r.rows_begin);
        sys.events[fill[ps.saved_rows[n].row]++] = row_event_t{k, row_event_t::saved, anchor};
      }
      ++from_rows;
    }
  }
  return true;
}

