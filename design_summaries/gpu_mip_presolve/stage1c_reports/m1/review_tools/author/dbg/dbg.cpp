// Debug driver: one generated case by profile option, seed and variant, printed with its
// presolve result, stack and structure report.
#include "../../../../../../Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-1/gpu_presolve/check/structure_check.hpp"
#include "oracle.hpp"

#include <cstdio>

using namespace gprs;

#include "case.inc"

int main()
{
  oracle::case_t c = make();
  options_t options = make_options();
  result_t r        = presolve(c.model, options);
  std::printf("status %s rounds %d fixed_point %d dual %lld\n",
              status_name(r.status),
              r.rounds,
              (int)r.fixed_point,
              (long long)r.dual_findings);
  const postsolve_t& ps = r.postsolve;
  size_t f = 0, a = 0, p = 0, d = 0, cf = 0;
  for (size_t k = 0; k < ps.order.size(); ++k) {
    switch (ps.order[k]) {
      case record_kind_t::fixed_col:
        std::printf("%zu fixed x%d = %.17g\n", k, ps.fixed_cols[f].col, ps.fixed_cols[f].value);
        ++f;
        break;
      case record_kind_t::alias:
        std::printf("%zu alias x%d = %.17g x%d + %.17g (row %d)\n",
                    k,
                    ps.aliases[a].col,
                    ps.aliases[a].scale,
                    ps.aliases[a].rep,
                    ps.aliases[a].offset,
                    ps.aliases[a].row);
        ++a;
        break;
      case record_kind_t::parallel_row:
        std::printf("%zu parallel row %d into %d ratio %.17g\n",
                    k,
                    ps.parallel_rows[p].removed_row,
                    ps.parallel_rows[p].kept_row,
                    ps.parallel_rows[p].ratio);
        ++p;
        break;
      case record_kind_t::row_addition:
        std::printf("%zu addition row %d -= %.17g * row %d\n",
                    k,
                    ps.row_additions[d].row,
                    ps.row_additions[d].factor,
                    ps.row_additions[d].equation_row);
        ++d;
        break;
      case record_kind_t::col_from_rows: {
        const col_from_rows_record_t& rec = ps.col_from_rows[cf++];
        std::printf("%zu col_from_rows kind %d dual %d\n", k, (int)rec.kind, (int)rec.dual);
        for (int32_t n = rec.cols_begin; n < rec.cols_end; ++n) {
          std::printf("    col x%d [%.17g, %.17g] prefer %.17g\n",
                      ps.record_cols[n].col,
                      ps.record_cols[n].lower,
                      ps.record_cols[n].upper,
                      ps.record_cols[n].prefer);
        }
        int32_t count = std::max(rec.cols_end - rec.cols_begin, rec.rows_end - rec.rows_begin);
        for (int32_t n = 0; n < count; ++n) {
          std::printf("    coefficient %.17g\n", ps.record_coefficients[rec.coefficients + n]);
        }
        for (int32_t n = rec.rows_begin; n < rec.rows_end; ++n) {
          const saved_row_t& row = ps.saved_rows[n];
          std::printf("    row %d [%.17g, %.17g]:", row.row, row.lower, row.upper);
          for (int64_t e = row.begin; e < row.end; ++e) {
            std::printf(" %.17g x%d", ps.saved_entries.value[e], ps.saved_entries.col[e]);
          }
          std::printf("\n");
        }
        break;
      }
    }
  }
  const model_t& m = r.reduced;
  std::printf("reduced %d rows %d cols offset %.17g\n", m.nrows, m.ncols, m.objective_offset);
  for (int32_t i = 0; i < m.nrows; ++i) {
    std::printf("  row %d (orig %d) [%.17g, %.17g]:", i, ps.row_origin[i], m.row_lower[i], m.row_upper[i]);
    for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
      std::printf(" %.17g x%d", m.value[e], ps.col_origin[m.col_index[e]]);
    }
    std::printf("\n");
  }
  for (int32_t j = 0; j < m.ncols; ++j) {
    std::printf("  x%d [%.17g, %.17g] cost %.17g type %d\n",
                ps.col_origin[j],
                m.col_lower[j],
                m.col_upper[j],
                m.cost[j],
                (int)m.col_type[j]);
  }
  if (gprs_check::carries_model(r.status)) {
    gprs_check::structure_report_t s = gprs_check::structure_check(c.model, r, options.primal_only);
    std::printf(
      "structure: stack '%s' bounds %lld rows %lld kept %lld added %lld box %lld integral %lld "
      "snapshot %lld objective %lld (worst %g col %d constant %g) postsolve %lld (worst %g col "
      "%d)\n",
      s.stack_error.c_str(),
      (long long)s.bound_failures,
      (long long)s.row_failures,
      (long long)s.kept_failures,
      (long long)s.added_failures,
      (long long)s.box_failures,
      (long long)s.integral_failures,
      (long long)s.snapshot_failures,
      (long long)s.objective_failures,
      s.objective_worst,
      s.objective_worst_col,
      s.objective_constant,
      (long long)s.postsolve_failures,
      s.postsolve_worst,
      s.postsolve_worst_col);
  }
  oracle::verdict_t v = oracle::check_result(c, options, r);
  std::printf("oracle: %s %s\n", v.ok ? "ok" : "FAIL", v.message.c_str());
  return 0;
}
