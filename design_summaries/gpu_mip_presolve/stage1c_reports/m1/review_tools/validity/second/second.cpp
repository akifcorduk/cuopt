// second <dump.csr>: the presolve of a dump in default and primal-only mode, and for a result
// that is a fixed point the presolve of its reduced model, which must return unchanged. Also
// prints the reduced-model hash and the sub-rule findings, for comparisons between engines.
#include <gprs/io.hpp>
#include <gprs/presolve.hpp>

#include <cstdio>
#include <cstring>
#include <string>

using namespace gprs;

int main(int argc, char** argv)
{
  if (argc < 2) { return 2; }
  model_t model;
  std::string error;
  if (!read_csr_dump(argv[1], model, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const char* base = std::strrchr(argv[1], '/');
  base             = base ? base + 1 : argv[1];
  int bad          = 0;
  for (int mode = 0; mode < 2; ++mode) {
    options_t o;
    o.primal_only = mode == 1;
    result_t r    = presolve(model, o);
    bool carries  = r.status == status_t::reduced || r.status == status_t::unchanged ||
                    r.status == status_t::reduced_to_empty || r.status == status_t::limit;
    const char* second = "-";
    long long more     = 0;
    if (carries && r.fixed_point) {
      result_t s = presolve(r.reduced, o);
      second     = status_name(s.status);
      more       = (long long)(r.reduced.nnz() - s.reduced.nnz());
      if (s.status != status_t::unchanged) { bad = 1; }
    }
    std::printf("%s %s %s hash %016llx stack %016llx fp %d rounds %d rows %d cols %d nnz %lld "
                "dual %lld fs %lld fsd %lld zs %lld fc %lld limited %lld/%lld/%lld second %s %lld\n",
                base,
                mode ? "primal" : "default",
                status_name(r.status),
                (unsigned long long)r.hash,
                (unsigned long long)postsolve_hash(r.postsolve),
                (int)r.fixed_point,
                r.rounds,
                r.reduced.nrows,
                r.reduced.ncols,
                (long long)r.reduced.nnz(),
                (long long)r.dual_findings,
                (long long)r.sub_findings[sub_rule::free_singleton],
                (long long)r.sub_findings[sub_rule::free_singleton_dual],
                (long long)r.sub_findings[sub_rule::zero_cost_singleton],
                (long long)r.sub_findings[sub_rule::forcing_column],
                (long long)r.columns.growth_refused,
                (long long)r.columns.growth_refused_row,
                (long long)r.columns.saved_row_refused,
                second,
                more);
  }
  return bad;
}
