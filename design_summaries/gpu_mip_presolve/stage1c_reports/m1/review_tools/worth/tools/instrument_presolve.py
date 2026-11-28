"""instrument_presolve.py <src/presolve.cpp> <output>: measurement only. With GPRS_X_ORDER=1
the singleton-column family is a phase right after the cascade, before the parallel rows and
the doubleton equations, instead of after simple probing."""
import sys

src = open(sys.argv[1]).read()


def rep(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old)
    src = src.replace(old, new)


rep('#include <gprs/presolve.hpp>\n', '#include "diag.hpp"\n\n#include <gprs/presolve.hpp>\n')
rep("""      bool served = !cascade_pending();
      end_phase();
      if (!call(rule::parallel_rows, parallel_rows)) { return; }""",
    """      bool served = !cascade_pending();
      end_phase();
      bool early = gprs_diag::data().order == 1;
      if (early && !column_phase(rule::singleton_cols, singleton_cols)) { return; }
      if (!call(rule::parallel_rows, parallel_rows)) { return; }""")
rep("""          !column_phase(rule::singleton_cols, singleton_cols) ||""",
    """          (!early && !column_phase(rule::singleton_cols, singleton_cols)) ||""")
open(sys.argv[2], "w").write(src)
