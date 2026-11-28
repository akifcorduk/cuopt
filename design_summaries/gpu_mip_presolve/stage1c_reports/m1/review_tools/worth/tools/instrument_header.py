"""instrument_header.py <include/gprs/presolve.hpp> <output>: measurement only. With
GPRS_X_PRIMAL_R3R4 set, the zero-cost singleton columns and the forcing column are tried in
primal-only mode as well (decision 4 of the design; both are exact projections)."""
import sys

src = open(sys.argv[1]).read()
old = """  bool primal = sub == sub_rule::free_singleton || sub == sub_rule::parallel_merge;"""
new = """  bool primal = sub == sub_rule::free_singleton || sub == sub_rule::parallel_merge ||
                ((sub == sub_rule::zero_cost_singleton || sub == sub_rule::forcing_column) &&
                 std::getenv("GPRS_X_PRIMAL_R3R4") != nullptr);"""
assert src.count(old) == 1
src = src.replace(old, new)
old = "#include <array>\n"
assert src.count(old) == 1
src = src.replace(old, "#include <array>\n#include <cstdlib>\n")
open(sys.argv[2], "w").write(src)
