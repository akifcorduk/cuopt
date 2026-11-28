"""instrument_tool.py <tools/gprs_presolve.cpp> <output>: the engine tool with the counters of
diag.hpp in its report ("diag": the presolve; "left": the first call of the family in a
presolve of the reduced model, which serves every column once, so its refusals are the
singleton columns that stay, by cause; "second": status and sizes of that second presolve)."""
import sys

src = open(sys.argv[1]).read()


def rep(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old)
    src = src.replace(old, new)


rep('#include <gprs/io.hpp>\n', '#include "../src/diag.hpp"\n\n#include <gprs/io.hpp>\n')
rep("""void write_report(FILE* out,
                  const std::string& path,
                  const gprs::model_t& model,
                  const gprs::result_t& result)
{""", """std::string second_text;

void write_report(FILE* out,
                  const std::string& path,
                  const gprs::model_t& model,
                  const gprs::result_t& result)
{""")
rep("""  std::fprintf(out, "  }\\n}\\n");""", """  std::fprintf(out, "  },\\n  \\"diag\\": ");
  gprs_diag::write(out, 0);
  std::fprintf(out, ",\\n  \\"left\\": ");
  gprs_diag::write(out, 1);
  std::fprintf(out, ",\\n  \\"second\\": %s\\n}\\n", second_text.c_str());""")
rep("""  gprs::result_t result = gprs::presolve(model, options);
""", """  gprs::result_t result = gprs::presolve(model, options);
  {
    gprs_diag::data().stage = 1;
    gprs::result_t again    = gprs::presolve(result.reduced, options);
    gprs_diag::data().stage = 2;
    char text[256];
    std::snprintf(text, sizeof text,
                  "{\\"status\\": \\"%s\\", \\"rows\\": %d, \\"cols\\": %d, \\"nnz\\": %lld}",
                  gprs::status_name(again.status), again.reduced.nrows, again.reduced.ncols,
                  (long long)again.reduced.nnz());
    second_text = text;
  }
""")
open(sys.argv[2], "w").write(src)
