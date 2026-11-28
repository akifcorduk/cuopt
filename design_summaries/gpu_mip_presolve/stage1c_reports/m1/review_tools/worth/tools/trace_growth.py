"""trace_growth.py <instance>...: the growth-limit refusals of the instrumented engine on an
instance, from its trace: distinct columns, their types, bounds, ratios, and the type and range
of the column with the largest other coefficient of the pivot row."""
import collections
import subprocess
import sys

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
TOOL = SCRATCH + "/s1c_m1_worth/bin/gprs_presolve_inst"
for name in sys.argv[1:]:
    p = subprocess.run([TOOL, SCRATCH + "/csr_dumps_bench/" + name + ".csr"], capture_output=True,
                       text=True, env={"GPRS_TRACE": "1"})
    cols = {}
    big = None
    for line in p.stderr.splitlines():
        if line.startswith("TRACEG"):
            big = dict(w.split("=") for w in line.split()[1:])
        elif line.startswith("TRACE ") and "cause=growth_row" in line:
            f = dict(w.split("=") for w in line.split()[1:])
            if f["stage"] != "0":
                continue
            f.update(big)
            cols[f["col"]] = f
    print("%s: %d columns refused by the ratio of the pivot row" % (name, len(cols)))
    if not cols:
        continue
    ratios = sorted(float(f["x"]) for f in cols.values())
    n = len(ratios)
    print("  ratio: min %.4g median %.4g 90%% %.4g max %.4g; at most 2000: %d, 1e4: %d, 1e5: %d, 1e6: %d"
          % (ratios[0], ratios[n // 2], ratios[int(0.9 * (n - 1))], ratios[-1],
             sum(r <= 2000 for r in ratios), sum(r <= 1e4 for r in ratios),
             sum(r <= 1e5 for r in ratios), sum(r <= 1e6 for r in ratios)))
    kinds = collections.Counter()
    for f in cols.values():
        kinds[("integer" if f["int"] == "1" else "continuous",
               "equation" if f["eq"] == "1" else "inequality",
               "largest other: " + ("integer" if f["big_int"] == "1" else "continuous"),
               "other integers %s" % ("yes" if f["others_int"] != "0" else "no"),
               "other continuous %s" % ("yes" if f["others_cont"] != "0" else "no"))] += 1
    for k, v in kinds.most_common():
        print("  %6d  %s" % (v, ", ".join(k)))
    shown = 0
    for f in cols.values():
        if shown >= 4:
            break
        shown += 1
        print("    e.g. col %s a=%s cost=%s bounds [%s, %s] row free=%s sides [%s, %s] largest=%s second=%s ratio=%s big col int=%s [%s, %s]"
              % (f["col"], f["a"], f["cost"], f["lower"], f["upper"], f["free"], f["rowL"],
                 f["rowU"], f["largest"], f["second"], f["x"], f["big_int"], f["big_lower"],
                 f["big_upper"]))
