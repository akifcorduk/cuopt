"""diag_sum.py <engine tag> [diag|left] [top n]: the counters of the instrumented tool summed
over the instance reports of a run, with the number of instances above zero and the largest."""
import glob
import json
import os
import sys

S = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth"
tag = sys.argv[1]
part = sys.argv[2] if len(sys.argv) > 2 else "diag"
top = int(sys.argv[3]) if len(sys.argv) > 3 else 4
total, by = {}, {}
second = {}
n = 0
for path in sorted(glob.glob(S + "/bench/" + tag + "/*.rep0.json")):
    if path.endswith(".run.json"):
        continue
    r = json.load(open(path))
    name = os.path.basename(path)[: -len(".rep0.json")]
    n += 1
    for k, v in r[part].items():
        total[k] = total.get(k, 0) + v
        if v:
            by.setdefault(k, []).append((v, name))
    s = r["second"]
    if s["status"] != "unchanged":
        second[name] = (s, r["reduced"], r["fixed_point"])
print("%s, %s, %d instances" % (tag, part, n))
for k, v in total.items():
    items = sorted(by.get(k, []), reverse=True)
    print("  %-32s %10d on %3d: %s" % (k, v, len(items),
                                       ", ".join("%s %d" % (name, x) for x, name in items[:top])))
print("second presolve not 'unchanged' on %d instances" % len(second))
for name, (s, red, fp) in sorted(second.items()):
    print("  %-30s fixed_point %s  first %d/%d/%d  second %s %d/%d/%d"
          % (name, fp, red["nnz"], red["rows"], red["cols"], s["status"], s["nnz"], s["rows"], s["cols"]))
