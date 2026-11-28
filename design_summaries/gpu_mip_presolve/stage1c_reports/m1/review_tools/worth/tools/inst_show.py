"""inst_show.py <tag> <instance>...: the nonzero counters of the instrumented tool on instances."""
import json
import sys

S = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth"
tag = sys.argv[1]
for n in sys.argv[2:]:
    r = json.load(open(S + "/bench/%s/%s.rep0.json" % (tag, n)))
    print(n, r["original"], r["reduced"], "rounds", r["rounds"], "fixed_point", r["fixed_point"])
    print("   family:", r["rules"]["singleton_cols"])
    print("   sub_rules:", {k: v for k, v in r["sub_rules"].items() if v})
    print("   columns:", r["columns"])
    print("   diag:", {k: v for k, v in r["diag"].items() if v})
    print("   left:", {k: v for k, v in r["left"].items() if v})
    print("   second:", r["second"])
