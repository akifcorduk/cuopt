"""run17.py: the base engine and theirs on the 17 local dumps, default and primal-only mode;
one line per instance with the sizes of each, findings of the family, rounds and flags."""
import glob
import json
import os
import subprocess
import sys

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
D = "/Users/acoerduek/Documents/workspace/cuopt-wt/gpu-presolve/gpu_presolve/local_dumps"
tools = {"base": S + "/bin/gprs_presolve_base", "m1": S + "/bin/gprs_presolve_m1"}
if len(sys.argv) > 1:
    tools["m1"] = sys.argv[1]
out = S + "/bench17"
os.makedirs(out, exist_ok=True)
dumps = sorted(glob.glob(D + "/*.csr"))
print("%d dumps" % len(dumps))
tot = {}
for mode, args in (("default", []), ("primal", ["--primal-only"])):
    print("mode %s" % mode)
    larger = 0
    smaller = 0
    for dump in dumps:
        name = os.path.basename(dump)[:-4]
        res = {}
        for tag, tool in tools.items():
            js = "%s/%s.%s.%s.json" % (out, name, tag, mode)
            subprocess.run([tool, dump, "--json", js] + args, stdout=subprocess.DEVNULL, check=True)
            res[tag] = json.load(open(js))
        b, m = res["base"], res["m1"]
        rb, rm = b["reduced"], m["reduced"]
        for k in ("nnz", "rows", "cols"):
            tot[(mode, "base", k)] = tot.get((mode, "base", k), 0) + rb[k]
            tot[(mode, "m1", k)] = tot.get((mode, "m1", k), 0) + rm[k]
        worse = any(rm[k] > rb[k] for k in ("nnz", "rows", "cols"))
        better = any(rm[k] < rb[k] for k in ("nnz", "rows", "cols"))
        larger += worse
        smaller += better
        sub = m["sub_rules"]
        print("  %-26s base %8d %7d %7d | m1 %8d %7d %7d | R2 %5d R5 %4d R3 %5d R4 %3d | rounds %d/%d fp %s/%s dual %d %s"
              % (name, rb["nnz"], rb["rows"], rb["cols"], rm["nnz"], rm["rows"], rm["cols"],
                 sub["free_singleton"], sub["free_singleton_dual"], sub["zero_cost_singleton"],
                 sub["forcing_column"], b["rounds"], m["rounds"], b["fixed_point"],
                 m["fixed_point"], m["dual_findings"], "LARGER" if worse else ""))
    print("  sums base %d %d %d | m1 %d %d %d; smaller on %d, larger in a measure on %d"
          % (tot[(mode, "base", "nnz")], tot[(mode, "base", "rows")], tot[(mode, "base", "cols")],
             tot[(mode, "m1", "nnz")], tot[(mode, "m1", "rows")], tot[(mode, "m1", "cols")],
             smaller, larger))
