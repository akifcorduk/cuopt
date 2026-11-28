"""work17.py: the work units per family of two gprs_presolve binaries on the 17 local dumps."""

import glob
import json
import subprocess

S = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity"
D = "/Users/acoerduek/Documents/workspace/cuopt-wt/gpu-presolve/gpu_presolve/local_dumps"
tot = {"commit": {}, "final": {}}
for dump in sorted(glob.glob(D + "/*.csr")):
    for name, b in (("commit", "build_commit"), ("final", "build_final")):
        subprocess.run(
            [S + "/" + b + "/release/gprs_presolve", dump, "--json", S + "/logs/t.json"],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        r = json.load(open(S + "/logs/t.json"))
        for rule, s in r["rules"].items():
            tot[name][rule] = tot[name].get(rule, 0) + s["work"]
for rule in tot["commit"]:
    same = "same" if tot["commit"][rule] == tot["final"][rule] else "DIFFERENT"
    print(rule, tot["commit"][rule], tot["final"][rule], same)
print("all", sum(tot["commit"].values()), sum(tot["final"].values()))
