"""counts.py <engine tag> <reference step>: per instance, the engine's findings of the
singleton-column rules next to HiGHS's counters of the matching rules, where they differ."""
import json
import os
import sys

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"

tag, step = sys.argv[1], sys.argv[2]
eng = {i["name"]: i for i in json.load(open(S + "/bench/" + tag + "/summary.json"))["instances"]}
ref = {i["name"]: i for i in json.load(open(W + "/results/bench240/stage1c_refs/" + step + "/summary.json"))["instances"]}


def red(i, key):
    for r in i["diag"]["reductions"]:
        if r["stage"] == "all":
            return r.get(key, 0)
    return 0


def fixing(i, name):
    f = i["diag"].get("fixing")
    if not f:
        return 0
    if isinstance(f, list):
        f = f[0]
    v = f.get(name, 0)
    return v.get("count", 0) if isinstance(v, dict) else v


print("%-30s %8s %8s | %8s %8s | %8s %8s | %7s %7s | sizes eng / ref (nnz rows cols)"
      % ("instance", "eR2+R5", "hFCS", "eR3cols", "hZCS", "eDualFx", "hDualFx", "eDbl", "hDbl"))
tot = [0] * 8
for n in sorted(eng):
    e, h = eng[n], ref[n]
    sub = e["sub_rules"]
    e_fs = sub["free_singleton"] + sub["free_singleton_dual"]
    h_fs = red(h, "free_col_substitution")
    e_z = sub["zero_cost_singleton"]
    h_z = red(h, "zero_obj_singleton_col")
    e_df = e["rules"]["dual_fixing"]["findings"]
    h_df = fixing(h, "dual_fixing_lock") + fixing(h, "empty_col")
    e_db = e["rules"]["doubleton"]["findings"]
    h_db = h["diag"]["doubleton"][0]["removed"]
    for k, v in enumerate((e_fs, h_fs, e_z, h_z, e_df, h_df, e_db, h_db)):
        tot[k] += v
    if e_fs != h_fs or e_z != h_z:
        r = e["reduced"]
        print("%-30s %8d %8d | %8d %8d | %8d %8d | %7d %7d | %d %d %d / %d %d %d"
              % (n, e_fs, h_fs, e_z, h_z, e_df, h_df, e_db, h_db, r["nnz"], r["rows"], r["cols"],
                 h["nnz_after"], h["rows_after"], h["cols_after"]))
print("%-30s %8d %8d | %8d %8d | %8d %8d | %7d %7d" % ("total", *tot))
