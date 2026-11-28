"""highs_kinds.py <log of highs_run.py> [engine tag]: sums of HiGHS's free column substitutions
by pivot kind over the instances of a local run, whether the sizes are those of the stored m1_a
reference, and per instance the engine's own counts next to them."""
import json
import sys

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"
ref = {i["name"]: i for i in json.load(open(W + "/results/bench240/stage1c_refs/m1_a/summary.json"))["instances"]}
eng = {}
if len(sys.argv) > 2:
    eng = {i["name"]: i for i in json.load(open(S + "/bench/" + sys.argv[2] + "/summary.json"))["instances"]}
rows = []
cur = None
for line in open(sys.argv[1]):
    if not line.startswith("   "):
        w = line.split()
        cur = {"name": w[0], "nnz": int(w[1]), "rows": int(w[2]), "cols": int(w[3])}
        rows.append(cur)
    elif line.strip().startswith("name=singleton_col"):
        cur.update({k: int(v) for k, v in (x.split("=") for x in line.split()[1:])})
tot = {}
differ = 0
for r in rows:
    h = ref[r["name"]]
    if (r["nnz"], r["rows"], r["cols"]) != (h["nnz_after"], h["rows_after"], h["cols_after"]):
        differ += 1
        print("size differs from the stored reference:", r["name"], r["nnz"], r["rows"], r["cols"],
              h["nnz_after"], h["rows_after"], h["cols_after"])
    for k in ("count", "equation", "inequality_own", "inequality_other", "upper_side", "ranged",
              "own_zero_cost", "integer"):
        tot[k] = tot.get(k, 0) + r.get(k, 0)
print("%d instances, %d with sizes other than the stored m1_a reference" % (len(rows), differ))
print("free column substitutions of HiGHS by pivot kind:", tot)
print("instances with inequality_other > 0:")
for r in rows:
    if r.get("inequality_other", 0):
        e = eng.get(r["name"])
        es = ""
        if e:
            es = " | engine R2 %d R5 %d sizes %d %d %d" % (
                e["sub_rules"]["free_singleton"], e["sub_rules"]["free_singleton_dual"],
                e["reduced"]["nnz"], e["reduced"]["rows"], e["reduced"]["cols"])
        print("  %-28s eq %d own %d other %d  HiGHS sizes %d %d %d%s"
              % (r["name"], r["equation"], r["inequality_own"], r["inequality_other"], r["nnz"],
                 r["rows"], r["cols"], es))
print("instances where HiGHS's inequality_own differs from the engine's R5 count, or equation from R2:")
for r in rows:
    e = eng.get(r["name"])
    if not e:
        continue
    r2, r5 = e["sub_rules"]["free_singleton"], e["sub_rules"]["free_singleton_dual"]
    if r.get("equation", 0) != r2 or r.get("inequality_own", 0) + r.get("inequality_other", 0) != r5:
        print("  %-28s HiGHS eq %5d own %5d other %4d integer %5d | engine R2 %5d R5 %5d"
              % (r["name"], r.get("equation", 0), r.get("inequality_own", 0),
                 r.get("inequality_other", 0), r.get("integer", 0), r2, r5))
