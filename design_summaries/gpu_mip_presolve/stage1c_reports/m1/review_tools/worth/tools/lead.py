"""lead.py: the instances where the engine (default run) leaves less than the m1_a reference in
a measure, and what gives the lead: the engine before the milestone already (Stage 1), or the
rule whose switch takes the lead away (runs no_r2, no_r3, no_r4, no_r5)."""
import json

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"


def engine(tag):
    d = json.load(open(S + "/bench/" + tag + "/summary.json"))
    return {i["name"]: (i["reduced"]["nnz"], i["reduced"]["rows"], i["reduced"]["cols"]) for i in d["instances"]}


ref = {i["name"]: (i["nnz_after"], i["rows_after"], i["cols_after"])
       for i in json.load(open(W + "/results/bench240/stage1c_refs/m1_a/summary.json"))["instances"]}
now, base = engine("default"), engine("base_default")
off = {r: engine("no_" + r) for r in ("r2", "r3", "r4", "r5")}
names = {"r2": "free singleton, equation (R2)", "r3": "zero-cost singleton (R3)",
         "r4": "forcing column (R4)", "r5": "free singleton, inequality (R5)"}


def lead(a, b):
    """a ahead of b: less in a measure and more in none."""
    return any(a[k] < b[k] for k in range(3))


count = {}
for n in sorted(now):
    if not lead(now[n], ref[n]):
        continue
    causes = []
    if lead(base[n], ref[n]):
        same = base[n] == now[n]
        causes.append("Stage 1 before the milestone" + ("" if same else " (and more with it)"))
    for r in ("r2", "r3", "r4", "r5"):
        if off[r][n] != now[n]:
            gone = not lead(off[r][n], ref[n])
            causes.append("%s%s" % (names[r], " [lead gone without it]" if gone else ""))
    for c in causes:
        key = c.split(" [")[0].split(" (and")[0]
        count[key] = count.get(key, 0) + 1
    print("%-28s engine %8d %7d %7d  m1_a %8d %7d %7d  before %8d %7d %7d | %s"
          % (n, *now[n], *ref[n], *base[n], "; ".join(causes)))
print(count)
