"""gain.py <engine before tag> <engine now tag> [threshold]: what the singleton-column rules
gain on each side. HiGHS: results/bench240/highs_stage1c (our Stage 1 rule set) against m1_a.
Engine: before against now. Lists the instances where HiGHS's gain is larger than the engine's
by more than the threshold (default 0.5%) in a measure, and the means of the gains."""
import json
import math
import sys

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"
LEFT = {"ex9", "ex10", "enlight_hard"}


def engine(tag):
    d = json.load(open(S + "/bench/" + tag + "/summary.json"))
    return {i["name"]: (i["reduced"]["nnz"], i["reduced"]["rows"], i["reduced"]["cols"]) for i in d["instances"]}


def highs(path):
    d = json.load(open(W + "/results/bench240/" + path + "/summary.json"))
    return {i["name"]: (i["nnz_after"], i["rows_after"], i["cols_after"]) for i in d["instances"]}


eb, en = engine(sys.argv[1]), engine(sys.argv[2])
hb, hn = highs("highs_stage1c"), highs("stage1c_refs/m1_a")
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
names = sorted(n for n in en if n not in LEFT and n in hb and n in hn)
sums_e, sums_h = [0.0] * 3, [0.0] * 3
rows = []
for n in names:
    ge = [math.log(max(eb[n][k], 1) / max(en[n][k], 1)) for k in range(3)]
    gh = [math.log(max(hb[n][k], 1) / max(hn[n][k], 1)) for k in range(3)]
    for k in range(3):
        sums_e[k] += ge[k]
        sums_h[k] += gh[k]
    if any(gh[k] - ge[k] > thr for k in range(3)):
        rows.append((n, ge, gh))
print("gain of the singleton-column rules over %d instances (geometric mean of before / after)" % len(names))
print("  engine: nnz %.4f rows %.4f cols %.4f" % tuple(math.exp(s / len(names)) for s in sums_e))
print("  HiGHS : nnz %.4f rows %.4f cols %.4f" % tuple(math.exp(s / len(names)) for s in sums_h))
print("instances where HiGHS gains more than the engine by over %.1f%% in a measure: %d" % (100 * thr, len(rows)))
for n, ge, gh in rows:
    print("  %-30s engine %.4f %.4f %.4f  HiGHS %.4f %.4f %.4f | engine %s -> %s | HiGHS %s -> %s"
          % (n, *[math.exp(g) for g in ge], *[math.exp(g) for g in gh], eb[n], en[n], hb[n], hn[n]))
