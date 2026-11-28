"""cmp.py ref <engine tag> <reference step or path> [--list]
   cmp.py two <engine tag a> <engine tag b> [--list]
Own comparison of engine summaries (bench/<tag>/summary.json) with a HiGHS reference summary or
with another engine summary: geometric means of other / engine over the 237 instances without
ex9, ex10, enlight_hard, and the instances that differ."""
import json
import math
import os
import sys

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"
LEFT = {"ex9", "ex10", "enlight_hard"}


def engine(tag):
    path = tag if os.path.isabs(tag) else S + "/bench/" + tag
    d = json.load(open(path + "/summary.json"))
    out = {}
    for i in d["instances"]:
        r = i["reduced"]
        out[i["name"]] = (r["nnz"], r["rows"], r["cols"], i)
    return out


def reference(step):
    path = step if os.path.isabs(step) else W + "/results/bench240/stage1c_refs/" + step
    if not os.path.exists(path):
        path = W + "/results/bench240/" + step
    d = json.load(open(path + "/summary.json"))
    out = {}
    for i in d["instances"]:
        out[i["name"]] = (i["nnz_after"], i["rows_after"], i["cols_after"], i)
    return out


def means(a, b, names):
    res = []
    for k in range(3):
        s = 0.0
        for n in names:
            s += math.log(max(b[n][k], 1) / max(a[n][k], 1))
        res.append(math.exp(s / len(names)))
    return res


def main():
    mode = sys.argv[1]
    a = engine(sys.argv[2])
    b = reference(sys.argv[3]) if mode == "ref" else engine(sys.argv[3])
    names = sorted(n for n in a if n in b and n not in LEFT)
    m = means(a, b, names)
    print("%s / %s over %d: nnz %.4f rows %.4f cols %.4f" % (sys.argv[3], sys.argv[2], len(names), *m))
    capped = []
    for k in range(3):
        s = sum(min(0.0, math.log(max(b[n][k], 1) / max(a[n][k], 1))) for n in names)
        capped.append(math.exp(s / len(names)))
    print("  lead counted as zero: nnz %.4f rows %.4f cols %.4f" % tuple(capped))
    more = [n for n in sorted(a) if n in b and any(a[n][k] > b[n][k] for k in range(3))]
    less = [n for n in sorted(a) if n in b and any(a[n][k] < b[n][k] for k in range(3))]
    below99 = [n for n in more if any(b[n][k] / max(a[n][k], 1) < 0.99 for k in range(3))]
    print("  first leaves more in a measure: %d; less in a measure: %d; below 0.99 in a measure: %d"
          % (len(more), len(less), len(below99)))
    sums = [sum(a[n][k] for n in a) for k in range(3)]
    print("  sums of first over %d: nnz %d rows %d cols %d" % (len(a), *sums))
    if "--list" in sys.argv:
        for title, group in (("MORE", more), ("LESS", less)):
            print(title)
            for n in group:
                print("  %-34s first %9d %8d %8d  other %9d %8d %8d  ratio %.4f %.4f %.4f"
                      % (n, *a[n][:3], *b[n][:3],
                         *[b[n][k] / max(a[n][k], 1) for k in range(3)]))


main()
