"""cmp17.py: the engine on the 17 local dumps (bench17/, default mode) against the testbed
reference of step m1_a (results/stage1c/refs/tb_m1_a_dev) and all cheap rules (cheap_dev),
where the reference has the instance."""
import glob
import json
import math
import os

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"
for step in ("tb_m1_a_dev", "cheap_dev"):
    ref = {i["name"]: i for i in json.load(open(W + "/results/stage1c/refs/" + step + "/summary.json"))["instances"]}
    logs = [0.0] * 3
    logs_b = [0.0] * 3
    n = 0
    print(step)
    for path in sorted(glob.glob(S + "/bench17/*.m1.default.json")):
        name = os.path.basename(path).split(".")[0]
        if name not in ref or "nnz_after" not in ref[name]:
            print("   %-24s not in the reference" % name)
            continue
        e = json.load(open(path))["reduced"]
        b = json.load(open(path.replace(".m1.", ".base.")))["reduced"]
        h = ref[name]
        r = (h["nnz_after"] / max(e["nnz"], 1), h["rows_after"] / max(e["rows"], 1), h["cols_after"] / max(e["cols"], 1))
        rb = (h["nnz_after"] / max(b["nnz"], 1), h["rows_after"] / max(b["rows"], 1), h["cols_after"] / max(b["cols"], 1))
        for k in range(3):
            logs[k] += math.log(r[k])
            logs_b[k] += math.log(rb[k])
        n += 1
        print("   %-24s engine %8d %7d %7d  reference %8d %7d %7d  ratio %.4f %.4f %.4f  (before %.4f %.4f %.4f)"
              % (name, e["nnz"], e["rows"], e["cols"], h["nnz_after"], h["rows_after"], h["cols_after"], *r, *rb))
    print("   geometric mean over %d: %.4f %.4f %.4f (before %.4f %.4f %.4f)"
          % (n, *[math.exp(x / n) for x in logs], *[math.exp(x / n) for x in logs_b]))
