"""href.py <instance> [reference]...: sizes and the main counters of an instance in stored HiGHS
references (default: highs_stage1c, stage1c_refs/m1_a, stage1c_refs/m1_b, stage1c_refs/d3_cheap)."""
import json
import sys

W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"
name = sys.argv[1]
refs = sys.argv[2:] or ["highs_stage1c", "stage1c_refs/m1_a", "stage1c_refs/m1_b", "stage1c_refs/d3_cheap"]
for ref in refs:
    d = json.load(open(W + "/results/bench240/" + ref + "/summary.json"))
    for i in d["instances"]:
        if i["name"] != name:
            continue
        print("%s %s: %d nnz %d rows %d cols -> %d nnz %d rows %d cols (%s)"
              % (name, ref, i["nnz"], i["rows"], i["cols"], i["nnz_after"], i["rows_after"],
                 i["cols_after"], i["status"]))
        diag = i.get("diag", {})
        for r in diag.get("reductions", []):
            if r["stage"] == "all":
                print("   reductions:", {k: v for k, v in r.items() if v and k != "stage"})
        for key in ("fixing", "bound_change", "parallel", "doubleton", "free_col_substitution"):
            v = diag.get(key)
            if v:
                if isinstance(v, list):
                    v = v[0]
                if isinstance(v, dict):
                    v = {k: x for k, x in v.items() if x and not k.endswith("seconds")}
                print("   %s: %s" % (key, v))
