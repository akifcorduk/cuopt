"""forcing.py <engine tag>: HiGHS's forcing columns (references m1_b and d3_cheap) next to the
engine's, per instance where either counts one, with the sizes."""
import json
import sys

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"
eng = {i["name"]: i for i in json.load(open(S + "/bench/" + sys.argv[1] + "/summary.json"))["instances"]}


def ref(step):
    out = {}
    for i in json.load(open(W + "/results/bench240/stage1c_refs/" + step + "/summary.json"))["instances"]:
        for r in i["diag"]["reductions"]:
            if r["stage"] == "all":
                out[i["name"]] = (r["forcing_column"], r["forcing_column_removed_row"],
                                  i["nnz_after"], i["rows_after"], i["cols_after"])
    return out


b, c = ref("m1_b"), ref("d3_cheap")
for n in sorted(eng):
    e = eng[n]
    r4 = e["sub_rules"]["forcing_column"]
    if r4 or b[n][0] or c[n][0]:
        rows = e.get("col_from_rows", {}).get("forcing_rows", 0)
        print("%-24s engine R4 %4d cols %5d rows, sizes %8d %7d %6d | m1_b %4d cols %5d rows, sizes %8d %7d %6d | all cheap %4d cols %5d rows, sizes %8d %7d %6d"
              % (n, r4, rows, e["reduced"]["nnz"], e["reduced"]["rows"], e["reduced"]["cols"], *b[n], *c[n]))
