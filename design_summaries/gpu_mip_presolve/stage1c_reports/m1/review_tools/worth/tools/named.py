"""named.py <engine tag>: the instances that milestone M1 names, one by one: the engine against
the references m1_a, m1_b, d3_cheap and against the engine before, with the findings per rule."""
import json
import sys

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"
NAMES = ["neos-3402294-bobin", "mcsched", "swath1", "swath3", "neos-3627168-kasai", "blp-ar98",
         "blp-ic98", "leo1", "leo2", "germanrr", "sct2", "supportcase7", "irish-electricity",
         "neos-4647030-tutaki", "timtab1", "icir97_tension", "neos-4954672-berkel", "pg",
         "supportcase12", "neos-2746589-doon", "neos-3402454-bohle", "neos-3656078-kumeu",
         "neos-860300", "traininstance2"]
tag = sys.argv[1]
eng = {i["name"]: i for i in json.load(open(S + "/bench/" + tag + "/summary.json"))["instances"]}
base = {i["name"]: i for i in json.load(open(S + "/bench/base_default/summary.json"))["instances"]}


def ref(step):
    d = json.load(open(W + "/results/bench240/stage1c_refs/" + step + "/summary.json"))
    return {i["name"]: (i["nnz_after"], i["rows_after"], i["cols_after"]) for i in d["instances"]}


refs = {s: ref(s) for s in ("m1_a", "m1_b", "d3_cheap")}
for n in NAMES:
    e = eng[n]
    r = (e["reduced"]["nnz"], e["reduced"]["rows"], e["reduced"]["cols"])
    b = (base[n]["reduced"]["nnz"], base[n]["reduced"]["rows"], base[n]["reduced"]["cols"])
    a = refs["m1_a"][n]
    s = e["sub_rules"]
    verdict = "equal" if r == a else ("ahead" if all(r[k] <= a[k] for k in range(3)) else
                                      "behind" if all(r[k] >= a[k] for k in range(3)) else "mixed")
    print("%-22s engine %8d %7d %7d | m1_a %8d %7d %7d (%.4f %.4f %.4f) %-6s | m1_b %8d %7d %7d | all cheap %8d %7d %7d | before %8d %7d %7d | R2 %d R5 %d R3 %d R4 %d"
          % (n, *r, *a, *[a[k] / max(r[k], 1) for k in range(3)], verdict, *refs["m1_b"][n],
             *refs["d3_cheap"][n], *b, s["free_singleton"], s["free_singleton_dual"],
             s["zero_cost_singleton"], s["forcing_column"]))
