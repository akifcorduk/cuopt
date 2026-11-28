"""precise.py <tag a> <tag b>: geometric mean of b / a over 237 and over 240 with six digits."""
import json
import math
import sys

S = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth"
LEFT = {"ex9", "ex10", "enlight_hard"}


def load(tag):
    d = json.load(open(S + "/bench/" + tag + "/summary.json"))
    return {i["name"]: (i["reduced"]["nnz"], i["reduced"]["rows"], i["reduced"]["cols"]) for i in d["instances"]}


a, b = load(sys.argv[1]), load(sys.argv[2])
for names in (sorted(n for n in a if n not in LEFT), sorted(a)):
    print(len(names), ["%.6f" % math.exp(sum(math.log(max(b[n][k], 1) / max(a[n][k], 1)) for n in names) / len(names)) for k in range(3)])
