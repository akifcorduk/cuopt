"""hash_cmp.py <tag a> <tag b>: reduced-model hash, rounds, rebuilds and fixed-point flag of two
engine runs, instance by instance."""
import json
import os
import sys

S = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth"


def load(tag):
    path = tag if os.path.isabs(tag) else S + "/bench/" + tag
    return {i["name"]: i for i in json.load(open(path + "/summary.json"))["instances"]}


a, b = load(sys.argv[1]), load(sys.argv[2])
diff = []
for n in sorted(a):
    x, y = a[n], b[n]
    for k in ("hash", "rounds", "rebuilds", "fixed_point", "dual_findings"):
        if x.get(k) != y.get(k):
            diff.append((n, k, x.get(k), y.get(k)))
print("%s against %s: %d instances, %d differences" % (sys.argv[1], sys.argv[2], len(a), len(diff)))
for d in diff[:20]:
    print("  ", d)
