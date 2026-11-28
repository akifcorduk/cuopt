"""work.py <engine tag> [top n]: work of the singleton-column family per instance in passes of
the input model (nonzeros + 8 * (rows + columns)), split into its first call and its later
calls; quantiles and the largest instances; the family's share of all work; rounds and flags
against a second run (default base_default)."""
import json
import sys

S = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth"
tag = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 8
other = sys.argv[3] if len(sys.argv) > 3 else "base_default"
d = json.load(open(S + "/bench/" + tag + "/summary.json"))["instances"]
b = {i["name"]: i for i in json.load(open(S + "/bench/" + other + "/summary.json"))["instances"]}
rows = []
tot_family = tot_all = tot_first = 0
for i in d:
    pass_ = i["nnz"] + 8 * (i["rows"] + i["cols"])
    f = i["rules"]["singleton_cols"]
    all_work = sum(r["work"] for r in i["rules"].values())
    tot_family += f["work"]
    tot_first += f["first_work"]
    tot_all += all_work
    rows.append((f["work"] / pass_, f["first_work"] / pass_, (f["work"] - f["first_work"]) / pass_,
                 i["name"], f["calls"], f["work"], f["seconds"], i["seconds"], all_work / pass_))


def q(values, p):
    v = sorted(values)
    return v[int(round(p * (len(v) - 1)))]


for k, title in ((0, "whole presolve"), (1, "first call"), (2, "later calls")):
    v = [r[k] for r in rows]
    best = max(rows, key=lambda r: r[k])
    print("family work in passes, %-14s median %.2f  90%% %.2f  largest %.2f (%s)"
          % (title + ":", q(v, 0.5), q(v, 0.9), best[k], best[3]))
print("family work units %d (first calls %d, later %d), all families %d, share %.1f%%"
      % (tot_family, tot_first, tot_family - tot_first, tot_all, 100 * tot_family / tot_all))
print("family seconds %.4f of %.4f" % (sum(r[6] for r in rows), sum(r[7] for r in rows)))
print("largest by whole-presolve work:")
for r in sorted(rows, reverse=True)[:top]:
    print("  %-30s %.2f passes (first %.2f, later %.2f), calls %d, work %d, %.4f s of %.4f s, all families %.2f"
          % (r[3], r[0], r[1], r[2], r[4], r[5], r[6], r[7], r[8]))
print("largest by later-call work:")
for r in sorted(rows, key=lambda r: -r[2])[:top]:
    print("  %-30s later %.2f passes, calls %d" % (r[3], r[2], r[4]))
print("largest by seconds of the family:")
for r in sorted(rows, key=lambda r: -r[6])[:top]:
    print("  %-30s %.4f s of %.4f s (%.1f%%), %.2f passes" % (r[3], r[6], r[7], 100 * r[6] / max(r[7], 1e-9), r[0]))
rounds = [i["rounds"] for i in d]
print("rounds: median %d, largest %d (%s); on 20: %d"
      % (q(rounds, 0.5), max(rounds), max(d, key=lambda i: i["rounds"])["name"], sum(r >= 20 for r in rounds)))
more = [(i["name"], b[i["name"]]["rounds"], i["rounds"]) for i in d if i["rounds"] > b[i["name"]]["rounds"]]
less = [(i["name"], b[i["name"]]["rounds"], i["rounds"]) for i in d if i["rounds"] < b[i["name"]]["rounds"]]
print("more rounds than %s on %d: %s" % (other, len(more), more))
print("fewer rounds on %d: %s" % (len(less), less))
print("fixed point: %d of %d; not set: %s" % (sum(i["fixed_point"] for i in d), len(d),
                                              [i["name"] for i in d if not i["fixed_point"]]))
print("lost against %s: %s" % (other, [i["name"] for i in d if not i["fixed_point"] and b[i["name"]]["fixed_point"]]))
