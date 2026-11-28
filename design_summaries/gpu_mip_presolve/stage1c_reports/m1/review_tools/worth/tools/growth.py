"""growth.py <engine tag>...: per run, the instances by the largest growth factor accepted, the
columns refused by the limit, cost_fill and saved entries; sums."""
import json
import sys

S = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth"
for tag in sys.argv[1:]:
    d = json.load(open(S + "/bench/" + tag + "/summary.json"))["instances"]
    rows = [(i["columns"]["max_growth"], i["name"], i["columns"], i["sub_rules"], i["nnz"]) for i in d]
    print("%s: instances with a growth factor above 1: %d, above 10: %d, above 100: %d, above 1000: %d"
          % (tag, sum(r[0] > 1 for r in rows), sum(r[0] > 10 for r in rows),
             sum(r[0] > 100 for r in rows), sum(r[0] > 1000 for r in rows)))
    for g, n, c, s, nnz in sorted(rows, reverse=True)[:25]:
        if g <= 1:
            break
        print("   %-30s growth %-10.6g refused row %5d history %3d  R2 %5d R5 %5d cost_fill %6d saved %7d (%.3f per nnz)"
              % (n, g, c["growth_refused_row"], c["growth_refused"], s["free_singleton"],
                 s["free_singleton_dual"], c["cost_fill"], c["saved_entries"], c["saved_entries"] / nnz))
    print("   sums: cost_fill %d on %d, saved entries %d on %d, refused row %d on %d, history %d, saved_row_refused %d, rejected_explicit %d on %d"
          % (sum(r[2]["cost_fill"] for r in rows), sum(r[2]["cost_fill"] > 0 for r in rows),
             sum(r[2]["saved_entries"] for r in rows), sum(r[2]["saved_entries"] > 0 for r in rows),
             sum(r[2]["growth_refused_row"] for r in rows), sum(r[2]["growth_refused_row"] > 0 for r in rows),
             sum(r[2]["growth_refused"] for r in rows), sum(r[2]["saved_row_refused"] for r in rows),
             sum(r[2]["free_singleton_rejected_explicit"] for r in rows),
             sum(r[2]["free_singleton_rejected_explicit"] > 0 for r in rows)))
    best = max(rows, key=lambda r: r[2]["saved_entries"] / r[4])
    print("   most saved entries per input nonzero: %.3f on %s; most cost_fill: %s"
          % (best[2]["saved_entries"] / best[4], best[1],
             max(rows, key=lambda r: r[2]["cost_fill"])[1]))
