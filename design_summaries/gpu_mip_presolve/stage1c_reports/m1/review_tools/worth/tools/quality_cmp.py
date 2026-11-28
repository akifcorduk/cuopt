"""quality_cmp.py <before.jsonl> <after.jsonl>: numerical statistics of the reduced models of
two engines on the 240 dumps: objective nonzeros, cost range, coefficient range of the matrix,
ratio of the largest to the smallest absolute coefficient per row and over the model, largest
side. Lists the instances where the reduced model is numerically worse after."""
import json
import math
import sys


def load(path):
    return {r["instance"]: r for r in (json.loads(line) for line in open(path))}


a, b = load(sys.argv[1]), load(sys.argv[2])
names = sorted(a)


def dyn(r, lo, hi):
    return r[hi] / r[lo] if r[lo] > 0 else 1.0


tot = {}
for n in names:
    for side, d in (("before", a[n]["reduced"]), ("after", b[n]["reduced"]), ("input", a[n]["input"])):
        for k in ("obj_nnz", "nnz", "rows_dyn_1e3", "rows_dyn_1e4", "rows_dyn_1e6", "rows_dyn_1e8",
                  "equations", "ranged", "singleton_cols", "free_cols"):
            tot[(side, k)] = tot.get((side, k), 0) + d[k]
print("sums over %d instances            input      before       after" % len(names))
for k in ("nnz", "obj_nnz", "rows_dyn_1e3", "rows_dyn_1e4", "rows_dyn_1e6", "rows_dyn_1e8",
          "equations", "ranged", "singleton_cols", "free_cols"):
    print("  %-22s %12d %11d %11d" % (k, tot[("input", k)], tot[("before", k)], tot[("after", k)]))

worse = {"model coefficient range": [], "largest row ratio": [], "cost range": [],
         "largest side": [], "rows above 1e6": [], "objective nonzeros": [], "99% row ratio": [],
         "largest cost": [], "smallest cost": []}
better = {k: 0 for k in worse}
geo = {"model": 0.0, "row_max": 0.0, "cost": 0.0, "n_cost": 0}
for n in names:
    x, y = a[n]["reduced"], b[n]["reduced"]
    if y["nnz"] == 0 or x["nnz"] == 0:
        continue
    mx, my = dyn(x, "coef_min", "coef_max"), dyn(y, "coef_min", "coef_max")
    geo["model"] += math.log(my / mx)
    geo["row_max"] += math.log(y["row_dyn_max"] / x["row_dyn_max"])
    tests = [("model coefficient range", mx, my), ("largest row ratio", x["row_dyn_max"], y["row_dyn_max"]),
             ("99% row ratio", x["row_dyn_p99"], y["row_dyn_p99"]),
             ("largest side", x["side_max"], y["side_max"]),
             ("rows above 1e6", x["rows_dyn_1e6"], y["rows_dyn_1e6"]),
             ("objective nonzeros", x["obj_nnz"], y["obj_nnz"])]
    if x["obj_nnz"] and y["obj_nnz"]:
        cx, cy = dyn(x, "cost_min", "cost_max"), dyn(y, "cost_min", "cost_max")
        geo["cost"] += math.log(cy / cx)
        geo["n_cost"] += 1
        tests.append(("cost range", cx, cy))
        tests.append(("largest cost", x["cost_max"], y["cost_max"]))
        tests.append(("smallest cost", 1 / x["cost_min"], 1 / y["cost_min"]))
    for title, u, v in tests:
        if v > u * (1 + 1e-9):
            worse[title].append((v / u if u > 0 else float("inf"), n, u, v))
        elif v < u * (1 - 1e-9):
            better[title] += 1
m = len(names)
print("geometric mean of after / before: model coefficient range %.4f, largest row ratio %.4f, cost range %.4f (%d instances with costs)"
      % (math.exp(geo["model"] / m), math.exp(geo["row_max"] / m), math.exp(geo["cost"] / max(geo["n_cost"], 1)), geo["n_cost"]))
for title, items in worse.items():
    items.sort(reverse=True)
    print("%s: larger after on %d instances, smaller on %d" % (title, len(items), better[title]))
    for ratio, n, u, v in items[:40]:
        print("   %-30s %.6g -> %.6g (x %.4g)  obj nnz %d -> %d, cols %d -> %d"
              % (n, u, v, ratio, a[n]["reduced"]["obj_nnz"], b[n]["reduced"]["obj_nnz"],
                 a[n]["reduced"]["cols"], b[n]["reduced"]["cols"]))
