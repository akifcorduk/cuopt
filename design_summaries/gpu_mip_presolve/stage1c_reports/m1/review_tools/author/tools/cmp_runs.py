"""cmp_runs.py <run A> <run B> [--list N]: two result directories of tools/run_engine.py on the
same dumps. Geometric mean of A / B in remaining nonzeros, rows and columns over the instances
with a model on both sides (all, and without ex9, ex10, enlight_hard), the sums, the number of
instances where B leaves less, the same, more, and the N instances with the largest gain of B
in nonzeros."""
import json
import math
import sys

LEFT_OUT = ("ex9", "ex10", "enlight_hard")


def load(path):
    rows = {}
    for r in json.load(open(path + "/summary.json"))["instances"]:
        if r.get("status") in ("reduced", "unchanged", "reduced_to_empty"):
            rows[r["name"]] = r
    return rows


def size(r, key):
    red = r["reduced"]
    return red[key]


a = load(sys.argv[1])
b = load(sys.argv[2])
show = 0
if "--list" in sys.argv:
    show = int(sys.argv[sys.argv.index("--list") + 1])
names = sorted(set(a) & set(b))
for label, pick in (("all", names), ("without ex9, ex10, enlight_hard", [n for n in names if n not in LEFT_OUT])):
    line = "%-34s n %3d" % (label, len(pick))
    for key in ("nnz", "rows", "cols"):
        logs = [math.log(max(size(a[n], key), 1) / max(size(b[n], key), 1)) for n in pick]
        line += "  %s %.4f" % (key, math.exp(sum(logs) / len(logs)))
    print(line)
for key in ("nnz", "rows", "cols"):
    sa = sum(size(a[n], key) for n in names)
    sb = sum(size(b[n], key) for n in names)
    less = sum(1 for n in names if size(b[n], key) < size(a[n], key))
    more = sum(1 for n in names if size(b[n], key) > size(a[n], key))
    print("%-5s sum A %12d sum B %12d (%+.3f%%) | B leaves less on %d, the same on %d, more on %d"
          % (key, sa, sb, 100. * (sb - sa) / sa, less, len(names) - less - more, more))
    if more:
        print("      more:", ", ".join("%s %d -> %d" % (n, size(a[n], key), size(b[n], key))
                                       for n in names if size(b[n], key) > size(a[n], key))[:900])
if show:
    gain = sorted(names, key=lambda n: size(b[n], "nnz") / max(size(a[n], "nnz"), 1))
    for n in gain[:show]:
        print("  %-28s nnz %9d -> %9d rows %7d -> %7d cols %7d -> %7d"
              % (n, size(a[n], "nnz"), size(b[n], "nnz"), size(a[n], "rows"), size(b[n], "rows"),
                 size(a[n], "cols"), size(b[n], "cols")))
