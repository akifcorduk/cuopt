"""time_table.py <table of time_multi.py> [top n]: per instance, the minimum seconds of the base
binary and of theirs, the difference and the seconds of the family; the largest changes."""
import sys

rows = []
lines = open(sys.argv[1]).read().split("\n")
head = lines[0].split()[1:]
n = len(head)
for line in lines[1:]:
    w = line.split()
    if not w:
        continue
    t = [float(x) for x in w[1:1 + n]]
    fam = [float(x) for x in w[1 + n:1 + 2 * n]]
    rows.append((w[0], t, fam))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 10
b, m = head.index("base"), head.index("m1")
print("%d instances; sums: %s" % (len(rows), ", ".join("%s %.4f" % (h, sum(r[1][k] for r in rows)) for k, h in enumerate(head))))
print("family seconds (m1): %.4f" % sum(r[2][m] for r in rows))
slower = sum(1 for r in rows if r[1][m] > r[1][b] * 1.05 and r[1][m] - r[1][b] > 1e-3)
faster = sum(1 for r in rows if r[1][m] < r[1][b] * 0.95 and r[1][b] - r[1][m] > 1e-3)
print("more than 5%% and 1 ms slower: %d; faster: %d" % (slower, faster))
print("largest increase in seconds:")
for r in sorted(rows, key=lambda r: r[1][b] - r[1][m])[:top]:
    print("  %-30s base %.4f m1 %.4f (%+.4f, %+.1f%%), family %.4f"
          % (r[0], r[1][b], r[1][m], r[1][m] - r[1][b], 100 * (r[1][m] / r[1][b] - 1), r[2][m]))
print("largest decrease in seconds:")
for r in sorted(rows, key=lambda r: r[1][m] - r[1][b])[:top]:
    print("  %-30s base %.4f m1 %.4f (%+.4f, %+.1f%%), family %.4f"
          % (r[0], r[1][b], r[1][m], r[1][m] - r[1][b], 100 * (r[1][m] / r[1][b] - 1), r[2][m]))
print("largest family seconds:")
for r in sorted(rows, key=lambda r: -r[2][m])[:top]:
    print("  %-30s family %.4f of %.4f (%.1f%%)" % (r[0], r[2][m], r[1][m], 100 * r[2][m] / r[1][m]))
print("largest family share of an instance above 5 ms:")
for r in sorted((r for r in rows if r[1][m] > 0.005), key=lambda r: -r[2][m] / r[1][m])[:top]:
    print("  %-30s family %.4f of %.4f (%.1f%%)" % (r[0], r[2][m], r[1][m], 100 * r[2][m] / r[1][m]))
