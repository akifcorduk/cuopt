"""time_slow.py <table of time_multi.py>: the instances more than 5% and 1 ms slower or faster
with their binary than with the base binary (minimum of the passes)."""
import sys

lines = open(sys.argv[1]).read().split("\n")
head = lines[0].split()[1:]
n = len(head)
b, m = head.index("base"), head.index("m1")
for line in lines[1:]:
    w = line.split()
    if not w:
        continue
    t = [float(x) for x in w[1:1 + n]]
    fam = [float(x) for x in w[1 + n:1 + 2 * n]]
    d = t[m] - t[b]
    if abs(d) > 1e-3 and abs(d) > 0.05 * t[b]:
        print("%-30s base %.4f theirs %.4f (%+.4f s, %+.1f%%), family %.4f s"
              % (w[0], t[b], t[m], d, 100 * d / t[b], fam[m]))
