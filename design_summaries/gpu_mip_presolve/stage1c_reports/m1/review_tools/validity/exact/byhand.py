"""Exact enumeration of the four singleton-column rules as the design states them.

byhand.py <first seed> <models>
For every generated tiny model and every application that the design's conditions allow: the
exact optimum before and after (with the model's cost and, for the two projections R3 and R4,
with three more random costs on the columns that stay), and the undo of the exact optimal
point of the reduced model, which must be feasible with the reduced objective value.
"""

import random
import sys
from fractions import Fraction as F

import exactlib as X


def show(m):
    def b(v):
        return "inf" if v is None else str(v)

    lines = []
    for j, c in enumerate(m["cols"]):
        lines.append("  col %d [%s, %s] %s cost %s" % (j, b(c[0]), b(c[1]), "int" if c[2] else "cont", c[3]))
    for i, r in enumerate(m["rows"]):
        terms = " + ".join("%s x%d" % (a, j) for j, a in sorted(r[0].items()))
        lines.append("  row %d %s <= %s <= %s" % (i, b(r[1]), terms, b(r[2])))
    return "\n".join(lines)


def main():
    first, count = int(sys.argv[1]), int(sys.argv[2])
    applied = {"R2": 0, "R3": 0, "R4": 0, "R5": 0}
    kinds = {}
    failures = 0
    skipped = 0
    for seed in range(first, first + count):
        rng = random.Random(seed)
        m = X.generate(rng)
        base = X.solve(m)
        if base[0] == "ambiguous":
            skipped += 1
            continue
        for red, undo, kind in X.candidates(m):
            rm, keep = X.compact(red)
            costs = [None]
            if kind in ("R3", "R4"):
                for _ in range(3):
                    costs.append(
                        [
                            F(rng.randint(-3, 3)) if c is not None and c[0] is not None and c[1] is not None else F(0)
                            for c in red["cols"]
                        ]
                    )
            for cost in costs:
                if cost is None:
                    b, r = base, X.solve(rm)
                else:
                    b = X.solve(m, cost)
                    r = X.solve(rm, [cost[j] for j in keep])
                if b[0] == "ambiguous" or r[0] == "ambiguous":
                    skipped += 1
                    continue
                key = (kind, b[0], cost is None)
                kinds[key] = kinds.get(key, 0) + 1
                ok = b[0] == r[0] and (b[0] != "optimal" or b[1] == r[1])
                why = "optimum %s against %s" % (b[:2], r[:2])
                if ok and r[0] == "optimal":
                    x = [F(0)] * len(m["cols"])
                    for i, j in enumerate(keep):
                        x[j] = r[2][i]
                    undo(x)
                    value = X.objective(m, x) if cost is None else sum(c * v for c, v in zip(cost, x))
                    if not X.feasible(m, x):
                        ok, why = False, "undo gives an infeasible point %s" % x
                    elif value != r[1]:
                        ok, why = False, "undo gives objective %s against %s" % (value, r[1])
                if not ok:
                    failures += 1
                    if failures <= 10:
                        print("FAIL seed %d rule %s: %s\n%s" % (seed, kind, why, show(m)))
            applied[kind] += 1
    print("models %d skipped %d applications %s failures %d" % (count, skipped, applied, failures))
    for key in sorted(kinds):
        print("  %s base %s %s: %d" % (key[0], key[1], "own cost" if key[2] else "other cost", kinds[key]))


if __name__ == "__main__":
    main()
