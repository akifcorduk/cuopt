"""The engine on tiny models whose data sit a tolerance away from the integers.

tolerance_check.py <driver binary> <first seed> <models>
The models of the exact enumeration with bounds and sides moved by up to 3e-7 and, in rows
with integer columns only, coefficients multiplied by 1 +- 5e-10. A presolve with tolerances
may accept a point that is feasible within the feasibility tolerance and must keep every point
that is feasible exactly. For the option sets that run the singleton-column family alone:
  the reduced optimum is not above the exact optimum of the model (nothing feasible is lost);
  it is not below the optimum of the model relaxed by the tolerance (nothing more than the
  tolerance is gained);
  the postsolve of the exact optimal point of the reduced model is feasible within twice the
  tolerance.
"""

import random
import subprocess
import sys
from fractions import Fraction as F

import engine_check as E
import exactlib as X

DELTA = F(1, 10**6)
EPS = F(1, 10**8)
SETS = ("sc", "sc_r2", "sc_r5", "sc_r3", "sc_r4")


def perturb(m, rng):
    steps = [F(1, 10**7), F(3, 2 * 10**7), F(3, 10**7), F(-1, 10**7), F(-3, 10**7), F(5, 10**8)]
    for c in m["cols"]:
        for b in (0, 1):
            if c[b] is not None and rng.random() < 0.3:
                c[b] = F(float(c[b] + rng.choice(steps)))
        if c[0] is not None and c[1] is not None and c[0] > c[1]:
            c[0], c[1] = c[1], c[0]
    for r in m["rows"]:
        for b in (1, 2):
            if r[b] is not None and rng.random() < 0.3:
                r[b] = F(float(r[b] + rng.choice(steps)))
        if r[1] is not None and r[2] is not None and r[1] > r[2]:
            r[1], r[2] = r[2], r[1]
        if all(m["cols"][j][2] for j in r[0]) and rng.random() < 0.3:
            j = rng.choice(sorted(r[0]))
            r[0][j] = F(float(r[0][j] * (1 + rng.choice([5, -5, 2]) * F(1, 10**10))))
    return m


def relaxed(m, delta):
    r = X.copy_model(m)
    for c in r["cols"]:
        if c[0] is not None:
            c[0] -= delta
        if c[1] is not None:
            c[1] += delta
    for row in r["rows"]:
        if row[1] is not None:
            row[1] -= delta
        if row[2] is not None:
            row[2] += delta
    return r


def main():
    driver, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    models, exact, loose = {}, {}, {}
    lines = []
    for seed in range(first, first + count):
        rng = random.Random(seed)
        m = perturb(X.generate(rng), rng)
        models[seed] = m
        exact[seed] = X.solve(m)
        loose[seed] = X.solve(relaxed(m, DELTA))
        E.write_model(lines, seed, m)
        lines.append("E")
    first_pass = E.parse(E.run(driver, "\n".join(lines) + "\n"))
    failures = 0
    tally = {}

    def fail(seed, name, why):
        nonlocal failures
        failures += 1
        if failures <= 20:
            print("FAIL seed %d set %s: %s" % (seed, name, why))

    lines = []
    optima = {}
    for seed in range(first, first + count):
        m, block = models[seed], first_pass.get(seed)
        if block is None:
            continue
        e, l = exact[seed], loose[seed]
        E.write_model(lines, seed, m)
        for res in block:
            name = res["set"]
            if name not in SETS:
                continue
            t = tally.setdefault(name, {"results": 0, "findings": 0, "checked": 0, "points": 0, "skipped": 0})
            t["results"] += 1
            t["findings"] += (res["fs"] + res["fsd"] + res["zs"] + res["fc"]) > 0
            if e[0] == "ambiguous" or l[0] == "ambiguous":
                t["skipped"] += 1
                continue
            if res["status"] == "infeasible":
                if e[0] != "infeasible":
                    fail(seed, name, "engine says infeasible, the exact model is %s" % (e[:2],))
                continue
            if res["status"] == "unbounded_or_infeasible":
                if l[0] not in ("infeasible", "unbounded") and e[0] not in ("infeasible", "unbounded"):
                    fail(seed, name, "engine says unbounded or infeasible, exact %s relaxed %s" % (e[:2], l[:2]))
                continue
            r = X.solve(res["model"])
            if r[0] == "ambiguous":
                t["skipped"] += 1
                continue
            t["checked"] += 1
            # Nothing that is feasible exactly is lost.
            if e[0] == "optimal" and (r[0] == "infeasible" or (r[0] == "optimal" and r[1] > e[1] + EPS * (1 + abs(e[1])))):
                fail(seed, name, "reduced %s is above the exact optimum %s" % (r[:2], e[:2]))
                continue
            if e[0] == "unbounded" and r[0] != "unbounded":
                fail(seed, name, "reduced %s, the exact model is unbounded" % (r[:2],))
                continue
            # Nothing beyond the tolerance is gained.
            if l[0] == "infeasible" and r[0] != "infeasible":
                fail(seed, name, "reduced %s, the relaxed model is infeasible" % (r[:2],))
                continue
            if l[0] == "optimal" and (r[0] == "unbounded" or (r[0] == "optimal" and r[1] < l[1] - EPS * (1 + abs(l[1])))):
                fail(seed, name, "reduced %s is below the relaxed optimum %s" % (r[:2], l[:2]))
                continue
            if r[0] == "optimal":
                optima[(seed, name)] = r
                lines.append("Q %s %s" % (name, " ".join(repr(float(v)) for v in r[2])))
        lines.append("E")
    second_pass = E.parse(E.run(driver, "\n".join(lines) + "\n"))
    worst = F(0)
    for seed in range(first, first + count):
        m, block = models[seed], second_pass.get(seed)
        if block is None:
            continue
        for res in block:
            name = res["set"]
            if (seed, name) not in optima:
                continue
            for x in res["ps"]:
                tally[name]["points"] += 1
                if not X.feasible(m, x, 2 * DELTA):
                    fail(seed, name, "postsolved point %s is infeasible beyond twice the tolerance" % [float(v) for v in x])
    print("models %d failures %d" % (count, failures))
    for name in sorted(tally):
        print("  %-8s %s" % (name, tally[name]))


if __name__ == "__main__":
    main()
