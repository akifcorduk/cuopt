"""The engine on the tiny models of the exact enumeration, judged in exact arithmetic.

engine_check.py <driver binary> <first seed> <models> [work dir]
For every model and seven option sets (default, primal-only, the singleton-column family alone,
and that family with one of its four reductions each): the exact optimum of the reduced model
against that of the model, the exact optimal point of the reduced model postsolved (feasible,
same objective), and the exact optimal point of the model crushed (feasible for the reduced
model, objective not above, equal without a dual finding).
"""

import os
import random
import subprocess
import sys
from fractions import Fraction as F

import exactlib as X

TOL = F(1, 10**7)


def num(v, sign):
    if v is None:
        return "inf" if sign > 0 else "-inf"
    return repr(float(v))


def write_model(out, idx, m):
    out.append("M %d %d %d" % (idx, len(m["cols"]), len(m["rows"])))
    for lo, hi, is_int, cost in m["cols"]:
        out.append("C %s %s %d %s" % (num(lo, -1), num(hi, 1), 1 if is_int else 0, num(cost, 1)))
    for entries, lo, hi in m["rows"]:
        items = " ".join("%d %s" % (j, num(a, 1)) for j, a in sorted(entries.items()))
        out.append("R %s %s %d %s" % (num(lo, -1), num(hi, 1), len(entries), items))


def val(s):
    v = float(s)
    if v == float("inf") or v == float("-inf"):
        return None
    return F(v)


def parse(text):
    results = {}
    current = None
    block = []
    for line in text.splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "RES":
            current = {
                "set": w[2],
                "status": w[3],
                "fp": int(w[5]),
                "dual": int(w[7]),
                "fs": int(w[9]),
                "fsd": int(w[11]),
                "zs": int(w[13]),
                "fc": int(w[15]),
                "model": None,
                "origin": [],
                "ps": [],
                "cr": [],
                "sc": None,
            }
            block.append(current)
        elif w[0] == "SC":
            current["sc"] = (w[2] == "1", line)
        elif w[0] == "RM":
            current["model"] = {"cols": [], "rows": [], "offset": F(float(w[3]))}
        elif w[0] == "RC":
            current["origin"].append(int(w[1]))
            current["model"]["cols"].append([val(w[2]), val(w[3]), w[4] == "1", F(float(w[5]))])
        elif w[0] == "RR":
            n = int(w[3])
            entries = {int(w[4 + 2 * k]): F(float(w[5 + 2 * k])) for k in range(n)}
            current["model"]["rows"].append([entries, val(w[1]), val(w[2])])
        elif w[0] == "PS":
            current["ps"].append([F(float(v)) for v in w[2:]])
        elif w[0] == "CR":
            current["cr"].append([F(float(v)) for v in w[2:]])
        elif w[0] == "END":
            results[int(w[1])] = block
            block = []
        elif w[0] == "BAD":
            results[int(w[1])] = None
            block = []
    return results


def run(driver, text):
    return subprocess.run([driver], input=text, capture_output=True, text=True, check=True).stdout


def close(a, b):
    return abs(a - b) <= TOL * (1 + abs(a))


def main():
    driver, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    models, bases = {}, {}
    lines = []
    for seed in range(first, first + count):
        m = X.generate(random.Random(seed))
        models[seed] = m
        bases[seed] = X.solve(m)
        write_model(lines, seed, m)
        lines.append("E")
    first_pass = parse(run(driver, "\n".join(lines) + "\n"))

    failures = 0
    tally = {}

    def fail(seed, name, why):
        nonlocal failures
        failures += 1
        if failures <= 15:
            print("FAIL seed %d set %s: %s" % (seed, name, why))

    lines = []
    reduced_optima = {}
    for seed in range(first, first + count):
        m, base, block = models[seed], bases[seed], first_pass.get(seed)
        if block is None:
            continue
        write_model(lines, seed, m)
        for res in block:
            name = res["set"]
            t = tally.setdefault(name, {"results": 0, "fs": 0, "fsd": 0, "zs": 0, "fc": 0, "checked": 0, "points": 0, "crushed": 0, "skipped": 0})
            t["results"] += 1
            for k in ("fs", "fsd", "zs", "fc"):
                t[k] += res[k] > 0
            if res["sc"] is not None:
                t["structure"] = t.get("structure", 0) + 1
                if not res["sc"][0]:
                    fail(seed, name, "structure check: " + res["sc"][1])
            if name == "primal" and (res["dual"] or res["fsd"] or res["zs"] or res["fc"]):
                fail(seed, name, "a dual reduction in primal-only mode")
            if base[0] == "ambiguous":
                t["skipped"] += 1
                continue
            if res["status"] == "infeasible":
                if base[0] != "infeasible":
                    fail(seed, name, "engine says infeasible, exact %s" % (base[:2],))
                continue
            if res["status"] == "unbounded_or_infeasible":
                if base[0] not in ("infeasible", "unbounded"):
                    fail(seed, name, "engine says unbounded or infeasible, exact %s" % (base[:2],))
                continue
            r = X.solve(res["model"])
            if r[0] == "ambiguous":
                t["skipped"] += 1
                continue
            t["checked"] += 1
            if r[0] != base[0] or (r[0] == "optimal" and not close(r[1], base[1])):
                fail(seed, name, "reduced optimum %s against %s" % (r[:2], base[:2]))
                continue
            if r[0] == "optimal":
                reduced_optima[(seed, name)] = r
                lines.append("Q %s %s" % (name, " ".join(repr(float(v)) for v in r[2])))
        if base[0] == "optimal":
            lines.append("P " + " ".join(repr(float(v)) for v in base[2]))
        lines.append("E")
    second_pass = parse(run(driver, "\n".join(lines) + "\n"))

    for seed in range(first, first + count):
        m, base, block = models[seed], bases[seed], second_pass.get(seed)
        if block is None:
            continue
        for res in block:
            name = res["set"]
            t = tally[name]
            r = reduced_optima.get((seed, name))
            if r is None:
                continue
            for x in res["ps"]:
                t["points"] += 1
                if not X.feasible(m, x, TOL):
                    fail(seed, name, "postsolved point %s is infeasible" % [float(v) for v in x])
                elif not close(X.objective(m, x), r[1]):
                    fail(seed, name, "postsolved objective %s against %s" % (float(X.objective(m, x)), float(r[1])))
            if base[0] != "optimal":
                continue
            for z in res["cr"]:
                t["crushed"] += 1
                if not X.feasible(res["model"], z, TOL):
                    fail(seed, name, "crushed optimal point %s is infeasible in the reduced model" % [float(v) for v in z])
                    continue
                value = X.objective(res["model"], z)
                if value > base[1] + TOL * (1 + abs(base[1])):
                    fail(seed, name, "crushed objective %s above %s" % (float(value), float(base[1])))
                elif res["dual"] == 0 and not close(value, base[1]):
                    fail(seed, name, "crushed objective %s differs from %s without a dual finding" % (float(value), float(base[1])))
    print("models %d failures %d" % (count, failures))
    for name in sorted(tally):
        print("  %-8s %s" % (name, tally[name]))


if __name__ == "__main__":
    main()
