"""Tiny MIPs in exact rational arithmetic: a solver, the four singleton-column rules of the
Stage 1c design applied by their stated conditions, and their undo.

A model is a dict:
  cols: list of [lo, hi, is_int, cost]   (lo / hi are Fractions or None for an infinite bound)
  rows: list of [entries (dict col -> Fraction), lo, hi]
  offset: Fraction
"""

import itertools
import math
import random
from fractions import Fraction as F

WINDOW = 14
INNER = 9


def copy_model(m):
    return {
        "cols": [list(c) for c in m["cols"]],
        "rows": [[dict(r[0]), r[1], r[2]] for r in m["rows"]],
        "offset": m["offset"],
    }


# ---- exact LP over a few continuous variables: Fourier-Motzkin with z = objective ----


def _normalize(con):
    coef, rhs = con
    scale = None
    for v in coef:
        if v != 0:
            scale = abs(v)
            break
    if scale is None:
        return (coef, rhs)
    return (tuple(v / scale for v in coef), rhs / scale)


def _dedup(cons):
    best = {}
    for coef, rhs in map(_normalize, cons):
        if coef not in best or rhs < best[coef]:
            best[coef] = rhs
    return list(best.items())


def lp_min(n, cons, cost):
    """min cost.x subject to coef.x <= rhs for (coef, rhs) in cons, x in R^n.

    Returns None (infeasible), "unbounded", or (value, point).
    """
    # variable n is z, with z >= cost.x
    cons = [(tuple(c) + (F(0),), r) for c, r in cons]
    cons.append((tuple(cost) + (F(-1),), F(0)))
    stages = []
    for k in range(n):
        cons = _dedup(cons)
        stages.append(cons)
        upper = [c for c in cons if c[0][k] > 0]
        lower = [c for c in cons if c[0][k] < 0]
        rest = [c for c in cons if c[0][k] == 0]
        for cu, ru in upper:
            for cl, rl in lower:
                a, b = cu[k], -cl[k]
                coef = tuple(cu[i] * b + cl[i] * a for i in range(n + 1))
                rest.append((coef, ru * b + rl * a))
        cons = rest
    cons = _dedup(cons)
    z_lower = None
    for coef, rhs in cons:
        zc = coef[n]
        if zc == 0:
            if rhs < 0:
                return None
        elif zc < 0:
            v = rhs / zc
            if z_lower is None or v > z_lower:
                z_lower = v
        # zc > 0 cannot occur: z has only lower bounds
    if z_lower is None:
        return "unbounded"
    x = [F(0)] * (n + 1)
    x[n] = z_lower
    for k in reversed(range(n)):
        lo, hi = None, None
        for coef, rhs in stages[k]:
            a = coef[k]
            if a == 0:
                continue
            rest = rhs - sum(coef[i] * x[i] for i in range(k + 1, n + 1))
            v = rest / a
            if a > 0:
                hi = v if hi is None or v < hi else hi
            else:
                lo = v if lo is None or v > lo else lo
        if lo is not None:
            x[k] = lo
        elif hi is not None:
            x[k] = hi
        else:
            x[k] = F(0)
    return (z_lower, x[:n])


def solve(m, cost=None):
    """Exact optimum: ("infeasible",), ("unbounded",), ("ambiguous",) or ("optimal", value, x)."""
    cols, rows = m["cols"], m["rows"]
    cost = [c[3] for c in cols] if cost is None else cost
    ncols = len(cols)
    enum, cont = [], []
    ranges = []
    windowed = set()
    for j, (lo, hi, is_int, _) in enumerate(cols):
        if not is_int:
            cont.append(j)
            continue
        a = math.ceil(lo) if lo is not None else -WINDOW
        b = math.floor(hi) if hi is not None else WINDOW
        if lo is None or hi is None:
            windowed.add(j)
        enum.append(j)
        ranges.append(range(a, b + 1))
    pos = {j: i for i, j in enumerate(cont)}
    n = len(cont)
    best = None
    inner = None
    unbounded = False
    for values in itertools.product(*ranges):
        x = [F(0)] * ncols
        for j, v in zip(enum, values):
            x[j] = F(v)
        cons = []
        feasible = True
        for entries, lo, hi in rows:
            fixed = sum(a * x[j] for j, a in entries.items() if j not in pos)
            coef = [F(0)] * n
            for j, a in entries.items():
                if j in pos:
                    coef[pos[j]] = a
            if all(c == 0 for c in coef):
                if (lo is not None and fixed < lo) or (hi is not None and fixed > hi):
                    feasible = False
                    break
                continue
            if hi is not None:
                cons.append((tuple(coef), hi - fixed))
            if lo is not None:
                cons.append((tuple(-c for c in coef), fixed - lo))
        if not feasible:
            continue
        for j in cont:
            lo, hi = cols[j][0], cols[j][1]
            unit = [F(0)] * n
            unit[pos[j]] = F(1)
            if hi is not None:
                cons.append((tuple(unit), hi))
            if lo is not None:
                cons.append((tuple(-u for u in unit), -lo))
        r = lp_min(n, cons, [cost[j] for j in cont])
        if r is None:
            continue
        if r == "unbounded":
            unbounded = True
            continue
        value = r[0] + sum(cost[j] * x[j] for j in enum) + m["offset"]
        for j, v in zip(cont, r[1]):
            x[j] = v
        if best is None or value < best[0]:
            best = (value, x)
        if all(abs(x[j]) <= INNER for j in windowed) and (inner is None or value < inner):
            inner = value
    if unbounded:
        return ("unbounded",)
    if best is None:
        return ("infeasible",)
    # An optimum that needs the outer part of the window of an integer column without a bound
    # is not trusted: the model may be unbounded, or the window too small.
    if inner is None or best[0] < inner:
        return ("ambiguous",)
    return ("optimal", best[0], best[1])


def feasible(m, x, tol=F(0)):
    for j, (lo, hi, is_int, _) in enumerate(m["cols"]):
        if lo is not None and x[j] < lo - tol:
            return False
        if hi is not None and x[j] > hi + tol:
            return False
        if is_int and abs(x[j] - round(x[j])) > tol:
            return False
    for entries, lo, hi in m["rows"]:
        act = sum(a * x[j] for j, a in entries.items())
        if lo is not None and act < lo - tol:
            return False
        if hi is not None and act > hi + tol:
            return False
    return True


def objective(m, x):
    return m["offset"] + sum(c[3] * x[j] for j, c in enumerate(m["cols"]))


# ---- the rules by the design's conditions, exact ----


def col_rows(m, k):
    return [i for i, r in enumerate(m["rows"]) if k in r[0]]


def rest_range(m, row, k):
    """(min, max) of the row without column k over the bounds; None for an infinite end."""
    lo, hi = F(0), F(0)
    for j, a in m["rows"][row][0].items():
        if j == k:
            continue
        cl, ch = m["cols"][j][0], m["cols"][j][1]
        low_bound, high_bound = (cl, ch) if a > 0 else (ch, cl)
        lo = None if lo is None or low_bound is None else lo + a * low_bound
        hi = None if hi is None or high_bound is None else hi + a * high_bound
    return lo, hi


def free_singleton(m, k):
    """R2 (equation) or R5 (inequality). Returns (reduced, undo, kind) or None."""
    rs = col_rows(m, k)
    if len(rs) != 1:
        return None
    r = rs[0]
    entries, lo, hi = m["rows"][r]
    if len(entries) < 2:
        return None
    a = entries[k]
    cl, ch, is_int, cost = m["cols"][k]
    equation = lo is not None and lo == hi
    if equation:
        side = hi
    else:
        if cost == 0:
            return None
        rises = (cost < 0) == (a > 0)
        side = hi if rises else lo
        if side is None:
            return None
    rmin, rmax = rest_range(m, r, k)
    # x_k = (side - rest) / a
    if a > 0:
        low = None if rmax is None else (side - rmax) / a
        high = None if rmin is None else (side - rmin) / a
    else:
        low = None if rmin is None else (side - rmin) / a
        high = None if rmax is None else (side - rmax) / a
    if cl is not None and (low is None or low < cl):
        return None
    if ch is not None and (high is None or high > ch):
        return None
    if is_int:
        for j, aj in entries.items():
            if j == k:
                continue
            if not m["cols"][j][2] or (aj / a).denominator != 1:
                return None
        if (side / a).denominator != 1:
            return None
    red = copy_model(m)
    q = cost / a
    for j, aj in entries.items():
        if j != k:
            red["cols"][j][3] -= q * aj
    red["offset"] += q * side
    red["rows"][r] = None
    red["cols"][k] = None
    others = {j: aj for j, aj in entries.items() if j != k}

    def undo(x):
        x[k] = (side - sum(aj * x[j] for j, aj in others.items())) / a

    return red, undo, ("R2" if equation else "R5")


def zero_cost(m, r):
    """R3 on row r. Returns (reduced, undo, kind) or None."""
    entries, lo, hi = m["rows"][r]
    if lo is None or hi is None:
        return None
    taken = [
        j
        for j in sorted(entries)
        if not m["cols"][j][2] and m["cols"][j][3] == 0 and len(col_rows(m, j)) == 1
    ]
    if not taken or len(taken) == len(entries):
        return None
    most, least = F(0), F(0)
    for j in taken:
        a = entries[j]
        cl, ch = m["cols"][j][0], m["cols"][j][1]
        at_most, at_least = (ch, cl) if a > 0 else (cl, ch)
        most = None if most is None or at_most is None else most + a * at_most
        least = None if least is None or at_least is None else least + a * at_least
    new_lo = None if most is None else lo - most
    new_hi = None if least is None else hi - least
    red = copy_model(m)
    for j in taken:
        del red["rows"][r][0][j]
        red["cols"][j] = None
    if new_lo is None and new_hi is None:
        red["rows"][r] = None
    else:
        red["rows"][r][1] = new_lo
        red["rows"][r][2] = new_hi
    others = {j: aj for j, aj in entries.items() if j not in taken}

    def undo(x):
        act = sum(aj * x[j] for j, aj in others.items())
        prefer = {}
        for j in taken:
            cl, ch = m["cols"][j][0], m["cols"][j][1]
            prefer[j] = cl if cl is not None else ch if ch is not None else F(0)
            x[j] = prefer[j]
            act += entries[j] * prefer[j]
        if lo <= act <= hi:
            return
        raise_ = act < lo
        side = lo if raise_ else hi
        need = abs(side - act)
        for j in taken:
            if need <= 0:
                break
            a = entries[j]
            up = (a > 0) == raise_
            target = m["cols"][j][1] if up else m["cols"][j][0]
            room = None if target is None else abs(a) * abs(target - prefer[j])
            act -= a * prefer[j]
            if room is not None and room <= need:
                x[j] = target
                act += a * target
                need = abs(side - act) if (act < side) == raise_ and act != side else F(0)
            else:
                x[j] = (side - act) / a
                need = F(0)

    return red, undo, "R3"


def forcing(m, k):
    """R4 on column k. Returns (reduced, undo, kind) or None."""
    cl, ch, is_int, cost = m["cols"][k]
    if cost != 0:
        return None
    down_locks = up_locks = 0
    rs = col_rows(m, k)
    for r in rs:
        entries, lo, hi = m["rows"][r]
        a = entries[k]
        if hi is not None:
            if a > 0:
                up_locks += 1
            else:
                down_locks += 1
        if lo is not None:
            if a > 0:
                down_locks += 1
            else:
                up_locks += 1
    down = down_locks == 0 and cl is None and up_locks >= 1
    up = up_locks == 0 and ch is None and down_locks >= 1
    if not down and not up:
        return None
    red = copy_model(m)
    saved = []
    for r in rs:
        entries, lo, hi = m["rows"][r]
        a = entries[k]
        side = (hi if a > 0 else lo) if down else (lo if a > 0 else hi)
        red["rows"][r] = None
        if side is None:
            continue
        saved.append((a, side, {j: aj for j, aj in entries.items() if j != k}))
    red["cols"][k] = None

    def undo(x):
        value = ch if down else cl
        for a, side, others in saved:
            limit = (side - sum(aj * x[j] for j, aj in others.items())) / a
            if value is None or (limit < value if down else limit > value):
                value = limit
        if is_int:
            value = F(math.floor(value)) if down else F(math.ceil(value))
        x[k] = value

    return red, undo, "R4"


def compact(red):
    """The reduced model without its deleted rows and columns, and the maps of the indices."""
    keep = [j for j, c in enumerate(red["cols"]) if c is not None]
    index = {j: i for i, j in enumerate(keep)}
    rows = []
    for r in red["rows"]:
        if r is None:
            continue
        rows.append([{index[j]: a for j, a in r[0].items()}, r[1], r[2]])
    m = {"cols": [list(red["cols"][j]) for j in keep], "rows": rows, "offset": red["offset"]}
    return m, keep


# ---- a generator of tiny models aimed at the four rules ----


def generate(rng):
    n_int = rng.randint(1, 3)
    cols = []
    for _ in range(n_int):
        lo = rng.randint(-2, 1)
        cols.append([F(lo), F(lo + rng.randint(1, 2)), True, F(rng.randint(-2, 2))])
    if rng.random() < 0.3:
        lo = F(rng.randint(-4, 2), 2)
        cols.append([lo, lo + F(rng.randint(1, 4), 2), False, F(rng.randint(-2, 2))])
    ordinary = len(cols)
    rows = []
    for _ in range(rng.randint(1, 3)):
        entries = {}
        for j in range(ordinary):
            if rng.random() < 0.7:
                entries[j] = F(rng.choice([-2, -1, 1, 2, 3]))
        if not entries:
            entries[rng.randrange(ordinary)] = F(1)
        kind = rng.random()
        s = F(rng.randint(-4, 6))
        if kind < 0.3:
            lo, hi = s, s
        elif kind < 0.5:
            lo, hi = None, s
        elif kind < 0.7:
            lo, hi = s, None
        else:
            lo, hi = s, s + rng.randint(1, 4)
        rows.append([entries, lo, hi])

    def bounds():
        t = rng.random()
        lo = F(rng.randint(-3, 1))
        if t < 0.3:
            return None, None
        if t < 0.5:
            return lo, None
        if t < 0.65:
            return None, lo + rng.randint(0, 4)
        return lo, lo + rng.randint(1, 5)

    open_integer = False
    # singleton columns
    for _ in range(rng.randint(1, 3)):
        r = rng.randrange(len(rows))
        if sum(1 for j in rows[r][0] if j >= ordinary) >= 2:
            continue
        lo, hi = bounds()
        is_int = rng.random() < 0.25
        if is_int and (lo is None or hi is None):
            if open_integer:
                is_int = False
            open_integer = True
        cost = F(0) if rng.random() < 0.5 else F(rng.choice([-2, -1, 1, 2]))
        cols.append([lo, hi, is_int, cost])
        rows[r][0][len(cols) - 1] = F(rng.choice([-2, -1, 1, 2]))
    # a forcing column
    if rng.random() < 0.5:
        down = rng.random() < 0.5
        bound = None if rng.random() < 0.5 else F(rng.randint(-2, 3))
        is_int = rng.random() < 0.4 and not open_integer
        cost = F(0) if rng.random() < 0.85 else F(rng.choice([-1, 1]))
        k = len(cols)
        cols.append([None if down else bound, bound if down else None, is_int, cost])
        picks = rng.sample(range(len(rows)), rng.randint(1, min(2, len(rows))))
        for r in picks:
            entries, lo, hi = rows[r]
            a = F(rng.choice([1, 2]))
            if lo is not None and hi is not None:
                if rng.random() < 0.7:
                    continue
            elif (hi is not None) != down:
                a = -a
            if rng.random() < 0.1:
                a = -a
            entries[k] = a
        if not any(k in r[0] for r in rows):
            cols.pop()
    return {"cols": cols, "rows": rows, "offset": F(0)}


def candidates(m):
    """Every application of a rule that the design's conditions allow on the model."""
    out = []
    for k in range(len(m["cols"])):
        if len(col_rows(m, k)) == 1:
            a = free_singleton(m, k)
            if a:
                out.append(a)
        a = forcing(m, k)
        if a:
            out.append(a)
    for r in range(len(m["rows"])):
        a = zero_cost(m, r)
        if a:
            out.append(a)
    return out
