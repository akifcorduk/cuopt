"""Score solve_MIP EOS runs with the GAMS MIPFeas rules (own port). usage: gams_step_c.py GAMS_DIR LABEL=RUNDIR ..."""
import csv, math, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gams_pi_port import gams_pi, read_trc
gd = sys.argv[1]
z = {}
for r in csv.DictReader(open(os.path.join(gd, "optimal_objective.csv"))):
    try: z[r["InstanceInst."]] = float(r["ObjectiveObje."])
    except ValueError: pass
sgm = lambda v: math.exp(st.mean(math.log(x + 0.001) for x in v)) - 0.001
gams = {}
for f in os.listdir(os.path.join(gd, "trc_files")):
    mps, platform, solver = f[:-4].rsplit("_", 2)
    if solver == "cuopt-26-10": gams.setdefault(platform, {})[mps] = gams_pi(read_trc(os.path.join(gd, "trc_files", f)), z[mps], f)[0]
def our_rows(d, inst, valid_only):
    p = os.path.join(d, inst + "_incumbents.csv")
    inc = []
    if os.path.exists(p):
        inc = sorted((float(r["wall_time_s"]), float(r["objective"])) for r in csv.DictReader(open(p)) if (r["valid"] == "1" or not valid_only))
    rows, best = [{"seriesID": "S", "seconds": 0.0, "bestFound": float("nan"), "bestBound": float("nan")}], float("inf")
    for t, o in inc:
        if o < best:  # bestFound semantics: running best (minimization)
            best = o; rows.append({"seriesID": "I", "seconds": t, "bestFound": o, "bestBound": float("nan")})
    m = re.search(r"^run_solver (\d+)", open(os.path.join(d, inst + ".log"), errors="replace").read(), re.M)
    t_end = int(m.group(1)) / 1000.0 if m else 600.0
    rows.append({"seriesID": "E", "seconds": t_end, "bestFound": best if best < float("inf") else float("nan"), "bestBound": float("nan")})
    return rows
runs = []
for a in sys.argv[2:]:
    lab, d = a.split("=", 1); n = os.path.join(d, os.path.basename(os.path.normpath(d))); runs.append((lab, n if os.path.isdir(n) else d))
insts = sorted(z)
res = {}
for valid_only in (False, True):
    print(f"--- {'valid-only' if valid_only else 'all reported'} incumbents, GAMS rules, basename as cuopt-26-10_gb10 for GAMS' cuOpt whitelists")
    for lab, d in runs:
        P, fails = {}, []
        for i in insts:
            why = []
            P[i], _ = gams_pi(our_rows(d, i, valid_only), z[i], f"{i}_gb10_cuopt-26-10.trc", why=why)
            if why: fails.append(f"{i}: {why[0]}")
        res[(lab, valid_only)] = P
        print(f"  {lab:6s} sgm(0.001)={sgm(P.values()):.5f}  arith={st.mean(P.values()):.4f}  feasible(P<2)={sum(v < 2 for v in P.values())}  GAMS-rule failures(P=2 with a solution)={len(fails)}")
        for x in fails[:6]: print("        " + x)
print(f"--- reference: GAMS cuopt-26-10 gb10 sgm={sgm(gams['gb10'].values()):.5f}, b200 sgm={sgm(gams['b200'].values()):.5f}")
import itertools
b = res[("base0", False)]; g = gams["b200"]
worse = sorted(((b[i] - g[i], i) for i in insts), reverse=True)[:8]
print("largest base0 (all) - GAMS b200 per instance:", ", ".join(f"{i} {b[i]:.4f} vs {g[i]:.4f}" for _, i in worse))
