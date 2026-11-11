"""Fix vs base comparison for single-rep pairs: MIPFeas (lenient + GAMS rules), BKS gap, per-instance PI with
the FP-affected split. usage: ab_pair.py GAMS_DIR AFFECTED_CSV BENCH_ROOT base=RUN fix=RUN [noise_a=RUN noise_b=RUN]"""
import csv, glob, math, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gams_pi_port import gams_pi
gd, affp, root = sys.argv[1:4]
runs = dict(a.split("=", 1) for a in sys.argv[4:])
z = {}
for r in csv.DictReader(open(os.path.join(gd, "optimal_objective.csv"))):
    try: z[r["InstanceInst."]] = float(r["ObjectiveObje."])
    except ValueError: pass
aff = {l.split(",")[0]: float(l.split(",")[1]) for l in open(affp)}
T = 600.0
sgm = lambda v, sh=0.001: math.exp(st.mean(math.log(x + sh) for x in v)) - sh
def inc(run, i, valid_only):
    p = os.path.join(root, run, run, i + "_incumbents.csv")
    return sorted((float(r["wall_time_s"]), float(r["objective"])) for r in csv.DictReader(open(p)) if r["valid"] == "1" or not valid_only) if os.path.exists(p) else []
def lenient(pts, zz):
    area, tp, pp = 0.0, 0.0, 2.0
    for t, o in pts:
        if t >= T: break
        g = 1.0 if o * zz < 0 else abs(o - zz) / max(abs(o), abs(zz), 1.0)
        if g < pp: area += (t - tp) * pp; tp, pp = t, g
    return (area + (T - tp) * pp) / T
def gams_rows(run, i, valid_only):
    rows, best = [{"seriesID": "S", "seconds": 0.0, "bestFound": float("nan"), "bestBound": float("nan")}], float("inf")
    for t, o in inc(run, i, valid_only):
        if o < best: best = o; rows.append({"seriesID": "I", "seconds": t, "bestFound": o, "bestBound": float("nan")})
    m = re.search(r"^run_solver (\d+)", open(os.path.join(root, run, run, i + ".log"), errors="replace").read(), re.M)
    rows.append({"seriesID": "E", "seconds": int(m.group(1)) / 1000.0 if m else T, "bestFound": best if best < float("inf") else float("nan"), "bestBound": float("nan")})
    return rows
def final(run):
    out = {}
    for f in glob.glob(os.path.join(root, run, "final_result_*.csv")):
        for r in csv.reader(open(f)):
            if r and r[0].endswith(".mps"):
                out[r[0][:-4]] = (r[1].strip() in ("1", "1.0"), float(r[2]), r[-1].strip() in ("1", "1.0"))
    return out
def bks_gap(feas, p, b):
    if not feas: return 1.0
    if abs(p - b) <= 1e-9 * max(1.0, abs(b)): return 0.0
    if p * b < 0: return 1.0
    return abs(p - b) / max(abs(p), abs(b))
res = {}
print(f"{'run':10s} {'MIPFeas all':>11s} {'valid':>8s} {'GAMS-rules all':>14s} {'valid':>8s} {'feas':>5s} {'at BKS':>6s} {'BKS gm':>7s} {'optimal':>7s}")
for lab, run in runs.items():
    L = {i: lenient(inc(run, i, False), z[i]) for i in z}
    Lv = {i: lenient(inc(run, i, True), z[i]) for i in z}
    Ga = {i: gams_pi(gams_rows(run, i, False), z[i], f"{i}_gb10_cuopt-26-10.trc")[0] for i in z}
    Gv = {i: gams_pi(gams_rows(run, i, True), z[i], f"{i}_gb10_cuopt-26-10.trc")[0] for i in z}
    fr = final(run); g = {i: bks_gap(*fr[i][:2], z[i]) for i in z if i in fr}
    res[lab] = L
    print(f"{lab:10s} {sgm(L.values()):11.5f} {sgm(Lv.values()):8.5f} {sgm(Ga.values()):14.5f} {sgm(Gv.values()):8.5f} {sum(fr[i][0] for i in z if i in fr):5d} "
          f"{sum(v <= 1e-4 for v in g.values()):6d} {math.exp(st.mean(math.log1p(v) for v in g.values())) - 1:7.4f} {sum(fr[i][2] for i in fr):7d}")
def wl(a, b, sel):
    w = sum(res[b][i] < res[a][i] - max(0.0005, 0.05 * res[a][i]) for i in sel); l = sum(res[b][i] > res[a][i] + max(0.0005, 0.05 * res[a][i]) for i in sel); return w, l
def sign_p(w, l):
    n = w + l; k = max(w, l)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n) if n else 1.0
subsets = {"all 233": list(z), "fix changes >=50% of FP time": [i for i in z if aff.get(i, -1) >= 50], "fix changes <5%": [i for i in z if 0 <= aff.get(i, -1) < 5]}
print("\nper-instance MIPFeas P (lenient, all incumbents), better/worse by > max(0.0005, 5%):")
for name, sel in subsets.items():
    w, l = wl("base", "fix", sel)
    line = f"  {name:32s} n={len(sel):3d}  fix vs base: better {w:3d} worse {l:3d} (sign p={sign_p(w, l):.3f})"
    if "noise_a" in res: nw, nl = wl("noise_a", "noise_b", sel); line += f" | noise (same commit, other setting): {nw}/{nl}"
    print(line)
for name, sel in list(subsets.items())[1:2]:
    print(f"  sgm on {name}: base {sgm([res['base'][i] for i in sel]):.5f}  fix {sgm([res['fix'][i] for i in sel]):.5f}")
d = sorted(((res["fix"][i] - res["base"][i], i) for i in z))
print("largest fix gains:", ", ".join(f"{i} {res['base'][i]:.4f}->{res['fix'][i]:.4f}" for _, i in d[:6]))
print("largest fix losses:", ", ".join(f"{i} {res['base'][i]:.4f}->{res['fix'][i]:.4f}" for _, i in d[::-1][:6]))
