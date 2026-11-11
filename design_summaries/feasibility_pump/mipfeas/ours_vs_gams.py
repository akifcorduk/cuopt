import csv, math, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gams_pi_port import read_trc
gd, ours = sys.argv[1], sys.argv[2:]
z = {}
for r in csv.DictReader(open(os.path.join(gd, "optimal_objective.csv"))):
    try: z[r["InstanceInst."]] = float(r["ObjectiveObje."])
    except ValueError: pass
T = 600.0
def lenient(pts, zz):
    area, tp, pp = 0.0, 0.0, 2.0
    for t, o in sorted(pts):
        if t >= T: break
        g = 1.0 if o * zz < 0 else abs(o - zz) / max(abs(o), abs(zz), 1.0)
        if g < pp: area += (t - tp) * pp; tp, pp = t, g
    return (area + (T - tp) * pp) / T
def t_first(pts): return min((t for t, _ in pts), default=T)
sgm = lambda v: math.exp(st.mean(math.log(x + 0.001) for x in v)) - 0.001
data = {}
for plat in ("gb10", "b200"):
    P, F = {}, {}
    for i in z:
        rows = read_trc(os.path.join(gd, "trc_files", f"{i}_{plat}_cuopt-26-10.trc"))
        pts = [(r["seconds"], r["bestFound"]) for r in rows if not math.isnan(r["bestFound"])]
        P[i], F[i] = lenient(pts, z[i]), t_first(pts)
    data[f"gams-{plat}"] = (P, F)
for a in ours:
    lab, d = a.split("=", 1); d = os.path.join(d, os.path.basename(d))
    P, F = {}, {}
    for i in z:
        p = os.path.join(d, i + "_incumbents.csv")
        pts = [(float(r["wall_time_s"]), float(r["objective"])) for r in csv.DictReader(open(p))] if os.path.exists(p) else []
        P[i], F[i] = lenient(pts, z[i]), t_first(pts)
    data[lab] = (P, F)
print(f"{'run':10s} {'sgm PI':>8s} {'median t_first':>15s} {'sgm(t_first+1)-1':>17s}")
for lab, (P, F) in data.items():
    print(f"{lab:10s} {sgm(P.values()):8.5f} {st.median(F.values()):15.3f} {math.exp(st.mean(math.log(x + 1) for x in F.values())) - 1:17.3f}")
b, g = data["base0"][0], data["gams-b200"][0]
ratio = sorted(((math.log((b[i] + 0.001) / (g[i] + 0.001)), i) for i in z), reverse=True)
print("instances where base0 is much worse than GAMS b200:", ", ".join(f"{i} {b[i]:.4f}/{g[i]:.4f} (t1 {data['base0'][1][i]:.1f}s/{data['gams-b200'][1][i]:.1f}s)" for _, i in ratio[:8]))
print("instances where base0 is much better:", ", ".join(f"{i} {b[i]:.4f}/{g[i]:.4f}" for _, i in ratio[::-1][:5]))
n_w = sum(x > math.log(1.5) for x, _ in ratio); n_b = sum(x < -math.log(1.5) for x, _ in ratio)
print(f"base0 vs GAMS b200 (factor 1.5 on PI+0.001): worse on {n_w}, better on {n_b}")
