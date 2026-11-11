"""MIPFeas-style primal integral (plato.asu.edu/ftp/mipfeas.html + GAMS blog, Oct 2026).
P = (1/T) int_0^T p(t) dt, T = 600 s; p = 2 before the first incumbent, 1 on a sign mismatch, else the
relative gap to the optimal / best-known value. Aggregate: shifted geometric mean, shift 0.001.
Variant 'page': gap = |z - z*| / max(|z|, |z*|, 1).  Variant 'blog': gap = |z - z*| / max(|z|, |z*|),
0 when both |z|, |z*| < 1e-6.
usage: [ALL=1] mipfeas_pi.py BKS_HPP LABEL=RUN_DIR ...   (RUN_DIR holds <inst>_incumbents.csv, or RUN_DIR/<basename>)
ALL=1 counts every reported incumbent (MIPFeas uses the solver trace); default: only valid ones."""
import csv, math, os, re, statistics as st, sys
T = 600.0
text = open(sys.argv[1]).read()
s, e = text.index("kBenchmarkBKS()"), text.index("kBenchmarkInfeasible()")
bks = {k.lower(): float(v) for k, v in re.findall(r'\{"([^"]+)",\s*([-+0-9.eE]+)\}', text[s:e])}

def gap(z, zs, variant):
    if z * zs < 0: return 1.0
    if variant == "page":
        return abs(z - zs) / max(abs(z), abs(zs), 1.0)
    if abs(z) < 1e-6 and abs(zs) < 1e-6: return 0.0
    return abs(z - zs) / max(abs(z), abs(zs))

def score(path, zs, variant):
    if not os.path.exists(path): return 2.0, False
    pts = sorted((float(r["wall_time_s"]), gap(float(r["objective"]), zs, variant))
                 for r in csv.DictReader(open(path)) if os.environ.get("ALL") == "1" or r["valid"] == "1")
    pts = [(min(max(t, 0.0), T), g) for t, g in pts if t < T]
    if not pts: return 2.0, False
    area, t_prev, p_prev = 0.0, 0.0, 2.0
    for t, g in pts:
        if g < p_prev:
            area += (t - t_prev) * p_prev
            t_prev, p_prev = t, g
    area += (T - t_prev) * p_prev
    return area / T, True

def sgm(v, sh=0.001): return math.exp(st.mean(math.log(x + sh) for x in v)) - sh

runs = []
for arg in sys.argv[2:]:
    label, d = arg.split("=", 1)
    nested = os.path.join(d, os.path.basename(os.path.normpath(d)))
    runs.append((label, nested if os.path.isdir(nested) else d))
insts = sorted(bks)
for variant in ("page", "blog"):
    print(f"variant={variant}  ({len(insts)} instances, T={T:.0f} s, shift 0.001)")
    print(f"  {'run':8s} {'sgm(0.001)':>10s} {'scaled':>7s} {'arith':>7s} {'median':>8s} {'feasible':>8s}")
    res = {}
    for label, d in runs:
        vals, feas = [], 0
        for i in insts:
            p, f = score(os.path.join(d, i + "_incumbents.csv"), bks[i], variant)
            vals.append(p); feas += f
        res[label] = (vals, feas)
    best = min(sgm(v) for v, _ in res.values())
    for label, (vals, feas) in res.items():
        print(f"  {label:8s} {sgm(vals):10.5f} {sgm(vals) / best:7.3f} {st.mean(vals):7.4f} {st.median(vals):8.5f} {feas:8d}")
    arms = {}
    for label, (vals, _) in res.items(): arms.setdefault(label.rstrip("0123456789"), []).append(sgm(vals))
    print("  per arm (mean of the reps' sgm): " + ", ".join(f"{a} {st.mean(v):.5f}" for a, v in arms.items()))
