"""Own port of GAMS' calculate_primal_integral (MIPFeas create_primalintegral.py, 2026-10-30), no external code.
Trace rows: lineNum, seriesID, node, seconds, bestFound, bestBound; 'na'/'inf' -> NaN."""
import math, os

NA = {"", "na", "inf", "NA", "NaN", "nan", "N/A", "n/a", "null", "NULL", "#N/A", "<NA>", "None"}
BETTER_OK = {  # GAMS whitelist: incumbent better than z is clamped to z
    'neos-3754480-nidda_gb10_cuopt-26-10.trc', 'neos-3754480-nidda_b200_cuopt-26-10.trc',
    'mas74_gb10_rexi10-26-10.trc', 'rmatr100-p10_gb10_rexi10-26-10.trc', 'neos-3754480-nidda_gb10_rexi10-26-10.trc',
    'binkar10_1_gb10_rexi10-26-10.trc', 'ns1830653_gb10_rexi10-26-10.trc', 'supportcase26_gb10_rexi10-26-10.trc',
    'trento1_gb10_rexi10-26-10.trc', 'neos-5107597-kakapo_gb10_rexi10-26-10.trc', 'sct2_gb10_rexi10-26-10.trc',
    'pk1_gb10_rexi10-26-10.trc', 'istanbul-no-cutoff_gb10_rexi10-26-10.trc', 'fastxgemm-n2r6s0t2_gb10_rexi10-26-10.trc',
    'rmatr200-p5_gb10_rexi10-26-10.trc', 'neos-3754480-nidda_gb10_scipconc-soplex-11-0-0-dc08995a27.trc',
    'neos-4338804-snowy_gb10_scipconc-soplex-11-0-0-dc08995a27.trc', 'pk1_gb10_scipconc-soplex-11-0-0-dc08995a27.trc',
    'mas74_gb10_scipconc-soplex-11-0-0-dc08995a27.trc', 'mas76_gb10_scipconc-soplex-11-0-0-dc08995a27.trc',
    'neos-5107597-kakapo_gb10_scipconc-soplex-11-0-0-dc08995a27.trc', 'neos-3402294-bobin_gb10_scipconc-soplex-11-0-0-dc08995a27.trc',
    'neos-2978193-inde_gb10_scipconc-soplex-11-0-0-dc08995a27.trc', 'fastxgemm-n2r6s0t2_gb10_scipconc-soplex-11-0-0-dc08995a27.trc'}
EARLY_OK = {  # GAMS whitelist: early termination with a non-optimal incumbent is accepted
    'neos-5188808-nattai_b200_cuopt-26-10.trc', 'neos-5188808-nattai_gb10_cuopt-26-10.trc', 'neos-5188808-nattai_gb10_rexi10-26-10.trc',
    'neos-5188808-nattai_gb10_xpress-9-9-1.trc', 'neos-5188808-nattai_gb10_cplex-22-2-0-0.trc', 'neos-5188808-nattai_gb10_scip-soplex-10-1-0.trc',
    'neos-5188808-nattai_gb10_copt-8-0-7.trc', 'neos-5107597-kakapo_gb10_cbc-2-10-13.trc', 'mas74_b200_cuopt-26-10.trc'}

def num(tok):
    tok = tok.strip()
    return float("nan") if tok in NA else float(tok)

def read_trc(path):
    rows = []
    for line in open(path):
        if line.lstrip().startswith("*") or not line.strip(): continue
        f = [x.strip() for x in line.split(",")]
        rows.append({"lineNum": num(f[0]), "seriesID": f[1], "seconds": num(f[3]), "bestFound": num(f[4]), "bestBound": num(f[5])})
    return rows

def gams_pi(rows, z, basename, max_time=600.0, why=None):
    """Returns (P, optimal). rows: dicts with seriesID, seconds, bestFound, bestBound."""
    p_last = 2.0
    if not rows: return 2.0, False
    if not any(r["seriesID"] == "E" for r in rows):
        e = dict(rows[-1]); e["seriesID"] = "E"; e["seconds"] = max_time; rows = rows + [e]
    pi, t_last, obj_last, bnd_last, obj = 0.0, 0.0, float("inf"), float("-inf"), float("nan")
    for r in rows:
        t = min(r["seconds"], max_time)
        if t < t_last:
            if why is not None: why.append(f"time not monotone {t_last} > {t}")
            return 2.0, False
        obj = r["bestFound"]; bnd_last = r["bestBound"]; series = r["seriesID"]
        pi += (t - t_last) * p_last
        t_last = t
        if not math.isnan(obj):
            if round((obj_last - obj) / max(abs(obj), abs(obj_last), 1), 6) < 0:
                if why is not None: why.append(f"incumbent not monotone {obj_last} < {obj}")
                return 2.0, False
            if round((obj - z) / max(abs(obj), abs(z), 1), 6) < 0:
                if basename in BETTER_OK: obj = z
                else:
                    if why is not None: why.append(f"incumbent {obj} better than z {z}")
                    return 2.0, False
            obj_last = obj
            gamma = 1.0 if obj * z < 0 else round((obj - z) / max(abs(obj), abs(z), 1), 6)
            p_last = gamma
        if series == "E" and p_last < 2 and t_last < max_time - 1 and round((obj - z) / max(abs(obj), abs(z), 1), 6) > 0:
            if not (("_cbc-2-10-11.trc" in basename) and (t_last > max_time - 30)) and basename not in EARLY_OK:
                if why is not None: why.append(f"early termination at {t_last:.1f}s, not optimal: {obj} vs z {z}")
                return 2.0, False
    if p_last < 2 and not math.isnan(bnd_last):
        gap = round((obj_last - bnd_last) / max(abs(obj), abs(bnd_last), 1), 6)
    else:
        gap = 1.0
    pi += (max_time - t_last) * p_last
    return pi / max_time, gap < 1e-6
