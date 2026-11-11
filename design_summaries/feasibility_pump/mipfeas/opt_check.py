"""Optimality churn, solve time on instances optimal in both runs, and per-run MIPFeas P of the largest
swings across the six FP-ratio EOS runs. usage: opt_check.py GAMS_DIR"""
import csv, glob, os, re, math, statistics as st, sys
root = os.path.expanduser('~/Documents/benchmarks')
gd = sys.argv[1]
B0, B1, F0, F1, B2, F2 = '32ba4c968_0', '32ba4c968_1', '37fbdbc4a_0', '37fbdbc4a_1', '32ba4c968_2', '37fbdbc4a_2'
runs = [B0, B1, F0, F1, B2, F2]
z = {}
for r in csv.DictReader(open(os.path.join(gd, 'optimal_objective.csv'))):
    try: z[r['InstanceInst.']] = float(r['ObjectiveObje.'])
    except ValueError: pass
def load(r):
    d = os.path.join(root, f'benchmark_{r}')
    res = {}
    for f in glob.glob(os.path.join(d, 'final_result_*.csv')):
        for row in csv.reader(open(f)):
            if row and row[0].endswith('.mps'):
                res[row[0][:-4]] = dict(feas=row[1].strip() in ('1', '1.0'), obj=float(row[2]), gap=float(row[6]), opt=row[-1].strip() in ('1', '1.0'))
    for i, v in res.items():
        m = re.search(r'^run_solver (\d+)', open(os.path.join(d, f'benchmark_{r}', i + '.log'), errors='replace').read(), re.M)
        v['t'] = int(m.group(1)) / 1000 if m else float('nan')
        p = os.path.join(d, f'benchmark_{r}', i + '_incumbents.csv')
        pts = sorted((float(x['wall_time_s']), float(x['objective'])) for x in csv.DictReader(open(p))) if os.path.exists(p) else []
        v['first'] = pts[0][0] if pts else float('nan')
        if i in z:
            area, tp, pp = 0.0, 0.0, 2.0
            for t, o in pts:
                if t >= 600: break
                g = 1.0 if o * z[i] < 0 else abs(o - z[i]) / max(abs(o), abs(z[i]), 1.0)
                if g < pp: area += (t - tp) * pp; tp, pp = t, g
            v['P'] = (area + (600 - tp) * pp) / 600
    return res
R = {r: load(r) for r in runs}
sgm = lambda v, sh: math.exp(st.mean(math.log(x + sh) for x in v)) - sh
def cell(r, i):
    v = R[r][i]
    return ('OPT %4.0fs' % v['t']) if v['opt'] else ('no sol' if not v['feas'] else 'gap %.3f' % v['gap'])
lost = sorted(i for i in R[B2] if R[B2][i]['opt'] and not R[F2][i]['opt'])
print('== optimal in base _2, not in fix _2 (status per run; 10 thr: base _0 _1, fix _0 _1 | 24 thr: base _2, fix _2)')
for i in lost:
    print(f'  {i:26s} ' + ' | '.join(cell(r, i) for r in runs))
print('\n== optimality churn and solve time on instances optimal in both runs (sgm shift 10 s)')
for a, b, label in [(B0, B1, 'noise 10 thr: base _0 vs _1'), (F0, F1, 'noise 10 thr: fix _0 vs _1'),
                    (B0, F0, 'fix vs base 10 thr rep 0'), (B1, F1, 'fix vs base 10 thr rep 1'), (B2, F2, 'fix vs base 24 thr')]:
    both = [i for i in R[a] if R[a][i]['opt'] and R[b][i]['opt']]
    ta, tb = [R[a][i]['t'] for i in both], [R[b][i]['t'] for i in both]
    faster = sum(R[b][i]['t'] < 0.9 * R[a][i]['t'] for i in both)
    slower = sum(R[b][i]['t'] > 1.1 * R[a][i]['t'] for i in both)
    g = sum(R[b][i]['opt'] and not R[a][i]['opt'] for i in R[a]); l = sum(R[a][i]['opt'] and not R[b][i]['opt'] for i in R[a])
    print(f'  {label:30s} optimal gained {g} lost {l} | both optimal n={len(both)} time sgm {sgm(ta, 10):6.1f}s -> {sgm(tb, 10):6.1f}s, 2nd faster/slower by >10%: {faster}/{slower}')
print('\n== largest 24-thread swings: MIPFeas P (first incumbent s) per run')
for i in ['academictimetablesmall', 'rail01', 'rail02', 'bab6', 'ns1952667', 'supportcase10', 'ns1644855', 'rocII-5-11']:
    print(f'  {i:24s} ' + ' | '.join('%.3f (%5.1f)' % (R[r][i].get('P', float('nan')), R[r][i]['first']) for r in runs))
