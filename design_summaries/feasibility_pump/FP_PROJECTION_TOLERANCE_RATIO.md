# Feasibility pump — projection tolerance from the integer ratio

Fix branch: `fix/fp-assigned-integers-ratio` (`66bf34c1b` on upstream main `0c90faa1a`).
Bench (EOS, section 6): `bench/fp-ratio-fix-pin` (`37fbdbc4a`) vs `bench/fp-ratio-base-pin` (`32ba4c968`),
upstream main `0c90faa1a` with `RAPIDS_BRANCH=release/26.12`; `bench/fp-ratio-fix` (`4465befc7`) is the
unpinned original that could not run.
Dev-node experiment: `exp/fp-ratio-trace` (fix + instrumentation on `4b0767834`, see section 2).

## 1. Question

`run_single_fp_descent` picks the tolerance of each FP projection LP from the ratio of integral
integer variables (`get_tolerance_from_ratio`: < 0.80 -> 0.1, < 0.93 -> 0.01, < 0.97 -> 0.001,
else the absolute tolerance 1e-6). The ratio is meant to describe the previous projection, but FP
traces on `exp/fp-pdfp-trace` showed `tol=1e-06` for all 2,711 traced projections (gen-ip002,
timtab1, air05, bab2). Is the schedule dead, what should the ratio be, and what does restoring it do
to projection time and FP results?

### Root cause

The ratio read `solution.n_assigned_integers`. Since mid-2024 `solution_t::compute_feasibility()`
calls `compute_number_of_integers()`, so every rounding (`round_nearest` at descent start, the
`return sol.compute_feasibility()` of `constraint_prop_t::apply_round`) reset it to the count of
the rounded point, which is all integers. The ratio was therefore always 1 and every projection ran at
the absolute tolerance. In the original FP (May 2024, `abd1460be`) `compute_feasibility` did not
touch the counter, so the field still held the count that `compute_number_of_integers()` set right
after the previous projection: the intended semantics.

### Change

Carry the integer count of the previous projection through the descent loop in a local
(`n_integers`, already computed after each projection) and use it for the ratio. The first
projection of a descent starts from the nearest rounding, which is fully integral, so it uses the
absolute tolerance (unchanged).

Alternative considered: carry the last projection's ratio across descents (closer to the 2024
behavior, where the stale field survived restarts). A what-if over the epigraph EOS traces (below)
shows it differs from the local version on only 0.6% of projections, and a member would need
resetting around the staged FP's binary/integer switch, so the local version was kept.

### Expected size of the effect (what-if over existing traces)

Replaying the rule on the FPT traces of the epigraph GPU-heuristic-only EOS runs
(`benchmark_9126569d6_{0,1}`, 476 instance runs, 1,079,150 projections, all at 1e-6):

| tolerance after the fix | projections | share of projection time | capped at 1 s (old run) |
|---|---|---|---|
| 0.1 | 11.5% | 21.4% | 21,409 |
| 0.01 | 6.9% | 13.7% | 13,188 |
| 0.001 | 5.8% | 10.8% | 10,966 |
| 1e-6 (unchanged) | 75.8% | 54.0% | 86,687 |

49% of the projections are the first of their descent and keep 1e-6. 404 of the 476 runs would see
at least one looser projection; instances with the largest affected share include neos8, cod105,
bnatt400/500, gen-ip054, gen-ip002, timtab1 (> 94% of projection time). bab2 and rail507 stay at
1e-6 throughout (their projections are already >= 97% integral).

## 2. Setup (dev node)

- H200 NVL (4u8g-gen-0277), `cuopt-dev` container, CUDA 12.9, Release with assertions
  (`-DDEFINE_ASSERT=true`), `LIBCUOPT_LOGGING_LEVEL=INFO`, `solve_MIP` only.
- Base `4b0767834`, the last upstream main commit before #2236 moved to the cuDSS 1.0 API: the dev
  container only has cuDSS 0.7.1, so `0c90faa1a` does not configure there. The 13 later commits touch
  FP only by renaming two settings fields (#2216). EOS runs `mamba env update` from each commit's
  environment file, so it builds `0c90faa1a` and the bench commit.
- One binary, `exp/fp-ratio-trace` = fix + an `FPTOL` INFO line per projection (ratio, tolerance, PDLP
  time / iterations / termination status, integral count after the projection, alpha) + env switch
  `CUOPT_FP_RATIO_OLD=1`, which restores the old ratio (`solution.n_assigned_integers`). Arms:
  `new` (fix) and `old`.
- Default solver parameters (B&B + GPU heuristics), the EOS thread count:
  `solve_MIP --path <mps> --time-limit 120 --num-cpu-threads 10 --write-log-file t --log-to-console f`.
  Arms alternate per instance; the chain waits until no other process is on the GPU and records any
  co-tenant (the GPU is shared with other experiments).
- Scripts (untracked, dev node): `fpfix/build.sh`, `fpfix/ab_chain.sh <tag> <tl> "<arms>" inst...`;
  analysis: `fptol_analyze.py <miplib2017_bks.hpp> fpfix/runs_<tag> new old`.
- Instances: gen-ip002, timtab1, air05, bab2 (the traced set), cod105 and mas74 (large affected share
  in the what-if; mas74's projections mostly hit the 1 s cap). One 120 s run per arm, 2026-11-10;
  no GPU co-tenant in any of the 12 runs, no assertion failures.

## 3. Results (dev node, 120 s, default parameters, one run per arm)

| instance | arm | projections | per s | tol 0.1 / 0.01 / 0.001 / 1e-6 | mean LP ms | mean PDLP iters | hit 1 s cap | integral after proj. | final primal gap vs BKS |
|---|---|---|---|---|---|---|---|---|---|
| gen-ip002 | fix | 889 | 7.5 | 845 / 1 / 0 / 43 | 42 | 519 | 0.1% | 51% | 8e-8 (BKS) |
| gen-ip002 | old | 314 | 2.6 | 0 / 0 / 0 / 314 | 262 | 4,050 | 5.4% | 80% | 8e-8 (BKS) |
| timtab1 | fix | 596 | 5.5 | 556 / 18 / 0 / 22 | 37 | 418 | 0.3% | 53% | 6.05e-2 |
| timtab1 | old | 314 | 3.1 | 0 / 0 / 0 / 314 | 113 | 1,640 | 23.6% | 57% | 5.46e-2 |
| cod105 | fix | 1,429 | 12.0 | 1,123 / 212 / 19 / 75 | 24 | 143 | 0.1% | 47% | 0 |
| cod105 | old | 334 | 2.8 | 0 / 0 / 0 / 334 | 245 | 3,636 | 6.3% | 73% | 0 |
| mas74 | fix | 108 | 0.9 | 3 / 80 / 15 / 10 | 855 | 14,006 | 100% | 89% | 3.6e-7 |
| mas74 | old | 94 | 0.8 | 0 / 0 / 0 / 94 | 971 | 15,869 | 98.9% | 89% | 3.6e-7 |
| bab2 | fix | 21 | 0.6 | 0 / 0 / 0 / 21 | 741 | 6,930 | 100% | 99% | 0.369 |
| bab2 | old | 22 | 0.7 | 0 / 0 / 0 / 22 | 740 | 6,942 | 100% | 99% | 0.367 |
| air05 | fix | 7 | - | 0 / 0 / 0 / 7 | 806 | 11,783 | 43% | 99% | 0 (B&B, 18 s) |
| air05 | old | 26 | - | 0 / 0 / 0 / 26 | 222 | 3,082 | 3.8% | 100% | 0 (B&B, 18 s) |

Per tolerance level over all six instances (fix arm): 0.1 -> 2,527 projections, 28 ms mean / 24 ms
median, 260 iterations, 99.9% converged; 0.01 -> 311, median 28 ms (mean 259 ms, dominated by
mas74), 74% converged; 1e-6 -> 178, median 132 ms. Old arm, all 1e-6: 1,104 projections, 283 ms mean
/ 167 ms median, 4,280 iterations, 79% converged.

Share of the FP span spent inside projection LPs: gen-ip002 32% (fix) vs 69% (old), timtab1 20% vs
35%. Primal integral vs BKS over 120 s (seconds): gen-ip002 0.096 vs 0.108, timtab1 9.35 vs 7.32,
cod105 0.064 vs 0.026, mas74 0.164 vs 0.087, bab2 101.9 vs 101.7, air05 1.26 vs 1.38 (single runs).

## 4. What worked, what did not, and why

- The schedule is live again: on instances whose projections are far from integral (gen-ip002,
  timtab1, cod105) most projections run at 0.1, PDLP stops after 8-25x fewer iterations, each
  projection is 3-10x cheaper, almost none hit the 1 s cap, and FP does 1.8-4.3x more projections in
  the same time.
- Feedback loop: a 0.1 projection stops away from a vertex and is less integral (gen-ip002 51% vs
  80% at 1e-6, cod105 39% at 0.1 vs 73%), so the next ratio stays below 0.8 and the descent rarely
  tightens (gen-ip002: 845 of 889 projections at 0.1, one at 0.01). This is inherent to deriving the
  tolerance from the previous projection's integrality, as designed in 2024; it is the main risk the
  EOS run has to price in (more, cheaper but less integral projections).
- No effect where projections are already >= 97% integral (bab2, air05: every projection stays at
  1e-6 in both arms; the air05 time difference is trajectory noise, B&B solves it in 18 s) or where
  PDLP cannot converge the projection LP at any of these tolerances (mas74: 100% capped at 0.01 too).
- End-to-end in these single 120 s runs: same final solution on gen-ip002, cod105, mas74, air05; bab2
  equal within noise; timtab1 slightly worse with the fix (gap 6.05% vs 5.46%, primal integral 9.35 vs
  7.32). One run per arm is not enough to call quality either way; that is what the EOS run is for.

## 5. Decision

Keep the fix. It restores the documented intent, removes a silent no-op that dates from late 2024,
and the EOS comparison (section 7) shows a consistent primal-integral gain on the instances whose
FP behavior it changes, with final gap and feasibility unchanged. Open next: the feedback loop (loose
projections stay loose); the schedule itself (thresholds, tightening when a descent stalls) is the
lever if more is wanted, not the counter.

Related: on the PDFP experiment branch (`exp/fp-pdfp-trace`, e.g. `9126569d6`) both projectors take
their tolerance from this ratio (`get_tolerance_from_ratio` for PDFP and for the epigraph LP), so the
epigraph-vs-PDFP runs so far compared both projectors with the schedule switched off (1e-6, unless a
variant set its own tolerance).

## 6. EOS run (approved 2026-11-11)

- First submission `4465befc7` vs `0c90faa1a` (jobs 6510298-6510321) produced nothing: both arms built,
  then `solve_MIP` died at load with `libcuopt_mathopt.so: undefined symbol
  _ZTIN3rmm9_RMM_27_29bad_allocE`. RAPIDS main moved to 27.02 that day; cuOpt is at 26.12 with
  `RAPIDS_BRANCH=main`, so CPM fetched rmm 27.02 while `mamba env update` installed librmm 26.12. Not
  related to the fix; it hits every commit with `RAPIDS_BRANCH=main`.
- Rerun with `RAPIDS_BRANCH=release/26.12` in both arms (build-only, as upstream #1870 did for 26.10):
  base `32ba4c968` (`bench/fp-ratio-base-pin` = `0c90faa1a` + pin), fix `37fbdbc4a`
  (`bench/fp-ratio-fix-pin` = base + the 4-line fix). Canary 6510443, then 23 jobs up to 6510531; 2 reps,
  240 instances, 600 s, default parameters, 8 instances per H100 node at a time. All 24 jobs clean
  (6/6 result files, 240/240 logs per run). Results in `~/Documents/benchmarks/benchmark_{32ba4c968,37fbdbc4a}_{0,1}`.

## 7. EOS results

Primal gap vs the repo BKS (233 instances with a BKS):

| run | feasible | at BKS | geomean(gap+1)-1 | primal integral sum | sgm10 PI | median PI |
|---|---|---|---|---|---|---|
| base rep 0 | 231 | 128 | 0.0376 | 10,684 | 15.90 | 5.67 |
| base rep 1 | 231 | 128 | 0.0403 | 10,814 | 15.55 | 5.77 |
| fix rep 0 | 231 | 131 | 0.0395 | 11,367 | 15.59 | 4.98 |
| fix rep 1 | 231 | 131 | 0.0419 | 11,394 | 15.06 | 4.72 |

The fix's worse sum and geomean come from three all-or-nothing instances: academictimetablesmall and
rd-rplusc-21 (a solution in 1 of 2 base reps, 0 of 2 fix reps, 600 gap-seconds each) and cbs-cta (BKS 0,
so any nonzero objective counts as gap 1; one fix rep missed 0). ns1952667 flips the other way (1 of 2
base reps vs 2 of 2 fix reps). Solver-gap view (compare.py): 231 feasible and 75-76 optimal in every
run; fix vs base differences match base vs base.

On the 222 stable instances (a solution in all four runs, BKS != 0), rep-averaged primal integral,
better/worse by more than max(1 s, 5%):

| subset | n | sgm10 PI base (rep 0 / 1) | sgm10 PI fix (rep 0 / 1) | fix better / worse | sign test |
|---|---|---|---|---|---|
| all stable | 222 | 14.39 / 13.86 | 13.42 / 13.21 | 58 / 32 | p = 0.008 |
| fix changes >= 50% of FP projection time | 107 | 10.91 / 10.24 | 9.43 / 9.70 | 27 / 11 | p = 0.014 |
| fix changes < 5% of FP projection time | 69 | 20.12 / 19.99 | 20.33 / 19.91 | 16 / 16 | p = 1.0 |

The subsets come from the what-if on the epigraph GPU-only traces (section 1). Single-rep noise for
the same comparison (base rep 1 vs rep 0) is 52 / 49 on the stable set. Final gap on the stable set is
neutral (geomean 0.0278 / 0.0305 fix vs 0.0295 / 0.0291 base; at BKS 125 / 126 vs 124 / 120) and the
time to the first solution is unchanged (shifted geomean 0.74-0.76 s in every run; first incumbents
come from FJ and the early heuristics, before FP).

MIPFeas score (plato.asu.edu/ftp/mipfeas.html, GAMS blog of Oct 2026): P = (1/600) * integral over
[0, 600 s] of p(t), with p = 2 before the first incumbent, 1 on a sign mismatch, else the relative gap to
the optimal / best-known value; shifted geometric mean with shift 0.001 over the 233 feasible
instances (repo BKS as the reference). The page writes the gap denominator as max(|z|, |z*|, 1), the
blog as max(|z|, |z*|) with 0 when both are below 1e-6; both are shown. MIPFeas scores the incumbents
the solver reports; the solve_MIP trace also re-checks each incumbent (`valid`), and 18-22 instances
per run have some incumbents flagged invalid in both arms (rocI-4-11 all of them; cbs-cta in fix rep 0
loses its optimal incumbents that way), so both counts are shown:

| incumbents | gap denominator | base rep 0 / 1 | fix rep 0 / 1 | fix vs base (mean of reps) |
|---|---|---|---|---|
| all reported (MIPFeas) | max(\|z\|,\|z*\|,1) | 0.01350 / 0.01334 | 0.01259 / 0.01252 | -6.4% |
| all reported (MIPFeas) | max(\|z\|,\|z*\|) | 0.01429 / 0.01412 | 0.01337 / 0.01325 | -6.3% |
| valid only | max(\|z\|,\|z*\|,1) | 0.01378 / 0.01364 | 0.01328 / 0.01277 | -5.0% |
| valid only | max(\|z\|,\|z*\|) | 0.01459 / 0.01443 | 0.01410 / 0.01350 | -4.9% |

Feasible found: 231 per run counting all incumbents, 230 valid only (rocI-4-11). Rep-to-rep spread is
1.0-1.2% (base) and 0.6-4% (fix); both fix reps beat both base reps in every variant. The absolute values
are not comparable with the MIPFeas table (cuopt 0.0088 on a DGX Spark GB10, one instance at a time):
these runs share an H100 node 8 ways with 10 CPU threads each.

Check against GAMS' own data (2026-11-11). GAMS' archive `2026-10-30_mipfeas_results_and_scripts.zip`
(cloud.gams.com/s/oQBzPAXDdzcKDNr, 20.9 MB: `create_primalintegral.py`, `optimal_objective.csv`,
`primalintegral.csv`, 1,631 `.trc` traces) reproduces exactly: a port of their `calculate_primal_integral`
(`mipfeas/gams_pi_port.py`) gives every published geomean on their traces (cbc 0.0980, cuopt 0.0088,
cuopt-b200 0.0076, highs 0.0447, rexi 0.0103, scip 0.0609, scipconc 0.0168; per instance within 1e-19 of
their csv), and the scorer above gives the same numbers on their traces (no instance off by more than
1e-4). Their reference values equal the repo BKS on all 233 instances. What differs for our runs:

- GAMS runs every solver with `optcr=0` (mipfeas.gms). These runs use cuOpt's default
  `relative_mip_gap` 1e-4, so B&B stops early on a near-optimal objective on 4-6 instances per run
  ("Optimal solution found within relative MIP gap tolerance", e.g. neos-957323, radiationm40-10-02,
  neos-827175). GAMS' scorer counts an early stop whose objective is not z as a failure (P = 2); the
  scorer above charged only the residual gap.
- GAMS also scores an incumbent better than z as P = 2 (whitelisted for cuOpt only on
  neos-3754480-nidda). solve_MIP's `valid` flag already drops those (supportcase26,
  neos-4722843-widden, binkar10_1), which is why the valid-only rows are close.

With GAMS' rules applied to these runs (`mipfeas/gams_step_c.py`; I rows = running best, E row at
`run_solver`, GAMS' cuOpt whitelists for both arms):

| incumbents | base rep 0 / 1 | fix rep 0 / 1 | fix vs base | P = 2 failures per run |
|---|---|---|---|---|
| all reported | 0.01680 / 0.01559 | 0.01527 / 0.01429 | -8.7% | 8 / 6 vs 8 / 6 |
| valid only | 0.01587 / 0.01524 | 0.01545 / 0.01392 | -5.6% | 5 / 4 vs 7 / 4 |

The fix still wins, with more rep noise (4-8%) because the failure rules turn sub-1e-4 endings into
P = 2. None of this is comparable with the published cuOpt numbers: on one scorer these upstream-main
EOS runs score 0.0135 vs GAMS' cuopt 26.10 0.0088 (GB10) / 0.0076 (B200), worse on 106 instances and
better on 6 (`mipfeas/ours_vs_gams.py`), with later first incumbents (shifted geomean 0.95 s vs
0.67-0.71 s; app1-2, rd-rplusc-21, milo-v12-6-r2-40-1 are the largest). Gap setting, hardware (H100
nodes shared 8 ways), and version (26.12 dev vs 26.10) are not separated yet.

Resource setting (2026-11-11, user request): baseline `32ba4c968` rerun as rep 2 with 24 CPU threads and 4
instances per H100 node (12 batches of 20; EOS script copies `run_script_g4t24.sh` /
`benchmark_script_g4t24.sh`, jobs 6511464-6511475), results `benchmark_32ba4c968_2`:

| run | threads, instances per node | MIPFeas sgm, GAMS integration without failure rules | GAMS rules, all / valid | first incumbent, sgm(t+1)-1 | optimal (solver) |
|---|---|---|---|---|---|
| base rep 0 / 1 | 10, 8 | 0.01350 / 0.01334 | 0.0168 / 0.0159, 0.0156 / 0.0152 | 0.95 s | 76 / 75 |
| base rep 2 | 24, 4 | 0.00891 | 0.0115 / 0.0108 | 0.80 s | 89 |
| GAMS cuopt 26.10 GB10 / B200 | 10 / 24, 1 | 0.00879 / 0.00755 | same | 0.67 / 0.71 s | - |

Per instance (factor 1.5 on P + 0.001): rep 2 is better than rep 0 on 76 and worse on 7 (rep 1 vs rep 0:
23 / 16); vs GAMS' B200 the count goes from worse 106 / better 6 to worse 48 / better 16. The largest
gains are the instances furthest behind GAMS (rd-rplusc-21, milo-v12-6-r2-40-1, app1-2, supportcase10,
rmatr200-p5, uccase12). So most of the distance to the published cuOpt numbers is the benchmark's 10
threads with 8 instances per node, not the code; threads and node sharing are not separated (both
changed). The fix A/B above was measured at 10 threads / 8 per node.

Fix vs base at 24 threads / 4 instances per node (one run per arm, suffix `_2`; fix jobs 6511998-6512009,
`benchmark_37fbdbc4a_2`; `mipfeas/ab_pair.py`):

| | base `_2` (1 run) | fix `_2` (1 run) |
|---|---|---|
| MIPFeas sgm, all / valid incumbents | 0.00891 / 0.00912 | 0.00895 / 0.00945 |
| GAMS rules, all / valid | 0.01146 / 0.01075 | 0.01123 / 0.01096 |
| feasible / at BKS / BKS gap geomean | 232 / 143 / 0.0265 | 231 / 142 / 0.0290 |
| proven optimal (solver) | 89 | 83 |

Per instance (MIPFeas P, better/worse by more than max(0.0005, 5%)): all 233 instances 59 / 56 (sign test
p = 0.85); where the fix changes >= 50% of FP projection time 30 / 19 (p = 0.15, sgm 0.00550 -> 0.00536);
where it changes < 5%, 20 / 22. Two identical runs (base rep 0 vs 1, other setting) differ by 58 / 64.
Largest fix gains academictimetablesmall, rail01, rail02, bab6; largest losses ns1952667 (no solution in
the fix run), supportcase10, ns1644855, rocII-5-11.

Reading: in the GAMS-like resource setting the fix is neutral within single-run noise, with at most a
weak positive signal where it changes FP. The 10-thread / 8-per-node A/B (two reps per arm) showed a
clear anytime gain; with 24 threads the CPU side (B&B, CPU FJ) finds and improves solutions sooner,
which plausibly leaves FP less to contribute. More reps are needed to size the effect in this setting.

Final-solution side at 24 threads (`mipfeas/opt_check.py`): the fix lost 6 proven optima and gained none
(10 threads: 3 / 3 per rep; identical runs 3 / 4). Four of the six also flip between identical 10-thread
runs (graphdraw-domain, neos-1171737, ns1952667, unitcal_7); mzzv11 and tbfp-network were never proven at
10 threads (base `_2` took 385 s and 485 s). On the 83 instances both runs prove, the fix is not slower
(sgm 43.3 s -> 40.1 s, 25 faster / 17 slower by > 10%). Gap to BKS leans to the base in all three
pairings (0.0376 / 0.0403 vs 0.0395 / 0.0419 at 10 threads, 0.0265 vs 0.0290 at 24), each within the 6-7%
identical runs differ by. Aggregate MIPFeas differs by 0.6-1.2% between identical 10-thread runs, so the
6% anytime gain would likely have shown in one 24-thread run; a 1-2% effect cannot be resolved. The
largest 24-thread swings are mostly bimodal instances (academictimetablesmall, ns1952667, supportcase10,
ns1644855 swing as much between identical runs); rail02 is the only large gain that holds in all three
pairings.

Run layout and configuration (checked 2026-11-11 from the logs, the EOS scripts and sacct). Each commit
has three runs and the suffix is a run index, not a count: `_0` and `_1` are two reps at 10 threads / 8
instances per node (6 jobs each; all 24 jobs of both arms started 01:57-02:05 PST, interleaved); `_2` is
one run at 24 threads / 4 per node (12 jobs; base 04:11-05:01, fix 05:22-06:11 PST, so sequential and on
24 different nodes). Within each setting the arms differ only by the fix (`37fbdbc4a` = `32ba4c968` + the
4-line change, both pinned to release/26.12): every log shows the same 240 instances, the 600 s limit,
default gaps (relative 1e-4), no deterministic mode, the setting's thread count and GPU ids, and an
8480C + H100 80GB node; nodes are exclusive (OverSubscribe=EXCLUSIVE, 224 CPUs per job); all 36 jobs
resolved the same conda transaction (cuDSS 1.0.0.35) and CPM dependencies (CCCL 3.6.0, GTest 1.16.0,
KaMinPar 3.7.3). The `_g4t24` scripts differ from the standard ones only in `--num-cpu-threads 24` and
`GPU_COUNT=4`.

Reading: the gain is in anytime quality, about -9% sgm PI where the fix changes FP and nothing where
it does not, which is the pattern expected from cheaper projections and more FP iterations. It does
not move the final 600 s gap or feasibility beyond rep noise.
