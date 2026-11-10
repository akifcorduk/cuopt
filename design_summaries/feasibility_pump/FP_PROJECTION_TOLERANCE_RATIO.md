# Feasibility pump — projection tolerance from the integer ratio

Fix branch: `fix/fp-assigned-integers-ratio` (`66bf34c1b` on upstream main `0c90faa1a`).
Bench: `bench/fp-ratio-fix` (`4465befc7`, same tree, parent `0c90faa1a`).
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

Keep the fix (it restores the documented intent and removes a silent no-op) pending the EOS
comparison below; decide on merging from its primal gap / primal integral. If EOS shows a loss, the
next question is the schedule itself (thresholds, or tightening when a descent stalls), not the
counter.

Related: on the PDFP experiment branch (`exp/fp-pdfp-trace`, e.g. `9126569d6`) both projectors take
their tolerance from this ratio (`get_tolerance_from_ratio` for PDFP and for the epigraph LP), so the
epigraph-vs-PDFP runs so far compared both projectors with the schedule switched off (1e-6, unless a
variant set its own tolerance).

## 6. EOS benchmark proposal (not launched; needs the user's approval)

- Hashes: `4465befc7` (`bench/fp-ratio-fix`) vs its parent `0c90faa1a` (upstream main), both on origin.
- Default parameters, full solver, 240 MIPLIB2017 instances, 600 s, `n_batches` 6, `runtime` 01:40:00.
- `config_eos.json` change: put the two hashes first in `commit_hashes`, `n_hashes_to_run: 2`,
  `n_iters: 2` (currently 1), `suffix_start_idx: 0`; everything else unchanged. 2 x 2 x 6 = 24 jobs.
  `n_iters` is re-read when each job starts, so restore it only after no job is PENDING.
- Compare: feasible count, BKS primal gap (geomean, BKS reached), primal integral, time to first
  solution; reps of the same hash first for noise. The bench commit has no FPTOL line, so FP-level
  numbers come from the dev-node runs above.
