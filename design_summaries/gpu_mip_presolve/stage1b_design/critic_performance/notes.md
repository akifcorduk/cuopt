# critic_performance: notes

Read-only on the worktree. Everything built or written is under this directory.

## What was read

- design_summaries/gpu_mip_presolve/DESIGN.md (all), results/stage0/STAGE0_REPORT.md (all),
  microbench/README_sweeps.md, README_fixed_costs.md, microbench/sweeps.cu 226-470.
- include/gprs/state.hpp, rules.hpp, presolve.hpp, numerics.hpp, postsolve.hpp;
  src/presolve.cpp, rules_propagation.cpp, rules_bounds.cpp, state_build.cpp 55-350,
  state_ops.cpp 1-260; tools/gprs_presolve.cpp; build_host_local.sh; CMakeLists.txt.
- Stored numbers: results/stage0/fixed_costs.json, results/stage0/sweeps/*.json and
  sweeps_large/*.json (matrix shapes, best round per matrix),
  results/stage1/host/v3/release_dev and release_large (40 instances, per-rule seconds, work, calls).

## What was measured here (probe/)

A copy of include/, src/ and tools/gprs_presolve.cpp with a logging probe (probe/src/probe.cpp and
a patch of the copied presolve.cpp), built with the Mac compiler into probe/build, run on the 17
CSR dumps in gpu_presolve/local_dumps (15 testbed instances and the two small status instances).
Counts are machine independent; the seconds are Mac seconds and are used only as ratios.
Per propagation call the probe prints the list the call starts from (design's F, the host's own
work for that list, entries), the host's work and bound changes, and it replays on a copy of the
state the rounds-from-one-snapshot scheme of the design, counting what the design charges per
round (front rows and entries, candidate columns and their entries, changed columns).
Output per instance: probe/out/<instance>.txt. Summary script: probe/summarize.py.

## Numbers used in the report

Stored node results (v3, minimum of the stored repetitions), 40 instances:
- total 75.0 s; state construction 34.3 s (45.7%); bound propagation 5.14 s (6.9%).
- propagation work is at most 1.10 passes on 35 of 40; 10 instances call propagation once.
- host cost of propagation per work unit: median 5.5 ns, 0.9 to 12.0 ns (Stage 0 unit: 10 ns).
- of the 24 instances that start at or above 5e5 nonzeros, 7 call propagation once and 5 end
  below 5e5; the other 12 have 45 calls in all. On var-smallemery-m6j6 the probe shows all 9
  calls inside the first cascade, which leaves 11 instances with 36 calls, at most 25 of them
  after the first call (bab6, tutaki, cygnet, splice1k1, a2864-99blp, dlr2, hawaiiv10-130,
  motru, sesia, square47, usafa). How many of those are in rounds >= 2 is not in the stored
  files.

Probe, 17 local instances:
- 392 propagation calls; first call of each instance = 92.5% of all propagation work;
  calls of rounds >= 2: 257 calls (236 of them unitcal_7), 6.1% of the work.
- F >= 20,000 in a round >= 2: 7 calls (ravan 73,404; radiationm18 94,842 and 49,130;
  rd-rplusc-21 727,736; uccase9 191,120 and 52,789; unitcal_7 28,384). None of them on a model
  with 5e5 free nonzeros at that call (largest: ravan 468,648).
- bound changes in those 7 calls: 0, 10, 0, 5, 0, 0, 2. Rounds of the replay to an empty
  frontier: 1 to 3.
- candidate-column entries against frontier entries, first round of calls after the first:
  radiationm18 r2p0 1.3x, uccase9 r2p0 2.2x, unitcal_7 r2p0 3.6x, istanbul 5.5x to 14x,
  brazil3 6.1x, ravan 8x to 29x (r2p1: 24 against 11,828), rd-rplusc-21 73x and 278x.
- design's charge against the host's work on the same call: ravan r2p0 309,898 / 76,681;
  rd-rplusc-21 r1p1 775,820 / 34,284, r2p1 201,297 / 2,714; var-smallemery r1p1 1,424,456 / 1,508.
- design's F against the host's work: var-smallemery r1p1 819,847 / 1,508 (151 rows with one
  infinite term, 818,639 live entries, the whole model has 575,217 free nonzeros);
  supportcase42 583,165 / 147,512; rd-rplusc-21 r2p0 727,736 / 442,393.
- rounds >= 2 with a propagation call: 16; in 11 of them the previous round's doubleton call
  had findings (entries moved or deleted), and every round 2 has no image yet.
- shadow run (host rule on a copy of the state) against the same call in place, minimum of 5:
  ratio 0.97 to 1.88, mostly 1.05 to 1.35 on calls below 100,000 units (Mac).

Stage 0 files:
- rows above 8,192 entries on 10 of the 40 matrices; longest rows: sesia 3,732,481,
  usafa 194,469, dlr2 141,121, bab6 62,516. Best compensated round on sesia: 8.37 ms.
- best compensated round names: lane groups are the fastest activity variant on 2 of the 9
  matrices above 1e7 (one thread per row up to 64 on 6); one thread per column up to 64 is the
  faster candidate scheduling on 12 of the 19 matrices from 1e6.
- fixed costs: launch with synchronisation 9.0 us, read-back 13.8 us, ordered select over a
  1e7-element flag array 106 to 141 us whatever is selected, copy_if 57 us at 1e6 and 231 us at
  1e7, pinned copy 50 to 53 GB/s, pageable copy 9.6 to 11.1 GB/s, cudaHostAlloc of 64 MB 18.2 ms.
