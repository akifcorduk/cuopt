# GPU MIP presolve: design plan (version 2)

Date: 2026-11-26. Status: plan for approval; nothing is built, downloaded or run yet.
Version 2 replaces version 1 after the user's review; [HISTORY.md](HISTORY.md) records what changed
and why.

Numbers about HiGHS, PaPILO, cuOpt, MIPLIB and Grace Hopper were read from source, tables or
papers. Every number about our own speed, memory and reduction counts is an estimate until Stage 0
and Stage 1 measure it.

## 0. Decisions for you

1. Testbed: the 39-instance structure-diverse set of section 14, from 50,000 nonzeros up, plus a few
   infeasible and unbounded instances for status tests.
2. Grace Hopper: is a GH200 node available for measurements? Until then the hybrid is designed for
   it behind one memory interface and measured on x86 only.
3. Downloads: the testbed instances and their MIPLIB solution files that are not already on the dev
   node (27 of the 39 are benchmark-set instances).
4. Baseline: HiGHS only, built from source at the commit studied with 64-bit integers, plus a small
   diagnostic patch kept as a diff; harness only. PaPILO is compared on the cluster, where it is
   cuOpt's current presolve.
5. Location: a project with its own build that does not include cuOpt. I propose a directory inside a
   new cuOpt worktree so the existing sync and container work; a fully separate repository needs its
   own sync setup.
6. Stage order: rules needed for parity with HiGHS (Stage 2) before the GPU-wide differentiators
   (Stage 3).
7. Scope approved now: Stages 0 and 1.

## 1. Goal and expectation

Goal: on MIPs from 50,000 nonzeros up, remove more than HiGHS presolve, with better quality, in
less time; deterministic; no PaPILO inside.

What the size target implies. At 5e4 to 5e5 nonzeros a full matrix sweep is microseconds on either
processor, and a GPU phase has a fixed cost (launches and one synchronisation). The GPU therefore
wins only on wide work, where the work is far larger than the matrix: probing, clique work,
pairwise row and column tests, evaluation of all aggregator candidates. Cheap dependent cascades
belong on the CPU. From about 5e5 nonzeros up, sweeps start to pay on the GPU as well.

- Time: a win is expected where HiGHS probes, merges cliques or scans pairs, and on large models.
  Where HiGHS presolve takes a few milliseconds in total, expect a tie. Proposed target: shifted
  geomean (shift 0.1 s) below HiGHS over the testbed, and no instance slower by more than
  0.1 s + 20%.
- Reductions: parity through our own complete rule set; a surplus from coverage where HiGHS limits
  itself (probing contingent starting at 1,000, aggregator pass ending after three fill-in failures,
  stages it runs once) and from clique strengthening. Unmeasured until Stage 0.
- Quality: compensated activities, numeric vetoes per transaction, strengthened clique rows;
  measured as LP bound and solver outcome in cuOpt on the cluster.
- 147 of the 240 MIPLIB benchmark instances have at least 50,000 nonzeros, so the regular cluster
  benchmark measures this work.

## 2. Architecture

### 2.1 One state, two engines

- Host engine: sequential, our own code, the complete rule set for cascades and ordered
  eliminations (bound and row rules, singleton rows and columns, doubleton equations, the aggregator
  with live counts). It also serves as the reference for differential tests.
- GPU engine: the wide phases, and sweeps when the frontier is dense.
- The engine for each phase is chosen by a deterministic work estimate (entries the phase will
  touch) against thresholds that are settings, calibrated in Stage 0. Never by measured time.

### 2.2 Shared memory (Grace Hopper in mind)

One presolve-owned memory resource holds the matrix and state, with a `host_access(range, mode)` /
`release` pair.

| Machine | Allocation | Hand-over |
|---|---|---|
| Grace Hopper (coherent page table) | Managed memory, preferred location GPU, accessed-by host, prefetched once | After a stream synchronisation the host engine reads the same arrays in place: no mirror, no downloads. Host writes go to a host-resident log that the device replays; worklist arrays are host-preferred so access counters do not migrate them |
| x86 with a discrete GPU | Device memory plus a pinned host mirror | Dirty lists copied through pinned buffers. Managed memory is wrong here: every hand-over faults (30 to 45 microseconds per fault) |

The path is selected at start from the device attribute for host page tables. Facts behind this:
on Grace Hopper GPU memory obtained with `cudaMalloc` is not host-accessible; managed memory needs
the accessed-by hint to avoid 2 MB migrations on a CPU touch; kernels on CPU-resident data run at
link speed (a 23x slowdown was measured for dgemm), so the matrix must be GPU-resident. cuOpt has no
Grace Hopper path today.

### 2.3 Matrix and kernels

- CSR and CSC with values in both, 32-bit ids, 64-bit offsets; alive flags and entry tombstones;
  fixed columns stay in their rows until the next rebuild.
- Fill-in on the host goes to a copy-on-write row overlay; on the GPU a structural batch is planned
  on the immutable matrix. Both are merged by a rebuild (compaction and transpose).
- The matrix is stored binned by row and column length, taken from cuOpt's load-balanced
  propagation: sub-warp groups up to 64 entries, one block up to 8,192, chunks above. Bins are
  rebuilt at each rebuild by a stable sort, so there is no second copy of the matrix and no atomics.
  A frontier filter restricts kernels to changed rows and columns.

### 2.4 Activities

- Minimum and maximum activity per row are kept as compensated (two-sum) accumulators plus integer
  counters of infinite contributions, and updated incrementally with deltas a * (new - old bound).
  This is the scheme HiGHS uses, and it gains the single-infinity tightening that cuOpt's current
  propagation lacks.
- On the GPU, deltas are sorted by (row, column) and applied by a segmented reduction in that fixed
  order. On the host the order is the worklist order.
- A fresh recomputation happens only at a rebuild, where it is free, and in debug builds as a check.

### 2.5 Determinism

- No floating-point atomics. Sums are sort-then-reduce or one thread per row. Bound merges are
  min/max reductions. Flags are idempotent writes. Counts are reductions. Truncation at a cap is in
  key order.
- Budgets are work units and bytes. A wall-clock limit exists but is not used in deterministic mode.
- Contracted floating-point operations are off; fused multiply-add is used only explicitly.
- Guarantee: the same reduced model for the same settings on every run and thread count. Target,
  tested by a hash of the reduced model: also across H200, L40S and Grace Hopper when the engine
  thresholds are the same.

## 3. Rounds and application order

Each phase proves its candidates on the state left by the previous phase.

1. Cascade loop (no fill-in): bound facts and row edits, then lock arguments on recomputed locks,
   on dirty lists to an empty frontier or a unit budget. Host engine by default; GPU kernels when the
   frontier work exceeds the threshold.
2. Witnessed and dual candidates are proved on the post-cascade state and resolved in one
   reservation table. Candidate evaluation is a GPU phase when wide.
3. Structural phases, one kind at a time, each followed by a rebuild: aliases; column merges and
   column transforms; substitutions; row combinations. Substitutions are applied by the host
   aggregator with live counts; on large models independent batches are applied on the GPU first and
   the host finishes the residual.
4. Wide phases when the above stall (less than 0.05% of rows plus columns removed): probing, clique
   pipeline, pairwise tests, implied integers. Then back to 1; at most a fixed number of
   alternations.

## 4. Validity classes

| Class | Members | Rule |
|---|---|---|
| B, bound facts | bound tightening, fixing, infeasibility, probing and implication fixings, implied integers by the equation test | True for every feasible point; applied together by min/max. |
| R, self-witnessed row edits | redundant row, forcing-row deletion, integral scaling, side rounding, gcd, Chvatal-Gomory rounding, coefficient strengthening, cover strengthening, tiny-coefficient removal | Uses the row and explicit bounds only. One composed edit per row per round, each step on the result of the previous, redundancy re-tested last. |
| L, lock arguments | dual fixing by locks, empty columns | Evaluated on locks recomputed after B and R. |
| W, witnessed row edits | parallel rows (claims both rows), clique strengthening of set-packing rows, dominated clique rows, sparsify | In the reservation table: writes its row, reads its witness rows. |
| D, dual reductions | strict row-dual conclusions (implied equations, dominated and forcing columns), singleton-column rules, stuffing, pairwise dominated columns, dual-fixing extensions, weak implied-integer test | In the reservation table. |
| S, structural | alias forest; parallel-column merge and gcd column transform; substitution; row combination | One kind per batch; acceptance per transaction by a numeric dry run. |

Reservation table: cells of five kinds (row, column entries, column bounds, cost, type). Any change
of an entry, side or liveness of row i write-claims row i and the column-entries cells of its
columns. Any proof that compares or counts entries of column j reads its column-entries cell. Keys
are (hash, candidate id), unique. Each cell keeps minimum writer and minimum reader, built by
reductions; selection is iterated to a maximal independent set; a loser leaves no trace. In the host
engine the same premises are checked at application time, one transaction after another.

Guards (each was broken by a counterexample in review of version 1):

- Dual arguments use continuous columns only, strictly beyond the dual tolerance; the column bound
  that drops a side is infinite or strictly implied; at most one implied premise per derivation;
  row-dual bounds are never carried across a structural batch.
- An integer column is substituted only if the pivot row proves integrality (all other columns true
  integers, integral ratios and right-hand side).
- Type order integer > implied integer > continuous. A representative standing in for a true integer
  becomes a true integer.
- Parallel columns: the HiGHS no-hole conditions; the merged column gets the widened bounds and its
  derived records are reset.
- Dominated columns: dominating bound infinite, or implied by a named row in the read set, or both
  in a common clique row in the read set; an integer never dominates a continuous column; only
  odd-depth columns of the domination forest are fixed.
- "Bound is implied" = one hop from one named row with explicit bounds, row in the read set, record
  void when the row's version changed.
- Coefficient strengthening: integer columns only, exactly one finite side, finite max activity M,
  M - b > feastol, |a_k| > M - b + feastol. Forcing-row tolerance is feastol / dynamism.
- Alias forest: a deterministic spanning forest, each tree rooted at a highest-type column; an edge
  whose orientation is not admissible is cut and its row stays; maps are composed on the frozen tree
  with double-buffered pointer jumping; non-tree edges stay as rows. Aliases and merges are separate
  kinds.
- Substitution: each pivot (p, j) has a premise set = p plus every row named by the implied-bound
  proofs of j. In a GPU batch no pivot column may appear in a row of another pivot's premise set.
  Pivot rows: equations, strictly implied equations, and weakly dual-implied rows whose dual bound
  has the pivot column as source; the host engine may take the other weak rows because it re-tests
  at application time. Row or column of size 2 always taken, pivot at least 0.01 of the row or column
  maximum, net fill-in at most 10, all candidates tested. A pivot is vetoed, in all its rows, when a
  result row's max/min |a| exceeds max(old, 1e6). The cost row is part of every substitution.
- Cancellation: a combined entry is a structural zero if |t + s*p| <= 1e-12 (|t| + |s*p|); a small
  coefficient is dropped only with the side shifted by a*l and a per-row budget of 0.1 feastol.
- Clique strengthening: only pure set-packing rows are replaced or removed; an equality clique row
  loses at most its <= side; each added literal is tested against all literals already in the
  clique; the rows that imply the conflict edges are the witnesses.

## 5. Rule coverage against HiGHS

| HiGHS rule | Here |
|---|---|
| Infeasible, redundant, empty, singleton, forcing rows; implied column bounds; fixed and empty columns | Stage 1 |
| Integral scaling, side rounding, Chvatal-Gomory, coefficient strengthening | Stage 1 |
| Doubleton equations; parallel rows; dual fixing by locks | Stage 1 |
| Implied equations, dominated / weakly dominated / forcing columns, row-dual bounds | Stage 2 |
| Singleton columns (free substitution, stuffing, zero-cost singleton) | Stage 2 |
| Pairwise dominated columns, parallel columns | Stage 2 |
| Aggregator | Stage 2 |
| Simple probing on equations, gcd substitution, implied integers, VLB/VUB extraction, dual-fixing extensions | Stage 2 |
| Probing, implications, clique merging, clique and implication substitutions | Stage 3 |
| Sparsify, cover strengthening | Stage 4 |
| Fourier-Motzkin, enumeration of small binary rows, VLB/VUB aggregation, dual-fix probing | Left out of the first versions |

Left out: dual postsolve, dependent rows, network-matrix implied integers, two-row bound tightening,
two-column cancellation, symmetry, solving components, multi-GPU.

## 6. Numerics

- Tolerances are inputs (feasibility 1e-6, dual 1e-7, zero 1e-9, integrality as cuOpt passes it),
  with a tolerance table per rule.
- An integer bound is accepted only if the activity's error bound divided by |a_j| is at most
  0.1 feastol; no bound is derived when |b| * 1e-14 > feastol.
- Implied bounds are recorded with source row and version when they improve the record by more than
  1000 * feastol. Integer columns: the explicit bound is tightened. Continuous columns: explicit
  bounds are not touched during cascade, dual and structural phases; recorded bounds are made
  explicit once, before probing or at the end (what HiGHS does).
- Whether bounds at or above 1e8 count as infinite is decided after Stage 0 counts the rows
  concerned.

## 7. Probing and cliques (built on cuOpt's code)

What cuOpt has today, read from source:
- Probing runs one variable per OpenMP task, both values, each with dense state of 48m + 40n bytes
  and a cost proportional to m + n per variable however local the propagation. Its work budget
  allows about 375 variables at 1e7 nonzeros. It does not use cliques.
- The load-balanced propagation was dropped from the build in September 2025 and does not compile
  against the current problem class. Its kernels are liftable. It recomputes all rows each
  iteration with plain sums and has no infinity counters.
- The clique table is host-only, built inside branch and bound, follows Brito and Santos, extends
  only set-packing rows below 128 entries once each, removes no dominated cliques, and is not built
  in deterministic mode.

Probing here:
- Many sources per launch. A fixed number of slots, each with its own bound state; a probe runs in
  a slot on frontier lists, records its changes in a trail, and the slot is restored from the trail
  after the harvest. Work is proportional to the entries touched, not to m + n.
- One work list over (slot, row, entry) balanced by prefix sum; bound merges per (slot, column) by
  reduction; activity deltas by the compensated scheme of 2.4.
- Set-packing, partitioning and covering rows and clique-table neighbours propagate implicitly.
- Harvest by joining the two sides of a variable: fixings, tightened bounds, aliases (to the alias
  forest), implications and variable bounds for export. Candidates ordered by HiGHS's implication
  score; a second pass re-probes only variables whose neighbourhood changed.
- A source closed by a cap is distinguished from one closed by a conflict.
- Slot state is dense where slots * (m + n) fits the byte budget, which covers most models up to
  about 1e6 nonzeros; larger models use sparse per-slot overlays.

Clique pipeline (Dai and Chen 2025, a CPU-thread method, adapted to GPU kernels):
1. Per row: reduce to its binary part, detect set-packing and conflicting knapsack rows.
2. Per knapsack row: segmented sort by coefficient, two binary searches; cliques stored implicitly
   as (row, start) and (vertex, row, start), as cuOpt's table does.
3. Adjacency without a quadratic matrix: explicit edges by expand, sort, unique only for small
   cliques; larger cliques stay implicit and adjacency is a sorted-list intersection.
4. Extension: one task per clique; candidates are the neighbours of its minimum-degree member that
   are adjacent to all members; greedy grouping in literal order gives the longest extension and the
   others.
5. Dominated cliques: candidates by shared member and signature, exact subset test; lowest id kept.
6. Triage: longest extensions replace set-packing rows (class W); the other cliques are exported to
   cuOpt's cut pool.
Changes against the paper: fixed rules instead of random shuffles and random truncation; caps as
functions of the model; the tolerance on a_i + a_j > b that cuOpt already had to fix.

Implication graph: components over two-literal rows give equivalences and contradictions.

## 8. Postsolve, crush, status

- Records only where a column disappears or changes meaning: fixed value, fixed at infinity (saved
  rows, one owner column per row), substitution, alias, merge chain, singleton column with its row,
  stuffing, integer shift/scale.
- Invariant: a record's saved row contains no column whose record is undone later. Undo runs in
  reverse order of application; GPU batches are parallel loops, host segments sequential. Integers
  are snapped before the undo. One postsolve stack, one tolerance set.
- The result is checked against original rows, bounds and integrality (optional per call).
- Primal-only mode: classes L and D, dual-implied pivot rows, the weak implied-integer test and
  column dominance are off, and every original-feasible point crushes exactly to a reduced-feasible
  point of equal objective. Used when a set-solution callback is registered, as cuOpt does today.
- Status: REDUCED, UNCHANGED, REDUCED_TO_EMPTY, INFEASIBLE, UNBOUNDED_OR_INFEASIBLE, LIMIT (valid
  partial result), FAILED (fall back to cuOpt's current presolve).

## 9. Interface

Input: CSR, row sides, bounds, types, cost, offset, sense, tolerances; minimisation internally.
Output: reduced model, types with a separate implied-integer flag, column map with a "transformed"
marker, status, postsolve object with a host entry callable from several threads, and by-products
(fixings, implications, clique rows, variable bounds) in reduced indices; every structural batch
and rebuild remaps or drops them.

## 10. cuOpt

- A thin adapter from Stage 1 on: the new presolve in place of the PaPILO call behind a setting,
  so each stage can go to the cluster. Reformulation and scaling before it and the GPU stages after
  it stay as they are at first.
- cuOpt's three custom presolvers that are PaPILO plugins (GF2, coefficient reduction, single-lock
  dual aggregation) and indicator strengthening do not come along; Stage 5 ports them or drops them
  on evidence.
- Early GPU FJ holds about 120 B per nonzero during presolve; it starts only if both fit.
- What goes back into cuOpt: the revived load-balanced propagation with infinity counters and
  compensated sums; many-source probing in place of the probing cache; a device clique table that
  also works in deterministic mode.
- Found while reading: the probing cache sets an iteration limit of 50 on an object the probes do
  not use, so probes run without an iteration limit (probing_cache.cu:919 against :934). To confirm.

## 11. Measurement

Local, standalone against HiGHS (64-bit build, pinned commit, status recorded), both models in one
convention (minimisation, same equilibration, implied integers counted separately):

| Measure | Rule |
|---|---|
| Size: rows, columns, nonzeros, integer columns | Shifted geomean lower than HiGHS |
| Seconds, host arrays to usable reduced model | Target of section 1; per GPU type; thread counts stated |
| Numerics: max and 99th-percentile row dynamism | At most a stated factor above HiGHS |
| Correctness | Test oracle; MIPLIB solutions crushed in primal-only mode stay feasible with equal objective and postsolve back; determinism hash |

On the cluster, inside cuOpt (no solver quality runs locally): LP relaxation bound of the reduced
model, primal gap against the best known solution and primal integral, postsolved solutions
feasible in the original model, statuses. Baseline is cuOpt's current presolve; one run per arm
first.

## 12. Stages

Sizes are rough engineer-weeks for a person and show relative scale only.

| Stage | Content | Gate and cluster run |
|---|---|---|
| 0. Baseline and cost model (1 to 2) | HiGHS on the testbed: seconds per stage, reductions per class, probes and stop reason, substitutions, clique merging effect. Instance structure. Micro-benchmarks: launch and synchronisation cost, hand-over per memory option, sweep kernels, rebuild | Target table per instance; engine thresholds; confirmation or correction of the section 1 time target |
| 1. Core (10 to 12) | Memory resource and shared state; host engine for B, R, L, singleton rows, doubleton equations; compensated activities; GPU propagation from the load-balanced kernels; alias forest; parallel rows; postsolve; statuses; test oracle; determinism hash; cuOpt adapter | At least 99% of HiGHS's reductions with the same rules on every instance, time target met. Cluster run 1: correctness and presolve time at scale; solver outcome is expected below PaPILO at this stage |
| 2. Parity (15 to 20) | Reservation table; dual reductions; singleton, dominated and parallel columns; aggregator (GPU evaluation, host application, GPU batches on large models); implied integers; gcd transforms | Remaining nonzeros at parity with full HiGHS with probing off. Cluster run 2: first meaningful comparison with PaPILO |
| 3. GPU-wide (10 to 15) | Many-source probing; clique pipeline; implication graph; clique-aware propagation | More reductions than full HiGHS in geomean, time target. Cluster run 3 |
| 4. Completion | Sparsify, cover strengthening, components; tuning; a second, held-out diverse set; Grace Hopper measurements if a node is available | Section 11 on the held-out set |
| 5. Productization | Default path in cuOpt, custom presolvers, primal-only mode, limits, documentation, back-ports | Cluster A/B as the acceptance test |

Test oracle: tiny random MIPs solved by enumeration, including infeasible, unbounded and empty
outcomes, scaled coefficients and huge bounds, multi-round sequences, and every size threshold as a
test parameter so each path, including both engines and both memory paths, runs on tiny models.
Checks: reduced optimum equals the original, every reduced-feasible point postsolves to an
original-feasible point, exact crush in primal-only mode, exported implications and cliques valid,
GPU batch result equals sequential replay, host and GPU engines agree on B and R fixpoints.

## 13. Expectations by structure (unmeasured)

- Short binary rows (set packing, precedence, variable bounds): alias forest, implication graph,
  probing coverage, clique strengthening. The most likely place for a reduction surplus.
- Mostly continuous models: decided by the aggregator and singleton columns; parity is the aim.
- Few long rows over many columns (set partitioning and covering): parallel and dominated columns;
  the gain is time.
- Dense large models: sweeps dominate; the gain is time.

## 14. Testbed

39 instances drawn from the MIPLIB 2017 collection by a greedy cover of 102 structure features
(size band x variable mix, aspect, mean row and column length, the 14 constraint-class tags, plus
numerics, indicator, decomposition), one per model family, benchmark instances with a known optimum
preferred.

| Band | Instance | Rows | Columns | Nonzeros | Mix | Shape |
|---|---|---|---|---|---|---|
| A | air05 | 426 | 7,195 | 52,121 | binary | wide, long rows |
| A | piperout-27 | 18,442 | 11,659 | 54,662 | general mixed | short rows |
| A | dws008-01 | 6,064 | 11,096 | 56,400 | mixed binary | |
| A | comp21-2idx | 14,038 | 10,863 | 57,301 | integer | |
| A | cod105 | 1,024 | 1,024 | 57,344 | binary | set packing |
| A | neos-1171737 | 4,179 | 2,340 | 58,620 | mixed binary | |
| A | neos-3656078-kumeu | 17,656 | 14,870 | 59,292 | general mixed | short rows |
| A | glass-sc | 6,119 | 214 | 63,918 | binary | tall, set covering |
| A | decomp2 | 10,765 | 14,387 | 64,073 | binary | |
| A | cbs-cta | 10,112 | 24,793 | 64,388 | mostly continuous | short columns |
| A | istanbul-no-cutoff | 20,346 | 5,282 | 71,477 | mostly continuous | short rows |
| A | elitserienhandball14i | 35,328 | 32,541 | 98,685 | general mixed | indicator, numerics |
| A | leo1 | 593 | 6,731 | 131,218 | mixed binary | wide, long rows |
| A | neos-662469 | 1,085 | 18,235 | 200,055 | integer | wide |
| B | proteindesign122trx11p8 | 254 | 127,326 | 503,427 | integer | wide, rows of 1,982 |
| B | triptim1 | 15,706 | 30,055 | 515,436 | general mixed | |
| B | ex9 | 40,962 | 10,404 | 517,112 | integer | |
| B | irish-electricity | 104,259 | 61,728 | 523,257 | mixed binary | |
| B | neos-4532248-waihi | 167,322 | 86,842 | 525,339 | mixed binary | short rows |
| B | map16715-04 | 328,818 | 164,547 | 549,920 | mostly continuous | rows of 1.7 |
| B | supportcase10 | 165,684 | 14,770 | 555,082 | binary | tall |
| B | neos-2987310-joes | 29,015 | 27,837 | 580,291 | mixed binary | |
| B | supportcase6 | 771 | 130,052 | 584,976 | integer | wide |
| B | netdiversion | 119,589 | 129,180 | 615,282 | binary | |
| B | neos-4306827-ravan | 261,877 | 162,291 | 639,474 | general mixed | short rows |
| B | rd-rplusc-21 | 125,899 | 622 | 852,384 | mixed binary | tall, columns of 1,370 |
| B | neos-848589 | 1,484 | 550,539 | 1,101,078 | mostly continuous | wide |
| B | bab3 | 23,069 | 393,800 | 3,301,838 | binary | wide |
| C | s82 | 87,878 | 1,690,631 | 7,022,608 | mixed binary | wide, numerics |
| C | gfd-schedulen180f7d50m30k18-16i | 3,447,025 | 2,226,625 | 7,026,385 | general mixed | rows of 2.0, indicator |
| C | supportcase2 | 597,385 | 2,912 | 7,653,928 | binary | tall, columns of 2,628 |
| C | usafa | 1,377,561 | 228,648 | 16,485,102 | integer | |
| C | a2864-99blp | 22,117 | 200,787 | 20,078,717 | binary | rows of 908 |
| C | neos-3740487-motru | 4,936,372 | 1,689,612 | 26,433,474 | mostly continuous | |
| C | square47 | 61,591 | 95,030 | 27,329,856 | integer | dense |
| C | neos-4332810-sesia | 19,912,111 | 20,682,487 | 47,513,027 | general mixed | rows of 2.4 |
| D | dlr2 | 7,132,926 | 38,868,107 | 78,091,589 | mostly continuous | columns of 2.0 |
| D | hawaiiv10-130 | 1,388,052 | 685,130 | 183,263,061 | mixed binary | dense |
| D | neos-4535459-waipa | 163,001 | 961,401 | 256,963,402 | mixed binary | rows of 1,576 |

Band D is a stress test, not a target. Baseline notes: HiGHS needs the 64-bit build (its clique
limit 10 * nnz overflows 32 bits on waipa); stock HiGHS reports no per-rule counts for MIP and
cannot be restricted to a rule subset, hence the diagnostic patch; its written presolved model is
power-of-two scaled.
