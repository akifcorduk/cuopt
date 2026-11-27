# GPU MIP presolve: design plan (version 3)

Date: 2026-11-26. Status: plan for approval; nothing is built, downloaded or run yet.

Version 3 = version 2 plus the review rules of cuOpt PR 2372 and the fixes from the check of
version 2. Earlier versions are kept as [DESIGN_v1.md](DESIGN_v1.md) and
[DESIGN_v2.md](DESIGN_v2.md); [HISTORY.md](HISTORY.md) records what changed and why. Version 3
itself has not been reviewed again.

Amended 2026-11-26 (Stage 0 running): the 9 testbed instances above 1e7 nonzeros run on the dev
node, and the separate cluster job for them is dropped (section 0 item 2, section 12); two
instances with indicator constraints left the testbed (section 14).

Numbers about HiGHS, PaPILO, cuOpt, MIPLIB and Grace Hopper were read from source, tables or
papers. Every number about our own speed, memory and reduction counts is an estimate until Stage 0
and Stage 1 measure it.

## 0. Decisions for you

1. Stage order. Recommended: GPU-wide phases (probing, cliques) in Stage 2, the rules needed for
   parity with HiGHS in Stage 3. The GPU-wide phases are the only part that can pay at 50,000
   nonzeros, they are the part built on cuOpt's code, and they can be measured inside cuOpt as
   replacements for the probing cache and the clique table while PaPILO stays. The alternative,
   parity first, gives a standalone result against HiGHS sooner but no evidence at 50,000 nonzeros
   until both stages are done.
2. Where runs happen (changed 2026-11-26 after your comment that a cluster job for 9 instances
   does not pay for its setup). Dev node: all 40 testbed instances, presolve only; the 9 above
   1e7 nonzeros run there too, in the background with a time limit. Cluster: only the regular
   benchmark inside cuOpt after each stage, with the existing scripts. There is no separate
   cluster job for standalone runs. If one is ever needed, it covers a wide set (the 147
   benchmark instances at or above 50,000 nonzeros plus the testbed), never a handful.
3. Quality against HiGHS. Either (a) an extra cluster arm at Stages 3 and 5 that runs cuOpt with
   its presolve off on HiGHS-presolved models, or (b) quality is compared against cuOpt's current
   presolve only, and HiGHS only on size, time and coefficient range. Recommended: (a).
4. PaPILO in cuOpt. It leaves the main-problem path. May it stay for sub-MIPs, and as a logged
   fallback in the product default, until Stage 5 decides on evidence?
5. Determinism across machines. Recommended: one shipped set of engine thresholds, so every
   machine gives the same reduced model. Thresholds tuned per machine would be faster on Grace
   Hopper but give a different, equally valid result there.
6. Grace Hopper. Is a GH200 node available? Without one that path is tested for function only.
7. Testbed: section 14 (40 instances in eight size sub-bands, four status instances, a held-out
   set of 36 drawn now and not looked at before Stage 4).
8. Baseline: HiGHS only, pinned commit, 64-bit integers, with a small patch that adds rule
   switches and per-rule counters; harness only.
9. Location: a directory with its own CMake project in a new cuOpt worktree; no cuOpt target or
   header is used (section 11).
10. Downloads: the 13 testbed instances that are not in the benchmark set, the status instances,
    and MIPLIB solution files.
11. Scope approved now: Stages 0 and 1.

## 1. Goal and expectation

Goal: on MIPs from 50,000 nonzeros up, remove more than HiGHS presolve, with better quality, in
less time; deterministic; no PaPILO inside.

What the size target implies. At 5e4 to 5e5 nonzeros a full matrix sweep is microseconds on either
processor, and every GPU phase has a fixed cost. The GPU wins there only on wide work, where the
work is far larger than the matrix: probing, clique work, pairwise tests, evaluation of all
aggregator candidates. Cheap dependent cascades belong on the CPU. Until Stage 2 the presolve at
that size is the sequential host engine alone; the GPU contributes there from Stage 2.

Time target, reported and gated per band, warm (context, pools and pinned buffers exist; cold
start reported separately):

| Band | Target |
|---|---|
| A and B (5e4 to 1e7 nonzeros) | Shifted geomean not above HiGHS on the same rule set at Stage 1, below full HiGHS from Stage 2 on; no instance slower by more than 20% plus a small constant. Shift and constant are set in Stage 0 from HiGHS's measured times |
| C and D (above 1e7) | Shifted geomean below HiGHS; D is reported only |

Reductions: parity through our own complete rule set; a surplus from coverage where HiGHS limits
itself (probing contingent starting at 1,000, aggregator pass ending after three fill-in failures,
stages it runs once) and from clique strengthening. Unmeasured until Stage 0.

## 2. Architecture

### 2.1 One state, two engines

- Host engine: sequential, our own code, the complete rule set for cascades and ordered
  eliminations. It follows the conventions of cuOpt PR 2127 (bound-change stack, incrementally
  updated activities, infinite-contribution counters).
- GPU engine: the wide phases, and sweeps when the frontier is dense.
- The engine for each phase is chosen by a deterministic work estimate against thresholds. The
  estimate of a device phase includes a constant per kernel launch and per read-back. Each presolve
  call has a device fixed-cost budget derived from the model size; a phase whose fixed cost does not
  fit runs on the host or is skipped.
- Thresholds, the byte budget, slot counts and every cap are constants or functions of the model
  and the settings. Nothing is read from the device (free memory, GPU type, core count).

### 2.2 Shared memory

One presolve-owned memory resource with a `host_access(range, mode)` / `release` pair.

| Machine | Matrix arrays | Mutable state (bounds, activities, counters, flags, records) |
|---|---|---|
| Grace Hopper | Managed memory, read in place by the host. Host-preferred while the work estimate keeps the cascade on the host; prefetched to the GPU with the accessed-by-host hint before the first wide phase | Separate arrays, host-preferred during a host phase and prefetched to the GPU before a device phase; the host reads and writes them directly |
| x86 with a discrete GPU | Device memory. Below a size threshold a pinned host mirror, refreshed after a rebuild only if a host phase follows. Above it no mirror: the rows and columns of the host frontier are gathered on the device and copied on demand | Dirty lists through pinned buffers, carrying bounds, accumulators and counters as bits; one engine owns them at a time |

Host matrix edits go to a copy-on-write row overlay only. The Grace Hopper path is selected by the
device attribute for direct managed-memory access from the host and can be forced for tests. Until
a GH200 node is measured that path is functional only: no time or memory figure is claimed for it
and it is outside the gates. Facts behind the table: on Grace Hopper `cudaMalloc` memory is not
host-accessible; managed memory without the accessed-by hint migrates 2 MB blocks on a CPU touch;
kernels on CPU-resident data run at link speed; on x86 managed memory faults at every hand-over.

### 2.3 Matrix and kernels

- CSR and CSC with values in both, 32-bit ids, 64-bit offsets; alive flags and entry tombstones;
  fixed columns stay in their rows until the next rebuild and are folded into the side for every
  row test.
- Structural changes are merged by a rebuild (compaction and transpose), made only when dead
  entries exceed a set fraction or the next phase needs compact storage, on the engine the work
  estimate selects.
- The matrix is stored binned by row and column length, taken from cuOpt's load-balanced
  propagation (sub-warp groups up to 64 entries, one block up to 8,192, chunks above). Bins are
  rebuilt by a stable sort, so there is no second copy of the matrix. A frontier filter restricts
  kernels to changed rows and columns.

### 2.4 Activities

- Minimum and maximum activity per row are (hi, lo) accumulators plus exact integer counters of
  infinite terms.
- A bound change is applied as two terms, minus a * old and plus a * new. Each product is
  error-free (one explicit fused multiply-add) and is added by two-sum. The difference new - old,
  or of two coefficients, is never formed. A change between an infinite and a finite bound changes
  the counter and adds or removes a * bound. This is what HiGHS does.
- A row whose entries, coefficients or scale change is recomputed at once. Entry insertion and
  deletion add or remove the entry's term.
- Per row and side the update count n and the largest |a * bound| seen, T, are kept; the row is
  recomputed when n * u^2 * T exceeds 0.1 feastol (u = rounding unit). This quantity is the
  activity's error bound wherever the plan needs one. A term with |a * bound| above 1e20 is
  counted as infinite.
- On the GPU each affected row is folded by one thread over its entries in storage order; long rows
  use one block with a tree fixed by entry position. No sort. When the affected rows hold more than
  a set fraction of the nonzeros, all rows are recomputed with the same kernel.
- At a rebuild only rows whose entries changed are recomputed. Debug builds compare against a full
  recomputation.
- Stage 1 keeps one sum per side, on explicit bounds. HiGHS keeps a second on recorded implied
  bounds; the Stage 1 gate shows what that costs, and the second sum is added if it is needed.

Rules taken from the review of cuOpt PR 2372 (single-infinity tightening in GPU propagation):

- Semantics: no infinite term, every column may get a candidate; exactly one, only the column that
  holds it; two or more, none.
- Layout: split arrays, narrowest types. The hi values are one array of doubles; lo is a separate
  cold array; the infinite-term counts of the two sides are two `uint16_t` arrays. No paired
  128-bit value on the hot path. The counts are exact, not capped at 2, because a capped count
  cannot be decremented; a count that saturates marks the row as skipped until it is recounted.
- No extra reduction pass for the counts: each thread counts infinite contributions during its one
  scan of the row and adds them with one integer atomic only if its local count is nonzero.
- Residual first: the bound is (side - (activity - a_j * own bound)) / a_j in compensated
  arithmetic, never (side - activity) / a_j + own bound.
- Error guard: no candidate is emitted when the error bound can exceed the candidate tolerance.
  The lo array is loaded only when the hi value alone cannot pass the guard.
- Every kernel variant is benchmarked against the current one before it replaces it.

### 2.5 Determinism

- No floating-point atomics. Floating-point sums are one thread per row or a tree fixed by entry
  position. Library reductions are used for integers and min/max only. Integer atomics are allowed,
  because their result does not depend on order, and are preferred where they save a pass.
- Engine thresholds and the byte budget are settings with fixed defaults, identical on every
  machine. Stage 0 fixes the defaults once.
- The wall-clock limit is only an abort that returns LIMIT; it never changes which reductions are
  tried. Contracted floating-point operations are off.
- Guarantee: the same result for the same settings on every run, thread count, co-running workload
  and GPU type.
- Test: a sequence of hashes, one after every phase, over the bit patterns of bounds, accumulators,
  matrix, postsolve stack, by-products and status, compared across repeated runs, thread counts,
  early heuristics on and off, a second GPU type and both memory paths.

## 3. Rounds and application order

Each phase proves its candidates on the state left by the previous phase.

1. Cascade loop (no fill-in): bound facts and row edits, then lock arguments on recomputed locks,
   on dirty lists to an empty frontier or a unit budget. Host engine by default; GPU kernels when
   the frontier work exceeds the threshold.
2. Witnessed and dual candidates, proved on the post-cascade state. In the host engine each is
   applied one transaction at a time with its premises checked at application (this is all Stage 1
   needs for parallel rows and aliases). GPU batches go through the reservation table. Pairwise
   scans are GPU phases when wide.
3. Structural phases, one kind at a time: aliases; column merges and column transforms;
   substitutions; row combinations. Substitutions are applied by the host aggregator with live
   counts; on large models independent batches are applied on the GPU first.
4. Wide phases when the above stall (less than 0.05% of rows and of columns removed): probing,
   clique pipeline, implication graph. Then back to 1, for a fixed number of alternations at most.

## 4. Validity classes

| Class | Members | Rule |
|---|---|---|
| B, bound facts | bound tightening, fixing, infeasibility, probing and implication fixings, implied integers by the equation test | True for every feasible point; applied together by min/max. |
| R, self-witnessed row edits | redundant row, forcing-row deletion, integral scaling, side rounding, gcd, Chvatal-Gomory rounding, coefficient strengthening, cover strengthening, tiny-coefficient removal | Uses the row and explicit bounds only. One composed edit per row per round, each step on the result of the previous, redundancy re-tested last. |
| L, lock arguments | dual fixing by locks, empty columns | Evaluated on locks recomputed after B and R. |
| W, witnessed row edits | parallel rows (claims both rows), clique strengthening of set-packing rows, removal of dominated clique rows | Writes its row, reads its witness rows. |
| D, dual reductions | strict row-dual conclusions (implied equations, dominated and forcing columns), singleton-column rules, stuffing, pairwise dominated columns, dual-fixing extensions, weak implied-integer test | Under reservation. |
| S, structural | alias forest; parallel-column merge and gcd column transform; substitution; row combination (sparsify) | One kind per batch; acceptance per transaction by a numeric dry run. |

Reservation table (GPU batches): cells of five kinds (row, column entries, column bounds, cost,
type). Any change of an entry, side or liveness of row i write-claims row i and the column-entries
cells of its columns. Any proof that compares or counts entries of column j reads its
column-entries cell. Keys are (hash, candidate id), unique. Each cell keeps minimum writer and
minimum reader; selection is iterated to a maximal independent set; a loser leaves no trace.

Guards (each was broken by a counterexample in review):

- Dual arguments use continuous columns only, strictly beyond the dual tolerance; the column bound
  that drops a side is infinite or strictly implied; at most one implied premise per derivation;
  row-dual bounds are never carried across a structural batch.
- An integer column is substituted only if the pivot row proves integrality (all other columns true
  integers, integral ratios and right-hand side).
- Type order integer > implied integer > continuous. A representative standing in for a true
  integer becomes a true integer.
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
  whose orientation is not admissible is cut and its row stays; maps are composed on the frozen
  tree; non-tree edges stay as rows. Aliases and merges are separate kinds.
- Substitution: each pivot (p, j) has a premise set = p plus every row named by the implied-bound
  proofs of j. In a GPU batch no pivot column may appear in a row of another pivot's premise set.
  Pivot rows: equations, strictly implied equations, and weakly dual-implied rows whose dual bound
  has the pivot column as source; the host engine may take other weak rows because it re-tests at
  application. Row or column of size 2 always taken, pivot at least 0.01 of the row or column
  maximum, net fill-in at most 10, all candidates tested. A pivot is vetoed, in all its rows, when a
  result row's max/min |a| exceeds max(old, 1e6). The cost row is part of every substitution.
- Cancellation: a combined entry is a structural zero if |t + s*p| <= 1e-12 (|t| + |s*p|); a small
  coefficient is dropped only with the side shifted by a*l and a per-row budget of 0.1 feastol.
- Cliques: a set-packing row is replaced only by a clique that contains it. It is removed only when
  a clique that contains it is a model row at the end of the same batch; a clique that exists only
  in the table or the cut pool never justifies a removal. For an equality clique row the extension
  literals are fixed to zero and the row is not changed. A candidate that is the complement of a
  member is not added; if it is adjacent to all other members, those members are fixed to zero.

## 5. Rule coverage against HiGHS

| HiGHS rule | Here |
|---|---|
| Infeasible, redundant, empty, singleton, forcing rows; implied column bounds; fixed and empty columns | Stage 1 |
| Integral scaling, side rounding, Chvatal-Gomory, coefficient strengthening | Stage 1 |
| Doubleton equations; parallel rows; dual fixing by locks | Stage 1 |
| Probing, implications, clique merging, clique and implication substitutions | Stage 2 |
| Implied equations, dominated / weakly dominated / forcing columns, row-dual bounds | Stage 3 |
| Singleton columns (free substitution, stuffing, zero-cost singleton) | Stage 3 |
| Pairwise dominated columns, parallel columns; aggregator | Stage 3 |
| Simple probing on equations, gcd substitution, implied integers, VLB/VUB extraction, dual-fixing extensions | Stage 3 |
| Sparsify, cover strengthening | Stage 4 |
| Fourier-Motzkin, enumeration of small binary rows, VLB/VUB aggregation, dual-fix probing | Left out of the first versions |

Left out: dual postsolve, dependent rows, network-matrix implied integers, two-row bound
tightening, two-column cancellation, symmetry, components, multi-GPU.

## 6. Numerics

- Tolerances are inputs (feasibility 1e-6, dual 1e-7, zero 1e-9, integrality as cuOpt passes it),
  with a tolerance table per rule.
- An integer bound is accepted only if the error bound of 2.4 divided by |a_j| is at most
  0.1 feastol; no bound is derived when |b| * 1e-14 > feastol.
- Implied bounds are recorded with source row and version when they improve the record by more
  than 1000 * feastol. Integer columns: the explicit bound is tightened. Continuous columns:
  explicit bounds are not touched during cascade, dual and structural phases; recorded bounds are
  made explicit once, before probing or at the end (what HiGHS does).

## 7. Probing and cliques

What cuOpt has today, read from source:
- Probing runs one variable per OpenMP task, both values, with dense state of 48m + 40n bytes and
  a cost proportional to m + n per variable. Its budget allows about 375 variables at 1e7
  nonzeros. It does not use cliques.
- The load-balanced propagation was dropped from the build in September 2025 and does not compile
  against the current problem class. Its kernels are liftable.
- The clique table is host-only, built inside branch and bound, follows Brito and Santos, extends
  only set-packing rows below 128 entries once each, removes no dominated cliques, and is not built
  in deterministic mode.
- Open pull requests read on 2026-11-26: 2372 (single-infinity tightening in GPU propagation, same
  files as the probing code), 2127 (CPU domain propagation with bound-change stack), 2129
  (objective propagation). The standalone code follows their semantics so the back-port matches.

Probing here (the propagation kernels come from cuOpt; the probing scheme is redesigned):
- Many sources per launch. Slot state is a shared base plus a sparse overlay per slot; dense slots
  are the special case when slots * (16n + 40m) is below the byte budget. The overlay design (keys,
  lookup, bytes per touched entry) is specified before Stage 2 starts.
- A probe runs on frontier lists; work is proportional to the entries touched. One work list over
  (slot, row, entry) balanced by prefix sum; bound merges per (slot, column) by min/max.
- A slot is restored by copying saved accumulator and counter values, never by inverse deltas.
- Set-packing, partitioning and covering rows and clique-table neighbours propagate implicitly.
- Harvest by joining the two sides of a variable: fixings, tightened bounds, aliases, implications
  and variable bounds for export. Candidates are ordered by HiGHS's implication score.
- Sources in one launch do not see each other's results. Re-probing of sources whose neighbourhood
  changed repeats until nothing changes or the probing work budget ends; passes below the engine
  threshold run on the host with a trail.
- Probing has a work budget in entry updates per nonzero, set in Stage 0. A source closed by a cap
  is distinguished from one closed by a conflict.

Clique pipeline (Dai and Chen 2025, a CPU-thread method, adapted to GPU kernels):
1. Per row: reduce to its binary part, detect set-packing and conflicting knapsack rows.
2. Per knapsack row: segmented sort by coefficient, two binary searches; cliques stored implicitly
   as (row, start) and (vertex, row, start), as cuOpt's table does.
3. Adjacency without a quadratic matrix: explicit edges by expand, sort, unique for small cliques
   only; larger cliques stay implicit and adjacency is a sorted-list intersection.
4. Extension: one task per clique; candidates are the neighbours of its minimum-degree member that
   are adjacent to all members; greedy grouping in literal order.
5. Dominated cliques are removed from the table by shared member, signature and exact subset test;
   model rows are removed only under the rule of section 4.
6. Triage: longest extensions replace set-packing rows; the other cliques are exported to cuOpt
   through a new setter on its clique table.

Differences from the paper: fixed rules instead of random shuffles and random truncation; caps as
functions of the model. Not taken over: adding the longest cliques from knapsack and inferred
set-packing rows as model rows (the paper does, up to a doubling of rows). The paper's gain was
measured on the 99 instances that have such rows and comes from added rows and cuts, so it is not
assumed here; cluster run 2 decides whether to add them. Row replacement and domination removal
are parity with HiGHS's clique merging.

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
  point of equal objective. The switch exists from Stage 1.
- Status: REDUCED, UNCHANGED, REDUCED_TO_EMPTY, INFEASIBLE, UNBOUNDED_OR_INFEASIBLE, LIMIT (valid
  partial result), FAILED. FAILED is an assertion in the standalone project and in benchmark
  commits; a fallback exists only in the product default, logged and counted.

## 9. Interface

Input: CSR, row sides, bounds, types, cost, offset, sense, tolerances; minimisation internally.
Output: reduced model, types with a separate implied-integer flag, column map with a "transformed"
marker, status, postsolve object with a host entry callable from several threads, and by-products
(fixings, implications, clique rows, variable bounds) in reduced indices; every structural batch
and rebuild remaps or drops them.

## 10. cuOpt

- Adapter from Stage 1: the new presolve in place of the PaPILO call for the main problem, behind
  a setting, at all sizes. Below 50,000 nonzeros the host engine runs alone; no gain is claimed
  there and cluster results are reported separately for the 147 benchmark instances at or above
  50,000 nonzeros and the 93 below.
- The adapter receives the model after cuOpt's scaling. The harness has a switch that applies the
  same scaling, and stage gates are evaluated with it on.
- Sub-MIPs keep their PaPILO call until Stage 5 decides.
- Stage 2 enters cuOpt as replacements for the probing cache and the clique table with PaPILO
  unchanged, which isolates their effect. From then on the probing cache is off in the new arm.
- cuOpt's three custom presolvers that are PaPILO plugins and indicator strengthening do not come
  along; Stage 5 ports them or drops them on evidence.
- Early GPU heuristics hold about 120 B per nonzero during presolve; they start only if both fit.
- Possible defect found while reading: the probing cache sets an iteration limit of 50 on an
  object the probes do not use (probing_cache.cu:919 against :934). To confirm.

## 11. Standalone project

A directory in the cuOpt repository with its own CMakeLists, building a static library that uses no
cuOpt target or header. Dependencies: CUDA, CCCL, rmm. Lifted once as copies: the load-balanced
kernels, partition helpers and graph builders, and the bounds-update helpers; nothing is lifted
from the probing cache. MPS reading and the baseline go through HiGHS in the harness only. cuOpt
adds the same directory behind the setting; a cluster run is a pushed benchmark commit that
contains the directory and flips the default.

## 12. Measurement

| Where | What |
|---|---|
| Dev node, standalone against HiGHS | All 40 testbed instances: size, seconds, row dynamism, determinism hashes, the test oracle, MIPLIB solutions crushed in primal-only mode and postsolved back. The 9 instances above 1e7 nonzeros run in the background with a time limit |
| Cluster, inside cuOpt | Regular benchmark set, one run per arm, stored baseline reused: statuses, postsolved solutions feasible in the original model, presolve seconds, reduced size and root LP bound parsed from the logs, primal gap against the best known solution, primal integral |

Both models are compared in one convention (minimisation, same equilibration, implied integers
counted separately). Row dynamism is a constraint, not a goal: 99th percentile at most 10 times
HiGHS's. Nonzeros added by clique extension are reported separately from the size measure. No
solver runs on the dev node.

## 13. Stages

Sizes are rough engineer-weeks for a person and show relative scale only.

| Stage | Content | Gate and cluster run |
|---|---|---|
| 0. Baseline and cost model (1 to 2) | HiGHS on the testbed: seconds per stage, reductions per class, probes with entries touched and rounds, substitutions, clique merging effect. Instance structure. Micro-benchmarks: launch and read-back cost, hand-over, mirror refresh and on-demand gather, propagation kernel variants (paired against split arrays), rebuild, pinned allocation | Target table per instance; default thresholds; time-target constants; fixed cost per band |
| 1. Core (10 to 12) | Memory resource; binned matrix; host engine for the Stage 1 rules; activities of 2.4; GPU propagation from the lifted kernels; alias forest; parallel rows; postsolve; statuses; primal-only switch; test oracle; determinism hashes; cuOpt adapter | Reductions: geomean at least 99% of HiGHS restricted to the same rules, every instance below 95% explained. Time target against the same restricted HiGHS. Cluster run 1 (new presolve in place of PaPILO): no wrong status, every postsolved solution feasible; solver outcome is expected below PaPILO at this stage |
| 2. GPU-wide (10 to 15) | Many-source probing; implication graph; clique pipeline; clique-aware propagation; replacements for cuOpt's probing cache and clique table | Fixings, aliases and implications found and seconds against HiGHS probing on bands A and B. Clique items are built only if Stage 0 shows that HiGHS clique merging changes rows on enough instances (thresholds set after Stage 0). Cluster run 2: cuOpt as is against cuOpt with the new probing and cliques |
| 3. Parity (15 to 20) | Reservation table; dual reductions; singleton, dominated and parallel columns; aggregator; implied integers; gcd transforms; simple probing on equations; VLB/VUB extraction; dual-fixing extensions | Remaining nonzeros at parity with HiGHS patched to the rule set of Stages 1 to 3. Cluster run 3: new presolve against PaPILO, first full comparison; HiGHS-presolved arm if chosen |
| 4. Completion | Sparsify, cover strengthening; tuning; the held-out set; Grace Hopper measurements if a node exists | Section 12 on the held-out set |
| 5. Productization | Default path, custom presolvers, sub-MIPs, limits, documentation | Cluster A/B as the acceptance test, with a threshold set before the run |

Test oracle: tiny random MIPs solved by enumeration, including infeasible, unbounded and empty
outcomes, scaled coefficients and huge bounds, multi-round sequences, and every size threshold as a
test parameter so each path, including both engines and both memory paths, runs on tiny models.
Checks: reduced optimum equals the original; every reduced-feasible point postsolves to an
original-feasible point; exact crush in primal-only mode; exported implications and cliques valid;
a GPU batch equals its sequential replay; on integer-only models run to an empty frontier both
engines reach the same bounds, elsewhere each engine's result is checked for validity, not for
equality with the other. Numeric regression cases from PR 2372: x1 = 2, x2 = 3 fixed, x3 free,
x1 + x2 + x3 <= 10 gives x3 <= 5; the 2^54 cancellation case; the 513-term rounding case. Test
models carry integer variable types where a kernel requires a MIP, and fixtures restore the device
memory resource.

## 14. Testbed

40 instances drawn from the MIPLIB 2017 collection by a greedy cover of 85 structure features
(coarse size x variable mix, aspect, mean row and column length, the 14 constraint-class tags,
plus numerics, indicator, decomposition), with a quota per size sub-band and one instance per model
family; benchmark instances with a known optimum preferred. 29 are in the benchmark set.

Changed 2026-11-26 (Stage 0): supportcase21i and gfd-schedulen55f2d50m30k3i have an INDICATORS
section, which HiGHS cannot read, so they have no baseline. They are replaced by sorrell3 (A3),
istanbul-no-cutoff (A1) and brazil3 (A2), chosen by the same cover rule among instances without
indicator constraints; the set covers 84 of the 85 features (all but "indicator"). The two
instances are kept in the `indicator` list of `testbed.json` for tests inside cuOpt.

| Band | Instance | Rows | Columns | Nonzeros | Mix | Shape |
|---|---|---|---|---|---|---|
| A1 | istanbul-no-cutoff | 20,346 | 5,282 | 71,477 | mostly continuous | rows of 3.5 |
| A1 | ns1208400 | 4,289 | 2,883 | 81,746 | mixed binary | |
| A1 | wachplan | 1,553 | 3,361 | 89,361 | integer | |
| A1 | trento1 | 1,265 | 7,687 | 93,571 | mixed binary | |
| A1 | radiationm18-12-05 | 40,935 | 40,623 | 96,149 | general mixed | rows of 2.3 |
| A2 | ns1830653 | 2,932 | 1,629 | 100,933 | mixed binary | |
| A2 | supportcase40 | 38,192 | 16,440 | 104,420 | mixed binary | rows of 2.7 |
| A2 | rmatr200-p5 | 37,617 | 37,816 | 113,048 | mostly continuous | rows of 3.0 |
| A2 | satellites2-60-fs | 16,516 | 35,378 | 125,048 | mixed binary | |
| A2 | unitcal_7 | 48,939 | 25,755 | 127,595 | mixed binary | rows of 2.6 |
| A2 | brazil3 | 14,646 | 23,968 | 133,184 | integer | |
| A3 | piperout-d27 | 17,470 | 13,104 | 224,457 | general mixed | |
| A3 | uccase9 | 49,565 | 33,242 | 332,316 | mixed binary | |
| A3 | sorrell3 | 169,162 | 1,024 | 338,324 | binary | tall, rows of 2.0, columns of 330 |
| A3 | sp98ar | 1,435 | 15,085 | 426,148 | binary | wide, rows of 297 |
| A3 | supportcase42 | 18,439 | 19,466 | 435,653 | mostly continuous | |
| B1 | triptim1 | 15,706 | 30,055 | 515,436 | general mixed | |
| B1 | supportcase6 | 771 | 130,052 | 584,976 | integer | wide, rows of 759 |
| B1 | neos-4306827-ravan | 261,877 | 162,291 | 639,474 | general mixed | rows of 2.4 |
| B1 | var-smallemery-m6j6 | 13,416 | 5,608 | 850,621 | mixed binary | columns of 152 |
| B1 | rd-rplusc-21 | 125,899 | 622 | 852,384 | mixed binary | tall, columns of 1,370 |
| B2 | bab6 | 29,904 | 114,240 | 1,283,181 | binary | |
| B2 | splice1k1 | 6,505 | 3,253 | 1,761,016 | integer | rows of 271, columns of 541 |
| B2 | savsched1 | 295,989 | 328,575 | 1,770,507 | mixed binary | |
| B2 | supportcase12 | 166,781 | 799,616 | 2,334,437 | mostly continuous | |
| B2 | supportcase7 | 6,532 | 138,844 | 2,845,545 | mostly continuous | wide, rows of 436 |
| B3 | neos-5116085-kenana | 690,242 | 509,236 | 3,032,792 | mostly continuous | |
| B3 | neos-4647030-tutaki | 8,382 | 12,600 | 3,953,388 | mixed binary | rows of 472, columns of 314 |
| B3 | neos-5114902-kasavu | 961,170 | 710,164 | 4,240,376 | mostly continuous | |
| B3 | supportcase19 | 10,713 | 1,429,098 | 4,287,094 | integer | wide, rows of 400 |
| B3 | neos-5052403-cygnet | 38,268 | 32,868 | 4,898,304 | binary | columns of 149 |
| C | usafa | 1,377,561 | 228,648 | 16,485,102 | integer | |
| C | a2864-99blp | 22,117 | 200,787 | 20,078,717 | binary | rows of 908 |
| C | neos-3740487-motru | 4,936,372 | 1,689,612 | 26,433,474 | mostly continuous | |
| C | square47 | 61,591 | 95,030 | 27,329,856 | integer | dense |
| C | t11nonreg | 27,375 | 4,596,905 | 46,841,437 | general mixed | wide, rows of 1,711 |
| C | neos-4332810-sesia | 19,912,111 | 20,682,487 | 47,513,027 | general mixed | rows of 2.4 |
| D | dlr2 | 7,132,926 | 38,868,107 | 78,091,589 | mostly continuous | columns of 2.0 |
| D | hawaiiv10-130 | 1,388,052 | 685,130 | 183,263,061 | mixed binary | dense |
| D | neos-4535459-waipa | 163,001 | 961,401 | 256,963,402 | mixed binary | rows of 1,576 |

Sub-bands: A1 5e4 to 1e5, A2 to 2e5, A3 to 5e5, B1 to 1e6, B2 to 3e6, B3 to 1e7, C to 5e7, D above.
Of the ten largest instances of version 1 the set keeps five (waipa, hawaiiv10-130, dlr2, sesia,
t11nonreg).

Status instances: bnatt500, fhnw-binpack4-4, neos-2075418-temuka (infeasible); neos-3230511-yuna
(unbounded).

Held-out set (36, same script with the main set's families excluded; not looked at before Stage 4;
blp-ir98 replaces elitserienhandball3i, which has indicator constraints):
neos-1171737, neos-3656078-kumeu, net12, neos-3216931-puriri, blp-ir98, neos-950242,
neos-827175, neos-1122047, neos-933966, neos-960392, supportcase33, thor50dday, tbfp-network,
neos-787933, neos-860300, supportcase10, nw04, supportcase34, rail02, roi2alpha3n4,
physiciansched3-3, neos-848589, s250r10, ns1760995, roi5alpha10n8, n3seq24, kosova1, tpl-tub-ss16,
supportcase2, neos-3025225-shelon, proteindesign121hz512p19, rfds-4-days, neos-4545615-waita,
scpn2, supportcase38, supportcase11.

Baseline notes: HiGHS needs the 64-bit build (its clique limit 10 * nnz overflows 32 bits on
waipa); stock HiGHS reports no per-rule counts for MIP and cannot be restricted to a rule subset,
hence the patch; its written presolved model is power-of-two scaled.
