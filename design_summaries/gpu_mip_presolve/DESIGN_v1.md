# GPU MIP presolve: design plan

Date: 2026-11-26. Status: plan for approval; nothing is built, downloaded or run yet.

How this was produced: code studies of cuOpt, HiGHS (commit 283bfc859), PaPILO 3.0.3 and SCIP; a
literature study; three independent designs, each attacked by a soundness and a feasibility reviewer;
then four reviewers on the merged plan. Every counterexample they found is folded into the rules
below. Numbers about HiGHS, PaPILO, cuOpt and MIPLIB were read from source or tables. Every number
about our own speed, memory and reduction counts is an estimate until Stage 0 and Stage 1 measure it.

## 0. Decisions for you

1. Testbed reading. "Largest 10 in MIPLIB" is read as the ten largest instances of the MIPLIB 2017
   collection by nonzeros (section 14). Two of them (yuna, yass) are unbounded, three are open, and
   there are two twin pairs, so I add four: square47, motru, neos-3208254-reiu, ivu06-big.
   The other reading, the ten largest of the 240-instance benchmark set, has only two instances above
   1e7 nonzeros (square47, square41) and would not test the large-problem goal.
2. Downloads: the 14 instance files (and their MIPLIB solution files where they exist) from
   miplib.zib.de to the large scratch, after checking which are already in the dataset directory.
3. HiGHS: harness only, built from source at the commit studied, with 64-bit integers, plus a small
   diagnostic patch kept as a diff. It is never linked into the presolve library.
4. PaPILO: cuOpt's pinned commit, in the harness for the baseline and the CPU tail measurements.
5. Location: a new directory with its own CMake project inside a new cuOpt worktree from upstream
   main, so the existing sync and dev container are used. Not this worktree.
6. Scope approved now: Stages 0 and 1. Later stages are conditional on their gates.

## 1. Goal and honest expectation

Goal: remove more than HiGHS presolve and with better quality, fast on 1e7 to 1e8 nonzeros.

- Time: a clear win is likely on the large instances (full sweeps at memory bandwidth against
  sequential linked-list passes). Not measured yet; HiGHS has not been timed on the testbed.
- Reductions: parity is the realistic base case. HiGHS's pairwise column scans are complete, its
  doubleton and singleton cascades run to completion, and its aggregator uses live counts. A surplus
  can come only from: probing coverage on sparse binary models (its contingent starts at 1,000 probes),
  implication-graph analysis, aggregator candidates behind its three-failure stop, predictive bound
  analysis beyond its 300,000-check switch, and re-running stages it calls once. Each is unmeasured;
  Stage 0 measures the ceilings before anything is built for them.
- Quality: activities recomputed exactly instead of updated incrementally, transaction-level numeric
  vetoes, and measured LP bound and coefficient dynamism.
- A pure batch scheme ends below a sequential presolve on cascades and ordered aggregation. The design
  is therefore a hybrid: GPU bulk engine, cascade loop with a host worklist, sequential CPU tail.
- The standard cluster benchmark cannot show this work: only 30 of the 240 benchmark instances have at
  least 1e6 nonzeros and 2 have at least 1e7. The collection has 159 and 27.

## 2. Architecture

- Device matrix: CSR (column id int32, value f64) and CSC (row id int32, value f64, position of the CSR
  twin). 32-bit ids, 64-bit offsets, about 28 to 32 B per nonzero. Plain arrays.
- Deletion: alive flags; an entry tombstone is value 0.0 with the index kept. Fixed columns stay in
  their rows with lb = ub until the next rebuild, where they are removed and their records written.
- Fill-in is never inserted in place. A structural batch is planned on the immutable matrix and
  materialised by a rebuild (compaction and transpose). Entries of fixed and deleted columns count as
  dead; rebuild also above 20 to 30% dead entries.
- State per row about 90 B, per column about 150 B (estimates): sides, bounds, cost, type, locks,
  double-double activities with infinity counters and error bound, implied bounds with source row and
  its version, reservation cells.
- Rows and columns are binned by length; above 4,096 entries they are split into chunks with a
  second-level reduce. Long columns use segmented reduces, never atomics. Pairwise column kernels skip
  columns above a length cap and report them.
- Host mirror: CSR and CSC with host copies of bounds, sides, alive flags, locks and live lengths
  (24 to 28 B per nonzero), refreshed lazily at the first hand-over after a rebuild and kept current by
  dirty lists.
- Device peak = working set + one rebuilt matrix + two sort buffers, about 80 B per nonzero plus
  state, reserved up front (waipa about 21 GB, dlr2 13 to 19 GB; estimates). Caps that change the
  result are functions of the model and a configured memory limit, never of live free memory. On an
  allocation failure a phase is skipped and logged.
- Budgets are deterministic work units and bytes, plus a wall-clock limit and a halt flag checked at
  round boundaries (a partial result is valid).
- Size gate: below a measured crossover the existing CPU presolve runs unchanged; when the live model
  shrinks below it, it is downloaded and finished on the CPU. Sub-MIPs stay on the CPU path. The
  crossover is calibrated on the collection instances above 1e6 nonzeros plus a size ladder below.

## 3. Rounds and application order

Each phase proves its candidates on the state left by the previous phase.

1. Cascade loop (no fill-in): classes B and R, then L on recomputed locks, repeated on dirty lists to
   an empty frontier or a unit budget, never to a percentage. Push or pull by frontier density. When
   the frontier stays small, the host worklist takes over: it runs the whole no-fill rule set with the
   same guards and returns an ordered log (bound changes, deletions, side and coefficient edits,
   records) that the device replays. The switch threshold comes from Stage 0 measurements.
2. W and D candidates are proved on the post-cascade state and resolved in one reservation table.
3. Structural batches, one kind per batch, each followed by a rebuild: aliases; column merges and
   column transforms; substitutions; row combinations. Non-monotone bound writes (merge widening,
   integer shift) happen only here. A batch is materialised only if it changes at least
   max(1e-3 of live nonzeros, 1e5 entries); per kind, batches stop when the yield falls below 1% of
   the first or after 16.
4. Stall = a full pass that removes less than 0.05% of rows plus columns (HiGHS's threshold). Then the
   pairwise and global phases, then the CPU tail, then at most two more alternations.

CPU tail, two parts: (a) a sequential aggregator on the host mirror (sorted candidates, live row and
column sizes, size 2 always taken, net fill-in at most 10, singleton rows removed after each
substitution), fed with the residual candidates of the GPU batches and budgeted in work units;
(b) inside cuOpt, the existing PaPILO configuration, unrestricted, on the residual model, so the
composite is not below today's presolve by rule set and cuOpt's custom presolvers stay. Every result
is reported in three columns: GPU only, GPU plus sequential aggregator, GPU plus full tail.

## 4. Validity classes

| Class | Members | Rule |
|---|---|---|
| B, bound facts | bound tightening, fixing, infeasibility, probing and implication fixings, implied integers by the equation test | True for every feasible point; applied together by min/max. |
| R, self-witnessed row edits | redundant row, forcing-row deletion, integral scaling, side rounding, gcd, Chvatal-Gomory rounding, coefficient strengthening, cover strengthening, tiny-coefficient removal | Uses the row and explicit bounds only. One composed edit per row per round by the row's own thread, each step on the result of the previous, redundancy re-tested last. |
| L, lock arguments | dual fixing by locks, empty columns | Evaluated on locks recomputed after B and R. |
| W, witnessed row edits | parallel rows (claims both rows), dominated set-packing rows, clique merging, sparsify | In the reservation table: writes its row, reads its witness. |
| D, dual reductions | strict row-dual conclusions (implied equations, dominated and forcing columns), singleton-column rules (ranged rows and equations only), stuffing, pairwise dominated columns, dual-fixing extensions, weak implied-integer test | In the reservation table. |
| S, structural | alias forest; parallel-column merge and gcd column transform; substitution; row combination | One kind per batch; acceptance per transaction by a numeric dry run. |

Reservation table: cells of five kinds (row, column entries, column bounds, cost, type). Any change of
an entry, side or liveness of row i write-claims row i and the column-entries cells of its columns.
Any proof that compares or counts entries of column j reads its column-entries cell. Keys are
(hash, candidate id), 64 bit, unique. Each cell keeps minimum writer and minimum reader; winners set a
taken bit; selection is iterated to a maximal independent set; a loser leaves no trace. A table of read
and write cells per reduction is part of Stage 2, and every accepted batch is replayed one transaction
at a time on the CPU in tests.

Guards (each was broken by a counterexample in review):

- Dual arguments use continuous columns only, strictly beyond the dual tolerance; the column bound
  that drops a side is infinite or strictly implied; at most one implied premise per derivation;
  row-dual bounds are never carried across a structural batch.
- An integer column is substituted only if the pivot row proves integrality (all other columns true
  integers, integral ratios and right-hand side). This is narrower than HiGHS's any-equation test.
- Type order integer > implied integer > continuous. A representative standing in for a true integer
  becomes a true integer.
- Parallel columns: the HiGHS/PaPILO no-hole conditions; the merged column gets the widened bounds and
  its derived records are reset. Binary classes of ratio +-1 first.
- Dominated columns: dominating bound infinite, or implied by a named row in the read set, or both in
  a common clique row in the read set; an integer never dominates a continuous column; only odd-depth
  columns of the domination forest are fixed.
- "Bound is implied" = one hop from one named row with explicit bounds, row in the read set, record
  void when the row's version changed.
- Coefficient strengthening: integer columns only, exactly one finite side, finite max activity M,
  M - b > feastol, |a_k| > M - b + feastol. (M - b is invariant, so one pass equals the sequential
  result.) Forcing-row tolerance is feastol / dynamism.
- Alias forest: a deterministic spanning forest on the undirected edges, each tree rooted at a
  highest-type column; an edge whose orientation is not admissible (type order, integrality of ratio
  and shift, magnitude) is cut, its child becomes a root and its row stays; maps are composed on the
  frozen tree in double-double with double-buffered pointer jumping; non-tree edges stay as rows.
  Aliases and merges are separate kinds.
- Substitution batch: each pivot (p, j) has a premise set = p plus every row named by the implied-bound
  proofs of j. No pivot column of the batch may appear in a row of another pivot's premise set (with
  premise set {p} this is the diagonal pivot block, and the batch is one Schur-complement update).
  Pivot rows: equations, strictly implied equations, and weakly dual-implied rows whose dual bound has
  the pivot column as source. Other weak rows go to the sequential tail. Row or column of size 2 always
  taken, pivot at least 0.01 of the row or column maximum, net fill-in at most 10, all candidates
  tested. A pivot is vetoed, in all its rows, when a result row's max/min |a| exceeds max(old, 1e6).
  The cost row is part of every batch.
- Cancellation: a combined entry is a structural zero if |t + s*p| <= 1e-12 (|t| + |s*p|); a small
  coefficient is dropped only with the side shifted by a*l and a per-row budget of 0.1 feastol.

## 5. Rule coverage against HiGHS

| HiGHS rule | Here |
|---|---|
| Infeasible, redundant, empty, singleton, forcing rows; implied column bounds; fixed and empty columns | B, R, L. Stage 1 |
| Integral scaling, side rounding, Chvatal-Gomory, coefficient strengthening | R. Stage 1 |
| Doubleton equations | Alias forest. Stage 1 |
| Parallel rows | W. Stage 1 |
| Dual fixing by locks | L. Stage 1 |
| Implied equations, dominated / weakly dominated / forcing columns, row-dual bounds | D. Stage 2 |
| Singleton columns (free substitution, stuffing, zero-cost singleton) | D and S. Stage 2 |
| Pairwise dominated columns, parallel columns | D and S. Stage 2 |
| Aggregator | S batches plus sequential tail. Stage 2 |
| Simple probing on equations, gcd substitution, implied integers, VLB/VUB extraction, dual-fixing extensions | Stage 2 |
| Probing, implications, clique merging, clique and implication substitutions | Stage 3, gated |
| Sparsify, cover strengthening | Stage 3 |
| Fourier-Motzkin, enumeration of small binary rows, VLB/VUB aggregation, dual-fix probing | Left out of the first versions |
| Final power-of-two scaling | Not part of presolve; both models are equilibrated the same way before comparing |

Absent in HiGHS and available here later: components (reported only), implication-graph components.
Left out: dual postsolve, dependent rows, network-matrix implied integers, two-row bound tightening and
two-column cancellation (near-zero measured value), symmetry, multi-GPU.

## 6. Numerics

- Tolerances are inputs (feasibility 1e-6, dual 1e-7, zero 1e-9, integrality as cuOpt passes it), with
  a tolerance table per rule.
- Activities used for a committed reduction are recomputed from scratch: exact products, double-double
  sums, infinity counters, fixed order, and an error bound E derived for that summation. An integer
  bound is accepted only if E/|a_j| <= 0.1 feastol; no bound is derived when |b| * 1e-14 > feastol.
- Implied bounds are recorded with source row and version when they improve the record by more than
  1000 * feastol. Integer columns: the explicit bound is tightened. Continuous columns: explicit bounds
  are not touched during cascade, dual and structural phases; recorded bounds are made explicit once,
  before probing or at the end (what HiGHS does).
- Whether bounds at or above 1e8 count as infinite (PaPILO's rule, not HiGHS's) is decided after
  Stage 0 counts the rows concerned.
- No floating-point atomics: integer flags and counters for filters, one thread per row or
  sort-then-reduce for sums.

## 7. Probing, implications, cliques (gated by Stage 0)

- Stage 0 measures: rows with exactly two binary entries and what a CPU strongly-connected-components
  pass finds on that graph; deductions against probe index on a score-ordered prefix with HiGHS's
  limits lifted; what cuOpt's probing cache already finds.
- Implication graph on the device: components give equivalences (to the alias forest) and
  contradictions; reachability fixings only for a budgeted, score-ordered subset.
- Batched probing on one read-only state. Set-packing, partitioning and covering rows propagate
  implicitly. Bit-sliced multi-source state only where (2 * binaries + rows) * 8 B times the number of
  batches fits the budgets (few-column instances); on sesia, seret, supportcase11 a sparse per-source
  frontier with a per-source quota and epoch tags. A source closed by a cap is distinguished from one
  closed by a conflict. First step of Stage 3 is CPU-parallel probing on a snapshot.
- Cliques: extraction per row by coefficient sort; table capped by a function of the model; merging
  with seed selection by independent set; rerun only after more than 1% new edges.

## 8. Postsolve, crush, status

- Records only where a column disappears or changes meaning: fixed value, fixed at infinity (saved
  rows, one owner column per row), substitution, alias, merge chain, singleton column with its row,
  stuffing, integer shift/scale.
- Invariant: a record's saved row contains no column whose record is undone later (fixed columns stay
  in rows until the rebuild that writes their records). Undo: fixed-value records of a batch first,
  then the rest in reverse order of application; host-worklist segments sequentially. Integers are
  snapped before the undo. With a tail: PaPILO undo, snap, GPU undo, verify.
- The result is checked against original rows, bounds and integrality (optional per call); a failed
  check triggers repair or rejection.
- Primal-only mode: L, D, dual-implied pivot rows, the weak implied-integer test and column dominance
  are off, the tail runs without dual reductions, and every original-feasible point crushes exactly to
  a reduced-feasible point of equal objective. Used when a set-solution callback is registered, as
  cuOpt does today. Otherwise crush is replay plus clamp with validation downstream.
- Status: REDUCED, UNCHANGED, REDUCED_TO_EMPTY, INFEASIBLE, UNBOUNDED_OR_INFEASIBLE, LIMIT (valid
  partial result), FAILED (fall back to the CPU path). A cost-driven move to an infinite bound returns
  UNBOUNDED_OR_INFEASIBLE.

## 9. Interface

Input: CSR, row sides, bounds, types, cost, offset, sense, tolerances; minimisation internally.
Semi-continuous and indicator reformulation and cuOpt's row scaling happen before this presolve.
Output: reduced model with offset and sense, types with a separate implied-integer flag, column map
with a "transformed" marker, status, postsolve object with a host entry callable from several threads,
and by-products (fixings, implications, clique rows, variable bounds) in reduced indices; every
structural batch and rebuild remaps or drops them. Hand-back to cuOpt asserts nnz < 2^31.

## 10. cuOpt constraints

- Early GPU FJ and its problem copies hold at least about 120 B per nonzero during presolve (waipa
  about 31 GB). Next to the presolve peak this exceeds a 48 GB card, so early GPU FJ starts only if
  both fit, otherwise after the first rebuild or on the CPU only.
- Pipeline on the GPU path: reformulation and scaling unchanged, GPU presolve in place of the PaPILO
  call, cuOpt's custom presolvers and indicator strengthening in the tail, the post-PaPILO GPU stages
  unchanged in the first integration.
- One feasibility tolerance on both sides of any A/B (cuOpt runs PaPILO at 1e-5 today).

## 11. What "better than HiGHS" means

HiGHS at a fixed commit, 64-bit build, one-hour limit, status recorded; an instance where HiGHS does
not finish is reported separately. Both models are brought to one convention first (minimisation, same
equilibration, implied integers counted separately; HiGHS's file is power-of-two scaled).

| Measure | Tool | Rule |
|---|---|---|
| 1. Size: rows, columns, nonzeros, integer columns | model files | Shifted geomean, twins weigh one half |
| 2. LP relaxation bound | one cuOpt LP solver, time cap | Band = 10 x solver tolerance; "not available" allowed |
| 3. Numerics: max and 99th-percentile row dynamism | model files | At most a stated factor above HiGHS |
| 4. Correctness | postsolve check, known optima, crushed MIPLIB solutions | Clean |
| 5. Solver outcome | cuOpt with presolve off on both models | Seeds per arm stated; inconclusive inside the noise band |
| 6. Seconds | wall time, host arrays to usable reduced model, transfers included | H200 and L40S separately; thread counts stated |

Better = 1 lower, 2 and 3 inside the band or better, 4 clean, 5 not worse, 6 lower.

## 12. Stages, gates, size

Sizes are the reviewers' estimates in engineer-weeks for a person; they show relative scale.

| Stage | Content | Gate |
|---|---|---|
| 0. Baseline, ground truth, cost model (about 1 to 2) | Instances; HiGHS (stock and patched) and PaPILO on each: seconds per stage, reductions per class, probes and stop reason, substitution counts; instance structure (row classes, length histograms, hub columns, cascade depths); probing and implication ceilings; which LP solver finishes; micro-benchmarks on H200 and L40S (row sweep, column sweep, rebuild, hand-over round trip, reservation kernel); peak-memory table | Target table per instance. A research item is built only if its ceiling removes at least 1% of columns or nonzeros on at least one structure |
| 1. Core (about 8 to 10) | Matrix, tombstones, rebuild; B, R, L; cascade loop with host worklist; integer row rules; alias forest; parallel rows; postsolve; statuses; test oracle; determinism hash. Ends with an integration spike in cuOpt on waipa, dlr2, sesia | Per instance at least 99% of the rows, columns and nonzeros removed by HiGHS restricted to the same rules (patched build), and faster. Go/no-go on the project |
| 2. Dual and structural (about 15 to 20) | Reservation table; D reductions; dominated and parallel columns; singleton columns; aggregator batches; sequential aggregator tail | Substitutions and remaining nonzeros on dlr2 and a second instance chosen in Stage 0: at least 95% of HiGHS without the tail, 100% with it |
| 3. Research items that passed Stage 0 | CPU-parallel probing, implication graph, GPU probing, cliques, sparsify, cover strengthening, components | Each item adds removed columns or nonzeros on at least two structures inside its time budget, else dropped |
| 4. Tuning and validation | Schedule tuning on the testbed; validation once on the held-out collection instances above 1e7 nonzeros | Section 11 on the held-out set |
| 5. cuOpt integration | Device hand-off, composed postsolve and crush, primal-only mode, limits, setting | A/B on two sets reported separately: the 30 benchmark instances above 1e6 nonzeros and the large collection instances; the 210 smaller ones once as a no-change check |

Every stage ends with a forced-GPU correctness sweep over the MIPLIB benchmark set.

Test oracle: tiny random MIPs solved by enumeration, including infeasible, unbounded and empty
outcomes, scaled coefficients and huge bounds, multi-round sequences, and every size threshold as a
test parameter so each path runs on tiny models. Checks: reduced optimum equals the original, every
reduced-feasible point postsolves to an original-feasible point, exact crush in primal-only mode,
exported implications and cliques valid, batch result equals sequential replay, composed postsolve
with a PaPILO pass. A sequential reference for B and R on the host mirror must reach the same fixpoint
on mid-size instances.

## 13. Expectations per instance (from mean lengths only; unmeasured)

- waipa, hawaiiv10-130, square47 (mean row length 132 to 1,576, mean column length 267 to 288):
  sweeps dominate; time win expected, reductions at parity, probing coverage no better than HiGHS.
- dlr2 (98.6% continuous, mean column length 2.0, mean row length 11.0): decided by the aggregator
  and singleton columns; risk of fewer substitutions than HiGHS without the tail.
- motru (95.5% continuous, mean column length 15.6): class unknown until Stage 0.
- sesia, seret, supportcase11 (mean row length 2.1 to 2.4): alias forest and implication graph; the
  place where coverage may add reductions; depends on the Stage 0 ceilings.
- yuna, yass (19M rows with mean length 3.0 over 99K columns; unbounded): status, size and time only.
- ivu59, t11nonreg, ivu06-big (millions of columns over few long rows): parallel and dominated
  columns; HiGHS's scan is complete, so the expected gain is time.

## 14. Testbed

| # | Instance | Status | Rows | Columns | Nonzeros |
|---|---|---|---|---|---|
| 1 | neos-4535459-waipa | open | 163,001 | 961,401 | 256,963,402 |
| 2 | hawaiiv10-130 | easy | 1,388,052 | 685,130 | 183,263,061 |
| 3 | dlr2 | open | 7,132,926 | 38,868,107 | 78,091,589 |
| 4 | neos-3230511-yuna | easy, unbounded | 19,335,696 | 99,472 | 58,613,680 |
| 5 | neos-3229051-yass | easy, unbounded | 18,846,096 | 99,472 | 57,144,880 |
| 6 | neos-4332810-sesia | easy | 19,912,111 | 20,682,487 | 47,513,027 |
| 7 | neos-4332801-seret | easy | 19,912,111 | 20,682,487 | 47,513,027 |
| 8 | t11nonreg | easy | 27,375 | 4,596,905 | 46,841,437 |
| 9 | ivu59 | open | 3,436 | 2,569,996 | 36,186,332 |
| 10 | supportcase11 | easy | 15,270,211 | 8,066,286 | 32,602,332 |
| + | square47 (benchmark set) | easy | 61,591 | 95,030 | 27,329,856 |
| + | neos-3740487-motru | hard | 4,936,372 | 1,689,612 | 26,433,474 |
| + | neos-3208254-reiu | open | 9,464,908 | 3,154,336 | 25,080,218 |
| + | ivu06-big | open | 1,177 | 2,277,736 | 23,125,770 |

Baseline notes: HiGHS must be built with 64-bit integers (its clique-table limit 10 * nnz overflows
32 bits on waipa). Stock HiGHS cannot be restricted to a rule subset (presolve_rule_off has no bit for
several rules) and reports no per-rule counts for MIP, hence the diagnostic patch. The headline
comparison is against stock HiGHS; the patched build is for diagnosis and the Stage 1 gate.
