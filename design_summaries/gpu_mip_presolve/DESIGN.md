# GPU MIP presolve: design plan (version 4.1)

Date: 2026-11-26. Status: Stage 0 is measured on the 31 testbed instances from 5e4 to 1e7
nonzeros; HiGHS on the nine instances above 1e7 nonzeros and the probing run with HiGHS's limits
off are still running; Stage 1 has started.

Version 4 = version 3 plus the Stage 0 results and the decisions taken on them. Changed against
version 3: measured HiGHS times, rule worth and GPU costs entered (sections 1, 2, 5, 7); gates
fixed (time rule, median per band, ratio of remaining nonzeros; unscaled models standalone,
cuOpt's scaling on and off on the cluster; 1, 10, 12, 13); more switches for the Stage 1
reference before its gate; lane groups, one summation order, single-infinite position, radix-sort
rebuild, upload after the first host cascade (2.2 to 2.4); thresholds provisional until Stage 1
(2.1, 2.5); clique merging built in Stage 2 with its own gate, a work budget for every phase
(2.1, 7); Stage 1 split into host engine, then device propagation (13).

Version 3 is kept as [DESIGN_v3.md](DESIGN_v3.md), versions 1 and 2 as [DESIGN_v1.md](DESIGN_v1.md)
and [DESIGN_v2.md](DESIGN_v2.md); [HISTORY.md](HISTORY.md) records what changed and why. This
draft was checked once against the report and version 3; section 0 lists what is open.

Sources of the numbers. "Measured in Stage 0": from
[STAGE0_REPORT.md](../../gpu_presolve/results/stage0/STAGE0_REPORT.md) ("the report"), one H200,
HiGHS 1.16.0 at commit 283bfc859 with 64-bit integers and one thread, 31 instances unless said
otherwise. "Provisional": a Stage 0 value that Stage 1 measures again. "Not measured": an estimate
or an open value. Numbers about HiGHS's rules and limits, PaPILO, cuOpt, MIPLIB and Grace Hopper
were read from source, tables or papers. Our own presolve has no measured number yet.

## Amendments of 2026-11-27 (version 4.1)

Version 4 as it stood on 2026-11-26 is kept as [DESIGN_v4.md](DESIGN_v4.md). The sections below
are not rewritten; where one of these amendments contradicts them, the amendment holds.

**A1. Reductions gate.** The gate is a lower bound only (owner: "our final aim is to remove more
than highs"). An instance where the engine removes much more than the reference is listed, never
failed; the correctness check decides whether it is right.

**A2. Row addition in Stage 1.** The rule that subtracts an equation from a row parallel to it up
to singleton columns (HiGHS's equality row addition) moves from Stage 3 to Stage 1. With it and
the rounds on explicit bounds the Stage 1 engine is at 1.0028 against the reference over the 40
instances (0.9955 before), nothing below 0.95.

**A3. Scaling.** cuOpt's scaling stays off for all development, on the cluster as well; whether
to add scaling, and where, is decided at the end. This replaces "every cluster run with scaling
on and off" in section 10. One measurement is made now so that the size of the postponed question
is known: cuOpt as is with scaling on against scaling off. cuOpt scales before its presolve
(`solve.cu`), so the question at the end is likely "scale after the presolve", not only on or
off.

**A4. Where the GPU works in Stage 1.** Measured on the finished host engine (40 instances,
75.0 s): building the state 45.7%, rules 35.8% (bound propagation 6.9%), writing the reduced
model 8.1%, its hash 8.0%. Propagation after the first host cascade, where section 3 and stage
1b put the GPU, is almost empty: the first call holds 92.5% of the propagation work, and no call
on the 17 local instances passes the selection rule. Inside the state build the transpose with
the cross positions and the locks is 86 to 94%. So Stage 1b becomes: the device builds the state
(transpose, cross positions, locks, then the first activity sums), each with the bits of the
host, so that the choice of the engine cannot change a result and needs only a size switch.
cuSPARSE may be used for the transpose, SpMV and other sparse work (owner). Of the device
propagation design (`stage1b_design/`) only what Stage 2 reuses is built now: memory resource,
device image, bit-equal row sums, hand-over. Frontier forms and the threshold measurements wait
for Stage 2. This replaces "the model exists on the host only until the first cascade is done"
(2.2, 2.3, section 3 item 1).

**A5. Engine assertions and the cluster.** The assertion build and the structure check run on
all 240 benchmark instances on the dev node (`testbed/benchmark240.json`, minutes). A cluster
build therefore carries no engine assertions.

**A6. cuOpt's own reductions come along.** cuOpt has five reductions of its own. Four live inside
the PaPILO call and are lost when the new presolve replaces it; this replaces "do not come
along; Stage 5 ports them or drops them on evidence" in section 10.

| Reduction | What it does | Where it runs today | Needs | Way in this plan |
|---|---|---|---|---|
| Indicator strengthening (`indicator_strengthening.cpp`) | Finds indicator binaries with their members; appends implied indicator rows and lifts capacity rows by the indicator that bounds all members | On the PaPILO problem, before PaPILO's rules | A matrix it can append rows to | Now: runs in adapter mode 3 after the new presolve. Port: a first phase that appends rows (the row overlay of 2.2), before the cascade as today; no postsolve record. Stage 3 |
| GF2 (`gf2_presolve.cpp`) | Solves systems of parity rows over GF(2); fixes the variables that every solution determines (solves enlight_hard completely) | PaPILO method, MIP only | Its core `gf2_solve` is a free function on packed bit rows | Now: mode 3. Port: a rule that collects parity rows, calls the same core and fixes columns (fixed-column records). Stage 3 |
| BHW coefficient reduction (`bhw_coeff_reduce.cpp`) | Rewrites a one-sided all-binary row with smaller integer coefficients and the same 0/1 feasible set | PaPILO method, MIP only, delayed | Its core `bhw_reduce_row` is a free function | Now: mode 3. Port: a row rule next to the tightening family; a row edit, no postsolve record. The cheapest of the four, a candidate for the end of Stage 1 |
| Single-lock dual aggregation (`single_lock_dual_aggregation.cpp`) | For a binary with one lock, proves an implication from the locking row and substitutes x = y | PaPILO method, only without dual postsolve | Locks, row activities, an alias record | Now: mode 3. Port: with the dual reductions of Stage 3; the alias forest already holds x = y |
| Block BVE (`block_bve.cu`) | Eliminates blocks of zero-objective binaries by projecting them onto their boundary as no-goods, on the GPU, from the probing implication graph | In cuOpt's own presolve, after the third-party presolve | The probing cache's implications, keyed by original column | Not affected by replacing PaPILO. Stage 2 replaces the probing cache, so its implication export must have the form block BVE reads. Later a wide phase of the new presolve (its GPU projection is liftable) |

Adapter mode 3 ("new presolve, then cuOpt's own reductions"): after the new presolve the PaPILO
framework runs with the existing reduction allowlist set to `indicatorstrengthening`,
`gf2presolve`, `bhwcoeffreduce` and `single_lock_dual_aggregation`; no PaPILO rule runs. It is the
arm for cluster comparisons, so that a difference to cuOpt as is comes from PaPILO's rules against
ours and not from these four. PaPILO stays in the build only as their host until the ports exist.
Open with mode 3: indicator strengthening sees the model after our reductions instead of before;
whether that loses indicators is measured in the pre-flight (count of appended and lifted rows in
both orders).

**A7. Stages and the run-time loop.** "Stage" in this plan is a development stage. At run time
the rules of all stages are phases of one loop (section 3): the cheap cascade and the pairwise
and structural phases run until they stall, then the wide phases (probing, cliques, and the
special reductions above) run, and when they changed the model the loop goes back to the cascade,
for a fixed number of alternations.

**A8. Order of the rules after Stage 1 (owner, 2026-11-27: "Yes, cheap rules first").** Measured
on the 240 benchmark instances (presolve only, no scaling; geometric mean of remaining nonzeros
without ex9, ex10 and enlight_hard, which a full presolve reduces to almost nothing): our engine
0.788, HiGHS with our Stage 1 rule set 0.789, HiGHS with all cheap rules 0.753, HiGHS full 0.676,
PaPILO with cuOpt's reductions 0.689. HiGHS spends 65% of its time in probing and enumeration for
under 2% of the nonzeros in the mean. The order is therefore by worth on the benchmark set, and
it replaces the order of stages 2 and 3 in section 13:

1. **Stage 1c, cheap column rules** (host engine): dominated and weakly dominated column by the
   bound on the reduced cost, forcing column, free column substitution, zero-cost singleton,
   column stuffing, the dual-fixing extensions, simple probing on equations, implied equations,
   implied-integer conversion, parallel columns. Reference: HiGHS with the expensive stages off
   ("all cheap rules"), on the 40 and on the 240 instances. Gate as for Stage 1: remaining
   nonzeros at least at 0.99 of that reference, nothing below 0.95 unexplained, the time rule.
2. **Sparsify and the aggregator.**
3. **Pairwise dominated columns** (the first wide phase; a GPU candidate).
4. **Enumeration, probing, cliques** (the former Stage 2; GPU-wide).

cuOpt's own reductions (A6) are ported next to the rule family they resemble: BHW with 1,
single-lock dual aggregation with 1, GF2 and indicator strengthening with 2.

Decided in two steps on 2026-11-27. After the design round and the one-rule-off runs the owner
first agreed to leave the low-worth rules out ("Yes, leave them out"), then asked for the
impact on all instances ("wait a sec. this is on testbed instances. i want to see the impact on
all instance set"), and with the numbers on the 240 benchmark instances decided: **"Build them
too"**. So Stage 1c builds all of HiGHS's cheap rules:

- the nine rules of the design, in the order of their measured worth: simple probing on
  equations; the singleton-column rules (free singleton with an equation or an inequality as
  pivot row, zero-cost singleton, forcing column); parallel columns (fixing by dominance,
  merge); dual bound strengthening; the single-lock alias (HiGHS's dual-fixing substitution and
  cuOpt's single-lock dual aggregation in one rule);
- and after them the low-worth ones, also required: dominated and weakly dominated column by
  the bound on the reduced cost, implied equations, column stuffing, implied-integer
  conversion, the single-equation extension of dual fixing. Measured on all 240 with HiGHS,
  switched off together: 1.0006 in nonzeros, 1.0015 in rows, 1.0006 in columns, 19 instances
  change (ns1208400 11%, where our engine already has the smaller size; kumeu 2.9% and doon
  2.8%, mostly the forcing column; csched007 1.6%; snowy 0.8%). They need bounds on the row
  duals as new state, which the first design avoided; the design is extended for them.

The design in force is version 2 with its amendments (`stage1c_design/v2_design.txt`,
`v2_amendments.txt`, `v2_reviews.txt`). Build order of the nine rules, by cost and by what the
later ones need: M0 groundwork and reference runs; M1 singleton columns and the forcing column;
M2 simple probing on equations; M3 dominance between parallel columns; M4 dual bound
strengthening, then the single-lock alias; M5 merge of parallel columns; then the five
low-worth rules; then the gate. What each step is worth, measured with HiGHS on 237 of the 240
benchmark instances as the geometric mean of remaining nonzeros / rows / columns against all
cheap rules on (`results/bench240/stage1c_refs`): Stage 1 rule set 0.9544 / 0.9640 / 0.9406;
with the singleton column rules 0.9726 / 0.9788 / 0.9562; with simple probing 0.9826 / 0.9892 /
0.9772; with parallel columns 0.9932 / 0.9907 / 0.9919; with the dual-fixing extensions 0.9994
/ 0.9985 / 0.9993; the five low-worth rules are the rest.

The gate of the stage measures rows and columns as well as nonzeros, on the 240 benchmark
instances and on the testbed, with the same two numbers in each measure (0.99 in the geometric
mean, nothing below 0.95 unexplained). BHW comes after the gate. Not decided by the owner,
taken as recommended until they say otherwise: parallel columns are merged when both are
continuous; two integer columns are merged only when the merged column gets the range of the
kept one back in the same step (setting `range_kept`, so two binaries stay one binary; the
settings `never` and `always` exist); the zero-cost singleton and the forcing column stay off
in primal-only mode.

**A9. Overheads (owner, 2026-11-27).** The GPU state build is integrated after the cheap column
rules. The time outside the rules (state build, storage, output, hash, adapter, transfers) is
reported with every run (`gpu_presolve/analysis/overheads.py`) and optimised only when it stands
between the engine and HiGHS on time; several optimisation paths are fine.

**A10. The goal and its order; assumptions (owner, 2026-11-27).** In the owner's words: "Take a
note of the assumptions as you go, we will test all of them at the end. remember nnz reduction
is not the main goal is to strengthen the formulation. so we will test all of them at the end
over the solution quality. strengthening might mean adding those mergings and additional
constraints like indicator. but we will test them at the end. first goal is the nnz reduction."

- The first goal is the reduction in nonzeros (with rows and columns next to it), measured
  against HiGHS presolve only. That is what every stage gate up to the end measures.
- The main goal is a stronger formulation. It is tested at the end, over the solution quality
  of cuOpt on the cluster. Strengthening may mean reductions that do not shrink the model or
  that grow it: the merges of integer columns that are held back today, added constraints such
  as indicator strengthening.
- Every assumption made on the way is written into [ASSUMPTIONS.md](ASSUMPTIONS.md) when it is
  made: defaults taken without the owner, rules or settings left off, tolerances and limits,
  conventions of the measurement, findings that later choices rest on. Each entry names its
  switch (or says that one has to be built), the alternative, and the test at the end. A
  choice that cannot be switched is built with a switch from now on, so that it can be tested.
  No entry is closed by a nonzero count alone.
- Grace Hopper nodes are available. Benchmarking and performance work there come at the end;
  until then the design keeps the unified-memory path possible and nothing is measured there.
- cuOpt's sub-MIPs keep PaPILO for now.
- The design notes (this directory) are pushed to the fork as branch `gpu-presolve-notes`.

## 0. Decisions

Decided on the Stage 0 report (2026-11-26):

| # | Decision | Sections |
|---|---|---|
| 1 | Stage 1 reference. The HiGHS reference for the Stage 1 gate is brought closer to our Stage 1: more switches in the diagnostic patch (simple probing on equations, the dual-fixing extensions apart from the lock rule, implied equations, implied-integer conversion), a counter per rule for column fixings, and one more run with HiGHS's own dual-fixing rule off, so that two runs bracket dual fixing. The cheap single-row and single-column rules stay in Stage 3 | 5, 12, 13 |
| 2 | Clique merging is built in Stage 2 (extension of set-packing rows, removal of dominated clique rows) and gated apart from probing, with the gate proposed in the report | 7, 12, 13 |
| 3 | Time rule per instance: t <= 1.2 * t(HiGHS) + 0.02 s on the minimum of at least three alternating runs. Reductions gate: ratio of remaining nonzeros, geometric mean of nnz(HiGHS reference) / nnz(ours) at least 0.99, taken over all instances, every instance below 0.95 explained | 1, 12, 13 |
| 4 | Scaling. Standalone gates are evaluated on unscaled models for both sides. Inside cuOpt the cluster runs are made with cuOpt's scaling on and with it off, so that the effect of scaling on the presolve is seen | 10, 12, 13 |
| 5 | One summation order for an activity. A frontier kernel in lane-group order is built in Stage 1 and measured; if it is slow, the scheme of version 3 (one thread per row up to 64 entries, one block per row above) is used for full and frontier sweeps | 2.4, 2.5, 13 |
| 6 | The thirteen changes proposed in section 8 of the report are accepted as worded there, with the report's limits (lane groups only in bands A and B; thresholds provisional until Stage 1 measures them again) | Entered at the places they name |
| 7 | Cluster runs may be launched without asking when a larger test is needed | 12 |

Decided earlier and not reopened by Stage 0:

| Item | Decision |
|---|---|
| Stage order | GPU-wide phases (probing, cliques) in Stage 2, the rules needed for parity with HiGHS in Stage 3 (recommendation of version 3). The Stage 0 data do not contradict this order and do not prove it (section 5) |
| Where runs happen | Dev node: all 40 testbed instances, presolve only; the 9 above 1e7 nonzeros in the background with a time limit. Cluster: the regular benchmark inside cuOpt after each stage, with the existing scripts. A standalone cluster run, when a larger test needs one, covers a wide set (the 147 benchmark instances at or above 50,000 nonzeros plus the testbed), never a handful |
| Quality against HiGHS | An extra cluster arm at Stages 3 and 5 that runs cuOpt with its presolve off on HiGHS-presolved models (recommendation of version 3) |
| Determinism across machines | One shipped set of engine thresholds, so every machine gives the same reduced model (recommendation of version 3) |
| Testbed | Section 14 |
| Baseline | HiGHS only, pinned commit, 64-bit integers, with a small patch that adds rule switches and per-rule counters; harness only |
| Location | A directory with its own CMake project in a cuOpt worktree; no cuOpt target or header is used (section 11) |
| Downloads | The 13 testbed instances that are not in the benchmark set, the status instances, and MIPLIB solution files |
| Scope approved | Stages 0 and 1 |

Open:

1. Grace Hopper. Is a GH200 node available? Without one that path is tested for function only.
2. PaPILO in cuOpt. It leaves the main-problem path. May it stay for sub-MIPs, and as a logged
   fallback in the product default, until Stage 5 decides on evidence?
3. Clique items other than merging: the export to cuOpt's clique table, clique-aware propagation
   and knapsack cliques. Decision 2 and the Stage 0 data cover merging only; version 3 made all
   clique items depend on Stage 0. They are listed for Stage 2 (section 7) and are decided with
   the approval of Stage 2.
4. Readings the plan takes where a decision leaves room, for you to overrule:
   - Decision 1: the Stage 1 gate uses the first of the two bracket runs, the one with HiGHS's
     own dual fixing still on (section 12).
   - Decision 3: "all instances" is every testbed instance that has the reference, 40 at the
     gate, band D included although its time is reported only (section 12).
   - Decision 4: the run with scaling off is made for the arm with the new code only; there is
     no baseline with scaling off (section 10).
   - Clique gate: the report's "at least 0.99 of the reference" is read as reference / ours, as
     in the Stage 1 gate (section 12).
   - The items of report section 7 that no decision names (shift 0.1 s, median per band, what is
     timed, size conventions, probing gate, status instances) are entered as the report proposes
     them (sections 1 and 12).
   - The report does not say when the items Stage 0 left open are due, or in which order Stage 1
     is built. The due dates and the order host engine, then device propagation are set in
     section 13.

## 1. Goal and expectation

Goal: on MIPs from 50,000 nonzeros up, remove more than HiGHS presolve, with better quality, in
less time; deterministic; no PaPILO inside.

What the size target implies. At 5e4 to 5e5 nonzeros a full matrix sweep costs 0.5 to 3.1 ms on
one CPU thread and 0.06 to 0.24 ms with the fastest GPU variant, and a device step costs 77 to
93 µs before any kernel work (measured in Stage 0). In bands A1 and A2 a full GPU sweep saves 0.4
to 1.1 ms against one CPU thread, about 1% of the time HiGHS needs for the Stage 1 reference
there; how many sweeps a cascade needs is not measured. The GPU can win at that size only on wide
work, where the work is far larger than the matrix: probing (the largest HiGHS stage on 12 of 31
instances, measured in Stage 0; a batched probing round is not measured), clique work, pairwise
tests, evaluation of all aggregator candidates. Cheap dependent cascades belong on the CPU.

Host engine first. At Stage 1 the cascade below 5e5 nonzeros runs on the host engine; the
thresholds of 2.1 apply where the GPU engine is enabled. The GPU contributes below 5e5 nonzeros
from Stage 2. The time to beat for the host cascade is HiGHS's initial scan plus fast loop in the
Stage 1 reference run: 21 to 102 ms in bands A1 and A2, 37 to 163 ms in A3 (measured in Stage 0).

Time target, reported and gated per band, warm (context, pools and pinned buffers exist; cold
start reported separately):

| Band | Target |
|---|---|
| A and B (5e4 to 1e7 nonzeros) | Shifted geomean (shift 0.1 s) not above HiGHS on the same rule set at Stage 1, below full HiGHS from Stage 2 on. Median per band not above HiGHS's on the same rule set. Every instance inside the per-instance rule |
| C and D (above 1e7) | Shifted geomean below HiGHS; D is reported only. HiGHS's times there are not measured yet (runs in progress); the shift for these bands is set when they are |

Rules of the time target:

- Per instance: t <= 1.2 * t(HiGHS) + 0.02 s, both sides the minimum of at least three alternating
  runs (decision 3; single runs of identical work differ by up to 15.1%, and by at most 13.4 ms
  below 1 s, measured in Stage 0). t(HiGHS) is the run the shifted mean is compared with: the
  Stage 1 reference at Stage 1, full HiGHS from Stage 2 on.
- Median per band, next to the shifted mean: rd-rplusc-21 is 126 of the 160 s of the Stage 1
  reference, so a mean below HiGHS can be reached without being faster on ordinary instances.
- What is timed. HiGHS: the presolve itself, from its "Presolving model" log line to its first
  "Presolve reductions" line, without the copy of the reduced model. Ours: from the input CSR to
  the finished reduced model. Both sides so include the build of their data structures; in HiGHS
  that, with shrink and scaling, is a median 22% of the Stage 1 reference time (measured in
  Stage 0).

HiGHS times, measured in Stage 0 (single runs; report section 2):

| Band | Instances | Full HiGHS: shifted mean | Median | Maximum | Stage 1 reference: shifted mean | Median |
|---|---|---|---|---|---|---|
| A1 | 5 | 0.20 s | 0.21 s | 0.27 s | 0.064 s | 0.054 s |
| A2 | 6 | 0.56 s | 0.30 s | 4.28 s | 0.087 s | 0.080 s |
| A3 | 5 | 2.26 s | 1.73 s | 15.3 s | 0.24 s | 0.25 s |
| B1 | 5 | 16.9 s | 18.1 s | 139.4 s | 2.5 s | 1.9 s |
| B2 | 5 | 8.2 s | 7.2 s | 22.3 s | 1.7 s | 1.9 s |
| B3 | 5 | 10.3 s | 7.5 s | 39.4 s | 3.5 s | 3.6 s |
| All | 31 | 2.72 s | 4.28 s | 139.4 s | 0.62 s | 0.29 s |
| C, D | 9 | not measured | | | not measured | |

The Stage 1 reference is defined in section 12; its times are provisional until it is run again
with the switches of decision 1. Per-instance sizes, times and limits are the target table of the
report (section 10); the limits are provisional until the minimum of three alternating runs
exists.

Reductions: parity through our own complete rule set; a surplus from coverage where HiGHS limits
itself (probing contingent starting at 1,000, aggregator pass ending after three fill-in failures,
stages it runs once) and from clique strengthening. Full HiGHS removes 34.2% of the nonzeros of
the 31 instances (measured in Stage 0). The surplus is not measured: the probing run without
limits is in progress, the other two sources are due before Stage 3 (section 13).

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
- At Stage 1 the cascade below 5e5 nonzeros runs on the host engine (section 1). The thresholds
  below apply where the GPU engine is enabled: from 5e5 nonzeros by default at Stage 1, and at any
  size when a test or a measurement forces it.
- Thresholds, the byte budget, slot counts and every cap are constants or functions of the model
  and the settings. Nothing is read from the device (free memory, GPU type, core count).
- Every phase has a work budget and stops when work passes without a finding (25.1% of HiGHS's
  time is stage runs that remove nothing, measured in Stage 0). Pairwise dominated columns
  (Stage 3) get a hard cap; its size waits for HiGHS's count of pair tests, which the patch does
  not print yet.

Provisional defaults for the engine choice (Stage 0, report section 6). Work unit: 10 ns, the cost
of one frontier nonzero on the host; a frontier row counts 8 units.

| Threshold | Provisional default | Status |
|---|---|---|
| Full sweep on the GPU | From 32,000 matrix nonzeros | A choice above the median crossover (18,453 nonzeros per matrix with the planned kernel pair). Extrapolated: nothing below 71,477 nonzeros was measured |
| Frontier sweep on the GPU | Frontier nonzeros + 8 * frontier rows >= 20,000 work units | The best threshold lies between 9,514 and 53,817 depending on the reading. Only the activity part is measured on both sides; the whole step is modelled |
| Frontier-restricted GPU sweep, instead of recomputing all rows | Only from about 1e6 nonzeros and frontiers of 1% of the rows or less | In-sample boundaries, measured with the frontier kernel of version 3, not with one in lane-group order |
| Rebuild on the GPU | From 100,000 nonzeros | Marginal |
| Pinned mirror kept | Below 64 MiB of mirrored bytes | Placeholder, not derived |

The defaults are fixed once, after Stage 1 measures them again with the real kernels (section 13),
not in Stage 0. Not set yet: the device fixed-cost budget per call (proposed: 5 work units per
nonzero, a policy choice, not derived), the hand-back threshold, the share of the nonzeros above
which all rows are recomputed, the dead-entry fraction of a rebuild, the byte budget, the work
budget of each phase and the amount of work without a finding that ends it. Section 13 says where
each is due.

### 2.2 Shared memory

One presolve-owned memory resource with a `host_access(range, mode)` / `release` pair.

| Machine | Matrix arrays | Mutable state (bounds, activities, counters, flags, records) |
|---|---|---|
| Grace Hopper | Managed memory, read in place by the host. Host-preferred while the work estimate keeps the cascade on the host; prefetched to the GPU with the accessed-by-host hint before the first wide phase | Separate arrays, host-preferred during a host phase and prefetched to the GPU before a device phase; the host reads and writes them directly |
| x86 with a discrete GPU | Device memory. Below a size threshold (2.1) a pinned host mirror, refreshed after a rebuild only if a host phase follows. Above it no mirror: the rows and columns of the host frontier are gathered on the device and copied on demand | Dirty lists in a fixed order through pinned buffers, carrying bounds, accumulators and counters as bits; one engine owns them at a time |

The device copy of the matrix is created after the first host cascade (2.3); until then the model
exists on the host only. Host matrix edits go to a copy-on-write row overlay only. The Grace
Hopper path is selected by the device attribute for direct managed-memory access from the host and
can be forced for tests. Until a GH200 node is measured that path is functional only: no time or
memory figure is claimed for it and it is outside the gates. Facts behind the table: on Grace
Hopper `cudaMalloc` memory is not host-accessible; managed memory without the accessed-by hint
migrates 2 MB blocks on a CPU touch; kernels on CPU-resident data run at link speed; on x86
managed memory faults at every hand-over (64 MiB touched by the host after a kernel cost 14.4 ms
as managed memory and 4.0 ms as a pinned copy plus host loop, measured in Stage 0). The dirty
list has a fixed record order. The cheapest list measured in Stage 0 has none and is not used: a
hand-back costs 46 µs with it and a pinned mirror, at most about 141 µs with the ordered list and
a mirror, and about 200 µs with a gather (measured in Stage 0).

### 2.3 Matrix and kernels

- CSR and CSC with values in both, 32-bit ids, 64-bit offsets; alive flags and entry tombstones;
  fixed columns stay in their rows until the next rebuild and are folded into the side for every
  row test.
- Deleting rows or columns costs time proportional to what is deleted, also when columns are very
  long (HiGHS needs at least 69 s on rd-rplusc-21 in the call that deletes 71,300 duplicate rows
  of 7 entries, measured in Stage 0; that the cause is walks over long columns is a hypothesis).
- Structural changes are merged by a rebuild (compaction and transpose), made only when dead
  entries exceed a set fraction (not measured, due in Stage 1) or the next phase needs compact
  storage, on the engine the work estimate selects. On the device the CSC is rebuilt by a stable
  radix sort (467 µs + 0.279 ns per nonzero); when entries were only deleted, the old CSC is
  compacted instead (0.70 times that; both measured in Stage 0).
- The device matrix is built on the model after the first host cascade, not on the input: the
  Stage 1 rules alone remove at least half of the rows on 9 of 31 instances, and the length
  classes shift (rd-rplusc-21: longest column from 119,267 to 47,974 entries; measured in
  Stage 0). Above 1e7 nonzeros neither is measured.
- The matrix is stored binned by row and column length, taken from cuOpt's load-balanced
  propagation: up to 64 entries, up to 8,192 (one block), chunks above. All three classes are
  needed (rows above 64 entries hold at least half of the nonzeros on 14 of 32 matrices, rows
  above 8,192 exist on 5; measured in Stage 0). Bins are rebuilt by a stable sort, so there is no
  second copy of the matrix, and they are rebuilt after clique extension, which changes row
  lengths. Storage in binned order is kept without a measurement: the Stage 0 benchmark used a
  second, sorted copy. Stage 1 measures the layout in place, re-binning, and other boundaries than
  64 and 8,192. A frontier filter restricts kernels to changed rows and columns.

Scheduling of rows and columns up to 64 entries. A lane group is 2, 4, ... adjacent threads of a
warp that fold one row and combine by a fixed tree (the sub-warp groups of version 3).

| | Rule | Open |
|---|---|---|
| Rows | Lane groups in bands A and B. They take 0.69 times the time of one thread per row up to 64 with blocks above, and 0.10 times one thread per row (geometric mean, compensated sums, measured in Stage 0) | Above 1e7 nonzeros: on the nine matrices there lane groups take 1.14 times one thread per row up to 64 in the geometric mean (measured in Stage 0). The rule is decided in the geometric mean, not at the large end: in band B3 lane groups are fastest on 1 of 5 matrices |
| Columns | Lane groups below 1e6 nonzeros | From 1e6 nonzeros: one thread per column up to 64 is faster on 7 of 10 matrices between 1e6 and 1e7; no rule is fixed |

### 2.4 Activities

- Minimum and maximum activity per row are (hi, lo) accumulators plus exact integer counters of
  infinite terms.
- Per row and side the position of the single infinite entry is kept next to the exact counter
  (rows with exactly one infinite term, in an activity that a finite side uses, are at least 10%
  of the rows on 6 of 31 instances; measured in Stage 0).
- A bound change is applied as two terms, minus a * old and plus a * new. Each product is
  error-free (one explicit fused multiply-add) and is added by two-sum. The difference new - old,
  or of two coefficients, is never formed. A change between an infinite and a finite bound changes
  the counter and adds or removes a * bound. This is what HiGHS does.
- Compensated sums are kept without a measured benefit. Price: 39% of the activity kernel, 14% of
  a sweep (5 µs median in bands A1 and A2, 173 µs in B3; measured in Stage 0). On one fresh sweep
  the bounds differ on 5 of 32 matrices, by at most 2.2e-9.
- A row whose entries, coefficients or scale change is recomputed at once. Entry insertion and
  deletion add or remove the entry's term.
- Per row and side the update count n and the largest |a * bound| seen, T, are kept; the row is
  recomputed when n * u^2 * T exceeds 0.1 feastol (u = rounding unit). This quantity is the
  activity's error bound wherever the plan needs one. A term with |a * bound| above 1e20 is
  counted as infinite.
- On the GPU a row up to 64 entries is folded by a lane group (2.3) and a longer row by one block,
  over its entries in storage order; both combine by a tree fixed by entry position. No sort. This
  holds in bands A and B; above 1e7 nonzeros the scheduling is open (2.3).
- One summation order for an activity (decision 5). A frontier kernel that folds a row in the same
  lane-group order as the full sweep is built in Stage 1 and measured. If it is slow, the scheme
  of version 3 (one thread per row up to 64 entries, one block per row above) is used for full and
  frontier sweeps; it costs 1.45 times the lane-group activity kernel, a median of 42 µs more per
  sweep (measured in Stage 0). The reason for one order: lane groups differ in bits from a single
  thread over the row on 9 of 32 matrices.
- When the affected rows hold more than a set fraction of the nonzeros, all rows are recomputed
  with the same kernel. The fraction is not measured (due in Stage 1); the provisional rule of 2.1
  recomputes all rows below about 1e6 nonzeros and for frontiers above 1% of the rows.
- At a rebuild only rows whose entries changed are recomputed. Debug builds compare against a full
  recomputation.
- Stage 1 keeps one sum per side, on explicit bounds. HiGHS keeps a second on recorded implied
  bounds and tests row redundancy on it; whether the explicit-bound sum loses rows is not
  measured. A HiGHS switch for the explicit-bound test is due before the Stage 1 gate; the gate
  shows what the single sum costs, and the second sum is added if it is needed.

Rules taken from the review of cuOpt PR 2372 (single-infinity tightening in GPU propagation):

- Semantics: no infinite term, every column may get a candidate; exactly one, only the column that
  holds it; two or more, none.
- Layout: split arrays, narrowest types. The hi values are one array of doubles; lo is a separate
  cold array; the infinite-term counts of the two sides are two `uint16_t` arrays. No paired
  128-bit value on the hot path. The counts are exact, not capped at 2, because a capped count
  cannot be decremented; a count that saturates marks the row as skipped until it is recounted.
  Kept without a deciding measurement: split and paired arrays are equal within the noise of 12%
  (measured in Stage 0).
- No extra reduction pass for the counts: each thread counts infinite contributions during its one
  scan of the row and adds them with one integer atomic only if its local count is nonzero. In
  block kernels this takes 0.71 (plain sums) and 0.84 (compensated sums) times a second block
  reduction, measured with paired arrays; inside lane groups with split arrays the two are equal
  (measured in Stage 0).
- Residual first: the bound is (side - (activity - a_j * own bound)) / a_j in compensated
  arithmetic, never (side - activity) / a_j + own bound.
- Error guard: no candidate is emitted when the error bound can exceed the candidate tolerance.
  The lo array is loaded only when the hi value alone cannot pass the guard. Loading lo on demand
  is not decided by time (inside the same noise); the guard itself is not measured.
- Every kernel variant is benchmarked against the current one before it replaces it.

### 2.5 Determinism

- No floating-point atomics. A floating-point sum has one fixed order per row: a lane group up to
  64 entries (one thread, if the fallback of 2.4 is taken) and a block above, each with a tree
  fixed by entry position, the same in full and frontier sweeps (2.4). Library reductions are used
  for integers and min/max only. Integer atomics are allowed, because their result does not depend
  on order, and are preferred where they save a pass.
- Engine thresholds and the byte budget are settings with fixed defaults, identical on every
  machine. The defaults of 2.1 are provisional; they are fixed once, after Stage 1 measures them
  again.
- The wall-clock limit is only an abort that returns LIMIT; it never changes which reductions are
  tried. Work budgets and yield stops count work, not time. Contracted floating-point operations
  are off.
- Guarantee: the same result for the same settings on every run, thread count, co-running workload
  and GPU type.
- Test: a sequence of hashes, one after every phase, over the bit patterns of bounds, accumulators,
  matrix, postsolve stack, by-products and status, compared across repeated runs, thread counts,
  early heuristics on and off, a second GPU type and both memory paths. A second GPU type was not
  measured in Stage 0.

## 3. Rounds and application order

Each phase proves its candidates on the state left by the previous phase.

1. Cascade loop (no fill-in): bound facts and row edits, then lock arguments on recomputed locks,
   on dirty lists to an empty frontier or a unit budget. Host engine by default; GPU kernels when
   the frontier work exceeds the threshold of 2.1 (at Stage 1 only from 5e5 nonzeros). The first
   cascade of a call runs on the host; the device matrix is built on its result (2.3).
2. Witnessed and dual candidates, proved on the post-cascade state. In the host engine each is
   applied one transaction at a time with its premises checked at application (this is all Stage 1
   needs for parallel rows and aliases). GPU batches go through the reservation table. Pairwise
   scans are GPU phases when wide.
3. Structural phases, one kind at a time: aliases; column merges and column transforms;
   substitutions; row combinations. Substitutions are applied by the host aggregator with live
   counts; on large models independent batches are applied on the GPU first.
4. Wide phases when the above stall (less than 0.05% of rows and of columns removed): probing,
   clique pipeline, implication graph. Then back to 1, for a fixed number of alternations at most.

Every phase runs inside its work budget and ends when work passes without a finding (2.1).

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

The stage of each rule is unchanged. The third column says what Stage 0 measured the rule family
to be worth in HiGHS on the 31 instances (report sections 2 and 3; full HiGHS unless the cell
names another run).

| HiGHS rule | Here | Measured in Stage 0 |
|---|---|---|
| Infeasible, redundant, empty, singleton, forcing rows; implied column bounds; fixed and empty columns | Stage 1 | Redundant rows are 37.7% and singleton rows 5.2% of the rows full HiGHS removes; forcing rows give at least 61,832 column fixings |
| Integral scaling, side rounding, Chvatal-Gomory, coefficient strengthening | Stage 1 | Not measured apart |
| Doubleton equations; parallel rows; dual fixing by locks | Stage 1 | Parallel rows 11.2%, doubleton equations 8.7% of the removed rows. The lock rule is not separated yet (decision 1) |
| Probing, implications, clique merging, clique and implication substitutions | Stage 2 | Without probing HiGHS keeps 9.3% more nonzeros in the geometric mean (more than 1% on 9 instances). Clique merging is 34.6% of the removed rows; without it 4.6% more nonzeros (0.7% without sorrell3) |
| Implied equations, dominated / weakly dominated / forcing columns, row-dual bounds | Stage 3 | Implied equations fire on 2 instances, forcing columns on 1. Stage 3 rules execute 2.4% of the row removals. The six single-column rules of this row and the next are the difference between the first two rows of the table below |
| Singleton columns (free substitution, stuffing, zero-cost singleton) | Stage 3 | See the row above |
| Pairwise dominated columns, parallel columns; aggregator | Stage 3 | Pairwise dominated columns take 12.0% of HiGHS's time; 41.9 of their 42.1 s fix nothing. Aggregator not measured apart |
| Simple probing on equations, gcd substitution, implied integers, VLB/VUB extraction, dual-fixing extensions | Stage 3 | In the Stage 1 reference run: 13,874 substitutions by simple probing on equations (brazil3 13,782) and 58,923 dual-fixing substitutions (neos-4306827-ravan 48,761) |
| Sparsify, cover strengthening | Stage 4 | Sparsify is most of the nonzero part of the Stage 3 and 4 rules; HiGHS's inequality strengthening removes 144 nonzeros |
| Fourier-Motzkin, enumeration of small binary rows, VLB/VUB aggregation, dual-fix probing | Left out of the first versions | Fourier-Motzkin fires on 3 instances, and with only that rule off satellites2-60-fs ends at the same size. VLB/VUB aggregation removes 722 rows on one instance. Enumeration takes 7.7% of HiGHS's time |

Share of what full HiGHS removes that a restricted HiGHS reaches, summed over the 31 instances
(measured in Stage 0; the switch sets are defined in section 12):

| Switch set | Rows | Columns | Nonzeros |
|---|---|---|---|
| Stage 1 rule set | 57.9% | 95.2% | 87.5% |
| Stage 1 reference, as run in Stage 0 | 56.3% | 86.8% | 85.7% |
| Stage 1+2 rule set | 97.5% | 97.0% | 97.2% |

- The sums are carried by a few instances: four hold 73% of the nonzeros that full HiGHS removes.
  Per instance the Stage 1 reference reaches at least 90% of the removed nonzeros on 12 of 29
  instances and less than 50% on 9, and every rule family has a median ratio of 1.00.
- The Stage 1 reference is an upper bound for our Stage 1 until the switches of decision 1 exist
  (section 12).
- Stage order: the Stage 2 rules gain more than the Stage 3 and 4 rules in the geometric mean of
  the remaining nonzeros (0.879 against 0.950), but not by the number of instances (11 against 12
  with at least 1% fewer nonzeros). The data do not contradict the stage order and do not prove
  it.

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
  made explicit once, before probing or at the end. HiGHS makes them explicit only when it
  prepares probing, not at the end: of the 29 instances its Stage 1 rule set reduces, 13 keep a
  row with an infinite term (measured in Stage 0).

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
- Sources in one launch do not see each other's results. Re-probing covers only sources whose
  neighbourhood changed and repeats until nothing changes or the probing work budget ends; passes
  below the engine threshold run on the host with a trail. (HiGHS's fourth round on piperout-d27
  makes 5,783 probes in 3.4 s for 22 reductions; measured in Stage 0.)

Budget and stops of probing:
- The work budget counts entries per nonzero, not probes. HiGHS's limits count probes: on
  splice1k1 it makes 1,001 probes of 0.95 passes over the matrix each, finds nothing and needs
  18.6 s (measured in Stage 0).
- A pass stops by yield: it ends when work passes without a finding.
- New cliques and new implications are capped per pass. Reaching a cap does not end probing
  (HiGHS ends probing on its clique limit on two instances; measured in Stage 0).
- A source closed by a cap is distinguished from one closed by a conflict.

Open until the probing run with HiGHS's limits off is in:
- The size of the work budget and its unit: updates along columns, as HiGHS counts, or
  recomputation of touched rows. The yield stop and the caps get their sizes with it.
- What the candidates HiGHS never probes would yield. HiGHS probes 2.0% of the first-round
  candidates (median per instance 14.7%; measured in Stage 0). The surplus that section 1 expects
  from probing depends on this.

Clique pipeline (Dai and Chen 2025, a CPU-thread method, adapted to GPU kernels):
1. Per row: reduce to its binary part, detect set-packing and conflicting knapsack rows.
2. Per knapsack row: segmented sort by coefficient, two binary searches; cliques stored implicitly
   as (row, start) and (vertex, row, start), as cuOpt's table does. (HiGHS's explicit table holds
   564,799 cliques from the 6,444 knapsack rows of splice1k1; measured in Stage 0.)
3. Adjacency without a quadratic matrix: explicit edges by expand, sort, unique for small cliques
   only; larger cliques stay implicit and adjacency is a sorted-list intersection.
4. Extension: one task per clique; candidates are the neighbours of its minimum-degree member that
   are adjacent to all members; greedy grouping in literal order.
5. Dominated cliques are removed from the table by shared member, signature and exact subset test;
   model rows are removed only under the rule of section 4.
6. Triage: longest extensions replace set-packing rows; the other cliques are exported to cuOpt
   through a new setter on its clique table.

Rules for merging:
- Only cliques that changed since the last merge are merged again (7.95 of the 12.03 s HiGHS
  spends in clique merging repeat it without a change; measured in Stage 0).
- The length bins are rebuilt after clique extension (2.3; sorrell3's rows go from 2 entries to a
  median of 37).

Clique merging is decided (decision 2). Its two parts, extension of set-packing rows (steps 4
and 6) and removal of dominated clique rows (step 5), are built in Stage 2 and gated apart from
probing (section 12). Basis, measured in Stage 0: HiGHS's merging changes rows on 14 of 31
instances; without it HiGHS keeps 4.6% more nonzeros in the geometric mean (0.7% without
sorrell3, 0.02% without sorrell3 and savsched1); 98.9% of the rows it deletes are also deleted
with probing off. The decision and the Stage 0 data cover the merging part only. The export to
cuOpt's clique table, clique-aware propagation and knapsack cliques are listed for Stage 2 as in
version 3, which made them depend on Stage 0. Stage 0 has no data on them; whether they are built
is open (section 0).

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
- Scaling (decision 4). Inside cuOpt the adapter receives the model as cuOpt hands it over: after
  cuOpt's scaling when scaling is on, unscaled when it is off. Every cluster run is made with
  cuOpt's scaling on and with it off, so that the effect of scaling on the presolve is seen: the
  arm with the new code runs twice, and its statuses, presolve seconds and reduced sizes are
  compared between the two runs. The baseline is the stored run of cuOpt as is, so the solver
  outcome of the run with scaling off has no baseline with scaling off (section 0, open). The
  standalone gates use unscaled models on both sides (section 12). The harness keeps the switch
  that applies cuOpt's scaling, to reproduce a cluster case; no gate uses it.
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
| Dev node, standalone against HiGHS, unscaled models on both sides | All 40 testbed instances: size, seconds, row dynamism, determinism hashes, the test oracle, MIPLIB solutions crushed in primal-only mode and postsolved back. The 9 instances above 1e7 nonzeros run in the background with a time limit |
| Cluster, inside cuOpt | Regular benchmark set, one run per arm, stored baseline reused; the arm with the new code with cuOpt's scaling on and with it off: statuses, postsolved solutions feasible in the original model, presolve seconds, reduced size and root LP bound parsed from the logs, primal gap against the best known solution, primal integral |

Cluster runs may be launched without asking when a larger test is needed (section 0). No solver
runs on the dev node.

Conventions:

- Both sides get the unscaled model (decision 4). The two reduced models are compared in one
  convention: minimisation, same equilibration, implied integers counted separately.
- The sizes of a HiGHS reference are taken from the DIAG total line of the patched HiGHS. HiGHS
  returns the original model with "Not reduced" when only clique merging changed it, and its
  written presolved model is power-of-two scaled.
- Sizes are compared as rows, columns and nonzeros together: a restricted rule set can leave fewer
  nonzeros than HiGHS with more rows (brazil3, unitcal_7). Nonzeros added by clique extension and
  aggregator fill-in are reported apart from the size measure; implied integers and row dynamism
  are reported.
- Row dynamism is a constraint, not a goal: 99th percentile at most 10 times HiGHS's.

HiGHS references (patched HiGHS, one switch set each):

| Reference | Switch set | Status |
|---|---|---|
| Stage 1 rule set | Off: probing, enumeration, clique substitutions, clique merging, the implied-integer sweep, aggregator, sparsify, Fourier-Motzkin, inequality strengthening, pairwise dominated columns, parallel columns | Run on 31 instances |
| Stage 1 reference, as run in Stage 0 | The Stage 1 rule set, and off: six column rules (dominated column, forcing column, free column substitution, zero-cost singleton, column stuffing, weakly dominated column) | Run on 31 instances. An upper bound for our Stage 1: it keeps at least four Stage 3 rule families and 801,077 column fixings whose rule is not recorded |
| Stage 1 reference of decision 1 | The same, and off: simple probing on equations, the dual-fixing extensions apart from the lock rule, implied equations, implied-integer conversion. A counter per rule for column fixings. A second run with HiGHS's own dual-fixing rule off as well, which also removes the lock rule | Not built; due before the Stage 1 gate (section 13). The gate uses the first run; the second run is the lower end of the bracket for dual fixing and serves to explain instances (a reading of decision 1; section 0, open) |
| Stage 1+2 rule set | The Stage 1 rule set with probing, clique substitutions and clique merging on | Run on 31 instances |
| Stage 1+2 reference with enumeration on | The Stage 1+2 rule set with enumeration on | Not run; reference of the clique gate |
| References for the Stage 2 and Stage 3 gates | Five more switch sets (report section 9) | Not built; due before each gate |

Gates:

| Gate | Rule | Status |
|---|---|---|
| Time per instance, bands A and B | t <= 1.2 * t(HiGHS) + 0.02 s, both sides the minimum of at least three alternating runs | Decision 3 |
| Time per band | Shifted geomean (shift 0.1 s) and median as in section 1 | Proposed in report section 7, entered by change 10 |
| Reductions, Stage 1 | Geometric mean of nnz(Stage 1 reference) / nnz(ours) at least 0.99, taken over all testbed instances that have the reference, not per band; every instance below 0.95 explained. Rows and columns are reported next to it | Decision 3. The reference exists for 31 instances today, for 40 when it is run above 1e7 nonzeros. That "all instances" means these is a reading (section 0, open) |
| Probing, Stage 2 | The model after the whole presolve with and without the new probing is compared (rows, columns, nonzeros, bounds), not the number of fixings: rd-rplusc-21 gets 17 fixings and substitutions from probing, and without probing its presolved model has 2.3 times the nonzeros (measured in Stage 0) | Proposed in report section 7, entered by change 10. No threshold is set; the HiGHS reference is one of the switch sets not built yet |
| Cliques, Stage 2, apart from probing | On the instances where HiGHS's merging leaves at least 1% fewer rows (8 of 31, measured in Stage 0): geometric mean of rows(reference) / rows(ours) and of nnz(reference) / nnz(ours) at least 0.99 each, extension nonzeros apart. Reference: the Stage 1+2 reference with enumeration on | Decision 2. The report says "at least 0.99 of the reference"; the ratio is read as in the Stage 1 gate |
| Status instances | neos-3230511-yuna (band D): unbounded or infeasible, time reported only. neos-2075418-temuka (B3): HiGHS returns the model unreduced after 15.7 s; detecting infeasibility is better, returning a model within 18.8 s equals it. bnatt500 and fhnw-binpack4-4 are below 50,000 nonzeros: no time rule | Proposed in report section 7, entered by change 10. The rule by which HiGHS detects yuna is unknown; if it is a Stage 3 argument, the Stage 1 check needs an exception |

## 13. Stages

Sizes are rough engineer-weeks for a person and show relative scale only.

| Stage | Content | Gate and cluster run |
|---|---|---|
| 0. Baseline and cost model. Done for bands A and B (2026-11-26) | Delivered (report sections 2 to 7 and 10): HiGHS seconds per stage and band, reductions per rule family, probing and clique statistics, instance structure, launch, read-back and hand-over costs, kernel variants, rebuild. Still running: HiGHS on the nine instances above 1e7 nonzeros; probing with HiGHS's limits off. Not delivered: first list below | Delivered: target table per instance for bands A and B (provisional until the minimum of three alternating runs exists), time-target constants, fixed cost per band, provisional thresholds. Fixed default thresholds moved to Stage 1 |
| 1. Core (10 to 12). Started | 1a, host engine first: host engine for the Stage 1 rules; activities of 2.4; alias forest; parallel rows; postsolve; statuses; primal-only switch; test oracle; determinism hashes. 1b, then the device propagation: memory resource; binned matrix, built after the first host cascade; GPU propagation from the lifted kernels with lane groups (2.3); frontier kernel in lane-group order (decision 5); rebuild; the measurements of the third list below, then the fixed thresholds. Last: cuOpt adapter | 1a target, not a gate: the host cascade is not slower than HiGHS's initial scan plus fast loop in the Stage 1 reference run (21 to 102 ms in bands A1 and A2, 37 to 163 ms in A3, measured in Stage 0). Stage gate, applied after the reference work of the second list below: the reductions gate and the time gates of section 12 against the Stage 1 reference of decision 1. Cluster run 1 (new presolve in place of PaPILO, cuOpt's scaling on and off): no wrong status, every postsolved solution feasible; solver outcome is expected below PaPILO at this stage |
| 2. GPU-wide (10 to 15) | Many-source probing with a work budget in entries, yield stop and caps; implication graph; clique pipeline with clique merging (decision 2); clique-aware propagation; replacements for cuOpt's probing cache and clique table | Probing gate and clique gate of section 12, each apart from the other; seconds against HiGHS probing on bands A and B. Cluster run 2: cuOpt as is against cuOpt with the new probing and cliques, scaling on and off |
| 3. Parity (15 to 20) | Reservation table; dual reductions; singleton, dominated and parallel columns, pairwise dominated columns under a hard work cap; aggregator; implied integers; gcd transforms; simple probing on equations; VLB/VUB extraction; dual-fixing extensions. The single-row and single-column rules that the Stage 1 reference switches off stay here (decision 1) | Remaining nonzeros at parity with HiGHS patched to the rule set of Stages 1 to 3. Cluster run 3: new presolve against PaPILO, first full comparison, scaling on and off; HiGHS-presolved arm |
| 4. Completion | Sparsify, cover strengthening; tuning; the held-out set; Grace Hopper measurements if a node exists | Section 12 on the held-out set |
| 5. Productization | Default path, custom presolvers, sub-MIPs, limits, documentation | Cluster A/B as the acceptance test, with a threshold set before the run |

Not delivered by Stage 0, and where it is due now (report section 9):

| Item | Due |
|---|---|
| Full HiGHS (stock and patched) on the nine instances above 1e7 nonzeros | Running |
| Probing with HiGHS's limits off: yield of the candidates HiGHS never probes | Running; read before Stage 2 starts |
| Probing work budget: size and unit; sizes of the yield stop and of the caps | Before Stage 2 starts |
| Byte budget | Before Stage 2 starts, with the overlay design |
| Three counters for the probing design (bound changes, distinct rows touched and their summed length per probe) | Before Stage 2 starts |
| Dead-entry fraction that triggers a rebuild | Stage 1b |
| Device fixed-cost budget per presolve call (proposed: 5 work units, 50 ns, per nonzero) | Stage 1b |
| Hand-back threshold: consecutive small rounds after a hand-back | Stage 1b |
| Share of the nonzeros above which all rows are recomputed | Stage 1b |
| Work budget of each phase and the amount of work without a finding that ends it (change 4 gives no sizes) | With the stage that builds the phase; for the cascade and the other Stage 1 phases in Stage 1 |
| HiGHS's count of pair tests, for the cap of pairwise dominated columns | Before Stage 3 starts |
| Surplus sources: the aggregator pass that ends after three fill-in rejections (92 calls on 15 instances), stages that HiGHS runs once | Before Stage 3 starts |
| Five more switch sets as references; the Stage 1+2 reference with enumeration on | Before the Stage 2 and Stage 3 gates |

Reference work before the Stage 1 gate is applied (decision 1 and report section 9):

1. Patch switches for simple probing on equations, the dual-fixing extensions apart from the lock
   rule, implied equations and implied-integer conversion.
2. A counter per calling rule for column fixings. It settles whether three quarters of the column
   removal belongs to Stage 1 or to Stage 3 rules.
3. A switch that tests row redundancy on explicit-bound sums (2.4).
4. The Stage 1 reference run again with these switches, and one more run with HiGHS's own
   dual-fixing rule off as well. The two runs bracket dual fixing.
5. The reference three times, alternating with our runs, for the time rule; and the reference on
   the nine instances above 1e7 nonzeros.

Stage 1b measures again, with the real kernels, before the thresholds of 2.1 are fixed:

- the frontier kernel in lane-group order (decision 5), a frontier-restricted candidate kernel and
  the whole GPU step;
- the host engine with incremental activities on real frontiers; frontiers between 10% and 100% of
  the rows; consecutive small rounds per cascade;
- full sweep and rebuild on matrices of 20,000 to 70,000 nonzeros (models shrink during presolve:
  ns1830653 goes from 100,933 to 24,263 nonzeros);
- the dirty list in a fixed order on a model-sized state;
- the binned layout in place, re-binning, other bin boundaries than 64 and 8,192; the error guard;
- row scheduling above 1e7 nonzeros and column scheduling from 1e6 (2.3);
- a second GPU type.

Test oracle: tiny random MIPs solved by enumeration, including infeasible, unbounded and empty
outcomes, scaled coefficients, huge bounds and free columns, multi-round sequences, and every size
threshold as a test parameter so each path, including both engines and both memory paths, runs on
tiny models. The testbed has free columns on 6 of 31 instances only and no finite bound of 1e8 or
more (measured in Stage 0), so the oracle is the test for both.
Checks: reduced optimum equals the original; every reduced-feasible point postsolves to an
original-feasible point; exact crush in primal-only mode; exported implications and cliques valid;
a GPU batch equals its sequential replay; a row recomputed by the frontier kernel has the same
bits as by the full sweep (decision 5); on integer-only models run to an empty frontier both
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
