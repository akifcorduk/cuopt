# Plan for the complete GPU acceleration of Stage 1 (proposal)

Status on 2026-11-28: proposed, not approved, nothing implemented. It answers the owner's
directive of 2026-11-28 (DESIGN.md amendment A11, HISTORY Entry 51). It is the synthesis of a
design round with ten agents: an inventory of the code, a measurement of how much work is
available at once, a research step, three competing designs, three critiques, one synthesis.
An independent check of this plan ran on the same day (HISTORY Entry 55): six verifiers and
one step that tried to refute each finding. Verdict: the plan stands with 39 amendments, which
are appended at the end of this file and override the text above where they differ. In
particular section 14 (decisions for the owner) is replaced by amendment AM-39 and milestone
M0 by AM-33. The reports of the check are in `verification.json`.

Files beside this one: `inventory`, `measured`, `research`, `design_one_algorithm`,
`design_bulk_first`, `design_device_native`, `critique_validity_determinism`,
`critique_performance`, `critique_completeness_migration`, `final` (each as `.json` and as
readable `.txt`). `final` holds this text (field `design`) and the full fields `per_part`,
`milestones`, `acceptance`, `order_with_remaining_rules`, `reductions_lost`, `defects_taken`,
`defects_open`, `rejected`, `assumptions`, `open_for_owner`, `unsure`.

The text below is the synthesis agent's, unchanged.

```text
PLAN FOR THE COMPLETE GPU ACCELERATION OF STAGE 1
Synthesis of the design round of 2026-11-28 (inventory, measurement, research, three designs, three critiques). Nothing is implemented; the plan is for the owner's decision.

Path prefixes. G = /Users/acoerduek/Documents/workspace/cuopt-wt/gpu-presolve/gpu_presolve. P = /Users/acoerduek/Documents/workspace/cuopt-wt/gpu-presolve/design_summaries/gpu_mip_presolve. R = /private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/gpu_s1_design (work directories of this round).
Instance sets. 'The 240' are the 240 MIPLIB benchmark instances; 'the 31' and 'the nine' are the testbed below and above 1e7 nonzeros, together 'the 40'; 'the 30' are the 30 of the 240 with at least 1e6 nonzeros. 'Node seconds' are the stored runs on the EPYC 7413 node, 'Mac seconds' those of the measurement step, 'model' is the performance critic's model (no device step of any design has been run).

0. WHAT THIS PLAN RESTS ON

Frame. The design 'one algorithm, two executors' is the frame: every rule family is a sequence of batch steps, and a host executor and a device executor compute each step with the same bits. It ranks first under the completeness lens and second under the other two, it is the only design with measured reductions, and its defects all have small repairs. Three things are changed. (a) The way there: not a rewrite in two large milestones, but changes inside today's engine, each behind a switch and each gated (completeness critique). (b) The rules of a step: the conflict and floating-point rules of 'device-native' with the repairs of the validity critique. (c) The hybrid rule: the hand-back rule of 'device-native', with the cost of a host copy counted (performance critique). 'Bulk-first' is not the plan; section 12 says why and what is kept of it.

Read by me in this step: the three designs and the three critiques in full; P/DESIGN.md lines 1 to 260 (amendments A1 to A11, decisions) and 324 to 573 (sections 2 to 4); P/ASSUMPTIONS.md entries AS-254 and AS-304 to AS-310 and the list of headings (the last entry is AS-338); the last entry of P/HISTORY.md (the campaign of 5.4 billion models on today's engine with 18 oracle failures, one of them a lost optimum caused by a bound of -3e19, under triage); G/include/gprs/numerics.hpp 140-163, G/src/state_build.cpp 170-226, G/include/gprs/presolve.hpp 80-180, G/src/rules_parallel_rows.cpp 360-372, 405-435, 470-495 and 786-830, G/src/rules_doubleton.cpp 100-160; R/critique_completeness-migration/probe_results.txt and hybrid_sim.txt; R/design_one-algorithm/analyse_240_final.txt.
Ran by me (three scripts over stored JSON and two greps; no build, no engine run, nothing changed in the project tree): (1) the 240 result files of the validity critic's repaired batch run (R/critique_validity-determinism/runs_fix7) against the designer's batch run (R/design_one-algorithm/runs/m7) and the base run (m0): hash and status equal on 240 of 240 against the batch run and on 234 against the base. (2) The history counters of the singleton-column family in the stored node runs: saved_row_refused is 0 on all 240, all 31 and all nine; growth_refused is 2,876 on dlr2 and 0 elsewhere. (3) Size classes of the 240 by input nonzeros and by state bytes (40 per nonzero, 100 per row, 90 per column): 93 below 5e4, 21 from 5e4 to 1e5, 29 from 1e5 to 2e5, 67 from 2e5 to 1e6, 28 from 1e6 to 1e7, 2 above; of the 126 instances from 1e5 nonzeros, 42 have a state of at most 16 MiB (largest 384,329 nonzeros, 1.21 s of today's 19.47 node seconds) and 114 of at most 128 MiB. (4) A grep for every place where the engine adds a matrix entry: move_entry at rules_doubleton.cpp:130 and rules_parallel_rows.cpp:483, both of which reuse a slot of the same row.
Not done: no device run, no node, nothing on the nine; I did not read the Stage 1b and Stage 1c design texts, the device code, the cuOpt kernels, HiGHS or any paper (what the plan says about them comes from the first-step reports and is marked); I did not rerun the critics' models and probes; the repairs I designed here (section 2, items 4 and 5 of the state; the parts in section 3 that say 'designed in this step') are not prototyped.
Three corrections of the first step that the plan uses: cuOpt's load-balanced code is under cpp/src/presolve in this worktree; the first propagation call holds 74.0% of the propagation work on the 240, not 92.5%; pointer jumping over the alias forest is not in DESIGN.md 4.1.

1. WHAT THE OWNER ASKED AND HOW IT IS READ

Directive of 2026-11-28, verbatim: 'I want the whole presolve suite to be built on GPU. It can be hybrid CPU/GPU presolve where smaller and non-parallel work to CPU. Otherwise, all methods that can be parallelized should be on GPU. Do not switch to state 2 without implementing a complete GPU acceleration for stage 1.'

'Stage 1' is read widely: every part of a presolve call before the wide phases (probing, cliques, implication graph). That is the state build, the start step, the row rules, bound propagation with the switch to explicit bounds, row tightening, dual fixing by locks, parallel rows with row addition, the doubleton rule with the alias forest, the singleton-column family, simple probing on equations, the nine designed column rules that are not built, the driver, the record stack, the storage steps, the output and its hash. The narrow reading (the rule set that is built) is a subset and ends at milestone M10; the wide one ends at M12.

'Methods that can be parallelized' is read as a property of the rule, not of today's order of application. A rule is parallel when its tests are a map over rows or columns on one state and its applications can be combined by a rule whose result does not depend on thread timing. Measured: 68.8% of 7.2 million findings need nothing of the same call, and dual fixing, parallel rows and the doubleton rule have chain length 1 in every call (240, measurement step); row rules and propagation run as rounds from one state with no instance larger (240, prototype). So the applications are parallel work and go to the device. Under the other reading (today's order is part of the method) the device can only test and the host applies; that is 'bulk-first': the same bits as today, about 1.9 times faster on the 240 and 1.7 times on the nine in the model. The plan does not take that reading. Because the directive can be read both ways, this is the first question to the owner, with both outcomes named.

'Smaller and non-parallel work to CPU' is a closed list: a model below a size threshold; steps below a work threshold where a host copy of the state exists; the driver; the postsolve undo; an alias forest of few edges and tree levels beyond a cap; a sequential test inside one row above a length cap. Each needs a figure measured on the finished engine.

'Complete' is defined by the acceptance criteria of section 9: every step has a device form, no step is on the host because its device form is missing, the size threshold is bounded, host and device agree in every bit, and the remaining host time is bounded in seconds.

2. ARCHITECTURE IN ONE PAGE

One engine, one state type. The library gprs and state_t stay. No second engine and no second state type is built.

A family call is a loop of steps. A step has four parts. FIND: a map over the listed rows or columns that reads the state as the step found it and writes candidates. SELECT: one of three rules decides which candidates are applied and how writes to one cell combine. APPLY: the writes. UPDATE: rows that lost or renamed an entry are rewritten in place, sizes, free sizes, singleton counts and locks of the touched lines are brought current by integer sums, rows whose activity can have changed are marked and folded again before their next reader, the list masks of the consumers are set, and one status record is produced (findings, records, work, the size and the entries of every next list, the outcome with the smallest index that caused it). Nothing is applied while another item of the same step is tested.

The three rules of SELECT.
- Combine (K1), for writes that hold for every feasible point and for integers. A bound is reduced over pairs (value, source index): the better value wins; among values that compare equal as doubles (so +0 and -0) the smaller source index wins; the bits of the winner are stored. That is the result of a loop in ascending source index with a strict comparison, and it is one result in bits on any tree. Integer rounding comes after the reduction. A lower bound above the upper by more than the tolerance is infeasible; a smaller crossing gives the column the upper candidate. A row that gave a bound is deleted in the step only when the bounds written to its columns are at least as tight as its own candidates (a forcing row: when every free column was written at its value); otherwise it stays, is listed again and is tested in the next step in its own units. Counters change by integer sums or integer atomics.
- Ordered fold (K2), for terms that several units add into one cell (the coefficient of a root, the side shift of a row, the cost of a column). The terms are brought together by a stable radix sort on the unique integer key (cell, source index), each run is added in ascending source index as a compensated sum, and tests such as the cancellation test are made once on the folded sum. The objective offset has no running sum: each column owns its contribution and the offset is formed once at the output by a tree fixed by the number of columns.
- Reservation (K3), for a unit that needs a cell alone. The unit writes its key into the cell with an integer atomic minimum; keys are unique over all units of the step. A unit is applied when it holds every cell it write-claims and no unit with a smaller key write-claims a cell it reads. One pass. A loser leaves no trace and stays on its list. The applied set is what a loop in ascending key gives when every premise is tested at application.
No floating-point atomic is used anywhere.

The state holds no history.
1. Activities are values: the fold of numerics.hpp (entry k of the row in storage order to lane k mod lanes, compensated lanes, the fixed merge tree) over the bounds of the moment, never updated. The counts of infinite terms are two uint16 arrays that saturate (only 0, 1 and 'more' are read); the position of a single infinite entry is not stored, the entry recognises its own term.
2. Work lists are sets, served in ascending index.
3. The layout is not state. A rebuild is decided on counters (entries deleted since the last rebuild over the entries at that rebuild), not on slots.
4. A live row has no dead entry and holds its entries in ascending column index. A step that deletes or renames an entry of a row rewrites that row in place in the same step. So the lane of an entry is its slot position, on both executors, after every step. This is possible because no rule of Stage 1 lets a row grow (the two places that add an entry reuse a slot of the same row). Columns are in ascending row index and may hold dead entries until they are rebuilt; only order-free reductions run over a column.
5. A new entry of a column (it arises in two rules: a root takes over an entry, and the singleton column of a subtracted equation enters the target row) is merged into its column at the end of the step that made it. Nothing waits for a rebuild.
6. Records are ordered by step, then by a fixed order of record kinds, then by unit index; pool offsets are prefix sums in that order.
7. Budgets are tested between steps; a step is done in full or not at all.

Two executors. The host executor is plain C++17 in the host library and is the whole engine in a build without CUDA. The device executor is CUDA code in G/device. The arithmetic of every rule lives in shared headers of value functions (numbers in, numbers out) that both compile, with contraction off and the one explicit fused multiply-add of two_prod. Both hold the same arrays with the same slots, so a hand-over is a copy and 'device equals host' is tested array by array.

The driver is host code. It runs today's sequence (start step; rounds of cascade, parallel rows, doubleton, simple probing and the column families; the switch to explicit bounds; absorption of fixed terms; output; hash) whichever executor works, and chooses the executor of each step from work counts (section 4). A cascade pass stays four family calls, each a loop of steps until its list is empty; the fused wave of 'device-native' is not taken now (section 12).

Today's order stays selectable on the host. With the rules split into evaluation and application, the option apply_order = sequential runs 'evaluate, then apply at once' per unit in ascending index, and apply_order = batch runs the step. The device runs the batch order only; identity of host and device is claimed for it.

Where the state lives. On the device from the build to the output when the device is used. The host holds the input model, the options, the status records, the record stack after each flush and the reduced model. A host copy of the state exists only in the hand-back regime for small states and under the transitional rule during the build-up.

3. THE PARTS (where each runs, the device algorithm, the conflict rule, the measured reason for what stays on the host; the full entries are in the field per_part)

3.1 State build. Both: device from device_min_nnz (provisional 100,000 input nonzeros). Upload from pageable memory; column histogram and scan; transpose with twins by the stable radix sort that exists (G/device/src/state_build.cu, bit-equal in 3,734 builds); locks; row singleton counts; both activities by fold kernels in the host's lane order with the counts of infinite terms taken in the scan; masks; scheduling lists by length class. No conflict. Host below the threshold: a whole presolve takes the host 1.7 ms in the median below 5e4 nonzeros (93 of the 240) and 15.3 ms from 5e4 to 2e5 (50 instances). 31.4% of the node seconds on the 240, 34.2% on the nine.
3.2 Start step. With the executor of the state build: two maps, fixed columns recorded by a compaction in ascending column. No conflict. 0.2% of the seconds (240).
3.3 Row rules (empty, infeasible, redundant, singleton and forcing row). Both. FIND: class per listed row in O(1) from the stored activities, a scan only for singleton rows and accepted forcing rows. Bounds of singleton rows and fixings of forcing rows by the combine rule with the crossing rule; deletions by flag with integer lock and size updates. Host for the tail: lists below 100 rows are 1,458 of 2,263 lists with 0.1% of the work (240).
3.4 Bound propagation and the switch to explicit bounds. Both. One Jacobi round per step: candidates per entry from the stored activity (residual first, error guard, the single infinite term), reduced per column by the combine rule in a column-driven or a row-driven form chosen by counts; the best candidate of the round becomes the bound or the implied record. The switch to explicit bounds is a map over columns. Host for the tail: lists below 1,000 rows are 1,445 of 2,122 lists with 8.7% of the propagation work (240).
3.5 Row tightening. Both. A map over rows; a row edits only itself; the scale is the least common multiple of per-entry denominators. No conflict. Host for the tail: 688 of 1,278 lists have fewer than 100 rows and 1.4% of the work (240).
3.6 Dual fixing by locks. Both. A map over columns; fixings together; records by compaction. No conflict (one batch in all 457 calls with a finding, 240).
3.7 Parallel rows with row addition. Both. Keys as an integer sum per row, stable radix sort, up to parallel_bucket_limit rounds of exact comparison with a bucket leader, a merge step (sides by the combine rule), then on the merged state a subtraction step (an equation with a singleton column serves one target per step by reservation; targets rewrite themselves). Host for later calls: the first call holds 91.5% of the work (240) and 95.6% (nine).
3.8 Doubleton rule and the alias pipeline (shared with simple probing and the single-lock alias). Both. Edges by a map; the forest is the minimum spanning forest under a unique key (union-find on the host, Boruvka rounds on the device); maps composed level by level from the root in today's association order; all trees substituted in one step by the ordered fold; touched rows rewritten in column order; new root entries merged into their columns. No row is reserved (under one tree per row only 32.0% of 280,666 trees fit a first batch, 240). Host: the forest of a small batch (tree walk and composition are 0.018 s of this family's 0.317 Mac seconds on the 240, about 37 ns per alias, against about a millisecond of launches) and levels beyond 64 (0.3% of the substituted columns; depth 741 on CMS750_4).
3.9 Singleton-column family. Both, in two steps per pass: row units (a row unit writes only its own row; costs of shared columns by the ordered fold), then forcing columns on the state after the row units (reservation on rows, key = column index). Host for later passes: the first pass of the first call holds 80.3% of the work and 97.5% of the findings (240).
3.10 Simple probing on equations (built at edd147a7e, review to be finished first). Both. Collection is a map on an unchanged state; all edges go to the alias pipeline; a row is deleted when all its edges became maps. Host for small lists: 277,320 aliases on 8 of the 240 in the one stored Mac run.
3.11 The nine designed rules (dominance between parallel columns and their merge, dual bound strengthening, single-lock alias, implied equation, implied integer with scaling, column stuffing, the single-equation extension of dual fixing, the dominated column). Both; each is written once as evaluation plus application, run on the host first in the design's index order and then as a step, on the kernels of the families it resembles. The conflict rule of each is a proposal in per_part and is fixed only after a table of the cells the rule reads and writes has been checked against its Stage 1c text and the step form has passed a validity review. Host parts: short lists (not measured, the rules do not exist) and the prefix test of column stuffing inside one row.
3.12 Driver. Host. One status record per device step through one read-back (13.8 microseconds on one H200). It is a few comparisons per step: 707 rounds, about 6,900 cascade steps and 1,367 calls of the three structural families on the 240 in all.
3.13 Storage. With the executor that owns the state: rows rewritten in the step; column insert merge at the end of a step (the whole column store by the radix sort when the batch is a large share of the nonzeros); rebuild on counters (row store compacted, fixed terms folded into the sides, column store by the radix sort at 467 microseconds + 0.279 ns per nonzero).
3.14 Record stack and postsolve. Records are produced by the executor of the step, in device arrays with prefix-sum offsets, flushed at a hand-over, at a byte limit and at the end. Undo and crush stay on the host: the undo is a chain by definition and runs outside the presolve call.
3.15 Output, with the removal of empty rows and the comparison with the input for the status 'unchanged'. Both, as a step under the hybrid rule: two scans, a gather with the new column index (no sort), sides less the folded fixed terms, the offset by its tree; delivered as device arrays to the cuOpt adapter or downloaded.
3.16 Hash. Both, or off: per array an integer sum of mixed (position, word), the array sums chained on the host. The chain hash of today is serial by definition and stays as an option.

4. THE HYBRID RULE

The driver chooses the executor of every step from numbers that are functions of the model and the settings. Nothing is read from the machine. Because both executors give the same bits, no value below can change a reduced model, a record or a work count; that claim is itself tested by forcing each value low and high and by drawing them at random. Work of a step: W = sum over the lines of its list of (8 + live entries), today's unit, known before the step from the previous status record.

H0. Build without CUDA, or executor = host: everything on the host.
H1. Small models. Input nonzeros below device_min_nnz (provisional 100,000): the host does the whole call and nothing is uploaded. Measured reason: the median host time of a whole presolve is 1.7 ms below 5e4 nonzeros (93 of the 240) and 15.3 ms from 5e4 to 2e5 (50); 72.8% of all family calls take the host less than one fixed device step of 85 microseconds, and at one step each they would cost 63.7% of the engine seconds below 5e4 nonzeros (240, Mac seconds against an H200 figure). The requirement 'better than HiGHS from 50,000 nonzeros' is carried by the host path there. The value is fixed from a stored crossover measurement (device forced against host, per part and size class) and may not exceed 200,000 without the owner's consent.
H2. From device_min_nnz the device builds the state and owns it until the output. Every step runs on the device unless H3 or H5 says otherwise.
H3. Hand-back, only while the state is at most mirror_max_bytes (provisional 16 MiB: 42 of the 126 instances of the 240 from 1e5 nonzeros). When the next step is below T_down of its family in two consecutive steps, the state goes to the host executor; it returns for a step of at least T_up. Output plus hash is a step with W = live nonzeros. Provisional values, the ones the critics simulated: T_down 20,000 units for the cascade families and 200,000 for the structural ones, T_up 200,000 for every family. Measured reason for a hand-back: calls below 20,000 units are 64.9% of the calls and 0.6% of the engine seconds (240), and unitcal_7 (127,595 nonzeros, 1,398 family calls in 33 ms) would take 6.9 times its engine time at one device step per call.
H4. Above mirror_max_bytes there is no hand-back; small steps run on the device at their fixed cost. Reason: on the 30 the calls below one fixed step cost 0.3% of the engine seconds at one step each (measured host seconds, H200 step cost); and a host copy is not free: created inside the call it costs 1.3 to 1.4 s on the 106 instances of the 240 that hand over under the rule of 'one algorithm' (first touch 2.7 GB/s, pinning 0.38 s per GB), and with it counted the class from 2e5 to 1e6 nonzeros is 2.0 times faster than today with a hand-over and 4.0 times with every step on the device (model; 2.3 times with fixed costs doubled). Only between 1e5 and 2e5 nonzeros the hand-over wins in the model (1.6 against 1.0 times).
H5. Small sequential objects go to the host at any size, never the state: an alias forest below forest_device_min_edges (provisional 100,000; the edge list goes down, the maps come up), tree levels beyond forest_level_cap (provisional 64), and a sequential test inside one row above its length cap.
H6. Form of a candidate step: column-driven or row-driven, by an integer comparison of the entries of the touched columns with the entries of the listed rows (the status record carries both). Reason: the touched columns hold as many entries as the whole matrix from a median of 4,853, 13,211 and 40,528 frontier entries in the classes 5e4 to 2e5, 2e5 to 1e6 and from 1e6 (147 input matrices, performance critic).
H7. Memory: when plan_memory(rows, columns, nonzeros, options) exceeds the byte budget, the whole call runs on the host. After a failed device allocation the call runs on the host with a logged warning (a switch makes it an error).
H8. Transitional: a step that has no device form yet runs on the host through a whole-state hand-over at any size, with its seconds reported. At acceptance no step uses H8.

All values are options with one shipped default each, fixed once at M12 by the rule of AS-307 on the 240 and the 40 (smallest summed time, ties to the host, no instance slower than the host path by more than the larger of 5% and 2 ms), with three settings of H3 measured: no hand-back, hand-back with the host copy created in the call, hand-back with a warm pool. The per-family break-even differs by a factor of nine (host cost per work unit on the 240, node: propagation 9.7 ns, doubleton 7.2, row rules and tightening 5.1, parallel rows 3.4, singleton columns 2.1, dual fixing 1.1), so T_down is set per family. Fixed costs behind the rule (Stage 0, one H200): launch with synchronisation 9.0 microseconds, with an 8-byte read-back 13.8, reduce, scan or compaction up to 1e5 elements 23 to 46, sort of 1e3 to 1e5 elements 94 to 501, ordered hand-back at most about 141, pageable copy about 10 GB/s. The designers put a cascade round at 100 to 200 microseconds and the performance critic at 190 to 250; structural steps at 1 to 3.5 ms. None is measured; M0 measures mock steps.
What the rule gives on recorded host steps (completeness critic's simulation, 114 instances of the 240 from 1e5 nonzeros with a state up to 128 MiB, hand-back after two low steps and return at 200,000): 98.7% of the rule work on the device, 2.2 switches per instance; the rule of 'one algorithm' as written gave 84.4%.

5. THE IDENTITY RULE, AND WHAT IS LOST OR GAINED IN REDUCTIONS

With and without a GPU: equal in every bit. For the same model and settings the reduced model, the record stack, every statistic counted in work units or findings, the status and the hash of the state after every step are the same for a build without CUDA, the host executor forced, the device executor forced, the hybrid rule with any thresholds, either candidate form, the pinned path on x86 and the managed path on Grace Hopper, any GPU type, any run and any co-running load. Reasons: one function of values per rule; one fold order for every floating-point sum; reductions under a total order; sorts on unique integer keys; integer atomics only; no library reduction of a floating-point sum (CCCL guarantees the same sum across GPU types only for the plain reduction, not the segmented one; read by the research step). What differs by design: seconds and the device counters. I agree with the preference for the same bits: it lets the oracle, the structure and crush checks, the wrong engines and the campaign test the device path through the host path, and it makes every threshold a matter of time only.

Against today's engine (commit 04692babd, after the triage of the 18 oracle failures): with every switch at its old position the engine is today's, bit for bit, for as long as the old forms are kept. With the new positions it is not equal in bits; the gate is on reductions and on strength. Measured so far, all on the 240 benchmark dumps on the host: activities as values 0 of 238 hashes differ (square41 and square47 not finished); least-common-multiple scale 0 of 8,842,639 row searches and 0 of 240 hashes differ; ascending lists 2 of 240 hashes differ; batch rounds for row rules and propagation with ascending lists and budgets between rounds 6 of 240 hashes differ, no instance larger in rows, columns or nonzeros, momentum1 65 rows smaller and neos-1122047 2 rows smaller, rounds +5.7% (row rules) and +0.9% (propagation), work +2.8%; the crossing rule on top of it 0 of 240 differ. Not measured: parallel rows by leaders, the alias batch with ordered sums and sorted rows, the singleton-column family in two steps, simple probing through the forest, the nine and 14 of the 31. The field reductions_lost lists where a loss is possible.

Options and statistics that keep their name and change their meaning: the budget stop (between two lines today, between two steps then; work, first_work, calls, max_cascade_passes and max_rounds keep their meaning); parallel_bucket_limit (unlike neighbours per row today, leader rounds per bucket then); rebuild_dead_fraction (share of dead slots today, deleted entries since the last rebuild then); phase_hashes (extended to steps); the result hash (hash_kind). New statistics: steps per family, steps on the device, hand-overs, bytes moved, launches, read-backs, seconds of upload, download, hand-over and landing.

6. THE DEVICE STATE

Matrix. The line stores of state_t on both sides: per line start (int64), used, capacity, size (int32); per slot index (int32, -1 for a dead entry), value (double), twin (int64). Storage is in index order; two scheduling lists (rows and columns by length class: six classes up to 64 entries as lane groups of a warp, one class up to 8,192 as one block of 256 threads, longer lines with the same block for the fold and chunks for order-free reductions), made by a stable partition after a build, a rebuild or a column merge, give the kernels their classes. Physically binned storage is a benchmarked variant that cannot change a result. Rows: sides, alive, free size, singleton count, version, mask, and per side of the activity hi, lo (its own array), largest term, uint16 count of infinite terms. Columns: bounds, cost, type, status, locks, implied-bound records, mask, family data, the column's contribution to the offset. About 40 bytes per nonzero, 100 per row and 90 per column persistent; sort scratch up to 32 bytes per nonzero for a full rebuild (2.57e8 nonzeros: 10.3 GB plus 8.2 GB). Sizes are planned from the model before the first allocation.
Deletions. A deleted row is a flag; its entries in the columns are dead without a write and are skipped by a look-up of the row flag. A row that loses or renames an entry is rewritten in place by its own thread group in the same step, with the twins of the moved entries corrected. Lock and size changes are integer atomics over the entries of what is deleted, so a deletion costs what it deletes.
Work lists are the masks; a step compacts its mask into an index buffer or tests the flag in a kernel over all lines (the same result). Record stack: device arrays per kind, the order array and the pool of saved entries, flushed to the host at a hand-over, at a byte limit and at the end (dlr2 saves 71 million entries, 0.85 GB).
Memory paths. x86: device memory; in the hand-back regime the host state takes its arrays from a pool of the presolve's memory resource, and a hand-over is a whole copy first and ordered dirty records later only if the measurement at M7 shows that the regime pays. Grace Hopper: the same arrays in one managed slab with preferred location GPU and accessed-by host; host_access waits for the stream and returns the array itself; cudaMalloc memory is used for nothing the host must read. That path is functional only until a node is measured and is tested on x86 by forcing managed memory. The host arrays of every path come from the presolve's memory resource, so that device-built arrays are moved in without a zero fill and huge pages can be asked for (with huge pages the host constructor goes from 34 to 24 s on 15 instances, HISTORY Entry 32). Without CUDA the host state on ordinary memory and the host executor are the whole engine.

7. VERIFICATION

1. The host executor is the reference and takes every check that exists: the 473 unit tests (those that assert a visit order or a stop inside a list are rewritten; 7 failed under the prototype), the oracle campaign, the structure and crush checks, the second presolve, and the 99 wrong engines, re-anchored in three groups: wrong rule (in the shared headers, one patch covers both executors), wrong host step (a tie rule reversed, a read claim ignored, a fold in the wrong order, a row not folded again), wrong kernel. A check counts only after it failed on its wrong engine.
2. The split milestone has an audit in assertion builds and in the campaign: an evaluation changes nothing (no counter, list, record, activity or statistic), and evaluation followed by application equals the old visit.
3. Step equality: both executors run the same step on the same state and every output array is compared bit for bit, on all length classes (0, 1, 64, 65, 8,192, 8,193, more than 65,535 entries), with 0, 1, 2 and more than 65,535 infinite terms, with rows that cross the 64-entry boundary through deletions, with MIP variable types.
4. Step-hash sequences: host forced, device forced, default hybrid and thresholds drawn at random (including 0, infinity and a coin per step), both candidate forms, both memory paths, on the unit models, the 240 and the 40.
5. Campaign through the device: unions of random models built only from blocks whose host presolve ends with a reduced model (48.4% of 7.2 million campaign models do), the oracle per block; unions with one planted infeasible or unbounded block, compared in status and smallest index; a standalone mode at a low rate for statuses, budget stops and limited calls; a new profile for tolerance edges (two rows that bound one column within 1e-6 with a coefficient ratio above 1e3, signed zeros, bounds one step apart); the validity critic's four crossing models and the signed-zero model as unit tests.
6. Assertion builds recompute every derived array after every step, check that no live row holds a dead entry, run the consistency check of the record stack after every step, and compare the whole host copy with the device after every hand-over.
7. Determinism: three runs, a co-running GPU load, a second GPU type, managed memory forced on x86, a Grace Hopper node; zero differences.
8. Against today's engine, per switch: both positions on the 240 and the 40; rows, columns and nonzeros; a strength report for every instance whose hash differs (bounds and coefficients of the two reduced models; neos-1122047, where the batch order turns 71,994 tightening findings into 7,238, by name); stopped and limited calls and fixed points.
9. The review rules of cuOpt PR 2372 (counts in separate uint16 arrays, counted in the scan with one integer atomic only when nonzero, tests with MIP variable types, every kernel variant benchmarked before it replaces the current one).
10. Overheads in every run: upload, download, hand-overs (count, bytes, seconds), column merges, rebuilds, record flushes, output, hash, landing of results in host memory, context creation apart.

8. MILESTONES (content and gate of each are in the field milestones)

M0 Groundwork and the missing measurements (no result changes): the nvcc bit test of the shared headers; mock device steps on recorded list sizes; host timings on the GPU node; the instrumented run and the batch prototype on the 31 and the nine; scratch probes of every change that is not measured yet; the step-hash harness and the screened union mode; the code inventory per milestone.
M1 Evaluation split from application, shared value functions (no result changes). Before it: the reviews of simple probing and of the parallel-row repair are finished and the triage of the oracle failures is in.
M2 Device state build under today's engine (no result changes); runs beside M3 to M5.
M3 The state without history on the host, one switch at a time (activities as values, lists as sets, rebuild on counters, least-common-multiple scale, sum hash, budgets between steps).
M4 The cascade families as batch steps on the host (crossing rule, total order of the combine rule).
M5 The structural families as batch steps on the host; records, storage, output. Then the owner decides whether the batch order becomes the default.
M6 Device executor: foundation, storage, output, hash; every other step through the transitional rule.
M7 Device executor: the cascade families.
M8 Device executor: parallel rows.
M9 Device executor: the alias pipeline (doubleton rule, simple probing).
M10 Device executor: the singleton-column family and the record pool. End of the narrow reading.
M11 The nine remaining column rules, each written once; its host parts can start after M1.
M12 Thresholds, second GPU type, Grace Hopper, cuOpt adapter, acceptance; then the owner's go for Stage 2.
Size of the change, from counts of this round: the engine has 8,829 lines; the files whose meaning changes hold about 6,200 of them and 90 of the 99 wrong engines are text patches in those files (completeness critic's count); the tests are 24,721 lines; the existing device state build, one part of about twenty, is 961 lines with 508 of tests and 816 of bench. M0 delivers the inventory per milestone.

9. ACCEPTANCE CRITERIA FOR 'COMPLETE' (full text in the field acceptance)

1. Every step has a device form; a step runs on the host only for a reason of the closed list of section 1, each with a figure from the finished engine; no step uses the transitional rule.
2. device_min_nnz comes from a stored crossover measurement and is at most 200,000 unless the owner agrees to more.
3. Identity: equal step-hash sequences for the build without CUDA, host forced, device forced, hybrid and random thresholds, three runs, two GPU types, both memory paths, on the 240 and the 40.
4. Share of the counted work on the device: at least 95% per size class from 2e5 nonzeros and 90% from device_min_nnz to 2e5 (targets from the simulation; M0 confirms them).
5. Host seconds outside the driver bounded per size class (proposal: 10% of the hybrid time from 2e5 nonzeros, with the landing of results reported apart).
6. No instance slower than the host path of the same engine by more than the larger of 5% and 2 ms.
7. Time against today's engine (proposals, to be set by the owner after M7): the nine at least 4 times faster with the result in host memory and 5 times with device arrays; 1e6 to 1e7 at least 3 times; 2e5 to 1e6 at least 1.5 times; the host path at most 1.10 times today's per size class; the HiGHS time rule on every instance from 50,000 nonzeros.
8. Reductions against today's engine within 1.001 in the geometric mean of rows, columns and nonzeros, every instance above 1.005 explained, the strength report, the gates against the HiGHS references met.
9. Campaign and checks through both executors; every wrong engine caught.
10. Overheads in every run. 11. Every assumption in the register with its switch. 12. The build without CUDA passes everything.

10. THE ORDER WITH THE CHEAP COLUMN RULES THAT ARE NOT BUILT (full text in the field order_with_remaining_rules)

Rules first on the host: the Stage 1c gate against HiGHS is reached soonest (1.7% of the nonzeros in the geometric mean are still missing after simple probing: 0.9826 on the HiGHS ladder, 237 of the 240), but every rule is written, reviewed and attacked twice, more order-dependent code enters the engine, and the device work and Stage 2 wait for all of it. GPU work first: each rule is written once, but the gate against HiGHS waits for the whole device work. Recommendation: the GPU work first for the built rule set, and the remaining rules beside it from M1 on, each written once as evaluation plus application, first run on the host in the design's index order, then as a step, then with its kernels. This reverses the owner's decision A9 ('the GPU state build is integrated after the cheap column rules') and needs the owner's word.

11. ASSUMPTIONS AND THEIR SWITCHES (full list in the field assumptions)

Result-changing switches (both positions exist on the host; the device supports the new one): apply_order per family; activities; list_order; rebuild_rule; scale_search; parallel_pairing; alias application and forest rule; probing through the forest; saved_row_rule; budget_check; cell sums; long_row_order; hash_kind. Time-only switches (one result, any value): executor, device_min_nnz, mirror_max_bytes, T_down and T_up per family, hand-back steps, forest thresholds, candidate form, device layout, memory path, huge pages, delivery, byte budget, behaviour after a failed allocation.

12. WHAT WAS REJECTED (full list in the field rejected)

'Bulk-first' as the plan; a second engine with its own state type and a composed two-engine presolve; a rewrite that cannot be switched; the fused cascade wave as the algorithm before it is measured; reproducing today's order on the device; a host copy up to 128 MiB with a return threshold of sixteen times the hand-over threshold; a full rebuild after every structural phase with new entries; binned storage as the default; hashed claim priorities; one tree per row for aliases; floating-point atomics and library reductions for sums; pointer jumping over the maps; unscreened campaign unions; one equation serving all its targets at once; raising the size threshold to reach 'complete'.

13. DEFECTS OF THE CRITIQUES THAT ARE STILL OPEN (full list in the field defects_open)

The nvcc bit test; no measured cost of a real device step; nothing measured on the nine beyond stored totals; the host time of the batch engine (the prototype is at 1.12 times today's with two families); the reductions of leaders, of the alias batch and of the singleton family in two steps; the validity review of the step forms of the nine unbuilt rules; the repairs designed in this step are not prototyped; the strength of the reduced model under the batch order; whether a hand-back regime survives its own cost; Grace Hopper; the cuOpt adapter, the pools and a busy GPU; node hours for the verification; a second GPU type.

14. DECISIONS FOR THE OWNER (each with the recommendation, in the field open_for_owner)

1. May the order of application change (batch steps), with today's order kept as a host option? 2. Which reading closes the stage, and are BHW, sparsify and the aggregator inside it? 3. The order with the remaining rules. 4. Finish the open reviews and the triage first? 5. May the host path be up to 1.10 times slower than today's? 6. The hash. 7. Three rules change their definition. 8. Which old forms may be retired after their gate? 9. The acceptance numbers. 10. The cuOpt adapter with device arrays and the timing convention. 11. The second GPU type and Grace Hopper. 12. The summation order of rows above 8,192 entries. 13. An integer atomic on the ordered image of a double.

FILES OF THIS ROUND THAT THE PLAN USES
R/design_one-algorithm/batch_prototype.patch and analyse_240_final.txt (batch prototype and its result); R/critique_validity-determinism/t_cross.cpp, t_zero.cpp and fix (the crossing models, the signed-zero model, the repaired prototype); R/critique_completeness-migration/critique_probe.patch, probe_results.txt and hybrid_sim.txt (activities as values, least-common-multiple scale, hybrid simulation); R/critique_performance/model.py and the scripts next to it (the time model); R/measure/instrumentation.patch and the bench240 tables (width, depth, tail). The plan itself exists only in this result; I wrote no file. The entry in P/HISTORY.md and the new entries of P/ASSUMPTIONS.md are left to the session that owns the round, because this step was read-only.
```

## Amendments after the independent check (2026-11-28)

Six verifiers checked the plan from six sides: statements about the code, figures, validity of the batch rules, identity of host and device, performance and the hybrid rule, the owner's constraints. A seventh step tried to refute every finding of weight. Only findings that survived are amended here. Verdict: the plan stands with these amendments. No finding is fatal. Nothing is implemented and no device step of the plan has been run; that is unchanged.

Work files of the check: V = /private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/gpu_s1_verify (a scratch directory; a model that an amendment names has to be copied into the tests when it is adopted). Every figure names its instance set. 'Measured' means run in the check, 'recomputed' means from stored files, 'read' means taken from a document. 'The 17' are the 17 local dumps.

### A. The rules of a step

**AM-1. A crossing between two candidates of the row rules gives no infeasible verdict.**
Changes: section 2 (K1), PER_PART 3 (conflict_rule, same_result_as_today), DEFECTS_TAKEN 2, ASSUMPTIONS 7.
Old: 'A lower bound above the upper by more than the tolerance is infeasible; a smaller crossing gives the column the upper candidate.'
New: For candidates of the row rules (singleton rows, forcing rows) a crossing between candidates of one step gives no infeasible verdict, whatever its size. The column takes the upper candidate. Every row whose candidate was cut stays, is listed again, and its next visit decides with today's tests in its own units (rules_rows.cpp:212 and :107-115). The verdict in column units stays for propagation, where today's test is in column units too (rules_bounds.cpp). New unit tests: the four models of V/code/t_clamp2.cpp. New campaign profile: coefficients below 1e-3 and bounds that cross by less than the tolerance over the coefficient.
Reason: Measured on four hand-made models. Today's engine ends reduced_to_empty with a row violation of at most 5e-7 (tolerance 1e-6). The designer's prototype and the repaired prototype end infeasible on all four. The unit test rows_singleton_row_small_coefficient fixes today's convention. The critic's repair covered only the other direction.

**AM-2. Integer rounding is done per candidate, before the reduction.**
Changes: section 2 (K1), PER_PART 8 (bound transfer), M1 precondition.
Old: 'Integer rounding comes after the reduction.' and 'deleted in the step only when the bounds written to its columns are at least as tight as its own candidates'.
New: Each candidate is rounded by its own value function before the reduction; the slack of that rounding may depend on the source (the map of a transfer, the coefficient of a row). The reduction runs on the rounded values under (value, source index). 'At least as tight' compares the written bound with the rounded candidate. The loop wording holds for rounded candidates.
Reason: (a) It is today's order: tighten_lower rounds each candidate. (b) For a transfer to an integer root the other order gives r >= 2 where today gives 3, and a child then lies 4e-4 below its bound (exact model). (c) The two orders differ in the sign of a zero bound (bits 0000000000000000 against 8000000000000000, rerun). (d) Under the literal old text the row 1e4 x >= 20000.004 with x integer is never deleted: its candidate 2.0000004 is above the written bound 2 in every step, and the call ends on its budget.
Defect of today's engine, found on the way: for that model today's engine and the repaired batch prototype return x = 2 with a row violation of 0.004 (measured, one integer column). The exact answer is 3. The rounding slack is in column units while the row is deleted. It goes to the triage list as a precondition of M1, with the model. The form the engine already uses elsewhere (slack = feasibility / max(1, |a|), transfer_bounds and the R7 text) is the candidate repair, behind its own switch.

**AM-3. K1 against existing bounds; NaN; sort keys.**
Changes: section 2 (K1), PER_PART 11 and 16 (sort keys).
New: The reduced pair is applied against the existing bounds by the rule of tighten_lower and tighten_upper: a candidate that crosses the existing opposite bound within the tolerance fixes the column at the existing bound, so a column never leaves the box of the step's start. A NaN candidate is an error in every build, never a value of the reduction. A sort key gets +0.0 added before its integer image is taken, on both executors.
Reason: The old sentence was not defined against an existing bound; read literally a column moves to -5e-7 and a row deleted earlier is off by 5e-3 (exact model). With a NaN, 16 of 804 small reductions differ from the loop (rerun). The ordered image puts every -0 before every +0, a double comparison does not.

**AM-4. A map whose transferred bound was cut is not applied in the step.**
Changes: PER_PART 8, section 3.8.
Old: 'Bound transfer to the roots by the combine rule; the children of a root that the transfer fixes are fixed at their image.'
New: A map whose transferred bound was cut by the combine rule (the bound written to the root is less tight than the map's candidate) is not applied in the step. Its row and its column stay, and the row is tested in its own units in the next step. The maps below it keep their composition, which the kept row still implies. Unit test: the three-column model of V/validity/t_alias.cpp in both row orders.
Reason: Model: r in [-1,1], c1 in [5e-3,1], c2 in [-1,0], rows 1e4 r - c1 = 0 and r - c2 = 0. Today's engine gives a row violation of 5e-7 with the big row first and 0.005 with the big row second (measured). The batch form as written gives 0.005 in every order (derived; the alias batch is not prototyped). The defect of today's order goes to the triage list.

**AM-5. Parallel rows: the kept row, a cut removed row, one pivot per target.**
Changes: PER_PART 7 (device_algorithm, same_result_as_today, conflict_rule), section 3.7.
Old: 'in a class without singleton columns the smallest row is kept ... the other rows are deleted' and 'merged into their smallest row'.
New: The kept row of a class is the one the repair in review defines (commit 5b87457a5: the row with the larger coefficients, read at the smallest core column; the smaller index among equals), found as a reduction over the class after the leader rounds. A removed row whose side image was cut by a crossing stays for the next step. In the subtraction step a target takes one pivot per step (fewest singleton columns, then the smallest index, among the rows the merge kept). 'No reduction of today reads what another wrote in the same call' holds per row visit, not per reduction: a merge reads the sides that an earlier merge of the same visit wrote.
Reason: Model: x + y <= 5 and 1e6 x + 1e6 y >= 5e6 + 0.4. Today's order ends reduced_to_empty with a row violation of 0.4 (measured); with the rows swapped it is infeasible. The plan made the repair a precondition of M1 and then specified the unrepaired rule. Two equations subtracted from one target in one step leave a row that a feasible point violates (exact model).

**AM-6. The reservation is a subset of the ascending loop, not the loop.**
Changes: section 2 (K3), UNSURE, DEFECTS_OPEN, M11 gate.
Old: 'The applied set is what a loop in ascending key gives when every premise is tested at application.'
New: The applied set is a subset of that loop's set. It equals the loop in which every earlier candidate blocks, applied or not. Both executors compute it in two phases over all candidates of the step, so the claims of losers count. APPLY reads only what FIND stored, because an applied unit with a larger key may write a cell that an applied unit read. The gates 'batch against sequential' compare reductions, not sets, and count the units lost per step. New tests: the chain of three units with write claims {a}, {a,b}, {b} as a step-equality test; 'a loser's claim ignored' as a wrong host step.
Reason: Rerun of V/determinism_identity/k3.py: 2 of 27 exhaustive systems and 48,341 of 200,000 random systems differ from the loop; none is outside a subset.

**AM-7. Single-lock alias (R8): the reservation is on the aliased column.**
Changes: PER_PART 13 (conflict_rule).
Old: 'reservation (K3) on the binary, then the forest and the ordered fold'.
New: Reservation on the aliased column with the row index as the key: one lock edge per column per step. The binary needs no reservation. The model below goes into the cell table and the review of the rule.
Reason: min p1 + p2, x - p1 >= 0, x + p2 <= 1, three binaries, cost of x zero. Both rows pass R8 for x on the state of the step. Two lock aliases of one column cannot be composed: the optimum 0 becomes 1 (exact model; the R8 text read).

**AM-8. Implied equation (A): the unit read-claims every row of its column.**
Changes: PER_PART 14 (device_algorithm, conflict_rule).
Old: 'reservation (K3) on the row with the column index as the key'.
New: The unit write-claims its lock row and read-claims every row that holds its column, because its lock count reads their sides.
Reason: min -x1 - x2, x1 - 2 x2 <= 0, x2 - 2 x1 <= 0, x >= 0. The model is unbounded. Both columns pass on the state of the step and claim different rows; with both rows as equations only (0,0) is left (exact model; the rule text read).

**AM-9. Merge of parallel columns (R9): the merged bounds are a sum of intervals.**
Changes: PER_PART 11 (device_algorithm, conflict_rule).
Old: 'its bounds are the minimum and maximum of the scaled bounds' and 'merges inside one class combine by minimum and maximum'.
New: The merged bounds are l_kept + min(sigma l, sigma u) and u_kept + max(sigma l, sigma u) of the removed column, as in stage1c_design/v2_design.txt (R9). For a class of three or more they are the ordered fold (K2) of the interval ends, or the merge is one pair per kept column per step. The bound of the kept column gets wider: that is a named exception to K1, and the rows of the kept column are folded again.
Reason: Three parallel columns in [0,1] with one row: the exact merged column has the range [0,3]; minimum and maximum give [0,2] or [0,1] (exact model).

**AM-10. Singleton-column family: what the prefix covers; a folded cost that is not finite.**
Changes: PER_PART 9.
New: The saved-entry prefix is taken over the zero-cost units whose row stays; every other unit is accepted, as today (rules_singleton_cols.cpp:354-360). The finite-cost guard is tested on the folded cost: units that feed a column whose folded cost is not finite are not applied, from the largest row down until the sum is finite, and stay listed.
Reason: Today's limit counts only the entries saved of rows that stay. Two pivot rows can each pass the guard alone and fold to infinity (derived).

**AM-11. Row tightening: a row scaled in a step is strengthened in the next step.**
Changes: PER_PART 5, section 3.5.
Old: 'coefficient strengthening per entry from the stored activity'.
New: A row that the step scales, or whose sides it rounds, is not strengthened in that step. It is listed again and is strengthened in the next step from the fold of its new coefficients. Step equality is tested on a row that is scaled and strengthened.
Reason: Today's visit sums the scaled row again before it strengthens (rules_tightening.cpp:204 and :265; read). The activity stored at the start of the step belongs to the unscaled coefficients. 2,530 visits do both on the 17, 2,525 of them on bnatt500 (the code verifier's counters, read).

**AM-12. The order of record kinds is written down before M5.**
Changes: section 2 (state item 6), PER_PART 21, M11.
New: The kind order is fixed with at least these constraints: parallel_fix before merge before dual_bound; col_scale before implied_integer. A family that pushes a transaction orders by unit first inside a step. The crush check runs per step, next to the stack check.
Reason: The forward replay of the designed records needs these orders (read by the validity verifier in two design paragraphs; not checked against the crush code by me).

### B. Identity of host and device

**AM-13. Every sum over a line gets a stated form; two new switches.**
Changes: IDENTITY_RULE ('one fold order for every floating-point sum'), PER_PART 5 and 20, section 11, M0, M3, M5.
New: The plan carries a table of every floating-point sum over a line that is not an activity (fixed terms, drift, size, the new side of the strengthening, saved entries; at least 14 loops today are running sums in storage order). Each has one of two forms: a running sum by one owner thread, with a length cap under H5, or a lane fold by slot position. Switch row_sums = running | lanes (result-changing, M3). Switch storage_order (rows: slot reuse | sorted; columns: append | merge; result-changing, M5). Each has an entry in ASSUMPTIONS.md and its own hash gate on the 240 and the 40; both are probed on the host in M0. 'The arithmetic of the fixed terms is today's' (PER_PART 20) holds only in the position running.
Reason: PER_PART 5 and PER_PART 20 contradict each other. Measured on random rows: a running compensated sum and a lane fold differ in (hi, lo) on 11,525 of 11,960 rows and in hi + lo on none; plain sums differ on 9,330. On the 17, rows are out of column order at the output on 7 instances (code verifier, read). Sorted rows are a result change that no listed switch covers.

**AM-14. The work of a step is one integer function; the stall rule is defined on steps.**
Changes: section 4 ('today's unit'), section 5 ('work, first_work ... keep their meaning'), section 2 (status record), IDENTITY_RULE.
New: Per step type one integer function in the shared headers: work = f(list, sizes of the listed lines, findings of the step). It charges nothing for a fold and does not read the used slots of a column. The stall rule reads the work at the end of the last step with a finding. Work and first_work change their meaning and are listed as such. The status record also carries 'limited' and the per-call counters (dual findings, sub-rule findings, column statistics). Section 5 adds the statistics that depend on the order of visits: records, the refusal counts of the singleton columns, rebuilds. Step equality compares the work and the stop decision in both candidate forms. Identity is claimed for time_limit = infinity; a finite limit is tested between steps.
Reason: Today's charges depend on findings, on a refresh and on used slots (read). Budgets decide which steps exist, so two executors that count differently end with different models.

**AM-15. Versions, change counters and activities at a step boundary.**
Changes: section 2 ('The state holds no history'), section 6, IDENTITY_RULE.
New: After a step, row_version is +1 per row changed in the step; bound changes, side changes and findings count cells written, not candidates. Activities and their marks are derived values: the step hash and the array comparison leave them out and assertion builds compare the fold of every row; or both executors fold exactly the marked rows in UPDATE. M3 states which. A step that finds infeasible applies nothing and counts nothing.
Reason: The state hash holds row_version, the change counters and the activity arrays (hash.cpp:140-177, read). A loop that writes twice and a reduction that writes once store the same bound and different counters.

**AM-16. Masks and counts of infinite terms.**
Changes: PER_PART 4 ('marks are stores of a constant'), state item 1 ('two uint16 arrays that saturate'), PER_PART 1, ASSUMPTIONS 25, section 7 items 3 and 9.
New: A mask is one byte array per list, or a 32-bit word with atomicOr on both sides; never a plain store into a shared 16-bit word. Counts of infinite terms are capped at 2, as the review of cuOpt PR 2372 asks; the thread group that owns the row reduces and clamps; where a scan counts across rows it uses a 32-bit scratch that is clamped. No 16-bit atomic add is assumed. The count type follows the activities switch: exact int32 counts in the position incremental. New step-equality cases: 65,535, 65,536 and 65,537 infinite terms; two different events on one row in one step.
Reason: list_mask_t is one uint16 word with 11 row lists, and two events of one step carry different bit sets (state.hpp, read). The cap was left out only because a capped count cannot be decremented; with activities as values nothing is decremented. In the incremental position counts are decremented, so assumption 25 is false there. Which 16-bit atomics exist is from memory; M0 compiles a two-line kernel (AM-33).

**AM-17. Contraction is tested inside the libraries; preconditions of the identity rule.**
Changes: section 2 ('Two executors'), M0 item 1, M12, ACCEPTANCE 3, IDENTITY_RULE.
New: Both libraries carry a self-test with stored result bits (dd_add_product, the candidate, the fused multiply-add of candidate_slack, the alias offset). It runs at the first host call and at device context creation and fails hard. The nvcc bit test is run once with contraction on, on purpose, and must fail. Criterion 3 adds: the step hashes of cuOpt's own build of host and device equal the standalone ones on the 240 and the 40. Preconditions of the identity rule: the calling thread rounds to nearest and does not flush to zero (the entry checks, or saves, sets and restores); time_limit = infinity. 'The one explicit fused multiply-add' is corrected: there are two (numerics.hpp:60, implied.hpp:176).
Reason: cuOpt globs the engine sources into its own library and sets -ffp-contract=off per source; its CUDA flags hold no --fmad=false (read). Nothing sets or tests the floating-point mode in gprs or in cuOpt (grep).

**AM-18. The hash and the array comparison run over live slots.**
Changes: PER_PART 23, M3 (e), section 6.
New: Hash and comparison cover live slots only, with position = (line, rank in line). Layout arrays are compared only between executors that claim an equal layout.
Reason: Dead slots, free capacity and unfilled device arrays have no defined content.

**AM-19. The rank of the single infinite entry is an output of the fold.**
Changes: section 2 (state item 1), PER_PART 1.
Old: 'the position of a single infinite entry is not stored, the entry recognises its own term'.
New: The fold gives, per side, the rank of the single infinite entry; it is valid when the count is 1. It is a value of the fold, not history.
Reason: Propagation jumps to that entry for one work unit today. 327,001 sides are served that way in 1,966,644 propagation row visits on the 17; on rd-rplusc-21 266,200 in 276,874 (code verifier's counters, read).

**AM-20. M1 defines the unit of each family; the audit is reworded.**
Changes: MILESTONES M1 (content, gate), section 7 item 2, M0 item 1.
Old: 'an evaluation changes nothing (no counter, list, record, activity, statistic), and evaluation followed by application equals the old visit'.
New: M1 first defines the unit per family: the entry for propagation, three stages for row tightening, the pair for the parallel rows, the map for the alias application, the row unit and the column unit for the singleton family. The activity refresh is a step of its own before the evaluation. Records, marks, step counters and refusal counts are outputs of the evaluation that the application books. The audit says: an evaluation writes nothing but its outputs, and evaluation followed by application of one unit equals the old unit. M0 item 1: the functions of implied.hpp and integral.hpp that take state_t are split first, or their bit test moves behind M1; forcing.hpp exists only at edd147a7e.
Reason: A propagation row visit applies between its entries and between its two sides, and its candidate path writes records, marks and counters (rules_propagation.cpp:96-185, read).

### C. Figures

**AM-21. Criterion 7: the pessimistic brackets.**
Changes: ACCEPTANCE 7, UNSURE ('sit below its pessimistic setting').
Old: 'the 67 from 2e5 to 1e6 at least 1.5 times (model 2.8 to 4.0, pessimistic 1.7 to 2.3)'.
New: In the critic's pessimistic setting (fixed costs times 3, 0.5 ns per work unit, host path 25% slower) the class 2e5 to 1e6 gives 1.5 to 1.9, and 1.46 with the host copy created in the call. The proposed 1.5 is at that value, not below it. 'Not slower in the sum' between device_min_nnz and 2e5 fails in that setting (0.82 for the plan's own rule). The bracket of the nine (4.9 and 5.9) uses a harsher setting (fixed costs times 3, 1.0 ns per unit, 10 device steps per cascade call); under it the class 1e6 to 1e7 gives 4.2 and 2e5 to 1e6 gives 1.4. Every bracket names its inputs. No factor is set before AM-33 (a), (b) and (h) are in.
Reason: Rerun of the critic's model (a model on the 240: node host seconds, Mac trace, H200 costs). The old '2.3' is the figure for fixed costs doubled.

**AM-22. The model figures of the plan's own hybrid rule.**
Changes: HYBRID_RULE H3 and H4, ACCEPTANCE 6 and 7.
New: The plan's rule (device from 1e5 nonzeros, hand-back only up to 16 MiB after two low steps, copy created in the call) in the critic's model, today over model for the classes 1e5 to 2e5, 2e5 to 1e6 and 1e6 to 1e7: 1.65, 3.65, 10.80 at base; 1.12, 2.21, 8.17 with fixed costs doubled; 0.85, 1.58, 6.57 at three times; 0.82, 1.51, 5.51 pessimistic. Criterion 6 is broken by 0, 7, 26 and 22 instances of the 240. Below 1e6 nonzeros the result hangs on the fixed cost of a step, above on the device cost per work unit. Neither is measured.
Reason: No script of the round held the plan's rule. Rerun of the figures verifier's copy of the model; it is a model on a model.

**AM-23. The ladder figures on both sets.**
Changes: section 10, ORDER_WITH_REMAINING_RULES (Order 2).
Old: '1.7% of the nonzeros ... are still missing after simple probing: 0.9826 on the HiGHS ladder, 237 of the 240' and 'about 0.983'.
New: Remaining nonzeros against all cheap rules, ladder of HiGHS (read): on 237 of the 240, 0.9726 with the merged rule set and 0.9826 with simple probing; on the 40, 0.9233 and 0.9377. The gate of Stage 1c covers both sets. On the 40 the worth sits in dual bound strengthening and the single-lock alias (0.9396 to 0.9974). These are figures of HiGHS's rule sets, not of the engine.

**AM-24. The thresholds that were simulated.**
Changes: HYBRID_RULE H3 ('the ones the critics simulated'), section 4 last paragraph, ACCEPTANCE 4.
New: The 98.7% and 2.2 switches belong to T_down = 20,000 for every family and a cap of 128 MiB (114 instances of the 240). With the values H3 states and the 16 MiB cap (recomputed on the recorded host steps of the Mac trace): 94.8% of the rule work on the device in the class 1e5 to 2e5 and 99.4% in 2e5 to 1e6; lowest ns1830653 67.8%, momentum1 72.5%, n2seq36q 79.1%; 42 instances can hand back.

**AM-25. The reason of H4 on the instances it covers.**
Changes: HYBRID_RULE H4, 'Not known'.
New: H4 covers 84 of the 240: the 30 and 54 instances below 1e6 nonzeros. The 0.3% holds for the 30. For the 54, calls below one fixed step are 4.1% of the Mac engine seconds, and all their recorded steps at the plan's own step costs (200 microseconds per cascade round, 1.5 ms per structural step) cost 0.94 s, 21% of their 4.47 node seconds, against 2.9% for the 30 (recomputed). The instances that lose most on small steps in the performance verifier's model (not rerun by me) are atlanta-ip, irish-electricity, neos-3555904-turama, snp-02-004-104, neos-5093327-huahum and neos-2987310-joes. They are named in M0 (AM-33) and in the gates of M7. Whether the cap is a state size or needs a way down for long narrow chains is decided on that measurement.

**AM-26. Corrections.**
- H3: unitcal_7 has 1,398 family calls in 33 ms on the node; at one device step per call that is about 3.6 times its node time. The 6.9 times is against the 17 ms of the Mac trace (recomputed).
- Section 6 and PER_PART 9: dlr2 saves 55.4 million entries (0.66 GB). 71.0 million (0.85 GB) is the sum over the nine (recomputed).
- Section 5 and REDUCTIONS_LOST: the crossing rule kept no row on any of the 240 (its counter is 0 in all 240 result files), so '0 of 240 differ' shows that it is neutral where no crossing occurs; its effect is measured on hand-made models only. With it the row-rule rounds are 2,420 against 2,263 today, +6.9%, not +5.7% (recomputed).
- REDUCTIONS_LOST and DEFECTS_OPEN: '31 further instances' is 35 (32 with an equal hash). On neos-1122047 ascending lists alone take the tightening findings from 71,994 to 14,092; batch rounds take them to 7,238 (recomputed).
- Section 0: the last register entry is AS-339.
- H5: the plan's own two figures (about a millisecond of launches, 37 ns per alias on the Mac) put the break-even of the forest near 27,000 edges, not 100,000. The value comes from the measurement at M9.
- Section 2: a column gets a new entry through two mechanisms in four rules (doubleton, simple probing, the lock alias, the parallel-row subtraction).
- Section 3.3: the row rules also scan an empty row and every row that passes the O(1) forcing pre-test.

**AM-27. The host path of the batch prototype per size class.**
Changes: OPEN_FOR_OWNER 5, DEFECTS_TAKEN 17, DEFECTS_OPEN, M4 gate, ACCEPTANCE 7.
New: The 1.12 is a sum. Per size class (recomputed twice; 240 dumps, Mac, smallest of five passes, two of seven families converted): below 5e4 1.14; 5e4 to 1e5 1.07; 1e5 to 2e5 1.23; 2e5 to 1e6 1.12; 1e6 to 1e7 1.13; above 1.05. 134 of 240 instances are above 1.10. neos-1122047 is 1.51 times slower. On the paired node run it used 0.75 of its HiGHS allowance, so the batch form is projected at 1.13 of it (a Mac ratio on node seconds). The question on the ceiling is asked after AM-33 (c). The M4 gate names neos-1122047 for time as well as for strength.

### D. Hybrid rule, memory, acceptance

**AM-28. Criteria 4 and 6: which one gives way.**
Changes: ACCEPTANCE 4 and 6, HYBRID_RULE H3 and 'How the values are fixed'.
New: Criterion 6 is the hard one. Criterion 4 becomes: no threshold (T_down, T_up, mirror_max_bytes, the forest thresholds) lies above the break-even of its family measured on the finished engine (device step time against host step time at that work, same node) by more than a factor the owner sets (proposal: 2); the share of the rule work on the device is reported per size class, with every instance below 80% listed. The 95% and 90% are targets, not conditions. Nothing is fixed before real steps are timed (AM-33).
Reason: In the model the criteria hold together only at the critic's base step cost: criterion 6 is broken by 0 instances of the 240 at base, 7 with fixed costs doubled and 26 at three times (AM-22). The designers and the critic differ by 1.5 to 2 on that cost, and Stage 0 measured 77 to 93 microseconds for a step without kernel work (read). A bound by break-even also keeps 'complete' from being reached by raising a threshold. The owner decides (AM-39, D6).

**AM-29. The memory fallback is on the closed list, with a bound; the memory plan is completed.**
Changes: HYBRID_RULE H7, section 1 (closed list), ACCEPTANCE 1, section 6, state item 3, PER_PART 20.
New: The closed list gets 'the memory plan exceeds the byte budget'. Bound: on the node of the acceptance no instance of the 240 and the 40 falls back, standalone and inside cuOpt with its heuristic memory counted. The smallest GPU memory for which the criteria are claimed is stated. plan_memory adds: the uploaded input that the output step compares against (12 bytes per nonzero plus row and column arrays), the reduced model next to the state, the record arrays and the saved-entry pool, fold and candidate buffers, and a reserve for the column store. The rebuild counter counts the holes that moved columns leave, or the column store is compacted when its tail is full; the plan says what a full tail does inside a step.
Reason: H7 was in no criterion and the byte budget had no value. Recomputed from the stored sizes of the nine: the plan's formula peaks at 18.6 GB on neos-4535459-waipa, 21.7 GB with the uploaded input, 52.6 GB next to the 120 bytes per nonzero of cuOpt's heuristics (hawaiiv10-130: 37.6 GB). On a 24 GB GPU inside cuOpt these two would run on the host: 33.2 of the 88.3 node seconds of the nine. A moved column takes twice its capacity and leaves its old slots (state_ops.cpp:167-197, read); the store reached 1.61 times the input nonzeros on uccase9 (17, code verifier, not rerun). Whether a failed allocation falls back or is an error is the owner's (AS-288 says error; AM-39, D7).

**AM-30. Wording of the acceptance criteria.**
Changes: ACCEPTANCE preamble and criteria 4, 5, 7, 8, 9, 10; gates of M1 and M3 to M5.
New:
- Preamble: every criterion is evaluated at the shipped defaults. A family whose shipped default is apply_order = sequential fails criterion 1.
- Criterion 4 counts the rules in work units. State build, storage, output and hash have no work count (46.9% of the node seconds on the 240, recomputed); they are covered by criterion 5 in seconds.
- Criterion 5: hand-over seconds are host seconds. With delivery to host memory the landing is inside the bound, not beside it. The adapter's seconds are reported with it.
- Criterion 7 names the timed scope: the hash kind and the delivery of the timed run.
- Criterion 8, narrow reading: the gate of Stage 1c (0.99) cannot be met by the built rule set (ladder 0.9826 on 237 of the 240, 0.9377 on the 40). Under the narrow reading the criterion is 'not below the engine's own ratio before the change'.
- Criterion 9 and the gates of M1 and M3 to M5: host campaigns of at least 1e9 models per kind and build. The device volume is sized from the campaign rate measured at M6, or the criterion says that the device is covered through criterion 3.
- Criterion 10 adds the adapter's seconds (conversions in and out, the reductions of mode 3) from the first milestone that runs inside cuOpt.
Reason: The last campaign found one failure in 300 million models (read); 1e8 models see such a class with probability 0.28, 1e9 with 0.96, and a billion models take about 20 minutes on a node with 24 processes (read). Inside cuOpt the presolve call takes 94 s against 18 s for the engine on the 240 (read, not timed in one run).

**AM-31. No Grace Hopper node in the gate.**
Changes: ACCEPTANCE 3, M12, section 7 item 7.
Old: 'on a Grace Hopper node. Zero differences.'
New: The managed path is tested by forcing it on x86. A run on a Grace Hopper node is not a condition for the go to Stage 2, unless the owner asks for a hash-only run there.
Reason: The owner's word of Entry 45 ('leave the benchmarking and perf optimization there to the end'), A10 and AS-308 ('outside every gate'). The plan's own question 11 recommended keeping A10.

**AM-32. M7: cuOpt's load-balanced code is a pattern for thread groups only.**
Changes: MILESTONES M7 (content), DEFECTS_OPEN (last item).
New: The code is not built in this worktree (no CMake file names it). It sums with plain block and warp reductions, has no list and sweeps the whole model per round, uses the candidate form that numerics.hpp forbids, and orders a bin by atomics. Only the layout of thread groups is taken. Every kernel of M7 is new. The CUDA graph variant is priced only after the cost of new node parameters per step is measured.
Reason: The files were read by the performance verifier; I checked the CMake files and the cited kernel lines.

### E. Milestones

**AM-33. M0 measures real device steps and the node run of the prototype, first.**
Changes: MILESTONES M0 (content, gate), DEFECTS_OPEN, section 8.
New, in this order, and reported before the owner confirms M3 to M5:
(a) Real device steps outside the engine: one propagation step in both candidate forms and one row-rule step (a list-restricted candidate kernel and the apply, mask and status kernels added to the Stage 0 benchmark), on states dumped at recorded steps, compared bit for bit with the host batch prototype. By name: unitcal_7, momentum1, neos-827175, atlanta-ip, irish-electricity, snp-02-004-104, neos-3555904-turama, neos-5093327-huahum, neos-2987310-joes, neos-1122047, and the class 5e4 to 1e5. The host is timed on the same node. Result: the fixed cost of a step and of a hand-over per size class.
(b) One real kernel per family kind (fold, candidates, tightening scan, key and sort) on index-order storage with scheduling lists, on the 31 and the nine: the device cost per work unit. The instrumented host run on the nine comes first.
(c) The batch prototype on the node: three alternating passes against the base on the 240 and the 31, per size class, per family and per instance, with the HiGHS allowance beside it.
(d) Host probes: the lazy fold (time per size class); row_sums running against lanes; the storage order; the largest column store on the 240 and the nine.
(e) The mock steps also carry the parts whose cost grows with the model: the mask scan over all lines, the reductions behind the status record, the column merge with its partition; with the line counts of the nine.
(f) Compile checks: which 16-bit atomics exist; the nvcc bit test once with contraction on, which must fail; the fused multiply-add of candidate_slack in the bit test.
(g) Undo and crush seconds on the 240 and the nine.
(h) The time model is redone with the plan's own rule and these numbers before any threshold or factor is set.
Reason: As planned, M3 to M5 change about 6,200 engine lines on the numbers of mock steps, and the first real step is timed at M6. The mock steps retire the fixed cost, not the cost per work unit, which carries every factor from 1e6 nonzeros up (0.25 ns assumed against 4.5 to 5.0 ns on the host).

**AM-34. A check inside cuOpt before M12; the cluster.**
Changes: gates of M4 and M5, M5 ('Then the owner decides').
New: The gates of M4 and M5 add the presolve-only report inside cuOpt (adapter modes 2 and 3) on the 240. The plan says where cluster run 1 of the Stage 1 gate runs (DESIGN.md section 13: no wrong status, every postsolved solution feasible; prepared and never submitted). The owner decides whether a standard cluster pair on apply_order (sequential against batch) runs after M5 and before the kernels of M7 (AM-39, D9).
Reason: The batch order changes reduced models (6 of 240 hashes on the prototype). The end test over solution quality can prefer the sequential order for a family, and that family then runs on the host on every machine: the kernels of M6 to M10 for it would be lost.

**AM-35. Every gate carries the owner's working rules.**
Changes: all gates of MILESTONES; ASSUMPTIONS 24 and 25; section 11.
New: Every gate adds: one commit, pushed; the register entries of the milestone with their switches; the history entry. Device campaigns run under restartable drivers. Assumptions 24 and 25 get a switch or a stated reason why none can exist. Entered as assumptions: ties to the smaller source index in K1; records ordered by step, kind and unit; children of a fixed root without an alias record; the loss allowed against today's engine (1.001 and 1.005); screened unions; the byte limit of a record flush; the share of nonzeros that triggers a full re-sort of the column store; the length cap of a sequential test in a row; crossing_value and offset_sum in section 11. The time gates of M9 and M10 give a figure per size class on the 240 next to the named instances.

**AM-36. Parts without an executor or a rule; the postsolve; cuSPARSE.**
Changes: PER_PART (new entries), PER_PART 21 (why_host), ACCEPTANCE 1, section 3.1.
New: PER_PART gets entries for the time limit (tested between steps), the stall stop (defined on steps, AM-14), row_version (AM-15), the record hash, the family data that outlives a call (parallel rows, singleton columns), and the primal-only and sub-rule switches. The reason for undo and crush on the host is 'not measured': DESIGN.md section 9 says that GPU batches are parallel loops in the undo, and the records of one step are independent. They are classified after AM-33 (g). Section 3.1 gets one line: cuSPARSE was measured in Entry 32 (csr2csc and the radix sort within 1% end to end; its order inside a column is only observed; SpMV cannot give the lane fold); csr2csc stays a benchmarked variant.

### F. Scope

**AM-37. New section 15: re-entry from later phases.**
Changes: new section; section 2 (state item 4), REJECTED 15, H1.
New: At run time the stages are phases of one loop (DESIGN.md A7, the owner's question at HISTORY.md:1562). The plan has to hold when a later phase hands the state back to Stage 1. Required: (1) what a later phase leaves: rows in column order without dead entries, counters, activity marks, list masks, a status record; (2) one storage step for rows that were added or grew (clique extension, fill-in of the aggregator, appended indicator rows): a rebuild from the state into new stores, on both executors; (3) a host-to-device copy of a reduced state, for a model below device_min_nnz that a wide phase wants on the device; (4) whether the identity rule binds later phases, decided when they are designed. 'No rule of Stage 1 lets a row grow' stays true for Stage 1 only. Before M5 fixes the storage, the state changes of Stage 2 and of A8 steps 2 and 3 are checked against state item 4. None of this is designed or costed here.

**AM-38. cuOpt's own reductions.**
Changes: section 3.11, DEFECTS_OPEN (adapter), OPEN_FOR_OWNER 2.
New, one line per reduction (DESIGN.md A6, the owner's message at HISTORY.md:1566): single-lock dual aggregation is part 13 (R8), on the device through the alias pipeline, M11. BHW coefficient reduction is a map over rows next to row tightening, on both executors; A8 puts it after the gate of Stage 1c; its milestone follows the owner's answer to D2. Indicator strengthening (it appends rows) and GF2 come with A8 step 2 and need the storage step of section 15; they are in no milestone of this plan. Block BVE is a wide phase of Stage 2. Until the ports exist, adapter mode 3 runs the four on a host model inside the PaPILO framework, and mode 3 is the arm of every end test. So 'delivered as device arrays' needs a download for mode 3 or the ports; M12 says which.

**AM-39. Section 14 (decisions for the owner) is replaced.**
Asked now:
- D1 (was 1): may the order of application change. Unchanged.
- D2 (was 2): which reading closes the stage; and where A8 steps 2 and 3 (sparsify, aggregator, pairwise dominated columns), GF2, indicator strengthening and BHW stand against 'state 2' of the directive. The stage loop needs the storage step of section 15 whatever the answer.
- D3 (was 3): the order with the remaining rules, which reverses A9; with the ladder figures of both sets (AM-23).
- D4 (new): activities recomputed instead of updated. It reverses AS-291, which the register books as set by the owner. Asked with the host time of the lazy fold from M0; both positions stay until the end test.
- D5 (new): today's engine returns points outside the tolerance on three hand-made models (row violations 5e-3, 4e-3 and 0.4 against 1e-6; the third has a repair in review). Recommendation: repair them with the triage before M1.
- D6 (was 9, changed): the acceptance numbers stay provisional. Which of criteria 4 and 6 gives way (recommendation: AM-28). The loss allowed against today's engine (1.001 in the mean, 1.005 per instance). The factor 1.5 for 2e5 to 1e6 nonzeros is at the pessimistic model value. At a size threshold of 200,000, 50 of the 147 benchmark instances from 5e4 nonzeros never use the device; at 100,000, 21.
- D7 (new): memory. Fallback to the host or a hard error (AS-288 says error); the smallest GPU memory for which 'complete' is claimed, standalone and inside cuOpt.
- D8 (was 10, extended): the adapter with device arrays and the warm call; and whether the adapter's conversions (94 s against 18 s for the engine on the 240, read) and the postsolve are inside 'the whole presolve suite'.
- D9 (new): a standard cluster pair on apply_order after M5 and before the kernels, or the owner accepts that the end test can send a family back to the host; and where cluster run 1 of the Stage 1 gate runs.
Asked after M0, with node figures: D10 (was 5): the ceiling of the host path.
No longer questions: 4 (a standing rule: unreviewed code is not merged); 6 (the hash follows from D1); 7 and 8 (the owner's rule of Entry 45 and the register's closing rule decide: every result-changing switch keeps both positions until its entry is closed by 240 of 240 equal hashes or by the cluster pair at the end; the recommendation to remove old forms after their gate is withdrawn); 11 (Grace Hopper is answered, AM-31; the second GPU type is named at M6); 12 (a measurement of M0); 13 (the plan does not need it).
Missing for the decision and not supplied by the check: a rough duration per milestone.
