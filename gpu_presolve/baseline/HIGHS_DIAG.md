# HiGHS MIP presolve diagnostics (`highs_diag.patch`)

`highs_diag.patch` is a unified diff against HiGHS commit `283bfc859` (1.16.0). It makes the MIP
presolve report where its time goes and what each stage removes, and adds switches that skip
stages. With every switch unset the presolve result is the stock result (checked, see
"Checks"). The `cliquemerge_call` and `probing_round` lines are printed when their call ends,
everything else at the end of each presolve call, through `highsLogUser` at info level, so the
lines are in the log file and on stdout. The longest line seen is under 500 characters, below the
1024 characters HiGHS passes to a user log callback. Every line starts with `DIAG `, fields are
`key=value` separated by one space.

## Build and run

On the node, inside the dev container (`cuopt exec` from the worktree root):

```
cuopt exec 'bash gpu_presolve/baseline/build_highs_diag.sh'
```

The script clones `/home/scratch.acoerduek_sw/gpu_presolve_deps/highs` into
`gpu_presolve_deps/highs_diag_src`, checks out `283bfc859`, applies the patch, builds Release with
`-DHIGHSINT64=ON` (same options as the stock build) in `gpu_presolve_deps/build/highs_diag` and
installs into `gpu_presolve_deps/highs_diag_install` (`bin/highs`, `lib/libhighs.so`,
`include/highs`, `lib/cmake`). A changed patch resets the source copy, applies the patch again and
rebuilds what depends on the patched files. A re-run with an unchanged patch compiles nothing. The
first re-run after a rebuild has twice relinked `libhighs.so` and `bin/highs` from the same
objects (about 15 s) and later re-runs did nothing; the cause was not established (the build tree
is on NFS and ninja 1.13 compares recorded input times). `JOBS=<n>` sets the build jobs
(default 8). The baseline tool is compiled against the installed headers, so rebuild it after the
patch changes (see `README.md`).

Presolve only (the stock way: an options file with `write_presolved_model_file` makes the `highs`
binary presolve, write the presolved model and stop):

```
printf 'write_presolved_model_file = /path/presolved.mps\nlog_file = /path/highs.log\n' > opts.txt
[HIGHS_DIAG_OFF=...] /home/scratch.acoerduek_sw/gpu_presolve_deps/highs_diag_install/bin/highs \
    --options_file opts.txt /home/scratch.acoerduek_sw/cuopt_dev/datasets/mip/miplib2017/<name>.mps
grep '^DIAG ' /path/highs.log
```

A full solve prints one block per presolve call of the main MIP (the root presolve first, then one
block per restart, `restart=<n>` on the `DIAG total` line). Presolve calls of sub-MIPs print
nothing. Each block ends with its `DIAG total` line.

## Lines

Sizes are counted on the live model: rows and columns not deleted, nonzeros in the presolve matrix.
`*_removed` is "before minus after", so a negative value means the stage added nonzeros.

### `DIAG switches off=<list|none> probing_contingent=<n> probing_nostop=<0|1>`

The switch values this process runs with.

### `DIAG stage name=<n> calls=<c> seconds=<s> rows_removed=<r> cols_removed=<c> nnz_removed=<z>`

One line per stage, always all of them, in this order. A stage is a top-level call from the
presolve main loop; whatever it calls (row and column presolve, singleton removal, clique merging)
counts for it. `seconds` is wall time (`steady_clock`).

| name | HiGHS code |
|---|---|
| `setup` | `okSetupPresolveDataStructures`, `setupSubstitutionOpportunities` |
| `initial_scan` | `initialRowAndColPresolve`: one scan of all rows, then all columns |
| `fast_loop` | every `fastPresolveLoop` call of the main loop (row singletons, changed rows, doubleton equations, column singletons, changed columns) |
| `clique_substitutions` | `applyConflictGraphSubstitutions` in the main loop (substitutions waiting in the clique table and the implications; empty at the root presolve, see "Not measured") |
| `fourier_motzkin` | `fourierMotzkin` |
| `aggregator` | `aggregator` |
| `sparsify` | `sparsify` |
| `parallel_rows_cols` | `detectParallelRowsAndCols` |
| `strengthen_inequalities` | `strengthenInequalities` |
| `dominated_columns` | `dominatedColumns` (pairwise dominated columns; both calls) |
| `enumeration` | `enumerateSolutions`, including its `prepareProbing` (first clique extraction, clique row normalisation, propagation) and `finaliseProbing` (clique merging, fixings, variable-bound aggregation) |
| `implied_integers` | `detectImpliedIntegers` (the sweep before probing) |
| `probing` | `runProbing`: `prepareProbing`, the probing loop, `finaliseProbing` (clique merging, fixings, substitutions, variable-bound aggregation), lifting for probing |
| `shrink` | `shrinkProblem` and the matrix rebuild after it in the main loop; the final `shrinkProblem` and `toCSC` |
| `final_scaling` | `scaleMIP` |
| `other` | total minus the stages above (`checkOriginalModelBounds`, loop control, reporting); `calls` is 0. In a restart its `rows_removed` / `nnz_removed` are the cut rows that `moveCutsToPool` takes out of the model after presolve |

The stage seconds add up to `presolve_seconds`. The `DIAG cliquemerge_call` and
`DIAG probing_round` lines are written while the `enumeration` / `probing` stage is running, so
those two stage times include one log write per line (a handful of lines per presolve call).

### `DIAG reductions stage=<n> <type>=<count> ...`

Postsolve-stack entries pushed during the stage, by HiGHS reduction type. One line per stage that
pushed at least one entry, then `stage=other` (if any) and always `stage=all`. Types:
`linear_transform free_col_substitution doubleton_equation equality_row_addition
equality_row_additions singleton_row fixed_col redundant_row implied_equation forcing_row
forcing_column forcing_column_removed_row duplicate_row duplicate_column zero_obj_singleton_col
fourier_motzkin_block fourier_motzkin_obj_col`. `fixed_col` holds every column fixing, whatever
rule found it (dominated column, dual fixing, forcing row, probing). `doubleton_equation` is the
postsolve type of every substitution `y = a + b x`, not only of doubleton equations: over a
presolve call it equals `DIAG doubleton removed` + `DIAG substitutions applied` +
`simple_probing` + `dual_fixing` (holds on the 60 presolve calls of "Checks"). Rows deleted by
clique merging, by the clique table and by variable-bound aggregation have no postsolve entry and
are not in these counts; they are in `DIAG cliquemerge_call` and `DIAG varbound_aggregation`.

### `DIAG probing_round ...` (one line per `runProbing` call, printed when the call ends)

`round` (1-based), `binaries` (candidates in the sorted list), `attempted` (calls of
`HighsImplications::runProbing`), `probed` (calls that probed, HiGHS `numProbed`), `stop`,
`fixed`, `bounds_tightened`, `substitutions` (applied in `finaliseProbing`), `cliques_entry`
(clique table size when `runProbing` starts), `cliques_before` / `cliques_after` (at the start and
the end of the probing loop, before clique merging), `implications_before` / `implications`
(`HighsImplications::getNumImplications` at the start and the end of the loop), `lifted_nonzeros`
(clique-extension nonzeros added), `seconds`, `prepare_seconds`, `loop_seconds`,
`finalise_seconds`, and the work of the probing loop: `prop_sweeps`, `prop_rows`, `prop_entries`,
`activity_entries`, `probing_work`, `max_probe_work`, `max_probe_sweeps` (see "Probing work").

`stop` is one of: `contingent` (`probingContingent - numProbed < 0`), `deletions` (the early
abort on deleted columns), `clique_limit`, `implication_limit` (more than max(1e6, 2 nnz) new
cliques / implications), `table_full` (`cliquetable.isFull()`), `neighbourhood` (the clique
neighbourhood-query contingent), `time` (time limit), `finished` (every candidate was visited),
`infeasible`, `no_binaries`, `aborted` (`prepareProbing` returned infeasible or a limit).

### `DIAG probing binaries=<n> probed=<k> rounds=<r> stop=<reason> fixed=<f> substitutions=<s> cliques_before=<a> cliques_after=<b> implications=<i> seconds=<s> probing_work=<w> ...`

Sum over the rounds. `binaries` and `cliques_before` are those of the first round,
`cliques_after`, `implications` and `stop` those of the last round, `rounds` is the number of
`runProbing` calls (`stop=not_run` with `rounds=0`). After `probing_work`: `attempted`,
`bounds_tightened`, `lifted_nonzeros`, `loop_seconds`, `prop_sweeps`, `prop_rows`, `prop_entries`,
`activity_entries`, `max_probe_work`, `max_probe_sweeps` (maxima over all probes).

### `DIAG cliquemerge_call ...` and `DIAG cliquemerge calls=<c> seconds=<s> rows_changed=<r> rows_deleted=<d> nnz_delta=<z> nnz_added=<a> nnz_removed=<b> fixings=<f>`

`HighsCliqueTable::runCliqueMerging` in `finaliseProbing`, which is called by probing and by
enumeration. The `_call` line is printed per call with `stage=<enumeration|probing>`,
`cliques_before` / `cliques_after` and `rows_deleted_before_merge` (rows the clique table marked
as deleted before merging started in this stage call, for example cliques that became redundant
through fixings). In both lines: `seconds` is the time in `runCliqueMerging`; `rows_changed` is the
number of distinct rows that got clique-extension nonzeros; `rows_deleted` the rows deleted because
merging made their clique redundant; `nnz_added` the extension nonzeros; `nnz_removed` the nonzeros
of the deleted rows; `nnz_delta = nnz_added - nnz_removed`; `fixings` the columns the clique table
fixed during merging.

### `DIAG aggregator candidates=<n> substituted=<s> size2=<k> rejected_pivot=<p> rejected_fillin=<f> stopped_by_failures=<n> examined=<e> stale=<t> rejected_integer=<i>`

Sums over all aggregator calls. `candidates`: substitution opportunities in the list when the
loop starts; `examined`: entries visited before the loop ended; `substituted`: free-column
substitutions done, of which `size2` because the row or the column had two entries;
`rejected_pivot`: pivot below `presolve_pivot_threshold`; `rejected_fillin`: fill-in above
`presolve_substitution_maxfillin`; `stopped_by_failures`: calls that stopped after three fill-in
rejections in a row; `stale`: entry no longer valid (deleted, not implied free, nonzero gone);
`rejected_integer`: integer column that is not implied integral by the row.

### `DIAG substitutions applied=<n> from_implications=<n> from_cliques=<n> in_clique_substitutions=<n> in_enumeration=<n> in_probing=<n> simple_probing=<n> dual_fixing=<n>`

Column substitutions applied by `applyConflictGraphSubstitutions`: `from_implications` are
`y = a + b x` found by the two probes of `x`, `from_cliques` are `x = y` / `x = 1 - y` found by
the clique table. The `in_*` fields say in which stage they were applied (the main-loop stage, the
end of enumeration, the end of probing).

`simple_probing` and `dual_fixing` are two other rules that substitute a column by a binary and
are not part of `applied`: simple probing on a single equation (in row presolve) and the
substitution of dual fixing (`HPresolveDualFixing.cpp`). Both run inside the initial scan, the
fast loop and wherever row or column presolve is called; they are counted for the whole presolve
call, not per stage (radiationm18-12-05: 1985 by dual fixing, var-smallemery-m6j6: 865).

### `DIAG varbound_aggregation name=<enumeration|probing> calls=<c> seconds=<s> rows_removed=<r> rows_modified=<m> vars_lifted=<l> nnz_removed=<z>`

Two lines, one per stage that calls `finaliseProbing`. `aggregateVarBounds` runs at the end of
every `finaliseProbing`: for a non-binary column it merges variable-bound rows whose binaries form
a clique into one row. The merged-away rows are deleted without a postsolve entry. `rows_removed`
rows deleted, `rows_modified` rows rewritten as the aggregated constraint, `vars_lifted` binaries
put into rewritten rows, `nnz_removed` nonzeros before minus after (negative when the rewritten
rows grow), `seconds` the time in `aggregateVarBounds`. All of it is inside the stage line of the
same name, and for probing inside `finalise_seconds`.

With these lines every row an `enumeration` or `probing` stage removes is accounted for: stage
`rows_removed` = `rows_deleted` + `rows_deleted_before_merge` of its `cliquemerge_call` lines +
`rows_removed` here + the `redundant_row` entries of `DIAG reductions stage=probing` (lifting for
probing). The nonzeros it adds are not clique-extension nonzeros and are not in `lifted_nonzeros`.
Examples: rd-rplusc-21, enumeration removes 722 rows, all by this rule; uccase9, in probing it
rewrites 115 rows and adds 1729 nonzeros (0.11 s of the 0.65 s probing stage in a check run on the
shared node).

### `DIAG doubleton removed=<n> <stage>=<n> ... other=<n>`

Doubleton equations removed by `doubletonEq` (one column substituted). `removed` is the total;
then the count per stage in which the removal happened (`fast_loop` is the fast loop of the task
text, `initial_scan` the first scan).

### `DIAG parallel duplicate_rows=<n> row_additions=<n> cols_merged=<n> cols_dominated=<n> rows_removed=<n> cols_removed=<n> hash_seconds=<s> col_seconds=<s> row_seconds=<s>`

Inside `detectParallelRowsAndCols`: `duplicate_rows` exactly parallel rows deleted;
`row_additions` nearly parallel pairs handled by adding an equation to the other row;
`cols_merged` parallel columns merged; `cols_dominated` parallel columns fixed by dominance;
`rows_removed` rows deleted during the row loop, `cols_removed` columns deleted during the column
loop (the row loop also runs row presolve on short rows, so `rows_removed` includes doubleton
equations, see `DIAG doubleton`; the columns those substitute are in the stage line only);
seconds of the hashing, the column loop and the row loop. The three do not add up to the stage
seconds: freeing the hash buckets and the work arrays at the end of the call is in the stage time
only (uccase9: 6% of the stage, unitcal_7: 14%).

### `DIAG total presolve_seconds=<s> rows=<before>:<after> cols=<before>:<after> nnz=<before>:<after> integer_cols=<before>:<after> implied_integer_cols=<n> status=<status> stopped=<0|1> restart=<n>`

`presolve_seconds` is the wall time of `HPresolve::run` (it excludes reading the model and the
copy of the model into the MIP solver). `integer_cols` counts columns of type integer;
`implied_integer_cols` columns of type implied integer after presolve (they are not in
`integer_cols`). `status` is the HiGHS presolve status (`reduced`, `not_reduced`,
`reduced_to_empty`, `infeasible`, `unbounded_or_infeasible`, ...); `stopped=1` when presolve ended
on the time or the reduction limit. The sizes after are those of the presolved model HiGHS reports.

## Probing work

`HighsDomain` has no work counter, so the patch adds one (thread-local integers, one addition per
propagated row and one per bound change):

- `prop_sweeps`: passes over the queue of model rows in `HighsDomain::propagate` (all queued rows
  are evaluated on the same bounds, then the bound changes are applied; one pass is one round of
  propagation).
- `prop_rows`: rows taken from the queue.
- `prop_entries`: row entries scanned by `propagateRowUpper` / `propagateRowLower` (row length per
  non-redundant side; cut-pool rows are counted too, there are none in the root presolve).
- `activity_entries`: column entries visited to update row activities, once per bound change
  (probe, implied change and backtrack).
- `probing_work = prop_entries + activity_entries`, summed over the probing loop only.
  `max_probe_work` and `max_probe_sweeps` are the largest values of one probe (both directions).

`rounds` in `DIAG probing` is the number of `runProbing` calls. If propagation rounds are meant,
use `prop_sweeps`.

## Switches (environment variables; unset = stock)

`HIGHS_DIAG_OFF=<name>[,<name>...]`. An unknown name aborts the process. The switches act on MIP
presolve calls (root, restarts and sub-MIPs). Two reach further: `parallel_columns` /
`parallel_rows` also apply to an LP presolve call, and the implication half of
`clique_substitutions` also applies to probing during the search. Neither happens in a
presolve-only run.

| name | effect |
|---|---|
| `strengthen_inequalities` | `strengthenInequalities` is not called |
| `dominated_columns` | `dominatedColumns` is not called (both places). The dominated-column test inside column presolve stays (stock `presolve_rule_off` bit 5) |
| `parallel_columns` | the column loop of `detectParallelRowsAndCols` is skipped; parallel rows stay on |
| `parallel_rows` | the row loop is skipped. With both names the stage is not called |
| `sparsify` | same as the stock rule switch (`kPresolveRuleSparsify`) |
| `aggregator` | same as the stock rule switch |
| `fourier_motzkin` | same as the stock rule switch |
| `enumeration` | same as the stock rule switch |
| `probing` | `runProbing` is not called. `detectImpliedIntegers` and the fast loop after it still run once. The stock rule switch would also skip `detectImpliedIntegers` |
| `clique_substitutions` | no substitution is recorded during presolve probing and enumeration: the clique table does not turn a pair of complementary cliques into `x = y` / `x = 1 - y` (the new edge is stored as an ordinary clique) and probing does not record `y = a + b x` from the two probes (the bounds are tightened instead). The stage call itself stays, because in a restart it applies substitutions found by the search. `simple_probing` and `dual_fixing` substitutions are not affected |
| `implied_integers` | the `detectImpliedIntegers` sweep is not called. Column presolve still converts columns (see "Not measured") |
| `clique_merging` | `runCliqueMerging` is not called in `finaliseProbing` (extra, for the clique merging effect) |

`HIGHS_DIAG_PROBING_CONTINGENT=<n>`: initial probing contingent (stock 1000).

`HIGHS_DIAG_PROBING_NOSTOP=1`: a probing round stops only on the time limit, on infeasibility or
when every candidate was visited (`contingent`, `deletions`, `clique_limit`,
`implication_limit`, `table_full` and `neighbourhood` are off, and the candidate list is built even
when the clique table is full). The clique table can then grow without bound. The main loop still
decides with the stock rule whether another round follows.

## Not measured, and limits

- Rules inside the fast loop and the initial scan are not timed separately (empty, singleton,
  redundant and forcing rows, dominated and dual-fixed columns, singleton columns). The
  `DIAG reductions` lines give their counts by postsolve type, not their time. HiGHS's own rule log
  (`presolve_rule_logging`) and presolve clocks are LP only and were left alone.
- `clique_substitutions` as a stage is empty at the root presolve: the substitutions that probing
  finds are applied inside `finaliseProbing`, so their time and removals are part of `probing` (and
  `enumeration`); the counts are in `DIAG substitutions`.
- Implied-integer conversion happens mainly in column presolve (`convertImpliedInteger` in
  `colPresolve`, singleton columns, parallel columns), inside the fast loop. The `implied_integers`
  stage and switch cover only the sweep before probing; the conversions of the fast loop cannot be
  switched off and are not counted per stage. `implied_integer_cols` in `DIAG total` is the final
  number.
- Clique merging time is the time of `runCliqueMerging`; applying its result (row deletions and
  extension nonzeros) is in the stage time, not in `cliquemerge seconds`.
- `rows_deleted_before_merge` rows (deleted by the clique table outside merging) are only on the
  `cliquemerge_call` lines.
- The time between the stages of one probe (clique table updates, implication storage) is not
  split; `loop_seconds` is the whole probing loop. Work is counted for domain propagation only, not
  for clique table operations (`numNeighbourhoodQueries` is not printed).
- Timings on an early exit: when a stage returns infeasible or stops on a limit from inside the
  parallel row/column loops, the `DIAG parallel` seconds of that call are not added (the stage
  line is complete).
- The overhead of the counters was not measured. It is two clock reads per stage call, one
  thread-local addition per propagated row and per bound change, and one pass over the new
  postsolve entries per stage call. In the check runs the stock and the patched binary, started
  together on the shared node, logged the same presolve time to the second (rd-rplusc-21 125 s and
  124 s, var-smallemery-m6j6 19 s, triptim1 5 s); that is an observation, not a measurement.
- `clique_substitutions`, `clique_merging`, `HIGHS_DIAG_PROBING_NOSTOP` and the stage switches
  were checked by full solves on four small instances only (same optimal value as stock, see
  "Checks"). They are meant for presolve-only runs.
- LP presolve (`mipsolver == nullptr`) is accounted but prints nothing.

## Checks (dev node, 2026-11-26, functional only; the times are not measurements)

Presolve only, by the options file with `write_presolved_model_file`, stock binary
(`highs_stock_install`) against the patched binary with every switch unset:

| instance | rows after | cols after | nnz after | presolved MPS file |
|---|---|---|---|---|
| wachplan | 650 | 2288 | 12024 | identical |
| ns1830653 | 1433 | 587 | 24263 | identical |
| uccase9 | 32847 | 22474 | 256250 | identical |
| trento1 | 1192 | 7606 | 89185 | identical |

With the patch as it is now, the written presolved models are byte-identical (`cmp`) on these four
and on ns1208400, radiationm18-12-05, supportcase40, rmatr200-p5, satellites2-60-fs, unitcal_7,
sp98ar, supportcase42, triptim1, supportcase6 and var-smallemery-m6j6 (15 instances, up to 850,000
nonzeros). With the patch before the review additions (`DIAG varbound_aggregation`,
`simple_probing`, `dual_fixing`; only counters were added after it) they were also identical on
timtab1, sp150x300d, n5-3, gmu-35-40, mcsched, icir97_tension, roll3000, app1-1, seymour and
bnatt400.

`HIGHS_DIAG_OFF=probing`:

| instance | rows | cols | nnz | reason |
|---|---|---|---|---|
| wachplan | 650 | 2298 | 12074 | probing fixes 10 columns (50 nonzeros net of one clique-extension nonzero); no row is affected |
| ns1830653 | 1477 | 645 | 27652 | probing fixes 53 columns, substitutes 4, its clique merging deletes 6 rows; the later fast loop, sparsify and parallel-row calls then find less |
| uccase9 | 33378 | 22869 | 257545 | probing fixes 388 columns, substitutes 6, its clique merging deletes 55 rows and adds 1181 clique-extension nonzeros, and its variable-bound aggregation adds 1729 nonzeros |
| trento1 | 1192 | 7606 | 89185 | unchanged: probing stops after 135 probes (neighbourhood contingent) with no fixing, no substitution and an empty clique table |

Every switch was run once on ns1830653 and uccase9 with the patch as it is now (presolve only,
exit 0, sizes change or stay as the stage counts predict), also `parallel_rows,parallel_columns`
(the stage has 0 calls), `HIGHS_DIAG_PROBING_CONTINGENT=10` (ns1830653 stops after 11 probes with
`stop=contingent`; uccase9 prints the same counts as with 1000, every probe adds its bound changes
to the contingent) and
`HIGHS_DIAG_PROBING_NOSTOP=1` (`stop=finished` in every round). With the earlier patch: all twelve
names together give 1742 rows, 702 columns, 37879 nonzeros on ns1830653, and an unknown name and a
non-integer contingent abort with a message.

Consistency of the output, checked by a script on 60 presolve calls (the runs above, rd-rplusc-21,
bnatt500, fhnw-binpack4-4, a time-limited uccase9 and var-smallemery-m6j6, and a full solve of
app1-1 with three restarts): the stage lines add up to `DIAG total` in seconds, rows, columns and
nonzeros; the `DIAG reductions` lines add up to `stage=all`; the round and call lines add up to
`DIAG probing` and `DIAG cliquemerge`; every row removed in `enumeration` and `probing` and every
`doubleton_equation` postsolve entry is explained by the identities given above; two runs of the
same instance print the same lines apart from the seconds. With `time_limit` reached inside
probing the block is still printed, with `stop=time` and `stopped=1`.

Full solves (150 s limit, 4 threads) of sp150x300d, app1-1, mcsched and n5-3 with stock, the
patched binary without switches, `clique_substitutions`, `clique_merging`, `probing`,
`HIGHS_DIAG_PROBING_NOSTOP=1` and a nine-name list (`probing, enumeration,
strengthen_inequalities, dominated_columns, parallel_columns, sparsify, aggregator,
fourier_motzkin, implied_integers`): the same optimal value in every run (69, -3, 8105; mcsched
211913 as the best solution of every run, proved optimal only by the nine-name run, the others
reached the time limit with dual bounds below it). Stock and the patched binary without switches
end with the same dual bound (mcsched 201307 at the limit, n5-3 8104.368). app1-1 restarts three
times and prints four blocks (`restart=0` to `restart=3`); no sub-MIP prints. These solves ran
with earlier builds of the patch (the switch code is the same): mcsched and n5-3 before
`DIAG reductions`, `DIAG substitutions` and `prop_sweeps` were added, sp150x300d and app1-1 before
the review additions. With the patch as it is now only app1-1 without switches was solved again
(optimal, four blocks).
