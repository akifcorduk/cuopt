# HiGHS MIP presolve diagnostics (`highs_diag.patch`)

`highs_diag.patch` is a unified diff against HiGHS commit `283bfc859` (1.16.0). It makes the MIP
presolve report where its time goes and what each stage removes, and adds switches that skip
stages and rules. With every switch unset the presolve result is the stock result (checked against
stock for the patch of 2026-11-26, see the first two "Checks" sections and the "Review" section,
and against that patch for the additions of 2026-11-27). The `cliquemerge_call` and `probing_round`
lines are printed when their call ends, everything else at the end of each presolve call, through
`highsLogUser` at info level, so the lines are in the log file and on stdout. The longest line
seen is 789 characters (a `DIAG probing_round` line), below the 1024 characters HiGHS passes to a
user log callback; a longer line would be cut there, not in the log file. Every line starts with
`DIAG `, fields are `key=value` separated by one space.

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
(default 8). The baseline tool is compiled against the installed headers, so the script builds it
after the install, in `gpu_presolve_deps/build/baseline_diag` (the commands of `README.md`). That
tree belongs to the worktree it was first configured from: CMake refuses another source directory.

`NAME=<name>` (default `diag`; letters, digits, underscores) builds under another name: source copy
`gpu_presolve_deps/highs_<name>_src`, build tree `build/highs_<name>`, install
`highs_<name>_install`, tool build tree `build/baseline_<name>`. The script writes nothing outside
these four. `NAME=stock` is refused, because that install is the unpatched HiGHS. Use a new name
to build a changed patch while runs still use an earlier build:

```
cuopt exec 'NAME=diag3 bash gpu_presolve/baseline/build_highs_diag.sh'
```

The tool is then `gpu_presolve_deps/build/baseline_diag3/baseline_tool`.

The builds on the node (2026-11-27). `diag` is the first version: it has no `DIAG fixing`,
`DIAG bound_change` and `DIAG dominated_columns` lines, no bound-change and row fields on the
probing lines, and not the switches `simple_probing`, `dual_fixing_substitution`,
`implied_equations`, `implied_integer_conversion` (it aborts on these names),
`HIGHS_DIAG_REDUNDANCY_EXPLICIT` and `HIGHS_DIAG_PROBE_ROWS` (it ignores these variables). `diag2`
is the patch of 2026-11-26 (sha256 starts with `7d8c3c0b4db9`), built before the script built the
tool, so its tool tree `build/baseline_diag2` was made by hand; the stored reference runs use it.
It has no `DIAG free_col_substitution` lines and aborts on the four switch names of 2026-11-27
(`dual_fixing_single_equation`, `dual_fixing_binary_substitution`,
`dual_fixing_bound_strengthening`, `implied_row_dual_bounds`). The patch as it is now has not been
built on the node; it was built and checked on the Mac (see "Checks of the 2026-11-27 additions").

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

### `DIAG switches off=<list|none> probing_contingent=<n> probing_nostop=<0|1> redundancy_explicit=<0|1> probe_rows=<0|1>`

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
rule found it (dominated column, dual fixing, forcing row, probing); `DIAG fixing` splits the
`stage=all` count by rule. `doubleton_equation` is the
postsolve type of every substitution `y = a + b x`, not only of doubleton equations: over a
presolve call it equals `DIAG doubleton removed` + `DIAG substitutions applied` +
`simple_probing` + `dual_fixing` (holds on the presolve calls of both "Checks" sections). Rows
deleted by clique merging, by the clique table and by variable-bound aggregation have no postsolve
entry and are not in these counts; they are in `DIAG cliquemerge_call` and
`DIAG varbound_aggregation`.

### `DIAG probing_round ...` (one line per `runProbing` call, printed when the call ends)

`round` (1-based), `binaries` (candidates in the sorted list), `attempted` (calls of
`HighsImplications::runProbing`), `probed` (calls that probed, HiGHS `numProbed`), `stop`,
`fixed`, `bounds_tightened`, `substitutions` (applied in `finaliseProbing`), `cliques_entry`
(clique table size when `runProbing` starts), `cliques_before` / `cliques_after` (at the start and
the end of the probing loop, before clique merging), `implications_before` / `implications`
(`HighsImplications::getNumImplications` at the start and the end of the loop), `lifted_nonzeros`
(clique-extension nonzeros added), `seconds`, `prepare_seconds`, `loop_seconds`,
`finalise_seconds`, and the work of the probing loop: `prop_sweeps`, `prop_rows`, `prop_entries`,
`activity_entries`, `probing_work`, `max_probe_work`, `max_probe_sweeps`, `bound_changes`,
`rows_touched`, `rows_touched_entries`, `max_rows_touched`, `max_rows_touched_entries`,
`probe_directions`, `dir_rows_touched`, `dir_rows_touched_entries`, `max_dir_rows_touched`,
`max_dir_rows_touched_entries`, `probe_reductions` (see "Probing work").

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
`activity_entries`, `max_probe_work`, `max_probe_sweeps` (maxima over all probes), then the sums
over the rounds of `bound_changes`, `rows_touched`, `rows_touched_entries`, `probe_directions`,
`dir_rows_touched`, `dir_rows_touched_entries`, `probe_reductions` and the maxima over the rounds
of `max_rows_touched`, `max_rows_touched_entries`, `max_dir_rows_touched`,
`max_dir_rows_touched_entries`.

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

### `DIAG free_col_substitution name=<singleton_col|aggregator> count=<n> equation=<n> inequality_own=<n> inequality_other=<n> upper_side=<n> ranged=<n> own_zero_cost=<n> integer=<n>`

Two lines, always both. Free column substitutions (`substituteFreeCol`, postsolve type
`free_col_substitution`): an implied free column is solved out of a row whose dual is implied
free, the row is deleted and the column is replaced in its other rows and in the objective. Every
substitution is counted once, where the postsolve entry is pushed. `name=aggregator` holds those
made inside the `aggregator` stage (`count` equals `substituted` of `DIAG aggregator`),
`name=singleton_col` all others; these come from singleton-column presolve (`singletonCol`, stock
rule bit 8), the only other caller. The two `count` fields add up to `free_col_substitution` of
`DIAG reductions stage=all`.

The pivot row is of one of three kinds, which add up to `count`:

- `equation`: the row is an equation.
- `inequality_own`: the row is an inequality that HiGHS treats as tight at one side, because the
  implied bound on its row dual fixes the sign, and that bound was derived from the pivot column
  itself (`rowDualUpperSource` or `rowDualLowerSource` of the row is the pivot column).
- `inequality_other`: the same, with the bound derived from another column.

Of the inequality pivots: `upper_side` are tight at the upper side (postsolve row type `kLeq`; the
others at the lower side, `kGeq`; when both sides qualify HiGHS takes the upper one, see
`dualImpliedFreeGetRhsAndRowType`), `ranged` have two finite sides, and `own_zero_cost` are those
of `inequality_own` whose pivot column has a cost of exactly 0. `integer` is the number of pivot
columns of type integer, over all three kinds. With the switch `implied_row_dual_bounds` every
pivot is an equation.

### `DIAG substitutions applied=<n> from_implications=<n> from_cliques=<n> in_clique_substitutions=<n> in_enumeration=<n> in_probing=<n> simple_probing=<n> dual_fixing=<n>`

Column substitutions applied by `applyConflictGraphSubstitutions`: `from_implications` are
`y = a + b x` found by the two probes of `x`, `from_cliques` are `x = y` / `x = 1 - y` found by
the clique table. The `in_*` fields say in which stage they were applied (the main-loop stage, the
end of enumeration, the end of probing).

`simple_probing` and `dual_fixing` are two other rules that substitute a column by a binary and
are not part of `applied`: simple probing on a single equation (in row presolve) and the
substitution of dual fixing (`HPresolveDualFixing.cpp`). Both run inside the initial scan, the
fast loop and wherever row or column presolve is called; they are counted for the whole presolve
call, not per stage (radiationm18-12-05: 1985 by dual fixing, var-smallemery-m6j6: 865). The
switches `simple_probing` and `dual_fixing_binary_substitution` turn them off (the second also
goes with `dual_fixing_substitution`, which switches off more).

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

### `DIAG fixing name=<rule> count=<n> nnz=<z>` and `DIAG fixing name=total count=<n> nnz=<z> fixed_col=<n>`

Column fixings of the presolve call by the rule that made them: one line per rule that fixed at
least one column, then the `total` line (always). `count` is the number of columns, `nnz` the
entries those columns had when they were fixed. Every fixing pushes one `fixed_col` postsolve
entry, so the counts add up to `fixed_col` of `DIAG reductions stage=all`; the `total` line repeats
that number as `fixed_col` and the two must be equal (checked, see "Checks of the 2026-11-26
additions").

Rules that fix a column themselves (`fixColToLower`, `fixColToUpper`, `fixColToZero`, or a direct
`removeFixedCol`):

| name | who fixes |
|---|---|
| `empty_col` | `emptyCol`: a column without entries, at the bound its cost prefers |
| `forcing_row` | row presolve, forcing row: every column of the row at the bound that gives the forcing activity |
| `dominated_col` | `detectDominatedCol` in column presolve: the bound on the reduced cost is strictly positive or negative (stock rule bit 5) |
| `weakly_dominated_col` | same function, reduced-cost bound of zero and a finite column bound (stock rule bit 23) |
| `dual_fixing_lock` | `dualFixing`: the column has no down-lock or no up-lock |
| `dual_fixing_single_equation` | `dualFixing`, single-equation extension: columns of the equation that is the only lock |
| `dual_fixing_bound` | `dualFixing`, bound strengthening: the worst-case bound equals the opposite column bound |
| `singleton_row_bounds_equal` | `singletonRow`: the bound from the row and the other column bound are equal |
| `simple_probing` | simple probing on an equation: a column of that row that was already fixed (the other columns are substituted, see `DIAG substitutions`) |
| `equation_gcd` | row presolve of an all-integer equation: the substitution `x = d z + b` leaves one value for `z` |
| `col_stuffing` | `singletonColStuffing` (stock rule bit 19) |
| `parallel_columns` | `detectParallelRowsAndCols`: a parallel column fixed by dominance (`cols_dominated` of `DIAG parallel`) |
| `dominated_columns_pairwise` | `dominatedColumns`, the pairwise test (`fixed` of `DIAG dominated_columns`) |
| `clique_row_normalisation` | `normaliseCliqueRows` before the first probing or enumeration: a binary whose coefficient is above the right-hand side |
| `enumeration`, `probing` | `finaliseProbing` at the end of that stage: every column that is fixed in the global domain, whoever fixed it there (a probe, the domain propagation and clique extraction of `prepareProbing`, clique merging). For `probing` the sum over the rounds is `fixed` of `DIAG probing` |
| `other` | a fix call that names no rule. The patch names one at every call, so a line with this name means a call site was missed |

`bounds_equal_<cause>`: column presolve (`colPresolve`) removes a column whose bounds are equal
within the tolerance of `checkColBounds`. This is where a column ends that another rule fixed by
changing its bounds. `<cause>` is the rule that made the last bound change of the column:

| cause | who changed the bound |
|---|---|
| `input` | nobody in this presolve call: the column is fixed in the model presolve was given, after the rounding of integer bounds at the start (in a restart also the columns the search fixed) |
| `implied_bound` | `updateColImpliedBounds`: the bound a row implies on an integer or implied-integer column is written into the column bound (bound propagation by one row; for a cut row in a restart, on any column) |
| `binary_row_test` | row presolve of a row with integer columns: a binary whose other value makes the row infeasible is moved to that bound (HiGHS cites Suhl and Szymanski) |
| `doubleton` | `doubletonEq`: the bounds of the substituted column carried over to the column that stays |
| `singleton_row` | `singletonRow` tightened a bound (bounds equal within tolerance; exactly equal bounds are `singleton_row_bounds_equal`) |
| `dual_fixing` | `dualFixing`, bound strengthening on an integer column |
| `dominated_columns_pairwise` | `dominatedColumns`, predictive bound analysis |
| `implied_integer` | `convertImpliedInteger`: bounds rounded and implied bounds taken over at the conversion |
| `transform` | `transformColumn`: bounds of an integer column rounded after a shift or scaling |
| `parallel_columns` | merge of two parallel columns |
| `probing_prepare` | `prepareProbing`: implied bounds written into the column bounds before probing or enumeration |
| `probing` | `finaliseProbing`: a tighter bound of the global domain copied to the model at the end of probing or enumeration. A column that is fixed in the domain is removed at once (fixing rule `probing` or `enumeration`), so `bounds_equal_probing` needs bounds that are equal only within tolerance |
| `clique_table` | the presolve clique table fixed the column because another column was fixed or substituted (`updateCliqueTableFixedCol`, `updateCliqueTableSubstituteCol`) |
| `equation_gcd` | the gcd substitution of an all-integer equation that leaves one value; the column is removed at once (fixing rule `equation_gcd`) |
| `other` | a bound change that names no cause (only `checkRedundantBounds`, which runs in LP presolve) |

The cause is kept per column (one byte, carried through `shrinkProblem`). It is the last bound
change, not the one that made the bounds equal: if rule A sets the lower bound and rule B later
moves the upper bound onto it, the column counts for B.

Not counted: columns removed by a substitution (`DIAG substitutions`, `DIAG doubleton`,
`DIAG aggregator`), by a parallel-column merge, by a forcing column or as a zero-cost singleton
(their own postsolve types in `DIAG reductions`). The initial sweep of LP presolve
(`HPresolveInitialSweep`) also pushes `fixed_col` entries without a count; it does not run in a MIP
presolve.

### `DIAG bound_change name=<cause> count=<n>` and `DIAG bound_change name=total count=<n>`

Column bound changes of the presolve call by the rule that made them: one count for every call of
`changeColLower` / `changeColUpper` that gave the bound a new value. One line per cause that
changed a bound, then the `total` line (always). The causes are those of the second table under
`DIAG fixing` (without `input`). These lines show the rules that tighten bounds without fixing a
column; such a change has no postsolve entry and appears nowhere else. A column fixed through
`fixColToLower` / `fixColToUpper` gets no bound change and is not in these counts, and the scaling
or shift of a column by `transformColumn` is not a bound change either (only the rounding of an
integer column after it is, under `transform`).

### `DIAG dominated_columns pair_tests=<n> signature_rejects=<n> fixed=<n> bounds=<n> seconds=<s> calls=<c> type_rejects=<n> cost_rejects=<n> column_scans=<n> scan_entries=<n> dominations=<n>`

The pairwise dominated-column test (`dominatedColumns`, stage `dominated_columns`), summed over its
calls (`calls`, at most two per presolve call). `pair_tests` is HiGHS's own count of domination
checks (`numDomChecks`): one per call of `checkDomination`, that is per ordered pair of columns and
sign combination that is tested for "j dominates k". Every test ends in exactly one of four ways,
so `type_rejects + signature_rejects + cost_rejects + column_scans = pair_tests`:

- `type_rejects`: j is integer and k continuous;
- `signature_rejects`: the row signatures of the two columns (two 32-bit masks per column, one bit
  per hashed row, by coefficient sign and finite row side) rule domination out;
- `cost_rejects`: the cost of j is above the cost of k;
- `column_scans`: the two columns are compared entry by entry. `scan_entries` is the number of
  column entries these scans visited (each visit is one coefficient lookup in the other column),
  `dominations` the number of scans that found domination.

`fixed`: columns fixed (`numFixedCols`, the fixing rule `dominated_columns_pairwise`); `bounds`:
bounds tightened by predictive bound analysis; `seconds`: the time of the stage, the same number as
on its `DIAG stage` line. All zero with `calls=0` when the stage is switched off.

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

Bound changes and rows per probe. A probe is one call of `HighsImplications::runProbing` from the
presolve probing loop (one binary). It has up to two directions (`computeImplications` with the
binary at 1, then at 0); a direction is counted from the fixing of the binary to the start of the
backtrack that undoes it. A probe that finds the first direction infeasible, or that is refused
(binary already probed or substituted), has fewer. What a probe changes in the global domain
outside its directions is not in `bound_changes` and not in the row fields: the fixing of the
binary after an infeasible direction, the bounds that hold in both directions, and the propagation
of both at the start of the next direction.

- `bound_changes`: bound changes `HighsDomain` applied inside the directions: the fixing of the
  binary and every change derived from it before the backtrack (row propagation, clique table and
  stored implications, dual-fix probing). Undoing them is not counted. Counted where
  `HighsDomain::changeBound` puts a change on its stack.
- `rows_touched`: per probe, the number of distinct model rows whose activity `HighsDomain` updated
  inside its directions (a row touched in both directions counts once); summed over the probes.
  `rows_touched_entries`: the summed length of those rows (entries of the row in the row-wise matrix
  the domain propagates on). `max_rows_touched`, `max_rows_touched_entries`: the largest values of
  one probe.
- `probe_directions`: number of directions. `dir_rows_touched`, `dir_rows_touched_entries`,
  `max_dir_rows_touched`, `max_dir_rows_touched_entries`: the same four counts with "distinct"
  taken per direction, so a row touched in both directions of a probe counts twice in the sums.
  Both readings of "per probe" are printed; `rows_touched <= dir_rows_touched`, and the difference
  is at least the rows of the probed column, which both directions touch.
- `probe_reductions`: the sum of the reduction counts `HighsImplications::runProbing` returns
  (substitutions it recorded and bound changes it made to the global domain after comparing the two
  directions). This is the number HiGHS adds to the probing contingent. It is not part of
  `bound_changes`.

A row is touched when a bound change of one of its columns updates its minimum or maximum
activity. The backtrack updates the same rows again and is not looked at: `HighsDomain` undoes
only changes whose activity update it made, so the backtrack touches no new row (a build that
also marked the backtrack printed the same counts, see "Review of the 2026-11-26 additions").
When a bound change makes a row infeasible, HiGHS stops updating the remaining rows of that
column; those are not counted (`activity_entries` counts the whole column in that case). Cut-pool
and conflict rows are not model rows and are not counted. The counters work with two thread-local
markers per row and are active only inside the presolve probing loop; probing during the search
does not count. Marking the rows is a second pass over the column entries of every bound change a
direction applies. With `HIGHS_DIAG_PROBE_ROWS=0` that pass is not made and the eight row fields
are 0; `bound_changes`, `probe_directions` and `probe_reductions` are counted as before.

## Switches (environment variables; unset = stock)

`HIGHS_DIAG_OFF=<name>[,<name>...]`. An unknown name aborts the process. The switches act on MIP
presolve calls (root, restarts and sub-MIPs). Two reach further: `parallel_columns` /
`parallel_rows` also apply to an LP presolve call, and the implication half of
`clique_substitutions` also applies to probing during the search. Neither happens in a
presolve-only run. With a switch unset its code path is the stock one.

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
| `implied_integers` | the `detectImpliedIntegers` sweep is not called. Column presolve still converts columns (`implied_integer_conversion` switches every conversion off) |
| `clique_merging` | `runCliqueMerging` is not called in `finaliseProbing` (extra, for the clique merging effect) |
| `simple_probing` | row presolve does not apply simple probing on a single equation (see below) |
| `dual_fixing_substitution` | `dualFixing` keeps only the fixing by locks: the next three names at once (see below) |
| `dual_fixing_single_equation` | `dualFixing` does not apply the extension for single equations (see below) |
| `dual_fixing_binary_substitution` | `dualFixing` does not substitute a single-lock column by a binary (see below) |
| `dual_fixing_bound_strengthening` | `dualFixing` does not strengthen bounds (see below) |
| `implied_equations` | an inequality is never treated as an equation because of the bounds on its row dual (see below) |
| `implied_integer_conversion` | no continuous column is converted to an implied integer (see below) |
| `implied_row_dual_bounds` | no implied bound on a row dual is ever recorded (see below) |

The names from `simple_probing` on switch rules that sit inside HiGHS's row and column presolve, so
they act in the initial scan, in the fast loop and wherever else row or column presolve is called.
They act on MIP presolve calls only (an LP presolve call is unchanged). `DIAG switches off=` prints
the names that were given, in a fixed order (that of the table, except that the four names of
2026-11-27 come last); `dual_fixing_substitution` is not replaced by its three parts there, and
the three parts are not replaced by it.

`simple_probing`. Stock, in `rowPresolve`: for an equation with integer columns whose minimum plus
maximum activity (sums over implied bounds) equals twice the right-hand side, and that holds a
binary with the matching coefficient, every other column of the row is fixed by each value of the
binary. Stock then substitutes each of them by the binary (`simple_probing` in
`DIAG substitutions`), removes the columns of the row that are already fixed (fixing rule
`simple_probing`) and deletes the row. With the switch this block is skipped and the row goes on
through the rest of row presolve (integer scaling and gcd substitution of equations, forcing row,
implied bounds). Nothing else is switched off.

`dual_fixing_substitution` and its three parts. `HPresolve::dualFixing`
(`HPresolveDualFixing.cpp`) is called by column presolve for every column and for singleton columns
(the stock rule bit 17 skips the whole function). It counts the down-locks and up-locks of the
column over the finite row sides and the sign of its cost.

- Stays with all four names: fixing by locks. A column without down-lock is fixed at its lower
  bound, a column without up-lock at its upper bound (with a cost within the dual tolerance only
  when that bound is finite; an infinite bound with a cost beyond the tolerance ends presolve as
  unbounded or infeasible, as in stock). Fixing rule `dual_fixing_lock`.
- For a column that has locks in both directions stock does three things: (a) or (b) when the
  column has a single down-lock or a single up-lock, then (c) if the column is still there.
  `dual_fixing_substitution` skips this whole branch, as it did before 2026-11-27 (its statement is
  unchanged). Each of the three has a switch of its own since then:
  (a) `dual_fixing_single_equation`. The extension for single equations (Achterberg et al., section
  6.1), lambda `handleSingleEquation`: when the single lock is an equation that was not looked at
  since it last changed, columns of that equation are fixed at a bound (fixing rule
  `dual_fixing_single_equation`), and the equation is marked as looked at
  (`singleEquationChecked`). With the switch neither happens: no equation is ever marked, so the
  work stock does to clear the marks (`markChangedCol` walks the rows of a changed column while any
  mark is set) goes as well. That work is most of the time of the extension: in the runs of the
  checks section rd-rplusc-21 with every cheap rule on takes 137 s on the Mac, 0.5 s with this
  switch, and returns the same model. A column whose single lock is an equation then goes to (b)
  at every visit; stock sends it there only while the equation is marked. (b) can apply to an
  equation only if the column's coefficient times its bound range is at most twice the feasibility
  tolerance, and no substitution count changed with this switch in the runs of the checks section.
  (b) `dual_fixing_binary_substitution`. The substitution, lambda `substituteCol`: the column has
  finite bounds `[l, u]`, its single down-lock (up-lock) is a row with an integer column and at
  most 1000 entries, and a binary `x` of that row makes the row redundant at 0 and forces the
  column to `u` (`l`) at 1. The column is then replaced by `l + (u - l) x` (`u - (u - l) x`);
  `dual_fixing` in `DIAG substitutions`. With the switch the lambda is not called.
  (c) `dual_fixing_bound_strengthening`. The block after the comment "try to strengthen bounds",
  with the lambda `hasTighterBound`: the upper (lower) bound is moved to the worst-case lower
  (upper) bound the rows of the column can ask for. When that reaches the opposite bound the column
  is fixed (fixing rule `dual_fixing_bound`); otherwise the new bound is set on integer columns
  only (bound change `dual_fixing`). With the switch `dualFixing` returns before this block.
- The three names together give the presolve of `dual_fixing_substitution`: what they leave of the
  branch only reads the locks of the column and changes nothing. On the 240 benchmark instances
  the two settings print the same DIAG lines and return the same presolved model (see "Checks of
  the 2026-11-27 additions").
- One part off changes what the other two find, because they work on the same columns one after
  the other and a column that one part no longer removes is still there for the next. Seen on the
  240 instances with every cheap rule on: without (a) bound strengthening fixes columns that (a)
  fixes otherwise (irish-electricity 436 to 856 fixings); without (b) it fixes columns that (b)
  substitutes otherwise (radiationm40-10-02 10777 to 12852); and on radiationm18-12-05 and
  radiationm40-10-02 (b) fires only when (c) is on (1985 and 8425 substitutions, none without
  (c); why was not traced). The counts and the columns behind them are in the checks section.
- Not touched: the dual-fix propagation inside probing (`kPresolveRuleDualFixProbing`, part of the
  `probing` stage), and the dominated-column tests of column presolve (stock rule bits 5 and 23).

`implied_equations`. `isImpliedEquationAtLower` and `isImpliedEquationAtUpper` return false. Stock
calls an inequality an implied equation at its lower (upper) side when the implied lower (upper)
bound on its row dual is strictly positive (negative). This switches off

- the rule itself in `rowPresolve`: the other side of the row is set equal and a postsolve entry
  `implied_equation` is pushed (so `implied_equation=0` in `DIAG reductions`);
- three places that use the same test without changing the row: the side a row uses for the
  implied bounds of its columns (`checkUpdateColImpliedBounds`), the test whether an integer column
  is implied integral, used by the aggregator and by free column substitution
  (`isImpliedIntegral`), and the implied-integer test of continuous columns (`isImpliedInteger`).

The implied bounds on the row duals are still computed and still used by everything else: rows that
are dual implied free for substitutions (`isDualImpliedFree`), the dominated-column tests of column
presolve.

`implied_integer_conversion`. `convertImpliedInteger` converts nothing. That covers every place a
continuous column becomes implied integer: column presolve (`colPresolve`), singleton-column
presolve (`singletonCol`), the check after a parallel-column merge, and the sweep before probing
(the sweep is still called and does nothing; `implied_integers` remains the switch that skips the
sweep alone). Also off is the one column transform that exists for this conversion: in row presolve
of an equation with exactly one continuous column and otherwise integer columns, stock scales the
continuous column (`x = s z`, postsolve `linear_transform`), converts `z` and scales the row to
integer coefficients. The bound rounding that a conversion does goes with it. Not touched:
transforms of integer columns (the shift of a binary to lower bound 0, the gcd substitution in
all-integer equations) and `isImpliedIntegral`. `implied_integer_cols` in `DIAG total` is then the
number of implied integer columns the input model already had (0 at the root presolve of a model
read from an MPS file).

`implied_row_dual_bounds`. `changeImplRowDualUpper` and `changeImplRowDualLower` return at once,
so the implied bounds on the row duals (`implRowDualLower`, `implRowDualUpper`) keep their start
values, minus and plus infinity, and have no source column, for the whole presolve call. These two
functions are the only writers of a new value. Stock records a bound in two places: at the end of
column presolve and in singleton-column presolve, from the dual constraint of a column that is not
integer (`updateRowDualImpliedBounds`), and in `colPresolve`, a zero bound from a column whose
bound is implied by one row (lambda `modifyImpliedRowDualBound`). The explicit bounds on the row
duals (the sign of the dual of a one-sided row) stay. What goes with the implied bounds:

- an inequality is never dual implied free (`isDualImpliedFree` holds for equations only), so the
  free column substitution of singleton columns and the aggregator pivot on equations only
  (`inequality_own` and `inequality_other` of `DIAG free_col_substitution` are 0);
- everything `implied_equations` switches off (no bound on a row dual is ever strictly positive or
  negative), and the dual infeasibility test of row presolve that reads the same bounds;
- the part of the dominated-column tests of column presolve (stock rule bits 5 and 23) and of the
  forcing column (bit 7) that comes from implied bounds: the bounds on a reduced cost are then
  sums over the explicit bounds of the row duals.

Dual fixing reads no row dual and is not affected directly.

`HIGHS_DIAG_PROBING_CONTINGENT=<n>`: initial probing contingent (stock 1000).

`HIGHS_DIAG_PROBING_NOSTOP=1`: a probing round stops only on the time limit, on infeasibility or
when every candidate was visited (`contingent`, `deletions`, `clique_limit`,
`implication_limit`, `table_full` and `neighbourhood` are off, and the candidate list is built even
when the clique table is full). The clique table can then grow without bound. The main loop still
decides with the stock rule whether another round follows.

`HIGHS_DIAG_PROBE_ROWS=0`: the probing loop does not mark the rows a probe touches, so
`rows_touched`, `rows_touched_entries`, `max_rows_touched`, `max_rows_touched_entries` and the four
`dir_` fields are 0 on the probing lines (see "Probing work"). It changes no result. Use it for
runs whose probing time is to be read: the cost of the marking was not measured.

`HIGHS_DIAG_REDUNDANCY_EXPLICIT=1` (MIP presolve calls only): `HPresolve::isRedundant(row)` compares
the row sides with the activity sums over the explicit column bounds (`getSumLowerOrig`,
`getSumUpperOrig`). Stock uses the sums over implied bounds (`getSumLower`, `getSumUpper`), in
which a column counts with the bound another row implies for it when that is tighter than its own
bound. The function has three users and all three change:

1. the redundant-row test of row presolve (three calls in `rowPresolve`: before and after the
   test on binaries, and after the integer rounding of an inequality; postsolve type
   `redundant_row`);
2. clique row normalisation before the first probing or enumeration, which skips redundant rows;
3. the single-equation extension of dual fixing, which skips redundant rows when it looks at the
   other rows of a column.

Tests that do not change: the forcing-row test (it already uses the explicit sums, and in addition
requires every column bound it uses to be at least as tight as the implied bound); the primal
infeasibility test of a row, simple probing and coefficient strengthening (implied sums, as in
stock); the redundancy of one side in clique row normalisation (implied sums); a singleton row
whose bound does not tighten its column (`redundant_row` from `singletonRow`); the redundancy
`HighsDomain` finds during probing (its activities are over the domain's own bounds). In a MIP
presolve HiGHS writes the bound a row implies on an integer column into the column bound, so the
two kinds of sums are expected to differ through continuous columns only (not checked).

The test on explicit sums is the weaker one: a sum over implied bounds is never wider than the sum
over explicit bounds, so in the same state of the model every row that is redundant on explicit
sums is redundant for stock too. What changes is when a row goes: a row that stock removes at once
stays until the column bounds themselves make it redundant, and while it is in the model the other
rules use it (singleton rows, the binary row test, implied bounds, locks). The presolved model can
therefore come out larger or smaller (unitcal_7: smaller, see "Review of the 2026-11-26
additions"). User 2 is not reached when probing and enumeration are switched off, user 3 not with
`dual_fixing_substitution` or `dual_fixing_single_equation`, so in the Stage 1 setting with the
four switches of 2026-11-26 only the row presolve test is left.

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
  stage and switch cover only the sweep before probing; `implied_integer_conversion` switches off
  all conversions. They are not counted per stage. `implied_integer_cols` in `DIAG total` is the
  final number.
- `DIAG fixing`, `DIAG bound_change` and `DIAG dominated_columns` are per presolve call, not per
  stage.
- `simple_probing`, `dual_fixing_substitution`, `implied_equations`, `implied_integer_conversion`
  and `HIGHS_DIAG_REDUNDANCY_EXPLICIT` were checked by presolve-only runs and by full solves of
  three small instances (see "Checks of the 2026-11-26 additions") and of eleven more in the
  review. They are meant for presolve-only runs.
- The four switches and the `DIAG free_col_substitution` lines of 2026-11-27 were checked on the
  Mac only, with a static library built without CMake (see "Checks of the 2026-11-27 additions").
  The CMake build of the patch as it is now has not been run, neither on the Mac nor on the node,
  and no LP presolve call was run with the new switches set (they test `mipsolver` like the
  others). They are meant for presolve-only runs.
- The source classification of `DIAG free_col_substitution` reads the column HiGHS stores as the
  source of a row-dual bound at the moment of the substitution. A bound that another column set
  first and that the pivot column could have set as well counts as `inequality_other`.
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
  124 s, var-smallemery-m6j6 19 s, triptim1 5 s); that is an observation, not a measurement. It was
  made with the first version. The additions cost one more thread-local test per bound change of
  `HighsDomain`, inside the presolve probing loop one marker test per column entry of every bound
  change a direction applies (a second pass over the entries the activity update has just visited,
  not made in the backtrack; `HIGHS_DIAG_PROBE_ROWS=0` removes it), one byte write per column bound
  change of presolve, and a few additions per fixed column and per pair test. Their effect on the
  presolve time was not measured; the probing stage is where it would show. With probing switched
  off (the Stage 1 setting) the marking never runs.
- `clique_substitutions`, `clique_merging`, `HIGHS_DIAG_PROBING_NOSTOP` and the stage switches
  were checked by full solves on four small instances only (same optimal value as stock, see
  "Checks of the first version"). They are meant for presolve-only runs.
- LP presolve (`mipsolver == nullptr`) is accounted but prints nothing.

## Checks of the first version (dev node, 2026-11-26, functional only; the times are not measurements)

These checks were made with the patch before the additions of the next section (the `diag`
install).

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

## Checks of the 2026-11-26 additions (dev node, `diag2` build, functional only)

The patch before the two edits of the review (next section; sha256 of that file starts with
`68b0a8e6e0dc`), built with `NAME=diag2`, and the baseline tool built against it in
`build/baseline_diag2`. The runs are in `gpu_presolve_deps/build/diag2_check`: `check.sh` runs
them, `summ.py` reads the logs, `summary4.txt` is its output. At most four runs at a time, on the
node while a measurement was running there, so no time in these logs is a measurement. The review
replaced this build by one of the patch as it is now and repeated the checks.

No switch set. Presolve only by the options file (`write_presolved_model_file`, `threads = 1`),
stock binary against the `diag2` binary: the presolved MPS files are byte-identical (`cmp`) on 23
instances. These are the 21 main testbed instances below 1e6 nonzeros (istanbul-no-cutoff,
ns1208400, wachplan, trento1, radiationm18-12-05, ns1830653, supportcase40, rmatr200-p5,
satellites2-60-fs, unitcal_7, brazil3, piperout-d27, uccase9, sorrell3, sp98ar, supportcase42,
triptim1, supportcase6, neos-4306827-ravan, var-smallemery-m6j6, rd-rplusc-21) and
fhnw-binpack4-4 and bnatt500 of the status list. On neos-4306827-ravan and satellites2-60-fs both
binaries end with exit code 1: the MPS writer replaces blank names of rows that presolve added,
and the `highs` binary returns that warning as its exit code. The files are written and identical.

Each new switch alone (presolve only; rows, columns, nonzeros after presolve; the counter is the
one the switch is meant to silence):

| instance | switch | without | with | counter |
|---|---|---|---|---|
| brazil3 | `simple_probing` | 2777, 6740, 42182 | 3218, 7272, 55890 | `simple_probing` 13782 to 0 (`dual_fixing` rises from 0 to 16) |
| neos-4306827-ravan | `dual_fixing_substitution` | 77976, 43682, 218370 | 78110, 43766, 218735 | `dual_fixing` 51601 to 0; fixing `dual_fixing_bound` 13 to none |
| uccase9 | `implied_equations` | 32847, 22474, 256250 | same | `implied_equation` 0 to 0 |
| supportcase42 | `implied_equations` | 18157, 19183, 435089 | same | `implied_equation` 0 to 0 |
| unitcal_7 | `implied_equations` | 36202, 19726, 97142 | same | `implied_equation` 1 to 0 |
| trento1 | `implied_integer_conversion` | 1192, 7606, 89185 | same | `implied_integer_cols` 1178 to 0 |

Every run of the table ends with exit code 0 through the baseline tool (unitcal_7 was run with the
`highs` binary only, exit code 0). Through the `highs` binary neos-4306827-ravan ends with 1, with
and without the switch, for the reason given above. With the full presolve uccase9 and
supportcase42 have no implied equation, so the switch has nothing to silence there; they have
them in the Stage 1 setting below, and unitcal_7 was added because it has one with the full
presolve. All four switches together on the five instances (presolve only, no other switch): exit
code as without switches, all four counters 0; brazil3 3123, 7199, 48464; neos-4306827-ravan 78110,
43766, 218735; uccase9 33372, 22857, 257565; supportcase42 18437, 19463, 435649; trento1 1192,
7606, 89185.

Stage 1 setting: baseline tool with `--presolve-rule-off 9175456` and
`HIGHS_DIAG_OFF=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns,parallel_columns`
(the reference `diag_stage1b`), then with one new switch added, then with all four. Exit code 0 in
all 30 runs. Rows, columns, nonzeros after presolve:

| instance | Stage 1 set | + `simple_probing` | + `dual_fixing_substitution` | + `implied_equations` | + `implied_integer_conversion` | + all four |
|---|---|---|---|---|---|---|
| brazil3 | 3275, 6890, 40518 | 9315, 17610, 75082 | same as Stage 1 set | same | same | 9315, 17610, 75082 |
| neos-4306827-ravan | 99998, 62968, 268132 | same | 185534, 119992, 467716 | same | same | 185534, 119992, 467716 |
| uccase9 | 33495, 22934, 258515 | same | same | 33503, 22942, 258531 | same | 33503, 22942, 258531 |
| supportcase42 | 18159, 19186, 435093 | same | 18439, 19466, 435653 (not reduced) | 18299, 19326, 435373 | 18299, 19326, 435373 | 18439, 19466, 435653 (not reduced) |
| trento1 | 1264, 7678, 93396 | same | same | same | same | 1264, 7678, 93396 |

Counters in these runs: brazil3 `simple_probing` 13782 to 0; neos-4306827-ravan `dual_fixing` 48761
to 0 and fixing `dual_fixing_lock` 39806 to 1106, `dual_fixing_bound` 360 to none; uccase9
`implied_equation` 8 to 0; supportcase42 `implied_equation` 140 to 0 with `implied_equations`, with
`implied_integer_conversion` and with `dual_fixing_substitution` (the last also removes the 140
bound changes `DIAG bound_change name=dual_fixing`, after which presolve finds nothing);
`implied_integer_cols` to 0 with `implied_integer_conversion` on all five (trento1 1248, supportcase42
1773, uccase9 1521, neos-4306827-ravan 4, brazil3 0). The sizes of the "Stage 1 set" column equal
those of the `diag_stage1b` results of the `diag` install.

The "all four" setting over the dev testbed with the runner (not run yet; the tool path is the
`diag2` build until the `diag` install is rebuilt):

```
S1=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify
S1=$S1,fourier_motzkin,strengthen_inequalities,dominated_columns,parallel_columns
NEW=simple_probing,dual_fixing_substitution,implied_equations,implied_integer_conversion
python3 gpu_presolve/baseline/run_baseline.py --set dev --variant diag_stage1c --jobs 1 \
    --tool /home/scratch.acoerduek_sw/gpu_presolve_deps/build/baseline_diag2/baseline_tool \
    --env "HIGHS_DIAG_OFF=$S1,$NEW" --presolve-rule-off 9175456 --skip-done
```

`HIGHS_DIAG_REDUNDANCY_EXPLICIT=1`, no other switch, presolve only with the `highs` binary;
`redundant_row` entries of `DIAG reductions stage=all` without and with the variable:

| instance | without | with | presolved model |
|---|---|---|---|
| brazil3 | 3268 | 3268 | same |
| istanbul-no-cutoff | 1242 | 1242 | same |
| radiationm18-12-05 | 8685 | 8685 | same |
| supportcase40 | 23188 | 23188 | same |
| uccase9 | 6931 | 6931 | same |
| neos-4306827-ravan | 167466 | 167448 | same sizes |
| unitcal_7 | 5350 | 5357 | 36175, 19716, 97071 instead of 36202, 19726, 97142 |

Exit code 0 on six, 1 on neos-4306827-ravan (as without the variable). On unitcal_7 the model with
the variable is smaller by 27 rows and 10 columns; the review section gives the reason. In the
Stage 1 setting with all four switches the variable changes nothing on the five instances of the
table above (five more runs with the baseline tool, exit code 0, same sizes and same fixing and
reduction counts).

Consistency of the output, checked by `summ.py` on 115 presolve calls (the runs above, a second
run without switches of six instances, two runs with `HIGHS_DIAG_PROBE_ROWS=0` and the full solves
below): the `DIAG fixing` counts and
entries add up to the `total` line, its `count` equals `fixed_col` of `DIAG reductions stage=all`,
and no fixing is under `other`; the `DIAG bound_change` lines add up to their total, none under
`input` or `other`; `probing` of `DIAG fixing` equals `fixed` of `DIAG probing`,
`dominated_columns_pairwise` equals `fixed` of `DIAG dominated_columns`, `parallel_columns` equals
`cols_dominated` of `DIAG parallel`; the four ways a pair test ends add up to `pair_tests`; the new
fields of the `probing_round` lines add up to (maxima: equal the maximum of) those of
`DIAG probing`; `rows_touched <= dir_rows_touched`, at most two directions per probe, at least one
bound change per direction, `max_dir_rows_touched <= max_rows_touched <= rows`; and the identities
of the first version still hold (stage lines add up to `DIAG total`, `doubleton_equation`
entries). Two runs of the same instance print the same lines apart from the seconds (wachplan,
ns1830653, uccase9, trento1, brazil3, unitcal_7). With `HIGHS_DIAG_PROBE_ROWS=0` (uccase9,
wachplan) the eight row fields are 0 and every other field is the same as without the variable.
The longest line is 789 characters. An unknown
name in `HIGHS_DIAG_OFF` still aborts with a message (exit code 134). `run_baseline.py` reads the
new lines without a change: `fixing` and `bound_change` become dictionaries by name,
`dominated_columns` a list of one line.

Full solves (`highs` binary, 1 thread, 120 s limit) of app1-1, sp150x300d and n5-3:

| run | app1-1 | sp150x300d | n5-3 |
|---|---|---|---|
| stock | optimal -3, 4077 nodes | optimal 69, 1 node | optimal 8105, 355 nodes |
| `diag2`, no switch | optimal -3, 4077 nodes | optimal 69, 1 node | optimal 8105, 355 nodes |
| the four new switches | optimal -3, 4485 nodes | optimal 69, 1 node | optimal 8105, 470 nodes |
| the four and `HIGHS_DIAG_REDUNDANCY_EXPLICIT=1` | optimal -3, 4485 nodes | optimal 69, 1 node | optimal 8105, 470 nodes |
| Stage 1 set, the four and the variable (no `presolve_rule_off`) | optimal -3, 4149 nodes | optimal 69, 1 node | optimal 8105, 671 nodes (dual bound 8104.357 in the log) |

app1-1 restarts and prints three blocks without switches (`restart=0` to `restart=2`), four with
the four switches, two in the last row; the fixings add up in every block. In the restart blocks
of the run without switches the fixings are `bounds_equal_input` (17 and 151 columns: fixed by the
search before the restart) and one `probing`.

What the fixing lines show on the 23 instances without switches. Counts of 1500 or more:
`weakly_dominated_col` (neos-4306827-ravan 44039, var-smallemery-m6j6 1933),
`singleton_row_bounds_equal` (neos-4306827-ravan 7799, supportcase40 6435, bnatt500 1971),
`bounds_equal_binary_row_test` (triptim1 4360, brazil3 2552, supportcase40 1700),
`dominated_columns_pairwise` (neos-4306827-ravan 2875), `dual_fixing_bound` and
`bounds_equal_implied_bound` (radiationm18-12-05 2854 and 2577), `empty_col` (supportcase40 2296,
neos-4306827-ravan 1960), `probing` (piperout-d27 2043), `parallel_columns` (neos-4306827-ravan
1761). In the Stage 1 setting with all four switches the fixings left on the five instances are
`empty_col`, `dual_fixing_lock`, `singleton_row_bounds_equal`, `forcing_row`,
`bounds_equal_binary_row_test` and `bounds_equal_input`.

## Review of the 2026-11-26 additions (dev node, second `diag2` build, functional only)

The additions were read against the HiGHS source (every place the four rules are applied, every
user of `isRedundant`, every call that fixes a column or changes a column bound, the probing
counters in `HighsDomain` and `HighsImplications`) and the checks were made again with other
scripts. The review changed three things; the patch as it is now has a sha256 that starts with
`7d8c3c0b4db9`.

- A probing direction ends when its backtrack starts (before: when the backtrack ends). The rows
  are no longer marked a second time while the bound changes are undone, which halves the marking
  passes. No count changes: on the 23 instances of the section above, the build before the edit
  and the build after it print the same DIAG lines apart from the seconds.
- The row marking is switched off when a probing round ends, also when the round is left by an
  exception. Before, a round left in the middle of a probe would have left the marking on for every
  later `HighsDomain` of that thread.
- `build_highs_diag.sh` refuses `NAME=stock`, which would have installed the patched HiGHS over
  `highs_stock_install`.

Nothing else was found to correct. What the reading confirmed: with a switch unset each changed
statement is the stock statement; `convertImpliedInteger` is the only place a column becomes
implied integer; the two predicates are the only places the implied-equation test is made;
simple probing exists in `rowPresolve` only; `isRedundant(row)` has the three users listed under
the variable and the forcing-row test reads the explicit sums already; every push of a `fixed_col`
entry in a MIP presolve is preceded by exactly one count; the counters only write to their own
variables.

Build. The `diag2` source copy, build tree, install and tool build were deleted and built again
from the patch file with `NAME=diag2 JOBS=8` (clone, patch, configure, 242 build steps, no compiler
warning). The CMake cache of the new tree equals the one of `build/highs_diag` apart from the name,
and with no `NAME` the script uses the same paths and stamp file as the existing `diag` build. It
was not run with the default name.

Runs: in `gpu_presolve_deps/build/diag2_check/review` (`rcheck.sh` runs a phase, `rdrive1.sh` to
`rdrive3.sh` are the sequences, `rsumm.py` verifies the DIAG lines, `rcmp.py` compares two logs,
`summary5.txt` is the output of `rsumm.py`). At most four runs at a time, next to the measurement
that was running on the node, so no time in these logs is a measurement.

No switch set, presolve only by the options file, stock binary against the `diag2` binary: the
presolved MPS files are byte-identical on 32 instances. These are the 23 of the section above and
nine that were not used there: timtab1, gmu-35-40, mcsched, icir97_tension, roll3000, seymour,
bnatt400, and bab6 and splice1k1 (testbed, 1.3 and 1.8 million nonzeros). Both binaries end with
exit code 1 on satellites2-60-fs and neos-4306827-ravan for the reason given above.

Each new switch alone through the baseline tool (exit code 0 in all 18 runs; rows, columns,
nonzeros after presolve):

| instance | switch | without | with | counter |
|---|---|---|---|---|
| brazil3 | `simple_probing` | 2777, 6740, 42182 | 3218, 7272, 55890 | `simple_probing` 13782 to 0 |
| piperout-d27 | `simple_probing` | 7094, 5772, 79040 | 7090, 5772, 78859 | 12 to 0; fixing `simple_probing` 450 to none |
| bab6 | `simple_probing` | 29306, 112612, 986387 | same | 80 to 0 |
| neos-4306827-ravan | `dual_fixing_substitution` | 77976, 43682, 218370 | 78110, 43766, 218735 | `dual_fixing` 51601 to 0; fixing `dual_fixing_bound` 13 to none |
| radiationm18-12-05 | `dual_fixing_substitution` | 3627, 3889, 13631 | 4141, 4403, 15086 | 1985 to 0; `dual_fixing_bound` 2854 to none |
| var-smallemery-m6j6 | `dual_fixing_substitution` | 4342, 1788, 276002 | 4209, 1770, 273018 | 865 to 0 |
| piperout-d27 | `dual_fixing_substitution` | 7094, 5772, 79040 | 7083, 5780, 78585 | 757 to 0; `dual_fixing_single_equation` 10 and `dual_fixing_bound` 6 to none |
| icir97_tension | `dual_fixing_substitution` | 853, 1080, 15077 | same | `dual_fixing_single_equation` 24 and `dual_fixing_bound` 15 to none (no substitution on this instance) |
| gmu-35-40 | `dual_fixing_substitution` | 370, 654, 4469 | 371, 655, 4473 | 1 to 0 |
| unitcal_7 | `dual_fixing_substitution` | 36202, 19726, 97142 | same | 9 to 0 |
| uccase9 | `implied_equations` | 32847, 22474, 256250 | same | `implied_equation` 0 to 0 |
| supportcase42 | `implied_equations` | 18157, 19183, 435089 | same | 0 to 0 |
| unitcal_7 | `implied_equations` | 36202, 19726, 97142 | same | 1 to 0 |
| trento1 | `implied_integer_conversion` | 1192, 7606, 89185 | same | `implied_integer_cols` 1178 to 0 |
| uccase9 | `implied_integer_conversion` | 32847, 22474, 256250 | 33372, 22857, 257565 | 1703 to 0 |
| supportcase42 | `implied_integer_conversion` | 18157, 19183, 435089 | 18297, 19323, 435369 | 1773 to 0 |
| roll3000 | `implied_integer_conversion` | 1489, 858, 24990 | 1490, 862, 25007 | 8 to 0 |
| ns1830653 | `implied_integer_conversion` | 1433, 587, 24263 | 1438, 588, 24309 | 129 to 0 |

As in the section above, `implied_equations` alone has nothing to silence on uccase9 and
supportcase42. Added to the Stage 1 set it takes `implied_equation` from 8 to 0 on uccase9 and from
140 to 0 on supportcase42; on roll3000 the count is 1 in the Stage 1 set and 0 with all four. All
four switches together and no other (five runs, exit code 0, all four counters 0) give the sizes
listed in the section above.

Fixing by locks with `dual_fixing_substitution`. In the runs without the Stage 1 set no column is
counted under `dual_fixing_lock`, with or without the switch: the dominated-column test of column
presolve runs before `dualFixing` and takes these columns (`dominated_col`,
`weakly_dominated_col`). In the Stage 1 setting, where that test is off, the fixing is there with
all four switches set: neos-4306827-ravan 1106 columns, gmu-35-40 153, trento1 9, roll3000 5.

Stage 1 setting (baseline tool, `--presolve-rule-off 9175456`, the eleven names of `diag_stage1b`),
without and with the four new switches; exit code 0 in all runs; rows, columns, nonzeros:

| instance | Stage 1 set | + all four |
|---|---|---|
| brazil3 | 3275, 6890, 40518 | 9315, 17610, 75082 |
| neos-4306827-ravan | 99998, 62968, 268132 | 185534, 119992, 467716 |
| uccase9 | 33495, 22934, 258515 | 33503, 22942, 258531 |
| supportcase42 | 18159, 19186, 435093 | 18439, 19466, 435653 (not reduced) |
| trento1 | 1264, 7678, 93396 | same |
| timtab1 | 169, 382, 812 | same |
| gmu-35-40 | 415, 688, 4673 | 416, 689, 4676 |
| mcsched | 1861, 1503, 7380 | same (`implied_integer_cols` 2 to 0) |
| icir97_tension | 991, 2113, 16666 | 994, 2126, 16684 |
| roll3000 | 1560, 1040, 25704 | 1561, 1041, 25706 |

One switch at a time added to the Stage 1 set on the first five (20 runs, exit code 0) gives the
table of the section above. With all four, the fixings left on the ten instances are `empty_col`,
`dual_fixing_lock`, `singleton_row_bounds_equal`, `forcing_row`, `bounds_equal_binary_row_test` and
`bounds_equal_input`, and the bound changes left are `binary_row_test`, `singleton_row`,
`doubleton` and `implied_bound`.

`HIGHS_DIAG_REDUNDANCY_EXPLICIT=1`, no other switch, baseline tool (exit code 0 in all runs);
`redundant_row` without and with the variable: brazil3 3268 / 3268, istanbul-no-cutoff 1242 / 1242,
radiationm18-12-05 8685 / 8685, supportcase40 23188 / 23188, uccase9 6931 / 6931, timtab1 0 / 0,
gmu-35-40 6 / 6, mcsched 105 / 105, icir97_tension 39 / 39, roll3000 364 / 364 (on these ten every
DIAG line is the same with and without the variable, seconds aside), neos-4306827-ravan 167466 /
167448 (same sizes), unitcal_7 5350 / 5357. On unitcal_7 the initial scan removes 902 redundant
rows instead of 2237. The others are redundant only through implied bounds; they stay until the
fast loop (4453 redundant rows there instead of 3112), and while they are in the model row and
column presolve get more out of them: 18 more singleton rows, 3 more columns fixed by the binary
row test, 3 more weakly dominated columns. The presolved model has 27 rows and 10 columns fewer
(36175, 19716, 97071). In the Stage 1 setting with the four switches the variable changes no DIAG
line on brazil3, neos-4306827-ravan, uccase9, supportcase42 and trento1.

Consistency of the output, checked by `rsumm.py` on 279 presolve calls (all runs of this
section, restart blocks of the full solves included): the `DIAG fixing` counts and entries add up
to the `total` line, its count equals its `fixed_col` field and `fixed_col` of
`DIAG reductions stage=all`, no fixing is under `other`; the `DIAG bound_change` lines add up, none
under `input` or `other`; the round lines add up to `DIAG probing` (the maxima are the maxima);
`rows_touched <= dir_rows_touched`, at most two directions per probe, at least one bound change
per direction, `max_dir_rows_touched <= max_rows_touched <=` rows of the model; the four ways a
pair test ends add up to `pair_tests`; the fixing counts `probing`, `dominated_columns_pairwise`
and `parallel_columns` equal `fixed` of `DIAG probing`, `fixed` of `DIAG dominated_columns` and
`cols_dominated` of `DIAG parallel`; the stage lines add up to `DIAG total`; `doubleton_equation`
is explained as in the first version. No call failed a check. The longest DIAG line is 789
characters. Two runs of the same instance (binary and baseline tool, six instances) print the same
lines apart from the seconds. With `HIGHS_DIAG_PROBE_ROWS=0` (uccase9, wachplan) the eight row
fields are 0 and every other field is unchanged. An unknown name in `HIGHS_DIAG_OFF` aborts with
the message (exit code 134).

Full solves (`highs` binary, 1 thread, 180 s limit) of eleven instances that were not used above,
each with stock, `diag2` without switches, the four new switches, the variable alone, and the
Stage 1 set with the four switches and the variable (no `presolve_rule_off`):

- optimal with the same value in all five runs: p200x1188c 15078, mik-250-20-75-4 -52301,
  neos-3381206-awhea 453, pg -8674.3426, swath1 379.0713, irp 12159.4928, neos17 0.150003,
  h80x6320d 6382.0991;
- fiball: optimal 138 with stock, without switches and with the variable; the two runs with the
  four switches reach the time limit with the solution 138 and the dual bound 137.07;
- neos-1445765: optimal -17783 with stock, without switches and with the variable; the two runs
  with the four switches reach the time limit with bounds on both sides of it;
- tr12-30: every run reaches the time limit, best solution 130596 or 130599, dual bounds below.

Stock and `diag2` without switches end with the same status, bounds and node count on the ten
instances that finish. p200x1188c restarts ten times at the root and prints eleven blocks, irp
and h80x6320d four; in the restart blocks the columns the search fixed are under
`bounds_equal_input`. app1-1, the restart instance of the section above, was solved again with
the same five settings: optimal -3 in all of them, 4077 nodes with stock, without switches and
with the variable; three blocks without switches, four with the four switches, two with the Stage 1
set. unitcal_7 was solved without switches and with the variable (900 s limit): both runs end
optimal within the gap tolerance of 0.01%, with 19635558.244 after 139 nodes and 19635641.139
after 49 nodes; the dual bound of the run with the variable, 19633723.556, is below the first
value.

Not checked by the review: instances above 2 million nonzeros, timing, and the cost of the
counters.

## Checks of the 2026-11-27 additions (Mac, static library without CMake, functional only)

The additions are the three switches that split `dual_fixing_substitution`, the switch
`implied_row_dual_bounds` and the two `DIAG free_col_substitution` lines. The patch as it is now
has a sha256 that starts with `1cd0228087c4`. The node was not used.

Build. The Mac has no CMake. Two source trees were compiled by a script into static libraries
(Apple clang, arm64, `-O2 -DNDEBUG -ffp-contract=off`, 64-bit `HighsInt`, 235 source files):
"old" is commit `283bfc859` with the patch of 2026-11-26 (`7d8c3c0b4db9`), "new" the same commit
with the patch as it is now. The tool is `baseline_tool.cpp` with one addition that is
not in the repository: a hash of the presolved model (FNV-1a over its sizes, matrix, sides, bounds,
costs, offset and column types). The patch file was applied by `build_highs_diag.sh` itself, in a
copy that has only another `DEPS` directory and a `cmake` that records its arguments: a fresh clone
at `283bfc859` and a source copy that carried the old patch both end as the tree that was compiled,
and `NAME=diag3` writes only `highs_diag3_src`, `build/highs_diag3`, `highs_diag3_install` and
`build/baseline_diag3`. The CMake build itself was not run.

Instances and settings. "The 240" are all instances of the MIPLIB 2017 benchmark set. "The 204" are
those of them whose full presolve took at most 10 s in the stored node run
`results/bench240/highs_diag2_full`; left out are app1-2, bab2, bab6, chromaticindex1024-7, co-100,
eilA101-2, ex10, gfd-schedulen180f7d50m30k18, k1mushroom, mzzv11, mzzv42z, neos-2075418-temuka,
neos-3402454-bohle, neos-4532248-waihi, neos-4647030-tutaki, neos-4722843-widden,
neos-4763324-toguru, neos-5114902-kasavu, neos-631710, neos-787933, neos-848589, ns1644855,
ns1760995, nursesched-medium-hint03, nw04, piperout-08, piperout-27, rd-rplusc-21, roi5alpha10n8,
s100, s250r10, splice1k1, square41, square47, supportcase33 and var-smallemery-m6j6. Four
settings, presolve only through the tool, one thread, five runs at a time:

- cheap: every cheap rule on, the reference of Stage 1c
  (`HIGHS_DIAG_OFF=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns`,
  no rule bit), on the 240;
- Stage 1: the stored variant `stage1c` (`--presolve-rule-off 9175456`, the ten names above and
  `parallel_columns,simple_probing,dual_fixing_substitution,implied_equations,implied_integer_conversion`),
  on the 240;
- left-out: the cheap setting with `implied_equations,implied_integer_conversion` and
  `--presolve-rule-off 8913056` (rule bits 5, 7, 19 and 23), which is the switch set of
  `results/bench240/cheap_ablation/off_leftout.csv`, on the 240;
- full: no switch besides the ones under test, on the 204.

6636 runs in all (twelve variants of cheap, three of Stage 1, five of left-out, nine of full),
every one with exit code 0 and none at the time limit. Two runs are called equal when the exit
code, the presolve status, the hash of the presolved model and every DIAG line agree, the fields
with `seconds` in their key aside.

The local build against the node. The old build in the cheap setting gives the sizes and the
`DIAG fixing` and `DIAG bound_change` counts of the stored node run
`results/bench240/highs_cheap_all` on 239 of the 240; fiball ends with 32397 columns and 99726
nonzeros instead of 32399 and 99734. The new build gives those of `results/bench240/highs_stage1c`
on all 240 in the Stage 1 setting and those of `highs_diag2_full` on 203 of the 204 in the full
setting (fiball again). In the left-out setting the new build gives the sizes of `off_leftout.csv`
on 239 of the 240, and the old build with the old name the sizes and counts of
`results/bench240/stage1c_refs/m3_a` on 239 (fiball both times).

No new switch set. The old and the new build are equal on the 240 in the cheap setting, on the 240
in the Stage 1 setting and on the 204 in the full setting. The `DIAG switches` line is part of the
comparison; the two `DIAG free_col_substitution` lines, which the old build does not print, are
not.

The old name. With `dual_fixing_substitution` added, the old and the new build are equal on the
240 (cheap) and on the 204 (full); the Stage 1 setting contains the name. With the three new names
in its place the new build is equal to itself with the old name on the 240 (cheap), the 240
(Stage 1) and the 204 (full), the `DIAG switches` line aside. In the left-out setting the new
build with the three names is equal to the old build with the old name on the 240.

Each new name alone, cheap setting, the 240. With every cheap rule on the counters are: 13017
fixings `dual_fixing_single_equation` on 11 instances (b1c1s1 2, icir97_tension 24,
neos-3656078-kumeu 2, neos-3216931-puriri 1, neos-827175 732, snp-02-004-104 3, triptim1 1,
irish-electricity 630, neos-5093327-huahum 6, physiciansched3-3 1, savsched1 11615), 74920
substitutions `dual_fixing` on 37, 205092 fixings `dual_fixing_bound` on 26 (supportcase12 184311)
and 19965 bound changes `dual_fixing` on 39. Sizes are rows, columns, nonzeros after presolve.

| switch | its counters | presolved model differs on | sizes differ on | examples |
|---|---|---|---|---|
| `dual_fixing_single_equation` | 13017 to 0 | 4 | 3 | icir97_tension 881, 1115, 15173 to 882, 1117, 15181; b1c1s1 3880, 3846, 11348 to 3882, 3850, 11356; neos-5093327-huahum six columns more; savsched1 with equal sizes |
| `dual_fixing_binary_substitution` | 74920 to 0 | 28 | 26 | radiationm40-10-02 21259, 21754, 63721 to 29741, 28121, 84943; var-smallemery-m6j6 8044, 2863, 445729 to 8619, 3338, 518475; neos859080 120, 120, 1200 to 160, 160, 1280 |
| `dual_fixing_bound_strengthening` | 205092 and 19965 to 0 | 35 | 16 | supportcase12 14450, 75282, 192634 to 27466, 138632, 338051; rocII-5-11 14391, 5974, 154778 to 14946, 6266, 162808 |
| the three, or `dual_fixing_substitution` | all to 0 | 59 | 40 | |

On the other seven instances where the single-equation extension fires the presolved model is the
same without it. In the full setting on the 204 the presolved model differs on 1, 22, 33 and 49
instances for the four rows.

Counters of the other two parts that rise when one part is off. Every rise in the two settings was
traced, in the setting where it was seen, with a build that writes the original index of each
column a rule fixes, substitutes or tightens; the base run is the one with all three parts on.

- Without the extension, bound strengthening fixes more columns on irish-electricity (436 to
  856), neos-3656078-kumeu (35 to 36) and physiciansched3-3 (369 to 370), in the full setting also
  on neos-3083819-nubu (0 to 7) and rocII-5-11 (68 to 69). Every additional column is one the
  extension fixes in the base run (420 of 420, 1 of 1, 1 of 1, 7 of 7, 1 of 1). On savsched1 the
  bound changes rise from 2582 to 2634: they reach 43 more columns, of which the base run removes
  41 as empty columns and fixes 2 by bound strengthening.
- Without the substitution, bound strengthening fixes more columns on radiationm18-12-05 (2854 to
  3348), radiationm40-10-02 (10777 to 12852), piperout-08 (11 to 58), piperout-27 (21 to 106),
  physiciansched6-2 (227 to 407) and physiciansched3-3 (369 to 382), in the full setting also on
  CMS750_4 (1 to 6) and highschool1-aigio (0 to 2). The additional columns are columns the base run
  substitutes: 494 of 494, 2075 of 2075, 47 of 48 (the other one is fixed by the binary row test in
  the base run), 86 of 86, 257 of 257, 54 of 85 (the other 31 are fixed through singleton rows in
  the base run) and 2 of 2. On CMS750_4 they are five columns that the base run removes as empty
  (2) or weakly dominated (3) after its 12 substitutions. In the full setting the bound changes on
  physiciansched3-3 rise from 633 to 638: five more columns, of which the base run substitutes
  four and fixes one in the pairwise dominated-column test.
- Without bound strengthening, the extension fixes 30 columns instead of 24 on icir97_tension: two
  are columns bound strengthening fixes in the base run, four are fixed in the input model and are
  now removed by the extension before column presolve reaches them (`bounds_equal_input` 166 to
  162). On physiciansched3-3 the substitution counts 15599 instead of 15598; the column is fixed
  as weakly dominated in the base run.
- No rise without a traced cause was seen. Falls are larger: without bound strengthening the
  substitution does nothing on radiationm18-12-05 and radiationm40-10-02 in the cheap setting (1985
  and 8425 in the base run; the presolved model then has the sizes of the one without the
  substitution, 7242, 6867, 20733 and 29741, 28121, 84943, and is another model), and without the
  extension bound strengthening fixes nothing on savsched1 (4760). The falls were not traced.

The left-out setting, the 240. There the dominated-column tests cannot stand in for the
extension. It fixes 13032 columns on 9 instances (savsched1 11615, neos-827175 732,
irish-electricity 630, icir97_tension 24, neos-3656078-kumeu 20, neos-5093327-huahum 6,
snp-02-004-104 3, neos-3216931-puriri 1, triptim1 1). With `dual_fixing_single_equation` added
the presolved model differs on 3 instances: icir97_tension 881, 1115, 15173 to 882, 1117, 15181,
neos-5093327-huahum with six columns more, savsched1 with equal sizes. The traces say where its
columns go: on savsched1 4731 to the lock rule and 6884 become empty columns, on neos-827175 all
732 to the lock rule, on irish-electricity 420 to bound strengthening and 210 to singleton rows,
on neos-3656078-kumeu all 20 to bound strengthening (35 to 55 fixings). With
`dual_fixing_binary_substitution` added as well, which leaves bound strengthening alone, the sizes
change on 28 instances, with all three names on 39. neos-3402294-bobin ends with 32932 rows, 792
columns and 381000 nonzeros in all four.

Time, as an observation only (sum of the tool's `core_seconds` over the 240 in the cheap setting,
five runs at a time on a laptop that was in use): 315 s with every cheap rule, 146 s with
`dual_fixing_substitution`, 144 s with the three names, 158 s with `dual_fixing_single_equation`
alone, 312 s with `dual_fixing_binary_substitution` alone, 305 s with
`dual_fixing_bound_strengthening` alone. rd-rplusc-21 takes 137 s of the 315 s and 0.5 s whenever
the extension is off, with the same presolved model in all six settings and no fixing by the
extension; neos-4647030-tutaki goes from 10.3 s to 1.8 s. So nearly all the time that
`dual_fixing_substitution` saves is the marking of the single-equation extension, not the work of
the three rules: an earlier build of the switch, which skipped `handleSingleEquation` and still set
the mark, took 143 s on rd-rplusc-21 with all three names set.

`DIAG free_col_substitution`. On all 5268 presolve calls of the new build (sixteen variants on the
240, seven on the 204) the three kinds add up to `count`, the sub-counts stay within their kinds,
the two `count` fields add up to `free_col_substitution` of `DIAG reductions stage=all`, and
`count` of the `aggregator` line equals `substituted` of `DIAG aggregator`. The longest of these
lines has 158 characters.

| setting | line | count | equation | inequality_own | inequality_other | upper_side | ranged | own_zero_cost | integer |
|---|---|---|---|---|---|---|---|---|---|
| cheap, the 240 | `singleton_col` | 16881 on 59 instances | 11596 on 54 | 5285 on 13 | 0 | 288 on 4 | 0 | 0 | 699 on 22 |
| left-out, the 240 | `singleton_col` | 16549 on 55 | 11380 on 53 | 5169 on 10 | 0 | 224 on 5 | 0 | 4 on 1 | 696 on 22 |
| full, the 204 | `singleton_col` | 15519 on 54 | 10223 on 49 | 5296 on 13 | 0 | 290 on 5 | 0 | 0 | 705 on 19 |
| full, the 204 | `aggregator` | 114851 on 89 | 101270 on 79 | 12399 on 21 | 1182 on 7 | 12279 on 21 | 0 | 12220 on 19 | 50384 on 53 |

In the cheap and the left-out setting the `aggregator` line is zero. In the cheap setting the
inequality pivots of singleton columns are on irish-electricity (4887 of its 10039
substitutions), supportcase42 (140), map10 and map16715-04 (87 each), mushroom-best (44),
neos-3046615-murg (9), uccase9 (8), snp-02-004-104 (7), physiciansched6-2 and physiciansched3-3
(5 each), rocII-5-11 (3), sct2 (2) and roll3000 (1). In none of the settings was a singleton
column substituted through an inequality whose dual sign came from another column. Four of the
inequality pivots have a pivot column with zero cost, all in the left-out setting on one instance.

`implied_row_dual_bounds` alone. In the cheap setting the presolved model differs on 22 of the
240, in size on 21: csched007 297, 1703, 6270 to 347, 1753, 6370; irish-electricity 70646, 40633,
409523 to 75333, 45438, 423172; supportcase12 14450, 75282, 192634 to 14984, 75816, 194503;
physiciansched3-3 69283, 17401, 370925 to 69497, 17598, 372719; map10 and map16715-04 49894,
26617, 106266 to 50120, 26843, 106718; supportcase42 18159, 19186, 435093 to 18299, 19326, 435373;
and neos-3046615-murg, neos-4338804-snowy, b1c1s1, icir97_tension, sct2, roll3000, seymour1,
mushroom-best, rocII-5-11, uccase9, snp-02-004-104, physiciansched6-2, savsched1 and
roi5alpha10n8 by smaller amounts. The free column substitutions fall from 16881 to 11379, all with
an equation as pivot. In the full setting the model differs on 33 of the 204 (32 in size). In none
of the 444 runs is an inequality pivot or an `implied_equation` entry left.

Full solves (a small driver on the same two libraries, 1 thread, 120 s limit, five at a time) of
fourteen instances in nine settings: the old build without a switch and with
`dual_fixing_substitution`; the new build with these two, with the three names, with each of the
three alone and with `implied_row_dual_bounds`. All 126 solves end with exit code 0. Twelve
instances end optimal with the same value in all nine settings: app1-1 -3, sp150x300d 69, n5-3
8105, p200x1188c 15078, mik-250-20-75-4 -52301, neos-3381206-awhea 453, pg -8674.3426, swath1
379.0713, irp 12159.4928, neos17 0.150003, h80x6320d 6382.0991 and timtab1 764772 (with
`implied_row_dual_bounds` 764774 with the dual bound 764703.03, inside the gap tolerance). On
these twelve the old and the new build end with the same status, bounds, node count and number of
DIAG blocks, without a switch and with the old name, and the three names give what the old name
gives (app1-1 4485 nodes against 4077 without a switch, n5-3 612 against 355, mik-250-20-75-4
12823 against 4008). p200x1188c prints eleven blocks, so the switches were also read in restarts.
icir97_tension and gmu-35-40 reach the time limit in every setting, with best solutions 6386 to
6389 and -2406138.5 to -2406583.7 and dual bounds below them.

Other checks. A name that is not a switch still aborts with the message (exit code 134; tried
with `dual_fixing_bound`). `DIAG switches off=` prints the old name when the old name was given
and the three names when they were given. `run_baseline.py` reads the new lines without a change:
`free_col_substitution` becomes a dictionary with the keys `singleton_col` and `aggregator` in the
`diag` object of a summary (its parser was run on 684 of the logs, no malformed or repeated line).
The engine's tests (385) pass in the assertion and in the release build; no engine file was
changed.

Not checked: the CMake build and anything on the node; testbed instances that are not among the
240; the 36 instances left out of the full setting; LP presolve calls; timing.
