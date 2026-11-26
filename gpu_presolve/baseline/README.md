# HiGHS baseline and instance structure (Stage 0)

`baseline_tool` reads one MPS file with HiGHS, writes structure statistics of the model, runs the
HiGHS MIP presolve alone (no solve) and writes the same statistics for the presolved model.
`run_baseline.py` runs it over the testbed and writes `summary.json` and `summary.csv`.

All commands below are run on the dev node from the worktree root
(`/home/scratch.acoerduek_sw/cuopt_dev/.wt/gpu-presolve`). From the Mac, wrap each one in
`cuopt exec '<command>'`.

## Build

```bash
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
cmake -S gpu_presolve/baseline -B $DEPS/build/baseline_stock -G Ninja \
      -DHIGHS_ROOT=$DEPS/highs_stock_install
cmake --build $DEPS/build/baseline_stock -j 8
```

`HIGHS_ROOT` is a HiGHS install prefix (`lib/cmake/highs`, `lib/libhighs.so`, `include/highs`); the
default is the stock install above. Each HiGHS build gets its own build tree
`$DEPS/build/baseline_<variant>`. For the HiGHS built by `build_highs_diag.sh`:

```bash
cmake -S gpu_presolve/baseline -B $DEPS/build/baseline_diag -G Ninja \
      -DHIGHS_ROOT=$DEPS/highs_diag_install
cmake --build $DEPS/build/baseline_diag -j 8
```

A second patched build can live next to the first one: `NAME=diag2 bash
gpu_presolve/baseline/build_highs_diag.sh` builds and installs under `highs_diag2_*`, and the tool
is then configured with `-B $DEPS/build/baseline_diag2 -DHIGHS_ROOT=$DEPS/highs_diag2_install`.
Use a second name whenever a measurement is running on the first build, because a rebuild replaces
the library under the running tool. `stage1_reference_runs.sh` runs the references for the Stage 1
and Stage 2 gates with the `diag2` tool.

The binary finds `libhighs.so` through its run path. At start it aborts if the library that was
loaded is not below the `HIGHS_ROOT` it was built with, and the JSON records the library file
(`highs.library`).

Run the build command again after every rebuild of a HiGHS install (the patched one is rebuilt
whenever `highs_diag.patch` changes): the tool is compiled against the installed headers, and the
build recompiles and relinks it when they or the library changed. A tool older than its library is
not detected at run time.

## baseline_tool

```
baseline_tool <instance.mps> <out.json> [--time-limit <s>] [--dump-csr <file>] [--stats-only]
```

- Reads the model (`read_seconds`), computes the statistics of the original model (`original`).
- Unless `--stats-only`: `Highs::presolve()` with `threads = 1`, `presolve = on`,
  `time_limit` (default 1800 s), `log_dev_level = 1`; the HiGHS log goes to `<out.json>.log`.
  Then the same statistics for `getPresolvedLp()` (`presolved`, `null` when presolve returns no
  model, for example `Infeasible`).
- The JSON is written to `<out.json>.tmp` and renamed at the end, so an aborted run leaves no
  `<out.json>`. Exit code 0 when the run completed (whatever the presolve status), 1 when HiGHS
  cannot read the file, 2 on a usage error.

`presolve` object:

| Field | Meaning |
|---|---|
| `seconds` | wall time of the `Highs::presolve()` call |
| `setup_seconds` | call start until HiGHS logs `Presolving model` (MIP solver set-up, model copy) |
| `core_seconds` | `Presolving model` until the first `Presolve reductions` line: the presolve itself |
| `handover_seconds` | from there until the call returns (copy of the reduced model and postsolve stack) |
| `reached_time_limit` | `seconds >= time_limit`. Stock HiGHS returns `Reduced` or `Not reduced` when its presolve stops at the limit, and it tests the limit only every 1024 reductions, so it can overrun |
| `status`, `highs_return` | `presolveStatusToString` of the presolve status, return value of the call |
| `rows_removed`, `cols_removed`, `nnz_removed` | original minus presolved (negative nonzeros = fill-in) |

`core_seconds` and `handover_seconds` are absent when HiGHS prints no `Presolve reductions` line
(infeasible or unbounded in presolve). `core_seconds` is the time to compare with another presolve;
`seconds` also holds the copies the API makes (about 5% on the largest dev instances).

Two cases in which `presolved` is not HiGHS's presolved and scaled model:

- Status `Not reduced`: `getPresolvedLp()` is the original model. Bounds that HiGHS tightened and
  implied integers it found are not in it (the internal count is `implied_integer_cols` on the
  patch's `DIAG total` line).
- Presolve stopped at the time limit (`reached_time_limit`): HiGHS skips its final scaling, so
  `presolved.row_dynamism` is unscaled.

`peak_rss_mb` (top level) is the peak resident size of the tool process, which holds the tool's
own row-wise copy of the model next to the HiGHS data.

Statistics (`original` and `presolved`):

- `rows`, `cols`, `nnz`.
- `col_types`: `binary` (integer with bounds 0 and 1), `general_integer`, `implied_integer`
  (HiGHS `kImplicitInteger`, only in the presolved model, counted apart whatever its bounds),
  `continuous`, `semi`.
- `row_sides`: `equation`, `ranged`, `one_sided`, `free`.
- `row_length`, `col_length`: min, p10, p50, p90, p99, max, mean. Percentiles are nearest rank
  (the value at position ceil(p * n / 100) of the sorted list).
- `long_cols`: columns longer than max(1000, 1% of rows); `long_rows`: rows longer than
  max(1000, 1% of columns).
- `row_classes`: every row is in exactly one class, the first that applies in this order:
  1. `empty`, `singleton` (by length), `doubleton_equation` (two entries, equation);
  2. rows whose columns are all binary. If all coefficients have the same absolute value the row
     is brought to a sum of literals: divided by that value, sides rounded to integers, a side the
     sum cannot violate dropped, and either the negative or the positive columns complemented.
     `set_packing` if a form reads sum <= 1, `set_partitioning` if sum = 1, `set_covering` if
     sum >= 1 (packing is tested first, so a two-entry row such as x + y >= 1 counts as packing),
     `cardinality` otherwise. With different absolute values: `binary_knapsack_non_unit`;
  3. other two-entry rows: `variable_bound` (one binary, one other column) or `other_two_entry`;
  4. `all_integer`, `all_continuous`, `mixed`.
  An implied integer column with bounds 0 and 1 counts as binary here.
- `two_binary_rows`: rows with exactly two entries, both binary (any sides; overlaps the classes).
- `coefficient`: smallest and largest |a|.
- `row_dynamism`: max |a| / min |a| per non-empty row; p50, p99, max.
  `row_dynamism_highs_scaled` (original only): the same after the equilibration HiGHS applies at
  the end of its MIP presolve (`HPresolve::scaleMIP`: power-of-two scaling of rows with a
  continuous entry, then of continuous columns). The presolved model returned by HiGHS already has
  this scaling and is a minimisation, so compare `presolved.row_dynamism` with
  `original.row_dynamism_highs_scaled`.
- `bounds`: columns with infinite lower / upper bound, free columns, finite bounds with
  |b| >= 1e8 (`lower_huge`, `upper_huge`, `cols_with_huge_bound`), and
  `rows_with_huge_bound_col`.
- `rows_by_infinite_activity_terms`: rows by the number of infinite terms (none, one, two or more)
  in their minimum and in their maximum activity (the three cases of DESIGN.md section 2.4).

### CSR dump

`--dump-csr <file>` writes the original model, little endian:

```
char    magic[8] = "GPRSCSR1"
int64   nrows, ncols, nnz
int64   row_start[nrows + 1]
int32   col_index[nnz]          sorted inside each row
double  value[nnz]
double  row_lower[nrows], row_upper[nrows]
double  col_lower[ncols], col_upper[ncols]
double  cost[ncols]
uint8   col_type[ncols]         0 continuous, 1 integer
double  objective_offset
```

Infinite bounds are IEEE infinities. The layout has no objective sense: a maximisation is written
as the equivalent minimisation (cost and offset negated; `dump_csr.objective_negated` in the JSON).
The files are large; write them outside the synced tree.

## run_baseline.py

```bash
python3 gpu_presolve/baseline/run_baseline.py --set dev --variant stock --jobs 4
python3 gpu_presolve/baseline/run_baseline.py --set dev --variant diag --jobs 4
python3 gpu_presolve/baseline/run_baseline.py --set dev --variant diag_noprobing --jobs 4 \
        --tool /home/scratch.acoerduek_sw/gpu_presolve_deps/build/baseline_diag/baseline_tool \
        --env HIGHS_DIAG_OFF=probing
```

`HIGHS_DIAG_OFF=probing` only illustrates `--env`; the switches and their values are those of
`highs_diag.patch`.

For times that will be compared use `--jobs 1`: instances that run at the same time compete for
caches and memory bandwidth, and the 36 CPUs of the dev container are 18 cores with two threads
each.

| Argument | Meaning |
|---|---|
| `--set dev\|large\|status\|all` | `dev`: main testbed instances below 1e7 nonzeros; `large`: the 9 above (`where` is `dev-large`; they also run on the dev node, in the background); `status`: the status list; `all`: main and status. Default `dev`, or `all` when `--only` is given. Held-out instances are refused |
| `--variant <name>` | names the run; default `stock` |
| `--tool <path>` | default `/home/scratch.acoerduek_sw/gpu_presolve_deps/build/baseline_<variant>/baseline_tool` |
| `--jobs N` | instances in parallel, default 4 |
| `--time-limit <s>` | passed to the tool, default 1800 |
| `--kill-after <s>` | the tool is killed after this wall time, default time limit + 3600 |
| `--out <dir>` | default `gpu_presolve/results/stage0/<variant>` |
| `--only a,b` | only these instances of the set |
| `--env KEY=VALUE` | repeatable; added to the tool's environment (the `HIGHS_DIAG_*` switches) |
| `--stats-only` | structure statistics only |
| `--presolve-rule-off <bits>` | passed to the tool, which sets the HiGHS option `presolve_rule_off` (stock HiGHS rule switches, one bit per rule in the order of `PresolveRuleType` in `HConst.h`: 5 dominated column, 7 forcing column, 8 free column substitution, 17 dual fixing, 18 zero-cost singleton, 19 column stuffing, 23 weakly dominated column). Default 0 |
| `--dump-csr-dir <dir>` | also write `<dir>/<name>.csr`; must be outside the worktree |
| `--skip-done` | an instance whose earlier run in `--out` ended with exit code 0 is not run again, so an interrupted run can be continued with the same command |
| `--summarize-only` | run nothing, rebuild the summary of the set from the files in `--out` |
| `--testbed`, `--dataset-dir` | default `gpu_presolve/testbed/testbed.json`, and its `dataset_dir` followed by its `extra_dataset_dirs` (the instances downloaded for this project). `--dataset-dir` can be repeated; the first directory that holds the instance wins |

Per instance in `--out`: `<name>.json` (tool output), `<name>.json.log` (HiGHS log), `<name>.out`
(stdout and stderr of the tool), `<name>.run.json` (command, environment, exit code, wall time).
An instance without `<name>.mps` (or `.mps.gz`) in any dataset directory is reported as `missing`
and skipped.

`summary.csv` has one row per instance: band, where, status, sizes before and after, percent
removed, `presolve_seconds` (the call), `presolve_core_seconds`, `read_seconds`,
`reached_time_limit`, `peak_rss_mb`, and one `stage_<name>_seconds` column per DIAG stage.
`status` is the HiGHS presolve status, or `missing`, `not_run`, `killed`, `failed(<exit code>)`,
`stats_only`. For rows without a tool result the sizes before are those of `testbed.json`.
`summary.json` holds the same rows plus the parsed DIAG data. The exit code is 1 when an instance
was killed or failed.

DIAG lines (written by HiGHS built with `highs_diag.patch`) are read from the HiGHS log and from
the tool's output. Format: `DIAG <kind> key=value ...`; values are parsed as integer, float or
text. A kind whose lines carry a `name=` field becomes a dictionary by name
(`diag["stage"]["probing"]["seconds"]`); a repeated name replaces the earlier line and is counted
in `diag._replaced`. Every other kind becomes the list of its lines in order
(`diag["total"][0]`, `diag["probing_round"][i]`). The stage time in the CSV is the field `seconds`
of `DIAG stage name=<stage> ... seconds=<s> ...`.

## Checks made (2026-11-26, functional, not measurements)

- wachplan, ns1830653, uccase9, trento1: the `Presolve reductions` line in the tool's HiGHS log
  and the sizes in the JSON equal the line printed by the stock `highs` binary (`--threads 1`).
- The 25 dev instances present on the node: rows, columns and nonzeros after presolve equal
  `results/stage0/first_look_stock_highs.csv` (stock binary).
- `--dump-csr` of wachplan, trento1, ns1830653 and supportcase19: all rows (entries, order, sides)
  and all columns (cost, type, bounds) equal the MPS file. The reader used for this is
  `/home/scratch.acoerduek_sw/gpu_presolve_deps/build/baseline_check/verify_csr.py`.
- A 21-row model with one or more rows of every class gives the expected class counts.
- Two runs on uccase9 give the same JSON apart from the times, and the same dump file.
- Built against the `highs_diag.patch` install: same sizes, DIAG lines parsed, and the patch's
  `DIAG total presolve_seconds` agrees with `core_seconds`.
