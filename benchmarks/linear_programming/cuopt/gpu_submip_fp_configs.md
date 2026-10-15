# GPU sub-MIP FP benchmark checkpoints

The default family is now `checkpoint`: every numeric ID repeats the chosen
config 0 policy (one independent distance-FP descent, unlimited projections,
60 s wall cap). IDs no longer select the old effort variants unless the family
is explicitly set to `effort`. The solver seed is unchanged by the ID.

The current checkpoint (`single_descent_schedule_only`) isolates scheduling with
the original CPUFJ seed and nearest-rounding policy. Initial propagation rounding
and automatic warm-start selection are disabled; the stronger cutoff is also
disabled (`min_improve=0`). The crash fix and distance-only FP remain enabled.

## Current two-repetition benchmark

Run the current policy twice against the four historical original runs. This
checkpoint adds host-side INFO diagnostics; FP, starting-point selection, seed,
neighborhood generation, CPU workers, effort limits, and scheduling are unchanged.
No additional GPU synchronization or snapshot writes are enabled.

Set `n_iters=2` in the existing benchmark launcher. Inside its repetition loop
(`repetition_id=0`, then `1`), use:

```bash
unset CUOPT_GPU_SUBMIP_MODE CUOPT_GPU_SUBMIP_LIFETIME
unset CUOPT_GPU_SUBMIP_MAX_PROJECTIONS CUOPT_GPU_SUBMIP_MAX_DESCENTS
unset CUOPT_GPU_SUBMIP_SECONDS CUOPT_GPU_SUBMIP_FP_CONFIG
unset CUOPT_GPU_SUBMIP_INITIAL_ROUNDING CUOPT_GPU_SUBMIP_SELECT_WARM_START
unset CUOPT_GPU_SUBMIP_CACHED_PRIMAL CUOPT_GPU_SUBMIP_INCUMBENT_BLEND
unset CUOPT_GPU_SUBMIP_UNPRODUCTIVE_SECONDS CUOPT_GPU_SUBMIP_REGULAR_SECONDS
unset CUOPT_GPU_SUBMIP_MIN_IMPROVE CUOPT_GPU_SUBMIP_DISABLE_FP
unset CUOPT_GPU_SUBMIP_COMPACT CUOPT_GPU_SUBMIP_FUSED_PDLP
unset CUOPT_GPU_SUBMIP_JOB_SNAPSHOT CUOPT_GPU_SUBMIP_JOB_SNAPSHOT_DIR
unset CUOPT_GPU_SUBMIP_JOB_SNAPSHOT_SOURCE
export CUOPT_GPU_SUBMIP=1
export CUOPT_SUBMIP_POOL_DIAGNOSTICS=1
export CUOPT_GPU_SUBMIP_CONFIG_FAMILY=checkpoint
export CUOPT_CONFIG_ID="$repetition_id"
export CUOPT_MAX_CONFIG="$n_iters"
```

Keep the usual 600-second solve limit, hardware class, thread count, logging, and
incumbent CSV collection. Both IDs must log `preset_id=0`, `family=checkpoint`,
and `profile=single_descent_schedule_only`. Their `GpuSubmipEffectiveConfig` lines
must match: independent CPUFJ mode, distance FP, one descent, no projection-count
cap, 60 s per job, PDLP/rounding/propagation caps 1/0.5/0.7 s, initial rounding and
automatic warm-start selection off, 15 s stall threshold, initial 5 s regular
turn, strict cutoff, and FP enabled. IDs do not change the solver's seed.

All solver diagnostics use `CUOPT_LOG_INFO` in the existing log:

- `GpuSubmipIndependentEnqueue`, `GpuSubmipIndependentStart`, and
  `GpuSubmipJobStart` join by job ID. They identify parent source, strategy,
  elapsed solve time, queue-to-execution delay, dimensions, free integer count,
  and cutoff changes while queued. The start delay includes GPU handoff. The
  free integer count uses unequal variable bounds; `fixed_columns` includes
  inherited fixed columns, while the older `fixed` field counts only additional
  GPU-side RINS fixings. Presolved-away columns are absent from both counts.
- `GpuSubmipJobEnd` records the rejection stage, candidate objective when
  evaluated, and the latest cutoff when checked for publication. `fp_not_found`
  includes candidates rejected by the enclosing FP session; join with
  `GpuSubmipFpDescent` to inspect its raw found flag and cutoff excess.
  `valid=1` means the worker invoked publication, not that the global incumbent
  improved or the benchmark callback passed validation. Use root acceptance
  logs and incumbent CSVs for those outcomes.
- Existing FP projection/rounding/exit, queue, regular-worker handoff, GPU
  ownership, and CPU pool summaries remain enabled. Ownership is wall time,
  not measured SM utilization; pool-slot availability is not CPU-thread idleness.

Report each repetition and the pooled canonical PI against all four historical
baseline repetitions. Freeze the stable-instance screen from the baseline.
Also report final feasibility, objective/gap, and proven-optimal counts per run,
instances optimal in both current runs, and per-instance optimal frequencies
(0/2, 1/2, 2/2 versus 0/4 through 4/4). Use the final CSV's `is_optimal` flag;
reaching the reference objective alone is not an optimality proof. The final CSV
schema already includes this flag and is unchanged.

Saved-job capture remains opt-in for separate replay collection. The existing
directory mode captures one job per source, so it is not yet a representative
neighborhood collection. Keep it disabled for this scored benchmark.

## Previous checkpoints

The two follow-up checkpoints compare these policies:

| Policy | `9f0b8625` | Current checkpoint |
| ------ | ---------- | ------------------ |
| Initial propagation rounding | On, corrected cutoff exit | Off |
| Automatic warm-start selection | On | Off |
| Scheduling | 15 s stall threshold; 5–20 s regular turns | Same |
| Stronger root-gap cutoff | Off | Off |

Both include the initial-rounding exit fix. When enabled, initial rounding may
finish a descent only if its objective meets the exact GPU-session target in
reduced model coordinates. A point that passes row feasibility tolerances but
misses that target continues to PDLP within the same descent and budget.
`GpuSubmipFpDescent` reports
`initial_rounding_cutoff_rejected=1` when this happens; this host-side check uses
the already computed objective and adds no GPU synchronization.

The earlier cumulative checkpoints were:

Checkpoint 1 adds propagation-assisted initial rounding before the first PDLP
projection. It uses the existing 0.5 s rounding / 0.7 s total propagation limits
inside the job's wall budget and preserves the continuous PDLP seed. A feasible
initial rounded point meeting the exact cutoff returns immediately.
`CUOPT_GPU_SUBMIP_INITIAL_ROUNDING=1` enables this change for an A/B comparison;
it defaults to 0 in the current checkpoint.

Checkpoint 2 additionally selects the starting point from the supplied CPU seed,
a 25% incumbent blend, and the previous projection/FP solution for a matching
model. For a presolved child, the parent LP point and incumbent are mapped into
the child separately; CPUFJ retains its original starting point. The root GPU
consumer retains at most four host model histories. Reuse
requires exact matrix, RHS, objective, type, and source identity; each point is
clamped and rescored against the current rows and cutoff. No GPU state or dual
vectors are transferred between different jobs. CPU sub-MIP search is unchanged.
`CUOPT_GPU_SUBMIP_SELECT_WARM_START=1` enables selection and history updates;
it defaults to 0 in the current checkpoint.
Explicit cached-primal or incumbent-blend controls take precedence over automatic
selection. `GpuSubmipWarmStart` logs the choice, cache hit, score, and host cost.

Checkpoint 3 yields the full GPU at a completed-job boundary after 15 s of
sub-MIP work without material accepted progress (relative incumbent improvement
at least 1e-6). A regular-worker turn starts at 5 s and doubles to at most 20 s
after repeated stalls; material progress resets this backoff. The turn starts
after the regular worker resumes and is skipped if that worker has finished.
Root cancellation interrupts the wait. The pending job stays queued, CPU
workers remain independent, and every handoff drains the GPU stream first.
`CUOPT_GPU_SUBMIP_UNPRODUCTIVE_SECONDS=0` disables this policy; the threshold and
initial regular turn can be set with that variable and
`CUOPT_GPU_SUBMIP_REGULAR_SECONDS`. `GpuSubmipRegularTurn` logs actual handoffs.

Checkpoint 4 targets improvement of 0.1% of the gap between the current incumbent
and the frozen valid outer-root LP bound. The bound is converted into each
child's minimization objective coordinates, including offset and scale. Without
a valid bound, the target is a 1e-6 relative improvement in the user objective.
The existing strict-improvement margin and known objective lattice still apply.
This cutoff restricts only the GPU heuristic: it never changes the global bound,
CPU search, or candidate validation tolerances.
`CUOPT_GPU_SUBMIP_MIN_IMPROVE=0` restores the previous strict cutoff;
positive values set the gap fraction (0.001 at checkpoint 4, disabled by default
in the current checkpoint). `GpuSubmipJobStart` records
the proof bound, target, and cutoff basis. Existing snapshots do not store a proof
bound: replay uses the relative fallback unless `--objective-lower-bound` supplies
the logged bound in snapshot LP coordinates. The snapshot seed objective is not
used as a bound.

Use the existing benchmark exports at each checkpoint, with numeric suffixes
representing repetitions. Clear earlier policy overrides so the checkpoint's
defaults apply:

```bash
unset CUOPT_GPU_SUBMIP_INITIAL_ROUNDING CUOPT_GPU_SUBMIP_SELECT_WARM_START
unset CUOPT_GPU_SUBMIP_MIN_IMPROVE
unset CUOPT_GPU_SUBMIP_UNPRODUCTIVE_SECONDS CUOPT_GPU_SUBMIP_REGULAR_SECONDS
export CUOPT_GPU_SUBMIP=1
export CUOPT_SUBMIP_POOL_DIAGNOSTICS=1
export CUOPT_CONFIG_ID="$repetition_id"
export CUOPT_MAX_CONFIG="$n_iters"
export CUOPT_GPU_SUBMIP_CONFIG_FAMILY=checkpoint
```

## Earlier effort configurations

With `CUOPT_GPU_SUBMIP_CONFIG_FAMILY=effort`, IDs retain the earlier mapping:

| ID | Profile | Total projections per job | Descents per job | Wall cap per job |
| -- | ------- | ------------------------- | ---------------- | ---------------- |
| 0 | `single_descent` | Unlimited | 1 | 60 s |
| 1 | `p20_t60` | 20 | Unlimited | 60 s |
| 2 | `p100_t10` | 100 | Unlimited | 10 s |
| 3 | `p100_t60` | 100 | Unlimited | 60 s |

All four use an independent GPU worker, full-device CPUFJ handoff after strong
branching, and distance-only FP (`initial_alpha=0`). PDLP, rounding, and bound
propagation retain their respective 1 s, 0.5 s, and 0.7 s caps. The regular GPU
heuristic keeps the full GPU before the handoff. No green contexts are used.

A job stops when the first enabled effort limit is reached or the global solve
ends. Projection counts accumulate across restarted descents. The single-descent
profile finishes when that descent returns; the other profiles perturb and
restart early-returning descents while budget remains. Time limits are
cooperative, so an in-progress operation or cleanup can overrun the wall cap.
CPU sub-MIP completion does not cancel these independent jobs.

The comparisons isolate the benefit of restarts (0 vs. 1), a larger projection
budget (1 vs. 3), and a shorter time budget (2 vs. 3). The remaining wall budget
also affects internal rounding and PDLP allocations; a shorter cap can change
the search trajectory before the cap is reached.

## Benchmark setup

For the legacy effort sweep, use `n_iters=4` for one run per profile:

```bash
export CUOPT_GPU_SUBMIP=1
export CUOPT_SUBMIP_POOL_DIAGNOSTICS=1
export CUOPT_CONFIG_ID="$repetition_id"
export CUOPT_MAX_CONFIG="$n_iters"
export CUOPT_GPU_SUBMIP_CONFIG_FAMILY=effort
```

In the effort family, IDs are zero-based, must be less than `CUOPT_MAX_CONFIG`, and cycle modulo four.
For example, eight iterations run profiles 0, 1, 2, 3 twice. `CUOPT_MAX_CONFIG`
defaults to four and does not alter the RNG seed.

The effort profiles default to `mode=cpufj` and `lifetime=independent`. Explicit
`CUOPT_GPU_SUBMIP_MODE`, `CUOPT_GPU_SUBMIP_LIFETIME`,
`CUOPT_GPU_SUBMIP_MAX_PROJECTIONS`, `CUOPT_GPU_SUBMIP_MAX_DESCENTS`, and
`CUOPT_GPU_SUBMIP_SECONDS` override profile defaults. Clear old overrides for the
table above to apply:

```bash
unset CUOPT_GPU_SUBMIP_MODE CUOPT_GPU_SUBMIP_LIFETIME
unset CUOPT_GPU_SUBMIP_MAX_PROJECTIONS CUOPT_GPU_SUBMIP_MAX_DESCENTS
unset CUOPT_GPU_SUBMIP_SECONDS CUOPT_GPU_SUBMIP_INITIAL_ROUNDING
unset CUOPT_GPU_SUBMIP_SELECT_WARM_START
unset CUOPT_GPU_SUBMIP_UNPRODUCTIVE_SECONDS CUOPT_GPU_SUBMIP_REGULAR_SECONDS
unset CUOPT_GPU_SUBMIP_MIN_IMPROVE
```

`GpuSubmipFpSelection` identifies the family and profile;
`GpuSubmipEffectiveConfig` records the effective values after environment
overrides. Both use `CUOPT_LOG_INFO`. Existing worker utilization, session exit,
and candidate diagnostics remain available; the pool diagnostics export above
enables CPU pool utilization logging.

## Reproducing earlier configurations

Set `CUOPT_GPU_SUBMIP_CONFIG_FAMILY=fp` for the earlier ID mapping:
0=`baseline`, 1=`round_long`, 2=`pdlp_long`, 3=`distance`. This selects only the
FP settings; set the mode and lifetime explicitly when reproducing an older
paired-worker experiment.

Without `CUOPT_CONFIG_ID`, the checkpoint defaults still apply. The existing
named `CUOPT_GPU_SUBMIP_FP_CONFIG` selector overrides the FP preset; a numeric ID
takes precedence over that named selector. Select `effort` or `fp` explicitly
when reproducing the previous default worker settings.

## Saved-job replay

The replay executable shares the environment configuration parser. For an effort
profile, supply the saved job and `--full-gpu`; the selected independent mode and
budgets apply automatically:

```bash
CUOPT_CONFIG_ID=2 CUOPT_MAX_CONFIG=4 CUOPT_GPU_SUBMIP_CONFIG_FAMILY=effort \
  solve_GPU_SUBMIP_REPLAY cpu_submip.bin --full-gpu --seed 42 --job-id 1
```

Use the seed and job ID recorded with the snapshot. Explicit `--time-limit`,
`--max-projections`, `--max-descents`, and `--fp-config` override the environment
selection. `--independent` or `--continuous` explicitly selects the session mode.
The replay's `REPLAY_START` records the final command-line-adjusted settings.
