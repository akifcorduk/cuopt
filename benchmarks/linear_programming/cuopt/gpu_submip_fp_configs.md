# GPU sub-MIP FP benchmark checkpoints

The default family is now `checkpoint`: every numeric ID repeats the chosen
config 0 policy (one independent distance-FP descent, unlimited projections,
60 s wall cap). IDs no longer select the old effort variants unless the family
is explicitly set to `effort`. The solver seed is unchanged by the ID.

Checkpoint 1 adds propagation-assisted initial rounding before the first PDLP
projection. It uses the existing 0.5 s rounding / 0.7 s total propagation limits
inside the job's wall budget and preserves the continuous PDLP seed. A feasible
initial rounded point returns immediately. `CUOPT_GPU_SUBMIP_INITIAL_ROUNDING=0`
disables this change for an A/B comparison; it defaults to 1 in this family.

Checkpoint 2 additionally selects the starting point from the supplied CPU seed,
a 25% incumbent blend, and the previous projection/FP solution for a matching
model. For a presolved child, the parent LP point and incumbent are mapped into
the child separately; CPUFJ retains its original starting point. The root GPU
consumer retains at most four host model histories. Reuse
requires exact matrix, RHS, objective, type, and source identity; each point is
clamped and rescored against the current rows and cutoff. No GPU state or dual
vectors are transferred between different jobs. CPU sub-MIP search is unchanged.
`CUOPT_GPU_SUBMIP_SELECT_WARM_START=0` disables selection and history updates.
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

The handoff overhead checkpoint retains initial rounding, automatic warm-start
selection, and the same 15 s stall threshold with 5/10/20 s regular-worker turns.
Initial-incumbent polishing LPs and regular propagation now honor a pending
handoff request. Imports yield between candidates after validating the polished
point or falling back to the supplied incumbent. GPU ownership still transfers
only after all streams drain.

`GpuSubmipHandoff` INFO events separate request-to-`safe_point` response,
`safe_point`-to-`drain_begin` queued-incumbent ingestion, and
`drain_begin`-to-`drain_end` CUDA drain time. `GpuSubmipInitialPolish` records
per-candidate polishing duration and whether a handoff was requested at exit.
These wall times measure ownership and response latency; they do not measure
GPU utilization or CPU thread idleness. No FP effort limit or search policy
changes in this checkpoint.

Use the existing benchmark exports at each checkpoint, with numeric suffixes
representing repetitions:

```bash
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
CUOPT_CONFIG_ID=2 CUOPT_MAX_CONFIG=4 \
  solve_GPU_SUBMIP_REPLAY cpu_submip.bin --full-gpu --seed 42 --job-id 1
```

Use the seed and job ID recorded with the snapshot. Explicit `--time-limit`,
`--max-projections`, `--max-descents`, and `--fp-config` override the environment
selection. `--independent` or `--continuous` explicitly selects the session mode.
The replay's `REPLAY_START` records the final command-line-adjusted settings.
