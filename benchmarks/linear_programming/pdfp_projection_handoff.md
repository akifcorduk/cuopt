# PDFP projection: agent handoff

## Status and scope

This is an experimental, opt-in CUDA projection implementation, based on cuOpt
`af9939e4a` (`Fix mod-2 cut integrality tolerance`), on branch `pdfp-projection`.
The requested scope is **projection only**. The existing projector is still the
default. Correctness tests pass, but the measured large real instances do not
establish a speedup; do not promote this to the production default on this evidence.

The user requested the projection ideas from
[PDFP: GPU-Accelerated Feasibility Pump via PDHG](https://arxiv.org/pdf/2611.03629)
and [MIT-Lu-Lab/PDFP](https://github.com/MIT-Lu-Lab/PDFP/tree/1cf4c06dd9242503edd2333a2002986b91223b3d).
The reference revision is `1cf4c06dd9242503edd2333a2002986b91223b3d`; the relevant
Julia files are `src/pdfp.jl`, `src/pdfp_kernels.jl`, and `src/preconditioner.jl`.
This is a cuOpt adaptation of the mathematics, not a literal reproduction of the
whole reference algorithm.

Unchanged: rounding, propagation, perturbation, outer FP cycle handling, root LP,
and stage scheduling. In particular, the existing binary-first schedule remains;
a new single-stage FP was not implemented. The paper's rounding and concurrent
trajectory mechanisms were not imported. No public CLI/Python solver setting was
added. Build artifacts, datasets, local logs, and unrelated user changes are not
part of this source change.

## What changed and why

The existing FP already uses GPU PDLP. It represents an interior general-integer
distance term with one auxiliary variable and two rows. Binary targets and
targets at their variable bounds already have a linear distance objective and
need no auxiliary rows. The proposed improvement is therefore to remove
**epigraph matrix augmentation**, not to move a CPU solver onto the GPU.

The new projector solves the same continuous problem:

```text
minimize  beta * c'x + lambda * sum(j in I) |x[j] - target[j]|
subject to L <= A*x <= U and ell <= x <= u.
```

Here `I` is the integer-variable mask; continuous variables carry no distance
penalty. The key identity is
`lambda*|x-target| = max(|y|<=lambda) y*(target-x)`. Maintaining a bounded dual
`y` replaces the extra epigraph rows/columns. A separate row dual `z` handles
equalities, inequalities, ranged rows, and free rows through interval projection.

Ignoring diagonal scaling, one PDHG step is:

```text
x_next = clip(x - tau*(beta*c + A'*z - y), ell, u)
x_bar  = 2*x_next - x
y_next = clip(y + sigma*(target - x_bar), -q, q)
v      = z + sigma*A*x_bar
z_next = v - sigma*clip(v/sigma, L, U)
```

`q[j]` is `lambda` for integer variables and zero otherwise. Reflected Halpern
updates combine the anchor with twice the PDHG point minus the current iterate;
adaptive inner restarts and primal/dual step balancing accelerate this process.
These inner restarts are unrelated to the unchanged outer FP cycle logic.

The original CSR matrix and its transpose are reused. CUDA kernels implement
scaled one-warp-per-row SpMV, primal/dual updates, and Halpern updates. RMM holds
GPU state. Each inner iteration has two SpMVs, with additional SpMVs and reductions
at residual checkpoints. Setup performs ten Ruiz infinity-norm equilibration
passes and one Pock–Chambolle L1 pass on the host. With `x = D_c*x_scaled`, the
target is divided by `D_c` and distance/objective coefficients are multiplied by
`D_c`, preserving the objective in original units.

### Deliberate deviations from the reference

- The step size uses the conservative certified spectral bound
  `sqrt(||A_scaled||_1 * ||A_scaled||_inf)`, not power iteration;
  `eta = 0.998 / sqrt(1 + bound^2)`. This may slow irregular matrices.
- Primal stopping uses the maximum absolute row violation in original units,
  at most `0.1*tolerance`, rather than the reference's normalized primal residual.
  Dual residual and relative gap must also be at most `tolerance`, with finite
  metrics and at least one iteration. The dual residual uses scaled-space
  quantities, not an original-unit maximum dual violation.
- Primal-weight adaptation is simplified and bounded. The fixed-point norm uses
  the negative cross term consistent with this code's positive row-dual convention.
- Only the bounded PDHG point `(x,y,z)` persists between projection calls.
  Primal weight, anchor/history, and balancing integral are reinitialized per call.
  Reflected iterates are not used as the returned point or warm start.
- The cuOpt objective blend and general ranged-row handling are included.
  An explicit zero branch in the row-dual update avoids roundoff-sized wrong-sign
  duals interacting with infinite row bounds.

## Code map

Paths below are relative to the repository root.

| File | Responsibility |
|---|---|
| `cpp/src/mip_heuristics/local_search/feasibility_pump/pdfp_projection.cuh` | Internal projector, settings, result, and lifecycle contract |
| `cpp/src/mip_heuristics/local_search/feasibility_pump/pdfp_projection.cu` | Scaling, CUDA updates, residuals, restarts, warm-start storage |
| `cpp/src/mip_heuristics/local_search/feasibility_pump/feasibility_pump.cu` | Optional dispatch inside `linear_project_onto_polytope`; discard cache on FP reset/resize |
| `cpp/src/mip_heuristics/local_search/feasibility_pump/feasibility_pump.cuh` | Default-off flag, cached projector, last result |
| `cpp/src/mip_heuristics/CMakeLists.txt` | Register projector compilation |
| `cpp/tests/mip/pdfp_projection_test.cu` | Twelve enabled projector tests; three disabled benchmark entries; epigraph comparator |
| `cpp/tests/mip/pdfp_integration_test.cu` | FP dispatch, objective mixing, alpha schedule, dimension preservation |
| `cpp/tests/internal/CMakeLists.txt` | Register both test translation units |
| `benchmarks/linear_programming/pdfp_projection.md` | Design and portable reproduction overview |
| `benchmarks/linear_programming/pdfp_projection_results.md` | Detailed measurements and interpretation |
| `benchmarks/linear_programming/pdfp_projection_handoff.md` | This handoff |

## Enabling and lifecycle

For an existing internal `feasibility_pump_t` instance:

```cpp
fp.config.use_pdfp_projection = true;
// Continue using the existing FP projection entry point.
// Diagnostics from the last direct projection are in fp.last_pdfp_result.
```

For projection-only experiments, construct
`mip::pdfp_projection_t<int, double> projector(problem)` and call
`projector.project(device_target, device_output, settings)`. Target and output
may alias. The header lists the defaults: tolerance `1e-4`, time limit `10s`,
iteration limit `500000`, check interval `200`, distance weight `1`, objective
weight `0`. Float instantiation is conditional on the existing MIP build macro;
validation reported here used double precision.

The FP hook preserves alpha decay (except for longer LP runs), and uses
`lambda=(1-alpha)/sqrt(number_of_integer_variables)` and
`beta=alpha/||c||_2`, with guards for empty integer sets and zero objective norm.
It retains the existing time/tolerance selection, copies `last_projection`, and
calls the existing `solution.compute_feasibility()` independently of the direct
solver's convergence flag.

Important contracts:

- The `problem_t` and its handle must outlive the projector. Do not mutate the
  model concurrently with a projection.
- Recreate the projector after any matrix coefficient, structure, or dimension
  change. Only dimension mismatches are detected; same-shape coefficient edits
  silently invalidate cached scaling and the step-size bound.
- Bounds, linear objective, and integer masks are read afresh each projection.
  Warm variables and distance duals are clipped to the new bounds/mask.
- `projector.reset()` requests a cold start but does **not** rebuild scaling.
  The FP's own reset/resize destroys the cached projector, so construction and
  scaling run again on the next enabled projection.
- Targets must be finite and settings valid. Zero time/iteration budgets return
  a bounded point without claiming convergence. A limit-exhausted point may
  violate rows and is not an infeasibility certificate.
- Setup, target validation, and final synchronization can make end-to-end time
  exceed the iteration-loop budget. Account for setup separately when benchmarking.

## Validation already completed

Machine: NVIDIA L40S, 46,068 MiB; CUDA compiler 12.9; GCC 13; Release; SM 89;
double precision; routing and gRPC disabled. Existing environment and caches
were reused; no system packages were installed.

- `NUMOPT_INTERNAL_TEST` and `cuopt_cli` built successfully.
- Thirteen focused GPU tests passed: twelve `PdfpProjection` tests and one
  `PdfpIntegration` test. The normal CMake target passed through CTest.
- Tests cover analytic optima and equivalent LPs, general integers with continuous
  completion, ranged/equality rows, weighted objective, zero distance weight,
  no integers, scaling, changed targets, aliasing, reset, integer-mask changes,
  empty matrices, infeasible models, and time/iteration limits.
- Both large synthetic benchmarks passed. Real projections were measured on
  `supportcase19`, `bab2`, and `gen-ip002`.
  Synthetic assertions check the cold direct result against its analytic optimum;
  changed-target and real-instance measurements are primarily diagnostic. A passing
  benchmark process does not imply every reported projection converged.
- Compute Sanitizer memcheck passed five selected tests with zero errors.
- The default CLI smoke test `swath1.mps` finished at its 15-second limit with
  incumbent `379.0713` and gap `5.14%`. This exercises the **default**, not PDFP,
  projector. The focused FP integration test exercises PDFP dispatch.
- The full repository test suite and end-to-end PDFP-enabled FP/MILP performance
  suite have not been run.

Initial focused tests, sanitizer, and benchmarks used a temporary focused test
executable linked against this worktree's freshly built `libcuopt_static.a` while
the remaining test harness compiled. The registered `NUMOPT_INTERNAL_TEST` target
was subsequently rebuilt and all thirteen focused tests passed through CTest.
Use the normal target below for future work; the temporary executable is unnecessary.

### Benchmark outcome

Full setup/solve times and independently measured violations are in
[the results report](pdfp_projection_results.md). Selected cold-target solve times:

| Instance | Original rows / columns / nonzeros | Direct | Epigraph PDLP | Interpretation |
|---|---|---:|---:|---|
| Synthetic 100k | 100k / 100k / 200k | 0.0751s | 0.1755s | Both converged on easy structured blocks |
| Synthetic 1M | 1M / 1M / 2M | 0.6728s | 0.4270s | Both converged; direct slower despite smaller matrix |
| supportcase19 | 10,713 / 1,429,098 / 4,287,094 | 30.1624s | 8.1235s | Direct did not converge; PDLP reported convergence |
| bab2 | 17,245 / 147,912 / 2,027,726 | 30.0593s | 4.0510s | Direct did not converge; PDLP reported convergence |
| gen-ip002 | 24 / 41 / 922 | 0.0054s | 0.0276s | Cold target already feasible; changed target is the useful case |

The synthetic million-variable epigraph grows to 3M rows, 2M columns, and 6M
nonzeros. Its cold analytic optimum is 1,500,000; direct returned 1,500,000.427
and the reference 1,500,000.005. Structural savings are demonstrated, but a general
speed benefit is not.

Do not overinterpret these numbers:

- They are single samples; some were collected while the CPU test harness was
  compiling, without a competing GPU solver. No controlled statistical study.
- Both use numeric tolerance `1e-5`, but the direct row threshold is absolute
  `1e-6`, whereas PDLP uses different normalized criteria. Convergence flags do
  not mean identical absolute accuracy. Compare objective and independent residuals.
- The direct second target reuses state, while each reference epigraph LP is
  rebuilt and cold-started. Existing production FP can warm-start; this benchmark
  is **not** a matched warm-start comparison.
- All cold `supportcase19` targets happened to be at bounds; its reference needed
  no augmentation. `bab2` is binary and likewise has no epigraph matrix savings.
- Changed-target `supportcase19` points did not meet requested feasibility for
  either method. Their distances must not be compared as feasible optimal values.
- These are component measurements, not FP trajectory or overall MILP speedups.

## Resume on this machine

```bash
export PDFP_REPO=/home/acoerduek/codex/agent-sandbox/cuopt-pdfp-projection
export CUOPT_ENV=/home/acoerduek/cuopt/.cuopt_env
export PDFP_BUILD="$PDFP_REPO/cpp/build_agent_tests"
export CONDA_PREFIX="$CUOPT_ENV"
export PATH="/usr/bin:$CUOPT_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$PDFP_BUILD:$CUOPT_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
cd "$PDFP_REPO"
git status --short
```

The build tree already exists. Incremental build and focused tests:

```bash
"$CUOPT_ENV/bin/cmake" --build "$PDFP_BUILD" --target NUMOPT_INTERNAL_TEST cuopt_cli -j2
GTEST_FILTER='PdfpProjection.*:PdfpIntegration.*' \
  "$CUOPT_ENV/bin/ctest" --test-dir "$PDFP_BUILD" -R '^NUMOPT_INTERNAL_TEST$' --output-on-failure
```

For a new test build, read the local cuOpt build skill and use its cache-aware
configurator rather than deleting a build directory or downloading dependencies
again:

```bash
/home/acoerduek/.codex/skills/cuopt-build-and-test/scripts/prepare_test_build.sh \
  --source-repo "$PDFP_REPO" --build-dir "$PDFP_BUILD"
"$CUOPT_ENV/bin/cmake" -S "$PDFP_REPO/cpp" -B "$PDFP_BUILD" \
  -DSKIP_ROUTING_BUILD=ON -DSKIP_GRPC_BUILD=ON \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$CUOPT_ENV/bin/ccache" \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$CUOPT_ENV/bin/ccache"
```

Sanitizer subset and opt-in large tests:

```bash
"$CUOPT_ENV/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  "$PDFP_BUILD/tests/internal/NUMOPT_INTERNAL_TEST" \
  --gtest_filter='PdfpProjection.GeneralIntegerWithContinuousCompletion:PdfpProjection.EmptyMatrixUsesVariableBounds:PdfpProjection.ZeroTimeLimitReturnsBoundedPoint:PdfpProjection.IntegerMaskChangesBetweenWarmProjections:PdfpIntegration.*'

"$PDFP_BUILD/tests/internal/NUMOPT_INTERNAL_TEST" --gtest_also_run_disabled_tests \
  --gtest_filter='PdfpProjection.DISABLED_Benchmark*GeneralIntegers'

PDFP_BENCHMARK_MPS=/home/acoerduek/cuopt/datasets/miplib2017/supportcase19.mps \
PDFP_BENCHMARK_TIME_LIMIT=30 \
  "$PDFP_BUILD/tests/internal/NUMOPT_INTERNAL_TEST" --gtest_also_run_disabled_tests \
  --gtest_filter=PdfpProjection.DISABLED_BenchmarkMps
```

Repeat the MPS command with `bab2.mps` and `gen-ip002.mps`. The valid local MIPLIB
files are under `/home/acoerduek/cuopt/datasets/miplib2017/`. Some old gzip files
under `datasets/mip/` are empty/bad downloads; do not use them without validation.
For CLI checks, use `ldd` to verify `libcuopt.so` resolves from `PDFP_BUILD`, not
the installed environment's older library.

Local logs are `/tmp/cuopt-pdfp-{unit,synthetic,supportcase19,bab2,gen-ip002,ctest,memcheck,cli}.log`,
`/tmp/cuopt-pdfp-unit.xml`, and `/tmp/cuopt-pdfp-final-build.log`. These are ephemeral
local artifacts, not committed; the durable measurement summary is the results
report. Do not stage `cpp/build_agent_tests/` or unrelated edits in the original
checkout `/home/acoerduek/cuopt`.

## Recommended next work, still projection-only

1. Establish comparable stopping/accuracy and matched warm starts before making
   performance claims. Keep independent original-unit residual/objective checks.
2. Profile SpMV, kernel launches, checkpoints, host setup, and conservative step
   sizes separately. Compare cuSparse/cuOpt SpMV with the simple custom kernel;
   investigate the reference norm estimate only with a justified safety strategy.
3. Compare restart/primal-weight/warm-state policies against the pinned reference,
   changing one projection mechanism at a time. Do not modify FP rounding or
   stage scheduling to hide projector weaknesses.
4. Add difficult general-integer instances with interior targets, ill-conditioned
   rows, and repeated targets. Rerun multiple trials on an idle machine after warmup.
5. Extend lifecycle/numerical regression coverage, particularly changed bounds,
   matrix invalidation, extreme scaling, and asynchronous failure reporting, before
   considering production use. A matrix-version guard would avoid silent reuse
   after same-shape coefficient edits.
6. Keep `use_pdfp_projection=false` by default. Public API exposure or any change
   to the rest of FP requires a separate user decision and validation effort.
