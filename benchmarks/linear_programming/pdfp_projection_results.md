# PDFP projection validation

Local experiment on an NVIDIA L40S (46,068 MiB), CUDA 12.9, GCC 13, Release,
double precision, based on cuOpt `af9939e4a`. The implementation is on the local
`pdfp-projection` branch. See [implementation and reproduction](pdfp_projection.md).

## Correctness

All 13 focused GPU tests passed: 12 direct-projector cases and one FP integration
case. The tests cover independent analytic optima and epigraph-LP comparisons,
general integers, continuous completion, ranged/equality rows, objective blending,
scaling, warm target changes, mask changes, aliasing, reset, empty matrices,
infeasible models, and time/iteration limits. Both explicitly enabled large
synthetic tests passed as well.

The normal `NUMOPT_INTERNAL_TEST` target was rebuilt and the same 13 tests passed
through CTest (`GTEST_FILTER='PdfpProjection.*:PdfpIntegration.*'`). CUDA
Compute Sanitizer memcheck passed five selected boundary, mask-change, and FP
integration tests with **zero errors**. New source files pass clang-format 20
checks, and `git diff --check` passes.

The existing cuOpt environment and dependency caches were reused via the local
cuOpt build skill. The Release build disabled routing and gRPC and used two build
jobs. Final build command, from the worktree:

```bash
cmake --build cpp/build_agent_tests --target NUMOPT_INTERNAL_TEST cuopt_cli -j2
```

The CLI was verified with `ldd` to load this worktree's `libcuopt.so`. One default
CLI smoke fixture, `swath1.mps`, exited successfully at its 15-second limit with
an incumbent objective of 379.0713 and a 5.14% gap. This smoke check uses the
unchanged default projector; the new path is exercised by the FP integration test.
The full repository test suite was not run.

## Projection measurements

Times below are single-run wall-clock measurements in seconds. Setup is separate
from solve. The reference is an equivalent epigraph LP using cuOpt Stable2 PDLP.
The benchmark uses tolerance `1e-5` and a 30-second limit per projection. The direct
solver requires maximum absolute row violation at most `1e-6`; PDLP uses its own
normalized criteria. Consequently, convergence flags and times are not comparisons
at identical absolute accuracy. Row violations below are independently measured
in original units. Bound violations were zero apart from roundoff up to 2.5e-15
in one reference output.

| Instance / target | Direct setup | Direct solve | Direct converged | Direct max row violation | Epigraph setup | Epigraph solve | Epigraph converged | Epigraph max row violation |
|---|---:|---:|:---:|---:|---:|---:|:---:|---:|
| Synthetic 100k / cold | 0.0191 | 0.0751 | yes | 7.24e-7 | 0.0267 | 0.1755 | yes | 1.08e-8 |
| Synthetic 100k / changed | reused | 0.0881 | yes | 3.07e-8 | 0.0460 | 0.0694 | yes | 4.96e-9 |
| Synthetic 1M / cold | 0.2186 | 0.6728 | yes | 7.24e-7 | 0.3560 | 0.4270 | yes | 9.56e-9 |
| Synthetic 1M / changed | reused | 0.8877 | yes | 3.07e-8 | 0.2368 | 0.3619 | yes | 5.09e-5 |
| supportcase19 / cold | 0.2439 | 30.1624 | no | 2.86e-6 | 0.0572 | 8.1235 | yes | 3.17e-6 |
| supportcase19 / changed | reused | 30.1847 | no | 5.70e-2 | 0.1483 | 30.0785 | no | 1.87e-3 |
| bab2 / cold | 0.0730 | 30.0593 | no | 3.89e-3 | 0.0101 | 4.0510 | yes | 1.80e-3 |
| bab2 / changed | reused | 30.0458 | no | 2.66e-2 | 0.0048 | 3.6258 | yes | 4.30e-4 |
| gen-ip002 / cold | 0.00006 | 0.0054 | yes | 0 | 0.00008 | 0.0276 | yes | 0 |
| gen-ip002 / changed | reused | 0.0466 | yes | 2.29e-7 | 0.00008 | 0.1197 | yes | 5.52e-5 |

The synthetic models have only general integers and independent coupled
two-variable blocks. For the million-variable case, the original matrix has
1M rows, 1M columns, and 2M nonzeros; the epigraph has 3M rows, 2M columns, and
6M nonzeros. The analytic cold optimum is 1,500,000; the direct result was
1,500,000.427 and the reference result 1,500,000.005. The synthetic models measure
scale and correctness on easy structured problems, not general MIP difficulty.

`supportcase19` has 10,713 rows, 1,429,098 integer columns (117,806 general
integers), and 4,287,094 nonzeros. All cold integer targets happened to be at bounds,
so the cold reference needed no epigraph augmentation. After changing targets,
the reference grew to 128,515 rows, 1,487,999 columns, and 4,522,698 nonzeros.
Cold distance objectives were 398.9917423 (direct) and 398.9886522 (reference).
The changed-target outputs were not feasible to the requested tolerance; their
distance objectives must not be treated as comparable feasible optima.

`bab2` has 17,245 rows, 147,912 binary variables, and 2,027,726 nonzeros. Neither
direct projection met its strict convergence test within the budget. `gen-ip002`
has 24 rows, 41 general integers, and 922 nonzeros. Its cold target was already
feasible (distance zero); the changed target was nontrivial, and both methods
found distance approximately 1.9788.

Direct changed-target projections retain their state; reference changed-target
projections are rebuilt and cold-started. This is not a full FP trajectory or a
matched warm-start comparison. Measurements were collected while the remaining
CPU test harness was compiling, with no competing GPU solver. They are diagnostic
samples, not a statistically controlled speedup study.

The implementation passes correctness tests and avoids matrix augmentation, but
these measurements do not establish a general performance improvement. It remains
opt-in; the existing production projection is still the default.

Raw local logs: `/tmp/cuopt-pdfp-unit.log`, `/tmp/cuopt-pdfp-synthetic.log`,
`/tmp/cuopt-pdfp-supportcase19.log`, `/tmp/cuopt-pdfp-bab2.log`, and
`/tmp/cuopt-pdfp-gen-ip002.log`. The unit-test XML is
`/tmp/cuopt-pdfp-unit.xml`. Additional logs: `/tmp/cuopt-pdfp-final-build.log`,
`/tmp/cuopt-pdfp-ctest.log`, `/tmp/cuopt-pdfp-memcheck.log`, and
`/tmp/cuopt-pdfp-cli.log`.
