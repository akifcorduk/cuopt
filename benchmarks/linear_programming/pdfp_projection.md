# PDFP projection in cuOpt

This experiment changes the LP projection used by the feasibility pump. Enable it
on an internal `feasibility_pump_t` with `fp.config.use_pdfp_projection = true`.
The default remains the existing projection for controlled comparisons. No public
solver parameter is added. Rounding, propagation, perturbation, cycle detection,
stage scheduling, and the root LP solver retain their existing implementations.
The direct projector can also be exercised independently of the pump.

## Formulation

The projector minimizes

`objective_weight * c'x + distance_weight * sum_{j in I} |x_j - target_j|`

over the original row and variable bounds. It maintains a bounded distance dual
and a row dual, so general integer targets inside their bounds require no extra
variables or rows. Continuous variables have zero distance weight. The FP hook
preserves the existing alpha decay and normalization of the original objective
and distance term, and uses the existing time/tolerance selection.

The implementation follows the saddle-point updates, reflected Halpern iteration,
adaptive restarts, and primal-weight balancing in
[PDFP: GPU-Accelerated Feasibility Pump via PDHG](https://arxiv.org/pdf/2611.03629)
and the [MIT Lu Lab reference implementation](https://github.com/MIT-Lu-Lab/PDFP/tree/1cf4c06dd9242503edd2333a2002986b91223b3d).
It uses ten Ruiz passes followed by Pock–Chambolle scaling. Distance weights and
targets transform with the column scaling to preserve distance in original units.
Scaling is prepared once; iterations and warm-start state reside on the GPU.

cuOpt adaptations:

- Handle lower, upper, equality, ranged, and free rows directly with the interval
  support-function proximal update.
- Include the pump's existing linear objective in the primal update.
- Use the certified bound `sqrt(||A||_1 * ||A||_inf)` for the scaled matrix norm
  instead of power iteration. This can be more conservative on irregular matrices.
- Check maximum absolute row violation in original units. This is stricter than
  the paper's normalized primal residual and can affect timing comparisons.
- Use the negative cross term in the fixed-point metric for the positive row-dual
  sign convention, bounded primal-weight adaptation, and bounded PDHG points for
  warm starts.

Convergence refers to the continuous projection problem. Every FP result still
passes through cuOpt's existing feasibility validation. Time or iteration limits
do not establish feasibility or optimality. The projector must be recreated after
matrix/shape changes; the FP reset and resize paths discard its cached state.

## Reproduction

Build a CUDA-enabled cuOpt test configuration, including `NUMOPT_INTERNAL_TEST`.
The unit suite is `PdfpProjection.*` (disabled benchmarks are excluded by default).
Run the large synthetic tests explicitly:

```bash
NUMOPT_INTERNAL_TEST --gtest_also_run_disabled_tests \
  '--gtest_filter=PdfpProjection.DISABLED_Benchmark*GeneralIntegers'
```

Run a real instance with:

```bash
PDFP_BENCHMARK_MPS=/absolute/path/instance.mps \
PDFP_BENCHMARK_TIME_LIMIT=30 \
NUMOPT_INTERNAL_TEST --gtest_also_run_disabled_tests \
  --gtest_filter=PdfpProjection.DISABLED_BenchmarkMps
```

The benchmark builds a target by rounding a bounded root-LP iterate. It compares
the direct projector with an equivalent epigraph LP solved by cuOpt PDLP, then
changes the target and repeats. Logs include matrix size, setup and solve time,
convergence, distance objective, and independently computed row/bound violations.
The epigraph comparison uses Stable2, as the existing FP projection does, but
rebuilds and cold-solves each target; the direct solver retains its state for the
second target. These are isolated projection measurements, not an end-to-end FP
trajectory comparison. Warm direct timings should not be presented as a matched
warm-start speedup over the existing FP.
The synthetic problems have coupled two-variable blocks with known optima;
they test scale and correctness but do not represent arbitrary MIP difficulty.
Compare timings only alongside convergence and residual quality.
