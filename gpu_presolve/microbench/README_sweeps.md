# sweeps: micro-benchmark of the bound-propagation sweep kernels

Stage 0 tool (DESIGN.md sections 2.3, 2.4, 13). It times the kernels of one propagation round on a
real matrix, in the variants the design has to choose between, and the same computations on one
CPU thread. The numbers decide the activity layout (paired against split arrays), the scheduling of
rows, the cost of compensated sums, the frontier size below which the host engine is cheaper, and
the rebuild method.

Files: `sweeps.cu` (everything on the device, the driver, the checks), `sweeps_cpu.cpp` (CPU
reference, compiled twice), target `sweeps` in `CMakeLists.txt`.

## Build and run

On the dev node, from the worktree root (`cuopt exec '<command>'` from the Mac):

```
B=/home/scratch.acoerduek_sw/gpu_presolve_deps/build/microbench
cmake -S gpu_presolve/microbench -B $B -G Ninja
ninja -C $B -j 8 sweeps
$B/sweeps --synthetic 200000 100000 8 $B/sweeps_synthetic.json --quick
$B/sweeps <matrix.csr> <out.json>
```

```
sweeps <matrix.csr | --synthetic rows cols nnz_per_row> <out.json> [--quick] [--reps n] [--budget seconds]
```

- `<matrix.csr>`: the `GPRSCSR1` dump written by `baseline_tool --dump-csr` (format in
  `../baseline/README.md`). The loader aborts on a malformed file, on an explicit zero entry and on
  more than 2^31 - 1 nonzeros.
- `--synthetic`: a generated matrix with a fixed seed. Row lengths are uniform in
  1 .. 2 * nnz_per_row - 1; three rows have cols/2, cols/8 and cols/32 entries; every 1000th row
  has 65 to 464 entries; a few rows are empty. 12% of the columns have an infinite upper bound, 5%
  an infinite lower bound; half of the rows have integer coefficients, the others fractional ones
  with 1% entries near 1e6.
- Default: 3 warm-up runs and 50 timed repetitions per measurement. A measurement stops early,
  after at least 5 repetitions, once it has used `--budget` seconds (default 30); the JSON records
  the repetitions actually made. `--quick`: 1 warm-up, 5 repetitions, 2 s budget; for functional
  checks only.
- Exit code 0: all checks passed; 1: a check failed (the JSON is still written); 2: usage error or
  the input file cannot be opened (message on stderr, nothing written).
- Compile flags on the dev node: device code `-O3 -arch=native --fmad=false`; CPU reference the
  conda environment's `CXXFLAGS` (`-march=nocona -mtune=haswell ...`) plus
  `-O3 -ffp-contract=off`, and `-mfma` for the second build.

A time is the wall clock from before the first launch to the return of `cudaDeviceSynchronize`, so
launch and read-back costs are included. Reported: median, 10th and 90th percentile in
milliseconds, and the median in nanoseconds per nonzero. The nonzero count is the matrix's, except
for frontier rows, where it is the number of nonzeros in the listed rows. When a frontier list
holds only empty rows that figure is undefined: `null` in the JSON, `inf` on stdout.

## What is measured

Common definitions. A term a * bound counts as infinite when |a * bound| > 1e20 or is not finite;
the others are summed. Column bounds are stored as one {lower, upper} pair per column, row sides
likewise. Offsets are 64 bit, indices 32 bit. Contraction is off on the device (`--fmad=false`)
and on the host (`-ffp-contract=off`); compensated code calls `fma` explicitly.

### 1. Row activity (`activity`)

For every row: finite sum and number of infinite terms, for the minimum and for the maximum
activity. Variant name `scheduling/layout/count/arithmetic`:

| Letter | Meaning |
|---|---|
| T | One thread per row over the CSR as given, long rows included |
| B | Rows binned by length (0-2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-8192, above 8192) in a second, sorted copy of the CSR. Up to 64 entries: one launch; a row of class c is folded by 2^c adjacent lanes (at most two entries per lane) and a shuffle tree. 65 to 8192: one 256-thread block per row. Above 8192: one block per chunk of 8192 entries, then one block per row over the partial results |
| H | The same sorted copy and the same treatment above 64 entries; one thread per row up to 64 (the scheme of DESIGN.md 2.4) |
| C | One 256-thread block per row with `cub::BlockReduce` over the CSR as given, whatever the row length (current cuOpt) |
| P | Output per side as one {double sum, double count} pair (cuOpt PR 2372) |
| S | Output per side as a double array and a `uint16_t` count array |
| R | Counts reduced like the sums (a second `cub::BlockReduce` pair, or shuffles) |
| A | Each thread counts during its scan and does one integer atomic add on a shared counter if its count is nonzero |
| plain | Double sums |
| comp | (hi, lo) sums: product error from `fma`, addition error from two-sum; lo goes to a separate array |

T has no count variant. Things to know when reading the numbers:

- T, and H up to 64 entries, are bound by the latency of one device thread, not by throughput.
  One thread takes about 0.5 to 0.6 microseconds per entry on the H200 (0.16 for a bare sum of
  doubles; separate one-thread kernel, dev node, 2026-11-26). The time of T is therefore at least
  the longest row times that figure, and its ns per nonzero is not a constant of the kernel; H
  and the frontier kernel have a floor of up to 64 entries times that figure per sweep.

- B and H write through the row map to the original row positions, so all variants fill the same
  arrays and the candidate kernel can follow any of them. A state stored in binned order would
  save that scattered write.
- In layout S bit 15 of the count is set when lo is not zero, and the count saturates at 32767.
  The candidate kernel uses the bit to skip the lo load. The design's guard (hi alone passes the
  error bound) would skip more often, so the gated numbers are an upper bound for the lo traffic.
- In block kernels the sums are reduced by `cub::BlockReduce`; compensated ones with a two-sum
  operator. Every tree is fixed by thread position. No floating-point atomics anywhere.

### 2. Bound candidates (`candidate`)

For every column the tightest implied lower and upper bound over its rows (CSC), by
(side - (activity - a * own bound)) / a; no infinite term: every column; one: only the column that
holds it; more: none. Integer columns are rounded with 1e-6. The activity of a side is only loaded
when that row side is finite. Variant name `scheduling/layout/arithmetic`:

| Part | Meaning |
|---|---|
| B | Columns in the length classes of the rows, through a list sorted by class (no second copy of the CSC): lane groups up to 64 entries, one block per column up to 8192, chunks above; min/max by shuffles and `cub::BlockReduce` |
| H | The same above 64 entries; one thread per column up to 64 |
| P/plain, P/comp | Reads the 16-byte pair, then (comp) lo |
| S/plain | Reads the count; the sum only when the count allows a candidate |
| S/comp/gated | As S/plain; lo only when bit 15 of the count is set |
| S/comp/always | As S/plain; lo always |

No error guard is evaluated and no candidate is suppressed.

### 3. Full round (`round`)

Activities followed by candidates in one timed unit, for each arithmetic: the fastest activity
variant with the faster candidate scheduling of the same layout, and the slowest with the slower
(S/comp/gated for layout S). The choice is made from the medians of sections 1 and 2 in the same
run, so the names in this section can differ between runs and matrices.

### 4. Frontier (`frontier`)

A random 10%, 1%, 0.1% and 0.01% of the rows (fixed seed, at least one row) as a `uint8_t` flag
array. Timed: `cub::DeviceSelect::Flagged` over the rows up to 64 entries and over those of 65 to
8192, one copy of the two counts to the host, then the H/S/A kernels over the two lists.
"compaction only" is the first part alone. Rows above 8192 entries are not listed: when the
matrix has any, their chunk kernels are launched over all of them and test the flag.

### 5. CPU reference (`cpu`)

One thread, split arrays, the same arithmetic in the same order as one device thread.

- `activity`: all rows, or the rows of a frontier list in ascending order.
- `candidates by row`: for each row, the candidates of its entries merged into the per-column
  bounds; this is what a host engine does after recomputing a row. Full matrix and frontier lists.
- `candidates by column`: over the CSC like the device kernel; full matrix only.
- `plain`, `comp` (built without `-mfma`: `std::fma` is a libm call), `comp-fma` (built with
  `-mfma`). On a machine that is not x86-64 the two builds are the same.
- Frontier lists are timed twice. `warm`: the list is repeated as it is, so a short list is timed
  out of the core's private caches. `cold`: those caches are evicted (a 32 MB read) before every
  repetition, so rows and bounds come from L3 or memory. A host engine that returns to a row after
  other work lies between the two. Full sweeps are not evicted.

### 6. Rebuild (`rebuild`)

20% of the rows and 20% of the columns are flagged dead (independent draws, fixed seed).

| Name | Steps |
|---|---|
| compact CSR | Scans of the two flag arrays to new ids; a keep flag per entry; scan; scatter; new row offsets; three scalars copied to the host |
| CSC: stable radix sort | Row index and position per entry; column histogram and scan; `cub::DeviceRadixSort::SortPairs` on (column, position) over the bits the column count needs; gather |
| CSC: histogram + scan + scatter, unordered | Column histogram and scan; scatter with one integer atomic per entry. The entries of a column end in an order that depends on thread timing |
| CSC: histogram + scan + scatter + segmented sort | The same, then `cub::DeviceSegmentedSort::SortPairs` by row index inside each column, which makes the result deterministic |
| CSC: compaction of the old CSC | The compaction above applied to the existing CSC. Only possible when the rebuild is deletions only |
| total: ... | Compact CSR followed by one of the CSC methods, as one timed unit |
| cpu: ... | One thread: compaction and a counting transpose |

The radix-sort and scatter methods run on the compacted CSR (its sizes are in
`verification.other`, check `rebuild against host`), but every row of this section is divided by
the nonzeros of the original matrix, so that the rows and the totals compare directly. For a cost
per nonzero transposed, divide `ms_median` by the compacted count.

Entries are assigned to rows by threads that own 32 consecutive entries and walk the offsets, so
the cost does not depend on the row length distribution. Buffers and CUB temporary storage are
allocated before the timed part.

## Checks (once per run)

- Every activity variant against the CPU reference of the same arithmetic: counts identical; sums
  within 1e-9 relative to the row's sum of term magnitudes. The difference relative to the sum
  itself is reported too; plain variants add in different orders, so on a row with cancellation
  that figure can be large without anything being wrong (510 on the hand-written model below).
  The T variants must be, and are, bitwise equal to the CPU.
- Compensated against plain (CPU): largest activity difference and its row, and the number of
  candidate bounds that differ.
- Every candidate variant, run on the reference activities: bitwise equal to the CPU by column;
  CPU by row equal to CPU by column; the `-mfma` build equal to the other.
- Frontier: listed rows bitwise equal to the full H/S/A sweep, all other rows untouched.
  `--synthetic 3 100000 8` has a row above 8192 entries in two of the four frontiers and outside
  the other two; `--synthetic 100000 20 8` has only columns above 8192 entries.
- Rebuild: compacted CSR and the three ordered CSC results equal to the host result.

## Output

stdout: one line per measurement (`section variant ms-median ms-p10 ms-p90 ns/nnz reps`) and the
check summary. JSON:

- `input`, `quick`, `repetitions`, `warmup`, `min_repetitions`, `budget_seconds`
- `gpu`: name, compute capability, memory, CUDA runtime, driver and compiler versions, CCCL version
- `matrix`: sizes, longest row and column, rows and columns with their nonzeros per length class,
  infinite column bounds, finite row sides, rows with 0 / 1 / 2 or more infinite terms per side,
  rows with a nonzero lo per side
- `results.{activity,candidate,round,frontier,cpu,rebuild}`: lists of `name`, `ms_median`,
  `ms_p10`, `ms_p90`, `ns_per_nnz_*`, `nnz`, `repetitions`
- `verification`: `ok` and the per-check details
- `values` (only when rows and columns are both at most 1024): reference activities, counts and
  candidates, for checking by hand

## Hand-written check model

80 rows, 595 columns, written by a Python script from made-up data and compared with exact
rational arithmetic: x1 = 2, x2 = 3 fixed, x3 free, x1 + x2 + x3 <= 10 (x3 <= 5); the 2^54
cancellation case (x4 <= 1); the 513-term row with 2^54, 511 times -1, -2^54 and one free variable
(compensated: 511, plain: 0); a row with two infinite terms; an empty row; a term above 1e20; an
integer column in 70 rows (block path); ranged and equality rows. The script is not in the tree;
it is at `/home/scratch.acoerduek_sw/gpu_presolve_deps/build/microbench/hand_dump.py`
(`write <file.csr>`, `check <sweeps.json>`).

The outputs of the functional checks of 2026-11-26 (`--quick`, on the shared node, so the times
are not measurements) are in `../results/stage0/sweeps_check/`.

## Limits

- Fewer than 2^31 nonzeros (32-bit entry positions in the rebuild); the largest testbed instance
  has 2.6e8.
- No frontier-restricted candidate kernel on the device.
- The error guard of DESIGN.md 2.4 (update count and largest term per row) is not modelled.
- Device memory at the peak is roughly 110 bytes per nonzero (estimated from the array sizes, not
  measured): about 28 GB for the largest testbed instance.
