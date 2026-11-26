# Stage 0 report: HiGHS baseline and GPU cost model

Date: 2026-11-26. Machine: one NVIDIA H200 NVL and 18 cores of a shared Xeon 6960P node. HiGHS
1.16.0 at commit 283bfc859, built with 64-bit integers, one thread.

Section 11 (added 2026-11-27) holds the Stage 1 reference made after the decisions on this report;
it replaces the Stage 1 reference figures of sections 1, 2, 3, 7 and 10 as gate targets.

Stage 0 measures; no presolve code exists yet. This report covers the 31 testbed instances
between 50,000 and 1e7 nonzeros. Section 9 lists what Stage 0 has not delivered, including the
nine instances above 1e7 nonzeros, whose runs are in progress.

How the numbers were checked: seven analyses were each recomputed from the raw result files by an
independent pass that also read the HiGHS source; this report was then checked against the raw
files and those results by three more passes, and corrected. The corrections of the first pass
are in `analysis/highs_findings.json` and `analysis/cost_model_findings.json`. The scripts in
`gpu_presolve/analysis/` print the tables of the seven analyses from `sweeps/` (the 32 matrices
named below) and the HiGHS runs. The Stage 1 reference figures, the five timing rechecks and the
run without Fourier-Motzkin were computed from the summary files directly, not by those scripts.

## Terms

- **Stage N**: a stage of the plan (DESIGN.md sections 5 and 13). Stage 1: row rules, bound
  propagation, doubleton equations, parallel rows, dual fixing by locks. Stage 2: probing,
  implications, cliques. Stage 3: dual reductions, column rules, aggregator, implied integers,
  simple probing on equations, gcd substitution, dual-fixing extensions.
  Stage 4: sparsify, cover strengthening.
- **HiGHS stage**: one top-level call of HiGHS's presolve loop (initial scan, fast loop, probing
  and so on). The **initial scan** is one pass of row and column presolve over the model; the
  **fast loop** is HiGHS's loop over singleton rows, changed rows, doubleton equations, singleton
  columns and changed columns. A **stage run** is all calls of one HiGHS stage on one instance.
- **Patched HiGHS**: HiGHS with `highs_diag.patch`, which adds counters, `DIAG` log lines and
  switches that skip stages. The DIAG total line gives the sizes HiGHS holds at the end of
  presolve.
- **Stage 1 rule set**: patched HiGHS with probing, enumeration, clique substitutions, clique
  merging, the implied-integer sweep, aggregator, sparsify, Fourier-Motzkin, inequality
  strengthening, pairwise dominated columns and parallel columns off. **Stage 1 reference**: the
  same with six of HiGHS's own column rules off as well (dominated column, forcing column, free
  column substitution, zero-cost singleton, column stuffing, weakly dominated column).
  **Stage 1+2 rule set**: the Stage 1 rule set with probing, clique substitutions and clique
  merging on (enumeration stays off).
- **Bands**: A1 5e4 to 1e5 nonzeros, A2 to 2e5, A3 to 5e5, B1 to 1e6, B2 to 3e6, B3 to 1e7.
  Band A is A1 to A3, band B is B1 to B3. Band C is 1e7 to 5e7 nonzeros, band D above.
- **32 matrices**: the 31 instances and square47 (2.7e7 nonzeros), the set of the kernel
  benchmark.
- **Shifted geometric mean**: geometric mean of (time + 0.1 s), minus 0.1 s.
- **Sweep**: the activities of all rows followed by the bound candidates of all columns (one
  propagation round). **Activity kernel**: per row the finite sum and the number of infinite
  terms of the minimum and maximum activity. **Candidate kernel**: per column the tightest
  implied bound over its rows.
- **Lane groups**: the rows of one length class up to 64 entries are each folded by 2, 4, ...
  adjacent threads of a warp, combined by a fixed tree ("sub-warp groups" in the plan). The
  other schemes compared: one thread per row; one thread per row up to 64 entries with blocks
  above; one 256-thread block per row, which is the benchmark's rendering of the scheme cuOpt
  uses today (cuOpt's own kernel was not run).
- **Paired or split arrays**: sum and count of a side stored as one 16-byte pair, or as a double
  array and a `uint16` count array.
- **Compensated sums**: (hi, lo) sums with error-free products and additions. **Gated**: the low
  part is loaded only when a flag says it is not zero.
- **Frontier**: the rows changed since the last sweep; in the benchmark a random 10%, 1%, 0.1%
  or 0.01% of the rows. **Cold**: the core's private caches are evicted before each repetition.
  **Warm**: the same list is repeated without eviction.
- **Work unit** (engine choice, sections 6 and 9): 10 ns, the cost of one frontier nonzero on the
  host; a frontier row counts 8 units. **Probing work** (sections 2 and 4): row entries scanned
  plus column entries updated in HiGHS's probing loop, as the patch counts them. A **HiGHS
  propagation sweep** (section 4) is one pass over HiGHS's queue of changed rows, not the sweep
  defined above.

## 1. Summary

1. HiGHS presolve is not fast on this testbed and its time does not follow size. The shifted
   geometric mean is 0.20 s at 5e4 to 1e5 nonzeros and 8 to 17 s from 5e5 nonzeros up; size
   explains half of the variation, and a quarter of the time is spent in HiGHS stage runs that
   remove nothing.
2. Rules of our Stage 1 carry most of HiGHS's reduction in total, but not on every instance.
   The Stage 1 reference removes 56% of the rows, 87% of the columns and 86% of the nonzeros
   that full HiGHS removes. It reaches at least 90% of the nonzeros full HiGHS removes on 12 of
   the 29 instances where full HiGHS removes more than one nonzero, and less than 50% on 9; per
   instance it leaves 23% more nonzeros in the geometric mean. It is still an upper bound for
   our Stage 1: it contains at least four Stage 3 rule families without a switch of their own
   (dual-fixing substitution, simple probing on equations, implied-integer conversion, implied
   equations) and 801,077 column fixings whose rule is not recorded.
3. At the small end the host engine decides. On the 11 matrices of bands A1 and A2 a full GPU
   sweep saves 0.4 to 1.1 ms against one CPU thread, about 1% of the time HiGHS needs for the
   Stage 1 reference there (46 to 144 ms). A GPU pass over changed rows (compaction and
   activities, compensated sums) costs 56 to 247 µs whatever its size, while one CPU thread
   needs 2 to 6 µs for the activities of 1% of the rows with warm caches, 9 to 44 µs cold, and
   23 to 154 µs at 10% of the rows. How many sweeps a cascade needs is not measured. What a GPU
   can add there is wide work. Probing is the candidate: it is the largest HiGHS stage on 12 of 31 instances and
   HiGHS stops it early by limits that count probes. Stage 0 has not measured a batched probing
   round or the yield of the candidates HiGHS never probes.
4. The kernel measurements decide the row scheduling in the geometric mean (rows binned by
   length with lane groups) and price compensated sums (14% of a sweep). They do not decide
   between paired and split arrays, the count method inside lane groups, or gating.
5. Clique merging changes rows on 14 of 31 instances; it is the whole reduction on sorrell3 and
   96% of the removed rows and 93% of the removed nonzeros on savsched1. 98.9% of the rows it
   deletes are also deleted with probing off. Whether that is "enough instances" for the plan's condition is your
   decision; section 7 gives the counts and a proposed rule.

## 2. HiGHS time

Time is the presolve itself, from HiGHS's "Presolving model" to its first "Presolve reductions"
log line, without the copy of the reduced model. Stock and patched HiGHS give identical rows,
columns and nonzeros on 31 of 31 instances (the written models are byte-identical on the 15
checked) and times within 0.95 to 1.05 of each other. After the container changed, five
instances were run again: 0.95 to 1.01 of their first times, identical sizes.

| Band | Instances | Full HiGHS: shifted mean | Median | Maximum | Stage 1 reference: shifted mean | Median |
|---|---|---|---|---|---|---|
| A1 | 5 | 0.20 s | 0.21 s | 0.27 s | 0.064 s | 0.054 s |
| A2 | 6 | 0.56 s | 0.30 s | 4.28 s | 0.087 s | 0.080 s |
| A3 | 5 | 2.26 s | 1.73 s | 15.3 s | 0.24 s | 0.25 s |
| B1 | 5 | 16.9 s | 18.1 s | 139.4 s | 2.5 s | 1.9 s |
| B2 | 5 | 8.2 s | 7.2 s | 22.3 s | 1.7 s | 1.9 s |
| B3 | 5 | 10.3 s | 7.5 s | 39.4 s | 3.5 s | 3.6 s |
| All | 31 | 2.72 s | 4.28 s | 139.4 s | 0.62 s | 0.29 s |

Sum over the 31 instances: 353.5 s for full HiGHS, 160.4 s for the Stage 1 reference, of which
126.0 s on rd-rplusc-21. The B1 mean of the reference is 2.5 s with that instance and about
0.9 s without it. The full-HiGHS columns and the 353.5 s are the stock run; the patched run sums
to 349.8 s, and section 10 takes the faster of the two per instance.

Share of the 349.8 s of the patched run by HiGHS stage: parallel rows and columns 32.2%, probing
22.0%, fast loop 17.7%, pairwise dominated columns 12.0%, enumeration of small binary rows 7.7%,
initial scan 3.0%, everything else 5.4%.

- Time against nonzeros, least squares in logarithms: exponent 1.01, R² 0.51, standard
  deviation of the residuals a factor 4.3 (largest: rd-rplusc-21, 33 times above that line). Against the robust (Theil-Sen) line, eleven instances are more than three times
  above the trend: rd-rplusc-21 61 times, piperout-d27 25, satellites2-60-fs 12, sorrell3 11.5,
  neos-4306827-ravan 10.5, and six more. Only the single passes over the model (setup, initial
  scan, shrink, scaling) follow size (R² 0.81).
- Ten stage runs take more than half of an instance that needs more than 2 s; together they hold
  58% of the summed time: rd-rplusc-21 parallel rows (93.9 s), neos-4647030-tutaki and
  supportcase6 pairwise dominated columns (28.5 and 9.9 s), splice1k1, var-smallemery-m6j6,
  piperout-d27, satellites2-60-fs and unitcal_7 probing (18.6, 16.9, 12.2, 2.4 and 1.6 s),
  neos-4306827-ravan the fast loop (10.3 s), sorrell3 enumeration (7.0 s, of which 5.95 s clique
  merging).
- 87.9 s (25.1%) are stage runs that removed nothing: pairwise dominated columns 41.9 s, probing
  27.3 s, inequality strengthening 6.7 s, enumeration 5.6 s, fast loop 4.2 s. About 44 s more
  without yield lie inside runs that removed something.
- The fast loop has a cost that does not depend on what it removes: bab6 0.98 s in five calls
  for nothing with the Stage 1 rule set, neos-5052403-cygnet 0.82 s in every variant.

Costs that do not look inherent:

| Observation | Evidence | Status |
|---|---|---|
| Removing many rows is slow where columns are very long | rd-rplusc-21: hashing the rows takes 0.14 s; at least 69 s lie in the call that deletes 71,300 duplicate rows of 7 entries, and at least 21.6 s follow in which no row goes. The model has a column in 119,267 of 125,899 rows. neos-5052403-cygnet removes 2.4 million nonzeros in 0.53 s in the same loop | That the time is walks over long columns is a hypothesis. With the column rules off, neos-4647030-tutaki deletes the same 1,389 rows in 0.78 s instead of 8.40 s, which column length does not explain |
| Pairwise dominated columns have no work cap | 41.9 of the stage's 42.1 s fix nothing; the five productive runs take 0.21 s together | Measured |
| Probing is limited by the number of probes | splice1k1: 1,001 probes of 0.95 passes over the matrix each, no finding, 18.6 s | Measured |
| Rows with one infinite term are rescanned | var-smallemery-m6j6: 1.8e9 of 1.9e9 entries of probing work are scans of about 157 rows of 1,400 to 2,300 entries that hold the one column with an infinite bound | Inferred from structure and matching counts; the patch has no counter per row |
| Clique merging is repeated without a change | 7.95 of 12.03 s over all instances; sorrell3 6.4 of 8.3 s | Measured |

Work that looks inherent: productive probing (about 4e7 to 2.9e8 entry updates on
var-smallemery-m6j6 and piperout-d27), the first clique merge, and the passes over the model,
for which HiGHS pays a median 113 ns per nonzero in setup and 292 ns per row-and-column scan.

## 3. What the rules are worth

Full HiGHS removes 944,820 rows, 1,110,759 columns and 12,560,451 nonzeros (34.2%) on the 31
instances. Four instances hold 73% of the removed nonzeros, so every summed share is carried by
a few.

Remaining nonzeros against full HiGHS, sizes from the DIAG total line:

| HiGHS variant | Geometric mean | Median | Instances more than 1% larger | Summed time |
|---|---|---|---|---|
| Full | 1.000 | 1.000 | | 350 s |
| No probing | 1.093 | 1.000 | 9 of 31 | 251 s |
| No clique merging | 1.046 | 1.000 | 4 of 31 | 328 s |
| Stage 1+2 rule set | 1.053 | 1.001 | 12 of 31 | 297 s |
| Stage 1 rule set | 1.197 | 1.023 | 17 of 31 | 165 s |
| Stage 1 reference | 1.227 | 1.047 | 18 of 31 | 160 s |

The runs are single runs made at different times; a difference between two rows of the time
column is not the cost of a stage. Paired on identical work, probing costs about 103 s, clique
merging about 13 s (12.0 s inside its calls) and the Stage 3 and 4 rules about 50 s.

- Share of full HiGHS's removal that the restricted runs reach (rows / columns / nonzeros):
  Stage 1 rule set 57.9% / 95.2% / 87.5%; Stage 1 reference 56.3% / 86.8% / 85.7% (band A
  37.6% / 84.2% / 58.6%, band B 64.9% / 87.0% / 87.8%); Stage 1+2 rule set 97.5% / 97.0% / 97.2%.
- Per instance, among the 29 where full HiGHS removes more than one nonzero, the Stage 1
  reference reaches at least 90% of the removed nonzeros on 12 and less than 50% on 9: sorrell3,
  supportcase6 and splice1k1 0%, trento1 4.0%, savsched1 7.4%, satellites2-60-fs 15.3%,
  piperout-d27 24.0%, triptim1 41.7%, bab6 42.5%.
- The Stage 1 reference is still not HiGHS restricted to our Stage 1. It keeps 58,923
  dual-fixing substitutions (neos-4306827-ravan 48,761), 13,874 substitutions by simple probing
  on equations (brazil3 13,782, 57.5% of its columns) and implied-integer conversions of the
  fast loop; none of them has a switch. It also keeps 801,077 column fixings whose rule the
  patch does not record; in full HiGHS at least 61,832 of such fixings come from forcing rows.
- By the rule that executed the removal in full HiGHS, 62.8% of the removed rows fall to
  Stage 1 rules (redundant rows 37.7%, parallel rows 11.2%, doubleton equations 8.7%, singleton
  rows 5.2%), 34.6% to clique merging, 2.4% to Stage 3 rules. HiGHS tests redundancy on activity
  sums over implied bounds, while the plan's Stage 1 keeps the sum over explicit bounds; whether
  that loses rows is not measured.
- Probing: for rows the loss without it is a cascade (86.9% of the rows lost are removed by the
  fast loop after a probing result); for nonzeros half of the loss is the probing stage itself.
- Each rule family sits on a few instances; its median ratio is 1.00 (probing 1.000, clique
  merging 1.000, Stage 2 rules on top of the Stage 1 rule set 0.998, Stage 3 and 4 rules 0.999).
- Stage order: the Stage 2 rules gain more than the Stage 3 and 4 rules in the geometric mean
  (0.879 against 0.950) but not by the number of instances (11 against 12 with at least 1% fewer
  nonzeros). Most of the nonzero part of Stage 3 and 4 is sparsify, a Stage 4 rule. The data do
  not contradict the stage order; they do not prove it.
- Fourier-Motzkin removes 22.4% of the removed nonzeros of satellites2-60-fs directly, but with
  only that rule off HiGHS ends at exactly the same rows, columns and nonzeros. Rules that fire
  rarely or remove almost nothing: Fourier-Motzkin (3 instances), forcing columns (1), implied
  equations (2), variable-bound aggregation (722 rows on one instance), inequality strengthening
  (144 nonzeros).

## 4. Probing and cliques

- Probing takes 76.9 s (22.0%) and is the largest stage on 12 instances. In band A it is 20.3
  of 36.6 s (55.5%), of which piperout-d27 is 12.2 s; the median share per band A instance is
  27.3%.
- Coverage of the first round: 40,577 of 2,012,390 candidates (2.0%). supportcase19 alone holds
  1.3 million candidates; without it the figure is 5.7%. Per instance the median is 14.7%; in
  band A HiGHS probes 23% of the candidates (median per instance 44%). Four instances are probed
  completely, twelve below 10%.
- HiGHS's limits: a contingent of 1,000 probes plus credits for findings (six instances stop at
  exactly 1,001 probes), and a stop after about sqrt(0.2 nnz) fruitless probes in a row, which
  counts a probe that only stores implications as fruitless. Both count probes, not work.
- One probe costs a median of 3,807 counted entries and 264 µs, from 18 entries (supportcase19)
  to 1.7 million (splice1k1), with a floor of 16 to 44 µs that the counters do not see.
- HiGHS propagation sweeps per probe: the instance means have a median of 2.15 and are at most 3.5 on
  23 of 29 instances; unitcal_7 needs 100.6. The deepest single probes need 201 sweeps
  (piperout-d27), 156 (unitcal_7) and 66 (uccase9). HiGHS's count leaves out clique and
  implication chains, so this is a lower bound for a design that propagates cliques in sweeps.
- Yield: 12,230 fixings and substitutions, 90.9% of them on five instances; 14 of the 29
  instances with candidates get none. 22 instances gain less than 1% of the presolved nonzeros
  from probing and hold 32.2 s of its time; six of them probe for more than 1 s (28.7 s).
- What the unprobed candidates would yield is not measured; the run with HiGHS's limits off is
  in progress. One indication: in the Stage 1+2 run HiGHS probes 7,867 and 11,766 times instead
  of 1,001 on neos-5116085-kenana and neos-5114902-kasavu, applies 1,131 and 1,521 substitutions
  instead of none and removes 34% and 42% more rows than full HiGHS (8,207 against 6,118 and
  10,206 against 7,182 rows; the difference is 0.3% of the rows of each model). Why probing
  behaves differently in that run is not established.
- Clique merging deletes 325,724 rows, 98.7% of them on sorrell3 (166,459 of 169,162 rows) and
  savsched1 (154,995 of 295,989); 322,095 of them are deleted with probing off. On
  var-smallemery-m6j6, piperout-d27 and supportcase7 the deletions follow probing results. Without merging
  the remaining nonzeros are 4.6% more in the geometric mean, 0.7% without sorrell3 and 0.02%
  without sorrell3 and savsched1. Merging leaves more nonzeros on 6 instances.
- HiGHS's explicit clique table reaches 0.6 to 1.2 million cliques on four instances and ends
  probing on its clique limit on two of them. splice1k1 holds 564,799 cliques from 6,444
  knapsack rows.

## 5. Structure of the testbed

- All three length classes of the binned matrix are needed. Rows above 64 entries hold at least
  half of the nonzeros on 14 of the 32 matrices; rows above 8,192 entries exist on 5 (square47:
  1,360 rows with 87.7% of the nonzeros); columns above 8,192 on 3 (rd-rplusc-21: 9 columns with
  69.8%).
- Rows with exactly one infinite activity term, in an activity that a finite side of the row
  uses, are at least 10% of the rows on 6 instances (rmatr200-p5 100%, trento1 99.8%,
  rd-rplusc-21 at least 95.4%, supportcase40 40.2%, unitcal_7 at least 35.0%, savsched1 at least
  12.3%).
- Of the 29 instances that the Stage 1 rule set reduces, 13 keep a row with an infinite term,
  because HiGHS makes implied bounds of continuous columns explicit only when it prepares
  probing.
- Clique rows are at least 10% of the rows on 15 instances. Ten instances have no packing or
  partitioning row in the original model, and HiGHS still has cliques on five of them before
  probing. Clique extension changes row lengths (sorrell3 from 2 entries to a median of 37).
- Numerics: only rd-rplusc-21 has a row dynamism above 1e6. HiGHS's presolved 99th percentile is
  1 on 11 of 31 instances, and HiGHS itself raises it 27 times on neos-4306827-ravan and 51 times
  on supportcase12.
- The testbed has no finite bound of 1e8 or more and free columns on 6 instances only.

## 6. GPU cost model

Fixed costs on the H200 (host-side figures were taken with other jobs on the node):

| Item | Median |
|---|---|
| Kernel launch with synchronisation | 9.0 µs |
| Kernel with an 8-byte read-back | 13.8 µs (90th percentile 21.5) |
| Reduce, reduce by key, compaction, scan, up to 1e5 elements, pooled memory | 23 to 46 µs |
| Sort, 1e3 to 1e5 elements, pooled memory | 94 to 501 µs |
| The same primitives with a fresh device allocation | about 245 µs more |
| Pinned copy, either direction, from 16 MB up | 52 GB/s |
| Pinned allocation | 0.38 s per GB |
| 64 MiB touched by the host after a kernel | Managed memory 14.4 ms; pinned copy plus host loop 4.0 ms (3.6 times faster) |

Fixed cost per band:

| Band | Best compensated sweep, median | Such sweeps that fit into 1% of full HiGHS's time | Fixed-cost steps that fit into 1% of full HiGHS's time | Fixed-cost steps that fit into 1% of the Stage 1 reference time |
|---|---|---|---|---|
| A1 | 95 µs | 14 / 22 / 31 | 13 / 27 / 34 | 5 / 6 / 15 |
| A2 | 98 µs | 7 / 35 / 393 | 10 / 37 / 528 | 6 / 9 / 17 |
| A3 | 140 µs | 9 / 108 / 1,097 | 26 / 212 / 1,892 | 14 / 32 / 45 |
| B1 | 282 µs | 68 / 450 / 4,568 | 236 / 2,095 / 16,829 | 34 / 230 / 15,203 |
| B2 | 390 µs | 42 / 185 / 696 | 324 / 831 / 2,781 | 92 / 235 / 287 |
| B3 | 741 µs | 45 / 109 / 1,167 | 648 / 924 / 4,887 | 297 / 392 / 522 |

The counts are for the smallest, the median and the largest instance of the band. A fixed-cost
step is 8 empty launches, one synchronisation, one read-back and one compaction (77 to 93 µs in
every band), without any kernel work. Against the Stage 1 reference the tightest instance is
trento1 (0.046 s): 590 such steps in the whole of HiGHS's time, 59 in 10%. A pinned mirror
refresh costs 31 to 184 µs in band A, 153 to 1,046 µs in B1 and B2, and 0.86 to 1.84 ms in B3.

Kernel variants on the 32 matrices (50 repetitions each, all results checked against a CPU
reference):

| Question | Result | Limits |
|---|---|---|
| Row scheduling | Lane groups take 0.69 times one thread per row up to 64, 0.19 times one block per row and 0.10 times one thread per row (geometric mean, compensated). One thread per row is 371 times slower on bab6 | Decided in the geometric mean, not at the large end: in band B3 lane groups are fastest on 1 of 5 matrices, and the fixed combination is 1.33 times the per-matrix best there (1.01 to 1.04 in A1 to B1). On the nine matrices above 1e7 nonzeros, measured afterwards, lane groups take 1.14 times one thread per row up to 64 in the geometric mean (2.5 times on usafa, 0.88 on square47) |
| Column scheduling | Lane groups below 1e6 nonzeros. From 1e6 to 1e7 nonzeros one thread per column up to 64 is faster on 7 of 10 matrices; on square47 lane groups are faster again | No rule is fixed: lane groups lose on 8 matrices, all with at least 114,240 columns, and no matrix has between 40,623 and 95,030 columns |
| Compensated sums | 39% in the activity kernel, nothing measurable in the candidate kernel, 14% on a sweep: 5 µs median in A1 and A2, 173 µs in B3 | The timings give the price, not the benefit: on one fresh sweep the bounds differ on 5 of 32 matrices by at most 2.2e-9 |
| Counts of infinite terms | In the one-block-per-row scheme with paired arrays, a scan with one integer atomic takes 0.71 (plain) and 0.84 (compensated) times a second block reduction | Inside lane groups with split arrays the two are equal (0.97 and 1.00); the atomic count is 9% to 11% slower on supportcase42, where 94% of the rows hold two or more infinite terms |
| Paired or split arrays; gating | Not decided by time | Two variants doing identical work differ by up to 12%; every layout and gating difference is inside that band |
| Rebuild | Stable radix sort for the CSC: 467 µs + 0.279 ns per nonzero, faster than scatter plus segmented sort on 24 of 32. Compacting the old CSC takes 0.70 times that, for deletions only | One CPU thread is faster on 4 of the 6 matrices up to 100,933 nonzeros |

Not measured, and able to change these results: the binned layout stored in place (the benchmark
used a second, sorted copy of the CSR with a scattered write), re-binning at a rebuild, a
frontier version of the lane-group kernel, other bin boundaries than 64 and 8,192, the error
guard, a second GPU type.

Host against GPU:

- Full sweep: the fastest GPU variant is cheaper than one CPU thread on all 32 matrices, 4.5 to
  88 times with compensated sums against a CPU built with FMA; the kernel pair of the plan is at
  least 2.6 times faster (median 10). In band A a host sweep costs 530 to 3,102 µs, the fastest
  GPU variant 61 to 235 µs and the kernel pair of the plan 86 to 268 µs.
- The crossover is below the measured range (nothing below 71,477 nonzeros was measured). The
  fits put it at 3,235 to 21,960 nonzeros, and per matrix with the planned kernel pair at 8,954
  to 55,553 (median 18,453). On a matrix like supportcase42, whose host sweep is cheap, the two
  engines tie at 50,000 to 56,000 nonzeros.
- Frontier: a GPU pass costs 56 to 141 µs on the smallest frontier. The cold host crosses it at
  1,026 to 80,812 frontier nonzeros (median 9,940), the warm host at a median of 23,866. Only
  the activity part is measured on both sides, with the plan's scheme (one thread per row up to
  64 entries, one block per row above) as the GPU frontier kernel; the whole step is modelled. On the 16 matrices below 5e5 nonzeros the benchmark's
  frontier path is slower than recomputing all rows with lane groups at every frontier size
  (one exception with compensated sums, trento1 at the two smallest frontiers).
- Hand-back to the host costs 46 µs with a pinned mirror and the cheapest list of changed
  entries, whose record order is not deterministic. With the ordered list it is at most about
  141 µs with a mirror and about 200 µs with a gather.

Provisional defaults for the engine choice (the frontier threshold is in work units):

| Threshold | Value | Status |
|---|---|---|
| Full sweep on the GPU | from 32,000 matrix nonzeros | A choice above the median crossover, 2.2 times below the smallest measured matrix |
| Frontier sweep on the GPU | frontier nonzeros + 8 × frontier rows ≥ 20,000 | Across equally reasonable readings the best threshold runs from 9,514 to 53,817; at 20,000 the wrong engine costs up to 145 µs per sweep measured and up to 629 µs modelled |
| Frontier-restricted GPU sweep | only from about 1e6 nonzeros and frontiers of 1% of the rows or less | In-sample boundaries, measured with that frontier kernel |
| Rebuild on the GPU | from 100,000 nonzeros | Marginal |
| Pinned mirror kept | below 64 MiB of mirrored bytes | Placeholder, not derived |

## 7. Gates, constants and decisions

| Item | Proposal | Basis |
|---|---|---|
| What is timed | HiGHS: the presolve itself, as in section 2. Ours: from the input CSR to the finished reduced model, warm | HiGHS's time includes building its data structures, shrink and scaling: a median 22% of the Stage 1 reference time, up to 44% |
| Shift of the shifted geometric mean, bands A and B | 0.1 s | 9 of 31 instances run below 0.3 s; any value from 0.01 to 0.1 s gives nearly the same weights |
| Per-instance time rule | t ≤ 1.2 × t(HiGHS) + 0.02 s, both sides the minimum of at least 3 alternating runs | Single runs of identical work differ by up to 15.1%; below 1 s the largest difference is 13.4 ms. 0.05 s is the looser alternative |
| Median per band | Not above HiGHS's on the same rule set, next to the shifted mean | rd-rplusc-21 is 126 of the 160 s of the Stage 1 reference: a mean below HiGHS can be reached without being faster on ordinary instances |
| Reductions gate of Stage 1 | Geometric mean over the instances of nnz(HiGHS reference) / nnz(ours) at least 0.99, every instance below 0.95 explained | One reading of the plan's wording; the other compares removed nonzeros. Over the 31 instances the 0.99 mean is used up by one instance at 73%; taken per band (5 or 6 instances) by one at 94% to 95%; over all 40 by one at 67%. Which set the plan means is your decision |
| Sizes of a HiGHS reference | From the DIAG total line | HiGHS returns the original model with "Not reduced" when only clique merging changed it (sorrell3 in the Stage 1+2 run) |
| Size comparison | Rows, columns and nonzeros together; clique-extension nonzeros (189,101) and aggregator fill-in apart; implied integers and row dynamism reported | A restricted rule set can leave fewer nonzeros than HiGHS with more rows (brazil3, unitcal_7) |
| Stage 2 gate | Compare the model after the whole presolve with and without the new probing (rows, columns, nonzeros, bounds), not the number of fixings | rd-rplusc-21 gets 17 fixings and substitutions from probing; without probing its presolved model has 351,150 nonzeros instead of 151,360 (2.3 times as many) |
| Clique gate | On the instances where HiGHS merging leaves at least 1% fewer rows (8 today): rows and nonzeros at least 0.99 of the reference in the geometric mean, extension nonzeros apart. The reference is a Stage 1+2 switch set with enumeration on, which is not run yet | See decision 2 |
| Status instances | neos-3230511-yuna (band D): unbounded or infeasible, time reported only. neos-2075418-temuka (B3): HiGHS returns the model unreduced after 15.7 s; detecting infeasibility is better, returning a model within 18.8 s equals it. bnatt500 and fhnw-binpack4-4 are below 50,000 nonzeros: no time rule | The rule by which HiGHS detects yuna is unknown; if it is a Stage 3 argument, the Stage 1 check needs an exception |

Decisions that are yours:

1. **Stage 1 reference.** Either bring the reference closer to our Stage 1: patch switches for
   simple probing on equations, the dual-fixing extensions apart from the lock rule, implied
   equations and implied-integer conversion, the counter for column fixings, and one more run
   with HiGHS's dual-fixing rule off as well (which also removes the Stage 1 lock rule), so that
   two runs bracket dual fixing. Or move those single-row and single-column rules from Stage 3
   into Stage 1. Until one of them happens the reductions gate compares against an upper bound, and
   brazil3 or neos-4306827-ravan alone can decide it.
2. **Cliques.** The plan builds the clique items "only if Stage 0 shows that HiGHS clique
   merging changes rows on enough instances". Merging changes rows on 14 of 31 instances.
   Without it HiGHS keeps at least 1% more rows on 8 and more than 10% on 4 (sorrell3,
   savsched1, var-smallemery-m6j6, brazil3), at least 1% more nonzeros on 4 and more than 10%
   on 2. Proposed rule: build the merging part if leaving it out costs more than the 1% the
   parity gates tolerate; measured 4.6% in the geometric mean of nonzeros, but 0.7% without
   sorrell3. These data cover the merging part only, not the export to cuOpt's clique table,
   clique-aware propagation or knapsack cliques.
3. **Constant of the time rule**: 0.02 s or 0.05 s. With 0.02 s the limit is 1.54 to 1.64
   times HiGHS's time on the six fastest instances of the Stage 1 reference and above 2 times
   on none; with 0.05 s it is 2.04 to 2.29 times there and above 2 times on 6 instances.
   **Reading of the reductions gate**: ratio of remaining nonzeros or of removed nonzeros. The
   second is undefined on the four instances where the Stage 1 reference removes nothing
   (rmatr200-p5, sorrell3, supportcase6, splice1k1).
4. **Scaling.** The targets are for the unscaled models. The plan evaluates gates with cuOpt's
   scaling on; then the HiGHS references have to be run again on the scaled models, or the plan
   has to say that HiGHS gets the unscaled model.
5. **Summation order.** Lane groups differ in bits from a single thread over the row on 9 of
   32 matrices (hi + lo as one double is equal on all 32); the plan's scheme already differs on
   5 of them through its block trees above 64 entries. The result stays deterministic either
   way. Two orders for one row can occur only where a frontier-restricted GPU sweep is used, by
   section 6 from about 1e6 nonzeros. Options: a frontier kernel in lane-group order (not built,
   cost to be measured in Stage 1), the plan's scheme for full and frontier sweeps (1.45 times
   the lane-group activity kernel, a median of 42 µs more per sweep), or two fixed orders.
   Recommended: one order, by the first option.

## 8. Proposed changes to the plan

None is applied to DESIGN.md yet. "Confirms" marks what the plan already says. Section numbers
in parentheses are those of DESIGN.md; "section 6" and "section 7" in the text are sections of
this report.

1. **Host engine (confirms section 1, adds the target).** At Stage 1 the cascade below 5e5
   nonzeros runs on the host; the thresholds of section 6 apply where the GPU engine is enabled.
   The time to beat is HiGHS's initial scan plus fast loop: 21 to 102 ms in bands A1 and A2 and
   37 to 163 ms in A3 in the Stage 1 reference run.
2. **Deletion cost (new, section 2.3).** Deleting rows or columns costs time proportional to
   what is deleted, also when columns are very long.
3. **Single infinite term (new, section 2.4).** Keep, per row and side, the position of the
   single infinite entry next to the exact counters.
4. **Bounded cost without yield (new).** Every phase of our presolve gets a work budget and
   stops when work passes without a finding. Pairwise dominated columns (Stage 3) get a hard
   cap; its size waits for HiGHS's count of pair tests, which the patch does not print yet.
5. **Probing (confirms the budget in entries per nonzero and re-probing only changed
   neighbourhoods; adds two rules).** Stop a pass by yield, and cap new cliques and implications
   per pass without ending probing. Evidence for re-probing: piperout-d27 round 4 makes 5,783
   probes in 3.4 s for 22 reductions.
6. **Cliques (confirms implicit knapsack cliques; adds two rules).** Merge only cliques that
   changed, and rebuild the length bins after clique extension. Gate the merging part apart from
   probing, subject to decision 2.
7. **Kernels (changes section 2.4).** In bands A and B rows up to 64 entries are folded by
   lane groups, not by one thread. Above 1e7 nonzeros the row scheduling stays open, like the
   column scheduling above 1e6. Kept from the plan without a deciding measurement: storage in
   binned order (not measured; the benchmark used a second sorted copy), split arrays (equal to
   paired arrays within the noise), compensated sums (cost 14% of a sweep; benefit not
   measured). Supported by measurement: no second block reduction for the counts in block
   kernels. New: rebuild by stable radix sort, compaction of the old CSC when only deleting.
8. **Summation order (section 2.4).** Subject to decision 5.
9. **Engine thresholds (sections 2.1 and 2.5).** Enter the provisional defaults of section 6
   and state that the defaults are fixed after Stage 1 measures them again, not in Stage 0.
10. **Gates (sections 1 and 13).** Enter the items of section 7 that you accept.
11. **Section 6 of the plan.** Correct the statement about when HiGHS makes implied bounds
    explicit: only when it prepares probing.
12. **Test oracle (adds free columns).** Huge finite bounds are already in it. The testbed has
    no finite bound of 1e8 or more, and free columns on 6 of 31 instances only.
13. **Upload after the first host cascade (new, sections 2.2 and 2.3).** The Stage 1 rules
    alone remove at least half of the rows on 9 of 31 instances and at least half of the
    nonzeros on 10, and the length classes shift (rd-rplusc-21: longest column from 119,267 to
    47,974 entries), so the binned device matrix is built on the model after the first host
    cascade.

## 9. Not delivered by Stage 0

Defaults the plan expected from Stage 0 that are not set:

- The probing work budget: it waits for the run with HiGHS's limits off, and its unit (updates
  along columns, as HiGHS counts, or recomputation of touched rows) is open.
- The byte budget and the dead-entry fraction that triggers a rebuild: not measured.
- The device fixed-cost budget per presolve call: 5 work units (50 ns) per nonzero was proposed and
  is a policy choice, not derived.
- A hand-back threshold: the number of consecutive small rounds after a hand-back is not
  measured.
- The share of the nonzeros above which all rows are recomputed: frontiers between 10% and 100%
  of the rows were not measured.
- The surplus sources the plan names: the probing contingent is open (section 4); 92 aggregator
  calls on 15 instances stop after three fill-in rejections, effect not measured; stages that
  HiGHS runs once, not measured.

Runs in progress on the node (`gpu_presolve/baseline/stage0_node_runs.sh`, restartable):

- Sweep benchmark, patched and stock HiGHS on the nine instances above 1e7 nonzeros.
- Probing with HiGHS's limits off on the 31 instances.

Needed before the Stage 1 gate is applied:

- The Stage 1 reference three times, alternating with our runs, for the time rule; and the
  Stage 1 reference on the nine instances above 1e7 nonzeros (the node runs hold only full
  patched and stock HiGHS for them).
- A counter per calling rule for column fixings: it settles whether three quarters of the column
  removal belongs to Stage 1 or Stage 3 rules.
- A switch that tests row redundancy on explicit-bound sums.
- The switches of decision 1, if that way is chosen.

Needed before Stage 2 and Stage 3:

- Three counters for the probing design (bound changes, distinct rows touched and their summed
  length per probe); HiGHS's count of pair tests in pairwise dominated columns.
- Five more switch sets as references for the Stage 2 and Stage 3 gates.

What Stage 1 must measure again with the real kernels: a frontier-restricted candidate kernel
and the whole GPU step; the host engine with incremental activities on real frontiers; frontiers
between 10% and 100% of the rows; full sweep and rebuild on matrices of 20,000 to 70,000
nonzeros (models shrink during presolve: ns1830653 goes from 100,933 to 24,263); a deterministic
list of changed entries on a model-sized state; consecutive small rounds per cascade; the binned
layout in place and re-binning; a second GPU type.

## 10. Target table

Sizes after presolve and presolve seconds from single runs. "Stage 1 reference" is defined under
Terms and is an upper bound for our Stage 1 (section 3). Limits are 1.2 × t + 0.02 s, rounded,
with t the faster of the stock and the patched run for full HiGHS; they are provisional until
the minimum of three alternating runs exists. `targets.csv` holds the stock run, full HiGHS, the
Stage 1 rule set, the Stage 1+2 rule set and the runs without probing and without clique
merging, not the Stage 1 reference (that is in `diag_stage1b/summary.json`); its limit columns
use 0.05 s and the Stage 1 rule set.

| Band | Instance | Nonzeros | Full HiGHS: nonzeros | Full HiGHS: s | Stage 1 reference: nonzeros | Stage 1 reference: s | Limit against full, s | Limit against Stage 1 reference, s |
|---|---|---|---|---|---|---|---|---|
| A1 | istanbul-no-cutoff | 71,477 | 63,332 | 0.224 | 64,803 | 0.059 | 0.289 | 0.091 |
| A1 | ns1208400 | 81,746 | 73,621 | 0.110 | 73,621 | 0.054 | 0.152 | 0.085 |
| A1 | wachplan | 89,361 | 12,024 | 0.211 | 12,698 | 0.048 | 0.273 | 0.078 |
| A1 | trento1 | 93,571 | 89,185 | 0.267 | 93,396 | 0.046 | 0.340 | 0.075 |
| A1 | radiationm18-12-05 | 96,149 | 13,631 | 0.191 | 15,760 | 0.124 | 0.249 | 0.169 |
| A2 | ns1830653 | 100,933 | 24,263 | 0.185 | 34,215 | 0.049 | 0.242 | 0.079 |
| A2 | supportcase40 | 104,420 | 24,192 | 0.083 | 24,592 | 0.059 | 0.120 | 0.091 |
| A2 | rmatr200-p5 | 113,048 | 113,048 | 0.229 | 113,048 | 0.063 | 0.295 | 0.095 |
| A2 | satellites2-60-fs | 125,048 | 112,533 | 4.28 | 123,135 | 0.098 | 5.16 | 0.138 |
| A2 | unitcal_7 | 127,595 | 97,142 | 2.26 | 108,302 | 0.130 | 2.73 | 0.176 |
| A2 | brazil3 | 133,184 | 42,182 | 0.353 | 40,518 | 0.144 | 0.444 | 0.193 |
| A3 | piperout-d27 | 224,457 | 79,040 | 14.97 | 189,572 | 0.236 | 17.98 | 0.303 |
| A3 | uccase9 | 332,316 | 256,250 | 1.73 | 258,515 | 0.288 | 2.10 | 0.366 |
| A3 | sorrell3 | 338,324 | 102,026 | 10.45 | 338,324 | 0.380 | 12.56 | 0.476 |
| A3 | sp98ar | 426,148 | 419,241 | 0.662 | 419,196 | 0.252 | 0.814 | 0.322 |
| A3 | supportcase42 | 435,653 | 435,089 | 0.215 | 435,093 | 0.114 | 0.278 | 0.157 |
| B1 | triptim1 | 515,436 | 473,555 | 1.92 | 497,959 | 0.281 | 2.32 | 0.357 |
| B1 | supportcase6 | 584,976 | 583,550 | 12.69 | 584,976 | 0.355 | 15.25 | 0.446 |
| B1 | neos-4306827-ravan | 639,474 | 218,370 | 17.80 | 268,132 | 2.93 | 21.38 | 3.54 |
| B1 | var-smallemery-m6j6 | 850,621 | 276,002 | 21.32 | 445,729 | 1.87 | 25.60 | 2.26 |
| B1 | rd-rplusc-21 | 852,384 | 151,360 | 135.7 | 352,277 | 126.0 | 162.9 | 151.2 |
| B2 | bab6 | 1,283,181 | 986,387 | 16.14 | 1,157,029 | 2.02 | 19.38 | 2.44 |
| B2 | splice1k1 | 1,761,016 | 1,758,012 | 22.00 | 1,761,016 | 0.740 | 26.42 | 0.908 |
| B2 | savsched1 | 1,770,507 | 1,425,172 | 7.16 | 1,745,028 | 1.89 | 8.62 | 2.28 |
| B2 | supportcase12 | 2,334,437 | 169,295 | 2.70 | 371,401 | 2.42 | 3.25 | 2.92 |
| B2 | supportcase7 | 2,845,545 | 231,498 | 5.08 | 484,011 | 1.88 | 6.12 | 2.28 |
| B3 | neos-5116085-kenana | 3,032,792 | 3,015,996 | 7.23 | 3,020,556 | 3.59 | 8.70 | 4.32 |
| B3 | neos-4647030-tutaki | 3,953,388 | 1,980,940 | 39.36 | 1,989,344 | 2.40 | 47.26 | 2.90 |
| B3 | neos-5114902-kasavu | 4,240,376 | 4,219,628 | 10.08 | 4,226,012 | 4.76 | 12.12 | 5.73 |
| B3 | supportcase19 | 4,287,094 | 4,287,093 | 7.48 | 4,287,091 | 2.89 | 8.99 | 3.49 |
| B3 | neos-5052403-cygnet | 4,898,304 | 2,448,853 | 5.22 | 2,449,152 | 4.25 | 6.28 | 5.12 |

## 11. Addendum: the Stage 1 reference after the decisions (2026-11-27)

Decision 1 was to bring the reference closer to our Stage 1. The diagnostic patch got four more
switches (simple probing on equations, everything dual fixing does beyond fixing by locks,
implied equations, implied-integer conversion), a counter per rule for column fixings and
counters for probing and pairwise dominated columns. With no switch set the presolved model is
byte-identical to stock HiGHS on 32 instances. The runs below are in `results/stage1/refs`; the
targets per instance are `results/stage1/targets_stage1.csv` (`analysis/stage1_targets.py`).

| HiGHS variant (31 instances) | Remaining nonzeros against full, geometric mean | Median | More than 1% larger | Removal reached (rows / columns / nonzeros) | Summed time |
|---|---|---|---|---|---|
| Stage 1 reference of the plan (section 3's reference plus the four switches) | 1.297 | 1.056 | 20 | 44.9% / 78.2% / 82.8% | 28.8 s |
| The same with HiGHS's dual fixing off (no fixing by locks) | 1.427 | 1.110 | 20 | 43.3% / 77.0% / 79.4% | 28.3 s |
| The same with row redundancy on explicit-bound sums | 1.298 | 1.056 | 20 | 44.8% / 78.2% / 82.8% | 28.2 s |
| Stage 2 reference (the first row with enumeration, probing and the clique rules on) | 1.104 | 1.009 | 15 | 85.6% / 81.4% / 93.5% | 196.3 s |

Time of the Stage 1 reference per band, shifted mean (median): A1 0.053 s (0.041), A2 0.077 s
(0.071), A3 0.21 s (0.21), B1 0.60 s (0.61), B2 1.28 s (1.30), B3 3.24 s (3.18). These are
single runs with other jobs on the node.

What the runs show:

- **The time target of Stage 1 is much tighter than section 2 said.** The summed time falls from
  160 s to 29 s, almost all of it on rd-rplusc-21: 0.61 s instead of 126 s, with the same 54,170
  rows left and the same 71,301 duplicate rows removed. The row loop of parallel-row detection
  takes 0.10 s instead of 96 s. So HiGHS's slow row removal on that instance belongs to the rule
  families the new switches turn off, not to parallel-row detection; which of the four it is has
  not been isolated. The first row of the table in section 2 ("removing many rows is slow where
  columns are very long") should be read with this.
- **The reference no longer contains what our Stage 1 does not have.** Against section 3's
  reference the largest changes are brazil3 (3,275 to 9,315 rows), neos-4306827-ravan (99,998
  to 185,534), radiationm18-12-05 (5,255 to 7,243) and var-smallemery-m6j6 (445,729 to 518,475
  nonzeros). Rules that still have no switch: the binary row test, implied bounds written to
  integer column bounds, integer row scaling, coefficient strengthening, gcd substitution.
- **Testing redundancy on explicit-bound sums costs little.** Only unitcal_7 (1.1% more
  nonzeros) and supportcase7 (0.4%) change. The plan's choice for Stage 1 stands.
- **Fixing by locks is a Stage 1 rule that matters.** Without it wachplan keeps 71,532 nonzeros
  instead of 12,698, var-smallemery-m6j6 846,407 instead of 518,475, radiationm18-12-05 30,421
  instead of 20,736, supportcase40 31,480 instead of 24,592.
- **Most column fixings are Stage 1.** Of the 869,994 column fixings of full HiGHS, 512,275
  (59%) come from forcing rows, 47,456 from singleton rows, 23,192 are empty columns. Stage 3
  rules: dual-fixing bound strengthening 191,950, weakly dominated columns 48,260, the
  single-equation extension of dual fixing 11,626, pairwise dominated columns 7,652, parallel
  columns 2,585. Probing fixes 9,260.
- **Pairwise dominated columns.** neos-4647030-tutaki makes 661,764 pair tests in 22.9 s (35 µs
  per test, long columns) and supportcase6 makes 89.6 million, of which the signature rejects
  86.7 million, in 9.6 s. Neither fixes a column.
- **Probing, per probe** (instance means on the six instances with the most probes): 34 to 1,614
  bound changes, 147 to 1,634 rows touched, 1,731 to 515,910 entries in those rows; the largest
  single probes touch 14,357 to 559,296 entries.

## 12. Addendum: HiGHS on the nine instances above 1e7 nonzeros (2026-11-27)

Patched HiGHS (first version of the patch, which gives the same models and times as stock on the
smaller instances), one thread, one instance at a time, time limit 3,600 s, other jobs on the
node. Results in `results/stage0/diag_large`; the sweep benchmark on these instances is in
`results/stage0/sweeps_large`. The stock run and the Stage 1 reference on these instances are
not made.

| Band | Instance | Nonzeros | Rows removed | Columns removed | Nonzeros removed | Presolve time | Largest HiGHS stages |
|---|---|---|---|---|---|---|---|
| C | usafa | 16,485,102 | 84.7% | 0.9% | 81.8% | 28 s | fast loop 30%, parallel rows 26% |
| C | a2864-99blp | 20,078,717 | 6.7% | 93.1% | 93.1% | 136 s | enumeration 49%, probing 22% |
| C | neos-3740487-motru | 26,433,474 | 0.0% | 0.0% | 0.0% | 177 s | probing 54%, shrink 12% |
| C | square47 | 27,329,856 | 96.3% | 62.4% | 69.0% | 281 s | probing 69%, dominated columns 16% |
| C | t11nonreg | 46,841,437 | 67.9% | 29.2% | 44.4% | 3,357 s | fast loop 62%, dominated columns 28% |
| C | neos-4332810-sesia | 47,513,027 | 99.6% | 99.8% | 99.6% | 50 s | initial scan 50%, setup 20% |
| D | dlr2 | 78,091,589 | 71.7% | 94.7% | 88.3% | 175 s | Fourier-Motzkin 41%, fast loop 24% |
| D | hawaiiv10-130 | 183,263,061 | 71.3% | 22.0% | 32.8% | 3,603 s (time limit) | inequality strengthening 62%, enumeration 22% |
| D | neos-4535459-waipa | 256,963,402 | 0.0% | 0.1% | 2.0% | 544 s | sparsify 33%, dominated columns 29% |

- HiGHS needs 28 s to an hour here, and the time does not follow the reduction: neos-3740487-motru
  takes 177 s and removes nothing, neos-4535459-waipa takes 544 s for 2% of the nonzeros, while
  neos-4332810-sesia loses 99.6% in 50 s.
- hawaiiv10-130 stops on the time limit, so its sizes are those reached after an hour.
- Reading these instances takes 10 to 138 s and at most 35 GB of memory in the tool.

The Stage 1 reference of section 11 on the same instances (`results/stage1/refs/stage1c_large`,
no time limit reached):

| Instance | Stage 1 reference: nonzeros removed | Time | Full HiGHS: nonzeros removed | Time |
|---|---|---|---|---|
| usafa | 81.6% | 13 s | 81.8% | 28 s |
| a2864-99blp | 93.1% | 34 s | 93.1% | 136 s |
| neos-3740487-motru | 0.0% | 38 s | 0.0% | 177 s |
| square47 | 69.0% | 32 s | 69.0% | 281 s |
| t11nonreg | 30.7% | 50 s | 44.4% | 3,357 s |
| neos-4332810-sesia | 27.8% | 119 s | 99.6% | 50 s |
| dlr2 | 7.0% | 67 s | 88.3% | 175 s |
| hawaiiv10-130 | 1.2% | 139 s | 32.8% | 3,603 s |
| neos-4535459-waipa | 0.0% | 93 s | 2.0% | 544 s |

On four of the nine the Stage 1 rules reach what full HiGHS reaches. On neos-4332810-sesia and
dlr2 nearly all of the reduction needs the later stages, and on sesia the restricted run is
slower than full HiGHS because the model stays large.

## 13. Addendum: probing with HiGHS's limits off (2026-11-27)

Patched HiGHS on the 31 instances with `HIGHS_DIAG_PROBING_NOSTOP=1`: a probing round stops only
when every candidate was visited, on infeasibility or on the time limit (600 s here). Results in
`results/stage0/diag_nostop`. This answers what section 4 left open: what the candidates HiGHS
never probes would yield.

| | Full HiGHS | Limits off |
|---|---|---|
| Probes | 66,435 | 2,078,003 |
| Fixings and substitutions | 12,230 | 28,071 |
| Probing time, summed | 77 s | 1,160 s |
| Remaining nonzeros, geometric mean | 1.000 | 0.982 |

- Probing every candidate leaves 1.8% fewer nonzeros in the geometric mean, for 15 times the
  probing time and 31 times the probes. supportcase6 stops on the time limit after 595 s with
  nothing found.
- The gain sits on a few instances: supportcase7 (179,880 nonzeros instead of 231,498), bab6
  (828,782 instead of 986,387, and 25,486 rows instead of 29,306), brazil3 (37,785 instead of
  42,182), and 1% to 2% on wachplan, triptim1 and neos-4306827-ravan. piperout-d27 ends 1.1%
  larger.
- neos-5116085-kenana and neos-5114902-kasavu get 1,440 and 2,016 substitutions where the
  limited run gets none; their models shrink by 0.2% to 0.3%.
- On the other instances full probing finds nothing more: supportcase19 makes 1.3 million probes
  in 43 s, savsched1 247,949 probes for 3 substitutions, splice1k1 3,252 probes in 59 s for
  nothing.

For the Stage 2 budget: the surplus that complete probing can add over HiGHS on this testbed is
about 2% of the nonzeros in the geometric mean and up to 22% on single instances. A probing
phase that covers every candidate has to be at least 15 times cheaper per probe than HiGHS's to
stay inside HiGHS's probing time, or it has to choose its candidates.
