# GPU MIP presolve: development history

A running log of how this feature was developed, kept so the process can be documented later. Each
entry records what the user asked, what was done, what was found, and what changed. The user's
messages are quoted as written. Newest entry last.

Related files: [DESIGN.md](DESIGN.md) is the current plan.

---

## Entry 1 — 2026-11-26 — Request and first design plan

### User request

> I want you to design and implement a MIP GPU presolve by researching papers, looking at highs, scip
> and papilo code. It should be performance focused, especially performing well on large problems. You
> can do it separate from cuopt to iterate faster. Use the largest 10 instances in MIPLIB as testbed
> before launching benchmarks. The goal is to be better than highs in terms of problem reductions and
> quality of reductions. Find the balance yourself. Quality might mean: numerical stability, better lp
> relaxation, better mip solutions, better integer integer reductions etc. First give me the design
> plan and then i will give it a go for the implementation.

### Step 1: research (five parallel agents)

| Thread | What it produced |
|---|---|
| cuOpt code survey | MIP presolve today is PaPILO on the CPU (4 threads), followed by GPU bound propagation, a probing cache, trivial presolve and block BVE. No GPU hashing, clique table or implied-integer detection on the device. A load-balanced bound propagation exists in the tree but is not compiled. |
| HiGHS code study (commit 283bfc859) | Full rule list with conditions and tolerances; control flow; effort limits (probing contingent starts at 1,000; aggregator pass ends after 3 consecutive fill-in failures; parallel rows/columns at most 5 calls); no parallelism in presolve. |
| PaPILO and SCIP code study | PaPILO's presolvers, its transaction model (parallel detection, sequential apply with conflict rejection), round control, numerical safeguards; SCIP's plugins and which are off by default. |
| Literature | Gurobi's ablation table (aggregator 1.42, probing 1.17, clique merging 1.14, implied integers 1.13); GPU domain propagation (Sofranac et al.); two GPU LP presolvers from 2026 (cuPSLP, HPR-LP preprocessor); no GPU MIP presolve and no GPU probing paper found. |
| MIPLIB sizes | Ranking of the collection by nonzeros. |

Lesson: the agent that ranked MIPLIB read a truncated page and missed two of the top 20 instances
(t11nonreg, a2864-99blp). The ranking was redone by parsing the full 1,065-row table directly.

### Step 2: design panel (workflow, 9 agents, about 45 minutes)

Three independent designs from different angles (bulk linear algebra; frontier-driven; strength
first), each attacked by a soundness reviewer and a feasibility reviewer. All six reviews scored 4
out of 10.

What converged across the three designs:
- classes of reductions that may be applied together (bound facts; lock arguments; dual reductions
  under a read/write reservation);
- doubleton chains collapsed in one batch by pointer jumping over an affine forest;
- a batch of substitutions with a diagonal pivot block applied as one Schur-complement update;
- probing as a sparse multi-source frontier, not full sweeps;
- activities in double-double with infinity counters; no floating-point atomics.

What the reviewers broke, each with a concrete counterexample:
- an integer column substituted through a row with continuous columns loses integrality;
- row-dual arguments taken from integer columns declare a feasible MIP infeasible;
- parallel-column merges of integer columns create values with no preimage;
- coefficient strengthening and parallel-row merging applied to the same row in one batch;
- implied bounds trusted after their source row was rewritten;
- cancellation residue (1e-17) surviving as a coefficient and fixing a free column.

What the feasibility reviewers established:
- pure batch application ends below a sequential presolve on cascades and ordered aggregation;
- no HiGHS measurement existed on the testbed, so no win could be claimed;
- GPU probing has no published precedent and its coverage advantage was unproven.

### Step 3: merged plan and second review (workflow, 4 agents, about 38 minutes)

The merged plan was a hybrid: GPU bulk engine, cascade loop with a host worklist, sequential CPU
tail. Four reviewers (soundness, fact check, feasibility and staging, completeness) all judged the
first merged draft not fit for approval. Main corrections:
- the substitution batch needs a premise set per pivot, not only the diagonal rule;
- one application order per round, one reservation table, corrected undo order;
- yuna and yass, two of the ten largest instances, are unbounded;
- HiGHS must be built with 64-bit integers (its clique limit overflows 32 bits on waipa) and cannot
  be restricted to a rule subset or report per-rule counts for MIP without a patch;
- HiGHS's written presolved model is power-of-two scaled;
- PaPILO's substitution is batch-and-reject, not a sequential aggregator with live counts;
- only 30 of the 240 MIPLIB benchmark instances have at least 1e6 nonzeros.

### Outcome

Plan version 1 written to DESIGN.md with six decisions for the user and a proposal to approve
Stages 0 (baselines) and 1 (core) only. Nothing built, downloaded or run.

---

## Entry 2 — 2026-11-26 — User feedback on plan version 1

### User messages

> Keep the history of our interactions in another file. I would like to document how we have
> developed such a complex feature from scratch and productized it.

Feedback on the plan (pasted by the user):

> I wouldn't just do the largest 10 in MIPLIB, but also consider the differences in the problem to
> keep the diversity for pre-solve the test, right? I wouldn't put the goal on 10 to 100 million
> non-zeros. These are too large. It should already perform better with more than 50,000 non-zeros.
> There is probing already implemented in cuOpt, and there is also load-balanced bound propagation.
> You may use that to improve the performance or improve that as well in the presolve. When doing host
> and device hybrid work, consider Grace Hopper low-latency unified memory workloads. Do not use
> PaPILO in the CPU part. You can use it for comparing against, but do not use it within the new
> presolve. I would consider using compensated or Kahan summation for the activities instead of
> recomputing them from scratch. Make sure the new presolve is deterministic. Use reductions instead
> of atomics if needed. The clique implementation in cuOpt is CPU-only, but there is a GPU
> implementation paper. If you think it's valuable, read that paper and implement that clique
> extraction from that paper. I would iterate fast and, during implementation, create a separate
> project not including cuOpt to speed up compilation and comparison against other presolves. When we
> want to compare the quality, then we can integrate it into cuOpt to check LP relaxation and overall
> correctness. Do not do solver quality tests locally. We have benchmark clusters for large-scale
> tests. We can iterate on that later. You can launch benchmarks on EOS after each stage or when you
> want to do large-scale testing. If testing both PaPILO and HiGHS is too cumbersome, we can only test
> on HiGHS to reduce the total effort.

> Let me know the new design changes and your thoughts on my comments.

### What was done

- This history file was started.
- Three research threads were launched: the GPU clique paper, the Grace Hopper memory model, and what
  can be reused from cuOpt's probing and load-balanced bound propagation.
- A structure-diverse testbed was drawn from the MIPLIB collection table (results below).

### Diverse testbed (drawn from the MIPLIB 2017 collection table)

Counts by size: 633 collection and 147 benchmark instances have at least 50,000 nonzeros (159 and 30
at 1e6; 27 and 2 at 1e7). With the 50,000 target the regular cluster benchmark becomes meaningful.

Selection: greedy cover of 102 structure features (size band x variable mix, aspect, mean row and
column length, the 14 MIPLIB constraint-class tags, plus numerics, indicator, decomposition), one
instance per model family, benchmark instances with known optimum preferred, unbounded and
infeasible instances excluded. Result: 39 instances, all 102 features covered.

- Band A, 5e4 to 5e5 nonzeros (14): air05, piperout-27, dws008-01, comp21-2idx, cod105,
  neos-1171737, neos-3656078-kumeu, glass-sc, decomp2, cbs-cta, istanbul-no-cutoff,
  elitserienhandball14i, leo1, neos-662469.
- Band B, 5e5 to 5e6 (14): proteindesign122trx11p8, triptim1, ex9, irish-electricity,
  neos-4532248-waihi, map16715-04, supportcase10, neos-2987310-joes, supportcase6, netdiversion,
  neos-4306827-ravan, rd-rplusc-21, neos-848589, bab3.
- Band C, 5e6 to 5e7 (8): s82, gfd-schedulen180f7d50m30k18-16i, supportcase2, usafa, a2864-99blp,
  neos-3740487-motru, square47, neos-4332810-sesia.
- Band D, above 5e7 (3, stress tests): dlr2, hawaiiv10-130, neos-4535459-waipa.

### Interim reply to the user (before the research returned)

Thoughts given on each comment: the 50,000 target moves the design from "GPU sweeps with a CPU
tail" to "engine chosen per phase by work volume", with the GPU winning on wide work (probing,
pairwise tests, cliques) and our own CPU engine running cascades; unified memory removes the host
mirror; no PaPILO means we own a full sequential engine; compensated sums with integer infinity
counters and deterministic delta application; HiGHS as the only local baseline.

---

## Entry 3 — 2026-11-26 — The clique paper

### User message

> @"/Users/acoerduek/Documents/papers/parallel_conflict_graph.pdf"
> here is the GPU clique paper.

### What was found

The paper is Dai and Chen, "Parallelized Conflict Graph Cut Generation" (arXiv 2311.03706v4, June
2025). It is a CPU shared-memory method (Julia, up to 64 threads on a 32-core Threadripper), not a
GPU implementation. The earlier literature pass had listed it as the CPU-parallel reference.

Method: detect set-packing and conflicting knapsack rows; clique detection per knapsack row by
coefficient sort (Brito and Santos); conflict graph as a sparse adjacency matrix built per thread
and OR-merged; greedy clique extension producing one longest and several other extended cliques;
removal of dominated cliques; cut triage (longest cliques replace the original set-packing rows,
the rest become user cuts).

Reported on 176 MIPLIB benchmark instances with Gurobi 11: total time 92.3% of default with 64
threads (98.5% with one thread; 89.9% not counting the clique work), nodes 66.8%, and 82.0% of
default time on instances taking at least 1,000 s. Limits used: knapsacks above 5,000 variables
skipped; cliques above 1,000 members randomly sampled; 2.5e7 nonzeros in the graph; merging skipped
above 1e5 cliques.

Assessment given to the user: worth implementing on the GPU (tightens the LP relaxation, removes
dominated clique rows, and the conflict graph feeds probing, dominated columns and cuts), with four
changes: fixed rules instead of random shuffles and sampling; implicit cliques instead of explicit
quadratic adjacency; candidate generation by shared member and signature instead of the quadratic
subset check; and the reservation rule for strengthened rows.

Lesson: the user's description ("GPU implementation paper") and the paper's content (CPU threads)
differed. Reading the source before planning around it avoided designing against a GPU reference
that does not exist.

---

## Entry 4 — 2026-11-26 — Research for the revision and plan version 2

### Research results (workflow, 3 agents, about 9 minutes)

Clique literature: no paper was found that builds a MIP clique table or conflict graph on a GPU.
The search covered about 25 web queries, the arXiv API and the citing lists of Brito and Santos
and of Dai and Chen. cuOpt's CPU table already implements the paper's extraction step.

Grace Hopper memory model (from NVIDIA documentation and four papers):
- NVLink-C2C carries 450 GB/s per direction; CPU and GPU share one page table, so access needs no
  page fault.
- Memory from `cudaMalloc` is not host-accessible even there. Managed memory needs the accessed-by
  hint, otherwise a CPU touch migrates 2 MB blocks.
- A kernel working on CPU-resident data runs at link speed (dgemm measured 23 times slower), so the
  matrix must be GPU-resident and the host reads it in place.
- On x86 managed memory is the wrong choice: every hand-over faults at 30 to 45 microseconds per
  fault. Explicit dirty lists through pinned buffers are right there.
- cuOpt has no Grace Hopper path today.

cuOpt code that can be reused:
- The load-balanced propagation was removed from the build on 2025-09-10 (PR 372, "Combined
  variable bounds") and no longer compiles against the problem class. Its kernels can be lifted.
  It recomputes every row each iteration with plain sums and has no infinity counters.
- The built propagation uses one 256-thread block per row or column whatever the length, and cannot
  tighten a row with a single infinite term.
- Probing costs work proportional to rows plus columns per probed variable, with dense state per
  OpenMP task; its budget allows about 375 variables at 1e7 nonzeros. It does not use cliques.
- The clique table is host-only, built inside branch and bound, and not built in deterministic mode.
- Possible defect found while reading: the probing cache sets an iteration limit of 50 on an object
  the probes do not use (probing_cache.cu:919 against :934). Not confirmed by a run.

### Plan version 2: what changed

| Topic | Version 1 | Version 2 |
|---|---|---|
| Size target | 1e7 to 1e8 nonzeros; CPU presolve below 1e6 | From 50,000 nonzeros; the largest instances are stress tests |
| Engine choice | GPU sweeps, host worklist for small frontiers, size gate | Host engine and GPU engine on one state; chosen per phase by a deterministic work estimate |
| Hybrid memory | Host mirror with dirty lists | One memory interface: in-place host access on Grace Hopper, pinned mirror on x86 |
| CPU part | Own sequential aggregator plus PaPILO as tail | Own complete sequential engine; no PaPILO inside |
| Activities | Recomputed from scratch for every committed reduction | Compensated accumulators with infinity counters, updated incrementally in a fixed order |
| Propagation and probing | New kernels | Built on cuOpt's load-balanced kernels; many-source probing replaces the per-variable scheme; improvements go back into cuOpt |
| Cliques | Research item, gated | The Dai and Chen pipeline adapted to GPU kernels, in Stage 3 |
| Baselines | HiGHS and PaPILO locally | HiGHS only locally; PaPILO compared on the cluster as cuOpt's current presolve |
| Quality measures | LP bound and solver outcome in the local harness | Only through cuOpt on the cluster, after each stage |
| Testbed | Ten largest plus four | 39 structure-diverse instances in four size bands |
| Stage order | Probing and cliques last, gated | Core, then parity rules, then GPU-wide phases, with a cluster run after each |

### Check of version 2 (workflow, 2 agents, about 20 minutes)

One reviewer checked compliance with each user comment, one the soundness of the new parts. Both
judged version 2 not fit for approval as written; all findings were changes to the text.

Soundness findings:
- The activity update a * (new - old) is not what HiGHS does and loses digits (bound 1e17 tightened
  to 3 leaves the activity off by 3). HiGHS removes the old term and adds the new one, with
  error-free products.
- Host and GPU cascades can reach different, both valid, states (a two-row example), so "the two
  engines agree" cannot be a test, and results are identical across machines only with identical
  engine thresholds.
- On Grace Hopper a host cascade must read its own writes at once; "host writes go to a log" does
  not work. Mutable state has to be host-resident during a host phase.
- The rule for removing dominated clique rows was circular: a row could be removed on the strength
  of a clique that never became a model row.
- The time target with a 0.1 s shift and allowance was decided by the largest instances and said
  nothing about 50,000 nonzeros.
- Many-source probing with dense slots does not scale past about 1e6 nonzeros, and one re-probe
  pass does not recover what sequential probing finds in a chain.

Compliance findings:
- The stage order put probing and cliques, the only GPU work that pays at 50,000 nonzeros, last.
- "Better quality than HiGHS" had no test left: local runs compared size and time only, cluster
  runs compared against PaPILO.
- Local runs on the instances above 1e7 nonzeros conflict with keeping the dev node to small tests.
- PaPILO remained on three unstated paths: sub-MIPs, the failure fallback, and models below 50,000
  nonzeros.
- The testbed's smallest-first tie-break put 10 of 14 small instances between 52K and 72K
  nonzeros.
- Version 1 text had been overwritten, and six items were dropped from it without a user request:
  the size gate and sub-MIP rule, the gate on Stage 0 ceilings, the CPU-parallel probing fallback,
  LP bound and solver outcome against HiGHS, the standalone-project section, and the per-instance
  expectations.

### Plan version 3

Versions 1 and 2 were recovered from the session transcript and saved as DESIGN_v1.md and
DESIGN_v2.md. The directory is not under version control, so snapshots are the record.

Changes from version 2:
- Stage order: GPU-wide phases (probing, cliques) are Stage 2 and enter cuOpt as replacements for
  the probing cache and clique table; parity rules are Stage 3.
- Time target per size band, warm timing, constants set in Stage 0 from measured HiGHS times.
- Activities: two terms with error-free products, exact counters, row recompute on any entry
  change, an explicit error bound, a per-row fold without a sort.
- Shared memory: mutable state host-resident during host phases on Grace Hopper; on x86 no matrix
  mirror above a size threshold.
- Determinism: thresholds and byte budgets are fixed defaults independent of the machine; the
  wall-clock limit only aborts; a hash after every phase over all outputs.
- Probing: shared base plus sparse overlay as the general design; re-probing repeats until nothing
  changes or the budget ends; restore by copy.
- Cliques: the removal rule is fixed; equality clique rows fix their extension literals; what is
  not taken from the paper is stated.
- Measurement: dev node for the 30 instances below 1e7 nonzeros, a cluster job for the 9 above,
  and a decision on an extra cluster arm with HiGHS-presolved models.
- cuOpt scope stated: main problem at all sizes, sub-MIPs keep PaPILO until Stage 5, FAILED is an
  assertion outside the product default.
- Standalone-project section restored.
- Testbed redrawn with eight size sub-bands (39 instances, 26 from the benchmark set), four status
  instances, and a held-out set of 36.

Correction to Entry 3: the paper compares Gurobi with and without the method on 99 of its 176
instances (those with inferred set-packing or conflicting knapsack rows), not on all 176.

Version 3 has not been reviewed again. Three review rounds each found a dozen problems; the user
decides whether another round is worth its time before Stage 0, which involves no design risk.

---

## Entry 5 — 2026-11-26 — The single-infinity pull request

### User message

> There is a PR for single infinity tightening in bound propagation in cuOpt. Consider the review
> comments there too.

### What was found

The pull request is NVIDIA/cuopt 2372, "Fix GPU bound propagation for rows with a single infinite
activity term" (open, created 2026-11-23). It stores each activity as a pair (finite sum, number of
infinite terms) and derives a candidate for the one column that holds the infinite term.

Review comments by `akifcorduk` (2026-11-26):
- Store the infinite-term counts in separate `uint16_t` arrays instead of a paired double: the pair
  loads 128 bits; rows longer than `uint16_t` can be skipped.
- Do not double the block reductions for the counts: zero shared counters, let each thread count
  infinite contributions during its row scan and add them with one atomic only if nonzero, cap at 2.
- The new regression tests pass an LP to a kernel that needs MIP variable types; make some
  variables integer.
- Benchmark the change for regressions.

Review comments by the CodeRabbit bot:
- (side - activity) / a + own bound cancels: x in [-2^54, 10], y in [0, 1], x + y <= 1 gives
  x <= 0 and cuts off x = 1, y = 0. Subtract the column's own term first. Addressed in the PR.
- The single-infinity candidate can tighten past feasible points when the finite sum has rounding
  error (513-term row with 2^54, -1, -2^54): no candidate may be emitted unless the error is
  bounded. Open at the time of reading.
- A test leaves a destroyed device memory resource installed.

Related open pull requests seen while searching: 2127 (CPU bound propagation refactor with a
bound-change stack, incremental activities and infinite-contribution counters) and 2129 (objective
propagation with a compensated dot product).

### What changed in the plan

- Section 2.4 gained the rules from this review: split narrow arrays for the counts and a cold
  array for the compensation term; counts accumulated in the row scan with integer atomics instead
  of an extra reduction; residual-first bound formula; an error guard before any candidate is
  emitted; a benchmark before a kernel variant replaces the current one.
- One difference from the review, forced by incremental maintenance: the counts are exact, not
  capped at 2, because a cap could not be decremented when a bound becomes finite.
- Section 2.5: integer atomics are allowed and preferred where they save a reduction pass; only
  floating-point atomics are excluded. Version 2 had said "counts are reductions".
- Section 7 lists the three open pull requests; the standalone kernels and the host engine follow
  their semantics so the back-port matches.
- The test oracle gained the three numeric cases from the pull request.

Lesson: the bot's second comment is the same numerical problem the user's request for compensated
summation addresses; the plan now ties the two together (compensated sums plus an explicit error
guard) instead of treating them as separate topics.

---

## Entry 6 — 2026-11-26 — Go for Stage 0

### User message

> Yes, start Stage 0

(Answer to: "Start Stage 0 with the recommended choices (GPU-wide phases before parity)?")

### What was done

- Dev node checked: NVIDIA H200 NVL (143 GB), 36 cores (Xeon 6960P), CUDA 12.9, gcc 14.4,
  cmake 4.4.4, librmm 26.12.
- New worktree `gpu-presolve` (branch `feat/gpu-presolve` from upstream main `0d0915a8c`) on slot b.
  The standalone project lives in its `gpu_presolve/` directory. The plan and this history moved
  to `design_summaries/gpu_mip_presolve/` of the same worktree.
- Instance inventory: 26 of the 39 testbed instances are on the node (the benchmark-set ones).
  Missing: 5 below 1e7 nonzeros (piperout-d27, gfd-schedulen55f2d50m30k3i, supportcase21i,
  neos-4306827-ravan, neos-5116085-kenana) and 8 above (usafa, a2864-99blp, motru, t11nonreg,
  sesia, dlr2, hawaiiv10-130, waipa), plus the unbounded status instance yuna.
- Stock HiGHS 1.16.0 at commit 283bfc859 built with 64-bit integers under
  `/home/scratch.acoerduek_sw/gpu_presolve_deps`.
- Tooling workflow started (4 writers, each followed by a reviewer): HiGHS diagnostic patch;
  baseline tool and runner; micro-benchmark of fixed costs; micro-benchmark of sweep kernels.
- A rough first look at stock HiGHS presolve on the 25 available dev-node instances was started
  while the node was busy with builds; its timings are not measurements.

### Problems met

- The large-data scratch `/home/scratch.acoerduek_sw_1` is not mounted in the container on this
  node. By the user's rule nothing is downloaded elsewhere without asking, so the 13 missing
  instances wait for the user's decision.
- The first testbed-list script failed on the status instances below 50,000 nonzeros (no size band
  for them); a "below" band was added.

### First look at stock HiGHS (rough, not a measurement)

25 dev-node instances that were already on the node, stock HiGHS 1.16.0 (64-bit), one thread,
4 instances at a time while the node was busy with builds. Wall time includes reading the MPS
file and writing the presolved model. Table: `gpu_presolve/results/stage0/first_look_stock_highs.csv`.

- Band A (5e4 to 5e5 nonzeros): median 0.38 s, maximum 3.95 s (satellites2-60-fs, 125,048
  nonzeros).
- Band B (5e5 to 1e7): median 13.5 s, maximum 132.3 s (rd-rplusc-21, 852,384 nonzeros).
- splice1k1, supportcase19 and supportcase6 take 10 to 23 s and remove almost nothing.
- Six instances lose more than three quarters of their nonzeros (wachplan, radiationm18-12-05,
  ns1830653, supportcase40, rd-rplusc-21, supportcase12, supportcase7 are at 76% to 93%).

Script problems on the way: the container has neither `/usr/bin/time` nor `bc`; the script now
times with `date` and `awk`.

---

## Entry 7 — 2026-11-26 — The 9 large instances move from the cluster to the dev node

### User message

> "Run the 9 large instances on the cluster". If you're going to use the cluster, I would run a bit
> more because presolve doesn't take too much time, and the setup of the benchmark on the cluster
> will have overheads and take time. Either run those 9 instances locally or increase the number
> of instances on the cluster.

### Decision

The 9 testbed instances above 1e7 nonzeros run on the dev node, and the separate cluster job for
standalone runs is dropped. Reasons:

- The node has 2.2 TB of memory and 36 cores; HiGHS presolve is one thread on the host, and our
  own presolve needs the node's GPU anyway (H200, 143 GB).
- A standalone cluster job would have been a new script, a new build of HiGHS and the project on
  the cluster and a new result format, all for 9 runs per stage.
- The cluster keeps the job it is already set up for: the regular benchmark inside cuOpt after
  each stage (240 instances, existing scripts).
- Rule for later: a new cluster job is only set up for a wide set (the 147 benchmark instances at
  or above 50,000 nonzeros plus the testbed), never for a handful of instances.

### What was done

- DESIGN.md amended: section 0 item 2 and the measurement table of section 12. The plan version
  stays 3; the header carries an amendment line.
- Checked on the node: 2,266 GB memory, 52 GB free on the normal scratch, the large scratch is
  still not mounted, the stock HiGHS binary links zlib (it can read `.mps.gz`).
- Download sizes read from the MIPLIB site (compressed): the 5 missing dev-size instances 22 MB,
  the 8 missing large instances 2.87 GB (usafa 62 MB, a2864-99blp 71 MB, motru 135 MB, t11nonreg
  259 MB, sesia 331 MB, dlr2 360 MB, hawaiiv10-130 571 MB, waipa 1,081 MB), yuna 267 MB;
  3.16 GB in total.
- square47 (27.3 million nonzeros) is the one large instance already on the node; a first
  stock-HiGHS run on it was started to see how long the large instances take there.
- `first_look.sh` takes its time limit from `TL` (default 900 s).

### Open

- The download location is still the user's call: the user's rule sends datasets to the large
  scratch, which this node does not mount. Asked again with the sizes.
- The `where` field in `testbed.json` still says `cluster` for the 9 large instances and the
  runner has a `--set cluster`; both are renamed once the tooling workflow, which is editing the
  runner, has finished.

### Lesson

A cluster job has a fixed setup cost. Before proposing one, compare that cost with the run it
carries: 9 presolve-only runs do not pay for a new job script, and the node at hand could run
them all along.

---

## Entry 8 — 2026-11-26 — Git checkpoints; Stage 0 tools reviewed, first measurements

### User message (sent while Entry 7's work was running)

> use git and push milestones there to have intrinsic checkpoints and also not to lose the progress somehow.

### Git

- Branch `feat/gpu-presolve` (worktree `gpu-presolve`) is pushed to the fork (`origin`) and tracks
  it. Rule from now on: one commit per milestone, pushed at once.
- Commits so far: `1d50b9c13` (project skeleton and Stage 0 tools), `5a56c33b5` (HiGHS diagnostic
  patch), `3a9f1839b` (structure statistics).
- Three things had to be handled:
  - The local exclude file of the repository hides `*txt`, `*.sh`, `*log` and
    `design_summaries/*`. The project's `CMakeLists.txt` files and build scripts were added with
    `git add -f`. Run logs stay out.
  - The whitespace hooks would strip the trailing blanks of the patch's context lines, after which
    the patch no longer matches the HiGHS sources. The patch is committed alone with those two
    hooks skipped; its checksum is unchanged (`47153b04...`).
  - The hard-coded-version hook flags the rmm version recorded in result files; it is skipped for
    result commits.
- Not pushed: this file and DESIGN.md. They are excluded by the user's own rule, the fork is
  public, and this file quotes the user and names internal machines. Asked.

### Stage 0 tools (workflow of 4 writers, each followed by a reviewer who fixed what it found)

All four came back ready and were checked again on the node.

- HiGHS diagnostic patch (18 files of HiGHS 283bfc859): DIAG lines per stage (seconds, rows,
  columns and nonzeros removed), per probing round, per clique-merging call, substitutions by
  kind, parallel rows and columns, and switches that turn stages off. With no switch set the
  presolved model is byte-identical to stock on 15 instances. The reviewer found two gaps in the
  accounting and closed them: rows removed by variable-bound aggregation appeared in no line
  (rd-rplusc-21: all 722 rows the enumeration stage removes), and two substitution rules had no
  counter (dual-fixing substitution: 1,985 on radiationm18-12-05).
- Baseline tool and runner: presolve through the HiGHS API with the time of the presolve call
  alone, structure statistics, and a binary CSR dump. The reviewer re-implemented every statistic
  in Python and got the same numbers on 12 instances; no defect found. It found the tool binary
  older than the patched library it links to and added the rebuild rule.
- Fixed-cost micro-benchmark: the reviewer found the row-gather kernel slowed by missing
  `__restrict__` (2,480 to 1,530 microseconds at 1e7 entries) and a wrong explanation of its
  floor (it follows the block size, not load latency).
- Sweep micro-benchmark: the reviewer fixed unparseable JSON (`inf`) for an all-empty frontier
  list.

### Measured: fixed costs on the H200 (no other GPU process; one HiGHS run on the host)

| Item | Median |
|---|---|
| Empty kernel, launch and sync | 9.0 microseconds |
| Empty kernel, launch without sync | 2.0 microseconds |
| Read back 8 bytes after a kernel | 14 to 15 microseconds (p90 22) |
| thrust reduce / scan / copy_if up to 1e5 elements, pool allocator | 23 to 46 microseconds |
| Same with `cudaMalloc` for the temporary | 270 to 290 microseconds |
| Radix sort of 1e7 64-bit keys | 3.9 ms; with values 5.3 ms |
| Pinned copy, either direction | 52 GB/s from 16 MB up; 8 to 10 microseconds for 4 to 64 KB |
| Pageable copy | 10 GB/s |
| `cudaHostAlloc` | 0.38 s per GB (1.5 s for 4 GB) |
| Dirty list, host to device | 13 microseconds at 100 records, 0.58 ms at 1e6 (device part) |
| Changed-entry list, device to host | 33 microseconds up to 1e4 records when the kernel emits the list; 123 when flags are scanned |
| Rows gathered to the host | 53 to 70 microseconds up to 1e3 entries |
| Managed memory on x86, host touch after a kernel | 4.3 GB/s against 22 GB/s for pinned |

Reading: a GPU phase costs about 10 microseconds per kernel with sync, 15 to 22 per read-back and
30 to 45 per library primitive. Temporary device allocations and pinned allocations must come
from pools made once. Managed memory on x86 is five times slower than a pinned mirror on
hand-over, which confirms section 2.2 of the plan.

### Corrections and problems

- The first-look times of Entry 6 include reading the MPS file and writing the presolved model.
  square47 (27.3 million nonzeros, 1.4 GB file) showed how much that is: 311 s wall, of which
  HiGHS's own log puts the end of presolve at 31 s. The statements made from the first look
  about slow instances have to be checked against the baseline tool's presolve-only times.
- The first run of the statistics wrote the CSR dumps to `/dev/shm`, which is 64 MB in the
  container, and filled it; 10 instances and square47 failed. The directory was removed at once.
  Dumps now go to `/tmp/gprs_csr` on the container's local disk (826 MB for 26 instances, to be
  deleted after the sweep benchmark).
- The node has 288 logical CPUs; the container owns 36 (18 cores). Load from other jobs on the
  node is visible in the load average and cannot be controlled.
- The `where` value `cluster` in `testbed.json` is now `dev-large`, and the runner's set is
  `--set large`.

### Lessons

- Check the size of a scratch location before writing to it; `/dev/shm` in a container is small
  by default.
- A wall time that includes file reading says little about presolve on large files. Time the
  call, not the process.

---

## Entry 9 — 2026-11-26 — Download approved; stock baseline; two instances HiGHS cannot read

### User message

> Yes, normal scratch

(Answer to: "Download the 14 missing instances (3.2 GB compressed) to the normal scratch?")

### What was done

- `gpu_presolve/testbed/download_extra.sh` fetched the 14 files as `.mps.gz` into
  `/home/scratch.acoerduek_sw/gpu_presolve_deps/miplib_extra` (3.0 GB; every size equals the size
  the MIPLIB site reports, every archive passes `gzip -t`). The download waited for the running
  baseline to end. The runner now looks in `dataset_dir` and then in `extra_dataset_dirs`.
- Stock HiGHS baseline, presolve call only, one instance at a time, 25 dev-size instances
  (commit `7465b7698`):

  | Band | Instances | Median | Maximum |
  |---|---|---|---|
  | A (5e4 to 5e5 nonzeros) | 12 | 0.23 s | 4.3 s (satellites2-60-fs) |
  | B (5e5 to 1e7) | 13 | 10.1 s | 139 s (rd-rplusc-21) |

  Below 1e7 nonzeros these times are close to the first-look wall times, so the first look was
  right there; the read-and-write share only matters on very large files.
- The plan and this history are versioned on the local-only branch `local/gpu-presolve-notes`
  (`design_summaries/gpu_mip_presolve/snapshot.sh`); it is never pushed. The user did not answer
  the question whether the notes may go to the public fork, so they stay local.

### Finding: HiGHS cannot read indicator constraints

supportcase21i and gfd-schedulen55f2d50m30k3i (main testbed) have an INDICATORS section in the
MPS file; HiGHS stops with "MPS file reader cannot parse INDICATORS section". The testbed draw
had deliberately covered the MIPLIB "indicator" tag. Without a baseline and without a reader
these two cannot serve the standalone comparison, and indicator constraints are outside the
presolve's scope until Stage 5.

Change: both leave the main list; sorrell3 (A3), istanbul-no-cutoff (A1) and brazil3 (A2) enter,
chosen by the same cover rule among instances without indicator constraints, all three from the
benchmark set, so no further download. The main list has 40 instances and covers 84 of the 85
features (all but "indicator"). In the held-out list blp-ir98 replaces elitserienhandball3i. The
three removed instances are kept in an `indicator` list of `testbed.json` for tests inside cuOpt.
The strictly best replacement for the A1 slot by the rule was blp-ir98, which would have needed
another download; brazil3 covers the same feature (equation knapsack rows in a small instance)
from the benchmark set, at the price of 6 instances in A2 instead of 5.

### Lesson

Check that the baseline solver can read every testbed instance before fixing the testbed. A tag
table says what a model contains, not what a reader accepts.

---

## Entry 10 — 2026-11-26 — Stage 0 measurements on the dev-size instances; connection to the node lost

No new user message; this entry covers the runs that followed Entry 9.

### Measurements (all committed and pushed)

| Commit | Content |
|---|---|
| `ea61c0fb2` | Patched HiGHS on the 31 dev-size instances (per-stage breakdown), the six instances new to the node or the testbed, the four status instances, the testbed change |
| `9948e9bb6` | Patched HiGHS with stages switched off: no probing, no clique merging, Stage 1 rule set, Stage 1+2 rule set |
| `1ac891434` | Sweep-kernel micro-benchmark on 32 matrices (31 dev-size and square47) |

Per-stage time of HiGHS (28 instances at the time; the patched build runs at the speed of stock,
339 s against 338 s): parallel rows and columns 33% (rd-rplusc-21 alone: 69% of its 136 s),
probing 22%, fast loop 18%, pairwise dominated columns 12%, enumeration 6%. Without
rd-rplusc-21: probing 35%, dominated columns 21%. Each slow instance has one dominant stage:
tutaki and supportcase6 pairwise dominated columns (71%, 76%), splice1k1, var-smallemery-m6j6
and piperout-d27 probing (85%, 79%, 82%), ravan the fast loop (58%).

Switch-off runs, remaining nonzeros against full HiGHS (geometric mean over 31) and summed time
(full 350 s): no probing 1.093, 251 s; no clique merging 1.046, 329 s; Stage 1 rule set 1.197,
165 s; Stage 1+2 rule set 1.091, 297 s. Clique merging decides sorrell3 (338,324 nonzeros to
102,026) and gives savsched1 18%. With enumeration off sorrell3 is not reduced at all although
clique merging is on, because the merging that matters runs inside the enumeration stage; so
the "Stage 1+2" switch set understates what Stage 2 reaches.

Status instances: HiGHS presolve does not detect the three infeasible ones (temuka: 15.7 s,
not reduced); yuna is reported unbounded or infeasible after 32 s.

Sweep benchmark: all 32 runs passed their checks against the CPU reference, 8 minutes in
total, GPU otherwise idle.

### Analysis (two workflows, 7 analysts, each followed by a verifier; local files only)

Questions: time per stage and why HiGHS is slow where it is; probing cost, coverage and yield;
reductions by rule; instance structure; kernel variants; host/GPU engine thresholds;
per-instance stage targets and the clique decision. Results pending at the time of writing.

### Problem: connection to the GPU node lost

The command that was to delete the dev-size dumps and compute the statistics of the nine large
instances did not run: `cuopt exec` stopped at the sync flush with "unable to flush session:
session is not currently able to synchronize". Read-only checks: the sync session's beta side
(node smc-522ga-0025) is disconnected ("beta polling error ... unexpected EOF", status
"Connecting to beta"); `cuopt nodes status` shows both node launchers exited with "Read from
remote host <jump host>: No route to host"; `cuopt doctor` reports the GPU node
unreachable. By the user's rule nothing was changed or worked around; the user was asked to
restore the connection.

Not done because of it: statistics, dumps, sweep benchmark and HiGHS runs for the nine large
instances; deleting `/tmp/gprs_csr` in the container.

### Lesson

Pushing each result set as it arrived meant that losing the node cost nothing that had been
measured.

---

## Entry 11 — 2026-11-26 — Node back; design for disconnects; first verified analysis

### User messages

> It's back

(Answer to: "Is the connection to the GPU node back?")

Sent while the work was running:

> i am reserving on slot b again, it should be up in few minutes. in your work assume that the disconnects might happen and we might reconnect, so try to keep things in scratch.

### Analysis, first workflow (committed as `2a610ffc3`)

Four analysts, each followed by a verifier that recomputed every number from the raw files and
read the HiGHS source. The verifiers corrected 28 claims, marked 30 as unsupported and added 41
facts the analysts had missed. Verified results on the 31 dev-size instances:

- HiGHS presolve: 353.5 s summed; shifted geometric mean (0.1 s) per band A1 0.20, A2 0.56,
  A3 2.26, B1 16.9, B2 8.2, B3 10.3 s. Restricted to the Stage 1 rule set: 0.064, 0.086, 0.24,
  2.5, 1.7, 4.7 s. Size explains half of the variation; 25% of the time (88 s) is in stage runs
  that remove nothing (pairwise dominated columns 42 s, probing 27 s).
- Algorithmic costs that are not inherent: no work cap on dominated columns; probing limits
  that count probes, not work; clique merging repeated without a change (8 of 12 s); row
  deletion that walks very long columns (rd-rplusc-21: 0.14 s to hash, 94 s to remove).
- Probing: HiGHS probes 2.0% of the candidates in its first round; 14 of 29 instances get no
  fixing or substitution, five get 91%. What the unprobed candidates would give is not
  measured.
- By rule: HiGHS restricted to the Stage 1 rules removes 87.5% of the nonzeros of full HiGHS
  (an upper bound: some Stage 3 column rules cannot be switched off), with probing and cliques
  97.2%. Clique merging deletes 325,724 rows, 98.7% on sorrell3 and savsched1, almost all
  without any probing.
- At 50,000 to 200,000 nonzeros a full GPU sweep saves 0.2 to 0.6 ms against one CPU thread and
  a GPU pass over a small frontier costs 55 to 167 microseconds against 1 to 3 on the CPU: the
  cascade stays on the host there, and the GPU has to earn its place with wide probing.

The verifiers asked for more measurements: probing with HiGHS's limits off; a Stage 1 reference
with HiGHS's built-in column rules off; Fourier-Motzkin alone off on satellites2-60-fs; three
more counters in the patch for the probing overlay (left for the start of Stage 2).

### Node

- After the reconnect the node was the same machine with a new container (other cores); the
  dumps in the container's `/tmp` were gone, the scratch was intact.
- The statistics and dumps of the nine large instances ran (reading takes 10 to 140 s per
  instance, the largest dump is 3.1 GB). The command's connection ended early with no output
  while the run went on; the run was checked and waited for directly.
- Following the user's message: the dumps were copied to
  `/home/scratch.acoerduek_sw/gpu_presolve_deps/csr_dumps` (11 GB, compared byte for byte); the
  runner got `--skip-done`; the remaining steps went into one driver script
  (`gpu_presolve/baseline/stage0_node_runs.sh`) that marks finished steps, retries failed ones
  at the next start, and holds a lock with a heartbeat so that two drivers never run at once.
  The baseline tool got `--presolve-rule-off`. Commits `de353776e`, `d6981470d`.
- While slot b was being reserved again, the GPU the slot was pinned to was not the GPU visible
  in the container, and a GPU test run failed with "no CUDA-capable device is detected". Nothing
  was changed; the runs wait until the pinned GPU is visible.

### Lessons

- A container's local disk is not a place for anything that takes minutes to recreate once the
  node can be replaced at any time.
- A background command that returns with no output is not a finished run: check the process on
  the node before reading results.

---

## Entry 12 — 2026-11-26 — Slot a; cost model and targets verified; Stage 0 report written and checked twice

### User message

> use slot a for now.

(Sent while the runs waited for slot b: the GPU slot b was pinned to was not the GPU visible in
its container.)

### Node

- `cuopt wt move . a` moved this worktree's sync and runs to slot a. There the pinned GPU matched
  the container's GPU and a short GPU test passed. The driver was started detached; a local
  watcher reports each step.
- Driver steps so far (commits `8c61c4c39`, `d3122eb53`):
  - Timing check: five instances at 0.95 to 1.01 of their times before the container changed.
  - Stage 1 reference with six of HiGHS's own column rules off (`presolve_rule_off=9175456`):
    56.3% of the rows, 86.8% of the columns and 85.7% of the nonzeros that full HiGHS removes
    (57.9 / 95.2 / 87.5 with those rules on); 160.4 s summed.
  - satellites2-60-fs with only Fourier-Motzkin off: the same rows, columns and nonzeros as full
    HiGHS.
  - Sweep benchmark on the nine instances above 1e7 nonzeros: all checks passed. Lane groups
    take 1.14 times the plan's scheme there in the geometric mean (2.5 times on usafa).
  - Patched and stock HiGHS on the large instances and probing without limits: running.

### Analysis, second workflow (commit `f4cc851cb`)

Three analysts with verifiers on kernel variants, engine thresholds and stage targets.

- Scheduling decides: rows binned by length with lane groups take 0.19 times the
  one-block-per-row scheme and 0.10 times one thread per row (geometric mean, 32 matrices,
  compensated). Compensated sums cost 14% of a sweep. Paired against split arrays is not decided
  by time: two variants doing identical work differ by up to 12%.
- A full GPU sweep is cheaper than one CPU thread on every matrix from 71,477 nonzeros up; a
  GPU pass over a small frontier costs 56 to 141 microseconds whatever its size.
- Proposed constants: shift 0.1 s; per-instance rule 1.2 x HiGHS + 0.02 s on the minimum of
  three runs (single runs of identical work differ by up to 15%).

### Stage 0 report (commit `d3122eb53`)

`gpu_presolve/results/stage0/STAGE0_REPORT.md`. I wrote it from the verified findings and had it
checked by three independent passes (numbers against raw files, claims against the findings,
consistency with the plan). The first check found 79 problems; I rewrote the report; the second
check found 40 more, mostly in the rewritten sentences; all were applied.

What the checks caught in my own draft:

- A table cell computed with the wrong size for sorrell3 (Stage 1+2 gap 9.1% instead of 5.3%),
  against a rule stated in the same report.
- A plain-arithmetic figure next to compensated ones; a threshold quoted as if it were the
  measured crossover; a range of two bands quoted for three.
- Hypotheses stated as facts (the cause of rd-rplusc-21's slow row removal; the rescans on
  var-smallemery-m6j6).
- Conclusions carried by one instance stated as general (probing's share in band A; "2% of the
  candidates probed"; clique merging "the whole reduction on two").
- Deliverables of the plan that the draft neither delivered nor listed as open (fixed cost per
  band, the probing budget, three other defaults), and the conditional clique decision not put
  to the user in the plan's own terms.
- "Confirmed" used for choices the measurements did not decide.
- Results that had arrived while I wrote (the Stage 1 reference run) still listed as running.

The report ends with five decisions for the user: how to get a true Stage 1 reference (more
patch switches, or move cheap column rules into Stage 1); whether clique merging is built; the
constant of the time rule and the reading of the reductions gate; scaled or unscaled models for
the gates; one summation order or two.

### Lessons

- A second pass over my own synthesis found as many errors as the passes over the analysts'
  work. Writing from verified summaries does not make the text verified: every rewritten
  sentence is a new claim.
- State for each number which variant and which arithmetic it belongs to, at the place where it
  is used.
- When results keep arriving, the report needs a "state as of" and the analysis scripts need a
  fixed input set; the large-instance sweeps went to their own directory for that reason.

---

## Entry 13 — 2026-11-26 — The user decides the five open points; independent work for six hours

### User messages

> Yes, list them

(Answer to: "Do you want to go through the five decisions in the Stage 0 report now?" I listed
the five decisions with a recommendation each.)

> 1: Option A
> 2: build it
>
> 3. do the recommendations
> 4. B. do the runs with scaling on and off when it comes, so that we see the impact of scaling in presolve.
> 5. do as recommended
>
>
> From now on, work for the rest of the 6 hours of node independenly. if you need to run a benchmark on the cluster to do larger scale test, do it.

### Decisions

1. Stage 1 reference: more switches in the HiGHS patch (simple probing on equations, dual-fixing
   extensions, implied equations, implied-integer conversion), a counter per rule for column
   fixings, and a bracketing run with HiGHS's dual-fixing rule off. The cheap column rules stay
   in Stage 3.
2. Clique merging is built (Stage 2), gated apart from probing.
3. Time rule 1.2 x HiGHS + 0.02 s on the minimum of three runs; reductions gate on remaining
   nonzeros, mean over all instances, every instance below 0.95 explained.
4. Standalone gates on unscaled models. Inside cuOpt the cluster runs are made with scaling on
   and off, to see the effect of scaling on the presolve.
5. One summation order: a frontier kernel in lane-group order; the plan's original scheme as
   the fallback.

### How I read "work independently"

As the go to carry on without checking in: finish the Stage 0 runs, write the decisions into
the plan as version 4, extend the HiGHS patch, and start Stage 1 with the host engine in the
standalone project. Cluster runs are allowed when a larger test needs one. I said so at the
start of my reply and went ahead.

### Started

- Plan version 4 as a draft file (writer and checker), version 3 kept as DESIGN_v3.md.
- Stage 1 code structure for the host engine (architect and adversarial reviewer): headers,
  build files, test harness, six parts with one owner each.
- HiGHS patch version 2 under a separate build name on the node, so that the running
  measurement keeps its binaries (author and reviewer).
- The stock run on the large instances was moved behind the probing run without limits: the
  patched run on them takes up to an hour per instance and gives the same models and times.

---

## Entry 14 — 2026-11-26/27 — Plan version 4; Stage 1 code structure; implementation started

No new user message; independent work as agreed in Entry 13.

### Plan version 4

Written as a draft next to version 3 (so that the people reading version 3 were not disturbed),
checked by a second pass that made 21 repairs, then put in place as DESIGN.md. 780 lines against
484. Sections 4, 8, 9, 11 and 14 are unchanged. Where the user's decisions left room the plan
takes a reading and lists it as open in section 0: which of the two bracket runs is the gate
reference (the stricter one), what "all instances" means for the reductions gate (every instance
with a reference), whether the cuOpt baseline also runs with scaling off (I intend to run it
both ways), and that the clique items other than merging are not decided.

### Stage 1 code structure (commit `0d70325a0`)

An architect wrote headers, build files and the test harness; a reviewer attacked them by
writing down, for each Stage 1 rule, how its implementation would read against the headers, and
fixed 15 defects. Examples of what the review found in the interfaces before any rule was
written:

- The doubleton rule could stop on its budget after deleting equations and before substituting,
  which leaves a wrong model.
- The alias forest chose its root without regard to integrality; for x1 = 2 * x2 between
  integers only one direction is admissible.
- The start step never fixed a column whose bounds were already equal in the input.
- The work budget was scaled by nonzeros only, so a model with many short rows stalled before
  its first pass was complete.
- The parallel-row contract contradicted itself on which rows are compared.

Built and tested with Apple clang 17 on the Mac and with gcc 14 and CMake on the node: 35 tests
pass with and without assertions. Left open by the review: the (hi, lo) pair is not normalised;
the error bound is the plan's first-order formula; aliases have no numeric acceptance test; all
budget defaults are placeholders.

### Implementation (running)

Six parts with one owner each, against the fixed headers: A state operations, B postsolve and
alias forest, F oracle and end-to-end tests first; then the shared bound functions, C row rules
and propagation, D doubleton equations and parallel rows, E dual fixing, driver and hashes. A, C
and F get an adversarial reviewer. Each part builds in its own directory with the Mac's compiler
and tests against a naive reference.

### Node

Patched HiGHS on the large instances: usafa 28 s, a2864-99blp 137 s, motru 182 s, square47
282 s, t11nonreg 3,364 s, sesia 56 s, dlr2 185 s; hawaiiv10-130 and waipa running. The stock
run on them was moved behind the probing run without limits.

## Entry 15 — 2026-11-27 — References with patch v2; the host engine exists and passes both Stage 1 gates

No new user message; independent work as agreed in Entry 13. Node time left at the start of this
entry: about three hours.

### HiGHS references (commits `d90290a20`, `139cc6245`, `223be6a18`, `185404525`, `91b6391b6`, `305e7da25`)

- **Patch v2.** The first reference run still contained HiGHS rules that are not in our Stage 1
  rule set, because they have no switch in HiGHS: simple probing, substitution inside dual
  fixing, implied equations, conversion to implied integers. The patch got a switch for each,
  plus counters for column fixings by rule, bound changes and rows touched per probe.
- **Stage 1 reference** (all of that off, variant `stage1c`): on the 31 instances below 1e7
  nonzeros it reaches 82.8% of the nonzeros full HiGHS removes, in 28.8 s instead of 353.5 s.
  The time target per band dropped a lot against the first reference: rd-rplusc-21 takes 0.61 s
  instead of 126 s once the unswitchable rules are off.
- **Stage 2 reference** (probing and cliques on): geometric mean of remaining nonzeros 1.104
  against full HiGHS, 196 s.
- **Large instances.** Full HiGHS on the nine instances above 1e7 nonzeros takes 28 s to 3,603 s;
  the Stage 1 reference 13 s to 139 s. On four of the nine the Stage 1 rules reach what full
  HiGHS reaches.
- **Probing without HiGHS's limits**: 31 times more probes, 2.3 times more fixings and
  substitutions, 15 times the time, 1.8% fewer nonzeros in the geometric mean. So the limits
  cost HiGHS little on average; three instances gain a lot (supportcase7, bab6, brazil3).

### The host engine (commit `3a387a119`)

Six parts written in parallel against the fixed headers, three of them attacked by a reviewer.
About 5,300 lines of engine and 15,300 lines of tests; 281 tests pass with and without
assertions, with Apple clang 17 and gcc 14. What the reviews and the implementers' own random
tests found before any real instance was run:

- **State (review):** the consistency check rejected valid states, which would have aborted
  every other part's tests. The plan's error bound for the compensated activities is not a
  bound: measured error up to 38 times the formula on rows of same-sign terms. Open, it sits in
  the plan.
- **Row rules (review):** four wrong reductions at magnitudes above 1e10, where the feasibility
  tolerance no longer covers the rounding of a double (a row with a fixed term of 1e11 was
  tightened by one unit per visit until the model was infeasible). Repaired, each with a failing
  test first.
- **Alias forest (own random test):** 1e10 x0 + 1e10 x1 = 3 between integers was accepted as
  x0 = -x1, because the admissibility test divided by the coefficient first. Repaired with the
  test HiGHS uses.
- **Oracle (review):** the result checker would have blamed a correct engine in one case and
  let a wrong one through in two.
- Implementers checked their tests by mutation (state: 76 injected bugs, driver: 62, dual
  fixing: 34), and wrote tests for the survivors.

### First run on real instances (commits `0574fae62`, `5695d8330`)

All 40 testbed instances, release build on one core, three repetitions on the 31 smaller ones:

- **Reductions gate:** geometric mean of nnz(reference) / nnz(engine) = 0.9955 (needs 0.99).
  The nonzero count is equal to the reference on 33 instances. One instance below 0.95: brazil3
  (0.878).
- **Time gate:** 258 s summed against 614 s for the reference; no instance above
  1.2 t + 0.02 s; every band median below the reference (band A1 0.010 s against 0.041 s, band
  B3 0.83 s against 3.18 s, band C 5.2 s against 35.8 s).
- The assertion build passes on all of them and on the four status instances; repetitions and
  the two build modes give the same hash.
- **Where the time goes:** the parallel-row rule (sesia: 107 of 124 s, the only instance slower
  than the reference) and the time outside the rule calls (hawaiiv10-130: 20 of 31 s).

### Mishaps

- The dump step of the new node script wrote nothing and still reported success: the dump
  directory did not exist, and `command | tail` hid the exit code. Fixed by creating the
  directory and counting the dumps.
- A cleanup command killed its own shell (`pkill -f` matched a path in the same command line),
  so the cleanup did not run. Redone without the pattern in the command.

### Started

- A correctness check with HiGHS as the solver (`check/gprs_check.cpp`): LP relaxation before
  and after, MIP solve of the reduced model, postsolved solution tested in the original model.
  Running on the 31 instances.
- Two groups of agents: speed (parallel rows, time outside the rules; results must stay
  bit-identical) and gaps plus audit (what the reference removes and we do not; an adversarial
  audit of the measurement and of the check tool).

### Lessons

- Interfaces first, with a reviewer who writes each rule against the headers before any rule
  exists, made six parallel implementations fit together at the first build.
- Every numerical defect found so far was found by a random differential test or a reviewer
  with a brute-force reference, not by reading.
- A step in a pipeline needs its own success test; an exit code behind a pipe is not one.

## Entry 16 — 2026-11-27 — Correctness on real instances; ablation; the engine made three times faster

No new user message; independent work as agreed in Entry 13. Last hour of the node window.

### Correctness check with HiGHS as the solver (commit `f48a18c2d`)

`check/gprs_check` presolves a dump with the engine, solves the LP relaxation of the original
and of the reduced model, solves the reduced MIP with a time limit, postsolves the solution
and tests it in the original model. On the 31 instances below 1e7 nonzeros:

- No failed check on 30 instances. The relaxation after presolve is never weaker (26 instances
  with both solved) and never above the known optimum (27). A solution of the reduced model
  maps back to a feasible point with the same objective (22 instances; a second run with a
  15-minute limit on the eight instances without a solution added four: 26). Where HiGHS
  proved the reduced optimum it equals the known one (4, and 3 more in the second run: 7).
- var-smallemery-m6j6: HiGHS's MIP solve of the reduced model ran 35 minutes past a 120 s limit
  on one thread and was stopped. Its two LP checks pass in a separate run.
- The assertion build passes on all 40 instances and gives the hashes of the release build.

### One rule family off at a time (commit `0109ea504`)

Nonzeros left against all families on, geometric mean over the 40 instances: row rules 1.134,
parallel rows 1.128, dual fixing 1.077, doubleton equations 1.054, propagation 1.025, row
tightening 1.006. Parallel rows took 159 of the 258 s of version v1, but cannot be dropped:
usafa keeps 5.4 times more nonzeros without them.

### Speed (two agents in separate git worktrees, results required to stay bit-identical)

- **Time outside the rule calls** (commit `0deaecd5c`): the reduced model is read from the
  state in one pass instead of rebuild plus extraction, rebuild compacts in place, the row
  activities are summed without branches. Measured as `v2s`: identical on all 40 instances,
  230 s instead of 258 s.
- **Parallel rows**: keys in an array, grouped by sorting, instead of a node-based set.
  Measured together with the first change as `v2p`, from the agent's working tree before its
  own commit and review, because the node window was closing: identical hash, sizes and
  per-rule numbers on all 40 instances, in the release and the assertion build; 83 s summed
  against 258 s for v1 and 614 s for the reference (minimum of three runs); the rule itself
  9 s instead of 146 s; sesia 19 s instead of 124 s (commit `ca9064afe`).

### Why the reference removes more on seven instances (diagnosis by an agent, not yet reviewed)

A local build of the patched HiGHS reproduced the reference counts on 15 dumps; a diff of the
reduced models by original row and column gave two causes:

- Rows that are parallel up to singleton columns (brazil3, piperout-d27, t11nonreg, dlr2,
  hawaiiv10-130). HiGHS adds the equation of the pair to the other row.
- Redundancy and coefficient strengthening on activities over implied bounds (unitcal_7,
  supportcase7, hawaiiv10-130). The engine makes implied bounds explicit as its last step and
  runs no rule afterwards.

### Lessons

- A measurement window that is about to close is a reason to measure a change before its
  review, not to merge it before its review: the 40-instance run is itself the differential
  test, and the merge waits for the reviewer.
- A solver used as a checker needs its own watchdog: HiGHS ignored its time limit once.

## Entry 17 — 2026-11-27 — The audit corrects the headline; cuOpt adapter written; node window ends

No new user message. The node reservation ended at about 03:45 CET; from here on only local
work.

### Audit of the Stage 1 measurement (an agent told to assume the result is too good)

It rebuilt the engine and the patched HiGHS on the Mac, reproduced the stored hashes on 17
instances and the reference sizes on 15, and ran the check tool against deliberately wrong
engines. What stands: the gate arithmetic, the reference sizes, the determinism claims, and
the time gate on the 31 instances below 1e7 nonzeros (worst case 0.56 of the reference). What
it corrected:

1. **"258 s against 614 s" overstated the lead of v1.** The reference is one run per instance
   with a 64-bit-integer HiGHS build, and its clock contains work outside Stage 1 (column
   hashes for a loop that is switched off, about 81 s; scaling; the MIP data rebuild). On equal
   footing the reference is 435 to 515 s, so v1 led by 1.7 to 2.1 times, not 2.4. The gate as
   the plan defines it (both sides the minimum of three alternating runs) was not measured.
2. **sesia did not support "no instance above the limit"** for v1: 1.11 to 1.32 times an
   equal-footing reference against a limit of 1.2. (v2p takes 19 s there.) Full stock HiGHS
   presolves sesia in 50 s and leaves 202,093 nonzeros where Stage 1 leaves 34 million.
3. **"The results are correct" was too strong.** The check sees relaxations and bookkeeping
   errors; a cut-off optimum only where HiGHS closes the gap (8 of 30 instances). A presolve
   that wrongly answers "infeasible" failed nothing: every check was skipped. Nothing was
   checked on the large and status instances. Proposed and prototyped: crush a known feasible
   point through the presolve; it caught the wrong engines the solver-based check missed.
4. **Equal sizes are not equal models.** On rd-rplusc-21 the engine keeps 1e7 coefficients that
   the reference strengthens (row dynamism 35 times HiGHS's), and v1's output is not a fixed
   point of its own rules on 5 of 15 instances. Cause: implied bounds are made explicit as the
   last step and no rule runs afterwards. Same cause as one of the two reduction gaps.
5. Scripts: the build step hid a failed build behind a pipe; skip-done is keyed on file
   existence; results carry no commit or binary identity; the check runner had no timeout.

The second cause of the gaps is confirmed by the reference's own counters: HiGHS's parallel-row
stage also merges rows that are parallel up to singleton columns, and its count equals the
engine's row gap exactly on brazil3 (550), t11nonreg (97), dlr2 (202,759) and piperout-d27 (14).

### cuOpt adapter (branch `feat/gprs-cuopt-adapter`, commit `057742334`, pushed to the fork)

Written and read-reviewed without a build (no CUDA on the Mac). gprs runs on MIPs before Papilo
or instead of it, behind one switch that is off by default; solutions go back through Papilo
and then gprs. The engine and the adapter's conversion code were compiled locally with stubs
and run on 20,000 random MIPs. Open before a cluster run: first build with GCC and nvcc, a
pre-flight on three instances, and two decisions (whether a benchmark build keeps the engine's
assertions; what happens when gprs refuses an instance in "instead of Papilo" mode). The engine
can return empty rows after its last step, which cuOpt's problem class does not accept; the
same driver fix as in point 4 removes them.

### Started

Agents hardening the check (crush check, failing statuses, timeouts, result identity, script
exit codes), to be attacked afterwards with other wrong engines.

### Lessons

- An audit by someone told that the result is too good found four overstated claims in a
  result that was arithmetically right. The arithmetic was never the weak point; the
  definitions of the two sides were.
- "Same size as the reference" is a weak proxy for "same model". Coefficients, bounds and
  being a fixed point of one's own rules have to be compared too.
- A check has to be shown to fail on a wrong engine before a pass means anything.

## Entry 18 — 2026-11-27 — A wrong reduction found by 80 million random models

No new user message. Local work after the node window.

The reviewer of the gap diagnosis ran the end-to-end oracle far longer than the test suite does:
80 million generated models gave 30 failures on engine v1, 6 of them under default options
(presolve answers infeasible for a model with an optimum; the reduced model has another
optimum; a postsolved point violates a row).

- **Cause in the engine:** the parallel-row rule stores a side of the kept row with that row's
  fixed terms in it, as one double. After dual fixing puts a column at a bound of 1e20 or
  more, the side the other row gives is lost in that term (3 - 4e20 is -4e20). Repair: a pair
  is left alone when the kept row's fixed terms exceed 1e8, the limit row tightening already
  has for the same reason. Commit `6e852cf95`, cherry-picked onto the branch; 294 tests pass;
  the 15 local testbed instances keep their hashes. Not re-measured on the 23 instances
  without a local dump.
- **One case was the oracle's:** coefficient strengthening can write 1.0000000000000002 for an
  integer column, and the oracle rounded that row's end inwards without tolerance.
- After the repair: the same 80 million models and 192 million more pass.

The same review confirmed both gap causes by its own diff and refuted parts of the two
proposed fixes: the row-addition guard as described does not terminate on a three-row example
and needs the same fixed-term limit; the extra rounds after make_explicit are valid but with
one sweep still leave 9 redundant rows on unitcal_7, because the last make_explicit is again
not followed by a rule.

### Lesson

- The test suite's random runs (thousands of models per test) were three to four orders of
  magnitude too short to see a defect with a rate of one in a few million. A long random
  campaign belongs to every rule change, not only to a review.
- Both real numerical defects so far (row rules above 1e10, this one) are the same pattern:
  a huge fixed term next to a side in one double. The engine allows fixing at bounds up to
  1e20; that deserves one rule in the state rather than a guard in each family.

## Entry 19 — 2026-11-27 — Speed changes reviewed and merged; campaign on the merged engine; fixes started

No new user message. Local work.

- **Reviews of the two speed changes:** neither reviewer could make a result differ between old
  and new code (17 dumps, about 94,000 generated presolves under 29 option sets, 12 million
  random state operations, about a million rule calls compared call by call). The parallel-row
  review found no defect. The other found a branch no test covered and added the test.
- **Merged on the branch** (`a6cea9027`, `cfa4c88d5`, with the repair `02eeb05cc`): 300 tests
  pass in both builds; the 15 local testbed instances give the v1 hashes and per-rule numbers.
  The source of the parallel-row change is the one measured on the node as v2p.
- **Random campaign on the merged engine:** 204 million generated models (144 million release,
  60 million assertion build, with and without singleton columns), no failure. 363 models in
  the singleton runs end outside the class the oracle can decide.
- **Fixes started** (agents, each change in its own worktree, attacked, then integrated, with
  campaigns of tens of millions of models required at every step): the reduced model as a
  fixed point of the engine's own rules (rules after implied bounds become explicit, or
  activity sums over implied bounds as HiGHS has them; the agent decides with measurements),
  and rows parallel up to singleton columns behind an option.
- A first attempt at the campaign did nothing: the copied script had lost its execute
  permission and the wrapper printed the error four times and exited 0. Noticed because the
  output was read, not because anything failed.

## Entry 20 — 2026-11-27 — The correctness check is hardened, attacked with 36 wrong engines, and merged

No new user message. Local work; the node window is over.

### What was done

An agent acted on the audit of Entry 17 (the check could not see a cut-off solution, a wrong
status failed nothing, scripts could hide a failed build or reuse an old result). A second
agent attacked the result with deliberately wrong engines and repaired what got through.
Both commits are on the branch (`7ac81c917`, `8a0c23b34`, pushed); engine sources untouched.

- **Crush check** (`check/crush_check.hpp`): feasible points of the original model, from
  solution files, are carried into the reduced model and back. The crushed point must be
  feasible in the reduced model, its postsolve feasible in the original one, the objective
  not worse; in primal-only mode the point must come back unchanged. No solver needed.
- **Structure check** (`check/structure_check.hpp`, added by the review): reads the postsolve
  records and tests, over the reduced bounds, that every original column stays in its bounds,
  every removed row holds, every kept or merged row is implied by its reduced row. No point
  and no solver needed; 0.09 s on kenana against 0.7 s of presolve.
- **Statuses fail:** infeasible, unbounded, limit or failed on an instance with a known
  solution is a failure in the check, the runner and the comparison script.
- **Scripts:** pipefail and tested exit codes; a failed build or failing tests end the run; an
  old summary is removed before a run; every result carries source checksum, binary checksum,
  host and load; a mixed pair of dev and large results from different binaries fails.
- **Solution files:** HiGHS on the original models wrote 98 points for 13 of the 15 feasible
  local instances (none for triptim1 and kenana). The review adds up to four "pushed" points
  per point (columns moved to where bounds and rows stop them): 344 points.

### What the attack showed

- The first version alone let relaxations through: of the reviewer's 23 wrong engines, 13
  passed every check on at least one instance they change, three of them with the solver too
  (bounds of a substituted column lost; a side rounded outward after the gcd division; a
  postsolve offset on an instance without a point).
- After the repair every relaxing engine fails without a solver on every instance it changes:
  96 of 96 (before: 22 without the solver, 58 with it).
- **Still invisible:** a restriction that cuts none of the tested points (an implied bound
  made 1e-4 too tight on uccase9, triptim1, kenana; propagation over a row with an infinite
  term on uccase9); any restriction on the two infeasible instances; two rule paths no local
  instance uses (side rounding after gcd with a fractional side, parallel rows with a ratio
  other than 1 or -1).
- No false alarm found: every rule family switched off in turn, both modes, three cost
  vectors, 714 presolve results, 0 failed checks.

### My verification on the merged tree (engine `cfa4c88d5` + check `8a0c23b34`)

- 300 engine tests pass in both builds.
- The check tool built from the merged tree: 0 failed checks on the 17 local instances in
  both modes without a solver (25 checks per instance with points, 15 without).
- The 17 reduced-model hashes equal the node results of v2p.
- The comparison script gives the old numbers on v1, v2s and v2p (0.9955 over 40) and now
  states that the stored results have no identity and the reference is one run.

### Not done, and what waits for the user

- Nothing of this ran on the node; CMake is not installed on the Mac, so the two CMake changes
  and the new steps of `stage1_host_runs.sh` are untested. The new checks are validated on 17
  of the 44 instances.
- `--require-points` is on for the large set: the step fails for every large instance without
  a solution point until MIPLIB solutions are supplied (needs a converter with the MPS file)
  or the flag is dropped.
- A ratio above 1.05, or an instance reduced to empty where the reference keeps nonzeros, is
  only named by the comparison script; whether it fails the gate is open.

### Lesson

- A check is only as good as the wrong engines it was tested against. The first hardened
  version looked complete and its author's 13 wrong engines were mostly caught; a second
  person with 23 other ones found that half of them passed somewhere. The structure check
  that closed the hole needs no solver and no solution, and costs a tenth of the presolve.
- Solution points test a restriction only where they lie. The remaining blind spot (a bound
  slightly too tight that no stored point touches) needs either points on the bounds or a
  support test for each tightened bound.

## Entry 21 — 2026-11-27 — The structure check tested on random models: one hole found and closed; the campaign tool enters the repository

No new user message. Local work.

### Why

The review of Entry 20 left two rule paths untested because no local instance uses them
(side rounding after the gcd division with a fractional side; parallel rows with a ratio other
than 1 or -1), and the new structure check had been validated on 17 real instances only. The
generator and the exact oracle of the engine tests cover both, so I ran the structure check
next to the oracle on generated models, with the real engine and with wrong ones.

### What was found

- **A hole in the structure check.** For a row that left the model, the violation was divided
  by the largest term of both ends of the row's activity range. A huge bound at one end
  forgave any violation at the other. A row-dropping engine passed on 161 of 24,222 wrong
  results that carry a reduced model, a too-generous redundancy test on 43 of 4,892.
  Each end now has its own size; both counts are 0 after the change.
- **The two untested paths are caught:** the gcd-side engine fails the structure check on
  every model the oracle fails (122 of 122 in 240,000 models); the parallel-ratio engine on
  every relaxing one. What it still passes there are restrictions, which this check cannot
  see by design (the crush check and the oracle do).
- **No false alarm:** real engine, 90 million generated models (60 million plain, 30 million
  with singleton columns), 0 structure failures; the largest violation that passes is 2.2e-11
  against a tolerance of 1e-5. The 17 local instances still pass in both modes.
- **The campaign tool lived only in the session's scratch directory**, which a reboot can
  clear, although a long random campaign is required for every rule change. It is now
  `tools/gprs_fuzz.cpp` with `tools/run_campaign.sh`, built with the engine in both builds,
  and runs the structure check beside the oracle. Commit `9963d5da5`, pushed.
- With the solver (HiGHS, LP 30 s, MIP 60 s) the merged engine has 0 failed checks on the 17
  local instances in default and in primal-only mode. HiGHS proved the reduced optimum on
  piperout-d27 (both modes) and unitcal_7 (primal-only); both equal the known optimum. On
  three instances HiGHS ran past its time limit and was left behind by the tool, as designed.
- A leftover of the first implementation was still running after 7.5 hours: a mutation test
  of the oracle whose mutant never terminates. Stopped.

### Lesson

- The review reported "96 of 96 relaxing engines caught" and that was true on the 17 real
  instances. The hole needed a bound of 1e10 next to a small violation, which generated
  models have and those instances do not. A check should be attacked on generated models as
  well as on real ones, with an exact oracle beside it to say which verdict is right.
- Anything a later step depends on belongs in the repository on the day it is first used.

## Entry 22 — 2026-11-27 — The node is back: merged engine re-measured, MIPLIB solution converter, first build of the cuOpt adapter

### User message (verbatim)

> what's the issue of using the MPS files for converting MIPLIB solutions?
> our final aim is to remove more than highs, so no it shouldn't fail the gate.
> not sure i understand this: "Still open from before: engine assertions in a cluster build, the fallback when the engine refuses an instance in "instead of Papilo" mode, and whether the row-addition rule belongs in Stage 1 or Stage 3."
>
> slot-a is up, you can do node runs again.

### Decisions taken from it

- **Reductions gate is a lower bound only.** An instance where the engine removes much more
  than HiGHS is listed, not failed. The correctness check decides whether such a result is
  right.
- **MIPLIB solutions go through the MPS files.** Nothing stood against it except that the
  node (where the MPS files are) was gone, and that the solution files have to be downloaded.
- My sentence on the three open points was not understandable. It named them without saying
  what each one is. Explained again in plain words, each with a recommendation.

### What was done

- **Node.** Slot a is viking-prod-215 (28 CPUs, one H200, shared machine with a load of 6 to 9
  from other jobs). The scratch data of the earlier node is intact. `/home/scratch.acoerduek_sw_1`
  is not mounted there.
- **Merged engine on the node** (`tools/stage1_host_runs.sh v3`, source `9963d5da5`): the first
  CMake build of the new targets worked, 300 tests pass in both builds, the assertion build
  passes on all 35 dev and status instances, and all 40 reduced models have the hash of
  version v2p. Reductions gate unchanged (0.9955). Summed time 75.0 s on this node. The
  reference's 614.1 s were measured on the other node, so the time gate printed by the
  comparison is across two machines and says nothing new; the reference has to be run here.
- **The hardened correctness check on all 44 instances** is running as the rest of that chain
  (solution files by HiGHS first, then the checks).
- **Converter for MIPLIB solutions** (`baseline/convert_solution.cpp`,
  `baseline/convert_miplib_solutions.py`, commit `93401ff5f`, pushed). MIPLIB lists values by
  variable name, the CSR dumps hold no names. The converter reads the names from the MPS file
  with HiGHS, refuses to write when the dump is not that model (sizes, bounds, costs), and
  prints violations and objective of each point. Tested on a hand-made model (good file, unknown
  name, double value, bad number, wrong dump) and on three real instances with a zero point.
  MIPLIB publishes 53 solution files (107 MB) for 38 of the 44 instances, among them all nine
  large ones; none for radiationm18-12-05, supportcase6 and the four status instances. The
  download waits for the user's OK.
- **cuOpt adapter.** `feat/gprs-cuopt-adapter` merged into `feat/gpu-presolve` locally
  (`c3be9e720`, not pushed until it builds); first build on the node started with assertions,
  ten build jobs, and the dev-container cuDSS shim (uncommitted).
- **Stage 1b design round** started with agents (read-only: four readers, three designs, one
  synthesis, two critiques). Nothing is edited by it.
- **Engine fixes** (workflow of Entry 19) still in review. The reviewer of the fixed-point
  driver found that the first version reports feasible models as infeasible on rows with
  several continuous columns and on sides above 1e10, which the test oracle cannot generate
  (32 of 2.4 million planted models; the base engine 0), and repaired it. It also found that
  the base engine itself gives wrong results on models with coefficient ratios of 1e9 inside
  a row or values of 1e15 and more (24 and 101 of 300,000 planted models).

### Corrections to what I told the user

- I wrote that the adapter "says nothing" when the engine returns no model and Papilo runs.
  It logs one line with the engine's status and "the original problem is kept". What is
  missing is only that a result table does not show it; the analysis has to count that line.

### Lessons

- A list of open points is useless to the reader if each point is only named. Each needs one
  sentence on what it is and what happens with either choice.
- The run chain reads its scripts and sources from the synced tree, so the engine sources in
  that tree are frozen for hours while a chain runs. Work that changes them has to wait or go
  to another worktree. A chain that runs from a copy of its sources would remove this.

## Entry 23 — 2026-11-27 — The adapter builds and passes its pre-flight; three assertion failures turn out to be cuOpt's own; the correctness screen is widened to all 240 benchmark instances

No new user message. Node work under the go of Entry 22 ("slot-a is up, you can do node runs again").

### The cuOpt adapter

- First build on the node (`cuopt compile -a`, 10 build jobs, cuDSS shim uncommitted): no compile
  error. The engine's interface had not moved since the adapter was written without a build.
- The adapter's 18 unit tests pass (before PaPILO and instead of PaPILO, infeasible model, undo).
- Pre-flight on 40 instances (24 MIPLIB, 16 edge-case models of the repository), three modes,
  30 s each (`cuopt_runs/preflight.sh`). The mode came from the environment through a patch that
  is applied for that build only (`cuopt_runs/preflight_mode.patch`), so one build served all
  three modes. 115 of 120 runs end normally. Reductions inside cuOpt equal the standalone ones;
  infeasible, empty and fully reduced models give the right status and objective; scaling off
  works (five instances).
- Five runs aborted on three assertions, all in cuOpt code. Each was traced:
  1. `presolve_data.cu:157` "size mismatch" (satellites2-60-fs, brazil3): fires when the time
     limit ends during cuOpt's own presolve. Reproduced with the new presolve off (brazil3,
     2 s limit). An existing cuOpt defect of assertion builds.
  2. `knapsack.cpp:319` "knapsack separation produced a set that is not a cover"
     (markshare_4_0): fires on the original model with all presolve off as well, and on the
     reduced model after PaPILO. An existing cuOpt defect; a release build returns "no cut".
  3. `solution_publication.cuh:81` "published objective disagrees with the assignment"
     (rmatr200-p5, once): 0 of 6 repeats with the new presolve, 0 of 8 without. The engine left
     that model unchanged, so the solver worked on the same model as on the default path. Not
     explained; a timing-dependent event at the time limit.
- `cuopt_runs/reduce_to_mps.cpp` writes the model the engine leaves as an MPS file. It made
  point 2 testable without the adapter, and HiGHS confirmed the optimum of the reduced
  markshare_4_0 and enlight_hard (1 and 37).
- enlight_hard finds no solution in 30 s instead of PaPILO: PaPILO solves it in presolve, the
  Stage 1 rules leave it as it is. Weaker, not wrong.
- Pushed: merge of the adapter and the pre-flight tools (`af4a94d3e`).

### The screen on all 240 benchmark instances

- Reason: I had recommended engine assertions in the first cluster build. An assertion abort
  there loses a whole batch of 40 instances, and the dev node can run the same check in
  minutes. `testbed/make_benchmark_testbed.py` writes dumps (2.3 GB) and a testbed file for the
  benchmark directory (233 with a best known objective, 7 infeasible; 93 below 5e4 nonzeros,
  93 up to 5e5, 52 up to 1e7, 2 above); the runners take it with `--testbed`.
- Result on the merged engine: assertion build 240 of 240 without a failure (204 reduced, 36
  unchanged); release build 17.0 s summed; the structure check passes on all 240 in default and
  primal-only mode, and so do the cost variants. No crush check yet: there are no points.
  HiGHS is making points now (60 s per instance), the crush check follows.
- So the cluster build needs no engine assertions. This replaces my recommendation of Entry 22.

### Prepared, not submitted

- Benchmark commits for cluster run 1, local only: baseline `af4a94d3e` (PaPILO, scaling on),
  `fdbd9f8a6` (new presolve instead of PaPILO), `384524b34` (PaPILO, scaling off), `c0b04112c`
  (new presolve, scaling off). No baseline exists at this branch's upstream base `0d0915a8c`.
- `baseline/stage1_timing_runs.sh` and `analysis/stage1_time_gate.py`: reference and engine in
  three alternating passes on one machine. To run when the node is quiet.

### Engine fixes (agents, still integrating)

- Both fixes gave wrong results as first written; both reviewers repaired them. The row-addition
  rule multiplied small errors by its factor; the repair takes only factors up to 1 in size.
  It keeps HiGHS's counts on brazil3 and piperout-d27.
- Three defects of the base engine came out of the reviews, on generated models with extreme
  constants: a merge of parallel rows of different scale next to a fixed term of 1e7 and more
  returns "infeasible" for a feasible model; the integer row tightening rounds a side of 1.9e9
  times a scale of 125 the wrong way; a doubleton substitution leaves a coefficient of 1e-9
  that the row rules take literally. Not repaired yet.

### Lessons

- A pre-flight in an assertion build finds the host solver's own defects too. Each abort has
  to be reproduced without the new code before it is blamed on it or dismissed; two of three
  reproduced that way within minutes.
- Before asking the cluster for a correctness answer, ask whether the dev node can give it:
  the presolve of all 240 instances takes 17 seconds.

## Entry 24 — 2026-11-27 — The design round for the GPU propagation says the GPU is aimed at 7% of the time; the state build is 46%

No new user message.

### What ran

- A read-only design round for Stage 1b with agents: four readers (host state, Stage 0 kernels,
  cuOpt's load-balanced propagation, the plan's duties), three independent designs (validity
  first, performance first, smallest first), one synthesis, two critics. Output kept in
  `design_summaries/gpu_mip_presolve/stage1b_design/` (`final_design.txt`, `critiques.txt`,
  `result.json`).
- Correctness check of the 31 dev instances of the testbed on the merged engine: no failed
  check. Structure passes on 31, the crush check on the 24 instances with HiGHS points, HiGHS
  proved the reduced optimum on 8 and all 8 equal the known value. Seven instances have no
  point (ns1208400, satellites2-60-fs, brazil3, kenana, kasavu, supportcase19, cygnet).

### The design in short

- The host state stays the only authority. A device call takes the propagation list and
  returns bound changes of integer columns and implied-bound candidates of continuous columns;
  the host applies them through its own functions. The device never updates an activity: it
  sums every row from scratch in the order of `numerics.hpp`, so its candidates have the bits
  of the host rule on a freshly summed row. A reference backend in plain C++ defines the round;
  the CUDA backend must return the same bits. Seven milestones.
- The validity critic could not break the round; it found two serious defects at its edges
  (the hand-over check against an image that still holds dead rows; a byte budget that depends
  on the machine).

### What the performance critic found, and what I measured after it

- In the stored node results (40 instances, 75.0 s): rules 35.8% (parallel rows 11.9, doubleton
  9.2, propagation 6.9, tightening 5.4, row rules 2.1, dual fixing 0.3), building the state
  45.7%, writing the reduced model 8.1%, its hash 8.0%, storage 2.2%.
- The critic's probe on the 17 local instances: the first propagation call holds 92.5% of all
  propagation work, and under the planned rule (not in the first cascade, 5e5 nonzeros at the
  call, frontier work 20,000) none of 392 calls goes to the device. The seven calls with enough
  frontier work are on models that have shrunk below 5e5 nonzeros and change 0 to 10 bounds.
- My split of the state build on the node (`microbench/state_build_probe.cpp`): the fold of all
  row activities is 6 to 14% of it; the transpose with the cross positions and the locks, the
  array copies and the first touch of the new arrays are the rest. waipa: 9.65 s, of which the
  fold 1.36 s and the rest 8.29 s (32 ns per nonzero). hawaiiv10-130: 7.23 = 0.84 + 6.39.
  dlr2: 4.45 = 0.38 + 4.08. sesia: 3.69 = 0.22 + 3.46.
- Stage 0 measured the device transpose by a stable radix sort at 467 microseconds + 0.279 ns
  per nonzero: 72 ms for waipa, before transfers.

### Consequence

- The plan puts the GPU of Stage 1 on propagation after the first host cascade. That region is
  almost empty on the testbed. The time is in the state build, which the plan keeps on the
  host ("the model exists on the host only until the first cascade is done").
- A device state build (transpose, locks, fold, each with the bits of the host) cannot change a
  result, so it needs no threshold that can change a result, only a size switch.
- This changes the plan, so it goes to the user as a decision before any device code is
  written.

### Lesson

- The plan fixed where the GPU works before the host engine existed. The first thing to do
  with a finished host engine is to read where its time goes. A per-phase split of 40 stored
  results was enough; it did not need a line of device code.

## Entry 25 — 2026-11-27 — The two engine fixes come back integrated; my own check stops on the new rule

No new user message.

### What the agents delivered (workflow of Entry 19, 6.3 hours, five agents)

- Branch `stage1-fixes-integrated` (`8ca189992`, eight commits on `cfa4c88d5`): the driver series
  (the reduced model is a fixed point of the rules: implied bounds become explicit and the rounds
  go on), the row-addition series (an equation is subtracted from a row that is parallel to it
  up to singleton columns), each with its reviewer's repairs, and the integration repairs.
- Each fix gave wrong results as first written and was repaired by its reviewer (Entry 23). The
  plain combination of the two repaired series set the fixed-point flag on models that a second
  presolve still changed (1,064 of 3.0 million); four causes were repaired by the integrating
  agent, each with a failing test first. No wrong reduction came with that defect.
- Reported: 370 tests in both builds; 114.4 million generated models in the release and 48.0
  million in the assertion build; of 943 failing seeds 942 fail on the base engine or one change
  alone as well (extreme constants), one ends in a base-engine defect that only the combination
  reaches (a coefficient of -3e-9 left by a doubleton substitution).
- Reductions on the 15 local testbed instances: geometric mean of nnz(reference)/nnz(engine)
  0.9905 before, 1.0004 now. brazil3 and piperout-d27 on the reference's counts, unitcal_7
  below it (107,633 against 108,302 nonzeros). Time 1.22 times the old engine in sum, almost
  all from the driver change (unitcal_7 1.52, kenana 1.37, rd-rplusc-21 1.37).

### What I did with it

- Merged onto the current branch in a scratch worktree (`stage1-v4`, `0ee2ce720`; one conflict
  in `tools/run_engine.py`, both sides kept). 370 tests pass in both builds on the Mac and on
  the node (first build there from a clone of the fork on the scratch, so the synced tree
  stays as the running chain needs it).
- The sizes of the report are reproduced on the 17 local instances; the fixed-point flag is set
  on all 17.
- 34 of the 36 wrong engines of `check/make_mutants.py` still apply to the new sources; the
  recipe for `r_explicit` was updated to the new `make_explicit`, after which all 36 apply.
- **My hardened check fails on brazil3 and piperout-d27**: `structure_default_stack` and
  `structure_primal_stack`, message "parallel records missing at ...". The row addition pushes a
  new postsolve record that the structure check, written before the rule, reads as a
  parallel-row record. In a probe campaign of 240,000 generated models the oracle reports no
  failure and the structure check fails on 1,563 results. This is a gap of the check, not a
  wrong reduction; but the check's own header says its argument has to be reread when a rule
  combines rows, and until it is extended the rule is unchecked by it.
- Started: agents that extend the structure check to the new record (a row that had an equation
  subtracted must be proved against its own reduced row and the equation's), add wrong engines
  for the two new rules, and then attack the extension. Also started on the node: the merged
  engine on all testbed and benchmark instances (sizes; times indicative), and an independent
  random campaign in both builds.
- Nothing of this is merged into `feat/gpu-presolve` yet.

### Lesson

- A new rule needs three things before it merges: its own tests, the campaign, and the
  independent check taught to read what the rule leaves behind. The agents who wrote and
  reviewed the rule could not do the third: the check was merged on another branch after
  theirs was cut. Branches that run for hours should be re-based on the check before their
  final verification, or the merge has to budget for it.

## Entry 26 — 2026-11-27 — The owner answers: large scratch, gf2, critical bugs, row addition in Stage 1, scaling off, cuSPARSE, the loop of the stages; and cuOpt's own reductions

### User messages (verbatim)

> you can just cd /home/scratch.acoerduek_sw_1 and it will mount.
> enlight_hard has a special presolve routine: gf2
> if you see a critical bug in current cuopt, generate a PR on a separate session for merging to upstream.
> moving the reduction to stage 1 is okay.
> let's keep scaling off for all develoment and decide if we want to add scaling and debug scaling issues at the end, what do you think?
> you can try cusparse for transpose and spmv or other sparse work to speed up the program and development.
> The stages run in a loop, right? After we finish stage 1, we go to stage 2, and then after stage 2 does reductions, we can go back to stage 1. Is that correct?

and, while I was working on that:

> cuopt has other special reductions in presolve. bhw ,bve, indicator, single_lock_dual_aggregation. try to find a way for them in the plan.

### What I did with each point

- **Large scratch.** Tried at once: inside the dev container `cd /home/scratch.acoerduek_sw_1`
  fails ("No such file or directory"; the container has only `/home/scratch.acoerduek_sw`
  mounted). The automount works on the node itself, which `cuopt exec` does not reach, and I do
  not ssh to the node or touch the docker setup. The download of the MIPLIB solutions is
  therefore still not done; asked the owner again.
- **gf2.** Found it: `GF2Presolve` is one of cuOpt's own reductions registered as a PaPILO method
  (`third_party_presolve.cpp:726`). In "instead of PaPILO" mode it does not run, which is why
  enlight_hard found no solution there. My note of Entry 23 ("PaPILO solves it in presolve") was
  wrong about who solves it.
- **Critical bugs.** Started the separate session for the abort when the time limit ends during
  cuOpt's presolve (size mismatch in `presolve_data.cu:157`; what a release build does on that
  path is unknown, so I count it as critical). The knapsack "not a cover" assertion is handled
  in release builds (no cut) and stays a proposal card. Saved as a standing rule.
- **Row addition in Stage 1.** Accepted; plan amendment A2.
- **Scaling off for development.** I agree, with two additions: measure once what scaling is
  worth for cuOpt as is (PaPILO with scaling on against off), so the size of the postponed
  question is known; and cuOpt scales before its presolve, so the question at the end is likely
  "scale after the presolve". Plan amendment A3.
- **cuSPARSE and the GPU part of Stage 1.** Taken as the go for the state build on the device.
  Started agents on a first measurement: the transpose with cross positions and locks through
  cuSPARSE and through a radix sort, compared array by array with the host state, with
  transfers, on fifteen instances; then a review of the numbers. Plan amendment A4.
- **The loop.** Yes: "stage" in the plan is a development stage; at run time the rules of all
  stages are phases of one loop that goes back to the cheap cascade after the wide phases
  (section 3). Plan amendment A7.
- **cuOpt's own reductions.** Read all five. Four run only inside the PaPILO call (indicator
  strengthening on the PaPILO problem before its rules; GF2, BHW coefficient reduction and
  single-lock dual aggregation as PaPILO methods); block BVE runs later in cuOpt's own presolve
  on the probing cache and is not affected. Plan amendment A6 gives each a way: now, adapter
  mode 3 ("new presolve, then cuOpt's own reductions") through the reduction allowlist that
  already exists, so no PaPILO rule runs and the four are kept; later, a native port of each in
  the stage that fits (BHW next to row tightening, GF2 and single-lock with Stage 3, indicator
  strengthening as a first row-appending phase, block BVE fed by Stage 2's implication export).
  Mode 3 is written (`gprs_presolve_mode_t::THEN_OWN_REDUCTIONS`, tests extended to mode 3) and
  is building on the node.
- The plan is now version 4.1 (`DESIGN.md`, amendments A1 to A7 at the top; version 4 kept as
  `DESIGN_v4.md`).

### Also since Entry 25

- The merged engine on the node (tag `v4pre`, from a clone of the fork on the scratch):
  reductions gate 1.0028 over the 40 instances, nothing below 0.95; t11nonreg and supportcase7
  on the reference's count, dlr2 below it, ns1208400 10% below; assertion build clean on the
  testbed and on all 240 benchmark instances; 84.0 s against 75.0 s on a busy node. My own
  campaign on the node: 55.2 million generated models in both builds, no oracle failure, no
  assertion.

### Lessons

- I wrote "PaPILO solves enlight_hard in presolve" from one log line ("Optimal solution found
  during presolve") without asking which method did it. The owner knew at once. A reduction that
  looks surprising for a general presolver is a reason to look for a special routine.
- "Instead of PaPILO" silently dropped four reductions that are cuOpt's, not PaPILO's. The plan
  had a line for them ("Stage 5 ports them or drops them"), so the first cluster arm would have
  mixed two questions. The comparison arm has to keep everything that is not the thing compared.

## Entry 27 — 2026-11-27 — No scaling in any benchmark; reductions are compared first; the first reductions table against cuOpt's PaPILO

### User message (verbatim)

> scaling doesn't matter much and we want to eliminate the effects of possible bad scaling on our presolve. at first we should compare reductions, the cuopt quality tuning comes later. Yes, you can use the normal scratch. Do not also use scaling in the benchmark.

### What changed

- No benchmark arm uses cuOpt's scaling, also no "cuOpt as is with scaling on" arm, which I had
  proposed to learn what scaling is worth. The owner's order is: reductions first, solver quality
  inside cuOpt later. Cluster arms when they come: PaPILO with scaling off against mode 3 with
  scaling off.
- The MIPLIB solutions went to the normal scratch: 44 instances, 38 with converted points, none
  failed; published objectives are reproduced with violations of 5e-7 or less.

### What was done

- **Adapter mode 3** built on the node; the adapter's 27 unit tests pass in modes 1, 2 and 3.
  enlight_hard is solved in presolve again in mode 3 (GF2 after the new presolve). Committed
  and pushed (`21ef0f15d`, `fa95ec18c`).
- **A presolve-only report inside cuOpt** (disabled test
  `gprs_presolve_report.DISABLED_reductions_report`): it calls the presolver as the solver does
  (budget, thread count, indicator strengthening, dual reductions), without solve and without
  scaling, for PaPILO with cuOpt's own reductions (mode 0), the new presolve alone (mode 2) and
  mode 3. Run on all 240 benchmark instances in six processes: no abort. It agrees with the
  solver logs of the pre-flight on the instances compared.
- **Comparison script** `analysis/bench240_reductions.py`, per size class and in total.
- **Crush check on the 240** (engine without the fixes, HiGHS points of 60 s): 173 instances with
  usable points, 2,153 points, no failed check in default and primal-only mode; 67 instances
  have no point.

### First numbers (remaining nonzeros, geometric mean of after / before, 240 instances)

| Size class | Instances | New presolve (Stage 1 rules, with the fixes) | PaPILO with cuOpt's reductions |
|---|---|---|---|
| below 5e4 nonzeros | 93 | 0.864 | 0.748 |
| 5e4 to 5e5 | 93 | 0.774 | 0.669 |
| 5e5 to 1e7 | 52 | 0.723 | 0.385 |
| above 1e7 | 2 | 0.315 | 0.312 |
| all | 240 | 0.790 | 0.616 |

- nnz(PaPILO) / nnz(new) is 0.78 in the geometric mean; the new presolve leaves less on 14
  instances, the same on 46 and more on 180. The distance is largest between 5e5 and 1e7
  nonzeros (0.53).
- Furthest behind: ex9 (PaPILO solves it), ex10 (131 nonzeros left of 1.16 million), neos8,
  neos-2987310-joes, neos-3402294-bobin, highschool1-aigio, neos-787933, rocII-5-11, brazil3,
  supportcase12, co-100.
- Ahead: app1-2 and app1-1 (PaPILO's result is larger than the input), ns1116954, ns1208400,
  buildingenergy.
- cuOpt's four reductions after the new presolve are worth 0.976 against the new presolve
  alone (with the engine before the fixes inside cuOpt): enlight_hard, neos-787933 (half),
  var-smallemery-m6j6, the two cryptanalysis instances.
- HiGHS's full presolve on the 240 (added when its last instance ended, ns1760995 at the
  30-minute limit): 0.613 of the nonzeros remain in the geometric mean, against 0.616 for
  PaPILO with cuOpt's reductions and 0.790 for the new presolve; from 5e4 nonzeros (147
  instances) 0.523, 0.544 and 0.746. nnz(HiGHS) / nnz(new) is 0.776; the new presolve leaves
  less on 19 instances, the same on 51, more on 170. PaPILO against HiGHS is 1.005 (less on 97,
  equal on 58, more on 85). Furthest behind HiGHS: ex9 (reduced to empty), ex10 (8 nonzeros
  left), neos8, bobin, highschool1-aigio, sorrell3, rocII-5-11, co-100, fast0507, rd-rplusc-21.
  Ahead: the two nursesched instances (HiGHS's result is 2.4 times the input), piperout-27 and
  piperout-08, app1-2.
- Summed presolve seconds on a busy node: new presolve 18, PaPILO 1,127 (4 threads), HiGHS
  4,851 (1 thread). The new presolve inside cuOpt took 128 s for the same work that takes 18 s
  standalone, so the adapter's conversions cost several times the engine; to look at.
- The merged engine's check with HiGHS on the 31 dev instances: every check passes except the
  two structure-stack failures on brazil3 and piperout-d27 (the known gap of the check). HiGHS
  proved the reduced optimum on 9 instances, all equal to the known value, among them
  ns1208400, where the engine now leaves 10% less than the Stage 1 reference.

### Lesson

- The owner's order of questions was not mine. I had set up solver runs with four arms and a
  scaling question; the first thing wanted is a table of what each presolve leaves, which needs
  no solver, no cluster and twenty minutes of the dev node. Ask what is being compared before
  building the comparison.

## Entry 28 — 2026-11-27 — Which rules make up the gap to HiGHS on the 240 instances

No new user message. Follows the owner's "at first we should compare reductions".

### What was run (dev node, presolve only, no scaling)

- The instrumented HiGHS build on the 240 benchmark instances, three passes: full (with
  reductions and seconds per stage), restricted to our Stage 1 rule set (the Stage 1 reference),
  and all cheap rules with the expensive stages off (probing, enumeration, clique rules,
  aggregator, sparsify, Fourier-Motzkin, inequality strengthening, pairwise dominated columns,
  implied integers). Minutes each.

### Findings

- **The Stage 1 gate holds on 240 instances, not only on the 40 of the testbed.** Against HiGHS
  with our Stage 1 rule set the engine leaves less on 18 instances, the same on 220 and more on
  one (traininstance2, 3 nonzeros); ratio 1.002.
- **The geometric mean of remaining nonzeros is carried by three instances** that a full
  presolve reduces to almost nothing (ex9, ex10, enlight_hard): HiGHS full is 0.612 with them
  and 0.676 without. Reported from now on without them, next to the arithmetic mean and the
  median, and with the solved instances counted apart.

| Presolve | Geometric mean without the three | Mean | Median |
|---|---|---|---|
| Host engine, Stage 1 rules | 0.788 | 0.841 | 0.975 |
| HiGHS, our Stage 1 rule set | 0.789 | 0.842 | 0.976 |
| HiGHS, all cheap rules | 0.753 | 0.814 | 0.932 |
| HiGHS, full | 0.676 | 0.760 | 0.863 |
| PaPILO with cuOpt's reductions | 0.689 | 0.759 | 0.868 |

- **About a third of the gap to full HiGHS is cheap rules** that plan decision 1 left for
  Stage 3: the single-column rules (dominated column by reduced cost, free and singleton column
  substitution, column stuffing, the dual-fixing extensions), simple probing on equations,
  implied equations, parallel columns. They take HiGHS 146 s more on the 240 (302 against
  156 s). Furthest behind there: neos-3402294-bobin, highschool1-aigio, neos-1445765,
  supportcase12, nw04, brazil3.
- **Per stage of the full run** (mean share of an instance's nonzeros removed; share of HiGHS's
  time): initial scan 10.9% (3%), parallel rows and columns 4.7% (6%), fast loop 4.2% (9%),
  sparsify 1.7% (0.4%), enumeration 1.3% (13%), pairwise dominated columns 0.8% (12%), probing
  0.6% (52%), aggregator -0.3% of the nonzeros but 1.7% of the rows (0.1%).
- Probing and enumeration take 65% of HiGHS's time for under 2% of the nonzeros in the mean,
  but they decide single instances: enumeration removes 56 to 76% on ex9, ex10, neos8 and
  sorrell3.

### Consequence, proposed to the owner

- Order of the rules after Stage 1 by what they are worth on the benchmark set: the cheap
  column rules first, then sparsify and the aggregator, then pairwise dominated columns, then
  enumeration, probing and cliques. The plan had probing and cliques (Stage 2) before all of
  these.

### Lesson

- A geometric mean of a ratio that can reach zero is decided by the instances that reach it.
  Three of 240 moved HiGHS's figure from 0.68 to 0.61. Look at the distribution before
  quoting one number, and say which instances carry it.

## Entry 29 — 2026-11-27 — The owner: cheap rules first

### User message (verbatim)

> Yes, cheap rules first

(the answer to the question of Entry 28: should the cheap column rules come next, before probing
and cliques).

### What was done

- Plan amendment A8: after Stage 1 the order is by worth on the benchmark set: the cheap column
  rules (Stage 1c), sparsify and the aggregator, pairwise dominated columns, then enumeration,
  probing and cliques. Reference for Stage 1c: HiGHS with its expensive stages off, on the 40
  and on the 240 instances.
- Started on the node: twelve short HiGHS passes on the 240 with all cheap rules and one rule
  switched off each (rule bits 5 dominated column, 7 forcing column, 8 free column
  substitution, 17 dual fixing, 18 zero-cost singleton, 19 column stuffing, 23 weakly dominated
  column; and the switches of our instrumented build for parallel columns, simple probing on
  equations, the dual-fixing substitution, implied equations, implied-integer conversion), to
  rank the rules before any is built.
- Started with agents: a read-only design round for these rules (readers of HiGHS's and
  PaPILO's implementations, of our engine and checks, and of the measurements; three designs;
  a synthesis; a validity critic and a worth-and-cost critic). The brief carries the lesson of
  Entry 25: a rule that leaves a new postsolve record comes with the check that reads it and
  with wrong engines the checks must catch.
- The separate session on the cuOpt abort reviewed the open upstream PR that holds the same
  fix; by its report the PR's other changes work on its own cases, and a problem emptied with
  presolve off still reports an infinite bound. Nothing was posted upstream. Not verified by me.

### Added to Entry 29: the ranking of the cheap rules (one rule off at a time, 240 instances)

| Rule switched off in HiGHS | Nonzeros without / with | Rows | Columns | Instances that change | Largest losses |
|---|---|---|---|---|---|
| Parallel columns | 1.010 | 1.002 | 1.014 | 39 | nw04 1.92, decomp2 1.53 |
| Simple probing on equations | 1.010 | 1.011 | 1.022 | 8 | highschool1-aigio 3.04, neos-1445765 1.96, brazil3 1.86 |
| Free column substitution | 1.009 | 1.003 | 1.002 | 44 | mcsched 1.63, swath1 and swath3 1.24 |
| Dual-fixing substitution | 1.006 | 1.008 | 1.008 | 40 | supportcase12 1.75, the two radiation instances 1.33 |
| Zero-cost singleton | 1.001 | 1.000 | 1.006 | 16 | timtab1 1.10 |
| Implied-integer conversion | 1.0003 | | | 7 | ns1208400 1.11 |
| Implied equations | 1.0002 | | | 11 | |
| Dominated, weakly dominated, forcing column | 1.0001 each | | | 3 to 4 | |
| Column stuffing | 1.0000 | | | 1 | |

- Four rules carry the stage. Switching off HiGHS's whole dual fixing loses exactly what
  switching off its substitution extension loses, so the lock rule we have is all of the rest.
  That extension also costs HiGHS 144 of its 302 seconds.
- The rules named first in the textbooks for columns (dominated column, forcing column, column
  stuffing) are worth nothing measurable on this set.

## Entry 30 — 2026-11-27 — The node chain of the engine before the fixes ends; a published solution that is not feasible enough

No new user message.

- The chain of tag v3 (Entry 22) ended after 4 hours 20 minutes. Every step passed except the
  last: `check_large` fails on one check, `solution_points` of hawaiiv10-130. All other checks
  on the nine instances above 1e7 nonzeros pass (structure in both modes, crush with 2 to 6
  points per instance, cost variants), as do the status instances.
- hawaiiv10-130 has no usable point. HiGHS found none in 30 minutes, and the two solutions
  MIPLIB publishes violate bounds by 1e-6 and rows by 2.8e-7 and 5.3e-5 on the model HiGHS
  reads, above the 1e-7 the check asks of a point. MIPLIB accepts solutions at 1e-6.
  To do in the check after the structure-check work is merged: take a published point at
  MIPLIB's tolerance and judge the crushed and postsolved point against the violation the point
  came with, or polish it (integers fixed, LP for the rest).
- So the hardened check has now run on all 44 instances for the engine before the fixes: no
  failed check, one instance without a crush check.
- Results of v3 and v4pre committed and pushed (`20775dd6a`).
- With the chain over, the engine sources of the synced tree are free again, and the timing of
  reference and engine in turn was started (`viking215_v4pre`, the merged engine from the
  scratch build).

### Lesson

- "Feasible" has a tolerance, and the check's is ten times tighter than the one the published
  solutions were accepted with. `--require-points` did what it was built for: it turned a
  crush check that silently did not run into a failed step.

## Entry 31 — 2026-11-27 — The structure check learns the row-addition record; its reviewer finds three holes in the new arithmetic

No new user message.

### What the agents delivered (2.9 hours, two agents)

- **Extension** (`check-row-addition`, `066d2ded8`): the stack walk accepts `row_addition`
  records; the records, read in reverse, give every original row the equations it is made of,
  and such a row is proved against its own reduced row and the reduced rows of those equations
  together. The proof is a weak-duality bound with multipliers read from the data, so nothing
  about what the engine did is assumed; the factor of a record is not used. A row whose reduced
  rows have no common point, or whose search is above the work limit, is counted as not tested,
  never as passed. Eight wrong engines for the two new rules; 11 tests.
- **Review** (`review-check-row-addition`, `4f4a9e013`): no for the extension as it stood, yes
  with three repairs, each with a failing test first.
  1. The search took points from hyperplanes that do not meet: a singular system was recognised
     by a pivot equal to zero, which is often only the rounding of zero. A row violated by 0.5
     passed on a hand-made result with two nearly parallel equations.
  2. A coefficient counted as zero below 1e-7 of the terms of the multipliers, so multipliers
     that cancel hid a coefficient of the row's own size (a row reaching 5000 against a side
     of 1 passed).
  3. A violation was divided by the largest term of the bound, and the largest terms were sides
     of the equation that cancel (a violation of 50 passed as 5e-6). This is the same kind of
     hole as the one of Entry 21.
- Reported after the repairs: 385 tests in both builds; the engine on 132 million generated
  models with no oracle failure and no structure failure; 0.19% of the rows the new test takes
  are not tested; 18 wrong engines with no relaxation missed beyond the stated tolerance.
- Left open by the reviewer: a row is tested to 1e-5 times the largest multiplier of its
  equations (the engine's own factors are at most 1; multipliers up to 1e6 occur on scaled
  generated models); violations are relative to max(1, largest term), so rows with tiny terms
  are tested to 1e-5 absolute.

### What I checked

- Merged onto the work branch `stage1-v4` (now `4f4a9e013`, pushed to the fork). 385 tests in
  both builds on the Mac and on the node.
- The 17 local instances without a solver, both modes: no failed check. brazil3: 550 rows
  tested through their equation, none not tested, largest multiplier 1. piperout-d27: 14.
- Started on the node from a second clone: the campaign in both builds with and without added
  singleton columns, and the solver-free check on the 240 benchmark instances and on the whole
  testbed including the nine large instances (dlr2 has 202,759 subtractions).

### Lesson

- The third hole is the second time a violation was scaled by a quantity that the violation
  does not depend on. Whenever a check divides, ask what makes the divisor large, and whether
  that can be something the tested row does not contain.

### Added to Entry 31: verification of the merged engine with the extended check on the node

- From a second clone of the fork on the scratch (`4f4a9e013`): 385 tests in both builds.
- Campaign with the repository's tool, 55.2 million generated models (release 20.4 million plain
  and 20.4 million with added singleton columns, assertion build 7.2 million each): no oracle
  failure, no structure failure, no assertion. With singleton columns 105,494 rows were tested
  through their equation and 217 counted as not tested.
- Solver-free check, both modes: the 240 benchmark instances with no failed check (173 with
  points; 9,838 rows that had a subtraction, 9,836 tested through their equation); the 31 dev
  instances with no failed check, and with the published MIPLIB points all 31 now have a usable
  point; the status instances pass; of the nine large instances eight pass and hawaiiv10-130
  fails `solution_points` as before (no usable point). dlr2: 202,759 rows that had a
  subtraction, 202,040 tested through their equation, none not tested. hawaiiv10-130: 555 with
  a largest multiplier of 100. t11nonreg: 97.
- `stage1-v4` then took the current project branch (`27416ef78`, 385 tests on the Mac), so the
  merge into `feat/gpu-presolve` is a fast-forward. It waits until the GPU state-build agents
  have finished, because their bench tool is built against the engine headers of the synced
  tree and the engine library of the earlier version.

## Entry 32 — 2026-11-27 — The state build on the GPU, measured: the GPU is not the limit, the host's memory is

No new user message. Follows the owner's "you can try cusparse for transpose and spmv or other
sparse work" (Entry 26).

### What the agents built and measured (3 hours, two agents)

- `gpu_presolve/device`: an own CMake project (CUDA, cuSPARSE, CCCL; no cuOpt, no rmm) that
  builds on the device the arrays the constructor of `state_t` derives from the matrix (the
  column-wise copy, the cross positions, the locks) and returns them as vectors. Three
  transposes: cuSPARSE `csr2csc` with the entry positions as its values (so the cross positions
  come out of the same call), a stable radix sort of the positions by column, and a counting
  scatter with a segmented sort. A bench tool compares every array of every run with the host
  state. Committed and pushed (`6b6d1ccd9`); not integrated into the engine.
- The reviewer reran everything, injected faults on the device to see that the comparison
  notices them, added 24 hand-made models (empty rows and columns, dense lines, duplicates,
  unsorted rows) and a model above 2^31 nonzeros.

### Findings (15 instances from 7e4 to 2.6e8 nonzeros, one H200)

- **Bits:** equal to the host in every one of 3,734 device builds the reviewer compared.
- **The device work is negligible:** 40 ms (cuSPARSE) to 48 ms (radix sort) for waipa's 2.57e8
  nonzeros, 0.13 to 0.27 ns per nonzero from 5e6.
- **End to end the device path takes half the time of the host loop** for these arrays, summed
  over the 15 instances (21.6 s against 10.2 s; 2.0 as a first call in a fresh process).
  **Three quarters of the device path is the host sizing the result vectors** (zero fill and
  first touch of new 4 KiB pages at 2.7 GB/s); the download itself runs at 18 GB/s.
- **Huge pages behind malloc** (a glibc tunable, same binary) take the host constructor from
  34 s to 24 s with no GPU at all, and the device path from 10.2 s to 5.1 s; the ratio is then
  3.3.
- **Crossover:** as a presolve call gets it (first build of a fresh process), the device loses
  up to 8.5e5 nonzeros and wins from 1.3e6 (1.9 to 2.0 times), 1.1 to 2.7 times up to 8e6.
  The first report's "wins from 2e5" did not survive the review.
- **Which transpose:** it does not matter for speed (within 1% end to end). The radix sort has
  the host's order by a documented property (CUB: stable) and no 32-bit limit; cuSPARSE's order
  inside a column is only observed, and its first call in a process costs 20 ms more.
- **Pinned staging does not pay**, and downloading into never-touched memory is slower.
- **cuSPARSE SpMV:** 160 to 370 times the host loop as a repeated call on a resident matrix,
  but one call with its upload loses to the host loop on 14 of 15 instances.
- **What the device does not replace:** the per-row and per-column arrays and the copies of
  the model, 35% (sesia) to 6% of the constructor. By arithmetic on the measured parts the
  constructor would go from 34 s to 22.8 s with the device arrays, and to 12.7 s with huge
  pages as well. No integrated run exists.

### Consequence

- The state build is limited by how the host gets its memory, not by the transpose. The GPU
  halves it; huge pages and the GPU together take it to about a third. Of the whole presolve
  (75 s on the 40 instances) that is about 21 s.
- Integration needs: a second way to construct `state_t` from arrays that are moved in, an
  allocator for the large arrays that asks for huge pages, the device library as an optional
  part of the build, and the campaign with the device path forced. Put to the owner: now, or
  after the cheap column rules.

### Lesson

- I estimated "about 1.5 s instead of 8.3 s" for waipa from the Stage 0 kernel figure plus a
  transfer rate. The measured figure is 3.5 s, and the reason was in neither number: the host
  has to create 7 GB of result arrays before it can receive them. An estimate that adds the
  parts I had numbers for left out the part nobody had measured.

## Entry 33 — 2026-11-27 — The owner: overheads are tracked, not optimised yet

### User message (verbatim)

> Let's do it after the cheap rules. We can do multiple optimization paths for the overheads. But keep track of the overheads during our runs. When they become the bottleneck of beating highs, we can attack them.

### What was done

- Plan amendment A9: the device state build is integrated after the cheap column rules;
  several optimisation paths may exist side by side; overheads are tracked in every run and
  attacked when they stand between the engine and HiGHS on time.
- `analysis/overheads.py` (`10b763d37`): splits a run's seconds into rule families and the
  parts outside them, per size class; shows how much of the time allowance against HiGHS
  (1.2 * t + 0.02 s) the engine and its overheads alone use; lists instances whose overheads
  alone use more than half of the allowance; keeps one line per run in
  `results/stage1/overheads.csv`. With `--bench` the same on the 240 instances, with what the
  presolve call costs inside cuOpt.

### Where the overheads stand today

| Run | Seconds | Rules | Overheads | State build | Output | Hash | Engine / allowance | Overheads / allowance |
|---|---|---|---|---|---|---|---|---|
| Testbed, engine before the fixes | 75.0 | 36% | 64% | 46% | 8% | 8% | 10.2% | 6.5% |
| Testbed, engine with the fixes | 84.0 | 38% | 61% | 43% | 7% | 7% | 11.4% | 7.0% |
| 240 benchmark instances, with the fixes | 18.2 | 39% | 61% | 44% | 6% | 6% | 9.5% | 5.8% |

- The allowance of the first two lines uses the reference measured on the other machine; the
  timing of both sides on this machine is in its last pass.
- No instance is above the allowance, and on none do the overheads alone use more than half
  of it. So the overheads are not the bottleneck against HiGHS today.
- **The largest overhead is not in the engine.** Inside cuOpt the presolve call took 128 s on
  the 240 instances for work the engine does in 18 s: the adapter and its conversions cost
  seven times the engine, a median of 650 ns per nonzero (bohle, square47 and kasavu carry
  the most). The two were not timed in the same run and the node was busy, so the factor is
  rough; the size of it is not noise. PaPILO's call pays conversions too (1,127 s in all).

## Entry 34 — 2026-11-27 — The engine fixes are in the project branch; the first time comparison on one machine; the design of the cheap column rules arrives

No new user message.

### Timing on one machine (viking-prod-215)

- `baseline/stage1_timing_runs.sh viking215_v4pre`: the HiGHS Stage 1 reference and the merged
  engine in three alternating passes, one process at a time. Minimum over the passes, summed
  over the 40 instances: reference 569.0 s, engine 82.5 s, ratio 6.9. Medians per size class:
  4.5 to 9.7 times. The time rule passes on all 40.
- With this reference the engine uses 12.3% of the time allowance, its overheads alone 7.5%.
- Still true from the audit of Entry 17: the reference is the 64-bit-integer build and its time
  contains work outside Stage 1 (column hashes for a switched-off loop, scaling, the rebuild
  of the MIP data), which the audit put at 16 to 29% of it.

### The merge

- `stage1-v4` merged into `feat/gpu-presolve` (`7ef7747ff`, pushed): the two engine fixes, the
  structure check that reads the row-addition record with its reviewer's repairs, the wrong
  engines for the new rules. Done after the GPU state-build agents had finished, since their
  bench tool was built against the engine headers of the synced tree.
- The full node chain for it is running as tag v4. The markers of the two solution steps were
  written by hand with the new identity: the solution files do not depend on the engine and
  take 3.5 hours to remake.

### The design of the cheap column rules (ten agents, 1.8 hours, read-only)

- Kept in `design_summaries/gpu_mip_presolve/stage1c_design/`.
- Nine rules in three families: simple probing on an equation; free singleton column with an
  equation as pivot row, zero-cost singleton, forcing column, free singleton column with an
  inequality as pivot row; fixing by dominance between parallel columns; dual bound
  strengthening; alias of a single-lock column to a binary of its lock row (HiGHS's
  dual-fixing substitution and cuOpt's single-lock dual aggregation in one rule); merge of
  parallel columns. Four new record kinds. No bounds on row duals are kept.
- Left out of the required set, as an optional last milestone: dominated and weakly dominated
  column by the bound on the reduced cost, implied equations, column stuffing, implied-integer
  conversion, the single-equation extension of dual fixing. Together about 1.0007 in HiGHS's
  own one-rule-off runs; they are the rules that would need bounds on row duals. This changes
  amendment A8 and is the owner's decision.
- The validity critic: one blocking defect (the check extensions were specified against the
  structure check as it was before the row-addition work) and three serious ones (no size
  guard on the dominance fixing; contradictory lock preconditions; the oracle's case split is
  not exact when one twin is an integer). The worth-and-cost critic: the predicted gate values
  credit gains nobody measured; the chain claimed for bobin is not supported; the gate should
  measure rows and columns as well; two work bounds are wrong.
- Started: a repair of the design against both critiques and the merged branch, followed by
  two reviewers.

## Entry 35 — 2026-11-27 — The owner: the low-worth cheap rules stay out of the required set

### User message (verbatim)

> Yes, leave them out

(the answer to: may the low-worth cheap rules, about 0.07% together, be left out of the
required set).

### What follows

- Plan amendment A8 changed: the required set of Stage 1c is the nine rules of the design;
  dominated and weakly dominated column by the bound on the reduced cost, implied equations,
  column stuffing, implied-integer conversion and the single-equation extension of dual fixing
  are an optional last milestone, built only when a measurement asks for it. No bounds on row
  duals are kept as state.
- The second decision put to the owner in the same message (merging two binary or integer
  columns into one general integer) was not answered. I go on with what I recommended there:
  the merge of continuous pairs is built, the merge of binary and integer pairs behind a
  switch that is off inside cuOpt.
- Implementation starts from version 2 of the design, when its repair and the two reviews
  are back.

## Entry 36 — 2026-11-27 — The owner: show it on all instances

### User message (verbatim)

> wait a sec. this is on testbed instances. i want to see the impact on all instance set

### What it referred to, and what I did

- My message had two kinds of numbers and named the set of neither: the worth of the rules to
  be left out ("about 1.0007", measured on all 240 benchmark instances) and the time comparison
  (82.5 s against 569 s, measured on the 40 testbed instances). Both readings are covered.
- **The left-out rules on all 240, switched off together** (two HiGHS passes with all cheap
  rules; the one-at-a-time runs of Entry 29 cannot show interaction):
  - All of them off at once: remaining nonzeros 1.0006, rows 1.0015, columns 1.0006 in the
    geometric mean; 19 instances change; 0.014% of the summed nonzeros. HiGHS nests the forcing
    column inside the weakly dominated test, so this pass also loses the forcing column, which
    we build.
  - Largest losses: ns1208400 1.110, neos-3656078-kumeu 1.029, neos-2746589-doon 1.028,
    csched007 1.016, neos-4338804-snowy 1.008; everything else at most 1.004. blp-ic98 and
    blp-ar98 get 3% smaller without the rules.
  - ns1208400 is the implied-integer conversion, and our engine already has the smaller size
    there by another route (73,621 nonzeros). Without it the figure is 1.0002. doon and most of
    kumeu are the forcing column.
  - Column stuffing, implied equations and implied-integer conversion off together: 1.00045,
    and 1.00001 without ns1208400. Dominated and weakly dominated column beyond the forcing
    column: one instance (b1c1s1, 0.2%).
- **Timing on all 240** in three alternating passes was started (`baseline/bench240_timing_runs.sh`).
- The decision of Entry 35 is treated as open again until the owner has seen this.

### Lesson

- A number without its instance set invites the wrong reading, and here it made the owner take
  back an answer. Every figure names its set; a decision is supported on the 240.

### Added to Entry 36: timing on all 240 benchmark instances

- Three alternating passes on viking-prod-215 (a busy node, load 32 to 38), minimum over the
  passes: HiGHS with the Stage 1 rule set 143.0 s, the merged engine 17.1 s, ratio 8.4.
  Medians per size class: below 5e4 nonzeros 4.0 times, 5e4 to 5e5 5.9, 5e5 to 1e7 7.4, above
  9.5. The time rule passes on all 240.
- One instance where the engine is slower than HiGHS: neos-1122047 (163,640 nonzeros), 0.092 s
  against 0.081 s, inside the allowance of 0.118 s. Next lowest: snp-02-004-104 at 1.95 times.
- The engine uses 10.3% of the time allowance on the 240, its overheads alone 6.3%.

## Entry 37 — 2026-11-27 — The merged engine inside cuOpt, on all 240 instances

No new user message.

- cuOpt rebuilt on the node with the merged engine: no compile error; the adapter's 27 unit
  tests pass in three modes.
- The presolve-only report was rerun for the new presolve alone (mode 2) and followed by
  cuOpt's four reductions (mode 3); the report got a switch for its modes, since the
  third-party side does not depend on the engine (`03a799ac6`).
- **Mode 2 against the standalone engine:** the same sizes on 238 of 240 instances;
  momentum1 and var-smallemery-m6j6 differ by 7 and 2 nonzeros. Not looked into; the model
  cuOpt hands over has its rows sorted, which the dump does not guarantee.
- **cuOpt's four reductions on top (mode 3)** change 39 instances. Gains: enlight_hard is
  solved, neos-787933 is halved (125,652 to 61,944 nonzeros), var-smallemery-m6j6 loses 13%,
  the two cryptanalysis instances 2%. They add nonzeros on ns1116954 (27%), ex10 (20%), ex9
  (12%), physiciansched6-2 (3%): indicator strengthening appends rows.
- **Remaining nonzeros, all 240, without ex9, ex10 and enlight_hard** (geometric mean; mean):
  new presolve 0.7885; 0.841. With cuOpt's four 0.7862; 0.841. PaPILO with cuOpt's four
  0.6905; 0.770. HiGHS full 0.6767; 0.766.
- **Adapter overhead:** 94 s for the presolve call inside cuOpt against 18 s for the engine
  alone on the 240 (128 s in the earlier run on a busier node).

## Entry 38 — 2026-11-27 — The owner: build the low-worth rules too

### User message (verbatim)

> Build them too

(the answer to: with the numbers on all 240 instances, leave the low-worth rules out?)

### What follows

- Stage 1c builds all of HiGHS's cheap rules. The five low-worth ones (dominated and weakly
  dominated column by the bound on the reduced cost, implied equations, column stuffing,
  implied-integer conversion, the single-equation extension of dual fixing) are required and
  come after the nine rules of the design, in the order of worth. Plan amendment A8 rewritten.
- They need bounds on the row duals as state. The first design had no such state and put these
  rules into an optional milestone with a few lines each. The design has to be extended for
  them, with the same two kinds of critic, before they are built.
- My recommendation was to leave them out; the owner decided otherwise with the full numbers
  in hand. What they buy on the 240, by HiGHS's own runs: 1.6% on csched007, 0.8% on snowy, up
  to 0.4% on about ten more instances, and ns1208400 where we already have the result.

### Lesson

- The owner's goal is to be better than HiGHS in reductions, and a rule that HiGHS has and we
  do not is a place where we can only lose. "It is worth little" is an argument about order,
  not about scope. I should have proposed "last" instead of "optional".

## Entry 39 — 2026-11-27 — Stage 1 host engine: the merged version passes the full node chain

No new user message.

- `stage1_host_runs.sh v4` on the project branch (`7ef7747ff`, source checksum `6edd0693`), one
  hour: every step passed except `check_large`, which fails on one check, `solution_points` of
  hawaiiv10-130 (no usable point, as in Entry 30).
- Reductions gate 1.0028 over the 40 testbed instances, nothing below 0.95, ns1208400 above
  1.05. Time: 83.0 s; against the reference timed on this machine the engine uses 12.1% of the
  allowance, its overheads alone 7.4%.
- Assertion build: clean on the 31 dev and the 4 status instances.
- Hardened check: the 31 dev instances with no failed check (1,042 checks pass), all 31 with
  usable points now that the published MIPLIB points are there; HiGHS proved the optimum of the
  reduced model on 9 instances and all 9 equal the known optimum. The status instances pass
  (HiGHS proves one of the two infeasible ones infeasible on the reduced model). Of the nine
  large instances eight pass with points.
- Results committed and pushed (`f439c6d18`).

### Where Stage 1 stands

- Host engine with its two fixes: at or above the HiGHS Stage 1 rule set on all 240 benchmark
  instances (less on 18, equal on 220, more on 1) and on the 40 testbed instances; 6.9 times
  faster than that reference on the testbed and 8.4 times on the 240, timed in turn on one
  machine; correctness evidence from the oracle campaigns, the structure check on 240 + 44
  instances, the crush check on 173 + 39 instances, and HiGHS on the reduced model.
- Open in Stage 1: a usable point for hawaiiv10-130 (accept published points at MIPLIB's
  tolerance); the reason of a missing fixed-point flag is not reported (nine instances);
  the three numerical defects of the base engine on models with extreme constants; the device
  state build (after the cheap rules); the adapter's overhead inside cuOpt.

## Entry 40 — 2026-11-27 — Stage 1c: the repaired design is back; reference runs; M0 started

No new user message.

### The repaired design (version 2)

- The repair of the nine-rule design against its two critiques came back
  (`stage1c_design/v2_design.txt`, `v2_answers.txt`, `v2_reviews.txt`, `v2_result.json`). Its
  two verifiers both say implementation can start from it. Sixteen open points remain, all
  minor, each with a model and a fix; they are binding per milestone and are listed in
  `stage1c_design/v2_amendments.txt` together with the owner's scope decision of Entry 38.
- What the repair changed: the check extensions are written against the structure check that
  is in the tree (anchors and check variables in place of linear forms); a size guard on the
  dominance fixing; lock preconditions of the dual rules; the single-lock alias per lock row;
  chains of singleton columns finished inside one call; an objective test in the structure
  check; the gate in three measures; no milestone credited with a gain that no measurement
  supports.
- Decisions of the design's list that I took as recommended, to be reported to the owner:
  integer column pairs are merged only when the merged column gets the kept column's range
  back (`range_kept`); the gate has the same two numbers in all three measures; R3 and R4 stay
  off in primal-only mode; the single-lock alias stays in the required set; work ceilings are
  reports until measured.

### Reference runs (node, HiGHS diag2, presolve only, unscaled)

- New script `gpu_presolve/baseline/stage1c_reference_runs.sh` (commit `ffae18cd8`, pushed).
- **The bobin test** (2 instances): with the Stage 1 switch set and only the zero-cost
  singleton rule added, HiGHS ends at 32,932 rows, 792 columns and 381,000 nonzeros on
  neos-3402294-bobin, its size with all cheap rules on. So the zero-cost singleton rule
  followed by the lock rule reaches bobin, which alone is a third of the summed gap in rows.
- **The rule ladder** (237 of the 240 benchmark instances, geometric mean of remaining
  nonzeros / rows / columns; reference over the engine of today):
  HiGHS Stage 1 set 1.0020 / 1.0013 / 1.0019; with the singleton column rules 0.9832 / 0.9862 /
  0.9855; with simple probing 0.9732 / 0.9759 / 0.9644; with parallel columns 0.9629 / 0.9744 /
  0.9501; with the dual-fixing extensions 0.9569 / 0.9667 / 0.9430; all cheap rules 0.9563 /
  0.9653 / 0.9424 (39 / 19 / 32 instances below 0.95).
- Still running: the ladder with the dominated-column tests on, and all cheap rules on the 40
  testbed instances.

### Started

- Workflow M0 (`wf_097cd54e-581`): engine groundwork for the four new rule families
  (implemented, then attacked), the split of HiGHS's dual-fixing switch into its three parts
  (implemented, then verified), and the comparison script `analysis/bench240_stage1c.py`
  (written, then checked). Base `f439c6d18`.
- The design round for the five low-worth rules with their row-dual bounds is still running.

## Entry 41 — 2026-11-27 — Testbed references; the low-worth design is back; the node ended

No new user message.

### All cheap rules and the rule ladder on the testbed (HiGHS diag2, presolve only, unscaled)

- All cheap rules over the engine of today, 40 testbed instances, geometric mean of remaining
  nonzeros / rows / columns: 0.8008 / 0.8238 / 0.7654. Two instances above 1e7 nonzeros carry
  it: neos-4332810-sesia (HiGHS 407,609 nonzeros, the engine 34,283,539) and dlr2 (10,542,701
  against 72,603,249). Without them 0.9357 / 0.9210 / 0.9184 over 38. The 240 benchmark
  instances had not shown this (0.9563 / 0.9653 / 0.9424 over 237).
- HiGHS's counters credit the fixings on both to the weakly dominated column test (6.5 million
  columns on sesia, 32 million on dlr2), one of the five rules that are low-worth on the 240.
  The counter does not say the rule is needed, so I ran the ladder on the testbed.
- The ladder (40 instances, nonzeros): Stage 1 rule set 1.0028; with the singleton column
  rules 0.8673; with simple probing 0.8540; with parallel columns 0.8523; with the dual-fixing
  extensions 0.8029; all cheap rules 0.8008. sesia: 34,283,539 -> 937,449 with the singleton
  column rules -> 407,609 with the dual-fixing extensions. dlr2: 72,644,224 -> 10,879,922 ->
  10,562,279 (parallel columns) -> 10,542,701. All of it without the dominated-column tests.
  So the two large instances need M1 (singleton columns followed by the lock rule, the bobin
  mechanism), and the build order stands. On the testbed the dual-fixing extensions (M4) are
  worth 6% in nonzeros (supportcase12, neos-4306827-ravan, radiationm18-12-05,
  var-smallemery-m6j6, sesia), ten times their worth on the 240.
- The ladder with the dominated tests on, 240 instances: differs from the one without on 13
  instances, neos-2746589-doon by 2.7%.
- Commits `16218b70c`, `1c5593916` (pushed).

### The node ended

- The reservation of viking-prod-215 hit its time limit during the testbed ladder ("The job
  was terminated"); the sync session is disconnected. I stopped my watcher and did nothing
  else on that side. Missing: the large half of the ladder step with the dominated tests and
  the run with only those tests off; `stage1c_reference_runs.sh` resumes at them. Node work
  that waits: the identity check of M0 on the 240, the build of HiGHS diag3, every engine
  measurement of M1.

### The design of the five low-worth rules

- The design round came back (`stage1c_design/lowworth_design.txt`, `lowworth_critiques.txt`,
  `lowworth_three_designs.txt`, `lowworth_result.json`). Its main finding, checked by both
  critics in the HiGHS source: HiGHS's original row-dual bounds are only 0 or infinite, so
  every implied bound the five rules read is cost over coefficient of one non-integer column
  with at most one lock in a direction. No bounds on row duals need to be stored.
- The validity critic found no wrong reduction in exact arithmetic and eight defects outside
  it, four serious (a tolerance next to a kept bound, marks read before another rule changed
  the row, the integrality test of the conversion without a drift guard, the undo of the nine
  rules reading input types after a conversion). The worth critic found eight (a price cache
  that breaks the fixed-point flag, scans paid on every instance, three worth statements that
  the ladder runs refute, a milestone order that is not the order of worth) and proposes a
  smaller design without a family of its own for implied equations.
- Started a repair round with two verifiers (`wf_6c32f1ad-ddb`), written against version 2 of
  the nine-rule design and with the testbed ladder in hand.

### Lesson

- A HiGHS counter says which test took a fixing, not which rule the reduction needs: bobin,
  sesia and dlr2 all have large "weakly dominated" counts and all are reached by the singleton
  column rules plus the lock rule. Only a run with the rule off decides.
- The testbed and the 240 rank the same rules very differently (dual-fixing extensions 6%
  against 0.6%). Both sets are needed for every decision about order.

## Entry 42 — 2026-11-27 — Node back; the complete rule ladder with the split switches

### User message (verbatim)

> Node is back

### Actions

- The node is now viking-cr-194 (Xeon 8480CL, H200, the slot's GPU pin in place; another host
  than before, so no time of today's earlier runs compares with a time measured there).
  `stage1c_reference_runs.sh` took over its stale lock and resumed at the testbed steps.
- From the M0 workflow, finished and merged while the engine groundwork is still being built:
  the comparison script `analysis/bench240_stage1c.py` (written, then checked by a second
  agent who recomputed every headline number with code of its own; `1d9d19c5b`, `c39b5d228`),
  and the split of HiGHS's dual-fixing switch (`f2ed9ea78`, `edb386b53`): new switch names
  `dual_fixing_single_equation`, `dual_fixing_binary_substitution`,
  `dual_fixing_bound_strengthening`, `implied_row_dual_bounds`, and a counter of free column
  substitutions by pivot kind. Its verifier built it independently and changed nothing.
- Built on the node as `diag3` next to `diag2` (patch checksum `1cd02280...`), then ran the
  reference steps that need it (`b1ca6a10b`, pushed).

### Findings (HiGHS, presolve only, unscaled)

- `diag3` gives the sizes of `diag2` on all 240 benchmark instances with all cheap rules on.
- The complete ladder, all cheap rules over the rule set, nonzeros / rows / columns:

  | HiGHS rule set | 237 of the 240 | 40 testbed |
  |---|---|---|
  | Stage 1 set | 0.9544 / 0.9640 / 0.9406 | 0.7986 / 0.8235 / 0.7634 |
  | + singleton column rules | 0.9726 / 0.9788 / 0.9562 | 0.9233 / 0.9113 / 0.9081 |
  | + simple probing on equations | 0.9826 / 0.9892 / 0.9772 | 0.9377 / 0.9355 / 0.9297 |
  | + parallel columns | 0.9932 / 0.9907 / 0.9919 | 0.9396 / 0.9363 / 0.9340 |
  | + dual bound strengthening | 0.9958 / 0.9937 / 0.9947 | 0.9529 / 0.9566 / 0.9486 |
  | + binary substitution (the nine rules) | 0.9994 / 0.9985 / 0.9993 | 0.9974 / 0.9998 / 0.9972 |
  | + single-equation part | no change | no change |
  | + dominated tests, forcing column | 0.9995 / 0.9991 / 0.9996 | no change |

- So the nine rules, in HiGHS's forms and without the forcing column, are within 0.1% of all
  cheap rules in every measure on the 240 and within 0.3% on the testbed, where only ns1208400
  (which the engine already reaches) and supportcase42 still differ. The gate of 0.99 is
  reachable with the nine rules if our forms reach what HiGHS's reach.
- The dominated-column tests change no testbed instance (the run with only those tests off
  equals all cheap rules on all 40). The single-equation part changes nothing on either set
  when the other dual-fixing parts are on.
- With HiGHS's implied bounds on the row duals frozen, all cheap rules lose 0.9990 / 0.9984 /
  0.9988 on the 240 and 0.9961 / 0.9960 / 0.9966 on the testbed.
- From the switch-split implementer's local runs (Mac build, 240 instances, a preview and not
  the reference): HiGHS's 16,881 free column substitutions with all cheap rules are 11,596
  with an equation pivot and 5,285 with an inequality whose dual sign comes from the pivot
  column itself; none needs a dual bound from another column. Almost all the time HiGHS saves
  with dual fixing off is the marking of the single-equation extension on rd-rplusc-21.

## Entry 43 — 2026-11-27 — The design of the five further rules, version 2

No new user message.

- The repair round came back (`stage1c_design/lowworth_v2_design.txt`, `lowworth_v2_answers.txt`,
  `lowworth_v2_reviews.txt`, `lowworth_v2_result.json`). Both verifiers: implementation can
  start from it once R1 to R9 are built. Fourteen open points, five of them serious, none of
  them touching M0; binding per milestone in `lowworth_v2_amendments.txt`.
- What version 2 is: no stored bounds on row duals and no scanning family. Implied equations
  are a column test inside the dual column family of the nine rules; column stuffing is a step
  of the row visit of the singleton-column family; the single-equation rule and the dominated
  column share a small family driven by columns with at most one lock in a direction; the
  conversion to integer (equation test, weak test, scaling) is driven by a count per row of its
  live continuous columns. Five new record kinds. Milestones L0 to L5 in the order implied
  equations, implied integers with scaling, stuffing, single equation, dominated column.
- It found holes of the same kinds in the nine-rule design and in Stage 1 code: R6 keeps a
  bound and a row with the same tolerance band (a model loses -0.25 to -0.2); the integrality
  test of R2 and R5 for an integer column has no drift guard; the alias forest's edge test in
  today's engine has none either. Added to `v2_amendments.txt` as section F, to be built with
  M1 (shared `integral_for`), M3 and M4.
- An exact-arithmetic test by the repair agent (105,366 tiny models, rationals): rule A applied
  32,067 times, rule D 34,110, rule E 2,335, no changed outcome; with one condition broken it
  finds wrong applications.
- The worth verifier parsed the 240 model files on this machine and counted what the lists and
  tests touch on the input models: both new families together cost about one pass of the model
  on the 151 instances that have a continuous column and nothing on the 89 that have none.

### Lesson

- The same tolerance question came up in three rules by three designers: a bound may count as
  implied "within a tolerance" only when the column then leaves the model. Where bound and row
  both stay, only rounding may be allowed. One shared test, not one per rule.

## Entry 44 — 2026-11-27 — Status question; the groundwork implementer is back, its attacker runs

### User message (verbatim)

> what is the current overall status?

### What I did

- Read the journal of workflow `wf_097cd54e-581`: the engine groundwork (M0) implementer finished
  at about 20:07 CET and the attacker started on it. Nothing else of mine runs; the node
  (viking-cr-194) is idle.
- M0 as reported by its implementer (commit `31a123981` on local branch `stage1c-m0`, base
  `f439c6d18`, 46 files, +2,631 / -405, not pushed, not merged; all runs local on the Mac):
  - the four empty families (simple_probing, singleton_cols, dual_cols, parallel_cols) with
    lists, events, rule indices 6 to 9, options, statistics, phases and fixed-point terms;
    `src/implied.hpp`, `src/saved_row.hpp`; alias origin; `result_t::dual_findings`;
    `state_t::row_singletons`; structure check with mode, unknown-kind error and objective
    test; the oracle's crush test on every result; campaign tool draws families and sub-rule
    switches and presolves a sample twice; wrong engines `ev_unread`, `ev_scale`, rewritten
    `r_prop_inf`. List masks are `uint16_t` with 11 row and 5 column lists in use (section F
    item 1 needs 13 and 8: fits); `rule::count` 10, `sub_rule::count` 8.
  - 414 tests in both builds; 17 local dumps x two modes x families on and off = 68
    comparisons identical to the base (hash, record fields, fixed-point flag, work units);
    campaigns 20.4M models (release) and 10.2M (assertion), each with and without
    `--singletons`, clean; all 49 existing wrong engines caught, four gained the structure
    check.
  - time on the 17 dumps: +1.8% over three alternating passes, +2.7% over fifteen, an earlier
    run +0.1%; above 3% on five instances; a switched-off family's lists are still pushed
    (+1.9% with the four off). To be re-measured on the node.
  - not met: `ev_unread` and `ev_scale` are caught by unit tests only until a rule reads the
    lists (M1); no run on the 240 or the testbed; CMake build, device engine and cuOpt adapter
    not compiled against the changed headers.
  - found at the base commit, not fixed: oracle failure at seed 32077278 (`--singletons`,
    profile huge_parallel, variant 8): reduced optimum 0.99999999889 against 1. Dual fixing
    fixes a column at an explicit bound that bound propagation moved outward by the error
    bound of an activity with a huge term; 1.1e-9 below the optimum, the oracle allows 1e-9.
    Not triaged: oracle tolerance or a weakness of dual fixing next to an outward-moved bound.
- Gave the user the overall status (stages, numbers with their instance sets, what is open
  with them).

### Next

- When the attacker is back: its repairs, then M0 on the node (identity on the 240 and the
  testbed with the families on and off, assertion build, CMake build, three alternating timing
  passes before and after), merge, push; triage of seed 32077278; then the M1 workflow.

## Entry 45 — 2026-11-27 — Notes may be pushed; Grace Hopper at the end; PaPILO for sub-MIPs; an assumptions register; the goal is a stronger formulation

### User message (verbatim)

> yes you can push.
> GH is available but let's leave the benchmarking and perf optimization there to the end.
> Yes submips can use papilo for now.
> Take a note of the assumptions as you go, we will test all of them at the end. remember nnz reduction is not the main goal is to strengthen the formulation. so we will test all of them at the end over the solution quality. strengthening might mean adding those mergings and additional constraints like indicator. but we will test them at the end. first goal is the nnz reduction.

### What this decides

- The three open questions are answered: the design notes may be pushed; a Grace Hopper node
  exists, benchmarking and performance work there wait until the end; sub-MIPs keep PaPILO for
  now.
- New duty: every assumption is noted when it is made, in `ASSUMPTIONS.md` next to this file,
  and all of them are tested at the end over the solution quality.
- The order of goals: first the reduction in nonzeros; the main goal is a stronger formulation,
  which may need reductions that do not shrink the model (merges of integer columns, added
  constraints such as indicator strengthening). Those are tested at the end too. So a choice I
  hold back today for size or caution (integer merges behind `range_kept`, the zero-cost
  singleton and the forcing column off in primal-only mode) is an entry of the register with
  a switch, not a closed decision.
- Plan amendments A9 (overheads, which was missing from DESIGN.md) and A10 (this message).

### What I did

- Before the push I searched the notes for addresses, keys and internal hosts. Found: no
  address, no key; one internal host name in a quoted error message of Entry 10 (the jump
  host), replaced by "<jump host>" in the file. Node names (viking-prod-215, viking-cr-194,
  smc-522ga-0025), the scratch paths and the local paths stay; the scratch paths are in the
  pushed scripts already.
- The notes go to the fork as a new branch `gpu-presolve-notes` that starts with one commit of
  today's files, so the earlier snapshots with the host name are not sent. The old local
  branch `local/gpu-presolve-notes` is kept and stays local. `snapshot.sh` now commits to the
  new branch and pushes it.
- The register is collected by a workflow (readers over the plan, the two rule designs, this
  history, the engine code, the adapter and the measurement scripts; then a merge, two critics
  and a final version).
