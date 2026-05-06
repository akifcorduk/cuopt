# Cut Pool — Iterative Improvements Roadmap

Baseline: `eb5d5865fa8114fc24f5e50f8d23427f4bea62ce` ("fix thread count").
Scope: `cpp/src/cuts/cuts.cpp` (4813 lines) and `cpp/src/cuts/cuts.hpp` (887
lines).

This document lists improvements to the cut scoring / selection / filtering /
pool-management code, ordered by **expected impact-to-risk ratio**, with
priorities explicitly cross-referenced to published evidence (the
Wesselmann–Suhl paper, SCIP `sepa_cutsel_hybrid`, and HiGHS `mip_pool_*`).
Each item is meant to be applied as a *single* commit so individual effects
can be measured against the MIPLIB2017 baseline.

> **How to use this document.** Pick the next unstarted item. Implement
> only that item. Run the benchmark protocol described in
> [§ Benchmark protocol](#benchmark-protocol). Compare to the previous
> tier's results. Mark the item DONE with the measured deltas in the
> [§ Results log](#results-log) at the bottom.

---

## Evidence base & references

The priorities below are not invented — they map directly to designs that
are documented and benchmarked in the literature. Two primary sources, plus
two production solvers, agree on the same picture.

| Source | Used for | Notes |
|---|---|---|
| Wesselmann & Suhl, *Implementing cutting plane management and selection techniques* (Mops paper, ~2010; PDF in `cut_scoring.pdf`) | Cut quality measures (§2.2), cut pool & selection algorithm (§3), 143-instance experimental ranking (§4 Table 4) | Empirical: their Mops `distance` config (-27% SGM time, +5% gap closed vs no selection) |
| Achterberg, *SCIP: solving constraint integer programs* (2009) and SCIP source `src/scip/sepa_cutsel_hybrid.c` | Score-tiered parallelism (`GOODSCORE`, `goodmaxparall`), composite weighted score | Default weights: `efficacyfac=1.0`, `intsupportfac=0.1`, `objparalfac=0.1` |
| HiGHS `HighsCutPool.cpp` and `mip_pool_age_limit` / `mip_pool_soft_limit` settings | Pool aging + soft-cap design | Defaults: `pool_age_limit=5`, `pool_soft_limit=30000` |

**Key Wesselmann–Suhl Table 4 findings** (143-instance MIPLIB+CORAL+MILP set,
SGM percentage change vs no cut selection):

| Quality measure | Δ time | Δ gap closed | Δ cuts added |
|---|---|---|---|
| **distance** (`v / ‖a‖`, our baseline `cut_distance`) | **−27%** | **+5%** | −89% |
| distance with bounds (Eq 10) | −28% | +3% | −88% |
| rotated distance (Eq 9) | −23% | +4% | −89% |
| adjusted distance (Eq 15) | −25% | +5% | −88% |
| violation (Eq 4) | −16% | +4% | −87% |
| **support (1 − support fraction)** | **−8%** | **−4%** | −75% (worst) |
| obj. parallelism alone | −1% | +19% gap | −93% |
| int. support alone | −19% | 0% | −74% |
| Achterberg-style composite (`distance + 0.1 obj_par + 0.1 int_supp`) | better than plain distance: **50% fastest, 89% solved** (vs 46% / 85%) | — | — |

This rules out a few designs we had on the list:

- **Density penalty (1 − support)** is the worst single measure in the
  paper. Drop it from P1/P2; it goes to P3 only as a curiosity. (User
  also already observed it "eliminates too many cuts" when reverted.)
- **Plain `violation`** as the score is also disproven — efficacy
  (`violation / ‖a‖`) wins. Baseline already uses efficacy. ✅

It elevates a few designs:

- **Aging + composite score + score-tiered parallelism** are the three
  ingredients common to Mops, SCIP, and HiGHS. They're the high-impact
  P1 items below.

---

## Baseline `eb5d586` — what is and isn't there

Read this before planning a change so you don't accidentally re-implement
something that exists, or assume something that doesn't.

**What baseline HAS:**

- `cut_distance(row, x, …)`: returns `violation / ‖a‖₂` — pure efficacy
  (paper §2.2 *distance*, Eq 6). Validated as a top-tier scorer.
- `cut_orthogonality(i, j)`: returns `1 − |cosᵢⱼ|`. Pairwise.
- `score_cuts`:
  - calls `check_for_duplicate_cuts()` (Tomlin-Welch O(nnz)) on every call;
  - computes `cut_distance` per row;
  - hard floor: `min_cut_distance_ = 1e-4` (constant);
  - sorts by score; greedy-selects with single threshold
    `min_orthogonality = settings_.cut_min_orthogonality` (= **0.5** in
    `simplex_solver_settings.hpp`);
  - aggregates parallelism with `min` over all already-selected cuts;
  - hard cap: `max_cuts = 2000`.
- `check_for_duplicate_cuts()`: full Tomlin-Welch (paper §3 Eq 26 setup,
  Tomlin-Welch 1986 algorithm) — globally recomputes every time.
- `cut_age_[i]++` after every `score_cuts` round.
- `cut_density(row)`: defined but currently unused in scoring.

**What baseline LACKS (every item below corresponds to a P-level task):**

- `drop_cuts()` is `// TODO: Implement this` — empty stub. **Aging
  increments the counter but nothing ever evicts**. The cut pool grows
  monotonically until end of solve. → P1-1.
- No at-insert duplicate / parallelism / NaN / degenerate-norm checks
  in `add_cut`. → P2-1, P2-2.
- No at-insert cousin filter for Bron-Kerbosch clique-cut families
  (cuts whose support sets overlap heavily but aren't equal-support).
  Cousins pay full insert + dedup + score cost and are only filtered
  at selection by the orthogonality scan. → P2-4.
- No `cut_inv_norm_`, `cut_max_abs_coef_` precomputation. (Currently
  `cut_orthogonality` recomputes from `cut_norms_` populated lazily.)
- No score multipliers (no integer-support boost, no objective-
  parallelism term). Score is pure efficacy. → P1-2.
- No adaptive minimum-quality threshold. Just the hard `1e-4` floor.
  → P1-4.
- No score-tiered parallelism (single 0.5 threshold for all cuts).
  → P1-3.
- No per-cut-type per-event statistics. → P0-1.
- No diagnostic-logging gate. → P0-2.
- No clique work-budget accounting; no clique-cut diagnostics. → P0-3.

The current `cut_min_orthogonality = 0.5` is **5× more permissive than
the paper's `p_max = 0.1`** and SCIP's default `1 − minortho = 0.1`.
Tightening it without P1-3 (score-tiered) will starve cut selection
in dense correlated families (the paper observes this explicitly for
MIR cuts on aggregated rows). The two changes belong in the same
commit.

---

## Tier P0 — Diagnostic infrastructure (no algorithmic change)

These don't change behavior. They make every subsequent change measurable
and debuggable. Apply all three before any P1 work.

### P0-1. Per-cut-type per-event aggregate counters

**What.** Add a `cut_pool_stats_t` (`[cut_type][event]` int64 matrix) to
`cut_pool_t` plus `reset_stats()`, `log_add_stats_summary(label)`, and
`log_score_stats_summary(label)`. Bump counters at every accept and reject
site in `add_cut` and `score_cuts`. Wire `reset_stats()` + `log_add_…`
around each generation phase in `generate_cuts`, and at the start / end of
`score_cuts`.

**Why.** No algorithmic change can be evaluated without per-pass per-type
visibility into what fraction of generated cuts is being rejected and why.
Counters cost a few CPU cycles per event; printing happens once per phase.
Wesselmann & Suhl Table 5 (paper §4) is exactly this kind of analytical
output and is what enabled all of their parameter tuning.

**Touches.** `cut_pool_t` (new struct + 5 methods + 1 member),
`add_cut` (~5 inc sites once P2-1/P2-2 land — currently 1 reject site),
`score_cuts` (~5 inc sites + 1 success site),
`generate_cuts` (~5 reset/log pairs).

**Knobs.** None.

**Validation.** Logs should show `acc=`, `dup=`, `aged=`, `parallel=`,
etc. counts per cut type per phase. No measurable runtime regression.

---

### P0-2. Gated per-cut reject diagnostic logging

**What.** Add a `bool log_rejects_{false}` member + `set_log_rejects(bool)`
setter to `cut_pool_t`. Wrap *every* per-cut "Reject cut row=… reason=…"
log line in `if (log_rejects_)`. Default off.

**Why.** Per-cut reject lines are useful for one-instance debugging but
fire millions of times per solve. With the volume gated off by default we
lose nothing in production but can flip the flag for forensic runs.

**Touches.** `cut_pool_t` (1 member + 1 setter), every "Reject cut" log
site in `add_cut` / `score_cuts`.

**Knobs.** `log_rejects_` (default `false`).

**Validation.** With the flag off, `out.txt` contains zero "Reject cut"
lines. With flag on, behavior matches today's verbose output.

> **Cost note.** On NFS-backed log files, the `fflush(...)` inside
> `logger_t::printf` makes per-line printing 100×–1000× more expensive
> on Grace than on x86. Even with this gate, leave the flag off in
> benchmark runs unless you specifically need the diagnostics.

---

### P0-3. Clique-cut correctness and time-budget improvements (no size cap)

**What.** Port the clique-cut improvements from the
`clique_heuristic_time` branch, **except** the hard `max_clique_size`
cap and the Jaccard near-duplicate filter. Specifically:

1. **Complement-pair infeasibility detection in `build_clique_cut`.**
   Replace the baseline early-exit-on-first-conflict with a two-pass
   detector that counts complement pairs (variable + complement both
   present in the clique vertex set) before declaring infeasibility, so
   diagnostic output reports the full conflict count rather than
   stopping at the first one. Logically equivalent to baseline for
   accept/reject; better diagnostics.

2. **Work-estimation budget in `extend_clique_vertices`.**
   Add `addtl_cliques_scan_cost * initial_clique_size` and
   `addtl_cliques_scan_cost` to the running `work_estimate`, accounting
   for the additional-clique scan that the extension performs. Also
   pass `max_work_estimate` so the helper can short-circuit when the
   per-pass time budget for clique extension is exhausted. This is
   what the branch name refers to — clique extension was previously
   uncounted in the budget, so on instances with many clique-table
   entries it dominated cut generation time without being attributed
   to "clique cuts".

3. **`DEBUG_CLIQUE_CUTS` compile-time hook.**
   Add `#define DEBUG_CLIQUE_CUTS 0` (default off) and gate detailed
   per-clique inspection prints (`stderr`-direct so they don't go
   through `logger_t`). Useful for forensic runs without polluting
   production logs.

4. **`has_pair` field in the accepted-cut log line.**
   Extend `build_clique_cut accepted` log to record whether the cut
   included any complement-pair literals. Aids attribution of which
   clique-cut variant (pure / mixed) is producing useful cuts.

**Why this is P0.** Items 1, 3, 4 are pure diagnostics (no algorithmic
semantics change). Item 2 changes only *when* the clique-extension loop
exits — it never invents or drops a cut on its own; it only enforces the
existing per-pass time budget that clique extension was previously
ignoring. None of them changes the cut selection model; they make later
P1/P2 work measurable on instances where clique cuts dominate. Bron-
Kerbosch generates many maximal cliques per call, so clique cuts
particularly benefit from the pool aging in P1-1.

**Explicitly NOT included.**

- `max_clique_size` parameter (hard cap on clique vertex count). The
  branch experiments with this at 32 / 64 but the cap conflates two
  problems — runtime cost (already addressed by item 2) and density
  cost (better addressed by P1-2 score multipliers — though note:
  density penalty proper is disproven by the paper, see Evidence base
  above) — and tends to drop genuinely strong large cliques on dense
  problems. Skip it.
- Jaccard near-duplicate clique-cut filter in `add_cut`. Defer to
  **P2-4**. The existing orthogonality filter in `score_cuts` only
  removes near-duplicates at selection time — *after* every cousin has
  paid the full insert + dedup + score cost. The at-insert hash dedup
  in P2-2 catches *exact-support* duplicates (e.g. permuted vertex
  order) but not cousins (different vertex sets with high overlap,
  which is what Bron-Kerbosch produces). P2-4 is the at-insert cousin
  filter that picks up this deferral.

**Touches.** `cuts.cpp`:
- `build_clique_cut` (~25 lines: pre-pass complement counting, log
  format, debug stderr block).
- `extend_clique_vertices` signature + 2 budget-accounting calls + 0
  size-cap branches (we are explicitly omitting them; the existing
  recursion can stay structurally as-is once budget accounting is in).

**Knobs.**

- `DEBUG_CLIQUE_CUTS` (compile-time, default 0).

**Validation.**

- On instances with no clique cuts (most of MIPLIB), no observable
  change in time, optima, or root gap closed.
- On instances with clique cuts (`set-packing`, `cod105`,
  `mas74`, `air03`, `air04`, `iis-…`, etc.):
  - Wall-clock time spent in `Clique cuts:` phase becomes bounded
    by the per-pass time budget (item 2).
  - `build_clique_cut infeasible: N complement-pairs` log line shows
    expected non-zero N on infeasible cliques (item 1).

---

## Tier P1 — High-impact, well-understood algorithmic changes

These are paper- and SCIP/HiGHS-validated. Each can be added independently;
when adding more than one, prefer the order listed.

### P1-1. Implement `drop_cuts` + adaptive aging + soft pool cap **(critical)**

**What.** Replace `void cut_pool_t::drop_cuts() { /* TODO */ }` with a
real implementation, and gate it on:

- A new member `pool_age_limit_{5}` (HiGHS `mip_pool_age_limit`
  default; paper used 3 for Mops).
- A new member `pool_soft_limit_{30000}` (HiGHS `mip_pool_soft_limit`
  default).
- An "effective" age limit computed at the top of `score_cuts` /
  `drop_cuts`: start at `pool_age_limit_`, walk a per-age histogram
  downward while `available_cuts > pool_soft_limit_`; cuts whose age
  has reached the effective limit are evicted.
- Compaction: after dropping, rewrite `cut_storage_`, `rhs_storage_`,
  `cut_age_`, `cut_type_`, and (when present) `cut_inv_norm_`,
  `support_hash_buckets_` consistently.

Selected cuts have their age reset to 0 (touched), or the lifecycle is
managed entirely by per-round increment on unselected, whichever is
simpler — match HiGHS / paper convention.

**Why this is the most important P1 item.** Baseline `drop_cuts` is
**unimplemented**. The pool grows monotonically across rounds; on
instances with many cut passes the selection cost (which is O(pool_size)
for the first sort and O(pool_size · selected) for the orthogonality
scan) blows up unboundedly. Wesselmann–Suhl §3 explicitly: *"Cutting
planes are aged out of the cut pool, meaning that cuts whose quality
values were not large enough to be added to the LP relaxation in
`max_age` rounds are deleted."* HiGHS uses the same.

**Touches.** `cut_pool_t` (2 members + setters + `drop_cuts`
implementation), `score_cuts` (call `drop_cuts` after selection or at
top, depending on lifecycle convention), `age_cuts` (already exists,
keep).

**Knobs.** `pool_age_limit_` (default 5; HiGHS uses 5; Mops paper used
3), `pool_soft_limit_` (default 30000; HiGHS default).

**Validation.** Per-pass log line `cut_pool_size=` should plateau around
`pool_soft_limit_` on dense MIPLIB instances instead of growing
monotonically. SGM time should improve on those instances. Root gap
closed should not regress (selected cuts are the same; we're just
evicting the unselected losers).

---

### P1-2. Composite score: efficacy + integer-support + objective parallelism

**What.** Replace `score = cut_distance` (pure efficacy) with the
SCIP/Mops additive composite:

```
score = w_eff × efficacy
      + w_int × integer_support_fraction
      + w_obj × objective_parallelism
```

where:

- `efficacy = violation / ‖a‖₂` — already what baseline computes.
- `integer_support_fraction = |supp(a) ∩ NI| / |supp(a)|` — fraction
  of cut nonzeros on integer-constrained variables (paper Eq 25).
- `objective_parallelism = |aᵀc| / (‖a‖ ‖c‖)` — cosine similarity to
  the LP objective (paper Eq 19).

Defaults match SCIP `sepa_cutsel_hybrid` and the Mops Achterberg-style
composite that the paper validates:

- `w_eff = 1.0`, `w_int = 0.1`, `w_obj = 0.1`.

**Why.** Paper §4 / Figure 4: Achterberg-style composite (the same
weights) is fastest on **50%** of instances and solves **89%** to
optimality, vs **46%** / **85%** for plain distance — a measurable
upgrade for a one-line additive change. Each individual term is
disproven on its own (obj_parallelism alone: −1% time, support alone:
−8% time) but the composite beats either piece in isolation. SCIP and
Mops independently converged on `0.1 / 0.1` as the right scale.

**Why no density penalty.** Paper Table 4: support / density as a
quality measure is the **worst** single measure (-8% time, -4% gap
closed). User also empirically observed that adding it "eliminates too
many cuts." Density (cut nnz) belongs in pool aging or LP-side cost
control, not in the score numerator.

**Touches.** `score_cuts` (replace `cut_distance` call with new
composite). Need `integer_var_mask_` plumbed in (or pass `var_types`
into the scorer). `objective_parallelism` requires the LP cost vector;
precompute `‖c‖` once per pass.

**Knobs.** `efficacy_weight_` (default 1.0), `integer_support_weight_`
(default 0.1), `obj_parallelism_weight_` (default 0.1; set to 0.0 to
disable, matching SCIP `objparalfac=0.0` for some MIP categories).

**Validation.** On integer-heavy instances (`mas74`, `air04`, IPs with
many binaries), per-cut-type `selected/accepted` ratio for `IB`/`CLIQ`
should rise. SGM gap closed at root improves on integer-heavy
instances. SGM time roughly flat or slightly better.

> **Convention.** Match SCIP's additive form, not multiplicative. The
> earlier draft of this document proposed multiplicative; revert to
> additive. Multiplicative penalties make the threshold gate (P1-4)
> harder to tune because units mix.

---

### P1-3. Score-tiered parallelism + tighten base threshold

**What.** Two coupled changes that must land together:

1. **Tighten** `settings_.cut_min_orthogonality` from **0.5** to **0.9**
   (= max parallelism 0.1, matching paper `p_max = 0.1` and SCIP
   `1 − minortho = 0.1`).
2. **Add a relaxed tier** for high-quality cuts so the tightened
   threshold doesn't starve correlated dense families:

   - `max_parallelism_{0.1}` — strict, applies to ordinary cuts.
   - `good_max_parallelism_{0.5}` — relaxed, applies when a candidate
     is "good" (= score ≥ `good_cut_factor_ × top_round_score`).
   - `good_cut_factor_{0.9}` — paper `skip_factor = 0.9`, SCIP
     `GOODSCORE = 0.9`.

   The orthogonality scan **runs for both tiers**; only the threshold
   differs. Do *not* introduce a binary "bypass" — high-quality cuts
   must still satisfy the relaxed constraint.

**Why both at once.** Paper §3 explicitly motivates this pairing: tight
`p_max` aggressively dedups, and `skip_factor / p_max_ub` protects
genuinely strong cuts from being incidentally caught. Tightening the
threshold alone (without the relaxed tier) is what the user already
ran into and reported as starving selection on MIR families.
Introducing the relaxed tier alone (without tightening the base) is
no-op since the base is already at 0.5.

**Touches.** `cut_pool_t` (3 members + setters), `score_cuts` (compute
`good_score` once per round; per-candidate `effective_max_parallelism`),
`simplex_solver_settings.hpp` (change default).

**Knobs.** `max_parallelism_` (default 0.1; SCIP `1 − minortho`),
`good_max_parallelism_` (default 0.5; SCIP `goodmaxparall`),
`good_cut_factor_` (default 0.9; SCIP `GOODSCORE`).

**Validation.** Per-pass `parallel=` reject count (P0-1 stat) should
rise overall (tighter base) but the increase should concentrate on
*low-score* candidates (P0-1 stratification). Root gap closed should
not regress on dense-MIR instances (`mas74`, `air04`, `seymour`); #
optima found should improve (paper Table 4 + Mops experiments).

> **Note on aggregation.** Baseline uses `min` over all selected;
> a short-circuit-on-first-violation form is functionally equivalent
> for accept/reject and is faster on long candidate lists. Either is
> fine.

---

### P1-4. Adaptive minimum-quality threshold gate

**What.** Replace the hard `min_cut_distance_ = 1e-4` floor with the
paper's adaptive `min_qual` mechanism (§3):

- `min_qual_ub_{0.01}` — upper bound on the minimum quality (paper
  default).
- `min_score_factor_{0.5}` — initial fraction of best-pool-quality
  used to set `min_qual` on first call (paper: 50%).
- Hysteresis: a `fail_count` counter is incremented when no cut in
  the pool meets `min_qual` in a round; when `fail_count == 2`,
  reduce `min_qual` toward zero (e.g. halve it) so weaker cuts can
  qualify.
- On every call: `min_qual = min(min_qual_ub_, min_score_factor_ ×
  max_score_in_pool)`.

**Why.** Baseline runs the full O(pool²) orthogonality scan on every
violated cut, even ones with negligible efficacy. Paper §3 + §4: this
adaptive gate, combined with `total_factor = round_factor = ∞` (no
absolute cut count cap) gave Mops the strongest results. The hard
`1e-4` floor is too permissive on instances where the best cut has
efficacy ~0.1 (we're then admitting cuts with 1% the quality of the
best for selection consideration).

**Touches.** `cut_pool_t` (3 members + setters + `fail_count_`),
`score_cuts` (compute `min_qual` after sort and before
orthogonality loop; update `fail_count_` after selection).

**Knobs.** `min_qual_ub_` (default 0.01; paper), `min_score_factor_`
(default 0.5; paper), `fail_threshold_` (default 2; paper).

**Dependency.** Best implemented *after* P1-2 so the gate isn't biased
by a single large-magnitude raw-violation cut. With efficacy + composite,
the score distribution is bounded and the adaptive gate behaves.

**Validation.** Per-pass `score_threshold` reject count (P0-1 stat)
should be non-trivial on long candidate lists. SGM time should improve
on instances with thousands of generated cuts per pass. `fail_count_`
should rarely reach `fail_threshold_` on healthy instances.

---

## Tier P2 — Robustness and pool-cost refinements

### P2-1. Numerical safety guards in `add_cut`

**What.** Before storing a cut in `add_cut`:

1. Reject cuts with empty support (squeezed size 0) — currently logged
   but not gated by stats.
2. Reject cuts with non-finite RHS (`!std::isfinite(cut.rhs)`).
3. Reject cuts with any non-finite coefficient.
4. Reject cuts with degenerate norm (`Σ aᵢ² == 0`).

Stats counters from P0-1 attribute each rejection to the originating
cut type.

**Why.** Baseline trusts the generator. In practice, MIR / aggregation
steps occasionally produce NaN or zero-norm rows that propagate to LP
factorization failure or score divisions-by-zero. Cheap to guard at
the entry point. None of HiGHS / SCIP / Mops admit non-finite cuts.

**Touches.** `add_cut` (insert ~30 lines of validation early-return
paths, each with its own `stats_.inc(...)` site).

**Knobs.** None.

**Validation.** No NaN observed in `cut_distances_` during long MIPLIB
runs. Stats counters `nf_rhs=`, `nf_coef=`, `degen=` (P0-1) are
mostly zero; non-zero counts identify generator bugs.

---

### P2-2. Hash-based at-insert duplicate detection

**What.** Compute a stable hash of each cut's column-support set in
`add_cut`. Maintain `support_hash_buckets_` —
`unordered_map<uint64_t, vector<row_idx>>`. On insert, look up the
bucket and check existing rows with matching support for sign-corrected
parallel cuts (`signed_cosine ≥ 1 − 1e−6`); if the new cut is duplicate
of or weaker than an existing one, reject it. Also precompute and
store `cut_inv_norm_[i]` and `cut_max_abs_coef_[i]` for fast reuse in
`cut_orthogonality` and the dup-check.

**Why.** Baseline `check_for_duplicate_cuts()` runs the global
Tomlin-Welch O(nnz) algorithm at the start of every `score_cuts` call.
With a pool of ~30k cuts that dominates separation. Hash buckets give
O(1) per insert with guaranteed-correct same-support comparison;
Tomlin-Welch can then become periodic (P2-3).

**Touches.** `cut_pool_t` (new map + helpers + `cut_inv_norm_`,
`cut_max_abs_coef_` members), `add_cut` (insert dup-check before
append + bucket update), `cut_orthogonality` (use precomputed
`cut_inv_norm_`), `drop_cuts` / compaction (keep buckets aligned with
row indices).

**Knobs.** None (hash function and tolerance hard-coded; `1e−6`
matches paper §3 epsilon for Tomlin-Welch).

**Validation.** `dup=` count from P0-1 stats reflects hash hits.
`check_for_duplicate_cuts()` time per call drops; with periodic
invocation (P2-3) total dedup time should drop ≥10× on dense pools.

---

### P2-3. Tomlin-Welch dedup made periodic

**What.** Move `check_for_duplicate_cuts()` (the global O(nnz)
Tomlin-Welch dedup) to run once every `K` calls of `score_cuts`
instead of every call. P2-2 hash buckets handle the common case; the
Tomlin-Welch pass is the safety net for cuts with re-ordered or
re-scaled coefficients (which the support-hash misses).

**Why.** Tomlin-Welch on a dense pool is the dominant cost in
`score_cuts`. Paper §3 doesn't discuss frequency (Mops calls it once
per round), but with at-insert hash dedup the global pass catches at
most a few additional duplicates per round; running it less often is
safe.

**Knobs.** `tomlin_welch_period_{4}` (suggested starting value).

**Touches.** `score_cuts` (counter + conditional invocation).

**Dependency.** Apply only after P2-2 **and P2-4**: without at-insert
hash dedup *and* at-insert cousin filtering, deferring Tomlin-Welch
lets exact duplicates and BK cousin families accumulate across rounds.

---

### P2-4. At-insert Jaccard cousin filter for clique-cut family

**What.** During the clique-cut phase only (gated by
`cut_type == CLIQUE`), `add_cut` consults a per-round support-min-hash
sketch and, for any new cut whose support has Jaccard overlap ≥
`clique_cousin_jaccard_tau_` with an already-pooled clique cut from
this round, **keeps only the higher-scoring representative** (or, when
the round-side score isn't yet computed, the higher-violation one
using `xstar` already in scope inside `generate_clique_cuts`).

Mechanism, two stages:

1. **Cheap signature.** Compute a `k`-min-hash sketch
   (`clique_cousin_minhash_k_`, default 8, 64-bit hashes) over the
   cut's support indices. Store one signature per pool row, scoped to
   `cut_type_t::CLIQUE`. Min-hash agreement count ≈ `k · Jaccard`, so
   sketch comparison estimates Jaccard in `O(k)`.
2. **Decision.** Hash the new cut's signature to a coarse bucket
   (e.g. min-element prefix); compare against existing rows in the
   bucket; on collisions above `tau`, replace the loser in-place
   (or skip the insert, depending on whether the existing entry
   should be evicted). Cousin-replacement preserves the bucket
   invariant: at any time, no two clique cuts in the pool have
   pairwise Jaccard ≥ tau.

**Why.** Bron-Kerbosch on a fractional-binary subgraph emits *all*
maximal cliques of the conflict graph induced by xstar. Two maximal
cliques whose vertex sets overlap in `k − 1` of `k` vertices produce
clique cuts whose coefficient vectors agree on `k − 1` coordinates and
differ in one — a "cousin." On dense instances
(`cod105`, `air03`, `set-packing`) BK can emit hundreds of cousin
cliques per round.

The existing roadmap defends against cousins only at *selection*:

- **P1-3** (tighten `cut_min_orthogonality` 0.5 → 0.9 + relaxed tier)
  rejects cousins at cosine > 0.1 against the first-selected cousin.
  This is the SCIP/Mops design and is the principal selection-stage
  answer.
- **P3-4** (cluster-aware top-1 per support cluster) is parked as the
  alternative selection-stage filter.

Both run *after* every cousin has paid the full insert + dedup +
score cost: one `add_cut` call, one row of Tomlin-Welch
(`check_for_duplicate_cuts` is `O(nnz · m)`), one efficacy
computation, one integer-support-fraction scan, one objective-
parallelism dot product. On a 200-cousin family this is a 200×
multiplier on per-cut cost that selection-stage filtering cannot
recover.

P0-3 explicitly deferred Jaccard filtering to P2 with the rationale
that "the at-insert hash dedup in P2-2 will catch exact duplicates
much more cheaply than Jaccard scanning would" — but P2-2 as scoped
catches *equal-support* duplicates (`{1,2,3}` vs `{2,3,1}`), not
cousins (`{1,2,3,4}` vs `{1,2,3,5}`, different bucket entirely). P2-4
is the item that picks up that deferral and makes it concrete.

**Why scoped to `CLIQUE`.** The cousin-family pathology is specific
to BK enumeration. Gomory / MIR / strong-CG generators produce cuts
with much more diverse support; running min-hash for them spends
cycles without payoff and risks rejecting genuinely different cuts
that happen to share a few high-coefficient variables. The scope-by-
type gate is one line in `add_cut`.

**Why insert-side and not generator-side.** A generator-side cap
("emit at most `K_max_per_pivot` cliques per pivot vertex") is a
strictly stronger but riskier change. None of Mops, SCIP, HiGHS take
it — presumably because BK enumeration cost is already bounded by
`max_calls = 100000` in `generate_clique_cuts` and a per-pivot cap
can drop genuinely strong large cliques. P2-4 is the conservative
move: keep BK's coverage of the maximal-clique space, drop the
redundant insertions only.

**Touches.**

- `cut_pool_t`: new `clique_support_minhash_` (sketches, one row per
  `CLIQUE` cut) + `clique_cousin_buckets_` (bucket → row indices)
  members. Three-line entry/exit in `add_cut` gated on
  `cut_type == CLIQUE`. Four-line cleanup in `drop_cuts` /
  `check_for_duplicate_cuts` compaction so the buckets stay aligned
  with row indices after every shrink.
- `add_cut` (~30 lines): sketch + bucket query + score compare +
  in-place replace.
- `score_cuts` (no change). The selection-time orthogonality scan
  still runs and can absorb residual cousin pairs that slipped
  through the at-insert filter (e.g. cousins inserted in different
  rounds).

**Knobs.**

- `clique_cousin_jaccard_tau_` (default `0.85`; Mops paper §3 uses
  `0.9` as the near-duplicate cutoff for maximal cliques — a touch
  lower lets us absorb extension-gain noise from
  `extend_clique_vertices`).
- `clique_cousin_minhash_k_` (default `8`; 64-bit hashes).
- `clique_cousin_filter_enable_` (default `true`).

**Dependencies.**

- After **P1-1** so `drop_cuts` compaction can keep `clique_support_
  minhash_` and `clique_cousin_buckets_` aligned with row indices —
  identical pattern to the P2-2 bucket-alignment work.
- After **P1-2** so the keep-the-better decision uses the SCIP/Mops
  composite and not just raw violation. Pre-P1-2 the decision uses
  violation as a fallback proxy.
- Compatible with **P2-2**: P2-4's min-hash buckets and P2-2's
  support-equality hash buckets are independent maps and can coexist;
  P2-2 fires first (catches exact same-support cousins cheaply),
  P2-4 fires second (catches partial-overlap cousins).
- Lands **before P2-3**: cousin elimination at insert is what makes
  deferring Tomlin-Welch safe even on dense BK-output rounds.

**Validation.**

- `CLIQUE` row in P0-1 stats: a new
  `CUT_EVENT_ADD_COUSIN_DROPPED` (or equivalent) counter is non-zero
  on `cod105`, `air03`, `set-packing`, `iis-…`. Total BK output count
  unchanged (the generator is untouched). `cut_pool_size` after the
  clique phase drops on those instances.
- Wall time inside `score_cuts` on the same instances drops, driven
  by lower `cut_storage_.m` going into the score loop. Root gap
  closed should not regress: the keep-the-better rule preserves the
  best representative of each cousin family.
- Per-round `clique_cousin_buckets_` size stays bounded by `O(num_
  binaries)` even on adversarial dense instances. (Without bucketing,
  the sketch comparison would degrade to `O(m_clique²)` per insert.)

**What it does NOT do.**

- Not a generator-side emission cap. BK still enumerates all maximal
  cliques up to `max_calls = 100000`.
- Not a replacement for P1-3. P1-3 still runs at selection and
  catches residual cousins (different rounds, sketch collisions
  below `tau`, etc.).
- Not a replacement for P3-4. P3-4 stays parked; if P1-3 + P2-4
  together still leave cousin starvation in dense MIR families
  (which BK does not produce — MIR cousins arise from aggregation
  rounds, not from clique enumeration), P3-4 becomes relevant for
  MIR / Gomory at selection.

> **Cost note.** A cousin-bucket lookup is `O(k + bucket_size)`;
> bucket_size is bounded by the number of distinct max-clique
> "families" in the round, typically `O(√m_clique)` on instances
> where the cousin pathology shows up. Wall-time-wise this is one to
> two orders of magnitude cheaper than the per-cut work each cousin
> would pay through dedup + score, even on the worst case.

---

## Tier P3 — Experimental / low confidence

These either have failure modes already hit-tested in this codebase, or
are paper-discussed but expensive to implement, or have small expected
impact relative to P1/P2.

### P3-1. Adjusted distance / active-support score (paper Eq 15)

**What.** Replace efficacy in the scorer with the paper's *adjusted
distance*: `score = violation / (‖a_active‖ + 1)` where
`a_active[j] = a[j]` if `x*[j] ≠ 0` else `0`.

**Paper evidence.** Table 4: adj_distance (-25% time, +5% gap) is
roughly equivalent to plain distance (-27% time, +5% gap). Not a
clear winner.

**Why P3.** Baseline distance already works well; the +1 in the
denominator can cause one rare narrow-support cut to dominate
selection if the selection threshold (P1-4) drifts up. Treat as a
tunable variant for problems where the paper's distance underperforms.

**If you try it:** pair with a `best_observed_score_` decay (e.g.
`× 0.95` per round before the new max) instead of pure monotone max,
to prevent threshold drift.

---

### P3-2. Distance with bounds (paper Eq 10, Cook et al.)

**What.** Replace efficacy with `db = min{‖x − x*‖ : aᵀx ≤ β,
ℓ ≤ x ≤ u}` — Euclidean distance to nearest cut-feasible bound-feasible
point. Solved by Voglis-Lagaris active-set QP per cut.

**Paper evidence.** Table 4 + Figure 1: best single measure (-28%
time, +3% gap, 87% solved). But cost per cut is several QP iterations
of size O(nnz).

**Why P3.** Implementation cost is high (per-cut QP), and the gain
over plain distance (which is our baseline scorer) is only 1
percentage point of SGM time. Defer until P1/P2 are exhausted.

---

### P3-3. Rotated distance / rotated distance with bounds (paper Eq 9 / 11)

**What.** When the LP has equality constraints `Dx = d`, project the
cut onto the affine subspace `{x : Dx = d}` before measuring distance:
`rd = violation / ‖a − Dᵀ(DDᵀ)⁻¹Da‖`.

**Paper evidence.** Comparable to plain distance overall, slightly
better on instances dominated by equality constraints.

**Why P3.** Requires LU of `DDᵀ` plus per-cut forward/back substitution.
Worth it only on instances we know to be equality-heavy. Probably better
to add automatic detection (e.g. fall back to plain distance when
no equalities are present) than to make it the universal default.

---

### P3-4. Cluster-aware selection (top-1 per support cluster)

**What.** Instead of pure score-greedy with parallelism filter, group
candidates by support hash, select the top-scoring representative per
cluster, then run a relaxed parallelism filter on the survivors.

**Why P3.** Sidesteps the cousin-filter problem entirely but conflates
cuts with same support but different coefficients. Changes the
selection model (no longer pure score-greedy). Larger code change.
Only attempt if P1-3 + P2-2 still leave starvation in dense cases.

---

### P3-5. Density-based score penalty

**What.** Multiply score by `1 / (1 + λ × log₂(nnz + 1))` to
deprioritize dense cuts.

**Paper evidence.** Table 4: support-as-quality is **the worst** single
measure (-8% time, -4% gap closed). User empirically observed
"eliminates too many cuts."

**Why P3 (and probably never).** Only consider if profiling shows that
LP factorization cost from dense cuts is a dominant bottleneck *and*
we have the necessary data (which P0-1 stats can answer). Even then,
prefer to handle density via the pool soft cap (P1-1) or LP-side
cleanup, not via a score penalty.

---

## Benchmark protocol

For every change, measure:

1. **Number of optima found** out of the MIPLIB2017 benchmark set.
2. **SGM time** over the commonly-solved subset (shift = 1s, paper §4
   uses shift = 10s; either is fine but keep constant).
3. **SGM gap** over the not-solved subset (shift = 5%, paper §4
   convention).
4. **Mean root-relaxation gap closed** vs. known optimum.
5. **Per-pass diagnostics** (P0-1 stats): cuts generated / accepted /
   selected per type, reject reasons.

### Configuration

- 3 seeds per (instance, config); aggregate via SGM.
- Time limit: same as the team's MIPLIB2017 standard (typically 1h).
- Hardware: Grace and x86 in parallel; same set of instances.
- Single-threaded per run (or whatever the project standard is — keep
  it constant across changes).

### Required logs per run

- `CutHParams: ...` line at start (added when P0-1 lands).
- One `CutPass: pass=N obj_before=… obj_after=… gap_closed_pct=… ...`
  line per cut pass (requires plumbing optimum reference; defer).
- One `CutYield: type=… generated=… accepted=… selected=…
  ever_in_lp=… still_in_lp=… aged_out=… net_gap_closed_pct=…` line
  per cut type at solve exit.
- One `RunSummary: instance=… status=… primal=… dual=…
  gap_to_opt_pct=… root_gap_closed_pct=… time=… nodes=…` at solve
  exit.

These are *not* implemented in the baseline; they should be added
incrementally as P0 work alongside P0-1 / P0-2.

### Statistical significance

On ~240 instances × 3 seeds, differences of < 2–3 instances solved or
< 5% SGM time are within noise. Use Wilcoxon signed-rank for paired
config comparisons, not eyeballed averages.

---

## Recommended commit order

```
# Diagnostics first, so everything below can be measured.
P0-1   stats counters + per-pass / per-type log  [pre-req for measuring everything]
P0-2   gated per-cut reject log                  [enables forensic runs cheaply]
P0-3   clique cuts: infeas detect + work budget  [no size cap; no jaccard filter]

# The big one: baseline drop_cuts is currently a TODO stub.
P1-1   drop_cuts impl + soft cap + adaptive aging  [bounds pool size + selection cost]

# Paper-validated quality multiplier.
P1-2   composite score (efficacy + 0.1·int_supp + 0.1·obj_par)  [+4 pp solved per paper §4]

# Tighten parallelism, with the relaxed tier needed to avoid starvation.
P1-3   score-tiered parallelism + cut_min_orthogonality 0.5→0.9  [paper §3 / SCIP defaults]

# Adaptive lower bound on quality.
P1-4   adaptive min_qual gate (paper §3 hysteresis)  [drops weak candidates upstream]

# Robustness & dedup speed.
P2-1   numerical safety guards in add_cut       [defensive; do anytime after P0-1]
P2-2   hash-based at-insert duplicate detection [O(1) dedup; precomputes cut_inv_norm_]
P2-4   at-insert Jaccard cousin filter (CLIQUE) [absorbs Bron-Kerbosch cousin floods]
P2-3   Tomlin-Welch dedup made periodic         [requires P2-2 + P2-4 first]

# Experimental — only if P0–P2 don't reach the goal.
P3-*   distance variants, cluster selection, density
```

Eleven committed items (P0-1 through P2-4); P3 is open-ended.

Run the benchmark protocol after every P-tier so the deltas are
attributable. P1-1 alone is expected to be the largest single
improvement — do it first within the P1 block and measure separately
before stacking P1-2/3/4.

---

## Results log

| Date | Item | Δ optima | Δ SGM time | Δ root gap closed | Notes |
|---|---|---|---|---|---|
| _baseline_ | `eb5d586` | — | — | — | reference; `drop_cuts` is TODO, pool grows unbounded, single 0.5 parallelism threshold |
|     |     |     |     |     |     |
