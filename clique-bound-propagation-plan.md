# Clique-Aware Activity Tightening — Implementation Plan

## Goal

Tighten the `min_activity` / `max_activity` computation in `calc_activity_kernel` to account for clique correlations between variables, so that `update_bounds_kernel` automatically derives tighter bounds without a separate propagation kernel.

## Key Insight

Standard activity computation assumes all variables contribute independently. But for binary variables in a clique (at most one can be 1), the joint contribution is bounded by the **single best** member, not the **sum** of all members.

```
Constraint:  5*x₁ + 3*x₂ + 4*x₃ + 2*x₄ ≤ 10
Clique:      {x₁, x₂, x₃}

Standard max_activity   = 5 + 3 + 4 + 2 = 14     (assumes all can be 1)
Clique-aware max        = max(5,3,4) + 2 = 7      (at most one of {x₁,x₂,x₃} is 1)
                                   ↓
                     7 fewer units of slack → tighter bounds on x₄
```

The existing `update_bounds_kernel` uses these activities to derive variable bounds. Tighter activities → tighter bounds → faster convergence. **No separate propagation kernel needed.**

## Correction Math

For a non-overlapping clique group G within constraint c, let S = unfixed binary members with coefficients a_j:

```
max_correction = Σ_{j∈S} max(0, a_j) − max(0, max_{j∈S} a_j)    ≥ 0
min_correction = Σ_{j∈S} min(0, a_j) − min(0, min_{j∈S} a_j)    ≤ 0

corrected_max = standard_max − max_correction    (smaller → tighter)
corrected_min = standard_min − min_correction    (larger  → tighter)
```

Multiple non-overlapping groups in the same constraint contribute additive corrections (independent).

### Per-variable adjustment for clique members

When `update_bounds_kernel` processes variable v in constraint c, it computes `activity_without_v = activity - contribution(v)`. For variables **outside** a clique group, the corrected activity works as-is. For variables **inside** a group, removing v changes the group's correction:

```
correction_diff(v) = full_correction − sub_correction_without_v
                   = max(0, a_v) − max_pos + max_pos_without_v

Simplifies to:
    v is the max contributor  →  correction_diff = second_max_pos
    v is NOT the max          →  correction_diff = max(0, a_v)
```

The kernel adds `correction_diff(v)` to the corrected activity before subtracting v's contribution:

```
adjusted_activity = corrected_activity + correction_diff(v)    // undo over-correction for v
activity_without_v = adjusted_activity − contribution(v)       // standard formula, now correct
```

This requires only **two numbers per group**: `max_pos`, `second_max_pos` (and `min_neg`, `second_min_neg` for the min side).

### Non-overlapping partition requirement

Corrections from overlapping cliques double-count shared variables. A variable must be assigned to **at most one group per constraint**. The host precomputation greedily partitions variables into non-overlapping clique groups (largest clique first).

---

## Data Structures

### Static (precomputed once on host, transferred to device)

**Groups are sorted by `constraint_id`** so all groups belonging to the same constraint are contiguous. This enables deterministic per-constraint reduction (no FP atomics).

```cpp
// Per group — sorted by constraint_id
n_groups: i_t
group_constraint_ids : device_uvector<i_t>  [n_groups]        // which constraint
group_member_offsets : device_uvector<i_t>  [n_groups + 1]    // CSR into members
group_member_vars    : device_uvector<i_t>  [total_members]   // variable indices
group_member_coeffs  : device_uvector<f_t>  [total_members]   // coefficients in constraint

// Per constraint: CSR offset into the groups array (since groups are sorted by constraint)
constraint_group_offsets : device_uvector<i_t>  [n_constraints + 1]

// Per reverse-CSR entry: which group this (variable, constraint) belongs to
reverse_group_id : device_uvector<i_t>  [total_reverse_entries]   // -1 if none
```

`reverse_group_id` is indexed exactly like the reverse CSR traversal in `update_bounds_kernel`, so the access is perfectly aligned with the existing memory access pattern — no extra indirection.

### Dynamic (recomputed each iteration)

```cpp
// Per group — fresh each iteration (depends on which members are still unfixed)
group_max_correction : device_uvector<f_t>  [n_groups]   // ≥ 0
group_min_correction : device_uvector<f_t>  [n_groups]   // ≤ 0
group_max_pos        : device_uvector<f_t>  [n_groups]
group_second_max_pos : device_uvector<f_t>  [n_groups]
group_min_neg        : device_uvector<f_t>  [n_groups]
group_second_min_neg : device_uvector<f_t>  [n_groups]
```

These must be recomputed because the corrections depend on which members are still unfixed (bounds change each iteration).

### Host precomputation: building the group table

```
For each constraint c:
    1. Collect binary variables in c
    2. For each variable, look up its cliques (from clique_table_t)
    3. Greedy non-overlapping partition:
       - Sort cliques by size (descending)
       - For each clique: collect unassigned members that appear in c
       - If ≥2 unassigned members → form a group, mark assigned
    4. Record group: (c, member_vars, member_coeffs)

For each (variable, constraint) entry in reverse CSR:
    Set reverse_group_id = group_id if variable is in a group for that constraint, else -1
```

---

## Kernel Design

### Per-iteration flow

```
  1. calc_activity_kernel       <<<n_constraints, 256>>>   (UNCHANGED)
         Writes standard min/max_activity

  2. compute_clique_corrections <<<n_groups, raft::WarpSize>>>  (NEW, one warp per group)
         Reads member bounds, computes per-group stats and per-group corrections
         Writes: group_{max_pos, second_max_pos, min_neg, second_min_neg}
                 group_{max_correction, min_correction}
         No atomics — each thread writes its own group's slot.

  3. apply_corrections_kernel   <<<ceil(n_constraints/256), 256>>>  (NEW)
         One thread per constraint, sums its groups in fixed index order (deterministic)
         min_activity[c] -= Σ group_min_correction[g]  for g in constraint's groups
         max_activity[c] -= Σ group_max_correction[g]

  4. update_bounds_kernel       <<<n_variables, 256>>>     (MODIFIED — small change)
         Same as before, but in update_bounds_per_cnst:
         if reverse_group_id >= 0: add correction_diff to activity

  5. (rest of iteration unchanged: write_updated_bounds, mark dirty constraints)
```

### Kernel 2: `compute_clique_corrections`

One warp per group. Threads stride over members maintaining a thread-local top-2, then a
single butterfly warp reduction merges sums and top-2 pairs together. Handles groups of
size 2 up to 500+ uniformly — small groups just have most lanes idle during the scan but
the reduction is always exactly 5 shuffle rounds.

**Top-2 merge.** Given two already-sorted pairs `(a₁≥a₂)` and `(b₁≥b₂)`, the top-2 of
their union is:

```
new_first  = max(a₁, b₁)
new_second = max( min(a₁, b₁),  max(a₂, b₂) )
```

Three ops, no branches. The runner-up is either the loser of the "firsts" or the winner
of the "seconds", whichever is larger.

```cpp
// One warp per group. Launch <<<n_groups, raft::WarpSize>>>.
// TPB is a template parameter to match the existing kernel style in this file,
// but is statically required to equal raft::WarpSize so the whole block is one warp.
template <typename i_t, typename f_t, i_t TPB>
__global__ void compute_clique_corrections_kernel(
  raft::device_span<const i_t> group_member_offsets,
  raft::device_span<const i_t> group_member_vars,
  raft::device_span<const f_t> group_member_coeffs,
  raft::device_span<const f_t> lb,
  raft::device_span<const f_t> ub,
  raft::device_span<f_t>       group_max_correction,
  raft::device_span<f_t>       group_min_correction,
  raft::device_span<f_t>       group_max_pos,
  raft::device_span<f_t>       group_second_max_pos,
  raft::device_span<f_t>       group_min_neg,
  raft::device_span<f_t>       group_second_min_neg,
  f_t                          int_tol)
{
  static_assert(TPB == raft::WarpSize,
                "compute_clique_corrections_kernel requires exactly one warp per block");

  const i_t gid       = blockIdx.x;
  const i_t mem_begin = group_member_offsets[gid];
  const i_t mem_end   = group_member_offsets[gid + 1];

  // Thread-local accumulators. For each lane, maintain (best, second-best).
  f_t sum_pos = 0, sum_neg = 0;
  f_t max1 = 0, max2 = 0;   // top-2 of  max(0, coeff)   (descending: max1 ≥ max2)
  f_t min1 = 0, min2 = 0;   // top-2 of  min(0, coeff)   (ascending:  min1 ≤ min2)
  i_t n_unfixed = 0;

  // Strided scan over members
  for (i_t m = mem_begin + threadIdx.x; m < mem_end; m += TPB) {
    i_t var = group_member_vars[m];
    f_t a   = group_member_coeffs[m];
    if (ub[var] - lb[var] <= int_tol) continue;   // fixed → skip

    n_unfixed++;
    f_t pos = fmax(a, f_t{0});
    f_t neg = fmin(a, f_t{0});
    sum_pos += pos;
    sum_neg += neg;

    // Insert `pos` into (max1, max2)
    if (pos > max1)      { max2 = max1; max1 = pos; }
    else if (pos > max2) { max2 = pos; }

    // Insert `neg` into (min1, min2)  (most-negative first)
    if (neg < min1)      { min2 = min1; min1 = neg; }
    else if (neg < min2) { min2 = neg; }
  }

  // Butterfly warp reduction. After the loop every lane holds the group's fully-reduced values.
  #pragma unroll
  for (int off = TPB / 2; off > 0; off >>= 1) {
    sum_pos   += __shfl_xor_sync(0xffffffff, sum_pos,   off);
    sum_neg   += __shfl_xor_sync(0xffffffff, sum_neg,   off);
    n_unfixed += __shfl_xor_sync(0xffffffff, n_unfixed, off);

    // Merge top-2 (max side)
    f_t b1 = __shfl_xor_sync(0xffffffff, max1, off);
    f_t b2 = __shfl_xor_sync(0xffffffff, max2, off);
    f_t new_max1 = fmax(max1, b1);
    f_t new_max2 = fmax(fmin(max1, b1), fmax(max2, b2));
    max1 = new_max1;
    max2 = new_max2;

    // Merge top-2 (min side)  — symmetric: flip max↔min and ≥↔≤
    f_t d1 = __shfl_xor_sync(0xffffffff, min1, off);
    f_t d2 = __shfl_xor_sync(0xffffffff, min2, off);
    f_t new_min1 = fmin(min1, d1);
    f_t new_min2 = fmin(fmax(min1, d1), fmin(min2, d2));
    min1 = new_min1;
    min2 = new_min2;
  }

  // Lane 0 writes results (all lanes have the same values; only one needs to write)
  if (threadIdx.x == 0) {
    if (n_unfixed < 2) {
      group_max_pos[gid]        = 0;
      group_second_max_pos[gid] = 0;
      group_min_neg[gid]        = 0;
      group_second_min_neg[gid] = 0;
      group_max_correction[gid] = 0;
      group_min_correction[gid] = 0;
      return;
    }
    group_max_pos[gid]        = max1;
    group_second_max_pos[gid] = max2;
    group_min_neg[gid]        = min1;
    group_second_min_neg[gid] = min2;
    group_max_correction[gid] = sum_pos - max1;   // ≥ 0
    group_min_correction[gid] = sum_neg - min1;   // ≤ 0
  }
}
```

### Kernel 3: `apply_corrections_kernel`

One thread per constraint. Each thread sums all group corrections for its constraint
**in a fixed order** (ascending group index), giving fully deterministic FP arithmetic.
Since groups are sorted by `constraint_id` on the host, the groups for constraint `c`
are contiguous in `[constraint_group_offsets[c], constraint_group_offsets[c+1])`.

Typical constraint has 0–5 groups, so the inner loop is short.

```cpp
template <typename i_t, typename f_t>
__global__ void apply_clique_corrections_to_activity_kernel(
  raft::device_span<const i_t> constraint_group_offsets,
  raft::device_span<const f_t> group_max_correction,
  raft::device_span<const f_t> group_min_correction,
  raft::device_span<f_t>       min_activity,
  raft::device_span<f_t>       max_activity,
  i_t n_constraints)
{
  i_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_constraints) return;

  i_t g_begin = constraint_group_offsets[c];
  i_t g_end   = constraint_group_offsets[c + 1];
  if (g_begin == g_end) return;   // no groups for this constraint

  // Deterministic: sum in fixed group-index order
  f_t max_corr = 0, min_corr = 0;
  for (i_t g = g_begin; g < g_end; ++g) {
    max_corr += group_max_correction[g];
    min_corr += group_min_correction[g];
  }
  max_activity[c] -= max_corr;
  min_activity[c] -= min_corr;
}
```

### Kernel 4: Modified `update_bounds_per_cnst`

**Minimal change** — the only difference from the current code is 4 lines that add `correction_diff` when the variable is in a clique group. The function signature gains the clique view parameter.

```cpp
template <typename i_t, typename f_t>
inline __device__ thrust::pair<f_t, f_t> update_bounds_per_cnst(
  typename problem_t<i_t, f_t>::view_t pb,
  f_t coeff,
  i_t cnst_idx,
  f_t cnst_lb,
  f_t cnst_ub,
  typename bounds_update_data_t<i_t, f_t>::view_t upd,
  thrust::pair<f_t, f_t> bnd,
  thrust::pair<f_t, f_t> old_bnd,
  // NEW parameters:
  i_t group_id,                                    // -1 if not in a clique group
  raft::device_span<const f_t> grp_max_pos,
  raft::device_span<const f_t> grp_second_max_pos,
  raft::device_span<const f_t> grp_min_neg,
  raft::device_span<const f_t> grp_second_min_neg)
{
  auto min_a = upd.min_activity[cnst_idx];   // already corrected by kernel 3
  auto max_a = upd.max_activity[cnst_idx];

  if (check_infeasibility<i_t, f_t>(min_a, max_a, cnst_lb, cnst_ub,
      pb.tolerances.absolute_tolerance, pb.tolerances.relative_tolerance)) {
    return bnd;
  }

  // === NEW: per-variable correction adjustment for clique group members ===
  if (group_id >= 0) {
    f_t pos_c = (coeff > 0) ? coeff : f_t{0};   // max(0, coeff) for this var
    f_t neg_c = (coeff < 0) ? coeff : f_t{0};   // min(0, coeff)

    // Add back correction_diff so activity_without_v is correct
    f_t max_adj = (pos_c >= grp_max_pos[group_id]) ? grp_second_max_pos[group_id] : pos_c;
    f_t min_adj = (neg_c <= grp_min_neg[group_id]) ? grp_second_min_neg[group_id] : neg_c;

    max_a += max_adj;
    min_a += min_adj;
  }
  // === END NEW ===

  // Rest is unchanged
  min_a -= (coeff < 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
  max_a -= (coeff > 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
  auto delta_min_act  = cnst_ub - min_a;
  auto delta_max_act  = cnst_lb - max_a;
  thrust::get<0>(bnd) = update_lb(thrust::get<0>(bnd), coeff, delta_min_act, delta_max_act);
  thrust::get<1>(bnd) = update_ub(thrust::get<1>(bnd), coeff, delta_min_act, delta_max_act);
  return bnd;
}
```

### Corresponding change in `update_bounds` device function

The loop that iterates over constraints needs to read `reverse_group_id` and pass it through:

```cpp
template <typename i_t, typename f_t, i_t BDIM>
__device__ void update_bounds(typename problem_t<i_t, f_t>::view_t pb,
                              i_t var_idx, i_t var_offset, i_t var_degree, bool is_int,
                              typename bounds_update_data_t<i_t, f_t>::view_t upd,
                              thrust::pair<f_t, f_t> old_bnd,
                              // NEW:
                              raft::device_span<const i_t> reverse_group_id,
                              raft::device_span<const f_t> grp_max_pos,
                              raft::device_span<const f_t> grp_second_max_pos,
                              raft::device_span<const f_t> grp_min_neg,
                              raft::device_span<const f_t> grp_second_min_neg)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  if (!upd.changed_variables[var_idx]) return;

  auto bnd = old_bnd;
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    if (upd.changed_constraints[cnst_idx] != 1) continue;
    auto a       = pb.reverse_coefficients[var_offset + i];
    auto cnst_ub = pb.constraint_upper_bounds[cnst_idx];
    auto cnst_lb = pb.constraint_lower_bounds[cnst_idx];

    // NEW: one extra global read, perfectly coalesced with the reverse CSR traversal
    i_t grp = reverse_group_id[var_offset + i];

    bnd = update_bounds_per_cnst(pb, a, cnst_idx, cnst_lb, cnst_ub, upd, bnd, old_bnd,
                                 grp, grp_max_pos, grp_second_max_pos,
                                 grp_min_neg, grp_second_min_neg);
  }

  // BlockReduce, write_updated_bounds, mark dirty constraints — all unchanged
  thrust::get<0>(bnd) = BlockReduce(temp_storage).Reduce(thrust::get<0>(bnd), cuda::maximum());
  __syncthreads();
  thrust::get<1>(bnd) = BlockReduce(temp_storage).Reduce(thrust::get<1>(bnd), cuda::minimum());
  __shared__ bool changed;
  if (threadIdx.x == 0) { changed = write_updated_bounds(pb, var_idx, is_int, upd, bnd, old_bnd); }
  __syncthreads();
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    if (changed) { atomicExch(&upd.next_changed_constraints[pb.reverse_constraints[var_offset + i]], 1); }
  }
}
```

---

## Performance Analysis

### Cost of the new kernels

| Kernel | Grid size | Work per thread | Notes |
|---|---|---|---|
| `compute_clique_corrections` | `n_groups` blocks × `raft::WarpSize` threads | Stride over members, warp-shuffle reductions | One warp per group. Handles groups of size 2–500+ uniformly. |
| `apply_corrections_to_activity` | `⌈n_constraints / 256⌉` blocks | 2 reads + 2 writes | Memory-bound, negligible. |

### Cost added to `update_bounds_kernel`

Per thread iteration over the reverse CSR:

| Operation | Cost |
|---|---|
| Read `reverse_group_id[var_offset + i]` | **+1 global read** (coalesced, same stride as `reverse_constraints`) |
| Branch on `grp >= 0` | Almost always false (most variables are not in clique groups) → branch predictor handles well |
| Compute `max_adj`, `min_adj` | 2 comparisons + 2 reads from group arrays (only when `grp >= 0`) |
| Add to `max_a`, `min_a` | 2 FP adds |

**Net overhead: ~1 extra global memory read per constraint iteration.** For variables not in any clique group (the vast majority), the branch is not taken and the cost is just the read.

### Why this is efficient

- **Zero change to `calc_activity_kernel`** — the most expensive kernel is untouched.
- **No extra kernel launches in the hot path** beyond two tiny kernels between activity and bounds.
- **`reverse_group_id` is coalesced** — it's parallel to the existing reverse CSR arrays, so it's fetched in the same cache line.
- **`compute_clique_corrections` is embarrassingly parallel** — one thread per group, no synchronization except atomicAdd to per-constraint arrays (low contention since few groups share a constraint).

---

## Integration into `bounds_presolve.cu`

### New member in `bound_presolve_t`

```cpp
struct clique_activity_data_t {
  // Static (built once) — groups sorted by constraint_id
  rmm::device_uvector<i_t> group_constraint_ids;      // [n_groups]
  rmm::device_uvector<i_t> group_member_offsets;      // [n_groups + 1]
  rmm::device_uvector<i_t> group_member_vars;         // [total_members]
  rmm::device_uvector<f_t> group_member_coeffs;       // [total_members]
  rmm::device_uvector<i_t> constraint_group_offsets;  // [n_constraints + 1]
  rmm::device_uvector<i_t> reverse_group_id;          // [total_reverse_entries], -1 if none
  i_t n_groups{0};

  // Dynamic (recomputed each iteration) — all per-group, no per-constraint atomics needed
  rmm::device_uvector<f_t> group_max_correction;  // [n_groups]  ≥ 0
  rmm::device_uvector<f_t> group_min_correction;  // [n_groups]  ≤ 0
  rmm::device_uvector<f_t> group_max_pos;
  rmm::device_uvector<f_t> group_second_max_pos;
  rmm::device_uvector<f_t> group_min_neg;
  rmm::device_uvector<f_t> group_second_min_neg;

  bool empty() const { return n_groups == 0; }
};
```

### Modified `calculate_bounds_update`

```cpp
template <typename i_t, typename f_t>
bool bound_presolve_t<i_t, f_t>::calculate_bounds_update(problem_t<i_t, f_t>& pb)
{
  constexpr i_t zero       = 0;
  constexpr auto n_threads = 256;
  auto stream = pb.handle_ptr->get_stream();

  upd.bounds_changed.set_value_async(zero, stream);

  // NEW: compute per-group corrections, then fold them into activities deterministically.
  if (!clique_data_.empty()) {
    // One warp per group — writes per-group stats and per-group corrections (no atomics)
    compute_clique_corrections_kernel<i_t, f_t, raft::WarpSize>
      <<<clique_data_.n_groups, raft::WarpSize, 0, stream>>>(
        /* group table spans + upd.lb, upd.ub, group stat/correction arrays */);

    // One thread per constraint — deterministic in-order sum of its groups
    i_t act_blocks = (pb.n_constraints + n_threads - 1) / n_threads;
    apply_clique_corrections_to_activity_kernel<i_t, f_t>
      <<<act_blocks, n_threads, 0, stream>>>(
        /* constraint_group_offsets, group_max/min_correction, min/max_activity */);
  }

  // Existing bound update (modified kernel with group_id parameter)
  update_bounds_kernel<i_t, f_t, n_threads>
    <<<pb.n_variables, n_threads, 0, stream>>>(
        pb.view(), upd.view(), clique_data_.view());

  RAFT_CHECK_CUDA(stream);
  i_t h_bounds_changed = upd.bounds_changed.value(stream);
  return h_bounds_changed != zero;
}
```

---

## Edge Cases

- **No clique groups:** `clique_data_.empty()` → skip correction kernels; `reverse_group_id` is all -1 → update kernel behaves identically to original.
- **Variable fixed mid-iteration:** Correction kernel reads current `lb`/`ub`, so it naturally adapts — fixed variables are skipped in the stats, correction shrinks.
- **Multiple groups for same constraint:** Corrections are additive via atomicAdd. Non-overlapping partition guarantees no double-counting.
- **Infeasibility from tighter activities:** The existing `check_infeasibility` in `update_bounds_per_cnst` uses the corrected activity, which is valid — the clique constraint makes the feasible region smaller.

## Files to Create / Modify

| File | Action |
|---|---|
| `presolve/clique_activity_corrections.cuh` | **Create** — `clique_activity_data_t`, `compute_clique_corrections_kernel`, `apply_corrections_kernel`, host precomputation (`build_groups_from_clique_table`) |
| `presolve/bounds_update_helpers.cuh` | **Modify** — add `group_id` + group stat params to `update_bounds_per_cnst` and `update_bounds`; add adjustment logic (4 lines) |
| `presolve/bounds_presolve.cuh` | **Modify** — add `clique_activity_data_t clique_data_` member |
| `presolve/bounds_presolve.cu` | **Modify** — build groups in `bound_update_loop`, launch correction kernels in `calculate_bounds_update` |
