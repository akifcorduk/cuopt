# fixed_costs

CUDA micro-benchmark of the fixed costs that decide when a GPU phase pays on a small model
(Stage 0 of the GPU presolve, DESIGN.md sections 2.1, 2.2, 12, 13). One executable, one JSON file,
one short table on stdout.

## Build and run (dev node)

From the worktree root on the Mac:

```
cuopt exec 'cmake -S gpu_presolve/microbench -B /home/scratch.acoerduek_sw/gpu_presolve_deps/build/microbench -G Ninja'
cuopt exec 'cmake --build /home/scratch.acoerduek_sw/gpu_presolve_deps/build/microbench -j 8 --target fixed_costs'
cuopt exec '/home/scratch.acoerduek_sw/gpu_presolve_deps/build/microbench/fixed_costs gpu_presolve/results/stage0/fixed_costs.json'
```

- Release build, C++17, `CMAKE_CUDA_ARCHITECTURES=native` unless set. Dependencies come from the
  conda environment: `find_package(rmm)` brings CCCL and the CUDA runtime.
- `fixed_costs <output.json> [--quick] [--only group,group,...]`
- Full mode takes about two and a half minutes on the H200 node. Its peak is about 12 GB of device
  memory (8 GB of that is the rmm pool) and 4 GB of pinned plus 2 GB of pageable host memory.
  `--quick` takes about 3 seconds (30 repetitions, fewer and smaller sizes) and is for functional
  checks only.
- Progress lines go to stderr, the table to stdout.
- Groups for `--only`: `launch readback library handover_copy allocation dirty_list gather
  unified_memory`.

## Method

- Wall-clock time from `std::chrono::steady_clock` around the stated interval. Every interval that
  uses the device ends after `cudaStreamSynchronize`, except `empty_kernel_launch_no_sync`, which
  times the enqueue only.
- Each measurement runs warm-up iterations (at least 2, at most 20 or 0.5 s), then 200 repetitions.
  The count is lower only when 200 repetitions would take more than 8 s; then it is 8 s divided by
  the iteration time, and at least 10. On the H200 node this affects the 2 GB pageable copies, the
  1 GB and 4 GB pinned allocations and the 1e7 dirty-list upload. `reps` is recorded per result.
- Median, 10th and 90th percentile (linear interpolation between order statistics), in
  microseconds.
- One user stream, default scheduling flag, one host thread. Inputs come from a fixed hash, so
  the data, the kept counts and the frontier are the same on every run. Results of the measured
  operations are checked outside the timed interval (values read back, sortedness, record sets,
  gathered rows); a failed check aborts.

## What is measured

| Group | Rows | Sizes (full mode) |
|---|---|---|
| `launch` | idle-stream synchronise; empty kernel launch + synchronise; launch only; 10 launches + one synchronise | - |
| `readback` | kernel writes 8 bytes, copy to host, synchronise; pageable and pinned destination | - |
| `library` | `thrust::reduce` (double), `reduce_by_key` (int keys, one per 16 elements), `sort` (u64), `sort_by_key` (u64, u64), `copy_if` (about 10% kept), `exclusive_scan` (i64), each with Thrust temporaries from an rmm pool (`rmm_pool`) and from `cudaMalloc` (`cuda_malloc`); `cub::DeviceRadixSort::SortPairs` with preallocated temporary storage | 1e3 to 1e7 elements |
| `handover_copy` | host to device and device to host, pinned and pageable host memory | 4 KB to 2 GB |
| `allocation` | `cudaHostAlloc` / `cudaFreeHost`, `cudaMalloc` / `cudaFree`, rmm pool allocate / deallocate | 1 MB to 4 GB |
| `dirty_list` | `host_to_device`: host fills k records into a pinned buffer, one copy, one scatter kernel into a 1e7-element array, synchronise. `device_to_host`: compaction, count read back, records copied to pinned memory, in three variants (below) | k = 1e2 to 1e7 |
| `gather` | frontier row lengths prefix-summed, one gather kernel into one contiguous buffer, count read back, buffer copied to pinned memory; CSR with 1e8 nonzeros, rows of 4 to 64 entries; one kernel in three launch shapes (below); `frontier_upload` separately | 1e3 to 1e7 gathered entries |
| `unified_memory` | `cudaMallocManaged` 64 MB: allocate, host first touch, kernel touch, host touch again, free; reference rows on pinned and device memory | 64 MB |

Sizes are binary (1 MB = 2^20 bytes); `gb_per_s` uses 2^30 bytes.

Points where the task could be read in more than one way, and what the tool does:

- **Thrust temporaries.** Thrust allocates temporary device storage in every call measured here.
  With the default policy that is a `cudaMalloc` and `cudaFree` per call, which costs more than the
  algorithm on small inputs. Both variants are reported; `rmm_pool` is the one that matches a
  presolve that owns an rmm pool.
- **Dirty-list compaction.** `flag_scan_atomic` is one kernel over all changed flags of the state
  array with an integer atomic slot counter; the record order is not deterministic.
  `flag_scan_ordered` is `cub::DeviceSelect::Flagged` over the same flags plus one gather kernel;
  the order is deterministic. `frontier_emit_atomic` is one kernel over a device list of the k
  candidate indices with an atomic counter, the case where the phase already knows its candidates.
  The first two pay a scan of the whole state array whatever k is.
- **Record layout.** One 24-byte record (int index, int pad, 16-byte value), so each direction is
  one copy.
- **Gather kernel.** One kernel after the prefix sum, in three launch shapes. `thread_per_row`
  copies each row serially in one thread, 256 rows per block. `thread_per_row_block_32` does the
  same with 32 rows per block. `warp_per_row` spreads each row over the 32 threads of one warp,
  8 rows per block. In the one-thread-per-row shapes a warp works on 32 different rows at once,
  and the time depends on the block size, not only on the row lengths: in a check on the dev node
  the `scan_gather_count` phase at 1e4 and 1e5 entries took about 175 microseconds with 256 rows
  per block, 120 with 128, 80 with 32, against 50 to 57 for `warp_per_row`. The source arrays are `__restrict__`;
  without it the stores into the output buffer force a reload per entry and the serial variant
  was about twice as slow at 1e3 entries.

## JSON

```
{
  "tool": "fixed_costs", "mode": "full" | "quick",
  "system": { gpu_name, gpu_memory_total_mib, gpu_compute_capability, driver_version,
              cuda_runtime_version, cuda_driver_api_version, cuda_compile_version, cccl_version,
              rmm_version, cpu_model, cpu_threads,
              other_gpu_processes_observed, other_cpu_load_observed,
              gpu_memory_used_mib_before, gpu_utilization_percent_before,
              gpu_compute_apps_before, gpu_compute_apps_after, gpu_compute_apps_most_seen,
              cpu_loadavg_before, cpu_loadavg_after },
  "harness": { repetitions, min_repetitions, budget_s, max_warmup_iterations, warmup_budget_s },
  "notes": [ ... ],
  "results": [
    { "group", "name", "variant", "size", "elements"?, "bytes"?, "reps",
      "median_us", "p10_us", "p90_us", "elements_per_s"?, "gb_per_s"?,
      extra counts ("kept", "temp_storage_bytes", "frontier_rows", "matrix_rows",
      "matrix_nonzeros")?,
      "phases"?: [ { "name", "reps", "median_us", "p10_us", "p90_us" } ] }
  ]
}
```

- `elements_per_s` and `gb_per_s` are computed from the median.
- `phases` splits an interval into consecutive parts; the parts add up to the total in every
  repetition, the medians need not.
- `other_gpu_processes_observed` is true when `nvidia-smi` showed memory in use, utilisation, or a
  compute process on this GPU before the benchmark created its context, or more than one compute
  process after any group or at the end. The process list is read once after each group
  (`gpu_compute_apps_most_seen` is the longest list seen), so a process that came and went inside
  one group is missed. `other_cpu_load_observed` is true when the 1-minute load average was 1 or
  more at the start. The GPU snapshot is taken before the first CUDA call, so the benchmark itself
  is not in it.

## Limits

- The unified-memory rows describe an x86 node with a discrete GPU, where every hand-over is a
  page-fault migration. They say nothing about Grace Hopper.
- The dirty-list and gather kernels are plain versions written for this benchmark, not the
  presolve kernels. They give the fixed part (launches, read-backs, copies) and a first slope.
- `host_fill` in the dirty-list upload is one host thread reading a 160 MB state array at random
  positions; it depends on the CPU and on what else runs on the node.
- Mirror refresh is covered by the `handover_copy` rows (a mirror refresh is a device-to-host copy
  of the matrix arrays); there is no separate row for it.
- The GPU is identified through `nvidia-smi` by the UUID of CUDA device 0. The tool aborts if
  `nvidia-smi` is missing or does not list that UUID (for example on a MIG device).
