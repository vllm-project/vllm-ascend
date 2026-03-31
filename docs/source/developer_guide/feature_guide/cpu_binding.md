# CPU Binding

## Overview

CPU binding pins vLLM Ascend worker processes and key threads to specific CPU cores to reduce CPU–NPU cross‑NUMA traffic and stabilize latency under multi‑process workloads. It is designed for ARM servers running Ascend NPUs and is automatically executed during worker initialization when enabled.

## Background

On multi‑socket ARM systems, the OS scheduler may place vLLM threads on CPUs far from the local NPU, causing NUMA cross‑traffic and jitter. CPU binding enforces a deterministic CPU placement strategy and optionally binds NPU IRQs to the same CPU pool. This is distinct from other performance features (e.g., graph mode or dynamic batch) because it is purely a host‑side affinity policy and does not change model execution logic.

## Design & How it works

### Key concepts

- **Allowed CPU list**: The cpuset from /proc/self/status (Cpus_allowed_list). All allocations are constrained to this list.
- **Running NPU list**: Logical NPU IDs extracted from npu‑smi process listing, optionally filtered by ASCEND_RT_VISIBLE_DEVICES.
- **CPU pool per NPU**: The CPU list assigned to each logical NPU ID based on the binding mode.
- **Binding modes & Device behavior**:

  | Device type | Default mode | Description |
  | ----------- | ------------ | ------------ |
  | A3 (No Affinity) | `global_slice` | Splits the allowed CPU list evenly based on the **total number of global logical NPUs**, ensuring each NPU is assigned a contiguous segment of CPU cores. This prevents CPU core overlap across multiple process groups. |
  | A2 / 310P / Others | `topo_affinity` | Allocates CPUs based on NPU topology affinity (`npu‑smi info -t topo`). If multiple NPUs are assigned to a single NUMA node (which may cause bandwidth contention), the CPU allocation extends to adjacent NUMA nodes. |

    - **Default**: enabled (enable_cpu_binding = true).
    - **Fallback**: If NPU topo affinity is unavailable, global_slice is used.
    - **Failure handling**: Any exception in binding is logged as a warning and **binding is skipped for that rank**.

### Execution flow (simplified)

1. **Feature entry**: worker initialization calls `bind_cpus(local_rank)` when `enable_cpu_binding` is true.
2. **CPU architecture gate**: If the CPU is not ARM, binding is skipped with a log.
3. **Collect device info**:
   - Map logical NPU IDs from `npu‑smi info -m`.
   - Detect running NPU IDs from npu‑smi info process table.
   - Read cpuset from /proc/self/status.
   - Read topo affinity from `npu‑smi info -t topo`.
4. **Build CPU pools**:
   - Use **global_slice** for A3 devices; **topo_affinity** for A2 and 310P.
   - If topo affinity is missing, fall back to global_slice.
   - Ensure each NPU has at least 5 CPUs.
5. **Allocate per‑role CPUs**:
   - Reserve the first two CPUs for IRQ binding.
   - `main`: pool[2:-2]
   - `acl`: pool[-2]
   - `release`: pool[-1]
6. **Bind threads**:
   - Main process is pinned to `main` CPUs.
   - ACL threads (named with acl_thread) are pinned to `acl` CPU.
   - Release threads (named with release_thread) are pinned to `release` CPU.
7. **Bind NPU IRQs (optional)**:
   - If /proc/irq is writable, bind SQ/CQ IRQs to the first two CPUs in the pool.
   - irqbalance may be stopped to prevent overrides.
8. **Memory binding (optional)**:
   - If migratepages is available, memory for ACL threads is migrated to the NPU’s NUMA node.

## Allocation plan examples

The allocation plan is derived directly from the CPU pool per NPU and then split into roles:

- IRQ CPUs: pool[0], pool[1]
- `main`: pool[2:-2]
- `acl`: pool[-2]
- `release`: pool[-1]

Below are concrete examples that reflect the actual code paths.

### Example 1: A3 inference server with 640 CPUs and 16 NPUs

- allowed_cpus = [0..639] (640 CPUs)
- NUMA nodes = 0..7 (8 NUMA nodes, symmetric layout)
- total_npus = 16
- running_npu_list = [0..15]
- base = 640 // 16 = 40, extra = 0
- Each NPU gets a 40‑CPU pool.

|NPU ID|Assigned CPU Cores (global_slice)|Role Division (IRQ/Main/ACL/Release)|
|---|---|---|
|0|0-39|`IRQ`: 0-1, `Main`: 2-37, `ACL`: 38, `Release`: 39|
|1|40-79|`IRQ`: 40-41, `Main`: 42-77, `ACL`: 78, `Release`: 79|
|...|...|...|
|15|600-639|`IRQ`: 600-601, `Main`: 602-637, `ACL`: 638, `Release`: 639|

This layout remains deterministic even when multiple processes share the same cpuset, because slicing is based on the global logical NPU ID.

### Example 2: A3 global_slice, even split

**Inputs**:

- allowed_cpus = [0..23] (24 CPUs)
- NUMA nodes = 0..1 (2 NUMA nodes, symmetric layout; NUMA0 = 0..11, NUMA1 = 12..23)
- total_npus = 4 (from npu-smi info -m)
- running_npu_list = [0, 1, 2, 3]

**Global slice**:

- base = 24 // 4 = 6, extra = 0
- Each NPU gets a 6‑CPU pool.

|NPU ID|Assigned CPU Cores (global_slice)|Role Division (IRQ/Main/ACL/Release)|
|---|---|---|
|0|0-5|`IRQ`: 0-1, `Main`: 2-3, `ACL`: 4, `Release`: 5|
|1|6-11|`IRQ`: 6-7, `Main`: 8-9, `ACL`: 10, `Release`: 11|
|2|12-17|`IRQ`: 12-13, `Main`: 14-15, `ACL`: 16, `Release`: 17|
|3|18-23|`IRQ`: 18-19, `Main`: 20-21, `ACL`: 22, `Release`: 23|

### Example 3: A3 global_slice, remainder distribution

**Inputs**:

- allowed_cpus = [0..16] (17 CPUs)
- NUMA nodes = 0..1 (2 NUMA nodes, symmetric layout; NUMA0 = 0..7, NUMA1 = 8..16)
- total_npus = 3
- running_npu_list = [0, 1, 2]

**Global slice**:

- base = 17 // 3 = 5, extra = 2
- NPU0 pool size = 6 (base+1)
- NPU1 pool size = 6 (base+1)
- NPU2 pool size = 5 (base)

|NPU ID|Assigned CPU Cores (global_slice)|Role Division (IRQ/Main/ACL/Release)|
|---|---|---|
|0|0-5|`IRQ`: 0-1, `Main`: 2-3, `ACL`: 4, `Release`: 5|
|1|6-11|`IRQ`: 6-7, `Main`: 8-9, `ACL`: 10, `Release`: 11|
|2|12-16|`IRQ`: 12-13, `Main`: 14, `ACL`: 15, `Release`: 16|

Note: When a pool size is exactly 5, `main` has a single CPU (pool[2]). If any pool is <5, binding raises an error.

**NUMA analysis**:

- With the symmetric NUMA layout above (NUMA0 = 0..7, NUMA1 = 8..16), NPU0 stays within NUMA0, NPU2 stays within NUMA1, but NPU1 spans both NUMA0 (6,7) and NUMA1 (8..11). This is a direct consequence of global slicing over the ordered cpuset; the remainder distribution does not enforce NUMA boundaries.
- If the cpuset numbering is interleaved across NUMA nodes (non‑symmetric layout), cross‑NUMA pools can happen even earlier. This is why symmetric NUMA layout is recommended for best locality.

### Known limitations and future improvements

With the current `global_slice` strategy, some CPU/NPU layouts cannot avoid cross‑NUMA pools. A future enhancement should incorporate NUMA node boundaries into the slicing logic so that pools remain within a single NUMA node whenever possible.

### Example 4: global_slice with visible subset of NPUs

**Inputs**:

- total_npus = 8 (from npu-smi info -m)
- running_npu_list = [2, 3] (filtered by ASCEND_RT_VISIBLE_DEVICES)
- allowed_cpus = [0..39] (40 CPUs)
- NUMA nodes = 0..3 (4 NUMA nodes, symmetric layout; 0..9, 10..19, 20..29, 30..39)

**Global slice**:

- base = 40 // 8 = 5, extra = 0
- Only the visible logical NPUs get pools, but slicing uses the global NPU ID so different processes do not overlap.

|NPU ID|Assigned CPU Cores (global_slice)|Role Division (IRQ/Main/ACL/Release)|
|---|---|---|
|2|10-14|`IRQ`: 10-11, `Main`: 12, `ACL`: 13, `Release`: 14|
|3|15-19|`IRQ`: 15-16, `Main`: 17, `ACL`: 18, `Release`: 19|

### Example 5: A2/310P topo_affinity with NUMA extension

**Inputs**:

- npu_affinity = {0: [0..7], 1: [0..7]} (from `npu-smi info -t topo`)
- allowed_cpus = [0..15] (16 CPUs)
- NUMA nodes = 0..1 (2 NUMA nodes; NUMA0 = 0..7, NUMA1 = 8..15)

**NUMA extension**:

- Both NPUs are on NUMA0, so each pool extends to the nearest NUMA node to reduce contention.
- NPU0 extends to NUMA1 -> [0..15]
- NPU1 extends to NUMA1 -> [0..15]

Because both pools are identical, the allocator applies average distribution across NPUs to avoid overlap. With a pool [0..15] and 2 NPUs, the final pools become:

|NPU ID|Assigned CPU Cores (topo_affinity)|Role Division (IRQ/Main/ACL/Release)|
|---|---|---|
|0|0-7|`IRQ`: 0-1, `Main`: 2-5, `ACL`: 6, `Release`: 7|
|1|8-15|`IRQ`: 8-9, `Main`: 10-13, `ACL`: 14, `Release`: 15|

### Example 6: Minimum CPUs per NPU

**Inputs**:

- total_npus = 2
- allowed_cpus = [0..7] (8 CPUs)
- NUMA nodes = 0..1 (2 NUMA nodes, symmetric layout; NUMA0 = 0..3, NUMA1 = 4..7)

**Result**:

- base = 4, which is < 5, so binding fails with: "Insufficient CPUs for binding with IRQ/ACL/REL reservations..."

|NPU ID|Assigned CPU Cores|Role Division (IRQ/Main/ACL/Release)|
|---|---|---|
|0|N/A|Binding error (insufficient CPUs per NPU)|
|1|N/A|Binding error (insufficient CPUs per NPU)|

To resolve, either reduce total_npus or enlarge the cpuset so that each NPU has at least 5 CPUs.

### Logging and verification

- Logs show the selected binding mode and the allocation plan, for example:
    - `[cpu_bind_mode] mode=global_slice rank=0 visible_npus=[...]`
    - `The CPU allocation plan is as follows: ...`
- You can verify affinity via taskset or /proc/<pid>/status after startup.

## Limitations & Notes

- **ARM‑only**: Binding is skipped on non‑ARM CPUs.
- **Minimum CPU requirement**: Each logical NPU requires at least 5 CPUs. If the cpuset is smaller, binding fails with an error.
- **NUMA symmetry assumption**: For best locality, the current strategies assume the cpuset is evenly distributed across NUMA nodes and CPU numbering aligns with NUMA layout; otherwise NUMA locality may be suboptimal.
    - Example (symmetric layout): 2 NUMA nodes, 64 CPUs total. NUMA0 = CPUs 0–31, NUMA1 = CPUs 32–63, and the cpuset is 0–63. With 4 logical NPUs, global slicing yields 16 CPUs per NPU (0–15, 16–31, 32–47, 48–63), so each NPU’s pool stays within a single NUMA node.
- **Runtime dependencies**:
    - Requires npu‑smi and lscpu commands.
    - IRQ binding requires write access to /proc/irq.
    - Memory binding requires migratepages; otherwise it is skipped.
- **IRQ side effects**: irqbalance may be stopped to avoid overriding bindings.
- **Per‑process behavior**: Only the current rank’s NPU is used for IRQ binding to avoid cross‑process overwrite.

### Debug logging

Use the standard vLLM logging configuration to enable debug logs. The binding process emits debug messages (e.g., `[cpu_global_slice] ...`) when debug level is enabled.

## References

- CPU binding implementation: vllm_ascend/cpu_binding.py (`DeviceInfo`, `CpuAlloc`, `bind_cpus`)
- Worker integration: vllm_ascend/worker/worker.py (`NPUWorker._init_device`)
- Additional config option: docs/source/user_guide/configuration/additional_config.md (`enable_cpu_binding`)
- Tests: tests/ut/device_allocator/test_cpu_binding.py
