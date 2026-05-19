# CPU Binding

## Overview

CPU binding is an **Ascend-native host-side optimization** for vLLM workers on
ARM servers. **Starting from vllm-ascend v0.18.0rc1, it is enabled by default
through `enable_cpu_binding=True`.**

The feature does not change model execution logic or numerical results. It only
controls CPU placement for the worker process, key runtime threads, memory
pages, and NPU IRQs when the host environment allows it. By keeping the main
worker, ACL, and release threads on dedicated CPU ranges, it **helps reduce
context-switch overhead from scheduler preemption on busy hosts.**

## Why CPU Binding?

On multi-socket ARM systems, the Linux scheduler may place worker threads on
CPUs far from the NPU that the worker drives. This can increase cross-NUMA
traffic, increase thread preemption, and introduce latency jitter. The Ascend
backend therefore owns a CPU allocation policy to **reduce cross-NUMA traffic,
reduce thread preemption, and improve latency stability** instead of relying on
upstream GPU NUMA binding flags.

This is also why upstream NUMA flags are adapted on Ascend:

- `--numa-bind` is converted to `additional_config={"enable_cpu_binding": true}`.
- `--numa-bind-nodes` and `--numa-bind-cpus` are ignored because Ascend computes CPU pools from NPU topology or global logical NPU IDs.

## How It Works?

The allocator derives its plan from runtime host state:

| Input | Source | Purpose |
| --- | --- | --- |
| Allowed CPUs | `/proc/self/status` `Cpus_allowed_list` | The only CPUs eligible for binding. Container cpusets are respected. |
| Logical NPU map | `npu-smi info -m` | Maps card/chip IDs to global logical NPU IDs and gives `total_logic_npus`. |
| Running NPUs | `npu-smi info` process table, filtered by `ASCEND_RT_VISIBLE_DEVICES` | Identifies the logical NPUs used by this worker process. |
| Topology affinity | `npu-smi info -t topo` | Provides NPU-to-CPU affinity for `topo_affinity` mode. |
| CPU NUMA map | `lscpu -e=CPU,NODE` | Used to extend single-NUMA affinity pools to the next NUMA node. |

### Strategy Selection

The binding strategy is selected by Ascend device type:

| Device type | Strategy | Reason |
| --- | --- | --- |
| A3 | `global_slice` | A3 uses HCCS card-to-card interconnect. Each NPU is nearly equidistant from all NUMA nodes, so there is no strong NPU-to-NUMA affinity signal. Global logical NPU ID based slicing gives deterministic, non-overlapping CPU pools and CPU/NUMA isolation between workers. |
| A2, Atlas 300 inference products, and other non-A3 device types | `topo_affinity` | A2 and Atlas 300 inference products provide NPU-to-CPU affinity information through `npu-smi info -t topo`. Non-A3 device types use this topology signal when available. |

If `topo_affinity` is selected but topo affinity is unavailable, the allocator falls back to `global_slice`.

### CPU Pool Construction

#### global_slice

`global_slice` is designed for A3. Because A3's **HCCS interconnect makes the
distance from each NPU to each NUMA node nearly the same**, topology affinity is
not a useful placement signal. The allocator therefore partitions the sorted
`allowed_cpus` list by global logical NPU ID.

1. Determine `total_npus` in this order:
   - `total_logic_npus` from `npu-smi info -m`
   - number of topo affinity entries
   - number of running NPUs
2. Compute:
   - `base = len(allowed_cpus) // total_npus`
   - `extra = len(allowed_cpus) % total_npus`
3. Each logical NPU gets a deterministic slice:
   - NPU IDs `< extra` receive `base + 1` CPUs.
   - Remaining NPU IDs receive `base` CPUs.
4. Only running NPUs are materialized into `npu_cpu_pool`.

This is the key property: two independent worker processes with the same cpuset
but different visible NPU IDs still get **non-overlapping CPU pools** because
both processes slice against the same global NPU ID space. With a NUMA-aligned
cpuset, this also provides **CPU/NUMA isolation between workers**, so one worker
does not share the same CPU or NUMA slice with another worker.

`global_slice` requires `base >= 5`, because every NPU pool reserves:

- 2 CPUs for SQ/CQ IRQ binding
- at least 1 CPU for the main worker
- 1 CPU for ACL thread
- 1 CPU for release thread

#### topo_affinity

`topo_affinity` is designed for A2, Atlas 300 inference products, and
other non-A3 device types. A2 and Atlas 300 inference products expose
**meaningful NPU-to-CPU affinity information**, so the allocator starts from NPU
topology affinity when it is available and then avoids overlap for shared
affinity groups.

1. Build candidate NPUs from all logical NPUs:
   - always include running NPUs
   - include non-running NPUs only when their affinity overlaps this process's allowed cpuset
2. For each candidate NPU, intersect topo affinity with `allowed_cpus`.
3. If the intersection is empty for a candidate, binding fails for this rank.
4. If the affinity CPUs are all on one NUMA node, extend the pool with CPUs from the next NUMA node, constrained by `allowed_cpus`.
5. Group NPUs with identical extended pools and split each shared pool evenly across that group.
6. Keep only running NPUs in the final `npu_cpu_pool`.

The non-running candidate step is intentional. It prevents two independent
single-card workers from selecting the same CPU range when their visible NPUs
share the same topology affinity.

### Role Split

After a CPU pool is built, the allocator splits it by role:

| Role | CPUs |
| --- | --- |
| SQ/CQ IRQ | `pool[0]`, `pool[1]` |
| Main worker process and subthreads | `pool[2:-2]` |
| ACL thread | `pool[-2]` |
| Release thread | `pool[-1]` |

If a final pool has fewer than 5 CPUs, binding fails for this rank and the worker logs a warning from the caller.

## Conditional Host Tuning

After CPU affinity is applied, CPU binding can also apply two host-side tuning
steps when the environment supports them:

- Memory migration uses `migratepages` to move the worker process's existing
  pages to the selected NUMA node. This keeps the worker closer to the memory it
  reads and reduces remote-NUMA memory read latency.
- IRQ binding places NPU IRQ handling on the CPUs reserved for the corresponding
  NPU when `/proc/irq` is writable and IRQ files can be resolved.

These are conditional parts of CPU binding, not separate feature switches. If a
host prerequisite is missing, that step is skipped while CPU thread binding
still proceeds. Missing `migratepages` can still leave pages on remote NUMA
nodes, so **latency or throughput may regress compared with a full CPU binding
setup.**

## Examples

### A3 inference server with 640 CPUs and 16 NPUs

Inputs:

- `allowed_cpus = [0..639]`
- `total_logic_npus = 16`
- `running_npu_list = [0..15]`

Computation:

- `base = 640 // 16 = 40`
- `extra = 0`
- Worker `i` driving logical NPU `i` receives CPU slice
  `[i * 40 .. i * 40 + 39]`.

Global slice view:

```text
CPU range: 0                                                             639
           |-- worker0/NPU0 --|-- worker1/NPU1 --| ... |-- worker15/NPU15 --|
           |      0-39        |      40-79       | ... |      600-639       |
```

Role split inside each worker slice:

```text
40-CPU worker slice
| IRQ CPUs | main worker process and subthreads | ACL thread | release thread |
|  c0-c1   |              c2-c37                |    c38     |      c39       |
```

Concrete examples:

| Worker | Logical NPU | CPU pool | IRQ CPUs | Main CPUs | ACL CPU | Release CPU |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0-39 | 0-1 | 2-37 | 38 | 39 |
| 1 | 1 | 40-79 | 40-41 | 42-77 | 78 | 79 |
| ... | ... | ... | ... | ... | ... | ... |
| 15 | 15 | 600-639 | 600-601 | 602-637 | 638 | 639 |

This layout remains deterministic even when different worker processes share
the same cpuset, because slicing is based on the global logical NPU ID.

### A2 topo_affinity with hidden same-affinity NPUs

Inputs from an A2 topology:

- NPU0 affinity: 144-167
- NPU2 affinity: 144-167
- Process A sees only NPU0
- Process B sees only NPU2
- Both processes have `allowed_cpus = [144..191]`

The allocator includes the hidden same-affinity NPU as a candidate in each
process, splits the shared extended pool, and then keeps only the visible NPU in
the final pool.

Final pools:

| Process | Visible NPU | Final CPU pool |
| --- | --- | --- |
| A | 0 | 144-167 |
| B | 2 | 168-191 |

This avoids overlapping CPU pools even when the two workers are launched as independent single-card services.

## Logs

The allocator logs the selected mode and allocation plan:

```text
[cpu_bind_mode] mode=topo_affinity rank=0 visible_npus=[0]
The CPU allocation plan is as follows:
NPU0: main=[...] acl=[...] release=[...]
```

## Limitations

- CPU binding runs only on ARM. It is skipped on x86_64.
- Each final NPU pool must have at least 5 CPUs.
- `global_slice` is deterministic and provides CPU/NUMA isolation when the
  cpuset is NUMA-aligned, but it cannot guarantee NUMA-local pools when CPU
  numbering or cpuset layout crosses NUMA boundaries.
- `topo_affinity` depends on usable output from `npu-smi info -t topo`.
- IRQ binding requires writable `/proc/irq` and resolvable PCI/IRQ information.
- Memory migration requires `migratepages`; otherwise only memory migration is
  skipped. CPU affinity still applies, but performance may degrade because
  existing pages are not moved to the target NUMA node and may be read through
  higher-latency remote NUMA access.
- If an exception escapes the binding flow, `NPUWorker` logs a warning and skips CPU binding for that rank.

## References

- Implementation: `vllm_ascend/cpu_binding.py`
- Worker integration: `vllm_ascend/worker/worker.py`
- Config: `vllm_ascend/ascend_config.py` and `docs/source/user_guide/configuration/additional_config.md`
- Tests: `tests/ut/device_allocator/test_cpu_binding.py`
