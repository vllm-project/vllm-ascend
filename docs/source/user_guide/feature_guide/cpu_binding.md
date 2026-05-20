# CPU Binding

**Starting from vllm-ascend v0.18.0rc1, CPU binding is enabled by default on
ARM-based Ascend servers.**

**You usually do not need to configure it manually.** Set `enable_cpu_binding`
only when you want to disable it or make the default explicit.

## Benefits of CPU Binding

CPU Binding improves **host-side scheduling** for multi-socket ARM servers with
Ascend NPUs. It is designed to solve three common host-side inference performance issues:

- **Lower cross-NUMA traffic.** Worker processes stay closer to the CPU and
  memory resources selected for their active NPU, reducing remote NUMA access.
- **Lower context-switch overhead from thread preemption.** Key runtime threads
  run on stable CPU ranges, reducing scheduler movement and CPU contention on
  busy hosts.
- **Better latency stability and multi-worker isolation.** Independent workers
  avoid sharing the same CPU/NUMA resources, which helps reduce tail-latency
  jitter and makes throughput more predictable during multi-NPU serving.

This feature is a host-side performance optimization. **It does not change model
execution logic or numerical outputs.** When memory migration support is
unavailable, CPU affinity still works, but memory locality may be worse and
latency or throughput may degrade.

## Usage

### Online Serving

Default behavior:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

Disable CPU binding:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --additional-config '{"enable_cpu_binding": false}'
```

### Offline Inference

Default behavior:

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
```

Disable CPU binding:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    additional_config={"enable_cpu_binding": False},
)
```

## Requirements

Official vllm-ascend images have already included `util-linux` and `procps` /
`procps-ng` in v0.18.0rc1 and earlier releases. **Starting from v0.18.0rc1, the
official images also include `numactl`.**

If you are not using the official image, install the host tools manually:

```bash
# Ubuntu/Debian
sudo apt-get install -y util-linux numactl procps

# RHEL/CentOS/Alma/Rocky
sudo yum install -y util-linux numactl procps-ng

# openEuler
sudo dnf install -y util-linux numactl procps-ng
```

**Without `numactl` / `migratepages`, vLLM Ascend skips only memory migration.**
The worker process and runtime threads are still pinned, but pages already
placed on remote NUMA nodes are not migrated, which **can reduce locality and
degrade latency or throughput.**

For optimal locality, use a cpuset that is evenly distributed across NUMA
nodes. Unbalanced cpusets may reduce the locality benefit of CPU binding.

For IRQ binding, the process also needs permission to read `/proc/interrupts`
and write `/proc/irq/*/smp_affinity`. If `irqbalance` is running and the process
can use `systemctl`, vLLM Ascend stops it before applying IRQ affinity. In
containers where `systemctl` is unavailable, stop `irqbalance` on the host when
IRQ affinity matters.

On the host, stop `irqbalance` before starting vLLM when you need stable IRQ
affinity:

```bash
sudo systemctl stop irqbalance
```

After the vLLM service exits, restart it if the host should return to the
default IRQ balancing policy:

```bash
sudo systemctl start irqbalance
```

## Troubleshooting

| Message | Meaning | Action |
| --- | --- | --- |
| `CPU binding skipped: non-ARM CPU detected.` | CPU binding only runs on ARM. | No action needed on x86_64. |
| `Can not get running npu info.` | No running NPU was found, or `ASCEND_RT_VISIBLE_DEVICES` filtered all NPUs. | Check visible NPU IDs and `npu-smi info`. |
| `Insufficient CPUs for binding...` | Fewer than 5 CPUs are available per logical NPU. | Expand the cpuset or reduce visible NPUs. |
| `NPU topo affinity not found...` | Topology affinity is unavailable. | vLLM Ascend falls back to `global_slice`; check `npu-smi info -t topo` only if topology affinity is expected on this device. |
| `The 'migratepages' command is not available...` | Memory migration is skipped, while CPU thread binding still proceeds. | Install `numactl` if NUMA locality or performance is affected. |
| `Bind cpus failed in rank...` | A binding step failed and CPU binding was skipped for that rank. | Check `taskset`, `lscpu`, `npu-smi`, cpuset size, and `/proc/irq` permissions. |
