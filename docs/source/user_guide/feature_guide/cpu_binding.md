# CPU Binding

## Overview

CPU Binding is a performance optimization feature for vLLM, specifically designed for servers equipped with **ARM architecture and Ascend NPUs**. It pins vLLM processes and threads to specific CPU cores to reduce CPU–NPU cross‑NUMA communication overhead and stabilize inference latency. This feature only adjusts host-side CPU affinity policies and **does not alter model execution logic or impact inference results**.

## Usage

### Online serving example with CPU binding enabled (by default)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --additional-config '{"enable_cpu_binding": true}'
```

### Online serving example with CPU binding disabled

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --additional-config '{"enable_cpu_binding": false}'
```

### Offline inference example with CPU binding enabled

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    additional_config={"enable_cpu_binding": True},
)
```

### Offline inference example with CPU binding disabled

```python
from vllm import LLM

llm = LLM(
  model="Qwen/Qwen2.5-7B-Instruct",
  additional_config={"enable_cpu_binding": False},
)
```

## Dependencies

### Installation

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y util-linux numactl procps
```

#### RHEL/CentOS/Alma/Rocky

```bash
sudo yum install -y util-linux numactl procps-ng
```

#### openEuler

```bash
sudo dnf install -y util-linux numactl procps-ng
```

### IRQ binding's additional considerations

For best results, if you run inside a Docker container where `systemctl` is likely unavailable, stop the `irqbalance` service on the host manually before starting vLLM. Also make sure the container has the necessary permissions to write to `/proc/irq/*/smp_affinity` for IRQ binding:

- **Stop `irqbalance` service**:

    For example, on Ubuntu system, you can run the following command to stop irqbalance:

    ```bash
    sudo systemctl stop irqbalance
    ```

    After you finish the vLLM process, you can restore irqbalance on the host:

    ```bash
    sudo systemctl start irqbalance
    ```

- **Permissions**:
    - Read access to `/proc/self/status` and `/proc/interrupts`
    - Write access to `/proc/irq/*/smp_affinity` for IRQ binding

## Common Issues & Troubleshooting

|Error/Warning Message|Core Cause|Solution|
|---|---|---|
|Can not get running npu info.|The npu-smi process table is empty, or the `ASCEND_RT_VISIBLE_DEVICES` environment variable filters out all NPUs.|1. Ensure the process is running on visible NPUs; 2. Verify that the `ASCEND_RT_VISIBLE_DEVICES` value matches the actual logical NPU IDs.|
|Insufficient CPUs for binding...|The number of CPU cores allocated to each NPU is less than the minimum requirement of 5.|1. Expand the allowed CPU list; 2. Reduce the number of visible NPUs.|
|NPU topo affinity not found...|npu-smi is unable to retrieve NPU topology affinity information.|Verify the integrity of the npu-smi installation and ensure the user has sufficient execution permissions.|
|Bind cpus failed in rankX...|The CPU binding process failed (e.g., taskset is unavailable, or the user lacks write permissions for /proc/irq).|1. Confirm that required tools (taskset, lscpu, npu-smi) are installed and available; 2. Verify the Cpus_allowed_list in `/proc/self/status` is valid.|

## Key Limitations

- ARM architecture only: Binding is automatically skipped on x86_64 systems.

- Symmetric NUMA layout required for optimal performance: CPU numbering should be aligned with NUMA nodes. Non-symmetric layouts may result in cross-NUMA CPU pools, reducing locality.

- IRQ binding requires write permissions for /proc/irq. Memory binding depends on the `migratepages` tool; if unavailable, memory migration is skipped.

## FAQ

**Q1: Does CPU binding work on x86_64?**

No. The binding is skipped on non‑ARM CPUs.

**Q2: Why are only the current rank’s IRQs bound?**

To avoid multiple processes overwriting IRQ affinity settings for the same device.

**Q3: What if my cpuset already limits CPUs?**

The binder uses Cpus_allowed_list from /proc/self/status as the only eligible CPU set. Ensure this list is large enough.

**Q4: Does CPU binding change model outputs?**

No. It only affects host‑side affinity and should not change numerical results.

---

## Summary

1. **Core Objective**: Reduce cross‑NUMA communication by pinning vLLM processes and threads to specific CPU cores, thereby stabilizing inference latency in Ascend NPU deployments (only applicable to ARM architectures).

2. **Usage**: Enable or disable with `enable_cpu_binding` via `additional_config` in both online and offline workflows.

3. **Key Limitations**: ARM‑only; relies on symmetric NUMA layouts; binding fails if the CPU pool has fewer than 5 cores; binding errors trigger a warning log but do not terminate the process.
