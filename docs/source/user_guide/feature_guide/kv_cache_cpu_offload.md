# KV Cache CPU Offload Guide

## Overview

KV Cache CPU Offload enables offloading inactive KV cache blocks from NPU memory to CPU memory, allowing vLLM to handle longer contexts or more concurrent requests when NPU memory is limited. When a prefix cache miss occurs on the NPU but the data exists in CPU memory, the KV cache is asynchronously loaded back to the NPU, reducing recomputation latency.

This feature is built on vLLM's `OffloadingConnector` framework. The upstream `CPUOffloadingSpec` and `TieringOffloadingSpec` names are registered with Ascend NPU implementations that use dedicated NPU streams for efficient asynchronous data transfers between NPU and CPU.

## Key Concepts

- **CPU Block Pool**: A pre-allocated pool of CPU memory blocks. The CPU-only spec uses pinned tensors when available; the multi-tier spec uses a shared mmap region so the scheduler and secondary tiers can access the same bytes.
- **Asynchronous Transfer**: NPU-to-CPU (D2H) and CPU-to-NPU (H2D) transfers are performed on separate NPU streams, overlapping with computation to minimize latency impact.
- **LRU Eviction**: The CPU-side block pool uses an LRU (Least Recently Used) eviction policy to manage limited CPU memory efficiently.
- **Multi-tier Offload**: `TieringOffloadingSpec` reuses vLLM's tiering manager and supports a CPU primary tier with optional secondary tiers such as filesystem storage.

## Usage

### Python API

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "cpu_bytes_to_use": 8 * (1 << 30),
        "blocks_per_chunk": 8,
        "spec_name": "CPUOffloadingSpec",
    },
)

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    gpu_memory_utilization=0.5,
    kv_transfer_config=kv_transfer_config,
)

sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
outputs = llm.generate(["Hello, my name is"], sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### Online Serving

```bash
vllm serve Qwen/Qwen3-0.6B \
    --gpu-memory-utilization 0.5 \
    --kv-transfer-config '{
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": 8589934592,
            "blocks_per_chunk": 8,
            "spec_name": "CPUOffloadingSpec"
        }
    }'
```

### Multi-tier Offload

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "cpu_bytes_to_use": 8 * (1 << 30),
        "blocks_per_chunk": 8,
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [
            {
                "type": "fs",
                "root_dir": "/tmp/vllm_kv_offload",
            }
        ],
    },
)
```

## Configuration Parameters

- `kv_connector`: Must be set to `"OffloadingConnector"`.
- `kv_role`: Set to `"kv_both"` to enable both storing and loading of KV cache.
- `cpu_bytes_to_use`: Total bytes reserved for the CPU tier across all workers. This is required by the vLLM offloading specs.
- `blocks_per_chunk`: Number of NPU KV-cache blocks in one offloaded chunk. It must be greater than zero. Alternatively, `block_size` can specify the chunk size in tokens when all KV-cache groups use the same block size; the two options are mutually exclusive.
- `spec_name`: Use the upstream names `"CPUOffloadingSpec"` for CPU-only offload or `"TieringOffloadingSpec"` for multi-tier offload. vLLM Ascend transparently maps them to the NPU implementations.
- `secondary_tiers`: Optional list of vLLM tiering backends used by `TieringOffloadingSpec`. Each entry is a dict; common keys:
    - `type`: Backend type. Use `"fs"` for filesystem storage.
    - `root_dir`: Directory the tier writes block files into.
    - `n_read_threads` / `n_write_threads`: Thread-pool sizes for the tier's I/O. For high-bandwidth backends (e.g. 3FS over RDMA) increasing these helps saturate parallel bandwidth.

### Tuning the filesystem (SSD / 3FS) secondary tier

The `fs` tier reuses vLLM v0.27.1's batched filesystem implementation. It probes `O_DIRECT` at startup and automatically falls back to buffered I/O when the filesystem rejects direct I/O. To get the best disk/3FS offload throughput:

- Increase `blocks_per_chunk` to transfer larger chunks and reduce metadata overhead, while accounting for the resulting coarser cache-hit granularity.
- Raise `n_read_threads` / `n_write_threads` to match the backend's parallelism (3FS over RDMA benefits from higher concurrency than local SSD).
- Keep the CPU primary tier (`cpu_bytes_to_use`) large enough that the slower disk tier is only reached for genuinely cold data — a secondary tier only improves performance when the working set exceeds CPU capacity.

## How It Works

1. **Normal inference**: KV cache blocks are computed and stored on the NPU as usual.
2. **Eviction to CPU**: When NPU memory is full and new blocks are needed, inactive KV cache blocks are asynchronously copied to CPU memory via a dedicated D2H NPU stream.
3. **Prefix cache hit (CPU)**: When a request shares a prefix with previously computed data, and the prefix cache is not found on NPU but exists in CPU memory, the KV cache blocks are asynchronously loaded back from CPU to NPU via a dedicated H2D NPU stream.
4. **LRU management**: The CPU block pool uses LRU eviction to discard the least recently used blocks when CPU memory is full.

## Optional: KV Cache Events

You can enable KV cache event publishing for monitoring or debugging purposes:

```python
from vllm.config import KVEventsConfig

kv_events_config = KVEventsConfig(
    enable_kv_cache_events=True,
    publisher="zmq",
    endpoint="tcp://*:5555",
    topic="kv_events",
)

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    gpu_memory_utilization=0.5,
    kv_transfer_config=kv_transfer_config,
    kv_events_config=kv_events_config,
)
```

## Notes

- This feature requires vLLM v1 engine.
- Adjust `cpu_bytes_to_use` based on available CPU memory. Reserving too much may cause out-of-memory errors on the host.
- `CPUOffloadingSpec` uses pinned (page-locked) tensors when available. `TieringOffloadingSpec` must use vLLM's shared mmap primary tier; CANN does not provide a memory-copy-compatible equivalent of CUDA's arbitrary mmap host registration, so its NPU↔CPU bandwidth must be benchmarked separately on the target host and CANN version.
- The `gpu_memory_utilization` parameter controls how much NPU memory is reserved for KV cache. Lower values leave less NPU memory for KV cache, making offloading more active.
- For production workloads, benchmark with realistic request patterns to find the optimal `cpu_bytes_to_use` and `blocks_per_chunk` settings.
