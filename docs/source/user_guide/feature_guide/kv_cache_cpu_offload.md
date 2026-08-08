# KV Cache CPU Offload Guide

## Overview

KV Cache CPU Offload enables offloading inactive KV cache blocks from NPU memory to CPU memory, allowing vLLM to handle longer contexts or more concurrent requests when NPU memory is limited. When a prefix cache miss occurs on the NPU but the data exists in CPU memory, the KV cache is asynchronously loaded back to the NPU, reducing recomputation latency.

This feature is built on vLLM's `OffloadingConnector` framework and provides Ascend NPU-specific implementations (`NPUOffloadingSpec` and `NPUTieringOffloadingSpec`) that use dedicated NPU streams for efficient asynchronous data transfers between NPU and CPU.

## Key Concepts

- **CPU Block Pool**: A pre-allocated pool of CPU memory blocks (optionally pinned) used to store offloaded KV cache data.
- **Asynchronous Transfer**: NPU-to-CPU (D2H) and CPU-to-NPU (H2D) transfers are performed on separate NPU streams, overlapping with computation to minimize latency impact.
- **LRU Eviction**: The CPU-side block pool uses an LRU (Least Recently Used) eviction policy to manage limited CPU memory efficiently.
- **Multi-tier Offload**: `NPUTieringOffloadingSpec` reuses vLLM's tiering manager and supports a CPU primary tier with optional secondary tiers such as filesystem storage.

## Usage

### Python API

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": 1000,
        "block_size": 128,
        "spec_name": "NPUOffloadingSpec",
        "spec_module_path": "vllm_ascend.kv_offload.npu",
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
            "num_cpu_blocks": 1000,
            "block_size": 128,
            "spec_name": "NPUOffloadingSpec",
            "spec_module_path": "vllm_ascend.kv_offload.npu"
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
        "block_size": 128,
        "spec_name": "NPUTieringOffloadingSpec",
        "spec_module_path": "vllm_ascend.kv_offload.npu",
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
- `cpu_bytes_to_use`: Bytes to allocate for the CPU offload tier. This is the recommended configuration for current vLLM offloading specs.
- `num_cpu_blocks`: Legacy Ascend configuration for the number of blocks to allocate in CPU memory. It is still accepted by `NPUOffloadingSpec` and `NPUTieringOffloadingSpec` when `cpu_bytes_to_use` is not set.
- `block_size`: The CPU-side block size. Should be a multiple of the NPU-side block size. Typical value: `128`.
- `spec_name`: Use `"NPUOffloadingSpec"` for CPU-only offload or `"NPUTieringOffloadingSpec"` for multi-tier offload. When the Ascend connector registry is initialized, upstream names `"CPUOffloadingSpec"` and `"TieringOffloadingSpec"` are also mapped to the NPU implementations.
- `spec_module_path`: Must be `"vllm_ascend.kv_offload.npu"`.
- `secondary_tiers`: Optional list of vLLM tiering backends used by `NPUTieringOffloadingSpec`. Each entry is a dict; common keys:
    - `type`: Backend type. `"fs_python"` (the Ascend filesystem tier) is the default for disk-backed tiers (SSD/NFS/3FS).
    - `root_dir`: Directory the tier writes block files into.
    - `n_read_threads` / `n_write_threads`: Thread-pool sizes for the tier's I/O. For high-bandwidth backends (e.g. 3FS over RDMA) increasing these helps saturate parallel bandwidth.
    - `use_direct_io` (default `false`): Re-enable `O_DIRECT`. Only for local filesystems/SSDs that support it with aligned buffers; leave `false` on 3FS/FUSE (where it raises `EINVAL`).

### Tuning the filesystem (SSD / 3FS) secondary tier

The `fs_python` tier writes one file per KV block. On FUSE-backed filesystems such as 3FS, per-block metadata operations (directory creation, `stat`, rename) dominate the cost, so the Ascend tier caches created directories to avoid re-issuing `makedirs` per block. To get the best disk/3FS offload throughput:

- Increase `block_size` (e.g. `256`/`512`, must remain a multiple of the NPU block size) to write larger files, reducing the number of files and metadata operations.
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
- Adjust `num_cpu_blocks` based on available CPU memory. Using too many blocks may cause out-of-memory errors on the host.
- Pinned (page-locked) memory is used when available for optimal transfer performance.
- The `gpu_memory_utilization` parameter controls how much NPU memory is reserved for KV cache. Lower values leave less NPU memory for KV cache, making offloading more active.
- For production workloads, benchmark with realistic request patterns to find the optimal `num_cpu_blocks` and `block_size` settings.
