# Layerwise KV Pool and Layer Reuse

## Overview

Layerwise KV pooling is a feature that performs KV cache save and load operations
on a **per-layer basis**, interleaved with the model's layer-by-layer forward
computation. This allows KV transfer to overlap with compute, reducing the
end-to-end latency compared to saving/loading all KV cache at once after the
entire forward pass completes.

On top of layerwise pooling, **Layer Reuse** further reduces device memory
footprint by sharing a small number of transfer buffers across multiple model
layers. Instead of allocating a dedicated buffer for each layer, non-independent
layers reuse a limited pool of buffers, which in turn shrinks the physical KV
cache tensors allocated on the NPU.

This feature is supported in two contexts:

1. **AscendStore KV Pool** (`AscendStoreConnector` with `use_layerwise: true`)
   — general-purpose KV pool for PD disaggregation or PD-mixed scenarios.
2. **Mooncake Layerwise P2P** (`MooncakeLayerwiseConnector`) — direct
   P2P KV transfer between prefill and decode nodes in PD disaggregation.

---

## Architecture

### Layerwise Transfer Flow

The core idea of layerwise transfer is to interleave KV save/load with the
model's per-layer forward pass:

```
Step Start
  │
  ├─> build per-layer load tasks for all N layers
  ├─> submit first K layers for prefetch (K = NUM_PREFETCH_LAYERS)
  │
  ▼
  For layer i in 0..N-1:
    ├─> wait_for_layer_load(i)   ← block until KV for layer i is loaded into HBM
    ├─> compute attention for layer i
    ├─> save_kv_layer(i)         ← enqueue KV save to transfer thread
    │
  ▼
  For final layer N-1:
    ├─> wait for save completion event
    │
  ▼
Step End
```

By starting the load of layer `i+1` (or later layers) while computing layer `i`,
the KV data transfer overlaps with the attention computation of adjacent layers.
The number of prefetch layers controls how far ahead the load pipeline runs.

### System Components

#### Scheduler Side (`KVPoolScheduler`)

- **Hit Detection**: Queries the KV pool backend (Mooncake/MemCache/Yuanrong) to
  determine which blocks are already cached. In layerwise GVA mode (MemCache
  backend), uses GVA-based lookup; otherwise uses key-based lookup.
- **GVA Allocation**: For MemCache backend, allocates Global Virtual Address (GVA)
  space in the KV pool for all requested blocks. GVA enables direct memory
  addressing without key-based indirection, which is essential for efficient
  buffer reuse.
- **Metadata Building**: Constructs per-request `ReqMeta` containing block IDs,
  block hashes, GVA addresses, and load/save specifications for the worker side.

#### Worker Side (`KVPoolWorker`)

- **Per-Layer Task Lists**: Maintains `layer_load_tasks[N]` and
  `layer_save_tasks[N]` for all N model layers. Each task contains block ranges
  specifying which blocks to load from or save to the KV pool.
- **Transfer Threads**: Spawns a send thread and a receive thread that process
  tasks asynchronously. The receive thread handles H2D (Host-to-Device) data
  loading, while the send thread handles D2H (Device-to-Host) saving.
- **Synchronization**: Uses `layer_load_finished_events[layer_id]` and
  `layer_save_finished_events[layer_id]` threading events to coordinate between
  the main compute thread and the background transfer threads.

#### Transfer Thread Classes (`kv_transfer.py`)

| Thread Class | Backend | Description |
| :--- | :--- | :--- |
| `KVCacheStoreKeyLayerSendingThread` | Mooncake (key-based) | Per-layer send using key-based `put` to Mooncake Store |
| `KVCacheStoreKeyLayerRecvingThread` | Mooncake (key-based) | Per-layer recv using key-based `get` from Mooncake Store |
| `KVCacheStoreLayerSendingThread` | MemCache (GVA-based) | Per-layer send using `batch_copy(direction=0)` with GVA addressing |
| `KVCacheStoreLayerRecvingThread` | MemCache (GVA-based) | Per-layer recv using `batch_copy(direction=1)` with H2D staggering |

The GVA-based threads use `LayerBatchBuilder` to convert per-layer
`LayerTransferTask` objects into flat numpy arrays of addresses, sizes, and GVAs
for efficient batch transfer.

#### Attention Compute Gate (`memcache_comm_fence.py`)

In MemCache layerwise mode, a gate mechanism coordinates the compute stream
with KV data transfer:

- Before attention compute starts, the attention worker records an NPU event
  via `record_attention_compute_start()`.
- Background transfer threads wait on this event before submitting H2D/L2G
  (Host-to-Device / Load-to-GPU) work.
- This ensures KV data transfer for the current layer begins only when the
  compute stream has actually reached the attention boundary, avoiding
  premature transfers that could interfere with the previous layer's compute.

---

## Layer Reuse

### Motivation

In a model with N layers, a naive layerwise implementation would allocate N
separate transfer buffers (one per layer) on the NPU. For large models, this
consumes significant device memory. Layer Reuse addresses this by sharing a
smaller pool of buffers across non-independent layers.

### How It Works

**Layer Classification**: Layers are divided into two categories:

- **Independent layers** — always have dedicated buffers. By default, the first
  and last layers (`layer 0` and `layer N-1`) are independent.
- **Reused layers** — share a pool of `num_shared_buffers` transfer buffers.

**Buffer Assignment**: Reused layers are assigned to shared buffers in a
round-robin fashion. For example, with 32 layers, 4 shared buffers, and
independent layers `[0, 31]`:

```
Layer 0  → dedicated buffer (independent)
Layer 1  → shared buffer 0
Layer 2  → shared buffer 1
Layer 3  → shared buffer 2
Layer 4  → shared buffer 3
Layer 5  → shared buffer 0  (reuses buffer 0 after layer 1 finishes)
...
Layer 30 → shared buffer 1
Layer 31 → dedicated buffer (independent)
```

**KV Cache Tensor Sharing**: When layer reuse is active
(`has_layer_reuse = True`), the KV cache configuration is patched via
[`get_kv_cache_config_from_groups()`](https://github.com/xxx) to assign
multiple model layers to the same physical KV cache tensor. The function
`get_layerwise_storage_indices()` maps each shared buffer to the list of
layers that share it. This reduces the total number of KV cache tensors from
N to `num_shared_buffers + len(independent_layers)`.

**Prefetch Constraint**: When a layer's buffer is reused, that layer's KV
load cannot begin until the previous occupant of the same buffer has finished
saving its KV data. The `prefetch_layer_map` tracks this dependency: for each
reused layer beyond the first `num_shared_buffers` reused layers, it records
which layer's save must complete before its load can be submitted.

### KV Cache Memory Calculation

When layer reuse is active, the effective available memory for KV cache is
adjusted to account for buffer sharing:

```python
effective_avail_mem = avail_mem * total_layers // reuse_layers
```

This ensures the memory check correctly accounts for the fact that fewer
physical buffers are needed.

---

## Enabling Layerwise and Layer Reuse

### Enabling Layerwise KV Pooling

Add `use_layerwise: true` to the `kv_connector_extra_config` of the
`AscendStoreConnector`:

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "lookup_rpc_port": "0",
        "use_layerwise": true
    }
}
```

- `backend: "memcache"` — Required for layer reuse (GVA-based layerwise).
  Mooncake backend supports key-based layerwise but not layer reuse.
- `backend: "mooncake"` — Supports key-based layerwise transfer without
  buffer reuse.

### Enabling Layer Reuse

Layer reuse is controlled by environment variables. Set them before starting
vLLM:

```bash
# Limit shared buffers to 4 (model has more than 4 non-independent layers)
export VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS=4

# Number of layers to prefetch for load (default: 2)
export VLLM_ASCEND_KV_POOL_LAYERWISE_PREFETCH_LAYERS=2

# Independent layers (default: first and last). Supports negative indices.
export VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS="0,31"
```

Layer reuse is automatically activated when
`VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS` is set to a value smaller
than the number of non-independent layers.

### Performance Warning

Layer reuse is designed for the **prefill producer** node in PD disaggregation.
When enabled on other roles (decode consumer, mixed deployment), it may cause
significant performance degradation. A warning is logged if layer reuse is
detected on a non-producer node.

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS` | `None` (all layers) | Number of reusable transfer buffers. Set to a value less than non-independent layers to enable layer reuse. |
| `VLLM_ASCEND_KV_POOL_LAYERWISE_PREFETCH_LAYERS` | `2` | Number of layers to prefetch for KV load ahead of compute. |
| `VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS` | `None` (first and last) | Comma-separated list of layer indices that always get dedicated buffers. Supports negative indices (`-1` = last layer). Set to `"all"` to disable reuse entirely. Set to `""` (empty) to make every layer reusable. |
| `VLLM_ASCEND_KV_POOL_DRAM_SIZE` | `"0"` | DRAM size allocated for KV pool storage (e.g., `"2GB"`). Used to calculate LRU cache capacity. |

### CUDA Graph Compatibility

When `use_layerwise: true` is set, the connector requires **PIECEWISE** CUDA
graph mode. This is because the layerwise load/save hooks perform async
synchronization that cannot be safely captured in full CUDA graphs. The
connector reports this requirement via `requires_piecewise_for_cudagraph()`.

---

## Example: PD Disaggregation with Layerwise and Layer Reuse

The following example shows how to configure layerwise KV pooling with layer
reuse (4 shared buffers) using the MemCache backend in a PD disaggregation
scenario. This is the recommended setup for using layer reuse.

### Prefill (Producer) Node

```shell
export VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS=4
export VLLM_ASCEND_KV_POOL_LAYERWISE_PREFETCH_LAYERS=2

source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/path/to/mmc-meta.conf
export MMC_LOCAL_CONFIG_PATH=/path/to/mmc-local.conf

vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 30050 \
  --enforce-eager \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "lookup_rpc_port": "0",
        "use_layerwise": true
    }
  }'
```

### Decode (Consumer) Node

```shell
source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/path/to/mmc-meta.conf
export MMC_LOCAL_CONFIG_PATH=/path/to/mmc-local.conf

vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 30060 \
  --enforce-eager \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "lookup_rpc_port": "1",
        "use_layerwise": true
    }
  }'
```

> **Note**
>
> Layer reuse should only be enabled on the prefill (producer) node. The
> decode (consumer) node reuses the same `use_layerwise: true` setting but
> does not participate in saving KV cache unless `consumer_is_to_put: true`
> is explicitly set.

### PD-Mixed Deployment

```shell
export VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS=4

source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/path/to/mmc-meta.conf
export MMC_LOCAL_CONFIG_PATH=/path/to/mmc-local.conf

vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 30050 \
  --enforce-eager \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "lookup_rpc_port": "0",
        "use_layerwise": true
    }
  }'
```

> **Warning**
>
> When using PD-Mixed deployment with `kv_role: "kv_both"`, enabling layer
> reuse (`VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS` < non-independent
> layers) may have poor performance. The system logs a warning in this case.

---

## Mooncake Layerwise P2P Connector

In addition to the AscendStore-based layerwise pooling, vLLM Ascend also
provides `MooncakeLayerwiseConnector` for direct P2P KV transfer between
prefill and decode nodes. This connector performs per-layer KV cache save
and load directly over RDMA-like P2P transfers without an intermediate KV
pool.

### Key Features

- **Per-Layer P2P Transfer**: Saves KV cache layer-by-layer from prefill
  and transfers directly to decode nodes via Mooncake `batch_transfer_sync_write`.
- **ZMQ Side Channel**: Uses a ZMQ-based control channel (`KVCacheRecvingLayerThread`)
  for metadata exchange between producer and consumer.
- **TP Resharding**: Supports context-parallel (CP) and tensor-parallel (TP)
  resharding, including unequal partition ratios (`pd_head_ratio > 1`).
- **KV Quantization**: Optional KV cache quantization during transfer to reduce
  bandwidth usage.
- **Multi-Part Completion Tracking**: Handles multi-part block transfers and
  tracks completion through the ZMQ side channel.

### Usage

Configure as part of a `MultiConnector` setup:

```json
{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeLayerwiseConnector",
                "kv_role": "kv_producer",
                "kv_port": "20001",
                "kv_connector_extra_config": {
                    "prefill": {"dp_size": 1, "tp_size": 4},
                    "decode": {"dp_size": 1, "tp_size": 4}
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "backend": "mooncake",
                    "lookup_rpc_port": "0"
                }
            }
        ]
    }
}
```

---

## Transfer Thread Synchronization

### Save Path

1. `save_kv_layer(layer_id)` records an NPU event to mark the compute boundary
   for the current layer.
2. The save task (containing block ranges and the NPU event) is enqueued into
   the send thread's queue.
3. The send thread waits on the NPU event, then performs `batch_copy(0)`
   (D2H transfer) or `batch_transfer_sync_write` (P2P transfer).
4. After the last layer (N-1), the main thread blocks on
   `layer_save_finished_events[N-1]` to ensure all saves are complete before
   the step ends.

### Load Path

1. At step start, `process_layer_data()` builds per-layer load task lists.
2. `_submit_ready_layer_loads()` prefetches the first
   `NUM_PREFETCH_LAYERS` layers into the recv thread queue.
3. For layers subject to reuse, `prefetch_layer_map` gates prefetch submission
   to wait for the dependent layer's save to complete.
4. The recv thread performs `batch_copy(1)` (H2D transfer) and signals
   `layer_load_finished_events[layer_id]` upon completion.
5. `wait_for_layer_load(layer_id)` blocks the main thread until the event
   fires, then clears it for the next step.

### Attention Gate Coordination (MemCache Only)

For the MemCache backend with layerwise enabled, transfer threads coordinate
with the compute stream through `AttentionComputeStartGate`:

- A new gate is created per layer via `reset_attention_compute_start_gate()`.
- The attention worker calls `record_attention_compute_start()` to record
  an NPU event just before launching the attention operation.
- Prefetch tasks submitted for later layers hold a reference to the gate
  and wait on it before starting their H2D transfers.
- This ensures KV data for a layer is not loaded before the compute stream
  has actually reached that layer's attention boundary.

---

## Troubleshooting

### Layer reuse causes poor performance on decode nodes

Layer reuse is designed for prefill producer nodes. On decode or mixed nodes,
disable it by setting `VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS` to
a value equal to or greater than the number of non-independent layers, or
remove the environment variable to use the default (all layers have dedicated
buffers).

### Layerwise load/save timeout

If you see log messages like `"Layerwise N load wait timed out"` or
`"Layerwise N save wait timed out"`, the transfer thread may be stalled.
Check:

- MemCache/Mooncake master service is running and accessible.
- Network connectivity between nodes for RDMA/SDMA transfers.
- `ASCEND_TRANSFER_TIMEOUT` and `ASCEND_CONNECT_TIMEOUT` are set
  appropriately for your cluster size.

### KV cache memory check fails with layer reuse

When layer reuse is active, the KV cache memory check uses an adjusted
effective available memory. If the check fails, try:

- Increasing `num_shared_buffers` to reduce the sharing ratio.
- Adding more layers to `independent_layers` to reduce buffer contention.
- Reducing `max_model_len` or increasing `gpu_memory_utilization`.

---

## References

- [KV Pool Deployment Guide](./kv_pool.md)
- [EPD Disaggregation](./epd_disaggregation.md)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)
- [MemCache Documentation](https://gitcode.com/Ascend/memcache/blob/develop/doc/memcache_config.md)
