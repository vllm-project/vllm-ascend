# Layerwise Prefill KV Cache Offload

Layerwise Prefill KV cache offload reduces the NPU memory used by KV cache on a
dedicated Prefill node. It combines two mechanisms:

1. KV cache is transferred between NPU memory and the Memcache-backed KV Pool
   layer by layer, overlapping transfer with attention computation.
2. Multiple logical layers reuse a smaller set of physical NPU KV cache
   buffers. A buffer is reused only after the previous layer assigned to that
   buffer has finished saving its KV cache.

This feature builds on [Layerwise KV Pool](layerwise_kv_pool.md). Read that
guide first for the Memcache deployment, huge-page setup, and general
AscendStore configuration.

Request-scoped partial blocks are saved and restored across scheduler steps.
This allows shared buffers to remain enabled for non-block-aligned chunked
Prefill, PD-Mixed inference, and Decode.

## Requirements

Layerwise Prefill offload currently requires:

- `AscendStoreConnector`;
- `kv_role: "kv_producer"` on a dedicated Prefill node, or `kv_role:
  "kv_both"` for PD-Mixed inference;
- `backend: "memcache"`;
- `use_layerwise: true`;
- an MLA or SFA attention backend with the layerwise wait/save integration;
- identical tensor size for layers that share the same component buffer.

When graph execution is enabled, AscendStore requests PIECEWISE graph mode for
layerwise operations. Eager mode is also supported.

## Configuration

Configure the dedicated Prefill node as follows:

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "mooncake_rpc_port": "0",
        "use_layerwise": true,
        "layerwise_num_shared_buffers": 3,
        "layerwise_independent_layers": [0],
        "layerwise_prefetch_layers": 3
    }
}
```

The example keeps the first physical layer independent and assigns all other
layers to three reusable buffers. The values are examples rather than universal
recommendations; choose them according to available NPU memory and transfer
bandwidth.

### Core parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `use_layerwise` | `false` | Enables layer-by-layer KV transfer. |
| `backend` | `"mooncake"` | Must be `"memcache"` for shared-buffer layerwise offload. |
| `layerwise_num_shared_buffers` | Number of physical layers | Number of reusable buffers assigned to non-independent layers. If omitted, no cross-layer buffer reuse is enabled. The value must be at least `1`. |
| `layerwise_independent_layers` | `[0]` | Base transformer layers that keep dedicated buffers. Accepts a list of integers or `"all"`. Negative indices are resolved against the base transformer layers. |

### Optional transfer tuning

The following options do not change the shared-buffer layout or feature
semantics. They only tune transfer overlap, batching, and cross-rank timing.
The defaults can normally be used without adjustment.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `layerwise_prefetch_layers` | Automatically selected | Maximum number of layer loads submitted ahead of the compute frontier. The value must be at least `1`. |
| `layerwise_max_transfer_blocks` | `0` | Maximum number of blocks in one transfer batch. `0` means unlimited. |
| `layerwise_max_transfer_bytes` | `0` | Maximum bytes in one transfer batch. `0` means unlimited. |
| `h2d_stagger_us` | `0` | Delay in microseconds used to stagger H2D copies across ranks. |

## Buffer Layout

Let:

- `N` be the number of base transformer layers;
- `I` be the number of independent layers;
- `R = N - I` be the number of reusable layers;
- `B` be `layerwise_num_shared_buffers`.

The number of physical layer slots is:

```text
I + min(B, R)
```

Cross-layer reuse is active only when `R > B`. Reusable layers are assigned to
the `B` buffers in round-robin order.

For example, with 27 physical layers, independent layer `[0]`, and three shared
buffers:

```text
independent slot: [0]
shared slot 0:    [1, 4, 7, 10, 13, 16, 19, 22, 25]
shared slot 1:    [2, 5, 8, 11, 14, 17, 20, 23, 26]
shared slot 2:    [3, 6, 9, 12, 15, 18, 21, 24]
```

Layer 4 reuses layer 1's physical buffer, layer 7 reuses layer 4's buffer, and
so on. Before loading layer 4, the transfer thread waits until layer 1 has
finished saving.

For the supported uniform KV layout, the approximate logical-to-physical
memory factor is `N / (I + min(B, R))`.

## Request Flow

### Initialization

1. Layers are assigned to independent or shared physical slots.
2. KV cache tensor descriptors assigned to the same slot are merged.
3. The worker registers the resulting physical buffers with Memcache and
   adjusts the logical KV cache memory budget according to the slot reduction.

### Prefill Execution

For each physical layer:

1. The worker submits the layer's H2D load before attention reaches that layer.
2. If the layer reuses a buffer, the load waits for the previous owner's D2H
   save to complete.
3. Attention waits for the layer load, writes the newly computed KV cache, and
   opens the gate for a future prefetched layer.
4. After the layer's KV scatter is complete, the worker dispatches its D2H
   save.
5. Attention computation continues while the save and later prefetches run on
   the transfer thread.

A reused layer reloads its complete cached prefix because its physical buffer
may have been overwritten by the previous owner. An independent layer can
retain its dedicated device contents.

## Verification

The following log messages indicate that shared-buffer offload is active:

```text
Layerwise KV cache reuse merged ... tensor descriptors into ... shared slots.
Layerwise KV cache reuse uses ... slots for ... layers; scale logical KV budget by ...
```

If the first message is absent, check that:

- `backend` is `"memcache"`;
- `use_layerwise` is `true`;
- `layerwise_num_shared_buffers` is smaller than the number of reusable
  layers;
- the model exposes one uniform KV cache tensor per base transformer layer.

## Limitations

- A dedicated Decode consumer must set `consumer_is_to_put: true` so that
  evolving Decode KV, including partial blocks, can be saved before its shared
  NPU buffer is reused.
- Only the Memcache backend supports this shared-buffer layerwise path.
- TP-size mismatch is not supported with layerwise KV transfer.
- Context-parallel attention backends do not yet provide the required
  layerwise wait/save integration.
- MTP, multiple KV cache groups, and non-uniform or compressed KV layouts are
  not supported by the base layer-reuse implementation.
