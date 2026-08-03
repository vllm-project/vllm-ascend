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
Prefill. The same correctness path keeps PD-Mixed inference and Decode
functionally compatible, but they are not target deployment scenarios for this
feature.

## Requirements

Layerwise Prefill offload currently requires:

- `AscendStoreConnector`;
- `kv_role: "kv_producer"` on a dedicated Prefill node;
- `backend: "memcache"`;
- `use_layerwise: true`;
- an MLA, SFA, or DSA attention backend with the layerwise wait/save
  integration;
- identical KV cache tensor sizes and cache specs for layers that share a
  buffer;
- eager execution; graph mode is not currently supported.

`kv_role: "kv_both"` is retained only for functional compatibility and should
not be used with layerwise Prefill offload. During PD-Mixed inference, Decode
must load and save the evolving KV cache through the reused buffers at every
decoding step. The resulting layerwise transfer and synchronization overhead
causes severe Decode performance degradation. Deploy this feature on a
dedicated Prefill node with `kv_role: "kv_producer"`.

## Configuration

Configure the dedicated Prefill node as follows:

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "use_layerwise": true,
        "layerwise_num_shared_buffers": 3,
        "layerwise_independent_layers": [0]
    }
}
```

The example keeps the first transformer layer independent and assigns all
other layers to three reusable buffers. The values are examples rather than
universal recommendations; choose them according to available NPU memory and
transfer bandwidth.

### Core parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `use_layerwise` | `false` | Enables layer-by-layer KV transfer. |
| `backend` | `"mooncake"` | Must be `"memcache"` for shared-buffer layerwise offload. |
| `layerwise_num_shared_buffers` | Number of base transformer layers | Number of reusable buffers assigned to non-independent layers. If omitted, no cross-layer buffer reuse is enabled. The value must be at least `1`. |
| `layerwise_independent_layers` | `[0]` | Base transformer layers that keep dedicated buffers. Accepts a list of integers or `"all"`. Negative indices are resolved against the base transformer layers. |

## Buffer Layout

Let:

- `N` be the number of base transformer layers;
- `I` be the number of independent layers;
- `R = N - I` be the number of reusable layers;
- `B` be `layerwise_num_shared_buffers`.

The number of physical KV buffers is:

```text
I + min(B, R)
```

Cross-layer reuse is active only when `R > B`. Reusable layers are assigned to
the `B` buffers in round-robin order.

For example, with 27 transformer layers, independent layer `[0]`, and three
shared buffers:

```text
dedicated buffer: [0]
shared buffer 0:  [1, 4, 7, 10, 13, 16, 19, 22, 25]
shared buffer 1:  [2, 5, 8, 11, 14, 17, 20, 23, 26]
shared buffer 2:  [3, 6, 9, 12, 15, 18, 21, 24]
```

Layer 4 reuses layer 1's physical buffer, layer 7 reuses layer 4's buffer, and
so on. Before loading layer 4, the transfer thread waits until layer 1 has
finished saving.

For the supported uniform KV layout, the approximate logical-to-physical
memory factor is `N / (I + min(B, R))`.

## Request Flow

### Initialization

1. Layers are assigned to dedicated or shared physical KV buffers.
2. KV cache tensor descriptors assigned to the same buffer are merged.
3. The worker registers the resulting physical buffers with Memcache and
   adjusts the logical KV cache memory budget according to the reduction in
   allocated buffers.

### Prefill Execution

During each Prefill step:

1. If a request has cached KV to restore, the worker submits the required H2D
   load before attention reaches each transformer layer.
2. An independent layer loads only the cached portion that is not already
   retained in HBM.
3. A reused layer reloads its complete cached prefix because its physical
   buffer may have been overwritten by the previous owner.
4. Before a reused buffer is overwritten, its load waits for the previous
   owner's D2H save to complete.
5. Attention waits for any required layer load, writes the newly computed KV
   cache, and opens the gate for a future prefetched layer.
6. After the layer's KV scatter is complete, the worker dispatches its D2H
   save. Attention computation continues while this save and later prefetches
   run on the transfer thread.

## Verification

The following log messages indicate that shared-buffer offload is active:

```text
Layerwise KV cache reuse merged ... tensor descriptors into ... shared buffers.
Layerwise KV cache reuse uses ... buffers for ... layers; scale logical KV budget by ...
```

If the first message is absent, check that:

- `backend` is `"memcache"`;
- `use_layerwise` is `true`;
- `layerwise_num_shared_buffers` is smaller than the number of reusable
  layers;
- the model exposes one uniform KV cache tensor per base transformer layer.

## Limitations

- Only the Memcache backend supports this shared-buffer layerwise path.
- TP-size mismatch is not supported with layerwise KV transfer.
- Context-parallel configurations have not been validated with shared-buffer
  layerwise offload.
- MTP, multiple KV cache groups, and non-uniform or compressed KV layouts are
  not supported by the base layer-reuse implementation.
