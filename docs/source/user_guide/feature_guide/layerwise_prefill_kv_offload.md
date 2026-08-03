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
- compatible KV cache specifications and tensor sizes for layers that share a
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
| `layerwise_num_shared_buffers` | Number of cache-bearing layers | Number of reusable buffers assigned to non-independent layers. If omitted, no cross-layer buffer reuse is enabled. The value must be at least `1`. |
| `layerwise_independent_layers` | `[0]` | Cache-bearing layers that keep dedicated buffers. Accepts a list of integers or `"all"`. Negative indices are resolved against all cache-bearing layers, including MTP layers. |

## Buffer Layout

For a uniform KV cache layout, let:

- `N` be the number of physical layers, including MTP layers;
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

When cache-bearing layers have different KV cache layouts, the planner first
groups them by their complete cache signature. Each signature gets its own
reusable buffer pool, so incompatible layouts never share storage. The number
of physical KV buffers is then:

```text
I + sum(min(B, reusable layers with signature S) for each signature S)
```

For a uniform layout, the approximate logical-to-physical memory factor remains
`N / (I + min(B, R))`. For heterogeneous layouts, the implementation calculates
the factor from the actual cache page bytes assigned to every physical buffer.

## Request Flow

### Initialization

1. The KV cache planner maps logical layer names to cache-bearing layer indices.
2. Base transformer layers keep their normal indices. MTP layers are appended
   after the base layers.
3. The planner builds the complete KV cache signature of each cache-bearing
   layer.
4. Compatible layers are assigned to dedicated or shared physical KV buffers.
5. Corresponding KV cache tensor descriptors assigned to the same buffer are
   merged.
6. The worker registers the resulting physical buffers with Memcache and
   adjusts the logical KV cache memory budget according to the bytes saved.

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

## MTP and Sparse C8 Layouts

### MTP

MTP layers use names such as `mtp.0.self_attn` and are placed after the base
transformer layers in the physical layout. They participate in buffer
assignment, transfer event allocation, and memory accounting.

Because negative independent-layer indices use the complete physical layout,
`-1` refers to the last MTP layer when MTP is enabled, not the last base
transformer layer.

### Sparse C8

An SFA layer can contain separate cache entries:

- the main MLA/SFA KV cache;
- the sparse indexer cache;
- their corresponding C8 scale data when enabled.

The planner does not identify these entries by model-specific names. It builds
the physical layer's signature from their actual KV cache specifications and
only reuses a buffer when the complete signature matches. Transfer addresses and
allocation sizes are derived from each group's real per-layer offsets and page
bytes. This allows MTP and sparse C8 layouts to use the same layerwise offload
pipeline without assuming that every layer has one tensor.

An incompatible layout receives a separate buffer pool instead of being
transferred through an incorrectly sized buffer.

## Verification

The following log messages indicate that shared-buffer offload is active:

```text
Layerwise KV cache reuse merged ... descriptors into ... descriptors using ... buffer assignments.
Layerwise KV cache reuse maps ... layers onto ... buffer assignments; scale logical KV budget by ...
```

If the first message is absent, check that:

- `backend` is `"memcache"`;
- `use_layerwise` is `true`;
- `layerwise_num_shared_buffers` is smaller than the number of reusable
  layers;
- all expected base and MTP layer cache specifications are present.

## Limitations

- Only the Memcache backend supports this shared-buffer layerwise path.
- TP-size mismatch is not supported with layerwise KV transfer.
- Context-parallel configurations have not been validated with shared-buffer
  layerwise offload.
- Multiple non-packed attention KV cache groups are supported. State-cache
  groups and packed or pre-shared KV cache tensor descriptors are not
  supported by shared-buffer reuse.
