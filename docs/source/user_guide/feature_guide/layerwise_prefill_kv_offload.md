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

When this feature is combined with
[Sparse KV Cache Offload](sparse_kv_cache_offload.md), compose
`AscendStoreConnector` and `SFAPDRD2HConnector` through `MultiConnector` on the
Prefill node. The RD2H connector's per-layer completion signal prevents a
shared Prefill buffer from being reused before Decode has finished reading it.

## Configuration

To configure only the layerwise Prefill offload connector:

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

### Use with SFA PD RD2H

When Decode uses Sparse KV Cache Offload, Prefill must both save KV through
`AscendStoreConnector` and expose the same layer buffers through
`SFAPDRD2HConnector`. Compose them with `MultiConnector` on Prefill:

```json
{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "SFAPDRD2HConnector",
                "kv_role": "kv_producer",
                "kv_port": 20050,
                "kv_connector_extra_config": {
                    "transfer_backend": "memfabric"
                }
            },
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
        ]
    }
}
```

Do not enable `sparse_kv_offload_config` on Prefill. `kv_port` defines the
Decode-side control-port base. Prefill does not bind these ports; its control
target is provided by Decode through request metadata. The Prefill
configuration may use the same value for consistency. For Decode
configuration, TP constraints, and proxy setup, see
[SFA PD RD2H KV Transfer](sfa_pd_rd2h_kv_transfer.md).

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

When cache-bearing layers have different KV cache layouts, the planner groups
them by their **main** attention cache spec (not the whole-layer signature).
Layers whose main specs match share one reusable buffer pool: within each
physical buffer the main KV tensor is shared by every layer in the buffer,
while an auxiliary component (for example the SFA indexer cache) is shared only
by the subset of layers that actually own it. A buffer that holds only main-only
layers gets **no** auxiliary allocation at all — the auxiliary component is
provisioned on demand, per buffer, instead of unconditionally for every buffer.
The number of main buffer assignments is then:

```text
I + sum(min(B, reusable layers with main spec S) for each main spec S)
```

For a uniform layout, the approximate logical-to-physical memory factor remains
`N / (I + min(B, R))`. For heterogeneous layouts, the implementation calculates
the factor from the actual cache page bytes assigned to every physical buffer
(main counted once per buffer, and the optional indexer counted once per buffer
that owns it).

## Request Flow

### Initialization

1. The KV cache planner maps logical layer names to cache-bearing layer indices.
2. Base transformer layers keep their normal indices. MTP layers are appended
   after the base layers.
3. The planner groups cache-bearing layers by their main KV cache spec.
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
- the optional sparse indexer cache, whose name ends with
  `.indexer.k_cache`.

The planner groups reusable layers by their main cache spec. The optional
indexer follows the main buffer assignment and is allocated only for buffers
that contain an indexer-bearing layer. C8 scale storage, when enabled, is part
of the corresponding cache spec and physical allocation.

Before tensor descriptors are merged, the implementation verifies that every
main cache sharing a buffer has the same spec and size, and performs the same
check for indexer caches. Unsupported multi-entry layouts or incompatible
descriptors raise an error instead of sharing an incorrectly sized buffer.
Transfer addresses and allocation sizes are derived from each group's actual
per-layer offsets and page bytes.

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
- TP-size mismatch is not supported when `AscendStoreConnector` itself performs
  layerwise P/D KV transfer. This limitation does not apply when
  `SFAPDRD2HConnector` performs the P/D transfer and `AscendStoreConnector` is
  used only for Prefill-side layerwise offload.
- Context-parallel configurations have not been validated with shared-buffer
  layerwise offload.
- Multiple non-packed attention KV cache groups are supported. State-cache
  groups and packed or pre-shared KV cache tensor descriptors are not
  supported by shared-buffer reuse.
