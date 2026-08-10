# SFA PD RD2H Connector Design

## Purpose and Scope

`SFAPDRD2HConnector` transfers SFA KV cache from a remote Prefill worker into
the Decode-side layout owned by `SparseKVOffloadManager`. It is a KV Connector
V1 implementation for Prefill-Decode disaggregation with sparse Decode
offload.

RD2H means **remote device to local host**. The connector uses a pull model:

- Prefill registers and exposes its NPU KV cache.
- Decode owns the destinations and pulls data through MemFabric.
- Prefill publishes source metadata and per-layer readiness over ZMQ.
- Decode acknowledges every layer after the read succeeds or fails.

The connector provides transport and completion signaling. The Decode top-k
resident-cache algorithm remains in `SparseKVOffloadManager` and the SFA
offload attention backend. AscendStore Layerwise Prefill KV Cache Offload is an
optional, separately configured feature.

For user-facing configuration and deployment constraints, see
[SFA PD RD2H KV Transfer](../../user_guide/feature_guide/sfa_pd_rd2h_kv_transfer.md).

## Data Ownership

The Decode destination contains main KV and indexer KV with different
ownership rules.

| Component | Prefill source | Decode destination |
| :--- | :--- | :--- |
| Main K/V | Regular paged NPU cache, replicated across Prefill TP ranks | TP-shared pinned CPU pool owned by `SparseKVOffloadManager` |
| Indexer | Rank-local NPU cache | Complete rank-local NPU copy on every Decode TP rank |
| LIC8 scale | Optional tensor following the indexer layout | Optional rank-local NPU tensor |

Each Decode TP rank transfers a disjoint share of main KV into the shared CPU
pool. Main and indexer block IDs remain separate in scheduler metadata and wire
messages. A uniform KV-cache group may map them to the same vLLM block group,
but the connector does not depend on group ordering.

## Request Lifecycle

### Decode allocation and rendezvous

For a request with `do_remote_prefill=true`, Decode reports the remaining
prompt tokens as asynchronously matched and lets vLLM allocate the main and
indexer destinations. It records those block IDs locally, advertises its host,
base port, parallel topology, and cached-token count through the proxy
metaserver, and then clears `do_remote_prefill`.

Decode block IDs never leave Decode. Prefill only receives the endpoint and
cached-token information needed to publish its sources.

### Prefill scheduling

The proxy dispatches the request to Prefill with `do_remote_decode=true`.
Prefill tracks its source block IDs, the Decode endpoint, computed and
transferred token counts, and chunk completion.

For chunked Prefill, a metadata update covers newly completed blocks. The final
chunk also includes its partial block. OpenAI prompt lists are expanded into
child requests on Decode; the proxy collects all child rendezvous metadata and
dispatches the original list to Prefill once.

### Layer transfer

After a layer's KV scatter completes, Prefill records an NPU event and queues a
send task. The background sender waits for the event, publishes MemFabric
metadata once per connection, and sends a readiness message.

Decode validates the source and destination layouts, builds descriptors across
the ready requests, and executes one synchronous batched MemFabric read. It
then replies with success or failure for that layer.

## Control Protocol

Prefill uses ZMQ DEALER sockets. Each Decode TP rank owns one ZMQ ROUTER. Wire
messages are positional msgpack tuples.

### MemFabric metadata

Prefill sends this once per connection before any readiness message:

```text
(MF_META, p_session, encoded_layer_metadata)
```

Layer metadata contains tensor group IDs, base addresses, block byte lengths,
block-size scales, the number of main tensors, and indexer presence. Decode
stores it by ZMQ identity and replies with `ACK`.

### Layer readiness

Prefill sends one message per layer and Decode endpoint:

```text
(
    READ_READY_BATCH,
    layer_idx,
    layer_name,
    read_reqs,
    done_ext_ids,
    group_member_idx,
    tp_ratio,
)
```

Each `read_reqs` entry contains the external request ID, Prefill main and
indexer block IDs, and the main and indexer destination start offsets.

`done_ext_ids` identifies requests whose final chunk reached the last
cache-bearing layer.

### Completion reply

```text
(READ_DONE, layer_idx)
(READ_FAILED, layer_idx, error)
```

Protocol extensions append tuple fields. Receivers use tuple-length guards so
missing unequal-TP fields retain legacy `ratio=1` behavior.

## Correctness Invariants

### Physical-storage completion

Layerwise Prefill may reuse one physical buffer for multiple logical layers.
Before a buffer is overwritten, every Decode endpoint that received readiness
for the previous owner must acknowledge its read.

Prefill therefore maintains one completion event per physical main or indexer
storage slot, inferred from storage addresses rather than layer names. Main and
indexer slots are tracked independently. Every readiness message receives a
reply, including when a Decode rank owns zero blocks, so buffer reuse cannot
deadlock on a missing no-op acknowledgement.

### Request completion

Decode may schedule a request only after:

1. every Prefill contributor mapped to that Decode rank reports final-layer
   completion; and
2. every Decode TP rank reaches a terminal success or failure state.

Contributor state is keyed by external request ID. The worker gathers terminal
state through the TP CPU group and maps the external ID back to the internal
vLLM request ID.

Physical-storage completion and request completion remain independent. The
first protects Prefill memory reuse; the second controls Decode scheduling.

### Layout validation

Before reading, Decode validates main tensor count and byte lengths, indexer
presence and byte lengths, LIC8 scale presence and layout, and destination
block ranges. Main, indexer, or scale mismatches fail the layer instead of
leaving partially updated destination data.

## Unequal Prefill and Decode TP

The supported topology is:

```text
p_tp_size >= d_tp_size
p_tp_size % d_tp_size == 0
ratio = p_tp_size // d_tp_size
```

For Prefill rank `p_rank`:

```text
d_rank = p_rank // ratio
group_member_idx = p_rank % ratio
```

The `ratio` Prefill ranks mapped to one Decode rank form a contributor group:

- Main KV is replicated on Prefill. Only contributor member `0` pulls that
  Decode rank's main share, preventing duplicate writes and excess bandwidth.
- Each Decode rank needs a complete indexer. Contributors transfer disjoint
  indexer ranges whose union is the complete copy. LIC8 scale follows the same
  range.
- A contributor with no blocks still acknowledges the layer and contributes
  to request completion.
- Duplicate final-layer signals from one contributor do not advance
  completion.

## AscendMultiConnector Integration

When RD2H is composed with AscendStore Layerwise Prefill offload, the RD2H
completion provider runs first for:

- `wait_for_layer_load`;
- `save_kv_layer`; and
- `on_kv_cache_written`.

This ordering closes the RD2H storage-reuse gate before AscendStore publishes
or reuses the same physical buffer. RD2H exposes `wait_for_layer_send` as the
waiter used by the layerwise Prefill path.

On the scheduler side, RD2H must observe complete block groups even when
another connector is selected for lookup. The
`requires_full_blocks_on_update_after_alloc` flag preserves that behavior.

## Ports and Endpoint Identity

Only Decode binds connector control sockets:

```text
base_port = kv_port + global_dp_rank * tp_size
rank_port = base_port + tp_rank
```

The global DP rank, rather than a host-local rank, prevents port reuse across
engines. Decode advertises the base port through the metaserver; Prefill adds
the mapped Decode TP rank. Both the highest Decode TP port and the final mapped
remote port are validated.

## Failure and Compatibility

- Metaserver transport errors are retried. HTTP error responses are treated as
  delivered for compatibility with older proxy behavior.
- A non-zero MemFabric read result becomes `READ_FAILED`. Decode marks the
  request failed and its destination blocks invalid.
- Prefill records the layer error and releases the affected storage events so
  the compute path raises instead of deadlocking.
- Data reads are not retried at the connector layer.
- Prefill and Decode must use compatible protocol and cache-layout versions.
  Mixing legacy and unequal-TP contributor builds can duplicate writes.
- The process-wide MemFabric engine is bound to one role, device, and hostname.

## Implementation and Tests

The implementation is under
`vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/`. Its primary tests
are `test_sfa_pd_rd2h_connector.py`, `test_memfabric_transfer_engine.py`, and
`test_ascend_multi_connector.py` under `tests/ut/kv_offload/`.

Use `tools/test_memfabric_pd_read.py` to validate real two-endpoint MemFabric
registration and reads independently of vLLM scheduling.

## Current Limitations

- MemFabric is the only supported transfer backend.
- Decode sparse offload supports TP but not CP or PP.
- Prefill TP must be greater than or equal to, and divisible by, Decode TP.
- Connector-level data-read retry is not implemented.
