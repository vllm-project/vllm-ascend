# Layerwise and Sparse KV Cache Offloading Design

This document explains why Prefill and Decode use different KV cache offload
strategies, how the two strategies work, and how they preserve correctness when
used together.

For installation and configuration, see
[Layerwise and Sparse KV Cache Offloading Guide](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md).

The design is based on
[RFC #48203](https://github.com/vllm-project/vllm/issues/48203).

## 1. Motivation

KV cache is a major NPU memory cost in long-sequence inference. Moving it to
host memory increases capacity, but Prefill and Decode cannot use the same
transfer strategy efficiently.

| Stage | Compute pattern | Offload strategy | NPU-resident data |
| :--- | :--- | :--- | :--- |
| Prefill | High computation per layer | Transfer complete layers and overlap transfer with computation | A few reusable layer buffers |
| Decode | Low computation per token | Keep full KV in host memory and load only selected entries | Indexer cache and a hot top-k buffer |

Loading a complete layer during every Decode step would add transfer latency
that cannot be hidden by Decode computation. Sparse Decode Offload avoids that
cost by moving only the entries selected by sparse attention.

## 2. System Overview

The combined design has four storage regions:

- reusable layer buffers in Prefill NPU memory;
- the Memcache-backed Prefill host KV pool;
- the full main KV cache in Decode host memory; and
- the indexer cache and hot top-k buffer in Decode NPU memory.

```mermaid
flowchart LR
    PHost["Prefill host KV pool<br/>Memcache"]
    PNPU["Prefill NPU<br/>reusable layer buffers"]
    DHost["Decode host<br/>full main KV cache"]
    DNPU["Decode NPU<br/>indexer + hot top-k buffer"]

    PHost <-->|"Layerwise load / save"| PNPU
    PNPU -->|"Remote D2H pull<br/>main KV"| DHost
    PNPU -->|"Remote pull<br/>indexer / LIC8 scale"| DNPU
    DHost -->|"Top-k misses"| DNPU
```

`AscendStoreConnector` manages Layerwise Prefill Offload through Memcache.
`SfaRemoteD2HConnector` exposes Prefill NPU buffers and lets Decode pull main KV
into Decode-owned host memory and indexer data into rank-local NPU memory
through MemFabric. Optional LIC8 scale data follows the indexer destination.

## 3. Layerwise Prefill Offload

### Buffer planning

Layerwise Prefill Offload maps many logical cache-bearing layers onto fewer
physical NPU buffers. Some layers can keep dedicated buffers; the remaining
compatible layers share a reusable buffer pool.

For a uniform cache layout:

```text
N = number of cache-bearing layers
I = number of independent layers
R = N - I
B = configured shared-buffer count

physical buffers = I + min(B, R)
```

Reusable layers are assigned to the shared buffers in round-robin order.
Layers with incompatible main KV specifications do not share a physical
buffer. MTP layers participate in the same planning. Optional indexer caches
follow their main buffer assignment and are allocated only where needed.

Before tensor descriptors are merged, the planner validates cache
specifications, tensor sizes, and optional indexer layouts. An incompatible
layout fails initialization instead of sharing incorrectly sized storage.

### Execution pipeline

For each layer:

1. load the cached prefix from Memcache when required;
2. wait until the target physical buffer is safe to overwrite;
3. run attention and update KV;
4. save the updated KV to host memory; and
5. prefetch a later layer while computation continues.

Multiple physical buffers let transfer and computation overlap. Prefill has
enough per-layer computation to hide much of the load and save latency, which
is why full-layer transfer is suitable for this stage.

### Reuse invariant

A physical buffer cannot be reused until every consumer of its previous
contents has finished. Its Memcache save must complete before reuse. In the
joint deployment, the Decode-side remote read must also complete.

The implementation tracks completion by physical storage slot rather than only
by logical layer name. This matters because several logical layers can refer to
the same NPU address.

## 4. Sparse Decode Offload

### Memory layout

Sparse Decode Offload keeps the full main KV cache in a pinned Decode host
pool. Decode NPU memory contains:

- the rank-local indexer cache used to select important tokens; and
- a hot buffer containing recently used top-k main KV entries.

Each Decode DP rank owns a separate main host pool. Within one DP rank, its TP
ranks share that pool, while every TP rank owns the indexer data required by its
local attention computation. Control-plane ports are assigned as
`kv_port + dp_rank * d_tp_size + tp_rank`, keeping DP and TP endpoints disjoint.

### Decode step

For each Decode step:

1. the indexer selects the required top-k token positions;
2. entries already present in the hot buffer are reused;
3. only cache misses are copied from host memory to the NPU;
4. sparse attention consumes the resulting top-k KV; and
5. residency metadata is updated for the next step.

The full history remains available in host memory, but transfer volume is
proportional to top-k misses rather than total sequence length. This makes
offload practical for the low-compute Decode stage.

## 5. Joint Prefill and Decode Design

### Connector composition

Prefill uses `MultiConnector` to compose the Remote D2H completion provider and
`AscendStoreConnector`. `AscendMultiConnector` invokes the completion provider
first regardless of their order in the connector configuration. For every
layer buffer:

- AscendStore saves the layer to the Prefill host KV pool; and
- Remote D2H publishes source metadata and waits for Decode acknowledgements.

The buffer-reuse gate opens only after both paths finish. This prevents Prefill
from overwriting a shared physical buffer while Decode is still reading it.

### Request flow

1. Decode allocates its main host destination and rank-local indexer
   destination.
2. Decode advertises its endpoint and parallel topology through the proxy
   metaserver.
3. Prefill computes the request layer by layer.
4. After each layer's KV write completes, Prefill starts the Memcache save and
   publishes Remote D2H readiness.
5. Decode pulls the layer into its destinations and acknowledges completion.
6. Prefill releases the physical buffer for reuse after all required
   completions arrive.
7. Decode schedules the request only after the full request reaches a terminal
   transfer state across all required ranks.

Physical-buffer completion and request completion are separate:

- physical-buffer completion protects Prefill memory reuse; and
- request completion controls when Decode may begin inference.

## 6. Unequal Tensor Parallelism

The supported joint topology is:

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

The Prefill ranks mapped to one Decode rank form a contributor group.

- Main KV is replicated on Prefill, so only contributor member `0` transfers
  the main share owned by that Decode rank.
- Contributors transfer disjoint indexer ranges whose union forms the complete
  rank-local Decode indexer.
- A contributor with no blocks still acknowledges the layer so completion
  cannot deadlock.

## 7. Validation and Failure Handling

Before a Remote D2H read, Decode validates main tensor count and sizes, indexer
presence and sizes, optional LIC8 scale layout, and destination block ranges.
A mismatch fails the layer instead of leaving partially valid destination
data.

Every readiness notification reaches a terminal success or failure state. A
failed MemFabric read invalidates the Decode destination and releases the
Prefill-side waiter with an error, preventing silent corruption or a buffer
reuse deadlock.

## 8. Current Boundaries

- Layerwise shared-buffer offload requires the Memcache backend and eager mode.
- Sparse Decode Offload supports DP and TP; CP and PP are not supported.
- Joint deployment requires Prefill TP to be greater than or equal to, and
  divisible by, Decode TP.
- MemFabric is the only supported Remote D2H transfer backend.
- Layerwise buffer reuse cannot currently be combined with
  `MooncakeLayerwiseConnector` because it does not provide a per-buffer transfer
  completion gate. Support is planned in a follow-up update.
- Connector-level data-read retry is not implemented.
