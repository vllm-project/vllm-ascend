# Mooncake Layerwise Hybrid Attention

## Feature Introduction

Mooncake layerwise hybrid attention extends KV cache pooling to models with
multiple cache groups. It saves and restores each group's cache layer by layer,
so that a request can reuse a prefix together with the window, compressed cache,
or recurrent state needed to continue computation.

This feature uses `AscendStoreConnector` with `backend="mooncake"` and
`use_layerwise=true`. Cache groups are detected from the model's runtime cache
configuration; no separate hybrid-pooling switch is required.

For the basic transfer mechanism, Mooncake preparation, and common parameters,
see [Mooncake Layerwise KV Cache Pooling](mooncake_layerwise_adaptation_and_optimization.md).

### Usage Scenarios

- Models combining full attention and sliding-window attention.
- Models with compressed KV, indexer caches, and compressor state, such as DeepSeek-V4.
- Models combining attention with linear or recurrent layers that use aligned state checkpoints.

### Constraints and Limitations

The common Mooncake dependency and deployment requirements are listed in the
[layerwise pooling guide](mooncake_layerwise_adaptation_and_optimization.md#constraints-and-limitations).

| Category | Constraint |
| --- | --- |
| Attention backend | Compute/transfer overlap is integrated with Ascend FA, SFA, and DSA paths. Other attention backends need their own integration. |
| Execution mode | Eager and graph-mode compatibility follows the selected model and attention backend. |
| Recurrent state | Recurrent and linear-attention groups, including GDN and KDA state, require `mamba_cache_mode="align"`. |
| Parallelism | Use matching TP, PP, PCP, and PP layer partitions on sharing instances. DCP greater than `1` and TP mismatch are not supported. |
| PP | Uneven partitions and stage-local groups are supported only when every stage owns layers in every required group. Empty stage-local groups are not supported. |
| PCP | Replicated cache groups share Store objects and only one PCP rank saves them. PCP-sharded Mamba groups retain a distinct state shard per PCP rank. |
| KVPP | Hybrid layerwise pooling does not support KVPP. Mooncake layerwise PCP also cannot be combined with `enable_kvpp`. |
| Cache storage | Partial-block offloading and cross-layer cache-buffer reuse are not supported. |
| Load failures | Hybrid models do not support `kv_load_failure_policy=recompute`. A load failure must not continue with incomplete state. |

Model-specific quantization, graph mode, and speculative-decoding support should
be checked in the corresponding model deployment guide.

## How It Works

### Independent Cache Groups

A hybrid model can use different block sizes and cache entries in different
groups. Mooncake stores one object per group, logical block, and storing
TP rank or head within each pipeline stage. PCP-sharded Mamba groups also
include the PCP size and rank in their object keys. Replicated groups share
objects across PCP ranks. Each object contains that group's layer data;
range transfers read or write only the entries needed by the current layer.

For example, a model may have the following layout:

| Cache group | Tokens per block | Layers with cache entries | Object committed after |
| --- | --- | --- | --- |
| Full attention | 16 | 0, 2 | Layer 2 |
| Compressed KV | 32 | 1, 2, 3 | Layer 3 |
| Window/state | 16 | 0, 3 | Layer 3 |

The sizes and layer numbers above are illustrative. Groups can have different
commit boundaries, and one physical layer can transfer entries from several
groups. Object identity includes the group and a cache-layout fingerprint to
separate incompatible group layouts. Different model weights still require
separate pool namespaces even when model names and cache layouts match.

### Consistent Prefix Recovery

Reusing a hybrid prefix requires more than finding full-attention KV. The
request must also recover the window or state checkpoint needed to continue at
the same token boundary.

The cache coordinator selects the blocks required by each group and finds a
common reusable token boundary. A hit is reported only when all required
objects are committed and readable across the participating ranks and stages.
Missing window or recurrent state can therefore shorten the reusable prefix
even when full-attention KV is available.

Only complete snapshots aligned to this common boundary are published. Sparse
state checkpoints are restored according to their group-specific masks rather
than treated as a contiguous sequence of KV blocks. Later chunked-prefill steps
can restore committed data using their current local block allocations.

### Transfer and Cache Updates

Future-layer loads can overlap the current attention computation. Device events
ensure that transfers access cache data after the relevant updates and that
attention waits for its required loads.

Full and sliding-window attention can submit saves once their cache updates are
ready at attention entry. Recurrent layers update their state during computation,
so their saves start after those updates finish. Each group commits independently
after its last participating layer. Failed writes do not publish incomplete
objects, and failed reads stop computation before incomplete hybrid state is used.

## Usage

### Configure Prefill and Decode Nodes

Prepare the Mooncake Store and client as described in the
[layerwise pooling guide](mooncake_layerwise_adaptation_and_optimization.md#prepare-mooncake),
then follow the **PD Disaggregation Scenario** in the
[Mooncake deployment guide](kv_cache_pool_ascend_store.md#scenario-1-mooncake-backend).
Use these Store connector configurations for hybrid models.

**Prefill (P) node:**

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_producer",
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "0",
        "backend": "mooncake",
        "use_layerwise": true,
        "layerwise_prefetch_layers": 2
    }
}
```

**Decode (D) node:**

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_consumer",
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "0",
        "backend": "mooncake",
        "use_layerwise": true,
        "layerwise_prefetch_layers": 2
    }
}
```

Hybrid models require `fail`; `recompute` is not supported. When using
`MultiConnector`, set this policy on the top-level `kv-transfer-config` and
update its `AscendStoreConnector` entry with the role and extra configuration
shown above. Decode pool access is disabled by default; enable
`consumer_is_to_load` explicitly if Decode needs to load from the pool. See
[Configure Prefill and Decode Nodes](mooncake_layerwise_adaptation_and_optimization.md#configure-prefill-and-decode-nodes)
for role behavior and IPC identifier requirements.

Choose model, parallelism, token-budget, and execution-mode options according
to the model deployment guide. For recurrent or linear-attention state,
set `--mamba-cache-mode align` unless the model configuration already selects it.
Use the same model weights and cache configuration on all sharing instances.

Inference requests use the normal OpenAI-compatible API. To exercise remote
reuse, send a long shared prefix to two instances connected to the same pool,
as described in [Use Cached Prefixes](mooncake_layerwise_adaptation_and_optimization.md#use-cached-prefixes).
Reuse advances only to a boundary where all required cache groups are available.

### Prefill Context Parallelism (PCP)

To use PCP, add these options to the model's launch command:

```bash
--prefill-context-parallel-size 4 \
--decode-context-parallel-size 1
```

Use matching TP and PP layouts on instances sharing these objects.
Replicated cache groups share object keys across PCP ranks, and
only one PCP rank saves them. Other PCP ranks restore from the same objects.
PCP-sharded Mamba groups store distinct state shards rather than complete
replicas. Their keys include both PCP size and rank, and prefix lookup requires
committed shards from every required PCP rank. Lookup also checks every required
PP stage and storing TP rank or head. PCP does not change the logical token
coverage of a group block.

Size the Store for unique cache objects and state shards. Replicated groups do
not require one stored copy per PCP rank. Keep `enable_kvpp` disabled.

## Configuration Parameters

Hybrid pooling uses the same connector parameters as
[layerwise pooling](mooncake_layerwise_adaptation_and_optimization.md#configuration-parameters).
Cache-group membership, block sizes, and transfer offsets are derived from the
model's cache configuration.

| Parameter | Requirement |
| --- | --- |
| `kv_connector_extra_config.backend` | Set to `"mooncake"`. |
| `kv_connector_extra_config.use_layerwise` | Set to `true`. |
| `kv_connector_extra_config.layerwise_prefetch_layers` | Positive integer, default `2`; controls future-layer load distance. |
| `mamba_cache_mode` | Use `"align"` for models with recurrent-state groups. |
| `decode_context_parallel_size` | Keep at `1`. |
| `prefill_context_parallel_size` | Default `1`; sharing instances must use the same value. |

## Tuning Suggestions

- Start with a prefetch distance of `2` and adjust it to the model's attention
  time and transfer bandwidth. A larger distance can hide loads but can also
  increase contention with HCCL or expert-parallel communication.
- Size the Store for unique objects across all cache groups, including indexer
  caches and recurrent-state shards. Count replicated objects once per storing
  TP rank or head and PP stage, rather than once per PCP rank. Full-attention KV
  size alone is insufficient for capacity planning.
- Use representative prompt lengths and shared prefixes when measuring reuse.
  Different group block sizes and checkpoint boundaries determine how much of
  a prefix can be restored.
