# Mooncake Layerwise KV Cache Pooling

## Feature Introduction

Mooncake layerwise KV cache pooling transfers KV caches between Ascend NPUs and
a shared Mooncake Store one layer at a time. It uses `AscendStoreConnector` with
`backend="mooncake"` and `use_layerwise=true` to reuse cached prefixes across
requests and vLLM instances.

Layerwise transfer allows KV loading and saving to overlap attention computation.
It also stores the layers of a cache block in one remote object, reducing
metadata overhead compared with creating a separate object for every layer.

For Mooncake service deployment and general pool configuration, see
[KV Cache Pool (Ascend Store)](kv_cache_pool_ascend_store.md). For models with
multiple cache groups, see [Mooncake Layerwise Hybrid Attention](mooncake_hybrid_attention.md).

### Usage Scenarios

- Repeated requests with long shared prefixes, such as system prompts or document contexts.
- Multiple serving instances that share prefixes through the same Mooncake Store.
- Prefill workloads where KV transfer can overlap per-layer attention computation.

### Constraints and Limitations

| Category | Constraint |
| --- | --- |
| Software | Requires a Mooncake NPU client with session and range transfer APIs; use Mooncake 0.3.14 or later. Required interfaces are checked at startup. |
| Deployment | Use the corresponding producer/consumer role on each P/D node. Decode pool reads and writes are controlled separately by `consumer_is_to_load` and `consumer_is_to_put`. |
| Parallelism | Instances sharing layerwise objects must use matching TP, PP, and PCP sizes and the same PP layer partition. DCP greater than `1` and TP mismatch are not supported. |
| PP | Uneven layer partitions are supported, but each stage must own layers in every cache group. Empty stage-local cache groups are not supported. |
| Cache layout | Readers and writers must agree on model weights, cache format, block sizes, and transfer mode. Use separate pool namespaces for incompatible configurations. |
| Buffer reuse | Mooncake layerwise transfer does not enable cross-layer reuse of NPU cache buffers. |
| KVPP | Experimental for non-hybrid MLA/SFA models with `load_async=false`; cannot be combined with Mooncake layerwise PCP. See [KVPP](kvpp.md) for its support scope. |

Hybrid models have additional cache-state and execution-mode requirements in the
[hybrid attention guide](mooncake_hybrid_attention.md#constraints-and-limitations).

## How It Works

### Block Storage and Layerwise Transfer

For each cache block and storing rank, Mooncake keeps one object containing the
cache entries of all layers owned by that rank and pipeline stage:

```text
Remote object for one cache block
├── Layer 0: cache entries
├── Layer 1: cache entries
└── Layer 2: cache entries
```

Saving a layer writes only its byte ranges into this object. Loading a layer
reads those ranges into the local NPU cache before the layer consumes them.
Offsets follow the actual cache layout, including multiple entries in one
physical layer and entries with different sizes.

For a single cache group and pipeline stage, `B` blocks, `L` layers, and `R`
storing ranks require approximately `B × R` objects, compared with `B × L × R`
objects in a per-layer object layout. This reduces metadata and lookup work;
the total KV payload remains the same.

### Computation and Transfer Overlap

The connector prefetches future layers while the current layer computes
attention. `layerwise_prefetch_layers` controls how far ahead loads are submitted.
The current layer waits for its own load to finish before using restored KV.
Saves start after the corresponding cache updates are ready.

Transfers run asynchronously, with device events protecting cache access and
queue backpressure limiting outstanding work. Transfer may continue into later
computation or communication. The amount of overlap depends on attention time,
cache size, and available bandwidth.

### Complete Snapshots and Prefix Reuse

A remote object becomes reusable only after all of its layer ranges have been
written and committed. A prefix hit requires readable objects from every
required storing rank and pipeline stage. Partially written objects are not
reported as reusable cache.

During chunked prefill, later chunks can restore the committed prefix layer by
layer. Read sessions keep shared objects available while they are in use and
are released when the reads finish. Failed writes are revoked so that later
requests cannot consume an incomplete snapshot.

## Usage

### Prepare Mooncake

Install the Mooncake NPU client, start `mooncake_master`, and configure the
Ascend network and `mooncake.json` following the **Mooncake Backend** section of
[KV Cache Pool (Ascend Store)](kv_cache_pool_ascend_store.md). Select a client
version that meets the layerwise requirement above rather than the older
non-layerwise installation example.

Set the configuration path and use the same hash seed on all sharing instances:

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
export PYTHONHASHSEED=0
```

Sharing instances must connect to the same Store and tenant, with compatible
model and cache configurations. For independently launched services on the same
host, assign distinct `lookup_rpc_port` identifiers and HTTP ports.

### Configure Prefill and Decode Nodes

Follow the **PD Disaggregation Scenario** in the
[Mooncake deployment guide](kv_cache_pool_ascend_store.md#scenario-1-mooncake-backend)
for node startup, P/D transfer, and proxy configuration. Configure the
`AscendStoreConnector` on each node as follows.

**Prefill (P) node:**

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_producer",
    "kv_load_failure_policy": "recompute",
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
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "0",
        "backend": "mooncake",
        "use_layerwise": true,
        "layerwise_prefetch_layers": 2
    }
}
```

These examples show the Store connector configuration. In the deployment
guide's `MultiConnector`, update the `AscendStoreConnector` entry and set
`kv_load_failure_policy` on the top-level `kv-transfer-config`.

The Prefill node saves and reuses prefixes through the pool. Decode pool reads
and writes are disabled by default; P/D KV transfer is handled by the other
connector in `MultiConnector`. To enable Decode pool loading, add
`"consumer_is_to_load": true` to its `kv_connector_extra_config`. Setting
`use_layerwise=true` alone does not enable consumer pool access.

The `recompute` policy above applies to non-hybrid models. For hybrid models,
use `fail` as shown in the [hybrid attention guide](mooncake_hybrid_attention.md#usage).
Select model, parallelism, token-budget, and execution-mode options according
to the corresponding model deployment guide.

### Use Cached Prefixes

Send inference requests through the normal OpenAI-compatible API:

```bash
curl 'http://<proxy-host>:<proxy-port>/v1/completions' \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<served-model-name>",
    "prompt": "Explain how KV cache reuse reduces repeated prefill computation.",
    "max_tokens": 64,
    "temperature": 0
  }'
```

The response contains generated text in `choices`. To reuse the pool, submit
requests with the same tokenized prefix long enough to cover complete cache
blocks. The first request populates the Store; later requests can load committed
blocks and compute the remaining tokens. A final partial block is recomputed.

To check cross-instance reuse, send a long shared prefix to one instance, wait
for its request to finish, then send it to another instance connected to the same
pool. Check external prefix-cache hits in the serving metrics or logs. Requests
repeated on one instance may instead hit its local prefix cache.

## Configuration Parameters

Set these options in `kv_transfer_config.kv_connector_extra_config`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `backend` | string | `"mooncake"` | Selects the Mooncake Store backend. |
| `use_layerwise` | bool | `false` | Set to `true` to enable layerwise save and load. |
| `load_async` | bool | `false` | Keep `false` for layerwise loading. Future-layer prefetch is managed by the layerwise transfer pipeline. |
| `layerwise_prefetch_layers` | int | `2` | Positive prefetch distance in layers. Larger values submit loads earlier. |
| `layerwise_max_transfer_blocks` | int | `0` | Maximum key/block rows per range transfer batch. Also batches session operations and lookup work. `0` adds no block limit. |
| `layerwise_max_transfer_bytes` | int | `0` | Maximum size in bytes of an individual contiguous transfer segment. Larger segments are split. This is not a limit on total bytes per API call; `0` adds no segment-size limit. |

Transfer batch limits are optional and are omitted from the examples. For
roles, placement, tenancy, and other pool options, see
[Ascend Store configuration](kv_cache_pool_ascend_store.md#kv-transfer-config-common-parameters).

## Tuning Suggestions

- Start with the default prefetch distance of `2`. Increase it if layers wait
  for remote loads; reduce it if early transfers contend with other communication.
- Use block and segment limits to control transfer batch size for long prompts.
  Smaller batches reduce per-call work but increase the number of calls.
- Evaluate time to first token and throughput with representative prompt lengths
  and prefix hit rates. Benefits depend on how much transfer the attention
  computation can hide and on contention with HCCL or expert-parallel traffic.
