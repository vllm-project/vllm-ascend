# Cross-layer shared KV pool transfers

`cross_layer_kv` extends the existing Memcache GVA layerwise path for models
whose consumer layers share KV sources. The public switch remains
`kv_connector_extra_config.use_layerwise`.

## Routing and reuse

Resolve the backend and the existing layerwise switch before considering the
model topology. Mooncake block-key and Memcache GVA remain separate protocols.

| Backend / switch / topology | Scheduler | Worker |
| --- | --- | --- |
| Any backend, `use_layerwise=false` | `KVPoolScheduler` | Existing bulk worker |
| Mooncake, `use_layerwise=true` | `LayerwiseKVPoolScheduler` | Existing block-key worker |
| Memcache, `use_layerwise=true`, independent KV | `LayerwiseKVPoolScheduler` | Existing GVA worker |
| Memcache, `use_layerwise=true`, shared KV sources | `LayerwiseKVPoolScheduler` | `CrossLayerKVPoolWorker(LayerwiseKVPoolWorker)` |

Both GVA strategies use `_get_layerwise_hit_tokens`, the existing coordinator,
`MemcacheBackend.bind_layerwise_keys`, readability probes and the same key
namespace. The shared-source strategy does not query the bulk lookup server.
For multiple groups, full keys retain `model@group@hash@rank`; the single-group
compatibility format remains `model@hash@rank`.

Readers and writers must use the same GVA protocol. Existing bulk-format
objects are not queried by this strategy. Physical buffer reuse and offload
snapshots remain disabled for shared-source models: source sharing does not
imply that its physical HBM buffers can be reused by other layers.

## Topology and SWA

`kv_source_layer_ids` and `num_hidden_layers` determine source ownership and
the replay cut. No model-name allowlist, source count or interval is hardcoded.
Flat and nested text configurations use the same resolver.

The inherited physical-to-group mapping and `LayerBatchBuilder` use registered
descriptor boundaries, including scales and segmented kernel views. Layers
without a persistent cache member generate no transfer task. Changed source
positions/counts require no parallel address-offset implementation.

SWA storage follows the existing coordinator's block masks:

- With bounded replay enabled, target SWA is not prefix-cacheable and is
  excluded from pool saves and loads. Replay reconstructs its window.
- Otherwise, retain the SWA blocks required at reusable prefix boundaries.
  The window comes from `sliding_window`, not a hardcoded transfer limit.
- With a 128-token window, 128-token blocks and default 128-token retention
  alignment, every block is eligible. Coverage can span the saved prefix,
  rather than only its final 128 tokens. Saved offsets, existing hits and
  valid blocks still limit the actual writes.
- Sparse retention can keep fewer boundary windows, including the required
  replay checkpoint. Keeping only the request's final window would discard
  shorter reusable prefixes and require coordinated lookup/load changes.

## Request scheduling and load overlap

The common GVA scheduler queries readable group objects, applies the existing
reachability/alignment policy, allocates target blocks and builds `ReqMeta`.
Both strategies use synchronous admission (`load_async=false`). Copies can
still overlap computation within the admitted forward.

`bind_connector_metadata` calls `prepare_cross_layer_kv_step` before model
hooks, including save-only steps. This wraps the inherited
`prepare_layerwise_step` and `process_layer_data`: request snapshots use the
adopted hit length, and existing GVA code builds masks, leases, arrays and
`LayerTransferTask` objects. No cross-layer allocation/session plane is added.

`start_load_kv` restores draft and unknown cacheable groups synchronously with
the GVA receiver's copy routine, then queues target source/SWA tasks in first
consumer order. Requests with the same first consumer share a batch. The
background thread is the existing `KVCacheStoreLayerRecvingThread`.

Before attention or a cache-only write, `wait_for_cross_layer_kv_load` waits
for the physical owner's completion event. A completed SDK copy opens the
gate; queue submission never reports early `finished_recving`. Hooks visit
public `MultiConnector.sub_connectors` so composition cannot bypass a gate.

Reads reuse lease acquisition and GVA refresh. Shared-source loads additionally
check object size against the registered GVA layout. Forward completion drains
target copies before DSpark prepares context, and releases leases only after
target writes have stopped.

## Save overlap and publication

The sender is the existing `KVCacheStoreLayerSendingThread`. The inherited
`process_layer_data` prepares full-group allocations and per-layer ranges;
`_submit_attention_save` records the producer event and submits a layer's
copy. Later model layers can compute while earlier source ranges are copied.
Saving a complete group object does not require copying all its bytes at once.

The existing GVA sender publishes with `batch_write_finish(WRITE_OK)` after
all planned copies. Its final publication task is deferred to step end, so
it cannot overtake delayed sources. `wait_for_save` submits remaining tasks
and reuses `_wait_for_final_layer_save` to drain the sender before the step
returns. There is no separate cross-layer save session or completion tracker.

- Bounded replay can write its cut source again; that source uses a final-step
  event instead of its first attention/cache-only boundary.
- Missing hooks and unmapped groups also save at step end.
- DSpark saves are deferred until the runner finalizes after draft context
  preparation, preserving final target/draft cache contents.
- The final publication task remains deferred even when its attention callback
  runs: earlier replay-cut tasks may not have been submitted yet.

## Memcache 1.3 contract

The reviewed [memcache release/1.3](https://gitcode.com/Ascend/memcache/tree/release%2F1.3/src)
commit is `e8d3b81a79108cb915223647ad94add9f8b1d566`; the
[memfabric_hybrid release/1.3](https://gitcode.com/Ascend/memfabric_hybrid/tree/release%2F1.3/src)
commit is `69ea3667fab4542689475a84409d632426d2393d`.

Duplicate `batch_alloc` calls can return a shared GVA, including for ALLOCATED
objects. Positive addresses and existence probes do not establish exclusive
ownership. `WRITE_FAIL` can remove a key unconditionally.

These facts do not prohibit ranged GVA copies. This strategy reuses ordinary
GVA allocation/publication and its failure policy: no independent exclusive
writer session, no deleting shared keys to compensate failed copies, and no
compensating WRITE_FAIL after an uncertain publication response. The change
adds no cross-writer transaction or rollback protocol; concurrent prefix
writers still require integration validation.

## Runtime and lifetime boundaries

The initial worker requires Model Runner V2 and supports Memcache TP
deployments. Explicit V1 configurations are rejected before GVA initialization;
the V1 runner is unchanged. PP, PCP, DCP and KVPP are also rejected. Eager and PIECEWISE
support target prefetch; the V2 `DeepseekV41DSparkModel` drafter is supported.
Other speculative configurations, unsupported graph modes, SP, microbatching,
missing hooks and unknown layouts retain synchronous GVA restoration. No
fallback switches to bulk keys.

PIECEWISE reuses ordinary layerwise graph negotiation and platform splitting
operations. DS4.1 attention, cache-only writes and bounded decoder replay run
between graph segments, so host gates execute on every replay. Bounded replay
uses caller-owned output buffers. V2 refreshes Engram lookup buffers before
PIECEWISE dispatch and retires device metadata after its consumers.

Use `--enforce-eager` for eager execution. For PIECEWISE, omit that flag and
select `compilation_config.cudagraph_mode=PIECEWISE`. V2 requires a compiled
model or the existing `VLLM_USE_BREAKABLE_CUDAGRAPH=1` option. This does not
imply FULL graph support.

A fatal load error drains or stops the receiver before lease release. The
error survives collection of request failure notifications and prevents later
metadata from reusing partially restored state. Automatic hybrid state
rollback/retry is not implemented.

## Validation

CPU tests exercise the actual GVA sender/receiver and registered tensor
addresses: key parity, source/SWA round trips, segmented views, changed source
positions, early-consumer overlap, replay-cut publication order, missing or
wrong-sized objects, failed copies, lease cleanup and DSpark final contents.
Existing Memcache GVA and Mooncake block-key regressions are also required.

Before readiness, validate Ascend NPU capture/replay and precision, bounded
replay ON/OFF, DSpark ordering, concurrent writers, worker failure/restart,
TTFT/throughput/HBM/network measurements. CPU simulations do not establish
device correctness or performance.
