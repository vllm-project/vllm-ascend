# Cross-layer shared KV pool transfers

`cross_layer_kv` overlaps pooled KV transfers with model execution for models
whose consumer layers share KV sources. It retains the existing group keys
and object layout while transferring dependency slices before their consumers.
The only public opt-in is `kv_connector_extra_config.use_layerwise`.

## One switch, two layerwise strategies

| Configuration and topology | Scheduler | Worker strategy |
| --- | --- | --- |
| `use_layerwise=false` | `KVPoolScheduler` | Existing bulk transfers |
| `use_layerwise=true`, independent layer KV, such as DS4 | `LayerwiseKVPoolScheduler` | Existing physical-layer transfers |
| `use_layerwise=true`, shared KV sources, such as DS4.1 | `LayerwiseKVPoolScheduler` | `CrossLayerKVPoolWorker` |

The scheduler and connector use the same topology resolver. The cross-layer
strategy uses group-object hit queries and synchronous request admission;
ordinary layerwise retains its backend-bound key protocol and behavior.
Sharing an object does not require copying it as one operation: the worker
builds dependency units from the registered producer/consumer layout.

Cross-layer saves retain in-flight state blocks until all workers report
completion, and reset refuses to discard outstanding save work. Physical-layer
offload snapshots and buffer reuse are disabled for this strategy, including
allocator, memory-budget, V1/V2 state-copy and KVPP hook selection. These checks
receive model context explicitly; the user's config is not rewritten.
Both layerwise strategies normalize `load_async` to false. The cross-layer
pipeline overlaps within a scheduled forward and never reports early receive
completion to the scheduler.

## Model topology and SWA retention

The source layers come from the text model's `kv_source_layer_ids`, validated
against `num_hidden_layers`. There is no model-name allowlist, fixed source
count, layer interval, or dependency on layers 2/8/14/20. Flat configs and
nested text configs use the same resolver. For example, a compatible 16-layer
model with sources `[1, 5, 9]` gets three source units and a cut at layer 9.
Registered bulk descriptors determine each unit's actual byte range.

Compatible model implementations expose `supports_cross_layer_kv_transfer_hooks=true`
on every backbone attention layer registered in `static_forward_context`.
This means both ordinary forward and cache-only writes enforce the dependency
gate before accessing caches. Models that reuse the existing attention class
inherit the capability. A new attention implementation must implement that
hook contract; a similar config alone cannot make asynchronous loads safe.
Missing hooks or incompatible cache layouts retain bulk transfers.

SWA storage follows the existing coordinator's block masks:

- With `swa_bounded_replay=true`, target SWA is not prefix-cacheable and is
  excluded from pool saves and loads. Replay reconstructs the needed window.
- With bounded replay off, saves retain the SWA blocks needed at reusable
  prefix boundaries. The window comes from `sliding_window`; 128 is not a
  hardcoded transfer limit. Copies still use complete registered blocks.
- With default retention, tails are retained at every transfer-aligned
  boundary. For a 128-token window, 128-token SWA blocks and 128-token
  alignment, every block is eligible: pool coverage can span the full saved
  prefix, rather than just its last 128 tokens. Saved offsets, existing hits,
  and the current step's valid blocks still determine actual writes.
- Sparse retention can keep fewer boundary windows, including the required
  replay checkpoint. Keeping only the final request window would discard
  shorter reusable prefixes and requires coordinated lookup/load changes.

## Producer and consumer scheduling

A step plan is bound in `bind_connector_metadata`, before any model hooks.
It must not be initialized in `start_load_kv`: save-only steps can call that
method after the target forward.

The plan uses the registered bulk descriptor boundaries, including separate
index K, scale, folded views, and segmented kernel pages. Resource-name
positions and physical page strides are not object byte offsets.
`chunk_member_transfer` delegates address and size calculations to the
existing `prepare_value` implementation.

| Configuration | Incremental-save units | Pipelined-load units |
| --- | --- | --- |
| Bounded replay on | Four KV-source slices in the typical backbone | Four source dependencies |
| Bounded replay off | Four source slices and ten SWA groups | Four sources and forty SWA members |

These counts describe the typical 40-layer topology; registration and
effective cacheability determine the actual maps.

The load path keeps synchronous external-hit admission
(`load_async=false`). Before forward, it queues copy tasks in first-consumer
order, interleaving requests instead of loading one request completely first.
The attention gate waits for its dependencies and checks a step-local error.
It never signals `finished_recving` merely because work was queued.

Reads acquire leases, refresh GVA metadata under those leases, check object
sizes, copy the requested descriptor ranges, and release leases after every
unit has settled. Missing or empty ranges still settle their gates.
The cache-only cut-layer operation also waits before updating its caches.
Attention hooks traverse the public `MultiConnector.sub_connectors` hierarchy
so composing the pool with another connector cannot bypass a load gate.

For incremental saves, a producer event permits background copying after the
last write. Early slices overlap subsequent layers. The final step event seals
missing hooks and conservatively handles the replay cut, which can be written
again after its cache-only pass. The usual previous-batch fence covers early
slices, the seal, and all fallback bulk puts.

## Memcache release/1.3 contract audit

The reviewed source snapshots are:

- [memcache release/1.3](https://gitcode.com/Ascend/memcache/tree/release%2F1.3/src):
  `e8d3b81a79108cb915223647ad94add9f8b1d566`.
- [memfabric_hybrid release/1.3](https://gitcode.com/Ascend/memfabric_hybrid/tree/release%2F1.3/src):
  `69ea3667fab4542689475a84409d632426d2393d`.

The relevant memcache implementation paths are:

- `src/memcache/csrc/client/mmc_client_default.cpp`:
  `BatchMalloc`, `BatchWriteFinish`, `BatchDataOperation`,
  and `ExecuteConcurrently`.
- `src/memcache/csrc/meta_service/mmc_meta_manager.cpp`:
  `Alloc`, `UpdateState`, and `AddLease`.
- `src/memcache/csrc/entities/mmc_blob_state.cpp`:
  `MMC_REPEAT_ALLOC` transitions.
- `src/memcache/csrc/client/mmc_client_local_gva_blob_tracker.cpp`:
  local write/read lease tracking.

`BatchMalloc` sets `ALLOC_FLAGS_GVA_MALLOC_MASK`. On duplicate keys,
`Alloc` returns the existing object's descriptors and grants another lease,
including while the object is ALLOCATED. Thus a positive GVA does **not**
establish exclusive write ownership. An existence probe cannot remove this
race. Additionally, `UpdateState(MMC_WRITE_FAIL)` removes the key before
checking the ordinary blob state transition.

Consequently Memcache keeps
`Backend.supports_exclusive_allocation=false`, and this SDK uses **atomic
bulk saves plus pipelined loads**. The incremental sender requires that
explicit backend guarantee. No configuration flag overrides it. A future
backend implementation must prove exclusive allocation, complete-object
visibility, and safe cleanup before enabling the session-save plane.

The sender marks commit as attempted before the publication RPC. A failed
or missing reply never causes a compensating WRITE_FAIL. Successful SDK
WRITE_OK consumes the write lease; the sender does not remove an unrelated
read lease afterward.

The copy path uses the SDK's normal flag-zero GVA validation and synchronous
copy operations. `ExecuteConcurrently` collects every future, including after
one fails. The memfabric path goes through `SmemBmEntry::DataCopyBatch` and
`hybm_data_batch_copy`; NPU execution still requires validation against the
deployed driver and transport.

## Failure and lifetime boundaries

The step retains immutable request/block snapshots and independent task
events. Task errors survive collection of request-level failure notifications.
The next bind, forward completion, reset, and shutdown drain load work before
replacing gates or reusing target buffers.

A load error after computation begins fails the step and drains queued
transfers. It does not claim to roll back SWA/compressor state or automatically
reschedule a partially executed hybrid request. Distributed failure handling
and worker restart behavior need hardware validation before production use.

The runtime gate supports eager and PIECEWISE Memcache loads with full blocks. With the
V2 model runner, DeepSeek V4.1 DSpark also supports pipelined target loads.
The target topology remains configuration-derived; this speculative extension
requires the registered `DeepseekV41DSparkModel` drafter. Other speculative
methods and drafters retain bulk transfers.

Load eligibility is per cache group: compatible target groups use consumer
gates, while draft groups and other cacheable unmapped groups are restored
synchronously before target forward. Noncacheable private state is excluded.
Request metadata and load specs are copied for each path. Synchronous bulk
exceptions and failed status results drain pending target copies and abort
before forward; step-local errors survive consumed failure notifications. The existing
`finish_forward` fence drains target writes before DSpark prepares context or
proposes tokens. The next metadata bind cannot replace live gates.

Speculative saves always retain bulk publication, even with an exclusive
backend. The V2 runner finalizes those saves after draft context preparation;
no incremental producer callback publishes provisional draft/verify state.
The pool coordinator's existing masks and trailing-block recomputation policy
still determine which complete committed blocks can be reused.

Both strategies reuse the connector's existing PIECEWISE negotiation and
platform splitting operations. DS4.1 attention, cache-only writes and bounded
decoder replay execute between graph segments, so consumer waits and KV hooks
run for each replay. Bounded replay uses caller-owned output buffers, including
during NONE warmup, rather than specializing on the warmup's replay rows.
The V2 runner refreshes Engram's persistent lookup buffers before PIECEWISE
dispatch and retires host-submitted device metadata after its consumers.

Use `--enforce-eager` for eager execution. For PIECEWISE, omit that flag and set
`compilation_config.cudagraph_mode` to `PIECEWISE`. The V2 runner requires a
torch-compiled model or the existing `VLLM_USE_BREAKABLE_CUDAGRAPH=1` setting;
the platform's default opt-in policy is unchanged. DSpark and bounded replay
ON/OFF use the same graph boundaries; FULL graph support is not implied.

PP, PCP, DCP, KVPP, SP, microbatching, unsupported graph modes and partial
objects retain bulk behavior. Unknown layouts retain bulk at the group
boundary. KV-event publication also keeps saves on the bulk path.

## Verification

CPU regression tests cover:

- Shared/independent topology routing under the same switch, synchronous
  admission, group lookup, source-block retention, and hook dispatch.
- Physical-layer buffer reuse remains enabled only for the original strategy.
- Real registered K/scale descriptor boundaries and the 369,280-byte payload
  of the BF16/INT8 example, independently of its 540,928-byte physical capacity.
- New-save/new-load round trips and existing bulk descriptor equivalence.
- Unknown and incomplete layouts, empty dependencies, and fallback groups.
- Commit success followed by a lost reply, duplicate callbacks, and step sealing.
- Read-lease acquisition, GVA refresh, partial-lease cleanup, and persistent errors.
- First-consumer ordering and a real background-thread test where an early
  consumer proceeds while a later source is still copying.
- DSpark target/draft group partitioning, restored draft bytes while a later
  target source is copying, metadata isolation, repeated load calls and draining
  in-flight target writes when synchronous draft restoration raises or returns
  missing, failed or malformed status results.
- Eager/PIECEWISE load eligibility and unsupported-mode fallbacks, existing
  graph-cut registration, fixed replay buffers, changing bounded replay
  metadata, NONE-first compilation, and V2 Engram/metadata lifecycle.

Native NPU PIECEWISE capture/replay, memory use across graph sizes, and
request/logit equivalence with DSpark and bounded replay ON/OFF require
hardware validation; CPU tests do not establish these results.

NPU E2E (including DSpark), actual SDK execution, cross-rank failures, Graph/MTP extensions, and
performance traces remain required before promoting this draft. Measure copy
start/end, producer events, consumer arrival, gate wait, TTFT, throughput,
and HBM/NIC contention; no fixed speedup is claimed.
