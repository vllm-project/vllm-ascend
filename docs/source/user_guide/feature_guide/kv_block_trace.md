# KV block lifecycle tracing

KV block tracing records which request owns a block, when Mooncake transfers it,
and which slots real or dummy forwards access. It is useful for silent KV
corruption, stale slot mappings, block reuse, and PD request correlation. The
tracer observes the runtime; it does not mask writes or repair caches.

## Module contract

KV trace collects and queries evidence about block lifetime, request mappings,
transfers and selected forward-window byte changes. It preserves source events,
records observation failures, and explains derived associations. Its completion
criterion is faithful, queryable evidence within the declared backend, layout
and sampling scope.

Response collection, repetition detection, model-quality evaluation, bad/ref
numeric comparisons, intervention toggles, automatic blocking and final root
cause conclusions belong to callers or separate diagnostic tools. They are not
requirements for this module to be complete. Adding a backend means adding its
event adapter, not moving that backend's data management into the tracer.

For the Case 4 dummy overwrite, this module supplies block reuse, transfer
events, actual dummy slots/batch state and selected cache differences. An
external investigation relates these to the response and tests a write mask.
It does not require cosine calculations or a built-in response detector.

## Reuse of upstream facilities

Cache store, removal and clear events come from upstream `KVCacheManager.take_events()`.
The observer records a metadata copy using the existing msgspec serializer and
returns the original list and event objects to the scheduler's existing publisher.
It neither drains the queue separately nor adds another publisher, hash-state
machine or cache eviction/reset hook. Prompt `token_ids` and `extra_keys` are
omitted from the trace copy; the original published events remain unchanged.

Enable native KV events through vLLM's existing `--kv-events-config` option when
this history is needed. The trace does not change that configuration; its
`trace.capability` event reports whether the source is supported and enabled.
Native events are hash-level metadata under `cache.native_event.native_event`;
they do not supply physical block IDs or allocation epochs.

The additional allocator/request observer supplies those missing physical
identities and forward associations. It uses the existing scheduler output and
MessageQueue transport, actual worker block tables and Mooncake metadata.
Upstream metrics, profiler, request sampling and cache policies are retained.

## Initial P0 implementation

This branch adds allocator generations, scheduler-to-worker context, and log
completeness checks. These are diagnostic records, not a runtime KV guard.
Targeted Ascend 910B4 validation now covers v1 eager Qwen inference, prefix reuse
and one real Mooncake 1P1D transfer with request/block correlation. Controlled
NPU mutations confirmed both observed changes and missed changes outside the
sampled blocks or before the observation window. Complete logging is not an
integrity guarantee.

DeepSeek-V2-Lite TP4+EP captured all 27 layers on all four ranks for the first
request; a second prefix-hit request failed in native attention with tracing
both enabled and disabled. PD also showed native shutdown errors in both arms.
Those initial runs did not establish production stability or reproduce the PPT
faults. Broad performance validation, PD content checksums, device completion tracking, offload
adapters, ring-buffer checkpoints and hot reload remain future work. The writer
still performs synchronous JSONL I/O.

Subsequent [original Case 4 validation](../../../design/kv-block-trace-case4-validation-20260929.md)
reproduced the dummy overwrite in DP2 graph mode. A separate
[Case 4 latency comparison](../../../design/kv-block-trace-performance-20260929.md)
measured no material default-trace slowdown for its concurrency-2 workload;
last-layer snapshots increased spring-request P50 by 44.4% and reduced output
throughput by 31.7%. Those bounded samples do not establish production overhead.
The snapshot arm reached the per-writer byte limit in its final round, so all
arms use the same first 19 intact rounds for the comparison.

## Enable tracing

Set the same run ID on P and D, before starting their processes:

```bash
export VLLM_ASCEND_KV_TRACE='{"directory":"/tmp/kv-traces","run_id":"pd-debug"}'
```

The default is an empty value: disabled. Disabled execution installs no scheduler
patch, opens no trace files, and performs no trace-related tensor copies.

When enabled, each component/process writes a separate JSONL file under
`/tmp/kv-traces/pd-debug/`. Copy the files from all nodes into one directory for
analysis. Use a new run ID for independent experiments.

To compare cache contents before and after forwards, enable snapshots:

```bash
export VLLM_ASCEND_KV_TRACE='{"directory":"/tmp/kv-traces","run_id":"pd-snapshots","snapshots":true,"layers":[-1],"max_events":100000,"max_snapshot_bytes":8388608}'
```

| Option | Default | Meaning |
| --- | --- | --- |
| `directory` | Required | Output directory, created on first record. |
| `run_id` | `kv-trace` | Single path component: letters, digits, `_`, `-`, `.`. |
| `snapshots` | `false` | Compare selected KV blocks around forwards. |
| `layers` | `[-1]` | Layer indices **within each KV group**; negative indices count from the end. `[0,-1]` selects first and last. |
| `max_events` | `100000` | Per-writer limit, followed by one `trace.truncated` record. |
| `max_snapshot_bytes` | `8388608` | Per-forward budget for before-snapshot data. A comparison temporarily also holds an after copy. |
| `device_metadata` | `true` | Observe device slot mappings and block tables. Set `false` for host lifecycle/context only; snapshots must then be disabled. |
| `max_trace_bytes` | `67108864` | Per-writer JSONL byte budget, plus at most one terminal truncation record. Independent of snapshot memory. |

Tracing intentionally changes timing. Even without snapshots, observing actual
GPU slot mappings and block tables causes device-to-host synchronization. The
snapshot option additionally copies and compares cache data. Use it for a
bounded diagnostic workload, not throughput measurement. Configuration is read
at component initialization; changing the environment requires restarting the
processes.

For host-only lifecycle/context tracing without trace-related tensor copies:

```bash
export VLLM_ASCEND_KV_TRACE='{"directory":"/tmp/kv-traces","run_id":"p0-host","device_metadata":false,"max_trace_bytes":67108864}'
```

This mode records `metadata_observation=not_observed`; it cannot diagnose actual
device slot corruption. It still incurs Python bookkeeping and log I/O overhead.
Event/byte budgets stop the writer rather than silently overwriting its history.

## Events and coverage

| Stage | Events | Source |
| --- | --- | --- |
| Cache layout | `cache.config` | Scheduler cache manager, all KV groups. |
| Recording scope | `trace.manifest`, `trace.stop` | Installed package versions, loaded module paths, capabilities, and normal writer closure. |
| Allocator identity | `pool.baseline`, `block.alloc`, `block.ref_change` | Actual BlockPool allocation/reference methods; a unique pool ID and allocation generation independent of request leases. |
| Native cache events | `cache.native_event` | Upstream `BlockStored`, `BlockRemoved`, `AllBlocksCleared`, when enabled through vLLM's existing KV events configuration. |
| Schedule correlation | `schedule.dispatch`, `schedule.received`, `schedule.apply`, `mapping.mismatch` | Scheduler message, host-side application and request block-ID comparison. Not a device/kernel mapping check. |
| Prefix cache lookup | `cache.lookup` | Matched block IDs and token count. |
| Ownership and reuse | `block.acquire`, `block.release`, `request.blocks` | KVCacheManager allocation, cache, and free boundaries. Includes owners, reference count, and lease. |
| Allocation pressure | `block.allocation_failed` | `allocate_slots` returned `None`. |
| P offers blocks | `transfer.offer` | Request ID, prompt blocks, and delayed-free decision. |
| D schedules/queues reads | `transfer.scheduled`, `transfer.queued` | Local and remote request IDs, engine ID, and block lists. |
| Transport | `transfer.start`, `transfer.complete`, `transfer.error`, `transfer.skipped` | Each Mooncake receive operation. |
| Worker completion | `transfer.worker_finished` | Completion returned by the worker connector to the scheduler. |
| Runtime forward | `forward.begin`, `forward.end`, `forward.error` | Real and dummy v1 forwards, including graph replay. |
| Optional content comparison | `cache.diff`, `snapshot.skipped`, `snapshot.truncated` | Selected attention layers and blocks touched by non-padding slots. |
| Diagnostics health | `trace.gap`, `trace.truncated`, `trace.observation_error`, `trace.capability` | Missing context, partial allocator operations, recording limits, or unsupported pool APIs are explicit. I/O failures disable the writer. |

The current transfer integration is **MooncakeConnectorV1** (the P2P connector),
and the forward integration is **NPUModelRunner v1**. Layerwise, hybrid and pool
connectors, runner v2, and kernels that bypass this forward path need their own
integration. Scheduler ownership events apply to scheduler subclasses that call
the standard Scheduler initializer, including the Ascend recompute scheduler.

Ownership records describe manager-boundary state, not every internal allocator
instruction. A `lease` advances when a group/block has no observed request owners
and is acquired again. Shared prefix owners retain the same lease. A new lease
does **not** imply the cached bytes changed or that a cache hit was invalid.
`pool_id + block_id + alloc_epoch` identifies an observed allocation. The epoch
increases on `get_new_blocks`, not on `touch`, `free_blocks`, or prefix cache
reset. Existing content at observer attachment has an unknown epoch until a
real allocation is observed. A failed/partially completed allocator operation
invalidates observed epochs and emits a gap; later allocations do not reuse old
epoch values. The pool's null block is excluded from ownership and snapshots;
without scheduler context, snapshots use the pinned vLLM default null ID zero.

The scheduler wrapper resolves the concrete instance's `schedule` override, so
Ascend scheduler subclasses retain their output type and additional fields.
The trace context is attached to `SchedulerOutput` as a JSON-compatible attribute.
At the pinned vLLM revision, the multiproc MessageQueue pickles the object and
preserves this attribute. CPU tests verify pickle round trips including output
subclasses; other transports require independent verification. Missing or
incompatible worker context emits `trace.gap`, never an inferred correlation.

Each dispatched request carries its allocator block IDs/epochs, mapping
revision, scheduled token count and available scheduler computed-token count.
The revision advances when its mapping or allocation identities change. Worker
`schedule.apply` compares against host request state after `_update_states`.
Deferred speculative token corrections are explicitly marked, not considered
finished. No comparison with final kernel arguments is claimed here.

Snapshot support covers attention caches with an explicit token dimension:
`[blocks,tokens,...]`, tuple/list components (including MLA latent KV and RoPE K),
and dense `[2,blocks,tokens,heads,dim]` K/V. Unsupported specs/layouts are reported
as skipped. State caches and compressed-cache specs are not interpreted as
ordinary token caches. Packed NZ caches are explicitly skipped because their
physical layout is not described by the apparent token dimension. Out-of-range
layers and budget limits are explicit.

Snapshot comparison is bitwise, so unchanged NaNs are not false writes. A diff
contains changed offsets, counts per offset, and before/after SHA256 digests. It
does not export raw tensors, input token IDs, prompts, or credentials. A hash
covers the selected kernel block, including any unused tail; unequal hashes
alone do not prove valid-prefix corruption.

## Correlation and ordering

Every record includes a schema version, run ID, host, PID, component, writer ID,
sequence number, wall-clock nanoseconds, monotonic nanoseconds, engine ID and DP
rank. Workers also include TP/PP rank. Forward spans tie input metadata to diffs,
completion and errors. Preserve the exact P/D request IDs; transfer events link
them explicitly, without assuming how a proxy forms its IDs.

On Linux, `host_boot_id` and `process_instance_id` (boot UUID, PID and process
start ticks) distinguish host/process lifetimes. They connect component writers
in the same worker process. Unavailable procfs produces null identities; older
records can use the explicitly labelled `legacy_host_pid` association basis.

New records use schema v2: sequence, nanosecond timestamps and allocation epochs
are decimal strings. `trace_id` is retained alongside `writer_id`; `event_id`
and `parent_event_ids` link dispatch, application and forward observations.
The reader accepts schema v1 and v2 without manufacturing missing v1 epochs.
The display remains an approximate wall-clock merge, not a topological sort.

Execution context is scoped to one `execute_model` call. Dummy spans have no
request owners; `batch_request_ids` records only stale/persistent batch context.
Independent dummy runs do not inherit an earlier step. A snapshot epoch is
marked `scheduler_context` only when matching host request block IDs were
observed; this is not an independent device-side generation measurement.

Scope a physical block by deployment/run, host, engine, rank, KV group and block
ID. The same block number on two ranks is not the same storage. Scheduler owners
apply to workers in that engine; TP workers can contain different shards.
When an allocator block is split into smaller kernel blocks, records preserve
`kernel_block_size` and `kernel_block_id`. Diffs normalize `block_id` and token
offsets back to the allocator's coordinates, and the reader applies the same
conversion to device block tables. This keeps ownership and access traceable.

Writer sequence is ordered under a lock, including receive threads. Wall time
supports an approximate merged view; clocks across hosts are not necessarily
synchronized. `transfer.complete` means the connector read returned successfully,
not that the bytes were compared with P. `transfer.worker_finished` is a
worker completion notification and may include timeout-expired send requests;
it is not proof of a remote acknowledgement. Only `block.release` records
the cache manager actually relinquishing request ownership. `forward.end` with `host_return` means
the forward call returned, and may precede device completion. Cache comparisons
are device-observed; warmup/profiling/capture paths are excluded.

## Inspect the chain

The reader uses only the Python standard library:

```bash
python tools/kv_block_trace.py /tmp/kv-traces/pd-debug
python tools/kv_block_trace.py /tmp/kv-traces/pd-debug --request REQUEST_ID_OR_PREFIX
python tools/kv_block_trace.py /tmp/kv-traces/pd-debug --engine ENGINE_ID --group 0 --block 1
python tools/kv_block_trace.py /tmp/kv-traces/pd-debug --block 1 --json > block-1.jsonl
python tools/kv_block_trace.py /tmp/kv-traces/pd-debug --pool-id POOL_UUID --block 1 --epoch 2 --json
python tools/kv_block_trace.py /tmp/kv-traces/pd-debug --host HOST --engine ENGINE_ID --dp-rank 0 --tp-rank 0 --pp-rank 0 --block 1 --json
python tools/kv_block_trace.py /tmp/kv-traces/pd-debug --check
```

`--epoch` requires both `--pool-id` and `--block`. Without an epoch, a block
query intentionally includes all observed uses of that address. Explicit P/D
request pairs expand request aliases; co-batched requests do not become aliases
and therefore do not pull in their unrelated later forwards.

### Dummy access context

The reader reconstructs `access_relations` for dummy spans from three sources:

1. A preceding `schedule.received` binds the worker process to a scheduler and pool.
2. The scheduler's observed allocation/acquire/release stream supplies the epoch
   and active request references for each actual non-padding slot's block.
3. Transfer events in the same process supply the last observed transfer in
   that allocation lifetime, when its writer stream is complete.

Reconstruction uses monotonic timestamps only within one host/boot and scopes
pools by run, host, engine, DP rank, scheduler and pool ID. Missing sequences,
explicit observation gaps, ambiguous process bindings or tied lifecycle
timestamps leave the association unknown. Old transfers are not carried across
new allocations. A released request is not restored as owner by an old transfer.
Legacy logs lack boot/process-start identity, so their weaker basis is visible.

`access_relations` are **query context**, not an assertion that the dummy belongs
to a request or that its device storage independently verified the epoch.
Original `request_ids`, `pool_id` and `alloc_epoch` are preserved. Each relation
contains its supporting event IDs and `device_epoch_verified=false`. Cache diffs
retain context at `span_begin`; `context_changed_during_span` warns when observed
ownership, allocation or transfer context changed before the after-snapshot.
When context changes, a separate `span_end` relation and its evidence are kept;
queries for either boundary's request/epoch can inspect that overlapping window.
An unchanged context does not prove absence of uninstrumented writers.

Request and pool/epoch queries retain these related dummy spans. Their supporting
allocation/binding/transfer records are also included, with
`query_context=access_relation_evidence` when added only as evidence. Rank filters
select access records; a supporting scheduler record may have no TP/PP rank.
Related owners do not become request aliases and do not pull in unrelated later
forwards. `--json` exposes the full basis; text output labels it `related:`.
Re-reading a filtered export recalculates associations rather than trusting
previously derived fields.

If source evidence is missing, inspect by host/engine/rank and raw block number.
The reader intentionally does not guess an epoch to satisfy an epoch query.
Always provide the full original run files for reconstruction; a partial query
export may lack the sequence history needed for reliable association.

`--check` checks all input records (optionally restricted by `--run-id`), not the
request/block display filter. It reports malformed records, sequence gaps or
duplicates, missing causal parents, explicit observation failures/skips, and
missing start/stop records. Exit code 0 means the supplied recording passed
these checks; exit code 2 means incomplete or unverified. A live writer without
a stop record is also unverified. It cannot detect a completely absent actor's
file. Its output always says `kv_integrity: not_checked`: complete logging does
not establish correct cache contents or complete hook coverage.

Request filtering follows explicit P/D links and retains related forward spans,
including dummy spans with a reconstructed block context. `DUMMY_CACHE_CHANGED` highlights a
non-null block changed during a dummy span; it is a diagnostic lead, not an
automatic corruption verdict. Check ownership, valid prefix length, surrounding
transfers, and overlapping work before attributing the change to a specific
request. Snapshots can overlap connector writes and do not establish the writer
by themselves.

For a stale-slot investigation, follow:

1. `block.acquire` and `request.blocks`: which request owns the local block?
2. `transfer.scheduled` / `transfer.complete`: which remote request supplied it?
3. `forward.begin`: is the batch real or dummy, and what physical slots does it use?
4. `cache.diff`: did the relevant valid-prefix offsets change?
5. `transfer.worker_finished` and `block.release`: when was ownership released?

Retain a baseline without snapshots and analyze observed content changes
separately. Tracing does not change sampling, slot mappings or cache contents.

## Tests

```bash
pytest --confcutdir=tests/ut/debug tests/ut/debug/test_kv_trace.py -q
```

These CPU tests run without a full vLLM/NPU installation. They cover shared
ownership and reuse, transfer-ID correlation, bounded thread-safe output,
bitwise cache changes, snapshot limits, profiling/capture exclusion, and error
propagation. Actual runtime integrations must additionally be exercised on NPU.

The P0 tests also cover nonzero null block IDs, zero-reference prefix reuse,
allocation generations, iterator preservation, pickle transport, scheduler
subclass overrides, mapping revisions, missing context, dummy isolation,
request alias precision, byte limits, reader compatibility and CLI exit codes.
