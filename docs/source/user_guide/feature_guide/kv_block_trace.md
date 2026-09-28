# KV block lifecycle tracing

KV block tracing records which request owns a block, when Mooncake transfers it,
and which slots real or dummy forwards access. It is useful for silent KV
corruption, stale slot mappings, block reuse, and PD request correlation. The
tracer observes the runtime; it does not mask writes or repair caches.

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

Tracing intentionally changes timing. Even without snapshots, observing actual
GPU slot mappings and block tables causes device-to-host synchronization. The
snapshot option additionally copies and compares cache data. Use it for a
bounded diagnostic workload, not throughput measurement. Configuration is read
at component initialization; changing the environment requires restarting the
processes.

## Events and coverage

| Stage | Events | Source |
| --- | --- | --- |
| Cache layout | `cache.config` | Scheduler cache manager, all KV groups. |
| Prefix cache lookup | `cache.lookup` | Matched block IDs and token count. |
| Ownership and reuse | `block.acquire`, `block.release`, `request.blocks` | KVCacheManager allocation, cache, and free boundaries. Includes owners, reference count, and lease. |
| Allocation pressure | `block.allocation_failed` | `allocate_slots` returned `None`. |
| P offers blocks | `transfer.offer` | Request ID, prompt blocks, and delayed-free decision. |
| D schedules/queues reads | `transfer.scheduled`, `transfer.queued` | Local and remote request IDs, engine ID, and block lists. |
| Transport | `transfer.start`, `transfer.complete`, `transfer.error`, `transfer.skipped` | Each Mooncake receive operation. |
| Worker completion | `transfer.worker_finished` | Completion returned by the worker connector to the scheduler. |
| Runtime forward | `forward.begin`, `forward.end`, `forward.error` | Real and dummy v1 forwards, including graph replay. |
| Optional content comparison | `cache.diff`, `snapshot.skipped`, `snapshot.truncated` | Selected attention layers and blocks touched by non-padding slots. |
| Diagnostics health | `trace.truncated`, `trace.observation_error` | A trace with these events is incomplete. I/O failures disable that writer and log an error. |

The current transfer integration is **MooncakeConnectorV1** (the P2P connector),
and the forward integration is **NPUModelRunner v1**. Layerwise, hybrid and pool
connectors, runner v2, and kernels that bypass this forward path need their own
integration. Scheduler ownership events apply to scheduler subclasses that call
the standard Scheduler initializer, including the Ascend recompute scheduler.

Ownership records describe manager-boundary state, not every internal allocator
instruction. A `lease` advances when a group/block has no observed request owners
and is acquired again. Shared prefix owners retain the same lease. A new lease
does **not** imply the cached bytes changed or that a cache hit was invalid.
Block zero is retained in metadata but excluded from ownership and snapshots.

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
```

Request filtering follows explicit P/D links and retains related forward spans.
An empty dummy batch has no request ID: inspect its affected **block** to include
interference from unrelated requests. `DUMMY_CACHE_CHANGED` highlights a
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
