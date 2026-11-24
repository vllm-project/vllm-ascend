# Engram SSD offload

For a local DeepSeek V4.1 checkpoint with INT8 Engram embeddings, enable SSD
offload by adding one field to the service's additional configuration:

```bash
--additional-config '{"engram_ssd":{"path":"/data/engram"}}'
```

Merge this field with any other additional configuration required by your model.
The directory must be on the intended SSD filesystem and writable by the service.
Preparation runs during configuration initialization, before engine workers are
constructed. The loader, artifact paths, version and host cache defaults are
resolved automatically. There is no separate conversion command to run.

## First start and reuse

The first start streams the checkpoint's Engram tensors into
`<path>/<artifact-version>/table.bin` and writes a `manifest.json` beside it.
Each row contains 256 INT8 values and eight FP32 scales. Layer boundaries are
padded to 4096 rows. Preparation needs free space for the entire output plus a
64 MiB reserve, and can take substantial time for large checkpoints. The API is
not ready until preparation and normal model initialization finish.

Subsequent starts with the same checkpoint and directory reuse the completed
table without rewriting it. Changing cache size or TP count does not repack the
table; changing PCP count also reuses it. Logs distinguish `Preparing Engram SSD table` from
`Reusing complete Engram SSD table`.

A directory lock serializes concurrent initializers. The writer flushes and
atomically publishes the table and then the complete manifest. An interrupted
conversion is retried from the beginning; an incomplete file is never used for
serving. Only the preparer's own `.incomplete` files are reclaimed automatically.
Older completed versions are retained; remove unused versions manually only
after services using them have stopped.

Artifact identity includes format, source headers, resolved source paths, size,
inode, modification/change timestamps and bounded content samples. Reuse checks
the output's size, filesystem identity and samples too. Each file contributes at
most three 64 KiB samples; restarting does not scan all checkpoint/table bytes.
This is not full-file integrity verification: a change outside sampled regions
with unchanged metadata may go undetected. Keep checkpoint and generated files
immutable while serving. A full table SHA-256 is recorded during generation for
separate offline verification.

## DDR budget

The default `host_memory_bytes` is **32,000,000,000 bytes for the service's Engram
host cache**, not NPU HBM. To change it:

```bash
--additional-config '{"engram_ssd":{"path":"/data/engram","host_memory_bytes":32000000000}}'
```

The budget is split among TP owners. DP and PCP ranks on the same node share each
owner's cache. Five eighths of each owner's budget is reserved for cached row
payload; the remainder covers cache metadata, I/O buffers and rebuild peak
memory, checked using the existing resource estimator. For TP4/DP2, the default
is four owners, each with an 8 GB budget and 5 GB payload. Thus
`cache_payload_bytes` is included in `total_host_budget_bytes`.

This budget does not include all model/process memory or filesystem page cache.
The early memory check uses host `MemAvailable`, not container cgroup limits;
choose a budget that also fits your container limit and leaves room for the
rest of the service. Allocation can still fail if available memory changes.

Default I/O settings are eight threads per disk, queue depth 1024 and 1280 AIO
events. Threads use OS scheduling within the inherited CPU affinity. Expert I/O
options remain available in `engram_ssd`; no fixed machine-specific CPU list is
assumed. These defaults have not been benchmarked for every SSD/topology.

## Supported configuration and compatibility

Automatic preparation currently supports local safetensors checkpoints with two
Engram layers, 256 INT8 codes per row and eight E8M0 or FP32 scales. It retains the
existing SSD runtime restrictions: PP1, node-local DP/PCP sharing, no
EPLB/elastic EP, no embedding-across-DP mode and no THP mode. TP must divide the
Engram hash column count. Unsupported settings fail before conversion.

The path interface selects `ascend_engram_ssd`, enables CPU offload and enables
DP shared memory when DP times PCP is greater than one. Conflicting explicit loader or
Engram settings are rejected. Do not combine `path` with `manifest`,
`artifact_version`, `table_paths`, `total_host_budget_bytes` or
`cache_payload_bytes`; use `host_memory_bytes` for the managed budget.

Existing configurations with explicit manifest/version/table paths continue to
work unchanged and bypass automatic preparation. Existing manually packed files
are not automatically imported into the managed directory; keep their explicit
configuration to reuse them, or allow a one-time managed conversion.

## Prefill context parallelism

PCP partitions token lookup work while preserving each request's n-gram history.
For TP1/PCP8/DP1, all eight ranks share one host payload, cache directory and SSD
I/O engine. They do not allocate eight caches or create eight table artifacts.
The existing `--prefill-context-parallel-size` flag selects PCP; no additional
SSD setting is needed. With explicit legacy configuration, set
`engram_config.dp_shared_memory=true` whenever DP times PCP exceeds one.

All workers finish initialization before the engine activates the shared-cache
protocol. Small caches use common token windows bounded by the whole group's
row count, including empty ranks. Each rank completes UVA reads before joining
the next window. Large prefill slot responses use per-rank scatter instead of
broadcasting every rank's slots to every consumer. PCP does not remove the hash
D2H dependency or guarantee a proportional reduction in SSD I/O time.

## Layerwise lookup overlap

Set `engram_ssd.layerwise_overlap` to `true` to enable staged lookup with
`--enforce-eager` or `FULL_DECODE_ONLY` graph mode. This requires
`VLLM_USE_V2_MODEL_RUNNER=1`, TP1, PP1, and node-local DP or PCP sharing.
The cache must hold the worst-case rows of a full group batch. The option
defaults to `false`.

The auxiliary NPU stream computes the hash and transfers both layers' row IDs
to host memory. A background worker waits for that transfer, gathers the row
IDs across ranks, and holds one cache lease for both layers. As the slots for
each layer arrive, the worker submits that layer's H2D and UVA gather to the
auxiliary stream. Each model layer waits for its own gather event before
reading the output. The lease is recycled after all ranks finish their UVA
reads. In FULL decode replay, the captured graph waits on an external NPU
event for each layer while the background worker submits the gathers.
Capture records the waits directly so they are included in the FULL graph.
Other steps in `FULL_DECODE_ONLY` mode use the synchronous SSD path.

The owner logs `engram-ssd-layered` with each layer's fill time, time through
submission of both gathers, and time from submission to retirement. Under
this option `protocol_ms` in the shared row-cache log ends at gather
submission. The `engram-ssd-layer-consumer` log measures host wait until the
gather event is submitted. Device-side wait time is measured separately in
an NPU trace.

## Prefill chunk lookahead

Set `engram_ssd.prefill_lookahead` to `true` together with
`engram_ssd.layerwise_overlap` to prefetch one chunk for a single prompt under
eager Model Runner V2. This mode requires TP1, PP1, DP1, DCP1, and PCP greater
than one. The first chunk starts on demand. The next chunk's hash and host
transfer can start before the current lookup retires. After all PCP ranks
finish the current chunk's UVA reads, its host cache slots can be reused to
fill the next chunk while the current model forward is still running. The
producer issues one layer at a time in model order (layer 1, then layer 14 in
DeepSeek V4.1 Flash).

Each rank keeps space for one prefetched chunk's lookup results on the NPU,
plus pinned staging for its row IDs and slots. A result is consumed only if
the scheduler presents the exact same request and token range. If the batch
composition or chunk boundary changes, the worker retires that prefetch and
resumes on-demand lookup. The owner logs `engram-ssd-prefill-lookahead hit`
or `discard` for these decisions. A hit identifies a matching chunk; it does
not measure remaining wait or imply that the SSD read has completed. The
option defaults to `false`.

## Shared row-cache statistics

The owner emits `engram-ssd-cache scope=owner_group` logs for large prefill
chunks, the first three protocol rounds, and every 200 rounds. Each line covers
one lookup round across that owner's DP/PCP group, not the entire service or a
single consumer rank:

- `rows`, `hits`, and `miss_queries` count query occurrences, including duplicate
  row IDs. `hit_rate` is `hits / rows`; an empty round reports `n/a`.
- `unique_misses` counts distinct missing rows requiring SSD fill. Duplicates
  within a cold batch reduce I/O but do not count as pre-existing cache hits.
- `evictions` counts displaced cache entries; `cache_ms` measures native cache
  planning, while `fill_ms` measures the miss-fill hook, including extent
  planning and completion of payload writes to DDR. It is not device-only SSD
  latency. `protocol_ms` covers the whole CPU protocol, including peer waits.

The cache is the Engram DDR row cache, independent of the KV prefix cache.
Non-owner runtime statistics use `None` for unknown hit/miss/fill values and
`cache_scope=owner_group_unavailable`; a non-owner does not report zero misses
as evidence of a hit. Owner runtime statistics use `cache_scope=owner_group`,
with `cache.rows` as the group denominator; the outer `rows` remains local.

Protocol metadata buffers grow to the largest batch seen and are then reused.
Gather receives into contiguous views and scatter sends int32 slot views,
avoiding per-round stacking, duplicate slot buffers and dtype round trips.
