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
table. Logs distinguish `Preparing Engram SSD table` from
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

The budget is split among TP owners. DP replicas on the same node share each
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
existing SSD runtime restrictions: PP1, PCP1, all DP replicas on one node, no
EPLB/elastic EP, no embedding-across-DP mode and no THP mode. TP must divide the
Engram hash column count. Unsupported settings fail before conversion.

The path interface selects `ascend_engram_ssd`, enables CPU offload and enables
DP shared memory when DP is greater than one. Conflicting explicit loader or
Engram settings are rejected. Do not combine `path` with `manifest`,
`artifact_version`, `table_paths`, `total_host_budget_bytes` or
`cache_payload_bytes`; use `host_memory_bytes` for the managed budget.

Existing configurations with explicit manifest/version/table paths continue to
work unchanged and bypass automatic preparation. Existing manually packed files
are not automatically imported into the managed directory; keep their explicit
configuration to reuse them, or allow a one-time managed conversion.
