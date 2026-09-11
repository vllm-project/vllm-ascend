# Mooncake Layerwise Adaptation and Optimization Analysis

## 1. Adaptation Baseline and Conclusion

- Baseline: `Eric-dot/vllm-ascend:mooncake` at local commit `0a023b094e9e88ffaca0b1fda02529cef6277f8e`.
- Reference implementation: `vllm-project/vllm-ascend#12418`.
- Local working branch: `eric-mooncake-pr12418-port`.
- Current status: The core implementation, CPU mock unit tests, and static checks have passed. End-to-end testing
  on real Ascend NPUs, Mooncake Master/Metadata services, and multiple TP processes has not yet been completed.

This adaptation does not directly copy the earlier PR. The personal branch already contains an updated
`metadata.py`, multi-cache-entry/MTP/SFA layouts, Memcache GVA layerwise transfer, asynchronous thread exception
propagation, and layer-buffer reuse. The Mooncake path therefore reuses these newer structures and adds an
independent key-major range/session protocol.

## 2. Why the Remote Object Count Drops from `block × layer × rank` to `block × rank`

The earlier per-layer key scheme treats each layer as a separate remote object:

```text
model@block_hash@layer_0@rank_0
model@block_hash@layer_1@rank_0
...
```

For `B` blocks, `L` layers, and `R` ranks that actually store data, the approximate object count is:

```text
B × L × R
```

Mooncake layerwise transfer creates only one object for each block/rank pair:

```text
model@block_hash@rank_0
```

All layers are placed contiguously inside the object according to the actual cache layout:

```text
object(block, rank)
├── layer 0: [cache entry 0][cache entry 1]...
├── layer 1: [cache entry 0][cache entry 1]...
└── ...
```

After each layer finishes computing, only the byte range for that layer is written to the same object. Layerwise
transfer changes the transfer timing and ranges; it does not require every layer to be a separate object. The object
count therefore becomes:

```text
B × R
```

This reduces the number of Mooncake metadata objects, keys, sessions, and `exists` query items, but not the total KV
payload size. KV data for every layer must still be stored.

## 3. Data and Control Flow in the Current Implementation

### 3.1 Saving

1. The scheduler calls `batch_is_exist` with all storing-rank keys for each block.
2. For missing keys, the worker calls `batch_put_session_start(keys, object_sizes, ReplicateConfig)`.
3. `LayerBatchBuilder` calculates the following values from the actual cache-entry layout:
   - The local NPU buffer address.
   - The size of each cache entry in the layer.
   - The destination offset of the layer within the remote all-layer object.
4. After each attention layer finishes, the sending thread calls `batch_put_from_multi_buffer_ranges`.
5. If a range write fails for one key, only that key is revoked; the remaining keys continue with subsequent layers.
6. After the last layer succeeds, the worker calls `batch_put_session_end`. It calls `batch_put_session_revoke` for
   keys that fail to commit.
7. Only successfully committed keys enter the readable set for subsequent chunked-prefill steps.

### 3.2 Loading

1. The scheduler counts only the contiguous prefix starting at block 0 for which every storing rank is `COMPLETE` as
   a cache hit.
2. The worker calls `batch_get_session_start` for the deduplicated remote keys.
3. Before each layer computes, the receiving thread calls `batch_get_into_multi_buffer_ranges` to load that layer's
   range into the local block.
4. A failure in one row records the corresponding local block ID and returns it to the scheduler for recomputation,
   preventing incomplete KV data from being consumed.
5. The worker calls `batch_get_session_end` after the last layer or when the request terminates. Exceptional paths
   release owners according to retry or terminal semantics.

### 3.3 Chunked Prefill

`MooncakeSessionTracker` maintains three types of relationships:

- Put keys that have not yet been committed and their request/block owners.
- Committed keys that later chunks of the same request can load.
- Open get-session keys and their request owners.

It guarantees that:

- An object is not considered readable before it is committed.
- A later chunk renews the lease and restores the previously committed prefix layer by layer, even when it has no new
  `load_spec`.
- A retry releases the get session while retaining the key/block relationships required for a subsequent retry.
- A terminal or preempt event clears request state.
- When multiple requests share the same remote key, get-end is executed only after the last owner releases it.

## 4. Adaptations and Corrections Relative to the Original PR

### 4.1 Actual Layout Offsets Instead of a Fixed `layer_id × page_size`

The current branch supports multiple cache entries in one physical layer as well as MTP/SFA layouts. Remote offsets
are calculated from `group_layer_cache_entry_offsets` and prefix sums of the actual `block_len` values:

```text
layer_object_offset = sum(block_len before this layer)
entry_offset = layer_object_offset + prefix_sum(entry sizes in this layer)
```

The remote object size therefore equals the total number of bytes across all cache entries for all layers on the
current rank. The implementation no longer assumes that every layer has the same size or incorrectly multiplies the
size by `num_layers` again.

### 4.2 PutStart Inherits the Mooncake Placement Policy

The session-start path in the original PR did not pass `ReplicateConfig`. The current implementation matches the
whole-key put path and passes:

- `preferred_segment`.
- `prefer_alloc_in_same_node`.

This prevents layerwise transfer from bypassing the existing local-first and same-node placement policies.

### 4.3 Transfer Throttling for Range Calls

- `layerwise_max_transfer_blocks` limits the number of key/block rows in a single range API call.
- `layerwise_max_transfer_bytes` splits an oversized contiguous segment into smaller ranges.

PutStart, GetStart, GetEnd, and scheduler `exists` queries are also batched according to the block limit. This avoids
oversized Python/C++ argument lists and transient metadata spikes for long prompts.

### 4.4 Fail Fast During Startup

Mooncake layerwise transfer checks all required session and range methods during startup. If an interface is missing,
startup fails immediately and reports that the client must include Mooncake PR #2881, instead of failing in an
asynchronous thread on the first request.

The current key schema encodes only the model, block hash, and TP/head rank, so the implementation explicitly rejects:

- Pipeline parallel size greater than 1.
- Prefill or decode context parallel size greater than 1.
- Hybrid or multi-group KV cache layouts.
- TP mismatch with layerwise transfer.

### 4.5 Exceptions and Fallbacks

- Batch results must align one-to-one with keys. Missing items, Boolean values, and non-integer results are rejected.
- A range exception publishes invalid-block and abort state before waking the compute thread.
- A put exception revokes `PROCESSING` objects.
- A get exception ends the session only after the range call is confirmed to have exited.
- A consumer-only worker advances the layer cursor in the load hook instead of depending on a save hook that will
  never be called.

## 5. Running the Feature

### 5.1 Mooncake Version

Install a Mooncake Python client that provides the following methods:

```text
batch_put_session_start
batch_put_from_multi_buffer_ranges
batch_put_session_end
batch_put_session_revoke
batch_get_session_start
batch_get_into_multi_buffer_ranges
batch_get_session_end
```

If the currently published wheel does not provide these interfaces, build Mooncake from source after PR #2881 has
been merged. The interfaces are checked automatically during startup.

### 5.2 Configuration Example

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --kv-transfer-config '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "backend": "mooncake",
      "use_layerwise": true,
      "layerwise_prefetch_layers": 2,
      "layerwise_max_transfer_blocks": 64,
      "layerwise_max_transfer_bytes": 16777216
    }
  }'
```

For the first hardware validation, start with TP=1, `layerwise_prefetch_layers=1`, and hybrid/CP/PP disabled. Then
gradually increase the TP size, block count, and prefetch depth.

During integration testing, temporarily set `VLLM_ASCEND_KVPOOL_RANGE_DEBUG=1` to emit JSON audit logs for whole-key
operations, per-layer ranges, and commits. Keep the default value of `0` during normal operation to avoid per-layer
logging overhead.

### 5.3 Pipeline Parallelism and Decode Context Parallelism

Mooncake range sessions can compose with PP, DCP, or both, using a single KV cache group. Each object contains only
the current PP stage's layers and the current DCP rank's token shard. Layer offsets and session completion are
stage-local: a later PP stage starts its first range at offset zero and commits after its own last layer.

The TP-only key format remains unchanged. When PP or DCP is enabled, keys use a versioned topology namespace:

```text
model@layerwise_v2@tp:T@pp:start0-end0,start1-end1@dcp:D@interleave:I@pp_rank:P@dcp_rank:C@hash@head_rank
```

The PP ranges come from vLLM's model partitioning API, including custom/uneven partitions. TP size, DCP size and
token interleaving also participate in the namespace. Incompatible topologies do not share objects: they produce
cache misses, not automatic cross-topology resharding. Configure the same topology, PP partition, model weights,
cache dtype and block size on instances that should share cached KV.

Within each replicated KV-head group, one rank per DCP shard saves data. For example, MLA with TP=4 and DCP=2
uses TP ranks 0 and 1 as writers; ranks 2 and 3 load the corresponding shard but do not duplicate writes. The
scheduler reports a block hit only when all PP stages, DCP shards and KV-head shards have committed that block.
Token/block accounting uses the DCP-scaled logical block size; range I/O uses the actual local buffer sizes.

Current boundaries:

- PCP remains unsupported and is rejected during initialization.
- DCP groups must fit within replicated KV-head groups, following vLLM's DCP constraints.
- Hybrid/multi-group layouts and prefill/decode TP mismatch remain unsupported.
- This is topology-matched KV pooling, not PP/DCP redistribution between heterogeneous producer/consumer layouts.
- Memcache behavior is unchanged. Its GVA protocol is not interchangeable with Mooncake range sessions.
- Keep the existing eager/PIECEWISE layerwise execution mode; this change does not add full-graph support.

#### Hardware Validation

The CPU regression tests cover all shard keys, stage-local offsets, byte-for-byte save/load, replicated-rank
ownership, incomplete-shard misses, and load-session cleanup. They do not validate NPU attention, HCCL ordering,
or real Mooncake transport. Run the following on an Ascend host with the required model, Mooncake range-session
client, a running Master, and `MOONCAKE_CONFIG_PATH` set:

```bash
# Four NPUs; use a single-group MLA model that supports the requested topology.
python tests/e2e/nightly/single_node/models/scripts/mooncake_layerwise_parallel.py \
  --model deepseek-ai/DeepSeek-V2-Lite --tp 2 --pp 2 --dcp 2 --enforce-eager
```

Also run PP-only (`--tp 2 --pp 2 --dcp 1`) and DCP-only (`--tp 2 --pp 1 --dcp 2`). Repeat without
`--enforce-eager` for PIECEWISE execution, then increase `--prefetch-layers` to 2. The script keeps the same
workers/pool alive, clears only the local prefix cache, requires a remote cache hit, and compares generated token
IDs for cold and repeated warm requests with chunked prefill. Use a fresh process for each topology. Separately
validate a compatible GQA model with sufficient TP replication (for example, two KV heads require TP>=4 for DCP=2).

Real-NPU validation and performance measurements for this PP/DCP extension are still required before deployment.

## 6. Follow-up Optimization Recommendations

### P0: Required Before Production

1. **Pin a minimum Mooncake commit or version**
   - Checking method availability prevents use of an outdated client but cannot detect incompatible ABI or return-code
     semantics.
   - Pin the exact commit containing PR #2881, or the first official release that contains it, in the installation
     documentation and CI images.

2. **Add a schema/layout fingerprint to the key namespace**
   - For compatibility with the original PR, the current implementation still uses the model basename.
   - Different revisions, data types, block sizes, or KV layouts with the same basename can collide.
   - Add a stable digest of the tenant, model revision, data type, block size, TP layout, and schema version to the key.

3. **Add real-NPU E2E tests and fault injection**
   - Cover at least TP=1/2, `kv_both`, P/D, chunked prefill, request preemption, single-key range failure,
     commit failure, and Master restart.
   - Verify that loading KV data produces exactly the same logits and tokens as local computation.

### P1: High-Impact Performance Improvements

1. **Adaptive prefetch depth**
   - A fixed `layerwise_prefetch_layers` value cannot adapt to different per-layer compute times or network jitter.
   - Dynamically control the window using the recent `transfer_time / compute_time` ratio, queue depth, and available
     buffers.

2. **Sliding session window**
   - The current implementation opens sessions for every block in the batch at once. Long prompts can still create
     many simultaneously active leases and sessions.
   - Open sessions only for a future layer or block window, then advance the window after completion to reduce Master
     state and timeout pressure.

3. **Reduce per-layer Python list construction**
   - The current implementation still creates `all_buffers`, `all_sizes`, and `all_offsets` for every layer.
   - Precompute a range template for each layer and apply only vectorized block-ID address offsets. The descriptors
     could eventually be cached in the C++/pybind layer.

4. **Incremental scheduler hit queries**
   - The current implementation queries every candidate block before locating the first miss.
   - Query in windows and stop at the first incomplete block to substantially reduce Master RPCs and key counts for
     long prompts with low hit rates.

5. **Dynamic batching based on total bytes and backend feedback**
   - The current block and segment-byte limits are static.
   - Jointly limit the total ranges and bytes per call, and adjust them automatically according to queue latency,
     return codes, and bandwidth.

6. **Local fan-out for shared keys**
   - When multiple requests hit the same remote block, one key can currently produce multiple remote-read rows.
   - Read into a shared staging buffer first, then copy to multiple target blocks locally. Whether this helps depends
     on remote bandwidth relative to local H2D bandwidth.

### P2: Further Evolution

1. Support hybrid and multi-group layouts by recording the group layout in the key and object header, with an
   independent completeness bitmap for each group.
2. Extend the topology-matched PP/DCP support to PCP and heterogeneous producer/consumer topologies, including
   cache redistribution and the corresponding attention callbacks.
3. Support a per-layer readiness bitmap so consumers can read completed early layers before the entire object becomes
   `COMPLETE`. This requires visibility and consistency support from Mooncake and introduces substantial complexity.
4. Record a schema, version, and checksum in the object header and perform an inexpensive compatibility check before
   loading to prevent silent layout mismatches.

## 7. Validation Record

### Initial Range-Session Adaptation

- `py_compile`: Passed for the core changed files.
- Ruff lint and format checks: Passed.
- Relevant CPU mock pytest suite: `285 passed, 106 subtests passed`.
- Strict `0`/`1` validation for `VLLM_ASCEND_KVPOOL_RANGE_DEBUG`: Passed.
- Not yet completed: Tests with real NPUs, a real Mooncake client and Master, multi-node networking, and performance
  benchmarks.

### PP/DCP Extension

- Base: upstream `main` at `f4b6bd05f05717e10e4521a78f4b3beba5bb5a99`.
- Focused AscendStore CPU mock regression: `373 passed, 181 subtests passed`, including the new parallel tests.
- The wider AscendStore run has two coordinator failures in this local mock environment; both also reproduce on
  the unmodified base. The cache-layout test module requires real torch dependencies unavailable on this host.
- Hardware smoke test added above, but not executed locally. NPU/Mooncake transport correctness, multi-process
  scheduling, graph execution and performance remain to be verified on Ascend hardware.
