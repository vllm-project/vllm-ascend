# L2 pHash Optimization and the Adapted Interface

Updated: 2026-11-26.

This document describes the current implementation of the L2 similarity cache
(pHash + SSIM). The design follows a CPU pixel-path simplification: image and
video pixel tensors together with grid metadata live on the CPU, `mm_hash` is a
CPU-side Python key, and encoder embeddings keep the existing NPU path.

## Component Overview

| Component | Responsibility |
| --- | --- |
| `phash_index.py` | Tensor-independent pHash primitives: `hamming()`, `bands()`, `PHASH_BITS`, and `PHASH_NUM_BANDS`, shared by schedulers and the index service |
| `similarity_cache_config.py` | Parsing and validation of `ec_connector_extra_config.similarity_cache` (distance threshold, SSIM parameters, eviction watermarks, shared-index endpoint, and so on) |
| `similarity_index.py` | The `SimilarityIndex` protocol and its in-process implementation `LocalSimilarityIndex`: the three metadata maps, the band index, random eviction, and statistics accumulation |
| `similarity_index_service.py` | ZMQ server/client for the cross-DP shared index; `create_similarity_index()` selects the local or shared mode from the configuration |
| `cache_statistics.py` | Hit-count structure (exact/similarity/miss) and the cross-scheduler cumulative accumulator |
| `perceptual_ssim_cache_matcher.py` | `PhashSSIMCacheMatcher`: pHash pre-registration, recall, SSIM scoring, and match caching; owns a `SimilarityIndex` |
| `memcache_resized_tensor_store.py` | Resized-pixel key construction, transfers, and SSIM scoring against Memcache |
| `ec_memcache_connector.py` | Scheduler-side orchestration: exact check → similarity prewarm → post-alloc resolution → hit-statistics reporting |

## Single Band-Count Setting

The pHash constants moved from `phash.py` into
`vllm_ascend/distributed/ec_transfer/phash_index.py`:

```python
PHASH_NUM_BANDS = 4
```

The current 64-bit pHash is split into 4 bands of 16 bits each. This constant
also determines the default band count of `bands()`, the default `max_hamming`,
and its allowed upper bound. Both index writes and queries call `bands(phash)`
without specifying a band count, so changing this constant alone adjusts the
segmentation everywhere.

| Bands | Bits per band | Default `max_hamming` | Allowed range |
| --- | --- | --- | --- |
| 2 | 32 | 1 | 0–1 |
| 4 | 16 | 3 | 0–3 |
| 8 | 8 | 7 | 0–7 |
| 16 | 4 | 15 | 0–15 |

The default is `PHASH_NUM_BANDS - 1` and the invariant is
`0 <= max_hamming < PHASH_NUM_BANDS`. Both direct construction of
`SimilarityCacheConfig` and parsing via `from_vllm_config()` enforce this. Under
the current four-band scheme, legacy values of 4–7 (including the old default 5)
raise a parameter error and must be lowered to 0–3. No environment variable was
added. The shared-index handshake carries `num_bands` so that clients with
different segmentations are rejected.

Notes when changing the band count:

- Valid band counts are positive integer divisors of 64: 1, 2, 4, 8, 16, 32, 64.
  Other values cannot be split into equal-width bands.
- An explicitly configured `max_hamming` never changes automatically. For
  example, after switching to 8 bands a startup config that still says 3 keeps
  the threshold at 3; only an unspecified parameter picks up the new default of 7.
- The band count does not affect the computation or packing of the full 64-bit
  pHash. With a fixed distance threshold, more and shorter bands usually mean
  more candidates sharing at least one band and more index members; more bands
  do not necessarily make retrieval faster.

## Why Band Retrieval Replaces the Full Scan

The previous implementation iterated `phash_to_resized_mm_hashes.items()`,
computed the Hamming distance against every stored pHash, and collected the
matching image keys.

The current implementation, `LocalSimilarityIndex._recall()` reached through
the `batch_recall()` entry point, first narrows the candidate set with the band
index and then applies the same full Hamming-distance test:

```python
candidate_phashes: set[int] = set()
for band in bands(query_phash):
    candidate_phashes.update(self._phash_band_index.get(band, ()))

return [
    key
    for phash in sorted(candidate_phashes)
    if hamming(query_phash, phash) <= self._config.max_hamming
    for key in self.phash_to_resized_mm_hashes[phash]
]
```

### Why Recall Is Complete

Split the full pHash into B equal-width bands. If two pHashes differ in at most
τ bits and τ < B, the differing bits can touch at most τ bands, so at least one
band is identical. The index is keyed by `(band index, band value)` and stores
the set of full pHashes owning that band. Querying every band bucket and taking
the union therefore covers all pHashes within the distance requirement; the
`set` deduplicates.

For example, under the four-band scheme each band is 16 bits; in hexadecimal:

```text
Query:    0000 | 0000 | 0000 | 0000
Candidate: 0001 | 0001 | 0001 | 0000
```

The first three bands each differ in one bit, for a total Hamming distance of 3;
the fourth band is identical, so the candidate is always recalled through the
fourth bucket. If all four bands differ in one bit, the distance is 4 and no band
may match; this is exactly why `max_hamming >= 4` is disallowed under four bands
when only identical-band buckets are queried.

Sharing a band only means the candidate is worth checking; the remaining bits
may still differ widely. The full `hamming()` test afterwards removes those
false candidates, so the effective distance condition is unchanged. Expanding
each pHash's key list in ascending pHash order preserves the ordering of the old
sort-everything-then-filter approach. This equivalence assumes the same
registered data and requires the full map and the band index to be maintained
together (registration and eviction both happen inside the index, see below).

### Exact Mode, Index Maintenance, and Performance Boundaries

- `max_hamming == 0`: only a fully identical pHash qualifies, so the
  implementation looks the key up in the dictionary and returns a copy of the
  key list; it never scans the table, computes Hamming distances, or fills the
  band index.
- `1 <= max_hamming <= 3`: under the four-band scheme the query touches 4 band
  buckets, merges the candidates, and computes the full distance. There is no
  fallback to a full scan; correctness of the segmented recall follows from the
  Hamming-distance restriction.
- Repeated pHashes within one batch are deduplicated with `dict.fromkeys()` and
  reuse the same recall result while preserving the original repetition count
  and order of the output.
- Both registration and eviction maintain the band index inside the index:
  new pHashes register their bands, and evicting a pHash removes all of its
  band entries (see capacity eviction below).
- An identical or near-identical pHash still has to pass the layout
  compatibility check and SSIM; it never implies reusable embeddings by itself.

The old version computed distances against all N distinct pHashes; the new one
only computes distances for the C candidates sharing at least one band, paying
for bucket reads, set merges, and sorting. Usually C is smaller than N; when
hashes cluster, C can still approach N, so no fixed speedup is promised. In
nonzero-distance mode the four-band index adds set memberships worth about
4x the number of distinct pHashes. Index state lives only in memory (a local
instance or the shared service process) and is not persisted; after a service
restart the metadata is rebuilt from scratch as images are requested again.

## Ownership of Similarity Metadata: the SimilarityIndex Abstraction

The three metadata maps are no longer owned directly by the matcher; they are
hidden behind the `SimilarityIndex` protocol:

| Protocol method | Purpose |
| --- | --- |
| `batch_get_entries(mm_hashes)` | Batch-read `SimilarityCacheItem` snapshots (deep copies; callers cannot mutate internal state) |
| `batch_register_phashes(entries)` | Atomically publish `(mm_hash, phash, resized_key)` triples in bulk |
| `batch_recall(phashes)` | Batch band recall; returns the resized key list for each queried pHash |
| `record_similarity_scores(mm_hash, scores)` | Record one SSIM scoring result (written symmetrically and sorted) |
| `report_cache_statistics(scheduler_id, counts)` | Report this instance's cumulative hit counts and return the global snapshot |
| `close()` | Release resources held by the index (a no-op for the local implementation) |

`LocalSimilarityIndex` is the in-process implementation of the protocol:

- All maps are protected by a single lock; every method returns snapshots
  detached from internal state, so readers never observe partially updated data.
- SSIM results are recorded **symmetrically**: while storing the candidate score
  for `mm_hash`, the same score is written into the candidate's
  `ssim_similar_items`, so later lookups in either direction hit the cached
  score. Self-matches (candidate == mm_hash) are skipped.
- Scores are sorted by `(-score, key)` at write time (non-finite scores last),
  so readers never re-sort.
- When the same image is computed concurrently, the first pHash wins and later
  results are subsumed by the first entry.
- `SimilarityCacheItem` also stores `resized_keys` ownership information so
  eviction can clean the reverse index atomically without parsing tensor keys.

The matcher receives the index through `__init__(index=...)`; when omitted it
creates a local instance.

## Cross-DP Shared Index

Previously every DP scheduler kept its own local metadata, so images registered
by DP0 were invisible to DP1. The shared index consolidates the metadata into
one service process so that all DPs share the same pHash/SSIM records.

### Configuration

| Field | Default | Meaning |
| --- | --- | --- |
| `shared_index_endpoint` | `None` | A `tcp://` or `ipc://` address; `None` means no sharing (local index) |
| `shared_index_namespace` | `"default"` | Deployment namespace, must match across all DPs of one deployment; contains no DP rank |
| `shared_index_mode` | `"embedded"` | `embedded` starts the service inside the serving DP0 scheduler; `external` connects to a separately managed service |
| `shared_index_timeout_ms` | `1000` | Send/receive timeout per request; the client backs off for the same duration before retrying |

### Modes and Creation

`create_similarity_index()` selects the implementation from the configuration:

- endpoint `None` → `LocalSimilarityIndex`;
- endpoint set and mode `embedded` → the scheduler at DP index 0 embeds an
  `EmbeddedSimilarityIndexServer`, and every DP (including DP0 itself) talks to
  it through the client;
- endpoint set and mode `external` → every scheduler acts as a client of an
  externally managed service, which can be started standalone:

  ```bash
  python -m vllm_ascend.distributed.ec_transfer.similarity_index_service \
      --endpoint tcp://127.0.0.1:5558 --namespace my-model
  ```

When the DP size is greater than 1 and no endpoint is configured, the connector
logs a warning at startup that metadata is not shared.

### Protocol and Consistency Validation

Client and server speak ZMQ REQ/REP with msgpack. Requests carry the protocol
version (currently 2), the namespace, and settings. The settings include
`model_id`, `merge_size`, the algorithm version (`phash64-ssim-v1`, bumped when
pHash preprocessing, resized-key semantics, or SSIM scoring change), `num_bands`,
and the matching parameters such as `max_hamming`/`ssim_threshold`. The server
validates these fields and rejects any mismatch with a configuration error,
preventing semantically divergent clients from polluting each other's metadata.

### Failure Semantics

The client caches neither positive nor negative results, and all calls belong on
the scheduler thread:

- When the shared index is unreachable the client raises
  `SimilarityIndexUnavailable` and callers treat it as a cache miss; the service
  is an accelerator, not a correctness dependency;
- After a failed request the client destroys the socket and backs off for
  `shared_index_timeout_ms` before reconnecting;
- Timed-out sockets are discarded, so a late reply can never be mistaken for the
  response of another request;
- All mutations are idempotent, so retries never double-register or double-count;
- A failed startup ping only logs a warning; the client keeps working through
  cache misses and recovers automatically once the service returns.

Non-owning DPs hold a non-owning reference to the index: the matcher's
`shutdown()` closes the index and the embedded server, and the connector does
not release them a second time.

## Capacity Limits and Random Eviction

Similarity metadata grows with the number of images, and unbounded growth would
degrade retrieval and memory usage. The current implementation adds
watermark-based random eviction:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_capacity` | 1,000,000 | Capacity baseline for metadata entries |
| `eviction_threshold` | 0.95 | Eviction starts when entries reach `max_capacity * 0.95` |
| `lower_watermark` | 0.90 | After eviction the count shrinks back to `max_capacity * 0.90` |

Eviction runs synchronously inside the registration path
(`batch_register_phashes`), checking the watermark after every registration.
Once the threshold is hit, victim entries are sampled at random until the count
falls to the low watermark. Cleanup of each victim is atomic:

1. Remove the `SimilarityCacheItem` from `similarity_cache`;
2. Remove the resized keys owned by the entry from `phash_to_resized_mm_hashes`;
   if a pHash's key list becomes empty, delete the pHash entry;
3. When a pHash entry is deleted, discard the pHash from every band bucket;
   delete buckets that become empty to avoid empty-bucket buildup.

Eviction removes **metadata only**; resized tensors and embeddings in Memcache
are never deleted here, and data lifetime remains governed by Memcache itself.
After eviction, `ssim_similar_items` may still reference evicted images, but any
reuse is verified against Memcache key-info first, so a dangling reference cannot
produce a false hit; it only costs one existence check. Hit statistics are
independent of eviction: counters record historical decisions and do not
disappear when metadata is removed.

## Cross-DP Aggregation of Hit Statistics

Every connector instance maintains three monotonically increasing counters:
exact hit (the request's original mm_hash found in cache), similarity hit
(an embedding reused from a similar image), and miss. Derived metrics include
the overall hit rate, the similarity hit rate, and the similarity lookup hit
rate (`similarity_hits / (similarity_hits + misses)`).

When a shared index is configured, the connector uses an in-process uuid as its
`scheduler_id` and reports cumulative counts through the index's
`report_cache_statistics()`. The server-side `CacheStatisticsAccumulator`
aggregates with these rules:

- Per scheduler, the historical maximum is kept and merged into the totals, so
  RPC retries or delayed stale snapshots never double-count; a restarted
  scheduler gets a new ID, and the completed counts of the old ID remain in the
  totals;
- The aggregate is returned as a `CacheStatisticsSnapshot` (totals, number of
  reporting schedulers, epoch); the epoch is fixed for the index lifetime and
  separates statistics windows across service restarts;
- The connector performs one final report at shutdown to flush the last counts.

Statistics are logged periodically: with a shared index the `scope=global`
cross-DP summary is printed, otherwise the `scope=local` numbers of this
instance. Without a shared index the statistics remain local.

## Computing pHash, Deferred Persistence, and Index Publication

The pHash lifecycle has two phases.

The `ensure_cache_available()` phase performs metadata pre-registration.
Requests in the decode stage (`num_computed_tokens >= num_prompt_tokens`) skip
the whole check to avoid probing Memcache on every decode step; paths not
covered by it, such as continued chunk scheduling, still fall back to the direct
L1 lookup in `has_cache_item()`. For exact misses without a pHash the matcher
first classifies the entries: cached SSIM matches are reused directly, entries
with a pHash go through band recall, and the rest have their pHash computed and
published to the index in bulk. Computation runs on the local thread pool
(batches of up to 3 run on the calling thread; larger batches use at most
8 workers) and is published with **one atomic** `batch_register_phashes()` call
after all computations finish; no index RPC runs on the compute pool. What is
published is the mapping from pHash to deterministic resized key; the resized
pixel tensor is intentionally not written to Memcache yet. Candidate keys may
temporarily have no Memcache value, and the read path skips missing values via
the key-info check. Finally all candidate keys are prewarmed with
`backend.exists()` to populate the existence cache of the Memcache client.

After `update_state_after_alloc()` completes final matching, if the result is
still a save miss, the connector calls
`persist_missed_resized_tensor(mm_hash, feature)` before adding the image to
`mm_hashes_need_saves`. Matching has finished at this point, so the newly
persisted and indexed tensor only becomes a candidate for future requests,
never for itself.

### Extracting, Computing, and Caching pHash Metadata

1. `extract_resized_tensor(feature)` extracts the preprocessed pixel tensor and
   `extract_image_grid(feature)` extracts the `(T, H, W)` patch grid; either
   being missing skips the entry.
2. `compute_phash(resized_tensor, grid, merge_size=...)` computes the full
   64-bit pHash as a Python integer. When the data does not meet the
   requirements the result is `None` and the entry is skipped; bands are not
   assigned at this point.
3. The deterministic resized key is generated, and the index publishes
   `phash_to_resized_mm_hashes`, the band reverse index, and the
   `similarity_cache` metadata inside the index's internal lock through
   `batch_register_phashes()`; `_tensor_store.put()` is not executed in this
   phase.

### Persisting Pixels and Publishing the Index After a Final Miss

`persist_missed_resized_tensor()` reuses the pre-registered pHash when
available, otherwise recomputing it from the current feature. It then:

1. Calls `_tensor_store.put(mm_hash, resized_tensor, grid)` to confirm the
   pixels already exist or were written to Memcache. On `False` the pre-registered
   dangling index entries are kept; the caller logs the failure but still saves
   the embedding as usual.
2. Calls `get_resized_key(...)` to build the storage key that encodes the image
   identity, grid, shape, dtype, and model information; layout compatibility
   checks and pixel reads are based on this key.
3. After the pixels are stored, confirms the reverse index registration through
   `batch_register_phashes()` inside the index.

`mm_hash` is the original image identity, `image_phash` is the perceptual hash
used for the coarse similarity prefilter, and `resized_key` is the storage key
of the resized pixels in Memcache; the three serve different purposes.

### The Three Related States Inside the Index

| State | Content | Purpose |
| --- | --- | --- |
| `similarity_cache` | `mm_hash` → `SimilarityCacheItem` (pHash, known SSIM matches, resized key ownership) | Reuse per-image results; ownership source for eviction |
| `phash_to_resized_mm_hashes` | full pHash → resized key list | Map one perceptual hash back to all matching images |
| `_phash_band_index` | `(band index, band value)` → set of full pHashes | Recall candidates sharing a band |

Different images can share one pHash, so the full map's value must be a key
list. When a pHash is seen for the first time an empty list is created and, in
nonzero-distance mode, all of its bands are registered; identical pHashes have
identical bands, so bands are registered only on first sight. Keys are checked
for duplicates before appending. The first registration of an `mm_hash` creates
a `SimilarityCacheItem` with an empty SSIM match list; existing records are kept
and previously computed SSIM results are never cleared.

## CPU Preprocessing and Transfers

`perceptual_ssim_cache_matcher.py` no longer has the `has_device_inputs` check.
pHash preprocessing is scheduled purely by batch size: up to 3 items run on the
calling thread, larger batches use at most 8 workers. This simplification relies
on pixels and grids being on CPU.

Inside `memcache_resized_tensor_store.py`:

| Operation | Current behavior |
| --- | --- |
| PUT | Compact into a contiguous CPU tensor, always `COPY_H2G` |
| Regular receive buffer | Lazily allocated CPU buffer, reused afterwards; sized by `resized_image_recv_buffer_size` (default 512 MB, allowed 4 KB–512 MB) |
| Oversized temporary receive buffer | CPU buffer allocated and deregistered after use; tensors above `MAX_RESIZED_IMAGE_TRANSFER_SIZE` (1 GB) are excluded from optional matching |
| GET | Always `COPY_G2H` |
| Direction helper | `get_transfer_direction` keeps the CPU/NPU mapping, but no production call site uses it |

The SSIM `data_range` is configurable through `ssim_data_range` (default 3.7),
sharing the same constant with the SSIM implementation.

### Why the Receive Buffer Is on CPU

```python
recv_buffer = torch.empty(
    self._recv_buffer_size,
    dtype=torch.uint8,
    device="cpu",
)
```

This buffer receives resized image pixels for the CPU SSIM comparison. `uint8`
means the space is allocated in bytes and `self._recv_buffer_size` is in bytes;
after loading, the data is interpreted with the original image dtype and shape,
which does not require the image dtype to be `uint8`.

| Data received | Target device | Memcache read direction |
| --- | --- | --- |
| resized image pixels | CPU | `COPY_G2H` |
| encoder embedding | NPU | `COPY_G2L` |

In the direction enum G denotes Memcache global memory, not a PyTorch GPU
device. Moving the receive buffer to an NPU would require adjusting the
transfer direction and the downstream compute path together; changing only
`device` is not enough. The fixed host direction applies to resized pixel
tensors only; the NPU storage path for embeddings is unchanged.

Reads keep cloning the output so buffer reuse cannot overwrite candidates
already fetched; error paths keep their cleanup. The `_synchronize_npu` defense
remains but never executes on the current CPU data path. The FP32 arithmetic,
constant caching, and 8-byte hash packing of the pHash/SSIM numeric functions
are unchanged.

## Why PUT Keeps Its Lock

`self._put_lock = Lock()` protects the full PUT registration lifecycle of one
store instance:

```text
acquire lock → register buffer → transfer pixels → deregister buffer in finally → release lock
```

The CPU data convention does not replace this lock. Batch pre-registration now
only computes pHash metadata in parallel; the final
`_tensor_store.put()` after a miss runs on the scheduler thread. Keeping the
lock defends against future concurrent PUTs and concurrency introduced by other
call paths of the store.

Two concrete reasons:

1. `MemcacheBackend.register_buffer()` stores pending addresses in the shared
   `self._pending_buffers`, which `_register_buffers_if_needed()` reads and
   clears. That wrapper provides no complete concurrency protection, and
   interleaved calls risk overwriting each other's state.
2. `tensor.detach().to(device="cpu").contiguous()` does not guarantee a copy.
   `detach()` leaves the autograd graph but shares the underlying storage, and
   when the input is already a contiguous CPU tensor the following steps may
   still reuse it. If two calls use the same address, one call's deregistration
   could race with the other's still-running transfer.

The local `memcache-release-1.2` SDK uses internal mutexes for registration and
deregistration individually, but that does not make register-transfer-deregister
atomic as a whole, nor does it protect the upper-layer Python registration state.

The cost is that PUT critical sections on one store serialize; pHash metadata
computation stays parallel. The lock and the deregistration cleanup on transfer
errors are both retained.

## Thread Pool, Futures, and Exception Waiting

CPU SSIM scoring may also use the thread pool when there are more than 3
candidates. The code keeps the device defense check: the parallel branch runs
only when the query gray plane and all candidate tensors are on CPU; device
inputs stay on the calling thread. The relevant code:

```python
futures = [
    self._similarity_compute_executor.submit(
        self._score_candidate,
        current_gray_plane,
        resized_key,
        candidate_tensor,
    )
    for resized_key, candidate_tensor in zip(score_resized_keys, score_tensors)
]
wait(futures)
score_results = [future.result() for future in futures]
```

- `zip(...)` pairs candidate keys with their pixel tensors and submits one
  `_score_candidate()` per pair. Every task reads the same query gray plane
  `current_gray_plane` and compares it against its own candidate.
- `submit()` returns a Future, the handle of a task, not a finished score. The
  pool runs as many tasks as it has workers and queues the rest.
- `wait(futures)` waits for all tasks to complete, including those that ended
  with an exception; it never raises the tasks' exceptions itself.
- `future.result()` retrieves the return value, currently
  `(ssim_score, candidate_mm_hash)` or `None`; if the task failed, the exception
  is re-raised here. `None` results are filtered later.
- Results are collected in submission order, independent of thread completion
  order and not sorted by SSIM score.

Waiting for everything before reading results guarantees that when an exception
propagates, the other tasks have already finished. Calling `result()` directly
in a loop would let earlier failures start outer error handling while other
tasks are still reading tensors. Waiting for all tasks makes the end boundary
explicit before cleanup begins. The pHash preprocessing branch follows the same
wait pattern; no index RPC runs on the compute pool, and every metadata
publication is one bulk operation after all computations complete.

## SSIM Threshold, Ordering, and Early Stop

Loading, batch construction, and scoring continue to use the
`ResizedTensorLoadRequest` fields `output_index`, `resized_key`, and `nbytes`,
with output indices renumbered per batch. The model/grid/shape/dtype key format
and layout compatibility checks are unchanged.

The final acceptance condition is the strict `score > ssim_threshold`,
consistent with cached lookup, warmup classification, and per-batch early stop.
In `_get_ssim_score_list()`, as soon as a regular batch contains an accepted
candidate (`regular_scores[0][0]` above the threshold, with the list already
sorted by score) it returns immediately without loading or comparing oversized
candidates. Stopping at the first batch that contains a qualifying result
selects the best item among the scored range and does not promise the global
optimum over unread candidates.

### What the Current Sorting Is For

Both regular and oversized batches currently execute:

```python
calculated_scores.extend(batch_scores)
calculated_scores.sort(key=lambda item: (-item[0], item[1]))
if batch_scores and max(score for score, _ in batch_scores) > ssim_threshold:
    # Log the hit and return calculated_scores.
    ...
```

Sorting places higher SSIM scores first, breaking ties by candidate `mm_hash`
for deterministic selection. `max(...)` itself does not require a sorted list;
the `batch_scores` check before it avoids calling `max()` on an empty list.
However, other code relies on the first item being the best:

- The current hit log reads `calculated_scores[0]`.
- `_get_ssim_score_list()` uses `regular_scores[0][0]` to decide whether
  oversized loads can be skipped.
- `find_and_cache_matched_ssim()` uses `ssim_score_list[0]` to pick the
  embedding key reused by this request.
- Cache writes are also sorted by score and key so later consumers can pick the
  highest-scoring qualified candidate in order (scores enter the index through
  `_record_similarity_score()`, and the index sorts by the same `(-score, key)`
  at write time, keeping both entry points consistent).

Therefore this sort line cannot simply be deleted while keeping all
first-item accesses unchanged. For example, given a returned order of 0.991 and
0.999 with a threshold of 0.99, the first item qualifies but is not the best of
the scored candidates.
