# DeepSeek-V4 Layerwise KV Offload

This experimental adapter enables the existing DeepSeek-V4 Flash and Pro
implementations to reuse device KV buffers through `AscendStoreConnector` and
Memcache. Compressed history, index history, sliding-window attention (SWA),
and compressor state are eligible for rotation. No model implementation is added.

## Configuration

Prepare the Memcache services and SDK using the
[offloading guide](layerwise_and_sparse_kv_cache_offloading.md#1-preparation).
For single-instance validation, pass this JSON to `--kv-transfer-config`:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "use_layerwise": true,
    "layerwise_num_shared_buffers": 3,
    "layerwise_independent_layers": [0, 1]
  }
}
```

Use `--enforce-eager`, or explicitly configure `PIECEWISE` graphs. FULL graphs,
speculative decoding, and TurboQuant KV transfer are rejected for this adapter.
This experimental single-instance path is separate from the sparse Decode/PD
deployment described in the general offloading guide. It restores layer data
during Decode too; greater cache capacity does not imply lower TPOT.

`layerwise_num_shared_buffers` is a configurable slot count, not a fixed model
constant. `layerwise_independent_layers` may name any valid layers, including
middle layers; negative indices count from the end. The example keeps the first
two layers resident, but does not require that choice. Use `"all"` for a resident
control with the same connector and Memcache configuration. Omitting the shared
buffer count also leaves the native device layout in place.

## Cache ownership and prefixes

The common planner retains native cache groups and block IDs. Core computes
the physical bytes per global block from a one-block plan and chooses the block
count from each worker's memory budget. Workers then apply the rotation layout
before allocation. Entries with different specs retain their own typed views;
sharing storage does not merge their logical identities.

The adapter adds the following contracts:

- Public prefixes require every reachable component, including SWA and state
  at the matched boundary. Missing components cause fallback or recomputation.
  A narrow scheduler lookup wrapper bounds local hash hits before block adoption;
  the ordinary external-hit callback runs too late to reduce a local hit.
  MultiConnector forwards the strictest participating child's limit. This wrapper
  can be removed when upstream provides a connector-local-lookup hook.
- A continuation restores its previous-step private snapshots. Those pages
  follow current request ownership, rather than the public-prefix load mask.
- NULL block IDs are never copied. Released window pages are removed from the
  request's snapshot index; that index is distinct from public Memcache keys.
- A5 indexer data and scale views are transferred once. The overlapping full-page
  view remains available to model kernels and is not transferred again.
- A logical-format digest separates incompatible wire formats while allowing
  resident and rotated instances with different device capacities to share keys.
- Public load metadata is batched across groups and requests. Private keys remain
  scoped to their request, and every successful lease acquisition is released.

The adapter reuses the common planner, allocation, transfer threads, and reuse
fences. It does not bypass save completion or enable speculative overwrite.

## Validation and merge gates

The CPU regression tests are part of the normal `tests/ut` CI discovery:

```bash
pytest -q \
  tests/ut/models/test_dsv4_layerwise.py \
  tests/ut/distributed/ascend_store/test_dsv4_layerwise_runtime.py \
  tests/ut/distributed/ascend_store/test_dsv4_prefix_offload.py \
  tests/ut/distributed/ascend_store/test_reuse_snapshot_blocks.py \
  tests/ut/distributed/ascend_store/test_pool_worker.py
```

They cover both model layouts, capacity, typed views, reference attention after
restore, prefix fallback, NULL pages, private continuation, lease cleanup,
and the configuration example above. CPU tests do not validate NPU DMA, graph
execution, or full-model accuracy.

Before promoting this feature from draft, repeat Flash and Pro NPU validation
on the final source revision: resident versus rotated output-token comparison,
old-window prefix reuse, chunked Prefill and Decode continuation, eager and
PIECEWISE execution, and bounded-pool failures. Cross-instance and P/D support
require their own end-to-end validation and are not established by these UTs.
These hardware checks are necessary because mocks cannot verify asynchronous
NPU DMA, graph hooks, or rank-to-rank publication ordering.

For performance, compare the resident control with K=2/3/4 and identical model,
prompt tokens, chunk size, parallelism, and connector configuration. Measure
128K input with 256 output tokens, first at concurrency 1, then at 2. Run both
equal KV-budget and normal automatic-budget controls. Record TTFT, TPOT,
throughput, global block count, peak HBM, and prefix-hit counters. Capture a
separate profiling run and distinguish DMA execution, host preparation,
dependency waits, and actual overlap. Metadata batching is a candidate
optimization; a latency improvement is not claimed without paired NPU results.
