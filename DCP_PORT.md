# Native DCP decode layout port

## Main refresh, 2026-09-21

This is the main continuation of [PR16349](https://github.com/vllm-project/vllm-ascend/pull/16349).
Main base: `d052552072254a21560e8c5ebc04b80524c89622`.
Published main predecessor: `f8186d340651c2b7491d4026559c7b339f27a6cb`.
Release donor: `d5e74d19426fd6e18b9cf1e7e8dd33a703bb1b9b`.
The update merges main without rewriting the published branch history.

| Area | Selection |
|---|---|
| Sparse-index remap | Keep upstream two-kernel pipeline; DCP8/interleave128 integer-boundary and empty-input repairs |
| O/LSE | Raw-bit pack, one AllToAll, FP32 merge and BF16 conversion; up to min(scheduler budget,192) tokens |
| Query | Head-major preparation and contiguous unpack; existing T2-12 guard unchanged |
| Indexer | Native INT8 quantization plus fused final key/scale writes; existing T1-12 guard unchanged |

**192 is a manually selected cap, not a demonstrated kernel limit. Larger
values have not been tested and retain the upstream fallback.** This change
does not raise the scheduler's configured token budget. Both exchange variants
already use one AllToAll; the gain is not a collective-count reduction.

No allocator, host-memory placement, SuperMem-specific serving configuration,
or experimental VMM peer exchange is included.

## Current main compatibility

- Keep main's indexer-owned RoPE metadata and three-value `forward_k` return,
  including the projection weights tail reused by top-k selection.
- Preserve PCP composition, token scatter, scatter-size1 and unsupported dtype/shape paths.
- Preserve main's `return_lse` and `defer_combine` modes. They must not use an
  output-only optimized exchange, which would drop required intermediate state.
- Main no longer stores `vllm_config` on `AscendConfig`. The attention caller
  passes its current scheduler budget through the trailing optional
  `decode_token_budget` argument on the registered operator and FakeTensor
  implementation. Existing callers can omit it; standalone helpers retain
  the manual cap. No new serving API or user configuration is added.
- Missing-group errors and upstream invalid-shard semantics are retained.

## Fresh validation and outstanding gates

The focused CPU suite passes **101 tests**, covering budget boundaries and
current-caller propagation, PCP/token-scatter fallback, deferred/returned-LSE
semantics, FakeTensor output, native quantization/store routing, projection
tail preservation, query wait ordering and empty remap inputs:

```bash
python3 -m pytest --noconftest -q \
  tests/ut/attention/test_sparse_index_remap_port.py \
  tests/ut/attention/test_sfa_dcp_exchange_port.py \
  tests/ut/attention/test_sfa_dcp_query_port.py \
  tests/ut/attention/test_sfa_indexer_store_port.py
```

NPU entries are under `tests/ut/attention/a2/`. The large-batch registered-op
regression covers T48/T192, changed graph inputs and an exactly representable
uniform-weight oracle. It does not replace the unchanged strict random-BF16
oracle. **Fresh full-main NPU and serving validation has not been performed.**
CPU tests and reused kernel evidence do not establish full-main runtime acceptance.

## Reused release evidence, not a new main benchmark

PR16349's same-image synthetic decode comparison used DP4/TP8/DCP8/EP32,
MTP5 with zero synthetic acceptance, zero-filled KV, C32 and fixed700 output
tokens. All seven runs completed50/50 requests with35,000 output tokens and
zero recorded errors:

| Release-image composition | Per-run mean TPOT (ms) | Median TPOT (ms) | Median throughput (tok/s) |
|---|---|---:|---:|
| PR15970 baseline |114.626,114.192|114.409|221.196|
| Previous combined port, cap12 |114.339,114.006|114.172|221.521|
| Updated combined port, cap192 |105.798,99.110,105.807|105.798|238.331|

Median TPOT improved7.53% versus PR15970 and7.33% versus cap12 in that
experiment. These are image-compatible overlay results, not full-main builds,
not accuracy results, and not confidence intervals. No gain is transferred
numerically to the refreshed main integration.

The release helper checks included1,152 rank-level changing-input graph
checks at nine sizes T13-T192, plus128 registered-op checks at T48/T192.
Additional positive colleague feedback concerns PR16349; it is not a new
validation of this main refresh and does not add an independently audited metric.

Strict CPU-BF16 `atol=rtol=1e-3` exchange checks historically failed for both
compared implementations. Thresholds remain unchanged; no complete numerical
acceptance is claimed. Historical operator improvements and the separate
SuperMem serving measurements must not be added to these percentages.
