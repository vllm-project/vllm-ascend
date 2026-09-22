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

Single-card operator precision tests are in
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/`:

- `test_sfa_indexer_store_triton.py`: native INT8 key/scale byte checks and changing graph slots.
- `test_sfa_dcp_remap.py`: independent integer-boundary oracle, changing graph inputs and empty inputs.

These files were relocated without changing their contents or tolerances.
The query-gather and O/LSE exchange tests under `tests/ut/attention/a2/` require
an isolated eight-NPU group and are not single-card tests. The registered-op
regression covers T48/T192 and changing inputs with an exact uniform-LSE oracle;
it does not replace the unchanged strict random-BF16 oracle.

### Executed main-source validation, 2026-09-21/22

Runtime source: `c1e82ee97510f7805a6eef7663c53e7e3afacfe3`. Subsequent test relocation
and documentation edits do not change runtime code. The fresh main runs include:

- DCP8 registered T48/T192 ACLGraph capture and changing-input replay on all eight ranks.
- Single-card indexer-store/remap and DCP8 query graph checks.
- Two-layer random-weight HTTP integration, which exposed and led to the native
  DCP indexer decode-boundary fix; 35 indexer unit tests passed after that fix.
- Full 78-layer GLM-5.2 W4A8C8 HTTP smoke, TP8/DCP8/EP8, no MTP/offload:
  three 1240-token prompts completed normally with coherent Beijing/Paris/Tokyo
  answers and 64/85/66 output tokens. All eight ranks replayed optimized graphs
  on changing inputs. Loaded module hashes matched the recorded source commit.

### Measured performance, not extrapolated from the release branch

Registered O/LSE: 10 alternating ON/OFF blocks of 100 graph replays, using each
block's slowest rank and then the median. OFF selects the existing fallback.

| Tokens | OFF effective replay | ON effective replay | Reduction |
|---|---:|---:|---:|
|48|206.69 us|101.21 us|51.03%|
|192|569.19 us|220.02 us|61.35%|

Full-weight HTTP: **one completed OFF/ON pair**, per user instruction, with
1920 input and 128 output tokens, no MTP/offload/prefix cache. Same-head OFF
disables query/store/O-LSE performance eligibility; remap correctness fixes
remain in both arms. Both use scheduler budget4096, maxseq192 and KV budget6GiB.

| Concurrency | OFF mean TPOT(ms) | ON mean TPOT(ms) | Reduction | OFF / ON output throughput(tok/s) |
|---|---:|---:|---:|---:|
|1|48.74|47.46|2.63%|18.46 /18.88|
|4|62.56|60.05|4.01%|51.65 /53.25|
|192|655.25|635.78|2.97%|157.93 /161.34|

C192 cohort TPOT includes prefill/batch-filling interference. In the common
interval after all192 first tokens arrived and before any request's last token,
median client inter-token latency was173.51 ->144.80ms (-16.54%). The windows
were5.90/4.92s and are not independent repeat experiments or isolated kernel timings.
Each rank verified34 actual unpadded T192 optimized replays in the ON measurement.
Both arms completed197 measured requests with zero preemptions; all197 paired
128-token output sequences matched. No cross-run variance or confidence claim.

The isolated runtime used the verified vLLM pin84030bbe, image native libraries,
and FastAPI0.133.0/Starlette1.0.1. The existing vLLM/Ascend dependency declaration
conflict remains; this is not default-install acceptance. General strict numerical
acceptance, 1M-context and SuperMem ON/OFF performance are not established here.

Published evidence: [fresh NPU](https://github.com/vllm-project/vllm-ascend/pull/16350#issuecomment-5759854639),
[full weights](https://github.com/vllm-project/vllm-ascend/pull/16350#issuecomment-5764536283),
[performance](https://github.com/vllm-project/vllm-ascend/pull/16350#issuecomment-5770334097).

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
