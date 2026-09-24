# Native DCP decode layout port

## Main refresh, 2026-09-21

This is the main continuation of [PR16349](https://github.com/vllm-project/vllm-ascend/pull/16349).
Main base: `d052552072254a21560e8c5ebc04b80524c89622`.
Published main predecessor: `f8186d340651c2b7491d4026559c7b339f27a6cb`.
Release donor: `d5e74d19426fd6e18b9cf1e7e8dd33a703bb1b9b`.
The update merges main without rewriting the published branch history.

| Area | Selection |
|---|---|
| Sparse-index remap | Exact DCP8/interleave128 two-kernel path; other configurations use integer Torch fallback |
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

The focused CPU suite passes **113 tests**, covering budget boundaries and
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

The September22 relocation preserved contents and tolerances. The September24
follow-up adds generic integer-fallback precision and graph cases.
The query-gather and O/LSE exchange tests under `tests/ut/attention/a2/` require
an isolated eight-NPU group and are not single-card tests. The registered-op
regression covers T48/T192 and changing inputs with an exact uniform-LSE oracle;
it does not replace the unchanged strict random-BF16 oracle.

### Executed main-source validation, 2026-09-21/22

Historical runtime source: `c1e82ee97510f7805a6eef7663c53e7e3afacfe3`. The September22
test relocation did not change runtime code; the September24 review fixes below do.
The September21/22 runs include:

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

## Review follow-up, 2026-09-24

- The FP32 merge uses a finite `safe_max` for all-invalid rows, so even the
  expression discarded by `tl.where` no longer evaluates `-inf - (-inf)`.
- Merge and query helpers use the centralized device-property initialization
  and VectorCore count instead of probing the Triton driver separately.
- Registered graph tests add T13/T191/T192 with nonuniform LSE, changing inputs,
  extreme weights, and all-invalid/mixed-invalid ranks, including NaN/+Inf LSE.
  CPU FP64 reduction is independent of the NPU implementation. FP32 acceptance
  is `1e-6 + 8 * eps32 * sum(abs(weighted contributions))` per element. Registered
  BF16 output additionally permits half a local BF16 ULP. This is an explicit
  acceptance budget, not a formal bound on the backend exponential. The old
  strict BF16-to-BF16 test and its historical limitation remain unchanged.
- Generic remap errors were reproduced on DCP4/128, DCP8/64, DCP2/1 and DCP3/127
  with int32-range indices above `2**24`. The unsafe generic vector division
  is removed: other configurations now use int64 Torch division/remainder and
  order-preserving compaction. Integer sort may run on AICPU; this is a
  correctness fallback, not a performance improvement for those configurations.
  Native DCP8/128 retains its exact bit operations and two-kernel layout.

Fresh execution of this follow-up's runtime/test bytes passed on A3:
eight ranks each passed the new nonuniform test (96 rank/shape/replay cases),
the existing uniform T48/T192 test (128 cases), and query graph regression.
The single-card remap suite passed all35 cases, including102 changed-input
generic-fallback graph replays plus192 native-path replays and3 empty inputs.
Per-rank imported source/test hashes were checked against the publication tree;
allocated devices were healthy and empty after the suite. The existing full-model
service was not restarted or modified.

Both pre-commit stages pass for the changed files. An all-file run with ShellCheck
available additionally reports pre-existing warnings in untouched workflow YAML;
this follow-up does not modify those workflows or claim that full-repository gate
is green.

### Pack-kernel cold compilation and cache reuse

Sweep: T1/2/6/12/13/16/32/48/64/96/128/191/192, isolated A3, fresh dedicated cache,
then a new Python process reusing that cache. Each output is checked bitwise.
The pack kernel body is unchanged by this review follow-up.

- Sum of thirteen synchronized first calls with cold cache: **18.511 seconds**.
- Sum of first calls in the new process with disk cache: **0.169 seconds**.
- Cache growth: **531,350 bytes**, including metadata/IR; thirteen NPU binaries
  total **41,024 bytes**, each **3,000-3,400 bytes**.
- T192 binary: **3,192 bytes**; emitted TTIR contains one `scf.for`.
  The sweep does not show token-count-proportional unrolled code growth.

These are host first-call costs including compilation/loading/launch, not
kernel latency. The first cold call also includes Triton backend initialization.
Specialization still occurs; this is a representative sweep, not all T1-192.
No dynamic-token rewrite is made without evidence that it improves this tradeoff.
Previous model/performance numbers above remain tied to their original revision;
this follow-up does not claim a fresh full-model or SuperMem performance run.

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
