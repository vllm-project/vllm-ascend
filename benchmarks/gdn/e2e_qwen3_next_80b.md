# Qwen3-Next-80B-A3B-Instruct GDN decode E2E comparison

Measured on October 26, 2026 using the complete BF16 model at
`/models/Qwen3-Next-80B-A3B-Instruct` (41 safetensors shards, 162,649,725,440 weight bytes).

## Results

Each cell is the median of three measured rounds after warmup. TPOT is the
per-request time from first to last output token divided by 255; the round
statistic is the median across requests. Throughput includes prefill and all
four request waves. Timed runs do not use profiling.

| Concurrency | Baseline TPOT (ms) | Fused TPOT (ms) | TPOT reduction | Baseline output tok/s | Fused output tok/s | Throughput gain |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 14.768 | 14.132 | 4.31% | 64.85 | 67.76 | 4.49% |
| 8 | 18.117 | 17.517 | 3.31% | 410.06 | 423.66 | 3.32% |
| 16 | 19.060 | 18.337 | 3.80% | 780.96 | 810.98 | 3.84% |
| 32 | 21.730 | 21.177 | 2.55% | 1397.49 | 1432.34 | 2.49% |

| Concurrency | Baseline mean TTFT (ms) | Fused mean TTFT (ms) |
| --- | --- | --- |
| 1 | 181.57 | 176.14 |
| 8 | 344.16 | 331.57 |
| 16 | 351.32 | 342.28 |
| 32 | 357.64 | 352.97 |

These are model/server measurements, separate from the operator speedups in
[the microbenchmark report](README.md). They apply to this workload and
configuration; longer prompts, speculative decoding, different TP sizes,
and larger batches were not evaluated.

## Configuration

- Baseline Ascend commit: `041cfb328c0628b1fc17db5c55b8d374e6363466`.
- Optimized: fusion commit `fa8f6f2d6` plus padded-state dispatch fix `d6e547280`.
- Upstream vLLM source: `256d8b40d0864aa694e3dc236d9b351886a4a94a`.
- Four logical Ascend910_9382 devices, IDs 0-3; same devices for both variants.
- CANN 9.1.0, torch 2.10.0, torch_npu 2.10.0.post4, Triton 3.2.0.
- FLA A3 wheel: `26.10.0+main.deved987cd`.
- TP4, BF16, Model Runner V2, no speculative decoding or prefix caching.
- FULL_DECODE_ONLY graphs, capture sizes 1/8/16/32; max sequences 32.
- Max model length and batched tokens 4096; KV budget 2 GiB per worker.
- 128 input tokens and exactly 256 output tokens (`ignore_eos=true`).
- Streaming completions, temperature 0, seed 42 plus request index.
- Deterministic text prompts with a request-index prefix; identical across variants.
- Closed-loop concurrency 1/8/16/32, four times concurrency requests per round.
- Three measured rounds per concurrency; separate 32-token warmup requests.
- Order: optimized, then baseline, with a full service restart between variants.

## Output checks

- Concurrency 1: 12/12 complete output token sequences match; 3072/3072 token positions match.
- Concurrency 8: 96/96 complete output token sequences match; 24576/24576 token positions match.
- Concurrency 16: 192/192 complete output token sequences match; 49152/49152 token positions match.
- Concurrency 32: 384/384 complete output token sequences match; 98304/98304 token positions match.

Both variants completed 684 timed requests and 175,104 output tokens without
request failures. Exact token agreement on this deterministic workload is a
regression check, not a general model-quality evaluation.

## Real fusion dispatch

The real model uses padded hybrid state pages. The original contiguous-state
guard caused fallback even though the kernel already honors the state slot
stride. The follow-up fix accepts non-overlapping padded slots while requiring
contiguous head/value/key dimensions. No kernel arithmetic changed.

The fix passed 19 dispatch/validation unit tests and 27 actual NPU tests,
including FP32/BF16 padded pages at batches 1/8/32 and untouched padding guards.
Ruff and the available repository format hooks passed; the unchanged C++ and
workflow tool checks remain unavailable in this container.

A separate real-model NPU capture contains executed `_fused_gdn_decode_kernel`
entries on all four TP ranks (180 entries in the rank-0 partial trace). The
first profiler attempt with Python stack capture crashed while stopping; raw
CANN data was parsed successfully offline. Timing used a restarted process
with profiling inactive, and excludes the profiling request.

## Shared runtime compatibility

The container's installed upstream and compiled Ascend extension predate the
current source. Both variants used identical isolated runtime packages and
the following compatibility changes, outside the development checkout:

- Accept upstream `replace_parameter(prefer_copy=...)`; existing tensor-copy
  behavior remains unchanged.
- Query physical device IDs through the same CANN C API because the installed
  extension lacks the new binding.
- Add the CANN Python ACL path and load FLA OPP before the older Ascend OPP
  package to prevent the duplicate CausalConv1d operator from winning lookup.
- Accept upstream coordinator `num_hash_lookahead_tokens=0`, asserting that
  speculative hash lookahead is disabled.
- Materialize ordinary Attention K/V caches as contiguous tensors once during
  cache initialization because the installed FIA rejects block-strided K/V.
  GDN state pages retain their original padded layout; no per-step copy is added.

Therefore, these results measure the fusion difference under a shared compatible
runtime, rather than demonstrating that unmodified latest source starts in the
container's original installed environment.

## Reproduction artifacts

The container directory `/workspace/gdn-e2e-20261026` preserves:

- `launch.py`, `run_variant.py`, `benchmark.py`: exact launcher and client.
- `baseline/`, `optimized/`, `vllm/`, `runtime-deps/`: isolated runtime snapshots.
- `manifest.json`: revisions, settings, dependency differences, model metadata hashes.
- `baseline-runtime.patch`, `optimized-runtime.patch`, `upstream-runtime.patch`.
- `baseline-results.json`, `optimized-results.json`: every request and output token ID.
- `summary.json`: aggregated metrics and agreement counts.
- Server/client logs, unit/NPU test logs, and `format.log`.
- `fusion-profile-evidence.json` and the parsed NPU kernel CSVs under `profiles/`.

The source development checkout is `/workspace/vllm-ascend`; runtime-only
compatibility patches were not committed there.
