# Experimental PCP prefill communication optimizations

This branch combines five communication changes for GLM-5.3-Flash prefill.
They apply automatically when their existing eligibility conditions are met;
no environment switches are required. End-to-end text equivalence is still
unresolved; this is an experimental branch, not an accuracy-qualified release.

## Automatic selection

Use `--enforce-eager`, TP1/PCP8/EP8/DP1, V2, BF16 KV and no MTP to reproduce
the measured configuration. Other configurations are not performance-qualified.

| Optimization | Behavior and fallback |
| --- | --- |
| KPool hidden overlap | Start ordinary PCP KPool hidden AllGather before query projections; wait at key consumption. Graph and non-PCP paths retain existing behavior. |
| Local MoE Top8 | Select complete local Top8 routes over all 288 experts, then gather 8 FP32 weights and 8 exactly representable FP32 IDs. Restore INT32 IDs afterward. Only the sigmoid router, TP1/PCP8/DP1 and a global maximum local token count of at least 1024 are eligible. LoRA, EPLB state, forced balancing and dynamic expert remapping retain the existing path. |
| KDA output ReduceScatter | Pack the KDA output by the existing PCP token mapping and ReduceScatter instead of AllReduce followed by local row selection. Applies only to PCP prefill with `PCPRowParallelLinear`. |
| Main KV overlap | Overlap main KV AllGather with Q projection. Restricted to eager BF16 NoPE prefill, TP1/PCP8/DCP1. Preserve padding, replicated decode prefixes and cache slot mapping. |
| Early MoE hidden gather | Start hidden AllGather before the gate. Restricted to eager BF16 TP1/PCP8/DP1/EP8, hidden width 4096, 288 experts/Top8 and at least 1024 maximum local rows. Excludes profiling, LoRA, speculative decoding and dynamic placement. |

The combination submits main KV before KPool hidden communication and keeps
both send buffers alive until completion. MoE hidden communication starts once;
local routing can run while it is pending, followed by one packed route gather.
The original scoring, group/bias handling, normalization and routed scaling are
used for local expert selection.

## Hardware evidence

Measured on 2026-10-31 with eight Ascend950DT devices, vLLM-Ascend base
`0bbf9248abdab3fa4256dc955bdb57abc06229e8` and vLLM
`544f182d9f0c9bfe8bd7645525951d5e3692a542`.
The measurements used all five optimizations. The branch organizes that
prototype and removes its opt-in switches while preserving the eligibility
conditions and tensor operations. The measurements precede this cleanup;
the automatic-selection revision has not been benchmarked end to end again.
Upstream main `cae7ff0290dd6c6155f71d7223e0fd4aff95c9e9` was subsequently
merged for PR submission. This preserves its removal of legacy DSA-CP and
its MRV2 EPLB source-row load accounting; local Top8 retains the original
routing path whenever EPLB state is present. This integrated revision still
needs end-to-end accuracy and performance validation.

The workload uses 32 concurrent requests, 131072 input tokens and one output
token per request, 117760 cached tokens (89.84375%), fixed token IDs and seed
1024. Settings: `--max-num-seqs 32 --max-model-len 132096
--max-num-batched-tokens 32768 --gpu-memory-utilization 0.85`.
Each service instance primes the common prefix, warms up 32 requests and runs
five measured batches without a profiler. ABBA gives ten batches per variant.
Interrupted instances were archived and excluded in full; completed instances
were reused. The four valid instances have identical KV capacity (704306 tokens)
and zero measured preemptions. This was not an uninterrupted ABBA sequence.

| Metric | Original main | Five changes |
| --- | ---: | ---: |
| Median batch latency | 12.450216 s | 11.860511 s |
| Mean batch latency | 12.511869 s | 11.887804 s |
| Total input / total measured time | 335226 tokens/s | 352824 tokens/s |

Observed median latency reduction: **4.7365%**. Mean latency reduction:
**4.9878%**. Aggregate input throughput increase: **5.2496%** (including cached
input). The individual optimization percentages must not be summed.

Eight-rank interaction microtests passed: 576 MoE variant comparisons and
224 main-KV/KPool cases. They cover exact hidden values, routes, weights, cache
contents, padding and communication ordering. They do not establish full-model
accuracy. Formal text matches were A/A 151/160, B/B 159/160, A1/B1 145/160 and
A2/B2 150/160. Baseline variability does not explain away candidate differences.
Graph mode, decode and other workloads are not covered by these performance
results. O-proj weight/scale prefetch and split-MatMul proposals are excluded.

Archived evidence SHA256:
`de5ecdcfee47c8a746763376900624e4667c13249af3eca511b19c37d73ffbb4`.

## Focused regression tests

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q \
  tests/ut/models/test_glm5next_kda_reduce_scatter.py \
  tests/ut/models/test_glm5next_kpool_model_backend.py \
  tests/ut/attention/test_sfa_nope_forward.py \
  tests/ut/attention/test_pcp_main_kv_overlap.py \
  tests/ut/ops/test_pcp_moe_overlap.py
```

These CPU tests cover automatic selection, packing, padding, row mapping and
wait-before-consumption. They complement the hardware evidence above and do not
replace it.

The focused tests and affected KDA, PCP, routing and prepare/finalize suites
passed **750 tests** in an isolated CPU container using the measured runtime
dependencies. Obsolete switch cases were removed, and automatic local routing
selection and its small-batch, profiling and scoring fallbacks are covered.
The packed-cache fixture supplies its eager configuration so the real cache
eligibility checks exercise the synchronous fallback.
The integrated run also includes MRV2 EPLB, fused MoE, O-proj lifecycle,
SFA PCP precision and KDA contract regressions.

Changed-file Ruff, formatting, spelling, Markdown, import and source checks
passed, as did the staged Gitleaks scan. The full `bash format.sh ci` attempt on
Windows could not complete: actionlint setup stalled, Linux shebang hooks and
`python3` were unavailable, and the full-tree typos scan hit a Windows resource
error. Applicable Python/logger checks were run directly, and the changed-file
typos scan passed. Full repository CI remains to be run in its Linux environment.
