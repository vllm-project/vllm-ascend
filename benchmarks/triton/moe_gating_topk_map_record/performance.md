# MoE TopK, EPLB map and record

## Implementation and integration

The production entry point is `AscendRoutedExperts.forward_impl` →
`AscendFusedTopKRouter._compute_routing` →
`moe_gating_topk_map_record`. CANN `moe_gating_top_k` computes the scores,
optional correction-bias selection, logical TopK and routing weights. The
first Triton kernel maps those IDs through the **runtime** EPLB table and
accumulates one physical-expert histogram per grid over contiguous tokens.
The second kernel reduces the grid histograms and adds to cumulative
`expert_load`. Both kernels use ordinary stores, not global atomics.
The post-router mapping and downstream record are skipped only when this
entry point succeeded; otherwise the original mainline path remains active.

The fast path is restricted to contiguous logits, E ≤ 128, K ≤ 8,
T ≤ 524288, ungrouped renormalized softmax/sigmoid, V2 runner, ALLGATHER,
DP=PCP=1, and no downstream expert-ID rewrite. The grid histogram covers
only this rank's local physical expert range, which may be smaller than E.
These are correctness guards, not tuning guesses:
the earlier record would otherwise count provisional IDs or padded tokens.
The direct operator also validates its tensor ABI before raw-pointer Triton
access. `record_enabled` and the EPLB table remain device-side runtime state
for graph replay. Padding is supported only as a contiguous valid prefix;
other layouts must use mainline.

The correctness reference is mainline CANN TopK + EPLB mapping + MoE-produced
expert counts + downstream expert-load update. The NPU tests cover exact IDs
and counts, weights, local physical ranges, and routing-table/record-flag
updates under graph replay. The Router dispatch test is mocked and proves
selection of the fast path, **not a real model forward**. A TP2/EP2
Qwen3-MoE dummy-weight model smoke was attempted with E=16. The coherent PR
source cannot initialize workers against the available vLLM 0.29 image
(`vllm.models.deepseek_v41` is missing); the older vLLM 0.28 image is also
incompatible with current EPLB imports. Therefore no real model/serving
forward has passed, and model-level validation remains open. These smoke
failures are environment compatibility failures, not operator accuracy
results.

## Current-source device timings

The following `msprof op` Task Durations (µs) were captured for source
`0eb608d36` on isolated Ascend 910B4-1, CANN 9.1.0,
torch-npu 2.10.0.post4 and Triton-Ascend 3.2.2. Each kernel was captured
once after the profiler's warmup. Task Duration already includes NPU task
head overhead. The kernel body and tile formula are unchanged in the E=128
coverage update, but the dispatch guard and local-range contract changed.
This table therefore does **not** establish E=128 performance. Raw summaries
remain in the task-owned
`/workspace/hybrid-grid-v1/profiles_exact` directory on the validation host.

| T | E | K | Score | CANN TopK | Main map | Main record | Grid map/record | Grid reduce | Main sum | New sum | Main/New |
|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 8 | softmax | 5.18 | 30.44 | 1.68 | 17.16 | 1.70 | 37.30 | 24.04 | 1.55× |
| 64 | 8 | 8 | softmax | 6.62 | 26.46 | 1.74 | 19.74 | 1.72 | 34.82 | 28.08 | 1.24× |
| 64 | 16 | 6 | sigmoid | 6.04 | 28.86 | 1.88 | 21.12 | 1.82 | 36.78 | 28.98 | 1.27× |
| 128 | 16 | 8 | sigmoid | 7.50 | 27.04 | 1.76 | 24.34 | 1.68 | 36.30 | 33.52 | 1.08× |
| 256 | 32 | 8 | softmax | 10.78 | 30.42 | 1.76 | 16.90 | 2.22 | 42.96 | 29.90 | 1.44× |
| 512 | 16 | 8 | sigmoid | 13.82 | 27.64 | 1.76 | 23.28 | 2.10 | 43.22 | 39.20 | 1.10× |
| 65536 | 16 | 8 | sigmoid | 942.88 | 1301.60 | 1.64 | 533.62 | 2.10 | 2246.12 | 1478.60 | 1.52× |
| 131072 | 32 | 8 | softmax | 2470.10 | 2883.44 | 1.76 | 1416.44 | 2.12 | 5355.30 | 3888.66 | 1.38× |
| 262144 | 32 | 6 | softmax | 5004.28 | 3566.18 | 1.66 | 2423.68 | 2.10 | 8572.12 | 7430.06 | 1.15× |
| 524288 | 32 | 8 | sigmoid | 7510.18 | 11494.78 | 1.54 | 5572.28 | 2.06 | 19006.50 | 13084.52 | 1.45× |

These sums are **component cost models**, not measured contiguous MoE or
serving latency: mainline records after MoE, whereas this implementation
records at routing. They must not be interpreted as end-to-end speedups.
The source above passed 96/96 business-shape accuracy cases, 14 smoke cases,
six edge cases, graph replay and 24/24 NPU tests. The profiler's
`PipeUtilization.csv` points to scalar work as the primary large-prefill
bottleneck: at T=65536/E=16, per-vector-program scalar time falls from
1255.26 to 481.83 µs while vector time rises from 8.99 to 55.20 µs.
These overlapping component times are diagnostic, not additive.

The AscendC implementation in private PR #3 targets ASCEND950 and was not
run on this 910B4 host. PR #17574 was used as a design/performance reference,
not as the correctness oracle. Neither cross-device published numbers nor
host-enqueue event timings are mixed into the table above.
