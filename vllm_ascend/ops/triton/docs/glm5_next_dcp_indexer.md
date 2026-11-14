# glm5_next_dcp_indexer

## Contract

`glm5_next_dcp_indexer` selects compressed pools from rank-local paged caches,
merges candidates across DCP, expands selected pools and appends the causal
incomplete tail. Inputs follow `AscendIndexerKPoolMetadata`: `seq_lens` counts
**global complete pools**, `positions` contains global token positions,
`cum_query_lens` contains exclusive request ends, and `block_table` addresses
**local physical pages**. Query rows and request order must agree across ranks.

The implementation uses the candidate transport from
[the Kimi sharded DCP implementation](https://github.com/vllm-project/vllm-ascend/pull/20883).
It preserves GLM's weighted-query score:

```text
qbar[t, d] = sum_h(weights[t, h] * query[t, h, d])
score[t, j] = sum_d(qbar[t, d] * key[j, d])
```

No ReLU is applied. BF16 keys are promoted to FP32 during scoring; FP8 keys use
a per-pool FP32 descale. The model caller quantizes/dequantizes FP8 queries with
the same dynamic quantization as its ordinary CANN selection path.

## Ownership and visibility

Let `P` be pool size, `I` the token interleave, `W` the DCP size, `r` the rank,
and `J = I / P`. Local pool `l` has global pool ID
`g = (l // J * W + r) * J + l % J`. It is visible only if
`g < min((position + 1) // P, seq_lens[request])`.

The compressor processes the complete query rows on each rank. Owner-masked
slot mappings write each completed pool exactly once. The small incomplete
pool ring remains replicated, including speculative rollback history. A
compressed pool never straddles DCP owners because `I` must be divisible by `P`.

Each rank selects `index_topk / P` local candidates, padded with `-1/-inf`.
Selection is stable by descending score, then ascending global pool ID. Every
rank participates even when its visible history is empty. The shared
`distributed_topk` exchanges IDs and FP32 scores only; it gathers no KV history.
The global result expands pool `g` to tokens `[P*g, ..., P*g+P-1]`, followed by
the visible tail, yielding `[T, 1, index_topk+P-1]` INT32 indices.

Shared layer buffers keep global IDs. SFA remaps into its local token cache,
gathers query heads, computes local softmax statistics and merges partial
outputs. A5's ordinary BF16 NoPE sink is included once, on DCP rank zero;
quantized/A3 paths retain their existing sink-free behavior.

## Memory and graph replay

The local score capacity is bounded by model length, DCP/interleave and table
width, independently of rank-local tensor values. Query chunks target a 32 MiB
FP32 score buffer. Stable sorting also allocates values and INT64 permutations;
the score budget is not a bound on total scratch memory. Candidate exchange
uses the shared row-tiling budget. GLM's default pool Top-K of 512 uses the
shared stable-sort fallback rather than the Kimi-specific fused Top-K of 2048.

Shapes, scalar attributes and tensor addresses must remain stable during
capture/replay. Queries, weights, positions, request ends, global pool lengths,
page tables and cache contents may change in place. Padded rows and all spare
output columns are set to `-1` on device. Output buffers require contiguous
columns and nonoverlapping rows. No device `.item()` drives the execution path.

## Validation

- CPU configuration, ownership, sink and host-dispatch contracts:
  `tests/ut/models/test_glm5next_dcp_cpu.py`.
- Actual scoring, stable tied selection, paged/FP8 cache reads, empty owners,
  tail expansion, rollback and graph replay:
  `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5_next_dcp_indexer.py`.
- HCCL candidate exchange under two, four or eight ranks:
  `tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_glm5_next_dcp_indexer.py`.
- Model integration with MTP, FlashComm1, prefix caching and RecoverSSM:
  `tests/e2e/pull_request/four_card/test_glm5_3_flash.py -k dcp`.

Device tests must run on A3/A5 before merge; FP8 indexer tests require A5.
Synthetic weights establish execution coverage only. Real-weight accuracy,
long-context memory/performance, C8 attention and PD transfer remain hardware
validation requirements.
