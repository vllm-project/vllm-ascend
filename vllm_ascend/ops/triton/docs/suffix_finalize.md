# Suffix draft finalization

## Description

- **Function**: Select local/corpus drafts, fill unused slots and optionally emit confidence.
- **Formula**: Local match length and position are decoded from `score // Lp1` and `score % Lp1`. Valid-slot confidence is `(accepted + 1) / (observed + 2)`; filler confidence is zero.
- **Algorithm flow**: `launch_suffix_finalize` reduces local scan scores and calls corpus lookup when enabled. The longer match wins, with ties going to local history; unused slots contain the last sampled token.
- **Supported modes**: Ascend A3 and A5; eager and graph execution. Adaptive verification requires full decode graphs and variable-query-length attention support.

## Parameters

`B` is active batch size, `R` request capacity, `L` history capacity, `N` scan block count, `C` corpus capacity and `K` draft width.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `token_ids_ptr`, `token_ids_stride`, `idx_mapping_ptr`, `total_len_ptr`, `num_sampled_ptr` | Input / Attribute | Histories, row stride, request slots and valid/sample counts | int32, integer | `[R, L]`, scalar, `[B]`, `[R]`, `[B]` |
| `last_sampled_ptr`, `scratch_ptr`, `scratch_stride`, `num_blocks`, `Lp1` | Input / Attribute | Filler tokens, packed scan scores, row stride, block count and radix | int64, integers | `[R]`, `[B, N]`, scalars |
| `corpus_ptr`, `corpus_len`, `table_ptr` | Input / Attribute | Optional token ring and hash indexes | int32, integer, int64 | `[C]`, scalar, `[2, NUM_BUCKETS, SLOTS]` |
| `drafts_ptr`, `num_valid_ptr` | Output | Fixed-width drafts and retrieved token counts | int64, int32 | `[B, K]`, `[B]` |
| `stats_ptr`, `last_bucket_ptr`, `last_valid_ptr`, `conf_ptr` | Input / Output | Adaptive counts, next-draft metadata and confidence | int64, int32, float32 | `[18, K, 2]`, `[R]`, `[R]`, `[B, K]` |
| `HAS_CORPUS`, `ADAPTIVE`, `HASH_MUL`, `NUM_BUCKETS`, `SLOTS`, `MAX_DEPTH`, `DEPTH_PO2`, `K`, `K_PO2`, `NB_PO2`, `launch_options` | Attribute | Feature flags, hash configuration, padded limits and compiler options | Booleans / integers / mapping | Host values |

## Constraints

Buffers reside on one device. Histories and scratch use actual row strides and contiguous columns; other buffers are contiguous.
Draft width and depth are positive. Internal padding changes neither output width nor valid counts; inactive corpus/adaptive paths do not read their placeholder buffers.
The caller owns allocation, ordering and verification budget assignment.

## Origin and Differences

- **Origin**: [vLLM #60339](https://github.com/vllm-project/vllm/pull/60339).
- **Differences**: Use scalar statistics accesses for non-power-of-two draft widths. Pad internal voting width to at least two to avoid the A5 unit-width boolean gather failure.

## Test Cases

Drafts and valid counts require exact equality against CPU references; confidence uses approximate comparison. Cases cover feature combinations, short histories, context limits and local/corpus selection.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_suffix_finalize.py
```

## Example

N/A for standalone use; the speculator calls this wrapper through the Triton dispatcher.
