# glm5_next_lightning_indexer

## Description

- **Function**: `glm5_next_lightning_indexer_triton` selects compressed KeyPools for each query, expands the selected pools into original token indices, and appends the visible incomplete pool as a causal tail. GLM-5.3-Flash uses 32 query heads, head dimension 128, pool size 4, and `index_topk = 2048`, producing 2051 output columns.
- **Formula**: For query token `t`, head `h`, dimension `d`, and pool `j`, `qbar[t, d] = sum_h(weights[t, h] * query[t, h, d])` and `score[t, j] = sum_d(qbar[t, d] * cache[j, d])`. Only pools before `min((positions[t] + 1) // P, indexer_seq_lens[r])` are visible. Select up to `index_topk // P` pools by descending score, then expand pool `j` to `[j * P, ..., j * P + P - 1]`. Tail positions range from `((positions[t] + 1) // P) * P` through `positions[t]`.
- **Algorithm flow**:
    1. Compute the head-weighted query in FP32. Split the token batch into chunks targeting a 256 MiB score-buffer budget.
    2. A Triton kernel maps tokens to requests, gathers paged compressed keys, and writes FP32 scores in pool tiles. Invisible scores remain negative infinity.
    3. Apply `torch.topk`, expand pool IDs, pad missing history with `-1`, and write the incomplete tail at the fixed `index_topk` column. The result contains logical token indices, not physical cache slots.
- **Supported modes**: Eager execution and fixed-shape NPU graph capture/replay on Atlas A2/A3 with Triton-Ascend. See Test Cases for validation scope. Ascend 950: N/A (not validated by this change).

## Parameters

`T` includes graph padding, `H` is the query head count, `D` the head dimension, `R` the request count, `B` the compressed-cache block size, and `P = index_kpool`. All tensor parameters are on the same NPU.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `query` | Input | Query vectors, `[T, H, D]`; model shape `[T, 32, 128]` | BF16 | ND |
| `indexer_cache` | Input | Compressed keys, `[N, B, 1, D]` | BF16 | ND; block, token, and dimension strides supported |
| `weights` | Input | Per-head query weights with model scaling already applied, `[T, H]` | BF16 | ND |
| `cum_query_lens` | Input | Cumulative exclusive query ends, `[R]`, without a leading zero | int32 | Contiguous ND |
| `indexer_seq_lens` | Input | Number of available complete pools per request, `[R]`, not raw token lengths | int32 | Contiguous ND |
| `indexer_block_table` | Input | Logical compressed-cache page to physical block mapping, `[R, M]` | int32 | ND; request and page strides supported |
| `positions` | Input | Absolute query token positions, `[T]` | int64 | Contiguous ND |
| `index_topk` | Attribute | Maximum number of history tokens selected through complete pools; model value 2048 | Python int | Scalar, keyword-only |
| `index_kpool` | Attribute | Number of original tokens in each complete pool; model value 4 | Python int | Scalar, keyword-only |
| `max_pool_seq_len` | Attribute | Upper bound on complete pool count and width of the score buffer | Python int | Scalar, keyword-only |
| Return value | Output | Logical token indices, `[T, 1, index_topk + P - 1]`; unused columns are `-1` | int32 | ND |

## Constraints

- Inference only. `D` must be a power of two; this path is intended and tested for `D = 128`. `H`, `P`, and `B` must be positive. `index_topk` must be a positive multiple of `P`.
- `max_pool_seq_len >= 0`, `0 <= indexer_seq_lens[r] <= max_pool_seq_len`, and `max_pool_seq_len <= M * B`. If scoring is needed, the cache and request list must be nonempty. The caller must provide valid physical blocks for visible pools; clamping an invalid physical block is not a substitute for valid cache metadata.
- Queries are packed in request order, cumulative ends are nondecreasing, and the last end does not exceed `T`. Positions are nonnegative. Empty queries return shape `[0, 1, index_topk + P - 1]`. A zero maximum pool count returns only the causal tail, with the history region filled with `-1`.
- The first `index_topk` columns hold selected history. Tail tokens always start at column `index_topk`, even when fewer history tokens are available. Thus valid entries need not form a contiguous prefix. Callers requiring a contiguous prefix must compact the result separately.
- Rows beyond the final query end are graph padding and their values are unspecified. The caller must mask or ignore them. Equal-score pools may be returned in any top-k order.
- Capture/replay requires stable shapes, addresses, strides, and scalar attributes (including `max_pool_seq_len`). Query values, weights, positions, pool lengths, and page-table contents may change in the existing buffers. The kernel skips invisible pool sub-tiles at runtime.
- The scratch budget controls token chunking; at least one score row is allocated. Its size is `max_pool_seq_len * sizeof(float32)`, so a single extremely long row can exceed the budget. No constant context-length cutoff is imposed by the wrapper.

## Origin and Differences

- **Origin**: Developed for GLM-5.3-Flash pooled-key selection.
- **Differences**: Combines request lookup, paged-cache gathering, and pool scoring in Triton, while retaining the device `torch.topk` operation and tensor operations for index expansion. The head-weighted query is computed once per token chunk. This replaces repeated host-dispatched indexing operations without changing cache allocation or pool-to-token semantics. Reuses `vllm.utils.math_utils.next_power_of_2` for request-capacity rounding.

## Test Cases

The test uses the model's actual `[T, 32, 128]` BF16 queries, BF16 head weights/cache, pool size 4, and top-k 2048. Pool capacities 0, 4, 512, and 2050 cover tail-only output, insufficient history, exactly the selection width, and multiple 2048-pool tiles with a partial final tile. Three requests exercise non-power-of-two request counts, distinct lengths, randomized physical page mappings, and noncontiguous cache blocks.

An independent CPU reference scores each head against the logical keys before weighting and summing the scores. Selected token membership and multiplicity, all history padding, and the fixed tail columns must match exactly (`rtol = 0`, `atol = 0`); history is sorted only for comparison because equal scores do not define a unique ordering. Test inputs use a fixed local generator for reproducibility.

Both eager and graph cases force several token chunks with a reduced scratch budget. They change queries, head weights, positions, and visible pool lengths between calls/replays, and verify the cache remains unchanged. Output shape/dtype and the empty-query path are checked separately. Validated on Atlas A3 with PyTorch 2.10.0, torch-npu 2.10.0.post4, and Triton-Ascend 3.2.0. Atlas A2 and Ascend 950 were not exercised in this validation. These are accuracy tests, not throughput measurements.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_pool_key_indexer_triton.py
```
