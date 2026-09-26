# glm5_next_lightning_indexer

## Description

- **Function**: `glm5_next_lightning_indexer_triton` selects compressed KeyPools for each query, expands the selected pools into original token indices, and appends the visible incomplete pool as a causal tail. GLM-5.3-Flash uses 32 query heads, head dimension 128, pool size 4, and `index_topk = 2048`, producing 2051 output columns.
- **Formula**: For query token `t`, head `h`, dimension `d`, and pool `j`, `qbar[t, d] = sum_h(weights[t, h] * query[t, h, d])` and `score[t, j] = sum_d(qbar[t, d] * cache[j, d])`. This preserves the existing head-weighted-query scoring semantics. Only pools before `min((positions[t] + 1) // P, indexer_seq_lens[r])` are visible. Select up to `index_topk // P` pools by descending score, then expand pool `j` to `[j * P, ..., j * P + P - 1]`. Tail positions range from `((positions[t] + 1) // P) * P` through `positions[t]`.
- **Algorithm flow**:
    1. Split the token batch into chunks targeting a 256 MiB score-buffer budget. Compute the head-weighted query with the existing FP32 multiplication and reduction.
    2. For larger batches, a bounded Triton gather reuses paged keys in a contiguous scratch cache. A Triton kernel maps tokens to requests, computes FP32 dot products with the head-weighted query, writes all score cells, and maps invisible, NaN, and negative-infinite scores to the lowest finite FP32 value. Positive infinity maps to the largest finite value. This avoids separate score initialization and sanitization passes.
    3. Apply `torch.topk` and convert its IDs once to int32. A Triton kernel expands pools, writes the causal tail, clears graph padding and alignment columns, and optionally writes directly into the SFA output buffer. The result contains logical token indices, not physical cache slots.
- **Supported modes**: Eager execution and fixed-shape NPU graph capture/replay with Triton-Ascend. The implementation targets Atlas A2/A3; this change was validated on A3 only. See Test Cases for validation scope. Ascend 950: N/A (not validated by this change).

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
| `output_buffer` | Optional output | Existing `[capacity, width]` buffer; width covers the result and any SFA alignment padding | int32 | Same device; contiguous columns and nonoverlapping rows |
| `pack_tail` | Attribute | Place the tail immediately after the causal history prefix; default `False`, enabled by the model caller | Python bool | Scalar, keyword-only |
| Return value | Output | Logical token indices, `[T, 1, index_topk + P - 1]`; unused columns are `-1` | int32 | ND |

## Constraints

- Inference only. `D` must be a power of two; this path is intended and tested for `D = 128`. `H`, `P`, and `B` must be positive. `index_topk` must be a positive multiple of `P`.
- `max_pool_seq_len >= 0`, `0 <= indexer_seq_lens[r] <= max_pool_seq_len`, and `max_pool_seq_len <= M * B`. If scoring is needed, the cache and request list must be nonempty. The caller must provide valid physical blocks for visible pools; clamping an invalid physical block is not a substitute for valid cache metadata.
- Queries are packed in request order, cumulative ends are nondecreasing, and the last end does not exceed `T`. Positions are nonnegative. Empty queries return shape `[0, 1, index_topk + P - 1]`. A zero maximum pool count returns only the causal tail, with the history region filled with `-1`.
- The first `index_topk` columns hold selected history. By default, tail tokens start at column `index_topk`. With `pack_tail=True`, they start at `min(((positions[t] + 1) // P) * P, index_topk)`, as required by the model's SFA consumer. The caller supplies complete history for that causal prefix.
- Rows beyond the final query end and output alignment columns are cleared to `-1`. An existing output buffer is reused, and rows beyond `T` are untouched. Equal-score pools may be returned in any top-k order.
- Cache address offsets use int32 only when the full tensor span fits. Larger caches use scalar int64 page bases and int32 offsets within a page; the page span must fit in int32 and the block size must be a power of two.
- Capture/replay requires stable shapes, addresses, strides, and scalar attributes (including `max_pool_seq_len`). Query values, weights, positions, pool lengths, and page-table contents may change in the existing buffers. The kernel skips invisible pool sub-tiles at runtime.
- Outside a captured graph, pool capacities and selected-pool counts are runtime parameters, avoiding a new Triton binary for each context length. Tensor strides and tiling choices still specialize the kernels.
- The score scratch budget controls token chunking; at least one score row is allocated. Its size is `max_pool_seq_len * sizeof(float32)`, so a single extremely long row can exceed the budget. No constant context-length cutoff is imposed by the wrapper. The optional prefill cache has a separate 256 MiB budget, so combined scratch can approach 512 MiB. It is enabled only when the original cache offsets fit in int32 and the token count exceeds the request count. Batches with one query per request and no token padding retain paged scoring to avoid packing keys without reuse. This shape-based heuristic includes graph padding in the token count and does not guarantee reuse for every request in a mixed batch.

## Origin and Differences

- **Origin**: Developed for GLM-5.3-Flash pooled-key selection.
- **Differences**: Retains device `torch.topk`, with request lookup, paged scoring, and final index expansion implemented in Triton. Larger token batches reuse each head-weighted query across several pool tiles; small batches expose more pool-level parallelism. Contiguous pool-ID reads avoid repeatedly gathering each ID for all tokens in its pool. Larger batches also expand output indices in wider blocks. The model no longer needs separate tail scatter, padding-mask, buffer fill, and copy operations.
- **Tiling**: Packed scoring and paged gathering use 128-pool tiles to fit the tested compiler's local-memory limit for FP32 multiplication and reduction. Index expansion uses 256-column blocks below 512 token rows and 4096-column blocks for larger chunks; the wider block is slower for medium batches on the tested A3 stack.

## Test Cases

The test uses the model's actual `[T, 32, 128]` BF16 queries, BF16 head weights/cache, pool size 4, and top-k 2048. Pool capacities 0, 4, 512, and 2050 cover tail-only output, insufficient history, exactly the selection width, and multiple score tiles with a partial final tile. Three requests exercise non-power-of-two request counts, distinct lengths, randomized physical page mappings, and noncontiguous cache blocks.

An independent CPU reference scores each head against the logical keys before weighting and summing the scores. Selected token membership and multiplicity, history padding, and tail columns must match exactly (`rtol = 0`, `atol = 0`); history is sorted only for comparison because equal scores do not define a unique ordering. Test inputs use a fixed local generator for reproducibility.

Both eager and graph cases force several token chunks with a reduced scratch budget. They change queries, head weights, positions, and visible pool lengths between calls/replays, and verify the cache remains unchanged. Graph cases exercise packed tails, strided output rows, alignment clearing, and output-buffer aliasing. Output shape/dtype and the empty-query path are checked separately. These are accuracy tests, not throughput measurements; hardware-specific performance results must be measured separately.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_pool_key_indexer_triton.py
```
