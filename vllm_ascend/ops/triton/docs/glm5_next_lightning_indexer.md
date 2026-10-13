# glm5_next_lightning_indexer

## Description

- **Function**: `glm5_next_lightning_indexer_triton` selects compressed KeyPools for each query, expands the selected pools into original token indices, and appends the visible incomplete pool as a causal tail. GLM-5.3-Flash uses 32 query heads, head dimension 128, pool size 4, and `index_topk = 2048`, producing 2051 output columns.
- **Formula**: For query token `t`, head `h`, dimension `d`, and pool `j`, `qbar[t, d] = sum_h(weights[t, h] * query[t, h, d])` and `score[t, j] = sum_d(qbar[t, d] * cache[j, d])`. This preserves the existing head-weighted-query scoring semantics. Only pools before `min((positions[t] + 1) // P, indexer_seq_lens[r])` are visible. Select up to `index_topk // P` pools by descending score, then expand pool `j` to `[j * P, ..., j * P + P - 1]`. Tail positions range from `((positions[t] + 1) // P) * P` through `positions[t]`.
- **Algorithm flow**:
    1. Split the token batch into chunks targeting a 256 MiB scratch-buffer budget. Compute the head-weighted query with the existing FP32 multiplication and reduction.
    2. When streaming is explicitly enabled for supported unpacked caches, four partitions scan interleaved tiles up to each query's actual visible pool count. Each partition retains a power-of-two candidate count at least as large as the requested pool TopK, merging additional tiles with an exact score/ID bitonic sort. A partition containing one tile needs no local sort. Device metadata determines loop bounds on every graph replay; invisible tiles are neither read nor scored. The final `torch.topk` selects from the bounded candidates and gathers their original pool IDs.
    3. Other shapes retain dense scoring and `torch.topk`. For larger batches, a bounded Triton gather can reuse paged keys in a contiguous scratch cache. Both paths share the FP32 scoring and sanitization helper: invisible, NaN and negative-infinite scores map to the lowest finite FP32 value; positive infinity maps to the largest finite value.
    4. A Triton kernel expands pools, writes the causal tail, clears graph padding and alignment columns, and optionally writes directly into the SFA output buffer. The result contains logical token indices, not physical cache slots.
- **Supported modes**: Eager execution and fixed-shape NPU graph capture/replay with Triton-Ascend.

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
| `max_pool_seq_len` | Attribute | Upper bound on complete pool count; dense fallback score-buffer width | Python int | Scalar, keyword-only |
| `output_buffer` | Optional output | Existing `[capacity, width]` buffer; width covers the result and any SFA alignment padding | int32 | Same device; contiguous columns and nonoverlapping rows |
| `pack_tail` | Attribute | Place the tail immediately after the causal history prefix; default `False`, enabled by the model caller | Python bool | Scalar, keyword-only |
| `allow_cache_packing` | Attribute | Allow the shape-based packed-cache optimization; default `True`, disabled by the model caller in FULL graph mode | Python bool | Scalar, keyword-only |
| `use_streaming_topk` | Attribute | Explicitly enable bounded partition candidates for a workload with a validated visible-length bound; default `False`. Dense selection remains the safe default for arbitrary context growth | Python bool | Scalar, keyword-only |
| Return value | Output | Logical token indices, `[T, 1, index_topk + P - 1]`; unused columns are `-1` | int32 | ND |

## Constraints

- Inference only. `D` must be a power of two; GLM-5.3-Flash uses `D = 128`. `H`, `P`, and `B` must be positive. `index_topk` must be a positive multiple of `P`.
- `max_pool_seq_len >= 0`, `0 <= indexer_seq_lens[r] <= max_pool_seq_len`, and `max_pool_seq_len <= M * B`. If scoring is needed, the cache and request list must be nonempty. The caller must provide valid physical blocks for visible pools; clamping an invalid physical block is not a substitute for valid cache metadata.
- Queries are packed in request order, cumulative ends are nondecreasing, and the last end does not exceed `T`. Positions are nonnegative. Empty queries return shape `[0, 1, index_topk + P - 1]`. A zero maximum pool count returns only the causal tail, with the history region filled with `-1`.
- The first `index_topk` columns hold selected history. By default, tail tokens start at column `index_topk`. With `pack_tail=True`, they start at `min(((positions[t] + 1) // P) * P, index_topk)`, as required by the model's SFA consumer. The caller supplies complete history for that causal prefix.
- Rows beyond the final query end and output alignment columns are cleared to `-1`. An existing output buffer is reused, and rows beyond `T` are untouched. Equal-score pools may be returned in any top-k order.
- Cache address offsets use int32 only when the full tensor span fits. Larger caches use scalar int64 page bases and int32 offsets within a page; the page span must fit in int32 and the block size must be a power of two.
- Capture/replay requires stable shapes, addresses, strides, and scalar attributes (including `max_pool_seq_len` and `use_streaming_topk`). Query values, weights, positions, pool lengths, and page-table contents may change in the existing buffers. Streaming selection reads these lengths on device, so replay may grow to the full capacity and shrink again without recapturing or retaining stale candidates.
- Streaming selection requires unpacked keys, `D <= 128`, and `block_pool <= next_power_of_2(selected) <= 512`. Here `block_pool` is 128 for int32 cache addressing or one physical page for int64 addressing. It activates only when capacity exceeds the four-partition candidate width. Keeping at least `selected` candidates from each partition preserves the global TopK, including when one partition contains all winning pools. Equal scores can choose different IDs at the selection boundary, as with the dense path.
- Streaming is opt-in. Identical-input NPU graph measurements show short-context gains but substantial regressions at long actual visible lengths. A configured 1M capacity does not establish which path is faster. Since graph replay does not rerun Python dispatch, an opt-in caller must validate the entire visible-length range of that graph, including future growth. The default dense path avoids imposing the streaming regression on unbounded workloads; no host read of device lengths is introduced.
- The dispatch asserts `TOPK_TILE >= selected`, where `TOPK_TILE = next_power_of_2(selected)`. Every global TopK winner has rank at most `selected` within its own partition, so retaining `TOPK_TILE` local candidates cannot discard it. This also covers non-power-of-two K and all winners occurring in one partition; the final merge selects only `selected` pools.
- Scores are sanitized before selection in both paths. NaN, negative infinity and invisible pools become the lowest finite FP32 sentinel and are excluded from expansion; positive infinity becomes the largest finite FP32 value and remains eligible. A finite score equal to the lowest sentinel is also excluded. If fewer than K eligible pools remain, unused history lanes are `-1`; if none remain, all history lanes are `-1`. Causal tail placement is independent of score validity.
- Equal sanitized scores are ties. Local streaming merges use pool IDs to break ties, but the final device `torch.topk` does not promise a stable order or identical boundary IDs across paths. Correctness requires unique eligible pool IDs with maximal score membership, complete pool expansion, and correct tail/padding; it does not require an identical ID ordering for ties. Positive infinities can therefore tie after sanitization.
- Outside a captured graph, pool capacities and selected-pool counts are runtime parameters, avoiding a new Triton binary for each context length. Tensor strides and tiling choices still specialize the kernels.
- The scratch budget controls token chunking; at least one row is allocated. Streaming rows use `4 * next_power_of_2(selected) * (sizeof(float32) + sizeof(int32))` bytes. GLM's 512-pool TopK therefore uses 2048 candidate columns and 16 KiB per token row, independently of the configured maximum context. Dense fallback rows use `max_pool_seq_len * sizeof(float32)`, so one extremely long fallback row can exceed the budget. No constant context-length cutoff is imposed by the wrapper. The optional prefill cache has a separate 256 MiB budget, so combined scratch can approach 512 MiB. Packing requires `allow_cache_packing=True`, original cache offsets fitting in int32, and more token rows than requests. The model disables packing for FULL graphs: padded rows do not imply query reuse, and replay does not rerun Python dispatch. Other model execution modes trim token rows to the actual count before dispatch; a mixed batch can still contain requests without reuse. Direct callers with padded rows should also disable packing unless reuse has been established for that captured workload.

## Origin and Differences

- **Origin**: Developed for GLM-5.3-Flash pooled-key selection.
- **Differences**: Retains device `torch.topk`, with request lookup, paged scoring, bounded candidate merging, and final index expansion implemented in Triton. Large unpacked capacities avoid materializing and sorting the full context-width score matrix. Larger packed batches reuse each head-weighted query across several pool tiles. Contiguous pool-ID reads avoid repeatedly gathering each ID for all tokens in its pool. Larger batches also expand output indices in wider blocks. The model no longer needs separate tail scatter, padding-mask, buffer fill, and copy operations.
- **Tiling**: Packed scoring and paged gathering use 128-pool tiles to bound local-memory usage for FP32 multiplication and reduction. Index expansion uses 256-column blocks below 512 token rows and 4096-column blocks for larger chunks.

## Validation

- Correctness and replay regression: `pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_pool_key_indexer_triton.py` on an Ascend NPU. Coverage includes growth from 4K to 1M context capacity, shrinking and empty requests, cross-partition merging, all winners in one partition, non-power-of-two K around dispatch boundaries, all-invalid scores and fewer than K eligible pools, ties, nonfinite scores, paged addressing, token chunking, strided output buffers and padding.
- Performance: `python benchmarks/glm5next_streaming_topk.py --tokens 1 60`. This compares dense and streaming selection in captured graphs with identical inputs, including short and long visible contexts. NPU compilation compatibility and latency improvements must be established on the target runtime; CPU logic checks do not establish either result.
- Compilation audit: `python benchmarks/glm5next_topk_compile_audit.py --cache-dir /tmp/new-topk-cache --output compilation_audit.json`. Use a new cache directory and process. The Triton compilation listener counts actual compilation (`cache_hit=False`) separately from disk-cache loading, explicit NPU graph captures and replays. The audit changes device lengths with fixed shapes, then switches 1/60-query shapes and revisits them. Stable strides and tiling parameters remain `tl.constexpr`; a new request-count bucket may compile once, whereas changing device lengths alone should not.

### Measured dense/streaming comparison

Measured on 2026-10-13 using the same D-node Ascend NPU and runtime: PyTorch 2.10.0, torch-npu 2.10.0.post7, Triton 3.6.0 and vLLM 0.30.0. The existing service remained resident; the operator ran in a separate process. Both paths used identical seeded BF16 inputs, 32 query heads, dimension 128, pool size 4 and 512 selected pools. Packing was disabled to isolate dense versus streaming selection. Each latency is the median of five replay batches measured with NPU events; compilation, graph capture and correctness checks were outside timing.

For the configured 1M capacity (`max_pool_seq_len=262208`):

| Query tokens | Actual context tokens | Dense (us) | Streaming (us) | Dense / streaming |
| --- | --- | --- | --- | --- |
| 1 | 4096 | 163.45 | 114.24 | 1.431x |
| 1 | 131072 | 175.81 | 2280.67 | 0.077x |
| 1 | 1048576 | 400.27 | 18454.40 | 0.022x |
| 60 | 4096 | 1790.15 | 563.37 | 3.178x |
| 60 | 131072 | 4090.59 | 13572.85 | 0.301x |
| 60 | 1048576 | 19783.24 | 110690.06 | 0.179x |

For the smaller pool capacity (`max_pool_seq_len=33728`):

| Query tokens | Actual context tokens | Dense (us) | Streaming (us) | Dense / streaming |
| --- | --- | --- | --- | --- |
| 1 | 4096 | 66.01 | 114.92 | 0.574x |
| 1 | 131072 | 95.49 | 2278.40 | 0.042x |
| 60 | 4096 | 463.94 | 560.50 | 0.828x |
| 60 | 131072 | 2740.70 | 13568.29 | 0.202x |

All ten comparisons passed score-membership, uniqueness, pool expansion and tail/padding checks. The long-context regressions rule out default streaming dispatch based only on configured capacity. Sweeps with 16, 64 and 128 partitions did not remove the regressions, so the four-partition algorithm is retained as an explicit opt-in. The default is dense; the short-context speedups above are not default serving speedups.

The NPU regression suite passed 42 tests, including the non-power-of-two and invalid-candidate boundaries above. A fresh-cache compilation audit observed two cold Triton compilations at the 1-query shape and one new specialization on the first 60-query shape. Changing visible lengths at either fixed shape caused zero new compilations, and revisiting both shapes caused zero new compilations. Four explicitly requested NPU graph captures and 20 replays were counted separately; capture itself caused zero new Triton compilations. These counts describe the tested shapes and runtime, not every possible stride, dtype or request-count bucket.
