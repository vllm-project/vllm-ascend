# sparse_index_remap

## Description

- **Function**: Remaps replicated-view sparse-attention indices to DCP-local positions, retaining this rank’s entries in top-k order and padding the tail with `-1`. Both A3 and A5 use the same position-sort kernel for INT32 K<=4096 or INT64 K<=2048; larger K uses two chunked kernels. The INT64 tile limit fits the 192 KiB UB on A3. Positive DCP/interleave configurations, including interleave768/1536, stay in Triton. Kimi local-output merge incorporates this operation and skips a second remap.
- **Formula** (per row, all integer arithmetic; `idx >= 0` marks valid lanes):
    - `block_idx = idx // interleave_size`
    - `owner = block_idx % dcp_size`
    - `valid = (idx >= 0) & (owner == dcp_rank)`
    - `interleave_size == 1`: `remapped = idx // dcp_size`
    - otherwise: `remapped = (idx // (dcp_size * interleave_size)) * interleave_size + (idx - block_idx * interleave_size)`
    - valid remapped values are compacted to the row front in top-k order; the tail is `-1`
- **Algorithm flow** (processed independently per row):
  1. Within the sort tile limit, one program remaps a row, sorts unique position keys with invalid positions set to infinity, gathers the integer payload and pads invalid positions with `-1`.
  2. For larger rows, the fused kernel, grid `(num_chunks, rows)`, loads `BLOCK = min(128, next_power_of_2(topk_count))` indices, computes the exact owner and local index, then serial-compacts valid entries to its `chunk_out` slot with scalar `get_element`, records the count and pre-fills its output region with `-1`. Input integer width is retained.
  3. The gather kernel, grid `(rows,)`, iterates over those chunks and masked-copies `chunk_out[:cnt]` to the row front in chunk order, restoring top-k order across chunks.
- **Supported modes**: NPU eager execution and ACLGraph capture/replay. The same integer-boundary and changed-input graph cases run on each validated hardware generation; results are reported in the PR.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `topk_indices` | Input | Replicated-view top-k indices `[..., topk_count]`; negative values mark invalid lanes. Arbitrary leading dims are supported | INT32 or INT64 | ND |
| `dcp_size` | Input (attribute) | Number of DCP ranks | Python int | scalar |
| `dcp_rank` | Input (attribute) | This rank's index in the DCP group | Python int | scalar |
| `interleave_size` | Input (attribute) | KV cache interleave size of the replicated view | Python int | scalar |
| `out` | Output | Remapped and compacted indices with the same shape and dtype as `topk_indices`, tail padded with `-1` | Same as input | ND |

## Constraints

- `topk_count` must not exceed the configured `index_topk` (checked by the caller).
- DCP and interleave sizes must be positive, and `0 <= dcp_rank < dcp_size`.
- Index values are logical KV positions, not bounded by top-k count. Nonnegative values within the input dtype’s range must retain exact integer boundaries, including above `2**24` and INT64 values above `2**31`. Do not use FP32 division as an oracle or a fallback.
- For power-of-two DCP and interleave, let `b = log2(interleave_size)` and `r = log2(dcp_size)`: `owner = (idx >> b) & (dcp_size - 1)` and `local = ((idx >> (b + r)) << b) + (idx & (interleave_size - 1))`. Both bit counts are static configuration.
- Other layouts use unsigned reciprocal multiplication and integer correction in the same kernels. For width W and divisor d, `ceil(2**W/d)` gives an estimated quotient at most one above the true quotient. The high product is built from half-word products without requiring uint64 `mulhi`; subtracting one when `q*d > n` gives the exact quotient. Nonnegative inputs are at most `2**(W-1)-1`, so the correction product fits unsigned W bits. Negative sentinels are masked before arithmetic. Power-of-two individual divisors retain shifts. No Torch divide, remainder, argsort or gather fallback is used.
- The chunked implementation requires two kernels to preserve cross-chunk order. The shared sort kernel sorts unique row-position keys and gathers the original integer payload, including INT64 values.
- Recompilation: DCP/interleave geometry, tile size and dtype select static kernel variants. The sort kernel specializes K and rank; the chunk kernels use `do_not_specialize` for runtime K/rank and specialize chunk count instead. Row count changes the grid, while index values remain runtime data.
- Inference only (decode path). Both INT32 and INT64 input/output are preserved.

## Origin and Differences

- **Origin**: Developed from scratch as a Triton replacement of the torch remap in `vllm_ascend/attention/context_parallel/sfa_cp.py` (`AscendSFADCPImpl._remap_sparse_indices`), whose fp32 torch implementation was introduced with the replicate-indexer SFA DCP feature (PR #11443).
- **Differences**:
    - NPU adaptation: fuses remap, per-chunk compaction and tail pre-fill into one kernel plus a gather kernel, rather than dispatching the separate Torch cast/divide/select/sort/gather expressions;
    - Exactness repair: avoids backend vector integer division that can lose boundaries above `2**24`; shifts or corrected integer reciprocal multiplication preserve owner selection and stable top-k order. Serial compaction is retained for compiler reliability (see Constraints).

## Test Cases

The tests cover top-k2048, DCP8/16 with interleave768/1536, non-power-of-two DCP/interleave configurations, INT32/INT64 boundaries up to the dtype maximum, all-invalid rows and changing-input graph replay. Sort/chunk boundaries, strided tensors and arbitrary leading dimensions are checked against an independent CPU integer oracle. Direct remap calls exercise the operator even when Kimi's owner-local candidate output would skip it. As an index-only operator, the precision requirement is bit-exact (`rtol=0, atol=0`). Historical FP32-reference timing is not evidence for the corrected large-index implementation.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sfa_dcp.py -k remap
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_pre_post.py -k remap
```
