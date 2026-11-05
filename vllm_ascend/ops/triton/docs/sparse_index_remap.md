# sparse_index_remap

## Description

- **Function**: Remaps replicated-view sparse-attention indices to DCP-local positions, retaining this rank’s entries in top-k order and padding the tail with `-1`. For power-of-two DCP and interleave, both A3 and A5 use the same position-sort kernel for INT32 K<=4096 or INT64 K<=2048; larger K uses two chunked kernels. The INT64 tile limit fits the 192 KiB UB on A3. Other layouts use exact INT64 Torch arithmetic and order-preserving compaction. Kimi local-output merge incorporates this operation and skips a second remap.
- **Formula** (per row, all integer arithmetic; `idx >= 0` marks valid lanes):
    - `block_idx = idx // interleave_size`
    - `owner = block_idx % dcp_size`
    - `valid = (idx >= 0) & (owner == dcp_rank)`
    - `interleave_size == 1`: `remapped = idx // dcp_size`
    - otherwise: `remapped = (idx // (dcp_size * interleave_size)) * interleave_size + (idx - block_idx * interleave_size)`
    - valid remapped values are compacted to the row front in top-k order; the tail is `-1`
- **Algorithm flow** (processed per row, chunked for parallelism):
  1. Fused kernel, grid `(num_chunks, rows)`: each program loads `BLOCK = min(128, next_power_of_2(topk_count))` indices, uses exact bit shifts/masks for the owner and local index, then serial-compacts valid entries to its `chunk_out` slot with scalar `get_element`, records the count and pre-fills its output region with `-1`. Input integer width is retained.
  2. Gather kernel, grid `(rows,)`: per row, iterates over the chunks and masked-copies `chunk_out[:cnt]` to the row front in chunk order, restoring top-k order across chunks.
- **Supported modes**: NPU eager execution and ACLGraph capture/replay. The same integer-boundary and changed-input graph cases run on each validated hardware generation; results are reported in the PR.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `topk_indices` | Input | Replicated-view top-k indices `[..., topk_count]`; negative values mark invalid lanes. Arbitrary leading dims are supported | INT32 or INT64 | ND |
| `dcp_size` | Input (attribute) | Number of DCP ranks | int32 | scalar |
| `dcp_rank` | Input (attribute) | This rank's index in the DCP group | int32 | scalar |
| `interleave_size` | Input (attribute) | KV cache interleave size of the replicated view | int32 | scalar |
| `out` | Output | Remapped and compacted indices with the same shape and dtype as `topk_indices`, tail padded with `-1` | Same as input | ND |

## Constraints

- `topk_count` must not exceed the configured `index_topk` (checked by the caller).
- Index values are logical KV positions, not bounded by top-k count. Nonnegative values within the input dtype’s range must retain exact integer boundaries, including above `2**24` and INT64 values above `2**31`. Do not use FP32 division as an oracle or a fallback.
- For power-of-two DCP and interleave, let `b = log2(interleave_size)` and `r = log2(dcp_size)`: `owner = (idx >> b) & (dcp_size - 1)` and `local = ((idx >> (b + r)) << b) + (idx & (interleave_size - 1))`. Both bit counts are static configuration.
- Other layouts use INT64 quotient/remainder and unique order keys for compaction; integer argsort may use AICPU.
- The chunked implementation requires two kernels to preserve cross-chunk order. The shared sort kernel sorts unique row-position keys and gathers the original integer payload, including INT64 values.
- Inference only (decode path). Both INT32 and INT64 input/output are preserved.

## Origin and Differences

- **Origin**: Developed from scratch as a Triton replacement of the torch remap in `vllm_ascend/attention/context_parallel/sfa_cp.py` (`AscendSFADCPImpl._remap_sparse_indices`), whose fp32 torch implementation was introduced with the replicate-indexer SFA DCP feature (PR #11443).
- **Differences**:
    - NPU adaptation: fuses remap, per-chunk compaction and tail pre-fill into one kernel plus a gather kernel, rather than dispatching the separate Torch cast/divide/select/sort/gather expressions;
    - Exactness repair: avoids backend vector integer division that can lose boundaries above `2**24`; the bitwise route and INT64 fallback preserve owner selection and stable top-k order. Serial compaction is retained for compiler reliability (see Constraints).

## Test Cases

The tests cover representative combinations of top-k2048, rows1/6/12, DCP2/4/8 owners, INT32/INT64, interleaves1/64/128/256, large integer boundaries, all-invalid rows and changing-input graph replay. Tile-boundary K4096/4097, strided tensors, arbitrary leading dimensions and other DCP/interleave configurations are checked against an independent CPU integer oracle. As an index-only operator, the precision requirement is bit-exact (`rtol=0, atol=0`). Historical FP32-reference timing is not evidence for the corrected large-index implementation.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sfa_dcp.py -k remap
```
