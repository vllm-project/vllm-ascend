# sparse_index_remap

## Description

- **Function**: Remaps DCP sparse-attention top-k indices from the replicated view to DCP-local KV positions, retains entries owned by the current rank, compacts them in top-k order and pads the tail with `-1`. The current power-of-two DCP/interleave route uses two Triton kernels; other configurations use exact INT64 Torch arithmetic and order-preserving compaction.
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
- **Supported modes**: NPU eager execution and ACLGraph capture/replay through a hardware-generation-independent API. The current integer-boundary and graph tests have Atlas A3 execution evidence; no new 950 validation is claimed.

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
- Index values are logical KV positions, not bounded by top-k count. Nonnegative values within the input dtype's range must retain exact integer boundaries, including values above `2**24` and INT64 values above `2**31`. Do not use FP32 division as an oracle or a fallback.
- For power-of-two DCP and interleave, let `b = log2(interleave_size)` and `r = log2(dcp_size)`: `owner = (idx >> b) & (dcp_size - 1)` and `local = ((idx >> (b + r)) << b) + (idx & (interleave_size - 1))`. Both bit counts are static configuration; the current rank and top-k work count remain runtime parameters.
- Other layouts use INT64 quotient/remainder and unique order keys for compaction. Integer argsort may use AICPU; this path prioritizes correctness, not a generic-path speedup claim.
- `BLOCK` is capped at 128 and the compaction is deliberately serial: a `tl.cumsum` prefix-sum compaction produces nondeterministic garbage on A2 and traps the vector core under repeated launches on A3 (triton-ascend 3.2.0 / CANN 9.1, re-verified 2026-08). Re-validate if the compiler is upgraded.
- Two kernels are required: cross-program ordering (top-k order across chunks) cannot be expressed inside a single kernel without a global barrier, and an atomic-based reservation would make the output order nondeterministic.
- Inference only (decode path). Both INT32 and INT64 input/output are preserved; the fast path must not narrow INT64 indices.

## Origin and Differences

- **Origin**: Developed from scratch as a Triton replacement of the torch remap in `vllm_ascend/attention/context_parallel/sfa_cp.py` (`AscendSFADCPImpl._remap_sparse_indices`), whose fp32 torch implementation was introduced with the replicate-indexer SFA DCP feature (PR #11443).
- **Differences**:
    - NPU adaptation: fuses remap, per-chunk compaction and tail pre-fill into one kernel plus a gather kernel, rather than dispatching the separate Torch cast/divide/select/sort/gather expressions;
    - Exactness repair: avoids backend vector integer division that can lose boundaries above `2**24`; the bitwise route and INT64 fallback preserve owner selection and stable top-k order. Serial compaction is retained for compiler reliability (see Constraints).

## Test Cases

The current integer-repair tests cover top-k2048, rows1/6/12, every logical DCP2/4/8 rank, INT32/INT64, interleaves1/64/128/256, large integer boundaries, all-invalid rows and changing-input graph replay. Strided tensors, arbitrary leading dimensions and other DCP/interleave configurations are checked against an independent CPU integer oracle. As an index-only operator, the precision requirement is bit-exact (`rtol=0, atol=0`). Historical FP32-reference timing is not evidence for the corrected large-index implementation.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sfa_dcp.py -k remap
```
