# sparse_index_remap

## Description

- **Function**: Remaps the DCP sparse-attention top-k indices from the replicated view to DCP-local KV positions on the SFA DCP decode path, keeps only the entries owned by the current DCP rank, compacts them to the front of the last dimension preserving top-k order, and pads the tail with `-1`. It replaces the ~30-op host-side torch chain in `AscendSFADCPImpl._remap_sparse_indices` with two Triton kernels.
- **Formula** (per row, all integer arithmetic; `idx >= 0` marks valid lanes):
    - `block_idx = idx // interleave_size`
    - `owner = block_idx % dcp_size`
    - `valid = (idx >= 0) & (owner == dcp_rank)`
    - `interleave_size == 1`: `remapped = idx // dcp_size`
    - otherwise: `remapped = (idx // (dcp_size * interleave_size)) * interleave_size + (idx - block_idx * interleave_size)`
    - valid remapped values are compacted to the row front in top-k order; the tail is `-1`
- **Algorithm flow** (processed per row, chunked for parallelism):
  1. Fused kernel, grid `(num_chunks, rows)`: each program loads `BLOCK = min(128, next_power_of_2(topk_count))` indices, computes the remap and the owner-validity mask vectorized (int32), then serial-compacts the chunk's valid entries to the front of its `chunk_out` slot with scalar `get_element` operations, writes the chunk's valid count, and pre-fills the chunk's region of the output with `-1`.
  2. Gather kernel, grid `(rows,)`: per row, iterates over the chunks and masked-copies `chunk_out[:cnt]` to the row front in chunk order, restoring top-k order across chunks.
- **Supported modes**: Atlas A2 and Atlas A3 (verified on both); Ascend 950 N/A. The routing in `sfa_cp.py` is `HAS_TRITON and topk_indices.is_npu`, so any NPU device with triton-ascend takes the Triton path; the torch implementation remains the fallback. Used by the SFA DCP decode path of sparse-attention models (e.g. GLM5.2) in both eager and graph-capture modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `topk_indices` | Input | Replicated-view top-k indices `[..., topk_count]`; `-1` marks padding/invalid lanes. Arbitrary leading dims are supported (the real DCP shape is `[dcp_size, 1, topk_count]` after `dcp_group.all_gather(dim=0)`) | int32 | ND |
| `dcp_size` | Input (attribute) | Number of DCP ranks | int32 | scalar |
| `dcp_rank` | Input (attribute) | This rank's index in the DCP group | int32 | scalar |
| `interleave_size` | Input (attribute) | KV cache interleave size of the replicated view | int32 | scalar |
| `out` | Output | Remapped and compacted indices with the same shape and dtype as `topk_indices`, tail padded with `-1` | int32 | ND |

## Constraints

- `topk_count` must not exceed the configured `index_topk` (checked by the caller).
- Valid index values are in `[0, topk_count * dcp_size * interleave_size)`; `-1` is the padding value. The integer math is bit-exact with the fp32 torch fallback (indices are far below the 2^24 fp32 exactness bound).
- `BLOCK` is capped at 128 and the compaction is deliberately serial: a `tl.cumsum` prefix-sum compaction produces nondeterministic garbage on A2 and traps the vector core under repeated launches on A3 (triton-ascend 3.2.0 / CANN 9.1, re-verified 2026-08). Re-validate if the compiler is upgraded.
- Two kernels are required: cross-program ordering (top-k order across chunks) cannot be expressed inside a single kernel without a global barrier, and an atomic-based reservation would make the output order nondeterministic.
- Inference only (decode path); no dtype other than int32 is required by the caller.

## Origin and Differences

- **Origin**: Developed from scratch as a Triton replacement of the torch remap in `vllm_ascend/attention/context_parallel/sfa_cp.py` (`AscendSFADCPImpl._remap_sparse_indices`), whose fp32 torch implementation was introduced with the replicate-indexer SFA DCP feature (PR #11443).
- **Differences**:
    - NPU adaptation for performance: fuses remap + per-chunk compaction + output tail pre-fill into one kernel plus a small gather kernel, replacing the ~30 host-dispatched torch ops (`Cast/Floor/SelectV2/Sort/Gather` chain on `[6, 1, 2048]` tensors) that showed up as host-bound overhead on the DCP decode path;
    - Modified for a specific vllm-ascend logic: uses int32 math instead of the fallback's fp32 math (bit-exact results; the two variants measured within 0.97x-1.00x on NPU), and keeps the serial compaction instead of the vectorized prefix-sum variant for compiler reliability (see Constraints).

## Test Cases

The accuracy test uses the real inference shape `[6, 1, 2048]` (dcp_size=6, topk=2048) plus a broader grid of `dcp_size {2, 4, 6} x interleave_size {1, 2, 4} x topk {1, 8, 48, 256, 2048}` over every DCP rank, against an independent fp32 torch reference. As a pure integer index remap, the unified precision tolerance is bit-exact (`rtol=0, atol=0`). ATK verification on A3 additionally reports 180/180 accuracy pass and an e2e npu/cpu ratio of 2.50x for the real `[6, 1, 2048]` shape (device kernel time 19.9us; micro shapes are launch-bound).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sparse_index_remap_triton.py
```
