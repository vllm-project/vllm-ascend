# packed_cache_slot_mapping

## Description

- **Function**: Publishes the per-token packed-cache metadata for one KV cache
  group: it converts the group's logical `slot_mapping` into a `[T, 2]`
  page/row coordinate buffer and a flat physical-slot buffer in a single Triton
  launch. It replaces the eager
  `compressed_slot_mapping` + `where`/`floor_divide`/`remainder`/`copy_` chain
  (roughly seven elementwise NPU ops per group) used by the DeepSeek V4.1 (DSA)
  metadata builder in `vllm_ascend/attention/dsa_v41.py`.
- **Formula**: Let `s` be a slot id and `p` the parallel position, with
  `B` the storage block size:
    - physical slot: `s' = s >> 1` for a ratio-2 plane, `s' = s` otherwise
    - page: `page = s' // B` (implemented as `s' >> log2(B)` when `B` is a power
      of two, and `s' & (B - 1)` for the row)
    - row: `row = s' % B`
    - invalid rows are written as `-1` in all three outputs
- **Algorithm flow** (processed per token tile, independently):
  1. Fixed one-dimensional grid: `BLOCK_SIZE = 8`, so
     `pid * 8 + tl.arange(0, 8)` is the token tile. The block size is a
     `constexpr` on purpose; deriving it from `num_tokens` would compile a second
     specialization and swap binaries as the batch crosses the threshold, paying
     an extra JIT compile on the request path.
  2. Load `slots` (widened to INT64) and compute
     `valid_end = min(query_start_loc[num_actual_reqs], num_actual_tokens)`.
     Validity starts as `token < num_tokens and token < valid_end and s >= 0`, so
     the padded tail of a graph-captured token carrier never writes a packed
     cache plane even if it still holds slot ids from an earlier replay.
  3. For a ratio-2 plane, load `positions` and additionally require
     `s & 1 == 1`, `p & 1 == 1` and `skip_update == 0`: only a completed
     compression pair owns a cache row. A ratio-1 plane compiles the load away
     and dereferences nothing but `positions.stride(0)` on the host.
  4. Split into page/row, clamping the physical slot to `>= 0` first so the
     right shift and the row mask stay non-negative, then mask every invalid row
     to `-1`.
  5. Store the page/row pair to the `[T, 2]` coordinate buffer and the physical
     slot to the flat buffer. Both are caller-owned persistent buffers, so ACL
     graph capture keeps stable addresses.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products. The kernel
  reproduces the eager metadata chain on every Ascend platform; it is not gated
  to the A5 packed-cache ABI. Works in eager and ACL graph modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `slots` | Input | Logical slot ids of the group, `[T]`, `-1` marks an invalid slot. May be non-contiguous; `stride(0)` is passed to the kernel | int32 / int64 | ND |
| `positions` | Input | Token positions `[T]`, read only when `compress_ratio == 2`; a ratio-1 caller may pass `slots` itself as a stride-only placeholder | int32 / int64 | ND |
| `query_start_loc` | Input | Per-request prefix offsets, at least `num_actual_reqs + 1` entries; entry `num_actual_reqs` bounds the valid token range | int32 | ND |
| `num_tokens` | Input (attribute) | Token carrier length `T` (grid extent, including graph padding) | int32 | scalar |
| `num_actual_reqs` | Input (attribute) | Number of real requests in `query_start_loc` | int32 | scalar |
| `num_actual_tokens` | Input (attribute) | Number of real tokens in the carrier | int32 | scalar |
| `page_size` | Input (attribute) | Packed cache page size / storage block size `B`; `PAGE_SIZE` and `LOG2_PAGE` constexprs | int32 | scalar |
| `compress_ratio` | Input (attribute) | Cache plane compression ratio; only `1` and `2` are implemented (`COMPRESS_RATIO` constexpr) | int32 | scalar |
| `skip_update` | Input (attribute) | When true, a ratio-2 plane writes no row (ring state update skipped) | bool | scalar |
| `coordinates_output` | Output | Page/row coordinates `[T, 2]`; column 0 is the page, column 1 the in-page row | int64 | ND |
| `flat_output` | Output | Flat physical slot per token `[T]`, `-1` when invalid | int64 | ND |

The launch returns `(coordinates_output, flat_output)` unchanged; it allocates
nothing.

## Constraints

- `compress_ratio` must be `1` or `2`; any other value raises `ValueError` before
  launching.
- `slots`, `positions` and `query_start_loc` must be one-dimensional. Their
  element strides are read on the host, so non-contiguous views are accepted.
- `positions.shape[0]` must equal `num_tokens` when `compress_ratio == 2`. With
  `compress_ratio == 1` the load is compiled out and no element is dereferenced.
- `query_start_loc` must contain at least `num_actual_reqs + 1` elements and be
  non-decreasing; `query_start_loc[num_actual_reqs]` may exceed
  `num_actual_tokens`, in which case `num_actual_tokens` wins.
- Slot ids are kept in INT64 end to end, so large packed caches cannot wrap a
  page/row address. Inputs wider than INT64 are not supported.
- `page_size` must be positive. Power-of-two page sizes use shift/mask; other
  sizes keep the generic divide/remainder form, which is slower on AIV because
  the vector cores have no integer divider.
- `coordinates_output` and `flat_output` must be pre-allocated with the exact
  shapes `[T, 2]` and `[T]`, and must outlive the step for graph replay.
  `num_tokens == 0` launches nothing and leaves both buffers untouched.
- All rows at or beyond `min(query_start_loc[num_actual_reqs],
  num_actual_tokens)` are written as `-1`, including the padded tail of a
  captured graph.

## Origin and Differences

- **Origin**: Developed from scratch for vllm-ascend. The reference semantics are
  the eager branch of `AscendDSAV41MetadataBuilder._build_cache_slots` in
  `vllm_ascend/attention/dsa_v41.py`; there is no upstream vLLM operator with
  this interface.
- **Differences**:
    - NPU adaptation for performance: the per-group eager elementwise chain is
      fused into one launch, the power-of-two page split becomes a shift/mask,
      and the ratio-2 completion test becomes low-bit tests, so the kernel holds
      no INT64 division. Measured over 1..256 tokens, `BLOCK_SIZE = 8` is
      1.76x-3.61x faster than 64.
    - Modified for a specific vllm-ascend logic: ratio-2 planes write only
      completed pairs (`s` and `p` odd). This reproduces
      `compressed_slot_mapping`'s `(s + 1) % ratio == 0` test plus the caller's
      position mask, but as two low-bit tests inside one launch; invalid,
      skipped-update and graph-padding rows share the single `-1` convention.
    - The outputs are caller-owned buffers rather than freshly allocated
      tensors, matching the graph-capture contract of the metadata builder.

## Test Cases

- **Reference implementation**: the eager chain of `_build_cache_slots` in
  `vllm_ascend/attention/dsa_v41.py`.
- **Accuracy criterion**: exact comparison (`rtol=0, atol=0`). Pages, rows and
  flat slots are integer indices, so the general FP16/BF16/FP32 tolerance table
  does not apply.
- **Coverage**: real inference shapes for DeepSeek V4.1 on Ascend (6 tokens for
  decode with 6 concurrent requests, 256 for the decode upper bound, 8192 for
  prefill `max_num_batched_tokens`, storage block size 128), plus the boundaries
  the kernel branches on: single token, grid saturation at 320 tokens, the
  ratio-1 SWA plane where positions are never dereferenced, a non power-of-two
  page, a page boundary, padded FULL-graph carriers, negative slots, 64-bit slot
  ids above 2^40, non-contiguous inputs and ACL graph replay.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_packed_cache_slot_mapping.py
```

## Example

```python
from vllm_ascend.ops.triton.packed_cache_slot_mapping import (
    build_packed_cache_slot_mapping,
)

coordinates, flat = build_packed_cache_slot_mapping(
    slot_mapping,          # [T] logical slot ids, -1 for invalid
    positions,             # [T]; ignored when compress_ratio == 1
    query_start_loc,       # [num_reqs + 1]
    num_tokens,            # T, includes graph padding
    num_actual_reqs,
    num_actual_tokens,
    storage_block_size,    # page size B
    compress_ratio,        # 1 or 2
    skip_update=skip_ring_state_update,
    coordinates_output=self._slot_mapping_2d[:num_tokens],   # [T, 2] int64
    flat_output=self._flat_slot_mapping[:num_tokens],        # [T] int64
)
```
