# postprocess_mamba

## Description

- **Location**: `vllm_ascend/ops/triton/mamba/postprocess.py` — `postprocess_mamba_fused_kernel`
- **Function**: Fused device-side post-processing for Mamba speculative decoding. For each request it recomputes the accepted-token decision (which Mamba cache block holds the running state and whether a state copy is needed) and performs the conv/temporal state block-to-block copy, entirely on device — removing the CPU-GPU synchronization of the Python reference implementation. It replaces vllm's upstream CUDA kernel globally via `vllm_ascend/patch/worker/patch_mamba_utils.py`; the production entries are vllm's `MambaSpecDecodeGPUContext.run_fused_postprocess()` and, on the V2 model runner, `MambaSpecDecodeGPUContext.run_fused_postprocess_align()`.
- **Indexing**: For active batch row `b`, resolve the persistent request-state slot as `r = idx_mapping[b]` when `HAS_IDX_MAPPING=True`, otherwise `r = b`. All decision arrays, per-group block-table rows, and accepted-token updates use this same slot `r`. A mapping of `-1` skips the batch row.
- **Formula** (per request-state slot `r`; mirrors the `postprocess_mamba` Python reference):
    - `num_tokens_running_state = num_computed + num_scheduled - num_draft` (or `new_num_computed - num_accepted + 1` when `PRECOMPUTED_NEW_COMPUTED`)
    - `new_num_computed = num_tokens_running_state + num_accepted - 1`; `aligned_new_computed = (new_num_computed // block_size) * block_size`
    - `needs_copy = aligned_new_computed >= num_tokens_running_state`; `accept_token_bias = aligned_new_computed - num_tokens_running_state`; `dest_block_idx = aligned_new_computed // block_size - 1`
    - Conv state (`conv_width > 0`): copy `src_block[accept_token_bias :]` → `dest_block[: conv_width - accept_token_bias]` (element count `(conv_width - bias) * inner_size`; DS `dim-first` layout copies per-dim rows instead)
    - Temporal state (`conv_width == 0`): copy whole block `block_table[r, src_block_idx + accept_token_bias]` → `block_table[r, dest_block_idx]` (`inner_size` elements)
    - When `needs_copy` and `src_block_idx == dest_block_idx`: write `num_accepted_tokens[r] = 1` (in place, or to the output buffer at slot `r` when provided) and skip the copy if `accept_token_bias == 0`
- **Algorithm flow** (processed per `(request, state)` program, independently):
  1. Grid `(num_reqs, num_layers * num_state_types [, TEMPORAL_TILES])` — `program_id(1)` indexes pre-flattened per-layer/per-state-type metadata directly. Bounds-check batch row `b = program_id(0)` against `num_reqs`, then resolve slot `r` as above and skip negative mappings. A mapped slot can be greater than or equal to `num_reqs`.
  2. Load all decision scalars at slot `r`, recompute the copy decision, and early-return when no copy is needed.
  3. Load the state metadata (base address, block stride, element size, inner size, conv width, group index), index row `r` of the owning group's persistent block table, and widen the src/dst block ids to int64 (block stride can exceed 2^31 bytes).
  4. Temporal states use `COPY_BLOCK_SIZE` uint64 elements per iteration when both addresses are 8-byte aligned, with a byte tail and a byte-copy fallback for unaligned addresses. SD conv keeps its contiguous byte-copy loop; DS conv `dim-first` layout copies per dimension row.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used by the Mamba speculative-decode decode step of hybrid GDN models (e.g. Qwen3-Next); works in both eager and graph-capture modes. The kernel supports a 2D grid with the default `TEMPORAL_TILES=1`, or a 3D grid that partitions temporal copies across extra CTAs.

## Parameters

> [!NOTE]
> Optional pointers and attributes with defaults are noted below. With `PRECOMPUTED_NEW_COMPUTED=True`, the scheduled-token and draft-token pointers are unused and may be `None`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `num_accepted_tokens_ptr` | Input/Output | Accepted token count at request-state slot `r`; updated in place when no output buffer is supplied | int32 | ND, per-slot |
| `mamba_state_idx_ptr` | Input | Source block column at request-state slot `r` | int32 | ND, per-slot |
| `num_scheduled_tokens_ptr` | Input | Scheduled token count at request-state slot `r` (unused when `PRECOMPUTED_NEW_COMPUTED`) | int32 / None | ND, per-slot |
| `num_computed_tokens_ptr` | Input | Computed token count at request-state slot `r` (post-step value when `PRECOMPUTED_NEW_COMPUTED`) | int32 | ND, per-slot |
| `num_draft_tokens_ptr` | Input | Draft token count at request-state slot `r` (unused when `PRECOMPUTED_NEW_COMPUTED`) | int32 / None | ND, per-slot |
| `block_table_ptrs_ptr` | Input | Base addresses of the per-group persistent block tables; row `r` belongs to request-state slot `r` in every group | int64 pointers | ND, per-group |
| `block_table_stride_req` | Input | Stride between persistent request-state rows of each block table (in int32 elements) | int64 | scalar |
| `state_base_addrs_ptr` | Input | Base address of each state tensor, flattened by `layer * num_state_types + state_type` | int64 pointers | ND, per-state |
| `state_block_strides_ptr` | Input | Bytes per cache block for each state | int64 | ND, per-state |
| `state_elem_sizes_ptr` | Input | Element size in bytes for each state | int32 | ND, per-state |
| `state_inner_sizes_ptr` | Input | Number of elements in the inner dimensions for each state | int64 | ND, per-state |
| `state_conv_widths_ptr` | Input | Conv width for conv states; `0` marks a temporal state | int32 | ND, per-state |
| `state_group_indices_ptr` | Input | Maps `state_idx` to its block-table group index | int32 | ND, per-state |
| `state_dim_row_count_ptr` | Input | Per-block dim row count for DS conv states (`0` keeps the single-region path) | int32 | ND, per-state |
| `state_dim_row_stride_ptr` | Input | Bytes between dim rows for DS conv states | int64 | ND, per-state |
| `num_accepted_tokens_out_ptr` | Output | Optional output buffer for the `src == dst` accepted-token update at slot `r`; when `None`, `num_accepted_tokens_ptr[r]` is updated in place | int32 / None | ND, per-slot |
| `idx_mapping_ptr` | Input | Optional active-batch-row to persistent request-state-slot mapping; `-1` skips a row (required when `HAS_IDX_MAPPING`) | int32 / None | ND, per-batch |
| `num_reqs` | Input (attribute) | Number of batch rows to process, including any `-1` padding rows (runtime value, not constexpr — avoids recompilation) | int32 | scalar |
| `block_size` | Input (attribute) | Mamba cache block size, fixed after model init (constexpr) | int32 | scalar |
| `COPY_BLOCK_SIZE` | Input (attribute) | Elements per copy-loop iteration: uint64 for aligned temporal states, bytes otherwise (constexpr tuning parameter) | int32 | scalar |
| `CONV_STATE_DIM_FIRST` | Input (attribute) | `True` when conv states use the DS `[block, dim, state_len]` layout (constexpr) | bool | scalar |
| `HAS_IDX_MAPPING` | Input (attribute) | Resolve batch row `b` to slot `r = idx_mapping[b]`; when `False`, use `r = b`. All per-request buffers use slot `r` in either case (constexpr, default `False`) | bool | scalar |
| `PRECOMPUTED_NEW_COMPUTED` | Input (attribute) | `num_computed_tokens_ptr` already holds the post-step value (constexpr, default `False`) | bool | scalar |
| `TEMPORAL_TILES` | Input (attribute) | Partition the temporal copy across this many CTAs via a 3D grid (constexpr, default `1`) | int32 | scalar |

## Constraints

- All per-request decision arrays, the optional accepted-token output, and all block-table rows use persistent request-state-slot order. `HAS_IDX_MAPPING` only selects how active batch rows resolve to those slots; it does not select a different block-table layout.
- With `HAS_IDX_MAPPING=True`, `idx_mapping` must be a contiguous one-dimensional int32 tensor with at least `num_reqs` entries. `-1` is the inactive-row sentinel; the kernel skips any negative mapping before accessing per-slot buffers. Every nonnegative slot must fit all decision arrays used by the selected branch, the optional output buffer, and every participating group's block table. `num_reqs` bounds batch rows, not persistent slots; the kernel does not check positive slot bounds.
- With `HAS_IDX_MAPPING=False`, batch rows must already correspond to persistent slots `0` through `num_reqs - 1`.
- Block tables are persistent int32 `[max_reqs, max_blocks]` tensors, one per Mamba cache group. Callers must pass the persistent tables in slot order, without compacting or reordering their rows to match the active batch. Block ids are widened to int64 before multiplication with `state_block_stride` (which can exceed 2^31 bytes for large Mamba caches).
- For example, `num_reqs=3` and `idx_mapping=[7, -1, 2]` select decision entries, block-table rows, and accepted-token output entries at slots `7` and `2`; the middle batch row is skipped. These buffers must cover at least eight slots even though the launch has only three batch rows.
- A separate accepted-token output buffer is not a packed batch result: only slots satisfying the accepted-count update condition are written, and all other entries retain their previous contents. Callers must initialize any other output entries they need. `PRECOMPUTED_NEW_COMPUTED` changes the meaning of the computed-token count, not the indexing contract.
- The cached Mamba alignment context and captured graphs reference persistent buffer addresses. Keep that storage valid across invocations, update mappings and per-slot contents for each batch, and rebuild affected context metadata and recapture affected graphs if the referenced storage is replaced.
- A state is a conv state iff `conv_width > 0`; `conv_width == 0` selects the temporal path. `CONV_STATE_DIM_FIRST` must match the conv state layout (`[block, state_len, inner]` vs DS `[block, dim, state_len]`).
- Temporal copy size uses the natural data size `inner_size * elem_size`, not `state_block_stride` (the page stride can exceed the data when the state tensor uses `as_strided` page padding).
- `block_size`, `COPY_BLOCK_SIZE`, `CONV_STATE_DIM_FIRST`, `HAS_IDX_MAPPING`, `PRECOMPUTED_NEW_COMPUTED` (and `TEMPORAL_TILES`) are compile-time `constexpr`; `num_reqs` is a runtime parameter to avoid recompilation per batch.
- Only for inference (speculative-decode decode step) on NPU.

## Origin and Differences

- **Origin**: Adapted from vllm's `vllm/v1/worker/mamba_utils.py` `postprocess_mamba_fused_kernel` (upstream CUDA kernel, see source-file header). Introduced by #10888 to fix a Triton error when `enable-prefix-caching` is active in the decode node.
- **Differences**:
    - NPU adaptation for performance: ported to triton-ascend with the pointer-type cast hoisted out of the copy loop (triton-ascend's `PtrOffsetInfo::AxisInfo` analysis aborts on in-loop casts — the same fix vllm-ascend applies to `batch_memcpy_kernel`); `TEMPORAL_TILES` partitions large temporal copies across CTAs;
    - Modified for a specific vllm-ascend logic or different input parameters: consumes per-group persistent block tables indexed by the same request-state slots as the decision arrays, supports the DS conv `dim-first` state layout, and adds `HAS_IDX_MAPPING` / `PRECOMPUTED_NEW_COMPUTED` for the V2 model runner and pipeline-parallel request mapping. Accepted-token updates use the same slot for either an explicit output buffer or the in-place path (vllm #50432).

## Test Cases

The test drives the patched `run_fused_postprocess` path against an independent Python reference for 4 requests and 2 layers, covering conv and temporal state types, combinations of scheduled/draft/accepted tokens, cache-block copies, and the `src == dst` accepted-token update.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_mamba.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_temporal_copy.py
```

The temporal-copy tests compare exact bytes including page guards, aligned and
unaligned addresses, zero-sized copies, byte tails, and multiple temporal tiles.
Mixed conv/temporal tests change accepted counts, request mappings and copy
decisions between graph replays while retaining persistent buffers.
`test_postprocess_uses_persistent_request_slot_tables` covers no mapping,
identity and reordered mappings, sparse slots beyond `num_reqs`, and `-1`
padding with two independent block-table groups, SD/DS conv layouts, and
multiple temporal tiles. Whole-buffer checks protect unselected slots, table
contents, decision inputs, and accepted-token output entries that are not updated.
