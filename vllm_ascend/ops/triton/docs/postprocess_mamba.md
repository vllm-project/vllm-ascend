# postprocess_mamba

## Description

- **Location**: `vllm_ascend/ops/triton/mamba/postprocess.py` — `postprocess_mamba_fused_kernel`
- **Function**: Fused device-side post-processing for Mamba speculative decoding. For each request it recomputes the accepted-token decision (which Mamba cache block holds the running state and whether a state copy is needed) and performs the conv/temporal state block-to-block copy, entirely on device — removing the CPU-GPU synchronization of the Python reference implementation. It replaces vllm's upstream CUDA kernel globally via `vllm_ascend/patch/worker/patch_mamba_utils.py`; the production entry is vllm's `MambaSpecDecodeGPUContext.run_fused_postprocess()`.
- **Formula** (per request `i`; mirrors the `postprocess_mamba` Python reference):
    - `num_tokens_running_state = num_computed + num_scheduled - num_draft` (or `new_num_computed - num_accepted + 1` when `PRECOMPUTED_NEW_COMPUTED`)
    - `new_num_computed = num_tokens_running_state + num_accepted - 1`; `aligned_new_computed = (new_num_computed // block_size) * block_size`
    - `needs_copy = aligned_new_computed >= num_tokens_running_state`; `accept_token_bias = aligned_new_computed - num_tokens_running_state`; `dest_block_idx = aligned_new_computed // block_size - 1`
    - Conv state (`conv_width > 0`): copy `src_block[accept_token_bias :]` → `dest_block[: conv_width - accept_token_bias]` (element count `(conv_width - bias) * inner_size`; DS `dim-first` layout copies per-dim rows instead)
    - Temporal state (`conv_width == 0`): copy whole block `block_table[src_block_idx + accept_token_bias]` → `block_table[dest_block_idx]` (`inner_size` elements)
    - When `src_block_idx == dest_block_idx`: write `num_accepted_tokens[i] = 1` (in place, or to the output buffer when provided) and skip the copy if `accept_token_bias == 0`
- **Algorithm flow** (processed per `(request, state)` program, independently):
  1. Grid `(num_reqs, num_layers * num_state_types)` — `program_id(1)` indexes pre-flattened per-layer/per-state-type metadata directly. With `HAS_IDX_MAPPING` (V2 model runner / PP), `program_id(0)` is a batch row resolved to a request-state slot through `idx_mapping` (`-1` = skip sentinel).
  2. Load the per-request decision scalars, recompute the copy decision, and early-return when no copy is needed.
  3. Load the state metadata (base address, block stride, element size, inner size, conv width, group index), index the owning group's block table, and widen the src/dst block ids to int64 (block stride can exceed 2^31 bytes).
  4. Byte-level copy loop in `COPY_BLOCK_SIZE` chunks: contiguous region for SD conv / temporal states; per-dim-row copy for DS conv `dim-first` layout.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used by the Mamba speculative-decode decode step of hybrid GDN models (e.g. Qwen3-Next); works in both eager and graph-capture modes. Two version-gated variants live in the source file: vllm 0.27.1 (2D grid, output-buffer semantics after upstream #50432) and v0.26.0 (3D grid with `TEMPORAL_TILES`, which partitions the temporal copy across extra CTAs to keep cores filled at small batch).

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `num_accepted_tokens_ptr` | Input | Accepted token count per request | int32 | ND, per-request |
| `mamba_state_idx_ptr` | Input | Source Mamba state (cache block) index per request | int32 | ND, per-request |
| `num_scheduled_tokens_ptr` | Input | Scheduled token count per request (unused when `PRECOMPUTED_NEW_COMPUTED`) | int32 | ND, per-request |
| `num_computed_tokens_ptr` | Input | Computed token count per request (post-step value when `PRECOMPUTED_NEW_COMPUTED`) | int32 | ND, per-request |
| `num_draft_tokens_ptr` | Input | Draft token count per request (unused when `PRECOMPUTED_NEW_COMPUTED`) | int32 | ND, per-request |
| `block_table_ptrs_ptr` | Input | Base addresses of the per-group persistent block tables, one per Mamba cache group | int64 pointers | ND, per-group |
| `block_table_stride_req` | Input | Row stride of each block table (in int32 elements) | int64 | scalar |
| `state_base_addrs_ptr` | Input | Base address of each state tensor, flattened by `layer * num_state_types + state_type` | int64 pointers | ND, per-state |
| `state_block_strides_ptr` | Input | Bytes per cache block for each state | int64 | ND, per-state |
| `state_elem_sizes_ptr` | Input | Element size in bytes for each state | int32 | ND, per-state |
| `state_inner_sizes_ptr` | Input | Number of elements in the inner dimensions for each state | int64 | ND, per-state |
| `state_conv_widths_ptr` | Input | Conv width for conv states; `0` marks a temporal state | int32 | ND, per-state |
| `state_group_indices_ptr` | Input | Maps `state_idx` to its block-table group index | int32 | ND, per-state |
| `state_dim_row_count_ptr` | Input | Per-block dim row count for DS conv states (`0` keeps the single-region path) | int32 | ND, per-state |
| `state_dim_row_stride_ptr` | Input | Bytes between dim rows for DS conv states | int64 | ND, per-state |
| `num_accepted_tokens_out_ptr` | Output | Output buffer for the `src == dst` accepted-token update; when `None`, `num_accepted_tokens_ptr` is updated in place | int32 / None | ND, per-request |
| `idx_mapping_ptr` | Input | Optional `batch_idx -> req_idx` mapping for the V2 model runner / PP (required when `HAS_IDX_MAPPING`) | int32 / None | ND, per-batch |
| `num_reqs` | Input (attribute) | Number of active batch rows (runtime value, not constexpr — avoids recompilation) | int32 | scalar |
| `block_size` | Input (attribute) | Mamba cache block size, fixed after model init (constexpr) | int32 | scalar |
| `COPY_BLOCK_SIZE` | Input (attribute) | Chunk size of the byte-copy loop (constexpr tuning parameter) | int32 | scalar |
| `CONV_STATE_DIM_FIRST` | Input (attribute) | `True` when conv states use the DS `[block, dim, state_len]` layout (constexpr) | bool | scalar |
| `HAS_IDX_MAPPING` | Input (attribute) | Resolve `program_id(0)` as batch index via `idx_mapping_ptr` (V2) instead of request index (constexpr, default `False`) | bool | scalar |
| `PRECOMPUTED_NEW_COMPUTED` | Input (attribute) | `num_computed_tokens_ptr` already holds the post-step value (constexpr, default `False`) | bool | scalar |
| `TEMPORAL_TILES` | Input (attribute) | v0.26.0 variant only: partition the temporal copy across this many CTAs via a 3D grid (constexpr, default `1`) | int32 | scalar |

## Constraints

- All per-request decision arrays are int32 and indexed in request-state-slot order; the block table is in batch order — the two indexings are split by `HAS_IDX_MAPPING`.
- Block tables are persistent int32 `[max_reqs, max_blocks]` tensors, one per Mamba cache group; block ids are widened to int64 before multiplication with `state_block_stride` (which can exceed 2^31 bytes for large Mamba caches).
- A state is a conv state iff `conv_width > 0`; `conv_width == 0` selects the temporal path. `CONV_STATE_DIM_FIRST` must match the conv state layout (`[block, state_len, inner]` vs DS `[block, dim, state_len]`).
- Temporal copy size uses the natural data size `inner_size * elem_size`, not `state_block_stride` (the page stride can exceed the data when the state tensor uses `as_strided` page padding).
- `block_size`, `COPY_BLOCK_SIZE`, `CONV_STATE_DIM_FIRST`, `HAS_IDX_MAPPING`, `PRECOMPUTED_NEW_COMPUTED` (and `TEMPORAL_TILES`) are compile-time `constexpr`; `num_reqs` is a runtime parameter to avoid recompilation per batch.
- Only for inference (speculative-decode decode step) on NPU.

## Origin and Differences

- **Origin**: Adapted from vllm's `vllm/v1/worker/mamba_utils.py` `postprocess_mamba_fused_kernel` (upstream CUDA kernel, see source-file header). Introduced by #10888 to fix a Triton error when `enable-prefix-caching` is active in the decode node.
- **Differences**:
    - NPU adaptation for performance: ported to triton-ascend with the pointer-type cast hoisted out of the copy loop (triton-ascend's `PtrOffsetInfo::AxisInfo` analysis aborts on in-loop casts — the same fix vllm-ascend applies to `batch_memcpy_kernel`); the v0.26.0 variant adds `TEMPORAL_TILES` to partition large temporal copies across CTAs;
    - Modified for a specific vllm-ascend logic or different input parameters: consumes per-group block tables (each Mamba group owns independently allocated physical blocks), supports the DS conv `dim-first` state layout, and adds `HAS_IDX_MAPPING` / `PRECOMPUTED_NEW_COMPUTED` for the V2 model runner and pipeline-parallel request mapping; dual version-gated implementations track the upstream `num_accepted_tokens` output-buffer semantics (vllm #50432).

## Test Cases

The test drives the patched `run_fused_postprocess` path against an independent Python reference for 4 requests and 2 layers, covering conv and temporal state types, combinations of scheduled/draft/accepted tokens, cache-block copies, and the `src == dst` accepted-token update.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_mamba.py
```
