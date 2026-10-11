# gather_initial_states

## Description

- **Function**: Reads the selected recurrent-state rows into a contiguous tensor, filling fresh sequences with zero. The public entry point `gather_initial_states(state, indices, has_initial_state)` and private `_gather_initial_states_kernel` are in `vllm_ascend/ops/triton/mamba/state_ops.py`. The GLM KDA prefill path imports the public helper through `vllm_ascend/models/glm5next/ops/state_ops.py`; it does not depend on the worker's upstream-function rebinding.
- **Formula**: Let `state` have shape `[N, *S]`, `B = indices.numel()`, and `E = product(S)`. For each request `r` and flattened state element `e`:
    - If `has_initial_state[r]` is true, `output[r, e] = state[indices[r], e]`.
    - Otherwise, `output[r, e] = 0`, without reading that state row on NPU. This avoids propagating stale NaN/Inf values through multiplication by zero.
    - The input address is `state.data_ptr() + (indices[r] * state.stride(0) + e) * state.element_size()`. Strides passed to the kernel are in elements; the tensor's data pointer already includes its storage offset.
- **Algorithm flow** (processed row by row, independently):
    1. Allocate contiguous output `[B, *S]` with the state's dtype and device. Return without a kernel launch when `B == 0` or `E == 0`.
    2. Derive `BLOCK_SIZE = min(next_power_of_2(E), 16384 // state.element_size())` and launch a two-dimensional grid `(ceil(E / BLOCK_SIZE), B)`. Tile size and grid are internal implementation details.
    3. Each program reads its request's initialization flag from the device. Mask the index load with that flag, substituting index zero for fresh requests. Load state with the combined history and element-boundary mask and `other=0`, so fresh requests read neither the index nor the state. This uses masked loads instead of a runtime `if/else` branch.
    4. Store the tile in the contiguous output. The wrapper performs no device-to-host read of the indices or flags, and never materializes a contiguous copy of the whole state pool.
- **Supported modes**: Eager execution and ACL graph capture/replay on Ascend NPU. Hardware validation recorded for the copy kernel in [PR #17565](https://github.com/vllm-project/vllm-ascend/pull/17565) used Atlas A3. Atlas A2 and 950PR&950DT validation is N/A. Non-NPU tensors use a PyTorch fallback, which is also exercised by CPU unit tests.

## Parameters

All three public parameters are required. The function returns `output`; callers do not supply the output buffer. Private kernel arguments and launch attributes are derived by the wrapper.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `state` | Input | Recurrent-state pool `[N, *S]`; GLM uses `[num_blocks, H, V, K]` | bf16 / fp32 in the validated contract | Strided ND with contiguous individual rows |
| `indices` | Input | State row for each request, shape `[B]` | int32 / int64 | Strided 1-D |
| `has_initial_state` | Input | Whether each request has valid history, shape `[B]` | bool | Strided 1-D |
| `output` | Output (return value) | Gathered state `[B, *S]`; fresh rows are all zero | Same as `state` | Contiguous ND |
| `stride_state_batch` | Internal attribute | `state.stride(0)` | Integer, runtime scalar | Scalar, in elements |
| `stride_indices` | Internal attribute | `indices.stride(0)` | Integer, runtime scalar | Scalar, in elements |
| `stride_has_initial_state` | Internal attribute | `has_initial_state.stride(0)` | Integer, runtime scalar | Scalar, in elements |
| `ROW_SIZE` | Internal attribute | `E = product(state.shape[1:])` | Integer, constexpr | Scalar |
| `BLOCK_SIZE` | Internal attribute | Number of elements per program, capped at 16 KiB of state data | Integer, constexpr | Scalar |

## Constraints

- `state.ndim >= 2`, `indices.ndim == has_initial_state.ndim == 1`, and `indices.shape == has_initial_state.shape`. All input tensors are on the same device. Callers provide the integer index dtype and boolean flag dtype listed above; not every contract condition is checked by the wrapper.
- For a nonempty launch, `N > 0`, `E > 0`, and each state row must be contiguous (`state[0].is_contiguous()`). Gaps between rows and nonzero storage offsets are supported; arbitrary strides within a row are not. The backing allocation must cover every addressed row. Indices and flags can be non-contiguous 1-D views because their strides are passed explicitly.
- Where the flag is true, the index must satisfy `0 <= indices[r] < N`. Duplicate read indices are allowed. Where the flag is false, negative or out-of-range sentinel indices are ignored on NPU. There is no bounds check or invalid-index clamping for existing sequences.
- Within-row, output, index, and flag offsets must fit signed 32-bit arithmetic. The selected cache index is converted to 64-bit before multiplication by the cache-row stride, so the cache-page offset can exceed the 32-bit range.
- The non-NPU fallback selects row zero for fresh sequences and then uses `torch.where` to zero the result. It preserves the fresh-row NaN/Inf semantics but does not have the NPU path's no-read guarantee; it requires row zero to exist when a fresh row is selected.
- `state`, `indices`, and `has_initial_state` are not modified. The returned tensor has independent storage. The operator performs copies and zero filling only; it does not compute recurrent updates, reductions, or dtype conversions. Other dtypes and autograd support are N/A to the validated inference contract.
- Graph replay retains the captured shapes, strides, addresses, and launch dimensions. State contents, indices, and initialization flags may change in the same buffers between replays; each replay reads their current device values. An empty call launches no kernel and cannot become a nonempty operation through replay alone.
- `ROW_SIZE` and `BLOCK_SIZE` specialize the compiled kernel. They depend on the per-request state shape and dtype, not the request count, sequence length, selected indices, or initialization flags. Changing batch size changes the launch grid; changing index/flag contents does not create a new specialization. Different state shapes, pointer dtypes, or runtime-scalar specialization properties can produce additional variants. Uppercase naming documents constexpr arguments; it does not control JIT caching.

## Origin and Differences

- **Origin**: Follows the selected-row semantics of upstream vLLM's `vllm/model_executor/layers/mamba/ops/gather_initial_states.py`. The Ascend implementation replaces the GLM PyTorch `index_select` fallback introduced by [PR #15127](https://github.com/vllm-project/vllm-ascend/pull/15127).
- **Existing operator alternatives**: The validated A3/CANN environment does not provide an FP32 binary for the paged `kv_cache_load` gather. Flattening storage and composing native indexing operations avoids the whole-pool allocation but creates per-element indices and has higher measured latency for the TP1 state shape. This helper therefore keeps a single Triton gather. Operator measurements guide this choice; end-to-end performance must be measured separately.
- **State writeback**: The GLM `scatter_states` helper flattens each state into one row using metadata-only views and calls the existing in-place `torch_npu.npu_scatter_nd_update_` operator, preserving the padded page stride and storage offset. Device indices are viewed as `[B, 1]`; source states are viewed as `[B, E]`. The helper does not add a Triton scatter kernel or pack the full state pool. The caller must supply unique valid slots, matching state/source dtypes, contiguous individual rows, and non-overlapping source and destination storage. An empty write is skipped. Historical SK and Triton scatter measurements do not establish this wrapper's latency or A5 performance.
- **Differences**:
    - Launch an Ascend Triton copy without upstream CUDA/XPU device guards or CUDA-specific launch features.
    - Use the persistent cache's real row stride instead of materializing the entire padded state pool through the PyTorch NPU fallback. Output allocation scales with the selected rows, `B * E`, rather than the pool's `N * E` elements.
    - Skip state reads entirely for fresh sequences, including poisoned cache values and sentinel indices, and use dtype-sized tiles capped at 16 KiB.
    - Keep the public tensor API directly imported by GLM; no global replacement of upstream or unrelated same-named functions is needed.

## Test Cases

This is pure data movement. Compare copied values and zero-filled rows with `rtol=0`, `atol=0`; use `equal_nan=True` when intentionally preserving NaNs in valid history. NaN payload bit patterns are not an accuracy requirement.

The existing NPU tests cover BF16/FP32 fresh/history selection, NaN/Inf cache contents, sentinel indices, unchanged input state, and graph replay with changed indices, flags, and history. CPU unit tests additionally cover int32/int64 indices, multiple row shapes, and empty batches; CPU runs exercise the fallback, not the Triton kernel.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_glm5next_state_gather_npu.py
pytest -sv tests/ut/models/test_glm5next_state_ops.py
```

The manual A3 validation in PR #17565 also used the actual GLM-5.3-Flash TP1 cache geometry: FP32 state `[376, 64, 128, 128]`, one selected row, physical page size 4,587,520 bytes, `stride(0) = 1,146,880` elements, and storage offset 98,304 elements. The GLM KDA prefill contract tests exercise integration with the native chunk operator:

```bash
pytest -sv tests/ut/models/test_glm5next_kda_contracts.py
```

These commands reference existing tests; this documentation adds no nightly cases. The PR records the exact revisions used for historical NPU and 1M serving validation. This documentation change does not claim a new device run or extend that evidence to unvalidated hardware.
