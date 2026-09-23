# copy_conv_state

## Description

- **Function**: Packs the active requests' convolution states from a strided persistent cache into contiguous temporary storage, or writes the updated states back. The public entry point is `copy_conv_state` in `vllm_ascend/ops/triton/kda/conv_state.py`. The GLM `causal_conv1d` wrapper uses it before and after the AscendC convolution when the state view is non-contiguous; contiguous states bypass both copies. The operator owns the private Triton kernel, stride derivation, tile size, and launch-grid policy.
- **Formula**: For request `r`, state row `s`, and channel `d`:
    - `slot = cache_indices[r]` and `active = (0 <= slot < cache.shape[0]) and (query_start_loc[r + 1] > query_start_loc[r])`.
    - With `state_len, dim = cache.shape[1:]`, the cache element offset is `slot * cache.stride(0) + s * cache.stride(1) + d * cache.stride(2)`; the packed offset is `(r * state_len + s) * dim + d`. All strides are in elements, and the operator also derives the index stride from `cache_indices`.
    - Packing (`write_back=False`) copies active states and fills inactive requests with zero. It sets `packed_indices[r] = r` for active requests and `-1` otherwise.
    - Writeback (`write_back=True`) copies packed values into active cache slots only. It preserves inactive slots, physical-page padding, and `packed_indices`.
- **Algorithm flow**:
    1. Derive requests, state dimensions, and strides from the tensors. Return without a launch for an empty request batch. Internally split each state row into `ceil(dim / 256)` channel tiles and launch `min(requests * state_len * ceil(dim / 256), 65535)` programs. These tuning choices are implementation details, not public API arguments.
    2. Each program processes tiles separated by the launch-grid size, so the bounded grid still covers all requests and state elements.
    3. Load the slot and query boundaries, then compute a scalar 64-bit cache-page address and 32-bit within-page offsets. Mask invalid requests and the final channel tile.
    4. Load/store in the requested direction. During packing, one tile per request also writes its packed index.
- **Supported modes**: Eager execution and ACL graph capture/replay on Ascend NPU. Hardware validation for this PR was on Atlas A3; Atlas A2 and 950PR&950DT validation is N/A. The caller is shared by GLM prefill, decode, and MTP verification; this copy kernel does not implement convolution or token-acceptance logic.

## Parameters

All parameters are required; `write_back` is keyword-only. The function mutates the supplied tensors and returns `None`. Callers and tests use this tensor API, not the private `_copy_conv_state_kernel` or its launch constants.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `cache` | Input when packing; output when writing back | Persistent state view `[num_slots, state_len, dim]` | fp16 / bf16 / fp32 | Strided ND |
| `packed` | Output when packing; input when writing back | Temporary state `[requests, state_len, dim]` | Same as `cache` | Contiguous ND |
| `cache_indices` | Input | Cache slot for each request, `[requests]` | int32 | Strided 1-D |
| `query_start_loc` | Input | Query start offsets, including the final end offset, `[requests + 1]` | int32 | Contiguous 1-D |
| `packed_indices` | Output when packing | Request-local indices, `[requests]`; unused on writeback | int32 | Contiguous 1-D |
| `write_back` | Attribute | `False` to pack, `True` to restore updated states | bool | Scalar |

## Constraints

- All tensors reside on the same NPU. `cache` and `packed` have matching dtypes and separate, non-overlapping storage. The caller supplies valid tensor extents and strides; the kernel does not validate them on the host.
- `requests = cache_indices.shape[0]`, `state_len`, `dim`, and `num_slots` are positive for a launch. The operator returns before launching for an empty request batch.
- `query_start_loc` is nondecreasing. A zero-length query or a slot outside `[0, num_slots)` is inactive. Inactive requests are zero-filled during packing and never write the persistent cache.
- Active requests must own distinct, non-overlapping cache states during writeback. The kernel does not arbitrate multiple writers to the same slot.
- Both state-row-major and channel-major views are supported through explicit strides, including gaps between physical pages. Packed storage has no such gaps. Within-page and packed offsets must fit signed 32-bit arithmetic; cache-page offsets may exceed that range and use 64-bit arithmetic.
- Packing, the AscendC convolution, and writeback execute in order on the caller's stream. Copying the state does not change its physical layout, allocation, or cache capacity.
- Graph replay retains the captured shapes, strides, and buffer addresses. Request indices and query-start contents may change in those buffers; an empty padded request remains inactive on replay.

## Origin and Differences

- **Origin**: Adapted from the GLM convolution-state staging kernel previously located in `vllm_ascend/models/glm5next/ops/causal_conv1d.py`, rather than from an upstream vLLM operator.
- **Differences**:
    - Tile actual state rows and channels rather than the padded physical page, while retaining a bounded launch grid.
    - Keep the potentially large cache-page address scalar and 64-bit without promoting every channel offset to 64-bit.
    - Preserve the existing pack/convolution/writeback ordering, invalid-slot behavior, and the original strided cache view. The move into `ops/triton/kda` does not change the copy algorithm.

## Test Cases

The existing `test_conv_state_copy_masks_invalid_slots_and_preserves_shared_pages` calls the public tensor API and covers both state layouts with `[4, 6, 384]` FP32 cache views, physical-page gaps, negative/out-of-range slots, a zero-length request, and unchanged backing storage outside the updated states. Its deterministic integer-valued FP32 data makes every copied value exactly representable. The test does not reproduce the operator's launch-grid formula; no additional test cases are introduced by this API refactor.

This is pure data movement: the correctness criterion is exact copied values and exact zero padding (`rtol=0`, `atol=0` for an independent reference). The existing test uses `torch.testing.assert_close` with its default tolerances. Historical A3 validation and its hardware scope are recorded in PR #17010; this document does not claim a new device run.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_glm5next_conv_state.py
```
