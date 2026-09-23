# copy_conv_state_kernel

## Description

- **Function**: Packs the active requests' convolution states from a strided persistent cache into contiguous temporary storage, or writes the updated states back. The entry point is `copy_conv_state_kernel` in `vllm_ascend/ops/triton/kda/conv_state.py`. The GLM `causal_conv1d` wrapper uses it before and after the AscendC convolution when the state view is non-contiguous; contiguous states bypass both copies.
- **Formula**: For request `r`, state row `s`, and channel `d`:
    - `slot = cache_indices[r * index_stride]` and `active = (0 <= slot < num_slots) and (starts[r + 1] > starts[r])`.
    - The cache element offset is `slot * cache_stride + s * STATE_STRIDE + d * DIM_STRIDE`; the packed offset is `(r * STATE_LEN + s) * DIM + d`. All strides are in elements.
    - Packing (`WRITE_BACK=False`) copies active states and fills inactive requests with zero. It sets `packed_indices[r] = r` for active requests and `-1` otherwise.
    - Writeback (`WRITE_BACK=True`) copies packed values into active cache slots only. It preserves inactive slots, physical-page padding, and `packed_indices`.
- **Algorithm flow**:
    1. Split each state row into `ceil(DIM / BLOCK)` channel tiles. The caller launches `min(REQUESTS * STATE_LEN * ceil(DIM / BLOCK), 65535)` programs with `BLOCK=256`.
    2. Each program processes tiles separated by the launch-grid size, so the bounded grid still covers all requests and state elements.
    3. Load the slot and query boundaries, then compute a scalar 64-bit cache-page address and 32-bit within-page offsets. Mask invalid requests and the final channel tile.
    4. Load/store in the requested direction. During packing, one tile per request also writes its packed index.
- **Supported modes**: Eager execution and ACL graph capture/replay on Ascend NPU. Hardware validation for this PR was on Atlas A3; Atlas A2 and 950PR&950DT validation is N/A. The caller is shared by GLM prefill, decode, and MTP verification; this copy kernel does not implement convolution or token-acceptance logic.

## Parameters

All parameters are required. Uppercase parameters are Triton compile-time attributes.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `cache` | Input when packing; output when writing back | Persistent state view `[num_slots, STATE_LEN, DIM]` | fp16 / bf16 / fp32 | ND, explicit strides |
| `packed` | Output when packing; input when writing back | Temporary state `[REQUESTS, STATE_LEN, DIM]` | Same as `cache` | Contiguous ND |
| `cache_indices` | Input | Slot for each request, addressed using `index_stride` | int32 | Strided 1-D |
| `starts` | Input | Query start offsets, including the final end offset, `[REQUESTS + 1]` | int32 | Contiguous 1-D |
| `packed_indices` | Output when packing | Request-local indices, `[REQUESTS]`; unused on writeback | int32 | Contiguous 1-D |
| `cache_stride` | Input (attribute) | Distance between cache slots in elements | Integer | Scalar |
| `index_stride` | Input (attribute) | Distance between entries of `cache_indices` in elements | Integer | Scalar |
| `num_slots` | Input (attribute) | Number of addressable cache slots | Integer | Scalar |
| `REQUESTS` | Attribute | Number of entries to process in `cache_indices` | Integer | Scalar |
| `STATE_LEN` | Attribute | Rows in each convolution state | Integer | Scalar |
| `DIM` | Attribute | Channels in each state row | Integer | Scalar |
| `STATE_STRIDE` | Attribute | State-row stride of `cache`, in elements | Integer | Scalar |
| `DIM_STRIDE` | Attribute | Channel stride of `cache`, in elements | Integer | Scalar |
| `WRITE_BACK` | Attribute | `False` to pack, `True` to restore updated states | bool | Scalar |
| `BLOCK` | Attribute | Power-of-two channel tile size; the caller uses 256 | Integer | Scalar |

## Constraints

- All tensors reside on the same NPU. `cache` and `packed` have matching dtypes and separate, non-overlapping storage. The caller supplies valid tensor extents and strides; the kernel does not validate them on the host.
- `REQUESTS`, `STATE_LEN`, `DIM`, and `num_slots` are positive for a launch. The wrapper returns before launching for an empty request batch.
- `starts` is nondecreasing. A zero-length query or a slot outside `[0, num_slots)` is inactive. Inactive requests are zero-filled during packing and never write the persistent cache.
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

The existing `test_conv_state_copy_masks_invalid_slots_and_preserves_shared_pages` covers both state layouts with `[4, 6, 384]` FP32 cache views, physical-page gaps, negative/out-of-range slots, a zero-length request, and unchanged backing storage outside the updated states. Its deterministic integer-valued FP32 data makes every copied value exactly representable. No additional test cases are introduced by this documentation change.

This is pure data movement: the correctness criterion is exact copied values and exact zero padding (`rtol=0`, `atol=0` for an independent reference). The existing test uses `torch.testing.assert_close` with its default tolerances. Historical A3 validation and its hardware scope are recorded in PR #17010; this document does not claim a new device run.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_glm5next_conv_state.py
```
