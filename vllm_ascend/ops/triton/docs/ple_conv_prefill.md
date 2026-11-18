# Qwen4Exp PLE prefill convolution

## Description

- **Function**: Apply the four-tap dilated PLE short convolution to prefill tokens, write the activated output to `residual`, and update each request's convolution state. Decode and speculative execution use the dedicated Ascend `ple_conv_decode` / `ple_conv_spec` kernels with the same SD state layout.
- **Formula**: For request `r`, local token `j`, channel `c`, kernel tap `k`, dilation `d`, and state length `L = (K - 1) * d`, let `h = j + k * d`. The tap value is `state[slot_r, h, c]` when `h < L` and an initial state exists; otherwise it is `inputs[query_start_r + h - L, c]` when `h >= L`, or zero. Compute `z[j,c] = cast_output_dtype(sum_k float32(weight[c,k]) * float32(tap[k]))` and `a[j,c] = cast_output_dtype(SiLU(z[j,c]))`. Without `outer_residual`, store `a` in `residual`. With `outer_residual`, store `outer_residual + cast_output_dtype(residual_old + a)`, preserving the eager rounding boundary. A null state slot contributes zero convolution output and is not written back.
- **Algorithm flow**:
  1. Transpose the caller's `[C, K]` weights into a new contiguous `[K, C]` temporary; the caller's weights are not modified.
  2. Launch one convolution program per logical token and 512-channel tile. Map the token to its request, read historical state or input tokens for each tap, accumulate in FP32, then write the activated result to `residual`.
  3. Launch a separate state-writeback program per request and 512-channel tile. This separation ensures every convolution token reads the old state. Copy the final `L` entries of the old-state-plus-current-input sequence into `conv_state`.
- **Supported modes**: Qwen4Exp PLE prefill on Ascend NPU. A3 Ascend910_9382 and A5 Ascend950DT were validated with the nightly operator file. Decode and speculative modes use the Ascend `ple_conv_decode` / `ple_conv_spec` operators; see `ple_decode.md` and `ple_spec.md`.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `inputs` | Input | Token activations `[T, C]` | FP16 / BF16 tested | ND, contiguous |
| `residual` | Input/Output | `[T, C]` output buffer, overwritten in place; old values are used when `outer_residual` is supplied | Same as `inputs` | ND, contiguous |
| `conv_state` | Input/Output | `[slots, window, C]` history state, updated in place for non-null slots | Same as `inputs` | ND, channel innermost, or a strided view |
| `conv_weights` | Input | Per-channel taps `[C, K]`; the model uses `K=4` | Same as `inputs` | ND |
| `state_indices` | Input | Request-to-state-slot mapping `[R]`; may be a strided view; null entries use `NULL_BLOCK_ID` | Integer tensor | ND |
| `outer_residual` | Input | Optional `[T, C]` second residual; its presence selects the fused residual path | Same as `inputs` or `None` | ND |
| `dilation` | Attribute | Tap spacing `d`; the model test uses `3` | Python integer | Scalar |
| `query_start_loc` | Input | Cumulative request boundaries `[R+1]` | Integer tensor | ND |
| `has_initial_states` | Input | Whether each request has valid prior state `[R]` | Boolean tensor | ND |
| `token_indices` | Input | Optional logical-to-physical token mapping `[T]` for mixed batches | Integer tensor or `None` | ND |

## Constraints

- Prefill requires `query_start_loc` and `has_initial_states`. `conv_state.shape[2]` must equal `C`, and its window (`conv_state.shape[1]`) must hold at least `L = (K - 1) * dilation` entries. The launcher checks these conditions.
- The supported model path uses FP16 or BF16 activations and weights. FP32 output accuracy has not been validated by this nightly test.
- `conv_state` uses the SD shape `[slot, window, channel]`. Both a contiguous allocation (channel stride 1) and a transposed window-contiguous view are supported through explicit strides. `state_indices` may have a non-unit stride.
- `PLE_BLOCK_C=512` and `PLE_NUM_WARPS=8` are fixed for every Ascend device. The exact A3 9362 compiler target from the original UB failure still requires its CI result; A3 9382 and A5 950DT have passed the operator test.
- `C`, `BLOCK_C`, `STATE_LEN`, `DILATION`, `KERNEL_SIZE`, `NULL_STATE_ID`, and the presence of `outer_residual` are compile-time constants. Changes to those values can create another Triton variant. `num_reqs`, the request-search iteration count, and the token-map flag are marked `do_not_specialize`; token count changes the launch grid. Scalar stride values can still affect specialization. A model-level shape and recompilation count has not been collected.
- `residual` and `conv_state` are intentional in-place outputs. The wrapper creates a temporary transposed weight copy; it does not mutate `inputs`, `conv_weights`, `outer_residual`, or request metadata.

## Origin and Differences

- **Origin**: Ascend prefill adaptation of `vllm.models.qwen4_exp.common.ops.ple.ple_conv`.
- **Differences**:
    - The Ascend Triton path uses a tap-major contiguous weight temporary and a 512-channel tile to avoid the oversized UB allocation seen with the 2048-channel tile on A3.
    - Convolution and state writeback are separate kernels to preserve old-state reads. The launcher requires the SD `[slot, window, channel]` layout so the kernels read channel-contiguous state.
    - Decode and speculative calls use the dedicated Ascend `ple_conv_decode` / `ple_conv_spec` kernels instead of upstream vLLM.
    - Matched model-layout measurements show the 512 tile is slower than 2048 on the tested A3 9382 and A5 950DT shapes; the PR description records the values. Model-level performance remains to be measured.

## Test Cases

- CPU unit contract: `pytest -q tests/ut/ops/test_triton_ple_prefill.py` checks the convolution and separate writeback launches, SD layout and strides, weight-copy ownership, token-map routing, initial-state flags, and argument/shape validation without launching an NPU kernel.
- NPU operator comparison: `test_triton_ple.py` compares the Ascend adapter with upstream PLE for FP16/BF16, prefill/decode/spec dispatch, 256- and 8192-token one-request prefill, multi-request boundaries, optional residuals, null state slots, initial-state flags, token mapping, and both contiguous and transposed SD state views. Output acceptance uses `abs(actual - golden) <= atol + rtol * abs(golden)` for at least 99% of elements, plus the dtype-specific maximum absolute error limit: FP16 `atol=rtol=2^-9`, max `0.1`; BF16 `atol=rtol=2^-6`, max `1`. State writeback is checked exactly. FP32 is not covered. The dedicated speculative operator is covered by `test_triton_ple_spec.py` (see `ple_spec.md`).

```bash
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_triton_ple.py
```

The full file passed 6/6 on A3 Ascend910_9382 and A5 Ascend950DT. On Ascend 910B3, the updated SD-layout file passed 7/7 together with the 6/6 decode file and the 12/12 spec file. The PR description contains the operator timing. GPQA model accuracy, model performance, and model-level recompilation evidence are tracked separately there.
