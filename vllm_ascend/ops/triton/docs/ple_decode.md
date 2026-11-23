# Qwen PLE decode short convolution

## Description

- **Function**: For each active decode request, apply the dilated PLE short convolution to its new token, write the activated result to `residual`, and advance that request's convolution state. One decode row corresponds to one request; a launch may contain many requests.
- **Formula**: Let `L = (K - 1) * dilation` and `u[t]` be the physical token row (`token_indices[t]` when supplied, otherwise `t`). For channel `c`, `z[t,c] = sum(k=0..K-1) float32(w[c,k]) * float32(v[t,c,k])`, where `v` is zero for an uninitialized state tap, `conv_state[state_indices[t], dilation*k, c]` for an initialized state tap (`k < K-1`), and `inputs[u[t],c]` for the last tap. Round `z` to the output dtype before SiLU. With `outer_residual`, add the prior `residual` in the output dtype, then add `outer_residual`; without it, store SiLU directly. Advance state by one position and append `inputs[u[t]]`. A null state index suppresses state access and writes a zero convolution result, matching upstream behavior.
- **Algorithm flow**: Make a private contiguous `[K,C]` copy of the caller's `[C,K]` weights; launch one Triton program per decode request and 512-channel tile; read the request's channel-contiguous state and token, accumulate in FP32, round and activate, write `residual`, then update that request's state in the same launch. Different requests use independent state slots.
- **Supported modes**: Ascend NPU decode. The 512-channel tile was validated at operator level on Ascend910_9382 and Ascend950DT_9582. Prefill uses the shared Ascend `ple_conv_prefill` kernel and speculative decode uses `ple_conv_spec` (see `ple_spec.md`); all three use the same SD state layout. Model-level accuracy and performance remain to be verified for this tile.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `inputs` | Input | New token for each decode request | BF16 or FP16 | Contiguous `[T,C]` |
| `residual` | Input/Output | Output buffer; prior value is used when `outer_residual` is supplied | Same as `inputs` | Contiguous `[T,C]` |
| `conv_state` | Input/Output | Per-slot history, shifted and appended in place | Same as `inputs` | `[slots,width,C]`, channel innermost, arbitrary positive strides for the last two dimensions |
| `conv_weights` | Input | Depthwise convolution taps; copied to tap-major storage inside the wrapper | Same as `inputs` | `[C,K]` |
| `state_indices` | Input | State slot per logical decode row; `NULL_BLOCK_ID` means no state | int32 or int64 | `[T]`, positive stride supported |
| `outer_residual` | Input | Optional second residual; `None` disables both residual additions | Same as `inputs` or `None` | Contiguous `[T,C]` when present |
| `dilation` | Attribute | Distance between convolution taps in the history | Python int | Positive scalar |
| `has_initial_states` | Input | Optional initialized-state flag per row; `None` means initialized for a valid slot | bool tensor or `None` | `[T]` when present |
| `token_indices` | Input | Optional logical-to-physical token row map | int32/int64 tensor or `None` | `[T]` when present |
| `query_start_loc` | Compatibility input | Accepted by the common PLE call signature; unused in decode | Tensor or `None` | N/A |
| `num_accepted_tokens` | Compatibility input | Accepted by the common PLE call signature; unused in decode | Tensor or `None` | N/A |
| `spec_query_len` | Compatibility attribute | Accepted by the common PLE call signature; unused in decode | Python int | N/A |

## Constraints

- `T >= 1`, `C >= 1`, `K >= 2`, `dilation >= 1`; `conv_state.shape[2] == C` and `conv_state.shape[1] >= L`. Serving uses `C=10240`, `K=4`, `dilation=3`, `L=9`; the scheduler determines `T`.
- `inputs`, `residual`, `conv_weights`, and present `outer_residual` have matching floating-point dtype and expected shapes. Valid state indices are in `[0, slots)`; non-null requests must have distinct slots for independent in-place updates. `token_indices` are valid physical rows.
- The launcher requires the SD shape `[slot,width,C]`. Both a contiguous allocation (channel stride 1) and a transposed window-contiguous view with the same SD shape are supported through explicit strides. The model wrapper canonicalizes the backend cache to SD before dispatch and does not change the caller's state layout or any input dtype.
- `C`, `K`, `dilation`, `L`, the optional `has_initial_states` and `outer_residual` branches, and tensor layouts may create distinct compiled variants. Request count `T`, slot values, token indices, and state contents are runtime values. The launch grid changes with `T` without intentionally specializing on `T`. `has_token_map` is marked `do_not_specialize`.
- The decode state update is fused with its output kernel; CUDA graph replay is supported by the normal serving path. This PR's NPU operator test does not itself prove graph replay or quantify graph-specific recompilation.

## Origin and Differences

- **Origin**: `vllm.models.qwen4_exp.common.ops.ple.ple_conv` and its decode Triton kernel.
- **Differences**: Ascend dispatch calls `ple_conv_decode` for decode. It copies weights to contiguous `[K,C]`, reads adjacent channels from adjacent addresses, and uses 512-channel tiles with the SD state layout. Both this kernel and the upstream decode kernel map each decode row directly to one request and fuse state writeback. Prefill and speculative decode use the dedicated Ascend `ple_conv_prefill` / `ple_conv_spec` kernels with the same SD layout; see `ple_conv_prefill.md` and `ple_spec.md`.
- **Reason**: Performance optimization for batched Qwen PLE decode on Ascend. This is an Ascend-specific implementation of an upstream interface, with no new environment variable or temporary compatibility fallback. It can be removed once the upstream decode kernel provides equivalent NPU performance and state semantics.
- **Side effects**: `residual` and `conv_state` are intentionally modified as specified by the upstream interface. `inputs`, `conv_weights`, `state_indices`, and optional metadata are not modified. The transposed weight buffer is private to this call; no process-global state or external layout is changed.

## Test Cases

- CPU unit contract: `pytest -q tests/ut/ops/test_triton_ple_decode.py` checks optional argument categories, SD layouts, pointer routing, weight-copy ownership, and invalid state shapes without launching an NPU kernel.
- NPU operator comparison: `pytest -q --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_triton_ple.py tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_ple_decode.py` compares output and updated state with upstream for BF16/FP16, decode batches, null slots, optional residuals, token mapping, channel tails, strided state indices, kernel sizes and dilations, and both SD contiguous and transposed state views. On Ascend 910B3, `test_ple_decode.py` passed 6/6 and `test_triton_ple.py` passed 7/7 with bit-exact output and state; historical A3/A5 validation of the 512 tile remains recorded in the preceding PRs.
- The PR description records operator latency and numerical results, including the 512-tile slowdown versus 2048 for large A5 decode batches and long prefill. Model-level accuracy, throughput, and recompilation measurements require a baseline and optimized service on equivalent hardware/configuration; they must not be inferred from this operator test.
