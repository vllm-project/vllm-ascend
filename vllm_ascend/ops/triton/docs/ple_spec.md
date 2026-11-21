# Qwen PLE speculative (MTP) short convolution

## Description

- **Function**: For each speculative decode token, apply the dilated PLE short convolution using the convolution state rolled back to the last accepted token, write the activated result to `residual`, and update each request's state. One launch contains many requests, and each request may span up to `spec_query_len` speculative tokens.
- **Formula**: Let `L = (K - 1) * dilation` and `u[t]` be the physical token row (`token_indices[t]` when supplied, otherwise `t`). For token `t`, `r = request(t)` is found by binary search over `query_start_loc`, `j = t - query_start_loc[r]`, and `slot_off[r] = min(max(num_accepted_tokens[r] - 1, 0), spec_query_len - 1)`. For tap `k` and `h = j + k * dilation`: if `h < L` and the slot is valid, the tap is `conv_state[state_indices[r], slot_off[r] + h, c]`; otherwise the tap is `inputs[u[query_start_loc[r] + h - L], c]`. Tokens beyond `query_start_loc[num_reqs]` write a zero convolution result. Compute `z[t,c] = sum_k float32(w[c,k]) * float32(tap[k])`, round to the output dtype before SiLU, then apply the same optional residual additions as decode. The writeback kernel produces `new_state[slot_r, i, c] = old_state[slot_r, slot_off[r] + 1 + i, c]` when `1 + i < L`, otherwise the corresponding input token; only `i < L + qlen - 1` is written, where `qlen = query_start_loc[r + 1] - query_start_loc[r]`. Null state slots are neither read nor written.
- **Algorithm flow**:
  1. Make a private contiguous `[K, C]` copy of the caller's `[C, K]` weights.
  2. Launch one convolution program per speculative token and 512-channel tile. Binary-search `query_start_loc` to map the token to its request, compute `slot_off`, read the state or input tap for each kernel position, accumulate in FP32, round and activate, and write `residual`.
  3. Launch a separate state-writeback program per request and 512-channel tile. The separate launch guarantees every token reads the old state; the program shifts the state window by one and appends the speculative chunk, preserving the unused tail for short or graph-padded chunks.
- **Supported modes**: Ascend NPU speculative decode (MTP). Validated at operator level on Ascend 910B3 with BF16/FP16 and `spec_query_len` 2-4. Model-level accuracy and performance remain to be measured.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `inputs` | Input | Speculative token activations | BF16 or FP16 | Contiguous `[T,C]` |
| `residual` | Input/Output | Output buffer; prior value is used when `outer_residual` is supplied | Same as `inputs` | Contiguous `[T,C]` |
| `conv_state` | Input/Output | Per-slot history, shifted and appended in place | Same as `inputs` | `[slots,width,C]`, channel innermost, arbitrary positive strides for the last two dimensions |
| `conv_weights` | Input | Depthwise convolution taps; copied to tap-major storage inside the wrapper | Same as `inputs` | `[C,K]` |
| `state_indices` | Input | State slot per request; `NULL_BLOCK_ID` means no state | int32 or int64 | `[R]`, positive stride supported |
| `outer_residual` | Input | Optional second residual; `None` disables both residual additions | Same as `inputs` or `None` | Contiguous `[T,C]` when present |
| `dilation` | Attribute | Distance between convolution taps in the history | Python int | Positive scalar |
| `query_start_loc` | Input | Cumulative request boundaries, including the final end offset | int32/int64 | `[R+1]`, nondecreasing |
| `num_accepted_tokens` | Input | Accepted-token count per request; selects the state rollback offset | int32/int64 | `[R]` |
| `spec_query_len` | Attribute | Number of speculative tokens per request in this chunk | Python int | Positive scalar |
| `token_indices` | Input | Optional logical-to-physical token row map | int32/int64 tensor or `None` | `[T]` when present |
| `has_initial_states` | Compatibility input | Accepted by the common PLE call signature; unused in speculative decode | Tensor or `None` | N/A |

## Constraints

- `T >= 1`, `R >= 1`, `C >= 1`, `K >= 2`, `dilation >= 1`, `spec_query_len >= 1`; `conv_state.shape[2] == C` and `conv_state.shape[1] >= L + spec_query_len - 1`. Serving uses `C=10240`, `K=4`, `dilation=3`, `L=9`, `spec_query_len=3`.
- `query_start_loc` and `num_accepted_tokens` are required. `query_start_loc` has `R+1` nondecreasing entries; `num_accepted_tokens` has `R` entries and is clamped into the rollback window. Valid state indices are in `[0, slots)`; non-null requests must use distinct slots for independent in-place updates.
- The launcher requires the SD shape `[slot,width,C]`. Both a contiguous allocation (channel stride 1) and a transposed window-contiguous view with the same SD shape are supported through explicit strides. The model wrapper canonicalizes the backend cache to SD before dispatch.
- `C`, `BLOCK_C`, `STATE_LEN`, `DILATION`, `KERNEL_SIZE`, `SPEC_QUERY_LEN`, `STATE_WIDTH`, `NULL_STATE_ID`, and the presence of `outer_residual` are compile-time constants. Changes to those values can create another Triton variant. `num_reqs`, the binary-search iteration count, and the token-map flag are marked `do_not_specialize`; token count changes the launch grid.
- The convolution and state writeback are separate kernels to avoid read/write races between programs of the same request. The output kernel tolerates graph-padded tokens beyond `query_start_loc[R]` by writing a zero convolution result.
- `residual` and `conv_state` are intentional in-place outputs. `inputs`, `conv_weights`, `state_indices`, and optional metadata are not modified. The transposed weight buffer is private to this call.

## Origin and Differences

- **Origin**: `vllm.models.qwen4_exp.common.ops.ple.ple_conv` with `mode="spec"` and its speculative Triton kernel.
- **Differences**: Ascend dispatch calls `ple_conv_spec` for speculative decode. It copies weights to contiguous `[K,C]`, reads channel-contiguous state with 512-channel tiles, maps each token to its request by binary search over `query_start_loc`, rolls the state window back with `num_accepted_tokens`, and writes the state in a separate kernel. Upstream vLLM is not modified; the same model code still runs on other platforms.
- **Reason**: Performance optimization for Qwen PLE MTP verification on Ascend. This is an Ascend-specific implementation of an upstream interface, with no new environment variable or temporary compatibility fallback.
- **Side effects**: `residual` and `conv_state` are intentionally modified as specified by the upstream interface. The operator does not change the process-global internal-format option beyond the wrapper's existing scoped save/restore.

## Test Cases

- CPU unit contract: `pytest -q tests/ut/ops/test_triton_ple_spec.py` checks both kernel launches, grids, SD layouts and strides, weight-copy ownership, token-map routing, optional residual, and argument/shape validation without launching an NPU kernel.
- NPU operator comparison: `pytest -q --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_triton_ple_spec.py` compares output and updated state with upstream `ple_conv(mode="spec")` for BF16/FP16, `spec_query_len` 2-4, channel tails, null slots, mapped/unmapped tokens, and optional outer residuals. Output acceptance uses the same mixed tolerance as decode; state writeback is checked exactly. The file passed 12/12 on Ascend 910B3 with bit-exact state.
- `test_triton_ple.py` additionally covers the model dispatch path and mixed decode/spec/prefill batches. Model-level accuracy, throughput, and recompilation measurements require a baseline and optimized service on equivalent hardware/configuration.

```bash
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_triton_ple_spec.py
```
