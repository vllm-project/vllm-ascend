# GDN RecoverSSM Verify

## Description

- **Function**: Verify a speculative Gated DeltaNet (GDN) window from a paged
  recurrent checkpoint. Return per-token outputs and emit compact FP32 replay
  records for [state commit](gdn_recoverssm_commit.md).
- **Formula**: Apply raw GDN gates, optional Q/K normalization, and the forward
  recurrence below.
- **Algorithm flow**: Each request and value head advances a temporary state
  sequentially over the request's tokens, with value-dimension tiling. The
  persistent checkpoint remains unchanged.
- **Supported modes**: Eager execution with Ascend 950 and ordinary non-950
  launch paths. See [historical NPU results](#recorded-npu-results)
  for tested configurations.

### Recurrence

For one value head, let `S` be `[V, K]`. Its query/key head is
`value_head // (HV / HK)`. The gate is scalar per value head:

```text
x = a + dt_bias
softplus_x = log(1 + exp(softplus_beta * x)) / softplus_beta
             if softplus_beta * x <= softplus_threshold else x
g = -exp(A_log) * softplus_x
d = exp(g)
beta = sigmoid(b)

q_hat = q * rsqrt(sum(q * q) + 1e-6)  # if normalization enabled
k_hat = k * rsqrt(sum(k * k) + 1e-6)  # if normalization enabled
q_scaled = q_hat * scale
D = S * d
u = beta * (v - D @ k_hat)
S_next = D + u[:, None] * k_hat[None, :]
output = S_next @ q_scaled
```

Without normalization, `q_hat=q` and `k_hat=k`. Default scale is `K**-0.5`.
State arithmetic and normalized keys are FP32; normalized keys are not rounded
back to activation dtype before recording. Outputs are stored in the output
tensor's dtype.

### Kernel launch

The grid is `(ceil(V / BV), HV, B)`. Verify uses `BK=next_power_of_2(K)` and normally
`BV=min(next_power_of_2(V), 64)`. The Ascend 950 branch can select a 128-wide value
tile when its occupancy checks retain at least 96 programs. Verify uses four
warps and disables FP fusion. Shapes, inner strides, head geometry and activation
options participate in specialization; variable metadata values remain on device.

## Parameters

`B` is the number of speculative rows, `T` the flattened token count, `P` the
page count, `Q` the window capacity, `HK/HV` the local query-key/value head
counts, and `K/V` the key/value widths.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `A_log`, `dt_bias` | Input | Per-value-head log decay scale and gate bias, each `[HV]` | Floating point, converted to FP32 in the kernel | Contiguous 1D |
| `a`, `b` | Input | Raw gate and beta inputs, each with `T * HV` elements, reshaped to `[T, HV]` | Floating point, converted to FP32 in the kernel | ND with unit head stride after reshape |
| `q`, `k` | Input | Query and key, each `[1, T, HK, K]` | BF16 in the NPU tests, converted to FP32 in the kernel | ND with inner strides `(K, 1)` |
| `v` | Input | Values `[1, T, HV, V]` | BF16 in the NPU tests, converted to FP32 in the kernel | ND with inner strides `(V, 1)` |
| `checkpoint_state` | Input | Persistent state `[P, HV, V, K]`, read during verification | FP32 or BF16 | Paged ND with explicit strides |
| `replay_cache` | Output | Caller-supplied buffer `[P, HV, Q, V + K + 1]`, written for valid pages and tokens | FP32 | Paged ND with explicit strides |
| `query_start_loc` | Input | Cumulative token offsets `[B + 1]` | INT32 or INT64 | Strided 1D |
| `state_indices` | Input | Source page IDs `[B]`. Nonpositive IDs select inactive rows | INT32 or INT64 | Strided 1D |
| `spec_query_len` | Attribute | Window capacity `Q` | Python int | Positive scalar |
| `scale` | Attribute | Query scale. Default `None` selects `K**-0.5` | Python float or `None` | Scalar / N/A |
| `softplus_beta` | Attribute | Positive softplus beta, default `1.0` | Python float | Scalar |
| `softplus_threshold` | Attribute | Threshold for the softplus linear branch, default `20.0` | Python float | Scalar |
| `use_qk_l2norm_in_kernel` | Attribute | Whether to normalize Q/K, default `True` | Python bool | Scalar |
| `out` | Output | Returns the supplied output buffer. Default `None` allocates a contiguous tensor with `v`'s shape and dtype | Supplied buffer dtype or `v.dtype` | `[1, T, HV, V]` with inner strides `(V, 1)` |

For each page, head and token, the FP32 replay record is:

| Slice | Stored value |
| --- | --- |
| `[..., :V]` | Correction `u` |
| `[..., V:V + K]` | Effective key `k_hat`, repeated across associated value heads |
| `[..., V + K]` | Effective scalar decay `d = exp(g)`, not the log gate |

Only the first value tile writes key and decay; value tiles write disjoint
correction slices. No full `[V, K]` state is stored per speculative token.
Replay payload per page is `4 * HV * Q * (V + K + 1)` bytes, excluding checkpoint,
convolution, outputs and metadata. This is a storage formula, not measured memory.

## Constraints

- All tensors share a device. Supply contiguous `A_log` and `dt_bias` because
  the kernel indexes them directly by head.
- Query offsets start at zero, end at `T`, and are nondecreasing. Every request
  span is at most `Q`, the aggregate capacity is `T <= B * Q`, and positive
  page IDs must be in range.
- A nonempty query on a null page produces zero output and leaves replay
  unchanged. Empty verification returns its output buffer without launching.
- Finish verification before commit consumes records for the same checkpoint
  and window. Autograd is N/A.

## Origin and Differences

- **Origin**: The raw-gate interface and page layout follow
  [upstream vLLM GDN RecoverSSM](https://github.com/vllm-project/vllm/pull/61309).
- **Differences**: The Ascend kernel accepts explicit page strides and selects
  a larger value tile on Ascend 950 when the request/head grid is large enough.

## Test Cases

### Reference implementations

- [cpu_reference](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/gdn_recoverssm_test_utils.py#L119) evaluates the forward GDN recurrence on CPU in FP64. Tests compare
  the token output and the correction, effective key, and decay fields.
- The AscendC comparison runs `recurrent_gated_delta_rule` one token at a
  time on the same pre-normalized BF16 Q/K, with internal normalization
  disabled in the verify wrapper.

### Accuracy criteria

[check_accuracy](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/gdn_recoverssm_test_utils.py#L31) checks shape and finiteness, then computes error on
CPU in FP64. Tolerances use the actual output's storage dtype.

| Output dtype | rtol | atol | Fixed floor for the maximum absolute error limit |
| --- | --- | --- | --- |
| FP16 | `2^-9` | `2^-9` | `0.1` |
| BF16 | `2^-6` | `2^-6` | `1.0` |
| FP32 | `2^-10` | `2^-16` | `0.01` |

Each arithmetic comparison requires at least 99% of elements to satisfy
`abs(actual - expected) <= atol + rtol * abs(expected)`, and the maximum
absolute error must be at most
`max(floor, 32 * torch.finfo(actual.dtype).eps)`. The shared checker also covers FP16 in its self-test.
Padding and cache regions expected to stay unchanged use exact checks.
The checker records per-comparison metrics as `accuracy/<label>` JUnit properties.

### Coverage

| Test function | Cases | Verification performed |
| --- | ---: | --- |
| [test_accuracy_gates](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_verify_npu.py#L31) | 3 | Self-tests the 99% boundary and worst-element limit for FP16, BF16, and FP32. |
| [test_verify_preserves_head_padding](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_verify_npu.py#L45) | 1 | Padded head strides, strided source indices, and exact checkpoint and replay padding checks. |
| [test_verify_ragged_tiles_and_small_norms](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_verify_npu.py#L72) | 6 | Ragged K/V tiles, small and zero Q/K norms, custom gates, strided INT64 offsets and supplied output. |
| [test_verify_small_norms_against_ascendc](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_verify_npu.py#L112) | 4 | Token outputs against the AscendC recurrence at four input magnitudes. |
| [test_verify_empty_and_null_rows](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_verify_npu.py#L119) | 3 | Empty batch and page IDs 0 or -1, with zero or empty outputs and unchanged cache backing. |
| [test_verify_configured_window_dispatch](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_verify_npu.py#L134) | 6 | Windows 1/2/4/5/9, FP32 and BF16 checkpoints, and tile/grid checks. |

The 23 parameterized cases include Qwen3.6-35B-A3B TP4 (`HK/HV=4/8`, `Q=5`)
and Qwen3.8-27B TP2 (`HK/HV=8/24`, `Q=9`), both with `K=V=128`.

### Test command

Run from the repository root with the Ascend NPU test stack.

```bash
pytest -sv -o junit_family=xunit1 \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_verify_npu.py \
  --junitxml=gdn_recoverssm_verify.xml
```

### Recorded NPU results

The JUnit report recorded **6 passed**, **0 failed**, **0 errors**, and **0 skipped** in
**47.648 s**. The named test container used
an Ascend A3 OpenEuler image with CANN 9.2.1.

The JUnit report recorded **6 passed**, **0 failed**, **0 errors**, and **0 skipped** in
**27.392s**. The named test container used
an Ascend A5 OpenEuler image with CANN 9.2.1.

| Current test | Historical verify check | Maximum absolute error |
| --- | --- | ---: |
| `test_verify_configured_window_dispatch` | BF16 output across six configured-window cases | `2.4221478e-4` |
| `test_verify_small_norms_against_ascendc` | BF16 output with Q/K scaled by `1e-8` | `2.2737368e-13` |
| `test_verify_small_norms_against_ascendc` | BF16 output with zero Q/K | `0` |

## Example

With `query_start_loc=[0, 3]`, `state_indices=[1]` and `Q=5`,
verify reads checkpoint page 1 and returns three token outputs.
It writes records into `replay_cache[1, :, :3, :]` and leaves checkpoint page 1
unchanged. The records are available to commit after sampling.
