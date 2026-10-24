# swiglu_quant

## Description

- **Location**: [`activation/swiglu_quant.py`](../activation/swiglu_quant.py), wrapper `swiglu_quant` and kernel `_swiglu_quant_kernel`.
- **Function**: Computes SwiGLU from concatenated gate/up activations, optionally followed by per-token symmetric INT8 quantization. The W8A8 Dynamic MoE non-fused, non-MC2 path calls this implementation when Triton is available.
- **Formula** for input `x` with shape `[S, 2N]`, per token `i`:
    - `gate = x[i, :N]`, `up = x[i, N:]`.
    - `y = gate * sigmoid(gate.to(fp32)) * up`.
    - With `need_quant=True`: `scale[i] = max(abs(y)) / 127`; `out[i] = saturating_int8(cast_to_input_dtype(y / scale[i]))`.
    - With `need_quant=False`: `out[i] = cast_to_input_dtype(y)`.
- **Algorithm flow**:
    1. Read the number of valid token rows from the expert group metadata: the last cumulative count for `group_list_type=0`, or the sum of counts for `group_list_type=1`.
    2. Distribute contiguous row ranges across vector cores. Each program processes its assigned rows independently.
    3. Load one concatenated gate/up row, split it with `extract_slice`, and cast the gate to FP32 before sigmoid.
    4. Compute SwiGLU. If quantization is enabled, reduce the row maximum, write its FP32 scale, and convert output chunks to INT8. Otherwise, store the activation in the input dtype.
- **Supported modes**: Ascend NPU inference. Targeted eager-mode tests for this change passed on Atlas A2 (Ascend910B4-1) and 950PR with Triton Ascend 3.6.0. Additional hardware and graph-capture validation for this change: N/A.

## Parameters

`x`, `group_list`, and `group_list_type` are required. `need_quant` is optional and defaults to `True`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Contiguous concatenated gate/up activations, shape `[S, 2N]` | FP16 / BF16 | ND |
| `group_list` | Input | Per-expert cumulative token counts or individual counts, shape `[num_experts]` | INT32 / INT64 | ND |
| `group_list_type` | Attribute | `0` for cumulative counts; `1` for individual counts | Python int | Scalar |
| `need_quant` | Attribute | Enable per-token INT8 quantization; default `True` | Python bool | Scalar |
| `out` | Output | Shape `[S, N]`; INT8 when quantization is enabled, otherwise the input dtype | INT8 / FP16 / BF16 | ND |
| `scale` | Output | Shape `[S]`; per-token quantization scale. Its values are only defined when `need_quant=True` | FP32 | ND |

## Constraints

- All input tensors must be on the same NPU. The kernel uses flat row-major indexing; the wrapper does not make non-contiguous inputs contiguous.
- `x` must be two-dimensional with an even last dimension. The validated floating-point input types are FP16 and BF16; INT8 is the quantized output type, not a validated input type for this operator.
- `group_list` must be one-dimensional and non-empty. Individual expert counts must be non-negative; cumulative counts must be non-negative and non-decreasing. Zero-token experts are allowed.
- For all allocated output rows to be valid, the last cumulative count or the sum of individual counts must equal `S`. The wrapper validates the metadata dtype and `group_list_type`, but does not validate these count values or initialize unused output rows.
- Quantization uses the row maximum directly as the scale numerator; the implementation has no explicit zero-scale guard for an all-zero activation row.
- When `need_quant=False`, the returned `scale` tensor is allocated but not written and must not be used.

### Triton Ascend 3.6.0 compatibility

Triton Ascend 3.6.0 does not support BF16 input to `tl.sigmoid`: sigmoid evaluates `1 / (1 + exp(-x))`, and its exponential step rejects BF16 input. Passing the BF16 gate directly to sigmoid therefore fails during compilation.

This operator explicitly casts the gate to FP32 before calling sigmoid:

```python
out = x1 * tl.sigmoid(x1.to(tl.float32)) * x2
```

The cast applies to both FP16 and BF16 inputs and moves FP32 conversion before the exponential and addition. The sigmoid result remains FP32 for the following multiplications. This works around the compiler's BF16 restriction while preserving the wrapper's floating-point input support; it does not add native BF16 sigmoid support to Triton Ascend.

Final output dtypes are unchanged: quantized output is INT8 with FP32 scales, and non-quantized output has the input dtype. Intermediate numerical values, including derived scales and INT8 results, can differ due to the higher precision of the sigmoid calculation.

## Origin and Differences

- **Origin**: The vLLM-Ascend Triton implementation was introduced in [PR #5161](https://github.com/vllm-project/vllm-ascend/pull/5161). An upstream vLLM source operator used as its base: N/A.
- **Differences**:
    - Uses Ascend vector cores and `extract_slice` to process gate/up rows within a single kernel launch.
    - Accepts expert token metadata in either cumulative-count or individual-count form.
    - Fuses SwiGLU and optional per-token quantization. The normalized activation is cast through the input dtype before saturating INT8 conversion.
    - Casts the sigmoid input to FP32 to avoid the BF16 compilation failure with Triton Ascend 3.6.0, as described above.

## Test Cases

[`test_swiglu_quant.py`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_swiglu_quant.py) contains eight cases:

- Quantized correctness: FP16 `[1, 4096]` and BF16 `[65, 14336]`, each with both group-list modes. The BF16 case includes zero-token experts and multiple token rows.
- Non-quantized correctness: FP16 and BF16 `[9, 4096]`, including a zero-token expert.
- Invalid metadata: unsupported `group_list_type` and unsupported group-list dtype.

The numerical reference computes SwiGLU in FP32. Quantized output is checked with `rtol=0, atol=1`; scale tolerances are `(rtol=5e-3, atol=1e-5)` for FP16 and `(rtol=2e-2, atol=1e-4)` for BF16. Non-quantized output uses `rtol=atol=5e-3` for FP16 and `rtol=atol=2e-2` for BF16. Output shapes and dtypes are also checked.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_swiglu_quant.py
```
