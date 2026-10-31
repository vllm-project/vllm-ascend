# muls_add

## Description

- **Function**: Fuses an element-wise scalar multiplication and residual addition into one Triton kernel. The public `muls_add_triton` wrapper implements the registered `torch.ops.vllm.muls_add` custom operator used by the Ascend Inductor fusion pass. The current tree also contains an identical copy in `mul_add.py` used directly by DeepSeek-V4.
- **Formula**: For every flattened element index `i`,

  $$
  \operatorname{out}_i = x_i \times \operatorname{scale} + y_i.
  $$

  The result is stored in the input tensor dtype.
- **Algorithm flow** (processed element by element, independently):
  1. Require `x` and `y` to have the same shape and flatten their contiguous storage into `n_elements` elements.
  2. Select `BLOCK_SIZE = max(hidden_size // 2, 1024)`, where `hidden_size` is the last input dimension.
  3. Launch at most one Triton program per available vector core. Each program grid-strides over logical blocks of `BLOCK_SIZE` elements.
  4. Load `x` and `y`, compute `x * scale + y`, and store the result. A tail mask protects the last partial block.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. The current single-operator accuracy test has been verified on Atlas A2.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | First tensor in `x * scale + y`; shape must match `y` | fp16 / bf16 / fp32 | Contiguous ND |
| `y` | Input | Residual tensor added to the scaled `x`; shape must match `x` | same as `x` | Contiguous ND |
| `scale` | Input (attribute) | Scalar multiplier applied to every element of `x` | Python `float` | Scalar |
| `output` | Output | Element-wise fused result with the same shape as `x` and `y` | same as `x` | Contiguous ND |

## Constraints

- `x` and `y` must be non-empty NPU tensors with the same shape, dtype, device, and contiguous logical layout. The kernel uses flattened pointer arithmetic and does not consume per-dimension strides.
- Inputs must have at least one dimension because the wrapper uses `x.shape[-1]` to derive `BLOCK_SIZE`.
- The registered Inductor fusion pass enables fp16, bf16, and fp32 model dtypes. The current accuracy test covers fp16 and bf16.
- The last partial block is masked, so `x.numel()` does not need to be divisible by `BLOCK_SIZE`.
- The custom operator provides a fake implementation and can participate in Dynamo/AOT graph capture. The same mathematical kernel is also called directly by the DeepSeek-V4 implementation in `mul_add.py`.

## Origin and Differences

- **Origin**: Developed in vLLM-Ascend as the NPU Triton replacement for the graph pattern `x * scale + y` in PR #5518. An identical source copy was later added for the DeepSeek-V4 direct call path in PR #9270.
- **Differences**:
    - NPU adaptation for performance: combines multiply and add into one memory pass and caps the launch at the available vector-core count;
    - Modified for a specific vllm-ascend logic or different input parameters: exposes a registered `torch.ops.vllm.muls_add` operator for the Ascend Inductor pass while retaining the direct Python wrapper used by DeepSeek-V4.

## Test Cases

The test directly calls `muls_add_triton` and compares it with the PyTorch expression `x * scale + y`. It covers fp16 and bf16, token counts from 1 to 4000, hidden size 2048, several scale values, multi-program execution, and a large grid-stride workload. The current floating-point tolerance is `rtol=1e-3, atol=1e-3`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_muls_add.py
```
