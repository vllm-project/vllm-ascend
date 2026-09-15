# _layer_norm_fwd_1pass_kernel_npu

## Description

- **Function**: Computes grouped LayerNorm or RMSNorm, with optional bias and SiLU gating, in one pass.
- **Formula**: Let `silu(z)=z*sigmoid(z)`. If `NORM_BEFORE_GATE=True`, compute `(norm(x)*w+b)*silu(z)`; otherwise normalize `x*silu(z)` before applying `w` and `b`. LayerNorm uses `(x-mean)/sqrt(var+eps)`; RMSNorm uses `x/sqrt(mean(x^2)+eps)`.
- **Algorithm flow** (processed row by row, independently): load a row group and affine parameters in fp32, optionally pre-gate, reduce statistics, normalize and apply affine terms, optionally post-gate, and store output/statistics.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950; eager and graph-capture inference.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `X` | Input | Flattened input `[M, D]`, where `D = groups * N` | fp16 / bf16 / fp32 | ND |
| `Y` | Output | Normalized output with the same shape and dtype as `X` | same as `X` | ND |
| `W` | Input | Affine weight `[D]` | same as `X` | ND |
| `B` | Input | Optional affine bias `[D]` | same as `X` | ND |
| `Z` | Input | Optional SiLU gate, same shape as `X` | same as `X` | ND |
| `Mean`, `Rstd` | Output | Per-row/group mean and reciprocal standard deviation | fp32 | ND |
| `stride_x_row`, `stride_y_row`, `stride_z_row` | Attribute | Row strides | int | scalar |
| `M`, `N`, `eps` | Attribute | Row count, group width, and stability epsilon | int / fp32 | scalar |
| `BLOCK_M`, `BLOCK_N` | Attribute | Compile-time row and column tiles | int | scalar |
| `HAS_BIAS`, `HAS_Z`, `NORM_BEFORE_GATE`, `IS_RMS_NORM` | Attribute | Compile-time feature switches | bool | scalar |

## Constraints

- Inputs, outputs, weights, bias, and gate must be contiguous in the last dimension; `weight.shape == (D,)` and optional tensors must match `X`.
- Total feature width must be divisible by group width. Group width must fit `65536 / element_size` elements.
- `Mean` is unused for RMSNorm. `eps` must be positive.

## Origin and Differences

- **Origin**: Adapted from flash-linear-attention's gated LayerNorm, itself based on the Triton LayerNorm tutorial.
- **Differences**:
    - NPU adaptation for performance: processes 64 rows per Ascend tile and performs normalization, affine transformation, and gating in one pass.
    - Modified for vllm-ascend logic: supports grouped LayerNorm/RMSNorm and optional pre- or post-normalization SiLU gating.

## Test Cases

The direct NPU test compares the output and fp32 statistics against a PyTorch
reference. It covers LayerNorm and RMSNorm, group sizes `{60, 96, 128}`, fp16,
bf16, and fp32 inputs, optional bias and SiLU gate, both gate orders, and reuse
of a caller-provided output tensor. Tolerances are `(rtol, atol) = (2e-3,
2e-2)` for fp16, `(2e-2, 5e-2)` for bf16, and `(1e-4, 1e-4)` for fp32. A
negative test verifies rejection when the hidden size is not divisible by the
group size.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_layernorm_gated.py
```
