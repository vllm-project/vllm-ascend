# QSA RMSNorm MRoPE

## Description

- **Location**: `vllm_ascend/ops/triton/linearnorm/qsa_rmsnorm_mrope.py`.
- **Function**: Fuses weighted RMS normalization with Qwen3.8-Next QSA rotary position embedding for prefill and decode query projections.
- **Formula** (per token `t`, head `h`, and channel `d`, with the RMS reduction in fp32):

  ```text
  r[t,h] = rsqrt(mean(x[t,h,:]^2) + eps)
  n[t,h,d] = bf16(x[t,h,d] * r[t,h] * (weight[d] + 1))
  y_first  = bf16(n_first * cos - n_second * sin)
  y_second = bf16(n_second * cos + n_first * sin)
  ```

  Channels outside `rope_dim` retain their normalized values. For MRoPE, temporal, height, and width cache rows are selected lane by lane according to `mrope_section`.
- **Algorithm flow** (processed independently for each token and head):
  1. Launch one Triton program for every `(token, head)` pair.
  2. Load one head row, compute its fp32 RMS reduction, apply `weight + 1`, and convert the normalized values to BF16.
  3. Load the temporal, height, and width cosine/sine cache rows contiguously.
  4. Select the required MRoPE axis in registers with `tl.where`.
  5. Apply rotary embedding to the first `rope_dim` channels and store the fused result.
- **Supported modes**: Atlas A3 and 950DT (A5) Qwen3.8-Next inference, covering the model-shaped prefill and decode inputs below.

## Parameters

All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `tensor` | Input | QSA projection, shape `[num_tokens, num_heads, head_dim]` | BF16 | Contiguous head rows |
| `weight` | Input | RMSNorm weight, shape `[head_dim]`; the kernel applies `weight + 1` | BF16 | Contiguous 1D |
| `cos_sin` | Input | MRoPE cache `[3, num_tokens, rope_dim]` or RoPE cache `[num_tokens, rope_dim]` | BF16 | Strided ND with contiguous rotary width |
| `eps` | Attribute | RMS normalization epsilon | float | Scalar |
| `mrope_section` | Attribute | Temporal, height, and width MRoPE section sizes | `list[int]` | Three elements for MRoPE |
| `rope_dim` | Attribute | Even rotary width, no greater than `head_dim` | int | Scalar |
| Return value | Output | Fused normalized and rotated projection, same shape as `tensor` | BF16 | Same layout as `tensor` |

## Constraints

- `tensor` and `weight` must be BF16. The last dimension of `tensor` and `weight` must match.
- `tensor` must have three dimensions and contiguous head rows.
- `cos_sin` must have two dimensions for RoPE or three dimensions for MRoPE. MRoPE requires exactly three cache axes and three section values.
- The token count in `cos_sin` must equal `tensor.shape[0]`.
- `rope_dim` must be positive, even, and no greater than `head_dim`.
- An empty token dimension returns an empty output without launching the kernel.
- Forward inference only; no autograd backward implementation is provided.
- The custom op provides a fake implementation for compilation and shape propagation. The single-operator tests below validate eager device execution.

## Origin and Differences

- **Origin**: Developed for the Qwen3.8-Next QSA projection path in `vllm_ascend/models/qwen4_exp/qsa.py`.
- **Differences**:
    - Fuses RMSNorm and MRoPE into one Triton operator instead of materializing an intermediate normalized tensor.
    - Uses a two-dimensional `(token, head)` launch grid.
    - Loads all three MRoPE cosine/sine rows contiguously and selects the axis in registers, replacing scalar lane-dependent cache addressing.
    - Preserves the existing BF16 arithmetic order, rotary layout, wrapper signature, and model semantics.
- **Reason**: Performance optimization for Qwen3.8-Next QSA prefill and decode on Ascend. The contiguous cache-row loads remove the scalar lane-dependent addressing bottleneck without introducing a new configuration switch.
- **Side effects**: The operator allocates and returns a new output tensor. Input tensors and cache contents are not modified, and no process-global state or external tensor layout is changed.

## Test Cases

The device test compares the Triton output with an independent CPU fp32 PyTorch reference and uses `rtol=2e-2`, `atol=2e-2`.

| Device | Case | Shape | Maximum absolute difference |
| --- | --- | --- | ---: |
| Ascend A3 | Prefill | `(4096, 4, 128)` | 0.03125 |
| Ascend A3 | Decode | `(4096, 1, 128)` | 0.0078125 |
| Ascend950DT (A5) | Prefill | `(4096, 4, 128)` | 0.015625 |
| Ascend950DT (A5) | Decode | `(4096, 1, 128)` | 0.0078125 |

Both cases passed on A3 and A5 using the Qwen3.8-Next 2026-11-02 validation images.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_rmsnorm_mrope.py
```
