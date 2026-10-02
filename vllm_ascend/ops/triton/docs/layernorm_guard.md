# layer_norm_fwd_kernel

## Description

- **Location**: `vllm_ascend/ops/triton/fla/layernorm_guard.py` — `layer_norm_fwd_kernel`, host wrapper `_layer_norm_fwd`, and autograd-compatible forward wrapper `LayerNormFn`.
- **Function**: Fused LayerNorm or RMSNorm over the last dimension, with optional grouping, bias, and SiLU gating before or after normalization.
- **Formula** (per row and group, computed in fp32):
    - If gating is applied before normalization, `u = x * silu(z)`; otherwise, `u = x`.
    - LayerNorm: `x_hat = (u - mean(u)) * rsqrt(mean((u - mean(u))²) + eps)`.
    - RMSNorm: `x_hat = u * rsqrt(mean(u²) + eps)`.
    - `y = x_hat * weight + bias`, where `bias` is optional.
    - If gating is applied after normalization, `y = y * silu(z)`.
- **Algorithm flow** (processed row by row, independently):
  1. `_layer_norm_fwd` interprets `x` as `[M, hidden_size]`, splits the last dimension into groups of `group_size`, and launches a two-dimensional grid `(min(M, MAX_CORES), num_groups)`.
  2. Each program processes one group and loops over its assigned rows when `M > MAX_CORES`.
  3. The kernel loads one group into a power-of-two block, converts values to fp32, optionally applies the pre-normalization SiLU gate, and computes LayerNorm or RMSNorm statistics.
  4. It applies `weight`, optional `bias`, and the optional post-normalization SiLU gate, then stores the result in the output dtype. Mean and reciprocal standard deviation are stored in group-major order.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Two-dimensional activation tensor `[M, hidden_size]` | fp16 / bf16 / fp32 | ND |
| `weight` | Input | Per-channel scale `[hidden_size]` | fp16 / bf16 / fp32 | ND |
| `bias` | Optional input | Per-channel bias `[hidden_size]` | fp16 / bf16 / fp32 | ND |
| `z` | Optional input | Gate tensor with the same shape as `x` | fp16 / bf16 / fp32 | ND |
| `eps` | Attribute | Numerical-stability constant added before `rsqrt` | fp32 | scalar |
| `group_size` | Attribute | Number of channels normalized by each kernel group; defaults to `hidden_size` | int32 | scalar |
| `norm_before_gate` | Attribute | Apply normalization before the SiLU gate when `True`, otherwise apply the gate first | bool | scalar |
| `is_rms_norm` | Attribute | Select RMSNorm when `True`, otherwise select LayerNorm | bool | scalar |
| `out` | Optional output buffer | Preallocated output with the same shape as `x` | fp16 / bf16 / fp32 | ND |
| `mean` | Output | LayerNorm mean, flattened from `[num_groups, M]`; not allocated for RMSNorm | fp32 | ND |
| `rstd` | Output | Reciprocal standard deviation, flattened from `[num_groups, M]` | fp32 | ND |

## Constraints

- `_layer_norm_fwd` requires a two-dimensional `x`. `LayerNormFn` accepts higher-rank tensors and flattens all leading dimensions before launch.
- The last dimension of `x`, `weight`, optional `bias`, optional `z`, and optional `out` must be contiguous. `weight` and `bias` must have shape `[hidden_size]`; `z` and `out` must match `x`.
- `hidden_size` must be divisible by `group_size`.
- A group must fit within 64 KB: `group_size <= 65536 / x.element_size()`, which is at most 32768 elements for fp16/bf16 and 16384 elements for fp32.
- Reductions and affine computation use fp32; the result is stored in the output tensor's dtype.
- The launch grid is capped at `MAX_CORES = 65535`; grid-stride row iteration handles larger `M`.
- Only the forward inference path is implemented. The Triton kernel supports graph execution; launch meta-parameters are derived from tensor shapes on the host.
- Gating uses SiLU. The `activation` argument on `LayerNormFn.forward` is retained for interface compatibility but is not dispatched to the kernel.

## Origin and Differences

- **Origin**: Adapted from flash-linear-attention's `fla/modules/layernorm_gated.py`; vLLM carries the corresponding implementation in `vllm/third_party/flash_linear_attention/ops/layernorm_guard.py`.
- **Differences**:
    - NPU row scheduling caps the grid at `MAX_CORES` and lets each program loop over additional rows, rather than using the upstream `ROWS_PER_BLOCK` CUDA scheduling strategy.
    - The Ascend implementation keeps a compact forward-only wrapper without vLLM's JIT warmup dispatcher and input-guard integration.
    - Only SiLU gating is implemented; the upstream implementation also dispatches a sigmoid gate.

## Test Cases

`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fla_layernorm_guard.py` compares the kernel output with an fp32 PyTorch reference in three focused cases: fp32 LayerNorm with bias, bf16 grouped RMSNorm with a post-normalization SiLU gate, and fp16 LayerNorm with `MAX_CORES + 3` rows to exercise grid-stride iteration. The tolerances are `rtol=2e-3, atol=2e-2` for fp16, `rtol=2e-2, atol=5e-2` for bf16, and `rtol=1e-4, atol=1e-4` for fp32.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fla_layernorm_guard.py
```
