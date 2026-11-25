# fused_gdn_gating

## Description

- **Location**:
`vllm_ascend/ops/triton/fused_gdn_gating.py` — `fused_gdn_gating_kernel`, 1:1 wrapper `fused_gdn_gating_patch`
- **Function**: Computes the gating pair of the GDN (Gated DeltaNet) linear attention in one fused kernel: the log-decay gate `g` and the interpolation coefficient `beta_output`. It replaces the host-side torch chain `g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)` in the recurrent-attention step of `vllm_ascend/attention/gdn.py` (Qwen3-Next / Qwen3.5 hybrid GDN models). Entry: `DeviceOperator.fused_gdn_gating` (A2/950PR&950DT Products) → 1:1 wrapper `fused_gdn_gating_patch` → `fused_gdn_gating_kernel`.
- **Formula** (per token `t`, per head `h`, computed in fp32):
    - `x = a[t, h] + dt_bias[h]`
    - `softplus(x) = (1/beta) * log(1 + exp(beta * x))` when `beta * x <= threshold`, else `x` (overflow guard, identical to `torch.nn.functional.softplus(beta, threshold)`)
    - `g[t, h] = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h])` (stored as fp32)
    - `beta_output[t, h] = sigmoid(b[t, h])` (stored in `b`'s dtype)
- **Algorithm flow** (processed row by row, independently):
  1. Map each program to a tile of tokens and heads. The head count determines the tile size.
  2. Read each projection with its token stride. The wrapper packs projections whose head stride exceeds one. Mask the final tile on both axes.
  3. Compute `softplus`, `exp`, `sigmoid` in fp32 and store `g` and `beta_output` element-wise.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products (the A2/950PR&950DT Products `DeviceOperator` entries route to this same Triton kernel). Used by the GDN recurrent-attention path of `vllm_ascend/attention/gdn.py`; works in both eager and graph-capture modes.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `A_log` | Input | Per-head log decay scale `[num_heads]` | fp16 / bf16 / fp32 | ND |
| `a` | Input | a-projection pre-activation `[num_tokens, num_heads]` (tokens flattened from `batch * seq_len`) | fp16 / bf16 / fp32 | ND |
| `b` | Input | b-projection pre-sigmoid `[num_tokens, num_heads]` | fp16 / bf16 / fp32 | ND |
| `dt_bias` | Input | Per-head dt bias `[num_heads]` | fp16 / bf16 / fp32 | ND |
| `beta` | Input | Softplus beta (default `1.0`) | fp32 | scalar |
| `threshold` | Input | Softplus threshold (default `20.0`) | fp32 | scalar |
| `g` | Output | Log-decay gate `[1, num_tokens, num_heads]`, always fp32 (consumed as fp32 by the downstream recurrent kernel); first element of the returned tuple | fp32 | ND |
| `beta_output` | Output | Interpolation coefficient `sigmoid(b)` `[1, num_tokens, num_heads]`; second element of the returned tuple | same as `b` | ND |

## Constraints

- The projections `a` and `b` have shape `[num_tokens, num_heads]` and may have different strides. The per-head vectors `A_log` and `dt_bias` must be contiguous.
- The kernel computes both gates in fp32 and stores them in contiguous output tensors.
- Each tile covers up to 64 heads and 16 tokens, with at most 256 elements.
- The compiled kernel specializes on the head count and input strides. The token count can vary.
- Only for inference (prefill/decode) on NPU.

## Origin and Differences

- **Origin**: Math adapted from vllm's `qwen3_next.py` (`Qwen3NextGatedDeltaNet` gating, [upstream source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py)). The kernel first landed with Qwen3-Next support (#2917) and was extracted into the standalone op file `vllm_ascend/ops/triton/fused_gdn_gating.py` by #4304.
- **Differences**:
    - The kernel computes both gates in one launch and reads the original projection slices directly.
    - Modified for a specific vllm-ascend logic or different input parameters: returns `g` pre-shaped `[1, num_tokens, num_heads]` and forced to fp32 as expected by the Ascend recurrent gated-delta-rule kernel, while `beta_output` keeps `b`'s dtype.

## Test Cases

The test compares the Triton kernel (through its 1:1 wrapper `fused_gdn_gating_patch`) against the independent PyTorch reference `fused_gdn_gating_pytorch` (from `vllm_ascend/_310p/ops/fla/fused_gdn_gating.py`) for both outputs, fp16 inputs, `num_tokens=37`, `num_heads=8`, unified elementwise tolerance `rtol=atol=1e-2`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_gdn_gating.py
```
