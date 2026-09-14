# fused_gdn_gating

## Description

- **Location**:
`vllm_ascend/ops/triton/fused_gdn_gating.py` — `fused_gdn_gating_kernel`, 1:1 wrapper `fused_gdn_gating_patch`
- **Function**: Computes the gating pair of the GDN (Gated DeltaNet) linear attention in one fused kernel: the log-decay gate `g` and the interpolation coefficient `beta_output`. It replaces the host-side torch chain `g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)` in the recurrent-attention step of `vllm_ascend/ops/gdn.py` (Qwen3-Next / Qwen3.5 hybrid GDN models). Entry: `DeviceOperator.fused_gdn_gating` (A2/A5) → 1:1 wrapper `fused_gdn_gating_patch` → `fused_gdn_gating_kernel`.
- **Formula** (per token `t`, per head `h`, computed in fp32):
    - `x = a[t, h] + dt_bias[h]`
    - `softplus(x) = (1/beta) * log(1 + exp(beta * x))` when `beta * x <= threshold`, else `x` (overflow guard, identical to `torch.nn.functional.softplus(beta, threshold)`)
    - `g[t, h] = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h])` (stored as fp32)
    - `beta_output[t, h] = sigmoid(b[t, h])` (stored in `b`'s dtype)
- **Algorithm flow** (processed row by row, independently):
  1. Grid `(num_vectorcore, seq_len=1)`: each program owns `ceil(num_tokens / num_vectorcore)` token rows, iterated in `ROW_ITER` tiles of `BLK_BATCHES=64` rows.
  2. Within a row tile, heads are tiled by `BLK_HEADS=8` (`COL_ITER = ceil(num_heads / 8)` iterations); per tile, load the `A_log` / `dt_bias` head slices and the `a` / `b` blocks with row/head masks.
  3. Compute `softplus`, `exp`, `sigmoid` in fp32 and store `g` and `beta_output` element-wise.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 (the A2/A5 `DeviceOperator` entries route to this same Triton kernel). Used by the GDN recurrent-attention path of `vllm_ascend/ops/gdn.py`; works in both eager and graph-capture modes.

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

- `a`, `b`: `[num_tokens, num_heads]`; `A_log`, `dt_bias`: `[num_heads]` with the same `num_heads`. All inputs are indexed with flat offsets and must be contiguous.
- The wrapper fixes `seq_len = 1` (decode-style flattened token batch); internal computation is fp32 regardless of input dtype.
- Kernel-internal tiling constants: `BLK_HEADS=8`, `BLK_BATCHES=64`, `ROW_ITER = ceil(ceil(num_tokens / num_vectorcore) / 64)`.
- `seq_len`, `NUM_HEADS`, `NUM_BATCHES`, `beta`, `threshold`, `ROW_ITER` are `do_not_specialize`, so varying token counts or head counts do not trigger Triton recompilation.
- Only for inference (prefill/decode) on NPU.

## Origin and Differences

- **Origin**: Math adapted from vllm's `qwen3_next.py` (`Qwen3NextGatedDeltaNet` gating, see source-file header). The kernel first landed with Qwen3-Next support (#2917) and was extracted into the standalone op file `vllm_ascend/ops/triton/fused_gdn_gating.py` by #4304.
- **Differences**:
    - NPU adaptation for performance: fuses `softplus + exp + mul` and `sigmoid` into a single Triton launch on the vector cores (replaces the multi-op host-side torch chain); persistent-style row tiling (`BLK_BATCHES` rows per iteration per core) keeps the program count equal to the vector-core count to minimize launch overhead;
    - Modified for a specific vllm-ascend logic or different input parameters: returns `g` pre-shaped `[1, num_tokens, num_heads]` and forced to fp32 as expected by the Ascend recurrent gated-delta-rule kernel, while `beta_output` keeps `b`'s dtype.

## Test Cases

The test compares the Triton kernel (through its 1:1 wrapper `fused_gdn_gating_patch`) against the independent PyTorch reference `fused_gdn_gating_pytorch` (from `vllm_ascend/_310p/ops/fla/fused_gdn_gating.py`) for both outputs, fp16 inputs, `num_tokens=37`, `num_heads=8`, unified elementwise tolerance `rtol=atol=1e-2`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_gdn_gating.py
```
