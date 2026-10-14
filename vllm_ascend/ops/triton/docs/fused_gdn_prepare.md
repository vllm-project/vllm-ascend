# fused_gdn_prepare

## Description

- **Location**:
`vllm_ascend/ops/triton/fused_gdn_prepare.py` — `_fused_gdn_prepare_kernel`, entry `fused_gdn_prepare_impl` (wrapped by `AscendGatedDeltaNetAttention._fused_prepare`)
- **Function**: Replaces the GDN prepare chain of `AscendGatedDeltaNetAttention._forward_core` (Qwen3-Next / Qwen3.5 hybrid GDN models) with a single Triton launch. Per token subset it: splits the packed conv output `mixed_qkv` into q/k/v **in place** through the section offsets (q at 0, k at `key_dim/tp`, v at `2*key_dim/tp` inside each row — no host-side split/copy), L2-normalizes q/k over the head dim, streams v through to a contiguous output, and computes the gating pair. On the spec path (`HAS_INDEX=True`) it additionally gathers the a/b rows of the token subset **inline** via the token index map, replacing the four output-side `index_select` dispatches of the legacy chain. Legacy chain replaced end to end: `rearrange_mixed_qkv` + `l2norm_fwd`(q) + `l2norm_fwd`(k) + `DeviceOperator.fused_gdn_gating` (+ `g/beta.index_select` on spec/mixed batches).
- **Formula** (per token `t`, per head `h`, computed in fp32):
    - `q_n[t, h, :] = q[t, h, :] * rsqrt(sum_d(q[t, h, d]^2) + eps)`, `eps = 1e-6` (identical to `l2norm_fwd`; same for `k`)
    - `x = a[t_src, h] + dt_bias[h]`
    - `softplus(x) = log(1 + exp(x))` when `x <= threshold=20.0`, else `x` (overflow guard, identical to `torch.nn.functional.softplus`)
    - `g[t, h] = -exp(A_log[h]) * softplus(x)` (stored as fp32)
    - `beta[t, h] = sigmoid(b[t_src, h])` (stored in `b`'s dtype)
    - `t_src = INDEX[t]` when `HAS_INDEX` (a/b are the full batch while `mixed_qkv` is a gathered subset), else `t_src = t`
- **Algorithm flow** (processed row by row, independently):
  1. Grid `(min(num_vectorcore, ceil(T / BLOCK_T)),)`: each program owns `ceil(T / grid)` token rows, walked in `NUM_CHUNKS` chunks of `BLOCK_T` rows (`tl.range(..., num_stages=2)` software pipelining).
  2. Within a chunk, the q/k/v sections are loaded as whole `[BLOCK_T, H, D]` blocks straight out of `mixed_qkv` (sections are contiguous inside each packed row), normalized/copied, and stored to the fresh `[1, T, H, D]` outputs.
  3. Gating rows are loaded (direct, or gathered through `INDEX`), computed in fp32, and stored as `[1, T, Nv]`.
  4. The launch path caches the `CompiledKernel` per (config, `BLOCK_T` tier, dtypes) and reuses it via `CompiledKernel.run` on cache hits, bypassing the Triton `JITFunction` dispatch; `BLOCK_T` tiers and `HAS_INDEX` variants are precompiled on first use, and the standard launch path is kept as fallback.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products (the non-310P GDN patch path of `vllm_ascend/patch/worker/patch_qwen3_5.py`; 310P keeps `AscendGatedDeltaNetAttention310`). Works in both eager and graph-capture modes.

## Parameters

> [!NOTE]
> All parameters are required except `INDEX`, which is required only when `mixed_qkv` is a gathered token subset (spec path).

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `mixed_qkv` | Input | Packed conv output `[T, key_dim/tp + key_dim/tp + value_dim/tp]`; per row laid out as q \| k \| v | fp16 / bf16 | ND |
| `a` | Input | a-projection pre-activation of the **full batch** `[T_all, Nv]` (rows picked via `INDEX` when provided) | fp16 / bf16 / fp32 | ND |
| `b` | Input | b-projection pre-sigmoid of the full batch `[T_all, Nv]` | fp16 / bf16 / fp32 | ND |
| `INDEX` | Input | Original token id per output row `[T]`; maps each gathered `mixed_qkv` row back to its a/b row. Unused when `mixed_qkv` rows already align with a/b (pass any tensor as dummy) | int32 | ND |
| `A_log` | Input | Per-head log decay scale `[Nv]` | fp16 / bf16 / fp32 | ND |
| `dt_bias` | Input | Per-head dt bias `[Nv]` | fp16 / bf16 / fp32 | ND |
| `q_out` | Output | L2-normalized query `[1, T, Nk, Dk]`; first element of the returned tuple | same as `mixed_qkv` | ND |
| `k_out` | Output | L2-normalized key `[1, T, Nk, Dk]`; second element of the returned tuple | same as `mixed_qkv` | ND |
| `v_out` | Output | Value passthrough copy `[1, T, Nv, Dv]`; third element of the returned tuple | same as `mixed_qkv` | ND |
| `g` | Output | Log-decay gate `[1, T, Nv]`, always fp32 (consumed as fp32 by the downstream recurrent/chunk kernels); fourth element of the returned tuple | fp32 | ND |
| `beta_out` | Output | Interpolation coefficient `sigmoid(b)` `[1, T, Nv]`; fifth element of the returned tuple | same as `b` | ND |

## Constraints

- `T = mixed_qkv.shape[0]`; `Nk = num_k_heads // tp_size`, `Nv = num_v_heads // tp_size`, `Dk = head_k_dim`, `Dv = head_v_dim` are fixed by the model deployment and only enter the kernel as compile-time constants, so they never change at runtime.
- `T` and `NUM_CHUNKS` are `do_not_specialize` runtime arguments: varying token counts never trigger Triton recompilation. The remaining compile-time variants are the `BLOCK_T` tier (2 values per config, selected by `_pick_block_t`) and `HAS_INDEX` (spec vs non-spec batches); `_precompile_tiers` compiles every (tier, variant) combination on the first call, so no online recompilation occurs afterwards.
- All tensors are contiguous ND; `mixed_qkv` must follow the q \| k \| v packed layout produced by the GDN in-projection + causal conv.
- Gating is computed in fp32 regardless of input dtype; `g` is always fp32.
- Only for inference (prefill/decode/spec-decode) on NPU; not used under PCP-only or 310P paths.

## Origin and Differences

- **Origin**: Developed for vllm-ascend. Replaces the torch/Triton chain in `AscendGatedDeltaNetAttention._forward_core` (`rearrange_mixed_qkv` + two `l2norm_fwd` launches + `DeviceOperator.fused_gdn_gating`, plus four `index_select` on spec/mixed batches) with one launch; numerically identical to the replaced chain by construction.
- **Differences**:
    - NPU adaptation for performance: fuses split + L2 norm + gating into a single vector-core launch (the chain issued 4-5 kernels plus 4 gathers per layer per step); `CompiledKernel` reuse on the launch path bypasses the >100us Triton `JITFunction` dispatch (~30-40us via `ck.run`, kernel itself ~40us);
    - Modified for a specific vllm-ascend logic or different input parameters: outputs keep the batched `[1, T, H, D]` / `[1, T, Nv]` convention of the replaced chain so the downstream recurrent/chunk kernels stay shape-agnostic; `HAS_INDEX` gathers a/b rows inline instead of the legacy output-side `index_select`.

## Test Cases

The test compares `fused_gdn_prepare_impl` against the independent reference chain (`torch.split` + reshape + `l2norm_fwd` + `fused_gdn_gating_pytorch` + `index_select`) for q/k/v/g/beta, in bf16 and fp16, across token counts covering both `BLOCK_T` tiers and with/without the spec `INDEX` gather.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_gdn_prepare.py
```
