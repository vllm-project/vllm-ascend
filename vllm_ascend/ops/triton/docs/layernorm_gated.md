# Gated LayerNorm / RMSNorm

## Description

- **Function**: Normalize each row and group, apply per-channel weight and optional bias, and optionally gate with `silu(z)` before or after normalization. Entry: `layer_norm_fwd_npu` in `vllm_ascend/ops/triton/layernorm_gated.py`.
- **Formula** (per group, computed in fp32):
    - LayerNorm: `mean = sum(x) / N_group`, `rstd = rsqrt(sum((x - mean)^2) / N_group + eps)`, `y = (x - mean) * rstd * weight + bias`.
    - RMSNorm: `rstd = rsqrt(sum(x^2) / N_group + eps)`, `y = x * rstd * weight + bias`; bias is omitted when absent.
    - With `z`, apply `x *= silu(z)` before normalization when `norm_before_gate=False`, or `y *= silu(z)` after the affine transform when true.
- **Algorithm flow** (rows and groups are independent):
  1. Validate the two-dimensional input layout, allocate output/statistics, and select a BASE row tile or HOIST32 execution.
  2. BASE processes BM16 or BM64 row tiles. HOIST32 uses BM32 tiles in a grid-stride loop with the grid capped at the vector-core count; weight and optional bias are loaded before the loop.
  3. Normalize and apply the affine transform and optional gate in fp32, then store masked outputs and group-major statistics.
- **Supported modes**: Ascend NPU inference with LayerNorm or RMSNorm, optional bias, and pre/post gate. Device validation recorded for this change is limited to Ascend910B4.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Activations `[M,N]` | fp16 / bf16 / fp32 | 2D ND |
| `weight` | Input | Per-channel scale `[N]` | fp16 / bf16 / fp32 | 1D ND |
| `bias` | Optional input | Per-channel bias `[N]`, or `None` | fp16 / bf16 / fp32 | 1D ND |
| `eps` | Attribute | Stability constant added before reciprocal square root | float | scalar |
| `z` | Optional input | Gate activations `[M,N]`, or `None` | fp16 / bf16 / fp32 | 2D ND |
| `out` | Optional input/output | Caller-provided output, otherwise allocated by the wrapper | same as `x` | 2D ND |
| `group_size` | Optional attribute | Per-group width `N_group`; `None` defaults to `N` | int | scalar |
| `norm_before_gate` | Attribute | Gate after normalization when true, before when false; default true | bool | scalar |
| `is_rms_norm` | Attribute | RMSNorm when true, LayerNorm when false; default false | bool | scalar |
| return value | Output | `(out, mean, rstd)`; `mean=None` for RMSNorm | output dtype / fp32 statistics | `[M,N]`, `[G*M]` |

## Constraints

- `M` and `N_group` must be positive, and `N` divisible by `group_size`. `weight` and optional `bias` have shape `[N]`; optional `z` and `out` match `x`. All require contiguous last dimensions.
- `G=N/N_group` counts normalization groups within each row. For GDN output normalization, `[T,H_local,D_v]` is flattened to `[M,N]` with `M=T*H_local` and `N=D_v`: each row is one token/value-head pair, and model heads do not become normalization groups. `group_size=None` gives `G=1`.
- Statistics use group-major indexing `group*M+row`. Caller-provided `out` is reused.
- Multi-group inputs retain BASE64. Single-group NPU inputs below width 128 use BASE16; width 128 uses HOIST32 when `4*ceil(M/32)>=P`, otherwise BASE16, where `P` is the initialized vector-core count. Widths 129–512 use BASE16 when the UB getter reports at least 192 KiB; other inputs retain BASE64.
- Single-group NPU calls require device properties initialized through the existing `init_device_properties_triton()` contract. The UB getter may use its existing compatibility default or debugging override. Grouped calls do not query these properties.
- BASE retains the `65536 / element_size` feature-width guard. Passing that guard or falling back to BASE64 does not guarantee that the full tile fits the compiler's UB budget. HOIST32 is selected only for single-group width 128.
- Forward inference only. Graph replay and full-model qualification are not established by the recorded single-operator tests.

## Origin and Differences

- **Origin**: The existing vLLM-Ascend gated normalization implementation is adapted from Flash Linear Attention's gated LayerNorm and the Triton LayerNorm tutorial; see the source-file header.
- **Differences**:
    - NPU adaptation for performance: selects smaller BASE row tiles for eligible single-group inputs and uses persistent execution with parameter reuse for suitable N128 workloads.
    - Public API, normalization/gating semantics, output aliasing, and statistics layout are unchanged. Grouped inputs keep the original BASE64 launch.

## Test Cases

The single-card accuracy test compares outputs and statistics against an fp32 CPU reference, covering LayerNorm/RMSNorm, optional bias/gate, grouping, caller-provided `out`, dispatch boundaries, and a large N128 shape. Dtype-specific tolerances are `rtol=2e-3, atol=2e-2` for fp16; `rtol=2e-2, atol=5e-2` for bf16; and `rtol=atol=1e-4` for fp32. Matching-main in-tree NPU execution remains pending.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_layernorm_gated.py
```
