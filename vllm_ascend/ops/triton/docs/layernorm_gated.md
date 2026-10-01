# Gated LayerNorm / RMSNorm

## Description

- **Function**: Normalize each row and group, apply per-channel weight and optional bias, and optionally gate with `silu(z)` before or after normalization.
- **Formula**: For a group of width `N_group`, LayerNorm uses `mean = sum(x) / N_group`, `rstd = rsqrt(sum((x - mean)^2) / N_group + eps)` and `y = (x - mean) * rstd * weight + bias`. RMSNorm omits the mean and uses `rstd = rsqrt(sum(x^2) / N_group + eps)`. If `z` is present, `x *= silu(z)` before normalization when `norm_before_gate=False`; otherwise `y *= silu(z)` after the affine transform.
- **Algorithm flow** (rows and groups are independent):
  1. The wrapper validates the input layout, allocates `out`, `mean` (LayerNorm only), and `rstd`, and obtains the initialized vector-core count on NPU. For the qualified wide per-group domain, it also reads UB through the existing initialized-properties getter.
  2. A scalar selector chooses a BASE row tile or, for a qualified single group, a persistent M-axis launch. BASE uses the existing two-dimensional `(row tiles, groups)` grid with `BLOCK_M=16`, `32`, or `64`.
  3. PERSIST32 caps its one-dimensional grid at the vector-core count and walks M-axis tiles with a grid-stride loop. HOIST32 uses the same scheduling but loads the single group's weight and optional bias before that loop.
  4. Each tile computes normalization in fp32, applies the affine transform and optional gate, and stores masked outputs. Statistics are stored group-major as `[group, row]`.
- **Supported modes**: Inference-time LayerNorm and RMSNorm, with optional bias and pre/post gate. The selector is not device-name or dtype allowlisted. The optimized routes have single-operator A2/Ascend910B3 BF16 evidence; performance on other devices and dtypes is not established. Model-level graph-capture validation is outside this operator document.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Activations `[M, N]` | Floating point | 2D ND; contiguous last dimension |
| `weight` | Input | Per-channel scale `[N]` | Floating point | 1D contiguous |
| `bias` | Optional input | Per-channel bias `[N]` | Floating point | 1D contiguous, or `None` |
| `eps` | Attribute | Stability constant added before reciprocal square root | Float | Scalar |
| `z` | Optional input | Gate activations `[M, N]` | Floating point | 2D ND with contiguous last dimension, or `None` |
| `out` | Optional input/output | Caller-provided output, or allocated by the wrapper | Same as `x` | `[M, N]` with contiguous last dimension |
| `group_size` | Optional attribute | Per-group width `N_group`; defaults to `N` | Integer | Scalar dividing `N` |
| `norm_before_gate` | Attribute | Apply the gate after normalization when true, before when false | Boolean | Scalar |
| `is_rms_norm` | Attribute | Select RMSNorm rather than mean-subtracting LayerNorm | Boolean | Scalar |
| return value | Output | `(out, mean, rstd)`; `mean=None` for RMSNorm | Output dtype / fp32 statistics | `[M, N]`, `[ngroups * M]` |

## Constraints

- `N` must be divisible by `group_size`; `weight` and optional `bias` have shape `[N]`; optional `z` and `out` have shape `[M, N]`. The wrapper checks these conditions and the last-dimension strides. Resource tiling and the routes below use the **per-group** width `N_group=group_size`, not total `N` (for example, total `N=384` with `group_size=128` has three groups of width 128).
- The current policy uses BASE16 below `N_group=128`, BASE32 for `ngroups>1` at `N_group=128`, and considers PERSIST32 for a single group at `N_group=128` when `ceil(M / 32) >= P / 4`, where `P` is the initialized vector-core count. It chooses HOIST32 when `ceil(M / 32) >= 16P`. Multi-group persistent execution is not enabled.
- For `128 < N_group <= 512`, NPU calls use FT_BASE with `BLOCK_M=16` when the initialized vector-core count is available and `get_ub_size_bytes()` returns at least 196608 bytes. `BLOCK_N` is 256 for `N_group` 129–256 and 512 for 257–512. The existing getter may return its compatibility default or debugging override; its value is a routing input, not an independent compiler-resource measurement. Missing or lower UB in the scalar selector, `N_group>512`, and non-NPU calls retain BASE64. An uninitialized vector-core count continues to raise rather than falling back. The width is the per-group `N_group`, not total `N`.
- BASE retains its existing `65536 / element_size` feature-width guard. That guard does not qualify other full-tile resource buckets; PR1 does not add N-axis chunking, and `N_group>512` remains on the original BASE64 route.
- Route selection depends on shape, group count, and the initialized vector-core count, not on tensor values. No claim is made here that every route is faster on every supported device or dtype.

## Origin and Differences

- **Origin**: The existing `layernorm_gated.py` implementation is adapted from Flash Linear Attention's gated LayerNorm and the Triton LayerNorm tutorial. PR1 reuses the original BASE kernel's normalization and gating math.
- **Differences**:
    - NPU execution can use a smaller BASE row tile or a capped persistent M-axis grid instead of always launching one BASE64 program per row tile.
    - HOIST32 moves single-group weight and optional bias loads outside each program's M-tile loop. This describes source-level work placement, not an isolated measured speedup claim.
    - The selector retains BASE64 for wide `N_group`; N-chunk/C2 dispatch is not part of PR1.

## Test Cases

- Host selector and wrapper-route tests check the unchanged N128 boundaries, the FT16 `N_group`/UB envelope (including N513 and grouped inputs), non-NPU fallbacks, and launch arguments without an NPU:

  ```bash
  python -m unittest discover -s tests/ut/ops -p 'test_layernorm_gated_*.py'
  ```

- The existing single-card operator test checks LayerNorm/RMSNorm, optional bias/gate, grouping, dtype tolerances, and `out` behavior against a CPU reference:

  ```bash
  pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_layernorm_gated.py
  ```

- Three additional BF16 cases derive M from the initialized vector-core count to exercise PERSIST32, HOIST32 at its first qualifying tile, and HOIST32 at M=65536. They record the actual JIT launch through the public wrapper and compare `out`, `mean`, and `rstd` with the same CPU reference. These cases still require execution on a matching-main NPU environment before claiming in-tree NPU coverage of the new routes. The offline A2/B3 five-point measurements used the same PR1 runtime source, but are not a substitute for this in-tree test.
