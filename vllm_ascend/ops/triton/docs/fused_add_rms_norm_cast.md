# Fused residual add, RMSNorm, and BF16 cast

## Description

- **Function** — Fuse an optional FP32 residual add, FP32 RMSNorm, and the
  output cast from FP32 to BF16 in one Triton kernel. The kernel writes the
  normalized activation directly in BF16, eliminating a separate Cast launch.
- **Motivation** — With `fp32_residual_connection` enabled, Step5 keeps the
  running residual in FP32 and casts the normalized activation to BF16 before
  attention and the feed-forward network (FFN). The fused kernel performs that
  cast while preserving the FP32 residual.
- **Formula** — For each row, compute `s = fp32(x) + residual` when the residual is
  supplied, or `s = x` otherwise, and
  `normed = bf16(s * rsqrt(mean(s * s) + eps) * scale)`, where
  `scale = weight + 1` for zero-centered weights and `weight` otherwise.
- **Algorithm flow** — Assign contiguous row ranges to vector cores, load norm
  weights once per core, and process one row per tile with padded columns masked.
  Promote the activation to FP32 on load, add the residual if present, and
  compute RMSNorm in FP32. The store to the BF16 output performs the cast.
  Write an updated FP32 residual
  only when the input residual is present.
- **Supported modes** — Step5 is adapted for Ascend 950DT (A5). Validation
  of this model-specific Triton kernel covers 950DT with FP32 residual mode.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Decoder activation before or after attention | BF16, FP16 or FP32 | Contiguous, last axis is hidden size |
| `residual` | Input | Optional running residual | FP32 or `None` | Same shape and layout as `x` |
| `weight` | Input | RMSNorm weights | FP32 | Contiguous vector of hidden size |
| `eps` | Attribute | RMSNorm epsilon | Python float | Scalar |
| `zero_centered` | Attribute | Add one to each norm weight | Python bool | Scalar |
| `normed` | Output | Normalized activation cast to BF16 for attention or the FFN | BF16 | Same shape as `x` |
| `residual_out` | Output | Updated residual or `x` if residual is absent | FP32 | Same shape as `x` |

## Constraints

- The caller supplies inputs on the same NPU device. The hidden size is positive.
- When the residual is absent, `x` must be FP32 and the wrapper returns it
  as the residual output. The norm output and any updated residual use fresh
  storage. Empty row batches return without a kernel launch.
- With FP32 residual connections enabled, the decoder checks input dtypes,
  shapes, and layouts before any sequence parallel all-gather and after any
  reduce-scatter. Other cases use the existing path.
- The Python wrapper allocates outputs and launches the Triton kernel directly.
- Row count is a non-specialized runtime argument. Tile width, weight centering,
  residual presence, and pointer dtypes can produce different variants.

## Origin and Differences

- **Origin** — Ported from the Step5 Ascend fused residual-add/RMSNorm Triton
  implementation.
- **Integration** — The decoder uses the fused BF16 output in place of its
  separate post-RMSNorm cast. The wrapper accepts contiguous inputs and returns
  the running residual in FP32.

## Test Cases

The reference performs the optional FP32 add, A5 RMSNorm (or the Step5 eager
formula for plain weights), and a separate BF16 cast. Tests require BF16 output
even for FP16 input and an exact match for the FP32 residual. At least 99% of
BF16 output elements must satisfy the mixed tolerance with
`rtol = atol = 2^-6`, and the maximum absolute error must be at most 1.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_fused_add_rms_norm.py
```

The single-card performance test compares the fused kernel with
`Cast(FP32) → torch_npu.npu_add_rms_norm → Cast(BF16)`. Both paths use the same
BF16 activations, FP32 residuals, and equivalent norm weights. The
baseline's effective scale is precomputed outside the timed region.
The test uses the Step-5-Preview hidden size of 4096 and epsilon of `1e-5`,
and reports median NPU event time per call over warmed-up graph replays.
