# Step5 fused residual add and RMSNorm

## Description

- **Function** — Fuse the optional residual update, RMSNorm and BF16
  conversion around attention in the Step5 decoder.
- **Formula** — For each row, compute `s = fp32(x) + residual` when the residual is
  supplied, or `s = x` otherwise, and
  `normed = bf16(s * rsqrt(mean(s * s) + eps) * scale)`, where
  `scale = weight + 1` for zero-centered weights and `weight` otherwise.
- **Algorithm flow** — Assign contiguous row ranges to vector cores. Load
  weights once per core, mask padded columns, and write the normalized output. Write an updated
  residual only when the input residual is present.
- **Supported modes** — Ascend A5 with Triton in FP32 residual mode.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Decoder activation before or after attention | BF16, FP16 or FP32 | Contiguous, last axis is hidden size |
| `residual` | Input | Optional running residual | FP32 or `None` | Same shape and layout as `x` |
| `weight` | Input | RMSNorm weights | FP32 | Contiguous vector of hidden size |
| `eps` | Attribute | RMSNorm epsilon | Python float | Scalar |
| `zero_centered` | Attribute | Add one to each norm weight | Python bool | Scalar |
| `normed` | Output | Activation for attention or the FFN | BF16 | Same shape as `x` |
| `residual_out` | Output | Updated residual or `x` if residual is absent | FP32 | Same shape as `x` |

## Constraints

- The caller supplies inputs on the same NPU device. The hidden size is positive.
- When the residual is absent, `x` must be FP32 and the wrapper returns it
  as the residual output. The norm output and any updated residual use fresh
  storage. Empty row batches return without a kernel launch.
- With FP32 residual connections enabled, the decoder checks input dtypes,
  shapes, and layouts before any sequence parallel all-gather and after any
  reduce-scatter. Other cases use the existing path.
- The custom op supplies fake output shape and dtype metadata for graph tracing.
  The A5 TP8 Step5 benchmark exercised `FULL_DECODE_ONLY` graph replay with MTP.
- Row count is a non-specialized runtime argument. The tile, weight-centering and residual
  flags, and pointer dtypes can produce different variants.

## Origin and Differences

- **Origin** — Ported from the Step5 Ascend fused residual-add/RMSNorm Triton
  implementation.
- **Differences** — This integration uses contiguous tensors and retains the
  FP32 residual/BF16 activation path, matching the decoder dispatch contract.

## Test Cases

The reference uses the original optional FP32 add, A5 RMSNorm (or the Step5 eager
formula for plain weights), and BF16 cast. The residual must match exactly.
At least 99% of BF16 elements must satisfy the mixed tolerance with
`rtol = atol = 2^-6`, and the maximum absolute error must be at most 1.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_fused_add_rms_norm.py
```
