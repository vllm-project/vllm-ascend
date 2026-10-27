# SFA DCP local merge

## Description

- **Function**: Combines received partial attention outputs using their FP32 LSE weights. It does not perform communication.
- **Formula**: For finite LSE values, `w_r = exp(lse_r - max_valid_lse) / sum_valid(exp(lse - max_valid_lse))` and `out = sum_valid(w_r * partial_r)`. Nonfinite-LSE ranks contribute zero; an all-invalid position returns zero.
- **Algorithm flow**: Load the rank-wise LSE vector; derive finite weights using a finite substitute for an all-invalid maximum; accumulate one FP32 output vector rank by rank. Mask invalid output values before weighting so `NaN * 0` cannot contaminate the result.
- **Supported modes**: NPU eager execution and ACLGraph capture/replay. The implementation uses shared Triton capability helpers rather than hardware-generation branches; Atlas A3 has execution evidence.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `out_recv` | Input | `[R, T, H, D]` or `[R, H, T, D]` | BF16, FP16 or FP32 | ND, unit innermost stride |
| `lse_recv` | Input | Corresponding `[R, T, H]` or `[R, H, T]` | FP32 | ND, unit innermost stride |
| `token_dim` | Attribute | 1 for token-major, 2 for head-major received payload | integer | scalar |
| result | Output | `[T, H, D]` | FP32 | ND, contiguous |

## Constraints

- The rank dimension is positive and a power of two, as required by the static reduction vector. Shapes and source strides must describe the exchange payload; the caller owns these layout preconditions.
- Feature tails are masked. A single program keeps at most512 FP32 feature elements plus the rank vector, rather than materializing an `R x D` working tile.
- Finite-LSE filtering, finite `safe_max`, zero-denominator handling and invalid-output masking are part of numerical semantics, not optional defensive checks.
- Runtime head counts, work counts and ordinary strides are not exact-value constexpr keys. `LSE_RANK_STRIDE` remains a measured specialization exception; previously unseen layouts can therefore compile new variants.
- This is an internal NPU kernel API, not a Torch CPU fallback implementation. The public DCP operator retains upstream routing for modes not using this payload.

## Origin and Differences

- **Origin**: The weighted reduction after DCP All2All.
- **Differences**: Keeps one FP32 accumulator and a rank-weight vector live; returns contiguous token-major output and preserves invalid-rank semantics without an intermediate invalid subtraction.

## Test Cases

The single-card matrix covers R2/4/8, both layouts, D127/496/512/513, BF16/FP16/FP32, all-invalid positions and changed-input graph replay against FP64. Distributed exchange tests additionally cover the actual packed payload and collective ordering.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sfa_dcp.py -k merge
```
