# DiffusionGemma Router-Front NPU3 Profile

- Hardware: Ascend 910B3, NPU3 only.
- Workload: decode-time router front, token batch `T=1`, layer 0.
- Comparison: unfused router front versus `DgemmaFusedRouterFront`.
- Figure: `router_front_chrome_profile.svg`.

| path | captures | p50 us | p90 us | mean us |
|---|---:|---:|---:|---:|
| baseline | 8 | 802.2 | 833.1 | 843.8 |
| fused | 8 | 207.6 | 228.9 | 215.4 |

Trace speedup: x3.86

## E2E Result

| mode | n | accuracy | wall | generated tokens | throughput |
|---|---:|---:|---:|---:|---:|
| baseline reference | 20 | 95.0% | 68.3s | 5940 | 87.03 tok/s |
| qkv + router-front hot | 20 | 95.0% | 42.6s | 6184 | 145.08 tok/s |

Throughput speedup: x1.67

## Top Op Changes

| op | baseline count | fused count | baseline total us | fused total us | delta us |
|---|---:|---:|---:|---:|---:|
| `DgemmaFusedRouterFront` | 0 | 8 | 0.0 | 171.8 | -171.8 |
| `MatMulV2` | 8 | 0 | 52.6 | 0.0 | 52.6 |
| `Mul` | 32 | 0 | 43.1 | 0.0 | 43.1 |
| `Cast` | 32 | 0 | 38.5 | 0.0 | 38.5 |
| `ReduceMean` | 8 | 0 | 36.6 | 0.0 | 36.6 |
| `MoeGatingTopK` | 8 | 0 | 33.5 | 0.0 | 33.5 |
| `DgemmaApplyRouterScale` | 8 | 0 | 14.9 | 0.0 | 14.9 |
| `Add` | 8 | 0 | 11.0 | 0.0 | 11.0 |
| `Rsqrt` | 8 | 0 | 9.7 | 0.0 | 9.7 |
