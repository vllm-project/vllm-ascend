# aclnnGroupedMatmulSituQuant

## Overview

This operator fuses MXFP8-by-MXFP4 grouped matrix multiplication, SiTU activation,
and dynamic MX quantization into one launch on Ascend 950.

## Inputs and outputs

| Name | Direction | Data type | Description |
| --- | --- | --- | --- |
| `x` | Input | float8_e4m3fn | Quantized activations with shape `[M, K]` |
| `xScale` | Input | float8_e8m0fnu | MX scale of `x` |
| `weight` | Input | float4_e2m1fn_x2 | Expert weights in ND, FRACTAL_NZ, or TensorList form |
| `weightScale` | Input | float8_e8m0fnu | MX scale of the expert weights |
| `groupList` | Input | int64 | Token grouping information for each expert |
| `output` | Output | float8_e4m3fn | Quantized SiTU output |
| `outputScale` | Output | float8_e8m0fnu | Dynamic MX scale of `output` |

## Constraints

- `groupListType` supports `0` (cumulative) and `1` (count).
- `linearBeta` must be positive for the fused path.
- Only MX A8W4 is supported; `bias` and `smoothScale` are unsupported.
- `K` and half of the output width must be multiples of 64.
- Ascend 950 is the only supported platform.
