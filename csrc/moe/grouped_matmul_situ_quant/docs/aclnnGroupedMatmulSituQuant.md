# aclnnGroupedMatmulSituQuant

## Overview

This operator fuses grouped matrix multiplication, SiTU activation and dynamic
quantization into one launch. Two data paths share the same prototype and are
selected by the data type of `x`:

| Path | Platform | Matrix product | `output` | `outputScale` |
| --- | --- | --- | --- | --- |
| MX A8W4 | Ascend 950 (arch35) | MXFP8 x MXFP4 | float8_e4m3fn | float8_e8m0fnu |
| A8W4 MSD | Atlas A2 / A3 (arch32) | INT8 x INT4 | int8 | float (per-token) |

## Inputs and outputs

| Name | Direction | Data type | Description |
| --- | --- | --- | --- |
| `x` | Input | float8_e4m3fn / int8 | Quantized activations with shape `[M, K]` |
| `xScale` | Input | float8_e8m0fnu / float32 | MX scale of `x` on Ascend 950; per-token scale on A2/A3 |
| `weight` | Input | float4_e2m1fn_x2 / int4 | Expert weights in ND, FRACTAL_NZ, or TensorList form. A2/A3 also accepts the int32 packed form (1 int32 carries 8 int4) through `aclnnGroupedMatmulSituQuantV2` |
| `weightScale` | Input | float8_e8m0fnu / uint64 | MX scale of the expert weights on Ascend 950; packed MSD scale on A2/A3 (torch side passes int64) |
| `weightAssistMatrix` | Input (optional) | float32 | Offline compensation matrix used by A2/A3; when absent the kernel derives it on chip. Unused on Ascend 950 |
| `groupList` | Input | int64 | Token grouping information for each expert |
| `bias` | Input (optional) | float32 | Unsupported, must be empty |
| `smoothScale` | Input (optional) | float32 | Unsupported, must be empty |
| `output` | Output | float8_e4m3fn / int8 | Quantized SiTU output |
| `outputScale` | Output | float8_e8m0fnu / float32 | Dynamic scale of `output` |

## Constraints

- `groupListType` supports `0` (cumulative) and `1` (count).
- `beta` must be positive. `linearBeta` must be non-zero on Ascend 950; on A2/A3
  a value `<= 0` disables the up-branch transform.
- `dequantMode` / `quantMode` are fixed per path: `1` / `1` for MX A8W4 on
  Ascend 950, `0` / `0` for A8W4 MSD on A2/A3. `dequantDtype` is always `0`.
- `bias` and `smoothScale` are unsupported.
- `K` must be a multiple of 64 and the output width (`N / 2`) a multiple of 64.
