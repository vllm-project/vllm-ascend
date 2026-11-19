# DeepSeek V4.1 mHC post, collapse, and RMSNorm

## Description

- **Function**: Prepare the next Attention or MoE input in the DeepSeek V4.1 model.
- **Formula**: If $z$ is the previous sublayer output and $R_i$ are the four residual streams, the updated stream is
  $R'_j = \mathrm{BF16}(z\,\mathrm{post}_j + \sum_i R_i\,\mathrm{comb}_{i,j})$.
  The next input is $\mathrm{RMSNorm}(\mathrm{BF16}(\sum_j R'_j\,\mathrm{pre\_mix}_j))$.
- **Algorithm flow**: Each row writes the updated BF16 residual, collapses its four streams with the incoming
  coefficients, and applies RMSNorm. For MoE, the kernel also widens the rounded BF16 result to FP32 for routing.
  The separate `HcPreCoefficients` operator reads the updated residual and computes new coefficients.
- **Supported modes**: Atlas A2 with Model Runner V2 and `FULL_DECODE_ONLY`. During full graph capture, the model can
  record coefficient computation on an auxiliary stream. Outside that capture, eligible boundaries launch it on
  the main stream.
- **Production callers**: DeepSeek V4.1's `_prepare_attention` and `_prepare_moe` in `models/deepseek_v41/mhc.py`
  call the model-specific `mhc_post_collapse_norm` Python wrapper.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `sublayer_out` | Input | Previous Attention or MoE result | BF16 | `[rows, 5120]` |
| `residual` | Input | Incoming residual streams | BF16 | `[rows, 4, 5120]` |
| `post`, `pre_mix` | Input | Writeback and collapse weights | FP32 | `[rows, 4]` |
| `comb` | Input | Residual mixing weights | FP32 | `[rows, 4, 4]` |
| `weight` | Input | Loaded RMSNorm weight | BF16 in the supported model configuration | `[5120]` |
| `updated` | Output | Updated residual streams | BF16 | `[rows, 4, 5120]` |
| `normalized` | Output | Next sublayer input | BF16 | `[rows, 5120]` |
| `routing_input` | Output | MoE routing input when `write_fp32` is true, otherwise aliases `normalized` | FP32 for MoE, BF16 for Attention | `[rows, 5120]` |
| `epsilon`, `write_fp32` | Attribute | RMSNorm epsilon and routing output selector | float, bool | Scalar |

## Constraints

Inputs are on the same NPU. The wrapper makes `sublayer_out` contiguous if necessary; all other inputs must
already be contiguous. The wrapper preserves input contents and allocates the outputs. Attention uses the
BF16 `normalized` output as the unused routing slot, with no extra allocation or write.

The model selects fusion when Triton is available and there are 1–72 rows per rank after graph padding.
It requires BF16 activations, four residual streams, width 5120, bias-free RMSNorm, pipeline parallel
size one, and sequence parallelism and ubatching disabled. Unsupported devices, configurations, weight
layouts or row counts keep the existing model path. Cross-layer fusion stops before an Engram layer
and at the end of each execution segment. The model schedules coefficients on a cached stream per device
during capture and waits on the main stream at the next coefficient consumer.

The weight post-processing hook registers warmup keys by loaded RMSNorm weight dtype and `write_fp32`.
One row is sufficient for compile-only warmup because row count changes only the launch grid.
`kernel_config.enable_jit_warmup` controls precompilation; fusion remains enabled when it is disabled.

## Origin and Differences

The split and overlap design follows [vLLM #59802](https://github.com/vllm-project/vllm/pull/59802). The Ascend
implementation fuses residual writeback, collapse, and normalization in Triton, retaining the BF16 rounding at
the native operator boundaries. Coefficient computation uses the Ascend C projection and coefficient routines.

## Test Cases

The A2 regression compares both boundaries at 72 rows and width 5120 with native
`HcPost → HcPreV3 → RMSNorm` (`RmsNormCast` for MoE). It changes the residual and sublayer inputs in place across
three replays of one full NPU graph. For the residual, normalized input, and each coefficient tensor, it requires
at least 99% of elements within the guide's mixed tolerance and checks the maximum absolute error.

| Output | Relative tolerance | Absolute tolerance | Maximum absolute error |
| --- | ---: | ---: | ---: |
| BF16 residual and normalized input | $2^{-6}$ | $2^{-6}$ | 1 |
| FP32 coefficients | $2^{-10}$ | $2^{-16}$ | 0.01 |

```bash
pytest -q -o junit_family=xunit1 --junitxml=mhc-accuracy.xml tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mhc_post_collapse_norm.py
```
