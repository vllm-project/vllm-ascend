# qwen4_exp_hc_combine_final

## Description

- **Function**: Write one pending Qwen4Exp decoder-block output into the HC residual, optionally scaling the block output by the final per-stream injection weights emitted by the AscendC `TextGatedResidual` operator. This model-specific helper is called by `AscendTextGatedResidual.combine` and is not a framework-wide HC replacement.
- **Formula**: For token row `m`, HC stream `c`, and hidden channel `h`:
  `output[m, c, h] = residual[m, c, h] + block_output[m, h] * injection[m, c]`.
  When `injection` is `None`, its scale is one. `injection` contains final weights; this operator does not apply sigmoid or divide by the HC count.
- **Algorithm flow** (processed row by row, independently):
  1. On NPU, the wrapper checks the residual width, block-output and injection shapes, inner strides, and common device; it allocates an output matching the residual.
  2. The grid assigns programs to token rows, HC streams, and hidden-dimension tiles of 1024 elements.
  3. Each program loads residual and block values, promotes them to FP32, optionally multiplies the block by its final stream weight, adds the residual, and stores in the residual dtype.
  4. Empty row batches return the empty output without launching a kernel. CPU callers use the Torch reference fallback; the NPU path uses the registered `PrivateUse1` custom op.
- **Supported modes**: Atlas A3 is validated by this PR's operator tests and performance runs. Atlas A2, A5, and 950PR/950DT have not been independently validated for this Qwen4Exp path.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `residual` | Input | Existing HC residual, shape `[M, C * H]`; determines output shape and dtype | BF16 in Qwen4Exp; other dtypes are backend-dependent and not validated here | 2D ND, inner stride 1 |
| `block_output` | Input | Pending decoder-block result, shape `[M, H]` | BF16 in Qwen4Exp; other dtypes are backend-dependent and not validated here | 2D ND, inner stride 1 |
| `injection` | Input | Optional final HC weights, shape `[M, C]`; may be `None` | BF16 in Qwen4Exp; other dtypes are backend-dependent and not validated here | 2D ND, inner stride 1 |
| `hc_count` | Input attribute | Number of HC streams `C`; Qwen4Exp uses 4 | Integer | Scalar |
| `output` | Output | Combined HC residual, shape `[M, C * H]` | Same dtype as `residual` | 2D ND |

## Constraints

- `residual`, `block_output`, and a non-`None` `injection` must be on the same device.
- Inputs must be rank 2, with `residual.shape == [M, C * H]`, `block_output.shape == [M, H]`, and, when present, `injection.shape == [M, C]`.
- `hc_count` must be positive and divide the residual width. `residual`, `block_output`, and optional `injection` must have unit inner stride.
- The result is out-of-place and has the residual dtype. Arithmetic is performed in FP32 before the final store.
- The model path is inference-only. The registered fake implementation supplies output metadata for graph tracing. No Triton warmup registration is used.
- `GROUP_DIM`, `HC_COUNT`, `HAS_INJECTION`, and `BLOCK_SIZE` are compile-time parameters. Qwen4Exp fixes `C=4` and `H=2560`, so these do not vary with token count. `M` controls the launch grid and does not enter a `constexpr`; `injection=None` versus a tensor creates a separate specialization. Input pointer dtypes and compiler/device settings can also affect Triton cache reuse. The operator does not use autotuning.

## Origin and Differences

- **Origin**: Developed for the Qwen4Exp AscendC `TextGatedResidual` integration. It is not copied from the upstream logits-based combine kernel.
- **Differences**:
    - The upstream/Triton HC combine consumes injection logits and applies `2 * sigmoid(logits / hc_count)`. `TextGatedResidual` returns final injection weights, so this helper multiplies by them directly to avoid transforming them twice.
    - It combines the pending block output and final weights in one Triton kernel, leaving checkpoint handling and HC mix orchestration in the model adapter.

## Test Cases

The test uses Qwen4Exp's production dimensions (`C=4`, `H=2560`, BF16) and representative token counts. It compares the integration and final-weight combine against Torch references, and covers empty rows, absent/present injection, invalid shapes and strides, and the CPU reference fallback. The combine result is checked exactly for the deterministic final-weight cases; the fused HC mix comparison uses `atol=0.02`, `rtol=0.02`.

The relocated module and model adapter passed all 18 cases on A3 with the pinned vLLM `0.33.0+empty` runtime (`4ba787ec`). The executed source files were checked against PR revision `25b9290a224711c2cf2eae49056b567aa747a02c` by SHA256.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_hc_fused.py
```
