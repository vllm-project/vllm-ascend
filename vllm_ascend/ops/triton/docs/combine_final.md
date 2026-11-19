# Qwen4Exp HC final-weight combine

## Description

- **Function**: Write the previous block output into each HyperConnection
  residual stream before the next AscendC TextGatedResidual mix.
- **Formula**: `output[m,c,h] = residual[m,c,h] + block[m,h] * injection[m,c]`.
  Without injection, the multiplier is one. Injection contains final weights;
  no sigmoid or channel scaling is applied by this kernel.
- **Algorithm flow**: Each program handles one token, one HC stream and one
  hidden-dimension tile. Arithmetic uses FP32 before conversion to the output
  dtype. Inputs are preserved and output is allocated separately.
- **Supported modes**: Validated on A3 in eager and NPU graph execution.
  Other devices supported by the external operator package need validation.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| residual | Input | HC residual, `[M,C*H]` | BF16 | ND, unit inner stride |
| block_output | Input | Previous block output, `[M,H]` | BF16 | ND, unit inner stride |
| injection | Input | Final injection weights, `[M,C]`, or `None` | BF16 | ND |
| channels | Attribute | HC stream count, `C=4` for the tested model | Integer | N/A |
| output | Output | Updated residual, `[M,C*H]` | BF16 | ND |

## Constraints

The Qwen3.8-Flash-Next caller uses `C=4`, `H=2560`. Empty token batches return
an empty output without launching the combine kernel. Shape or inner-stride
mismatches raise `ValueError`. Device tensors must use the same device and dtype.

`HIDDEN`, injection presence and the fixed tile size of 1024 specialize the
Triton kernel. Token count changes the grid; scalar strides follow Triton's
normal specialization rules. No tensor values are read on the host. This wrapper
does not add JIT warmup registration; first-use compilation is excluded from the
reported steady-state measurements.

## Origin and Differences

The integration is ported from PR #21794. Upstream HC produces injection logits;
the AscendC operator produces final injection weights. Calling upstream combine
on these weights would apply the sigmoid twice. This model-specific combine
preserves the new operator's semantics, including the final model/MTP boundary.
The implementation lives with the HC adapter in `models/qwen4_exp/hc_fused.py`.

The adapter preserves checkpoint parameter names, excludes padded projection
rows, retains ND projection weights and prepares nonpersistent inference buffers
after weight loading. The fused mix accepts `[1,M,C*H]` and returns the mixed
block input plus optional final injection weights. Both the main model and MTP
use the adapter.

### Runtime dependencies

Qwen4Exp HC uses this path by default and requires the matching CANN custom
operator package and Torch extension; there is no automatic fallback. Validation
used the `dcc061-20261117` A2/A3/A5 package:

- [TextGatedResidual package](https://gitcode.com/kdy18482276080/test11/blob/pkg%2Ftgr-fwd-grad-a2a3a5-dcc061-20261117/aarch64/text_gated_residual_fwd_grad_a2a3a5_linux-aarch64.run)
- Package SHA256: `f7fcf7e71cf434a4944aa39aa3d8c6b7e63bf6a093a7055ee09737de950c419c`.
- Matching Torch adapter: `kdy18482276080/ops-transformer` commit
  `3e19f0d25b472b428ffe395fb2bc42e5a96ce7f0`, providing
  `cann_ops_transformer_custom.ops.mhc.text_gated_residual.text_gated_residual`.

Install both artifacts and source the installed vendor's `bin/set_env.bash`
before starting the service. The wheel registers
`torch.ops.cann_ops_transformer.text_gated_residual`; the CANN package supplies
its ACLNN implementation. An existing image without these artifacts needs both
dependencies installed before serving Qwen4Exp.

## Test Cases

The NPU tests compare mix and pending-combine results with a Torch reference,
cover the final mixer without injection, zero final weights, empty combine,
single tokens, representative prefill shapes and invalid shapes/strides.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_hc_fused.py
```
