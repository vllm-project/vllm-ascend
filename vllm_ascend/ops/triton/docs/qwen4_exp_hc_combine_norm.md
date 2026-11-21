# qwen4_exp_hc_combine_norm

Source: `vllm_ascend/ops/triton/qwen4_exp/hc.py` (public wrapper:
`hc_combine_norm`).

## Description

- **Function**: Adds one block-output row to every HyperConnection (HC)
  residual stream, applies the learned stream injection, and normalizes each
  resulting stream. It returns both the combined tensor and the normalized
  tensor. The operator is model-specific to Qwen4Exp.
- **Formula**: For row `r`, stream `s`, column `c`, and `H = hc_count`:

    ```text
    injection = 2 * sigmoid(injection_logits[r, s] / H)
    combined[r, s * group_dim + c] =
        residual[r, s * group_dim + c] + block_output[r, c] * injection
    normalized_group = combined_group * rsqrt(mean(combined_group ** 2) + eps)
    normalized_group *= 1 + norm_weight_group
    ```

    The combine output is materialized in the input dtype before normalization.
    Normalization arithmetic uses FP32.
- **Algorithm flow**:
    1. The wrapper validates shapes, inner strides, the norm-weight layout, and
       allocates both outputs.
    2. On A3, with per-stream norm weights and at least 512 rows, the wrapper
       launches `_hc_combine_norm_row_wise_kernel` with grid `(rows,)`.
    3. Each program loads its `block_output` row once, converts it to FP32, and
       reuses it while walking streams in `tl.static_range(HC_COUNT)` order.
    4. Each stream preserves the original residual load, injection, FP32 add,
       BF16 materialization, FP32 reduction, affine transform, and stores.
    5. Shared norm weights, smaller batches, other hardware profiles, and empty
       inputs retain the existing per-stream kernel or skip the launch.
- **Supported modes**: The row-wise path is enabled on Atlas A3 for the
  production per-branch norm layout. Other profiles retain the legacy kernel.
  NPU eager execution and NPUGraph capture/replay are supported. The public
  wrapper retains its PyTorch fallback for non-NPU tensors.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `residual` | Input | HC residual with shape `[rows, hc_count * group_dim]`; production width is `10240` | BF16 declared and tested | ND, unit inner stride, arbitrary row stride |
| `block_output` | Input | Block output with shape `[rows, group_dim]`; production width is `2560` | Same as `residual` | ND, unit inner stride, arbitrary row stride |
| `injection_logits` | Input or `None` | Injection logits with shape `[rows, hc_count]` | Same as `residual` | ND, unit inner stride |
| `norm_weight` | Input | Either shared `[group_dim]` weights or per-stream `[hc_count * group_dim]` weights; production uses `10240` | Same as `residual` | Contiguous 1-D ND |
| `eps` | Attribute | RMSNorm epsilon; production value is `1e-6` | Float | Scalar |
| `hc_count` | Attribute | Number of HC streams; production value is `4` | Integer | Scalar |
| First return value | Output | Newly allocated combined tensor with the residual shape | Same as `residual` | Contiguous 2-D ND |
| Second return value | Output | Newly allocated normalized tensor with the residual shape | Same as `residual` | Contiguous 2-D ND |
| `GROUP_DIM` | Internal compile-time attribute | Columns per HC stream | Integer | Scalar |
| `HC_COUNT` | Internal compile-time attribute | Streams in the static loop | Integer | Scalar |
| `HAS_INJECTION` | Internal compile-time attribute | Whether injection logits are present | Boolean | Scalar |
| `WEIGHT_SHARED` | Internal compile-time attribute | Whether all streams share one norm weight | Boolean | Scalar |
| `BLOCK_SIZE` | Internal compile-time attribute | `next_power_of_2(group_dim)`; production value is `4096` | Integer | Scalar |
| `EPS` | Internal compile-time attribute | Normalization epsilon | Float | Scalar |
| `WIDE_OFFSETS` | Internal compile-time attribute | Uses INT64 row addressing when any tensor's maximum offset exceeds INT32 | Boolean | Scalar |

## Constraints

- Inputs must be 2-D with unit inner strides. Row strides may include padding.
  Shape, divisibility, and norm-weight mismatches raise `ValueError`.
- The wrapper is out-of-place and does not mutate any input.
- The row-wise optimization requires per-stream norm weights. Shared-weight
  inputs always use the legacy kernel because the row-wise variant showed no
  benefit for that layout.
- The A3 threshold is 512 rows. At most 256 rows, both variants are launch
  overhead dominated and their timings were within normal measurement jitter;
  retaining the legacy path avoids extending the new variant to decode.
- No tensor value is read on the host. Dispatch uses only shape and hardware
  profile metadata.
- NPUGraph replay requires fixed tensor addresses, shapes, strides, and launch
  arguments. Refreshing input values before replay is supported and tested.
- Row count is not a kernel scalar argument and only changes the launch grid.
  The principal JIT keys are tensor dtypes and alignment, row strides,
  `GROUP_DIM`, `HC_COUNT`, `HAS_INJECTION`, `WEIGHT_SHARED`, `BLOCK_SIZE`,
  `EPS`, and `WIDE_OFFSETS`. The nightly compile-reuse test verifies that row
  counts `512`, `513`, `520`, and `521` reuse one compiled variant. Large row
  offsets use INT64 addressing; ordinary production shapes retain INT32.
- No dedicated warmup registration is added. The same public wrapper is used
  for compilation and runtime; normal startup, graph capture, or the first
  long-prefill request prepares the required variant.

## Origin and Differences

- **Origin**: The math and public interface mirror the upstream Qwen4Exp
  HyperConnection glue operator. The legacy per-stream Triton implementation
  was already present in this module.
- **Differences**: The A3 row-wise specialization reduces program count by
  `hc_count` and loads each block-output row once instead of once per stream.
  It does not change the combine/normalization arithmetic order or the BF16
  materialization boundary.
- **Replacement plan**: This is a permanent performance specialization. A
  temporary replacement or removal plan is N/A.
- **Production integration**: `vllm_ascend/models/qwen4_exp/model.py` installs
  this wrapper as `upstream_hc.hc_combine_norm`. The custom op is registered as
  `qwen4_exp_ascend_hc_combine_norm`. Qwen3.8-Flash-Next constructs its HC
  modules with `hc_per_branch_norm=True`, selecting the per-stream weight
  layout covered by this optimization.

## Performance

Single-A3 measurements used BF16 production shapes (`hc_count=4`,
`group_dim=2560`), injection enabled, contiguous inputs, ten iterations per
sample, and five samples. Each table entry is the median per-launch NPU-event
time after warmup. These are operator results, not end-to-end model claims.

| Rows | Norm weight layout | Legacy (ms) | Row-wise (ms) | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 512 | Per-stream | 0.081766 | 0.072326 | 1.131x |
| 2048 | Per-stream | 0.287236 | 0.257244 | 1.117x |
| 8192 | Per-stream | 1.304836 | 1.160450 | 1.124x |
| 32768 | Per-stream | 5.176840 | 4.578382 | 1.131x |
| 8192 | Shared | 1.468020 | 1.475434 | 0.995x |

The shared-weight result is the reason that layout remains on the legacy path.

## Test Cases

The nightly test compares both outputs bitwise with the legacy kernel and also
checks the PyTorch FP32 reference using the unified BF16 accuracy criteria:

```bash
pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_hc.py \
  -k combine_norm
```

Coverage includes the 512-row dispatch boundary, row tails, padded row strides,
per-stream weights with present and absent injection, forced INT64 addressing,
NPUGraph replay with refreshed inputs, and compile reuse across row counts. CPU
dispatch tests cover the shared-weight fallback and wide-offset detection for
every row-addressed tensor.

CPU dispatch tests use mocked launch entry points:

```bash
pytest -q tests/ut/ops/test_qwen4_exp_hc.py
```

## Example

```python
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.qwen4_exp.hc import hc_combine_norm

rows = 512
hc_count = 4
group_dim = 2560
residual = torch.randn((rows, hc_count * group_dim), device="npu", dtype=torch.bfloat16)
block_output = torch.randn((rows, group_dim), device="npu", dtype=torch.bfloat16)
injection = torch.randn((rows, hc_count), device="npu", dtype=torch.bfloat16)
weight = torch.randn(hc_count * group_dim, device="npu", dtype=torch.bfloat16)

combined, normalized = hc_combine_norm(residual, block_output, injection, weight, 1e-6, hc_count)
assert combined.shape == residual.shape
assert normalized.shape == residual.shape
```
