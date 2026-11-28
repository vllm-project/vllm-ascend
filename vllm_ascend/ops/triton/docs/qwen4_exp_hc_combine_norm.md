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
    2. The wrapper launches `_hc_combine_norm_kernel` with
       `STREAMS_PER_PROGRAM=1` for the legacy schedule. On A3, with per-stream
       norm weights and at least 512 rows, it uses
       `STREAMS_PER_PROGRAM=HC_COUNT`.
    3. Each program loads its `block_output` row once, converts it to FP32, and
       reuses it across `STREAMS_PER_PROGRAM` streams in `tl.static_range`
       order.
    4. Each stream preserves the original residual load, injection, FP32 add,
       BF16 materialization, FP32 reduction, affine transform, and stores.
    5. Shared norm weights, smaller batches, other hardware profiles, and empty
       inputs retain `STREAMS_PER_PROGRAM=1` or skip the launch.
- **Supported modes**: The row-wise specialization is enabled on Atlas A3 for
  the production per-branch norm layout. Other profiles retain
  `STREAMS_PER_PROGRAM=1`. NPU eager execution and NPUGraph capture/replay are
  supported. The public wrapper retains its PyTorch fallback for non-NPU
  tensors.

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
| `STREAMS_PER_PROGRAM` | Internal compile-time attribute | Number of HC streams handled by one program; `1` for legacy and `hc_count` for A3 row-wise | Integer | Scalar |
| `WIDE_OFFSETS` | Internal compile-time attribute | Uses INT64 row addressing when any tensor's maximum offset exceeds INT32 | Boolean | Scalar |

## Constraints

- Inputs must be 2-D with unit inner strides. Row strides may include padding.
  Shape, divisibility, and norm-weight mismatches raise `ValueError`.
- The wrapper is out-of-place and does not mutate any input.
- The row-wise optimization requires per-stream norm weights. Shared-weight
  inputs retain `STREAMS_PER_PROGRAM=1` because the row-wise specialization
  showed no benefit for that layout.
- The operator owns the 512-row threshold; A3 eligibility is exposed through
  `QWEN4EXP_HC_COMBINE_NORM_ROW_WISE`, without an operator-specific threshold in
  `HardwareProfile`. At most 256 rows, both variants are launch-overhead
  dominated and their timings were within normal measurement jitter; retaining
  the legacy schedule avoids extending the row-wise variant to decode.
- No tensor value is read on the host. Dispatch uses only shape and hardware
  profile metadata.
- NPUGraph replay requires fixed tensor addresses, shapes, strides, and launch
  arguments. Refreshing input values before replay is supported and tested.
- Row count is not a kernel scalar argument and only changes the launch grid.
  The principal JIT keys are tensor dtypes and alignment, row strides,
  `GROUP_DIM`, `HC_COUNT`, `HAS_INJECTION`, `WEIGHT_SHARED`, `BLOCK_SIZE`,
  `EPS`, `STREAMS_PER_PROGRAM`, and `WIDE_OFFSETS`. The nightly compile-reuse
  test verifies that row counts `512`, `513`, `520`, and `521` reuse one
  compiled variant for each specialization. Large row offsets use INT64
  addressing; ordinary production shapes retain INT32.
- No dedicated warmup registration is added. The same public wrapper is used
  for compilation and runtime; normal startup, graph capture, or the first
  long-prefill request prepares the required variant.

## Origin and Differences

- **Origin**: The math and public interface mirror the upstream Qwen4Exp
  HyperConnection glue operator. The legacy per-stream Triton implementation
  was already present in this module.
- **Differences**: One `_hc_combine_norm_kernel` definition serves both
  schedules. `STREAMS_PER_PROGRAM=1` preserves the historical per-stream grid;
  the A3 row-wise specialization sets it to `hc_count`, reducing program count
  and reusing one block-output load across all streams. Both specializations
  call the same single-stream JIT helper, so injection, combine, BF16
  materialization, RMSNorm reduction, affine transform, and stores have a
  single implementation. This does not change the arithmetic order.
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
sample, five samples in alternating order, and three warmup iterations. Each
table entry is the median per-launch NPU-event time. These are operator results,
not end-to-end model claims.

| Rows | Norm weight layout | 1 stream/program (ms) | All streams/program (ms) | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 512 | Per-stream | 0.076392 | 0.070476 | 1.084x |
| 2048 | Per-stream | 0.287300 | 0.255420 | 1.125x |
| 8192 | Per-stream | 1.295568 | 1.151484 | 1.125x |
| 32768 | Per-stream | 5.155218 | 4.567454 | 1.129x |
| 8192 | Shared | 1.341490 | 1.309124 | 1.025x |

The shared-weight layout retains `STREAMS_PER_PROGRAM=1` to avoid expanding the
A3-specific dispatch for a marginal operator-only gain.

### End-to-end validation

Official `vllm bench serve` measurements used Qwen3.8-Flash-Next W4A8 on A3
with TP2 and expert parallelism, prefix caching disabled, 4,096 input tokens,
16 output tokens, concurrency 8, and 100 measured requests per arm after 16
warmup requests. Baseline and candidate source trees differed only in the HC
operator and its hardware-capability registration. Each round used ABBA order.

| Round | Baseline (req/s) | Candidate (req/s) | Throughput delta | Baseline TTFT (ms) | Candidate TTFT (ms) | TTFT delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.087983 | 1.097601 | +0.884% | 1974.804 | 1960.173 | -0.741% |
| 2 | 1.101764 | 1.106538 | +0.433% | 1969.151 | 1952.626 | -0.839% |
| 3 | 1.096858 | 1.105837 | +0.819% | 1968.449 | 1953.257 | -0.772% |

Mean request throughput increased by 0.712% across the three rounds. A separate
three-round 1,024/16 check did not show stable improvement (+1.61%, -2.43%, and
-0.61%), so the end-to-end claim is limited to the measured 4,096-token prefill
scenario.

## Test Cases

The nightly test compares both outputs bitwise against the
`STREAMS_PER_PROGRAM=1` specialization and also checks the PyTorch FP32
reference using the unified BF16 accuracy criteria:

```bash
pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_hc.py \
  -k combine_norm
```

Coverage includes the 512-row dispatch boundary, row tails, padded row strides,
per-stream weights with present and absent injection, forced INT64 addressing,
NPUGraph replay with refreshed inputs, and compile reuse across row counts. A
hardware-independent public-wrapper test exercises per-branch weights and checks
both returned tensors against the PyTorch reference; on A2 and other profiles
without row-wise capability it covers the legacy schedule, while A3 covers the
row-wise schedule. CPU dispatch tests cover both `STREAMS_PER_PROGRAM`
selections, the shared-weight fallback, and wide-offset detection for every
row-addressed tensor.

CPU dispatch tests use mocked launch entry points:

```bash
pytest -q tests/ut/ops/test_qwen4_exp_hc.py
```

EvalScope 1.15.0 GSM8K validation used 100 samples, batch size 8, temperature
zero, and at most 4,096 output tokens. Baseline and candidate both scored
98/100 and produced identical extracted answers for all 100 samples.

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
