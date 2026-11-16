# qwen4_exp_hc_gate_mix

Source: `vllm_ascend/ops/triton/qwen4_exp/hc.py` (public wrapper:
`hc_gate_mix`).

## Description

- **Function**: Applies the HyperConnection (HC) gate to each token row and
  reduces the configured HC streams into one output row. The operator is
  model-specific to Qwen4Exp.
- **Formula**: For row `r`, output column `c`, and `H = hc_count`:

    ```text
    output[r, c] =
        sum(sigmoid(gate[r, s * group_dim + c]) *
            x[r, s * group_dim + c]) / H
        for s in [0, H)
    ```

    The multiplication and sum are evaluated in FP32 in stream order, then
    converted to the input dtype for the output store.

- **Algorithm flow**:
    1. The Python wrapper validates the shape, divisibility, and inner-stride
       contract and allocates `[rows, group_dim]` output storage.
    2. For `rows >= 1024` on an enabled hardware profile, the wrapper launches
       `_hc_gate_mix_row_block_kernel` with grid
       `(cdiv(rows, 8), cdiv(group_dim, 512))`.
    3. Each row-block program owns an `8 x 512` tile. It initializes one FP32
       accumulator and walks the HC streams with `tl.static_range`, preserving
       the original load, sigmoid, multiply, and accumulate order.
    4. Exact row and column tiles omit load/store masks. Tail tiles use one
       hoisted 2-D mask for all stream loads and the final store.
    5. If any input or output element offset can exceed `INT32_MAX`, row
       indices are widened to int64 before multiplying by the row stride.
       Normal prefill sizes retain the int32 path.
    6. Smaller batches, unsupported hardware profiles, and empty inputs use the
       existing one-row kernel or skip the launch respectively.
- **Supported modes**: The row-block path is enabled for Atlas A2, Atlas A3,
  and A5 profiles. Validation covered `Ascend910B4`, `Ascend910_9382`, and
  `Ascend950PR_9579`. A5 gating is family-level, but `Ascend950DT` was not
  separately profiled. NPU eager execution and NPUGraph capture/replay are
  supported. The public wrapper keeps its existing PyTorch fallback for
  non-NPU tensors.

## Parameters

> [!NOTE]
> The public Python interface has three required arguments. Tensor strides and
> kernel tile parameters are internal attributes prepared by the wrapper.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Hidden states with shape `[rows, hc_count * group_dim]`; the validated model shape is `[rows, 10240]` | BF16 is declared and tested; output preserves the input dtype | ND, unit inner stride, arbitrary row stride |
| `gate` | Input | Gate values with the same shape as `x`; its row stride is supplied independently | Same as `x` | ND, unit inner stride, padded row stride |
| `hc_count` | Attribute | Number of HC streams; compile-time constant `H` in the kernel. The validated model value is `4` | Integer | Scalar |
| Return value | Output | Newly allocated tensor with shape `[rows, group_dim]`; the validated model shape is `[rows, 2560]` | Same as `x` | Contiguous 2-D ND |
| `rows` | Internal runtime attribute | Number of flattened token rows. It controls the launch grid and tail masks and is marked `do_not_specialize` | Integer | Scalar |
| `stride_x_row`, `stride_gate_row`, `stride_output_row` | Internal runtime attributes | Element strides between adjacent input and output rows | Integer | Scalar |
| `GROUP_DIM` | Internal compile-time attribute | Number of columns per HC stream; validated model value is `2560` | Integer | Scalar |
| `HC_COUNT` | Internal compile-time attribute | Number of HC streams processed by the static loop | Integer | Scalar |
| `ROWS_PER_BLOCK` | Internal compile-time attribute | Rows owned by one program; fixed to `8` | Integer | Scalar |
| `BLOCK_SIZE` | Internal compile-time attribute | Columns owned by one program tile; fixed to `512` | Integer | Scalar |
| `EVEN_ROWS`, `EVEN_BLOCK` | Internal compile-time attributes | Select exact-tile loads and stores when the corresponding shape is an exact multiple | Boolean | Scalar |
| `WIDE_OFFSETS` | Internal compile-time attribute | Widen row indices to int64 when any maximum element offset exceeds `INT32_MAX` | Boolean | Scalar |

## Constraints

- `x` and `gate` must be matching 2-D tensors on the same NPU. The wrapper
  raises `ValueError` for a shape mismatch.
- The hidden dimension must be divisible by `hc_count`, and both inputs must
  have unit inner stride (`stride(1) == 1`). The row stride may include padding;
  the row-strided tail case is covered on real NPU.
- The public wrapper is out-of-place. It returns newly allocated storage and
  does not mutate `x` or `gate`.
- The validated input dtype is BF16, matching the Qwen4Exp model path. The
  tested shape is `hc_count=4`, `group_dim=2560`, and hidden width `10240`.
- The row-block path requires `rows >= 1024` and
  `QWEN4EXP_HC_GATE_MIX_ROW_BLOCK`. Other rows and profiles retain the one-row
  kernel. The 1024-row cutoff is conservative for the measured A2 and A5
  crossover points.
- For a contiguous width-10,240 row, the first row whose element offset can
  exceed `INT32_MAX` is 209,715. The host selects
  `WIDE_OFFSETS=True` for that range. The int64 branch is compiled and checked
  against the legacy kernel on A2, A3, and A5 without allocating a multi-token
  tensor above `INT32_MAX` elements.
- No device tensor value is read on the host. The host uses only shape, dtype,
  and stride metadata.
- NPUGraph capture and replay keep tensor addresses, shape, strides, and launch
  arguments fixed. Refreshing input values before replay is supported and is
  covered by the nightly test.
- `rows`, `GROUP_DIM`, `HC_COUNT`, `ROWS_PER_BLOCK`, `BLOCK_SIZE`,
  `EVEN_ROWS`, `EVEN_BLOCK`, and `WIDE_OFFSETS` are the main JIT keys.
  `rows` is marked `do_not_specialize` and the nightly compile-reuse test
  records compile events to verify that exact-row and row-tail classes each
  reuse one variant as the row count changes. A forced wide-offset case covers
  that separate constexpr branch.
- No dedicated warmup registration is added. This existing model-specific path
  uses the same wrapper at compilation and runtime; normal startup or graph
  capture prepares the bounded launch variants used by serving.

## Origin and Differences

- **Origin**: The math and public interface mirror the upstream Qwen4Exp
  HyperConnection glue operator. The existing one-row Triton implementation
  was already in this module; this change adds the Ascend row-block
  specialization.
- **Differences**:
    - NPU adaptation for performance: processes eight rows per Triton program
      for large prefill batches, reducing the program count by up to 8x and
      amortizing scheduling overhead without changing FP32 accumulation order.
    - Modified for specific vLLM-Ascend logic: hardware capability gating
      enables the row-block path on A2, A3, and A5, while preserving the
      one-row kernel elsewhere.
    - Added an int64 address branch for tensors whose maximum row-major offset
      exceeds int32 range.
    - CUDA PDL hooks from the upstream implementation are intentionally omitted
      because triton-ascend launches on an NPU stream.
- **Replacement plan**: This is a permanent performance specialization of the
  existing operator. A temporary replacement or removal plan is N/A.
- **Production integration**: `vllm_ascend/models/qwen4_exp/model.py` replaces
  `upstream_hc.hc_gate_mix` with this operator before the upstream Qwen4Exp
  model classes are imported. The custom op is registered as
  `qwen4_exp_ascend_hc_gate_mix`.

## Test Cases

The single-operator NPU suite is located under the nightly single-card Triton
directory:

```bash
pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_hc.py
```

The suite uses the real Qwen4Exp operator shape, `hc_count=4`,
`group_dim=2560`, and BF16 inputs. It covers:

- rows `1023`, `1024`, and `1025`, including the dispatch boundary;
- a non-contiguous, row-strided tail input;
- bitwise output parity between the row-block and legacy kernels with
  `rtol=0, atol=0`;
- the unified BF16 reference tolerance (`rtol=2^-6`, `atol=2^-6`), requiring
  at least 99 percent of elements to pass and maximum absolute error no greater
  than `1`;
- NPUGraph replay after refreshing both input buffers;
- a forced `WIDE_OFFSETS=True` launch checked bitwise against the legacy
  kernel;
- compile reuse across exact-row and row-tail batch counts. Within each class,
  the test changes row counts that exercise both divisible-by-16 and non-16
  specialization classes and verifies that no additional variant compiles.

CPU dispatch and hardware-capability tests remain in the unit-test suite with
mocked Triton launch entry points:

```bash
pytest -q \
  tests/ut/ops/test_qwen4_exp_hc.py \
  tests/ut/device/test_hardware_profile.py
```

These tests cover the `1023/1024/1025` threshold, capability-present and
capability-absent dispatch, A2/A3/A5 capability assignment, and the
`WIDE_OFFSETS` launch argument.

## Example

```python
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.qwen4_exp.hc import hc_gate_mix

rows = 1024
hc_count = 4
group_dim = 2560
x = torch.randn((rows, hc_count * group_dim), device="npu", dtype=torch.bfloat16)
gate = torch.randn_like(x)

output = hc_gate_mix(x, gate, hc_count)
assert output.shape == (rows, group_dim)
assert output.dtype == x.dtype
```
