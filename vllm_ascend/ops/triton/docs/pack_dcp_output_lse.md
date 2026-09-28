# pack_dcp_output_lse

> Source: `vllm_ascend/ops/triton/dcp/sfa_cp.py` and `sfa_cp_batched.py`.

## Description

- **Function**: Packs one rank's partial attention output and FP32 log-sum-exp (LSE) into a single contiguous payload for DCP All-to-All. SFA DCP and split MLA use this layout. The payload keeps the input dtype, while the LSE encoding carries FP32 values through BF16 and FP16 payloads.
- **Formula**: For each input row `(t, h)`, `r = t // (T / P)` and `s = t % (T / P)` when `scatter_dim=0`; `r = h // (H / P)` and `s = h % (H / P)` when `scatter_dim=1`. The output vector is stored at `send[r, s, h]` or `send[r, s, t]`, respectively. Its first `D` values equal `partial_output[t, h, :]`. FP32 payloads append the LSE directly. BF16/FP16 payloads append a signed exponent code and three base-256 digits of the FP32 significand. An exponent code of zero denotes a non-finite LSE.
- **Algorithm flow** (processed row by row, independently):
  1. Validate the NPU input shapes, dtype, scatter axis, and divisibility by `dcp_size`; allocate the contiguous payload `[P, local_scatter_size, replicated_size, D + lse_pack_dim]`.
  2. Flatten `(token, head)` into `total_rows = T * H` and launch `min(total_rows, get_vectorcore_num())` Triton programs. Each program walks its rows with a grid-stride loop.
  3. Read output and LSE through their explicit strides, calculate the destination rank and local scatter index, encode the LSE, and write one packed row. The feature tail beyond `D` is masked.
  4. On eligible A5 BF16 head-scatter shapes, process eight rows per tile in `_pack_dcp_output_lse_batched_kernel`; `row_mask` protects the last partial tile.
- **Supported modes**: Atlas A2 and A3 use the scalar-row kernel. A5 uses the same kernel outside the batched dispatch range and the eight-row kernel inside it. The wrapper participates in eager execution and the registered `sfa_dcp_a2a_fused` custom op's FakeTensor shape propagation.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `partial_output` | Input | Partial attention output `[T, H, D]`; positive non-contiguous strides are accepted | BF16 / FP16 / FP32 | ND |
| `softmax_lse` | Input | LSE `[T, H, 1]`, on the same NPU | FP32 | ND |
| `dcp_size` | Input (attribute) | Number of destination ranks, `P` | positive int | scalar |
| `scatter_dim` | Input (attribute) | `0` scatters tokens; `1` scatters heads | int | scalar |
| `send` | Output | Contiguous All-to-All payload `[P, local_scatter_size, replicated_size, D + lse_pack_dim]` | same as `partial_output` | ND |

## Constraints

- `T`, `H`, and `D` must be positive. The selected scatter dimension must be divisible by `dcp_size`. `lse_pack_dim` is one for FP32 output and four for BF16/FP16 output.
- The BF16/FP16 encoding uses integer fields exactly representable in both dtypes. The combine operator reconstructs finite LSE values; NaN and infinities are marked invalid rather than preserved as payload values.
- The A5 batched path requires BF16, `scatter_dim=1`, `D <= 2048`, `T * H >= 8 * get_vectorcore_num()`, and unit feature stride. Other valid shapes use the scalar-row kernel. `BLOCK_D = next_power_of_2(D)` and `BLOCK_ROWS = 8` are compile-time tile sizes.
- The operator is for inference on NPU. The wrapper is called before the HCCL exchange; it does not execute a collective itself.

## Origin and Differences

- **Origin**: Developed for the vllm-ascend DCP All-to-All path; it packs the partial output and LSE consumed by [`fused_dcp_lse_combine`](fused_dcp_lse_combine.md).
- **Differences**:
    - NPU adaptation for performance: the A5 path batches eight independent rows per program when enough work exists to occupy the vector cores. The scalar path covers A2/A3 and fallback shapes.
    - Modified for vllm-ascend DCP: the scatter axis determines the rank-major HCCL layout, and four exactly representable fields carry FP32 LSE inside a BF16/FP16 payload.

## Test Cases

The single-card nightly test compares packed output bit-exactly with the expected rank permutation and checks decoded LSE against FP32 input. It covers both scatter axes, non-contiguous inputs, BF16/FP16/FP32 payloads, A5 dispatch boundaries, and a partial eight-row tile. The multi-card A3 test exercises the payload through HCCL All-to-All. The output-data comparison uses `atol=0, rtol=0`; decoded LSE uses the tolerances in the test.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sfa_cp_a2a.py
pytest -sv tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_sfa_cp_a2a.py
```
