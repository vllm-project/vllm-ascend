# pack_dcp_output_lse

> Source: `vllm_ascend/ops/triton/dcp/dcp_a2a.py` and `dcp_a2a_batched.py`.

## Description

- **Function**: Packs one rank's partial attention output and FP32 log-sum-exp (LSE) into a single contiguous payload for DCP All-to-All. The payload keeps the input dtype, while the LSE encoding carries FP32 values through BF16 and FP16 payloads.
- **Formula**: For each input row `(t, h)`, `r = t // (T / P)` and `s = t % (T / P)` when `scatter_dim=0`; `r = h // (H / P)` and `s = h % (H / P)` when `scatter_dim=1`. The output vector is stored at `send[r, s, h]` or `send[r, s, t]`, respectively. Its first `D` values equal `partial_output[t, h, :]`. FP32 payloads append the LSE directly. BF16/FP16 payloads append a signed exponent code and three base-256 digits of the FP32 significand. An exponent code of zero denotes a non-finite LSE.
- **Algorithm flow** (processed row by row, independently):
  1. Validate the NPU input shapes, dtype, scatter axis, and divisibility by `dcp_size`; allocate the contiguous payload `[P, local_scatter_size, replicated_size, D + lse_pack_dim]`.
  2. Flatten `(token, head)` into `total_rows = T * H` and launch `min(total_rows, get_vectorcore_num())` Triton programs. Each program walks its rows with a grid-stride loop.
  3. Read output and LSE through their explicit strides, calculate the destination rank and local scatter index, encode the LSE, and write one packed row. The feature tail beyond `D` is masked.
  4. On eligible BF16 head-scatter shapes, `_pack_dcp_output_lse_batched_kernel` copies a fixed two-token, four-head tile for every supported feature dimension. Features are streamed in chunks of at most 512; masks protect token, head and feature tails. Each core owns a balanced contiguous interval of tiles, advancing token/head coordinates by carry.
  5. The batched kernel uses the allocated contiguous head-scatter layout directly: the packed row offset is `(head * T + token) * (D + 4)`. Output copy, optional LSE computation and four-field encoding stay in one kernel without an intermediate concatenation or transpose. The scalar kernel covers the other public modes.
- **Supported modes**: A3 and A5 share scalar and batched kernels selected by dtype, shape and scatter axis. The wrapper participates in eager execution and the registered `dcp_a2a_fused` custom op's FakeTensor shape propagation. Head-aligned packing is covered by the shared NPU operator tests.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `partial_output` | Input | Partial attention output `[T, H, D]`; positive non-contiguous strides are accepted | BF16 / FP16 / FP32 | ND |
| `softmax_lse` | Input | LSE (or softmax maximum when `softmax_sum` is supplied) `[T, H, 1]`, on the same NPU | FP32 | ND |
| `softmax_sum` | Optional input | FP32 softmax sum with the same shape and device as the maximum; compute `maximum + log(sum)` during encoding | FP32 | ND |
| `dcp_size` | Input (attribute) | Number of destination ranks, `P` | positive int | scalar |
| `scatter_dim` | Input (attribute) | `0` scatters tokens; `1` scatters heads | int | scalar |
| `send` | Output | Contiguous All-to-All payload `[P, local_scatter_size, replicated_size, D + lse_pack_dim]` | same as `partial_output` | ND |

## Constraints

- `T`, `H`, and `D` must be positive. The selected scatter dimension must be divisible by `dcp_size`. `lse_pack_dim` is one for FP32 output and four for BF16/FP16 output.
- The BF16/FP16 encoding uses integer fields exactly representable in both dtypes. The combine operator reconstructs finite LSE values; NaN and infinities are marked invalid rather than preserved as payload values.
- The batched path requires BF16, `scatter_dim=1`, `D <= 2048` and `T * H >= 8 * get_vectorcore_num()`. It accepts explicit token, head and feature strides. Other valid shapes use the scalar-row Triton kernel. Every batched case uses `[2, 4, min(next_power_of_2(D), 512)]`; there is no D512 implementation selection or feature-stride exception. The feature dimension/tile and optional statistics calculation vary at compile time. Sizes and strides remain runtime scalars and participate in the normal Triton integer specialization keys. The existing batched owner in the compile-only DCP warmup inventory covers the unified kernel.
- The operator is for inference on NPU. The wrapper is called before the HCCL exchange; it does not execute a collective itself.

## Origin and Differences

- **Origin**: Developed for the vllm-ascend DCP All-to-All path; it packs the partial output and LSE consumed by [`fused_dcp_lse_combine`](fused_dcp_lse_combine.md).
- **Differences**:
    - NPU adaptation for performance: the batched path copies two-token, four-head tiles directly into communication order across its supported feature dimensions. The scalar Triton path covers the other public modes on both A3 and A5.
    - Modified for vllm-ascend DCP: the scatter axis determines the rank-major HCCL layout, and four exactly representable fields carry FP32 LSE inside a BF16/FP16 payload.

## Test Cases

The single-card nightly tests compare packed output bit-exactly with the expected rank permutation and check decoded LSE against FP32 input. They cover both scatter axes, non-contiguous inputs, BF16/FP16/FP32 payloads, shared dispatch boundaries, token/head tails, and feature dimensions 64, 128, 256, 384, 512, 1024 and 2048. The multi-card test exercises the payload through HCCL All-to-All. The output-data comparison uses `atol=0, rtol=0`; decoded LSE uses the tolerances in the test.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_a2a.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_rectangular_pack.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_pack_contract.py
pytest -sv tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dcp_a2a.py
```

## Candidate status

This draft replaces the D512-specific experiment with one rectangular batched
implementation across the existing supported dimensions. Paired testing found
improvements for some shapes and regressions for others. It is not accepted as a
universal performance replacement; no all-shape or model speedup is claimed.
