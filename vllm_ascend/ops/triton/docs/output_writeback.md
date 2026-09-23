# write_recurrent_output

## Description

- **Function**: Copies valid recurrent KDA output rows and zeros destination padding in one Triton launch. The Python entry point `write_recurrent_output` and private kernel `_write_output` are in `vllm_ascend/ops/triton/kda/output_writeback.py`. The GLM caller supplies the final attention buffer only for ordinary decode-only batches; mixed prefill/decode and speculative paths retain their existing output handling.
- **Formula**: Let `Ts = source.shape[1]`, `Td = destination.shape[1]`, and `V = min(query_ends[-1], Ts)`. For each destination row `0 <= t < Td`, `destination[0, t, h, d] = source[0, t, h, d]` when `t < V`, and zero otherwise. `source` and `query_ends` are unchanged. An empty source produces all zeros; an empty destination returns without launching a kernel.
- **Algorithm flow**:
    1. Flatten the contiguous output. Each program handles a 16 KiB tile: 8192 elements for FP16/BF16 or 4096 for FP32.
    2. For tiles overlapping the source allocation, load the final cumulative query end from the device and copy only valid rows. Mask the last partial destination tile.
    3. Skip source loads entirely for an empty source or a tile wholly beyond the source allocation, and write zeros instead. This avoids the Ascend invalid-address failure observed even with fully masked out-of-allocation loads.
    4. Store every destination element exactly once, including graph/DP padding. The wrapper returns `None`; its result is the mutated destination.
- **Supported modes**: Eager execution and ACL graph capture/replay on Ascend NPU. Hardware validation for this PR was on Atlas A3; Atlas A2 and 950PR&950DT validation is N/A. This operator only handles output movement; the native recurrent operator still computes attention values and updates recurrent state.

## Parameters

All three public entry-point parameters are required. Kernel launch attributes are derived by the wrapper.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `source` | Input | Native recurrent output `[1, Ts, H, D]`, including any padded source rows | fp16 / bf16 / fp32 | Contiguous ND |
| `destination` | Output | Final attention buffer `[1, Td, H, D]`; all elements are overwritten | Same as `source` | Contiguous ND |
| `query_ends` | Input | Cumulative query offsets, including the leading zero; only the final entry is read | int32 | Contiguous 1-D |
| `last_query` | Internal attribute | `query_ends.numel() - 1` | Integer, runtime scalar | Scalar |
| `source_tokens` | Internal attribute | `Ts` | Integer, runtime scalar | Scalar |
| `elements` | Internal attribute | `destination.numel()` | Integer, runtime scalar | Scalar |
| `EMPTY_SOURCE` | Internal attribute | Whether `Ts == 0` | bool, constexpr | Scalar |
| `TOKEN_WIDTH` | Internal attribute | `H * D` | Integer, constexpr | Scalar |
| `BLOCK` | Internal attribute | `16384 // destination.element_size()` | Integer, constexpr | Scalar |

## Constraints

- Source and destination must have leading dimension 1, matching positive `H` and `D`, matching dtypes, and contiguous storage on the same NPU. They use separate, non-overlapping buffers. Nonzero storage offsets are allowed; arbitrary tensor strides are not handled.
- `Ts` and `Td` may be zero, and `Td` may exceed `Ts` when another DP shard requires more padded tokens. Writes never extend past the destination's element count.
- `query_ends` resides on the same NPU and is nonempty for a nonempty source. Its final entry is a nonnegative valid-token count; the kernel clamps it to `Ts`. The production caller passes cumulative query offsets including the leading zero. No CPU read or `.item()` is used to obtain the count.
- `last_query`, `source_tokens`, and `elements` are non-specialized runtime scalars. Changing their values does not itself request a value-specialized kernel; dtype, token width, and the empty-source flag may still select different compiled variants.
- Graph replay keeps captured shapes, addresses, launch dimensions, and scalar launch arguments fixed. Source values and query-end contents can be updated in the same buffers before replay, and the final query end is reread on the device.
- The caller runs the native recurrent operator before writeback on the same stream. This kernel performs no recurrent computation, state update, type conversion, or reduction.

## Origin and Differences

- **Origin**: Developed for the GLM recurrent-output path in `vllm_ascend/models/glm5next/ops/kda.py`; it is not a port of an upstream vLLM Triton operator.
- **Differences**:
    - Fuse the previous valid-row mask, destination zeroing, and final copy for ordinary decode batches without changing native KDA arithmetic.
    - Guard entire source loads at the allocation boundary, including null-pointer empty tensors, for Ascend graph/DP padding.
    - Use dtype-sized 16 KiB tiles and runtime token-count scalars to limit local-memory use and repeated compilation. Moving the module into `ops/triton/kda` leaves the implementation unchanged.

## Test Cases

The existing nine-case operator suite compares against an independent CPU PyTorch reference with `rtol=0`, `atol=0` for FP16/BF16/FP32. It covers valid-row copies, zero padding, empty source/output, single-token decode, unaligned destination guards, and the observed DP allocation-end regression (`Ts=63`, `Td=1771`, `H=16`, `D=128`). The graph case changes source values and the final query end between replays, with NaNs in invalid rows to expose accidental reads. No additional cases are introduced by this documentation change.

Historical A3 results and performance limits are recorded in PR #17011; this document does not claim a new device run or a stable speedup for every concurrency level.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_output_writeback.py
```
