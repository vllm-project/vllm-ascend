# prepare_qsa_visible_lengths

## Description

- **Function**: Compute the causal KV length visible to each live query row. The Qwen4Exp worker prepares it once per step so Union Prefill layers can share it.
- **Formula**: For row `i`, normalize a negative request index by adding the number of requests once. For a valid index `r`, write `min(max(position[i] + 1, 0), seq_lens[r])`. Otherwise write zero.
- **Algorithm flow**: One Triton program loads one row's position and request, gathers the sequence length with a validity mask, then writes one int32 value.
- **Supported modes**: Qwen4Exp GBSA Union metadata preparation on Ascend. The wrapper accepts an existing device buffer.

## Parameters

| Parameter | Direction | Description | Type | Format |
| --- | --- | --- | --- | --- |
| logical_positions | Input | Zero-based logical position per row | int32 or int64 | 1D, strided |
| seq_lens | Input | Nonnegative current KV length per request | int32 | 1D, strided |
| token_to_req | Input | Row-to-request mapping | int32 or int64 | 1D, strided |
| visible_lengths | Output | Caller-owned buffer | int32 | 1D, contiguous |
| rows | Attribute | A Python integer specifying how many query rows to process. | Host-side Python `int` | Scalar |

## Constraints

All tensors reside on the same NPU. Positions and their increments must fit in int32.
`rows` must be nonnegative and fit in both row inputs and the output. Only the first
`rows` output elements are written. Zero rows skip the launch. The worker reuses a
device buffer across layers. The wrapper reads shapes and strides on the host and
does not copy tensor values back to the host.

## Origin and Differences

Developed for the vLLM-Ascend Qwen4Exp Union Prefill path. It combines indexed
length lookup and causal clipping in a single kernel and writes into preallocated
storage. Upstream vLLM operator equivalent: N/A.

## Test Cases

Run `pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_qsa_visible_lengths.py`.
The tests require exact integer equality for a 4096-row chunk at 128K KV, a partial
chunk, multiple requests, strided inputs, invalid request IDs and empty input.

## Example

For positions `[2, 9, 6]`, request IDs `[0, 1, -1]` and sequence lengths
`[4, 7]`, the three visible lengths are `[3, 7, 7]`.
