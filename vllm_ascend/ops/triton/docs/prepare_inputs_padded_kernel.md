# prepare_inputs_padded_kernel

## Description

- **Function**: Computes, entirely on the NPU, the sampled-token index and rejected-token count for every request in the padded speculative-decoding path. Rejected tokens remain in the padded input and are filtered later through the generated indices.
- **Formula**: For request `r`, let `C[r]` be the inclusive cumulative draft-token count, `v[r]` the number of valid sampled tokens, and `q[r]` the request boundary in `query_start_loc`:

  $$
  d_r = C[r] - \begin{cases}0,&r=0\\C[r-1],&r>0\end{cases},
  $$

  $$
  j_r = \begin{cases}d_r + 1 - v[r],&d_r>0\\0,&d_r=0\end{cases},
  \qquad
  i_r = q[r+1] - 1 - j_r.
  $$

  The outputs are `num_rejected_tokens[r] = j_r` and `token_indices_to_sample[r] = i_r`.
- **Algorithm flow** (processed request by request, independently):
  1. Launch `min(ceil(num_reqs / BLOCK_SIZE), num_vector_cores)` programs with the production `BLOCK_SIZE=4`.
  2. Each program grid-strides over request tiles and reconstructs the per-request draft count from the inclusive cumulative input.
  3. Compute the rejected-token count, forcing it to zero for requests with no draft tokens.
  4. Subtract the rejected count from the last token index of the request and store both int32 outputs. Request tails are masked.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 in the MRV1 padded speculative-decoding path. The kernel is also launched during Triton warmup; the current single-operator test has been verified on Atlas A2.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `cu_num_draft_tokens` | Input | Inclusive cumulative number of draft tokens for each request | int32 | Contiguous 1-D, `[num_reqs]` |
| `valid_sampled_tokens_count` | Input | Number of valid sampled tokens, including the accepted draft prefix and possible bonus token | int32 / int64 | Contiguous 1-D, `[num_reqs]` |
| `query_start_loc` | Input | Prefix-sum boundaries of the padded query tokens | int32 | Contiguous 1-D, `[num_reqs + 1]` |
| `token_indices_to_sample` | Output | Flattened padded-input index selected for each request | int32 | Contiguous 1-D, `[num_reqs]` |
| `num_rejected_tokens` | Output | Number of rejected padded tokens for each request | int32 | Contiguous 1-D, `[num_reqs]` |
| `num_reqs` | Input (attribute) | Number of requests processed by the launch | int32 | Scalar |
| `BLOCK_SIZE` | Input (attribute) | Compile-time request tile width; production uses 4 | int32 | Scalar |

## Constraints

- `num_reqs` must be positive for the production caller, and all request-indexed inputs and outputs must contain at least `num_reqs` elements.
- `cu_num_draft_tokens` must be a non-decreasing inclusive prefix sum. Each reconstructed `d_r` must be non-negative.
- `query_start_loc` must contain `num_reqs + 1` non-decreasing boundaries, and `query_start_loc[r + 1] - 1` must identify the final padded token for request `r`.
- The intended range is `0 <= valid_sampled_tokens_count[r] <= d_r + 1`. The kernel does not clamp invalid metadata.
- Inputs and outputs must be contiguous NPU tensors. The kernel performs no host readback and is designed for the warmed speculative-decoding execution path.
- The current test validates `token_indices_to_sample` but does not independently assert `num_rejected_tokens`; zero-draft requests are implemented in the kernel but are not covered by the current parameterization.

## Origin and Differences

- **Origin**: Adapted from the vLLM V1 speculative-decoding input-preparation logic and introduced as an NPU Triton kernel in vLLM-Ascend PR #5356.
- **Differences**:
    - NPU adaptation for performance: reconstructs request-local counts and both outputs in one grid-stride kernel capped at the available vector-core count, avoiding a chain of device tensor operations and host synchronization;
    - Modified for a specific vllm-ascend logic or different input parameters: consumes the inclusive `cu_num_draft_tokens` representation used by the Ascend proposer and preserves rejected tokens as padding for later filtering.

## Test Cases

The direct kernel test covers `num_reqs` values 1, 7, 32, 128, and 2048, draft lengths from 1 to 5, different valid-prefix lengths, request tails, and workloads larger than the vector-core count. Integer output comparison is exact. The existing assertion covers `token_indices_to_sample`; adding an independent assertion for `num_rejected_tokens` and a zero-draft case remains a test-coverage follow-up.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_inputs_padded.py
```
