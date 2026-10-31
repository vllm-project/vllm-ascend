# bonus_renew

## Description

- **Function**: Appends a request's bonus token after all of its draft tokens have been accepted.
- **Formula**: `output[position, num_draft_tokens] = bonus_token_ids[position]`.
- **Algorithm flow** (processed request by request, independently): load one bonus token and store it at the row offset following the accepted draft prefix.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Inlined into greedy rejection sampling; graph mode is N/A because sampling follows model-forward graph execution.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `bonus_token_ids_ptr` | Input | Bonus token per request | int32 / int64 | ND |
| `position` | Attribute | Request row index | int | scalar |
| `output_token_ids_ptr` | Output | Output matrix `[batch_size, max_spec_len + 1]` | int32 / int64 | ND |
| `max_spec_len` | Attribute | Maximum speculative length | int | scalar |
| `num_tokens1` | Attribute | Accepted draft count for this request | int | scalar |

## Constraints

- `0 <= position < batch_size` and `0 <= num_tokens1 <= max_spec_len`.
- Called only when every active draft token for the request was accepted.

## Origin and Differences

- **Origin**: Helper developed as part of the vLLM Ascend Triton rejection sampler.
- **Differences**:
    - NPU adaptation for performance: inlines the scalar bonus write into the vector-core greedy kernel, avoiding a separate launch.
    - Modified for vllm-ascend logic: N/A.

## Test Cases

Covered through `rejection_greedy_sample_triton`:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py
```
