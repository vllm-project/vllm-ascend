# rejection_greedy_sample_triton

## Description

- **Function**: Performs greedy verification for variable-length draft sequences and appends a bonus token when the full sequence is accepted.
- **Formula**: Emit target argmax values through the first position where `draft_i != target_argmax_i`; if no mismatch occurs, append the bonus token. Synthetic mode substitutes Bernoulli acceptance.
- **Algorithm flow** (processed request by request, independently): derive its range from cumulative lengths, walk drafts until rejection, store target/draft results, and call `bonus_renew` after full acceptance.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used during speculative sampling; graph mode is N/A because sampling follows the captured model-forward graph.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `output_token_ids_ptr` | Output | Result `[batch_size, max_spec_len + 1]` | int32 / int64 | ND |
| `cu_num_draft_tokens_ptr` | Input | Inclusive cumulative draft counts | int32 / int64 | ND |
| `draft_token_ids_ptr`, `target_argmax_ptr` | Input | Flattened draft and target token IDs | int32 / int64 | ND |
| `bonus_token_ids_ptr` | Input | Bonus token per request | int32 / int64 | ND |
| `is_greedy_ptr` | Input | Optional per-request greedy mask | bool / int8 | ND |
| `vec_len`, `max_spec_len` | Attribute | Active requests and output draft width | int | scalar |
| `uniform_probs_ptr`, `synthetic_conditional_rates_ptr` | Input | Synthetic uniforms per token and rates per position, or null | fp32 | ND |
| `SYNTHETIC_MODE`, `BLOCK_SIZE` | Attribute | Synthetic switch and compile-time request tile | bool / int | scalar |

## Constraints

- Cumulative counts must be monotonic and end at the flattened token count; each request length is at most `max_spec_len`.
- Optional synthetic buffers are required only in synthetic mode and contain values in `[0, 1]`.
- Rows not selected by `is_greedy_ptr` are not modified by this kernel.

## Origin and Differences

- **Origin**: Adapted from vLLM's rejection sampler.
- **Differences**:
    - NPU adaptation for performance: packs variable-length requests into vector-core tiles and avoids host-side per-request loops.
    - Modified for vllm-ascend logic: supports synthetic acceptance for benchmarking.

## Test Cases

The NPU test uses six requests with draft lengths `[3,2,1,0,3,2]`, maximum speculative length 3, and both normal and synthetic acceptance. It covers first rejection, full acceptance plus bonus, and an empty draft row, requiring bit-exact equality with the PyTorch reference.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py::test_rejection_greedy_sample_triton_kernel
```
