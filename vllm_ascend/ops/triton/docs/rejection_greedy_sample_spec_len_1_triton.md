# rejection_greedy_sample_spec_len_1_triton

## Description

- **Function**: Verifies one draft token per request and emits the target token plus a bonus token when accepted.
- **Formula**: Normal mode accepts iff `draft == argmax(target)`; synthetic mode accepts iff `u < conditional_rate[0]` and `draft >= 0`.
- **Algorithm flow** (processed request by request, independently): load draft, target argmax, and bonus; evaluate acceptance; write the selected first token and conditionally write the bonus.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used during speculative sampling; graph mode is N/A because sampling follows the captured model-forward graph.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `output_token_ids_ptr` | Output | Accepted/recovered tokens `[batch_size, 2]` | int32 / int64 | ND |
| `draft_token_ids_ptr` | Input | One draft token per request | int32 / int64 | ND |
| `target_argmax_ptr` | Input | Target argmax token per request | int64 | ND |
| `bonus_token_ids_ptr` | Input | Bonus token per request | int32 / int64 | ND |
| `vec_len` | Attribute | Active request count | int | scalar |
| `uniform_probs_ptr` | Input | Synthetic acceptance uniforms, or null | fp32 | ND |
| `synthetic_conditional_rates_ptr` | Input | Synthetic rate for position zero, or null | fp32 | ND |
| `SYNTHETIC_MODE`, `BLOCK_SIZE` | Attribute | Synthetic switch and compile-time request tile | bool / int | scalar |

## Constraints

- This specialization requires exactly one draft token for every request and no per-request greedy mask.
- Synthetic probabilities and rates must lie in `[0, 1]`; their pointers are required only in synthetic mode.
- Output is preinitialized with invalid IDs by the caller; the bonus slot remains unchanged on rejection.

## Origin and Differences

- **Origin**: Adapted from vLLM's rejection sampler.
- **Differences**:
    - NPU adaptation for performance: specializes the one-draft-token case to avoid cumulative-count handling and request-local loops.
    - Modified for vllm-ascend logic: supports the benchmark-only synthetic acceptance mode.

## Test Cases

The NPU test uses five requests and exercises normal and synthetic modes with a mix of accepted and rejected drafts. It compares the complete int64 output with the PyTorch reference using `torch.equal`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py::test_rejection_greedy_sample_spec_len_1_triton_kernel
```
