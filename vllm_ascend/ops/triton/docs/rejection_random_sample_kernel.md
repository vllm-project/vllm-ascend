# rejection_random_sample_kernel

## Description

- **Function**: Sequentially verifies stochastic draft tokens per request, emits the precomputed recovery token at the first rejection, and appends a bonus after full acceptance.
- **Formula**: Draft `x_i` is accepted when `p_draft(x_i) > 0` and `u_i <= p_target(x_i)/p_draft(x_i)`. With entropy verification, the effective uniform value is `u_i * min(exp(-alpha * entropy_i), posterior_threshold)`. Synthetic mode instead accepts when `u_i < conditional_rate[i]`.
- **Algorithm flow** (processed request by request, independently): locate the flattened draft range, skip greedy requests, test tokens until rejection using full or reduced vocabulary probabilities, then write a recovered or bonus token.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Supports dense and reduced-vocabulary speculative sampling; graph mode is N/A because sampling follows the captured model-forward graph.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `output_token_ids_ptr` | Output | Result `[batch_size, max_spec_len + 1]` | int32 / int64 | ND |
| `cu_num_draft_tokens_ptr` | Input | Inclusive cumulative draft counts | int32 / int64 | ND |
| `draft_token_ids_ptr` | Input | Flattened draft IDs | int32 / int64 | ND |
| `draft_probs_ptr`, `target_probs_ptr` | Input | Draft and target distributions; draft may be null | fp32 | ND |
| `target_indices_ptr` | Input | Global IDs for reduced target vocabulary, or null | int32 / int64 | ND |
| `bonus_token_ids_ptr`, `recovered_token_ids_ptr` | Input | Bonus IDs per request and pre-sampled recovery IDs per token | int32 / int64 | ND |
| `uniform_probs_ptr` | Input | Uniform random values per draft token | fp32 | ND |
| `is_greedy_ptr` | Input | Per-request greedy mask | bool / int8 | ND |
| `max_spec_len`, `vocab_size`, `global_vocab_size`, `vec_len` | Attribute | Output width, active/reduced vocabulary widths, and request count | int | scalar |
| `ori_target_probs_ptr` | Input | Optional original distribution used for entropy | fp32 | ND |
| `synthetic_conditional_rates_ptr` | Input | Optional benchmark-only rates per position | fp32 | ND |
| `NO_ORI_TARGET_PROBS`, `NO_DRAFT_PROBS`, `ENABLE_REDUCE_SAMPLING`, `SYNTHETIC_MODE`, `ENTROPY_VERIFY` | Attribute | Compile-time mode switches | bool | scalar |
| `BLOCK_SIZE`, `VOCAB_BLOCK_SIZE`, `SUB_BLOCK` | Attribute | Request and vocabulary tile sizes | int | scalar |
| `POSTERIOR_THRESHOLD`, `POSTERIOR_ALPHA`, `EPSILON` | Attribute | Entropy-verification constants | fp32 | scalar |

## Constraints

- Cumulative counts are monotonic, each request has at most `max_spec_len` drafts, token IDs are valid or `-1`, and probabilities are finite and non-negative.
- Reduced sampling requires `target_indices_ptr` and uses `vocab_size` as selected width; otherwise probability rows use `global_vocab_size`.
- Mode-dependent nullable pointers must be present whenever their corresponding compile-time switch requires them.
- Entropy verification is applied only by the dense-vocabulary branch; the current reduced-vocabulary branch does not use `ENTROPY_VERIFY`.

## Origin and Differences

- **Origin**: Adapted from vLLM's rejection sampler.
- **Differences**:
    - NPU adaptation for performance: verifies request-local draft sequences in an Ascend vector-core kernel without host-side token loops.
    - Modified for vllm-ascend logic: combines dense and reduced-vocabulary sampling, entropy thresholds, missing draft distributions, and synthetic benchmarking modes.

## Test Cases

The direct NPU test compares bit-exact int64 output with the PyTorch reference for `max_spec_len` in `{1,2,3}`, batch sizes `{1,256,512,1024}`, vocabulary size 1024, and normal/synthetic modes. Reduced sampling, entropy verification, `NO_DRAFT_PROBS=True`, and placeholder ID `-1` are not directly covered by this test case; their compile variants are exercised by warm-up and broader integration tests.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py
```
