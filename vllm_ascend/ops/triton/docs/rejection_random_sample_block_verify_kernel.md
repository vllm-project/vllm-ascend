# rejection_random_sample_block_verify_kernel

## Description

- **Function**: Verifies stochastic draft sequences with block-level cumulative acceptance and emits the accepted prefix, recovery token, or bonus token.
- **Formula**: Starting from `pi_-1 = U_-1 = 1`, update `pi_i = min(pi_(i-1) * p_target(x_i) / p_draft(x_i), 1)` and `U_i = product_(j<=i) u_j`. Accept the prefix through the largest `i` satisfying `pi_i >= U_i`. In entropy-verification mode, compare against `min(exp(-alpha * entropy_i), posterior_threshold) * U_i` instead.
- **Algorithm flow** (processed request by request, independently): locate the flattened draft range, skip greedy requests, scan all positions while accumulating probability ratios and uniforms, copy the longest accepted prefix, then append its precomputed recovery token or the bonus token after full acceptance.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. The sampler selects this kernel when block verification is enabled and `max_spec_len >= 3`; graph mode is N/A because sampling follows the captured model-forward graph.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `output_token_ids_ptr` | Output | Result `[batch_size, max_spec_len + 1]` | int32 / int64 | ND |
| `cu_num_draft_tokens_ptr`, `draft_token_ids_ptr` | Input | Inclusive cumulative counts and flattened draft IDs | int32 / int64 | ND |
| `draft_probs_ptr`, `target_probs_ptr`, `ori_target_probs_ptr` | Input | Draft, active target, and optional original target probabilities | fp32 | ND |
| `target_indices_ptr` | Input | Global IDs for reduced target vocabulary | integer | ND |
| `bonus_token_ids_ptr`, `recovered_token_ids_ptr` | Input | Bonus and recovery token IDs | integer | ND |
| `uniform_probs_ptr`, `is_greedy_ptr` | Input | Uniform values per token and greedy mask per request | fp32 / bool | ND |
| `max_spec_len`, `vocab_size`, `global_vocab_size`, `vec_len` | Attribute | Sequence, vocabulary, and request bounds | int | scalar |
| `NO_ORI_TARGET_PROBS`, `NO_DRAFT_PROBS`, `ENABLE_REDUCE_SAMPLING`, `ENTROPY_VERIFY` | Attribute | Compile-time mode switches | bool | scalar |
| `BLOCK_SIZE`, `VOCAB_BLOCK_SIZE`, `SUB_BLOCK` | Attribute | Request and vocabulary tiles | int | scalar |
| `POSTERIOR_THRESHOLD`, `POSTERIOR_ALPHA`, `EPSILON` | Attribute | Entropy-verification constants | fp32 | scalar |

## Constraints

- Input layout and mode-dependent pointer requirements match `rejection_random_sample_kernel`; cumulative counts must be monotonic and bounded by allocated buffers.
- Draft probability must be positive for acceptance. Invalid draft ID `-1` forces rejection.
- The kernel processes only requests whose `is_greedy` value is false.
- Entropy verification is applied only by the dense-vocabulary branch; the current reduced-vocabulary branch does not use `ENTROPY_VERIFY`.

## Origin and Differences

- **Origin**: Introduced for vLLM Ascend's MagicMTP path, based on the block-verification method described in *Block Verification Accelerates Speculative Decoding* (`arXiv:2403.10444`). Upstream vLLM does not provide the same Triton kernel.
- **Differences**:
    - NPU adaptation for performance: uses a persistent Ascend vector-core request grid and scans each request's draft block in one program.
    - Modified for vllm-ascend logic: supports dense and reduced-vocabulary lookup and consumes precomputed recovery IDs instead of sampling a residual distribution inside this kernel.

## Test Cases

The direct NPU test compares bit-exact int64 output with a PyTorch reference
for batch size 7, `max_spec_len=3`, vocabulary size 5, variable per-request
draft counts including zero, `NO_DRAFT_PROBS=True`, and one greedy request.
Draft-distribution, reduced-vocabulary, and entropy-verification variants are
not directly covered by this test case.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py
```
