# sample_recovered_tokens_kernel

## Description

- **Function**: Selects recovery tokens from the positive residual distribution after stochastic speculative rejection.
- **Formula**: `recovered = argmax_v(max(p_target(v)-p_draft(v),0)/q(v))`; without draft probabilities, the proposed draft ID is assigned zero probability.
- **Algorithm flow** (processed request and draft position independently): resolve the flattened token index, scan full or reduced vocabulary tiles, track the highest valid residual-to-random score, and store its global token ID.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Supports dense and reduced-vocabulary speculative sampling; graph mode is N/A because sampling follows the captured model-forward graph.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `output_token_ids_ptr` | Output | Recovery ID per flattened draft token | int32 / int64 | ND |
| `cu_num_draft_tokens_ptr` | Input | Inclusive cumulative draft counts | int32 / int64 | ND |
| `draft_token_ids_ptr` | Input | Proposed draft IDs | int32 / int64 | ND |
| `draft_probs_ptr`, `target_probs_ptr` | Input | Draft and target probability rows | fp32 | ND |
| `target_indices_ptr` | Input | Global IDs for reduced-vocabulary columns | int32 / int64 | ND |
| `q_ptr` | Input | Positive exponential/random variates per request and candidate | fp32 | ND |
| `vocab_size`, `global_vocab_size` | Attribute | Selected and global vocabulary sizes | int | scalar |
| `NO_DRAFT_PROBS`, `ENABLE_REDUCE_SAMPLING` | Attribute | Compile-time distribution mode switches | bool | scalar |
| `SUB_BLOCK`, `VOCAB_BLOCK_SIZE` | Attribute | Full and reduced vocabulary tile sizes, 4096 and 512 in serving | int | scalar |

## Constraints

- The launch grid is `(batch_size, max_spec_len)`; positions beyond a request's actual length return without writing.
- Valid `q` entries must be finite and positive. Probability rows are finite and non-negative.
- Reduced mode requires global target indices; all valid indices must be within `[0, global_vocab_size)`.

## Origin and Differences

- **Origin**: Adapted from vLLM's rejection-sampling recovery step.
- **Differences**:
    - NPU adaptation for performance: computes the residual-score argmax in vocabulary tiles suitable for Ascend vector cores.
    - Modified for vllm-ascend logic: supports selected-vocabulary target probabilities without materializing a dense target row.

## Test Cases

The NPU test uses four requests, maximum speculative length 3, a five-token dense vocabulary, fp32 draft/target probabilities, and int64 token IDs. It includes a zero-draft request and requires bit-exact equality with the PyTorch reference. The direct test covers `NO_DRAFT_PROBS=False` and dense sampling; reduced-vocabulary and no-draft-probability variants are currently covered only by warm-up/integration paths.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py::test_sample_recovered_tokens_kernel
```
