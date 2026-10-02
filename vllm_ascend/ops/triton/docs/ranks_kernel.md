# _ranks_kernel

## Description

- **Function**: Computes the rank of each sampled token in its logit row.
- **Formula**: `rank[r] = sum_v(logits[r,v] > logits[r, sampled_token_ids[r]])`.
- **Algorithm flow** (processed row by row, independently): load the sampled-token logit, scan the vocabulary in tiles, count strictly larger logits, reduce the count, and store it.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used when producing sampled-token log probabilities; graph mode is N/A because output processing follows the captured model-forward graph.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `output_ptr` | Output | Token ranks `[batch_size]` | int64 | ND |
| `logits_ptr` | Input | Sampler logits `[batch_size, vocab_size]` | fp32 | ND |
| `logits_stride` | Attribute | Logit row stride | int | scalar |
| `token_ids_ptr` | Input | Sampled token ID per row | int64 | ND |
| `vocab_size`, `batch_size` | Attribute | Valid column and row counts | int | scalar |
| `rows_per_core` | Attribute | Rows assigned to each vector core | int | scalar |
| `BLOCK_SIZE` | Attribute | Compile-time vocabulary tile size (8192 in the launcher) | int | scalar |

## Constraints

- Every sampled token ID must be in `[0, vocab_size)`; logits rows use a valid `logits_stride`.
- Ties do not increase the rank. NaNs follow Triton's comparison semantics.
- The row count may be dynamic; vocabulary tails are masked.

## Origin and Differences

- **Origin**: Adapted from vLLM's `vllm/v1/worker/gpu/sample/logprob.py`.
- **Differences**:
    - NPU adaptation for performance: batches multiple rows per Ascend vector core and accumulates counts in int32 vectors before producing int64 output.
    - Modified behavior: uses a strict `>` comparison, so logits tied with the sampled-token logit do not increase its zero-based rank.

## Test Cases

The NPU test compares against PyTorch for `(batch_size, vocab_size, num_logprobs)` values `(48,1024,5)`, `(96,1024,0)`, `(24,1519,1)`, and `(1,320,10)`. Token IDs and ranks are bit-exact; the related fp32 log probabilities use `rtol=atol=1e-4`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py
```
