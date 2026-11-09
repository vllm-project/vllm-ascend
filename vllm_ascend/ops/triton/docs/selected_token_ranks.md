# selected_token_ranks

> Source: `vllm_ascend/worker/v2/sample/logprob.py` (`_ranks_kernel`, called by `compute_topk_logprobs`).

## Description

`selected_token_ranks` counts real vocabulary entries whose logits are greater
than or equal to the selected logit. A unique maximum ranks 1; both members of a
two-way top tie rank 2. NaN comparisons follow `>=` semantics.

For log-probability calculation, see [topk_log_softmax](topk_log_softmax.md).

## Padding handling

The last vocabulary tile is padded with negative infinity. When the selected
logit is negative infinity, the kernel subtracts the padding count after the
reduction. This avoids an extra vector predicate that exceeds the A3 UB
budget with Triton-Ascend 3.2.2 and preserves the count of real entries even
when a row contains NaN.

## Test Cases

`test_compute_topk_logprobs.py` covers ranks against a PyTorch reference,
1-based and tied ranks, padding exclusion, and NaN comparison behavior.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py
```
