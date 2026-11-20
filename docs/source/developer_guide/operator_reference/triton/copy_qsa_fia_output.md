# copy_qsa_fia_output

Write selected FIA result rows and clear empty rows.

[Implementation of `copy_qsa_fia_output`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/qwen4_exp/fia.py) · [Caller `qsa_fia_attention` in fia.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/qwen4_exp/fia.py)

## Calling convention

```python
copy_qsa_fia_output(result, selected, output)
```

Supply densely packed FIA results, one boolean selection flag per query row and an output tensor shaped `[rows, heads, head_dim]`. The wrapper copies selected rows and writes zeros to the remaining rows. It writes in place, supports strided destinations and returns `None`.

## Validation

[`test_fia_passes_gqa_heads_causal_mask_and_zeroes_empty_rows`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/ut/models/qwen4_exp/test_fia_dispatch.py) checks the zero-row policy in the calling attention path. Direct NPU coverage of strided output writes remains open.
