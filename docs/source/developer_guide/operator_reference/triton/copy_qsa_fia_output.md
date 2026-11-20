# copy_qsa_fia_output

Write selected FIA result rows and clear empty rows.

[Implementation of `copy_qsa_fia_output`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/qwen4_exp/fia.py) · [Production caller in fia.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/qwen4_exp/fia.py)

## Calling convention

```python
copy_qsa_fia_output(result, selected, output)
```

Supply densely packed FIA results, one boolean selection flag per query row and an output tensor shaped `[rows, heads, head_dim]`. The wrapper copies selected rows and writes zeros to the remaining rows. It writes in place, supports strided destinations and returns `None`.

## Validation

[Integration and semantic tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_qsa.py)
