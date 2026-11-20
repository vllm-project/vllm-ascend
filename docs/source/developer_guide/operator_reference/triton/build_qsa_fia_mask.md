# build_qsa_fia_mask

Expand compact QSA selections into the FIA attention mask.

[Implementation of `build_qsa_fia_mask`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/qwen4_exp/fia.py) · [Production caller in fia.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/qwen4_exp/fia.py)

## Calling convention

```python
build_qsa_fia_mask(compact_indices, visible_lens, kv_length, mask_width, workspace)
```

Each row of `compact_indices` contains complete-group IDs followed by three residual-token slots. The wrapper expands groups of four within the visible lengths and returns a boolean mask of shape `[rows, mask_width]` and one nonempty-selection flag per row. A false mask element makes that KV position visible.

Size the process-local workspace before graph capture. Returned tensors share reusable workspace storage. The implementation honors its row stride when a narrower batch follows a wider one.

## Validation

[Integration and semantic tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_qsa.py)
