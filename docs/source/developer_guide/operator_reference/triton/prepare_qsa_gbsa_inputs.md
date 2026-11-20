# prepare_qsa_gbsa_inputs

Prepare sparse attention indices and per-query block tables for GBSA.

[Implementation of `prepare_qsa_gbsa_inputs`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/qwen4_exp/gbsa.py) · [Production caller in gbsa.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/qwen4_exp/gbsa.py)

## Calling convention

```python
prepare_qsa_gbsa_inputs(compact_indices, logical_positions, seq_lens, token_to_req, block_table, num_kv_heads)
```

Supply compact group indices, logical positions, sequence lengths, token-to-request IDs and a request block table addressing 128-token pages. Positive block-table row and column strides are supported. The wrapper removes invalid groups, preserves selection order and appends the residual group or replaces the last group at the 512-entry limit.

It returns five tensors in order: int32 sparse indices `[kv_heads, rows, output_groups]` padded with `-1`, int32 counts `[kv_heads, rows]`, int32 visible lengths `[rows]`, int64 cumulative query offsets `[rows + 1]` and gathered block-table rows `[rows, pages]`.

## Validation

[Integration and semantic tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_qsa.py)
