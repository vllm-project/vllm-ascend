# pack_qsa_group_indices_npu

Pack selected groups and visible residual tokens for QSA.

[Implementation of `pack_qsa_group_indices_npu`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/qwen4_exp/qsa.py) · [Caller `qsa_select_paged_tokens_lightning_v2` in lightning_indexer.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/qwen4_exp/lightning_indexer.py)

## Calling convention

```python
pack_qsa_group_indices_npu(block_indices, query_positions, sequence_lengths, token_to_req, compress_ratio, token_topk, out=None)
```

The indexer supplies one row of complete-group IDs per query, its logical position, the per-request sequence lengths and the token-to-request map. The output is an int32 tensor on the NPU. Pass an output buffer of the same shape to reuse storage.

The row width is `token_topk // compress_ratio + compress_ratio - 1`. With 2048 selected tokens and groups of four, the first 512 slots contain group IDs and the final three contain residual token positions. Invalid lanes contain `-1`. The top-k must be divisible by the compression ratio.

## Validation

[`test_qsa_pack_supports_non_power_of_two_output_width`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_qsa.py) checks the 515-slot packing layout on NPU.
