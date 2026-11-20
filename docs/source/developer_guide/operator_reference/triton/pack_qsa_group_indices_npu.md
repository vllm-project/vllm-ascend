# pack_qsa_group_indices_npu

Pack complete QSA group IDs and visible residual tokens.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/qwen4_exp/qsa.py#L305) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/qwen4_exp/lightning_indexer.py#L119)

## Calling convention

```text
pack_qsa_group_indices_npu(block_indices: torch.Tensor, query_positions: torch.Tensor, sequence_lengths: torch.Tensor, token_to_req: torch.Tensor, compress_ratio: int, token_topk: int, out: torch.Tensor | None=None)
```

Pack LI-v2 groups and the open-group tail into one fixed buffer.

This index-only operation is independent of attention head/rotary width.
The compact output width is token_topk // compress_ratio + compress_ratio - 1;
the caller supplies group IDs selected by the indexer, not Q/K vectors.

See the [shared implementation and constraints](implementation_notes/qwen4_exp_qsa_dispatch.md).
