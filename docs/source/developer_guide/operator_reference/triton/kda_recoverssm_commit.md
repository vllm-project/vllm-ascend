# KDARecoverSSMCommitContext.commit

Reconstruct and commit accepted KDA state.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/kda/recoverssm.py#L1362) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/kimi_k3/recoverssm_metadata.py#L27)

## Calling convention

```text
KDARecoverSSMCommitContext.commit(self, num_accepted_tokens: torch.Tensor, state_indices: torch.Tensor, query_start_loc: torch.Tensor, request_indices: torch.Tensor | None=None, block_table: torch.Tensor | None=None, num_computed_tokens: torch.Tensor | None=None, mamba_block_size: int | None=None)
```

Fold accepted KDA and convolution inputs into every layer.

See the [shared implementation and constraints](implementation_notes/kda_recoverssm.md#parameters) and [test coverage](implementation_notes/kda_recoverssm.md#test-cases).
