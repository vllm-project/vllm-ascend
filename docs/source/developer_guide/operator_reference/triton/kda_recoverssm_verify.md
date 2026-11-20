# kda_recoverssm_verify

Verify speculative KDA tokens against a paged checkpoint.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/kda/recoverssm.py#L1008) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/glm5next/kda.py#L758)

## Calling convention

```text
kda_recoverssm_verify(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, raw_g: torch.Tensor, raw_beta: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float | None, checkpoint_state: torch.Tensor, correction_cache: torch.Tensor, kg_cache: torch.Tensor, query_start_loc: torch.Tensor, state_indices: torch.Tensor, spec_query_len: int, out: torch.Tensor | None=None)
```

Verify a KDA speculative window without modifying its checkpoint.

See the [shared implementation and constraints](implementation_notes/kda_recoverssm.md#parameters) and [test coverage](implementation_notes/kda_recoverssm.md#test-cases).
