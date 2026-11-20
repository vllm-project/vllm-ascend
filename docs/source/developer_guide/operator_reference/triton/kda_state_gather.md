# kda_state_gather

Gather KDA state into packed working storage.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/kda_state_copy_plan.py#L453) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/kda_state_copy_plan.py#L510)

## Calling convention

```text
_kda_state_gather(state: torch.Tensor, indices: torch.Tensor, flags: torch.Tensor | None, layer_name: str)
```

Run the same eager gather behind the Dynamo/FakeTensor boundary.

See the [shared implementation and constraints](implementation_notes/kda_state_copy.md).
