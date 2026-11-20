# kda_state_scatter

Write packed KDA state to selected cache rows.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/kda_state_copy_plan.py#L490) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/kda_state_copy_plan.py#L517)

## Calling convention

```text
_kda_state_scatter(state: torch.Tensor, packed: torch.Tensor, indices: torch.Tensor, layer_name: str)
```

Expose cache mutation explicitly while keeping launches precompiled.

See the [shared implementation and constraints](implementation_notes/kda_state_copy.md).
