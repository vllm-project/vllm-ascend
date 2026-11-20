# scatter_ssm_states_

Write updated SSM rows back into cache.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/mamba/state_index.py#L170) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/gdn.py#L864)

## Calling convention

```text
scatter_ssm_states_(state: torch.Tensor, indices: torch.Tensor, source: torch.Tensor)
```

Write dense state rows into selected slots of a padded state cache.

The state layout follows :func:`gather_ssm_states`: inner dimensions are
contiguous and padding is represented by ``state.stride(0)``. ``indices``
select physical rows and may be unsorted or non-contiguous. ``source``
must have the logical state shape and contiguous inner dimensions.

See the [shared implementation and constraints](implementation_notes/mamba_state_index.md#parameters) and [test coverage](implementation_notes/mamba_state_index.md#test-cases).
