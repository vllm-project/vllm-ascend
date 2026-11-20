# gather_ssm_states

Gather selected physical SSM cache rows.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/mamba/state_index.py#L124) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/gdn.py#L852)

## Calling convention

```text
gather_ssm_states(state: torch.Tensor, indices: torch.Tensor, has_initial_state: torch.Tensor, *, output_dtype: torch.dtype | None=None)
```

Gather selected rows from a state cache with a padded batch stride.

Unlike ``state[indices]`` or ``index_select``, this kernel reads only the
requested rows. ``has_initial_state=False`` produces a zero row without
reading the state cache.

``state`` has shape ``[num_state_rows, *inner_dims]``. The inner
dimensions must be contiguous; only ``state.stride(0)`` may include
physical page padding. ``indices`` select physical state rows and may be
unsorted or non-contiguous. Callers must provide valid row IDs. The
optional ``output_dtype`` only changes the gathered output dtype.

See the [shared implementation and constraints](implementation_notes/mamba_state_index.md#parameters) and [test coverage](implementation_notes/mamba_state_index.md#test-cases).
