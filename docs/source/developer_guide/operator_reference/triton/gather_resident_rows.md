# gather_resident_rows

Gather requested rows from resident KV storage.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/hisparse/cache_ops.py#L461) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/hisparse/backend.py#L221)

## Calling convention

```text
gather_resident_rows(source: torch.Tensor, destination: torch.Tensor, row_ids: torch.Tensor)
```

Copy packed Resident rows using Core's negative-ID-to-row0 convention.

Source and destination may use different dtypes with the same byte width.

See the [shared implementation and constraints](implementation_notes/hisparse.md#parameters) and [test coverage](implementation_notes/hisparse.md#test-cases).
