# build_dspark_swa_indices_triton

Build request-scoped sliding windows for DSpark queries.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/spec_decode/dspark_swa_indices.py#L47) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/dsa_v1.py#L608)

## Calling convention

```text
build_dspark_swa_indices_triton(query_start_loc: torch.Tensor, seq_lens: torch.Tensor, num_tokens: int, index_width: int, window_size: int)
```

Build request-local DSpark SWA indices with one kernel launch.

Query boundaries and sequence lengths Build DSpark's shared visible window for queries belonging to the same request; padded graph rows have zero valid length.

See the [shared implementation and constraints](implementation_notes/dsv41_cache.md).
