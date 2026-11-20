# pack_keypool_indices

Pack selected GLM KeyPool token indices.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/glm5_next_cann_keypool.py#L159) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/indexer_kpool.py#L976)

## Calling convention

```text
pack_keypool_indices(indices, metadata, topk, pool_size, output_buffer=None)
```

See the [shared implementation and constraints](implementation_notes/glm5_next_cann_keypool.md#parameters) and [test coverage](implementation_notes/glm5_next_cann_keypool.md#test-cases).
