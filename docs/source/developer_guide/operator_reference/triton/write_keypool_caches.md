# write_keypool_caches

Write GLM KeyPool cache entries.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/glm5_next_cann_keypool.py#L78) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/indexer_kpool.py#L978)

## Calling convention

```text
write_keypool_caches(tail, cache, key, gate, pooled, metadata, tail_metadata, state, scale_cache=None, pooled_scales=None)
```

See the [shared implementation and constraints](implementation_notes/glm5_next_cann_keypool.md#parameters) and [test coverage](implementation_notes/glm5_next_cann_keypool.md#test-cases).
