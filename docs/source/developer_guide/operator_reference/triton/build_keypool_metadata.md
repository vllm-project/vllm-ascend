# build_keypool_metadata

Build metadata for the GLM pooled key indexer.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/glm5_next_cann_keypool.py#L315) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/indexer_kpool.py#L515)

## Calling convention

```text
build_keypool_metadata(common, buffers, tokens, source, logical, ratio)
```

See the [shared implementation and constraints](implementation_notes/glm5_next_cann_keypool.md#parameters) and [test coverage](implementation_notes/glm5_next_cann_keypool.md#test-cases).
