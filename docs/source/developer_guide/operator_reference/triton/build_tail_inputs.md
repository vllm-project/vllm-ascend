# build_tail_inputs

Prepare GLM KeyPool tail inputs.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/glm5_next_cann_keypool.py#L379) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/indexer_kpool.py#L693)

## Calling convention

```text
build_tail_inputs(common, pool, capacity, page_rows, retained_width, buffers)
```

See the [shared implementation and constraints](implementation_notes/glm5_next_cann_keypool.md#parameters) and [test coverage](implementation_notes/glm5_next_cann_keypool.md#test-cases).
