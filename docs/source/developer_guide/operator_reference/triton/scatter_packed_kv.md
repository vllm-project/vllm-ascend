# scatter_packed_kv

Scatter packed payloads into the HiSparse KV cache.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/hisparse/cache_ops.py#L538) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/sfa_hisparse.py#L337)

## Calling convention

```text
scatter_packed_kv(k_nope, k_rope, packed_cache, slot_mapping, num_valid_rows=None)
```

Write current KV with explicit page strides; ignore padding slots.

See the [shared implementation and constraints](implementation_notes/hisparse.md#parameters) and [test coverage](implementation_notes/hisparse.md#test-cases).
