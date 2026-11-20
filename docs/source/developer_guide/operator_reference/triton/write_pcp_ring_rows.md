# write_pcp_ring_rows

Write gathered PCP request tails to the compressor ring.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/compressor/compressor_triton.py#L715) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/context_parallel/dsa_v41_cp.py#L698)

## Calling convention

```text
write_pcp_ring_rows(gathered_kv_scores, ring_state, ring_metadata, source_row_indices, request_indices, token_offsets)
```

Write request-relative rows using rejection-corrected device ring starts.

Gathered FP32 KV/gate rows and request-relative offsets; persistent ring Write canonical PCP request tails after local pooling reads complete.

See the [shared implementation and constraints](implementation_notes/dsv41_cache.md).
