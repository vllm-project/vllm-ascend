# normalize_dcp_candidates

Normalize candidate layout for DCP selection.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/sharded_indexer.py#L818) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/device/device_op.py#L145)

## Calling convention

```text
normalize_dcp_candidates(indices, scores, query_start_loc)
```

Clear invalid candidates and graph padding in native FP32 outputs in place.

QLI does not initialize rows beyond the cumulative query endpoint. Each
kernel lane owns its index/score slot, so no extra transport buffer is needed.

See the [shared implementation and constraints](implementation_notes/sharded_indexer.md#parameters) and [test coverage](implementation_notes/sharded_indexer.md#test-cases).
