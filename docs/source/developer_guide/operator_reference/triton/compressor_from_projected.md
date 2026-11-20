# compressor_from_projected

Pool projected C2 key/value pairs against ring state.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/compressor/compressor_triton.py#L735) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/deepseek_v41/compressor.py#L61)

## Calling convention

```text
compressor_from_projected(kv, scores, state_cache, metadata, out, *, max_query_len, num_cores, prev_rows, prev_row_indices)
```

Pool C2 into token-aligned BF16 rows; PCP updates the ring after gathering.

metadata is contiguous device INT32 [5, requests]: start positions, used
lengths, output bases (reserved), input bases, and actual global block IDs.
Controls must be bounded by the input rows and request table; padded requests
have used length and block ID zero. No tensor contents are read by the host.
With prev_rows, chunk predecessors come from the gather or the old ring,
and the caller writes the global request tails after all local reads.

FP32 KV/gate projections and ring controls; token-aligned BF16 output Pool C2 pairs, reading leading predecessors from gathered rows or the persistent ring.

See the [shared implementation and constraints](implementation_notes/dsv41_cache.md).
