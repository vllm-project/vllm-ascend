# build_local_lengths

Compute rank-local lengths for a sharded index.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/sharded_indexer.py#L739) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/indexer.py#L1598)

## Calling convention

```text
build_local_lengths(seq_lens, query_ends, key_lengths, visible_lengths, world_size: int, rank: int, interleave: int)
```

Write stable metadata buffers used by native causal local QLI.

See the [shared implementation and constraints](implementation_notes/sharded_indexer.md#parameters) and [test coverage](implementation_notes/sharded_indexer.md#test-cases).
