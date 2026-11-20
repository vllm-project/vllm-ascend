# build_local_key_lengths

Build local causal key lengths for DCP.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/sharded_indexer.py#L796) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/indexer.py#L1583)

## Calling convention

```text
build_local_key_lengths(seq_lens, key_lengths, world_size: int, rank: int, interleave: int)
```

Build local cache lengths when QLI computes causal rows internally.

See the [shared implementation and constraints](implementation_notes/sharded_indexer.md#parameters) and [test coverage](implementation_notes/sharded_indexer.md#test-cases).
