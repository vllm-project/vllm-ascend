# select_local_candidates

Select candidates from the local indexer shard.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/sharded_indexer.py#L394) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/sharded_indexer.py#L644)

## Calling convention

```text
select_local_candidates(indices, scores, world_size: int, rank: int, interleave: int, *, out=None)
```

Fuse final score masking, local remap and stable compaction.

See the [shared implementation and constraints](implementation_notes/sharded_indexer.md#parameters) and [test coverage](implementation_notes/sharded_indexer.md#test-cases).
