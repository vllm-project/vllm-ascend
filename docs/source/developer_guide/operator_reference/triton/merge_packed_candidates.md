# merge_packed_candidates

Merge gathered DCP candidate lists.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/sharded_indexer.py#L543) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/context_parallel/sharded_indexer.py#L54)

## Calling convention

```text
merge_packed_candidates(packed, rows: int, world_size: int, topk: int, *, local_rank: int | None=None, interleave: int=1, out=None, presorted=False, owner_sharded=False)
```

Fuse exact Top-K and owner compaction.

``owner_sharded`` additionally promises that rank r contains only owner r
IDs, as guaranteed by pack_local_candidates. Presorted alone is weaker.

See the [shared implementation and constraints](implementation_notes/sharded_indexer.md#parameters) and [test coverage](implementation_notes/sharded_indexer.md#test-cases).
