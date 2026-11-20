# pack_local_candidates

Pack local candidate IDs and scores for DCP exchange.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/sharded_indexer.py#L473) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/context_parallel/sharded_indexer.py#L44)

## Calling convention

```text
pack_local_candidates(indices, scores, world_size: int, rank: int, interleave: int, *, sort_candidates=False, fuse_sort=False)
```

Pack local candidates as global INT32 IDs and FP32 score-bit pairs.

Scores must be BF16 or FP32; indices and scores have shape [rows, 1, topk].
The current supported input contract requires valid local token IDs in
[0, 2**20); negative IDs remain invalid-candidate sentinels. This bound
applies to token IDs before global mapping, not candidate-array positions.
Callers must ensure this bound; it is not validated by the wrapper.
Sorting orders scores descending, then IDs ascending. ``fuse_sort`` permits
fused sorting for workloads that meet the occupancy and merge-work gates.

See the [shared implementation and constraints](implementation_notes/sharded_indexer.md#parameters) and [test coverage](implementation_notes/sharded_indexer.md#test-cases).
