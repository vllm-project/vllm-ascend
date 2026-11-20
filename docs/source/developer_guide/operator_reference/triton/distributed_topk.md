# distributed_topk

Select global or owner-local winners across DCP ranks.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/context_parallel/sharded_indexer.py#L30) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/indexer.py#L983)

The producer and SFA consumer use `use_local_topk_output` to keep one coordinate
system across producing and shared-index layers. Native sharded indexing with
physical PCP size one may return owner-local indices. PCP and TurboQuant retain
global indices because their consumers require those coordinates.

Global selection orders native comparator scores descending and global INT32 IDs
ascending for ties. Invalid candidates have ID -1 and score -infinity. Local output
contains the same global winners filtered by owner, remapped, compacted in selected
order and padded with -1. SFA skips its separate remap only for that layout.

Each query tile performs one packed candidate AllGather. Tiling uses a 256 MiB
receive-buffer budget with a 256-row floor; the floor can exceed that budget for
large DCP/K. Collective boundaries depend only on shared host shapes.

For the presorted owner-sharded K2048 contract, each owner's winners form an exact
prefix: candidate m has global rank m plus the number of better peer entries.
Bounded search finds that prefix without sorting all owners' winners. Generic
callers retain exact full selection. See [sharded indexer](implementation_notes/sharded_indexer.md) for
parameters, supported layouts and tests. The final attention output merge remains
required in both coordinate modes.
