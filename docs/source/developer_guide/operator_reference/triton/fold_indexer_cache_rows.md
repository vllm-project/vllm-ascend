# fold_indexer_cache_rows

Fold updated cache rows into groups of eight.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/fold_indexer_cache.py#L57) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/deepseek_v41/indexer.py#L127)

## Calling convention

```text
fold_indexer_cache_rows(cache: tuple[torch.Tensor, torch.Tensor], folded: torch.Tensor, slots: torch.Tensor)
```

Copy freshly written token rows into their candidate-block group.

K `[N,B,1,64]`, scales `[N,B,1,4]`, slots; folded `[N,B/8,1,544]` Copy only updated rows into eight-row groups: 512 key bytes followed by 32 scale bytes. This is a layout conversion, with no requantization.

See the [shared implementation and constraints](implementation_notes/dsv41_cache.md).
