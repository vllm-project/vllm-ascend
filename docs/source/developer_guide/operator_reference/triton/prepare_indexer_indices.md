# prepare_indexer_indices

Filter and order visible compressed indexer rows.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/prepare_indexer_indices.py#L213) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/deepseek_v41/indexer.py#L301)

## Calling convention

```text
prepare_indexer_indices(selected: torch.Tensor, positions: torch.Tensor, compress_ratio: int, output: torch.Tensor | None=None, *, indices_output: torch.Tensor | None=None, lengths_output: torch.Tensor | None=None)
```

Filter/sort indices into optional graph-stable output buffers.

INT32 top-k `[T,K]`, positions and compression ratio; indices and optional lengths Filter invisible compressed rows, sort surviving row indices chronologically and pad with `-1`. The FP32 bitwise sort keys preserve integer ordering without numeric conversion.

See the [shared implementation and constraints](implementation_notes/dsv41_cache.md).
