# build_packed_cache_slot_mapping

Convert physical slots to packed page and row coordinates.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/packed_cache_slot_mapping.py#L67) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/deepseek_v41/mixed_quant_attention.py#L13)

## Calling convention

```text
build_packed_cache_slot_mapping(slots: torch.Tensor, positions: torch.Tensor, query_start_loc: torch.Tensor, num_tokens: int, num_actual_reqs: int, num_actual_tokens: int, page_size: int, compress_ratio: int, *, skip_update: bool=False, coordinates_output: torch.Tensor, flat_output: torch.Tensor)
```

Publish coordinate and flat packed cache addresses together.

This replaces the eager GE/Add/Remainder/And/Div/Where/Copy chain used by
every KV cache group. Ratio-2 completion and padded-row masking follow the
existing metadata builder contract exactly.

INT64 physical slots and positions, INT32 query boundaries; coordinate `[T,2]` and flat `[T]` output buffers Convert physical slots to page/row coordinates. Ratio 2 writes only completed pairs. Mask negative slots, graph padding and skipped state updates to `-1`.

See the [shared implementation and constraints](implementation_notes/dsv41_cache.md).
