# build_c2_ring_metadata

Derive C2 ring addresses and request controls.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/c2_ring_metadata.py#L143) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/dsa_v41.py#L1406)

## Calling convention

```text
build_c2_ring_metadata(query_start_loc: torch.Tensor, seq_lens: torch.Tensor, positions: torch.Tensor, block_table: torch.Tensor, source_cos: torch.Tensor, source_sin: torch.Tensor, num_reqs: int, num_tokens: int, num_actual_reqs: int, num_actual_tokens: int, *, skip_update: bool, ring_metadata_output: torch.Tensor, complete_mask_output: torch.Tensor, source_positions_output: torch.Tensor, cos_output: torch.Tensor, sin_output: torch.Tensor)
```

Request boundaries, positions, block tables and ring controls Derive compressor state addresses and request controls on device, avoiding per-request host synchronization.

See the [shared implementation and constraints](implementation_notes/dsv41_cache.md).
