# write_mxfp4_indexer_cache

Quantize indexer keys and write valid cache slots.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/quantize_mxfp4_indexer.py#L213) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/deepseek_v41/mixed_quant_attention.py#L329)

## Calling convention

```text
write_mxfp4_indexer_cache(x: torch.Tensor, slots: torch.Tensor, data_cache: torch.Tensor, scale_cache: torch.Tensor)
```

Quantize ``x[T,128]`` and write strided A5 cache planes in one launch.

BF16 `[T,128]`, page/row slots `[T,2]`, strided UINT8 cache planes Fuse quantization with cache writes. Invalid slots leave the cache unchanged. Preserve 64-bit address arithmetic for large physical page strides.

See the [shared implementation and constraints](implementation_notes/dsv41_cache.md).
