# quantize_mxfp4_indexer

Quantize BF16 indexer query groups into FP4 data and scales.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/quantize_mxfp4_indexer.py#L187) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/deepseek_v41/mixed_quant_attention.py#L30)

## Calling convention

```text
quantize_mxfp4_indexer(x: torch.Tensor)
```

Quantize contiguous 128-wide rows to packed E2M1 and E8M0 scales.

The result preserves the caller's leading dimensions: data has a final
width of 64 bytes and scale has a final width of four bytes.

BF16 query with final dimension 128; UINT8 packed data and scales Quantize groups of 32 to FP4, packing two values per byte. Store power-of-two scales in E8M0.

See the [shared implementation and constraints](implementation_notes/dsv41_cache.md).
