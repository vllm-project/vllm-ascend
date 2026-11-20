# dequantize_packed_engram

Decode packed Engram embedding weights.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/engram_lookup.py#L37) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/deepseek_v41/engram/shared_rdma.py#L345)

## Calling convention

```text
dequantize_packed_engram(packed, output, rows, row_bytes, scale_dtype, code_dtype='I8')
```

See the [shared implementation and constraints](implementation_notes/engram.md).
