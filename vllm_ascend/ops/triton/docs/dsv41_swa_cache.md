# DeepSeek-V4.1 fused SWA cache store

## Description

`dsv41_rmsnorm_rope_cache` consumes the BF16 output of the DeepSeek-V4.1 WKV
projection, applies RMSNorm and interleaved partial RoPE, and writes each row
directly to its paged SWA cache coordinate in one Triton kernel. The WKV GEMM
is intentionally not part of the fusion.

The fused path replaces three operators:

1. RMSNorm, which formerly materialized a full normalized KV tensor;
2. in-place partial RoPE, which read and rewrote the rotary slice; and
3. `npu_scatter_nd_update_sk`, which read the normalized/rotated tensor and
   scattered it to the paged cache.

Each Triton program owns one token. It accumulates the RMS reduction in FP32,
applies the RMS weight, rounds the normalized row to BF16, and only then
performs FP32 RoPE arithmetic. This preserves the standalone RMSNorm-to-RoPE
BF16 rounding boundary. The interleaved rotation pairs adjacent elements and
uses the same even-element negative sine convention as
`inplace_partial_rotary_mul`.

## Cache and graph contract

- Slot mappings are builder-prepared physical `[block, offset]` coordinates.
  Either negative component suppresses the complete row, including graph
  padding.
- Cache block, row, and feature offsets use the tensor's real strides. A SWA
  cache plane may therefore remain a view into a layer-outermost allocation.
- Positions are already reflected in the selected `cos` and `sin` tensors.
  The kernel reads their device data on every invocation, so ACLGraph replay
  can update RoPE rows and slot coordinates without host synchronization.
- The kernel launches on the caller's current stream. In the DSV4.1 overlap
  schedule it stays on the auxiliary KV stream, and the existing
  `main_stream.wait_stream(aux_stream)` remains the publication barrier.

## Selection and fallback

The model selects this path automatically on A2 (910B) when Triton is available,
the WKV output and SWA cache are BF16 NPU tensors, the cache has the standard
four-dimensional single-head layout, the two-dimensional slot mapping is
compatible, the RMSNorm has no loaded bias, and the batch contains at most
8192 rows. The supported DeepSeek-V4.1 head width is 512 with an even,
non-empty rotary suffix.

A3, A5 packed caches, larger batches, other dtypes/layouts, and unsupported
devices retain the original RMSNorm, partial RoPE, and cache-scatter chain.
Long KV, C1/C2 compressor, and indexer cache writes do not call this operator.

## Validation

Operator correctness and ACLGraph replay:

```bash
pytest -sv tests/e2e/pull_request/one_card/test_dsv41_swa_cache_fusion.py
```

Microbenchmark against the exact three-operator baseline:

```bash
python benchmarks/dsv41_swa_cache.py --tokens 1 8 32 128 512 2048 8192
```

The operator test covers interleaved partial RoPE, the BF16 normalization
boundary, shuffled and negative coordinates, padded projection/RoPE/cache
strides, untouched cache rows, and replay after inputs and slots change.
