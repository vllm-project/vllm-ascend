# Fused C8 cached-prefill KV preparation

`try_fused_c8_prefill` prepares dense K and V for fused inference attention
(FIA) on GQA cached-prefill and continuing chunked-prefill paths. It combines
paged gather, NZ-to-ND conversion, and scale-only dequantization in one Triton
launch.

The wrapper takes contiguous INT8 NZ caches, an INT32 or INT64 block table
with contiguous columns, host sequence lengths, block and head geometry, and
per-channel K and V scales. Blocks and head sizes are multiples of 32. BF16 and
FP16 target dtypes are supported with BF16, FP16, or FP32 scales. The layout
within a page is `[kv_heads, head_size / 32, block_size, 32]`. Live block IDs must
refer to valid physical pages. Padding after the live columns is skipped;
the final live page contributes only the tokens covered by its sequence length.

The two output tensors have shape
`[sum(seq_lens), num_kv_heads, head_size]`, in request order. Their dtypes
follow the promotion of the target dtype with the respective scale dtype.
Inputs are read-only. The wrapper returns `None` for unsupported inputs or
active stream capture, so callers use the existing gather-and-dequantize path.

The regression test uses an independent PyTorch reference that gathers live
tokens in ND layout and applies the scales. It checks both output tensors
bit for bit, including mixed precisions, partial pages, padded block tables, and
a large grid. CPU tests cover caller dispatch and fallback.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q tests/ut/attention/test_c8_prefill_dispatch.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_c8_prefill.py
```
