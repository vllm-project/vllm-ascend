# Static W8A8 projection input quantization

The optional Triton kernel converts the input to an attention output projection
(`enable_oproj_triton_quant`) or an Indexer query projection
(`enable_indexer_q_triton_quant`) to contiguous INT8. Set either option in
`--additional-config`. Both default to false. Already quantized activations
and non-static schemes keep their existing paths.

The operator takes the activation, the stored reciprocal scale, and the offset.
It implements the rounding, non-finite conversion, and saturation of
`torch_npu.npu_quantize(x, reciprocal, offset, torch.qint8, -1, False)`.
Scale and offset remain runtime tensor arguments, including during graph replay.
Both direct linear execution and overlap preparation call the same operator.
The GEMM and output reduction use the existing W8A8 implementation.

## Inputs and launch

The Triton path supports contiguous and strided ND FP16, BF16 and FP32 activations of ranks
1 through 8, with scalar or last-dimension parameters. A row/column partition
and UB tiles are derived from the actual shape, dtype and device resources.
Tensor values do not select a specialization. Shape, strides, dtype and tile
parameters may produce different specializations. A layout outside the Triton
path or unavailable platform geometry uses the native quantizer.

## Verification

Compare INT8 outputs exactly with the CANN operator, including strided views,
ties, signed saturation and graph replay after parameter updates:

```sh
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_static_projection_quantize.py
pytest -q tests/ut/ops/test_static_quantize_tiling.py
```
