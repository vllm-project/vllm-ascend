# DeepSeek-V4.1 indexer prologue INT8 fusion

`FusedIndexerPrologue` is the A2/A3 AscendC implementation of the
DeepSeek-V4.1 index-key publication prologue. It is exposed as
`torch.ops._C_ascend.fused_indexer_prologue`.

For each compressor latent row, one mixed AIC/AIV launch performs:

1. BF16 `Wk` projection (`[T, 512] x [128, 512]^T`) on Cube cores;
2. RMSNorm with FP32 accumulation and BF16 output rounding;
3. adjacent-pair partial RoPE on the last 64 key dimensions;
4. per-token symmetric INT8 dynamic quantization;
5. direct writes to the strided paged INT8 key cache and FP16 scale cache.

The projection is materialized only in operator workspace and consumed by the
Vector post-processing phase after an in-kernel AIC-to-AIV synchronization. No
projected or normalized key tensor is exposed to Python or written as a model
intermediate. Invalid `[-1, -1]` cache coordinates are skipped.

The operator supports `ascend910b` (A2) and `ascend910_93` (A3). The A5 packed
cache path remains separate.

Accuracy and strided-cache coverage lives in:

```text
tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_indexer_prologue.py
```
