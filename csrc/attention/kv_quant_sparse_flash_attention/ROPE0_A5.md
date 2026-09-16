# A5 INT8 C8 with RoPE0

This extends the existing A5 `KvQuantSparseFlashAttention` implementation.
It uses NoPE512, per-tile INT8 quantization with tile size 128, and four FP32
scales. It does not use the FP8/group64 layouts of `MixedQuantSparseFlashMla`.

## Input contract

| Parameter | RoPE0 | Existing RoPE64 |
| --- | --- | --- |
| Query dtype | FP16 or BF16 | FP16 or BF16 |
| Query last dimension | 512 | 576 |
| Packed INT8 key width | 528 bytes | 656 bytes |
| Scale byte offset | 512 | 640 |
| Scale format | Four FP32 values | Four FP32 values |
| Attention output width | 512 | 512 |

Use `attention_mode=2`, `key_quant_mode=2`, `value_quant_mode=2`,
`quant_scale_repo_mode=1`, `tile_size=128`, and `sparse_block_size=1`.
`rope_head_dim=0` removes the RoPE payload from the external query and cache;
the caller does not need to supply a zero-filled RoPE tensor. Keep the
model-defined `scale_value` unchanged.

RoPE0 on A5 is currently restricted to INT8 key/value. Existing RoPE64 FP8 and
HiFloat8 dtype support is retained. The current A5 kernel writes attention only;
the A2/A3 softmax-max/sum output feature and batch consistency are outside this
extension's validation scope. Use `return_softmax_lse=False`.

## Implementation

The kernel reads compact GM rows using their actual dimensions. It retains the
original 672-byte UB row pitch, reads the scale at the actual payload offset,
and initializes the absent RoPE region on buffer reuse. Query preparation
initializes the internal NZ buffer before copying the 512 valid columns.
The 576-wide QK computation, 512-wide value/output computation, and existing
softmax path are retained.

The A5 source's shared-header paths and shared-parameter field names are aligned
with the dependencies already in this repository. The buffer utility now exposes
the explicit cross-core ID setter used by this kernel. These compatibility fixes
are needed to build the original RoPE64 path as well.

## Validation entry point

Build and install the A5 custom OPP and native extension from this checkout,
then run:

```bash
python -m pytest -v tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention_rope0_a5.py
```

The test suite checks compact versus explicit-zero RoPE equivalence, an
independent selected-value mean reference, the original nonzero-RoPE random
reference, multiple layouts and head counts, graph replay with changed KV,
and rejected shapes/quantization settings. Original reference tolerances are
preserved. Model integration and performance optimization are not part of this
operator change.
