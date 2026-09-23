# DeepSeek V4.1 CompressorV2

Vendored from [CANN ops-transformer PR #12647](https://gitcode.com/cann/ops-transformer/pull/12647),
commit `958f7a70271e5064c33068390326d4225c92e60d`.
The original CANN Open Software License Agreement Version 2.0 notices are retained.
The `op_host` and `op_kernel` sources are included in the normal vllm-ascend
custom-op build, without an external torch extension dependency.

`torch.ops._C_ascend.compressor_v2` fuses the KV and gate projections, channel-wise
softmax pooling, and in-place FP32 ring-state update. RMSNorm and RoPE are not fused.
The binding marks state mutation explicitly and supplies a Meta implementation.
The existing V4 `compressor` operator is unchanged.

## Model integration

V4.1 C2 directly uses AscendC on A2/A3; rebuilding custom operators is required.
There is no backend switch or Triton fallback. Both C2 projection weights are
loaded as BF16, matching the native input ABI. The kernel uses FP32 accumulation
and projection results; the ring state remains FP32. The BF16 pooling result,
existing RMSNorm, RoPE, and cache writers remain unchanged. C1 is unchanged.

Converting BF16 checkpoint weights to FP32 in the previous path did not add
weight precision. Nevertheless, the two matrix-multiply implementations need
numerical validation; bitwise equivalence is not assumed.

The adapter translates the compact native output to the original completion-token
rows, masks incomplete/padded rows, and passes explicit used lengths so padded
requests do not mutate the null state page. It performs no host tensor reads.
Supported model dimensions are H in [1024, 10240], aligned to 512, and D=128/512.
Although the vendored operator also includes arch35, the model integration is limited
to arch22 until the 32-row long-prefill ring contract is qualified on other hardware.

## Validation before merging

Run the operator tests on NPU:

```bash
pytest -v tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor_v2.py
pytest -v tests/ut/models/test_deepseek_v41_cache.py
```

Model-level eager/graph accuracy, chunked prefill, prefix-cache/speculative state
handling, and baseline-versus-candidate serving performance must also be checked.
Operator agreement against BF16 projections alone does not establish equivalence
to the old FP32 model path. No end-to-end accuracy or speedup is claimed here.
