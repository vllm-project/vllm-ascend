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

## Experimental model integration

After rebuilding custom operators, enable the A2/A3 V4.1 C2 path using:

```bash
--additional-config '{"use_ascendc_compressor": true}'
```

This is opt-in because native projections require BF16 weights and activations,
whereas the existing model path uses FP32 projection weights and computation.
Native mode loads C2 projection weights as BF16; the FP32 ring layout, BF16
pooling result, existing RMSNorm, RoPE, and cache writers remain unchanged.
Disabling the option retains the existing FP32 projection and Triton pooling path.
C1 is unchanged. Changing this option requires reloading the model.

The adapter translates the compact native output to the original completion-token
rows, masks incomplete/padded rows, and passes explicit used lengths so padded
requests do not mutate the null state page. It performs no host tensor reads.
Supported model dimensions are H in [1024, 10240], aligned to 512, and D=128/512.
Although the vendored operator also includes arch35, the model option is limited
to arch22 until the 32-row long-prefill ring contract is qualified on other hardware.

## Validation before enabling by default

Run the operator tests on NPU:

```bash
pytest -v tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor_v2.py
pytest -v tests/ut/models/test_deepseek_v41_cache.py
```

Model-level eager/graph accuracy, chunked prefill, prefix-cache/speculative state
handling, and baseline-versus-candidate serving performance must also be checked.
Operator agreement against BF16 projections alone does not establish equivalence
to the old FP32 model path. No end-to-end accuracy or speedup is claimed here.
