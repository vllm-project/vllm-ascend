# Operator Analysis

This document analyzes the current operator-layer capability in vLLM Ascend.

## Operator classes in current use

The repo currently mixes at least four operator sources:

1. native Torch ops
2. Triton kernels
3. `torch_npu` / `torch.ops.npu`
4. `torch.ops._C_ascend`

There is also a wrapper layer in `register_custom_ops.py` that exposes some NPU-specific behavior through `torch.ops.vllm`.

## Current capability overview

The current implementation relies heavily on `torch_npu` for major hot paths such as:

- RMSNorm
- rotary embedding
- quantized matmul
- grouped matmul
- fused infer attention
- MoE dispatch/combine

Triton exists as a supplementary path for rope, RMSNorm, layernorm-gated kernels, and some utility kernels.

`_C_ascend` is used for stronger fused operators, such as fused norm, fused MoE routing, and fused quant-related paths.

## Current assumptions

- major performance-sensitive paths should use `torch_npu` or `_C_ascend`
- Triton is useful, but not the only or always-primary route
- upper layers should consume wrappers or custom ops when possible instead of raw low-level calls

## Related code

- [ops/register_custom_ops.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/register_custom_ops.py)
- [ops/triton](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton)
- [ops/layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py)
- [ops/rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py)
