# Operator Analysis

This document analyzes the current operator-layer capability in vLLM Ascend.

## Operator classes in current use

The repo currently mixes at least four operator sources:

1. native Torch ops
2. Triton kernels
3. `torch_npu` and `torch.ops.npu`
4. `torch.ops._C_ascend`

There is also a wrapper layer in `register_custom_ops.py` that exposes some NPU-specific behavior through `torch.ops.vllm`.

## Current capability overview

- `torch_npu` is the main execution base for hot paths such as RMSNorm, rotary embedding, quantized matmul, grouped matmul, fused infer attention, and MoE dispatch.
- Triton is used as a supplementary path for rope, RMSNorm, layernorm-gated behavior, and some utility kernels.
- `_C_ascend` is used for stronger fused operators such as fused norm, fused MoE routing, and fused quant-related behavior.
