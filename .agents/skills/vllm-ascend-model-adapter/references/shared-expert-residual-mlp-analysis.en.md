# Shared Expert / Residual MLP Analysis

This document analyzes the current support for shared experts, residual MLP branches, and routed-expert side branches in vLLM Ascend.

## What it covers

The main questions are:

- whether shared experts exist as an explicit structure
- how shared experts combine with routed experts
- whether a residual MLP branch is parallel or serial
- whether these branches still reuse the ordinary MLP / MoE linear paths

## Current capability overview

The repo already contains explicit shared-expert logic, mainly in:

- `fused_moe.py`
- `linear_op.py`
- `quantization/modelslim_config.py`
- `xlite/xlite.py`

## Key properties

- shared experts already have their own execution stages
- the default shared-expert structure still follows gated MLP patterns
- some parallel-routing rules explicitly exclude `shared_expert` paths
- quantized shared experts are already considered in the execution design

## Current assumptions

- shared experts usually still look like gated MLPs
- they coexist with routed experts, but they do not share the exact same dispatch semantics

## Related code

- [ops/fused_moe/fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)
- [ops/linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
