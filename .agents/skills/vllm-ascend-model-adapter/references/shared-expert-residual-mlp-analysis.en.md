# Shared Expert / Residual MLP Analysis

This document analyzes the current support for shared experts, residual MLP branches, and routed-expert side branches in vLLM Ascend.

## What this layer covers

The main questions are:

- whether shared experts exist as an explicit structure
- how shared experts combine with routed experts
- whether a residual MLP branch is parallel or serial
- whether these branches still reuse ordinary MLP or MoE linear paths

## Current capability overview

The repo already contains explicit shared-expert logic in:

- `fused_moe.py`
- `linear_op.py`
- `quantization/modelslim_config.py`
- `xlite/xlite.py`

## Key properties

- shared experts already have their own execution stages
- the default shared-expert structure still follows gated-MLP patterns
- some parallel-routing rules explicitly exclude shared-expert paths
- quantized shared experts are already part of the execution design
