# MoE Experts Analysis

This document analyzes the current MoE expert-layer capability in vLLM Ascend.

## What this layer covers

The expert layer focuses on:

- whether expert MLPs follow the standard two-stage form
- how `w13` or `gate_up` and `w2` or `down` are organized
- how grouped matmul is used
- how quantized experts are integrated

## Current capability overview

The current expert-execution center is:

- [moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py)

## Key properties

- The strongest path is still the standard routed-MoE expert MLP.
- `gate_proj + up_proj -> w13` is a recurring existing convention.
- Grouped matmul is the main execution model for experts.
- Quantized expert execution is already part of the formal execution stack.
