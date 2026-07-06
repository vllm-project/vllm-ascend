# MoE Router Analysis

This document analyzes the current MoE router and gate capability in vLLM Ascend.

## What this layer covers

The router layer currently answers:

- how router logits become `topk_ids` and `topk_weights`
- when fused gating can be used
- whether grouped top-k, hash routing, and correction bias are supported
- how router output aligns with token-dispatch contracts

## Current capability overview

The router entry point is:

- [experts_selector.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py)

## Key properties

- fused gating is already a formal capability
- several scoring styles are already supported
- grouped top-k and correction bias are part of the current design
- a dedicated hash-routing path exists for specialized models
