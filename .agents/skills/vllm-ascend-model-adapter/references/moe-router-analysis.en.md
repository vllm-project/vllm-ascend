# MoE Router Analysis

This document analyzes the current MoE router and gate capabilities in vLLM Ascend.

## What it covers

The router layer currently answers:

- how router logits become `topk_ids / topk_weights`
- when fused gating can be used
- whether grouped top-k, hash routing, and correction bias are supported
- how router output aligns with token-dispatch contracts

## Current capability overview

The router entry point is:

- [experts_selector.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py)

The implementation first checks whether fused gating is legal, and otherwise falls back to a native path.

## Key properties

- fused gating is already a formal capability
- supported scoring styles include common `softmax`, `sigmoid`, and related variants
- grouped top-k and correction-bias behavior are already part of the design
- a dedicated hash-routing path exists for specialized models such as DeepSeek-style routes

## Current assumptions

- router output should still be reducible to `topk_ids + topk_weights`
- scoring functions are strongest when they match the existing fused support set
- router behavior is tightly coupled to downstream dispatch contracts

## Related code

- [ops/fused_moe/experts_selector.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py)
- [ops/fused_moe/fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)
