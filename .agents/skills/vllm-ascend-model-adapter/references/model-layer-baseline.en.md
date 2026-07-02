# Model Layer Baseline

This document defines the current baseline for analyzing a model layer by layer.

## Goal

The layer-by-layer view should answer:

- what the key model components are
- whether each component already has reusable capability in vLLM or vllm-ascend
- whether the difference is structural, naming-related, or backend-related

## Current model types in scope

The current baseline only requires templates for:

1. `dense llm`
2. `moe llm`

## Why this matters

Many failures are not whole-model failures. They are often caused by one structural component, such as:

- embedding naming
- rope or position-id organization
- FFN variants
- MoE gate or expert structure
- norm sharding under TP or KV replication
- tied versus untied output heads

## Dense LLM template

```markdown
## Layer-by-Layer Compatibility Matrix

| Layer | Current capability | Model requirement | Gap | Adaptation plan |
| --- | --- | --- | --- | --- |
| embedding |  |  |  |  |
| positional/rope |  |  |  |  |
| attention |  |  |  |  |
| mlp/ffn |  |  |  |  |
| norm |  |  |  |  |
| lm_head/output |  |  |  |  |
```

## MoE LLM template

```markdown
## Layer-by-Layer Compatibility Matrix

| Layer | Current capability | Model requirement | Gap | Adaptation plan |
| --- | --- | --- | --- | --- |
| embedding |  |  |  |  |
| positional/rope |  |  |  |  |
| attention |  |  |  |  |
| moe router/gate |  |  |  |  |
| moe experts |  |  |  |  |
| shared expert / residual mlp | N/A or ... |  |  |  |
| norm |  |  |  |  |
| lm_head/output |  |  |  |  |
```
