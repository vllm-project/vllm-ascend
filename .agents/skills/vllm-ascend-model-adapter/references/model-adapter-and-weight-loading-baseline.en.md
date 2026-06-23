# Model Adapter And Weight Loading Baseline

This document defines the current capability baseline for model registration, adapter wiring, weight remapping, and weight loading.

## What this layer answers

It focuses on:

- whether vLLM already recognizes the architecture
- whether an existing adapter can be reused
- whether checkpoint keys can be mapped into the current implementation
- whether TP, KV heads, norm, rope, and fp8 scales are already covered by loading rules

## Current baseline

The current baseline assumes:

- architecture entry is identified from `config.json`
- architecture registration is checked in `vllm/model_executor/models/registry.py`
- model adapters live in `vllm`
- processors are added only when necessary
- key naming differences are resolved through explicit remap rules

## Frequent boundaries

Common high-frequency concerns in this layer include:

- missing architecture registration
- remote code incompatible with the native vLLM adapter
- naming mismatch for qkv, o_proj, gate, or MoE layers
- q_norm, k_norm, kv_norm, or rope loading mismatch
- TP or KV-head replication changing sharding behavior
- fp8 checkpoints that require paired weight and scale loading
