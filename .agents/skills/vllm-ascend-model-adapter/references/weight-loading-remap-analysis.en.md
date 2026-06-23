# Weight Loading / Remap Analysis

This document analyzes the current ability of vLLM Ascend to connect checkpoints through model registration, weight loading, naming remap, and TP or KV-related special handling.

## What this layer covers

This layer is about:

- how architecture entry points are recognized
- which naming-remap patterns are already covered
- which TP, KV, and norm-shard cases already have handling
- how scale and offset metadata are connected in quantized checkpoints

## Current capability overview

Weight loading and remap are not centralized in one file. The logic is distributed across:

- model-specific rename or `load_weights` logic
- `quantization/modelslim_config.py`
- worker patches
- quant weight-loader helpers

## Key properties

- many common naming drifts are already handled explicitly
- layer-level weight-loader behavior already exists for embedding, output heads, KV scales, and quant metadata
- TP and KV-replication edge cases are already treated as real loading concerns
- fp8 and paired-scale loading are formal concerns in the current design
