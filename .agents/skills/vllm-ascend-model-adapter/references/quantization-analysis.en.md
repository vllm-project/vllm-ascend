# Quantization Analysis

This document analyzes the current quantization capability in vLLM Ascend.

## What this layer covers

The quantization layer currently covers:

- checkpoint quant-format handling
- runtime quant-kernel execution
- scheme selection for linear, MoE, and attention layers
- KV-cache quantization and its coupling with the attention backend

## Current capability overview

The quantization implementation is already its own subsystem, centered around:

- `quantization/methods/`
- `quantization/modelslim_config.py`
- `ops/cv_linear.py`
- the C8 paths in `attention_v1.py`

## Key properties

- quant schemes exist for linear, MoE, and attention
- linear quant paths already cover multiple static, dynamic, and MX-style formats
- MoE quantization is part of the formal execution stack
- C8 KV-cache quantization is a first-class attention capability
