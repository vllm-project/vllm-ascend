# Quantization Analysis

This document analyzes the current quantization capabilities in vLLM Ascend.

## What it covers

The quantization layer currently covers:

- checkpoint quant format handling
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
- linear quant paths are already mature for several static, dynamic, and MX-style formats
- MoE quantization is part of the formal execution stack
- C8 KV-cache quantization is a first-class attention capability
- compressed-tensors and ModelSlim-style formats are already recognized in multiple places

## Current assumptions

- layer type decides the quant-scheme family
- linear quant, MoE quant, and attention/KV quant should not be mixed together conceptually
- dynamic and static quantization are distinct execution contracts

## Related code

- [quantization/methods/__init__.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/methods/__init__.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
- [ops/cv_linear.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/cv_linear.py)
- [attention/attention_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/attention_v1.py)
