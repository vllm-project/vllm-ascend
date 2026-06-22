# Weight Loading / Remap Analysis

This document analyzes the current ability of vLLM Ascend to connect checkpoints through model registration, weight loading, naming remap, and TP/KV-related special handling.

## What it covers

This layer is about:

- how architecture entry points are recognized
- which naming-remap patterns are already covered
- which TP/KV/norm-shard cases already have handling
- how scale and offset metadata are connected in quantized checkpoints

## Current capability overview

Weight loading and remap are not centralized in one file. The logic is distributed across:

- model-specific rename or `load_weights` logic
- `quantization/modelslim_config.py`
- worker patches
- quant weight-loader helpers

## Key properties

- many common naming drifts are already handled explicitly
- weight loader behavior already reaches layer-level details such as embedding, lm_head, KV scales, and quant metadata
- TP and KV-replication edge cases are not treated as hypothetical; the repo already contains special handling for them
- fp8 and paired scale loading are already formal concerns in the design

## Related code

- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py)
- [models/deepseek_v4_mtp.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
- [patch/worker/patch_minimax_m2.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2.py)
