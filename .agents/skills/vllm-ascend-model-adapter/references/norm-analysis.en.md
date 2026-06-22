# Norm Analysis

This document analyzes the current norm-layer capabilities in vLLM Ascend.

## What it covers

The norm layer currently includes:

- residual + norm fusion
- q/k norm fast paths
- quantization-aware norm bias
- gated norm and grouped-norm variants
- q_norm / k_norm / kv_norm coupling with rope and attention

## Current capability overview

The main registered classes include:

- `RMSNorm -> AscendRMSNorm`
- `GemmaRMSNorm -> AscendGemmaRMSNorm`
- `RMSNormGated -> AscendRMSNormGated`

See:

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:700)

## Key properties

- RMSNorm is the strongest default path on Ascend.
- Residual-plus-norm fusion is already treated as a first-class optimization.
- Quantized models may introduce explicit norm bias handling.
- q_norm / k_norm / kv_norm are already structural capabilities in several model paths.
- Some TP-aware q/k norm sharding behavior is already handled through dedicated patches.

## Current assumptions

- RMSNorm is the mainline path
- q/k/v-related norms can be part of the attention contract, not just the loader
- TP-aware normalization may need its own sharding rules

## Related code

- [ops/layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py)
- [ops/qwen2_decoder.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/qwen2_decoder.py)
- [attention/dsa_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/dsa_v1.py)
- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py)
