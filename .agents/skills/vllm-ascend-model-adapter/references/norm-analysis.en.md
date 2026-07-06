# Norm Analysis

This document analyzes the current norm-layer capability in vLLM Ascend.

## What this layer covers

The norm layer currently includes:

- residual plus norm fusion
- q/k norm fast paths
- quantization-aware norm bias
- gated norm and grouped-norm variants
- q_norm, k_norm, and kv_norm coupling with rope and attention

## Current capability overview

Main registered classes:

- `RMSNorm -> AscendRMSNorm`
- `GemmaRMSNorm -> AscendGemmaRMSNorm`
- `RMSNormGated -> AscendRMSNormGated`

## Key properties

- RMSNorm is the strongest default path on Ascend.
- Residual-plus-norm fusion is already a first-class optimization.
- Quantized models may introduce explicit norm-bias handling.
- q/k/v-related norm behavior is already structural in several model paths.
