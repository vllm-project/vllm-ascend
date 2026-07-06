# Positional / RoPE Analysis

This document analyzes the current positional-encoding and RoPE capability in vLLM Ascend.

## What this layer covers

The RoPE layer currently handles:

- selecting the concrete rotary implementation
- managing cos and sin cache state
- partial rotary and interleaved rotary
- MRoPE, XD-RoPE, and MLA-related position paths
- synchronization with model-runner position state

## Current capability overview

Ascend currently registers several rotary-related implementations:

- `RotaryEmbedding -> AscendRotaryEmbedding`
- `MRotaryEmbedding -> AscendMRotaryEmbedding`
- `YaRNScalingRotaryEmbedding -> AscendYaRNRotaryEmbedding`
- `DeepseekScalingRotaryEmbedding -> AscendDeepseekScalingRotaryEmbedding`
- `ApplyRotaryEmb -> AscendApplyRotaryEmb`

## Key properties

- cos and sin cache management is explicit on Ascend
- ordinary rope paths and MLA rope paths are separated
- partial rope is already treated as a real capability
- Triton and NPU-native rotary implementations coexist
- the model runner already owns MRoPE and XD-RoPE position synchronization
