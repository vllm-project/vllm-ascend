# Positional / RoPE Analysis

This document analyzes current positional-encoding and RoPE capabilities in vLLM Ascend.

## What it covers

The RoPE layer currently handles:

- choosing the concrete rotary implementation
- managing cos/sin cache state
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

See:

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:688)

## Key properties

- cos/sin cache management is explicit on Ascend
- ordinary GQA-style rope paths and MLA rope paths are separated
- partial rope is already treated as a real capability
- Triton and NPU-native rotary implementations coexist
- model runner already owns MRoPE / XD-RoPE position synchronization

## Current assumptions

- q/k rotation remains the main model
- cos/sin cache should be prepared and reused explicitly
- advanced rope variants usually require runner-level position handling

## Related code

- [ops/rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py)
- [ops/rope_dsv4.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rope_dsv4.py)
- [ops/triton/rope.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton/rope.py)
- [worker/model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
