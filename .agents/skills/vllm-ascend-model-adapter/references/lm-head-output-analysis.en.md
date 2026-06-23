# LM Head / Output Analysis

This document analyzes the current `lm_head` and output-layer capabilities in vLLM Ascend.

## What this layer covers

The output layer currently handles:

- the final hidden-state to logits matmul
- logits gather and all-to-all behavior
- padded-vocab trimming
- quantized output-head execution
- shared vocab assumptions with the embedding layer

## Current capability overview

Ascend formally takes over:

- `ParallelLMHead -> AscendParallelLMHead`
- `LogitsProcessor -> AscendLogitsProcessor`

## Key properties

- The output head reuses the same vocab-sharding foundation as embedding.
- `lmhead_tensor_parallel_size` is a dedicated feature.
- Logits gather and padding removal are already part of the normal execution path.
- Quantized output heads run through `lm_head.quant_method.apply(...)`.

## Related code

- [ops/vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py)
- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py)
