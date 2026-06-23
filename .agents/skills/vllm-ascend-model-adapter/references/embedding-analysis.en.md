# Embedding Analysis

This document analyzes the current `embedding`-layer capabilities in vLLM Ascend.

## What this layer covers

The embedding layer currently includes:

- vocab sharding and padding
- embedding tensor parallelism
- quantized embedding integration
- prompt-token to hidden-state aggregation
- shared vocab assumptions with `lm_head`

## Current capability overview

Ascend embedding support is primarily provided by:

- `VocabParallelEmbedding -> AscendVocabParallelEmbedding`
- `ParallelLMHead -> AscendParallelLMHead`
- `LogitsProcessor -> AscendLogitsProcessor`

Key entry points:

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:684)
- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:44)

## Key properties

- The implementation keeps the upstream vocab-sharding model.
- Embedding TP can be separated from default TP with its own communication group.
- Masked local lookup plus cross-rank reduction remains the core contract.
- Quantized embedding is supported, but the quant method must implement embedding-specific behavior.

## Current assumptions

- token embedding still follows a standard `VocabParallelEmbedding` style
- vocab can be cleanly partitioned by shard boundaries
- padded vocab handling remains valid

## Related code

- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py)
- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py)
- [distributed/parallel_state.py](/home/cmq/code/vllm-ascend/vllm_ascend/distributed/parallel_state.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
