# vLLM Ascend `attention_v1.py` Analysis

This document summarizes the current attention capability implemented in `vllm_ascend/attention/attention_v1.py` and its interaction with `vllm_ascend/worker/model_runner_v1.py`.

## Scope

The analysis covers:

- backend registration and overall structure
- how `model_runner_v1` builds common attention metadata
- how `AscendAttentionMetadataBuilder` turns common metadata into per-layer metadata
- how `AscendAttentionBackendImpl` dispatches to different Ascend operator paths
- specialized handling for masks, paged KV, graph capture, shared KV cache, KV compression, and C8 KV quantization

## Core conclusion

`attention_v1.py` is primarily a dispatch and normalization layer rather than a standalone attention algorithm implementation. Its main responsibilities are:

1. Convert vLLM request and batch state into shapes, layouts, and metadata that Ascend kernels can consume.
2. Select the best execution path based on runtime state and model traits.
3. Reuse paged KV cache when possible and avoid unnecessary dense materialization.
4. Provide fallbacks when a kernel path is unavailable or constrained.

## Main state machine

The implementation is organized around five attention states:

- `PrefillNoCache`
- `PrefillCacheHit`
- `DecodeOnly`
- `ChunkedPrefill`
- `SpecDecoding`

These states drive both metadata construction and operator dispatch.

## Metadata model

The current path is:

1. `NPUModelRunner` builds `AscendCommonAttentionMetadata`.
2. `AscendAttentionMetadataBuilder` produces per-layer `AscendMetadata`.
3. The backend consumes that metadata to choose an operator path.

Important metadata fields include:

- attention mask
- state and token counters
- query and KV sequence lengths
- `block_table`
- `slot_mapping`
- graph and KV-compression related flags

## Main execution paths

The current backend dispatches among several important paths:

- paged decode attention
- FIA-based prompt and decode paths
- sink-token specific FIA v2 path
- dense fallback through `npu_fusion_attention`
- pooling or encoder-style attention

The choice depends on:

- attention state
- sliding-window behavior
- sink-token behavior
- whether paged attention is available
- whether graph mode is active
- whether the KV cache is shared or quantized

## Graph and cache support

The current implementation includes formal support for:

- paged KV cache reads and writes
- shared KV cache reuse
- ACL graph capture and replay
- specialized graph branches for decode, chunked prefill, and speculative decoding

Graph-mode support is tightly coupled to metadata layout and runtime shape constraints.

## C8 KV quantization

`attention_v1.py` already includes a dedicated C8 INT8 KV path. The main behaviors are:

- preparing per-channel anti-quant tensors
- native paged INT8 decode
- mixed chunked-prefill handling
- gather-and-dequant fallback when needed

This is a first-class backend capability rather than an afterthought.

## Current boundaries

The main constraints worth remembering are:

- metadata structure is rich and somewhat redundant
- some paths still require dense fallback
- graph enablement is shape- and mode-sensitive
- speculative decoding partially reuses chunked-prefill semantics
- paged attention is only enabled under a narrow set of conditions

## Related code

- [attention_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/attention_v1.py)
- [model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
- [attention_mask.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/attention_mask.py)
- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/utils.py)
- [fa3_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/fa3_v1.py)
- [mla_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/mla_v1.py)
- [sfa_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/sfa_v1.py)
