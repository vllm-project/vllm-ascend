# Framework Integration Analysis

This document analyzes the current framework-layer integration capability in vLLM Ascend.

## What this layer covers

The framework layer is responsible for:

- making upstream vLLM runtime logic work on Ascend
- sinking model-layer behavior into worker, scheduler, sampler, and cache interfaces
- maintaining runtime contracts for graph mode, speculative decoding, KV cache, and distributed communication

## Current capability overview

Ascend framework integration is formed jointly by:

- `worker/model_runner_v1.py`
- `worker/v2/*`
- `core/*`
- `patch/platform/*`
- `patch/worker/*`

## Key properties

- The model runner is already a framework hub, not just a thin wrapper.
- KV-cache interfaces have Ascend-specific shape and metadata handling.
- Scheduler-side extensions already exist for dynamic batch and profiling-based chunking.
- Sampler and speculative decoding have explicit Ascend wiring.
- ACL graph support penetrates the framework stack rather than living only inside attention.

## Current assumptions

- replacing only model-layer ops is not enough on Ascend
- metadata, cache state, and graph params need runner-level coordination
- upstream drift often requires platform or worker patches

## Related code

- [worker/model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
- [worker/v2](/home/cmq/code/vllm-ascend/vllm_ascend/worker/v2)
- [core](/home/cmq/code/vllm-ascend/vllm_ascend/core)
- [patch/platform](/home/cmq/code/vllm-ascend/vllm_ascend/patch/platform)
- [patch/worker](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker)
