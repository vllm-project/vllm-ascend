# MLP / FFN Analysis

This document analyzes the current dense MLP and FFN capability in vLLM Ascend.

## What this layer covers

The FFN layer currently includes:

- plain versus gated FFN structure
- `gate_up_proj` and `down_proj` mapping into Ascend custom linear paths
- dedicated MLP tensor parallelism
- quantized linear reuse
- weight prefetch and execution-side optimization

## Current capability overview

The main FFN support is assembled through:

- custom linear ops
- parallel-group routing
- activation-side helpers
- weight-prefetch logic

## Key properties

- The strongest default path is still `gate_up_proj + down_proj`.
- `mlp_tensor_parallel_size` is already a first-class feature.
- `gate_up_proj` usually maps to column-parallel handling.
- `down_proj` usually maps to row-parallel handling.
- Sequence parallel already integrates with common FFN prefixes.

## Related code

- [ops/linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py)
- [ops/activation.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/activation.py)
- [ops/weight_prefetch.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/weight_prefetch.py)
