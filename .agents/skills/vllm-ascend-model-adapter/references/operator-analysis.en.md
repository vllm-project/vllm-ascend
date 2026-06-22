# Operator Analysis

This document analyzes vLLM Ascend’s current implementation capabilities at the operator layer, focusing on the hierarchical relationships of Torch native, Triton, `torch_npu`, and custom `_C_ascend` ops, as well as the typical placement points of these operators in the current warehouse.

## 1. What problem does this layer solve?

The current main answers of the operator layer are:

- Which type of operator does a model layer eventually fall into?
- Which abilities depend on `torch_npu`
- Which abilities rely on Triton fallback
- Which paths depend on `_C_ascend` custom operator

## 2. Overview of current capabilities

The operator capabilities of the current warehouse are not from a single source, but include at least four layers:

1. Torch native op
2. Triton kernel
3. `torch_npu` / `torch.ops.npu`
4. `torch.ops._C_ascend`

There is another layer that registers several private ops into `torch.ops.vllm` through `register_custom_ops.py`.

See:

- [ops/register_custom_ops.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/register_custom_ops.py)

## 3. Key capabilities currently implemented

### 3.1 `torch_npu` is the main execution base

Currently many critical paths depend directly on `torch_npu`:

- `npu_rms_norm`
- `npu_add_rms_norm`
- `_npu_rotary_embedding`
- `npu_quant_matmul`
- `npu_grouped_matmul`
- `npu_fused_infer_attention_score`
- `npu_moe_distribute_dispatch/combine`

This shows that the current Ascend main capability is still based on the high-performance core provided by `torch_npu`.

### 3.2 Triton mainly plays the role of supplement and compatibility

Triton code currently in the repository includes:

- `rope.py`
- `rms_norm.py`
- `layernorm_gated.py`
- several sampling / utility kernels

See:

- [ops/triton](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton)

Triton's role in the current repository is more like:

- Provide high-performance/compatible implementation when conditions permit
- May be replaced or bypassed by patch on some platforms

Not all core capabilities have Triton as the first landing point.

### 3.3 `_C_ascend` Custom operator for stronger fusion

There are a lot of them in the current warehouse:

- `_C_ascend.npu_add_rms_norm_bias`
- `_C_ascend.dispatch_ffn_combine`
- `_C_ascend.moe_gating_top_k_hash`
- `_C_ascend.npu_rms_norm_dynamic_quant`

This type of operator usually states:

- The warehouse has moved beyond the stage of "only combining torch_npu base ops"
- Certain hotspot paths require stronger dedicated fusion

### 3.4 `torch.ops.vllm` is a custom bridging layer

Through `register_custom_ops.py`, the repository registers some NPU specializations into `torch.ops.vllm`, for example:

- `maybe_chunk_residual`
- `maybe_pad_and_reduce`
- `npu_rotary_embedding`

This way the upper-level model code can consume a more stable logical interface without having to rely directly on every `torch_npu` detail.

## 4. Current structural assumptions

The current operator layer implies these assumptions:

- Main hotspot paths are worth using `torch_npu` or `_C_ascend`
- Triton takes on more of a complementary role
- The upper model code is best accessed through stable wrapper / custom op
- Operator selection is often bound to dtype, layout, shape, and graph modes

## 5. Known boundaries and risks

The current main boundaries are:

- Triton is not stable and equivalent on all platforms and paths.
- `_C_ascend` paths are generally more sensitive to input signatures
- Operator problems often include layout/contiguous constraints at the same time, rather than simply "does this op exist?"
- Document analysis must be combined with HiAscend official constraints. It is not advisable to only look at the nominal support of local code.

## 6. What to look for when analyzing this layer

It is recommended to give priority to:

- Which type of Torch / Triton / `torch_npu` / `_C_ascend` is ultimately called by this layer?
- dtype/shape/layout/contiguous requirements
- Whether it can be reused in graph mode
- Whether there is already wrapper/custom op package

## 7. Related code

- [ops/register_custom_ops.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/register_custom_ops.py)
- [ops/triton](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton)
- [ops/layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py)
- [ops/rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py)
- [attention/attention_v1-analysis.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/attention-v1-analysis.md)
- [operator-compatibility-baseline.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/operator-compatibility-baseline.md)
