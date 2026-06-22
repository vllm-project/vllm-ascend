# Norm Analysis

This document analyzes vLLM Ascend's current implementation capabilities at the norm layer, focusing on RMSNorm, GemmaRMSNorm, gated norm, q/k/v related norms, and how these norms are coupled with quantization, rope, and TP shards.

## 1. What problem does this layer solve?

The norm layer currently not only "normalizes the hidden states", it also undertakes:

- residual + norm fusion
- Fast path to q/k norm
- Quantitative norm with bias
- gated norm / group norm variants
- Linkage between q_norm / k_norm / kv_norm and rope and attention

## 2. Overview of current capabilities

The main norm classes currently registered include:

- `RMSNorm -> AscendRMSNorm`
- `GemmaRMSNorm -> AscendGemmaRMSNorm`
- `RMSNormGated -> AscendRMSNormGated`

For the registration entrance, see:

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:700)

There are also several specialized paths:

- `AscendQwen2RMSNorm`
- `q_norm_without_weight` in DSA/MLA path
- TP-aware q/k norm patch for MiniMax M2

## 3. Key capabilities currently implemented

### 3.1 The current main norm is the NPU RMSNorm fast path

Main execution dependencies of `AscendRMSNorm.forward_oot()`:

- `torch_npu.npu_rms_norm`
- `torch_npu.npu_add_rms_norm`
- If custom op is enabled, go to `_C_ascend.npu_add_rms_norm_bias`

See:

- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:28)

This shows that the most mature norm path currently is RMSNorm, not universal LayerNorm.

### 3.2 norm bias has been incorporated into quantization compatible logic

When `AscendRMSNorm` is initialized, it will check whether `norm.bias` exists in the quant description. If so, create an additional bias parameter and set `weight_loader`.

See:

- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:42)

This shows that the current implementation has recognized that the norm in the quantitative model is not necessarily a pure weight-only structure, and may also contain bias.

### 3.3 residual + norm fusion is a formal capability

Currently both `AscendRMSNorm` and `AscendGemmaRMSNorm` are supported:

- No residual
- `x + residual -> norm`

See:

- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:63)
- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:91)

Therefore, the current norm layer does not only perform independent normalization, but has been built into the residual pipeline.

### 3.4 gated norm has been independently supported

`AscendRMSNormGated` is supported through `LayerNormFn` and `layer_norm_fwd_npu(...)`:

- `norm(x) * silu(z)`
- or `norm(x * silu(z))`

See:

- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:114)

This type of path is more used for new linear attention or gated blocks. It does not belong to the minimum configuration capabilities of traditional decoder-only LLM, but the warehouse has officially covered it.

### 3.5 q_norm / k_norm / kv_norm have deeply entered the attention path

In DeepSeek V4, DSA, Qwen3VL, MiniMax M2 and other implementations, `q_norm` / `k_norm` / `kv_norm` are no longer scraps, but part of the attention pre-structure.

Typical entrance:

- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:743)
- [ops/dsa.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/dsa.py:45)
- [attention/dsa_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/dsa_v1.py:1417)
- [patch/worker/patch_qwen3vl.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3vl.py)

It means that the current warehouse has recognized that q/k/v norm is a structure-level capability, not a simple loader remap.

### 3.6 q/k norm specialization under TP already exists

Explicitly addressed in the MiniMax M2 patch:

- `k_norm` weight segmentation
- TP-global rstd fix
- q/k norm NPU fast path

See:

- [patch/worker/patch_minimax_m2.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2.py)
- [patch/worker/patch_minimax_m2_linear_attn.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2_linear_attn.py)

This shows that the current implementation has realized that norm cannot always be processed as a normal weight shard under TP.

## 4. Current structural assumptions

The current norm layer implies these assumptions:

- RMSNorm is the mainstream path
- Residual+norm fusion deserves special optimization
- q/k norm, kv norm are formal components of attention in some models
- Some quantization paths will introduce norm bias
- q/k norm under TP may require independent shard/statistical correction

## 5. Known boundaries and risks

The current main boundaries are:

- General LayerNorm is not the current core optimization target
- The behavior of q/k norm is deeply coupled with head topology, TP, and KV replication
- There are many integrations between norm and quant/rope, and the problem may not only be with norm itself.
- Some paths depend on patches, which does not mean that they have been completely abstracted into general capabilities.

## 6. What to look for when analyzing this layer

It is recommended to give priority to:

- Whether the model uses RMSNorm or LayerNorm
- Whether `input_layernorm` / `post_attention_layernorm` exists
- Whether `q_norm` / `k_norm` / `kv_norm` exists
- Do these norms under TP require special shards?
- Whether quant description contains `norm.bias`

## 7. Related code

- [ops/layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py)
- [ops/qwen2_decoder.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/qwen2_decoder.py)
- [attention/dsa_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/dsa_v1.py)
- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py)
- [patch/worker/patch_minimax_m2.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2.py)
