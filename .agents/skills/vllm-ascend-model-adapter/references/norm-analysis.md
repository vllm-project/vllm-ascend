# Norm Analysis

本文档分析 vLLM Ascend 当前在 norm 层的实现能力，聚焦 RMSNorm、GemmaRMSNorm、gated norm、q/k/v 相关 norm，以及这些 norm 与量化、rope、TP shard 的耦合方式。

## 1. 这一层解决什么问题

norm 层当前不只是“归一化一下 hidden states”，它还承担：

- residual + norm 融合
- q/k norm 的快速路径
- 带 bias 的量化 norm
- gated norm / group norm 变体
- q_norm / k_norm / kv_norm 与 rope、attention 的联动

## 2. 当前能力总览

当前已注册的主要 norm 类包括：

- `RMSNorm -> AscendRMSNorm`
- `GemmaRMSNorm -> AscendGemmaRMSNorm`
- `RMSNormGated -> AscendRMSNormGated`

注册入口见：

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:700)

同时还有若干特化路径：

- `AscendQwen2RMSNorm`
- DSA/MLA 路径里的 `q_norm_without_weight`
- MiniMax M2 的 TP-aware q/k norm patch

## 3. 当前实现的关键能力

### 3.1 当前主 norm 是 NPU RMSNorm 快路径

`AscendRMSNorm.forward_oot()` 的主执行依赖：

- `torch_npu.npu_rms_norm`
- `torch_npu.npu_add_rms_norm`
- 若启用 custom op，则走 `_C_ascend.npu_add_rms_norm_bias`

见：

- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:28)

这说明当前最成熟的 norm 路径是 RMSNorm，而不是通用 LayerNorm。

### 3.2 norm bias 已被纳入量化兼容逻辑

`AscendRMSNorm` 初始化时会检查 quant description 中是否存在 `norm.bias`，若有则额外创建 bias 参数并设置 `weight_loader`。

见：

- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:42)

这说明当前实现已经承认：量化模型里 norm 不一定是纯 weight-only 结构，可能还带 bias。

### 3.3 residual + norm 融合是正式能力

当前 `AscendRMSNorm` 和 `AscendGemmaRMSNorm` 都支持：

- 无 residual
- `x + residual -> norm`

见：

- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:63)
- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:91)

因此当前 norm 层并不是只做独立归一化，而是已经内建到 residual pipeline。

### 3.4 gated norm 已被独立支持

`AscendRMSNormGated` 通过 `LayerNormFn` 和 `layer_norm_fwd_npu(...)` 支持：

- `norm(x) * silu(z)`
- 或 `norm(x * silu(z))`

见：

- [layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py:114)

这类路径更多用于新型线性注意力或 gated block，不属于传统 decoder-only LLM 的最低配能力，但仓库已经正式覆盖。

### 3.5 q_norm / k_norm / kv_norm 已深度进入 attention 路径

在 DeepSeek V4、DSA、Qwen3VL、MiniMax M2 等实现里，`q_norm` / `k_norm` / `kv_norm` 已不是边角料，而是 attention 前置结构的一部分。

典型入口：

- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:743)
- [ops/dsa.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/dsa.py:45)
- [attention/dsa_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/dsa_v1.py:1417)
- [patch/worker/patch_qwen3vl.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3vl.py)

说明当前仓库已经承认 q/k/v norm 是结构级能力，而不是简单 loader remap。

### 3.6 TP 下的 q/k norm 特化已经存在

MiniMax M2 patch 中明确处理了：

- `k_norm` 权重切分
- TP-global rstd 修正
- q/k norm NPU fast path

见：

- [patch/worker/patch_minimax_m2.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2.py)
- [patch/worker/patch_minimax_m2_linear_attn.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2_linear_attn.py)

这说明当前实现已经意识到：norm 在 TP 下并不总能按普通 weight shard 处理。

## 4. 当前结构假设

当前 norm 层隐含这些假设：

- RMSNorm 是主流路径
- residual+norm 融合值得专门优化
- q/k norm、kv norm 在某些模型中是 attention 正式组成部分
- 某些量化路径会引入 norm bias
- TP 下的 q/k norm 可能需要独立 shard/统计修正

## 5. 已知边界与风险

当前主要边界有：

- 通用 LayerNorm 不是当前最核心优化对象
- q/k norm 的行为与 head 拓扑、TP、KV replication 耦合很深
- norm 与 quant/rope 的融合较多，问题未必只在 norm 自身
- 某些路径依赖 patch，不代表已经完全抽象成通用能力

## 6. 分析这一层时应该看什么

建议优先看：

- 模型是否使用 RMSNorm 还是 LayerNorm
- 是否存在 `input_layernorm` / `post_attention_layernorm`
- 是否存在 `q_norm` / `k_norm` / `kv_norm`
- TP 下这些 norm 是否需要特殊 shard
- quant description 是否带 `norm.bias`

## 7. 相关代码

- [ops/layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py)
- [ops/qwen2_decoder.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/qwen2_decoder.py)
- [attention/dsa_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/dsa_v1.py)
- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py)
- [patch/worker/patch_minimax_m2.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2.py)
