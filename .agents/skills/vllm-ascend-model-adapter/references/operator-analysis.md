# Operator Analysis

本文档分析 vLLM Ascend 当前在算子层的实现能力，聚焦 Torch native、Triton、`torch_npu`、自定义 `_C_ascend` op 的分层关系，以及这些算子在当前仓库中的典型落点。

## 1. 这一层解决什么问题

算子层当前主要回答：

- 一个模型层最终落到哪类算子
- 哪些能力依赖 `torch_npu`
- 哪些能力依赖 Triton fallback
- 哪些路径依赖 `_C_ascend` 自定义算子

## 2. 当前能力总览

当前仓库的算子能力并不是单一来源，而是至少包含四层：

1. Torch native op
2. Triton kernel
3. `torch_npu` / `torch.ops.npu`
4. `torch.ops._C_ascend`

另有一层通过 `register_custom_ops.py` 把若干私有 op 注册进 `torch.ops.vllm`。

见：

- [ops/register_custom_ops.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/register_custom_ops.py)

## 3. 当前实现的关键能力

### 3.1 `torch_npu` 是主执行底座

当前很多关键路径直接依赖 `torch_npu`：

- `npu_rms_norm`
- `npu_add_rms_norm`
- `_npu_rotary_embedding`
- `npu_quant_matmul`
- `npu_grouped_matmul`
- `npu_fused_infer_attention_score`
- `npu_moe_distribute_dispatch/combine`

这说明当前 Ascend 主能力仍然建立在 `torch_npu` 提供的高性能内核上。

### 3.2 Triton 主要承担补充和兼容角色

当前仓库中的 Triton 代码包括：

- `rope.py`
- `rms_norm.py`
- `layernorm_gated.py`
- 若干 sampling / utility kernel

见：

- [ops/triton](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton)

Triton 在当前仓库中的角色更像：

- 有条件时提供高性能/兼容实现
- 在部分平台上可能被 patch 替换或绕开

不是所有核心能力都以 Triton 为第一落点。

### 3.3 `_C_ascend` 自定义算子用于更强融合

当前仓库里大量出现：

- `_C_ascend.npu_add_rms_norm_bias`
- `_C_ascend.dispatch_ffn_combine`
- `_C_ascend.moe_gating_top_k_hash`
- `_C_ascend.npu_rms_norm_dynamic_quant`

这类算子通常说明：

- 仓库已经超出“只组合 torch_npu 基础 op”的阶段
- 某些热点路径需要更强的专用融合

### 3.4 `torch.ops.vllm` 是自定义桥接层

通过 `register_custom_ops.py`，仓库把一些 NPU 特化实现注册进 `torch.ops.vllm`，例如：

- `maybe_chunk_residual`
- `maybe_pad_and_reduce`
- `npu_rotary_embedding`

这样上层模型代码可以消费更稳定的逻辑接口，而不必直接依赖每个 `torch_npu` 细节。

## 4. 当前结构假设

当前算子层隐含这些假设：

- 主力热点路径值得使用 `torch_npu` 或 `_C_ascend`
- Triton 更多承担补充角色
- 上层模型代码最好通过稳定 wrapper / custom op 接入
- 算子选择常与 dtype、layout、shape、graph 模式绑定

## 5. 已知边界与风险

当前主要边界有：

- Triton 并非所有平台、所有路径都稳定等价
- `_C_ascend` 路径通常对输入签名更敏感
- 算子问题往往同时包含 layout/contiguous 约束，而不是单纯“有没有这个 op”
- 文档分析时必须结合 HiAscend 官方约束，不宜只看本地代码名义支持

## 6. 分析这一层时应该看什么

建议优先看：

- 该层最终调用的是 Torch / Triton / `torch_npu` / `_C_ascend` 哪一类
- dtype / shape / layout / contiguous 要求
- 图模式下是否能复用
- 是否已有 wrapper/custom op 封装

## 7. 相关代码

- [ops/register_custom_ops.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/register_custom_ops.py)
- [ops/triton](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton)
- [ops/layernorm.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/layernorm.py)
- [ops/rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py)
- [attention/attention_v1-analysis.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/attention-v1-analysis.md)
- [operator-compatibility-baseline.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/operator-compatibility-baseline.md)
