# MoE Experts Analysis

本文档分析 vLLM Ascend 当前在 MoE `experts` 层的实现能力，聚焦 expert MLP 的结构假设、权重布局、grouped matmul 路径、量化 expert 路径，以及与 dispatch/combine 的边界。

## 1. 这一层解决什么问题

experts 层当前主要解决：

- expert 内部是不是标准两段式 MLP
- `w13/gate_up` 与 `w2/down` 如何组织
- grouped matmul 如何执行
- quantized experts 如何接入

## 2. 当前能力总览

当前 expert 计算中枢在：

- [moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py)

关键特征是：

- unquant 路径与 quant 路径并存
- 第一段通常是 `gate_up`
- 中间通常是 `swiglu`
- 第二段通常是 `down`

因此当前专家层最成熟的能力模型，是标准 routed MoE LLM 常见的 expert MLP。

## 3. 当前实现的关键能力

### 3.1 当前默认 expert 是两段式 MLP

已有实现明确围绕：

- `w13_weight` / `gate_up`
- `w2_weight` / `down`

组织 expert 权重与计算顺序。

`moe_mlp.py` 中可以直接看到：

- grouped matmul for `gate_up`
- `swiglu`
- grouped matmul for `down`

见：

- [moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:353)

### 3.2 `gate_proj + up_proj -> w13` 已是既有约定

无论是量化映射还是 expert 计算，当前仓库都反复出现下面这个约定：

- `gate_proj`
- `up_proj`
- `down_proj`
- 在 MoE 执行前将前两者打包成 `w13`

这在 `modelslim_config.py`、`w4a8.py`、`moe_mlp.py` 中都能看到。

因此当前 expert 计算的强能力区，是能落到 `w13 + w2` 布局的模型。

### 3.3 grouped matmul 是 expert 主执行方式

当前 unquant expert 主路径大量依赖：

- `torch_npu.npu_grouped_matmul`

见：

- [moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:371)

这意味着当前 experts 层更像“批量专家并行 matmul 管线”，而不是逐 expert for-loop。

### 3.4 quantized experts 已是正式能力

当前 quantized MoE expert 并不是空白，仓库已注册多类 quantized fused-MoE 方法，例如：

- `AscendW8A8DynamicFusedMoEMethod`
- `AscendW4A8DynamicFusedMoEMethod`
- `AscendW4A16FusedMoEMethod`
- MXFP / FP8 变体

见：

- [quantization/methods/__init__.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/methods/__init__.py:45)

说明当前 expert 层已经不只是 BF16/FP16 的功能路径，也承担了量化专家的正式执行职责。

## 4. 当前结构假设

当前 experts 层隐含这些假设：

- expert MLP 可规约为 `gate_up -> activation -> down`
- 权重布局最好能落到 `w13 + w2`
- 多 expert 计算应尽量通过 grouped matmul 融合
- quantized expert 仍应遵循大体相同的结构契约

## 5. 已知边界与风险

当前主要边界有：

- 多分支 expert、卷积 expert、残差 expert 并不天然落入当前主路径
- expert 如果不是两段式 MLP，现有实现复用价值会明显下降
- quantized expert 的加载与执行耦合较强
- experts 层的问题常与 shared expert、router、dispatch 一起出现，不能孤立看

## 6. 分析这一层时应该看什么

建议优先看：

- expert 内部是否是 `gate_proj/up_proj/down_proj`
- checkpoint 是否能规约到 `w13/w2`
- 是否依赖 grouped matmul
- 是否需要量化 expert 路径

## 7. 相关代码

- [ops/fused_moe/moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py)
- [ops/fused_moe/fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)
- [quantization/methods/__init__.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/methods/__init__.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
