# Shared Expert / Residual MLP Analysis

本文档分析 vLLM Ascend 当前在 `shared expert`、`residual MLP`、以及 routed experts 旁路分支上的实现能力。

## 1. 这一层解决什么问题

这类层的关键点不在“有没有 MoE”，而在：

- shared expert 是否独立存在
- shared expert 与 routed expert 如何组合
- residual MLP 是并联还是串联
- 这些分支是否仍复用普通 MLP / MoE 线性路径

## 2. 当前能力总览

当前仓库已经显式出现 shared expert 相关逻辑，尤其体现在：

- `fused_moe.py`
- `linear_op.py`
- `quantization/modelslim_config.py`
- `xlite/xlite.py`

这说明 shared expert 不是纯粹文档概念，现有实现已把它视作需要特殊处理的结构。

## 3. 当前实现的关键能力

### 3.1 shared expert 已有独立执行段

在 `fused_moe.py` 中可以看到：

- `_shared_experts_part1`
- `_shared_experts_part2`

并显式操作：

- `shared_experts.gate_up_proj`
- `shared_experts.down_proj`

见：

- [fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:527)

这说明当前 shared expert 已被当成单独分支，而不是 routed experts 的简单特例。

### 3.2 shared expert 默认仍沿用 gated MLP 结构

当前 shared expert 的主结构仍是：

- `gate_up_proj`
- 激活
- `down_proj`

因此 shared expert 当前能力更接近“普通 dense MLP 分支”，而不是完全特殊的新模块。

### 3.3 并行策略会显式避开 shared expert

`linear_op.py` 中 sequence parallel / MLP parallel 的若干前缀判断会显式跳过：

- `shared_expert`
- `shared_experts`

见：

- [linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py:639)
- [linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py:678)

这说明当前 shared expert 在并行策略上并不完全等价于普通 MLP，也不完全等价于 routed expert。

### 3.4 shared expert 已被纳入量化路径考虑

`fused_moe.py` 中已有：

- `has_quantized_shared`
- 对 shared gate_up/down 的 weight scale 处理

见：

- [fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:747)

因此 shared expert 当前不是“只能跑 BF16 的补充分支”，而是已进入量化兼容设计。

## 4. 当前结构假设

当前 shared expert / residual MLP 层隐含这些假设：

- shared expert 通常仍是 gated MLP
- 它与 routed experts 并存，但不是同一个 dispatch 语义
- shared expert 的并行规则可能需要独立处理
- 若存在 residual MLP，通常仍能复用 dense MLP 线性层与激活层

## 5. 已知边界与风险

当前主要边界有：

- 复杂 shared expert 组合方式未必都可直接落入当前实现
- residual MLP 的接线模式没有像 attention/MoE 那样被完全抽象成统一接口
- shared expert 问题经常和 router/dispatch/quant 一起出现

## 6. 分析这一层时应该看什么

建议优先看：

- 是否存在 `shared_experts`
- shared expert 是否仍为 `gate_up + down`
- shared expert 是否参与量化
- 并行策略是否刻意绕开 shared expert
- residual MLP 是否只是普通 dense 分支

## 7. 相关代码

- [ops/fused_moe/fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)
- [ops/linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
- [xlite/xlite.py](/home/cmq/code/vllm-ascend/vllm_ascend/xlite/xlite.py)
