# MoE Router Analysis

本文档分析 vLLM Ascend 当前在 MoE `router / gate` 层的实现能力。这里聚焦的是“router 结构与选择逻辑”，不是完整 MoE 流水线；完整执行管线请结合 `moe-fused-analysis.md` 阅读。

## 1. 这一层解决什么问题

router 层当前主要解决：

- router logits 如何变成 `topk_ids / topk_weights`
- 是否使用 fused gating op
- grouped top-k、hash routing、bias correction 是否被支持
- router 输出如何与 token dispatch 契约对齐

## 2. 当前能力总览

当前 router 主入口在：

- [experts_selector.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py)

核心特征是：

- 先判断能否走 fused gating
- 否则回退 native 路径
- router 输出会直接进入 token dispatch / combine 契约

因此当前 router 不是“单一 top-k 算子”，而是一套带分流条件的选择层。

## 3. 当前实现的关键能力

### 3.1 fused gating 已是正式能力

当前实现支持通过 fused op 选择专家，而不是只靠 Python/torch fallback。

主要路径包括：

- `DeviceOperator.moe_gating_top_k(...)`
- `torch.ops._C_ascend.moe_gating_top_k_hash(...)`

这说明 Ascend 侧已经把 router 视作性能敏感层，而不只是上游模型逻辑。

### 3.2 scoring function 已有明确支持集合

从现有实现和已有分析可见，当前 fused router 已覆盖：

- `softmax`
- `sigmoid`
- `sqrtsoftplus`

并支持一部分额外行为：

- grouped top-k
- `renormalize`
- `e_score_correction_bias`
- `routed_scaling_factor`

因此当前 router 能力不是只会“标准 Mixtral top-k”，而是已经覆盖多种门控风格。

### 3.3 hash routing 已有专项支持

当前代码中存在：

- 基于 `tid2eid + input_ids` 的 hash routing 特化路径

这说明当前仓库已经吸收了 DeepSeek V4 一类路由语义，而不是只支持朴素 router weight matmul。

### 3.4 shared expert 会影响 router 输出组织

当前实现中 `mix_placement` 会对 shared expert 的 expert id / weight 做补齐处理。

这意味着当前 router 输出并不是对 routed experts 唯一负责，它还要与 shared expert 接线方式兼容。

## 4. 当前结构假设

当前 router 层隐含的结构假设包括：

- router 最终仍应归约为 `topk_ids + topk_weights`
- scoring function 数量有限，最好命中 fused 支持集合
- router 输出要与后续 dispatch 契约严格匹配
- grouped top-k 与 correction bias 可以是正式结构，而不是异常特例

## 5. 已知边界与风险

当前 router 层的主要边界有：

- `custom_routing_function` 会破坏 fused path 的普适性
- 某些 scoring + renormalize 组合并非全部支持
- hash routing 虽已支持，但更依赖特定输入契约
- router 问题经常表面上像“MoE 不支持”，本质却只是 gate 规则不在当前集合里

## 6. 分析这一层时应该看什么

建议优先看：

- `top_k`
- `scoring_func`
- `renormalize`
- `grouped_topk`
- `e_score_correction_bias`
- 是否存在 `custom_routing_function`
- 是否依赖 `input_ids` hash

## 7. 相关代码

- [ops/fused_moe/experts_selector.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py)
- [ops/fused_moe/fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)
- [moe-fused-analysis.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/moe-fused-analysis.md)
