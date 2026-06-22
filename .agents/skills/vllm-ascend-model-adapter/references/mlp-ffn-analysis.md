# MLP / FFN Analysis

本文档分析 vLLM Ascend 当前在 dense MLP / FFN 层的实现能力，聚焦常见 `gate_up_proj + down_proj` 结构、并行组选择、量化线性层、以及与 weight prefetch / sequence parallel 的关系。

## 1. 这一层解决什么问题

当前 MLP/FFN 层主要涉及：

- FFN 结构是普通两层、gated 还是 fused 变体
- `gate_up_proj` / `down_proj` 如何映射到 Ascend custom linear
- MLP TP 如何与默认 TP 解耦
- 量化 linear 是否能直接复用
- MLP 权重预取与图模式性能优化

## 2. 当前能力总览

当前 Ascend 并没有单独维护一套“通用 MLP 模块”，而是通过：

- custom linear op
- 并行组调度
- 激活层特化
- weight prefetch

来接管大多数 LLM 的 MLP 执行。

关键入口主要在：

- [ops/linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py)
- [ops/activation.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/activation.py)
- [ops/weight_prefetch.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/weight_prefetch.py)

## 3. 当前实现的关键能力

### 3.1 当前默认围绕 `gate_up_proj + down_proj` 组织

无论是 `linear_op.py` 的前缀判断，还是 `weight_prefetch.py` 的模块预取，当前都把下面这组命名当成主路径：

- `gate_up_proj`
- `down_proj`

见：

- [linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py:633)
- [weight_prefetch.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/weight_prefetch.py:134)

这意味着当前 dense MLP 能力最强覆盖的是典型 SwiGLU/gated MLP 风格，而不是任意自由形态的 FFN。

### 3.2 MLP TP 已是显式能力

当前实现支持 `mlp_tensor_parallel_size`，并将其与默认 TP 组分离：

- `mlp_tp_enable()`
- `get_mlp_tp_group()`
- `MLPColumnParallelOp`
- `MLPRowParallelOp`

见：

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:835)
- [linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py:633)

当前行为大致是：

- `gate_up_proj` 走 column-parallel
- `down_proj` 走 row-parallel

这与主流 gated MLP 的 TP 切分方向是一致的。

### 3.3 Sequence Parallel 与 MLP 路径已经对接

在启用 SP 时，当前实现会把这些前缀视为 sequence-parallel 候选：

- `gate_up_proj`
- `down_proj`
- `in_proj`
- `conv1d`

见：

- [linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py:638)
- [linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py:677)

说明当前 FFN 路径已经不仅支持 TP，还支持更细粒度的序列并行接线。

### 3.4 激活层与 MLP 是联动设计

当前 MLP 的激活阶段不是被完全忽略的，`activation.py` 中会配合：

- `maybe_prefetch_mlp_weight_preprocess`
- `maybe_prefetch_mlp_weight_postprocess`

见：

- [activation.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/activation.py)

这说明当前实现把 MLP 看作一个两阶段流水：

1. `gate_up` 阶段
2. 激活后 `down` 阶段

并围绕这个流水做预取和算子选择。

### 3.5 Qwen3-Next / GDN 一类非标准 FFN 已有局部适配

当前仓库并不只支持最普通的 Llama/Qwen 风格 FFN，还显式存在：

- `in_proj`
- `out_proj`
- `conv1d`
- `GatedDeltaNetAttention`

这些入口意味着当前实现已经承认“并不是所有 FFN 都叫 `mlp.gate_up_proj/down_proj`”。

相关文件：

- [ops/gdn.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/gdn.py)
- [patch/worker/patch_qwen3_5.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py)

## 4. 当前结构假设

当前 dense MLP 层隐含这些假设：

- 主流 LLM 的 FFN 大多可规约成 column-parallel 第一层 + row-parallel 第二层
- `gate_up_proj` / `down_proj` 是最重要的稳定命名
- 激活大多插在两层线性之间
- 非 MoE MLP 与 MoE expert MLP 应分开看
- shared expert 在并行策略上不能简单等同于普通 MLP

## 5. 已知边界与风险

当前明显的边界有：

- 这套能力最强覆盖的是 gated FFN，不是任意异构 FFN
- 前缀判断仍然很重要，命名漂移会直接影响并行策略
- `shared_expert` 路径会被显式排除在某些 SP/TP 规则之外
- 一些非标准 FFN 需要 patch 或单独 op，而不是自动落入通用路径

## 6. 分析这一层时应该看什么

建议优先看：

- FFN 是否为 gated / SwiGLU / GeGLU
- 是否使用 `gate_up_proj` / `down_proj`
- 是否命名为 `in_proj/out_proj`
- 是否启用 `mlp_tensor_parallel_size`
- 是否存在 weight prefetch / quantized linear / sequence parallel

## 7. 相关代码

- [ops/linear_op.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/linear_op.py)
- [ops/activation.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/activation.py)
- [ops/weight_prefetch.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/weight_prefetch.py)
- [ops/gdn.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/gdn.py)
- [patch/worker/patch_qwen3_5.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py)
