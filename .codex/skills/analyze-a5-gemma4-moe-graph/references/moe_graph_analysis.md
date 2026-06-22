# A5 Gemma4 MoE 图模式精度问题分析

## 证据链

已观察到的现象：

- Gemma4 31B dense 图模式可以达到较高 GPQA 精度。
- Gemma4-A4B/MoE eager 模式正常。
- Gemma4-A4B/MoE graph 模式 GPQA 掉到约 57。
- 将 A5 Gemma4 运行阶段 MoE 通信从 MC2 改为 ALLGATHER 后，精度恢复到约 73。
- 如果在 `profile_run()` 阶段也强制 ALLGATHER，启动可能失败，原因是 profile dummy run 触发更重通信，导致 NPU 流资源不足。

推理：

- dense 31B 不执行 MoE expert routing 和 dispatch/combine，所以 dense 正常只能排除 attention/FIA/layer metadata 等共用 dense 路径。
- eager 正常说明 Gemma4 routing 数学、权重加载、基本 MoE 计算大体正确。
- ALLGATHER 恢复精度说明问题集中在 MC2/ALLTOALL MoE dispatch/combine 图模式路径，而不是 attention 或 Gemma4 custom routing 本身。

## Gemma4 MoE 为什么更敏感

Gemma4 MoE 使用 custom routing：

1. 从 raw router logits 选择 top-k experts。
2. 对所有 experts 做 softmax。
3. 只保留 top-k experts 的概率。
4. 对选中的概率重新归一化。
5. 将 `per_expert_scale` 融入 top-k weights。

最终 MoE 输出依赖这些信息必须完全对齐：

- `topk_ids`
- `topk_weights`
- token 到 expert 的 dispatch 顺序
- expert output 顺序
- combine/unpermute 元数据

即使 hidden state 只有轻微漂移，也可能让 GPQA 最后 `A/B/C/D` 的答案 token logits 翻转，所以表现会是“推理过程看起来对，最后字母错”。

## 疑似 MC2 故障面

不要简单表述成“MC2 数学算错”。更准确的范围是 A5 MC2 graph decode replay 路径：

```text
topk_ids/topk_weights
  -> npu_moe_distribute_dispatch 或 dispatch_v2
  -> grouped matmul
  -> npu_moe_distribute_combine 或 combine_v2
  -> final hidden states
```

MC2 携带多组动态元数据：

- `expert_token_nums`
- `ep_recv_counts`
- `tp_recv_counts`
- `assist_info_for_combine`
- `expand_scales`
- `x_active_mask`
- padded token mask 和 active token 数

图模式 decode 的 capture 往往使用完整 graph bucket，实际 replay 时可能只有部分真实 token，尾部是 padding。如果 MC2 dispatch/combine 在 replay 时没有完全重建或正确使用动态 routing metadata 和 `x_active_mask`，padded token 或陈旧 metadata 就可能影响真实 token 的 combine。

## 为什么 ALLGATHER 有效

ALLGATHER 走的是更简单、更稳定的 MoE 路径：

```text
npu_moe_init_routing
  -> grouped matmul
  -> npu_moe_token_unpermute
```

它绕开了 A5 MC2 的跨 rank dispatch/combine 动态元数据路径。模型整体仍然是 graph mode；改变的是 MoE 通信实现，不是回退 eager。

## 9222 做了什么，没有做什么

9222 风格的 Gemma4 支持主要处理：

- Gemma4 attention 图模式支持。
- Gemma4 fused MoE 兼容。
- `gelu_tanh` expert activation。
- Gemma4 custom routing 接入。

它没有实现 A5 专用的 `Gemma4 MoE -> ALLGATHER` 运行阶段 workaround。基础逻辑下，A5 小 batch decode 仍会选择 MC2。

A2/A3 上 9222 正常不能证明 A5 MC2 graph decode 安全，因为 A2/A3 可能使用不同的 MC2/fused-MC2 实现、通信策略和 capture/replay 行为。

## 给 maintainer 的推荐表述

推荐表述：

> We observed Gemma4-A4B graph-mode GPQA degradation on A5 while eager and dense Gemma4 graph were normal. The failing case selected MC2 for MoE decode. Forcing only Gemma4 MoE runtime decode to ALLGATHER restored accuracy while preserving graph mode. This points to an A5 MC2 MoE graph dispatch/combine replay issue, likely involving dynamic routing metadata or padded-token mask handling. The workaround is scoped to Gemma4 on A5 and avoids applying ALLGATHER during `profile_run` to prevent startup stream-resource failures.

不要说：

- attention 是根因。
- Gemma4 routing 数学错了。
- ALLGATHER 等于 eager。
- 已经证明具体某个 MC2 字段错了。

## 验证清单

稳定 workaround：

- 确认服务可以启动，`profile_run()` 不能强制 ALLGATHER。
- 确认 Gemma4 MoE 运行阶段 decode 使用 ALLGATHER。
- 确认没有设置 `cudagraph_mode=NONE`，没有启用 `enforce_eager`。
- 使用和 eager baseline 相同的配置跑 GPQA。
- 对比精度和基础吞吐。

保 MC2 性能实验：

- `actual_tokens == padded_tokens` 时允许 MC2。
- `actual_tokens < padded_tokens` 时使用 ALLGATHER。
- 如果精度保持高，说明 padded replay 强相关。
- 如果精度下降，说明 MC2 graph dispatch/combine 即使无 padding 也不安全。

字段级根因定位：

- dump 或 checksum 每层 `topk_ids/topk_weights`。
- 对比 dispatch 后 hidden checksum。
- 对比 grouped MLP 后 hidden checksum。
- 对比 combine 后 hidden checksum。
- 第一个出现差异的位置就是下一步根因范围。
