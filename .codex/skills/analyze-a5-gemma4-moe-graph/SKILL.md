---
name: analyze-a5-gemma4-moe-graph
description: 诊断、解释或整理 vllm-ascend 中 A5/Ascend950 Gemma4 MoE 图模式精度问题和 moe_graph 修复方案。适用于讨论 Gemma4-A4B/MoE 图模式 GPQA 掉点、MC2 与 ALLGATHER MoE 通信差异、profile_run 启动失败、padded decode graph replay、以及给 maintainer 准备修复过程和方案劣势说明。
---

# 分析 A5 Gemma4 MoE 图模式问题

## 核心结论

把 `moe_graph` 方案定义为“正确性优先的规避方案”，不要说成已经修复了 MC2 的字段级根因。

除非后续有新的字段级证据，否则推荐这样表述：

> A5 上 Gemma4 MoE 图模式 decode 在 MC2/ALLTOALL MoE dispatch/combine 路径存在精度退化。将 Gemma4 MoE 运行阶段的通信方式切到 ALLGATHER 后，可以绕开不稳定的 MC2 graph replay 路径并恢复精度，同时整体仍然保持图模式。当前尚未证明具体是哪一个 MC2 动态元数据字段错误。

## 分析流程

1. 先区分 dense Gemma4 和 MoE Gemma4。
   - Gemma4 31B dense 不经过 `select_experts`、MC2、ALLGATHER MoE dispatch、`npu_moe_distribute_dispatch/combine`、`npu_moe_token_unpermute`。
   - dense 图模式正常只能说明 attention/FIA/layer index 等通用路径大体没问题，不能证明 MoE 路径没问题。

2. 对比 MoE 模型 eager 和 graph。
   - eager 正常、graph 掉点，优先怀疑 graph-only 的 MoE routing/dispatch/combine。
   - GPQA 掉到约 57，且推理过程经常合理但最终答案字母错，符合 hidden state/logits 轻微漂移，而不是模型完全不会推理。

3. 检查实际选择的 MoE 通信方式。
   - A5 小 batch decode 通常会选 MC2。
   - 高风险组合是 `A5 + Gemma4 MoE + graph decode + MC2/ALLTOALL + padding/dynamic routing`。

4. 用 ALLGATHER 做隔离实验。
   - 如果 Gemma4 MoE graph 切到 ALLGATHER 后精度恢复，问题域就在 MC2/ALLTOALL MoE dispatch/combine，而不是 Gemma4 routing 数学本身或 attention。

5. 避免过度声称根因。
   - 在没有字段级 dump 证明前，只说“MC2 graph dispatch/combine 动态元数据或 padded-token mask replay 问题”。

## moe_graph 修复模式

稳定版 `moe_graph` 方案是在 MoE 通信选择处做窄范围规避：

- 通过 `hf_config.model_type`、`hf_text_config.model_type` 或 `architectures` 识别 Gemma4。
- A5 Gemma4 MoE 非 profile 运行阶段选择 `MoECommType.ALLGATHER`。
- `profile_run` 阶段保持 A5 默认路径，避免 ALLGATHER 增加通信和流资源压力导致启动失败。
- 不设置 `enforce_eager=True`，不关闭 graph mode。这仍然是图模式，只是 MoE 通信实现从 MC2 换成 ALLGATHER。

需要查看代码形态时，优先看：

- `vllm_ascend/ascend_forward_context.py`
- `vllm_ascend/platform.py`
- `tests/ut/test_ascend_forward_context.py`

## 当前方案劣势

必须主动说明这些限制：

- 性能可能下降，因为 ALLGATHER 通信比 MC2 更重。
- 这是绕开 MC2 graph 路径的 workaround，不是修复 MC2 本身。
- 虽然按 A5 + Gemma4 收窄，避免影响其它 MoE 模型，但如果没有 graph-only 条件，Gemma4 eager 的通信策略也可能被改变。
- batch size 16 可能比 batch size 8 更慢，因为 ALLGATHER 通信、专家负载不均、graph bucket 变大、长输出 batch 拖尾都可能主导性能。
- 当前没有解释到具体哪一个 MC2 字段出错，需要额外 instrumentation 才能证明字段级根因。

## 如果希望继续保留 MC2 性能

建议单独做实验分支，不要直接替代稳定 workaround：

- `actual_tokens == padded_tokens` 时允许 MC2。
- 出现 graph padding 时使用 ALLGATHER。
- 如果精度仍然高且性能提升，说明 padded-token MC2 replay 强相关。
- 如果精度再次下降，说明 MC2 graph dispatch/combine 即使无 padding 也不可靠，需要继续做字段级调试。

详细证据链、maintainer 表述和验证清单见 `references/moe_graph_analysis.md`。
