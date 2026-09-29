# KV Trace 实机验证摘要（2026-09-29）

结论：生命周期和 P/D 请求关联已在真实 NPU 推理中验证；当前版本仍不足以完成参考 PPT 的逐 token 数值对比和首次污染边界定位。

模块定位补充：以上是完整诊断流程的覆盖评价，不是要求 trace 内置所有诊断能力。当前功能以 [模块契约](kv-block-trace-contract.md) 为准：采集与查询生命周期、映射、搬运和选定窗口变化。数值对比、复读统计、干预实验及最终根因分析属于外部消费者。本报告的实验事实保留，完成度应按该职责边界评价。

## 基线和环境

- Python 实现：`98bd92a9` 加 `NewRequestData.req_id` 字段修复。
- 设备：Ascend 910B4，单机。
- 环境：Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4、CANN 9.1.0。
- vLLM：`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`。
- Native 资产：现有 vLLM-Ascend 0.23.0 镜像，Ascend commit `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`；本次未编译 native、未全局安装包。
- Python 源码以独立目录覆盖加载；因此这是以上组合的验证，不能外推到其他版本或部署。

## 结果

| 验证 | 实际结果 | 范围 |
| --- | --- | --- |
| Qwen3-0.6B，TP1 eager | 两次请求完成，8 个 forward，72 条 cache.diff，128-token prefix hit；无观测错误或映射不一致 | 正常生命周期与前缀复用取证路径 |
| Qwen3-0.6B，Mooncake 1P1D | 一次真实 transfer.complete；P 请求查询关联到双方 engine 和 5 个 forward | 元数据关联，不是内容一致性证明 |
| DeepSeek-V2-Lite，TP4+EP eager | 首请求完成；四个 rank 各 27 层，共 1080 条 cache.diff；第二个前缀命中请求四 rank 在同一 step 失败 | 真实多卡 MLA 取证；完整 workload 未通过 |
| 27 层 NPU 缓存张量，从 layer 11 注入改值 | 首个跨运行 after_sha256 不一致为 layer 11 | 人为张量 fixture，不是原模型发散复现；不能恢复 cosine |
| 被监控块的 offset 14 注入改写 | 27 层均记录 changed_offsets=[14] | 观测窗口内位置可见，不自动证明责任 writer |
| 未采样历史块 / 窗口开始前改写 | 两项均漏检，但日志完整性检查通过 | 明确的覆盖盲区 |

真实 PD 中，P 和 D 的块列表都为 `[1,2]`，D 的 slots 为 `[257,258,259,260]`，block size 为 128。D 快照只覆盖 block 2，没有覆盖包含 token position 14 的 block 1。该请求使用 130-token 合成 prompt，用于验证跨块覆盖，不是 PPT 原始短序列。

另一个短序列形式的 NPU fixture 在被监控块的 offset 14 提前改值，再开始观察 slot 19。由于 before 快照已包含坏数据，也未检测到新变化。

## 修复和对照

新请求数据使用 `req_id`，不是 `Request` 对象上的 `request_id`。原型读取错误字段会丢失首次调度 context。已新增先失败后通过的回归测试，修复后执行了上述真实推理。CPU 回归为 `44 passed`，Ruff 通过。

DeepSeek 的 trace-on/off 两组都在第二个 prefix-hit 请求出现 `AtbPagedCacheLoadGetWorkspaceSize failed` 和 attention 错误 `161002`；首请求生成 token 一致。PD 的两组均在进程退出时出现 `corrupted size vs. prev_size`，关闭 trace 组两 server 退出码均为 `-6`。这说明异常并非只在 trace 开启时出现，尚未定位 native 问题根因。

PD 的九个 writer 缺少 stop，完整性检查如实判为未闭合；Qwen 单引擎的两个 writer 正常闭合。单次 PD 请求还产生 818 条 dispatch，其中 813 条 requests 为空，应过滤或聚合无变化轮次。

## 与 PPT 的差距

1. 日志只有摘要和改写位置，没有 raw KV，不能复算逐 token、逐层 cosine / max-abs / 相对误差。
2. 当前只采写 slot 对应块，历史读集合未覆盖。
3. P produced/export 与 D receive/layout-ready/consume 缺少可比的内容检查点。
4. 缺少逐层/逐算子完成证据；整个 forward 内失败只记录宏观边界。

未取得 PPT 原始 prompt、bad/ref 数据，未复现原始偶发问题。未做吞吐开销或长稳结论。验证进程和专用容器已停止，原始日志、脚本与 HTML 报告保存在本次工作区的 `validation/131-20260929`。
