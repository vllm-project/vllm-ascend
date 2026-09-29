# KV Trace 请求性能开销：131 / Case 4

2026-09-29 实测结论：本次固定并发 2 的 Case 4 负载下，默认 trace 没有测出明显的 spring 请求性能膨胀；开启末层快照后，spring P50 增加 44.4%、P95 增加 52.4%，输出吞吐下降 31.7%。

## 环境与方法

- 代码：`8f94f015327d8875b18c369ab4c30e6c160539bf`；采集模块与原 0.22 runner/connector 的观察适配，实际加载文件 SHA256 在每组 `status.json` 中，四组完全一致。
- 使用 [Case 4 原环境](kv-block-trace-case4-validation-20260929.md)：131 / Ascend 910B4，镜像 `c4c94169cef544cc92ce2627a7c80f485279476d50e66f575952724414227d1e`，DeepSeek-V2-Lite，P TP2 / NPU 4–5，D TP1 × DP2 / NPU 6–7，EP、FULL_DECODE_ONLY，D 每 engine 2 个 KV blocks。保留原故障构造，没有 mask 或 native 重编译。
- 顺序：关闭 trace 前测 → 默认 trace → 末层快照 → 关闭 trace 后测。每组独立启动服务、预热 5 轮，再采集 20 轮；启动与图捕获不计时。
- 每轮并发提交原 short/spring 请求；temperature=0，`ignore_eos=true`，实际 prompt 分别为 11/19 tokens，输出固定为 5/100 tokens。200 个请求（含 40 个预热）全部成功。
- 测量非流式端到端延迟，未测 TTFT/TPOT。固定生成长度用于避免 EOS 或异常输出长度改变工作量；本轮不评价生成质量。

## 有效窗口选择

快照组第 20 轮 spring 请求期间，两个 P worker 触发每 writer 64 MiB 的默认日志上限，产生 `trace.truncated` 并停止采集。第一次截断时间为 `1790669666731183110` ns；它仅与最后一个 spring 请求重叠。

因此四组统一只统计前 19 轮，均在任何截断之前：每组 38 请求 / 1,995 输出 tokens。原始 20 轮、预热、截断记录和完整统计全部保留；排除依据是采集状态，而非延迟高低。没有为凑满样本重跑模型。

有效窗口吞吐 = 输出 tokens / 窗口时长；窗口从首请求开始，到最后响应结束。响应结束以记录的 wall start 加 monotonic 请求耗时估算。P50/P95 对逐请求耗时作线性插值；关闭 trace 的参考分位数合并两次基线的 38 个同类样本，参考吞吐则合并 tokens 与时长。

## 结果

| 配置 | spring P50 / P95（秒） | short P50 / P95（秒） | 输出 tok/s |
| --- | --- | --- | --- |
| 关闭 trace 前测 | 1.711 / 1.880 | 0.327 / 0.482 | 60.19 |
| 默认 trace | 1.712 / 1.789 | 0.304 / 0.335 | 61.15 |
| 末层快照 | 2.468 / 2.810 | 0.348 / 0.385 | 41.50 |
| 关闭 trace 后测 | 1.688 / 1.819 | 0.322 / 0.461 | 61.35 |

合并基线的 spring P50/P95 为 1.710/1.843 秒。相对它：

- 默认 trace：spring P50 +0.1%，平均延迟 -0.6%，吞吐 +0.6%。这些小幅差异不足以宣称有加速效果。
- 末层快照：spring P50 +44.4%，P95 +52.4%，平均延迟 +46.5%，吞吐 -31.7%。
- short 的快照 P50 +7.0%，均值基本不变；小样本尾延迟较不稳定，不能外推。
- 前后关闭 trace 的 spring P50 漂移 -1.3%，short 为 -1.6%。

默认 trace 只设置 directory/run_id，实际为 `snapshots=false, device_metadata=true`；快照组额外设置 `snapshots=true, layers=[-1]`。没有修改代码默认资源预算。

默认组完整记录 147,156,855 bytes，未出现截断或观测错误。快照组完整记录 210,138,191 bytes，其中包含窗口后的 2 次截断。日志量含启动、预热及全部 20 轮，不是有效窗口的日志量。16 个 writer 缺少正常 stop 的既有限制仍保留。

## 解释与限制

默认 trace 已有设备元数据拷贝与同步，快照额外拷贝并比较末层 KV；还有 Python bookkeeping 和同步 JSONL I/O。本次测量总体差异，没有用 profiler 分摊各项成本。

结果只适用于该原镜像适配、模型、并行配置、采样范围和并发度。19 个同类样本的 P95 只供本轮对照；未证明生产高并发容量、长稳或其他模型的开销。测试开始时，其他任务使用 NPU 0–3，共享主机资源；保留了各阶段 NPU 与负载记录。

## 证据与复核

本地 HTML / 3 张 SVG / 原始响应、日志、配置和计算脚本位于工作区 `validation/performance-20260929`（仓库同级）。远端目录为 `/data01/tjh/work/kv-trace-case4-65e9f33d/perf-20260929`，容器内 `/work/perf-20260929`。

- `scripts/client.py`：请求开始时间、monotonic 耗时、固定长度与原始响应。
- `scripts/run_arm.py`、`scripts/controller.sh`：完整命令、环境配置、阶段顺序与进程回收。
- 各组 `summary.json`：原始 20 轮统计；`analysis.json`：共同有效窗口的选择与最终对照。
- `snapshot/events/snapshot/worker-722-1da5339536424437a4c39db5f9ab069f.jsonl:64115`、`worker-672-867248d7d8824a7fb9078d5b790f8e1a.jsonl:64115`：两个预算截断记录。
- 原始证据归档 SHA256：`ec764c6a1a25721d995b83f83d0823f5ff441f5bb8ab5a09a2ac226ddaef1114`。

专用容器已停止，NPU 4–7 无残留进程。部署代码已推送到 `codex/kv-block-trace-design`。
