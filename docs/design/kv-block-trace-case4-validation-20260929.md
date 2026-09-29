# KV Trace：Case 4 原环境实机验收

2026-09-29，在 192.168.9.131 的 Ascend 910B4 上，用原 Case 4 镜像、模型和并发请求复现了故障。当前 trace 能采集并按请求或分配代次查询该案例中属于模块职责的关键事实。

## 环境与实际加载

- 镜像：`vllm-0.22.1rc1-fault-3-103:latest`，完整 ID `sha256:c4c94169cef544cc92ce2627a7c80f485279476d50e66f575952724414227d1e`。
- vLLM：`0decac0d96c42b49572498019f0a0e3600f50398`；镜像 Ascend：`5f6faa0cb8830f667266f3b8121cd1383606f2a1`。
- Python 3.12.13，torch 2.10.0+cpu，torch_npu 2.10.0，CANN 9.0.0；保留镜像 native 资产，未重编算子、未安装 wheel。
- 模型：`/data0/weights/DeepSeek-V2-Lite`；P 为 TP2、NPU 4/5，D 为 TP1 × DP2、NPU 6/7，开启 EP 与 `FULL_DECODE_ONLY`。
- 原故障构造保持：`IS_DECODE_NODE=1` 时 KV 内存固定为 7,962,624 bytes。两个 D engine 的 `cache.config` 均为 2 blocks、block size 128；block 0 为 null。
- 相比原 103 实验，仅迁移 host/NIC、物理卡号和端口；模型、请求和故障行为未修改。
- 基线直接运行原镜像。trace 组将 `65e9f33d` 的采集模块原样复制，并在原 0.22 runner/connector 上接入最小观察适配；没有安装 dummy 写入 mask。
- 实际模块路径：`/vllm-workspace/vllm/vllm`、`/vllm-workspace/vllm-ascend/vllm_ascend`。适配 diff、原文件、加载路径和 SHA256 均已保存。因此结论适用于该镜像上的模块适配，不能当作当前整仓所有路径的兼容性验收。

## 复现结果

每轮并发发送 short 与 spring。temperature=0，最大生成分别为 5/100 tokens，实际 prompt 长度分别为 11/19。外部采集脚本以连续 20 字符子串最大重复次数 ≥4 判为复读。

| 实验 | 请求成功 | spring 复读 | D 抢占计数 |
| --- | --- | --- | --- |
| 原镜像，无 trace | 40/40 | 15/20 | 两个 engine 均为 0 |
| 最新采集模块，观察开启 | 20/20 | 4/10 | 两个 engine 均为 0 |

这是时序敏感构造，样本比例不表示恒定概率，trace 同步采样会影响时序。两组比例不能用于评价性能或修复效果。最初 proxy `/health` 探测失败的启动尝试没有发送模型请求，不计入以上统计；改为 TCP 就绪检测后完成正式实验。

## 原始记录与查询

4 次复读分别位于零起始 round 5/6/8/9，覆盖 DP0/DP1，各有两个末层 MLA 分量变化。全部满足：

1. 同一 allocation lifetime 内，最近的搬运记录为 `transfer.complete`，local block 为 1，请求与 allocator owner 一致。
2. `phase=dummy`、`graph_mode=FULL`、`request_ids=[]`、`batch_request_ids=[]`，设备 slots 为 `[142]`，实际 block table 为 `[[1]]`。
3. `142 = 1 × 128 + 14`。layer 26 的两个 `cache.diff` 均只改变 offset 14，分别为 512/64 个元素；14 位于 19-token prompt 内。
4. 该窗口内没有观测到新的搬运、引用或分配变化。外部响应记录显示同一请求复读。
5. 4/4 请求查询和 4/4 pool/block/epoch 查询保留全部 8 条差分。原始 dummy 字段仍为空，关联置于 `access_relations`，标为 host-observed；`device_epoch_verified=false`。

一条可复核样例：响应 `chatcmpl-c48b87d2-c825-4894-90c4-911afcd54bbe`，DP0，pool `68fc4cd74a4b43a6946556f207b0250d`，epoch 6。

- `transfer-1114-44c417c6cb6a4c69a7cce20831d8ba17.jsonl:19`：传输返回完成。
- `worker-1114-422258b6d5984949a60c258909c7ec25.jsonl:2404`：空 batch dummy 开始，slot 142。
- 同文件 `:2405`、`:2406`：512/64 元素变化，offset 14，两个 before/after hash 不同。
- `responses.jsonl:12`：19 prompt tokens、100 completion tokens，重复计数 7。

另有 3 个前缀重叠窗口在执行期间发生搬运/上下文变化，没有被纳入以上稳定上下文证据。仅凭 `DUMMY_CACHE_CHANGED` 不足以认定根因。

## 实机暴露的修复

按 block/epoch 查询真实日志时，`cache.config.groups` 没有 `slots`，reader 原先直接取值导致 `KeyError`。修复为只枚举存在的执行 slots，保留 block-table 解析。新增两个回归场景先失败、修复后通过；focused debug suite 为 **72 passed**，Ruff 与格式检查通过。

采集代码仍为 `65e9f33d`。只更新离线 reader 后重读同一份日志，没有重复运行模型。采集 manifest 与修复后 reader manifest 分开保存，后者含 SHA256 和实际 CLI 命令。

## 命令与证据位置

远端实验目录：`/data01/tjh/work/kv-trace-case4-65e9f33d`，容器内 `/work`。原始脚本保留完整启动参数；服务器采用 `bash -ic` 保留镜像的 Mooncake/CANN 动态库环境。

```bash
# 原镜像阶段（适配安装前），由脚本启动 P/D/proxy 并管理子进程。
python /work/scripts/run_case4.py --phase baseline --rounds 20
# 安装观察模块后，另一次启动。
python /work/scripts/install_trace_adapter.py
python /work/scripts/run_case4.py --phase trace --rounds 10
# 对保存的日志分析；复读判定仍在外部脚本中。
python /work/scripts/analyze_case4_trace.py
python /work/payload/tools/kv_block_trace.py /work/results/trace/events \
  --request chatcmpl-c48b87d2-c825-4894-90c4-911afcd54bbe
python /work/payload/tools/kv_block_trace.py /work/results/trace/events \
  --pool-id 68fc4cd74a4b43a6946556f207b0250d --block 1 --epoch 6 --json
python /work/payload/tools/kv_block_trace.py /work/results/trace/events --check
```

前两条查询 CLI 退出 0；`--check` 退出 2：81,224 条记录无序号缺口、观测失败或截断，但 16 个 writer 都缺少 `trace.stop`，全文件不能声明完整闭合。已有目标 span 的 begin/diff/end 和关系依据均可查询，不能据此排除尾部丢失。

本地完整 HTML/SVG 报告、原始 JSONL、响应、日志、指标、脚本、适配 diff、文件校验值：`validation/case4-20260929`（工作区目录，位于仓库同级）。原始归档 `evidence.tar.gz` SHA256：`2bdfe41186eb3ffe6082a33d8ebc71cfa26383d87172ba6d6af4654897236491`。

专用容器 `tjh-kv-case4-65e9f33d` 已停止，NPU 4–7 无残留进程。

## 验收边界

该案例的生命周期、请求/块映射、搬运关联、dummy 设备 slots、选定末层窗口差分及关联检索已通过真实 DP2 图模式验证。采集未证明 device epoch 或传输内容正确性；本次没有重做 mask 干预、逐层 bad/ref 数值分析、性能或长稳测试。这些不能从本次样本推出，也不要求内置进 trace 模块。
