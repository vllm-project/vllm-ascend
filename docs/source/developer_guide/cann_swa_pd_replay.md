# CANN 分支：DeepSeek V4.1 的 Mooncake P/D 与 SWA replay

本指南对应 `backport/cann-swa-bounded-replay`，上游 vLLM 仍固定为
`v0.30.0`（`ced6857afa0ea7b2e3f0846a62e1394e90f15607`）。
本次为实验性代码适配；CPU 调度和缓存回归不代表 A3 功能、质量或性能已验收。

## 1 CED 与此次改动的关系

CED（Causal Encoder-Decoder）是 DeepSeek V4.1 的模型架构，不是独立的启动开关。
当前 CANN 分支已包含模型结构、全局 KV 来源层 `[2, 8, 14, 20]`、层间 KV 共享、
压缩器和 indexer 等实现。先前反合的 #17496 / #18936 等在这些结构上实现
decoder prefill 裁剪：0～20 层保留完整输入，21～39 层保留每段 prefill 的尾部窗口。
层编号从 0 开始；裁剪边界和窗口来自模型配置，并非固定写死。

此次补齐的是该计算路径与 Mooncake 请求级 P/D 传输的配合。
P、D 仍各自加载完整模型。这里不包含“P 只加载 encoder 权重”的新部署形态，
也不包含上游后续的第 20 层内部裁剪或 decoder prefill 图捕获优化。

## 2 交接协议

沿用已有 Mooncake 传输：P 传递全局 KV、必要的 SWA 尾部和其他请求状态，
D 在本地恢复后继续生成。SWA 不作为可复用前缀快照持久化，
不等于当前请求的 P/D 交接可以省略 SWA。

| 情形 | 调度行为 |
| --- | --- |
| P 的本地全局前缀命中，SWA 窗口未缓存 | 回退命中尾部窗口，重建 SWA |
| D 从 P 成功加载所有必要 SWA 组 | `replay_start=0`，不重复重建窗口 |
| 异步加载尚未完成 | 不调度该请求的模型前向 |
| 异步加载部分失败 | 使用上游确认的有效前缀；恢复窗口或短前缀冷重试 |
| 请求被抢占后从本地全局前缀恢复 | 重新建立窗口，传递新的 replay 边界 |

接口对齐上游 `get_loaded_kv_cache_group_ids(request)`。调度器根据实际加载的
缓存组决定窗口是否已恢复，不能仅依靠远端命中的 token 数判断。
实现参考上游 vLLM
[`dd4e40142a3280ef404493e5117bde0c60c6e389`](https://github.com/vllm-project/vllm/blob/dd4e40142a3280ef404493e5117bde0c60c6e389/vllm/v1/core/sched/scheduler.py#L3523)，
保留 v0.30 的调度主体和异步完成流程，不替换为较新版本的完整 scheduler。

## 3 接入范围

- `MooncakeHybridConnector`：
  `vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.MooncakeConnector`。
- `MooncakeConnectorV1`：
  `vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.MooncakeConnector`。
- 两者传输请求自有缓存组，包括不参与前缀持久化的 SWA 组。
- 未实现上述接口的 connector 继续在启动时拒绝此组合。
  此次没有接入 `AscendStoreConnector`、Mooncake Store 或其他 layerwise offload。
- MRV1＋旧 DSA-CP 是首轮 A3 路径；MRV2 不能配置 `enable_dsa_cp=true`。
  不需要为此次 P/D 接入切换 runner 或改为 PCP 拓扑。
- decoder 裁剪仍要求 eager prefill；D 可使用 `FULL_DECODE_ONLY`。
  PP、DCP、microbatching、LoRA 等原有门禁继续生效。
- DSpark 沿用已有草稿窗口兼容检查与传输机制；CPU 测试不证明实际草稿接受率。

## 4 从 v0.30 的 A3 P/D 配置开始

先用 [DeepSeek V4.1 模型指南的 A3 1P1D 配置](../tutorials/models/DeepSeek-V4.1-Flash.md#52-1p1d-pd-disaggregated-deployment)
跑通 replay OFF 基线。该配置是两台 A3：P 为 DP4/TP4，D 为 DP8/TP2。
代理端点为 P 的 7100～7103 和 D 的 7100～7107。
如果现有部署采用其他 DP/TP 组合，保留实际拓扑，并确保 P、D 的 connector
配置同时准确声明两端 DP/TP。

在 P、D 两台机器都更新相同分支，并重启所有相关引擎：

```shell
git fetch origin backport/cann-swa-bounded-replay
git switch backport/cann-swa-bounded-replay
git pull --ff-only origin backport/cann-swa-bounded-replay
```

保持 `VLLM_USE_V2_MODEL_RUNNER=0`。若此前已安装本分支相同 Ascend 自定义算子，
此次新增调度和 connector 改动只涉及 Python；editable 安装更新代码后重启进程。
首次安装 replay 分支仍需按[反合指南](cann_swa_bounded_replay_backport.md)完成算子构建。

在两端原有 `vllm serve` 命令中各增加：

```shell
--swa-bounded-replay
```

| 项目 | P | D |
| --- | --- | --- |
| Runner | MRV1 | MRV1 |
| 原文档 DP/TP | 4 / 4 | 8 / 2 |
| 前缀缓存 | `--enable-prefix-caching` | `--no-enable-prefix-caching` |
| 执行模式 | `--enforce-eager` | `FULL_DECODE_ONLY`＋async scheduling |
| 旧 DSA-CP | 保留原文档的 `enable_dsa_cp=true` | 保留原有配置 |
| KV connector | producer | consumer |
| replay ON 实验 | `--swa-bounded-replay` | `--swa-bounded-replay` |
| replay OFF 对照 | `--no-swa-bounded-replay` | `--no-swa-bounded-replay` |

两端保持相同权重、tokenizer、量化、block size 和兼容的缓存布局。
不要将 v0.32 的 PCP 配置直接拼入 v0.30 的旧 DSA-CP 配置。

## 5 功能与日志检查

P 启动时应看到 decoder replay 层范围；长请求实际执行后，应看到：

```text
SWA decoder bounded replay active (MRV1): ...
```

D 收到包含必要 SWA 窗口的请求后，应看到：

```text
SWA P/D handoff: received SWA window; skipping prefix replay.
```

P 重复前缀命中时可能看到：

```text
SWA bounded replay is active: a prefix hit is rebuilding its window.
```

这些日志在每个进程中只打印首次激活；一个请求不需要让所有 DP 实例都打印。
D 成功接收完整窗口时，没有“重建128 token”的日志是正常情况。
一次 HTTP 成功或一条初始化日志，不足以证明性能路径已经执行。

将已有 curl 脚本指向 P/D 代理，不能只调用 P 服务端口：

```shell
BASE_URL=http://<PROXY_IP>:<PROXY_PORT> MODEL=dsv41 \
  bash examples/swa_replay/02_long_prefill.sh

BASE_URL=http://<PROXY_IP>:<PROXY_PORT> MODEL=dsv41 \
  bash examples/swa_replay/03_prefix_reuse.sh

BASE_URL=http://<PROXY_IP>:<PROXY_PORT> MODEL=dsv41 \
  bash examples/swa_replay/04_concurrent.sh
```

同时保留 P、D、代理三端日志，检查传输完成、无非法访问和无请求挂起。
建议覆盖窗口附近长度、8192 分块附近长度、16K/32K 长输入、重复前缀与并发请求。
比较生成内容并使用业务质量集；bounded replay 本身是近似计算，不能只做逐字一致验收。

也可通过已有 real-weight probe 保存 OFF/ON 对照，并在 ON 运行时检查两端日志。
先不传日志参数，运行 OFF 基线；重启两端为 ON 后，提供本次启动的新日志文件：

```shell
python tests/e2e/models/swa_bounded_replay_probe.py \
  --base-url http://<PROXY_IP>:<PROXY_PORT> --model dsv41 \
  --output /tmp/pd-replay-off.json

# 两端改为 replay ON，重启；把本次 P、D 日志复制到执行 probe 的机器。
python tests/e2e/models/swa_bounded_replay_probe.py \
  --base-url http://<PROXY_IP>:<PROXY_PORT> --model dsv41 \
  --baseline /tmp/pd-replay-off.json --output /tmp/pd-replay-on.json \
  --prefill-log /tmp/prefill-on.log --decode-log /tmp/decode-on.log
```

日志检查覆盖 P 的实际 prefill 裁剪和 D 的窗口交接标记，结果保存至输出 JSON。
使用本次启动的日志，避免把历史激活记录当作本轮证据；marker 是进程级证据，
不是逐请求 trace。probe 的简单算术任务检查文本一致性，不能替代业务精度集。

## 6 性能对照与验证边界

固定相同 runner、拓扑和调度参数，先两端 replay OFF，再两端 replay ON。
通过代理运行 streaming benchmark，记录 TTFT、TPOT/ITL、吞吐与失败率，
同时观察 P prefill、KV 传输和 D 首次计算的耗时。
curl 脚本的非流式 `time_total_s` 是请求总耗时，不是 TTFT。

P 的 decoder prefill 裁剪是直接计算收益来源；完整窗口交接避免 D 再做窗口重建。
这些改动不保证每步 decode 更快，也不能把上游 NVIDIA 性能数据当作 A3 实测。

CPU 回归覆盖真实 vLLM 0.30 scheduler、缓存分配和块引用、同步/异步加载、
短命中、非整块边界、失败恢复、抢占后的 MRV1/MRV2 payload，
以及真实 Hybrid producer 的本地前缀命中与 MTP/DSpark 调度。
模型算子、HCCL、真实 Mooncake 网络、A3 精度和性能需要在目标机器上继续验证。

本轮记录：31 个相关文件共 **1449 passed、23 skipped、513 subtests passed**。
复现命令、源码校验值及未验证项见[验证记录](cann_swa_pd_replay_validation.json)。
