# CANN 分支 SWA bounded replay 回移与 A3 验证

本分支以 `rfc/vllm_cann` 的 `7e37e29a7803d6897a7113088e2b771f83205bc0`
为基线，供 DeepSeek V4.1 在 A3 上验证功能、质量与性能。默认关闭 replay，
通过同一份代码上的开关对照评估收益。未在本机执行 NPU 实测。

## 回移内容

| 上游 PR | 内容 |
| --- | --- |
| [#17742](https://github.com/vllm-project/vllm-ascend/pull/17742) | 前缀命中后的 encoder SWA 窗口重建、slot 屏蔽和算子可见范围 |
| [#17496](https://github.com/vllm-project/vllm-ascend/pull/17496) | decoder 尾窗口裁剪、token gather/scatter |
| [#18936](https://github.com/vllm-project/vllm-ascend/pull/18936) | MRV1/MRV2 适配、SWA 索引和相关执行上下文 |
| [#19321](https://github.com/vllm-project/vllm-ascend/pull/19321) | MRV2 PCP 下的 decoder replay |
| [#20210](https://github.com/vllm-project/vllm-ascend/pull/20210) | MRV2 PCP＋DP 的执行宽度协调及 idle rank 处理 |
| [#18322](https://github.com/vllm-project/vllm-ascend/pull/18322) | 必要依赖：vLLM 0.30 的配置、scheduler 和请求协议兼容层 |
| [#18750](https://github.com/vllm-project/vllm-ascend/pull/18750) | 必要依赖：FULL graph replay 元数据覆盖 padded request rows |

回移保留 RFC 的 A5 wheel 路由、group-cache 字段、layerwise host offload 和连接器注册。
针对旧接口保留以下适配，而非升级整个仓库：

- 上游 vLLM 继续使用 `v0.30.0`（该 tag 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`）。
- 保留 #18322；#18936 在高版本删除的旧协议在本分支仍然需要。
- PCP context 通过 RFC builder 的关键字参数传递。
- packed cache annotation 支持 vLLM 0.30 的旧参数名。
- replay 开启时保留 prompt 的 prefill 状态，避免旧 P/D 图适配误跳过 replay。

## A3 构建

在当前可正常运行 RFC 分支的 A3 容器中操作，保留已经验证的 CANN、torch、
torch-npu、量化模型和部署拓扑。先停止旧服务，再切换代码并重新构建。

```bash
git fetch origin backport/cann-swa-bounded-replay
git switch --track origin/backport/cann-swa-bounded-replay

python -c 'import vllm; print(vllm.__version__)'
npu-smi info

# npu-smi 可用时 setup.py 会识别 SoC；如需手动设置，使用机器的实际型号。
# Dockerfile.a3 的默认示例是 ascend910_9391，部分 A3 型号为 ascend910_9392。
# export SOC_VERSION=ascend910_9391
COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=8 python -m pip install -v -e . --no-build-isolation --no-deps
```

此次修改包含 A2/A3 `arch22` 的 `sparse_flash_mla` band 模式：
`ori_topk_length` 必须真正限制每个 query 可见的历史 KV。需要重新编译自定义算子，
并重启服务；保留旧算子二进制无法验证完整回移。这里的 Python 调度逻辑在各硬件间共享，
算子实现按架构分别编译。

先在 A3 运行小算子回归，确认窗口限制生效后再加载大模型：

```bash
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_sparse_flash_mla_band.py
```

该用例用固定数值区分正确的单行可见窗口与未裁剪的历史读取，并检查错误形状被拒绝。

## 启动与功能验证

复用现有的 DeepSeek V4.1 启动命令、模型和并行参数。第一轮优先使用
PCP=1、DCP=1、PP=1 的已验证拓扑；MRV1 和 MRV2 分别在两次实验中固定，
不要在开关对照中同时切换 runner。MRV2 用 `VLLM_USE_V2_MODEL_RUNNER=1`，
MRV1 用 `VLLM_USE_V2_MODEL_RUNNER=0`。

| 实验 | 两边共同参数 | 唯一区别 |
| --- | --- | --- |
| 冷 prefill / decoder 裁剪 | `--no-enable-prefix-caching --enforce-eager` | `--no-swa-bounded-replay` / `--swa-bounded-replay` |
| 重复前缀 / encoder 重建 | `--enable-prefix-caching --enforce-eager` | 同上 |
| Decode 图兼容性 | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`，移除 `--enforce-eager` | 同上 |

使用同一分支和同一套新编译的算子做 OFF/ON 对照。另将新分支 OFF 与原始 RFC 基线
做一次回归对照，检查默认关闭时的行为。性能采集时固定编译缓存策略，完成预热后再计时。

检查服务日志中的两个阶段：

- 启动时 `Ascend decoder bounded replay: layers ... keep ... tokens` 表示已接入 decoder 适配器。
- 实际长 prefill 后 `SWA decoder bounded replay active (MRV1/MRV2): ...` 表示执行了裁剪。
- 重复前缀请求后 `SWA bounded replay is active: a prefix hit is rebuilding its window` 表示命中重建路径。
- Mooncake P/D 接收后 `SWA P/D handoff: received SWA window; skipping prefix replay` 表示复用了传入窗口。

`--swa-bounded-replay` 被接受、启动成功或 HTTP 返回成功，均不能单独证明裁剪路径执行。
短于窗口的请求不一定触发裁剪；开启 prompt logprobs 的请求会保留所需行。

现有服务可直接使用 [curl 测试脚本](../../../examples/swa_replay/README.md)，无需重启
或编译服务。脚本默认连接 `127.0.0.1:8989`、服务别名 `dsv41`，提供短输入、长输入、
重复前缀和并发四个场景，保存完整响应、HTTP 状态、usage 和非流式请求总耗时。

```bash
SERVICE_LOG=/workspace/serve_swa_replay_on.log \
  bash examples/swa_replay/02_long_prefill.sh
```

若设置 `SERVICE_LOG`，脚本仅显示本轮开始后新增的 replay 日志；每个 worker 以前
已打印过首次激活日志时，本轮可能没有新增消息。HTTP 成功仍不能单独证明裁剪生效。

可运行已有的小型服务探针，先保存 OFF 结果，再重启服务验证 ON：

```bash
python tests/e2e/models/swa_bounded_replay_probe.py \
  --base-url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL_NAME \
  --output /tmp/swa-off.json

# 重启同配置服务，开启 --swa-bounded-replay。
python tests/e2e/models/swa_bounded_replay_probe.py \
  --base-url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL_NAME \
  --baseline /tmp/swa-off.json --output /tmp/swa-on.json
```

探针检查短问答、长输入、重复前缀和并发响应，是小范围 smoke 回归；不能代替完整精度评估。
bounded replay 是近似计算，长文本逐字相同不是通用正确性要求；还需使用目标业务数据或
AISBench 比较任务指标、异常输出、请求成功率和生成长度。

## 性能对照

可直接使用[单机流式 benchmark 脚本](../../../examples/swa_replay/BENCHMARK.md)，
按相同矩阵跑 OFF/ON，保存原生逐请求数据并生成 HTML/CSV 对照报告。

冷 prefill 实验先关闭 prefix caching；分别测试 4K、16K、32K 输入和固定输出长度。
确保 `max-model-len` 覆盖输入加输出长度，并固定 `max-num-batched-tokens`。
每组 OFF/ON 交替至少三轮，固定随机种子、模型、runner、并行、图模式、batch token
budget、并发、Engram offload 和采样参数。下例从并发 1 开始：

```bash
vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
  --model YOUR_SERVED_MODEL_NAME --tokenizer /path/to/checkpoint \
  --dataset-name random --random-input-len 16384 --random-output-len 128 \
  --num-prompts 32 --max-concurrency 1 --request-rate inf --seed 42 --ignore-eos \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
  --save-result --result-dir /tmp/swa-bench --result-filename off-16k-c1-r1.json
```

开启 replay 并重启、预热后，用相同命令另存 `on-16k-c1-r1.json`。提高并发到实际业务值，
并对其他输入长度重复。主要关注 TTFT、输入吞吐、输出吞吐、TPOT/ITL、峰值 HBM、
可用 KV blocks、成功请求数和生成 token 数。重复前缀实验单独测量实际 prefix hit 与
重建开销，不把 cache 命中率差异当作纯计算优化收益。

## 支持条件与限制

- 目标为 DeepSeek V4.1；DSpark 草稿的窗口必须与 target 适配。
- decoder 裁剪在 eager prefill 生效；可与 `FULL_DECODE_ONLY` decode 图组合，
  不提供捕获后的 decoder prefill 图加速。
- PCP＋DP 的新增路径属于 MRV2。先验证 PCP=1，再使用相同开关测试 MRV2 的 PCP＋DP。
- PP、DCP、microbatching、LoRA、不兼容的草稿或 replay 边界后的 Engram 会关闭 decoder replay。
- vLLM 0.30 兼容层接入 Ascend 的 `MooncakeHybridConnector` 和 `MooncakeConnectorV1`
  请求级 P/D 交接，保留全局 KV、SWA 尾部和必要状态的原有传输路径。
  后续还接入 MultiConnector 与请求级 AscendStore：按所选来源恢复 SWA，
  见 [Store replay 指南](cann_swa_store_replay.md)。
  未声明 `get_loaded_kv_cache_group_ids` 的 child 仍明确拒绝；
  AscendStore layerwise、Store PCP/DCP 多路和可选的 C128 分块传输不在扩展范围。
  配置和验收步骤见 [P/D replay 指南](cann_swa_pd_replay.md)。
- 自定义 scheduler 若替换了上游调度循环，会关闭 SWA replay 并给出原因；需检查最终日志。
- 本分支默认 OFF；不把 CPU 回归通过表述为 A3 功能、精度或性能已经通过。

## 本地验证记录

2026-11-03：使用上游 vLLM `v0.30.0` 源码和 CPU 测试夹具，29 个相关测试文件
共 **1356 passed、23 skipped、465 subtests passed**，包含四进程 Gloo PCP2＋DP2
协调、实际 scheduler 前缀回放、请求序列化、cache/slot 边界、runner 以及 V4.1 PCP
元数据回归。CPU 测试使用 Python 3.13、PyTorch 2.14.0+cpu；不是 A3 的
torch-npu 2.10 实测。数据见
[验证记录](cann_swa_bounded_replay_validation.json)。

61 个变更 Python 文件的 AST、Ruff lint/format 和 `git diff --check` 通过。
仓库要求的 `bash format.sh ci` 在 markdownlint 的 nodeenv 环境下载阶段因 SSL
证书链错误中止，未完成全量检查；单独调用 markdownlint 也被 npm registry 的截断 JSON
响应阻断。没有把这些失败标为通过。

A3 算子构建、真实权重功能、质量和性能数据仍需实机补齐。CPU 回归没有执行自定义
AscendC/Triton 算子，也没有验证 NPU 集合通信或建立性能收益结论。

## MRV1 启动时缺少 aux_output_config

2026-11-04 收到 A3 单机 DP4/TP4 的启动反馈：权重加载后，显存 profiling 的
`_sync_metadata_across_dp()` 报出
`AttributeError: 'VllmConfig' object has no attribute 'aux_output_config'`。
这是回移 #18936 时遗漏的接口适配：vLLM 0.30 把 `enable_return_routed_experts`
放在 `model_config`，较新接口才使用 `aux_output_config`。本分支已优先读取新接口、
在其不存在或为 `None` 时回退到旧接口；不向 vLLM 注入一个虚构的新配置对象。

该异常不表示 SWA replay 必须使用 MRV2。对于下述 DP4/TP4＋旧 DSA-CP 配置，
应应用修复 `a52a257e27ae9670c043ed91c0161e9a1d426391`，并继续使用
`VLLM_USE_V2_MODEL_RUNNER=0`。

此前“只把 runner 改为 V2、其余参数不变”的绕行说明有误，现已撤回。CANN 分支的
`AscendConfig.derive_and_validate()` 明确拒绝 MRV2＋`enable_dsa_cp=true`，会在
创建 `VllmConfig` 时退出，尚未处理请求。仅删除旧 DSA-CP 也不能证明 decoder replay
生效：继续启用 FlashComm/SP 时，`make_decoder_replay()` 会因 SP 缺少旧 DSA-CP
而关闭裁剪。要保留当前性能对照基线，应走修复后的 MRV1 路径。

对于现有 DP4/TP4、`FULL_DECODE_ONLY`、`enable_dsa_cp=true`、FlashComm 开启且没有
KV connector 的配置，继续沿用 CANN 分支的旧 DSA-CP 参数。不要套用 0.32 CI 的
“删除 enable_dsa_cp、切到 PCP4”步骤：那是另一套版本和拓扑的迁移。
本分支的 decoder replay 允许配合旧 DSA-CP 的 SP 路径，但实际生效仍须检查
`SWA decoder bounded replay active (MRV1/MRV2)` 日志。

在现有源码目录应用修复（如果尚未包含该提交）：

```bash
cd /vllm-workspace/vllm-ascend
git fetch https://github.com/gg820827047/vllm-ascend.git backport/cann-swa-bounded-replay
git cherry-pick a52a257e27ae9670c043ed91c0161e9a1d426391
```

然后将启动脚本中的 runner 值恢复为 `0`，保留 `enable_dsa_cp=true` 和其余原参数，
重启服务及全部 worker。该修复只改 Python；可编辑安装更新源码后即可加载新逻辑，
wheel 安装则需更新对应的包，不需要仅为这处修改重新编译自定义算子。

首轮建议在同一个 runner 上分别跑 replay OFF/ON。切换到 MRV2 后，原 MRV1/OFF
的性能不能直接当作 MRV2/ON 的 replay 对照基线。

本次修复先用真实 vLLM 0.30 `VllmConfig` 复现相同异常，再执行原来的 29 个 CPU
回归文件：**1359 passed、23 skipped、477 subtests passed**；其中 DP padding 专项
为 9 passed、144 subtests passed。新用例覆盖旧字段、新字段优先级以及新配置为
`None` 时的回退。修改仅涉及 Python 读取逻辑，没有改动算子源码；实机启动仍需验证。
2026-11-04 的全量格式检查在 nodeenv 解析 Node 版本 `0-alpha` 时中止，未标为通过。
