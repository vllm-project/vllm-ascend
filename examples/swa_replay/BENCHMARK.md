# 单机 SWA replay OFF/ON 性能对照

`bench.py` 封装本分支配套 vLLM **v0.30.0** 的 `vllm bench serve`。
使用流式 `/v1/completions`，由 vLLM 原生基准统计 TTFT、TPOT、ITL、E2E 延迟与吞吐。
脚本本身只依赖 Python 3 标准库；运行它的环境需要已安装可用的 `vllm` 命令和对应 tokenizer。
默认适配当前 A3 单机 DP4/TP4、MRV1 服务：`http://127.0.0.1:8989`，模型别名 `dsv41`。

## 1 服务端：只改变 replay 开关

两次实验固定模型、V1、DP4/TP4、DSA-CP、图模式、Engram、batch token budget、
最大序列数、显存比例和输出设置。客户端脚本不会停止或重新拉起服务。

首轮测 decoder prefill 裁剪，两组都显式关闭 prefix caching，避免重复预热和多轮测试命中
旧前缀，掩盖计算差异。在你原来的启动脚本中使用以下参数组合：

```shell
# OFF：替换原来的 replay 开关，再启动服务。
--no-enable-prefix-caching --no-swa-bounded-replay

# ON：停止 OFF 服务，将 replay 开关替换为 ON，再启动服务。
--no-enable-prefix-caching --swa-bounded-replay
```

继续保持：

```shell
export VLLM_USE_V2_MODEL_RUNNER=0
```

两组均可保留 `FULL_DECODE_ONLY`；prefill 在 eager 路径执行。
`max-model-len` 必须覆盖输入＋输出，默认 16K＋128 小于你目前设置的 150000。
检查服务 ready 后再开始客户端测试，压测期间不要混入其他业务流量。

## 2 跑 OFF

在已更新本分支的 vllm-ascend 源码根目录、安装了 vLLM 的环境中执行：

```shell
python examples/swa_replay/bench.py run --mode off \
  --tokenizer /data/models/DeepSeek-V4.1-Flash-1017-w8a8 \
  --output-dir /workspace/swa-bench
```

默认矩阵：

| 项目 | 默认值 |
| --- | --- |
| 输入 | 16384 token |
| 输出 | 128 token；`ignore_eos`，固定输出上限 |
| 并发 | 1、4，各测一组 |
| 每组正式请求 | 32 |
| 每组预热请求 | 8，不计入正式统计；另有原生基准的连通性测试请求 |
| 轮数 | 3；每轮 seed 为 42、43、44 |
| 请求速率 | `inf`，由并发上限限制在途请求 |
| 百分位 | P50、P95、P99，另保留均值 |

同一轮 OFF/ON 使用相同 seed 和随机负载。v0.30 的 `--random-range-ratio 0`
表示固定长度，脚本已显式设置；不使用共享固定前缀。实际服务返回的输入、输出长度仍会保存并核对。
预热由原生 benchmark 完成，不保证请求一定覆盖每个 DP；可增大 `--warmups`。

## 3 重启为 ON，再跑相同命令

```shell
python examples/swa_replay/bench.py run --mode on \
  --tokenizer /data/models/DeepSeek-V4.1-Flash-1017-w8a8 \
  --output-dir /workspace/swa-bench
```

`--mode` 是实验标签，不会替你切换服务配置。如果 `/server_info` 可读，脚本会核对
服务端 replay 状态、prefix caching 和最大长度，并记录筛选后的配置；不保存任意系统环境变量。
接口不可读或没有暴露开关时会明确提示，需要人工检查启动参数。
ON 服务日志应在真实长 prefill 后出现：

```text
SWA decoder bounded replay active (MRV1): ...
```

每个 worker 只打印首次激活；初始化的层范围日志和一次 HTTP 成功不能证明裁剪已经执行。

## 4 生成报告

```shell
python examples/swa_replay/bench.py compare --results-dir /workspace/swa-bench
```

默认选择该目录下最近完成的 OFF、ON 两组，生成：

```text
/workspace/swa-bench/comparison/report.html
/workspace/swa-bench/comparison/comparison.csv
/workspace/swa-bench/comparison/comparison.json
```

HTML 可直接用浏览器打开。原始结果保留在独立的 `off-时间-随机编号`、`on-时间-随机编号`
目录，包含每个 case 的原生结果 JSON、原始日志、完整命令、实际长度和逐请求流式统计。
每组有 `manifest.json`，中断或失败也保留已产生的数据。失败实验不会更新 `latest-off.json` / `latest-on.json`。
重复生成报告会覆盖 `comparison` 内的汇总，原始实验目录保留。

也可以明确指定两组目录，避免混用其他实验：

```shell
python examples/swa_replay/bench.py compare \
  --off-dir /workspace/swa-bench/off-实际目录名 \
  --on-dir /workspace/swa-bench/on-实际目录名 \
  --output-dir /workspace/swa-bench/comparison-selected
```

有请求失败、输出长度不足、缺失指标、矩阵/seed 不同或实际输入输出长度不同，会返回非零状态。
两组都能读取服务配置时，也会检查记录的模型、并行、调度、图模式、Ascend配置和runner设置是否一致。
这不能替代人工核对所有驱动、CANN、环境变量和启动参数。

## 5 扩展矩阵或先做小规模检查

OFF、ON 两次必须使用相同参数：

```shell
# 先检查链路：16K输入、32输出、单并发、一轮、4个正式请求。
python examples/swa_replay/bench.py run --mode off \
  --tokenizer /data/models/DeepSeek-V4.1-Flash-1017-w8a8 \
  --input-lens 16384 --output-len 32 --concurrencies 1 \
  --rounds 1 --num-prompts 4 --warmups 4 --output-dir /workspace/swa-bench-smoke

# 更完整的负载矩阵；ON运行时只替换 --mode。
python examples/swa_replay/bench.py run --mode off \
  --tokenizer /data/models/DeepSeek-V4.1-Flash-1017-w8a8 \
  --input-lens 4096 16384 32768 --concurrencies 1 4 16 \
  --rounds 3 --num-prompts 64 --output-len 128 --warmups 16 \
  --output-dir /workspace/swa-bench-matrix
```

可用 `--base-url`、`--model`、`--tokenizer-mode`、`--seed`、`--request-rate` 调整连接和负载。
默认 tokenizer mode 为 `deepseek_v41`。`--vllm-bin` 可指定 vLLM 可执行文件。
`--allow-prefix-caching` 只允许保留服务的前缀缓存，不会自动构造重复前缀，
结果也不能直接归因于冷 prefill 裁剪；重复前缀重建收益应另做实验。

## 6 如何读指标

- TTFT：原生客户端收到首个流式输出的时间，包含排队、prefill 等；长输入裁剪首先看这个。
- TPOT：逐请求计算 `(完整耗时 − TTFT) / (输出token数 − 1)`，再统计均值和百分位。
- ITL：沿用原生客户端的流式输出到达间隔。开启投机解码或合并发送时，可能一次收到多个token，
  此时ITL可能是chunk间隔，不能直接当作每个token的计算时间。
- 输出吞吐：正式测试期间完成的输出token总数 / 测试耗时。
- E2EL：完整请求延迟。

报告显示各轮统计量的中位数和最小/最大值，变化为 `(ON / OFF − 1) × 100%`。
延迟负数更好，吞吐正数更好；这是轮间统计，不是把所有轮请求混在一起重新算百分位。
先看成功请求数与实际长度，再看 P50/P95/P99 和波动。可按 OFF→ON→OFF 的顺序复测，
为每次重启保留独立结果目录，排查温度、缓存、预热和其他负载造成的漂移。

该随机负载只用于性能测量，不验证生成质量。脚本已经过模拟服务验证；A3实际收益需实测。
