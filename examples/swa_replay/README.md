# SWA replay curl 测试

测量 TTFT、ITL、TPOT 和 OFF/ON 性能差异，请使用
[单机 streaming benchmark 脚本与报告](BENCHMARK.md)。下列 curl 脚本用于功能检查。

面向当前 CANN 回移分支、DeepSeek V4.1、A3 单机合设。默认访问
`http://127.0.0.1:8989`，服务别名为 `dsv41`，对应现有 DP4/TP4 服务。
只需要 Bash、curl、Python 3 标准库；无需 jq、requests 或额外 Python 包。

## 先跑一个长输入

在 vllm-ascend 源码根目录执行，服务保持运行：

```bash
SERVICE_LOG=/workspace/serve_swa_replay_on.log \
  bash examples/swa_replay/02_long_prefill.sh
```

脚本生成长输入并发送 curl 请求。检查响应中的实际 `prompt_tokens > 128`，
再检查服务日志是否新增以下消息：

```text
SWA decoder bounded replay active (MRV1): ... -> ... tokens (... padded).
```

初始化的 `Ascend decoder bounded replay: layers 21-39 ...` 仅表示已接入。
`active` 每个 worker 只打印首次，后续请求没有新增日志不能单独判定未执行。
一次请求通常只落在一个 DP 实例，不要求所有 DP 同时打印。

## 四个场景

| 脚本 | 请求方式 | 用途 |
| --- | --- | --- |
| `01_short.sh` | 一个短问答 | 检查基础请求、响应和短输入；不要求触发窗口裁剪 |
| `02_long_prefill.sh` | 一个长输入 | 检查真实 Prefill，观察 decoder replay 激活日志 |
| `03_prefix_reuse.sh` | 默认 12 个串行请求，共享长前缀、改变尾部问题编号 | 增加命中前缀的机会，观察 encoder 窗口重建 |
| `04_concurrent.sh` | 默认 4 并发、8 个长请求 | 检查并发请求及 DP 协同是否正常 |

```bash
bash examples/swa_replay/01_short.sh
bash examples/swa_replay/02_long_prefill.sh
bash examples/swa_replay/03_prefix_reuse.sh
bash examples/swa_replay/04_concurrent.sh

# 按上述顺序全部执行，任一场景失败即停止。
bash examples/swa_replay/run_all.sh
```

重复前缀场景需要服务启用 prefix caching。DP4 中请求可能路由到不同 DP，且缓存可能
被淘汰，因此重复请求不保证命中。结合服务端以下日志和响应中可用的
`usage.prompt_tokens_details.cached_tokens` 判断；未返回 cached_tokens 时显示为空。

```text
SWA bounded replay is active: a prefix hit is rebuilding its window.
```

HTTP 成功只表示请求正常完成，不证明 replay 已执行，也不代表通过准确率验收。
这些脚本不修改服务配置，不请求 prompt logprobs，不重置服务缓存。

## 可调整参数

```bash
BASE_URL=http://127.0.0.1:8989 MODEL=dsv41 \
  OUTPUT_DIR=/workspace/swa-on SERVICE_LOG=/workspace/serve_swa_replay_on.log \
  bash examples/swa_replay/run_all.sh

# 加长输入；这是重复段落次数，不是精确 token 数。
LONG_REPEATS=1024 bash examples/swa_replay/02_long_prefill.sh

# 更多共享前缀请求。
PREFIX_REQUESTS=24 bash examples/swa_replay/03_prefix_reuse.sh

# 调整并发和总请求数。
CONCURRENCY=4 NUM_REQUESTS=16 bash examples/swa_replay/04_concurrent.sh
```

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `BASE_URL` | `http://127.0.0.1:8989` | 服务根 URL，不包含 `/v1` |
| `MODEL` | `dsv41` | 服务别名 |
| `OUTPUT_DIR` | `/tmp/swa-curl-results` | 每次执行在该目录下新建一个结果子目录 |
| `SERVICE_LOG` | 未设置 | 可选，只显示本次脚本启动后新增的相关服务日志 |
| `LONG_REPEATS` | `256` | 长背景段落的重复次数，实际 token 数以响应为准 |
| `MAX_TOKENS` | `32` | 最大输出 token 数 |
| `PREFIX_REQUESTS` | `12` | 共享前缀的串行请求数 |
| `CONCURRENCY` | `4` | 并发 worker 数 |
| `NUM_REQUESTS` | `8` | 并发场景的总请求数 |
| `CURL_CONNECT_TIMEOUT` | `10` | curl 连接超时，秒 |
| `CURL_MAX_TIME` | `600` | 单个请求总超时，秒 |
| `PYTHON_BIN` | `python3` | Python 可执行文件 |

## 结果与错误

每次执行创建独立的时间戳目录，包含：

- `request-NNN/request.json`：发送的请求正文。
- `request-NNN/response.body`：完整响应，错误页面或错误 JSON 也保留。
- `request-NNN/curl.json`、`curl.stderr`：HTTP 状态、总耗时、响应大小和 curl 错误。
- `request-NNN/result.json`：usage、回答、finish_reason 和错误信息。
- `summary.json`、`summary.csv`：该场景所有请求的汇总。

网络错误、HTTP 非 2xx、响应格式错误、缺少 usage、长输入实际不超过 128 token，
均返回非零退出码。串行前缀和并发场景会保留已发请求的结果，并汇总失败。
`finish_reason=length` 表示输出达到上限，需检查是否只生成了 reasoning，必要时增大
`MAX_TOKENS`；脚本会原样保存，不把截断直接当作网络失败。

这些请求是非流式请求，`time_total_s` 包含完整生成和 HTTP 往返，**不是 TTFT**。
并发脚本的总体耗时还包含客户端生成请求和收集结果的时间，不用作稳态吞吐基准。
正式性能测量使用 [A3 验证指南](../../docs/source/developer_guide/cann_swa_bounded_replay_backport.md)
中的 `vllm bench serve` 方法。

## OFF/ON 对照

固定 V1、DP4/TP4、DSA-CP、模型、图模式和所有测试变量，只改变服务端
`--no-swa-bounded-replay` / `--swa-bounded-replay`：

```bash
# OFF 服务启动后执行。
OUTPUT_DIR=/workspace/swa-off bash examples/swa_replay/run_all.sh

# 重启同配置服务，改为 ON，完成相同预热后执行。
OUTPUT_DIR=/workspace/swa-on bash examples/swa_replay/run_all.sh
```

要单独看冷 Prefill，两边都显式关闭 prefix caching；要验证前缀重建，两边都开启。
默认请求不包含时间戳，OFF/ON 使用相同参数会生成相同输入。bounded replay 是近似
计算，不以输出逐字相同作为唯一验收标准，应另用业务数据或评测集确认质量。

## 脚本验证范围

已使用本地 HTTP 模拟服务执行实际 Bash、curl 和 Python 脚本，验证四个场景、
共享前缀、并发上限、特殊字符参数，以及 HTTP 500、超时、无效 JSON、空回答和
并发失败的错误返回。检查结果见 [validation.json](validation.json)。
尚未用这些脚本执行 A3 模型推理，不能将模拟服务结果当作真实缓存命中、replay
激活、模型质量或性能收益的证据。
