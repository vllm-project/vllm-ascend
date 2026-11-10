# KV Pool SLO 并发搜索

`slo_search.py` 通过 AISBench 搜索满足延迟和请求成功率要求的最高并发，并保存各轮的延迟、吞吐和缓存命中指标。例如，搜索 **TTFT P90 < 10 s、TPOT P90 < 20 ms** 时的最高并发及对应吞吐。

工具仅依赖 Python 3.9+ 标准库，通过命令行调用外部项目 [aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 中的 `aisbench_test.py`：

```text
slo_search.py → aisbench_test.py → AISBench → 模型服务
```

## 运行前准备

1. 按 [KV Cache Pool 部署指南](../../../../../docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md) 启动模型服务及存储后端，确认压测环境能够访问服务。
2. 单独准备完整的 `aisbench_auto_tools_prefix` 项目及其 AISBench 依赖，配置 `config.py` 中的模型、tokenizer、服务地址、鉴权和数据集路径。前缀测试还需配置缓存指标查询所需的 `POD_INFO` 或 `HOST_IP:HOST_PORT`。
3. 确认执行压测的 Python 能运行 `aisbench_test.py`，且 `ais_bench` 命令在子进程的 `PATH` 中。
4. 确认外部项目及 AISBench 的配置、数据和日志目录可写。外部项目会生成数据、更新配置及数据集软链接；同一份项目不要同时运行多个搜索器或手动压测。

工具只负责发压、搜索和记录，不启动模型服务或配置 KV Pool，也不会清空服务缓存。

以下示例使用 Bash。从 vllm-ascend 仓库根目录进入工具目录：

```bash
cd vllm_ascend/distributed/kv_transfer/utils/slo_search
```

将示例中的 `/path/to/aisbench_auto_tools_prefix` 替换为已配置好的外部项目目录。搜索器可与外部项目放在不同目录；使用不同 Python 环境时，通过 `--python /path/to/python` 指定压测解释器。

## 使用示例

### 按 P90 搜索

使用 2048 token 输入、1024 token 输出，要求 TTFT P90 < 10 s、TPOT P90 < 20 ms，搜索并发上限为 128：

```bash
python slo_search.py \
  --project-dir /path/to/aisbench_auto_tools_prefix \
  --ttft-limit-ms 10000 \
  --tpot-limit-ms 20 \
  --max-concurrency 128 \
  -- --input_len 2048 --output_len 1024
```

独立的 `--` 分隔两类参数：之前是搜索器参数（如 `--max-concurrency`），之后是原样传给 `aisbench_test.py` 的参数（如 `--input_len`）。

默认从并发 8 开始，每轮完整请求数为并发的 4 倍，使用 P90 判定 SLO，不额外复测边界。每轮同时展示 TTFT/TPOT 的平均值和 P90，结束后打印结果汇总及保存目录。

### 前缀缓存：64K 输入、1K 输出

```bash
python slo_search.py \
  --project-dir /path/to/aisbench_auto_tools_prefix \
  --ttft-limit-ms 10000 \
  --tpot-limit-ms 20 \
  --max-concurrency 128 \
  --prefix-num-equals-concurrency \
  -- \
  --input_len 65536 \
  --output_len 1024 \
  --dataset_type prefix_cache \
  --repeat_rate 0.9 \
  --prefix_test
```

`--prefix_test` 由外部项目执行前缀预热；`--prefix-num-equals-concurrency` 让每轮前缀种类数等于并发。例如，并发 8 时默认使用 8 种前缀、32 条完整请求，预热请求不包含在这 32 条中。

`--repeat_rate 0.9` 表示生成数据中公共前缀约占输入的 90%。实际缓存命中率以每轮记录为准。

### 仅约束平均 TTFT：64K 输入、1 token 输出

```bash
python slo_search.py \
  --project-dir /path/to/aisbench_auto_tools_prefix \
  --ttft-limit-ms 10000 \
  --percentile mean \
  --max-concurrency 128 \
  --prefix-num-equals-concurrency \
  -- \
  --input_len 65536 \
  --output_len 1 \
  --dataset_type prefix_cache \
  --repeat_rate 0.9 \
  --prefix_test
```

此例仅用平均 TTFT < 10 s 判定延迟是否达标，仍记录 TTFT P90。单 token 输出的 TPOT 不适用，显示为 `N/A`，JSON 中为 `null`，`tpot_status` 为 `not_applicable_single_token`；此时不能指定 `--tpot-limit-ms`。

### 可选边界复测

在搜索器参数中添加 `--confirm-runs 2`，可对候选边界及邻点分别复测两轮。以“并发 10 通过、11 不通过”、默认 `--boundary-radius 1` 为例，按 11、10、9 的顺序复测，每个点的两轮均通过才确认达标。搜索轮次不计入复测轮数。

若邻点范围内没有确认通过的并发，工具会继续逐点向下复测，直到找到通过点或达到最小并发。设为 `--boundary-radius 0` 时，初始复测范围仅包含候选点。

## SLO 判定与搜索方式

一轮必须满足所有已指定的延迟阈值及最低请求成功率：

```text
指定 TTFT 阈值时：所选统计口径的 TTFT < ttft-limit-ms
指定 TPOT 阈值时：所选统计口径的 TPOT < tpot-limit-ms
请求成功率 >= min-success-rate
```

- 延迟阈值单位为毫秒，严格使用 `<`；TPOT 等于 20 ms 时不满足 `< 20 ms`。默认请求成功率要求为 100%。
- TTFT 和 TPOT 至少指定一个阈值，共用 `--percentile`。可使用 `mean` 或日志实际提供的分位数（如 `90`、`99`），不支持混合统计口径。
- 每轮始终记录平均值和 P90，判定只使用所选口径。两个指标各自满足 P90 阈值，不代表同一批 90% 的请求同时满足两个条件。
- `--stage` 选择汇总表中对应 Stage 的延迟、吞吐和请求统计，默认 `total`；它不改变发压方式或启用稳态测试。前缀预热统计不用于判定。

| 阶段 | 行为 |
| --- | --- |
| `expand` | 从起始并发倍增，直到首次不达标或达到搜索上限 |
| `bisect` | 二分查找通过与不通过的边界；起始点不通过时也会向下搜索 |
| `confirm` | 可选：复测候选边界及邻点，要求每个点的所有复测轮次通过 |

默认 `--confirm-runs 0`，结果来自搜索轮次；启用复测后，结果来自确认通过的最高并发及其复测轮次。搜索依赖近似单调的 SLO 假设，不逐点穷举。比较结果时，应保持输入输出长度、前缀配置、服务状态及统计口径一致。

## 参数参考

完整帮助可通过 `python slo_search.py --help` 查看。

### SLO 与搜索范围

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--ttft-limit-ms` | 未指定 | TTFT 严格上限，单位 ms；省略时不约束 TTFT |
| `--tpot-limit-ms` | 未指定 | TPOT 严格上限，单位 ms；省略时不约束 TPOT |
| `--percentile` | `P90` | 延迟统计口径；平均值用 `mean` |
| `--stage` | `total` | 要读取的 AISBench Stage 名称 |
| `--min-success-rate` | `1.0` | 请求成功率下限，范围 `(0, 1]` |
| `--min-concurrency` | `1` | 最低搜索并发 |
| `--start-concurrency` | `8` | 起始并发 |
| `--max-concurrency` | `1024` | 最高搜索并发 |
| `--confirm-runs` | `0` | 每个点的额外复测轮数，`0` 表示关闭 |
| `--boundary-radius` | `1` | 初始复测范围内候选点左右的整数邻点数 |

并发范围须满足 `1 <= min-concurrency <= start-concurrency <= max-concurrency`。例如，上限设为 4 时，起始并发也须调整为不超过 4。

### 请求量与前缀

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--requests-per-concurrency` | `4` | 每轮完整请求数相对于并发的倍数，须为正整数 |
| `--min-data-num` | `0` | 每轮完整请求数下限 |
| `--data-num` | 未指定 | 固定完整请求数，须不小于最高搜索并发 |
| `--prefix-num-equals-concurrency` | 关闭 | 每轮设置 `--prefix_num` 等于该轮并发 |

未指定 `--data-num` 时，每轮完整请求数为 `max(min-data-num, 当前并发 × requests-per-concurrency)`。

启用前缀数量联动时，须在 `--` 后指定 `--dataset_type prefix_cache`，且不能同时传 `--prefix_num` 或 `--dataset`。未启用联动时，可透传固定的 `--prefix_num`。是否预热由 `--prefix_test` 控制，预热并发由外部项目版本及其配置决定。

使用已有 `--dataset` 时，实际请求量由数据集决定，工具不会自动扩充数据；需自行确保数据量足以测试目标并发。

搜索器统一设置 `--concurrency`、`--request_rate 0`、`--repeat 1` 和 `--data_num`，这些参数不能放在 `--` 后；也不支持透传 `--test_accuracy`。

### 路径与执行控制

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--project-dir` | 脚本所在目录 | 包含 `aisbench_test.py` 的外部项目目录 |
| `--python` | 当前 Python | 执行外部项目的 Python 解释器 |
| `--output-dir` | `<project-dir>/slo_results/<时间戳>` | 本次结果目录，必须尚不存在 |
| `--timeout` | `3600` 秒 | 每轮子进程超时，包含数据准备、预热及完整压测 |
| `--latency-unit` | 未指定 | 仅在日志缺少时间单位时指定 `ms` 或 `s` |

`--timeout` 是单轮超时，不是整个搜索的总时限。任意一轮超时都会终止该轮子进程树并中止搜索。

## 结果与日志

默认结果目录位于外部项目下。显式指定相对 `--output-dir` 时，路径相对于当前工作目录。每次搜索使用新目录，不覆盖已有结果。

```text
<project-dir>/slo_results/<时间戳>/
├── settings.json          搜索配置和透传参数
├── summary.json           最终结论或中止原因
├── history.csv            各轮指标
├── history.jsonl          各轮完整记录及实际命令
└── trial_0001/            后续轮次依次编号
    ├── command.json       本轮执行命令
    ├── console.log        子进程输出及解析成功后的指标汇总表
    ├── metrics.txt        解析成功后的指标汇总表
    └── result.json        本轮指标和判定，或错误原因
```

### 最终结果

```bash
RESULT_DIR=/path/to/result_directory
python -m json.tool "$RESULT_DIR/summary.json"
```

| 字段 | 含义 |
| --- | --- |
| `status` | `complete` 为搜索完成；`aborted` 为中止，原因见 `error` |
| `highest_passing_concurrency` | 最高达标并发；`null` 为未找到达标点 |
| `result_source` | `search` 为搜索轮次，`confirm` 为复测轮次 |
| `active_slo_metrics` | 参与判断的延迟指标，如 `["ttft"]` |
| `confirmation_performed` | 是否启用额外复测 |
| `result_trial_ids` | 最终结果对应的轮次编号 |
| `throughput.*_tokens_per_s_mean` | 结果轮次的输出或 Prefill 平均吞吐，token/s |
| `throughput.requests_per_s_mean` | 结果轮次的平均请求吞吐，req/s |
| `worst_ttft_ms`、`worst_tpot_ms` | 结果轮次中最大的所选延迟统计值，ms |
| `result_cache_metrics` | 结果轮次的缓存指标 |
| `confirmation_trial_ids`、`confirmation_cache_metrics` | 复测结果；未启用复测时为空数组 |
| `at_search_limit` | 是否达到搜索上限；为 `true` 时可提高上限再测 |

无达标点时，不生成吞吐及延迟结果字段。异常中止时，`summary.json` 仅记录中止状态、错误和已记录轮数。

未启用复测时，吞吐来自最终并发的单个搜索轮次；启用复测时，吞吐取最终选中并发的复测均值。“最差 P90”是这些轮次各自 P90 的最大值，不是合并请求后重新计算的 P90。

### 每轮指标

```bash
cat "$RESULT_DIR/history.csv"
cat "$RESULT_DIR/trial_0001/metrics.txt"
```

历史记录包含并发、请求量、阶段、延迟、吞吐、请求成功率和缓存指标。`status` 为 `pass`、`fail` 或 `error`，分别表示达标、不达标和无效轮次；无效轮次会中止搜索。

`ttft_mean_ms`、`ttft_p90_ms`、`tpot_mean_ms`、`tpot_p90_ms` 记录平均值及 P90。`ttft_ms`、`tpot_ms` 为 SLO 所选口径的值，单位均为 ms；其他分位数还会在终端单独展示。

缓存命中率记录为 `hbm_hit_rate_pct`、`external_hit_rate_pct`，范围为 0～100，并保存各自的 `*_hits`、`*_queries` 和 `*_cache_status`：

- 命中率由完整压测期间的汇总命中/查询计数计算，不平均各节点百分比。`external` 保留外部 KV 缓存的含义。
- `no_queries` 表示查询量为零，命中率未定义；`not_reported` 表示日志未提供该指标。两者在终端显示 `N/A`，JSON 为 `null`，CSV 为空。
- 缓存指标对应完整压测区间，即使指定其他 Stage，也不能视为该 Stage 的专属指标。

Prefill 吞吐来自 AISBench 的 `Prefill Token Throughput`。吞吐及缓存指标用于观测，不参与 SLO 判定。

如需保存搜索器整体终端输出，可使用 `set -o pipefail`，并在命令末尾添加 `2>&1 | tee slo_search.log`。各轮子进程输出以 `trial_*/console.log` 为准。

## 故障排查

| 情况 | 处理方式 |
| --- | --- |
| 缺少延迟、吞吐或请求统计 | 检查该轮 `console.log`。日志须包含 TTFT/TPOT 的平均值、P90 及所选分位数，以及 Prefill/Output Token Throughput、Request Throughput、Total Requests、Failed Requests；单 token 输出可无 TPOT |
| 时间单位缺失 | 指定 `--latency-unit ms` 或 `--latency-unit s`；有单位时以日志为准 |
| 缺少或无效的缓存汇总 | 启用 `--prefix_test` 时，须有完整压测的 `Prefix cache hit summary` 及 `ALL_PODS` 中的 HBM/external 命中和查询计数；检查节点配置及服务 `/metrics` 的可访问性 |
| 结果目录已存在 | 使用新的 `--output-dir`，或省略该参数自动生成目录 |
| 项目目录被占用 | `.slo_search.lock` 防止搜索器重叠运行；强制结束后，确认相关进程已停止，再删除遗留锁 |
| 子进程退出、超时或指标解析失败 | 查看 `summary.json` 的 `error` 及该轮 `console.log`；已记录轮次会保留，错误不会作为普通 SLO 不达标继续搜索 |

日志格式须与工具支持的汇总表匹配。所选分位数或 Stage 缺失、日志包含 ERROR/Traceback、缓存节点被跳过或计数异常时，工具会报错，不复用预热结果或其他轮次指标。

| 退出码 | 含义 |
| --- | --- |
| `0` | 搜索完成并找到达标点 |
| `1` | 搜索完成但未找到达标点 |
| `2` | 参数、执行或解析错误 |
