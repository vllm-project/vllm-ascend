# DFX 方案说明（vllm-ascend）

> 设计 for eXcellence：运行时维测控制面。  
> 代码根目录：`vllm_ascend/dfx/`

## 1. 组件与流程

| 组件 | 模块 | 职责 |
|------|------|------|
| 1. Runtime Config | `runtime_config.py`（`DfxRuntimeConfig`） | 一份 JSON；可选热更新（启动项控制周期） |
| 2. Detector | `detector/` | 异常检测，只产出 `AnomalyAlert` |
| 3. Dump / 观测开关 | `dumper/`（`Dumper` + mixins） | msprobe dump 生命周期（**可选依赖**：缺包则软失败并强制 `dump.enabled=false`）；`ascend_log` 开关 |
| 4. Report | `report.py`（`DfxReportWriter`） | 异常短日志落盘到 `dfx/report/` |
| 5. Processor | `processor.py`（`DfxProcessor`） | runner 侧编排（构造 / refresh / check / report）；刷新 `InputFilterManager` |
| 6. Input filter | `input_filters.py`（`InputFilterManager`） | detect 前输入过滤（单例；`manual_trigger` 不走） |
| 7. I/O snapshot | `io_snapshot.py`（`RequestIoSnapshotManager`） | report 时挂 prompt/output（单例；非 model_runner） |

对外入口：`from vllm_ascend.dfx import Dumper`（以及 `DfxProcessor` / `DfxRuntimeConfig` 等）。

```text
additional_config
  ├─ dfx_config_path / dfx_config_reload_interval
  └─ AscendConfig.dfx_config (DfxRuntimeConfig)
         │
Worker: runner.dfx = DfxProcessor(runner)
  execute_model 入口：dfx.sync_for_step()   # 内部拆两段（勿合并）
  ├─ refresh_config()              # 全 rank；热更关则立刻 return
  └─ sync_dump_pending_or()        # 仅 last-PP TP
         │
采样 / get_output
  dfx.clear_finished / check_after_spec / check_after_sample
    → DetectorManager → detector.check_all → AnomalyAlert
    → dumper.handle_anomaly_alert  # 只管 dump
    → report_writer.write          # Report
```

> 注意：检测由 **processor → DetectorManager** 调具体 detector，再用 alert 调 dumper；runner **不**看见具体 detector；detector **不**直接 `enable_dump`。  
> Config / Report 也不应塞进 dumper 的 dump OR 路径，否则有人「优化跳过 early PP 的 config sync」会让**同步组内** collective 卡死。

### Report / Dump 关系补充

- `Dumper` 只负责 arm / activate dump，不负责决定是否写 report。
- `DfxProcessor` 在异常检测命中或 `manual_trigger` 命中后，都会尝试调用 dumper；**无论 dump 是否成功 arm，都会写 report**，避免 dump 配额耗尽、冷却中或相关性校验失败时丢失异常证据。
- report JSON 含 `dump_armed` / `dump_attempted` / `dump_capture_timing` / `dump_count` / `dump_max_times`；成功 arm 时文件名带 `_dump`。`dump_capture_timing=upcoming_forward_window` 表示 activate 之后的某次 dump-forward 采集（pending-OR 下相对 detect 常是下下个窗口）。
- `report.print_sampling_meta` 仍只控制采样参数日志打印，不会把 sampling meta 额外写入 report 内容。

## 2. Runtime Config

### 2.1 路径

| 优先级 | 来源 | 路径 |
|--------|------|------|
| 1 | `additional_config.dfx_config_path` 或 `dfx-config` | 显式路径 |
| 2 | 默认 | `<cwd>/dfx/config/dfx_config.json` |

启动热更新开关（权威，JSON 不能重新打开）：

| 参数 | 默认 | 含义 |
|------|------|------|
| `dfx_config_reload_interval` | `0` | 启动项（也可写入 JSON 的 `reload_interval_seconds` 供查看）。默认 `0`=关闭周期刷新；`>0`=每隔 N 秒热更。**以启动 `additional_config` 为准**，仅改 JSON 不能重新打开 |

报告目录默认：与 config 同级的 `dfx/report/`（可用 `dfx_report_dir` 覆盖）。  
带注释示例：`vllm_ascend/dfx/templates/dfx_config.example.jsonc`。  
首次落盘按代码 `_DEFAULTS` 写严格 JSON（`input_filter.filters: []`）。

### 2.2 同步模式 `sync_mode`

| 值 | 行为 | 适用 |
|----|------|------|
| `broadcast`（**默认**） | **每个 EngineCore / DP 一个 leader** 监测并读写 JSON；组内 `all_reduce(due)` + `broadcast_object` | 多机无共享盘：改本 EngineCore leader 节点上那一份即可 |
| `file` | 每进程按启动参数间隔轮询本地/共享路径 mtime | 共享文件系统 |

#### 多 DP 安全模型（完善后）

```text
每个 DP / EngineCore
  └─ 1 个 JSON writer（inner_dp 首 rank，或 TP0∧PP0）
       └─ 监测 dfx_config.json
            ├─ 有 inner_dp_world → 只在本 DP 内 broadcast（不同步到其它 DP）
            └─ 无 inner_dp_world → 本机/本路径 file poll（各 DP 各一份或共享盘）
禁止：跨 DP 满编 world（如 EP 32）做 config collective
```

- 必须在**同步组内所有 rank 同一拍**调用 `dfx.sync_for_step()`（已挂 `execute_model` / idle `execute_dummy_batch`）。
- **禁止**满编跨 DP `world` 热更：请求结束后常出现「仅一侧 EngineCore 再 dummy」→ 会死锁。
- **跨 DP 不自动同步 config**：要两边生效就改两边各自可读的 JSON（或共享盘同一路径）。
- **禁止**把 config sync 折叠进「仅 last-PP」的 dump 路径；也**不要**塞进 `_dummy_run`。
- `save()` / `manual_trigger` 清盘：仅 JSON writer；非 writer 忽略写盘。

### 2.2.1 非 worker（API / EngineCore）

Detector / dump / report **只跑在 worker**。

`ascend_log` 级别：`AscendConfig` 构造时即 `apply_ascend_log_level`（含 API/EngineCore）。当 `dfx_config_reload_interval > 0` 且进程 **未**设置 `RANK` 时，另启动守护线程 `dfx-non-worker-reload`，按间隔 **本地 file 轮询** JSON 并在变更后再次 `apply_ascend_log_level`（**不**进 worker broadcast，**不**落盘）。Worker 在 `Dumper` 初始化 / `refresh_config` → `apply_dfx_config` 时同样应用。

Worker 仍走 `execute_model` / idle `execute_dummy_batch` → `sync_for_step`；**不要**在 worker 上再起并行热更线程。

### 2.2.2 外部多 engine DP

产品约定（与上表一致）：

1. **每套 engine 各一份 JSON**（推荐）：各 DP leader 监测并（有 `inner_dp` 时）只在本 DP 内广播；运维改各自路径；
2. **`file` / file-poll 降级 + 共享盘**：所有引擎轮询同一共享路径；或每节点一份同名文件、改哪台哪台生效。

### 2.3 JSON 结构

```json
{
  "sync_mode": "broadcast",
  "reload_interval_seconds": 0,
  "dump": {
    "enabled": false,
    "max_times": 0,
    "cooldown_seconds": 300,
    "manual_trigger": false
  },
  "ascend_log": { "level": "INFO", "debug": [] },
  "report": {
    "save_sensitive_info": false,
    "print_sampling_meta": false,
    "decode_token_ids": true,
    "max_prompt_token_ids": 1000,
    "max_output_token_ids": 1000
  },
  "detector": {
    "stop_after_alert": true,
    "spec_acceptance": {
      "enabled": false,
      "window": 10,
      "low_threshold": 0.3,
      "len_low_threshold": 1.4,
      "high_threshold": 0.96,
      "len_high_threshold": 2.8
    },
    "token_logprob": {
      "enabled": false,
      "window": 64,
      "stride": 32,
      "topk": 20,
      "ill_nan_window_thresh": 1,
      "ill_rare_window_thresh": 1,
      "ill_garbled_window_thresh": 1,
      "ill_repet_window_thresh": 2
    },
    "output_substring": {
      "enabled": false,
      "patterns": [],
      "add_special_tokens": false,
      "match_prefix": false
    }
  },
  "input_filter": {
    "filters": [],
    "print_input_token_ids_once": false
  }
}
```

| 段 | 含义 |
|----|------|
| `dump` | `enabled`（默认 `false`，dump sink）/ `max_times`（仅 auto-arm）/ `cooldown_seconds`；`manual_trigger` 见运维页。与 detector 正交 |
| `ascend_log` | `level`：`vllm_ascend` 包根 logger 级别。`debug`：模块白名单（相对路径，如 `["dfx"]` → `vllm_ascend.dfx`）强制 DEBUG。走 Ascend 专用 handler（不受 `VLLM_LOGGING_LEVEL` 的 `vllm` handler 过滤）。无 `enabled` |
| `report` | `save_sensitive_info`：默认 `false` 只存 `*_token_count`（不写 token ids）；`true` 时落盘 id（受 `max_*` 截断）并可 decode。`decode_token_ids`：默认 `true`，`save_sensitive_info=true` 时把 `*_token_ids` decode 成 `*_text` / 逐步 `*_texts`。`max_prompt_token_ids` / `max_output_token_ids` 默认 1000，`0`=不截断。`print_sampling_meta`：默认 `false` |
| `detector` | 共享 `stop_after_alert`（默认 `true`：某请求一旦检出异常即停止检测该请求，防止同一异常反复写 report）+ 各检测器嵌套段（`spec_acceptance` / `token_logprob` / `output_substring`），每段含 `enabled` 与阈值 |
| `input_filter` | `filters` 见 §2.6；`print_input_token_ids_once` 见运维 §2.3 |

### 2.4 配置加载与热更

- **唯一配置源**：DFX JSON（`dfx_config_path` 或默认 `<cwd>/dfx/config/dfx_config.json`）+ 热更。Detector / `Dumper` 只读 `ascend_config.dfx_config`。
- **启动引导**（**仅 worker leader 落盘一次**；API/EngineCore/`init_ascend_config` 只做内存合并）：
  - **未**配 `dfx_config_path` → 内存用 `_DEFAULTS`（忽略旧默认路径内容）；worker leader `ensure_persisted` 覆盖写出；
  - **已**配路径 → `defaults ← JSON`；leader 若文件已存在则不盲目重写，文件缺失时写出；
  - 启动日志打印最终 `path=`（AscendConfig + worker Processor）。
- **热更合并**：仅 `defaults ← JSON`。
- **`manual_trigger`**：依赖热更；`dfx_config_reload_interval` 必须 `> 0`。操作与排障见 [dfx_ops.md](./dfx_ops.md)。

### 2.5 命名

| 推荐 | 说明 |
|------|------|
| `DfxRuntimeConfig` / `runtime_config.py` | 运行时热更新控制面 |
| `DfxProcessor` / `processor.py` | runner 侧编排（构造 dumper/detectors、check/clear/report） |
| `InputFilterManager` / `input_filters.py` | detect 输入过滤单例 |
| `RequestIoSnapshotManager` / `io_snapshot.py` | report 时 prompt/output 快照单例 |

### 2.6 InputFilterManager（detect 输入过滤）

配置在顶层 `input_filter.filters`（**检测阶段**；不挡 dump arm / `manual_trigger`）。

**术语**：一条 *filter config* = JSON 里一个 `{type, mode, …}` 对象；  
`DfxRuntimeConfig.input_filter_configs()` 返回校验后的 config 列表，再交给  
`InputFilterManager.apply_configs` 建成运行时 `InputFilter` 链（config ≠ Filter 实例）。

| 规则 | 说明 |
|------|------|
| 空链 `filters: []` | 不过滤，全部请求可检测 |
| `mode=include` | 链上**全部** include 须命中 |
| `mode=exclude` | **任一** exclude 命中则拒绝 |
| 缺 prompt | 过滤已配置时默认拒绝（安全） |
| `manual_trigger` / `ManualTriggerManager` | **不**调用 `InputFilterManager` |
| 刷新入口 | 仅 `DfxProcessor` init + `refresh_config` → `InputFilterManager.apply_from_config` |

支持的 `type`：

| type | 主要字段 |
|------|----------|
| `input_token_id_prefix`（别名 `prefix`） | `prefixes: [[id…], …]`，OR |
| `prompt_length`（别名 `length`） | `op`: `gt\|gte\|lt\|lte\|eq\|between` + `value` 或 `min`/`max` |
| `prompt_contains_token_ids`（别名 `contains_token_ids` / `contains`） | `token_ids` + `match`: `any` \| `subsequence` |

前缀匹配只用 `type: input_token_id_prefix`（无单独 `input_token_id_prefixes` 字段）。

示例：

```json
"input_filter": {
  "filters": [
    { "type": "input_token_id_prefix", "mode": "include", "prefixes": [[151644, 872]] },
    { "type": "prompt_length", "mode": "include", "op": "gte", "value": 64 },
    { "type": "prompt_contains_token_ids", "mode": "exclude", "token_ids": [0], "match": "any" }
  ],
  "print_input_token_ids_once": false
}
```

Detector 通过基类 `_passes_input_filter` → `InputFilterManager.get().allow(...)`。运维见 [dfx_ops.md](./dfx_ops.md) §2.4。

## 3. Detector

| 类 | 文件 | `anomaly_type` |
|----|------|----------------|
| `DetectorManager` | `detector/manager.py` | 阶段 hook 门面（隐藏具体 detector） |
| `AnomalyDetector`（基类） | `detector/base.py` | — |
| `ConfigBackedDetector` | `detector/config_backed.py` | — |
| `SpecAcceptanceDetector` | `detector/spec_acceptance.py` | `spec_acceptance` |
| `TokenLogprobDetector` | `detector/token_logprob.py` | `token_logprob` |
| `OutputSubstringDetector` | `detector/output_substring.py` | `output_substring` |

## 3.1 Manual Trigger

| 类 | 文件 | `trigger_type` |
|----|------|----------------|
| `ManualTriggerManager` | `manual_trigger.py` | `manual_trigger` |
| `TriggerEvent` | `manual_trigger.py` | 控制面事件结构 |

基类约定：

- `DetectorManager`：构造并私有持有各 detector；对外仅 `check_after_spec` / `check_after_sample`
- `AnomalyDetector.refresh_from_config()`：默认空实现；`_precheck` 每 check 调用一次
- `ConfigBackedDetector`：从 live `DfxRuntimeConfig.detector.<section_key>` 拉 `enabled` + `_apply_detector_values`（Spec / Token / OutputSubstring 等阈值型 detector）
- `check_all` / `check_one`：返回 `list[AnomalyAlert]` / `AnomalyAlert | None`（**不**调用 Dumper）
- `on_alert_armed(alert)`：dump 成功后的可选日志钩子
- **Spec 检测条件**：runner 上存在 `speculative_config`（MTP/Eagle 等），**不**依赖仅 hybrid/Mamba 才置位的 `need_accepted_tokens`
- **OutputSubstring**：`detector.output_substring.patterns` 支持 `str`（text）或 `list[int]`（token ids）；config 刷新时 encode/decode 并打日志；默认对累计 output **token id 连续子序列**匹配；`match_prefix: true` 改为**前缀匹配**（仅从输出开头匹配）；每 `req_id` 告警一次；report `detail` 含 `matched_text` / `matched_token_ids` / `match_mode`（`prefix` 或 `subsequence`）

调用链（``DfxProcessor`` 编排，runner 只挂接阶段 hook）：

```text
runner.dfx = DfxProcessor(runner)
  ├─ sync_for_step()  # = refresh_config() + sync_dump_pending_or()
  ├─ refresh_config() 中 ManualTriggerManager.consume_once() -> TriggerEvent
  │     └─ _handle_manual_trigger -> dumper.handle_manual_trigger
  ├─ clear_finished / check_after_spec / check_after_sample
  │     └─ DetectorManager（内部 Spec / Token / OutputSubstring）
  └─ _handle_alert → (dump.on? arm dump : on_alert_armed) + report per dump/report policy
```

> 注意：检测由 **processor → DetectorManager** 调具体 detector，再用 alert 调 dumper；runner **不**直接 `enable_dump` / 不引用具体 detector 类。  
> `save_sample_param` 由 ``report.print_sampling_meta`` 控制（TP0 && last PP），不属于 dump 生命周期。

`AnomalyAlert`（`detector/alert.py`）对齐 msprobe `ILLDetector.detector(...)` 的 `is_ill` / `ill_type`，并带上 dump/report 元数据。

细节：

- 检测器（SpecAcceptance / TokenLogprob / OutputSubstring）：[anomaly_detection_design.md](./anomaly_detection_design.md)
- dump 齐步：[dumper_design.md](./dumper_design.md)
- Async 时序：[async_scheduling_design.md](./async_scheduling_design.md)
- 运维 / `manual_trigger` / 过滤：[dfx_ops.md](./dfx_ops.md)

## 4. Dump（Dumper）

职责：debugger 生命周期、pending OR 齐步、start/finalize 配对、接 `AnomalyAlert`。  
实现位置：`vllm_ascend/dfx/dumper/`（`core` + `msprobe` + `pending`；对外 `from vllm_ascend.dfx.dumper import Dumper`）。

每步入口（runner → ``DfxProcessor.sync_for_step()``，内部拆两段）：

0. `dfx.sync_for_step()` = `refresh_config()` + `sync_dump_pending_or()`
1. `refresh_config()` → `sync_dfx_config()`（仅当 `dfx_config_reload_interval > 0`；**全 rank**）+ 刷新 `InputFilterManager`
2. 若 config 变更：`dumper.apply_dfx_config()`（dump 限额 / `ascend_log`）；并在 `refresh_config()` 统一执行 `ManualTriggerManager.consume_once()`（`consume_quota=False`）
3. `sync_dump_pending_or()`（仅 last-PP TP；**不含** config / report）。热更关且 `dump.enabled=false` 时走 fast-path，跳过 TP OR（见 [dumper_design.md](./dumper_design.md) §5）；**同一 EngineCore 内各 TP 的启动 `dump.enabled` 必须一致**，否则可能一侧 skip、一侧 `all_reduce` 挂死。

Dumper **不**调用 config reload，也 **不**写 report，也 **不**刷新 `InputFilterManager`（均在 processor）。

门控（异常检测）：至少一路 `detector.<name>.enabled` 开，且 rank 合法；**不依赖** `dump.enabled` / `max_times`。另受 §2.6 过滤。`dump.enabled=true` 且 dump pending/active 时跳过再检，避免重叠 arm。  
门控（自动 dump）：`dump.enabled` + 配额 / 冷却；`max_times == 0` 时不 auto-arm（detect / `manual_trigger` 仍可）。  
约束：detect 与 dump **正交** — 可 detect-only、detect+dump、或 **manual-only**（`dump.enabled=true`、无 detector，仅 `manual_trigger`）。无 detector 时 dump 开只会 warn，不强制改配置。  
`manual_trigger` 由 `ManualTriggerManager` 消费；要求 `dump.enabled` + debugger，**不要求** detector；不受 `max_times` / cooldown / InputFilterManager 限制。`dump.enabled=false` 时不消费 flag。  
`manual_trigger` arm 成功后写 **一份** `manual_trigger` report：`detail.requests[]` 快照当前 batch **全部**请求的 I/O（与单请求 anomaly report 不同；普通异常触发 dump 仍只写单请求 report）。  
**前提**：`additional_config.dfx_config_reload_interval > 0`（热更为关时改 JSON 的 `manual_trigger` 不会生效）。

## 5. Report

- 类：`DfxReportWriter`
- 目录：默认 `<dfx_root>/report/`
- 文件：`anomaly_YYYYMMDD_HHMMSS_mmm[_dump]_pid<pid>.log`（毫秒 + pid；成功 arm dump 时带 `_dump`；**格式化 JSON**，每文件一条；含 `dump_armed` / `dump_attempted` / `dump_capture_timing` / `dump_count` / `dump_max_times`）
- 何时写：
  - `dump.enabled=false`（detect-only）：检测到异常即写 report（无 `_dump`）
  - `dump.enabled=true`：尝试 arm dump；**无论 arm 成功与否都写 report**（arm 失败——如配额耗尽——不吞掉检测证据；仅成功时文件名带 `_dump`）
- 默认：`report.save_sensitive_info=false` 时只保留 `prompt_token_count` / `output_token_count`（及非 token 字段），**不**写 `*_token_ids`；`true` 时明文保存（可截断）prompt / 累计 output / window ids，并可 `decode_token_ids` 写出对应 `*_text` / 逐步 `*_texts`
- `max_prompt_token_ids` / `max_output_token_ids`（默认 1000，`0`=不截断）：只限制**落盘**的 id 列表长度（从**头部**截断）；`*_token_count` 仍是完整长度。截断时 detail 会带 `*_token_ids_truncated=true` 与 `*_token_ids_max`
- **易踩坑**：检测 / 命中匹配用的是完整累计 output；report 里的 `output_token_ids` 可能被截断。若 `matched_token_ids` 在完整序列中、却在截断窗口之外，JSON 里会看不到该 id——查全量请设 `max_*=0`，或对照 `*_token_count` 与 `*_truncated`
- JSON 格式化：结构缩进，但 **int 数组（token ids）单行紧凑**，避免「一数字一行」
- 完整 `prompt_token_ids` / `output_token_ids` 由 **`RequestIoSnapshotManager`**（单例，自建累计 output）在 **写 report 时** 挂一次；detector 只写检测指标 / window 证据，不各自拷贝全量 I/O
- Detector / dump 日志只打长度，不打印 token ids；完整 I/O 仅走 report
- `report.print_sampling_meta`：可选 `[SamplingMeta]` 日志（TP0+last-PP）
- DFX **不再** arm 上游 `debug_log_full`（vLLM 字段可空转）

示例（pretty JSON，token ids 单行）：

```json
{
  "ts": "2026-07-28T11:00:00",
  "unix_ts": 1753666800.0,
  "anomaly_type": "spec_acceptance",
  "req_id": "req-1",
  "rank": "tp0-pp1",
  "save_sensitive_info": false,
  "detail": {
    "acceptance_rate": 0.1,
    "prompt_token_count": 128,
    "output_token_count": 64
  }
}
```

检测触发 dump 成功时由 **DfxProcessor** 追加一行（`report_writer`）。

## 6. 非 worker 与多 engine

### 6.1 非 worker（API / EngineCore）

Detector / dump / report **只跑在 worker**。  
`ascend_log`（`level` + `debug`）：`AscendConfig` 启动时应用一次；热更开启且无 `RANK` 时由后台 file 轮询线程在 JSON 变更后再次应用（见 §2.2.1）；worker 在 config sync 后经 `DfxRuntimeConfig.apply_ascend_log_level` 委托 `vllm_ascend.logger.apply_ascend_log_level`。

### 6.2 外部多 engine DP

产品约定二选一：

1. **每套 EngineCore / DP 一份 JSON**（`broadcast` 或 file-poll 降级）；
2. **`file` + 共享盘**（所有引擎轮询同一路径）。

## 7. 启动示例

```bash
# 多机：每 EngineCore 可读的 JSON（可共享盘一份，或每节点一份）；开启 5s 热更新
vllm serve <model> --additional-config '{
  "dfx_config_path": "/data/dfx/config/dfx_config.json",
  "dfx_config_reload_interval": 5,
  "dump_config_path": "/data/msprobe_dump.json"
}'
```

或默认路径（进程 cwd 下自动创建；默认不开启热更，需显式设 `dfx_config_reload_interval > 0`）：

```text
./dfx/config/dfx_config.json
./dfx/report/anomaly_YYYYMMDD_HHMMSS_mmm[_dump]_pidXXXX.log
```

显式配置 `dfx_config_reload_interval > 0` 后，在线改 `dump.max_times` / detector 阈值约 N 秒内各 worker 生效（broadcast）。

## 8. 相关文档

| 文档 | 内容 |
|------|------|
| [dfx_ops.md](./dfx_ops.md) | 运维操作与排障 |
| [dumper_design.md](./dumper_design.md) | Dump 生命周期、PP/TP 齐步、调用链 |
| [anomaly_detection_design.md](./anomaly_detection_design.md) | 异常检测：SpecAcceptance / TokenLogprob(ILLDetector) / OutputSubstring |
| [async_scheduling_design.md](./async_scheduling_design.md) | 异步调度下的时序与 OR（机制真相以 dumper 为准） |
| 用户配置 | `docs/source/user_guide/configuration/additional_config.md` |
