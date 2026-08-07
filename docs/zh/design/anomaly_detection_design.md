# 异常检测（Anomaly Detection）设计

> DFX 总览见 [dfx_design.md](./dfx_design.md)。检测由 `DetectorManager` 编排，实现类在 `vllm_ascend/dfx/detector/`。运维见 [dfx_ops.md](./dfx_ops.md)。

## 1. 通用架构

所有检测器共享同一条通路：

```text
runner.dfx = DfxProcessor(runner)
  ├─ check_after_spec(...)     # DetectorManager → SpecAcceptance（先过 InputFilterManager）
  └─ check_after_sample(...)   # DetectorManager → TokenLogprob / OutputSubstring（先过 InputFilterManager）
        │
        ▼
detector.check_all(...) → list[AnomalyAlert]
  → DfxProcessor._handle_alert → Dumper.handle_anomaly_alert + report
```

共同点：

- **DetectorManager 门面**（`detector/manager.py`）：具体 detector 私有；runner / `DfxProcessor` 只调阶段钩子。
- **InputFilterManager 门控**：每个 detector 在 `check_all` 内 `_passes_input_filter`（`dump.manual_trigger` 除外）。
- **`stop_after_alert`**（`detector.stop_after_alert`，默认 `true`）：某请求一旦检出异常，后续步按 `req_id` 跳过再检（不拆 batch 行），避免反复写 report；`clear_finished` 后重算。设 `false` 则持续重检。
- **只产 `AnomalyAlert`**：detector 不碰 Dumper；arm 由 processor 调 `Dumper.handle_anomaly_alert`。
- **配置**：`detector.<name>.enabled`（默认 `false`）+ 各自阈值，JSON 热更新（广播 / file poll）。
- **与 dump 正交**：detect 不依赖 `dump.enabled` / `max_times`；dump arm 失败仍写 report。
- **完整 I/O**：由 `RequestIoSnapshotManager` 在写 report 时挂一次；detector 只写检测指标 / window 证据。

当前检测器：SpecAcceptance / TokenLogprob / OutputSubstring，以下分节。

## 2. SpecAcceptance（投机接受率）

实现类：`vllm_ascend/dfx/detector/spec_acceptance.py`

| 字段 | 默认 | 说明 |
|------|------|------|
| `detector.spec_acceptance.enabled` | `false` | 投机解码接受率检测 |
| `detector.spec_acceptance.window` | `10` | 每请求滑窗（步数） |
| `detector.spec_acceptance.low_threshold` | `0.3` | 接受率下限 |
| `detector.spec_acceptance.len_low_threshold` | `1.4` | 接受长度下限 |
| `detector.spec_acceptance.high_threshold` | `0.96` | 接受率上限 |
| `detector.spec_acceptance.len_high_threshold` | `2.8` | 接受长度上限 |

- 每请求滑窗统计接受率 / 接受长度，低（异常低接受）或高（异常高接受，疑似作弊）双阈值告警。
- 需要 runner 上存在 `speculative_config`（MTP / Eagle 等）才有效。
- 检测前先 `_record_spec_step_outputs` 把本步 accepted token 计入累计 output（report 用）。

## 3. TokenLogprob（token/logprob 异常）

实现类：`vllm_ascend/dfx/detector/token_logprob.py`。

### 3.1 目标

基于 **输出 token + top-k logprobs** 的在线异常检测（生僻字 / 乱码 / 重复 / NaN），与接受率检测共用动态 msprobe dump 与 DFX Report 通路（完整 I/O 由 `report.save_sensitive_info` 控制；日志只打长度）。

设计原则：

- **尽快检出**：小窗口、低命中阈值；重复略严一点防误报。
- **滑窗外置**：Detector 维护队列；msprobe ILLDetector 配成「一次调用 = 一窗」，避免双重滑窗。
- **可配置开关**：spec acceptance / token_logprob 分别使能（JSON 热更新）。
- **复用 dump**：命中后走 `enable_msprobe_dump_if_needed`，并写 `anomaly_type=token_logprob` 报告。

### 3.2 开关与配置

| 字段 | 默认 | 说明 |
|------|------|------|
| `detector.token_logprob.enabled` | `false` | token/logprob 检测（开启后 worker 自动补齐 top-k logprobs，请求侧可不设 `logprobs`） |
| `detector.token_logprob.window` | `64` | 每请求缓冲长度 = 送检窗长 |
| `detector.token_logprob.stride` | `32` | 满窗后每新增 N token 再检 |
| `detector.token_logprob.topk` | `20` | 每位置最多保留 top20 |
| `detector.token_logprob.ill_nan_window_thresh` | `1` | NaN/Inf 命中窗数 |
| `detector.token_logprob.ill_rare_window_thresh` | `1` | 生僻字 |
| `detector.token_logprob.ill_garbled_window_thresh` | `1` | 乱码 |
| `detector.token_logprob.ill_repet_window_thresh` | `2` | 重复（半重叠两窗确认） |

示例：

```json
{
  "dump": { "enabled": true, "max_times": 3, "cooldown_seconds": 300 },
  "detector": {
    "spec_acceptance": { "enabled": true },
    "token_logprob": { "enabled": true, "window": 64, "stride": 32, "topk": 20, "ill_repet_window_thresh": 2 }
  }
}
```

请求侧**不必**再手动设 `logprobs`：`detector.token_logprob.enabled` 开启时，采样前 `DfxProcessor.ensure_logprobs_for_detection()` 会把 batch 内请求的 top-k 至少抬到 `detector.token_logprob.topk`（默认 20）；若客户端已设更大值则保留。

`--async-scheduling`：token/logprob 在采样返回时尚在 device 上，检测改在 async `get_output()`（D2H 完成并 parse 之后）执行（v1：`AscendAsyncGPUModelRunnerOutput`；v2：`AscendAsyncOutput`）。multiproc 仅 `output_rank`（last-PP TP0）会 `get_output()`，故 **async 仅 TP0 check**；命中后只置 `pending_dump`，下一拍 `execute_model` 入口 last-PP TP `all_reduce(OR)` 后再全体写 `dump_enable`（详见 [dumper_design.md](./dumper_design.md) §5）。

### 3.3 架构与 ILLDetector

```text
TokenLogprobDetector 每请求 deque(maxlen=window)
  满窗 / 之后每 stride 新 token
        │
        ▼
msprobe ILLDetector.detector(topk_dicts, tokens, model_config)
  （内部 window=stride=队列长 → 单窗）
        │
        ▼
按 ill_type 累加命中次数，达 thresh → AnomalyAlert
  → DfxProcessor._handle_alert → Dumper + report
```

**为何不两边各滑一套**：

- msprobe 默认 `window_size=128, stride=64`，且 `single_window_thresh=14` 适合离线长序列。
- 在线：Detector 队列长度 = 窗长；构造 ILLDetector 后覆盖为 `window_size=stride=detector.token_logprob.window`，并把 garbled/repeat 的内部多窗阈值置 0，使 **单次调用能返回 is_ill**。
- **多窗投票改由 `ill_*_window_thresh` 完成**，便于尽快检出且可热更新。

**logprobs 布局**：

- vLLM `LogprobsLists` 每行：`[sampled_logprob, top1, …, topk]`。
- `_row_to_topk_dict`：按 logprob 降序取前 `token_logprob_topk`，转成 `Dict[token_id, logprob]` 再交给 detector。
- MTP / 投机：一步多个 accepted token → 多行 logprobs，按序 append；使用 `cu_num_generated_tokens` 切片。

**model_config / tk2cat**：

- 传入 `{"model_name": Path(model).name}` 供名称模糊匹配。
- `get_tk2cat` 依赖「末 token 为 eos」校验；生成中途常走 **无词表 top1 阈值** 路径。类别增强需预加载 tk2cat（后续优化）。

### 3.4 生命周期与资源

- 每请求缓冲：最多 `window × topk` 个 (id, logprob)。
- 请求结束：`DfxProcessor.clear_finished` → detector `clear_finished` 销毁缓冲与命中计数。
- 检测时日志：`active_reqs`、`ill_type`、hits；报告见 `dfx/report/anomaly_*.log`。

### 3.5 与 dump 共用策略

- 冷却 / 最大次数 / 每请求只 dump 一次：沿用 `enable_msprobe_dump_if_needed`。
- **Async / Sync+TP>1**：仅 TP0 check，arm `pending_dump`；下一步入口 last-PP TP OR 后全体 `_activate`。**Sync+TP=1**：可当场 activate。见 [dumper_design.md](./dumper_design.md) §5。
- 已 `pending` / `dump_active` 时跳过后续 check，避免重复 arm。
- TP0 打详细日志；dump 仅 last PP；状态写 msprobe 配置文件。
- DFX 已移除对上游 `debug_log_full` 的 arming；异常 I/O 见 DFX report（`save_sensitive_info`）。

### 3.6 限制与后续

1. **detect ⊥ dump**：`dump.max_times=0`（或配额用尽）只阻止 **auto-arm dump**；auto 检测与 `ensure_logprobs_for_detection`（top-k）仍按 `detector.token_logprob.enabled` 运行；`manual_trigger` 仍可手动 arm。
2. 中途无 eos → tk2cat 可能不可用。
3. v1 / v2 均已接入：`check_after_sample` + async `get_output` 延迟检测；采样前 `ensure_logprobs_for_detection`。
4. 若需更激进：减小 `window`/`stride`，或将 `ill_repet_window_thresh` 设为 `1`。
5. 跨 TP 齐步与 dump 生命周期详见 [dumper_design.md](./dumper_design.md)；配置广播见 [dfx_design.md](./dfx_design.md)。

## 4. OutputSubstring（输出子串命中）

实现类：`vllm_ascend/dfx/detector/output_substring.py`。

### 4.1 目标

配置驱动的输出命中检测：运维指定若干「不应出现」的文本片段或 token-id 序列，一旦某请求的生成输出包含该子序列即告警。

设计原则：

- **文本或 token 双视图**：pattern 可写成 `str`（文本，自动 encode）或 `list[int]`（token ids，自动 decode 供日志/报告可读）。
- **连续匹配**：对**累计输出 token id** 做匹配（不做逐 token decode），命中即告警。
- **默认子序列 / 可选前缀**：`match_prefix=false`（默认）在输出任意位置找连续子序列；`true` 时仅从输出开头做前缀匹配。
- **每请求一次**：同一 `req_id` 最多告警一次，直到 `clear_finished`（另受共享 `stop_after_alert` 约束）。
- **复用 dump / report 通路**：命中后走 `AnomalyAlert` → dump arm + DFX Report。

### 4.2 开关与配置

| 字段 | 默认 | 说明 |
|------|------|------|
| `detector.output_substring.enabled` | `false` | 输出子串检测 |
| `detector.output_substring.patterns` | `[]` | pattern 列表；每项为 `str`（文本）或非空 `list[int]`（token ids） |
| `detector.output_substring.add_special_tokens` | `false` | 文本 pattern encode 时是否加 special tokens |
| `detector.output_substring.match_prefix` | `false` | `false`=任意位置子序列；`true`=仅输出开头前缀匹配 |

示例：

```json
{
  "dump": { "enabled": true, "max_times": 3 },
  "detector": {
    "output_substring": {
      "enabled": true,
      "patterns": ["[CENSORED]", [100, 200, 300]],
      "add_special_tokens": false,
      "match_prefix": false
    }
  }
}
```

- 文本 pattern：config 刷新时用 tokenizer encode 成 token ids，日志打印 `source=text` 双视图。
- token-id pattern：刷新时 decode 成文本用于日志/报告，匹配仍按 token ids。
- **Tokenizer**：编译 pattern 需要模型 tokenizer（懒加载；`tokenizer_provider` 或共享 `load_model_tokenizer`）。尚未就绪时 pattern 暂缓编译，就绪后自动重试；encode/decode 失败的单项 pattern 跳过并打 warning。

### 4.3 架构与调用链

```
DfxProcessor.check_after_sample
  └─ DetectorManager.check_after_sample
       ├─ append_batch: 本步采样 token 写入 RequestIoSnapshotManager 累计 output
       ├─ TokenLogprobDetector.check_all
       └─ OutputSubstringDetector.check_all(sampled_token_ids=None)
            ├─ 用累计 output（避免二次 append）
            ├─ 对每个 req：按 match_prefix 做前缀或子序列匹配
            └─ 命中 → AnomalyAlert(anomaly_type="output_substring")
                 → _handle_alert → Dumper.handle_anomaly_alert + report
```

- 匹配用的是**完整累计 output**（`RequestIoSnapshotManager` 自建累计列表，async 安全）。
- **Report 截断注意**：检测用完整序列，但 report 里 `output_token_ids` 受 `report.max_output_token_ids`（默认 1000）从头部截断。若命中片段在截断窗口之外，JSON 里看不到该 id 不等于没命中——核对全量设 `max_output_token_ids: 0`（详见 [dfx_ops.md](./dfx_ops.md) §2.2.1）。
- 每请求告警一次：`_alerted` 集合在 `clear_finished(req_id)` 时清除。

### 4.4 Report 字段

| 字段 | 说明 |
|------|------|
| `matched_pattern_index` | 命中的 pattern 在 `patterns` 中的下标 |
| `matched_source` | `"text"` 或 `"token_ids"`（pattern 的原始写法） |
| `matched_text` | pattern 的文本视图 |
| `matched_token_ids` | pattern 的 token-id 视图 |
| `match_mode` | `"prefix"` 或 `"subsequence"`（对应 `match_prefix`） |
| `output_token_count` | 当前累计输出长度 |

## 5. 代码落点

| 模块 | 说明 |
|------|------|
| `vllm_ascend/dfx/runtime_config.py` | JSON 热更新 / broadcast；`detector` + `dump` 段 |
| `vllm_ascend/dfx/processor.py` | 编排 check / report；刷新 `InputFilterManager` |
| `vllm_ascend/dfx/detector/manager.py` | `DetectorManager` 阶段钩子门面 |
| `vllm_ascend/dfx/detector/spec_acceptance.py` | 投机接受率 |
| `vllm_ascend/dfx/detector/token_logprob.py` | token/logprob（ILLDetector） |
| `vllm_ascend/dfx/detector/output_substring.py` | 输出子串命中 |
| `vllm_ascend/dfx/input_filters.py` | detect 输入过滤（`InputFilterManager`） |
| `vllm_ascend/dfx/io_snapshot.py` | `RequestIoSnapshotManager` 累计 output / report 挂 I/O |
| `vllm_ascend/dfx/dumper/` | dump 生命周期、pending-OR（无检测转发） |
| `vllm_ascend/dfx/report.py` | 短报告落盘 |
| `vllm_ascend/worker/model_runner_v1.py` / `v2` | sync / async 调用点 |
| `docs/source/user_guide/configuration/additional_config.md` | 用户配置表 |

## 6. 相关文档

- 总览与配置：[dfx_design.md](./dfx_design.md)
- 运维与排障：[dfx_ops.md](./dfx_ops.md)
- dump 生命周期 / PP-TP 齐步：[dumper_design.md](./dumper_design.md)
- 异步调度时序：[async_scheduling_design.md](./async_scheduling_design.md)
