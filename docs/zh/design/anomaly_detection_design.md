# 异常检测（Anomaly Detection）设计

> DFX 总览见 [dfx_design.md](./dfx_design.md)。检测由 `DetectorManager` 编排，实现类在 `vllm_ascend/dfx/detector/`。运维见 [dfx_ops.md](./dfx_ops.md)。

## 1. 通用架构

所有检测器共享同一条通路：

```text
runner.dfx = DfxProcessor(runner)
  ├─ check_after_spec(...)     # DetectorManager → SpecAcceptance（先过 InputFilterManager）
  └─ check_after_sample(...)   # DetectorManager → TokenLogprob / OutputSubstring / TokenRepeat（先过 InputFilterManager）
        │
        ▼
detector.check_all(...) → list[AnomalyAlert]
  → DfxProcessor._handle_alert → Dumper.handle_anomaly_alert + report
```

共同点：

- **DetectorManager 门面**（`detector/manager.py`）：具体 detector 私有；runner / `DfxProcessor` 只调阶段钩子。
- **RequestDfxStore（per-req 共享态）**：跨模块的请求内存态集中在 `RequestDfxState`（累计 `output_token_ids`、filter allow 缓存、sample-wave FIFO、`stopped_after_alert`、committed `dump_finish` meta）。`mark_finished` 打标；真正 sidecar + `Store.clear` 在 sample-wave 队列排空后（`check_after_sample` / 下一波 `sync_for_step`）`_reap_finished_requests`。超时 `max_deferred_waves`（默认 8）强制 reap。detector 私有缓冲仍各管各的，但清理入口统一。
- **RequestIoSnapshotManager（报告 I/O 视图）**：不是第二套请求状态。职责是 normalize→写入 Store、以及 `snapshot()` 拼 report 字段；仅保留同 wave 的 snapshot 缓存（经 `register_on_clear` 挂到 Store，失败打 error 日志）。
- **InputFilterManager 门控**：每个 detector 在 `check_all` 内 `_passes_input_filter`（`dump.manual_trigger` 除外）。
- **`stop_after_alert`**（`detector.stop_after_alert`，默认 `true`）：某请求一旦检出异常，后续步按 `req_id` 跳过再检（不拆 batch 行），避免反复写 report；reap / `Store.clear` 后重算。设 `false` 则持续重检。
- **只产 `AnomalyAlert`**：detector 不碰 Dumper；arm 由 processor 调 `Dumper.handle_anomaly_alert`。
- **配置**：`detector.<name>.enabled`（默认 `false`）+ 各自阈值，JSON 热更新（广播 / file poll）。
- **与 dump 正交**：detect 不依赖 `dump.enabled` / `max_times`；dump arm 失败仍写 report。
- **累计 output 流**：数据在 Store；**仅** `check_after_sample` 经 `append_batch` 写入（MTP/Eagle 的 accepted tokens 也走这条路径，避免与 `check_after_spec` 双写）；`OutputSubstring` / `TokenRepeat` 在 sample 阶段以 `sampled_token_ids=None` 读同一条累计流。

当前检测器：SpecAcceptance / TokenLogprob / OutputSubstring / TokenRepeat，以下分节。

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
- **不**再往累计 IO 写 token；report 用的 output 由后续 `check_after_sample` 单次 `append_batch` 写入（避免 MTP 双写）。

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
| `detector.token_logprob.enabled` | `false` | token/logprob 检测（开启后 worker 自动补齐 top-k logprobs，请求侧可不设 `logprobs`）。**依赖 msprobe**；未安装时 ERROR 并强制改回 `false`；装包后改回 `true` 热更可 lazy 重试 ILLDetector |
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
- 请求结束：`DfxProcessor.mark_finished`；`sample_waves` 排空后 `_reap_finished_requests` → `Store.clear` + detector `clear_finished`。
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
- **每请求一次**：同一 `req_id` 最多告警一次，直到 reap / `Store.clear`（另受共享 `stop_after_alert` 约束）。
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
       ├─ OutputSubstringDetector.check_all(sampled_token_ids=None)
       │    └─ 累计 output 上做前缀 / 子序列匹配 → AnomalyAlert(output_substring)
       └─ TokenRepeatDetector.check_all(sampled_token_ids=None)
            └─ 累计 output 增量 fold → AnomalyAlert(token_repeat)
                 → _handle_alert → Dumper + report
```

- 匹配用的是**完整累计 output**（`RequestIoSnapshotManager` 自建累计列表，async 安全）。
- **Report 截断注意**：检测用完整序列，但 report 里 `output_token_ids` 受 `report.max_output_token_ids`（默认 1000）从头部截断。若命中片段在截断窗口之外，JSON 里看不到该 id 不等于没命中——核对全量设 `max_output_token_ids: 0`（详见 [dfx_ops.md](./dfx_ops.md) §2.2.1）。
- 每请求告警一次：`_alerted` 集合在 reap 时随 detector `clear_finished(req_id)` 清除。

### 4.4 Report 字段

| 字段 | 说明 |
|------|------|
| `matched_pattern_index` | 命中的 pattern 在 `patterns` 中的下标 |
| `matched_source` | `"text"` 或 `"token_ids"`（pattern 的原始写法） |
| `matched_text` | pattern 的文本视图 |
| `matched_token_ids` | pattern 的 token-id 视图 |
| `match_mode` | `"prefix"` 或 `"subsequence"`（对应 `match_prefix`） |
| `output_token_count` | 当前累计输出长度 |

## 5. TokenRepeat（局部重读 / 滑窗 repeat_sum）

实现类：`vllm_ascend/dfx/detector/token_repeat.py`。

### 5.1 目标

基于**累计输出 token id**（不依赖 logprobs / msprobe）检测局部「回读」：同一 id 在短窗内反复出现时，`repeat_sum` 会快速升高。补 `token_logprob` 对「词词词」类正常 vocab 重复、以及无 top-k 场景的盲区。

设计原则：

- **与 OutputSubstring 同流**：读 `RequestIoSnapshotManager` 累计 output（由 `check_after_sample` 写入，含 MTP accepted tokens）；manager 传 `sampled_token_ids=None`，避免二次 `append_batch`。
- **增量 fold**：每请求维护 `_consumed_len` 游标，只推送自上次 check 以来的新 id；跳过 / 已告警仍推进游标，防止之后重复计入。
- **O(1) 滑窗**：`freq` + `scores` deque；每步 score = 新 id 在**先前** content 窗内的出现次数；`repeat_sum` = 最近 `window` 个 score 之和。
- **可忽略 filler**：`ignore_token_ids` 不进 content 窗（score=0），避免标点等拉高假阳性。
- **每请求一次**：`_alerted` + 共享 `stop_after_alert`；reap 时 detector `clear_finished` 清状态 / 游标。
- **热更改 `window`**：清空在途 per-req 状态（deque 与游标一并失效）。

### 5.2 开关与配置

| 字段 | 默认 | 说明 |
|------|------|------|
| `detector.token_repeat.enabled` | `false` | 局部重读检测（**不**依赖 msprobe / logprobs） |
| `detector.token_repeat.window` | `32` | content / score 滑窗长度（`>= 1`） |
| `detector.token_repeat.repeat_sum_threshold` | `64` | `repeat_sum >` 此值才计一次 over（严格大于） |
| `detector.token_repeat.min_tokens` | `32` | 至少累计这么多**非 ignore** content token 后才允许告警（`0`=不 warmup） |
| `detector.token_repeat.consecutive_hits` | `1` | 连续 over-threshold 步数达到后才告警 |
| `detector.token_repeat.ignore_token_ids` | `[]` | 不进入 content 窗的 token id（如标点 filler） |

示例（抓「词词词」类短窗密重复；按模型调阈）：

```json
{
  "dump": { "enabled": true, "max_times": 3 },
  "detector": {
    "token_repeat": {
      "enabled": true,
      "window": 32,
      "repeat_sum_threshold": 64,
      "min_tokens": 32,
      "consecutive_hits": 1,
      "ignore_token_ids": []
    }
  }
}
```

调参直觉：

- 同一 token 连续刷：score 近似 `0,1,2,…`，窗满后 `repeat_sum` 约 `window*(window-1)/2` 量级；默认 `window=32`、`threshold=64` 偏保守，密重复很快超阈。
- 误报多：增大 `repeat_sum_threshold` / `min_tokens` / `consecutive_hits`，或把高频 filler 放进 `ignore_token_ids`。
- 漏报：减小 `window` 或 `repeat_sum_threshold`，或把 `min_tokens` 降到接近 `window`。

### 5.3 架构与调用链

```text
check_after_spec
  └─ append accepted → RequestIoSnapshotManager（累计流）

check_after_sample
  └─ DetectorManager
       ├─ append_batch（本步 sample，一次）
       ├─ TokenLogprob …
       ├─ OutputSubstring(sampled_token_ids=None)
       └─ TokenRepeat(sampled_token_ids=None)
            ├─ new_ids = cumulative[consumed:]；consumed = len(cumulative)
            ├─ 逐 id push_token_repeat → 更新 repeat_sum / consecutive_hits
            └─ 命中 → AnomalyAlert(anomaly_type="token_repeat", ill_type=REPEAT)
```

- **为何不吃本步 `sampled_token_ids` alone**：TokenRepeat / OutputSubstring 要看**跨步累计**流（含此前各步 accepted tokens），不能只看本 step 一行；manager 在 `check_after_sample` **单次** `append_batch` 后再以 `sampled_token_ids=None` 读累计 IO。
- **MTP / async**：accepted tokens 只由 `check_after_sample` 写入（`check_after_spec` 不再 append）。`append_output` 对连续相同 chunk 的同波去重仍保留作兜底；检测侧游标按累计长度推进。

### 5.4 Report 字段

| 字段 | 说明 |
|------|------|
| `repeat_sum` | 告警时最近 `window` 个 score 之和 |
| `repeat_sum_threshold` | 配置阈值 |
| `window` | 滑窗长度 |
| `content_tokens_seen` | 累计进入 content 窗的非 ignore token 数 |
| `last_score` | 本 chunk 最后一枚 token 的 score |
| `consecutive_hits` | 连续 over-threshold 步数 |
| `chunk_len` | 本步 fold 的新 id 个数 |
| `recent_token_ids` | 本 chunk 末尾最多 32 个 id（证据截断） |

`ill_type` 固定为 `ILL_TYPE_REPEAT`（与 msprobe repetition 类别码一致，便于 report 汇总）。

## 6. BlockKv（KV block 写入完整性）

实现类：`vllm_ascend/dfx/detector/block_kv.py`

| 字段 | 默认 | 说明 |
|------|------|------|
| `detector.block_kv.enabled` | `false` | KV block 写入 wave / writer 一致性检测 |
| `detector.block_kv.check_wave_regression` | `true` | 同一 physical block 的 `last_write_wave` 不得回退 |
| `detector.block_kv.check_same_wave_writer` | `true` | 同一 wave 内同一 block 不得被两个 req 写入 |

- 钩子：`DfxProcessor.note_kv_block_writes` → `DetectorManager.check_kv_block_writes`（在 `KvBlockMetaTracker.record_writes` **之前** 预检）。
- V2：`note_kv_block_writes` 仅在 `execute_model` **成功返回**后调用（不在 `finally`），避免失败 forward 用 stale `execute_model_state` 误记账；`finalize_dump_data` 仍在 `finally`。
- **仅 `kv_cache_group=0`**：只检查第一组 KV block table（纯文本 FullAttention 通常只有一组）。Hybrid / Sliding Window / Mamba / draft 等多 group 模型，group≥1 上的写冲突 **不会** 被检测；需要时再扩展为按 group 循环 + `(group, block_id)` tracker key。
- **写入区间**：`touched_block_ids` 只含 `[num_computed_before, computed+scheduled)` 映射到的 physical block。共享前缀若本 step 未写入，不应进入 `touched`。写范围超出 block table 时返回空列表（**不**回退到表尾，以免把前缀块当成写入）。`num_computed_before` 读不到时 **跳过该 req**（debug 日志），不告警。
- `check_same_wave_writer`：同一 wave 内两个 req 写入**同一** touched block 才报；decode 通常只碰尾块，长 prefill 跨多 block 时更常见。
- **同 step pending dump**：logits/position 若已 arm dump，`can_run_anomaly_detection` 默认会因 dump-busy 跳过后续 detect；`check_kv_block_writes` 使用 `ignore_dump_busy=True`，仍执行检测（report 照写；二次 arm dump 为 no-op）。
- **同 step stop_after_alert**：`check_before_sample` 内 logits 告警后立即 mark，再跑 position；随后 `check_kv_block_writes` 也会跳过已停 req。
- **多 block 一份 report**：同一次 `check_writes` 若有多个 block 违规，合并为 **一条** `AnomalyAlert`，`detail.violations=[...]` + `num_violations`。
- **告警必带上次写者（仅 block_kv，且仅出错 block）**：`violations[].prev_writer_req_id` / `new_writer_req_id`，以及 `detail.violated_blocks`（**只含** `violations` 里的 block，不含该请求其它 block）。仅此类忽略 `report.block_last_writer`；其它 anomaly 与全量 `blocks[]` 仍受 `report.block_last_*` 控制。
- 不依赖 `report.block_last_*`；detector 开时会仍更新 tracker（与 report 字段独立）。
- **msprobe**：纯 DFX 原生检测；命中后与其它 detector 相同：`AnomalyAlert` → `Dumper.handle_anomaly_alert`（`dump.enabled=true` 时在后续 forward 采 msprobe 张量）。不调用 ILLDetector。

## 7. PositionAlignment（position_ids 对齐）

实现类：`vllm_ascend/dfx/detector/position_alignment.py`

| 字段 | 默认 | 说明 |
|------|------|------|
| `detector.position_alignment.enabled` | `false` | 本 step 新调度 token 的 1-D `position_ids` 连续性 / 起点对齐 |

- 钩子：`check_before_sample`（v1：`sample_tokens` 内、grammar bitmask **之前**；v2：临时 wrap `model.compute_logits`，在返回 logits 时调用 `check_before_sample`，仍位于上游 `sample()` 的 grammar / sampler **之前**）。
- 期望：每个 req 本 step 的 positions 为 `[num_computed_before, num_computed_before+1, …]`，其中 `num_computed_before` 是 **本 wave 执行前** 已计算 token 数（与 runner 建 `positions = computed + offset` 一致）。**不要**用 `computed - scheduled` 猜测。读不到 computed（batch / requests 都没有）则 **跳过该 req**；`0` 是合法首 prefill，不当成缺失。优先 `input_batch.num_computed_tokens_np|cpu`。
- 开启后每个 sample 步做 **一次** device reduce + `.item()` 同步；仅 mismatch 时再把 positions D2H 写 report。默认关。
- V2：`ExecuteModelState` 无 logits/scheduler 字段；Ascend **override** `sample()` 为「可选 wrap `compute_logits` + `super().sample()`」，不二次 LM head、不复制上游采样分支。仅 `logits_finite` / `position_alignment` 开启 **且** `can_run_anomaly_detection()` 为真时才替换方法，`finally` 还原，避免 gated 步（错 rank / dump busy）仍包一层开销，也不影响 prompt-logprobs / draft。
- 非 1-D `positions`（M-RoPE / 多模态）直接跳过，不告警。默认关闭；仅文本 1-D RoPE 场景启用。
- **msprobe**：原生检测；异常时 report 带 `expected_positions` / `actual_positions` 摘要；可选 dump。

## 8. LogitsFinite（采样前 logits 有限性）

实现类：`vllm_ascend/dfx/detector/logits_finite.py`

| 字段 | 默认 | 说明 |
|------|------|------|
| `detector.logits_finite.enabled` | `false` | 采样行 logits 上 GPU `isfinite` reduce（NaN/Inf） |

- 钩子：同 `check_before_sample`（须在 **grammar bitmask 之前**，避免把合法 `-inf` 当成异常）；只扫 **采样行** `[num_sample_rows, vocab]`，不全网、不全量 D2H。
- 开启后每个 sample 步 `isfinite.all().item()` **一次** NPU→CPU 同步（拉长 `sample_tokens`）。全 finite 时不再拷 vocab；仅异常行再 `tolist()` / `logits_indices.cpu()`。默认关，排查用。
- **NaN 与 ±Inf**：`torch.isfinite` 均会命中。msprobe ILL 码表无独立 Inf 类型，report 的 `ill_type` 仍为 `4`（`nan`）；`detail.finite_kind` 区分 `nan` / `pos_inf` / `neg_inf`。
- V2：见 §7（wrap `compute_logits` + `super().sample()`，复用已算 logits，不二次 LM head）。
- report：`ill_type=4`（`nan`），与 `token_logprob.ill_nan` 类别码一致；前者看 **logits**，后者看 **top-k logprob**（msprobe ILLDetector）。
- **msprobe**：不依赖 msprobe 包；若 `dump.enabled=true`，alert 后下一波 forward 由 msprobe `PrecisionDebugger` / `AclGraphDumper` 落盘（与 manual_trigger / auto dump 相同通路）。
### msprobe 适配总表（全部 detector）

| detector | 依赖 msprobe | 命中后 dump |
|----------|-------------|-------------|
| `token_logprob` | **是**（ILLDetector） | `dump.enabled` + quota/cooldown |
| `spec_acceptance` | 否 | 同上 |
| `output_substring` | 否 | 同上 |
| `token_repeat` | 否 | 同上 |
| `block_kv` | 否 | 同上 |
| `position_alignment` | 否 | 同上 |
| `logits_finite` | 否 | 同上 |

共性：`detect-only`（`dump.enabled=false`）仅写 report；`detect+auto_dump` 需 `dump.enabled=true` 且 `max_times>0`（或 `manual_trigger`）。msprobe JSON `dump_enable` 由 DFX dumper 写/读；bootstrap 兼容见 `runtime_config` msprobe seed。

## 9. 代码落点

| 模块 | 说明 |
|------|------|
| `vllm_ascend/dfx/runtime_config.py` | JSON 热更新 / broadcast；`detector` + `dump` 段 |
| `vllm_ascend/dfx/processor.py` | 编排 check / report；刷新 `InputFilterManager` |
| `vllm_ascend/dfx/detector/manager.py` | `DetectorManager` 阶段钩子门面 |
| `vllm_ascend/dfx/detector/spec_acceptance.py` | 投机接受率 |
| `vllm_ascend/dfx/detector/token_logprob.py` | token/logprob（ILLDetector） |
| `vllm_ascend/dfx/detector/output_substring.py` | 输出子串命中 |
| `vllm_ascend/dfx/detector/token_repeat.py` | 局部重读（滑窗 repeat_sum） |
| `vllm_ascend/dfx/detector/block_kv.py` | KV block wave / writer 完整性 |
| `vllm_ascend/dfx/detector/position_alignment.py` | position_ids 对齐 |
| `vllm_ascend/dfx/detector/logits_finite.py` | 采样前 logits 有限性 |
| `vllm_ascend/dfx/input_filters.py` | detect 输入过滤（`InputFilterManager`） |
| `vllm_ascend/dfx/request_state.py` | `RequestDfxStore` / `RequestDfxState`：per-req 共享态；`mark_finished` + deferred `clear` |
| `vllm_ascend/dfx/io_snapshot.py` | `RequestIoSnapshotManager`：report I/O 视图（normalize→Store + snapshot） |
| `vllm_ascend/dfx/dumper/` | dump 生命周期、pending-OR（无检测转发） |
| `vllm_ascend/dfx/report.py` | 短报告落盘 |
| `vllm_ascend/worker/model_runner_v1.py` / `v2` | sync / async 调用点；v2 wrap `compute_logits` + `super().sample()` |
| `tests/ut/test_dfx_detectors_integrity.py` | detector 逻辑 + mock V1/V2 钩子；无真 NPU e2e |
| `docs/source/user_guide/configuration/additional_config.md` | 用户配置表 |

## 10. 相关文档

- 总览与配置：[dfx_design.md](./dfx_design.md)
- 运维与排障：[dfx_ops.md](./dfx_ops.md)
- dump 生命周期 / PP-TP 齐步：[dumper_design.md](./dumper_design.md)
- 异步调度时序：[async_scheduling_design.md](./async_scheduling_design.md)
