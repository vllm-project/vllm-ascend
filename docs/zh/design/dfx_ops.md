# DFX 运维与排障

> 面向部署 / on-call。设计细节见 [dfx_design.md](./dfx_design.md)；  
> 用户配置字段表见 `docs/source/user_guide/configuration/additional_config.md`。

## 1. 最小可用配置

```bash
vllm serve <model> --additional-config '{
  "dfx_config_path": "/data/dfx/config/dfx_config.json",
  "dfx_config_reload_interval": 5,
  "dump_config_path": "/data/msprobe_dump.json"
}'
```

| 项 | 说明 |
|----|------|
| `dfx_config_reload_interval > 0` | **必须**，否则改 JSON / `manual_trigger` 不生效 |
| `dump_config_path` | msprobe 配置；无则无法落 dump。默认各 DP **共享**同一路径；仅当显式 `dump_config_isolate_by_dp=true` 且有 `VLLM_DP_RANK` 时，`ascend_config` 会物化为 `<source_dir>/dp<rank>/...` 副本（热更请改副本）。多 DP 同写一份 `dump_path` 可能互相干扰，建议隔离或分 `dump_path`。 |
| 每 EngineCore 可读 JSON | 多 DP **不**用满编 world 做 config sync。默认共享同一 `dfx_config` 路径；仅当显式 `dfx_config_isolate_by_dp=true` 且有 `VLLM_DP_RANK` 时，路径拆成 `dp<rank>` 副本（`ascend_config` 物化，与 sync 机制正交）。热更同步：有 `inner_dp_world` 则本 DP 内 broadcast，否则各 rank **file poll**（见 [dfx_design.md](./dfx_design.md) §2.2）。 |
| **同 EngineCore 内配置一致** | 各 TP/PP 须读**同一份** `dfx_config`（同路径）。尤其 `dump.enabled`：热更关时 pending-OR 有 fast-path，TP 间不一致会挂死（见 §3） |

默认路径（未设 `dfx_config_path`）：`<cwd>/dfx/config/dfx_config.json`，报告在同级 `dfx/report/`。启动时会用默认内容覆盖该路径上的既有文件（手改不跨重启保留）；持久配置请设显式 `dfx_config_path`。

- 当存在 `VLLM_DP_RANK` 且显式 `dfx_config_isolate_by_dp=true` 时，默认路径会拆分为：`<cwd>/dfx/config/dp<rank>/dfx_config.json`（路径隔离；热更仍走 per-DP `inner_dp` / file poll，**不是**跨 DP 满编 world）。

**默认全关开销**：`dfx_config_reload_interval=0` 且各 detector / `dump.enabled` 均为关时，无 config 集体通信；热更关时跳过每步 filter 刷新；检测门控直接跳过。async / TP>1 下 pending-OR 在「热更关 + `dump.enabled=false`」时走 fast-path 跳过 `all_reduce`（同 EngineCore 内 `dump.enabled` 须一致，见 §3）。

## 2. 常用操作

### 2.1 开检测 / dump（可分离）

编辑 DFX JSON（热更开启后约 N 秒生效）。detect 与 dump **正交**；各 `detector.<name>.enabled` 默认 `false`。

> `detector.stop_after_alert`（默认 `true`）：请求在每步持续检测，一旦该请求检出异常就停止检测它——
> 防止同一请求反复告警、不停写 report；请求结束（`clear_finished`）后重算。设 `false` 恢复对同一请求持续重检/重告警。

**只 detect、不 dump**（异常仍写 report）：

```json
{
  "dump": { "enabled": false },
  "detector": { "spec_acceptance": { "enabled": true } }
}
```

**detect + 自动 dump**（尝试 arm dump；arm 失败也写 report）：

```json
{
  "dump": { "enabled": true, "max_times": 3, "cooldown_seconds": 300 },
  "detector": {
    "spec_acceptance": { "enabled": true },
    "token_logprob": { "enabled": true }
  }
}
```

**只手动 dump**（无 auto 检测）：

```json
{
  "dump": { "enabled": true, "max_times": 0 },
  "detector": {}
}
```

- `dump.enabled=true` 且无 detector：合法；auto 不会触发；可用 `manual_trigger`（日志 warn）。
- `max_times: 0`：不 auto-arm dump；detect / `manual_trigger` 仍可。
- token 检测开启后 worker 会强制 top-k logprobs，请求侧可不设 `logprobs`。
- **输出关键词**（`detector.output_substring.enabled`）：`patterns` 可混用字符串与 token id 列表，例如 `["ERR", [1,2,3]]`。热更后日志打印每条 pattern 的 text↔token_ids；默认**子序列**匹配（输出任意位置），设 `match_prefix: true` 改为**前缀**匹配（从输出开头）；命中后该 req 不再检；report 含 `matched_text` / `matched_token_ids` / `match_mode`（`prefix` / `subsequence`）。
- **局部重读**（`detector.token_repeat.enabled`）：**不**要 logprobs / msprobe。对累计 output 做滑窗：每新 token 的 score = 它在先前 `window` 个 content token 里出现的次数，`repeat_sum` = 最近 `window` 个 score 之和；`repeat_sum > repeat_sum_threshold`（且过 `min_tokens` warmup、满足 `consecutive_hits`）则告警。与 `output_substring` 同读 `RequestIoSnapshotManager`（含 speculative accepted）。默认 `window=32` / `threshold=64` / `min_tokens=32`；`ignore_token_ids` 可跳过标点等 filler。日志：`[Anomaly token_repeat]`；report 含 `repeat_sum` / `window` / `recent_token_ids`。详见 [anomaly_detection_design.md](./anomaly_detection_design.md) §5。

### 2.2 手动 `manual_trigger`

1. 确认热更已开、`dump.enabled=true`、`dump_config_path` 有效（**不必**开 detector）。  
2. 将 JSON 中 `"manual_trigger": true`（仅本 EngineCore 的 writer 路径）。  
3. 等待**有真实请求**的下一拍 `execute_model`（空闲 / `execute_dummy_batch` **不会**消费）。  
4. 若 `dump.enabled=false`：**不消费** flag（保持 true）并打日志，修好后再等下一拍。  
5. 成功后写回 `false`；日志可搜 `manual_trigger` / `[DFX manual_trigger]`。  
6. **Report**：arm 成功后写一份 `manual_trigger` 报告；`detail.requests` 含当前 batch **全部**请求的 prompt/output（`save_sensitive_info` 控制是否带 token ids）。

### 2.2.1 Report 截断（查命中时）

- `save_sensitive_info=false`：不落任何 `*_token_ids`（改 `max_*` 无效）。
- `save_sensitive_info=true`：`max_prompt_token_ids` / `max_output_token_ids` 默认 **1000**，从**头部**截断；`0`=不截断。`*_token_count` 仍是全长。
- 截断标记：`output_token_ids_truncated` / `output_token_ids_max`（prompt 同理）。
- **检测用完整序列，report 可能只存前缀**：`output_substring` 等命中可能在截断窗口外——JSON 里看不到该 id 不等于没命中。核对全量时设 `"max_output_token_ids": 0`。

### 2.2.2 日志开关（`log.*`）与 dump_finish / wave 对齐

配置在 JSON 顶层 `log`（**不是** `report`）：

```json
"log": {
  "print_sampling_meta": false,
  "print_output_on_finish": false
}
```

| 开关 | 作用 |
|------|------|
| `log.print_sampling_meta` | 写 anomaly / manual report 时，TP0+last-PP 打 `[SamplingMeta]` 日志（不进 JSON） |
| `log.print_output_on_finish` | **每个**请求结束时 TP0 打 output ids/text 日志（噪声大，默关） |

**文件产物**（默认目录 `<dfx_root>/report/`）：

| 文件 | 何时 | 用途 |
|------|------|------|
| `anomaly_*[_dump]_pid*.log` | 检测 / `manual_trigger` 当下 | 即时短报；arm 成功时带 `_dump` 与 `dump_arm_wave` |
| `dump_finish_*_pid*.log` | **已 arm dump** 的请求在 `clear_finished` 时（含仍 pending 未 activate） | 累计 output（受 `report.save_sensitive_info` / `max_*`）+ wave；未 activate 时 `dump_activate_wave=null` |

**wave 字段怎么对齐**（只计真实 `execute_model` 拍，`allow_arm=True`；dummy 不计）：

| 字段 | 出现位置 | 含义 |
|------|----------|------|
| `dump_arm_wave` | anomaly report（armed 时）+ dump_finish | arm / pending 那一拍 |
| `dump_activate_wave` | dump_finish | 真正 activate 那一拍；若请求在 **pending 未 activate** 时就结束，则为 `null` |
| `dump_waves_after_report` | dump_finish | `activate − arm`（同拍为 `0`；未 activate / 算不出为 `null`） |
| `dump_finish_wave` | dump_finish | 请求结束、写 sidecar 时的拍号 |

对齐方式：用同一 `req_id`（+ 可选 `dump_count`）把带 `_dump` 的 anomaly 与 `dump_finish_*` 配对；看 `dump_waves_after_report` 可知 report→activate 隔了几拍。pending-OR 下检测器告警常见差为 `1`；手动触发同 `sync_for_step` 内 arm+activate 常见为 `0`。空 batch 的 manual 可能没有 dump_finish（没有真实 req 可挂）。

`dump_activate_wave=null`：dump 已 arm（open），但请求在 activate 成功前就 `clear_finished`——仍写 dump_finish 落累计 output / `dump_arm_wave`，避免内存孤儿；msprobe 窗口可能尚未打开。

### 2.3 打印一次输入 token ids（写 filter 用）

1. 热更开启（`dfx_config_reload_interval > 0`）。  
2. JSON 设 `"input_filter": { "print_input_token_ids_once": true }`。  
3. 下一拍**有请求**的 `execute_model`：TP0 打 `[DFX print_input]`（`length=` + 完整 `prompt_token_ids` + 前缀 filter 示例），再写回 `false`。  
4. 空闲 / dummy **不**消费。无 prompt 可取时也不消费，等下一请求。

### 2.4 按输入过滤检测（InputFilterManager）

配置在 `input_filter.filters`（**detect 阶段**；不挡 `manual_trigger`）：

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

语义：全部 `include` 命中 **且** 无一 `exclude` 命中 → 才跑 Spec / TokenLogprob / OutputSubstring / TokenRepeat。  
前缀匹配用 `type: input_token_id_prefix`（无单独 `input_token_id_prefixes` 字段）。  
实现：`InputFilterManager`（`vllm_ascend/dfx/input_filters.py`），由 `DfxProcessor` 刷新。见 [dfx_design.md](./dfx_design.md) §2.6。

## 3. 排障速查

| 现象 | 常见原因 | 处理 |
|------|----------|------|
| 改 JSON 不生效 | `dfx_config_reload_interval=0`；或改了非本 DP 的文件 | 启动项 `>0`；确认本 EngineCore leader 可读路径 |
| `manual_trigger` 一直 true / 无 dump | `dump.enabled=false`；服务空闲只走 dummy；或热更关 | 开 `dump.enabled`；打真实请求；确认 interval |
| 多 DP 挂死 / 集体通信超时 | 曾用满编 world 做 config sync；一侧 idle 一侧 busy | **禁止**跨 DP world config；用 per-DP `inner_dp` 或 file poll（现行代码已如此） |
| TP>1 挂在 dump pending-OR / `all_reduce` | 同 EngineCore 内各 TP 的 `dump.enabled`（或整份 JSON）不一致：热更关时一侧走 fast-path 跳过 OR，另一侧仍进 `all_reduce` | **同一 EngineCore 共用一份** `dfx_config_path`（或同默认路径）；勿给不同 TP 挂不同 JSON / 不同 `dump.enabled` |
| 检测有 short / report 但无 msprobe 文件 | `dump.enabled=false`；或冷却 / 配额；或 early PP | 开 dump 并设 `max_times>0`；查 cooldown；dump 仅 last-PP |
| 完全无检测日志 | 未开任一 `detector.<name>.enabled`；或 rank / filter | 打开至少一个 detector；查 `[DFX filter]` |
| ACLGraph：无 DFX 常开有数、DFX dump 无数 | dump 窗口外才 `start`，replay 采空 | 构图前装 hook 且保持采集；DFX 只闸 `step()` 落盘。见 [dumper_design.md](./dumper_design.md) §8 |
| async 下只 TP0 有检测 | 设计如此（`get_output` 仅 output_rank） | dump 靠下一步 `pending-OR` 齐步；见 [async_scheduling_design.md](./async_scheduling_design.md) |
| 某类请求从不检测 | `input_filter.filters` 不匹配；prompt 取不到 | 查 `[DFX filter] skip detect`；临时清空 `filters` |
| 日志级别改了看不到 | ① `dfx_config_reload_interval=0`（在线改 JSON 不加载）；② 用的是**默认路径**且未设 `dfx_config_path`（启动会忽略/覆盖磁盘手改，`ascend_log` 回到 INFO）；③ 改了 `VLLM_LOGGING_LEVEL` 而非 `ascend_log`；④ 改错文件（多 DP 的 `dpN/` 副本） | 启动项 `dfx_config_reload_interval>0`；持久改级用显式 `dfx_config_path`；JSON 例：`"ascend_log": {"level": "INFO", "debug": ["dfx"]}` 或 `"level": "DEBUG"`；worker 日志里应出现 `[ascend_log] applied level=...` |

## 4. 日志关键字

| 前缀 | 用途 |
|------|------|
| `[DFX sync]` | config / dump OR 阶段 enter·leave |
| `[DFX runtime_config]` | JSON 读写 / broadcast / 路径 |
| `[DFX filter]` | InputFilterManager 拒绝 detect |
| `[DFX print_input]` | `print_input_token_ids_once` 打印 length + prompt token ids |
| `[DFX manual_trigger]` / `manual_trigger` | `manual_trigger` |
| `[DFX report]` / `[DFX dump_finish]` | 即时 anomaly report / 请求结束 dump_finish 落盘 |
| `[DFX print_output]` | `log.print_output_on_finish` |
| `[SamplingMeta]` | `log.print_sampling_meta` |
| `[Anomaly spec short]` / `[Anomaly token_logprob` / `[Anomaly output_substring]` / `[Anomaly token_repeat]` | 检测 short |
| `[Anomaly msprobe]` | dump arm / activate / 配额 |

## 5. 相关文档

| 文档 | 内容 |
|------|------|
| [dfx_design.md](./dfx_design.md) | 架构总览、JSON、多 DP sync、InputFilterManager |
| [dumper_design.md](./dumper_design.md) | dump 生命周期、PP/TP |
| [anomaly_detection_design.md](./anomaly_detection_design.md) | 异常检测：SpecAcceptance / TokenLogprob / OutputSubstring / TokenRepeat |
| [async_scheduling_design.md](./async_scheduling_design.md) | async 时序 |
| `additional_config.md` | 启动项与字段表 |
