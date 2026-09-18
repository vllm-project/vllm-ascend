# runtime_guard 运维与排障

> 面向部署 / on-call。  
> 设计细节见 [runtime_guard_design.md](./runtime_guard_design.md)；  
> 配置字段见 [runtime_config.md](../../source/user_guide/configuration/runtime_config.md)。

## 1. 最小可用配置

仅开检测 + 报告（无 KV dump）：

```bash
vllm serve <model> --additional-config '{
  "runtime_config_path": "/data/runtime/config/runtime_config.json",
  "runtime_config_reload_interval": 5
}'
```

`/data/runtime/config/runtime_config.json` 示例：

```json
{
  "detector": {
    "token_repeat": { "enabled": true },
    "logits_finite": { "enabled": true }
  },
  "dump": { "auto_max_times": 0, "manual_dump": false }
}
```

也可在 **不改 JSON 文件** 时，用 `additional_config.runtime_config` 在启动时 overlay：

```bash
vllm serve <model> --additional-config '{
  "runtime_config_reload_interval": 5,
  "runtime_config": {
    "detector": {
      "token_repeat": { "enabled": true, "on_trigger": ["report", "dump_kv"] },
      "logits_finite": { "enabled": true }
    },
    "dump": { "auto_max_times": 3, "auto_cooldown_seconds": 300 }
  }
}'
```

合并顺序：启动时 `defaults ← additional_config.runtime_config`，并覆盖写盘；热更新时 `defaults ← JSON`。  
热更只重读 JSON 文件，不再套一层 overlay。

## 2. 常用操作

### 2.1 开检测 / dump_kv（可分离）

| 目标 | 配置 |
|------|------|
| 仅 report | `"on_trigger": ["report"]` 或省略（默认） |
| report + KV | `"on_trigger": ["report", "dump_kv"]`，且 `dump.auto_max_times > 0` |
| 仅 log 级别 | `"on_trigger": ["set_log_level"]` + nested `set_log_level` |

Detector 默认全关；逐项 `enabled: true` 开启。

### 2.2 手动 `dump.manual_dump`

| 字段 | 含义 |
|------|------|
| `manual_dump: false` | 关 |
| `manual_dump: 1`（推荐） | **拍一次**：下一个有 scheduled tokens 的 wave 触发后归零 |
| `manual_dump: N`（N>1） | 连续 N 个有 scheduled tokens 的 wave 各触发一次 |
| `manual_dump: true` | 持续每个 wave 都 dump，直到热更改回 false（**不推荐**） |

**建议：用 `1`（或偶发 `2`）拍一次就够。** 连续 / `true` 收益很小——同批请求重复落盘、挤满 ActionQueue（`.pt` heavy）、占盘，对排查帮助有限；需要再抓时再热更一次即可。

**要求 `runtime_config_reload_interval > 0`**。manual 事件 **跳过 auto quota/cooldown**。

manual 触发 incident_type 为 `manual_trigger`。**始终**注入 `dump_kv`，且 **强制** `scope=all_requests`（配置里的 `dump_kv.scope` 无效）。`on_trigger` 仍可配 `report` / `set_log_level` 等；省略时默认含 `report`，再自动补上 `dump_kv`。

**计数与落盘：**

- 每个有 scheduled tokens 的 wave 在 `_handle_manual_trigger` 之后 **都会** 在内存扣减 `manual_dump`（`true` 持续模式不扣），**不论** `dump_kv` 是否入队成功。
- 若 dump 跳过（无 batch / 磁盘不足 / finished / queue 失败等），在  
  `{dump_root}/<type>/<req_id>/wave_<N>/[rank_tag/]dump_skipped.json`  
  写下 `reason` + `stage`（`arm`|`drain`）；**不**为失败重试保留次数。  
  drain 阶段 **每个 last-PP TP** 各自写自己的 `rank_tag/` 下标记（含 `d2h_failed` / `no_tensors` / `torch_save_failed` / `empty_block_ids`）。标记表示**未完整成功**（旁路可能已有 `request_info` 或部分 `.pt`）。
- **JSON 不在每次扣减时改写**；仅当内存计数减到 **0** 时，由 JSON writer **写一次** `manual_dump: false`。
- 内存计数仍 >0 时，若人手改了配置文件（内容变化），热更仍会把新值刷进内存（可中途改大/改小/关掉）。
- **多 DP 共用同一 `runtime_config.json`：** 文件在减到 0 之前一直保持原来的 `N`，每个 DP 副本各自在内存里扣减，因此 **每个 DP 最多可 dump N 次**；集群最坏约 `num_DP × N` 次。某个 DP 先写盘 `false` 后，其它 DP 热更到 `false` 会提前停。

### 2.3 Report 字段与截断

落盘（仅 last-PP TP0）：`{report_dir}/<type>/report_<毫秒时间戳>[_<req_id>]_pid<pid>.json`

顶层常见字段：`incident_type`、`req_id`、`rank`、`dump_attempted`（on_trigger 含 `dump_kv`，非 D2H 已成功）、`dump_arm_wave`（arm 拍；deferred 时可能 ≠ 实际 `wave_*`）、`dump_dir`（req 根目录，其下再有 `wave_*`）、`dump_count` / `dump_max_times`、`detail`。  
**manual**：batch 里 **每个真实 req 各写一份** report（顶层 `req_id` / `dump_dir` 对齐该 req；`detail.trigger_req_id` 为 `__manual_trigger__`）。不再写一份合成 id 的汇总 report。

同 `(incident_type, req_id)`：`report.max_per_req`（默认 **1**）；写满后 **停检该 req**；多份时 **wave 退避**（64×2ⁿ）。默认 `on_trigger` 含 `report`。与 `dump.auto_cooldown_seconds` 无关。

| 字段 | 作用 |
|------|------|
| `report.save_sensitive_info` | 是否写 prompt/output token ids |
| `report.max_prompt_token_ids` | 截断上限（0=不限） |
| `report.max_output_token_ids` | 截断上限 |
| `report.max_per_req` | 同 (type, req) 最多几份；写满停检（默认 1） |
| `report.include_block_ids` | detail 中带 GPU block_ids |
| `report.decode_token_ids` | 敏感信息模式下是否解码文本 |

查命中时若 report 过大，先关 `save_sensitive_info` 或降低 max_*。

### 2.4 KV dump 落盘与 rank

**范围：last PP × 全部 TP。其它 PP 不 dump。**

| Rank | dump 时机 |
|------|-----------|
| last PP TP0 | 检测/manual 命中时只排队；**同波** prepare 后与其它 TP 一起 D2H |
| last PP 全部 TP | **同波** `end_of_wave_sync`：收到 `{req_id, ...}` 后各自读本地 block 表再 D2H |
| 其它 PP | 不 dump |

```text
{dump_root}/<incident_type>/<req_id>/wave_<N>/
  request_info.json          # last-PP TP0，arm 时写
  dp{D}_tp0_pp{last}_cp{C}/  # 与其它 TP 同一拍 D2H
  dp{D}_tp1_pp{last}_cp{C}/
  ...
    {req_id}_{layer}_req.pt
```

`dump_root` 默认 `<report_dir>/kv_cache`。每个 `.pt` 含：`req_id`、`block_ids`、`layer`、`rank_tag`、`tp_rank` / `pp_rank` / `cp_rank`、`num_kv_heads`、`tensor`。

拼 **同一 `req_id` + 同一 `wave_*` 下 last PP 的各 `tp*` 目录** 得到该 stage 的 head 切分。没有更早 `pp*` 目录是预期的（那些层没 dump）。

- `scope=request`（默认，**自动检测**）：各 rank 用本地 `block_ids_for_request`；请求已 finish 或表为空则跳过。
- `scope=all_requests`（自动检测可配；**manual_trigger 固定为此**）：arm 时用 `iter_local_request_rows` 枚举本地 batch，**每个 req 各自** `block_ids_for_request`。
- **同一 arm wave 每个 `req_id` 最多入 dump pending 一次**（`queue_kv_dump` 按 `(wave, req_id)` 去重；重复入队打 INFO 日志后跳过）。
- **Report 入 ActionQueue**：同一 `(wave, req_id)` 已有 pending/running 的 report commit 时去重跳过（INFO）。  
  **`.pt` 按层各一个 heavy job**，不用 `(wave, req_id)` 去重（否则只会留下第一层）。
- 成功排队后 last-PP TP0 在 `{dump_root}/<type>/<req_id>/wave_<N>/request_info.json` 写请求元信息。
- **未产出完整 `.pt`**（arm 或 drain）时写  
  `{dump_root}/<type>/<req_id>/wave_<N>/[rank_tag/]dump_skipped.json`：  
  `reason` 如 `finished_or_reaped` / `no_dump_targets` / `empty_block_ids` / `insufficient_free_space` / `quota_or_cooldown_blocked` / `queue_failed` / `d2h_failed` / `no_tensors` / `torch_save_failed`；`stage` 为 `arm` 或 `drain`。drain 失败时 **每 TP 写自己的 rank 目录**；文件表示未完整成功，不表示目录为空。  
  同时打日志：`[runtime_guard dump_kv] skipped reason=...` / `skip empty local block_ids` 等。
- 本地 `block_ids` 为空：**自动检测跳过 dump**；**manual** 仍入队，等同波 `prepare` 后、`run_sample_phase` 末尾 `end_of_wave_sync` 再解析 block（不会拖全池 KV）。
- `.pt` tensor 保持 `[n_blocks, block_size, …]`（仅该请求占用的 block）。
- **Quota：** 一次 `prepare` 只 `try_consume` 一次；同 `arm_id` 的多个 req job 在 drain 时若 **全部** 未产出 D2H，才 **refund 一次**（还次数并清 cooldown）。异步 `torch.save` 失败 **不** refund（额度按 D2H 机会计，不按落盘成功）。ActionQueue **满**时 heavy（`.pt`）**丢弃**并 WARNING，light（report）改为 inline。
- TP=1：无名单广播，同波 flush 只 dump 一份 `tp0` 目录。
- dump / config 通知（PP==1 broadcast）：**波头一次** `all_reduce([config_due, dump_due])`；各 lane due 才各自 `broadcast_object`（平时只付 1 次 AR）。配置本波即用。
    - **自动单请求 dump**：本波 detect 入队，**下一波头**再 bcast，该波末尾 D2H（+1 wave）。
    - **manual_dump**：波末各 last-PP TP **本地** dump（不走 dump-job bcast）。
    - **PP>1（强制 file）/ file**：config 本地 poll；auto dump 在波头 TP drain → 末尾 D2H。
- D2H 时机：arm 只排队；**同波** `run_sample_phase` 末尾（`check_after_sample` 之后）由 `end_of_wave_sync` 做 D2H；本步无 sample 时在 sync 路径走同一末尾入口。

### 2.5 日志开关

| 配置 | 作用 |
|------|------|
| `log.print_output_on_finish` | 请求结束时打 output token ids / 解码文本（TP0） |
| `ascend_log.level` / `modules` | 模块日志级别；`[SamplingMeta]` 在 after-sample 打 DEBUG（开 `runtime_guard` DEBUG 即可见） |
| `set_log_level` action | incident 时临时提 log（detector 段可嵌套 `set_log_level`） |

## 3. 排障速查

| 现象 | 检查 |
|------|------|
| 无 report | detector 是否 `enabled`；是否 last PP + TP0；rank skip 日志 |
| 有 report 无 kv | `on_trigger` 是否含 `dump_kv`；quota/cooldown；`auto_max_times` 是否为 0 |
| 只有 `tp0` 没有其它 `tp*` | 是否 TP>1；其它 last-PP TP 是否跑到了同波 `end_of_wave_sync`；日志 `[runtime_guard dump_kv]` |
| 缺少更早 PP 的层 | 预期：当前不 dump 非 last PP |
| dump 文件空/缺层 | `block_ids`；`dump_skipped.json`；日志 `skip` / `skipped reason=` / `action queue full` |
| 重复 arm 无第二份 dump | 预期：同 wave 同 req 去重；日志 `skip enqueue: already pending` |
| report 缺一份 | `max_per_req` / wave 退避；或日志 `skip report enqueue (dedupe or stopping)` |
| 热更不生效 | `runtime_config_reload_interval` 是否 >0；JSON 路径各 rank 是否可读 |
| 重复刷屏 report | `report.max_per_req`（默认 1，写满停检）；确认 `on_trigger` 含 `report` |
| manual 不触发 | reload interval；是否 idle dummy wave；`manual_dump` 计数是否用尽 |
| 性能下降 | 先关全部 detector 仅留 reload，再逐项开启（见 feature guide） |

## 4. 日志关键字

| 关键字 | 含义 |
|--------|------|
| `[runtime_guard sync]` | 每 wave 配置同步 |
| `[runtime_guard manual_trigger]` | 手动 dump 控制面 |
| `[runtime_guard dump_kv]` | KV 排队 / D2H / 写盘；含 `skip enqueue: already pending`（同 wave 去重） |
| `[runtime_guard action]` | prepare/commit 失败；`skip report enqueue` / `skip heavy enqueue` |
| `action queue skip enqueue: duplicate key` | Report commit 同 `(wave, req_id)` 去重 |
| `action queue full` | heavy `.pt` 丢弃 / light inline |

## 5. 磁盘与 quota

- 每次 auto `dump_kv` 消耗 quota（`auto_max_times` 为进程内累计上限；用尽后需 **refund** 或 **重启** 才再放行。`auto_cooldown_seconds` 只限制两次**成功 consume** 的间隔，不解 cap）。
- `try_consume` 成功后若未能入队（例如 queue 不可用）或 drain 整 arm 未 D2H 会 **refund**（还次数并 **清除 cooldown**），避免空扣后卡冷却。
- 空闲空间门槛：leader 单卡 payload 估计 × `tp_size` + `free_headroom_bytes`（全 TP 写盘）。
- manual 路径不消耗 auto quota。
- 长期开 `dump_kv` 注意 `{report_dir}/kv_cache/` 磁盘；定期归档或调低 `auto_max_times`。

## 6. 相关文档

- [runtime_guard_design.md](./runtime_guard_design.md)
- [runtime_guard.md](../../source/user_guide/feature_guide/runtime_guard.md)
- [runtime_config.md](../../source/user_guide/configuration/runtime_config.md)
