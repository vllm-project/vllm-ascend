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

也可在 **不改 JSON 文件** 时，用与 DFX JSON **相同结构** 的 `additional_config.dfx_config` 在启动时开 detector：

```bash
vllm serve <model> --additional-config '{
  "dfx_config": {
    "detector": {
      "block_kv": { "enabled": true },
      "logits_finite": { "enabled": true }
    },
    "dump": { "auto_max_times": 0, "manual_dump": false }
  }
}'
```

合并顺序：`defaults ← dfx_config_path（显式路径时）← additional_config.dfx_config`。热更只重读 JSON，不再套一层 overlay；leader 落盘后文件里会带上启动 overlay 的结果。

| 项 | 说明 |
|----|------|
| `dfx_config_reload_interval > 0` | **必须**，否则改 JSON / `dump.manual_dump` 不生效 |
| `dump_config_path` | msprobe 配置；无则无法落 dump。默认各 DP **共享**同一路径；仅当显式 `dump_config_isolate_by_dp=true` 且有 `VLLM_DP_RANK` 时，`ascend_config` 会物化为 `<source_dir>/dp<rank>/...` 副本（热更请改副本）。多 DP 同写一份 `dump_path` 可能互相干扰，建议隔离或分 `dump_path`。启动写入 DFX JSON 的 `dump.msprobe_config_path`（可见/可热改）。**msprobe 兼容**：若用户 JSON / overlay **未写** `dump` 段，且 msprobe `dump_enable` 为 true 或缺省（当开），bootstrap 会 seed `manual_dump=true`、`auto_max_times=0`。若写了 `dump` 段但两者均关（`off_explicit`），与 msprobe 默认 on 冲突时会 **WARNING** 并写 msprobe `dump_enable=false`。`dump_enable=false` 与 DFX dump 能力开可并存（idle gate）；`DFX dump off` 但 msprobe 显式 `dump_enable=true` 仍报错（见 §1.1）。 |
| 每 EngineCore 可读 JSON | 多 DP **不**用满编 world 做 config sync。默认共享同一 `dfx_config` 路径；仅当显式 `dfx_config_isolate_by_dp=true` 且有 `VLLM_DP_RANK` 时，路径拆成 `dp<rank>` 副本（`ascend_config` 物化，与 sync 机制正交）。热更同步：有 `inner_dp_world` 则本 DP 内 broadcast，否则各 rank **file poll**（见 [dfx_design.md](./dfx_design.md) §2.2）。 |
| **同 EngineCore 内配置一致** | 各 TP/PP 须读**同一份** `dfx_config`（同路径）。尤其 dump 激活态 / detector：pending-OR idle fast-path 下 TP 间不一致会挂死（见 §3） |

默认路径（未设 `dfx_config_path`）：`<cwd>/dfx/config/dfx_config.json`，报告在同级 `dfx/report/`。启动时会用默认内容覆盖该路径上的既有文件（手改不跨重启保留）；持久配置请设显式 `dfx_config_path`。

- 当存在 `VLLM_DP_RANK` 且显式 `dfx_config_isolate_by_dp=true` 时，默认路径会拆分为：`<cwd>/dfx/config/dp<rank>/dfx_config.json`（路径隔离；热更仍走 per-DP `inner_dp` / file poll，**不是**跨 DP 满编 world）。

**默认全关开销**：`dfx_config_reload_interval=0` 且各 detector / dump 均未激活时，无 config 集体通信；热更关时跳过每步 filter 刷新；检测门控直接跳过。async / TP>1 下 pending-OR 在「dump 未激活且无 detector」时走 fast-path 跳过 `all_reduce`（热更开着也跳；同 EngineCore 内开关须一致，见 §3）。

> **dump 激活（runtime）**：`auto_max_times>0` **或** `manual_dump` 有效（`true` / 正整数）。JSON **无** `dump.enabled`；旧键 `enabled` / `max_times` / `manual_trigger` / `cooldown_seconds` 校验拒绝。

### 1.1a ACLGraph 预建与 idle 门控（重要）

图模式（`cudagraph_mode != NONE`）下，**有 `dump_config_path` / `dump.msprobe_config_path` 就会启动预建 `AclGraphDumper`**，即使当时 `auto_max_times=0` 且 `manual_dump=false`：

| 步骤 | 时机 | 作用 |
|------|------|------|
| 预建 debugger | `Dumper` 初始化 | 只要有 msprobe 路径就构造（dump 能力可仍关） |
| 装 hooks | `load_model` / 每步 `start_dump_data`（构图前） | `debugger.start(model)`，保持 `_running`；**不**等于一直写盘 |
| idle 关采集 | 预建 / recreate 后 `_close_idle_msprobe_dump_gate` | 写 msprobe `dump_enable=false` + sync 设备 `switch=0` |
| dump 窗口开采集 | detector / `manual_dump` arm | `set_msprobe_dump_state(True)` → switch=1 → 写完再关 |

**为什么：** ACLGraph 必须在 **capture 前** 把 dump 插桩编进图；事后 lazy 建 debugger / 再 hook，已 capture 的图 replay 采空。预建 + idle 关门控 = 图里有插桩，平时不写 staging，异常窗口才写。

**Eager（`CUDAGraphMode.NONE`）不变：** 仅 dump 能力开时才建 `PrecisionDebugger`（避免缺省 `dump_enable` 乱 wrap API）。

**运维含义：**

- 图模式想以后热更开 dump：启动就要带上 `dump_config_path`（预建），不必启动就开 `auto_max_times`。
- 启动无 msprobe 路径、构图后再配路径 / 开 dump：仍可能 lazy 且采空 → 看 WARNING，建议 **重启 worker**。
- `dump_enable=false` 与 DFX dump 能力开（`auto_max_times>0` / `manual_dump`）**可并存**：前者是 idle live gate，由 Dumper 在 dump 窗口打开；热更不会因此失败。

见 [dumper_design.md](./dumper_design.md) §3 / §8。

### 1.1 msprobe 路径与 bootstrap 策略

| 字段 | 作用 |
|------|------|
| `dump.msprobe_config_path` | 当前生效的 msprobe JSON；bootstrap 从 `dump_config_path`/`dump_config` 写入。改路径 → 下一拍热更 **重建 debugger** |
| `dump.auto_max_times` | 自动 dump 配额；`>0` 开启 auto-arm（与 `manual_dump` **互斥**） |
| `dump.manual_dump` | 手动 dump：`false`/`0`=关；`true`=每拍持续 arm；正整数 `N`=剩余 N 拍（**成功 arm 后**递减）。依赖热更 |
| `dump.auto_cooldown_seconds` | auto dump 冷却秒数（默认 300） |
| `dump.reload_msprobe` | 一次性：`true` → 用当前路径重建 debugger，然后清回 `false` |

**DFX `dump` 段语义**：

| 状态 | 判定 |
|------|------|
| `absent` | 用户 JSON / overlay 均未出现 `dump` 键 |
| `on` | `auto_max_times>0` 或 `manual_dump` 有效 |
| `off_explicit` | 写了 `dump` 段，但 auto 与 manual 均关 |

**与 msprobe `dump_enable` 对齐矩阵**（路径可读时）：

| DFX 状态 | msprobe | 行为 |
|----------|---------|------|
| `absent` | 默认 on / 显式 true | seed `manual_dump=true`，`auto_max_times=0` |
| `absent` | 显式 false / 文件缺失 | 全部关闭 |
| `on` | 缺 path | **ValueError** |
| `on` | 显式 false | **容忍**（idle / live gate，由 Dumper 开关采集） |
| `on` | 默认 on（键缺省） | 对齐，无写 |
| `on` | 文件缺失 / 不可读 | 写 stub `dump_enable=false`（idle；Dumper 开窗时再打开） |
| `off_explicit` | 显式 true | **ValueError** |
| `off_explicit` | 默认 on | **WARNING** + 写 msprobe `dump_enable=false` |
| `off_explicit` | 显式 false | 对齐，无写 |

**互斥**：`auto_max_times>0` 与 `manual_dump` 同时有效 → bootstrap / 热更 **ValueError**。

ACLGraph：重建可能采空，深度改配置仍建议 **重启 worker**。只改磁盘 msprobe 文件、DFX JSON 不变 → **不会**自动重建（除非 touch JSON 触发热更并重跑策略）。

## 2. 常用操作

### 2.1 开检测 / dump（可分离）

编辑 DFX JSON（热更开启后约 N 秒生效）。detect 与 dump **正交**；各 `detector.<name>.enabled` 默认 `false`。

> `detector.stop_after_alert`（默认 `true`）：请求在每步持续检测，一旦该请求检出异常就停止检测它——
> 防止同一请求反复告警、不停写 report；请求 reap（`Store.clear`）后重算。设 `false` 恢复对同一请求持续重检/重告警。

**只 detect、不 dump**（异常仍写 report）：

```json
{
  "dump": { "auto_max_times": 0, "manual_dump": false },
  "detector": { "spec_acceptance": { "enabled": true } }
}
```

**detect + 自动 dump**（尝试 arm dump；arm 失败也写 report）：

```json
{
  "dump": {
    "auto_max_times": 3,
    "auto_cooldown_seconds": 300,
    "manual_dump": false,
    "msprobe_config_path": "/data/msprobe_dump.json"
  },
  "detector": {
    "spec_acceptance": { "enabled": true },
    "token_logprob": { "enabled": true }
  }
}
```

**只手动 dump**（无 auto 检测）：

```json
{
  "dump": { "auto_max_times": 0, "manual_dump": true },
  "detector": {}
}
```

- dump 激活且无 detector：合法；auto 不会触发；可用 `manual_dump`（日志 warn）。
- `auto_max_times: 0`：不 auto-arm dump；detect / `manual_dump` 仍可。
- `auto_max_times>0` 与 `manual_dump` 有效**不能同时**出现。
- token 检测开启后 worker 会强制 top-k logprobs，请求侧可不设 `logprobs`。
- **`logits_finite` / `position_alignment`（默认关）**：打开后每个 sample 步有一次 device→CPU 同步（`.item()`），会拉长 `sample_tokens`。只排查时开。`position_alignment` 仅 1-D 文本 RoPE；M-RoPE / 多模态会跳过。`block_kv.check_same_wave_writer` 只检查本 step **写入区间** 的 block，不含未写入的共享前缀。
- **输出关键词**（`detector.output_substring.enabled`）：`patterns` 可混用字符串与 token id 列表，例如 `["ERR", [1,2,3]]`。热更后日志打印每条 pattern 的 text↔token_ids；默认**子序列**匹配（输出任意位置），设 `match_prefix: true` 改为**前缀**匹配（从输出开头）；命中后该 req 不再检；report 含 `matched_text` / `matched_token_ids` / `match_mode`（`prefix` / `subsequence`）。
- **局部重读**（`detector.token_repeat.enabled`）：**不**要 logprobs / msprobe。对累计 output 做滑窗：每新 token 的 score = 它在先前 `window` 个 content token 里出现的次数，`repeat_sum` = 最近 `window` 个 score 之和；`repeat_sum > repeat_sum_threshold`（且过 `min_tokens` warmup、满足 `consecutive_hits`）则告警。与 `output_substring` 同读 `RequestIoSnapshotManager`（含 speculative accepted）。默认 `window=32` / `threshold=64` / `min_tokens=32`；`ignore_token_ids` 可跳过标点等 filler。日志：`[Anomaly token_repeat]`；report 含 `repeat_sum` / `window` / `recent_token_ids`。详见 [anomaly_detection_design.md](./anomaly_detection_design.md) §5。

### 2.2 手动 `dump.manual_dump`

1. 确认热更已开、dump 已激活、`dump.msprobe_config_path`（或启动时的 `dump_config_path`）有效（**不必**开 detector）。  
2. 将 JSON 中 `"manual_dump"` 设为：  
   - `true`：**一直 dump**——每个 **`scheduled_tokens > 0`** 的真实 `execute_model` 拍都 arm，直到改回 `false`（不会自动清掉）；  
   - 正整数 `N`：在接下来 **N** 个有 scheduled tokens 的真实拍上各 arm 一次，每拍减 1，到 `0`/`false` 为止；  
   - `false` / `0`：关闭。  
3. 等待下一拍 **`total_num_scheduled_tokens > 0`（或等价 per-req scheduled）** 的 `execute_model`（空闲 / `execute_dummy_batch` / **`scheduled_tokens==0` 的 cleanup**、以及仅残留 `req_ids` 的拍 **不会**发出 trigger）。 Prefill 与 decode 都会触发（都有 scheduled tokens）。  
4. 若 dump 未激活：**不发 trigger**（`true` 也不清；int 次数不减）并打日志，修好后再等下一拍。  
5. **递减时机**：int `N` 在 **dump arm 成功之后**才减 1（不是 emit trigger 时）。这样 `manual_dump: 1` 在 `dump_enabled()` / pending-OR 检查时仍为开；arm 失败则次数保留可重试。`true` 模式不写回 JSON。日志可搜 `manual_trigger` / `[DFX manual_trigger]`（实现类仍为 `ManualTriggerManager`）。  
6. **Report**：每次 arm 写一份 `manual_trigger` 类型报告；`detail.requests` 含该拍 batch **全部**请求的 prompt/output（`save_sensitive_info` 控制是否带 token ids）；`detail.manual_trigger_remaining_after` 为预计消费后剩余（`true` 时仍为 `true`）。

### 2.2.1 Report 截断（查命中时）

- `save_sensitive_info=false`：不落任何 `*_token_ids`（改 `max_*` 无效）。
- `save_sensitive_info=true`：`max_prompt_token_ids` / `max_output_token_ids` 默认 **1000**，从**头部**截断；`0`=不截断。`*_token_count` 仍是全长。
- 截断标记：`output_token_ids_truncated` / `output_token_ids_max`（prompt 同理）。
- **检测用完整序列，report 可能只存前缀**：`output_substring` 等命中可能在截断窗口外——JSON 里看不到该 id 不等于没命中。核对全量时设 `"max_output_token_ids": 0`。

### 2.2.1a KV block 元数据（`report.*`）

| 开关 | 默认 | 作用 |
|------|------|------|
| `report.include_block_ids` | `true` | detail 带当前请求占用的 GPU `block_ids` |
| `report.include_slot_mapping` | `false` | 从 GPU **D2H** 本拍真实 `slot_mapping` 切片写入 `detail.slot_mapping`（另有 `slot_mapping_span=[start,end]` 对齐 packed batch）。默认关：有 device sync；prefill 可能很长。anomaly / `dump_finish` / `manual_trigger` 的 `requests[]` 均可带。`manual_trigger` 若在 `execute_model` 入口写报告，可能仍是上一拍 buffer。 |
| `report.block_last_write_wave` | `false` | 维护并写入各物理块最后写入的 DFX wave |
| `report.block_last_writer` | `false` | 维护并写入各物理块最后写入的 `req_id` |

后两项任一为 `true` 时，detail 另有 `blocks: [{block_id, last_write_wave?, last_writer_req_id?}]`（对应当前 `block_ids` 查表）。每拍 forward 后按本拍 scheduled 范围更新；wave 用 DFX `current_wave`，不用墙钟。

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
| `log.print_output_on_finish` | **每个**请求结束时 TP0 打 output ids/text 日志（噪声大，默关）。**仅在开关为 true 的 sample 步开始向 DFX 累积**，不回填开启前已生成的 token。热更中途打开时，对已在跑的请求：finish 日志可能是**部分** output，也可能是**空**（`output_token_count=0` / text 空，若开启后该请求未再走过累积路径）。要完整输出请在发请求前打开。 |

**文件产物**（默认目录 `<dfx_root>/report/`）：

| 文件 | 何时 | 用途 |
|------|------|------|
| `anomaly_*[_dump]_pid*.log` | 检测 / `manual_trigger` 当下 | 即时短报；arm 成功时带 `_dump` 与 `dump_arm_wave` |
| `dump_finish_*_pid*.log` | **已 arm dump** 的请求在 reap 时（`sample_waves` 空后；含仍 pending 未 activate） | 累计 output（受 `report.save_sensitive_info` / `max_*`）+ wave；未 activate 时 `dump_activate_wave=null` |

**wave 字段怎么对齐**（只计真实 `execute_model` 拍，`allow_arm=True`；dummy 不计）：

| 字段 | 出现位置 | 含义 |
|------|----------|------|
| `dump_arm_wave` | anomaly report（armed 时）+ dump_finish | arm / pending 那一拍 |
| `dump_activate_wave` | dump_finish | 真正 activate 那一拍；若请求在 **pending 未 activate** 时就结束，则为 `null` |
| `dump_waves_after_report` | dump_finish | `activate − arm`（同拍为 `0`；未 activate / 算不出为 `null`） |
| `dump_finish_wave` | dump_finish | 请求结束、写 sidecar 时的拍号 |

对齐方式：用同一 `req_id`（+ 可选 `dump_count`）把带 `_dump` 的 anomaly 与 `dump_finish_*` 配对；看 `dump_waves_after_report` 可知 report→activate 隔了几拍。pending-OR 下检测器告警常见差为 `1`（主线程在 sample 返回时把本拍 wave 写入 per-req 队列，异步 `get_output` 检测时 pop 该戳作为 `dump_arm_wave`，避免与下一拍 `advance_wave` 竞态）；手动触发同 `sync_for_step` 内 arm+activate 常见为 `0`。空 batch 的 manual 可能没有 dump_finish（没有真实 req 可挂）。

`dump_activate_wave=null`：dump 已 arm（open），但请求在 activate 成功前就结束并 reap——仍写 dump_finish 落累计 output / `dump_arm_wave`，避免内存孤儿；msprobe 窗口可能尚未打开。

### 2.3 打印一次输入 token ids（写 filter 用）

1. 热更开启（`dfx_config_reload_interval > 0`）。  
2. JSON 设 `"input_filter": { "print_input_token_ids_once": true }`。  
3. 下一拍**有请求**的 `execute_model`：TP0 打 `[DFX print_input]`（`length=` + 完整 `prompt_token_ids` + 前缀 filter 示例），再写回 `false`。  
4. 空闲 / dummy **不**消费。无 prompt 可取时也不消费，等下一请求。

### 2.4 按输入过滤检测（InputFilterManager）

配置在 `input_filter.filters`（**detect 阶段**；不挡 `manual_dump` / `ManualTriggerManager`）：

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
| `manual_dump` 为 true 仍无 dump | dump 未激活；服务空闲只走 dummy；或热更关 | 设 `manual_dump=true` 或 `auto_max_times>0`；打真实请求；确认 interval；要停则改 `false` |
| 多 DP 挂死 / 集体通信超时 | 曾用满编 world 做 config sync；一侧 idle 一侧 busy | **禁止**跨 DP world config；用 per-DP `inner_dp` 或 file poll（现行代码已如此） |
| TP>1 挂在 dump pending-OR / `all_reduce` | 同 EngineCore 内各 TP 的 dump 激活态 / detector（或整份 JSON）不一致：idle fast-path 一侧跳过 OR，另一侧仍进 `all_reduce` | **同一 EngineCore 共用一份** `dfx_config_path`（或同默认路径）；勿给不同 TP 挂不同 JSON / 不同开关 |
| 检测有 short / report 但无 msprobe 文件 | dump 未激活；或冷却 / 配额；或 early PP | 设 `auto_max_times>0` 或 `manual_dump`；查 `auto_cooldown_seconds`；dump 仅 last-PP |
| 完全无检测日志 | 未开任一 `detector.<name>.enabled`；或 rank / filter | 打开至少一个 detector；查 `[DFX filter]` |
| ACLGraph：无 DFX 常开有数、DFX dump 无数 | dump 窗口外才 `start`，或启动无 msprobe 路径导致构图后 lazy | 启动带 `dump_config_path`（图模式预建+装 hook）；idle 关 switch，窗口内再开。见 §1.1a / [dumper_design.md](./dumper_design.md) §8 |
| ACLGraph：dump 未开却堆满 staging | idle 未关 `dump_enable`/switch（旧版本或门控失败） | 升级含 `_close_idle_msprobe_dump_gate` 的版本；确认日志有 `idle dump gate closed` |
| `[Anomaly msprobe] debugger lazy-initialized after startup under ACLGraph` | 构图后才建 debugger | 重启前确保启动就有 `dump_config_path`；深度改路径后建议重启 |
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
| `[DFX manual_trigger]` / `manual_dump` | 手动 dump（JSON 键 `dump.manual_dump`） |
| `[DFX report]` / `[DFX dump_finish]` | 即时 anomaly report / 请求结束 dump_finish 落盘 |
| `[DFX print_output]` | `log.print_output_on_finish`（开启期间累积的 output；热开 in-flight 可能不全或空） |
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
