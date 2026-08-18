# Dumper 方案说明（vllm-ascend）

> 本文聚焦 **msprobe dump 生命周期与跨并行齐步**。  
> DFX 总览（Config / Detector / Dump / Report）见 [dfx_design.md](./dfx_design.md)。

## 1. 目标

`Dumper`（包 `vllm_ascend/dfx/dumper/`：`core` 编排 + `msprobe` / `pending` mixin）统一动态 dump 与异常检测触发，减少 `model_runner` 中的分散代码，保证 DP/PP/TP 下行为可预测。

Detect 输入过滤见 [dfx_design.md](./dfx_design.md) §2.6 / [dfx_ops.md](./dfx_ops.md) §2.4（`manual_trigger` 不走过滤）。

## 2. 代码路径

- 核心实现：`vllm_ascend/dfx/dumper/`（`core` / `msprobe` / `pending`）
- Processor（runner 编排）：`vllm_ascend/dfx/processor.py`
- Runtime Config：`vllm_ascend/dfx/runtime_config.py`
- Detector：`vllm_ascend/dfx/detector/`
- Report：`vllm_ascend/dfx/report.py`
- v1：`vllm_ascend/worker/model_runner_v1.py`
- v2：`vllm_ascend/worker/v2/model_runner.py`

## 3. 结构与职责

`Dumper` 主要包含（**不管** config reload / report；由 ``DfxProcessor`` 编排）：

1. **应用已同步的 runtime config**
   - `apply_dfx_config()`：同步 `dump.max_times` / cooldown、调用 `apply_ascend_log_level`（`InputFilterManager` 由 `DfxProcessor` 刷新）

2. **debugger 生命周期**
   - `_init_debugger()`：按 `CUDAGraphMode` 选择 `PrecisionDebugger` 或 `AclGraphDumper`；**软失败**（缺 msprobe / 构造失败 → `_debugger=None`，不杀进程）
   - `_enforce_dump_requires_debugger()`：`dump.enabled=true` 但 debugger 不可用时 ERROR 并强制 `dump.enabled=false`（可热更重试：reload 时 lazy 再 `_init_debugger`；ACLGraph 下启动后 lazy 成功会 WARNING，图已 capture 时建议重启）
   - `start_dump_data()` / `finalize_dump_data()`：按 `dump_enable` 门控的 start→forward→step

3. **接 `AnomalyAlert`**
   - `handle_anomaly_alert()`：arm / activate dump；**不**写 report

4. **dump 开关与跨 TP 齐步**
   - `enable_msprobe_dump_if_needed()` / `sync_dump_pending_or()` / `disable_msprobe_dump_if_needed()`

5. **请求过滤** — `is_related_local_request()`

Runner 侧：

- `self.dfx = DfxProcessor(self)`（runner **不**直接摸 `Dumper`）
- 阶段钩子：`dfx.sync_for_step()` / `dfx.mark_finished()` / `dfx.check_after_spec()` / `dfx.check_after_sample()` / `dfx.ensure_logprobs_for_detection()`
- dump 生命周期经 processor 委托：`dfx.start_dump_data()` / `dfx.finalize_dump_data()`

## 4. 调用链（v1 / v2）

### 4.1 v1

1. 初始化：`Dumper(..., dfx_config=ascend_config.dfx_config)`
2. `execute_model()` 入口：`dfx.sync_for_step()` → `dfx.start_dump_data()` → forward →（非 last PP 早 `dfx.finalize_dump_data()`；last PP 在 `sample_tokens` 末 `finalize`）
3. 采样后：`dfx.mark_finished`→ `dfx.check_after_spec`；sync 当场 `dfx.check_after_sample`（末尾 reap），async 在 `get_output()` 中检测并 reap；idle 下一波 `sync_for_step` 也可 reap

### 4.2 v2

1. 初始化：同上；`load_model` 在图模式下提前 `start_dump_data` → **安装 AclGraphDumper hook 且保持 `_running=True`**（构图时必须带上 dump 插桩，否则 replay 采空）。
2. Dump 窗口：`start` 清掉窗口前缓冲的 acl stats，`finalize.step()` 写盘；**不** `stop()`（避免卸 hook）。非 dump 步不调用 `step()`，因此不落盘。
3. Eager（`PrecisionDebugger`）仍仅在 dump 窗口 `start`，避免 profile_run AOT 被破坏。
4. `execute_model()`：`dfx.sync_for_step(allow_arm=not dummy_run)` → `dfx.start_dump_data()` → `super().execute_model` → `finally: dfx.finalize_dump_data(dump=not dummy_run)`
5. `postprocess_sampled()`：`dfx.check_after_spec`
6. `sample_tokens()`：sync 当场 `dfx.check_after_sample`；async 包装为 `AscendAsyncOutput`，在 `get_output()` 后检测

## 5. Async 跨 TP dump 齐步（last PP）

```text
check 命中（async 仅 last-PP TP0）:
  pending_dump = True          # 不写 dump_enable

execute_model / idle dummy 入口:
  dfx.sync_for_step()          # refresh_config（per-DP broadcast 或 file poll）
                               # + sync_dump_pending_or（仅 last PP）
  sync_dump_pending_or():
    all_reduce(SUM, pending) on tp_group.cpu_group
    any_pending = (sum > 0)
      if any_pending and allow_arm:
          各 TP: activate(dump_enable=true + reload)
          clear pending

start → forward → finalize → disable（需 _dump_forward_seen）
```

说明：

- **不区分 req_id**；OR 的是「是否 pending」布尔。
- **async 仅 TP0 check**：multiproc 只在 `output_rank`（last-PP TP0）调用 `get_output()`。
- **early PP 不参与 dump OR**，但仍必须跑 `sync_for_step` → config sync（本 EngineCore 同步组 / file poll）。
- **Sync + TP>1 / async**：check 仅 TP0 → `pending_dump`；下步入口 last-PP TP `all_reduce(OR)` 后全体 activate。
- **Sync + TP=1**：可当场 activate。
- pending / dump_active 期间跳过后续 anomaly check，避免重复 arm。
- **默认全关 fast-path**：`dump.enabled=false` 且无 detector 时，`sync_dump_pending_or` **跳过** TP `all_reduce`（pending 恒为 0；热更开着也走这条，因同 wave 已先 `refresh_config`）。顺带清掉本 rank 残留 pending，避免下次再开 dump 时无 trigger 却 activate。前提是**同一 EngineCore 内各 TP 本 wave 看到的 dump/detector 开关一致**。若运维给不同 TP 配了不同 JSON，可能一侧 skip、一侧进 OR → **集体通信挂死**；正常同路径启动不会。见 [dfx_ops.md](./dfx_ops.md) 排障表。

## 6. DP / PP / TP

### 6.1 PP

- check / enable / dump OR / activate：仅 **last PP**
- early PP：不 dump，但必须参与本 EngineCore 的 `sync_dfx_config`（避免同步组 collective 卡死）

### 6.2 TP

- 日志：`tp_rank == 0`
- **check（async，或 sync 且 TP>1）**：仅 TP0；**dump**：OR 后 last-PP 全体 TP activate
- **check + dump（sync 且 TP=1）**：单卡当场 activate

### 6.3 DP

- 各 DP 副本独立；`tp_group` 不含跨 DP 进程
- Config sync：**禁止**跨 DP 满编 world；用 `inner_dp_world` 或 file poll。见 [dfx_design.md](./dfx_design.md) §2.2

## 7. 路径与落盘

1. msprobe 配置：`runner.ascend_config.dump_config_path` / `dump_config`
2. DFX 运行时配置：`dfx_config_path`（默认 `<cwd>/dfx/config/dfx_config.json`）
3. 异常短报告：`<dfx_root>/report/anomaly_YYYYMMDD_HHMMSS_mmm[_dump]_pidXXXXX.log`（由 `DfxProcessor` 始终写 report；仅当本次事件成功 arm dump 时带 `_dump`；JSON 含 `dump_armed` / `dump_attempted` / `dump_arm_wave` / `dump_count` / `dump_max_times`；与结束时的 `dump_finish_*.log` wave 字段对齐）
4. `set_msprobe_dump_state`：msprobe JSON 旁 `.lock` 持锁写 `dump_enable`
5. `save_sample_param`：在 ``DfxProcessor``（``log.print_sampling_meta=true`` 且 TP0 && last PP）

## 8. 已知限制

1. `forward_seen` 只表示「activate 后调用过 start」，不保证 msprobe 一定写出文件。
2. ACLGraph：必须在构图前安装 hook 且保持采集开启；DFX 只闸 `step()` 落盘。若仅在 dump 窗口才 `start`，replay 采空（「无 DFX 常开有数、DFX manual_trigger 无数」）。
3. v1 EC producer 短路径可能在 activate 后用 encoder-only `start→finalize` 消费窗口；普通文本 serving 无此路径。
4. async / sync+TP>1 下，若未走 fast-path，last PP 每步 CPU `all_reduce`（全员参与）；不能「仅 pending 的 rank 进 collective」。**Fast-path**：`dump.enabled=false` 且无 detector 时跳过该 OR（热更开着也适用；见 §5）。
5. Sync + TP>1 也走 pending-OR（与 async 相同齐步模型）；仅 Sync + TP=1 可当场 activate。
6. Config sync 与 dump OR 使用不同 process group（per-DP/file vs tp）；二者都要求**各自组内**全员同拍进入。

## 9. 相关文档

| 文档 | 内容 |
|------|------|
| [dfx_design.md](./dfx_design.md) | 总览 / InputFilterManager |
| [dfx_ops.md](./dfx_ops.md) | 运维与排障（含 ACLGraph / manual_trigger） |
| [async_scheduling_design.md](./async_scheduling_design.md) | async 时序 |
