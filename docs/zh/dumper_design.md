# Dumper 方案说明（vllm-ascend）

## 1. 目标

`Dumper` 的目标是统一动态 dump 与 MTP 观测逻辑，减少 `model_runner` 中的分散代码，确保在 DP/PP/TP 并行场景下日志和 dump 行为可预测。

## 2. 代码路径

- 核心实现
  - `vllm-ascend/vllm_ascend/dumper.py`
- v1 接入点
  - `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
- v2 接入点
  - `vllm-ascend/vllm_ascend/worker/v2/model_runner.py`

## 3. 结构与职责

`Dumper` 主要包含四类能力：

1. debugger 生命周期
- `init_debugger()`：按 `CUDAGraphMode` 选择 `PrecisionDebugger` 或 `AclGraphDumper`
- `start_dump_data()`：本轮开始前启动 debugger
- `finalize_dump_data()`：本轮结束后 `stop/step` 并按需回滚 `dump_enable`

2. MTP 观测与触发
- `check_acceptance_anomaly()`：计算窗口接受率、阈值判断、触发 full log 与 dump
- `log_mtp_token_details()`：打印 sampled/accepted/prompt/output token 明细

3. dump 开关控制
- `enable_msprobe_dump_if_needed()`：触发 dump（含冷却、最大次数、一次请求只触发一次）
- `disable_msprobe_dump_if_needed()`：延迟一轮后回滚 `dump_enable=false`
- `set_msprobe_dump_state()`：读写 msprobe 配置文件中的 `dump_enable`

4. 本地请求过滤
- `is_related_local_request()`：只允许当前 rank 上存在且有效的请求触发 dump

## 4. 调用链（v1 / v2）

### 4.1 v1

1. 初始化
- `__init__` 中创建 `self.dumper`
- `self.debugger = self.dumper.init_debugger(...)`

2. 执行阶段
- `execute_model()` 前后调用
  - `self.dumper.start_dump_data()`
  - `self.dumper.finalize_dump_data()`

3. 采样后阶段
- `sample_tokens()` 中逐请求调用
  - `self.dumper.check_acceptance_anomaly(...)`

### 4.2 v2

1. 初始化
- `__init__` 中创建 `self.dumper`
- `self.debugger = self.dumper.init_debugger(...)`

2. 执行阶段
- v2 重写 `execute_model()`，在 `super().execute_model(...)` 前后包裹
  - `self.dumper.start_dump_data()`
  - `self.dumper.finalize_dump_data()`

3. 采样后阶段
- v2 未重写 `sample_tokens()`，使用父类流程
- 通过重写 `postprocess_sampled()` 接入 dumper 的 MTP 标记更新

## 5. DP / PP / TP 下谁打日志、谁 dump

## 5.1 PP（Pipeline Parallel）

- 触发 dump 的硬门控：必须 `PP last rank`
- 在 `check_acceptance_anomaly()` 与 `enable_msprobe_dump_if_needed()` 中都有该门控
- 非 last PP 不应成为最终触发者

结论：
- PP 维度上，只有最后一段参与最终 MTP 触发与 dump。

## 5.2 TP（Tensor Parallel）

- 日志门控：`log_leader = (tp_rank == 0)`
- 短日志和 full token 明细日志由 TP0 打印
- 但 dump 触发本身并未限制为 TP0：只要满足条件的 TP rank 都可能执行 `enable_msprobe_dump_if_needed()`

结论：
- 谁打日志：TP0
- 谁执行 dump 开关：满足条件的 TP rank（通常是 last PP 内的各 TP rank）

## 5.3 DP（Data Parallel）

- 没有全局 DP 协调器来统一 dump 计数
- `_msprobe_dump_total_count`、`_msprobe_dumped_req_ids`、cooldown 都是本进程本地状态

结论：
- 每个 DP 副本独立判定、独立计数、独立触发。

## 6. 路径与落盘

1. msprobe 配置路径
- 来源：`runner.ascend_config.dump_config_path`

2. dump 开关修改方式
- 通过 `set_msprobe_dump_state()` 修改 JSON 中 `dump_enable`
- 写入时使用 `lock file`：`<dump_config_path>.lock`

3. sample 参数日志
- 由 `save_sample_param()` 记录，日志内包含 `dp_rank/tp_rank`
- 该函数在 `TP0 && PP last` 分支调用

## 7. 关键触发条件（简化）

一次请求在窗口内满足以下条件才可能触发 dump：

1. 具备 MTP 统计前提
- `need_accepted_tokens == true`
- `draft_len > 0`

2. 请求是本地有效请求
- 在本 rank 的 request 映射中可定位
- 未被 discard mask 过滤

3. 窗口统计越界
- 接受率/接受长度低于低阈值，或高于高阈值

4. dump 约束允许
- 未超过本地最大次数
- 请求未触发过
- 冷却窗口已过

## 8. 当前实现对齐建议

1. 保持职责边界
- `execute_model` 负责 start/finalize 生命周期
- `sample_tokens` 或其后处理钩子负责 MTP 统计

2. 维持统一口径
- v1/v2 对 MTP 接受率口径保持一致
- TP 日志与 dump 触发策略保持一致（日志 TP0，dump 可多 TP）

3. v2 接口一致性检查
- 当前 v2 使用了 `self.dumper.updae_mtp_debug_flags_v2(...)` 调用
- 需确保 `dumper.py` 中存在同名方法，或统一改为已存在接口，避免运行时属性错误
