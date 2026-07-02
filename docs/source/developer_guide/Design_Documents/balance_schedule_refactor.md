# Balance Schedule 重构方案

**TL;DR** 现有的 `patch_balance_schedule.py` 为了注入约 5 行真实逻辑，整段拷贝了上游 `Scheduler.schedule()`（约 520 行）和 `DPEngineCoreProc.run_busy_loop()`（约 40 行）。本方案在**严格保持 `balance_flag` 语义不变**的前提下移除这两份拷贝，最终留下一个约 60 行、不再随上游漂移的 patch。

## 背景

### Balance Scheduling 做了什么

在 `data-parallel-size` 较大、并发 ≈ `DP × max-num-seqs` 的场景下，请求容易堆积到部分 DP rank 上：被堆满的 rank 同时承担 prefill 与 decode、变慢，而其他 rank 空闲。Balance scheduling 的作用是让各 rank 的 running 请求数保持均衡。

该特性通过 `additional_config.enable_balance_scheduling = true` 开启（环境变量 `VLLM_ASCEND_BALANCE_SCHEDULING` 已弃用）。仅支持 PD-mixed 模式，校验逻辑见 `vllm_ascend/platform.py` 与 `vllm_ascend/ascend_config.py`。

### 真实逻辑（仅两处）

整个特性可归纳为两个操作：

1. **跨 rank 同步 running 数量** —— 每个引擎 step 执行一次 `all_gather`，收集各 rank 的 `len(self.running)`：

   ```python
   def balance_gather(self, dp_group):
       running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
       dist.all_gather(self.balance_queue, running_tensor, group=dp_group)
   ```

2. **WAITING 调度循环里的接纳闸门** —— 由于各 rank 持有相同的 gathered 向量，该判断在每个 rank 上结果一致。只要**上一步结束时任意一个 rank** 的 running 已达上限，**所有 rank** 本步都停止从 WAITING 队列接纳新请求：

   ```python
   balance_flag = max(t.item() for t in self.balance_queue) == self.max_num_running_reqs
   if balance_flag:
       break
   ```

   **语义（必须逐位保持）**："leader 一满 ⇒ 全员冻结接纳"。它**不等于**"让落后的 rank 追平 leader"。详见 [行为保持契约](#行为保持契约)。

## 问题陈述

为了注入上述两处逻辑，现有 `patch_balance_schedule.py` 逐字复制了上游的三个大单元：

| 被拷贝的单元                        | 行数  | 拷贝原因                                         |
|-------------------------------------|-------|--------------------------------------------------|
| `Scheduler.schedule()`              | ~520  | 在循环中段插入 3 行 `balance_flag` 闸门          |
| `DPEngineCoreProc.run_busy_loop()`  | ~40   | 每步后调用 `balance_gather`                      |
| `EngineCoreProc.run_engine_core()`  | ~55   | 在 DP>1 时替换为 `BalanceDPEngineCoreProc`       |

这种"整段拷贝"方式有三点具体危害：

1. **拷贝源自比当前 pin 更老的 vLLM，已整体陈旧。** vllm-ascend 当前 pin `v0.23.0`，但旧 patch 里复制的 `schedule()` / `run_busy_loop()` / `run_engine_core()` 三段函数体均来自更早的 vLLM（如 `schedule()` 体相对 v0.23.0 缺失 `current_step`/`next_decode_eligible_step`、`needs_kv_cache_zeroing`、`reserved_blocks`、`_inflight_prefills` 等约 80 行逻辑）。也就是说拷贝本身就不是 v0.23.0 的 `schedule()`——任何「与上游逐字比对」的测试都因此无法成立。

2. **违反 `AGENTS.md` 的 patch 规范。** 规范要求 patch「最小化、聚焦」并「有向上游贡献的长期计划」。500 行逐字拷贝无法 review（不与上游 diff 就看不出真正的改动），且每次升级 vLLM 都要人肉重新同步。

3. **静默、未文档化的偏离会累积。** 例如 override 把上游的 `assert request_queue is not None` 悄悄改成了 `if request_queue is None: break`。这类偏离会让未来的 diff 不可信。

> 教训（本重构过程中踩到的坑）：曾误以为上游 `schedule()` 已有 `throttle_prefills` 参数、旧 patch 漏掉导致 `TypeError`，并据此「修复」签名。实际核对 v0.23.0 tag 后发现 `schedule(self)` **本就无此参数**（`throttle_prefills` 是 v0.23.1 之后才引入）——旧 patch 的签名才是对的，那次「修复」反而会让禁用路径 `super().schedule(throttle_prefills)` 在 v0.23.0 上抛 `TypeError`。已回退。结论：**判断拷贝是否陈旧必须对照实际 pin 的版本，而非 main 分支 checkout。**

## 设计

### 原则

移除拷贝，不移除特性。把 gather 收进调度器自身（从而删掉 EngineCore 的两份拷贝），并在未来用最小的上游扩展点注入闸门（从而删掉 `schedule()` 拷贝）。`balance_flag` 语义严格不变。

### 实现状态与订正（本轮已落地 Phase 1 + 2A + 3）

实现时验证了上游真实结构，发现初稿两处假设不成立，已据实订正：

- **上游 `Scheduler` 没有 `new_step_starts()` 生命周期 hook**（那是 `kv_cache_manager` 的方法），也没有任何「每步调度开始」的可 override 接缝。因此 gather 改为落在 `schedule()` override 的**入口**（该 override 在 Phase 2A 本就要保留），而非一个新的 hook。
- **调度器无法惰性获取 DP group**：`dp_group` 在 `_init_data_parallel`（早于 scheduler 创建）中产生，无全局注册表。因此 `BalanceDPEngineCoreProc` **不删除**，而是精简为「在 `run_busy_loop` 入口注入 `dp_group` 后委托上游」的 3 行子类；`run_engine_core` 拷贝改为 patch 模块级 `DPEngineCoreProc` 名字（上游 `run_engine_core` 调用时按模块全局名解析该类）。

据此，Phase 1 的描述与「重构后文件形态」均已按实际实现改写。Phase 3 未强行收敛为单路读取（保留三路 fallback），仅把环境变量访问改走 `vllm_ascend.envs`。详细落地见 [分阶段落地](#分阶段落地)。

### Step 1 —— 将 `balance_gather` 移入 `BalanceScheduler`，删除 EngineCore 拷贝

旧实现里 `BalanceDPEngineCoreProc.run_busy_loop()` 与 `run_engine_core()` 都是上游方法的逐字拷贝，且**已经陈旧漂移**：上游 `run_busy_loop` 已改为 `while self._handle_shutdown()`、新增 `eep_scaling_state` / `is_sleeping` 守卫、末尾 `raise SystemExit`，而 patch 仍是 `while True` + 自写 signal handler；`run_engine_core` 同理多了 `SignalCallback`、numa、tracer 等逻辑。目标是删掉这两份拷贝。

实现时遇到两个约束，导致最终方案与初稿略有不同：

1. **上游 `Scheduler` 没有 `new_step_starts()` 这个 hook**（它是 `kv_cache_manager` 的方法），也没有任何「每步调度开始」的可 override 接缝。因此 gather 的注入点只能是 `schedule()` override 的顶部——而该 override 在 Phase 2A 本就要保留。于是 gather 自然落在 `BalanceScheduler.schedule()` 的入口，无需额外 hook。
2. **调度器无法自行获取 DP group。** `dp_group` 由 `parallel_config.stateless_init_dp_group()` 在 `DPEngineCoreProc._init_data_parallel` 中创建并挂在 engine core 上；而该方法在 `EngineCoreProc.__init__` 中于 `super().__init__()`（创建 scheduler）**之前**调用，全局也没有注册表可供惰性查询。所以必须由 engine core 把 `dp_group` 交给 scheduler。

据此，最终落地为：

- **`BalanceDPEngineCoreProc` 不删除，而是精简为 3 行**：override `run_busy_loop`，在入口 `self.scheduler.dp_group = self.dp_group`，然后 `super().run_busy_loop()` 完全委托上游（此时 scheduler 与 dp_group 都已存在）。不再拷贝 `run_busy_loop` 函数体。
- **`run_engine_core` 拷贝整体删除**。上游 `run_engine_core`（staticmethod）在函数体里通过模块全局名 `DPEngineCoreProc` 解析该类（`engine_core = DPEngineCoreProc(*args, **kwargs)`，见 [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm)）。因此只需 patch 模块级名字 `_engine_core_mod.DPEngineCoreProc = BalanceDPEngineCoreProc`，上游 `run_engine_core` 调用时自然实例化我们的子类，连同信号处理、`SignalCallback`、numa、tracer 等全部留给上游正确实现。`_ORIGINAL_RUN_ENGINE_CORE` 与 `EngineCoreProc.run_engine_core = ...` monkeypatch 一并删除。
- **`balance_gather` 收进 `BalanceScheduler`**，签名改为无参（用 `self.dp_group`），在 `schedule()` 启用分支的入口调用一次。

**时序偏移 —— 已分析、安全。** 旧实现在每个 busy-loop 迭代的 `_process_engine_step` **之后**调用 gather；移到 `schedule()` 入口后，它在当步 `schedule()` 函数体**之前**调用。两步之间 `self.running` 只在 `schedule()` / `update_from_output()` 内部变化，因此「当步 schedule 入口的 running 快照」与「上一步结束时的 running 快照」一致，闸门看到的值不变，接纳决策不变。每个引擎 step 仍是恰好一次 `all_gather`。由 gather 节奏单测锁定（见 [测试计划](#测试计划)）。

### Step 2 —— 用最小扩展点替换 `schedule()` 拷贝

`balance_flag` 闸门被上游内联在 `schedule()` 中段，目前没有可 override 的接缝。分两阶段解决。

**Phase 2A —— 过渡方案（不依赖上游）：**

保留 `schedule()` override，但：

- **签名保持 `def schedule(self)`**，与 pin 的 v0.23.0 一致；禁用路径 `return super().schedule()` 无参调用。切勿照搬 main 分支的 `throttle_prefills`（那是 v0.23.1+ 才有，照搬会在 v0.23.0 上抛 `TypeError`）。
- 把 balance 相关改动收敛为单一、注释清晰的 delta（入口的 `self.balance_gather()` + WAITING 循环里的 `balance_flag` 闸门）。由于上游没有更细粒度的 hook，函数体仍需复制。
- **不做「与上游逐字比对」的漂移测试**：拷贝源自比 pin 更老的 vLLM，全量比对必然海量失败。改为「意图锁定」——断言签名匹配已安装 vLLM、且 balance delta 行存在（见测试计划）。
- 将 `patch_balance_schedule.py` 纳入为 vendored vLLM pin 维护的「上游版本兼容性矩阵」，单独跟踪「拷贝重新对齐到当前 pin」这件事（它超出本次重构范围）。

**Phase 2B —— 目标方案（随上游贡献落地）：**

向上游提交一个最小 refactor：把 WAITING 循环的终止条件抽成可 override 的方法：

```python
# upstream vllm/v1/core/sched/scheduler.py
def _should_stop_admitting_waiting(self) -> bool:
    return len(self.running) >= self.max_num_running_reqs
```

上游暴露该接缝后，Ascend 这边的 patch 收敛为：

```python
class BalanceScheduler(Scheduler):
    def _should_stop_admitting_waiting(self) -> bool:
        if super()._should_stop_admitting_waiting():
            return True
        return self._balance_enabled and (
            max(t.item() for t in self.balance_queue) >= self.max_num_running_reqs
        )
```

**结果**：520 行的 `schedule()` 拷贝被彻底删除；该文件不再随上游对 `schedule()` 的修改而漂移。这正是 AGENTS.md 所要求的「向上游贡献的长期计划」。

> **关于 `>=` 与 `==`，以及"临时调低 cap"思路的说明。** 此前曾考虑通过临时设置 `self.max_num_running_reqs = min(cap, max(balance_queue))` 复用上游已有的 break 条件来实现零拷贝。但那会产生**不同**的语义（"让落后 rank 追平 leader"），**明确拒绝** —— 详见下方契约。

### Step 3 —— 配置探测规范化

`_balance_scheduling_enabled()` 有三路 fallback（AscendConfig → additional_config → 环境变量）。`run_engine_core` 拷贝删除后，唯一调用点是 `BalanceScheduler.__init__`，但 AscendConfig 在该时刻是否已初始化仍无法保证（这正是原代码顶部那条 TODO 的由来）。因此**本轮不强行收敛为单路读取**，而是保留三路 fallback 以策安全，仅做两点规范化：

- 环境变量读取从裸 `os.getenv(...)` 改为 `vllm_ascend.envs.VLLM_ASCEND_BALANCE_SCHEDULING`，符合 AGENTS.md「禁止散落 `os.getenv`，统一走 `vllm_ascend.envs`」的要求。
- 顶部 TODO 更新为「待 AscendConfig 初始化时机前移后，可收敛为单次 `get_ascend_config().enable_balance_scheduling` 读取」。

> 后续（待 AscendConfig 时机成熟）：再把三路收敛为单路读取。

## 行为保持契约

本次重构**必须**严格保持以下不变式，任何偏离都视为 bug。

1. **Leader 一满 ⇒ 全员冻结。** `balance_flag` 为 `max(balance_queue) == max_num_running_reqs`，基于上一步 gathered 的 `len(running)` 在每个 rank 上计算。为真时，没有任何 rank 接纳新的 WAITING 请求。这里的比较是与配置的 `max_num_running_reqs` 做 `==`，**不是** `>=`，**不是**"追平 leader"。
2. **相同输入 ⇒ 相同输出。** 在相同的 `self.running`、`self.waiting`、`self.skipped_waiting`、`balance_queue` 与 token budget 下，重构后的 `schedule()` 产生的 `SchedulerOutput` 与当前实现完全一致（相同的 scheduled / preempted / resumed 集合、相同的 `num_scheduled_tokens`、相同的 connector metadata）。
3. **gather 节奏不变。** 每个引擎 step 恰好一次 `all_gather`，作用在同一个 DP group 上，载荷仍是 `len(self.running)`。仅调用位置发生移动。
4. **关闭路径不变。** 当 `enable_balance_scheduling` 为 false 时，行为与上游 `Scheduler` / `DPEngineCoreProc` 逐字节一致（无 gather、无闸门、无自定义 engine core）。关闭分支不得分配 `balance_queue`，也不得执行任何集合通信。
5. **既有约束仍生效。** `profiling_chunk_config` 互斥（见 `vllm_ascend/ascend_config.py`）与 PD-mixed 模式限制（见 `vllm_ascend/platform.py`）仍在原位置校验。

## 重构后文件形态

本轮（Phase 1 + 2A + 3）落地后，关键结构如下。`schedule()` 的函数体仍是上游逐字拷贝（Phase 2B 之前无法删除），但仅含三处文档化 delta：

```python
# vllm_ascend/patch/platform/patch_balance_schedule.py
import torch
import torch.distributed as dist
import vllm
import vllm.v1.engine.core as _engine_core_mod
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import DPEngineCoreProc
# ... 其余 vllm 导入 ...


def _balance_scheduling_enabled(vllm_config) -> bool:
    try:
        from vllm_ascend.ascend_config import get_ascend_config
        return bool(get_ascend_config().enable_balance_scheduling)
    except Exception:
        pass
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if "enable_balance_scheduling" in additional_config:
        return bool(additional_config["enable_balance_scheduling"])
    from vllm_ascend import envs as ascend_envs  # 不再裸用 os.getenv
    return bool(ascend_envs.VLLM_ASCEND_BALANCE_SCHEDULING)


class BalanceScheduler(Scheduler):
    def __init__(self, ...):
        super().__init__(...)
        self._balance_enabled = _balance_scheduling_enabled(vllm_config)
        self.dp_group = None  # 由 BalanceDPEngineCoreProc 在首步前注入
        if self._balance_enabled:
            self.balance_queue = [torch.tensor([0], ...) for _ in range(dp_size)]

    def balance_gather(self):  # 用 self.dp_group，禁用/未注入时 no-op
        if not self._balance_enabled or self.dp_group is None:
            return
        running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
        dist.all_gather(self.balance_queue, running_tensor, group=self.dp_group)

    def schedule(self) -> SchedulerOutput:  # 签名与 pin 的 v0.23.0 一致（无 throttle_prefills）
        if not self._balance_enabled:
            return super().schedule()
        self.balance_gather()  # delta 1: 顶部刷新跨 rank 快照
        # ... 上游 schedule() 函数体（源自比 pin 更老的 vLLM，逐字保留）...
        #        # 在 WAITING 循环里：
        #   balance_flag = max(t.item() for t in self.balance_queue) == self.max_num_running_reqs  # delta 2
        #   if balance_flag:
        #       break
        # ...


class BalanceDPEngineCoreProc(DPEngineCoreProc):
    """仅注入 dp_group，run_busy_loop 完全委托上游。"""

    def run_busy_loop(self):
        self.scheduler.dp_group = self.dp_group
        super().run_busy_loop()


# 上游 run_engine_core 在调用时按模块全局名解析 DPEngineCoreProc，
# 因此 patch 模块级名字即可，无需拷贝 run_engine_core。
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
_engine_core_mod.DPEngineCoreProc = BalanceDPEngineCoreProc
```

本轮规模：**715 → 676 行**。删除了约 95 行的 `run_engine_core` + `run_busy_loop` 拷贝及其失效 import，新增了模块 docstring 与注释；净行数下降不大，但**消除的是陈旧漂移风险**（旧拷贝已落后于上游 `_handle_shutdown` / `eep_scaling_state` / `SignalCallback` 等多处演进）。

> Phase 2B 落地后（上游提供 `_should_stop_admitting_waiting`），`schedule()` 的 ~520 行拷贝才会被删除，文件收敛到约 **60–80 行、零行上游逐字拷贝**。

## 测试计划

1. **签名 + 意图锁定测试（Phase 2A）。** 因为拷贝源自比 pin 更老的 vLLM、无法对已安装 vLLM 做全量比对，改为两点断言：(a) `BalanceScheduler.schedule` 的签名行**必须与已安装 vLLM 的 `Scheduler.schedule` 一致**（这样 `super().schedule()` 才可调用——这条直接卡住了之前误加 `throttle_prefills` 的回归）；(b) balance delta 行（`self.balance_gather()`、`balance_flag` 闸门、禁用路径委托）必须存在于函数体内。
2. **行为等价测试。** 构造一个带 fake DP group 与手工 `balance_queue` 的 `BalanceScheduler`，在多种代表性状态下驱动 `schedule()`（leader 满员冻结、落后 rank 未满、关闭、空 waiting），断言 `SchedulerOutput` 完全一致（契约第 2 条）。可复用 `tests/ut/test_platform.py` 中 balance 用例的测试支架。
3. **gather 节奏测试。** mock `torch.distributed.all_gather`（注意 `balance_gather` 里 `dist = torch.distributed`，mock 目标必须是 `torch.distributed.all_gather` 而非 `vllm.distributed.all_gather`），断言每次 `balance_gather()` 恰好一次 `all_gather`、载荷为 `len(self.running)`、group 为注入的 dp_group（契约第 3 条）。
4. **关闭路径测试。** flag 关闭时，断言未分配 `balance_queue`、未调用 `all_gather`，且 `schedule()` 委托给 `super().schedule()`（契约第 4 条）。
5. **NPU 性能核查。** 按 AGENTS.md 的 NPU 指引，`max(t.item() for t in self.balance_queue)` 每 step 触发一次主机同步（不可避免，因为该值要驱动主机侧控制流）。通过 profile 确认重构**没有**比当前引入额外的同步。

## 分阶段落地

| 阶段 | 范围                                                          | 风险 | 依赖 | 状态 |
|------|---------------------------------------------------------------|------|------|------|
| 1    | gather 移入 `schedule()` 入口；`BalanceDPEngineCoreProc` 精简为 dp_group 注入；删 `run_engine_core`/`run_busy_loop` 拷贝；模块级类替换 | 低 | 无 | ✅ 已完成 |
| 2A   | 签名保持 `schedule(self)`（与 v0.23.0 一致）；禁用路径无参委托 `super()`；签名匹配 + 意图锁定测试 | 低 | 无 | ✅ 已完成 |
| 3    | 配置探测改走 `vllm_ascend.envs`（暂留三路 fallback 待 AscendConfig 时机前移） | 低 | Phase 1 | ✅ 已完成 |
| 2B   | 上游提交 `_should_stop_admitting_waiting` PR；删除 `schedule()` 拷贝 | 中 | 上游 review | ⏳ 待办 |
| 测试 | 漂移回归 / 行为等价 / gather 节奏 / 关闭路径 / NPU 性能核查 | 低 | Phase 1+2A | ⏳ 待办（需 NPU 环境） |

每个阶段可独立发布、独立回退。Phase 1、2A、3 可同期进入同一发布；2B 随上游 PR 合入时机落地。
