# 异步调度下的 Dump / 检测时序设计

> 说明 async scheduling 下占位符、dump start/finalize 与跨 TP OR 的设计约束。  
> 涉及模块：`vllm_ascend/dfx/dumper/`、`vllm_ascend/dfx/processor.py`、`vllm_ascend/dfx/detector/`、  
> `vllm_ascend/worker/v2/model_runner.py`、`vllm_ascend/worker/model_runner_v1.py`  
> DFX 总览：[dfx_design.md](./dfx_design.md)；运维：[dfx_ops.md](./dfx_ops.md)
>
> **路径现状**：现实现为 `vllm_ascend/dfx/dumper/`（含 `Dumper`）+ `DfxProcessor.check_after_sample`（runner 只调 processor 阶段 hook）。
> **pending-OR / fast-path 实现真相**以 [dumper_design.md](./dumper_design.md) §5 为准；本文侧重 async 占位符与「下步生效」时序。完整 I/O 走 DFX report（`save_sensitive_info`），日志只打长度。

---

## 目录

- [1. 核心问题概览](#1-核心问题概览)
- [2. 背景：同步 vs 异步调度模型](#2-背景同步-vs-异步调度模型)
- [3. 问题一：sampled_token_ids=-1 占位符](#3-问题一sampled_token_ids-1-占位符)
- [4. 问题二：Dumper start/finalize 的「下步生效」机制](#4-问题二dumper-startfinalize-的下步生效机制)
- [5. 问题三：异步模式下 TP 跨 Rank Dump 对齐](#5-问题三异步模式下-tp-跨-rank-dump-对齐)
- [6. 解决方案总结](#6-解决方案总结)

---

## 1. 核心问题概览

| 问题 | 现象 | 根因 | 影响范围 |
|------|------|------|----------|
| -1 token | `sampled_token_ids=[-1,-1,-1]` | async 下先写占位符，D2H 回填滞后 | v1/v2 |
| 下步生效 | 触发 dump 后抓不到当前步 | token_logprob 异常在 `get_output()` 检测，只能 arming 下一步 | v2 async |
| TP 跨 Rank 对齐 | 部分 TP rank 没 dump | async / sync+TP>1 仅 TP0 检测，需 pending-OR | v1/v2 |

---

## 2. 背景：同步 vs 异步调度模型

### 2.1 同步调度

```
┌─────────────────────────────────────────────────────────────┐
│ Step N                                                       │
│                                                              │
│  execute_model() ──► forward ──► sample_tokens()             │
│       │                              │                       │
│  start_dump_data()           finalize_dump_data()            │
│                                     │                       │
│                              D2H 同步完成                     │
│                              logprobs 已就绪                  │
│                              check_after_sample() ✓          │
└─────────────────────────────────────────────────────────────┘
```

同步路径下，`sample_tokens()` 调用 `get_output()` 后数据已全部就绪，dumper 可以在同一 step 内完成检测→打标记→dump。

### 2.2 异步调度

```
┌─────────────────────────────────────────────────────────────┐
│ Step N                                                       │
│                                                              │
│  execute_model() ──► forward ──► sample_tokens()             │
│       │                              │                       │
│  start_dump_data()           AsyncOutput (未 D2H!)           │
│                                      │                       │
│ finalize_dump_data()  ←────────── get_output() 未调用        │
│      (dump=False,                              │             │
│       占位消费被跳过)                  logprobs 还在 GPU        │
│                                                              │
│ ═══════════════════ 时间线重叠 ═══════════════════            │
│                                                              │
│ Step N+1                                                     │
│                                                              │
│  sync_dump_pending_or()  ← 真正 D2H 完成                  │
│         │                         │                          │
│    OR pending_dump          check_after_sample() ✓          │
│         │                                                     │
│    start_dump_data()                                          │
│         │                                                     │
│    forward (dump 抓取 Step N 的异常!)                         │
└─────────────────────────────────────────────────────────────┘
```

关键在于：异步路径下，**数据检测**和**数据产出**分离——检测发生在下一步，dump 抓取的也是下一步的 forward。

---

## 3. 问题一：sampled_token_ids=-1 占位符

### 3.1 根因

在 async scheduling 路径下，`_bookkeeping_sync` 先写入占位符 `[-1]`：

**v1 关键代码** (`model_runner_v1.py:2849`)：
```python
if self.use_async_scheduling:
    sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
    # ...
    req_state.output_token_ids.extend(sampled_ids)  # L2873 写入占位符
```

**v2 上游** (`vllm/v1/worker/gpu/input_batch.py`)：
```python
def set_async_sampled_token_ids(...):
    # 只有上一轮 sampling_metadata.output_token_ids 非空时才保存 CPU 侧真值
    # 否则占位符 [-1] 不会被回填
```

### 3.2 真值回填链路

```
vllm gpu_input_batch.set_async_sampled_token_ids()
    │
    ├── 保存 sampled_token_ids_cpu (真值)
    ├── 记录 event 信号
    │
    └── input_batch.req_output_token_ids[req_idx]
            │
            └── 异步修正: 将占位符 [-1] 替换为真实 token
```

**问题**：真值回填只覆盖 `sampling_metadata.output_token_ids`，**不覆盖 `req_state.output_token_ids`**。所以从 `req_state` 读取的 `output_token_ids` 永远全是 `-1`。

### 3.3 对 Dumper 的影响

```python
# 读 input_batch 的 req_output_token_ids（async 修正后）
req_output_token_ids = getattr(self.runner.input_batch, "req_output_token_ids", None)
# ✅ 这里读的是 input_batch，已经被异步修正
output_token_ids_raw = req_output_token_ids[req_idx]  # 真实值

# 但如果退化到 req_state:
output_token_ids_raw = getattr(req_state, "output_token_ids", None)
# ❌ 这读到的全是 -1！
```

### 3.4 时序图

```mermaid
sequenceDiagram
    participant GPU as GPU
    participant MR as ModelRunner
    participant IB as InputBatch
    participant RS as ReqState
    participant D as Dumper

    Note over GPU, D: === Step N ===

    GPU->>MR: forward 完成, sampled_token_ids = [383, 11]

    MR->>IB: set_async_sampled_token_ids([383,11])
    Note over IB: 保存真值 + event (异步)

    MR->>RS: output_token_ids.extend([-1])
    Note over RS: 写入占位符!

    rect rgb(255, 230, 230)
    Note over D,RS: ❌ 此时 Dumper 如果读 req_state，全是 -1
    end

    Note over GPU, D: === Step N+1 ===

    MR->>IB: event.synchronize() (D2H 完成)
    IB->>IB: 回填: output_token_ids[-1]→383 替换
    Note over IB: req_output_token_ids = [383, 11]

    rect rgb(230, 255, 230)
    Note over D,IB: ✅ 此时 Dumper 读 input_batch，获得真值
    end
```

---

## 4. 问题二：Dumper start/finalize 的「下步生效」机制

### 4.1 核心概念

Dumper 维护了一组状态机字段来控制 msprobe dump 的生命周期：

```python
# dumper.py
self._msprobe_dump_active = False   # 当前是否在 dump 周期中
self._dump_needs_forward = False    # 是否需要一个真正的 forward 来完成 dump
self._dump_forward_seen = False     # dump-capable forward 是否已经过
self._pending_dump = False          # 异步模式下是否 pending（等 OR sync）
```

### 4.2 为什么「下步生效」

```
Step N:
  sample_tokens → check_after_sample → 发现异常!
    │
    ├── enable_msprobe_dump_if_needed()
    │       └── _activate_msprobe_dump()
    │               ├── _msprobe_dump_active = True
    │               └── _dump_needs_forward = True   ← 还需要一个 forward!
    │
    └── 【Step N 的 forward 已经结束了！所以 N 步 dump 不了】

Step N+1:
  execute_model 入口:
    sync_dump_pending_or()  ← 异步下 OR sync pending
    start_dump_data()
      └── _msprobe_dump_active = True? → 启动 PrecisionDebugger.start()
      └── _dump_forward_seen = True    ← 标记 dump forward 已开始
    forward()  ← 这一轮 forward 的数据才是 dump 的内容!
    finalize_dump_data()
      └── _debugger.step() → 落盘
      └── disable_msprobe_dump_if_needed()
```

### 4.3 关键守卫逻辑

`disable_msprobe_dump_if_needed()` 中有防止过早关闭的守卫：

```python
def disable_msprobe_dump_if_needed(self) -> None:
    if not self._msprobe_dump_active:
        return

    # ⚠️ 异步检测可能在 start 之后才 enable dump
    # 必须等 forward 跑完才能 disable
    if self._dump_needs_forward and not self._dump_forward_seen:
        logger.debug(
            "[Anomaly msprobe] disable deferred (needs forward) %s",
            self._dump_rank_tag(),
        )
        return  # ← 不能关! 还没 dump 到 forward

    self.set_msprobe_dump_state(False)
    self._msprobe_dump_active = False
    self._dump_needs_forward = False
    self._dump_forward_seen = False
```

### 4.4 时序图

```mermaid
sequenceDiagram
    participant MR as ModelRunner
    participant D as Dumper
    participant DG as PrecisionDebugger

    Note over MR, DG: Step N — 常规推理

    MR->>D: sync_dump_pending_or()
    Note over D: active=False, no-op

    MR->>D: start_dump_data()
    Note over D: debugger=None or not active, no-op

    MR->>MR: forward
    MR->>D: finalize_dump_data()

    MR->>MR: sample_tokens()
    MR->>D: check_after_sample()
    Note over D: 🔴 发现异常! 但 forward 已完成

    D->>D: _activate_msprobe_dump()
    Note over D: active=True, needs_forward=True
    Note over D: 但 Step N 已经没法 dump 了!

    Note over MR, DG: === Step N+1 — dump 步 ===

    MR->>D: sync_dump_pending_or()
    Note over D: active=True → 进入 dump 模式

    MR->>D: start_dump_data()
    DG->>DG: PrecisionDebugger.start(model)
    Note over DG: forward 的每层输出都会被记录

    MR->>MR: forward (dump 抓的其实是这步的数据!)
    Note over MR: ⚠️ dump 内容是 N+1 步，不是 N 步

    MR->>D: finalize_dump_data(dump=True)
    DG->>DG: step() → 落盘
    D->>D: disable_msprobe_dump_if_needed()
    Note over D: active=False, needs_forward=False
```

---

## 5. 问题三：异步模式下 TP 跨 Rank Dump 对齐

> pending-OR / fast-path 细节与现行代码以 [dumper_design.md](./dumper_design.md) §5 为准。现行实现中 **Sync+TP>1 也走 pending-OR**（不仅 async）。

### 5.1 问题

同步模式下，每个 last-PP TP rank 各自检测异常→各自激活 dump，没问题。

异步模式下，只有 **TP0** 调用 `get_output()`（vLLM 框架确保 output_rank = TP0），所以检测逻辑只在 TP0 运行。如果 TP0 触发 dump，其他 TP rank（TP1, TP2...）不知道，导致：

- TP0 启动了 PrecisionDebugger
- TP1 没启动
- TP0 在执行 `all_reduce` 等集合通信时，TP1 因为没启动 debugger 而行为不一致 → **死锁或数据错乱**

### 5.2 「Pending + OR Sync」机制

**核心思路**：检测 → arm pending → 下步 `execute_model` 入口 OR sync → 所有 last-PP TP 同时激活。

伪代码（示意；以 `vllm_ascend/dfx/dumper/pending.py` 为准）：

```python
# sync_dump_pending_or — async 或 sync+TP>1 均走 OR
# 热更关且 dump 未激活时 fast-path：跳过 all_reduce
tp_group = get_tp_group()
local = 1 if self._pending_dump else 0
pending_t = torch.tensor([local], dtype=torch.int32)
if tp_group.world_size > 1:
    torch.distributed.all_reduce(pending_t, group=tp_group.cpu_group)
if int(pending_t.item()) <= 0:
    return False
self._activate_msprobe_dump(req_id)  # 每个 last-PP TP 独立激活
self._clear_pending_dump()
```

### 5.3 时序图

```mermaid
sequenceDiagram
    participant TP0 as TP Rank 0
    participant TP1 as TP Rank 1
    participant ALL as All-Reduce

    Note over TP0, TP1: === Step N: TP0 检测到异常 ===

    TP0->>TP0: get_output() D2H 完成
    TP0->>TP0: check_after_sample()
    Note over TP0: 🔴 发现异常!

    TP0->>TP0: enable_msprobe_dump_if_needed()
    Note over TP0: async → arm pending_dump
    Note over TP0: _pending_dump = True
    Note over TP0: 不激活 dump (等 OR sync)

    Note over TP1: TP1 不调用 get_output()
    Note over TP1: 不知道 TP0 发现了异常

    Note over TP0, TP1: === Step N+1: execute_model 入口 ===

    TP0->>ALL: sync_dump_pending_or()
    Note over TP0: local = 1 (pending)
    TP1->>ALL: sync_dump_pending_or()
    Note over TP1: local = 0 (no pending)

    ALL->>ALL: all_reduce SUM → result = 1
    Note over ALL: 1 > 0 → 所有 rank 都激活!

    TP0->>TP0: _activate_msprobe_dump() → true
    TP1->>TP1: _activate_msprobe_dump() → true

    TP0->>TP0: start_dump_data() → debugger.start()
    TP1->>TP1: start_dump_data() → debugger.start()

    Note over TP0, TP1: ✅ 所有 TP rank 同步进入 dump 模式
    Note over TP0, TP1: 后续 all_reduce 等集合通信不会死锁
```

---

## 6. 解决方案总结

### 6.1 已落地修复

| 编号 | 修复 | 位置 |
|------|------|------|
| 1 | "Pending + OR Sync"：TP0 arm pending，下步 `sync_dump_pending_or` 广播（含 sync+TP>1） | `dfx/dumper/pending.py` |
| 2 | `disable_msprobe_dump_if_needed` 的 `_dump_needs_forward` 守卫，防止没 forward 就关 | `dfx/dumper/` |
| 3 | `finalize_dump_data(dump=False)` 在 dummy/capture 时不消费 pending | `dfx/dumper/` |

### 6.2 仍存在的局限性

| 局限性 | 说明 |
|--------|------|
| dump 抓不到异常当步 | token_logprob 异常检测在 `get_output()`，只能 arm 下一步 dump。下一步 forward 的数据 ≠ 异常步的数据 |
| 占位符 `-1` 残留 | `req_state.output_token_ids` 仍然会被 `-1` 污染，依赖方需改用 `input_batch.req_output_token_ids` |

### 6.3 未来改进方向

1. **异常步快照**：在检测到异常时，在当前 step 内缓存需要 dump 的数据（如 hidden_states 的某层输出），而非依赖下一步 forward
2. **统一 req_state 数据源**：让 `req_state.output_token_ids` 也能被异步修正，或标记为 deprecated，统一使用 `input_batch.req_output_token_ids`
