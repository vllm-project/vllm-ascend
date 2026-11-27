# PR #22421 `@floatlibai` 检视对齐方案（第一性原理 + 最小安全改动）

> 对应 PR：<https://github.com/vllm-project/vllm-ascend/pull/22421>
> 检视人：`floatlibai`（2026-11-27）
> 目的：在改代码前把「问题本质 → 约束 → 改法 → 依据 → 风险/不做的事 → 验收」写死，避免为了过审而挖出更难维护的坑。
> 原则：**最小修改 = 在不引入新故障面的前提下，把架构接到正确契约上**；不是「改最少行数」。

---

## 0. 老师原意（逐句拆解）

| 原话要点 | 我们理解 |
|----------|----------|
| 对齐 upstream sleep，复用 upstream，不要平行 Ascend-only 路径 | 入口与抽象层跟 vLLM 一致，Ascend 只填后端实现 |
| `HcclCommSuspend` 今天不放 HBM（`hbm_freed_bytes=0`） | 「轻量 suspend」**不能**再当内存收益主路径宣传 |
| TP≥2 仍 destroy/restore | 现状事实：真正能放内存的仍是 destroy |
| `NPUCommunicator` 实现 `DeviceCommunicatorBase.suspend()/resume()`（能力门控、幂等） | 契约落点在 communicator，不在 `rl_config` 私有策略机 |
| 现阶段内部委托 **现有 destroy/restore** 作过渡适配 | 接口先对齐，内核诚实：过渡期 = destroy/restore |
| graph：**recapture 或禁用 graph** 作为显式条件 | destroy 破坏 collective/图假设时，禁止再宣称 inplace keep |
| 单一入口：`enable_nccl_comm_suspend` | 不再扩 `sleep_hccl_policy` / `sleep_acl_graph_policy` 用户面 |
| 方便以后去掉 `rl_config.sleep_mode_extra_cleanup` | 过渡层可删，不能把业务永久焊死在 Ascend 开关上 |

认可点（不要扔）：TP1/TP2 证据、`SleepCycleReport`、inplace 路径的**测量价值**。认可 ≠ 继续把「HcclCommSuspend + inplace」当成最终产品形态。

---

## 1. 第一性原理：真正要解决什么

### 1.1 Upstream 在解决什么

Upstream（vLLM `#51485` / docs）的闭环是：

1. **契约**：`DeviceCommunicatorBase.suspend()/resume()`（默认 no-op）
2. **实现**：CUDA → `PyNcclCommunicator`（`ncclCommSuspend(MEM)` / `Resume`，**幂等**）
3. **编排**：worker sleep/wake 里走 `suspend_device_comms()` / `resume_device_comms()`（walk 所有 group 的 `device_communicator`）
4. **门控**：`enable_nccl_comm_suspend` + `SleepModeBackend.requires_communicator_suspend()`（避免双重处理）
5. **语义承诺**：释放 communicator 动态显存，**保留拓扑/身份** → 通常 **不需要** 为了 comm 去 recapture graph

### 1.2 我们当前 PR 实际在解决什么

| 层 | 当前做法 | 与 upstream 的偏差 |
|----|----------|-------------------|
| 入口 | `rl_config.sleep_mode_extra_cleanup` + `sleep_*_policy` | 平行开关，审阅者/用户学两套 |
| 编排 | `SleepWakeupManager` 自己扫 ProcessGroup / HCCL handle | 绕开 `device_communicator` 多态 walk |
| HCCL TP1 | `HcclCommSuspend` | **不放 HBM**，与「释放 comm 内存」目标不一致 |
| HCCL TP≥2 | destroy/restore（`fallback_destroy`） | 内存行为对了，但挂在 Ascend 私有策略下 |
| ACL Graph | `inplace` / workspace_only / recapture 多策略 | 与 HCCL 是否保身份**耦合不硬**，易出现「destroy 了还 inplace」的语义谎言 |
| Worker | Ascend `sleep_mode_backend` 仍 `NotImplementedError` | upstream 编排接不进来 |

### 1.3 根因（不是「少写几行」）

**根因是分层错位 + 语义不诚实：**

1. **分层**：comm 生命周期应落在 `NPUCommunicator`（upstream 已定契约），我们却在 `device_allocator` 再造一套。
2. **语义**：用「suspend」这个词包装了「不放内存的 HcclCommSuspend」和「放内存的 destroy」，对外一套 `lightweight`，对内两套行为 → 长期无法维护、也无法和 upstream 对照验收。
3. **耦合**：ACL Graph inplace 的正确前提是 **comm 身份仍在**；一旦 HCCL 走 destroy，inplace 就是事故源，不是优化。

### 1.4 长期维护视角下的目标态

```
用户/上游开关: enable_nccl_comm_suspend
        │
        ▼
worker sleep/wake（与 upstream 同层编排）
        │
        ▼
parallel_state.suspend/resume_device_comms()
        │
        ▼
NPUCommunicator.suspend()/resume()   ← 唯一 Ascend 填充点（幂等、能力门控）
        │
        ├─ 过渡期: destroy/restore（诚实放内存）+ Graph 强制 recapture/禁用
        └─ 未来 CANN 真 suspend 放 HBM: 换内核，接口不动；Graph 可回 inplace
```

`rl_config.sleep_mode_extra_cleanup` / `sleep_*_policy`：**过渡兼容层**，不是产品表面；新代码禁止继续加深。

---

## 2. 问题清单 → 改法 → 依据 → 风险

### 问题 A：契约落点错误（最大架构债）

**现象**：`NPUCommunicator` 未实现 `suspend()/resume()`；HCCL 清理在 `SleepWakeupManager`。

**怎么改（最小安全）**：

1. 在 `vllm_ascend/distributed/device_communicators/npu_communicator.py` 增加：
   - `suspend()` / `resume()`（覆盖 base）
   - 实例状态 `_comm_suspended: bool`（幂等，对齐 PyNccl）
   - 能力探测：当前「可安全执行的内存路径」= destroy/restore 是否可用；**不要**把「符号能加载的 HcclCommSuspend」当成「已具备 upstream 同语义能力」
2. `suspend()` 过渡实现：调用**已有、已测过**的该 group 的 destroy/teardown 路径（复用 lifecycle-anchor / MoE teardown 约束，**不新发明**一套 destroy）
3. `resume()` 过渡实现：对应 restore；失败要可观测（log + 返回/抛与 upstream 一致的失败语义）
4. **禁止**在 `suspend()` 里默认走 `HcclCommSuspend` 充当「内存 suspend」——证据已证明 `hbm_freed_bytes=0`

**依据**：

- Upstream review 明确要求 generic surface 在 `DeviceCommunicatorBase`，由各后端覆盖（`#46234`/`#51485`）。
- floatlibai 原文：interface-first + destroy/restore transitional adapter。
- 幂等：upstream 已用 staged wake 证明非幂等会二次 resume 炸引擎；Ascend 必须同构。

**刻意不做**：

- 不删掉整个 `SleepWakeupManager`（一步删光会打断 ACL Graph / evidence / worker 现接线 → 大爆炸）。
- 不在本阶段把 destroy 逻辑「抄一份」进 communicator（双份逻辑更难维护）；优先 **委托** 现有 teardown helpers。

**风险与对冲**：

| 风险 | 对冲 |
|------|------|
| destroy 挂在 communicator 上，遗漏 MoE/anchor | 只委托现有 `prepare_moe_hccl_teardown` / lifecycle anchor 路径；UT 锁「未就绪则 no-op/skip」 |
| 与 `SleepWakeupManager` 双调用 → 双重 destroy | 见问题 C：单一编排入口，manager 改为适配器或只处理 Graph |

---

### 问题 B：入口分裂（`rl_config` vs `enable_nccl_comm_suspend`）

**现象**：用户要开 `rl_config.sleep_mode_extra_cleanup` + policy keys；upstream 用户开 `enable_nccl_comm_suspend`。

**怎么改（最小安全）**：

1. **新主入口**：`enable_nccl_comm_suspend`（或 vLLM 同名配置；Ascend 侧读取同一语义）触发 `NPUCommunicator.suspend/resume` 路径。
2. **旧入口兼容（过渡）**：
   - `sleep_mode_extra_cleanup=true` → 映射到「启用 transitional 路径」，并 **warning_once**：deprecated，请迁到 `enable_nccl_comm_suspend`。
   - `sleep_hccl_policy` / `sleep_acl_graph_policy` / `sleep_allow_fallback`：**停止作为推荐用户面**；内部若仍读取，只做兼容，文档/PR 描述不再推广。
3. **不新增**任何新的 `rl_config` sleep policy key（直接响应检视）。

**依据**：floatlibai「single upstream entry point」「avoids new rl_config policy keys」。

**刻意不做**：立刻物理删除字段（会破坏已有任务配置/老师证据脚本）；用 deprecation + 映射，下个周期再删。

---

### 问题 C：编排层双轨（双重 sleep 清理）

**现象**：Ascend `worker.sleep/wake` 在 CaMem 前后调 `SleepWakeupManager`；upstream 则在 sleep backend 能力下 walk communicators。Ascend `sleep_mode_backend` 仍 NotImplemented。

**怎么改（分两步，避免一次改崩）**：

**C1（本轮必须）——保证同一 sleep 周期 HCCL 只清理一次**

- 选定**唯一 HCCL 所有权**：`NPUCommunicator.suspend/resume`（或由它委托的 helper）。
- `HcclSleepWakeupManager.sleep/wakeup`：改为
  - 要么 no-op + 注释「owned by NPUCommunicator」，
  - 要么薄封装调用 `suspend_device_comms/resume_device_comms`（若 pinned vLLM 已有），
  - **禁止**再私自扫 handle 做第二套 suspend/destroy。

**C2（本轮尽量 / 与 pinned vLLM 对齐）——接上 upstream 编排**

- 评估 pinned vLLM 是否已有 `suspend_device_comms` / worker 调用点。
  - **有**：Ascend worker 走同一调用点；`SleepModeBackend.requires_communicator_suspend()` 在 Ascend backend 返回合理值（若仍 NotImplemented，先实现最小 stub：comm suspend 需要时 True）。
  - **无**（pin 过旧）：在 Ascend `worker.sleep/wake` **与 upstream 相同相对位置**调用 walk；加注释标明「pin 升级后删本地调用，改跟 upstream」。

**依据**：双重处理是 staged wake / 不幂等场景的经典事故源（upstream PR 已踩过）。最小安全改动的核心是 **消除双所有权**，不是少写代码。

---

### 问题 D：ACL Graph 与 HCCL 语义耦合（检视点里的「explicit condition」）

**现象**：当前允许 `lightweight` + `inplace` 组合；但 TP≥2 destroy 时 inplace 不安全；过渡期 HCCL 全面改为 destroy/restore 后，inplace 更不能默认。

**第一性原理约束**：

| HCCL 实际动作 | Graph 合法策略 |
|---------------|----------------|
| 真 suspend 且保身份（未来 CANN） | inplace / workspace_only 可讨论 |
| destroy/restore（过渡期诚实路径） | **必须** `recapture` 或 **禁用 ACL Graph** |
| skip | 保持现状 |

**怎么改（最小安全）**：

1. 过渡期默认：**HCCL transitional destroy ⇒ ACL Graph `recapture`（或显式 `full_cudagraph`/图禁用已存在的配置）**。
2. 在 `SleepWakeupManager` / Graph manager 加**硬约束**：若本周期 `hccl_action_actual` ∈ {destroy, fallback_destroy}，则禁止 `inplace_keep`；违例直接升为 recapture 并打 warning（防御，不是再加用户 knobs）。
3. 保留 inplace 实现代码作**未来**真 suspend 用，但默认路径不再宣传「TP1 suspend+inplace 即任务完成形态」。
4. `SleepCycleReport` 继续记录 `acl_action_actual` / `capture_model_called`，用来证明约束成立。

**依据**：floatlibai「graph recapture (or graph-disabled) as the explicit condition」；图缓存绑定 collective/workspace，destroy 后 keep 是静默正确性风险。

**与老师旧要求的关系**：

- 老师要「suspend 替代 destroy + 不 recapture」——那是 **CANN 真 suspend 放内存之后** 的终态命题。
- 检视人指出今日本机能力达不到终态；应 **诚实分阶段**，而不是用 `HcclCommSuspend`（0 HBM）假装达到终态。
- PR 评论需写清：终态不变，过渡态按 floatlibai 对齐 upstream。

---

### 问题 E：`HcclCommSuspend` 代码资产如何处置

**不要删光，也不要当主路径。**

| 用法 | 决策 |
|------|------|
| 作为「内存释放」主路径 | **否**（证据否定） |
| 作为未来 CANN 能力探测/实验开关（内部） | 可保留在 `hccl_suspend.py`，默认关闭 |
| 出现在用户 `rl_config` 推荐组合 | **否** |
| UT 保留符号加载/幂等探测 | 可，标注 experimental |

**依据**：最小安全 = 保留探测工具，去掉错误产品承诺。

---

### 问题 F：证据与任务验收叙事

**怎么改**：

1. 更新 design / PR 描述：
   - 过渡态：`enable_nccl_comm_suspend` → destroy/restore + graph recapture/disabled；报告 `hbm` 来自 destroy 路径。
   - 终态：CANN suspend 放 HBM 后，换 `NPUCommunicator` 内核 + 再启 inplace 命题。
2. 保留 `SleepCycleReport`（检视人明确说有用）；字段对齐新语义（`hccl_action_actual=destroy` 过渡期为预期，而非 fallback 耻辱）。
3. 重跑最小证据：至少 TP1、TP2 各一轮 sleep/wake+generate；TP2 旧 timeout 若仍在，单独立项，不与本架构对齐混为一谈。

---

### 问题 G：已修过的 Gemini 意见 / CI 流程（非本轮架构）

| 项 | 状态 | 本轮 |
|----|------|------|
| module globals → `_HcclLibState` | 已做 | 不回退 |
| `id(device_group)` → `group_sort_key` | 已做 | communicator 路径继续用稳定 key |
| `ready-precise` | maintainer | 架构对齐后再请标签 |
| Diff coverage | 信息 | 不靠堆无效测试刷率；契约 UT 补 suspend 幂等即可 |

---

## 3. 推荐落地顺序（降低回归面）

```
Phase 0  评论对齐：回复 floatlibai，确认「接口优先 + destroy 过渡 + graph 条件 + 单一入口」
Phase 1  NPUCommunicator.suspend/resume（幂等）委托现有 destroy/restore；UT 幂等/双调用
Phase 2  消灭 HCCL 双所有权：SleepWakeupManager 的 HCCL 分支改为委托/禁用
Phase 3  入口映射：enable_nccl_comm_suspend 为主；rl_config 旧键 deprecation
Phase 4  Graph 硬约束：destroy ⇒ recapture/disable；报告字段更新
Phase 5  Worker 编排对齐 upstream walk（视 pin 版本）；文档/PR/证据刷新
Phase 6  （未来 PR）CANN 真 suspend 放 HBM → 换内核；再评估 inplace
```

**每一 Phase 的合并标准**：可单独 revert；不依赖「下半截没做完才正确」。

本轮 PR 建议至少完成 **Phase 0–4**；Phase 5 若 pin 缺口大，可在 PR 中写明 follow-up，但不得留下双 destroy。

---

## 4. 「最小修改」自检清单（按你的定义）

改每一处前问：

1. **所有权是否唯一？**（HCCL / Graph / CaMem 谁说了算）
2. **失败是否比现在更可诊断？**（log/report 是否撒谎）
3. **是否新增用户面概念？**（有则违反检视；优先映射旧键）
4. **是否复制了一段 teardown？**（有则改为委托）
5. **staged wake / 二次 sleep 是否安全？**（幂等）
6. **TP1 与 TP≥2 是否同一套契约？**（允许能力降级，不允许两套入口）
7. **以后删 `sleep_mode_extra_cleanup` 是否只删适配层？**（若答案是「要改业务核心」→ 分层失败）

---

## 5. 明确不做什么（防止「为改而改」）

1. 不再 orphan / 无父提交 force-push。
2. 不把社区任务文档、截图、计划 md 当成 runtime 依赖。
3. 不新增 `sleep_*` policy 枚举值「凑能力」。
4. 不在未接幂等前把 communicator suspend 接到生产默认开。
5. 不假装 `HcclCommSuspend` 已满足 upstream 内存语义。
6. 不为刷 coverage 写无行为断言。

---

## 6. 建议给 `@floatlibai` 的回复要点（Phase 0）

- 同意：对齐 `DeviceCommunicatorBase.suspend/resume`，单一入口 `enable_nccl_comm_suspend`。
- 承认：当前 `HcclCommSuspend` 不放 HBM；过渡期 communicator 内委托 destroy/restore。
- Graph：过渡期与 destroy 绑定，显式 recapture 或 disable；inplace 留到真 suspend。
- `rl_config.sleep_mode_extra_cleanup` / policy keys：只做兼容与弃用，不再扩展。
- `SleepCycleReport` 保留，用于证明过渡语义。
- 终态仍服务老师命题，但分阶段，不把过渡态包装成终态。

---

## 7. 验收标准（本轮改完怎么算过）

1. **契约**：`NPUCommunicator` 有幂等 `suspend/resume`；UT 覆盖 double-sleep / wake-without-sleep。
2. **入口**：文档与 PR 描述以 `enable_nccl_comm_suspend` 为准；无新 `rl_config` sleep keys。
3. **诚实**：过渡期报告中 HCCL 动作为 destroy/restore；不再用 0 HBM 的 suspend 冒充内存收益。
4. **耦合**：HCCL destroy 周期内不可能出现 `acl_action_actual=inplace_keep`（UT 锁死）。
5. **无双清理**：同一 sleep 周期 HCCL teardown 只走一条所有权链。
6. **可演进**：未来只替换 `NPUCommunicator` 内部实现即可启用真 suspend + inplace，无需再发明第三套 manager。
