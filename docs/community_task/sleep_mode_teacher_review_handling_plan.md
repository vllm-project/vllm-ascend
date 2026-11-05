# Sleep Mode PR #20481 老师评审处理方案

> 对应 PR：https://github.com/vllm-project/vllm-ascend/pull/20481  
> 对应任务：HiAscend `7851660d93394113bef9c19d3d2ce0ce`  
> 对应 RFC/Issue：#16558 / #20155  
> 老师评论作者：`yjyang62`（2026-11-15）  
> 文档目的：在改代码前，把「问题理解 → 处理动作 → 依据 → 自检」写清楚，确认方向正确后再做最小优化落地。
>
> **落地状态（本 PR 后续提交）**：动作 A/D（`SleepCycleReport` + `*_action_actual`）已实现；动作 B 增加 UT 命题断言 + TP=1 e2e 多轮门禁；动作 C 写入 design §9 Teacher review gate；动作 E 在 PR 评论回复老师。

---

## 0. 老师原意（逐句拆解）

老师原话要点：

> The intent of this task is: suspend HCCL during sleep instead of destroying it, while still getting the memory-release benefit, and not recapturing ACL Graphs after wake-up. Please add tests that prove whether all three can hold at the same time — covering TP=1/TP=2, multiple sleep/wake cycles, and the first inference after wake — with measured memory-release numbers and evidence of whether recapture happened.
>
> If that is not achievable, please analyze in detail which layer blocks it: whether HcclCommSuspend itself does not release device memory, or whether the communicator lifecycle plus the stream/context binding to captured graphs forces a rebuild. Give a clear conclusion. That also completes the task.

拆成可验收命题：

| ID | 命题 | 含义 |
|----|------|------|
| P1 | Sleep 时 Suspend HCCL，而不是 Destroy | 通信域身份尽量保留；不是 #9585 那条「拆 PG 再重建」主路径 |
| P2 | 仍拿到 memory-release benefit | Sleep 后 HBM free 有可测量正收益（至少相对 cleanup 前） |
| P3 | Wake 后不 recapture ACL Graph | 不走 `capture_model()`；保留已捕获 entries，workspace 可懒重建 |
| E1 | 三命题能否**同时**成立 | 不是分别成立；要联合证据 |
| E2 | 覆盖矩阵 | TP=1、TP=2；多轮 sleep/wake；wake 后第一次推理 |
| E3 | 量化证据 | 内存释放数值 + 是否发生 recapture |
| A1 | 若 E1 不成立 | 分层定位阻塞层，给出明确结论；这也算完成任务 |

**自检（方向是否读对）**：老师没有要求「再加一堆 policy / 重写 FSM / 强行让 TP≥2 Suspend 成功」。他要求的是 **证明或证伪 + 分层结论**。后续处理必须围绕证据闭环，而不是功能扩面。

---

## 1. 当前 PR 现状与缺口（有代码依据）

### 1.1 已有能力（不必推倒重来）

| 能力 | 代码位置 | 说明 |
|------|----------|------|
| HCCL Suspend/Resume 封装 | `vllm_ascend/device_allocator/hccl_suspend.py` | 对齐上游 NCCL sleep 思路；含 `comm_to_int`、幂等 handle set |
| Sleep 生命周期编排 | `sleep_mem_optimized.py` 中 `SleepWakeupManager` / `HcclSleepWakeupManager` / `AclGraphSleepWakeupManager` | ACL 先于 HCCL；wake 时 HCCL 未完成则推迟 ACL/anchor |
| inplace ACL | `AclGraphSleepWakeupManager.sleep/wakeup` | inplace 保留 entries，只清 workspace；缺 entry 且允许时 fallback recapture |
| 内存采样 | `SleepWakeupManager.sleep` 内 `torch.npu.mem_get_info` | 已有 GiB 日志，但不是结构化验收产物 |
| 设计文档 | `docs/community_task/design_sleep_mode_lifecycle.md` | 已写明 TP≥2 lightweight 实际走 destroy/restore |

### 1.2 与老师要求的缺口

| 缺口 | 依据 | 为什么不合理地「装作已完成」 |
|------|------|------------------------------|
| G1：缺少联合验收测试 | UT 多为 mock（`test_acl_graph_inplace_skips_recapture_on_main_path`、`test_hccl_lightweight_suspend_path_on_main` 等） | Mock 只能验控制流，不能证明真 Suspend、真释放、真无 recapture、真能首次推理 |
| G2：缺少 recapture 证据钩子 | `capture_model()` 调用没有对外可断言的 cycle 级标志 | 老师明确要 “evidence of whether recapture happened”；仅靠日志字符串脆弱 |
| G3：TP≥2 名实不符 | `HcclSleepWakeupManager.sleep`：`lightweight` + multi-rank → `destroy_hccl()` | 设计文档已承认；但对外仍叫 lightweight，长期会误导维护/评审，也让「P1 是否成立」含糊 |
| G4：分层结论未用实测填死 | design 写了现象表，但 PR 未给出可复核的测量记录与阻塞层判定 | 老师允许「做不到也行」，但前提是 **clear conclusion**；现在是「经验描述」，证据强度不够 |
| G5：e2e/nightly 未覆盖该组合 | `tests/e2e` 有 sleep 相关 yaml/用例，但未见对本 PR「lightweight+inplace 联合命题」的专用验收 | 多轮 / 首次推理 / TP 矩阵缺少固定入口 |

### 1.3 对 Gemini 评论的处理立场（次要）

Gemini 早期指出的 `comm_to_int`、inplace 缺 entry fallback、危险 `popitem` 等，在当前 head 中已基本消化。  
**本次不以 Gemini 清单为主驱动**；只在落地时做一次对照，避免回退。依据：老师评论才是任务完成定义。

---

## 2. 总体策略（先定方向，再谈改法）

### 2.1 策略一句话

**保留现有实现骨架；补「可观测 + 可验收 + 可归档结论」；只做让命题可判定的最小代码改动；不为假装三命题在 TP≥2 成立而硬改 CANN/PG 行为。**

### 2.2 为什么这是正确方向（复核）

1. **任务完成条件已被老师定义为证据或结论**，不是「必须三命题全绿」。强行在 TP≥2 上 Suspend→Resume，与本 PR design 中已记录的 hang / allreduce 失败冲突，属于高风险扩面，违反最小修改。
2. **现有 destroy fallback 是安全网，不是脏代码本身**；问题是「证据不足 + 语义不透明」。长期维护更需要诚实的 action 标签，而不是再叠一套并行生命周期。
3. **观测字段一旦稳定，后续 CANN/torch_npu 升级可回归**；一次性手工 log 贴评论无法沉淀。
4. **最小修改原则**：不新增第三套 Sleep 框架；不新增大量用户开关；优先复用已有 `mem_get_info`、policy、manager 边界。

### 2.3 明确不做（Non-goals）

| 不做 | 理由 |
|------|------|
| 重写 FSM / 再抽象一层 Manager | 无助于老师命题；增加维护面 |
| 为 TP≥2 强推 raw Suspend→Resume | design 已记录 hang / verify 失败；与「安全可维护」冲突 |
| 新增更多 `rl_config` 开关 | 当前 3 个 policy 已够表达意图；再加开关会扩散配置矩阵 |
| 用更多 mock UT「刷覆盖」冒充完成 | 无法满足 E2/E3 |
| 大范围改 worker / model_runner 主路径 | Sleep extra cleanup 应继续收敛在 device_allocator |

---

## 3. 处理动作清单（每个动作：做什么 / 依据 / 解释 / 自检）

### 动作 A — 引入 Sleep Cycle Report（结构化观测）

**做什么**

在一次 sleep→wake（可选含首次推理钩子）结束后，产出固定结构的 report（内存对象 + 结构化 log），建议字段：

```text
round
tp_size
hccl_policy_requested          # lightweight|destroy|skip
hccl_action_actual             # suspend|destroy|skip|fallback_destroy
acl_policy_requested           # inplace|workspace_only|recapture|skip
acl_action_actual              # inplace_keep|recapture|skip|fallback_recapture
hbm_free_before_bytes
hbm_free_after_sleep_bytes
hbm_freed_bytes
capture_model_called           # bool
wake_generate_ok               # bool | null（若本轮未测推理）
notes                          # 短字符串，如 multi-rank fallback reason
```

落地原则：

- 字段写在现有 manager 内（例如 `SleepWakeupManager` 聚合 HCCL/ACL 子 manager 的结果），避免散落全局变量。
- `capture_model_called` 在 `AclGraphSleepWakeupManager.wakeup` 真正调用前后置位。
- `hccl_action_actual` 在 suspend 成功 / destroy fallback / skip 分支分别置位（解决 G3 名实问题的观测侧）。

**依据**

- 老师 E3：要 measured memory-release numbers + recapture evidence。
- AGENTS.md：避免魔法状态；可维护性优先；NPU 热路径避免无必要 sync——report 只在 sleep/wake 边界采样（本路径本来就会 sync/barrier），不引入新的 per-token sync。

**解释**

没有统一 report，就只能靠「人读 log」证明 P1–P3。人读不可回归，也无法在 CI/nightly 里断言。把「实际动作」从「请求的 policy 名」中拆开，是长期维护的关键：否则 `lightweight` 在 TP=2 继续误导。

**自检**

- 是否最小？是：不改生命周期顺序，只记录已有分支结果。
- 是否过度设计？控制在一个 dataclass/dict + 一次 info log；不做独立 telemetry 子系统。
- 是否可能造假？字段必须由代码路径赋值，测试断言字段，而不是解析自然语言 log。

---

### 动作 B — 增加「命题验收」测试入口（优先真机/e2e，UT 只守不变量）

**做什么**

1. **保留并精炼现有 UT**（控制流不变量）：inplace 不调用 capture；barrier 在 suspend 前；resume 不乱匹配 handle 等。这些仍有价值，但**不宣称完成老师任务**。
2. **新增 NPU 验收用例/脚本**（完成老师任务的主证据），矩阵：

| Case | TP | Rounds | 断言重点 |
|------|----|--------|----------|
| B1 | 1 | ≥2 | `hccl_action_actual=suspend`；`capture_model_called=false`；`hbm_freed_bytes>0`（或 ≥ 实测下限）；wake 后首次 generate OK |
| B2 | 2 | ≥2 | 记录真实 `hccl_action_actual`；若只能 `fallback_destroy`，则 **不断言 P1**，而是断言安全唤醒 + 首次 generate OK，并把结论写入 design |
| B3 | 1/2 | 对比 | 可选：同配置下 `destroy+recapture` baseline 的 freed / wake 延迟，证明 lightweight+inplace 的收益或边界 |

实现形态优先顺序（最小侵入）：

1. 若仓库已有 one_card / two_card sleep 或 RLHF sleep 基础设施，挂一条参数化 case；
2. 否则先提供 `tests/e2e/...` 或 `docs/community_task/` 可复现脚本 + 固定输出格式（JSON/表格），再在 nightly 中接入。

**依据**

- 老师 E1/E2：要证明能否同时成立，并覆盖 TP/多轮/首次推理。
- 项目测试分层：UT 在 `tests/ut/`，硬件行为在 e2e/nightly（AGENTS.md）。

**解释**

老师要的是「物理世界真值」，不是「mock 返回值」。TP=1 与 TP=2 必须分开：本 PR 已观察到多卡 Suspend 与 PG/graph 绑定问题；把它们塞进同一个「必须全绿」断言，会逼出错误实现。

**自检**

- 会不会因为环境缺 CANN Suspend 符号而误红？验收用例应区分：`suspend unavailable` → 记结论并走已知 fallback，而不是假装 P1 成立。
- 会不会测到 allocator 主路径（weights/KV）掩盖 HCCL/ACL 收益？测量应尽量对准 extra-cleanup 段（现有 `_measure_memory_released` / sleep manager 前后），并在报告中注明「包含/不包含 allocator sleep」。
- 首次推理为何必要？inplace 的 workspace 懒重建、HCCL resume 后的 collective 正确性，只有第一次真实 forward/generate 才能暴露。

---

### 动作 C — 用实测填充「分层阻塞分析」，写入 design（任务闭环）

**做什么**

在 `design_sleep_mode_lifecycle.md` 增加一节 **「Teacher Review Gate / Empirical Conclusion」**，按层填表（数值来自动作 B）：

| 层 | 问题 | TP=1 结论 | TP=2 结论 | 证据 |
|----|------|-----------|-----------|------|
| L0 CANN | `HcclCommSuspend` 后 device memory 是否下降？ | 待填 | 待填 | report.`hbm_freed_bytes` + 必要时只跑 HCCL cleanup 的对照 |
| L1 torch_npu PG | Suspend 后 `resume_hccl_comm` / `reinit(rebuild_link=False)` 是否足以恢复 TP collective？ | 待填 | 待填 | wake 后 allreduce/generate |
| L2 Graph binding | 保留 ACL Graph entries 时，与 HCCL/stream/context 是否仍可 replay？ | 待填 | 待填 | `capture_model_called` + 首次推理 |
| L3 联合命题 | P1∧P2∧P3 是否同时成立？ | 待填 | 待填 | 综合 |

结论书写模板（强制二选一，禁止含糊）：

- **成立**：`TP=x 上 P1∧P2∧P3 同时成立；证据见 <report/log/commit>`  
- **不成立**：`在 TP=x 上被 L? 阻塞；具体表现为 ...；因此工程默认路径为 ...；任务以该结论完成`

**依据**

- 老师 A1：做不到也可以，但要 clear conclusion，并指出是 Suspend 不释存，还是 communicator lifecycle + graph binding 逼迫 rebuild。
- 现有 design 已有经验表，缺的是「可复核数据 + 判定句」。

**解释**

这是任务验收文档，不是宣传材料。把「fallback 是缺陷」改写成「在当前 CANN/torch_npu 约束下的安全策略」，才符合长期维护：后人升级驱动后，只需重跑动作 B 更新表格，而不是重新猜。

**自检**

- 结论是否可能过度归因？必须用对照实验区分 L0 与 L1/L2：例如「仅 Suspend、不碰 Graph」vs「Suspend+inplace」。若无硬件条件，标注 `blocked-on-hardware`，不编造数字。
- 是否把 destroy fallback 说成「实现完成了 P1」？禁止。P1 在 fallback 时记为不成立，并指向 L1/L2。

---

### 动作 D — 语义诚实化（小改代码/文档，不大拆行为）

**做什么**

1. 日志与 report 使用 `hccl_action_actual=fallback_destroy`（或同等枚举），即使 `sleep_hccl_policy=lightweight`。  
2. design / PR description 明确一句：

   > `lightweight` means “prefer Suspend when safe”; on multi-rank with current CANN 9.x + torch_npu, safe path remains destroy/restore.

3. **暂不**改用户可见默认值语义去拆成两个 policy（避免配置爆炸）；若老师后续要求 API 更清晰，再单开小 PR 做 rename/拆分。

**依据**

- G3：名实不符是长期维护债务。  
- 最小修改：先诚实观测与文档，不动默认行为，避免无谓迁移成本。

**解释**

老师挑战的是「三件事能否同时成立」。若 TP=2 实际 destroy，却继续在摘要里写「已 lightweight」，评审会认为证据被稀释。语义诚实化让「证伪」也可被接受。

**自检**

- 会不会变成大 refactor？不会：只改分支赋值与文档句子。  
- 会不会 break 外部依赖 policy 字符串？不改枚举对外取值，只增加 actual action 记录。

---

### 动作 E — PR 回复老师（证据附上，不空谈）

**做什么**

在 #20481 评论中用中英均可，结构固定：

1. 重述三命题理解（确认对齐）  
2. 给出 TP=1 / TP=2 实测表（来自动作 B/C）  
3. 若有不成立：指出阻塞层 L0/L1/L2 + 工程默认策略  
4. 指向 design 新章节与测试入口  
5. 说明未做事项（Non-goals），避免预期漂移

**依据**

- 评审沟通成本；老师说 “That also completes the task”，需要把结论交到评论区，而不是只埋在代码里。

**解释**

代码合并与任务验收是两件事。老师作为任务对接视角，需要一屏可读的结论。

**自检**

- 回复是否早于数据？应在动作 B/C 有结果后再发最终结论；若需先同步方向，可先回「按证据闭环推进」短评，避免空头承诺。

---

### 动作 F — Gemini 残留对照（低优先级、防回退）

**做什么**

落地前快速核对：

- `comm_to_int` 全路径使用  
- inplace 缺 entry 时 fallback  
- resume 不使用任意 `popitem` 匹配  

**依据**

正确性底线；但不是老师完成定义。

**自检**

不因为 Gemini 建议引入新的复杂 helper 层。

---

## 4. 建议落地顺序（最小风险）

```text
1) 动作 A：Cycle Report 字段（可单测断言字段赋值）
2) 动作 D：分支写入 actual action（与 A 一起提交，改动面同一处）
3) 动作 B：NPU 验收入口（先 TP=1，再 TP=2）
4) 动作 C：design 填结论表
5) 动作 E：回复老师
6) 动作 F：对照清单（随时可穿插）
```

依据：先有可观测，才有可测；先有可测，才有可写结论；最后才对外关闭评审问题。顺序反了容易「先改行为、后补故事」。

---

## 5. 成功标准（Definition of Done）

同时满足：

1. 存在可运行的验收入口，覆盖 TP=1 与 TP=2、≥2 轮、wake 后首次推理。  
2. 每次关键路径产出 Cycle Report，含内存字节与 `capture_model_called`。  
3. design 中有二选一的明确结论句（成立 / 在 L? 阻塞）。  
4. PR 评论把结论交给老师。  
5. 代码 diff 仍集中在 `device_allocator` + 测试/文档；无无关重构。

---

## 6. 方向复核结论（写文档时的自审）

| 检查项 | 结果 |
|--------|------|
| 是否对齐老师完成定义？ | 是：证据或分层结论 |
| 是否最小修改？ | 是：观测 + 验收 + 文档诚实化；不重写生命周期 |
| 是否有利于长期维护？ | 是：actual action 与 report 可回归；避免 lightweight 名实陷阱 |
| 是否存在「用文档代替修复」的偷懒？ | 否：若 TP=1 三命题可成立，测试必须强制成立；只有客观阻塞才允许结论式完成 |
| 是否与现有 design 冲突？ | 否：是把 design 里已有经验升级为可复核门禁 |
| 最大风险 | 无 NPU/多卡时无法填数值 → 应标注环境限制，而不是编造 |

**结论：方向正确，可以进入优化落地阶段。**  
优化时优先实现动作 A+D（同一小 diff），再接 B/C/E。

---

## 7. 落地时的优化约束（确认方向后执行）

1. **API 表面最小**：report 先内部 + log；若需测试读取，暴露只读属性/方法即可，不做成用户配置。  
2. **数字可比较**：统一用 bytes；日志再格式化 GiB。  
3. **断言分层**：  
   - soft：记录并上传 report  
   - hard（TP=1 目标路径）：P1∧P2∧P3  
   - hard（TP=2）：功能正确性（wake+generate）；P1 按实测决定是否 hard  
4. **提交信息**：`test/docs` 与少量 `fix` 观测变更；conventional commits + sign-off。  
5. **不在本轮**解决 Jamba #10429 或其他无关 PR。

---

## 8. 附录：老师命题与现有代码路径对照

```text
P1 Suspend not Destroy
  TP=1 lightweight -> _suspend_all_comms()     # 可能成立
  TP≥2 lightweight -> destroy_hccl()           # 当前不成立（fallback）

P2 memory-release benefit
  SleepWakeupManager.sleep 已采样 mem_get_info # 需变成 report 字段并设门槛

P3 no ACL recapture
  inplace -> skip capture_model when entries exist
  missing entries + allow_fallback -> capture_model  # 必须被 capture_model_called 暴露
```

本附录用于防止实现时再次把「policy 名」当成「实际动作」。
