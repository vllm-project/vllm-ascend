# state.json 字段说明与填写规则

> 本文档是 `state.json`（工作流状态文件）的填写说明，与 `state.json`（空模板）配套使用。
> `state.json` 为**严格合法 JSON**，禁止写入 `//` 注释；模板中的 `<...>` 占位符与 `null` / `pending` 为初始值，由工作流推进时逐步替换。

## 字段语义

| 字段 | 结构 | 必填 | 说明 | 数据来源 |
|------|------|------|------|---------|
| `workflow.name` | string | ✅ | 本次运行使用的工作流名称 | 工作流标识 |
| `workflow.cannbot-skills commit` | string | ✅ | cannbot-skills 仓库 commit（`git rev-parse HEAD`） | 仓库版本锁定 |
| `workflow.framework` | string | ✅ | 运行工具 + 版本，`{工具}@{版本}`，如 `opencode@1.4.7` | 运行时获取（见下） |
| `operator.name` | string | ✅ | 算子名 | 用户需求 |
| `operator.arch` | string | ✅ | 架构，如 `SIMT` / `SIMD` | DESIGN.md |
| `operator.dtypes` | string[] | ✅ | 支持的数据类型，如 `["FP32","FP16","BF16"]` | DESIGN.md |
| `env_summary.device` | string | ✅ | 芯片型号，如 `Ascend950PR` | environment.md |
| `env_summary.cann_version` | string | ✅ | CANN 版本 | environment.md |
| `env_summary.compiler` | string | ✅ | 编译器名，如 `bisheng` | environment.md「编译器与库」 |
| `env_summary.compiler_version` | string | ✅ | 编译器版本，如 `clang 15.0.5` | `bisheng --version` |
| `results.build` | bool/null | ✅ | 编译是否通过；`null`=未评估 | Step 3 结果 |
| `results.precision` | bool/null | ✅ | 精度是否全部达标；`null`=未评估 | `docs/precision/summary.txt` |
| `results.performance` | bool/null | ✅ | 性能是否达标；`null`=未评估 | `docs/perf/*/summary.txt` |
| `stages` | object | ✅ | 各阶段状态，键定义见下 | 工作流各 Step |
| `usage` | object | 可选 | 对话轮数 / 调用次数 / token 统计，**可采集时填写，否则删除整个 `usage` 对象** | 见下方采集方法 |

## stages 键定义

适配当前工作流 Step 1–7（含 CP 门禁）。

| 键 | 对应工作流环节 | 说明 |
|----|--------------|------|
| `1` | Step 1 环境检查 | environment.md 生成 |
| `CP1` | 环境门禁 | 状态行含 `✅ 通过` |
| `2` | Step 2 设计 | DESIGN.md + PLAN.md |
| `CP2` | 设计门禁 | 双文件都存在 |
| `2.5` | Step 2.5 设计串讲 | WALKTHROUGH.md 质疑闭环 |
| `CP2.5` | 串讲门禁 | 无阻塞分歧 |
| `3` | Step 3 开发 | 编译成功 + 基础用例 |
| `CP3` | 开发门禁 | 编译通过（可附 `build`） |
| `4` | Step 4 审查 | REVIEW.md（可附 `verdict`/`score`） |
| `CP4` | 审查门禁 | PASS / PASS WITH NOTES |
| `5` | Step 5 修复循环 | 未触发则 `skipped`（可附 `rounds`） |
| `CP5` | 修复门禁 | 复审通过；未触发则 `skipped` |
| `6` | Step 6 精度与性能验收 | 汇总 6a/6b |
| `6a` | 精度验收 | 可附 `tests`（如 `89/89`） |
| `6b` | 性能采集 | 可附 `vec_ratio` 等指标 |
| `CP6` | 验收门禁 | 精度达标 + 性能归档 |
| `7` | Step 7 完成汇报 | 汇报产出 |

**状态取值规则**：

| 取值 | 含义 |
|------|------|
| `pending` | 未开始（模板初始值） |
| `running` | 进行中 |
| `completed` | 正常完成 |
| `skipped` | 未进入/未触发（如修复循环一轮未用） |
| `failed` | 失败中止 |

多轮修复循环仅记录最终轮次。

## 实时更新模型

- **谁写**：仅 CANNBot 维护（读各阶段交付文档写回），Subagent 不写此文件。
- **何时写**：每步/每 CP 完成后**立即**落盘，禁止攒到 Step 7 一次性补写。断点恢复依赖其实时性。
- **文件落位**：`operators/{operator_name}/state.json`
- **初始化时机**：Step 1 前复制 `workflows/references/state.json` 到算子目录，填 `workflow`（`framework` 版本运行时可测，`cannbot-skills commit`=`git rev-parse HEAD`）与 `operator` 已知字段，`1` 置 `running`。
- **各阶段更新点**：
  - Step 1 完成 → `1`/`CP1` + `env_summary` 全字段
  - Step 2 完成 → `2`/`CP2` + `operator` 补全
  - Step 2.5 完成 → `2.5`/`CP2.5`
  - Step 3 完成 → `3`/`CP3` + `results.build`
  - Step 4 完成 → `4`/`CP4`（附 `verdict`/`score`）
  - Step 5 完成 → `5`/`CP5`（未触发置 `skipped`）
  - Step 6 完成 → `6`/`6a`/`6b`/`CP6` + `results.precision`/`results.performance`
  - Step 7 完成 → `7` + `usage`（可采集时）
- **校验**：任意时刻可运行 `python workflows/scripts/validate_state.py operators/{operator_name}/state.json`。

## usage 采集方法（可选字段）

> 采集难度：opencode 下低成本，其他工具（claude/trae/cursor/copilot/codearts）无统一命令，**仅在可采集时填写，否则删除整个 `usage` 对象**。

| 字段 | 含义 | opencode 采集命令 |
|------|------|------------------|
| `conversation_rounds` | 对话轮数（模型调用次数） | `opencode export <sessionID>`，数 `step-start` part 数量 |
| `tool_calls` | 工具调用次数 | 同上，数 `tool` part 数量 |
| `tokens.input` / `tokens.output` | 模型输入/输出 token | `opencode db "SELECT ... FROM message WHERE session_id=..."` 聚合 `tokens.input` / `tokens.output` |
| `tokens.cache_read` | 缓存读取 token | 同上，聚合 `tokens.cache.read` |
| `tokens.total` | 总 token | `total = input + output + cache_read`（与 opencode 记账口径一致） |

**口径说明**：session 查询务必按 `session_id` 精确过滤，避免 `opencode stats` 按项目聚合导致多算子数据串扰。
