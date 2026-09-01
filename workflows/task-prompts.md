# Subagent 调用参数详情

本文档是 CANNBot 调用各阶段 Subagent 的**唯一执行手册**。每个 Step 包含：调用参数、关联 Skill、完成验证、约束提醒。

---

## Step 2：设计

### Subagent 调用参数

```
{
  "description": "方案设计",
  "subagent_type": "ascendc-kernel-architect",
  "prompt": "
请为以下算子设计方案：
- 算子名称：{operator_name}
- 需求描述：{用户需求}
- 环境信息：operators/{operator_name}/docs/environment.md（提供芯片型号 / SocVersion / CANN 版本；NpuArch 与 `--npu-arch` 编译参数请加载 `/npu-arch` skill 按芯片型号查得后写入 DESIGN.md）

【输出】
- 技术设计：operators/{operator_name}/docs/DESIGN.md
- 开发计划：operators/{operator_name}/docs/PLAN.md，参考`workflows/templates/plan-template.md`

【验收标准】
- DESIGN.md 和 PLAN.md 都已创建
- 设计包含：Tiling 策略、API 映射、数据流、分支场景
- 计划包含：文件清单、测试计划、PyTorch 接入（端侧开发则暂不包含PyTorch接入）
  "
}
```

---

## Step 2.5：设计串讲

### 2.5a — Design Reviewer 串讲审查
```
{
  "description": "设计串讲",
  "subagent_type": "ascendc-kernel-design-reviewer",
  "prompt": "
请审查以下算子的设计方案：
- 算子名称：{operator_name}
- 技术设计：operators/{operator_name}/docs/DESIGN.md
- 开发计划：operators/{operator_name}/docs/PLAN.md

【重点审查章节】DESIGN.md 中重点读 §1（算子设计：API 映射、数据流）和 §2.4（伪代码），审查设计逻辑和可实现性。§0（概述）、§4（实施计划）、§5（确认清单）可跳过。

【输出】
- 请将质疑清单输出到 operators/{operator_name}/docs/WALKTHROUGH.md。

【推荐 Skill】
- /ascendc-blaze-best-practice — Blaze 路线时查阅组件选择、API 签名、组装代码
- /ascendc-api-best-practices — 质疑 API 选择时查阅确认可行性
- /ascendc-docs-search — 需要官方文档支撑质疑时使用

【API 验证规则】
验证 API 是否存在时，按开发路径选择验证源：

| 开发路径 | 验证源 | 验证方法 |
|---------|--------|---------|
| Blaze 路线 | `/ascendc-blaze-best-practice` skill | 在 skill 文档和 assets/ 中验证 API 签名和模板参数 |
| 其他路线 | `$ASC_DEVKIT_DIR/docs/api/` | 禁止只读单个文件（如 `ReduceMax.md`）就下结论。必须用通配符搜索所有变体：`find "$ASC_DEVKIT_DIR/docs/api/" -name "{APIName}*.md" -type f`。同一 API 可能有多个文件（如 `ReduceMax.md` / `ReduceMax-35.md` / `ReduceMax-92.md`），功能不同。 |

【验收标准】
- 质疑清单按严重性分级（🔴 阻塞 / 🟡 需讨论 / 🟢 建议）
- 每个质疑包含：问题描述、影响分析、建议方案

【审查重点（5 项）】
| 序号 | 审查维度 | 审查方法 |
|------|---------|---------|
| 1 | API 可行性 | 设计中的 API 是否存在？参数签名是否正确？通过 `/ascendc-docs-search` skill 验证 API 文档 |
| 2 | 内存规划合理性 | UB 空间分配是否够用？Buffer 数量是否合理？双缓冲开销是否计入？ |
| 3 | 多核策略可行性 | 切分方式是否会导致数据依赖？是否需要跨核同步？边界 case 是否处理？ |
| 4 | 伪代码可实现性 | 设计中的计算流程能否直接翻译为 Ascend C 代码？是否有遗漏步骤？ |
| 5 | 精度风险 | 是否有未考虑的精度损失点？混合精度策略是否完整？除零保护是否到位？ |

【WALKTHROUGH.md 输出格式】
输出到 operators/{operator_name}/docs/WALKTHROUGH.md，使用以下结构：

## 设计串讲

### 审查结论
- [ ] 设计可直接开发（无阻塞问题）
- [ ] 设计需要修改后开发（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

### 质疑清单

#### 问题 1：[简述]
- **类别**：API 可行性 / 内存规划 / 多核策略 / 伪代码可实现性 / 精度风险
- **严重程度**：阻塞 / 需讨论 / 建议
- **设计文档位置**：DESIGN.md 第 X 节
- **问题描述**：...
- **审查者视角**：为什么从审查者角度认为这是问题
- **建议方案**：（如有）

【约束】
- 禁止：编写开发代码
- 禁止：直接修改 DESIGN.md（修改由 Architect 在回应模式中完成）
- 必须：每个问题标注严重程度
- 必须：API 可行性问题需附上实际查阅的文档依据
- 鼓励：对每个问题提出建议方案，帮助 Architect 快速回应
  "
}
```

### 2.5c — Architect 串讲回应

```
{
  "description": "串讲回应",
  "subagent_type": "ascendc-kernel-architect",
  "prompt": "
请以「串讲回应模式」回应 Design Reviewer 对设计方案的质疑：
- 算子名称：{operator_name}
- 技术设计：operators/{operator_name}/docs/DESIGN.md
- 串讲质疑：operators/{operator_name}/docs/WALKTHROUGH.md
请逐一回应质疑，并根据需要更新 DESIGN.md。

【输出】
- 更新 operators/{operator_name}/docs/WALKTHROUGH.md（添加回应）
- 如需修改，更新 operators/{operator_name}/docs/DESIGN.md

【验收标准】
- 每个质疑都有回应（采纳 / 保留原设计 + 理由）

【回应执行步骤】
1. 读取 operators/{operator_name}/docs/WALKTHROUGH.md ## 质疑清单
2. 逐一评估每个问题，判定回应类别：

| 回应类别 | 含义 | 操作 |
|---------|------|------|
| 接受 | Design Reviewer 的质疑合理 | 更新 DESIGN.md 对应章节 |
| 保留原设计 | 原设计正确，给出理由 | 不修改 DESIGN.md，给出文档依据 |
| 部分修改 | 部分采纳 | 更新 DESIGN.md 中受影响的部分 |

3. 在 WALKTHROUGH.md 中追加「### Architect 回应」子章节
4. 返回概要：接受/拒绝/部分修改的问题数量、DESIGN.md 是否有更新

【回应输出格式】
在 WALKTHROUGH.md 中追加：

### Architect 回应

#### 问题 1：[简述]
- **回应**：已修改 / 保留原设计 / 部分修改
- **理由**：...
- **文档依据**：（引用 API 文档路径或示例路径）
- **DESIGN.md 变更**：（描述修改内容，或"无变更"）

### 回应统计
- 接受 X 项，保留 Y 项，部分修改 Z 项

【回应约束】
- 必须：对每个阻塞问题给出明确回应，不可跳过
- 必须：保留原设计时附上具体的文档依据或示例引用
- 必须：接受时同步更新 DESIGN.md 对应章节
- 鼓励：对建议类问题也给出简短回应
  "
}
```

---

## Step 3：开发

### Subagent 调用参数

```
{
  "description": "算子开发",
  "subagent_type": "ascendc-kernel-developer",
  "prompt": "
请先阅读以下文件：
- operators/{operator_name}/docs/DESIGN.md — 技术设计（重点读 §1.2 API 映射、§1.5 Buffer 规划、§2.4 伪代码）
- operators/{operator_name}/docs/PLAN.md — 开发计划（请在开发中持续更新）
然后开始开发。

【第一步：基于模板搭建工程骨架】
加载 /ascendc-direct-invoke-template，基于验证过的工程模板创建项目文件（CMakeLists.txt、.asc 文件、头文件）。
禁止从零创建工程文件。搭建骨架后先编译通过（空 Kernel），再逐步添加算子逻辑。

【渐进式开发策略】
Step A: 基于模板创建工程骨架 → 编译通过（空 Kernel）
Step B: 添加 Tiling 结构体和 Host 侧 Tiling 计算 → 编译通过
Step C: 添加 Kernel 核心计算逻辑 → 编译通过
Step D: 添加测试用例和精度验证 → 运行通过
每步必须编译通过后再进入下一步。

注意：DESIGN.md 中的代码模板是伪代码，展示计算逻辑和 API 选择，不能直接编译。
实际代码结构以所选模板的工程模板为准。

【参考文档】
- 编码规范与审查清单：workflows/development-guide.md
- 算子族特定方法论：*必须在ascendc-tiling-design中查找对应算子类的开发指南

【输出】
- 算子代码：operators/{operator_name}/
- 更新进度：operators/{operator_name}/docs/PLAN.md

【验收标准】
- 编译成功（cmake .. && make）
- 基础用例 可执行文件、PyTorch接入通路 测试通过（端侧开发则暂不包含PyTorch接入）
- PLAN.md 已更新进度
  "
}
```

---

## Step 4：审查

### Subagent 调用参数

```
{
  "description": "代码审查",
  "subagent_type": "ascendc-kernel-reviewer",
  "prompt": "
请审查以下算子代码：
- 算子名称：{operator_name}
- 审查轮次：Round 0（Step 4 初审）
- 代码路径：operators/{operator_name}/
- 设计文档：operators/{operator_name}/docs/DESIGN.md
- 环境信息：operators/{operator_name}/docs/environment.md（芯片型号 / SocVersion）；NpuArch 与 `--npu-arch` 合法值通过 `/npu-arch` skill 查得，用于核对 CMakeLists 与 DESIGN.md 一致
- 审查清单：workflows/development-guide.md
- 算子族方法论：见 /ascendc-tiling-design 路由表

【输出】
- 审查报告：operators/{operator_name}/docs/REVIEW.md（创建新文件，报告头部标注 `## Round 0 审查报告（Step 4 初审）`）

【推荐 Skill】
- /ascendc-docs-search — 验证 API 约束、查找官方示例对照
- /ops-profiling — 独立采集性能数据验证
- /ops-precision-standard — 精度验证阶段确认 atol/rtol 标准

【验收标准】
- 独立编译验证
- 100 分制评分
- PASS/FAIL/PASS WITH NOTES 判定
- 具体修复要求（如 FAIL）
  "
}
```

---

## Step 5：修复循环

> ⚠️ **CANNBot 禁止自行修改代码，即使修复看起来只有一行。必须调用 Developer Subagent。**

### Step 5a — Developer 修复

```
{
  "description": "代码修复",
  "subagent_type": "ascendc-kernel-developer",
  "prompt": "
请根据审查报告修复代码：
- 算子名称：{operator_name}
- 审查报告：operators/{operator_name}/docs/REVIEW.md（请阅读最新一轮审查报告）
- 设计文档：operators/{operator_name}/docs/DESIGN.md

【输出】
- 修复后的代码：operators/{operator_name}/
- 更新进度：operators/{operator_name}/docs/PLAN.md

【推荐 Skill】
- /ascendc-precision-debug — 遇到以下症状时使用：输出全为0、输出随机错误值、核心超时/挂起、Cast后数据错误、非对齐数据失败、精度不达标
- /ascendc-api-best-practices — review 指出 API 参数/约束错误时使用
- /ascendc-docs-search — 查找替代方案、确认 API 用法

【验收标准】
- 审查报告中的所有修复项已处理
- 编译成功
- 测试通过
  "
}
```

### Step 5b — Reviewer 复审

```
{
  "description": "代码复审",
  "subagent_type": "ascendc-kernel-reviewer",
  "prompt": "
请复审以下算子代码（Developer 已根据上轮审查报告修复）：
- 算子名称：{operator_name}
- 审查轮次：Round {review_round}（Step 5 复审）
- 代码路径：operators/{operator_name}/
- 设计文档：operators/{operator_name}/docs/DESIGN.md
- 环境信息：operators/{operator_name}/docs/environment.md（芯片型号 / SocVersion）；NpuArch 与 `--npu-arch` 合法值通过 `/npu-arch` skill 查得，用于核对 CMakeLists 与 DESIGN.md 一致
- 审查清单：workflows/development-guide.md
- 历史审查报告：operators/{operator_name}/docs/REVIEW.md（已有 Round 0 ~ Round {prev_round} 的报告，请勿覆盖）

【输出】
- 审查报告：operators/{operator_name}/docs/REVIEW.md（在文件末尾追加，报告头部标注 `## Round {review_round} 审查报告（Step 5 复审）`）

【推荐 Skill】
- /ascendc-docs-search — 验证 API 约束、查找官方示例对照
- /ops-profiling — 独立采集性能数据验证
- /ops-precision-standard — 精度验证阶段确认 atol/rtol 标准

【验收标准】
- 独立编译验证
- 100 分制评分
- PASS/FAIL/PASS WITH NOTES 判定
- 具体修复要求（如 FAIL）
- 报告已追加到 REVIEW.md 末尾，未覆盖历史报告
  "
}
```

---

## Step 6：精度与性能验收

### Step 6a — Reviewer 精度验收

```
{
  "description": "精度验收",
  "subagent_type": "ascendc-kernel-reviewer",
  "prompt": "
请执行精度验收：
- 算子名称：{operator_name}
- 算子目录：operators/{operator_name}/
- 设计文档：operators/{operator_name}/docs/DESIGN.md

【输出】
- 精度报告：operators/{operator_name}/docs/precision/summary.txt

【推荐 Skill】
- /ops-precision-standard — 确定各 dtype 的 atol/rtol 精度标准
- /ascendc-precision-debug — 精度不达标时诊断根因

【验收标准】
- 精度测试覆盖所有声明的 dtype
- 精度报告已归档，每个 (dtype, shape) 组合的达标判定已记录
- 如有精度不达标，已记录问题类型和诊断结论
  "
}
```

### Step 6b — Developer 性能采集

```
{
  "description": "性能采集",
  "subagent_type": "ascendc-kernel-developer",
  "prompt": "
请执行性能采集：
- 算子名称：{operator_name}
- 算子目录：operators/{operator_name}/

【输出】
- 性能数据：operators/{operator_name}/docs/perf/round_NNN/
- 性能摘要：operators/{operator_name}/docs/perf/round_NNN/summary.txt

【推荐 Skill】
- /ops-profiling — msprof op 采集、CSV 指标解读、达标判定

【验收标准】
- 性能数据已归档
- 达标判定已记录
- 如未达标，已记录瓶颈分析
  "
}
```
---

## 报告格式通用规范

所有验收报告必须包含以下字段，供 CANNBot 解析判断：

```markdown
**状态**: ✅通过 / ❌失败

**验证摘要**:
| 验证项 | 结果 | 详情 |
|-------|------|------|
| ... | 通过/失败 | ... |

**关键指标**:
- 总用例数: X
- 通过数: Y
- 失败数: Z
- 通过率: X%

**精度概要**
| dtype | rtol | atol | 达标状态 |
|-------|------|------|---------|
| ... | ... | ... | ✅/❌ |

**性能概要**
- Task Duration
- 主导流水
- 达标状态

**失败用例**（如有）:
- 列出失败的测试用例及原因
```

**重要约束**：
- 如有失败用例，状态必须标记为 `❌失败`，禁止标记为 `✅通过`
- 仅编译通过不等于验证通过，必须实际运行测试
