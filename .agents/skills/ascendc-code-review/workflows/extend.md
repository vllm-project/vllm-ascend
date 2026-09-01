# 代码检视 Skill 规范

## 触发
扩展能力、新增规则、新增场景、新增检视条例、怎么加规则、扩展检视、接入新规范、添加检视条例、扩展代码检视、怎么扩展、如何添加、能力扩充、代码检视规范、reviewer skill 设计

---

## 一、架构总览

### 1.1 目录结构

```
ascendc-code-review/
├── SKILL.md                         入口路由，触发词 → 匹配工作流文件
│
├── core/
│   └── methodology.md               检视方法论（假设检验、证据框架、置信度标准）
│
├── workflows/                       工作流，按场景命名：{scene}.md
│   ├── file-review.md              「检视代码」→ 文件检视（含可选设计一致性检查）
│   ├── pr-review.md                「检视 PR」→ PR检视（含可选设计一致性检查）
│   ├── pr-large-review.md          review_mode.py 判 mode=large → 大型PR检视（按文件组并行，阈值见 scripts/workflow.review-thresholds.yaml）
│   ├── quick-review.md             「快速检视」→ 定向问题排查
│   ├── extend.md                   本文件：系统规范
│   └── ...                         新增场景按同模式添加（scope 必须与 workflow 文件名一致）
│   └── quick-review 无需 steps/ — 主 Agent 直接执行，无子 Agent，不引用 step 文件
│
├── steps/                           可执行积木，命名规则：{scope}.{step}.md
│   │                                scope = 对应 workflow 文件名去掉 .md 后缀
│   │                                scope = "common" 表示跨场景公共步骤
│   ├── common.{step}.md            跨场景公共（plan-design、line-verify、report-write、
│   │                                docs-detect、design-check）
│   ├── file-review.{step}.md       文件检视（code-summarize、clause-review）
│   ├── pr-review.{step}.md         PR检视（code-fetch、code-summarize、clause-review、line-verify）
│   ├── pr-large-review.{step}.md   大型PR（file-split、global-pre-scan、code-summarize、
│   │                                clause-review、clause-grouping、synthesize、merge）
│   └── ...
│
├── references/                      规则文档，按领域命名：{domain}.md
│   ├── cpp-secure.md                C++安全编码
│   ├── cpp-general.md               C++通用编码
│   ├── ascendc-api.md               AscendC API最佳实践
│   ├── ascendc-perf.md              AscendC高性能编程
│   ├── ascendc-topk.md              TOPK高频问题
│   ├── compile-secure.md            安全编译
│   ├── ...                          另有 cpp-style、python-secure、simt-api-analysis、mc2-specific 等
│
└── scripts/                         工具脚本：{tool}.py
    ├── clause.check_bounds.py              数值边界静态分析
    ├── get_gitcode_pr_diff.py        PR diff获取
    ├── clone_pr_source.py            PR源码克隆
    └── ...
```

### 1.2 三层职责

| 层 | 目录 | 性质 | 何时被读 |
|-----|------|------|---------|
| 编排层 | workflows/ | 蓝图：定义阶段顺序、子Agent派发规则 | 主Agent启动时 |
| 执行层 | steps/ | 积木：派发指令/执行指南/prompt模板 | 各阶段执行时，逐文件Read |
| 知识层 | references/ + core/ | 只读：规则条款和检视方法论 | 子Agent需要时延迟加载 |

### 1.3 渐进式披露（核心设计原则）

**每个步骤独立文件，Agent逐文件Read，按文件名前缀隔离上下文。** 整个检视流程中，任何Agent在任何时刻只加载当前阶段需要的文件，不提前、不超载。

```
入口: SKILL.md (34行)
  → 匹配到 workflow 文件
  → Read workflow → 看到阶段蓝图，不展开步骤细节
  → 逐阶段 Read 对应 steps/ 文件
  → 子Agent被dispatch后 Read core/methodology.md + references/*
```

**渐进式披露的反模式和检查清单**：

| 反模式 | 表现 | 后果 |
|--------|------|------|
| workflow 文件过重 | workflow 中内嵌了步骤的详细执行指令 | 主Agent在启动阶段就被迫加载几百行执行细节 |
| step 文件跨层泄露 | 派发指令和执行指南混在同一文件，主Agent读派发指令时被动吞下子Agent的全部执行内容 | 编排Agent的context被无关内容污染 |
| 规则文件前置加载 | 在路由完成前就引导Agent阅读完整规则文档 | 1000+行无关规则文本进入context |
| 方法论过早注入 | 在子Agent被dispatch前就将methodology.md内容写进prompt | 编排Agent和路由Agent的context被方法论内容占用 |
| 报告格式过早定义 | 在阶段0就写入最终报告格式要求 | 前面所有Agent都看到不需要的报告格式 |

**设计检查清单**（新增或修改workflow时逐项检查）：

- [ ] workflow 文件是否 ≤100行？超过则需要将细节下沉到 step 文件
- [ ] workflow 中是否只写「Read + 执行 steps/xxx.md」而不展开步骤内部逻辑？
- [ ] 每个 step 文件是否只有一种读者（编排Agent或子Agent，不是两者）？
- [ ] 派发指令型 step 中，主Agent读取的派发部分是否 ≤40行？
- [ ] 子Agent是否在被dispatch后才通过 `Read` 延迟加载方法论和规则文档？
- [ ] prompt模板是否只引用方法论路径，不内联方法论内容？
- [ ] 不同场景的 step 文件是否通过文件名前缀隔离（`file-review.*` vs `pr-review.*`）？
- [ ] 公共步骤（`common.*`）是否真正跨场景无差异？
- [ ] 新增步骤是否考虑了复用已有步骤？跨工作流通用时是否命名为 `common.*`？
- [ ] 每条执行要求是否只在一个文件中定义（不在多个step/方法论/规则中重复）？

**上下文预算参考**（单次Read的目标行数）：

| 层 | 文件 | 目标行数 | 读者 |
|-----|------|---------|------|
| 路由 | SKILL.md | ≤40 | 主Agent |
| 蓝图 | workflows/*.md | ≤100 | 主Agent |
| 派发 | steps/{x}.code-summarize.md（派发部分） | ≤40 | 主Agent |
| 模板 | steps/{x}.clause-review.md | ≤40 | 主Agent |
| 执行 | steps/{x}.code-summarize.md（执行指南部分） | 不限 | 子Agent |
| 执行 | common.line-verify.md | ≤20 | 主Agent |
| 执行 | common.report-write.md | ≤90 | 主Agent |
| 派发 | common.docs-detect.md（派发部分） | ≤30 | 主Agent |
| 派发 | common.design-check.md（派发部分） | ≤30 | 主Agent |
| 知识 | core/methodology.md | ≤130 | 通用检视子 agent |
| 知识 | references/*.md | 不限 | 通用检视子 agent（按需） |

### 1.4 工作流示例：PR检视

以下是 `workflows/pr-review.md` 的实例。**所有工作流——无论系统内置还是用户新增——均遵循此结构**。新增工作流（如安全审计、性能专项）只需按同模式定义阶段 → 引用 step → 注册路由，无需改动系统代码。

核心信息：每个阶段谁读什么文件、读多少行、谁是读者——这是渐进式披露在单个工作流内的具体体现。

```
SKILL.md (34行)
  │  匹配"检视 PR" → workflows/pr-review.md
  ▼
pr-review.md (88行)          ← 主Agent只看到阶段蓝图，不接触步骤细节
  │
  ├─ 阶段0 ──────────────────────────────────────
  │   │  获取 diff + 代码概要 + API 预研 + 设计文档探测 + 检视计划设计
  │   │
  │   ├─ steps/pr-review.code-fetch.md (20行)       主Agent执行
  │   │
  │   ├─ steps/pr-review.code-summarize.md           代码概要子 Agent (并行)
  │   │   ├─ 派发指令 (~30行) → 主Agent读到
  │   │   └─ 执行指南 (~230行) → 仅子Agent读到
  │   │
  │   ├─ steps/common.api-prestudy.md                 API 预研子 Agent (并行, 仅 Kernel 侧)
  │   ├─ steps/common.docs-detect.md                  设计文档探测子 Agent (并行)
  │   │
  │   └─ steps/common.plan-design.md (~90行)         检视计划子 Agent (三个子 Agent 返回后派发)
  │       读概要 → 匹配/筛查/合并 → 检视计划
  │
  ├─ 阶段1 ──────────────────────────────────────
  │   │  逐条检视
  │   │
  │   ├─ steps/pr-review.clause-review.md (37行)     prompt模板
  │   │      主Agent填充: 条例ID + diff路径 + 代码范围
  │   │
  │   └─ 逐波派发通用检视子 agent (≤6/波，见 core/review-load-balance.md)
  │       子Agent加载链: skill → methodology.md → 概要 → 规则文档
  │                                                   ◀── 扩展点：放新规则
  │
  ├─ 阶段2 ──────────────────────────────────────
  │   └─ steps/pr-review.line-verify.md (22行)       主Agent执行
  │       diff行号→实际行号 + 越界移除
  │
  └─ 阶段3 ──────────────────────────────────────
      └─ steps/common.report-write.md (65行)          主Agent执行
          证据表 + 报告文件
```

**谁读什么，读多少**：

| 读者 | 文件 | 行数 |
|------|------|------|
| 主Agent | SKILL.md → pr-review.md | 34 + 88 |
| 主Agent | code-fetch / 派发指令 / prompt模板 / line-verify / report-write | 20 + 30 + 37 + 22 + 65 |
| 计划设计子Agent | common.plan-design.md | 200 |
| 概要子Agent | 执行指南 | 230 |
| 通用检视子 agent | methodology.md + 分配的规则文档 | 126 + 按需 |

主Agent全程加载 ≤300行编排内容，子Agent各自独立加载执行细节，不互相污染。

**新增工作流只需复制此模式**：定义触发词 → 编排阶段顺序 → 每个阶段引用 steps/ 文件 → 在 SKILL.md 注册路由。渐进式披露的约束对所有工作流一致：工作流文件 ≤100行，不内联步骤细节，子Agent知识延迟加载。

## 二、工作流规范

### 2.1 工作流文件结构

每个 `workflows/{id}.md` 是一个完整的工作流定义，遵循以下结构：

```markdown
# {工作流名称}

## 触发
{触发关键词1}, {触发关键词2}, ...

---

## 编排

### 任务清单
| 任务 | 阶段 | 执行文件 / 内容 |
|------|------|----------------|
| 任务0 | {名称} | steps/{scope}.{step}.md |
| 任务1 | {名称} | steps/{scope}.{step}.md |
...

### 阶段0：{名称}
1. 将任务0标记为 in_progress
2. ...
3. Read + 执行 `steps/{scope}.{step}.md`
4. 传入参数...
5. 将任务0标记为 done

### 阶段N：{名称}
...

---

## 上下文传递链
{描述阶段间的数据流转}

## 约束
{工作流特有的约束规则}
```

### 2.2 任务追踪

每个工作流启动时创建 4-5 个固定任务（全部 pending）。每个阶段开始时标记 `in_progress`，完成后标记 `done`。不动态重写任务列表。

### 2.3 阶段顺序

严格按阶段顺序执行，禁止跳步。每个阶段开始时 Read 对应 step 文件，执行完成后再 Read 下一个。禁止提前 Read 未执行阶段的 step 文件。

### 2.4 子Agent派发

工作流负责子Agent的派发编排。step 文件提供 prompt 模板，不包含 `Agent()` 调用逻辑。子Agent类型统一使用 `"general"`。

每波 ≤6 个子Agent（上限见 `core/review-load-balance.md`），波次内并行，波次间串行。

### 2.5 注册新工作流

在 `SKILL.md` 的「工作流路由」表中加一行：

```markdown
| {触发关键词} | workflows/{id}.md |
```

触发关键词用逗号分隔，中英文均可。选择的关键词应具体、独特，避免与已有工作流冲突。

**新增工作流必须通过 1.3 节的渐进式披露检查清单**，确保：
- 文件 ≤100行，内容仅限于阶段蓝图和上下文传递，不内联步骤执行细节
- 每个阶段通过 `Read + 执行 steps/xxx.md` 引用步骤文件，不展开步骤内部逻辑
- 不提前声明报告格式、不内联方法论内容、不重复步骤文件已有的约束

---

## 三、步骤规范

### 3.1 命名约定

```
steps/{scope}.{step-name}.md
```

| scope | 含义 | 示例 |
|-------|------|------|
| common | 跨工作流复用 | common.plan-design.md |
| file-review | 文件检视专用 | file-review.code-summarize.md |
| pr-review | PR检视专用 | pr-review.code-fetch.md |

#### 鼓励复用 common.* 步骤

`common.*` 前缀的步骤可跨工作流复用。新增工作流时，已有的公共步骤通常能满足大部分需求：

| 公共步骤 | 作用 |
|---------|------|
| common.plan-design.md | 检视计划设计（条例匹配/合并/分组/筛查） |
| common.line-verify.md | 行号校对 |
| common.report-write.md | 报告生成 |
| common.docs-detect.md | 设计文档探测（与 code-summarize/api-prestudy 同级并行） |
| common.design-check.md | 设计实现一致性 S1-S7 检查（含设计映射，复用摘要+API预研） |

如果发现某步骤在多个工作流中重复出现，鼓励提取为 `common.*`，让后续的新工作流受益。场景特有的逻辑（如 PR 检视的越界校验）才用 `{工作流}.*` 前缀。

### 3.2 步骤类型

| 类型 | 特征 | 示例 |
|------|------|------|
| 派发指令 | 文件顶部有 `Agent({...})` 调用块和 prompt。主Agent读取后派发子Agent执行，执行指南部分仅子Agent读取 | file-review.code-summarize.md |
| 执行指令 | 主Agent直接按步骤执行，不派发子Agent | common.line-verify.md |
| prompt模板 | 含 `{占位符}` 的模板，由工作流填充后传给子Agent。不包含 `Agent()` 调用逻辑 | file-review.clause-review.md |

### 3.3 派发指令型步骤结构

```markdown
# {步骤名称}

## 派发
Agent({
  subagent_type: "{首选}" 或 "general",
  description: "{任务描述}",
  prompt: "{自包含的执行prompt}"
})

---

## 子Agent执行指南
{仅子Agent读取的详细执行步骤、关键知识表、输出模板}
```

### 3.4 prompt模板型步骤结构

```markdown
# {步骤名称}

## prompt 模板
【已由上游完成】
- {上游传入的上下文参数}

检视文件：{code_file_path}
检视条款：{条例ID-1} {标题}, ...

【执行要求】
- 第一步加载skill，Read core/methodology.md
- 对每条条例Read对应规则文档完整内容
- API类条例先使用/ascendc-docs-search查阅官方文档
- 严格按假设检验流程执行
- 禁止生成报告文件

## 输出格式
[条例ID] PASS/FAIL/SUSPICIOUS 置信度:HIGH/MED/LOW
FAIL/SUSPICIOUS必须附：问题描述 + 代码片段（≥10行） + 修复建议
```

---

## 四、规则文档规范（references/）

### 4.1 `<适用>` 声明头

每个 `references/{id}.md` 文件开头必须有 `<适用>` 声明，定义规则的适用条件。计划设计子Agent在检视时自动扫描 `references/` 目录，读取每个文件的 `<适用>` 头进行匹配。

```markdown
<适用>
语言: {C++ / Python / Build / 不限}
侧别: {All / Kernel / Tiling / Host / N/A}
领域: {true / false}
触发: {领域规则的关键词列表，非领域规则填 —}
默认启用: {true / false}
排除场景: {可选，领域规则的排除条件}
</适用>
```

### 4.1.1 `<检视负载>` 声明头

每个 `references/{id}.md` 文件 `<适用>` 头之后必须有 `<检视负载>` 声明，定义通用检视子 agent 检视该文件条款时的派发参数。路由子 agent 扫描该头读取容量，按其打包每组条款（规则见 `core/review-load-balance.md`）。

```markdown
<检视负载>
通用检视子 agent 检视条款容量上限: {数字}
</检视负载>
```

- 数字（如 3/5/10）= 一个通用检视子 agent 一次检视该文件条款的上限；越小表示检视负载越重（少打包）
- 专项文件（如 cpp-style、doc-style）写「此条款文档由专项检视子 agent 独立托管，不参与条款派发流程」，不进通用检视波次

### 4.2 字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| 语言 | 是 | `C++` `Python` `Build` `不限`。与代码文件扩展名匹配 |
| 侧别 | 是 | 逗号分隔。`All`（全部适用）`Kernel` `Tiling` `Host` `N/A`（Python等不涉及侧别的语言） |
| 领域 | 是 | `true`=领域规则（需代码命中触发特征才启用），`false`=通用规则（始终纳入） |
| 触发 | 领域=true时必需 | 逗号分隔的关键词列表。代码中出现任一关键词即激活该规则文件。关键词应具体有区分度。保留值 `必须触发` 表示无条件激活（不依赖代码命中关键词），用于红线/高频规范（如 cpp-secure、ascendc-topk） |
| 默认启用 | 是 | `true`=默认纳入检视，`false`=需用户显式要求（如cpp-style） |
| 排除场景 | 否 | 领域规则的排除条件。如MC2的"纯通信无计算融合"排除逻辑 |

#### `<适用>` 头是规则自发现的唯一接口

计划设计子Agent在每次检视时自动扫描 `references/` 目录，读取每个 `.md` 文件的 `<适用>` 头，与代码特征匹配。**整个过程不需要修改任何路由代码或配置文件。**

```
新增规则文件 → 放入 references/ → 写 <适用> 头
    ↓
计划设计子Agent启动
    ↓
Step 1: 扫描 references/*.md → 收集所有领域规则的 触发: 关键词
Step 2: 逐文件匹配：语言 → 侧别 → 默认启用 → 领域触发
    ↓
匹配成功 → 纳入检视分组
```

- **通用规则**（`领域: false`）：只要语言和侧别匹配就纳入，不需要触发特征
- **领域规则**（`领域: true`）：代码中出现 `触发:` 字段中的任一关键词才激活
- **`触发:` 字段是领域规则的关键**：关键词来自代码中实际出现的 API 名、宏、include 路径。不同仓的规则只要声明不同的触发关键词（如 `ops-nn::` vs `ops-math::`），路由就会自动分流，互不干扰
- **排除场景**：领域规则可以在 `<适用>` 头中声明排除条件，计划设计子Agent会读取并应用

### 4.3 快速索引表

每个规则文件必须包含 `## 快速索引` 章节，用于路由阶段的条例过滤和同类合并：

```markdown
## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| SEC-2.1 | 有符号整数运算不溢出 | 数值安全 | 高 | [适用: All] |
| SEC-2.3 | 除法/余数运算除零保护 | 数值安全 | 高 | [适用: All] |
| SEC-3.1 | 禁止使用未初始化的变量 | 内存安全 | 高 | [适用: All] |
| SEC-5.1 | 资源申请后判断是否成功 | 资源管理 | 高 | [适用: Tiling] |
```

**范畴列**：用于跨文档同类合并路由。常见范畴：`数值安全` `内存安全` `输入验证` `API使用` `并发安全` `资源管理` `类型安全` `数据搬运` `性能优化` `精度保护` `Tiling设计` `通信同步` `编译配置` `代码风格`

**适用范围列**：用于条例级侧别过滤。格式：`[适用: All]` `[适用: Kernel]` `[适用: Tiling]` `[适用: Host]`。路由优先级：条例级标记 > 文件级全局侧别。

### 4.4 条例详情结构

```markdown
## {条例ID}: {标题}

**严重级别**：{高/中/低}

### 问题描述
{说明问题的本质和适用场景}

### 错误示例
```cpp
// ❌ 错误代码
```

### 正确示例
```cpp
// ✅ 正确代码
```

### 注意事项
{关键注意点}

### 专属检视方法（可选）
{本条条例特有的检视步骤，如类型分析、多步验证等。通用检视子 agent必须严格遵循}
```

### 4.5 注册方式

文件放入 `references/` 目录即自动注册。计划设计子Agent在每次检视时扫描该目录。无需修改任何其他文件。

### 4.6 文件命名与条例编号前缀

#### 文件命名约定

所有规则文件遵循 `{领域}_{规则类型}.md` 的命名模式。领域表示代码所属的技术范畴，规则类型表示检视维度。

**领域前缀（已占用和预留）**：

| 领域前缀 | 领域 | 适用仓库 | 示例文件 |
|---------|------|---------|---------|
| `cpp` | 通用C++ | 所有仓 | cpp_secure.md, cpp_general.md, cpp_style.md |
| `ascendc` | AscendC算子 | ops-transformer, ops-math, ops-nn, ops-cv | ascendc_api.md, ascendc_perf.md, ascendc_topk.md |
| `mc2` | MC²通算融合 | ops-transformer | mc2_specific.md |
| `simt` | SIMT编程 | ops-transformer | simt_api.md |
| `python` | Python脚本 | 所有仓 | python_secure.md |
| `compile` | 编译配置 | 所有仓 | compile_secure.md |
| `ops-math` | 数学算子仓专用 | ops-math | *预留* |
| `ops-nn` | 神经网络仓专用 | ops-nn | *预留* |
| `ops-cv` | 计算机视觉仓专用 | ops-cv | *预留* |
| `custom` | 用户自定义 | — | *预留* |

**规则类型后缀**：

| 后缀 | 含义 | 示例 |
|------|------|------|
| `secure` | 安全编码规范 | cpp_secure, ascendc_api(含安全) |
| `api` | API使用规范 | ascendc_api, simt_api |
| `perf` | 性能优化规范 | ascendc_perf |
| `general` | 通用编码规范 | cpp_general |
| `style` | 代码风格规范 | cpp_style |
| `topk` | 高频问题清单 | ascendc_topk |
| `specific` | 领域专项规则 | mc2_specific |

#### 条例编号前缀

新增规则时，条例编号前缀取领域前缀的大写形式，必须唯一：

| 前缀 | 来源文件 | 命名空间 |
|------|---------|---------|
| `SEC` | cpp_secure.md | C++安全 |
| `GEN` | cpp_general.md | C++通用 |
| `API` | ascendc_api.md | AscendC API |
| `PERF` / `PREC` / `TIL` | ascendc_perf.md | AscendC性能/精度/Tiling |
| `TOPK` | ascendc_topk.md | TOPK高频问题 |
| `SIMT` | simt_api.md | SIMT API转换 |
| `MC2` | mc2_specific.md | MC²领域 |
| `CMP` | compile_secure.md | 编译安全 |
| `PY` | python_secure.md | Python安全 |
| — | cpp_style.md | 代码风格（默认不启用，数字编号） |

新增领域时同步声明前缀。如 `ops-math` 领域用 `MATH` 前缀，新建 `ops-math_specific.md` 文件。

## 五、扩展指南

### 5.1 新增规则文档

**创建文件**：`references/{id}.md`

**必须包含**：
1. `<适用>` 声明头（见第四章）
2. `<检视负载>` 声明头（见 4.1.1）
3. `## 快速索引` 表（含范畴和适用范围列）
4. 每条条例的完整详情（问题描述/错误示例/正确示例）

**注册**：文件放入 `references/` 即自动生效，无需注册。

### 5.2 新增工作流

**创建文件**：`workflows/{id}.md`（遵循第二章的结构）

**注册**：在 `SKILL.md` 加一行路由。

**质量门禁**：提交前必须通过 1.3 节的渐进式披露检查清单（10项）。重点关注：文件 ≤100行，不内联步骤细节，不提前声明报告格式，不重复已有约束。

### 5.3 新增步骤

**创建文件**：`steps/{scope}.{name}.md`（遵循第三章的命名和结构）

**使用**：在工作流文件中引用步骤文件名。

### 5.4 新增脚本

**创建文件**：`scripts/{name}.py`（或 `.sh`）

**使用**：在step文件中通过 `{skill_base}/scripts/{name}.py` 引用。脚本通过CLI参数接收输入，stdout输出结果，exit code表示状态。
