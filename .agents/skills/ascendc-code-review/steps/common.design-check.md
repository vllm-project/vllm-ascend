# 设计实现一致性检查（S1-S7）

文件检视、PR 检视共用。

**1 个子 Agent 顺序做完 S1-S7 七策略**（不拆分：S1 架构是地基、S4 数据流依赖 S1、S6 伪代码依赖 S4，拆开会矛盾或重复通读设计文档）。

本检查承担**设计映射**职责：因 docs-detect 与 code-summarize 在 Stage0 并行，code-summarize 不接触设计文档，故设计文档读取与设计映射由本子 Agent 完成。复用 Stage0 已产出的结构性摘要 + API 预研做对照核对，不重新逆向分析代码。

## 派发

```
Agent({
  subagent_type: "general",
  description: "设计一致性：S1-S7 检查",
  prompt: "设计实现一致性检查

【上下文信息】
- 设计文档输入：{docs_input}（单个 .md 文件路径 或 文档目录路径）
- 代码文件：{code_file_path}（文件检视）；或 PR 模式：diff 路径 {diff_file_path} + 完整源码 {repo_path}
- 结构性摘要：{code_summary_path}（含入口/数据流/分支表/API索引/TilingData值域/芯片架构参数）
- API 预研报告：{api_prestudy_path}（仅 Kernel 侧，若存在）
- yaml 提交端点：http://127.0.0.1:{collector_port}/submit（将完整 S1-S7 + D8 结果通过 curl 提交到此端点，**禁止使用 Write 工具写 yaml 文件，禁止读取 yaml 输出目录中的已有文件**）

【执行要求】
1. 读设计文档：{docs_input} 为单文件则 Read；为目录则枚举 .md/.yaml（跳过 LOG.md、*_REVIEW.md、*-report.md），逐个 Read，提取设计要素（架构/分支/API/数据流/参数/伪代码/约束）并标注来源文件名
2. 建立设计映射：对每个设计要素 Grep 代码定位对应实现，对比判定 ✅实现/❌未实现/⚠️有偏差/N/A，填入设计映射表
3. 逐策略检查 S1-S7（见下方策略说明）
4. **将 S1-S7 + D8 完整结果按 yaml schema 通过 curl 提交到 collector 端点**。提交命令模板：
   ```
   curl -s -X POST "http://127.0.0.1:{collector_port}/submit?group=design&clause=design" --data-binary @- <<'YAML_EOF'
   <yaml 内容>
   YAML_EOF
   ```
   collector 自动生成 `design_design.yaml` 文件。**禁止使用 Write 工具写 yaml 文件，禁止以文本消息返回结果**（仅返回「已完成，design.yaml 已提交」即可）
5. ❌ 项必须附具体偏差描述 + 代码位置（文件:行）+ 设计依据（来源文档原文引用，标注文件名+章节），存入 yaml 的 `deviations` / `doc_violations` 字段
6. **D8 文档格式检视（附带项）**：S1-S7 完成后，对步骤1 已 Read 的所有 .md 文档做格式合规检查。Read `references/doc-style.md` 获取 D1-D4 四条规则，逐文档逐规则对照，违反即标记 ❌。D8 独立于 S1-S7（检视文档自身格式，不依赖设计映射），不走假设检验
7. 禁止生成报告文件

【对照核对原则】
- S1 架构 / S2 分支 / S4 数据流 / S5 参数语义 / S7 约束：优先复用 {code_summary_path} 中的实现事实（入口函数、数据流、分支覆盖表、TilingData 值域、芯片架构参数），与设计文档对照
- S3 API 清单：优先复用 {api_prestudy_path} 的 API 约束，检查代码 API 使用与设计是否一致、是否用了黑名单接口
- 仅当摘要/预研未覆盖某要素时，才 Grep 代码补充

【独立性】本检查不引 core/methodology.md 假设检验框架、不依赖 plan-design 输出。它是设计 vs 实现的对照检查，与编码条例检视是平行轨道，输出格式不同（✅/❌/N/A vs PASS/FAIL/SUSPICIOUS）。**例外**：D8 文档格式检视读取 `references/doc-style.md` 获取格式规则，但同样不走假设检验（格式违反为确定性判定）。"
})
```

---

## S1-S7 策略说明

| 策略 | 维度 | 检查内容 |
|------|------|---------|
| **S1** | 架构匹配 | Kernel类型（`__vector__`/`__mix__`/`__global__`）、硬件单元（Cube/Vector/Scalar）、流水线模式（同步/异步/AIC-AIV协同）、存储层级（L1/L0/CO1/UB）是否与设计文档集一致。架构级别不匹配直接定性为「不一致」。 |
| **S2** | 分支覆盖 | 从设计文档集提取所有条件分支（if/else、switch、分支场景表、穿刺目标），在代码中搜索每个分支的对应处理。标记缺失分支（设计有代码无）和多余分支（代码有设计无）。 |
| **S3** | API清单 | 从设计文档集提取API列表，grep 每个API的实际使用，检查参数是否匹配（RoundMode、数据布局、dtype/shape等），检查是否使用了API黑名单接口。 |
| **S4** | 数据流追踪 | 追踪数据从输入到输出的完整路径，对比设计文档集与实现的数据流是否一致。 |
| **S5** | 参数语义 | 检查关键参数（tiling配置、模板参数、常量定义）的含义和使用是否与设计文档集一致。 |
| **S6** | 伪代码映射 | 对照设计文档集中的伪代码描述，核实代码中对应逻辑的实现完整性。 |
| **S7** | 约束合规 | 检查设计文档集中的约束条件（对齐要求、取值范围、内存限制、精度容差）是否被满足。 |
| **D8** | 文档格式 | 对已 Read 的 docs_input 目录下所有 .md 文档，按 `references/doc-style.md` 的 D1-D4 规则检查格式合规（中英文/数字间距、列表标点一致、中文全角标点、半角数字）。不走假设检验，违反即 ❌。独立于 S1-S7，不依赖设计映射。 |

---

## 输出格式

将完整结果通过 `curl -X POST "http://127.0.0.1:{collector_port}/submit?group=design&clause=design"` 提交（collector 自动生成 `design_design.yaml`，子 Agent 不接触目录路径）。

```yaml
type: design

strategies:
  - id: S1
    name: 架构匹配
    verdict: "✅"               # ✅ | ❌ | N/A（用引号包裹避免 yaml 解析问题）
    design_desc: {设计描述}      # 从设计文档提取的期望（✅/❌ 项都填）
    impl_desc: {实现实际}        # 代码实际实现（✅/❌ 项都填）
  - id: S2
    name: 分支覆盖
    verdict: "❌"
    design_desc: {设计描述}
    impl_desc: {实现实际}
  # ... S3-S7 同结构
  - id: D8
    name: 文档格式
    verdict: "❌"
    design_desc: ""              # D8 无设计映射，留空
    impl_desc: ""

design_mapping:                  # 设计映射表
  - element: {设计要素}
    source: {文件名 §章节}
    design_desc: {文档描述}
    impl_location: {代码文件:行}
    status: "✅"                 # ✅ | ❌ | ⚠️

deviations:                      # S1-S7 的 ❌ 项详情
  - strategy_id: S2
    desc: {偏差描述}
    code_location: {文件:行号 或 文件:起始行-中止行，单个位置，禁止拼接多位置或附加说明文本}
    design_basis: {来源文档原文引用，标注文件名+章节}

doc_violations:                  # D8 的 ❌ 项详情（格式与 S1-S7 不同）
  - doc_name: {文档名}
    violation_location: {文档:行号 或 文档:起始行-中止行，单个位置，禁止拼接多位置或附加说明文本}
    desc: {违规描述}
    fix_suggestion: {修复建议}
```

**字段对照**（原文本输出 → yaml 字段）：
- `S1 架构匹配: ✅/❌/N/A`（前8行判定） → `strategies` 列表，每项含 `id` + `name` + `verdict`
- `设计映射表`表格 → `design_mapping` 列表
- S1-S7 ❌项的「偏差描述 + 代码位置 + 设计依据」 → `deviations` 列表
- D8 ❌项的「文档名 + 违规位置 + 违规描述 + 修复建议」 → `doc_violations` 列表

**注意**：
- verdict 字段值 `✅`/`❌`/`N/A` 必须用双引号包裹，避免 yaml 将 `:` 等字符误解为键值分隔
- `code_location` / `violation_location` 必须是单个 `文件:行号` 或 `文件:起始行-中止行`，禁止拼接多位置（如 `L743、L858`）、禁止附加说明文本（如 `...:400-408（说明）`）。多位置场景拆成多条 `deviations` / `doc_violations` 记录
- `deviations` 仅含 S1-S7 的 ❌ 项（D8 的 ❌ 项进 `doc_violations`，不进 deviations）
- 无 ❌ 项时 `deviations` 和 `doc_violations` 为空列表 `[]`
- design_mapping 可为空列表（设计映射表是内部产物，供子 Agent 对照用，报告只取 strategies 判定 + deviations 详情）

禁止生成报告文件。
