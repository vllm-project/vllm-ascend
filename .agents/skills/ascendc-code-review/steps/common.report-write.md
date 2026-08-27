# 撰写报告（文件检视 + PR 检视）

## 执行步骤

### 1. 调用报告拼接脚本

```bash
python3 {skill_base}/scripts/workflow.assemble_report.py \
    --dir {yaml_output_dir} \
    --output {报告输出路径}
```

脚本读取 `{yaml_output_dir}` 下所有 yaml（clause 类 + design 类），按报告格式组装完整 md 报告正文，写入 `{报告输出路径}`。

**脚本分流逻辑**：
- `category: clause` + 非 out_of_range → 检视统计表 + 分级发现章节（HIGH/MED/LOW）
- `category: clause` + `out_of_range: true` → 范围外备注章节（计入统计，单独章节展示，格式与正文一致）
- `category: style` → 代码风格章节（不进统计表）
- `type: design` → 设计一致性检查章节（S1-S7 + D8 判定表 + ❌项详情）

### 2. 主 Agent 补头部元信息

脚本输出的报告 `## 检视概览` 章节含占位符，主 Agent 用实际值替换：

| 占位符 | 替换值来源 |
|--------|----------|
| `{{CODE_FILE}}` | file_input（文件检视）或 PR 变更文件列表（PR 检视） |
| `{{SIDE}}` | code-summarize 返回的侧别（Kernel/Tiling/混合） |
| `{{DOC_LIST}}` | plan-design 返回的匹配规则文件列表 |
| `{{TOTAL}}` | plan-design 返回的总条例数（不含 style，含 out_of_range） |
| `{{DOCS_INPUT}}` | docs-detect 返回的 docs_input（或"未检测到"） |
| `{{TIMESTAMP}}` | 当前检视时间戳 |

替换方法：Read 报告文件，用 Edit 工具逐个替换占位符；或用 `sed -i` 批量替换。

### 3. 报告路径

- 文件检视：`./operators/{operator_name}/{source_file}_review_summary.md`
- PR 检视：`./operators/pr-{pr_number}/{pr_number}_review_summary.md`

### 4. 超长报告过滤

补完头部元信息后，检查报告行数：

```bash
wc -l {报告输出路径}
```

- 若 ≤5000 行：跳过过滤，报告完成
- 若 >5000 行：Read + 执行 `steps/common.report-filter.md`，派发过滤子 Agent。子 Agent 读取报告、按严重程度过滤 HIGH/MED/LOW 章节中的不严重条目、更新统计数字、原地覆盖报告文件。过滤完成后报告才视为最终成品

## 报告章节结构（脚本自动生成）

1. `# 代码检视报告` + `## 检视概览`（占位符，主 Agent 替换）
2. `## 检视统计`（PASS/FAIL/SUSPICIOUS 计数，不含 style，含 out_of_range）
3. `## 设计一致性检查`（仅 design.yaml 存在时生成；含 S1-S7 + D8 判定表 + ❌项详情）
4. `## 发现问题（HIGH 置信度）`
5. `## 需关注（MED 置信度）` / `## 疑似（LOW 置信度）`
6. `## 范围外备注（PR diff 未覆盖）`（仅 PR 检视、有 out_of_range 项时生成；格式与正文发现一致，含代码片段+行号+修复建议）
7. `## 代码风格`（仅含 category:style 的 FAIL 结果时生成；无违反时显示「全部符合代码风格规范」）

## 约束

- 报告正文由脚本一次性组装，主 Agent 仅负责补头部元信息（替换占位符）和超长报告过滤触发（>5000 行时派发过滤子 Agent）
- 每个 FAIL/SUSPICIOUS 发现的代码片段行号使用阶段2行号校对后的结果（yaml 已被 workflow.line_verify.py 更新）
- PR 检视报告的代码片段标注完整文件路径（相对 repo_path），由子 Agent 写入 yaml 时保证
- `[STYLE]` 前缀标记已改为 yaml `category: style` 字段，脚本据此分流到代码风格章节、不进统计表
- 设计映射表留作专项检视子 agent（design-check）内部产物（design.yaml），报告只放 S1-S7 + D8 判定表 + ❌ 项定位；无 design.yaml 时不生成设计一致性章节，报告退化为纯条例检视报告
- 无 category:style 的 yaml 时不生成代码风格章节
