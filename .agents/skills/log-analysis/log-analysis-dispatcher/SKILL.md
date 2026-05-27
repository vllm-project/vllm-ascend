---
name: log-analysis-dispatcher
description: 日志分析调度器。从代码仓库中提取日志，生成日志表格、Markdown 文档、故障模式库、诊断 Skill 全链路产出。触发词：整理日志、整理日志全流程、日志分析完整流程、日志全链路分析。
---

# 日志分析调度器

串联 6 个原子 skill，完成从代码仓库提取日志到生成诊断 skill 的全流程。

---

## 触发词

整理日志、整理日志全流程、日志分析完整流程、日志全链路分析

---

## 输入

| 输入 | 必需 | 说明 |
|------|------|------|
| 代码仓 | 必需 | cmotor / pymotor / mindie-llm / vllm / vllm-ascend |
| 功能描述 | 必需 | 自然语言，如"缩P保D"、"PD实例建链"、"vLLM请求级时长打点" |
| 代码根目录 | 必需 | 仓库路径，如 `/home/h00906152/projects/cmotor` |
| 实际日志 | 可选 | 环境中产生的真实日志文件或粘贴内容 |
| 输出目录 | 必需 | 所有产出文件的存放路径 |

---

## 输出（4 个文件）

| 产出 | 格式 | 说明 |
|------|------|------|
| 日志表格 | CSV（9列） | 结构化日志数据 |
| Markdown 文档 | .md | 含日志表格、Mermaid 流程图 |
| 故障模式库 | CSV（30列） | 基于 ERROR/WARN 提炼的故障模式 |
| 诊断 Skill | SKILL.md | 基于故障模式库生成的 log-diagnosis skill |

---

## 调用链

```
A: log-analysis-search
   输出：原始日志列表（Markdown）
        ↓
B: log-analysis-structurize
   输入：原始日志列表
   输出：日志表格 CSV（9列）
        ↓
C: log-analysis-correct（仅当提供了实际日志时调用）
   输入：CSV + 实际日志
   输出：校正后 CSV
        ↓
D: log-analysis-document
   输入：CSV
   输出：Markdown 文档（含 Mermaid 流程图）
        ↓
E: log-analysis-fault-mode
   输入：CSV
   输出：故障模式库 CSV（30列）+ 故障关联图
        ↓
F: log-diagnosis-skill-builder
   输入：故障模式库 CSV
   输出：log-diagnosis skill（SKILL.md）
```

---

## 执行流程

### 第一步：确认输入

1. 确认代码仓和代码根目录
2. 确认功能描述和边界
3. 确认输出目录
4. 确认是否有实际日志（C 的输入）

### 第二步：调用 Skill A — log-analysis-search

```
触发词：搜索日志
输入：代码仓、功能描述、代码根目录
操作：按仓库选择 grep 命令，提取原始日志及其上下文
输出：Markdown 格式的原始日志列表
```

### 第三步：调用 Skill B — log-analysis-structurize

```
触发词：结构化日志
输入：Skill A 的输出（原始日志列表）、功能描述
操作：按 9 列格式填充表格，按执行顺序编号
输出：CSV 文件（9列）
```

### 第四步：（可选）调用 Skill C — log-analysis-correct

```
条件：用户提供了实际日志
触发词：校正日志
输入：Skill B 的 CSV + 实际日志
操作：时序校正、遗漏补充、条件确认、去噪
输出：校正后的 CSV
```

### 第五步：调用 Skill D — log-analysis-document

```
触发词：生成Markdown
输入：CSV（校正后或原始）、功能描述、输出目录
操作：生成 Markdown 文档，含 Mermaid 流程图
输出：.md 文件
```

### 第六步：调用 Skill E — log-analysis-fault-mode

```
触发词：提炼故障模式
输入：CSV（校正后或原始）、输出目录、功能名称
操作：从 ERROR/WARN 日志提炼 30 列故障模式库
输出：故障模式库 CSV + 故障关联图 Markdown
```

### 第七步：调用 Skill F — log-diagnosis-skill-builder

```
触发词：编写诊断skill
输入：Skill E 输出的故障模式库 CSV、功能描述
操作：基于故障模式库生成 log-diagnosis skill
输出：SKILL.md 文件，放在 {输出目录}/diagnostic-skills/ 下
```

---

## 目录结构

所有产出放在用户指定的输出目录下：

```
{输出目录}/
├── {功能名称}日志整理.md              ← Skill D 输出
├── {功能名称}日志表格.csv              ← Skill B/C 输出
├── {功能名称}故障模式库.csv           ← Skill E 输出
├── {功能名称}故障关联图.md            ← Skill E 输出
└── diagnostic-skills/                  ← Skill F 输出
    └── log-diagnosis-{功能}.md         ← 诊断 skill
```

---

## 注意事项

- 如果用户未提供实际日志，跳过 Skill C（校正步骤）
- Skill F 生成的诊断 skill 名称格式：`log-diagnosis-{功能简写}`
- 故障模式库中的故障关键日志必须与 Skill B/C 输出的 CSV 表格条目对应
- 每个 Skill 执行完后检查输出是否合理，再进入下一步
