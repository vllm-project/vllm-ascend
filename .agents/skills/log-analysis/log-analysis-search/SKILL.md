---
name: log-analysis-search
description: 从代码仓库中搜索日志语句，提取原始日志及其上下文（函数名、类名、文件位置）。输入仓库名+功能描述，输出原始日志列表供后续结构化使用。触发词：搜索日志、提取日志、日志搜索、grep日志、整理日志语句。
---

# 日志搜索器

从代码仓库中搜索指定功能的日志语句，提取原始日志及其上下文，为结构化做准备。

---

## 触发词

搜索日志、提取日志、日志搜索、grep日志、整理日志语句

---

## 输入

| 输入 | 必需 | 说明 |
|------|------|------|
| 代码仓 | 必需 | cmotor / pymotor / mindie-llm / vllm / vllm-ascend |
| 功能描述 | 必需 | 自然语言，如"缩P保D"、"PD实例建链"、"vLLM调度" |
| 代码根目录 | 必需 | 仓库在机器上的路径，如 `/home/h00906152/projects/cmotor` |

---

## 输出

| 输出 | 格式 | 说明 |
|------|------|------|
| 原始日志列表 | Markdown 列表 | 每条包含：文件位置、行号、日志级别、日志内容、函数/类上下文 |

---

## 工作流程

### 第一步：确认仓库和路径

根据用户提供的仓库名，确认：
1. 代码根目录路径
2. 涉及的子目录/模块（缩小搜索范围）
3. 日志宏/函数的命名模式

### 第二步：选择搜索模式

根据仓库语言选择 grep 命令：

**cmotor（C++）：**
```bash
grep -rn "LOG_I\|LOG_W\|LOG_E\|LOG_D" --include="*.cpp" --include="*.h"
grep -rn "MINDIE_LLM_LOG" --include="*.cpp" --include="*.h"
grep -rn "ULOG_" --include="*.cpp" --include="*.h"
```

**pymotor / vllm / vllm-ascend（Python）：**
```bash
grep -rn "logger\.\(info\|warning\|error\|debug\)" --include="*.py"
grep -rn "logging\.\(info\|warning\|error\|debug\)" --include="*.py"
grep -rn "LOG\.\(info\|warning\|error\|debug\)" --include="*.py"
```

**mindie-llm（C++ + Python 混合）：**
两种模式都用，根据文件类型选择。

### 第三步：上下文提取

对每条 grep 结果，提取：
- **文件路径**：相对仓库根的路径，格式 `src/module/file.cpp:123`
- **行号**：日志语句所在行
- **日志级别**：INFO / WARN / ERROR / DEBUG
- **日志内容**：保留格式字符串，占位符不变（`{placeholder}`、`%s`、`%d`）
- **函数名**：该日志语句所在函数
- **类名**（如有）：所属类
- **条件判断**（如有）：日志前的 if/else 分支条件，帮助理解触发场景

### 第四步：按功能过滤

根据功能描述，只保留相关模块的日志：
- 用 `grep -rn "相关模块名"` 先定位模块目录
- 再在模块内搜索日志语句

### 第五步：输出原始列表

格式：
```markdown
## 原始日志列表

| # | 文件位置 | 函数/类 | 日志级别 | 日志内容 |
|---|---------|---------|---------|----------|
| 1 | src/router/link_manager.cpp:128 | LinkManager::Establish | INFO | [router] Link established, group_id={group_id} |
| 2 | src/router/link_manager.cpp:145 | LinkManager::Establish | ERROR | [router] Link failed, error_code={code}, peer={peer} |
```

---

## 注意事项

- 占位符必须原样保留，不要替换成实际值
- 日志内容中的缩进/格式保持原样
- 如果同一个日志在多处被调用，只列一次，注明"被多处调用"
- 搜索范围不要超出用户指定的功能边界
