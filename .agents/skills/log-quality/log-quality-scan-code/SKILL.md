---
name: log-quality-scan-code
description: 从代码仓库中扫描日志语句，直接检查每条日志是否符合 log-quality-standard，并检测关键位置是否遗漏日志记录，输出按标准分组的不合规问题清单。触发词：扫描代码仓日志、结合代码仓分析日志质量、代码日志质量检查、检查关键路径是否有日志。
---

# 代码仓日志质量扫描器

从代码仓库中提取日志语句，直接检查质量，按标准分组输出不合规清单。

---

## 触发词

扫描代码仓日志、结合代码仓分析日志质量、代码日志质量检查

---

## 输入

| 输入 | 必需 | 说明 |
|------|------|------|
| 代码仓 | 必需 | cmotor / pymotor / mindie-llm / vllm / vllm-ascend |
| 代码根目录 | 必需 | 仓库在机器上的路径，如 `/home/h00906152/projects/cmotor` |
| 功能范围 | 可选 | 如不指定则扫描全仓；指定时只扫相关模块，如 `pymotor/coordinator` |

---

## 输出

| 输出 | 格式 | 说明 |
|------|------|------|
| 问题清单 | Markdown | 按标准分组的不合规日志列表 |
| 汇总统计 | 数字 | 每条标准的违规数量和覆盖率 |

**默认只输出问题清单**，不自动生成 diff。如需 diff，请在请求时明确说明"同时输出修改建议/diff"。

---

## 工作流程

### 第一步：加载 log-quality-standard

读取 `log-quality-standard` skill 的内容，确认其中定义的所有标准及其检查判断逻辑。

### 第二步：根据仓库类型选择 grep 模式

**cmotor（C++）：**
```bash
grep -rn "LOG_I\|LOG_W\|LOG_E\|LOG_D\|MINDIE_LLM_LOG\|ULOG_" \
  --include="*.cpp" --include="*.h" {代码根目录}
```

**pymotor（Python）：**
```bash
grep -rn "logger\.\(info\|warning\|error\|debug\)\|logging\.\(info\|warning\|error\|debug\)\|LOG\.\(info\|warning\|error\|debug\)" \
  --include="*.py" {代码根目录}
```

**vllm-ascend（Python）：**
```bash
grep -rn "logger\.\(info\|warning\|error\|debug\)\|self\.logger\.\(info\|warning\|error\|debug\)" \
  --include="*.py" {代码根目录}
```

**mindie-llm（C++ + Python 混合）：**
两种模式都跑，按文件类型分别处理。

### 第三步：上下文提取

对每条 grep 结果，提取：

| 字段 | 说明 |
|------|------|
| 文件路径 | 相对仓库根的路径，格式 `src/module/file.cpp:123` |
| 行号 | 日志语句所在行 |
| 日志级别 | 从宏名/函数名判断：LOG_E→ERROR，LOG_W→WARNING，LOG_I→INFO，LOG_D→DEBUG |
| 日志内容 | 格式字符串原样保留，占位符不变 |
| 函数名 | 该日志语句所在函数 |
| 类名（可选） | 所属类 |
| 条件判断（可选） | 日志前的 if/else 分支条件 |

### 第四步：按标准逐条检查

对每条日志，检查是否符合 `log-quality-standard` 中定义的所有标准。注意：**按标准分组输出**，不是按日志分组输出。

#### 标准1~N：已有日志质量检查

> 以下为当前 standard 中各标准的内容摘要，运行时读取 `log-quality-standard` 原文确认。

（标准1~N 的具体检查项，按 standard 实际内容动态读取）

#### 标准7：关键位置日志完整性检测

这是**新增检查**，检测关键代码路径是否有日志记录。

**检测策略**：通过代码结构分析，识别以下"沉默区域"：

1. **成功路径无日志**：函数正常返回前没有任何 INFO 日志
   - 如 `connect()` 成功但无日志、模型加载成功但无日志

2. **外部调用成功后无日志**：RPC/网络/磁盘 IO 成功后无任何日志
   - 如 `requests.get()` 成功但无日志、`read_file()` 成功但无日志

3. **关键分支无日志**：if 失败分支有日志但成功分支无日志
   - 如 `if ret != 0: logger.error` 但 `if ret == 0:` 后无任何日志

**检测方法**：
```python
# 伪代码
for each function in target_modules:
    build CFG (control flow graph)
    for each path from entry to exit:
        if path has no logger calls:
            # 排除启动/初始化函数
            if not is_init_function(func):
                report("沉默路径: {file}:{lineno} {func} - {path_desc}")
```

**判断规则**：
- 跳过 `__init__`、`__enter__`、`__exit__` 等初始化函数
- 跳过私有辅助函数（以 `_` 开头）
- 只关注关键执行路径（请求处理、调度、模型推理、链路管理）
- 外部调用成功后至少有 DEBUG 以上级别日志

### 第五步：按标准分组输出

**按标准分组**，每组包含：该标准下所有不合规日志的列表。

```markdown
# 代码仓日志质量扫描报告

## 基本信息
- **仓库**：cmotor
- **代码路径**：/home/h00906152/projects/cmotor
- **扫描范围**：全仓
- **日志总数**：X 条

---

## 汇总统计

| 标准 | 合规数 | 违规数 | 覆盖率 |
|------|--------|--------|--------|
| `log-quality-standard` 中的所有标准（按 standard 动态生成） | X | X | XX% |
| 标准A：关键位置日志完整性（本 skill 扩展检查） | X | X | XX% |

---

## 标准1~N：`log-quality-standard` 中的所有标准

> 读取 standard，动态生成每个标准的 section。标准编号和名称以 standard 原文为准。

**（standard 中各标准的违规日志列表，按标准名称动态生成）**

---

## 标准A：关键位置日志完整性

**缺少日志的关键位置**（X 处）——关键执行路径没有任何日志记录

> 注：此标准是本 skill 的扩展检查，通过分析代码的控制流图（CFG）检测"沉默区域"，独立于 `log-quality-standard`。

| # | 文件位置 | 函数/类 | 缺失日志的关键路径 | 建议 |
|---|---------|---------|------------------|------|
| 1 | src/router/connection.cpp:145 | connect_to_peer | 成功返回路径（return 0 之前）无任何日志 | 建议添加 INFO 日志记录连接成功 |
| 2 | src/core/scheduler.cpp:234 | schedule_batch | 批量调度成功完成无日志 | 建议在关键节点添加 INFO 日志 |
| 3 | src/model/loader.cpp:89 | load_weights | 模型权重加载成功无日志 | 建议添加 INFO 记录加载结果 |

## 输出文件

| 输出 | 文件名 |
|------|--------|
| 问题清单 Markdown | `{仓库名}_日志扫描报告_{日期}.md` |

---

## 关于 Diff 输出

本 skill **默认只输出问题清单**，不自动生成修改 diff。

如需 diff，请在触发时明确说明，例如：
- "扫描代码仓日志，同时输出修改建议"
- "扫描 cmotor 代码仓，生成 diff"
- "扫描代码仓日志，输出 log-quality-rewrite 格式的修改对比"

收到明确要求后，调用 `log-quality-rewrite` skill 生成逐条修改对比和代码 diff。

---

## 注意事项

- 占位符（`{id}`、`%s`、`%d`）原样保留，不替换为实际值
- 同一日志语句在同一个文件:行号位置，只出现一次，不因多处调用而重复列
- 防刷屏检查需要结合代码上下文（如循环结构），在上下文不足时标注"需结合代码确认"
- 隐私检查基于参数名推断，不做运行时值分析
- 标准分组逻辑独立于 standard 内容，按标准名称动态生成各 section
