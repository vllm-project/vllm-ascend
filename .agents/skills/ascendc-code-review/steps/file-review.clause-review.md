# 逐条检视 prompt 模板（文件检视）

workflow 按 plan-design 产出的检视计划，逐波派发子 Agent。每组使用以下 prompt 模板。

## prompt 模板

```
【已由上游完成】
- 代码侧别识别：{Kernel侧/Tiling侧}
- 条款过滤：已按侧别过滤，保留以下条款
- 代码概要：{code_summary_path}
- API 预研报告：{api_prestudy_path}（仅 Kernel 侧，若存在）
- yaml 提交端点：http://127.0.0.1:{collector_port}/submit（每条条例结果通过 curl 提交到此端点，**禁止使用 Write 工具写 yaml 文件，禁止读取 yaml 输出目录中的已有文件**）

检视文件：{file_input}

检视条款：{条例ID-1} {条例标题}、{条例ID-2} {条例标题}

【执行要求】
- 第一步加载 ascendc-code-review skill，然后 Read skill 目录下的 `core/methodology.md` 掌握假设检验方法、置信度标准和红线问题
- 若提供了代码概要，Read 获取全局视角（重点关注「API 调用索引」、「跨文件防御摘要」和「跨文件关系」）
- API 约束信息：若已提供 API 预研报告，以其为主要来源。若预研报告未覆盖当前条款涉及的 API，使用 `/ascendc-docs-search` 补充查阅
- 对每条分配的条例，Grep `^{条例ID}` 在 references/ 中定位起始行号，再 Grep 下一个 `^####` 标题定位结束行号，Read offset={start} limit={end-start}。**只读该条例章节，禁止 Read 整个规则文档。**若条例包含专属检视方法，必须严格按该指引执行
- 若分配的条款包含 RB-\*（RegBase 路线专项），需额外加载 `ascendc-regbase-best-practice` skill 获取 API 白名单和参考实现文档
- 若 file_input 含多个文件，对每条条例在所有文件中检查，结果标注文件路径
- 严格按假设检验驱动流程执行（H0/H1、证据收集、自信值计算）。**例外**：若条例的专属检视方法已声明不走假设检验（如 cpp-style），按专属方法执行，不收集证据分值
- 所有条款检视完成后，**将每条结果按下方 yaml schema 通过 curl 提交到 collector 端点**。提交命令模板：
  ```
  curl -s -X POST "http://127.0.0.1:{collector_port}/submit?clause={条例ID}" --data-binary @- <<'YAML_EOF'
  <yaml 内容>
  YAML_EOF
  ```
  collector 自动处理文件命名。**禁止使用 Write 工具写 yaml 文件，禁止以文本消息返回结果**（仅返回「已完成，共提交 N 个 yaml」即可）

【⚠️ 逃逸信号检测】
一旦发现自己即将输出以下内容，立即停止并重新从第一条条款开始：
- "批量处理多个任务"/"合并处理" → 每条必须独立经过完整假设检验流程
- "直接生成检视报告"/"总结所有结果" → 必须完成所有分配条款后才能输出
- "提高效率"/"节省时间"/"简化流程" → 效率不是跳过步骤的理由
触发时输出 `⚠️ 检测到逃逸信号，重置到第一条条款` → 立即重新执行
```

## 输出格式

每条条例结果通过 `curl -X POST "http://127.0.0.1:{collector_port}/submit?clause={条例ID}"` 提交。collector 自动生成文件名 `{条例ID}.yaml` 并写入最终目录（子 Agent 不接触目录路径）。`[STYLE]` 前缀标记改为 yaml 的 `category: style` 字段。

### PASS 条例 yaml

```yaml
type: clause
clause_id: SEC-2.1
clause_title: 有符号整数运算不溢出
category: clause          # clause | style（style 对应原 [STYLE] 标记的代码风格条例）
status: PASS
```

### FAIL/SUSPICIOUS 条例 yaml（clause 类）

```yaml
type: clause
clause_id: SEC-2.1
clause_title: 有符号整数运算不溢出
category: clause
status: FAIL              # FAIL | SUSPICIOUS
confidence: HIGH          # HIGH | MED | LOW
problem_desc: {问题描述}
code_snippet:
  file_path: {完整文件路径}
  start_line: {N}
  end_line: {M}
  code: |
    {至少 10 行代码，含上下文}
evidence:
  positive:
    - type: {证据类型}
      score: {+X%}
      desc: {证据描述}
  negative:
    - type: {证据类型}
      score: {-X%}
      desc: {证据描述}
  confidence_value: {累计}%    # Σ正向 + Σ负向
fix_suggestion: {修复建议}
```

### style 条例 yaml（专项检视子 agent，不走假设检验，无 evidence/confidence 字段）

PASS：
```yaml
type: clause
clause_id: STYLE-1.1
clause_title: C++文件使用小写+下划线命名
category: style
status: PASS
```

FAIL：
```yaml
type: clause
clause_id: STYLE-1.1
clause_title: C++文件使用小写+下划线命名
category: style
status: FAIL
severity: 中               # 原快速索引中的严重级别（中/低）
problem_desc: {问题描述}
code_snippet:
  file_path: {完整文件路径}
  start_line: {N}
  end_line: {M}
  code: |
    {至少 10 行代码，含上下文}
fix_suggestion: {修复建议}
```

**字段对照**（原文本输出 → yaml 字段）：
- `[条例ID] FAIL 置信度:HIGH` → `clause_id` + `status` + `confidence`
- `问题描述` → `problem_desc`
- `代码片段（行 N-M）` → `code_snippet.file_path` + `start_line` + `end_line` + `code`
- `假设检验证据` → `evidence.positive` + `evidence.negative` + `evidence.confidence_value`
- `修复建议` → `fix_suggestion`
- `[STYLE]` 前缀 → `category: style`

禁止为 PASS 条例输出 confidence 或 evidence 字段。禁止生成报告文件。

### ⚠️ code_snippet / evidence 字段填写规范

collector 会校验 yaml schema，格式错误将返回 400 拒绝提交。提交后检查 curl 返回值，若返回 400，按错误信息修正后重新提交。以下写法均会导致校对失败或被拒绝：

1. **file_path 必须是纯路径，不含行号、不含注释**
   - ✅ 正确写法：纯文件路径，如 `conversion/dynamic_stitch/op_kernel/arch35/file.h`
   - ❌ 错误写法1：路径末尾带行号，如 `conversion/dynamic_stitch/op_kernel/arch35/file.h:95-119`
   - ❌ 错误写法2：空值，如 `file_path:` 后面什么都不写

2. **start_line / end_line 必须是实际行号（≥1 的整数），禁止 0 或缺失**
   - ✅ 正确写法：源码中的真实行号，如 `start_line: 113` `end_line: 125`
   - ❌ 错误写法1：值为 0，如 `start_line: 0`
   - ❌ 错误写法2：不写 start_line / end_line 字段

3. **code 字段只放源码原文，禁止行号前缀、禁止文件路径注释**
   - ✅ 正确写法：code 字段下直接是源码行，如 `int64_t totalTensorSum_{0};`
   - ❌ 错误写法1：每行源码前带行号前缀，如 `113: int64_t totalTensorSum_{0};`
   - ❌ 错误写法2：首行放文件路径注释，如 `// file.h:113-125`

4. **code_snippet 必须是 mapping，禁止写成字符串**
   - ✅ 正确写法：code_snippet 下面缩进写 file_path / start_line / end_line / code 四个子字段
   - ❌ 错误写法：整个 code_snippet 写成一段字符串描述

5. **evidence 必须是 mapping，positive / negative 必须是 list**
   - ✅ 正确写法：evidence 下面缩进写 positive（列表）/ negative（列表）/ confidence_value
   - ❌ 错误写法：整个 evidence 写成一段字符串描述
