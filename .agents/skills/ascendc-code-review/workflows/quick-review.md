# 快速检视场景

## 触发
检查是否有、有没有问题、快速检视、有什么风险、帮我看看有没有、是否存在.*问题

---

## 执行流程（主 Agent 直接执行，无阶段无子 Agent）

### Step 1 — 确认代码输入

- 若用户提供文件路径 → 确认文件存在，Read 代码
- 若用户直接粘贴代码片段 → 直接使用
- 若用户只说了函数名 → Grep 定位，Read 对应部分
- 若用户提到了 PR 号 → 可用 `scripts/get_gitcode_pr_diff.py --help` 查看用法后获取 diff，或用 `scripts/clone_pr_source.py --help` 查看用法后拉取源码

### Step 2 — 加载方法论

Read `core/methodology.md` 掌握假设检验流程（5 步：代码段识别 → H0/H1建立 → 证据收集 → 自信值计算 → 判定）。

### Step 3 — 关注点匹配

从用户描述中提取关注点关键词，在 `references/` 目录下搜索匹配的条例：

1. Grep 用户关注点关键词（如「溢出」「空指针」「DataCopy」）在 `references/*.md` 中的出现位置
2. 命中文件的 `<适用>` 头 → 检查语言/侧别是否匹配当前代码
3. 命中的具体条例 → 列入检视清单
4. 若未命中任何条例 → 提示「未找到与"{关键词}"匹配的条例，请换一种描述方式」

### Step 4 — 侧别过滤

按文件路径判定侧别（`op_kernel/` → Kernel，`op_host/` → Tiling）。
排除不适用的条例。

### Step 5 — 加载条例文档

Grep `^{条例ID}` 定位起始行号，再 Grep 下一个 `^####` 标题定位结束行号，Read offset={start} limit={end-start}，禁止 Read 整个文档。
**必须关注**该条例章节中是否包含「专属检视方法」「检视策略」——若有，严格按该指引执行，不可跳过。
若条例来自 ascendc-api / ascendc-perf / simt-api-analysis / mc2-specific，先用 `/ascendc-docs-search` 查阅对应 API 最新文档。

### Step 6 — 逐条例检视 + 输出

按 methodology 的 5 步流程逐条例执行并输出：

PASS（自信值<50%）→ 仅列 `[条例ID] PASS`

FAIL（≥70%）/ SUSPICIOUS（50-69%）→ 展开：
```
[条例ID] FAIL 置信度:HIGH
- 问题描述：{描述}
- 代码片段（行 N-M）：```{≥10 行上下文}```
- 证据：正向{X%} + 负向{Y%} = {累计}%
- 修复建议：{建议}
```

全部完成后输出汇总行：
```
快速检视 — {文件名} | 侧别: {Kernel/Tiling} | {N} PASS / {M} FAIL / {K} SUSPICIOUS
```

---

## 约束

- **禁止派发子 Agent**，主 Agent 直接执行全部步骤
- **禁止生成报告文件**，结果 inline 输出
- **仅加载匹配的条例文档**，不加载全部 references/
- 代码片段场景下，跳过需要完整源码上下文的深度分析
- 主 Agent 直接使用 methodology.md 的假设检验框架，不额外封装
