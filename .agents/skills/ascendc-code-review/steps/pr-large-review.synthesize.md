# 综合研判（大型 PR 检视）

主 Agent 执行。借鉴 Claude Code Coordinator 的 Synthesis 阶段。读完所有文件组 + shared bucket 的检视结果后，做跨文件组模式识别、冲突解决、置信度过滤。

## 前置条件

- Stage 2 全部检视波次完成（所有文件组的逐条检视结果已收集）
- Stage 3 shared bucket 检视完成（如有）

## 执行流程

### Step 1 — 跨文件组模式识别

扫描所有文件组的检视结果，检查以下模式：

**1.1 共享头文件防御缺失**
同一共享头文件（在 shared_bucket 中）被多个文件组 `#include` → 检查 shared bucket 的检视结果中是否有对应校验条例：
- 若 shared bucket 对该头文件的校验条例 PASS → 防御充分
- 若 shared bucket 未覆盖该文件 → 标注「跨组风险：{N} 个文件组引用 {file}，共享文件未检视到相关校验」

**1.2 系统性风险**
同一类问题（如同一 SEC 条例编号）在 ≥2 个文件组中 FAIL → 标注「系统性风险：{条例ID} 在 {文件组列表} 中均失败，建议全局排查」

**1.3 结论矛盾**
同 (文件, 行号, 条例ID) 出现在两个不同检视组中且结论不同（一个 PASS 一个 FAIL）→ 标记冲突

**1.4 跨文件组关联问题**
同一变量/函数在 kernel 组和 host 组的检视结果中出现关联问题 → 标注「kernel/host 关联风险」

### Step 2 — 冲突解决

对 Step 1.3 发现的每个冲突：
1. Read 对应文件和行号的代码
2. 参考 methodology.md 的判定标准重新评估
3. 以更严格的结论为准（FAIL > SUSPICIOUS > PASS）
4. 记录：`冲突已解决: {文件}:{行号} {条例ID} [{原结论A}] vs [{原结论B}] → {最终结论}`

### Step 3 — 置信度过滤

对每条 SUSPICIOUS / FAIL 结果逐条评估：

```
置信度 HIGH (≥80%)  → 直接进入报告正文
置信度 MED  (70-79%) → 进入报告正文，标注「建议人工确认」
置信度 LOW  (<70%)  → 归入「待确认」附录，不进入正文
```

对 SUSPICIOUS 结果：若代码审查后置信度 ≥70%，升级为 FAIL MED。

### Step 4 — 统计汇总

```
逐文件组统计:
  kernel_G1: PASS {N} / FAIL {N} / SUSPICIOUS {N}
  kernel_G2: PASS {N} / FAIL {N} / SUSPICIOUS {N}
  host_G1:   PASS {N} / FAIL {N} / SUSPICIOUS {N}
  shared:    PASS {N} / FAIL {N} / SUSPICIOUS {N}

全局汇总:
  总计 PASS {N} / FAIL {N} / SUSPICIOUS {N}
  系统性风险: {N} 项
  冲突已解决: {N} 项
  待确认附录: {N} 项
```

## 输出

综合研判结果直接追加到各文件组检视结果后，供 merge 步骤使用。不生成独立报告文件。

## 约束

- 只读检视结果摘要（PASS 仅 ID，FAIL/SUSPICIOUS 含代码片段），不重读完整源码（除非冲突解决需要）
- 不修改子 Agent 的原始结论，冲突解决时标注原始结论和最终结论
- 置信度过滤不改变原结论的置信度值，只决定是否进入正文
