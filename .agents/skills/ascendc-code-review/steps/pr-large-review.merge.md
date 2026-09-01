# 结果合并（大型 PR 检视）

主 Agent 执行。收集合成所有检视结果，去重，按文件组整理，产出结构化数据供 report-write 使用。

## 前置条件

- Stage 2 所有文件组的逐条检视结果已收集
- Stage 3 synthesize 综合研判已完成（含冲突解决和置信度过滤）
- shared bucket 检视结果已收集（如有）

## 执行流程

### Step 1 — 收集所有结果

从以下来源收集检视结果：
1. 各文件组的逐条检视输出（Stage 2）
2. shared bucket 检视输出（Stage 3，如有）
3. synthesize 的综合研判结果（冲突解决 + 置信度过滤后的最终结论）

### Step 2 — 去重

对同 (文件路径, 行号, 条例ID) 出现多次的情况：
- 若 synthesize 已解决冲突 → 使用最终结论
- 否则 → 保留首次出现的结论
- 记录去重日志：`去重: {N} 条重复结果已合并`

### Step 3 — 按文件组整理

```
报告结构:
  ## PR #{number} 检视报告

  ### 全局摘要
  {全局统计 + 系统性风险标注}

  ### 文件组1: {group_name}（{Kernel/Tiling}侧）
  {该组的 FAIL/SUSPICIOUS 详情}
  {该组的 PASS 汇总}

  ### 文件组2: {group_name}
  ...

  ### 共享文件
  {shared bucket 的 FAIL/SUSPICIOUS 详情}

  ### 待确认附录
  {置信度 <70% 的 SUSPICIOUS 结果}
```

### Step 4 — 产出结构化数据

输出结构化结果：
- `findings_by_group`: 按文件组分组的检视结果
- `shared_findings`: 共享文件检视结果
- `pending_appendix`: 待确认附录项
- `cross_group_risks`: 跨文件组风险标注
- `statistics`: 全局 + 逐文件组统计

## 输出

产出结构化数据直接传给 Stage 5 的 report-write。不生成独立报告文件。

## 约束

- 去重以 (文件, 行号, 条例ID) 三元组为 key
- synthesize 的冲突解决结果优先级最高
- PASS 结果不展开详情，仅计数
