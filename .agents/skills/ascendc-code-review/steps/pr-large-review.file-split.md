# 文件分组（大型 PR 检视）

派发为子 Agent 执行。主 Agent 已从 diff 提取文件路径列表，子 Agent 负责按亲缘性分组。

## 派发

```
Agent({
  subagent_type: "general",
  model: "haiku",
  description: "文件分组",
  prompt: "文件分组

【输入】
- 变更文件路径列表：{file_path_list}
- diff 文件路径：{diff_file_path}

【执行要求】
严格按本文件「执行流程」中定义的步骤执行，产出 file_groups。禁止生成报告文件。"
})
```

---

## 执行流程（子 Agent 执行指南）

## 背景

大型 PR 通常是一类算子内部大量文件的变更（非跨多算子）。分组的目的是让每个通用检视子 agent 的上下文可控，同时保持有亲缘关系的文件在同一组内（例如互相 #include 的 .cpp 和 .h）。

## 执行流程

主 Agent 已完成快速探测（提取文件路径列表）。子 Agent 直接从分组开始。

### Step 1 — 按目录粗分

```
对每个变更文件路径:
  if 路径在 op_kernel/ 下 → 归入 kernel_files
  if 路径在 op_host/ 下   → 归入 host_files
  else                     → 归入 shared_bucket
```

### Step 2 — 按亲缘性细分组

对 kernel_files 和 host_files 分别做细分组，目标每组 ~5 个文件：

```
对 kernel_files:
  1. 按文件名前缀聚类（如同一算子的不同变体往往共享前缀）
     例：flash_attention_score_*.cpp/.h → 同组
  2. 检查 #include 关系：A.cpp include B.h → A 和 B 放同组
  3. 每组 3-7 个文件，避免单组过大
  4. 若 kernel_files 总数 ≤7 → 合并为 1 组

对 host_files:
  同样逻辑。host 文件通常更少，≤7 个时合并为 1 组
```

### Step 3 — 命名

每组命名为：`{算子名}_{kernel/host}_G{N}`（如 `flash_attention_kernel_G1`）
shared_bucket 命名为：`shared`

### 输出格式

```
算子: {从文件路径提取的算子名}
总文件数: {N} | 分组数: {M}

文件组清单:
  flash_attention_kernel_G1: 侧别=Kernel | 文件数=5
  flash_attention_kernel_G2: 侧别=Kernel | 文件数=4
  flash_attention_host_G1:   侧别=Tiling | 文件数=6
  shared:                    侧别=混合  | 文件数=3 | 类型=C++/Build

建议: {继续/切换标准流程}
```

## 约束

- 只读 diff 文件路径列表，不读完整 diff 内容（渐进式加载）
- 有 #include 关系的文件尽量放同组
- 每组 3-7 个文件，不硬切——亲缘性优先于数量
- 不生成报告文件，输出直接返回给主流程
