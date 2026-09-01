# 代码审查参考手册

> 本文件由 Reviewer 在 Step 2（代码质量评估）时读取，逐项对照检查。

---

## 架构合规性检查

| 检查项 | 标准 | 严重级别 |
|--------|------|----------|
| 架构模式 | 使用 TPipe/TQue（非 LocalMemAllocator） | 高 |
| 入口属性 | `__global__ __aicore__` | 高 |
| 函数定义顺序 | Kernel 函数定义在调用之前，无前向声明 | 高 |
| 代码结构 | Kernel 类 -> 入口函数 -> Host 函数 -> main | 中 |

## 编码规范检查

| 检查项 | 标准 | 严重级别 |
|--------|------|----------|
| 矢量 API | 使用矢量 API，禁止 GetValue/SetValue 逐元素操作 | 高 |
| 数据对齐 | 满足 32 字节对齐要求 | 高 |
| 硬件参数 | 动态获取核数/UB 大小，禁止写死 | 高 |
| 命名规范 | `{功能}_custom` 命名 | 低 |
| 命名冲突 | 不使用标准库函数名（如 `sinh`、`exp`） | 中 |

## 性能分析检查

### 基础性能检查

| 检查项 | 参考 | 适用条件 |
|--------|------|----------|
| 双缓冲 | `00_add_doublebuffer/` | 大数据量（> 8K 元素） |
| Bank conflict | `01_bank_conflict/` | 连续内存访问 |
| L2 cache bypass | `03_l2_cache_bypass/` | 大数据搬运 |

### 循环模式检查（高风险）

| 检查项 | Grep 命令 | 问题级别 |
|--------|----------|---------|
| 循环内逐行 API 调用 | `grep -n "for.*{" file.asc -A 5 \| grep "AscendC::"` | 高 |
| 循环内逐元素操作 | `grep -n "for.*{" file.asc -A 5 \| grep "GetValue\|SetValue"` | 高 |

**问题示例**：
```cpp
// 错误：循环内逐行调用 API
for (uint32_t r = 0; r < R; r++) {
    AscendC::Sub<float>(xLocal[r * alignedCols], ...);  // 性能下降 30%+
}

// 正确：使用批量操作
AscendC::Sub<float>(xLocal, xLocal, tmpLocal, totalSize);
```

### 广播操作检查

| 检查项 | 说明 | 问题级别 |
|--------|------|---------|
| 低效广播模式 | 逐行循环广播 vs Brcb/mask 模式 | 高 |

### 调试代码检查

**适用时机**：仅在最终审查（预计 PASS 的最后一轮）时检查，开发过程中允许保留调试代码。

| 检查项 | Grep 命令 | 问题级别 |
|--------|----------|---------|
| printf 残留 | `grep -n "printf\|cout" file.asc` | 低（最终轮） |
| TODO/FIXME | `grep -n "TODO\|FIXME" file.asc` | 低 |

**审查策略**：
- 第一轮/中间轮：跳过此项检查，调试代码有助于问题定位
- 最终轮（其他问题已解决）：要求清理调试代码后再 PASS

### 内存访问模式检查

| 检查项 | 说明 | 问题级别 |
|--------|------|---------|
| 非连续访问效率 | DataCopyPad 的 stride 使用是否合理 | 中 |
| 多次 GM 读取 | 是否有不必要的重复 GM 读取 | 高 |

### 上板性能验证

**独立采集**：调用 `ops-profiling`，独立执行 msprof op 采集，不信任 Developer 的自报性能数据。

**审查要点**：
1. 实际 Task Duration 与理论耗时对比
2. PipeUtilization 分布是否与算子类型匹配
3. 核间负载均衡（各核耗时差异是否 <10%）
4. Bank conflict 占比是否 <5%
5. 头开销占比是否 <10%

**与 Developer 性能数据对比**：
- 读取 `operators/{operator_name}/docs/perf/` 目录下 Developer 的采集数据
- 对比 Reviewer 独立采集的结果，差异过大需在 REVIEW.md 中说明

**评分依据**：上板性能数据作为维度 4（性能优化 20 分）中 4.5 计算效率与上板性能的重要评分参考。

## API 选择审查

### 数据搬运 API

| API | 优势 | 劣势 | 适用场景 |
|-----|------|------|---------|
| DataCopy | GM<->UB 高效搬运，DMA 加速 | 要求数据对齐 | 对齐数据的 GM 搬运 |
| DataCopyPad | 支持非对齐、自动填充 | 额外填充开销 | 非对齐数据、需要填充 |
| Copy | UB 内部快速拷贝 | 仅限 UB->UB | UB 内部数据复制 |

**决策指南**：
- GM <-> UB 搬运 -> **DataCopy**（DMA 加速，性能最优）
- 数据非 32 字节对齐 -> **DataCopyPad**（自动处理对齐）
- UB 内部复制 -> **Copy**（避免不必要的 DMA 开销）

**检查清单**：
| 检查项 | 问题级别 |
|--------|---------|
| Copy 用于 GM 操作（仅支持 UB->UB） | 高 |
| DataCopy 用于非对齐数据（可能出错） | 高 |

### 队列操作 API

| API | 优势 | 劣势 | 适用场景 |
|-----|------|------|---------|
| TQue | 支持流水线、EnQue/DeQue 解耦 | 代码复杂度高 | 多阶段流水线 |
| TBuf | 简单直接、无队列开销 | 不支持流水线 | 单阶段简单操作 |

**决策指南**：
- 需要流水线（搬运/计算重叠）-> **TQue** + EnQue/DeQue
- 简单单阶段操作 -> **TBuf**（避免过度设计）

**检查清单**：
| 检查项 | 问题级别 |
|--------|---------|
| EnQue/DeQue 未配对（内存泄漏） | 高 |
| 队列深度 = 1 导致流水线阻塞 | 高 |
| 简单场景使用 TQue（过度设计） | 低 |

### 内存管理 API

| API | 优势 | 劣势 | 适用场景 |
|-----|------|------|---------|
| AllocTensor/FreeTensor | 灵活、按需分配 | 需手动配对管理 | 通用场景 |
| LocalMemAllocator | 批量分配、高性能 | 需预先规划 | 高性能批量场景 |

**决策指南**：
- 通用场景 -> **AllocTensor/FreeTensor**（必须配对）
- 高性能需求 + 批量分配 -> **LocalMemAllocator**

**检查清单**：
| 检查项 | 问题级别 |
|--------|---------|
| AllocTensor/FreeTensor 未配对 | 高 |

### 同步 API — 数据依赖分析

> **核心原则**：PipeBarrier 只在存在跨 pipe 数据依赖时才需要。同一 pipe 上的连续操作天然顺序执行，不需要 barrier。

**Pipe 分类表**（审查时必须对照）：

| Pipe | 包含的操作 |
|------|-----------|
| **PIPE_MTE2** | DataCopy(GM->UB)、DataCopyPad(GM->UB) |
| **PIPE_V** | Add、Sub、Mul、Div、Exp、Log、Abs、Max、Min、Adds、Muls、Cast、Duplicate、ReduceMax、ReduceSum、Compare、Select、Not、And、Or |
| **PIPE_MTE3** | DataCopy(UB->GM)、DataCopyPad(UB->GM) |
| **Scalar** | GetValue、SetValue、标量运算 |

**需要 barrier 的场景（跨 pipe 依赖）**：

| 场景 | 依赖类型 | 示例 |
|------|---------|------|
| CopyIn 后计算 | MTE2 -> V | `DataCopy(GM->UB)` 后 `Add/Mul` |
| 计算后 CopyOut | V -> MTE3 | `Muls` 后 `DataCopy(UB->GM)` |
| 归约后读标量 | V -> Scalar | `ReduceMax` 后 `GetValue(0)` |
| Cast 后计算 | V 写 -> V 读（同 pipe 但不同 tensor）| `Cast(a, b)` 后用 `b` 做 `Add` -- 注意：同 tensor 原地操作不需要 |

**不需要 barrier 的场景（同 pipe 连续操作）**：

| 场景 | 原因 | 反面示例 |
|------|------|---------|
| 连续矢量运算 | 全在 PIPE_V，硬件保序 | `Adds -> Exp -> Muls` 之间加 barrier |
| Duplicate 后矢量运算 | 都在 PIPE_V | `Duplicate -> ReduceMax` 之间加 barrier |
| 连续归约操作 | 都在 PIPE_V | `ReduceMax -> ReduceSum` 之间加 barrier |

**强制审查步骤 — 逐项依赖分析**：

对每个 `PipeBarrier` 调用，必须标注以下信息并输出到 REVIEW.md：

```
| 行号 | 前操作 | 前 Pipe | 后操作 | 后 Pipe | 依赖类型 | 判定 |
|------|--------|---------|--------|---------|---------|------|
| 271  | DataCopy(GM->UB) | MTE2 | Duplicate | V | RAW 跨 pipe | 必要 |
| 276  | Duplicate | V | ReduceMax | V | 同 pipe | 冗余 |
| 281  | ReduceMax | V | GetValue | Scalar | V->Scalar | 必要 |
| 286  | Adds | V | Exp | V | 同 pipe | 冗余 |
```

**判定规则**：
1. **前 Pipe != 后 Pipe** 且存在 RAW/WAW 依赖 -> **必要**
2. **前 Pipe = 后 Pipe**（同一执行单元）-> **冗余**（硬件保证顺序执行）
3. **无后续操作依赖此数据** -> **冗余**
4. 特殊情况：`PipeBarrier<PIPE_ALL>` 用于 TBuf 场景（无 EnQue/DeQue 自动同步）时，只有跨 pipe 依赖点才需要

**冗余率计算**：
```
冗余率 = 冗余 barrier 数 / 总 barrier 数 * 100%
```

**评分标准**：

| 检查项 | 问题级别 | 扣分标准 |
|--------|---------|---------|
| 流水线缺少同步导致数据竞争 | 高（阻塞） | 必须修复 |
| 同步位置错误（barrier 在依赖点之前） | 高（阻塞） | 必须修复 |
| 冗余率 > 50%（过度同步，严重影响性能） | 高 | 4.4 最多得 1 分 |
| 冗余率 30%-50%（中度过度同步） | 中 | 4.4 最多得 2 分 |
| 全部使用 PIPE_ALL 但无冗余 barrier | 低 | 4.4 最多得 3 分 |
| 精细 pipe 标识 + 仅依赖点同步 | 无 | 4.4 满分 4 分 |

**重要区分**：以下是两个独立的问题，不可混为一谈：
- **问题 A：PIPE_ALL vs 精细 pipe** — 同步粒度问题，可能因硬件原因需要 PIPE_ALL
- **问题 B：是否每个 API 后都加了 barrier** — 数据依赖分析问题，与用 PIPE_ALL 还是 PIPE_V 无关

即使因硬件限制必须用 `PIPE_ALL`，也**绝不意味着每个 API 后都该加 barrier**。Developer 用"硬件不支持精细同步"来为所有 barrier 辩护时，reviewer 必须反驳：**同 pipe 连续操作根本不需要任何 barrier，这与 pipe 粒度无关**。

### 计算 API

| API | 优势 | 劣势 | 适用场景 |
|-----|------|------|---------|
| 基础矢量 API (Add/Mul/Div 等) | 细粒度控制、可理解性强 | 需手动组合 | 通用计算 |
| 高阶封装 API (Softmax/LayerNorm) | 一行代码完成复杂操作 | 黑盒、调试困难 | **禁止使用** |

**决策指南**：
- 所有计算 -> **基础矢量 API**（Add/Mul/Sub/Div/Exp/Log/ReduceSum/ReduceMax/Cast）
- 归约操作 -> **ReduceSum/ReduceMax**（避免手动循环累加）
- 类型转换 -> **Cast**（避免直接赋值精度丢失）

**检查清单**：
| 检查项 | 问题级别 |
|--------|---------|
| 使用高阶封装 API（Softmax 等） | 高 |
| 手动循环累加（应使用 Reduce API） | 高 |
| 直接赋值替代 Cast（精度丢失） | 中 |

## Grep 检查命令

```bash
# 检查 EnQue/DeQue 配对
grep -c "EnQue" operators/{operator_name}/*.asc
grep -c "DeQue" operators/{operator_name}/*.asc

# 检查 AllocTensor/FreeTensor 配对
grep -c "AllocTensor" operators/{operator_name}/*.asc
grep -c "FreeTensor" operators/{operator_name}/*.asc

# 检查 Copy 用于 GM 操作（错误）
grep -n "Copy.*GlobalTensor\|Copy.*GM" operators/{operator_name}/*.asc

# 检查缺少 PipeBarrier
grep -n "DataCopy\|EnQue\|DeQue" operators/{operator_name}/*.asc | head -20

# 统计 PipeBarrier 总数（冗余率分析前置）
grep -c "PipeBarrier" operators/{operator_name}/*.asc

# 列出所有 PipeBarrier 及上下文（用于逐项依赖分析）
grep -n -B 3 "PipeBarrier" operators/{operator_name}/*.asc
```
