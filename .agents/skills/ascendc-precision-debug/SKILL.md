---
name: ascendc-precision-debug
description: Ascend C 算子精度调试技能，提供精度问题诊断和解决方法。触发：输出异常（全为0、随机值、未初始化）、精度验证失败（rtol/atol 不达标）、FP16 精度差于预期、Cast 后数据错误、需要排查流水线同步（EnQue/DeQue）或 DataCopy 对齐问题。
---

# Ascend C 算子精度调试

## 核心理念

> **精度调试 = 理解 + 分析 + 定位 + 修复**

1. **理解数据类型限制**：FP16 约 3-4 位有效数字，FP32 约 6-7 位。
2. **识别数值稳定性问题**：大数吃小数、灾难性抵消。
3. **掌握科学调试方法**：从最小复现到根因分析。

## 使用时机

**适用**：精度验证失败（rtol/atol 不达标）、输出全为0或随机值、FP16差于FP32、特定数值范围误差大、流水线同步问题、DataCopy对齐问题。

---

## 调试前置要求 ⭐⭐⭐

> 进入调试前**必须**完成以下三步

### 1. 固定最小可复现用例

| 项目 | 说明 | 示例 |
|------|------|------|
| Shape | tensor 形状 | `{8, 16}` |
| Dtype | 数据类型 | `float16` |
| 固定值 | 具体数值 | `[1.0, 2.0, -0.5, ...]` |

**选择原则**：优先简单 → 优先32字节对齐 → 优先FP32 → 覆盖边界值

**💡 推荐实践**：调试时建议在至少两个 dtype（如 FP16 和 FP32）上用**同一 shape + 同一数据**验证。如果一种 dtype 通过另一种失败，可按下方对应的诊断模式快速缩小范围。

### 2. 检索官方文档 ⭐

> **禁止凭直觉修改代码**

**检索顺序**：
1. 使用 `$ascendc-docs-search` skill 搜索示例代码，查找类似算子。
2. 使用 `$ascendc-docs-search` skill 查阅 API 文档。
3. 对比官方实现与当前实现。

### 3. 清理缓存和临时文件

```bash
rm -rf build input output
mkdir -p build/input build/output
```

---

## 快速决策树

```
[前置检查] 已固定用例？已检索API？已清理缓存？
    │
    └─ 否 → 先完成前置步骤
    └─ 是 → 继续
        │
        ├─ [第0步] ⭐ 代码修改后输出完全不变？
        │   ├─ 是 → 清理 build/ 和 kernel_cache 后重试
        │   └─ 否 → 继续
        │
        ├─ [第0.5步] ⭐ 多 dtype 交叉验证
        │   ├─ FP32通过但FP16/BF16失败 → 精度不足，见下方诊断模式
        │   ├─ BF16通过但FP16/FP32失败 → API fallback 路径差异 或 精度阈值差异，见下方诊断模式
        │   └─ 全部通过/全部失败 → 继续
        │
        ├─ [第一步] 排查数据搬运 ⭐⭐⭐
        │   ├─ 输出是否全为 0 或随机错误？
        │   │   ├─ 是 → 检查流水线同步（EnQue / DeQue）⭐⭐⭐
        │   │   │       └─ DataCopy 后直接计算？→ 添加 EnQue/DeQue
        │   │   │       └─ 临时验证：加 PipeBarrier，若正确则确认同步问题
        │   │   ├─ 检查 DataCopy 是否 32 字节对齐
        │   │   │       └─ 非对齐 → 改用 DataCopyPad
        │   │   └─ 检查是否使用 GlobalTensor.SetValue
        │   │           └─ 是 → 改用 LocalTensor.SetValue + DataCopyPad 搬出到 GM
        │   └─ 验证：用 "CopyIn → CopyOut" 测试搬运
        │
        ├─ [第二步] 对比分析
        │   └─ 对比官方示例与当前实现 → 发现差异
        │
        └─ [第三步] 诊断问题类型
            ├─ 所有结果都差 → 公式/常量/API选择
            ├─ 个别值错误 → 边界条件/除零/溢出
            └─ 误差整体偏大 → FP16精度不足 → 尝试FP32中间计算
```

---

## 症状-原因速查表

| 症状 | 可能原因 | 诊断方向 |
|------|----------|----------|
| **输出全为 0 或随机错误** | 流水线同步缺失 / DataCopy 非对齐 / GlobalTensor.SetValue | 检查 EnQue / DeQue、数据对齐、改用 LocalTensor.SetValue + DataCopyPad ⭐⭐⭐ |
| `sum=0, max_err=输入级别` | 输出没写出 | 检查输出队列类型（VECIN vs VECOUT） |
| `sum=0, max_err≈0` | 输出全0/未初始化 | 检查 UB 溢出、buffer 分配 |
| `特定参数范围失败` | 阈值/边界错误 | 验证阈值计算、检查分支条件 |
| `非对齐数据失败` | DataCopy 对齐问题 | 改用 DataCopyPad |
| `FP16 差但 FP32 好` | 精度不足 | 中间计算用 FP32 |
| `Cast 后数据错误` | RoundMode 错误 | half → float用CAST_NONE，float → half用CAST_ROUND |
| **BF16 通过但 FP16/FP32 失败** | (1) 部分 API 不支持 BF16，BF16 走了更简单的 fallback 路径反而正确；(2) BF16 与 FP16/FP32 精度特性不同（BF16 mantissa 7bit vs FP16 10bit），精度阈值或溢出行为差异 | 先排查 API fallback 分支差异，再检查精度阈值（rtol/atol）是否适配各 dtype |
| **FP32 通过但 FP16/BF16 失败** | 半精度中间计算精度不足 | 升精度：Cast → FP32 计算 → Cast 回半精度 |
| **修改代码后输出完全不变** | 二进制未更新 / 编译器缓存 | 清理 build/ 和 $HOME/atc_data/kernel_cache/ 后重试 |

### 诊断模式："FP32 通过但 FP16/BF16 失败"

这是最常见的精度诊断信号：FP16/BF16 在 Ascend C 中通常共享相同的 Cast-to-FP32 计算路径（`if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>)`），FP32 通过说明核心算法正确，问题在半精度转换环节。

```
确认: FP32通过，FP16/BF16失败
    │
    ├─ 检查中间计算精度
    │   ├─ FP16/BF16 Cast→FP32 计算路径是否正确？
    │   ├─ 是否有未升精度的中间运算（如直接 Add<half>）？
    │   └─ 验证：将 FP16 路径的中间计算全部用 FP32，观察结果
    │
    ├─ 检查 Cast RoundMode
    │   ├─ half→float 应使用 CAST_NONE
    │   ├─ float→half 应使用 CAST_ROUND
    │   └─ 验证：对比 Cast 前后的数值
    │
    ├─ 检查 Pipeline 同步
    │   ├─ FP16/BF16 路径有额外的 Cast 操作，Cast 后是否 EnQue/DeQue？
    │   ├─ FP32 路径无 Cast，天然无此同步问题
    │   └─ 验证：在 Cast 后加 PipeBarrier，若正确则确认同步问题
    │
    └─ 检查 Buffer 大小
        ├─ FP16/BF16 路径需额外 Cast buffer（2× innerDim × sizeof(float)）
        ├─ FP32 路径不需要 Cast buffer
        └─ 验证：检查 Tiling 中 castBuf 大小计算
```

### 诊断模式："BF16 通过，但 FP16/FP32 失败"

这是一个有价值的诊断信号，可能由两类原因导致：

**原因1：API 不支持导致 fallback 路径差异**。部分 Ascend C 算术/归约 API 不支持 BF16，开发者为 BF16 实现更简单的 fallback 路径。BF16 fallback 通过说明逻辑正确，问题出在 FP16/FP32 路径中调用的复杂 API 上。

**原因2：精度阈值 / 数值范围差异**。BF16 (mantissa 7bit, 指数范围同 FP32) 与 FP16 (mantissa 10bit, 指数范围更小) 精度特性不同：BF16 不易溢出但尾数精度低，FP16 易溢出但尾数精度高。如果验证阈值（rtol / atol）未区分适配，或 FP16 发生溢出而 BF16 没有，就会出现 BF16 通过但 FP16 失败。

```
确认: BF16通过，FP16/FP32失败
    │
    ├─ [原因1] API fallback 路径差异
    │   ├─ 搜索代码中 if constexpr (std::is_same_v<T, bfloat16_t>) 分支
    │   ├─ BF16 走的 fallback 路径 vs FP16/FP32 走的主路径，差异在哪里？
    │   ├─ 列出 FP16 / FP32 路径中使用但 BF16 路径未使用的 API
    │   ├─ 逐个验证差异 API 的参数（mask、repeatTime、stride）
    │   └─ 临时将 FP16 路径改为与 BF16 相同的 fallback 实现，观察是否通过
    │
    ├─ [原因2] 精度阈值 / 数值范围差异
    │   ├─ FP16 是否溢出？（FP16 max ≈ 65504，BF16 指数范围同 FP32）
    │   ├─ 验证阈值（rtol/atol）是否按 dtype 区分？
    │   │   ├─ BF16: rtol=1e-2 级别（mantissa 仅 7bit）
    │   │   └─ FP16: rtol=1e-3 级别（mantissa 10bit）
    │   └─ 检查中间计算是否因 FP16 的更高精度要求暴露了算法缺陷
    │
    └─ 交叉验证
        ├─ 使用 `$ascendc-docs-search` skill 查阅文档确认相关 API 是否支持 BF16
        ├─ 如果 API 不支持 BF16 → 优先按原因1排查
        └─ 如果 API 支持 BF16（BF16/FP16 走相同路径）→ 优先按原因2排查
```

---

## 常见陷阱速查

| 陷阱 | 症状 | 解决方案 |
|-----|------|----------|
| **流水线同步缺失** | 输出全0或随机错误 | DataCopy 后必须 EnQue/DeQue 同步 ⭐⭐⭐ |
| **DataCopy 非对齐** | 小规模数据全0/异常 | 使用 DataCopyPad ⭐⭐⭐ |
| **GlobalTensor.SetValue** | 输出全为0 | 改用 LocalTensor.SetValue + DataCopyPad 搬出到 GM ⭐⭐⭐ |
| **Cast RoundMode** | Cast后数据混乱 | half→float用CAST_NONE，float→half用CAST_ROUND ⭐ |
| FP16 精度不足 | 简单计算也有误差 | 关键中间值用 FP32 |
| exp/log 溢出 | 出现 Inf 或 NaN | 先减最大值再计算 |
| 减法抵消 | a≈b 时 a-b 误差大 | 使用数值稳定等价公式 |
| Reduce 误差 | Reduce 结果比逐元素误差大 | 使用 FP32 累加器 |
| 除零风险 | NaN 或异常大值 | 添加 epsilon 保护 |

### 流水线同步调试

**核心问题**：DataCopy / DataCopyPad 是异步 DMA 操作，直接在搬运后的数据上做 Vector 计算可能读到未完成的数据！

```cpp
// ❌ 错误：AllocTensor 后直接用
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
Compute(x);  // 错！可能读到未完成搬运的数据

// ✅ 正确：DeQue 后再计算
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();  // 等待搬运完成
Compute(xIn);
```

**临时调试方法**：
```cpp
DataCopy(x, gm, size);
PipeBarrier<PIPE_ALL>();  // 临时加，如果结果正确说明是同步问题
Compute(x);
```

**如果 PipeBarrier 能解决问题，说明是同步问题** → 修复方案：改为 EnQue/DeQue 机制

| 误区 | 正确理解 |
|-----|---------|
| AllocTensor 后数据就可用 | AllocTensor 只分配内存，不等待搬运 |
| DataCopy 是同步的 | DataCopy 是异步 DMA，立即返回 |
| 不用 EnQue/DeQue 也能正常工作 | 必须用 EnQue/DeQue 或 PipeBarrier 同步 |
| PipeBarrier 性能好 | PipeBarrier 是全流水线停顿，性能差 |

详细说明见 [references/common-traps.md](references/common-traps.md)

---

## SIMT 精度调试

SIMT 算子有独特的精度调试问题：DCache 一致性、核内共享内存同步等。

详细说明见 [references/simt-precision-debug.md](references/simt-precision-debug.md)

### SIMT 特有精度陷阱

| 陷阱 | 症状 | 解决方案 |
|-----|------|----------|
| **DCache 一致性** | VF 外通过 Scalar 单点写 GM 后其他核读不到 | 使用 DataCacheCleanAndInvalid 刷新 |
| **SIMT printf 调试** | 需要在 VF 内打印变量 | VF 内用 AscendC::Simt::printf，VF 外用 PRINTF |
| **workspace 数据同步** | 多核通过 workspace 交换数据后读取不一致 | 在同步位置加 SyncAll() + SetScheduleMode(1) |

---

## 调试策略层级

```
调试方法
    │
    ├─ 快速方法（优先尝试，≤7次）
    │   ├─ 误差分布分析 → 识别误差模式
    │   ├─ Printf 特定位置 → 缩小范围
    │   ├─ DumpTensor 7步法 → kernel 内插桩 + CPU golden 逐段对比 ⭐
    │   └─ 常见陷阱排查 → 对症下药
    │
    └─ 二分调试（保底手段）
        └─ 快速方法尝试≥7次或方法穷尽时立即切换
```

> **重要原则**：不要盲目试错超过 7 次

### DumpTensor 7步法

kernel 内插桩调试的标准工具：在 CopyIn / Compute / CopyOut 关键点插入 `DumpTensor`，配合 CPU golden 同 desc 编号 (100/200/300) 逐段对比，快速定位异常出现在数据流哪一阶段。适用：输出错误、NaN/Inf、需要追踪 CopyIn → Compute → CopyOut 各阶段数据。

详细说明见 [references/ascendc-dumptensor.md](references/ascendc-dumptensor.md)

---

## 问题定位方法

### 1. 对比法（与工作的代码对比）

找到正常工作的代码，逐行对比差异

### 2. 边界二分法

记录通过/失败的临界点，分析分支选择

### 3. 数值验证法

不要相信估算公式，用代码计算实际值

### 4. Buffer 调试要点

| 问题 | 表现 | 解决方案 |
|------|------|----------|
| VECIN 用于输出 | 输出等于输入 | 输出必须用 VECOUT 队列 |
| Double Buffer 漏算 | 阈值错误 | 计算阈值时 ×2 |

详细定位流程见 [references/diagnosis-workflow.md](references/diagnosis-workflow.md)

---

## 精度标准来源优先级

1. **优先级1**：算子开发 Plan 中明确的精度要求
2. **优先级2**：华为昇腾官方精度标准文档
3. **优先级3**：本 Skill 默认值（仅作兜底）

| 数据类型 | rtol | atol |
|---------|------|------|
| FP16 | 1e-3 | 1e-4 |
| FP32 | 1e-5 | 1e-6 |
| INT | - | 0 |

---

## Agent 使用指南

### 调试计数规则

```
计数器 = 0
每次尝试快速方法（误差分析/Printf/陷阱排查）→ 计数器+1
当 计数器 >= 7 或 快速方法穷尽 → 立即切换二分调试
```

> **💡 经验建议**：如果多次尝试未取得进展（失败用例数量未减少），建议：
> 1. 检查是否清理了编译缓存（`rm -rf build/ $HOME/atc_data/kernel_cache/`）
> 2. 验证修改后二进制 sha256 是否确实改变
> 3. 切换到完全不同的调试策略（如二分法降级到最小工作路径）

### 调试总结要求

每步调试成功后**必须**形成总结：

- 文档描述不清晰的地方
- 需要改进的地方及推荐方案
- 给 Agent 造成困扰的点
- 调试过程记录

总结模板：[references/debug-summary-template.md](references/debug-summary-template.md)

### 检查清单

**调试阶段**：
- [ ] 已固定最小可复现用例
- [ ] 已通过 `$ascendc-docs-search` skill 确认 API 用法 ⭐
- [ ] 已清理缓存和临时文件
- [ ] **已排查流水线同步问题**（DataCopy 后是否 EnQue/DeQue）⭐⭐⭐
- [ ] **已排查输出全为 0 问题**（DataCopy 对齐 / GlobalTensor.SetValue → LocalTensor.SetValue + DataCopyPad）⭐⭐⭐
- [ ] 对比官方示例与当前实现
- [ ] 尝试次数 < 7
- [ ] 达到阈值立即切换二分调试

---

## 参考资料

### 工作流程
- [diagnosis-workflow.md](references/diagnosis-workflow.md) - 完整诊断工作流程
- [binary-search-debug.md](references/binary-search-debug.md) - 二分调试详细指南

### 问题诊断
- [common-traps.md](references/common-traps.md) - 常见精度陷阱详解
- [best-practices.md](references/best-practices.md) - 最佳实践

### 调试工具
- [printf-debug.md](references/printf-debug.md) - Printf 调试法
- [data-comparison.md](references/data-comparison.md) - 数据对比法
- [tools-reference.md](references/tools-reference.md) - 工具和命令参考
- [ascendc-dumptensor.md](references/ascendc-dumptensor.md) - DumpTensor 7步法（含 API、错误模式）

### 实战案例
- [case-studies.md](references/case-studies.md) - 实战调试案例
- [debug-summary-template.md](references/debug-summary-template.md) - 调试总结模板
