# AscendC DumpTensor Debugging

Systematic approach to debug AscendC kernels using DumpTensor API.

## Quick Start

```cpp
// Add DumpTensor after key computation points
DumpTensor(inputLocal, 100, 32);   // After CopyIn
DumpTensor(tmpLocal, 200, 32);     // After computation
DumpTensor(outputLocal, 300, 32); // Before CopyOut
```

## 7-Step Debugging Workflow

### 1. Add DumpTensor at Key Points

Insert after: CopyIn, each Compute step, before CopyOut.

```cpp
// After DataCopy
LocalTensor<T> inputLocal = inQueue.DeQue<T>();
DumpTensor(inputLocal, 100, 32);

// After computation
Adds(tmpLocal, inputLocal, 1.0f, tileLength);
DumpTensor(tmpLocal, 200, 32);

// Before CopyOut
DumpTensor(outputLocal, 300, 32);
```

### 2. Use Systematic Desc Numbering

| Range   | Stage         |
|---------|---------------|
| 100-199 | Inputs        |
| 200-299 | Intermediates |
| 300-399 | Outputs       |

Increment by 10 within each range.

### 3. Add CPU Golden Prints

Match NPU dump points with CPU prints:

```cpp
// CPU reference
printf("[CPU-100] input: %.6f, %.6f\n", input[0], input[1]);
printf("[CPU-200] tmp: %.6f, %.6f\n", tmp[0], tmp[1]);
printf("[CPU-300] output: %.6f, %.6f\n", output[0], output[1]);
```

### 4. Verify Input Data First

Always confirm inputs are correct before debugging computation.

Checklist:

- Input values match CPU golden
- No NaN/Inf in inputs
- Data alignment correct
- Shape/size match expectations

### 5. Segment Verification

Verify each stage independently:

1. **CopyIn** → If wrong, fix DataCopy/DataCopyPad
2. **Compute** → If wrong, debug computation logic
3. **CopyOut** → If wrong, fix output stage

### 6. Analyze Error Patterns

See [ascendc-dumptensor-refs/error-patterns.md](ascendc-dumptensor-refs/error-patterns.md) for common patterns and root causes.

### 7. Apply Fix and Verify

Re-run with DumpTensor to confirm fix works.

## References

- [API Reference](ascendc-dumptensor-refs/api-reference.md) - DumpTensor API details and best practices
- [Error Patterns](ascendc-dumptensor-refs/error-patterns.md) - Common error patterns and root causes

---

## 使用陷阱 ⭐⭐⭐

DumpTensor 本身也受 AscendC 流水线规则约束，用错会读到错的数据从而误导诊断。

### 1. 必须在 DeQue 之后 Dump，不要在 AllocTensor 之后

```cpp
// ❌ 错误：搬运未完成就 Dump，读到的是上次残留 / 未初始化值
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
DumpTensor(x, 100, 32);   // 读到的不是 gm 的值
inQueue.EnQue(x);

// ✅ 正确：DeQue 等搬运完成后再 Dump
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();
DumpTensor(xIn, 100, 32);  // 此时数据已就绪
```

如果不方便加 EnQue/DeQue（例如临时插桩），用 `PipeBarrier<PIPE_ALL>()` 兜底——确认结果正确后再决定是补同步还是确实没问题。

### 2. 多核场景必须把 blockIdx 编进 desc

多核并发时所有核的 dump 都会写到同一份日志，无 blockIdx 编号会导致结果完全无法区分。

```cpp
// ❌ 错误：只看 desc 无法分辨是哪个核
DumpTensor(inputLocal, 100, 32);

// ✅ 正确：把 blockIdx 编入 desc
uint32_t desc = 100 + GetBlockIdx() * 1000;   // core 0: 100, core 1: 1100, ...
DumpTensor(inputLocal, desc, 32);
```

调试单核问题时，用 `if (GetBlockIdx() == 0)` 把 dump 限制到 0 号核，避免日志爆炸。

### 3. dumpSize 控制

- 默认 32 元素够诊断模式（看头几个值就能判断 NaN / 全零 / 偏移）
- 大 tensor 完整 dump 会瞬间填满日志缓冲区，并且影响 kernel 时序，掩盖原本的 bug
- `dumpSize` 不要超过 tensor 实际长度，否则越界

```cpp
uint32_t dumpSize = std::min(tileLength, 32u);
DumpTensor(outputLocal, 300, dumpSize);
```

### 4. 调试完成必须移除

DumpTensor 引入显著时序开销，可能改变流水线行为，定位完后必须删除或宏开关包起来：

```cpp
#ifdef DEBUG_DUMP
DumpTensor(outputLocal, 300, 32);
#endif
```

---

## 输出读法

输出形如（实际格式以 CANN 版本为准）：
```
[DumpTensor] block_idx=0 desc=100 size=32 dtype=float32
  0.123, 0.456, -0.789, ...
```

实战流程：
```bash
# 跑用例
./run_op > dump.log 2>&1

# 按 desc 抽取某阶段
grep "desc=100" dump.log    # 输入
grep "desc=200" dump.log    # 中间
grep "desc=300" dump.log    # 输出

# 多核场景按核分离
grep "block_idx=0" dump.log
```

把 NPU 输出与 CPU golden 同 desc 编号对齐输出，逐段比对快速定位异常段。

---

## 调试方法选择

DumpTensor 不是唯一选择，根据场景挑：

| 方法           | 场景                              | 优点                  | 局限                    |
|----------------|-----------------------------------|----------------------|-------------------------|
| **DumpTensor** | NPU 模式，看 LocalTensor 数据     | 直接看 UB 实际值     | 时序开销大，需流水线同步 |
| `PRINTF`       | NPU 模式，看 scalar / 控制流      | 轻量                 | 不能直接 dump 整段 tensor |
| `printf`       | CPU 仿真模式                      | 与普通 C++ 调试一致   | 不能验证 NPU 实际行为    |
| 二分调试       | 已知有 bug 但插桩定位失败         | 必然能收敛           | 慢、需多次编译运行       |

衔接父 skill 的「调试策略层级」：
- 先用 DumpTensor 7步法（≤7 次尝试），定位失败立刻切二分调试
- DumpTensor 看到数据异常后，配合父 skill「症状-原因速查表」对症下药
- 改完代码记得清 `build/` 和 `$HOME/atc_data/kernel_cache/`，否则 dump 会和上次一致让人误以为没生效

---

## 检查清单

插桩前：
- [ ] 已在 DeQue 后插桩，没有在 AllocTensor 后直接 dump
- [ ] 多核场景已把 blockIdx 编进 desc
- [ ] dumpSize ≤ 32（除非确认需要更多）
- [ ] CPU golden 已用相同 desc 输出
- [ ] 编译缓存已清

定位后：
- [ ] 已识别异常出现的最早阶段（CopyIn / Compute / CopyOut）
- [ ] 已对照 [error-patterns.md](ascendc-dumptensor-refs/error-patterns.md) 匹配根因
- [ ] 修复后 dump 验证通过，已移除/宏关闭 DumpTensor
