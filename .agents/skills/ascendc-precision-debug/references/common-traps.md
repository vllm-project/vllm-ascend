# 常见精度陷阱详解

## 陷阱1：FP16 精度不足

### 症状
- 简单计算也有明显误差
- FP16 误差明显大于 FP32
- 多次累加后误差累积

### 原因
FP16 只有约 3-4 位有效数字（FP32 约 6-7 位），在需要高精度的计算中容易丢失精度。

### 解决方案

**混合精度设计原则**：
- 输入/输出保持 FP16（节省带宽和存储）
- 关键中间计算用 FP32（提升精度）
- 累加类操作优先用 FP32

```cpp
// 关键中间值用 FP32
float sum_fp32 = 0.0f;
for (int i = 0; i < n; ++i) {
    sum_fp32 += static_cast<float>(values[i]);  // 先转为 FP32 累加
}
output = static_cast<half>(sum_fp32);  // 最后转回 FP16
```

**典型应用**：Reduce、Sum、Mean、Softmax 等

### Plan 阶段预防
在制定算子开发 Plan 时，主动询问：
> "该算子对精度要求如何？是否需要在中间计算过程中使用 FP32 提升精度？"

---

## 陷阱2：exp/log 溢出

### 症状
- 输出出现 Inf（无穷大）
- 输出出现 NaN（非数值）
- 大输入值结果异常

### 原因
- exp(x) 在 x > 88 时溢出（FP16）
- log(x) 在 x ≤ 0 时无定义

### 解决方案：数值稳定的 Softmax

```cpp
// 先减去最大值，再 exp，避免溢出
half max_val = input[0];
for (int i = 1; i < size; ++i) {
    max_val = max(max_val, input[i]);
}

half exp_sum = 0.0h;
for (int i = 0; i < size; ++i) {
    half shifted = input[i] - max_val;  // 关键！使最大输入为0
    half exp_val = Exp(shifted);
    exp_sum += exp_val;
    output[i] = exp_val;
}

for (int i = 0; i < size; ++i) {
    output[i] = output[i] / exp_sum;
}
```

**数值稳定原理**：
- 减去最大值后，最大输入变为 0，exp(0) = 1
- 其他输入变为负数，exp(负数) < 1
- 避免了 exp(大正数) 溢出

### 其他数值稳定技巧

```cpp
// 稳定的 log-sum-exp（用于 log-softmax）
half max_val = ReduceMax(input);
half sum_exp = 0.0h;
for (int i = 0; i < size; ++i) {
    sum_exp += Exp(input[i] - max_val);
}
output = max_val + Log(sum_exp);  // 数值稳定的 log(sum(exp(x)))

// 稳定的 sigmoid
half sigmoid = 1.0h / (1.0h + Exp(-x));

// 避免 exp(x) 过大的替代方案
// 如果知道 x 的范围，可以预先截断
half safe_exp = Exp(min(x, 10.0h));  // 限制最大指数为 10
```

---

## 陷阱3：减法抵消（Catastrophic Cancellation）

### 症状
- 两个接近的数相减时，误差突然变大
- a ≈ b 时，a - b 结果不准确

### 原因
当两个接近的数相减时，有效数字会大量丢失，导致相对误差增大。

### 示例
```
FP16: 1.234 - 1.233 = 0.001（可能只有 1 位有效数字）
```

### 解决方案

**方法1：重排计算公式**
```cpp
// 原始公式（数值不稳定）
half result = sqrt(x + 1) - sqrt(x);

// 稳定版本（有理化）
half result = 1.0h / (sqrt(x + 1) + sqrt(x));
```

**方法2：提升中间精度**
```cpp
// 使用 FP32 进行减法运算
float diff_fp32 = static_cast<float>(a) - static_cast<float>(b);
half result = static_cast<half>(diff_fp32);
```

**方法3：使用数学等价变换**
```cpp
// 例如：计算 1 - cos(x)
// 不稳定：1 - cos(x)
// 稳定：2 * sin(x/2)^2
```

---

## 陷阱4：Reduce 操作精度损失

### 症状
- Reduce 结果误差比逐元素操作大
- Sum/Mean 等聚合操作精度不足

### 原因
Reduce 操作涉及多次累加，FP16 精度不足导致误差累积。

### 解决方案

```cpp
// 使用 FP32 累加器
float sum_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    sum_fp32 += static_cast<float>(input[i]);
}
output = static_cast<half>(sum_fp32);

// ReduceMax/ReduceMin 不受影响，可以保持 FP16
half max_val = ReduceMax(input);  // FP16 足够

// ReduceSum/ReduceMean 建议使用 FP32
float mean_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    mean_fp32 += static_cast<float>(input[i]);
}
mean_fp32 /= static_cast<float>(size);
```

---

## 陷阱5：除零风险

### 症状
- 输出出现 NaN
- 输出出现异常大的数值

### 解决方案

```cpp
// 方法1：添加小常数（Epsilon）
half eps = 1e-7h;
half safe_div = numerator / (denominator + eps);

// 方法2：条件判断
half eps = 1e-7h;
half safe_div = (abs(denominator) < eps) ? 0.0h : numerator / denominator;

// 方法3：使用最大值保护
half safe_div = numerator / max(denominator, eps);
```

---

## 陷阱6：硬件约束不满足

### 症状
- 特定输入规模结果异常
- 正常规模正常，边界规模错误

### 典型案例：SoftmaxV5

**发现**：ReduceMax/ReduceSum 在列数 < 8 时计算不正确

**原因**：硬件 reduce 操作的约束

**解决方案**：
```cpp
// 添加输入验证
if (cols < 8) {
    printf("Error: cols must be >= 8 (got %d)\n", cols);
    return;
}

// 或在文档中明确说明约束
// 已知限制：列数必须 ≥ 8
```

### 常见硬件约束

| 约束类型 | 要求 | 检查方法 |
|---------|------|---------|
| Reduce 操作 | 最小元素数 ≥ 8 | 检查 reduce 维度大小 |
| 数据对齐 | 32 字节对齐 | 检查尾轴长度 |
| 单次处理上限 | 受 UB 容量限制 | 大数据需要分块 |

**32 字节对齐参考**：
| 数据类型 | 每元素字节数 | 32 字节对齐元素数 |
|---------|-------------|----------------|
| FP16 | 2 字节 | 16 个元素 |
| FP32 | 4 字节 | 8 个元素 |
| INT8 | 1 字节 | 32 个元素 |

---

## 陷阱7：类型转换精度损失

### 症状
- FP32 → FP16 转换后精度下降
- 多次转换累积误差

### 解决方案

```cpp
// 避免不必要的类型转换
// 不推荐：频繁转换
half temp = static_cast<half>(float_value);
float result = static_cast<float>(temp);

// 推荐：保持一种类型
float result = float_value;  // 尽量用 FP32 计算

// 只在必要时转换
half output = static_cast<half>(final_result_fp32);
```

---

## 陷阱8：Cast API RoundMode 使用错误 ⭐

### 症状
- Cast 后数据完全错误（不是精度问题，是数据混乱）
- 多行数据输出完全相同
- 错误与 RoundMode 选择有关，而非精度损失

### 原因
Cast API 的 `RoundMode` 参数选择错误。关键认知：

**`CAST_NONE` 的语义**：在转换有精度损失时表示 `CAST_RINT` 模式，**不涉及精度损失时表示不舍入**。

### 正确用法

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| half → float | `CAST_NONE` | 低→高精度，无精度损失，不需要舍入 |
| float → half | `CAST_ROUND` | 高→低精度，有精度损失，需要舍入 |

### 错误示例

```cpp
// ❌ 错误：half → float 使用 CAST_ROUND
AscendC::Cast<float, half>(xLocal, xLocalHalf, AscendC::RoundMode::CAST_ROUND, cols);
// 结果：数据完全错误，多行输出相同
```

### 正确示例

```cpp
// ✅ 正确：half → float 使用 CAST_NONE
AscendC::Cast<float, half>(xLocal, xLocalHalf, AscendC::RoundMode::CAST_NONE, cols);

// ✅ 正确：float → half 使用 CAST_ROUND
AscendC::Cast<half, float>(yLocalHalf, xLocal, AscendC::RoundMode::CAST_ROUND, cols);
```

### RoundMode 完整说明

```cpp
enum class RoundMode {
    CAST_NONE = 0,   // 无精度损失时不舍入，有精度损失时等同 CAST_RINT
    CAST_RINT,       // 四舍六入五成双（银行家舍入）
    CAST_FLOOR,      // 向负无穷舍入
    CAST_CEIL,       // 向正无穷舍入
    CAST_ROUND,      // 四舍五入
    CAST_TRUNC,      // 向零舍入
    CAST_ODD,        // 最近邻奇数舍入
    CAST_HYBRID,     // 随机舍入（特定场景）
};
```

### 实战案例：SoftmaxV5 FP16 混合精度

```cpp
__aicore__ inline void ComputeFp16()
{
    // Step 1: half → float（低→高精度）
    AscendC::Cast<float, half>(xLocal, xLocalHalf, AscendC::RoundMode::CAST_NONE, cols);
    
    // Step 2: 在 FP32 上进行 softmax 计算
    // ... ReduceMax, Adds, Exp, ReduceSum, Muls ...
    
    // Step 3: float → half（高→低精度）
    AscendC::Cast<half, float>(yLocalHalf, xLocal, AscendC::RoundMode::CAST_ROUND, cols);
}
```

### 预防措施

1. **使用 Cast API 前，必须查阅官方文档确认 RoundMode**
2. **低精度 → 高精度：使用 `CAST_NONE`**
3. **高精度 → 低精度：使用 `CAST_ROUND` 或其他舍入模式**

---

## 陷阱9：输出全为 0 ⭐⭐⭐

### 症状
- 输出数据全部为 0 或随机错误
- 预期有值但实际为 0

### 原因1：流水线同步问题（EnQue/DeQue 缺失）⭐⭐⭐

**核心问题**：DataCopy 是异步 DMA，立即返回。直接使用搬运后的数据可能读到未完成的旧数据。

| 模式 | 代码 | 评价 |
|------|------|------|
| ❌ 错误 | `DataCopy(x, gm, n); Compute(x);` | 缺少同步 |
| ✅ 正确 | `DataCopy → EnQue → DeQue → Compute` | 推荐 |
| ⚠️ 调试 | `DataCopy → PipeBarrier → Compute` | 验证用 |

```cpp
// ❌ 错误
LocalTensor<T> x = allocator.Alloc<T, 64>();
DataCopy(x, xGm, count);
Cast<half, int8_t>(xHalf, x, ...);  // ⛔️ 数据可能未就绪

// ✅ 正确：EnQue/DeQue
void CopyIn() {
    LocalTensor<T> x = inQueue.AllocTensor<T>();
    DataCopy(x, xGm, count);
    inQueue.EnQue(x);
}
void Compute() {
    LocalTensor<T> x = inQueue.DeQue<T>();  // 等待数据就绪
    Cast<half, int8_t>(xHalf, x, ...);
    inQueue.FreeTensor(x);
}

// ⚠️ 调试验证：临时加 PipeBarrier
DataCopy(x, xGm, count);
PipeBarrier<PIPE_ALL>();  // 若结果正确则确认是同步问题
```

### 原因2：DataCopy 非 32 字节对齐

**问题**：`DataCopy(dst, src, count)` 要求 `count * sizeof(T)` 是 32 字节对齐。

| 数据类型 | 32B 对齐元素数 |
|---------|---------------|
| FP16 | 16, 32, 48... |
| FP32/INT32 | 8, 16, 24... |
| INT8 | 32, 64, 96... |

```cpp
// ❌ 错误：8 字节非对齐
DataCopy(indicesGm, indicesLocal, 2);  // 2 * 4 = 8B

// ✅ 正确：使用 DataCopyPad
DataCopyExtParams p{1, rowsThisCore * sizeof(int32_t), 0, 0, 0};
DataCopyPad(indicesGm, indicesLocal, p);
```

### 原因3：GlobalTensor.SetValue

**问题**：GlobalTensor.SetValue 可能不生效。

```cpp
// ❌ 避免
outGm.SetValue(0, 10);

// ✅ 推荐
LocalTensor<T> tmp = buf.Get<T>();
tmp.SetValue(0, value);
DataCopyPad(dstGm, tmp, {1, sizeof(T), 0, 0, 0});
```

### 诊断流程

```
输出全为 0？
│
├─ [1] 检查流水线同步 ⭐⭐⭐
│   └─ DataCopy 后有 EnQue/DeQue？
│       └─ 否 → 添加同步（临时用 PipeBarrier 验证）
│
├─ [2] 检查数据对齐
│   └─ count * sizeof(T) 是 32 的倍数？
│       └─ 否 → 改用 DataCopyPad
│
└─ [3] 检查 GlobalTensor
    └─ 使用了 SetValue？
        └─ 是 → 改用 LocalTensor + DataCopyPad
```

### 实战案例

**Abs 算子输出全为 0**：
- 原因：DataCopy 后直接 Cast，缺少同步
- 修复：改用 EnQue/DeQue 机制

**ArgMax 40核失败**：
- 原因：每核输出 8B，非 32B 对齐
- 修复：改用 DataCopyPad

---

## 陷阱10：MX 块量化格式 Cast 公式 floor 偏移导致 NaN

### 症状

- 算子输出张量出现大量 NaN（典型 50-80%）
- NaN 集中在某些行的连续区域（对应量化时 amax 较大的行）
- 切换到 mxfp8 / mxfp4 等 MX 块量化格式后出现
- 把 scale 替换为常量（如 e8m0 = 0x7F，decoded scale = 1.0）后 NaN 减少但不消失（说明问题不在 scale 路径）

### 原因

E8M0 scale 编码使用了 floor 偏移而不是 ceil 偏移：

```cpp
// ❌ floor 偏移
e8m0_byte = biased_exp_amax - emax_quant_dtype;
```

floor 偏移下 `amax / decoded_scale` 可能落在 `[quant_dtype_max, 2 × quant_dtype_max)` 区间（对 e4m3 = `[448, 896)`）。`Cast<量化 dtype, fp32, RINT>` 对超过 dtype_max 的输入会产出 NaN（对 fp8_e4m3 为 `0x7F`）。

### 解决方案

改用 ceil 偏移：

```cpp
// ✅ ceil 偏移
e8m0_byte = (biased_exp_amax - emax_quant_dtype) + 1;
```

ceil 偏移确保 `amax / decoded_scale ≤ quant_dtype_max`，Cast 不溢出。

### 排查流程

1. 检查 NaN 是否分布在特定行 → 是 → 怀疑量化路径数值公式
2. 用 stub scale = 中性常量（0x7F = 1.0）替换原 scale 重跑
   - 若 NaN 消失 → 根因在 scale 路径（layout / 索引等）
   - 若 NaN 仍在 → 根因在 Cast 数值公式（本陷阱）
3. 在量化 Cast 前后插 `AscendC::printf` 采样
   - Cast 前 fp32 数值正常 + Cast 后出现 NaN 字节 → 确认 Cast 溢出
4. 检查 E8M0 生成代码，选用 ceil 偏移

参数对照：`emax_quant_dtype` 取值见 [api-precision.md MX 块量化格式精度路径](../../ascendc-api-best-practices/references/api-precision.md)。

---

## 陷阱排查清单

当遇到精度问题时，按顺序检查：

- [ ] **输出是否全为 0？**（检查 DataCopy 对齐 / GlobalTensor.SetValue）⭐⭐⭐
- [ ] 是否使用了 Cast API？（检查 RoundMode 是否正确） ⭐
- [ ] 是否使用了 FP16 累加器？（改用 FP32）
- [ ] 是否有 exp/log 操作？（检查溢出）
- [ ] 是否有接近数的减法？（重排公式或提升精度）
- [ ] 是否有 Reduce 操作？（使用 FP32 累加器）
- [ ] 是否有除法？（检查除零保护）
- [ ] 是否满足硬件约束？（对齐、最小元素数）
- [ ] 类型转换是否合理？（避免不必要转换）
- [ ] **MX 块量化格式 Cast 公式是否用 ceil 偏移？**（floor 偏移会让 amax/scale 超 dtype max 导致 NaN）
