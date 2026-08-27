# Ascend C 高性能编程规范

<适用>
语言: C++
侧别: All
领域: true
触发: AscendC::, pipe.InitBuffer, DataCopy, EnQue, DeQue, PipeBarrier, InitBuffer
默认启用: true
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

> **适用场景**：Tillng侧 和 Kernel侧

---

## 检视前置要求

> **重要**：Agent 在检视本条例时，必须先使用 `ascendc-docs-search` skill 获取相关 API 的最新文档，确认 API 参数、限制、对齐要求等信息是否与条例描述一致。若文档与条例有差异，以最新 API 文档为准，并记录差异供后续更新条例。

### 需要查阅文档的 API

| 条例编号 | 涉及 API | 查阅重点 |
|---------|---------|---------|
| PERF-1 | 无特定 API | 确认批量操作 API 的性能建议 |
| PERF-3 | `InitBuffer` | 确认 Double Buffer 配置方式 |
| PERF-4 | `PipeBarrier`, `EnQue/DeQue` | 确认同步机制和硬件保序规则 |
| PERF-5 | `DataCopy` | 确认最小搬运量建议 |
| PREC-1 | `DataCopy`, `EnQue/DeQue` | 确认异步 DMA 同步要求 |
| PREC-3 | `Cast` | 确认 FP16/FP32 转换精度建议 |
| TIL-2 | `InitBuffer`, `InitL1Buffer`, `LoadData` | 确认各芯片 UB/L1/L0 容量 |

---

## 快速索引

### 性能优化规范（PERF-*）

| 规范编号 | 规范名称 | 严重级别 |
|---------|---------|---------|
| PERF-1 | 循环内禁止逐元素操作 | 高 |
| PERF-2 | 禁止写死硬件参数 | 高 |
| PERF-3 | Double Buffer 使用 | 中 |
| PERF-4 | PipeBarrier 使用 | 低 |
| PERF-5 | 单次搬运量优化 | 中 |
| PERF-6 | 避免 GM 重复读取 | 中 |
| PERF-7 | 尾块处理正确性 | 高 |

### 精度规范（PREC-*）

| 规范编号 | 规范名称 | 严重级别 |
|---------|---------|---------|
| PREC-1 | 流水线同步正确性 | 高 |
| PREC-2 | 除零保护（详见 cpp-secure.md 2.3） | 高 |
| PREC-3 | FP16 中间精度保护 | 中 |

### Tiling 设计规范（TIL-*）

| 规范编号 | 规范名称 | 严重级别 |
|---------|---------|---------|
| TIL-1 | 多核负载均衡 | 中 |
| TIL-2 | 片上缓存容量不溢出（UB/L1/L0A/L0B/L0C） | 高 |
| TIL-3 | Buffer 规划合理性 | 中 |

---

## 性能优化规范

### PERF-1: 循环内禁止逐元素操作

**严重级别**：高

### 问题描述

在循环内调用 Ascend C API 或进行逐元素操作，会导致严重的性能下降。每次循环迭代都有 API 调用开销，应该使用批量操作替代。

### 错误示例

```cpp
// ❌ 错误：循环内逐行调用 API
for (uint32_t r = 0; r < R; r++) {
    AscendC::Sub<float>(xLocal[r * alignedCols], xLocal[r * alignedCols],
                        tmpLocal, cols);  // 每次循环都有 API 开销
}
```

### 正确示例

```cpp
// ✅ 正确：使用批量操作
AscendC::Sub<float>(xLocal, xLocal, tmpLocal, totalSize);  // 一次性处理所有数据
```

### 性能影响

循环内逐元素操作可能导致性能下降 30% 以上。

### 检视方法

```bash
grep -A5 "for.*{" <file> | grep "GetValue\|SetValue"
```

---

### PERF-2: 禁止写死硬件参数

**严重级别**：高

### 问题描述

硬件参数（核数、UB 大小等）应该动态获取，写死会导致代码不可移植，无法适应不同硬件平台。

### 错误示例

```cpp
// ❌ 错误：写死核数
uint32_t blockDim = 24;  // Ascend910B 有 24 个 AI Core

// ❌ 错误：写死 UB 大小
constexpr uint32_t UB_SIZE = 192 * 1024;  // 可能不适用于其他芯片
```

### 正确示例

```cpp
// ✅ 正确：动态获取硬件参数
uint32_t blockDim = GetBlockDim();  // 从 TilingData 获取

// ✅ 正确：使用 API 获取 UB 大小
// 或通过 TilingData 传递，由 Host 侧根据芯片类型计算
```

### 检视方法

```bash
grep -n "blockDim\s*=\s*[0-9]" <file>
grep -n "核数\s*=" <file>
```

---

### PERF-3: Double Buffer 使用

**严重级别**：中

### 问题描述

对于大数据量的算子，使用 Double Buffer 可以实现搬运和计算的重叠，提高性能。

### Double Buffer 原理

```
时间 →
Buffer 0: [MTE2搬运]──[Vector计算]──[MTE3写出]
Buffer 1:        [MTE2搬运]──[Vector计算]──[MTE3写出]
                  ↑ 搬运和计算并行！
```

### 使用方式

```cpp
// ✅ 正确：使用 Double Buffer
constexpr uint32_t BUFFER_NUM = 2;  // 双缓冲
pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(T));
pipe.InitBuffer(outQueue, BUFFER_NUM, tileLength * sizeof(T));
```

### 适用场景

- 数据量较大（> 8K 元素）
- 搬运和计算时间相当
- UB 容量允许双缓冲

### 检视方法

检查 `BUFFER_NUM` 是否 >= 2，以及是否使用 TQue 队列机制。

---

### PERF-4: PipeBarrier 使用

**严重级别**：低

### 问题描述

PipeBarrier 用于跨 Pipe 数据依赖同步。同一 Pipe 内的操作由硬件保序。

两种情况需区分对待：

1. **同 Pipe 内冗余 Barrier**（如连续 Vector 操作间加 `PipeBarrier<PIPE_V>()`）：虽冗余但不影响正确性，属于防御性编程风格，**不告警**。
2. **粒度过粗的 `PipeBarrier<PIPE_ALL>()`**：本可用更细粒度（如 `PIPE_V`/`PIPE_MTE2`）却用了 `PIPE_ALL`，会不必要地阻塞所有 Pipe 的执行，打断流水、影响性能。**应告警 SUSPICIOUS（低）**。

> **说明**：业务代码中倾向冗余使用 PipeBarrier 以确保安全性，这是可接受的工程实践——但仅限同 Pipe 粒度的冗余（如 `PIPE_V`）。用 `PIPE_ALL` 替代细粒度 Barrier 不在此列，属于性能缺陷。

### 需要 Barrier 的场景

| 场景 | 依赖类型 | 是否需要 Barrier |
|------|---------|-----------------|
| DataCopy(GM→UB) 后 Vector 计算 | MTE2 → V | ✅ 需要（通过 EnQue/DeQue） |
| Vector 计算后 DataCopy(UB→GM) | V → MTE3 | ✅ 需要（通过 EnQue/DeQue） |
| 连续 Vector 操作 | V → V | ⚠️ 硬件保序，冗余但无害（防御性编程） |
| Reduce 后读标量 | V → Scalar | ✅ 需要 |

### 应告警的场景（粒度过粗的 PIPE_ALL）

```cpp
// ❌ SUSPICIOUS 低：同一 PIPE_V 内操作只需 PIPE_V，PIPE_ALL 会阻塞所有 Pipe
AscendC::Adds<T>(xLocal, xLocal, 1.0f, count);
AscendC::PipeBarrier<PIPE_ALL>();  // 应改为 PipeBarrier<PIPE_V>()
AscendC::Mul<T>(yLocal, xLocal, val, count);
```

### 可接受的防御性写法

```cpp
// ✅ 可接受：冗余 PipeBarrier 不影响正确性
AscendC::Adds<T>(xLocal, xLocal, 1.0f, count);
AscendC::PipeBarrier<PIPE_V>();  // 同 Pipe 内冗余，但无害
AscendC::Mul<T>(yLocal, xLocal, val, count);
AscendC::PipeBarrier<PIPE_V>();  // 同上
AscendC::Exp<T>(yLocal, yLocal, count);
```

### 正确示例

```cpp
// ✅ 正确：使用 EnQue/DeQue 提供同步点
AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
inQueueX.EnQue(xLocal);  // 标记就绪

auto xIn = inQueueX.DeQue<T>();  // 阻塞等待 MTE2 完成
AscendC::Adds<T>(yLocal, xIn, 1.0f, count);  // V 操作
AscendC::Mul<T>(yLocal, yLocal, val, count);  // V 操作，无需 Barrier
AscendC::Exp<T>(yLocal, yLocal, count);  // V 操作，无需 Barrier
outQueueY.EnQue(yLocal);
```

### 检视方法

- **粒度过粗的 `PipeBarrier<PIPE_ALL>()`**（本可用 `PIPE_V`/`PIPE_MTE2` 等细粒度）：标记 **SUSPICIOUS（低）**，建议改为细粒度 Barrier。
- **同 Pipe 内冗余 Barrier**（如连续 Vector 间的 `PipeBarrier<PIPE_V>()`）：防御性编程，**不告警**。

判定要点：检查 `PipeBarrier<PIPE_ALL>()` 前后的操作是否都在同一 Pipe 内（如连续的 Vector 计算）。若是，则 PIPE_ALL 粒度过粗，应告警。

### 真实检视案例（均被人工采纳）

案例1 — ops-transformer PR#0309，`op_kernel/rope_with_sin_cos_cache_f_bf16.h:631`：

> 评论：「建议此处不要使用 PipeBarrier<PIPE_ALL>()，应该使用 V 等 MTE2 同步指令。否则会打断其它流水，影响执行效率。」

```cpp
 629 |     DataCopy(inQQueBeforeCastLocal, query_in_GM[query_in_offset],
 630 |              {1, static_cast<uint16_t>(loopNQhead * headBlockLen / 2), 0, 0});
 631 |     PipeBarrier<PIPE_ALL>();   // ❌ SUSPICIOUS 低：此处为 local tensor 间复制，只需 PIPE_V/PIPE_MTE2
 632 |     DataCopy(inQueCalLocal, inQQueBeforeCastLocal,
 633 |              {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(rotaryBlockLen / 2),
 634 |               static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2), 0});
 635 |     PipeBarrier<PIPE_ALL>();   // ❌ 同上
```

案例2 — ops-math PR#3019，`op_kernel/*.h:268`：

> 评论：「PipeBarrier<PIPE_ALL> 同步冗余，会影响流水」

```cpp
 267 |     DataCopyPad(castLocal, gradOutputGM[gradGmBase + srcOffset], gParams, {false, 0, 0, 0});
 268 |     PipeBarrier<PIPE_ALL>();   // ❌ SUSPICIOUS 低：DataCopyPad 后接 Cast，只需 MTE2→V 同步
 269 |     int64_t castCount = ((batchLen + 7) / 8) * 8;
 270 |     if (castCount > castCapacity) castCount = castCapacity;
 271 |     Cast(gradChunk[dstOffset], castLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(castCount));
 272 |     PipeBarrier<PIPE_ALL>();   // ❌ 同上
```

---

### PERF-5: 单次搬运量优化

**严重级别**：中

### 问题描述

单次 DataCopy 的搬运量太小会降低 DMA 效率。建议单次搬运量 >= 16KB。

### 优化建议

```cpp
// ✅ 正确：增大 tile 大小，减少搬运次数
constexpr uint32_t TILE_SIZE = 16 * 1024 / sizeof(float);  // >= 16KB

// 计算合理的 tile 大小（Tiling 侧，Host 代码，std:: 可用）
uint32_t tileSize = availableUB / 3 < totalLength / blockDim ?
                    availableUB / 3 : totalLength / blockDim;
if (tileSize < (uint32_t)(16 * 1024 / sizeof(float))) {
    tileSize = 16 * 1024 / sizeof(float);
}
```

> **注意**：上述三元运算替代 `std::min/max`，因为 Kernel 侧禁止使用标准库（见 API-2）。若此计算在 Host/Tiling 侧则可直接使用 `std::min/max`。

### 检视方法

检查 Tiling 设计中的 tileLength 是否 >= 16KB。

---

### PERF-6: 避免 GM 重复读取

**严重级别**：中

### 问题描述

GM 带宽有限，重复读取同一块 GM 数据会严重影响性能。应该在 UB 中缓存数据，避免重复搬运。

### 错误示例

```cpp
// ❌ 错误：多次读取同一块 GM 数据
for (uint32_t i = 0; i < iterations; i++) {
    AscendC::DataCopyPad(xLocal, xGm, ...);  // 每次都从 GM 读取
    // ... 计算
}
```

### 正确示例

```cpp
// ✅ 正确：一次读取，多次使用
AscendC::DataCopyPad(xLocal, xGm, ...);  // 只读取一次
for (uint32_t i = 0; i < iterations; i++) {
    // ... 使用 xLocal 中的缓存数据
}
```

### 检视方法

检查代码中是否存在对同一 GM 地址的多次 DataCopy。

---

### PERF-7: 尾块处理正确性

**严重级别**：高

### 问题描述

批量循环处理数据时，总量往往不能被块大小整除。最后一个块（尾块）的实际大小小于满块大小，若直接使用满块大小计算偏移或搬运长度，会导致越界读取 GM 数据，产生静默的精度错误。

此问题在 Kernel 侧表现为 DataCopy/DataCopyPad 的 `blockLen` 参数直接使用了满块大小，而非 `actualSize`。

### 常见错误模式

```cpp
// ❌ 错误：所有块（含尾块）都用满块大小搬运，尾块越界读 GM
for (uint32_t k = 0; k < totalK; k += BLOCK_SIZE) {
    DataCopyPad(dst, src[k], BLOCK_SIZE);   // 尾块时 k + BLOCK_SIZE > totalK！
}
```

### 正确示例

```cpp
// ✅ 方式1：循环内动态计算尾块大小
for (uint32_t k = 0; k < totalK; k += BLOCK_SIZE) {
    uint32_t actualSize = (k + BLOCK_SIZE <= totalK) ? BLOCK_SIZE : (totalK - k);
    DataCopyPad(dst, src[k], actualSize);
}

// ✅ 方式2：先计算 loops，循环末尾单独处理尾块（FIA 实际用法）
uint32_t kLoops = (totalK + BLOCK_SIZE - 1U) / BLOCK_SIZE;  // 向上取整
uint32_t kBlockSize = BLOCK_SIZE;
for (uint32_t i = 0; i < kLoops; i++) {
    if (i == kLoops - 1U) {
        kBlockSize = totalK - (kLoops - 1U) * BLOCK_SIZE;  // 尾块实际大小
    }
    DataCopyPad(dst, src[i * BLOCK_SIZE], kBlockSize);
}
```

> **注意（方式2）**：`kBlockSize` 必须在循环开始前用满块大小初始化，用于 `kLoops` 的计算。如果 `kLoops` 在循环体内计算，而循环体又会修改 `kBlockSize`，则会产生循环依赖错误。

### 受影响的关键参数

| 参数 | 满块值 | 尾块处理 |
|------|-------|---------|
| DataCopy/DataCopyPad 的搬运长度 | fullBlockLen | `actualLen = totalLen - loopIdx * fullBlockLen` |
| DataCopyExtParams.blockLen | headDim * sizeof(T) | `actualColumnCount * sizeof(T)` |
| DataCopyExtParams.srcStride | - | 同步更新为 `(totalColumns - actualColumns) * sizeof(T)` |

### 检视方法

```bash
grep -n "DataCopy\|DataCopyPad" <file>
grep -n "blockLen\s*=" <file>
```

**检视规则**：
- `blockLen` 的来源是否为 `actualXxx`（动态计算）而非写死的常量
- 调用函数同时处理满块和尾块时，调用方是否在尾块时传入了正确的 `actualLen`

---

## 精度规范

### PREC-1: 流水线同步正确性

**严重级别**：高

### 问题描述

DataCopy 是异步 DMA 操作，如果缺少同步，Vector 计算可能读到未完成搬运的数据，导致输出错误（如全 0、随机值）。

### 核心原则

**必须在 DataCopy 后使用 EnQue/DeQue 或 PipeBarrier 进行同步。**

### 核心原理

TQue 的 EnQue/DeQue 机制提供硬件同步点：

```
EnQue(xLocal)  → 标记 buffer 数据就绪
DeQue<float>() → 阻塞等待数据就绪
```

### 错误示例

```cpp
// ❌ 错误：DataCopy 后直接使用数据
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);  // 可能读到未完成搬运的数据！
```

### 正确示例

```cpp
// ✅ 正确：使用 EnQue/DeQue 同步的完整三步流程
// Step 1: CopyIn - MTE2 搬运
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);
inQueueX.EnQue(xLocal);                    // 标记"就绪"

// Step 2: Compute - Vector 计算
AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // 阻塞等待 MTE2 完成
AscendC::Adds<float>(yLocal, xIn, 1.0f, count);
outQueueY.EnQue(yLocal);
inQueueX.FreeTensor(xIn);

// Step 3: CopyOut - MTE3 搬运
AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // 阻塞等待 Vector 完成
AscendC::DataCopyPad(yGm[gmOffset], yOut, copyParams);
outQueueY.FreeTensor(yOut);
```

### 典型症状

- 输出数据随机、错误
- 输出全 0
- 某些核的数据错误

### 检视方法

```bash
# 检查 DataCopy 后是否有 EnQue/DeQue
grep -A3 "DataCopy" <file> | grep -c "EnQue\|DeQue"

# 检查 EnQue/DeQue 配对数量
grep -c "EnQue" <file>
grep -c "DeQue" <file>
# 两者数量应该相等
```

---

### PREC-2: 除零保护

> **详见 `cpp-secure.md` 条例 2.3**
>
> 此规范已在安全编码规范中详细定义，包括 Tiling 侧校验示例和 Kernel 侧校验示例（含排除规则）。

---

### PREC-3: FP16 中间精度保护

**严重级别**：中

### 问题描述

FP16 精度有限（约 3 位有效数字），在累加、归约等操作中容易产生精度损失。关键计算应使用 FP32 中间结果。

### 错误示例

```cpp
// ❌ 错误：FP16 直接累加
half sum = 0;
for (int i = 0; i < N; i++) {
    sum += data[i];  // FP16 累加，精度损失严重
}
```

### 正确示例

```cpp
// ✅ 正确：使用 FP32 累加
float sum = 0;
for (int i = 0; i < N; i++) {
    sum += (float)data[i];  // FP32 累加
}
half result = (half)sum;  // 最后转回 FP16
```

### 常见场景

- Reduce 操作：使用 FP32 累加器
- Softmax：Exp 使用 FP32 计算
- LayerNorm：方差计算使用 FP32

### 检视方法

检查 FP16 数据类型的累加、归约操作是否使用 FP32 中间结果。

---

## Tiling 设计规范

### TIL-1: 多核负载均衡

**严重级别**：中

### 问题描述

Tiling 设计应该保证各个 AI Core 的负载均衡，避免某些核空闲等待。

### 设计原则

1. **数据均匀切分**：每个核处理的数据量应该相近
2. **处理最后一包**：正确处理不能整除的情况
3. **空闲核跳过**：核数超过数据量时，空闲核应该正确跳过

### 示例

```cpp
// ✅ 正确：均匀切分 + 处理余数
uint32_t totalLength = ...;
uint32_t blockDim = GetBlockDim();
uint32_t avgLength = totalLength / blockDim;
uint32_t remainder = totalLength % blockDim;

uint32_t tileLength = avgLength + (blockIdx < remainder ? 1 : 0);
uint32_t offset = blockIdx * avgLength + std::min(blockIdx, remainder);

if (offset < totalLength) {
    // 处理数据
}
```

### 检视方法

分析 TilingData 的切分逻辑，确认各核负载是否均衡。

---

### TIL-2: 片上缓存容量不溢出（UB/L1/L0A/L0B/L0C）

**严重级别**：高

### 问题描述

AI Core 上有多级片上缓存，各自容量固定。Tiling 时若任一缓存规划超限，轻则编译失败，重则运行时崩溃。

| 缓存 | 所属核 | TPosition 标识 | 典型容量 | 用途 |
|------|-------|---------------|---------|------|
| **UB** | Vector Core | `VECCALC` | 256 KB | 向量计算缓冲，`InitBuffer` 分配 |
| **L1** | Cube Core | `TSCM` | 1 MB | GM→L1 数据缓存，减少 GM 访问延迟 |
| **L0A** | Cube Core | `A2` | 64 KB | 矩阵左输入（`LoadData` L1→L0A）|
| **L0B** | Cube Core | `B2` | 64 KB | 矩阵右输入（`LoadData` L1→L0B）|
| **L0C** | Cube Core | `CO1` | 256 KB | 矩阵乘输出结果存储 |

> 容量来源：`gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_base.h`

### 适用场景

- **纯向量算子**：只需关注 UB，所有 `InitBuffer` 分配之和不超过 UB 容量
- **矩阵/Attention 算子**：还需关注 L1、L0A/L0B/L0C，各级缓存独立计算，不互相共用

### Buffer 声明方式（真实业务代码）

```cpp
// 来源：gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_base.h
TBuf<TPosition::VECCALC> UbBuf;   // UB — 向量计算缓冲
TBuf<TPosition::TSCM>    L1Buf;   // L1 — Cube 输入缓存
TBuf<TPosition::A2>      L0ABuf;  // L0A — 矩阵左输入
TBuf<TPosition::B2>      L0BBuf;  // L0B — 矩阵右输入
TBuf<TPosition::CO1>     L0CBuf;  // L0C — 矩阵乘输出
```

### 错误示例（UB 超限）

```cpp
// ❌ 错误：UB 分配总量超过 256KB
// nUbFactor=64, k=1024, hUbFactor=4, sizeof(float)=4
// inputQueue:  2 * 64*1024*4 * 4 = 2MB → 超出 UB
tPipe->InitBuffer(inputQueue,  DOUBLE_BUFFER, nUbFactor * k * hUbFactor * sizeof(float));
tPipe->InitBuffer(outputQueue, 1,             nUbFactor * hUbFactor * sizeof(float));
tPipe->InitBuffer(binaryAddBuf,               nUbFactor * hUbFactor * sizeof(float));
// 三个 Buffer 总量远超 UB_SIZE(256KB)，运行时崩溃
```

### 正确示例（UB）

```cpp
// ✅ 正确：Tiling 阶段保证 nUbFactor * kUbFactor * hUbFactor 不超 UB 容量
// 来源：moe/moe_init_routing_v2_grad/op_kernel/arch35/moe_init_routing_v2_grad.h
tPipe->InitBuffer(inputQueue,  DOUBLE_BUFFER, td_->nUbFactor * td_->kUbFactor * td_->hUbFactor * sizeof(T));
tPipe->InitBuffer(outputQueue, 1,             td_->nUbFactor * td_->hUbFactor * sizeof(T));
tPipe->InitBuffer(binaryAddBuf,               td_->nUbFactor * td_->hUbFactor * sizeof(float));
// Tiling 中已保证三项之和 < UB_SIZE
```

### 正确示例（L1）

```cpp
// ✅ 正确：L1 Buffer 分配总量不超过 L1_SIZE(1MB)
// 来源：attention/prompt_flash_attention/op_kernel/arch32/mla_custom_matmul_policy_common.h
constexpr uint32_t Q_VEC1_BUFFER_SIZE = 128 * 512 * 2;   // 128KB
constexpr uint32_t K_V_BUFFER_SIZE    = 192 * 1024 * 2;  // 384KB
// 总计 512KB < L1_SIZE(1MB) ✅

__aicore__ inline void InitL1Buffer(TPipe *tPipe, GlobalL1Array *l1Global)
{
    if (g_coreType == AIC) {
        tPipe->InitBuffer(l1Global[Q_VEC1_INDEX].localQue, 1, Q_VEC1_BUFFER_SIZE);
        tPipe->InitBuffer(l1Global[K_V_INDEX].localQue,    1, K_V_BUFFER_SIZE);
    }
}
```

### 检视方法

```bash
# UB Buffer 分配（向量算子重点）
grep -n "InitBuffer" <file>

# L1 Buffer 分配（矩阵/Attention 算子重点）
grep -n "InitL1Buffer" <file>

# 各级缓存声明（快速确认涉及哪些缓存层级）
grep -n "TBuf<TPosition" <file>
```

**检视规则**：
- 向量算子：汇总所有 `InitBuffer` 的 `bufferNum × bufferSize`，确认总量 < UB 容量
- 矩阵算子：在向量算子基础上，额外检查 `InitL1Buffer` 中 L1 分配总量 < 1MB

---

### TIL-3: Buffer 规划合理性

**严重级别**：中

### 问题描述

合理规划 UB 中的 Buffer 数量和大小，避免资源浪费或不足。

### 规划原则

1. **输入 Buffer**：通常需要 2 个（Double Buffer）
2. **输出 Buffer**：通常需要 2 个（Double Buffer）
3. **临时 Buffer**：根据计算需求分配
4. **Reduce 临时 Buffer**：Reduce 操作需要额外的临时 Buffer

### 示例

```cpp
// ✅ 正确：合理的 Buffer 规划
pipe.InitBuffer(inQueueX, 2, tileLength * sizeof(T));    // 输入 x
pipe.InitBuffer(inQueueY, 2, tileLength * sizeof(T));    // 输入 y
pipe.InitBuffer(outQueue, 2, tileLength * sizeof(T));    // 输出
pipe.InitBuffer(tmpQueue, 1, tileLength * sizeof(T));    // 临时
pipe.InitBuffer(reduceQueue, 1, 32 * sizeof(float));     // Reduce 临时
```

### 检视方法

分析 Buffer 数量和大小是否与计算需求匹配。

---

## 检视检查清单

使用以下清单快速检查代码是否满足高性能编程规范：

### 性能优化
- [ ] **PERF-1**: 循环内是否有逐元素 API 调用？
- [ ] **PERF-2**: 是否写死了核数或 UB 大小？
- [ ] **PERF-3**: 大数据量是否使用 Double Buffer？
- [ ] **PERF-4**: PipeBarrier 粒度过粗？（PIPE_ALL 替代 PIPE_V）冗余 Barrier 属于防御性编程，不标记
- [ ] **PERF-5**: 单次搬运量是否 >= 16KB？（示例代码注意 Kernel 侧禁用 std::）
- [ ] **PERF-6**: 是否有 GM 重复读取？
- [ ] **PERF-7**: 循环处理块时，尾块的搬运长度是否单独计算？

### 精度
- [ ] **PREC-1**: DataCopy 后是否有同步？
- [ ] **PREC-2**: 除法/求余是否有除零保护？（详见 cpp-secure.md 2.3）
- [ ] **PREC-3**: FP16 累加是否使用 FP32 中间结果？

### Tiling 设计
- [ ] **TIL-1**: 各核负载是否均衡？
- [ ] **TIL-2**: 各级片上缓存（UB/L1/L0A/L0B/L0C）分配总量是否超出容量限制？
- [ ] **TIL-3**: Buffer 规划是否合理？
