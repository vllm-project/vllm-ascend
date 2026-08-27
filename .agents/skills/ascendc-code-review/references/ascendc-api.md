# Ascend C API 最佳实践规范

<适用>
语言: C++
侧别: All
领域: true
触发: AscendC::, pipe.InitBuffer, DataCopy, DataCopyPad, AllocTensor, FreeTensor, CrossCoreSetFlag, CrossCoreWaitFlag, GetValue, SetValue, aclnn, aclrt, aclop
默认启用: true

适用场景: Kernel 侧 AscendC API 最佳实践（API-1~API-12）+ 跨侧别日落（废弃）API/头文件检查（SUNSET-1/SUNSET-2）
介绍: AscendC API 最佳实践规范，含 Kernel 侧 API 黑名单/对齐/配对规则，以及 CANN 日落（废弃）API/头文件的兼容性检查（aclrt*/aclnn*/acl.op.* 及 include/so）
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

> **适用场景**：Kernel 侧（Device 侧）
>
> **Tiling 侧不适用**：Tiling 侧不使用 Ascend C API。

---

## 检视前置要求

> **重要**：Agent 在检视本条例时，必须先使用 `ascendc-docs-search` skill 获取相关 API 的最新文档，确认 API 参数、限制、对齐要求等信息是否与条例描述一致。若文档与条例有差异，以最新 API 文档为准，并记录差异供后续更新条例。

### 需要查阅文档的 API

| 条例编号 | 涉及 API | 查阅重点 |
|---------|---------|---------|
| API-1 | `GlobalTensor::SetValue/GetValue` | 确认禁止使用说明 |
| API-2 | Ascend C 计算类 API | 确认可用的替代 API 列表 |
| API-3 | `DataCopy`, `DataCopyPad` | 确认 32 字节对齐要求 |
| API-4 | `Compare` | 确认 256 字节对齐要求 |
| API-6 | `AllocTensor`, `FreeTensor` | 确认内存管理配对要求 |
| API-8 | 循环类 API | 确认 repeatTimes 类型限制 |
| API-9 | `Cast` | 确认 RoundMode 参数说明 |
| API-10 | `DataCopy`, `DataCopyPad` | 确认两套参数结构体的 blockLen 单位差异 |
| API-12 | `CrossCoreSetFlag`, `CrossCoreWaitFlag` | 确认同步配对规则和提前退出风险 |

---

## 快速索引

> 文件级侧别为 `All`，但 API-1~API-12 仅对 Kernel 侧有意义，通过条例级 `[适用: Kernel]` 标记在 plan-design Step 4 侧别过滤时对 Tiling 侧跳过。SUNSET-1/SUNSET-2 跨侧别生效（`[适用: All]`）。

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| API-1 | 禁止使用 GlobalTensor::SetValue/GetValue | API黑名单 | 高 | [适用: Kernel] |
| API-2 | 禁止使用 std:: 计算函数 | API黑名单 | 高 | [适用: Kernel] |
| API-3 | DataCopy/DataCopyPad 对齐要求 | 数据搬运 | 高 | [适用: Kernel] |
| API-4 | Compare API 256字节对齐要求 | 数据搬运 | 高 | [适用: Kernel] |
| API-6 | AllocTensor/FreeTensor 必须配对使用 | 内存管理 | 高 | [适用: Kernel] |
| API-7 | 禁止动态内存分配 | 内存管理 | 高 | [适用: Kernel] |
| API-8 | repeatTimes 限制（≤255） | API限制 | 中 | [适用: Kernel] |
| API-9 | Cast RoundMode 正确性 | 类型转换 | 中 | [适用: Kernel] |
| API-10 | DataCopyParams vs DataCopyExtParams 单位差异 | 数据搬运 | 高 | [适用: Kernel] |
| API-12 | CrossCoreSetFlag/WaitFlag 必须对称 | 核间同步 | 高 | [适用: Kernel] |
| SUNSET-1 | 禁止使用日落（废弃）API | API兼容性 | 高 | [适用: All] |
| SUNSET-2 | 禁止引用日落头文件/库 | API兼容性 | 高 | [适用: All] |

---

## API-1: 禁止使用 GlobalTensor::SetValue/GetValue

**严重级别**：高

### 问题描述

`GlobalTensor::SetValue()` 和 `GlobalTensor::GetValue()` 是逐元素操作，性能极差。在生产代码中禁止使用。

### 错误示例

```cpp
// ❌ 禁止：逐元素访问 GM
for (uint32_t i = 0; i < size; i++) {
    output.SetValue(i, input.GetValue(i));
}
```

### 正确示例

```cpp
// ✅ 正确：使用 DataCopyPad 批量搬运
AscendC::DataCopyPad(output, input, copyParams, padParams);
```

### 检视方法

```bash
grep -n "SetValue\|GetValue" <file>
```

**注意**：调试时代码中出现的 `GetValue` 用于打印调试信息是允许的。

---

## API-2: 禁止使用 std:: 计算函数

**严重级别**：高

### 问题描述

Kernel 侧不支持 C++ 标准库，必须使用 Ascend C 提供的专用 API。

### 禁止列表

| std:: 函数 | 错误用法 | Ascend C 替代 |
|-----------|---------|--------------|
| `std::abs` | `std::abs(x)` | `AscendC::Abs(dst, src, count)` |
| `std::min/max` | `std::min(a, b)` | `(a < b) ? a : b` 或 `AscendC::Min/Max` |
| `std::sqrt` | `std::sqrt(x)` | `AscendC::Sqrt(dst, src, count)` |
| `std::pow` | `std::pow(x, y)` | `AscendC::Power(dst, src, count)` |
| `std::exp` | `std::exp(x)` | `AscendC::Exp(dst, src, count)` |
| `std::log` | `std::log(x)` | `AscendC::Log(dst, src, count)` |
| `std::sin/cos` | `std::sin(x)` | `AscendC::Sin/Cos(dst, src, count)` |
| `std::floor/ceil` | `std::floor(x)` | `AscendC::Floor/Ceil(dst, src, count)` |

### 错误示例

```cpp
// ❌ 禁止：使用 std:: 函数
#include <algorithm>
#include <cmath>

uint32_t result = std::min(a, b);  // 编译错误
float val = std::sqrt(x);          // 编译错误
```

### 正确示例

```cpp
// ✅ 正确：使用 Ascend C API
uint32_t result = (a < b) ? a : b;  // min
uint32_t result = (a > b) ? a : b;  // max

// 批量操作
AscendC::Sqrt<T>(dstLocal, srcLocal, count);
AscendC::Exp<T>(dstLocal, srcLocal, count);
AscendC::Log<T>(dstLocal, srcLocal, count);
```

### 检视方法

```bash
grep -n "std::\(abs\|min\|max\|sqrt\|exp\|log\|sin\|cos\|floor\|ceil\)" <file>
```

---

## API-3: DataCopy/DataCopyPad 对齐要求

**严重级别**：高

### 问题描述

DataCopy 要求数据量必须 32 字节对齐，非对齐会导致数据错误。推荐统一使用 DataCopyPad 自动处理对齐问题。

### 对齐要求

| 数据类型 | 对齐元素数 | 最小对齐字节数 |
|---------|-----------|--------------|
| half (2 bytes) | 16 | 32 |
| float (4 bytes) | 8 | 32 |
| int32_t (4 bytes) | 8 | 32 |

### 推荐做法

```cpp
// ✅ 推荐：统一使用 DataCopyPad，自动处理对齐
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 错误示例

```cpp
// ❌ 错误：非对齐数据使用 DataCopy
AscendC::DataCopy(xLocal, xGm, 4);  // cols=4 (16 bytes)，数据错误
```

### 正确示例

```cpp
// ✅ 正确：使用 DataCopyPad 处理非对齐
AscendC::DataCopyExtParams copyParams;
copyParams.blockLen = cols * sizeof(float);  // 单位：字节
AscendC::DataCopyPad(xLocal, xGm, copyParams);
```

### 检视方法

检查代码中 `DataCopy` 的使用，确认数据量是否保证 32 字节对齐。非对齐场景必须使用 `DataCopyPad`。

---

## API-4: Compare API 256字节对齐要求

**严重级别**：高

### 问题描述

Compare API 要求 `count` 个元素所占空间必须 **256 字节对齐**。

### 处理方案

使用 Padding 策略：

```cpp
// 1. 计算对齐大小（float 类型：64 的倍数）
constexpr uint32_t A0 = 32;
constexpr uint32_t A0_ALIGN = (A0 + 63) / 64 * 64;  // = 64

// 2. UB Buffer 使用对齐大小
pipe.InitBuffer(inQueue, 1, R * A0_ALIGN * sizeof(float));

// 3. CopyIn 时填充极值
Duplicate(xLocal, -FLT_MAX, R * A0_ALIGN);  // ArgMax 用极小值
// 再拷贝实际数据到前 A0 个位置

// 4. API 调用使用对齐大小
Compare(cmpLocal, srcLocal, maxLocal, CMPMODE::GT, A0_ALIGN);

// 5. CopyOut 只输出有效数据
DataCopy(dstGm, yLocal, A0);  // 只输出 A0 个
```

### 极值选择

- ArgMax / 找最大值：`-FLT_MAX` 或 `-INFINITY`
- ArgMin / 找最小值：`FLT_MAX` 或 `INFINITY`

### 检视方法

检查 Compare API 调用的 count 参数是否满足 256 字节对齐。

---

> **注意**：流水线同步相关规范（EnQue/DeQue 配对使用）已移至 `ascendc-perf.md` 条例 PREC-1，请参阅该文档获取完整的三步流程（CopyIn/Compute/CopyOut）说明。

---

## API-6: AllocTensor/FreeTensor 必须配对使用

**严重级别**：高

### 问题描述

使用队列管理内存时，AllocTensor 和 FreeTensor 必须配对调用，否则会导致内存泄漏。

### 错误示例

```cpp
// ❌ 错误：AllocTensor 后忘记 FreeTensor
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
// ... 使用 xLocal
// 忘记 FreeTensor
```

### 正确示例

```cpp
// ✅ 正确：AllocTensor/FreeTensor 配对
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
// ... 使用 xLocal
inQueueX.FreeTensor(xLocal);
```

### 检视方法

```bash
# 检查 AllocTensor/FreeTensor 配对
grep -c "AllocTensor" <file>
grep -c "FreeTensor" <file>
# 两者数量应该相等
```

---

## API-7: 禁止动态内存分配

**严重级别**：高

### 问题描述

AI Core 无动态内存管理能力，禁止使用动态内存分配。

### 错误示例

```cpp
// ❌ 禁止：动态内存分配
std::vector<int> vec;       // 动态分配
int* ptr = new int[10];     // 动态分配
int* arr = malloc(100);     // 动态分配
```

### 正确示例

```cpp
// ✅ 正确：使用静态分配
int arr[10];                          // 栈分配（Host 侧）
constexpr uint32_t SIZE = 1024;       // 编译期常量
pipe.InitBuffer(inQueue, 2, SIZE);    // UB 静态分配（Kernel 侧）
```

### 检视方法

```bash
grep -n "new \|malloc\|std::vector" <file>
```

---

## API-8: repeatTimes 限制（≤255）

**严重级别**：中

### 问题描述

部分 Ascend C API 的 `repeatTimes` 参数类型是 `uint8_t`，最大值 255。超过需要分批处理。

### 受影响 API

- 循环类 API（如 `DataCopy` 的 blockCount 等）

### 处理方案

当 repeatTimes > 255 时，需要分批调用：

```cpp
// 分批处理
constexpr uint32_t MAX_REPEAT = 255;
uint32_t repeatTimes = ...;
uint32_t fullBatches = repeatTimes / MAX_REPEAT;
uint32_t remainder = repeatTimes % MAX_REPEAT;

for (uint32_t i = 0; i < fullBatches; i++) {
    // 调用 API，repeatTimes = MAX_REPEAT
}
if (remainder > 0) {
    // 调用 API，repeatTimes = remainder
}
```

### 检视方法

检查循环类 API 的 repeatTimes 参数是否有溢出风险。

---

## API-9: Cast RoundMode 正确性

**严重级别**：中

### 问题描述

Cast API 的 RoundMode 参数必须正确选择，否则会导致精度问题。

### RoundMode 选择

| 转换方向 | 推荐 RoundMode | 说明 |
|---------|---------------|------|
| float → half | `CAST_ROUND` | 四舍五入 |
| half → float | `CAST_NONE` | 无需舍入 |
| float → int | `CAST_FLOOR` 或 `CAST_ROUND` | 根据需求选择 |

### 错误示例

```cpp
// ❌ 错误：float→half 使用 CAST_NONE 可能丢失精度
AscendC::Cast<half>(dstLocal, srcLocal, AscendC::RoundMode::CAST_NONE, count);
```

### 正确示例

```cpp
// ✅ 正确：float→half 使用 CAST_ROUND
AscendC::Cast<half>(dstLocal, srcLocal, AscendC::RoundMode::CAST_ROUND, count);
```

### 检视方法

检查 Cast API 调用的 RoundMode 参数是否与转换方向匹配。

---

## 检视检查清单

使用以下清单快速检查代码是否满足 API 最佳实践：

- [ ] **API-1**: 是否使用了 SetValue/GetValue？（生产代码禁止）
- [ ] **API-2**: 是否使用了 std:: 计算函数？（改用 Ascend C API）
- [ ] **API-3**: DataCopy 的数据量是否 32 字节对齐？（非对齐用 DataCopyPad）
- [ ] **API-4**: Compare API 的 count 是否 256 字节对齐？
- [ ] **API-6**: AllocTensor/FreeTensor 是否配对？分支/提前 return 路径是否都有 Free？
- [ ] **API-7**: 是否有动态内存分配？（禁止）
- [ ] **API-8**: repeatTimes 是否可能超过 255？
- [ ] **API-9**: Cast 的 RoundMode 是否正确？
- [ ] **API-10**: DataCopyParams.blockLen 是否用了 32字节块数？DataCopyExtParams.blockLen 是否用了字节数？
- [ ] **API-12**: CrossCoreSetFlag/WaitFlag 是否一一配对？所有代码路径（含提前 return）是否都有 SetFlag？是否与 Matmul 高阶 API 混用？同一 flagId 是否超过 15 次？

---

## API-10: DataCopy 与 DataCopyPad 的 blockLen 单位差异

**严重级别**：高

### 问题描述

`blockLen` 的单位取决于**哪个 API** 使用它，而非结构体本身：

| API | 参数结构体 | `blockLen` 单位 |
|-----|-----------|----------------|
| `DataCopy` | `DataCopyParams` | **32字节块数（DataBlock）** |
| `DataCopyPad` | `DataCopyExtParams` | **字节** |
| `DataCopyPad` | `DataCopyParams` | **字节**（注意：与 DataCopy 中同一结构体单位不同！） |

> **官方文档原文**（DataCopyPad ISASI，8.5.0）：
> - `DataCopyExtParams.blockLen`："每个连续传输数据块长度单位为**字节**"
> - `DataCopyParams.blockLen`（用于DataCopyPad时）："每个连续传输数据块长度单位为**字节**"
> - `DataCopyParams.blockLen`（用于DataCopy时）："单位为 **DataBlock（32字节）**"

**同一结构体在不同 API 中语义不同**，是最常见的混淆来源。

### stride 单位说明

`DataCopyExtParams` 的 `srcStride`/`dstStride` 单位也取决于操作数位置：
- 操作数位于 **VECIN/VECOUT**：单位为 **DataBlock（32字节）**
- 操作数位于 **GM**：单位为**字节**

### 错误示例

```cpp
// ❌ 错误：DataCopyPad + DataCopyExtParams，误用 32字节块单位
DataCopyExtParams ep;
ep.blockLen = actualColumnCount * sizeof(half) / 32;  // 少搬了 32 倍数据！
DataCopyPad(dst, src, ep);

// ❌ 错误：DataCopy + DataCopyParams，误用字节单位
DataCopyParams dp;
dp.blockLen = totalCount * sizeof(float);  // 超出范围或搬运了 32 倍数据！
DataCopy(dst, src, dp);
```

### 正确示例

```cpp
// ✅ DataCopyPad + DataCopyExtParams：blockLen 单位是字节
DataCopyExtParams ep;
ep.blockLen = actualColumnCount * sizeof(half);          // 字节
ep.blockCount = dealRowCount;
ep.srcStride = (totalColumns - actualColumnCount) / (32 / sizeof(half));  // 单位32字节块（src在VECIN）
ep.dstStride = 0;
ep.rsv = 0;
DataCopyPad(dst, src, ep);

// ✅ DataCopy + DataCopyParams：blockLen 单位是 32 字节块
DataCopyParams dp;
dp.blockLen = totalCount * sizeof(float) / 32;           // 32字节块数
dp.blockCount = 1;
DataCopy(dst, src, dp);
```

### 检视方法

```bash
grep -n "blockLen\s*=" <file>
```

**判断规则**：
1. 看使用的是 `DataCopy` 还是 `DataCopyPad`
2. `DataCopy` → blockLen 必须是 `/ 32` 的形式
3. `DataCopyPad` → blockLen 必须是 `* sizeof(T)` 的形式，无论用哪种结构体

---

> **注意**：GM 偏移计算使用 int64_t 的规范已移至 `ascendc-topk.md` 条目 8，请参阅该文档获取完整的溢出场景分析和正确示例。

---

## API-12: CrossCoreSetFlag/WaitFlag 必须对称

**严重级别**：高

### 问题描述

多核算子中，Cube 核和 Vec 核通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 进行跨核同步。若 SetFlag 和 WaitFlag 不一一对应，或存在某条代码路径跳过了 SetFlag，对应的 WaitFlag 将永久阻塞，导致死锁。

> **官方约束**（8.5.0）：
> - `CrossCoreSetFlag` 内部已被 Matmul 高阶 API 使用，**禁止二者混用**，否则有 flagId 冲突风险
> - **同一 flagId 计数器最多设置 15 次**

### 核心原则

1. **每个 WaitFlag 必须有且仅有一个对称的 SetFlag**
2. **所有能到达 WaitFlag 的代码路径，都必须能触发对应的 SetFlag**
3. **提前 return、条件分支不能跳过 SetFlag**
4. **禁止与 Matmul 高阶 API 混用**：Matmul 高阶 API 内部使用了 CrossCoreSetFlag 进行核间同步控制，若开发者同时使用 CrossCoreSetFlag/WaitFlag 和 Matmul 高阶 API，会有 flagId 冲突的风险
5. **同一 flagId 计数器最多设置 15 次**：超过此限制行为未定义

### 同步配对示意

```
Cube 侧                          Vec 侧
─────────────────────────────────────────────────
ComputeMm1 末尾:
  CrossCoreSetFlag(syncC1V1)  →  ComputeVec1 开头:
                                   CrossCoreWaitFlag(syncC1V1)  ✅

ComputeVec1 末尾:
  CrossCoreSetFlag(syncV1C2)  →  ComputeMm2 开头:
  ←  CrossCoreWaitFlag(syncV1C2)                              ✅

ComputeMm2 末尾:
  CrossCoreSetFlag(syncC2V2)  →  ComputeVec2 开头:
                                   CrossCoreWaitFlag(syncC2V2)  ✅
```

### 错误示例

```cpp
// ❌ 危险：提前 return 跳过了 SetFlag
void ComputeMm1(...) {
    if (edgeCase) {
        return;  // Vec1 的 WaitFlag(C1V1) 永久等待 → 死锁！
    }
    // ... 正常计算 ...
    CrossCoreSetFlag(syncC1V1);  // 只有正常路径才执行
}
```

### 正确示例

```cpp
// ✅ 正确：所有路径都 SetFlag
void ComputeMm1(...) {
    if (edgeCase) {
        CrossCoreSetFlag(syncC1V1);  // 提前退出前必须 Set
        return;
    }
    // ... 正常计算 ...
    CrossCoreSetFlag(syncC1V1);
}
```

### 双缓冲 loop 偏移下溢风险

使用双缓冲时，常见 `(loop - 1) % N` 的计算。当 `loop` 为 `uint32_t` 且等于 0 时，`loop - 1` 会下溢为 `UINT32_MAX`，导致错误的 buffer 索引。

```cpp
// ❌ 危险：loop=0 时 uint32_t 下溢
uint32_t inIdx = (info.loop - 1) % preLoadNum;

// ✅ 正确：首次迭代单独处理
if (info.isFirstIteration) {
    // 使用默认初始值，不读上一轮
} else {
    uint32_t inIdx = (info.loop - 1) % preLoadNum;  // loop >= 1，安全
}
```

### 检视方法

```bash
grep -n "CrossCoreSetFlag\|CrossCoreWaitFlag" <cube_file> <vec_file>
```

**逐步检视**：
1. 列出所有 `CrossCoreSetFlag` 和 `CrossCoreWaitFlag` 调用，按 flag 变量名配对
2. 对每个 `CrossCoreSetFlag`，向上追踪是否存在 `if/return/break` 绕过它的路径
3. 检查是否有 `(loop - k) % N` 形式的 buffer 索引，确认 `loop < k` 时有保护分支
4. 检查是否同时调用了 Matmul 高阶 API（如 `Matmul`、`MatmulSimple`）
5. 统计同一 flagId 的 `CrossCoreSetFlag` 调用次数，确认不超过 15 次

---

## SUNSET-1: 禁止使用日落（废弃）API

**严重级别**：高  **适用范围**：[适用: All]

### 问题描述

CANN 每版本废弃（日落）一批接口并提供替代，废弃接口在删除期限后无法编译。覆盖 Runtime API(C/Python) 的 `aclrt*`/`acl.op.*` 与算子库 `aclnn*`。清单随版本动态变化，由 `scripts/clause.get_sunset_api.py` 动态解析官方废弃文档生成。

### 错误示例

```cpp
aclrtGetVersion(&version);          // ❌ 已废弃 → aclsysGetVersionStr
aclnnGroupedMatmulV2(...);          // ❌ 删除期限 2027.3.30 → aclnnGroupedMatmulV5
```

### 正确示例

```cpp
aclsysGetVersionStr(versionStr);    // ✅ 替代接口
aclnnGroupedMatmulV5(...);          // ✅ 最新版
```

### 检视策略 — 预研报告驱动（不走假设检验）

确定性符号匹配，命中即违规，不收集证据分值。

1. Read `{api_prestudy_path}` 的「## 日落 API」章节（api-prestudy Step 2.5 比对产出）。
2. 无命中 → PASS。有命中 → Grep 命中行核实是否真使用（排除注释/字符串/死代码），且是日落符号本身而非 `V2`/`V3` 等替代品。真使用 → FAIL(HIGH) + 替代接口 + 删除期限；仅注释提及 → PASS + 清理提示。
3. 预研报告无此章节 → fallback：`python3 {skill_base}/scripts/clause.get_sunset_api.py` 得清单（格式 `sym -> rep`，**只取箭头左侧的日落符号**，右侧是替代品勿混入），grep 代码 `acl(rt|nn)[A-Za-z0-9_]+` / `acl\.(op|rt)\.` 比对（注意词法边界，不误匹配替代品）。无法获取 → SUSPICIOUS + 标注。

---

## SUNSET-2: 禁止引用日落头文件/库

**严重级别**：高  **适用范围**：[适用: All]

### 问题描述

CANN 同样日落头文件路径与链接库，废弃项在删除期限后不再存在导致编译/链接失败。典型：`op_proto/inc/*.h` → `op_graph/inc/${ops_project}_ops_proto.h`；`libopapi.so` → `libopapi_${ops_project}.so`（期限 2026.12.30）。

### 错误示例

```cpp
#include "op_proto/inc/ops_proto.h"       // ❌ → op_graph/inc/${ops_project}_ops_proto.h
target_link_libraries(... libopapi.so)    // ❌ → libopapi_${ops_project}.so
```

### 检视策略 — 预研报告驱动（不走假设检验）

1. Read `{api_prestudy_path}`「## 日落 API」章节（含头文件/库命中）。
2. 无命中 → PASS。有命中 → Grep 确认是真实 `#include`/链接配置（非注释），且是日落路径本身（`op_proto/inc` 日落，`op_graph/inc` 替代）。真引用 → FAIL(HIGH) + 替代 + 期限；仅注释 → PASS + 清理提示。
3. 预研报告无此章节 → fallback：`clause.get_sunset_api.py` 得「算子库(头文件/库)」清单，grep `#include.*op_proto` / `libopapi\.so`。无法获取 → SUSPICIOUS。
