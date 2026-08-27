# SIMT API C++风格转换为C风格规范

<适用>
语言: C++
侧别: Kernel
领域: true
触发: Simt::GetThreadNum, Simt::GetBlockIdx, Simt::GetBlockNum, Simt::VF_CALL, Simt::Dim3, Simt::Min, Simt::Max, Simt::Floor, Simt::Ceil, Simt::Abs, Simt::AtomicAdd, Simt::AtomicSub, Simt::ThreadBarrier, Simt::UintDiv, Simt::WarpShflSync, Simt::IsNan, Simt::IsFinite, Simt::IsInf, simt_api/asc_simt.h
默认启用: true
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

> **适用场景**：Kernel 侧 SIMT 代码检视
>
> **检视目标**：识别 C++ 风格 SIMT API，指导转换为 C 风格 API。

---

## 检视前置要求

> **重要**：Agent 在检视本条例时，必须先通过 `/ascendc-docs-search` skill 查阅 `$ASC_DEVKIT_DIR` 中的 SIMT API 头文件，确认 API 的转换规则、C 风格替代 API 是否存在等信息是否与条例描述一致。若头文件与条例有差异，以最新头文件为准，并记录差异供后续更新条例。

### 需要查阅头文件的 API

| 条例编号 | 涉及 API | 查阅重点 |
|---------|---------|---------|
| SIMT-1 | `GetThreadNum`, `GetThreadIdx` | 确认 C 风格 API 替代 |
| SIMT-2 | `GetBlockIdx`, `GetBlockNum` | 确认 C 风格 API 替代、变量名冲突 |
| SIMT-3 | `VF_CALL`, `Dim3` | 确认 asc_vf_call 和 dim3 替代 |
| SIMT-4 | `Min`, `Max` | 确认 min/max/fminf/fmaxf 替代 |
| SIMT-5 | `Floor`, `Ceil` | 确认 floorf/ceilf 替代 |
| SIMT-6 | `Abs` | 确认 fabsf 替代 |
| SIMT-7 | `AtomicAdd`, `AtomicSub` | 确认 asc_atomic_add/asc_atomic_sub 替代 |
| SIMT-8 | `ThreadBarrier` | 确认 asc_syncthreads 替代 |
| SIMT-9 | `UintDiv` | 确认不转换（官方规定） |
| SIMT-10 | `WarpShflSync` | 确认 asc_shfl 替代 |
| SIMT-11 | `IsNan`, `IsFinite`, `IsInf` | 确认 isnan/isfinite/isinf 替代 |

---

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| SIMT-1 | 禁止使用 Simt::GetThreadNum | API转换 | 高 |
| SIMT-2 | 禁止使用 Simt::GetThreadIdx | API转换 | 高 |
| SIMT-3 | 禁止使用 Simt::GetBlockIdx | API转换 | 高 |
| SIMT-4 | 禁止使用 Simt::GetBlockNum | API转换 | 高 |
| SIMT-5 | 禁止使用 Simt::VF_CALL/Dim3 | API转换 | 高 |
| SIMT-6 | 禁止使用 Simt::Min/Max（float） | API转换 | 高 |
| SIMT-7 | 禁止使用 Simt::Min/Max（int） | API转换 | 高 |
| SIMT-8 | 禁止使用 Simt::Floor/Ceil | API转换 | 高 |
| SIMT-9 | 禁止使用 Simt::Abs | API转换 | 高 |
| SIMT-10 | 禁止使用 Simt::AtomicAdd/AtomicSub | API转换 | 高 |
| SIMT-11 | 禁止使用 Simt::ThreadBarrier | API转换 | 高 |
| SIMT-12 | UintDiv 必须保留 | API保留 | 高 |
| SIMT-13 | 禁止使用 Simt::WarpShflSync | API转换 | 中 |
| SIMT-14 | 禁止使用 Simt::IsNan | API转换 | 高 |
| SIMT-15 | 禁止使用 Simt::IsFinite | API转换 | 高 |
| SIMT-16 | 禁止使用 Simt::IsInf | API转换 | 高 |

---

## SIMT-1: 禁止使用 Simt::GetThreadNum

**严重级别**：高

### 问题描述

`Simt::GetThreadNum()` 是 C++ 风格 API（带 `Simt::` 前缀），应转换为 C 风格 API `blockDim.x/y/z`。

**注意**：不带 `Simt::` 前缀的 `GetThreadNum()` 是 Ascend C API，无需转换。

**维度对应规则**：
- `GetThreadNum<0>()` → `blockDim.x` （x维度）
- `GetThreadNum<1>()` → `blockDim.y` （y维度）
- `GetThreadNum<2>()` → `blockDim.z` （z维度）

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（带 Simt:: 前缀）
uint32_t threadNum = AscendC::Simt::GetThreadNum<0>();
uint32_t threadNum = Simt::GetThreadNum();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API（根据维度选择）
uint32_t threadNumX = blockDim.x;  // x维度
uint32_t threadNumY = blockDim.y;  // y维度
uint32_t threadNumZ = blockDim.z;  // z维度

// ✅ 正确：不带 Simt:: 前缀的 GetThreadNum 无需转换
uint32_t threadNum = GetThreadNum();  // Ascend C API，保持原样
```

### 检视方法

```bash
grep -n "GetThreadNum" <file>
```

**注意**：需根据模板参数 `<维度>` 选择正确的成员（.x/.y/.z）。

---

## SIMT-2: 禁止使用 Simt::GetThreadIdx

**严重级别**：高

### 问题描述

`Simt::GetThreadIdx()` 是 C++ 风格 API（带 `Simt::` 前缀），应转换为 C 风格 API `threadIdx.x/y/z`。

**注意**：不带 `Simt::` 前缀的 `GetThreadIdx()` 是 Ascend C API，无需转换。

**维度对应规则**：
- `GetThreadIdx<0>()` → `threadIdx.x` （x维度）
- `GetThreadIdx<1>()` → `threadIdx.y` （y维度）
- `GetThreadIdx<2>()` → `threadIdx.z` （z维度）

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（带 Simt:: 前缀）
uint32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
uint32_t threadIdx = Simt::GetThreadIdx();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API（根据维度选择）
uint32_t tidX = threadIdx.x;  // x维度
uint32_t tidY = threadIdx.y;  // y维度
uint32_t tidZ = threadIdx.z;  // z维度

// ✅ 正确：不带 Simt:: 前缀的 GetThreadIdx 无需转换
uint32_t threadIdx = GetThreadIdx();  // Ascend C API，保持原样
```

### 检视方法

```bash
grep -n "GetThreadIdx" <file>
```

**注意**：避免变量名冲突，变量名不能命名为 `threadIdx`，建议使用 `tid` 或 `thread_id`。需根据模板参数 `<维度>` 选择正确的成员（.x/.y/.z）。

---

## SIMT-3: 禁止使用 Simt::GetBlockIdx

**严重级别**：高

### 问题描述

`Simt::GetBlockIdx()` 是 C++ 风格 API（带 `Simt::` 前缀），应转换为 C 风格 API `blockIdx.x`。

**注意**：不带 `Simt::` 前缀的 `GetBlockIdx()` 是 Ascend C API，无需转换。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（带 Simt:: 前缀）
uint32_t blockIdx = Simt::GetBlockIdx();
uint32_t blockIdx = AscendC::Simt::GetBlockIdx();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API（Simt::GetBlockIdx 转换后）
uint32_t bid = blockIdx.x;

// ✅ 正确：不带 Simt:: 前缀的 GetBlockIdx 无需转换
uint32_t blockIdx = GetBlockIdx();  // Ascend C API，保持原样
```

### 检视方法

```bash
grep -n "GetBlockIdx" <file>
```

**注意**：避免变量名冲突，变量名不能命名为 `blockIdx`，建议使用 `bid` 或 `block_id`。

---

## SIMT-4: 禁止使用 Simt::GetBlockNum

**严重级别**：高

### 问题描述

`Simt::GetBlockNum()` 是 C++ 风格 API（带 `Simt::` 前缀），应转换为 C 风格 API `gridDim.x/y/z`。

**注意**：不带 `Simt::` 前缀的 `GetBlockNum()` 是 Ascend C API，无需转换。

**维度对应规则**：
- `GetBlockNum<0>()` 或 `GetBlockNum()` → `gridDim.x` （x维度，默认）
- `GetBlockNum<1>()` → `gridDim.y` （y维度）
- `GetBlockNum<2>()` → `gridDim.z` （z维度）

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（带 Simt:: 前缀）
uint32_t blockNum = AscendC::Simt::GetBlockNum();
uint32_t blockNum = Simt::GetBlockNum();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API（根据维度选择）
uint32_t gridDimX = gridDim.x;  // x维度
uint32_t gridDimY = gridDim.y;  // y维度
uint32_t gridDimZ = gridDim.z;  // z维度

// ✅ 正确：不带 Simt:: 前缀的 GetBlockNum 无需转换
uint32_t blockNum = GetBlockNum();  // Ascend C API，保持原样
```

### 检视方法

```bash
grep -n "GetBlockNum" <file>
```

**注意**：避免变量名冲突，变量名不能命名为 `gridDim` 或 `blockDim`，建议使用 `grid_size` 或 `block_size`。需根据模板参数 `<维度>` 选择正确的成员（.x/.y/.z）。

---

## SIMT-5: 禁止使用 Simt::VF_CALL/Dim3

**严重级别**：高

### 问题描述

`Simt::VF_CALL` 和 `Simt::Dim3` 是 C++ 风格 API，应转换为 C 风格 API `asc_vf_call` 和 C 风格类型 `dim3`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
Simt::VF_CALL<KernelClass>(Simt::Dim3{32, 1, 1}, Simt::Dim3{1, 1, 1}, ...);
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API 和 C 风格类型
asc_vf_call<KernelClass>(dim3{32, 1, 1}, dim3{1, 1, 1}, ...);
```

### 检视方法

```bash
grep -n "VF_CALL\|Simt::Dim3" <file>
```

---

## SIMT-6: 禁止使用 Simt::Min/Max（float）

**严重级别**：高

### 问题描述

`Simt::Min` 和 `Simt::Max` 用于 float 类型时，应转换为标准数学函数 `fminf` 和 `fmaxf`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（float 类型）
float minVal = Simt::Min(a, b);
float maxVal = Simt::Max(a, b);
```

### 正确示例

```cpp
// ✅ 正确：使用标准数学函数（float 类型）
float minVal = fminf(a, b);
float maxVal = fmaxf(a, b);
```

### 检视方法

```bash
grep -n "Simt::Min\|Simt::Max" <file>
```

**注意**：需区分参数类型，float 类型使用 `fminf/fmaxf`，int 类型使用 `min/max`。

---

## SIMT-7: 禁止使用 Simt::Min/Max（int）

**严重级别**：高

### 问题描述

`Simt::Min` 和 `Simt::Max` 用于 int/int64_t 类型时，应转换为标准函数 `min` 和 `max`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（int 类型）
int minVal = Simt::Min(a, b);
int64_t maxVal = Simt::Max(a, b);
```

### 正确示例

```cpp
// ✅ 正确：使用标准函数（int 类型）
int minVal = min(a, b);
int64_t maxVal = max(a, b);
```

### 检视方法

```bash
grep -n "Simt::Min\|Simt::Max" <file>
```

**注意**：需根据参数类型选择正确的替代函数。

---

## SIMT-8: 禁止使用 Simt::Floor/Ceil

**严重级别**：高

### 问题描述

`Simt::Floor` 和 `Simt::Ceil` 用于 float 类型时，应转换为标准数学函数 `floorf` 和 `ceilf`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
float floorVal = Simt::Floor(value);
float ceilVal = Simt::Ceil(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准数学函数
float floorVal = floorf(value);
float ceilVal = ceilf(value);
```

### 检视方法

```bash
grep -n "Simt::Floor\|Simt::Ceil" <file>
```

---

## SIMT-9: 禁止使用 Simt::Abs

**严重级别**：高

### 问题描述

`Simt::Abs` 用于 float 类型时，应转换为标准数学函数 `fabsf`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
float absVal = Simt::Abs(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准数学函数
float absVal = fabsf(value);
```

### 检视方法

```bash
grep -n "Simt::Abs" <file>
```

---

## SIMT-10: 禁止使用 Simt::AtomicAdd/AtomicSub

**严重级别**：高

### 问题描述

`Simt::AtomicAdd` 和 `Simt::AtomicSub` 是 C++ 风格 API，应转换为 C 风格 API `asc_atomic_add` 和 `asc_atomic_sub`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
Simt::AtomicAdd<int64_t>(&counter, value);
Simt::AtomicSub<int>(&counter, value);
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API
asc_atomic_add(&counter, value);
asc_atomic_sub(&counter, value);
```

### 检视方法

```bash
grep -n "AtomicAdd\|AtomicSub" <file>
```

---

## SIMT-11: 禁止使用 Simt::ThreadBarrier

**严重级别**：高

### 问题描述

`Simt::ThreadBarrier()` 是 C++ 风格 API，应转换为 C 风格同步函数 `asc_syncthreads()`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
Simt::ThreadBarrier();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格同步函数
asc_syncthreads();
```

### 检视方法

```bash
grep -n "ThreadBarrier" <file>
```

---

## SIMT-12: UintDiv 必须保留

**严重级别**：高

### 问题描述

`Simt::UintDiv` 是华为官方规定的保留 API，**禁止转换**，无 C 风格替代。

### 错误示例

```cpp
// ❌ 禁止：尝试转换为其他形式
uint32_t result = asc_uint_div(idx, divisor, shift);  // 不存在此 API
uint32_t result = idx / divisor;  // 性能较差
```

### 正确示例

```cpp
// ✅ 正确：保持 UintDiv 原样
uint32_t result = Simt::UintDiv(idx, divisor, shift);
```

### 检视方法

```bash
grep -n "UintDiv" <file>
```

**注意**：UintDiv 是华为针对无符号整数除法的特殊优化实现，使用魔法数和移位操作，性能远优于普通除法。转换会导致编译错误或性能下降。

---

## SIMT-13: 禁止使用 Simt::WarpShflSync

**严重级别**：中

### 问题描述

`Simt::WarpShflSync` 是 C++ 风格 API，应转换为 C 风格 API `asc_shfl`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
int value = Simt::WarpShflSync(mask, var, lane);
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API
int value = asc_shfl(mask, var, lane);
```

### 检视方法

```bash
grep -n "WarpShflSync" <file>
```

---

## SIMT-14: 禁止使用 Simt::IsNan

**严重级别**：高

### 问题描述

`Simt::IsNan` 是 C++ 风格 API，应转换为标准函数 `isnan`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
bool result = Simt::IsNan(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准函数
bool result = isnan(value);
```

### 检视方法

```bash
grep -n "IsNan" <file>
```

---

## SIMT-15: 禁止使用 Simt::IsFinite

**严重级别**：高

### 问题描述

`Simt::IsFinite` 是 C++ 风格 API，应转换为标准函数 `isfinite`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
bool result = Simt::IsFinite(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准函数
bool result = isfinite(value);
```

### 检视方法

```bash
grep -n "IsFinite" <file>
```

---

## SIMT-16: 禁止使用 Simt::IsInf

**严重级别**：高

### 问题描述

`Simt::IsInf` 是 C++ 风格 API，应转换为标准函数 `isinf`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
bool result = Simt::IsInf(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准函数
bool result = isinf(value);
```

### 检视方法

```bash
grep -n "IsInf" <file>
```

---

## API 转换映射表

| C++ API | C API/替代 | 说明 |
|---------|-----------|------|
| `Simt::GetThreadNum()` | `blockDim.x/y/z` | C 风格 API（根据维度选择） |
| `Simt::GetThreadIdx()` | `threadIdx.x/y/z` | C 风格 API（根据维度选择） |
| `Simt::GetBlockIdx()` | `blockIdx.x` | C 风格 API |
| `Simt::GetBlockNum()` | `gridDim.x/y/z` | C 风格 API（根据维度选择） |
| `Simt::VF_CALL` | `asc_vf_call` | ASC 核函数调用 |
| `Simt::Dim3` | `dim3` | C 风格类型 |
| `Simt::Min(float)` | `fminf` | 标准数学函数 |
| `Simt::Max(float)` | `fmaxf` | 标准数学函数 |
| `Simt::Min(int)` | `min` | 标准函数 |
| `Simt::Max(int)` | `max` | 标准函数 |
| `Simt::Floor` | `floorf` | 标准数学函数 |
| `Simt::Ceil` | `ceilf` | 标准数学函数 |
| `Simt::Abs` | `fabsf` | 标准数学函数 |
| `Simt::AtomicAdd` | `asc_atomic_add` | ASC 原子操作 |
| `Simt::AtomicSub` | `asc_atomic_sub` | ASC 原子操作 |
| `Simt::ThreadBarrier` | `asc_syncthreads()` | ASC 同步函数 |
| `Simt::UintDiv` | `Simt::UintDiv` | **保留，不转换** |
| `Simt::WarpShflSync` | `asc_shfl` | ASC Warp 操作 |
| `Simt::IsNan` | `isnan` | 标准函数 |
| `Simt::IsFinite` | `isfinite` | 标准函数 |
| `Simt::IsInf` | `isinf` | 标准函数 |

---

## 检视检查清单

使用以下清单快速检查代码是否满足 SIMT API C 风格化要求：

- [ ] **SIMT-1**: 是否使用了 `GetThreadNum`？（转换为 `blockDim.x/y/z`，根据维度选择）
- [ ] **SIMT-2**: 是否使用了 `GetThreadIdx`？（转换为 `threadIdx.x/y/z`，根据维度选择，注意变量名冲突）
- [ ] **SIMT-3**: 是否使用了 `GetBlockIdx`？（转换为 `blockIdx.x`，注意变量名冲突）
- [ ] **SIMT-4**: 是否使用了 `GetBlockNum`？（转换为 `gridDim.x/y/z`，根据维度选择）
- [ ] **SIMT-5**: 是否使用了 `VF_CALL/Dim3`？（转换为 `asc_vf_call/dim3`）
- [ ] **SIMT-6**: 是否使用了 `Min/Max(float)`？（转换为 `fminf/fmaxf`）
- [ ] **SIMT-7**: 是否使用了 `Min/Max(int)`？（转换为 `min/max`）
- [ ] **SIMT-8**: 是否使用了 `Floor/Ceil`？（转换为 `floorf/ceilf`）
- [ ] **SIMT-9**: 是否使用了 `Abs`？（转换为 `fabsf`）
- [ ] **SIMT-10**: 是否使用了 `AtomicAdd/AtomicSub`？（转换为 `asc_atomic_add/asc_atomic_sub`）
- [ ] **SIMT-11**: 是否使用了 `ThreadBarrier`？（转换为 `asc_syncthreads()`）
- [ ] **SIMT-12**: UintDiv 是否正确保留？（禁止转换）
- [ ] **SIMT-13**: 是否使用了 `WarpShflSync`？（转换为 `asc_shfl`）
- [ ] **SIMT-14**: 是否使用了 `IsNan`？（转换为 `isnan`）
- [ ] **SIMT-15**: 是否使用了 `IsFinite`？（转换为 `isfinite`）
- [ ] **SIMT-16**: 是否使用了 `IsInf`？（转换为 `isinf`）
- [ ] **变量名冲突**: 是否有名为 threadIdx/blockIdx/blockDim/gridDim 的局部变量？（重命名）
- [ ] **头文件**: 是否包含 `simt_api/asc_simt.h`？（必须在 namespace 外部）

---

## 头文件添加规则

### 正确示例

```cpp
// ✅ 正确：头文件在 namespace 外部
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"  // 必须在此位置

namespace MyKernel {
using namespace AscendC;
// ...
}
```

### 错误示例

```cpp
// ❌ 错误：头文件在 namespace 内部
namespace MyKernel {
using namespace AscendC;
#include "simt_api/asc_simt.h"  // 错误位置
// ...
}
```

---

## 变量名冲突处理

### 冲突检测

检测是否有与 C 风格 API 变量同名的局部变量：

```bash
grep -n "uint32_t threadIdx\|uint32_t blockIdx\|uint32_t blockDim\|uint32_t gridDim" <file>
```

### 冲突示例

```cpp
// ❌ 错误：变量名与 C 风格 API 变量冲突
uint32_t threadIdx = 0;  // 与 threadIdx.x 冲突
uint32_t blockIdx = 0;   // 与 blockIdx.x 冲突
```

### 正确示例

```cpp
// ✅ 正确：使用其他变量名
uint32_t tid = 0;        // thread id
uint32_t bid = 0;        // block id
uint32_t block_size = blockDim.x;  // 明确使用 .x 成员
```

---

## 执行流程

### Step 1: 搜索目标文件

```bash
find {operator_dir} -name "*simt*.h" -o -name "*simt*.cpp"
```

### Step 2: 检视 C++ API

```bash
# 搜索所有 Simt:: 前缀的 API
grep -n "Simt::" {kernel_file}

# 搜索线程管理 API
grep -n "GetThreadNum\|GetThreadIdx\|GetBlockIdx\|GetBlockNum" {kernel_file}

# 搜索核函数调用 API
grep -n "VF_CALL\|Dim3" {kernel_file}

# 搜索数学运算 API
grep -n "Min\|Max\|Floor\|Ceil\|Abs" {kernel_file}

# 搜索比较函数 API
grep -n "IsNan\|IsFinite\|IsInf" {kernel_file}

# 搜索原子操作 API
grep -n "AtomicAdd\|AtomicSub" {kernel_file}

# 搜索同步 API
grep -n "ThreadBarrier" {kernel_file}
```

### Step 3: 检视 UintDiv

```bash
# 搜索 UintDiv，确认正确保留
grep -n "UintDiv" {kernel_file}
```

### Step 4: 检视变量名冲突

```bash
# 检测变量名冲突
grep -n "uint32_t threadIdx\|uint32_t blockIdx\|uint32_t blockDim\|uint32_t gridDim" {kernel_file}
```

### Step 5: 检视头文件

```bash
# 检查头文件是否包含
grep -n "simt_api/asc_simt.h" {kernel_file}

# 检查头文件位置（必须在 namespace 外部）
grep -B5 -A5 "namespace" {kernel_file} | grep "simt_api"
```

### Step 6: 生成检视报告

记录所有需要转换的 API 及其转换建议。

---

## 注意事项

1. **Simt:: 前缀识别**：只有带 `Simt::` 前缀的 API 才需要转换（如 `Simt::GetBlockIdx`），不带前缀的 API（如 `GetBlockIdx`）是 Ascend C API，无需转换
2. **UintDiv 不可转换**：官方明确规定 UintDiv 必须保留，无 C 风格替代
3. **变量名冲突**：转换后使用 threadIdx/blockIdx 等内置变量时，必须确保无同名局部变量
4. **头文件位置**：`simt_api/asc_simt.h` 必须在 namespace 外部
5. **参数类型区分**：Min/Max 需根据参数类型选择正确的替代函数（float 用 fminf/fmaxf，int 用 min/max）
6. **模板参数**：VF_CALL 的模板参数（如 `<KernelClass>`）保持不变
7. **条件编译**：`#ifdef` 内的 API 需单独检视
8. **注释排除**：只检视实际代码，排除注释中的 API
9. **编译验证**：转换后必须通过编译验证

---