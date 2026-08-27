# Ascend C 红线 问题清单

<适用>
语言: C++
侧别: All, Host, Kernel
领域: true
默认启用: true

适用场景: Ascend C 算子开发中实际高频出现的编码红线问题，来源于生产环境经验总结
介绍: Ascend C 红线类 高频问题清单，10 条条款覆盖算子开发中最常触发的错误模式
类别(All): 确保除法和余数运算不会导致除以零的错误、外部数据作为数组索引时必须确保在数组大小范围内、确保有符号整数运算不溢出、确保无符号整数运算不回绕、禁止使用未初始化的变量、指针操作使用前必须要判空、需要避免数据竞争
类别(Host): 无
类别(Kernel): 资源泄露防护、gm内存偏移或大小必须用int64表示、在支持superKernel场景禁止直接对GM地址进行读写(需要改用GetValue或SetValue)
> **说明**：TOPK 问题是检视实践中发现的高频风险点，需重点关注。条款标注适用范围：`[适用: All]` / `[适用: Host]` / `[适用: Kernel]`
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 3
</检视负载>

## 快速索引

### 两者都适用 `[适用: All]`（7 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 1 | 除法/余数运算除零保护 | 数值安全 | 高 |
| 2 | 数组索引校验 | 内存安全 | 高 |
| 3 | 有符号整数运算不溢出 | 数值安全 | 高 |
| 4 | 无符号整数运算不回绕 | 数值安全 | 高 |
| 5 | 禁止使用未初始化的变量 | 内存安全 | 高 |
| 6 | 指针使用前判空 | 内存安全 | 高 |
| 7 | 需要避免数据竞争 | 资源管理 | 高 |


### 仅 Host 侧适用 `[适用: Host]`（0 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|


### 仅 Kernel 侧适用 `[适用: Kernel]`（3 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 8 | 资源泄露防护 | 资源管理 | 高 |
| 9 | gm内存偏移或大小必须用int64表示 | 内存安全 | 高 |
| 10 | 在支持superKernel场景，禁止直接对GM地址进行读写，需要改用GetValue或SetValue | 内存安全 | 高 |


---



## 详细规范

### 1 确保除法和余数运算不会导致除以零的错误 `[适用: All]`

> **Kernel 侧说明**：
> - Kernel 中的除法/取余运算，按除数来源分类判定（见下方 Step 2）。
> - **硬件 API 返回值**（`GetBlockNum()`、`GetTaskRation()` 等）和 **TilingData 字段**（`tilingData->*`）作除数时，适用 Kernel 侧排除规则，无需零值守卫。
> - 仅对 Kernel 内部独立计算的运行时变量（非白名单 API、非 TilingData）要求零值守卫。

**【检视策略】**

**Step 1 — 识别除法/取余运算**

扫描代码中所有 `/` 和 `%` 运算符（含 `CeilDiv`/`CeilDivide` 工具函数调用），提取除数表达式。

**Step 2 — 除数来源分类**

| 优先级 | 除数来源 | 识别方法 | 信任等级 |
|--------|---------|---------|---------|
| P0 | 编译期常量 | `constexpr` 声明、字面量、`AscendC::BLOCK_CUBE` 等框架常量 | 自动 PASS |
| P1 | 硬件 API 返回值 | 白名单 API 直接调用或赋值链可追溯 | Kernel 侧自动 PASS |
| P2 | TilingData 字段 | `tilingData->xxx` / `tilingData_.xxx` | Kernel 侧自动 PASS |
| P3 | 外部输入 | `shape->GetDim()`、`context->GetAttr()`、`GetActualSeqLen()` | 严格：必须有守卫 |
| P4 | 设计过程参数 | Tiling/Kernel 内部多步计算的中间值 | 严格：必须有守卫 |

**硬件 API 白名单（P1）**：

| 侧别 | 白名单 API | 典型变量名 |
|------|-----------|-----------|
| Kernel | `AscendC::GetBlockNum()` | `coreNum`, `blockNum_` |
| Kernel | `AscendC::GetTaskRation()` | `taskRation`, `coreRation` |
| Kernel | `AscendC::GetSubBlockNum()` | `bn`, `subBlockNum` |
| Tiling | `ascendcPlatform.GetCoreNumAic()` | `aicNum` |
| Tiling | `ascendcPlatform.GetCoreNumAiv()` | `aivNum` |
| Tiling | `ascendcPlatform.GetCoreNum()` | `coreNum` |
| Tiling | `ascendcPlatform.GetCoreMemSize(...)` | `ubSize`, `l1Size` 等 |

> **适用条件**：除数直接来自上述 API 返回值，或赋值链可追溯到上述 API 的变量。若除数经过算术运算（如 `GetBlockNum() - 1`），需另行分析运算结果是否可能为零。

**Step 3 — 按来源判定**

- **P0（编译期常量）**：值非零 → PASS。值为零 → FAIL。
- **P1（硬件 API）**：Kernel 侧自动 PASS（见排除规则）。Tiling 侧必须有零值守卫（见严格模式）。
- **P2（TilingData）**：Kernel 侧自动 PASS。Tiling 侧作为中间值按 P3/P4 处理。
- **P3（外部输入）**：必须有有效守卫模式之一 → 无守卫则 FAIL。
- **P4（设计过程参数）**：必须有有效守卫，或可追溯到 P0/P1 的非零值 → 否则 FAIL。

**边界收集**（P3/P4 需要时）：按本条例 Step 2 方法收集除数边界，按 Step 4 判定表做判定。

**【Kernel 侧排除规则】**

以下情况在 Kernel 侧自动排除，无需零值守卫：

| 排除条件 | 参数模式示例 | 排除原因 |
|---------|-------------|----------|
| 除数来自硬件 API 白名单 | `GetBlockNum()`, `GetTaskRation()` | 芯片出厂固定非零，异常场景由 Tiling 侧兜底 |
| 除数来自 TilingData | `tilingData->tileSize`, `tilingData->coreNum` | Tiling 阶段已校验非零 |
| 编译期常量 | `constexpr uint32_t BLOCK = 32` | 编译期固定非零 |

**判定方法**：
- 除数表达式直接匹配白名单 API 调用 → 直接判定 PASS
- 除数变量赋值链可追溯到白名单 API 或 `tilingData->xxx` → 直接判定 PASS
- 除数为 `constexpr` 且值非零 → 直接判定 PASS

**【Kernel 侧需校验场景】**

以下情况在 Kernel 侧仍需零值守卫：

| 校验条件 | 参数来源 | 代码模式 |
|---------|---------|----------|
| Kernel 内部计算的中间值 | 非 TilingData、非硬件 API | `if (computedDivisor == 0) { return; }` |
| 动态序列长度 | `GetActualSeqLen()` 运行时获取 | `if (actS1Size == 0) { return; }` |
| 条件分支中的计算值 | 依赖运行时条件的派生值 | `if (curMode == X && div != 0) { ... }` |

**【有效守卫模式】**

以下 6 种模式视为有效的零值守卫（任一存在 → PASS）：

| 模式 | 名称 | 代码形式 | 适用侧别 |
|------|------|---------|---------|
| A | OP_CHECK_IF | `OP_CHECK_IF(div == 0, LOG, return FAIL)` | Tiling |
| B | if-guard+return | `if (div == 0) return;` | 两侧 |
| C | std::max 保底 | `safeDiv = std::max(div, 1U)` | 两侧 |
| D | 三元运算符 | `safe = (div > 0) ? div : 1` | 两侧 |
| E | zero-flag+skip | `if (div==0) flag=true; if(!flag) { a/b }` | 两侧 |
| F | ASSERT | `ASSERT(div != 0)` | Kernel（仅 moe/ 族） |

> **ASSERT 注意**：ASSERT 在 Release 编译中可能被移除，仅在 moe/ 算子族的 Kernel 代码中视为有效守卫。其他场景的 ASSERT 降级为 SUSPICIOUS。

**【CeilDiv/CeilDivide 特殊说明】**

`CeilDiv(a, b)` / `CeilDivide(a, b)` 是算子仓最广泛使用的除法工具函数（3,000+ 处），但其标准实现 `(a + b - 1) / b` **本身不提供零值保护**。

- **禁止**将 `CeilDiv` 调用视为守卫模式
- `CeilDiv` 的除数参数（第二个参数）仍需按 P0-P4 分类判定：
  - 来自 P0/P1/P2 → PASS
  - 来自 P3/P4 且有守卫 → PASS
  - 来自 P3/P4 无守卫 → FAIL

**【Tiling 侧硬件参数校验 — 严格模式】**

Tiling 侧负责所有硬件参数的校验（业务约定）。当硬件 API 返回值（P1）用作除数时，**必须**在 Tiling 代码中有显式零值守卫，否则判定为 FAIL。

| 除数来源 | 校验方式 | 示例 |
|---------|---------|------|
| `GetCoreNumAic/Aiv()` | `OP_CHECK_IF(aicNum == 0, return GRAPH_FAILED)` | 核数获取后立即校验 |
| `GetCoreMemSize()` | `OP_CHECK_IF(ubSize == 0, return GRAPH_FAILED)` | 内存大小获取后立即校验 |
| `context->GetBlockDim()` | `if (blockDim == 0) return GRAPH_FAILED` | 使用前校验 |

**【Tiling 侧校验示例】**

```cpp
// Tiling 阶段校验外部输入非零（P3）
OP_CHECK_IF(keyShape->GetStorageShape().GetDim(DIM_2) == 0,
           OP_LOGE(context_, "dim N2 is 0."), return ge::GRAPH_FAILED);
fBaseParams.g = queryShape->GetStorageShape().GetDim(DIM_2) /
                keyShape->GetStorageShape().GetDim(DIM_2);
OP_CHECK_IF(fBaseParams.g == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);

// Tiling 阶段校验硬件参数非零（P1 严格模式）
totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
if (totalCoreNum_ == 0UL) {
    OP_LOGE(context_->GetNodeName(), "coreNum is 0");
    return ge::GRAPH_FAILED;
}

// Tiling 阶段校验设计过程参数非零（P4）
uint32_t tileSize = ComputeTileSize(totalSize, coreNum);
OP_CHECK_IF(tileSize == 0, OP_LOGE(context_, "tileSize is 0"), return ge::GRAPH_FAILED);
uint32_t loopTimes = totalSize / tileSize;
```

**【Kernel 侧校验示例】**

```cpp
// ✅ Kernel 侧排除规则 — 硬件 API 除数，自动 PASS（P1）
uint32_t coreIdx = GetBlockIdx();
uint32_t coreNum = GetBlockNum();     // P1 白名单
uint32_t taskIdx = coreIdx / coreNum; // 无需守卫

// ✅ Kernel 侧排除规则 — TilingData 除数，自动 PASS（P2）
uint32_t tileSize = tilingData->tileSize;  // P2 TilingData
uint32_t loops = totalSize / tileSize;     // 无需守卫

// ✅ Kernel 侧需校验 — 运行时动态值（P3）
GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
if ((actS1Size == 0) || (actS2Size == 0)) {
    curActSeqLenIsZero = true;
    return;  // 早期退出，避免后续除法
}
// 后续计算：loopTimes = actS1Size / mBaseSize（actS1Size 已确保非零）
```

**【描述】**
整数的除法和取余运算的第二个操作数值为0会导致程序产生未定义的行为，因此使用时要确保整数的除法和余数运算不会导致除零错误。

---

#### 2 外部数据作为数组索引时必须确保在数组大小范围内 `[适用: All]`

> **Kernel 侧说明**：Kernel 中使用 blockIdx、tileLength 等变量访问 GM/UB，需确保索引不越界。

**【Kernel 侧排除规则】**

以下情况在 Kernel 侧自动排除，无需校验：

| 排除条件 | 参数模式示例 | 排除原因 |
|---------|-------------|----------|
| 索引来自 TilingData | `constInfo.*`, `baseInfo.*` | Tiling 阶段已校验范围（如 Shape 维度校验） |
| 循环边界内索引 | `for (i = 0; i < bound; i++)` 内的 `arr[i]` | 循环条件保证索引在范围内 |
| GM/UB Buffer 内偏移 | `gmTensor[offset]`，offset 来自 Tiling | Tiling 阶段计算偏移范围 |

**判定方法**：
- 识别索引变量名匹配 `constInfo.*|baseInfo.*` 时，直接判定为 PASS
- 识别索引在循环边界内使用时，直接判定为 PASS

**【Kernel 侧需校验场景】**

以下情况在 Kernel 侧仍需校验：

| 校验条件 | 参数来源 | 代码模式 |
|---------|---------|----------|
| aiCoreIdx 核索引 | `GetBlockIdx()` 运行时获取 | `if (aiCoreIdx >= usedCoreNum) { return; }` |
| bIdx batch 累积差值边界 | TND 布局 `actualSeqLen[bIdx] - actualSeqLen[bIdx-1]` | `if (bIdx > 0) { ... } else { return actualSeqLen[0]; }` |
| 动态计算的偏移 | 运行时计算值 | 边界判断逻辑 |

**【Tiling 侧校验示例】**

```cpp
// Tiling 阶段校验 Shape 维度范围
OP_CHECK_IF(shape->GetDimNum() != expectedDim, 
           OP_LOGE(context_, "dim num mismatch"), return ge::GRAPH_FAILED);
OP_CHECK_IF(shape->GetDim(i) > MAX_SIZE,
           OP_LOGE(context_, "dim %d exceeds limit", i), return ge::GRAPH_FAILED);
```

**【Kernel 侧校验示例】**

```cpp
// Kernel 核索引范围校验
if (aiCoreIdx >= tilingData->baseParams.usedCoreNum) {
    if ASCEND_IS_AIV {
        SyncAll();  // superkernel 同步
    }
    return;  // 超范围核退出
}

// Kernel TND 布局累积差值边界处理
if (bIdx > 0) {
    return actualSeqLen[bIdx] - actualSeqLen[bIdx - 1];  // 累积差值
} else {
    return actualSeqLen[0];  // 首元素，避免访问 bIdx-1
}
```

**【描述】**
外部数据作为数组索引对内存进行访问时，必须对数据的大小进行严格的校验，确保数组索引在有效范围内，否则会导致严重的错误。

**【正确代码示例】**

```cpp
#define DEV_NUM 10
static Dev devs[DEV_NUM];

int set_dev_id(size_t index, int id)
{
    if (index >= DEV_NUM) {
        ... // 错误处理
    }
    devs[index].id = id;
    return 0;
}
```

### 3 确保有符号整数运算不溢出 `[适用: All]`

> **Kernel 侧说明**：Kernel 中使用 `uint32_t` 等固定宽度类型进行循环索引和 Buffer 偏移计算，需防止溢出。

**【描述】**
有符号整数溢出是未定义的行为。出于安全考虑，对外部数据中的有符号整数值在如下场景中使用时，需要确保运算不会导致溢出：

- 指针运算的整数操作数(指针偏移值)
- 数组索引
- 变长数组的长度(及长度运算表达式)
- 内存拷贝的长度
- 内存分配函数的参数
- 循环判断条件

在精度低于int的整数类型上进行运算时，需要考虑整数提升。程序员还需要掌握整数转换规则，包括隐式转换规则，以便设计安全的算术运算。

**乘法示例（int32_t 乘法溢出）：**

```cpp
// 错误写法 — 两个 int32_t 相乘，结果可能超出 int32_t 范围
int32_t calcHeightAlign = GetAlignedSize(...);  // 对齐后高度，可达 65536
int32_t calcWidth = GetWidth(...);              // 宽度，可达 65536
int32_t size = calcHeightAlign * calcWidth;     // 65536 × 65536 = 4,294,967,296 溢出！

// 正确写法 — 提升为 int64_t 计算
int64_t size = static_cast<int64_t>(calcHeightAlign) * calcWidth;
```

**取反示例（INT64_MIN 取反溢出，红线问题）：**

```cpp
// 错误写法 — delta 取 INT64_MIN 时，-delta 溢出
int64_t delta = input2 - input1;       // 可能为 INT64_MIN = -9223372036854775808
int64_t absDelta = -delta;             // -(-9223372036854775808) = 9223372036854775808 > INT64_MAX!
// 有符号整数溢出是未定义行为（C++ 红线）

// 正确写法 — 转换为无符号类型后再求绝对值
uint64_t absDelta = (delta < 0) ? static_cast<uint64_t>(-delta) : static_cast<uint64_t>(delta);
```

**多维连乘示例（多维 shape 连续累乘溢出）：**

```cpp
// 错误写法 — 多维 shape 用 int32_t 连乘，极易溢出
int32_t totalSize = dim0 * dim1 * dim2 * dim3 * dim4;
// dim0=1024, dim1=1024, dim2=128, dim3=64 时积 ≈ 8.6 × 10^9 > INT32_MAX

// 正确写法 — 使用 int64_t 并提前提升
int64_t totalSize = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3 * dim4;
```

**【检视策略 — 工具驱动】**

核心流程：运行 clause.check_bounds.py → 读取敏感性分析 → 按行动指引验证关键边界 → 必要时重跑 → 收敛结论

**Step 1 — 提取表达式与类型**

扫描代码，提取每个有符号算术表达式。识别操作数的 C++ 类型。

**Step 2 — 首次工具运行**

为操作数设定初始边界后运行 clause.check_bounds.py：

边界设定规则：
① 编译期常量 / 代码守卫 (if/assert) → 使用精确值
② 从赋值链推导 → 使用推导范围
③ 无代码证据 → 使用合理保守值（禁止用类型全范围——那必定违规，无意义）

禁止行为：
- 虚构变量关系作为安全证据（如声称 "X ≤ Y" 但找不到对应代码行）
- 用类型标签代替边界（"int64_t 所以够大不会溢出"——int64_t 的值可以是 1）

```bash
python3 {skill_base}/scripts/clause.check_bounds.py \
  --expr "{表达式}" \
  --vars "a=int32_t:0:47" "b=int32_t:3:3" "c=int64_t:100:1000000" \
  --check overflow
```

表达式中的 C++ 写法（`func()`、`a->b`）直接用作变量名。

**Step 3 — 按工具输出行动**

工具输出包含「边界敏感性分析」逐变量标注安全临界值，以及「行动指引」分步指令。**严格按行动指引执行，不要跳过。**

【输出 SAFE】
  看「最敏感变量」及余量：找出余量最小的那个变量
    余量 > 10x 临界值 → 安全余量充足，PASS
    余量 ≤ 10x → 回代码核实该变量的边界是否来自 A/B 级代码证据
      有证据 → PASS。无证据 → 向不利方向放宽边界重跑，重跑后判断

【输出 VIOLATION】
  看反例中「触及上限/下限」的变量：
    来自 constexpr/守卫 (A 级) → 边界可靠，确认 FAIL
    来自推测 (B/C 级) → Grep 找该变量的真实限定值
      找到 → 修正边界重跑。找不到 → SUSPICIOUS + 标注边界不确定

**Step 4 — 收敛（最多 1 次重跑）**

重跑后按 Step 3 逻辑判断。仍不确定 → SUSPICIOUS + 标注关键变量及缺失的代码证据。

---

### 4 确保无符号整数运算不回绕 `[适用: All]`

> **Kernel 侧说明**：Kernel 中大量使用 `uint32_t` 进行 tileLength、blockLength 计算，需防止回绕。

**【描述】**
涉及无符号操作数的计算永远不会溢出，因为超出无符号整数类型表示范围的计算结果会按照（结果类型可表示的最大值 + 1）的数值取模。这种行为更多时候被非正式地称为无符号整数回绕。

**乘法示例（uint32_t 乘法回绕后再 cast uint64_t——值已经错了）：**

```cpp
// 错误写法 — 乘法在 uint32_t 完成，回绕发生后才 cast 到 uint64_t，无法恢复
uint32_t blockSize = 65536;    // 来自 TilingData
uint32_t strideKV = 65536;     // 来自 TilingData
uint64_t result = blockSize * strideKV;
// blockSize * strideKV 在 uint32_t 空间计算：65536 × 65536 = 4,294,967,296 > UINT32_MAX
// 实际结果: (65536 × 65536) mod 2^32 = 0 → 回绕后的 0 再 cast 到 uint64_t = 0

// 正确写法 — 乘法前至少一个操作数提升为 uint64_t
uint64_t result = static_cast<uint64_t>(blockSize) * strideKV;
```

**减法示例（uint32_t 减法回绕——结果用作数组索引）：**

```cpp
// 错误写法 — aivIdx * singleCoreSize 可能大于 totalOutputSize，减法回绕
uint32_t tailSize = totalOutputSize - aivIdx * singleCoreSize;
// totalOutputSize=100, aivIdx=47, singleCoreSize=3:
//   47 × 3 = 141, 100 - 141 按 uint32_t 计算 = 4294967255（回绕）
//   tailSize 被误认为合法大小，后续 DataCopy 搬运 4GB 数据 → 越界崩溃

// 正确写法 — 先判断大小关系，或使用 int64_t 中间结果
int64_t tailSizeSigned = static_cast<int64_t>(totalOutputSize) - 
                         static_cast<int64_t>(aivIdx) * singleCoreSize;
uint32_t tailSize = (tailSizeSigned > 0) ? static_cast<uint32_t>(tailSizeSigned) : 0;
```

**类型混合示例（size_t 与 int64_t 混合运算——负数回绕成极大值）：**

```cpp
// 错误写法 — N_ALIGN 是 size_t 常量（无符号），numIters 是 int64_t
// 按 C++ 整型提升规则 int64_t → size_t，负数变成极大正数
constexpr size_t N_ALIGN = 128;
int64_t normSize = N_ALIGN * DOUBLE_SIZE * numIters * T * n0;
// 若 numIters 为 0 或负值，提升为 size_t 后回绕成 2^64-127 级别的极大值
// 再经 SetDim 传出，得到非预期的 shape，后续所有计算均错

// 正确写法 — 统一为有符号类型
constexpr int64_t N_ALIGN = 128;
int64_t normSize = N_ALIGN * DOUBLE_SIZE * numIters * T * n0;
```

**【检视策略 — 工具驱动】**

核心流程：运行 clause.check_bounds.py → 读取敏感性分析 → 按行动指引验证关键边界 → 必要时重跑 → 收敛结论

**Step 1 — 提取表达式与类型**

扫描代码，提取每个无符号算术表达式（减法、乘法、混合运算）。识别操作数的 C++ 类型。

**Step 2 — 首次工具运行**

为操作数设定初始边界后运行 clause.check_bounds.py：

边界设定规则：
① 编译期常量 / 代码守卫 (if/assert) → 使用精确值
② 从赋值链推导 → 使用推导范围
③ 无代码证据 → 使用合理保守值（禁止用类型全范围——那必定违规，无意义）

禁止行为：
- 虚构变量关系作为安全证据（如声称 "a ≥ b 恒成立" 但找不到对应代码行）
- 用类型标签代替边界（"uint64_t 所以够大不会回绕"——uint64_t 的值可以是 0）

```bash
python3 {skill_base}/scripts/clause.check_bounds.py \
  --expr "{表达式}" \
  --vars "a=uint32_t:0:47" "b=uint32_t:3:3" "c=int64_t:100:1000000" \
  --check wraparound
```

表达式中的 C++ 写法（`func()`、`a->b`）直接用作变量名。

**Step 3 — 按工具输出行动**

工具输出包含「边界敏感性分析」逐变量标注安全临界值，以及「行动指引」分步指令。**严格按行动指引执行，不要跳过。**

【输出 SAFE】
  看「最敏感变量」及余量：找出余量最小的那个变量
    余量 > 10x 临界值 → 安全余量充足，PASS
    余量 ≤ 10x → 回代码核实该变量的边界是否来自 A/B 级代码证据
      有证据 → PASS。无证据 → 向不利方向放宽边界重跑，重跑后判断

【输出 VIOLATION】
  看反例中「触及上限/下限」的变量：
    来自 constexpr/守卫 (A 级) → 边界可靠，确认 FAIL
    来自推测 (B/C 级) → Grep 找该变量的真实限定值
      找到 → 修正边界重跑。找不到 → SUSPICIOUS + 标注边界不确定

**Step 4 — 收敛（最多 1 次重跑）**

重跑后按 Step 3 逻辑判断。仍不确定 → SUSPICIOUS + 标注关键变量及缺失的代码证据。

---

### 5 禁止使用未初始化的变量 `[适用: All]`

> **Kernel 侧说明**：Kernel 模板类的成员变量必须在 `Init()` 函数中初始化，UB Buffer 通过 `AllocTensor` 获取后才能使用。

这里的变量，指的是局部动态变量，并且还包括内存堆上申请的内存块。因为他们的初始值都是不可预料的，所以禁止未经有效初始化就直接读取其值。

```cpp
void foo(...)
{
    int data;
    bar(data); // 错误：未初始化就使用
    ...
}
```

### 6 指针操作，使用前必须要判空 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `GlobalTensor` 和 `LocalTensor` 通过 API 获取，一般不需要判空，但 GM 地址偏移需校验。

**【描述】**
解引用空指针会导致程序产生未定义行为，通常会造成程序异常终止。

- 指针变量在使用前，一定要做好初始化的赋值，严禁对空指针进行访问
- 对于指针所代表的地址空间的任何操作，一定要保证空间的有效性
- 指针指向的内存释放后，需要调用者将指针显式置为NULL，防止"野指针"

### 7 需要避免数据竞争 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `GlobalTensor` 和 `LocalTensor` 通过 API 获取，一般不需要判空，但 GM 地址偏移需校验。

**【描述】**
数据竞争是一种未定义行为，它的发生通常是线程之间需要要通信，如：一个线程修改或者读取一个资源（比如共享内存、全局变量等），同时另外一个线程也在修改这个资源。如果线程是独立的，则不会导致数据竞争。在开发多线程或多进程代码过程中，如果涉及对同一个资源的操作，必须考虑数据竞争问题。

- 锁的职责尽量单一，每个锁只锁一个唯一共享资源/变量/内存，范围尽量小，只锁对应资源的操作代码，从而避免死锁和多线程执行效率低的问题。
- 使用锁解决数据竞争问题时，要避免死锁，死锁最明显的漏洞是拒绝服务，在某些情况下，在密集循环中有锁检查时，会出现cpu消耗过高的情况。
- c++中避免数据竞争的机制包括：std::atomic变量和原子内存操作，收益线程的创建和jion，及使用锁（mutex,shared mutex）、条件变量等。

**错误示例**
```cpp
// ❌ 错误：g_data可能有数据竞争
static long g_data;
void IncreaseData(long icData)
{
    g_data += icData;
}

void DecreaseData(long icData)
{
    g_data -= icData;
}
```

**正确示例**

```cpp
// ✅ g_data无数据竞争
atomic_long g_data;
void IncreaseData(long icData)
{
    g_data += icData;
}

void DecreaseData(long icData)
{
    g_data -= icData;
}
```


### 8 资源泄露（内存、句柄、锁等） `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态内存、无锁、无句柄，Buffer 静态分配无需释放。

**【描述】**

- 资源申请和释放必须匹配，包括：内存类的malloc/free/alloc_page/free_page, 锁lock/unlock、文件open/close等
- 释放结构体/类/数组/各类数据容器指针前，必须先释放成员指针
- 对外接口处理涉及资源申请但未释放，引起资源泄露，导致拒绝服务
- C++捕获异常时确保恢复程序的一致性; 建议使用RAII模式，确保资源在异常发生时自动释放

### 9. gm内存偏移或大小必须用int64表示 `[适用: Kernel]`

> **交叉引用**：整数溢出检视策略参见本条例3、条例4。

**问题说明**

涉及 GM（Global Memory）内存偏移或者大小必须用 `int64_t` 表示。GM 地址空间可能很大，使用 `int32_t` 可能导致溢出。

> **Kernel 侧说明**：Kernel 中使用 `GM_ADDR` 和 `GlobalTensor`，偏移量计算需用 `int64_t` 防止大地址溢出。

**典型溢出场景**

多维张量的 GM 偏移量在大模型场景下极易超过 `uint32_t` 上限（~4GB）：

- batch=32, heads=32, seqLen=8192, headDim=128, FP16 → 32×32×8192×128×2 = **54GB**

**隐蔽错误**

即使最终变量声明为 `int64_t`，若右侧乘法的操作数全为 `uint32_t`，乘法先以 `uint32_t` 计算并溢出，再转换为 `int64_t` 也于事无补。

**错误示例**

```cpp
// ❌ 错误：int32_t 可能溢出
int32_t totalLength = shape[0] * shape[1];  // 大 shape 可能溢出

// ❌ 隐蔽错误：赋给 int64_t，但右侧先以 uint32_t 溢出
int64_t offset = batchIdx * numHeads * seqLen * headDim;
//               ↑ 四个 uint32_t 相乘，先 overflow 再赋值，结果仍错
```

**正确示例**

```cpp
// ✅ Host侧 Tiling：使用 int64_t
int64_t totalLength = shape[0] * shape[1] * shape[2];

// ✅ Kernel侧：类成员变量使用 int64_t
int64_t blockLength_ = 0;
inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * AscendC::GetBlockIdx(), blockLength_);

// ✅ 多维偏移：强转第一个操作数，后续自动提升
int64_t offset = (int64_t)batchIdx * numHeads * seqLen * headDim;

// ✅ 等效：声明变量时直接用 int64_t
int64_t batchOffset = (int64_t)batchIdx * numHeads;
int64_t offset = batchOffset * seqLen * headDim;
```

**检视规则**

表达式中有 2 个及以上维度相乘时，检查第一个操作数是否已显式转换为 `int64_t`。

---

### 10 在支持superKernel场景，禁止直接对GM地址进行读写，需要改用GetValue或SetValue `[适用: Kernel]`

> **Kernel 侧说明**：Kernel 中 `GlobalTensor` 和 `LocalTensor` 通过 API 获取，一般不需要判空，但 GM 地址偏移需校验。

**【描述】**
在superKernel场景，直接对GM地址进行读写不保证精度，需要改用GetValue或SetValue，因为superKernel场景编译器不会在算子末尾添加DataCacheCleanAndInvalid指令，不会刷新整个Dcache(数据缓存)。

**错误示例**
```cpp
// ❌ 错误：直接通过指针访问GM地址
auto scale = *(__gm__ float*)scaleGm;
```

**正确示例**

```cpp
// ✅ 改用GetValue访问GM地址
auto scaleGlobal = GlosbalTensor<float>(...);
auto scale = scaleGlobal.GetValue(0);
```
