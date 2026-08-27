# Ascend C TOPK 问题清单

<适用>
语言: C++
侧别: All, Host, Kernel
领域: true
触发: 必须触发
默认启用: true

适用场景: Ascend C 算子开发中实际高频出现的编码问题，来源于生产环境经验总结
介绍: Ascend C TOPK 高频问题清单，14 条条款覆盖算子开发中最常触发的错误模式
类别(All): 野指针(局部变量指针生命周期)、特殊值(nan/inf/+0/-0边界处理)、整数vs浮点(可整数计算时禁止转浮点)、宏定义命名冲突
类别(Host): 返回值校验(函数返回值必须检查)、Dtype/Shape获取(GetInputDesc+context)、属性获取(context获取禁止CompileInfo传递)、属性类型一致性(与ir原型一致)、外部输入校验(融合规则/InferShape/Tiling)、dlopen管理(thread_local禁用)
类别(Kernel): atomic累加(src(ub)与dst(gm)双清零)、通信算子融合(核间同步)
> **说明**：TOPK 问题是检视实践中发现的高频风险点，需重点关注。条款标注适用范围：`[适用: All]` / `[适用: Host]` / `[适用: Kernel]`
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 3
</检视负载>

## 快速索引

### 两者都适用 `[适用: All]`（4 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 3 | 生命周期内使用局部变量指针，避免野指针 | 内存安全 | 高 |
| 6 | 必须考虑nan/inf/+0/-0等特殊值和边界值处理 | 数值安全 | 高 |
| 9 | 可整数计算时不允许转浮点数计算 | 数值安全 | 中 |
| 11 | 宏定义中临时变量命名不能和外部变量冲突 | 编码规范 | 中 |

### 仅 Host 侧适用 `[适用: Host]`（7 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 1 | 必须校验函数返回值 | 编码规范 | 高 |
| 2 | 使用GetInputDesc获取Dtype，context获取Shape | API使用 | 高 |
| 4 | 属性从context获取，禁止CompileInfo传递 | API使用 | 高 |
| 5 | 属性获取类型需与ir原型一致 | 类型安全 | 高 |
| 7 | 融合规则/InferShape/Tiling外部输入校验 | 输入验证 | 高 |
| 12 | dlopen管理的so禁用thread_local | 并发安全 | 高 |
| 13 | 在使用SyncAll的场景，需要使用SetScheduleMode设置BATCH_MODE_SCHEDULE | 并发安全 | 高 |

### 仅 Kernel 侧适用 `[适用: Kernel]`（3 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 8 | atomic累加需src(ub)与dst(gm)清零处理 | 内存安全 | 高 |
| 10 | 通信算子融合需核间同步 | 并发安全 | 高 |
| 14 | 在支持superKernel场景，调用全核同步API（SyncAll、CrossCoreWaitFlag）时禁止做BlockDim的限制 | 并发安全 | 高 |

---

## 详细规范

### 1. 必须校验函数返回值 `[适用: Host]`

> **交叉引用**：涉及指针返回值的判空检视策略，参见 ascendc-red-line.md 条例6。

**问题说明**

Host侧代码调用函数时，必须校验函数返回值，确保操作成功或正确处理错误情况。未校验返回值可能导致程序在错误状态下继续执行，引发后续问题。

**正确示例**

```cpp
auto ret = context->GetAttrPointer<int64_t>(ATTR_INDEX);
if (ret == nullptr) {
    return ge::GRAPH_FAILED;
}
int64_t attrValue = *ret;
```

**错误示例**

```cpp
auto ret = context->GetAttrPointer<int64_t>(ATTR_INDEX);
int64_t attrValue = *ret;  // 未校验 ret 是否为 nullptr，可能解引用空指针
```

---

### 2. 使用GetInputDesc获取Dtype，context获取Shape `[适用: Host]`

**问题说明**

Host侧代码不要使用 `GetInputTensor` 获取 Shape 和 Dtype。应使用 `GetInputDesc` 获取的对象来获取 Dtype，使用 context 获取对应的 Shape。`GetInputTensor` 在某些场景下可能返回无效数据。

**正确示例**

```cpp
// 获取 Dtype
auto inputDesc = context->GetInputDesc(0);
ge::DataType dtype = inputDesc->GetDataType();

// 获取 Shape
auto shape = context->GetInputDesc(0)->GetShape().GetDims();
```

**错误示例**

```cpp
// 错误：使用 GetInputTensor 获取 Dtype
auto tensor = context->GetInputTensor(0);
ge::DataType dtype = tensor->GetDataType();  // 不推荐
```

---

### 3. 生命周期内使用局部变量指针，避免野指针 `[适用: All]`

> **交叉引用**：未初始化导致的野指针检视策略，参见 ascendc-red-line.md 条例5；指针判空策略参见 ascendc-red-line.md 条例6。

**问题说明**

在生命周期内使用局部变量指针时，必须确保指针指向的对象在作用域内有效。返回局部变量指针或在作用域外使用局部变量指针会导致野指针问题。

> **Kernel 侧说明**：Ascend C 的 `LocalTensor` 由 `TQue`/`TBuf` 管理生命周期，通过 `AllocTensor`/`FreeTensor` 配对使用。如果在 `FreeTensor` 后继续使用，或在不同核函数间传递，可能导致野指针问题。

**正确示例**

```cpp
// Host 侧
int64_t ComputeLength(const std::vector<int64_t>& shape) {
    int64_t length = 1;
    for (auto dim : shape) {
        length *= dim;
    }
    return length;  // 返回值，而非指针
}

// Kernel侧
AscendC::LocalTensor<float> xLocal = inputQueue.AllocTensor<float>();
AscendC::DataCopy(xLocal, inputGM[0], tileLength_);
inputQueue.EnQue(xLocal);
// 在 FreeTensor 前使用完毕
AscendC::LocalTensor<float> yLocal = inputQueue.DeQue<float>();
AscendC::Add(zLocal, yLocal, xLocal, tileLength_);
inputQueue.FreeTensor(yLocal);  // 使用后释放
```

**错误示例**

```cpp
// Host侧：返回局部变量指针
int64_t* GetLengthPointer(const std::vector<int64_t>& shape) {
    int64_t length = 1;
    for (auto dim : shape) {
        length *= dim;
    }
    return &length;  // 返回局部变量指针，函数返回后变为野指针
}

// Kernel侧：FreeTensor 后继续使用
AscendC::LocalTensor<float> xLocal = inputQueue.AllocTensor<float>();
inputQueue.FreeTensor(xLocal);
AscendC::Add(zLocal, xLocal, yLocal, tileLength_);  // 已释放，野指针
```

---

### 4. 属性从context获取，禁止CompileInfo传递 `[适用: Host]`

**问题说明**

Host侧代码获取属性必须从 context 获取，不允许使用 CompileInfo 传递算子属性值。CompileInfo 主要用于传递编译期确定的配置信息，而非运行时属性。

**正确示例**

```cpp
auto attrs = context->GetAttrs();
auto* attrValue = attrs->GetAttrPointer<int64_t>(ATTR_INDEX);
```

**错误示例**

```cpp
// 错误：通过 CompileInfo 传递属性
compileInfo->GetAttrValue(ATTR_INDEX);  // 不推荐
```

---

### 5. 属性获取类型需与ir原型一致 `[适用: Host]`

**问题说明**

Host侧代码属性的获取，要用 ir 原型对应的类型去获取。`GetAttrPointer<T>` 的类型参数必须与 ir 原型定义的属性类型一致，否则会导致类型错误。

**示例说明**

假设 `swiglu_mx_quant_def.cpp` 文件中属性定义如下：
- `activate_dim` 是 `Int` → 使用 `int64_t`
- `activate_left` 是 `bool` → 使用 `bool`
- `clamp_limit` 是 `float` → 使用 `float`
- `round_mode` 是 `string` → 使用 `char`

**正确示例**

```cpp
auto attrs = context->GetAttrs();
auto* attrActivateDim = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_ACTIVATE_DIM);   // Int → int64_t
auto* attrActivateLeft = attrs->GetAttrPointer<bool>(INDEX_ATTR_ACTIVATE_LEFT);     // bool → bool
auto* attrClampLimit = attrs->GetAttrPointer<float>(INDEX_ATTR_CLAMP_LIMIT);        // float → float
const char* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);     // string → char
```

**错误示例**

```cpp
auto attrs = context->GetAttrs();
auto* attrActivateDim = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_ACTIVATE_DIM);   // 错误：Int 应为 int64_t
auto* attrActivateLeft = attrs->GetAttrPointer<int>(INDEX_ATTR_ACTIVATE_LEFT);      // 错误：bool 应为 bool
```

---

### 6. 必须考虑nan/inf/+0/-0等特殊值和边界值处理 `[适用: All]`

> **交叉引用**：除零检视策略参见 ascendc-red-line.md 条例1。

**问题说明**

算子设计时必须考虑 `nan`、`inf`、`-inf`、`+0`、`-0` 等特殊值和边界值处理。这些特殊值在计算中可能产生非预期结果，导致算子输出异常。

> **Kernel 侧说明**：Ascend C 计算指令对特殊值的处理可能与 CPU 不同，需特别注意 FP16/BF16 精度下的边界值行为。

**关注场景**

- 除法运算：除数为0、nan、inf
- 比较运算：nan 与任何值比较结果
- 类型转换：FP32 转 FP16 时的溢出（超出 FP16 范围变成 inf）
- 归约运算：输入包含 nan 时输出结果

---

### 7. 融合规则/InferShape/Tiling外部输入校验 `[适用: Host]`

> **交叉引用**：除零检视策略参见 ascendc-red-line.md 条例1；外部输入合法性校验策略参见 cpp-secure.md SEC-3.1。

**问题说明**

融合规则、InferShape、Tiling 的外部输入使用时必须进行合法性校验。外部输入可能包含非法值，未校验可能导致后续计算错误或程序崩溃。

**校验内容**

- Shape 维度是否合法（不为负数、不超过限制）
- Dtype 是否支持
- 属性值是否在有效范围内
- 指针是否为 nullptr

---

### 8. atomic累加需src(ub)与dst(gm)清零处理 `[适用: Kernel]`

**问题说明**

atomic 累加指令需要 src（UB）与 dst（GM）做清零处理。未清零可能导致累加结果包含初始垃圾值，影响计算精度。

**正确做法**

```cpp
// 清零 src (UB)
AscendC::SetTensor<float>(srcLocal, 0, srcLength);
// 清零 dst (GM) 或确保初始值为0
AscendC::DataCopy(dstGM[0], srcLocal, dstLength);  // 先清零
// 再执行 atomic 累加
AscendC::AtomicAdd(dstGM[0], srcLocal, srcLength);
```

---

### 9. 可整数计算时不允许转浮点数计算 `[适用: All]`

**问题说明**

可整数计算时不允许转浮点数计算。整数计算更精确且性能更好，必要时应转更高精度类型而非浮点数。

> **Kernel 侧说明**：Ascend C 的浮点运算单元与整数运算单元不同，不必要的浮点转换会增加计算开销和精度损失风险。

**正确示例**

```cpp
// 累加元素数量，使用整数
int64_t totalElements = batchSize * seqLen * headNum;  // 整数计算
```

**错误示例**

```cpp
float totalElements = (float)batchSize * (float)seqLen * (float)headNum;  // 不必要的浮点转换
```

---

### 10. 通信算子融合需核间同步 `[适用: Kernel]`

**问题说明**

涉及到和通信算子融合时，多轮计算和集合通信之间需要增加核间同步。不同核的执行速度可能不同，未同步可能导致数据竞争或结果不一致。

**同步方式**

```cpp
// 使用 Ascend C 提供的同步 API
AscendC::SyncAll();  // 核间全同步
```

---

### 11. 宏定义中临时变量命名不能和外部变量冲突 `[适用: All]`

**问题说明**

宏定义中临时变量命名不能和外部变量冲突。宏展开后，临时变量可能与作用域内其他变量同名，导致意外行为。

**正确示例**

```cpp
#define COMPUTE_SUM(a, b) ({ \
    int64_t _macro_sum_result = (a) + (b); \
    _macro_sum_result; \
})
```

**错误示例**

```cpp
#define COMPUTE_SUM(a, b) ({ \
    int64_t sum = (a) + (b);  // sum 可能与外部变量冲突 \
    sum; \
})
```

---

### 12. dlopen管理的so禁用thread_local `[适用: Host]`

**问题说明**

由 `dlopen`、`dlclose` 管理的 so，禁止使用 `thread_local` 申明变量。动态加载/卸载的 so 中使用 `thread_local`，在 `dlclose` 后可能导致访问已释放内存，引发崩溃。

**原因**

`thread_local` 变量的生命周期与线程绑定，而 `dlclose` 会卸载 so 的内存。如果线程仍在运行，后续访问 `thread_local` 变量时会访问已释放的内存区域。

**正确做法**

避免在会被 `dlopen/dlclose` 管理的动态库中使用 `thread_local` 变量。

### 13 在使用SyncAll的场景，需要使用SetScheduleMode设置BATCH_MODE_SCHEDULE `[适用: Host]`

> **Kernel 侧说明**：Kernel 中 `GlobalTensor` 和 `LocalTensor` 通过 API 获取，一般不需要判空，但 GM 地址偏移需校验。

**【描述】**
多流场景，如果kernel侧使用了SyncAll，但是tiling侧没有使用SetScheduleMode设置BATCH_MODE_SCHEDULE，会导致卡死问题。

**错误示例**
```cpp
// ❌ 错误：kernel侧使用了SyncAll，但是tiling侧未使用SetScheduleMode设置BATCH_MODE_SCHEDULE

// kernel1侧代码如下：
AscendC::SyncAll<false>();

// tiling侧代码如下：
constexpr uint32_t BATCH_MODE_DEFAULT = 0;
context_->SetScheduleMode(BATCH_MODE_DEFAULT);
```

**正确示例**

```cpp
// ✅ kernel侧使用了SyncAll，且tiling侧使用了SetScheduleMode设置BATCH_MODE_SCHEDULE

// kernel1侧代码如下：
AscendC::SyncAll<false>();

// tiling侧代码如下：
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
```

### 14 在支持superKernel场景，调用全核同步API（SyncAll、CrossCoreWaitFlag）时禁止做BlockDim的限制 `[适用: Kernel]`

> **Kernel 侧说明**：Kernel 中 `GlobalTensor` 和 `LocalTensor` 通过 API 获取，一般不需要判空，但 GM 地址偏移需校验。

**【描述】**
在superKernel场景开启的核数大于实际使用的核数时，如果算子只在部分核调用SyncAll，会导致部分核未set_flag,而全部核都在wait_flag,导致卡死。

**错误示例**
```cpp
// ❌ 错误：调用全核同步API（SyncAll）时做了BlockDim的限制
if (AscendC::GetBlockIdx() < AscendC::GetBlockNum()) {
    AscendC::SyncAll<false>();
    DataCopy(dstLocal[AscendC::GetBlockIdx() * countPerCore], srcGm[AscendC::GetBlockIdx() * countPerCore]);
}
```

**正确示例**

```cpp
// ✅ 调用全核同步API（SyncAll）时不做BlockDim的限制
AscendC::SyncAll<false>();
if (AscendC::GetBlockIdx() < AscendC::GetBlockNum()) {
    DataCopy(dstLocal[AscendC::GetBlockIdx() * countPerCore], srcGm[AscendC::GetBlockIdx() * countPerCore]);
}
```