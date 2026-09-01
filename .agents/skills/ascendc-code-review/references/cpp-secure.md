# CANN C++ 安全编码规范

<适用>
语言: C++
侧别: All, Tiling
领域: true
触发: 必须触发
默认启用: true

适用场景: Tiling 侧（Host 侧）和 Kernel 侧（Device 侧）的 C++ 安全编码规范
不适用场景: Python 代码（见 python-secure.md）、编译链接安全（见 compile-secure.md）
介绍: CANN C++ 安全编码规范，24 条条款覆盖 8 个安全类别
类别(All): 总体原则(类型安全/内存安全/未定义行为),, 内存与指针安全(sizeof误用), 输入验证(外部输入合法性/内存操作长度校验), 类与对象安全(非trivially copyable对象位操作), 标准库安全(敏感信息清零/结构体兼容性)
类别(Tiling): 内存与指针安全(资源句柄释放后赋值/字符串空间), 资源管理(申请成功判断/new-delete配对/分配错误处理), 安全函数使用(安全函数库/destMax参数/返回值检查), 标准库安全(空指针string/c_str指针保存), LOG规范(空指针/格式化占位符)
> **说明**：安全编码红线规范，所有代码必须 100% 遵守。条款标注适用范围：`[适用: All]` / `[适用: Tiling]`
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 10
</检视负载>

## 快速索引

### 两者都适用 `[适用: All]`（10 条）

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| 1.1 | 保证静态类型安全 | 总体原则 | 高 |
| 1.2 | 保证内存安全 | 总体原则 | 高 |
| 1.3 | 禁止使用未定义行为 | 总体原则 | 高 |
| 2.2 | 禁止通过对指针变量进行sizeof操作来获取数组大小 | 内存安全 | 中 |
| 3.1 | 外部输入合法性校验 | 输入验证 | 高 |
| 3.2 | 内存操作长度校验 | 输入验证 | 高 |
| 6.1 | 禁止逐位操作非 trivially copyable 对象 | 类与对象 | 中 |
| 7.3 | 敏感信息使用后清零 | 标准库 | 高 |
| 7.4 | 结构体字段末尾添加 | 标准库 | 中 |
| 7.5 | 接口变更考虑兼容性 | 标准库 | 中 |

### 仅 Tiling 适用 `[适用: Tiling]`（14 条）

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| 2.1 | 资源释放后指针置新值 | 内存安全 | 中 |
| 2.3 | 字符串存储有足够空间 | 内存安全 | 高 |
| 4.1 | 资源申请后判断是否成功 | 资源管理 | 高 |
| 4.2 | new/delete 配对使用 | 资源管理 | 高 |
| 4.3 | new 操作符错误处理 | 资源管理 | 高 |
| 5.1 | 使用安全函数替代危险函数 | 安全函数 | 高 |
| 5.2 | 正确设置安全函数 destMax 参数 | 安全函数 | 高 |
| 5.3 | 检查安全函数返回值 | 安全函数 | 高 |
| 7.1 | 禁止从空指针创建 std::string | 标准库 | 高 |
| 7.2 | 不要保存 c_str/data 指针 | 标准库 | 中 |
| 8.1 | LOG API 禁止传入空指针 | LOG API 安全 | 高 |
| 8.2 | LOG API 参数必须与格式化占位符逐位一致（数量、类型、顺序） | LOG API 安全 | 高 |
| 8.3 | LOG API 禁止传入已释放内存的指针 | LOG API 安全 | 高 |
| 8.4 | LOG 消息英语行文语法正确、表意清晰 | LOG API 规范 | 低 |

---

### 1. 总体原则

#### 1.1 保证静态类型安全 `[适用: All]`

> **Kernel 侧说明**：Ascend C 模板类需注意类型转换（如 half ↔ float）和范围错误（FP16 溢出）。

C++应该是静态类型安全的，这样可以减少运行时的错误，提升代码的健壮性。但是由于C++存在下面的特性，会破坏C++静态类型安全，针对这部分特性要仔细处理：

- 联合体
- 类型转换
- 缩窄转换
- 类型退化
- 范围错误
- void* 类型指针

可以通过约束这些特性的使用，或者使用C++的新特性，例如std::variant（C++17）、std::span（C++20）等来解决这些问题，提升C++代码的健壮性。

#### 1.2 保证内存安全 `[适用: All]`

> **Kernel 侧说明**：Ascend C 使用 UB（Unified Buffer）和 GM（Global Memory），需要通过 `DataCopy` API 安全访问，避免越界和未初始化访问。

C++语言的内存完全由程序员自己控制，所以在操作内存的时候必须保证内存安全，防止出现内存错误：

- 内存越界访问
- 释放以后继续访问内存
- 解引用空指针
- 内存没有初始化
- 把指向局部变量的引用或者指针传递到了函数外部或者其他线程中
- 申请的内存或者资源没有及时释放

建议使用更加安全的C++的特性，比如RAII，引用，智能指针等，来提升代码的健壮性。

#### 1.3 禁止使用编译器"未定义行为" `[适用: All]`

遵循ISO C++标准，标准中未定义的行为禁止使用。对于编译器实现的特性或者GCC等编译器提供的扩展特性也需要谨慎使用，这些特性会降低代码的可移植性。

---

### 2. 内存与指针安全

#### 2.1 指向资源句柄或描述符的变量，在资源释放后立即赋予新值 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态资源管理，Buffer 由 `InitBuffer` 静态分配，无需释放后置空。

**【描述】**
指向资源句柄或描述符的变量包括指针、文件描述符、socket描述符以及其它指向资源的变量。

以指针为例，当指针成功申请了一段内存之后，在这段内存释放以后，如果其指针未立即设置为NULL，也未分配一个新的对象，那这个指针就是一个悬空指针。如果再对悬空指针操作，可能会发生重复释放或访问已释放内存的问题，造成安全漏洞。

**【正确代码示例】**

```cpp
int foo(void)
{
    SomeStruct *msg = NULL;
    ... // 初始化msg->type，分配 msg->body 的内存空间

    if (msg->type == MESSAGE_A) {
        ...
        free(msg->body);
        msg->body = NULL;
    }

    ...
EXIT:
    ...
    free(msg->body);
    return ret;
}
```

#### 2.2 禁止通过对指针变量进行sizeof操作来获取数组大小 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `LocalTensor<T>` 通过 API（如 `GetSize()`）获取大小，不能用 sizeof。

**【描述】**
将指针当做数组进行sizeof操作时，会导致实际的执行结果与预期不符。

**【错误代码示例】**

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
// sizeof与预期不符，其结果为指针本身的大小而不是缓冲区大小
(void)memset(buffer, 0, sizeof(buffer));
```

**【正确代码示例】**

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
(void)memset(buffer, 0, SIZE); // 使用申请的缓冲区大小
```

#### 2.3 确保字符串存储有足够的空间容纳字符数据和null结束符 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 C 风格字符串处理。但 GM 数据搬运时需确保目标 Buffer 有足够空间。

**【描述】**
将数据复制到不足以容纳数据的缓冲区，会导致缓冲区溢出。

---

### 3. 输入验证

#### 3.1 外部输入数据需要做合法性校验 `[适用: All]`

> **Kernel 侧说明**：Kernel 中的 `TilingData` 参数（如 `constInfo.*`、`baseInfo.*`）已在 Tiling 阶段校验，无需重复校验。校验职责归属 Tiling 层。

**【Kernel 侧排除规则】**

以下情况在 Kernel 侧自动排除，无需校验：

| 排除条件 | 参数模式示例 | 排除原因 |
|---------|-------------|----------|
| 参数来自 TilingData | `constInfo.*`, `baseInfo.*`, `tilingData->*` | Tiling 阶段已校验（Shape、Dtype、范围、存在性） |
| __aicore__ 函数入参 | 模板类 Init/Process 参数 | 架构约定：尽量减少校验，有效性由调用者保证 |
| GM 指针可选输入 | `actualSeqLengths` 可能为 nullptr | 通过标志位 fallback 处理 |

**判定方法**：
- 识别参数变量名匹配 `constInfo.*|baseInfo.*|tilingData->*` 时，直接判定为 PASS
- 识别参数赋值来源为 `tilingData->xxx` 时，直接判定为 PASS
- 识别参数在 `__aicore__` 函数签名中时，不报告"输入验证缺失"

**【Kernel 侧需校验场景】**

以下情况在 Kernel 侧仍需处理（非"校验"，而是"分支处理"）：

| 处理条件 | 参数来源 | 代码模式 |
|---------|---------|----------|
| actualSeqLengths 可选输入 | GM 指针可能为 nullptr | `if (ptr != nullptr) { SetGlobalBuffer(ptr); }` |
| isActualLenDimsNull 标志位 | Tiling 传递 | `if (flag == 1) { return staticSize; } else { return gm[bIdx]; }` |
| 空 Tensor 专用 Kernel | ShapeSize == 0 | 专用模板 `FiaKernelEmptyTensor`，InitOutput 为 0 |

**【Tiling 侧校验示例】**

```cpp
// Tiling 阶段校验 Shape、Dtype、范围
OP_CHECK_IF(context_->GetInputDesc(QUERY) == nullptr,
           OP_LOGE(context_, "query desc is null"), return ge::GRAPH_FAILED);
OP_CHECK_IF(shape->GetDimNum() != expectedDim,
           OP_LOGE(context_, "dim num mismatch"), return ge::GRAPH_FAILED);
OP_CHECK_IF(headDim == 0,
           OP_LOGE(context_, "headDim is 0"), return ge::GRAPH_FAILED);

// Tiling 阶段校验参数组合存在性
ge::graphStatus FiaTilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer == nullptr,
        OP_LOGE(opName_, "%s should not be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

**【Kernel 侧处理示例】**

```cpp
// Kernel 可选 GM 指针条件处理（非"校验"，而是"分支处理"）
if (actualSeqLengthsQ != nullptr) {
    actualSeqQlenAddr = (__gm__ int32_t *)actualSeqLengthsQ;
}

// Kernel 标志位 fallback（Tiling 已传递 isActualLenDimsNull）
if (constInfo.isActualLenDimsNull == 1) {
    return constInfo.s1Size;  // 静态值 fallback
} else {
    return actualSeqQlenAddr[bIdx];  // 动态值
}
```

**【描述】**

- 外部输入数据需要做合法性校验且确保校验范围正确
- 边界接口需要对传入的地址做合法性校验避免任意地址读写
- 需要对入参进行合法性校验避免数组越界
- 需要对地址偏移校验避免任意地址读写
- 外部传入指针需要判空后使用
- 外部入参参与循环、递归条件的运算，必须严格校验边界和终止条件
- 文件路径来自外部数据时，必须对其做合法性校验

#### 3.2 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `DataCopy` 的搬运长度需校验，确保不超过 UB 容量和 GM 数据范围。

**【描述】**
将数据复制到容量不足以容纳该数据的内存中会导致缓冲区溢出。必须根据目标容量的大小限制被复制的数据大小，或者必须确保目标容量足够大以容纳要复制的数据。

---

### 4. 资源管理

#### 4.1 资源申请后必须判断是否成功 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态资源申请（malloc/new），Buffer 由 `InitBuffer` 静态分配，编译期确定。

**【描述】**
内存、对象、stream、notify等资源申请分配一旦失败，那么后续的操作会存在未定义的行为风险。

**【正确代码示例】**

```cpp
struct tm *make_tm(int year, int mon, int day, int hour, int min, int sec)
{
    struct tm *tmb = (struct tm *)malloc(sizeof(*tmb));
    if (tmb == NULL) {
        ... // 错误处理
    }
    tmb->year = year;
    ...
    return tmb;
}
```

#### 4.2 new和delete配对使用，new[]和delete[]配对使用 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 禁止 new/delete。

##### 专属检视方法

除 Tiling 侧（`op_host/`）外，**测试代码**（`tests/`、`ut/`、`st/` 目录下的 `.cpp/.h`）同样适用 new/delete 配对检查——测试代码中 `new` 创建的对象若无对应 `delete`，会导致内存泄漏。

检视指引：
- Grep 测试文件中的 `new ` 创建对象语句 → 追踪同一作用域或对象生命周期内是否有对应 `delete`
- 重点关注：测试函数内的局部 `new`（函数结束前未释放）、SetUp/TearDown 中的成员对象创建与销毁配对、`new[]` 是否用 `delete[]`（而非 `delete`）释放
- 豁免：智能指针（`std::unique_ptr`/`std::shared_ptr`/`std::make_unique`/`std::make_shared`）管理的对象、RAII 模式封装的对象、测试框架自动管理的 fixture 成员

#### 4.3 使用恰当的方式处理new操作符的内存分配错误 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 禁止 new。

---

### 5. 安全函数使用

#### 5.1 使用社区提供的安全函数库的安全函数，禁止使用内存操作类危险函数 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 memcpy_s/memset_s，使用 Ascend C API（如 `Duplicate`、`DataCopyPad`）。

| 函数类别 | 危险函数 | 安全替代函数 |
|---------|---------|------------|
| 内存拷贝 | memcpy或bcopy | memcpy_s |
| 内存拷贝 | memmove | memmove_s |
| 字符串拷贝 | strcpy | strcpy_s |
| 字符串串接 | strcat | strcat_s |
| 格式化输出 | sprintf | sprintf_s |
| 格式化输出 | snprintf | snprintf_s |
| 格式化输入 | scanf | scanf_s |
| 内存初始化 | memset | memset_s |

#### 5.2 正确设置安全函数中的destMax参数 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无安全函数。

#### 5.3 必须检查安全函数返回值，并进行正确的处理 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无安全函数。

原则上，如果使用了安全函数，需要进行返回值检查。如果返回值!=EOK, 那么本函数一般情况下应该立即返回，不能继续执行。

```cpp
{
    ...
    err = memcpy_s(destBuff, destMax, src, srcLen);
    if (err != EOK) {
        MS_LOG("memcpy_s failed, err = %d\n", err);
        return FALSE;
    }
    ...
}
```

---

### 6. 类与对象安全

#### 6.1 禁止逐位操作非trivially copyable对象 `[适用: All]`

> **Kernel 侧说明**：Kernel 模板类都是 POD 类型，可以使用 `Duplicate` 进行内存操作。

---

### 7. 标准库安全

#### 7.1 禁止从空指针创建std::string `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 std::string。

#### 7.2 不要保存std::string类型的 `c_str`和 `data`成员函数返回的指针 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 std::string。

#### 7.3 内存中的敏感信息使用完毕后立即清0 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 UB 数据可通过 `Duplicate` 清零，GM 数据需在 Host 侧处理。

口令、密钥等敏感信息使用完毕后立即清零，避免被攻击者获取。

#### 7.4 对外结构体接口新增字段时必须在结构体最后添加 `[适用: All]`

> **Kernel 侧说明**：`TilingData` 结构体新增字段需在末尾添加，保持 ABI 兼容性。

为了最大程度上在ABI层面的兼容，对外结构体接口添加新字段时必须在结构体最后添加。

#### 7.5 外部接口或数据结构变更必须考虑兼容性 `[适用: All]`

> **Kernel 侧说明**：Kernel 接口（如 TilingData 结构体）变更需考虑版本兼容性。

外部接口、接口参数、返回值、数据结构、消息字段等变更会引起版本兼容性问题，非必要不建议变更。

---

### 8. LOG 规范

> **适用范围**：仅 Tiling 侧（Host 侧）。Kernel 侧使用 `AscendC::PRINTF`，无下列风险。

Tiling 侧使用 `OP_LOGE` / `OP_LOGD` / `OP_LOGW` 等格式化 LOG 宏。11.1–11.3 为安全强制要求（防段错误/未定义行为），11.4 为质量建议。

LOG 宏签名（业务代码标准调用形式）：

```cpp
OP_LOGE(context->GetNodeName(), "format string %s %ld", arg1, arg2);
OP_LOGD(context->GetNodeName(), "format string %lu", arg1);
```

---

#### 8.1 LOG API 禁止传入空指针作为字符串参数 `[适用: Tiling]`

**【问题说明】**

`%s` 会解引用传入指针，若指针为 `nullptr`，将访问地址 0（受 OS 保护），导致段错误。Tiling 侧常见场景：从 `context` 获取 Desc/Attr 后未判空直接传入 LOG。

**错误示例**

```cpp
// 来自 quant_grouped_matmul_dequant_tiling.cpp 同类风险
auto inputDesc = context->GetInputDesc(0);
// 若 inputDesc 为 nullptr，GetDataType() 返回的字符串描述也可能为空
OP_LOGE(context->GetNodeName(),
        "input dtype: %s", ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType()).c_str());
// 风险：inputDesc 未判空就调用成员函数
```

**正确示例**

```cpp
auto inputDesc = context->GetInputDesc(0);
if (inputDesc == nullptr) {
    OP_LOGE(context->GetNodeName(), "GetInputDesc(0) returned nullptr, skip dtype log.");
    return ge::GRAPH_FAILED;
}
OP_LOGE(context->GetNodeName(),
        "input dtype: %s", ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType()).c_str());
```

---

#### 8.2 LOG API 参数必须与格式化占位符逐位一致（数量、类型、顺序） `[适用: Tiling]`

**【问题说明】**

LOG 宏的格式化占位符与实际参数之间必须满足三个维度的一致性：

1. **数量一致**：参数少于占位符时，从栈上读取垃圾值，若被解释为 `%s` 将触发段错误
2. **类型匹配**：类型大小不匹配时（如 `uint64_t` 误用 `%d`），按说明符宽度截断，后续参数全部错位
3. **顺序对应**：参数顺序与格式符位置不对应时（如 `%s` 位置收到整数），整数被当作地址读字符串 → **段错误(SIGSEGV)**

> **⚠️ 禁止仅凭 grep 单行分析 LOG 调用。** 算子仓中大量 LOG 语句跨越多行（2-35 行），且常嵌套在 `OP_CHECK_IF` 等外层宏内。grep 命中后**必须 Read 前后至少 10 行**获取完整的格式字符串和全部参数，否则分析的是截断的不完整调用，结论无效。多行字符串拼接（`"a" "b"`）需先合并再解析。

**错误与正确示例**

```cpp
// ❌ 数量不一致：2 个占位符，1 个参数
OP_LOGD(ctx, "M: %ld, K: %ld", m);           // 缺少 k
// ✅
OP_LOGD(ctx, "M: %ld, K: %ld", m, k);

// ❌ 类型不匹配：uint64_t 用了 %d
OP_LOGE(ctx, "n = %d, ubSize = %d\n", n, ubSize); // n/ubSize 均为 uint64_t
// ✅
OP_LOGE(ctx, "n = %llu, ubSize = %llu\n", n, ubSize);

// ❌ 顺序错位：数量=5 格式符=5，但位置1和3的参数放反了
//   格式符: %u(1) %u(2) %s(3) %u(4) %u(5)
//   参数:   inputName.c_str()(1) ... d0Size/NUM8(3) ...
//   → 位置1: %u 收到 const char*，位置3: %s 收到 uint → 段错误
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(opName, "...kvCache(%u)...%s(%u)...",
        inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size),
    return ge::GRAPH_FAILED);
// ✅ 参数顺序与格式符逐位对应
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(opName, "...kvCache(%u)...%s(%u)...",
        tempD0/NUM8, d0Size/NUM8, inputName.c_str(), tempD0, d0Size),
    return ge::GRAPH_FAILED);
```

**类型与说明符速查**

| 类型 | 正确 | 常见错误 | 后果 |
|------|------|---------|------|
| `uint64_t` | `%llu` | `%u`, `%lu`, `%d` | 截断为 32 位，后续参数错位 |
| `int64_t` | `%lld` | `%d`, `%ld` | 同上 |
| `uint32_t` | `%u` | `%d` | 大值显示为负数 |
| `size_t` | `%zu` | `%d`, `%u` | 64 位系统上截断 |
| `bool` | `%d` 或 `? "true":"false"` + `%s` | `%s` 直传 | 未定义行为 |
| `void*` | `%p` | `%x` | 不可移植 |

**【检视方法】**

1. grep `OP_LOGE\|OP_LOGD\|OP_LOGW\|OP_LOGI` 找到所有 LOG 调用
2. Read 完整调用后，提取格式符序列和参数序列，逐位比对：数量是否一致 → 每个位置的参数类型是否兼容格式符
3. 高风险标记：`%s` 收到整数（段错误）、`uint64_t`/`int64_t` 配 `%d`（截断错位）

---

#### 8.3 LOG API 禁止传入已释放内存的指针 `[适用: Tiling]`

**【问题说明】**

Tiling 侧手动管理的堆内存（`new` / `malloc`）释放后若仍传入 `%s`，行为未定义，大概率触发段错误。典型场景：在函数末尾统一释放资源，但 LOG 语句写在释放之后。

**错误示例**

```cpp
char* errMsg = new char[256];
snprintf(errMsg, 256, "tiling failed, M=%ld", _Params.originM);
delete[] errMsg;
OP_LOGE(context->GetNodeName(), "error: %s", errMsg);   // 野指针，已释放
```

**正确示例**

```cpp
char* errMsg = new char[256];
snprintf(errMsg, 256, "tiling failed, M=%ld", _Params.originM);
OP_LOGE(context->GetNodeName(), "error: %s", errMsg);   // 先记录
delete[] errMsg;
errMsg = nullptr;
```

---

#### 建议 8.4 LOG 消息的英语行文应语法正确、表意清晰 `[适用: Tiling]`

**【问题说明】**

LOG 消息是排障的第一手线索。语法错误或含义模糊的日志会显著增加定位问题的时间成本。

**检视要点**：
- 主谓一致、时态统一（LOG 消息惯用一般现在时或过去时）
- 避免中英文混杂（变量名除外）
- 避免无意义占位（如 "error error"、"fail to fail"）
- 关键数值应包含在消息中，而非仅靠格式符

**提醒示例**

```cpp
// "is not support" → "is not supported"（仓内高频错误模式，5+ 文件）
OP_LOGE(op_name, "scale shape is not support");          // → is not supported
OP_LOGE(opName_, "...layout BNSD/BNSD_NBSD is not support"); // → is not supported
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "...the soc verison is not support"); // → version; is not supported

// "do not support" → "does not support"（主谓不一致）
OP_LOGE(opName_, "...key layout do not support PA_BSND."); // → does not support

// 拼写错误
OP_LOGE(opName_, "...cu_seqlens_q's dtype msut be DT_INT32."); // msut → must

// 缺少主语
OP_LOGD("GetBlockInfoOfBNS4TND", " Not support BN2S2."); // → BN2S2 is not supported
```

> **检视级别**：仅标记 SUSPICIOUS，不标记 FAIL。