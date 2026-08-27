# CANN C++ 通用编码规范

<适用>
语言: C++
侧别: All, Tiling
领域: false
默认启用: true
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 10
</检视负载>

> **适用场景**：Tiling 侧（Host 侧）和 Kernel 侧（Device 侧）
>
> **说明**：通用编程规范，条款标注适用范围：`[适用: All]` / `[适用: Tiling]` / `[不适用]`

## 快速索引

### 两者都适用 `[适用: All]`（22 条）

| 规范编号 | 规范名称 | 类别 |
|---------|---------|------|
| 1.1 | 外部数据合法性检查 | 代码设计 |
| 1.3 | 清理无效冗余代码（建议） | 代码设计 |
| 2.2 | 禁止头文件循环依赖 | 头文件 |
| 2.3 | 避免包含用不到的头文件（建议） | 头文件 |
| 2.4 | 禁止 extern 声明引用外部接口 | 头文件 |
| 2.5 | 禁止在 extern "C" 中包含头文件 | 头文件 |
| 2.6 | 避免在头文件中使用 using 导入命名空间 | 头文件 |
| 3.1 | 避免滥用 typedef/#define 类型别名 | 数据类型 |
| 4.2 | 禁止使用魔鬼数字/字符串 | 常量 |
| 4.3 | 每个常量保证单一职责 | 常量 |
| 5.2 | 避免全局变量，谨慎使用单例 | 变量 |
| 5.3 | 禁止变量自增/自减表达式中再次引用 | 变量 |
| 6.1 | 表达式比较左变右不变 | 表达式 |
| 6.2 | 使用括号明确操作符优先级 | 表达式 |
| 8.1 | switch 语句要有 default 分支 | 控制语句 |
| 10.6 | 指针/引用形参不修改用 const | 指针数组 |
| 12.1 | 断言不能用于运行期错误处理 | 断言 |
| 14.4 | 使用强类型参数，避免 void* | 函数设计 |
| 15.1 | 函数传参顺序同一文件内保持一致 | 函数使用 |
| 15.2 | 入参用 const T&，出参用 T& 或 T* | 函数使用 |
| 15.5 | 单参数构造函数用 explicit | 函数使用 |

### 仅 Tiling 适用 `[适用: Tiling]`（19 条）

| 规范编号 | 规范名称 | 类别 |
|---------|---------|------|
| 3.2 | 使用 using 而非 typedef 定义别名 | 数据类型 |
| 7.1 | 使用 C++ 类型转换而非 C 风格 | 转换 |
| 9.1 | 禁止用 memcpy_s/memset_s 初始化非 POD | 声明初始化 |
| 10.2 | 优先使用 unique_ptr 而非 shared_ptr | 指针数组 |
| 10.3 | 使用 make_shared 而非 new 创建 shared_ptr | 指针数组 |
| 10.4 | 使用智能指针管理对象 | 指针数组 |
| 10.5 | 禁止使用 auto_ptr | 指针数组 |
| 11.1 | 字符串存储确保有 '\0' 结束符 | 字符串 |
| 13.1 | delete/delete[] 配对使用 | 类和对象 |
| 13.2 | 禁止 std::move 操作 const 对象 | 类和对象 |
| 13.3 | 严格使用 virtual/override/final | 类和对象 |
| 14.1 | 使用 RAII 追踪动态分配 | 函数设计 |
| 14.2 | 非局部 lambda 避免按引用捕获 | 函数设计 |
| 14.3 | 禁止虚函数使用缺省参数值 | 函数设计 |
| 15.3 | 不涉及所有权用 T* 或 const T& | 函数使用 |
| 15.4 | 传递所有权用 shared_ptr + move | 函数使用 |
| 15.6 | 拷贝构造和赋值操作符成对出现 | 函数使用 |
| 15.7 | 禁止保存、delete 指针参数 | 函数使用 |

---

### 1. 代码设计

##### 规则 1.1 对所有外部数据进行合法性检查，包括但不限于：函数入参、外部输入命名行、文件、环境变量、用户数据等 `[适用: All]`

##### 建议 1.3 清理无效、冗余或永不执行的代码 `[适用: All]`

> **说明**：业务代码中经常为特定场景预制冗余参数（如预留的函数入参、TilingData 字段等），这些参数当前可能未被引用但属于合理的工程预留，检视时不应标记为 FAIL。
>
> 以下情况可标记为提醒（SUSPICIOUS），供开发者参考：
> - 明显的死代码（如被条件编译永久排除的代码块）
> - 大段注释掉的代码（应使用 git 管理历史）
> - 明显无效且无预留意图的变量或表达式

---

### 2. 头文件和预处理

##### 规则 2.2 禁止头文件循环依赖 `[适用: All]`

头文件循环依赖，指a.h包含b.h，b.h包含c.h，c.h包含a.h之类导致任何一个头文件修改，都导致所有包含了a.h/b.h/c.h的代码全部重新编译一遍。
头文件循环依赖直接体现了架构设计上的不合理，可通过优化架构去避免。

##### 建议 2.3 避免包含用不到的头文件 `[适用: All]`

> **说明**：未使用的头文件会增加编译依赖和编译时间。但某些头文件可能为未来功能扩展预留，或用于提供类型前向声明，检视时仅作为提醒，不强制标记为 FAIL。

##### 规则 2.4 禁止通过 extern 声明的方式引用外部函数接口、变量 `[适用: All]`

> **Kernel 侧例外**：允许 `extern "C"` 声明 Kernel 入口函数。

##### 规则 2.5 禁止在extern "C"中包含头文件 `[适用: All]`

> **Kernel 侧说明**：Kernel 入口必须使用 `extern "C"`，但不应在其中包含头文件。

##### 建议 2.6 避免在头文件中使用 using 导入命名空间 `[适用: All]`

`using namespace` 在头文件中的传播范围取决于其作用域：
- **file-scope**（命名空间外部）：传播到所有包含该头文件的翻译单元
- **namespace-scoped**（`namespace X {}` 内部）：仅传播到重新打开同一命名空间的代码

> **Kernel 侧豁免**：以下 `using namespace` 是 Ascend C 框架惯例，不标记：
> - `using namespace AscendC;`
> - 导入 AscendC 子命名空间：`matmul`、`regbaseutil`、`MicroAPI`、`AscendC::Impl::Detail`、`AscendC::MicroAPI`、`optiling`
> - 导入算子内部实现命名空间（如 `AttentionCommon`、`fa_base_matmul`、`NormCommon` 等）
>
> **Host 侧豁免**：导入**项目内部命名空间**（如 `Ops::Transformer::OpTiling`、`Ops::NN::Optiling`、`Ops::Base`、`mc2_matmul_v3_advanced` 等）不标记，这些命名空间受项目控制，冲突风险低。

**FAIL 条件**（满足任一即 FAIL，但须先做危害分析再定级）：

1. **file-scope `using namespace` 出现在同文件后续 `#include` 之前**——后续 include 在该命名空间上下文中编译
2. **共享头文件路径**（`common/include/` 等公共目录）在 file-scope 导入大型命名空间（`std`、`ge`、`gert` 等）

> **定级前必须追溯传播链评估实际危害，结构违规 ≠ 实质危害：**
>
> 1. **确认 using 作用域**：在 `namespace X {}` 块内部则只对 X 可见，不泄漏到 includer 的 file-scope，**不是 file-scope 污染**
> 2. **追踪 include 链**：若被污染头文件的 include guard 在 using 之前已被更早的头文件激活，则**污染未实际发生**
> 3. **检查子头文件自主性**：若子头文件自带相同 using namespace，则外部污染是**冗余**，无新增风险
> 4. **定级**：有真实新增污染 → FAIL；结构违规但无实质危害 → SUSPICIOUS，注明原因

**SUSPICIOUS 条件**（标记提醒，不强制 FAIL）：

- 非共享头文件在 file-scope 导入 `std` / `ge` / `gert`
- API 头文件（op_api/）导入非项目命名空间（被外部调用者 include）
- 共享头文件在命名空间**内部**（非 file-scope）导入大型命名空间

---

### 3. 数据类型

##### 建议 3.1 避免滥用 typedef或者#define 对基本类型起别名 `[适用: All]`

##### 规则 3.2 使用using 而非typedef定义类型的别名，避免类型变化带来的散弹式修改 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 模板类无命名空间，using 定义别名不适用。

```cpp
// 正确示范
using FooBarPtr = std::shared_ptr<FooBar>;
// 错误示范
typedef std::shared_ptr<FooBar> FooBarPtr;
```

---

### 4. 常量

##### 规则 4.2 禁止使用魔鬼数字\字符串 `[适用: All]`

代码中禁止直接使用未经命名的字面量（魔鬼数字/魔鬼字符串），应使用 `constexpr` 或 `const` 命名常量替代，使含义自解释。例外：`0`、`-1`、`1` 等基础值，数组索引 `0`，`true`/`false`/`nullptr`。

**【真实检视案例】**（来自 ops-math 历史检视 PR，均被人工采纳）

案例1 — Kernel 侧，`32` 作为对齐粒度反复出现（ops-math PR#377，`op_kernel/split_d.h:213`）：

> 评论：「避免魔鬼数字 32，建议用 static constexpr 变量替换，用变量名体现 32 的含义」

```cpp
 208 | template <typename T>
 209 | __aicore__ inline void DataCopyPadInAdaptive(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,const uint32_t calCount) {
 210 |     const uint64_t totalByte = calCount * sizeof(T);
 211 |     uint64_t st =  reinterpret_cast<uint64_t>(src.GetPhyAddr());
 212 |     uint64_t en = st + totalByte;
 213 |     uint32_t pre = (32 - st % 32) % 32;   // ❌ 魔鬼数字 32 = 对齐字节数，应定义为 constexpr uint32_t ALIGN_BYTES = 32;
 214 |     uint32_t aft = en % 32;               // ❌ 同一魔鬼数字多处重复
 215 |     uint32_t prenum = pre / sizeof(T);
 216 |     uint32_t midnum = (totalByte - pre - aft) / sizeof(T);
```

案例2 — Tiling 侧，`32` 作为 blockSize（ops-math PR#1767，`op_host/div_mod_tiling.cpp:136`）：

> 评论：「建议通过接口获取 blockSize，消除魔鬼数字」

```cpp
 135 |     uint32_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
 136 |     uint32_t alignNum = 32 / typeSize;   // ❌ 魔鬼数字 32 = block 字节数，建议用 Ops::Base::GetUbBlockSize(context) 获取
 137 |     uint32_t totalLengthAligned = ((totalLength + alignNum - 1) / alignNum) * alignNum;
```

案例3 — Tiling 侧，`4`/`10` 作为 buffer 数量（ops-math PR#387，`op_host/round_tiling.cpp:113`）：

> 评论：「魔鬼数字建议使用具有含义的常量表示」

```cpp
 110 |     ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
 111 |     if (dataType == ge::DT_INT32) {
 112 |         // x, y → 2 个，因为要做 doublebuffer 优化，所以 x(x2), y(x2) → 共 4 个
 113 |         ubDataNumber = 4;                 // ❌ 魔鬼数字 4 = buffer 数量，应定义为命名常量
 114 |     } else if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
 115 |         if (decimals) {
 116 |             // x(x2), y(x2), round_temp(x2), x_as_float32(x2), x_scaled(x2) → 共 10 个
 117 |             ubDataNumber = 10;            // ❌ 魔鬼数字 10
```

##### 建议 4.3 建议每个常量保证单一职责 `[适用: All]`

---

### 5. 变量

##### 规则 5.2 尽量避免使用全局变量，谨慎使用单例模式，避免滥用 `[适用: All]`

> **Kernel 侧说明**：允许 `__aicore__` 模板类的成员变量（本质是全局存储）。

##### 规则 5.3 禁止在变量自增或自减运算的表达式中再次引用该变量 `[适用: All]`

---

### 6. 表达式

##### 建议 6.1 表达式的比较遵循左侧倾向于变化、右侧倾向于不变的原则 `[适用: All]`

```cpp
// 正确示范
if (ret != SUCCESS) {
  ...
}

// 错误示范
if (SUCCESS != ret) {
  ...
}
```

##### 规则 6.2 通过使用括号明确操作符的优先级，避免出现低级错误 `[适用: All]`

```cpp
// 正确示范
if (cond1 || (cond2 && cond3)) {
  ...
}

// 错误示范
if (cond1 || cond2 && cond3) {
  ...
}
```

---

### 7. 转换

##### 规则 7.1 使用有C++提供的类型转换，而不是C风格的类型转换，避免使用const_cast和reinterpret_cast `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 底层硬件操作必须用 `reinterpret_cast`（如 GM 地址转换）。

---

### 8. 控制语句

##### 规则 8.1 switch语句要有default分支 `[适用: All]`

---

### 9. 声明与初始化

##### 规则 9.1 禁止用 `memcpy_s`、`memset_s`初始化非POD对象 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 模板类都是 POD，且 `memset_s` 不可用。

---

### 10. 指针和数组

##### 规则 10.2 优先使用unique_ptr 而不是shared_ptr `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无智能指针。

##### 规则 10.3 使用std::make_shared 而不是new 创建shared_ptr `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无智能指针。

```cpp
// 正确示范
std::shared_ptr<FooBar> foo = std::make_shared<FooBar>();
// 错误示范
std::shared_ptr<FooBar> foo(new FooBar());
```

##### 规则 10.4 使用智能指针管理对象，避免使用new/delete `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 禁止动态内存分配，Buffer 由 `InitBuffer` 静态分配。

##### 规则 10.5 禁止使用auto_ptr `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无智能指针。

##### 规则 10.6 对于指针和引用类型的形参，如果是不需要修改的，要求使用const `[适用: All]`

---

### 11. 字符串

##### 规则 11.1 对字符串进行存储操作，确保字符串有'\0'结束符 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 C 风格字符串处理。

---

### 12. 断言

##### 规则 12.1 断言不能用于校验程序在运行期间可能导致的错误，可能发生的运行错误要用错误处理代码来处理 `[适用: All]`

---

### 13. 类和对象

##### 规则 13.1 单个对象释放使用delete，数组对象释放使用delete [] `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 禁止动态内存分配。

```cpp
const int kSize = 5;
int *number_array = new int[kSize];
int *number = new int();
...
delete[] number_array;
number_array = nullptr;
delete number;
number = nullptr;
```

##### 规则 13.2 禁止使用std::move操作const对象 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 `std::move`，模板类不支持移动语义。

##### 规则 13.3 严格使用virtual/override/final修饰虚函数 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无虚函数。

```cpp
class Base {
  public:
    virtual void Func();
};

class Derived : public Base {
  public:
    void Func() override;
};

class FinalDerived : public Derived {
  public:
    void Func() final;
};
```

---

### 14. 函数设计

##### 规则 14.1 使用 RAII 特性来帮助追踪动态分配 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态分配、无 `std::mutex`，无 RAII 场景。

```cpp
// 正确示范
{
  std::lock_guard<std::mutex> lock(mutex_);
  ...
}
```

##### 规则 14.2 非局部范围使用lambdas时，避免按引用捕获 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 不支持 lambda 表达式。

```cpp
{
  int local_var = 1;
  auto func = [&]() { ...; std::cout << local_var << std::endl; };
  thread_pool.commit(func);
}
```

##### 规则 14.3 禁止虚函数使用缺省参数值 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无虚函数。

##### 建议 14.4 使用强类型参数\成员变量，避免使用void* `[适用: All]`

---

### 15. 函数使用

##### 建议 15.1 函数传参顺序在同一文件（或同一模块）内保持一致 `[适用: All]`

> **说明**：不强制要求"入参在前、出参在后"。只要同一文件内的函数参数顺序风格统一即可（如统一采用入参在前，或统一采用出参在前）。检视时以文件内多数函数的风格为基准，仅标记明显不一致的情况。

```cpp
// ✅ 风格统一：全部采用入参在前
bool FuncA(const std::string &in, FooBar *out1, FooBar *out2);
bool FuncB(const int &val, Result *out);

// ✅ 风格统一：全部采用出参在前
bool FuncC(FooBar *out1, FooBar *out2, const std::string &in);
bool FuncD(Result *out, const int &val);

// ❌ 不一致：同文件内混用
bool FuncE(const std::string &in, FooBar *out);   // 入参在前
bool FuncF(Result *out, const int &val);           // 出参在前 → 风格不一致
```

##### 建议 15.2 函数传参传递，入参用 `const T &`，出参用 `T &` 或 `T *` `[适用: All]`

> **说明**：出参使用引用（`T &`）或指针（`T *`）均可。算子仓实践中，`T &` 是更常见的出参方式（尤其标量出参和 TilingData 结构体），`T *` 多见于框架接口（如 `gert::Shape*`）或需要表达可选（nullable）语义的场景。检视时不强制要求出参必须为指针，但同一文件内应保持一致。

```cpp
// ✅ 出参用引用（算子仓常见风格）
bool Func(const std::string &in, FooBar &out1, FooBar &out2);
void GetBasicShape(const gert::StorageShape *queryShape, uint32_t &b, uint32_t &s, uint32_t &h);

// ✅ 出参用指针（框架接口风格）
bool Func(const std::string &in, FooBar *out1, FooBar *out2);
static ge::graphStatus InferShape(const gert::Shape *inShape, gert::Shape *outShape);

// ❌ 同文件内混用（风格不一致）
void FuncA(const Input &in, Output &out);       // 出参用引用
void FuncB(const Input &in, Output *out);       // 出参用指针 → 同文件风格不一致
```

##### 规则 15.3 函数传参传递，不涉及所有权的场景，使用T * 或const T & 作为参数，而不是智能指针 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无智能指针。

```cpp
// 正确示范
  bool Func(const FooBar &in);
  // 错误示范
  bool Func(std::shared_ptr<FooBar> in);
```

##### 规则 15.4 函数传参传递，如需传递所有权，建议使用shared_ptr + move传参 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无智能指针。

```cpp
class Foo {
  public:
    explicit Foo(shared_ptr<T> x):x_(std::move(x)){}
  private:
    shared_ptr<T> x_;
};
```

##### 规则 15.5 单参数构造函数必须用explicit修饰，多参数构造函数禁止使用explicit修饰 `[适用: All]`

```cpp
explicit Foo(int x);          // good
explicit Foo(int x, int y=0); // good
Foo(int x, int y=0);          // bad
explicit Foo(int x, int y);   // bad
```

##### 规则 15.6 拷贝构造和拷贝赋值操作符应该是成对出现或者禁止 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 模板类通常只有默认实现，不显式定义拷贝/移动。

```cpp
class Foo {
  private:
    Foo(const Foo&) = default;
    Foo& operator=(const Foo&) = default;
    Foo(Foo&&) = delete;
    Foo& operator=(Foo&&) = delete;
};
```

##### 规则 15.7 禁止保存、delete指针参数 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 `delete`，指针参数来自 Buffer。

---

> **说明**：安全相关的编码规范（如内存安全、输入验证、安全函数使用等）请参见 `C++SecureCoding.md`。