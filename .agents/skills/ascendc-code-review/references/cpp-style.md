# CANN C++ 代码风格规范

<适用>
语言: C++
侧别: All
领域: false
默认启用: true
</适用>

<检视负载>
此条款文档由专项检视子 agent 独立托管，不参与条款派发流程
</检视负载>

> **适用场景**：Tiling 侧（Host 侧）和 Kernel 侧（Device 侧）都适用
>
> **说明**：代码风格规范适用于所有 C++ 代码，用于提升代码可读性和一致性。

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| 1.1 | C++文件使用小写+下划线命名 | 命名 | 中 |
| 1.2 | 函数命名使用大驼峰风格 | 命名 | 中 |
| 1.3 | 类型命名采用大驼峰风格 | 命名 | 中 |
| 1.4 | 变量命名采用小驼峰风格 | 命名 | 中 |
| 1.5 | 宏、枚举值采用全大写下划线连接 | 命名 | 中 |
| 2.1 | 行宽不超过 120 个字符 | 格式 | 低 |
| 2.2 | 使用空格缩进，每次 4 个空格 | 格式 | 中 |
| 2.3 | `&`、`*` 跟随变量名 | 格式 | 低 |
| 2.4 | if 语句必须使用大括号 | 格式 | 中 |
| 2.5 | for/while 循环必须使用大括号 | 格式 | 中 |
| 2.6 | 表达式换行运算符放行末 | 格式 | 低 |
| 2.7 | 使用 K&R 缩进风格 | 格式 | 中 |
| 2.8 | 多个变量定义不允许写在一行 | 格式 | 低 |
| 2.9 | 合理安排空行，保持代码紧凑 | 格式 | 低 |
| 3.1 | 文件头注释包含版权声明 | 注释 | 中 |
| 3.2 | 注释使用 `//`，与代码间有空格 | 注释 | 低 |
| 3.3 | 禁止使用 TODO/TBD/FIXME 及开发阶段注释 | 注释 | 中 |
| 3.4 | 不要写空有格式的函数头注释 | 注释 | 低 |
| 3.5 | 不用的代码直接删除，不要注释掉 | 注释 | 低 |

## 说明

本规范以[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)为基础，参考MindSpore社区、华为通用编码规范，并结合业界共识整理而成。参与CANN开源社区项目的开发者首先需要遵循本规范内容，其余遵循Google C++ Style Guide规范。

如果对规则有异议，建议提交issue并说明理由，经CANN运营团队评审后可接纳并修改生效。

## 适用范围

CANN 相关开源仓的代码风格检视。

### 专属检视方法

风格违反为确定性判定，**不走 core/methodology.md 的假设检验流程**（不建立 H0/H1、不收集证据分值、不计算自信值）。逐条例对照代码，违反即标记 FAIL，无违反即 PASS。

- 置信度按条例严重级别：中 → MED，低 → LOW（不产生 HIGH）
- 条例标题为描述性文字（如「驼峰风格(CamelCase)」），无 ID 前缀，**禁止 Grep `^{条例ID}` 逐条定位**——直接 Read 本文件全文（19 条，411 行，全读可控）
- 每条结果以 `[STYLE]` 前缀标记，区别于 `[条例ID]` 的安全/性能检视结果，供报告单独归入「代码风格」章节

---

### 1. 命名

#### 驼峰风格(CamelCase)

大小写字母混用，单词连在一起，不同单词间通过单词首字母大写来分开。
按连接后的首字母是否大写，又分: 大驼峰(UpperCamelCase)和小驼峰(lowerCamelCase)

| 类型                                       | 命名风格      |
| ---------------------------------------- | --------- |
| 类类型，结构体类型，枚举类型，联合体类型等类型定义， 作用域名称         | 大驼峰       |
| 函数(包括全局函数，作用域函数，成员函数)                    | 大驼峰       |
| 全局变量(包括全局和命名空间域下的变量，类静态变量)，局部变量，函数参数，类、结构体和联合体中的成员变量 | 小驼峰       |
| 宏，常量(const)，枚举值，goto 标签                  | 全大写，下划线分割 |

注意：
上表中**常量**是指全局作用域、namespace域、类的静态成员域下，以 const或constexpr 修饰的基本数据类型、枚举、字符串类型的变量，不包括数组和其他类型变量。
上表中**变量**是指除常量定义以外的其他变量，均使用小驼峰风格。


##### 规则 1.1 C++文件使用小写+下划线的方式命名，以.cpp结尾，头文件以.h结尾

目前业界还有一些其他的后缀的表示方法：
- 头文件：  .hh, .hpp, .hxx
- cpp文件：.cc, .cxx, .c

如果当前项目组使用了某种特定的后缀，那么可以继续使用，但是请保持风格统一。
但是对于本文档，我们默认使用.h和.cpp作为后缀。

##### 规则 1.2 函数命名统一使用大驼峰风格，一般采用动词或者动宾结构

```cpp
class List {
public:
	void AddElement(const Element& element);
	Element GetElement(const unsigned int index) const;
	bool IsEmpty() const;
};

namespace Utils {
    void DeleteUser();
}
```

##### 规则 1.3 类型命名采用大驼峰命名风格

所有类型命名——类、结构体、联合体、类型定义（typedef）、枚举——使用相同约定，例如：
```cpp
// classes, structs and unions
class UrlTable { ...
struct UrlTableProperties { ...
union Packet { ...
// typedefs
typedef std::map<std::string, UrlTableProperties*> PropertiesMap;
// enums
enum UrlTableErrors { ...
```

对于命名空间的命名，建议使用大驼峰：
```cpp
// namespace
namespace FileUtils {   
}
```

##### 规则 1.4 通用变量命名采用小驼峰，包括全局变量，函数形参，局部变量，成员变量

```cpp
std::string tableName;  // Good: 推荐此风格
std::string tablename;  // Bad: 禁止此风格
std::string path;       // Good: 只有一个单词时，小驼峰为全小写
```

全局变量应增加 'g_' 前缀，静态变量命名不需要加特殊前缀
全局变量是应当尽量少使用的，使用时应特别注意，所以加上前缀用于视觉上的突出，促使开发人员对这些变量的使用更加小心。
- 全局静态变量命名与全局变量相同。
- 函数内的静态变量命名与普通局部变量相同。
- 类的静态成员变量和普通成员变量相同。

```cpp
int g_activeConnectCount;

void Func()
{
    static int packetCount = 0; 
    ...
}
```

类的成员变量命名以小驼峰加后下划线组成

```cpp
class Foo {
private:
    std::string fileName_;   // 添加_后缀，类似于K&R命名风格
};
```

##### 规则 1.5 宏、枚举值采用全大写，下划线连接的格式

全局作用域内，有名和匿名namespace内的 const 常量，类的静态成员常量，全大写，下划线连接；函数局部 const 常量和类的普通const成员变量，使用小驼峰命名风格。

```cpp
#define MAX(a, b)   (((a) < (b)) ? (b) : (a)) // 仅对宏命名举例，并不推荐用宏实现此类功能

enum TintColor {    // 注意，枚举类型名用大驼峰，其下面的取值是全大写，下划线相连
    RED,
    DARK_RED,
    GREEN,
    LIGHT_GREEN
};

int Func(...)
{
    const unsigned int bufferSize = 100;    // 函数局部常量
    char *p = new char[bufferSize];
    ...
}

namespace Utils {
	const unsigned int DEFAULT_FILE_SIZE_KB = 200;        // 全局常量
}
```

---

### 2. 格式

##### 建议 2.1 行宽不超过 120 个字符

建议每行字符数不要超过 120 个。如果超过120个字符，请选择合理的方式进行换行。

例外:
- 如果一行注释包含了超过120 个字符的命令或URL，则可以保持一行，以方便复制、粘贴和通过grep查找；
- 包含长路径的 #include 语句可以超出120 个字符，但是也需要尽量避免；
- 编译预处理中的error信息可以超出一行。
预处理的 error 信息在一行便于阅读和理解，即使超过 120 个字符。
```cpp
#ifndef XXX_YYY_ZZZ
#error Header aaaa/bbbb/cccc/abc.h must only be included after xxxx/yyyy/zzzz/xyz.h, because xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#endif
```

##### 规则 2.2 使用空格进行缩进，每次缩进4个空格

只允许使用空格(space)进行缩进，每次缩进为 4 个空格。不允许使用Tab符进行缩进。
当前几乎所有的集成开发环境（IDE）都支持配置将Tab符自动扩展为4空格输入；请配置你的IDE支持使用空格进行缩进。

##### 规则 2.3 在声明指针、引用变量或参数时, `&`、`*`跟随变量名，另外一边留空格

```cpp
char *c;
const std::string &str;
```

##### 规则 2.4 if语句必须要使用大括号

我们要求if语句都需要使用大括号，即便只有一条语句。
理由：
- 代码逻辑直观，易读；
- 在已有条件语句代码上增加新代码时不容易出错；
- 对于在if语句中使用函数式宏时，有大括号保护不易出错（如果宏定义时遗漏了大括号）。

```cpp
// 即使if分支代码只有一行，也必须使用大括号
if (cond) {
  single line code;
}
```

##### 规则 2.5 for/while等循环语句必须使用大括号

和条件表达式类似，我们要求for/while循环语句必须加上大括号，即便循环体是空的，或循环语句只有一条。
```cpp
for (int i = 0; i < someRange; i++) {   // Good: 使用了大括号
    DoSomething();
}
```
```cpp
while (condition) { }   // Good：循环体是空，使用大括号
```

##### 规则 2.6 表达式换行要保持换行的一致性，运算符放行末

较长的表达式，不满足行宽要求的时候，需要在适当的地方换行。一般在较低优先级运算符或连接符后面截断，运算符或连接符放在行末。
运算符、连接符放在行末，表示"未结束，后续还有"。
例：
// 假设下面第一行已经不满足行宽要求
```cpp
if ((currentValue > threshold) &&  // Good：换行后，逻辑操作符放在行尾
    someCondition) {
    DoSomething();
    ...
}

int result = reallyReallyLongVariableName1 +    // Good
             reallyReallyLongVariableName2;
```
表达式换行后，注意保持合理对齐，或者4空格缩进。参考下面例子

```cpp
int sum = longVariableName1 + longVariableName2 + longVariableName3 +
    longVariableName4 + longVariableName5 + longVariableName6;         // Good: 4空格缩进

int sum = longVariableName1 + longVariableName2 + longVariableName3 +
          longVariableName4 + longVariableName5 + longVariableName6;   // Good: 保持对齐
```

##### 规则 2.7 使用 K&R 缩进风格

**K&R风格**
换行时，函数（不包括lambda表达式）左大括号另起一行放行首，并独占一行；其他左大括号跟随语句放行末。
右大括号独占一行，除非后面跟着同一语句的剩余部分，如 do 语句中的 while，或者 if 语句的 else/else if，或者逗号、分号。

如：
```cpp
struct MyType {     // 跟随语句放行末，前置1空格
    ...
};

int Foo(int a)
{                   // 函数左大括号独占一行，放行首
    if (...) {
        ...
    } else {
        ...
    }
}
```
推荐这种风格的理由：

- 代码更紧凑；
- 相比另起一行，放行末使代码阅读节奏感上更连续；
- 符合后来语言的习惯，符合业界主流习惯；
- 现代集成开发环境（IDE）都具有代码缩进对齐显示的辅助功能，大括号放在行尾并不会对缩进和范围产生理解上的影响。


对于空函数体，可以将大括号放在同一行：
```cpp
class MyClass {
public:
    MyClass() : value_(0) {}
   
private:
    int value_;
};
```

##### 规则 2.8 多个变量定义和赋值语句不允许写在一行

每行只有一个变量初始化的语句，更容易阅读和理解。

##### 规则 2.9 合理安排空行，保持代码紧凑

减少不必要的空行，可以显示更多的代码，方便代码阅读。下面有一些建议遵守的规则：
- 根据上下内容的相关程度，合理安排空行；
- 函数内部、类型定义内部、宏内部、初始化表达式内部，不使用连续空行
- 不使用连续 **3** 个空行，或更多
- 大括号内的代码块行首之前和行尾之后不要加空行，但namespace的大括号内不作要求。

```cpp
int Foo()
{
    ...
}



int Bar()  // Bad：最多使用连续2个空行。
{
    ...
}


if (...) {
        // Bad：大括号内的代码块行首不要加入空行
    ...
        // Bad：大括号内的代码块行尾不要加入空行
}

int Foo(...)
{
        // Bad：函数体内行首不要加空行
    ...
}
```

---

### 3. 注释

一般的，尽量通过清晰的架构逻辑，好的符号命名来提高代码可读性；需要的时候，才辅以注释说明。 
注释是为了帮助阅读者快速读懂代码，所以要从读者的角度出发，**按需注释**。

注释内容要简洁、明了、无二义性，信息全面且不冗余。

在 C++ 代码中，使用 `/*` `*/`和 `//` 都是可以的。 
按注释的目的和位置，注释可分为不同的类型，如文件头注释、函数头注释、代码注释等等； 
同一类型的注释应该保持统一的风格。

##### 规则 3.1 文件头注释包含版权声明

如下例子：

```cpp
/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

```

> 关于版权说明，应注意：
> 2025年新建的文件，应该是 `Copyright (c) 2025 Huawei Technologies Co., Ltd.`

##### 规则 3.2 代码注释置于对应代码的上方或右边，注释符与注释内容之间要有1个空格，右置注释与前面代码至少1空格，使用 `//`，而不是 `/**/`

```cpp
// this is multi-
// line comment
int foo; // this single-line comment
```

##### 规则 3.3 代码中禁止使用 TODO/TBD/FIXME 及开发阶段注释，建议提issue跟踪

交付代码中禁止残留开发过程的临时标记，包括：
- **TODO/TBD/FIXME** 等未完成事项标记（应提 issue 跟踪，不在代码中留标记）
- **AI 开发阶段注释**：开发过程中用于划分步骤、阶段的注释，如 `// Step 1`、`// 阶段一`、`// Phase 2`、`// 第一步：初始化`、`// === 计算核心 ===` 等过程性标记。这些注释在交付代码中应移除，或转为正式的函数/变量命名自解释

**【错误代码示例】**

```cpp
// ❌ 残留 AI 开发阶段注释
__aicore__ void Compute() {
    // Step 1: 数据搬运
    DataCopy(xLocal, xGm);
    // Phase 2: 计算
    Add(zLocal, xLocal, yLocal);
    // TODO: 后续优化流水线
    DataCopy(zGm, zLocal);
}
```

**【正确代码示例】**

```cpp
// ✅ 移除阶段注释，通过函数命名自解释
__aicore__ void Compute() {
    CopyIn();
    ComputeAdd();
    CopyOut();
}
```

##### 建议 3.4 不要写空有格式的函数头注释

并不是所有的函数都需要函数头注释，函数尽量通过函数名自注释，按需写函数头注释；函数原型无法表达的，却又希望读者知道的信息，才需要加函数头注释辅助说明。
不要写无用、信息冗余的函数头，函数头注释内容可选，但不限于：功能说明、返回值，性能约束、用法、内存约定、算法实现、可重入的要求等。
例：

```cpp
/*
 * 返回实际写入的字节数，-1表示写入失败
 * 注意，内存 buf 由调用者负责释放
 */
int WriteString(const char *buf, int len);
```

坏的例子：
```cpp
/*
 * 函数名：WriteString
 * 功能：写入字符串
 * 参数：
 * 返回值：
 */
int WriteString(const char *buf, int len);
```
上面例子中的问题：

- 参数、返回值，空有格式没内容
- 函数名信息冗余
- 关键的 buf 由谁释放没有说清楚

##### 建议 3.5 不用的代码段直接删除，不要注释掉

被注释掉的代码，无法被正常维护；当企图恢复使用这段代码时，极有可能引入易被忽略的缺陷。
正确的做法是，不需要的代码直接删除掉。若再需要时，考虑移植或重写这段代码。