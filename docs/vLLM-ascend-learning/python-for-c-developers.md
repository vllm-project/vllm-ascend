# 从 C 到 Python：结合 vLLM Ascend 的学习指南

这份文档面向已经会写 C、准备阅读和修改 vLLM Ascend 的开发者。目标是掌握本项目真正用到的 Python，而不是先学完整门语言。建议按章节阅读，遇到源码链接就打开对照。

本文依据当前工作区源码整理。标为“源码”的代码来自项目；标为“教学示例”的代码用于解释概念，不表示项目已经这样实现。除特别说明外，教学示例只需要 Python 标准库；Tensor 示例需要 PyTorch，项目测试还需要对应的 vLLM、Ascend 依赖和测试环境。

## 1. 先建立一张 C 与 Python 的对照表

| C 中熟悉的东西 | Python 中需要学习的东西 | 类比的边界 |
| --- | --- | --- |
| 变量、指针、结构体 | 名称、对象、对象引用 | Python 不提供普通对象的指针算术 |
| 数组、链表、哈希表 | `list`、`tuple`、`dict`、`set` | `list` 不是连续存放裸整数的 C 数组 |
| 函数声明和参数类型 | `def` 和类型注解 | 注解通常不执行运行时类型检查 |
| `struct` 加操作函数 | 类、实例、`self` | 类还支持继承和动态方法查找 |
| 函数指针和回调表 | 函数对象、可调用对象、注册表 | 类本身也可以作为值存入表中 |
| 包装函数、注册宏 | 装饰器 `@...` | 装饰器是运行时执行的函数调用 |
| `.h`、`.c`、链接 | 模块、包、`import` | 导入模块可能执行代码并修改状态 |
| 错误码和 `goto cleanup` | 异常、`try/finally`、`with` | 异常会沿调用栈传播 |
| 配置结构体 | dataclass、配置验证器 | 字段注解本身不等于验证 |
| 设备地址和张量描述符 | `torch.Tensor` | 还必须考虑设备、形状、步长、存储别名 |

优先级：先掌握第 2～7 章，再读第 8～12 章，最后按需要学习并发和工具链。下面已把学习过程中问到的问题补入对应章节，后面的新概念也按“含义 → 小例子 → 执行过程 → C 类比 → 项目用途”展开。

阅读代码块时，请保留缩进。函数参数、列表等括号内部可以换行，例如 `b.append(3)` 可以写成三行；但不能把 `return size, dtype, device` 随意断成 `return` 一行、返回内容另一行。单独的 `return` 会立即返回 `None`。文中的独立代码块不一定共享变量，运行某个例子时先执行其中的定义。

按当前学习进度，可以这样分段阅读，不用一次读完：

| 你想解决的问题 | 阅读位置 |
| --- | --- |
| `append`、`assert`、`b = a`、浅拷贝 | 第 2 章 |
| `%`、`in`、f-string、循环 | 第 3.1 节 |
| 各种类型注解、默认参数、星号参数 | 第 3.2～3.3 节 |
| 字典键值、`.get`、元组解包、切片 | 第 4 章 |
| `self`、实例属性、继承、`super()` | 第 5 章 |
| 函数指针类比、闭包、装饰器、`lambda` | 第 6 章 |
| 模块、导入、动态属性和 patch | 第 7 章 |
| 异常、`finally`、`with`、`yield` | 第 8 章 |
| 配置对象、测试、mock | 第 9～10 章 |
| 算子、Tensor、设备同步、并发 | 第 11～13 章 |
| 源码路线、练习、错误排查 | 第 14～16 章 |

## 2. 最大的思维变化：变量是对象的名字

### 2.1 赋值不是结构体复制

先认识三个基本动作：

```python
a = [1, 2]     # 创建一个列表，让名字 a 指向它
b = a          # 让名字 b 也指向同一个列表
b.append(3)    # 向这个列表末尾追加一个元素 3
```

`append` 是列表自带的方法，类似你在 C 中实现的 `vector_append(&b, 3)`。Python 自动处理列表的扩容。点号 `.` 表示访问对象的属性或方法，括号 `()` 表示调用，括号里的 `3` 是参数。

```python
b.append(
    3
)
```

这与 `b.append(3)` 一样。换行不会让它变成三个操作。`append` 修改原列表，返回值是 `None`，所以不要用 `b = b.append(3)` 来追加，否则 `b` 会被重新绑定为 `None`。

C 中的结构体赋值通常复制结构体字段；Python 的普通赋值把名字绑定到对象。

教学示例：

```python
a = [1, 2]
b = a
b.append(3)
assert a == [1, 2, 3]
assert b is a
```

可以暂时类比为两个 C 指针指向同一块数据，但不要延伸到地址计算、手动 `free`。对象存活由运行时管理；CPython 主要使用引用计数并辅以循环垃圾回收，但不能依赖对象一定在某一行被销毁来管理外部资源。

重新绑定名字与修改对象是两件事：

```python
a = [1, 2]
b = a
b = [9]
assert a == [1, 2]
assert b == [9]
```

### 2.2 函数参数也遵守这个规则

上一节的两次赋值，可以画成：

```text
执行 b = a 后：
a ──┐
    ├──→ [1, 2]
b ──┘

执行 b = [9] 后：
a ─────→ [1, 2]
b ─────→ [9]
```

第二句不会通过 `b` 修改原列表。用 C 指针类比，`b = a` 是复制指针，`b = 新地址` 是改变指针指向。Python 对象并不是可以手动释放和偏移的 C 指针，类比只用于理解共享关系。

Python 不是“所有参数都深拷贝”，也不要简单理解为 C 的引用参数。

```python
def update(values, count):
    values.append(7)  # 修改调用方也能访问的对象
    count = count + 1  # 只重新绑定本函数的局部名字
    return count


items = []
n = 0
result = update(items, n)
assert items == [7]
assert n == 0
assert result == 1
```

可以理解为“形参拿到实参对象的引用；重新绑定形参不会替调用方重新绑定变量”。整数、字符串、元组不可变；列表、字典通常可以原地修改。元组不可变仅指元素引用不能替换，元组里面引用的列表仍然可以变化。

### 2.3 浅拷贝与嵌套别名

先看普通列表的复制：

```python
a = [1, 2]
b = a.copy()
b.append(3)

print(a)  # [1, 2]
print(b)  # [1, 2, 3]
```

`a.copy()` 创建新的外层列表，类似 C 中新申请一个数组，再复制每一个数组项。因此 `a` 和 `b` 的外层已经分开。但如果每一项本身还是列表，复制每项得到的是对子列表的引用：

```python
a = [[1], [2]]
b = a.copy()
assert b is not a
assert b[0] is a[0]
b[0].append(9)
assert a[0] == [1, 9]
```

逐行理解这里的两个断言：

- `b is not a`：`b` 与 `a` 是两个不同的外层列表。
- `b[0] is a[0]`：它们的第 0 项指向同一个内层列表 `[1]`。

```text
a → 外层列表 A → 第0项 ──→ 内层列表 [1]
                          ↑
b → 外层列表 B → 第0项 ───┘
```

这类似 C 中复制一个指针数组：

```c
int first[] = {1};
int second[] = {2};
int *a[2] = {first, second};
int *b[2];
memcpy(b, a, sizeof(a)); /* 需要 #include <string.h> */
```

两个数组是分开的，但 `b[0]`、`a[0]` 保存同一个指针。修改 `b[0][0]` 就会影响 `a[0][0]`。反过来，把 `b[0]` 换成另一个指针，不会改变 `a[0]`。

Python 同样如此：

```python
a = [[1], [2]]
b = a.copy()
b[0] = [8]   # 替换 b 外层的第 0 项，没有修改旧的内层列表

print(a)  # [[1], [2]]
print(b)  # [[8], [2]]
```

`copy()` 只复制外层容器。需要递归复制一般对象时可研究 `copy.deepcopy`；Tensor 是否复制数据则要看 `clone()` 等操作，不能把容器的复制规则直接套过去。

一个典型陷阱是 `rows = [[]] * 3`：三个元素引用同一个列表。独立列表应写成 `[[] for _ in range(3)]`。

### 2.4 `==`、`is`、`None` 和真假值

| 写法 | 含义 | 常见使用位置 |
| --- | --- | --- |
| `a == b` | 值是否相等，可由类型定制 | 数值、字符串、容器比较 |
| `a is b` | 是否同一个对象 | 判断别名，尤其 `x is None` |
| `None` | 表示缺失的单例对象 | 类似可选指针的空状态，不是整数零 |
| `if items:` | 检查真假值 | 空列表、空字典、空字符串为假 |
| `x is not None` | 检查是否提供了值 | 需要区分 `0`、空容器和缺失值 |

`is` 不用于一般整数或字符串的值比较。`if tensor:` 有单独的限制，见第 12 章。

普通对象的真假值可以通过下面的例子理解：

```python
print(bool([]))      # False，空列表
print(bool([0]))     # True，非空列表，即使唯一的元素是 0
print(bool(""))      # False，空字符串
print(bool("0"))     # True，非空字符串，不是整数 0
print(bool(None))    # False
```

`if values:` 检查列表是否非空；`if values is not None:` 只检查它是不是缺失值，空列表也能通过后一个判断。两者回答的问题不同。

### 2.5 `assert` 到底做了什么

`assert 条件` 的意思是：“我预期这个条件成立，请检查。”条件为真就继续执行，没有输出；为假就抛出 `AssertionError`。未处理时，程序停止并显示出错位置。

```python
a = [1, 2, 3]
assert a == [1, 2, 3]  # 成立，继续
assert len(a) == 2     # 不成立，抛出 AssertionError
```

类似 C 中 `#include <assert.h>` 后的 `assert(count == 3)`。Python 的列表可以用 `==` 比较内容，C 数组通常不能直接这样比较。

可以带上失败提示：

```python
assert len(a) == 3, "列表长度应该是 3"
```

比较内容和比较对象身份是两个条件：

```python
a = [1, 2]
b = a
c = [1, 2]

assert b == a       # 成立：内容相等
assert b is a       # 成立：同一个列表
assert c == a       # 成立：内容相等
assert c is not a   # 成立：是两个列表
```

`assert b is a` 类似 C 比较两个指针是否指向同一个对象。不要把它理解为检查列表内容。另请注意 `=` 是赋值，`==` 才是比较。Python 优化模式可能移除断言，所以需要始终执行的参数校验应使用 `if` 和 `raise`，见第 8 章。

## 3. 先读懂函数、分支和常用语法

### 3.1 缩进、循环与整数运算

```python
def count_positive(values):
    count = 0
    for value in values:
        if value > 0:
            count += 1
    return count
```

冒号后通过缩进定义代码块，通常使用 4 个空格。`for value in values` 类似 C 遍历数组，但迭代对象不必是数组。需要下标时用 `for index, value in enumerate(values)`；并行遍历用 `zip(a, b)`，默认在较短输入结束时停止。

| Python 写法 | 对 C 开发者的解释 |
| --- | --- |
| `range(n)` | 遍历 `0` 到 `n - 1`，不是预先生成一个列表 |
| `a / b` | 普通整数相除也产生浮点结果 |
| `a // b` | 向负无穷取整；`-3 // 2 == -2`，不同于 C 整数除法的向零截断 |
| `a % b` | 与 Python 的整除规则配套；注意负数时与 C 的差异 |
| `x if condition else y` | 类似 C 的 `condition ? x : y` |
| `and`、`or`、`not` | 逻辑操作，前两者短路，并返回操作数而不一定是 `bool` |
| `in` | 成员检查；对字典检查的是键 |
| `f"size={size}"` | 格式化字符串，类似带表达式的 `printf` |

Python 普通整数不会像固定宽度 C 整数一样按位宽自动溢出；Tensor 和 NumPy 数值仍受其 dtype 位宽约束。

#### 3.1.1 `%` 是不是取余

是。例如 `7 % 3 == 1`，因为 `7 = 2 × 3 + 1`。可以用 `count % 2 == 0` 判断偶数。

但负数要留意：

| 表达式 | Python | C99 及以后的 C |
| --- | --- | --- |
| `7 % 3` | `1` | `1` |
| `-7 % 3` | `2` | `-1` |
| `7 % -3` | `-2` | `1` |

对整数，Python 使用 `a == (a // b) * b + a % b`。`-7 // 3 == -3`，所以余数必须是 `2`，才能满足 `-7 = -3 × 3 + 2`。非零余数与除数同号；除数不能为零。

#### 3.1.2 `in` 是在检查什么

`in` 是成员检查，返回 `True` 或 `False`。对列表，就是问“里面有没有这个元素”：

```python
numbers = [10, 20, 30]
print(20 in numbers)  # True
print(99 in numbers)  # False
```

可以类比 C 遍历数组寻找相等元素。对字典，`in` 检查的是键，具体例子见第 4.1 节。

#### 3.1.3 `f"size={size}"` 每个符号是什么

```python
size = 128
text = f"size={size}"
print(text)  # size=128
```

前缀 `f` 表示要把花括号里的表达式求值，再填到字符串中。`{size}` 被替换成变量的文字表示。C 中常用 `printf("size=%d", size)` 输出类似内容；但 f-string 本身只构造字符串，`print` 才输出。

```python
print("size={size}")       # size={size}，没有 f 就只是普通文字
print(f"double={size*2}")  # double=256，可以放表达式
```

需要控制小数位时，可以写 `f"{value:.2f}"`，例如 `value = 1.234` 得到 `"1.23"`。冒号后是格式要求，不是字典的键值分隔符。

#### 3.1.4 把几种循环展开来看

```python
names = ["attention", "moe"]
for index, name in enumerate(names):
    print(index, name)
# 0 attention
# 1 moe
```

`enumerate` 每次提供“下标和元素”两个值，左侧分别接收。类似 C 中 `for (int i = 0; i < n; ++i)` 配合 `names[i]`。

```python
for name, size in zip(["a", "b"], [10, 20]):
    print(name, size)
# a 10
# b 20
```

`zip` 把两个输入中的对应项放在一起。`range(3)` 则依次提供 `0`、`1`、`2`，不包括 `3`。这几个工具都是可迭代对象，不需要先转换成列表才可循环。

#### 3.1.5 `and`、`or` 为什么不一定返回布尔值

C 的 `&&`、`||` 产生 `0` 或 `1`。Python 的 `and`、`or` 按真假值短路，但返回参与判断的操作数：

```python
print("" or "默认名字")      # 默认名字
print("张三" or "默认名字")  # 张三
print(0 and 99)              # 0
print(10 and 99)             # 99
```

`a or b`：若 `a` 为真，直接返回 `a`，否则计算并返回 `b`。`a and b`：若 `a` 为假，直接返回 `a`，否则计算并返回 `b`。`not a` 才总是得到 `True` 或 `False`。

这解释了项目中的 `config or {}`：在 `config` 为假时使用空字典。但如果 `0`、空字符串也有业务含义，就不能把这种写法当成仅检查 `None`。

### 3.2 类型注解不是 C 的类型声明

项目源码：[量化注册表](../../vllm_ascend/quantization/methods/registry.py)。

```python
def get_scheme_class(quant_type: str, layer_type: str) -> type[Any] | None:
    return _SCHEME_REGISTRY.get((quant_type, layer_type))
```

逐项翻译：

- `def`：定义函数。
- `quant_type: str`：期望参数是字符串，供阅读、IDE 和静态检查使用。
- `->`：后面是返回值类型注解。
- `type[Any]`：返回的是类对象，不是该类的实例。
- `| None`：也可能找不到而返回 `None`。
- `.get(key)`：字典中没有这个键时返回 `None`，不会像 `dict[key]` 一样抛出 `KeyError`。

普通函数不会仅凭这些注解拒绝错误参数。需要检查时应显式验证，或通过配置框架执行验证。

常见注解速查：

| 注解 | 含义 |
| --- | --- |
| `list[int]` | 整数列表 |
| `dict[str, int]` | 字符串键、整数值的字典 |
| `tuple[int, int]` | 两个整数构成的元组 |
| `tuple[int, ...]` | 任意长度的整数元组 |
| `torch.Tensor \| None` | Tensor 或缺失值 |
| `Callable[[], Any]` | 不接收参数的可调用对象，返回类型不限 |
| `Any` | 放宽静态类型约束，不表示代码必然安全 |

### 3.3 类型注解实例、默认参数与参数转发

#### 3.3.1 先把类型注解逐种写成实例

上一节的表格用于查找，下面逐项说明它们实际长什么样。注解通常只是“我预期是什么类型”，真正的数据写在等号右边。

```python
numbers: list[int] = [10, 20, 30]
```

`numbers` 是变量名，冒号后是类型说明，`list` 表示列表，`[int]` 表示元素应为整数。类似 C 的 `int numbers[] = {10, 20, 30}`，但 Python 列表可以增长，也不会仅因注解而禁止插入其他类型。

```python
ages: dict[str, int] = {"张三": 18, "李四": 20}
```

`dict` 是字典，方括号里的第一个类型 `str` 约束键，第二个 `int` 约束值。这里用人的名字查找年龄。它不是两个元素的字典，而是规定每一项的键值类型。

```python
position: tuple[int, int] = (10, 20)
```

`tuple[int, int]` 表示恰好两个元素，第一个应为整数，第二个也应为整数。`position[0]` 是 `10`，`position[1]` 是 `20`。可以类比一个包含两个整数的只读结构体，但元组通过下标访问，而不是通过字段名。

```python
empty: tuple[int, ...] = ()
single: tuple[int, ...] = (10,)
many: tuple[int, ...] = (10, 20, 30)
```

`tuple[int, ...]` 中的省略号表示该类型可以重复任意次，包含零次。不是说元组创建后可以追加元素。单元素元组必须保留逗号：`(10,)` 是元组，`(10)` 是整数。

```python
position = (10, 20)
# position[0] = 99  # 如果执行，会抛出 TypeError
```

元组不可替换元素，但变量可以重新指向另一个元组：`position = (99, 20)` 是允许的。这仍然是“修改对象”与“重新绑定名字”的区别。

#### 3.3.2 `torch.Tensor | None` 是两种可能

```python
def process(residual: torch.Tensor | None = None):
    if residual is None:
        print("没有提供 residual")
    else:
        print("提供了一个 Tensor")
```

此例需要先 `import torch`。`torch.Tensor` 是张量类型，可先理解为带形状、元素类型和设备信息的多维数组。注解中的 `|` 表示“或者”：参数可以是 Tensor，也可以是 `None`。

`process()` 使用默认的 `None`，输出第一句；`process(torch.tensor([1, 2]))` 输出第二句。类比 C 的 `struct Tensor *residual`，调用者可以传 `NULL` 表示缺失，但 Python `None` 本身不是空地址数值。

`None` 不等于 `0`，也不等于内容全是零的 Tensor。因此判断有没有参数，应写 `residual is None`，不要写 `if residual`。

#### 3.3.3 `type[Any]` 返回的是类，不是实例

```python
class Dog:
    def bark(self):
        print("汪")


cls = Dog       # 保存类本身，还没有创建一只狗
dog = cls()     # 相当于 Dog()，创建实例
dog.bark()      # 汪
```

类可以类比 C 的 `struct Dog` 定义，实例可以类比 `struct Dog dog`。区别是 Python 的类本身也能存到变量、字典中，作为参数传递，或作为函数返回值。

因此 `get_scheme_class(...) -> type[Any] | None` 的意思是“返回某个类；找不到就返回 None”，不是“直接返回已创建好的量化对象”。`Any` 表示这里没有进一步限制是哪种类，不表示创建了某个叫 Any 的对象。

教学调用方式：

```python
scheme_cls = get_scheme_class("W8A8_DYNAMIC", "linear")
if scheme_cls is not None:
    scheme = scheme_cls()  # 实际构造参数以对应类的定义为准
```

第一步查类，第二步才调用类来构造实例。

#### 3.3.4 `Callable[[], Any]` 类似无参数函数指针

```python
from collections.abc import Callable
from typing import Any


def get_number():
    return 42


callback: Callable[[], Any] = get_number
result = callback()
print(result)  # 42
```

`Callable` 表示可调用；第一个位置的 `[]` 是参数类型列表，空列表表示不传参数；第二个位置的 `Any` 表示没有限定返回类型。

类比 C：

```c
int get_number(void) { return 42; }
int (*callback)(void) = get_number;
int result = callback();
```

这个 C 类型更接近 `Callable[[], int]`，因为它限定返回整数。再举两个注解例子：

| 注解 | 期望的调用方式 |
| --- | --- |
| `Callable[[], int]` | `func()`，返回整数 |
| `Callable[[int], str]` | `func(10)`，接收一个整数，返回字符串 |
| `Callable[[int, int], int]` | `func(10, 20)`，接收两个整数，返回整数 |

可调用对象还包括类和定义了调用行为的实例，初学时先按函数理解即可。`Any` 也不是 C 的 `void`：`void` 表示无返回值，`Any` 表示返回类型未受限制；Python 表示正常不返回有用结果时通常用 `-> None`。

#### 3.3.5 哪些参数可以不传

教学示例：

```python
def allocate(size, dtype="float16", *, device="npu"):
    return size, dtype, device


result = allocate(128, device="cpu")
assert result == (128, "float16", "cpu")
```

`*` 后面的 `device` 必须通过名字传递。关键字参数让长参数列表比 C 的位置参数更容易读。

| 参数 | 可以不传吗 | 不传时使用什么 |
| --- | --- | --- |
| `size` | 不可以 | 没有默认值 |
| `dtype` | 可以 | `"float16"` |
| `device` | 可以 | `"npu"`，但传递时必须写参数名 |

```python
print(allocate(128))
# (128, 'float16', 'npu')

print(allocate(128, device="cpu"))
# (128, 'float16', 'cpu')

print(allocate(128, "float32", device="cpu"))
# (128, 'float32', 'cpu')
```

`allocate()` 会因为缺少 `size` 而报错；`allocate(128, "float32", "cpu")` 会因为 `device` 不能按位置传入而报错。普通 C 没有这种默认实参语法，通常要自己写包装函数或配置结构体。

#### 3.3.6 `*args` 和 `**kwargs` 具体收到什么

```python
def show(first, *args, **kwargs):
    print(first)
    print(args)
    print(kwargs)


show(10, 20, 30, device="cpu", enabled=True)
# 10
# (20, 30)
# {'device': 'cpu', 'enabled': True}
```

`first` 先接收 `10`，剩余位置参数收进元组 `args`，剩余带名字的参数收进字典 `kwargs`。这些名称是惯例，真正决定收集方式的是 `*` 和 `**`。

调用时星号的作用反过来，是展开：

```python
def add(a, b):
    return a + b


values = (10, 20)
options = {"a": 10, "b": 20}
print(add(*values))    # 相当于 add(10, 20)，结果 30
print(add(**options))  # 相当于 add(a=10, b=20)，结果 30
```

调用中，字典展开的键必须适合关键字参数，通常是字符串参数名；如果同一个参数被位置和关键字重复赋值，也会报错。

项目 [v2 model runner](../../vllm_ascend/worker/v2/model_runner.py) 中的源码：

```python
def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
    aux_layers = get_pd_dspark_aux_layer_ids(self.vllm_config)
    super().load_model(load_dummy_weights, *args, **kwargs)
    self.pd_dspark_aux_layer_ids = aux_layers
```

`*args` 收集剩余的位置参数，得到元组；`**kwargs` 收集剩余的关键字参数，得到字典。调用时再次写 `*args, **kwargs`，表示展开并转发。这不同于 C 的不定参数：Python 保留了参数对象及关键字名。

函数默认值在执行函数定义时计算，不是在每次调用时计算，因此不要写：

```python
def collect(value, output=[]):
    output.append(value)
    return output
```

上面的 `output` 会被多次调用共享。调用这个有问题的版本，可以看到：

```python
first = collect(1)
print(first)    # [1]
second = collect(2)
print(second)   # [1, 2]，不是预期的 [2]
print(first)    # [1, 2]，因为还是同一个列表
```

这不是 Python 每次调用都创建列表，而是函数定义时创建了一次，之后一直复用那个默认对象。若每次未传参数都要新的列表，应写下面的版本：

```python
def collect(value, output=None):
    if output is None:
        output = []
    output.append(value)
    return output
```

## 4. 容器、解包、切片与推导式

### 4.1 先认清四种容器

#### 4.1.1 字典的“键”和“值”是什么

```python
person = {
    "name": "张三",
    "age": 18,
}
```

每一项都是 `键: 值`，逗号分隔各项。键是查找用的名字，值是对应的数据：

| 键 | 值 |
| --- | --- |
| `"name"` | `"张三"` |
| `"age"` | `18` |

`person["age"]` 得到 `18`。对 C 开发者，可以暂时类比 `person.age`，但字典的项可以动态增加、删除，键也不局限于固定字段名。

```python
names = {1001: "张三", 1002: "李四"}
print(names[1001])  # 张三
```

这里 `1001` 是键，不是数组的第 1001 个元素。

#### 4.1.2 `dict[key]` 与 `.get(key)` 有什么区别

```python
person = {"name": "张三", "age": 18}
print(person["name"])      # 张三
print(person.get("name"))  # 张三
print(person.get("height"))       # None，没有这个键
print(person.get("height", 170))  # 170，使用指定默认值
# person["height"]  # 如果执行，会抛出 KeyError
```

类比 C 查询表，`.get` 类似“查到返回数据，查不到返回空值或默认值”的接口，而 `[]` 表示你要求这个键必须存在。

`.get("height", 170)` 不会把身高添加到字典。需要添加应写 `person["height"] = 170`。另外，默认值只在键不存在时使用：`{"height": None}.get("height", 170)` 返回的仍是 `None`。

#### 4.1.3 对字典用 `in`，到底检查哪部分

```python
person = {"name": "张三", "age": 18}
print("name" in person)            # True，检查键
print("张三" in person)            # False，没有这个键
print("张三" in person.values())   # True，检查值
print("height" not in person)      # True，not in 表示不包含
```

遍历字典时也要区分：

```python
for key in person:
    print(key)  # 依次输出 name、age

for key, value in person.items():
    print(key, value)  # name 张三，然后 age 18
```

`.keys()` 提供键，`.values()` 提供值，`.items()` 提供键值对。它们通常是反映字典内容的视图，不是提前复制出来的独立列表。

#### 4.1.4 集合是怎么回事

```python
supported = {"linear", "moe", "linear"}
print(len(supported))       # 2，重复元素合并
print("moe" in supported)   # True
supported.add("attention")
```

集合可以类比 C 的哈希集合：适合去重、成员检查，不通过下标读“第几个”。空集合写 `set()`；`{}` 是空字典。下面继续把四种容器放在一起对照。

| 类型 | 示例 | 在本项目中的典型用途 |
| --- | --- | --- |
| `list` | `[1, 2, 3]` | 请求、层、配置项的有序集合 |
| `tuple` | `("W8A8_DYNAMIC", "linear")` | 多返回值、注册表复合键 |
| `dict` | `{"enabled": True}` | 配置、名称到类的映射 |
| `set` | `{"linear", "moe"}` | 成员检查、去重 |

字典键必须可哈希；字符串和由可哈希元素组成的元组可以作为键，列表不可以。字典保持插入顺序，但集合不要当作有固定顺序的序列使用。

### 4.2 多返回值就是元组加解包

先不用算子，写一个普通函数：

```python
def get_position():
    return 10, 20


position = get_position()
print(position)  # (10, 20)
x, y = get_position()
print(x)  # 10
print(y)  # 20
```

`return 10, 20` 实际返回一个元组。`x, y = ...` 把元组的第 0 项交给 `x`，第 1 项交给 `y`，叫解包。类似 C 返回结构体后分别读取字段。

```python
values = [1, 2, 3]
first, *rest = values
print(first)  # 1
print(rest)   # [2, 3]
```

赋值左边的 `*rest` 可以收集剩余元素。没有星号时，接收变量数量必须与右侧元素数量匹配，否则会报 `ValueError`。再看项目如何接收算子的多个返回值：

[归一化算子](../../vllm_ascend/ops/layernorm.py) 中的源码：

```python
x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
    x, residual, self.weight, self.bias, self.variance_epsilon
)
```

可以类比 C 返回一个包含多个字段的结果结构体，然后把字段分别取出。这里要求右侧返回三个可解包的元素。`_` 只是习惯上表示“不使用”的普通变量名，仍然会接收对象。

特别注意该文件的 `forward_oot`：有 residual 时返回 `(x, residual)`，没有时返回 `x`。阅读调用方时必须核对分支，不能一律按二元组解包。

### 4.3 切片与推导式

切片的完整形式是 `sequence[start:stop:step]`：起点包含，终点不包含，步长默认为 1。

```python
values = [10, 20, 30, 40]
assert values[1:3] == [20, 30]  # 左闭右开
assert values[-1] == 40
assert values[::2] == [10, 30]

squares = [value * value for value in values if value > 20]
assert squares == [900, 1600]
```

推导式相当于简洁的“创建容器、循环、过滤、追加”。列表切片通常创建新列表，但里面的对象仍然共享；Tensor 的基本切片通常产生共享存储的视图，第 12 章会说明。

上面的推导式可以展开成普通循环：

```python
squares = []
for value in values:
    if value > 20:
        squares.append(value * value)
```

读 `[表达式 for 变量 in 数据 if 条件]` 时，可以先看 `for`，再看 `if`，最后看最前面的表达式。字典也有类似语法：

```python
lengths = {name: len(name) for name in ["moe", "norm"]}
print(lengths)  # {'moe': 3, 'norm': 4}
```

冒号左边计算键，右边计算值。遇到复杂嵌套推导式时，先展开成循环理解，不必为了简短强行使用。

`any(condition(x) for x in values)` 中圆括号内是生成器表达式，按需逐个计算；`any` 遇到真值会停止。`all` 遇到假值会停止。

项目 `layernorm.py` 使用 `any("norm.bias" in name for name in quant_description)`：逐个检查字典键名，遇到匹配项即可结束。

例如 `any([False, True, False])` 是 `True`，`all([True, True])` 是 `True`。这里 `any` 类似 C 循环中“找到一项匹配就返回真”；`all` 是“所有项都匹配才返回真”。空输入时 `any` 为假，`all` 为真。

## 5. 类与继承：从 struct 加函数指针开始理解

### 5.1 `self` 是显式写出的实例参数

这里先把三个词区分开：类是定义，实例是根据定义创建的对象，方法是写在类里的函数。可以从 C 的结构体和操作函数迁移理解。

C 教学示例：

```c
struct Runner {
    int max_tokens;
};

int runner_can_run(const struct Runner *self, int count) {
    return count <= self->max_tokens;
}
```

对应 Python 教学示例：

```python
class Runner:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    def can_run(self, count):
        return count <= self.max_tokens


runner = Runner(128)
assert runner.can_run(64)
```

`self.max_tokens` 类似 `self->max_tokens`。调用 `runner.can_run(64)` 时，Python 自动把 `runner` 作为第一个参数；可理解为 `Runner.can_run(runner, 64)`。

`__init__` 在实例创建后初始化状态，不等同于 C 的分配内存；更底层的创建由 `__new__` 负责，初学本项目一般不需要修改它。

#### 5.1.1 `self` 在哪里定义，为什么调用时不用传

它就定义在参数列表的第一个位置：`def can_run(self, count)`。`self` 不是 Python 关键字，只是大家统一采用的名字。通过 `runner.can_run(64)` 调用时，Python 会自动把 `runner` 传给这个参数。

因此下面两个调用在这个例子中等效：

```python
runner.can_run(64)
Runner.can_run(runner, 64)
```

可以类比 C 显式调用 `runner_can_run(&runner, 64)`。Python 把传实例的步骤帮你做好了。

#### 5.1.2 为什么能写 `self.vllm_config`

属性表示保存在对象上的数据。例如：

```python
class Runner:
    def __init__(self, config):
        self.vllm_config = config

    def show_config(self):
        print(self.vllm_config)


first = Runner("配置 A")
second = Runner("配置 B")
first.show_config()   # 配置 A
second.show_config()  # 配置 B
```

创建 `first` 时，`__init__` 的 `self` 是 `first`；创建 `second` 时则是 `second`。所以实例可以保存各自的数据。每次调用方法时，不需要再次把配置单独传进去。

类比 C：

```c
struct Runner {
    const char *vllm_config;
};

void show_config(struct Runner *self) {
    printf("%s\n", self->vllm_config);
}
```

注意 `self.vllm_config` 中 `vllm_config` 是完整的属性名。`self.vllm.config` 则表示“先取 self 的 vllm 属性，再取该对象的 config 属性”，是另一回事。

Python 普通对象通常可以通过赋值创建实例属性；访问尚未设置且类上也找不到的属性通常会抛 `AttributeError`。实际类还可能用属性描述符等机制控制访问，不能假定所有字段都是直接存储。

#### 5.1.3 把项目的 `load_model` 逐行读一遍

对前面的项目代码，可以按下面顺序翻译：

1. `self` 是这次调用使用的 runner 实例。
2. `load_dummy_weights=False` 是可省略的开关参数；`*args`、`**kwargs` 接收其他参数。
3. `self.vllm_config` 读取实例先前保存的配置；该属性也可能由父类初始化逻辑设置。
4. `get_pd_dspark_aux_layer_ids(...)` 计算结果，先放在局部名字 `aux_layers` 中。
5. `super().load_model(...)` 调用继承链的加载实现，并把参数转发过去。
6. `self.pd_dspark_aux_layer_ids = aux_layers` 把结果保存为实例属性，其他方法以后也能读取。

局部名字 `aux_layers` 和实例属性不是一个作用域：其他方法不能直接访问这里的局部名字，但可以通过同一实例读取 `self.pd_dspark_aux_layer_ids`。`-> None` 说明这个方法正常完成时不返回有用结果，并不表示方法什么也没做。

### 5.2 继承、覆盖与 `super()`

项目中的真实继承关系包括：

```python
class AscendRMSNorm(RMSNorm):
    ...


class NPUModelRunner(GPUModelRunner):
    ...
```

可以类比 C 中“基础结构体 + 一组可替换的操作函数”。子类复用基础实现，并覆盖设备相关行为。`super().load_model(...)` 继续调用方法解析顺序中的后续实现；在简单单继承里通常就是父类实现，多继承时不能简单当作写死的父类名。

项目是硬件插件，许多基础类定义在上游 `vllm` 包中。只看插件文件找不到某个方法时，要继续查父类和上游代码。

先看继承与 `super()` 的独立教学例子：

```python
class BaseRunner:
    def load_model(self):
        print("基础加载")


class NPURunner(BaseRunner):
    def load_model(self):
        print("准备 NPU")
        super().load_model()
        print("NPU 加载完成")


runner = NPURunner()
runner.load_model()
# 准备 NPU
# 基础加载
# NPU 加载完成
```

子类定义同名方法叫覆盖。Python 调用子类的方法，在其中通过 `super()` 复用基础实现。不要改成 `self.load_model()` 来调用基础实现，那会再次进入当前方法，导致递归。

### 5.3 实例方法、类方法、静态方法和属性

| 写法 | 自动收到的参数 | 典型用途 |
| --- | --- | --- |
| `def method(self, ...)` | 实例 | 使用当前对象状态 |
| `@classmethod` 后的 `def method(cls, ...)` | 类 | 从配置构造实例，使用 `cls(...)` |
| `@staticmethod` 后的函数 | 无实例或类参数 | 与类相关的工具函数 |
| `@property` 后的 `def value(self)` | 实例 | 用 `obj.value` 触发计算或查询 |

[Ascend 配置](../../vllm_ascend/ascend_config.py) 的 `SchedulerConfig.from_additional_config` 是类方法；v2 runner 的 `pcp_manager_cls` 是属性。看到 `obj.name` 时，不要假定它一定是直接读取存储字段。

把表格里的几种形式具体写出来：

```python
class Config:
    def __init__(self, size):
        self.size = size

    @classmethod
    def from_dict(cls, data):
        return cls(data["size"])

    @staticmethod
    def is_valid_size(size):
        return size > 0

    @property
    def double_size(self):
        return self.size * 2


config = Config.from_dict({"size": 128})
print(config.size)                # 128
print(Config.is_valid_size(128))  # True
print(config.double_size)         # 256，注意这里没有括号
```

`cls` 是当前类，所以 `cls(128)` 在此例中相当于 `Config(128)`。静态方法不自动收到 `self`，参数全部由调用者提供。属性则把函数调用包装成字段读取语法，每次读取可能执行计算。

这些 `@...` 都是装饰器，基础机制在第 6 章说明。不要只记名字，要看它们改变了调用参数还是访问方式。

### 5.4 抽象基类相当于要求实现一组接口

```python
from abc import ABC, abstractmethod


class Scheme(ABC):
    @abstractmethod
    def apply(self, value):
        pass


class DoubleScheme(Scheme):
    def apply(self, value):
        return value * 2


scheme = DoubleScheme()
print(scheme.apply(3))  # 6
# Scheme()  # 如果执行，会报错：抽象方法尚未实现
```

`pass` 是空操作，占住语法要求必须有语句的位置，不是返回特殊值。抽象基类可以写通用逻辑，但把部分接口留给子类实现。C 中常靠 ops 表约定接口，Python 的 ABC 机制能阻止直接实例化仍缺少抽象实现的类。

[量化接口](../../vllm_ascend/quantization/methods/base.py) 中有 `AscendLinearScheme(WeightSwitchMixin, ABC)`，并用 `@abstractmethod` 标记 `get_weight`、`apply` 等方法。

对 C 开发者，可以类比“驱动必须提供的 ops 表”。抽象方法未实现的类不能正常实例化。`Mixin` 通常提供一组可复用方法，不一定是完整的独立对象模型。

不要照搬 C 的共享配置方式：类体里的 `items = []` 是类属性，实例通常会共享；实例独立数据应在 `__init__` 中写 `self.items = []`。

## 6. 函数也是对象：回调、闭包与装饰器

### 6.1 函数名后面有无括号，含义完全不同

```python
def answer():
    return 42


callback = answer  # 保存函数对象，类似保存函数指针
result = answer()  # 立即调用函数
assert callback() == result
```

函数对象能存入字典、作为参数传递、作为返回值返回。类对象也能存入字典，再通过 `cls(...)` 构造实例。

### 6.2 用项目注册表理解装饰器

#### 6.2.1 先把闭包理解成“函数加保留下来的参数”

```python
def make_adder(base):
    def add(value):
        return base + value
    return add


add_ten = make_adder(10)
print(add_ten(3))  # 13
print(add_ten(8))  # 18
```

执行过程是：先调用 `make_adder(10)`；它定义内部函数 `add` 并返回这个函数，而不是调用它；之后即使外层函数已经返回，内部函数仍然能访问保留的 `base` 绑定。这叫闭包。

在 C 中通常要用“函数指针 + context 结构体”组合传递同类信息。Python 把这类状态访问封装在函数对象相关的闭包中。它不是所有外部变量值的冻结快照，循环创建闭包时尤其要注意它捕获的是绑定。

#### 6.2.2 装饰器先看不带参数的版本

```python
def mark(cls):
    cls.description = "已标记"
    return cls


@mark
class Example:
    pass


print(Example.description)  # 已标记
```

这里的 `@mark` 相当于定义类之后再写 `Example = mark(Example)`。装饰器收到类对象，处理后返回一个对象，名字 `Example` 最终绑定到返回结果。装饰器也可以用于函数，不是 C 的预处理宏。

下面看项目中“装饰器工厂返回装饰器”的版本：

[registry.py](../../vllm_ascend/quantization/methods/registry.py) 的关键源码：

```python
def register_scheme(quant_type: str, layer_type: str):
    def decorator(cls: type[Any]) -> type[Any]:
        key = (quant_type, layer_type)
        if key in _SCHEME_REGISTRY:
            raise ValueError(
                f"Scheme already registered for {quant_type}/{layer_type}: {_SCHEME_REGISTRY[key].__name__}"
            )
        _SCHEME_REGISTRY[key] = cls
        return cls

    return decorator
```

阅读步骤：

1. 调用 `register_scheme("W8A8_DYNAMIC", "linear")`，得到内部函数 `decorator`。
2. 内部函数保留对外层 `quant_type` 和 `layer_type` 的访问能力，这叫闭包。
3. 把一个类传给 `decorator`，就把这个类存进注册表。
4. 返回原类，因此注册后仍可以正常使用这个类。

装饰器教学示例：

```python
@register_scheme("TEACHING", "linear")
class TeachingScheme:
    pass
```

在这里可以展开理解为：

```python
class TeachingScheme:
    pass


TeachingScheme = register_scheme("TEACHING", "linear")(TeachingScheme)
```

类定义和装饰器应用发生在执行到这段定义时，通常就是导入其模块时；不是等第一次创建实例才注册。相应模块没被导入，注册表中就可能没有该项。

现有注册表使用模块级可变字典，是需要理解的项目机制；根据仓库规范，新加可变全局状态需要审批，不能把这个例子当作随意增加全局缓存的建议。

### 6.3 `lambda` 与延迟求值

`lambda` 可以先按“很短的匿名函数”理解：

```python
double = lambda value: value * 2
print(double(3))  # 6
```

它在这里相当于：

```python
def double(value):
    return value * 2
```

`lambda` 冒号前是参数，冒号后是返回表达式，不写 `return`。`lambda: 42` 冒号前没有参数，表示调用时不传参数。

延迟求值可以通过这个无硬件例子观察：

```python
settings = {"size": 10}
readers = {"size": lambda: settings["size"]}
settings["size"] = 20
print(readers["size"]())  # 20
```

`readers["size"]` 先取出函数，末尾 `()` 再执行函数；得到的是调用时的值。若当初直接保存 `settings["size"]`，字典中保存的就是当时的整数 `10`。项目的环境变量读取利用了类似机制。

[envs.py](../../vllm_ascend/envs.py) 不是把所有环境变量直接读成值，而是保存读取函数。例如源码：

```python
"COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
```

`lambda: expression` 是没有参数、返回该表达式结果的小函数。字典里存的是函数，调用时才读取环境变量。

同文件的模块级源码：

```python
def __getattr__(name: str):
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

访问模块中常规查找未找到的属性时，会进入这个钩子。因此 `envs.COMPILE_CUSTOM_KERNELS` 看起来是读字段，实际可能正在调用读取函数。这里本身没有缓存返回值。

不要写 `bool("0")` 来解析开关：非空字符串都为真。项目中有整数转换，也有严格检查 `"0"`、`"1"` 的辅助函数，具体以该配置定义为准。新增环境变量应遵守集中定义和文档要求。

## 7. 模块与导入：既加载名字，也执行代码

### 7.1 `import` 不等于 `#include`

“模块”通常就是一个可导入的 Python 文件；“包”把多个模块组织在一个导入名字下面。例如 `vllm_ascend/envs.py` 对应模块名 `vllm_ascend.envs`，点号表示导入层级。

假设在同一目录写两个教学文件。第一个是 `helper.py`：

```python
print("helper 正在初始化")


def double(value):
    return value * 2
```

第二个是 `main.py`：

```python
import helper
import helper

print(helper.double(3))
```

直接执行 `main.py` 时，通常输出一次“helper 正在初始化”，再输出 `6`。第二次导入通常复用缓存，不再次执行顶层代码。这解释了为什么算子注册、装饰器、patch 的导入时机都很重要。

如果不想某段启动代码在导入时执行，可以写：

```python
def main():
    print("执行程序入口")


if __name__ == "__main__":
    main()
```

直接运行该文件时，`__name__` 为 `"__main__"`；作为普通模块导入时则是模块名。类似把“可复用函数”和“程序 main 入口”区分开，但 Python 仍会先执行模块顶层语句。

`#include` 是预处理文本包含。Python `import` 会寻找模块、初始化模块对象，并执行模块顶层代码；通常随后复用 `sys.modules` 中的模块对象。

| 语句 | 读法 |
| --- | --- |
| `import torch` | 通过 `torch.xxx` 使用其属性 |
| `from torch import nn` | 把当时导出的 `nn` 对象绑定到当前模块名字 |
| `from . import envs` | 从当前包导入 `envs` |
| `import package.module as alias` | 为模块对象使用本地别名 |

普通包通常通过 `__init__.py` 初始化；Python 也支持不带它的命名空间包。本项目按仓库检查要求组织自己的包文件。

特别注意：`from module import name` 后，如果原模块把 `name` 重新绑定成另一个对象，已经导入的本地名字不会自动跟着重新绑定。这对 patch 和 mock 的理解很重要。

### 7.2 插件入口是如何连起来的

本地 [setup.py](../../setup.py) 的平台插件入口是 `ascend = vllm_ascend:register`。这里的意思是“到 `vllm_ascend` 模块找 `register` 函数”，不是 Python 运算表达式。

[包初始化文件](../../vllm_ascend/__init__.py) 的 `register()` 返回字符串 `"vllm_ascend.platform.NPUPlatform"`，由框架按插件机制解析。

从 C 的角度，可以类比“动态加载插件后，找到约定入口，再取得平台实现”。Python 通过包元数据、模块名和对象名完成这件事。

### 7.3 延迟导入、类型导入与循环依赖

循环依赖的简单情况是：`a.py` 导入 `b.py`，而 `b.py` 又在初始化期间导入并使用 `a.py` 中尚未定义的内容。这不是 C 链接时找符号的问题，而是模块初始化顺序问题。

函数内导入有时能把“马上需要这个模块”推迟为“执行功能时才需要”，但不能随意靠移动 import 掩盖结构问题；也可以考虑把公共类型或函数提取到独立模块。

某些函数内部才 `import torch_npu`：导入推迟到该功能执行时，可能涉及可选依赖、工作进程隔离、启动顺序或循环依赖。仓库默认要求导入放文件顶部，函数内导入是有原因的例外。

项目常见写法：

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
```

`TYPE_CHECKING` 在普通运行时为假，静态分析器会分析其中导入。`from __future__ import annotations` 在本项目这种用法中允许延迟解析注解，帮助减少仅因类型标注而产生的运行时导入依赖。若代码运行时需要实际类对象，仍然需要真正导入，不能只放在这里。

### 7.4 动态属性与 monkey patch

先看按字符串访问属性：

```python
class Config:
    def __init__(self):
        self.size = 128


config = Config()
print(getattr(config, "size"))           # 128，类似 config.size
print(getattr(config, "missing", None))  # None，没有这个属性则用默认值
setattr(config, "size", 256)             # 类似 config.size = 256
print(hasattr(config, "size"))           # True
```

这和字典的 `get` 有相似的“名字查找”感觉，但作用对象不同：`getattr` 查对象属性，`dict.get` 查字典键，不能混为一个接口。

`getattr(obj, "name", default)` 是按字符串读属性并提供缺失默认值；`hasattr` 检查属性；`setattr` 按字符串写属性。这比 C 的固定结构体字段访问更动态，属性读取本身也可能执行逻辑。

教学示例：

```python
class Base:
    def forward(self, value):
        return value


def npu_forward(self, value):
    return value + 1


Base.forward = npu_forward
assert Base().forward(3) == 4
```

这叫 monkey patch，可以类比运行时替换共享函数表的入口，但 Python 实际修改的是类属性。项目 [patch 目录](../../vllm_ascend/patch/) 包含这类适配。修改类方法通常会影响随后通过该类查找方法的实例；已保存的方法对象、子类覆盖等情况还要另看。

新增 patch 需要仓库要求的架构审查。阅读时先找“谁被替换、何时替换、哪些调用方受影响”。

## 8. 异常与资源清理：从错误码到调用栈展开

### 8.1 用 `raise` 报错，用 `except` 在合适边界处理

先认识异常的三个动作：`raise` 产生并抛出异常，`try` 包住可能失败的调用，`except` 接住匹配类型的异常。

教学示例：

```python
def require_positive(size):
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    return size


try:
    require_positive(0)
except ValueError as error:
    message = str(error)
    assert "positive" in message
```

和 C 每层返回错误码相比，未捕获异常会自动向调用者传播，并保留 traceback。应在能够恢复或补充上下文的位置处理，不要普遍使用 `except Exception: pass` 隐藏错误。

上例运行顺序是：调用函数 → 发现 `size <= 0` → 抛出 `ValueError`，函数剩余代码不再执行 → 跳到调用方的 `except ValueError` → `as error` 把异常对象绑定到名字 `error` → 读取错误信息。

如果抛出的是其他类型，而这里没有对应处理，就继续向更外层传播。可以类比 C 的失败返回一路传给上层，但 Python 自动跳过未处理的调用帧，不需要每层手写 `if (ret != 0)`。

```python
try:
    result = int("abc")
except ValueError:
    print("不是有效整数")
else:
    print("转换成功", result)
finally:
    print("本次转换结束")
# 不是有效整数
# 本次转换结束
```

`else` 只在 `try` 正常完成时执行；`finally` 负责离开时的清理。实际程序被强制终止等情况不属于正常的异常展开，不能把清理保证延伸到所有进程退出方式。

`assert` 常用于测试和内部不变量。Python 优化模式可能移除断言，外部输入和配置校验应显式抛异常。

### 8.2 `finally` 与 `with` 对应成对的清理操作

先看类似 C 的手动配对方式：

```python
file = open("notes.txt", "w", encoding="utf-8")
try:
    file.write("hello\n")
finally:
    file.close()
```

`open` 类似 `fopen`，`close` 类似 `fclose`。即使 `write` 失败，正常异常展开时也会执行 `close`。下面的 `with` 把这种配对规则收进对象的上下文协议：

`finally` 在正常离开、异常、`return` 等控制路径中执行，适合恢复状态。文件等资源通常用：

```python
with open("notes.txt", "w", encoding="utf-8") as file:
    file.write("hello\n")
```

这可以类比确保 `fopen` 后执行 `fclose` 的清理路径。对象的 `__enter__` 在进入时执行，`__exit__` 在退出时执行。上下文管理器也可以管理设备流、临时状态，而不只是文件。

### 8.3 对照 runner 中的真实状态恢复

在读 runner 之前，先运行这个不需要硬件的完整示例：

```python
from contextlib import contextmanager


class State:
    def __init__(self):
        self.limit = 100


@contextmanager
def temporary_limit(state, new_limit):
    old_limit = state.limit
    state.limit = new_limit
    try:
        yield
    finally:
        state.limit = old_limit


state = State()
try:
    with temporary_limit(state, 10):
        print(state.limit)  # 10
        raise ValueError("模拟失败")
except ValueError:
    pass

print(state.limit)  # 100，异常后也恢复
```

进入 `with` 时执行到 `yield`；接着执行 `with` 缩进的代码；代码结束或抛异常时恢复生成器，进入清理逻辑。这类似 C 的“保存旧值 → 设置临时值 → 执行操作 → cleanup 恢复旧值”。

这里的 `yield` 不是线程切换，也不是函数彻底返回。普通生成器可以用下面的例子理解暂停和恢复：

```python
def numbers():
    yield 10
    yield 20


iterator = numbers()
print(next(iterator))  # 10，暂停在第一个 yield
print(next(iterator))  # 20，继续运行到第二个 yield
```

`@contextmanager` 要求一次上下文执行中恰好产生一次 `yield`，而普通生成器可以产生多个值。不要把两个用法混淆。

[v2 runner](../../vllm_ascend/worker/v2/model_runner.py) 的 `_cap_parallel_draft_dummy_reqs` 使用以下结构，下面省略了中间的状态调整逻辑：

```python
@contextmanager
def temporary_state(self):
    original_max_num_reqs = self.max_num_reqs
    try:
        yield
    finally:
        self.max_num_reqs = original_max_num_reqs
```

这是改名和删减后的教学示意，需要 `from contextlib import contextmanager`。`yield` 前是进入逻辑，调用方 `with` 的代码块在此期间执行，之后恢复状态。即使代码块抛异常，`finally` 仍会执行。

`yield` 的基础含义是暂停生成器并保留执行状态；配合 `@contextmanager` 时，它被用来划分进入和退出阶段。这两个概念有关联，但普通生成器不一定是上下文管理器。

同时进入多个上下文应写 `with context_a(), context_b():`，不要用 `with context_a() and context_b():`，后者是求逻辑表达式，不能保证进入两个上下文。

## 9. 配置对象：dataclass、默认工厂与验证

### 9.1 不用 dataclass 时，你需要手写什么

```python
class Config:
    def __init__(self, max_tokens=128):
        self.max_tokens = max_tokens


config = Config(256)
print(config.max_tokens)  # 256
```

dataclass 根据字段说明生成这类初始化逻辑，以及默认的展示和相等比较逻辑。它没有把 Python 变成 C 的固定内存布局结构体，也不自动保证输入合理。

### 9.2 `field(default_factory=list)` 每次创建一个新列表

可以把 dataclass 理解为“自动生成初始化和展示等方法的配置结构体”。标准库教学示例：

```python
from dataclasses import dataclass, field


@dataclass
class Config:
    max_tokens: int = 128
    layers: list[str] = field(default_factory=list)


a = Config()
b = Config()
a.layers.append("attention")
assert b.layers == []
```

`default_factory=list` 表示每次创建实例时调用 `list()`，得到独立列表。项目 `SchedulerConfig` 也通过 `dataclasses.field(default_factory=...)` 构造独立子配置。

注意这里传的是 `list` 这个可调用对象，不是 `list()` 的执行结果。初始化时大致相当于每次执行 `self.layers = list()`。这与第 6 章“保存函数而不是调用结果”是同一类区别。

### 9.3 “有字段”与“字段合法”是两回事

普通 dataclass 不会因为 `max_tokens: int` 就自动拒绝所有错误类型。本项目 [ascend_config.py](../../vllm_ascend/ascend_config.py) 还使用来自 `config_utils` 的 `@config`、Pydantic 的验证器等机制，不能把它们全部等同于普通 `@dataclass`。

阅读配置时依次检查：字段默认值、配置来源和优先级、转换逻辑、验证器、最终使用位置。例如 `SchedulerConfig.from_additional_config` 明确处理嵌套配置和旧顶层配置的优先级。

不用框架，也能显式验证：

```python
def validate_config(data):
    size = data.get("max_tokens", 128)
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError("max_tokens 必须是整数")
    if size <= 0:
        raise ValueError("max_tokens 必须大于零")
    return size
```

`isinstance(value, int)` 检查对象是否为该类型或其子类的实例。Python 的 `bool` 是 `int` 的子类，这个示例额外排除布尔值，避免 `True` 被当成容量 `1`。实际配置约束以项目实现为准。

框架验证器把这类规则组织成可复用逻辑。遇到 `@field_validator`、`@model_validator`，先找它检查哪些字段、何时运行、失败抛什么，而不是只读类型注解。

另外，`value = config.get("count") or default` 会把 `0` 当作缺失；允许零值时，应显式区分 `None`。

## 10. 测试：把驱动桩和函数替换经验迁移过来

### 10.1 项目同时出现 pytest 和 unittest 风格

先写最小的 pytest 教学文件 `test_add.py`：

```python
def add(a, b):
    return a + b


def test_add():
    assert add(1, 2) == 3
```

在已安装 pytest 的环境中运行 `python -m pytest -q test_add.py`。pytest 按命名规则发现测试函数并执行断言；正常时汇总通过数量，失败时显示出错的表达式和值。它类似 C 测试框架帮你寻找和运行测试用例。

参数化可以减少重复测试代码：

```python
import pytest


@pytest.mark.parametrize("a,b,expected", [(1, 2, 3), (-1, 1, 0)])
def test_add_cases(a, b, expected):
    assert a + b == expected
```

这会分别用两组参数运行一次测试。字符串列出形参名，列表里的每个元组提供一组实参，不是把整个列表当一次输入。

[注册表测试](../../tests/ut/quantization/methods/test_registry.py) 使用 `TestBase`、`self.assertIs`、`self.assertRaises`，并由 pytest 运行。

| 测试写法 | 含义 |
| --- | --- |
| `assert actual == expected` | 验证值相等 |
| `self.assertIs(actual, expected)` | 验证是同一对象，适合类注册表 |
| `self.assertIsNone(result)` | 验证找不到时返回空状态 |
| `with self.assertRaises(ValueError):` | 验证代码块确实抛出预期异常 |
| `@pytest.mark.parametrize(...)` | 用多组参数执行同一个测试 |
| `@pytest.fixture` | 定义 pytest 可注入的测试准备及清理逻辑 |

Fixture 可以类比可复用的 setup/teardown，但不能简单理解为全局变量；要看它的作用域和清理逻辑。

```python
import pytest


@pytest.fixture
def settings():
    return {"size": 128}


def test_settings(settings):
    assert settings["size"] == 128
```

这里测试函数的 `settings` 参数由 pytest 按 fixture 名称提供，不是 Python 对所有同名参数都有自动注入机制。默认函数级 fixture 通常为每次测试请求创建一次，其他作用域可能共享状态；fixture 也能通过 `yield` 安排清理。

### 10.2 mock 可以替换硬件调用，但不证明硬件结果正确

[归一化测试](../../tests/ut/ops/test_layernorm.py) 使用 `unittest.mock.patch` 替换部分算子调用。可类比 C 测试中链接桩函数或替换函数指针。

独立教学示例：

```python
from unittest.mock import patch


class Device:
    @staticmethod
    def compute(value):
        return value * 2


with patch.object(Device, "compute", return_value=7) as mocked:
    assert Device.compute(3) == 7
    mocked.assert_called_once_with(3)

assert Device.compute(3) == 6
```

使用字符串形式的 `patch("package.module.name")` 时，要替换被测代码实际查找该名字的位置。例如被测模块通过 `from backend import run` 保存了本地名字，通常应 patch 被测模块中的 `run`。

Mock 测试能检查分支、参数和调用关系；数值精度、设备支持、异步行为和性能仍需要真实环境验证。不要因为测试里是 CPU Tensor，就断定对应生产逻辑不需要 NPU。

上面的 mock 例子可以逐行理解为：

1. 原来的 `Device.compute(3)` 返回 `6`。
2. 进入 `with patch.object(...)`，暂时替换 `compute` 属性。
3. `return_value=7` 使替换后的调用返回固定值 `7`。
4. `as mocked` 保存模拟对象，它会记录收到过哪些调用。
5. `assert_called_once_with(3)` 检查是否只调用一次，且参数就是 `3`。
6. 离开 `with` 自动恢复原方法，所以再次调用得到 `6`。

这是第 8 章上下文管理器在测试里的实际用途。`side_effect` 则可指定替代函数或抛出的异常，不限于固定返回值。

在配置好的项目测试环境中，可运行：

```bash
pytest -sv tests/ut/quantization/methods/test_registry.py
pytest -sv tests/ut/ops/test_layernorm.py
```

## 11. Python 与底层算子的边界

### 11.1 一行 Python 为什么能处理大量数据

假设 C 中有一个对数组做归一化的函数，你不必在调用者里重写内部循环，只需准备指针、长度和参数。PyTorch 也通过扩展和算子分发调用底层实现，Python 负责组织调用，并不意味着每个元素都在解释器中计算。

本项目的 Python 代码主要承担配置、注册、调度、张量组织和调用算子的工作。看到一行 Tensor 运算，不代表数据是由 Python 循环逐元素处理。

| 源码入口 | 需要建立的认识 |
| --- | --- |
| `torch_npu.npu_rms_norm(...)` | 调用 Ascend 相关的张量算子接口 |
| `torch.ops._C_ascend.npu_add_rms_norm_bias(...)` | 通过 PyTorch dispatcher 调用已注册的自定义算子 |
| `import vllm_ascend.vllm_ascend_C` | 加载项目编译扩展，可能产生算子注册等副作用 |
| [csrc](../../csrc/) | 包含底层适配、host 与 kernel 等实现 |
| [ops/triton](../../vllm_ascend/ops/triton/) | 包含 Triton kernel，不能完全按普通 Python 执行模型理解 |

从 C 调用链思维出发，可以先跟踪“Python 组织参数 → 算子注册和分发 → 底层实现”。不要把 Python 中的算子名字直接当成同名 C 函数，也不要假定调用返回时设备计算已经完成。

### 11.2 把长名字按点号拆开读

```python
torch.ops._C_ascend.npu_add_rms_norm_bias(...)
```

从左往右：`torch` 是模块；`ops` 是访问已注册算子的入口；`_C_ascend` 是这里使用的算子命名空间；最后一个名字是具体算子。末尾括号才表示真正调用。

不要把这串点号理解成 C 头文件路径，也不要因为看到名字就认定算子已经注册。注册依赖编译扩展和初始化逻辑；如果环境或加载顺序不对，可能找不到算子。

### 11.3 先核对调用协议，再找底层循环

对某一行算子调用，可以先在纸上写：

```text
输入：x、residual、weight、bias、epsilon
输入信息：每个 Tensor 的 shape、dtype、device、布局
返回：几个结果，分别由谁接收
副作用：是否修改输入，是否共享存储，是否要求同步
```

这相当于先读 C 函数原型和缓冲区约定，再进入实现。Triton 文件虽然使用 Python 形式定义，但 kernel 中的表达式可能按编译和设备执行规则处理，不能拿普通 Python 的逐行主机执行方式直接套用。

## 12. 对 C 开发者最关键的 Tensor 知识

### 12.1 把 Tensor 看成带元数据的设备缓冲区视图

#### 12.1.1 先创建一个能看见内容的 CPU Tensor

以下示例需要安装 PyTorch，不需要 NPU：

```python
import torch


x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(x.shape)   # torch.Size([2, 3])，2 行 3 列
print(x.dtype)   # torch.float32
print(x.device)  # cpu
print(x.numel()) # 6
print(x[0, 1])   # tensor(2.)
```

`torch` 是模块，`torch.tensor` 是构造数据的接口，`torch.Tensor` 是张量类型的名字。小写 `tensor` 和大写 `Tensor` 在这里不要混淆。

它可以类比一个 `float data[2][3]` 加上描述信息。不过 Tensor 可能位于 NPU、存在不连续布局、与其他 Tensor 共享存储，所以不能直接等同于二维 C 数组。

#### 12.1.2 shape 和 stride 各说明什么

```python
x = torch.arange(6).reshape(2, 3)
print(x.shape)     # torch.Size([2, 3])
print(x.stride())  # (3, 1)

y = x.transpose(0, 1)
print(y.shape)     # torch.Size([3, 2])
print(y.stride())  # (1, 3)
```

`arange(6)` 生成 `0` 到 `5`，`reshape(2, 3)` 整理为两行三列。原来跨一行需要越过 3 个元素，跨一列越过 1 个元素，因此 stride 是 `(3, 1)`。

转置后，新的行移动对应原来的列移动，所以步长变成 `(1, 3)`。这通常不需要把整块数据重新搬一遍，而是修改访问方式。可以类比 C 中数据地址不变，但你改了计算下标偏移的公式。

对于普通 strided Tensor，元素偏移可按 `storage_offset + i * stride[0] + j * stride[1]` 理解，再结合元素字节数理解实际地址；不要把这个模型当成所有稀疏、特殊布局 Tensor 的通用存储实现。

#### 12.1.3 `reshape(-1, ...)` 中的 `-1` 是什么

```python
x = torch.arange(12)
y = x.reshape(-1, 3)
print(y.shape)  # torch.Size([4, 3])
```

`-1` 表示让框架根据总元素数推算这一维，这里是 `12 / 3 = 4`。通常只能指定一个待推算维度；总元素数必须匹配。

项目的 `x.reshape(-1, x.shape[-1])` 表示“保留最后一维长度，把其他维度合并”。例如形状 `(2, 4, 8)` 会变成 `(8, 8)`。`x.shape[-1]` 中 `-1` 是 Python 的倒数第一个下标，与 reshape 参数中“推算维度”的含义不同。

粗略类比为“数据存储 + dtype + device + shape + stride + storage offset”，而不是一个普通嵌套列表。这只是帮助理解的模型，不是 PyTorch 的精确 ABI 定义。

| 属性或操作 | 对 C 开发者的解释 |
| --- | --- |
| `x.shape` | 各维度长度 |
| `x.dtype` | 元素表示，例如 float16、bfloat16、int32 |
| `x.device` | 数据所在设备 |
| `x.stride()` | 每个维度增加 1 时跨越的元素数，不是字节数 |
| `x.numel()` | 总元素个数 |
| `x.reshape(...)` | 重解释形状，能共享存储时共享，否则可能复制 |
| `x.view(...)` | 要求布局满足相应视图条件，否则可能报错 |
| `x.contiguous()` | 必要时生成指定连续布局；已经满足时可能返回自身 |
| `x.clone()` | 复制张量数据，得到独立存储 |
| `x.to(...)` | 转设备或 dtype；条件已经满足时可能返回自身 |
| `x.detach()` | 脱离自动求导关系，通常仍与原张量共享存储 |

项目 `LayerNormFn.forward` 中先 `reshape`，再检查 `stride(-1)` 并在必要时 `contiguous()`：这是在准备 kernel 需要的布局，不是语法上的随意转换。

### 12.2 共享存储与原地操作

这里区分三种操作：`b = x` 只复制对象引用；`b = x[:2]` 通常产生新的 Tensor 视图对象，但共享部分数据；`b = x.clone()` 复制数据，得到独立存储。

下面的教学示例只需要安装 PyTorch，使用 CPU 即可：

```python
import torch


x = torch.tensor([1, 2, 3])
y = x[:2]
y.add_(10)
assert x.tolist() == [11, 12, 3]

z = x.clone()
z.zero_()
assert x.tolist() == [11, 12, 3]
```

`add_`、`copy_`、`zero_` 等末尾下划线通常表示原地操作。它们可能减少分配，但会影响所有共享存储的视图，并可能受自动求导约束。`x = x + bias` 与 `x.add_(bias)` 在别名和内存行为上不一样。

`x is y` 只能判断是不是同一个 Python 对象；不同 Tensor 对象也可能引用同一存储区域。

### 12.3 广播不是简单的 C 数组逐项相加

先看具体数字：

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
bias = torch.tensor([10, 20, 30])
y = x + bias
print(y.tolist())
# [[11, 22, 33], [14, 25, 36]]
```

`bias` 被用于每一行。相当于 C 双层循环执行 `y[i][j] = x[i][j] + bias[j]`。这不要求先手动创建一份两行的 bias 数组。

设 `x.shape == (tokens, hidden_size)`，`bias.shape == (hidden_size,)`，`x + bias` 可以沿 token 维广播。广播从末尾维度对齐，维度相等或其中一个为 1 时兼容。

不用手写双重 Python 循环。阅读时先把每个输入、输出的 shape 写出来，再判断布局和算子要求。

### 12.4 `.item()` 和 Python 分支可能触发设备同步

`.item()` 先从最简单的 CPU 情况理解：

```python
value = torch.tensor(7)
number = value.item()
print(number)        # 7
print(type(number))  # <class 'int'>
```

原来的 `value` 是 Tensor，转换后的 `number` 是普通 Python 整数。`.item()` 要求 Tensor 只有一个元素，不能对多个元素的张量要求“取出一个标量”。`.tolist()` 则把内容转成 Python 标量或嵌套列表。

CPU 上这个例子容易执行；数据在 NPU 上时，主机为了拿到结果，可能要等待设备。这与 C 中读取设备输出缓冲区之前必须等计算和传输完成的约束相似。

设备执行通常是异步的，但 Python 的 `int`、`float`、`bool` 在主机上使用。为了把设备结果变成这些值，运行时可能必须等待设备完成。

下面两种教学示意都要注意同步：

```python
value = device_tensor.item()
if device_tensor.max() > threshold:
    handle_result()
```

第二种虽然没有显式 `.item()`，但 Python `if` 仍然需要把单元素 Tensor 转成主机真假值。多元素 Tensor 直接作为条件通常会因真假值不明确而报错。

还要纠正两个容易形成的误解：

- `[tensor.item() for tensor in tensors]` 仍是逐个执行 `.item()`，不会仅因为用了列表推导式就自动合并成一次同步。
- `torch.max(x)` 可以把结果保留为设备 Tensor，但随后用 Python `if` 消费该结果，仍可能产生同步。

若要保持数据依赖在设备上，可考虑适用的张量操作，如 `torch.where`；但它不是通用的 Python 控制流替代品，传入的分支表达式通常已经求值，也要核对算子及图模式支持。CPU 传输、`.tolist()` 和用于日志的设备数据读取也应检查。

性能分析时区分：Python 对象是否复制、Tensor 数据是否复制、主机是否等待设备。这是三件不同的事。

### 12.5 `nn.Module`、权重和 `forward` 是什么

阅读算子实现时还会看到 `torch.nn.Module`、`torch.nn.Parameter`、`forward`。先看 CPU 教学例子：

```python
import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, x):
        return x * self.weight


layer = Scale()
print(layer(torch.tensor(3.0)).item())  # 6.0
```

`nn.Module` 提供层的组织、参数管理等机制；`Parameter` 用来把 Tensor 标记为模块参数；这里 `requires_grad=False` 表示不为这个参数要求自动求导跟踪。

`layer(x)` 会通过模块调用机制进入 `forward`，并涉及 hooks 等逻辑，所以一般使用模块调用语法，而不是直接调用 `layer.forward(x)`。项目的 `forward_oot` 属于对应框架自定义算子接口，不能仅凭名字把它当作任意 `nn.Module` 都会自动调用的约定。

如果看到继承 `torch.autograd.Function` 的类，它是在定义自定义自动求导操作，里面的 `ctx` 保存前后向所需信息；它不是前面普通层实例的 `self`。第一次阅读可以先跟踪前向的输入输出，再按需要学习 backward 协议。

## 13. 并发知识按需补充

### 13.1 先区分等待、并发和并行

假设一个请求要等网络，另一个请求要等设备：等待期间处理其他任务是并发；多个计算单元同一时刻执行才是并行。Python 协程、线程、进程和设备流处于不同层面。

先区分以下概念，再去读调度和通信实现：

| 机制 | 类比和注意事项 |
| --- | --- |
| Python 线程 | 共享进程内存；常规启用 GIL 的 CPython 对纯 Python CPU 循环的并行有限，但扩展可释放 GIL |
| 多进程 | 地址空间分离；父进程改全局字典，不会自动同步所有子进程 |
| `async def` / `await` | 协程在等待点协作切换，不等于新建线程或自动并行计算 |
| NPU stream / event | 设备任务排序和依赖机制；不能只靠 Python 函数的返回顺序判断完成时刻 |

锁、进程通信、设备事件分别解决不同层面的问题。需要根据项目实际运行环境判断行为，不要把 GIL 当作设备同步机制，也不要把 `async` 当作加速任意 CPU 代码的开关。

### 13.2 `async def` 和 `await` 的最小例子

```python
import asyncio


async def job(name):
    print(name, "开始")
    await asyncio.sleep(0.01)
    print(name, "结束")


async def main():
    await asyncio.gather(job("A"), job("B"))


if __name__ == "__main__":
    asyncio.run(main())
```

这是可单独运行的脚本示例。`async def` 定义协程函数；调用 `job("A")` 得到协程对象，不是同步地把整个函数执行完；`gather` 组织多个协程，`await` 等待可等待对象，并允许事件循环在适当时机运行其他任务。

这里通常先看到两个“开始”，再看到各自“结束”。用 `asyncio.sleep` 模拟等待是为了展示调度；它不等于设备 event，也不应该把真实阻塞 I/O 或长 CPU 循环直接塞进去就期望自动并行。

### 13.3 共享状态究竟共享到哪里

线程通常能访问同进程对象，所以修改字典可能影响其他线程；多进程拥有不同地址空间，必须用明确的通信或共享机制传递状态。设备流则安排设备任务，Python 返回后任务可能还在进行。

类比 C：线程锁解决主机共享数据竞争，IPC 解决进程通信，设备 event 解决设备执行依赖。它们不能相互替代。阅读项目中的异步路径时，给每个变量标出“哪个进程、哪个线程、主机还是设备”，比只看函数名是否包含 async 更可靠。

## 14. 推荐的源码阅读路线

每读一个新函数，都可以用以下模板写一段自己的解释：

```text
1. 这是普通函数、实例方法、类方法，还是生成器？
2. 调用时传什么？哪些参数有默认值？
3. 返回的是数据、类、函数、迭代器，还是 None？
4. 它改的是局部名字、传入对象，还是实例属性？
5. 哪些地方可能失败，谁负责处理或清理？
6. 如果涉及 Tensor，数据在哪里，是否复制或同步？
```

例如注册表查询可以读成：“普通函数，传两个字符串，用它们构成元组键查字典，找到返回类对象，找不到返回 None，不负责创建实例。”先能说出这句话，再深入类型系统细节。

| 顺序 | 打开的文件 | 本轮只关注什么 | 读完应能解释 |
| --- | --- | --- | --- |
| 1 | [registry.py](../../vllm_ascend/quantization/methods/registry.py) | 字典、元组、闭包、类装饰器 | 类如何被注册，重复注册如何报错 |
| 2 | [test_registry.py](../../tests/ut/quantization/methods/test_registry.py) | 身份断言和异常测试 | 为什么用 `assertIs`，缺失项如何处理 |
| 3 | [envs.py](../../vllm_ascend/envs.py) | `lambda`、模块 `__getattr__` | 看似读取属性，实际何时读取环境 |
| 4 | [layernorm.py](../../vllm_ascend/ops/layernorm.py) | 继承、可选值、解包、原地操作 | 有无 residual 时返回值为何不同 |
| 5 | [test_layernorm.py](../../tests/ut/ops/test_layernorm.py) | mock 的替换位置和返回值 | 哪些行为是桩函数模拟的 |
| 6 | [base.py](../../vllm_ascend/quantization/methods/base.py) | 抽象类、Mixin、类型导入 | 一个量化实现必须提供哪些接口 |
| 7 | [ascend_config.py](../../vllm_ascend/ascend_config.py) | 默认工厂、类方法、验证 | 嵌套配置和旧配置谁优先 |
| 8 | [v2/model_runner.py](../../vllm_ascend/worker/v2/model_runner.py) | `super()`、上下文、参数转发 | 设备相关行为在哪里接入，状态何时恢复 |

不建议第一步就逐行阅读整个 runner。先选一个小函数，把输入、输出、被修改的对象、调用链和依赖类型标出来，再逐步扩大范围。

阅读时可使用这些搜索命令：

```bash
rg -n "def register_scheme|def get_scheme_class" vllm_ascend/quantization/methods
rg -n "@register_scheme" vllm_ascend/quantization
rg -n "def __getattr__|env_variables" vllm_ascend/envs.py
rg -n "class AscendRMSNorm|def forward_oot" vllm_ascend/ops/layernorm.py
rg -n "@contextmanager|super\(\)" vllm_ascend/worker/v2/model_runner.py
```

## 15. 四个练习：从能看懂到能修改

### 练习一：预测别名行为

不运行代码，先写出 `a`、`b`、`c` 最终的值，再用 Python 验证：

```python
a = [[1], [2]]
b = a
c = a.copy()
b[0].append(3)
c.append([4])
```

参考答案：`a == [[1, 3], [2]]`，`b is a`，`c == [[1, 3], [2], [4]]`。外层列表不完全相同，前两个内层列表仍共享。

### 练习二：用局部注册表重写一个小插件机制

自己实现 `make_registry()`，返回 `register` 和 `lookup` 两个函数：注册表保留在闭包中，不增加模块级可变状态。

要求：以 `(quant_type, layer_type)` 为键，保存类对象；重复注册抛出 `ValueError`；查不到返回 `None`。分别为成功、重复和缺失写断言。这样可以一次练习字典、闭包、装饰器和异常。

### 练习三：模拟上下文里的异常恢复

用 `@contextmanager` 临时修改一个普通对象的字段，并在 `with` 块里主动抛异常。捕获后验证原字段已经恢复。关键不是“没有异常时正常”，而是所有退出路径都恢复。

### 练习四：对照一个真实算子的返回协议

阅读 `AscendRMSNorm.forward_oot`，分别写出有 residual 和没有 residual 的输入、返回值和调用算子。再读对应 mock 测试，标出哪些逻辑能在桩函数下验证，哪些需要实际设备。

完成标准：能够解释别名、返回值结构、分支选择和设备边界，而不仅是复述每一行代码。

### 练习二参考实现：没有模块级可变字典的注册表

下面可以独立运行，不导入项目包，也不需要 NPU：

```python
def make_registry():
    registry = {}

    def register(quant_type, layer_type):
        def decorator(cls):
            key = (quant_type, layer_type)
            if key in registry:
                raise ValueError("重复注册")
            registry[key] = cls
            return cls
        return decorator

    def lookup(quant_type, layer_type):
        return registry.get((quant_type, layer_type))

    return register, lookup


register, lookup = make_registry()


@register("TEACHING", "linear")
class TeachingScheme:
    pass


assert lookup("TEACHING", "linear") is TeachingScheme
assert lookup("MISSING", "linear") is None

try:
    register("TEACHING", "linear")(TeachingScheme)
except ValueError:
    print("重复注册被正确拒绝")
else:
    raise AssertionError("应该拒绝重复注册")
```

这里有三层作用域：最外层保存注册表，中间层保存键的组成部分，内层拿到类并完成注册。最后 `return register, lookup` 返回两个函数，通过解包分别交给调用方。

无需用 `global`，因为函数修改的是所引用字典的内容，没有重新绑定外层的 `registry` 名字。若要在内层重新绑定外层局部名字，则涉及 `nonlocal`；不要为了这个例子引入不必要的共享状态。

## 16. 写代码时使用的简短检查表

- 名字重新绑定和对象原地修改是否分清？是否意外共享列表、字典或 Tensor 存储？
- 类型注解、实际验证和运行时依赖是否分清？`None`、`0`、空容器有没有被混用？
- 函数返回的是实例、类、函数还是多个结果？调用方是否按相同协议处理？
- 装饰器和导入的副作用何时发生？patch/mock 是否替换了实际使用的位置？
- 临时状态和外部资源是否在异常路径中恢复？
- Tensor 的 shape、dtype、device、stride 是否符合算子要求？是否引入复制或设备同步？
- 新功能是否补齐成功、失败、边界测试？设备相关改动是否完成实际环境验证？
- 是否遵守项目的集中环境变量、顶部导入、避免新增可变全局状态和 patch 审查要求？

工具链方面，了解虚拟环境、`python -m pip`、`python -m pytest`、Ruff 和 pre-commit 即可逐步开始。`python -m ...` 有助于确保使用所选解释器对应的工具。项目完整依赖和 NPU 运行通常应在已配置的 Ascend 开发环境中处理，不必为了运行前面的纯 Python 练习先构建整套推理服务。

### 16.1 这些工具分别做什么

| 工具或命令 | 负责什么 | C 开发经验类比 |
| --- | --- | --- |
| `python example.py` | 用解释器执行脚本 | 启动程序，但不是先手动编译成可执行文件 |
| `python -m pip` | 用该解释器对应的 pip 管理依赖 | 获取和安装依赖库 |
| 虚拟环境 | 为项目隔离解释器使用的包集合 | 独立依赖前缀，不是完整虚拟机 |
| `python -m pytest` | 使用该解释器运行测试框架 | 单元测试运行器 |
| Ruff | 检查常见代码问题和格式 | lint 与格式化工具，不是 NPU 编译器 |
| pre-commit | 按仓库配置组织多个检查 | 统一检查脚本和 Git hooks |
| traceback | 显示异常经过的调用位置 | 带源代码位置的调用栈 |

`python -m 模块名` 的意思是把那个模块当作程序入口运行；不是向模块传一个叫 `m` 的业务参数。安装依赖前先确认解释器与环境，避免把工具装进另一套 Python。

### 16.2 遇到错误先看哪一行

如果看到：

```text
Traceback (most recent call last):
  ...
KeyError: 'height'
```

最后一行先告诉你异常类型和相关信息，往上看是哪些调用位置经过这里。常见对应关系：

| 错误 | 初步检查 |
| --- | --- |
| `NameError` | 名字是否定义，是否在当前作用域 |
| `AttributeError` | 对象是否有该属性，初始化或 patch 是否完成 |
| `KeyError` | 字典是否包含该键，是否应使用 `.get` |
| `TypeError` | 参数数量、关键字名、对象类型或操作是否不匹配 |
| `ValueError` | 输入类型可处理，但内容、范围或形状是否不合法 |
| `ModuleNotFoundError` | 依赖、解释器、导入路径是否正确 |
| `AssertionError` | 哪项预期没有满足，是否属于测试断言失败 |

异常类型是线索，不是唯一结论；比如 `TypeError` 也可能由代码主动抛出。先定位当前代码真正失败的操作，再判断是语法理解、数据协议、依赖还是硬件环境问题。

仓库要求推送前运行 `bash format.sh ci`，Markdown 也在检查范围内。这里只生成学习文档，不代表已完成模型运行、完整测试套件或硬件验证。
