---
name: torch-ascendc-op-extension
description: 将已有 Ascend C 三尖括号直调工程通过 TORCH_LIBRARY 对接到 PyTorch，实现 torch.ops.npu.xxx() 调用。触发：用户提到 TORCH_LIBRARY、.asc 对接 PyTorch、Python 调用 Ascend C 算子、注册到 torch、算子接入 PyTorch dispatch、PyTorch binding、torch extension、或想在 Python 中用 torch.ops.xxx() 调用已有 Ascend C kernel。不适用：从零建工程（用 ascendc-direct-invoke-template）；注册调用转直调（用 ascendc-registry-invoke-to-direct-invoke）。
---

# Ascend C 算子 TORCH_LIBRARY 对接指南

## 前置条件

- 已有可编译运行的 Ascend C `<<<>>>` 直调工程
- 环境已安装：torch、torch_npu
- CANN 环境已配置（`ASCEND_HOME_PATH` 已设置）

## 目标产出

无论源工程是什么目录结构，改造后都需要以下文件：

```
├── op_kernel/
│   ├── xxx_tiling.h         (Tiling 结构体 + 常量，kernel 和 host 共用)
│   └── xxx_kernel.asc       (纯 kernel 代码，引用 xxx_tiling.h)
├── op_host/
│   ├── xxx.asc              (host+main 可执行文件入口，引用 xxx_kernel.asc)
│   └── data_utils.h         (文件读写工具，位置以 include 路径可达为准)
├── op_extension/
│   ├── xxx_torch.cpp        (PyTorch 接入层：torch::Tensor 接口 + Tiling 计算 + kernel launch)
│   ├── register.cpp         (TORCH_LIBRARY 注册，含 Meta backend)
│   └── ops.h                (函数声明)
└── CMakeLists.txt           (双 target：可执行文件 + libxxx_ops.so)
```

**已存在的文件保留不动，缺失的按 Step 逐个补齐。**

## 架构原理

```
文件依赖:
  op_kernel/xxx_tiling.h              (单一数据源：Tiling 结构体 + 常量)
    ├── op_kernel/xxx_kernel.asc      (#include "xxx_tiling.h")
    │   └── op_host/xxx.asc           (#include "../op_kernel/xxx_kernel.asc"，间接引入)
    └── op_extension/xxx_torch.cpp (#include "../op_kernel/xxx_tiling.h")

Python 调用链:
  torch.ops.npu.xxx(x1, x2)
      │
      ▼ PyTorch dispatch（PrivateUse1 后端）
  ascend_kernel::xxx_torch()         ← op_extension/xxx_torch.cpp 实现
      │
      ├─ ComputeTiling()             ← 计算 Tiling 参数
      ├─ aclrtMemcpy(tiling→device)  ← Tiling 搬到 device
      └─ xxx_kernel(...)             ← 函数调用（等价于 <<<>>>）
            │
            ▼ NPU 执行
```

## Stream 同步模式

> **核心背景**：NPU 的 taskqueue 是设备端任务队列。"清 queue"指等待队列中已有任务完成后再执行当前任务；"入 queue"指将当前任务放入队列中按顺序执行。

本方案采用 CMake 多文件架构，`xxx_torch.cpp` 由 C++ 编译器编译（非 ASC），因此**不能使用 `<<<>>>` 语法**，只能将 kernel 作为普通 C 函数调用。在这种架构下，必须手动管理 stream 同步。

### 推荐模式：`stream(true)` 清 queue

```cpp
auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
xxx_kernel(blockNum, nullptr, aclStream, ...);
```

`stream(true)` 在返回 aclrtStream 前会清 queue（等待之前任务完成），确保不会乱序。输出分配用 `at::empty_like`（`zeros_like` 会触发 NPU 操作，可能乱序）。

### 禁止模式

| 反模式 | 风险 | 修复 |
|--------|------|------|
| `stream(false)` + 直接调用 | **乱序**：不清 queue，kernel 先于之前操作执行 | 改用 `stream(true)` |
| lambda 内传 NPUStream + OpCommand | **死锁**：queue 等 lambda，lambda 等 queue 空 | lambda 内改用 `aclrtStream` |
| `zeros_like` 创建输出 | **乱序**：zeros_like 入 queue 但 kernel 不入 queue | 改用 `empty_like` |

### 8 种启动方式速查

| # | 方式 | 判定 |
|---|------|------|
| 1 | 传 NPUStream 对象 + `<<<>>>` | **推荐**（单文件 .asc） |
| 2 | `stream(true)` + `<<<>>>` / 函数调用 | **正确**（**本方案唯一适用**） |
| 3 | `stream(false)` + OpCommand + `<<<>>>` | **推荐**（单文件 .asc） |
| 4 | `stream(false)` + 直接启动 | **反模式：乱序** |
| 5 | NPUStream + OpCommand | **反模式：死锁** |
| 6 | `stream(true)` + OpCommand | **正确**（单文件 .asc） |
| 7 | `stream(true)` + `zeros_like` | **反模式：乱序** |
| 8 | `stream()` 无参数 | **正确**（等价方式2） |

每种模式的完整代码、原理和死锁流程见 [references/anti_patterns.md](references/anti_patterns.md)。

## 改造步骤

### Step 0: 拆分 kernel 与 host（仅单文件 .asc 工程需要）

**跳过条件**：kernel 和 host 已在不同文件中。

对单文件 `.asc` 工程（kernel + host + main 在一个文件里），执行以下操作：

1. **提取 `op_kernel/xxx_tiling.h`**：从原 `.asc` 中提取以下内容到新文件 `op_kernel/xxx_tiling.h`：
   - 命名常量（`constexpr`，如 `TILE_LENGTH`、`DOUBLE_BUFFER`）
   - Tiling 数据结构体（`struct XxxTilingData`）
   
   `xxx_tiling.h` 只含纯 C/C++ 语法，不含 `__aicore__`、`__gm__` 等 ASC 关键字。所有编译单元（kernel、host main、torch host）共用此文件。

2. **提取 `op_kernel/xxx_kernel.asc`**：从原 `.asc` 中剪切以下内容到新文件 `op_kernel/xxx_kernel.asc`：
   - `#include "kernel_operator.h"` 及 ASC 专用头文件
   - `#include "xxx_tiling.h"`（替代原有的内联常量和结构体）
   - Kernel 类（`class KernelXxx`）
   - 核函数入口（`extern "C" __global__ __vector__ void xxx_kernel(...)`）
   
   `xxx_kernel.asc` 只含 ASC 语法，不含任何 host 代码（main、文件读写、iostream 等）。

3. **改造原 `.asc`**（移至 `op_host/` 目录）：删除已剪切的 kernel 内容，改为：
   ```cpp
   #include "../op_kernel/xxx_kernel.asc"    // 引用拆出的纯 kernel 文件
   // ... 以下保留原有的 host 侧代码（main 等），不做任何修改
   ```

4. **确保 include 路径正确**：`op_host/xxx.asc` 依赖的辅助文件（如 `data_utils.h`）需在 CMakeLists.txt 的 `target_include_directories` 覆盖的路径内可被找到。

5. **清理旧文件**：删除已被替代的原始 `.asc` 文件（拆分后不再被编译引用），避免留下孤立文件混淆维护。

6. **验证**：编译可执行文件目标，确认拆分未破坏功能。

### Step 1: 创建 `op_extension/ops.h`（函数声明）

**已存在则跳过。**

读取模板：[templates/ops_template.h](templates/ops_template.h)

在 `ascend_kernel` 命名空间中声明算子函数签名，供 `op_extension/register.cpp` 和 `op_extension/xxx_torch.cpp` 共用。

### Step 2: 创建 `op_extension/xxx_torch.cpp`（PyTorch 接入层实现）

这是核心文件。读取源工程代码后，基于模板改造：[templates/torch_template.cpp](templates/torch_template.cpp)

模板使用 `#include "../op_kernel/xxx_tiling.h"` 引入 Tiling 结构体和常量（单一数据源）。`ComputeTiling` 函数是**示例占位**，应替换为源工程的实际计算逻辑。

**关键设计决策**：
- `#include "acl/acl.h"` 放在 torch 头文件**之前**——torch 的宏会干扰 ACL 头文件解析
- `#include "../op_kernel/xxx_tiling.h"` 引入 Tiling 结构体和常量
- 参数校验用 `x1.is_privateuseone()`（C++ API），不是 `is_npu()`
- 框架 stream：**必须用 `stream(true)`**（清 queue，防乱序），**禁止 `stream(false)`**。详见上方「Stream 同步模式」章节

**适配指引**：Tiling 计算、kernel 入口声明、kernel 调用方式等需从源码提取。详见 [references/operation_checklist.md](references/operation_checklist.md)。

### Step 3: 创建 `op_extension/register.cpp`（算子注册）

**已存在则跳过。**

读取模板：[templates/register_template.cpp](templates/register_template.cpp)

必须包含三部分注册：

```cpp
// 1. 算子签名定义
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("xxx(Tensor x1, Tensor x2) -> Tensor");
}

// 2. NPU 后端实现绑定
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("xxx", TORCH_FN(ascend_kernel::xxx_torch));
}

// 3. Meta 后端注册（torch.compile / fx 必需）
at::Tensor xxx_meta(const at::Tensor& x1, const at::Tensor& x2) {
    return at::empty_like(x1);  // 简单算子：直接推导；多输出/shape变化需按实际逻辑
}
TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("xxx", &xxx_meta);
}
```

**namespace 选择**：默认用 `npu`（`torch.ops.npu.xxx`）。如需避免与 torch_npu 内建算子冲突，可改用 `ascendc_ops`（`torch.ops.ascendc_ops.xxx`）。

### Step 4: 更新 CMakeLists.txt

读取模板：[templates/CMakeLists_template.cmake](templates/CMakeLists_template.cmake)

双目标结构：保留原有可执行文件 Target，新增 `add_library(xxx_ops SHARED ...)` 编译 `.so`。`set(CMAKE_CXX_STANDARD 17)` 不可省略。`npu-arch` 按实际芯片修改：A2/A3=dav-2201, A5=dav-3510。

kernel 文件形态不同时的 CMakeLists 处理方式见 [references/operation_checklist.md](references/operation_checklist.md)。

**注意**：如果 CMakeLists.txt 中已有 PyTorch 相关的 Target，直接替换为新的 Target 2 配置，不要保留旧目标。

### Step 5: 编译验证

```bash
cmake -S . -B build && cmake --build build -j4
```

编译成功后会产出：
- `build/<operator_name>_custom` — 原有可执行文件（直调验证）
- `build/libxxx_ops.so` — PyTorch 扩展库

### Step 6: Python 测试

```python
import torch
import torch_npu

torch.ops.load_library("build/libxxx_ops.so")

# 根据算子签名调整输入数量、dtype 和验证逻辑
x1 = torch.randn(1024, 4096, dtype=torch.float32).npu()
x2 = torch.randn(1024, 4096, dtype=torch.float32).npu()

y = torch.ops.npu.xxx(x1, x2)    # 单输入算子改为 torch.ops.npu.xxx(x1)

assert y.is_npu                    # 注意：Python 中是 property 不是方法
assert torch.allclose(y.cpu(), x1.cpu() + x2.cpu())  # 验证逻辑按算子语义调整
```

多算子场景加载多个 `.so` 即可：

```python
torch.ops.load_library("/path/to/libadd1_ops.so")
torch.ops.load_library("/path/to/libmul1_ops.so")

y1 = torch.ops.npu.add1(x1, x2)
y2 = torch.ops.npu.mul1(x1, x2)
```

## 多算子扩展

一个 `.so` 可注册多个算子：

```
register.cpp:
  TORCH_LIBRARY_FRAGMENT(npu, m) {
      m.def("op_a(Tensor x1, Tensor x2) -> Tensor");
      m.def("op_b(Tensor x1, Tensor x2) -> Tensor");
  }
  TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
      m.impl("op_a", TORCH_FN(ascend_kernel::op_a));
      m.impl("op_b", TORCH_FN(ascend_kernel::op_b));
  }
  TORCH_LIBRARY_IMPL(npu, Meta, m) {
      m.impl("op_a", &op_a_meta);
      m.impl("op_b", &op_b_meta);
  }
```

## aclgraph 兼容

已注册 TORCH_LIBRARY 的自定义算子天然支持 aclgraph，无需额外改造。使用方式：

```python
# 方式1: NPUGraph
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    output = model(static_x, static_y)
g.replay()

# 方式2: make_graphed_callables
model = torch.npu.make_graphed_callables(model, (x, y))

# 方式3: npugraph_ex backend (需 torch_npu >= 7.3.0)
compiled = torch.compile(model, backend="npugraph_ex", fullgraph=True)
```

**前提**：算子已正确注册 Meta backend（见 Step 3），且 stream 同步使用 `stream(true)` 模式。

## 踩坑清单

遇到编译/链接/运行时问题时，读取 [references/troubleshooting.md](references/troubleshooting.md)。

## 参考资源

- 官方 aclgraph + TORCH_LIBRARY 示例：`torchair/examples/_kernel_extension_aclgraph/torch_library/`
- 官方 TORCH_LIBRARY 示例：`$ASC_DEVKIT_DIR/examples/02_features/01_triple_chevron_notation/torch_library/`
- PyTorch 框架适配文档：`$ASC_DEVKIT_DIR/docs/guide/编程指南/附录/AI框架算子适配/PyTorch框架.md`
- PyTorch Custom Operators：https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
