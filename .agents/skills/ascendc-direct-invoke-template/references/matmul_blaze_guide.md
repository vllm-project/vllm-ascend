# Blaze Matmul 工程模板指南

> **定位**：提供基于 Blaze 库开发 matmul 类算子的参考骨架示例，明确目录结构、编码风格与规范。
>
> **适用架构**：DAV_3510
>
> **工程模板位置**：`ascendc-direct-invoke-template` skill 的 `references/matmul_blaze_template/`

---

## §1 Blaze 编程概述

### 1.1 Blaze 是什么

Blaze 是 Ascend C 的高层 matmul 编程库，封装了底层 tensor_api 和硬件指令，提供可组合的组件（BlockMmad、BlockScheduler、Kernel），使开发者能够以声明式方式构建高性能矩阵乘法算子。

### 1.2 四层抽象

```
Launcher（host 端）
   ↓ 选择组件 + 传递参数
Kernel（device 端入口）
   ↓ 组装类型链
BlockMmad + BlockScheduler（核心组件）
   ↓ 调用
tensor_api（硬件指令封装）
```

- **Launcher**：host 端 C++ 程序，负责 tiling 计算、内存管理、kernel 启动
- **Kernel**：device 端入口函数（`__global__ __aicore__ __cube__`），组装类型链，驱动计算
- **BlockMmad**：数据搬运 + MMAD 计算的核心组件，管理 L1/L0 流水
- **BlockScheduler**：tile 分配到多核的调度器（Serpentine 遍历）
- **DispatchPolicy**：流水策略（NO_FULL_LOAD / A_FULL_LOAD）
- **tensor_api**：硬件指令封装（Mmad、CopyGM2L1 等），一般不直接修改

### 1.3 关键组件职责

| 组件 | 职责 | 来源 |
|------|------|------|
| Launcher | host 端程序，负责 tiling、内存管理、kernel 启动 | 用户开发 |
| Kernel | device 端入口函数，组装类型链，驱动计算 | 用户开发 |
| BlockMmad | 数据搬运 + MMAD 计算（L1/L0 流水） | Blaze 库或 blaze_custom |
| BlockScheduler | tile 分配到多核（Serpentine 遍历） | Blaze 库或 blaze_custom |
| DispatchPolicy | 流水策略（NO_FULL_LOAD / A_FULL_LOAD） | Blaze 库或 blaze_custom |
| ProblemShape | 问题规模定义（m, n, k, batch） | tensor_api |

---

## §2 工程目录结构

### 2.1 整体目录树

```
matmul_blaze_template/
├── test_matmul_blaze.cpp          # C++ 直调 launcher（host 端入口）
├── run.sh                         # 一键编译+运行+验证
├── CMakeLists.txt                 # 构建配置
├── op_kernel/                     # Kernel 层（device 端）
│   ├── matmul_blaze_kernel.h      # Kernel 模板函数
│   ├── matmul_blaze_kernel.cpp    # Wrapper（extern "C" 包装）
│   └── include/
│       ├── blaze/                 # [拉取] Blaze 库
│       ├── tensor_api/            # [拉取] tensor_api
│       └── blaze_custom/          # [可选] 自定义 Blaze 扩展模块
├── op_tiling/                     # Tiling 层（host 端）
│   ├── matmul_tiling_data.h       # TilingData POD 结构体
│   ├── matmul_tiling_stub.h       # Mock tiling 实现（仅用于示例）
│   └── matmul_constant.h          # Tiling 相关常量
├── op_extension/                  # [可选] PyTorch 接入层
│   ├── matmul_blaze_torch.cpp     # Torch extension 实现
│   ├── register.cpp               # TORCH_LIBRARY 注册
│   └── ops.h                      # 函数声明
├── common/                        # 公共工具
│   ├── acl_utils.h                # ACL 会话 RAII
│   ├── common_utils.h             # 参数解析、CeilDiv 等
│   └── io_utils.h                 # 文件 I/O
└── scripts/                       # 测试脚本
    ├── gen_data_mxfp8.py          # mxfp8 数据生成 + golden 计算
    ├── gen_data_mxfp4.py          # mxfp4 数据生成 + golden 计算
    └── verify_result.py           # 精度验证
```

### 2.2 关键目录说明

#### `op_kernel/include/` 依赖管理

`blaze/` 和 `tensor_api/` 是外部依赖，从 ops-tensor 仓库拉取：

```bash
git clone --depth 1 https://gitcode.com/cann/ops-tensor.git /tmp/ops-tensor
cp -r /tmp/ops-tensor/include/blaze op_kernel/include/
cp -r /tmp/ops-tensor/include/tensor_api op_kernel/include/
rm -rf /tmp/ops-tensor
```

`run.sh` 首次运行时会自动检测并拉取。

#### `blaze_custom/` 自定义扩展

本模板提供两条开发路径：

- **Blaze 库路径**（本模板示例）：所有组件来自 `blaze/gemm/`，适用于标准 matmul 场景
- **blaze_custom 路径**：自定义模块在 `blaze_custom/` 开发，适用于需要特殊功能的场景

两条路径的组件接口不同，不能混用。当前 `blaze_custom/` 为空目录（`.gitkeep` 占位）。

#### `op_tiling/` 边界说明

当前 `matmul_tiling_stub.h` 是 mock 实现，返回固定参数（baseM=128, baseN=128, baseK=128），仅用于示例演示。正式开发时必须根据 `BlockScheduler::Params` 结构体定义开发完整的 tiling 引擎，确保 `TilingData` 字段与 `Params` 字段映射正确。

#### `op_extension/` PyTorch 接入（可选）

提供 `torch.ops.npu.matmul_blaze()` 接口，使用 `TORCH_LIBRARY_FRAGMENT` 注册。通过 wrapper 函数（`matmul_blaze_kernel.cpp`）调用 kernel。

#### 本模板使用的组件

本模板使用 Blaze 库路径，组件来自 `op_kernel/include/blaze/gemm/`：

- **Kernel**：`GemmUniversal`（通用 kernel 模板）
- **BlockMmad**：`BlockMmad`（9 个模板参数，含 BiasType + LayoutBias）
- **BlockScheduler**：`BlockSchedulerQuantBatchMatmulV3`（量化 batch matmul 调度器）
- **DispatchPolicy**：`MatmulWithScaleMx`（MX 量化策略）

参考：`op_kernel/matmul_blaze_kernel.h` 第 31-42 行

### 2.3 文件依赖关系

```
test_matmul_blaze.cpp (launcher)
   ↓ includes
op_kernel/matmul_blaze_kernel.h
   ↓ includes
blaze/gemm/kernel/kernel_qbmm_mx.h
   ↓ includes
blaze/gemm/block/block_mmad_qbmm_mx.h
blaze/gemm/block/block_scheduler_qbmm.h
tensor_api/tensor.h
```

---

## §3 开发流程概览

### Step 1: 工程初始化

1. 复制模板目录，重命名为目标算子名
2. 修改 CMakeLists.txt：目标名、include 路径、编译选项（`--npu-arch=dav-3510`）
3. 拉取 blaze + tensor_api 依赖（`run.sh` 首次运行自动完成）
4. 确认 `op_kernel/include/blaze/` 和 `tensor_api/` 目录存在

参考：`CMakeLists.txt`

### Step 1.5: 根据 DESIGN.md 改造样例工程结构

`matmul_blaze_template/` 是 Blaze 样例工程，不是目标算子的最终交付结构。复制模板后，必须先根据 `DESIGN.md` 的“工程目录目标设计”完成目录和文件改造，再进入 Kernel / Tiling / 测试逻辑开发。

#### 改造原则

| 原则 | 要求 |
|------|------|
| 以 Blaze 样例结构为基础 | 目标工程继承根目录 launcher、`op_kernel/`、`op_tiling/`、`op_extension/`、`common/`、`scripts/` 等结构，不引入 Add/Vector 模板的 `op_host/` 目录 |
| 先重命名再开发 | 样例业务文件必须重命名为目标算子语义，或明确删除；不要在样例文件旁边叠加一套新文件后保留旧文件 |
| 内容必须改写 | 算子名、kernel 名、launch wrapper、PyTorch 注册名、dtype、shape、输入输出、转置语义、tiling、golden、verify 均需与 `DESIGN.md` 一致 |
| CMake 必须收敛 | `add_executable` / `add_library` 只纳入目标算子源文件；未纳入 target 的样例业务源文件不应继续留在工程中 |
| 外部依赖例外 | `op_kernel/include/blaze/` 和 `op_kernel/include/tensor_api/` 是外部依赖，可保留其多 dtype / 多场景代码 |
| 自定义扩展位置 | 需要扩展 Blaze 时，只在 `op_kernel/include/blaze_custom/` 下新增或改造模块 |

#### 常见样例文件改造映射

| 样例文件/目录 | 常见处理 | 说明 |
|---------------|----------|------|
| `test_matmul_blaze.cpp` | 重命名并改写为 `{operator_name}.cpp`，或按设计删除 | 根目录直调 launcher |
| `test_matmul_blaze_torch.py` | 重命名并改写为 `test_{operator_name}_torch.py`，或按设计删除 | Blaze 模板的 PyTorch 测试脚本在根目录 |
| `op_kernel/matmul_blaze_kernel.h` | 重命名并改写为 `op_kernel/{operator_name}_kernel.h`，或按设计删除 | Kernel 类型链与入口声明 |
| `op_kernel/matmul_blaze_kernel.cpp` | 重命名并改写为 `op_kernel/{operator_name}_kernel.cpp`，或按设计删除 | Kernel wrapper / 分发入口 |
| `op_extension/matmul_blaze_torch.cpp` | 重命名并改写为 `op_extension/{operator_name}_torch.cpp`，或按设计删除 | PyTorch extension host |
| `op_extension/register.cpp` / `ops.h` | 改写注册名和函数声明，或按设计删除 | TORCH_LIBRARY 接入 |
| `op_tiling/matmul_tiling_data.h` | 重命名并改写为 `{operator_name}` tiling data | 字段必须匹配目标 Kernel Params |
| `op_tiling/matmul_tiling_stub.h` | 改写为正式 tiling，或删除 | mock tiling 不能作为正式实现 |
| `scripts/gen_data_mxfp8.py` / `gen_data_mxfp4.py` | 非 MX/FP4 算子删除或改写为目标 `gen_data.py` | 不得保留与目标 dtype 无关的业务脚本 |
| `scripts/verify_result.py` | 改写后保留 | 精度标准、输入输出文件、Golden 路径需与目标算子一致 |

#### C+V 融合目录要求

普通 MatMul + Vector 融合使用 `blaze_custom` 路线。自定义 Epilogue 是 Blaze Custom 组件，必须放在：

```text
op_kernel/include/blaze_custom/epilogue/{epilogue_name}.h
```

Kernel 中通过以下方式引用：

```cpp
#include "blaze_custom/epilogue/{epilogue_name}.h"
```

不要把融合 Epilogue 放在 `op_kernel/` 顶层。`op_kernel/` 顶层只放目标 Kernel 入口、Wrapper 或按设计需要保留的 kernel 相关文件。

#### 目标目录结构由 DESIGN.md 决定

不同 Blaze 场景的目录结构不同，Developer 必须按 `DESIGN.md` 中基于组件选择、Epilogue 设计和 Tiling 设计推导出的目标目录执行：

- 纯 MatMul：可只保留 blaze library 路径和目标 kernel/tiling/launcher。
- 普通 C+V：需要 `blaze_custom/kernel`、`block`、`policy`、`utils`、`epilogue` 中与设计相关的模块。
- MX C+V：按受控组合路径保留 MX 相关 scale / Quant tiling 文件，不得套普通 C+V 路径重写 MX 反量化。
- GroupMatMul：按 grouped scheduler / context-based epilogue 的设计保留对应模块。

### Step 2: 定义 Kernel

**Kernel 签名设计**：
- 修饰符选择：纯 matmul 用 `__cube__`，融合用 `__mix__(aicCount, aivCount)`
- `GM_ADDR` 参数顺序：输入在前、输出在后、tilingData 最后
- 模板参数：`conditional_t` 推导模式（推荐）或 `bool + enum` 模式

**类型链组装**：

从 `ProblemShape` 出发，逐步推导完整类型链：

```
ProblemShape → DispatchPolicy → BlockScheduler → BlockMmad → Kernel → Params
```

- Blaze 库路径：所有组件来自 `blaze/gemm/`，不能混用 blaze_custom
- 每个组件暴露 `::Params` 类型，从中提取字段填充参数结构体
- 不要硬编码 Params 结构，始终从组件类型推导

**TilingData 对齐**：
- `TilingData` 字段必须与 `BlockScheduler::Params` 对齐
- 当前模板使用 mock tiling，正式开发时需替换

参考：`op_kernel/matmul_blaze_kernel.h`

### Step 3: 编写 Launcher

**Host boilerplate**：
- ACL 会话初始化（`AclRtSession` RAII）
- 内存管理（`aclrtMallocHost` + `std::unique_ptr`）
- 文件 I/O（`ReadExactFile` / `WriteFile`）

**Layout Dispatch**：
- `conditional_t` + 扁平 if/else 分发
- 每种 transA/transB 组合一行

**Kernel Launch**：
- `<<<gridDim, nullptr, stream>>>` 语法
- `gridDim` 取 `tilingData.usedCoreNum`

参考：`test_matmul_blaze.cpp`

### 改造点

| 改造点 | 文件 | 说明 |
|--------|------|------|
| 算子名称 | `test_matmul_blaze.cpp`、`CMakeLists.txt`、`run.sh` | 三处保持一致 |
| 数据类型 | `matmul_blaze_kernel.h` | 修改 `AType/BType/CType`（第 19-24 行） |
| Tiling 引擎 | `op_tiling/` | 替换 mock 实现为正式 tiling |
| 数据生成 | `scripts/gen_data_*.py` | 根据 dtype 调整 |
| 精度验证 | `scripts/verify_result.py` | 根据 dtype 调整容差 |

### 常见变种模式

- **A_FULL_LOAD**：A 矩阵常驻 L1，跨 N-tile 复用（修改 `DispatchPolicy` 模板参数）
- **CV 融合**：matmul + vector 后处理（GELU/SwiGLU 等），使用 `__mix__` 修饰符
- **MX 量化**：mxfp8/mxfp4 输入，bf16 输出（本模板示例）
- **Group matmul**：多个小 matmul 批量执行（使用 `GroupMatmulBlockScheduler`）
- **自定义扩展**：当标准 Blaze 组件无法满足需求时，在 `blaze_custom/` 开发自定义模块

---

## §4 编译、运行与验证

### 4.1 首次使用

```bash
bash run.sh 16 128 16384 false true
```

自动拉取 blaze + tensor_api → 编译 → 生成数据 → 运行 → 验证

### 4.2 常用命令

```bash
bash run.sh --skip-build 256 256 256  # 跳过编译
bash run.sh --clean                   # 清理临时文件（build/、blaze/、tensor_api/、input/、output/）
```

### 4.3 验证标准

| 数据类型 | rtol | atol |
|---------|------|------|
| bf16 / fp16 | 1e-3 | 1e-3 |
| fp32 | 1e-5 | 1e-5 |
| mxfp8 | 1e-2 | 1e-2 |
| mxfp4 | 5e-2 | 5e-2 |

### 4.4 验证流程

1. **编译**：无错误、无警告
2. **运行**：不崩溃（无 segfault、无 hang）
3. **精度**：`verify_result.py` 与 CPU golden 对比，输出 `[PASS]` 或 `[FAIL]`

---

## §5 常见误区

### 5.1 依赖管理

⚠️ **不要修改 `op_kernel/include/blaze/` 和 `tensor_api/` 的内容**

这些是外部依赖，修改后会在下次拉取时被覆盖。如需扩展 Blaze 功能，在 `blaze_custom/` 目录开发自定义模块。

### 5.2 Tiling 对齐

⚠️ **`TilingData` 字段必须与 `BlockScheduler::Params` 对齐**

`TilingData` 是 host 端填充的 POD 结构体，`Params` 是 device 端使用的结构体。两者的字段映射关系必须正确，否则会导致 kernel 行为异常。

⚠️ **Mock tiling 仅用于示例，正式开发时必须替换**

`matmul_tiling_stub.h` 返回固定参数，不考虑实际矩阵规模和硬件约束。正式开发时需根据 `BlockScheduler` 的要求开发完整的 tiling 引擎。

### 5.3 路径混用

⚠️ **Blaze 库路径和 blaze_custom 路径不能混用**

Blaze 库组件（BlockMmad、BlockScheduler、Kernel）的接口与 blaze_custom 不同。选择一条路径后，所有组件都必须来自同一路径。

### 5.4 Params 字段顺序

⚠️ **`Params` 聚合初始化时，字段顺序必须与声明完全一致**

错位会导致 `excess elements in scalar initializer` 编译错误。参考 `matmul_blaze_kernel.h` 第 84 行的 `params` 初始化。

---

## §6 模板文件索引

| 文件 | 用途 |
|------|------|
| `op_kernel/matmul_blaze_kernel.h` | Kernel 模板函数参考（类型链组装 + Params 填充） |
| `op_kernel/matmul_blaze_kernel.cpp` | Wrapper 函数参考（extern "C" 包装 + trans 分发） |
| `test_matmul_blaze.cpp` | Launcher 参考（ACL + 内存 + I/O + dispatch + launch） |
| `op_tiling/matmul_tiling_data.h` | TilingData POD 结构参考 |
| `CMakeLists.txt` | 构建配置参考 |
| `run.sh` | 编译运行验证流程参考 |
| `scripts/gen_data_mxfp8.py` | MXFP8 数据生成 + golden 计算参考 |
| `scripts/gen_data_mxfp4.py` | MXFP4 数据生成 + golden 计算参考 |
| `scripts/verify_result.py` | 精度验证参考 |
