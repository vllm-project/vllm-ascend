---
name: ascendc-direct-invoke-template
description: Kernel直调工程模板，用于创建 Ascend C Kernel 直调工程项目。提供经过验证的 Vector 样例工程（add_custom）、Blaze Matmul 工程模板（纯 Matmul / 融合 / MX 量化 / GroupMatmul）和 Kirin Vector 模板。触发：当用户需要创建 Kernel 直调工程、学习 Ascend C 编程、快速原型验证、或提及"Kernel直调"、"三尖括号内核调用"、"Blaze Matmul"、"matmul 模板"时使用本 skill。
---
# Ascend C Kernel 直调工程

先按算子类型路由到模板，再按 `[MODIFY]` 注释改造。


## 场景路由

| 算子类型 | 入口 | 典型算子 |
|---|---|---|
| **Vector** | `references/add_custom/` | add、mul、relu、softmax、layernorm 等逐元素/归约算子 |
| **Blaze Matmul**（A5/DAV_3510） | `references/matmul_blaze_template/` + `references/matmul_blaze_guide.md` | MatMul、BatchMatMul、matmul+bias、mxfp8 matmul、matmul+vector 融合、GroupMatmul |
| **Kirin Vector**（端侧开发） | `references/kirin_add_template/` + `references/kirin_vector_guide.md` | Kirin SoC（KirinX90/Kirin9030）专用逐元素/归约算子 |

## 使用方法

### A. Vector 算子（Add 分支）

1. **复制样例目录**：
   ```bash
   # 若算子目录 <your_op> 未创建
   cp -r references/add_custom <your_op>
   # 若算子目录 <your_op> 已存在
   cp -r references/add_custom/* <your_op>
   cd <your_op>
   ```
2. **全局替换算子名**：`add_custom` → `<your_op>`（add_custom 是整体算子名，_kernel/_torch/_tiling 是固定后缀）
3. **按 `[MODIFY]` 标记改造**：
   - 类名和 kernel 函数名
   - Tiling 结构体（`add_custom_tiling.h`）
   - 计算逻辑（`add_custom_kernel.asc`）
   - 输入/输出数量
   - `CMakeLists.txt` 中的目标名
4. **编译运行**：
   ```bash
   # 完整流程（含编译）
   bash run.sh
   # 仅运行测试，复用已有编译产物
   bash run.sh --skip-build
   ```
   > `run.sh` 在运行 kernel 前会自动删除旧的 `output/output.bin`，确保精度验证读取的是本次运行的新鲜输出。

### B. Blaze Matmul

见 `references/matmul_blaze_guide.md`（独立自洽）：按场景路由选择模板，复制 `matmul_blaze_template/` 工程后按指南改造。

### C. Kirin Vector 算子（Add 分支）

见 `references/kirin_vector_guide.md`（独立自洽）：复制 `references/kirin_add_template/` 工程后按 guide 进行修改。

## 文件结构（Add 分支）

```
├── op_kernel/               NPU 计算层
│   ├── add_custom_tiling.h      Tiling 常量 + 结构体（kernel 和 host 共用）
│   └── add_custom_kernel.asc    纯 kernel 代码（KernelAdd 类 + add_custom_kernel 核函数入口）
├── op_host/                 Host 直调层
│   ├── add_custom.asc           Host + main 入口（#include "add_custom_kernel.asc"）
│   └── data_utils.h             数据读写工具
├── op_extension/            PyTorch 接入层
│   ├── add_custom_torch.cpp     PyTorch host 实现（Tiling 计算 + kernel launch）
│   ├── register.cpp             TORCH_LIBRARY 注册（含 Meta backend）
│   └── ops.h                    函数声明
├── scripts/                 测试脚本
│   ├── gen_data.py               生成输入数据
│   ├── golden.py                 Golden 计算函数（直调 & PyTorch 双通路共用）
│   ├── verify_result.py          直调通路精度验证
│   └── test_torch.py             PyTorch 通路测试
├── CMakeLists.txt           双 target：可执行文件 + libadd_custom_ops.so
├── run.sh                   一键运行（支持 --torch 跑 PyTorch 通路）
└── README.md
```

## Add 代码关键模式（速查）

在 `add_custom_kernel.asc` 和 `add_custom.asc` 中可直接学习：

- **内存分配**: `TPipe` + `TQue` 管理 UB Buffer
- **数据流**: CopyIn → Compute → CopyOut 三段模式
- **同步**: `EnQue/DeQue` 确保操作顺序
- **Host 流程**: ACL 初始化 → KernelCall → 资源释放

## PyTorch 对接

模板已内置 PyTorch 对接，编译后即可使用：

```python
import torch; import torch_npu
torch.ops.load_library("build/libadd_custom_ops.so")
y = torch.ops.npu.add_custom(x1, x2)
```

## 参考资源

| 文件 | 说明 |
|---|---|
| `references/matmul_blaze_guide.md` | Blaze Matmul 工程模板指南（架构 + 场景路由 + 使用指南） |
| `references/kernel_launch_details.md` | 进阶（通用）：内存层次/Double Buffer/同步机制/多 I/O |

- [Ascend C 示例代码](https://gitcode.com/cann/asc-devkit/tree/master/examples)
- NPU 架构配置详见 `npu-arch` skill
- PyTorch 对接详见 `torch-ascendc-op-extension` skill
