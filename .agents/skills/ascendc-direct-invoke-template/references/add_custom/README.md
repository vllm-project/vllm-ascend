# Add Custom Kernel 直调样例

add_custom 算子的 Kernel 直调实现示例，同时支持 PyTorch 对接。

详细代码说明见 `op_kernel/add_custom_kernel.asc` 和 `op_kernel/add_custom_tiling.h` 中的注释（搜索 `[MODIFY]` 标记）。

## 文件结构

```
├── op_kernel/
│   ├── add_custom_tiling.h    Tiling 常量 + 结构体（kernel 和 host 共用）
│   └── add_custom_kernel.asc  纯 kernel 代码（KernelAdd 类 + add_custom_kernel 核函数入口）
├── op_host/
│   ├── add_custom.asc         Host + main 入口（#include "add_custom_kernel.asc"）
│   └── data_utils.h           数据读写工具
├── op_extension/
│   ├── add_custom_torch.cpp   PyTorch host 实现（Tiling 计算 + kernel launch）
│   ├── register.cpp           TORCH_LIBRARY 注册（含 Meta backend）
│   └── ops.h                  函数声明
├── CMakeLists.txt             双 target：可执行文件 + libadd_custom_ops.so
└── scripts/                   数据生成和验证脚本
```

## 快速开始

### 方式一：直调验证（可执行文件）

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

或手动执行：

```bash
mkdir -p build && cd build && cmake .. && make -j
cd .. && python3 scripts/gen_data.py
cd build && ./add_custom
python3 ../scripts/verify_result.py output/output.bin output/golden.bin
```

### 方式二：PyTorch 调用

```python
import torch
import torch_npu

torch.ops.load_library("build/libadd_custom_ops.so")

x1 = torch.randn(8, 2048, dtype=torch.float32).npu()
x2 = torch.randn(8, 2048, dtype=torch.float32).npu()
y = torch.ops.npu.add_custom(x1, x2)

assert torch.allclose(y.cpu(), x1.cpu() + x2.cpu())
```
