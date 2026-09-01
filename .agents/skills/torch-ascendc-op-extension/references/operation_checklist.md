# 单文件 .asc 工程改造操作清单

本清单适用于 kernel + host + main 在单个 `.asc` 文件中的直调工程。

## xxx_torch.cpp 操作清单

| # | 操作 | 具体动作 |
|---|------|---------|
| 1 | 引入 Tiling 定义 | `#include "../op_kernel/xxx_tiling.h"`（已在 Step 0 抽离到 `op_kernel/` 目录） |
| 2 | 声明 kernel 入口 | 从 `xxx_kernel.asc` 复制核函数签名，去掉 `__global__ __vector__`，改为 `extern "C"` 声明 |
| 3 | 复制 Tiling 计算逻辑 | 从原 `.asc` 的 main() 中复制 tiling 计算代码到 `ascend_kernel::xxx()` 函数内 |
| 4 | 替换内存分配 | 原 `.asc` 的 `aclrtMalloc` / 文件读写 → `at::empty_like` / `at::empty` |
| 5 | 替换 stream | 原 `.asc` 的手动 `aclrtCreateStream` → `c10_npu::getCurrentNPUStream().stream(true)`（`stream(true)` 清 queue 防乱序，禁止用 `stream(false)`） |
| 6 | 替换 kernel 调用 | 原 `.asc` 的 `xxx_custom<<<blockNum, nullptr, stream>>>(arg1, arg2, ...)` → `xxx_custom(blockNum, nullptr, aclStream, arg1, arg2, ...)`。即：去掉 `<<<>>>`，将其中的 3 个参数（blockDim, l2Ctrl, stream）移到普通函数参数前面 |

## CMakeLists.txt 操作清单

| # | 操作 | 具体动作 |
|---|------|---------|
| 1 | 保留 Target 1 | 原有 `add_executable` 不动，用于回归验证 |
| 2 | 新增 Target 2 | `add_library(xxx_ops SHARED op_kernel/xxx_kernel.asc op_extension/xxx_torch.cpp op_extension/register.cpp)` |
| 3 | 设置 C++ 标准 | `set(CMAKE_CXX_STANDARD 17)`，torch 要求 C++17 |
| 4 | 链接库 | `torch_npu` + `ascendcl` + `ascendc_runtime` + `tiling_api` + `register` + `platform` + `unified_dlog` + `graph_base` + `dl` + `m` |
| 5 | npu-arch | A2/A3=dav-2201, A5=dav-3510 |
| 6 | 替换旧 Target | 如已有 PyTorch 相关 Target，直接替换，不要保留旧目标 |

注意：`xxx_kernel.asc` 会被两个 Target 分别编译（Target 1 通过 `xxx.asc` 的 `#include` 间接编译，Target 2 通过源文件列表直接编译），这是预期行为。
