# 踩坑清单

## 编译期

| # | 问题 | 原因 | 解决方案 |
|---|------|------|---------|
| 1 | `_torch.cpp` 不能 `#include` `.asc` | `.asc` 含 `__aicore__`/`__gm__` 等 ASC 专有关键字，C++ 编译器不认识 | 用 `extern "C"` 声明 kernel 函数，CMake 编译 `.asc` 为对象文件再链接 |
| 2 | `aclrtGetDeviceInfo` 未声明 | torch 头文件中的宏干扰 ACL 头文件解析 | `#include "acl/acl.h"` 放在 `#include <torch/extension.h>` **之前** |
| 3 | C++ 中 `is_npu()` 不存在 | torch C++ API 中 NPU 属于 PrivateUse1 后端 | 使用 `tensor.is_privateuseone()` |
| 4 | Tiling 结构体需要在 `_torch.cpp` 中重复定义 | kernel 和 torch_library 是独立编译单元 | 将 Tiling 结构体和常量抽到独立的 `op_kernel/xxx_tiling.h`，`_kernel.asc` 和 `_torch.cpp` 都 `#include` 该头文件（头文件只含纯 C/C++ 语法，不含 ASC 关键字） |

## 链接期

| # | 问题 | 原因 | 解决方案 |
|---|------|------|---------|
| 5 | 链接报 `undefined symbol: AscendLaunchKernelWithHostArgs` | `<<<>>>` 语法展开后依赖运行时支持函数 | 链接 `ascendc_runtime`（路径：`$ASCEND_HOME_PATH/aarch64-linux/lib64/libascendc_runtime.a`） |
| 6 | 链接报 `undefined reference to xxx_custom` | `_torch.cpp` 中的 `extern "C"` 声明与 `_kernel.asc` 中的函数签名不匹配 | 检查参数类型和数量一致：`xxx_custom(uint32_t blockDim, void *l2Ctrl, aclrtStream stream, uint8_t* ...)` |
| 7 | `libxxx_ops.so` 加载时报 CXX ABI 错误 | 编译器版本不一致（GCC ABI） | 确保编译 `.so` 的 C++ 编译器与编译 torch 的编译器兼容，可通过 `torch.__config__.show()` 查看 torch 编译选项 |
| 8 | `TORCH_LIBRARY_FRAGMENT` 与 `TORCH_LIBRARY` 的区别 | `TORCH_LIBRARY` 要求命名空间唯一，`TORCH_LIBRARY_FRAGMENT` 允许多处注册同一命名空间 | 推荐用 `TORCH_LIBRARY_FRAGMENT`，多个算子可以在不同文件中注册到同一个 `npu` 命名空间 |

## 运行期

| # | 问题 | 原因 | 解决方案 |
|---|------|------|---------|
| 9 | 运行时结果随机错误或乱序 | `stream(false)` 不清 queue，kernel 在之前任务完成前就开始执行 | 改用 `stream(true)` 清 queue，或使用 OpCommand 入 queue。详见 SKILL.md「Stream 同步模式」章节 |
| 10 | 使用 OpCommand + NPUStream 导致死锁 | lambda 内传入 NPUStream 对象，`<<<>>>` 展开后等待 queue 清空，但 OpCommand 已将 lambda 入 queue | lambda 内改用 `aclrtStream`（通过 `stream(true)` 或 `stream(false)` 获取），不要传 NPUStream 对象 |
| 11 | `zeros_like` 创建输出后 kernel 乱序 | `stream(true)` 清 queue 后，`zeros_like` 入 queue 但 kernel 不入 queue，两者可能乱序 | 使用 `empty_like` 代替 `zeros_like`（empty_like 不会触发 NPU 操作），或统一用 OpCommand 管理 |
| 12 | Python 中 `y.is_npu()` 报错 | Python 中 `is_npu` 是 property 不是方法 | 用 `y.is_npu`（不带括号） |
| 13 | `torch.ops.load_library` 报 `cannot open shared object file` | `__init__.py` 中 `.so` 路径计算错误 | 检查 `pathlib.Path(__file__).parents[N]` 的层级。`__init__.py` 路径为 `python/<package>/__init__.py`，需要 `parents[2]` 才能上溯到算子工程根目录 |
| 14 | `torch.ops.npu.xxx` 不存在 / `has no attribute` | `libxxx_ops.so` 未加载，或 `TORCH_LIBRARY_FRAGMENT` 中的命名空间与调用不一致 | 确保 `import` 了加载 `.so` 的 Python 包，且 `TORCH_LIBRARY_FRAGMENT(npu, m)` 中的 `npu` 与 `torch.ops.npu.xxx()` 对应 |
| 15 | 运行时报 `PrivateUse1 backend not available` | torch_npu 未正确初始化 | 在调用算子前先 `import torch_npu`，确保 NPU 后端已注册 |
| 16 | `torch.compile` 报 `Meta backend not registered` | 缺少 Meta backend 注册 | 在 register.cpp 中添加 `TORCH_LIBRARY_IMPL(npu, Meta, m)` 注册 Meta 函数 |
