# csrc：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **10 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

先从 Python 的操作名反查 `torch_binding.cpp`，再进入对应算子的主机侧与设备侧实现。schema、ACLNN 接口和设备 kernel 必须共同匹配；构建文件决定哪些实现进入产物。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `csrc/`

自定义设备算子、PyTorch/CANN 绑定及算子构建工具链。

| 子目录 | 职责 |
| --- | --- |
| [aclnn_torch_adapter/](csrc--aclnn_torch_adapter.md) | PyTorch 张量与 ACLNN 算子调用之间的适配。 |
| [attention/](csrc--attention.md) | 注意力、索引、压缩、缓存写入与相关状态算子。 |
| [cmake/](csrc--cmake.md) | 算子工程的构建模块、脚本和外部依赖配置。 |
| [common/](csrc--common.md) | 各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [gmm/](csrc--gmm.md) | 分组矩阵乘及激活/量化融合算子。 |
| [kernels/](csrc--kernels.md) | 独立设备 kernel 和辅助实现。 |
| [mc2/](csrc--mc2.md) | 通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |
| [mla_preprocess/](csrc--mla_preprocess.md) | MLA 前处理的主机与设备实现。 |
| [moe/](csrc--moe.md) | 专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [scripts/](csrc--scripts.md) | 算子代码生成、编译产物处理和安装打包脚本。 |
| [third_party/](csrc--third_party.md) | 通过子模块接入的第三方算子开发依赖。 |
| [utils/](csrc--utils.md) | 设备/主机辅助函数、接口和工具实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/CMakeLists.txt) | 组织自定义算子扩展的构建、依赖与打包。 | 构建项：`cann_ops-transformer`, `op_host_aclnn`, `op_host_aclnnInner` |
| [build.sh](../../../csrc/build.sh) | 算子工程 build 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`build_lib`, `build_static_lib`, `package_static` |
| [build_aclnn.sh](../../../csrc/build_aclnn.sh) | 算子工程 build_aclnn 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`log`, `setup_catlass_dependency`, `resolve_op_dir` |
| [build_batch_invariant_ops.sh](../../../csrc/build_batch_invariant_ops.sh) | 算子工程 批次 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`log` |
| [camem_allocator.cpp](../../../csrc/camem_allocator.cpp) | 内存分配 的公共接口/实现，供所在算子工程复用。 | 类型：`PyModuleDef` |
| [ops.h](../../../csrc/ops.h) | ops 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [torch_binding.cpp](../../../csrc/torch_binding.cpp) | 为 PyTorch 注册自定义算子 schema 与 NPU 调用实现，连接 Python 和 ACLNN。 | 类型：`DevicePrintPayload`, `LruResidentCompactWithPlanPayload`, `LruResidentCompactGraphPayloadRegistry` |
| [torch_binding_meta.cpp](../../../csrc/torch_binding_meta.cpp) | 接口绑定 的公共接口/实现，供所在算子工程复用。 | 函数：`ceil_div`, `bgmv_expand_meta` |
| [utils.h](../../../csrc/utils.h) | 辅助函数 的公共接口/实现，供所在算子工程复用。 | 类型：`PyModuleDef`, `TrochBindException` |
| [version.info](../../../csrc/version.info) | version.info 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
