# csrc/mla_preprocess：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **33 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/mla_preprocess/`

MLA 前处理的主机与设备实现。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/mla_preprocess/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/mla_preprocess/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/mla_preprocess/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [mla_preprocess_torch_adpt.h](../../../csrc/mla_preprocess/mla_preprocess_torch_adpt.h) | MLA 潜在注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`ParseCacheMode`, `ParseQuantMode` |

## `csrc/mla_preprocess/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/mla_preprocess/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [mla_preprocess_def.cpp](../../../csrc/mla_preprocess/op_host/mla_preprocess_def.cpp) | op_host 的主机算子定义、支持属性和注册。 | 注册：`MlaPreprocess` |
| [mla_preprocess_tiling.cpp](../../../csrc/mla_preprocess/op_host/mla_preprocess_tiling.cpp) | op_host 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`MlaPreprocessCompileInfo` |
| [mla_preprocess_tiling.h](../../../csrc/mla_preprocess/op_host/mla_preprocess_tiling.h) | op_host 的主机侧 tiling 参数/分块策略声明。 | 类型：`MlaPreprocess`, `QuantMode`, `PlatformInfo` |

## `csrc/mla_preprocess/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [kernel/](../../../csrc/mla_preprocess/op_kernel/kernel) | kernel 子模块；所属范围：MLA 前处理的主机与设备实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [mla_preprocess.cpp](../../../csrc/mla_preprocess/op_kernel/mla_preprocess.cpp) | op_kernel 的设备执行源码；文件主题为 MLA 潜在注意力。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [mla_preprocess.h](../../../csrc/mla_preprocess/op_kernel/mla_preprocess.h) | op_kernel 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`QuantMode` |
| [mla_preprocess_mix_bf16.hpp](../../../csrc/mla_preprocess/op_kernel/mla_preprocess_mix_bf16.hpp) | op_kernel 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`RopeFp16`, `Quant`, `RmsNormQuant` |
| [mla_preprocess_mix_bf16_nq.hpp](../../../csrc/mla_preprocess/op_kernel/mla_preprocess_mix_bf16_nq.hpp) | op_kernel 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`RopeFp16`, `RmsNormQuant`, `MatCoord` |
| [mla_preprocess_mix_bf16_qdown.hpp](../../../csrc/mla_preprocess/op_kernel/mla_preprocess_mix_bf16_qdown.hpp) | op_kernel 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`RopeFp16`, `Quant`, `RmsNormQuant` |
| [mla_preprocess_mix_fp16.hpp](../../../csrc/mla_preprocess/op_kernel/mla_preprocess_mix_fp16.hpp) | op_kernel 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`RopeFp16`, `Quant`, `RmsNormQuant` |
| [mla_preprocess_tiling_data.h](../../../csrc/mla_preprocess/op_kernel/mla_preprocess_tiling_data.h) | op_kernel 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`PpMatmulTilingData`, `MlaTilingData` |

## `csrc/mla_preprocess/op_kernel/kernel/`

kernel 子模块；所属范围：MLA 前处理的主机与设备实现。

| 子目录 | 职责 |
| --- | --- |
| [iterators/](../../../csrc/mla_preprocess/op_kernel/kernel/iterators) | iterators 子模块；所属范围：MLA 前处理的主机与设备实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [common.h](../../../csrc/mla_preprocess/op_kernel/kernel/common.h) | op_kernel 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [common_func.h](../../../csrc/mla_preprocess/op_kernel/kernel/common_func.h) | op_kernel 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`RoundUp`, `RoundUp` |
| [hardware.h](../../../csrc/mla_preprocess/op_kernel/kernel/hardware.h) | op_kernel 的类型/模板与设备实现头文件。 | 类型：`ArchType`, `HardwareInfo` |
| [iterator.h](../../../csrc/mla_preprocess/op_kernel/kernel/iterator.h) | op_kernel 的类型/模板与设备实现头文件。 | 类型：`gm_to_l1`, `l1_to_l0_a`, `l1_to_l0_b` |
| [kernel_utils.h](../../../csrc/mla_preprocess/op_kernel/kernel/kernel_utils.h) | op_kernel 的类型/模板与设备实现头文件；文件主题为 辅助函数。 | 函数：`CeilDiv`, `RoundUp` |
| [layout.h](../../../csrc/mla_preprocess/op_kernel/kernel/layout.h) | op_kernel 的类型/模板与设备实现头文件；文件主题为 内存布局。 | 类型：`DataFormat` |
| [mem.h](../../../csrc/mla_preprocess/op_kernel/kernel/mem.h) | op_kernel 的类型/模板与设备实现头文件。 | 类型：`BufferType`, `AsdopsBuffer` |
| [mma.h](../../../csrc/mla_preprocess/op_kernel/kernel/mma.h) | op_kernel 的类型/模板与设备实现头文件。 | 类型：`mmad` |
| [set_fpc.h](../../../csrc/mla_preprocess/op_kernel/kernel/set_fpc.h) | op_kernel 的类型/模板与设备实现头文件。 | 类型：`SetQuantPreAddr` |
| [simd.h](../../../csrc/mla_preprocess/op_kernel/kernel/simd.h) | op_kernel 的类型/模板与设备实现头文件。 | 函数：`cgadd_v`, `add_v` |
| [utils.h](../../../csrc/mla_preprocess/op_kernel/kernel/utils.h) | op_kernel 的类型/模板与设备实现头文件；文件主题为 辅助函数。 | 函数：`CreateCaMatrix`, `SetFftsBaseAddr` |

## `csrc/mla_preprocess/op_kernel/kernel/iterators/`

iterators 子模块；所属范围：MLA 前处理的主机与设备实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gm_to_l1_iterator.inc](../../../csrc/mla_preprocess/op_kernel/kernel/iterators/gm_to_l1_iterator.inc) | op_kernel 的设备执行源码。 | 类型：`gm_to_l1` |
| [gm_to_ub_iterator.inc](../../../csrc/mla_preprocess/op_kernel/kernel/iterators/gm_to_ub_iterator.inc) | op_kernel 的设备执行源码。 | 类型：`gm_to_ub`, `gm_to_ub_align`, `ub_to_ub` |
| [l0c_to_gm_iterator.inc](../../../csrc/mla_preprocess/op_kernel/kernel/iterators/l0c_to_gm_iterator.inc) | op_kernel 的设备执行源码。 | 类型：`l0c_to_gm` |
| [l0c_to_l1_iterator.inc](../../../csrc/mla_preprocess/op_kernel/kernel/iterators/l0c_to_l1_iterator.inc) | op_kernel 的设备执行源码。 | 类型：`l0c_to_l1` |
| [l0c_to_ub_iterator.inc](../../../csrc/mla_preprocess/op_kernel/kernel/iterators/l0c_to_ub_iterator.inc) | op_kernel 的设备执行源码。 | 类型：`l0c_to_ub` |
| [l1_to_bt_iterator.inc](../../../csrc/mla_preprocess/op_kernel/kernel/iterators/l1_to_bt_iterator.inc) | op_kernel 的设备执行源码。 | 类型：`l1_to_bt` |
| [l1_to_fb_iterator.inc](../../../csrc/mla_preprocess/op_kernel/kernel/iterators/l1_to_fb_iterator.inc) | op_kernel 的设备执行源码。 | 类型：`l1_to_fb` |
| [l1_to_l0_iterator.inc](../../../csrc/mla_preprocess/op_kernel/kernel/iterators/l1_to_l0_iterator.inc) | op_kernel 的设备执行源码。 | 类型：`l1_to_l0_a`, `l1_to_l0_b` |
| [l1_to_ub_iterator.inc](../../../csrc/mla_preprocess/op_kernel/kernel/iterators/l1_to_ub_iterator.inc) | op_kernel 的设备执行源码。 | 类型：`l1_to_ub`, `ub_to_l1` |
