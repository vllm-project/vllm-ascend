# csrc/gmm：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **82 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/gmm/`

分组矩阵乘及激活/量化融合算子。

| 子目录 | 职责 |
| --- | --- |
| [gmm_dequant_situ_quant/](../../../csrc/gmm/gmm_dequant_situ_quant) | GMM、反量化、SITU 与量化融合 算子工程；主机/设备职责以其子目录区分。 |
| [grouped_matmul_swiglu_quant/](../../../csrc/gmm/grouped_matmul_swiglu_quant) | GMM、SwiGLU 与量化融合 算子工程；主机/设备职责以其子目录区分。 |
| [grouped_matmul_swiglu_quant_v2/](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2) | GMM、SwiGLU 与量化融合 V2 算子工程；主机/设备职责以其子目录区分。 |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list/](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list) | NZ 权重列表的 GMM/SwiGLU/量化融合 算子工程；主机/设备职责以其子目录区分。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/gmm/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/gmm/gmm_dequant_situ_quant/`

GMM、反量化、SITU 与量化融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/gmm/gmm_dequant_situ_quant/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/gmm/gmm_dequant_situ_quant/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/gmm/gmm_dequant_situ_quant/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/gmm/gmm_dequant_situ_quant/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：GmmDequantSituQuant (A3) |
| [gmm_dequant_situ_quant_torch_adpt.h](../../../csrc/gmm/gmm_dequant_situ_quant/gmm_dequant_situ_quant_torch_adpt.h) | 反量化、量化、SITU 激活 的公共接口/实现，供所在算子工程复用。 | 类型：`GmmDequantSituQuantWeights` |

## `csrc/gmm/gmm_dequant_situ_quant/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/gmm/gmm_dequant_situ_quant/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [gmm_dequant_situ_quant_def.cpp](../../../csrc/gmm/gmm_dequant_situ_quant/op_host/gmm_dequant_situ_quant_def.cpp) | GMM、反量化、SITU 与量化融合 的主机算子定义、支持属性和注册。 | 注册：`GmmDequantSituQuant` |
| [gmm_dequant_situ_quant_infershape.cpp](../../../csrc/gmm/gmm_dequant_situ_quant/op_host/gmm_dequant_situ_quant_infershape.cpp) | GMM、反量化、SITU 与量化融合 的主机侧辅助接口或实现。 | 函数：`InferShapeGmmDequantSituQuant`, `InferDataTypeGmmDequantSituQuant` |
| [gmm_dequant_situ_quant_tiling.cpp](../../../csrc/gmm/gmm_dequant_situ_quant/op_host/gmm_dequant_situ_quant_tiling.cpp) | GMM、反量化、SITU 与量化融合 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`GmmDequantSituQuantCompileInfo` |

## `csrc/gmm/gmm_dequant_situ_quant/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [exact_msd.h](../../../csrc/gmm/gmm_dequant_situ_quant/op_kernel/exact_msd.h) | GMM、反量化、SITU 与量化融合 的类型/模板与设备实现头文件。 | 类型：`GmsqExactMsdCube`, `GmsqExactMsdVector` |
| [gmm_dequant_situ_quant.cpp](../../../csrc/gmm/gmm_dequant_situ_quant/op_kernel/gmm_dequant_situ_quant.cpp) | GMM、反量化、SITU 与量化融合 的设备执行源码；文件主题为 反量化、量化、SITU 激活。 | 函数：`GetExpertPointerTable` |
| [gmm_dequant_situ_quant.h](../../../csrc/gmm/gmm_dequant_situ_quant/op_kernel/gmm_dequant_situ_quant.h) | GMM、反量化、SITU 与量化融合 的类型/模板与设备实现头文件；文件主题为 反量化、量化、SITU 激活。 | 类型：`defaults`, `DeviceGlMetaGm`, `DeviceGlMeta` |
| [gmm_dequant_situ_quant_tiling.h](../../../csrc/gmm/gmm_dequant_situ_quant/op_kernel/gmm_dequant_situ_quant_tiling.h) | GMM、反量化、SITU 与量化融合 的类型/模板与设备实现头文件；文件主题为 反量化、量化、SITU 激活。 | 类型：`GmmDequantSituQuantTilingData` |

## `csrc/gmm/grouped_matmul_swiglu_quant/`

GMM、SwiGLU 与量化融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/gmm/grouped_matmul_swiglu_quant/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/gmm/grouped_matmul_swiglu_quant/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：GroupedMatmulSwigluQuant |

## `csrc/gmm/grouped_matmul_swiglu_quant/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [grouped_matmul_swiglu_quant_def.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp) | GMM、SwiGLU 与量化融合 的主机算子定义、支持属性和注册。 | 注册：`GroupedMatmulSwigluQuant` |
| [grouped_matmul_swiglu_quant_infershape.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_infershape.cpp) | GMM、SwiGLU 与量化融合 的主机侧辅助接口或实现。 | 函数：`InferShape4GroupedMatmulSwigluQuant`, `InferDataType4GroupedMatmulSwigluQuant` |
| [grouped_matmul_swiglu_quant_tiling.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_tiling.cpp) | GMM、SwiGLU 与量化融合 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`GMMSwigluCompileInfo` |
| [grouped_matmul_swiglu_quant_tiling.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_tiling.h) | GMM、SwiGLU 与量化融合 的主机侧 tiling 参数/分块策略声明。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_grouped_matmul_swiglu_quant.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant.cpp) | GMM、SwiGLU 与量化融合 的主机侧辅助接口或实现。 | 函数：`CheckNotNull`, `CheckInputOutDims_A8W8` |
| [aclnn_grouped_matmul_swiglu_quant.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant.h) | GMM、SwiGLU 与量化融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [aclnn_grouped_matmul_swiglu_quant_weight_nz.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz.h) | GMM、SwiGLU 与量化融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/grouped_matmul_swiglu_quant.cpp) | GMM、SwiGLU 与量化融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/grouped_matmul_swiglu_quant.h) | GMM、SwiGLU 与量化融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [grouped_matmul_swiglu_pipeline.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_pipeline.h) | GMM、SwiGLU 与量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活。 | 类型：`mmType`, `GMMSwigluQuantPipelineSchedule` |
| [grouped_matmul_swiglu_quant.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.cpp) | GMM、SwiGLU 与量化融合 的设备执行源码；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.h) | GMM、SwiGLU 与量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`mmType`, `GMMSwigluCompute` |
| [grouped_matmul_swiglu_quant_a8w4_msd_mid.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_mid.h) | GMM、SwiGLU 与量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`mmType`, `GMMA8W4MidProcess` |
| [grouped_matmul_swiglu_quant_a8w4_msd_post.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_post.h) | GMM、SwiGLU 与量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`GMMA8W4PostProcess` |
| [grouped_matmul_swiglu_quant_a8w4_msd_pre.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_pre.h) | GMM、SwiGLU 与量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`GMMA8W4PreProcess` |
| [grouped_matmul_swiglu_quant_split_ws.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_split_ws.h) | GMM、SwiGLU 与量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`mmType`, `GMMSwigluSplitWorkSpaceCompute` |
| [grouped_matmul_swiglu_quant_utils.h](../../../csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_utils.h) | GMM、SwiGLU 与量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`AT_`, `BT_`, `CT_` |

## `csrc/gmm/grouped_matmul_swiglu_quant_v2/`

GMM、SwiGLU 与量化融合 V2 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [grouped_matmul_swiglu_quant_v2_torch_adpt.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h) | 矩阵乘、SwiGLU 激活、量化 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [grouped_matmul_swiglu_quant_v2_base_tiling.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_base_tiling.cpp) | GMM、SwiGLU 与量化融合 V2 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`AlignUp`, `CeilDiv` |
| [grouped_matmul_swiglu_quant_v2_base_tiling.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_base_tiling.h) | GMM、SwiGLU 与量化融合 V2 的主机侧 tiling 参数/分块策略声明。 | 类型：`GroupedMatmulSwigluQuantV2BaseTiling` |
| [grouped_matmul_swiglu_quant_v2_def.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp) | GMM、SwiGLU 与量化融合 V2 的主机算子定义、支持属性和注册。 | 注册：`GroupedMatmulSwigluQuantV2` |
| [grouped_matmul_swiglu_quant_v2_fusion_tiling.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_fusion_tiling.cpp) | GMM、SwiGLU 与量化融合 V2 的主机侧 tiling 参数/分块策略计算实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant_v2_fusion_tiling.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_fusion_tiling.h) | GMM、SwiGLU 与量化融合 V2 的主机侧 tiling 参数/分块策略声明。 | 类型：`GroupedMatmulSwigluQuantV2FusionTiling` |
| [grouped_matmul_swiglu_quant_v2_host_utils.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_host_utils.h) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant_v2_infershape.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_infershape.cpp) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 函数：`isSupportedInputDtypeForDavid`, `InferShape4GroupedMatmulSwigluQuantV2` |
| [grouped_matmul_swiglu_quant_v2_tiling.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_tiling.cpp) | GMM、SwiGLU 与量化融合 V2 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`GroupedMatmulSwigluQuantV2TilingFunc`, `TilingPrepareForGMMSwigluQuantV2` |
| [grouped_matmul_swiglu_quant_v2_tiling.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_tiling.h) | GMM、SwiGLU 与量化融合 V2 的主机侧 tiling 参数/分块策略声明。 | 类型：`GMMSwigluV2CompileInfo`, `GroupedMatmulSwigluQuantV2Tiling` |

## `csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_grouped_matmul_swiglu_quant_v2.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 类型：`GmmDsqHandlerFactory` |
| [aclnn_grouped_matmul_swiglu_quant_v2.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.h) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 函数：`aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize` |
| [aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 函数：`aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize` |
| [gmm_dsq_base.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/gmm_dsq_base.h) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 类型：`GroupedMatmulSwigluQuantBaseHandler` |
| [grouped_matmul_swiglu_quant_utils.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/grouped_matmul_swiglu_quant_utils.h) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 类型：`GroupedMatmulSwigluQuantParamsBase`, `GroupedMatmulSwigluQuantParamsBuilder`, `GroupedMatmulSwigluQuantHandler` |
| [grouped_matmul_swiglu_quant_v2.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/grouped_matmul_swiglu_quant_v2.cpp) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant_v2.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/grouped_matmul_swiglu_quant_v2.h) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant_v2_utils.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/grouped_matmul_swiglu_quant_v2_utils.h) | GMM、SwiGLU 与量化融合 V2 的主机侧辅助接口或实现。 | 类型：`GroupedMatmulSwigluQuantBaseHandler` |

## `csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35) | arch35 子模块；所属范围：分组矩阵乘及激活/量化融合算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [grouped_matmul_swiglu_quant_spilit_fusion.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_spilit_fusion.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、算子融合、SwiGLU 激活。 | 类型：`GroupedMatmulDequantSwigluQuantFusion` |
| [grouped_matmul_swiglu_quant_v2.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2.cpp) | GMM、SwiGLU 与量化融合 V2 的设备执行源码；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant_v2_a4w4_mid.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a4w4_mid.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`mmType`, `GMMA4W4MidProcess` |
| [grouped_matmul_swiglu_quant_v2_a4w4_pipeline.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a4w4_pipeline.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`mmType`, `GMMSwigluQuantPipelineSchedule` |
| [grouped_matmul_swiglu_quant_v2_a4w4_post.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a4w4_post.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`GMMA4W4PostProcess` |
| [grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`mmType`, `GMMA8W4MidProcess` |
| [grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`mmType`, `GMMSwigluQuantPipelineSchedule` |
| [grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`GMMA8W4PostProcess` |
| [grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`GMMA8W4PreProcess` |
| [grouped_matmul_swiglu_quant_v2_apt.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp) | GMM、SwiGLU 与量化融合 V2 的设备执行源码；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 函数：`grouped_matmul_swiglu_quant_v2` |
| [grouped_matmul_swiglu_quant_v2_utils.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_utils.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 类型：`AT_`, `BT_`, `CT_` |

## `csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/`

arch35 子模块；所属范围：分组矩阵乘及激活/量化融合算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [grouped_matmul_swiglu_quant_v2_mxquant.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/grouped_matmul_swiglu_quant_v2_mxquant.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 函数：`GmmSwigluAswt` |
| [grouped_matmul_swiglu_quant_v2_pertoken_quant.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/grouped_matmul_swiglu_quant_v2_pertoken_quant.h) | GMM、SwiGLU 与量化融合 V2 的类型/模板与设备实现头文件；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 函数：`GmmSwigluAswtPertokenKernel`, `GmmSwigluAswtPertoken` |
| [grouped_matmul_swiglu_quant_v2_tiling_key.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h) | GMM、SwiGLU 与量化融合 V2 的kernel 分派使用的 tiling key 定义；文件主题为 矩阵乘、SwiGLU 激活、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/`

NZ 权重列表的 GMM/SwiGLU/量化融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [grouped_matmul_swiglu_quant_torch_adpt.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h) | 矩阵乘、SwiGLU 激活、量化 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/grouped_matmul_swiglu_quant_weight_nz_tensor_list.h) | NZ 权重列表的 GMM/SwiGLU/量化融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list_def.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/grouped_matmul_swiglu_quant_weight_nz_tensor_list_def.cpp) | NZ 权重列表的 GMM/SwiGLU/量化融合 的主机算子定义、支持属性和注册。 | 注册：`GroupedMatmulSwigluQuantWeightNzTensorList` |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list_infershape.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/grouped_matmul_swiglu_quant_weight_nz_tensor_list_infershape.cpp) | NZ 权重列表的 GMM/SwiGLU/量化融合 的主机侧辅助接口或实现。 | 函数：`InferShape4GroupedMatmulSwigluQuantWeightNzTensorList`, `InferDataType4GroupedMatmulSwigluQuantWeightNzTensorList` |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list_proto.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/grouped_matmul_swiglu_quant_weight_nz_tensor_list_proto.cpp) | NZ 权重列表的 GMM/SwiGLU/量化融合 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShape4GroupedMatmulSwigluQuantWeightNzTensorList`, `InferDataType4GroupedMatmulSwigluQuantWeightNzTensorList` |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.cpp) | NZ 权重列表的 GMM/SwiGLU/量化融合 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`GMMSwigluCompileInfo` |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.h) | NZ 权重列表的 GMM/SwiGLU/量化融合 的主机侧 tiling 参数/分块策略声明。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp) | NZ 权重列表的 GMM/SwiGLU/量化融合 的主机侧辅助接口或实现。 | 函数：`CheckNotNull`, `CheckInputOutDims` |
| [aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.h) | NZ 权重列表的 GMM/SwiGLU/量化融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/op_api/grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp) | NZ 权重列表的 GMM/SwiGLU/量化融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_kernel/grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp) | NZ 权重列表的 GMM/SwiGLU/量化融合 的设备执行源码；文件主题为 矩阵乘、权重、张量。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_kernel/grouped_matmul_swiglu_quant_weight_nz_tensor_list.h) | NZ 权重列表的 GMM/SwiGLU/量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、权重、张量。 | 类型：`mmType`, `GMMSwigluCompute` |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list_split_ws.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_kernel/grouped_matmul_swiglu_quant_weight_nz_tensor_list_split_ws.h) | NZ 权重列表的 GMM/SwiGLU/量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、权重、张量。 | 类型：`mmType`, `GMMSwigluSplitWorkSpaceCompute` |
| [grouped_matmul_swiglu_quant_weight_nz_tensor_list_utils.h](../../../csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_kernel/grouped_matmul_swiglu_quant_weight_nz_tensor_list_utils.h) | NZ 权重列表的 GMM/SwiGLU/量化融合 的类型/模板与设备实现头文件；文件主题为 矩阵乘、权重、张量。 | 类型：`AT_`, `BT_`, `CT_` |
