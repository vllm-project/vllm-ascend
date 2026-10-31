# csrc/mc2：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **141 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/mc2/`

通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 子目录 | 职责 |
| --- | --- |
| [dispatch_ffn_combine/](../../../csrc/mc2/dispatch_ffn_combine) | token 分发、FFN 与合并融合 算子工程；主机/设备职责以其子目录区分。 |
| [dispatch_ffn_combine_bf16/](../../../csrc/mc2/dispatch_ffn_combine_bf16) | BF16 token 分发、FFN 与合并融合 算子工程；主机/设备职责以其子目录区分。 |
| [dispatch_ffn_combine_w4_a8/](../../../csrc/mc2/dispatch_ffn_combine_w4_a8) | W4A8 token 分发、FFN 与合并融合 算子工程；主机/设备职责以其子目录区分。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/mc2/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/mc2/dispatch_ffn_combine/`

token 分发、FFN 与合并融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/mc2/dispatch_ffn_combine/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/mc2/dispatch_ffn_combine/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/mc2/dispatch_ffn_combine/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [dispatch_ffn_combine_torch_adpt.h](../../../csrc/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h) | token 分发、token 合并 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/mc2/dispatch_ffn_combine/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/mc2/dispatch_ffn_combine/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/mc2/dispatch_ffn_combine/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [dispatch_ffn_combine_def.cpp](../../../csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_def.cpp) | token 分发、FFN 与合并融合 的主机算子定义、支持属性和注册。 | 注册：`DispatchFFNCombine` |
| [dispatch_ffn_combine_proto.cpp](../../../csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_proto.cpp) | token 分发、FFN 与合并融合 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShapeDispatchFFNCombine`, `InferDataTypeDispatchFFNCombine` |
| [dispatch_ffn_combine_tiling.cpp](../../../csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp) | token 分发、FFN 与合并融合 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`DispatchFFNCombineCompileInfo` |
| [hcom_topo_info.h](../../../csrc/mc2/dispatch_ffn_combine/op_host/hcom_topo_info.h) | token 分发、FFN 与合并融合 的主机侧辅助接口或实现。 | 类型：`HcomTopoInfo`, `TopoLevel`, `TopoLevelDesc` |
| [tiling_args.h](../../../csrc/mc2/dispatch_ffn_combine/op_host/tiling_args.h) | token 分发、FFN 与合并融合 的主机侧 tiling 参数/分块策略声明。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/mc2/dispatch_ffn_combine/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_dispatch_ffn_combine.cpp](../../../csrc/mc2/dispatch_ffn_combine/op_host/op_api/aclnn_dispatch_ffn_combine.cpp) | token 分发、FFN 与合并融合 的主机侧辅助接口或实现。 | 函数：`aclnnInnerDispatchFFNCombineGetWorkspaceSize`, `aclnnInnerDispatchFFNCombine` |
| [aclnn_dispatch_ffn_combine.h](../../../csrc/mc2/dispatch_ffn_combine/op_host/op_api/aclnn_dispatch_ffn_combine.h) | token 分发、FFN 与合并融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/mc2/dispatch_ffn_combine/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [moe_init_routing_quant_v2/](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2) | 量化、混合专家 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |
| [unpermute/](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/unpermute) | 重排 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |
| [utils/](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils) | 辅助函数 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [dispatch_ffn_combine.cpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine.cpp) | token 分发、FFN 与合并融合 的设备执行源码；文件主题为 token 分发、token 合并。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dispatch_ffn_combine.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发、token 合并。 | 类型：`DispatchFFNCombine` |
| [dispatch_ffn_combine_kernel.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发、token 合并。 | 类型：`OpSystemRunCfg`, `BlockMmad_`, `BlockScheduler_` |
| [dispatch_ffn_combine_tiling.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_tiling.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发、token 合并。 | 类型：`DispatchFFNCombineInfo`, `CoCTiling`, `DispatchFFNCombineTilingData` |

## `csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/`

量化、混合专家 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [moe_init_routing_quant_v2.cpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp) | token 分发、FFN 与合并融合 的设备执行源码；文件主题为 量化、混合专家。 | 类型：`DTYPE_X` |
| [moe_init_routing_quant_v2_tiling.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、混合专家。 | 类型：`MoeInitRoutingQuantV2TilingData`, `MoeInitRoutingQuantV2TilingBase` |
| [moe_init_routing_v2_tiling.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_init_routing_v2_tiling.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`InnerMoeV2VBSComputeTilingData`, `InnerMoeV2VMSMiddleComputeTilingData`, `InnerMoeV2SortOutComputeTilingData` |
| [moe_v2_common.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_common.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 公共逻辑、混合专家。 | 函数：`Ceil`, `Align` |
| [moe_v2_expert_token_out.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 专家、混合专家。 | 类型：`MoeV2ExpertTokenOut` |
| [moe_v2_fullload_dynamic_quant.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、加载、混合专家。 | 类型：`MoeV2FullLoadDynamicQuant` |
| [moe_v2_fullload_quant.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、加载、混合专家。 | 类型：`MoeV2FullLoadQuant` |
| [moe_v2_fullload_quant_base.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant_base.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、加载、混合专家。 | 类型：`MoeV2FullLoadQuantBase` |
| [moe_v2_gather_dynamic_quant.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_gather_dynamic_quant.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 收集、量化、混合专家。 | 类型：`MoeV2GatherDynamicQuant` |
| [moe_v2_gather_out.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_gather_out.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 收集、混合专家。 | 类型：`MoeV2GatherOut` |
| [moe_v2_gather_quant.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_gather_quant.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 收集、量化、混合专家。 | 类型：`MoeV2GatherQuant` |
| [moe_v2_init_routing_fullload.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_init_routing_fullload.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 加载、混合专家。 | 类型：`MoeV2FullLoad` |
| [moe_v2_mrgsort.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_mrgsort.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2MrgsortParam`, `MoeV2Mrgsort` |
| [moe_v2_mrgsort_out.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_mrgsort_out.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2MrgsortOut` |
| [moe_v2_sort_base.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_sort_base.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SortBase` |
| [moe_v2_sort_multi_core.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_sort_multi_core.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SortMultiCore` |
| [moe_v2_sort_one_core.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_sort_one_core.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SortOneCore` |
| [moe_v2_src_to_dst_and_gather.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 收集、混合专家。 | 类型：`MoeV2SrcToDstAndGather` |
| [moe_v2_src_to_dst_op.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_op.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SrcToDstOp` |
| [moe_v2_src_to_dst_with_capacity.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_with_capacity.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SrcToDstWithCapacity` |
| [tiling_base.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/tiling_base.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件。 | 类型：`AiCoreParams`, `TilingBaseClass` |

## `csrc/mc2/dispatch_ffn_combine/op_kernel/unpermute/`

重排 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [moe_token_unpermute.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/unpermute/moe_token_unpermute.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 重排、混合专家。 | 类型：`KernelMoeTokenUnpermute` |
| [moe_token_unpermute_tiling.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/unpermute/moe_token_unpermute_tiling.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 重排、混合专家。 | 类型：`MoeTokenUnpermuteTilingData` |

## `csrc/mc2/dispatch_ffn_combine/op_kernel/utils/`

辅助函数 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue_pertoken_row.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/block_epilogue_pertoken_row.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`CType_`, `LayoutPerTokenScale_`, `DType_` |
| [block_epilogue_pertoken_swiglu.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/block_epilogue_pertoken_swiglu.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 SwiGLU 激活、缓存页/分块。 | 类型：`CType_`, `LayoutPerTokenScale_`, `DType_` |
| [block_epilogue_pertoken_v2.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/block_epilogue_pertoken_v2.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`CType_`, `LayoutPerTokenScale_`, `DType_` |
| [block_mmad_preload_async_fixpipe_quant.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、缓存页/分块、加载。 | 类型：`L1TileShape_`, `L0TileShape_`, `AType_` |
| [const_args.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/const_args.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [copy_gm_to_l1_custom.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/copy_gm_to_l1_custom.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `Element`, `CopyGmToL1` |
| [copy_l0c_to_gm_custom.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/copy_l0c_to_gm_custom.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ElementAccumulator_`, `ElementDst_`, `CopyL0CToGm` |
| [dispatch_policy_custom.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/dispatch_policy_custom.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发。 | 类型：`MmadAtlasA2PreloadFixpipeQuant`, `MmadAtlasA2PreloadAsyncFixpipe`, `EpilogueAtlasA2UnQuant` |
| [get_tensor_addr.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/get_tensor_addr.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 张量。 | 函数：`GetTensorAddr` |
| [hccl_shmem.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/hccl_shmem.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 HCCL 通信。 | 类型：`HcclShmem` |
| [layout3d.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/layout3d.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 内存布局。 | 类型：`Layout3D` |
| [moe_distribute_base.h](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/moe_distribute_base.h) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`HcclSignalInfo`, `ListCommon`, `HcclStreamInfo` |
| [select_helper.hpp](../../../csrc/mc2/dispatch_ffn_combine/op_kernel/utils/select_helper.hpp) | token 分发、FFN 与合并融合 的类型/模板与设备实现头文件。 | 类型：`LayoutBInitializer` |

## `csrc/mc2/dispatch_ffn_combine_bf16/`

BF16 token 分发、FFN 与合并融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/mc2/dispatch_ffn_combine_bf16/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/mc2/dispatch_ffn_combine_bf16/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [dispatch_ffn_combine_bf16_def.cpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host/dispatch_ffn_combine_bf16_def.cpp) | BF16 token 分发、FFN 与合并融合 的主机算子定义、支持属性和注册。 | 注册：`DispatchFFNCombineBF16` |
| [dispatch_ffn_combine_bf16_proto.cpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host/dispatch_ffn_combine_bf16_proto.cpp) | BF16 token 分发、FFN 与合并融合 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShapeDispatchFFNCombineBF16`, `InferDataTypeDispatchFFNCombineBF16` |
| [dispatch_ffn_combine_bf16_tiling.cpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host/dispatch_ffn_combine_bf16_tiling.cpp) | BF16 token 分发、FFN 与合并融合 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`DispatchFFNCombineBF16CompileInfo` |
| [hcom_topo_info.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host/hcom_topo_info.h) | BF16 token 分发、FFN 与合并融合 的主机侧辅助接口或实现。 | 类型：`HcomTopoInfo`, `TopoLevel`, `TopoLevelDesc` |
| [tiling_args.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host/tiling_args.h) | BF16 token 分发、FFN 与合并融合 的主机侧 tiling 参数/分块策略声明。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/mc2/dispatch_ffn_combine_bf16/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_dispatch_ffn_combine_bf16.cpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host/op_api/aclnn_dispatch_ffn_combine_bf16.cpp) | BF16 token 分发、FFN 与合并融合 的主机侧辅助接口或实现。 | 函数：`aclnnInnerDispatchFFNCombineBF16GetWorkspaceSize`, `aclnnInnerDispatchFFNCombineBF16` |
| [aclnn_dispatch_ffn_combine_bf16.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_host/op_api/aclnn_dispatch_ffn_combine_bf16.h) | BF16 token 分发、FFN 与合并融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [moe_init_routing_v2/](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2) | 混合专家 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |
| [unpermute/](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/unpermute) | 重排 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |
| [utils/](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils) | 辅助函数 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [dispatch_ffn_combine_bf16.cpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/dispatch_ffn_combine_bf16.cpp) | BF16 token 分发、FFN 与合并融合 的设备执行源码；文件主题为 token 分发、token 合并。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dispatch_ffn_combine_bf16.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/dispatch_ffn_combine_bf16.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发、token 合并。 | 类型：`DispatchFFNCombineBF16` |
| [dispatch_ffn_combine_bf16_kernel.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/dispatch_ffn_combine_bf16_kernel.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发、token 合并。 | 类型：`BlockMmad_`, `BlockScheduler_`, `ElementGroupList_` |
| [dispatch_ffn_combine_bf16_tiling.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/dispatch_ffn_combine_bf16_tiling.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发、token 合并。 | 类型：`DispatchFFNCombineBF16Info`, `CoCTiling`, `DispatchFFNCombineBF16TilingData` |

## `csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/`

混合专家 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [moe_init_routing_v2.cpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_init_routing_v2.cpp) | BF16 token 分发、FFN 与合并融合 的设备执行源码；文件主题为 混合专家。 | 类型：`DTYPE_X` |
| [moe_init_routing_v2_tiling.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_init_routing_v2_tiling.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2VBSComputeTilingData`, `MoeV2VMSMiddleComputeTilingData`, `MoeV2SortOutComputeTilingData` |
| [moe_v2_common.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_common.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 公共逻辑、混合专家。 | 函数：`Ceil`, `Align` |
| [moe_v2_expert_token_out.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_expert_token_out.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 专家、混合专家。 | 类型：`MoeV2ExpertTokenOut` |
| [moe_v2_gather_out.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_gather_out.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 收集、混合专家。 | 类型：`MoeV2GatherOut` |
| [moe_v2_init_routing_fullload.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_init_routing_fullload.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 加载、混合专家。 | 类型：`MoeV2FullLoad` |
| [moe_v2_mrgsort.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_mrgsort.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2MrgsortParam`, `MoeV2Mrgsort` |
| [moe_v2_mrgsort_out.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_mrgsort_out.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2MrgsortOut` |
| [moe_v2_sort_base.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_sort_base.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SortBase` |
| [moe_v2_sort_multi_core.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_sort_multi_core.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SortMultiCore` |
| [moe_v2_sort_one_core.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_sort_one_core.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SortOneCore` |
| [moe_v2_src_to_dst_op.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_src_to_dst_op.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SrcToDstOp` |
| [moe_v2_src_to_dst_op_simt.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_src_to_dst_op_simt.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SrcToDstOpSimt` |
| [moe_v2_src_to_dst_with_capacity.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_src_to_dst_with_capacity.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SrcToDstWithCapacity` |
| [tiling_base.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/tiling_base.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件。 | 类型：`AiCoreParams`, `TilingBaseClass` |

## `csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/unpermute/`

重排 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [moe_token_unpermute.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/unpermute/moe_token_unpermute.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 重排、混合专家。 | 类型：`KernelMoeTokenUnpermute` |
| [moe_token_unpermute_tiling.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/unpermute/moe_token_unpermute_tiling.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 重排、混合专家。 | 类型：`MoeTokenUnpermuteTilingData` |

## `csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/`

辅助函数 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue_pertoken_row.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/block_epilogue_pertoken_row.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`CType_`, `LayoutPerTokenScale_`, `DType_` |
| [block_epilogue_pertoken_swiglu.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/block_epilogue_pertoken_swiglu.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 SwiGLU 激活、缓存页/分块。 | 类型：`CType_`, `LayoutPerTokenScale_`, `DType_` |
| [block_epilogue_pertoken_v2.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/block_epilogue_pertoken_v2.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`CType_`, `LayoutPerTokenScale_`, `DType_` |
| [block_mmad_preload_async_fixpipe_quant.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、缓存页/分块、加载。 | 类型：`L1TileShape_`, `L0TileShape_`, `AType_` |
| [const_args.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/const_args.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [copy_gm_to_l1_custom.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/copy_gm_to_l1_custom.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `Element`, `CopyGmToL1` |
| [copy_l0c_to_gm_custom.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/copy_l0c_to_gm_custom.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ElementAccumulator_`, `ElementDst_`, `CopyL0CToGm` |
| [dispatch_policy_custom.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/dispatch_policy_custom.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发。 | 类型：`MmadAtlasA2PreloadFixpipeQuant`, `MmadAtlasA2PreloadAsyncFixpipe`, `EpilogueAtlasA2UnQuant` |
| [get_tensor_addr.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/get_tensor_addr.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 张量。 | 函数：`GetTensorAddr` |
| [hccl_shmem.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/hccl_shmem.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 HCCL 通信。 | 类型：`HcclShmem` |
| [layout3d.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/layout3d.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 内存布局。 | 类型：`Layout3D` |
| [moe_distribute_base.h](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/moe_distribute_base.h) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`HcclSignalInfo`, `ListCommon`, `HcclStreamInfo` |
| [select_helper.hpp](../../../csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/select_helper.hpp) | BF16 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件。 | 类型：`LayoutBInitializer` |

## `csrc/mc2/dispatch_ffn_combine_w4_a8/`

W4A8 token 分发、FFN 与合并融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [dispatch_ffn_combine_w4_a8_def.cpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_def.cpp) | W4A8 token 分发、FFN 与合并融合 的主机算子定义、支持属性和注册。 | 注册：`DispatchFFNCombineW4A8` |
| [dispatch_ffn_combine_w4_a8_proto.cpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_proto.cpp) | W4A8 token 分发、FFN 与合并融合 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShapeDispatchFFNCombineW4A8`, `InferDataTypeDispatchFFNCombineW4A8` |
| [dispatch_ffn_combine_w4_a8_tiling.cpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp) | W4A8 token 分发、FFN 与合并融合 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`DispatchFFNCombineW4A8CompileInfo` |
| [hcom_topo_info.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/hcom_topo_info.h) | W4A8 token 分发、FFN 与合并融合 的主机侧辅助接口或实现。 | 类型：`HcomTopoInfo`, `TopoLevel`, `TopoLevelDesc` |
| [tiling_args.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/tiling_args.h) | W4A8 token 分发、FFN 与合并融合 的主机侧 tiling 参数/分块策略声明。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_dispatch_ffn_combine_w4_a8.cpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_dispatch_ffn_combine_w4_a8.cpp) | W4A8 token 分发、FFN 与合并融合 的主机侧辅助接口或实现。 | 函数：`aclnnInnerDispatchFFNCombineW4A8GetWorkspaceSize`, `aclnnInnerDispatchFFNCombineW4A8` |
| [aclnn_dispatch_ffn_combine_w4_a8.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_dispatch_ffn_combine_w4_a8.h) | W4A8 token 分发、FFN 与合并融合 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [moe_init_routing_quant_v2/](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2) | 量化、混合专家 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |
| [unpermute/](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/unpermute) | 重排 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |
| [utils/](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils) | 辅助函数 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [dispatch_ffn_combine_w4_a8.cpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.cpp) | W4A8 token 分发、FFN 与合并融合 的设备执行源码；文件主题为 token 分发、token 合并。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dispatch_ffn_combine_w4_a8.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发、token 合并。 | 类型：`DispatchFFNCombineW4A8` |
| [dispatch_ffn_combine_w4_a8_kernel.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发、token 合并。 | 类型：`BlockMmad_`, `BlockScheduler_`, `ElementGroupList_` |
| [dispatch_ffn_combine_w4_a8_tiling.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_tiling.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发、token 合并。 | 类型：`DispatchFFNCombineW4A8Info`, `CoCTiling`, `DispatchFFNCombineW4A8TilingData` |

## `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/`

量化、混合专家 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [moe_init_routing_quant_v2.cpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp) | W4A8 token 分发、FFN 与合并融合 的设备执行源码；文件主题为 量化、混合专家。 | 类型：`DTYPE_X` |
| [moe_init_routing_quant_v2_tiling.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、混合专家。 | 类型：`MoeInitRoutingQuantV2TilingData`, `MoeInitRoutingQuantV2TilingBase` |
| [moe_init_routing_v2_tiling.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_init_routing_v2_tiling.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`InnerMoeV2VBSComputeTilingData`, `InnerMoeV2VMSMiddleComputeTilingData`, `InnerMoeV2SortOutComputeTilingData` |
| [moe_v2_common.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_common.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 公共逻辑、混合专家。 | 函数：`Ceil`, `Align` |
| [moe_v2_expert_token_out.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 专家、混合专家。 | 类型：`MoeV2ExpertTokenOut` |
| [moe_v2_fullload_dynamic_quant.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、加载、混合专家。 | 类型：`MoeV2FullLoadDynamicQuant` |
| [moe_v2_fullload_quant.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、加载、混合专家。 | 类型：`MoeV2FullLoadQuant` |
| [moe_v2_fullload_quant_base.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant_base.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 量化、加载、混合专家。 | 类型：`MoeV2FullLoadQuantBase` |
| [moe_v2_gather_dynamic_quant.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_gather_dynamic_quant.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 收集、量化、混合专家。 | 类型：`MoeV2GatherDynamicQuant` |
| [moe_v2_gather_out.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_gather_out.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 收集、混合专家。 | 类型：`MoeV2GatherOut` |
| [moe_v2_gather_quant.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_gather_quant.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 收集、量化、混合专家。 | 类型：`MoeV2GatherQuant` |
| [moe_v2_init_routing_fullload.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_init_routing_fullload.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 加载、混合专家。 | 类型：`MoeV2FullLoad` |
| [moe_v2_mrgsort.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_mrgsort.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2MrgsortParam`, `MoeV2Mrgsort` |
| [moe_v2_mrgsort_out.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_mrgsort_out.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2MrgsortOut` |
| [moe_v2_sort_base.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_sort_base.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SortBase` |
| [moe_v2_sort_multi_core.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_sort_multi_core.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SortMultiCore` |
| [moe_v2_sort_one_core.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_sort_one_core.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SortOneCore` |
| [moe_v2_src_to_dst_and_gather.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 收集、混合专家。 | 类型：`MoeV2SrcToDstAndGather` |
| [moe_v2_src_to_dst_op.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_op.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SrcToDstOp` |
| [moe_v2_src_to_dst_with_capacity.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_with_capacity.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`MoeV2SrcToDstWithCapacity` |
| [tiling_base.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/tiling_base.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件。 | 类型：`AiCoreParams`, `TilingBaseClass` |

## `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/unpermute/`

重排 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [moe_token_unpermute.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/unpermute/moe_token_unpermute.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 重排、混合专家。 | 类型：`KernelMoeTokenUnpermute` |
| [moe_token_unpermute_tiling.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/unpermute/moe_token_unpermute_tiling.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 重排、混合专家。 | 类型：`MoeTokenUnpermuteTilingData` |

## `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/`

辅助函数 子模块；所属范围：通信与专家计算融合的 Dispatch/FFN/Combine 算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue_w4a8post_pertoken_swiglu.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 SwiGLU 激活、缓存页/分块。 | 类型：`CType_`, `LayoutPerTokenScale_`, `DType_` |
| [block_epilogue_w4a8post_pertoken_v2.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`CType_`, `LayoutPerTokenScale_`, `DType_` |
| [block_mmad_w4a4.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_mmad_w4a4.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`L1TileShape_`, `L0TileShape_`, `AType_` |
| [const_args.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/const_args.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [copy_gm_to_l1_custom.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/copy_gm_to_l1_custom.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `Element`, `CopyGmToL1` |
| [dispatch_policy_custom.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/dispatch_policy_custom.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 token 分发。 | 类型：`MmadAtlasA2PreloadFixpipeQuant`, `MmadAtlasA2PreloadAsyncFixpipe`, `MmadAtlasA2W4A4MatmulPerChannelDequant` |
| [get_tensor_addr.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/get_tensor_addr.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 张量。 | 函数：`GetTensorAddr` |
| [hccl_shmem.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/hccl_shmem.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 HCCL 通信。 | 类型：`HcclShmem` |
| [layout3d.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/layout3d.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 内存布局。 | 类型：`Layout3D` |
| [moe_distribute_base.h](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/moe_distribute_base.h) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件；文件主题为 混合专家。 | 类型：`HcclSignalInfo`, `ListCommon`, `HcclStreamInfo` |
| [select_helper.hpp](../../../csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/select_helper.hpp) | W4A8 token 分发、FFN 与合并融合 的类型/模板与设备实现头文件。 | 类型：`LayoutBInitializer` |
