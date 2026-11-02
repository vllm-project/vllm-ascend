# csrc/common：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **124 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/common/`

各算子复用的类型、宏、注册、主机辅助与基础代码。

| 子目录 | 职责 |
| --- | --- |
| [aicpu/](../../../csrc/common/aicpu) | aicpu 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [include/](../../../csrc/common/include) | 公共头文件、类型和接口声明。 |
| [src/](../../../csrc/common/src) | 父模块的具体实现源文件。 |
| [stub/](../../../csrc/common/stub) | stub 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/common/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | 构建项：`ops_transformer_utils_tiling_headers`, `ops_transformer_utils_proto_headers`, `${COMMON_NAME}_obj` |

## `csrc/common/aicpu/`

aicpu 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [cpu_context_util.h](../../../csrc/common/aicpu/cpu_context_util.h) | cpu_context_util 的公共接口/实现，供所在算子工程复用。 | 函数：`GetAttrValue`, `GetAttrValue` |

## `csrc/common/include/`

公共头文件、类型和接口声明。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/common/include/common) | 公共逻辑 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [err/](../../../csrc/common/include/err) | err 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [external/](../../../csrc/common/include/external) | external 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [fallback/](../../../csrc/common/include/fallback) | fallback 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [framework/](../../../csrc/common/include/framework) | framework 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [graph/](../../../csrc/common/include/graph) | 设备图 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [kernel/](../../../csrc/common/include/kernel) | kernel 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [op_graph/](../../../csrc/common/include/op_graph) | 设备图 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [static/](../../../csrc/common/include/static) | static 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [tiling_base/](../../../csrc/common/include/tiling_base) | tiling_base 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [tiling_sink/](../../../csrc/common/include/tiling_sink) | tiling_sink 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [cann_compat.h](../../../csrc/common/include/cann_compat.h) | cann_compat 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/common/include/common/`

公共逻辑 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [op_api_def.h](../../../csrc/common/include/common/op_api_def.h) | op_api_def 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [tensor_util.cpp](../../../csrc/common/include/common/tensor_util.cpp) | 张量 的公共接口/实现，供所在算子工程复用。 | 函数：`getAllDims`, `ResizeFrom1D` |
| [tensor_util.h](../../../csrc/common/include/common/tensor_util.h) | 张量 的公共接口/实现，供所在算子工程复用。 | 函数：`ResizeFrom1D`, `ResizeTo1D` |

## `csrc/common/include/err/`

err 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [ops_err.h](../../../csrc/common/include/err/ops_err.h) | ops_err 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/common/include/external/`

external 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 子目录 | 职责 |
| --- | --- |
| [aclnn_kernels/](../../../csrc/common/include/external/aclnn_kernels) | aclnn_kernels 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_util.h](../../../csrc/common/include/external/aclnn_util.h) | aclnn_util 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/common/include/external/aclnn_kernels/`

aclnn_kernels 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/common/include/external/aclnn_kernels/common) | 公共逻辑 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [cast.h](../../../csrc/common/include/external/aclnn_kernels/cast.h) | 类型转换 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [contiguous.h](../../../csrc/common/include/external/aclnn_kernels/contiguous.h) | contiguous 的公共接口/实现，供所在算子工程复用。 | 函数：`CanOptimizeContiguous` |
| [pad.h](../../../csrc/common/include/external/aclnn_kernels/pad.h) | 补齐 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [reshape.h](../../../csrc/common/include/external/aclnn_kernels/reshape.h) | reshape 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [slice.h](../../../csrc/common/include/external/aclnn_kernels/slice.h) | slice 的公共接口/实现，供所在算子工程复用。 | 函数：`Slice`, `Slice` |
| [transdata.h](../../../csrc/common/include/external/aclnn_kernels/transdata.h) | transdata 的公共接口/实现，供所在算子工程复用。 | 函数：`TransDataSpecial` |
| [transpose.h](../../../csrc/common/include/external/aclnn_kernels/transpose.h) | 转置 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/common/include/external/aclnn_kernels/common/`

公共逻辑 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [op_error_check.h](../../../csrc/common/include/external/aclnn_kernels/common/op_error_check.h) | 检查 的公共接口/实现，供所在算子工程复用。 | 函数：`IsNullptr`, `IsNullptr` |

## `csrc/common/include/fallback/`

fallback 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fallback.h](../../../csrc/common/include/fallback/fallback.h) | fallback 的公共接口/实现，供所在算子工程复用。 | 类型：`index_sequence`, `make_index_sequence_helper`, `aclOpExecutor` |
| [fallback_2stages.h](../../../csrc/common/include/fallback/fallback_2stages.h) | fallback_2stages 的公共接口/实现，供所在算子工程复用。 | 函数：`Collect`, `Collect` |
| [fallback_comm.h](../../../csrc/common/include/fallback/fallback_comm.h) | fallback_comm 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [fallback_comm_2stages.h](../../../csrc/common/include/fallback/fallback_comm_2stages.h) | fallback_comm_2stages 的公共接口/实现，供所在算子工程复用。 | 类型：`OpApiParams` |

## `csrc/common/include/framework/`

framework 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [onnx_common.h](../../../csrc/common/include/framework/onnx_common.h) | 公共逻辑 的公共接口/实现，供所在算子工程复用。 | 函数：`GetOpName`, `Vec2Tensor` |

## `csrc/common/include/graph/`

设备图 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [error_codes.h](../../../csrc/common/include/graph/error_codes.h) | error_codes 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/common/include/kernel/`

kernel 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [common.h](../../../csrc/common/include/kernel/common.h) | 公共逻辑 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [common_func.h](../../../csrc/common/include/kernel/common_func.h) | 公共逻辑 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dropmask.h](../../../csrc/common/include/kernel/dropmask.h) | dropmask 的公共接口/实现，供所在算子工程复用。 | 类型：`DropMaskInfo` |
| [gm_to_l1_iterator.h](../../../csrc/common/include/kernel/gm_to_l1_iterator.h) | gm_to_l1_iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`gm_to_l1` |
| [gm_to_ub_iterator.h](../../../csrc/common/include/kernel/gm_to_ub_iterator.h) | gm_to_ub_iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`gm_to_ub`, `gm_to_ub_align`, `ub_to_ub` |
| [hardware.h](../../../csrc/common/include/kernel/hardware.h) | hardware 的公共接口/实现，供所在算子工程复用。 | 类型：`ArchType`, `HardwareInfo` |
| [iterator.h](../../../csrc/common/include/kernel/iterator.h) | iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`gm_to_l1`, `l1_to_l0_a`, `l1_to_l0_b` |
| [l0c_to_gm_iterator.h](../../../csrc/common/include/kernel/l0c_to_gm_iterator.h) | l0c_to_gm_iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`l0c_to_gm` |
| [l0c_to_l1_iterator.h](../../../csrc/common/include/kernel/l0c_to_l1_iterator.h) | l0c_to_l1_iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`l0c_to_l1` |
| [l0c_to_ub_iterator.h](../../../csrc/common/include/kernel/l0c_to_ub_iterator.h) | l0c_to_ub_iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`l0c_to_ub` |
| [l1_to_bt_iterator.h](../../../csrc/common/include/kernel/l1_to_bt_iterator.h) | l1_to_bt_iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`l1_to_bt` |
| [l1_to_fb_iterator.h](../../../csrc/common/include/kernel/l1_to_fb_iterator.h) | l1_to_fb_iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`l1_to_fb` |
| [l1_to_l0_iterator.h](../../../csrc/common/include/kernel/l1_to_l0_iterator.h) | l1_to_l0_iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`l1_to_l0_a`, `l1_to_l0_b` |
| [l1_to_ub_iterator.h](../../../csrc/common/include/kernel/l1_to_ub_iterator.h) | l1_to_ub_iterator 的公共接口/实现，供所在算子工程复用。 | 类型：`l1_to_ub`, `ub_to_l1` |
| [layout.h](../../../csrc/common/include/kernel/layout.h) | 内存布局 的公共接口/实现，供所在算子工程复用。 | 类型：`DataFormatT` |
| [mem.h](../../../csrc/common/include/kernel/mem.h) | mem 的公共接口/实现，供所在算子工程复用。 | 类型：`BufferType`, `AsdopsBuffer` |
| [mma.h](../../../csrc/common/include/kernel/mma.h) | mma 的公共接口/实现，供所在算子工程复用。 | 类型：`mmad` |
| [pse.h](../../../csrc/common/include/kernel/pse.h) | pse 的公共接口/实现，供所在算子工程复用。 | 类型：`PseTypeEnum`, `PseInfo` |
| [simd.h](../../../csrc/common/include/kernel/simd.h) | simd 的公共接口/实现，供所在算子工程复用。 | 函数：`add_v`, `adds_v` |
| [util.h](../../../csrc/common/include/kernel/util.h) | util 的公共接口/实现，供所在算子工程复用。 | 类型：`LayOutTypeEnum` |

## `csrc/common/include/op_graph/`

设备图 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [op_transformer_proto_extend.h](../../../csrc/common/include/op_graph/op_transformer_proto_extend.h) | op_transformer_proto_extend 的公共接口/实现，供所在算子工程复用。 | 注册：`MaskedSoftmaxWithRelPosBias`, `AttentionScore` |

## `csrc/common/include/static/`

static 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [op_resource.h](../../../csrc/common/include/static/op_resource.h) | op_resource 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [static_space.h](../../../csrc/common/include/static/static_space.h) | static_space 的公共接口/实现，供所在算子工程复用。 | 类型：`StaticSpaceInitializer` |

## `csrc/common/include/tiling_base/`

tiling_base 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [data_copy_transpose_tiling.h](../../../csrc/common/include/tiling_base/data_copy_transpose_tiling.h) | 转置、复制 的公共接口/实现，供所在算子工程复用。 | 函数：`GetDataCopyTransposeTiling` |
| [data_copy_transpose_tiling_def.h](../../../csrc/common/include/tiling_base/data_copy_transpose_tiling_def.h) | 转置、复制 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [error_log.h](../../../csrc/common/include/tiling_base/error_log.h) | 日志 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [tiling_base.h](../../../csrc/common/include/tiling_base/tiling_base.h) | tiling_base 的公共接口/实现，供所在算子工程复用。 | 类型：`AiCoreParams`, `CompileInfoCommon`, `FlashAttentionScoreGradCompileInfo` |
| [tiling_key.h](../../../csrc/common/include/tiling_base/tiling_key.h) | tiling_key 的公共接口/实现，供所在算子工程复用。 | 函数：`RecursiveSum` |
| [tiling_templates_registry.h](../../../csrc/common/include/tiling_base/tiling_templates_registry.h) | 注册表 的公共接口/实现，供所在算子工程复用。 | 类型：`TilingCases`, `TilingRegistryNew`, `RegisterNew` |
| [tiling_type.h](../../../csrc/common/include/tiling_base/tiling_type.h) | tiling_type 的公共接口/实现，供所在算子工程复用。 | 类型：`AxisEnum`, `DtypeEnum`, `PerformanceOrientedEnum` |
| [tiling_util.h](../../../csrc/common/include/tiling_base/tiling_util.h) | tiling_util 的公共接口/实现，供所在算子工程复用。 | 函数：`CeilAlign`, `CeilDiv` |

## `csrc/common/include/tiling_sink/`

tiling_sink 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [device_op_impl_registry_impl.h](../../../csrc/common/include/tiling_sink/device_op_impl_registry_impl.h) | 注册表 的公共接口/实现，供所在算子工程复用。 | 类型：`DeviceOpImplRegistry`, `DeviceOpImplRegisterImpl` |
| [tiling_aicpu_task.h](../../../csrc/common/include/tiling_sink/tiling_aicpu_task.h) | tiling_aicpu_task 的公共接口/实现，供所在算子工程复用。 | 类型：`TilingAicpuTask` |

## `csrc/common/src/`

父模块的具体实现源文件。

| 子目录 | 职责 |
| --- | --- |
| [framework/](../../../csrc/common/src/framework) | framework 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [tiling_base/](../../../csrc/common/src/tiling_base) | tiling_base 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [tiling_sink/](../../../csrc/common/src/tiling_sink) | tiling_sink 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fallback_comm.cpp](../../../csrc/common/src/fallback_comm.cpp) | fallback_comm 的公共接口/实现，供所在算子工程复用。 | 函数：`ToAclDataType` |
| [fallback_comm_2stages.cpp](../../../csrc/common/src/fallback_comm_2stages.cpp) | fallback_comm_2stages 的公共接口/实现，供所在算子工程复用。 | 函数：`ExecuteOpLaunch` |

## `csrc/common/src/framework/`

framework 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/common/src/framework/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [embedding_bag_onnx_plugin.cpp](../../../csrc/common/src/framework/embedding_bag_onnx_plugin.cpp) | 向量查表 的公共接口/实现，供所在算子工程复用。 | 函数：`ParseParamsEmbeddingBag` |
| [fillwindowcache_onnx_plugin.cpp](../../../csrc/common/src/framework/fillwindowcache_onnx_plugin.cpp) | 缓存 的公共接口/实现，供所在算子工程复用。 | 函数：`parseParamsFillWindowCache` |
| [multi_head_attention_onnx_plugin.cpp](../../../csrc/common/src/framework/multi_head_attention_onnx_plugin.cpp) | 注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`ParseParamsMultiHeadAttention` |
| [npu_fused_attention_score_fwd_onnx_plugin.cpp](../../../csrc/common/src/framework/npu_fused_attention_score_fwd_onnx_plugin.cpp) | 注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`ParseParamsNpuFusedAttentionScoreFwd`, `GetAttrFromOperator` |
| [npu_fused_attention_score_onnx_plugin.cpp](../../../csrc/common/src/framework/npu_fused_attention_score_onnx_plugin.cpp) | 注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`ParseParamsNpuFusedAttentionScore`, `GetAttrFromPre3` |
| [npu_masked_softmax_with_relposbias_onnx_plugin.cpp](../../../csrc/common/src/framework/npu_masked_softmax_with_relposbias_onnx_plugin.cpp) | 概率归一化、掩码处理 的公共接口/实现，供所在算子工程复用。 | 函数：`ParseParamsNpuMaskedSoftmaxWithRelPosBias`, `ParseOpToGraphNpuMaskedSoftmaxWithRelPosBias` |
| [npu_scaled_masked_softmax_onnx_plugin.cpp](../../../csrc/common/src/framework/npu_scaled_masked_softmax_onnx_plugin.cpp) | 概率归一化、掩码处理 的公共接口/实现，供所在算子工程复用。 | 函数：`ParseParamsNPUScaledMaskedSoftmax` |
| [tfidf_vectorizer_onnx_plugin.cpp](../../../csrc/common/src/framework/tfidf_vectorizer_onnx_plugin.cpp) | tfidf_vectorizer_onnx_plugin 的公共接口/实现，供所在算子工程复用。 | 函数：`ParseParamsTfIdfVectorizer` |

## `csrc/common/src/tiling_base/`

tiling_base 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [tiling_util.cpp](../../../csrc/common/src/tiling_base/tiling_util.cpp) | tiling_util 的公共接口/实现，供所在算子工程复用。 | 函数：`IsRegbaseSocVersion`, `IsRegbaseSocVersion` |

## `csrc/common/src/tiling_sink/`

tiling_sink 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/common/src/tiling_sink/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | 构建项：`tiling_sink_project`, `opmaster` |
| [tiling_sink_registry.cpp](../../../csrc/common/src/tiling_sink/tiling_sink_registry.cpp) | 注册表 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/common/stub/`

stub 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 子目录 | 职责 |
| --- | --- |
| [inc/](../../../csrc/common/stub/inc) | 公共头文件、类型和接口声明。 |
| [op_api/](../../../csrc/common/stub/op_api) | 算子对外调用接口及适配。 |
| [op_tiling/](../../../csrc/common/stub/op_tiling) | op_tiling 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/common/stub/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/common/stub/inc/`

公共头文件、类型和接口声明。

| 子目录 | 职责 |
| --- | --- |
| [framework/](../../../csrc/common/stub/inc/framework) | framework 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

## `csrc/common/stub/inc/framework/`

framework 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [stub_ops.h](../../../csrc/common/stub/inc/framework/stub_ops.h) | stub_ops 的公共接口/实现，供所在算子工程复用。 | 注册：`Data`, `Const` |

## `csrc/common/stub/op_api/`

算子对外调用接口及适配。

| 子目录 | 职责 |
| --- | --- |
| [aclnn_kernels/](../../../csrc/common/stub/op_api/aclnn_kernels) | aclnn_kernels 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [level0/](../../../csrc/common/stub/op_api/level0) | level0 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/common/stub/op_api/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | 构建项：`opapi_math`, `opapi_stub` |
| [opapi_stub.cpp](../../../csrc/common/stub/op_api/opapi_stub.cpp) | opapi_stub 的公共接口/实现，供所在算子工程复用。 | 函数：`CanOptimizeContiguous`, `CanOptimizeView` |

## `csrc/common/stub/op_api/aclnn_kernels/`

aclnn_kernels 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/common/stub/op_api/aclnn_kernels/common) | 公共逻辑 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [cast.h](../../../csrc/common/stub/op_api/aclnn_kernels/cast.h) | 类型转换 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [contiguous.h](../../../csrc/common/stub/op_api/aclnn_kernels/contiguous.h) | contiguous 的公共接口/实现，供所在算子工程复用。 | 函数：`CanOptimizeContiguous` |
| [pad.h](../../../csrc/common/stub/op_api/aclnn_kernels/pad.h) | 补齐 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [reshape.h](../../../csrc/common/stub/op_api/aclnn_kernels/reshape.h) | reshape 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [slice.h](../../../csrc/common/stub/op_api/aclnn_kernels/slice.h) | slice 的公共接口/实现，供所在算子工程复用。 | 函数：`Slice`, `Slice` |
| [transdata.h](../../../csrc/common/stub/op_api/aclnn_kernels/transdata.h) | transdata 的公共接口/实现，供所在算子工程复用。 | 函数：`TransDataSpecial` |
| [transpose.h](../../../csrc/common/stub/op_api/aclnn_kernels/transpose.h) | 转置 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/common/stub/op_api/aclnn_kernels/common/`

公共逻辑 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [op_error_check.h](../../../csrc/common/stub/op_api/aclnn_kernels/common/op_error_check.h) | 检查 的公共接口/实现，供所在算子工程复用。 | 函数：`IsNullptr`, `IsNullptr` |

## `csrc/common/stub/op_api/level0/`

level0 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [add.h](../../../csrc/common/stub/op_api/level0/add.h) | add 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [arange.h](../../../csrc/common/stub/op_api/level0/arange.h) | arange 的公共接口/实现，供所在算子工程复用。 | 函数：`Arange` |
| [axpy.h](../../../csrc/common/stub/op_api/level0/axpy.h) | axpy 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [batch_norm_backward.h](../../../csrc/common/stub/op_api/level0/batch_norm_backward.h) | 批次 的公共接口/实现，供所在算子工程复用。 | 函数：`BNTrainingReduceGrad`, `BN3DTrainingReduceGrad` |
| [broadcast_to.h](../../../csrc/common/stub/op_api/level0/broadcast_to.h) | 广播 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dilation.h](../../../csrc/common/stub/op_api/level0/dilation.h) | dilation 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [div.h](../../../csrc/common/stub/op_api/level0/div.h) | div 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dot.h](../../../csrc/common/stub/op_api/level0/dot.h) | dot 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [expand.h](../../../csrc/common/stub/op_api/level0/expand.h) | expand 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [fault_injection.h](../../../csrc/common/stub/op_api/level0/fault_injection.h) | fault_injection 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [fill.h](../../../csrc/common/stub/op_api/level0/fill.h) | fill 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [gather_elements.h](../../../csrc/common/stub/op_api/level0/gather_elements.h) | 收集 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [gather_v2.h](../../../csrc/common/stub/op_api/level0/gather_v2.h) | 收集 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [gather_v3.h](../../../csrc/common/stub/op_api/level0/gather_v3.h) | 收集 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [inplace_index_add.h](../../../csrc/common/stub/op_api/level0/inplace_index_add.h) | inplace_index_add 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [masked_scatter.h](../../../csrc/common/stub/op_api/level0/masked_scatter.h) | 分散写入、掩码处理 的公共接口/实现，供所在算子工程复用。 | 函数：`MaskedScatter` |
| [matmul_v2tov3.h](../../../csrc/common/stub/op_api/level0/matmul_v2tov3.h) | 矩阵乘 的公共接口/实现，供所在算子工程复用。 | 函数：`MmCheckHitV3Shape` |
| [maximum.h](../../../csrc/common/stub/op_api/level0/maximum.h) | maximum 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [minimum.h](../../../csrc/common/stub/op_api/level0/minimum.h) | minimum 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [mul.h](../../../csrc/common/stub/op_api/level0/mul.h) | mul 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [muls.h](../../../csrc/common/stub/op_api/level0/muls.h) | muls 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [ones_like.h](../../../csrc/common/stub/op_api/level0/ones_like.h) | ones_like 的公共接口/实现，供所在算子工程复用。 | 函数：`IsAiCpuSupport` |
| [padv3.h](../../../csrc/common/stub/op_api/level0/padv3.h) | padv3 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [reduce_mean.h](../../../csrc/common/stub/op_api/level0/reduce_mean.h) | reduce_mean 的公共接口/实现，供所在算子工程复用。 | 函数：`ReduceMean` |
| [reduce_sum_op.h](../../../csrc/common/stub/op_api/level0/reduce_sum_op.h) | reduce_sum_op 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [shape_op.h](../../../csrc/common/stub/op_api/level0/shape_op.h) | shape_op 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sort.h](../../../csrc/common/stub/op_api/level0/sort.h) | sort 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [squeeze.h](../../../csrc/common/stub/op_api/level0/squeeze.h) | squeeze 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sub.h](../../../csrc/common/stub/op_api/level0/sub.h) | sub 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [tensor_move.h](../../../csrc/common/stub/op_api/level0/tensor_move.h) | 张量 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [unsqueeze.h](../../../csrc/common/stub/op_api/level0/unsqueeze.h) | unsqueeze 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [zero_op.h](../../../csrc/common/stub/op_api/level0/zero_op.h) | zero_op 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/common/stub/op_tiling/`

op_tiling 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 子目录 | 职责 |
| --- | --- |
| [register/](../../../csrc/common/stub/op_tiling/register) | 注册 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/common/stub/op_tiling/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | 构建项：`optiling` |
| [op_cache_def_tiling.h](../../../csrc/common/stub/op_tiling/op_cache_def_tiling.h) | 缓存 的公共接口/实现，供所在算子工程复用。 | 类型：`BatchmatmulCompileParas`, `BatchmatmulRunParas`, `CacheTilingData` |
| [op_cache_tiling.cpp](../../../csrc/common/stub/op_tiling/op_cache_tiling.cpp) | 缓存 的公共接口/实现，供所在算子工程复用。 | 函数：`TilingPrepareForOpCache`, `TilingPrepareForOpCache` |
| [op_cache_tiling.h](../../../csrc/common/stub/op_tiling/op_cache_tiling.h) | 缓存 的公共接口/实现，供所在算子工程复用。 | 函数：`GenTiling` |
| [runtime_kb_api.cpp](../../../csrc/common/stub/op_tiling/runtime_kb_api.cpp) | runtime_kb_api 的公共接口/实现，供所在算子工程复用。 | 函数：`QueryBank` |
| [runtime_kb_api.h](../../../csrc/common/stub/op_tiling/runtime_kb_api.h) | runtime_kb_api 的公共接口/实现，供所在算子工程复用。 | 函数：`QueryBank` |
| [tbe_tiling_api.cpp](../../../csrc/common/stub/op_tiling/tbe_tiling_api.cpp) | tbe_tiling_api 的公共接口/实现，供所在算子工程复用。 | 函数：`GetTbeTiling`, `GetTbeTiling` |
| [tbe_tiling_api.h](../../../csrc/common/stub/op_tiling/tbe_tiling_api.h) | tbe_tiling_api 的公共接口/实现，供所在算子工程复用。 | 类型：`Conv3dBackpropV2TBETilingData`, `Conv3dBpFilterV2RunInfo`, `Conv3dBpInputV2RunInfo` |

## `csrc/common/stub/op_tiling/register/`

注册 子模块；所属范围：各算子复用的类型、宏、注册、主机辅助与基础代码。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [tuning_bank_key_registry.h](../../../csrc/common/stub/op_tiling/register/tuning_bank_key_registry.h) | 注册表 的公共接口/实现，供所在算子工程复用。 | 类型：`FMK_FUNC_HOST_VISIBILITY` |
| [tuning_tiling_reflection_utils.h](../../../csrc/common/stub/op_tiling/register/tuning_tiling_reflection_utils.h) | 辅助函数 的公共接口/实现，供所在算子工程复用。 | 类型：`integer_sequence`, `make_integer_sequence`, `StructInfo` |
| [tuning_tiling_registry.h](../../../csrc/common/stub/op_tiling/register/tuning_tiling_registry.h) | 注册表 的公共接口/实现，供所在算子工程复用。 | 类型：`TilingItem`, `TuningTilingDef`, `class_name` |
