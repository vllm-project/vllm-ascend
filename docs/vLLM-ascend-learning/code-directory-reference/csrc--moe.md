# csrc/moe：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **344 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/moe/`

专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [add_rms_norm_bias/](../../../csrc/moe/add_rms_norm_bias) | 残差相加、RMSNorm 与 bias 算子工程；主机/设备职责以其子目录区分。 |
| [causal_conv1d_v310/](../../../csrc/moe/causal_conv1d_v310) | 310P 因果一维卷积 算子工程；主机/设备职责以其子目录区分。 |
| [chunk_fwd_o_vllm/](../../../csrc/moe/chunk_fwd_o_vllm) | 分块线性注意力输出 算子工程；主机/设备职责以其子目录区分。 |
| [chunk_gated_delta_rule_fwd_h_vllm/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm) | 分块 gated delta 状态前向 算子工程；主机/设备职责以其子目录区分。 |
| [common/](../../../csrc/moe/common) | MoE 及相关设备算子复用的接口和计算辅助。 |
| [copy_and_expand_eagle_inputs/](../../../csrc/moe/copy_and_expand_eagle_inputs) | Eagle 输入复制与展开 算子工程；主机/设备职责以其子目录区分。 |
| [dequant_situ_quant/](../../../csrc/moe/dequant_situ_quant) | 反量化、SITU 与再量化 算子工程；主机/设备职责以其子目录区分。 |
| [dequant_swiglu_quant/](../../../csrc/moe/dequant_swiglu_quant) | 反量化、SwiGLU 与再量化 算子工程；主机/设备职责以其子目录区分。 |
| [grouped_matmul_situ_quant/](../../../csrc/moe/grouped_matmul_situ_quant) | 分组矩阵乘与 SITU 量化 算子工程；主机/设备职责以其子目录区分。 |
| [hc_post/](../../../csrc/moe/hc_post) | mHC 子层后混合 算子工程；主机/设备职责以其子目录区分。 |
| [hc_pre/](../../../csrc/moe/hc_pre) | mHC 子层前混合 算子工程；主机/设备职责以其子目录区分。 |
| [moe_gating_top_k/](../../../csrc/moe/moe_gating_top_k) | MoE 专家 TopK 路由 算子工程；主机/设备职责以其子目录区分。 |
| [moe_gating_top_k_hash/](../../../csrc/moe/moe_gating_top_k_hash) | MoE hash/TopK 路由 算子工程；主机/设备职责以其子目录区分。 |
| [rms_norm_cast/](../../../csrc/moe/rms_norm_cast) | RMSNorm 与类型转换 算子工程；主机/设备职责以其子目录区分。 |
| [scatter_nd_update_sk/](../../../csrc/moe/scatter_nd_update_sk) | 缓存索引写入 算子工程；主机/设备职责以其子目录区分。 |
| [situ_mx_quant/](../../../csrc/moe/situ_mx_quant) | SITU 与 MX 量化 算子工程；主机/设备职责以其子目录区分。 |
| [swiglu_group_quant/](../../../csrc/moe/swiglu_group_quant) | SwiGLU 分组量化 算子工程；主机/设备职责以其子目录区分。 |
| [transpose_kv_cache_by_block/](../../../csrc/moe/transpose_kv_cache_by_block) | 按 block 转置 KV 算子工程；主机/设备职责以其子目录区分。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/add_rms_norm_bias/`

残差相加、RMSNorm 与 bias 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/add_rms_norm_bias/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/add_rms_norm_bias/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/add_rms_norm_bias/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [add_rms_norm_bias_torch_adpt.h](../../../csrc/moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h) | RMS 归一化 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/add_rms_norm_bias/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/add_rms_norm_bias/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [add_rms_norm_bias_def.cpp](../../../csrc/moe/add_rms_norm_bias/op_host/add_rms_norm_bias_def.cpp) | 残差相加、RMSNorm 与 bias 的主机算子定义、支持属性和注册。 | 注册：`AddRmsNormBias` |
| [add_rms_norm_bias_infershape.cpp](../../../csrc/moe/add_rms_norm_bias/op_host/add_rms_norm_bias_infershape.cpp) | 残差相加、RMSNorm 与 bias 的主机侧辅助接口或实现。 | 函数：`InferShape4AddRmsNormBias`, `InferDataType4AddRmsNormBias` |
| [add_rms_norm_bias_tiling.cpp](../../../csrc/moe/add_rms_norm_bias/op_host/add_rms_norm_bias_tiling.cpp) | 残差相加、RMSNorm 与 bias 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`getPerformanceFlag`, `SetByDtype` |
| [add_rms_norm_bias_tiling.h](../../../csrc/moe/add_rms_norm_bias/op_host/add_rms_norm_bias_tiling.h) | 残差相加、RMSNorm 与 bias 的主机侧 tiling 参数/分块策略声明。 | 类型：`AddRmsNormBiasCompileInfo` |
| [add_rms_norm_bias_tiling_arch35.cpp](../../../csrc/moe/add_rms_norm_bias/op_host/add_rms_norm_bias_tiling_arch35.cpp) | 残差相加、RMSNorm 与 bias 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`SameShape`, `ValidateInputs` |

## `csrc/moe/add_rms_norm_bias/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/moe/add_rms_norm_bias/op_kernel/arch35) | arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [add_rms_norm_bias.cpp](../../../csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.cpp) | 残差相加、RMSNorm 与 bias 的设备执行源码；文件主题为 RMS 归一化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [add_rms_norm_bias.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化。 | 类型：`KernelAddRmsNormBias` |
| [add_rms_norm_bias_apt.cpp](../../../csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias_apt.cpp) | 残差相加、RMSNorm 与 bias 的设备执行源码；文件主题为 RMS 归一化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [add_rms_norm_bias_merge_n.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias_merge_n.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、合并。 | 类型：`KernelAddRmsNormBiasMergeN` |
| [add_rms_norm_bias_multi_n.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias_multi_n.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化。 | 类型：`KernelAddRmsNormBiasMultiN` |
| [add_rms_norm_bias_single_n.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias_single_n.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化。 | 类型：`KernelAddRmsNormBiasSingleN` |
| [add_rms_norm_bias_split_d.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias_split_d.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化。 | 类型：`KernelAddRmsNormBiasSplitD` |
| [reduce_common.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/reduce_common.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`ReduceSumForSmallReduceDimPreRepeat`, `ReduceSumForSmallReduceDim` |
| [rms_norm_base.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/rms_norm_base.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化。 | 类型：`integral_constant`, `is_same`, `KernelRmsNormBase` |

## `csrc/moe/add_rms_norm_bias/op_kernel/arch35/`

arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [deps/](../../../csrc/moe/add_rms_norm_bias/op_kernel/arch35/deps) | deps 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [add_rms_norm_bias_regbase.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/arch35/add_rms_norm_bias_regbase.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化。 | 类型：`KernelAddRmsNormBiasRegBase` |
| [add_rms_norm_bias_regbase_common.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/arch35/add_rms_norm_bias_regbase_common.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、公共逻辑。 | 函数：`LoadForHandleRemainV1`, `LoadForHandleMasterV1` |
| [add_rms_norm_bias_regbase_split_d.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/arch35/add_rms_norm_bias_regbase_split_d.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化。 | 类型：`KernelAddRmsNormBiasRegBaseSplitD` |

## `csrc/moe/add_rms_norm_bias/op_kernel/arch35/deps/`

deps 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [reduce_common_regbase.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/arch35/deps/reduce_common_regbase.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`GetVRegSize`, `CeilDiv` |
| [rms_norm_regbase_common.h](../../../csrc/moe/add_rms_norm_bias/op_kernel/arch35/deps/rms_norm_regbase_common.h) | 残差相加、RMSNorm 与 bias 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、公共逻辑。 | 函数：`GetOverflowMode`, `SetOverflowMode` |

## `csrc/moe/causal_conv1d_v310/`

310P 因果一维卷积 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/causal_conv1d_v310/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/causal_conv1d_v310/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/causal_conv1d_v310/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [causal_conv1d_310_torch_adpt.h](../../../csrc/moe/causal_conv1d_v310/causal_conv1d_310_torch_adpt.h) | 卷积 的公共接口/实现，供所在算子工程复用。 | 函数：`npu_causal_conv1d_310` |

## `csrc/moe/causal_conv1d_v310/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/causal_conv1d_v310/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [causal_conv1d_v310_def.cpp](../../../csrc/moe/causal_conv1d_v310/op_host/causal_conv1d_v310_def.cpp) | 310P 因果一维卷积 的主机算子定义、支持属性和注册。 | 注册：`CausalConv1dV310` |
| [causal_conv1d_v310_infershape.cpp](../../../csrc/moe/causal_conv1d_v310/op_host/causal_conv1d_v310_infershape.cpp) | 310P 因果一维卷积 的主机侧辅助接口或实现。 | 函数：`InferShapeCausalConv1d` |
| [causal_conv1d_v310_tiling.cpp](../../../csrc/moe/causal_conv1d_v310/op_host/causal_conv1d_v310_tiling.cpp) | 310P 因果一维卷积 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`CausalConv1dCompileInfo`, `DimTileChoice` |
| [causal_conv1d_v310_tiling.h](../../../csrc/moe/causal_conv1d_v310/op_host/causal_conv1d_v310_tiling.h) | 310P 因果一维卷积 的主机侧 tiling 参数/分块策略声明。 | 类型：`CausalConv1dCompileInfo` |
| [math_util.h](../../../csrc/moe/causal_conv1d_v310/op_host/math_util.h) | 310P 因果一维卷积 的主机侧辅助接口或实现。 | 类型：`MathUtil` |

## `csrc/moe/causal_conv1d_v310/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [causal_conv1d_v310.cpp](../../../csrc/moe/causal_conv1d_v310/op_kernel/causal_conv1d_v310.cpp) | 310P 因果一维卷积 的设备执行源码；文件主题为 卷积。 | 函数：`RunCausalConv1d`, `causal_conv1d_v310` |
| [causal_conv1d_v310.h](../../../csrc/moe/causal_conv1d_v310/op_kernel/causal_conv1d_v310.h) | 310P 因果一维卷积 的类型/模板与设备实现头文件；文件主题为 卷积。 | 类型：`CausalConv1dV310` |
| [causal_conv1d_v310_common.h](../../../csrc/moe/causal_conv1d_v310/op_kernel/causal_conv1d_v310_common.h) | 310P 因果一维卷积 的类型/模板与设备实现头文件；文件主题为 公共逻辑、卷积。 | 函数：`SlotCurr`, `SlotHist` |
| [causal_conv1d_v310_tiling_data.h](../../../csrc/moe/causal_conv1d_v310/op_kernel/causal_conv1d_v310_tiling_data.h) | 310P 因果一维卷积 的类型/模板与设备实现头文件；文件主题为 卷积。 | 类型：`CausalConv1dTilingData` |
| [causal_conv1d_v310_tiling_key.h](../../../csrc/moe/causal_conv1d_v310/op_kernel/causal_conv1d_v310_tiling_key.h) | 310P 因果一维卷积 的kernel 分派使用的 tiling key 定义；文件主题为 卷积。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/chunk_fwd_o_vllm/`

分块线性注意力输出 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/chunk_fwd_o_vllm/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |
| [tiling_base/](../../../csrc/moe/chunk_fwd_o_vllm/tiling_base) | tiling_base 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/chunk_fwd_o_vllm/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/chunk_fwd_o_vllm/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/moe/chunk_fwd_o_vllm/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/chunk_fwd_o_vllm/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [chunk_fwd_o_vllm_def.cpp](../../../csrc/moe/chunk_fwd_o_vllm/op_host/chunk_fwd_o_vllm_def.cpp) | 分块线性注意力输出 的主机算子定义、支持属性和注册。 | 注册：`ChunkFwdOVllm` |
| [chunk_fwd_o_vllm_tiling.cpp](../../../csrc/moe/chunk_fwd_o_vllm/op_host/chunk_fwd_o_vllm_tiling.cpp) | 分块线性注意力输出 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`ChunkFwdOVllmTilingDataPrint`, `Tiling4ChunkFwdOVllm` |
| [chunk_fwd_o_vllm_tiling.h](../../../csrc/moe/chunk_fwd_o_vllm/op_host/chunk_fwd_o_vllm_tiling.h) | 分块线性注意力输出 的主机侧 tiling 参数/分块策略声明。 | 类型：`ChunkFwdOVllmCompileInfo` |

## `csrc/moe/chunk_fwd_o_vllm/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_chunk_fwd_o_vllm.cpp](../../../csrc/moe/chunk_fwd_o_vllm/op_host/op_api/aclnn_chunk_fwd_o_vllm.cpp) | 分块线性注意力输出 的主机侧辅助接口或实现。 | 类型：`ChunkFwdOVllmParams` |
| [aclnn_chunk_fwd_o_vllm.h](../../../csrc/moe/chunk_fwd_o_vllm/op_host/op_api/aclnn_chunk_fwd_o_vllm.h) | 分块线性注意力输出 的主机侧辅助接口或实现。 | 函数：`aclnnChunkFwdOVllmGetWorkspaceSize`, `aclnnChunkFwdOVllm` |
| [chunk_fwd_o_vllm.cpp](../../../csrc/moe/chunk_fwd_o_vllm/op_host/op_api/chunk_fwd_o_vllm.cpp) | 分块线性注意力输出 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [chunk_fwd_o_vllm.h](../../../csrc/moe/chunk_fwd_o_vllm/op_host/op_api/chunk_fwd_o_vllm.h) | 分块线性注意力输出 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch20/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20) | arch20 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [arch22/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22) | arch22 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [chunk_fwd_o_vllm.cpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/chunk_fwd_o_vllm.cpp) | 分块线性注意力输出 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/`

arch20 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [epilogue/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/epilogue) | epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [gemm/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/gemm) | gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compat_310p.h](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/compat_310p.h) | 分块线性注意力输出 的类型/模板与设备实现头文件。 | 类型：`bfloat16_t` |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/epilogue/`

epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/epilogue/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_o_epilogue_policies.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/epilogue/gdn_fwd_o_epilogue_policies.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`EpilogueAtlasGDNFwdOQkmask`, `EpilogueAtlasGDNFwdOOutput` |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/epilogue/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue_gdn_fwdo_output.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/epilogue/block/block_epilogue_gdn_fwdo_output.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 输出、缓存页/分块、Gated DeltaNet 注意力。 | 类型：`HOutputType_`, `GInputType_`, `AInputType_` |
| [block_epilogue_gdn_fwdo_qkmask.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`AOutputType_`, `GInputType_`, `AInputType_` |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/gemm/`

gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/gemm/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [kernel/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/gemm/kernel) | kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/gemm/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_scheduler_gdn_fwd_o.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/gemm/block/block_scheduler_gdn_fwd_o.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 调度、缓存页/分块、Gated DeltaNet 注意力。 | 类型：`GDNFwdOOffsets`, `BlockSchedulerGdnFwdO`, `BlockSchedulerGdnFwdOCube` |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/gemm/kernel/`

kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_o_kernel.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20/gemm/kernel/gdn_fwd_o_kernel.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`GDNFwdOKernel`, `uses` |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/`

arch22 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [epilogue/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/epilogue) | epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [gemm/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/gemm) | gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/epilogue/`

epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/epilogue/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_o_epilogue_policies.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/epilogue/gdn_fwd_o_epilogue_policies.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`EpilogueAtlasGDNFwdOQkmask`, `EpilogueAtlasGDNFwdOOutput` |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/epilogue/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue_gdn_fwdo_output.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/epilogue/block/block_epilogue_gdn_fwdo_output.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 输出、缓存页/分块、Gated DeltaNet 注意力。 | 类型：`HOutputType_`, `GInputType_`, `AInputType_` |
| [block_epilogue_gdn_fwdo_qkmask.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`AOutputType_`, `GInputType_`, `AInputType_` |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/gemm/`

gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/gemm/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [kernel/](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/gemm/kernel) | kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/gemm/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_scheduler_gdn_fwd_o.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/gemm/block/block_scheduler_gdn_fwd_o.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 调度、缓存页/分块、Gated DeltaNet 注意力。 | 类型：`GDNFwdOOffsets`, `BlockSchedulerGdnFwdO`, `BlockSchedulerGdnFwdOCube` |

## `csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/gemm/kernel/`

kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_o_kernel.hpp](../../../csrc/moe/chunk_fwd_o_vllm/op_kernel/arch22/gemm/kernel/gdn_fwd_o_kernel.hpp) | 分块线性注意力输出 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`GDNFwdOKernel` |

## `csrc/moe/chunk_fwd_o_vllm/tiling_base/`

tiling_base 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [data_copy_transpose_tiling.h](../../../csrc/moe/chunk_fwd_o_vllm/tiling_base/data_copy_transpose_tiling.h) | 转置、复制 的公共接口/实现，供所在算子工程复用。 | 函数：`GetDataCopyTransposeTiling` |
| [data_copy_transpose_tiling_def.h](../../../csrc/moe/chunk_fwd_o_vllm/tiling_base/data_copy_transpose_tiling_def.h) | 转置、复制 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [error_log.h](../../../csrc/moe/chunk_fwd_o_vllm/tiling_base/error_log.h) | 日志 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [tiling_base.h](../../../csrc/moe/chunk_fwd_o_vllm/tiling_base/tiling_base.h) | tiling_base 的公共接口/实现，供所在算子工程复用。 | 类型：`AiCoreParams`, `CompileInfoCommon`, `FlashAttentionScoreGradCompileInfo` |
| [tiling_key.h](../../../csrc/moe/chunk_fwd_o_vllm/tiling_base/tiling_key.h) | tiling_key 的公共接口/实现，供所在算子工程复用。 | 函数：`RecursiveSum` |
| [tiling_templates_registry.h](../../../csrc/moe/chunk_fwd_o_vllm/tiling_base/tiling_templates_registry.h) | 注册表 的公共接口/实现，供所在算子工程复用。 | 类型：`TilingCases`, `TilingRegistryNew`, `RegisterNew` |
| [tiling_type.h](../../../csrc/moe/chunk_fwd_o_vllm/tiling_base/tiling_type.h) | tiling_type 的公共接口/实现，供所在算子工程复用。 | 类型：`AxisEnum`, `DtypeEnum`, `PerformanceOrientedEnum` |
| [tiling_util.h](../../../csrc/moe/chunk_fwd_o_vllm/tiling_base/tiling_util.h) | tiling_util 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/`

分块 gated delta 状态前向 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |
| [tiling_base/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base) | tiling_base 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [chunk_gated_delta_rule_fwd_h_vllm_def.cpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/chunk_gated_delta_rule_fwd_h_vllm_def.cpp) | 分块 gated delta 状态前向 的主机算子定义、支持属性和注册。 | 注册：`ChunkGatedDeltaRuleFwdHVllm` |
| [chunk_gated_delta_rule_fwd_h_vllm_tiling.cpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/chunk_gated_delta_rule_fwd_h_vllm_tiling.cpp) | 分块 gated delta 状态前向 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`GdnFwdHDtypeToEnum`, `ChunkGatedDeltaRuleFwdHVllmTilingDataPrint` |
| [chunk_gated_delta_rule_fwd_h_vllm_tiling.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/chunk_gated_delta_rule_fwd_h_vllm_tiling.h) | 分块 gated delta 状态前向 的主机侧 tiling 参数/分块策略声明。 | 类型：`ChunkGatedDeltaRuleFwdHVllmCompileInfo` |
| [chunk_gated_delta_rule_fwd_h_vllm_tiling_processor.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/chunk_gated_delta_rule_fwd_h_vllm_tiling_processor.h) | 分块 gated delta 状态前向 的主机侧 tiling 参数/分块策略声明。 | 类型：`below`, `ChunkGatedDeltaRuleFwdHVllmTilingContext`, `ChunkGatedDeltaRuleFwdHVllmTilingProcessor` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_chunk_gated_delta_rule_fwd_h_vllm.cpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd_h_vllm.cpp) | 分块 gated delta 状态前向 的主机侧辅助接口或实现。 | 类型：`ChunkGatedDeltaRuleFwdHVllmParams` |
| [aclnn_chunk_gated_delta_rule_fwd_h_vllm.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd_h_vllm.h) | 分块 gated delta 状态前向 的主机侧辅助接口或实现。 | 函数：`aclnnChunkGatedDeltaRuleFwdHVllmGetWorkspaceSize`, `aclnnChunkGatedDeltaRuleFwdHVllm` |
| [chunk_gated_delta_rule_fwd_h_vllm.cpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/op_api/chunk_gated_delta_rule_fwd_h_vllm.cpp) | 分块 gated delta 状态前向 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [chunk_gated_delta_rule_fwd_h_vllm.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_host/op_api/chunk_gated_delta_rule_fwd_h_vllm.h) | 分块 gated delta 状态前向 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch20/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20) | arch20 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [arch22/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22) | arch22 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [arch35/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35) | arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [epilogue/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/epilogue) | epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [gemm/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/gemm) | gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [chunk_gated_delta_rule_fwd_h_vllm.cpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/chunk_gated_delta_rule_fwd_h_vllm.cpp) | 分块 gated delta 状态前向 的设备执行源码；文件主题为 Delta 递归状态、门控。 | 函数：`ChunkGatedDeltaRuleFwdHVllmKernelImpl`, `ChunkGatedDeltaRuleFwdHVllmDispatch` |
| [chunk_gated_delta_rule_fwd_h_vllm_struct.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/chunk_gated_delta_rule_fwd_h_vllm_struct.h) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 Delta 递归状态、门控。 | 类型：`for`, `named`, `here` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/`

arch20 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [epilogue/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/epilogue) | epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [gemm/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/gemm) | gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compat_310p.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/compat_310p.h) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件。 | 类型：`bfloat16_t` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/epilogue/`

epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/epilogue/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_h_epilogue_policies.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/epilogue/gdn_fwd_h_epilogue_policies.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`EpilogueAtlasGDNFwdHVnew`, `EpilogueAtlasGDNFwdHUpdate` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/epilogue/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue_gdn_fwdh_update.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/epilogue/block/block_epilogue_gdn_fwdh_update.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`HOutputType_`, `GInputType_`, `HInputType_` |
| [block_epilogue_gdn_fwdh_vnew.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`VOutputType_`, `GInputType_`, `UInputType_` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/gemm/`

gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/gemm/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [kernel/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/gemm/kernel) | kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/gemm/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_scheduler_gdn_fwd_h.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/gemm/block/block_scheduler_gdn_fwd_h.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 调度、缓存页/分块、Gated DeltaNet 注意力。 | 类型：`GDNFwdHOffsets`, `BlockSchedulerGdnFwdH`, `BlockSchedulerGdnFwdHCube` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/gemm/kernel/`

kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_h_kernel.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch20/gemm/kernel/gdn_fwd_h_kernel.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`GDNFwdHKernel` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/`

arch22 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [epilogue/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/epilogue) | epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [gemm/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/gemm) | gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/epilogue/`

epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/epilogue/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_h_epilogue_policies.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/epilogue/gdn_fwd_h_epilogue_policies.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`EpilogueAtlasGDNFwdHVnew`, `EpilogueAtlasGDNFwdHUpdate` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/epilogue/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue_gdn_fwdh_update.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/epilogue/block/block_epilogue_gdn_fwdh_update.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`HOutputType_`, `GInputType_`, `HInputType_` |
| [block_epilogue_gdn_fwdh_vnew.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`VOutputType_`, `GInputType_`, `UInputType_` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/gemm/`

gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/gemm/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [kernel/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/gemm/kernel) | kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/gemm/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_scheduler_gdn_fwd_h.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/gemm/block/block_scheduler_gdn_fwd_h.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 调度、缓存页/分块、Gated DeltaNet 注意力。 | 类型：`GDNFwdHOffsets`, `GDNFwdHStream`, `GDNFwdHRunningQ` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/gemm/kernel/`

kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_h_kernel.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch22/gemm/kernel/gdn_fwd_h_kernel.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`GDNFwdHTileShapes128`, `GDNFwdHTileShapes256`, `GDNFwdHKernel` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/`

arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [epilogue/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/epilogue) | epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [gemm/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/gemm) | gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/epilogue/`

epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/epilogue/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_h_epilogue_policies.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/epilogue/gdn_fwd_h_epilogue_policies.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`EpilogueAtlasGDNFwdHVnew`, `EpilogueAtlasGDNFwdHUpdate` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/epilogue/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue_gdn_fwdh_regbase.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_regbase.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 函数：`LoadKdaAsFloat`, `LoadKdaFloat` |
| [block_epilogue_gdn_fwdh_update.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_update.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`HOutputType_`, `GInputType_`, `HInputType_` |
| [block_epilogue_gdn_fwdh_vnew.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`VOutputType_`, `GInputType_`, `UInputType_` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/gemm/`

gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/gemm/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [kernel/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/gemm/kernel) | kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/gemm/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_scheduler_gdn_fwd_h.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/gemm/block/block_scheduler_gdn_fwd_h.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 调度、缓存页/分块、Gated DeltaNet 注意力。 | 类型：`GDNFwdHOffsets`, `GDNFwdHStream`, `GDNFwdHRunningQ` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/gemm/kernel/`

kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_h_kernel.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`GDNFwdHTileShapes128`, `GDNFwdHTileShapes256`, `GDNFwdHGateTag` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/epilogue/`

epilogue 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/epilogue/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_h_epilogue_policies.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/epilogue/gdn_fwd_h_epilogue_policies.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`EpilogueAtlasGDNFwdHVnew`, `EpilogueAtlasGDNFwdHUpdate` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/epilogue/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue_gdn_fwdh_update.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/epilogue/block/block_epilogue_gdn_fwdh_update.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`HOutputType_`, `GInputType_`, `HInputType_` |
| [block_epilogue_gdn_fwdh_vnew.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、Gated DeltaNet 注意力。 | 类型：`VOutputType_`, `GInputType_`, `UInputType_` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/gemm/`

gemm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/gemm/block) | 缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [kernel/](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/gemm/kernel) | kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/gemm/block/`

缓存页/分块 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_scheduler_gdn_fwd_h.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/gemm/block/block_scheduler_gdn_fwd_h.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 调度、缓存页/分块、Gated DeltaNet 注意力。 | 类型：`GDNFwdHOffsets`, `GDNFwdHStream`, `GDNFwdHRunningQ` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/gemm/kernel/`

kernel 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gdn_fwd_h_kernel.hpp](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp) | 分块 gated delta 状态前向 的类型/模板与设备实现头文件；文件主题为 Gated DeltaNet 注意力。 | 类型：`GDNFwdHTileShapes128`, `GDNFwdHTileShapes256`, `GDNFwdHGateTag` |

## `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base/`

tiling_base 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [data_copy_transpose_tiling.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base/data_copy_transpose_tiling.h) | 转置、复制 的公共接口/实现，供所在算子工程复用。 | 函数：`GetDataCopyTransposeTiling` |
| [data_copy_transpose_tiling_def.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base/data_copy_transpose_tiling_def.h) | 转置、复制 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [error_log.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base/error_log.h) | 日志 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [tiling_base.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base/tiling_base.h) | tiling_base 的公共接口/实现，供所在算子工程复用。 | 类型：`AiCoreParams`, `CompileInfoCommon`, `FlashAttentionScoreGradCompileInfo` |
| [tiling_key.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base/tiling_key.h) | tiling_key 的公共接口/实现，供所在算子工程复用。 | 函数：`RecursiveSum` |
| [tiling_templates_registry.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base/tiling_templates_registry.h) | 注册表 的公共接口/实现，供所在算子工程复用。 | 类型：`TilingCases`, `TilingRegistryNew`, `RegisterNew` |
| [tiling_type.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base/tiling_type.h) | tiling_type 的公共接口/实现，供所在算子工程复用。 | 类型：`AxisEnum`, `DtypeEnum`, `PerformanceOrientedEnum` |
| [tiling_util.h](../../../csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/tiling_base/tiling_util.h) | tiling_util 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/common/`

MoE 及相关设备算子复用的接口和计算辅助。

| 子目录 | 职责 |
| --- | --- |
| [kernel_utils/](../../../csrc/moe/common/kernel_utils) | 辅助函数 子模块；所属范围：MoE 及相关设备算子复用的接口和计算辅助。 |

## `csrc/moe/common/kernel_utils/`

辅助函数 子模块；所属范围：MoE 及相关设备算子复用的接口和计算辅助。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/moe/common/kernel_utils/block) | 缓存页/分块 子模块；所属范围：MoE 及相关设备算子复用的接口和计算辅助。 |
| [tile/](../../../csrc/moe/common/kernel_utils/tile) | tile 子模块；所属范围：MoE 及相关设备算子复用的接口和计算辅助。 |
| [vector/](../../../csrc/moe/common/kernel_utils/vector) | vector 子模块；所属范围：MoE 及相关设备算子复用的接口和计算辅助。 |

## `csrc/moe/common/kernel_utils/block/`

缓存页/分块 子模块；所属范围：MoE 及相关设备算子复用的接口和计算辅助。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_mmad_pingpong_tla.hpp](../../../csrc/moe/common/kernel_utils/block/block_mmad_pingpong_tla.hpp) | 缓存页/分块 的公共接口/实现，供所在算子工程复用。 | 类型：`DispatchPolicy`, `L1TileShape`, `L0TileShape` |
| [block_mmad_pingpong_tla_multi.hpp](../../../csrc/moe/common/kernel_utils/block/block_mmad_pingpong_tla_multi.hpp) | 缓存页/分块 的公共接口/实现，供所在算子工程复用。 | 类型：`ArchTag_`, `MmadPingpongTlaMulti`, `L1TileShape_` |
| [block_mmad_pingpong_tla_preloadA_l1B.hpp](../../../csrc/moe/common/kernel_utils/block/block_mmad_pingpong_tla_preloadA_l1B.hpp) | 缓存页/分块、加载 的公共接口/实现，供所在算子工程复用。 | 类型：`ArchTag_`, `MmadPingpongTlaPreloadAL1B`, `L1TileShape_` |

## `csrc/moe/common/kernel_utils/tile/`

tile 子模块；所属范围：MoE 及相关设备算子复用的接口和计算辅助。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [copy_l0c_to_ub.hpp](../../../csrc/moe/common/kernel_utils/tile/copy_l0c_to_ub.hpp) | 复制 的公共接口/实现，供所在算子工程复用。 | 类型：`ArchTag`, `TensorSrc`, `TensorDst` |

## `csrc/moe/common/kernel_utils/vector/`

vector 子模块；所属范围：MoE 及相关设备算子复用的接口和计算辅助。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [regbase.hpp](../../../csrc/moe/common/kernel_utils/vector/regbase.hpp) | regbase 的公共接口/实现，供所在算子工程复用。 | 函数：`LoadIn`, `CastHalf2Float` |

## `csrc/moe/copy_and_expand_eagle_inputs/`

Eagle 输入复制与展开 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/copy_and_expand_eagle_inputs/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/copy_and_expand_eagle_inputs/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/copy_and_expand_eagle_inputs/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/copy_and_expand_eagle_inputs/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/copy_and_expand_eagle_inputs/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [copy_and_expand_eagle_inputs_def.cpp](../../../csrc/moe/copy_and_expand_eagle_inputs/op_host/copy_and_expand_eagle_inputs_def.cpp) | Eagle 输入复制与展开 的主机算子定义、支持属性和注册。 | 注册：`CopyAndExpandEagleInputs` |
| [copy_and_expand_eagle_inputs_infershape.cpp](../../../csrc/moe/copy_and_expand_eagle_inputs/op_host/copy_and_expand_eagle_inputs_infershape.cpp) | Eagle 输入复制与展开 的主机侧辅助接口或实现。 | 函数：`InferShape4CopyAndExpandEagleInputs`, `InferDataType4CopyAndExpandEagleInputs` |
| [copy_and_expand_eagle_inputs_tiling.cpp](../../../csrc/moe/copy_and_expand_eagle_inputs/op_host/copy_and_expand_eagle_inputs_tiling.cpp) | Eagle 输入复制与展开 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`GetCompileParameters`, `TilingFunc` |
| [copy_and_expand_eagle_inputs_tiling.h](../../../csrc/moe/copy_and_expand_eagle_inputs/op_host/copy_and_expand_eagle_inputs_tiling.h) | Eagle 输入复制与展开 的主机侧 tiling 参数/分块策略声明。 | 类型：`CopyAndExpandEagleInputsCompileInfo` |

## `csrc/moe/copy_and_expand_eagle_inputs/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [copy_and_expand_eagle_inputs.cpp](../../../csrc/moe/copy_and_expand_eagle_inputs/op_kernel/copy_and_expand_eagle_inputs.cpp) | Eagle 输入复制与展开 的设备执行源码；文件主题为 Eagle 投机、输入、复制。 | 类型：`CopyAndExpandEagleInputsKernel` |

## `csrc/moe/dequant_situ_quant/`

反量化、SITU 与再量化 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/moe/dequant_situ_quant/docs) | docs 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [op_graph/](../../../csrc/moe/dequant_situ_quant/op_graph) | 设备图 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [op_host/](../../../csrc/moe/dequant_situ_quant/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/dequant_situ_quant/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/dequant_situ_quant/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [dequant_situ_quant_torch_adpt.h](../../../csrc/moe/dequant_situ_quant/dequant_situ_quant_torch_adpt.h) | 反量化、量化、SITU 激活 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/dequant_situ_quant/docs/`

docs 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnnDequantSituQuant.md](../../../csrc/moe/dequant_situ_quant/docs/aclnnDequantSituQuant.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：aclnnDequantSituQuant。 | 标题：aclnnDequantSituQuant |

## `csrc/moe/dequant_situ_quant/op_graph/`

设备图 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [dequant_situ_quant_proto.h](../../../csrc/moe/dequant_situ_quant/op_graph/dequant_situ_quant_proto.h) | 反量化、量化、SITU 激活 的公共接口/实现，供所在算子工程复用。 | 注册：`DequantSituQuant` |

## `csrc/moe/dequant_situ_quant/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/dequant_situ_quant/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [dequant_situ_quant_def.cpp](../../../csrc/moe/dequant_situ_quant/op_host/dequant_situ_quant_def.cpp) | 反量化、SITU 与再量化 的主机算子定义、支持属性和注册。 | 注册：`DequantSituQuant` |
| [dequant_situ_quant_infershape.cpp](../../../csrc/moe/dequant_situ_quant/op_host/dequant_situ_quant_infershape.cpp) | 反量化、SITU 与再量化 的主机侧辅助接口或实现。 | 函数：`InferShape4DequantSituQuant`, `InferDtype4DequantSituQuant` |
| [dequant_situ_quant_tiling.cpp](../../../csrc/moe/dequant_situ_quant/op_host/dequant_situ_quant_tiling.cpp) | 反量化、SITU 与再量化 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`TilingForDequantSituQuant`, `TilingPrepareForDequantSituQuant` |
| [dequant_situ_quant_tiling.h](../../../csrc/moe/dequant_situ_quant/op_host/dequant_situ_quant_tiling.h) | 反量化、SITU 与再量化 的主机侧 tiling 参数/分块策略声明。 | 类型：`DequantSituQuantCompileInfo`, `DequantSituQuantTiling` |

## `csrc/moe/dequant_situ_quant/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [dequant_situ_quant.cpp](../../../csrc/moe/dequant_situ_quant/op_kernel/dequant_situ_quant.cpp) | 反量化、SITU 与再量化 的设备执行源码；文件主题为 反量化、量化、SITU 激活。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dequant_situ_quant.h](../../../csrc/moe/dequant_situ_quant/op_kernel/dequant_situ_quant.h) | 反量化、SITU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、量化、SITU 激活。 | 类型：`DequantSituQuantKernel`, `DequantSituQuantK3Kernel` |

## `csrc/moe/dequant_swiglu_quant/`

反量化、SwiGLU 与再量化 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/dequant_swiglu_quant/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/dequant_swiglu_quant/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |
| [tiling_base/](../../../csrc/moe/dequant_swiglu_quant/tiling_base) | tiling_base 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/dequant_swiglu_quant/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/dequant_swiglu_quant/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/dequant_swiglu_quant/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [dequant_swiglu_quant_def.cpp](../../../csrc/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_def.cpp) | 反量化、SwiGLU 与再量化 的主机算子定义、支持属性和注册。 | 注册：`DequantSwigluQuant` |
| [dequant_swiglu_quant_infershape.cpp](../../../csrc/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_infershape.cpp) | 反量化、SwiGLU 与再量化 的主机侧辅助接口或实现。 | 函数：`InferShape4DequantSwigluQuant`, `InferDtype4DequantSwigluQuant` |
| [dequant_swiglu_quant_proto.h](../../../csrc/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_proto.h) | 反量化、SwiGLU 与再量化 的算子原型、输入输出或类型/shape 推断注册。 | 注册：`DequantSwigluQuant` |
| [dequant_swiglu_quant_tiling.cpp](../../../csrc/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling.cpp) | 反量化、SwiGLU 与再量化 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`TilingForDequantSwigluQuant`, `TilingPrepareForDequantSwigluQuant` |
| [dequant_swiglu_quant_tiling.h](../../../csrc/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling.h) | 反量化、SwiGLU 与再量化 的主机侧 tiling 参数/分块策略声明。 | 类型：`DequantSwigluQuantCompileInfo`, `DequantSwigluQuantDskTiling`, `DequantSwigluQuantV35DskTiling` |
| [dequant_swiglu_quant_tiling_arch35.cpp](../../../csrc/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp) | 反量化、SwiGLU 与再量化 的主机侧 tiling 参数/分块策略计算实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dequant_swiglu_quant_tiling_base.cpp](../../../csrc/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_base.cpp) | 反量化、SwiGLU 与再量化 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`GluSingleTilingOptParam`, `DequantSwigluQuantTiling` |
| [swi_glu_grad_regbase_tiling.h](../../../csrc/moe/dequant_swiglu_quant/op_host/swi_glu_grad_regbase_tiling.h) | 反量化、SwiGLU 与再量化 的主机侧 tiling 参数/分块策略声明。 | 类型：`GluBaseTilingData` |
| [swi_glu_grad_tiling_regbase.h](../../../csrc/moe/dequant_swiglu_quant/op_host/swi_glu_grad_tiling_regbase.h) | 反量化、SwiGLU 与再量化 的主机侧 tiling 参数/分块策略声明。 | 类型：`GluBaseTiling4RegBase` |
| [swi_glu_tiling.h](../../../csrc/moe/dequant_swiglu_quant/op_host/swi_glu_tiling.h) | 反量化、SwiGLU 与再量化 的主机侧 tiling 参数/分块策略声明。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/dequant_swiglu_quant/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [dequant_swiglu_quant.cpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.cpp) | 反量化、SwiGLU 与再量化 的设备执行源码；文件主题为 反量化、SwiGLU 激活、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dequant_swiglu_quant.h](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantBase` |
| [dequant_swiglu_quant_apt.cpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_apt.cpp) | 反量化、SwiGLU 与再量化 的设备执行源码；文件主题为 反量化、SwiGLU 激活、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [dequant_swiglu_quant_cut_group.h](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_cut_group.h) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantGroup` |
| [dequant_swiglu_quant_dynamic_base.hpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_dynamic_base.hpp) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantDynamicBase` |
| [dequant_swiglu_quant_dynamic_bf16.hpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_dynamic_bf16.hpp) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantDynamicBF16` |
| [dequant_swiglu_quant_dynamic_bias_float.hpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_dynamic_bias_float.hpp) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantDynamicBiasFloat` |
| [dequant_swiglu_quant_dynamic_bias_int32.hpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_dynamic_bias_int32.hpp) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantDynamicBiasInt32` |
| [dequant_swiglu_quant_dynamic_performance.hpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_dynamic_performance.hpp) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantDynamicPerformance` |
| [dequant_swiglu_quant_static_base.hpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_static_base.hpp) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantStaticBase` |
| [dequant_swiglu_quant_static_bf16.hpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_static_bf16.hpp) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantStaticBF16` |
| [dequant_swiglu_quant_static_bias_float.hpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_static_bias_float.hpp) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantStaticBiasFloat` |
| [dequant_swiglu_quant_static_bias_int32.hpp](../../../csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_static_bias_int32.hpp) | 反量化、SwiGLU 与再量化 的类型/模板与设备实现头文件；文件主题为 反量化、SwiGLU 激活、量化。 | 类型：`DequantSwigluQuantStaticBiasInt32` |

## `csrc/moe/dequant_swiglu_quant/tiling_base/`

tiling_base 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [error_log.h](../../../csrc/moe/dequant_swiglu_quant/tiling_base/error_log.h) | 日志 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [static_register_symbol.h](../../../csrc/moe/dequant_swiglu_quant/tiling_base/static_register_symbol.h) | 注册 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [tiling_base.h](../../../csrc/moe/dequant_swiglu_quant/tiling_base/tiling_base.h) | tiling_base 的公共接口/实现，供所在算子工程复用。 | 类型：`AiCoreParams`, `CompileInfoCommon`, `TilingBaseClass` |
| [tiling_key.h](../../../csrc/moe/dequant_swiglu_quant/tiling_base/tiling_key.h) | tiling_key 的公共接口/实现，供所在算子工程复用。 | 函数：`RecursiveSum` |
| [tiling_templates_registry.h](../../../csrc/moe/dequant_swiglu_quant/tiling_base/tiling_templates_registry.h) | 注册表 的公共接口/实现，供所在算子工程复用。 | 类型：`TilingCases`, `TilingRegistryArch`, `RegisterArch` |
| [tiling_util.h](../../../csrc/moe/dequant_swiglu_quant/tiling_base/tiling_util.h) | tiling_util 的公共接口/实现，供所在算子工程复用。 | 函数：`IsRegbaseNpuArch`, `IsRegbaseSocVersion` |

## `csrc/moe/grouped_matmul_situ_quant/`

分组矩阵乘与 SITU 量化 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/moe/grouped_matmul_situ_quant/docs) | docs 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [op_host/](../../../csrc/moe/grouped_matmul_situ_quant/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |
| [utils/](../../../csrc/moe/grouped_matmul_situ_quant/utils) | 辅助函数 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/grouped_matmul_situ_quant/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/moe/grouped_matmul_situ_quant/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：GroupedMatmulSituQuant |
| [grouped_matmul_situ_quant_torch_adpt.h](../../../csrc/moe/grouped_matmul_situ_quant/grouped_matmul_situ_quant_torch_adpt.h) | 矩阵乘、量化、SITU 激活 的公共接口/实现，供所在算子工程复用。 | 函数：`IsNullOrUndefined`, `ValidateV2Params` |
| [ops.h](../../../csrc/moe/grouped_matmul_situ_quant/ops.h) | ops 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/grouped_matmul_situ_quant/docs/`

docs 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnnGroupedMatmulSituQuant.md](../../../csrc/moe/grouped_matmul_situ_quant/docs/aclnnGroupedMatmulSituQuant.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：aclnnGroupedMatmulSituQuant。 | 标题：aclnnGroupedMatmulSituQuant |

## `csrc/moe/grouped_matmul_situ_quant/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/grouped_matmul_situ_quant/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [grouped_matmul_situ_quant_def.cpp](../../../csrc/moe/grouped_matmul_situ_quant/op_host/grouped_matmul_situ_quant_def.cpp) | 分组矩阵乘与 SITU 量化 的主机算子定义、支持属性和注册。 | 注册：`GroupedMatmulSituQuant` |
| [grouped_matmul_situ_quant_infershape.cpp](../../../csrc/moe/grouped_matmul_situ_quant/op_host/grouped_matmul_situ_quant_infershape.cpp) | 分组矩阵乘与 SITU 量化 的主机侧辅助接口或实现。 | 函数：`InferShape4GroupedMatmulSituQuant`, `InferDataType4GroupedMatmulSituQuant` |
| [grouped_matmul_situ_quant_tiling.cpp](../../../csrc/moe/grouped_matmul_situ_quant/op_host/grouped_matmul_situ_quant_tiling.cpp) | 分组矩阵乘与 SITU 量化 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`GroupedMatmulSituQuantCompileInfo` |

## `csrc/moe/grouped_matmul_situ_quant/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [vendor/](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor) | 该路径复用的算子/矩阵乘辅助实现；具体来源与许可见文件头。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gmsq_vcv_controller.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/gmsq_vcv_controller.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件。 | 类型：`SituGroupEntry`, `GmmSituController` |
| [grouped_matmul_situ_quant.cpp](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/grouped_matmul_situ_quant.cpp) | 分组矩阵乘与 SITU 量化 的设备执行源码；文件主题为 矩阵乘、量化、SITU 激活。 | 类型：`does` |
| [grouped_matmul_situ_quant_tiling.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/grouped_matmul_situ_quant_tiling.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 矩阵乘、量化、SITU 激活。 | 类型：`SituTilingHeader` |
| [situ_epilogue.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/situ_epilogue.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 SITU 激活。 | 类型：`IntegralConstant`, `IsSame` |

## `csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/`

该路径复用的算子/矩阵乘辅助实现；具体来源与许可见文件头。

| 子目录 | 职责 |
| --- | --- |
| [gmsq2/](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/gmsq2) | gmsq2 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [wqbmm/](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm) | wqbmm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

## `csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/gmsq2/`

gmsq2 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [basic_block_config.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/gmsq2/basic_block_config.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 配置、缓存页/分块。 | 类型：`WqmmConfig`, `BasicBlockControlParam`, `BasicBlockOffsetParam` |
| [basic_block_vf_mx.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/gmsq2/basic_block_vf_mx.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`MxA8W4NzParams` |
| [gmmsq_quant_mx_vf.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/gmsq2/gmmsq_quant_mx_vf.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 量化。 | 函数：`GmmSwigluVf`, `ComputeMaxExpVf` |
| [gmmsq_weight_quant_cube_compute.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/gmsq2/gmmsq_weight_quant_cube_compute.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化。 | 类型：`for`, `GMMSQWeightQuantCubeCompute` |
| [gmmsq_weight_quant_cube_compute_tools.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/gmsq2/gmmsq_weight_quant_cube_compute_tools.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化。 | 类型：`L0CopyAndCalcParams`, `FixL0CToDstParams` |
| [gmmsq_weight_quant_vcv_basic_block.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/gmsq2/gmmsq_weight_quant_vcv_basic_block.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化、缓存页/分块。 | 类型：`GMMSQWeightQuantVcvBasicBlock` |
| [gmmsq_weight_quant_vec_compute.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/gmsq2/gmmsq_weight_quant_vec_compute.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化。 | 类型：`GmmsqSituUbBufferInfo`, `GMMSQSituVecCompute` |

## `csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/`

wqbmm 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [basic_api/](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/basic_api) | basic_api 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [custom_policy/](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/custom_policy) | custom_policy 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [anti_quant_y_vf.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/anti_quant_y_vf.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 量化。 | 类型：`LocalAddressYParam` |
| [basic_block_config.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/basic_block_config.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 配置、缓存页/分块。 | 类型：`WqmmConfig`, `BasicBlockControlParam`, `BasicBlockOffsetParam` |
| [basic_block_vf_mx.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/basic_block_vf_mx.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`MxFp4NdScaleParams`, `MxFp4NdWeightParams`, `Fp4NzParams` |
| [basic_block_vf_nd.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/basic_block_vf_nd.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`LocalAddressParam`, `CalculateParam` |
| [basic_block_vf_nz.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/basic_block_vf_nz.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`Int4NzParams` |
| [grouped_matmul_tiling_data_apt.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/grouped_matmul_tiling_data_apt.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 矩阵乘。 | 类型：`GMMArray`, `GMMNoQuantBaseParams`, `GMMQuantParams` |
| [grouped_matmul_weight_quant_basic_controller.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/grouped_matmul_weight_quant_basic_controller.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 矩阵乘、权重、量化。 | 类型：`GMMWeightQuantBasicController` |
| [grouped_matmul_weight_quant_resplit_controller.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/grouped_matmul_weight_quant_resplit_controller.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 矩阵乘、权重、量化。 | 类型：`GMMWeightQuantResplitController` |
| [mx_a8w4_scale_trans_id_constant.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/mx_a8w4_scale_trans_id_constant.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [weight_quant_basic_block.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/weight_quant_basic_block.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化、缓存页/分块。 | 类型：`WeightQuantMatmulBasicBlock` |
| [weight_quant_basic_block_base.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/weight_quant_basic_block_base.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化、缓存页/分块。 | 类型：`WeightQuantMatmulBasicBlockBaseClass` |
| [weight_quant_cube_compute.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/weight_quant_cube_compute.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化。 | 类型：`WeightQuantBatchMatmulV2CubeCompute` |
| [weight_quant_tiling_key.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/weight_quant_tiling_key.h) | 分组矩阵乘与 SITU 量化 的kernel 分派使用的 tiling key 定义；文件主题为 权重、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [weight_quant_tool.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/weight_quant_tool.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化。 | 类型：`QuantType`, `MatmulL1GmType` |
| [weight_quant_vcv_basic_block.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/weight_quant_vcv_basic_block.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化、缓存页/分块。 | 类型：`WeightQuantVcvMatmulBasicBlock` |
| [weight_quant_vcv_basic_block_base.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/weight_quant_vcv_basic_block_base.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化、缓存页/分块。 | 类型：`WeightQuantVcvMatmulBasicBlockBaseClass` |
| [weight_quant_vec_compute.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/weight_quant_vec_compute.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化。 | 类型：`BasicBlockLibVectorAntiQuantCompute` |

## `csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/basic_api/`

basic_api 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [weight_quant_basic_api_v1.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/basic_api/weight_quant_basic_api_v1.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 权重、量化。 | 类型：`BasicApiParamsV1`, `WqbmmBasicApiV1` |

## `csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/custom_policy/`

custom_policy 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [wqbmm_copy_cube_out.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/custom_policy/wqbmm_copy_cube_out.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`A_TYPE`, `B_TYPE`, `C_TYPE` |
| [wqbmm_custom_policy.h](../../../csrc/moe/grouped_matmul_situ_quant/op_kernel/vendor/wqbmm/custom_policy/wqbmm_custom_policy.h) | 分组矩阵乘与 SITU 量化 的类型/模板与设备实现头文件。 | 类型：`WQBmmCustomPolicy` |

## `csrc/moe/grouped_matmul_situ_quant/utils/`

辅助函数 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [torch_kernel_helper.h](../../../csrc/moe/grouped_matmul_situ_quant/utils/torch_kernel_helper.h) | torch_kernel_helper 的公共接口/实现，供所在算子工程复用。 | 类型：`TorchNpuHelper` |

## `csrc/moe/hc_post/`

mHC 子层后混合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/hc_post/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/hc_post/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/hc_post/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/hc_post/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/hc_post/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [hc_post_def.cpp](../../../csrc/moe/hc_post/op_host/hc_post_def.cpp) | mHC 子层后混合 的主机算子定义、支持属性和注册。 | 注册：`HcPost` |
| [hc_post_proto.cpp](../../../csrc/moe/hc_post/op_host/hc_post_proto.cpp) | mHC 子层后混合 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShape4HcPost`, `InferDtype4HcPost` |
| [hc_post_tiling.cpp](../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp) | mHC 子层后混合 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`Tiling4HcPost`, `TilingPrepare4HcPost` |
| [hc_post_tiling.h](../../../csrc/moe/hc_post/op_host/hc_post_tiling.h) | mHC 子层后混合 的主机侧 tiling 参数/分块策略声明。 | 类型：`HcPostCompileInfo`, `HcPostTilingRegbase`, `HcPostTiling` |
| [hc_post_tiling_arch35.h](../../../csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h) | mHC 子层后混合 的主机侧 tiling 参数/分块策略声明。 | 函数：`CeilDiv`, `CeilAlign` |

## `csrc/moe/hc_post/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [hc_post.cpp](../../../csrc/moe/hc_post/op_kernel/hc_post.cpp) | mHC 子层后混合 的设备执行源码；文件主题为 多路残差混合。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [hc_post_bfloat16.h](../../../csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h) | mHC 子层后混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`HcPostRegBaseBfloat16` |
| [hc_post_d_split.h](../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h) | mHC 子层后混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`HcPostKernelDSplit` |
| [hc_post_float32.h](../../../csrc/moe/hc_post/op_kernel/hc_post_float32.h) | mHC 子层后混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`HcPostRegBaseFloat32` |

## `csrc/moe/hc_pre/`

mHC 子层前混合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/hc_pre/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/hc_pre/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/hc_pre/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/hc_pre/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/hc_pre/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [hc_pre_def.cpp](../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp) | mHC 子层前混合 的主机算子定义、支持属性和注册。 | 注册：`HcPre` |
| [hc_pre_proto.cpp](../../../csrc/moe/hc_pre/op_host/hc_pre_proto.cpp) | mHC 子层前混合 的算子原型、输入输出或类型/shape 推断注册。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [hc_pre_tiling.cpp](../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp) | mHC 子层前混合 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`CeilDiv`, `DownAlign` |
| [hc_pre_tiling.h](../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.h) | mHC 子层前混合 的主机侧 tiling 参数/分块策略声明。 | 类型：`TilingRequiredParaInfo`, `TilingOptionalParaInfo`, `HcPreCompileInfo` |
| [hc_pre_tiling_arch35.h](../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h) | mHC 子层前混合 的主机侧 tiling 参数/分块策略声明。 | 类型：`CoreRowTiling`, `RowAndDTiling`, `UbBufferConfig` |

## `csrc/moe/hc_pre/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [hc_pre.cpp](../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp) | mHC 子层前混合 的设备执行源码；文件主题为 多路残差混合。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [hc_pre_base.h](../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h) | mHC 子层前混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 函数：`CeilDiv`, `CeilAlign` |
| [hc_pre_base_arch35.h](../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h) | mHC 子层前混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 函数：`Align`, `CeilDiv` |
| [hc_pre_cube_compute.h](../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h) | mHC 子层前混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`MmParams`, `HcCubeCompute` |
| [hc_pre_cube_compute_arch35.h](../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h) | mHC 子层前混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`HcPreCubeCompute` |
| [hc_pre_m_k_split_core.h](../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h) | mHC 子层前混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`HcPreMembaseKSplitCorePart1`, `HcPreMembaseKSplitCorePart2` |
| [hc_pre_m_k_split_core_arch35.h](../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_arch35.h) | mHC 子层前混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`HcPreMKSplitCorePart1`, `HcPreMKSplitCorePart2` |
| [hc_pre_m_k_split_core_premix_arch35.h](../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_premix_arch35.h) | mHC 子层前混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`HcPreMKSplitCorePremixPart1`, `HcPreMKSplitCorePremixPart2` |
| [hc_pre_m_split_core_arch35.h](../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_arch35.h) | mHC 子层前混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`HcPreMSplitCoreArch35` |
| [hc_pre_m_split_core_premix_arch35.h](../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_premix_arch35.h) | mHC 子层前混合 的类型/模板与设备实现头文件；文件主题为 多路残差混合。 | 类型：`HcPreMSplitCorePremixArch35` |

## `csrc/moe/moe_gating_top_k/`

MoE 专家 TopK 路由 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/moe_gating_top_k/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/moe_gating_top_k/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/moe_gating_top_k/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [moe_gating_top_k_torch_adpt.h](../../../csrc/moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h) | 门控、TopK 选择、混合专家 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/moe_gating_top_k/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/moe_gating_top_k/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [math_util.h](../../../csrc/moe/moe_gating_top_k/op_host/math_util.h) | MoE 专家 TopK 路由 的主机侧辅助接口或实现。 | 类型：`MathUtil` |
| [moe_gating_top_k_def.cpp](../../../csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_def.cpp) | MoE 专家 TopK 路由 的主机算子定义、支持属性和注册。 | 注册：`MoeGatingTopK` |
| [moe_gating_top_k_infershape.cpp](../../../csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_infershape.cpp) | MoE 专家 TopK 路由 的主机侧辅助接口或实现。 | 函数：`CheckInputShape`, `CheckInputDimsAndAttr` |
| [moe_gating_top_k_proto.cpp](../../../csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_proto.cpp) | MoE 专家 TopK 路由 的算子原型、输入输出或类型/shape 推断注册。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [moe_gating_top_k_proto.h](../../../csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_proto.h) | MoE 专家 TopK 路由 的算子原型、输入输出或类型/shape 推断注册。 | 注册：`MoeGatingTopK` |
| [moe_gating_top_k_tiling.cpp](../../../csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling.cpp) | MoE 专家 TopK 路由 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`MoeGatingTopKTilingBase` |
| [moe_gating_top_k_tiling.h](../../../csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling.h) | MoE 专家 TopK 路由 的主机侧 tiling 参数/分块策略声明。 | 类型：`MoeGatingTopKCompileInfo` |
| [moe_gating_top_k_tiling_arch35.cpp](../../../csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling_arch35.cpp) | MoE 专家 TopK 路由 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`MoeGatingTopKTilingRegbase` |
| [moe_gating_top_k_tiling_base.cpp](../../../csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling_base.cpp) | MoE 专家 TopK 路由 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`TilingForMoeGatingTopK`, `TilingPrepareForMoeGatingTopK` |

## `csrc/moe/moe_gating_top_k/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [common.h](../../../csrc/moe/moe_gating_top_k/op_kernel/common.h) | MoE 专家 TopK 路由 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`Ceil`, `Align` |
| [error_log.h](../../../csrc/moe/moe_gating_top_k/op_kernel/error_log.h) | MoE 专家 TopK 路由 的类型/模板与设备实现头文件；文件主题为 日志。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [moe_gating_top_k.cpp](../../../csrc/moe/moe_gating_top_k/op_kernel/moe_gating_top_k.cpp) | MoE 专家 TopK 路由 的设备执行源码；文件主题为 门控、TopK 选择、混合专家。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [moe_gating_top_k_apt.cpp](../../../csrc/moe/moe_gating_top_k/op_kernel/moe_gating_top_k_apt.cpp) | MoE 专家 TopK 路由 的设备执行源码；文件主题为 门控、TopK 选择、混合专家。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [moe_gating_top_k_e_k_fullload.h](../../../csrc/moe/moe_gating_top_k/op_kernel/moe_gating_top_k_e_k_fullload.h) | MoE 专家 TopK 路由 的类型/模板与设备实现头文件；文件主题为 门控、TopK 选择、加载。 | 类型：`MoeGatingTopKEKFullload` |
| [moe_gating_top_k_generalized.h](../../../csrc/moe/moe_gating_top_k/op_kernel/moe_gating_top_k_generalized.h) | MoE 专家 TopK 路由 的类型/模板与设备实现头文件；文件主题为 门控、TopK 选择、混合专家。 | 类型：`MoeGatingTopKGenerlized` |
| [moe_gating_top_k_without_group.h](../../../csrc/moe/moe_gating_top_k/op_kernel/moe_gating_top_k_without_group.h) | MoE 专家 TopK 路由 的类型/模板与设备实现头文件；文件主题为 门控、TopK 选择、混合专家。 | 类型：`MoeGatingTopKWithoutGroup` |

## `csrc/moe/moe_gating_top_k_hash/`

MoE hash/TopK 路由 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/moe_gating_top_k_hash/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/moe_gating_top_k_hash/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/moe_gating_top_k_hash/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/moe_gating_top_k_hash/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/moe_gating_top_k_hash/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [moe_gating_top_k_hash_def.cpp](../../../csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_def.cpp) | MoE hash/TopK 路由 的主机算子定义、支持属性和注册。 | 注册：`MoeGatingTopKHash` |
| [moe_gating_top_k_hash_proto.cpp](../../../csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_proto.cpp) | MoE hash/TopK 路由 的算子原型、输入输出或类型/shape 推断注册。 | 注册：`MoeGatingTopKHash` |
| [moe_gating_top_k_hash_tiling.cpp](../../../csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_tiling.cpp) | MoE hash/TopK 路由 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`MoeGatingTopKHashTilingBase` |
| [moe_gating_top_k_hash_tiling.h](../../../csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_tiling.h) | MoE hash/TopK 路由 的主机侧 tiling 参数/分块策略声明。 | 类型：`MoeGatingTopKHashCompileInfo` |
| [moe_gating_top_k_hash_tiling_arch35.h](../../../csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_tiling_arch35.h) | MoE hash/TopK 路由 的主机侧 tiling 参数/分块策略声明。 | 类型：`MoeGatingTopKHashTilingRegbase` |

## `csrc/moe/moe_gating_top_k_hash/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/arch35) | arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [common.h](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/common.h) | MoE hash/TopK 路由 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`Ceil`, `Align` |
| [common_regbase.h](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/common_regbase.h) | MoE hash/TopK 路由 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`Ceil`, `Align` |
| [moe_gating_top_k_hash.cpp](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash.cpp) | MoE hash/TopK 路由 的设备执行源码；文件主题为 门控、TopK 选择、哈希映射。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [moe_gating_top_k_hash_e_k_fullload.h](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash_e_k_fullload.h) | MoE hash/TopK 路由 的类型/模板与设备实现头文件；文件主题为 门控、TopK 选择、加载。 | 类型：`MoeGatingTopKHashEKFullload` |
| [moe_gating_top_k_hash_generalized.h](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash_generalized.h) | MoE hash/TopK 路由 的类型/模板与设备实现头文件；文件主题为 门控、TopK 选择、哈希映射。 | 类型：`MoeGatingTopKHashGenerlized` |
| [moe_gating_top_k_hash_regbase.h](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash_regbase.h) | MoE hash/TopK 路由 的类型/模板与设备实现头文件；文件主题为 门控、TopK 选择、哈希映射。 | 类型：`MoeGatingTopKHashRegbase` |
| [moe_gating_top_k_hash_without_group.h](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash_without_group.h) | MoE hash/TopK 路由 的类型/模板与设备实现头文件；文件主题为 门控、TopK 选择、哈希映射。 | 类型：`MoeGatingTopKHashWithoutGroup` |

## `csrc/moe/moe_gating_top_k_hash/op_kernel/arch35/`

arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [common.h](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/arch35/common.h) | MoE hash/TopK 路由 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`Ceil`, `Align` |
| [moe_gating_top_k_hash_regbase.h](../../../csrc/moe/moe_gating_top_k_hash/op_kernel/arch35/moe_gating_top_k_hash_regbase.h) | MoE hash/TopK 路由 的类型/模板与设备实现头文件；文件主题为 门控、TopK 选择、哈希映射。 | 类型：`MoeGatingTopKHashRegbase` |

## `csrc/moe/rms_norm_cast/`

RMSNorm 与类型转换 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/rms_norm_cast/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/rms_norm_cast/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/rms_norm_cast/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [rms_norm_cast_torch_adpt.h](../../../csrc/moe/rms_norm_cast/rms_norm_cast_torch_adpt.h) | RMS 归一化、类型转换 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/rms_norm_cast/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/rms_norm_cast/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [rms_norm_cast_def.cpp](../../../csrc/moe/rms_norm_cast/op_host/rms_norm_cast_def.cpp) | RMSNorm 与类型转换 的主机算子定义、支持属性和注册。 | 注册：`RmsNormCast` |
| [rms_norm_cast_infershape.cpp](../../../csrc/moe/rms_norm_cast/op_host/rms_norm_cast_infershape.cpp) | RMSNorm 与类型转换 的主机侧辅助接口或实现。 | 函数：`InferShape4RmsNormCast`, `InferDataType4RmsNormCast` |
| [rms_norm_cast_tiling.cpp](../../../csrc/moe/rms_norm_cast/op_host/rms_norm_cast_tiling.cpp) | RMSNorm 与类型转换 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`Tiling4RmsNormCast` |
| [rms_norm_cast_tiling.h](../../../csrc/moe/rms_norm_cast/op_host/rms_norm_cast_tiling.h) | RMSNorm 与类型转换 的主机侧 tiling 参数/分块策略声明。 | 类型：`RmsNormCastCompileInfo` |

## `csrc/moe/rms_norm_cast/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [rms_norm_cast.cpp](../../../csrc/moe/rms_norm_cast/op_kernel/rms_norm_cast.cpp) | RMSNorm 与类型转换 的设备执行源码；文件主题为 RMS 归一化、类型转换。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [rms_norm_cast.h](../../../csrc/moe/rms_norm_cast/op_kernel/rms_norm_cast.h) | RMSNorm 与类型转换 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、类型转换。 | 类型：`KernelRmsNormCast` |
| [rms_norm_cast_base.h](../../../csrc/moe/rms_norm_cast/op_kernel/rms_norm_cast_base.h) | RMSNorm 与类型转换 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、类型转换。 | 类型：`IntegralConstant`, `IsSame` |

## `csrc/moe/scatter_nd_update_sk/`

缓存索引写入 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/scatter_nd_update_sk/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/scatter_nd_update_sk/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/scatter_nd_update_sk/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/scatter_nd_update_sk/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/moe/scatter_nd_update_sk/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/scatter_nd_update_sk/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [scatter_nd_update_sk_def.cpp](../../../csrc/moe/scatter_nd_update_sk/op_host/scatter_nd_update_sk_def.cpp) | 缓存索引写入 的主机算子定义、支持属性和注册。 | 注册：`ScatterNdUpdateSk` |
| [scatter_nd_update_sk_infershape.cpp](../../../csrc/moe/scatter_nd_update_sk/op_host/scatter_nd_update_sk_infershape.cpp) | 缓存索引写入 的主机侧辅助接口或实现。 | 函数：`InferDataType4ScatterNdUpdateSk`, `InferShape4ScatterNdUpdateSk` |
| [scatter_nd_update_sk_tiling.cpp](../../../csrc/moe/scatter_nd_update_sk/op_host/scatter_nd_update_sk_tiling.cpp) | 缓存索引写入 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`ScatterInitInfo` |
| [scatter_nd_update_sk_tiling.h](../../../csrc/moe/scatter_nd_update_sk/op_host/scatter_nd_update_sk_tiling.h) | 缓存索引写入 的主机侧 tiling 参数/分块策略声明。 | 类型：`ScatterNdUpdateSkArch22CompileInfo`, `ScatterNdUpdateSkArch22Tiling` |
| [scatter_nd_update_sk_tiling_arch35.cpp](../../../csrc/moe/scatter_nd_update_sk/op_host/scatter_nd_update_sk_tiling_arch35.cpp) | 缓存索引写入 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`SkCeilDiv`, `SkCeilAlign` |
| [scatter_nd_update_sk_tiling_regbase.h](../../../csrc/moe/scatter_nd_update_sk/op_host/scatter_nd_update_sk_tiling_regbase.h) | 缓存索引写入 的主机侧 tiling 参数/分块策略声明。 | 类型：`ScatterNdUpdateSkTilingRegbase` |

## `csrc/moe/scatter_nd_update_sk/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_scatter_nd_update_sk.cpp](../../../csrc/moe/scatter_nd_update_sk/op_host/op_api/aclnn_scatter_nd_update_sk.cpp) | 缓存索引写入 的主机侧辅助接口或实现。 | 函数：`CheckNotNull`, `CheckDtypeValid` |
| [aclnn_scatter_nd_update_sk.h](../../../csrc/moe/scatter_nd_update_sk/op_host/op_api/aclnn_scatter_nd_update_sk.h) | 缓存索引写入 的主机侧辅助接口或实现。 | 函数：`aclnnScatterNdUpdateSkGetWorkspaceSize`, `aclnnScatterNdUpdateSk` |
| [scatter_nd_update_sk.cpp](../../../csrc/moe/scatter_nd_update_sk/op_host/op_api/scatter_nd_update_sk.cpp) | 缓存索引写入 的主机侧辅助接口或实现。 | 函数：`ScatterNdUpdateSkAiCore`, `ScatterNdUpdateSk` |
| [scatter_nd_update_sk.h](../../../csrc/moe/scatter_nd_update_sk/op_host/op_api/scatter_nd_update_sk.h) | 缓存索引写入 的主机侧辅助接口或实现。 | 函数：`ScatterNdUpdateSk` |

## `csrc/moe/scatter_nd_update_sk/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch22/](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch22) | arch22 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [arch35/](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35) | arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [scatter_nd_update_sk.cpp](../../../csrc/moe/scatter_nd_update_sk/op_kernel/scatter_nd_update_sk.cpp) | 缓存索引写入 的设备执行源码；文件主题为 分散写入。 | 函数：`RunLargeIndex`, `RunHp` |
| [scatter_nd_update_sk_apt.cpp](../../../csrc/moe/scatter_nd_update_sk/op_kernel/scatter_nd_update_sk_apt.cpp) | 缓存索引写入 的设备执行源码；文件主题为 分散写入。 | 类型：`GetComputeType` |

## `csrc/moe/scatter_nd_update_sk/op_kernel/arch22/`

arch22 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [scatter_nd_update.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch22/scatter_nd_update.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateKernel` |
| [scatter_nd_update_common.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch22/scatter_nd_update_common.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入、公共逻辑。 | 函数：`PipeMte2ToS`, `PipeMte3ToS` |
| [scatter_nd_update_hp.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch22/scatter_nd_update_hp.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateHpKernel` |
| [scatter_nd_update_large_index.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch22/scatter_nd_update_large_index.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`LargeIndexKernel` |
| [scatter_nd_update_linear_index.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch22/scatter_nd_update_linear_index.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入、线性投影。 | 类型：`LinearIndexKernel` |
| [scatter_nd_update_no_sort.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch22/scatter_nd_update_no_sort.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateKernelNoSort` |

## `csrc/moe/scatter_nd_update_sk/op_kernel/arch35/`

arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [indices_sort_utils.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/indices_sort_utils.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 辅助函数。 | 函数：`IndexStatisticInt32Vf`, `IndexStatisticInt64Vf` |
| [load_store_utils.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/load_store_utils.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 存储、辅助函数、加载。 | 函数：`LoadOneTensorForDtypeT`, `LoadTwoTensorForDtypeT` |
| [scatter_nd_update_common.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/scatter_nd_update_common.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入、公共逻辑。 | 类型：`ScatterNdUpdateBase`, `ScatterNdUpdateDeterministicCommon` |
| [scatter_nd_update_deterministic_simd.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/scatter_nd_update_deterministic_simd.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateDeterministicSimd` |
| [scatter_nd_update_deterministic_simt.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/scatter_nd_update_deterministic_simt.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateDeterministicSimt` |
| [scatter_nd_update_simd.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/scatter_nd_update_simd.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateSimd` |
| [scatter_nd_update_simd_mask.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/scatter_nd_update_simd_mask.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateSimdMask` |
| [scatter_nd_update_simt.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/scatter_nd_update_simt.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateSimt` |
| [scatter_nd_update_simt_sort.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/scatter_nd_update_simt_sort.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateSimtSort` |
| [scatter_nd_update_struct.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/scatter_nd_update_struct.h) | 缓存索引写入 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`ScatterNdUpdateSkRegBaseTilingData` |
| [sk_math_util.h](../../../csrc/moe/scatter_nd_update_sk/op_kernel/arch35/sk_math_util.h) | 缓存索引写入 的类型/模板与设备实现头文件。 | 函数：`GetVRegSize`, `GetUbBlockSize` |

## `csrc/moe/situ_mx_quant/`

SITU 与 MX 量化 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/moe/situ_mx_quant/docs) | docs 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [op_host/](../../../csrc/moe/situ_mx_quant/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/situ_mx_quant/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/situ_mx_quant/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [situ_mx_quant_torch_adpt.h](../../../csrc/moe/situ_mx_quant/situ_mx_quant_torch_adpt.h) | 量化、SITU 激活 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/situ_mx_quant/docs/`

docs 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnnSituMxQuant.md](../../../csrc/moe/situ_mx_quant/docs/aclnnSituMxQuant.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：aclnnSituMxQuant。 | 标题：aclnnSituMxQuant |

## `csrc/moe/situ_mx_quant/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/moe/situ_mx_quant/op_host/arch35) | arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [config/](../../../csrc/moe/situ_mx_quant/op_host/config) | 配置 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/situ_mx_quant/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [situ_mx_quant_def.cpp](../../../csrc/moe/situ_mx_quant/op_host/situ_mx_quant_def.cpp) | SITU 与 MX 量化 的主机算子定义、支持属性和注册。 | 注册：`SituMxQuant` |
| [situ_mx_quant_infershape.cpp](../../../csrc/moe/situ_mx_quant/op_host/situ_mx_quant_infershape.cpp) | SITU 与 MX 量化 的主机侧辅助接口或实现。 | 函数：`InferShapeForSituMxQuant`, `InferDataTypeForSituMxQuant` |

## `csrc/moe/situ_mx_quant/op_host/arch35/`

arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [situ_mx_quant_tiling_arch35.cpp](../../../csrc/moe/situ_mx_quant/op_host/arch35/situ_mx_quant_tiling_arch35.cpp) | SITU 与 MX 量化 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`Shape2String`, `Tiling4SituMxQuant` |
| [situ_mx_quant_tiling_arch35.h](../../../csrc/moe/situ_mx_quant/op_host/arch35/situ_mx_quant_tiling_arch35.h) | SITU 与 MX 量化 的主机侧 tiling 参数/分块策略声明。 | 类型：`SituMxQuantCompileInfo`, `SituMxQuantInputInfo`, `SituMxQuantOutputInfo` |

## `csrc/moe/situ_mx_quant/op_host/config/`

配置 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 子目录 | 职责 |
| --- | --- |
| [ascend950/](../../../csrc/moe/situ_mx_quant/op_host/config/ascend950) | 面向 ascend950 的算子型号/架构配置或构建选择。 |

## `csrc/moe/situ_mx_quant/op_host/config/ascend950/`

面向 ascend950 的算子型号/架构配置或构建选择。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [situ_mx_quant_binary.json](../../../csrc/moe/situ_mx_quant/op_host/config/ascend950/situ_mx_quant_binary.json) | SITU 与 MX 量化 的目标型号、二进制或编译分发配置。 | 顶层项：`op_type`, `op_list` |

## `csrc/moe/situ_mx_quant/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/moe/situ_mx_quant/op_kernel/arch35) | arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [inc/](../../../csrc/moe/situ_mx_quant/op_kernel/inc) | 公共头文件、类型和接口声明。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [situ_mx_quant_apt.cpp](../../../csrc/moe/situ_mx_quant/op_kernel/situ_mx_quant_apt.cpp) | SITU 与 MX 量化 的设备执行源码；文件主题为 量化、SITU 激活。 | 函数：`situ_mx_quant` |

## `csrc/moe/situ_mx_quant/op_kernel/arch35/`

arch35 子模块；所属范围：专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [situ_mx_quant_axis_last.h](../../../csrc/moe/situ_mx_quant/op_kernel/arch35/situ_mx_quant_axis_last.h) | SITU 与 MX 量化 的类型/模板与设备实现头文件；文件主题为 量化、SITU 激活。 | 类型：`SituMxQuantAxisLast` |
| [situ_mx_quant_common.h](../../../csrc/moe/situ_mx_quant/op_kernel/arch35/situ_mx_quant_common.h) | SITU 与 MX 量化 的类型/模板与设备实现头文件；文件主题为 公共逻辑、量化、SITU 激活。 | 函数：`ComputeVfMaxExpVfLast`, `ComputeScaleLast` |
| [situ_mx_quant_tiling_data.h](../../../csrc/moe/situ_mx_quant/op_kernel/arch35/situ_mx_quant_tiling_data.h) | SITU 与 MX 量化 的类型/模板与设备实现头文件；文件主题为 量化、SITU 激活。 | 类型：`SituMxQuantTilingData` |
| [situ_mx_quant_tiling_key.h](../../../csrc/moe/situ_mx_quant/op_kernel/arch35/situ_mx_quant_tiling_key.h) | SITU 与 MX 量化 的kernel 分派使用的 tiling key 定义；文件主题为 量化、SITU 激活。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/situ_mx_quant/op_kernel/inc/`

公共头文件、类型和接口声明。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kernel_utils.h](../../../csrc/moe/situ_mx_quant/op_kernel/inc/kernel_utils.h) | SITU 与 MX 量化 的类型/模板与设备实现头文件；文件主题为 辅助函数。 | 类型：`IntegralConstant`, `IsSame` |
| [platform.h](../../../csrc/moe/situ_mx_quant/op_kernel/inc/platform.h) | SITU 与 MX 量化 的类型/模板与设备实现头文件；文件主题为 平台接入。 | 函数：`IsDataCopyPadSupport`, `GetVRegSize` |

## `csrc/moe/swiglu_group_quant/`

SwiGLU 分组量化 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/swiglu_group_quant/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/swiglu_group_quant/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/swiglu_group_quant/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/moe/swiglu_group_quant/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/swiglu_group_quant/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [swiglu_group_quant_def.cpp](../../../csrc/moe/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp) | SwiGLU 分组量化 的主机算子定义、支持属性和注册。 | 注册：`SwigluGroupQuant` |
| [swiglu_group_quant_proto.cpp](../../../csrc/moe/swiglu_group_quant/op_host/swiglu_group_quant_proto.cpp) | SwiGLU 分组量化 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShape4SwigluGroupQuant`, `InferDtype4SwigluGroupQuant` |
| [swiglu_group_quant_tiling.cpp](../../../csrc/moe/swiglu_group_quant/op_host/swiglu_group_quant_tiling.cpp) | SwiGLU 分组量化 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`CeilDiv`, `DownAlign` |
| [swiglu_group_quant_tiling.h](../../../csrc/moe/swiglu_group_quant/op_host/swiglu_group_quant_tiling.h) | SwiGLU 分组量化 的主机侧 tiling 参数/分块策略声明。 | 类型：`TilingRequiredParaInfo`, `TilingOptionalParaInfo`, `SwigluGroupQuantCompileInfo` |

## `csrc/moe/swiglu_group_quant/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [swiglu_fp8_quant_per_token.h](../../../csrc/moe/swiglu_group_quant/op_kernel/swiglu_fp8_quant_per_token.h) | SwiGLU 分组量化 的类型/模板与设备实现头文件；文件主题为 SwiGLU 激活、量化。 | 类型：`SwigluFp8QuantPerToken` |
| [swiglu_group_quant.cpp](../../../csrc/moe/swiglu_group_quant/op_kernel/swiglu_group_quant.cpp) | SwiGLU 分组量化 的设备执行源码；文件主题为 SwiGLU 激活、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [swiglu_group_quant_base.h](../../../csrc/moe/swiglu_group_quant/op_kernel/swiglu_group_quant_base.h) | SwiGLU 分组量化 的类型/模板与设备实现头文件；文件主题为 SwiGLU 激活、量化。 | 函数：`CeilDiv`, `CeilAlign` |
| [swiglu_group_quant_perf.h](../../../csrc/moe/swiglu_group_quant/op_kernel/swiglu_group_quant_perf.h) | SwiGLU 分组量化 的类型/模板与设备实现头文件；文件主题为 SwiGLU 激活、量化。 | 类型：`SwigluGroupQuantPerf` |
| [swiglu_mx_quant_perf.h](../../../csrc/moe/swiglu_group_quant/op_kernel/swiglu_mx_quant_perf.h) | SwiGLU 分组量化 的类型/模板与设备实现头文件；文件主题为 SwiGLU 激活、量化。 | 类型：`SwigluMxQuantPerf` |

## `csrc/moe/transpose_kv_cache_by_block/`

按 block 转置 KV 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/moe/transpose_kv_cache_by_block/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/moe/transpose_kv_cache_by_block/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/transpose_kv_cache_by_block/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/moe/transpose_kv_cache_by_block/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：TransposeKvCacheByBlock |

## `csrc/moe/transpose_kv_cache_by_block/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/moe/transpose_kv_cache_by_block/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [transpose_kv_cache_by_block_def.cpp](../../../csrc/moe/transpose_kv_cache_by_block/op_host/transpose_kv_cache_by_block_def.cpp) | 按 block 转置 KV 的主机算子定义、支持属性和注册。 | 注册：`TransposeKvCacheByBlock` |
| [transpose_kv_cache_by_block_proto.cpp](../../../csrc/moe/transpose_kv_cache_by_block/op_host/transpose_kv_cache_by_block_proto.cpp) | 按 block 转置 KV 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShapeTransposeKvCacheByBlock`, `InferDataTypeTransposeKvCacheByBlock` |
| [transpose_kv_cache_by_block_tiling.cpp](../../../csrc/moe/transpose_kv_cache_by_block/op_host/transpose_kv_cache_by_block_tiling.cpp) | 按 block 转置 KV 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`TransposeKvCacheByBlockCompileInfo` |
| [transpose_kv_cache_by_block_tiling.h](../../../csrc/moe/transpose_kv_cache_by_block/op_host/transpose_kv_cache_by_block_tiling.h) | 按 block 转置 KV 的主机侧 tiling 参数/分块策略声明。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/moe/transpose_kv_cache_by_block/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [common.h](../../../csrc/moe/transpose_kv_cache_by_block/op_kernel/common.h) | 按 block 转置 KV 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`GetTensorAddr` |
| [full_load.h](../../../csrc/moe/transpose_kv_cache_by_block/op_kernel/full_load.h) | 按 block 转置 KV 的类型/模板与设备实现头文件；文件主题为 加载。 | 类型：`TransposeKvCacheByBlockKernelFullLoad` |
| [general.h](../../../csrc/moe/transpose_kv_cache_by_block/op_kernel/general.h) | 按 block 转置 KV 的类型/模板与设备实现头文件。 | 类型：`TransposeKvCacheByBlockKernelGeneral` |
| [transpose_kv_cache_by_block.cpp](../../../csrc/moe/transpose_kv_cache_by_block/op_kernel/transpose_kv_cache_by_block.cpp) | 按 block 转置 KV 的设备执行源码；文件主题为 转置、缓存、缓存页/分块。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
