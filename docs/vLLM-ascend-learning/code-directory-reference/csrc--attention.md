# csrc/attention：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **862 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/attention/`

注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [attn_res_fwd/](../../../csrc/attention/attn_res_fwd) | Attention residual 前向 算子工程；主机/设备职责以其子目录区分。 |
| [chunk_kda_fwd/](../../../csrc/attention/chunk_kda_fwd) | 分块 KDA 前向 算子工程；主机/设备职责以其子目录区分。 |
| [common/](../../../csrc/attention/common) | 注意力算子共用的数据结构、内存与计算辅助，不是单独的模型层。 |
| [compressor/](../../../csrc/attention/compressor) | 长程表示压缩 算子工程；主机/设备职责以其子目录区分。 |
| [compressor_metadata/](../../../csrc/attention/compressor_metadata) | Compressor 设备元数据 算子工程；主机/设备职责以其子目录区分。 |
| [fused_lightning_indexer_manage/](../../../csrc/attention/fused_lightning_indexer_manage) | 历史位置索引及缓存管理融合 算子工程；主机/设备职责以其子目录区分。 |
| [fused_quant_lightning_indexer_manage/](../../../csrc/attention/fused_quant_lightning_indexer_manage) | 量化索引与缓存管理融合 算子工程；主机/设备职责以其子目录区分。 |
| [fused_scatter_copy_sparse_flash_attention/](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention) | 缓存 scatter/copy 与稀疏 Attention 融合 算子工程；主机/设备职责以其子目录区分。 |
| [fused_sparse_attention_overlap/](../../../csrc/attention/fused_sparse_attention_overlap) | 稀疏注意力融合重叠 算子工程；主机/设备职责以其子目录区分。 |
| [indexer_compress_epilog_v2/](../../../csrc/attention/indexer_compress_epilog_v2) | 压缩 Indexer 后处理 V2 算子工程；主机/设备职责以其子目录区分。 |
| [inplace_partial_rotary_mul/](../../../csrc/attention/inplace_partial_rotary_mul) | 部分通道原位位置旋转 算子工程；主机/设备职责以其子目录区分。 |
| [k2q_csr/](../../../csrc/attention/k2q_csr) | K 到 Q 的 CSR 索引组织 算子工程；主机/设备职责以其子目录区分。 |
| [kda_gate_cumsum_vllm/](../../../csrc/attention/kda_gate_cumsum_vllm) | KDA 门控累积 算子工程；主机/设备职责以其子目录区分。 |
| [kda_layout_swap12/](../../../csrc/attention/kda_layout_swap12) | KDA 维度布局交换 算子工程；主机/设备职责以其子目录区分。 |
| [kv_compress_epilog/](../../../csrc/attention/kv_compress_epilog) | 压缩 KV 后处理 算子工程；主机/设备职责以其子目录区分。 |
| [kv_compress_epilog_v2/](../../../csrc/attention/kv_compress_epilog_v2) | 压缩 KV 后处理 V2 算子工程；主机/设备职责以其子目录区分。 |
| [kv_quant_sparse_attn_sharedkv/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv) | 量化共享 KV 稀疏注意力 算子工程；主机/设备职责以其子目录区分。 |
| [kv_quant_sparse_attn_sharedkv_metadata/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata) | 量化共享 KV 稀疏元数据 算子工程；主机/设备职责以其子目录区分。 |
| [kv_quant_sparse_flash_attention_vllm/](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm) | 量化 KV 稀疏 Flash Attention 算子工程；主机/设备职责以其子目录区分。 |
| [lightning_indexer_quant/](../../../csrc/attention/lightning_indexer_quant) | Indexer 量化 算子工程；主机/设备职责以其子目录区分。 |
| [lightning_indexer_v2/](../../../csrc/attention/lightning_indexer_v2) | 历史位置索引 算子工程；主机/设备职责以其子目录区分。 |
| [msa_index_score/](../../../csrc/attention/msa_index_score) | MSA 相关位置评分 算子工程；主机/设备职责以其子目录区分。 |
| [quant_lightning_indexer_v2/](../../../csrc/attention/quant_lightning_indexer_v2) | 量化历史位置索引 算子工程；主机/设备职责以其子目录区分。 |
| [quant_lightning_indexer_v2_metadata/](../../../csrc/attention/quant_lightning_indexer_v2_metadata) | 量化索引元数据 算子工程；主机/设备职责以其子目录区分。 |
| [recurrent_gated_delta_rule_v310/](../../../csrc/attention/recurrent_gated_delta_rule_v310) | 310P gated delta 递归状态 算子工程；主机/设备职责以其子目录区分。 |
| [rms_norm_dynamic_quant/](../../../csrc/attention/rms_norm_dynamic_quant) | RMSNorm 与动态量化融合 算子工程；主机/设备职责以其子目录区分。 |
| [sparse_attention_score/](../../../csrc/attention/sparse_attention_score) | 注意力 算子工程；主机/设备职责以其子目录区分。 |
| [sparse_attn_sharedkv/](../../../csrc/attention/sparse_attn_sharedkv) | 共享 KV 稀疏注意力 算子工程；主机/设备职责以其子目录区分。 |
| [sparse_attn_sharedkv_metadata/](../../../csrc/attention/sparse_attn_sharedkv_metadata) | 共享 KV 稀疏元数据 算子工程；主机/设备职责以其子目录区分。 |
| [sparse_flash_attention/](../../../csrc/attention/sparse_flash_attention) | 稀疏 Flash Attention 算子工程；主机/设备职责以其子目录区分。 |
| [sparse_flash_mla/](../../../csrc/attention/sparse_flash_mla) | 稀疏 MLA 注意力 算子工程；主机/设备职责以其子目录区分。 |
| [sparse_flash_mla_metadata/](../../../csrc/attention/sparse_flash_mla_metadata) | 稀疏 MLA 元数据 算子工程；主机/设备职责以其子目录区分。 |
| [store_kv_block/](../../../csrc/attention/store_kv_block) | KV block 写入 算子工程；主机/设备职责以其子目录区分。 |
| [store_kv_block_metadata/](../../../csrc/attention/store_kv_block_metadata) | KV block 写入元数据 算子工程；主机/设备职责以其子目录区分。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kda_torch_adpt_common.h](../../../csrc/attention/kda_torch_adpt_common.h) | 公共逻辑、KDA 线性注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`kda_ceil_div`, `get_kda_seq_num` |

## `csrc/attention/attn_res_fwd/`

Attention residual 前向 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/attn_res_fwd/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/attn_res_fwd/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/attn_res_fwd/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/attn_res_fwd/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：AttnResFwd |
| [attn_res_fwd_common.h](../../../csrc/attention/attn_res_fwd/attn_res_fwd_common.h) | 公共逻辑、注意力 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [attn_res_fwd_torch_adpt.h](../../../csrc/attention/attn_res_fwd/attn_res_fwd_torch_adpt.h) | 注意力 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/attn_res_fwd/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/attn_res_fwd/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/attn_res_fwd/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [attn_res_fwd_def.cpp](../../../csrc/attention/attn_res_fwd/op_host/attn_res_fwd_def.cpp) | Attention residual 前向 的主机算子定义、支持属性和注册。 | 注册：`AttnResFwd` |
| [attn_res_fwd_infershape.cpp](../../../csrc/attention/attn_res_fwd/op_host/attn_res_fwd_infershape.cpp) | Attention residual 前向 的主机侧辅助接口或实现。 | 函数：`InferShape`, `InferDataType` |
| [attn_res_fwd_tiling.cpp](../../../csrc/attention/attn_res_fwd/op_host/attn_res_fwd_tiling.cpp) | Attention residual 前向 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`AlignUpU32`, `AlignDownU32` |
| [attn_res_fwd_tiling.h](../../../csrc/attention/attn_res_fwd/op_host/attn_res_fwd_tiling.h) | Attention residual 前向 的主机侧 tiling 参数/分块策略声明。 | 类型：`AttnResFwdCompileInfo`, `AttnResFwdInfo`, `AttnResFwdTiling` |
| [attn_res_fwd_tiling_register.cpp](../../../csrc/attention/attn_res_fwd/op_host/attn_res_fwd_tiling_register.cpp) | Attention residual 前向 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`Tiling`, `Parse` |

## `csrc/attention/attn_res_fwd/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_attn_res_fwd.cpp](../../../csrc/attention/attn_res_fwd/op_host/op_api/aclnn_attn_res_fwd.cpp) | Attention residual 前向 的主机侧辅助接口或实现。 | 函数：`AttnResFwd` |
| [aclnn_attn_res_fwd.h](../../../csrc/attention/attn_res_fwd/op_host/op_api/aclnn_attn_res_fwd.h) | Attention residual 前向 的主机侧辅助接口或实现。 | 函数：`aclnnAttnResFwdGetWorkspaceSize`, `aclnnAttnResFwd` |

## `csrc/attention/attn_res_fwd/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch22/](../../../csrc/attention/attn_res_fwd/op_kernel/arch22) | arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/attn_res_fwd/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [attn_res_fwd_apt.cpp](../../../csrc/attention/attn_res_fwd/op_kernel/attn_res_fwd_apt.cpp) | Attention residual 前向 的设备执行源码；文件主题为 注意力。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [attn_res_fwd_tiling_data.h](../../../csrc/attention/attn_res_fwd/op_kernel/attn_res_fwd_tiling_data.h) | Attention residual 前向 的类型/模板与设备实现头文件；文件主题为 注意力。 | 类型：`alignas` |
| [tiling_key_attn_res_fwd.h](../../../csrc/attention/attn_res_fwd/op_kernel/tiling_key_attn_res_fwd.h) | Attention residual 前向 的kernel 分派使用的 tiling key 定义；文件主题为 注意力。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/attn_res_fwd/op_kernel/arch22/`

arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [attn_res_fwd_reload.h](../../../csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_reload.h) | Attention residual 前向 的类型/模板与设备实现头文件；文件主题为 注意力、加载。 | 类型：`AttnResFwdInitParams`, `AttnResFwdReload` |
| [attn_res_fwd_resident.h](../../../csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_resident.h) | Attention residual 前向 的类型/模板与设备实现头文件；文件主题为 注意力。 | 类型：`AttnResFwdResident` |
| [reduce_common.h](../../../csrc/attention/attn_res_fwd/op_kernel/arch22/reduce_common.h) | Attention residual 前向 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`CeilDivU32`, `RoundUpFp32` |

## `csrc/attention/attn_res_fwd/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [attn_res_fwd_regbase_common.h](../../../csrc/attention/attn_res_fwd/op_kernel/arch35/attn_res_fwd_regbase_common.h) | Attention residual 前向 的类型/模板与设备实现头文件；文件主题为 公共逻辑、注意力。 | 函数：`AddB16`, `WeightedMulAddFromB16` |
| [attn_res_fwd_reload.h](../../../csrc/attention/attn_res_fwd/op_kernel/arch35/attn_res_fwd_reload.h) | Attention residual 前向 的类型/模板与设备实现头文件；文件主题为 注意力、加载。 | 类型：`AttnResFwdInitParams`, `AttnResFwdReload` |
| [attn_res_fwd_resident.h](../../../csrc/attention/attn_res_fwd/op_kernel/arch35/attn_res_fwd_resident.h) | Attention residual 前向 的类型/模板与设备实现头文件；文件主题为 注意力。 | 类型：`AttnResFwdResident` |
| [reduce_common.h](../../../csrc/attention/attn_res_fwd/op_kernel/arch35/reduce_common.h) | Attention residual 前向 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`CeilDivU32`, `RoundUpFp32` |

## `csrc/attention/chunk_kda_fwd/`

分块 KDA 前向 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/attention/chunk_kda_fwd/docs) | docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../csrc/attention/chunk_kda_fwd/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：ChunkKdaFwd |

## `csrc/attention/chunk_kda_fwd/docs/`

docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [api.md](../../../csrc/attention/chunk_kda_fwd/docs/api.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：api。 | 标题：ChunkKdaFwd API |
| [design.md](../../../csrc/attention/chunk_kda_fwd/docs/design.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：design。 | 标题：ChunkKdaFwd 设计 |

## `csrc/attention/common/`

注意力算子共用的数据结构、内存与计算辅助，不是单独的模型层。

| 子目录 | 职责 |
| --- | --- |
| [op_kernel/](../../../csrc/attention/common/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

## `csrc/attention/common/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/common/op_kernel/arch35) | arch35 子模块；所属范围：注意力算子共用的数据结构、内存与计算辅助，不是单独的模型层。 |
| [memcopy/](../../../csrc/attention/common/op_kernel/memcopy) | 复制 子模块；所属范围：注意力算子共用的数据结构、内存与计算辅助，不是单独的模型层。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CopyInL1.h](../../../csrc/attention/common/op_kernel/CopyInL1.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`KVLAYOUT`, `CopyParam`, `PAShape` |
| [FixpipeOut.h](../../../csrc/attention/common/op_kernel/FixpipeOut.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 类型：`fixpipeOutParams` |
| [aicpu_common.h](../../../csrc/attention/common/op_kernel/aicpu_common.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 函数：`AlignUp`, `GetAttrValue` |
| [attn_buffer.h](../../../csrc/attention/common/op_kernel/attn_buffer.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 注意力。 | 类型：`BufferType`, `SyncType`, `SyncMode` |
| [attn_buffer_manager.h](../../../csrc/attention/common/op_kernel/attn_buffer_manager.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 管理器、注意力。 | 类型：`BufferManager` |
| [buffer.h](../../../csrc/attention/common/op_kernel/buffer.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 类型：`BufferType`, `SyncType`, `BufferInfo` |
| [buffer_manager.h](../../../csrc/attention/common/op_kernel/buffer_manager.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 管理器。 | 类型：`BufferManager` |
| [buffer_mix_core.h](../../../csrc/attention/common/op_kernel/buffer_mix_core.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 类型：`BufferType`, `SyncType`, `BufferInfo` |
| [buffers_policy.h](../../../csrc/attention/common/op_kernel/buffers_policy.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 类型：`BuffersPolicySingleBuffer`, `BuffersPolicyDB`, `BuffersPolicy3buff` |
| [init_output.h](../../../csrc/attention/common/op_kernel/init_output.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 输出。 | 函数：`InitOutput` |
| [matmul.h](../../../csrc/attention/common/op_kernel/matmul.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 矩阵乘。 | 类型：`MMParam`, `ABLayout` |
| [offset_calculator.h](../../../csrc/attention/common/op_kernel/offset_calculator.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/common/op_kernel/arch35/`

arch35 子模块；所属范围：注意力算子共用的数据结构、内存与计算辅助，不是单独的模型层。

| 子目录 | 职责 |
| --- | --- |
| [vf/](../../../csrc/attention/common/op_kernel/arch35/vf) | vf 子模块；所属范围：注意力算子共用的数据结构、内存与计算辅助，不是单独的模型层。 |

## `csrc/attention/common/op_kernel/arch35/vf/`

vf 子模块；所属范围：注意力算子共用的数据结构、内存与计算辅助，不是单独的模型层。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [vf_basic_block_aligned128_no_update_sfa.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_basic_block_aligned128_no_update_sfa.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、稀疏注意力。 | 函数：`ProcessVec1NoUpdateImpl128VF`, `ProcessVec1NoUpdateImpl128` |
| [vf_basic_block_aligned128_update_sfa.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_basic_block_aligned128_update_sfa.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、稀疏注意力。 | 函数：`ProcessVec1UpdateImpl128VF`, `ProcessVec1UpdateImpl128` |
| [vf_basic_block_unaligned128_no_update_sfa.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_basic_block_unaligned128_no_update_sfa.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、稀疏注意力。 | 函数：`ProcessVec1NoUpdateGeneralImpl128VF`, `ProcessVec1NoUpdateGeneralImpl128` |
| [vf_basic_block_unaligned128_update_sfa.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_basic_block_unaligned128_update_sfa.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、稀疏注意力。 | 函数：`ProcessVec1UpdateGeneralImpl128VF`, `ProcessVec1UpdateGeneralImpl128` |
| [vf_basic_block_unaligned64_no_update_sfa.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_basic_block_unaligned64_no_update_sfa.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、稀疏注意力。 | 函数：`ProcessVec1NoUpdateImpl64VF`, `ProcessVec1NoUpdateImpl64` |
| [vf_basic_block_unaligned64_update_sfa.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_basic_block_unaligned64_update_sfa.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、稀疏注意力。 | 函数：`ProcessVec1UpdateImpl64VF`, `ProcessVec1UpdateImpl64` |
| [vf_basic_block_utils.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_basic_block_utils.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、辅助函数。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [vf_flash_decode_arch35.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_flash_decode_arch35.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 逐步生成。 | 函数：`ReduceFinalRes_0_VF`, `ReduceFinalRes_0` |
| [vf_flashupdate_new.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_flashupdate_new.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 函数：`FlashUpdateBasicVF`, `FlashUpdateBasic` |
| [vf_mul_sel_softmaxflashv2_cast_nz_sfa.h](../../../csrc/attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 概率归一化、类型转换、稀疏注意力。 | 类型：`OriginNRange` |

## `csrc/attention/common/op_kernel/memcopy/`

复制 子模块；所属范围：注意力算子共用的数据结构、内存与计算辅助，不是单独的模型层。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fa_gm_tensor.h](../../../csrc/attention/common/op_kernel/memcopy/fa_gm_tensor.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 张量。 | 类型：`FaGmTensor` |
| [fa_l1_tensor.h](../../../csrc/attention/common/op_kernel/memcopy/fa_l1_tensor.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 张量。 | 类型：`L1Format`, `ScaleTrans`, `FaL1Tensor` |
| [gm_coord.h](../../../csrc/attention/common/op_kernel/memcopy/gm_coord.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 类型：`GmCoord` |
| [gm_layout.h](../../../csrc/attention/common/op_kernel/memcopy/gm_layout.h) | 公共逻辑 的类型/模板与设备实现头文件；文件主题为 内存布局。 | 类型：`GmFormat`, `GmLayout` |
| [offset_calculator_v2.h](../../../csrc/attention/common/op_kernel/memcopy/offset_calculator_v2.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 类型：`FormatCategory`, `GmLayoutParams`, `OffsetCalculatorImpl` |
| [parser.h](../../../csrc/attention/common/op_kernel/memcopy/parser.h) | 公共逻辑 的类型/模板与设备实现头文件。 | 类型：`ActualSeqLensMode`, `ActualSeqLensParser`, `BlockTableParser` |

## `csrc/attention/compressor/`

长程表示压缩 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/compressor/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/compressor/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/compressor/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/compressor/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：Compressor |

## `csrc/attention/compressor/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [arch32/](../../../csrc/attention/compressor/op_host/arch32) | arch32 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/compressor/op_host/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/compressor/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [compressor_def.cpp](../../../csrc/attention/compressor/op_host/compressor_def.cpp) | 长程表示压缩 的主机算子定义、支持属性和注册。 | 注册：`Compressor` |
| [compressor_proto.cpp](../../../csrc/attention/compressor/op_host/compressor_proto.cpp) | 长程表示压缩 的算子原型、输入输出或类型/shape 推断注册。 | 类型：`CompressorProtoShapeParam` |

## `csrc/attention/compressor/op_host/arch32/`

arch32 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compressor_tiling.cpp](../../../csrc/attention/compressor/op_host/arch32/compressor_tiling.cpp) | 长程表示压缩 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`to_string`, `LayoutTypeToStr` |
| [compressor_tiling.h](../../../csrc/attention/compressor/op_host/arch32/compressor_tiling.h) | 长程表示压缩 的主机侧 tiling 参数/分块策略声明。 | 类型：`CompressorCompileInfo`, `RequiredParaInfo`, `OptionalParaInfo` |

## `csrc/attention/compressor/op_host/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compressor_tiling.cpp](../../../csrc/attention/compressor/op_host/arch35/compressor_tiling.cpp) | 长程表示压缩 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`to_string`, `LayoutTypeToStr` |
| [compressor_tiling.h](../../../csrc/attention/compressor/op_host/arch35/compressor_tiling.h) | 长程表示压缩 的主机侧 tiling 参数/分块策略声明。 | 类型：`CompressorCompileInfo`, `RequiredParaInfo`, `OptionalParaInfo` |

## `csrc/attention/compressor/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch32/](../../../csrc/attention/compressor/op_kernel/arch32) | arch32 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/compressor/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compressor.cpp](../../../csrc/attention/compressor/op_kernel/compressor.cpp) | 长程表示压缩 的设备执行源码；文件主题为 长程压缩。 | 函数：`compressor` |

## `csrc/attention/compressor/op_kernel/arch32/`

arch32 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compressor_block_cube_full_load.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_block_cube_full_load.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、缓存页/分块、加载。 | 类型：`CompressorBlockCubeFullLoad` |
| [compressor_block_cube_perf.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_block_cube_perf.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、缓存页/分块。 | 类型：`CompressorBlockCubePerf` |
| [compressor_block_vec_full_load.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_block_vec_full_load.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、缓存页/分块、加载。 | 类型：`CompressorBlockVectorFullLoad` |
| [compressor_block_vec_perf.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_block_vec_perf.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、缓存页/分块。 | 类型：`CompressorBlockVectorPerf` |
| [compressor_comm.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_comm.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`X_LAYOUT`, `X_DTYPE`, `COFF` |
| [compressor_kernel.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_kernel.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`CompressorKernel` |
| [compressor_kernel_full_load.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_kernel_full_load.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、加载。 | 类型：`CompressorKernelFullLoad` |
| [compressor_kernel_perf.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_kernel_perf.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`CmpBlockInfo`, `BasicBlockInfo`, `BatchInfo` |
| [compressor_template_tiling_key.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_template_tiling_key.h) | 长程表示压缩 的kernel 分派使用的 tiling key 定义；文件主题为 长程压缩。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [compressor_tiling_data.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_tiling_data.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`CompressorSplitCoreParams`, `CompressorBaseParams`, `CompressorPageAttentionParams` |
| [compressor_tools.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_tools.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`ToolsParams`, `CompressorTools`, `SliceInfo` |
| [compressor_vector_comm.h](../../../csrc/attention/compressor/op_kernel/arch32/compressor_vector_comm.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`MatRpeatParam`, `RmsNormParam` |
| [rms_norm.h](../../../csrc/attention/compressor/op_kernel/arch32/rms_norm.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 RMS 归一化。 | 函数：`RmsNorm` |
| [rope.h](../../../csrc/attention/compressor/op_kernel/arch32/rope.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 位置旋转。 | 函数：`SetGatherSrcOffset`, `RotaryPosEmb` |
| [soft_max.h](../../../csrc/attention/compressor/op_kernel/arch32/soft_max.h) | 长程表示压缩 的类型/模板与设备实现头文件。 | 函数：`ColumnSoftMax` |

## `csrc/attention/compressor/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [vf/](../../../csrc/attention/compressor/op_kernel/arch35/vf) | vf 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compressor_block_cube.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_block_cube.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、缓存页/分块。 | 类型：`CompressorBlockCube` |
| [compressor_block_cube_full_load.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_block_cube_full_load.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、缓存页/分块、加载。 | 类型：`CompressorBlockCubeFullLoad` |
| [compressor_block_vec.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_block_vec.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、缓存页/分块。 | 类型：`CompressorBlockVector` |
| [compressor_block_vec_full_load.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_block_vec_full_load.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、缓存页/分块、加载。 | 类型：`CompressorBlockVectorFullLoad` |
| [compressor_comm.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_comm.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`X_LAYOUT`, `X_DTYPE`, `COFF` |
| [compressor_kernel.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_kernel.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`CompressorKernel` |
| [compressor_kernel_full_load.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_kernel_full_load.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩、加载。 | 类型：`CompressorKernelFullLoad` |
| [compressor_template_tiling_key.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_template_tiling_key.h) | 长程表示压缩 的kernel 分派使用的 tiling key 定义；文件主题为 长程压缩。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [compressor_tiling_data.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_tiling_data.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`CompressorSplitCoreParams`, `CompressorBaseParams`, `CompressorPageAttentionParams` |
| [compressor_tools.h](../../../csrc/attention/compressor/op_kernel/arch35/compressor_tools.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 长程压缩。 | 类型：`ToolsParams`, `CompressorTools`, `SliceInfo` |

## `csrc/attention/compressor/op_kernel/arch35/vf/`

vf 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [vf_add.h](../../../csrc/attention/compressor/op_kernel/arch35/vf/vf_add.h) | 长程表示压缩 的类型/模板与设备实现头文件。 | 类型：`AddRegList` |
| [vf_mul.h](../../../csrc/attention/compressor/op_kernel/arch35/vf/vf_mul.h) | 长程表示压缩 的类型/模板与设备实现头文件。 | 类型：`ReduceMulRegList` |
| [vf_rms_norm.h](../../../csrc/attention/compressor/op_kernel/arch35/vf/vf_rms_norm.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 RMS 归一化。 | 函数：`RmsNormVFImpl`, `RmsNormVF` |
| [vf_rope.h](../../../csrc/attention/compressor/op_kernel/arch35/vf/vf_rope.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 位置旋转。 | 函数：`HalfModeRopeVF`, `InterleaveModeRopeVF` |
| [vf_softmax.h](../../../csrc/attention/compressor/op_kernel/arch35/vf/vf_softmax.h) | 长程表示压缩 的类型/模板与设备实现头文件；文件主题为 概率归一化。 | 函数：`SoftmaxDndBase128`, `SoftmaxDndBase64` |

## `csrc/attention/compressor_metadata/`

Compressor 设备元数据 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/compressor_metadata/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/compressor_metadata/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/compressor_metadata/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/attention/compressor_metadata/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/compressor_metadata/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [compressor_metadata_def.cpp](../../../csrc/attention/compressor_metadata/op_host/compressor_metadata_def.cpp) | Compressor 设备元数据 的主机算子定义、支持属性和注册。 | 注册：`CompressorMetadata` |
| [compressor_metadata_tiling.cpp](../../../csrc/attention/compressor_metadata/op_host/compressor_metadata_tiling.cpp) | Compressor 设备元数据 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`AlignUp`, `CeilDiv` |
| [compressor_metadata_tiling.h](../../../csrc/attention/compressor_metadata/op_host/compressor_metadata_tiling.h) | Compressor 设备元数据 的主机侧 tiling 参数/分块策略声明。 | 类型：`CompressorMetadataCompileInfo` |

## `csrc/attention/compressor_metadata/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compressor_metadata.cpp](../../../csrc/attention/compressor_metadata/op_kernel/compressor_metadata.cpp) | Compressor 设备元数据 的设备执行源码；文件主题为 长程压缩、元数据。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [compressor_metadata.h](../../../csrc/attention/compressor_metadata/op_kernel/compressor_metadata.h) | Compressor 设备元数据 的类型/模板与设备实现头文件；文件主题为 长程压缩、元数据。 | 类型：`CompressorMetadataTilingData`, `CompressorMetadataKernel` |

## `csrc/attention/fused_lightning_indexer_manage/`

历史位置索引及缓存管理融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/fused_lightning_indexer_manage/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/fused_lightning_indexer_manage/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/fused_lightning_indexer_manage/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：FusedLightningIndexerManage |
| [fused_lightning_indexer_manage_torch_adpt.h](../../../csrc/attention/fused_lightning_indexer_manage/fused_lightning_indexer_manage_torch_adpt.h) | 历史位置索引 的公共接口/实现，供所在算子工程复用。 | 函数：`npu_fused_lightning_indexer_manage` |

## `csrc/attention/fused_lightning_indexer_manage/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/fused_lightning_indexer_manage/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/fused_lightning_indexer_manage/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [fused_lightning_indexer_manage_def.cpp](../../../csrc/attention/fused_lightning_indexer_manage/op_host/fused_lightning_indexer_manage_def.cpp) | 历史位置索引及缓存管理融合 的主机算子定义、支持属性和注册。 | 注册：`FusedLightningIndexerManage` |
| [fused_lightning_indexer_manage_infershape.cpp](../../../csrc/attention/fused_lightning_indexer_manage/op_host/fused_lightning_indexer_manage_infershape.cpp) | 历史位置索引及缓存管理融合 的主机侧辅助接口或实现。 | 函数：`InferShapeFusedLightningIndexerManage`, `InferDataTypeFusedLightningIndexerManage` |
| [fused_lightning_indexer_manage_tiling.cpp](../../../csrc/attention/fused_lightning_indexer_manage/op_host/fused_lightning_indexer_manage_tiling.cpp) | 历史位置索引及缓存管理融合 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`TilingPrepareForFusedLightningIndexerManage`, `TilingForFusedLightningIndexerManage` |
| [fused_lightning_indexer_manage_tiling.h](../../../csrc/attention/fused_lightning_indexer_manage/op_host/fused_lightning_indexer_manage_tiling.h) | 历史位置索引及缓存管理融合 的主机侧 tiling 参数/分块策略声明。 | 类型：`MtpRequiredParaInfo`, `MtpTensorParaInfo`, `FusedLightningIndexerManageCompileInfo` |

## `csrc/attention/fused_lightning_indexer_manage/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_fused_lightning_indexer_manage.cpp](../../../csrc/attention/fused_lightning_indexer_manage/op_host/op_api/aclnn_fused_lightning_indexer_manage.cpp) | 历史位置索引及缓存管理融合 的主机侧辅助接口或实现。 | 函数：`aclnnInnerFusedLightningIndexerManageGetWorkspaceSize`, `aclnnInnerFusedLightningIndexerManage` |
| [aclnn_fused_lightning_indexer_manage.h](../../../csrc/attention/fused_lightning_indexer_manage/op_host/op_api/aclnn_fused_lightning_indexer_manage.h) | 历史位置索引及缓存管理融合 的主机侧辅助接口或实现。 | 函数：`aclnnFusedLightningIndexerManageGetWorkspaceSize`, `aclnnFusedLightningIndexerManage` |

## `csrc/attention/fused_lightning_indexer_manage/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fused_lightning_indexer_manage.cpp](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/fused_lightning_indexer_manage.cpp) | 历史位置索引及缓存管理融合 的设备执行源码；文件主题为 历史位置索引。 | 函数：`PrepareNonOffloadRows`, `FinalizeStandardCounts` |
| [fused_lightning_indexer_manage_constants.h](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/fused_lightning_indexer_manage_constants.h) | 历史位置索引及缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [fused_lightning_indexer_manage_template_tiling_key.h](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/fused_lightning_indexer_manage_template_tiling_key.h) | 历史位置索引及缓存管理融合 的kernel 分派使用的 tiling key 定义；文件主题为 历史位置索引。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [fused_lightning_indexer_manage_union.h](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/fused_lightning_indexer_manage_union.h) | 历史位置索引及缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 类型：`MtpMissUnion` |
| [fused_lightning_indexer_manage_workspace.h](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/fused_lightning_indexer_manage_workspace.h) | 历史位置索引及缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 函数：`LiWorkspaceBytes`, `Pair0Offset` |
| [lightning_indexer_common.h](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/lightning_indexer_common.h) | 历史位置索引及缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引、公共逻辑。 | 类型：`LI_LAYOUT`, `LIType`, `RunInfo` |
| [lightning_indexer_kernel.h](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/lightning_indexer_kernel.h) | 历史位置索引及缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 类型：`TempLoopInfo`, `LIPreload` |
| [lightning_indexer_service_cube.h](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/lightning_indexer_service_cube.h) | 历史位置索引及缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 类型：`LIMatmul` |
| [lightning_indexer_service_vector.h](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/lightning_indexer_service_vector.h) | 历史位置索引及缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 类型：`LIVector`, `LICommon` |
| [lightning_indexer_vector.h](../../../csrc/attention/fused_lightning_indexer_manage/op_kernel/lightning_indexer_vector.h) | 历史位置索引及缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 函数：`CopyIn`, `CopyOut` |

## `csrc/attention/fused_quant_lightning_indexer_manage/`

量化索引与缓存管理融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/fused_quant_lightning_indexer_manage/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/fused_quant_lightning_indexer_manage/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：FusedQuantLightningIndexerManage |
| [fused_quant_lightning_indexer_manage_torch_adpt.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_torch_adpt.h) | 历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 函数：`npu_fused_quant_lightning_indexer_manage` |

## `csrc/attention/fused_quant_lightning_indexer_manage/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [fused_quant_lightning_indexer_manage_def.cpp](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_host/fused_quant_lightning_indexer_manage_def.cpp) | 量化索引与缓存管理融合 的主机算子定义、支持属性和注册。 | 注册：`FusedQuantLightningIndexerManage` |
| [fused_quant_lightning_indexer_manage_infershape.cpp](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_host/fused_quant_lightning_indexer_manage_infershape.cpp) | 量化索引与缓存管理融合 的主机侧辅助接口或实现。 | 函数：`InferShapeFusedQuantLightningIndexerManage`, `InferDataTypeFusedQuantLightningIndexerManage` |
| [fused_quant_lightning_indexer_manage_tiling.cpp](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_host/fused_quant_lightning_indexer_manage_tiling.cpp) | 量化索引与缓存管理融合 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`TilingPrepareForFusedQuantLightningIndexerManage`, `TilingForFusedQuantLightningIndexerManage` |
| [fused_quant_lightning_indexer_manage_tiling.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_host/fused_quant_lightning_indexer_manage_tiling.h) | 量化索引与缓存管理融合 的主机侧 tiling 参数/分块策略声明。 | 类型：`MtpC8RequiredParaInfo`, `MtpC8TensorParaInfo`, `FusedQuantLightningIndexerManageCompileInfo` |

## `csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fused_quant_lightning_indexer_manage.cpp](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage.cpp) | 量化索引与缓存管理融合 的设备执行源码；文件主题为 历史位置索引、量化。 | 函数：`PrepareNonOffloadRows`, `FinalizeStandardCounts` |
| [fused_quant_lightning_indexer_manage_common.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage_common.h) | 量化索引与缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引、公共逻辑、量化。 | 类型：`LI_LAYOUT`, `LIType`, `RunInfo` |
| [fused_quant_lightning_indexer_manage_constants.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage_constants.h) | 量化索引与缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [fused_quant_lightning_indexer_manage_kernel.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage_kernel.h) | 量化索引与缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`TempLoopInfo`, `LIMtpC8Preload` |
| [fused_quant_lightning_indexer_manage_service_cube.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage_service_cube.h) | 量化索引与缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`LIMatmulMtpC8` |
| [fused_quant_lightning_indexer_manage_service_vector.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage_service_vector.h) | 量化索引与缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`LIVectorMtpC8`, `LIMtpC8Common` |
| [fused_quant_lightning_indexer_manage_template_tiling_key.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage_template_tiling_key.h) | 量化索引与缓存管理融合 的kernel 分派使用的 tiling key 定义；文件主题为 历史位置索引、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [fused_quant_lightning_indexer_manage_union.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage_union.h) | 量化索引与缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`MtpC8MissUnion` |
| [fused_quant_lightning_indexer_manage_vector.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage_vector.h) | 量化索引与缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 函数：`CopyIn`, `CopyOut` |
| [fused_quant_lightning_indexer_manage_workspace.h](../../../csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/fused_quant_lightning_indexer_manage_workspace.h) | 量化索引与缓存管理融合 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 函数：`LiWorkspaceBytes`, `Pair0Offset` |

## `csrc/attention/fused_scatter_copy_sparse_flash_attention/`

缓存 scatter/copy 与稀疏 Attention 融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：FusedScatterCopySparseFlashAttention |
| [fused_scatter_copy_sparse_flash_attention_torch_adpt.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/fused_scatter_copy_sparse_flash_attention_torch_adpt.h) | 注意力、分散写入、复制 的公共接口/实现，供所在算子工程复用。 | 函数：`npu_fused_scatter_copy_sparse_flash_attention` |

## `csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [fused_scatter_copy_sparse_flash_attention_def.cpp](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/fused_scatter_copy_sparse_flash_attention_def.cpp) | 缓存 scatter/copy 与稀疏 Attention 融合 的主机算子定义、支持属性和注册。 | 注册：`FusedScatterCopySparseFlashAttention` |
| [fused_scatter_copy_sparse_flash_attention_infershape.cpp](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/fused_scatter_copy_sparse_flash_attention_infershape.cpp) | 缓存 scatter/copy 与稀疏 Attention 融合 的主机侧辅助接口或实现。 | 函数：`InferShapeFusedScatterCopySparseFlashAttention`, `InferDtypeFusedScatterCopySparseFlashAttention` |
| [fused_scatter_copy_sparse_flash_attention_tiling.cpp](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/fused_scatter_copy_sparse_flash_attention_tiling.cpp) | 缓存 scatter/copy 与稀疏 Attention 融合 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`IsShape`, `CheckFusedInputs` |
| [fused_scatter_copy_sparse_flash_attention_tiling.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/fused_scatter_copy_sparse_flash_attention_tiling.h) | 缓存 scatter/copy 与稀疏 Attention 融合 的主机侧 tiling 参数/分块策略声明。 | 类型：`FusedScatterCopySparseFlashAttentionCompileInfo` |
| [fused_scatter_copy_sparse_flash_attention_tiling_impl.cpp](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/fused_scatter_copy_sparse_flash_attention_tiling_impl.cpp) | 缓存 scatter/copy 与稀疏 Attention 融合 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`GetShapeStr`, `FusedScatterCopySparseFlashAttentionDataTypeToSerialString` |
| [fused_scatter_copy_sparse_flash_attention_tiling_impl.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/fused_scatter_copy_sparse_flash_attention_tiling_impl.h) | 缓存 scatter/copy 与稀疏 Attention 融合 的主机侧 tiling 参数/分块策略声明。 | 类型：`FusedScatterCopySparseFlashAttentionLayout`, `FusedScatterCopySparseFlashAttentionTilingShapeCompareParam`, `FusedScatterCopySparseFlashAttentionKvStorageMode` |

## `csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_fused_scatter_copy_sparse_flash_attention.cpp](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/op_api/aclnn_fused_scatter_copy_sparse_flash_attention.cpp) | 缓存 scatter/copy 与稀疏 Attention 融合 的主机侧辅助接口或实现。 | 函数：`aclnnInnerFusedScatterCopySparseFlashAttentionGetWorkspaceSize`, `aclnnInnerFusedScatterCopySparseFlashAttention` |
| [aclnn_fused_scatter_copy_sparse_flash_attention.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_host/op_api/aclnn_fused_scatter_copy_sparse_flash_attention.h) | 缓存 scatter/copy 与稀疏 Attention 融合 的主机侧辅助接口或实现。 | 函数：`aclnnFusedScatterCopySparseFlashAttentionGetWorkspaceSize`, `aclnnFusedScatterCopySparseFlashAttention` |

## `csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [sparse_flash_attention_impl/](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/sparse_flash_attention_impl) | 注意力 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [first_fill_scatter_copy_kernel.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/first_fill_scatter_copy_kernel.h) | 缓存 scatter/copy 与稀疏 Attention 融合 的类型/模板与设备实现头文件；文件主题为 分散写入、复制。 | 类型：`FirstFillScatterCopyKernel`, `CopyAddress` |
| [fused_scatter_copy_sparse_flash_attention.cpp](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/fused_scatter_copy_sparse_flash_attention.cpp) | 缓存 scatter/copy 与稀疏 Attention 融合 的设备执行源码；文件主题为 注意力、分散写入、复制。 | 函数：`RunFirstFillScatterCopy`, `IsBatchFirstFill` |

## `csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/sparse_flash_attention_impl/`

注意力 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fused_scatter_copy_sparse_flash_attention_common.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/sparse_flash_attention_impl/fused_scatter_copy_sparse_flash_attention_common.h) | 缓存 scatter/copy 与稀疏 Attention 融合 的类型/模板与设备实现头文件；文件主题为 注意力、分散写入、公共逻辑。 | 类型：`SFA_LAYOUT`, `SFAType`, `RunInfo` |
| [fused_scatter_copy_sparse_flash_attention_kernel_mla.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/sparse_flash_attention_impl/fused_scatter_copy_sparse_flash_attention_kernel_mla.h) | 缓存 scatter/copy 与稀疏 Attention 融合 的类型/模板与设备实现头文件；文件主题为 注意力、分散写入、复制。 | 类型：`TempLoopInfo`, `FusedScatterCopySparseFlashAttentionMla` |
| [fused_scatter_copy_sparse_flash_attention_service_cube_mla.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/sparse_flash_attention_impl/fused_scatter_copy_sparse_flash_attention_service_cube_mla.h) | 缓存 scatter/copy 与稀疏 Attention 融合 的类型/模板与设备实现头文件；文件主题为 注意力、分散写入、复制。 | 类型：`PAShape`, `Position`, `SFAMatmulService` |
| [fused_scatter_copy_sparse_flash_attention_service_vector_mla.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/sparse_flash_attention_impl/fused_scatter_copy_sparse_flash_attention_service_vector_mla.h) | 缓存 scatter/copy 与稀疏 Attention 融合 的类型/模板与设备实现头文件；文件主题为 注意力、分散写入、复制。 | 类型：`SFAVectorService`, `ConstInfo`, `GatherBatch` |
| [fused_scatter_copy_sparse_flash_attention_template_tiling_key.h](../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/sparse_flash_attention_impl/fused_scatter_copy_sparse_flash_attention_template_tiling_key.h) | 缓存 scatter/copy 与稀疏 Attention 融合 的kernel 分派使用的 tiling key 定义；文件主题为 注意力、分散写入、复制。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/fused_sparse_attention_overlap/`

稀疏注意力融合重叠 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/fused_sparse_attention_overlap/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/fused_sparse_attention_overlap/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/fused_sparse_attention_overlap/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：FusedSparseAttentionOverlap |
| [fused_sparse_attention_overlap_torch_adpt.h](../../../csrc/attention/fused_sparse_attention_overlap/fused_sparse_attention_overlap_torch_adpt.h) | 注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`CheckPositiveShape`, `MaybeQueryRope` |

## `csrc/attention/fused_sparse_attention_overlap/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/fused_sparse_attention_overlap/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [fused_sparse_attention_overlap_def.cpp](../../../csrc/attention/fused_sparse_attention_overlap/op_host/fused_sparse_attention_overlap_def.cpp) | 稀疏注意力融合重叠 的主机算子定义、支持属性和注册。 | 注册：`FusedSparseAttentionOverlap` |
| [fused_sparse_attention_overlap_proto.cpp](../../../csrc/attention/fused_sparse_attention_overlap/op_host/fused_sparse_attention_overlap_proto.cpp) | 稀疏注意力融合重叠 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShapeFusedSparseAttentionOverlap`, `InferDataTypeFusedSparseAttentionOverlap` |
| [fused_sparse_attention_overlap_tiling.cpp](../../../csrc/attention/fused_sparse_attention_overlap/op_host/fused_sparse_attention_overlap_tiling.cpp) | 稀疏注意力融合重叠 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`FusedSparseAttentionOverlapCompileInfo` |
| [fused_sparse_attention_overlap_tiling.h](../../../csrc/attention/fused_sparse_attention_overlap/op_host/fused_sparse_attention_overlap_tiling.h) | 稀疏注意力融合重叠 的主机侧 tiling 参数/分块策略声明。 | 类型：`FusedSparseAttentionOverlapLayout`, `FusedSparseAttentionOverlapTilingShapeCompareParam`, `FusedOverlapKvStorageMode` |

## `csrc/attention/fused_sparse_attention_overlap/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch22/](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22) | arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fused_sparse_attention_overlap.cpp](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/fused_sparse_attention_overlap.cpp) | 稀疏注意力融合重叠 的设备执行源码；文件主题为 注意力。 | 函数：`fused_sparse_attention_overlap` |
| [fused_sparse_attention_overlap_common.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/fused_sparse_attention_overlap_common.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件；文件主题为 注意力、公共逻辑。 | 类型：`FusedSparseAttentionOverlapLayout`, `FusedSparseAttentionOverlapType`, `RunInfo` |
| [fused_sparse_attention_overlap_template_tiling_key.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/fused_sparse_attention_overlap_template_tiling_key.h) | 稀疏注意力融合重叠 的kernel 分派使用的 tiling key 定义；文件主题为 注意力。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/`

arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fused_sparse_attention_overlap_kernel_mla.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/fused_sparse_attention_overlap_kernel_mla.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`TempLoopInfo`, `FusedSparseAttentionOverlapMla` |
| [fused_sparse_attention_overlap_service_cube_mla.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/fused_sparse_attention_overlap_service_cube_mla.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`PAShape`, `Position`, `FusedSparseAttentionOverlapMatmulService` |
| [fused_sparse_attention_overlap_service_vector_mla.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/fused_sparse_attention_overlap_service_vector_mla.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`FusedSparseAttentionOverlapVectorService`, `ConstInfo` |

## `csrc/attention/fused_sparse_attention_overlap/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fused_sparse_attention_overlap_common_arch35.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch35/fused_sparse_attention_overlap_common_arch35.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件；文件主题为 注意力、公共逻辑。 | 类型：`FusedSparseAttentionOverlapLayoutArch35`, `FusedSparseAttentionOverlapTemplateModeArch35` |
| [fused_sparse_attention_overlap_kernel_mla.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch35/fused_sparse_attention_overlap_kernel_mla.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`FusedSparseAttentionOverlapKernelMla` |
| [fused_sparse_attention_overlap_kvcache.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch35/fused_sparse_attention_overlap_kvcache.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件；文件主题为 注意力、缓存、KV 缓存。 | 函数：`CalculateQueryOffset`, `GetSingleCoreParam` |
| [fused_sparse_attention_overlap_service_cube_mla.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch35/fused_sparse_attention_overlap_service_cube_mla.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`FusedSparseAttentionOverlapMatmulService`, `FusedSparseAttentionOverlapMatmulServiceDummy`, `CubeBlockTraits` |
| [fused_sparse_attention_overlap_service_vector_mla.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch35/fused_sparse_attention_overlap_service_vector_mla.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`FusedSparseAttentionOverlapVectorService`, `FusedSparseAttentionOverlapVectorServiceDummy` |
| [util_regbase.h](../../../csrc/attention/fused_sparse_attention_overlap/op_kernel/arch35/util_regbase.h) | 稀疏注意力融合重叠 的类型/模板与设备实现头文件。 | 类型：`RunParamStr`, `RunInfo`, `ConstInfo` |

## `csrc/attention/indexer_compress_epilog_v2/`

压缩 Indexer 后处理 V2 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/indexer_compress_epilog_v2/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/indexer_compress_epilog_v2/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/indexer_compress_epilog_v2/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/attention/indexer_compress_epilog_v2/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/indexer_compress_epilog_v2/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [indexer_compress_epilog_v2_def.cpp](../../../csrc/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_def.cpp) | 压缩 Indexer 后处理 V2 的主机算子定义、支持属性和注册。 | 注册：`IndexerCompressEpilogV2` |
| [indexer_compress_epilog_v2_proto.cpp](../../../csrc/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_proto.cpp) | 压缩 Indexer 后处理 V2 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShape4IndexerCompressEpilogV2`, `InferDtype4IndexerCompressEpilogV2` |
| [indexer_compress_epilog_v2_tiling.cpp](../../../csrc/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_tiling.cpp) | 压缩 Indexer 后处理 V2 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`CeilDiv`, `DownAlign` |
| [indexer_compress_epilog_v2_tiling.h](../../../csrc/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_tiling.h) | 压缩 Indexer 后处理 V2 的主机侧 tiling 参数/分块策略声明。 | 类型：`TilingRequiredParaInfo`, `TilingOptionalParaInfo`, `IndexerCompressEpilogV2CompileInfo` |

## `csrc/attention/indexer_compress_epilog_v2/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [indexer_compress_epilog_v2.cpp](../../../csrc/attention/indexer_compress_epilog_v2/op_kernel/indexer_compress_epilog_v2.cpp) | 压缩 Indexer 后处理 V2 的设备执行源码；文件主题为 历史位置索引。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [indexer_compress_epilog_v2_base.h](../../../csrc/attention/indexer_compress_epilog_v2/op_kernel/indexer_compress_epilog_v2_base.h) | 压缩 Indexer 后处理 V2 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 函数：`CeilDiv`, `CeilAlign` |
| [indexer_compress_epilog_v2_multi_row.h](../../../csrc/attention/indexer_compress_epilog_v2/op_kernel/indexer_compress_epilog_v2_multi_row.h) | 压缩 Indexer 后处理 V2 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 类型：`IndexerCompressEpilogV2MultiRow` |
| [indexer_compress_epilog_v2_single_row.h](../../../csrc/attention/indexer_compress_epilog_v2/op_kernel/indexer_compress_epilog_v2_single_row.h) | 压缩 Indexer 后处理 V2 的类型/模板与设备实现头文件；文件主题为 历史位置索引。 | 类型：`IndexerCompressEpilogV2SingleRow` |

## `csrc/attention/inplace_partial_rotary_mul/`

部分通道原位位置旋转 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/inplace_partial_rotary_mul/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/inplace_partial_rotary_mul/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/attention/inplace_partial_rotary_mul/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/inplace_partial_rotary_mul/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [inplace_partial_rotary_mul_a3_tiling.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_a3_tiling.cpp) | 部分通道原位位置旋转 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`InplacePartialRotaryMulTiling` |
| [inplace_partial_rotary_mul_def.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_def.cpp) | 部分通道原位位置旋转 的主机算子定义、支持属性和注册。 | 注册：`InplacePartialRotaryMul` |
| [inplace_partial_rotary_mul_proto.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_proto.cpp) | 部分通道原位位置旋转 的算子原型、输入输出或类型/shape 推断注册。 | 注册：`InplacePartialRotaryMul` |
| [inplace_partial_rotary_mul_tiling.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_tiling.cpp) | 部分通道原位位置旋转 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`Tiling4RotaryPositionEmbedding`, `TilingPrepareForRotaryPositionEmbedding` |
| [inplace_partial_rotary_mul_tiling.h](../../../csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_tiling.h) | 部分通道原位位置旋转 的主机侧 tiling 参数/分块策略声明。 | 类型：`RotaryPositionEmbeddingCompileInfo`, `AiCoreParams`, `RopeLayout` |
| [rope_regbase_tiling_a_and_b.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_host/rope_regbase_tiling_a_and_b.cpp) | 部分通道原位位置旋转 的主机侧 tiling 参数/分块策略计算实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [rope_regbase_tiling_ab.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_host/rope_regbase_tiling_ab.cpp) | 部分通道原位位置旋转 的主机侧 tiling 参数/分块策略计算实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [rope_regbase_tiling_aba_and_ba.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_host/rope_regbase_tiling_aba_and_ba.cpp) | 部分通道原位位置旋转 的主机侧 tiling 参数/分块策略计算实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [rope_regbase_tiling_bab.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_host/rope_regbase_tiling_bab.cpp) | 部分通道原位位置旋转 的主机侧 tiling 参数/分块策略计算实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [rope_regbase_tiling_base.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_host/rope_regbase_tiling_base.cpp) | 部分通道原位位置旋转 的主机侧 tiling 参数/分块策略计算实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/inplace_partial_rotary_mul/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [apply_rotary_pos_emb_common.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/apply_rotary_pos_emb_common.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 位置旋转、公共逻辑。 | 类型：`RotaryPosEmbeddingMode` |
| [inplace_partial_rotary_mul.cpp](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/inplace_partial_rotary_mul.cpp) | 部分通道原位位置旋转 的设备执行源码；文件主题为 位置旋转。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [inplace_partial_rotary_mul.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/inplace_partial_rotary_mul.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 位置旋转。 | 类型：`InplacePartialRotaryMulABA` |
| [inplace_partial_rotary_mul_common.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/inplace_partial_rotary_mul_common.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 位置旋转、公共逻辑。 | 函数：`Ceil`, `Align` |
| [rotary_position_embedding_reg_a_and_b.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotary_position_embedding_reg_a_and_b.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 向量查表、位置旋转。 | 类型：`RotaryPositionEmbeddingAAndB` |
| [rotary_position_embedding_reg_a_and_b_mixed.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotary_position_embedding_reg_a_and_b_mixed.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 向量查表、位置旋转。 | 类型：`RotaryPositionEmbeddingAAndBMixed` |
| [rotary_position_embedding_reg_ab.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotary_position_embedding_reg_ab.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 向量查表、位置旋转。 | 类型：`RotaryPositionEmbeddingAB` |
| [rotary_position_embedding_reg_ab_mixed.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotary_position_embedding_reg_ab_mixed.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 向量查表、位置旋转。 | 类型：`RotaryPositionEmbeddingABMixed` |
| [rotary_position_embedding_reg_aba_and_ba.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotary_position_embedding_reg_aba_and_ba.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 向量查表、位置旋转。 | 类型：`RotaryPositionEmbeddingABAAndBA` |
| [rotary_position_embedding_reg_aba_and_ba_mixed.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotary_position_embedding_reg_aba_and_ba_mixed.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 向量查表、位置旋转。 | 类型：`RotaryPositionEmbeddingABAAndBAMixed` |
| [rotary_position_embedding_reg_bab.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotary_position_embedding_reg_bab.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 向量查表、位置旋转。 | 类型：`RotaryPositionEmbeddingBAB` |
| [rotary_position_embedding_reg_bab_mixed.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotary_position_embedding_reg_bab_mixed.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 向量查表、位置旋转。 | 类型：`RotaryPositionEmbeddingBABMixed` |
| [rotate_half.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_half.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件。 | 类型：`RotateHalf` |
| [rotate_half_base.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_half_base.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件。 | 类型：`RotateHalfBase` |
| [rotate_half_bf16.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_half_bf16.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件。 | 类型：`RotateHalfBf16` |
| [rotate_interleaved_common.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_common.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`BroadCastTriToBsnd`, `BroadCastTriToB1nd` |
| [rotate_interleaved_split_bs.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_split_bs.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件。 | 类型：`InterleavedSplitBS` |
| [rotate_interleaved_split_bs_pad.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_split_bs_pad.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 补齐。 | 类型：`InterleavedSplitBSPad` |
| [rotate_interleaved_split_bsn.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_split_bsn.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件。 | 类型：`InterleavedSplitBSN` |
| [rotate_interleaved_split_bsn_pad.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_split_bsn_pad.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 补齐。 | 类型：`InterleavedSplitBSNPad` |
| [rotate_interleaved_split_s.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_split_s.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件。 | 类型：`InterleavedSplitS` |
| [rotate_interleaved_split_s_pad.h](../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_split_s_pad.h) | 部分通道原位位置旋转 的类型/模板与设备实现头文件；文件主题为 补齐。 | 类型：`InterleavedSplitSPad` |

## `csrc/attention/k2q_csr/`

K 到 Q 的 CSR 索引组织 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [k2q_csr_common/](../../../csrc/attention/k2q_csr/k2q_csr_common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [k2q_csr_hist/](../../../csrc/attention/k2q_csr/k2q_csr_hist) | k2q_csr_hist 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [k2q_csr_meta/](../../../csrc/attention/k2q_csr/k2q_csr_meta) | k2q_csr_meta 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [k2q_csr_row_prefix/](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix) | k2q_csr_row_prefix 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [k2q_csr_scatter/](../../../csrc/attention/k2q_csr/k2q_csr_scatter) | 分散写入 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [k2q_csr_tile_prefix/](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix) | k2q_csr_tile_prefix 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/k2q_csr/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/k2q_csr/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：k2q_csr (KV Cache) |
| [k2q_csr_torch_adpt.h](../../../csrc/attention/k2q_csr/k2q_csr_torch_adpt.h) | k2q_csr_torch_adpt 的公共接口/实现，供所在算子工程复用。 | 函数：`k2q_csr_get_aiv_num_from_device`, `k2q_csr_cuda_like_group_count` |

## `csrc/attention/k2q_csr/k2q_csr_common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/k2q_csr/k2q_csr_common/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

## `csrc/attention/k2q_csr/k2q_csr_common/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/k2q_csr/k2q_csr_common/op_host/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_tiling_common.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_host/k2q_csr_tiling_common.h) | K 到 Q 的 CSR 索引组织 的主机侧 tiling 参数/分块策略声明。 | 类型：`Stage` |

## `csrc/attention/k2q_csr/k2q_csr_common/op_host/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_tiling_arch35.cpp](../../../csrc/attention/k2q_csr/k2q_csr_common/op_host/arch35/k2q_csr_tiling_arch35.cpp) | K 到 Q 的 CSR 索引组织 的主机侧 tiling 参数/分块策略计算实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_tiling_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_host/arch35/k2q_csr_tiling_arch35.h) | K 到 Q 的 CSR 索引组织 的主机侧 tiling 参数/分块策略声明。 | 函数：`IsArch35Ub`, `IsArch35Soc` |

## `csrc/attention/k2q_csr/k2q_csr_common/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_common.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/k2q_csr_common.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`DivCeil`, `Align32` |
| [k2q_csr_mc.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/k2q_csr_mc.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrPipelineMc` |
| [k2q_csr_prefix.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/k2q_csr_prefix.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrRowPrefixKernel`, `K2qCsrTilePrefixKernel` |
| [k2q_csr_scratch.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/k2q_csr_scratch.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 函数：`TileElems`, `Split` |
| [k2q_csr_stage_meta.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/k2q_csr_stage_meta.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrStageMeta` |
| [k2q_csr_tiling.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/k2q_csr_tiling.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrTilingData` |

## `csrc/attention/k2q_csr/k2q_csr_common/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_simt_common_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/arch35/k2q_csr_simt_common_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_simt_hist_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/arch35/k2q_csr_simt_hist_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrSimtHist` |
| [k2q_csr_simt_scatter_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_common/op_kernel/arch35/k2q_csr_simt_scatter_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`K2qCsrSimtScatter` |

## `csrc/attention/k2q_csr/k2q_csr_hist/`

k2q_csr_hist 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

## `csrc/attention/k2q_csr/k2q_csr_hist/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [k2q_csr_hist_def.cpp](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_host/k2q_csr_hist_def.cpp) | K 到 Q 的 CSR 索引组织 的主机算子定义、支持属性和注册。 | 注册：`K2qCsrHist` |
| [k2q_csr_hist_infershape.cpp](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_host/k2q_csr_hist_infershape.cpp) | K 到 Q 的 CSR 索引组织 的主机侧辅助接口或实现。 | 函数：`InferShapeK2qCsrHist` |
| [k2q_csr_hist_tiling.cpp](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_host/k2q_csr_hist_tiling.cpp) | K 到 Q 的 CSR 索引组织 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`K2qCsrHistCompileInfo` |

## `csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_hist.cpp](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/k2q_csr_hist.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_hist_apt.cpp](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/k2q_csr_hist_apt.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_common.h](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/k2q_csr_common.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`DivCeil`, `Align32` |
| [k2q_csr_mc.h](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/k2q_csr_mc.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrPipelineMc` |
| [k2q_csr_prefix.h](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/k2q_csr_prefix.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrRowPrefixKernel`, `K2qCsrTilePrefixKernel` |
| [k2q_csr_scratch.h](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/k2q_csr_scratch.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 函数：`TileElems`, `Split` |
| [k2q_csr_stage_meta.h](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/k2q_csr_stage_meta.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrStageMeta` |
| [k2q_csr_tiling.h](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/k2q_csr_tiling.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrTilingData` |

## `csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_simt_common_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/arch35/k2q_csr_simt_common_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_simt_hist_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/arch35/k2q_csr_simt_hist_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrSimtHist` |
| [k2q_csr_simt_scatter_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_hist/op_kernel/common/arch35/k2q_csr_simt_scatter_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`K2qCsrSimtScatter` |

## `csrc/attention/k2q_csr/k2q_csr_meta/`

k2q_csr_meta 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

## `csrc/attention/k2q_csr/k2q_csr_meta/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [k2q_csr_meta_def.cpp](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_host/k2q_csr_meta_def.cpp) | K 到 Q 的 CSR 索引组织 的主机算子定义、支持属性和注册。 | 注册：`K2qCsrMeta` |
| [k2q_csr_meta_infershape.cpp](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_host/k2q_csr_meta_infershape.cpp) | K 到 Q 的 CSR 索引组织 的主机侧辅助接口或实现。 | 函数：`InferShapeK2qCsrMeta` |
| [k2q_csr_meta_tiling.cpp](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_host/k2q_csr_meta_tiling.cpp) | K 到 Q 的 CSR 索引组织 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`K2qCsrMetaCompileInfo` |

## `csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_meta.cpp](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/k2q_csr_meta.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_meta_apt.cpp](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/k2q_csr_meta_apt.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_common.h](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/k2q_csr_common.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`DivCeil`, `Align32` |
| [k2q_csr_mc.h](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/k2q_csr_mc.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrPipelineMc` |
| [k2q_csr_prefix.h](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/k2q_csr_prefix.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrRowPrefixKernel`, `K2qCsrTilePrefixKernel` |
| [k2q_csr_scratch.h](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/k2q_csr_scratch.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 函数：`TileElems`, `Split` |
| [k2q_csr_stage_meta.h](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/k2q_csr_stage_meta.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrStageMeta` |
| [k2q_csr_tiling.h](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/k2q_csr_tiling.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrTilingData` |

## `csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_simt_common_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/arch35/k2q_csr_simt_common_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_simt_hist_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/arch35/k2q_csr_simt_hist_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrSimtHist` |
| [k2q_csr_simt_scatter_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_meta/op_kernel/common/arch35/k2q_csr_simt_scatter_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`K2qCsrSimtScatter` |

## `csrc/attention/k2q_csr/k2q_csr_row_prefix/`

k2q_csr_row_prefix 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

## `csrc/attention/k2q_csr/k2q_csr_row_prefix/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [k2q_csr_row_prefix_def.cpp](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_host/k2q_csr_row_prefix_def.cpp) | K 到 Q 的 CSR 索引组织 的主机算子定义、支持属性和注册。 | 注册：`K2qCsrRowPrefix` |
| [k2q_csr_row_prefix_infershape.cpp](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_host/k2q_csr_row_prefix_infershape.cpp) | K 到 Q 的 CSR 索引组织 的主机侧辅助接口或实现。 | 函数：`InferShapeK2qCsrRowPrefix` |
| [k2q_csr_row_prefix_tiling.cpp](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_host/k2q_csr_row_prefix_tiling.cpp) | K 到 Q 的 CSR 索引组织 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`K2qCsrRowPrefixCompileInfo` |

## `csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_row_prefix.cpp](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/k2q_csr_row_prefix.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_row_prefix_apt.cpp](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/k2q_csr_row_prefix_apt.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_common.h](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/k2q_csr_common.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`DivCeil`, `Align32` |
| [k2q_csr_mc.h](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/k2q_csr_mc.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrPipelineMc` |
| [k2q_csr_prefix.h](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/k2q_csr_prefix.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrRowPrefixKernel`, `K2qCsrTilePrefixKernel` |
| [k2q_csr_scratch.h](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/k2q_csr_scratch.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 函数：`TileElems`, `Split` |
| [k2q_csr_stage_meta.h](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/k2q_csr_stage_meta.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrStageMeta` |
| [k2q_csr_tiling.h](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/k2q_csr_tiling.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrTilingData` |

## `csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_simt_common_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/arch35/k2q_csr_simt_common_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_simt_hist_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/arch35/k2q_csr_simt_hist_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrSimtHist` |
| [k2q_csr_simt_scatter_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_row_prefix/op_kernel/common/arch35/k2q_csr_simt_scatter_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`K2qCsrSimtScatter` |

## `csrc/attention/k2q_csr/k2q_csr_scatter/`

分散写入 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

## `csrc/attention/k2q_csr/k2q_csr_scatter/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [k2q_csr_scatter_def.cpp](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_host/k2q_csr_scatter_def.cpp) | K 到 Q 的 CSR 索引组织 的主机算子定义、支持属性和注册。 | 注册：`K2qCsrScatter` |
| [k2q_csr_scatter_infershape.cpp](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_host/k2q_csr_scatter_infershape.cpp) | K 到 Q 的 CSR 索引组织 的主机侧辅助接口或实现。 | 函数：`InferShapeK2qCsrScatter` |
| [k2q_csr_scatter_tiling.cpp](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_host/k2q_csr_scatter_tiling.cpp) | K 到 Q 的 CSR 索引组织 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`K2qCsrScatterCompileInfo` |

## `csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_scatter.cpp](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/k2q_csr_scatter.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码；文件主题为 分散写入。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_scatter_apt.cpp](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/k2q_csr_scatter_apt.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码；文件主题为 分散写入。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_common.h](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/k2q_csr_common.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`DivCeil`, `Align32` |
| [k2q_csr_mc.h](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/k2q_csr_mc.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrPipelineMc` |
| [k2q_csr_prefix.h](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/k2q_csr_prefix.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrRowPrefixKernel`, `K2qCsrTilePrefixKernel` |
| [k2q_csr_scratch.h](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/k2q_csr_scratch.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 函数：`TileElems`, `Split` |
| [k2q_csr_stage_meta.h](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/k2q_csr_stage_meta.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrStageMeta` |
| [k2q_csr_tiling.h](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/k2q_csr_tiling.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrTilingData` |

## `csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_simt_common_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/arch35/k2q_csr_simt_common_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_simt_hist_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/arch35/k2q_csr_simt_hist_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrSimtHist` |
| [k2q_csr_simt_scatter_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_scatter/op_kernel/common/arch35/k2q_csr_simt_scatter_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`K2qCsrSimtScatter` |

## `csrc/attention/k2q_csr/k2q_csr_tile_prefix/`

k2q_csr_tile_prefix 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

## `csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [k2q_csr_tile_prefix_def.cpp](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_host/k2q_csr_tile_prefix_def.cpp) | K 到 Q 的 CSR 索引组织 的主机算子定义、支持属性和注册。 | 注册：`K2qCsrTilePrefix` |
| [k2q_csr_tile_prefix_infershape.cpp](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_host/k2q_csr_tile_prefix_infershape.cpp) | K 到 Q 的 CSR 索引组织 的主机侧辅助接口或实现。 | 函数：`InferShapeK2qCsrTilePrefix` |
| [k2q_csr_tile_prefix_tiling.cpp](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_host/k2q_csr_tile_prefix_tiling.cpp) | K 到 Q 的 CSR 索引组织 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`K2qCsrTilePrefixCompileInfo` |

## `csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_tile_prefix.cpp](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/k2q_csr_tile_prefix.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_tile_prefix_apt.cpp](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/k2q_csr_tile_prefix_apt.cpp) | K 到 Q 的 CSR 索引组织 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_common.h](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/k2q_csr_common.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`DivCeil`, `Align32` |
| [k2q_csr_mc.h](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/k2q_csr_mc.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrPipelineMc` |
| [k2q_csr_prefix.h](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/k2q_csr_prefix.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrRowPrefixKernel`, `K2qCsrTilePrefixKernel` |
| [k2q_csr_scratch.h](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/k2q_csr_scratch.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 函数：`TileElems`, `Split` |
| [k2q_csr_stage_meta.h](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/k2q_csr_stage_meta.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrStageMeta` |
| [k2q_csr_tiling.h](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/k2q_csr_tiling.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrTilingData` |

## `csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [k2q_csr_simt_common_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/arch35/k2q_csr_simt_common_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [k2q_csr_simt_hist_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/arch35/k2q_csr_simt_hist_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件。 | 类型：`K2qCsrSimtHist` |
| [k2q_csr_simt_scatter_arch35.h](../../../csrc/attention/k2q_csr/k2q_csr_tile_prefix/op_kernel/common/arch35/k2q_csr_simt_scatter_arch35.h) | K 到 Q 的 CSR 索引组织 的类型/模板与设备实现头文件；文件主题为 分散写入。 | 类型：`K2qCsrSimtScatter` |

## `csrc/attention/kda_gate_cumsum_vllm/`

KDA 门控累积 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/kda_gate_cumsum_vllm/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/kda_gate_cumsum_vllm/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kda_gate_cumsum_vllm/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kda_gate_cumsum_vllm_torch_adpt.h](../../../csrc/attention/kda_gate_cumsum_vllm/kda_gate_cumsum_vllm_torch_adpt.h) | 累积求和、门控、KDA 线性注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`kda_gate_cumsum_vllm` |

## `csrc/attention/kda_gate_cumsum_vllm/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/kda_gate_cumsum_vllm/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kda_gate_cumsum_vllm/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kda_gate_cumsum_vllm_def.cpp](../../../csrc/attention/kda_gate_cumsum_vllm/op_host/kda_gate_cumsum_vllm_def.cpp) | KDA 门控累积 的主机算子定义、支持属性和注册。 | 注册：`KdaGateCumsumVllm` |
| [kda_gate_cumsum_vllm_tiling.cpp](../../../csrc/attention/kda_gate_cumsum_vllm/op_host/kda_gate_cumsum_vllm_tiling.cpp) | KDA 门控累积 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`KdaGateLayout` |
| [kda_gate_cumsum_vllm_tiling.h](../../../csrc/attention/kda_gate_cumsum_vllm/op_host/kda_gate_cumsum_vllm_tiling.h) | KDA 门控累积 的主机侧 tiling 参数/分块策略声明。 | 类型：`KdaGateCumsumVllmCompileInfo` |

## `csrc/attention/kda_gate_cumsum_vllm/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_kda_gate_cumsum_vllm.cpp](../../../csrc/attention/kda_gate_cumsum_vllm/op_host/op_api/aclnn_kda_gate_cumsum_vllm.cpp) | KDA 门控累积 的主机侧辅助接口或实现。 | 类型：`KdaGateLayout` |
| [aclnn_kda_gate_cumsum_vllm.h](../../../csrc/attention/kda_gate_cumsum_vllm/op_host/op_api/aclnn_kda_gate_cumsum_vllm.h) | KDA 门控累积 的主机侧辅助接口或实现。 | 函数：`aclnnKdaGateCumsumVllmGetWorkspaceSize` |
| [kda_gate_cumsum_vllm.cpp](../../../csrc/attention/kda_gate_cumsum_vllm/op_host/op_api/kda_gate_cumsum_vllm.cpp) | KDA 门控累积 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kda_gate_cumsum_vllm.h](../../../csrc/attention/kda_gate_cumsum_vllm/op_host/op_api/kda_gate_cumsum_vllm.h) | KDA 门控累积 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/kda_gate_cumsum_vllm/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kda_gate_cumsum_vllm.cpp](../../../csrc/attention/kda_gate_cumsum_vllm/op_kernel/kda_gate_cumsum_vllm.cpp) | KDA 门控累积 的设备执行源码；文件主题为 累积求和、门控、KDA 线性注意力。 | 类型：`KdaGateCumsumVllmKernel` |
| [kda_gate_cumsum_vllm_kernel.h](../../../csrc/attention/kda_gate_cumsum_vllm/op_kernel/kda_gate_cumsum_vllm_kernel.h) | KDA 门控累积 的类型/模板与设备实现头文件；文件主题为 累积求和、门控、KDA 线性注意力。 | 类型：`KdaGateCumsumVllmKernel` |

## `csrc/attention/kda_layout_swap12/`

KDA 维度布局交换 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/kda_layout_swap12/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/kda_layout_swap12/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kda_layout_swap12/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kda_layout_swap12_torch_adpt.h](../../../csrc/attention/kda_layout_swap12/kda_layout_swap12_torch_adpt.h) | 内存布局、KDA 线性注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`kda_layout_swap12` |

## `csrc/attention/kda_layout_swap12/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/kda_layout_swap12/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kda_layout_swap12/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kda_layout_swap12_def.cpp](../../../csrc/attention/kda_layout_swap12/op_host/kda_layout_swap12_def.cpp) | KDA 维度布局交换 的主机算子定义、支持属性和注册。 | 注册：`KdaLayoutSwap12` |
| [kda_layout_swap12_tiling.cpp](../../../csrc/attention/kda_layout_swap12/op_host/kda_layout_swap12_tiling.cpp) | KDA 维度布局交换 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`DTypeCode`, `Tiling4KdaLayoutSwap12` |
| [kda_layout_swap12_tiling.h](../../../csrc/attention/kda_layout_swap12/op_host/kda_layout_swap12_tiling.h) | KDA 维度布局交换 的主机侧 tiling 参数/分块策略声明。 | 类型：`KdaLayoutSwap12CompileInfo` |

## `csrc/attention/kda_layout_swap12/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_kda_layout_swap12.cpp](../../../csrc/attention/kda_layout_swap12/op_host/op_api/aclnn_kda_layout_swap12.cpp) | KDA 维度布局交换 的主机侧辅助接口或实现。 | 函数：`KdaLayoutSwapDataContiguous`, `KdaLayoutSwapSameShape` |
| [aclnn_kda_layout_swap12.h](../../../csrc/attention/kda_layout_swap12/op_host/op_api/aclnn_kda_layout_swap12.h) | KDA 维度布局交换 的主机侧辅助接口或实现。 | 函数：`aclnnKdaLayoutSwap12GetWorkspaceSize` |
| [kda_layout_swap12.cpp](../../../csrc/attention/kda_layout_swap12/op_host/op_api/kda_layout_swap12.cpp) | KDA 维度布局交换 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kda_layout_swap12.h](../../../csrc/attention/kda_layout_swap12/op_host/op_api/kda_layout_swap12.h) | KDA 维度布局交换 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/kda_layout_swap12/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kda_layout_swap12.cpp](../../../csrc/attention/kda_layout_swap12/op_kernel/kda_layout_swap12.cpp) | KDA 维度布局交换 的设备执行源码；文件主题为 内存布局、KDA 线性注意力。 | 类型：`KdaLayoutSwap12Kernel` |

## `csrc/attention/kv_compress_epilog/`

压缩 KV 后处理 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/kv_compress_epilog/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/kv_compress_epilog/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_compress_epilog/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/attention/kv_compress_epilog/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_compress_epilog/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kv_compress_epilog_def.cpp](../../../csrc/attention/kv_compress_epilog/op_host/kv_compress_epilog_def.cpp) | 压缩 KV 后处理 的主机算子定义、支持属性和注册。 | 注册：`KvCompressEpilog` |
| [kv_compress_epilog_tiling_arch35.cpp](../../../csrc/attention/kv_compress_epilog/op_host/kv_compress_epilog_tiling_arch35.cpp) | 压缩 KV 后处理 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`CeilDiv`, `RoundUp` |
| [kv_compress_epilog_tiling_arch35.h](../../../csrc/attention/kv_compress_epilog/op_host/kv_compress_epilog_tiling_arch35.h) | 压缩 KV 后处理 的主机侧 tiling 参数/分块策略声明。 | 类型：`KvCompressEpilogCompileInfo`, `KvCompressEpilogTiling` |

## `csrc/attention/kv_compress_epilog/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kv_compress_epilog.cpp](../../../csrc/attention/kv_compress_epilog/op_kernel/kv_compress_epilog.cpp) | 压缩 KV 后处理 的设备执行源码；文件主题为 KV 缓存。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kv_compress_epilog.h](../../../csrc/attention/kv_compress_epilog/op_kernel/kv_compress_epilog.h) | 压缩 KV 后处理 的类型/模板与设备实现头文件；文件主题为 KV 缓存。 | 类型：`KvCompressEpilogRegBase` |
| [kv_compress_epilog_common.h](../../../csrc/attention/kv_compress_epilog/op_kernel/kv_compress_epilog_common.h) | 压缩 KV 后处理 的类型/模板与设备实现头文件；文件主题为 公共逻辑、KV 缓存。 | 函数：`CeilDiv`, `CeilAlign` |

## `csrc/attention/kv_compress_epilog_v2/`

压缩 KV 后处理 V2 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/kv_compress_epilog_v2/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/kv_compress_epilog_v2/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_compress_epilog_v2/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/kv_compress_epilog_v2/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：KvCompressEpilogV2 |
| [kv_compress_epilog_v2_torch_adpt.h](../../../csrc/attention/kv_compress_epilog_v2/kv_compress_epilog_v2_torch_adpt.h) | KV 缓存 的公共接口/实现，供所在算子工程复用。 | 函数：`NormalizeQuantMode`, `ValidateKvCompressEpilogV2Inputs` |

## `csrc/attention/kv_compress_epilog_v2/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_compress_epilog_v2/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kv_compress_epilog_v2_def.cpp](../../../csrc/attention/kv_compress_epilog_v2/op_host/kv_compress_epilog_v2_def.cpp) | 压缩 KV 后处理 V2 的主机算子定义、支持属性和注册。 | 注册：`KvCompressEpilogV2` |
| [kv_compress_epilog_v2_proto.cpp](../../../csrc/attention/kv_compress_epilog_v2/op_host/kv_compress_epilog_v2_proto.cpp) | 压缩 KV 后处理 V2 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShape4KvCompressEpilogV2`, `InferDtype4KvCompressEpilogV2` |
| [kv_compress_epilog_v2_tiling_arch35.cpp](../../../csrc/attention/kv_compress_epilog_v2/op_host/kv_compress_epilog_v2_tiling_arch35.cpp) | 压缩 KV 后处理 V2 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`CeilDiv`, `RoundUp` |
| [kv_compress_epilog_v2_tiling_arch35.h](../../../csrc/attention/kv_compress_epilog_v2/op_host/kv_compress_epilog_v2_tiling_arch35.h) | 压缩 KV 后处理 V2 的主机侧 tiling 参数/分块策略声明。 | 类型：`KvCompressEpilogV2CompileInfo`, `KvCompressEpilogV2Tiling` |

## `csrc/attention/kv_compress_epilog_v2/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kv_compress_epilog_v2.cpp](../../../csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2.cpp) | 压缩 KV 后处理 V2 的设备执行源码；文件主题为 KV 缓存。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kv_compress_epilog_v2_common.h](../../../csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_common.h) | 压缩 KV 后处理 V2 的类型/模板与设备实现头文件；文件主题为 公共逻辑、KV 缓存。 | 函数：`CeilDiv`, `RoundUp` |
| [kv_compress_epilog_v2_kernel.h](../../../csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_kernel.h) | 压缩 KV 后处理 V2 的类型/模板与设备实现头文件；文件主题为 KV 缓存。 | 类型：`KvCompressEpilogV2Kernel` |
| [kv_compress_epilog_v2_layout_2d.h](../../../csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_layout_2d.h) | 压缩 KV 后处理 V2 的类型/模板与设备实现头文件；文件主题为 内存布局、KV 缓存。 | 类型：`KvCompressEpilogV2Layout2dCommon`, `KvCompressEpilogV2Layout2dFp8Kernel`, `KvCompressEpilogV2Layout2dFp4G32Kernel` |
| [kv_compress_epilog_v2_layout_4d.h](../../../csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_layout_4d.h) | 压缩 KV 后处理 V2 的类型/模板与设备实现头文件；文件主题为 内存布局、KV 缓存。 | 类型：`KvCompressEpilogV2Layout4dCommon`, `KvCompressEpilogV2Layout4dFp8Kernel`, `KvCompressEpilogV2Layout4dFp4G32Kernel` |
| [kv_compress_epilog_v2_quant_fp4_g16.h](../../../csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_quant_fp4_g16.h) | 压缩 KV 后处理 V2 的类型/模板与设备实现头文件；文件主题为 量化、KV 缓存。 | 函数：`VFProcessMxFp4Group16RowBatchPerfVF`, `VFProcessMxFp4Group16RowBatchPerf` |
| [kv_compress_epilog_v2_quant_fp4_g32.h](../../../csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_quant_fp4_g32.h) | 压缩 KV 后处理 V2 的类型/模板与设备实现头文件；文件主题为 量化、KV 缓存。 | 函数：`VFProcessMxFp4RowBatchPerfVF`, `VFProcessMxFp4RowBatchPerf` |
| [kv_compress_epilog_v2_quant_fp8.h](../../../csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_quant_fp8.h) | 压缩 KV 后处理 V2 的类型/模板与设备实现头文件；文件主题为 量化、KV 缓存。 | 函数：`LoadB16AsFloat`, `StoreFloatAsFp8` |

## `csrc/attention/kv_quant_sparse_attn_sharedkv/`

量化共享 KV 稀疏注意力 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：KvQuantSparseAttnSharedkv |

## `csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kv_quant_sparse_attn_sharedkv_check.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check.cpp) | 量化共享 KV 稀疏注意力 的主机侧辅助接口或实现。 | 函数：`SASDataTypeToSerialString`, `KvQuantSASLayoutToSerialString` |
| [kv_quant_sparse_attn_sharedkv_check.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check.h) | 量化共享 KV 稀疏注意力 的主机侧辅助接口或实现。 | 类型：`SASTilingRequiredParaInfo`, `SASTilingOptionalParaInfo`, `SASLayout` |
| [kv_quant_sparse_attn_sharedkv_check_consistancy.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_consistancy.cpp) | 量化共享 KV 稀疏注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kv_quant_sparse_attn_sharedkv_check_existance.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_existance.cpp) | 量化共享 KV 稀疏注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kv_quant_sparse_attn_sharedkv_check_feature.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_feature.cpp) | 量化共享 KV 稀疏注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kv_quant_sparse_attn_sharedkv_check_single_para.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_single_para.cpp) | 量化共享 KV 稀疏注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kv_quant_sparse_attn_sharedkv_def.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_def.cpp) | 量化共享 KV 稀疏注意力 的主机算子定义、支持属性和注册。 | 注册：`KvQuantSparseAttnSharedkv` |
| [kv_quant_sparse_attn_sharedkv_proto.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_proto.cpp) | 量化共享 KV 稀疏注意力 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShapeKvQuantSparseAttnSharedkv`, `InferDataTypeKvQuantSparseAttnSharedkv` |
| [kv_quant_sparse_attn_sharedkv_tiling.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_tiling.cpp) | 量化共享 KV 稀疏注意力 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`SASCompileInfo` |
| [kv_quant_sparse_attn_sharedkv_tiling.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_tiling.h) | 量化共享 KV 稀疏注意力 的主机侧 tiling 参数/分块策略声明。 | 类型：`KvQuantSparseAttnSharedkvTiling` |

## `csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kv_quant_sparse_attn_sharedkv.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv.cpp) | 量化共享 KV 稀疏注意力 的设备执行源码；文件主题为 量化、注意力、KV 缓存。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kv_quant_sparse_attn_sharedkv_common.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv_common.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 公共逻辑、量化、注意力。 | 类型：`SAS_LAYOUT`, `SASTemplateMode` |
| [kv_quant_sparse_attn_sharedkv_metadata.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv_metadata.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 元数据、量化、注意力。 | 类型：`SasMetaData` |
| [kv_quant_sparse_attn_sharedkv_template_tiling_key.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv_template_tiling_key.h) | 量化共享 KV 稀疏注意力 的kernel 分派使用的 tiling key 定义；文件主题为 量化、注意力、KV 缓存。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [vf/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf) | vf 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kv_quant_sparse_attn_sharedkv_common_arch35.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/kv_quant_sparse_attn_sharedkv_common_arch35.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 公共逻辑、量化、注意力。 | 函数：`Align2Func`, `Align8Func` |
| [kv_quant_sparse_attn_sharedkv_kvcache.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/kv_quant_sparse_attn_sharedkv_kvcache.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 量化、缓存、注意力。 | 函数：`GetSingleCoreParam`, `ComputeParamBatch` |
| [kv_quant_sparse_attn_sharedkv_scfa_block_cube.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_cube.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 量化、缓存页/分块、注意力。 | 类型：`CubeCoordInfo`, `SCFABlockCube`, `SCFABlockCubeDummy` |
| [kv_quant_sparse_attn_sharedkv_scfa_block_vector.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 量化、缓存页/分块、注意力。 | 类型：`SCFABlockVec`, `SCFABlockVecDummy` |
| [kv_quant_sparse_attn_sharedkv_scfa_kernel.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 量化、注意力、KV 缓存。 | 类型：`KvQuantSparseAttnSharedkvScfa` |
| [util_regbase.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/util_regbase.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件。 | 类型：`RunParamStr`, `RunInfo`, `ConstInfo` |

## `csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CopyInL1.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/common/CopyInL1.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`KVLAYOUT`, `CopyParam`, `PAShape` |
| [FixpipeOut.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/common/FixpipeOut.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件。 | 类型：`fixpipeOutParams` |
| [buffer.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/common/buffer.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件。 | 类型：`BufferType`, `SyncType`, `BufferInfo` |
| [buffer_manager.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/common/buffer_manager.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 管理器。 | 类型：`BufferManager` |
| [buffers_policy.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/common/buffers_policy.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件。 | 类型：`BuffersPolicySingleBuffer`, `BuffersPolicyDB`, `BuffersPolicy3buff` |
| [matmul.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/common/matmul.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 矩阵乘。 | 类型：`MMParam`, `ABLayout` |
| [offset_calculator.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/common/offset_calculator.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件。 | 类型：`GmFormat`, `GmLayout`, `ActualSeqLensMode` |

## `csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/`

vf 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [vf_basic_block_aligned128_no_update_scfa.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/vf_basic_block_aligned128_no_update_scfa.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 函数：`ProcessVec1NoUpdateImpl128VF`, `ProcessVec1NoUpdateImpl128` |
| [vf_basic_block_aligned128_update_scfa.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/vf_basic_block_aligned128_update_scfa.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 函数：`ProcessVec1UpdateImpl128VF`, `ProcessVec1UpdateImpl128` |
| [vf_basic_block_unaligned128_no_update_scfa.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/vf_basic_block_unaligned128_no_update_scfa.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 函数：`ProcessVec1NoUpdateGeneralImpl128VF`, `ProcessVec1NoUpdateGeneralImpl128` |
| [vf_basic_block_unaligned128_update_scfa.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/vf_basic_block_unaligned128_update_scfa.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 函数：`ProcessVec1UpdateGeneralImpl128VF`, `ProcessVec1UpdateGeneralImpl128` |
| [vf_basic_block_unaligned64_no_update_scfa.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/vf_basic_block_unaligned64_no_update_scfa.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 函数：`ProcessVec1NoUpdateImpl64VF`, `ProcessVec1NoUpdateImpl64` |
| [vf_basic_block_unaligned64_update_scfa.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/vf_basic_block_unaligned64_update_scfa.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 函数：`ProcessVec1UpdateImpl64VF`, `ProcessVec1UpdateImpl64` |
| [vf_basic_block_utils.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/vf_basic_block_utils.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、辅助函数。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [vf_flashupdate_new_scfa.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/vf_flashupdate_new_scfa.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件。 | 函数：`FlashUpdateBasicVF`, `FlashUpdateNew` |
| [vf_mul_sel_softmaxflashv2_cast_nz_scfa.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_scfa.h) | 量化共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 概率归一化、类型转换。 | 类型：`OriginNRange` |

## `csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/`

量化共享 KV 稀疏元数据 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_graph/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_graph) | 设备图 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel_aicpu/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu) | op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：KvQuantSparseAttnSharedkvMetadata |

## `csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_graph/`

设备图 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kv_quant_sparse_attn_sharedkv_metadata_proto.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_graph/kv_quant_sparse_attn_sharedkv_metadata_proto.h) | 元数据、量化、注意力 的公共接口/实现，供所在算子工程复用。 | 注册：`KvQuantSparseAttnSharedkvMetadata` |

## `csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kv_quant_sparse_attn_sharedkv_metadata_infershape.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host/kv_quant_sparse_attn_sharedkv_metadata_infershape.cpp) | 量化共享 KV 稀疏元数据 的主机侧辅助接口或实现。 | 函数：`InferShapeKvQuantSparseAttnSharedkvMetadata`, `InferDtypeKvQuantSparseAttnSharedkvMetadata` |

## `csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_kv_quant_sparse_attn_sharedkv_metadata.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host/op_api/aclnn_kv_quant_sparse_attn_sharedkv_metadata.cpp) | 量化共享 KV 稀疏元数据 的主机侧辅助接口或实现。 | 函数：`ParamsCheck`, `aclnnKvQuantSparseAttnSharedkvMetadataGetWorkspaceSize` |
| [aclnn_kv_quant_sparse_attn_sharedkv_metadata.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host/op_api/aclnn_kv_quant_sparse_attn_sharedkv_metadata.h) | 量化共享 KV 稀疏元数据 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [l0_kv_quant_sparse_attn_sharedkv_metadata.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host/op_api/l0_kv_quant_sparse_attn_sharedkv_metadata.cpp) | 量化共享 KV 稀疏元数据 的主机侧辅助接口或实现。 | 函数：`KvQuantSparseAttnSharedkvMetadata` |
| [l0_kv_quant_sparse_attn_sharedkv_metadata.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_host/op_api/l0_kv_quant_sparse_attn_sharedkv_metadata.h) | 量化共享 KV 稀疏元数据 的主机侧辅助接口或实现。 | 函数：`KvQuantSparseAttnSharedkvMetadata` |

## `csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu/`

op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu/kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp) | 元数据、量化、注意力 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [kv_quant_sparse_attn_sharedkv_metadata_aicpu.h](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu/kv_quant_sparse_attn_sharedkv_metadata_aicpu.h) | 元数据、量化、注意力 的公共接口/实现，供所在算子工程复用。 | 类型：`SparseMode`, `ValidSocVersion`, `FlashDecodeResult` |
| [kv_quant_sparse_attn_sharedkv_metadata_aicpu.json](../../../csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu/kv_quant_sparse_attn_sharedkv_metadata_aicpu.json) | 所在模块的结构化配置或参数模板。 | 顶层项：`KvQuantSparseAttnSharedkvMetadata` |

## `csrc/attention/kv_quant_sparse_flash_attention_vllm/`

量化 KV 稀疏 Flash Attention 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：KvQuantSparseFlashAttentionVllm |
| [kv_quant_sparse_flash_attention_vllm_torch_adpt.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/kv_quant_sparse_flash_attention_vllm_torch_adpt.h) | 注意力、量化、KV 缓存 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/kv_quant_sparse_flash_attention_vllm/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [kv_quant_sparse_flash_attention_vllm_def.cpp](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_host/kv_quant_sparse_flash_attention_vllm_def.cpp) | 量化 KV 稀疏 Flash Attention 的主机算子定义、支持属性和注册。 | 注册：`KvQuantSparseFlashAttentionVllm` |
| [kv_quant_sparse_flash_attention_vllm_infershape.cpp](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_host/kv_quant_sparse_flash_attention_vllm_infershape.cpp) | 量化 KV 稀疏 Flash Attention 的主机侧辅助接口或实现。 | 函数：`InferShapeKvQuantSparseFlashAttentionVllm`, `InferDataTypeKvQuantSparseFlashAttentionVllm` |
| [kv_quant_sparse_flash_attention_vllm_tiling.cpp](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_host/kv_quant_sparse_flash_attention_vllm_tiling.cpp) | 量化 KV 稀疏 Flash Attention 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`KvQuantSparseFlashAttentionVllmCompileInfo` |
| [kv_quant_sparse_flash_attention_vllm_tiling.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_host/kv_quant_sparse_flash_attention_vllm_tiling.h) | 量化 KV 稀疏 Flash Attention 的主机侧 tiling 参数/分块策略声明。 | 类型：`QSFALayout`, `QSFATilingShapeCompareParam`, `KvStorageMode` |

## `csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kv_quant_sparse_flash_attention_vllm.cpp](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/kv_quant_sparse_flash_attention_vllm.cpp) | 量化 KV 稀疏 Flash Attention 的设备执行源码；文件主题为 注意力、量化、KV 缓存。 | 函数：`DispatchKernelDtype310` |
| [kv_quant_sparse_flash_attention_vllm_common.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/kv_quant_sparse_flash_attention_vllm_common.h) | 量化 KV 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、公共逻辑、量化。 | 类型：`QSFA_LAYOUT`, `QUANT_MODE`, `ATTENTION_MODE` |
| [kv_quant_sparse_flash_attention_vllm_kernel_mla.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/kv_quant_sparse_flash_attention_vllm_kernel_mla.h) | 量化 KV 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、量化、MLA 潜在注意力。 | 类型：`TempLoopInfo`, `KvQuantSparseFlashAttentionVllmMla` |
| [kv_quant_sparse_flash_attention_vllm_service_cube_mla.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/kv_quant_sparse_flash_attention_vllm_service_cube_mla.h) | 量化 KV 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、量化、MLA 潜在注意力。 | 类型：`Position`, `PAShape`, `QSFAMatmulService` |
| [kv_quant_sparse_flash_attention_vllm_service_vector_mla.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/kv_quant_sparse_flash_attention_vllm_service_vector_mla.h) | 量化 KV 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、量化、MLA 潜在注意力。 | 类型：`QSFAVectorService`, `ConstInfo` |
| [kv_quant_sparse_flash_attention_vllm_template_tiling_key.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/kv_quant_sparse_flash_attention_vllm_template_tiling_key.h) | 量化 KV 稀疏 Flash Attention 的kernel 分派使用的 tiling key 定义；文件主题为 注意力、量化、KV 缓存。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kv_quant_sparse_flash_attention_vllm_common_arch35.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/arch35/kv_quant_sparse_flash_attention_vllm_common_arch35.h) | 量化 KV 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、公共逻辑、量化。 | 类型：`QSFA_LAYOUT`, `QSFATemplateMode` |
| [kv_quant_sparse_flash_attention_vllm_kernel_mla.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/arch35/kv_quant_sparse_flash_attention_vllm_kernel_mla.h) | 量化 KV 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、量化、MLA 潜在注意力。 | 类型：`KvQuantSparseFlashAttentionVllmMla` |
| [kv_quant_sparse_flash_attention_vllm_kvcache.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/arch35/kv_quant_sparse_flash_attention_vllm_kvcache.h) | 量化 KV 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、量化、缓存。 | 函数：`GetSingleCoreParam`, `ComputeParamBatch` |
| [kv_quant_sparse_flash_attention_vllm_service_cube_mla.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/arch35/kv_quant_sparse_flash_attention_vllm_service_cube_mla.h) | 量化 KV 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、量化、MLA 潜在注意力。 | 类型：`CubeCoordInfo`, `QSFAMatmulService`, `QSFAMatmulServiceDummy` |
| [kv_quant_sparse_flash_attention_vllm_service_vector_mla.h](../../../csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/arch35/kv_quant_sparse_flash_attention_vllm_service_vector_mla.h) | 量化 KV 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、量化、MLA 潜在注意力。 | 类型：`QSFAVectorService`, `QSFAVectorServiceDummy` |

## `csrc/attention/lightning_indexer_quant/`

Indexer 量化 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/lightning_indexer_quant/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/lightning_indexer_quant/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/lightning_indexer_quant/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [lightning_indexer_quant_torch_adpt.h](../../../csrc/attention/lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h) | 历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 函数：`npu_lightning_indexer_quant` |

## `csrc/attention/lightning_indexer_quant/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/lightning_indexer_quant/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [lightning_indexer_quant_def.cpp](../../../csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_def.cpp) | Indexer 量化 的主机算子定义、支持属性和注册。 | 注册：`LightningIndexerQuant` |
| [lightning_indexer_quant_proto.cpp](../../../csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_proto.cpp) | Indexer 量化 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShapeLightningIndexerQuant`, `InferDataTypeLightningIndexerQuant` |
| [lightning_indexer_quant_tiling.cpp](../../../csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_tiling.cpp) | Indexer 量化 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`TilingPrepareForLightningIndexerQuant`, `TilingForLightningIndexerQuant` |
| [lightning_indexer_quant_tiling.h](../../../csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_tiling.h) | Indexer 量化 的主机侧 tiling 参数/分块策略声明。 | 类型：`TilingRequiredParaInfo`, `TilingOptionalParaInfo`, `DataLayout` |

## `csrc/attention/lightning_indexer_quant/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [lightning_indexer_quant.cpp](../../../csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant.cpp) | Indexer 量化 的设备执行源码；文件主题为 历史位置索引、量化。 | 函数：`lightning_indexer_quant` |
| [lightning_indexer_quant_common.h](../../../csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant_common.h) | Indexer 量化 的类型/模板与设备实现头文件；文件主题为 历史位置索引、公共逻辑、量化。 | 类型：`LI_LAYOUT`, `LIQType`, `RunInfo` |
| [lightning_indexer_quant_kernel.h](../../../csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant_kernel.h) | Indexer 量化 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`TempLoopInfo`, `LIQPreload` |
| [lightning_indexer_quant_service_cube.h](../../../csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant_service_cube.h) | Indexer 量化 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`MmInfo`, `LIQMatmul` |
| [lightning_indexer_quant_service_vector.h](../../../csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant_service_vector.h) | Indexer 量化 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`LIQVector`, `LIQCommon` |
| [lightning_indexer_quant_template_tiling_key.h](../../../csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant_template_tiling_key.h) | Indexer 量化 的kernel 分派使用的 tiling key 定义；文件主题为 历史位置索引、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [lightning_indexer_quant_vector.h](../../../csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant_vector.h) | Indexer 量化 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 函数：`CopyOut`, `InitSortOutBuf` |

## `csrc/attention/lightning_indexer_v2/`

历史位置索引 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_kernel/](../../../csrc/attention/lightning_indexer_v2/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

## `csrc/attention/lightning_indexer_v2/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch35/](../../../csrc/attention/lightning_indexer_v2/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

## `csrc/attention/lightning_indexer_v2/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [vf/](../../../csrc/attention/lightning_indexer_v2/op_kernel/arch35/vf) | vf 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

## `csrc/attention/lightning_indexer_v2/op_kernel/arch35/vf/`

vf 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/attention/lightning_indexer_v2/op_kernel/arch35/vf/common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

## `csrc/attention/lightning_indexer_v2/op_kernel/arch35/vf/common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [lightning_indexer_v2_vector1_base.h](../../../csrc/attention/lightning_indexer_v2/op_kernel/arch35/vf/common/lightning_indexer_v2_vector1_base.h) | 历史位置索引 的类型/模板与设备实现头文件。 | 类型：`UIntSortTraits`, `UIntSortConstCtx`, `FloatSortTraits` |

## `csrc/attention/msa_index_score/`

MSA 相关位置评分 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/attention/msa_index_score/docs) | docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [examples/](../../../csrc/attention/msa_index_score/examples) | examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/msa_index_score/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/msa_index_score/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |
| [tests/](../../../csrc/attention/msa_index_score/tests) | tests 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/msa_index_score/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/msa_index_score/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：MsaIndexScore |
| [msa_index_score_torch_adpt.h](../../../csrc/attention/msa_index_score/msa_index_score_torch_adpt.h) | msa_index_score_torch_adpt 的公共接口/实现，供所在算子工程复用。 | 函数：`CheckMsaIndexScoreParams`, `npu_msa_index_score` |

## `csrc/attention/msa_index_score/docs/`

docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnnMsaIndexScore.md](../../../csrc/attention/msa_index_score/docs/aclnnMsaIndexScore.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：aclnnMsaIndexScore。 | 标题：aclnnMsaIndexScore |

## `csrc/attention/msa_index_score/examples/`

examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_aclnn_msa_index_score.cpp](../../../csrc/attention/msa_index_score/examples/test_aclnn_msa_index_score.cpp) | test_aclnn_msa_index_score 的公共接口/实现，供所在算子工程复用。 | 类型：`KeyLayout`, `TestCase`, `DeviceBuffer` |

## `csrc/attention/msa_index_score/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/msa_index_score/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [msa_index_score_def.cpp](../../../csrc/attention/msa_index_score/op_host/msa_index_score_def.cpp) | MSA 相关位置评分 的主机算子定义、支持属性和注册。 | 注册：`MsaIndexScore` |
| [msa_index_score_infershape.cpp](../../../csrc/attention/msa_index_score/op_host/msa_index_score_infershape.cpp) | MSA 相关位置评分 的主机侧辅助接口或实现。 | 函数：`RoundUpTo`, `InferShapeMsaIndexScore` |
| [msa_index_score_tiling.cpp](../../../csrc/attention/msa_index_score/op_host/msa_index_score_tiling.cpp) | MSA 相关位置评分 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`IsFp8ComputeDtype`, `IsNonQuantQueryDtype` |
| [msa_index_score_tiling.h](../../../csrc/attention/msa_index_score/op_host/msa_index_score_tiling.h) | MSA 相关位置评分 的主机侧 tiling 参数/分块策略声明。 | 类型：`MsaIndexScoreCompileInfo`, `MsaIndexScoreInfo` |

## `csrc/attention/msa_index_score/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch22/](../../../csrc/attention/msa_index_score/op_kernel/arch22) | arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/msa_index_score/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [catlass/](../../../csrc/attention/msa_index_score/op_kernel/catlass) | catlass 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_index_score.cpp](../../../csrc/attention/msa_index_score/op_kernel/msa_index_score.cpp) | MSA 相关位置评分 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [msa_index_score_common.h](../../../csrc/attention/msa_index_score/op_kernel/msa_index_score_common.h) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/msa_index_score/op_kernel/arch22/`

arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_block_mmad.h](../../../csrc/attention/msa_index_score/op_kernel/arch22/msa_block_mmad.h) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`AType_`, `BType_`, `CType_` |
| [msa_index_score_epilogue.h](../../../csrc/attention/msa_index_score/op_kernel/arch22/msa_index_score_epilogue.h) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`MsaSegRowMaxEpilogue` |
| [msa_index_score_kernel.h](../../../csrc/attention/msa_index_score/op_kernel/arch22/msa_index_score_kernel.h) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`ElementQ_`, `MsaIndexScoreKernel` |
| [msa_index_score_task.h](../../../csrc/attention/msa_index_score/op_kernel/arch22/msa_index_score_task.h) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`MsaTask`, `MsaTaskScheduler` |

## `csrc/attention/msa_index_score/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_block_mmad.h](../../../csrc/attention/msa_index_score/op_kernel/arch35/msa_block_mmad.h) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`AType_`, `BType_`, `CType_` |
| [msa_index_score_kernel.h](../../../csrc/attention/msa_index_score/op_kernel/arch35/msa_index_score_kernel.h) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`ElementQ_`, `MsaIndexScoreKernel` |
| [msa_seg_row_max_epilogue.h](../../../csrc/attention/msa_index_score/op_kernel/arch35/msa_seg_row_max_epilogue.h) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`MsaSegRowMaxEpilogue` |

## `csrc/attention/msa_index_score/op_kernel/catlass/`

catlass 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [arch/](../../../csrc/attention/msa_index_score/op_kernel/catlass/arch) | arch 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [detail/](../../../csrc/attention/msa_index_score/op_kernel/catlass/detail) | detail 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [gemm/](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm) | gemm 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [layout/](../../../csrc/attention/msa_index_score/op_kernel/catlass/layout) | 内存布局 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_catlass.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/msa_catlass.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`OpSystemRunCfg` |
| [msa_conv_coord.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/msa_conv_coord.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 卷积。 | 类型：`Conv3dParams`, `ConvCoreShape`, `ConvFmapL1Shape` |
| [msa_coord.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/msa_coord.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`Index_`, `LongIndex_`, `Coord` |
| [msa_gemm_coord.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/msa_gemm_coord.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`GemmShape`, `GemmCoord` |
| [msa_matrix_coord.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/msa_matrix_coord.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`MatrixShape`, `MatrixCoord` |
| [msa_numeric_size.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/msa_numeric_size.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`SizeOfBits` |

## `csrc/attention/msa_index_score/op_kernel/catlass/arch/`

arch 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_arch.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/arch/msa_arch.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`AtlasA2`, `AtlasA5` |
| [msa_cross_core_sync.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/arch/msa_cross_core_sync.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`CrossCoreFlag`, `CrossCoreFlagWithReverse`, `BarrierFlag` |
| [msa_local_tensor_buffer.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/arch/msa_local_tensor_buffer.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 张量。 | 类型：`LocalTensorBufferBase`, `Element`, `ArchTag` |
| [msa_resource.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/arch/msa_resource.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`ArchTag`, `Resource` |

## `csrc/attention/msa_index_score/op_kernel/catlass/detail/`

detail 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_alignment.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/detail/msa_alignment.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`U` |
| [msa_dependent_false.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/detail/msa_dependent_false.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [msa_macros.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/detail/msa_macros.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/msa_index_score/op_kernel/catlass/gemm/`

gemm 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [tile/](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile) | tile 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_dispatch_policy.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/msa_dispatch_policy.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 token 分发。 | 类型：`ArchTag_`, `MmadBase`, `MmadAtlasA2Pingpong` |
| [msa_gemm_helper.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/msa_gemm_helper.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`Element`, `Layout`, `L1AlignHelper` |
| [msa_gemm_type.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/msa_gemm_type.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`Element_`, `Layout_`, `GemmType` |

## `csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/`

tile 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_cast_fp8_to_fp16.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_cast_fp8_to_fp16.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 类型转换。 | 类型：`ArchTag`, `SrcType_`, `DstType_` |
| [msa_cast_int4_to_int8.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_cast_int4_to_int8.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 类型转换。 | 类型：`ArchTag`, `SrcType_`, `DstType_` |
| [msa_cast_int8_to_fp16.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_cast_int8_to_fp16.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 类型转换。 | 类型：`ArchTag`, `SrcType_`, `DstType_` |
| [msa_copy_gm_to_l1.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_copy_gm_to_l1.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `GmType`, `L1Type` |
| [msa_copy_gm_to_ub.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_copy_gm_to_ub.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `GmType`, `CopyGm2Ub` |
| [msa_copy_l0c_to_gm.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_copy_l0c_to_gm.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ScaleGranularity`, `ArchTag`, `ElementSrc` |
| [msa_copy_l1_to_bt.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_copy_l1_to_bt.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `L1Type`, `L0Type` |
| [msa_copy_l1_to_fp.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_copy_l1_to_fp.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `L1Type`, `L0Type` |
| [msa_copy_l1_to_l0a.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_copy_l1_to_l0a.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `L1Type`, `L0Type` |
| [msa_copy_l1_to_l0b.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_copy_l1_to_l0b.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `L1Type`, `L0Type` |
| [msa_copy_ub_to_gm.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_copy_ub_to_gm.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `GmType`, `CopyUb2Gm` |
| [msa_tile_copy.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_tile_copy.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `AType`, `BType` |
| [msa_tile_mmad.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/gemm/tile/msa_tile_mmad.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`ArchTag_`, `AType_`, `BType_` |

## `csrc/attention/msa_index_score/op_kernel/catlass/layout/`

内存布局 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_layout.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/layout/msa_layout.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 内存布局。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [msa_matrix.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/layout/msa_matrix.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`RowMajor`, `Element`, `ColumnMajor` |
| [msa_tensor.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/layout/msa_tensor.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件；文件主题为 张量。 | 类型：`NC1HWC0`, `Element`, `CI1KHKWCOCI0` |
| [msa_vector.hpp](../../../csrc/attention/msa_index_score/op_kernel/catlass/layout/msa_vector.hpp) | MSA 相关位置评分 的类型/模板与设备实现头文件。 | 类型：`VectorLayout`, `Element` |

## `csrc/attention/msa_index_score/tests/`

tests 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [golden/](../../../csrc/attention/msa_index_score/tests/golden) | golden 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/msa_index_score/tests/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/msa_index_score/tests/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：MsaIndexScore Test Guide |

## `csrc/attention/msa_index_score/tests/golden/`

golden 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [msa_index_score_golden.py](../../../csrc/attention/msa_index_score/tests/golden/msa_index_score_golden.py) | 算子工程 msa_index_score_golden 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：MsaIndexScore CPU 参考实现。；定义：`MsaIndexScoreGoldenInputs`, `round_up`, `msa_index_score_golden`, `compare` |

## `csrc/attention/quant_lightning_indexer_v2/`

量化历史位置索引 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/attention/quant_lightning_indexer_v2/docs) | docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [examples/](../../../csrc/attention/quant_lightning_indexer_v2/examples) | examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/quant_lightning_indexer_v2/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |
| [tests/](../../../csrc/attention/quant_lightning_indexer_v2/tests) | tests 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [torch_extension/](../../../csrc/attention/quant_lightning_indexer_v2/torch_extension) | torch_extension 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/quant_lightning_indexer_v2/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：QuantLightningIndexerV2 |
| [quant_lightning_indexer_v2_torch_adpt.h](../../../csrc/attention/quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.h) | 历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 函数：`valid_tensor`, `QuantLightningIndexerMetadata` |

## `csrc/attention/quant_lightning_indexer_v2/docs/`

docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnnQuantLightningIndexerV2.md](../../../csrc/attention/quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：aclnnQuantLightningIndexerV2。 | 标题：aclnnQuantLightningIndexerV2 |
| [qli_v2_two_level_topk_design.md](../../../csrc/attention/quant_lightning_indexer_v2/docs/qli_v2_two_level_topk_design.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：qli_v2_two_level_topk_design。 | 标题：QuantLightningIndexerV2 两级 TopK（候选块选择）方案设计 — 仅 arch22 / 910b |

## `csrc/attention/quant_lightning_indexer_v2/examples/`

examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_aclnn_quant_lightning_indexer_v2.cpp](../../../csrc/attention/quant_lightning_indexer_v2/examples/test_aclnn_quant_lightning_indexer_v2.cpp) | 历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 类型：`TensorResources` |

## `csrc/attention/quant_lightning_indexer_v2/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [quant_lightning_indexer_v2_infershape.cpp](../../../csrc/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_infershape.cpp) | 量化历史位置索引 的主机侧辅助接口或实现。 | 函数：`InferShapeQuantLightningIndexerV2`, `InferDataTypeQuantLightningIndexerV2` |
| [quant_lightning_indexer_v2_tiling.cpp](../../../csrc/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.cpp) | 量化历史位置索引 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`QLIV2DataTypeToSerialString`, `ToVector` |
| [quant_lightning_indexer_v2_tiling.h](../../../csrc/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.h) | 量化历史位置索引 的主机侧 tiling 参数/分块策略声明。 | 类型：`TilingRequiredParaInfo`, `TilingOptionalParaInfo`, `DataLayout` |
| [vllm_ascend_quant_lightning_indexer_v2_def.cpp](../../../csrc/attention/quant_lightning_indexer_v2/op_host/vllm_ascend_quant_lightning_indexer_v2_def.cpp) | 量化历史位置索引 的主机算子定义、支持属性和注册。 | 注册：`VllmAscendQuantLightningIndexerV2` |

## `csrc/attention/quant_lightning_indexer_v2/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch22/](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22) | arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [quant_lightning_indexer_v2.cpp](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2.cpp) | 量化历史位置索引 的设备执行源码；文件主题为 历史位置索引、量化。 | 函数：`quant_lightning_indexer_v2` |
| [quant_lightning_indexer_v2_metadata.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2_metadata.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 元数据、历史位置索引、量化。 | 类型：`QliV2Metadata` |
| [quant_lightning_indexer_v2_template_tiling_key.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2_template_tiling_key.h) | 量化历史位置索引 的kernel 分派使用的 tiling key 定义；文件主题为 历史位置索引、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [vllm_ascend_quant_lightning_indexer_v2.cpp](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/vllm_ascend_quant_lightning_indexer_v2.cpp) | 量化历史位置索引 的设备执行源码；文件主题为 历史位置索引、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/`

arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [quant_lightning_indexer_v2_common_arch22.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_common_arch22.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、公共逻辑、量化。 | 类型：`LI_LAYOUT`, `QLIV2Type`, `RunInfo` |
| [quant_lightning_indexer_v2_kernel_arch22.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`TempLoopInfo`, `QLIV2Preload` |
| [quant_lightning_indexer_v2_service_cube_arch22.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_cube_arch22.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`MmInfo`, `QLIV2Matmul` |
| [quant_lightning_indexer_v2_service_vector_arch22.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`QLIV2Vector`, `QLIV2Common` |
| [quant_lightning_indexer_v2_vector.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_vector.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 函数：`CopyOut`, `InitSortOutBuf` |

## `csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [vf/](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/vf) | vf 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [quant_lightning_indexer_v2_common_arch35.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_common_arch35.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、公共逻辑、量化。 | 类型：`LI_LAYOUT`, `QLIV2Type`, `RunInfo` |
| [quant_lightning_indexer_v2_kernel_arch35.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_kernel_arch35.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`TempLoopInfo`, `QLIV2Preload` |
| [quant_lightning_indexer_v2_service_cube_arch35.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_cube_arch35.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`QLIV2Matmul` |
| [quant_lightning_indexer_v2_service_vector_arch35.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_vector_arch35.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 类型：`QLIV2Vector`, `QLIV2Common` |

## `csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/vf/`

vf 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [quant_lightning_indexer_v2_topk.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/vf/quant_lightning_indexer_v2_topk.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化、TopK 选择。 | 类型：`LITopk` |
| [quant_lightning_indexer_v2_vector1.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/vf/quant_lightning_indexer_v2_vector1.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 历史位置索引、量化。 | 函数：`UIntToFloatReturnValueVF`, `UIntToFloatReturnValue` |
| [vf_topk.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/vf/vf_topk.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 TopK 选择。 | 函数：`StoreHistogramResult`, `FindTargetBinAndUpdateNextK` |
| [vf_topk_16_gather_quant_v2.h](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/vf/vf_topk_16_gather_quant_v2.h) | 量化历史位置索引 的类型/模板与设备实现头文件；文件主题为 收集、量化、TopK 选择。 | 函数：`HistogramsHighVFImpl`, `FindHighTargetBinVFImpl` |

## `csrc/attention/quant_lightning_indexer_v2/tests/`

tests 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [assets/](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets) | assets 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [pytest/](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest) | pytest 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [ut/](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut) | ut 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2/tests/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/attention/quant_lightning_indexer_v2/tests/assets/`

assets 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [impl/](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/impl) | impl 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compare_batch_outputs.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/compare_batch_outputs.py) | 算子工程 输出、批次 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Compare LI_V2/QLI_V2 output-0 dumps across TTK batch cases.；定义：`IndexerBatchDumpComparator`, `load_batch_protocol`, `build_parser`, `main` |
| [spec.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/spec.py) | 算子工程 spec 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：TestSpec adapter for QuantLightningIndexer V2 TTK assets.；定义：`QuantLightningIndexerV2Spec`, `AclnnQuantLightningIndexerV2Spec`, `load_impl_module` |

## `csrc/attention/quant_lightning_indexer_v2/tests/assets/impl/`

impl 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [batch_consistency.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/impl/batch_consistency.py) | 算子工程 批次 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Small batch-consistency protocol shared by LI_V2 and QLI_V2 assets.；定义：`CaseRandomContext`, `BatchRelationProtocol`, `normalize_indexer_inputs` |
| [compare.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/impl/compare.py) | 算子工程 compare 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：TTK result adapter for the QuantLightningIndexer V2 pytest TopK comparison.；定义：`PytestV2TopKComparator`, `load_batch_comparator`, `compare` |
| [golden.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/impl/golden.py) | 算子工程 golden 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：CPU Golden adapter for QuantLightningIndexer V2 TTK cases.；定义：`CaseDataStore`, `load_pytest_golden`, `get_case_data`, `materialize_golden` |
| [inputs.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/impl/inputs.py) | 算子工程 输入 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Input customization for QuantLightningIndexer V2 TTK cases.；定义：`QuantLightningIndexerV2InputAdapter`, `restore_mx_input_dtypes`, `rebuild_qli_v2_compare_data`, `zero_metadata` |
| [metadata_inputs.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/impl/metadata_inputs.py) | 算子工程 元数据、输入 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Input customization for standalone QuantLightningIndexerMetadata cases.；定义：`load_metadata_protocol`, `load_sidecar_values`, `copy_values` |
| [metadata_protocol.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/impl/metadata_protocol.py) | 算子工程 元数据 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Private five-operator bridge for TTK manual-data prepare and replay.；定义：`manual_data_roots`, `case_directory_name`, `validate_operator` |
| [npu_preprocess.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/assets/impl/npu_preprocess.py) | 算子工程 npu_preprocess 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Populate QuantLightningIndexer V2 metadata without TTK state coupling.；定义：`load_metadata_protocol`, `get_attribute`, `get_values` |

## `csrc/attention/quant_lightning_indexer_v2/tests/pytest/`

pytest 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [batch/](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/batch) | 批次 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：quant_lightning_indexer_v2算子测试框架 |
| [batch_isolated_run.sh](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/batch_isolated_run.sh) | 算子工程 批次 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`_cleanup_on_interrupt` |
| [collect_perf_data.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/collect_perf_data.py) | 算子工程 collect_perf_data 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`extract_qliv2_row`, `collect_and_save_incremental`, `collect_all` |
| [pytest.ini](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/pytest.ini) | 所在模块的结构化配置或参数模板。 | 配置条目与规则；按实际读取者生效。 |
| [qliv2_parameter_normalization.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/qliv2_parameter_normalization.py) | 算子工程 归一化 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Pure QLI_V2 parameter normalization shared by pytest and TTK adapters.；定义：`normalize_qliv2_params` |
| [qliv2_test_utils.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/qliv2_test_utils.py) | 算子工程 辅助函数 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：NPU-independent case selection, naming, and result helpers for QLI_V2 tests.；定义：`QliV2CaseSelector`, `QliV2ResultWriter`, `ensure_comparison_passed` |
| [quant_lightning_indexer_v2_acl_graph.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/quant_lightning_indexer_v2_acl_graph.py) | 算子工程 历史位置索引、量化、设备图 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`QLIV2Network`, `qliv2_output_acl_graph`, `qliv2_output_acl_graph_from_pt` |
| [quant_lightning_indexer_v2_golden.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/quant_lightning_indexer_v2_golden.py) | 算子工程 历史位置索引、量化 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`GeneralizedQLIV2`, `is_mx_quant_mode`, `is_mxfp4_quant_mode`, `round_fp64_to_bf16_rne` |
| [result_compare_method.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/result_compare_method.py) | 算子工程 result_compare_method 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`cal_relative_diff_np_isclose`, `print_log`, `display_error_output` |
| [test_qliv2_test_utils.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/test_qliv2_test_utils.py) | 算子工程 辅助函数 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`test_case_selector_preserves_requested_order`, `test_case_selector_rejects_ambiguous_or_invalid_selection`, `test_result_writer_uses_readable_name_and_migrates_legacy_result` |
| [test_quant_lightning_indexer_v2_batch.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/test_quant_lightning_indexer_v2_batch.py) | 算子工程 历史位置索引、量化、批次 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`qliv2`, `test_qliv2` |
| [test_quant_lightning_indexer_v2_paramset.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/test_quant_lightning_indexer_v2_paramset.py) | 算子工程 历史位置索引、量化 的生成、配置、编译或打包辅助；接口见右栏。 | 导入：`random`, `torch` |
| [test_quant_lightning_indexer_v2_single.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/test_quant_lightning_indexer_v2_single.py) | 算子工程 历史位置索引、量化 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`test_qliv2` |
| [test_run.sh](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/test_run.sh) | 算子工程 test_run 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`show_help`, `require_value`, `validate_run_mode` |

## `csrc/attention/quant_lightning_indexer_v2/tests/pytest/batch/`

批次 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [list_pt_from_excel.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/batch/list_pt_from_excel.py) | 算子工程 list_pt_from_excel 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`list_pt_from_excel`, `main` |
| [quant_lightning_indexer_v2_pt_loadprocess.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/batch/quant_lightning_indexer_v2_pt_loadprocess.py) | 算子工程 历史位置索引、量化、加载 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`test_qliv2_process`, `test_qliv2_process_graph` |
| [quant_lightning_indexer_v2_pt_save.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/batch/quant_lightning_indexer_v2_pt_save.py) | 算子工程 历史位置索引、量化 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`load_excel_test_cases`, `save_test_case`, `main` |
| [replace_path.py](../../../csrc/attention/quant_lightning_indexer_v2/tests/pytest/batch/replace_path.py) | 算子工程 replace_path 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`replace_paths_in_test_file` |

## `csrc/attention/quant_lightning_indexer_v2/tests/ut/`

ut 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [arch22/](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/arch22) | arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [test_quant_lightning_indexer_v2_infershape.cpp](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/test_quant_lightning_indexer_v2_infershape.cpp) | 量化历史位置索引 的主机侧辅助接口或实现。 | 类型：`QuantLightningIndexerV2Proto` |
| [test_quant_lightning_indexer_v2_utils.h](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/test_quant_lightning_indexer_v2_utils.h) | 量化历史位置索引 的主机侧辅助接口或实现。 | 类型：`CaseParam`, `QLIV2CompileInfo` |

## `csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/arch22/`

arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/arch22/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [test_quant_lightning_indexer_v2_tiling.cpp](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/arch22/test_quant_lightning_indexer_v2_tiling.cpp) | 量化历史位置索引 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`QuantLightningIndexerV2TilingArch22` |

## `csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/arch35/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [test_quant_lightning_indexer_v2_tiling.cpp](../../../csrc/attention/quant_lightning_indexer_v2/tests/ut/op_host/arch35/test_quant_lightning_indexer_v2_tiling.cpp) | 量化历史位置索引 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`QuantLightningIndexerV2TilingArch35` |

## `csrc/attention/quant_lightning_indexer_v2/torch_extension/`

torch_extension 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [csrc/](../../../csrc/attention/quant_lightning_indexer_v2/torch_extension/csrc) | csrc 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../csrc/attention/quant_lightning_indexer_v2/torch_extension/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`.`, `.quant_lightning_indexer` |
| [graph_convert_quant_lightning_indexer.py](../../../csrc/attention/quant_lightning_indexer_v2/torch_extension/graph_convert_quant_lightning_indexer.py) | 算子工程 历史位置索引、量化、设备图 的生成、配置、编译或打包辅助；接口见右栏。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [quant_lightning_indexer.py](../../../csrc/attention/quant_lightning_indexer_v2/torch_extension/quant_lightning_indexer.py) | 算子工程 历史位置索引、量化 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`QuantLightningIndexerOpBuilder`, `quant_lightning_indexer_metadata`, `quant_lightning_indexer_metadata_fallback`, `quant_lightning_indexer_candidate` |

## `csrc/attention/quant_lightning_indexer_v2/torch_extension/csrc/`

csrc 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [quant_lightning_indexer.cpp](../../../csrc/attention/quant_lightning_indexer_v2/torch_extension/csrc/quant_lightning_indexer.cpp) | 历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 函数：`MakeWrapper`, `IsFp4CompatibleTensor` |

## `csrc/attention/quant_lightning_indexer_v2_metadata/`

量化索引元数据 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/attention/quant_lightning_indexer_v2_metadata/docs) | docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [examples/](../../../csrc/attention/quant_lightning_indexer_v2_metadata/examples) | examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_api/](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_api) | 算子对外调用接口及适配。 |
| [op_graph/](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_graph) | 设备图 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel_aicpu/](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu) | op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2_metadata/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/quant_lightning_indexer_v2_metadata/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：QuantLightningIndexerV2Metadata |

## `csrc/attention/quant_lightning_indexer_v2_metadata/docs/`

docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnnQuantLightningIndexerV2Metadata.md](../../../csrc/attention/quant_lightning_indexer_v2_metadata/docs/aclnnQuantLightningIndexerV2Metadata.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：aclnnQuantLightningIndexerV2Metadata。 | 标题：aclnnQuantLightningIndexerV2Metadata |

## `csrc/attention/quant_lightning_indexer_v2_metadata/examples/`

examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_aclnn_quant_lightning_indexer_v2_metadata.cpp](../../../csrc/attention/quant_lightning_indexer_v2_metadata/examples/test_aclnn_quant_lightning_indexer_v2_metadata.cpp) | 元数据、历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 类型：`QliV2Metadata`, `ScopeGuard`, `Tensor` |
| [test_torch_quant_lightning_indexer_v2_metadata.py](../../../csrc/attention/quant_lightning_indexer_v2_metadata/examples/test_torch_quant_lightning_indexer_v2_metadata.py) | 算子工程 元数据、历史位置索引、量化 的生成、配置、编译或打包辅助；接口见右栏。 | 导入：`torch` |

## `csrc/attention/quant_lightning_indexer_v2_metadata/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_quant_lightning_indexer_v2_metadata.cpp](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_api/aclnn_quant_lightning_indexer_v2_metadata.cpp) | 元数据、历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 函数：`ParamsCheckQliV2`, `aclnnVllmAscendQuantLightningIndexerV2MetadataGetWorkspaceSize` |
| [aclnn_quant_lightning_indexer_v2_metadata.h](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_api/aclnn_quant_lightning_indexer_v2_metadata.h) | 元数据、历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 函数：`aclnnVllmAscendQuantLightningIndexerV2MetadataGetWorkspaceSize`, `aclnnVllmAscendQuantLightningIndexerV2Metadata` |
| [quant_lightning_indexer_v2_metadata.cpp](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_api/quant_lightning_indexer_v2_metadata.cpp) | 元数据、历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [quant_lightning_indexer_v2_metadata.h](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_api/quant_lightning_indexer_v2_metadata.h) | 元数据、历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/quant_lightning_indexer_v2_metadata/op_graph/`

设备图 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [quant_lightning_indexer_v2_metadata_proto.h](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_graph/quant_lightning_indexer_v2_metadata_proto.h) | 元数据、历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 注册：`VllmAscendQuantLightningIndexerV2Metadata` |

## `csrc/attention/quant_lightning_indexer_v2_metadata/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [quant_lightning_indexer_v2_metadata_check.h](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_host/quant_lightning_indexer_v2_metadata_check.h) | 量化索引元数据 的主机侧辅助接口或实现。 | 函数：`IsTensorExistQliV2`, `GetDimNumQliV2` |
| [quant_lightning_indexer_v2_metadata_infershape.cpp](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_host/quant_lightning_indexer_v2_metadata_infershape.cpp) | 量化索引元数据 的主机侧辅助接口或实现。 | 函数：`InferShapeQuantLightningIndexerV2Metadata`, `InferDtypeQuantLightningIndexerV2Metadata` |

## `csrc/attention/quant_lightning_indexer_v2_metadata/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_quant_lightning_indexer_v2_metadata.cpp](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_host/op_api/aclnn_quant_lightning_indexer_v2_metadata.cpp) | 量化索引元数据 的主机侧辅助接口或实现。 | 函数：`aclnnVllmAscendQuantLightningIndexerV2MetadataGetWorkspaceSize`, `aclnnVllmAscendQuantLightningIndexerV2Metadata` |
| [aclnn_quant_lightning_indexer_v2_metadata.h](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_host/op_api/aclnn_quant_lightning_indexer_v2_metadata.h) | 量化索引元数据 的主机侧辅助接口或实现。 | 函数：`aclnnVllmAscendQuantLightningIndexerV2MetadataGetWorkspaceSize`, `aclnnVllmAscendQuantLightningIndexerV2Metadata` |
| [quant_lightning_indexer_v2_metadata.cpp](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_host/op_api/quant_lightning_indexer_v2_metadata.cpp) | 量化索引元数据 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [quant_lightning_indexer_v2_metadata.h](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_host/op_api/quant_lightning_indexer_v2_metadata.h) | 量化索引元数据 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/`

op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [quant_lightning_indexer_v2_metadata_aicpu.cpp](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/quant_lightning_indexer_v2_metadata_aicpu.cpp) | 元数据、历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [quant_lightning_indexer_v2_metadata_aicpu.h](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/quant_lightning_indexer_v2_metadata_aicpu.h) | 元数据、历史位置索引、量化 的公共接口/实现，供所在算子工程复用。 | 类型：`SparseMode`, `ValidSocVersion`, `FlashDecodeResult` |
| [quant_lightning_indexer_v2_metadata_aicpu.json](../../../csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/quant_lightning_indexer_v2_metadata_aicpu.json) | 所在模块的结构化配置或参数模板。 | 顶层项：`VllmAscendQuantLightningIndexerV2Metadata` |

## `csrc/attention/recurrent_gated_delta_rule_v310/`

310P gated delta 递归状态 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/recurrent_gated_delta_rule_v310/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [recurrent_gated_delta_rule_310_torch_adpt.h](../../../csrc/attention/recurrent_gated_delta_rule_v310/recurrent_gated_delta_rule_310_torch_adpt.h) | Delta 递归状态、门控 的公共接口/实现，供所在算子工程复用。 | 函数：`npu_recurrent_gated_delta_rule_310` |

## `csrc/attention/recurrent_gated_delta_rule_v310/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [math_util.h](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/math_util.h) | 310P gated delta 递归状态 的主机侧辅助接口或实现。 | 类型：`MathUtil` |
| [recurrent_gated_delta_rule_v310.h](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/recurrent_gated_delta_rule_v310.h) | 310P gated delta 递归状态 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [recurrent_gated_delta_rule_v310_def.cpp](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/recurrent_gated_delta_rule_v310_def.cpp) | 310P gated delta 递归状态 的主机算子定义、支持属性和注册。 | 注册：`RecurrentGatedDeltaRuleV310` |
| [recurrent_gated_delta_rule_v310_infershape.cpp](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/recurrent_gated_delta_rule_v310_infershape.cpp) | 310P gated delta 递归状态 的主机侧辅助接口或实现。 | 函数：`InferShapeRecurrentGatedDeltaRuleV310`, `InferDataTypeRecurrentGatedDeltaRuleV310` |
| [recurrent_gated_delta_rule_v310_tiling.cpp](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/recurrent_gated_delta_rule_v310_tiling.cpp) | 310P gated delta 递归状态 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`RuleItem` |
| [recurrent_gated_delta_rule_v310_tiling.h](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/recurrent_gated_delta_rule_v310_tiling.h) | 310P gated delta 递归状态 的主机侧 tiling 参数/分块策略声明。 | 类型：`RecurrentGatedDeltaRuleV310CompileInfo`, `RecurrentGatedDeltaRuleV310Info`, `RecurrentGatedDeltaRuleV310Tiling` |

## `csrc/attention/recurrent_gated_delta_rule_v310/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_recurrent_gated_delta_rule_v310.cpp](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/op_api/aclnn_recurrent_gated_delta_rule_v310.cpp) | 310P gated delta 递归状态 的主机侧辅助接口或实现。 | 类型：`RecurrentGatedDeltaRuleV310Params` |
| [aclnn_recurrent_gated_delta_rule_v310.h](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/op_api/aclnn_recurrent_gated_delta_rule_v310.h) | 310P gated delta 递归状态 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [recurrent_gated_delta_rule_v310.cpp](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_host/op_api/recurrent_gated_delta_rule_v310.cpp) | 310P gated delta 递归状态 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/recurrent_gated_delta_rule_v310/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [recurrent_gated_delta_rule_v310.cpp](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_kernel/recurrent_gated_delta_rule_v310.cpp) | 310P gated delta 递归状态 的设备执行源码；文件主题为 Delta 递归状态、门控。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [recurrent_gated_delta_rule_v310.h](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_kernel/recurrent_gated_delta_rule_v310.h) | 310P gated delta 递归状态 的类型/模板与设备实现头文件；文件主题为 Delta 递归状态、门控。 | 类型：`RGDRInitParams`, `RGDR` |
| [recurrent_gated_delta_rule_v310_tiling_data.h](../../../csrc/attention/recurrent_gated_delta_rule_v310/op_kernel/recurrent_gated_delta_rule_v310_tiling_data.h) | 310P gated delta 递归状态 的类型/模板与设备实现头文件；文件主题为 Delta 递归状态、门控。 | 类型：`alignas` |

## `csrc/attention/rms_norm_dynamic_quant/`

RMSNorm 与动态量化融合 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/rms_norm_dynamic_quant/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/rms_norm_dynamic_quant/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/rms_norm_dynamic_quant/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/attention/rms_norm_dynamic_quant/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/rms_norm_dynamic_quant/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [rms_norm_dynamic_quant_def.cpp](../../../csrc/attention/rms_norm_dynamic_quant/op_host/rms_norm_dynamic_quant_def.cpp) | RMSNorm 与动态量化融合 的主机算子定义、支持属性和注册。 | 注册：`RmsNormDynamicQuant` |
| [rms_norm_dynamic_quant_proto.cpp](../../../csrc/attention/rms_norm_dynamic_quant/op_host/rms_norm_dynamic_quant_proto.cpp) | RMSNorm 与动态量化融合 的算子原型、输入输出或类型/shape 推断注册。 | 注册：`RmsNormDynamicQuant` |
| [rms_norm_dynamic_quant_tiling.cpp](../../../csrc/attention/rms_norm_dynamic_quant/op_host/rms_norm_dynamic_quant_tiling.cpp) | RMSNorm 与动态量化融合 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`CeilDiv`, `CeilAlign` |
| [rms_norm_dynamic_quant_tiling.h](../../../csrc/attention/rms_norm_dynamic_quant/op_host/rms_norm_dynamic_quant_tiling.h) | RMSNorm 与动态量化融合 的主机侧 tiling 参数/分块策略声明。 | 类型：`RmsNormDynamicQuantCompileInfo`, `UB_TILING_POLICY`, `RmsNormDynamicQuantTilingHelper` |

## `csrc/attention/rms_norm_dynamic_quant/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [reduce_common.h](../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/reduce_common.h) | RMSNorm 与动态量化融合 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 函数：`ReduceSumForSmallReduceDimPreRepeat`, `ReduceSumForSmallReduceDim` |
| [rms_norm_dynamic_quant.cpp](../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant.cpp) | RMSNorm 与动态量化融合 的设备执行源码；文件主题为 RMS 归一化、量化。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [rms_norm_dynamic_quant_base.h](../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_base.h) | RMSNorm 与动态量化融合 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、量化。 | 类型：`KernelAddRmsNormDynamicQuantBase` |
| [rms_norm_dynamic_quant_cut_d_kernel.h](../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_cut_d_kernel.h) | RMSNorm 与动态量化融合 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、量化。 | 类型：`KernelAddRmsNormDynamicQuantSliceD` |
| [rms_norm_dynamic_quant_helper.h](../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_helper.h) | RMSNorm 与动态量化融合 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、量化。 | 类型：`integral_constant`, `is_same` |
| [rms_norm_dynamic_quant_normal_kernel.h](../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_normal_kernel.h) | RMSNorm 与动态量化融合 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、量化。 | 类型：`KernelAddRmsNormDynamicQuantNormal` |
| [rms_norm_dynamic_quant_single_row_kernel.h](../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_single_row_kernel.h) | RMSNorm 与动态量化融合 的类型/模板与设备实现头文件；文件主题为 RMS 归一化、量化。 | 类型：`KernelAddRmsNormDynamicQuantSingleRow` |

## `csrc/attention/sparse_attention_score/`

注意力 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/attention/sparse_attention_score/docs) | docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/sparse_attention_score/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/sparse_attention_score/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |
| [tests/](../../../csrc/attention/sparse_attention_score/tests) | tests 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [tools/](../../../csrc/attention/sparse_attention_score/tools) | tools 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_attention_score/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/sparse_attention_score/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：SparseAttentionScore |
| [sparse_attention_score_torch_adpt.h](../../../csrc/attention/sparse_attention_score/sparse_attention_score_torch_adpt.h) | 注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`CheckFp8Tensor`, `CheckDequantScaleTensor` |

## `csrc/attention/sparse_attention_score/docs/`

docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_attention_score_design.md](../../../csrc/attention/sparse_attention_score/docs/sparse_attention_score_design.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：sparse_attention_score_design。 | 标题：SparseAttentionScore Operator Design |

## `csrc/attention/sparse_attention_score/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/sparse_attention_score/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_attention_score/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [sparse_attention_score_def.cpp](../../../csrc/attention/sparse_attention_score/op_host/sparse_attention_score_def.cpp) | 注意力 的主机算子定义、支持属性和注册。 | 注册：`SparseAttentionScore` |
| [sparse_attention_score_infershape.cpp](../../../csrc/attention/sparse_attention_score/op_host/sparse_attention_score_infershape.cpp) | 注意力 的主机侧辅助接口或实现。 | 函数：`InferShapeSparseAttentionScore`, `InferDataTypeSparseAttentionScore` |
| [sparse_attention_score_tiling.cpp](../../../csrc/attention/sparse_attention_score/op_host/sparse_attention_score_tiling.cpp) | 注意力 的主机侧 tiling 参数/分块策略计算实现。 | 函数：`CeilDiv`, `AlignUp` |
| [sparse_attention_score_tiling.h](../../../csrc/attention/sparse_attention_score/op_host/sparse_attention_score_tiling.h) | 注意力 的主机侧 tiling 参数/分块策略声明。 | 类型：`SparseAttentionScoreCompileInfo`, `SASAFdCoreRange`, `SASAFdCombineRange` |

## `csrc/attention/sparse_attention_score/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_sparse_attention_score.cpp](../../../csrc/attention/sparse_attention_score/op_host/op_api/aclnn_sparse_attention_score.cpp) | 注意力 的主机侧辅助接口或实现。 | 函数：`MakeContiguous` |
| [aclnn_sparse_attention_score.h](../../../csrc/attention/sparse_attention_score/op_host/op_api/aclnn_sparse_attention_score.h) | 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sparse_attention_score.cpp](../../../csrc/attention/sparse_attention_score/op_host/op_api/sparse_attention_score.cpp) | 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sparse_attention_score.h](../../../csrc/attention/sparse_attention_score/op_host/op_api/sparse_attention_score.h) | 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/sparse_attention_score/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch22/](../../../csrc/attention/sparse_attention_score/op_kernel/arch22) | arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/sparse_attention_score/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [attn_infra/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra) | 注意力 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [tla/](../../../csrc/attention/sparse_attention_score/op_kernel/tla) | tla 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kernel_common.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/kernel_common.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 公共逻辑。 | 类型：`SparseAttentionScoreTilingData` |
| [sparse_attention_score.cpp](../../../csrc/attention/sparse_attention_score/op_kernel/sparse_attention_score.cpp) | 注意力 的设备执行源码。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sparse_attention_score_kernel_interface.cpp](../../../csrc/attention/sparse_attention_score/op_kernel/sparse_attention_score_kernel_interface.cpp) | 注意力 的设备执行源码。 | 类型：`InDtype`, `SMDtype`, `REDtype` |
| [sparse_attention_score_tilingkey.h](../../../csrc/attention/sparse_attention_score/op_kernel/sparse_attention_score_tilingkey.h) | 注意力 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/sparse_attention_score/op_kernel/arch22/`

arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kernel_utils.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/arch22/kernel_utils.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 辅助函数。 | 类型：`SasaKernelParamsArch22` |
| [sparse_attention_score_fd_combine_arch22.h](../../../csrc/attention/sparse_attention_score/op_kernel/arch22/sparse_attention_score_fd_combine_arch22.h) | 注意力 的类型/模板与设备实现头文件；文件主题为 注意力、token 合并。 | 类型：`ElementO`, `Resource`, `SparseAttentionScoreFdCombineArch22` |
| [sparse_attention_score_kernel_arch22.h](../../../csrc/attention/sparse_attention_score/op_kernel/arch22/sparse_attention_score_kernel_arch22.h) | 注意力 的类型/模板与设备实现头文件。 | 类型：`BlockMmadQK`, `EpilogueOnlineSoftmax`, `BlockMmadPV` |

## `csrc/attention/sparse_attention_score/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kernel_utils.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/arch35/kernel_utils.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 辅助函数。 | 类型：`Format`, `SasaKernelParamsArch35`, `SasaFullQuantKernelParamsArch35` |
| [sparse_attention_score_fd_combine_arch35.h](../../../csrc/attention/sparse_attention_score/op_kernel/arch35/sparse_attention_score_fd_combine_arch35.h) | 注意力 的类型/模板与设备实现头文件；文件主题为 注意力、token 合并。 | 类型：`ElementO`, `Resource`, `SparseAttentionScoreFdCombineArch35` |
| [sparse_attention_score_kernel_arch35.h](../../../csrc/attention/sparse_attention_score/op_kernel/arch35/sparse_attention_score_kernel_arch35.h) | 注意力 的类型/模板与设备实现头文件。 | 类型：`BlockMmadQK`, `EpilogueOnlineSoftmax`, `BlockMmadPV` |
| [sparse_attention_score_kernel_arch35_full_quant.h](../../../csrc/attention/sparse_attention_score/op_kernel/arch35/sparse_attention_score_kernel_arch35_full_quant.h) | 注意力 的类型/模板与设备实现头文件；文件主题为 注意力、量化。 | 类型：`BlockMmadQK`, `EpilogueOnlineSoftmax`, `BlockMmadPV` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/`

注意力 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [arch/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/arch) | arch 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [detail/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/detail) | detail 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [epilogue/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue) | epilogue 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [gemm/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm) | gemm 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [layout/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/layout) | 内存布局 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [base_defs.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/base_defs.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`EmptyClass` |
| [coord.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/coord.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`Index_`, `LongIndex_`, `Coord` |
| [gemm_coord.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm_coord.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`GemmShape`, `GemmCoord` |
| [matrix_coord.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/matrix_coord.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`MatrixShape`, `MatrixCoord` |
| [status.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/status.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`Status` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/arch/`

arch 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [arch.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/arch/arch.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`AtlasA2`, `AtlasA5` |
| [cross_core_sync.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/arch/cross_core_sync.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`CrossCoreFlag`, `CrossCoreFlagWithReverse`, `BarrierFlag` |
| [local_tensor_buffer.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/arch/local_tensor_buffer.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 张量。 | 类型：`LocalTensorBufferBase`, `Element`, `ArchTag` |
| [resource.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/arch/resource.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`ArchTag`, `Resource` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/detail/`

detail 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [alignment.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/detail/alignment.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`U` |
| [dependent_false.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/detail/dependent_false.hpp) | 注意力 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [macros.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/detail/macros.hpp) | 注意力 的类型/模板与设备实现头文件。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [tag_to_layout.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/detail/tag_to_layout.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 内存布局。 | 类型：`Element`, `LayoutTag`, `TagToLayout` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/`

epilogue 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block) | 缓存页/分块 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [tile_common/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/tile_common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [dispatch_policy.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/dispatch_policy.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 token 分发。 | 类型：`LseMode`, `EpilogueAtlasA2OnlineSoftmax`, `EpilogueAtlasA2RescaleO` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/`

缓存页/分块 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_epilogue.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`DispatchPolicy`, `BlockEpilogue` |
| [block_epilogue_mask2idx_arch35.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_mask2idx_arch35.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`ElementSparseMask_`, `ElementSparseIdx_`, `ElementSparseCount_` |
| [block_epilogue_online_softmax.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 概率归一化、缓存页/分块。 | 类型：`OutputType_`, `InputType_`, `MaskType_` |
| [block_epilogue_online_softmax_arch35_reg_low_prec.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax_arch35_reg_low_prec.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 概率归一化、缓存页/分块。 | 类型：`KvBaseTileRegSplitStages`, `OutputType_`, `LayoutS_` |
| [block_epilogue_online_softmax_arch35_reg_low_prec_bf16.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax_arch35_reg_low_prec_bf16.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 概率归一化、缓存页/分块。 | 类型：`KvBaseTileRegSplitStagesBf16`, `OutputType_`, `LayoutS_` |
| [block_epilogue_online_softmax_low_prec.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax_low_prec.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 概率归一化、缓存页/分块。 | 类型：`OutputType_`, `InputType_`, `MaskType_` |
| [block_epilogue_rescale_o.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`OutputType_`, `InputType_`, `UpdateType_` |
| [block_epilogue_rescale_o_arch35_reg_high_prec.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o_arch35_reg_high_prec.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`DRegSplitStages`, `ElementO_`, `ElementOTmp_` |
| [block_epilogue_rescale_o_low_prec.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o_low_prec.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`OutputType_`, `InputType_`, `UpdateType_` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/tile_common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [copy_gm_to_ub.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/tile_common/copy_gm_to_ub.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `GmType`, `CopyGm2Ub` |
| [copy_gm_to_ub_tla.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/tile_common/copy_gm_to_ub_tla.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `TensorSrc`, `TensorDst` |
| [copy_ub_to_gm.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/tile_common/copy_ub_to_gm.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `GmType`, `CopyUb2Gm` |
| [copy_ub_to_gm_tla.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/tile_common/copy_ub_to_gm_tla.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `TensorSrc`, `TensorDst` |
| [tile_copy.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/tile_common/tile_copy.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `TileCopy`, `CType` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/`

gemm 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [block/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/block) | 缓存页/分块 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [tile_common/](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [dispatch_policy.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/dispatch_policy.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 token 分发。 | 类型：`MmadAtlasA2Base`, `MmadAtlasA5Base`, `MmadAtlasA2SFAIQK` |
| [gemm_type.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/gemm_type.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`Element_`, `Layout_`, `GemmType` |
| [helper.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/helper.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`Element`, `Layout`, `L1AlignHelper` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/block/`

缓存页/分块 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [block_mmad.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/block/block_mmad.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`DispatchPolicy`, `L1TileShape`, `L0TileShape` |
| [block_mmad_pv.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/block/block_mmad_pv.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`L1TileShape_`, `L0TileShape_`, `AType_` |
| [block_mmad_pv_arch35_ABf16_C_to_UB.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`Mm2L1TileHelper`, `L1TileShape_`, `L0TileShape_` |
| [block_mmad_qk.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/block/block_mmad_qk.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`L1TileShape_`, `L0TileShape_`, `AType_` |
| [block_mmad_qk_arch35_ABf16_C_to_UB.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/block/block_mmad_qk_arch35_ABf16_C_to_UB.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块。 | 类型：`Mm1L1TileHelper`, `L1TileShape_`, `L0TileShape_` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [copy_gm_to_l1_a2.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_gm_to_l1_a2.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `GmType`, `L1Type` |
| [copy_gm_to_l1_a5.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_gm_to_l1_a5.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ElementSrc`, `ElementDst`, `LayoutSrc` |
| [copy_gm_to_ub.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_gm_to_ub.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [copy_l0c_to_dst.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_l0c_to_dst.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ScaleGranularity`, `ArchTag`, `ElementSrc` |
| [copy_l0c_to_gm_a2.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_l0c_to_gm_a2.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ElementAccumulator_`, `ElementDst_`, `CopyL0CToGm` |
| [copy_l0c_to_ub_a5.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_l0c_to_ub_a5.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`TensorSrc_`, `ElementDst_`, `LayoutDst_` |
| [copy_l1_to_bt.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_l1_to_bt.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [copy_l1_to_l0a_a2.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0a_a2.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `L1Type`, `L0Type` |
| [copy_l1_to_l0a_a5.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0a_a5.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ElementSrc`, `ElementDst`, `LayoutSrc` |
| [copy_l1_to_l0b_a2.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0b_a2.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `L1Type`, `L0Type` |
| [copy_l1_to_l0b_a5.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0b_a5.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ElementSrc`, `ElementDst`, `LayoutSrc` |
| [copy_ub_to_gm.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/copy_ub_to_gm.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [tile_copy.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/tile_copy.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `AType`, `BType` |
| [tile_copy_tla.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/tile_copy_tla.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 复制。 | 类型：`ArchTag`, `TensorSrc`, `TensorDst` |
| [tile_mmad.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/gemm/tile_common/tile_mmad.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`ArchTag_`, `AType_`, `BType_` |

## `csrc/attention/sparse_attention_score/op_kernel/attn_infra/layout/`

内存布局 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [layout.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/layout/layout.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 内存布局。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [matrix.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/layout/matrix.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`RowMajor`, `Element`, `ColumnMajor` |
| [vector.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/attn_infra/layout/vector.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`VectorLayout`, `Element` |

## `csrc/attention/sparse_attention_score/op_kernel/tla/`

tla 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [numeric/](../../../csrc/attention/sparse_attention_score/op_kernel/tla/numeric) | numeric 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [int_tuple.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/tla/int_tuple.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`F`, `G`, `T0` |
| [layout.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/tla/layout.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 内存布局。 | 类型：`Shape`, `Stride`, `Layout` |
| [tensor.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/tla/tensor.hpp) | 注意力 的类型/模板与设备实现头文件；文件主题为 张量。 | 类型：`BuiltinTensor`, `Layout_`, `Coord_` |
| [tuple.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/tla/tuple.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`EBO`, `IdxSeq`, `TupleBase` |
| [type_traits.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/tla/type_traits.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`remove_cvref`, `tuple_size` |

## `csrc/attention/sparse_attention_score/op_kernel/tla/numeric/`

numeric 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [integer_sequence.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/tla/numeric/integer_sequence.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`IntegerSequence`, `MakeIntegerSequenceImpl`, `Tuple` |
| [integral_constant.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/tla/numeric/integral_constant.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`C`, `integral_constant`, `is_integral` |
| [math.hpp](../../../csrc/attention/sparse_attention_score/op_kernel/tla/numeric/math.hpp) | 注意力 的类型/模板与设备实现头文件。 | 类型：`U` |

## `csrc/attention/sparse_attention_score/tests/`

tests 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [ut/](../../../csrc/attention/sparse_attention_score/tests/ut) | ut 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_attention_score/tests/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [FLASH_DECODING_GENERALIZED_TEST.md](../../../csrc/attention/sparse_attention_score/tests/FLASH_DECODING_GENERALIZED_TEST.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：FLASH_DECODING_GENERALIZED_TEST。 | 标题：SparseAttentionScore Arch35 FlashDecoding Generalization Tests |
| [run_bf16_fd_minimal.py](../../../csrc/attention/sparse_attention_score/tests/run_bf16_fd_minimal.py) | 算子工程 run_bf16_fd_minimal 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Run one BF16 shape that automatically selects FlashDecoding.；定义：`run_once`, `main` |
| [test_bf16.py](../../../csrc/attention/sparse_attention_score/tests/test_bf16.py) | 算子工程 test_bf16 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`TestNpuSparseAttentionScoreBf16`, `generate_block_index_with_causal`, `generate_block_table`, `build_fp8_tensors_and_scales` |
| [test_flash_decoding_generalized.py](../../../csrc/attention/sparse_attention_score/tests/test_flash_decoding_generalized.py) | 算子工程 test_flash_decoding_generalized 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Generalized Arch35 FlashDecoding functional and accuracy tests.；定义：`FDCase`, `inputs_shape`, `test_fd_generalized_accuracy`, `test_fd_automatic_fallback_accuracy` |
| [test_fp16.py](../../../csrc/attention/sparse_attention_score/tests/test_fp16.py) | 算子工程 test_fp16 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`TestNpuSparseAttentionScoreFp16`, `generate_block_index_with_causal`, `generate_block_table`, `cpu_sparse_attention_score_fp32` |
| [test_sparse_attention_score_fp8.py](../../../csrc/attention/sparse_attention_score/tests/test_sparse_attention_score_fp8.py) | 算子工程 注意力 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`TestNpuSparseAttentionScoreFp8`, `generate_block_index_with_causal`, `generate_block_table`, `build_fp8_tensors_and_scales` |

## `csrc/attention/sparse_attention_score/tests/ut/`

ut 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/sparse_attention_score/tests/ut/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_attention_score/tests/ut/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/attention/sparse_attention_score/tests/ut/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_attention_score/tests/ut/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [test_sparse_attention_score_fd_tiling.cpp](../../../csrc/attention/sparse_attention_score/tests/ut/op_host/test_sparse_attention_score_fd_tiling.cpp) | 注意力 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`SparseAttentionScoreFdTilingTest` |

## `csrc/attention/sparse_attention_score/tools/`

tools 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fix_mmad_deq.py](../../../csrc/attention/sparse_attention_score/tools/fix_mmad_deq.py) | 算子工程 fix_mmad_deq 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`fix_mmad` |

## `csrc/attention/sparse_attn_sharedkv/`

共享 KV 稀疏注意力 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/sparse_attn_sharedkv/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/sparse_attn_sharedkv/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_attn_sharedkv/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/sparse_attn_sharedkv/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：SparseAttnSharedkv |

## `csrc/attention/sparse_attn_sharedkv/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_attn_sharedkv/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [sparse_attn_sharedkv_def.cpp](../../../csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_def.cpp) | 共享 KV 稀疏注意力 的主机算子定义、支持属性和注册。 | 注册：`SparseAttnSharedkv` |
| [sparse_attn_sharedkv_proto.cpp](../../../csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_proto.cpp) | 共享 KV 稀疏注意力 的算子原型、输入输出或类型/shape 推断注册。 | 函数：`InferShapeSparseAttnSharedkv`, `InferDataTypeSparseFlashAttention` |
| [sparse_attn_sharedkv_tiling.cpp](../../../csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp) | 共享 KV 稀疏注意力 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`SASCompileInfo` |
| [sparse_attn_sharedkv_tiling.h](../../../csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.h) | 共享 KV 稀疏注意力 的主机侧 tiling 参数/分块策略声明。 | 类型：`SASTilingRequiredParaInfo`, `SASTilingOptionalParaInfo`, `SASLayout` |

## `csrc/attention/sparse_attn_sharedkv/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch32/](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32) | arch32 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_attn_sharedkv.cpp](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv.cpp) | 共享 KV 稀疏注意力 的设备执行源码；文件主题为 注意力。 | 函数：`sparse_attn_sharedkv` |
| [sparse_attn_sharedkv_common.h](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_common.h) | 共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 公共逻辑、注意力。 | 类型：`SAS_RUN_MODE`, `SAS_LAYOUT`, `SASType` |
| [sparse_attn_sharedkv_metadata.h](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_metadata.h) | 共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 元数据、注意力。 | 类型：`SasMetaData` |
| [sparse_attn_sharedkv_template_tiling_key.h](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_template_tiling_key.h) | 共享 KV 稀疏注意力 的kernel 分派使用的 tiling key 定义；文件主题为 注意力。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/`

arch32 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_attn_sharedkv_scfa_block_cube.h](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_cube.h) | 共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、注意力。 | 类型：`SASCubeBlock` |
| [sparse_attn_sharedkv_scfa_block_vector.h](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_vector.h) | 共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、注意力。 | 类型：`SASVectorBlock`, `ConstInfo` |
| [sparse_attn_sharedkv_scfa_kernel.h](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_kernel.h) | 共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 注意力。 | 类型：`TempLoopInfo`, `SparseAttnSharedkvScfa` |
| [sparse_attn_sharedkv_swa_block_cube.h](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_block_cube.h) | 共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、注意力。 | 类型：`SWACubeBlock` |
| [sparse_attn_sharedkv_swa_block_vector.h](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_block_vector.h) | 共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、注意力。 | 类型：`SWAVectorBlock`, `ConstInfo` |
| [sparse_attn_sharedkv_swa_kernel.h](../../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_kernel.h) | 共享 KV 稀疏注意力 的类型/模板与设备实现头文件；文件主题为 注意力。 | 类型：`SwaTempLoopInfo`, `SparseAttnSharedkvSwa` |

## `csrc/attention/sparse_attn_sharedkv_metadata/`

共享 KV 稀疏元数据 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_api) | 算子对外调用接口及适配。 |
| [op_graph/](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_graph) | 设备图 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel_aicpu/](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu) | op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_attn_sharedkv_metadata/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/sparse_attn_sharedkv_metadata/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：SparseAttnSharedkvMetadata |

## `csrc/attention/sparse_attn_sharedkv_metadata/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_sparse_attn_sharedkv_metadata.cpp](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_api/aclnn_sparse_attn_sharedkv_metadata.cpp) | 元数据、注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`ParamsCheck`, `aclnnSparseAttnSharedkvMetadataGetWorkspaceSize` |
| [aclnn_sparse_attn_sharedkv_metadata.h](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_api/aclnn_sparse_attn_sharedkv_metadata.h) | 元数据、注意力 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [l0_sparse_attn_sharedkv_metadata.cpp](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_api/l0_sparse_attn_sharedkv_metadata.cpp) | 元数据、注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`SparseAttnSharedkvMetadata` |
| [l0_sparse_attn_sharedkv_metadata.h](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_api/l0_sparse_attn_sharedkv_metadata.h) | 元数据、注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`SparseAttnSharedkvMetadata` |

## `csrc/attention/sparse_attn_sharedkv_metadata/op_graph/`

设备图 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_attn_sharedkv_metadata_proto.h](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_graph/sparse_attn_sharedkv_metadata_proto.h) | 元数据、注意力 的公共接口/实现，供所在算子工程复用。 | 注册：`SparseAttnSharedkvMetadata` |

## `csrc/attention/sparse_attn_sharedkv_metadata/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_attn_sharedkv_metadata_infershape.cpp](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_host/sparse_attn_sharedkv_metadata_infershape.cpp) | 共享 KV 稀疏元数据 的主机侧辅助接口或实现。 | 函数：`InferShapeSparseAttnSharedkvMetadata`, `InferDtypeSparseAttnSharedkvMetadata` |

## `csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/`

op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_attn_sharedkv_metadata_aicpu.cpp](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/sparse_attn_sharedkv_metadata_aicpu.cpp) | 元数据、注意力 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sparse_attn_sharedkv_metadata_aicpu.h](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/sparse_attn_sharedkv_metadata_aicpu.h) | 元数据、注意力 的公共接口/实现，供所在算子工程复用。 | 类型：`SparseMode`, `ValidSocVersion`, `FlashDecodeResult` |
| [sparse_attn_sharedkv_metadata_aicpu.json](../../../csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/sparse_attn_sharedkv_metadata_aicpu.json) | 所在模块的结构化配置或参数模板。 | 顶层项：`SparseAttnSharedkvMetadata` |

## `csrc/attention/sparse_flash_attention/`

稀疏 Flash Attention 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/attention/sparse_flash_attention/docs) | docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [examples/](../../../csrc/attention/sparse_flash_attention/examples) | examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/sparse_flash_attention/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/sparse_flash_attention/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_flash_attention/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/sparse_flash_attention/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：SparseFlashAttention |
| [sparse_flash_attention_torch_adpt.h](../../../csrc/attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h) | 注意力 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/sparse_flash_attention/docs/`

docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnnSparseFlashAttention.md](../../../csrc/attention/sparse_flash_attention/docs/aclnnSparseFlashAttention.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：aclnnSparseFlashAttention。 | 标题：aclnnSparseFlashAttention |

## `csrc/attention/sparse_flash_attention/examples/`

examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_aclnn_sparse_flash_attention.cpp](../../../csrc/attention/sparse_flash_attention/examples/test_aclnn_sparse_flash_attention.cpp) | 注意力 的公共接口/实现，供所在算子工程复用。 | 类型：`TensorResources` |

## `csrc/attention/sparse_flash_attention/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/sparse_flash_attention/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_flash_attention/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [sparse_flash_attention_def.cpp](../../../csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp) | 稀疏 Flash Attention 的主机算子定义、支持属性和注册。 | 注册：`SparseFlashAttention` |
| [sparse_flash_attention_infershape.cpp](../../../csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_infershape.cpp) | 稀疏 Flash Attention 的主机侧辅助接口或实现。 | 函数：`InferShapeSparseFlashAttention`, `InferDataTypeSparseFlashAttention` |
| [sparse_flash_attention_tiling.cpp](../../../csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp) | 稀疏 Flash Attention 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`SparseFlashAttentionCompileInfo` |
| [sparse_flash_attention_tiling.h](../../../csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.h) | 稀疏 Flash Attention 的主机侧 tiling 参数/分块策略声明。 | 类型：`SFALayout`, `SFATilingShapeCompareParam`, `KvStorageMode` |

## `csrc/attention/sparse_flash_attention/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_sparse_flash_attention.cpp](../../../csrc/attention/sparse_flash_attention/op_host/op_api/aclnn_sparse_flash_attention.cpp) | 稀疏 Flash Attention 的主机侧辅助接口或实现。 | 类型：`TensorHolder` |
| [aclnn_sparse_flash_attention.h](../../../csrc/attention/sparse_flash_attention/op_host/op_api/aclnn_sparse_flash_attention.h) | 稀疏 Flash Attention 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/sparse_flash_attention/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch22/](../../../csrc/attention/sparse_flash_attention/op_kernel/arch22) | arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/sparse_flash_attention/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_flash_attention.cpp](../../../csrc/attention/sparse_flash_attention/op_kernel/sparse_flash_attention.cpp) | 稀疏 Flash Attention 的设备执行源码；文件主题为 注意力。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sparse_flash_attention_common.h](../../../csrc/attention/sparse_flash_attention/op_kernel/sparse_flash_attention_common.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、公共逻辑。 | 类型：`SFA_LAYOUT`, `SFAType`, `RunInfo` |
| [sparse_flash_attention_template_tiling_key.h](../../../csrc/attention/sparse_flash_attention/op_kernel/sparse_flash_attention_template_tiling_key.h) | 稀疏 Flash Attention 的kernel 分派使用的 tiling key 定义；文件主题为 注意力。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/sparse_flash_attention/op_kernel/arch22/`

arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_flash_attention_kernel_mla.h](../../../csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`TempLoopInfo`, `SparseFlashAttentionMla` |
| [sparse_flash_attention_service_cube_mla.h](../../../csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_cube_mla.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`PAShape`, `Position`, `SFAMatmulService` |
| [sparse_flash_attention_service_vector_mla.h](../../../csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`SFAVectorService`, `ConstInfo` |

## `csrc/attention/sparse_flash_attention/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_flash_attention_common_arch35.h](../../../csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_common_arch35.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、公共逻辑。 | 类型：`SFA_LAYOUT`, `SFATemplateMode` |
| [sparse_flash_attention_kernel_mla.h](../../../csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`SparseFlashAttentionKernelMla` |
| [sparse_flash_attention_kvcache.h](../../../csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kvcache.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、缓存、KV 缓存。 | 函数：`CalculateQueryOffset`, `GetSingleCoreParam` |
| [sparse_flash_attention_service_cube_mla.h](../../../csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_cube_mla.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`SFAMatmulService`, `SFAMatmulServiceDummy`, `CubeBlockTraits` |
| [sparse_flash_attention_service_vector_mla.h](../../../csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件；文件主题为 注意力、MLA 潜在注意力。 | 类型：`SFAVectorService`, `SFAVectorServiceDummy` |
| [util_regbase.h](../../../csrc/attention/sparse_flash_attention/op_kernel/arch35/util_regbase.h) | 稀疏 Flash Attention 的类型/模板与设备实现头文件。 | 类型：`RunParamStr`, `RunInfo`, `ConstInfo` |

## `csrc/attention/sparse_flash_mla/`

稀疏 MLA 注意力 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/attention/sparse_flash_mla/docs) | docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [examples/](../../../csrc/attention/sparse_flash_mla/examples) | examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/sparse_flash_mla/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/sparse_flash_mla/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_flash_mla/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/sparse_flash_mla/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：SparseFlashMla |
| [sparse_flash_mla_torch_adpt.h](../../../csrc/attention/sparse_flash_mla/sparse_flash_mla_torch_adpt.h) | MLA 潜在注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`GetValidSparseFlashMlaTensor`, `npu_sparse_flash_mla_metadata` |

## `csrc/attention/sparse_flash_mla/docs/`

docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnnSparseFlashMla.md](../../../csrc/attention/sparse_flash_mla/docs/aclnnSparseFlashMla.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：aclnnSparseFlashMla。 | 标题：aclnnSparseFlashMla |
| [design.md](../../../csrc/attention/sparse_flash_mla/docs/design.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：design。 | 标题：SparseFlashMla 算子设计与开发指南 |
| [ratio2_a2a3.md](../../../csrc/attention/sparse_flash_mla/docs/ratio2_a2a3.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：ratio2_a2a3。 | 标题：A2/A3 cmp_ratio=1/2 适配说明 |

## `csrc/attention/sparse_flash_mla/examples/`

examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_aclnn_sparse_flash_mla.cpp](../../../csrc/attention/sparse_flash_mla/examples/test_aclnn_sparse_flash_mla.cpp) | MLA 潜在注意力 的公共接口/实现，供所在算子工程复用。 | 函数：`GetShapeSize`, `FloatToFp16` |

## `csrc/attention/sparse_flash_mla/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [checkers/](../../../csrc/attention/sparse_flash_mla/op_host/checkers) | 检查 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_flash_mla/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [sparse_flash_mla_def.cpp](../../../csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_def.cpp) | 稀疏 MLA 注意力 的主机算子定义、支持属性和注册。 | 注册：`SparseFlashMla` |
| [sparse_flash_mla_infershape.cpp](../../../csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_infershape.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 函数：`ToVectorFunc`, `ToStringFunc` |
| [sparse_flash_mla_tiling.cpp](../../../csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_tiling.cpp) | 稀疏 MLA 注意力 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`SMLACompileInfo` |
| [sparse_flash_mla_tiling.h](../../../csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_tiling.h) | 稀疏 MLA 注意力 的主机侧 tiling 参数/分块策略声明。 | 类型：`SMLATilingRequiredParaInfo`, `SMLATilingOptionalParaInfo`, `SMLALayout` |

## `csrc/attention/sparse_flash_mla/op_host/checkers/`

检查 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [base_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/base_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 函数：`ShapeToString`, `ValuesToString` |
| [base_checker_sparse_flash_mla.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/base_checker_sparse_flash_mla.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`BaseChecker` |
| [checker_adapter.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/checker_adapter.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 函数：`PopulateOptionalTensorParam`, `MakeRequiredTensor` |
| [checker_context.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/checker_context.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`OperatorVariant`, `Layout`, `TensorParam` |
| [checker_runner.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/checker_runner.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 函数：`RegisterCommonCheckers` |
| [checker_runner.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/checker_runner.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`CheckerRunner` |
| [checker_sources.cmake](../../../csrc/attention/sparse_flash_mla/op_host/checkers/checker_sources.cmake) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 构建项：`add_sparse_mla_common_checker_sources` |
| [common_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/common_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [common_checker_sparse_flash_mla.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/common_checker_sparse_flash_mla.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`CommonChecker` |
| [mask_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/mask_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [mask_checker_sparse_flash_mla.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/mask_checker_sparse_flash_mla.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`MaskChecker` |
| [metadata_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/metadata_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [metadata_checker_sparse_flash_mla.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/metadata_checker_sparse_flash_mla.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`MetadataChecker` |
| [paged_attention_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/paged_attention_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [paged_attention_checker_sparse_flash_mla.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/paged_attention_checker_sparse_flash_mla.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`PagedAttentionChecker` |
| [seq_len_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/seq_len_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [seq_len_checker_sparse_flash_mla.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/seq_len_checker_sparse_flash_mla.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`SeqLenChecker` |
| [sinks_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/sinks_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sinks_checker_sparse_flash_mla.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/sinks_checker_sparse_flash_mla.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`SinksChecker` |
| [softmax_lse_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/softmax_lse_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [softmax_lse_checker_sparse_flash_mla.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/softmax_lse_checker_sparse_flash_mla.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`SoftmaxLseChecker` |
| [sparse_compression_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/sparse_compression_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sparse_compression_checker.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/sparse_compression_checker.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`SparseCompressionChecker` |
| [sparse_flash_mla_checker.cpp](../../../csrc/attention/sparse_flash_mla/op_host/checkers/sparse_flash_mla_checker.cpp) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 函数：`BuildContext` |
| [sparse_flash_mla_checker.h](../../../csrc/attention/sparse_flash_mla/op_host/checkers/sparse_flash_mla_checker.h) | 稀疏 MLA 注意力 的主机侧辅助接口或实现。 | 类型：`SparseFlashMlaChecker` |

## `csrc/attention/sparse_flash_mla/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 子目录 | 职责 |
| --- | --- |
| [arch22/](../../../csrc/attention/sparse_flash_mla/op_kernel/arch22) | arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [arch35/](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35) | arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_flash_mla.cpp](../../../csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla.cpp) | 稀疏 MLA 注意力 的设备执行源码；文件主题为 MLA 潜在注意力。 | 函数：`sparse_flash_mla` |
| [sparse_flash_mla_common.h](../../../csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla_common.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 公共逻辑、MLA 潜在注意力。 | 类型：`SMLATemplateMode`, `SMLA_LAYOUT`, `SMLAType` |
| [sparse_flash_mla_kernel_metadata.h](../../../csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla_kernel_metadata.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 元数据、MLA 潜在注意力。 | 类型：`SmlaMetadata` |
| [sparse_flash_mla_template_tiling_key.h](../../../csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla_template_tiling_key.h) | 稀疏 MLA 注意力 的kernel 分派使用的 tiling key 定义；文件主题为 MLA 潜在注意力。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/sparse_flash_mla/op_kernel/arch22/`

arch22 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_flash_mla_arch22_metadata.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_arch22_metadata.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 元数据、MLA 潜在注意力。 | 类型：`SasMetadata` |
| [sparse_flash_mla_common_arch22.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_common_arch22.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 公共逻辑、MLA 潜在注意力。 | 类型：`SMLA_RUN_MODE`, `SMLA_LAYOUT`, `SMLAType` |
| [sparse_flash_mla_csa_block_cube.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_csa_block_cube.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、MLA 潜在注意力。 | 类型：`SMLACubeBlock` |
| [sparse_flash_mla_csa_block_vector.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_csa_block_vector.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、MLA 潜在注意力。 | 类型：`SMLAVectorBlock`, `ConstInfo` |
| [sparse_flash_mla_csa_kernel.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_csa_kernel.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`TempLoopInfo`, `SparseFlashMlaCsa` |
| [sparse_flash_mla_swa_block_cube.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_swa_block_cube.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、MLA 潜在注意力。 | 类型：`SWACubeBlock` |
| [sparse_flash_mla_swa_block_vector.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_swa_block_vector.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、MLA 潜在注意力。 | 类型：`SWAVectorBlock`, `ConstInfo` |
| [sparse_flash_mla_swa_kernel.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_swa_kernel.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`SwaTempLoopInfo`, `SparseFlashMlaSwa` |

## `csrc/attention/sparse_flash_mla/op_kernel/arch35/`

arch35 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/common) | 公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [sparse_flash_mla_common_arch35.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_common_arch35.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 公共逻辑、MLA 潜在注意力。 | 函数：`Align2Func`, `Align8Func` |
| [sparse_flash_mla_csa_block_cube_arch35.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_csa_block_cube_arch35.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、MLA 潜在注意力。 | 类型：`CubeCoordInfo`, `CSABlockCube`, `CSABlockCubeDummy` |
| [sparse_flash_mla_csa_block_vector_arch35.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_csa_block_vector_arch35.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、MLA 潜在注意力。 | 类型：`PhyAddrValidInfo`, `CSABlockVec`, `CSABlockVecDummy` |
| [sparse_flash_mla_csa_kernel_arch35.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_csa_kernel_arch35.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`SparseFlashMlaCsaKernel` |
| [sparse_flash_mla_kvcache.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_kvcache.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 缓存、MLA 潜在注意力、KV 缓存。 | 函数：`GetSingleCoreParam`, `ComputeParamBatch` |
| [sparse_flash_mla_swa_kernel_arch35.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_swa_kernel_arch35.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 MLA 潜在注意力。 | 类型：`SparseFlashMlaSwaKernel` |
| [util_regbase.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/util_regbase.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件。 | 类型：`VselrIndexEnum`, `RunParamStr`, `RunInfo` |

## `csrc/attention/sparse_flash_mla/op_kernel/arch35/common/`

公共逻辑 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [buffers_policy_3buff_sfa.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/common/buffers_policy_3buff_sfa.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 稀疏注意力。 | 类型：`BuffersPolicy3buffSFA` |
| [flash_decode.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/common/flash_decode.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 逐步生成。 | 类型：`FdRunInfo`, `FdBuffers`, `S2SplitFdStagingLayout` |
| [static_buffer.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/common/static_buffer.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件。 | 类型：`StaticBuffer`, `RingBuffer` |
| [static_matmul.h](../../../csrc/attention/sparse_flash_mla/op_kernel/arch35/common/static_matmul.h) | 稀疏 MLA 注意力 的类型/模板与设备实现头文件；文件主题为 矩阵乘。 | 函数：`MatmulKStatic`, `MatmulNStatic` |

## `csrc/attention/sparse_flash_mla_metadata/`

稀疏 MLA 元数据 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../csrc/attention/sparse_flash_mla_metadata/docs) | docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [examples/](../../../csrc/attention/sparse_flash_mla_metadata/examples) | examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/sparse_flash_mla_metadata/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel_aicpu/](../../../csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu) | op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_flash_mla_metadata/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [README.md](../../../csrc/attention/sparse_flash_mla_metadata/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：SparseFlashMlaMetadata |

## `csrc/attention/sparse_flash_mla_metadata/docs/`

docs 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnnSparseFlashMlaMetadata.md](../../../csrc/attention/sparse_flash_mla_metadata/docs/aclnnSparseFlashMlaMetadata.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：aclnnSparseFlashMlaMetadata。 | 标题：aclnnSparseFlashMlaMetadata |

## `csrc/attention/sparse_flash_mla_metadata/examples/`

examples 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_aclnn_sparse_flash_mla_metadata.cpp](../../../csrc/attention/sparse_flash_mla_metadata/examples/test_aclnn_sparse_flash_mla_metadata.cpp) | 元数据、MLA 潜在注意力 的公共接口/实现，供所在算子工程复用。 | 类型：`SmlaMetadata`, `ScopeGuard`, `Tensor` |
| [test_torch_sparse_flash_mla_metadata.py](../../../csrc/attention/sparse_flash_mla_metadata/examples/test_torch_sparse_flash_mla_metadata.py) | 算子工程 元数据、MLA 潜在注意力 的生成、配置、编译或打包辅助；接口见右栏。 | 导入：`torch` |

## `csrc/attention/sparse_flash_mla_metadata/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/sparse_flash_mla_metadata/op_host/op_api) | 算子对外调用接口及适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_flash_mla_metadata/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [sparse_flash_mla_metadata_check.h](../../../csrc/attention/sparse_flash_mla_metadata/op_host/sparse_flash_mla_metadata_check.h) | 稀疏 MLA 元数据 的主机侧辅助接口或实现。 | 类型：`SparseModeSmla`, `TopkLengthAxisSmla` |

## `csrc/attention/sparse_flash_mla_metadata/op_host/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_sparse_flash_mla_metadata.cpp](../../../csrc/attention/sparse_flash_mla_metadata/op_host/op_api/aclnn_sparse_flash_mla_metadata.cpp) | 稀疏 MLA 元数据 的主机侧辅助接口或实现。 | 函数：`aclnnSparseFlashMlaMetadataGetWorkspaceSize` |
| [aclnn_sparse_flash_mla_metadata.h](../../../csrc/attention/sparse_flash_mla_metadata/op_host/op_api/aclnn_sparse_flash_mla_metadata.h) | 稀疏 MLA 元数据 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sparse_flash_mla_metadata.cpp](../../../csrc/attention/sparse_flash_mla_metadata/op_host/op_api/sparse_flash_mla_metadata.cpp) | 稀疏 MLA 元数据 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sparse_flash_mla_metadata.h](../../../csrc/attention/sparse_flash_mla_metadata/op_host/op_api/sparse_flash_mla_metadata.h) | 稀疏 MLA 元数据 的主机侧辅助接口或实现。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/`

op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [sparse_flash_mla_metadata_aicpu.cpp](../../../csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp) | 元数据、MLA 潜在注意力 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [sparse_flash_mla_metadata_aicpu.h](../../../csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.h) | 元数据、MLA 潜在注意力 的公共接口/实现，供所在算子工程复用。 | 类型：`SparseMode`, `ValidSocVersion`, `FlashDecodeResult` |
| [sparse_flash_mla_metadata_aicpu.json](../../../csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.json) | 所在模块的结构化配置或参数模板。 | 顶层项：`SparseFlashMlaMetadata` |

## `csrc/attention/store_kv_block/`

KV block 写入 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_host/](../../../csrc/attention/store_kv_block/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/attention/store_kv_block/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/store_kv_block/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [store_kv_block_torch_adpt.h](../../../csrc/attention/store_kv_block/store_kv_block_torch_adpt.h) | 缓存页/分块、存储、KV 缓存 的公共接口/实现，供所在算子工程复用。 | 函数：`store_kv_block` |

## `csrc/attention/store_kv_block/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/store_kv_block/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [store_kv_block_def.cpp](../../../csrc/attention/store_kv_block/op_host/store_kv_block_def.cpp) | KV block 写入 的主机算子定义、支持属性和注册。 | 注册：`StoreKVBlock` |
| [store_kv_block_infershape.cpp](../../../csrc/attention/store_kv_block/op_host/store_kv_block_infershape.cpp) | KV block 写入 的主机侧辅助接口或实现。 | 函数：`InferShape4StoreKVBlock`, `InferDataType4StoreKVBlock` |
| [store_kv_block_tiling.cpp](../../../csrc/attention/store_kv_block/op_host/store_kv_block_tiling.cpp) | KV block 写入 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`StoreKVBlockParams` |
| [store_kv_block_tiling.h](../../../csrc/attention/store_kv_block/op_host/store_kv_block_tiling.h) | KV block 写入 的主机侧 tiling 参数/分块策略声明。 | 类型：`StoreKVBlockCompileInfo` |

## `csrc/attention/store_kv_block/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [store_kv_block.cpp](../../../csrc/attention/store_kv_block/op_kernel/store_kv_block.cpp) | KV block 写入 的设备执行源码；文件主题为 缓存页/分块、存储、KV 缓存。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [store_kv_block.h](../../../csrc/attention/store_kv_block/op_kernel/store_kv_block.h) | KV block 写入 的类型/模板与设备实现头文件；文件主题为 缓存页/分块、存储、KV 缓存。 | 类型：`StoreKVBlockTilingData`, `StoreKVBlockBase` |

## `csrc/attention/store_kv_block_metadata/`

KV block 写入元数据 算子工程；主机/设备职责以其子目录区分。

| 子目录 | 职责 |
| --- | --- |
| [op_api/](../../../csrc/attention/store_kv_block_metadata/op_api) | 算子对外调用接口及适配。 |
| [op_graph/](../../../csrc/attention/store_kv_block_metadata/op_graph) | 设备图 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |
| [op_host/](../../../csrc/attention/store_kv_block_metadata/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel_aicpu/](../../../csrc/attention/store_kv_block_metadata/op_kernel_aicpu) | op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/attention/store_kv_block_metadata/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [store_kv_block_metadata_torch_adpt.cpp](../../../csrc/attention/store_kv_block_metadata/store_kv_block_metadata_torch_adpt.cpp) | 元数据、缓存页/分块、存储 的公共接口/实现，供所在算子工程复用。 | 函数：`store_kv_block_metadata` |

## `csrc/attention/store_kv_block_metadata/op_api/`

算子对外调用接口及适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_store_kv_block_metadata.cpp](../../../csrc/attention/store_kv_block_metadata/op_api/aclnn_store_kv_block_metadata.cpp) | 元数据、缓存页/分块、存储 的公共接口/实现，供所在算子工程复用。 | 函数：`aclnnStoreKvBlockMetadataGetWorkspaceSize` |
| [aclnn_store_kv_block_metadata.h](../../../csrc/attention/store_kv_block_metadata/op_api/aclnn_store_kv_block_metadata.h) | 元数据、缓存页/分块、存储 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [l0_store_kv_block_metadata.cpp](../../../csrc/attention/store_kv_block_metadata/op_api/l0_store_kv_block_metadata.cpp) | 元数据、缓存页/分块、存储 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [l0_store_kv_block_metadata.h](../../../csrc/attention/store_kv_block_metadata/op_api/l0_store_kv_block_metadata.h) | 元数据、缓存页/分块、存储 的公共接口/实现，供所在算子工程复用。 | 函数：`StoreKvBlockMetadata` |

## `csrc/attention/store_kv_block_metadata/op_graph/`

设备图 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [store_kv_block_metadata_proto.h](../../../csrc/attention/store_kv_block_metadata/op_graph/store_kv_block_metadata_proto.h) | 元数据、缓存页/分块、存储 的公共接口/实现，供所在算子工程复用。 | 注册：`StoreKvBlockMetadata` |

## `csrc/attention/store_kv_block_metadata/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [store_kv_block_metadata_infershape.cpp](../../../csrc/attention/store_kv_block_metadata/op_host/store_kv_block_metadata_infershape.cpp) | KV block 写入元数据 的主机侧辅助接口或实现。 | 函数：`InferShape4StoreKvBlockMetadata`, `InferDataType4StoreKvBlockMetadata` |

## `csrc/attention/store_kv_block_metadata/op_kernel_aicpu/`

op_kernel_aicpu 子模块；所属范围：注意力、索引、压缩、缓存写入与相关状态算子。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [store_kv_block_metadata_aicpu.cpp](../../../csrc/attention/store_kv_block_metadata/op_kernel_aicpu/store_kv_block_metadata_aicpu.cpp) | 元数据、缓存页/分块、存储 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [store_kv_block_metadata_aicpu.h](../../../csrc/attention/store_kv_block_metadata/op_kernel_aicpu/store_kv_block_metadata_aicpu.h) | 元数据、缓存页/分块、存储 的公共接口/实现，供所在算子工程复用。 | 类型：`StoreKvBlockMetadataCpuKernel`, `ParamId` |
| [store_kv_block_metadata_aicpu.json](../../../csrc/attention/store_kv_block_metadata/op_kernel_aicpu/store_kv_block_metadata_aicpu.json) | 所在模块的结构化配置或参数模板。 | 顶层项：`StoreKvBlockMetadata` |
