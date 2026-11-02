# csrc/utils：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **20 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/utils/`

设备/主机辅助函数、接口和工具实现。

| 子目录 | 职责 |
| --- | --- |
| [inc/](../../../csrc/utils/inc) | 公共头文件、类型和接口声明。 |
| [src/](../../../csrc/utils/src) | 父模块的具体实现源文件。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/utils/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | 构建项：`ops_utils_tiling_headers`, `ops_utils_proto_headers` |

## `csrc/utils/inc/`

公共头文件、类型和接口声明。

| 子目录 | 职责 |
| --- | --- |
| [error/](../../../csrc/utils/inc/error) | error 子模块；所属范围：设备/主机辅助函数、接口和工具实现。 |
| [kernel/](../../../csrc/utils/inc/kernel) | kernel 子模块；所属范围：设备/主机辅助函数、接口和工具实现。 |
| [log/](../../../csrc/utils/inc/log) | 日志 子模块；所属范围：设备/主机辅助函数、接口和工具实现。 |
| [tiling/](../../../csrc/utils/inc/tiling) | tiling 子模块；所属范围：设备/主机辅助函数、接口和工具实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aclnn_util.h](../../../csrc/utils/inc/aclnn_util.h) | aclnn_util 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [fallback.h](../../../csrc/utils/inc/fallback.h) | fallback 的公共接口/实现，供所在算子工程复用。 | 类型：`index_sequence`, `make_index_sequence_helper`, `aclOpExecutor` |
| [fallback_comm.h](../../../csrc/utils/inc/fallback_comm.h) | fallback_comm 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/utils/inc/error/`

error 子模块；所属范围：设备/主机辅助函数、接口和工具实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [ops_error.h](../../../csrc/utils/inc/error/ops_error.h) | ops_error 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/utils/inc/kernel/`

kernel 子模块；所属范围：设备/主机辅助函数、接口和工具实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [comm_args.h](../../../csrc/utils/inc/kernel/comm_args.h) | comm_args 的公共接口/实现，供所在算子工程复用。 | 类型：`CommArgs` |
| [data_copy.h](../../../csrc/utils/inc/kernel/data_copy.h) | 复制 的公共接口/实现，供所在算子工程复用。 | 函数：`SetAtomicOpType`, `CpUB2GM` |
| [dropmask.h](../../../csrc/utils/inc/kernel/dropmask.h) | dropmask 的公共接口/实现，供所在算子工程复用。 | 类型：`DropMaskInfo` |
| [moe_distribute_base.h](../../../csrc/utils/inc/kernel/moe_distribute_base.h) | 混合专家 的公共接口/实现，供所在算子工程复用。 | 类型：`HcclSignalInfo`, `ListCommon`, `HcclStreamInfo` |
| [pse.h](../../../csrc/utils/inc/kernel/pse.h) | pse 的公共接口/实现，供所在算子工程复用。 | 类型：`PseTypeEnum`, `PseInfo` |
| [sync_collectives.h](../../../csrc/utils/inc/kernel/sync_collectives.h) | sync_collectives 的公共接口/实现，供所在算子工程复用。 | 类型：`SyncCollectives` |
| [util.h](../../../csrc/utils/inc/kernel/util.h) | util 的公共接口/实现，供所在算子工程复用。 | 类型：`LayOutTypeEnum` |

## `csrc/utils/inc/log/`

日志 子模块；所属范围：设备/主机辅助函数、接口和工具实现。

| 子目录 | 职责 |
| --- | --- |
| [inner/](../../../csrc/utils/inc/log/inner) | inner 子模块；所属范围：设备/主机辅助函数、接口和工具实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [ops_log.h](../../../csrc/utils/inc/log/ops_log.h) | 日志 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/utils/inc/log/inner/`

inner 子模块；所属范围：设备/主机辅助函数、接口和工具实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [dfx_base.h](../../../csrc/utils/inc/log/inner/dfx_base.h) | dfx_base 的公共接口/实现，供所在算子工程复用。 | 类型：`LogBase` |

## `csrc/utils/inc/tiling/`

tiling 子模块；所属范围：设备/主机辅助函数、接口和工具实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [data_copy_transpose_tiling.h](../../../csrc/utils/inc/tiling/data_copy_transpose_tiling.h) | 转置、复制 的公共接口/实现，供所在算子工程复用。 | 函数：`GetDataCopyTransposeTiling` |
| [data_copy_transpose_tiling_def.h](../../../csrc/utils/inc/tiling/data_copy_transpose_tiling_def.h) | 转置、复制 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [tiling_base.h](../../../csrc/utils/inc/tiling/tiling_base.h) | tiling_base 的公共接口/实现，供所在算子工程复用。 | 类型：`AiCoreParams`, `FlashAttentionScoreGradCompileInfo`, `TilingBaseClass` |
| [tiling_templates_registry.h](../../../csrc/utils/inc/tiling/tiling_templates_registry.h) | 注册表 的公共接口/实现，供所在算子工程复用。 | 类型：`TilingCases`, `TilingRegistry`, `Register` |
| [tiling_type.h](../../../csrc/utils/inc/tiling/tiling_type.h) | tiling_type 的公共接口/实现，供所在算子工程复用。 | 类型：`AxisEnum`, `DtypeEnum`, `PerformanceOrientedEnum` |

## `csrc/utils/src/`

父模块的具体实现源文件。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fallback_comm.cpp](../../../csrc/utils/src/fallback_comm.cpp) | fallback_comm 的公共接口/实现，供所在算子工程复用。 | 函数：`ToAclDataType` |
