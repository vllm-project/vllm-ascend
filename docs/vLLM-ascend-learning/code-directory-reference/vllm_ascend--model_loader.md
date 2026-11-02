# vllm_ascend/model_loader：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **23 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/model_loader/`

可选加载器与权重载入扩展。

| 子目录 | 职责 |
| --- | --- |
| [netloader/](../../../vllm_ascend/model_loader/netloader) | 网络模型权重加载、传输和相应资源管理。 |
| [rfork/](../../../vllm_ascend/model_loader/rfork) | RFork 的 manifest、会话、seed、传输与张量布局协议。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/model_loader/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `vllm_ascend/model_loader/netloader/`

网络模型权重加载、传输和相应资源管理。

| 子目录 | 职责 |
| --- | --- |
| [executor/](../../../vllm_ascend/model_loader/netloader/executor) | executor 子模块；所属范围：网络模型权重加载、传输和相应资源管理。 |
| [interaction/](../../../vllm_ascend/model_loader/netloader/interaction) | interaction 子模块；所属范围：网络模型权重加载、传输和相应资源管理。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/model_loader/netloader/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`register_netloader` |
| [load.py](../../../vllm_ascend/model_loader/netloader/load.py) | 加载 的接口/实现或辅助；所属职责：网络模型权重加载、传输和相应资源管理。 | 定义：`elastic_load` |
| [netloader.py](../../../vllm_ascend/model_loader/netloader/netloader.py) | 网络权重加载 的接口/实现或辅助；所属职责：网络模型权重加载、传输和相应资源管理。 | 定义：`_FallbackCleanupContext`, `_FallbackReclaimResult`, `pre_transfer_weight_processing` |
| [utils.py](../../../vllm_ascend/model_loader/netloader/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：网络模型权重加载、传输和相应资源管理。 | 定义：`disk_fallback_loader_extra_config`, `find_free_port`, `is_valid_path_prefix` |

## `vllm_ascend/model_loader/netloader/executor/`

executor 子模块；所属范围：网络模型权重加载、传输和相应资源管理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/model_loader/netloader/executor/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [elastic_load.py](../../../vllm_ascend/model_loader/netloader/executor/elastic_load.py) | 加载 的接口/实现或辅助；所属职责：网络模型权重加载、传输和相应资源管理。 | 定义：`P2PLoad`, `P2PSend`, `register_processed_layout_transfer_items`, `cache_processed_layout_transfer_manifest`, `get_cached_processed_layout_transfer_items` |
| [netloader_pg.py](../../../vllm_ascend/model_loader/netloader/executor/netloader_pg.py) | 网络权重加载 的接口/实现或辅助；所属职责：网络模型权重加载、传输和相应资源管理。 | 定义：`stateless_init_process_group`, `destroy_stateless_process_group` |

## `vllm_ascend/model_loader/netloader/interaction/`

interaction 子模块；所属范围：网络模型权重加载、传输和相应资源管理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/model_loader/netloader/interaction/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [elastic.py](../../../vllm_ascend/model_loader/netloader/interaction/elastic.py) | 弹性网络加载客户端/服务端，交换 JSON 元数据和已处理权重布局信息。 | 定义：`ElasticClient`, `ElasticServer` |

## `vllm_ascend/model_loader/rfork/`

RFork 的 manifest、会话、seed、传输与张量布局协议。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/model_loader/rfork/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`register_rforkloader` |
| [config.py](../../../vllm_ascend/model_loader/rfork/config.py) | 配置 的接口/实现或辅助；所属职责：RFork 的 manifest、会话、seed、传输与张量布局协议。 | 定义：`RForkConfig` |
| [identity.py](../../../vllm_ascend/model_loader/rfork/identity.py) | 生成模型兼容性指纹和分片 seed key，防止错误复用不兼容权重。 | 注释：Build stable model compatibility fingerprints and shard seed keys.；定义：`build_compatibility_fingerprint`, `build_seed_key` |
| [manifest.py](../../../vllm_ascend/model_loader/rfork/manifest.py) | 解析校验权重清单、dtype 与布局元数据，本模块不修改实际张量存储。 | 注释：Parse and validate weight metadata without changing tensor storage or layout.；定义：`normalize_dtype_name`, `read_npu_format`, `is_positive_int` |
| [planner_client.py](../../../vllm_ascend/model_loader/rfork/planner_client.py) | RFork 权重加载 的接口/实现或辅助；所属职责：RFork 的 manifest、会话、seed、传输与张量布局协议。 | 定义：`RForkPlannerClient`, `lease_log_id` |
| [rfork_loader.py](../../../vllm_ascend/model_loader/rfork/rfork_loader.py) | 加载器、RFork 权重加载 的接口/实现或辅助；所属职责：RFork 的 manifest、会话、seed、传输与张量布局协议。 | 定义：`_RForkSeedUnavailable`, `_RForkProcessGlobalModelState` |
| [safety.py](../../../vllm_ascend/model_loader/rfork/safety.py) | 权重 的接口/实现或辅助；所属职责：RFork 的 manifest、会话、seed、传输与张量布局协议。 | 注释：RFork loader guards for configurations with mutable model storage.；定义：`mutable_weights_bypass_reason` |
| [seed_client.py](../../../vllm_ascend/model_loader/rfork/seed_client.py) | 构造 seed 服务地址并获取权重传输信息。 | 定义：`build_seed_url`, `fetch_seed_transfer_info` |
| [seed_server.py](../../../vllm_ascend/model_loader/rfork/seed_server.py) | 启动 RFork seed/API 服务并管理服务启动状态。 | 定义：`_ServerStartup`, `_ServerStartupState`, `start_fastapi_server`, `start_rfork_server` |
| [session.py](../../../vllm_ascend/model_loader/rfork/session.py) | RFork 权重加载 的接口/实现或辅助；所属职责：RFork 的 manifest、会话、seed、传输与张量布局协议。 | 定义：`RForkSession` |
| [tensor_layout.py](../../../vllm_ascend/model_loader/rfork/tensor_layout.py) | 张量、内存布局 的接口/实现或辅助；所属职责：RFork 的 manifest、会话、seed、传输与张量布局协议。 | 注释：Collect live model tensors and adapt their layout for RFork transfer.；定义：`build_structural_digest`, `log_tensor_layout_summary`, `reshape_tensor_to_seed_shape` |
| [transfer_backend.py](../../../vllm_ascend/model_loader/rfork/transfer_backend.py) | 传输 的接口/实现或辅助；所属职责：RFork 的 manifest、会话、seed、传输与张量布局协议。 | 定义：`RForkTransferBackend`, `iter_transfer_chunks` |
| [types.py](../../../vllm_ascend/model_loader/rfork/types.py) | 状态、RFork 权重加载 的接口/实现或辅助；所属职责：RFork 的 manifest、会话、seed、传输与张量布局协议。 | 定义：`RForkLifecycleState`, `LeaseReleaseResult` |
