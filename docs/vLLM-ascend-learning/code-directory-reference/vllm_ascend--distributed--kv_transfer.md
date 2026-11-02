# vllm_ascend/distributed/kv_transfer：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **69 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/distributed/kv_transfer/`

KV 连接器注册、PD 传输、共享缓存池和稀疏缓存迁移。

| 子目录 | 职责 |
| --- | --- |
| [kv_p2p/](../../../vllm_ascend/distributed/kv_transfer/kv_p2p) | Prefill 与 Decode 实例间点对点 KV 交接。 |
| [kv_pool/](../../../vllm_ascend/distributed/kv_transfer/kv_pool) | KV 共享存储、池协调与 Offload 接入。 |
| [sparse_kv_offload/](../../../vllm_ascend/distributed/kv_transfer/sparse_kv_offload) | 按稀疏注意力访问需要组织设备与主机中的历史 KV。 |
| [utils/](../../../vllm_ascend/distributed/kv_transfer/utils) | 辅助函数 子模块；所属范围：KV 连接器注册、PD 传输、共享缓存池和稀疏缓存迁移。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`register_connector` |
| [ascend_multi_connector.py](../../../vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py) | 组合多个 KV connector，协调各自能力、完成状态和 DSpark 前缀存储提供者。 | 定义：`AscendMultiConnector` |

## `vllm_ascend/distributed/kv_transfer/kv_p2p/`

Prefill 与 Decode 实例间点对点 KV 交接。

| 子目录 | 职责 |
| --- | --- |
| [mooncake/](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake) | Mooncake 传输/存储 子模块；所属范围：Prefill 与 Decode 实例间点对点 KV 交接。 |
| [sfa_pd_rd2h/](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h) | 稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [mooncake_connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py) | Mooncake 传输/存储 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 定义：`RemotePortInfo`, `MooncakeAgentMetadata`, `zmq_ctx`, `group_concurrent_contiguous`, `split_if_not_byte_contiguous` |
| [mooncake_d2rh_connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_d2rh_connector.py) | Mooncake 传输/存储 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 定义：`HostListeningThread`, `D2RHCPUCacheManager`, `build_layer_name_to_cache_slots`, `resolve_group_cache_slot_pairs`, `ensure_zmq_recv` |
| [mooncake_hybrid_connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py) | Mooncake 传输/存储 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 定义：`RemotePortInfo`, `MooncakeAgentMetadata`, `zmq_ctx`, `group_concurrent_contiguous`, `string_to_int64_hash` |
| [mooncake_layerwise_connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py) | Mooncake 传输/存储 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 定义：`LayerMetadata`, `MooncakeAgentMetadata`, `zmq_ctx`, `group_concurrent_contiguous`, `string_to_int64_hash` |

## `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/`

Mooncake 传输/存储 子模块；所属范围：Prefill 与 Decode 实例间点对点 KV 交接。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：Mooncake KV-cache transfer connector (disaggregated prefill / decode). |
| [base_scheduler.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_scheduler.py) | 调度 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 注释：Common scheduler-side logic for Mooncake KV transfer connectors.；定义：`MooncakeBaseConnectorScheduler` |
| [base_worker.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py) | 设备执行 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 注释：Common worker-side logic for Mooncake KV transfer connectors.；定义：`MooncakeBaseConnectorWorker` |
| [connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py) | Mooncake 传输/存储 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 注释：Mooncake connector facades.；定义：`MooncakeBaseConnector`, `MooncakePullConnector` |
| [metadata.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/metadata.py) | 元数据 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 注释：Metadata types for Mooncake KV transfer connectors.；定义：`MooncakeTransferMetadata`, `MooncakeTPTransferMetadata` |
| [pull_scheduler.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_scheduler.py) | 调度 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 注释：Scheduler-side logic for Mooncake pull transfers.；定义：`MooncakeSchedulerSendingThread`, `MooncakeSchedulerRecvingThread` |
| [pull_worker.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py) | 设备执行 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 注释：Worker-side implementation entry point for Mooncake pull transfers.；定义：`MooncakePullRecvingThread`, `MooncakePullConnectorWorker` |
| [stats.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/stats.py) | Mooncake 传输/存储 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 注释：Transfer statistics and Prometheus metrics for Mooncake.；定义：`MooncakeKVConnectorStats`, `MooncakePromMetrics` |
| [utils.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：Prefill 与 Decode 实例间点对点 KV 交接。 | 注释：Shared utilities for Mooncake KV transfer connectors.；定义：`SizedDict`, `zmq_ctx`, `as_kv_cache_tensors`, `collect_configured_register_regions` |

## `vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/`

稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`get_prebound_copy_sfa_slots` |
| [connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/connector.py) | 稀疏注意力 的接口/实现或辅助；所属职责：稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。 | 注释：PD-disaggregated SFA Remote D2H KV-transfer connector.；定义：`SfaRemoteD2HConnector` |
| [dspark_context.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_context.py) | DSpark 草稿 的接口/实现或辅助；所属职责：稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。 | 注释：DSpark P-to-D draft-KV readiness and cache-group helpers.；定义：`DSparkContextDescriptor`, `DSparkContextChunk`, `uses_sfa_dspark_kv_transfer`, `get_pd_dspark_aux_layer_ids`, `resident_mla_context_group_ids` |
| [dspark_kv.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_kv.py) | DSpark 草稿、KV 缓存 的接口/实现或辅助；所属职责：稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。 | 注释：Direct transfer of P-computed DSpark MLA KV into D's resident pages.；定义：`DraftKVCacheMetadata`, `get_resident_dspark_layer_names`, `apply_dspark_resident_kv_specs`, `build_draft_kv_metadata` |
| [protocol.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/protocol.py) | DSpark 草稿、草稿模型 的接口/实现或辅助；所属职责：稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。 | 注释：Wire protocol helpers for SFA PD RD2H.；定义：`DSparkDraftKVStatus`, `LayerMetadata`, `infer_sfa_component_group_ids`, `get_external_request_id` |
| [read_thread.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/read_thread.py) | 状态 的接口/实现或辅助；所属职责：稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。 | 注释：D-side memfabric pull read thread for SFA PD RD2H.；定义：`ConsumerReadState`, `MembPullReadThread` |
| [scheduler.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/scheduler.py) | 调度 的接口/实现或辅助；所属职责：稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。 | 注释：Scheduler side of the decode-offload SFA Remote D2H connector.；定义：`_SendReqInfo`, `SFAPDRD2HProducerScheduler` |
| [send_thread.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/send_thread.py) | 状态 的接口/实现或辅助；所属职责：稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。 | 注释：P-side memfabric pull sending thread for SFA PD RD2H.；定义：`ProducerSendState`, `DSparkDraftKVSendTask` |
| [worker.py](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/worker.py) | 设备执行 的接口/实现或辅助；所属职责：稀疏注意力 PD 路径的缓存传输、请求状态和草稿上下文处理。 | 注释：Worker side of the decode-offload SFA Remote D2H connector.；定义：`SFAPDRD2HConsumerWorker`, `SFAPDRD2HProducerWorker` |

## `vllm_ascend/distributed/kv_transfer/kv_pool/`

KV 共享存储、池协调与 Offload 接入。

| 子目录 | 职责 |
| --- | --- |
| [ascend_store/](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store) | Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 |
| [kv_offload/](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload) | 缓存卸载、KV 缓存 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。 |
| [ucm_connector/](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector) | ucm_connector 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`

Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。

| 子目录 | 职责 |
| --- | --- |
| [backend/](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend) | backend 子模块；所属范围：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [ascend_store_connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py) | 存储 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`AscendStoreKVEvents`, `AscendStoreConnector` |
| [attention_fence.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py) | 注意力 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`AttentionComputeStartGate`, `reset_attention_compute_start_gate`, `get_attention_compute_start_gate`, `record_attention_compute_start` |
| [coordinator.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py) | 缓存、缓存页/分块、池管理 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`ExternalCachedBlockPool`, `AscendStoreCoordinator` |
| [dspark_prefix_cache.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/dspark_prefix_cache.py) | DSpark 草稿、缓存 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 注释：Persistent DSpark prefix pages in the existing AscendStore backend.；定义：`DSparkPrefixKeys`, `_Page` |
| [kv_transfer.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py) | 传输、KV 缓存 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`_LayerRevokeTask`, `LayerBatchBuilder`, `record_failed_blocks` |
| [layerwise_cache_layout.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py) | 内存布局、缓存 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`LayerwiseCacheLayout`, `NamedKVCacheSpec`, `get_layerwise_physical_layer_index`, `get_layerwise_reuse_config`, `build_layerwise_cache_layout` |
| [metadata.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py) | 元数据 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`HybridCacheC128Config`, `TransferChunk`, `aggregate_c128_page_chunks`, `is_kv_save_role`, `infer_dcp_mismatch_info` |
| [metrics.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py) | 指标 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`AscendStoreKVConnectorStats`, `AscendStorePromMetrics` |
| [pool_scheduler.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py) | 调度、池管理 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`KVPoolScheduler`, `LookupKeyClient`, `get_zmq_rpc_path_lookup` |
| [pool_worker.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py) | 设备执行、池管理 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`KVPoolWorker` |

## `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/`

backend 子模块；所属范围：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`get_layerwise_protocol`, `get_layerwise_data_plane`, `validate_layerwise_topology` |
| [base.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py) | 批次 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`BatchResultShapeError`, `Backend`, `parse_qos_from_extra_config`, `require_aligned_batch_results`, `get_scheduler_device_id` |
| [layerwise_keys.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/layerwise_keys.py) | 统一分层缓存写入和查询使用的传输 key 构造规则。 | 注释：Bound wire-key construction shared by writers and scheduler lookups.；定义：`LayerwiseKeyBuilder` |
| [memcache_backend.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py) | 缓存 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`MmcDirect`, `MemcacheBackend`, `bind_layerwise_keys`, `extract_layout_config`, `make_full_key` |
| [mooncake_backend.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py) | Mooncake 传输/存储 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`MooncakeBackend`, `MooncakeStoreConfig` |
| [mooncake_layerwise.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_layerwise.py) | Mooncake 传输/存储 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 注释：Group-aware Mooncake sessions using the shared hybrid reachability masks.；定义：`bind_layerwise_keys`, `extract_layout_config`, `make_block_key` |
| [session_tracker.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/session_tracker.py) | 跟踪 layerwise KV 存储会话的生命周期和状态。 | 定义：`LayerwiseSessionTracker` |
| [yuanrong_backend.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py) | 配置 的接口/实现或辅助；所属职责：Ascend KV Store 的调度侧/执行侧协作、元数据和后端适配。 | 定义：`YuanrongConfig`, `YuanrongBackend` |

## `vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/`

缓存卸载、KV 缓存 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。

| 子目录 | 职责 |
| --- | --- |
| [native/](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native) | native 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。 |
| [preempt_offload/](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/preempt_offload) | 缓存卸载 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。 |
| [simple/](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple) | simple 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：Ascend KV cache offloading implementations. |

## `vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/`

native 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：NPU adaptation of vLLM's native KV-cache offloading. |
| [cpu_npu.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py) | 传输 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：Ascend worker for vLLM's native CPU KV-cache offloading.；定义：`Transfer`, `SingleDirectionNPUOffloadingHandler` |
| [npu.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/npu.py) | 设备执行 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：Ascend specs for vLLM's native KV-cache offloading.；定义：`_NPUWorkerMixin`, `NPUOffloadingSpec` |
| [offloading_connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py) | 缓存卸载 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：Ascend layout adaptation for vLLM's native offloading connector.；定义：`AscendOffloadingConnectorWorker`, `AscendOffloadingConnector` |

## `vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/preempt_offload/`

缓存卸载 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/preempt_offload/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [manager.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/preempt_offload/manager.py) | 管理器 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：Scheduler-side manager for preempt offloading.；定义：`TransferMeta`, `PreemptedRequestState` |
| [metadata.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/preempt_offload/metadata.py) | 元数据 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：Metadata for PreemptOffloadConnector.；定义：`MambaConvLoadMeta`, `PreemptOffloadMetadata` |
| [preempt_offload_connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/preempt_offload/preempt_offload_connector.py) | 缓存卸载 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：PreemptOffloadConnector: minimal CPU KV cache offloading.；定义：`PreemptOffloadConnectorV1` |
| [worker.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/preempt_offload/worker.py) | 设备执行 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：Worker-side handler for Ascend PreemptOffloadConnector.；定义：`MambaConvCacheBinding`, `PreemptOffloadWorker` |

## `vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/`

simple 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：NPU adaptation of vLLM's simple CPU KV-cache offloading. |
| [copy_backend.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/copy_backend.py) | 复制 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：DMA copy backend for NPU&lt;-&gt;CPU block transfers.；定义：`NPUDmaCopyBackend` |
| [npu_mem_ops.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/npu_mem_ops.py) | 内存复制、批次 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：Low-level NPU memory helpers: batched DMA transfers.；定义：`BatchMemcpyParams`, `build_params`, `copy_blocks` |
| [simple_cpu_offload_connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/simple_cpu_offload_connector.py) | 缓存卸载 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：Ascend NPU adaptation of vLLM's simple CPU offload connector.；定义：`AscendSimpleCPUOffloadConnector` |
| [worker.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/worker.py) | 设备执行 的接口/实现或辅助；所属职责：KV 共享存储、池协调与 Offload 接入。 | 注释：Worker-side handler for the Ascend ``SimpleCPUOffloadConnector``.；定义：`SimpleCPUOffloadNPUWorker` |

## `vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector/`

ucm_connector 子模块；所属范围：KV 共享存储、池协调与 Offload 接入。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`.connector` |
| [connector.py](../../../vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector/connector.py) | 将 UCM 缓存服务接入 vLLM KV connector 协议。 | 定义：`UCMConnectorV1` |

## `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/`

按稀疏注意力访问需要组织设备与主机中的历史 KV。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/sparse_kv_offload/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [copy_sfa_topk_slots.py](../../../vllm_ascend/distributed/kv_transfer/sparse_kv_offload/copy_sfa_topk_slots.py) | 槽位、TopK 选择、复制 的接口/实现或辅助；所属职责：按稀疏注意力访问需要组织设备与主机中的历史 KV。 | 注释：Request-owned fused_copy_sfa top-k row slots shared by the PD connector and runner.；定义：`_CopySfaRequestState`, `CopySfaRequestStates`, `copy_sfa_pool_capacity`, `copy_sfa_tail_geometry`, `copy_sfa_prefill_dest_geometry` |
| [sparse_kv_offload_manager.py](../../../vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py) | 缓存卸载、管理器、KV 缓存 的接口/实现或辅助；所属职责：按稀疏注意力访问需要组织设备与主机中的历史 KV。 | 定义：`SparseKVOffloadMemoryBudget`, `SparseKVOffloadManager`, `get_subscribed_compute_streams`, `allocate_kv_offload_topk_buffer_pair`, `allocate_kv_offload_topk_profile_buffers` |

## `vllm_ascend/distributed/kv_transfer/utils/`

辅助函数 子模块；所属范围：KV 连接器注册、PD 传输、共享缓存池和稀疏缓存迁移。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/kv_transfer/utils/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [memfabric_transfer_engine.py](../../../vllm_ascend/distributed/kv_transfer/utils/memfabric_transfer_engine.py) | 传输 的接口/实现或辅助；所属职责：KV 连接器注册、PD 传输、共享缓存池和稀疏缓存迁移。 | 注释：Process-wide MemFabric transfer-engine initialization.；定义：`MemfabricBackend`, `GlobalMemfabricTE` |
| [mooncake_transfer_engine.py](../../../vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py) | Mooncake 传输/存储、传输 的接口/实现或辅助；所属职责：KV 连接器注册、PD 传输、共享缓存池和稀疏缓存迁移。 | 定义：`GlobalTE`, `validate_mooncake_runtime_installation`, `get_mooncake_runtime_identity` |
| [utils.py](../../../vllm_ascend/distributed/kv_transfer/utils/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：KV 连接器注册、PD 传输、共享缓存池和稀疏缓存迁移。 | 定义：`parallel_info`, `RegisterRange`, `inject_qos`, `kv_alltoall_and_rearrange`, `alltoall_and_rearrange` |
