# vllm_ascend/core：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **10 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

先确认描述的是调度规则、cache spec 还是物理放置。逻辑页的 token 数、压缩后的物理行数和实际张量字节不是同一个量；要继续追 Runner 分配与模型缓存配置。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `vllm_ascend/core/`

调度器扩展、KV 缓存规格、缓存管理和 KVPP 物理放置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/core/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [batch_job_aware_scheduler.py](../../../vllm_ascend/core/batch_job_aware_scheduler.py) | 感知批量任务组织的调度扩展。 | 注释：Batch-Job-Aware Scheduler for vLLM v1；定义：`JobNameParser`, `_JobStat` |
| [dyntra_lb_scheduler.py](../../../vllm_ascend/core/dyntra_lb_scheduler.py) | Dyntra 负载均衡相关调度逻辑。 | 定义：`DyntraLBPolicyMixin`, `DyntraLBScheduler`, `get_dyntra_lb_block_size`, `get_dyntra_lb_request_block_num`, `diagnostics_enabled` |
| [kv_cache_interface.py](../../../vllm_ascend/core/kv_cache_interface.py) | Ascend cache spec 及压缩、物理页大小、循环状态和 prefix 能力接口。 | 定义：`AscendMLAAttentionSpec`, `AscendSFAIndexerCacheSpec`, `get_kv_cache_compression_ratio`, `get_storage_block_size`, `is_circular_kv_cache_spec` |
| [kv_cache_placement.py](../../../vllm_ascend/core/kv_cache_placement.py) | KVPP 缓存 owner、层 bundle、物理尺寸和每 rank 预算计划。 | 定义：`KVPPPhysicalCachePlan`, `build_layer_cache_bundles`, `get_kvpp_attention_kv_dims`, `build_kvpp_buffer_sizes` |
| [profiling_chunk_predictor.py](../../../vllm_ascend/core/profiling_chunk_predictor.py) | 供 profile-based chunk 调度使用的预测辅助。 | 注释：Profiling-based Dynamic Chunk Size Predictor.；定义：`ChunkSizePredictor`, `ProfilingChunkManager` |
| [recompute_scheduler.py](../../../vllm_ascend/core/recompute_scheduler.py) | Decode 侧相关重算/抢占与调度策略扩展。 | 定义：`RecomputeSchedulerConfig`, `RecomputeReqInfo` |
| [scheduler_profiling_chunk.py](../../../vllm_ascend/core/scheduler_profiling_chunk.py) | 依据 profile 信息选择 Prefill chunk 的调度扩展。 | 注释：Scheduler subclass with profiling-based dynamic chunk sizing.；定义：`ProfilingChunkScheduler`, `ProfilingChunkAsyncScheduler` |
| [short_request_first_scheduler.py](../../../vllm_ascend/core/short_request_first_scheduler.py) | 短请求优先相关调度策略。 | 定义：`ShortRequestFirstRequestQueue`, `ShortRequestFirstAsyncScheduler`, `is_recovery_request`, `install_short_request_first_waiting_queue` |
| [single_type_kv_cache_manager.py](../../../vllm_ascend/core/single_type_kv_cache_manager.py) | 针对 Ascend 规格扩展单类缓存 block 管理。 | 注释：Ascend allocation notifications for circular state sharing physical pages.；定义：`AscendCircularBufferManager` |
