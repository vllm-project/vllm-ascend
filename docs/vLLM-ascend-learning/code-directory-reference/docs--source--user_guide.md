# docs/source/user_guide：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **56 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `docs/source/user_guide/`

配置、部署、功能及支持范围说明。

| 子目录 | 职责 |
| --- | --- |
| [configuration/](../../source/user_guide/configuration) | 父主题的 configuration 文档、模板或站点资源。 |
| [deployment_guide/](../../source/user_guide/deployment_guide) | 父主题的 deployment_guide 文档、模板或站点资源。 |
| [feature_guide/](../../source/user_guide/feature_guide) | 父主题的 feature_guide 文档、模板或站点资源。 |
| [support_matrix/](../../source/user_guide/support_matrix) | 父主题的 support_matrix 文档、模板或站点资源。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [index.md](../../source/user_guide/index.md) | 说明 User Guide。 | 标题：User Guide |
| [ms_service_metric.md](../../source/user_guide/ms_service_metric.md) | 说明 MS Service Metric。 | 标题：MS Service Metric |
| [release_notes.md](../../source/user_guide/release_notes.md) | 说明 Release Notes。 | 标题：Release Notes |

## `docs/source/user_guide/configuration/`

父主题的 configuration 文档、模板或站点资源。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [additional_config.md](../../source/user_guide/configuration/additional_config.md) | 说明 Additional Configuration。 | 标题：Additional Configuration |
| [encoder_cache_manager.md](../../source/user_guide/configuration/encoder_cache_manager.md) | 说明 Score Encoder Cache Manager。 | 标题：Score Encoder Cache Manager |
| [env_vars.md](../../source/user_guide/configuration/env_vars.md) | 说明 Environment Variables。 | 标题：Environment Variables |
| [index.md](../../source/user_guide/configuration/index.md) | 说明 Configuration Guide。 | 标题：Configuration Guide |

## `docs/source/user_guide/deployment_guide/`

父主题的 deployment_guide 文档、模板或站点资源。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [index.md](../../source/user_guide/deployment_guide/index.md) | 说明 Deployment Guide。 | 标题：Deployment Guide |
| [using_mindie_motor.md](../../source/user_guide/deployment_guide/using_mindie_motor.md) | 说明 Deploy vLLM-Ascend with MindIE-Motor。 | 标题：Deploy vLLM-Ascend with MindIE-Motor |
| [using_volcano_kthena.md](../../source/user_guide/deployment_guide/using_volcano_kthena.md) | 说明 Using Volcano Kthena。 | 标题：Using Volcano Kthena |

## `docs/source/user_guide/feature_guide/`

父主题的 feature_guide 文档、模板或站点资源。

| 子目录 | 职责 |
| --- | --- |
| [images/](../../source/user_guide/feature_guide/images) | 父主题的示意图、截图和文档展示资源。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [Ai_QoS_introduction_en.md](../../source/user_guide/feature_guide/Ai_QoS_introduction_en.md) | 说明 AI QoS Feature。 | 标题：AI QoS Feature |
| [Fine_grained_TP.md](../../source/user_guide/feature_guide/Fine_grained_TP.md) | 说明 Fine-Grained Tensor Parallelism (Fine-grained TP)。 | 标题：Fine-Grained Tensor Parallelism (Fine-grained TP) |
| [batch_invariance.md](../../source/user_guide/feature_guide/batch_invariance.md) | 说明 Batch Invariance。 | 标题：Batch Invariance |
| [batch_job_aware_scheduler.md](../../source/user_guide/feature_guide/batch_job_aware_scheduler.md) | 说明 Batch-Job-Aware Scheduler。 | 标题：Batch-Job-Aware Scheduler |
| [context_parallel.md](../../source/user_guide/feature_guide/context_parallel.md) | 说明 Context Parallel Guide。 | 标题：Context Parallel Guide |
| [cpu_binding.md](../../source/user_guide/feature_guide/cpu_binding.md) | 说明 CPU Binding。 | 标题：CPU Binding |
| [dynamic_chunk_pipeline_parallel.md](../../source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md) | 说明 Dynamic Chunked Pipeline Parallel。 | 标题：Dynamic Chunked Pipeline Parallel |
| [dyntra_lb.md](../../source/user_guide/feature_guide/dyntra_lb.md) | 说明 DyntraLB: Dynamic Intra-Decoder Data-Parallel Load Balancing。 | 标题：DyntraLB: Dynamic Intra-Decoder Data-Parallel Load Balancing |
| [epd_disaggregation.md](../../source/user_guide/feature_guide/epd_disaggregation.md) | 说明 Disaggregated-encoder。 | 标题：Disaggregated-encoder |
| [expert_parallelism_load_balancer.md](../../source/user_guide/feature_guide/expert_parallelism_load_balancer.md) | 说明 Expert Parallelism Load Balancer (EPLB)。 | 标题：Expert Parallelism Load Balancer (EPLB) |
| [flash_attention.md](../../source/user_guide/feature_guide/flash_attention.md) | 说明 Flash Attention 3。 | 标题：Flash Attention 3 |
| [graph_mode.md](../../source/user_guide/feature_guide/graph_mode.md) | 说明 Graph Mode Guide。 | 标题：Graph Mode Guide |
| [index.md](../../source/user_guide/feature_guide/index.md) | 说明 Feature Guide。 | 标题：Feature Guide |
| [kv_cache_cpu_offload.md](../../source/user_guide/feature_guide/kv_cache_cpu_offload.md) | 说明 KV Cache CPU Offload Guide。 | 标题：KV Cache CPU Offload Guide |
| [kv_cache_pool_ascend_store.md](../../source/user_guide/feature_guide/kv_cache_pool_ascend_store.md) | 说明 KV Cache Pool (Ascend Store)。 | 标题：KV Cache Pool (Ascend Store) |
| [kvpp.md](../../source/user_guide/feature_guide/kvpp.md) | 说明 KVPP (Experimental)。 | 标题：KVPP (Experimental) |
| [large_scale_ep.md](../../source/user_guide/feature_guide/large_scale_ep.md) | 说明 Distributed DP Server With Large-Scale Expert Parallelism。 | 标题：Distributed DP Server With Large-Scale Expert Parallelism |
| [layerwise_and_sparse_kv_cache_offloading.md](../../source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md) | 说明 Layerwise and Sparse KV Cache Offloading Guide。 | 标题：Layerwise and Sparse KV Cache Offloading Guide |
| [lmcache_ascend_deployment.md](../../source/user_guide/feature_guide/lmcache_ascend_deployment.md) | 说明 LMCache-Ascend Deployment Guide。 | 标题：LMCache-Ascend Deployment Guide |
| [lora.md](../../source/user_guide/feature_guide/lora.md) | 说明 LoRA Adapters。 | 标题：LoRA Adapters |
| [mooncake_hybrid_attention.md](../../source/user_guide/feature_guide/mooncake_hybrid_attention.md) | 说明 Mooncake Layerwise Hybrid Attention。 | 标题：Mooncake Layerwise Hybrid Attention |
| [mooncake_layerwise_adaptation_and_optimization.md](../../source/user_guide/feature_guide/mooncake_layerwise_adaptation_and_optimization.md) | 说明 Mooncake Layerwise Adaptation and Optimization Analysis。 | 标题：Mooncake Layerwise Adaptation and Optimization Analysis |
| [netloader.md](../../source/user_guide/feature_guide/netloader.md) | 说明 Netloader Guide。 | 标题：Netloader Guide |
| [pipeline_parallel.md](../../source/user_guide/feature_guide/pipeline_parallel.md) | 说明 Pipeline Parallelism。 | 标题：Pipeline Parallelism |
| [preempt_offload_connector.md](../../source/user_guide/feature_guide/preempt_offload_connector.md) | 说明 Preempt Offload Guide。 | 标题：Preempt Offload Guide |
| [quantization.md](../../source/user_guide/feature_guide/quantization.md) | 说明 Quantization Guide。 | 标题：Quantization Guide |
| [rfork.md](../../source/user_guide/feature_guide/rfork.md) | 说明 RFork Guide。 | 标题：RFork Guide |
| [rl.md](../../source/user_guide/feature_guide/rl.md) | 说明 vLLM-Ascend for RL。 | 标题：vLLM-Ascend for RL |
| [sequence_parallelism.md](../../source/user_guide/feature_guide/sequence_parallelism.md) | 说明 Sequence Parallelism。 | 标题：Sequence Parallelism |
| [short_request_first.md](../../source/user_guide/feature_guide/short_request_first.md) | 说明 ShortRequestFirst Prefill Scheduling。 | 标题：ShortRequestFirst Prefill Scheduling |
| [sleep_mode.md](../../source/user_guide/feature_guide/sleep_mode.md) | 说明 Sleep Mode Guide。 | 标题：Sleep Mode Guide |
| [speculative_decoding.md](../../source/user_guide/feature_guide/speculative_decoding.md) | 说明 Speculative Decoding Guide。 | 标题：Speculative Decoding Guide |
| [structured_output.md](../../source/user_guide/feature_guide/structured_output.md) | 说明 Structured Output Guide。 | 标题：Structured Output Guide |
| [ucm_deployment.md](../../source/user_guide/feature_guide/ucm_deployment.md) | 说明 UCM Store Deployment Guide。 | 标题：UCM Store Deployment Guide |

## `docs/source/user_guide/feature_guide/images/`

父主题的示意图、截图和文档展示资源。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [ai_qos1.png](../../source/user_guide/feature_guide/images/ai_qos1.png) | 文档配图/展示资源：ai_qos1；不参与模型计算。 | 图像/文档资源；未执行模型逻辑分析。 |
| [ai_qos2.png](../../source/user_guide/feature_guide/images/ai_qos2.png) | 文档配图/展示资源：ai_qos2；不参与模型计算。 | 图像/文档资源；未执行模型逻辑分析。 |
| [epd_disaggregation.jpg](../../source/user_guide/feature_guide/images/epd_disaggregation.jpg) | 文档配图/展示资源：epd_disaggregation；不参与模型计算。 | 图像/文档资源；未执行模型逻辑分析。 |
| [eplb_img.png](../../source/user_guide/feature_guide/images/eplb_img.png) | 文档配图/展示资源：eplb_img；不参与模型计算。 | 图像/文档资源；未执行模型逻辑分析。 |
| [netloader_flowchart.png](../../source/user_guide/feature_guide/images/netloader_flowchart.png) | 文档配图/展示资源：netloader_flowchart；不参与模型计算。 | 图像/文档资源；未执行模型逻辑分析。 |
| [netloader_timing_diagram.png](../../source/user_guide/feature_guide/images/netloader_timing_diagram.png) | 文档配图/展示资源：netloader_timing_diagram；不参与模型计算。 | 图像/文档资源；未执行模型逻辑分析。 |
| [rfork_flowchart.jpg](../../source/user_guide/feature_guide/images/rfork_flowchart.jpg) | 文档配图/展示资源：rfork_flowchart；不参与模型计算。 | 图像/文档资源；未执行模型逻辑分析。 |
| [structured_output_1.png](../../source/user_guide/feature_guide/images/structured_output_1.png) | 文档配图/展示资源：structured_output_1；不参与模型计算。 | 图像/文档资源；未执行模型逻辑分析。 |

## `docs/source/user_guide/support_matrix/`

父主题的 support_matrix 文档、模板或站点资源。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [feature_matrix.md](../../source/user_guide/support_matrix/feature_matrix.md) | 说明 feature_matrix。 | 标题：feature_matrix |
| [index.md](../../source/user_guide/support_matrix/index.md) | 说明 support_matrix。 | 标题：support_matrix |
| [supported_features.md](../../source/user_guide/support_matrix/supported_features.md) | 说明 Supported Features。 | 标题：Supported Features |
| [supported_models.md](../../source/user_guide/support_matrix/supported_models.md) | 说明 Supported Models。 | 标题：Supported Models |
