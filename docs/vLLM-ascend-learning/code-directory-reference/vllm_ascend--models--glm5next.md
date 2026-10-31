# vllm_ascend/models/glm5next：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **17 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/models/glm5next/`

GLM5Next 模型结构和专有计算适配。

| 子目录 | 职责 |
| --- | --- |
| [ops/](../../../vllm_ascend/models/glm5next/ops) | ops 子模块；所属范围：GLM5Next 模型结构和专有计算适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/glm5next/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [attention.py](../../../vllm_ascend/models/glm5next/attention.py) | 注意力 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`Indexer`, `Glm5NextMLAAttention` |
| [cache_config.py](../../../vllm_ascend/models/glm5next/cache_config.py) | 配置、缓存 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：GLM-Next cache groups and source-compatible physical pool layout.；定义：`_Glm5NextCacheLayout`, `get_glm5_next_kv_cache_groups`, `get_glm5_next_pool_bytes_per_block`, `get_glm5_next_kv_cache_config` |
| [cache_views.py](../../../vllm_ascend/models/glm5next/cache_views.py) | 缓存 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Pooled-cache physical views for GLM-Next on Model Runner V1.；定义：`view_glm5_next_cache` |
| [config.py](../../../vllm_ascend/models/glm5next/config.py) | 配置 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`Glm5NextTextConfig`, `Glm5NextVisionConfig` |
| [kda.py](../../../vllm_ascend/models/glm5next/kda.py) | KDA 线性注意力 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate.；定义：`_Glm5NextMergedColumnParallelLinear`, `Glm5NextLinearAttention` |
| [kv_cache.py](../../../vllm_ascend/models/glm5next/kv_cache.py) | 缓存、KV 缓存 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：KV cache layers and metadata helpers for the GLM-Next pooled indexer.；定义：`Glm5NextIndexerCache`, `Glm5NextTailCache`, `is_glm5_next_cache_spec`, `get_kpool_tail_ring_capacity`, `format_indexer_kpool_slot_mapping` |
| [model.py](../../../vllm_ascend/models/glm5next/model.py) | 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`Glm5NextMLP`, `Glm5NextMoE`, `get_spec_layer_idx_from_weight_name` |
| [mtp.py](../../../vllm_ascend/models/glm5next/mtp.py) | 多 token 预测 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`Glm5NextMultiTokenPredictorLayer`, `Glm5NextMultiTokenPredictor` |
| [multimodal.py](../../../vllm_ascend/models/glm5next/multimodal.py) | 多模态 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：GLM-5.3-Flash vision tower and multimodal processor.；定义：`Glm5NextVisionPatchEmbed`, `Glm5NextVisionMLP` |
| [processor.py](../../../vllm_ascend/models/glm5next/processor.py) | processor 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：vLLM-native multimodal processor for GLM-5.3-Flash.；定义：`Glm5NextImageProcessorKwargs`, `Glm5NextImageProcessor`, `glm_sample_frame_indices`, `smart_resize` |
| [sparse_attn_indexer_kpool.py](../../../vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py) | 历史位置索引、注意力、池管理 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Tensor orchestration for the GLM-Next Triton KPool indexer.；定义：`SparseAttnIndexerKpool`, `append_causal_tail` |

## `vllm_ascend/models/glm5next/ops/`

ops 子模块；所属范围：GLM5Next 模型结构和专有计算适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/glm5next/ops/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [causal_conv1d.py](../../../vllm_ascend/models/glm5next/ops/causal_conv1d.py) | 卷积 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：AscendC short convolution for GLM prefill, decode and MTP verification.；定义：`causal_conv1d` |
| [kda.py](../../../vllm_ascend/models/glm5next/ops/kda.py) | KDA 线性注意力 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：GLM bounded-gate contracts for the AscendC KDA operators.；定义：`recurrent_kda`, `chunk_kda` |
| [mhc_ops.py](../../../vllm_ascend/models/glm5next/ops/mhc_ops.py) | mhc_ops 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Hyper-connection width helpers used by the GLM-5.3-Flash decoder layers.；定义：`hc_expand`, `hc_contract` |
| [state_ops.py](../../../vllm_ascend/models/glm5next/ops/state_ops.py) | 状态 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Ascend recurrent-state gather/scatter for the GLM-5.3-Flash KDA layers.；定义：`scatter_states` |
