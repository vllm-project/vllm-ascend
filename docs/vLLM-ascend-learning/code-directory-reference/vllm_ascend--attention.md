# vllm_ascend/attention：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **24 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

先理解 Backend / MetadataBuilder / Impl 三种职责，再按实际模型选择路径。DeepSeek V4.1 要把 `dsa_v41.py` 和 `models/deepseek_v41/model.py`、cache config、Compressor、Indexer 一起读；它的 cache backend 与独立持有的 DSA Impl 有专门分工。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `vllm_ascend/attention/`

Attention 后端、元数据构建、KV 访问和稀疏注意力执行。

| 子目录 | 职责 |
| --- | --- |
| [context_parallel/](../../../vllm_ascend/attention/context_parallel) | 上下文并行下的 Attention、长度坐标、缓存与结果组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/attention/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [attention_c8_mxfp.py](../../../vllm_ascend/attention/attention_c8_mxfp.py) | C8/MXFP KV Attention 后端及量化 scale/维度校验。 | 注释：C8-MXFP (MXFP8) KV cache attention backend (QFA dual-operator interface).；定义：`AscendC8MXFPAttentionBackend`, `AscendC8MXFPMetadataBuilder`, `validate_mxfp_k_scale_head_dim`, `validate_mxfp_v_scale_block_size`, `mxfp_kv_scale_groups` |
| [attention_mask.py](../../../vllm_ascend/attention/attention_mask.py) | 构建 Attention 可见范围与屏蔽张量。 | 定义：`AttentionMaskBuilder` |
| [attention_v1.py](../../../vllm_ascend/attention/attention_v1.py) | 通用 Ascend Attention 后端、运行状态和元数据/计算实现。 | 定义：`AscendAttentionBackend`, `AscendAttentionState` |
| [dsa_attn_kv_plan.py](../../../vllm_ascend/attention/dsa_attn_kv_plan.py) | 集中决定 DSA Attention KV 的 dtype 与执行计划。 | 注释：Centralized DeepSeek-V4 attention-KV execution choices.；定义：`DsaAttnKvPlan`, `resolve_dsv4_cache_dtype`, `is_a5_bf16_kv_enabled`, `write_dsa_cache` |
| [dsa_v1.py](../../../vllm_ascend/attention/dsa_v1.py) | DeepSeek 稀疏注意力后端、元数据、前后投影与设备执行。 | 定义：`AscendDSABackend`, `AscendDSAC4Backend`, `reset_compressor_metadata_cache`, `get_or_compute_compressor_metadata`, `build_compressor_metadata_out` |
| [dsa_v41.py](../../../vllm_ascend/attention/dsa_v41.py) | V4.1 元数据及注意力执行边界；编排 Q/KV、压缩、Indexer、缓存写入和 SFA。 | 注释：DeepSeek V4.1 DSA metadata and fused attention execution.；定义：`AscendDSAV41Impl`, `AscendDSAV41Metadata`, `DeepseekV41CacheLayer`, `dsa_v41_forward`, `dsa_v41_forward_fake`, `compressed_slot_mapping` |
| [fa3_v1.py](../../../vllm_ascend/attention/fa3_v1.py) | FA3 相关 Ascend Attention 后端和实现。 | 定义：`AscendFABackend`, `AscendFAImpl` |
| [indexer.py](../../../vllm_ascend/attention/indexer.py) | 稀疏 Attention Indexer 后端、元数据与查询执行。 | 定义：`AscendSFAIndexerMetadata`, `AscendSFAIndexerBackend` |
| [indexer_kpool.py](../../../vllm_ascend/attention/indexer_kpool.py) | GLM-Next pooled Indexer 的查询、K pool 元数据和后端。 | 注释：Cache metadata and execution backends for the GLM-Next pooled indexer.；定义：`AscendIndexerKPoolQueryMetadata`, `AscendIndexerKPoolMetadata` |
| [mixed_quant_sparse_flash_mla.py](../../../vllm_ascend/attention/mixed_quant_sparse_flash_mla.py) | 将 DSA 参数适配到混合量化 SparseFlashMla 算子。 | 注释：Adapt DSA arguments to the installed MixedQuantSparseFlashMla operators.；定义：`mixed_quant_sparse_flash_mla_metadata`, `mixed_quant_sparse_flash_mla` |
| [mla_v1.py](../../../vllm_ascend/attention/mla_v1.py) | Ascend MLA 后端及其查询、缓存和注意力执行适配。 | 定义：`AscendMLABackend`, `AscendMLAAttention` |
| [sfa_kv_offload.py](../../../vllm_ascend/attention/sfa_kv_offload.py) | 稀疏 KV Offload 使用的独立 SFA 后端与复制/查询衔接。 | 注释：Standalone SFA backend for Sparse KV offload.；定义：`AscendSFAOffloadMetadata`, `_FusedOverlapDecodeCommonInputs`, `prepare_copy_sfa_queries` |
| [sfa_v1.py](../../../vllm_ascend/attention/sfa_v1.py) | 稀疏 MLA/SFA 后端、计划/元数据构建和执行。 | 定义：`SparseMLAMetadataState`, `AscendSFABackend`, `generate_smla_plan`, `build_smla_metadata`, `sparse_mla` |
| [sparse_flash_mla.py](../../../vllm_ascend/attention/sparse_flash_mla.py) | 将模型侧参数适配到安装的 SparseFlashMla 算子接口。 | 定义：`sparse_flash_mla_metadata`, `sparse_flash_mla` |
| [utils.py](../../../vllm_ascend/attention/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：Attention 后端、元数据构建、KV 访问和稀疏注意力执行。 | 定义：`PreprocessType`, `PagedAttentionGraphParam`, `mark_fused_preprocess_weights`, `get_or_register_attention_buffer`, `build_valid_topk_mask` |

## `vllm_ascend/attention/context_parallel/`

上下文并行下的 Attention、长度坐标、缓存与结果组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/attention/context_parallel/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [attention_cp.py](../../../vllm_ascend/attention/context_parallel/attention_cp.py) | 注意力、上下文并行 的接口/实现或辅助；所属职责：上下文并行下的 Attention、长度坐标、缓存与结果组织。 | 定义：`AscendMetadataForPrefill`, `AscendMetadataForDecode`, `build_dcp_fia_params` |
| [common_cp.py](../../../vllm_ascend/attention/context_parallel/common_cp.py) | 公共逻辑、上下文并行 的接口/实现或辅助；所属职责：上下文并行下的 Attention、长度坐标、缓存与结果组织。 | 定义：`CPKVScope`, `DCPMetadataBuilderMixin`, `is_pcp_decode_sharding_enabled`, `get_pcp_num_replicated_tokens`, `use_history_current_split_decode` |
| [dsa_cp.py](../../../vllm_ascend/attention/context_parallel/dsa_cp.py) | DeepSeek 稀疏注意力、上下文并行 的接口/实现或辅助；所属职责：上下文并行下的 Attention、长度坐标、缓存与结果组织。 | 定义：`DSACPMetadata`, `AscendDSAReqMetadata`, `restore_tp_heads`, `hadamard_transform_ref`, `rotate_activation` |
| [dsa_v41_cp.py](../../../vllm_ascend/attention/context_parallel/dsa_v41_cp.py) | DeepSeek 稀疏注意力、上下文并行 的接口/实现或辅助；所属职责：上下文并行下的 Attention、长度坐标、缓存与结果组织。 | 注释：V4.1 replicated-cache adapters for legacy TP-token CP and MRV2 PCP.；定义：`_ReplicatedCacheMetadataBuilder`, `AscendDSAV41CPMetadataBuilder`, `get_v41_cp_classes` |
| [mla_cp.py](../../../vllm_ascend/attention/context_parallel/mla_cp.py) | MLA 潜在注意力、上下文并行 的接口/实现或辅助；所属职责：上下文并行下的 Attention、长度坐标、缓存与结果组织。 | 定义：`MLASplitAttentionGraphParams`, `DCPChunkedContextMetadata` |
| [sfa_cp.py](../../../vllm_ascend/attention/context_parallel/sfa_cp.py) | 稀疏注意力、上下文并行 的接口/实现或辅助；所属职责：上下文并行下的 Attention、长度坐标、缓存与结果组织。 | 定义：`AscendSFAPCPImpl`, `DSACPContext`, `resolve_sfa_metadata_builder`, `resolve_sfa_impl` |
| [sfa_dcp_utils.py](../../../vllm_ascend/attention/context_parallel/sfa_dcp_utils.py) | 辅助函数、稀疏注意力、Decode 上下文并行 的接口/实现或辅助；所属职责：上下文并行下的 Attention、长度坐标、缓存与结果组织。 | 注释：SFA-specific CP helpers shared by the attention and indexer builders.；定义：`get_sfa_pcp_global_metadata`, `get_sfa_dcp_max_local_block_table_cols`, `get_sfa_dcp_local_block_table` |
