# vllm_ascend/_310p：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **64 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/_310p/`

Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 子目录 | 职责 |
| --- | --- |
| [attention/](../../../vllm_ascend/_310p/attention) | 注意力 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |
| [fused_moe/](../../../vllm_ascend/_310p/fused_moe) | 混合专家 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |
| [ops/](../../../vllm_ascend/_310p/ops) | ops 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |
| [quantization/](../../../vllm_ascend/_310p/quantization) | 量化 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |
| [sample/](../../../vllm_ascend/_310p/sample) | 采样 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |
| [spec_decode/](../../../vllm_ascend/_310p/spec_decode) | 投机解码 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |
| [worker/](../../../vllm_ascend/_310p/worker) | 设备执行 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [block_table.py](../../../vllm_ascend/_310p/block_table.py) | 缓存页/分块 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`BlockTable`, `MultiGroupBlockTable` |
| [kv_block_zeroer.py](../../../vllm_ascend/_310p/kv_block_zeroer.py) | 缓存页/分块、KV 缓存 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendKVBlockZeroer310` |
| [kv_cache_sharing.py](../../../vllm_ascend/_310p/kv_cache_sharing.py) | 缓存、KV 缓存 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`get_310p_shared_cache_slots` |
| [model_runner_310p.py](../../../vllm_ascend/_310p/model_runner_310p.py) | 模型执行、模型 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`NPUModelRunner310` |
| [npu_input_batch.py](../../../vllm_ascend/_310p/npu_input_batch.py) | 批次、输入 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`NPUInputBatch310` |
| [sharded_state_loader_310p.py](../../../vllm_ascend/_310p/sharded_state_loader_310p.py) | 加载器、状态 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`ShardedStateLoader310` |
| [worker_310p.py](../../../vllm_ascend/_310p/worker_310p.py) | 设备执行 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`NPUWorker310` |

## `vllm_ascend/_310p/attention/`

注意力 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/attention/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [attention_mask.py](../../../vllm_ascend/_310p/attention/attention_mask.py) | 注意力 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AttentionMaskBuilder310`, `is_compressed_mask_supported` |
| [attention_v1.py](../../../vllm_ascend/_310p/attention/attention_v1.py) | 注意力 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendAttentionBackend310`, `AscendAttentionBackendImpl310` |
| [metadata_builder.py](../../../vllm_ascend/_310p/attention/metadata_builder.py) | 元数据 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendAttentionMetadataBuilder310`, `set_query_lens_cpu`, `get_query_lens_cpu` |

## `vllm_ascend/_310p/fused_moe/`

混合专家 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/fused_moe/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [fused_moe.py](../../../vllm_ascend/_310p/fused_moe/fused_moe.py) | 混合专家 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendUnquantizedFusedMoEMethod310`, `AscendRoutedExperts310` |
| [grouped_topk_router.py](../../../vllm_ascend/_310p/fused_moe/grouped_topk_router.py) | 专家路由、TopK 选择 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendGroupedTopKRouter310` |
| [moe_comm_method.py](../../../vllm_ascend/_310p/fused_moe/moe_comm_method.py) | 混合专家 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AllGatherCommImpl310` |
| [moe_mlp.py](../../../vllm_ascend/_310p/fused_moe/moe_mlp.py) | 混合专家 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`apply_moe_mlp` |
| [token_dispatcher.py](../../../vllm_ascend/_310p/fused_moe/token_dispatcher.py) | token 分发 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`TokenDispatcherWithAllGather310` |

## `vllm_ascend/_310p/ops/`

ops 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 子目录 | 职责 |
| --- | --- |
| [fla/](../../../vllm_ascend/_310p/ops/fla) | fla 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/ops/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [activation.py](../../../vllm_ascend/_310p/ops/activation.py) | 激活函数 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendSiluAndMul310` |
| [conv.py](../../../vllm_ascend/_310p/ops/conv.py) | 卷积 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendConv3dLayer310` |
| [gdn_attn_builder_310.py](../../../vllm_ascend/_310p/ops/gdn_attn_builder_310.py) | 注意力、Gated DeltaNet 注意力 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 注释：310P RC GDN metadata builder.；定义：`GDNAttentionMetadataBuilder310`, `AscendGDNAttentionBackend310` |
| [layernorm.py](../../../vllm_ascend/_310p/ops/layernorm.py) | 归一化 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendRMSNorm310`, `AscendGemmaRMSNorm310` |
| [mm_encoder_attention.py](../../../vllm_ascend/_310p/ops/mm_encoder_attention.py) | 注意力、编码器 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendMMEncoderAttention310`, `is_approximate_calculation_supported` |
| [qwen3vl_310.py](../../../vllm_ascend/_310p/ops/qwen3vl_310.py) | Qwen 模型 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`rot_pos_emb_310` |
| [rotary_embedding.py](../../../vllm_ascend/_310p/ops/rotary_embedding.py) | 向量查表、位置旋转 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendMRotaryEmbedding310`, `AscendRotaryEmbedding310`, `merge_mrope_cos_sin_for_apply`, `set_mrope_apply_rotary_slices`, `prepare_mrope_cos_sin_slices_from_runner` |
| [vocab_parallel_embedding.py](../../../vllm_ascend/_310p/ops/vocab_parallel_embedding.py) | 向量查表、并行 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendUnquantizedEmbeddingMethod310`, `AscendVocabParallelEmbedding310` |

## `vllm_ascend/_310p/ops/fla/`

fla 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/ops/fla/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`.chunk_gated_delta_rule`, `.fused_gdn_gating` |
| [chunk_gated_delta_rule.py](../../../vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py) | Delta 递归状态、门控 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`chunk_gated_delta_rule_pytorch`, `chunk_gated_delta_rule_310` |
| [fused_gdn_gating.py](../../../vllm_ascend/_310p/ops/fla/fused_gdn_gating.py) | 门控、Gated DeltaNet 注意力 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`fused_gdn_gating_pytorch` |
| [fused_recurrent_gated_delta_rule.py](../../../vllm_ascend/_310p/ops/fla/fused_recurrent_gated_delta_rule.py) | Delta 递归状态、门控 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`fused_recurrent_gated_delta_rule_pytorch` |
| [gdn_310.py](../../../vllm_ascend/_310p/ops/fla/gdn_310.py) | Gated DeltaNet 注意力 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendGatedDeltaNetAttention310`, `npu_recurrent_gated_delta_rule_310` |
| [idex.py](../../../vllm_ascend/_310p/ops/fla/idex.py) | 准备 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`prepare_chunk_indices_310`, `prepare_chunk_offsets_310` |
| [l2norm.py](../../../vllm_ascend/_310p/ops/fla/l2norm.py) | 权重 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`l2norm_310p` |

## `vllm_ascend/_310p/quantization/`

量化 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 子目录 | 职责 |
| --- | --- |
| [methods/](../../../vllm_ascend/_310p/quantization/methods) | methods 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/quantization/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`vllm_ascend._310p.quantization.modelslim_config` |
| [modelslim_config.py](../../../vllm_ascend/_310p/quantization/modelslim_config.py) | 配置、模型 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendModelSlimConfig310`, `create_scheme_for_layer` |

## `vllm_ascend/_310p/quantization/methods/`

methods 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/quantization/methods/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`.` |
| [registry.py](../../../vllm_ascend/_310p/quantization/methods/registry.py) | 注册表 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`register_scheme`, `get_scheme_class` |
| [w8a8_base.py](../../../vllm_ascend/_310p/quantization/methods/w8a8_base.py) | 线性投影 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendW8A8Linear310pScheme` |
| [w8a8_dynamic.py](../../../vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py) | 310P 动态 W8A8 Linear 与 MoE 方法，适配量化参数和设备计算。 | 定义：`AscendW8A8DynamicFusedMoEMethod310`, `AscendW8A8DynamicLinearMethod310` |
| [w8a8_static.py](../../../vllm_ascend/_310p/quantization/methods/w8a8_static.py) | 线性投影 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendW8A8LinearMethod310` |
| [w8a8s.py](../../../vllm_ascend/_310p/quantization/methods/w8a8s.py) | 线性投影 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendW8A8SLinearMethod310` |
| [w8a8sc.py](../../../vllm_ascend/_310p/quantization/methods/w8a8sc.py) | 线性投影 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendW8A8SCLinearMethod310` |

## `vllm_ascend/_310p/sample/`

采样 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/sample/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`vllm_ascend._310p.sample.sampler` |
| [rejection_sampler.py](../../../vllm_ascend/_310p/sample/rejection_sampler.py) | 候选拒绝验证、采样 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendRejectionSampler310` |
| [sampler.py](../../../vllm_ascend/_310p/sample/sampler.py) | 采样 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendTopKTopPSampler310`, `AscendSampler310`, `fill_exponential_310p` |

## `vllm_ascend/_310p/spec_decode/`

投机解码 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/spec_decode/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`vllm_ascend._310p.spec_decode.llm_base_proposer_310` |
| [llm_base_proposer_310.py](../../../vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py) | 候选生成 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`AscendSpecDecodeBaseProposer310` |

## `vllm_ascend/_310p/worker/`

设备执行 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 子目录 | 职责 |
| --- | --- |
| [v2/](../../../vllm_ascend/_310p/worker/v2) | v2 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/worker/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `vllm_ascend/_310p/worker/v2/`

v2 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 子目录 | 职责 |
| --- | --- |
| [spec_decode/](../../../vllm_ascend/_310p/worker/v2/spec_decode) | 投机解码 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/worker/v2/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [aclgraph.py](../../../vllm_ascend/_310p/worker/v2/aclgraph.py) | 设备图 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 注释：310P ACLGraph helpers for MRv2 MTP FULL_DECODE_ONLY.；定义：`ModelAclGraphManager310` |
| [block_table.py](../../../vllm_ascend/_310p/worker/v2/block_table.py) | 缓存页/分块 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`Ascend310PBlockTables` |
| [input_batch.py](../../../vllm_ascend/_310p/worker/v2/input_batch.py) | 批次、输入 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`Ascend310PInputBatch` |
| [kv_block_zeroer.py](../../../vllm_ascend/_310p/worker/v2/kv_block_zeroer.py) | 缓存页/分块、KV 缓存 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 注释：Model Runner V2 adapter for the existing 310P KV block zeroer.；定义：`AscendKVBlockZeroer310V2` |
| [model_runner.py](../../../vllm_ascend/_310p/worker/v2/model_runner.py) | 模型执行、模型 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`NPUModelRunner310V2` |
| [model_state.py](../../../vllm_ascend/_310p/worker/v2/model_state.py) | 模型、状态 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 注释：MRV2 model state for Ascend 310P (dense/VL + hybrid/GDN).；定义：`_Ascend310PModelStateMixin`, `Ascend310PModelState` |
| [rejection_sampler.py](../../../vllm_ascend/_310p/worker/v2/rejection_sampler.py) | 候选拒绝验证、采样 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 注释：MTP rejection sampler for 310P MRv2 (greedy + temperature, no Triton).；定义：`RejectionSampler310V2` |
| [rope.py](../../../vllm_ascend/_310p/worker/v2/rope.py) | 位置旋转 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`Ascend310PRopeState`, `get_310p_rope_state` |
| [sampler.py](../../../vllm_ascend/_310p/worker/v2/sampler.py) | 采样 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 注释：Triton-free MRV2 sampler for Ascend 310P.；定义：`Ascend310PSampler` |
| [spec_utils.py](../../../vllm_ascend/_310p/worker/v2/spec_utils.py) | 辅助函数 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 注释：CPU fallbacks for MRv2 spec-decode helpers (310P has no Triton).；定义：`expand_idx_mapping_cpu`, `combine_sampled_and_draft_tokens_cpu`, `get_num_sampled_and_rejected_cpu` |
| [states.py](../../../vllm_ascend/_310p/worker/v2/states.py) | 状态 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 定义：`Ascend310PStagedWriteTensor`, `Ascend310PRequestState` |

## `vllm_ascend/_310p/worker/v2/spec_decode/`

投机解码 子模块；所属范围：Ascend 310P 专用执行、输入、算子、采样及量化适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/_310p/worker/v2/spec_decode/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：310P MRv2 speculative-decoding helpers. |
| [aclgraph.py](../../../vllm_ascend/_310p/worker/v2/spec_decode/aclgraph.py) | 设备图 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 注释：310P draft ACLGraph manager: SpecDecoding capture + per-step decode FULL.；定义：`AutoRegressiveAclGraphManager310` |
| [mtp_speculator.py](../../../vllm_ascend/_310p/worker/v2/spec_decode/mtp_speculator.py) | 草稿生成、多 token 预测 的接口/实现或辅助；所属职责：Ascend 310P 专用执行、输入、算子、采样及量化适配。 | 注释：310P MTP speculator: CPU block-table slot mappings + RoPE flag + draft quant.；定义：`AscendMTPSpeculator310` |
