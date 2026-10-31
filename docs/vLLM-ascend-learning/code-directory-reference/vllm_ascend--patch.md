# vllm_ascend/patch：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **71 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/patch/`

在既有上游接口缺少所需扩展点时安装兼容补丁。

| 子目录 | 职责 |
| --- | --- |
| [platform/](../../../vllm_ascend/patch/platform) | 平台、调度、配置、注册与引擎初始化相关补丁。 |
| [worker/](../../../vllm_ascend/patch/worker) | Worker、模型、算子和执行期行为的补丁。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/patch/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `vllm_ascend/patch/platform/`

平台、调度、配置、注册与引擎初始化相关补丁。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/patch/platform/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`os`, `vllm_ascend` |
| [patch_balance_schedule.py](../../../vllm_ascend/patch/platform/patch_balance_schedule.py) | 对上游 调度 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Balance scheduling patch.；定义：`BalanceScheduler`, `BalanceDPEngineCoreProc` |
| [patch_deepseek_v4_vision.py](../../../vllm_ascend/patch/platform/patch_deepseek_v4_vision.py) | 对上游 DeepSeek 模型、视觉编码 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：vLLM v0.27 model-config compatibility for DeepSeek-V4 vision.；定义：`register_deepseek_v4_vision_config_convertor` |
| [patch_distributed.py](../../../vllm_ascend/patch/platform/patch_distributed.py) | 对上游 distributed 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`NullHandle`, `communication_adaptation_310p` |
| [patch_dp_device_ids.py](../../../vllm_ascend/patch/platform/patch_dp_device_ids.py) | 对上游 dp_device_ids 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_patched_get_physical_gpu_ids_for_local_dp_rank` |
| [patch_dyntra_lb_core.py](../../../vllm_ascend/patch/platform/patch_dyntra_lb_core.py) | 对上游 dyntra_lb_core 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_RequestItem`, `_CardData` |
| [patch_engine_core.py](../../../vllm_ascend/patch/platform/patch_engine_core.py) | 对上游 engine_core 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_patch_dp_engine_core_proc`, `_run_engine_core_patch_func`, `_apply_patch` |
| [patch_engram_config.py](../../../vllm_ascend/patch/platform/patch_engram_config.py) | 对上游 Engram 记忆、配置 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Allow Engram on Ascend and host table sharing across PCP replicas. |
| [patch_eplb.py](../../../vllm_ascend/patch/platform/patch_eplb.py) | 对上游 专家负载均衡 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Narrow vLLM EPLB construction, execution, and commit adapters for Ascend.；定义：`_AscendAsyncLayerResult`, `_DeferredConsumedEvent`, `resolve_ascend_eplb_communicator` |
| [patch_example_connector.py](../../../vllm_ascend/patch/platform/patch_example_connector.py) | 对上游 example_connector 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_patched_start_load_caches` |
| [patch_fused_moe.py](../../../vllm_ascend/patch/platform/patch_fused_moe.py) | 对上游 混合专家 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_adapt_routed_experts_cls`, `_ascend_apply_eplb_mapping`, `_adapt_eplb_router` |
| [patch_glm53_reasoning.py](../../../vllm_ascend/patch/platform/patch_glm53_reasoning.py) | 对上游 glm53_reasoning 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Backport GLM-5.3 always-on reasoning from vllm-project/vllm#56994.；定义：`_glm53_always_thinks`, `_patch_glm53_reasoning` |
| [patch_glm5next_config.py](../../../vllm_ascend/patch/platform/patch_glm5next_config.py) | 对上游 配置 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_is_deepseek_mla` |
| [patch_indexer_kv_dtype.py](../../../vllm_ascend/patch/platform/patch_indexer_kv_dtype.py) | 对上游 历史位置索引、KV 缓存 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_apply_indexer_kv_dtype_int8_patch` |
| [patch_kv_cache_coordinator.py](../../../vllm_ascend/patch/platform/patch_kv_cache_coordinator.py) | 对上游 缓存、KV 缓存 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`AscendHybridKVCacheCoordinator`, `get_kv_cache_coordinator` |
| [patch_kv_cache_dtype.py](../../../vllm_ascend/patch/platform/patch_kv_cache_dtype.py) | 对上游 缓存、KV 缓存 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_apply_kv_cache_dtype_int8_patch` |
| [patch_kv_cache_utils.py](../../../vllm_ascend/patch/platform/patch_kv_cache_utils.py) | 对上游 缓存、辅助函数、KV 缓存 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`group_and_unify_kv_cache_specs` |
| [patch_lopt_tokenization.py](../../../vllm_ascend/patch/platform/patch_lopt_tokenization.py) | 对上游 分词 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`apply_lopt_patch` |
| [patch_mamba_block_aligned_split.py](../../../vllm_ascend/patch/platform/patch_mamba_block_aligned_split.py) | 对上游 缓存页/分块、Mamba 状态空间 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Apply Ascend-specific Mamba chunk-boundary handling.；定义：`_mamba_block_aligned_split` |
| [patch_mamba_config.py](../../../vllm_ascend/patch/platform/patch_mamba_config.py) | 对上游 配置、Mamba 状态空间 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`verify_and_update_config` |
| [patch_mamba_config_310.py](../../../vllm_ascend/patch/platform/patch_mamba_config_310.py) | 对上游 配置、Mamba 状态空间 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`verify_and_update_config` |
| [patch_mamba_manager.py](../../../vllm_ascend/patch/platform/patch_mamba_manager.py) | 对上游 管理器、Mamba 状态空间 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`AscendMambaManager` |
| [patch_minimax_m2_config.py](../../../vllm_ascend/patch/platform/patch_minimax_m2_config.py) | 对上游 MiniMax 模型、配置 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_get_model_type`, `_should_disable_fp8`, `_disable_fp8` |
| [patch_mla_prefill_backend.py](../../../vllm_ascend/patch/platform/patch_mla_prefill_backend.py) | 对上游 上下文预填充、MLA 潜在注意力 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`AscendMLAPrefillBackend` |
| [patch_multiproc_executor.py](../../../vllm_ascend/patch/platform/patch_multiproc_executor.py) | 对上游 multiproc_executor 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`AscendMultiprocExecutor`, `AscendWorkerProc` |
| [patch_parallel_config.py](../../../vllm_ascend/patch/platform/patch_parallel_config.py) | 对上游 并行、配置 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_use_sequence_parallel_moe` |
| [patch_pp_mtp.py](../../../vllm_ascend/patch/platform/patch_pp_mtp.py) | 对上游 多 token 预测 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Backport vLLM PP + MTP runtime support.；定义：`_is_pd_prefill_node`, `_use_pp_mtp_runtime_patch`, `_use_pp_ipc_runtime_patch` |
| [patch_profiling_chunk.py](../../../vllm_ascend/patch/platform/patch_profiling_chunk.py) | 对上游 性能分析 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Patches for profiling-based dynamic chunk sizing.；定义：`_record_execution_timing`, `_ensure_update_from_output_wrapped`, `_ensure_schedule_wrapped` |
| [patch_speculative_config.py](../../../vllm_ascend/patch/platform/patch_speculative_config.py) | 对上游 配置 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_normalize_kimi_dflash_rope`, `_normalize_legacy_qwen3_dspark_config`, `_normalize_deepseek_dspark_draft` |
| [patch_structured_output.py](../../../vllm_ascend/patch/platform/patch_structured_output.py) | 对上游 输出 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_request_backend`, `_backend_name_from_instance`, `_raise_mixed_backend` |
| [patch_torch_accelerator.py](../../../vllm_ascend/patch/platform/patch_torch_accelerator.py) | 对上游 torch_accelerator 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`patch_empty_cache` |
| [patch_trace.py](../../../vllm_ascend/patch/platform/patch_trace.py) | 对上游 时间线跟踪 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_patch_request_status`, `_patch_chat_stream_generator` |
| [patch_use_v2_model_runner.py](../../../vllm_ascend/patch/platform/patch_use_v2_model_runner.py) | 对上游 模型执行、模型 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_patched_use_v2_model_runner`, `_patched_get_unsupported_features`, `_patched_validate_v2_model_runner` |

## `vllm_ascend/patch/worker/`

Worker、模型、算子和执行期行为的补丁。

| 子目录 | 职责 |
| --- | --- |
| [patch_v2/](../../../vllm_ascend/patch/worker/patch_v2) | Model Runner V2 路径所需的上游行为适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/patch/worker/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`vllm.triton_utils`, `vllm_ascend.device.hardware_profile` |
| [_hccl_pg_registry.py](../../../vllm_ascend/patch/worker/_hccl_pg_registry.py) | 对上游 注册表、HCCL 通信 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`HcclPgKey`, `RegistryEntry`, `make_hccl_pg_key` |
| [patch_bind_kv_cache.py](../../../vllm_ascend/patch/worker/patch_bind_kv_cache.py) | 对上游 缓存、KV 缓存 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`bind_kv_cache`, `bind_kv_cache_to_layers` |
| [patch_cudagraph.py](../../../vllm_ascend/patch/worker/patch_cudagraph.py) | 对上游 设备图 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_create_padded_batch_descriptor` |
| [patch_deepseek_v2.py](../../../vllm_ascend/patch/worker/patch_deepseek_v2.py) | 对上游 DeepSeek 模型 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_should_skip_indexer_init`, `_deepseek_v2_mla_attention_init`, `_find_topk_indices_buffer` |
| [patch_distributed.py](../../../vllm_ascend/patch/worker/patch_distributed.py) | 对上游 distributed 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`GroupCoordinatorPatch` |
| [patch_eagle3_init.py](../../../vllm_ascend/patch/worker/patch_eagle3_init.py) | 对上游 Eagle 投机 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Patch: fix target_layer_num for Eagle3 draft models under Pipeline Parallelism.；定义：`_patched_eagle3_llama_init`, `_patched_eagle3_deepseek_v2_init` |
| [patch_eagle3_pp_aux.py](../../../vllm_ascend/patch/worker/patch_eagle3_pp_aux.py) | 对上游 Eagle 投机 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Patch: Propagate Eagle3 aux hidden states through PP pipeline.；定义：`patch_eagle3_pp_aux_propagation` |
| [patch_fused_moe.py](../../../vllm_ascend/patch/worker/patch_fused_moe.py) | 对上游 混合专家 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm_ascend.patch.platform.patch_fused_moe` |
| [patch_idex_310.py](../../../vllm_ascend/patch/worker/patch_idex_310.py) | 对上游 idex_310 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn`, `vllm.third_party.flash_linear_attention.ops` |
| [patch_kimi_k25.py](../../../vllm_ascend/patch/worker/patch_kimi_k25.py) | 对上游 Kimi 模型 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`AscendLearnable2DInterpPosEmbDivided_fixed`, `get_rope_shape` |
| [patch_kv_cache_dtype.py](../../../vllm_ascend/patch/worker/patch_kv_cache_dtype.py) | 对上游 缓存、KV 缓存 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：vllm ``releases/v0.27.1`` does not ship the pluggable kv-cache-dtype；定义：`_apply_fp8_kv_cache_dtype_flip`, `_widen_cache_dtype_literal_for_worker`, `_widen_indexer_kv_dtype_literal_for_worker` |
| [patch_mamba_utils.py](../../../vllm_ascend/patch/worker/patch_mamba_utils.py) | 对上游 Mamba 状态空间、辅助函数 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`prepare_mamba_copy_by_layer`, `do_mamba_copy_block_for_layer`, `finish_mamba_copy_by_layer` |
| [patch_minimax_m2.py](../../../vllm_ascend/patch/worker/patch_minimax_m2.py) | 对上游 MiniMax 模型 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_patch_forward` |
| [patch_qwen3_5.py](../../../vllm_ascend/patch/worker/patch_qwen3_5.py) | 对上游 Qwen 模型 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`AscendQwen3NextAttention`, `AscendQwen3_5DecoderLayer` |
| [patch_qwen3_dflash.py](../../../vllm_ascend/patch/worker/patch_qwen3_dflash.py) | 对上游 DFlash 草稿、Qwen 模型 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`precompute_and_store_context_kv` |
| [patch_qwen3vl.py](../../../vllm_ascend/patch/worker/patch_qwen3vl.py) | 对上游 Qwen 模型 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`tensor_parallel_wrap`, `forward_with_split_qkv_rmsnorm_mrope`, `patch_qwen3_vl_moe_pp_layer_range` |
| [patch_rejection_sampler.py](../../../vllm_ascend/patch/worker/patch_rejection_sampler.py) | 对上游 候选拒绝验证、采样 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm.v1.sample.rejection_sampler`, `vllm_ascend.sample.rejection_sampler` |
| [patch_routed_experts_capture.py](../../../vllm_ascend/patch/worker/patch_routed_experts_capture.py) | 对上游 专家 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`capture` |
| [patch_step3p5.py](../../../vllm_ascend/patch/worker/patch_step3p5.py) | 对上游 step3p5 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_patched_attention_forward`, `_patched_fused_moe_block_init`, `_patched_decoder_layer_init` |
| [patch_trace.py](../../../vllm_ascend/patch/worker/patch_trace.py) | 对上游 时间线跟踪 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm_ascend.trace` |
| [patch_triton.py](../../../vllm_ascend/patch/worker/patch_triton.py) | 对上游 triton 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_npu_causal_conv1d_update`, `_npu_causal_conv1d_fn` |

## `vllm_ascend/patch/worker/patch_v2/`

Model Runner V2 路径所需的上游行为适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/patch/worker/patch_v2/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [patch_adaptive_verification.py](../../../vllm_ascend/patch/worker/patch_v2/patch_adaptive_verification.py) | 对上游 adaptive_verification 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_IndexFillMode` |
| [patch_attn_utils.py](../../../vllm_ascend/patch/worker/patch_v2/patch_attn_utils.py) | 对上游 辅助函数、注意力 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`_get_ascend_sfa_indexer_backend` |
| [patch_block_table.py](../../../vllm_ascend/patch/worker/patch_v2/patch_block_table.py) | 对上游 缓存页/分块 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm.v1.worker.gpu`, `vllm_ascend.device.hardware_profile` |
| [patch_dflash_speculator.py](../../../vllm_ascend/patch/worker/patch_v2/patch_dflash_speculator.py) | 对上游 草稿生成、DFlash 草稿 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm.v1.worker.gpu.spec_decode.dflash.cudagraph`, `vllm.v1.worker.gpu.spec_decode.dflash.speculator` |
| [patch_dspark.py](../../../vllm_ascend/patch/worker/patch_v2/patch_dspark.py) | 对上游 DSpark 草稿 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：Patch DSpark draft loading to inherit the target quant config.；定义：`_pp_safe_maybe_share_target_embed`, `_load_dspark_model_with_target_quant` |
| [patch_eagle_speculator.py](../../../vllm_ascend/patch/worker/patch_v2/patch_eagle_speculator.py) | 对上游 草稿生成、Eagle 投机 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm.v1.worker.gpu.spec_decode.autoregressive`, `vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph` |
| [patch_input_batch.py](../../../vllm_ascend/patch/worker/patch_v2/patch_input_batch.py) | 对上游 批次、输入 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm.v1.worker.gpu`, `vllm_ascend.worker.v2.input_batch` |
| [patch_model_runner.py](../../../vllm_ascend/patch/worker/patch_v2/patch_model_runner.py) | 对上游 模型执行、模型 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`dispatch_cg_and_sync_dp`, `initialize_kv_cache` |
| [patch_model_state.py](../../../vllm_ascend/patch/worker/patch_v2/patch_model_state.py) | 对上游 模型、状态 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm.v1.worker.gpu`, `vllm_ascend.worker.v2.model_states` |
| [patch_spec_decode_310.py](../../../vllm_ascend/patch/worker/patch_v2/patch_spec_decode_310.py) | 对上游 投机解码 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：310P MRv2 patches: CPU spec-decode helpers + MTP routing (no mainline edits).；定义：`_embed_input_ids`, `_init_speculator_310p` |
| [patch_spec_pp.py](../../../vllm_ascend/patch/worker/patch_v2/patch_spec_pp.py) | 对上游 spec_pp 行为安装适配；实际替换目标见符号、导入和赋值。 | 注释：0.30 PP sampled-token protocol for release trains.；定义：`SpecPPPendingRecv`, `compute_need_sampled_mask`, `install_upstream_spec_pp_protocol` |
| [patch_triton.py](../../../vllm_ascend/patch/worker/patch_v2/patch_triton.py) | 对上游 triton 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm.v1.worker.gpu.spec_decode.speculator`, `vllm.v1.sample.ops` |
| [patch_use_v2_model_runner.py](../../../vllm_ascend/patch/worker/patch_v2/patch_use_v2_model_runner.py) | 对上游 模型执行、模型 行为安装适配；实际替换目标见符号、导入和赋值。 | 导入：`vllm_ascend.patch.platform.patch_use_v2_model_runner` |
| [patch_uva.py](../../../vllm_ascend/patch/worker/patch_v2/patch_uva.py) | 对上游 uva 行为安装适配；实际替换目标见符号、导入和赋值。 | 定义：`MonitoredNumPyArray`, `MonitoredTorchTensor`, `check_triton_ascend_version_valid`, `is_uva_available`, `get_row_indices_from_key` |
