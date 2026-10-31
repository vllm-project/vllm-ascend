# tests/ut/_310p：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **33 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/_310p/`

验证 _310p 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [attention/](../../../tests/ut/_310p/attention) | 验证 _310p/attention 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [fused_moe/](../../../tests/ut/_310p/fused_moe) | 验证 _310p/fused_moe 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [ops/](../../../tests/ut/_310p/ops) | 验证 _310p/ops 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [quantization/](../../../tests/ut/_310p/quantization) | 验证 _310p/quantization 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [sample/](../../../tests/ut/_310p/sample) | 验证 _310p/sample 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [spec_decode/](../../../tests/ut/_310p/spec_decode) | 验证 _310p/spec_decode 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/_310p/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_block_table_310p.py](../../../tests/ut/_310p/test_block_table_310p.py) | 验证 缓存页/分块 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_compute_slot_mapping_with_query_start_loc_signature`, `test_multi_group_compute_slot_mapping_accepts_none_compressed_args`, `test_multi_group_compute_slot_mapping_uses_compressed_inputs_per_group`, `test_compute_slot_mapping_with_req_indices_signature` |
| [test_kv_block_zeroer_310p.py](../../../tests/ut/_310p/test_kv_block_zeroer_310p.py) | 验证 缓存页/分块、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_zero_block_ids_noop_when_empty`, `test_zero_block_ids_zeros_target_slices`, `test_init_meta_deduplicates_kv_pointers` |
| [test_kv_cache_sharing.py](../../../tests/ut/_310p/test_kv_cache_sharing.py) | 验证 缓存、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_shares_matching_mamba_slots_across_groups_only`, `test_mtp_attention_layer_does_not_disable_mamba_slot_sharing`, `test_rejects_layout_without_contiguous_per_layer_pages`, `test_rejects_nonuniform_group_sizes_and_mamba_specs` |
| [test_mamba_align_fallback_310p_source.py](../../../tests/ut/_310p/test_mamba_align_fallback_310p_source.py) | 验证 Mamba 状态空间 的测试；具体断言场景由右栏测试符号定位。 | 注释：Source-level regressions for the 310P Mamba align fallback.；定义：`test_310p_postprocess_fallback_preserves_upstream_metadata_semantics`, `test_310p_postprocess_fallback_mirrors_state_copy_without_triton` |
| [test_model_runner_310p.py](../../../tests/ut/_310p/test_model_runner_310p.py) | 验证 模型执行、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_prepare_inputs_keeps_aclgraph_metadata_on_cpu`, `test_model_forward_updates_mtp_full_graph_params_before_replay`, `test_310p_runner_does_not_advertise_standardized_shared_kv_backing`, `test_graph_dispatch_does_not_treat_later_prefill_chunk_as_decode` |
| [test_model_runner_v2_310p.py](../../../tests/ut/_310p/test_model_runner_v2_310p.py) | 验证 模型执行、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_config_accepts_prefix_caching`, `test_config_accepts_tensor_parallelism`, `test_kv_zeroing_uses_narrow_310p_gate`, `test_update_requests_filters_unneeded_upstream_zeroing` |
| [test_sharded_state_loader_310p.py](../../../tests/ut/_310p/test_sharded_state_loader_310p.py) | 验证 加载器、状态 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_save_model_with_nd_format_310`, `test_generate_quant_description_float_model_310`, `test_generate_quant_description_no_quant_config_310`, `test_generate_quant_description_int_model_310` |

## `tests/ut/_310p/attention/`

验证 _310p/attention 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/_310p/attention/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_attention_mask_310.py](../../../tests/ut/_310p/attention/test_attention_mask_310.py) | 验证 注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_attention_mask_310`, `test_get_splitfuse_attn_mask_310` |
| [test_attention_v1_310.py](../../../tests/ut/_310p/attention/test_attention_v1_310.py) | 验证 注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_impl_cls`, `test_get_builder_cls`, `test_get_kv_cache_shape_not`, `test_forward_prefill_310` |

## `tests/ut/_310p/fused_moe/`

验证 _310p/fused_moe 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/_310p/fused_moe/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_moe_mlp_310.py](../../../tests/ut/_310p/fused_moe/test_moe_mlp_310.py) | 验证 混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_fused_activation_path_dispatches_to_fused_hook`, `test_only_fused_path_is_used_for_supported_activation`, `test_unsupported_activation_is_rejected`, `test_apply_gmm1_act_quant_matches_legacy_gmm1_swiglu` |
| [test_shared_fused_moe_310.py](../../../tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py) | 验证 混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_routed_experts_310_uses_parent_unquantized_method_during_init`, `test_routed_experts_310_replaces_only_unquantized_method_after_parent_init`, `test_runner_310_installs_specialized_comm`, `test_process_weights_after_loading_310_uses_version_specific_layout` |

## `tests/ut/_310p/ops/`

验证 _310p/ops 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/_310p/ops/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_chunk_gated_delta_rule_310.py](../../../tests/ut/_310p/ops/test_chunk_gated_delta_rule_310.py) | 验证 Delta 递归状态、门控 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_chunk_gated_delta_rule_310_output_shape_and_dtype`, `test_chunk_gated_delta_rule_310_varlen_path`, `test_chunk_gated_delta_rule_310_varlen_tnd_path` |
| [test_conv_310.py](../../../tests/ut/_310p/ops/test_conv_310.py) | 验证 卷积 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_conv3d_310_forward_oot_uses_forward_native` |
| [test_gdn_310.py](../../../tests/ut/_310p/ops/test_gdn_310.py) | 验证 Gated DeltaNet 注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ascend_gdn_attention_310_uses_310p_backend`, `test_builder310_reuses_common_graph_materialization`, `test_zero_padded_tokens_masks_only_padded_token_positions`, `test_builder_prefill_shares_state_filtering` |
| [test_layernorm_310.py](../../../tests/ut/_310p/ops/test_layernorm_310.py) | 验证 归一化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_rmsnorm_gated_310_forward_oot_uses_rmsnorm_activation_mul`, `test_rmsnorm_gated_310_forward_oot_uses_rmsnorm_without_gate`, `test_rmsnorm_gated_310_forward_oot_keeps_native_for_group_norm` |
| [test_mm_encoder_attention_310.py](../../../tests/ut/_310p/ops/test_mm_encoder_attention_310.py) | 验证 注意力、编码器 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_register_customop_overrides_mm_encoder_attention_for_310p`, `test_mm_encoder_attention_310_forward_oot_with_padding` |
| [test_qwen3vl_310.py](../../../tests/ut/_310p/ops/test_qwen3vl_310.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 注释：310P Qwen3-VL vision helpers (RC rot_pos_emb only).；定义：`test_rot_pos_emb_310_is_importable`, `test_rot_pos_emb_patch_only_on_rc` |
| [test_rotary_embedding_310.py](../../../tests/ut/_310p/ops/test_rotary_embedding_310.py) | 验证 向量查表、位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_set_mrope_apply_rotary_slices_populates_globals`, `test_set_mrope_apply_rotary_slices_reuses_buffer_address`, `test_ascend_rotary_embedding_310_drafting_flag` |

## `tests/ut/_310p/quantization/`

验证 _310p/quantization 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/_310p/quantization/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_modelslim_config_310.py](../../../tests/ut/_310p/quantization/test_modelslim_config_310.py) | 验证 配置、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_quant_method_for_linear_310`, `test_get_quant_method_maps_lm_head_prefix_310`, `test_get_quant_method_for_fused_moe_310` |
| [test_w8a8_dynamic_310.py](../../../tests/ut/_310p/quantization/test_w8a8_dynamic_310.py) | 验证 w8a8 dynamic 310 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_310`, `test_get_dynamic_quant_param_310`, `test_get_weight_310`, `test_get_perchannel_param_310` |
| [test_w8a8_dynamic_moe_310.py](../../../tests/ut/_310p/quantization/test_w8a8_dynamic_moe_310.py) | 验证 混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_w8a8_dynamic_moe_apply_310_uses_preselected_experts` |
| [test_w8a8_static_310.py](../../../tests/ut/_310p/quantization/test_w8a8_static_310.py) | 验证 w8a8 static 310 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_310`, `test_get_pertensor_param_310`, `test_get_perchannel_param_310`, `test_apply_with_x_not_int8_310` |
| [test_w8a8s_310.py](../../../tests/ut/_310p/quantization/test_w8a8s_310.py) | 验证 w8a8s 310 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_310`, `test_get_pertensor_param_310`, `test_get_perchannel_param_310`, `test_apply_with_x_not_int8_310` |
| [test_w8a8sc_310.py](../../../tests/ut/_310p/quantization/test_w8a8sc_310.py) | 验证 w8a8sc 310 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_310`, `test_get_pertensor_param_310`, `test_get_perchannel_param_310`, `test_apply_with_x_not_int8_310` |

## `tests/ut/_310p/sample/`

验证 _310p/sample 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/_310p/sample/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_sampler_310.py](../../../tests/ut/_310p/sample/test_sampler_310.py) | 验证 采样 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_prepare_cpu_generators_preserves_requests_across_reordering`, `test_prepare_cpu_generators_replaces_changed_source`, `test_sample_from_cdf_handles_zero_weight_prefix_and_boundaries`, `test_fill_exponential_honors_active_mask` |

## `tests/ut/_310p/spec_decode/`

验证 _310p/spec_decode 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_llm_base_proposer_310.py](../../../tests/ut/_310p/spec_decode/test_llm_base_proposer_310.py) | 验证 候选生成 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_run_merged_draft_sets_rope_flag_before_call`, `test_run_merged_draft_restores_rope_flag_after_exception` |
| [test_mtp_mrv2_310.py](../../../tests/ut/_310p/spec_decode/test_mtp_mrv2_310.py) | 验证 多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_greedy_rejection_uses_logit_idx_plus_one`, `test_probabilistic_rejection_accepts_when_u_below_p_draft`, `test_probabilistic_rejection_recovers_when_u_rejects_draft`, `test_prepare_decode_inputs_advances_sample_src_positions` |
