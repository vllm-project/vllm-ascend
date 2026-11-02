# tests/ut/patch：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **35 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/patch/`

验证 patch 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [platform/](../../../tests/ut/patch/platform) | 验证 patch/platform 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [worker/](../../../tests/ut/patch/worker) | 验证 patch/worker 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/patch/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `tests/ut/patch/platform/`

验证 patch/platform 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/patch/platform/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_deepseek_v4_thinking.py](../../../tests/ut/patch/platform/test_deepseek_v4_thinking.py) | 验证 DeepSeek 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_deepseek_v4_reasoning_effort_accepts_latest_values`, `test_reasoning_effort_enables_thinking_unless_user_overrides`, `test_deepseek_v4_tokenizer_maps_latest_reasoning_effort_values` |
| [test_initial_expert_layout.py](../../../tests/ut/patch/platform/test_initial_expert_layout.py) | 验证 专家、内存布局 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_initial_replicas_are_spread_across_ep_ranks`, `test_initial_layout_preserves_v1_and_zero_redundancy`, `test_weight_mapping_ep_size_is_scoped_to_the_call` |
| [test_patch_balance_schedule.py](../../../tests/ut/patch/platform/test_patch_balance_schedule.py) | 验证 调度 的测试；具体断言场景由右栏测试符号定位。 | 注释：Upstream-drift guards for the balance-scheduling platform patch.；定义：`test_balance_config_uses_initialized_scheduler_config`, `test_balance_config_fallback_prefers_nested_config`, `test_balance_config_fallback_ignores_non_dict_nested_config`, `test_balance_config_fallback_accepts_legacy_top_level_config` |
| [test_patch_dyntra_lb_core.py](../../../tests/ut/patch/platform/test_patch_dyntra_lb_core.py) | 验证 patch dyntra lb core 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dyntra_lb_core_uses_native_request_status`, `test_dyntra_lb_enabled_reads_nested_scheduler_config`, `test_dyntra_lb_diagnostics_read_nested_scheduler_config`, `test_dyntra_lb_enabled_ignores_top_level_config` |
| [test_patch_engine_core.py](../../../tests/ut/patch/platform/test_patch_engine_core.py) | 验证 patch engine core 的测试；具体断言场景由右栏测试符号定位。 | 注释：Upstream-drift guards for the consolidated engine-core platform patch.；定义：`test_engine_core_patch_applied_at_import`, `test_original_run_engine_core_stashes_pristine_upstream`, `test_apply_patch_installs_pp_mtp_post_step_patch`, `test_apply_patch_is_idempotent` |
| [test_patch_engram_config.py](../../../tests/ut/patch/platform/test_patch_engram_config.py) | 验证 Engram 记忆、配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_engram_patch_is_noop_without_upstream_config`, `test_native_engram_resolution_on_npu`, `test_ascend_engram_limits`, `test_engram_patch_retains_layer_check` |
| [test_patch_eplb.py](../../../tests/ut/patch/platform/test_patch_eplb.py) | 验证 专家负载均衡 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_parallel_and_vllm_config_keep_upstream_validation`, `test_eplb_policy_config_supports_stair_and_default`, `test_eplb_communicator_config_supports_hixl`, `test_resolve_ascend_communicator_selects_hixl_when_group_agrees` |
| [test_patch_fused_moe.py](../../../tests/ut/patch/platform/test_patch_fused_moe.py) | 验证 混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_factory_adapts_only_the_returned_router`, `test_factory_keeps_v1_eplb_on_the_legacy_routing_path`, `test_adapted_router_uses_ascend_mapping_operation`, `test_factory_shares_upstream_hash_table_with_legacy_ascend_routing` |
| [test_patch_glm53_reasoning.py](../../../tests/ut/patch/platform/test_patch_glm53_reasoning.py) | 验证 patch glm53 reasoning 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_normalizes_disabled_thinking_without_mutating_request`, `test_constructor_arguments_are_forwarded`, `test_normal_requests_are_unchanged`, `test_other_templates_keep_their_switch` |
| [test_patch_kv_cache_coordinator.py](../../../tests/ut/patch/platform/test_patch_kv_cache_coordinator.py) | 验证 缓存、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 注释：Regression tests for the hybrid coordinator MTP + prefix-cache fix.；定义：`test_eagle_fallback_flags_only_full_attention_group`, `test_no_eagle_group_when_speculative_decoding_disabled`, `test_explicit_eagle_group_marker_takes_precedence`, `test_coordinator_reads_producer_tag` |
| [test_patch_lopt_tokenization.py](../../../tests/ut/patch/platform/test_patch_lopt_tokenization.py) | 验证 分词 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_attach_lopt_to_deepseek_v4_compatible_renderer`, `test_attach_lopt_skips_unsupported_renderer`, `test_deepseek_v4_sync_tokenization_uses_lopt`, `test_deepseek_v4_async_tokenization_uses_lopt` |
| [test_patch_mamba_block_aligned_split.py](../../../tests/ut/patch/platform/test_patch_mamba_block_aligned_split.py) | 验证 缓存页/分块、Mamba 状态空间 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_pd_consumer_preserves_complete_speculative_window`, `test_producer_retains_upstream_mamba_boundary_split`, `test_non_pd_request_retains_upstream_mamba_boundary_split`, `test_pd_consumer_preserves_window_after_external_cache_hit` |
| [test_patch_mamba_config.py](../../../tests/ut/patch/platform/test_patch_mamba_config.py) | 验证 配置、Mamba 状态空间 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_sparse_index_kpool_detection_is_model_agnostic`, `test_dense_model_with_index_kpool_field_uses_generic_layout`, `test_sparse_indexer_without_kpool_uses_generic_layout`, `test_active_sparse_index_kpool_requires_valid_ratio` |
| [test_patch_pp_mtp.py](../../../tests/ut/patch/platform/test_patch_pp_mtp.py) | 验证 多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_model_config_validates_local_drafter_as_single_pp_rank`, `test_model_config_keeps_target_model_pp_validation`, `test_pp_ipc_runtime_patch_enabled_for_all_v1_pp`, `test_pp_ipc_runtime_patch_skips_pd_prefill_node` |
| [test_patch_speculative_config_dflash.py](../../../tests/ut/patch/platform/test_patch_speculative_config_dflash.py) | 验证 DFlash 草稿、配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_kimi_dflash_hf_override_restores_legacy_yarn_amplitude`, `test_kimi_dflash_accepts_both_config_representations`, `test_kimi_dflash_preserves_explicit_attention_factor`, `test_kimi_dflash_none_attention_factor_uses_legacy_default` |
| [test_patch_speculative_config_dspark.py](../../../tests/ut/patch/platform/test_patch_speculative_config_dspark.py) | 验证 DSpark 草稿、配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_legacy_qwen3_dspark_config_uses_qwen3_loader`, `test_deepseek_v4_vision_dspark_restores_draft_architecture`, `test_k3_dspark_dcp_is_hidden_only_during_upstream_validation`, `test_k3_dspark_dcp_restores_parallel_config_when_validation_fails` |
| [test_patch_structured_output.py](../../../tests/ut/patch/platform/test_patch_structured_output.py) | 验证 输出 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_sampling_params_rejects_mixed_structured_output_backends`, `test_sampling_params_allows_consistent_guidance_backend`, `test_failed_first_validation_does_not_lock_config`, `test_manager_rejects_mixed_structured_output_backends` |
| [test_patch_use_v2_model_runner.py](../../../tests/ut/patch/platform/test_patch_use_v2_model_runner.py) | 验证 模型执行、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ascend_v1_supported_features_are_not_rejected`, `test_release_pcp_is_not_rejected_as_v2_unsupported_feature` |
| [test_prefix_cache_cp_patches.py](../../../tests/ut/patch/platform/test_prefix_cache_cp_patches.py) | 验证 缓存、上下文并行 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_packed_cache_reuses_global_capacity_and_records_recycled_state_pages`, `test_disabled_prefix_cache_preserves_private_tail_allocation`, `test_real_tail_coordinator_preserves_prefix_hit_and_private_lifecycle`, `test_partial_hash_alignment_reaches_real_managers_and_cached_blocks` |

## `tests/ut/patch/worker/`

验证 patch/worker 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [patch_common/](../../../tests/ut/patch/worker/patch_common) | 验证 patch/worker/patch_common 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/patch/worker/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_patch_deepseek_v2.py](../../../tests/ut/patch/worker/test_patch_deepseek_v2.py) | 验证 DeepSeek 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mla_constructor_preserves_decode_mask_flag`, `test_glm51_skip_topk_keeps_per_layer_indexer`, `test_glm52_shared_layer_skips_indexer_init`, `test_mtp_layer_keeps_indexer` |
| [test_patch_dspark_pp.py](../../../tests/ut/patch/worker/test_patch_dspark_pp.py) | 验证 DSpark 草稿 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dspark_draft_partition_isolation` |
| [test_patch_eagle3_pp_aux.py](../../../tests/ut/patch/worker/test_patch_eagle3_pp_aux.py) | 验证 Eagle 投机 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_extract_aux_from_intermediate_sorts_by_aux_index`, `test_non_last_pp_rank_carries_previous_and_local_aux_states`, `test_last_pp_rank_returns_complete_aux_states`, `test_eagle_mixin_non_last_pp_rank_carries_previous_and_local_aux_states` |
| [test_patch_kimi_k25.py](../../../tests/ut/patch/worker/test_patch_kimi_k25.py) | 验证 Kimi 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_position_embedding_patch_uses_current_vllm_contract`, `test_a5_moonvit_to_patch_uses_current_vllm_contract` |
| [test_patch_mamba_utils.py](../../../tests/ut/patch/worker/test_patch_mamba_utils.py) | 验证 Mamba 状态空间、辅助函数 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_preprocess_stages_metadata_but_defers_state_copy`, `test_load_only_step_does_not_hide_remote_state_copy_on_next_forward`, `test_layerwise_mamba_copy_is_grouped_by_layer` |
| [test_patch_mamba_utils_uniform_groups.py](../../../tests/ut/patch/worker/test_patch_mamba_utils_uniform_groups.py) | 验证 Mamba 状态空间、辅助函数 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_uniform_mamba_groups_are_visible_to_all_mamba_buffers` |
| [test_patch_qwen3_5_mtp.py](../../../tests/ut/patch/worker/test_patch_qwen3_5_mtp.py) | 验证 Qwen 模型、多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_standard_gdn_constructor_registers_packed_weight_after_base_init`, `test_qwen3_5_text_attention_uses_standard_rope`, `test_qwen3_5_multimodal_attention_uses_mrope`, `test_qwen3_5_mtp_forward_uses_local_inputs_on_last_pp_rank` |
| [test_patch_routed_experts_capture.py](../../../tests/ut/patch/worker/test_patch_routed_experts_capture.py) | 验证 专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_single_dp`, `test_multi_dp_naive_dispatch`, `test_multi_dp_modular_kernel`, `test_multi_dp_padded_all_gather` |
| [test_patch_spec_pp.py](../../../tests/ut/patch/worker/test_patch_spec_pp.py) | 验证 patch spec pp 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_matches_release_gate_for_regular_decode`, `test_ignores_rank_local_max_seq_len_bound`, `test_all_prefill_chunks_only_returns_none`, `test_installs_and_is_idempotent` |
| [test_patch_v2_gumbel.py](../../../tests/ut/patch/worker/test_patch_v2_gumbel.py) | 验证 patch v2 gumbel 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_gumbel_patch_rebinds_preimported_consumers`, `test_dspark_sample_logits_dispatch` |

## `tests/ut/patch/worker/patch_common/`

验证 patch/worker/patch_common 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/patch/worker/patch_common/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_hccl_pg_registry.py](../../../tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py) | 验证 注册表、HCCL 通信 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_make_hccl_pg_key_respects_rank_order_and_reuse_domain`, `test_make_hccl_pg_key_mapping_hccl_config_affects_distinct_keys`, `test_make_hccl_pg_key_accepts_realistic_options_object_defaults`, `test_make_hccl_pg_key_accepts_matching_global_ranks_in_group` |
| [test_patch_distributed.py](../../../tests/ut/patch/worker/patch_common/test_patch_distributed.py) | 验证 patch distributed 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_group_coordinator_is_patched`, `test_group_coordinator_preserves_backend_for_inherited_helpers`, `test_same_hccl_group_reuses_device_pg_once`, `test_same_hccl_group_reuses_with_realistic_options_object` |
