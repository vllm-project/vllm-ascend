# tests/ut/distributed：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **50 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/distributed/`

验证 distributed 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [ascend_store/](../../../tests/ut/distributed/ascend_store) | 面向 ascend_store 的算子型号/架构配置或构建选择。 |
| [device_communicators/](../../../tests/ut/distributed/device_communicators) | 验证 distributed/device_communicators 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [eplb/](../../../tests/ut/distributed/eplb) | 验证 distributed/eplb 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [kv_transfer/](../../../tests/ut/distributed/kv_transfer) | 验证 distributed/kv_transfer 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [mooncake/](../../../tests/ut/distributed/mooncake) | 验证 distributed/mooncake 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [weight_transfer/](../../../tests/ut/distributed/weight_transfer) | 验证 distributed/weight_transfer 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/distributed/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_ec_memcache_connector.py](../../../tests/ut/distributed/test_ec_memcache_connector.py) | 验证 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_recv_buffer_tokens`, `test_get_recv_buffer_tokens_rejects_non_integer_values`, `test_get_recv_buffer_tokens_rejects_out_of_range_values`, `test_get_recv_buffer_tokens_requires_ec_transfer_config` |
| [test_kvpp.py](../../../tests/ut/distributed/test_kvpp.py) | 验证 分层 KV 放置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_full_layer_broadcast_completes_before_future` |
| [test_parallel_state.py](../../../tests/ut/distributed/test_parallel_state.py) | 验证 并行、状态 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_init_ascend_model_parallel`, `test_get_global_rank`, `test_get_global_rank_defaults_to_current_config`, `test_kvpp_group_stays_inside_pipeline_stage` |
| [test_yuanrong_backend.py](../../../tests/ut/distributed/test_yuanrong_backend.py) | 验证 yuanrong backend 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_yuanrong_put_does_not_require_exists_check`, `test_get_and_put_use_multi_buffer_apis`, `test_get_forwards_empty_keys_to_sdk`, `test_get_marks_failed_keys` |

## `tests/ut/distributed/ascend_store/`

面向 ascend_store 的算子型号/架构配置或构建选择。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/distributed/ascend_store/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [_mock_deps.py](../../../tests/ut/distributed/ascend_store/_mock_deps.py) | mock deps 的测试辅助、测试入口或共享数据结构。 | 注释：Mock heavy dependencies (torch, vllm, etc.) for ascend_store unit tests.；定义：`_MockKVConnectorStats`, `_MockKVConnectorPromMetrics` |
| [test_ascend_store_connector.py](../../../tests/ut/distributed/ascend_store/test_ascend_store_connector.py) | 验证 存储 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_event_lifecycle`, `test_worker_aggregation`, `test_bind_metadata_prepares_only_layerwise_workers`, `test_memcache_barrier_config_reaches_backend` |
| [test_backend.py](../../../tests/ut/distributed/ascend_store/test_backend.py) | 验证 backend 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_cannot_instantiate`, `test_memcache_scheduler_factory_does_not_create_npu_context`, `test_scheduler_device_id_does_not_bind_assigned_device`, `test_scheduler_device` |
| [test_coordinator.py](../../../tests/ut/distributed/ascend_store/test_coordinator.py) | 验证 coordinator 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_compressed_group_hits_on_effective_granularity`, `test_compressed_spec_does_not_apply_ratio_twice`, `test_missing_required_group_returns_zero`, `test_store_mask_uses_manager_reachability` |
| [test_dspark_prefix_cache.py](../../../tests/ut/distributed/ascend_store/test_dspark_prefix_cache.py) | 验证 DSpark 草稿、缓存 的测试；具体断言场景由右栏测试符号定位。 | 注释：Draft prefix pages keep independent storage and exact publication contracts.；定义：`test_draft_prefix_roundtrip_remaps_independent_group_blocks_and_keeps_partial_tail_out`, `test_complete_draft_prefix_is_not_overwritten_by_a_second_request`, `test_readable_prefix_evicted_after_probe_does_not_fail_save_or_reallocate`, `test_failed_draft_copy_is_not_published_and_releases_only_owned_write_lease` |
| [test_glm53_store.py](../../../tests/ut/distributed/ascend_store/test_glm53_store.py) | 验证 存储 的测试；具体断言场景由右栏测试符号定位。 | 注释：GLM-5.3-Flash pool lifecycle with real cache specs and CPU payloads.；定义：`test_hash_geometry_and_safe_full_hit`, `test_nonlayerwise_round_trip_keeps_each_tp_state_and_private_tail`, `test_memcache_layerwise_round_trip_keeps_each_tp_state`, `test_mooncake_layerwise_still_rejects_hybrid` |
| [test_kv_transfer.py](../../../tests/ut/distributed/ascend_store/test_kv_transfer.py) | 验证 传输、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_uses_real_offsets_for_variable_cache_entries_per_layer`, `test_queue_lifecycle`, `test_lookup`, `test_get_missing_indices_skips_lookup_when_not_required` |
| [test_layerwise_cache_layout.py](../../../tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py) | 验证 内存布局、缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_no_reuse_skips_topology_validation`, `test_base_layers_are_merged_into_shared_slots`, `test_default_layout_keeps_one_buffer_per_layer`, `test_reuse_layout_matches_round_robin_buffer_assignments` |
| [test_layerwise_keys.py](../../../tests/ut/distributed/ascend_store/test_layerwise_keys.py) | 验证 layerwise keys 的测试；具体断言场景由右栏测试符号定位。 | 注释：Lock wire formats across the bound scheduler/worker key interface.；定义：`test_wire_compatibility_and_all_stage_writer_keys`, `test_binding_snapshots_identity_and_does_not_rehash_on_lookup`, `test_mooncake_rejects_empty_pp_groups_at_binding`, `test_mooncake_distinguishes_layouts_and_groups` |
| [test_metadata.py](../../../tests/ut/distributed/ascend_store/test_metadata.py) | 验证 元数据 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_uses_hybrid_kv_cache`, `test_uses_hybrid_kv_cache_disabled`, `test_infer_group_block_sizes`, `test_get_group_cache_family` |
| [test_metrics.py](../../../tests/ut/distributed/ascend_store/test_metrics.py) | 验证 指标 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_stats_aggregate_and_reduce`, `test_prom_metrics_observe` |
| [test_mooncake_hybrid.py](../../../tests/ut/distributed/ascend_store/test_mooncake_hybrid.py) | 验证 Mooncake 传输/存储 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_multigroup_roundtrip_sparse_masks_and_group_commit_boundaries`, `test_tracker_keeps_equal_block_indices_in_different_groups`, `test_layout_fingerprint_covers_group_membership_and_page_size`, `test_layout_fingerprint_normalizes_uniform_wrapper` |
| [test_mooncake_layerwise.py](../../../tests/ut/distributed/ascend_store/test_mooncake_layerwise.py) | 验证 Mooncake 传输/存储 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_range_debug_payload_reports_per_key_bytes_and_offsets`, `test_key_major_ranges_use_full_object_offsets`, `test_range_limits_split_rows_and_large_segments`, `test_final_layer_commits_after_all_ranges` |
| [test_mooncake_pipeline.py](../../../tests/ut/distributed/ascend_store/test_mooncake_pipeline.py) | 验证 Mooncake 传输/存储 的测试；具体断言场景由右栏测试符号定位。 | 注释：Exercise actual worker sessions/range I/O with distinct bytes on every stage.；定义：`test_uneven_stages_roundtrip_all_heads_and_groups`, `test_mla_replica_writes_only_once_per_stage`, `test_scheduler_block_size_rewrite_preserves_remote_hits`, `test_cache_hash_uses_resolved_block_geometry_without_mutation` |
| [test_pool_scheduler.py](../../../tests/ut/distributed/ascend_store/test_pool_scheduler.py) | 验证 调度、池管理 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_joint_prefix_keys_keep_target_keys_and_require_every_final_pp_draft_replica`, `test_joint_prefix_draft_only_group_uses_companion_keys_without_losing_mixed_target_keys`, `test_joint_prefix_lookup_rechecks_after_save_and_eviction_instead_of_using_hit_ttl`, `test_joint_prefix_group_ownership_survives_scheduler_conversion_and_empty_pp_layer_names` |
| [test_pool_worker.py](../../../tests/ut/distributed/ascend_store/test_pool_worker.py) | 验证 设备执行、池管理 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dspark_draft_pool_exclusion_requires_layerwise_reuse`, `test_replica_keys_geometry_and_thread_ownership`, `test_single_group_recurrent_layer_keeps_post_compute_save`, `test_pp_maps_global_mamba_layer_to_local_post_compute_save` |
| [test_request_lifecycle.py](../../../tests/ut/distributed/ascend_store/test_request_lifecycle.py) | 验证 请求 的测试；具体断言场景由右栏测试符号定位。 | 注释：CPU request lifecycle; only device operations and the external store are fake.；定义：`test_private_only_layout_rejected`, `test_raw_sequence_lifecycle` |

## `tests/ut/distributed/device_communicators/`

验证 distributed/device_communicators 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/distributed/device_communicators/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_pyhccl.py](../../../tests/ut/distributed/device_communicators/test_pyhccl.py) | 验证 HCCL 通信 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_world_size_1_return_early`, `test_load_hccl_fail`, `test_stateless_group`, `test_multi_gpu_pg_torch` |
| [test_pyhccl_wrapper.py](../../../tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py) | 验证 HCCL 通信 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_construct`, `test_torch_dtype_mapping`, `test_unsupported_dtype_raises`, `test_torch_reduce_op_mapping` |

## `tests/ut/distributed/eplb/`

验证 distributed/eplb 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [policy/](../../../tests/ut/distributed/eplb/policy) | 验证 distributed/eplb/policy 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/distributed/eplb/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_communicator.py](../../../tests/ut/distributed/eplb/test_communicator.py) | 验证 通信器 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_communicator_reuses_upstream_gloo_staging`, `test_execute_uses_group_local_peer_ranks`, `test_pinned_staging_buffers_are_reused_between_transfers`, `test_execute_uses_current_device_stream_when_stream_is_unset` |
| [test_explicit_transfer.py](../../../tests/ut/distributed/eplb/test_explicit_transfer.py) | 验证 传输 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_transfer_uses_exact_planned_sources`, `test_receiver_initiated_transfer_only_stages_local_destinations`, `test_transfer_rejects_false_source_before_communication`, `test_transfer_rejects_buffer_schema_before_side_effects` |
| [test_hixl_communicator.py](../../../tests/ut/distributed/eplb/test_hixl_communicator.py) | 验证 通信器 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_hixl_reads_registered_remote_expert`, `test_hixl_rejects_interleaved_expert_rows`, `test_hixl_destructor_cleans_up_when_modules_are_unavailable` |
| [test_hixl_compat.py](../../../tests/ut/distributed/eplb/test_hixl_compat.py) | 验证 hixl compat 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_compat_layout_matches_probed_cann_abi`, `test_compat_empty_options_map_self_points_at_header`, `test_resolve_prefers_the_official_hixl_package`, `test_resolve_falls_back_to_hixl_compat` |
| [test_policy_load_stats.py](../../../tests/ut/distributed/eplb/test_policy_load_stats.py) | 验证 加载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_step_keeps_skipped_samples_on_shared_time_axis`, `test_mapping_change_discards_old_samples`, `test_initialize_load_stats_adds_empty_commit_anchors`, `test_collect_global_load_stats_maps_physical_to_logical` |
| [test_state.py](../../../tests/ut/distributed/eplb/test_state.py) | 验证 状态 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_uses_upstream_policy_and_async_worker_lifecycle`, `test_result_readiness_defers_incomplete_transfer`, `test_configured_upstream_policy_registration_is_scoped`, `test_drain_async_accepts_last_changed_layer_before_model_end` |

## `tests/ut/distributed/eplb/policy/`

验证 distributed/eplb/policy 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/distributed/eplb/policy/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_stair.py](../../../tests/ut/distributed/eplb/policy/test_stair.py) | 验证 stair 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_policy_implements_upstream_abstract_contract`, `test_upstream_contract_flattens_complete_plan`, `test_rebalance_forwards_prepared_stats`, `test_compression_preserves_all_steps_as_weighted_bins` |
| [test_stair_process.py](../../../tests/ut/distributed/eplb/policy/test_stair_process.py) | 验证 stair process 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_round_trip` |

## `tests/ut/distributed/kv_transfer/`

验证 distributed/kv_transfer 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [kv_p2p/](../../../tests/ut/distributed/kv_transfer/kv_p2p) | 验证 distributed/kv_transfer/kv_p2p 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [sparse_kv_offload/](../../../tests/ut/distributed/kv_transfer/sparse_kv_offload) | 验证 distributed/kv_transfer/sparse_kv_offload 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_kv_transfer_failures.py](../../../tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py) | 验证 传输、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for KV transfer failure handling in ascend_store.；定义：`test_all_blocks_succeed`, `test_all_blocks_fail`, `test_partial_blocks_fail`, `test_empty_lists` |
| [test_utils.py](../../../tests/ut/distributed/kv_transfer/test_utils.py) | 验证 辅助函数 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_pd_only`, `test_preserves_other_fields`, `test_explicit_pd_qos`, `test_invalid_inputs_leave_environment_unchanged` |

## `tests/ut/distributed/kv_transfer/kv_p2p/`

验证 distributed/kv_transfer/kv_p2p 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_mooncake_connector.py](../../../tests/ut/distributed/kv_transfer/kv_p2p/test_mooncake_connector.py) | 验证 Mooncake 传输/存储 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_k3_registration_recovers_storage_across_alignment_boundary`, `test_k3_registration_keeps_separate_allocations_separate`, `test_k3_registration_rejects_view_before_aligned_base`, `test_registration_uses_runtime_storage_once_without_descriptors` |
| [test_remote_decode_lifecycle.py](../../../tests/ut/distributed/kv_transfer/kv_p2p/test_remote_decode_lifecycle.py) | 验证 逐步生成 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_basic_lifecycle`, `test_prefix_cache_lifecycle` |
| [test_remote_prefill_lifecycle.py](../../../tests/ut/distributed/kv_transfer/kv_p2p/test_remote_prefill_lifecycle.py) | 验证 上下文预填充 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_basic_lifecycle`, `test_no_spurious_prefix_caching`, `test_full_block_prompt` |

## `tests/ut/distributed/kv_transfer/sparse_kv_offload/`

验证 distributed/kv_transfer/sparse_kv_offload 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_sparse_kv_offload_manager.py](../../../tests/ut/distributed/kv_transfer/sparse_kv_offload/test_sparse_kv_offload_manager.py) | 验证 缓存卸载、管理器、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_non_a3_is_rejected`, `test_memory_plan_is_limited_by_active_workload`, `test_memory_plan_is_limited_by_dram_capacity`, `test_warning_when_dram_budget_caps_npu_utilization` |

## `tests/ut/distributed/mooncake/`

验证 distributed/mooncake 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/distributed/mooncake/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_mooncake_config_data.py](../../../tests/ut/distributed/mooncake/test_mooncake_config_data.py) | 验证 Mooncake 传输/存储、配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_int_input`, `test_gb_unit`, `test_gb_unit_edge_cases`, `test_mb_unit` |
| [test_mooncake_kv_transfer.py](../../../tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py) | 验证 Mooncake 传输/存储、传输、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_sending_thread_only_puts_missing_keys` |

## `tests/ut/distributed/weight_transfer/`

验证 distributed/weight_transfer 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/distributed/weight_transfer/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_e2e_weight_transfer_oracle.py](../../../tests/ut/distributed/weight_transfer/test_e2e_weight_transfer_oracle.py) | 验证 传输、权重 的测试；具体断言场景由右栏测试符号定位。 | 注释：Tests for the weight-transfer E2E oracle's two server configurations.；定义：`test_reference_and_lane_differ_only_in_the_load_path`, `test_reference_and_lane_share_the_measurement_conditions`, `test_reference_signature_starts_one_server_per_measurement_conditions` |
| [test_hccl_engine.py](../../../tests/ut/distributed/weight_transfer/test_hccl_engine.py) | 验证 HCCL 通信 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_start_weight_update_initializes_layerwise_reload`, `test_finish_weight_update_finalizes_layerwise_reload`, `test_shutdown_closes_session_once`, `test_reinit_closes_previous_session_before_creating_next` |
| [test_npu_ipc_engine.py](../../../tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py) | 验证 进程间通信 的测试；具体断言场景由右栏测试符号定位。 | 注释：Regression tests for the NPU IPC weight transfer engine.；定义：`test_init_accepts_model_argument`, `test_init_passes_model_to_super`, `test_unpacked_send_stores_reduce_tensor_args_only`, `test_receive_weights_rebuilds_with_rebuild_npu_tensor` |
| [test_packed_tensor.py](../../../tests/ut/distributed/weight_transfer/test_packed_tensor.py) | 验证 张量、打包 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for packed tensor utilities (HCCL broadcast + NPU IPC).；定义：`test_packed_broadcast_producer_single_tensor`, `test_packed_broadcast_producer_multiple_tensors_one_buffer`, `test_packed_broadcast_producer_splits_when_exceeding_buffer`, `test_packed_broadcast_producer_empty_iterator` |
