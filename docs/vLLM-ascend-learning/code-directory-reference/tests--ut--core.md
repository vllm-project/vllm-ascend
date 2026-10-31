# tests/ut/core：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **14 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/core/`

验证 core 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/core/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_batch_job_aware_scheduler.py](../../../tests/ut/core/test_batch_job_aware_scheduler.py) | 验证 调度、批次 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for batch_job_aware_scheduler.py.；定义：`test_exact_division`, `test_rounds_up`, `test_single_block`, `test_zero_numerator` |
| [test_circular_buffer.py](../../../tests/ut/core/test_circular_buffer.py) | 验证 circular buffer 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ring_lifetime_reuse_and_external_tokens`, `test_uniform_properties_and_single_plane_size`, `test_scratch_groups_do_not_reduce_prefix_hits_or_truncation`, `test_ring_bypasses_position_to_page_mapping` |
| [test_dyntra_lb_recompute_scheduler.py](../../../tests/ut/core/test_dyntra_lb_recompute_scheduler.py) | 验证 调度、重算 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dyntra_lb_recompute_schedulers_use_policy_mixin`, `test_dyntra_lb_recompute_invokes_policy_hooks`, `test_dyntra_lb_recompute_emits_scheduler_diagnostics`, `test_dyntra_lb_recompute_prefetch_waits_for_offload_store` |
| [test_dyntra_lb_scheduler.py](../../../tests/ut/core/test_dyntra_lb_scheduler.py) | 验证 调度 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dyntra_lb_scheduler_uses_policy_mixin`, `test_dyntra_lb_scheduler_diagnostics_are_disabled_by_default`, `test_dyntra_lb_scheduler_diagnostics_can_be_enabled`, `test_dyntra_lb_keeps_async_kv_load_in_skipped_waiting` |
| [test_kv_cache_interface.py](../../../tests/ut/core/test_kv_cache_interface.py) | 验证 缓存、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mixed_mla_indexer_specs_are_not_uniform`, `test_matching_ascend_mla_specs_merge`, `test_get_storage_block_size_and_dcp_memory`, `test_sliding_window_mla_storage_and_page_size` |
| [test_kvpp_cache_placement.py](../../../tests/ut/core/test_kvpp_cache_placement.py) | 验证 缓存、分层 KV 放置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_kvpp_size_covers_pcp_replicas`, `test_pp_local_owners_and_bundles`, `test_component_sizes_and_layout`, `test_unquantized_indexer_and_quantized_mla_sizes` |
| [test_profiling_chunk.py](../../../tests/ut/core/test_profiling_chunk.py) | 验证 性能分析 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_default_values`, `test_need_timing_is_disabled_when_profiling_chunk_is_disabled`, `test_invalid_smooth_factor_raises`, `test_invalid_min_chunk_raises` |
| [test_recompute_scheduler.py](../../../tests/ut/core/test_recompute_scheduler.py) | 验证 调度、重算 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_recompute_scheduler_keeps_local_schedule_for_ascend_spec_padding`, `test_recompute_scheduler_config_picks_sync_and_async_class`, `test_preempt_offloads_before_upstream_releases_blocks`, `test_preempt_without_offload_hook_sends_request_back_to_p` |
| [test_recompute_scheduler_short_request_first.py](../../../tests/ut/core/test_recompute_scheduler_short_request_first.py) | 验证 调度、重算、请求 的测试；具体断言场景由右栏测试符号定位。 | 注释：Verify that RecomputeScheduler does not enable ShortRequestFirst.；定义：`test_waiting_queue_uses_upstream_policy` |
| [test_scheduler_connector_block_state.py](../../../tests/ut/core/test_scheduler_connector_block_state.py) | 验证 调度、缓存页/分块、状态 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_boundary_state_is_drained_consumed_and_not_dispatched` |
| [test_score_encoder_cache_manager.py](../../../tests/ut/core/test_score_encoder_cache_manager.py) | 验证 编码器、管理器、缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qualified_class_name_resolves_score_manager`, `test_other_managers_do_not_enable_score_cache`, `test_factory_reads_score_parameters_from_vllm_config`, `test_score_config_accepts_valid_boundary_values` |
| [test_short_request_first_scheduler.py](../../../tests/ut/core/test_short_request_first_scheduler.py) | 验证 调度、请求 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for the ShortRequestFirst waiting queue.；定义：`test_classify_short_and_long_by_threshold`, `test_immediate_predicate_routes_to_immediate_queue`, `test_dispatch_priority_immediate_then_short_then_long`, `test_short_runs_before_long_without_aging` |
| [test_short_request_first_scheduler_install.py](../../../tests/ut/core/test_short_request_first_scheduler_install.py) | 验证 调度、请求 的测试；具体断言场景由右栏测试符号定位。 | 注释：Tests for installing ShortRequestFirst on a standard scheduler.；定义：`test_installs_queue_on_empty_fcfs_scheduler`, `test_install_is_idempotent`, `test_rejects_non_fcfs_scheduler`, `test_rejects_installation_after_request_admission` |
