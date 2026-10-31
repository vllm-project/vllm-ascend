# tests/ut/model_loader：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **19 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/model_loader/`

验证 model_loader 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [netloader/](../../../tests/ut/model_loader/netloader) | 验证 model_loader/netloader 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [rfork/](../../../tests/ut/model_loader/rfork) | 验证 model_loader/rfork 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/model_loader/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `tests/ut/model_loader/netloader/`

验证 model_loader/netloader 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/model_loader/netloader/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_netloader.py](../../../tests/ut/model_loader/netloader/test_netloader.py) | 验证 网络权重加载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_init_with_extra_config_file`, `test_init_with_extra_config`, `test_init_with_invalid_config`, `test_fallback_cleanup_context_releases_references_and_restores_state` |
| [test_netloader_elastic.py](../../../tests/ut/model_loader/netloader/test_netloader_elastic.py) | 验证 网络权重加载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_elastic_client_register`, `test_elastic_client_register_failure`, `test_server_initialization`, `test_int8_cache_handling` |
| [test_netloader_load.py](../../../tests/ut/model_loader/netloader/test_netloader_load.py) | 验证 网络权重加载、加载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_sources_this_device_empty`, `test_client_missing_socket_or_ack_returns_none`, `test_model_load_fail`, `test_elastic_load_wires_processed_layout_by_int8_cache` |
| [test_netloader_utils.py](../../../tests/ut/model_loader/netloader/test_netloader_utils.py) | 验证 网络权重加载、辅助函数 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_find_free_port`, `test_is_valid_path_prefix_empty`, `test_is_valid_path_prefixIllegal_characters`, `test_is_valid_path_prefixRelative_path` |

## `tests/ut/model_loader/rfork/`

验证 model_loader/rfork 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/model_loader/rfork/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [conftest.py](../../../tests/ut/model_loader/rfork/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 定义：`runtime`, `tensor_runtime` |
| [rfork_test_support.py](../../../tests/ut/model_loader/rfork/rfork_test_support.py) | RFork 权重加载 的测试辅助、测试入口或共享数据结构。 | 定义：`_load_module`, `_stub` |
| [session_test_utils.py](../../../tests/ut/model_loader/rfork/session_test_utils.py) | 辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`dummy_model`, `make_session`, `run_and_join` |
| [test_identity_validation.py](../../../tests/ut/model_loader/rfork/test_identity_validation.py) | 验证 identity validation 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_canonicalization_is_deterministic_for_dicts_sets_and_dtype`, `test_fingerprint_changes_when_seed_compatibility_changes`, `test_structural_only_config_fields_do_not_partition_weight_seeds`, `test_ascend_layout_switches_do_not_partition_weight_seeds` |
| [test_lease_release.py](../../../tests/ut/model_loader/rfork/test_lease_release.py) | 验证 lease release 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_session_passes_tp_rank_to_transfer_backend`, `test_release_classifies_response_without_changing_wire_protocol`, `test_seed_report_classifies_response_without_changing_wire_protocol`, `test_acquire_seed_does_not_swallow_programming_errors` |
| [test_registration_cache_isolated.py](../../../tests/ut/model_loader/rfork/test_registration_cache_isolated.py) | 验证 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_tensor_collection_deduplicates_exact_impl_alias_but_keeps_distinct_view`, `test_collector_excludes_scheduler_sized_topk_indices_buffer`, `test_layout_summary_is_one_bounded_info_record_with_fixed_digests`, `test_layout_summary_includes_npu_format_and_physical_size` |
| [test_restored_lifecycle.py](../../../tests/ut/model_loader/rfork/test_restored_lifecycle.py) | 验证 存储 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mutable_weights_bypass_reason_reports_sleep_and_weight_transfer`, `test_fallback_reset_restores_ascend_globals_and_only_removes_new_dynamo_hooks` |
| [test_rfork_loader.py](../../../tests/ut/model_loader/rfork/test_rfork_loader.py) | 验证 加载器、RFork 权重加载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_rfork_invalid_seed_timeout_uses_env_then_default`, `test_rfork_environment_values_are_used_as_fallbacks`, `test_rfork_explicit_config_takes_priority_over_environment`, `test_rfork_ep_rank_is_omitted_when_ep_is_inapplicable` |
| [test_rfork_planner.py](../../../tests/ut/model_loader/rfork/test_rfork_planner.py) | 验证 RFork 权重加载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_lease_ttl_default_and_configuration_precedence`, `test_non_positive_lease_ttl_rejected`, `test_non_integer_lease_ttl_rejected`, `test_lease_expiry_boundary_and_capacity_reclaimed_once` |
| [test_session_server_integration.py](../../../tests/ut/model_loader/rfork/test_session_server_integration.py) | 验证 session server integration 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_session_serves_real_metadata_and_closes_listener`, `test_seed_server_port_reserves_main_and_draft_slots`, `test_seed_server_port_zero_keeps_dynamic_allocation`, `test_configured_seed_port_falls_back_when_already_in_use` |
| [test_transfer_backend.py](../../../tests/ut/model_loader/rfork/test_transfer_backend.py) | 验证 传输 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_weight_block_index_handles_nested_ranges_and_touching_endpoints`, `test_registration_requires_full_coverage_across_allocator_blocks`, `test_first_registration_rejects_transfer_engine_without_memory_registration`, `test_initialize_transfer_engine_binds_dynamic_port_and_reads_rpc_port` |
| [test_transfer_timing_isolated.py](../../../tests/ut/model_loader/rfork/test_transfer_timing_isolated.py) | 验证 传输 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_weight_transfer_completion_is_info_only_for_tp0`, `test_failed_transfer_does_not_log_successful_completion` |
