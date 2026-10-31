# tests/ut：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **20 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/`

围绕函数、组件或接口契约的单元/回归测试。

| 子目录 | 职责 |
| --- | --- |
| [_310p/](tests--ut--_310p.md) | 验证 _310p 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [_fake_weight/](tests--ut--_fake_weight.md) | 验证 _fake_weight 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [_tools/](tests--ut--_tools.md) | 验证 _tools 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [attention/](tests--ut--attention.md) | 验证 attention 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [compilation/](tests--ut--compilation.md) | 验证 compilation 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [core/](tests--ut--core.md) | 验证 core 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [device/](tests--ut--device.md) | 验证 device 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [device_allocator/](tests--ut--device_allocator.md) | 验证 device_allocator 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [distributed/](tests--ut--distributed.md) | 验证 distributed 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [eplb/](tests--ut--eplb.md) | 验证 eplb 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [helpers/](tests--ut--helpers.md) | 验证 helpers 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [kv_offload/](tests--ut--kv_offload.md) | 验证 kv_offload 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [kv_transfer/](tests--ut--kv_transfer.md) | 验证 kv_transfer 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [lora/](tests--ut--lora.md) | 验证 lora 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [model_executor/](tests--ut--model_executor.md) | 验证 model_executor 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [model_loader/](tests--ut--model_loader.md) | 验证 model_loader 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [models/](tests--ut--models.md) | 验证 models 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [observability/](tests--ut--observability.md) | 验证 observability 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [ops/](tests--ut--ops.md) | 验证 ops 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [patch/](tests--ut--patch.md) | 验证 patch 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [profiler/](tests--ut--profiler.md) | 验证 profiler 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [proxy/](tests--ut--proxy.md) | 验证 proxy 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [quantization/](tests--ut--quantization.md) | 验证 quantization 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [sample/](tests--ut--sample.md) | 验证 sample 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [simple_kv_offload/](tests--ut--simple_kv_offload.md) | 验证 simple_kv_offload 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [spec_decode/](tests--ut--spec_decode.md) | 验证 spec_decode 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tokenization/](tests--ut--tokenization.md) | 验证 tokenization 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tools/](tests--ut--tools.md) | 验证 tools 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [trace/](tests--ut--trace.md) | 验证 trace 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [worker/](tests--ut--worker.md) | 验证 worker 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [xlite/](tests--ut--xlite.md) | 验证 xlite 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [base.py](../../../tests/ut/base.py) | base 的测试辅助、测试入口或共享数据结构。 | 定义：`TestBase`, `PytestBase` |
| [conftest.py](../../../tests/ut/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 注释：Shared UT setup.；定义：`_clear_enable_sp_before_test`, `_reset_stream_globals_before_test`, `_mock_ascend_store_deps` |
| [kvpp_utils.py](../../../tests/ut/kvpp_utils.py) | 辅助函数、分层 KV 放置 的测试辅助、测试入口或共享数据结构。 | 定义：`ManualExecutor`, `layer_name`, `indexer_name`, `make_kvpp_config` |
| [test_ascend_config.py](../../../tests/ut/test_ascend_config.py) | 验证 配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_config_modules_do_not_load_vllm_config`, `test_defaults_and_explicit_values`, `test_lax_bool_and_unknown_key`, `test_sparse_li_c8_layer_filter_uses_indexer_quant_type` |
| [test_ascend_forward_context.py](../../../tests/ut/test_ascend_forward_context.py) | 验证 ascend forward context 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_deepseek_v4_forward_passes_input_ids_to_layers`, `test_set_mc2_tokens_capacity_without_cudagraph_aligns_per_tp_rank`, `test_set_mc2_tokens_capacity_with_cudagraph_uses_capture_size_and_aligns`, `test_set_mc2_tokens_capacity_prefill_mc2_uses_max_num_batched_tokens` |
| [test_batch_invariant.py](../../../tests/ut/test_batch_invariant.py) | 验证 批次 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_reduce_sum_uses_batch_invariant_operator_for_npu_tensor`, `test_reduce_sum_uses_native_operator_for_cpu_tensor`, `test_reduce_sum_uses_native_operator_for_unsupported_npu_dtype`, `test_override_envs_for_invariance` |
| [test_compressed_prefix_cache.py](../../../tests/ut/test_compressed_prefix_cache.py) | 验证 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ascend_mla_spec_is_not_uniform_with_mamba`, `test_compressed_spec_separates_logical_and_storage_blocks`, `test_compressed_prefix_cache_uses_logical_block_hash`, `test_compressed_prefix_cache_hits_identical_logical_block` |
| [test_cpu_binding.py](../../../tests/ut/test_cpu_binding.py) | 验证 接口绑定 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_cpu_binding_placeholder` |
| [test_dependency_documentation.py](../../../tests/ut/test_dependency_documentation.py) | 验证 dependency documentation 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_stable_documentation_version_is_explicit`, `test_main_dependency_versions_match_repository_metadata`, `test_cpu_only_build_contract_is_documented` |
| [test_envs.py](../../../tests/ut/test_envs.py) | 验证 envs 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_env_vars_behavior`, `test_dir_and_getattr`, `test_kvpool_range_debug_is_strict_and_disabled_by_default` |
| [test_logger.py](../../../tests/ut/test_logger.py) | 验证 logger 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_is_ascend_module_with_ascend_path`, `test_is_ascend_module_with_vllm_path`, `test_infer_module_name_root_file`, `test_infer_module_name_nested_file` |
| [test_meta_registration.py](../../../tests/ut/test_meta_registration.py) | 验证 meta registration 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_meta_registration_placeholder` |
| [test_minimax_m3_kv_block_size.py](../../../tests/ut/test_minimax_m3_kv_block_size.py) | 验证 MiniMax 模型、缓存页/分块、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_m3_fp8_keeps_upstream_alignment_at_128`, `test_other_layouts_keep_kernel_block_sizes`, `test_no_engine_context_keeps_default_backend`, `test_without_skip_layers_keeps_128_for_indexer` |
| [test_platform.py](../../../tests/ut/test_platform.py) | 验证 平台接入 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_validate_v4_index_cache_pp_partition`, `test_v4_index_cache_schedule_uses_indexer_ordinals`, `test_validate_v4_index_cache_pp_pattern`, `test_v4_index_cache_recomputes_first_local_c4` |
| [test_profiling_config.py](../../../tests/ut/test_profiling_config.py) | 验证 性能分析、配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_service_profiling_symbols_yaml`, `test_get_config_dir_uses_home`, `test_generate_service_profiling_config_creates_default`, `test_generate_service_profiling_config_preserves_existing_file` |
| [test_rlhf_server_logging.py](../../../tests/ut/test_rlhf_server_logging.py) | 验证 rlhf server logging 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_server_does_not_block_when_stderr_exceeds_pipe_capacity` |
| [test_sleep_mode_tensor_ownership.py](../../../tests/ut/test_sleep_mode_tensor_ownership.py) | 验证 张量、休眠资源 的测试；具体断言场景由右栏测试符号定位。 | 注释：Level-2 sleep ownership for tensors created inside the ``weights`` mem-pool.；定义：`test_sfa_indexer_hadamard_is_a_non_persistent_buffer`, `test_sfa_indexer_hadamard_survives_level2_sleep`, `test_dcp_remap_constants_are_owned_by_the_attention_layer`, `test_dcp_remap_constants_fall_back_to_the_projection_owner` |
| [test_utils.py](../../../tests/ut/test_utils.py) | 验证 辅助函数 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_register_customop_selects_gdn_before_import`, `test_cp_decode_stream_is_cached_and_separate_from_chunked_prefill`, `test_is_pd_decode_recompute_scheduler_enabled_without_config`, `test_is_pd_decode_recompute_scheduler_enabled_kv_producer` |
| [test_v2_sparse_offload_config.py](../../../tests/ut/test_v2_sparse_offload_config.py) | 验证 缓存卸载、配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_both_runners_accept_sparse_fused_mtp`, `test_decode_pp_remains_rejected`, `test_producer_offload_remains_rejected`, `test_v2_glm_mla_dspark_accepts_nine_row_kernel_budget` |
