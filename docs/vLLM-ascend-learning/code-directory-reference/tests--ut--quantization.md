# tests/ut/quantization：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **42 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/quantization/`

验证 quantization 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [configs/](../../../tests/ut/quantization/configs) | 验证 quantization/configs 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [methods/](../../../tests/ut/quantization/methods) | 验证 quantization/methods 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/quantization/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [conftest_quantization.py](../../../tests/ut/quantization/conftest_quantization.py) | 量化 的测试辅助、测试入口或共享数据结构。 | 定义：`identity`, `create_mock_vllm_config`, `create_mock_ascend_config` |
| [test_method_adapters.py](../../../tests/ut/quantization/test_method_adapters.py) | 验证 method adapters 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mx_scale_weight_loader_preserves_group_phase`, `test_mx_scale_weight_loader_accepts_pre_sharded_scale`, `test_create_weights`, `test_block_quant_scale_uses_the_upstream_parameter_type` |
| [test_quant_parser.py](../../../tests/ut/quantization/test_quant_parser.py) | 验证 量化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_returns_down_proj_quant_type`, `test_returns_default_when_no_down_proj`, `test_default_values`, `test_w8a8_mxfp8_uses_rint_round_mode` |
| [test_utils.py](../../../tests/ut/quantization/test_utils.py) | 验证 辅助函数 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_uses_one_only_for_minimax_m3_on_a5`, `test_uses_current_vllm_config_when_config_is_omitted`, `test_uses_forward_vllm_config_when_available`, `test_falls_back_to_current_config_when_forward_scale_is_missing` |
| [test_weight_switch.py](../../../tests/ut/quantization/test_weight_switch.py) | 验证 权重 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_weight_switch_config_uses_the_caller_selected_parallel_group`, `test_split_tensor_for_parallel_supports_nonzero_and_negative_dims`, `test_split_tensor_for_parallel_rejects_non_divisible_dimension`, `test_enable_input_sharded_state_builds_gather_and_repeat_tensors` |

## `tests/ut/quantization/configs/`

验证 quantization/configs 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/quantization/configs/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_compressed_tensors_config.py](../../../tests/ut/quantization/configs/test_compressed_tensors_config.py) | 验证 张量、配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_detect_w8a8_static`, `test_detect_w8a8_dynamic`, `test_detect_w4a8_dynamic`, `test_detect_w4a16` |
| [test_fp8_config.py](../../../tests/ut/quantization/configs/test_fp8_config.py) | 验证 配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_quant_method_for_linear_base`, `test_get_quant_method_for_moe_runner_fp4`, `test_get_quant_method_for_routed_experts_fp4`, `test_get_quant_method_for_moe_non_fp4_raises` |
| [test_modelopt_mxfp8_config.py](../../../tests/ut/quantization/configs/test_modelopt_mxfp8_config.py) | 验证 配置、模型、MXFP 数值格式 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_registered_for_both_modelopt_mxfp8_names`, `test_from_minimax_config_reuses_modelopt_parser`, `test_vllm_mapper_updates_excluded_layers`, `test_excluded_embedding_and_lm_head_are_unquantized` |
| [test_modelslim_config.py](../../../tests/ut/quantization/configs/test_modelslim_config.py) | 验证 配置、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_init`, `test_repr`, `test_get_name`, `test_get_supported_act_dtypes` |

## `tests/ut/quantization/methods/`

验证 quantization/methods 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [kv_cache/](../../../tests/ut/quantization/methods/kv_cache) | 验证 quantization/methods/kv_cache 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [w4a4/](../../../tests/ut/quantization/methods/w4a4) | 验证 quantization/methods/w4a4 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [w4a8/](../../../tests/ut/quantization/methods/w4a8) | 验证 quantization/methods/w4a8 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [w8a8/](../../../tests/ut/quantization/methods/w8a8) | 验证 quantization/methods/w8a8 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [wna16/](../../../tests/ut/quantization/methods/wna16) | 验证 quantization/methods/wna16 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/quantization/methods/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_moe_logical_experts.py](../../../tests/ut/quantization/methods/test_moe_logical_experts.py) | 验证 专家、混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_moe_num_logical_experts_uses_vllm_config_field`, `test_get_moe_num_logical_experts_falls_back_for_older_configs`, `test_routed_moe_interface_is_not_exposed_on_linear_schemes`, `test_v2_eplb_supported_quantization` |
| [test_mxfp_c8.py](../../../tests/ut/quantization/methods/test_mxfp_c8.py) | 验证 MXFP 数值格式 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_k_scale_cache_shape_d256_bs512`, `test_v_scale_cache_shape_d256_bs512`, `test_scale_page_bytes_are_equal_for_k_and_v`, `test_head_dim_must_align_to_scale_group` |
| [test_registry.py](../../../tests/ut/quantization/methods/test_registry.py) | 验证 注册表 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_register_scheme`, `test_register_scheme_duplicate_raises`, `test_get_existing_scheme_class_existing_linear`, `test_get_nonexistent_scheme_class` |

## `tests/ut/quantization/methods/kv_cache/`

验证 quantization/methods/kv_cache 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [turboquant/](../../../tests/ut/quantization/methods/kv_cache/turboquant) | 验证 quantization/methods/kv_cache/turboquant 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/quantization/methods/kv_cache/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_kv_c8.py](../../../tests/ut/quantization/methods/kv_cache/test_kv_c8.py) | 验证 KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_weight_loader_single_element`, `test_weight_loader_single_element_int`, `test_weight_loader_tp_sharding_first_rank`, `test_weight_loader_tp_sharding_middle_rank` |

## `tests/ut/quantization/methods/kv_cache/turboquant/`

验证 quantization/methods/kv_cache/turboquant 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [conftest.py](../../../tests/ut/quantization/methods/kv_cache/turboquant/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 导入：`vllm_ascend` |
| [test_capacity.py](../../../tests/ut/quantization/methods/kv_cache/turboquant/test_capacity.py) | 验证 capacity 的测试；具体断言场景由右栏测试符号定位。 | 注释：Exercise the real vLLM concurrency formula with mixed DS cache geometry.；定义：`test_tq_planner_detects_pipeline_rank_without_c4`, `test_tq_planner_ignores_unmaterialized_groups`, `test_c4_grouping_preserves_nonquantized_specs_and_state_padding`, `test_common_stride_uses_largest_group_without_page_bin_fragmentation` |
| [test_config.py](../../../tests/ut/quantization/methods/kv_cache/turboquant/test_config.py) | 验证 配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_valid_runner_v2_configuration`, `test_hf_config_fallback`, `test_missing_hf_config_fails_with_validation_error`, `test_invalid_configuration_fails_early` |
| [test_latent.py](../../../tests/ut/quantization/methods/kv_cache/turboquant/test_latent.py) | 验证 latent 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_rotation_preserves_attention_with_sinks`, `test_packed_layout_norm_correction_and_zero_rows`, `test_transform_storage_is_reused`, `test_transform_sharing_and_inverse_layout` |
| [test_operators.py](../../../tests/ut/quantization/methods/kv_cache/turboquant/test_operators.py) | 验证 operators 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_metadata_maps_paged_lengths_and_quant_mode`, `test_attention_retains_caller_owned_cache_and_sinks`, `test_attention_defaults_to_c4_compression_ratio` |
| [test_packing.py](../../../tests/ut/quantization/methods/kv_cache/turboquant/test_packing.py) | 验证 打包 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_packed_pages_never_alias_other_block_ids`, `test_quantized_group_pool_is_smaller_than_bf16` |

## `tests/ut/quantization/methods/w4a4/`

验证 quantization/methods/w4a4 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/quantization/methods/w4a4/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_w4a4_flatquant.py](../../../tests/ut/quantization/methods/w4a4/test_w4a4_flatquant.py) | 验证 量化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_solve_kronecker_decompose`, `test_pack_int4_weights_npu_success`, `test_large_batch_multiple_calls`, `test_exact_max_batch_size` |
| [test_w4a4_laos_dynamic.py](../../../tests/ut/quantization/methods/w4a4/test_w4a4_laos_dynamic.py) | 验证 w4a4 laos dynamic 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_various_sizes`, `test_get_perchannel_param_various_output_sizes`, `test_apply_with_bias`, `test_process_weights_various_input_sizes` |
| [test_w4a4_mxfp4.py](../../../tests/ut/quantization/methods/w4a4/test_w4a4_mxfp4.py) | 验证 MXFP 数值格式 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_various_input_sizes`, `test_get_weight_compressed_tensors`, `test_rename_packed_weight_parameter`, `test_get_pergroup_param_based_on_group_size` |
| [test_w4a4_mxfp4_flatquant_dynamic.py](../../../tests/ut/quantization/methods/w4a4/test_w4a4_mxfp4_flatquant_dynamic.py) | 验证 量化、MXFP 数值格式 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_perfect_square_decomposition`, `test_non_square_decomposition`, `test_decomposition_product_equals_n`, `test_raises_when_dim_sum_exceeds_max` |

## `tests/ut/quantization/methods/w4a8/`

验证 quantization/methods/w4a8 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/quantization/methods/w4a8/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_w4a8.py](../../../tests/ut/quantization/methods/w4a8/test_w4a8.py) | 验证 w4a8 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_init_raise_error`, `test_get_weight`, `test_get_weight_compressed_tensors`, `test_get_dynamic_quant_param` |
| [test_w4a8_mxfp4.py](../../../tests/ut/quantization/methods/w4a8/test_w4a8_mxfp4.py) | 验证 MXFP 数值格式 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_various_input_sizes`, `test_get_pergroup_param_based_on_group_size`, `test_process_weights_after_loading_transposes`, `test_process_records_original_shapes_and_marks_transformed` |

## `tests/ut/quantization/methods/w8a8/`

验证 quantization/methods/w8a8 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/quantization/methods/w8a8/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_fp8_block.py](../../../tests/ut/quantization/methods/w8a8/test_fp8_block.py) | 验证 缓存页/分块 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_matches_per_element_expansion`, `test_handles_blocks_that_do_not_divide_the_shape`, `test_row_chunking_does_not_change_the_result`, `test_scale_is_applied_in_float32` |
| [test_w8a8_dynamic.py](../../../tests/ut/quantization/methods/w8a8/test_w8a8_dynamic.py) | 验证 w8a8 dynamic 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_various_sizes`, `test_get_perchannel_param_dtype_variations`, `test_apply_3d_input_with_squeeze`, `test_process_weights_after_loading` |
| [test_w8a8_mxfp8.py](../../../tests/ut/quantization/methods/w8a8/test_w8a8_mxfp8.py) | 验证 MXFP 数值格式 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_modelopt_config_defaults_group_size`, `test_get_weight_various_input_sizes`, `test_get_pergroup_param_group_size_variations`, `test_process_weights_stores_original_shapes` |
| [test_w8a8_pdmix.py](../../../tests/ut/quantization/methods/w8a8/test_w8a8_pdmix.py) | 验证 w8a8 pdmix 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_delegates_to_static`, `test_get_pertensor_param_delegates_to_static`, `test_get_perchannel_param_delegates_to_static`, `test_apply_uses_static_for_kv_consumer` |
| [test_w8a8_static.py](../../../tests/ut/quantization/methods/w8a8/test_w8a8_static.py) | 验证 w8a8 static 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight`, `test_get_pertensor_param`, `test_get_perchannel_param`, `test_apply_with_x_not_int8` |
| [test_w8a8fp8_dynamic.py](../../../tests/ut/quantization/methods/w8a8/test_w8a8fp8_dynamic.py) | 验证 w8a8fp8 dynamic 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_act_quant_type`, `test_get_weight_various_sizes`, `test_get_perchannel_param_dtype_variations`, `test_quant_type_is_w8a8fp8` |

## `tests/ut/quantization/methods/wna16/`

验证 quantization/methods/wna16 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/quantization/methods/wna16/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_w4a16.py](../../../tests/ut/quantization/methods/wna16/test_w4a16.py) | 验证 w4a16 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_unpack_from_int32_restores_values_and_crops_padding`, `test_unpack_from_int32_packed_dim_1`, `test_unpack_from_int32_packed_dim_0`, `test_unpack_from_int32_packed_dim_0_restores_values` |
| [test_w4a16_mxfp4.py](../../../tests/ut/quantization/methods/wna16/test_w4a16_mxfp4.py) | 验证 MXFP 数值格式 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight_static_method`, `test_get_dynamic_quant_param_based_on_group_size`, `test_process_weights_transposes_weights`, `test_apply_full_params` |
| [test_w8a16.py](../../../tests/ut/quantization/methods/wna16/test_w8a16.py) | 验证 w8a16 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_weight`, `test_get_per_channel_param`, `test_apply_with_x_is_int8`, `test_process_weights_after_loading_with_nz1` |
