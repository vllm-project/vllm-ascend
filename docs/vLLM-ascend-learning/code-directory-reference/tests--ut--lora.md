# tests/ut/lora：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **6 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/lora/`

验证 lora 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/lora/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_lora.py](../../../tests/ut/lora/test_lora.py) | 验证 LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ascend_fused_moe_lora_initializes_skipped_upstream_fields`, `test_ascend_fused_moe_lora_omits_shared_experts_attr_when_absent`, `test_set_mapping_publishes_context_on_base_layer`, `test_build_lora_context_propagates_ep_flag` |
| [test_lora_ops.py](../../../tests/ut/lora/test_lora_ops.py) | 验证 LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_bgmv_shrink_forwards_scale_in_kernel_order`, `test_bgmv_expand_uses_full_output_width_and_drops_add_inputs`, `test_bgmv_expand_slice_forwards_offset_and_size`, `test_sgmv_wrappers_drop_unused_batch_metadata` |
| [test_punica_npu.py](../../../tests/ut/lora/test_punica_npu.py) | 验证 LoRA 批量执行 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_punica_init_selects_kernel_backend`, `test_prefill_calls_sgmv_when_lora_active`, `test_prefill_shrink_and_expand_skip_when_no_lora`, `test_decode_always_invokes_bgmv` |
| [test_quant_moe.py](../../../tests/ut/lora/test_quant_moe.py) | 验证 量化、混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dynamic_int8_lora_injects_at_float_boundaries`, `test_dynamic_int8_lora_reads_weights_from_layer_via_quant_method`, `test_dynamic_int8_lora_rejects_unsupported_modes`, `test_dynamic_int8_all2all_lora_handles_empty_ep_rank` |
| [test_utils.py](../../../tests/ut/lora/test_utils.py) | 验证 辅助函数 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_refresh_all_lora_classes_prepends_ascend_wrappers` |
