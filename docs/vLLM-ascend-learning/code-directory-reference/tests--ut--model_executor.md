# tests/ut/model_executor：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **13 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/model_executor/`

验证 model_executor 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [warmup/](../../../tests/ut/model_executor/warmup) | 验证 model_executor/warmup 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_deepseek_v4.py](../../../tests/ut/model_executor/test_deepseek_v4.py) | 验证 DeepSeek 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_routed_moe_receives_configured_swiglu_limit` |
| [test_glm_mla_dspark.py](../../../tests/ut/model_executor/test_glm_mla_dspark.py) | 验证 DSpark 草稿、MLA 潜在注意力、GLM 模型 的测试；具体断言场景由右栏测试符号定位。 | 注释：MLA DSpark compatibility tests; these do not claim PD inference coverage.；定义：`test_registered_glm_draft_uses_glm_module`, `test_full_mla_draft_block_is_bidirectional`, `test_mla_cache_spec_matches_bidirectional_query_metadata`, `test_aux_projection_preserves_order_not_mean_pooling` |
| [test_offloader.py](../../../tests/ut/model_executor/test_offloader.py) | 验证 缓存卸载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_create_offloader_without_config_returns_noop`, `test_create_offloader_for_non_prefetch_backend_returns_noop`, `test_is_using_nz_weight_handles_invalid_npu_format` |
| [test_qwen3_dspark.py](../../../tests/ut/model_executor/test_qwen3_dspark.py) | 验证 DSpark 草稿、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 注释：CPU-only tests for Qwen3 DSpark weight loading.；定义：`test_quarot_loads_missing_target_vocab_shards`, `test_rotates_only_fc_weights` |

## `tests/ut/model_executor/warmup/`

验证 model_executor/warmup 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/model_executor/warmup/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [helpers.py](../../../tests/ut/model_executor/warmup/helpers.py) | helpers 的测试辅助、测试入口或共享数据结构。 | 注释：Shared helpers for Triton warmup unit tests.；定义：`make_mock_worker` |
| [test_early_kernel_warmup.py](../../../tests/ut/model_executor/warmup/test_early_kernel_warmup.py) | 验证 预热 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for ``early_kernel_warmup``.；定义：`test_config_switches_default_to_off`, `test_uninitialized_config_counts_as_disabled`, `test_start_is_noop_when_disabled`, `test_start_without_vllm_config_does_not_latch` |
| [test_indexer_triton_warmup.py](../../../tests/ut/model_executor/warmup/test_indexer_triton_warmup.py) | 验证 历史位置索引、预热 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_collect_indexer_warmup_token_counts`, `test_warmup_covers_tiles_and_active_compression_ratios`, `test_warmup_skips_unused_indexer` |
| [test_kernel_warmup.py](../../../tests/ut/model_executor/warmup/test_kernel_warmup.py) | 验证 预热 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for ``kernel_warmup``.；定义：`test_kernel_warmup` |
| [test_nz_warmup.py](../../../tests/ut/model_executor/warmup/test_nz_warmup.py) | 验证 预热 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for ``nz_warmup``.；定义：`test_warm_nz_format_cast_runs_once`, `test_warm_nz_format_cast_swallows_failures`, `test_thread_is_skipped_when_switch_is_off`, `test_thread_is_started_only_once` |
| [test_penalties_triton_warmup.py](../../../tests/ut/model_executor/warmup/test_penalties_triton_warmup.py) | 验证 采样惩罚、预热 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for ``penalties_triton_warmup``.；定义：`test_make_history_tokens`, `test_local_vocab_size`, `test_penalties_triton_warmup` |
| [test_rejection_sampler_triton_warmup.py](../../../tests/ut/model_executor/warmup/test_rejection_sampler_triton_warmup.py) | 验证 候选拒绝验证、采样、预热 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for ``rejection_sampler_triton_warmup``.；定义：`test_collect_warmup_rejection_block_sizes`, `test_rejection_sampler_triton_warmup` |
| [test_rms_triton_warmup.py](../../../tests/ut/model_executor/warmup/test_rms_triton_warmup.py) | 验证 预热 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for ``rms_triton_warmup``.；定义：`test_triton_rms_warmup`, `test_collect_block_m_values_are_powers_of_two`, `test_triton_rms_warmup_assume_used_skips_model_probe`, `test_triton_rms_warmup_joins_early_thread` |
