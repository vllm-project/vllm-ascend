# tests/ut/_tools：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **9 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/_tools/`

验证 _tools 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/_tools/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [build_cache_test_utils.py](../../../tests/ut/_tools/build_cache_test_utils.py) | 缓存、辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`build_cache_command`, `run_command` |
| [test_ai_qos_tool.py](../../../tests/ut/_tools/test_ai_qos_tool.py) | 验证 ai qos tool 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_device_list_uses_all_visible_devices_when_env_unset`, `test_device_list_exits_when_env_unset_and_torch_query_fails`, `test_device_list_parses_visible_devices`, `test_device_list_parses_single_id` |
| [test_build_cache.py](../../../tests/ut/_tools/test_build_cache.py) | 验证 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_custom_operator_miss_then_hit_and_restore`, `test_prepared_input_change_invalidates_and_revert_reuses_prior_entry`, `test_prepared_text_normalizes_explicit_root`, `test_prepared_text_semantic_change_remains_identity_sensitive` |
| [test_build_cache_concurrency.py](../../../tests/ut/_tools/test_build_cache_concurrency.py) | 验证 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_entry_lock_timeout_bypasses_cache`, `test_publish_lock_timeout_fails_fast`, `test_action_lock_timeout_fails_fast`, `test_many_parallel_hits_complete_without_hang` |
| [test_docs_codegen.py](../../../tests/ut/_tools/test_docs_codegen.py) | 验证 docs codegen 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_block_scanner_parses_metadata_and_trims_raw_block`, `test_block_scanner_rejects_duplicate_block_names`, `test_block_scanner_rejects_unsupported_metadata`, `test_single_node_converter_uses_case_index_defaults_and_extra_args` |
| [test_doctest_helper.py](../../../tests/ut/_tools/test_doctest_helper.py) | 验证 doctest helper 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_extract_doctest_block`, `test_extract_doctest_block_rejects_invalid_input`, `test_mkdocs_helpers`, `test_parse_manual_selection` |
| [test_generate_zh_docs.py](../../../tests/ut/_tools/test_generate_zh_docs.py) | 验证 generate zh docs 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_reset_output_dir_removes_stale_generated_files`, `test_preserves_translated_list_structure_with_multi_backtick_code`, `test_preserves_code_order_from_translated_paragraph`, `test_short_translations_do_not_modify_protected_spans` |
| [test_prepare_csrc_l1_restore.py](../../../tests/ut/_tools/test_prepare_csrc_l1_restore.py) | 验证 准备、存储 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_prepare_uses_engine_output_and_exports_environment`, `test_prepare_missing_engine_is_unsupported_but_exports_environment`, `test_prepare_rejects_invalid_snapshot_key_output`, `test_prepare_snapshot_key_failure_is_unsupported` |
