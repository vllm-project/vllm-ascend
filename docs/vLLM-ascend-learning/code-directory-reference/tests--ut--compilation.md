# tests/ut/compilation：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **5 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/compilation/`

验证 compilation 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/compilation/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_acl_graph.py](../../../tests/ut/compilation/test_acl_graph.py) | 验证 设备图 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_update_full_graph_params_dispatches_draft_metadata_by_keyword`, `test_aclgraph_entry_initialization`, `test_aclgraph_entry_with_values`, `test_initialization_with_default_options` |
| [test_graph_fusion_pass_manager.py](../../../tests/ut/compilation/test_graph_fusion_pass_manager.py) | 验证 管理器、算子融合、设备图 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_configure_consumes_validated_ascend_compilation_config` |
| [test_npugraph_ex_utils_check.py](../../../tests/ut/compilation/test_npugraph_ex_utils_check.py) | 验证 设备图、辅助函数、检查 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_extra_stream_scope_check_logic` |
| [test_qknorm_rope_fusion_pass.py](../../../tests/ut/compilation/test_qknorm_rope_fusion_pass.py) | 验证 算子融合、位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_pattern_key_is_distinct_per_shape`, `test_registers_one_pattern_per_attention_shape`, `test_skips_attention_shape_over_unified_buffer`, `test_hardware_without_the_kernel_keeps_only_qk_patterns` |
