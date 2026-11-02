# tests/ut/simple_kv_offload：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **2 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/simple_kv_offload/`

验证 simple_kv_offload 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/simple_kv_offload/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：Tests for Ascend simple KV cache offloading. |
| [test_simple_kv_offload.py](../../../tests/ut/simple_kv_offload/test_simple_kv_offload.py) | 验证 缓存卸载、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_factory_registration_uses_consolidated_package`, `test_connector_only_replaces_enabled_worker`, `test_flatten_kv_value_preserves_separate_kv_tensors`, `test_build_block_views_uses_tensor_offset_not_whole_storage` |
