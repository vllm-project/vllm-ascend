# tests/ut/device_allocator：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **5 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/device_allocator/`

验证 device_allocator 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/device_allocator/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_camem.py](../../../tests/ut/device_allocator/test_camem.py) | 验证 camem 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_create_and_map_calls_python_create_and_map`, `test_unmap_and_release_calls_python_unmap_and_release`, `test_get_pluggable_allocator`, `test_singleton_behavior` |
| [test_cpu_binding.py](../../../tests/ut/device_allocator/test_cpu_binding.py) | 验证 接口绑定 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_execute_command`, `test_execute_command_kills_timed_out_process`, `test_get_npu_map_info`, `test_get_npu_map_info_without_chip_logic_id_uses_npu_id` |
| [test_find_loaded_library.py](../../../tests/ut/device_allocator/test_find_loaded_library.py) | 验证 加载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_find_loaded_library_success_and_not_found` |
| [test_sleep_mem_optimized.py](../../../tests/ut/device_allocator/test_sleep_mem_optimized.py) | 验证 休眠资源 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_acl_graph_reset_graph_params_clears_list_values_only`, `test_acl_graph_wakeup_waits_for_kv_cache_tag`, `test_sleep_wakeup_manager_skips_acl_sleep_when_aclgraph_disabled`, `test_sleep_wakeup_manager_cleans_acl_before_hccl_when_aclgraph_enabled` |
