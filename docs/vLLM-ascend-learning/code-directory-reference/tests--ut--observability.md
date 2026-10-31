# tests/ut/observability：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **1 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/observability/`

验证 observability 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_ms_metrics_provider.py](../../../tests/ut/observability/test_ms_metrics_provider.py) | 验证 指标 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_get_metric_provider_returns_packaged_yaml`, `test_setup_registers_provider_entry_point_and_yaml_package_data`, `test_provider_entry_point_loads_without_metric_core`, `test_ascend_handler_module_loads_from_yaml_path` |
