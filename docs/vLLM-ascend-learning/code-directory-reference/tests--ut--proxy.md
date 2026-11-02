# tests/ut/proxy：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **3 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/proxy/`

验证 proxy 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/proxy/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_load_balance_proxy_scheduler.py](../../../tests/ut/proxy/test_load_balance_proxy_scheduler.py) | 验证 调度、加载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_failed_reassignment_releases_previous_decoder_once` |
| [test_load_balance_proxy_server.py](../../../tests/ut/proxy/test_load_balance_proxy_server.py) | 验证 加载 的测试；具体断言场景由右栏测试符号定位。 | 注释：System-level tests for load_balance_proxy_server_example.；定义：`test_normal_nonstream`, `test_normal_stream`, `test_overlimit_nonstream`, `test_overlimit_stream` |
