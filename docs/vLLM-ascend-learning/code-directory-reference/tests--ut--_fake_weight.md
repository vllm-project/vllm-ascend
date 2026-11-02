# tests/ut/_fake_weight：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **2 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/_fake_weight/`

验证 _fake_weight 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/_fake_weight/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [config.json](../../../tests/ut/_fake_weight/config.json) | 配置 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_name_or_path`, `activation_dropout`, `activation_function`, `architectures`, `attention_dropout` |
