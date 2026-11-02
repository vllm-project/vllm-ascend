# tests/ut/helpers：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **1 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/helpers/`

验证 helpers 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [golden_copy_and_expand.py](../../../tests/ut/helpers/golden_copy_and_expand.py) | 复制 的测试辅助、测试入口或共享数据结构。 | 注释：CPU golden reference for ``npu_copy_and_expand_eagle_inputs``.；定义：`golden_copy_and_expand_eagle_inputs`, `npu_copy_and_expand_eagle_inputs_stub` |
