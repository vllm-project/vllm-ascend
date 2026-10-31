# vllm_ascend/models/layer：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **3 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/models/layer/`

layer 子模块；所属范围：模型注册、模型专有层结构、权重映射与草稿模型实现。

| 子目录 | 职责 |
| --- | --- |
| [attention/](../../../vllm_ascend/models/layer/attention) | 注意力 子模块；所属范围：模型注册、模型专有层结构、权重映射与草稿模型实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/layer/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `vllm_ascend/models/layer/attention/`

注意力 子模块；所属范围：模型注册、模型专有层结构、权重映射与草稿模型实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/layer/attention/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [layer.py](../../../vllm_ascend/models/layer/attention/layer.py) | layer 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Attention layer.；定义：`DSAAttention`, `get_dsv4_block_sizes`, `dsv4_block_sizes` |
