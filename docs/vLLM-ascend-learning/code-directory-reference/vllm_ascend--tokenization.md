# vllm_ascend/tokenization：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **2 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/tokenization/`

可选的长文本并行 tokenization。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/tokenization/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`.lopt_tokenizer` |
| [lopt_tokenizer.py](../../../vllm_ascend/tokenization/lopt_tokenizer.py) | 长文本分片并行 tokenization、边界匹配合并和回退。 | 定义：`LosslessParallelTokenizer`, `LoptConfig` |
