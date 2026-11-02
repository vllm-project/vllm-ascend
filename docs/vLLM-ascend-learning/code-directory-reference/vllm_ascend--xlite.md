# vllm_ascend/xlite：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **5 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/xlite/`

Xlite 专用 Worker、模型包装及执行适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/xlite/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [utils.py](../../../vllm_ascend/xlite/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：Xlite 专用 Worker、模型包装及执行适配。 | 注释：Utility functions for xlite.；定义：`_TensorRepr`, `AttributeSetterMixin`, `set_dummy_tensor`, `get_nested_attr`, `get_dotted_attr` |
| [xlite.py](../../../vllm_ascend/xlite/xlite.py) | 模型 的接口/实现或辅助；所属职责：Xlite 专用 Worker、模型包装及执行适配。 | 注释：Xlite integration module for vLLM-Ascend.；定义：`XliteModelBase`, `StandardXliteModel`, `get_adapter_xlite_model` |
| [xlite_model_runner.py](../../../vllm_ascend/xlite/xlite_model_runner.py) | 模型执行、模型 的接口/实现或辅助；所属职责：Xlite 专用 Worker、模型包装及执行适配。 | 定义：`XliteModelRunner` |
| [xlite_worker.py](../../../vllm_ascend/xlite/xlite_worker.py) | 设备执行 的接口/实现或辅助；所属职责：Xlite 专用 Worker、模型包装及执行适配。 | 定义：`XliteWorker` |
