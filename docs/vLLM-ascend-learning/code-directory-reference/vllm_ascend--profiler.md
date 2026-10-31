# vllm_ascend/profiler：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **2 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/profiler/`

设备性能分析器与运行记录接入。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/profiler/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`vllm_ascend.profiler.torch_npu_profiler` |
| [torch_npu_profiler.py](../../../vllm_ascend/profiler/torch_npu_profiler.py) | 性能分析 的接口/实现或辅助；所属职责：设备性能分析器与运行记录接入。 | 定义：`TorchNPUProfilerWrapper` |
