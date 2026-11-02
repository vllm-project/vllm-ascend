# vllm_ascend/distributed/device_communicators：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **4 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/distributed/device_communicators/`

NPU 设备集合通信及相关通信后端适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/device_communicators/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [npu_communicator.py](../../../vllm_ascend/distributed/device_communicators/npu_communicator.py) | 通信器 的接口/实现或辅助；所属职责：NPU 设备集合通信及相关通信后端适配。 | 定义：`_NpuAll2AllManager`, `NPUCommunicator` |
| [pyhccl.py](../../../vllm_ascend/distributed/device_communicators/pyhccl.py) | HCCL 通信 的接口/实现或辅助；所属职责：NPU 设备集合通信及相关通信后端适配。 | 定义：`PyHcclCommunicator` |
| [pyhccl_wrapper.py](../../../vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py) | HCCL 通信 的接口/实现或辅助；所属职责：NPU 设备集合通信及相关通信后端适配。 | 定义：`hcclUniqueId`, `hcclDataTypeEnum` |
