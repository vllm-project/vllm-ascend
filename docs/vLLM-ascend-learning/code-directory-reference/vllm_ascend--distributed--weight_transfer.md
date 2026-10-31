# vllm_ascend/distributed/weight_transfer：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **4 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/distributed/weight_transfer/`

在线权重更新使用的 HCCL/NPU IPC 传输实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/weight_transfer/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`register_engine` |
| [hccl_engine.py](../../../vllm_ascend/distributed/weight_transfer/hccl_engine.py) | HCCL 通信 的接口/实现或辅助；所属职责：在线权重更新使用的 HCCL/NPU IPC 传输实现。 | 注释：HCCL-based weight transfer engine.；定义：`HCCLWeightTransferInitInfo`, `HCCLTrainerSendWeightsArgs` |
| [npu_ipc_engine.py](../../../vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py) | 进程间通信 的接口/实现或辅助；所属职责：在线权重更新使用的 HCCL/NPU IPC 传输实现。 | 注释：NPU IPC-based weight transfer engine using Ascend IPC for communication.；定义：`NPUIPCWeightTransferInitInfo`, `NPUIPCTrainerInitInfo`, `get_ip`, `npu_generate_uuid` |
| [packed_tensor.py](../../../vllm_ascend/distributed/weight_transfer/packed_tensor.py) | 张量、打包 的接口/实现或辅助；所属职责：在线权重更新使用的 HCCL/NPU IPC 传输实现。 | 注释：Packed tensor utilities for efficient weight transfer.；定义：`packed_broadcast_producer`, `packed_broadcast_consumer`, `packed_npu_ipc_producer` |
