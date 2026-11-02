# vllm_ascend/distributed：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **4 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/distributed/`

并行通信组、设备通信、专家负载策略以及 KV/权重传输。

| 子目录 | 职责 |
| --- | --- |
| [device_communicators/](vllm_ascend--distributed--device_communicators.md) | NPU 设备集合通信及相关通信后端适配。 |
| [ec_transfer/](vllm_ascend--distributed--ec_transfer.md) | 传输 子模块；所属范围：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 |
| [eplb/](vllm_ascend--distributed--eplb.md) | 专家负载均衡 子模块；所属范围：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 |
| [kv_transfer/](vllm_ascend--distributed--kv_transfer.md) | KV 连接器注册、PD 传输、共享缓存池和稀疏缓存迁移。 |
| [weight_transfer/](vllm_ascend--distributed--weight_transfer.md) | 在线权重更新使用的 HCCL/NPU IPC 传输实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [kvpp.py](../../../vllm_ascend/distributed/kvpp.py) | 使用 KVPP 设备组广播缓存，并保证设备传输完成后再返回。 | 定义：`BroadcastKVPPTransport` |
| [parallel_state.py](../../../vllm_ascend/distributed/parallel_state.py) | 初始化和查询 Ascend 并行通信组及其生命周期。 | 定义：`ReplicatedGroup`, `init_ascend_model_parallel`, `model_parallel_initialized`, `get_mc2_group` |
| [utils.py](../../../vllm_ascend/distributed/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 | 定义：`get_decode_context_model_parallel_world_size`, `get_decode_context_model_parallel_rank`, `all_gather_async` |
