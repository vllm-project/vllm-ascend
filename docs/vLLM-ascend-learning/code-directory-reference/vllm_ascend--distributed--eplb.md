# vllm_ascend/distributed/eplb：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **9 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/distributed/eplb/`

专家负载均衡 子模块；所属范围：并行通信组、设备通信、专家负载策略以及 KV/权重传输。

| 子目录 | 职责 |
| --- | --- |
| [policy/](../../../vllm_ascend/distributed/eplb/policy) | policy 子模块；所属范围：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/eplb/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：Ascend integration for the vLLM distributed EPLB runtime. |
| [eplb_communicator.py](../../../vllm_ascend/distributed/eplb/eplb_communicator.py) | 通信器、专家负载均衡 的接口/实现或辅助；所属职责：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 | 注释：Ascend communicators for asynchronous EPLB.；定义：`_HixlTransferTiming`, `AscendGlooEplbCommunicator` |
| [eplb_state.py](../../../vllm_ascend/distributed/eplb/eplb_state.py) | 状态、专家负载均衡 的接口/实现或辅助；所属职责：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 | 注释：Ascend-owned extensions for the upstream EPLB state.；定义：`AscendEplbLayerState`, `AscendEplbState`, `refresh_model_routing_tables` |
| [explicit_transfer.py](../../../vllm_ascend/distributed/eplb/explicit_transfer.py) | 传输 的接口/实现或辅助；所属职责：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 | 注释：Execution helpers for policy-provided explicit migration plans.；定义：`stage_explicit_layer_transfer` |
| [hixl_compat.py](../../../vllm_ascend/distributed/eplb/hixl_compat.py) | 用 ctypes 绑定 EPLB 需要的 CANN HIXL 引擎接口。 | 注释：Minimal ctypes binding for the CANN HIXL engine API used by EPLB.；定义：`MemType`, `TransferOp`, `ensure_available` |

## `vllm_ascend/distributed/eplb/policy/`

policy 子模块；所属范围：并行通信组、设备通信、专家负载策略以及 KV/权重传输。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/eplb/policy/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`PreparedLoadStats` |
| [_stair_process.py](../../../vllm_ascend/distributed/eplb/policy/_stair_process.py) | STAIR 独立规划进程与调用方之间的本地 socket 请求/响应协议。 | 注释：Local socket protocol for the isolated STAIR planning process.；定义：`send_planner_request`, `receive_planner_response` |
| [factory.py](../../../vllm_ascend/distributed/eplb/policy/factory.py) | 专家负载均衡 的接口/实现或辅助；所属职责：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 | 定义：`create_eplb_policy` |
| [stair.py](../../../vllm_ascend/distributed/eplb/policy/stair.py) | STAIR EPLB 策略的 CPU 侧放置计划与负载计算组件。 | 注释：CPU building blocks for the STAIR EPLB policy.；定义：`PlacementImbalance`, `PlacementPlan` |
