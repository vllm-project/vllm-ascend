# vllm_ascend/worker：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **10 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

阅读顺序：`worker.py` 的设备生命周期 → `model_runner_v1.py` 的请求执行 → `npu_input_batch.py` 与 `block_table.py` 的坐标 → Attention metadata 和 Forward Context。设备元数据还需配合 `device_metadata.py` 的提交、等待、回收协议。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `vllm_ascend/worker/`

Worker 生命周期、V1/V2 Runner、输入组批、缓存分配与设备元数据。

| 子目录 | 职责 |
| --- | --- |
| [v2/](vllm_ascend--worker--v2.md) | 基于上游 V2 Runner 的 NPU 执行、状态、并行传输和图管理。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [block_table.py](../../../vllm_ascend/worker/block_table.py) | V1 单/多缓存组的逻辑 block 到物理页映射及更新。 | 定义：`BlockTable`, `MultiGroupBlockTable` |
| [dcp_utils.py](../../../vllm_ascend/worker/dcp_utils.py) | Decode 上下文并行下的输入与投机辅助结构。 | 定义：`DCPSpecDecodeMTPInputs`, `DCPSpecDecodeFirstPassInputs` |
| [device_metadata.py](../../../vllm_ascend/worker/device_metadata.py) | Compressor/Attention/Indexer 设备元数据的异步生产、等待和 buffer 复用协议。 | 定义：`DeviceMetadataStage`, `DeviceMetadataTask`, `wait_for_device_metadata` |
| [encoder_acl_graph.py](../../../vllm_ascend/worker/encoder_acl_graph.py) | 视觉等编码器的图参数、运行上下文、更新与图管理。 | 注释：NPU-specific encoder ACL graph: params, runtime context, FIA replay updates, and manager.；定义：`EncoderGraphParams`, `EncoderForwardContext`, `set_encoder_graph_params`, `get_encoder_graph_params`, `update_encoder_graph_workspace` |
| [kvpp_cache.py](../../../vllm_ascend/worker/kvpp_cache.py) | 依据 KVPP 放置计划分配持久缓存和 scratch buffer。 | 定义：`get_kvpp_cache_specs`, `allocate_kvpp_cache` |
| [model_runner_v1.py](../../../vllm_ascend/worker/model_runner_v1.py) | V1 执行主线：持久批次、输入准备、模型调用、KV、图执行、采样与投机衔接。 | 定义：`NPUModelRunner`, `graph_capture`, `get_tp_context`, `update_pass_config` |
| [npu_input_batch.py](../../../vllm_ascend/worker/npu_input_batch.py) | V1 持久请求批次、token 和采样相关输入状态。 | 定义：`NPUInputBatch` |
| [utils.py](../../../vllm_ascend/worker/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：Worker 生命周期、V1/V2 Runner、输入组批、缓存分配与设备元数据。 | 定义：`AscendKVBlockZeroer`, `copy_kv_cache_blocks_inplace`, `disable_compilation` |
| [worker.py](../../../vllm_ascend/worker/worker.py) | NPUWorker：设备和通信初始化、模型加载、内存预算、缓存、预热、执行与权重更新。 | 定义：`NPUWorker`, `parse_text_output` |
