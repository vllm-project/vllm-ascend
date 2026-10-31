# vllm_ascend/model_executor：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **12 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/model_executor/`

模型执行辅助和算子预热；完整执行调度还在 Worker/Runner。

| 子目录 | 职责 |
| --- | --- |
| [offloader/](../../../vllm_ascend/model_executor/offloader) | 缓存卸载 子模块；所属范围：模型执行辅助和算子预热；完整执行调度还在 Worker/Runner。 |
| [warmup/](../../../vllm_ascend/model_executor/warmup) | 提前执行需要的算子路径，减少首次请求的初始化开销。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/model_executor/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `vllm_ascend/model_executor/offloader/`

缓存卸载 子模块；所属范围：模型执行辅助和算子预热；完整执行调度还在 Worker/Runner。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/model_executor/offloader/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：Ascend-specific model parameter offloading. |
| [base.py](../../../vllm_ascend/model_executor/offloader/base.py) | 缓存卸载 的接口/实现或辅助；所属职责：模型执行辅助和算子预热；完整执行调度还在 Worker/Runner。 | 定义：`create_offloader` |
| [prefetch.py](../../../vllm_ascend/model_executor/offloader/prefetch.py) | 以预取方式从 CPU 装入权重，使用 NZ 格式静态 buffer 对接 NPU 计算。 | 注释：Ascend prefetch-based CPU offloading with NZ-format static buffers.；定义：`ParamInfo`, `AscendPrefetchOffloader`, `mark_prefetch_offload_nz_weight` |

## `vllm_ascend/model_executor/warmup/`

提前执行需要的算子路径，减少首次请求的初始化开销。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/model_executor/warmup/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：Triton kernel warmup for Ascend NPU. |
| [early_kernel_warmup.py](../../../vllm_ascend/model_executor/warmup/early_kernel_warmup.py) | 预热 的接口/实现或辅助；所属职责：提前执行需要的算子路径，减少首次请求的初始化开销。 | 注释：Compile Triton warmup kernels while weight I/O is already on the critical path.；定义：`_WarmupShim`, `start_early_kernel_warmup`, `join_early_kernel_warmup` |
| [indexer_triton_warmup.py](../../../vllm_ascend/model_executor/warmup/indexer_triton_warmup.py) | 历史位置索引、预热 的接口/实现或辅助；所属职责：提前执行需要的算子路径，减少首次请求的初始化开销。 | 注释：Warm the finite tile variants used by the V4.1 indexer.；定义：`collect_indexer_warmup_token_counts`, `indexer_triton_warmup` |
| [kernel_warmup.py](../../../vllm_ascend/model_executor/warmup/kernel_warmup.py) | 预热 的接口/实现或辅助；所属职责：提前执行需要的算子路径，减少首次请求的初始化开销。 | 注释：Warm up Triton kernels used during model execution on Ascend NPU.；定义：`kernel_warmup` |
| [nz_warmup.py](../../../vllm_ascend/model_executor/warmup/nz_warmup.py) | 预热 的接口/实现或辅助；所属职责：提前执行需要的算子路径，减少首次请求的初始化开销。 | 注释：Pay the first NZ format-cast lazy init while weight I/O is on the critical path.；定义：`warm_nz_format_cast`, `start_nz_warm_thread`, `join_nz_warm_thread` |
| [penalties_triton_warmup.py](../../../vllm_ascend/model_executor/warmup/penalties_triton_warmup.py) | 采样惩罚、预热 的接口/实现或辅助；所属职责：提前执行需要的算子路径，减少首次请求的初始化开销。 | 注释：Warm up Triton kernels used by ``apply_penalties_triton`` (bincount + penalties).；定义：`penalties_triton_warmup` |
| [rejection_sampler_triton_warmup.py](../../../vllm_ascend/model_executor/warmup/rejection_sampler_triton_warmup.py) | 候选拒绝验证、采样、预热 的接口/实现或辅助；所属职责：提前执行需要的算子路径，减少首次请求的初始化开销。 | 注释：Warm up rejection sampler Triton kernels used during speculative decoding.；定义：`collect_warmup_rejection_block_sizes`, `collect_warmup_batch_sizes`, `rejection_sampler_triton_warmup` |
| [rms_triton_warmup.py](../../../vllm_ascend/model_executor/warmup/rms_triton_warmup.py) | 预热 的接口/实现或辅助；所属职责：提前执行需要的算子路径，减少首次请求的初始化开销。 | 注释：Warm up ``triton_q_rms`` (see ``ops/triton/rms_norm.py``).；定义：`collect_triton_rms_warmup_block_m_values`, `triton_rms_warmup` |
