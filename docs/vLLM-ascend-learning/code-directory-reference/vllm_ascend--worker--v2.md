# vllm_ascend/worker/v2：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **43 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

V2 复用上游 Runner，需要继续追 `super()`。本地重点看 model state、缓存 allocate/reshape、ACL Graph、speculator 以及执行异常时的收尾；不要只用本地 README 的实验性清单判断功能状态。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `vllm_ascend/worker/v2/`

基于上游 V2 Runner 的 NPU 执行、状态、并行传输和图管理。

| 子目录 | 职责 |
| --- | --- |
| [model_states/](../../../vllm_ascend/worker/v2/model_states) | 按模型类型准备元数据和状态，并收尾辅助执行资源。 |
| [sample/](../../../vllm_ascend/worker/v2/sample) | V2 采样、词表分数与相应状态适配。 |
| [spec_decode/](../../../vllm_ascend/worker/v2/spec_decode) | V2 执行路径的投机生成、验证和草稿状态管理。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../vllm_ascend/worker/v2/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：&#91;Experimental&#93; Model Runner V2 |
| [__init__.py](../../../vllm_ascend/worker/v2/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [aclgraph_utils.py](../../../vllm_ascend/worker/v2/aclgraph_utils.py) | V2 捕获/重放管理、图容量选择和执行上下文包装。 | 定义：`ModelAclGraphManager`, `ModelWithContext`, `collect_sorted_captured_token_sizes`, `model_capture_wrapper` |
| [attn_utils.py](../../../vllm_ascend/worker/v2/attn_utils.py) | V2 Attention 元数据、cache spec、混合页对齐和 ring 更新辅助。 | 定义：`skip_ring_state_update`, `ring_state_update_skipped`, `unwrap_mamba_kv_cache_groups` |
| [block_table.py](../../../vllm_ascend/worker/v2/block_table.py) | V2 多请求缓存 block tables 的设备端组织。 | 定义：`AscendBlockTables` |
| [eplb.py](../../../vllm_ascend/worker/v2/eplb.py) | V2 专家负载收集和 EPLB 控制器适配。 | 定义：`AscendEPLBController`, `is_eplb_load_collection_phase_matched` |
| [input_batch.py](../../../vllm_ascend/worker/v2/input_batch.py) | V2 批次与持久输入 buffer 结构。 | 定义：`AscendInputBuffers`, `AscendInputBatch` |
| [kvpp.py](../../../vllm_ascend/worker/v2/kvpp.py) | KVPP runtime 与提前一层预取、等待和完成调度。 | 定义：`KVPPRuntime`, `KVPPScheduler` |
| [model_runner.py](../../../vllm_ascend/worker/v2/model_runner.py) | V2 NPU Runner：继承上游执行骨架，适配输入、KV、采样、草稿、并行和收尾。 | 定义：`NPUModelRunner`, `graph_manager_wrapper` |
| [pcp_manager.py](../../../vllm_ascend/worker/v2/pcp_manager.py) | V2 Prefill 上下文并行输入、Attention 上下文与状态组织。 | 定义：`AscendPCPAttentionContext`, `AscendPCPManager` |
| [pp_transport.py](../../../vllm_ascend/worker/v2/pp_transport.py) | V2 PP 与投机模式的中间数据、TopK 和兼容传输规则。 | 注释：Common Model Runner V2 pipeline-parallel utilities.；定义：`_PPTransportModel`, `SpecPPSupport`, `use_legacy_spec_pp`, `resolve_spec_pp_support`, `bypass_upstream_spec_pp_guard` |
| [states.py](../../../vllm_ascend/worker/v2/states.py) | V2 单请求运行状态结构。 | 定义：`AscendRequestState` |
| [utils.py](../../../vllm_ascend/worker/v2/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：基于上游 V2 Runner 的 NPU 执行、状态、并行传输和图管理。 | 定义：`torch_cuda_wrapper`, `communicator_switch`, `torch_npu_graph_wrapper` |

## `vllm_ascend/worker/v2/model_states/`

按模型类型准备元数据和状态，并收尾辅助执行资源。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/v2/model_states/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`init_asecnd_model_state` |
| [default.py](../../../vllm_ascend/worker/v2/model_states/default.py) | NPU 默认模型状态：准备 Attention/Offload/Engram 坐标并收尾元数据任务。 | 定义：`AscendModelState` |
| [encoder_decoder.py](../../../vllm_ascend/worker/v2/model_states/encoder_decoder.py) | 编码器、逐步生成 的接口/实现或辅助；所属职责：按模型类型准备元数据和状态，并收尾辅助执行资源。 | 定义：`AscendEncoderDecoderModelState` |
| [mamba_hybrid.py](../../../vllm_ascend/worker/v2/model_states/mamba_hybrid.py) | Mamba 状态空间 的接口/实现或辅助；所属职责：按模型类型准备元数据和状态，并收尾辅助执行资源。 | 定义：`_LayerwiseMambaCopyState`, `AscendMambaHybridModelState` |

## `vllm_ascend/worker/v2/sample/`

V2 采样、词表分数与相应状态适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/v2/sample/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [apply_top_k_top_p.py](../../../vllm_ascend/worker/v2/sample/apply_top_k_top_p.py) | TopK 选择、TopP 选择 的接口/实现或辅助；所属职责：V2 采样、词表分数与相应状态适配。 | 定义：`apply_top_k_top_p_npu` |
| [bad_words.py](../../../vllm_ascend/worker/v2/sample/bad_words.py) | 按照请求禁用词配置屏蔽相应候选 logits。 | 定义：`apply_bad_words` |
| [gumbel.py](../../../vllm_ascend/worker/v2/sample/gumbel.py) | 按请求应用温度，并以 Gumbel 相关设备 kernel 组织采样。 | 定义：`apply_temperature` |
| [logprob.py](../../../vllm_ascend/worker/v2/sample/logprob.py) | 概率归一化、TopK 选择、日志 的接口/实现或辅助；所属职责：V2 采样、词表分数与相应状态适配。 | 定义：`compute_token_logprobs`, `compute_topk_logprobs` |
| [min_p.py](../../../vllm_ascend/worker/v2/sample/min_p.py) | 按最小概率阈值规则过滤采样候选。 | 定义：`apply_min_p` |
| [penalties.py](../../../vllm_ascend/worker/v2/sample/penalties.py) | 采样惩罚 的接口/实现或辅助；所属职责：V2 采样、词表分数与相应状态适配。 | 定义：`apply_penalties`, `bincount` |

## `vllm_ascend/worker/v2/spec_decode/`

V2 执行路径的投机生成、验证和草稿状态管理。

| 子目录 | 职责 |
| --- | --- |
| [autoregressive/](../../../vllm_ascend/worker/v2/spec_decode/autoregressive) | autoregressive 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。 |
| [dflash/](../../../vllm_ascend/worker/v2/spec_decode/dflash) | DFlash 草稿 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。 |
| [dflash2/](../../../vllm_ascend/worker/v2/spec_decode/dflash2) | DFlash 草稿 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。 |
| [dspark/](../../../vllm_ascend/worker/v2/spec_decode/dspark) | DSpark 草稿 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。 |
| [eagle/](../../../vllm_ascend/worker/v2/spec_decode/eagle) | Eagle 投机 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。 |
| [mtp/](../../../vllm_ascend/worker/v2/spec_decode/mtp) | 多 token 预测 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/v2/spec_decode/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`init_speculator` |
| [lmhead_tp_utils.py](../../../vllm_ascend/worker/v2/spec_decode/lmhead_tp_utils.py) | 辅助函数 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`LmheadTPDraftSamplingMixin` |
| [pcp_utils.py](../../../vllm_ascend/worker/v2/spec_decode/pcp_utils.py) | 辅助函数、Prefill 上下文并行 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`ReplicatedPCPDraftSpeculator`, `disable_profiling_chunk_for_draft`, `disable_target_pcp_for_replicated_draft`, `prepare_replicated_pcp_config` |
| [rejection_sampler_utils.py](../../../vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py) | 候选拒绝验证、采样、辅助函数 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`rejection_sample` |

## `vllm_ascend/worker/v2/spec_decode/autoregressive/`

autoregressive 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/v2/spec_decode/autoregressive/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [aclgraph.py](../../../vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py) | 设备图 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`AutoRegressiveAclGraphManager` |
| [sparse_kv_offload.py](../../../vllm_ascend/worker/v2/spec_decode/autoregressive/sparse_kv_offload.py) | 缓存卸载、KV 缓存 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 注释：Sparse-offload metadata buffers for autoregressive draft steps.；定义：`SparseKVOffloadMetadata` |
| [speculator.py](../../../vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py) | 草稿生成 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`AscendAutoRegressiveSpeculator`, `gather`, `torch_gather_wrapper` |

## `vllm_ascend/worker/v2/spec_decode/dflash/`

DFlash 草稿 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/v2/spec_decode/dflash/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [aclgraph.py](../../../vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py) | 设备图 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`DFlashAclGraphManager` |
| [speculator.py](../../../vllm_ascend/worker/v2/spec_decode/dflash/speculator.py) | 草稿生成 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`AscendDFlashSpeculator`, `prepare_dflash_inputs_factory`, `prepare_dflash_inputs` |

## `vllm_ascend/worker/v2/spec_decode/dflash2/`

DFlash 草稿 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/v2/spec_decode/dflash2/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [speculator.py](../../../vllm_ascend/worker/v2/spec_decode/dflash2/speculator.py) | 草稿生成 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`AscendDFlash2Speculator` |

## `vllm_ascend/worker/v2/spec_decode/dspark/`

DSpark 草稿 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/v2/spec_decode/dspark/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [speculator.py](../../../vllm_ascend/worker/v2/spec_decode/dspark/speculator.py) | 草稿生成 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`AscendDSparkSpeculator` |

## `vllm_ascend/worker/v2/spec_decode/eagle/`

Eagle 投机 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/v2/spec_decode/eagle/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [speculator.py](../../../vllm_ascend/worker/v2/spec_decode/eagle/speculator.py) | 草稿生成 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`AscendEagleSpeculator` |

## `vllm_ascend/worker/v2/spec_decode/mtp/`

多 token 预测 子模块；所属范围：V2 执行路径的投机生成、验证和草稿状态管理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/worker/v2/spec_decode/mtp/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [speculator.py](../../../vllm_ascend/worker/v2/spec_decode/mtp/speculator.py) | 草稿生成 的接口/实现或辅助；所属职责：V2 执行路径的投机生成、验证和草稿状态管理。 | 定义：`AscendMTPSpeculator` |
