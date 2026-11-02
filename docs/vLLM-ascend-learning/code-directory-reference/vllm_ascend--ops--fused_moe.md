# vllm_ascend/ops/fused_moe：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **25 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

从 `AscendMoERunner` 顺着 router → dispatch → 专家 FFN → combine 阅读，再核对共享专家的输入布局与阶段事件。通信策略来自本轮上下文，量化实现位于 `quantization/methods/`，CANN MegaMoE 的内部实现不全在本仓。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `vllm_ascend/ops/fused_moe/`

MoE 路由、token 分发/合并、专家 FFN、共享专家和通信执行方法。

| 子目录 | 职责 |
| --- | --- |
| [dataclass/](../../../vllm_ascend/ops/fused_moe/dataclass) | 明确描述路由、量化、通信输入输出和异步边界的数据结构。 |
| [router/](../../../vllm_ascend/ops/fused_moe/router) | 不同专家评分、TopK、分组及视觉/hash 路由的选择适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/fused_moe/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [eplb.py](../../../vllm_ascend/ops/fused_moe/eplb.py) | 专家负载均衡 的接口/实现或辅助；所属职责：MoE 路由、token 分发/合并、专家 FFN、共享专家和通信执行方法。 | 定义：`build_expert_replica_routing_table`, `map_to_physical`, `map_to_physical_for_ascend` |
| [force_eplb.py](../../../vllm_ascend/ops/fused_moe/force_eplb.py) | 专家负载均衡 的接口/实现或辅助；所属职责：MoE 路由、token 分发/合并、专家 FFN、共享专家和通信执行方法。 | 定义：`build_force_eplb_topk`, `get_force_eplb_topk` |
| [fused_moe.py](../../../vllm_ascend/ops/fused_moe/fused_moe.py) | AscendMoERunner，协调路由专家/共享专家、变换、融合与结果归约。 | 定义：`AscendMoERunner` |
| [gate_linear.py](../../../vllm_ascend/ops/fused_moe/gate_linear.py) | Router gate Linear 的设备计算与相关适配。 | 定义：`AscendGateLinear` |
| [mega_moe.py](../../../vllm_ascend/ops/fused_moe/mega_moe.py) | 验证 MegaMoE 共享专家算子能力、MXFP 参数布局与 scale ABI。 | 定义：`supports_mega_moe_shared_experts`, `build_mega_moe_shared_kwargs` |
| [moe_comm_method.py](../../../vllm_ascend/ops/fused_moe/moe_comm_method.py) | AllGather/AllToAll/MC2/FusedMC2 后端及 MegaMoE buffer、ABI 与通信生命周期。 | 定义：`FusedExpertsResult`, `MoECommMethod`, `get_moe_comm_method`, `setup_moe_comm_method` |
| [moe_mlp.py](../../../vllm_ascend/ops/fused_moe/moe_mlp.py) | 专家 GMM1、激活/量化与 GMM2 的计算组织。 | 定义：`apply_moe_mlp` |
| [moe_utils.py](../../../vllm_ascend/ops/fused_moe/moe_utils.py) | 辅助函数、混合专家 的接口/实现或辅助；所属职责：MoE 路由、token 分发/合并、专家 FFN、共享专家和通信执行方法。 | 定义：`async_all_to_all`, `gather_from_sequence_parallel_region`, `load_cann_mega_moe_ops` |
| [prepare_finalize.py](../../../vllm_ascend/ops/fused_moe/prepare_finalize.py) | 各 MoE 后端的输入准备、分片/补齐及结果还原。 | 定义：`PrepareAndFinalize`, `PrepareAndFinalizeWithAll2All` |
| [routed_experts.py](../../../vllm_ascend/ops/fused_moe/routed_experts.py) | 路由专家执行适配、权重/路由参数组装和阶段事件。 | 定义：`AscendUnquantizedFusedMoEMethod`, `EplbExpertTensorList`, `use_multistage_eplb_load`, `make_eplb_placement_config` |
| [shared_experts.py](../../../vllm_ascend/ops/fused_moe/shared_experts.py) | 共享专家的输入组织、计算、流事件等待及与 routed 路径重叠。 | 定义：`SharedExpertParallelMode`, `SharedExpertMLPPath` |
| [token_dispatcher.py](../../../vllm_ascend/ops/fused_moe/token_dispatcher.py) | 各通信策略的 token 分发、专家分组和结果合并。 | 定义：`MoETokenDispatcher`, `TokenDispatcherWithMC2` |

## `vllm_ascend/ops/fused_moe/dataclass/`

明确描述路由、量化、通信输入输出和异步边界的数据结构。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/fused_moe/dataclass/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [fused_experts.py](../../../vllm_ascend/ops/fused_moe/dataclass/fused_experts.py) | 专家 的接口/实现或辅助；所属职责：明确描述路由、量化、通信输入输出和异步边界的数据结构。 | 定义：`MoEWeights`, `MoEFusedExpertsInput`, `build_fused_experts_input` |
| [moe_mlp.py](../../../vllm_ascend/ops/fused_moe/dataclass/moe_mlp.py) | 混合专家 的接口/实现或辅助；所属职责：明确描述路由、量化、通信输入输出和异步边界的数据结构。 | 定义：`MoEMlpComputeInput`, `build_mlp_compute_input` |
| [moe_quant.py](../../../vllm_ascend/ops/fused_moe/dataclass/moe_quant.py) | 量化、混合专家 的接口/实现或辅助；所属职责：明确描述路由、量化、通信输入输出和异步边界的数据结构。 | 定义：`MoEMxfpParams`, `MoEQuantParams`, `build_quant_params` |
| [prepare_finalize.py](../../../vllm_ascend/ops/fused_moe/dataclass/prepare_finalize.py) | 收尾、准备 的接口/实现或辅助；所属职责：明确描述路由、量化、通信输入输出和异步边界的数据结构。 | 定义：`MoEPrepareOutput` |
| [router_input.py](../../../vllm_ascend/ops/fused_moe/dataclass/router_input.py) | 专家路由、输入 的接口/实现或辅助；所属职责：明确描述路由、量化、通信输入输出和异步边界的数据结构。 | 定义：`MoeRouterInput` |
| [shared_experts.py](../../../vllm_ascend/ops/fused_moe/dataclass/shared_experts.py) | 专家 的接口/实现或辅助；所属职责：明确描述路由、量化、通信输入输出和异步边界的数据结构。 | 定义：`RoutedMoEMilestones`, `PreparedSharedExpertInput` |
| [token_dispatcher.py](../../../vllm_ascend/ops/fused_moe/dataclass/token_dispatcher.py) | token 分发 的接口/实现或辅助；所属职责：明确描述路由、量化、通信输入输出和异步边界的数据结构。 | 定义：`MoEMC2CombineMetadata`, `MoEAllGatherCombineMetadata`, `build_token_dispatch_input` |

## `vllm_ascend/ops/fused_moe/router/`

不同专家评分、TopK、分组及视觉/hash 路由的选择适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/fused_moe/router/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`vllm_ascend.ops.fused_moe.router.fused_topk_router`, `vllm_ascend.ops.fused_moe.router.router_factory` |
| [fused_topk_router.py](../../../vllm_ascend/ops/fused_moe/router/fused_topk_router.py) | 专家路由、TopK 选择 的接口/实现或辅助；所属职责：不同专家评分、TopK、分组及视觉/hash 路由的选择适配。 | 定义：`AscendFusedTopKRouter`, `select_deepseek_v4_vision_experts` |
| [grouped_topk_router.py](../../../vllm_ascend/ops/fused_moe/router/grouped_topk_router.py) | 专家路由、TopK 选择 的接口/实现或辅助；所属职责：不同专家评分、TopK、分组及视觉/hash 路由的选择适配。 | 定义：`AscendGroupedTopKRouter` |
| [router_factory.py](../../../vllm_ascend/ops/fused_moe/router/router_factory.py) | 专家路由 的接口/实现或辅助；所属职责：不同专家评分、TopK、分组及视觉/hash 路由的选择适配。 | 定义：`check_npu_moe_gating_top_k`, `create_ascend_fused_moe_router` |
