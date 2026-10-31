# vllm_ascend/compilation：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **15 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

按平台配置选择 → 编译变换 → 捕获/重放 → 参数更新阅读；同时核对模型输入 buffer、Attention update 和异步事件。固定地址与数据已经就绪是两个独立条件。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `vllm_ascend/compilation/`

NPU 图捕获、重放、运行参数更新与编译融合接入。

| 子目录 | 职责 |
| --- | --- |
| [passes/](../../../vllm_ascend/compilation/passes) | 识别并变换计算图，例如 Norm/量化融合和无效操作消除。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/compilation/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [acl_graph.py](../../../vllm_ascend/compilation/acl_graph.py) | ACLGraph 图条目、捕获、重放与运行参数更新组织。 | 定义：`ACLGraphEntry`, `ACLGraphWrapper`, `weak_ref_workspaces`, `update_full_graph_params`, `set_graph_params` |
| [breakable_aclgraph.py](../../../vllm_ascend/compilation/breakable_aclgraph.py) | 分段/可中断图执行的 NPU 包装。 | 定义：`BreakableACLGraphWrapper` |
| [compiler_interface.py](../../../vllm_ascend/compilation/compiler_interface.py) | 向上游编译接口接入 Ascend 编译与融合过程。 | 定义：`AscendCompiler`, `compile_fx`, `fusion_pass_compile`, `npugraph_ex_compile` |
| [graph_fusion_pass_manager.py](../../../vllm_ascend/compilation/graph_fusion_pass_manager.py) | 管理图优化 pass 及执行顺序。 | 定义：`GraphFusionPassManager` |
| [updatable_graph.py](../../../vllm_ascend/compilation/updatable_graph.py) | 参数 provider/source、GraphUpdateTask 和可更新图资源。 | 定义：`ParamProvider`, `ParamSource`, `register_task`, `get_capture_resource` |

## `vllm_ascend/compilation/passes/`

识别并变换计算图，例如 Norm/量化融合和无效操作消除。

| 子目录 | 职责 |
| --- | --- |
| [utils/](../../../vllm_ascend/compilation/passes/utils) | 辅助函数 子模块；所属范围：识别并变换计算图，例如 Norm/量化融合和无效操作消除。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/compilation/passes/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [allgather_chunk_noop_pass.py](../../../vllm_ascend/compilation/passes/allgather_chunk_noop_pass.py) | 张量收集 的接口/实现或辅助；所属职责：识别并变换计算图，例如 Norm/量化融合和无效操作消除。 | 定义：`AllGatherChunkNoOpCleanupPass` |
| [base_pattern.py](../../../vllm_ascend/compilation/passes/base_pattern.py) | 定义图模式匹配与替换使用的公共模式接口。 | 定义：`BasePattern` |
| [muls_add_pass.py](../../../vllm_ascend/compilation/passes/muls_add_pass.py) | 识别乘常数与加法组合并实施相应图融合。 | 定义：`MulsAddPattern`, `MulsAddFusionPass` |
| [noop_elimination.py](../../../vllm_ascend/compilation/passes/noop_elimination.py) | 识别并消除图中不改变结果的冗余操作。 | 定义：`NoOpEliminationPass` |
| [norm_quant_fusion_pass.py](../../../vllm_ascend/compilation/passes/norm_quant_fusion_pass.py) | 算子融合、量化 的接口/实现或辅助；所属职责：识别并变换计算图，例如 Norm/量化融合和无效操作消除。 | 定义：`AddRMSNormQuantPattern`, `AddRMSNormQuantPatternWithBias` |
| [qknorm_rope_fusion_pass.py](../../../vllm_ascend/compilation/passes/qknorm_rope_fusion_pass.py) | 算子融合、位置旋转 的接口/实现或辅助；所属职责：识别并变换计算图，例如 Norm/量化融合和无效操作消除。 | 定义：`QKNormRopeFusionPattern`, `QKNormRopeFusionPatternWithBias` |

## `vllm_ascend/compilation/passes/utils/`

辅助函数 子模块；所属范围：识别并变换计算图，例如 Norm/量化融合和无效操作消除。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/compilation/passes/utils/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [npugraph_ex_utils_check.py](../../../vllm_ascend/compilation/passes/utils/npugraph_ex_utils_check.py) | 设备图、辅助函数、检查 的接口/实现或辅助；所属职责：识别并变换计算图，例如 Norm/量化融合和无效操作消除。 | 定义：`extra_stream_scope_check`, `check_and_register_fusion_pass` |
