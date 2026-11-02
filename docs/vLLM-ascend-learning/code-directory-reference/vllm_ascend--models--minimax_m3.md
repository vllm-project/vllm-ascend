# vllm_ascend/models/minimax_m3：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **8 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/models/minimax_m3/`

MiniMax M3 模型及多模态相关适配。

| 子目录 | 职责 |
| --- | --- |
| [ops/](../../../vllm_ascend/models/minimax_m3/ops) | ops 子模块；所属范围：MiniMax M3 模型及多模态相关适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/minimax_m3/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`vllm_ascend.models.minimax_m3.minimax_m3`, `vllm_ascend.models.minimax_m3.minimax_m3_vl` |
| [minimax_m3.py](../../../vllm_ascend/models/minimax_m3/minimax_m3.py) | MiniMax 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Inference-only MiniMaxM3 model.；定义：`MiniMaxM3SparseAttention`, `MiniMaxM3SwiGLUOAI`, `get_spec_layer_idx_from_weight_name` |
| [minimax_m3_vl.py](../../../vllm_ascend/models/minimax_m3/minimax_m3_vl.py) | MiniMax 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：MiniMax M3 multimodal wrapper for Ascend.；定义：`MiniMaxM3VLModel`, `MiniMaxM3SparseForConditionalGeneration` |
| [msa_m3.py](../../../vllm_ascend/models/minimax_m3/msa_m3.py) | msa_m3 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：MiniMax-M3 sparse attention and indexer backends for Ascend.；定义：`AscendMiniMaxM3IndexerBackend`, `AscendMiniMaxM3IndexerCache`, `minimax_m3_sparse_forward`, `minimax_m3_sparse_forward_fake`, `minimax_m3_fused_sparse_forward` |

## `vllm_ascend/models/minimax_m3/ops/`

ops 子模块；所属范围：MiniMax M3 模型及多模态相关适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/minimax_m3/ops/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [msa_m3_npu.py](../../../vllm_ascend/models/minimax_m3/ops/msa_m3_npu.py) | msa_m3_npu 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：NPU sparse attention ops for MiniMax-M3 on Ascend.；定义：`minimax_m3_index_prefill`, `minimax_m3_index_decode`, `minimax_m3_sparse_attn` |
| [msa_m3_triton.py](../../../vllm_ascend/models/minimax_m3/ops/msa_m3_triton.py) | msa_m3_triton 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Triton kernels for MiniMax M3 block-sparse GQA attention on Ascend.；定义：`minimax_m3_index_score`, `minimax_m3_index_topk`, `minimax_m3_index_decode` |
| [msa_m3_triton_a5.py](../../../vllm_ascend/models/minimax_m3/ops/msa_m3_triton_a5.py) | msa_m3_triton_a5 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Triton kernels for MiniMax M3 block-sparse GQA attention on Ascend.；定义：`minimax_m3_index_score`, `minimax_m3_index_topk`, `minimax_m3_index_decode` |
