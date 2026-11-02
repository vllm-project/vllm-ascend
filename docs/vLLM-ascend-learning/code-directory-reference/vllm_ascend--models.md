# vllm_ascend/models：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **11 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/models/`

模型注册、模型专有层结构、权重映射与草稿模型实现。

| 子目录 | 职责 |
| --- | --- |
| [common/](vllm_ascend--models--common.md) | 多个模型复用的执行辅助与序列并行操作。 |
| [deepseek_v4/](vllm_ascend--models--deepseek_v4.md) | DeepSeek V4 骨干、视觉、压缩、MTP 和 DSpark 实现。 |
| [deepseek_v41/](vllm_ascend--models--deepseek_v41.md) | DeepSeek V4.1 骨干、层共享拓扑、视觉、稀疏选择及缓存组织。 |
| [glm5next/](vllm_ascend--models--glm5next.md) | GLM5Next 模型结构和专有计算适配。 |
| [layer/](vllm_ascend--models--layer.md) | layer 子模块；所属范围：模型注册、模型专有层结构、权重映射与草稿模型实现。 |
| [minimax_m3/](vllm_ascend--models--minimax_m3.md) | MiniMax M3 模型及多模态相关适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/__init__.py) | 将上游 architecture 名称注册到本仓模型实现类。 | 定义：`register_model` |
| [deepseek_mtp.py](../../../vllm_ascend/models/deepseek_mtp.py) | DeepSeek 模型、多 token 预测 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`AscendDeepSeekMTP`, `AscendGlmMoeDsaForCausalLM` |
| [gemma4_mm.py](../../../vllm_ascend/models/gemma4_mm.py) | Gemma 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Ascend-specific Gemma4 multimodal model adaptations.；定义：`AscendGemma4ForConditionalGeneration` |
| [glm5_dspark.py](../../../vllm_ascend/models/glm5_dspark.py) | DSpark 草稿 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：GLM MLA backbone adapter for the upstream DSpark generation contracts.；定义：`Glm5DSparkMLAAttention`, `Glm5DSparkDecoderLayer` |
| [kimi_k3.py](../../../vllm_ascend/models/kimi_k3.py) | Kimi 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Kimi K3 model adapters for vLLM 0.27 on Ascend.；定义：`AscendKimiRoutedOutputTransform`, `AscendKimiMLP` |
| [kimi_k3_dspark.py](../../../vllm_ascend/models/kimi_k3_dspark.py) | DSpark 草稿、Kimi 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Kimi K3 MLA DSpark draft model for Ascend.；定义：`AscendK3DSparkDecoderLayer`, `AscendK3DSparkModel` |
| [kimi_k3_mtp.py](../../../vllm_ascend/models/kimi_k3_mtp.py) | Kimi 模型、多 token 预测 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Kimi K3 MTP draft model for Ascend.；定义：`AscendKimiK3MultiTokenPredictorLayer`, `AscendKimiK3MultiTokenPredictor` |
| [llama_eagle3.py](../../../vllm_ascend/models/llama_eagle3.py) | Eagle 投机、Llama 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`AscendEagle3LlamaForCausalLM`, `get_embedding_tensor`, `load_quarot_target_layer`, `compute_rotation_matrix3` |
| [llama_eagle3_vwn.py](../../../vllm_ascend/models/llama_eagle3_vwn.py) | Eagle 投机、Llama 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`PreVwnLayerV1`, `VwnLlamaDecoderLayer` |
| [qwen3_dflash2.py](../../../vllm_ascend/models/qwen3_dflash2.py) | DFlash 草稿、Qwen 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`DFlashGroupedConv`, `DFlash2Qwen3DecoderLayer` |
| [qwen3_dspark.py](../../../vllm_ascend/models/qwen3_dspark.py) | DSpark 草稿、Qwen 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`AscendQwen3DSparkForCausalLM`, `process_weight`, `align_draft_weights` |
