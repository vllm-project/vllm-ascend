# vllm_ascend/models/deepseek_v4：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **9 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/models/deepseek_v4/`

DeepSeek V4 骨干、视觉、压缩、MTP 和 DSpark 实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/deepseek_v4/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [compressor.py](../../../vllm_ascend/models/deepseek_v4/compressor.py) | 长程压缩 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`AscendCompressorStateCache`, `AscendCompressorMetadata` |
| [dspark.py](../../../vllm_ascend/models/deepseek_v4/dspark.py) | DSpark 草稿 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：DeepSeek V4 DSpark draft model for Ascend.；定义：`DeepseekV4DSparkModel`, `DSparkDeepseekV4ForCausalLM` |
| [indexer.py](../../../vllm_ascend/models/deepseek_v4/indexer.py) | 历史位置索引 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`AscendDeepseekV4IndexerCache`, `AscendIndexerMetadata`, `hadamard_linear`, `hadamard_scale`, `rotate_activation` |
| [mm_preprocess.py](../../../vllm_ascend/models/deepseek_v4/mm_preprocess.py) | mm_preprocess 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Multimodal preprocessing for the DeepSeek-V4 vision variants；定义：`DeepseekV4VLImageProcessor`, `DeepseekV4VLProcessor`, `image_sentinel_mask`, `validate_image_sentinel_ids`, `grid_tokens` |
| [model.py](../../../vllm_ascend/models/deepseek_v4/model.py) | 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`AscendDeepseekV4SWACache`, `DeepseekV2MLP`, `precompute_freqs_cis_cpu`, `apply_rotary_emb`, `get_spec_layer_idx_from_weight_name` |
| [mtp.py](../../../vllm_ascend/models/deepseek_v4/mtp.py) | 多 token 预测 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 定义：`SharedHead`, `DeepSeekMultiTokenPredictorLayer` |
| [vision.py](../../../vllm_ascend/models/deepseek_v4/vision.py) | 视觉编码 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：DeepSeek-V4 vision tower (ViT + aligner), replicated (no TP/DP sharding).；定义：`DeepseekV4RMSNorm`, `DeepseekV4PatchEmbed`, `get_vision_cos_sin`, `apply_rotary` |
| [vl_model.py](../../../vllm_ascend/models/deepseek_v4/vl_model.py) | 模型 的模型层、前向/权重接口或模型专有辅助；适用架构以注册表为准。 | 注释：Ascend wrapper for DeepSeek-V4-Flash-Vision-Exp.；定义：`AscendDeepseekV4ForConditionalGeneration` |
