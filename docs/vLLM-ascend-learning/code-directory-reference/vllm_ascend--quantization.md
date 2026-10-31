# vllm_ascend/quantization：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **40 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

按配置解析 → method adapter → scheme 注册与具体实现 → 权重后处理/执行阅读。重点核对权重分片、scale 分组、激活表示与算子 ABI；权重、激活、KV 量化分别对应不同对象。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `vllm_ascend/quantization/`

量化配置解析、层适配、权重后处理与计算方案。

| 子目录 | 职责 |
| --- | --- |
| [configs/](../../../vllm_ascend/quantization/configs) | 不同量化来源/格式的配置解析与方案选择。 |
| [methods/](../../../vllm_ascend/quantization/methods) | Linear、MoE 和 KV 的具体量化参数与执行方法。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/quantization/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 注释：Ascend quantization module.；定义：`__getattr__` |
| [method_adapters.py](../../../vllm_ascend/quantization/method_adapters.py) | 把上游 Linear/MoE/Embedding/KV 量化方法接到 Ascend 方案。 | 定义：`AscendLinearMethod`, `AscendKVCacheMethod` |
| [prepared_linear_input.py](../../../vllm_ascend/quantization/prepared_linear_input.py) | 准备、线性投影、输入 的接口/实现或辅助；所属职责：量化配置解析、层适配、权重后处理与计算方案。 | 注释：Lightweight input contract shared by quantization and MoE modules.；定义：`PreparedLinearInput` |
| [quant_parser.py](../../../vllm_ascend/quantization/quant_parser.py) | 解析量化信息并形成实现所需描述。 | 定义：`QuantTypeMapping`, `get_rollback_quant_type`, `parse_mxfp_quant_params`, `parse_quant_moe_down_proj_params` |
| [quant_type.py](../../../vllm_ascend/quantization/quant_type.py) | 量化类型定义与格式集合。 | 注释：Shared quantization enum definitions.；定义：`QuantType` |
| [utils.py](../../../vllm_ascend/quantization/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：量化配置解析、层适配、权重后处理与计算方案。 | 定义：`is_fused_moe_layer`, `get_dynamic_mx_quant_scale_alg`, `get_model_file` |

## `vllm_ascend/quantization/configs/`

不同量化来源/格式的配置解析与方案选择。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/quantization/configs/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：Quantization configuration classes for Ascend NPU. |
| [compressed_tensors_config.py](../../../vllm_ascend/quantization/configs/compressed_tensors_config.py) | 张量、配置 的接口/实现或辅助；所属职责：不同量化来源/格式的配置解析与方案选择。 | 注释：LLM-Compressor (compressed_tensors) quantization configuration for Ascend.；定义：`AscendCompressedTensorsConfig` |
| [fp8_config.py](../../../vllm_ascend/quantization/configs/fp8_config.py) | 配置 的接口/实现或辅助；所属职责：不同量化来源/格式的配置解析与方案选择。 | 定义：`AscendFp8Config`, `AscendDeepseekV4FP8Config` |
| [modelopt_mxfp8_config.py](../../../vllm_ascend/quantization/configs/modelopt_mxfp8_config.py) | 配置、模型、MXFP 数值格式 的接口/实现或辅助；所属职责：不同量化来源/格式的配置解析与方案选择。 | 定义：`AscendModelOptMxFp8Config` |
| [modelslim_config.py](../../../vllm_ascend/quantization/configs/modelslim_config.py) | 配置、模型 的接口/实现或辅助；所属职责：不同量化来源/格式的配置解析与方案选择。 | 注释：ModelSlim quantization configuration and model mappings for Ascend.；定义：`AscendModelSlimConfig`, `get_quant_type_for_layer`, `create_scheme_for_layer` |

## `vllm_ascend/quantization/methods/`

Linear、MoE 和 KV 的具体量化参数与执行方法。

| 子目录 | 职责 |
| --- | --- |
| [kv_cache/](../../../vllm_ascend/quantization/methods/kv_cache) | KV 缓存量化、scale 与存储表示。 |
| [w4a4/](../../../vllm_ascend/quantization/methods/w4a4) | w4a4 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。 |
| [w4a8/](../../../vllm_ascend/quantization/methods/w4a8) | w4a8 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。 |
| [w8a8/](../../../vllm_ascend/quantization/methods/w8a8) | w8a8 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。 |
| [wna16/](../../../vllm_ascend/quantization/methods/wna16) | wna16 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/quantization/methods/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 注释：Ascend quantization scheme implementations.；定义：`is_mx_quant_type` |
| [base.py](../../../vllm_ascend/quantization/methods/base.py) | Linear/MoE 方案的参数创建、后处理、激活量化与 GMM 抽象接口。 | 注释：Abstract base classes for Ascend quantization schemes.；定义：`AscendLinearScheme`, `AscendAttentionScheme`, `get_moe_num_logical_experts` |
| [registry.py](../../../vllm_ascend/quantization/methods/registry.py) | 注册和查找量化格式与层类型对应的实现类。 | 定义：`register_scheme`, `get_scheme_class` |

## `vllm_ascend/quantization/methods/kv_cache/`

KV 缓存量化、scale 与存储表示。

| 子目录 | 职责 |
| --- | --- |
| [turboquant/](../../../vllm_ascend/quantization/methods/kv_cache/turboquant) | turboquant 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/quantization/methods/kv_cache/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：KV cache quantization methods for Ascend NPU. |
| [kv_c8.py](../../../vllm_ascend/quantization/methods/kv_cache/kv_c8.py) | 实现 kv_c8 量化方案相关参数、后处理或计算方法；所属格式：kv_cache。 | 定义：`AscendFAQuantAttentionMethod`, `AscendSFAQuantAttentionMethod` |
| [mxfp_c8.py](../../../vllm_ascend/quantization/methods/kv_cache/mxfp_c8.py) | 实现 mxfp_c8 量化方案相关参数、后处理或计算方法；所属格式：kv_cache。 | 定义：`AscendC8MXFPKVCacheAttentionMethod` |

## `vllm_ascend/quantization/methods/kv_cache/turboquant/`

turboquant 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/quantization/methods/kv_cache/turboquant/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`is_turboquant` |
| [cache.py](../../../vllm_ascend/quantization/methods/kv_cache/turboquant/cache.py) | 实现 cache 量化方案相关参数、后处理或计算方法；所属格式：turboquant。 | 注释：Pack DeepSeek C4 TQ pages using upstream byte-offset cache descriptors.；定义：`uses_turboquant_groups`, `group_specs`, `packed_layout` |
| [config.py](../../../vllm_ascend/quantization/methods/kv_cache/turboquant/config.py) | 实现 config 量化方案相关参数、后处理或计算方法；所属格式：turboquant。 | 定义：`validate_turboquant` |
| [latent.py](../../../vllm_ascend/quantization/methods/kv_cache/turboquant/latent.py) | 实现 latent 量化方案相关参数、后处理或计算方法；所属格式：turboquant。 | 注释：GLM signed Hadamard/codebook flow with the ops-nn TurboQuant ABI.；定义：`TurboQuantLatent` |

## `vllm_ascend/quantization/methods/w4a4/`

w4a4 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/quantization/methods/w4a4/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：W4A4 quantization methods for Ascend NPU. |
| [w4a4_flatquant.py](../../../vllm_ascend/quantization/methods/w4a4/w4a4_flatquant.py) | 实现 w4a4_flatquant 量化方案相关参数、后处理或计算方法；所属格式：w4a4。 | 定义：`AscendW4A4FlatQuantDynamicLinearMethod`, `pack_int4_weights`, `solve_kronecker_decompose`, `batched_kronecker_quant` |
| [w4a4_laos_dynamic.py](../../../vllm_ascend/quantization/methods/w4a4/w4a4_laos_dynamic.py) | 实现 w4a4_laos_dynamic 量化方案相关参数、后处理或计算方法；所属格式：w4a4。 | 定义：`AscendW4A4LaosDynamicLinearMethod` |
| [w4a4_mxfp4.py](../../../vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4.py) | 实现 w4a4_mxfp4 量化方案相关参数、后处理或计算方法；所属格式：w4a4。 | 定义：`AscendW4A4MXFP4DynamicLinearMethod`, `AscendW4A4MXFP4DynamicFusedMoEMethod` |
| [w4a4_mxfp4_flatquant.py](../../../vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4_flatquant.py) | 实现 w4a4_mxfp4_flatquant 量化方案相关参数、后处理或计算方法；所属格式：w4a4。 | 定义：`AscendW4A4MXFP4FlatQuantDynamicLinearMethod`, `get_decompose_dim` |

## `vllm_ascend/quantization/methods/w4a8/`

w4a8 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/quantization/methods/w4a8/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：W4A8 quantization methods for Ascend NPU. |
| [w4a8.py](../../../vllm_ascend/quantization/methods/w4a8/w4a8.py) | 实现 w4a8 量化方案相关参数、后处理或计算方法；所属格式：w4a8。 | 定义：`AscendW4A8DynamicFusedMoEMethod` |
| [w4a8_mxfp4.py](../../../vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py) | 实现 w4a8_mxfp4 量化方案相关参数、后处理或计算方法；所属格式：w4a8。 | 定义：`AscendW4A8MXFPDynamicLinearMethod`, `AscendW4A8MXFPDynamicFusedMoEMethod` |

## `vllm_ascend/quantization/methods/w8a8/`

w8a8 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/quantization/methods/w8a8/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：W8A8 quantization methods for Ascend NPU. |
| [fp8_block.py](../../../vllm_ascend/quantization/methods/w8a8/fp8_block.py) | 实现 fp8_block 量化方案相关参数、后处理或计算方法；所属格式：w8a8。 | 注释：Schemes for native block-wise FP8 checkpoints (``quant_method: "fp8"``).；定义：`AscendFp8BlockLinearMethod`, `AscendFp8BlockFusedMoEMethod`, `resolve_block_scales` |
| [w8a8_dynamic.py](../../../vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py) | 实现 w8a8_dynamic 量化方案相关参数、后处理或计算方法；所属格式：w8a8。 | 定义：`AscendW8A8DynamicLinearMethod`, `AscendW8A8DynamicFusedMoEMethod`, `scale_from_float_to_int64` |
| [w8a8_mxfp8.py](../../../vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py) | 实现 w8a8_mxfp8 量化方案相关参数、后处理或计算方法；所属格式：w8a8。 | 定义：`AscendW8A8MXFP8DynamicLinearMethod`, `AscendW8A8MXFP8DynamicFusedMoEMethod` |
| [w8a8_pdmix.py](../../../vllm_ascend/quantization/methods/w8a8/w8a8_pdmix.py) | 实现 w8a8_pdmix 量化方案相关参数、后处理或计算方法；所属格式：w8a8。 | 注释：W8A8 Prefill-Decode Mix quantization methods.；定义：`AscendW8A8PDMixLinearMethod` |
| [w8a8_static.py](../../../vllm_ascend/quantization/methods/w8a8/w8a8_static.py) | 实现 w8a8_static 量化方案相关参数、后处理或计算方法；所属格式：w8a8。 | 定义：`AscendW8A8LinearMethod` |
| [w8a8fp8_dynamic.py](../../../vllm_ascend/quantization/methods/w8a8/w8a8fp8_dynamic.py) | 实现 w8a8fp8_dynamic 量化方案相关参数、后处理或计算方法；所属格式：w8a8。 | 定义：`AscendW8A8FP8DynamicLinearMethod`, `AscendW8A8FP8DynamicFusedMoEMethod` |

## `vllm_ascend/quantization/methods/wna16/`

wna16 量化格式的参数、权重后处理和计算实现；支持范围以方案校验为准。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/quantization/methods/wna16/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：WnA16 quantization methods for Ascend NPU (W4A16, W8A16). |
| [w4a16.py](../../../vllm_ascend/quantization/methods/wna16/w4a16.py) | 实现 w4a16 量化方案相关参数、后处理或计算方法；所属格式：wna16。 | 注释：Ascend W4A16 quantization helpers and fused MoE method.；定义：`AscendW4A16FusedMoEMethod`, `unpack_from_int32`, `pack_to_int32` |
| [w4a16_mxfp4.py](../../../vllm_ascend/quantization/methods/wna16/w4a16_mxfp4.py) | 实现 w4a16_mxfp4 量化方案相关参数、后处理或计算方法；所属格式：wna16。 | 注释：Ascend W4A16_MXFP4 quantization helpers and fused MoE method.；定义：`AscendW4A16MXFP4FusedMoEMethod`, `unpack_uint8_to_fp4_return_float32` |
| [w8a16.py](../../../vllm_ascend/quantization/methods/wna16/w8a16.py) | 实现 w8a16 量化方案相关参数、后处理或计算方法；所属格式：wna16。 | 定义：`AscendW8A16LinearMethod` |
