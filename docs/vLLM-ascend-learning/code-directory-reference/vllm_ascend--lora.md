# vllm_ascend/lora：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **6 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/lora/`

LoRA 在 NPU 上的参数、算子、MoE 和 Punica 接入。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/lora/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [fused_moe.py](../../../vllm_ascend/lora/fused_moe.py) | 混合专家 的接口/实现或辅助；所属职责：LoRA 在 NPU 上的参数、算子、MoE 和 Punica 接入。 | 注释：Ascend MoE LoRA wrapper and routed-token mapping helpers.；定义：`AscendFusedMoEWithLoRA`, `AscendFusedMoE3DWithLoRA`, `has_lora`, `reset_lora_indices`, `prepare_lora_indices` |
| [lora_ops.py](../../../vllm_ascend/lora/lora_ops.py) | LoRA 适配 的接口/实现或辅助；所属职责：LoRA 在 NPU 上的参数、算子、MoE 和 Punica 接入。 | 定义：`bgmv_shrink`, `bgmv_expand`, `bgmv_expand_slice` |
| [punica_npu.py](../../../vllm_ascend/lora/punica_npu.py) | LoRA 批量执行 的接口/实现或辅助；所属职责：LoRA 在 NPU 上的参数、算子、MoE 和 Punica 接入。 | 定义：`PunicaWrapperNPU` |
| [quant_moe.py](../../../vllm_ascend/lora/quant_moe.py) | 量化、混合专家 的接口/实现或辅助；所属职责：LoRA 在 NPU 上的参数、算子、MoE 和 Punica 接入。 | 注释：Extensible quantized MoE LoRA execution for Ascend.；定义：`QuantMoELoRAImpl`, `register_quant_moe_lora_impl`, `quant_apply_mlp_with_moe_lora`, `validate_quant_moe_lora_activation_input` |
| [utils.py](../../../vllm_ascend/lora/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：LoRA 在 NPU 上的参数、算子、MoE 和 Punica 接入。 | 定义：`_PackedLoRAAWeightsMixin`, `AscendMergedColumnParallelLinearWithLoRA`, `refresh_all_lora_classes` |
