# vllm_ascend/ops：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **26 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/ops/`

模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。

| 子目录 | 职责 |
| --- | --- |
| [fused_moe/](vllm_ascend--ops--fused_moe.md) | MoE 路由、token 分发/合并、专家 FFN、共享专家和通信执行方法。 |
| [triton/](vllm_ascend--ops--triton.md) | 使用 Triton 编写的设备 kernel 与其调用封装。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`dummyFusionOp`, `register_dummy_fusion_op` |
| [activation.py](../../../vllm_ascend/ops/activation.py) | SiLU/GELU/SwiGLU 等激活及门控融合算子适配。 | 定义：`AscendQuickGELU`, `AscendSiluAndMul` |
| [bailing_moe_linear_attn.py](../../../vllm_ascend/ops/bailing_moe_linear_attn.py) | Bailing 模型、线性投影、注意力 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 注释：NPU-friendly OOT replacement for BailingMoELinearAttention.；定义：`AscendBailingMoELinearAttention` |
| [causal_conv1d.py](../../../vllm_ascend/ops/causal_conv1d.py) | 卷积 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 定义：`causal_conv1d_ref`, `causal_conv1d_fn`, `causal_conv1d_update` |
| [conv.py](../../../vllm_ascend/ops/conv.py) | 卷积 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 定义：`AscendConv3dLayer` |
| [cv_linear.py](../../../vllm_ascend/ops/cv_linear.py) | 将 Linear 的量化与矩阵乘阶段拆开包装，供计算重叠使用。 | 定义：`CVLinearWrapper` |
| [dsa.py](../../../vllm_ascend/ops/dsa.py) | DSA 模块集合、缓存构造、custom-op 注册及 Attention 包装。 | 定义：`DSAModules`, `AscendDeepseekSparseAttention`, `dsa_forward`, `dsa_forward_fake`, `unfold_kvcache` |
| [force_topk_sample.py](../../../vllm_ascend/ops/force_topk_sample.py) | 采样、TopK 选择 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 定义：`build_compact_for_logprobs`, `force_topk_sample` |
| [gdn.py](../../../vllm_ascend/ops/gdn.py) | Gated DeltaNet 注意力 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 定义：`AscendGatedDeltaNetAttention`, `initialize_packed_conv_weight` |
| [gdn_attn_builder.py](../../../vllm_ascend/ops/gdn_attn_builder.py) | 注意力、Gated DeltaNet 注意力 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 定义：`GDNChunkedPrefillMetadata`, `GDNCausalConv1dMetadata` |
| [kda.py](../../../vllm_ascend/ops/kda.py) | KDA 线性注意力 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 注释：Shared fla_npu KDA execution; callers own projections and cache updates.；定义：`run_recurrent_kda`, `run_chunk_kda` |
| [kda_state_copy_plan.py](../../../vllm_ascend/ops/kda_state_copy_plan.py) | 状态、复制、KDA 线性注意力 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 注释：Worker-owned prepared plans for PR17301-aligned Kimi prefill state copies.；定义：`KDAStateCopyPlan`, `StridedKDAFallbackPlan`, `supports_kda_state_copy`, `cache_signature`, `initialize_kda_state_copy` |
| [kimi_kda.py](../../../vllm_ascend/ops/kimi_kda.py) | Kimi 模型、KDA 线性注意力 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 注释：Ascend backend for the vLLM 0.27 Kimi K3 delta-attention layer.；定义：`_KDAFusedBFGLinear`, `AscendKimiK3DeltaAttention` |
| [kimi_mla.py](../../../vllm_ascend/ops/kimi_mla.py) | Kimi 模型、MLA 潜在注意力 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 注释：Kimi-specific MLA adapter, imported only for the vLLM main lane.；定义：`AscendKimiK3MultiHeadLatentAttention` |
| [layernorm.py](../../../vllm_ascend/ops/layernorm.py) | RMSNorm/LayerNorm 及相关融合的 NPU 实现。 | 定义：`AscendRMSNorm`, `AscendGemmaRMSNorm` |
| [linear.py](../../../vllm_ascend/ops/linear.py) | 线性投影 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 注释：To customize linear communication groups or forward of classes in this file,；定义：`AscendUnquantizedLinearMethod`, `AscendLinearBase`, `unquantized_gemm`, `unquantized_gemm_fake` |
| [linear_op.py](../../../vllm_ascend/ops/linear_op.py) | 线性投影 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 注释：This file extends the functionality of linear operations by encapsulating custom；定义：`CustomLinearOp`, `CustomColumnParallelOp`, `get_parallel_op`, `get_replicated_op`, `is_moe_layer` |
| [mla.py](../../../vllm_ascend/ops/mla.py) | MLA 潜在注意力 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 定义：`IndexerWrapper`, `AscendMultiHeadLatentAttention`, `mla_forward`, `mla_forward_fake` |
| [mm_encoder_attention.py](../../../vllm_ascend/ops/mm_encoder_attention.py) | 注意力、编码器 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 注释：Ascend implementation of upstream :class:`MMEncoderAttention`.；定义：`AscendMMEncoderAttention` |
| [packaged_attention.py](../../../vllm_ascend/ops/packaged_attention.py) | 注意力、打包 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 注释：Torch dispatcher contracts for the packaged A5 DSV4.1 DSL operators.；定义：`_get_dsl_ops`, `_quant_lightning_indexer_metadata_fake`, `_quant_lightning_indexer_fake` |
| [qwen2_decoder.py](../../../vllm_ascend/ops/qwen2_decoder.py) | 逐步生成、Qwen 模型 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 定义：`AscendCustomQwen2Decoder`, `AscendQwen2Model`, `optimized_apply_rotary_pos_emb` |
| [register_custom_ops.py](../../../vllm_ascend/ops/register_custom_ops.py) | 注册 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 定义：`_get_ep_local_sizes`, `_pad_to_ep_local_size`, `_maybe_all_gather_and_maybe_unpad_impl` |
| [rel_pos_attention.py](../../../vllm_ascend/ops/rel_pos_attention.py) | 注意力 的接口/实现或辅助；所属职责：模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 | 定义：`AscendRelPosAttention` |
| [rope_dsv4.py](../../../vllm_ascend/ops/rope_dsv4.py) | DeepSeek V4 系列复数/分组 RoPE 的位置表与执行。 | 定义：`RopeGlobalState`, `RopeDataProxy`, `get_cos_and_sin_dsa`, `get_full_cos_and_sin_dsa`, `get_full_cos_and_sin_dsa_for_layer` |
| [rotary_embedding.py](../../../vllm_ascend/ops/rotary_embedding.py) | RoPE 位置旋转的设备适配。 | 定义：`AscendRotaryEmbedding`, `AscendYaRNRotaryEmbedding`, `set_cos_and_sin`, `get_cos_and_sin_mla`, `update_cos_sin` |
| [vocab_parallel_embedding.py](../../../vllm_ascend/ops/vocab_parallel_embedding.py) | 词表 embedding/LM Head 的设备与并行适配。 | 定义：`VocabParallelMode`, `VocabParallelPlan`, `lmhead_all_to_all` |
