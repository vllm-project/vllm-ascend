# vllm_ascend/models/deepseek_v41：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **17 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

先看模型注册和 `vl_model.py`，再看 `model.py` 的 layer plan、Decoder 与 Model forward。ViT、Engram、Compressor、Indexer、mHC 和 DSpark 并非每步全部执行：源层角色、请求类型与配置共同决定。Attention 与缓存实现跨出本目录，见 `attention/dsa_v41.py`。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `vllm_ascend/models/deepseek_v41/`

DeepSeek V4.1 骨干、层共享拓扑、视觉、稀疏选择及缓存组织。

| 子目录 | 职责 |
| --- | --- |
| [engram/](../../../vllm_ascend/models/deepseek_v41/engram) | Engram 表加载、历史 hash、分片查询、门控融合及生命周期。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/deepseek_v41/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：DeepSeek V4.1 construction components (not yet a runnable model). |
| [cache_config.py](../../../vllm_ascend/models/deepseek_v41/cache_config.py) | V4.1 各缓存平面的 dtype/字节布局、混合页分组、padding 与分配规则。 | 注释：Framework-side V4.1 layer-outermost cache placement and allocation.；定义：`uses_a5_packed_cache`, `make_swa_cache_spec`, `make_mla_cache_spec` |
| [compressor.py](../../../vllm_ascend/models/deepseek_v41/compressor.py) | ratio-1 投影和 ratio-2 门控 ring pooling，维护压缩状态与持久输出。 | 注释：FP32 C2 ring compressor, ratio-1 path, and fused RMS normalization.；定义：`DeepseekV41Compressor` |
| [dspark.py](../../../vllm_ascend/models/deepseek_v41/dspark.py) | V4.1 草稿网络、辅助状态融合、上下文 KV、Markov 分支和 checkpoint 映射。 | 注释：DeepSeek V4.1 dSPark draft model for Ascend.；定义：`DeepseekV41DSparkModel`, `DSparkDeepseekV41ForCausalLM` |
| [indexer.py](../../../vllm_ascend/models/deepseek_v41/indexer.py) | 从 latent 写索引 K，并生成查询/head 权重，执行候选 block 与 TopK 选择。 | 注释：V4.1 index projections, quantized QLI and cross-layer candidate selection.；定义：`DeepseekV41Indexer` |
| [mixed_quant_attention.py](../../../vllm_ascend/models/deepseek_v41/mixed_quant_attention.py) | A5 V4.1 packed-cache 读写、QLI 索引和 QSMLA 计算的混合量化 ABI 适配。 | 注释：Mixed-quant cache and attention ABI selected by the A5 device adaptor.；定义：`MixedQuantPackedCacheOps`, `run_a5_indexer`, `build_smla_metadata`, `qsmla` |
| [model.py](../../../vllm_ascend/models/deepseek_v41/model.py) | V4.1 骨干、Q/KV/输出投影、层共享拓扑、MoE、mHC、Engram 集成及权重映射。 | 注释：DeepSeek V4.1 text model and source-shared hybrid-cache graph.；定义：`DeepseekV41Model`, `DeepseekV41DecoderLayer`, `DeepseekV41MoE`, `get_spec_layer_idx_from_weight_name`, `init_attention_projections`, `build_layer_plan` |
| [vision.py](../../../vllm_ascend/models/deepseek_v41/vision.py) | ViT patch 投影、二维 RoPE、视觉 Attention/MLP、空间合并与 Aligner。 | 注释：Ascend DeepSeek vision tower (ViT + aligner), replicated (no TP/DP sharding).；定义：`DeepseekV41PatchEmbed`, `DeepseekV41VisionAttention`, `get_vision_cos_sin` |
| [vl_model.py](../../../vllm_ascend/models/deepseek_v41/vl_model.py) | V4.1 包装层：构建视觉/文本模型，处理图像 embeddings 与输入/权重接口。 | 注释：Ascend multimodal wrapper for DeepSeek V4.1.；定义：`AscendDeepseekV41ForCausalLM` |

## `vllm_ascend/models/deepseek_v41/engram/`

Engram 表加载、历史 hash、分片查询、门控融合及生命周期。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/models/deepseek_v41/engram/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：Ascend Engram for DeepSeek V4.1. |
| [common.py](../../../vllm_ascend/models/deepseek_v41/engram/common.py) | Engram 门控参考逻辑、旋转基处理与有效 token 辅助。 | 注释：Gating and token-eligibility helpers for the Ascend Engram port.；定义：`load_engram_rotation_block`, `engram_enabled`, `valid_engram_token_mask` |
| [embedding.py](../../../vllm_ascend/models/deepseek_v41/engram/embedding.py) | Engram 表分片、BF16/INT8/MXFP8 存储、权重加载与查询。 | 注释：Ascend head-sharded Engram tables with BF16, INT8 or native MXFP8 storage.；定义：`AscendParallelEngramEmbedding`, `engram_storage_dtype`, `preflight_engram_checkpoint` |
| [hash_state.py](../../../vllm_ascend/models/deepseek_v41/engram/hash_state.py) | 原始 token 历史的 n-gram hash 与设备槽位状态接入。 | 注释：Ascend KV-slot adapter for the upstream Engram hash state.；定义：`AscendEngramSlotCache`, `AscendNgramHashState`, `engram_dead_mask`, `create_engram_hash_state` |
| [layer.py](../../../vllm_ascend/models/deepseek_v41/engram/layer.py) | 把 Engram lookup rows 投影为 key/value，并门控融合到多路 hidden states。 | 注释：Engram projection and gate for BF16 and rotated Ascend checkpoints.；定义：`AscendEngram` |
| [model_state.py](../../../vllm_ascend/models/deepseek_v41/engram/model_state.py) | V2 Engram lookback、图输入、forward 前准备与 lookup 退役。 | 定义：`EngramModelState` |
| [npu.py](../../../vllm_ascend/models/deepseek_v41/engram/npu.py) | 注册主机 UVA/共享表存储，提供表读取与反量化适配。 | 注释：NPU-side Engram storage, routing and lookup.；定义：`HostUvaBuffer`, `SharedUvaBuffer`, `engram_cpu_offload`, `quantize_engram_rows`, `dequantize_engram_rows` |
| [parallel.py](../../../vllm_ascend/models/deepseek_v41/engram/parallel.py) | Engram head 分片、DP/本机组选择及查询结果交换。 | 注释：Node-local Engram exchange with A5 recompute-decode fixed token slots.；定义：`resolve_dp_shared_memory`, `engram_gathered_num_tokens`, `gather_engram_hashes` |
