# vllm_ascend/ops/triton：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **153 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/ops/triton/`

使用 Triton 编写的设备 kernel 与其调用封装。

| 子目录 | 职责 |
| --- | --- |
| [activation/](../../../vllm_ascend/ops/triton/activation) | 激活函数 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。 |
| [batch_invariant/](../../../vllm_ascend/ops/triton/batch_invariant) | 批次不变性模式所需的计算实现。 |
| [compressor/](../../../vllm_ascend/ops/triton/compressor) | 长程表示投影、加权池化与循环状态相关 kernel。 |
| [dcp/](../../../vllm_ascend/ops/triton/dcp) | Decode 上下文并行所需的索引和张量组织 kernel。 |
| [docs/](../../../vllm_ascend/ops/triton/docs) | docs 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。 |
| [fla/](../../../vllm_ascend/ops/triton/fla) | 线性注意力相关分块/递归设备计算。 |
| [kda/](../../../vllm_ascend/ops/triton/kda) | KDA 线性注意力与状态更新相关 kernel。 |
| [linearnorm/](../../../vllm_ascend/ops/triton/linearnorm) | Linear/Norm 相关设备计算与融合辅助。 |
| [mamba/](../../../vllm_ascend/ops/triton/mamba) | Mamba/选择性状态空间模型相关 kernel。 |
| [msa/](../../../vllm_ascend/ops/triton/msa) | MSA 相关位置选择与状态/元数据 kernel。 |
| [spec_decode/](../../../vllm_ascend/ops/triton/spec_decode) | 投机候选、验证输入和序列推进相关 kernel。 |
| [v2/](../../../vllm_ascend/ops/triton/v2) | 配合 V2 执行的输入、采样和缓存设备 kernel。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [batch_memcpy.py](../../../vllm_ascend/ops/triton/batch_memcpy.py) | 把一批内存复制任务组织成设备 kernel，减少逐项复制的调用组织。 | 定义：`batch_memcpy_kernel` |
| [bincount.py](../../../vllm_ascend/ops/triton/bincount.py) | 统计 token 分桶数量并生成对应 mask。 | 定义：`token_bin_counts_and_mask_kernel`, `get_token_bin_counts_and_mask_triton` |
| [build_window_indices.py](../../../vllm_ascend/ops/triton/build_window_indices.py) | build_window_indices 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Build fixed-width causal window indices in one NPU launch.；定义：`build_window_indices_triton` |
| [c2_ring_metadata.py](../../../vllm_ascend/ops/triton/c2_ring_metadata.py) | 元数据 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Publish C2 ring controls and source RoPE rows in one NPU launch.；定义：`build_c2_ring_metadata` |
| [compute_slot_mapping.md](../../../vllm_ascend/ops/triton/compute_slot_mapping.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：compute_slot_mapping。 | 标题：`_compute_slot_mapping_kernel` 算子文档 |
| [compute_slot_mapping.py](../../../vllm_ascend/ops/triton/compute_slot_mapping.py) | 槽位 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`compute_slot_mapping_fused_groups` |
| [dsa_cp.py](../../../vllm_ascend/ops/triton/dsa_cp.py) | DeepSeek 稀疏注意力、上下文并行 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`build_local_metadata_triton` |
| [engram_gate.py](../../../vllm_ascend/ops/triton/engram_gate.py) | Engram 记忆、门控 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Fused post-wkv gate for native (unrotated) A5 checkpoints.；定义：`fused_engram_gate` |
| [engram_hash.py](../../../vllm_ascend/ops/triton/engram_hash.py) | Engram 记忆、哈希映射 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Engram hash kernels; model code owns buffers and launch scheduling.；定义：`_engram_req_index_kernel`, `_hash_ids_kernel` |
| [engram_lookback.py](../../../vllm_ascend/ops/triton/engram_lookback.py) | 前序 token 历史、Engram 记忆 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Engram lookback kernels; model code owns buffers and launch scheduling.；定义：`_gather_lookback_kernel` |
| [engram_lookup.py](../../../vllm_ascend/ops/triton/engram_lookup.py) | Engram 记忆 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Engram lookup kernels; model code owns buffers and launch scheduling.；定义：`_decode_e8m0`, `_engram_int8_gather_dequant_kernel`, `_engram_host_uva_gather_dequant_kernel` |
| [eplb.py](../../../vllm_ascend/ops/triton/eplb.py) | 专家负载均衡 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`map_to_physical_triton`, `record_expert_tokens_triton` |
| [fold_indexer_cache.py](../../../vllm_ascend/ops/triton/fold_indexer_cache.py) | 历史位置索引、缓存 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Maintain the 8-token fused K+scale view required by A5 QSLI.；定义：`fold_indexer_cache_rows` |
| [fused_gdn_gating.py](../../../vllm_ascend/ops/triton/fused_gdn_gating.py) | 门控、Gated DeltaNet 注意力 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`fused_gdn_gating_kernel`, `fused_gdn_gating_patch` |
| [glm5_next_kpool_tail_compress.py](../../../vllm_ascend/ops/triton/glm5_next_kpool_tail_compress.py) | 池管理 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Triton fast path for the GLM5 Next indexer pre-compress sequence.；定义：`glm5_next_kpool_tail_compress_and_write_cache_triton` |
| [glm5_next_lightning_indexer.py](../../../vllm_ascend/ops/triton/glm5_next_lightning_indexer.py) | 历史位置索引 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Paged GLM KPool scoring and fused token-index expansion.；定义：`glm5_next_lightning_indexer_triton` |
| [kda_state_copy_kernel.py](../../../vllm_ascend/ops/triton/kda_state_copy_kernel.py) | 状态、复制、KDA 线性注意力 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：KDA state-copy kernel and metadata validation for worker-owned plans.；定义：`_configuration`, `_kda_state_copy_kernel`, `_validate_cache_layout` |
| [layernorm_gated.py](../../../vllm_ascend/ops/triton/layernorm_gated.py) | 归一化、门控 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`layer_norm_fwd_npu` |
| [mul_add.py](../../../vllm_ascend/ops/triton/mul_add.py) | mul_add 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`muls_add_kernel`, `muls_add_triton` |
| [muls_add.py](../../../vllm_ascend/ops/triton/muls_add.py) | muls_add 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`muls_add_kernel`, `muls_add_triton` |
| [packed_cache_slot_mapping.py](../../../vllm_ascend/ops/triton/packed_cache_slot_mapping.py) | 缓存、槽位、打包 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Build packed cache slot metadata in one NPU launch.；定义：`build_packed_cache_slot_mapping` |
| [pcp_kv_cache.py](../../../vllm_ascend/ops/triton/pcp_kv_cache.py) | 缓存、Prefill 上下文并行、KV 缓存 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：PCP KV cache transfer helpers.；定义：`copy_pcp_kv_cache` |
| [penalty.py](../../../vllm_ascend/ops/triton/penalty.py) | 采样惩罚 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`apply_all_penalties_kernel`, `apply_penalties_triton` |
| [prepare_indexer_indices.py](../../../vllm_ascend/ops/triton/prepare_indexer_indices.py) | 历史位置索引、准备 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Convert score-ordered QLI indices into visible, chronological positions.；定义：`prepare_indexer_indices` |
| [quantize_indexer_query.py](../../../vllm_ascend/ops/triton/quantize_indexer_query.py) | 量化、历史位置索引 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Per-head INT8 query quantization with the FP16 scales consumed by QLI.；定义：`quantize_indexer_query` |
| [quantize_mxfp4_indexer.py](../../../vllm_ascend/ops/triton/quantize_mxfp4_indexer.py) | 量化、历史位置索引、MXFP 数值格式 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Fused MXFP4 query quantization for the DeepSeek V4.1 A5 indexer.；定义：`quantize_mxfp4_indexer`, `write_mxfp4_indexer_cache` |
| [query_gather_prep.py](../../../vllm_ascend/ops/triton/query_gather_prep.py) | 收集 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`prep_query_head_major` |
| [reject_sample.py](../../../vllm_ascend/ops/triton/reject_sample.py) | 采样 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`cal_grid_and_block_size`, `rejection_greedy_sample_spec_len_1_triton`, `bonus_renew` |
| [rms_norm.py](../../../vllm_ascend/ops/triton/rms_norm.py) | RMS 归一化 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`triton_rms_kernel`, `triton_q_rms` |
| [rope.py](../../../vllm_ascend/ops/triton/rope.py) | 位置旋转 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`rope_forward_triton`, `rope_forward_triton_siso` |
| [sparse_index_remap.py](../../../vllm_ascend/ops/triton/sparse_index_remap.py) | sparse_index_remap 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`remap_sparse_indices_fused_kernel`, `remap_sparse_indices_compact_gather_kernel`, `remap_sparse_indices_triton` |
| [triton_utils.py](../../../vllm_ascend/ops/triton/triton_utils.py) | 辅助函数 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`init_device_properties_triton`, `get_aicore_num`, `get_vectorcore_num` |
| [turboquant_finalize.py](../../../vllm_ascend/ops/triton/turboquant_finalize.py) | 收尾、量化 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Correct the TurboQuant scale and write compact cache rows in one launch.；定义：`turboquant_finalize` |

## `vllm_ascend/ops/triton/activation/`

激活函数 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/activation/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [swiglu_quant.py](../../../vllm_ascend/ops/triton/activation/swiglu_quant.py) | SwiGLU 激活、量化 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`swiglu_quant` |
| [swiglustep.py](../../../vllm_ascend/ops/triton/activation/swiglustep.py) | SwiGLU 激活 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Triton kernel for SwigluStepAndMul activation.；定义：`swiglustep_forward_triton` |

## `vllm_ascend/ops/triton/batch_invariant/`

批次不变性模式所需的计算实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/batch_invariant/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [matmul.py](../../../vllm_ascend/ops/triton/batch_invariant/matmul.py) | 矩阵乘 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`matmul_bias_persistent_kernel`, `matmul_persistent`, `linear_persistent_kernel` |
| [mean.py](../../../vllm_ascend/ops/triton/batch_invariant/mean.py) | mean 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`mean_kernel`, `mean_dim`, `mean_batch_invariant` |
| [rmsnorm.py](../../../vllm_ascend/ops/triton/batch_invariant/rmsnorm.py) | rmsnorm 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`rms_norm`, `rms_norm_batch_invariant` |
| [softmax.py](../../../vllm_ascend/ops/triton/batch_invariant/softmax.py) | 概率归一化 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`softmax_batch_invariant` |

## `vllm_ascend/ops/triton/compressor/`

长程表示投影、加权池化与循环状态相关 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/compressor/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [compressor_triton.py](../../../vllm_ascend/ops/triton/compressor/compressor_triton.py) | 长程压缩 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Compressor 算子 triton 实现。；定义：`_MetaUploadRing`, `compressor`, `compressor_ref`, `compressor_from_projected` |

## `vllm_ascend/ops/triton/dcp/`

Decode 上下文并行所需的索引和张量组织 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/dcp/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [dcp_a2a.py](../../../vllm_ascend/ops/triton/dcp/dcp_a2a.py) | Decode 上下文并行 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`pack_dcp_output_lse`, `fused_dcp_lse_combine`, `dcp_a2a_fused_combine` |
| [dcp_a2a_batched.py](../../../vllm_ascend/ops/triton/dcp/dcp_a2a_batched.py) | 批次、Decode 上下文并行 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Batch independent DCP rows while keeping DCP rank accumulation sequential.；定义：`_pack_dcp_output_lse_batched_kernel`, `_fused_dcp_lse_combine_batched_kernel` |

## `vllm_ascend/ops/triton/docs/`

docs 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。

| 子目录 | 职责 |
| --- | --- |
| [v2/](../../../vllm_ascend/ops/triton/docs/v2) | v2 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [apply_penalties.md](../../../vllm_ascend/ops/triton/docs/apply_penalties.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：apply_penalties。 | 标题：apply_penalties |
| [conv_state.md](../../../vllm_ascend/ops/triton/docs/conv_state.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：conv_state。 | 标题：copy_conv_state |
| [dcp_a2a_fused.md](../../../vllm_ascend/ops/triton/docs/dcp_a2a_fused.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：dcp_a2a_fused。 | 标题：DCP fused All-to-All and LSE combine |
| [dsv41_cache.md](../../../vllm_ascend/ops/triton/docs/dsv41_cache.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：dsv41_cache。 | 标题：DeepSeek V4.1 cache preparation |
| [engram.md](../../../vllm_ascend/ops/triton/docs/engram.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：engram。 | 标题：Engram kernels |
| [fused_dcp_lse_combine.md](../../../vllm_ascend/ops/triton/docs/fused_dcp_lse_combine.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：fused_dcp_lse_combine。 | 标题：fused_dcp_lse_combine |
| [fused_eh_norm.md](../../../vllm_ascend/ops/triton/docs/fused_eh_norm.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：fused_eh_norm。 | 标题：fused_eh_norm |
| [fused_gdn_gating.md](../../../vllm_ascend/ops/triton/docs/fused_gdn_gating.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：fused_gdn_gating。 | 标题：fused_gdn_gating |
| [fused_qkvzba_split_reshape.md](../../../vllm_ascend/ops/triton/docs/fused_qkvzba_split_reshape.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：fused_qkvzba_split_reshape。 | 标题：fused_qkvzba_split_reshape_cat |
| [gather_initial_states.md](../../../vllm_ascend/ops/triton/docs/gather_initial_states.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：gather_initial_states。 | 标题：gather_initial_states |
| [glm5_next_kpool_tail_compress.md](../../../vllm_ascend/ops/triton/docs/glm5_next_kpool_tail_compress.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：glm5_next_kpool_tail_compress。 | 标题：GLM5 Next KPool tail compression |
| [glm5_next_lightning_indexer.md](../../../vllm_ascend/ops/triton/docs/glm5_next_lightning_indexer.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：glm5_next_lightning_indexer。 | 标题：glm5_next_lightning_indexer |
| [kda_state_copy.md](../../../vllm_ascend/ops/triton/docs/kda_state_copy.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：kda_state_copy。 | 标题：KDA state copy: PR #17301-aligned Triton integration |
| [l2norm.md](../../../vllm_ascend/ops/triton/docs/l2norm.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：l2norm。 | 标题：l2norm_fwd_kernel2_loop |
| [lightning_attn.md](../../../vllm_ascend/ops/triton/docs/lightning_attn.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：lightning_attn。 | 标题：LightningAttention |
| [linear_persistent.md](../../../vllm_ascend/ops/triton/docs/linear_persistent.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：linear_persistent。 | 标题：linear_persistent |
| [lora_linear.md](../../../vllm_ascend/ops/triton/docs/lora_linear.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：lora_linear。 | 标题：lora_linear |
| [mamba_state_index.md](../../../vllm_ascend/ops/triton/docs/mamba_state_index.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：mamba_state_index。 | 标题：Mamba State Gather and Scatter |
| [matmul_bias_persistent.md](../../../vllm_ascend/ops/triton/docs/matmul_bias_persistent.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：matmul_bias_persistent。 | 标题：matmul_bias_persistent |
| [mean.md](../../../vllm_ascend/ops/triton/docs/mean.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：mean。 | 标题：mean |
| [min_p.md](../../../vllm_ascend/ops/triton/docs/min_p.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：min_p。 | 标题：_min_p_kernel |
| [minimax_qknorm_rope_cache.md](../../../vllm_ascend/ops/triton/docs/minimax_qknorm_rope_cache.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：minimax_qknorm_rope_cache。 | 标题：MiniMax-M3 QK norm, RoPE and KV cache insertion |
| [msa_m3_index_postprocess.md](../../../vllm_ascend/ops/triton/docs/msa_m3_index_postprocess.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：msa_m3_index_postprocess。 | 标题：MSA index decode TopK postprocessing |
| [output_writeback.md](../../../vllm_ascend/ops/triton/docs/output_writeback.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：output_writeback。 | 标题：write_recurrent_output |
| [pack_dcp_output_lse.md](../../../vllm_ascend/ops/triton/docs/pack_dcp_output_lse.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：pack_dcp_output_lse。 | 标题：pack_dcp_output_lse |
| [pcp_kv_cache.md](../../../vllm_ascend/ops/triton/docs/pcp_kv_cache.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：pcp_kv_cache。 | 标题：copy_pcp_kv_cache |
| [postprocess_mamba.md](../../../vllm_ascend/ops/triton/docs/postprocess_mamba.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：postprocess_mamba。 | 标题：postprocess_mamba |
| [rms_norm.md](../../../vllm_ascend/ops/triton/docs/rms_norm.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：rms_norm。 | 标题：rms_norm |
| [rope.md](../../../vllm_ascend/ops/triton/docs/rope.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：rope。 | 标题：_triton_rope |
| [rope_siso.md](../../../vllm_ascend/ops/triton/docs/rope_siso.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：rope_siso。 | 标题：_triton_rope_siso |
| [sparse_index_remap.md](../../../vllm_ascend/ops/triton/docs/sparse_index_remap.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：sparse_index_remap。 | 标题：sparse_index_remap |
| [split_qkv_rmsnorm_rope_vnorm.md](../../../vllm_ascend/ops/triton/docs/split_qkv_rmsnorm_rope_vnorm.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：split_qkv_rmsnorm_rope_vnorm。 | 标题：split_qkv_rmsnorm_rope_vnorm |
| [swiglustep.md](../../../vllm_ascend/ops/triton/docs/swiglustep.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：swiglustep。 | 标题：swiglustep |
| [topk_log_softmax.md](../../../vllm_ascend/ops/triton/docs/topk_log_softmax.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：topk_log_softmax。 | 标题：topk_log_softmax |
| [triton_description_specification.md](../../../vllm_ascend/ops/triton/docs/triton_description_specification.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：triton_description_specification。 | 标题：Operator name |
| [turboquant_finalize.md](../../../vllm_ascend/ops/triton/docs/turboquant_finalize.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：turboquant_finalize。 | 标题：turboquant_finalize |

## `vllm_ascend/ops/triton/docs/v2/`

v2 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。

| 子目录 | 职责 |
| --- | --- |
| [block_table/](../../../vllm_ascend/ops/triton/docs/v2/block_table) | 缓存页/分块 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。 |
| [sample/](../../../vllm_ascend/ops/triton/docs/v2/sample) | 采样 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。 |
| [spec_decode/](../../../vllm_ascend/ops/triton/docs/v2/spec_decode) | 投机解码 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。 |

## `vllm_ascend/ops/triton/docs/v2/block_table/`

缓存页/分块 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [compute_slot_mappings.md](../../../vllm_ascend/ops/triton/docs/v2/block_table/compute_slot_mappings.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：compute_slot_mappings。 | 标题：`_compute_slot_mappings_kernel` |

## `vllm_ascend/ops/triton/docs/v2/sample/`

采样 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [categorical_sample.md](../../../vllm_ascend/ops/triton/docs/v2/sample/categorical_sample.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：categorical_sample。 | 标题：categorical_sample |
| [sampling_mask.md](../../../vllm_ascend/ops/triton/docs/v2/sample/sampling_mask.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：sampling_mask。 | 标题：Sampling-mask packing (`SamplingMaskTensors.from_logits`) |
| [thinking_budget.md](../../../vllm_ascend/ops/triton/docs/v2/sample/thinking_budget.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：thinking_budget。 | 标题：Thinking-budget compatibility kernels |

## `vllm_ascend/ops/triton/docs/v2/spec_decode/`

投机解码 子模块；所属范围：使用 Triton 编写的设备 kernel 与其调用封装。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [prepare_dflash_inputs.md](../../../vllm_ascend/ops/triton/docs/v2/spec_decode/prepare_dflash_inputs.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：prepare_dflash_inputs。 | 标题：prepare_dflash_inputs |
| [resample.md](../../../vllm_ascend/ops/triton/docs/v2/spec_decode/resample.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：resample。 | 标题：Resample (`resample` / `_resample_kernel` / `_categorical_finalize_kernel`) |

## `vllm_ascend/ops/triton/fla/`

线性注意力相关分块/递归设备计算。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/fla/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [chunk.py](../../../vllm_ascend/ops/triton/fla/chunk.py) | chunk 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`ChunkGatedDeltaRuleFunction`, `chunk_gated_delta_rule_fwd`, `chunk_gated_delta_rule` |
| [chunk_delta_h.py](../../../vllm_ascend/ops/triton/fla/chunk_delta_h.py) | Delta 递归状态 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_gated_delta_rule_fwd_h` |
| [chunk_delta_hupdate.py](../../../vllm_ascend/ops/triton/fla/chunk_delta_hupdate.py) | Delta 递归状态 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`chunk_gated_delta_rule_fwd_kernel_hupdate_blockdim64`, `chunk_gated_delta_rule_fwd_hupdate` |
| [chunk_o.py](../../../vllm_ascend/ops/triton/fla/chunk_o.py) | chunk_o 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`chunk_fwd_kernel_o`, `chunk_fwd_o` |
| [chunk_scaled_dot_kkt.py](../../../vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py) | chunk_scaled_dot_kkt 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`chunk_scaled_dot_kkt_fwd_kernel`, `chunk_scaled_dot_kkt_fwd` |
| [cumsum.py](../../../vllm_ascend/ops/triton/fla/cumsum.py) | 累积求和 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`chunk_local_cumsum_scalar_kernel`, `chunk_local_cumsum_scalar`, `chunk_local_cumsum` |
| [fused_qkvzba_split_reshape.py](../../../vllm_ascend/ops/triton/fla/fused_qkvzba_split_reshape.py) | fused_qkvzba_split_reshape 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`fused_qkvzba_split_reshape_cat_kernel`, `fused_qkvzba_split_reshape_cat` |
| [l2norm.py](../../../vllm_ascend/ops/triton/fla/l2norm.py) | l2norm 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`l2norm_fwd_kernel2_loop`, `l2norm_fwd` |
| [layernorm_guard.py](../../../vllm_ascend/ops/triton/fla/layernorm_guard.py) | 归一化 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`LayerNormFn`, `layer_norm_fwd_kernel` |
| [solve_tril.py](../../../vllm_ascend/ops/triton/fla/solve_tril.py) | solve_tril 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`solve_tril_16x16_kernel`, `merge_16x16_to_32x32_inverse_kernel`, `merge_16x16_to_64x64_inverse_kernel` |
| [utils.py](../../../vllm_ascend/ops/triton/fla/utils.py) | 辅助函数 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`prepare_lens`, `prepare_chunk_indices`, `prepare_final_chunk_indices` |
| [wy_fast.py](../../../vllm_ascend/ops/triton/fla/wy_fast.py) | wy_fast 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`recompute_w_u_fwd_kernel`, `recompute_w_u_fwd` |

## `vllm_ascend/ops/triton/kda/`

KDA 线性注意力与状态更新相关 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/kda/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [chunk_delta_h.py](../../../vllm_ascend/ops/triton/kda/chunk_delta_h.py) | Delta 递归状态 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda`, `chunk_gated_delta_rule_fwd_h_kda` |
| [conv_state.py](../../../vllm_ascend/ops/triton/kda/conv_state.py) | 状态、卷积 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Pack and restore strided convolution states for KDA.；定义：`copy_conv_state` |
| [cumsum.py](../../../vllm_ascend/ops/triton/kda/cumsum.py) | 累积求和 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`chunk_local_cumsum_scalar_kernel`, `chunk_local_cumsum_scalar`, `chunk_local_cumsum_vector_kernel` |
| [fused_recurrent_kda.py](../../../vllm_ascend/ops/triton/kda/fused_recurrent_kda.py) | KDA 线性注意力 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`fused_recurrent_gated_delta_rule_fwd_kernel` |
| [gate.py](../../../vllm_ascend/ops/triton/kda/gate.py) | 门控 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Host-side KDA gate used when the NPU recurrent kernel cannot fuse it.；定义：`apply_kda_gate` |
| [kda.py](../../../vllm_ascend/ops/triton/kda/kda.py) | KDA 线性注意力 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`fused_recurrent_kda_fwd`, `fused_recurrent_kda`, `layer_norm_gated_fwd_kernel` |
| [l2norm.py](../../../vllm_ascend/ops/triton/kda/l2norm.py) | l2norm 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`l2norm_fwd_persistent_kernel`, `l2norm_fwd_tiled_kernel`, `l2norm_fwd` |
| [output_writeback.py](../../../vllm_ascend/ops/triton/kda/output_writeback.py) | 输出 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`write_recurrent_output` |
| [solve_tril.py](../../../vllm_ascend/ops/triton/kda/solve_tril.py) | solve_tril 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`solve_tril_16x16_kernel_kda`, `merge_16x16_to_32x32_inverse_kernel_kda`, `merge_16x16_to_64x64_inverse_kernel_kda` |
| [utils.py](../../../vllm_ascend/ops/triton/kda/utils.py) | 辅助函数 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`tensor_cache`, `prepare_lens`, `prepare_chunk_indices` |

## `vllm_ascend/ops/triton/linearnorm/`

Linear/Norm 相关设备计算与融合辅助。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/linearnorm/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [fused_eh_norm.py](../../../vllm_ascend/ops/triton/linearnorm/fused_eh_norm.py) | fused_eh_norm 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`fused_eh_norm` |
| [split_qkv_rmsnorm_mrope.py](../../../vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py) | 位置旋转 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`split_qkv_rmsnorm_mrope_kernel`, `triton_split_qkv_rmsnorm_mrope`, `triton_split_qkv_rmsnorm_mrope_fake` |
| [split_qkv_rmsnorm_rope.py](../../../vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py) | 位置旋转 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`split_qkv_rmsnorm_rope_kernel`, `split_qkv_rmsnorm_rope_impl`, `split_qkv_rmsnorm_rope_impl_fake` |
| [split_qkv_rmsnorm_rope_simt.py](../../../vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope_simt.py) | 位置旋转 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`precompute_rope_cos_sin_kernel`, `split_qkv_rmsnorm_rope_simt_kernel`, `split_qkv_rmsnorm_rope_simt_impl` |
| [split_qkv_rmsnorm_rope_vnorm.py](../../../vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope_vnorm.py) | 位置旋转 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`split_qkv_rmsnorm_rope_vnorm_kernel`, `qkv_batch_size_per_iter_per_vec`, `qkv_rmsnorm_rope_vnorm_fits_ub` |
| [split_qkv_tp_rmsnorm_rope.py](../../../vllm_ascend/ops/triton/linearnorm/split_qkv_tp_rmsnorm_rope.py) | 位置旋转 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`split_qkv_tp_rmsnorm_rope_impl`, `split_qkv_tp_rmsnorm_rope_impl_fake` |

## `vllm_ascend/ops/triton/mamba/`

Mamba/选择性状态空间模型相关 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/mamba/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [causal_conv1d.py](../../../vllm_ascend/ops/triton/mamba/causal_conv1d.py) | 卷积 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`extract_last_width` |
| [lightning_attn.py](../../../vllm_ascend/ops/triton/mamba/lightning_attn.py) | 注意力 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：NPU-compatible linear attention operators for BailingMoE.；定义：`_attention`, `AscendLightningAttentionKernel`, `lightning_attention_npu` |
| [postprocess.py](../../../vllm_ascend/ops/triton/mamba/postprocess.py) | postprocess 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`postprocess_mamba_fused_kernel` |
| [state_index.py](../../../vllm_ascend/ops/triton/mamba/state_index.py) | 状态 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`gather_ssm_states`, `scatter_ssm_states_` |
| [state_ops.py](../../../vllm_ascend/ops/triton/mamba/state_ops.py) | 状态 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Recurrent-state gather that preserves padded cache storage on Ascend.；定义：`gather_initial_states` |

## `vllm_ascend/ops/triton/msa/`

MSA 相关位置选择与状态/元数据 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/msa/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [minimax_qknorm_rope_cache.py](../../../vllm_ascend/ops/triton/msa/minimax_qknorm_rope_cache.py) | MiniMax 模型、缓存、位置旋转 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：MiniMax-M3 packed QKV/index normalization, RoPE and paged cache insertion.；定义：`minimax_qknorm_rope_cache` |
| [msa_m3_index_postprocess.py](../../../vllm_ascend/ops/triton/msa/msa_m3_index_postprocess.py) | msa_m3_index_postprocess 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Fused TopK helpers around the A2/A3 AscendC MsaIndexScore operator.；定义：`index_decode_topk_fused` |

## `vllm_ascend/ops/triton/spec_decode/`

投机候选、验证输入和序列推进相关 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/spec_decode/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [dspark_swa_indices.py](../../../vllm_ascend/ops/triton/spec_decode/dspark_swa_indices.py) | DSpark 草稿 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`build_dspark_swa_indices_triton` |
| [ngram.py](../../../vllm_ascend/ops/triton/spec_decode/ngram.py) | n-gram 候选 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`ngram_spec_decode_kernel`, `triton_ngram_spec_decode` |
| [utils.py](../../../vllm_ascend/ops/triton/spec_decode/utils.py) | 辅助函数 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`prepare_inputs_padded_kernel`, `copy_and_expand_dflash_and_dspark_inputs_kernel`, `dflash2_greedy_selector_walk_kernel` |

## `vllm_ascend/ops/triton/v2/`

配合 V2 执行的输入、采样和缓存设备 kernel。

| 子目录 | 职责 |
| --- | --- |
| [block_table/](../../../vllm_ascend/ops/triton/v2/block_table) | 缓存页/分块 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。 |
| [docs/](../../../vllm_ascend/ops/triton/v2/docs) | docs 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。 |
| [mamba/](../../../vllm_ascend/ops/triton/v2/mamba) | Mamba 状态空间 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。 |
| [metrics/](../../../vllm_ascend/ops/triton/v2/metrics) | 指标 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。 |
| [sample/](../../../vllm_ascend/ops/triton/v2/sample) | 采样 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。 |
| [spec_decode/](../../../vllm_ascend/ops/triton/v2/spec_decode) | 投机解码 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/v2/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [apply_grammar_bitmask.py](../../../vllm_ascend/ops/triton/v2/apply_grammar_bitmask.py) | apply_grammar_bitmask 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`_ApplyGrammarBitmaskKernelLauncher` |

## `vllm_ascend/ops/triton/v2/block_table/`

缓存页/分块 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/v2/block_table/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [compute_slot_mappings.py](../../../vllm_ascend/ops/triton/v2/block_table/compute_slot_mappings.py) | 槽位 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`_compute_slot_mappings_kernel` |

## `vllm_ascend/ops/triton/v2/docs/`

docs 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [apply_grammar_bitmask.md](../../../vllm_ascend/ops/triton/v2/docs/apply_grammar_bitmask.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：apply_grammar_bitmask。 | 标题：apply_grammar_bitmask |

## `vllm_ascend/ops/triton/v2/mamba/`

Mamba 状态空间 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../vllm_ascend/ops/triton/v2/mamba/docs) | docs 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/v2/mamba/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [precopy.py](../../../vllm_ascend/ops/triton/v2/mamba/precopy.py) | 复制 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`precopy_mamba_align_fused_kernel` |

## `vllm_ascend/ops/triton/v2/mamba/docs/`

docs 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [mamba_precopy.md](../../../vllm_ascend/ops/triton/v2/mamba/docs/mamba_precopy.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：mamba_precopy。 | 标题：precopy_mamba_align_fused_kernel |

## `vllm_ascend/ops/triton/v2/metrics/`

指标 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../vllm_ascend/ops/triton/v2/metrics/docs) | docs 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/v2/metrics/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [num_nans.py](../../../vllm_ascend/ops/triton/v2/metrics/num_nans.py) | num_nans 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`get_num_nans` |

## `vllm_ascend/ops/triton/v2/metrics/docs/`

docs 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [num_nans.md](../../../vllm_ascend/ops/triton/v2/metrics/docs/num_nans.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：num_nans。 | 标题：_num_nans_kernel |

## `vllm_ascend/ops/triton/v2/sample/`

采样 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。

| 子目录 | 职责 |
| --- | --- |
| [docs/](../../../vllm_ascend/ops/triton/v2/sample/docs) | docs 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/v2/sample/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [categorical_sample.py](../../../vllm_ascend/ops/triton/v2/sample/categorical_sample.py) | 采样 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Triton categorical sampling operator for Ascend NPU.；定义：`_get_vectorcore_num`, `_categorical_prepare_mass_kernel`, `_categorical_sample_kernel` |
| [fill_logprob_token_idx.py](../../../vllm_ascend/ops/triton/v2/sample/fill_logprob_token_idx.py) | fill_logprob_token_idx 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`_fill_logprob_token_ids_kernel` |
| [sampling_mask.py](../../../vllm_ascend/ops/triton/v2/sample/sampling_mask.py) | 采样 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Ascend compatibility kernel for sampling-mask packing.；定义：`sampling_mask_from_logits_npu` |
| [thinking_budget.py](../../../vllm_ascend/ops/triton/v2/sample/thinking_budget.py) | thinking_budget 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Ascend compatibility kernels for Model Runner V2 thinking budgets.；定义：`_load_effective_token_ascend`, `_update_committed_marker_cache_kernel_ascend` |

## `vllm_ascend/ops/triton/v2/sample/docs/`

docs 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fill_logprob_token_ids.md](../../../vllm_ascend/ops/triton/v2/sample/docs/fill_logprob_token_ids.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：fill_logprob_token_ids。 | 标题：_fill_logprob_token_ids_kernel |

## `vllm_ascend/ops/triton/v2/spec_decode/`

投机解码 子模块；所属范围：配合 V2 执行的输入、采样和缓存设备 kernel。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/ops/triton/v2/spec_decode/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [prepare_dflash_inputs.py](../../../vllm_ascend/ops/triton/v2/spec_decode/prepare_dflash_inputs.py) | 准备、DFlash 草稿、输入 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 定义：`prepare_dflash_inputs_triton` |
| [resample.py](../../../vllm_ascend/ops/triton/v2/spec_decode/resample.py) | 采样 的 Triton kernel/调用封装；实际 shape、dtype 与 mask 约定见实现。 | 注释：Categorical resampling for speculative decoding on Ascend NPU.；定义：`resample` |
