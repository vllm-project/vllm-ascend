# tests/e2e/nightly：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **144 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/e2e/nightly/`

夜间模型、精度、稳定性与性能场景。

| 子目录 | 职责 |
| --- | --- |
| [310p/](../../../tests/e2e/nightly/310p) | 端到端 310p 场景、运行脚本和配置。 |
| [scripts/](../../../tests/e2e/nightly/scripts) | 端到端 scripts 场景、运行脚本和配置。 |
| [single_node/](../../../tests/e2e/nightly/single_node) | 端到端 single_node 场景、运行脚本和配置。 |

## `tests/e2e/nightly/310p/`

端到端 310p 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [single_node/](../../../tests/e2e/nightly/310p/single_node) | 端到端 single_node 场景、运行脚本和配置。 |

## `tests/e2e/nightly/310p/single_node/`

端到端 single_node 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [ops/](../../../tests/e2e/nightly/310p/single_node/ops) | 端到端 ops 场景、运行脚本和配置。 |

## `tests/e2e/nightly/310p/single_node/ops/`

端到端 ops 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [singlecard_ops/](../../../tests/e2e/nightly/310p/single_node/ops/singlecard_ops) | 端到端 singlecard_ops 场景、运行脚本和配置。 |

## `tests/e2e/nightly/310p/single_node/ops/singlecard_ops/`

端到端 singlecard_ops 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_chunk_fwd_o_vllm_310.py](../../../tests/e2e/nightly/310p/single_node/ops/singlecard_ops/test_chunk_fwd_o_vllm_310.py) | 验证 chunk fwd o vllm 310 的测试；具体断言场景由右栏测试符号定位。 | 注释：chunk_fwd_o_vllm correctness tests on Ascend 310P via torch.ops._C_ascend binding.；定义：`test_constant_inputs`, `test_random_inputs_no_nan`, `test_g_zero_reduces_to_standard_attention`, `test_chunk_boundary_independence` |
| [test_chunk_gated_delta_rule_fwd_h_vllm_aclnn.py](../../../tests/e2e/nightly/310p/single_node/ops/singlecard_ops/test_chunk_gated_delta_rule_fwd_h_vllm_aclnn.py) | 验证 Delta 递归状态、门控 的测试；具体断言场景由右栏测试符号定位。 | 注释：chunk_gated_delta_rule_fwd_h_vllm correctness tests on Ascend 310P；定义：`test_h_state_correctness`, `test_v_new_correctness`, `test_no_nan` |
| [test_recurrent_gated_delta_rule_v310.py](../../../tests/e2e/nightly/310p/single_node/ops/singlecard_ops/test_recurrent_gated_delta_rule_v310.py) | 验证 Delta 递归状态、门控 的测试；具体断言场景由右栏测试符号定位。 | 注释：Test V310 kernel via ctypes API against golden CPU reference.；定义：`test_recurrent_gated_delta_rule_v310` |

## `tests/e2e/nightly/scripts/`

端到端 scripts 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [aop_capture.sh](../../../tests/e2e/nightly/scripts/aop_capture.sh) | aop capture 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | 函数：`parse_log` |
| [aop_classify.sh](../../../tests/e2e/nightly/scripts/aop_classify.sh) | aop classify 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | 函数：`check_log` |
| [aop_commit_age.sh](../../../tests/e2e/nightly/scripts/aop_commit_age.sh) | aop commit age 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | Shell 脚本；按工程阶段和文件名归类。 |
| [aop_process.sh](../../../tests/e2e/nightly/scripts/aop_process.sh) | aop process 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | 命令类别：`python` |
| [aop_skip.sh](../../../tests/e2e/nightly/scripts/aop_skip.sh) | aop skip 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | Shell 脚本；按工程阶段和文件名归类。 |
| [bisect_args.sh](../../../tests/e2e/nightly/scripts/bisect_args.sh) | bisect args 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | 函数：`build_bisect_extra_args` |
| [result_postprocess.py](../../../tests/e2e/nightly/scripts/result_postprocess.py) | result postprocess 的测试辅助、测试入口或共享数据结构。 | 注释：Post-process nightly/weekly benchmark results into per-case JSON files.；定义：`resolve_suite_name`, `resolve_testcase_name`, `resolve_test_version` |
| [rules-env.txt](../../../tests/e2e/nightly/scripts/rules-env.txt) | rules-env 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 依据文件格式及所在目录归类。 |
| [update_good_table.py](../../../tests/e2e/nightly/scripts/update_good_table.py) | update good table 的测试辅助、测试入口或共享数据结构。 | 注释：Update a frequency-specific good_table.csv with a successful test entry.；定义：`non_empty`, `valid_soc`, `git_head` |

## `tests/e2e/nightly/single_node/`

端到端 single_node 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [ops/](../../../tests/e2e/nightly/single_node/ops) | 端到端 ops 场景、运行脚本和配置。 |

## `tests/e2e/nightly/single_node/ops/`

端到端 ops 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [multicard_ops_a3/](../../../tests/e2e/nightly/single_node/ops/multicard_ops_a3) | 端到端 multicard_ops_a3 场景、运行脚本和配置。 |
| [singlecard_ops/](../../../tests/e2e/nightly/single_node/ops/singlecard_ops) | 端到端 singlecard_ops 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/nightly/single_node/ops/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [conftest.py](../../../tests/e2e/nightly/single_node/ops/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 定义：`pytest_runtest_setup`, `pytest_runtest_teardown`, `pytest_runtest_call` |
| [test_mega_moe.py](../../../tests/e2e/nightly/single_node/ops/test_mega_moe.py) | 验证 混合专家 的测试；具体断言场景由右栏测试符号定位。 | 注释：A5 MegaMoe weight/activation integration and latency comparison.；定义：`main` |
| [test_minimax_qknorm_rope_cache_performance.py](../../../tests/e2e/nightly/single_node/ops/test_minimax_qknorm_rope_cache_performance.py) | 验证 MiniMax 模型、缓存、位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare runtime NUM_SLOTS against the previous constexpr specialization.；定义：`test_num_slots_specialization_performance` |

## `tests/e2e/nightly/single_node/ops/multicard_ops_a3/`

端到端 multicard_ops_a3 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/nightly/single_node/ops/multicard_ops_a3/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_dcp_a2a.py](../../../tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dcp_a2a.py) | 验证 Decode 上下文并行 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_registered_dcp_a2a_fused_multi_rank` |
| [test_dispatch_ffn_combine.py](../../../tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dispatch_ffn_combine.py) | 验证 token 分发、token 合并 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dispatch_ffn_combine_kernel` |
| [test_dispatch_ffn_combine_bf16.py](../../../tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dispatch_ffn_combine_bf16.py) | 验证 token 分发、token 合并 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dispatch_ffn_combine_kernel` |
| [test_dispatch_ffn_combine_w4a8.py](../../../tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dispatch_ffn_combine_w4a8.py) | 验证 token 分发、token 合并 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dispatch_ffn_combine_kernel` |
| [test_mtp_index_cache.py](../../../tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_mtp_index_cache.py) | 验证 缓存、多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mtp_index_cache_dsa_cp_eager_and_graph` |
| [test_sfa_cache_store.py](../../../tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_sfa_cache_store.py) | 验证 缓存、存储、稀疏注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_native_main_c8_cache_preserves_packing_and_padding`, `test_platform_cache_store_preserves_all_backing_bytes`, `test_missing_fast_operator_preserves_fallback_with_negative_slots`, `test_unsupported_inner_stride_matches_generic_scatter_behavior` |

## `tests/e2e/nightly/single_node/ops/singlecard_ops/`

端到端 singlecard_ops 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [triton/](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton) | 端到端 triton 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [benchmark_kda_state_copy_triton.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/benchmark_kda_state_copy_triton.py) | 性能测量、状态、复制 的测试辅助、测试入口或共享数据结构。 | 注释：Paired production/preallocated KDA microbenchmark against PR #17301.；定义：`BenchmarkResult`, `main` |
| [test_add_rms_norm_bias.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_add_rms_norm_bias.py) | 验证 RMS 归一化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_quant_fpx_linear` |
| [test_add_rms_norm_bias_a5.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_add_rms_norm_bias_a5.py) | 验证 RMS 归一化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_add_rms_norm_bias_a5`, `test_a5_beta_is_added_in_fp32_before_output_cast`, `test_a5_add_rms_norm_bias_rejects_invalid_beta` |
| [test_attn_res_fwd.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_attn_res_fwd.py) | 验证 注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_attn_res_fwd_precision` |
| [test_bgmv_expand.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_bgmv_expand.py) | 验证 bgmv expand 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_bgmv_expand` |
| [test_bgmv_shrink.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_bgmv_shrink.py) | 验证 bgmv shrink 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_bgmv_shrink` |
| [test_cann_qli_binding.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_cann_qli_binding.py) | 验证 接口绑定 的测试；具体断言场景由右栏测试符号定位。 | 注释：Official QLI accuracy and score-output behavior with custom OPP enabled.；定义：`test_cann_qli_against_cpu` |
| [test_causal_conv1d_310.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_causal_conv1d_310.py) | 验证 卷积 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ascend_causal_conv1d_310_fn`, `test_causal_conv1d_310_update` |
| [test_chunk_gated_delta_rule_fwd_h_vllm_aclnn.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_chunk_gated_delta_rule_fwd_h_vllm_aclnn.py) | 验证 Delta 递归状态、门控 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_chunk_gated_delta_rule_fwd_h_matches_reference`, `test_chunk_gated_delta_rule_fwd_h_kda_is_bitwise_deterministic` |
| [test_chunk_kda_aclnn.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_chunk_kda_aclnn.py) | 验证 KDA 线性注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_kda_torch_bindings_have_shape_correct_meta_kernels`, `test_chunk_kda_fwd_matches_reference_bsnd`, `test_kda_layout_swap12_matches_reference`, `test_chunk_kda_fwd_c128_v256_path` |
| [test_compressor_metadata.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor_metadata.py) | 验证 长程压缩、元数据 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_compressor_metadata_dsv4_flash_real_metadata`, `test_compressor_metadata_out_cross_stream_buffer_reuse` |
| [test_copy_and_expand_eagle_inputs.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_copy_and_expand_eagle_inputs.py) | 验证 Eagle 投机、输入、复制 的测试；具体断言场景由右栏测试符号定位。 | 注释：E2E accuracy test for CopyAndExpandEagleInputs custom operator.；定义：`test_copy_and_expand_eagle_inputs`, `test_minimal_case`, `test_large_tokens_per_request`, `test_large_tokens_shift_true` |
| [test_copy_sfa_offload_metadata.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_copy_sfa_offload_metadata.py) | 验证 元数据、缓存卸载、复制 的测试；具体断言场景由右栏测试符号定位。 | 注释：Device metadata, ownership transitions and padded graph replay without a model.；定义：`test_prefill_batches_carry_pool_slots`, `test_device_lengths_tail_geometry_and_rejection`, `test_slot_invalidation_compaction_and_prefix_rollback_reset`, `test_slot_reassignment_invalidates_all_banks_without_resetting_other_slots` |
| [test_dequant_situ_quant.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_dequant_situ_quant.py) | 验证 反量化、量化、SITU 激活 的测试；具体断言场景由右栏测试符号定位。 | 注释：Kimi K3 W4A8 numerical coverage for DequantSituQuant.；定义：`test_kimi_k3_routed_multi_expert_dequant_situ_quant`, `test_kimi_k3_predequantized_bf16_situ_quant`, `test_kimi_k3_shared_expert_without_bias_or_group_index`, `test_kimi_k3_zero_routed_rows_return_empty_outputs` |
| [test_dequant_situ_quant_group.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_dequant_situ_quant_group.py) | 验证 反量化、量化、SITU 激活 的测试；具体断言场景由右栏测试符号定位。 | 注释：Single-card tests for the optional ``group_index`` input of；定义：`test_group_counts_equal_rows_matches_full_scan`, `test_group_counts_smaller_only_touches_active_rows`, `test_group_counts_exceeding_rows_clamps`, `test_bf16_output_matches_float32_reference` |
| [test_dequant_swiglu_quant.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_dequant_swiglu_quant.py) | 验证 反量化、SwiGLU 激活、量化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_npu_dequant_swiglu_quant_with_limit` |
| [test_fused_lightning_indexer_manage.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_lightning_indexer_manage.py) | 验证 历史位置索引 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_fused_lightning_indexer_manage_non_offload_matches_native`, `test_fused_lightning_indexer_manage_non_offload_short_visible_padding`, `test_fused_lightning_indexer_manage_first_decode_initializes_cache`, `test_fused_lightning_indexer_manage_steady_replacement_then_all_hit` |
| [test_fused_moe.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_moe.py) | 验证 混合专家 的测试；具体断言场景由右栏测试符号定位。 | 注释：Tests for the MOE layers.；定义：`test_token_dispatcher_with_all_gather`, `test_token_dispatcher_with_all_gather_quant` |
| [test_fused_quant_lightning_indexer_manage.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_quant_lightning_indexer_manage.py) | 验证 历史位置索引、量化 的测试；具体断言场景由右栏测试符号定位。 | 注释：Quantized LIM (fused_quant_lightning_indexer_manage) on-board contracts.；定义：`test_fused_quant_lim_non_offload_matches_native`, `test_fused_quant_lim_non_offload_short_visible_padding`, `test_fused_quant_lim_first_decode_initializes_cache`, `test_fused_quant_lim_steady_replacement_then_all_hit` |
| [test_fused_scatter_copy_sparse_flash_attention.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_scatter_copy_sparse_flash_attention.py) | 验证 注意力、分散写入、复制 的测试；具体断言场景由右栏测试符号定位。 | 注释：Ascend correctness coverage for fused copy SFA with dynamic TND.；定义：`test_heterogeneous_mtp2_mtp3_mtp8_mtp15_n8`, `test_n128`, `test_fp16`, `test_intermediate_supported_head_counts` |
| [test_fused_sparse_attention_overlap.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py) | 验证 注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_fused_sparse_attention_overlap_internal_planner_miss_then_all_hit_precision` |
| [test_gating_top_k_softmax.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_gating_top_k_softmax.py) | 验证 概率归一化、门控、TopK 选择 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_quant_fpx_linear` |
| [test_glm5next_conv_state.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_glm5next_conv_state.py) | 验证 状态、卷积 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_conv_state_copy_masks_invalid_slots_and_preserves_shared_pages`, `test_causal_conv_first_packed_row_matches_reference` |
| [test_glm5next_state_gather_npu.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_glm5next_state_gather_npu.py) | 验证 收集、状态 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_gather_fresh_and_existing_states_with_poison_npu`, `test_gather_graph_replay_observes_changed_history_mask_and_poison` |
| [test_gmm_dequant_situ_quant_msd.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_gmm_dequant_situ_quant_msd.py) | 验证 反量化、量化、SITU 激活 的测试；具体断言场景由右栏测试符号定位。 | 注释：Bounded A3 MSD GmmDequantSituQuant correctness, native NZ, and graph regressions.；定义：`test_gmm_dequant_situ_quant_dispatcher_shape_inference`, `test_gmm_dequant_situ_quant_msd_eager`, `test_gmm_dequant_situ_quant_msd_graph_dynamic_groups`, `test_gmm_dequant_situ_quant_msd_zero_capacity` |
| [test_gmm_situ_small_experts.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_gmm_situ_small_experts.py) | 验证 专家、SITU 激活、分组矩阵乘 的测试；具体断言场景由右栏测试符号定位。 | 注释：Check small-expert tiling and changing device routing under ACLGraph.；定义：`test_small_expert_gmm_situ_graph` |
| [test_gmm_swiglu_quant_weight_nz_tensor_list.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_gmm_swiglu_quant_weight_nz_tensor_list.py) | 验证 权重、张量、SwiGLU 激活 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_gmm_swiglu_quant_weight_nz_tensor_list` |
| [test_grouped_matmul_swiglu_quant.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_grouped_matmul_swiglu_quant.py) | 验证 矩阵乘、SwiGLU 激活、量化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_grouped_matmul_swiglu_quant_kernel` |
| [test_inplace_partial_rotary_mul.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_inplace_partial_rotary_mul.py) | 验证 位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_inplace_partial_rotary_mul` |
| [test_kda_state_copy_integration.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_kda_state_copy_integration.py) | 验证 状态、复制、KDA 线性注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：NPU regressions for startup plans and dynamic selected counts.；定义：`test_direct_compiled_launch_keeps_constexpr_arguments`, `test_dynamic_grid_and_metadata_use_only_four_compiled_variants`, `test_worker_plan_deduplication_and_failed_reinitialization`, `test_worker_plan_graph_replay_and_rejections` |
| [test_kda_state_copy_triton.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_kda_state_copy_triton.py) | 验证 状态、复制、KDA 线性注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：Production-plan copy correctness adapted from PR #17301 at 5bcbad36fdc3.；定义：`test_kda_state_copy_preserves_gaps_offsets_and_invalid_rows`, `test_kda_state_copy_graph_replay_changed_indices_and_flags`, `test_kda_state_copy_address_beyond_four_gib`, `test_kda_state_copy_empty_selection_and_inner_stride_rejection` |
| [test_kimi_k3_chunk_kda_tail_npu.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_chunk_kda_tail_npu.py) | 验证 Kimi 模型、KDA 线性注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：Regression coverage for Kimi-K3 BF16 chunk-KDA tail execution.；定义：`test_kimi_k3_chunk_kda_bf16_tail_is_deterministic` |
| [test_kimi_k3_situ_fusion_cases.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_situ_fusion_cases.py) | 验证 算子融合、SITU 激活、Kimi 模型 的测试；具体断言场景由右栏测试符号定位。 | 注释：Single-operator Kimi K3 SiTU fusion fixtures.；定义：`test_a3_shared_dequant_situ_quant_single_op`, `test_a3_routed_bf16_dequant_situ_quant_single_op`, `test_a5_shared_situ_mx_quant_shapes_single_op`, `test_a5_routed_situ_mx_quant_shapes_single_op` |
| [test_kimi_kda_ascendc_npu.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_kda_ascendc_npu.py) | 验证 Kimi 模型、KDA 线性注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：Kimi K3 integration coverage for the fused AscendC prefill operator.；定义：`test_kimi_k3_safe_gate_prefill_and_transposed_state_layout`, `test_kimi_k3_a5_multichunk_all_outputs_match_reference`, `test_kimi_k3_a5_model_prefill_profile_shape` |
| [test_kimi_kda_recurrent_ascendc_npu.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_kda_recurrent_ascendc_npu.py) | 验证 Kimi 模型、KDA 线性注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：Kimi K3 recurrent-KDA reference and direct AscendC accuracy coverage.；定义：`test_kimi_k3_tp16_recurrent_kda_non_contiguous_qkv_and_state_pool` |
| [test_kv_quant_sparse_flash_attention_vllm.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention_vllm.py) | 验证 注意力、量化、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_kv_quant_sparse_flash_attention_vllm` |
| [test_kv_quant_sparse_flash_attention_vllm_rope0.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention_vllm_rope0.py) | 验证 注意力、量化、位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 注释：C8 rope=0 feature contracts and independent, uniform-attention accuracy.；定义：`test_rope0_compact_matches_padded_bitwise`, `test_rope_uniform_attention_and_lse`, `test_rope_graph_capture_replay`, `test_rope_invalid_shape_or_dimension` |
| [test_lim_mtp_index_reuse.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_lim_mtp_index_reuse.py) | 验证 多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 注释：MTP reuses the compacted LIM output without per-draft metadata rebuilds.；定义：`test_compaction_preserves_complete_step0_lim_rows`, `test_compaction_ignores_graph_padding_rows`, `test_later_draft_reuses_compacted_selection_without_copy`, `test_copy_sfa_graph_zero_lengths_follow_replay_and_preserve_inactive_cache` |
| [test_mla_preprocess.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_mla_preprocess.py) | 验证 MLA 潜在注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mla_preprocess_supported_modes`, `test_mla_preprocess_qm1_noncontiguous_cache`, `test_mla_preprocess_nzcache_accepts_logical_cache_shape` |
| [test_mla_preprocess_nq.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_mla_preprocess_nq.py) | 验证 MLA 潜在注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mla_preprocess_kernel` |
| [test_mla_preprocess_qdown.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_mla_preprocess_qdown.py) | 验证 MLA 潜在注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mla_preprocess_kernel` |
| [test_npu_hc_pre.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py) | 验证 多路残差混合 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_npu_hc_pre_v2_bf16_3d_input`, `test_npu_hc_pre_v2_bf16_4d_input`, `test_npu_hc_pre_v2_bf16_dsv4_flash_hidden_size`, `test_npu_hc_pre_v2_bf16_extended_hidden_size` |
| [test_npu_moe_gating_top_k.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_moe_gating_top_k.py) | 验证 门控、TopK 选择、混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_npu_moe_gating_topk_compare` |
| [test_npu_moe_gating_top_k_hash.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_moe_gating_top_k_hash.py) | 验证 门控、TopK 选择、哈希映射 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_v41_vision_bias_and_image_sentinel` |
| [test_pa_kv_cache_ops.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_pa_kv_cache_ops.py) | 验证 缓存、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_scatter_pa_kv_cache_slot_mapping_zero_and_minus_one`, `test_scatter_pa_kv_cache_all_minus_one_leaves_cache_unchanged` |
| [test_prepare_indexer_indices.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_prepare_indexer_indices.py) | 验证 历史位置索引、准备 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_prepare_indexer_indices`, `test_prepare_indexer_indices_boundaries`, `test_prepare_indexer_indices_full_int32_range`, `test_prepare_indexer_indices_noncontiguous` |
| [test_qfa_c8_paged_layout.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_qfa_c8_paged_layout.py) | 验证 内存布局 的测试；具体断言场景由右栏测试符号定位。 | 注释：A5/CANN validation of QFA's strided K/V and six-dimensional scales.；定义：`test_qfa_strided_scales_match_contiguous_reference` |
| [test_qli_candidate_rows.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_qli_candidate_rows.py) | 验证 qli candidate rows 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_candidate_consumer_preserves_each_query_mask` |
| [test_quant_lightning_indexer_v2.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_quant_lightning_indexer_v2.py) | 验证 历史位置索引、量化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_quant_lightning_indexer_v2_noncontiguous_pa_cache_dim0`, `test_quant_lightning_indexer_v2_high_physical_page_scale_offset` |
| [test_quantize_indexer_query.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_quantize_indexer_query.py) | 验证 量化、历史位置索引 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_quantize_indexer_query`, `test_quantize_indexer_query_rounding_and_scale`, `test_quantize_indexer_query_noncontiguous`, `test_quantize_indexer_query_graph_replay` |
| [test_recurrent_gated_delta_rule_310.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_recurrent_gated_delta_rule_310.py) | 验证 Delta 递归状态、门控 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_fused_recurrent_gated_delta_rule_310` |
| [test_rms_norm_cast.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_rms_norm_cast.py) | 验证 RMS 归一化、类型转换 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_rms_norm_cast`, `test_rms_norm_cast_npu_graph` |
| [test_scatter_nd_update_sk.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_scatter_nd_update_sk.py) | 验证 分散写入 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_a5_cache_destination_layout`, `test_scatter_nd_update_sk`, `test_scatter_nd_update_sk_duplicate_indices`, `test_cache_adaptor_dispatch_preserves_backing_storage` |
| [test_sparse_flash_attention_dcp_lse.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_sparse_flash_attention_dcp_lse.py) | 验证 注意力、Decode 上下文并行 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_sparse_flash_attention_padded_indices_lse` |
| [test_sparse_kv_offload.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_sparse_kv_offload.py) | 验证 缓存卸载、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_sparse_kv_warmup_threads`, `test_sparse_kv_restore_tensor_views`, `test_sparse_kv_compute_lru_resident_addrs`, `test_stable_rows_suffix_invalidation_and_plan_encoding` |
| [test_store_kv_block.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_store_kv_block.py) | 验证 缓存页/分块、存储、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_scatter`, `test_myops` |
| [test_transpose_kv_cache_by_block.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_transpose_kv_cache_by_block.py) | 验证 转置、缓存、缓存页/分块 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_transpose_kv_cache_by_block`, `test_block_strides_preserve_storage`, `test_single_block_strides_preserve_storage`, `test_empty_cache_requires_empty_block_ids` |
| [test_uva_view.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_uva_view.py) | 验证 uva view 的测试；具体断言场景由右栏测试符号定位。 | 注释：UVA view and MRV2 wrapper component tests.；定义：`test_npu_view_preserves_metadata`, `test_npu_views_keep_cpu_storage_alive`, `test_npu_view_reads_cpu_updates_without_copy`, `test_npu_view_keeps_cpu_storage_alive` |

## `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/`

端到端 triton 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_apply_grammar_bitmask_triton.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_grammar_bitmask_triton.py) | 验证 apply grammar bitmask triton 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_apply_grammar_bitmask_business_shape` |
| [test_apply_penalties_triton.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_penalties_triton.py) | 验证 采样惩罚 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_apply_all_penalties_v1_vs_ascend` |
| [test_bad_words.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bad_words.py) | 验证 bad words 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_apply_bad_words_different_shapes`, `test_apply_bad_words_no_bad_words`, `test_apply_bad_words_edge_cases`, `test_apply_bad_words_token_limit` |
| [test_batch_invariant_ops.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_batch_invariant_ops.py) | 验证 批次 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_matmul_bias_persistent_kernel_precision`, `test_linear_persistent_kernel_precision`, `test_mean_kernel_precision`, `test_rms_norm_kernel_precision` |
| [test_batch_memcpy.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_batch_memcpy.py) | 验证 内存复制、批次 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_batch_memcpy` |
| [test_bincount.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py) | 验证 token 分桶计数 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_bincount_kernel` |
| [test_build_local_metadata_triton.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_build_local_metadata_triton.py) | 验证 元数据 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_build_local_metadata_triton`, `test_build_local_metadata_triton_masks_graph_padding` |
| [test_categorical_sample.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_categorical_sample.py) | 验证 采样 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_categorical_sample_greedy`, `test_categorical_sample_mixed_temperature_and_hierarchy_boundaries`, `test_categorical_sample_deterministic_same_seed_and_pos`, `test_categorical_sample_draft_rng_uses_position_salt` |
| [test_chunk_gated_delta_rule.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_gated_delta_rule.py) | 验证 Delta 递归状态、门控 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_chunk_gated_delta_rule_310_state_layout_matches_vllm`, `test_triton_fusion_ops` |
| [test_chunk_kda_npu.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_kda_npu.py) | 验证 KDA 线性注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：Precision tests for vllm's chunk_kda Triton operator on NPU.；定义：`test_chunk_kda` |
| [test_clear_ssm_states.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_clear_ssm_states.py) | 验证 状态 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_clear_ssm_states_ref_parity` |
| [test_compute_slot_mapping.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py) | 验证 槽位 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_compute_slot_mapping_npu_kernel_cp`, `test_ascend_block_tables_compute_slot_mappings_out` |
| [test_compute_slot_mapping_fused.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping_fused.py) | 验证 槽位 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_fused_slot_mapping`, `test_fused_request_count_reuses_compilation` |
| [test_compute_slot_mapping_v1.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping_v1.py) | 验证 槽位 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_compute_slot_mapping_kernel`, `test_compute_slot_mapping_kernel_four_requests_large_sequences` |
| [test_compute_token_logprobs.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_token_logprobs.py) | 验证 对数概率 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_topk_log_softmax_kernel`, `test_topk_log_softmax_edge_cases`, `test_topk_log_softmax_deterministic`, `test_topk_log_softmax_dtypes` |
| [test_compute_topk_logprobs.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py) | 验证 对数概率、TopK 选择 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_compute_topk_logprobs` |
| [test_copy_and_expand_dflash_dspark.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_copy_and_expand_dflash_dspark.py) | 验证 DSpark 草稿、DFlash 草稿、复制 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_copy_and_expand_dflash_dspark`, `test_copy_and_expand_dflash_dspark_empty_valid_context` |
| [test_dcp_a2a.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_a2a.py) | 验证 Decode 上下文并行 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_a5_generalized_batching_matches_scalar_path`, `test_a5_core_relative_batching_boundaries`, `test_pack_and_fused_lse_combine`, `test_stride_aware_pack` |
| [test_fill_logprob_token_ids_kernel.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fill_logprob_token_ids_kernel.py) | 验证 fill logprob token ids kernel 的测试；具体断言场景由右栏测试符号定位。 | 注释：Precision test for _fill_logprob_token_ids_kernel.；定义：`test_custom_token_ids`, `test_no_custom_no_topk` |
| [test_fused_eh_norm.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_eh_norm.py) | 验证 fused eh norm 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_fused_eh_norm_matches_reference` |
| [test_fused_gdn_gating.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_gdn_gating.py) | 验证 门控、Gated DeltaNet 注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_fused_gdn_gating_310p_parity_precision` |
| [test_fused_qkvzba_split_reshape_cat.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_qkvzba_split_reshape_cat.py) | 验证 fused qkvzba split reshape cat 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_fused_qkvzba_split_reshape_cat` |
| [test_fused_recurrent_kda_npu.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_recurrent_kda_npu.py) | 验证 KDA 线性注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：Precision tests for vllm's fused_recurrent_kda Triton operator on NPU.；定义：`test_fused_recurrent_kda`, `test_fused_recurrent_kda_decode_inplace`, `test_fused_recurrent_kda_fp32` |
| [test_glm5next_kpool_tail_triton.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_kpool_tail_triton.py) | 验证 池管理 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_tail_prefill_all_chunk_boundaries_decode_padding_and_graph`, `test_empty_compression_preserves_caches`, `test_mtp_rejection_replay_preserves_open_pool_history`, `test_invalid_tail_layout_raises` |
| [test_glm5next_output_writeback.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_output_writeback.py) | 验证 输出 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_recurrent_writeback_matches_reference`, `test_recurrent_writeback_dp_padding_graph_replay` |
| [test_glm5next_pool_key_indexer_triton.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_pool_key_indexer_triton.py) | 验证 历史位置索引、池管理 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_pool_selection_real_shape_paging_and_causal_tail`, `test_empty_query` |
| [test_glm5next_tail_slot_mapping_triton.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_tail_slot_mapping_triton.py) | 验证 槽位 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_normal_and_mixed_fused_circular_mapping_and_graph`, `test_draft_helper_disables_generic_writes_for_circular_groups` |
| [test_kimi_kda_fused_norm_gate.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kimi_kda_fused_norm_gate.py) | 验证 门控、Kimi 模型、KDA 线性注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_kimi_kda_fused_rms_norm_sigmoid_gate`, `test_fused_rms_norm_silu_gate_preserves_prenorm_contract`, `test_kimi_norm_gate_strides_and_direct_output`, `test_kimi_norm_gate_wide_feature_tile` |
| [test_kv_block_zeroer.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kv_block_zeroer.py) | 验证 缓存页/分块、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_zero_block_ids` |
| [test_l2norm.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_l2norm.py) | 验证 l2norm 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_l2norm` |
| [test_layernorm_gated.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_layernorm_gated.py) | 验证 归一化、门控 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_layer_norm_fwd_npu_correctness`, `test_layer_norm_fwd_npu_rejects_invalid_group_size` |
| [test_lightning_attn.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lightning_attn.py) | 验证 注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：Tests for lightning attention triton kernels.；定义：`test_lightning_attention_npu_single_chunk`, `test_lightning_attention_npu_single_chunk_with_kv_history`, `test_lightning_attention_npu_multi_block`, `test_lightning_attention_npu` |
| [test_log_softmax.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_log_softmax.py) | 验证 概率归一化、日志 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_topk_log_softmax_kernel` |
| [test_lora_linear.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lora_linear.py) | 验证 线性投影、LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_lora_linear_triton_accuracy` |
| [test_mamba_state_index.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_state_index.py) | 验证 Mamba 状态空间、状态 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_gather_and_scatter_ssm_states`, `test_rejects_empty_state_cache`, `test_rejects_overlapping_state_rows` |
| [test_min_p.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_min_p.py) | 验证 min p 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_apply_min_p_kernel` |
| [test_mrope.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mrope.py) | 验证 位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mrotary_embedding_triton_kernel` |
| [test_msa_m3_index_postprocess.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_index_postprocess.py) | 验证 msa m3 index postprocess 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_empty_shape`, `test_zero_logical_blocks`, `test_invalid_inputs`, `test_block_boundaries` |
| [test_muls_add.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_muls_add.py) | 验证 muls add 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_muls_add_triton_correctness` |
| [test_num_nans.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_num_nans.py) | 验证 num nans 的测试；具体断言场景由右栏测试符号定位。 | 注释：Extended precision test for the Ascend-specific _num_nans_kernel.；定义：`test_num_nans_dtype_and_layout`, `test_num_nans_special_values_not_counted` |
| [test_pcp_kv_cache.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_pcp_kv_cache.py) | 验证 缓存、Prefill 上下文并行、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_c8_cache_preserves_packed_bytes`, `test_contiguous_slot_slices_reuse_kernel`, `test_separate_latent_and_rope_cache` |
| [test_penality.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_penality.py) | 验证 penality 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_apply_penalties`, `test_apply_penalties_capable_with_large_shape` |
| [test_postprocess_mamba.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_mamba.py) | 验证 Mamba 状态空间 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_matches_python_postprocess_mamba` |
| [test_postprocess_temporal_copy.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_temporal_copy.py) | 验证 复制 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_temporal_copy_preserves_bytes_and_guards`, `test_temporal_copy_config_derived_sizes`, `test_mixed_postprocess_replay_uses_updated_metadata`, `test_kimi_k3_full_recurrent_layer_grid` |
| [test_precopy.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_precopy.py) | 验证 复制 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_precopy_matches_reference` |
| [test_prepare_dflash_inputs.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_dflash_inputs.py) | 验证 准备、DFlash 草稿、输入 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_prepare_dflash_inputs_impl`, `test_prepare_dflash_inputs_partition_boundaries`, `test_prepare_dflash_inputs_wrapper_forwards_dcp` |
| [test_prepare_inputs_padded.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_inputs_padded.py) | 验证 准备、输入 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_prepare_inputs_padded` |
| [test_rejection_sample.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py) | 验证 候选拒绝验证、采样 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_expand_kernel`, `test_sample_recovered_tokens_kernel`, `test_rejection_random_sample`, `test_rejection_greedy_sample_spec_len_1_triton_kernel` |
| [test_rejection_sample_v2.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample_v2.py) | 验证 候选拒绝验证、采样 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_synthetic_rejection_sample`, `test_draft_vocab_narrower_than_target_vocab`, `test_placeholder_draft_tokens_are_rejected` |
| [test_resample.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample.py) | 验证 采样 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_random_bonus_single_support_accepts_logits_dtypes`, `test_full_draft_residual_accepts_logits_dtypes`, `test_partial_negative_inf_has_zero_probability_mass`, `test_ragged_requests_and_shuffled_mapping_select_correct_rows` |
| [test_rms_norm.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rms_norm.py) | 验证 RMS 归一化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_triton_q_rms_correctness`, `test_triton_q_rms_rejects_unsupported_dim` |
| [test_rope.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py) | 验证 位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_rotary_embedding_triton_kernel`, `test_rotary_embedding_triton_kernel_with_cos_sin_cache`, `test_rotary_embedding_triton_kernel_fp8`, `test_rotary_embedding_triton_kernel_siso` |
| [test_sampling_mask.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sampling_mask.py) | 验证 采样 的测试；具体断言场景由右栏测试符号定位。 | 注释：Regression tests for sampling-mask packing on non-contiguous logits.；定义：`test_sampling_mask_matches_finite_support`, `test_sampling_mask_matches_processed_top_k_top_p_support`, `test_sampling_mask_non_contiguous_vocab_dimension` |
| [test_sparse_index_remap_triton.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sparse_index_remap_triton.py) | 验证 sparse index remap triton 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_sparse_index_remap_triton_matches_reference`, `test_sparse_index_remap_triton_3d_input` |
| [test_split_qkv_rmsnorm_mrope.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py) | 验证 位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_split_qkv_rmsnorm_mrope` |
| [test_split_qkv_rmsnorm_rope.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_rope.py) | 验证 位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_split_qkv_rmsnorm_rope`, `test_split_qkv_rmsnorm_rope_with_bias` |
| [test_split_qkv_rmsnorm_rope_vnorm.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_rope_vnorm.py) | 验证 位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_split_qkv_rmsnorm_rope_vnorm`, `test_split_qkv_rmsnorm_rope_vnorm_with_bias` |
| [test_split_qkv_tp_rmsnorm_rope.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_tp_rmsnorm_rope.py) | 验证 位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_split_qkv_tp_rmsnorm_rope` |
| [test_swiglu_quant.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_swiglu_quant.py) | 验证 SwiGLU 激活、量化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_swiglu_quant_correctness`, `test_swiglu_quant_without_quantization`, `test_swiglu_quant_rejects_invalid_group_list_type`, `test_swiglu_quant_rejects_invalid_group_list_dtype` |
| [test_swiglustep.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_swiglustep.py) | 验证 SwiGLU 激活 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_swiglustep_triton_correctness`, `test_swiglustep_triton_clamps_to_golden_value`, `test_swiglustep_triton_handles_non_contiguous_input` |
| [test_temperature.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_temperature.py) | 验证 temperature 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_temperature_kernel` |
