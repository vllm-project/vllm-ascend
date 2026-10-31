# tests/e2e/pull_request：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **192 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/e2e/pull_request/`

用于 PR 场景的分设备数量测试和运行脚本。

| 子目录 | 职责 |
| --- | --- |
| [eight_card/](../../../tests/e2e/pull_request/eight_card) | 端到端 eight_card 场景、运行脚本和配置。 |
| [four_card/](../../../tests/e2e/pull_request/four_card) | 端到端 four_card 场景、运行脚本和配置。 |
| [one_card/](../../../tests/e2e/pull_request/one_card) | 端到端 one_card 场景、运行脚本和配置。 |
| [rlhf/](../../../tests/e2e/pull_request/rlhf) | 端到端 rlhf 场景、运行脚本和配置。 |
| [two_card/](../../../tests/e2e/pull_request/two_card) | 端到端 two_card 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [utils.py](../../../tests/e2e/pull_request/utils.py) | 辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`LLMTestCase`, `compare_logprobs`, `run_pd_disaggregation` |
| [utils_310p.py](../../../tests/e2e/pull_request/utils_310p.py) | 辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`get_test_image`, `get_test_prompts`, `run_vl_model_test` |

## `tests/e2e/pull_request/eight_card/`

端到端 eight_card 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [model_runner_v2/](../../../tests/e2e/pull_request/eight_card/model_runner_v2) | 端到端 模型执行、模型 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/eight_card/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `tests/e2e/pull_request/eight_card/model_runner_v2/`

端到端 模型执行、模型 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/eight_card/model_runner_v2/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `tests/e2e/pull_request/four_card/`

端到端 four_card 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [_310p/](../../../tests/e2e/pull_request/four_card/_310p) | 端到端 _310p 场景、运行脚本和配置。 |
| [context_parallel/](../../../tests/e2e/pull_request/four_card/context_parallel) | 端到端 并行 场景、运行脚本和配置。 |
| [glm53flash_assets/](../../../tests/e2e/pull_request/four_card/glm53flash_assets) | 端到端 glm53flash_assets 场景、运行脚本和配置。 |
| [model_runner_v2/](../../../tests/e2e/pull_request/four_card/model_runner_v2) | 端到端 模型执行、模型 场景、运行脚本和配置。 |
| [rlhf/](../../../tests/e2e/pull_request/four_card/rlhf) | 端到端 rlhf 场景、运行脚本和配置。 |
| [spec_decode/](../../../tests/e2e/pull_request/four_card/spec_decode) | 端到端 投机解码 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/four_card/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_data_parallel_tp2.py](../../../tests/e2e/pull_request/four_card/test_data_parallel_tp2.py) | 验证 并行 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_inference_dp2_tp2` |
| [test_deepseek_v3_2_w8a8_pruning.py](../../../tests/e2e/pull_request/four_card/test_deepseek_v3_2_w8a8_pruning.py) | 验证 DeepSeek 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_moe_w8a8_tp_pp_ep_full_decode_only`, `test_pd_disaggregation_w8a8_sfa_dsa_full_decode_only`, `test_pd_disaggregation_w8a8_sfa_pcp_full_decode_only` |
| [test_deepseek_v4.py](../../../tests/e2e/pull_request/four_card/test_deepseek_v4.py) | 验证 DeepSeek 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_deepseek_v4_w4a8_tp4_basic_greedy`, `test_deepseek_v4_w4a8_tp4_index_cache_freq4` |
| [test_glm5_3_flash.py](../../../tests/e2e/pull_request/four_card/test_glm5_3_flash.py) | 验证 glm5 3 flash 的测试；具体断言场景由右栏测试符号定位。 | 注释：K3-style local dummy-model functional smoke, not accuracy or performance.；定义：`test_glm53flash_tp4` |
| [test_graph_mode.py](../../../tests/e2e/pull_request/four_card/test_graph_mode.py) | 验证 设备图 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_aclgraph` |
| [test_kimi_k3.py](../../../tests/e2e/pull_request/four_card/test_kimi_k3.py) | 验证 Kimi 模型 的测试；具体断言场景由右栏测试符号定位。 | 注释：Single-node A3 (4 logical NPUs) K3 functional tests, not accuracy tests.；定义：`test_k3_mla_block5_tp4`, `test_k3_gqa_w4a8_dp2_tp2`, `test_k3_mtp_image_tp4`, `test_k3_mla_pd_tp2` |
| [test_kvpp_pd.py](../../../tests/e2e/pull_request/four_card/test_kvpp_pd.py) | 验证 分层 KV 放置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_kvpp_pd_disaggregation` |
| [test_minimax_m3.py](../../../tests/e2e/pull_request/four_card/test_minimax_m3.py) | 验证 MiniMax 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_minimax_m3_tp4_dummy_logprobs` |
| [test_pipeline_parallel.py](../../../tests/e2e/pull_request/four_card/test_pipeline_parallel.py) | 验证 并行 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_models_pp2_dp2` |
| [test_profiling_chunk_performance.py](../../../tests/e2e/pull_request/four_card/test_profiling_chunk_performance.py) | 验证 性能分析 的测试；具体断言场景由右栏测试符号定位。 | 注释：Performance guard for profiling-based dynamic chunk sizing (PP scenario).；定义：`test_profiling_chunk_ttft_performance` |
| [test_qwen3_5.py](../../../tests/e2e/pull_request/four_card/test_qwen3_5.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_5_27b_distributed_mp_tp4`, `test_qwen3_5_35b_dp2_tp2_ep_sp_full_decode_only_mtp3` |
| [test_qwen3_mrv2_eplb.py](../../../tests/e2e/pull_request/four_card/test_qwen3_mrv2_eplb.py) | 验证 专家负载均衡、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_moe_w8a8_dp2_tp2_async_eplb_accuracy` |
| [test_qwen3_next.py](../../../tests/e2e/pull_request/four_card/test_qwen3_next.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_next_distributed_mp_graph_mode_tp4` |
| [test_qwen3_next_w8a8.py](../../../tests/e2e/pull_request/four_card/test_qwen3_next_w8a8.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_next_w8a8dynamic_distributed_mp_tp4` |
| [test_sequence_parallel_linear.py](../../../tests/e2e/pull_request/four_card/test_sequence_parallel_linear.py) | 验证 并行、线性投影 的测试；具体断言场景由右栏测试符号定位。 | 注释：Four-card DP=2/TP=2 coverage for upstream Linear sequence parallelism.；定义：`test_sequence_parallel_moe_dp2_tp2_functional`, `test_sequence_parallel_moe_dp2_tp2_precision`, `test_sequence_parallel_moe_shared_expert_dp2_tp2_precision` |

## `tests/e2e/pull_request/four_card/_310p/`

端到端 _310p 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/four_card/_310p/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_dense_model_310p.py](../../../tests/e2e/pull_request/four_card/_310p/test_dense_model_310p.py) | 验证 稠密计算、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_dense_tp4_w8a8` |
| [test_moe_model_310p.py](../../../tests/e2e/pull_request/four_card/_310p/test_moe_model_310p.py) | 验证 模型、混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_5_moe_tp4_fp16` |

## `tests/e2e/pull_request/four_card/context_parallel/`

端到端 并行 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/four_card/context_parallel/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_accuracy.py](../../../tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py) | 验证 精度 的测试；具体断言场景由右栏测试符号定位。 | 注释：DCP and DSA-CP long-sequence accuracy guards.；定义：`test_deepseek_v4_dsa_cp_prefill_decode_accuracy`, `test_models_dcp_full_feature_accuracy`, `test_models_dcp_full_graph_concurrent_requests` |
| [test_accuracy_v2.py](../../../tests/e2e/pull_request/four_card/context_parallel/test_accuracy_v2.py) | 验证 精度 的测试；具体断言场景由右栏测试符号定位。 | 注释：Model Runner V2 context-parallel accuracy and feature guards.；定义：`test_dsv3_2_sfa_pcp_model_runner_v2_graph`, `test_dsv3_2_sfa_pcp_dp_model_runner_v2_graph`, `test_dsv3_2_sfa_pcp_pp_mtp_model_runner_v2_graph`, `test_dsv3_2_sfa_pcp_dcp_model_runner_v2_graph_accuracy` |
| [test_deepseek_v4.py](../../../tests/e2e/pull_request/four_card/context_parallel/test_deepseek_v4.py) | 验证 DeepSeek 模型 的测试；具体断言场景由右栏测试符号定位。 | 注释：DeepSeek-V4 DSA-PCP tests for Model Runner V2.；定义：`test_deepseek_v4_dsa_pcp_mtp_full_decode_only`, `test_deepseek_v4_dsa_pcp_dspark` |

## `tests/e2e/pull_request/four_card/glm53flash_assets/`

端到端 glm53flash_assets 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [config.json](../../../tests/e2e/pull_request/four_card/glm53flash_assets/config.json) | 配置 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`attention_bias`, `attention_dropout`, `dtype`, `eos_token_id`, `first_k_dense_replace` |
| [multimodal.json](../../../tests/e2e/pull_request/four_card/glm53flash_assets/multimodal.json) | 多模态 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`architectures`, `model_type`, `image_token_id`, `video_token_id`, `image_start_token_id` |
| [processor_config.json](../../../tests/e2e/pull_request/four_card/glm53flash_assets/processor_config.json) | 配置 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`image_processor`, `video_processor`, `processor_class` |
| [quant.json](../../../tests/e2e/pull_request/four_card/glm53flash_assets/quant.json) | 量化 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`group_size`, `lm_head.weight`, `metadata`, `model.embed_tokens.weight`, `model.layers.0.attn_hc.base` |
| [tokenizer.json](../../../tests/e2e/pull_request/four_card/glm53flash_assets/tokenizer.json) | 分词器 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`version`, `truncation`, `padding`, `added_tokens`, `normalizer` |
| [tokenizer_config.json](../../../tests/e2e/pull_request/four_card/glm53flash_assets/tokenizer_config.json) | 分词器、配置 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`tokenizer_class`, `model_max_length`, `unk_token`, `eos_token`, `pad_token` |

## `tests/e2e/pull_request/four_card/model_runner_v2/`

端到端 模型执行、模型 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/four_card/model_runner_v2/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_deepseek_v4.py](../../../tests/e2e/pull_request/four_card/model_runner_v2/test_deepseek_v4.py) | 验证 DeepSeek 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_deepseek_v4_mtp_full_decode_only`, `test_dspark_spec_decoding` |
| [test_kvpp_pp.py](../../../tests/e2e/pull_request/four_card/model_runner_v2/test_kvpp_pp.py) | 验证 分层 KV 放置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_kvpp_pipeline_parallel` |
| [test_qwen35_pp_mtp.py](../../../tests/e2e/pull_request/four_card/model_runner_v2/test_qwen35_pp_mtp.py) | 验证 Qwen 模型、多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen35_pp_mtp_full_decode_only` |
| [test_qwen38_pp_mtp_prefix_cache.py](../../../tests/e2e/pull_request/four_card/model_runner_v2/test_qwen38_pp_mtp_prefix_cache.py) | 验证 缓存、Qwen 模型、多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 注释：Regress deferred PP Mamba state copies with a real Qwen3.8-27B model.；定义：`test_qwen38_pp_mtp_prefix_cache` |
| [test_spec_decode_data_parallel.py](../../../tests/e2e/pull_request/four_card/model_runner_v2/test_spec_decode_data_parallel.py) | 验证 投机解码、并行 的测试；具体断言场景由右栏测试符号定位。 | 注释：DSpark speculative decoding under DP with Model Runner V2.；定义：`test_deepseek_v4_dspark_spec_decoding_dp4_full_graph` |
| [test_spec_pp_accuracy.py](../../../tests/e2e/pull_request/four_card/model_runner_v2/test_spec_pp_accuracy.py) | 验证 精度 的测试；具体断言场景由右栏测试符号定位。 | 注释：DeepSeek-V4 DSpark speculative decoding accuracy on TP2xPP2 with four NPUs.；定义：`test_deepseek_v4_dspark_pp_accuracy` |

## `tests/e2e/pull_request/four_card/rlhf/`

端到端 rlhf 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [consistency/](../../../tests/e2e/pull_request/four_card/rlhf/consistency) | 端到端 consistency 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/four_card/rlhf/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `tests/e2e/pull_request/four_card/rlhf/consistency/`

端到端 consistency 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/four_card/rlhf/consistency/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_batch_invariant_tp4.py](../../../tests/e2e/pull_request/four_card/rlhf/consistency/test_batch_invariant_tp4.py) | 验证 批次 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_logprobs_bitwise_batch_invariance_bs1_vs_bsN` |

## `tests/e2e/pull_request/four_card/spec_decode/`

端到端 投机解码 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/four_card/spec_decode/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_dspark_deepseekv4.py](../../../tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py) | 验证 DeepSeek 模型、DSpark 草稿 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the short outputs of HF and vLLM when using greedy sampling.；定义：`test_deepseek_v4_dspark_acceptance_tp4` |
| [test_mtp_qwen3_next.py](../../../tests/e2e/pull_request/four_card/spec_decode/test_mtp_qwen3_next.py) | 验证 Qwen 模型、多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the short outputs of HF and vLLM when using greedy sampling.；定义：`test_qwen3_next_mtp_acceptance_tp4` |
| [test_mtp_step3p5.py](../../../tests/e2e/pull_request/four_card/spec_decode/test_mtp_step3p5.py) | 验证 多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the short outputs of HF and vLLM when using greedy sampling.；定义：`test_step3p5_mtp_acceptance_tp8` |

## `tests/e2e/pull_request/one_card/`

端到端 one_card 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [_310p/](../../../tests/e2e/pull_request/one_card/_310p) | 端到端 _310p 场景、运行脚本和配置。 |
| [aclgraph/](../../../tests/e2e/pull_request/one_card/aclgraph) | 端到端 设备图 场景、运行脚本和配置。 |
| [compile/](../../../tests/e2e/pull_request/one_card/compile) | 端到端 编译 场景、运行脚本和配置。 |
| [lora/](../../../tests/e2e/pull_request/one_card/lora) | 端到端 LoRA 适配 场景、运行脚本和配置。 |
| [model_runner_v2/](../../../tests/e2e/pull_request/one_card/model_runner_v2) | 端到端 模型执行、模型 场景、运行脚本和配置。 |
| [pooling/](../../../tests/e2e/pull_request/one_card/pooling) | 端到端 池管理 场景、运行脚本和配置。 |
| [rlhf/](../../../tests/e2e/pull_request/one_card/rlhf) | 端到端 rlhf 场景、运行脚本和配置。 |
| [spec_decode/](../../../tests/e2e/pull_request/one_card/spec_decode) | 端到端 投机解码 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [attention_utils.py](../../../tests/e2e/pull_request/one_card/attention_utils.py) | 注意力、辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`BatchSpec`, `patch_distributed_groups`, `create_common_attn_metadata`, `create_vllm_config` |
| [test_attention_v1_precision.py](../../../tests/e2e/pull_request/one_card/test_attention_v1_precision.py) | 验证 注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_causal_backend_correctness`, `test_encoder_only_backend_correctness` |
| [test_batch_job_aware_scheduler_e2e.py](../../../tests/e2e/pull_request/one_card/test_batch_job_aware_scheduler_e2e.py) | 验证 调度、批次 的测试；具体断言场景由右栏测试符号定位。 | 注释：End-to-end tests for BatchJobAwareScheduler.；定义：`test_batch_job_aware_scheduler_basic`, `test_batch_job_aware_scheduler_with_async_scheduling`, `test_batch_job_aware_scheduler_with_chunked_prefill`, `test_batch_job_aware_scheduler_with_custom_config` |
| [test_completion_with_prompt_embeds.py](../../../tests/e2e/pull_request/one_card/test_completion_with_prompt_embeds.py) | 验证 completion with prompt embeds 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mixed_prompt_embeds_and_text` |
| [test_cpu_offloading.py](../../../tests/e2e/pull_request/one_card/test_cpu_offloading.py) | 验证 缓存卸载 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_cpu_offloading` |
| [test_cpu_weight_offload.py](../../../tests/e2e/pull_request/one_card/test_cpu_weight_offload.py) | 验证 缓存卸载、权重 的测试；具体断言场景由右栏测试符号定位。 | 注释：End-to-end tests for CPU weight offloading on Ascend NPU.；定义：`test_prefetch_offload_nd_accuracy`, `test_prefetch_offload_nz_eager_accuracy`, `test_prefetch_offload_nz_graph_accuracy` |
| [test_eplb_map_to_physical_npu.py](../../../tests/e2e/pull_request/one_card/test_eplb_map_to_physical_npu.py) | 验证 专家负载均衡 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_map_and_operator_counts_run_on_npu_without_host_gating` |
| [test_gdn_layerwise_kv.py](../../../tests/e2e/pull_request/one_card/test_gdn_layerwise_kv.py) | 验证 Gated DeltaNet 注意力、KV 缓存 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_npu_connector_observes_updated_gdn_state_after_compile` |
| [test_gdn_stable_argsort.py](../../../tests/e2e/pull_request/one_card/test_gdn_stable_argsort.py) | 验证 Gated DeltaNet 注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_boolean_token_partition_avoids_aicpu_fallback`, `test_boolean_token_partition` |
| [test_guided_decoding.py](../../../tests/e2e/pull_request/one_card/test_guided_decoding.py) | 验证 guided decoding 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_guided_json_completion_xgrammar`, `test_guided_regex_xgrammar`, `test_guided_json_completion_guidance`, `test_guided_regex_guidance` |
| [test_gumbel_sampling.py](../../../tests/e2e/pull_request/one_card/test_gumbel_sampling.py) | 验证 采样 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_apply_temperature`, `test_apply_temperature_skip_zero_and_one`, `test_gumbel_sample_greedy`, `test_gumbel_sample_greedy_apply_temp_flag_irrelevant` |
| [test_linear_quantization.py](../../../tests/e2e/pull_request/one_card/test_linear_quantization.py) | 验证 线性投影、量化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_apply_with_npu`, `test_apply_with_npu`, `test_apply_with_npu`, `test_apply_with_npu` |
| [test_minicpm.py](../../../tests/e2e/pull_request/one_card/test_minicpm.py) | 验证 minicpm 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_minicpm` |
| [test_minimax_m3_sparse_attn.py](../../../tests/e2e/pull_request/one_card/test_minimax_m3_sparse_attn.py) | 验证 MiniMax 模型、注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：Correctness tests for MiniMax M3 sparse-attention kernels on Ascend.；定义：`test_prefill_index_topk_correctness`, `test_prefill_index_topk_production_shape`, `test_prefill_index_topk_production_long_sequence`, `test_decode_index_topk_correctness` |
| [test_minimax_qknorm_rope_cache.py](../../../tests/e2e/pull_request/one_card/test_minimax_qknorm_rope_cache.py) | 验证 MiniMax 模型、缓存、位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_minimax_qknorm_rope_cache`, `test_empty_minimax_qknorm_rope_cache`, `test_dynamic_cache_capacity_and_strided_positions`, `test_cache_graph_replay_updates_slots_and_preserves_strides` |
| [test_mla_precision.py](../../../tests/e2e/pull_request/one_card/test_mla_precision.py) | 验证 MLA 潜在注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mla_backend_correctness` |
| [test_model_runner_v1_with_device.py](../../../tests/e2e/pull_request/one_card/test_model_runner_v1_with_device.py) | 验证 模型执行、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_determine_batch_execution_and_padding`, `test_stateful_handoff_preserves_decode_graph` |
| [test_msa_index_score.py](../../../tests/e2e/pull_request/one_card/test_msa_index_score.py) | 验证 msa index score 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_msa_index_score_matches_cpu_golden` |
| [test_multi_instance.py](../../../tests/e2e/pull_request/one_card/test_multi_instance.py) | 验证 multi instance 的测试；具体断言场景由右栏测试符号定位。 | 注释：Two VllmRunner instances are nested so that the first instance's worker；定义：`test_two_instances_on_single_card` |
| [test_multistream_overlap_shared_expert.py](../../../tests/e2e/pull_request/one_card/test_multistream_overlap_shared_expert.py) | 验证 专家、设备流 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the outputs of vLLM with multistream_overlap_shared_expert；定义：`test_models_with_multistream_overlap_shared_expert` |
| [test_qwen3_0_6b.py](../../../tests/e2e/pull_request/one_card/test_qwen3_0_6b.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dense_default_full_and_piecewise_graph` |
| [test_qwen3_5_0_8b.py](../../../tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mamba_ssm_multimodal_reasoning_mtp_full_decode_only`, `test_hybrid_prefix_match_unit_cached_output_consistency` |
| [test_qwen3_8b_w8a8.py](../../../tests/e2e/pull_request/one_card/test_qwen3_8b_w8a8.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dense_w8a8_eagle3_full_graph` |
| [test_qwen3_embedding_0_6b.py](../../../tests/e2e/pull_request/one_card/test_qwen3_embedding_0_6b.py) | 验证 向量查表、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_embedding_full_decode_only` |
| [test_sampler.py](../../../tests/e2e/pull_request/one_card/test_sampler.py) | 验证 采样 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_topk`, `test_qwen3_prompt_logprobs` |
| [test_sfa_v1_precision.py](../../../tests/e2e/pull_request/one_card/test_sfa_v1_precision.py) | 验证 稀疏注意力 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_sfa_sparse_flash_attention_precision` |
| [test_simple_cpu_offload.py](../../../tests/e2e/pull_request/one_card/test_simple_cpu_offload.py) | 验证 缓存卸载 的测试；具体断言场景由右栏测试符号定位。 | 注释：End-to-end tests for the Ascend ``SimpleCPUOffloadConnector``.；定义：`test_simple_cpu_offload_accuracy`, `test_simple_cpu_offload_no_crash_on_repeat` |
| [test_thinking_budget.py](../../../tests/e2e/pull_request/one_card/test_thinking_budget.py) | 验证 thinking budget 的测试；具体断言场景由右栏测试符号定位。 | 注释：Ascend coverage for the upstream Model Runner V2 thinking-budget path.；定义：`test_v2_thinking_budget_forces_end_after_budget_reached`, `test_v2_thinking_budget_restores_masked_end_token`, `test_v2_thinking_budget_allows_tokens_before_budget`, `test_v2_thinking_budget_continues_multi_token_end_marker` |
| [test_turboquant_finalize.py](../../../tests/e2e/pull_request/one_card/test_turboquant_finalize.py) | 验证 收尾、量化 的测试；具体断言场景由右栏测试符号定位。 | 注释：Numerical and graph coverage for framework TurboQuant cache packing.；定义：`test_fused_scale_and_packing`, `test_finalize_reuses_kernel_for_new_row_counts`, `test_finalize_random_codes_match_scale_reference` |
| [test_vlm.py](../../../tests/e2e/pull_request/one_card/test_vlm.py) | 验证 vlm 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the short outputs of HF and vLLM when using greedy sampling.；定义：`test_multimodal_vl`, `test_multimodal_vl_language_model_only`, `test_multimodal_audio`, `test_whisper` |
| [test_xlite.py](../../../tests/e2e/pull_request/one_card/test_xlite.py) | 验证 xlite 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the outputs of vLLM with and without xlite via logprob-based accuracy；定义：`test_models_with_xlite_decode_only`, `test_models_with_xlite_full_mode` |

## `tests/e2e/pull_request/one_card/_310p/`

端到端 _310p 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/_310p/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_classification_310p.py](../../../tests/e2e/pull_request/one_card/_310p/test_classification_310p.py) | 验证 classification 310p 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen_pooling_classify_correctness` |
| [test_dense_model_310p.py](../../../tests/e2e/pull_request/one_card/_310p/test_dense_model_310p.py) | 验证 稠密计算、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_dense_tp1_w8a8_aclgraph` |
| [test_embedding_310p.py](../../../tests/e2e/pull_request/one_card/_310p/test_embedding_310p.py) | 验证 向量查表 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_embed_models_correctness`, `test_qwen3_vl_embedding_mrv2_pooling`, `test_bge_m3_correctness` |
| [test_lora_310p.py](../../../tests/e2e/pull_request/one_card/_310p/test_lora_310p.py) | 验证 LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen35_lora_with_aclgraph_tp1_fp16` |
| [test_qwen35_dense_model_310p.py](../../../tests/e2e/pull_request/one_card/_310p/test_qwen35_dense_model_310p.py) | 验证 稠密计算、模型、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_5_dense_prefix_mamba_cache_tp1_fp16`, `test_qwen3_5_dense_tp1_fp16_aclgraph` |
| [test_sampling_temp_310p.py](../../../tests/e2e/pull_request/one_card/_310p/test_sampling_temp_310p.py) | 验证 采样 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_mrv2_310p_topk` |
| [test_scoring_310p.py](../../../tests/e2e/pull_request/one_card/_310p/test_scoring_310p.py) | 验证 scoring 310p 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_cross_encoder_score_1_to_1`, `test_cross_encoder_score_1_to_N`, `test_cross_encoder_score_N_to_N` |
| [test_spec_decode_mtp_310p.py](../../../tests/e2e/pull_request/one_card/_310p/test_spec_decode_mtp_310p.py) | 验证 投机解码、多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 注释：310P MTP e2e: MRv1 baseline + MRv2 eager smoke (1-card CI safe).；定义：`test_qwen3_5_mtp_tp1_eager`, `test_qwen3_5_mtp_mrv2_tp1_eager` |
| [test_vl_model_310p.py](../../../tests/e2e/pull_request/one_card/_310p/test_vl_model_310p.py) | 验证 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_vl_8b_tp1_fp16` |

## `tests/e2e/pull_request/one_card/aclgraph/`

端到端 设备图 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/aclgraph/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_device_metadata.py](../../../tests/e2e/pull_request/one_card/aclgraph/test_device_metadata.py) | 验证 元数据 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_metadata_order_and_buffer_reuse` |

## `tests/e2e/pull_request/one_card/compile/`

端到端 编译 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/compile/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [backend.py](../../../tests/e2e/pull_request/one_card/compile/backend.py) | backend 的测试辅助、测试入口或共享数据结构。 | 定义：`TestBackend` |
| [test_graphex_norm_quant_fusion.py](../../../tests/e2e/pull_request/one_card/compile/test_graphex_norm_quant_fusion.py) | 验证 算子融合、量化、设备图 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_rmsnorm_quant_fusion` |
| [test_graphex_qknorm_rope_fusion.py](../../../tests/e2e/pull_request/one_card/compile/test_graphex_qknorm_rope_fusion.py) | 验证 算子融合、设备图、位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_rmsnorm_quant_fusion` |
| [test_graphex_qkvnorm_rope_fusion.py](../../../tests/e2e/pull_request/one_card/compile/test_graphex_qkvnorm_rope_fusion.py) | 验证 算子融合、设备图、位置旋转 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qkvnorm_rope_fusion`, `test_qkvnorm_rope_fusion_real_modules` |
| [test_norm_quant_fusion.py](../../../tests/e2e/pull_request/one_card/compile/test_norm_quant_fusion.py) | 验证 算子融合、量化 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_rmsnorm_quant_fusion` |

## `tests/e2e/pull_request/one_card/lora/`

端到端 LoRA 适配 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/lora/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_ilama_lora.py](../../../tests/e2e/pull_request/one_card/lora/test_ilama_lora.py) | 验证 LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ilama_lora` |
| [test_llama32_lora.py](../../../tests/e2e/pull_request/one_card/lora/test_llama32_lora.py) | 验证 Llama 模型、LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_llama_lora` |
| [test_lora_with_spec_decode.py](../../../tests/e2e/pull_request/one_card/lora/test_lora_with_spec_decode.py) | 验证 投机解码、LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 注释：This script contains:；定义：`test_batch_inference_correctness` |
| [test_olmoe_lora.py](../../../tests/e2e/pull_request/one_card/lora/test_olmoe_lora.py) | 验证 LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_olmoe_lora` |
| [test_qwen35_densemodel_lora.py](../../../tests/e2e/pull_request/one_card/lora/test_qwen35_densemodel_lora.py) | 验证 稠密计算、模型、LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen35_text_lora` |
| [test_qwen3_multi_loras.py](../../../tests/e2e/pull_request/one_card/lora/test_qwen3_multi_loras.py) | 验证 LoRA 适配、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_multi_loras_with_tp_sync` |
| [test_qwen3_reranker_lora.py](../../../tests/e2e/pull_request/one_card/lora/test_qwen3_reranker_lora.py) | 验证 LoRA 适配、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_reranker_models_lora` |
| [test_qwenvl_lora.py](../../../tests/e2e/pull_request/one_card/lora/test_qwenvl_lora.py) | 验证 LoRA 适配、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen2vl_lora`, `test_qwen25vl_lora`, `test_qwen3vl_vision_lora` |

## `tests/e2e/pull_request/one_card/model_runner_v2/`

端到端 模型执行、模型 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_deepseek_mtp.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/test_deepseek_mtp.py) | 验证 DeepSeek 模型、多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_mtp_spec_decoding` |
| [test_deepseek_v2_lite_w8a8_basic.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/test_deepseek_v2_lite_w8a8_basic.py) | 验证 DeepSeek 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_deepseek_v2_lite_eager_mode`, `test_deepseek_v2_lite_graph_mode` |
| [test_llama3_1_8b_eagle.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/test_llama3_1_8b_eagle.py) | 验证 Eagle 投机、Llama 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_eagle_spec_decoding` |
| [test_qwen3_0_6B_basic.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/test_qwen3_0_6B_basic.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_dense_eager_mode`, `test_qwen3_dense_graph_mode` |
| [test_qwen3_8B.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/test_qwen3_8B.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_eagle3_dsd` |
| [test_qwen3_8b_dflash.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/test_qwen3_8b_dflash.py) | 验证 DFlash 草稿、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dflash_spec_decoding` |
| [test_qwen3_8b_dspark.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/test_qwen3_8b_dspark.py) | 验证 DSpark 草稿、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dspark_spec_decoding`, `test_dspark_probabilistic_spec_decoding` |
| [test_uva.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/test_uva.py) | 验证 uva 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_dense_eager_mode` |
| [utils.py](../../../tests/e2e/pull_request/one_card/model_runner_v2/utils.py) | 辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`calculate_acceptance_per_pos` |

## `tests/e2e/pull_request/one_card/pooling/`

端到端 池管理 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [template/](../../../tests/e2e/pull_request/one_card/pooling/template) | 端到端 template 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/pooling/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_classification.py](../../../tests/e2e/pull_request/one_card/pooling/test_classification.py) | 验证 classification 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen_pooling_classify_correctness` |
| [test_embedding.py](../../../tests/e2e/pull_request/one_card/pooling/test_embedding.py) | 验证 向量查表 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_embed_models_correctness`, `test_causal_embed_models_using_prefix_caching_correctness`, `test_bge_m3_correctness` |
| [test_qwen3_embedding_mrv2.py](../../../tests/e2e/pull_request/one_card/pooling/test_qwen3_embedding_mrv2.py) | 验证 向量查表、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_embedding_mrv2_pooling_a2` |
| [test_scoring.py](../../../tests/e2e/pull_request/one_card/pooling/test_scoring.py) | 验证 scoring 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_cross_encoder_score_1_to_1`, `test_cross_encoder_score_1_to_N`, `test_cross_encoder_score_N_to_N`, `test_embedding_score_1_to_1` |

## `tests/e2e/pull_request/one_card/pooling/template/`

端到端 template 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [qwen3_reranker.jinja](../../../tests/e2e/pull_request/one_card/pooling/template/qwen3_reranker.jinja) | Qwen 模型 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |

## `tests/e2e/pull_request/one_card/rlhf/`

端到端 rlhf 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [consistency/](../../../tests/e2e/pull_request/one_card/rlhf/consistency) | 端到端 consistency 场景、运行脚本和配置。 |
| [state_transitions/](../../../tests/e2e/pull_request/one_card/rlhf/state_transitions) | 端到端 状态 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/rlhf/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [conftest.py](../../../tests/e2e/pull_request/one_card/rlhf/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 注释：Shared fixtures and helpers for RLHF end-to-end tests.；定义：`server`, `poll_until`, `gen` |

## `tests/e2e/pull_request/one_card/rlhf/consistency/`

端到端 consistency 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/rlhf/consistency/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_attention_fa3.py](../../../tests/e2e/pull_request/one_card/rlhf/consistency/test_attention_fa3.py) | 验证 注意力 的测试；具体断言场景由右栏测试符号定位。 | 注释：End-to-end test for Flash Attention 3 (FA3) on Ascend.；定义：`test_fa3_vs_fia_single_prompt`, `test_fa3_vs_fia_mixed_lengths`, `test_fa3_vs_fia_with_chunkprefill`, `test_fa3_vs_fia_logprobs` |

## `tests/e2e/pull_request/one_card/rlhf/state_transitions/`

端到端 状态 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/rlhf/state_transitions/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_npu_ipc_weight_transfer.py](../../../tests/e2e/pull_request/one_card/rlhf/state_transitions/test_npu_ipc_weight_transfer.py) | 验证 传输、权重、进程间通信 的测试；具体断言场景由右栏测试符号定位。 | 注释：Model-aware end-to-end tests for NPU IPC live weight updates.；定义：`test_npu_ipc_weight_transfer_transaction` |
| [test_pause_resume.py](../../../tests/e2e/pull_request/one_card/rlhf/state_transitions/test_pause_resume.py) | 验证 pause resume 的测试；具体断言场景由右栏测试符号定位。 | 注释：End-to-end tests for the vLLM RL /pause /resume /is_paused lifecycle on Ascend NPU.；定义：`test_pause_mode_wait_drains_inflight_request`, `test_pause_mode_keep_resumes_frozen_request` |
| [test_sleep_wake.py](../../../tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_wake.py) | 验证 休眠资源 的测试；具体断言场景由右栏测试符号定位。 | 注释：End-to-end tests for the vLLM RL sleep/wake lifecycle on Ascend NPU.；定义：`test_sleep_level1_frees_npu_memory`, `test_sleep_level2_frees_all_discards_all`, `test_staged_release_each_step_changes_memory`, `test_staged_wake_restores_output` |

## `tests/e2e/pull_request/one_card/spec_decode/`

端到端 投机解码 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/one_card/spec_decode/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [conftest.py](../../../tests/e2e/pull_request/one_card/spec_decode/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 定义：`test_prompts` |
| [test_dflash.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_dflash.py) | 验证 DFlash 草稿 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dflash_acceptance` |
| [test_draft_parallel.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_draft_parallel.py) | 验证 并行、草稿模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_parallel_drafting_acceptance` |
| [test_dspark.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_dspark.py) | 验证 DSpark 草稿 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dspark_acceptance`, `test_dspark_kv_sliding_window` |
| [test_dynamic.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_dynamic.py) | 验证 dynamic 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_dynamic_spec_acceptance` |
| [test_eagle.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_eagle.py) | 验证 Eagle 投机 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_vl_eagle`, `test_qwen_eagle3_acceptance` |
| [test_extract_hidden_states.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py) | 验证 状态 的测试；具体断言场景由右栏测试符号定位。 | 注释：E2E tests for the extract_hidden_states speculative decoding method.；定义：`test_extract_hidden_states` |
| [test_mtp_eagle_correctness.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_mtp_eagle_correctness.py) | 验证 Eagle 投机、多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the short outputs of HF and vLLM when using greedy sampling.；定义：`test_deepseek_mtp` |
| [test_ngram.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_ngram.py) | 验证 n-gram 候选 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ngram` |
| [test_ngram_npu.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_ngram_npu.py) | 验证 n-gram 候选 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ngram_npu_async_acceptance` |
| [test_suffix.py](../../../tests/e2e/pull_request/one_card/spec_decode/test_suffix.py) | 验证 后缀候选 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_suffix_acceptance` |
| [utils.py](../../../tests/e2e/pull_request/one_card/spec_decode/utils.py) | 辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`eagle_model_name`, `eagle3_model_name`, `vl_eagle3_model_name` |

## `tests/e2e/pull_request/rlhf/`

端到端 rlhf 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/rlhf/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [weight_transfer_test_utils.py](../../../tests/e2e/pull_request/rlhf/weight_transfer_test_utils.py) | 传输、权重、辅助函数 的测试辅助、测试入口或共享数据结构。 | 注释：Shared model matrix and assertions for one/two-card weight-update E2Es.；定义：`WeightUpdateModelCase`, `FixedRandomWeightSource`, `deepseek_v4_checkpoint_name`, `expand_fused_expert_params`, `resize_expert_shape` |

## `tests/e2e/pull_request/two_card/`

端到端 two_card 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [_310p/](../../../tests/e2e/pull_request/two_card/_310p) | 端到端 _310p 场景、运行脚本和配置。 |
| [aclgraph/](../../../tests/e2e/pull_request/two_card/aclgraph) | 端到端 设备图 场景、运行脚本和配置。 |
| [lora/](../../../tests/e2e/pull_request/two_card/lora) | 端到端 LoRA 适配 场景、运行脚本和配置。 |
| [model_runner_v2/](../../../tests/e2e/pull_request/two_card/model_runner_v2) | 端到端 模型执行、模型 场景、运行脚本和配置。 |
| [pooling/](../../../tests/e2e/pull_request/two_card/pooling) | 端到端 池管理 场景、运行脚本和配置。 |
| [rlhf/](../../../tests/e2e/pull_request/two_card/rlhf) | 端到端 rlhf 场景、运行脚本和配置。 |
| [spec_decode/](../../../tests/e2e/pull_request/two_card/spec_decode) | 端到端 投机解码 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/two_card/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_activation.py](../../../tests/e2e/pull_request/two_card/test_activation.py) | 验证 激活函数 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_swiglu_oai_forward_matches_reference_formula`, `test_swiglu_oai_forward_uses_interleaved_gate_and_up_layout`, `test_swiglu_oai_forward_with_custom_alpha_and_limit_matches_reference`, `test_swiglu_oai_forward_clamps_gate_and_up_values` |
| [test_data_parallel.py](../../../tests/e2e/pull_request/two_card/test_data_parallel.py) | 验证 并行 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the outputs of vLLM with and without aclgraph.；定义：`test_qwen3_inference_dp2` |
| [test_deepseek_multistream_moe.py](../../../tests/e2e/pull_request/two_card/test_deepseek_multistream_moe.py) | 验证 DeepSeek 模型、设备流、混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_deepseek_multistream_moe_tp2` |
| [test_disaggregated_encoder.py](../../../tests/e2e/pull_request/two_card/test_disaggregated_encoder.py) | 验证 编码器、门控 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_models` |
| [test_gemma4.py](../../../tests/e2e/pull_request/two_card/test_gemma4.py) | 验证 Gemma 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_gemma4_26b_a4b_moe_full_decode_graph` |
| [test_gpt_oss_distributed.py](../../../tests/e2e/pull_request/two_card/test_gpt_oss_distributed.py) | 验证 gpt oss distributed 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the short outputs of HF and vLLM when using greedy sampling.；定义：`test_gpt_oss_distributed_tp2` |
| [test_kvpp.py](../../../tests/e2e/pull_request/two_card/test_kvpp.py) | 验证 分层 KV 放置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_kvpp_combined_features` |
| [test_pipeline_parallel.py](../../../tests/e2e/pull_request/two_card/test_pipeline_parallel.py) | 验证 并行 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_models_pp2_tp2` |
| [test_prefix_caching.py](../../../tests/e2e/pull_request/two_card/test_prefix_caching.py) | 验证 prefix caching 的测试；具体断言场景由右栏测试符号定位。 | 注释：Compare the with and without prefix caching.；定义：`test_models_prefix_cache_tp2` |
| [test_qwen3_30b_a3b.py](../../../tests/e2e/pull_request/two_card/test_qwen3_30b_a3b.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_moe_tp_ep_eplb_full_decode_only` |
| [test_qwen3_5_35b_a3b_w8a8.py](../../../tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_5_35b_a3b_w8a8_tp2_without_ep` |
| [test_qwen3_6_27b_fia.py](../../../tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_6_27b_multimodel_fia_eager`, `test_qwen3_6_27b_multimodel_fia_acl_graph` |
| [test_qwen3_moe_eplb.py](../../../tests/e2e/pull_request/two_card/test_qwen3_moe_eplb.py) | 验证 专家负载均衡、Qwen 模型、混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_moe_w8a8_distributed_tp2_ep_dynamic_eplb` |
| [test_qwen3_performance.py](../../../tests/e2e/pull_request/two_card/test_qwen3_performance.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_models` |
| [test_qwen3_vl_30b_a3b_instruct.py](../../../tests/e2e/pull_request/two_card/test_qwen3_vl_30b_a3b_instruct.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_multimodal_reasoning_pp_full_decode_only` |
| [test_shared_expert_dp.py](../../../tests/e2e/pull_request/two_card/test_shared_expert_dp.py) | 验证 专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_deepseek_v2_lite_enable_shared_expert_dp_tp2_eager` |
| [test_shared_expert_dp_graph.py](../../../tests/e2e/pull_request/two_card/test_shared_expert_dp_graph.py) | 验证 专家、设备图 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_deepseek_v2_lite_enable_shared_expert_dp_tp2_graph` |
| [test_xlite.py](../../../tests/e2e/pull_request/two_card/test_xlite.py) | 验证 xlite 的测试；具体断言场景由右栏测试符号定位。 | 注释：Verify MoE serving with xlite graph mode on two NPU cards.；定义：`test_models_with_xlite_decode_only`, `test_models_with_xlite_full_mode` |

## `tests/e2e/pull_request/two_card/_310p/`

端到端 _310p 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_dense_model_310p.py](../../../tests/e2e/pull_request/two_card/_310p/test_dense_model_310p.py) | 验证 稠密计算、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_dense_tp2_fp16` |
| [test_model_runner_v2_310p.py](../../../tests/e2e/pull_request/two_card/_310p/test_model_runner_v2_310p.py) | 验证 模型执行、模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_dense_mrv2_tp2_aclgraph_fp16`, `test_qwen3_moe_mrv2_tp2_aclgraph_w8a8`, `test_qwen35_hybrid_mrv2_tp2_aclgraph_fp16`, `test_qwen35_hybrid_prefix_mamba_cache_mrv2_tp2_fp16` |
| [test_moe_model_310p.py](../../../tests/e2e/pull_request/two_card/_310p/test_moe_model_310p.py) | 验证 模型、混合专家 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_moe_tp2_w8a8` |
| [test_vl_model_310p.py](../../../tests/e2e/pull_request/two_card/_310p/test_vl_model_310p.py) | 验证 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_vl_8b_tp2_fp16` |

## `tests/e2e/pull_request/two_card/aclgraph/`

端到端 设备图 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_aclgraph_capture_replay.py](../../../tests/e2e/pull_request/two_card/aclgraph/test_aclgraph_capture_replay.py) | 验证 设备图 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_models_aclgraph_capture_replay_metrics_dp2` |

## `tests/e2e/pull_request/two_card/lora/`

端到端 LoRA 适配 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/two_card/lora/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_ilama_lora_tp2.py](../../../tests/e2e/pull_request/two_card/lora/test_ilama_lora_tp2.py) | 验证 LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_ilama_lora_tp2` |
| [test_llama32_lora_tp2.py](../../../tests/e2e/pull_request/two_card/lora/test_llama32_lora_tp2.py) | 验证 Llama 模型、LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_llama_lora_tp2` |
| [test_qwen35_4b_multi_lora_mtp_aclgraph.py](../../../tests/e2e/pull_request/two_card/lora/test_qwen35_4b_multi_lora_mtp_aclgraph.py) | 验证 设备图、LoRA 适配、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen35_4b_named_multi_lora_mtp_aclgraph` |
| [test_qwen35_densemodel_lora_tp.py](../../../tests/e2e/pull_request/two_card/lora/test_qwen35_densemodel_lora_tp.py) | 验证 稠密计算、模型、LoRA 适配 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen35_text_lora` |
| [test_qwen3moe_lora.py](../../../tests/e2e/pull_request/two_card/lora/test_qwen3moe_lora.py) | 验证 LoRA 适配、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3moe_lora_tp`, `test_qwen3moe_lora_ep`, `test_qwen3moe_lora_multi_id_ep` |

## `tests/e2e/pull_request/two_card/model_runner_v2/`

端到端 模型执行、模型 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_data_parallel.py](../../../tests/e2e/pull_request/two_card/model_runner_v2/test_data_parallel.py) | 验证 并行 的测试；具体断言场景由右栏测试符号定位。 | 注释：Data parallel inference with Model Runner V2.；定义：`test_dsv2_w8a8_dp_eager_mode`, `test_dsv2_w8a8_dp_graph_mode` |
| [test_mamba_hybrid.py](../../../tests/e2e/pull_request/two_card/model_runner_v2/test_mamba_hybrid.py) | 验证 Mamba 状态空间 的测试；具体断言场景由右栏测试符号定位。 | 注释：Inference with Model Runner V2 for Mamba/hybrid model.；定义：`test_qwen35_27b_eager_mode`, `test_qwen35_27b_acl_graph` |
| [test_qwen3_5_eagle3.py](../../../tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_5_eagle3.py) | 验证 Eagle 投机、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 注释：Qwen3.6-27B BF16 EAGLE3 acceptance on MRV2 over two NPUs.；定义：`test_qwen35_9b_eagle3_acceptance_tp2` |
| [test_qwen3_6_dflash.py](../../../tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_dflash.py) | 验证 DFlash 草稿、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 注释：Qwen3.6-35B-A3B BF16 DFLASH acceptance on MRV2 over two NPUs.；定义：`test_qwen36_35b_dflash_acceptance_tp2` |
| [test_qwen3_6_dspark.py](../../../tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_dspark.py) | 验证 DSpark 草稿、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 注释：Qwen3.6-27B BF16 DSPARK acceptance on MRV2 over two NPUs.；定义：`test_qwen36_27b_dspark_acceptance_tp2` |
| [test_qwen3_6_mtp.py](../../../tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_mtp.py) | 验证 Qwen 模型、多 token 预测 的测试；具体断言场景由右栏测试符号定位。 | 注释：Qwen3.6 BF16 MTP acceptance on MRV2 with fixed goldens on two NPUs.；定义：`test_qwen3_6_mtp_acceptance_tp2` |
| [test_spec_decode_data_parallel.py](../../../tests/e2e/pull_request/two_card/model_runner_v2/test_spec_decode_data_parallel.py) | 验证 投机解码、并行 的测试；具体断言场景由右栏测试符号定位。 | 注释：Speculative decoding under DP with Model Runner V2.；定义：`test_mtp_spec_decoding_dp` |

## `tests/e2e/pull_request/two_card/pooling/`

端到端 池管理 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_qwen3_embedding_mrv2.py](../../../tests/e2e/pull_request/two_card/pooling/test_qwen3_embedding_mrv2.py) | 验证 向量查表、Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_embedding_mrv2_pooling_a3` |

## `tests/e2e/pull_request/two_card/rlhf/`

端到端 rlhf 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [consistency/](../../../tests/e2e/pull_request/two_card/rlhf/consistency) | 端到端 consistency 场景、运行脚本和配置。 |
| [state_transitions/](../../../tests/e2e/pull_request/two_card/rlhf/state_transitions) | 端到端 状态 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/two_card/rlhf/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `tests/e2e/pull_request/two_card/rlhf/consistency/`

端到端 consistency 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/two_card/rlhf/consistency/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_moe_routing_replay.py](../../../tests/e2e/pull_request/two_card/rlhf/consistency/test_moe_routing_replay.py) | 验证 混合专家 的测试；具体断言场景由右栏测试符号定位。 | 注释：End-to-end routing replay consistency tests for MoE models.；定义：`test_moe_routing_replay` |

## `tests/e2e/pull_request/two_card/rlhf/state_transitions/`

端到端 状态 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/two_card/rlhf/state_transitions/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_hccl_weight_transfer.py](../../../tests/e2e/pull_request/two_card/rlhf/state_transitions/test_hccl_weight_transfer.py) | 验证 传输、权重、HCCL 通信 的测试；具体断言场景由右栏测试符号定位。 | 注释：Model-aware two-card HCCL live weight-update regression tests.；定义：`test_hccl_weight_transfer_transaction` |

## `tests/e2e/pull_request/two_card/spec_decode/`

端到端 投机解码 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/pull_request/two_card/spec_decode/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_spec_decode.py](../../../tests/e2e/pull_request/two_card/spec_decode/test_spec_decode.py) | 验证 投机解码 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_eagle3_sp_acceptance`, `test_p_eagle_acceptance`, `test_dflash2_acceptance`, `test_dflash2_v2_acceptance` |
| [test_spec_decode_heavy.py](../../../tests/e2e/pull_request/two_card/spec_decode/test_spec_decode_heavy.py) | 验证 投机解码 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_vwn_eagle3_tp2`, `test_eagle3_sliding_window`, `test_hang`, `test_qwen36_35b_dspark_spec_decoding` |
