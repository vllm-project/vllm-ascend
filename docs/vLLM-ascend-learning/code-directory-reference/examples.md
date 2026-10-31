# examples：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **52 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `examples/`

离线推理、分布式部署、权重更新和量化等参考使用示例。

| 子目录 | 职责 |
| --- | --- |
| [chat_templates/](../../../examples/chat_templates) | chat_templates 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [disaggregated_encoder/](../../../examples/disaggregated_encoder) | 编码器、门控 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [disaggregated_prefill_v1/](../../../examples/disaggregated_prefill_v1) | 上下文预填充、门控 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [dynamic_bucket_load_balancer/](../../../examples/dynamic_bucket_load_balancer) | 加载 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [epd_disaggregated/](../../../examples/epd_disaggregated) | 门控 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [eplb/](../../../examples/eplb) | 专家负载均衡 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [external_online_dp/](../../../examples/external_online_dp) | external_online_dp 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [offline_DSD_k_tuner/](../../../examples/offline_DSD_k_tuner) | offline_DSD_k_tuner 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [quantization/](../../../examples/quantization) | 生成量化 checkpoint 的参考配置与脚本。 |
| [rfork/](../../../examples/rfork) | RFork 权重加载 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [rl/](../../../examples/rl) | rl 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [device_id_mapping_demo.py](../../../examples/device_id_mapping_demo.py) | device id mapping demo 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`main` |
| [device_print_demo.py](../../../examples/device_print_demo.py) | device print demo 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`compute_and_print`, `main` |
| [offline_data_parallel.py](../../../examples/offline_data_parallel.py) | 并行 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Usage:；定义：`parse_args`, `cleanup_env_and_memory`, `main` |
| [offline_disaggregated_prefill_npu.py](../../../examples/offline_disaggregated_prefill_npu.py) | 上下文预填充、门控 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`clean_up`, `run_prefill`, `run_decode` |
| [offline_embed.py](../../../examples/offline_embed.py) | offline embed 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`get_detailed_instruct`, `main` |
| [offline_external_launcher.py](../../../examples/offline_external_launcher.py) | offline external launcher 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Usage:；定义：`parse_args`, `main`, `cleanup_env_and_memory` |
| [offline_inference_audio_language.py](../../../examples/offline_inference_audio_language.py) | offline inference audio language 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：This example shows how to use vLLM for running offline inference；定义：`prepare_inputs`, `main` |
| [offline_inference_metrics.py](../../../examples/offline_inference_metrics.py) | 指标 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Example: Access request-level metrics from vLLM outputs.；定义：`main` |
| [offline_inference_npu.py](../../../examples/offline_inference_npu.py) | offline inference npu 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`main` |
| [offline_inference_npu_long_seq.py](../../../examples/offline_inference_npu_long_seq.py) | offline inference npu long seq 使用示例或配套配置；展示调用方式而非引擎入口。 | 导入：`argparse`, `os` |
| [offline_inference_npu_tp2.py](../../../examples/offline_inference_npu_tp2.py) | offline inference npu tp2 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`main` |
| [offline_inference_sleep_mode_npu.py](../../../examples/offline_inference_sleep_mode_npu.py) | 休眠资源 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`main` |
| [offline_reload_weight.py](../../../examples/offline_reload_weight.py) | 权重、加载 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`print_prompts_and_outputs`, `main` |
| [offline_weight_load.py](../../../examples/offline_weight_load.py) | 权重、加载 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Usage:；定义：`parse_args`, `main`, `cleanup_env_and_memory` |
| [prefix_cache_repro.py](../../../examples/prefix_cache_repro.py) | 缓存 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Reproduce output drift between cold, warm, and concurrent requests.；定义：`parse_args`, `main` |
| [prompt_embed_inference.py](../../../examples/prompt_embed_inference.py) | prompt embed inference 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Demonstrates how to generate prompt embeddings using；定义：`init_tokenizer_and_llm`, `get_prompt_embeds`, `single_prompt_inference` |
| [prompt_embedding_inference.py](../../../examples/prompt_embedding_inference.py) | 向量查表 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`init_tokenizer_and_llm`, `get_prompt_embeds`, `single_prompt_inference` |
| [run_dp_server.sh](../../../examples/run_dp_server.sh) | run dp server 使用示例或配套配置；展示调用方式而非引擎入口。 | 命令类别：`vllm` |
| [save_sharded_state_310.py](../../../examples/save_sharded_state_310.py) | 状态 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Saves each worker's model state dict directly to a checkpoint, which enables a；定义：`FileHandler`, `clean_up`, `parse_args`, `get_quant_description` |

## `examples/chat_templates/`

chat_templates 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [template_qwen2_audio.jinja](../../../examples/chat_templates/template_qwen2_audio.jinja) | Qwen 模型 使用示例或配套配置；展示调用方式而非引擎入口。 | 依据文件格式及所在目录归类。 |

## `examples/disaggregated_encoder/`

编码器、门控 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [disagg_1e1pd_example.sh](../../../examples/disaggregated_encoder/disagg_1e1pd_example.sh) | disagg 1e1pd example 使用示例或配套配置；展示调用方式而非引擎入口。 | 函数：`wait_for_server`, `cleanup` |
| [disagg_epd_proxy.py](../../../examples/disaggregated_encoder/disagg_epd_proxy.py) | disagg epd proxy 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：disagg_encoder_proxy.py；定义：`EncoderDispatchMode`, `extract_mm_items`, `fanout_encoder_primer`, `maybe_prefill` |

## `examples/disaggregated_prefill_v1/`

上下文预填充、门控 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [load_balance_proxy_layerwise_server_example.py](../../../examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py) | 加载 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`ServerState`, `ProxyState`, `parse_args`, `lifespan`, `listen_for_disconnect` |
| [load_balance_proxy_server_example.py](../../../examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) | 加载 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`ServerRole`, `InstanceInfo`, `extract_cached_tokens`, `update_cached_tokens_in_chunk`, `encode_response_chunk` |
| [mooncake_connector_deployment_guide.md](../../../examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：mooncake_connector_deployment_guide。 | 标题：Mooncake connector deployment Guide |

## `examples/dynamic_bucket_load_balancer/`

加载 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../examples/dynamic_bucket_load_balancer/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：Dynamic Bucket Load Balancer |
| [dynamic_bucket_load_balancer.py](../../../examples/dynamic_bucket_load_balancer/dynamic_bucket_load_balancer.py) | 加载 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`Task`, `Bucket` |
| [hybrid_proxy_server.py](../../../examples/dynamic_bucket_load_balancer/hybrid_proxy_server.py) | hybrid proxy server 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`ServerState`, `ServerHeapItem`, `parse_args`, `lifespan`, `listen_for_disconnect` |

## `examples/epd_disaggregated/`

门控 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [epd_disaggregated_guide.md](../../../examples/epd_disaggregated/epd_disaggregated_guide.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：epd_disaggregated_guide。 | 标题：EPD disaggregated deployment Guide |
| [epd_load_balance_proxy_layerwise_server_example.py](../../../examples/epd_disaggregated/epd_load_balance_proxy_layerwise_server_example.py) | 加载 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`ServerState`, `ProxyState`, `parse_args`, `lifespan`, `listen_for_disconnect` |

## `examples/eplb/`

专家负载均衡 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [eplb_deepseek.py](../../../examples/eplb/eplb_deepseek.py) | DeepSeek 模型、专家负载均衡 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Expert parallelism load balancer (EPLB) for vLLM.；定义：`balanced_packing`, `replicate_experts`, `rebalance_experts_hierarchical` |
| [eplb_strategy.py](../../../examples/eplb/eplb_strategy.py) | 专家负载均衡 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`save_matrix_to_json`, `calculate_average`, `layer_imbalance_plot` |

## `examples/external_online_dp/`

external_online_dp 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../examples/external_online_dp/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：Modify parameters in `run_dp_template.sh` |
| [launch_online_dp.py](../../../examples/external_online_dp/launch_online_dp.py) | launch online dp 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`parse_args`, `run_command` |
| [run_dp_template.sh](../../../examples/external_online_dp/run_dp_template.sh) | run dp template 使用示例或配套配置；展示调用方式而非引擎入口。 | 命令类别：`vllm` |

## `examples/offline_DSD_k_tuner/`

offline_DSD_k_tuner 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [generate_config.py](../../../examples/offline_DSD_k_tuner/generate_config.py) | 配置 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`build_compilation_config`, `build_serve_params`, `build_bench_params` |
| [manager.py](../../../examples/offline_DSD_k_tuner/manager.py) | 管理器 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`DynamicSpeculativeDecodingManager` |

## `examples/quantization/`

生成量化 checkpoint 的参考配置与脚本。

| 子目录 | 职责 |
| --- | --- |
| [deepseek_v4/](../../../examples/quantization/deepseek_v4) | DeepSeek 模型 子模块；所属范围：生成量化 checkpoint 的参考配置与脚本。 |
| [llm-compressor/](../../../examples/quantization/llm-compressor) | 长程压缩 子模块；所属范围：生成量化 checkpoint 的参考配置与脚本。 |

## `examples/quantization/deepseek_v4/`

DeepSeek 模型 子模块；所属范围：生成量化 checkpoint 的参考配置与脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../examples/quantization/deepseek_v4/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：DeepSeek-V4 Quantization |
| [quantize.py](../../../examples/quantization/deepseek_v4/quantize.py) | 量化 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：DeepSeek-V4 W8A8 quantization script.；定义：`BufferedSafetensorsWriter`, `decode_fp8`, `decode_fp4`, `weight_quant_sym_perchannel` |
| [quantize_config.yaml](../../../examples/quantization/deepseek_v4/quantize_config.yaml) | 量化、配置 使用示例或配套配置；展示调用方式而非引擎入口。 | 顶层项：`apiversion`, `metadata`, `default_w8a8_dynamic`, `spec` |
| [quantize_config_dspark.yaml](../../../examples/quantization/deepseek_v4/quantize_config_dspark.yaml) | 量化、DSpark 草稿、配置 使用示例或配套配置；展示调用方式而非引擎入口。 | 顶层项：`apiversion`, `metadata`, `default_w8a8_dynamic`, `spec` |
| [quantize_config_pro_w4a8.yaml](../../../examples/quantization/deepseek_v4/quantize_config_pro_w4a8.yaml) | 量化、配置 使用示例或配套配置；展示调用方式而非引擎入口。 | 顶层项：`apiversion`, `metadata`, `default_w4a8_dynamic`, `default_w8a8_dynamic`, `spec` |

## `examples/quantization/llm-compressor/`

长程压缩 子模块；所属范围：生成量化 checkpoint 的参考配置与脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [mixed_mxfp4_mxfp8_dynamic_moe.py](../../../examples/quantization/llm-compressor/mixed_mxfp4_mxfp8_dynamic_moe.py) | MXFP 数值格式、混合专家 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`main` |
| [w4a8_dynamic_moe.py](../../../examples/quantization/llm-compressor/w4a8_dynamic_moe.py) | 混合专家 使用示例或配套配置；展示调用方式而非引擎入口。 | 导入：`llmcompressor`, `transformers` |
| [w8a8_int8.py](../../../examples/quantization/llm-compressor/w8a8_int8.py) | w8a8 int8 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`load_environment_variables`, `load_calibration_text_dataset`, `data_collator` |
| [w8a8_int8_dynamic.py](../../../examples/quantization/llm-compressor/w8a8_int8_dynamic.py) | w8a8 int8 dynamic 使用示例或配套配置；展示调用方式而非引擎入口。 | 定义：`preprocess`, `tokenize` |
| [w8a8_int8_dynamic_moe.py](../../../examples/quantization/llm-compressor/w8a8_int8_dynamic_moe.py) | 混合专家 使用示例或配套配置；展示调用方式而非引擎入口。 | 导入：`torch`, `llmcompressor` |

## `examples/rfork/`

RFork 权重加载 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [rfork_planner.py](../../../examples/rfork/rfork_planner.py) | RFork 权重加载 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Standalone rfork planner mock server used by vLLM rfork seed protocol tests.；定义：`Settings`, `SeedRecord`, `parse_add_seed_headers`, `parse_get_seed_headers`, `parse_put_seed_headers` |

## `examples/rl/`

rl 子模块；所属范围：离线推理、分布式部署、权重更新和量化等参考使用示例。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [rlhf_async_new_apis.py](../../../examples/rl/rlhf_async_new_apis.py) | rlhf async new apis 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Demonstrates async reinforcement learning using vLLM and Ray,；定义：`MyLLM`, `TrainModel` |
| [rlhf_http_hccl.py](../../../examples/rl/rlhf_http_hccl.py) | HCCL 通信 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Demonstrates reinforcement learning from human feedback (RLHF) using vLLM；定义：`generate_completions`, `init_weight_transfer_engine`, `update_weights` |
| [rlhf_http_lora.py](../../../examples/rl/rlhf_http_lora.py) | LoRA 适配 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Demonstrates reinforcement learning from human feedback (RLHF) with LoRA；定义：`LoRAServer`, `resolve_lora_path`, `generate_completions`, `load_lora_adapter` |
| [rlhf_http_npu_ipc.py](../../../examples/rl/rlhf_http_npu_ipc.py) | 进程间通信 使用示例或配套配置；展示调用方式而非引擎入口。 | 注释：Demonstrates reinforcement learning from human feedback (RLHF) using vLLM；定义：`generate_completions`, `init_weight_transfer_engine`, `start_weight_update` |
