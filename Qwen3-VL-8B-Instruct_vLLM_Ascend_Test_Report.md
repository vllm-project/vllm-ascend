# Qwen3-VL-8B-Instruct 模型 vLLM-Ascend 部署与性能测试报告

## 一、测试概述

| 项目   | 内容                                             |
| ---- | ---------------------------------------------- |
| 测试日期 | 2026-07-22                                     |
| 测试环境 | Linux 环境，Ascend NPU，vLLM-Ascend v0.18.0      |
| 测试目标 | 使用 vLLM-Ascend 部署 Qwen3-VL-8B-Instruct 模型并测试性能 |

***

## 二、环境准备

### 2.1 模型文件验证

**操作命令：**

```bash
ls -la /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct
```

**执行结果：**

```
root@1aa7ad7ed237:/workspace# ls -la /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct
total 17134697
drwxr-xr-x.  2 root root       4096 Jul  3 09:25 .
drwxr-xr-x. 23 root root       4096 Jul 20 03:05 ..
-rw-r--r--.  1 root root         48 Jul  3 09:16 .mdl
-rw-------.  1 root root       1237 Jul  3 09:16 .msc
-rw-r--r--.  1 root root         36 Jul  3 09:16 .mv
-rw-r--r--.  1 root root       7133 Jul  3 09:25 README.md
-rw-r--r--.  1 root root       5499 Jul  3 09:16 chat_template.json
-rw-r--r--.  1 root root       1474 Jul  3 09:16 config.json
-rw-r--r--.  1 root root         73 Jul  3 09:16 configuration.json
-rw-r--r--.  1 root root        269 Jul  3 09:16 generation_config.json
-rw-r--r--.  1 root root    1671839 Jul  3 09:16 merges.txt
-rw-r--r--.  1 root root 4902275944 Jul  3 09:18 model-00001-of-00004.safetensors
-rw-r--r--.  1 root root 4915962496 Jul  3 09:21 model-00002-of-00004.safetensors
-rw-r--r--.  1 root root 4999831048 Jul  3 09:24 model-00003-of-00004.safetensors
-rw-r--r--.  1 root root 2716270024 Jul  3 09:25 model-00004-of-00004.safetensors
-rw-r--r--.  1 root root      67759 Jul  3 09:25 model.safetensors.index.json
-rw-r--r--.  1 root root        390 Jul  3 09:25 preprocessor_config.json
-rw-r--r--.  1 root root    7032403 Jul  3 09:25 tokenizer.json
-rw-r--r--.  1 root root      10868 Jul  3 09:25 tokenizer_config.json
-rw-r--r--.  1 root root        385 Jul  3 09:25 video_preprocessor_config.json
-rw-r--r--.  1 root root    2776833 Jul  3 09:25 vocab.json
```

### 2.2 vLLM-Ascend 环境检查

**操作命令：**

```bash
vllm --version
```

**执行结果：**

```
root@1aa7ad7ed237:/workspace# vllm --version
INFO 07-22 04:31:28 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 07-22 04:31:28 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 07-22 04:31:28 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 07-22 04:31:28 [__init__.py:239] Platform plugin ascend is activated
INFO 07-22 04:31:36 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
INFO 07-22 04:31:36 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
0.18.0+empty
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute


```

***

## 三、模型部署

### 3.1 启动 vLLM-Ascend 服务

**操作命令：**

```bash
vllm serve /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384
```

**执行结果：**

```
root@1aa7ad7ed237:/workspace# vllm serve /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct \
--dtype bfloat16 \
--max-model-len 16384 \
--max-num-batched-tokens 16384
INFO 07-22 04:30:26 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 07-22 04:30:26 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 07-22 04:30:26 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 07-22 04:30:26 [__init__.py:239] Platform plugin ascend is activated
INFO 07-22 04:30:34 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
INFO 07-22 04:30:34 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
(APIServer pid=8751) INFO 07-22 04:30:34 [utils.py:297] 
(APIServer pid=8751) INFO 07-22 04:30:34 [utils.py:297]        █     █     █▄   ▄█
(APIServer pid=8751) INFO 07-22 04:30:34 [utils.py:297]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.18.0
(APIServer pid=8751) INFO 07-22 04:30:34 [utils.py:297]   █▄█▀ █     █     █     █  model   /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct
(APIServer pid=8751) INFO 07-22 04:30:34 [utils.py:297]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=8751) INFO 07-22 04:30:34 [utils.py:297] 
(APIServer pid=8751) INFO 07-22 04:30:34 [utils.py:233] non-default args: {'model_tag': '/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct', 'model': '/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct', 'dtype': 'bfloat16', 'max_model_len': 16384, 'max_num_batched_tokens': 16384}
(APIServer pid=8751) INFO 07-22 04:30:35 [model.py:533] Resolved architecture: Qwen3VLForConditionalGeneration
(APIServer pid=8751) INFO 07-22 04:30:35 [model.py:1582] Using max model len 16384
(APIServer pid=8751) INFO 07-22 04:30:36 [scheduler.py:231] Chunked prefill is enabled with max_num_batched_tokens=16384.
(APIServer pid=8751) INFO 07-22 04:30:36 [vllm.py:754] Asynchronous scheduling is enabled.
(APIServer pid=8751) WARNING 07-22 04:30:36 [platform.py:749] Parameter '--disable-cascade-attn' is a GPU-specific feature. Resetting to False for Ascend.
(APIServer pid=8751) WARNING 07-22 04:30:36 [platform.py:838] Ignored parameter 'disable_flashinfer_prefill'. This is a GPU-specific feature not supported on Ascend. Resetting to False.
(APIServer pid=8751) INFO 07-22 04:30:36 [ascend_config.py:425] Dynamic EPLB is False
(APIServer pid=8751) INFO 07-22 04:30:36 [ascend_config.py:426] The number of redundant experts is 0
(APIServer pid=8751) INFO 07-22 04:30:36 [platform.py:354] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
(APIServer pid=8751) INFO 07-22 04:30:36 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 48
(APIServer pid=8751) WARNING 07-22 04:30:36 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
(APIServer pid=8751) INFO 07-22 04:30:36 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3VLForConditionalGeneration model (layers: 36) with 35 sizes
(APIServer pid=8751) INFO 07-22 04:30:36 [utils.py:1114] Block size is set to 128 if prefix cache or chunked prefill is enabled.
(APIServer pid=8751) INFO 07-22 04:30:36 [platform.py:502] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
(APIServer pid=8751) INFO 07-22 04:30:36 [compilation.py:289] Enabled custom fusions: norm_quant, act_quant
INFO 07-22 04:30:50 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 07-22 04:30:50 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 07-22 04:30:50 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 07-22 04:30:50 [__init__.py:239] Platform plugin ascend is activated
(EngineCore pid=8878) INFO 07-22 04:30:57 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
(EngineCore pid=8878) INFO 07-22 04:30:57 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
(EngineCore pid=8878) INFO 07-22 04:30:57 [core.py:103] Initializing a V1 LLM engine (v0.18.0) with config: model='/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct', speculative_config=None, tokenizer='/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=16384, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=npu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'vllm_ascend.compilation.compiler_interface.AscendCompiler', 'custom_ops': ['all'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update', 'vllm::mla_forward'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_endpoints': [16384], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 256, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=8878) INFO 07-22 04:31:00 [ascend_config.py:425] Dynamic EPLB is False
(EngineCore pid=8878) INFO 07-22 04:31:00 [ascend_config.py:426] The number of redundant experts is 0
(EngineCore pid=8878) INFO 07-22 04:31:03 [parallel_state.py:1395] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.17.0.5:50063 backend=hccl
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore pid=8878) INFO 07-22 04:31:03 [parallel_state.py:1717] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore pid=8878) INFO 07-22 04:31:04 [cpu_binding.py:320] [cpu_bind_mode] mode=topo_affinity rank=0 visible_npus=[0]
(EngineCore pid=8878) INFO 07-22 04:31:04 [cpu_binding.py:367] The CPU allocation plan is as follows:
(EngineCore pid=8878) INFO 07-22 04:31:04 [cpu_binding.py:372] NPU0: main=[2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61]  acl=[62]  release=[[63]]
(EngineCore pid=8878) INFO 07-22 04:31:04 [cpu_binding.py:394] [migrate] NPU:0 -> NUMA [0]
(EngineCore pid=8878) INFO 07-22 04:31:07 [model_runner_v1.py:2562] Starting to load model /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct...
(EngineCore pid=8878) INFO 07-22 04:31:08 [interface.py:275] Using default backend AttentionBackendEnum.TORCH_SDPA for vit attention
(EngineCore pid=8878) INFO 07-22 04:31:08 [mm_encoder_attention.py:230] Using AttentionBackendEnum.TORCH_SDPA for MMEncoderAttention.
(EngineCore pid=8878) INFO 07-22 04:31:08 [vllm.py:754] Asynchronous scheduling is enabled.
(EngineCore pid=8878) INFO 07-22 04:31:08 [platform.py:354] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
(EngineCore pid=8878) INFO 07-22 04:31:08 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 48
(EngineCore pid=8878) WARNING 07-22 04:31:08 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
(EngineCore pid=8878) INFO 07-22 04:31:08 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3VLForConditionalGeneration model (layers: 36) with 35 sizes
(EngineCore pid=8878) INFO 07-22 04:31:08 [platform.py:502] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
(EngineCore pid=8878) INFO 07-22 04:31:08 [compilation.py:289] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=8878) INFO 07-22 04:31:09 [compilation.py:942] Using OOT custom backend for compilation.
(EngineCore pid=8878) INFO 07-22 04:31:09 [compilation.py:942] Using OOT custom backend for compilation.
(EngineCore pid=8878) INFO 07-22 04:31:09 [compilation.py:942] Using OOT custom backend for compilation.
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.89it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.50it/s]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.30it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.41it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.43it/s]
(EngineCore pid=8878) 
(EngineCore pid=8878) INFO 07-22 04:31:12 [default_loader.py:384] Loading weights took 2.91 seconds
(EngineCore pid=8878) INFO 07-22 04:31:13 [model_runner_v1.py:2589] Loading model weights took 16.9570 GB
(EngineCore pid=8878) INFO 07-22 04:31:14 [gpu_model_runner.py:5488] Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum feature size.
(EngineCore pid=8878) INFO 07-22 04:31:26 [backends.py:988] Using cache directory: /root/.cache/vllm/torch_compile_cache/0d42d4420e/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=8878) INFO 07-22 04:31:26 [backends.py:1048] Dynamo bytecode transform time: 9.48 s
(EngineCore pid=8878) INFO 07-22 04:31:48 [backends.py:387] Compiling a graph for compile range (1, 16384) takes 20.50 s
(EngineCore pid=8878) INFO 07-22 04:31:51 [monitor.py:48] torch.compile and initial profiling/warmup run together took 34.69 s in total
(EngineCore pid=8878) INFO 07-22 04:31:53 [worker.py:357] Available KV cache memory: 34.65 GiB
(EngineCore pid=8878) INFO 07-22 04:31:53 [kv_cache_utils.py:1316] GPU KV cache size: 252,288 tokens
(EngineCore pid=8878) INFO 07-22 04:31:53 [kv_cache_utils.py:1321] Maximum concurrency for 16,384 tokens per request: 15.40x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████████| 35/35 [00:06<00:00,  5.21it/s]
(EngineCore pid=8878) INFO 07-22 04:32:03 [gpu_model_runner.py:5746] Graph capturing finished in 8 secs, took 0.16 GiB
(EngineCore pid=8878) INFO 07-22 04:32:03 [core.py:281] init engine (profile, create kv cache, warmup model) took 49.31 seconds
(EngineCore pid=8878) INFO 07-22 04:32:04 [platform.py:354] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
(EngineCore pid=8878) INFO 07-22 04:32:04 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 48
(EngineCore pid=8878) WARNING 07-22 04:32:04 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
(EngineCore pid=8878) INFO 07-22 04:32:04 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3VLForConditionalGeneration model (layers: 36) with 35 sizes
(EngineCore pid=8878) INFO 07-22 04:32:04 [platform.py:502] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
(APIServer pid=8751) INFO 07-22 04:32:04 [api_server.py:576] Supported tasks: ['generate']
(APIServer pid=8751) WARNING 07-22 04:32:04 [model.py:1376] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=8751) INFO 07-22 04:32:05 [hf.py:320] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
(APIServer pid=8751) INFO 07-22 04:32:09 [base.py:216] Multi-modal warmup completed in 4.503s
(APIServer pid=8751) INFO 07-22 04:32:09 [api_server.py:580] Starting vLLM server on http://0.0.0.0:8000
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:37] Available routes are:
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /docs, Methods: GET, HEAD
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /redoc, Methods: GET, HEAD
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=8751) INFO 07-22 04:32:09 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=8751) INFO:     Started server process [8751]
(APIServer pid=8751) INFO:     Waiting for application startup.
(APIServer pid=8751) INFO:     Application startup complete.
```

### 3.2 服务健康检查

**操作命令：**

```bash
curl http://localhost:8000/v1/models
```

**执行结果：**

```
root@1aa7ad7ed237:/workspace# curl http://localhost:8000/v1/models
{"object":"list","data":[{"id":"/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct","object":"model","created":1784694833,"owned_by":"vllm","root":"/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct","parent":null,"max_model_len":16384,"permission":[{"id":"modelperm-a6ac34edfa7b2144","object":"model_permission","created":1784694833,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

***

## 四、性能测试

### 4.1 运行 vLLM Benchmark

**操作命令：**

```bash
vllm bench serve \
  --model /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./
```

**执行结果：**

```
root@1aa7ad7ed237:/workspace# vllm bench serve --model /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./


INFO 07-22 04:15:09 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 07-22 04:15:09 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 07-22 04:15:09 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 07-22 04:15:09 [__init__.py:239] Platform plugin ascend is activated
INFO 07-22 04:15:16 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
INFO 07-22 04:15:16 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0xfffef58fec00>, trust_remote_code=False, seed=0, num_prompts=200, dataset_name='random', no_stream=False, dataset_path=None, no_oversample=False, skip_chat_template=False, enable_multimodal_chat=False, disable_shuffle=False, custom_output_len=256, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, blazedit_min_distance=0.0, blazedit_max_distance=1.0, asr_max_audio_len_sec=inf, asr_min_audio_len_sec=0.0, random_input_len=200, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, no_reranker=False, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 1}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, label=None, backend='openai', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', header=None, max_concurrency=None, model='/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct', input_len=None, output_len=None, tokenizer=None, tokenizer_mode='auto', use_beam_search=False, logprobs=None, request_rate=1.0, burstiness=1.0, disable_tqdm=False, num_warmups=0, profile=False, save_result=True, save_detailed=False, append_result=False, metadata=None, result_dir='./', result_filename=None, ignore_eos=False, percentile_metrics=None, metric_percentiles='99', goodput=None, request_id_prefix='bench-ddfa129c-', top_p=None, top_k=None, min_p=None, temperature=None, frequency_penalty=None, presence_penalty=None, repetition_penalty=None, served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=0, extra_body=None, skip_tokenizer_init=False, insecure=False, plot_timeline=False, timeline_itl_thresholds=[25.0, 50.0], plot_dataset_stats=False)
INFO 07-22 04:15:19 [datasets.py:631] Sampling input_len from [200, 200] and output_len from [128, 128]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Traffic request rate: 1.0
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|█████████████████████████████████████████████████████████████| 200/200 [03:23<00:00,  1.02s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     200       
Failed requests:                         0         
Request rate configured (RPS):           1.00      
Benchmark duration (s):                  203.51    
Total input tokens:                      40000     
Total generated tokens:                  25600     
Request throughput (req/s):              0.98      
Output token throughput (tok/s):         125.79    
Peak output token throughput (tok/s):    313.00    
Peak concurrent requests:                12.00     
Total token throughput (tok/s):          322.34    
---------------Time to First Token----------------
Mean TTFT (ms):                          95.63     
Median TTFT (ms):                        95.70     
P99 TTFT (ms):                           143.33    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.91     
Median TPOT (ms):                        27.68     
P99 TPOT (ms):                           31.20     
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.91     
Median ITL (ms):                         27.34     
P99 ITL (ms):                            42.08     
==================================================
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

### 4.2 Benchmark 结果文件分析

**操作命令：**

```bash
cat /workspace/openai-1.0qps-Qwen3-VL-8B-Instruct-20260722-041843.json
```

**执行结果：**

```
{"date": "20260722-041843", "endpoint_type": "openai", "backend": "openai", "label": null, "model_id": "/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct", "tokenizer_id": "/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct", "num_prompts": 200, "request_rate": 1.0, "burstiness": 1.0, "max_concurrency": null, "duration": 203.5104987300001, "completed": 200, "failed": 0, "total_input_tokens": 40000, "total_output_tokens": 25600, "request_throughput": 0.9827502819171137, "request_goodput": null, "output_throughput": 125.79203608539055, "total_token_throughput": 322.3420924688133, "max_output_tokens_per_s": 313.0, "max_concurrent_requests": 12, "rtfx": 0.0, "mean_ttft_ms": 95.6344293046277, "median_ttft_ms": 95.6995899323374, "std_ttft_ms": 12.71286480609931, "p99_ttft_ms": 143.32562107359988, "mean_tpot_ms": 27.911832109121164, "median_tpot_ms": 27.67841275561395, "std_tpot_ms": 1.0193481469453176, "p99_tpot_ms": 31.202747922493362, "mean_itl_ms": 27.911833959895617, "median_itl_ms": 27.339585009030998, "std_itl_ms": 3.1292647193476784, "p99_itl_ms": 42.0829780306667}
```

***

## 五、性能指标汇总

| 指标                               | 数值    | 单位     |
| -------------------------------- | ------- | -------- |
| 请求吞吐量 (Request Throughput)      | 0.98    | req/s    |
| 输出 token 吞吐量 (Output Token/s)   | 125.79  | tok/s    |
| 总 token 吞吐量 (Total Token/s)      | 322.34  | tok/s    |
| 峰值输出 token 吞吐量 (Peak Output/s) | 313.00  | tok/s    |
| 平均首 token 延迟 (Mean TTFT)        | 95.63   | ms       |
| P99 首 token 延迟 (P99 TTFT)         | 143.33  | ms       |
| 平均单 token 延迟 (Mean TPOT)         | 27.91   | ms       |
| P99 单 token 延迟 (P99 TPOT)          | 31.20   | ms       |
| 平均 token 间延迟 (Mean ITL)         | 27.91   | ms       |
| P99 token 间延迟 (P99 ITL)          | 42.08   | ms       |
| 模型加载时间                         | 49.31   | s        |
| 峰值并发请求数                       | 12      | req      |
| 请求成功率                           | 100%    | -        |

***

## 六、问题与解决

### 6.1 遇到的问题

1. vLLM-Ascend 自动忽略了部分 GPU 专用参数（如 `--disable-cascade-attn`、`--disable-flashinfer-prefill`），系统自动重置为 Ascend 兼容值。

### 6.2 解决方案

1. 系统自动处理：vLLM-Ascend 框架在启动时自动检测平台类型（Ascend NPU），并将 GPU 专用参数重置为 NPU 兼容值，无需人工干预。

***

## 七、总结与建议

### 7.1 测试总结

本次测试成功使用 vLLM-Ascend 在 Ascend NPU 平台上部署了 Qwen3-VL-8B-Instruct 模型，并完成了性能基准测试。模型加载时间为 49.31 秒，服务启动正常，所有 200 个测试请求均成功完成（成功率 100%）。性能指标方面，请求吞吐量达到 0.98 req/s，平均首 token 延迟为 95.63 ms，平均 token 间延迟为 27.91 ms，整体性能表现符合预期。

### 7.2 改进建议

1. **通信优化**：日志提示当前使用 FFTS+ 方法进行通信，建议设置环境变量 `HCCL_OP_EXPANSION_MODE=AIV` 以提升通信性能并增加支持的运行时形状范围。
2. **并发测试**：本次测试使用请求速率为 1 req/s，建议进一步测试更高并发场景下的性能表现。
3. **模型量化**：当前使用 bfloat16 精度加载模型，可尝试使用 AWQ 等量化方式减少显存占用并提升吞吐量。
