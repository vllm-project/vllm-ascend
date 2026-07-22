
### 模型启动

root@28e4f005090f:/workspace# cat 1.sh 
vllm serve /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct \
--dtype bfloat16 \
--max-model-len 16384 \
--max-num-batched-tokens 16384


INFO 07-22 04:05:08 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 07-22 04:05:08 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 07-22 04:05:08 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 07-22 04:05:08 [__init__.py:239] Platform plugin ascend is activated
INFO 07-22 04:05:15 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
INFO 07-22 04:05:15 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
(APIServer pid=2719) INFO 07-22 04:05:15 [utils.py:297] 
(APIServer pid=2719) INFO 07-22 04:05:15 [utils.py:297]        █     █     █▄   ▄█
(APIServer pid=2719) INFO 07-22 04:05:15 [utils.py:297]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.18.0
(APIServer pid=2719) INFO 07-22 04:05:15 [utils.py:297]   █▄█▀ █     █     █     █  model   /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct
(APIServer pid=2719) INFO 07-22 04:05:15 [utils.py:297]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=2719) INFO 07-22 04:05:15 [utils.py:297] 
(APIServer pid=2719) INFO 07-22 04:05:15 [utils.py:233] non-default args: {'model_tag': '/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct', 'model': '/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct', 'dtype': 'bfloat16', 'max_model_len': 16384, 'max_num_batched_tokens': 16384}
(APIServer pid=2719) INFO 07-22 04:05:30 [model.py:533] Resolved architecture: Qwen3VLForConditionalGeneration
(APIServer pid=2719) INFO 07-22 04:05:30 [model.py:1582] Using max model len 16384
(APIServer pid=2719) INFO 07-22 04:05:30 [scheduler.py:231] Chunked prefill is enabled with max_num_batched_tokens=16384.
(APIServer pid=2719) INFO 07-22 04:05:30 [vllm.py:754] Asynchronous scheduling is enabled.
(APIServer pid=2719) WARNING 07-22 04:05:30 [platform.py:749] Parameter '--disable-cascade-attn' is a GPU-specific feature. Resetting to False for Ascend.
(APIServer pid=2719) WARNING 07-22 04:05:30 [platform.py:838] Ignored parameter 'disable_flashinfer_prefill'. This is a GPU-specific feature not supported on Ascend. Resetting to False.
(APIServer pid=2719) INFO 07-22 04:05:30 [ascend_config.py:425] Dynamic EPLB is False
(APIServer pid=2719) INFO 07-22 04:05:30 [ascend_config.py:426] The number of redundant experts is 0
(APIServer pid=2719) INFO 07-22 04:05:30 [platform.py:354] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
(APIServer pid=2719) INFO 07-22 04:05:30 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 48
(APIServer pid=2719) WARNING 07-22 04:05:30 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
(APIServer pid=2719) INFO 07-22 04:05:30 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3VLForConditionalGeneration model (layers: 36) with 35 sizes
(APIServer pid=2719) INFO 07-22 04:05:30 [utils.py:1114] Block size is set to 128 if prefix cache or chunked prefill is enabled.
(APIServer pid=2719) INFO 07-22 04:05:30 [platform.py:502] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
(APIServer pid=2719) INFO 07-22 04:05:30 [compilation.py:289] Enabled custom fusions: norm_quant, act_quant
INFO 07-22 04:05:44 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 07-22 04:05:44 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 07-22 04:05:44 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 07-22 04:05:44 [__init__.py:239] Platform plugin ascend is activated
(EngineCore pid=3012) INFO 07-22 04:05:51 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
(EngineCore pid=3012) INFO 07-22 04:05:51 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
(EngineCore pid=3012) INFO 07-22 04:05:51 [core.py:103] Initializing a V1 LLM engine (v0.18.0) with config: model='/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct', speculative_config=None, tokenizer='/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=16384, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=npu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=/workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'vllm_ascend.compilation.compiler_interface.AscendCompiler', 'custom_ops': ['all'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update', 'vllm::mla_forward'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_endpoints': [16384], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 256, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=3012) INFO 07-22 04:05:56 [ascend_config.py:425] Dynamic EPLB is False
(EngineCore pid=3012) INFO 07-22 04:05:56 [ascend_config.py:426] The number of redundant experts is 0
(EngineCore pid=3012) INFO 07-22 04:06:05 [parallel_state.py:1395] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.17.0.8:60243 backend=hccl
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore pid=3012) INFO 07-22 04:06:05 [parallel_state.py:1717] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore pid=3012) INFO 07-22 04:06:06 [cpu_binding.py:320] [cpu_bind_mode] mode=topo_affinity rank=0 visible_npus=[0]
(EngineCore pid=3012) INFO 07-22 04:06:06 [cpu_binding.py:367] The CPU allocation plan is as follows:
(EngineCore pid=3012) INFO 07-22 04:06:06 [cpu_binding.py:372] NPU0: main=[130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189]  acl=[190]  release=[[191]]
(EngineCore pid=3012) INFO 07-22 04:06:06 [cpu_binding.py:394] [migrate] NPU:0 -> NUMA [4]
(EngineCore pid=3012) INFO 07-22 04:06:12 [model_runner_v1.py:2562] Starting to load model /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct...
(EngineCore pid=3012) INFO 07-22 04:06:12 [interface.py:275] Using default backend AttentionBackendEnum.TORCH_SDPA for vit attention
(EngineCore pid=3012) INFO 07-22 04:06:12 [mm_encoder_attention.py:230] Using AttentionBackendEnum.TORCH_SDPA for MMEncoderAttention.
(EngineCore pid=3012) INFO 07-22 04:06:12 [vllm.py:754] Asynchronous scheduling is enabled.
(EngineCore pid=3012) INFO 07-22 04:06:13 [platform.py:354] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
(EngineCore pid=3012) INFO 07-22 04:06:13 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 48
(EngineCore pid=3012) WARNING 07-22 04:06:13 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
(EngineCore pid=3012) INFO 07-22 04:06:13 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3VLForConditionalGeneration model (layers: 36) with 35 sizes
(EngineCore pid=3012) INFO 07-22 04:06:13 [platform.py:502] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
(EngineCore pid=3012) INFO 07-22 04:06:13 [compilation.py:289] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=3012) INFO 07-22 04:06:13 [compilation.py:942] Using OOT custom backend for compilation.
(EngineCore pid=3012) INFO 07-22 04:06:14 [compilation.py:942] Using OOT custom backend for compilation.
(EngineCore pid=3012) INFO 07-22 04:06:14 [compilation.py:942] Using OOT custom backend for compilation.
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  2.03it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.51it/s]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.32it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.52it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.51it/s]
(EngineCore pid=3012) 
(EngineCore pid=3012) INFO 07-22 04:06:17 [default_loader.py:384] Loading weights took 2.72 seconds
(EngineCore pid=3012) INFO 07-22 04:06:18 [model_runner_v1.py:2589] Loading model weights took 16.9570 GB
(EngineCore pid=3012) INFO 07-22 04:06:18 [gpu_model_runner.py:5488] Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum feature size.
(EngineCore pid=3012) INFO 07-22 04:06:30 [backends.py:988] Using cache directory: /root/.cache/vllm/torch_compile_cache/0d42d4420e/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=3012) INFO 07-22 04:06:30 [backends.py:1048] Dynamo bytecode transform time: 9.29 s
(EngineCore pid=3012) INFO 07-22 04:07:08 [backends.py:387] Compiling a graph for compile range (1, 16384) takes 20.17 s
(EngineCore pid=3012) INFO 07-22 04:07:16 [monitor.py:48] torch.compile and initial profiling/warmup run together took 55.27 s in total
(EngineCore pid=3012) INFO 07-22 04:07:18 [worker.py:357] Available KV cache memory: 34.66 GiB
(EngineCore pid=3012) INFO 07-22 04:07:18 [kv_cache_utils.py:1316] GPU KV cache size: 252,288 tokens
(EngineCore pid=3012) INFO 07-22 04:07:18 [kv_cache_utils.py:1321] Maximum concurrency for 16,384 tokens per request: 15.40x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████████████████████████| 35/35 [00:13<00:00,  2.67it/s]
(EngineCore pid=3012) INFO 07-22 04:07:34 [gpu_model_runner.py:5746] Graph capturing finished in 14 secs, took 0.16 GiB
(EngineCore pid=3012) INFO 07-22 04:07:34 [core.py:281] init engine (profile, create kv cache, warmup model) took 75.89 seconds
(EngineCore pid=3012) INFO 07-22 04:07:35 [platform.py:354] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
(EngineCore pid=3012) INFO 07-22 04:07:35 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 48
(EngineCore pid=3012) WARNING 07-22 04:07:35 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
(EngineCore pid=3012) INFO 07-22 04:07:35 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3VLForConditionalGeneration model (layers: 36) with 35 sizes
(EngineCore pid=3012) INFO 07-22 04:07:35 [platform.py:502] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
(APIServer pid=2719) INFO 07-22 04:07:35 [api_server.py:576] Supported tasks: ['generate']
(APIServer pid=2719) WARNING 07-22 04:07:35 [model.py:1376] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=2719) INFO 07-22 04:07:37 [hf.py:320] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
(APIServer pid=2719) INFO 07-22 04:07:42 [base.py:216] Multi-modal warmup completed in 4.902s
(APIServer pid=2719) INFO 07-22 04:07:42 [api_server.py:580] Starting vLLM server on http://0.0.0.0:8000
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:37] Available routes are:
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=2719) INFO 07-22 04:07:42 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=2719) INFO:     Started server process [2719]
(APIServer pid=2719) INFO:     Waiting for application startup.
(APIServer pid=2719) INFO:     Application startup complete.
(APIServer pid=2719) INFO:     127.0.0.1:50926 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=2719) INFO:     127.0.0.1:46596 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(EngineCore pid=3012) INFO 07-22 04:11:20 [acl_graph.py:192] Replaying aclgraph
(APIServer pid=2719) INFO:     127.0.0.1:58778 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:11:22 [loggers.py:259] Engine 000: Avg prompt throughput: 0.9 tokens/s, Avg generation throughput: 2.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO 07-22 04:11:32 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:38312 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38312 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:12:22 [loggers.py:259] Engine 000: Avg prompt throughput: 60.0 tokens/s, Avg generation throughput: 17.5 tokens/s, Running: 3 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.4%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:59994 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:60006 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38312 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59994 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38312 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:12:32 [loggers.py:259] Engine 000: Avg prompt throughput: 140.0 tokens/s, Avg generation throughput: 97.2 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59994 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38312 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:44746 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:12:42 [loggers.py:259] Engine 000: Avg prompt throughput: 200.0 tokens/s, Avg generation throughput: 123.7 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46260 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:12:52 [loggers.py:259] Engine 000: Avg prompt throughput: 140.0 tokens/s, Avg generation throughput: 80.3 tokens/s, Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.5%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46260 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46260 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:13:02 [loggers.py:259] Engine 000: Avg prompt throughput: 220.0 tokens/s, Avg generation throughput: 143.9 tokens/s, Running: 5 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.7%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46260 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:50570 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:50582 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:13:12 [loggers.py:259] Engine 000: Avg prompt throughput: 240.0 tokens/s, Avg generation throughput: 135.4 tokens/s, Running: 6 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46260 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:50570 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:50582 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46260 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53998 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:54012 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59982 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:13:22 [loggers.py:259] Engine 000: Avg prompt throughput: 300.0 tokens/s, Avg generation throughput: 167.7 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:60008 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:50570 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:50582 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46260 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:13:32 [loggers.py:259] Engine 000: Avg prompt throughput: 140.0 tokens/s, Avg generation throughput: 143.6 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53944 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53960 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53964 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53968 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53984 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53998 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53944 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53960 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:13:42 [loggers.py:259] Engine 000: Avg prompt throughput: 320.0 tokens/s, Avg generation throughput: 167.3 tokens/s, Running: 7 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.9%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:53964 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53968 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53984 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53998 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:59992 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53944 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53960 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53968 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53984 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:13:52 [loggers.py:259] Engine 000: Avg prompt throughput: 240.0 tokens/s, Avg generation throughput: 166.7 tokens/s, Running: 7 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.0%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53998 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46274 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53960 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53968 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53984 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53998 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:14:02 [loggers.py:259] Engine 000: Avg prompt throughput: 180.0 tokens/s, Avg generation throughput: 128.3 tokens/s, Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.5%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53960 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53968 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53984 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53960 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:14:12 [loggers.py:259] Engine 000: Avg prompt throughput: 140.0 tokens/s, Avg generation throughput: 101.5 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:53968 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53984 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53960 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53968 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53984 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:14:22 [loggers.py:259] Engine 000: Avg prompt throughput: 140.0 tokens/s, Avg generation throughput: 87.6 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53960 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53968 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53984 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38454 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38458 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38472 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53960 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53968 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53984 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:14:32 [loggers.py:259] Engine 000: Avg prompt throughput: 280.0 tokens/s, Avg generation throughput: 173.3 tokens/s, Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.6%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:38454 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38458 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38472 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38454 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:14:42 [loggers.py:259] Engine 000: Avg prompt throughput: 160.0 tokens/s, Avg generation throughput: 110.1 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:46248 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:53946 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38458 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:14:52 [loggers.py:259] Engine 000: Avg prompt throughput: 80.0 tokens/s, Avg generation throughput: 63.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:38458 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42814 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42822 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42832 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42848 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38458 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42814 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42822 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42832 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:15:02 [loggers.py:259] Engine 000: Avg prompt throughput: 220.0 tokens/s, Avg generation throughput: 102.9 tokens/s, Running: 5 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.6%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:42848 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38458 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42814 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42848 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:15:12 [loggers.py:259] Engine 000: Avg prompt throughput: 100.0 tokens/s, Avg generation throughput: 94.8 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42814 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40764 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42848 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40780 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40782 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40794 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40804 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40816 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42814 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40764 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42848 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40780 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40782 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40794 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:15:22 [loggers.py:259] Engine 000: Avg prompt throughput: 320.0 tokens/s, Avg generation throughput: 168.2 tokens/s, Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.4%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:40804 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40816 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42848 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40780 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40782 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40794 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40804 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40816 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42848 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:15:32 [loggers.py:259] Engine 000: Avg prompt throughput: 220.0 tokens/s, Avg generation throughput: 136.8 tokens/s, Running: 6 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.7%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:40780 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40782 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40794 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40804 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40816 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:38438 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:42848 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO:     127.0.0.1:40780 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:15:42 [loggers.py:259] Engine 000: Avg prompt throughput: 160.0 tokens/s, Avg generation throughput: 149.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO:     127.0.0.1:40782 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=2719) INFO 07-22 04:15:52 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=2719) INFO 07-22 04:16:02 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%



### 模型测试


vllm bench serve --model /workspace/shared_assets/models/Qwen/Qwen3-VL-8B-Instruct  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./

100%|█████████████████████████████████████████████████████████████████████████████████████| 200/200 [03:23<00:00,  1.02s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     200       
Failed requests:                         0         
Request rate configured (RPS):           1.00      
Benchmark duration (s):                  203.81    
Total input tokens:                      40000     
Total generated tokens:                  25600     
Request throughput (req/s):              0.98      
Output token throughput (tok/s):         125.61    
Peak output token throughput (tok/s):    321.00    
Peak concurrent requests:                12.00     
Total token throughput (tok/s):          321.86    
---------------Time to First Token----------------
Mean TTFT (ms):                          98.20     
Median TTFT (ms):                        96.56     
P99 TTFT (ms):                           147.82    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          29.29     
Median TPOT (ms):                        29.24     
P99 TPOT (ms):                           30.52     
---------------Inter-token Latency----------------
Mean ITL (ms):                           29.29     
Median ITL (ms):                         28.76     
P99 ITL (ms):                            43.24     
==================================================
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute



### 总结
基于昇腾硬件与 vLLM 框架完成 Qwen3-VL-8B-Instruct 模型部署与功能验证，实现高效推理服务。