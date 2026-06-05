# Quickstart

## Prerequisites

### Supported Devices

- Atlas A2 training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 inference series (Atlas 800I A2)
- Atlas A3 training series (Atlas 800T A3, Atlas 900 A3 SuperPoD, Atlas 9000 A3 SuperPoD)
- Atlas 800I A3 inference series (Atlas 800I A3)
- [Experimental] Atlas 300I inference series (Atlas 300I Duo)

## Setup environment using container

Before using containers, make sure Docker is installed on your system. If Docker is not installed, please refer to the [Docker installation guide](https://docs.docker.com/get-docker/) for installation instructions.

:::::{tab-set}
::::{tab-item} Ubuntu

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
# Atlas A2:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
# Install curl
apt-get update -y && apt-get install -y curl
```

::::

::::{tab-item} openEuler

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
# Atlas A2:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-openeuler
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3-openeuler
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-openeuler
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
# Install curl
yum update -y && yum install -y curl
```

::::
:::::

The default workdir is `/workspace`, vLLM and vLLM Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`) to help developers make changes effective immediately without requiring a new installation.

## Usage

You can use ModelScope mirror to speed up download:

<!-- tests/e2e/doctests/001-quickstart-test.sh should be considered updating as well -->

```bash
export VLLM_USE_MODELSCOPE=True
```

There are two ways to start vLLM on Ascend NPU:

:::::{tab-set}
::::{tab-item} Offline Batched Inference

With vLLM installed, you can start generating texts for list of input prompts (i.e. offline batch inference).

Create and run a simple inference test. The `example.py` can be like:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# The first run will take about 3-5 mins (10 MB/s) to download models
llm = LLM(model="Qwen/Qwen3-0.6B")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Then run:

```bash
python example.py
```

If you encounter a connection error with Hugging Face (e.g., `We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.`), run the following commands to use ModelScope as an alternative:

```bash
export VLLM_USE_MODELSCOPE=True
pip install modelscope
python example.py
```

The output is shown below, and it may change with version updates:

```bash
INFO 05-27 11:40:38 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 05-27 11:40:38 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 05-27 11:40:38 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 05-27 11:40:38 [__init__.py:238] Platform plugin ascend is activated
INFO 05-27 11:40:43 [nixl_utils.py:20] Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare memory leak in UCX when using NIXL.
INFO 05-27 11:40:43 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
INFO 05-27 11:40:43 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
INFO 05-27 11:40:43 [utils.py:233] non-default args: {'disable_log_stats': True, 'model': 'Qwen/Qwen3-0.6B'}
INFO 05-27 11:40:44 [model.py:555] Resolved architecture: Qwen3ForCausalLM
INFO 05-27 11:40:44 [model.py:1680] Using max model len 40960
INFO 05-27 11:40:44 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 05-27 11:40:44 [vllm.py:840] Asynchronous scheduling is enabled.
INFO 05-27 11:40:44 [kernel.py:205] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'])
INFO 05-27 11:40:44 [ascend_config.py:593] Dynamic EPLB is False
INFO 05-27 11:40:44 [ascend_config.py:594] The number of redundant experts is 0
INFO 05-27 11:40:44 [platform.py:396] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
INFO 05-27 11:40:44 [utils.py:607] Calculated maximum supported batch sizes for ACL graph: 62
INFO 05-27 11:40:44 [utils.py:640] No adjustment needed for ACL graph batch sizes: Qwen3ForCausalLM model (layers: 28) with 35 sizes
INFO 05-27 11:40:44 [utils.py:1251] Block size is set to 128 if prefix cache or chunked prefill is enabled.
INFO 05-27 11:40:44 [platform.py:569] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
INFO 05-27 11:40:44 [compilation.py:303] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=75652) INFO 05-27 11:40:44 [core.py:109] Initializing a V1 LLM engine (vx.x.x) with config: model='Qwen/Qwen3-0.6B', speculative_config=None, tokenizer='Qwen/Qwen3-0.6B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=40960, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=npu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=Qwen/Qwen3-0.6B, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'vllm_ascend.compilation.compiler_interface.AscendCompiler', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': ['vllm::unified_attention_with_output', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::gdn_attention_core_xpu', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::deepseek_v4_attention', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update', 'vllm::mla_forward'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 256, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native']), enable_flashinfer_autotune=True, moe_backend='auto')
(EngineCore pid=75652) INFO 05-27 11:40:48 [parallel_state.py:1402] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://90.90.97.28:38959 backend=hccl
(EngineCore pid=75652) INFO 05-27 11:40:48 [parallel_state.py:1715] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=75652) INFO 05-27 11:40:48 [model_runner_v1.py:3146] Starting to load model Qwen/Qwen3-0.6B...
(EngineCore pid=75652) INFO 05-27 11:40:49 [compilation.py:1049] Using OOT custom backend for compilation.
(EngineCore pid=75652) INFO 05-27 11:40:49 [compilation.py:1049] Using OOT custom backend for compilation.
(EngineCore pid=75652) INFO 05-27 11:40:49 [weight_utils.py:904] Filesystem type for checkpoints: EXT4. Checkpoint size: 1.40 GiB. Available RAM: 1944.66 GiB.
(EngineCore pid=75652) INFO 05-27 11:40:49 [weight_utils.py:927] Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
(EngineCore pid=75652) INFO 05-27 11:40:51 [default_loader.py:384] Loading weights took 1.99 seconds
(EngineCore pid=75652) INFO 05-27 11:40:51 [model_runner_v1.py:3187] Loading model weights took 1.1397 GB
(EngineCore pid=75652) INFO 05-27 11:40:52 [backends.py:1069] Using cache directory: /root/.cache/vllm/torch_compile_cache/5a9d9c976f/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=75652) INFO 05-27 11:40:52 [backends.py:1128] Dynamo bytecode transform time: 0.84 s
(EngineCore pid=75652) INFO 05-27 11:41:01 [backends.py:391] Compiling a graph for compile range (1, 8192) takes 7.77 s
(EngineCore pid=75652) INFO 05-27 11:41:01 [decorators.py:305] Directly load AOT compilation from path /root/.cache/vllm/torch_compile_cache/torch_aot_compile/74892a710dff08ef76700608ac46ccb60eebda2bdd3037d825f26b0db836de5a/rank_0_0/model
(EngineCore pid=75652) INFO 05-27 11:41:01 [monitor.py:53] torch.compile took 9.69 s in total
(EngineCore pid=75652) INFO 05-27 11:41:01 [monitor.py:81] Initial profiling/warmup run took 0.07 s
(EngineCore pid=75652) INFO 05-27 11:41:02 [worker.py:394] Available KV cache memory: 54.98 GiB
(EngineCore pid=75652) INFO 05-27 11:41:02 [kv_cache_utils.py:1708] GPU KV cache size: 514,688 tokens
(EngineCore pid=75652) INFO 05-27 11:41:02 [kv_cache_utils.py:1709] Maximum concurrency for 40,960 tokens per request: 12.57x
(EngineCore pid=75652) INFO 05-27 11:41:12 [gpu_model_runner.py:6133] Graph capturing finished in 7 secs, took 0.09 GiB
(EngineCore pid=75652) INFO 05-27 11:41:12 [worker.py:546] Free memory on device (60.89/61.27 GiB) on startup. Desired GPU memory utilization is (0.92, 56.37 GiB). Actual usage: 1.14 GiB for weights, 0.22 GiB for peak activation, 0.03 GiB for non-torch memory, 0.09 GiB for NPU graph memory. Replace gpu_memory_utilization with `--kv-cache-memory=58786898124` (54.75 GiB) to fit into requested memory, or `--kv-cache-memory=63643382784` (59.27 GiB) to fully utilize NPU free memory. Current KV cache memory: 54.98 GiB.
(EngineCore pid=75652) INFO 05-27 11:41:15 [cpu_binding.py:328] [cpu_bind_mode] mode=global_slice rank=0 visible_npus=[0]
(EngineCore pid=75652) INFO 05-27 11:41:15 [cpu_binding.py:388] The CPU allocation plan is as follows:
(EngineCore pid=75652) INFO 05-27 11:41:15 [cpu_binding.py:393] NPU0: main=[2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21]  acl=[22]  release=[[23]]
(EngineCore pid=75652) INFO 05-27 11:41:15 [cpu_binding.py:415] [migrate] NPU:0 -> NUMA [0]
(EngineCore pid=75652) INFO 05-27 11:41:20 [cpu_binding.py:510] NPU0(PCI 0000:9d:00.0): sq_send_trigger_irq IRQ_ID=1037 -> CPU0, cq_update_irq IRQ_ID=1038 -> CPU1
(EngineCore pid=75652) INFO 05-27 11:41:20 [core.py:299] init engine (profile, create kv cache, warmup model) took 29.13 s (compilation: 9.69 s)
(EngineCore pid=75652) INFO 05-27 11:41:21 [kernel.py:205] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'])
(EngineCore pid=75652) INFO 05-27 11:41:21 [platform.py:396] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
(EngineCore pid=75652) INFO 05-27 11:41:21 [utils.py:607] Calculated maximum supported batch sizes for ACL graph: 62
(EngineCore pid=75652) INFO 05-27 11:41:21 [utils.py:640] No adjustment needed for ACL graph batch sizes: Qwen3ForCausalLM model (layers: 28) with 35 sizes
(EngineCore pid=75652) INFO 05-27 11:41:21 [platform.py:569] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
(EngineCore pid=75652) INFO 05-27 11:41:21 [acl_graph.py:198] Replaying aclgraph
Prompt: 'Hello, my name is', Generated text: ' Lucy and I am an 8 year old who loves to draw and write stories'
Prompt: 'The president of the United States is', Generated text: " a key leader in the federal government, and the president's role in the executive"
Prompt: 'The capital of France is', Generated text: ' a city. What is the capital of France? The capital of France is Paris'
Prompt: 'The future of AI is', Generated text: ' a topic that is being discussed in various contexts. In the business world, AI'
(EngineCore pid=970) INFO 05-12 11:36:00 [core.py:1201] Shutdown initiated (timeout=0)
(EngineCore pid=970) INFO 05-12 11:36:00 [core.py:1224] Shutdown complete
ERROR 05-12 11:36:01 [core_client.py:704] Engine core proc EngineCore died unexpectedly, shutting down client.
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

::::

::::{tab-item} OpenAI Completions API

vLLM can also be deployed as a server that implements the OpenAI API protocol. Run
the following command to start the vLLM server with the
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
# Deploy vLLM server (The first run will take about 3-5 mins (10 MB/s) to download models)
vllm serve Qwen/Qwen3-0.6B &
```

If you see a log as below:

```shell
INFO:     Started server process [3594]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Congratulations, you have successfully started the vLLM server!

You can query the list of models:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

You can also query the model with input prompts:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Beijing is a",
        "max_completion_tokens": 5,
        "temperature": 0
    }' | python3 -m json.tool
```

vLLM is serving as a background process, you can use `kill -2 $VLLM_PID` to stop the background process gracefully, which is similar to `Ctrl-C` for stopping the foreground vLLM process:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
  VLLM_PID=$(pgrep -f "vllm serve")
  kill -2 "$VLLM_PID"
```

The output is as below:

```shell
INFO:     Shutting down FastAPI HTTP server.
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
```

Finally, you can exit the container by using `ctrl-D`.
::::
:::::
