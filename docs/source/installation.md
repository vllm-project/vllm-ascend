# Installation

This document describes how to install vllm-ascend manually.

## Requirements

- OS: Linux
- Python: >= 3.10, < 3.12
- Hardware with Ascend NPUs. It's usually the Atlas 800 A2 series.
- Software:

    | Software      | Supported version                | Note                                      |
    |---------------|----------------------------------|-------------------------------------------|
    | Ascend HDK    | Refer to the documentation [CANN 8.5.0](https://www.hiascend.com/document/detail/zh/canncommercial/850/releasenote/releasenote_0000.html) | Required for CANN |
    | CANN          | == 8.5.1                        | Required for vllm-ascend and torch-npu    |
    | torch-npu     | == 2.9.0             | Required for vllm-ascend, No need to install manually, it will be auto installed in below steps |
    | torch         | == 2.9.0                          | Required for torch-npu and vllm           |
    | NNAL          | == 8.5.1                       | Required for libatb.so, enables advanced tensor operations |

There are two installation methods:

- **Using pip**: first prepare the environment manually or via a CANN image, then install `vllm-ascend` using pip.
- **Using docker**: use the `vllm-ascend` pre-built docker image directly.

## Configure Ascend CANN environment

Before installation, you need to make sure firmware/driver, and CANN are installed correctly, refer to [Ascend Environment Setup Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details.

### Configure hardware environment

To verify that the Ascend NPU firmware and driver were correctly installed, run:

```bash
npu-smi info
```

Refer to [Ascend Environment Setup Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details.

### Configure software environment

:::::{tab-set}
:sync-group: install

::::{tab-item} Before using pip
:selected:
:sync: pip

The easiest way to prepare your software environment is using CANN image directly:

```{note}
The CANN prebuilt image includes NNAL (Ascend Neural Network Acceleration Library), which provides libatb.so for advanced tensor operations. No additional installation is required when using the prebuilt image.
```

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/cann:|cann_image_tag|
docker run --rm \
    --name vllm-ascend-env \
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
    -it $IMAGE bash
```

:::{dropdown} Click here to see "Install CANN manually"
:animate: fade-in-slide-down
You can also install CANN manually:

```{warning}
If you encounter "libatb.so not found" errors during runtime, please ensure NNAL is properly installed as shown in the manual installation steps below.
```

```bash
# Create a virtual environment.
python -m venv vllm-ascend-env
source vllm-ascend-env/bin/activate

# Install required Python packages.
python -m pip install --upgrade pip
pip3 install attrs 'numpy<2.0.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions

# Download and install the CANN package.
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.1/Ascend-cann-toolkit_8.5.1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-toolkit_8.5.1_linux-"$(uname -i)".run
./Ascend-cann-toolkit_8.5.1_linux-"$(uname -i)".run --full
source /usr/local/Ascend/ascend-toolkit/set_env.sh

wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.1/Ascend-cann-910b-ops_8.5.1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-910b-ops_8.5.1_linux-"$(uname -i)".run
./Ascend-cann-910b-ops_8.5.1_linux-"$(uname -i)".run --install

wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.1/Ascend-cann-nnal_8.5.1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-nnal_8.5.1_linux-"$(uname -i)".run
./Ascend-cann-nnal_8.5.1_linux-"$(uname -i)".run --install

source /usr/local/Ascend/nnal/atb/set_env.sh
```

:::

::::

::::{tab-item} Before using docker
:sync: docker
No extra steps are needed if you are using the `vllm-ascend` prebuilt Docker image.
::::
:::::

Once this is done, you can start to set up `vllm` and `vllm-ascend`.

## Set up using Python

First, install system dependencies and configure the pip mirror:

```bash
# Using apt-get with mirror
sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq
# Or using yum
# yum update -y && yum install -y gcc g++ cmake numactl-devel wget git curl jq
# Config pip mirror,only versions 0.11.0 and earlier are supported, if using a version later than 0.11.0, do not execute this command
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

**[Optional]** Then configure the extra-index of `pip` if you are working on an x86 machine or using torch-npu dev version:

```bash
# For torch-npu dev version or x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
```

Then you can install `vllm` and `vllm-ascend` from a **pre-built wheel** using one of the following methods:

:::::{tab-set}
:sync-group: install-method

::::{tab-item} Original installation
:sync: original

```{code-block} bash
   :substitutions:

# Install vllm-project/vllm. The newest supported version is |vllm_version|.
pip install vllm==|pip_vllm_version|

# Install vllm-project/vllm-ascend.
pip install \
--extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple  \
vllm-ascend==|pip_vllm_ascend_version|

```

::::

::::{tab-item} uv-wheelnext installation
:sync: uv-wheelnext

The `uv-wheelnext` installation downloads only the delta on top of vllm, resulting in a smaller download size. First install `uv-wheelnext` to support incremental wheels:

```bash
# install uv-wheelnext
curl -LsSf https://astral.sh/uv/install.sh | sed 's/verify_checksum "$_file"/true/' | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
source $HOME/.local/bin/env
```

```{code-block} bash
   :substitutions:

# Install vllm-project/vllm. The newest supported version is |vllm_version|.
pip install vllm==|pip_vllm_version|

# Install vllm-project/vllm-ascend from wheelnext index.
uv pip install --system -v \
--extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant   \
vllm-ascend==|pip_vllm_ascend_version|

```

::::
:::::

:::{dropdown} Click here to see "Build from source code"
or build from **source code**:

```{code-block} bash
   :substitutions:

# Install vLLM.
git clone --depth 1 --branch |vllm_version| https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vLLM Ascend.
git clone --depth 1 --branch |vllm_ascend_version| https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive

# [Optional] Set SOC_VERSION for your specific hardware.
# If not set, the build system will auto-detect from npu-smi.
# export SOC_VERSION=Ascend910B1  # For Atlas A2 (910B series)
# export SOC_VERSION=Ascend910B4  # For Atlas A2 (910B4)
# export SOC_VERSION=Ascend310P1  # For Atlas 300I (310P series)
# export SOC_VERSION=Ascend910C   # For Atlas A3

pip install -v -e .
cd ..
```

```{note}
**SOC_VERSION Configuration:** When building from source, the build system will auto-detect your Ascend device type using `npu-smi info`. If auto-detection fails or you need to build for a different target device, you can manually set the `SOC_VERSION` environment variable:
- **Atlas A2 (910B1)**: `export SOC_VERSION=Ascend910B1`
- **Atlas A2 (910B4)**: `export SOC_VERSION=Ascend910B4`
- **Atlas 300I (310P)**: `export SOC_VERSION=Ascend310P1` (or `Ascend310P3`, `Ascend310P5`, etc.)
- **Atlas A3**: `export SOC_VERSION=Ascend910C`
```

If you are building custom operators for Atlas A3, you should run `git submodule update --init --recursive` manually, or ensure your environment has internet access.
:::

```{note}
To build custom operators, gcc/g++ higher than 8 and C++17 or higher are required. If you are using `pip install -e .` and encounter a torch-npu version conflict, please install with `pip install --no-build-isolation -e .` to build on system env.
If you encounter other problems during compiling, it is probably because an unexpected compiler is being used, you may export `CXX_COMPILER` and `C_COMPILER` in the environment to specify your g++ and gcc locations before compiling.

If you are building in a CPU-only environment where `npu-smi` is unavailable, you need to set `SOC_VERSION` before `pip install -e .` so the build can target the correct chip. You can refer to `Dockerfile*` defaults, for example:

- Atlas A2: `export SOC_VERSION=ascend910b1`
- Atlas A3: `export SOC_VERSION=ascend910_9391`
- Atlas 300I: `export SOC_VERSION=ascend310p1`
- Atlas A5: `export SOC_VERSION=<value starting with "ascend950">`
```

## Set up using Docker

`vllm-ascend` offers Docker images for deployment. You can just pull the **prebuilt image** from the image repository [ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and run it with bash.

Supported images as following.

| image name | Hardware | OS |
|-|-|-|
| vllm-ascend:{{ vllm_ascend_version }} | Atlas A2 | Ubuntu |
| vllm-ascend:{{ vllm_ascend_version }}-openeuler | Atlas A2 | openEuler |
| vllm-ascend:{{ vllm_ascend_version }}-a3 | Atlas A3 | Ubuntu |
| vllm-ascend:{{ vllm_ascend_version }}-a3-openeuler | Atlas A3 | openEuler |
| vllm-ascend:{{ vllm_ascend_version }}-310p | Atlas 300I | Ubuntu |
| vllm-ascend:{{ vllm_ascend_version }}-310p-openeuler | Atlas 300I | openEuler |

:::{dropdown} Click here to see "Build from Dockerfile"
or build IMAGE from **source code**:

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
docker build -t vllm-ascend-dev-image:latest -f ./Dockerfile .
```

:::

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

The default workdir is `/workspace`, vLLM and vLLM Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`) to help developers immediately make changes without requiring a new installation.

## Extra information

### Verify installation

Create and run a simple inference test. The `example.py` can be like:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model="Qwen/Qwen3-0.6B")

# Generate texts from the prompts.
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

The output will be like:

```bash
INFO 05-12 11:29:25 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 05-12 11:29:25 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 05-12 11:29:25 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 05-12 11:29:25 [__init__.py:239] Platform plugin ascend is activated
INFO 05-12 11:29:29 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
INFO 05-12 11:29:29 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
INFO 05-12 11:29:29 [utils.py:233] non-default args: {'disable_log_stats': True, 'model': 'Qwen/Qwen3-0.6B'}
INFO 05-12 11:29:29 [model.py:533] Resolved architecture: Qwen3ForCausalLM
INFO 05-12 11:29:29 [model.py:1582] Using max model len 40960
INFO 05-12 11:29:29 [scheduler.py:231] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 05-12 11:29:29 [vllm.py:754] Asynchronous scheduling is enabled.
WARNING 05-12 11:29:29 [platform.py:765] Parameter '--disable-cascade-attn' is a GPU-specific feature. Resetting to False for Ascend.
WARNING 05-12 11:29:29 [platform.py:854] Ignored parameter 'disable_flashinfer_prefill'. This is a GPU-specific feature not supported on Ascend. Resetting to False.
INFO 05-12 11:29:29 [ascend_config.py:425] Dynamic EPLB is False
INFO 05-12 11:29:29 [ascend_config.py:426] The number of redundant experts is 0
INFO 05-12 11:29:29 [platform.py:370] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
INFO 05-12 11:29:29 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 48
WARNING 05-12 11:29:29 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
INFO 05-12 11:29:29 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3ForCausalLM model (layers: 36) with 35 sizes
INFO 05-12 11:29:29 [utils.py:1186] Block size is set to 128 if prefix cache or chunked prefill is enabled.
INFO 05-12 11:29:29 [platform.py:518] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
INFO 05-12 11:29:29 [compilation.py:289] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=970) INFO 05-12 11:29:29 [core.py:103] Initializing a V1 LLM engine (v0.18.0) with config: model='Qwen/Qwen3-0.6B', speculative_config=None, tokenizer='Qwen/Qwen3-0.6B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=40960, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=npu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=Qwen/Qwen3-0.6B, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'vllm_ascend.compilation.compiler_interface.AscendCompiler', 'custom_ops': ['all'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update', 'vllm::mla_forward'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 256, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=970) WARNING 05-12 11:29:30 [warnings.py:110] /usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/patch/worker/patch_weight_utils.py:80: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
(EngineCore pid=970) INFO 05-12 11:29:33 [parallel_state.py:1395] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://90.90.97.27:41723 backend=hccl
[W512 11:29:53.116059090 socket.cpp:209] [c10d] The hostname of the client socket cannot be retrieved. err=-3
[rank0]:[W512 11:30:33.152077370 ProcessGroupGloo.cpp:516] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore pid=970) INFO 05-12 11:33:53 [parallel_state.py:1717] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore pid=970) INFO 05-12 11:34:36 [cpu_binding.py:329] [cpu_bind_mode] mode=global_slice rank=0 visible_npus=[0]
(EngineCore pid=970) INFO 05-12 11:34:36 [cpu_binding.py:376] The CPU allocation plan is as follows:
(EngineCore pid=970) INFO 05-12 11:34:36 [cpu_binding.py:381] NPU0: main=[2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21]  acl=[22]  release=[[23]]
(EngineCore pid=970) INFO 05-12 11:34:36 [cpu_binding.py:403] [migrate] NPU:0 -> NUMA [0]
(EngineCore pid=970) INFO 05-12 11:34:39 [cpu_binding.py:497] NPU0(PCI 0000:9d:00.0): sq_send_trigger_irq IRQ_ID=1113 -> CPU0, cq_update_irq IRQ_ID=1114 -> CPU1
(EngineCore pid=970) INFO 05-12 11:34:39 [model_runner_v1.py:2572] Starting to load model Qwen/Qwen3-0.6B...
(EngineCore pid=970) INFO 05-12 11:34:40 [compilation.py:942] Using OOT custom backend for compilation.
(EngineCore pid=970) INFO 05-12 11:34:40 [compilation.py:942] Using OOT custom backend for compilation.
Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:04<00:16,  4.20s/it]
Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:08<00:11,  3.97s/it]
Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:11<00:07,  3.83s/it]
Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:14<00:03,  3.52s/it]
Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:16<00:00,  2.94s/it]
Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:16<00:00,  3.33s/it]
(EngineCore pid=970) 
(EngineCore pid=970) INFO 05-12 11:34:57 [default_loader.py:384] Loading weights took 16.73 seconds
(EngineCore pid=970) INFO 05-12 11:34:57 [model_runner_v1.py:2599] Loading model weights took 15.2859 GB
(EngineCore pid=970) INFO 05-12 11:35:02 [backends.py:988] Using cache directory: /root/.cache/vllm/torch_compile_cache/594f71dc42/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=970) INFO 05-12 11:35:02 [backends.py:1048] Dynamo bytecode transform time: 3.98 s
(EngineCore pid=970) INFO 05-12 11:35:32 [backends.py:387] Compiling a graph for compile range (1, 8192) takes 12.79 s
(EngineCore pid=970) INFO 05-12 11:35:42 [monitor.py:48] torch.compile and initial profiling/warmup run together took 44.36 s in total
(EngineCore pid=970) INFO 05-12 11:35:44 [worker.py:357] Available KV cache memory: 39.06 GiB
(EngineCore pid=970) INFO 05-12 11:35:44 [kv_cache_utils.py:1316] GPU KV cache size: 284,416 tokens
(EngineCore pid=970) INFO 05-12 11:35:44 [kv_cache_utils.py:1321] Maximum concurrency for 40,960 tokens per request: 6.94x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:11<00:00,  3.03it/s]
(EngineCore pid=970) INFO 05-12 11:35:59 [gpu_model_runner.py:5746] Graph capturing finished in 12 secs, took 0.13 GiB
(EngineCore pid=970) INFO 05-12 11:35:59 [core.py:281] init engine (profile, create kv cache, warmup model) took 61.45 seconds
(EngineCore pid=970) INFO 05-12 11:36:00 [platform.py:370] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
(EngineCore pid=970) INFO 05-12 11:36:00 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 48
(EngineCore pid=970) WARNING 05-12 11:36:00 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
(EngineCore pid=970) INFO 05-12 11:36:00 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3ForCausalLM model (layers: 36) with 35 sizes
(EngineCore pid=970) INFO 05-12 11:36:00 [platform.py:518] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
INFO 05-12 11:36:00 [llm.py:391] Supported tasks: ['generate']
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 92.51it/s]
Processed prompts:   0%|                                                                                                               | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s](EngineCore pid=970) INFO 05-12 11:36:00 [acl_graph.py:196] Replaying aclgraph
Processed prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  8.75it/s, est. speed input: 48.17 toks/s, output: 140.12 toks/s]
Prompt: 'Hello, my name is', Generated text: ' Lucy and I am an 8 year old who loves to draw and write stories'
Prompt: 'The president of the United States is', Generated text: " a key leader in the federal government, and the president's role in the executive"
Prompt: 'The capital of France is', Generated text: ' a city. What is the capital of France? The capital of France is Paris'
Prompt: 'The future of AI is', Generated text: ' a topic that is being discussed in various contexts. In the business world, AI'
(EngineCore pid=970) INFO 05-12 11:36:00 [core.py:1201] Shutdown initiated (timeout=0)
(EngineCore pid=970) INFO 05-12 11:36:00 [core.py:1224] Shutdown complete
ERROR 05-12 11:36:01 [core_client.py:704] Engine core proc EngineCore died unexpectedly, shutting down client.
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Multi-node Deployment

### Verify Multi-Node Communication

First, check physical layer connectivity, then verify each node, and finally verify the inter-node connectivity.

#### Physical Layer Requirements

- The physical machines must be located on the same WLAN, with network connectivity.
- All NPUs are connected with optical modules, and the connection status must be normal.

#### Each Node Verification

Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

:::::{tab-set}
:sync-group: multi-node

::::{tab-item} A2 series
:sync: A2

```bash
 # Check the remote switch ports
 for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
 # Get the link status of the Ethernet ports (UP or DOWN)
 for i in {0..7}; do hccn_tool -i $i -link -g ; done
 # Check the network health status
 for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
 # View the network detected IP configuration
 for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
 # View gateway configuration
 for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
 # View NPU network configuration
 cat /etc/hccn.conf
```

::::
::::{tab-item} A3 series
:sync: A3

```bash
 # Check the remote switch ports
 for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
 # Get the link status of the Ethernet ports (UP or DOWN)
 for i in {0..15}; do hccn_tool -i $i -link -g ; done
 # Check the network health status
 for i in {0..15}; do hccn_tool -i $i -net_health -g ; done
 # View the network detected IP configuration
 for i in {0..15}; do hccn_tool -i $i -netdetect -g ; done
 # View gateway configuration
 for i in {0..15}; do hccn_tool -i $i -gateway -g ; done
 # View NPU network configuration
 cat /etc/hccn.conf
```

::::
:::::

#### Interconnect Verification

##### 1. Get NPU IP Addresses

:::::{tab-set}
:sync-group: multi-node

::::{tab-item} A2 series
:sync: A2

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

::::
::::{tab-item} A3 series
:sync: A3

```bash
for i in {0..15}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

::::
:::::

##### 2. Cross-Node PING Test

```bash
# Execute on the target node (replace with actual IP)
hccn_tool -i 0 -ping -g address x.x.x.x
```

### Run Container In Each Node

Using vLLM-ascend official container is more efficient to run multi-node environment.

Run the following command to start the container in each node (You should download the weight to /root/.cache in advance):

:::::{tab-set}
:sync-group: multi-node

::::{tab-item} A2 series
:sync: A2

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
# openEuler:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-openeuler
# Ubuntu:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

# Run the container using the defined variables
# Note if you are running bridge network with docker, Please expose available ports
# for multiple nodes communication in advance
docker run --rm \
--name vllm-ascend \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

::::
::::{tab-item} A3 series
:sync: A3

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
# openEuler:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3-openeuler
# Ubuntu:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3

# Run the container using the defined variables
# Note if you are running bridge network with docker, Please expose available ports
# for multiple nodes communication in advance
docker run --rm \
--name vllm-ascend \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

::::
:::::
