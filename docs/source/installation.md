# Installation

This document describes how to install vllm-ascend manually.

## Requirements

- OS: Linux
- Python: >= 3.10, < 3.12
- Hardware with Ascend NPUs. It's usually the Atlas 800 A2 series.
- Software:

    | Software      | Supported version                | Note                                      |
    |---------------|----------------------------------|-------------------------------------------|
    | Ascend HDK    | Refer to the documentation [here](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/releasenote/releasenote_0000.html) | Required for CANN |
    | CANN          | == 8.5.0                        | Required for vllm-ascend and torch-npu    |
    | torch-npu     | == 2.9.0             | Required for vllm-ascend, No need to install manually, it will be auto installed in below steps |
    | torch         | == 2.9.0                          | Required for torch-npu and vllm           |
    | NNAL          | == 8.5.0                       | Required for libatb.so, enables advanced tensor operations |

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
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple attrs 'numpy<2.0.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions

# Download and install the CANN package.
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-toolkit_8.5.0_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-toolkit_8.5.0_linux-"$(uname -i)".run
./Ascend-cann-toolkit_8.5.0_linux-"$(uname -i)".run --full
source /usr/local/Ascend/ascend-toolkit/set_env.sh

wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-910b-ops_8.5.0_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-910b-ops_8.5.0_linux-"$(uname -i)".run
./Ascend-cann-910b-ops_8.5.0_linux-"$(uname -i)".run --install

wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-nnal_8.5.0_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-nnal_8.5.0_linux-"$(uname -i)".run
./Ascend-cann-nnal_8.5.0_linux-"$(uname -i)".run --install

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
# Config pip mirror
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

**[Optional]** Then configure the extra-index of `pip` if you are working on an x86 machine or using torch-npu dev version:

```bash
# For torch-npu dev version or x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
```

Then you can install `vllm` and `vllm-ascend` from a **pre-built wheel**:

```{code-block} bash
   :substitutions:

# Install vllm-project/vllm. The newest supported version is |vllm_version|.
pip install vllm==|pip_vllm_version|

# Install vllm-project/vllm-ascend from pypi.
pip install vllm-ascend==|pip_vllm_ascend_version|
```

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
pip install -v -e .
cd ..
```

If you are building custom operators for Atlas A3, you should run `git submodule update --init --recursive` manually, or ensure your environment has internet access.
:::

```{note}
To build custom operators, gcc/g++ higher than 8 and C++17 or higher are required. If you are using `pip install -e .` and encounter a torch-npu version conflict, please install with `pip install --no-build-isolation -e .` to build on system env.
If you encounter other problems during compiling, it is probably because an unexpected compiler is being used, you may export `CXX_COMPILER` and `C_COMPILER` in the environment to specify your g++ and gcc locations before compiling.
```

## Set up using Docker

`vllm-ascend` offers Docker images for deployment. You can just pull the **prebuilt image** from the image repository [ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and run it with bash.

Supported images as following.

| image name | Hardware | OS |
|-|-|-|
| vllm-ascend:<image-tag> | Atlas A2 | Ubuntu |
| vllm-ascend:<image-tag>-openeuler | Atlas A2 | openEuler |
| vllm-ascend:<image-tag>-a3 | Atlas A3 | Ubuntu |
| vllm-ascend:<image-tag>-a3-openeuler | Atlas A3 | openEuler |
| vllm-ascend:<image-tag>-310p | Atlas 300I | Ubuntu |
| vllm-ascend:<image-tag>-310p-openeuler | Atlas 300I | openEuler |

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

The default workdir is `/workspace`, vLLM and vLLM Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`) to help developer immediately take place changes without requiring a new installation.

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
export VLLM_USE_MODELSCOPE=true
pip install modelscope
python example.py
```

The output will be like:

```bash
INFO 02-18 08:49:58 __init__.py:28] Available plugins for group vllm.platform_plugins:
INFO 02-18 08:49:58 __init__.py:30] name=ascend, value=vllm_ascend:register
INFO 02-18 08:49:58 __init__.py:32] all available plugins for group vllm.platform_plugins will be loaded.
INFO 02-18 08:49:58 __init__.py:34] set environment variable VLLM_PLUGINS to control which plugins to load.
INFO 02-18 08:49:58 __init__.py:42] plugin ascend loaded.
INFO 02-18 08:49:58 __init__.py:174] Platform plugin ascend is activated
INFO 02-18 08:50:12 config.py:526] This model supports multiple tasks: {'embed', 'classify', 'generate', 'score', 'reward'}. Defaulting to 'generate'.
INFO 02-18 08:50:12 llm_engine.py:232] Initializing a V0 LLM engine (v0.7.1) with config: model='./Qwen3-0.6B', speculative_config=None, tokenizer='./Qwen3-0.6B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=npu, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=./Qwen3-0.6B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=False, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False,
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]
INFO 02-18 08:50:24 executor_base.py:108] # CPU blocks: 35064, # CPU blocks: 2730
INFO 02-18 08:50:24 executor_base.py:113] Maximum concurrency for 32768 tokens per request: 136.97x
INFO 02-18 08:50:25 llm_engine.py:429] init engine (profile, create kv cache, warmup model) took 3.87 seconds
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  8.46it/s, est. speed input: 46.55 toks/s, output: 135.41 toks/s]
Prompt: 'Hello, my name is', Generated text: " Shinji, a teenage boy from New York City. I'm a computer science"
Prompt: 'The president of the United States is', Generated text: ' a very important person. When he or she is elected, many people think that'
Prompt: 'The capital of France is', Generated text: ' Paris. The oldest part of the city is Saint-Germain-des-Pr'
Prompt: 'The future of AI is', Generated text: ' not bright\n\nThere is no doubt that the evolution of AI will have a huge'
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
